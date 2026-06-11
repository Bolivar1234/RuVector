//! Latent reasoning memory: z_H snapshot + warm start (ADR-199 R3).
//!
//! The H-level state is an explicit compressed "what I've figured out"
//! tensor. This module snapshots the final `z_H` of a forward pass
//! ([`HrmTextModel::capture_state`]), exposes a mean-pooled embedding for
//! ruvector memory ([`HrmTextModel::embed_state`] — the ADR-199 `embed_state`
//! hook), and warm-starts the recurrence from a retrieved snapshot instead of
//! the learned init buffer alone ([`HrmTextModel::prefill_warm`]).
//!
//! ## v0 warm-start policy (research-gated)
//!
//! `z_L(pos) = z_l_init + mean_pool(snapshot.z_h)` — the mean-pooled snapshot
//! state is broadcast over the sequence and ADDED to the learned `z_L` init
//! buffer (matching the architecture's additive-injection convention). This
//! is the simplest principled blend; R3 is the most speculative ADR-199
//! track (pass gate: > 10 % latency or quality gain on repeated tasks; kill:
//! cross-prompt state contamination), so the policy stays deliberately
//! minimal until the research gate is evaluated on real weights.
//!
//! Snapshots carry a config fingerprint; warm-start rejects mismatches.

use serde::{Deserialize, Serialize};

use super::cache::HrmKvCaches;
use super::config::{HrmTextConfig, NormPosition};
use super::generate::HrmTextModel;
use super::mask::AttentionMaskMode;
use crate::error::{Result, RuvLLMError};

/// Snapshot of the final H-level hidden state of one forward pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HrmStateSnapshot {
    /// Per-position final `z_H`, `[seq_len, hidden_size]` row-major
    /// (pre-`final_norm`).
    pub z_h: Vec<f32>,
    /// Number of positions captured.
    pub seq_len: usize,
    /// Hidden size of the producing model.
    pub hidden_size: usize,
    /// Stable architecture fingerprint of the producing config
    /// ([`config_fingerprint`]); warm-start rejects mismatches.
    pub config_fingerprint: u64,
}

impl HrmStateSnapshot {
    /// Internal-consistency check.
    pub fn validate(&self) -> Result<()> {
        if self.seq_len == 0 || self.hidden_size == 0 {
            return Err(RuvLLMError::InvalidOperation(
                "HrmStateSnapshot must have non-zero seq_len and hidden_size".to_string(),
            ));
        }
        if self.z_h.len() != self.seq_len * self.hidden_size {
            return Err(RuvLLMError::InvalidOperation(format!(
                "HrmStateSnapshot z_h length {} != seq_len {} * hidden_size {}",
                self.z_h.len(),
                self.seq_len,
                self.hidden_size
            )));
        }
        Ok(())
    }

    /// Mean-pool `z_h` over positions, `[hidden_size]`.
    pub fn mean_pooled(&self) -> Vec<f32> {
        let mut pooled = vec![0.0f32; self.hidden_size];
        for row in self.z_h.chunks_exact(self.hidden_size) {
            for (p, v) in pooled.iter_mut().zip(row.iter()) {
                *p += v;
            }
        }
        let inv = 1.0 / self.seq_len as f32;
        for p in pooled.iter_mut() {
            *p *= inv;
        }
        pooled
    }
}

/// Stable FNV-1a hash of the ARCHITECTURE-relevant config fields: hidden /
/// intermediate sizes, layer and head counts, vocab size, `half_layers`,
/// RoPE theta, RMSNorm epsilon, and norm position.
///
/// Runtime knobs are deliberately EXCLUDED — `h_cycles`/`l_cycles` (the R1
/// override), `max_position_embeddings`, `prefix_bidirectional`, and token
/// ids — so a snapshot stays valid across cycle-depth re-routing.
pub fn config_fingerprint(config: &HrmTextConfig) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    for v in [
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_kv_heads,
        config.vocab_size,
        config.half_layers,
    ] {
        mix(&(v as u64).to_le_bytes());
    }
    mix(&config.rope_theta.to_bits().to_le_bytes());
    mix(&config.rms_norm_eps.to_bits().to_le_bytes());
    mix(&[match config.norm_position {
        NormPosition::PreNorm => 0u8,
        NormPosition::PostNorm => 1u8,
    }]);
    hash
}

impl HrmTextModel {
    /// Full-sequence forward capturing the final `z_H` as a snapshot.
    /// Allocates its own caches; returns `(logits, snapshot)` where logits
    /// are `[seq, vocab_size]` row-major.
    pub fn capture_state(
        &self,
        input_ids: &[u32],
        mask_mode: AttentionMaskMode,
    ) -> Result<(Vec<f32>, HrmStateSnapshot)> {
        let mut caches = HrmKvCaches::new(&self.config);
        let (logits, z_h) = self.forward_with_state(input_ids, mask_mode, &mut caches)?;
        let snapshot = HrmStateSnapshot {
            z_h,
            seq_len: input_ids.len(),
            hidden_size: self.config.hidden_size,
            config_fingerprint: config_fingerprint(&self.config),
        };
        Ok((logits, snapshot))
    }

    /// Mean-pooled final `z_H` for `input_ids`, `[hidden_size]` — the
    /// ruvector memory hook (ADR-199 `embed_state`). The whole input is
    /// treated as prompt: bidirectional over the full sequence when
    /// `prefix_bidirectional` is set, causal otherwise.
    pub fn embed_state(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let mask = if self.config.prefix_bidirectional {
            AttentionMaskMode::PrefixLm {
                prefix_len: input_ids.len(),
            }
        } else {
            AttentionMaskMode::Causal
        };
        let (_logits, snapshot) = self.capture_state(input_ids, mask)?;
        Ok(snapshot.mean_pooled())
    }

    /// Warm-start prefill (v0 policy, see module docs): like
    /// [`prefill`](Self::prefill), but `z_L` is initialized to
    /// `z_l_init + mean_pool(snapshot.z_h)` broadcast over the sequence.
    /// Returns the LAST position's logits, `[vocab_size]`. Rejects snapshots
    /// whose `config_fingerprint` does not match this model.
    pub fn prefill_warm(
        &self,
        input_ids: &[u32],
        prefix_len: usize,
        snapshot: &HrmStateSnapshot,
        caches: &mut HrmKvCaches,
    ) -> Result<Vec<f32>> {
        snapshot.validate()?;
        let expected = config_fingerprint(&self.config);
        if snapshot.config_fingerprint != expected {
            return Err(RuvLLMError::Config(format!(
                "snapshot fingerprint {:#018x} does not match model fingerprint {:#018x}; \
                 warm-start requires an architecture-identical producer",
                snapshot.config_fingerprint, expected
            )));
        }
        if snapshot.hidden_size != self.config.hidden_size {
            return Err(RuvLLMError::Config(format!(
                "snapshot hidden_size {} != model hidden_size {}",
                snapshot.hidden_size, self.config.hidden_size
            )));
        }
        self.check_cache_shape(caches)?;
        let seq = input_ids.len();
        if seq == 0 {
            return Err(RuvLLMError::InvalidOperation(
                "prefill_warm called with empty input_ids".to_string(),
            ));
        }
        if seq > self.config.max_position_embeddings {
            return Err(RuvLLMError::InvalidOperation(format!(
                "sequence length {} exceeds max_position_embeddings {}",
                seq, self.config.max_position_embeddings
            )));
        }
        if prefix_len > seq {
            return Err(RuvLLMError::InvalidOperation(format!(
                "prefix_len {} exceeds prompt length {}",
                prefix_len, seq
            )));
        }
        caches.clear();

        // v0 warm-start blend: learned buffer + broadcast pooled snapshot.
        let pooled = snapshot.mean_pooled();
        let warm_row: Vec<f32> = self
            .z_l_init
            .iter()
            .zip(pooled.iter())
            .map(|(a, b)| a + b)
            .collect();
        let mut z_l0 = Vec::with_capacity(seq * self.config.hidden_size);
        for _ in 0..seq {
            z_l0.extend_from_slice(&warm_row);
        }

        let mask = if self.config.prefix_bidirectional && prefix_len > 0 {
            AttentionMaskMode::PrefixLm { prefix_len }
        } else {
            AttentionMaskMode::Causal
        };
        let z_h = self.embed_sequence(input_ids)?;
        let positions: Vec<usize> = (0..seq).collect();
        let z_h = self.run_cycles_from(z_h, z_l0, &positions, mask, caches)?;
        let logits = self.project_logits(z_h)?;
        let vocab = self.config.vocab_size;
        Ok(logits[(seq - 1) * vocab..].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_stable_and_sensitive() {
        let config = HrmTextConfig::tiny_test();
        let a = config_fingerprint(&config);
        let b = config_fingerprint(&HrmTextConfig::tiny_test());
        assert_eq!(a, b, "fingerprint must be deterministic");

        let mut arch_changed = HrmTextConfig::tiny_test();
        arch_changed.rope_theta = 50000.0;
        assert_ne!(a, config_fingerprint(&arch_changed));

        // Runtime knobs do NOT change the fingerprint.
        let mut cycles_changed = HrmTextConfig::tiny_test();
        cycles_changed.h_cycles = 4;
        cycles_changed.l_cycles = 1;
        cycles_changed.max_position_embeddings = 128;
        assert_eq!(a, config_fingerprint(&cycles_changed));
    }

    #[test]
    fn test_snapshot_validate_and_mean_pool() {
        let snap = HrmStateSnapshot {
            z_h: vec![1.0, 3.0, 5.0, 7.0],
            seq_len: 2,
            hidden_size: 2,
            config_fingerprint: 0,
        };
        snap.validate().unwrap();
        assert_eq!(snap.mean_pooled(), vec![3.0, 5.0]);

        let bad = HrmStateSnapshot {
            z_h: vec![0.0; 3],
            seq_len: 2,
            hidden_size: 2,
            config_fingerprint: 0,
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_snapshot_serde_round_trip() {
        let config = HrmTextConfig::tiny_test();
        let model = HrmTextModel::new_random(config, 42).unwrap();
        let (_logits, snap) = model
            .capture_state(&[3, 5, 7], AttentionMaskMode::Causal)
            .unwrap();
        let json = serde_json::to_string(&snap).unwrap();
        let back: HrmStateSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back, snap);
    }
}
