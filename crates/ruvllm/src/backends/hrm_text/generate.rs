//! HRM-Text model assembly, forward recurrence, prefill/decode and greedy
//! generation (ADR-199 Phase D items 1-3).
//!
//! The forward pass is a direct port of the upstream no-carry recurrence:
//!
//! ```text
//! z_H <- embed(input_ids)
//! z_L <- learned z_l_init buffer, broadcast over the sequence
//! for i in 0..h_cycles {
//!     for k in 0..l_cycles { z_L = L(z_L + z_H) }   // additive injection
//!     z_H = H(z_H + z_L)
//! }
//! logits = lm_head(final_norm(z_H))
//! ```
//!
//! No ACT/halting exists upstream; cycle counts are fixed per forward but may
//! be overridden at runtime via [`HrmTextModel::cycle_override`] (R1 hook).

use super::cache::HrmKvCaches;
use super::config::HrmTextConfig;
use super::core::HrmRecurrentCore;
use super::mask::AttentionMaskMode;
use super::tensor::{Embedding, Linear, RmsNorm, WeightRng};
use crate::error::{Result, RuvLLMError};

/// Complete HRM-Text model with hierarchical H/L recurrent cores.
#[derive(Debug, Clone)]
pub struct HrmTextModel {
    /// Model configuration.
    pub config: HrmTextConfig,
    /// Token embedding table.
    pub embed_tokens: Embedding,
    /// Slow H-level core (`half_layers` decoder layers).
    pub h_level: HrmRecurrentCore,
    /// Fast L-level core (`half_layers` decoder layers).
    pub l_level: HrmRecurrentCore,
    /// Learned L-state init buffer, `[hidden_size]`, broadcast over the sequence.
    pub z_l_init: Vec<f32>,
    /// Final RMSNorm before the LM head.
    pub final_norm: RmsNorm,
    /// LM head, `[vocab_size, hidden_size]`.
    pub lm_head: Linear,
}

impl HrmTextModel {
    /// Build a model with deterministic seeded random weights
    /// (truncated-normal-ish init via a clamped-gaussian xorshift RNG).
    /// Same `(config, seed)` always yields identical weights.
    pub fn new_random(config: HrmTextConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        let mut rng = WeightRng::new(seed);

        let mut embed = vec![0.0f32; config.vocab_size * config.hidden_size];
        rng.fill_gaussian(&mut embed, 0.02);
        let embed_tokens = Embedding {
            weight: embed,
            vocab_size: config.vocab_size,
            dim: config.hidden_size,
        };

        let h_level = HrmRecurrentCore::new_random(&config, &mut rng);
        let l_level = HrmRecurrentCore::new_random(&config, &mut rng);

        let mut z_l_init = vec![0.0f32; config.hidden_size];
        rng.fill_gaussian(&mut z_l_init, 0.02);

        let final_norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps);
        let lm_head = Linear::new_random(config.hidden_size, config.vocab_size, &mut rng);

        Ok(Self {
            config,
            embed_tokens,
            h_level,
            l_level,
            z_l_init,
            final_norm,
            lm_head,
        })
    }

    /// Override the recurrence depth at runtime (R1 test-time-compute hook).
    ///
    /// Caveat: recurrent models are typically trained at a fixed cycle count;
    /// running at a different depth is out-of-distribution and may degrade
    /// quality beyond the training-time count (ADR-199 R1 kill criterion).
    /// Caches must be re-created ([`HrmKvCaches::new`]) after an override —
    /// forward/prefill/decode validate cache shape and will error otherwise.
    pub fn cycle_override(&mut self, h_cycles: usize, l_cycles: usize) -> Result<()> {
        if h_cycles == 0 || l_cycles == 0 {
            return Err(RuvLLMError::Config(
                "cycle_override requires h_cycles >= 1 and l_cycles >= 1".to_string(),
            ));
        }
        self.config.h_cycles = h_cycles;
        self.config.l_cycles = l_cycles;
        Ok(())
    }

    fn check_cache_shape(&self, caches: &HrmKvCaches) -> Result<()> {
        if caches.h_cycles() != self.config.h_cycles
            || caches.l_cycles() != self.config.l_cycles
            || caches.half_layers() != self.config.half_layers
            || caches.kv_dim() != self.config.kv_dim()
        {
            return Err(RuvLLMError::KvCache(format!(
                "cache shape (h={}, l={}, layers={}, kv_dim={}) does not match \
                 config (h={}, l={}, layers={}, kv_dim={}); re-create caches \
                 after cycle_override",
                caches.h_cycles(),
                caches.l_cycles(),
                caches.half_layers(),
                caches.kv_dim(),
                self.config.h_cycles,
                self.config.l_cycles,
                self.config.half_layers,
                self.config.kv_dim()
            )));
        }
        Ok(())
    }

    /// The nested H/L recurrence. `z_h0` is `[seq, hidden]` (embeddings);
    /// returns the final `z_H`.
    fn run_cycles(
        &self,
        z_h0: Vec<f32>,
        positions: &[usize],
        mask: AttentionMaskMode,
        caches: &mut HrmKvCaches,
    ) -> Result<Vec<f32>> {
        let seq = positions.len();
        let mut z_h = z_h0;
        // z_L starts from the learned init buffer broadcast over positions.
        let mut z_l = Vec::with_capacity(seq * self.config.hidden_size);
        for _ in 0..seq {
            z_l.extend_from_slice(&self.z_l_init);
        }
        for i in 0..self.config.h_cycles {
            for k in 0..self.config.l_cycles {
                let slot = caches.l_cycle(i, k)?;
                z_l = self.l_level.forward(&z_l, &z_h, positions, mask, slot)?;
            }
            let slot = caches.h_cycle(i)?;
            z_h = self.h_level.forward(&z_h, &z_l, positions, mask, slot)?;
        }
        Ok(z_h)
    }

    /// Logits from a final hidden state, `[n, hidden]` -> `[n, vocab]`.
    fn project_logits(&self, mut z_h: Vec<f32>) -> Result<Vec<f32>> {
        self.final_norm.apply_rows(&mut z_h)?;
        self.lm_head.forward(&z_h)
    }

    /// Full-sequence forward pass. Clears `caches` and refills them (full
    /// prefill semantics), returning logits for ALL positions,
    /// `[seq, vocab_size]` row-major.
    pub fn forward(
        &self,
        input_ids: &[u32],
        mask_mode: AttentionMaskMode,
        caches: &mut HrmKvCaches,
    ) -> Result<Vec<f32>> {
        self.check_cache_shape(caches)?;
        let seq = input_ids.len();
        if seq == 0 {
            return Err(RuvLLMError::InvalidOperation(
                "forward called with empty input_ids".to_string(),
            ));
        }
        if seq > self.config.max_position_embeddings {
            return Err(RuvLLMError::InvalidOperation(format!(
                "sequence length {} exceeds max_position_embeddings {}",
                seq, self.config.max_position_embeddings
            )));
        }
        caches.clear();

        let mut z_h = Vec::with_capacity(seq * self.config.hidden_size);
        for &token in input_ids {
            z_h.extend_from_slice(self.embed_tokens.lookup(token)?);
        }
        let positions: Vec<usize> = (0..seq).collect();
        let z_h = self.run_cycles(z_h, &positions, mask_mode, caches)?;
        self.project_logits(z_h)
    }

    /// Prefill the caches with a prompt and return the LAST position's
    /// logits, `[vocab_size]`.
    ///
    /// If `config.prefix_bidirectional` is set and `prefix_len > 0`, the
    /// first `prefix_len` tokens attend bidirectionally
    /// (`AttentionMaskMode::PrefixLm`); the bidirectional region is fixed at
    /// prefill time — subsequent decode steps are causal.
    pub fn prefill(
        &self,
        input_ids: &[u32],
        prefix_len: usize,
        caches: &mut HrmKvCaches,
    ) -> Result<Vec<f32>> {
        if prefix_len > input_ids.len() {
            return Err(RuvLLMError::InvalidOperation(format!(
                "prefix_len {} exceeds prompt length {}",
                prefix_len,
                input_ids.len()
            )));
        }
        let mask = if self.config.prefix_bidirectional && prefix_len > 0 {
            AttentionMaskMode::PrefixLm { prefix_len }
        } else {
            AttentionMaskMode::Causal
        };
        let logits = self.forward(input_ids, mask, caches)?;
        let vocab = self.config.vocab_size;
        let last = (input_ids.len() - 1) * vocab;
        Ok(logits[last..last + vocab].to_vec())
    }

    /// Single-token decode step at absolute position `pos`, using the
    /// per-cycle KV caches filled by [`HrmTextModel::prefill`] (and previous
    /// decode steps). Returns logits, `[vocab_size]`.
    ///
    /// Decode positions are always causal: the PrefixLM bidirectional region
    /// was fixed at prefill, and for `q >= prefix_len` the PrefixLM
    /// allowed-set equals the causal one.
    pub fn decode_step(
        &self,
        token: u32,
        pos: usize,
        caches: &mut HrmKvCaches,
    ) -> Result<Vec<f32>> {
        self.check_cache_shape(caches)?;
        if pos >= self.config.max_position_embeddings {
            return Err(RuvLLMError::InvalidOperation(format!(
                "decode position {} exceeds max_position_embeddings {}",
                pos, self.config.max_position_embeddings
            )));
        }
        let cached = caches.cached_len()?;
        if cached != pos {
            return Err(RuvLLMError::KvCache(format!(
                "decode_step at position {} but caches hold {} positions",
                pos, cached
            )));
        }
        let z_h = self.embed_tokens.lookup(token)?.to_vec();
        let z_h = self.run_cycles(z_h, &[pos], AttentionMaskMode::Causal, caches)?;
        self.project_logits(z_h)
    }

    /// Greedy generation: prefill `prompt_ids` (with `prefix_len`
    /// PrefixLM-bidirectional positions), then argmax-decode up to
    /// `max_new_tokens`. Stops at `eos_token_id` (eos is not included in the
    /// returned tokens). Returns only the newly generated tokens.
    pub fn generate_greedy(
        &self,
        prompt_ids: &[u32],
        prefix_len: usize,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        let mut caches = HrmKvCaches::new(&self.config);
        let mut last_logits = self.prefill(prompt_ids, prefix_len, &mut caches)?;
        let mut out = Vec::new();
        let mut pos = prompt_ids.len();

        for step in 0..max_new_tokens {
            let next = argmax(&last_logits)?;
            if next == self.config.eos_token_id {
                break;
            }
            out.push(next);
            if step + 1 == max_new_tokens || pos >= self.config.max_position_embeddings {
                break;
            }
            last_logits = self.decode_step(next, pos, &mut caches)?;
            pos += 1;
        }
        Ok(out)
    }
}

/// First-index argmax over logits (deterministic tie-break).
fn argmax(logits: &[f32]) -> Result<u32> {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    if best_val == f32::NEG_INFINITY {
        return Err(RuvLLMError::Generation(
            "argmax over empty or all -inf logits".to_string(),
        ));
    }
    Ok(best as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax_deterministic_tiebreak() {
        assert_eq!(argmax(&[0.0, 1.0, 1.0, 0.5]).unwrap(), 1);
        assert!(argmax(&[]).is_err());
    }

    #[test]
    fn test_forward_shape() {
        let config = HrmTextConfig::tiny_test();
        let model = HrmTextModel::new_random(config.clone(), 42).unwrap();
        let mut caches = HrmKvCaches::new(&config);
        let ids = [3u32, 5, 7, 11];
        let logits = model
            .forward(&ids, AttentionMaskMode::Causal, &mut caches)
            .unwrap();
        assert_eq!(logits.len(), ids.len() * config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(caches.cached_len().unwrap(), ids.len());
    }

    #[test]
    fn test_cycle_override_invalidates_old_caches() {
        let config = HrmTextConfig::tiny_test();
        let mut model = HrmTextModel::new_random(config.clone(), 42).unwrap();
        let mut old_caches = HrmKvCaches::new(&config);
        model.cycle_override(3, 3).unwrap();
        let err = model.forward(&[1, 2], AttentionMaskMode::Causal, &mut old_caches);
        assert!(err.is_err());
        let mut new_caches = HrmKvCaches::new(&model.config);
        assert!(model
            .forward(&[1, 2], AttentionMaskMode::Causal, &mut new_caches)
            .is_ok());
        assert!(model.cycle_override(0, 1).is_err());
    }

    #[test]
    fn test_decode_requires_matching_position() {
        let config = HrmTextConfig::tiny_test();
        let model = HrmTextModel::new_random(config.clone(), 42).unwrap();
        let mut caches = HrmKvCaches::new(&config);
        model.prefill(&[3, 5, 7], 0, &mut caches).unwrap();
        // Caches hold 3 positions; decoding at pos 5 must fail.
        assert!(model.decode_step(9, 5, &mut caches).is_err());
        assert!(model.decode_step(9, 3, &mut caches).is_ok());
    }
}
