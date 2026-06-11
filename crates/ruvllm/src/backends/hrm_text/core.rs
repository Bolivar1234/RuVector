//! Core building blocks for HRM-Text: the decoder layer (RMSNorm -> RoPE MHA
//! (GQA) -> residual -> RMSNorm -> SwiGLU MLP -> residual) and the H/L
//! recurrent core (ADR-199 Phase D items 1-3). Weight tensor primitives live
//! in [`super::tensor`].
//!
//! Pure CPU f32. Reuses the portable kernel entry points
//! (`apply_rope_neon`, `silu_vec`, and `rms_norm_neon` via
//! [`super::tensor::RmsNorm`] — each has a scalar fallback on non-aarch64),
//! and implements masked softmax attention locally because the existing
//! flash-attention kernels do not take a mask function.

use super::cache::KvLayerCache;
use super::config::{HrmTextConfig, NormPosition};
use super::mask::AttentionMaskMode;
use super::tensor::{Linear, RmsNorm, WeightRng};
use crate::error::{Result, RuvLLMError};
use crate::kernels::{apply_rope_neon, silu_vec};

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

/// One HRM-Text decoder layer: RMSNorm -> RoPE MHA (GQA, masked softmax,
/// per-slot KV cache) -> residual -> RMSNorm -> SwiGLU MLP -> residual.
/// Pre-norm by default; post-norm variant per [`NormPosition`].
#[derive(Debug, Clone)]
pub struct HrmDecoderLayer {
    /// Query projection `[num_heads * head_dim, hidden]`.
    pub q_proj: Linear,
    /// Key projection `[num_kv_heads * head_dim, hidden]`.
    pub k_proj: Linear,
    /// Value projection `[num_kv_heads * head_dim, hidden]`.
    pub v_proj: Linear,
    /// Output projection `[hidden, num_heads * head_dim]`.
    pub o_proj: Linear,
    /// SwiGLU gate projection `[intermediate, hidden]`.
    pub gate_proj: Linear,
    /// SwiGLU up projection `[intermediate, hidden]`.
    pub up_proj: Linear,
    /// SwiGLU down projection `[hidden, intermediate]`.
    pub down_proj: Linear,
    /// First norm (input norm for pre-norm; post-attention norm for post-norm).
    pub attn_norm: RmsNorm,
    /// Second norm (pre-MLP for pre-norm; post-MLP for post-norm).
    pub mlp_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    rope_theta: f32,
    norm_position: NormPosition,
}

impl HrmDecoderLayer {
    pub(crate) fn new_random(config: &HrmTextConfig, rng: &mut WeightRng) -> Self {
        let h = config.hidden_size;
        let head_dim = config.head_dim();
        let q_dim = config.num_attention_heads * head_dim;
        let kv_dim = config.kv_dim();
        Self {
            q_proj: Linear::new_random(h, q_dim, rng),
            k_proj: Linear::new_random(h, kv_dim, rng),
            v_proj: Linear::new_random(h, kv_dim, rng),
            o_proj: Linear::new_random(q_dim, h, rng),
            gate_proj: Linear::new_random(h, config.intermediate_size, rng),
            up_proj: Linear::new_random(h, config.intermediate_size, rng),
            down_proj: Linear::new_random(config.intermediate_size, h, rng),
            attn_norm: RmsNorm::new(h, config.rms_norm_eps),
            mlp_norm: RmsNorm::new(h, config.rms_norm_eps),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim,
            hidden_size: h,
            rope_theta: config.rope_theta,
            norm_position: config.norm_position,
        }
    }

    /// Layer forward. `hidden` is `[seq, hidden]`; `positions` are the
    /// absolute positions of those rows; new K/V is appended into `cache`.
    pub fn forward(
        &self,
        hidden: &[f32],
        positions: &[usize],
        mask: AttentionMaskMode,
        cache: &mut KvLayerCache,
    ) -> Result<Vec<f32>> {
        let seq = positions.len();
        if hidden.len() != seq * self.hidden_size {
            return Err(RuvLLMError::InvalidOperation(format!(
                "layer input length {} != seq {} * hidden {}",
                hidden.len(),
                seq,
                self.hidden_size
            )));
        }
        match self.norm_position {
            NormPosition::PreNorm => {
                let mut normed = hidden.to_vec();
                self.attn_norm.apply_rows(&mut normed)?;
                let attn = self.attention(&normed, positions, mask, cache)?;
                let mut x: Vec<f32> =
                    hidden.iter().zip(attn.iter()).map(|(a, b)| a + b).collect();
                let mut normed2 = x.clone();
                self.mlp_norm.apply_rows(&mut normed2)?;
                let mlp = self.mlp(&normed2)?;
                for (a, b) in x.iter_mut().zip(mlp.iter()) {
                    *a += b;
                }
                Ok(x)
            }
            NormPosition::PostNorm => {
                let attn = self.attention(hidden, positions, mask, cache)?;
                let mut x: Vec<f32> =
                    hidden.iter().zip(attn.iter()).map(|(a, b)| a + b).collect();
                self.attn_norm.apply_rows(&mut x)?;
                let mlp = self.mlp(&x)?;
                for (a, b) in x.iter_mut().zip(mlp.iter()) {
                    *a += b;
                }
                self.mlp_norm.apply_rows(&mut x)?;
                Ok(x)
            }
        }
    }

    /// Masked softmax attention with RoPE, GQA and per-slot KV caching.
    ///
    /// Cache invariant: the cache must already hold exactly `positions[0]`
    /// positions, and `positions` must be contiguous ascending — cached row
    /// index then equals absolute key position, which is what the mask
    /// function is evaluated against.
    fn attention(
        &self,
        x: &[f32],
        positions: &[usize],
        mask: AttentionMaskMode,
        cache: &mut KvLayerCache,
    ) -> Result<Vec<f32>> {
        let seq = positions.len();
        let hd = self.head_dim;
        let q_width = self.num_heads * hd;
        let kv_width = self.num_kv_heads * hd;

        let first = *positions.first().ok_or_else(|| {
            RuvLLMError::InvalidOperation("attention called with empty positions".to_string())
        })?;
        if cache.len() != first {
            return Err(RuvLLMError::KvCache(format!(
                "cache holds {} positions but chunk starts at position {}",
                cache.len(),
                first
            )));
        }
        if positions.windows(2).any(|w| w[1] != w[0] + 1) {
            return Err(RuvLLMError::InvalidOperation(
                "attention positions must be contiguous ascending".to_string(),
            ));
        }

        let mut q = self.q_proj.forward(x)?;
        let mut k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // RoPE on q and k, per (token, head). `apply_rope_neon` is the
        // portable kernel entry point (scalar fallback off-aarch64).
        for (t, &pos) in positions.iter().enumerate() {
            for h in 0..self.num_heads {
                let off = t * q_width + h * hd;
                apply_rope_neon(&mut q[off..off + hd], &[pos], hd, self.rope_theta);
            }
            for h in 0..self.num_kv_heads {
                let off = t * kv_width + h * hd;
                apply_rope_neon(&mut k[off..off + hd], &[pos], hd, self.rope_theta);
            }
        }

        cache.append(&k, &v)?;
        let kv_len = cache.len();
        let keys = cache.keys();
        let vals = cache.values();

        let group = self.num_heads / self.num_kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; seq * q_width];
        let mut scores = vec![0.0f32; kv_len];

        for (t, &qpos) in positions.iter().enumerate() {
            for h in 0..self.num_heads {
                let kvh = h / group;
                let qrow = &q[t * q_width + h * hd..t * q_width + (h + 1) * hd];

                // Masked scores with running max for stable softmax.
                let mut max_s = f32::NEG_INFINITY;
                for kp in 0..kv_len {
                    if mask.allowed(qpos, kp) {
                        let krow = &keys[kp * kv_width + kvh * hd..kp * kv_width + (kvh + 1) * hd];
                        let mut s = 0.0f32;
                        for (a, b) in qrow.iter().zip(krow.iter()) {
                            s += a * b;
                        }
                        let s = s * scale;
                        scores[kp] = s;
                        if s > max_s {
                            max_s = s;
                        }
                    } else {
                        scores[kp] = f32::NEG_INFINITY;
                    }
                }
                if max_s == f32::NEG_INFINITY {
                    // No visible key (cannot happen for valid causal inputs,
                    // since k <= q always includes the query itself).
                    continue;
                }
                let mut denom = 0.0f32;
                for s in scores.iter_mut() {
                    if s.is_finite() {
                        *s = (*s - max_s).exp();
                        denom += *s;
                    } else {
                        *s = 0.0;
                    }
                }
                let inv = 1.0 / denom;
                let orow = &mut out[t * q_width + h * hd..t * q_width + (h + 1) * hd];
                for (kp, &w) in scores.iter().enumerate() {
                    if w > 0.0 {
                        let vrow =
                            &vals[kp * kv_width + kvh * hd..kp * kv_width + (kvh + 1) * hd];
                        let wn = w * inv;
                        for (o, vv) in orow.iter_mut().zip(vrow.iter()) {
                            *o += wn * vv;
                        }
                    }
                }
            }
        }

        self.o_proj.forward(&out)
    }

    /// SwiGLU MLP: `down(silu(gate(x)) * up(x))`.
    fn mlp(&self, x: &[f32]) -> Result<Vec<f32>> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let mut act = silu_vec(&gate);
        for (a, u) in act.iter_mut().zip(up.iter()) {
            *a *= u;
        }
        self.down_proj.forward(&act)
    }
}

// ---------------------------------------------------------------------------
// Recurrent core
// ---------------------------------------------------------------------------

/// One HRM level (H or L): a stack of `half_layers` decoder layers applied to
/// `core(z + injection)` with per-layer KV caching into the given slot.
#[derive(Debug, Clone)]
pub struct HrmRecurrentCore {
    /// Decoder layers of this level.
    pub layers: Vec<HrmDecoderLayer>,
}

impl HrmRecurrentCore {
    pub(crate) fn new_random(config: &HrmTextConfig, rng: &mut WeightRng) -> Self {
        Self {
            layers: (0..config.half_layers)
                .map(|_| HrmDecoderLayer::new_random(config, rng))
                .collect(),
        }
    }

    /// Compute `core(z + injection)`, caching K/V per layer into `cache_slot`.
    /// `cache_slot` must hold exactly one [`KvLayerCache`] per layer.
    pub fn forward(
        &self,
        z: &[f32],
        injection: &[f32],
        positions: &[usize],
        mask: AttentionMaskMode,
        cache_slot: &mut [KvLayerCache],
    ) -> Result<Vec<f32>> {
        if z.len() != injection.len() {
            return Err(RuvLLMError::InvalidOperation(format!(
                "state/injection length mismatch: {} vs {}",
                z.len(),
                injection.len()
            )));
        }
        if cache_slot.len() != self.layers.len() {
            return Err(RuvLLMError::KvCache(format!(
                "cache slot has {} layer caches, core has {} layers",
                cache_slot.len(),
                self.layers.len()
            )));
        }
        // Additive input injection (upstream HRM "no-carry" forward).
        let mut x: Vec<f32> = z.iter().zip(injection.iter()).map(|(a, b)| a + b).collect();
        for (layer, cache) in self.layers.iter().zip(cache_slot.iter_mut()) {
            x = layer.forward(&x, positions, mask, cache)?;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_forward_shape_and_cache_growth() {
        let config = HrmTextConfig::tiny_test();
        let mut rng = WeightRng::new(1);
        let layer = HrmDecoderLayer::new_random(&config, &mut rng);
        let mut cache = KvLayerCache::new(config.kv_dim());
        let seq = 4;
        let x = vec![0.1f32; seq * config.hidden_size];
        let positions: Vec<usize> = (0..seq).collect();
        let y = layer
            .forward(&x, &positions, AttentionMaskMode::Causal, &mut cache)
            .unwrap();
        assert_eq!(y.len(), seq * config.hidden_size);
        assert_eq!(cache.len(), seq);
        // Decoding one more position appends exactly one row.
        let y2 = layer
            .forward(
                &x[..config.hidden_size],
                &[seq],
                AttentionMaskMode::Causal,
                &mut cache,
            )
            .unwrap();
        assert_eq!(y2.len(), config.hidden_size);
        assert_eq!(cache.len(), seq + 1);
    }

    #[test]
    fn test_layer_rejects_discontiguous_positions() {
        let config = HrmTextConfig::tiny_test();
        let mut rng = WeightRng::new(1);
        let layer = HrmDecoderLayer::new_random(&config, &mut rng);
        let mut cache = KvLayerCache::new(config.kv_dim());
        let x = vec![0.1f32; 2 * config.hidden_size];
        let err = layer.forward(&x, &[0, 2], AttentionMaskMode::Causal, &mut cache);
        assert!(err.is_err());
    }
}
