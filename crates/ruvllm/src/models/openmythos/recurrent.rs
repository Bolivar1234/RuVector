//! The recurrent block: the deep loop with loop-index embedding, LTI-stable
//! state injection, per-depth LoRA, and Adaptive Computation Time halting.

use candle_core::{DType, Device, Tensor};
use candle_nn::{ops, Linear, Module, RmsNorm, VarBuilder};

use super::attention::KvLayerCache;
use super::block::TransformerBlock;
use super::config::MythosConfig;
use super::rope::cand;
use crate::error::Result;
use crate::models::rdt::DepthTelemetry;

/// LTI-constrained state update guaranteeing a contractive (spectral radius < 1)
/// recurrence: `h_{t+1} = A·h_t + B·e + transformer_out`, with diagonal
/// `A = exp(-exp(log_dt + log_A)) ∈ (0, 1)`.
pub struct LtiInjection {
    log_a: Tensor,
    log_dt: Tensor,
    b_gain: Tensor,
}

impl LtiInjection {
    pub fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        Ok(Self {
            log_a: vb.get(dim, "log_a").map_err(cand)?,
            log_dt: vb.get(dim, "log_dt").map_err(cand)?,
            b_gain: vb.get(dim, "b_gain").map_err(cand)?,
        })
    }

    /// Diagonal `A ∈ (0,1)^dim`. The pre-activation is clamped to `[-20, 20]`
    /// to avoid f32 overflow in the double exponential.
    pub fn a_diag(&self) -> Result<Tensor> {
        let s = (&self.log_dt + &self.log_a)
            .map_err(cand)?
            .clamp(-20.0, 20.0)
            .map_err(cand)?;
        s.exp().map_err(cand)?.neg().map_err(cand)?.exp().map_err(cand)
    }

    pub fn forward(&self, h: &Tensor, e: &Tensor, trans_out: &Tensor) -> Result<Tensor> {
        let a = self.a_diag()?;
        let a_h = h.broadcast_mul(&a).map_err(cand)?;
        let b_e = e.broadcast_mul(&self.b_gain).map_err(cand)?;
        ((a_h + b_e).map_err(cand)? + trans_out).map_err(cand)
    }
}

/// Per-depth LoRA: `delta = (down(x) ⊙ scale[t]) @ B`, with a learned scale row
/// per loop iteration (clamped to the last row beyond the training maximum).
pub struct DepthLora {
    down: Linear,
    b_mat: Tensor,     // [rank, dim]
    scale: Tensor,     // [max_loops, rank]
    rank: usize,
    max_rows: usize,
}

impl DepthLora {
    pub fn load(vb: VarBuilder, cfg: &MythosConfig) -> Result<Self> {
        let down = candle_nn::linear_no_bias(cfg.dim, cfg.lora_rank, vb.pp("down")).map_err(cand)?;
        let b_mat = vb.get((cfg.lora_rank, cfg.dim), "b_mat").map_err(cand)?;
        let scale = vb
            .get((cfg.max_loop_iters, cfg.lora_rank), "scale")
            .map_err(cand)?;
        Ok(Self {
            down,
            b_mat,
            scale,
            rank: cfg.lora_rank,
            max_rows: cfg.max_loop_iters,
        })
    }

    pub fn forward(&self, x: &Tensor, t: usize) -> Result<Tensor> {
        let (b, seq, _dim) = x.dims3().map_err(cand)?;
        let row = t.min(self.max_rows - 1);
        let scale_t = self
            .scale
            .narrow(0, row, 1)
            .map_err(cand)?
            .reshape((1, 1, self.rank))
            .map_err(cand)?; // [1,1,rank]
        let d = self.down.forward(x).map_err(cand)?; // [b,seq,rank]
        let d = d.broadcast_mul(&scale_t).map_err(cand)?;
        let dim = self.b_mat.dim(1).map_err(cand)?;
        let delta = d
            .reshape((b * seq, self.rank))
            .map_err(cand)?
            .matmul(&self.b_mat)
            .map_err(cand)?
            .reshape((b, seq, dim))
            .map_err(cand)?;
        Ok(delta)
    }
}

/// The recurrent block executed repeatedly by the model.
pub struct RecurrentBlock {
    inject_norm: RmsNorm,
    block: TransformerBlock,
    lti: LtiInjection,
    lora: DepthLora,
    act_head: Linear,
    loop_dim: usize,
    dim: usize,
    act_threshold: f32,
    rope_theta: f32,
}

/// Output of one recurrent forward: ACT-weighted state plus the updated KV cache
/// (final-iteration keys/values for the processed positions).
pub struct RecurrentOut {
    pub hidden: Tensor,
    pub kv: KvLayerCache,
}

impl RecurrentBlock {
    pub fn load(vb: VarBuilder, cfg: &MythosConfig) -> Result<Self> {
        Ok(Self {
            inject_norm: candle_nn::rms_norm(cfg.dim, cfg.rms_norm_eps, vb.pp("inject_norm"))
                .map_err(cand)?,
            block: TransformerBlock::load(vb.pp("block"), cfg, cfg.use_moe)?,
            lti: LtiInjection::load(vb.pp("lti"), cfg.dim)?,
            lora: DepthLora::load(vb.pp("lora"), cfg)?,
            act_head: candle_nn::linear(cfg.dim, 1, vb.pp("act_head")).map_err(cand)?,
            loop_dim: cfg.loop_dim,
            dim: cfg.dim,
            act_threshold: cfg.act_threshold,
            rope_theta: cfg.rope_theta,
        })
    }

    pub(crate) fn lti(&self) -> &LtiInjection {
        &self.lti
    }

    /// Sinusoidal loop-index embedding added to the first `loop_dim` channels.
    fn loop_embedding(&self, t: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let half = self.loop_dim / 2;
        let mut data = vec![0f32; self.dim];
        for i in 0..half {
            let freq = 1.0f32 / self.rope_theta.powf(2.0 * i as f32 / self.loop_dim as f32);
            let angle = t as f32 * freq;
            data[i] = angle.sin();
            data[half + i] = angle.cos();
        }
        Tensor::from_vec(data, (1, 1, self.dim), device)
            .map_err(cand)?
            .to_dtype(dtype)
            .map_err(cand)
    }

    /// Run the recurrent loop. `past` is the recurrent KV cache for prior
    /// positions; `offset` is its length. `cos`/`sin`/`mask_fn` cover the current
    /// query positions over `offset + seq` keys. Returns the ACT-weighted state
    /// and the updated KV cache, recording per-token depth in `telemetry`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        h0: &Tensor,
        e: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        past: Option<&KvLayerCache>,
        n_loops: usize,
        telemetry: &DepthTelemetry,
    ) -> Result<RecurrentOut> {
        let (b, seq, _dim) = h0.dims3().map_err(cand)?;
        let device = h0.device().clone();
        let dtype = h0.dtype();
        let n = b * seq;

        let mut h = h0.clone();
        let mut h_out = h0.zeros_like().map_err(cand)?;
        let mut cumulative = vec![0f32; n];
        let mut halted = vec![false; n];
        let mut depth = vec![0usize; n];
        // KV cache of the final iteration (updated every iteration; the last one
        // wins, matching "cache the final recurrent state per position").
        let mut last_kv: Option<KvLayerCache> = None;

        for t in 0..n_loops {
            let loop_emb = self.loop_embedding(t, &device, dtype)?;
            let h_loop = h.broadcast_add(&loop_emb).map_err(cand)?;
            let injected = (h_loop + e).map_err(cand)?;
            let normed = self.inject_norm.forward(&injected).map_err(cand)?;
            let (trans_out, kv) = self.block.forward(&normed, cos, sin, mask, past)?;
            last_kv = Some(kv);
            let trans_out = (trans_out + self.lora.forward(&normed, t)?).map_err(cand)?;

            // Stable state update.
            h = self.lti.forward(&h, e, &trans_out)?;

            // ACT halting probability per token.
            let p = ops::sigmoid(&self.act_head.forward(&h).map_err(cand)?).map_err(cand)?;
            let p_vals: Vec<f32> = p.reshape((n,)).map_err(cand)?.to_vec1().map_err(cand)?;

            let mut weight = vec![0f32; n];
            let mut all_halted = true;
            for i in 0..n {
                if halted[i] {
                    continue;
                }
                depth[i] += 1;
                let pi = p_vals[i];
                if cumulative[i] + pi >= self.act_threshold {
                    weight[i] = (1.0 - cumulative[i]).max(0.0);
                    halted[i] = true;
                } else {
                    weight[i] = pi;
                    cumulative[i] += pi;
                    all_halted = false;
                }
            }

            let w = Tensor::from_vec(weight, (b, seq, 1), &device)
                .map_err(cand)?
                .to_dtype(dtype)
                .map_err(cand)?;
            h_out = (h_out + h.broadcast_mul(&w).map_err(cand)?).map_err(cand)?;

            tracing::trace!(loop = t, all_halted, "openmythos recurrent step");
            if all_halted {
                break;
            }
        }

        // Tokens still running at the ceiling contribute their remainder.
        let mut tail = vec![0f32; n];
        let mut needs_tail = false;
        for i in 0..n {
            if !halted[i] {
                tail[i] = (1.0 - cumulative[i]).max(0.0);
                needs_tail = true;
            }
        }
        if needs_tail {
            let w = Tensor::from_vec(tail, (b, seq, 1), &device)
                .map_err(cand)?
                .to_dtype(dtype)
                .map_err(cand)?;
            h_out = (h_out + h.broadcast_mul(&w).map_err(cand)?).map_err(cand)?;
        }

        telemetry.record(&depth);
        let kv = last_kv.expect("at least one loop iteration runs");
        Ok(RecurrentOut { hidden: h_out, kv })
    }
}
