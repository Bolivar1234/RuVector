//! Feed-forward networks: dense SwiGLU expert and fine-grained MoE.

use candle_core::Tensor;
use candle_nn::{ops, Linear, Module, VarBuilder};

use super::config::MythosConfig;
use super::rope::cand;
use crate::error::Result;

/// A single SwiGLU expert: `down(silu(gate(x)) * up(x))`.
pub struct Expert {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Expert {
    pub fn load(vb: VarBuilder, dim: usize, inter: usize) -> Result<Self> {
        Ok(Self {
            gate: candle_nn::linear_no_bias(dim, inter, vb.pp("gate")).map_err(cand)?,
            up: candle_nn::linear_no_bias(dim, inter, vb.pp("up")).map_err(cand)?,
            down: candle_nn::linear_no_bias(inter, dim, vb.pp("down")).map_err(cand)?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let g = ops::silu(&self.gate.forward(xs).map_err(cand)?).map_err(cand)?;
        let u = self.up.forward(xs).map_err(cand)?;
        self.down.forward(&(g * u).map_err(cand)?).map_err(cand)
    }
}

/// Either a dense SwiGLU FFN (prelude/coda) or fine-grained MoE (recurrent).
pub enum Ffn {
    Dense(Expert),
    Moe(MoeFfn),
}

impl Ffn {
    pub fn load(vb: VarBuilder, cfg: &MythosConfig, use_moe: bool) -> Result<Self> {
        Ok(if use_moe {
            Ffn::Moe(MoeFfn::load(vb.pp("moe"), cfg)?)
        } else {
            let inter = cfg.expert_dim * cfg.n_shared_experts.max(2);
            Ffn::Dense(Expert::load(vb.pp("ffn"), cfg.dim, inter)?)
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Ffn::Dense(e) => e.forward(xs),
            Ffn::Moe(m) => m.forward(xs),
        }
    }
}

/// Fine-grained Mixture-of-Experts with routed + always-on shared experts.
///
/// Routing computes a softmax over experts and keeps the top-`k` per token
/// (others masked to zero weight, kept weights renormalized). Shared experts
/// always contribute. The routed path computes each selected expert and sums
/// the gated outputs — correct and simple; a production deployment would replace
/// the per-expert loop with a sparse dispatch/gather kernel.
pub struct MoeFfn {
    router: Linear,
    routed: Vec<Expert>,
    shared: Vec<Expert>,
    top_k: usize,
}

impl MoeFfn {
    pub fn load(vb: VarBuilder, cfg: &MythosConfig) -> Result<Self> {
        let router =
            candle_nn::linear_no_bias(cfg.dim, cfg.n_experts, vb.pp("router")).map_err(cand)?;
        let rvb = vb.pp("experts");
        let mut routed = Vec::with_capacity(cfg.n_experts);
        for i in 0..cfg.n_experts {
            routed.push(Expert::load(rvb.pp(i), cfg.dim, cfg.expert_dim)?);
        }
        let svb = vb.pp("shared_experts");
        let mut shared = Vec::with_capacity(cfg.n_shared_experts);
        for i in 0..cfg.n_shared_experts {
            shared.push(Expert::load(svb.pp(i), cfg.dim, cfg.expert_dim)?);
        }
        Ok(Self {
            router,
            routed,
            shared,
            top_k: cfg.n_experts_per_tok,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq, dim) = xs.dims3().map_err(cand)?;
        let n_tok = b * seq;
        let flat = xs.reshape((n_tok, dim)).map_err(cand)?;

        let logits = self.router.forward(&flat).map_err(cand)?;
        let probs = ops::softmax_last_dim(&logits).map_err(cand)?;
        let rows: Vec<Vec<f32>> = probs.to_vec2().map_err(cand)?;

        let n_experts = self.routed.len();
        let mut weights = vec![0f32; n_tok * n_experts];
        for (t, row) in rows.iter().enumerate() {
            let mut idx: Vec<usize> = (0..n_experts).collect();
            idx.sort_by(|&a, &c| row[c].partial_cmp(&row[a]).unwrap());
            let keep = &idx[..self.top_k.min(n_experts)];
            let denom: f32 = keep.iter().map(|&e| row[e]).sum::<f32>().max(1e-9);
            for &e in keep {
                weights[t * n_experts + e] = row[e] / denom;
            }
        }

        let mut out = flat.zeros_like().map_err(cand)?;
        for (e, expert) in self.routed.iter().enumerate() {
            let col: Vec<f32> = (0..n_tok).map(|t| weights[t * n_experts + e]).collect();
            if col.iter().all(|&w| w == 0.0) {
                continue;
            }
            let gate = Tensor::from_vec(col, (n_tok, 1), flat.device())
                .map_err(cand)?
                .to_dtype(flat.dtype())
                .map_err(cand)?;
            let y = expert.forward(&flat)?;
            out = (out + y.broadcast_mul(&gate).map_err(cand)?).map_err(cand)?;
        }
        for expert in &self.shared {
            out = (out + expert.forward(&flat)?).map_err(cand)?;
        }
        out.reshape((b, seq, dim)).map_err(cand)
    }
}
