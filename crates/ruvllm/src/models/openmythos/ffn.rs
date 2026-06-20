//! Feed-forward networks: dense SwiGLU expert and fine-grained MoE.

use candle_core::Tensor;
use candle_nn::{ops, Linear, Module, VarBuilder};

use super::config::MythosConfig;
use super::rope::cand;
use crate::error::Result;
use crate::models::quantized::{QuantType, QuantizedLinear};

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

    /// Post-training quantize this expert's weights into a [`QuantizedExpert`].
    pub fn quantize(&self, qt: QuantType) -> Result<QuantizedExpert> {
        Ok(QuantizedExpert {
            gate: QuantizedLinear::from_linear(&self.gate, qt)?,
            up: QuantizedLinear::from_linear(&self.up, qt)?,
            down: QuantizedLinear::from_linear(&self.down, qt)?,
        })
    }
}

/// A SwiGLU expert with block-quantized weights (low-memory inference).
pub struct QuantizedExpert {
    gate: QuantizedLinear,
    up: QuantizedLinear,
    down: QuantizedLinear,
}

impl QuantizedExpert {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let g = ops::silu(&self.gate.forward(xs)?).map_err(cand)?;
        let u = self.up.forward(xs)?;
        self.down.forward(&(g * u).map_err(cand)?)
    }

    /// Total quantized weight storage in bytes.
    pub fn size_bytes(&self) -> usize {
        self.gate.size_bytes() + self.up.size_bytes() + self.down.size_bytes()
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
/// (kept weights renormalized). Shared experts always contribute. Each routed
/// expert runs only on the tokens routed to it (sparse dispatch via
/// `index_select` gather + `index_add` scatter), so FFN compute scales with
/// `top_k`, not `n_experts`.
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
        let device = xs.device().clone();
        let dtype = xs.dtype();
        let flat = xs.reshape((n_tok, dim)).map_err(cand)?;

        // Router probabilities (kept on-device, grad-carrying). NOTE: use the
        // differentiable `softmax` (not `softmax_last_dim`, which is `no_bwd` and
        // would silently sever the router's gradient). Cast to F32 so the
        // host-side top-k selection is precision-independent.
        let logits = self.router.forward(&flat).map_err(cand)?;
        let probs = ops::softmax(&logits, candle_core::D::Minus1)
            .map_err(cand)?
            .to_dtype(candle_core::DType::F32)
            .map_err(cand)?;
        let rows: Vec<Vec<f32>> = probs.to_vec2().map_err(cand)?;

        // Top-k selection (non-differentiable, as in standard MoE). Build a
        // host keep-mask and the per-expert token lists for sparse dispatch.
        let n_experts = self.routed.len();
        let mut tok_ids: Vec<Vec<u32>> = vec![Vec::new(); n_experts];
        let mut mask = vec![0f32; n_tok * n_experts];
        let mut order: Vec<usize> = (0..n_experts).collect();
        for (t, row) in rows.iter().enumerate() {
            order.sort_by(|&a, &c| row[c].partial_cmp(&row[a]).unwrap());
            let keep = &order[..self.top_k.min(n_experts)];
            for &e in keep {
                tok_ids[e].push(t as u32);
                mask[t * n_experts + e] = 1.0;
            }
        }

        // Gate magnitudes stay differentiable: renormalize the kept router probs
        // via tensor ops so gradient flows back to the router weights.
        let mask_t = Tensor::from_vec(mask, (n_tok, n_experts), &device).map_err(cand)?;
        let masked = (&probs * &mask_t).map_err(cand)?;
        let denom = masked
            .sum_keepdim(1)
            .map_err(cand)?
            .clamp(1e-9, f32::INFINITY as f64)
            .map_err(cand)?;
        let gate_full = masked.broadcast_div(&denom).map_err(cand)?; // [n_tok, n_experts]

        // Sparse dispatch: each expert processes only its routed tokens, so FFN
        // compute scales with top_k rather than n_experts.
        let mut out = flat.zeros_like().map_err(cand)?;
        for (e, expert) in self.routed.iter().enumerate() {
            let n_e = tok_ids[e].len();
            if n_e == 0 {
                continue;
            }
            let idx = Tensor::from_vec(tok_ids[e].clone(), (n_e,), &device).map_err(cand)?;
            let gathered = flat.index_select(&idx, 0).map_err(cand)?; // [n_e, dim]
            let y = expert.forward(&gathered)?;
            // Differentiable gate for this expert's tokens: [n_e, 1].
            // `narrow` yields a non-contiguous view; index_select needs contiguous.
            let gate_e = gate_full
                .narrow(1, e, 1)
                .map_err(cand)?
                .contiguous()
                .map_err(cand)?
                .index_select(&idx, 0)
                .map_err(cand)?
                .to_dtype(dtype)
                .map_err(cand)?;
            let y = y.broadcast_mul(&gate_e).map_err(cand)?;
            out = out.index_add(&idx, &y, 0).map_err(cand)?;
        }

        // Shared experts always contribute (dense over all tokens).
        for expert in &self.shared {
            out = (out + expert.forward(&flat)?).map_err(cand)?;
        }
        out.reshape((b, seq, dim)).map_err(cand)
    }

    /// Post-training quantize all experts (router stays dense f32).
    pub fn quantize(&self, qt: QuantType) -> Result<QuantizedMoeFfn> {
        let routed = self
            .routed
            .iter()
            .map(|e| e.quantize(qt))
            .collect::<Result<Vec<_>>>()?;
        let shared = self
            .shared
            .iter()
            .map(|e| e.quantize(qt))
            .collect::<Result<Vec<_>>>()?;
        Ok(QuantizedMoeFfn {
            router: self.router.clone(),
            routed,
            shared,
            top_k: self.top_k,
        })
    }
}

/// MoE FFN with block-quantized experts (router stays dense). Inference-only;
/// routing weights are applied as host scalars (no gradient path needed).
pub struct QuantizedMoeFfn {
    router: Linear,
    routed: Vec<QuantizedExpert>,
    shared: Vec<QuantizedExpert>,
    top_k: usize,
}

impl QuantizedMoeFfn {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq, dim) = xs.dims3().map_err(cand)?;
        let n_tok = b * seq;
        let device = xs.device().clone();
        let dtype = xs.dtype();
        let flat = xs.reshape((n_tok, dim)).map_err(cand)?;

        let logits = self.router.forward(&flat).map_err(cand)?;
        let probs = ops::softmax_last_dim(&logits)
            .map_err(cand)?
            .to_dtype(candle_core::DType::F32)
            .map_err(cand)?;
        let rows: Vec<Vec<f32>> = probs.to_vec2().map_err(cand)?;

        let n_experts = self.routed.len();
        let mut tok_ids: Vec<Vec<u32>> = vec![Vec::new(); n_experts];
        let mut tok_w: Vec<Vec<f32>> = vec![Vec::new(); n_experts];
        let mut order: Vec<usize> = (0..n_experts).collect();
        for (t, row) in rows.iter().enumerate() {
            order.sort_by(|&a, &c| row[c].partial_cmp(&row[a]).unwrap());
            let keep = &order[..self.top_k.min(n_experts)];
            let denom: f32 = keep.iter().map(|&e| row[e]).sum::<f32>().max(1e-9);
            for &e in keep {
                tok_ids[e].push(t as u32);
                tok_w[e].push(row[e] / denom);
            }
        }

        let mut out = flat.zeros_like().map_err(cand)?;
        for (e, expert) in self.routed.iter().enumerate() {
            let n_e = tok_ids[e].len();
            if n_e == 0 {
                continue;
            }
            let idx = Tensor::from_vec(tok_ids[e].clone(), (n_e,), &device).map_err(cand)?;
            let gathered = flat.index_select(&idx, 0).map_err(cand)?;
            let y = expert.forward(&gathered)?;
            let w = Tensor::from_vec(tok_w[e].clone(), (n_e, 1), &device)
                .map_err(cand)?
                .to_dtype(dtype)
                .map_err(cand)?;
            out = out.index_add(&idx, &y.broadcast_mul(&w).map_err(cand)?, 0).map_err(cand)?;
        }
        for expert in &self.shared {
            out = (out + expert.forward(&flat)?).map_err(cand)?;
        }
        out.reshape((b, seq, dim)).map_err(cand)
    }

    /// Total quantized expert storage in bytes.
    pub fn size_bytes(&self) -> usize {
        self.routed.iter().map(|e| e.size_bytes()).sum::<usize>()
            + self.shared.iter().map(|e| e.size_bytes()).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::quantized::dense_bytes;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn quantized_expert_approximates_dense_and_saves_memory() {
        let (dim, inter) = (64usize, 64usize);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let expert = Expert::load(vb, dim, inter).unwrap();

        let xs = Tensor::randn(0f32, 1.0, (4, dim), &Device::Cpu).unwrap();
        let dense: Vec<f32> = expert.forward(&xs).unwrap().flatten_all().unwrap().to_vec1().unwrap();

        let q = expert.quantize(QuantType::Q8_0).unwrap();
        let qy: Vec<f32> = q.forward(&xs).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert!(qy.iter().all(|x| x.is_finite()));

        // Relative L2 error should be small for Q8_0.
        let num: f32 = dense.iter().zip(qy.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        let den: f32 = dense.iter().map(|a| a * a).sum::<f32>().max(1e-9);
        let rel_l2 = (num / den).sqrt();
        assert!(rel_l2 < 0.15, "quantized expert rel-L2 too high: {rel_l2}");

        // Memory: Q8_0 (~1 byte/weight) well under f32 (4 bytes/weight).
        let dense_bytes_total = dense_bytes(inter, dim) * 2 + dense_bytes(dim, inter);
        assert!(q.size_bytes() < dense_bytes_total / 3, "insufficient memory savings");
    }

    #[test]
    fn moe_router_receives_gradient() {
        // Regression: the sparse-dispatch gate must stay differentiable so the
        // router trains. (Previously the gate was built from host floats, cutting
        // the router's gradient entirely.)
        let cfg = MythosConfig::tiny();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let moe = MoeFfn::load(vb.pp("moe"), &cfg).unwrap();

        let xs = Tensor::randn(0f32, 1.0, (1, 4, cfg.dim), &Device::Cpu).unwrap();
        let loss = moe.forward(&xs).unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let g = grads
            .get(moe.router.weight())
            .expect("router weight must receive a gradient");
        let gv: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            gv.iter().any(|&x| x.abs() > 1e-9),
            "router gradient is entirely zero"
        );
    }

    #[test]
    fn quantized_moe_matches_dense_and_saves_memory() {
        let mut cfg = MythosConfig::tiny();
        cfg.dim = 64; // block-size-aligned for Q8_0
        cfg.expert_dim = 64;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let moe = MoeFfn::load(vb.pp("moe"), &cfg).unwrap();

        let xs = Tensor::randn(0f32, 1.0, (1, 4, cfg.dim), &Device::Cpu).unwrap();
        let dense: Vec<f32> = moe.forward(&xs).unwrap().flatten_all().unwrap().to_vec1().unwrap();

        let q = moe.quantize(QuantType::Q8_0).unwrap();
        let qy: Vec<f32> = q.forward(&xs).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert!(qy.iter().all(|x| x.is_finite()));

        let num: f32 = dense.iter().zip(qy.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        let den: f32 = dense.iter().map(|a| a * a).sum::<f32>().max(1e-9);
        let rel_l2 = (num / den).sqrt();
        assert!(rel_l2 < 0.2, "quantized MoE rel-L2 too high: {rel_l2}");

        // 5 experts (4 routed + 1 shared) × 3 matrices of 64×64 f32.
        let dense_total = 5 * 3 * dense_bytes(64, 64);
        assert!(q.size_bytes() < dense_total / 3, "insufficient MoE memory savings");
    }
}
