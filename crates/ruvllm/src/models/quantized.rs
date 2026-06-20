//! Post-training quantization for recurrent-depth model weights.
//!
//! Provides [`QuantizedLinear`] — a drop-in replacement for a bias-free
//! `candle_nn::Linear` whose weights are stored in a GGML block-quantized format
//! (Q4_0 / Q8_0 / Q4_K / …) and multiplied via `candle`'s `QMatMul` kernels.
//! This is the building block for low-memory inference: an f32 projection of
//! `out × in` weights (4 bytes each) becomes ~0.5–1.1 bytes/weight.
//!
//! Quantization runs along the input dimension in blocks, so `in_dim` must be a
//! multiple of the format's [`QuantType::block_size`].

#![cfg(feature = "candle")]

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Module, Tensor};
use candle_nn::Linear;

use crate::error::{Result, RuvLLMError};

fn cand(e: candle_core::Error) -> RuvLLMError {
    RuvLLMError::Quantization(format!("candle (quant): {e}"))
}

/// Supported weight quantization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4K,
    Q6K,
    F16,
}

impl QuantType {
    /// The underlying candle GGML dtype.
    pub fn ggml(self) -> GgmlDType {
        match self {
            QuantType::Q4_0 => GgmlDType::Q4_0,
            QuantType::Q4_1 => GgmlDType::Q4_1,
            QuantType::Q5_0 => GgmlDType::Q5_0,
            QuantType::Q5_1 => GgmlDType::Q5_1,
            QuantType::Q8_0 => GgmlDType::Q8_0,
            QuantType::Q4K => GgmlDType::Q4K,
            QuantType::Q6K => GgmlDType::Q6K,
            QuantType::F16 => GgmlDType::F16,
        }
    }

    /// Block size (the quantized dimension must be a multiple of this).
    pub fn block_size(self) -> usize {
        self.ggml().block_size()
    }
}

/// A bias-free linear layer with block-quantized weights.
pub struct QuantizedLinear {
    inner: QMatMul,
    in_dim: usize,
    out_dim: usize,
    bytes: usize,
}

impl QuantizedLinear {
    /// Quantize an `[out, in]` weight tensor.
    pub fn from_weight(weight: &Tensor, qt: QuantType) -> Result<Self> {
        let (out_dim, in_dim) = weight.dims2().map_err(cand)?;
        let bs = qt.block_size();
        if in_dim % bs != 0 {
            return Err(RuvLLMError::Quantization(format!(
                "QuantizedLinear: in_dim ({in_dim}) must be a multiple of block_size ({bs}) for {qt:?}"
            )));
        }
        let w = weight.contiguous().map_err(cand)?;
        let qtensor = QTensor::quantize(&w, qt.ggml()).map_err(cand)?;
        let bytes = qtensor.storage_size_in_bytes();
        let inner = QMatMul::from_qtensor(qtensor).map_err(cand)?;
        Ok(Self {
            inner,
            in_dim,
            out_dim,
            bytes,
        })
    }

    /// Quantize the weights of an existing (bias-free) [`Linear`].
    pub fn from_linear(linear: &Linear, qt: QuantType) -> Result<Self> {
        Self::from_weight(linear.weight(), qt)
    }

    /// `xs @ Wᵀ` over `[..., in]` → `[..., out]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs).map_err(cand)
    }

    pub fn in_dim(&self) -> usize {
        self.in_dim
    }
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Quantized storage size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.bytes
    }
}

/// f32 dense storage size of an `out × in` weight, in bytes.
pub fn dense_bytes(out_dim: usize, in_dim: usize) -> usize {
    out_dim * in_dim * 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn weight(out: usize, inp: usize) -> Tensor {
        // Deterministic small-magnitude weights so quantization error is bounded.
        let n = out * inp;
        let data: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) / 16.0).collect();
        Tensor::from_vec(data, (out, inp), &Device::Cpu).unwrap()
    }

    #[test]
    fn q8_matmul_approximates_dense() {
        let w = weight(64, 64);
        let xs = Tensor::from_vec(
            (0..2 * 64).map(|i| ((i % 13) as f32 - 6.0) / 12.0).collect::<Vec<_>>(),
            (2, 64),
            &Device::Cpu,
        )
        .unwrap();

        let dense = xs.matmul(&w.t().unwrap()).unwrap();
        let ql = QuantizedLinear::from_weight(&w, QuantType::Q8_0).unwrap();
        let q = ql.forward(&xs).unwrap();

        let d: Vec<f32> = dense.flatten_all().unwrap().to_vec1().unwrap();
        let qv: Vec<f32> = q.flatten_all().unwrap().to_vec1().unwrap();
        let max_diff = d
            .iter()
            .zip(qv.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(qv.iter().all(|x| x.is_finite()));
        assert!(max_diff < 0.05, "Q8_0 deviates too much from dense: {max_diff}");
    }

    #[test]
    fn q4_is_finite_and_smaller_than_q8() {
        let w = weight(128, 64);
        let q4 = QuantizedLinear::from_weight(&w, QuantType::Q4_0).unwrap();
        let q8 = QuantizedLinear::from_weight(&w, QuantType::Q8_0).unwrap();
        let xs = Tensor::ones((3, 64), DType::F32, &Device::Cpu).unwrap();
        assert!(q4
            .forward(&xs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite()));

        // Memory ordering: Q4_0 < Q8_0 < dense f32.
        let dense = dense_bytes(128, 64);
        assert!(q4.size_bytes() < q8.size_bytes());
        assert!(q8.size_bytes() < dense);
        // Q4_0 should be roughly 1/8th of f32 (4.5 bits/weight incl. scales).
        assert!(q4.size_bytes() * 6 < dense, "Q4_0 not compact enough");
    }

    #[test]
    fn rejects_unaligned_in_dim() {
        let w = weight(32, 30); // 30 not divisible by 32
        assert!(QuantizedLinear::from_weight(&w, QuantType::Q8_0).is_err());
    }

    #[test]
    fn from_linear_quantizes_weights() {
        use candle_nn::{Linear, Module};
        let w = weight(64, 64);
        let lin = Linear::new(w.clone(), None);
        let ql = QuantizedLinear::from_linear(&lin, QuantType::Q8_0).unwrap();
        assert_eq!(ql.in_dim(), 64);
        assert_eq!(ql.out_dim(), 64);

        let xs = Tensor::ones((1, 64), candle_core::DType::F32, &Device::Cpu).unwrap();
        let dense = lin.forward(&xs).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let q = ql.forward(&xs).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let max_diff = dense.iter().zip(q.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max_diff < 0.05);
    }
}
