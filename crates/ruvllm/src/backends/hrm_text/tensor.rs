//! Plain-`Vec<f32>` weight tensors and deterministic seeded initialization
//! for the HRM-Text backend (ADR-199 Phase D).
//!
//! Pure CPU f32, no candle dependency. RMSNorm routes through the portable
//! `rms_norm_neon` kernel entry point (scalar fallback off-aarch64).

use crate::error::{Result, RuvLLMError};
use crate::kernels::rms_norm_neon;

// ---------------------------------------------------------------------------
// Deterministic weight initialization
// ---------------------------------------------------------------------------

/// Tiny xorshift64*-based RNG for deterministic, dependency-stable random
/// weight init (truncated-normal-ish: Box-Muller clamped to +/- 2 sigma).
#[derive(Debug, Clone)]
pub(crate) struct WeightRng {
    state: u64,
}

impl WeightRng {
    pub(crate) fn new(seed: u64) -> Self {
        // Avoid the all-zero fixed point and decorrelate small seeds.
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        state ^= state >> 33;
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f32(&mut self) -> f32 {
        // Uniform in (0, 1] (never exactly 0, safe for ln()).
        let bits = (self.next_u64() >> 40) as u32; // 24 random bits
        (bits as f32 + 1.0) / 16_777_217.0
    }

    /// Gaussian sample (Box-Muller) clamped to +/- 2 sigma.
    pub(crate) fn next_gaussian(&mut self, sigma: f32) -> f32 {
        let u1 = self.next_f32();
        let u2 = self.next_f32();
        let mag = (-2.0 * u1.ln()).sqrt();
        let z = mag * (2.0 * std::f32::consts::PI * u2).cos();
        (z * sigma).clamp(-2.0 * sigma, 2.0 * sigma)
    }

    pub(crate) fn fill_gaussian(&mut self, out: &mut [f32], sigma: f32) {
        for v in out.iter_mut() {
            *v = self.next_gaussian(sigma);
        }
    }
}

// ---------------------------------------------------------------------------
// Weight tensors
// ---------------------------------------------------------------------------

/// Token embedding table, `[vocab_size, dim]` row-major.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Flat weights.
    pub weight: Vec<f32>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding dimension.
    pub dim: usize,
}

impl Embedding {
    /// Embedding row for `token`.
    pub fn lookup(&self, token: u32) -> Result<&[f32]> {
        let idx = token as usize;
        if idx >= self.vocab_size {
            return Err(RuvLLMError::InvalidOperation(format!(
                "token id {} out of vocabulary ({})",
                token, self.vocab_size
            )));
        }
        Ok(&self.weight[idx * self.dim..(idx + 1) * self.dim])
    }
}

/// Dense linear layer with `[out, in]` row-major weights (no bias).
#[derive(Debug, Clone)]
pub struct Linear {
    /// Flat weights, `[out, in]` row-major.
    pub w: Vec<f32>,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
}

impl Linear {
    pub(crate) fn new_random(in_dim: usize, out_dim: usize, rng: &mut WeightRng) -> Self {
        let sigma = 1.0 / (in_dim as f32).sqrt();
        let mut w = vec![0.0f32; in_dim * out_dim];
        rng.fill_gaussian(&mut w, sigma);
        Self { w, in_dim, out_dim }
    }

    /// `input` is `[n, in_dim]` row-major; returns `[n, out_dim]`.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if self.in_dim == 0 || input.len() % self.in_dim != 0 {
            return Err(RuvLLMError::InvalidOperation(format!(
                "Linear input length {} not a multiple of in_dim {}",
                input.len(),
                self.in_dim
            )));
        }
        let n = input.len() / self.in_dim;
        let mut out = vec![0.0f32; n * self.out_dim];
        for r in 0..n {
            let x = &input[r * self.in_dim..(r + 1) * self.in_dim];
            let dst = &mut out[r * self.out_dim..(r + 1) * self.out_dim];
            for (o, d) in dst.iter_mut().enumerate() {
                let wrow = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
                let mut sum = 0.0f32;
                for (a, b) in x.iter().zip(wrow.iter()) {
                    sum += a * b;
                }
                *d = sum;
            }
        }
        Ok(out)
    }
}

/// RMSNorm weights (`gamma`) plus epsilon. Applies the portable
/// `rms_norm_neon` kernel per row.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// Learnable scale, `[dim]`.
    pub gamma: Vec<f32>,
    /// Numerical-stability epsilon.
    pub eps: f32,
}

impl RmsNorm {
    pub(crate) fn new(dim: usize, eps: f32) -> Self {
        Self {
            gamma: vec![1.0; dim],
            eps,
        }
    }

    /// Normalize `x` in place, treating it as rows of `gamma.len()` elements.
    pub fn apply_rows(&self, x: &mut [f32]) -> Result<()> {
        let dim = self.gamma.len();
        if dim == 0 || x.len() % dim != 0 {
            return Err(RuvLLMError::InvalidOperation(format!(
                "RmsNorm input length {} not a multiple of dim {}",
                x.len(),
                dim
            )));
        }
        for row in x.chunks_mut(dim) {
            rms_norm_neon(row, &self.gamma, self.eps);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_rng_deterministic_and_clamped() {
        let mut a = WeightRng::new(42);
        let mut b = WeightRng::new(42);
        for _ in 0..256 {
            let x = a.next_gaussian(0.1);
            assert_eq!(x, b.next_gaussian(0.1));
            assert!(x >= -0.2 && x <= 0.2);
        }
        let mut c = WeightRng::new(43);
        let differs = (0..16).any(|_| a.next_gaussian(0.1) != c.next_gaussian(0.1));
        assert!(differs);
    }

    #[test]
    fn test_linear_shapes() {
        let mut rng = WeightRng::new(7);
        let lin = Linear::new_random(4, 3, &mut rng);
        let out = lin.forward(&[1.0; 8]).unwrap(); // 2 rows
        assert_eq!(out.len(), 6);
        assert!(lin.forward(&[1.0; 7]).is_err());
    }

    #[test]
    fn test_embedding_bounds() {
        let emb = Embedding {
            weight: vec![0.0; 4 * 8],
            vocab_size: 4,
            dim: 8,
        };
        assert!(emb.lookup(3).is_ok());
        assert!(emb.lookup(4).is_err());
    }

    #[test]
    fn test_rms_norm_rows() {
        let norm = RmsNorm::new(4, 1e-5);
        let mut x = vec![1.0f32; 8]; // two rows
        norm.apply_rows(&mut x).unwrap();
        // RMS of all-ones is 1, so output stays ~1.0 with gamma=1.
        assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-3));
        let mut bad = vec![0.0f32; 7];
        assert!(norm.apply_rows(&mut bad).is_err());
    }
}
