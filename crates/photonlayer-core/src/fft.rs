//! Dependency-free, deterministic FFT (iterative radix-2 Cooley–Tukey).
//!
//! Restricted to power-of-two transform lengths. The optical engine pads
//! grids to powers of two before propagation, which keeps the transform
//! exact and bit-for-bit reproducible across platforms (no FFT library
//! threading or SIMD-order nondeterminism).

use crate::complex::Complex;
use core::f32::consts::PI;

/// Returns true if `n` is a power of two and non-zero.
#[inline]
pub fn is_pow2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// In-place 1D FFT. `inverse = true` computes the inverse transform and
/// applies the `1/N` normalization so that `ifft(fft(x)) == x`.
///
/// # Panics
/// Panics if `data.len()` is not a power of two.
pub fn fft_1d(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    assert!(is_pow2(n), "FFT length must be a power of two, got {n}");
    if n == 1 {
        return;
    }

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }

    // Danielson–Lanczos butterflies.
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let ang = sign * 2.0 * PI / len as f32;
        let wlen = Complex::from_phase(ang);
        let half = len / 2;
        let mut i = 0;
        while i < n {
            let mut w = Complex::ONE;
            for k in 0..half {
                let u = data[i + k];
                let v = data[i + k + half] * w;
                data[i + k] = u + v;
                data[i + k + half] = u - v;
                w = w * wlen;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv = 1.0 / n as f32;
        for c in data.iter_mut() {
            *c = c.scale(inv);
        }
    }
}

/// In-place 2D FFT on a row-major `width * height` buffer.
///
/// Both dimensions must be powers of two. Performs row transforms followed
/// by column transforms (separable DFT).
pub fn fft_2d(data: &mut [Complex], width: usize, height: usize, inverse: bool) {
    assert_eq!(data.len(), width * height, "buffer size mismatch");
    assert!(is_pow2(width) && is_pow2(height), "dims must be power of two");

    // Rows.
    for r in 0..height {
        let row = &mut data[r * width..(r + 1) * width];
        fft_1d(row, inverse);
    }

    // Columns (gather/scatter to keep the 1D kernel contiguous).
    let mut col = vec![Complex::ZERO; height];
    for c in 0..width {
        for r in 0..height {
            col[r] = data[r * width + c];
        }
        fft_1d(&mut col, inverse);
        for r in 0..height {
            data[r * width + c] = col[r];
        }
    }
}

/// 2D fftshift: swaps quadrants so the zero-frequency component moves to the
/// center. `width` and `height` must be even (always true for power-of-two).
pub fn fftshift_2d(data: &mut [Complex], width: usize, height: usize) {
    let hw = width / 2;
    let hh = height / 2;
    let mut out = vec![Complex::ZERO; data.len()];
    for r in 0..height {
        let nr = (r + hh) % height;
        for c in 0..width {
            let nc = (c + hw) % width;
            out[nr * width + nc] = data[r * width + c];
        }
    }
    data.copy_from_slice(&out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_1d() {
        let mut x: Vec<Complex> = (0..8).map(|i| Complex::new(i as f32, 0.0)).collect();
        let orig = x.clone();
        fft_1d(&mut x, false);
        fft_1d(&mut x, true);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a.re - b.re).abs() < 1e-4, "{a:?} vs {b:?}");
            assert!(a.im.abs() < 1e-4);
        }
    }

    #[test]
    fn dc_component_is_sum() {
        // FFT of a constant signal -> all energy in bin 0.
        let mut x = vec![Complex::new(2.0, 0.0); 16];
        fft_1d(&mut x, false);
        assert!((x[0].re - 32.0).abs() < 1e-3);
        for c in &x[1..] {
            assert!(c.abs() < 1e-3);
        }
    }

    #[test]
    fn roundtrip_2d() {
        let (w, h) = (8, 4);
        let mut x: Vec<Complex> = (0..w * h)
            .map(|i| Complex::new((i % 5) as f32, 0.0))
            .collect();
        let orig = x.clone();
        fft_2d(&mut x, w, h, false);
        fft_2d(&mut x, w, h, true);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a.re - b.re).abs() < 1e-3);
        }
    }
}
