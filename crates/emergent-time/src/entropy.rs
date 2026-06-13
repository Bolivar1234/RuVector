//! Von Neumann / Shannon entropy helpers.
//!
//! Entropy is the monotone that several emergent-time constructions use as the
//! internal clock variable (the cold-atom toy universe in particular). All
//! logarithms are natural, so entropy is measured in nats.

use crate::complex_matrix::{hermitian_eigenvalues, CMatrix};
use crate::real_matrix::RealMatrix;

const EPS: f64 = 1e-12;

/// Shannon / von Neumann entropy `S = -Σ p_k ln p_k` from a probability
/// spectrum (density-matrix eigenvalues). Entries `<= EPS` contribute nothing,
/// matching the `lim_{p->0} p ln p = 0` convention.
pub fn entropy_from_spectrum(probs: &[f64]) -> f64 {
    let mut s = 0.0;
    for &p in probs {
        if p > EPS {
            s -= p * p.ln();
        }
    }
    s
}

/// Von Neumann entropy of a real symmetric density matrix.
pub fn von_neumann_entropy_real(rho: &RealMatrix) -> f64 {
    let (eigs, _v) = rho.symmetric_eigen();
    entropy_from_spectrum(&eigs)
}

/// Von Neumann entropy of a complex Hermitian density matrix.
pub fn von_neumann_entropy_hermitian(rho: &CMatrix) -> f64 {
    entropy_from_spectrum(&hermitian_eigenvalues(rho))
}

/// Purity `Tr(ρ²) = Σ p_k²`. Equals 1 for a pure state, `1/d` for the maximally
/// mixed state of dimension `d`.
pub fn purity_from_spectrum(probs: &[f64]) -> f64 {
    probs.iter().map(|p| p * p).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_state_zero_entropy() {
        assert!(entropy_from_spectrum(&[1.0, 0.0, 0.0]).abs() < 1e-12);
    }

    #[test]
    fn maximally_mixed_is_ln_d() {
        let d = 4;
        let probs = vec![1.0 / d as f64; d];
        let s = entropy_from_spectrum(&probs);
        assert!((s - (d as f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn real_density_entropy() {
        let rho = RealMatrix::diag(&[0.5, 0.5]);
        assert!((von_neumann_entropy_real(&rho) - 2.0f64.ln()).abs() < 1e-10);
    }
}
