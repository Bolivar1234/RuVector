//! Page–Wootters relational time.
//!
//! Time is not assumed; it emerges from correlations inside a globally
//! *static* entangled state of clock ⊗ rest.
//!
//! ## Exact construction
//!
//! Diagonalize the system Hamiltonian `H_R = Σ_k E_k |E_k><E_k|`. Build a clock
//! of the same dimension whose energy eigenstate `|k>_C` carries energy `-E_k`,
//! i.e. `H_C = diag(-E_0, …, -E_{d-1})`. The global state
//!
//! ```text
//!   |Ψ> = (1/√d) Σ_k |k>_C ⊗ |E_k>_R
//! ```
//!
//! satisfies the Wheeler–DeWitt constraint **exactly**:
//!
//! ```text
//!   Ĵ|Ψ> = (H_C ⊗ I + I ⊗ H_R)|Ψ> = (1/√d) Σ_k (-E_k + E_k)|k>|E_k> = 0.
//! ```
//!
//! Yet an internal observer who reads the clock *pointer* state
//! `|t>_C = Σ_k e^{iE_k t}|k>_C` recovers Schrödinger evolution:
//!
//! ```text
//!   <t|Ψ>_R ∝ Σ_k e^{-iE_k t}|E_k>_R = e^{-iH_R t}|ψ_0>,   |ψ_0> = Σ_k|E_k>.
//! ```
//!
//! Evolution is what the rest sector *looks like* conditioned on the clock.

use crate::complex::{normalized, Complex};
use crate::complex_matrix::schrodinger_propagator;
use crate::real_matrix::RealMatrix;
use crate::state::condition_on_clock;

/// A relational clock paired with a system Hamiltonian.
pub struct PageWootters {
    /// System (rest-sector) Hamiltonian.
    pub h_r: RealMatrix,
    /// System energy levels `E_k`.
    pub energies: Vec<f64>,
    /// Energy eigenvectors as columns (`|E_k>` is column `k`).
    pub vecs: RealMatrix,
    /// Hilbert-space dimension of each sector.
    pub dim: usize,
}

impl PageWootters {
    /// Build from a real symmetric system Hamiltonian.
    pub fn new(h_r: RealMatrix) -> Self {
        let (energies, vecs) = h_r.symmetric_eigen();
        let dim = h_r.n;
        PageWootters {
            h_r,
            energies,
            vecs,
            dim,
        }
    }

    /// The reference state `|ψ_0> = Σ_k |E_k>` (equal superposition of energy
    /// eigenstates) that the emergent dynamics evolve.
    pub fn reference_state(&self) -> Vec<Complex> {
        let d = self.dim;
        (0..d)
            .map(|r| {
                let mut acc = 0.0;
                for k in 0..d {
                    acc += self.vecs.get(r, k);
                }
                Complex::real(acc)
            })
            .collect()
    }

    /// The clock Hamiltonian `H_C = diag(-E_k)`.
    pub fn clock_hamiltonian(&self) -> RealMatrix {
        let neg: Vec<f64> = self.energies.iter().map(|e| -e).collect();
        RealMatrix::diag(&neg)
    }

    /// The globally static entangled state `|Ψ>` (length `dim²`, `C ⊗ R` order).
    pub fn global_static_state(&self) -> Vec<Complex> {
        let d = self.dim;
        let inv = 1.0 / (d as f64).sqrt();
        let mut psi = vec![Complex::ZERO; d * d];
        for k in 0..d {
            // clock index = k, system component |E_k>_R has amplitude vecs[r][k]
            for r in 0..d {
                psi[k * d + r] = Complex::real(inv * self.vecs.get(r, k));
            }
        }
        psi
    }

    /// The clock pointer (bra) for reading "time `t`": `|t>_C = Σ_k e^{iE_k t}|k>`.
    pub fn clock_pointer(&self, t: f64) -> Vec<Complex> {
        self.energies
            .iter()
            .map(|&e| Complex::phase(e * t))
            .collect()
    }

    /// Conditional state of the rest sector when the clock reads `t`, normalized.
    /// This is the *emergent* evolved state — derived purely from a static `|Ψ>`.
    pub fn conditional_state(&self, t: f64) -> Vec<Complex> {
        let psi = self.global_static_state();
        let bra = self.clock_pointer(t);
        let raw = condition_on_clock(&psi, &bra, self.dim);
        normalized(&raw)
    }

    /// The ordinary Schrödinger-evolved reference state `e^{-iH_R t}|ψ_0>`,
    /// normalized — what the conditional state must reproduce.
    pub fn schrodinger_state(&self, t: f64) -> Vec<Complex> {
        let u = schrodinger_propagator(&self.h_r, t);
        let evolved = u.matvec(&self.reference_state());
        normalized(&evolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::fidelity;

    fn sample_h() -> RealMatrix {
        // A non-trivial symmetric 3-level Hamiltonian.
        RealMatrix::from_fn(3, |r, c| {
            if r == c {
                (r as f64) - 1.0
            } else {
                0.35
            }
        })
    }

    #[test]
    fn evolution_emerges_from_static_state() {
        let pw = PageWootters::new(sample_h());
        for &t in &[0.0, 0.5, 1.3, 2.7, -1.1] {
            let cond = pw.conditional_state(t);
            let schro = pw.schrodinger_state(t);
            let f = fidelity(&cond, &schro);
            assert!(f > 1.0 - 1e-8, "t={t}: fidelity {f} too low");
        }
    }

    #[test]
    fn distinct_times_give_distinct_states() {
        let pw = PageWootters::new(sample_h());
        let a = pw.conditional_state(0.0);
        let b = pw.conditional_state(1.5);
        // Generic Hamiltonian: states at different clock readings differ.
        assert!(fidelity(&a, &b) < 0.999);
    }
}
