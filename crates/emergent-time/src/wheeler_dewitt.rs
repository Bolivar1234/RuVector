//! Wheeler–DeWitt timeless constraint.
//!
//! The quantum state of a closed universe obeys `Ĥ|Ψ> = 0` — there is no
//! external time parameter. "Time" must be found *inside* the state. This
//! module builds bipartite constraint operators `Ĵ = H_C ⊗ I + I ⊗ H_R` and
//! locates their physical (kernel) states.

use crate::complex::Complex;
use crate::complex_matrix::CMatrix;
use crate::real_matrix::RealMatrix;
use crate::state::idx;

/// Build the bipartite constraint `Ĵ = H_C ⊗ I_{dr} + I_{dc} ⊗ H_R`.
///
/// Physical states `|Ψ>` of the joint clock+rest system satisfy `Ĵ|Ψ> = 0`:
/// the total "energy" (clock + rest) is constrained to vanish, which is what
/// removes the external time parameter.
pub fn bipartite_constraint(h_c: &RealMatrix, h_r: &RealMatrix) -> RealMatrix {
    let dc = h_c.n;
    let dr = h_r.n;
    let n = dc * dr;
    let mut j = RealMatrix::zeros(n);
    for c in 0..dc {
        for cp in 0..dc {
            let hc = h_c.get(c, cp);
            for r in 0..dr {
                for rp in 0..dr {
                    let mut v = 0.0;
                    if r == rp {
                        v += hc; // H_C ⊗ I
                    }
                    if c == cp {
                        v += h_r.get(r, rp); // I ⊗ H_R
                    }
                    if v != 0.0 {
                        j.set(idx(c, r, dr), idx(cp, rp, dr), v);
                    }
                }
            }
        }
    }
    j
}

/// A physical (timeless) state: the eigenvector of the constraint with the
/// eigenvalue closest to zero.
pub struct PhysicalState {
    /// The constraint eigenvalue actually achieved (≈ 0 for a true kernel).
    pub eigenvalue: f64,
    /// The normalized physical state vector.
    pub state: Vec<f64>,
}

/// Find the physical state `|Ψ>` solving `Ĵ|Ψ> ≈ 0` — the kernel direction of
/// the constraint operator.
pub fn solve_constraint(j: &RealMatrix) -> PhysicalState {
    let (vals, vecs) = j.symmetric_eigen();
    let mut best = 0usize;
    for k in 1..vals.len() {
        if vals[k].abs() < vals[best].abs() {
            best = k;
        }
    }
    PhysicalState {
        eigenvalue: vals[best],
        state: vecs.column(best),
    }
}

/// Residual `‖Ĵ|Ψ>‖` for a (possibly complex) state vector — the degree to
/// which the timeless equation `Ĥ|Ψ> = 0` is satisfied.
pub fn constraint_residual(j: &RealMatrix, psi: &[Complex]) -> f64 {
    let jc = CMatrix::from_real(j);
    let out = jc.matvec(psi);
    out.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page_wootters::PageWootters;

    fn sample_h() -> RealMatrix {
        RealMatrix::from_fn(3, |r, c| if r == c { (r as f64) - 1.0 } else { 0.3 })
    }

    #[test]
    fn page_wootters_state_is_in_kernel() {
        let pw = PageWootters::new(sample_h());
        let j = bipartite_constraint(&pw.clock_hamiltonian(), &pw.h_r);
        let psi = pw.global_static_state();
        let residual = constraint_residual(&j, &psi);
        // The static entangled state solves the Wheeler–DeWitt equation exactly.
        assert!(residual < 1e-8, "residual {residual} should be ~0");
    }

    #[test]
    fn constraint_has_zero_eigenvalue() {
        let pw = PageWootters::new(sample_h());
        let j = bipartite_constraint(&pw.clock_hamiltonian(), &pw.h_r);
        let phys = solve_constraint(&j);
        assert!(phys.eigenvalue.abs() < 1e-8);
        // Kernel state is a unit vector.
        let norm: f64 = phys.state.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-8);
    }
}
