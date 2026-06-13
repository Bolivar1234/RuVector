//! Entropic time — the cold-atom toy-universe clock.
//!
//! When a closed system is split into an observed sector and a hidden sector
//! exchanging entropy, the *change in entropy* of the observed sector defines
//! an internal time
//!
//! ```text
//!   τ_S = (S(λ) - S_0) / k
//! ```
//!
//! where `λ` is a lab control parameter (e.g. the barrier between sectors).
//! Derivatives reparametrize as
//!
//! ```text
//!   dX/dτ_S = (k / (dS/dλ)) · dX/dλ.
//! ```
//!
//! The "speed of internal time" `dτ_S/dλ = (dS/dλ)/k` tracks entropy
//! production: when entropy exchange stalls, internal time freezes; when it
//! accelerates, internal time speeds up.

use crate::entropy::entropy_from_spectrum;
use crate::real_matrix::RealMatrix;

/// Maps observed-sector entropy onto an internal time coordinate.
#[derive(Clone, Copy, Debug)]
pub struct EntropicClock {
    /// Reference entropy `S_0` (internal time origin).
    pub s0: f64,
    /// Clock scale `k` (nats of entropy per unit internal time).
    pub k: f64,
}

impl EntropicClock {
    pub fn new(s0: f64, k: f64) -> Self {
        EntropicClock { s0, k }
    }

    /// Internal time `τ_S` for an observed entropy `s`.
    pub fn tau(&self, s: f64) -> f64 {
        (s - self.s0) / self.k
    }

    /// Speed of internal time `dτ_S/dλ = (dS/dλ)/k`.
    pub fn rate(&self, ds_dlambda: f64) -> f64 {
        ds_dlambda / self.k
    }

    /// Convert a `λ`-derivative into a `τ_S`-derivative,
    /// `dX/dτ_S = (k/(dS/dλ)) dX/dλ`.
    ///
    /// Returns `None` when entropy production vanishes (internal time is frozen,
    /// so the rate of change per unit internal time is undefined / unbounded).
    pub fn convert_derivative(&self, dx_dlambda: f64, ds_dlambda: f64) -> Option<f64> {
        if ds_dlambda.abs() < 1e-12 {
            None
        } else {
            Some((self.k / ds_dlambda) * dx_dlambda)
        }
    }

    /// Reparametrize a `λ`-sampled observable trajectory into internal time.
    /// Each input sample is `(λ, S(λ), X(λ))`; output is `(τ_S, X)`.
    pub fn reparametrize(&self, samples: &[(f64, f64, f64)]) -> Vec<(f64, f64)> {
        samples.iter().map(|&(_l, s, x)| (self.tau(s), x)).collect()
    }
}

/// Gibbs (thermal) density matrix `ρ = e^{-βH}/Z` for a real symmetric
/// Hamiltonian — the standard entropy source for an observed sector at inverse
/// temperature `β`.
pub fn gibbs_density(h: &RealMatrix, beta: f64) -> RealMatrix {
    let (energies, vecs) = h.symmetric_eigen();
    // Shift by the ground-state energy for numerical stability of exp.
    let e_min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
    let weights: Vec<f64> = energies.iter().map(|&e| (-beta * (e - e_min)).exp()).collect();
    let z: f64 = weights.iter().sum();
    let probs: Vec<f64> = weights.iter().map(|w| w / z).collect();
    RealMatrix::from_spectrum(&probs, &vecs)
}

/// Von Neumann entropy of the Gibbs state at inverse temperature `β`.
pub fn gibbs_entropy(h: &RealMatrix, beta: f64) -> f64 {
    let (energies, _v) = h.symmetric_eigen();
    let e_min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
    let weights: Vec<f64> = energies.iter().map(|&e| (-beta * (e - e_min)).exp()).collect();
    let z: f64 = weights.iter().sum();
    let probs: Vec<f64> = weights.iter().map(|w| w / z).collect();
    entropy_from_spectrum(&probs)
}

/// Sweep a control parameter `λ ∈ [lo, hi]` (interpreted as inverse temperature),
/// returning `(λ, S(λ), τ_S(λ))` triples. Demonstrates how the internal clock
/// runs fast where entropy changes quickly and stalls where it saturates.
pub fn entropic_time_sweep(
    h: &RealMatrix,
    clock: &EntropicClock,
    lo: f64,
    hi: f64,
    steps: usize,
) -> Vec<(f64, f64, f64)> {
    let mut out = Vec::with_capacity(steps);
    for i in 0..steps {
        let frac = if steps <= 1 { 0.0 } else { i as f64 / (steps - 1) as f64 };
        let lam = lo + frac * (hi - lo);
        let s = gibbs_entropy(h, lam);
        out.push((lam, s, clock.tau(s)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_h() -> RealMatrix {
        RealMatrix::diag(&[0.0, 1.0, 2.0, 3.0])
    }

    #[test]
    fn gibbs_trace_one() {
        let rho = gibbs_density(&sample_h(), 0.8);
        let tr: f64 = (0..rho.n).map(|i| rho.get(i, i)).sum();
        assert!((tr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn entropy_monotone_in_temperature() {
        // Higher temperature (lower β) → higher entropy.
        let s_hot = gibbs_entropy(&sample_h(), 0.1);
        let s_cold = gibbs_entropy(&sample_h(), 5.0);
        assert!(s_hot > s_cold);
    }

    #[test]
    fn frozen_entropy_freezes_time() {
        let clock = EntropicClock::new(0.0, 1.0);
        assert!(clock.convert_derivative(1.0, 0.0).is_none());
        assert!(clock.convert_derivative(1.0, 2.0).unwrap().abs() > 0.0);
    }

    #[test]
    fn tau_tracks_entropy() {
        let clock = EntropicClock::new(0.5, 2.0);
        assert!((clock.tau(2.5) - 1.0).abs() < 1e-12);
    }
}
