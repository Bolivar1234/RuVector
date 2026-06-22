//! Time-of-flight reconstruction via SART (Simultaneous Algebraic
//! Reconstruction Technique).
//!
//! We solve the linear tomography system `A s = t`, where `s` is the per-cell
//! slowness (1/c), `A_ij` is the length ray `i` spends in cell `j`, and `t` is
//! the measured interior travel time. The same solver reconstructs attenuation
//! by swapping the right-hand side. A single SART sweep is equivalent to the
//! classic delay-backprojection baseline (ADR-0004); additional sweeps move the
//! estimate towards the least-squares solution.

use crate::acquisition::Acquisition;
use crate::grid::Grid;
use crate::types::{clamp, SPEED_MAX, SPEED_MIN, WATER_SPEED};

/// Reconstruction tuning.
#[derive(Debug, Clone, Copy)]
pub struct ReconConfig {
    /// Number of SART sweeps (1 == delay-backprojection baseline).
    pub iters: usize,
    /// Relaxation factor in `(0, 2)`; smaller is more stable.
    pub relaxation: f32,
}

impl Default for ReconConfig {
    fn default() -> Self {
        ReconConfig {
            iters: 6,
            relaxation: 0.9,
        }
    }
}

/// Generic SART solver over the supplied measurements.
///
/// `rhs(i)` returns the measured line integral for measurement `i` and
/// `init` is the constant initial field value. Returns the per-cell field.
fn sart<F>(acq: &Acquisition, n_cells: usize, cfg: ReconConfig, init: f32, rhs: F) -> Vec<f32>
where
    F: Fn(usize) -> f32,
{
    let mut field = vec![init; n_cells];
    let mut num = vec![0.0f32; n_cells];
    let mut den = vec![0.0f32; n_cells];

    for _ in 0..cfg.iters.max(1) {
        num.iter_mut().for_each(|v| *v = 0.0);
        den.iter_mut().for_each(|v| *v = 0.0);

        for (i, m) in acq.measurements.iter().enumerate() {
            if !m.valid {
                continue;
            }
            let rowsum = m.ray.interior_length();
            if rowsum <= 0.0 {
                continue;
            }
            // Predicted line integral with the current field estimate.
            let mut predicted = 0.0f32;
            for &(c, l) in &m.ray.cells {
                predicted += field[c] * l;
            }
            let residual = (rhs(i) - predicted) / rowsum;
            for &(c, l) in &m.ray.cells {
                num[c] += l * residual;
                den[c] += l;
            }
        }

        for j in 0..n_cells {
            if den[j] > 0.0 {
                field[j] += cfg.relaxation * num[j] / den[j];
            }
        }
    }
    field
}

/// Reconstruct the speed-of-sound map (m/s) on a grid shaped like `like`.
pub fn reconstruct_speed(acq: &Acquisition, like: &Grid, cfg: ReconConfig) -> Grid {
    let n_cells = like.len();
    let init = 1.0 / WATER_SPEED;
    // Interior travel time = measured travel time minus the exterior water leg.
    let slowness = sart(acq, n_cells, cfg, init, |i| {
        let m = &acq.measurements[i];
        let exterior = (m.path_length - m.ray.interior_length()).max(0.0);
        m.travel_time - exterior / WATER_SPEED
    });

    let mut out = like.clone();
    for (o, &s) in out.data.iter_mut().zip(&slowness) {
        let c = if s > 0.0 { 1.0 / s } else { WATER_SPEED };
        *o = clamp(c, SPEED_MIN, SPEED_MAX);
    }
    out
}

/// Reconstruct the attenuation map (Np/m) on a grid shaped like `like`.
pub fn reconstruct_attenuation(acq: &Acquisition, like: &Grid, cfg: ReconConfig) -> Grid {
    let n_cells = like.len();
    let field = sart(acq, n_cells, cfg, 0.0, |i| acq.measurements[i].attenuation);
    let mut out = like.clone();
    for (o, &a) in out.data.iter_mut().zip(&field) {
        *o = a.max(0.0);
    }
    out
}
