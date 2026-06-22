//! Reconstruction and segmentation quality metrics.

use crate::grid::Grid;
use crate::types::Tissue;

/// Dice similarity coefficient for one class between predicted and truth labels.
///
/// Returns 1.0 for a class that is absent from both maps (vacuously perfect),
/// matching the common convention for empty-class Dice.
pub fn dice(pred_labels: &Grid, true_labels: &Grid, class: Tissue) -> f32 {
    let c = class as u8 as f32;
    let mut inter = 0u64;
    let mut a = 0u64;
    let mut b = 0u64;
    for (p, t) in pred_labels.data.iter().zip(&true_labels.data) {
        let pp = *p == c;
        let tt = *t == c;
        if pp {
            a += 1;
        }
        if tt {
            b += 1;
        }
        if pp && tt {
            inter += 1;
        }
    }
    if a + b == 0 {
        return 1.0;
    }
    (2 * inter) as f32 / (a + b) as f32
}

/// Per-class Dice scores in `Tissue::ALL` order.
pub fn dice_all(pred_labels: &Grid, true_labels: &Grid) -> [f32; Tissue::COUNT] {
    let mut out = [0.0f32; Tissue::COUNT];
    for (i, &t) in Tissue::ALL.iter().enumerate() {
        out[i] = dice(pred_labels, true_labels, t);
    }
    out
}

/// Mean Dice across all classes.
pub fn mean_dice(pred_labels: &Grid, true_labels: &Grid) -> f32 {
    let d = dice_all(pred_labels, true_labels);
    d.iter().sum::<f32>() / d.len() as f32
}

/// Mean absolute speed-of-sound error (m/s) between two grids.
pub fn mae_speed(pred: &Grid, truth: &Grid) -> f32 {
    pred.mean_abs_diff(truth).unwrap_or(f32::NAN)
}

/// A compact bundle of quality metrics for one reconstruction.
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Mean absolute speed error (m/s).
    pub mae_speed: f32,
    /// Per-class Dice in `Tissue::ALL` order.
    pub dice: [f32; Tissue::COUNT],
    /// Mean Dice across classes.
    pub mean_dice: f32,
    /// Number of valid measurements used.
    pub measurements: usize,
}
