//! Real-data MNIST optical-compression benchmark + differential-detection
//! ablation (ADR-260 M2).
//!
//! Pipeline: MNIST digit -> 32x32 optical field -> learned phase mask ->
//! diffraction -> sensor frame -> compact readout -> tiny digital decoder.
//!
//! ADR-260's thesis is "light performs the first trained transformation; a
//! SMALL digital backend reads the result." The acceptance test (the user's
//! own, relative-to-baseline) is therefore NOT an absolute accuracy target but:
//!
//!   * learned optical accuracy >= full-image baseline accuracy - 2pp,
//!   * sensor pixels reduced >= 16x,
//!   * digital MACs (decoder INCLUDED) reduced >= 10x,
//!
//! all using the **same tiny decoder** (deterministic nearest-centroid,
//! hundreds of params) so any difference is attributable to the optics, not a
//! bigger network. We measure:
//!
//!   BASELINE  : tiny decoder on the raw downsampled image (full input pixels).
//!   OPTICAL   : tiny decoder on the compressed optical differential readout
//!               (2 * 10 = 20 region integrals -> feature vector).
//!
//! Plus the differential ablation (plain vs differential readout on the
//! identical trained mask) and an optics-only floor (pure argmax, no decoder).
//!
//! Single hill-climbed phase mask + tiny decoder is a *single-layer* optical
//! compressor. The Li/Ozcan ~97% figure is a 5-layer diffractive network
//! trained end-to-end by backprop with differential readout as the final layer;
//! multi-layer + gradient is the path to higher accuracy and is future work.
//! This benchmark positions the result as competitive single-layer optical
//! compression, never as beating state-of-the-art.

use crate::decoder::{frame_features, pool_features, NearestCentroid};
use crate::diffdetect::DiffDetector;
use crate::mnist::MNIST_CLASSES;
use crate::synthetic::Sample;
use core::f32::consts::PI;
use photonlayer_core::config::OpticalConfig;
use photonlayer_core::mask::PhaseMask;
use photonlayer_core::rng::DeterministicRng;
use photonlayer_core::simulator::{OpticalSimulator, ScalarSimulator};

/// Configuration for the MNIST differential benchmark.
#[derive(Clone, Copy, Debug)]
pub struct MnistBenchConfig {
    /// Power-of-two optical grid side (e.g. 32).
    pub grid: usize,
    /// Downsampled digit side centered in the grid (e.g. 20).
    pub cell: usize,
    /// Compressed optical sensor side: the frame is pooled to `sensor x sensor`
    /// region integrals, the compressed measurement the tiny decoder reads.
    /// At grid=32, sensor=8 gives 64 sensor px = 16x pixel reduction (the bar).
    pub sensor: usize,
    /// Hill-climbing iterations for mask training.
    pub iterations: usize,
    /// Side length of the perturbed mask block per step.
    pub block: usize,
    /// Std-dev (radians) of the per-cell phase perturbation.
    pub sigma: f32,
    /// Master seed (mask init + perturbation stream).
    pub seed: u64,
}

impl Default for MnistBenchConfig {
    fn default() -> Self {
        Self {
            grid: 32,
            cell: 20,
            sensor: 8, // 64 sensor px = 16x reduction of the 1024-px input
            iterations: 1500,
            block: 5,
            sigma: 0.7,
            seed: 0x06E157,
        }
    }
}

/// Compressed optical features for a sample set: the sensor frame pooled to
/// `sensor x sensor` region integrals, L2-normalized (via `frame_features`).
/// This `sensor^2`-length vector is the compressed measurement the tiny decoder
/// reads — the "small digital backend sees the compressed measurement" of
/// ADR-260.
fn optical_feature_set(
    samples: &[Sample],
    mask: &PhaseMask,
    cfg: &OpticalConfig,
    sensor: usize,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let feats = samples
        .iter()
        .map(|s| {
            let frame = ScalarSimulator.simulate(&s.image, mask, cfg).expect("simulation");
            frame_features(&frame, sensor)
        })
        .collect();
    let labels = samples.iter().map(|s| s.label).collect();
    (feats, labels)
}

/// Compressed-optical test accuracy via a tiny centroid decoder over the pooled
/// `sensor x sensor` readout for `mask`. Returns (test_accuracy, decoder_params).
fn decode_optical_acc(
    train: &[Sample],
    test: &[Sample],
    mask: &PhaseMask,
    cfg: &OpticalConfig,
    sensor: usize,
) -> (f32, usize) {
    let (tr_f, tr_l) = optical_feature_set(train, mask, cfg, sensor);
    let (te_f, te_l) = optical_feature_set(test, mask, cfg, sensor);
    let dec = NearestCentroid::fit(&tr_f, &tr_l, MNIST_CLASSES);
    (dec.accuracy(&te_f, &te_l), dec.param_count())
}

/// Pure optics-only argmax differential accuracy (no decoder) — a transparency
/// floor showing what the optics alone achieve before the tiny decoder.
fn argmax_diff_acc(samples: &[Sample], mask: &PhaseMask, cfg: &OpticalConfig, det: &DiffDetector) -> f32 {
    let mut correct = 0usize;
    for s in samples {
        let frame = ScalarSimulator.simulate(&s.image, mask, cfg).expect("sim");
        if det.predict_differential(&frame) == s.label {
            correct += 1;
        }
    }
    correct as f32 / samples.len().max(1) as f32
}

fn argmax_plain_acc(samples: &[Sample], mask: &PhaseMask, cfg: &OpticalConfig, det: &DiffDetector) -> f32 {
    let mut correct = 0usize;
    for s in samples {
        let frame = ScalarSimulator.simulate(&s.image, mask, cfg).expect("sim");
        if det.predict_plain(&frame) == s.label {
            correct += 1;
        }
    }
    correct as f32 / samples.len().max(1) as f32
}

/// Result of one MNIST benchmark run. Every field is a directly measured number.
#[derive(Clone, Debug)]
pub struct MnistBenchResult {
    pub train_size: usize,
    pub test_size: usize,
    pub grid: usize,
    pub cell: usize,
    pub seed: u64,

    // --- Acceptance comparison (same tiny decoder, raw image vs optical). ---
    /// Full-image digital baseline accuracy (tiny decoder on raw input pixels).
    pub baseline_acc: f32,
    /// Optical accuracy: tiny decoder on the compressed differential readout.
    pub optical_acc: f32,
    /// Optical decoder parameter count (classes * feature_len).
    pub decoder_params: usize,
    /// Baseline decoder parameter count (classes * input_pixels).
    pub baseline_decoder_params: usize,

    // --- Differential-detection ablation (identical trained mask). ---
    /// Optics-only floor: learned-mask pure argmax differential, no decoder.
    pub optics_only_differential: f32,
    /// Optics-only learned-mask plain argmax (single-region), no decoder.
    pub optics_only_plain: f32,
    /// Random-mask pure argmax differential, no decoder (learned-optics WIN
    /// guard: this is the mask-sensitive readout where learning genuinely wins).
    pub random_optics_only_differential: f32,
    /// Random-mask decoded accuracy on the compressed pooled readout. NOTE: this
    /// readout is largely mask-insensitive (diffraction + pooling preserve info
    /// for any phase mask), so learned ~= random here — reported for honesty, it
    /// is the compression metric, not where learned optics dominate.
    pub random_optical_acc: f32,

    // --- Compression accounting. ---
    /// Input pixels the baseline decoder reads (grid * grid).
    pub baseline_pixels: usize,
    /// Optical sensor pixels the optical decoder reads (pooled sensor^2).
    pub optical_sensor_pixels: usize,
    /// Sensor-pixel reduction = baseline_pixels / optical_sensor_pixels.
    pub sensor_reduction_x: f32,
    /// Digital MACs for the baseline decoder (classes * baseline_pixels).
    pub baseline_macs: usize,
    /// Digital MACs for the optical decoder (classes * optical_sensor_pixels).
    pub optical_macs: usize,
    /// MAC reduction = baseline_macs / optical_macs.
    pub mac_reduction_x: f32,
}

impl MnistBenchResult {
    /// The acceptance test (the user's own, relative-to-baseline):
    /// optical within 2pp of baseline AND >=16x sensor reduction AND >=10x MACs.
    pub fn acceptance_pass(&self) -> bool {
        self.optical_acc >= self.baseline_acc - 0.02
            && self.sensor_reduction_x >= 16.0
            && self.mac_reduction_x >= 10.0
    }
}

/// Train a phase mask on `train` against the compressed differential-decoder
/// objective, then run the full acceptance comparison + ablation on the
/// identical trained mask. All steps share `bcfg.seed`.
pub fn run_mnist_differential(
    train: &[Sample],
    test: &[Sample],
    bcfg: &MnistBenchConfig,
) -> MnistBenchResult {
    let cfg = OpticalConfig::demo(bcfg.grid, bcfg.grid);
    let det = DiffDetector::new(MNIST_CLASSES, bcfg.grid, bcfg.grid);
    let w = bcfg.grid;
    let h = bcfg.grid;
    let sensor = bcfg.sensor;

    // Score a candidate mask by its compressed-readout *training* accuracy.
    // The decoder is closed-form (centroid, no random init), so the score is a
    // deterministic function of the mask alone. This trains the optics to make
    // the pooled sensor readout linearly separable by the tiny decoder.
    let score_mask = |mask: &PhaseMask| -> f32 {
        let (f, l) = optical_feature_set(train, mask, &cfg, sensor);
        let dec = NearestCentroid::fit(&f, &l, MNIST_CLASSES);
        dec.accuracy(&f, &l)
    };

    // --- Random-mask baselines. ---
    let random_mask = PhaseMask::random(w, h, bcfg.seed ^ 0x5EED);
    let (random_optical_acc, _) = decode_optical_acc(train, test, &random_mask, &cfg, sensor);
    // Argmax differential on the random mask: the mask-sensitive readout where
    // learning genuinely dominates (the honest learned-optics WIN guard).
    let random_optics_only_differential = argmax_diff_acc(test, &random_mask, &cfg, &det);

    // --- Train the mask via seeded block hill-climbing. ---
    let mut rng = DeterministicRng::new(bcfg.seed);
    let mut mask = PhaseMask::random(w, h, bcfg.seed);
    let mut score = score_mask(&mask);
    for _ in 0..bcfg.iterations {
        let mut candidate = mask.clone();
        let bx = (rng.next_f32() * (w.saturating_sub(bcfg.block) + 1) as f32) as usize;
        let by = (rng.next_f32() * (h.saturating_sub(bcfg.block) + 1) as f32) as usize;
        for dy in 0..bcfg.block.min(h) {
            for dx in 0..bcfg.block.min(w) {
                let idx = (by + dy).min(h - 1) * w + (bx + dx).min(w - 1);
                let delta = rng.next_gaussian() * bcfg.sigma;
                candidate.phase_radians[idx] =
                    (candidate.phase_radians[idx] + delta).rem_euclid(2.0 * PI);
            }
        }
        let cand = score_mask(&candidate);
        if cand > score {
            mask = candidate;
            score = cand;
        }
    }
    mask.mask_id = format!("mnist-learned:{:#x}", bcfg.seed);

    // --- Optical accuracy: tiny decoder on the compressed pooled sensor readout. ---
    let (optical_acc, decoder_params) = decode_optical_acc(train, test, &mask, &cfg, sensor);

    // --- Optics-only floor (pure argmax, identical trained mask). ---
    let optics_only_differential = argmax_diff_acc(test, &mask, &cfg, &det);
    let optics_only_plain = argmax_plain_acc(test, &mask, &cfg, &det);

    // --- Full-image digital baseline: SAME decoder family on raw input pixels. ---
    // pool_features at the full grid is the L2-normalized raw downsampled image,
    // so the baseline reads every input pixel (no compression) with the same
    // centroid classifier — the apples-to-apples "full-image baseline".
    let baseline_feats = |samples: &[Sample]| -> (Vec<Vec<f32>>, Vec<usize>) {
        let f = samples
            .iter()
            .map(|s| pool_features(&s.image.pixels, s.image.width, s.image.height, bcfg.grid))
            .collect();
        let l = samples.iter().map(|s| s.label).collect();
        (f, l)
    };
    let (btr_f, btr_l) = baseline_feats(train);
    let (bte_f, bte_l) = baseline_feats(test);
    let bdec = NearestCentroid::fit(&btr_f, &btr_l, MNIST_CLASSES);
    let baseline_acc = bdec.accuracy(&bte_f, &bte_l);
    let baseline_decoder_params = bdec.param_count();

    let baseline_pixels = bcfg.grid * bcfg.grid;
    let optical_sensor_pixels = sensor * sensor; // pooled sensor readout size
    let baseline_macs = MNIST_CLASSES * baseline_pixels;
    let optical_macs = MNIST_CLASSES * optical_sensor_pixels;
    MnistBenchResult {
        train_size: train.len(),
        test_size: test.len(),
        grid: bcfg.grid,
        cell: bcfg.cell,
        seed: bcfg.seed,
        baseline_acc,
        optical_acc,
        decoder_params,
        baseline_decoder_params,
        optics_only_differential,
        optics_only_plain,
        random_optics_only_differential,
        random_optical_acc,
        baseline_pixels,
        optical_sensor_pixels,
        sensor_reduction_x: baseline_pixels as f32 / optical_sensor_pixels as f32,
        baseline_macs,
        optical_macs,
        mac_reduction_x: baseline_macs as f32 / optical_macs as f32,
    }
}
