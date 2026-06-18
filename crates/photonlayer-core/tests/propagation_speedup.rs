//! M1 proof: the cached + in-place `Propagator` is faster than the naive free
//! `propagate()` (which recomputes the transfer function H and clones the field
//! every call), and produces **bit-identical** output. No speedup claim without
//! this measured number.
//!
//! Run: `cargo test -p photonlayer-core --release --test propagation_speedup -- --ignored --nocapture`

use std::time::Instant;

use photonlayer_core::config::OpticalConfig;
use photonlayer_core::field::{InputImage, OpticalField};
use photonlayer_core::propagate::{propagate, Propagator};

const N: usize = 64; // grid (learn-loop regime where H-recompute is a large fraction)
const ITERS: usize = 3000;

fn test_field(n: usize) -> OpticalField {
    // Deterministic non-trivial pattern.
    let px: Vec<f32> = (0..n * n)
        .map(|i| {
            let (x, y) = ((i % n) as f32, (i / n) as f32);
            0.5 + 0.5 * ((x * 0.3).sin() * (y * 0.2).cos())
        })
        .collect();
    let img = InputImage::from_norm_f32(n, n, px).unwrap();
    OpticalField::from_image(&img, n, n).unwrap()
}

/// Always-on correctness gate: the cached + in-place path is bit-for-bit
/// identical to the free `propagate()`. Cheap; runs in the default suite.
#[test]
fn cached_propagator_is_bit_identical() {
    let field = test_field(N);
    let config = OpticalConfig::demo(N, N);
    let reference = propagate(&field, &config).unwrap();
    let prop = Propagator::new(N, N, &config).unwrap();
    let via_struct = prop.propagate(&field).unwrap();
    let mut buf = field.data.clone();
    prop.propagate_into(&mut buf).unwrap();
    assert_eq!(via_struct.data, reference.data, "Propagator::propagate must match free propagate");
    assert_eq!(buf, reference.data, "propagate_into must be bit-identical to free propagate");
}

/// Timing proof (M1). Release-only — wall-clock is meaningless in debug. Run:
/// `cargo test -p photonlayer-core --release --test propagation_speedup -- --ignored --nocapture`
#[test]
#[ignore = "timing benchmark — run with --release --ignored"]
fn cached_propagator_is_faster() {
    let field = test_field(N);
    let config = OpticalConfig::demo(N, N);

    // Warm up.
    for _ in 0..64 {
        let _ = propagate(&field, &config).unwrap();
    }

    // Naive: free propagate (recompute H + clone) every call.
    let t = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..ITERS {
        let out = propagate(&field, &config).unwrap();
        sink += out.data[0].re;
    }
    let naive = t.elapsed().as_secs_f64();

    // Optimized: build operator once; in-place propagate into a reused buffer.
    let prop = Propagator::new(N, N, &config).unwrap();
    let mut scratch = vec![photonlayer_core::complex::Complex::ZERO; N * N];
    let t = Instant::now();
    for _ in 0..ITERS {
        scratch.copy_from_slice(&field.data);
        prop.propagate_into(&mut scratch).unwrap();
        sink += scratch[0].re;
    }
    let opt = t.elapsed().as_secs_f64();
    std::hint::black_box(sink);

    let speedup = naive / opt;
    eprintln!(
        "propagation {N}x{N} x{ITERS}: naive={:.1}ms  cached+inplace={:.1}ms  speedup={speedup:.2}x",
        naive * 1e3,
        opt * 1e3
    );
    assert!(
        speedup >= 1.5,
        "cached+in-place propagator must be >= 1.5x the naive path; got {speedup:.2}x"
    );
}
