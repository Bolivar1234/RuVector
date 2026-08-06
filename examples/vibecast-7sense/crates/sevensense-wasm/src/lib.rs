//! Browser bindings for the 7sense analysis pipeline.
//!
//! This exposes a raw C ABI rather than using `wasm-bindgen`. The compiled
//! module is inlined into a self-contained page as a data URI, so binary size
//! is a correctness constraint -- `wasm-bindgen` would add a JS shim and
//! several hundred kilobytes of glue for an interface that is, in the end, a
//! handful of float buffers.
//!
//! # Calling convention
//!
//! The host allocates an input buffer with [`alloc_f32`], writes samples into
//! WebAssembly memory, and calls an analysis function. Each returns a count (or
//! a negative error code) and leaves its packed result in a module-owned output
//! buffer, readable via [`out_ptr`] and [`out_len`].
//!
//! Results are packed as flat `f32` arrays with a fixed stride per record, so
//! the host reads them with a single `Float32Array` view and no per-field
//! decoding. Field order is declared by the `*_STRIDE` constants below and must
//! stay in step with the JavaScript that reads it.

#![allow(clippy::missing_safety_doc)]

use sevensense_audio::features::{FeatureConfig, FeatureExtractor};
use sevensense_audio::streaming::{
    MemorySource, StreamConfig, StreamPipeline, StreamSegmenter, StreamSegmenterConfig,
};
use sevensense_vector::projection::PcaProjection;

/// Values per frame in the [`analyze_features`] result.
pub const FRAME_STRIDE: usize = 10;

/// Values per segment in the [`detect_segments`] result.
pub const SEGMENT_STRIDE: usize = 6;

/// Input was empty, too short, or otherwise unusable.
const ERR_INVALID_INPUT: i32 = -1;
/// Analysis failed for a reason reported by the underlying crate.
const ERR_ANALYSIS_FAILED: i32 = -2;

/// Module-owned buffer holding the most recent result.
///
/// Single-threaded by construction: WebAssembly without the threads proposal
/// runs one instance per worker, and each call fully overwrites this before
/// returning.
static mut OUTPUT: Vec<f32> = Vec::new();

/// Replaces the output buffer and returns how many records it holds.
fn publish(values: Vec<f32>, stride: usize) -> i32 {
    let records = if stride == 0 { 0 } else { values.len() / stride };
    // Safety: single-threaded module; no other reference is live here.
    unsafe {
        OUTPUT = values;
    }
    i32::try_from(records).unwrap_or(i32::MAX)
}

/// Allocates `len` floats in module memory and returns a pointer to them.
///
/// The host owns the result and must release it with [`dealloc_f32`].
#[no_mangle]
pub extern "C" fn alloc_f32(len: usize) -> *mut f32 {
    let mut buffer = Vec::<f32>::with_capacity(len);
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

/// Releases a buffer previously returned by [`alloc_f32`].
///
/// # Safety
/// `ptr` must come from [`alloc_f32`] with the same `len`, and must not have
/// been freed already.
#[no_mangle]
pub unsafe extern "C" fn dealloc_f32(ptr: *mut f32, len: usize) {
    if !ptr.is_null() {
        drop(Vec::from_raw_parts(ptr, 0, len));
    }
}

/// Pointer to the packed result of the most recent analysis call.
#[no_mangle]
pub extern "C" fn out_ptr() -> *const f32 {
    // Safety: returns a shared pointer; the host reads before the next call.
    unsafe { core::ptr::addr_of!(OUTPUT).cast::<Vec<f32>>().as_ref() }
        .map_or(core::ptr::null(), |v| v.as_ptr())
}

/// Number of floats in the most recent result.
#[no_mangle]
pub extern "C" fn out_len() -> usize {
    unsafe { core::ptr::addr_of!(OUTPUT).cast::<Vec<f32>>().as_ref() }.map_or(0, Vec::len)
}

/// Computes per-frame acoustic descriptors over `samples`.
///
/// Packs [`FRAME_STRIDE`] values per frame: time in milliseconds, spectral
/// centroid, spread, rolloff, tonality, crest, entropy, dominant frequency,
/// energy in dBFS, and a voiced flag.
///
/// Returns the frame count, or a negative error code.
///
/// # Safety
/// `ptr` must point to `len` readable floats.
#[no_mangle]
pub unsafe extern "C" fn analyze_features(ptr: *const f32, len: usize, sample_rate: u32) -> i32 {
    if ptr.is_null() || len == 0 || sample_rate == 0 {
        return ERR_INVALID_INPUT;
    }
    let samples = std::slice::from_raw_parts(ptr, len);

    let config = FeatureConfig {
        sample_rate,
        ..Default::default()
    };
    let Ok(extractor) = FeatureExtractor::new(config) else {
        return ERR_INVALID_INPUT;
    };
    let Ok(features) = extractor.extract(samples) else {
        return ERR_ANALYSIS_FAILED;
    };

    let mut packed = Vec::with_capacity(features.frames.len() * FRAME_STRIDE);
    for frame in &features.frames {
        packed.push(frame.time_ms as f32);
        packed.push(frame.centroid_hz);
        packed.push(frame.spread_hz);
        packed.push(frame.rolloff_hz);
        packed.push(frame.tonality);
        packed.push(frame.crest);
        packed.push(frame.entropy);
        packed.push(frame.dominant_hz);
        // A silent frame yields -inf dB, which serializes to NaN in JS and
        // breaks any chart drawn from it. Clamp to the display floor instead.
        packed.push(if frame.energy_db.is_finite() {
            frame.energy_db
        } else {
            -120.0
        });
        packed.push(if frame.voiced { 1.0 } else { 0.0 });
    }

    publish(packed, FRAME_STRIDE)
}

/// Returns the summary statistics for the last [`analyze_features`] call's
/// input, recomputed over `samples`.
///
/// Packs 12 values: mean centroid, centroid deviation, mean spread, mean
/// tonality, mean crest, mean entropy, mean dominant frequency, mean energy,
/// voiced fraction, AM rate, AM depth, and FM extent. Modulation fields are
/// zero when the input is too short to measure them.
///
/// # Safety
/// `ptr` must point to `len` readable floats.
#[no_mangle]
pub unsafe extern "C" fn analyze_summary(ptr: *const f32, len: usize, sample_rate: u32) -> i32 {
    if ptr.is_null() || len == 0 || sample_rate == 0 {
        return ERR_INVALID_INPUT;
    }
    let samples = std::slice::from_raw_parts(ptr, len);

    let config = FeatureConfig {
        sample_rate,
        ..Default::default()
    };
    let Ok(extractor) = FeatureExtractor::new(config) else {
        return ERR_INVALID_INPUT;
    };
    let Ok(features) = extractor.extract(samples) else {
        return ERR_ANALYSIS_FAILED;
    };

    let s = &features.summary;
    let modulation = s.modulation;
    let packed = vec![
        s.centroid_hz.mean,
        s.centroid_hz.std,
        s.spread_hz.mean,
        s.tonality.mean,
        s.crest.mean,
        s.entropy.mean,
        s.dominant_hz.mean,
        if s.energy_db.mean.is_finite() {
            s.energy_db.mean
        } else {
            -120.0
        },
        s.voiced_fraction,
        modulation.map_or(0.0, |m| m.am_rate_hz),
        modulation.map_or(0.0, |m| m.am_depth),
        modulation.map_or(0.0, |m| m.fm_extent_hz),
    ];

    publish(packed, 12)
}

/// Segments `samples` with the streaming pipeline.
///
/// Packs [`SEGMENT_STRIDE`] values per segment: start and end time in
/// milliseconds, peak amplitude, RMS energy, SNR in dB, and a truncated flag.
///
/// Returns the segment count, or a negative error code.
///
/// # Safety
/// `ptr` must point to `len` readable floats.
#[no_mangle]
pub unsafe extern "C" fn detect_segments(ptr: *const f32, len: usize, sample_rate: u32) -> i32 {
    if ptr.is_null() || len == 0 || sample_rate == 0 {
        return ERR_INVALID_INPUT;
    }
    let samples = std::slice::from_raw_parts(ptr, len).to_vec();

    let config = StreamSegmenterConfig {
        sample_rate,
        ..Default::default()
    };
    let Ok(mut segmenter) = StreamSegmenter::new(config) else {
        return ERR_INVALID_INPUT;
    };

    let mut events = segmenter.push(&samples);
    events.extend(segmenter.flush());

    let to_ms = |s: u64| s as f32 * 1000.0 / sample_rate as f32;
    let mut packed = Vec::new();
    for event in events {
        if let sevensense_audio::streaming::SegmentEvent::Closed(segment) = event {
            packed.push(to_ms(segment.start_sample));
            packed.push(to_ms(segment.end_sample));
            packed.push(segment.peak_amplitude);
            packed.push(segment.rms_energy);
            packed.push(segment.snr_db());
            packed.push(if segment.truncated { 1.0 } else { 0.0 });
        }
    }

    publish(packed, SEGMENT_STRIDE)
}

/// Runs the full streaming pipeline and reports how many analysis windows a
/// recording would produce, alongside the loss counters.
///
/// Packs 5 values: window count, segment count, discarded count, samples read,
/// and samples dropped to ring-buffer overwrite.
///
/// # Safety
/// `ptr` must point to `len` readable floats.
#[no_mangle]
pub unsafe extern "C" fn pipeline_stats(ptr: *const f32, len: usize, sample_rate: u32) -> i32 {
    if ptr.is_null() || len == 0 || sample_rate == 0 {
        return ERR_INVALID_INPUT;
    }
    let samples = std::slice::from_raw_parts(ptr, len).to_vec();

    let source = MemorySource::mono(samples, sample_rate);
    let Ok(mut pipeline) = StreamPipeline::new(source, StreamConfig::default()) else {
        return ERR_INVALID_INPUT;
    };
    let Ok(windows) = pipeline.run_to_completion() else {
        return ERR_ANALYSIS_FAILED;
    };

    let stats = pipeline.stats();
    publish(
        vec![
            windows.len() as f32,
            stats.segments as f32,
            stats.discarded as f32,
            stats.samples_read as f32,
            stats.samples_dropped as f32,
        ],
        5,
    )
}

/// Fits a PCA projection over `n_points` vectors of `dim` values each and
/// projects them to `out_dims` dimensions.
///
/// Input is row-major: point `i` occupies `[i * dim, (i + 1) * dim)`.
/// Packs `out_dims` coordinates per point, followed by one final record holding
/// the explained-variance ratio per component.
///
/// Returns the number of records (points plus the trailing variance record), or
/// a negative error code.
///
/// # Safety
/// `ptr` must point to `n_points * dim` readable floats.
#[no_mangle]
pub unsafe extern "C" fn project_pca(
    ptr: *const f32,
    n_points: usize,
    dim: usize,
    out_dims: usize,
    seed: u32,
) -> i32 {
    if ptr.is_null() || n_points == 0 || dim == 0 || out_dims == 0 || out_dims > dim {
        return ERR_INVALID_INPUT;
    }
    let flat = std::slice::from_raw_parts(ptr, n_points * dim);
    let data: Vec<Vec<f32>> = flat.chunks_exact(dim).map(<[f32]>::to_vec).collect();

    let Ok(pca) = PcaProjection::fit(&data, out_dims, u64::from(seed)) else {
        return ERR_INVALID_INPUT;
    };
    let Ok(mut projected) = pca.project_batch(&data) else {
        return ERR_ANALYSIS_FAILED;
    };

    // Bounds are discarded: the host frames the view from the normalized
    // coordinates alone, so the original scale is not needed here.
    let _bounds = sevensense_vector::projection::normalize_coordinates(&mut projected);

    let mut packed: Vec<f32> = projected.into_iter().flatten().collect();
    // Trailing record: explained variance per component, padded to the stride
    // so the host can read everything with one uniformly-strided view.
    let mut variance = pca.explained_variance().to_vec();
    variance.resize(out_dims, 0.0);
    packed.extend(variance);

    publish(packed, out_dims)
}

/// Version of the analysis pipeline this module was built from.
///
/// Encoded as `major * 10000 + minor * 100 + patch` so it crosses the ABI as a
/// single integer.
#[no_mangle]
pub extern "C" fn pipeline_version() -> i32 {
    let parse = |s: Option<&str>| -> i32 { s.and_then(|v| v.parse().ok()).unwrap_or(0) };
    let mut parts = env!("CARGO_PKG_VERSION").split('.');
    parse(parts.next()) * 10_000 + parse(parts.next()) * 100 + parse(parts.next())
}
