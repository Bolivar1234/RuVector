#![allow(
    clippy::all,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_must_use,
    unused_parens
)]
//! HRM-Text recurrence benchmarks (ADR-199 Phase D / R1 infrastructure).
//!
//! - `hrm_forward_cycles`: forward latency across cycle configs
//!   (1,1)/(2,2)/(3,3)/(4,4) at fixed model size — the latency axis of the
//!   R1 "test-time compute scaling on the cycle knob" experiment.
//! - `hrm_decode`: cached `decode_step` vs full-sequence recompute at
//!   seq 128 — demonstrates the per-cycle KV cache win.
//!
//! KV memory accounting for `hrm_text_l()` at 4K context is asserted (not
//! timed) in `tests/hrm_text_test.rs::kv_memory_hrm_text_l_at_4k_is_3x_standard`;
//! the number is also printed once here for convenience.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ruvllm::backends::hrm_text::{
    AttentionMaskMode, HrmCycles, HrmKvCaches, HrmTextConfig, HrmTextModel, NormPosition,
    SelfSpeculativeDecoder,
};

/// Small-but-not-trivial config: hidden 64, 4 total layers (2 per core),
/// 4 heads (head_dim 16). Cycle counts are the swept variable.
fn bench_config(h_cycles: usize, l_cycles: usize) -> HrmTextConfig {
    HrmTextConfig {
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        max_position_embeddings: 256,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        norm_position: NormPosition::PreNorm,
        h_cycles,
        l_cycles,
        half_layers: 2,
        prefix_bidirectional: true,
        bos_token_id: 1,
        eos_token_id: 2,
    }
}

fn token_ids(len: usize) -> Vec<u32> {
    // Deterministic pseudo-random token stream within the bench vocab.
    (0..len)
        .map(|i| (((i as u64).wrapping_mul(2654435761) >> 8) % 256) as u32)
        .collect()
}

/// R1 infrastructure: forward latency vs recurrence depth at zero extra
/// parameters. Expect roughly linear scaling in h*(l+1) core invocations.
fn bench_forward_cycles(c: &mut Criterion) {
    let mut group = c.benchmark_group("hrm_forward_cycles");
    group.sample_size(10);

    let seq = 64;
    let ids = token_ids(seq);

    for (h, l) in [(1usize, 1usize), (2, 2), (3, 3), (4, 4)] {
        let config = bench_config(h, l);
        let model = HrmTextModel::new_random(config.clone(), 42).expect("bench model");
        let id = BenchmarkId::new(format!("h{}xl{}", h, l), h * l);
        group.bench_function(id, |b| {
            b.iter_batched(
                || HrmKvCaches::new(&config),
                |mut caches| {
                    model
                        .forward(
                            black_box(&ids),
                            AttentionMaskMode::PrefixLm { prefix_len: 16 },
                            &mut caches,
                        )
                        .expect("forward")
                },
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

/// Cached single-token decode vs recomputing the whole sequence: the
/// per-cycle KV cache win at seq 128.
fn bench_decode_vs_recompute(c: &mut Criterion) {
    let mut group = c.benchmark_group("hrm_decode");
    group.sample_size(10);

    let seq = 128;
    let config = bench_config(2, 2);
    let model = HrmTextModel::new_random(config.clone(), 42).expect("bench model");
    let ids = token_ids(seq);

    // Prefill positions 0..seq-1 once; decode position seq-1 repeatedly on a
    // cloned cache (decode_step appends, so each iteration needs a fresh copy).
    let mut prefilled = HrmKvCaches::new(&config);
    model
        .prefill(&ids[..seq - 1], 16, &mut prefilled)
        .expect("prefill");
    let last_token = ids[seq - 1];

    group.bench_function("hrm_decode_step_seq128", |b| {
        b.iter_batched(
            || prefilled.clone(),
            |mut caches| {
                model
                    .decode_step(black_box(last_token), seq - 1, &mut caches)
                    .expect("decode_step")
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("hrm_full_recompute_seq128", |b| {
        b.iter_batched(
            || HrmKvCaches::new(&config),
            |mut caches| {
                model
                    .forward(
                        black_box(&ids),
                        AttentionMaskMode::PrefixLm { prefix_len: 16 },
                        &mut caches,
                    )
                    .expect("forward")
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Not timed: report the FP32 KV footprint for hrm_text_l() at 4K context
/// (the ADR-199 ~3x headline number). Asserted in tests/hrm_text_test.rs.
fn report_kv_memory(c: &mut Criterion) {
    let config = HrmTextConfig::hrm_text_l();
    let bytes = HrmKvCaches::projected_memory_bytes(&config, 4096);
    let standard = config.num_hidden_layers * 2 * 4096 * config.kv_dim() * 4;
    println!(
        "hrm_kv_memory: hrm_text_l @ 4096 ctx FP32 = {} bytes ({:.2} GiB), \
         {}x a standard 24-layer model ({} bytes)",
        bytes,
        bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        bytes / standard,
        standard
    );
    // Trivial criterion entry so the group shows up in reports without
    // timing anything meaningful.
    c.bench_function("hrm_kv_memory_accounting", |b| {
        b.iter(|| black_box(HrmKvCaches::projected_memory_bytes(&config, 4096)))
    });
}

/// R2 infrastructure: full-cycle greedy generation vs self-speculative
/// decoding (1x1 draft / 2x2 verify) producing the IDENTICAL token sequence.
/// On synthetic weights acceptance is near-random, so this measures
/// machinery overhead; the real-weight acceptance rate is the R2 gate
/// (> 50 % per ADR-199).
fn bench_self_speculative(c: &mut Criterion) {
    let mut group = c.benchmark_group("hrm_self_speculative");
    group.sample_size(10);

    let config = bench_config(2, 2);
    let prompt = token_ids(16);
    let max_new = 24;

    let model = HrmTextModel::new_random(config.clone(), 42).expect("bench model");
    group.bench_function("greedy_2x2_baseline", |b| {
        b.iter(|| {
            model
                .generate_greedy(black_box(&prompt), 16, max_new)
                .expect("greedy")
        })
    });

    let mut spec_model = HrmTextModel::new_random(config, 42).expect("bench model");
    group.bench_function("self_spec_draft1x1_verify2x2", |b| {
        b.iter(|| {
            let mut dec = SelfSpeculativeDecoder::new(
                &mut spec_model,
                HrmCycles::new(1, 1),
                HrmCycles::new(2, 2),
            )
            .expect("decoder");
            dec.generate_greedy(black_box(&prompt), 16, max_new)
                .expect("self-spec")
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_forward_cycles,
    bench_decode_vs_recompute,
    bench_self_speculative,
    report_kv_memory,
);
criterion_main!(benches);
