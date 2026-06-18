#![allow(clippy::all, unused_imports, dead_code, unexpected_cfgs)]
//! Recurrent-Depth Transformer benchmarks (RDT substrate + OpenMythos).
//!
//! Measures prefill forward, full-sequence forward at varying lengths, and
//! incremental KV-cache decode for the GQA and MLA attention variants.
//!
//! Run CPU benchmarks:
//! ```bash
//! cargo bench -p ruvllm --features candle --bench recurrent_depth_bench
//! ```
//!
//! Run GPU (CUDA) benchmarks:
//! ```bash
//! cargo bench -p ruvllm --features candle,cuda --bench recurrent_depth_bench
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "candle")]
mod candle_bench {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use ruvllm::models::openmythos::{MythosCache, MythosConfig, OpenMythos};
    use ruvllm::models::rdt::{RdtConfig, RdtModel};

    /// Moderate config: large enough to be representative, small enough to bench.
    fn mythos_cfg() -> MythosConfig {
        MythosConfig {
            vocab_size: 4096,
            dim: 512,
            n_heads: 8,
            n_kv_heads: 2,
            max_seq_len: 1024,
            max_loop_iters: 8,
            prelude_layers: 2,
            coda_layers: 2,
            attn_type: ruvllm::models::openmythos::AttnType::Gqa,
            kv_lora_rank: 128,
            q_lora_rank: 256,
            qk_rope_head_dim: 32,
            qk_nope_head_dim: 64,
            v_head_dim: 64,
            expert_dim: 512,
            n_experts: 8,
            n_shared_experts: 2,
            n_experts_per_tok: 2,
            use_moe: true,
            act_threshold: 0.99,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            loop_dim: 32,
            lora_rank: 8,
        }
    }

    fn rdt_cfg() -> RdtConfig {
        RdtConfig {
            hidden_size: 512,
            intermediate_size: 1376,
            num_heads: 8,
            num_kv_heads: 2,
            vocab_size: 4096,
            max_position_embeddings: 1024,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            num_shared_blocks: 1,
            max_loops: 8,
            halt_threshold: 0.9,
        }
    }

    fn rand_mythos_on(cfg: MythosConfig, device: &Device, dtype: DType) -> OpenMythos {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);
        OpenMythos::load(vb, cfg).expect("load mythos")
    }

    fn rand_rdt_on(cfg: RdtConfig, device: &Device, dtype: DType) -> RdtModel {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);
        RdtModel::load(vb, cfg).expect("load rdt")
    }

    fn ids_on(seq: usize, device: &Device) -> Tensor {
        let v: Vec<u32> = (0..seq as u32).map(|i| i % 4096).collect();
        Tensor::from_vec(v, (1, seq), device).unwrap()
    }

    // -----------------------------------------------------------------------
    // CPU benchmarks (F32)
    // -----------------------------------------------------------------------

    pub fn bench_mythos_forward_cpu(c: &mut Criterion) {
        let mut g = c.benchmark_group("cpu/mythos_forward_gqa_f32");
        let model = rand_mythos_on(mythos_cfg(), &Device::Cpu, DType::F32);
        for &seq in &[32usize, 128, 256] {
            let input = ids_on(seq, &Device::Cpu);
            g.bench_with_input(BenchmarkId::from_parameter(seq), &seq, |b, _| {
                b.iter(|| {
                    let out = model.forward(black_box(&input)).unwrap();
                    black_box(out);
                })
            });
        }
        g.finish();
    }

    pub fn bench_mythos_forward_mla_cpu(c: &mut Criterion) {
        let mut g = c.benchmark_group("cpu/mythos_forward_mla_f32");
        let mut cfg = mythos_cfg();
        cfg.attn_type = ruvllm::models::openmythos::AttnType::Mla;
        let model = rand_mythos_on(cfg, &Device::Cpu, DType::F32);
        for &seq in &[32usize, 128] {
            let input = ids_on(seq, &Device::Cpu);
            g.bench_with_input(BenchmarkId::from_parameter(seq), &seq, |b, _| {
                b.iter(|| {
                    let out = model.forward(black_box(&input)).unwrap();
                    black_box(out);
                })
            });
        }
        g.finish();
    }

    pub fn bench_mythos_decode_cpu(c: &mut Criterion) {
        let mut g = c.benchmark_group("cpu/mythos_decode_f32");
        let cfg = mythos_cfg();
        let model = rand_mythos_on(cfg.clone(), &Device::Cpu, DType::F32);
        let prompt: Vec<u32> = (0..32u32).collect();
        g.bench_function("prompt32_gen16", |b| {
            b.iter(|| {
                let out = model
                    .generate(black_box(&prompt), 16, cfg.max_loop_iters, None)
                    .unwrap();
                black_box(out);
            })
        });
        g.finish();
    }

    pub fn bench_rdt_forward_cpu(c: &mut Criterion) {
        let mut g = c.benchmark_group("cpu/rdt_forward_f32");
        let model = rand_rdt_on(rdt_cfg(), &Device::Cpu, DType::F32);
        for &seq in &[32usize, 128, 256] {
            let input = ids_on(seq, &Device::Cpu);
            g.bench_with_input(BenchmarkId::from_parameter(seq), &seq, |b, _| {
                b.iter(|| {
                    let out = model.forward(black_box(&input)).unwrap();
                    black_box(out);
                })
            });
        }
        g.finish();
    }

    // -----------------------------------------------------------------------
    // GPU benchmarks (CUDA) — F32 and BF16
    // -----------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    fn cuda_device() -> Device {
        Device::new_cuda(0).expect("CUDA device 0 not available")
    }

    #[cfg(feature = "cuda")]
    pub fn bench_mythos_forward_cuda_f32(c: &mut Criterion) {
        let dev = cuda_device();
        let mut g = c.benchmark_group("cuda/mythos_forward_gqa_f32");
        let model = rand_mythos_on(mythos_cfg(), &dev, DType::F32);
        for &seq in &[32usize, 128, 256, 512] {
            let input = ids_on(seq, &dev);
            g.bench_with_input(BenchmarkId::from_parameter(seq), &seq, |b, _| {
                b.iter(|| {
                    let out = model.forward(black_box(&input)).unwrap();
                    black_box(out);
                })
            });
        }
        g.finish();
    }

    #[cfg(feature = "cuda")]
    pub fn bench_mythos_forward_cuda_bf16(c: &mut Criterion) {
        let dev = cuda_device();
        let mut g = c.benchmark_group("cuda/mythos_forward_gqa_bf16");
        let model = rand_mythos_on(mythos_cfg(), &dev, DType::BF16);
        for &seq in &[32usize, 128, 256, 512] {
            let input = ids_on(seq, &dev);
            g.bench_with_input(BenchmarkId::from_parameter(seq), &seq, |b, _| {
                b.iter(|| {
                    let out = model.forward(black_box(&input)).unwrap();
                    black_box(out);
                })
            });
        }
        g.finish();
    }

    #[cfg(feature = "cuda")]
    pub fn bench_mythos_decode_cuda_bf16(c: &mut Criterion) {
        let dev = cuda_device();
        let mut g = c.benchmark_group("cuda/mythos_decode_bf16");
        let cfg = mythos_cfg();
        let model = rand_mythos_on(cfg.clone(), &dev, DType::BF16);
        let prompt: Vec<u32> = (0..32u32).collect();
        g.bench_function("prompt32_gen16", |b| {
            b.iter(|| {
                let out = model
                    .generate(black_box(&prompt), 16, cfg.max_loop_iters, None)
                    .unwrap();
                black_box(out);
            })
        });
        g.finish();
    }

    #[cfg(feature = "cuda")]
    pub fn bench_rdt_forward_cuda_f32(c: &mut Criterion) {
        let dev = cuda_device();
        let mut g = c.benchmark_group("cuda/rdt_forward_f32");
        let model = rand_rdt_on(rdt_cfg(), &dev, DType::F32);
        for &seq in &[32usize, 128, 256, 512] {
            let input = ids_on(seq, &dev);
            g.bench_with_input(BenchmarkId::from_parameter(seq), &seq, |b, _| {
                b.iter(|| {
                    let out = model.forward(black_box(&input)).unwrap();
                    black_box(out);
                })
            });
        }
        g.finish();
    }

    #[cfg(feature = "cuda")]
    pub fn bench_rdt_forward_cuda_bf16(c: &mut Criterion) {
        let dev = cuda_device();
        let mut g = c.benchmark_group("cuda/rdt_forward_bf16");
        let model = rand_rdt_on(rdt_cfg(), &dev, DType::BF16);
        for &seq in &[32usize, 128, 256, 512] {
            let input = ids_on(seq, &dev);
            g.bench_with_input(BenchmarkId::from_parameter(seq), &seq, |b, _| {
                b.iter(|| {
                    let out = model.forward(black_box(&input)).unwrap();
                    black_box(out);
                })
            });
        }
        g.finish();
    }
}

// CPU criterion groups (always registered)
#[cfg(feature = "candle")]
criterion_group!(
    cpu_benches,
    candle_bench::bench_mythos_forward_cpu,
    candle_bench::bench_mythos_forward_mla_cpu,
    candle_bench::bench_mythos_decode_cpu,
    candle_bench::bench_rdt_forward_cpu,
);

// CUDA criterion groups (only when cuda feature is active)
#[cfg(all(feature = "candle", feature = "cuda"))]
criterion_group!(
    cuda_benches,
    candle_bench::bench_mythos_forward_cuda_f32,
    candle_bench::bench_mythos_forward_cuda_bf16,
    candle_bench::bench_mythos_decode_cuda_bf16,
    candle_bench::bench_rdt_forward_cuda_f32,
    candle_bench::bench_rdt_forward_cuda_bf16,
);

// No-op stubs when features are absent
#[cfg(not(feature = "candle"))]
criterion_group!(cpu_benches, noop);
#[cfg(not(feature = "candle"))]
fn noop(_c: &mut Criterion) {}

#[cfg(all(feature = "candle", feature = "cuda"))]
criterion_main!(cpu_benches, cuda_benches);

#[cfg(all(feature = "candle", not(feature = "cuda")))]
criterion_main!(cpu_benches);

#[cfg(not(feature = "candle"))]
criterion_main!(cpu_benches);
