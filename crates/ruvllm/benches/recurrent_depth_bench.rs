#![allow(clippy::all, unused_imports, dead_code, unexpected_cfgs)]
//! Recurrent-Depth Transformer benchmarks (RDT substrate + OpenMythos).
//!
//! Measures prefill forward, full-sequence forward at varying lengths, and
//! incremental KV-cache decode for the GQA and MLA attention variants. Run with:
//!
//! ```bash
//! cargo bench -p ruvllm --features candle --bench recurrent_depth_bench
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

    fn rand_mythos(cfg: MythosConfig) -> OpenMythos {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        OpenMythos::load(vb, cfg).expect("load mythos")
    }

    fn rand_rdt(cfg: RdtConfig) -> RdtModel {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        RdtModel::load(vb, cfg).expect("load rdt")
    }

    fn ids(seq: usize) -> Tensor {
        let v: Vec<u32> = (0..seq as u32).map(|i| i % 4096).collect();
        Tensor::from_vec(v, (1, seq), &Device::Cpu).unwrap()
    }

    pub fn bench_mythos_forward(c: &mut Criterion) {
        let mut g = c.benchmark_group("mythos_forward_gqa");
        let model = rand_mythos(mythos_cfg());
        for &seq in &[32usize, 128] {
            let input = ids(seq);
            g.bench_with_input(BenchmarkId::from_parameter(seq), &seq, |b, _| {
                b.iter(|| {
                    let out = model.forward(black_box(&input)).unwrap();
                    black_box(out);
                })
            });
        }
        g.finish();
    }

    pub fn bench_mythos_forward_mla(c: &mut Criterion) {
        let mut g = c.benchmark_group("mythos_forward_mla");
        let mut cfg = mythos_cfg();
        cfg.attn_type = ruvllm::models::openmythos::AttnType::Mla;
        let model = rand_mythos(cfg);
        for &seq in &[32usize, 128] {
            let input = ids(seq);
            g.bench_with_input(BenchmarkId::from_parameter(seq), &seq, |b, _| {
                b.iter(|| {
                    let out = model.forward(black_box(&input)).unwrap();
                    black_box(out);
                })
            });
        }
        g.finish();
    }

    pub fn bench_mythos_decode(c: &mut Criterion) {
        let mut g = c.benchmark_group("mythos_decode");
        let cfg = mythos_cfg();
        let model = rand_mythos(cfg.clone());
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

    pub fn bench_rdt_forward(c: &mut Criterion) {
        let mut g = c.benchmark_group("rdt_forward");
        let cfg = RdtConfig {
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
        };
        let model = rand_rdt(cfg);
        for &seq in &[32usize, 128] {
            let input = ids(seq);
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

#[cfg(feature = "candle")]
criterion_group!(
    benches,
    candle_bench::bench_mythos_forward,
    candle_bench::bench_mythos_forward_mla,
    candle_bench::bench_mythos_decode,
    candle_bench::bench_rdt_forward,
);

#[cfg(not(feature = "candle"))]
criterion_group!(benches, noop);
#[cfg(not(feature = "candle"))]
fn noop(_c: &mut Criterion) {}

criterion_main!(benches);
