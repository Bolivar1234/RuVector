---
adr: 199
title: "RuvLLM HRM-Text Backend — Hierarchical Recurrent Transformer Inference in Rust"
status: proposed
date: 2026-06-11
authors: [ruvnet, claude-flow]
related: [ADR-002, ADR-180, ADR-181, ADR-183, ADR-189]
tags: [ruvllm, hrm, hierarchical-reasoning, inference, prefixlm, recurrence, candle, rust]
---

# ADR-199 — RuvLLM HRM-Text Backend: Hierarchical Recurrent Transformer Inference in Rust

## Status

**Proposed.** Targets `crates/ruvllm` v2.7. Inference-only in phase 1;
training remains in upstream PyTorch (`sapientinc/HRM-Text`).

## Question Being Answered

> Could [sapientinc/HRM-Text](https://github.com/sapientinc/HRM-Text) be used
> with ruvllm implemented in Rust?

**Yes — for inference, with moderate effort.** HRM-Text's architecture
decomposes at inference time into primitives ruvllm already ships (RoPE,
SwiGLU, RMSNorm, flash attention, KV caching), composed in a nested H/L
recurrence loop with additive input injection. Nothing in the forward pass
requires PyTorch, FlashAttention 3, or CUDA-specific behavior. The genuinely
new work is:

1. a **recurrence-aware forward loop** (H_cycles × L_cycles nested iteration),
2. **PrefixLM attention masking** (bidirectional over the prefix, causal over
   the response),
3. **per-cycle KV cache semantics** (each recurrence cycle re-attends over the
   sequence), and
4. a **safetensors/HF checkpoint loader** for HRM-Text's
   `conversion/convert_to_hf.py` export format (ruvllm today is GGUF-first).

Training (FSDP2, backprop warmup, adam_atan2, multipack sampling) is
explicitly **out of scope** — it stays in PyTorch.

## Context

### What HRM-Text is

HRM-Text (Sapient Inc., Apache-2.0) is a ~1B-parameter text generation model
family built on the Hierarchical Reasoning Model. Its pitch is extreme
pretraining efficiency: 130–600× less compute and 150–900× less data than
conventional pretraining (reference run: 1B params, 16×H100, 46 h, 84.7 %
GSM8k / 60.7 % MMLU).

Architecture (from `models/transformer.py` and
`models/baselines/hrm_nocarry_bp_warmup.py`):

| Component | HRM-Text | Already in ruvllm? |
|-----------|----------|--------------------|
| Attention | Gated MHA + RoPE (configurable theta) | ✅ `kernels::rope`, `flash_attention_neon`, Metal/WGSL shaders |
| FFN | SwiGLU, `intermediate = round(expansion × hidden × 2/3)` aligned to 256 | ✅ Phi-3/Gemma-2 SwiGLU paths |
| Norm | RMSNorm, pre-norm or post-norm | ✅ `rms_norm_neon` |
| Positioning | RoPE | ✅ `precompute_rope_tables_with_config` |
| **H/L hierarchy** | Two recurrent transformer cores; `half_layers: true` splits `n_layers` evenly between H and L | ❌ new |
| **Recurrence** | `for i in 0..H_cycles { for k in 0..L_cycles { z_L = L(z_L, inject=z_H) } ; z_H = H(z_H, inject=z_L) }` | ❌ new |
| **Input injection** | Additive: `hidden + injection` | ❌ new (trivial) |
| **z_L init** | Learned buffer (truncated normal), broadcast over sequence | ❌ new (trivial) |
| **PrefixLM** | Bidirectional attention over prefix, causal over response | ❌ new mask mode |
| ACT / halting | **None** — cycles are fixed and deterministic | n/a (simplifies port) |
| Checkpoints | FSDP2 → HF Transformers via `convert_to_hf.py` (EMA weights by default) | ⚠️ ruvllm is GGUF-first; needs safetensors path |

Model sizes (all within ruvllm's proven envelope — RuvLTRA-Medium is 3B):

| Size | Layers | Hidden | Heads |
|------|--------|--------|-------|
| B | 12 | 1024 | 8 |
| L (0.6B) | 24 | 1280 | 10 |
| XL (1B) | 32 | 1536 | 12 |
| XXL | 72 | 1792 | 14 |
| XXL_wide | 32 | 2560 | 20 |

### Why ruvllm wants this

- **Edge-class reasoning.** A 1B model scoring 84.7 % GSM8k is exactly the
  profile of ruvllm's deployment targets (M-series, Pi 5 + Hailo-10H cluster
  per ADR-173/179, WASM). Q4_K-quantized XL (~0.7 GB weights) fits everywhere
  RuvLTRA-Small does.
- **Cheap custom pretraining.** ~$1.5K to pretrain a 1B model makes
  RuvLTRA-class *custom* foundation models plausible (train in PyTorch, serve
  in ruvllm), instead of only fine-tuning third-party checkpoints.
- **Recurrence is compute-for-memory.** Effective depth =
  `cycles × n_layers/2` with only `n_layers` layers of weights. On
  memory-constrained edge nodes this is the right trade — weights are the
  bottleneck, not FLOPs (sparse attention ADR-183/189 already attacks the
  FLOP side).
- **SONA/MicroLoRA composability.** HRM's H-level state `z_H` is a natural
  attachment point for ruvllm's per-request MicroLoRA adapters and SONA
  instant loop — adapting the *reasoning* stream, not just output logits.

### Key constraint discovered during analysis

HRM-Text's upstream code targets **FlashAttention 3 on Hopper GPUs** and
PyTorch FSDP2. None of that is portable, and none of it is needed:
FlashAttention 3 is a kernel-level optimization of mathematically standard
attention. ruvllm's existing Flash Attention 2 (NEON/Metal/WGSL) computes the
same function. The PrefixLM **mask**, not the FA3 kernel, is the semantic
requirement.

## Decision

Implement an inference-only HRM-Text backend in `crates/ruvllm`, following the
existing per-architecture module pattern (`backends/phi3.rs`,
`backends/gemma2.rs`), in four phases.

### Phase 1 — Core architecture module (`backends/hrm_text.rs`)

New module exporting `HrmTextConfig`, `HrmRecurrentCore`, `HrmTextModel`,
registered in `backends/mod.rs` and added to the `ModelArchitecture` enum.

```rust
/// HRM-Text model configuration (mirrors HierarchicalReasoningModelConfig).
#[derive(Debug, Clone)]
pub struct HrmTextConfig {
    // -- standard transformer core (shared by H and L unless overridden) --
    pub hidden_size: usize,
    pub intermediate_size: usize,     // round(expansion * hidden * 2/3), 256-aligned
    pub num_hidden_layers: usize,     // TOTAL; half_layers splits H/L
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub norm_position: NormPosition,  // PreNorm | PostNorm
    pub use_flash_attention: bool,
    // -- HRM-specific --
    pub h_cycles: usize,              // upstream: H_cycles
    pub l_cycles: usize,              // upstream: L_cycles
    pub half_layers: bool,            // requires num_hidden_layers % 2 == 0
    pub h_override: Option<HrmCoreOverride>, // asymmetric H config (upstream H_override)
    // -- PrefixLM --
    pub prefix_bidirectional: bool,   // bidirectional attention over prompt region
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

impl HrmTextConfig {
    pub fn hrm_text_l() -> Self  { /* 24 layers, 1280, 10 heads — 0.6B */ }
    pub fn hrm_text_xl() -> Self { /* 32 layers, 1536, 12 heads — 1B   */ }
}
```

```rust
/// One recurrent core (H or L): a stack of standard transformer blocks
/// plus additive input injection at the bottom.
pub struct HrmRecurrentCore {
    layers: Vec<HrmDecoderLayer>,     // attention + SwiGLU, reusing phi3-style blocks
    rope: RopeTables,
}

impl HrmRecurrentCore {
    /// z' = core(z + injection)  — additive fusion per upstream.
    pub fn forward(
        &self,
        z: &Tensor,
        injection: &Tensor,
        mask: &AttentionMask,
        cache: Option<&mut CycleKvCache>,
    ) -> Result<Tensor> { ... }
}

pub struct HrmTextModel {
    embed_tokens: Embedding,
    h_level: HrmRecurrentCore,        // num_hidden_layers / 2 layers
    l_level: HrmRecurrentCore,        // num_hidden_layers / 2 layers
    z_l_init: Tensor,                 // learned buffer [hidden_size], broadcast over seq
    final_norm: RmsNorm,              // pre-norm variant only
    lm_head: Linear,
}
```

Forward pass — direct port of the upstream no-carry loop (no ACT/halting
exists upstream, which removes the hardest porting risk):

```rust
fn forward(&self, input_ids: &[u32], mask: &AttentionMask,
           caches: &mut HrmKvCaches) -> Result<Tensor> {
    let x = self.embed_tokens.forward(input_ids)?;
    let mut z_h = x;                                  // z_H ← input embeddings
    let mut z_l = self.z_l_init.broadcast_to_seq(input_ids.len())?;

    for i in 0..self.config.h_cycles {
        for k in 0..self.config.l_cycles {
            z_l = self.l_level.forward(&z_l, /*inject=*/ &z_h, mask,
                                       caches.l_cycle(i, k))?;
        }
        z_h = self.h_level.forward(&z_h, /*inject=*/ &z_l, mask,
                                   caches.h_cycle(i))?;
    }
    self.lm_head.forward(&self.final_norm.forward(&z_h)?)
}
```

### Phase 2 — PrefixLM attention mask + recurrence-aware KV cache

**PrefixLM mask.** Extend `AttentionConfig` with a mask mode:

```rust
pub enum AttentionMaskMode {
    Causal,                            // existing default
    PrefixLm { prefix_len: usize },    // bidirectional in [0, prefix_len), causal after
}
```

Semantics: `allowed(q, k) = k < prefix_len || k <= q`. This is a mask-function
change in the existing flash-attention inner loop (NEON, Metal, WGSL), not a
new kernel. Decode steps (`q >= prefix_len`) remain purely causal, so
**incremental generation is unchanged after prefill** — the bidirectional
region is fixed once the prompt is processed.

**Per-cycle KV cache.** The recurrence means each (cycle, layer) pair attends
with *different* K/V (states evolve every cycle). A naive port re-prefills
every cycle on every decode step — O(T) per token per cycle. Instead:

```rust
/// KV caches indexed by (level, cycle, layer).
/// Total slots = (h_cycles * l_cycles + h_cycles) * (num_hidden_layers / 2).
pub struct HrmKvCaches { ... }
```

For HRM-Text-XL with H_cycles=2, L_cycles=2 (representative): (2·2+2)·16 = 96
cache slots vs 32 for a standard 32-layer model — **3× KV memory**. This is
the real cost of recurrence at inference and the headline number to validate.
Mitigations, all existing ruvllm machinery applied per-slot:

- `TwoTierKvCache` (FP16 tail + Q4 store) — default on.
- TurboQuant 3-bit tier (ADR-181 lineage) for long contexts — 10.7× per slot.
- Optional `recompute_l_cache: true` mode: keep only H-level caches resident
  and recompute L-level prefill per decode step — memory parity with a
  standard model at ~`l_cycles×` decode FLOPs; right trade on Hailo/Pi nodes
  where the sparse-attention kernel (ADR-183/189 `decode_step`, O(log T))
  makes recompute cheap.

### Phase 3 — Checkpoint loading

HRM-Text ships `conversion/convert_to_hf.py` (FSDP2 → HF Transformers format,
EMA weights preferred). Loader strategy:

1. **safetensors loader** in `ruvllm::hub` — read the HF export directly
   (memory-mapped, like existing GGUF mmap). Tensor name mapping table
   `hf_name → (level, layer, role)` lives in `hrm_text.rs`, including the
   `z_L` init buffer and the H/L split implied by `half_layers`.
2. **GGUF conversion script** (`scripts/hrm_text_to_gguf.py`, upstream-side)
   writing custom KV metadata: `hrm.h_cycles`, `hrm.l_cycles`,
   `hrm.half_layers`, `hrm.prefix_bidirectional`. This unlocks the entire
   existing quantization path (Q4_K/Q5_K/Q8) and `ruvllm-cli quantize`
   unchanged.
3. Tokenizer: HF `tokenizer.json` via existing `ruvllm::tokenizer` — no new
   work expected.

Autodetect (`autodetect.rs`): presence of `hrm.h_cycles` metadata or
`z_l_init` tensor selects `ModelArchitecture::HrmText`.

### Phase 4 — Serving, CLI, and SONA integration

- `ruvllm-cli`: add HRM models to `models.rs` registry; `serve`/`chat`/
  `benchmark` work through the standard `Backend` trait once Phase 1–3 land.
- Continuous batching (ADR-180): requests with equal `(h_cycles, l_cycles)`
  batch together; the scheduler treats one full recurrence as one forward.
- Speculative decoding: HRM model as **target** with RuvLTRA-Small as draft
  is supported immediately; HRM-as-draft deferred (recurrence latency makes
  it a poor draft).
- SONA/MicroLoRA: attach rank-1/2 adapters to H-level Q/V projections —
  adapting the reasoning stream. Deferred to a follow-up ADR once baseline
  quality is verified.

### Explicit non-goals

- **Training in Rust** (FSDP2, bp-warmup gradient windowing, adam_atan2,
  multipack/LPT sampling) — stays upstream in PyTorch.
- **FlashAttention 3 parity** — not required; FA3 is a Hopper kernel
  optimization, not a semantic dependency.
- **ACT/adaptive halting** — upstream HRM-Text has none (fixed cycles);
  do not invent one.

## Consequences

### Positive

- 1B-class reasoning models (84.7 % GSM8k) become servable on every ruvllm
  target — Metal, CUDA, ANE, WASM, Hailo/Pi — with the full quantization,
  batching, and SONA stack.
- Opens a "pretrain custom 1B for ~$1.5K in PyTorch → serve in ruvllm"
  pipeline for RuvLTRA-class custom models.
- PrefixLM mask mode and the safetensors loader are independently useful
  (other PrefixLM/UL2-style checkpoints; non-GGUF model loading generally).
- Weight memory stays at `n_layers` while effective depth is
  `cycles × n_layers/2` — favorable for weight-bound edge nodes.

### Negative / costs

- **KV cache multiplied by ~(H_cycles·L_cycles + H_cycles)/2** vs a standard
  equal-depth model (≈3× for the representative config). Mitigated but not
  eliminated by two-tier/TurboQuant/recompute modes; must be benchmarked
  before `accepted`.
- Decode latency per token scales with total cycle count — HRM trades
  parameter count for sequential compute. Edge throughput numbers needed.
- New maintenance surface: recurrence loop, cache indexing, mask mode, and a
  second checkpoint format.
- Upstream HRM-Text is young; checkpoint format churn in
  `convert_to_hf.py` is a tracking burden (pin a verified commit/release).

### Licensing

HRM-Text is Apache-2.0; ruvllm is Apache-2.0/MIT. Architecture
reimplementation in Rust plus loading Apache-2.0 weights is unambiguously
compatible. Cite upstream BibTeX in the module docs.

## Alternatives considered

1. **Bind PyTorch via `tch-rs` and run the upstream model.** Rejected:
   drags libtorch (~2 GB) into every target, kills WASM/ANE/Hailo, defeats
   ruvllm's purpose.
2. **Wait for llama.cpp / GGUF ecosystem support.** Rejected: HRM recurrence
   + PrefixLM is unlikely to land upstream there soon, and ruvector controls
   its own kernel stack — this is exactly the differentiation ruvllm exists
   for.
3. **Distill HRM-Text into a standard RuvLTRA architecture.** Deferred, not
   rejected — distillation forfeits the weight-memory advantage but would
   need zero runtime changes. Reasonable fallback if Phase 2 KV costs prove
   prohibitive on the smallest targets.
4. **ONNX export → ruvector-onnx path (ADR-194).** Rejected for generation:
   recurrence with per-cycle KV caching maps poorly to static ONNX graphs;
   acceptable only for fixed-length scoring use cases.

## Verification plan (gate to `accepted`)

1. **Parity:** logits within 1e-3 (FP16) of upstream
   `simple_inference_engine.py` on a pinned HRM-Text-L checkpoint, 32 prompts,
   greedy decode, exact token-match over 128 tokens.
2. **Mask correctness:** property tests for `PrefixLm` mode — bidirectional
   block equals full attention restricted to prefix; causal region matches
   existing causal path bit-for-bit; decode-step equivalence to full-sequence
   forward (extend `tests/` pattern from ADR-186).
3. **Memory:** measured KV footprint at seq 4K/8K for XL across {FP16,
   two-tier, TurboQuant-3bit, recompute_l_cache} on M4 Pro and Pi 5.
4. **Throughput:** prefill + decode tok/s vs RuvLTRA-Medium 3B (similar
   quality class) on M4 Pro; report in `benches/e2e_bench.rs`.
5. **Quality:** GSM8k subset via the existing evaluation harness ≥ 95 % of
   upstream-reported score at Q4_K.

## References

- HRM-Text repository: https://github.com/sapientinc/HRM-Text (Apache-2.0)
  — `models/transformer.py`, `models/baselines/hrm_nocarry_bp_warmup.py`,
  `models/flash_attention_prefixlm_v2.py`, `conversion/convert_to_hf.py`
- Wang et al., *Hierarchical Reasoning Model*, arXiv:2506.21734
- ADR-002 (ruvllm integration), ADR-180 (continuous batching serving),
  ADR-181 (pi-quant/BitNet), ADR-183–190 (sparse attention kernel family,
  esp. ADR-189 KV-cache incremental decode)
- `crates/ruvllm/src/backends/phi3.rs`, `gemma2.rs` — the per-architecture
  module pattern this backend follows
