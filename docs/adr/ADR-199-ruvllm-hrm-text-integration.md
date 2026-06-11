---
adr: 199
title: "RuvLLM × HRM-Text — Hierarchical Recurrent Reasoning Kernel for the Governed Runtime"
status: proposed
date: 2026-06-11
authors: [ruvnet, claude-flow]
related: [ADR-002, ADR-180, ADR-181, ADR-183, ADR-189]
tags: [ruvllm, hrm, hierarchical-reasoning, reasoning-kernel, prefixlm, planner, verifier, router, vllm, rust]
---

# ADR-199 — RuvLLM × HRM-Text: Reasoning Kernel, Not "Just Another Model"

## Status

**Proposed.** Phase A/B target a new `crates/ruvllm-hrm` adapter crate behind
a local vLLM/SGLang endpoint. Native Rust inference (Phase D) is gated on the
acceptance tests below. Training remains in upstream PyTorch
(`sapientinc/HRM-Text`).

## Question Being Answered

> Could [sapientinc/HRM-Text](https://github.com/sapientinc/HRM-Text) be used
> with ruvllm implemented in Rust?

**Yes — but not as a drop-in chat model.** HRM-Text-1B is a hierarchical
recurrent model (slow H-level + fast L-level Transformer stacks iterating with
additive state injection — "effectively unbounded compute depth at bounded
parameter count"). Critically, the released checkpoint is **pre-alignment**:
PrefixLM-pretrained with condition prefix tokens, **not** instruction-tuned,
dialogue-tuned, or RLHF'd. Dropped into a chat stack as if it were Llama or
Qwen, it will look broken. Used as a **reasoning kernel inside a governed
runtime**, it is exactly the profile ruvllm wants: a 1B model that can
repeatedly think, route, verify, and compress state — cheap recurrent
reasoning for edge agentic systems where planning/verification steps should
not hit a giant remote model.

## Context

### What HRM-Text is

| Property | Value |
|----------|-------|
| Model | `sapientinc/HRM-Text-1B` (HuggingFace), Apache-2.0 |
| Architecture | Dual-timescale HRM: two Transformer cores (H-level slow / L-level fast), `H_cycles × L_cycles` nested recurrence, additive input injection, learned `z_L` init buffer; RoPE, SwiGLU, RMSNorm; `half_layers` splits depth between H and L |
| Objective | PrefixLM — prompt tokens attend bidirectionally, response tokens causally; inference requires `token_type_ids = ones` to match training |
| Conditioning | Condition prefix tags baked into pretraining: `direct`, `cot`, `noisy`, `synth` (composable, e.g. `synth,cot`) |
| Alignment | **Pre-alignment.** Model card: NLP tasks → `direct` + 2–8 few-shot; reasoning/math → `synth,cot`; chat behavior uneven by design |
| Tokenizer | Custom 65,536-token vocab with special condition tokens |
| Serving today | vLLM (`vllm serve sapientinc/HRM-Text-1B`) and SGLang, documented on the model card |
| Training stack | PyTorch FSDP2, FlashAttention 3 (Hopper), multipack/LPT batching, adam_atan2 — **not portable, not needed for inference** |
| Efficiency claim | 130–600× less pretraining compute, 150–900× less data; 1B reference run: 16×H100, 46 h (~$1.5K), 84.7 % GSM8k / 60.7 % MMLU |
| Caveats | English-only, weak at code (not code-trained), pre-alignment output quality |

### Why this fits ruvllm

The value is **not raw chat quality** — it is cheap recurrent reasoning as a
building block:

1. **Planning passes** — task + constraints + state → compact plan,
   decomposition, next action.
2. **Verifier passes** — generated answer + rules + evidence → pass/fail,
   contradictions, missing facts.
3. **Latent scratchpad generation** — symbolic task frame → intermediate
   reasoning tokens or compressed state.
4. **Router intelligence** — user intent + context → pick local model, tool,
   vector search, workflow, or remote model.
5. **ruvector-enhanced memory** — HRM hidden/generated state → embeddings,
   recurrence summaries, task memory.

The novel composition, and the actual decision of this ADR:

```
HRM-Text  = recurrent controller (plan / verify / route / compress)
ruvector  = memory and contrastive state
ruvLLM    = governed execution runtime
RuFlo     = workflow harness
```

A small model that repeatedly thinks is more valuable inside a governed
pipeline than as an endpoint. This is especially true on the edge targets
ruvllm already owns (M-series, Pi 5 + Hailo-10H per ADR-173/179, WASM).

## Decision

Integrate HRM-Text as a **reasoning kernel** in three layers, with native
Rust inference deferred until the kernel proves value. New crate:

```
crates/ruvllm-hrm/
  src/
    backend.rs      # LlmBackend trait + HrmTextBackend (vLLM/SGLang client)
    prompt.rs       # HrmMode → condition-prefix templates
    router.rs       # routing decisions (model/tool/vector-search/workflow/remote)
    verifier.rs     # verdict parsing, contradiction/evidence checks
    benchmark.rs    # acceptance-test harness
  examples/
    plan.rs
    verify.rs
    extract_json.rs
```

### Layer 1 — Rust adapter behind a local endpoint (do first)

Provider trait, so the transport can later swap from HTTP to native without
touching callers:

```rust
pub trait LlmBackend {
    async fn generate(&self, req: GenerateRequest) -> anyhow::Result<GenerateResponse>;
    async fn score(&self, req: ScoreRequest) -> anyhow::Result<ScoreResponse>;
    async fn embed_state(&self, req: StateRequest) -> anyhow::Result<StateResponse>;
}

pub struct HrmTextBackend {
    endpoint: String,        // http://localhost:8000/v1
    model: String,           // sapientinc/HRM-Text-1B
    tokenizer_mode: PrefixMode,
}
```

Serve upstream weights with zero porting risk:

```bash
pip install vllm
vllm serve sapientinc/HRM-Text-1B
```

Rust side calls the OpenAI-compatible completions endpoint (`reqwest` +
`serde_json`, temperature ~0.2 for kernel tasks, `max_tokens` ≤ 512).
HuggingFace documents vLLM serving for this checkpoint — fastest, lowest-risk
path.

> **Known risk — PrefixLM mask over the OpenAI API.** The model card requires
> `token_type_ids = torch.ones_like(input_ids)` so prompt tokens attend
> bidirectionally. Verify whether vLLM's HRM-Text integration applies this
> internally; if the generic completions path silently runs causal-only,
> quality degrades invisibly. Phase A includes a parity check (vLLM output vs
> upstream `simple_inference_engine.py` on 32 prompts) before anything builds
> on top. SGLang is the fallback serving path if vLLM's handling is wrong.

### Layer 2 — PrefixLM-aware prompting (the critical part)

The checkpoint is condition-token-trained, not instruction-tuned. ruvllm-hrm
must own the prompt contract — callers select a mode, never write raw
prompts:

```rust
pub enum HrmMode {
    Direct,    // NLP tasks: direct + 2-8 few-shot examples
    SynthCot,  // reasoning/math: synth,cot composite prefix
    Verify,    // contradiction / evidence checking
    Plan,      // task decomposition
    Extract,   // strict JSON output
}

fn build_hrm_prompt(mode: HrmMode, input: &str) -> String {
    match mode {
        HrmMode::Direct => format!(
            "direct\nTask: Extract the answer.\nInput:\n{input}\nOutput:"),
        HrmMode::SynthCot => format!(
            "synth,cot\nSolve carefully.\nProblem:\n{input}\nAnswer:"),
        HrmMode::Verify => format!(
            "direct\nCheck for contradictions, missing evidence, and invalid assumptions.\n{input}\nVerdict:"),
        HrmMode::Plan => format!(
            "synth,cot\nDecompose this into executable steps.\n{input}\nPlan:"),
        HrmMode::Extract => format!(
            "direct\nReturn strict JSON only.\n{input}\nJSON:"),
    }
}
```

`prompt.rs` additionally carries per-mode few-shot banks (model card
recommends 2–8 shots for `direct`), stop sequences, and output parsers
(`Verdict:` → structured pass/fail; `JSON:` → serde-validated value with one
retry on parse failure).

### Layer 3 — ruvllm controller (the actual product)

HRM-Text never answers user queries directly. It sits in the governed loop:

```
User request
  → ruvector retrieval                 (context, memory, patterns)
  → HRM planning pass    (Plan)        (decomposition, next action)
  → model/tool routing   (router.rs)   (local model | tool | vector search | workflow | remote)
  → execution model                    (RuvLTRA / Qwen / remote — the workhorse)
  → HRM verifier pass    (Verify)      (contradictions, missing facts → retry or escalate)
  → final answer
```

`router.rs` returns a typed `RoutingDecision`; `verifier.rs` returns a typed
`Verdict { pass, contradictions, missing_facts }`. Both compose with the
3-tier model routing already in the runtime (ADR-026 lineage): HRM occupies
the gap between Agent Booster (Tier 1) and remote frontier models (Tier 3) —
a local Tier-2 reasoning step at 1B-model latency. HRM generated/hidden state
feeds ruvector as recurrence summaries and task memory (`embed_state`).

### Phase D (deferred) — Native Rust inference in `crates/ruvllm`

Gated on the acceptance tests. The analysis says it is feasible: at inference
HRM-Text reduces to primitives ruvllm already ships (RoPE tables,
`flash_attention_neon`/Metal/WGSL, `rms_norm_neon`, Phi-3-style SwiGLU
blocks) composed in the nested recurrence. The native port, when justified,
follows the `backends/phi3.rs` module pattern (`backends/hrm_text.rs`) and
requires four additions:

1. **Recurrence loop** — direct port of the upstream no-carry forward
   (`z_H ← embed(x)`; `z_L ←` learned buffer; nested
   `H_cycles × L_cycles` with additive injection; no ACT/halting exists
   upstream, which removes the hardest porting risk).
2. **PrefixLM mask mode** — `AttentionMaskMode::PrefixLm { prefix_len }`,
   `allowed(q,k) = k < prefix_len || k <= q`; a mask-function change in the
   existing flash-attention loops, not a new kernel. Decode stays causal
   after prefill.
3. **Per-cycle KV caches** — each (cycle, layer) attends with different K/V:
   `(H_cycles·L_cycles + H_cycles) · n_layers/2` slots ≈ **3× KV memory** vs
   an equal-depth standard model for representative configs. Mitigations are
   existing machinery applied per-slot: `TwoTierKvCache`, TurboQuant 3-bit
   (ADR-181), or a `recompute_l_cache` mode trading FLOPs for memory on
   Hailo/Pi nodes where ADR-189's O(log T) `decode_step` makes recompute
   cheap. This 3× figure is the headline number to benchmark.
4. **Checkpoint loading** — safetensors loader for the
   `conversion/convert_to_hf.py` export (EMA weights default) plus a
   GGUF conversion script with `hrm.h_cycles` / `hrm.l_cycles` /
   `hrm.half_layers` metadata to unlock the existing Q4_K/Q5_K/Q8 path; the
   custom 65K tokenizer loads via existing `ruvllm::tokenizer`
   (`tokenizer.json`). FlashAttention 3 is a Hopper kernel optimization, not
   a semantic dependency — ruvllm's FA2 computes the same function.

### Decision matrix

| Option | Value | Risk | Take |
|--------|-------|------|------|
| vLLM endpoint from Rust | Fastest path, model-card-documented | Python dependency in the loop | **Do first (Phase A)** |
| SGLang endpoint from Rust | Good serving path | More moving parts | Good second / fallback |
| Direct Rust-native inference | High control, edge story, no Python | High effort (recurrence KV, PrefixLM mask, loader) | Later (Phase D, gated) |
| Convert to GGUF for llama.cpp | Great edge story | HRM recurrence + PrefixLM almost certainly unsupported upstream | Investigate only; native ruvllm path supersedes |
| Fine-tune for ruvLLM kernel tasks | High value (plan/verify/route formats) | Dataset work | After baseline proves value |

### Explicit non-goals

- Training in Rust (FSDP2, bp-warmup gradient windowing, adam_atan2,
  multipack/LPT) — stays upstream.
- Chat/assistant duty for HRM-Text — pre-alignment checkpoint; it is a
  kernel, not a persona.
- Inventing ACT/halting — upstream has fixed deterministic cycles.

## Acceptance tests (gate between phases)

Run 100 controlled tasks per category through the Phase A adapter before any
Phase D investment. Judge on subtasks, never freeform chat:

| Test | Pass target |
|------|-------------|
| JSON extraction validity (`Extract` mode, serde-valid) | ≥ 95 % |
| Plan usefulness (judged by stronger model) | ≥ 80 % |
| Contradiction detection (`Verify` mode) | ≥ 85 % |
| Routing accuracy vs labeled intents | ≥ 90 % |
| Latency vs local 7B (Qwen2.5-7B Q4K) on same hardware | ≥ 2× faster |

Additional Phase A gate: PrefixLM parity check (vLLM vs upstream reference
engine, 32 prompts, greedy) to rule out silent causal-only serving.

Phase D gates (if reached): logits within 1e-3 of upstream reference;
PrefixLM mask property tests (bidirectional block ≡ restricted full
attention; causal region bit-identical to existing path; decode-step ≡
full-sequence forward, extending the ADR-186 test pattern); KV footprint at
4K/8K across {FP16, two-tier, TurboQuant-3bit, recompute} on M4 Pro and
Pi 5; ≥ 95 % of upstream GSM8k subset score at Q4_K via the existing
evaluation harness.

## Research frontier — beyond-SOTA tracks

The integration above is the governed baseline. Five tracks could push past
state of the art on specific axes. None claims absolute frontier-model
quality — a 1B pre-alignment checkpoint will not beat frontier models on
open-ended tasks. The claims are Pareto claims: quality per parameter, per
watt, per dollar, per millisecond. Each track has a kill criterion so we
measure before we believe.

> **Calibration note.** The ARC Prize team's ablation of the original HRM
> found much of its gain came from the outer refinement loop and training
> procedure rather than the H/L hierarchy itself, and HRM-Text's 130–600×
> efficiency numbers are vendor claims on a young repo. Tracks R1–R3 are
> bets on the recurrence being real; R4–R5 pay off regardless.

### R1 — Test-time compute scaling on the cycle knob

*Hypothesis:* recurrence depth (`H_cycles × L_cycles`) works as a
per-request "think longer" dial — o1-style test-time compute scaling at
zero extra parameters and zero extra weight memory. Upstream ships fixed
cycles; no production runtime exposes recurrence depth as an inference
parameter.

*Experiment (first entry in `benchmark.rs`, runnable in Phase A):* sweep
cycles {1×1, 1×2, 2×2, 3×3, 4×4} on fixed GSM8k/MATH subsets (200 problems,
greedy, `SynthCot`), plot accuracy vs FLOPs. Requires a serving path that
honors a cycle override; if vLLM's integration hardcodes cycles, this
becomes the first native-path (Phase D) experiment instead.

*Beyond-SOTA form:* router schedules depth per task — hard tasks 4×4, easy
extraction 1×2. Follow-up: learned halting policy (original HRM had ACT;
HRM-Text dropped it) — a genuine research contribution if stable.

*Kill criterion:* accuracy flat or degrading beyond the training-time cycle
count (recurrent models often destabilize at out-of-distribution depth).
If flat, R1 and R2 die; fall back to R4/R5.

### R2 — Self-speculative decoding via cycle asymmetry

*Hypothesis:* a recurrent model can draft with itself at low cycle count
and verify at full cycle count — same weights, same tokenizer, no separate
draft model. No other runtime can offer this because no other runtime
serves a cycle-parameterized model. If acceptance rates hold, ~2–3× decode
speedup at zero extra memory, stacking on existing `ruvllm::speculative`.

*Experiment (Phase D):* measure token acceptance rate of 1×1-cycle drafts
against 2×2/4×4-cycle verification on kernel-task generation.

*Kill criterion:* acceptance < ~60 % (below the break-even of the existing
draft-model path with RuvLTRA-Small).

### R3 — Latent reasoning memory: persisting z_H into ruvector

*Hypothesis:* the H-level state is an explicit compressed "what I've
figured out" tensor. Snapshot z_H into ruvector keyed by task; warm-start
the recurrence from a retrieved state instead of the learned init buffer —
cross-session *latent* memory, closer to continuous-latent reasoning
(Coconut-style) than to text RAG. Only attemptable here because ruvector
and the inference engine share a process (Phase D).

*Experiment:* multi-turn task suites, warm-start vs cold-start, measure
quality and tokens-to-solution. Most speculative track — states may not
transfer across prompts at all.

*Kill criterion:* warm-start ≤ cold-start quality, or any cross-prompt
state contamination in verification suites.

### R4 — Edge agentic Pareto frontier (engineering, not research risk)

*Claim:* best agentic reasoning per watt / per dollar, fully offline —
the complete plan→route→execute→verify loop under 2 GB on Pi 5 +
Hailo-10H (ADR-173/179), via Q4_K 1B (~0.7 GB) + sparse attention
(ADR-183/189) + TurboQuant KV (ADR-181). Nobody publishes numbers for a
governed agent loop at this footprint; we define and own the benchmark:
GSM8k-class quality per watt, end-to-end loop latency, reproducible on
~$150 hardware.

*Kill criterion:* none — this works regardless of R1–R3; gated only on the
Phase A acceptance tests.

### R5 — Verifier-amplified small-model quality (cheapest jump)

*Hypothesis:* best-of-N with HRM-as-scorer plus the Verify-retry loop, at
1B-local prices (N=8 costs less than one 7B call), pushes GSM8k from the
published 84.7 % toward the high 80s/low 90s — above anything in the
≤1.5B class (Qwen2.5-1.5B ≈ 70 %). Combine with the ~$1.5K kernel
fine-tune on plan/verify/route/extract formats from the decision matrix.

*Experiment (Phase A):* GSM8k subset, N ∈ {1, 4, 8}, HRM `Verify`-mode
scoring vs majority vote vs single-shot.

*Caveat / kill criterion:* self-verification has known ceilings — a model
is weakest at its own systematic errors (the controller mitigates this by
having HRM verify the *execution model's* output, not its own). Kill if
best-of-8 gains < 2 points over majority vote.

### Sequencing

R1 and R5 run in days behind the Phase A vLLM adapter and are
prompt/sampling-level. Their results determine whether the Phase D native
port is scoped for cycle-parameterized serving and self-speculation
(R1→R2, R3) or only for the edge Pareto play (R4), which is justified
either way.

## Consequences

### Positive

- A governed plan→route→execute→verify loop with local 1B-latency reasoning,
  usable today via vLLM — no porting risk on the critical path.
- The `LlmBackend` trait isolates transport: native Phase D slots in without
  touching router/verifier/controller code.
- Opens "pretrain custom 1B for ~$1.5K in PyTorch → serve in ruvllm" for
  RuvLTRA-class custom kernels; fine-tuning on plan/verify/route formats is a
  natural follow-up.
- Phase D byproducts (PrefixLM mask mode, safetensors loader) are
  independently useful for other PrefixLM/UL2-style checkpoints.

### Negative / costs

- Phase A puts Python (vLLM) inside an otherwise-Rust runtime — acceptable
  scaffolding, but it weakens the edge story until Phase D and must not
  become load-bearing permanently.
- Pre-alignment model: every consumer must go through `HrmMode` templates;
  raw prompting will produce garbage and erode trust in the kernel.
- `token_type_ids` PrefixLM handling through OpenAI-compatible APIs is the
  top silent-failure risk; mitigated by the parity gate.
- Native path carries ~3× KV memory for the recurrence (mitigable), decode
  latency scaling with cycle count, and a second checkpoint format to
  maintain. Upstream is young — pin a verified commit/release of
  `convert_to_hf.py`.

### Licensing

HRM-Text code and the 1B checkpoint are Apache-2.0; ruvllm is
Apache-2.0/MIT. Serving the weights, reimplementing the architecture in
Rust, and fine-tuning are all compatible. Cite upstream BibTeX in module
docs.

## Alternatives considered

1. **Treat HRM-Text as a chat model behind the standard backend.** Rejected:
   pre-alignment checkpoint; chat behavior is uneven by the vendor's own
   card. Judging it there kills a good kernel for the wrong job.
2. **Bind PyTorch via `tch-rs`.** Rejected: ~2 GB libtorch, kills
   WASM/ANE/Hailo targets; the vLLM sidecar achieves the same speed with
   less coupling.
3. **Skip the endpoint phase, go straight to native Rust.** Rejected: weeks
   of effort before knowing whether HRM-Text clears the acceptance bar as
   planner/verifier/router. Endpoint-first converts that to days.
4. **Distill HRM-Text into a standard RuvLTRA architecture.** Deferred —
   forfeits the bounded-params/unbounded-depth advantage but needs zero
   runtime changes; fallback if Phase D KV costs prove prohibitive on the
   smallest targets.

## References

- Model card: https://huggingface.co/sapientinc/HRM-Text-1B (Apache-2.0;
  pre-alignment, PrefixLM `token_type_ids` requirement, condition tags
  `direct`/`cot`/`noisy`/`synth`, vLLM/SGLang serving, 65,536-token vocab)
- Repository: https://github.com/sapientinc/HRM-Text —
  `models/transformer.py`, `models/baselines/hrm_nocarry_bp_warmup.py`,
  `models/flash_attention_prefixlm_v2.py`, `conversion/convert_to_hf.py`,
  `simple_inference_engine.py`
- Wang et al., *Hierarchical Reasoning Model*, arXiv:2506.21734
- ADR-002 (ruvllm integration), ADR-180 (continuous batching serving),
  ADR-181 (pi-quant/BitNet quantization), ADR-183–190 (sparse attention
  kernel family; ADR-189 KV-cache incremental decode)
- `crates/ruvllm/src/backends/phi3.rs`, `gemma2.rs` — per-architecture
  module pattern for the Phase D native backend
