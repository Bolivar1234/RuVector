# ruvllm-hrm

HRM-Text reasoning-kernel adapter for the ruvllm governed runtime
(**ADR-199**, Phases A–C). HRM-Text-1B is a hierarchical recurrent,
PrefixLM-pretrained, **pre-alignment** checkpoint — a kernel, not a chat
model. This crate owns the prompt contract and wires the kernel into the
governed loop `retrieve → plan → route → execute → verify`.

## What's inside

| Module | Role |
|--------|------|
| `backend` | `LlmBackend` trait + `HrmTextBackend` (vLLM/SGLang OpenAI-compatible `/v1/completions` client; retries; `h_cycles`/`l_cycles` R1 hook; `embed_state` is Phase-D-only) |
| `mock` | Deterministic, programmable `MockBackend` for offline tests/benchmarks |
| `prompt` | `HrmMode` condition-prefix templates (verbatim from ADR-199), few-shot banks, stop sequences, recommended params |
| `parse` | Tolerant `Verdict` parser, strict-JSON parser with one corrective retry |
| `router` | Typed `RoutingDecision` (local model / tool / vector search / workflow / remote) with keyword fallback |
| `verifier` | `Verdict { pass, contradictions, missing_facts }` Verify-mode pass |
| `controller` | Governed loop with verify-fail retry and a ruvector retrieval hook |
| `benchmark` | ADR-199 acceptance suite (5 categories, targets 0.95/0.80/0.85/0.90/2x), R1 cycle sweep, R5 best-of-N |

## Running against vLLM

```bash
pip install vllm
vllm serve sapientinc/HRM-Text-1B   # http://localhost:8000/v1

HRM_ENDPOINT=http://localhost:8000/v1 cargo run -p ruvllm-hrm --example plan
HRM_ENDPOINT=http://localhost:8000/v1 cargo run -p ruvllm-hrm --example verify
HRM_ENDPOINT=http://localhost:8000/v1 cargo run -p ruvllm-hrm --example extract_json
```

`HRM_MODEL` overrides the model name (default `sapientinc/HRM-Text-1B`).

## Running offline (no server)

Every example takes `--mock`:

```bash
cargo run -p ruvllm-hrm --example plan -- --mock
cargo run -p ruvllm-hrm --example verify -- --mock
cargo run -p ruvllm-hrm --example extract_json -- --mock
```

The acceptance harness, R1 cycle sweep, and R5 best-of-N run end-to-end on
`MockBackend` in `cargo test -p ruvllm-hrm`.

## Caveats (from the model card / ADR-199)

- Pre-alignment checkpoint: always go through `HrmMode` templates; raw
  prompting produces garbage.
- PrefixLM requires `token_type_ids = ones`; verify your serving path with
  the ADR-199 parity gate before trusting quality (`PrefixMode` records the
  outcome).
- The `h_cycles`/`l_cycles` request fields are best-effort: servers that
  hardcode recurrence depth ignore them (the R1 sweep then shows a flat line).
- Offline plan-usefulness judging is a heuristic; the production gate uses a
  stronger judge model.
