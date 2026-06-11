//! # ruvllm-hrm — HRM-Text reasoning-kernel adapter (ADR-199)
//!
//! Integrates [`sapientinc/HRM-Text-1B`](https://huggingface.co/sapientinc/HRM-Text-1B)
//! into the ruvllm governed runtime as a **reasoning kernel**, not a chat model.
//! HRM-Text is a dual-timescale hierarchical recurrent model (slow H-level +
//! fast L-level Transformer stacks, `H_cycles x L_cycles` nested recurrence).
//! The released checkpoint is **pre-alignment**: PrefixLM-pretrained with
//! condition prefix tokens (`direct`, `cot`, `noisy`, `synth`), not
//! instruction-tuned. Raw prompting will produce garbage — every consumer must
//! go through the [`prompt::HrmMode`] templates owned by this crate.
//!
//! Phases implemented here (see `docs/adr/ADR-199-ruvllm-hrm-text-integration.md`):
//!
//! - **Phase A** — [`backend::HrmTextBackend`]: Rust adapter behind a local
//!   vLLM/SGLang OpenAI-compatible endpoint (`vllm serve sapientinc/HRM-Text-1B`).
//!   The [`backend::LlmBackend`] trait isolates transport so the native Rust
//!   path (Phase D) slots in without touching callers.
//! - **Phase B** — [`prompt`]: PrefixLM-aware prompting. Condition-prefix
//!   templates, per-mode few-shot banks (model card recommends 2–8 shots for
//!   `direct`), stop sequences, and output parsers ([`parse`]).
//! - **Phase C** — [`controller::Controller`]: the governed loop
//!   `retrieve -> plan -> route -> execute -> verify`, with [`router::HrmRouter`]
//!   and [`verifier::HrmVerifier`].
//! - **Acceptance harness** — [`benchmark`]: the ADR-199 acceptance suite
//!   (JSON extraction / plan usefulness / contradiction detection / routing
//!   accuracy / latency), the R1 cycle sweep, and the R5 best-of-N experiment.
//!   All runnable offline against [`mock::MockBackend`].
//!
//! ## Known risk (from ADR-199)
//!
//! The model card requires `token_type_ids = ones` so prompt tokens attend
//! bidirectionally (PrefixLM). If a generic completions path silently runs
//! causal-only, quality degrades invisibly. Phase A includes a parity check
//! against the upstream reference engine before anything builds on top; this
//! crate exposes [`backend::PrefixMode`] so callers can record which serving
//! mode was verified.
//!
//! ## Upstream citation
//!
//! Wang et al., *Hierarchical Reasoning Model*, arXiv:2506.21734.
//! HRM-Text code and the 1B checkpoint are Apache-2.0.

pub mod backend;
pub mod benchmark;
mod benchmark_tasks;
pub mod controller;
pub mod edge;
pub mod mock;
pub mod parse;
pub mod prompt;
pub mod router;
pub mod scoring;
pub mod verifier;

pub use backend::{
    BackendError, GenerateRequest, GenerateResponse, HrmCycles, HrmTextBackend, LlmBackend,
    PrefixMode, ScoreRequest, ScoreResponse, StateRequest, StateResponse,
};
pub use benchmark::{
    run_acceptance, run_best_of_n, run_cycle_sweep, AcceptanceReport, AcceptanceSuite,
    AcceptanceTask, BestOfNReport, CategoryReport, CycleSweepReport, Expected, ReasoningTask,
    TaskCategory, DEFAULT_CYCLE_GRID,
};
pub use controller::{Controller, ControllerResult, RetrievalFn};
pub use edge::pareto::{pareto_csv, pareto_frontier, ParetoPoint, PARETO_CSV_HEADER};
pub use edge::{
    default_edge_tasks, under_memory_gate, EdgeCheck, EdgeLoopBenchmark, EdgeLoopReport,
    EdgePowerProfile, EdgeTask, EdgeTaskRecord, MemorySampler, EDGE_MEMORY_GATE_GB,
};
pub use mock::MockBackend;
pub use parse::{extract_json_with_retry, parse_strict_json, parse_verdict};
pub use prompt::{build_hrm_prompt, FewShot, GenerateParams, HrmMode, PromptBuilder};
pub use router::{choose_cycles, ComputeClass, HrmRouter, Route, RoutingDecision};
pub use scoring::{agent_score, edge_score, AgentScoreInputs};
pub use verifier::{HrmVerifier, Verdict};
