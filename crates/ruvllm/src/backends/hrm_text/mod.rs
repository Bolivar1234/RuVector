//! HRM-Text Hierarchical Recurrent Transformer Backend (ADR-199 Phase D)
//!
//! Native Rust CPU inference for the `sapientinc/HRM-Text` architecture: a
//! dual-timescale hierarchical recurrent model with two Transformer cores
//! (slow H-level, fast L-level) iterated `h_cycles x l_cycles` times with
//! additive state injection — "effectively unbounded compute depth at bounded
//! parameter count".
//!
//! ## What this module provides
//!
//! - [`HrmTextConfig`] — architecture hyperparameters, with `hrm_text_l()` /
//!   `hrm_text_xl()` reference configs and `tiny_test()` for tests.
//! - [`AttentionMaskMode`] — `Causal` and `PrefixLm { prefix_len }` masking
//!   (`allowed(q, k) = k < prefix_len || k <= q`); HRM-Text is
//!   PrefixLM-pretrained and requires the bidirectional prefix to match
//!   training (the `token_type_ids = ones` requirement on the model card).
//! - [`HrmKvCaches`] — per-(level, cycle, layer) KV caches:
//!   `(h_cycles * l_cycles + h_cycles) * half_layers` slots, ~3x the KV
//!   memory of an equal-depth standard model (the ADR's headline number,
//!   measurable via `memory_bytes` / `projected_memory_bytes`).
//! - [`HrmTextModel`] — full forward / prefill / `decode_step` /
//!   `generate_greedy`, plus `cycle_override` (R1 test-time-compute hook).
//!
//! Pure CPU f32 with no candle dependency: compiles under default features
//! and under `--no-default-features --features minimal`.
//!
//! ## Non-goals (per ADR-199)
//!
//! Training, ACT/halting (upstream has fixed deterministic cycles), and
//! chat/persona duty (the released checkpoint is pre-alignment; it is a
//! reasoning kernel, not an endpoint). Checkpoint loading (safetensors/GGUF)
//! is ADR-199 Phase D item 4 and is not implemented here.
//!
//! ## Upstream reference
//!
//! - Model: <https://huggingface.co/sapientinc/HRM-Text-1B> (Apache-2.0)
//! - Code: <https://github.com/sapientinc/HRM-Text>
//! - Wang et al., *Hierarchical Reasoning Model*, arXiv:2506.21734
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::backends::hrm_text::{HrmKvCaches, HrmTextConfig, HrmTextModel};
//!
//! let config = HrmTextConfig::tiny_test();
//! let model = HrmTextModel::new_random(config.clone(), 42)?;
//! let tokens = model.generate_greedy(&[3, 5, 7, 11], 4, 16)?;
//! ```

mod cache;
mod config;
mod core;
mod generate;
mod mask;
mod tensor;

pub use cache::{HrmKvCaches, KvLayerCache};
pub use config::{HrmTextConfig, NormPosition};
pub use generate::HrmTextModel;
pub use mask::AttentionMaskMode;
pub use self::core::{HrmDecoderLayer, HrmRecurrentCore};
pub use tensor::{Embedding, Linear, RmsNorm};
