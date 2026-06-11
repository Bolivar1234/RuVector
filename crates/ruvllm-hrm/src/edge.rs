//! ADR-199 R4 — Edge agentic Pareto frontier benchmark (the rank-1 bet).
//!
//! Drives the **full governed loop** (`plan -> route -> execute -> verify`,
//! via [`crate::controller::Controller`]) over an end-to-end task set under
//! an operator-declared watt budget and a measured (or supplied) memory
//! footprint, then emits the data behind the ADR's "one chart that gates any
//! announcement": task quality vs latency/watts, with [`crate::scoring::edge_score`]
//! and [`crate::scoring::agent_score`] always published together — never
//! accuracy alone.
//!
//! Product gate (ADR-199 research-vs-product table): full loop **< 2 GB**,
//! within a defined watt budget, with stable execution. There is no research
//! kill criterion — this benchmark pays off regardless of R1–R3.
//!
//! Offline the benchmark runs against [`crate::mock::MockBackend`]; pointing
//! the same harness at live backends measures the real model (see
//! `examples/edge_loop.rs`).

pub mod budget;
pub mod pareto;

pub use budget::{EdgePowerProfile, MemorySampler};

use crate::backend::LlmBackend;
use crate::benchmark::Expected;
use crate::benchmark_tasks as tasks;
use crate::controller::Controller;
use crate::router::Route;
use crate::scoring::{agent_score, edge_score, AgentScoreInputs};
use pareto::ParetoPoint;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// ADR-199 R4 product gate: the whole governed loop must fit under 2 GB.
pub const EDGE_MEMORY_GATE_GB: f64 = 2.0;

/// Is `peak_memory_gb` strictly below the 2 GB product gate? (`< 2 GB` per
/// the ADR — exactly 2.0 GB fails.)
pub fn under_memory_gate(peak_memory_gb: f64) -> bool {
    peak_memory_gb < EDGE_MEMORY_GATE_GB
}

/// What the governed loop must get right for one edge task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeCheck {
    /// Router must pick this destination (route correctness applicable).
    Route(Route),
    /// Final answer must be serde-valid JSON equal to this value
    /// (JSON validity applicable).
    Json(serde_json::Value),
    /// Only the verifier verdict gates the task (plan-style goals).
    VerdictOnly,
}

impl EdgeCheck {
    fn kind(&self) -> &'static str {
        match self {
            EdgeCheck::Route(_) => "routing",
            EdgeCheck::Json(_) => "json_extraction",
            EdgeCheck::VerdictOnly => "verdict_only",
        }
    }
}

/// One end-to-end task: an input for the full governed loop paired with the
/// expected outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeTask {
    pub input: String,
    pub check: EdgeCheck,
}

/// Default 20-task end-to-end set, composed from the acceptance-harness
/// categories in `benchmark_tasks.rs` (no duplicated data): 8 routing
/// intents, 6 JSON extractions, 6 planning goals.
///
/// JSON inputs embed the strict-JSON instruction in the request itself
/// (`Return strict JSON only.\n…\nJSON:`) so any instruction-following
/// executor — and the offline mock — emits a machine-checkable answer.
pub fn default_edge_tasks() -> Vec<EdgeTask> {
    let mut out = Vec::with_capacity(20);
    for t in tasks::routing_tasks().into_iter().take(8) {
        if let Expected::Route(route) = t.expected {
            out.push(EdgeTask { input: t.input, check: EdgeCheck::Route(route) });
        }
    }
    for t in tasks::json_extraction_tasks().into_iter().take(6) {
        if let Expected::Json(v) = t.expected {
            out.push(EdgeTask {
                input: format!("Return strict JSON only.\n{}\nJSON:", t.input),
                check: EdgeCheck::Json(v),
            });
        }
    }
    for t in tasks::plan_tasks().into_iter().take(6) {
        out.push(EdgeTask { input: t.input, check: EdgeCheck::VerdictOnly });
    }
    out
}

/// Per-task outcome of one governed-loop run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeTaskRecord {
    pub input: String,
    /// `routing` | `json_extraction` | `verdict_only`.
    pub kind: String,
    /// End-to-end wall latency of the full loop, in milliseconds.
    pub latency_ms: f64,
    /// Verifier verdict on the final answer.
    pub verdict_pass: bool,
    /// Router picked the expected destination (`None` when not applicable).
    pub route_correct: Option<bool>,
    /// Answer was serde-valid JSON equal to the expected value (`None` when
    /// not applicable).
    pub json_valid: Option<bool>,
    /// Execution attempts the controller needed (1 = no verify-fail retry).
    pub attempts: usize,
    /// Task fully passed the loop: verdict pass AND every applicable check.
    pub passed: bool,
    /// Loop error, if the run failed outright (makes the report unstable).
    pub error: Option<String>,
}

/// Full R4 report — quality and cost together, plus the product-gate fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeLoopReport {
    pub power_profile: EdgePowerProfile,
    pub records: Vec<EdgeTaskRecord>,
    pub total_tasks: usize,
    /// Fraction of tasks fully passing the loop, in `[0, 1]`.
    pub quality: f64,
    /// Effective peak memory in GB: measured RSS, unless the operator
    /// supplied an override (which wins — it represents the real model
    /// server's footprint when inference runs out-of-process).
    pub peak_memory_gb: f64,
    /// Peak RSS actually sampled from this process, when available.
    pub sampled_peak_rss_gb: Option<f64>,
    /// `peak_memory_gb` came from the operator override, not the sampler.
    pub memory_overridden: bool,
    pub latency_mean_ms: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    /// Per-task budget (ms) behind the `latency_score` agent-score proxy.
    pub latency_budget_ms: u64,
    /// `quality / (mean_latency_s × watts × peak_memory_gb)`.
    pub edge_score: f64,
    /// ADR-199 governance composite with category proxies from this run.
    pub agent_score: f64,
    /// Product gate: `peak_memory_gb` strictly under 2 GB.
    pub under_2gb: bool,
    /// Product gate: no task errored.
    pub stable: bool,
}

impl EdgeLoopReport {
    /// This run as one point on the announcement-gating Pareto chart.
    pub fn pareto_point(&self, label: impl Into<String>) -> ParetoPoint {
        ParetoPoint {
            label: label.into(),
            latency_ms: self.latency_mean_ms,
            watts: self.power_profile.watts,
            quality: self.quality,
            memory_gb: self.peak_memory_gb,
            edge_score: self.edge_score,
        }
    }
}

/// The R4 benchmark runner. See module docs.
pub struct EdgeLoopBenchmark {
    kernel: Arc<dyn LlmBackend>,
    executor: Arc<dyn LlmBackend>,
    power: EdgePowerProfile,
    memory_gb_override: Option<f64>,
    tasks: Vec<EdgeTask>,
    latency_budget_ms: u64,
}

impl EdgeLoopBenchmark {
    /// Benchmark over [`default_edge_tasks`] with the acceptance harness's
    /// 1000 ms per-task latency budget (the offline 2x-vs-7B proxy).
    pub fn new(
        kernel: Arc<dyn LlmBackend>,
        executor: Arc<dyn LlmBackend>,
        power: EdgePowerProfile,
    ) -> Self {
        Self {
            kernel,
            executor,
            power,
            memory_gb_override: None,
            tasks: default_edge_tasks(),
            latency_budget_ms: 1000,
        }
    }

    /// Replace the task set.
    pub fn with_tasks(mut self, tasks: Vec<EdgeTask>) -> Self {
        self.tasks = tasks;
        self
    }

    /// Operator-supplied peak memory in GB. Required off-Linux (no RSS
    /// sampling); takes precedence over the sampler when the model serves
    /// from a separate process whose footprint this process cannot see.
    pub fn with_memory_gb_override(mut self, memory_gb: f64) -> Self {
        self.memory_gb_override = Some(memory_gb);
        self
    }

    /// Override the per-task latency budget behind the `latency_score` proxy.
    pub fn with_latency_budget_ms(mut self, budget_ms: u64) -> Self {
        self.latency_budget_ms = budget_ms;
        self
    }

    /// Run every task through the full governed loop and aggregate the report.
    pub async fn run(&self) -> anyhow::Result<EdgeLoopReport> {
        if self.tasks.is_empty() {
            anyhow::bail!("edge-loop benchmark has no tasks");
        }
        if let Some(gb) = self.memory_gb_override {
            if !gb.is_finite() || gb <= 0.0 {
                anyhow::bail!("memory_gb_override must be finite and positive, got {gb}");
            }
        }
        let controller = Controller::new(self.kernel.clone(), self.executor.clone());
        let mut sampler = MemorySampler::new();
        sampler.sample();

        let mut records = Vec::with_capacity(self.tasks.len());
        for task in &self.tasks {
            let start = Instant::now();
            let outcome = controller.run(&task.input).await;
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            sampler.sample();
            records.push(match outcome {
                Ok(r) => {
                    let route_correct = match &task.check {
                        EdgeCheck::Route(expected) => Some(&r.route.route == expected),
                        _ => None,
                    };
                    let json_valid = match &task.check {
                        EdgeCheck::Json(expected) => Some(
                            serde_json::from_str::<serde_json::Value>(r.answer.trim())
                                .map(|v| &v == expected)
                                .unwrap_or(false),
                        ),
                        _ => None,
                    };
                    let passed = r.verdict.pass
                        && route_correct.unwrap_or(true)
                        && json_valid.unwrap_or(true);
                    EdgeTaskRecord {
                        input: task.input.clone(),
                        kind: task.check.kind().to_string(),
                        latency_ms,
                        verdict_pass: r.verdict.pass,
                        route_correct,
                        json_valid,
                        attempts: r.attempts,
                        passed,
                        error: None,
                    }
                }
                Err(e) => EdgeTaskRecord {
                    input: task.input.clone(),
                    kind: task.check.kind().to_string(),
                    latency_ms,
                    verdict_pass: false,
                    route_correct: None,
                    json_valid: None,
                    attempts: 0,
                    passed: false,
                    error: Some(e.to_string()),
                },
            });
        }
        self.aggregate(records, sampler.peak_gb())
    }

    fn aggregate(
        &self,
        records: Vec<EdgeTaskRecord>,
        sampled_peak_rss_gb: Option<f64>,
    ) -> anyhow::Result<EdgeLoopReport> {
        let total = records.len();
        let frac = |n: usize| n as f64 / total as f64;
        let quality = frac(records.iter().filter(|r| r.passed).count());
        let stable = records.iter().all(|r| r.error.is_none());

        let mut sorted: Vec<f64> = records.iter().map(|r| r.latency_ms).collect();
        sorted.sort_by(f64::total_cmp);
        let latency_mean_ms = sorted.iter().sum::<f64>() / total as f64;
        let latency_p50_ms = percentile(&sorted, 50.0);
        let latency_p95_ms = percentile(&sorted, 95.0);

        let (peak_memory_gb, memory_overridden) =
            match (self.memory_gb_override, sampled_peak_rss_gb) {
                (Some(gb), _) => (gb, true),
                (None, Some(gb)) => (gb, false),
                (None, None) => anyhow::bail!(
                    "RSS sampling is unavailable on this platform; supply the model \
                     footprint via EdgeLoopBenchmark::with_memory_gb_override(..)"
                ),
            };

        // Floor at 1 µs so a pathological clock cannot zero the cost term.
        let mean_latency_s = (latency_mean_ms / 1000.0).max(1e-6);
        let edge_score = edge_score(quality, mean_latency_s, self.power.watts, peak_memory_gb)?;

        let json_applicable = records.iter().filter(|r| r.json_valid.is_some()).count();
        let json_validity = if json_applicable == 0 {
            1.0 // neutral, mirroring AcceptanceSuite's unobserved-memory proxy
        } else {
            records.iter().filter(|r| r.json_valid == Some(true)).count() as f64
                / json_applicable as f64
        };
        let agent_score = agent_score(&AgentScoreInputs {
            task_success: quality,
            verifier_accuracy: frac(records.iter().filter(|r| r.verdict_pass).count()),
            json_validity,
            latency_score: frac(
                records
                    .iter()
                    .filter(|r| r.latency_ms <= self.latency_budget_ms as f64)
                    .count(),
            ),
            // Headroom against the 2 GB product gate, clamped to [0, 1].
            memory_score: (1.0 - peak_memory_gb / EDGE_MEMORY_GATE_GB).clamp(0.0, 1.0),
        });

        Ok(EdgeLoopReport {
            power_profile: self.power.clone(),
            total_tasks: total,
            records,
            quality,
            peak_memory_gb,
            sampled_peak_rss_gb,
            memory_overridden,
            latency_mean_ms,
            latency_p50_ms,
            latency_p95_ms,
            latency_budget_ms: self.latency_budget_ms,
            edge_score,
            agent_score,
            under_2gb: under_memory_gate(peak_memory_gb),
            stable,
        })
    }
}

/// Nearest-rank percentile over an ascending-sorted, non-empty slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_gate_2gb_boundary() {
        assert!(under_memory_gate(1.999_999));
        assert!(!under_memory_gate(2.0), "gate is strict: exactly 2 GB fails");
        assert!(!under_memory_gate(2.000_001));
    }

    #[test]
    fn default_task_set_is_20_across_three_kinds() {
        let tasks = default_edge_tasks();
        assert_eq!(tasks.len(), 20);
        let count = |kind: &str| tasks.iter().filter(|t| t.check.kind() == kind).count();
        assert_eq!(count("routing"), 8);
        assert_eq!(count("json_extraction"), 6);
        assert_eq!(count("verdict_only"), 6);
    }

    #[test]
    fn percentile_nearest_rank() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&sorted, 50.0), 5.0);
        assert_eq!(percentile(&sorted, 95.0), 10.0);
        assert_eq!(percentile(&sorted, 100.0), 10.0);
        assert_eq!(percentile(&[42.0], 50.0), 42.0);
        assert_eq!(percentile(&[42.0], 95.0), 42.0);
    }
}
