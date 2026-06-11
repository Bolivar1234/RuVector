//! ADR-199 acceptance-test harness + research-track experiments.
//!
//! - [`run_acceptance`]: the gate between Phase A and Phase D — JSON
//!   extraction validity (>= 0.95), plan usefulness (>= 0.80), contradiction
//!   detection (>= 0.85), routing accuracy (>= 0.90), latency (>= 2x faster
//!   than a local 7B; offline this is a wall-clock budget proxy).
//! - [`run_cycle_sweep`] (R1): per-request recurrence depth as a test-time
//!   compute dial — accuracy + mean latency per `(H_cycles, L_cycles)`.
//! - [`run_best_of_n`] (R5): verifier-amplified small-model quality —
//!   best-of-N with kernel scoring vs majority vote vs single-shot.
//!
//! Everything runs offline against [`crate::mock::MockBackend`]; pointing
//! the same harness at a live [`crate::backend::HrmTextBackend`] measures
//! the real model. Plan usefulness is judged by a heuristic offline; the
//! production gate uses a stronger judge model.

use crate::backend::{HrmCycles, LlmBackend, ScoreRequest};
use crate::scoring::{agent_score, AgentScoreInputs};
use crate::benchmark_tasks as tasks;
use crate::parse::{extract_final_int, extract_json_with_retry};
use crate::prompt::{HrmMode, PromptBuilder};
use crate::router::{HrmRouter, Route};
use crate::verifier::HrmVerifier;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Acceptance task categories (ADR-199 acceptance table).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskCategory {
    JsonExtraction,
    PlanUsefulness,
    ContradictionDetection,
    RoutingAccuracy,
    Latency,
}

impl TaskCategory {
    pub const ALL: [TaskCategory; 5] = [
        TaskCategory::JsonExtraction,
        TaskCategory::PlanUsefulness,
        TaskCategory::ContradictionDetection,
        TaskCategory::RoutingAccuracy,
        TaskCategory::Latency,
    ];

    /// Pass-rate target from the ADR-199 acceptance table. For `Latency`
    /// the underlying target is "2x faster than a local 7B"; each task
    /// passes if it fits the suite's latency budget, gated at 0.90.
    pub fn target(self) -> f64 {
        match self {
            TaskCategory::JsonExtraction => 0.95,
            TaskCategory::PlanUsefulness => 0.80,
            TaskCategory::ContradictionDetection => 0.85,
            TaskCategory::RoutingAccuracy => 0.90,
            TaskCategory::Latency => 0.90,
        }
    }
}

/// Expected outcome for one acceptance task.
#[derive(Debug, Clone)]
pub enum Expected {
    /// Extract mode must produce exactly this JSON value.
    Json(serde_json::Value),
    /// Verifier must reach this pass/fail result.
    Verdict { pass: bool },
    /// Router must pick this destination.
    Route(Route),
    /// Plan judged useful by the heuristic (or a judge model online).
    PlanHeuristic,
    /// No expected output (latency tasks).
    None,
}

/// One acceptance task.
#[derive(Debug, Clone)]
pub struct AcceptanceTask {
    pub category: TaskCategory,
    pub input: String,
    /// Evidence handed to the verifier (contradiction tasks).
    pub evidence: Vec<String>,
    pub expected: Expected,
}

/// One reasoning task for the R1 sweep / R5 best-of-N.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTask {
    pub problem: String,
    /// Expected final integer answer.
    pub expected: i64,
}

/// Per-category results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryReport {
    pub category: TaskCategory,
    pub total: usize,
    pub passed_tasks: usize,
    pub pass_rate: f64,
    pub target: f64,
    pub passed: bool,
    pub mean_latency_ms: f64,
}

/// Full acceptance report (serde-serializable for CI artifacts).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptanceReport {
    pub categories: Vec<CategoryReport>,
    pub total_tasks: usize,
    pub all_passed: bool,
    /// Latency budget used as the offline 2x-vs-7B proxy.
    pub latency_budget_ms: u64,
    /// ADR-199 composite governance score:
    ///
    /// ```text
    /// 0.40*task_success + 0.20*verifier_accuracy + 0.15*json_validity
    ///   + 0.15*latency_score + 0.10*memory_score
    /// ```
    ///
    /// Quality and cost are always published together — never accuracy alone.
    pub agent_score: f64,
}

/// Acceptance suite: tasks + the latency budget.
pub struct AcceptanceSuite {
    pub tasks: Vec<AcceptanceTask>,
    /// Per-task wall-clock budget for the latency category. Default 1000ms:
    /// half of a ~2000ms local Qwen2.5-7B Q4K reference generation, i.e. the
    /// ADR's ">= 2x faster" target expressed as an absolute proxy.
    pub latency_budget_ms: u64,
    /// Memory score in [0,1] for the `agent_score` composite. The HTTP
    /// harness cannot observe server RSS, so this defaults to the neutral
    /// 1.0 proxy; edge runs (ADR-199 R4) supply the measured value.
    pub memory_score: f64,
}

impl Default for AcceptanceSuite {
    fn default() -> Self {
        let mut all = Vec::with_capacity(100);
        all.extend(tasks::json_extraction_tasks());
        all.extend(tasks::plan_tasks());
        all.extend(tasks::contradiction_tasks());
        all.extend(tasks::routing_tasks());
        all.extend(tasks::latency_tasks());
        Self {
            tasks: all,
            latency_budget_ms: 1000,
            memory_score: 1.0,
        }
    }
}

impl AcceptanceSuite {
    /// Run every task against `backend` and aggregate per-category reports.
    pub async fn run(&self, backend: Arc<dyn LlmBackend>) -> anyhow::Result<AcceptanceReport> {
        if self.tasks.is_empty() {
            anyhow::bail!("acceptance suite has no tasks");
        }
        let extract_builder = PromptBuilder::new(HrmMode::Extract);
        let plan_builder = PromptBuilder::bare(HrmMode::Plan);
        let direct_builder = PromptBuilder::bare(HrmMode::Direct);
        let router = HrmRouter::new(backend.clone());
        let verifier = HrmVerifier::new(backend.clone());

        let mut stats: HashMap<TaskCategory, (usize, usize, f64)> = HashMap::new();
        for task in &self.tasks {
            let start = Instant::now();
            let ok = match (&task.category, &task.expected) {
                (TaskCategory::JsonExtraction, Expected::Json(expected)) => {
                    match extract_json_with_retry(backend.as_ref(), &extract_builder, &task.input)
                        .await
                    {
                        Ok(v) => &v == expected,
                        Err(_) => false,
                    }
                }
                (TaskCategory::PlanUsefulness, Expected::PlanHeuristic) => {
                    let resp = backend.generate(plan_builder.to_request(&task.input)).await?;
                    judge_plan_useful(&resp.text)
                }
                (TaskCategory::ContradictionDetection, Expected::Verdict { pass }) => {
                    let verdict = verifier.verify(&task.input, &[], &task.evidence).await?;
                    verdict.pass == *pass
                }
                (TaskCategory::RoutingAccuracy, Expected::Route(expected)) => {
                    let decision = router.route(&task.input).await?;
                    &decision.route == expected
                }
                (TaskCategory::Latency, Expected::None) => {
                    backend.generate(direct_builder.to_request(&task.input)).await?;
                    start.elapsed().as_millis() as u64 <= self.latency_budget_ms
                }
                (cat, exp) => {
                    anyhow::bail!("task category {cat:?} has incompatible expectation {exp:?}")
                }
            };
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let entry = stats.entry(task.category).or_insert((0, 0, 0.0));
            entry.0 += 1;
            entry.1 += usize::from(ok);
            entry.2 += elapsed;
        }

        let mut categories = Vec::new();
        for cat in TaskCategory::ALL {
            let Some(&(total, passed_tasks, latency_sum)) = stats.get(&cat) else {
                continue;
            };
            let pass_rate = passed_tasks as f64 / total as f64;
            let target = cat.target();
            categories.push(CategoryReport {
                category: cat,
                total,
                passed_tasks,
                pass_rate,
                target,
                passed: pass_rate >= target,
                mean_latency_ms: latency_sum / total as f64,
            });
        }
        let all_passed = categories.iter().all(|c| c.passed);
        let rate = |cat: TaskCategory| {
            categories
                .iter()
                .find(|c| c.category == cat)
                .map(|c| c.pass_rate)
                .unwrap_or(0.0)
        };
        // task_success = the end-to-end task categories (plan + routing);
        // the remaining components map 1:1 onto their categories.
        let agent_score = agent_score(&AgentScoreInputs {
            task_success: (rate(TaskCategory::PlanUsefulness)
                + rate(TaskCategory::RoutingAccuracy))
                / 2.0,
            verifier_accuracy: rate(TaskCategory::ContradictionDetection),
            json_validity: rate(TaskCategory::JsonExtraction),
            latency_score: rate(TaskCategory::Latency),
            memory_score: self.memory_score,
        });
        Ok(AcceptanceReport {
            total_tasks: self.tasks.len(),
            categories,
            all_passed,
            latency_budget_ms: self.latency_budget_ms,
            agent_score,
        })
    }
}

/// Offline plan-usefulness heuristic: at least two enumerated steps and
/// non-trivial content. The production gate replaces this with a stronger
/// judge model (ADR-199: "judged by stronger model").
fn judge_plan_useful(plan: &str) -> bool {
    let steps = plan
        .lines()
        .filter(|l| {
            let t = l.trim();
            t.starts_with('-')
                || t.starts_with('*')
                || t.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)
        })
        .count();
    steps >= 2 && plan.trim().len() >= 30
}

/// Run the default acceptance suite against `backend`.
pub async fn run_acceptance(backend: Arc<dyn LlmBackend>) -> anyhow::Result<AcceptanceReport> {
    AcceptanceSuite::default().run(backend).await
}

/// One entry of the R1 cycle sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleSweepEntry {
    pub cycles: HrmCycles,
    pub accuracy: f64,
    pub mean_latency_ms: f64,
    pub tasks: usize,
}

/// R1 report: accuracy vs recurrence depth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleSweepReport {
    pub entries: Vec<CycleSweepEntry>,
}

/// R1 — sweep recurrence depth ([`HrmCycles`] forwarded as
/// `h_cycles`/`l_cycles`; servers that hardcode cycles produce a flat line,
/// which is itself the kill-criterion signal).
pub async fn run_cycle_sweep(
    backend: Arc<dyn LlmBackend>,
    cycles: &[HrmCycles],
    tasks: &[ReasoningTask],
) -> anyhow::Result<CycleSweepReport> {
    if cycles.is_empty() || tasks.is_empty() {
        anyhow::bail!("cycle sweep needs at least one cycle pair and one task");
    }
    let builder = PromptBuilder::bare(HrmMode::SynthCot);
    let mut entries = Vec::with_capacity(cycles.len());
    for &grid_point in cycles {
        let mut correct = 0usize;
        let mut latency_sum = 0.0f64;
        for task in tasks {
            let mut req = builder.to_request(&task.problem);
            req.temperature = 0.0; // greedy, per the ADR experiment design
            req.hrm_cycles = Some(grid_point);
            let start = Instant::now();
            let resp = backend.generate(req).await?;
            latency_sum += start.elapsed().as_secs_f64() * 1000.0;
            if extract_final_int(&resp.text) == Some(task.expected) {
                correct += 1;
            }
        }
        entries.push(CycleSweepEntry {
            cycles: grid_point,
            accuracy: correct as f64 / tasks.len() as f64,
            mean_latency_ms: latency_sum / tasks.len() as f64,
            tasks: tasks.len(),
        });
    }
    Ok(CycleSweepReport { entries })
}

/// One entry of the R5 best-of-N comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestOfNEntry {
    pub n: usize,
    pub single_shot_accuracy: f64,
    pub majority_vote_accuracy: f64,
    pub kernel_scored_accuracy: f64,
    pub tasks: usize,
}

/// R5 report: best-of-N with kernel scoring vs majority vote vs single-shot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestOfNReport {
    pub entries: Vec<BestOfNEntry>,
}

/// R5 — verifier-amplified small-model quality. `executor` drafts N
/// candidates per task; `kernel` scores them ([`LlmBackend::score`]).
pub async fn run_best_of_n(
    kernel: Arc<dyn LlmBackend>,
    executor: Arc<dyn LlmBackend>,
    n_values: &[usize],
    tasks: &[ReasoningTask],
) -> anyhow::Result<BestOfNReport> {
    if n_values.is_empty() || n_values.contains(&0) || tasks.is_empty() {
        anyhow::bail!("best-of-N needs non-zero n values and at least one task");
    }
    let builder = PromptBuilder::bare(HrmMode::SynthCot);
    let mut entries = Vec::with_capacity(n_values.len());
    for &n in n_values {
        let (mut single, mut majority, mut scored) = (0usize, 0usize, 0usize);
        for task in tasks {
            let prompt = builder.build(&task.problem);
            let mut candidates = Vec::with_capacity(n);
            for _ in 0..n {
                let mut req = builder.to_request(&task.problem);
                // Greedy for N=1 (single-shot baseline), sampled otherwise.
                req.temperature = if n == 1 { 0.0 } else { 0.7 };
                candidates.push(executor.generate(req).await?.text);
            }
            let answers: Vec<Option<i64>> =
                candidates.iter().map(|c| extract_final_int(c)).collect();
            // Single-shot baseline.
            if answers[0] == Some(task.expected) {
                single += 1;
            }
            // Majority vote over normalized answers.
            if majority_answer(&answers) == Some(task.expected) {
                majority += 1;
            }
            // Kernel-scored best-of-N.
            let mut best: Option<(f64, usize)> = None;
            for (i, candidate) in candidates.iter().enumerate() {
                if candidate.trim().is_empty() {
                    continue;
                }
                let resp = kernel
                    .score(ScoreRequest {
                        prompt: prompt.clone(),
                        completion: candidate.clone(),
                        hrm_cycles: None,
                    })
                    .await?;
                if best.map(|(s, _)| resp.score > s).unwrap_or(true) {
                    best = Some((resp.score, i));
                }
            }
            if let Some((_, i)) = best {
                if answers[i] == Some(task.expected) {
                    scored += 1;
                }
            }
        }
        let total = tasks.len() as f64;
        entries.push(BestOfNEntry {
            n,
            single_shot_accuracy: single as f64 / total,
            majority_vote_accuracy: majority as f64 / total,
            kernel_scored_accuracy: scored as f64 / total,
            tasks: tasks.len(),
        });
    }
    Ok(BestOfNReport { entries })
}

/// Most common extracted answer (ties broken by first occurrence).
fn majority_answer(answers: &[Option<i64>]) -> Option<i64> {
    let mut counts: Vec<(i64, usize)> = Vec::new();
    for a in answers.iter().flatten() {
        match counts.iter_mut().find(|(v, _)| v == a) {
            Some((_, c)) => *c += 1,
            None => counts.push((*a, 1)),
        }
    }
    counts.into_iter().max_by_key(|&(_, c)| c).map(|(v, _)| v)
}

/// Built-in reasoning tasks (shared by the R1 sweep and R5 best-of-N).
pub fn default_reasoning_tasks() -> Vec<ReasoningTask> {
    tasks::reasoning_tasks()
}

/// The ADR-199 R1 sweep grid: {1x1, 1x2, 2x2, 3x3, 4x4}.
pub const DEFAULT_CYCLE_GRID: [HrmCycles; 5] = [
    HrmCycles { h_cycles: 1, l_cycles: 1 },
    HrmCycles { h_cycles: 1, l_cycles: 2 },
    HrmCycles { h_cycles: 2, l_cycles: 2 },
    HrmCycles { h_cycles: 3, l_cycles: 3 },
    HrmCycles { h_cycles: 4, l_cycles: 4 },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_match_adr199_table() {
        assert_eq!(TaskCategory::JsonExtraction.target(), 0.95);
        assert_eq!(TaskCategory::PlanUsefulness.target(), 0.80);
        assert_eq!(TaskCategory::ContradictionDetection.target(), 0.85);
        assert_eq!(TaskCategory::RoutingAccuracy.target(), 0.90);
    }

    #[test]
    fn plan_heuristic() {
        assert!(judge_plan_useful("1. analyze the schema\n2. migrate rows\n3. verify counts"));
        assert!(!judge_plan_useful("just do it"));
        assert!(!judge_plan_useful(""));
    }

    #[test]
    fn majority_vote_logic() {
        assert_eq!(majority_answer(&[Some(1), Some(2), Some(2)]), Some(2));
        assert_eq!(majority_answer(&[None, None]), None);
        assert_eq!(majority_answer(&[Some(7)]), Some(7));
    }

    #[test]
    fn default_suite_covers_all_categories_with_100_tasks() {
        let suite = AcceptanceSuite::default();
        assert!(suite.tasks.len() >= 100);
        for cat in TaskCategory::ALL {
            assert!(
                suite.tasks.iter().filter(|t| t.category == cat).count() >= 20,
                "category {cat:?} below 20 tasks"
            );
        }
    }
}
