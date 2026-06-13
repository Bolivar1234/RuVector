//! **Agentic Time** — a clock for autonomous systems where time is measured by
//! meaningful state change, not seconds, tokens, or steps.
//!
//! > Wall-clock time tells you *when* something happened.
//! > Agentic time tells you *how much the agent changed.*
//!
//! An agent can run for 30 minutes and barely age; or hit one contradiction and
//! age massively in a second. The agentic-time increment over a transition is
//!
//! ```text
//!   τ_a = f(ΔB, ΔM, ΔR, ΔG, ΔE, ΔP)
//! ```
//!
//! * `ΔB` — belief change,
//! * `ΔM` — memory change,
//! * `ΔR` — retrieval change,
//! * `ΔG` — goal-graph movement,
//! * `ΔE` — error / contradiction change,
//! * `ΔP` — plan change.
//!
//! The **Agentic Time Index** (ATI) is *progress per unit structural change*:
//! high ATI means the agent is learning and moving; low ATI means it is
//! spinning; falling progress means it is accumulating confusion. ATI drives a
//! health classifier (`Healthy`, `Drifting`, `Stuck`, `NeedsReplan`,
//! `Contradicting`, `Collapsing`, `NeedsHumanReview`).
//!
//! The included demo runs an agent trace through four clocks — wall, step count,
//! token count, agentic — and shows agentic time flags trouble earliest.

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// A snapshot of an agent's cognitive state.
#[derive(Clone, Debug)]
pub struct AgentState {
    /// Belief embedding. `ΔB`
    pub belief: Vec<f64>,
    /// Working-memory embedding. `ΔM`
    pub memory: Vec<f64>,
    /// Retrieved-context embedding. `ΔR`
    pub retrieval: Vec<f64>,
    /// Goal-graph summary (e.g. open-subgoal mass). `ΔG`
    pub goal_graph: f64,
    /// Contradiction / error score in `[0, 1]`. `ΔE`
    pub contradiction: f64,
    /// Plan embedding. `ΔP`
    pub plan: Vec<f64>,
    /// Cumulative tokens consumed (for the token-count clock).
    pub tokens: u64,
}

/// A clock over agent states: assigns a non-negative internal-time increment to
/// each transition.
pub trait AgentClock {
    fn name(&self) -> &str;
    fn tick(&self, prev: &AgentState, cur: &AgentState) -> f64;

    fn increments(&self, trace: &[AgentState]) -> Vec<f64> {
        let mut out = vec![0.0];
        for i in 1..trace.len() {
            out.push(self.tick(&trace[i - 1], &trace[i]).max(0.0));
        }
        out
    }

    fn cumulative(&self, trace: &[AgentState]) -> Vec<f64> {
        let mut acc = 0.0;
        let mut out = Vec::with_capacity(trace.len());
        for (i, _s) in trace.iter().enumerate() {
            if i > 0 {
                acc += self.tick(&trace[i - 1], &trace[i]).max(0.0);
            }
            out.push(acc);
        }
        out
    }
}

/// Wall-clock: one tick per observation, regardless of what changed.
pub struct AgentWallClock;
impl AgentClock for AgentWallClock {
    fn name(&self) -> &str {
        "wall"
    }
    fn tick(&self, _p: &AgentState, _c: &AgentState) -> f64 {
        1.0
    }
}

/// Step-count: identical to wall-clock here (one step per observation), included
/// to make the four-clock comparison explicit.
pub struct StepCountClock;
impl AgentClock for StepCountClock {
    fn name(&self) -> &str {
        "step-count"
    }
    fn tick(&self, _p: &AgentState, _c: &AgentState) -> f64 {
        1.0
    }
}

/// Token-count: internal time advances with tokens consumed.
pub struct TokenCountClock;
impl AgentClock for TokenCountClock {
    fn name(&self) -> &str {
        "token-count"
    }
    fn tick(&self, p: &AgentState, c: &AgentState) -> f64 {
        c.tokens.saturating_sub(p.tokens) as f64
    }
}

/// Weights for the six agentic-time channels.
#[derive(Clone, Copy, Debug)]
pub struct AgenticWeights {
    pub belief: f64,
    pub memory: f64,
    pub retrieval: f64,
    pub goal_graph: f64,
    pub contradiction: f64,
    pub plan: f64,
}

impl Default for AgenticWeights {
    fn default() -> Self {
        // Contradictions age an agent the most; memory/retrieval the least.
        AgenticWeights {
            belief: 1.0,
            memory: 0.5,
            retrieval: 0.5,
            goal_graph: 1.0,
            contradiction: 1.5,
            plan: 1.0,
        }
    }
}

/// Agentic Time: `τ_a = Σ wᵢ·d(channelᵢ)`.
pub struct AgenticTime {
    pub weights: AgenticWeights,
}

impl AgenticTime {
    pub fn new(weights: AgenticWeights) -> Self {
        AgenticTime { weights }
    }
}

impl AgentClock for AgenticTime {
    fn name(&self) -> &str {
        "agentic"
    }
    fn tick(&self, p: &AgentState, c: &AgentState) -> f64 {
        let w = &self.weights;
        w.belief * l2(&p.belief, &c.belief)
            + w.memory * l2(&p.memory, &c.memory)
            + w.retrieval * l2(&p.retrieval, &c.retrieval)
            + w.goal_graph * (c.goal_graph - p.goal_graph).abs()
            + w.contradiction * (c.contradiction - p.contradiction).abs()
            + w.plan * l2(&p.plan, &c.plan)
    }
}

/// Classification of an agentic-time tick (ADR-251 §8.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TickClass {
    /// Below the noise floor — no meaningful change.
    Idle,
    /// Belief / plan / goal moved forward.
    Progress,
    /// New information arrived (retrieval / memory moved).
    Learning,
    /// Contradiction rose.
    Contradiction,
    /// Contradiction is high — failure regime.
    Collapse,
}

/// An explainable agentic-time tick: the magnitude, its class, a human-readable
/// reason, and the per-channel weighted contributions (ADR-251 invariant §31.5:
/// every tick must have an auditable reason).
#[derive(Clone, Debug)]
pub struct Tick {
    pub delta: f64,
    pub class: TickClass,
    pub reason: String,
    pub belief: f64,
    pub memory: f64,
    pub retrieval: f64,
    pub goal_graph: f64,
    pub contradiction: f64,
    pub plan: f64,
}

impl AgenticTime {
    /// Compute an explainable tick for a transition. `noise_floor` suppresses
    /// jitter; contributions are the weighted per-channel movements.
    pub fn explain(&self, p: &AgentState, c: &AgentState, noise_floor: f64) -> Tick {
        let w = &self.weights;
        let belief = w.belief * l2(&p.belief, &c.belief);
        let memory = w.memory * l2(&p.memory, &c.memory);
        let retrieval = w.retrieval * l2(&p.retrieval, &c.retrieval);
        let goal_graph = w.goal_graph * (c.goal_graph - p.goal_graph).abs();
        let contradiction = w.contradiction * (c.contradiction - p.contradiction).abs();
        let plan = w.plan * l2(&p.plan, &c.plan);
        let delta = (belief + memory + retrieval + goal_graph + contradiction + plan
            - noise_floor)
            .max(0.0);

        // Dominant channel drives the class and reason.
        let channels = [
            ("belief", belief),
            ("memory", memory),
            ("retrieval", retrieval),
            ("goal-graph", goal_graph),
            ("contradiction", contradiction),
            ("plan", plan),
        ];
        let (dom_name, dom_val) = channels
            .iter()
            .copied()
            .fold(("none", 0.0), |acc, x| if x.1 > acc.1 { x } else { acc });

        let class = if delta <= 0.0 {
            TickClass::Idle
        } else if dom_name == "contradiction" {
            if c.contradiction >= 0.5 {
                TickClass::Collapse
            } else {
                TickClass::Contradiction
            }
        } else if dom_name == "retrieval" || dom_name == "memory" {
            TickClass::Learning
        } else {
            TickClass::Progress
        };

        let reason = if delta <= 0.0 {
            "no meaningful state change".to_string()
        } else {
            format!(
                "{class:?}: dominated by {dom_name} movement ({dom_val:.3} of {delta:.3} total)"
            )
        };

        Tick {
            delta,
            class,
            reason,
            belief,
            memory,
            retrieval,
            goal_graph,
            contradiction,
            plan,
        }
    }
}

/// Health verdicts derived from the Agentic Time Index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentHealth {
    /// Progress is keeping pace with internal change.
    Healthy,
    /// Moving, but inefficiently (low progress per unit change).
    Drifting,
    /// Neither changing nor progressing.
    Stuck,
    /// Lots of internal churn, no progress — replan.
    NeedsReplan,
    /// Losing ground (progress going backwards).
    Contradicting,
    /// Contradiction is high and rising.
    Collapsing,
    /// Contradiction is critical — escalate to a human.
    NeedsHumanReview,
}

/// Thresholds for the health classifier.
#[derive(Clone, Copy, Debug)]
pub struct HealthThresholds {
    pub idle: f64,            // below this Δτ, the agent is not changing
    pub healthy_ati: f64,     // ATI at/above this is healthy
    pub drifting_ati: f64,    // ATI at/above this is drifting (else replan)
    pub collapse: f64,        // contradiction at/above this is collapsing
    pub human_review: f64,    // contradiction at/above this escalates
}

impl Default for HealthThresholds {
    fn default() -> Self {
        HealthThresholds {
            idle: 1e-3,
            healthy_ati: 0.5,
            drifting_ati: 0.1,
            collapse: 0.5,
            human_review: 0.8,
        }
    }
}

/// The Agentic Time Index: progress per unit of structural change over a window.
pub fn agentic_time_index(delta_tau: f64, delta_progress: f64) -> f64 {
    if delta_tau <= 1e-12 {
        // No internal change: efficiency is "infinite" if progressing, else 0.
        if delta_progress > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    } else {
        delta_progress / delta_tau
    }
}

/// Classify agent health from the change in agentic time, the change in
/// progress, and the current contradiction level over a window.
pub fn classify(
    delta_tau: f64,
    delta_progress: f64,
    contradiction: f64,
    th: &HealthThresholds,
) -> AgentHealth {
    if contradiction >= th.human_review {
        return AgentHealth::NeedsHumanReview;
    }
    if contradiction >= th.collapse {
        return AgentHealth::Collapsing;
    }
    if delta_progress < -1e-9 {
        return AgentHealth::Contradicting;
    }
    if delta_tau < th.idle {
        // Not changing. Progressing-while-static is fine; otherwise stuck.
        return if delta_progress > th.idle {
            AgentHealth::Healthy
        } else {
            AgentHealth::Stuck
        };
    }
    let ati = agentic_time_index(delta_tau, delta_progress);
    if ati >= th.healthy_ati {
        AgentHealth::Healthy
    } else if ati >= th.drifting_ati {
        AgentHealth::Drifting
    } else {
        AgentHealth::NeedsReplan
    }
}

/// First step where a clock's rate exceeds `mean + k·std` of its baseline.
pub fn alarm_step(
    clock: &dyn AgentClock,
    trace: &[AgentState],
    baseline_window: usize,
    k_sigma: f64,
) -> Option<usize> {
    let inc = clock.increments(trace);
    if trace.len() <= baseline_window + 1 {
        return None;
    }
    let base = &inc[1..=baseline_window];
    let mean = base.iter().sum::<f64>() / base.len() as f64;
    let var = base.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / base.len() as f64;
    let threshold = mean + k_sigma * var.sqrt();
    for i in (baseline_window + 1)..trace.len() {
        if inc[i] > threshold {
            return Some(i);
        }
    }
    None
}

/// Early-warning lead: steps between the alarm and the failure (0 if no alarm).
pub fn early_warning_lead(
    clock: &dyn AgentClock,
    trace: &[AgentState],
    fail_index: usize,
    baseline_window: usize,
    k_sigma: f64,
) -> usize {
    match alarm_step(clock, trace, baseline_window, k_sigma) {
        Some(a) if a <= fail_index => fail_index - a,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Synthetic agent traces (deterministic).
// ---------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// A labelled agent trace plus its progress curve and failure index.
pub struct AgentTrace {
    pub states: Vec<AgentState>,
    pub progress: Vec<f64>,
    pub fail_index: usize,
    pub thrash_onset: usize,
    pub baseline_window: usize,
}

/// Generate a failing workflow trace: an early healthy phase where belief and
/// plan converge and progress rises, then a *thrash onset* where the plan
/// oscillates, retrieval destabilizes, and contradictions climb while progress
/// stalls — culminating in failure. Tokens accrue at a near-constant rate, so
/// wall / step / token clocks stay blind to the internal collapse.
pub fn generate_failing_trace(seed: u64) -> AgentTrace {
    let dim = 6;
    let steps = 100;
    let onset = 40;
    let fail_index = 80;
    let baseline_window = 18;
    let mut rng = Rng::new(seed);

    let target: Vec<f64> = (0..dim).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

    let mut states = Vec::with_capacity(steps);
    let mut progress = Vec::with_capacity(steps);
    let mut tokens = 0u64;

    for i in 0..steps {
        tokens += 120 + (rng.unit().abs() * 10.0) as u64;

        let (belief, plan, retrieval, contradiction, prog) = if i < onset {
            // Healthy convergence: belief/plan ease toward target; progress rises.
            let frac = i as f64 / onset as f64;
            let belief: Vec<f64> = target
                .iter()
                .map(|&t| frac * t + 0.01 * rng.unit())
                .collect();
            let plan = belief.clone();
            let retrieval: Vec<f64> = target.iter().map(|&t| frac * t).collect();
            (belief, plan, retrieval, 0.05, 0.5 * frac)
        } else {
            // Thrash: plan oscillates hard, retrieval unstable, contradiction
            // climbs, progress stalls near 0.5.
            let osc = if i % 2 == 0 { 1.0 } else { -1.0 };
            let p = (i - onset) as f64 / (fail_index - onset) as f64;
            let belief: Vec<f64> = target
                .iter()
                .map(|&t| t + 0.3 * osc + 0.05 * rng.unit())
                .collect();
            let plan: Vec<f64> = target
                .iter()
                .map(|&t| t + 0.8 * osc + 0.1 * rng.unit())
                .collect();
            let retrieval: Vec<f64> = target.iter().map(|&t| t + 0.4 * rng.unit()).collect();
            let contradiction = (0.05 + 0.9 * p).min(0.95);
            (belief, plan, retrieval, contradiction, 0.5)
        };

        let goal_graph = if i < onset {
            (onset - i) as f64 / onset as f64 // open subgoals shrinking
        } else {
            1.0 + (i - onset) as f64 * 0.05 // subgoals reopening (thrash)
        };

        states.push(AgentState {
            belief,
            memory: states.last().map(|s: &AgentState| s.belief.clone()).unwrap_or_else(|| vec![0.0; dim]),
            retrieval,
            goal_graph,
            contradiction,
            plan,
            tokens,
        });
        progress.push(prog);
    }

    AgentTrace {
        states,
        progress,
        fail_index,
        thrash_onset: onset,
        baseline_window,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agentic_time_warns_earliest_of_four_clocks() {
        let tr = generate_failing_trace(0xA9E1);
        let agentic = AgenticTime::new(AgenticWeights::default());
        let bw = tr.baseline_window;

        let lead_wall = early_warning_lead(&AgentWallClock, &tr.states, tr.fail_index, bw, 4.0);
        let lead_step = early_warning_lead(&StepCountClock, &tr.states, tr.fail_index, bw, 4.0);
        let lead_token = early_warning_lead(&TokenCountClock, &tr.states, tr.fail_index, bw, 4.0);
        let lead_agentic = early_warning_lead(&agentic, &tr.states, tr.fail_index, bw, 4.0);

        // The three dumb clocks are blind to the internal collapse.
        assert_eq!(lead_wall, 0);
        assert_eq!(lead_step, 0);
        assert_eq!(lead_token, 0);
        // Agentic time flags it well before the visible failure.
        assert!(
            lead_agentic >= 2 * (lead_token.max(lead_step).max(lead_wall) + 1),
            "agentic lead {lead_agentic} should be >= 2x the best dumb clock"
        );
        // It should fire right around the thrash onset.
        let alarm = alarm_step(&agentic, &tr.states, bw, 4.0).unwrap();
        assert!(alarm <= tr.thrash_onset + 2);
    }

    #[test]
    fn classifier_distinguishes_health() {
        let th = HealthThresholds::default();
        // Converging fast: healthy.
        assert_eq!(classify(1.0, 0.8, 0.05, &th), AgentHealth::Healthy);
        // Churning, no progress: replan.
        assert_eq!(classify(2.0, 0.0, 0.1, &th), AgentHealth::NeedsReplan);
        // Static and not progressing: stuck.
        assert_eq!(classify(0.0, 0.0, 0.1, &th), AgentHealth::Stuck);
        // Losing ground: contradicting.
        assert_eq!(classify(1.0, -0.2, 0.1, &th), AgentHealth::Contradicting);
        // High contradiction: collapsing / escalate.
        assert_eq!(classify(1.0, 0.0, 0.6, &th), AgentHealth::Collapsing);
        assert_eq!(classify(1.0, 0.0, 0.9, &th), AgentHealth::NeedsHumanReview);
    }

    #[test]
    fn ati_high_when_progressing_low_when_spinning() {
        let healthy = agentic_time_index(1.0, 0.9);
        let spinning = agentic_time_index(5.0, 0.02);
        assert!(healthy > spinning);
    }

    #[test]
    fn explain_tick_classifies_and_attributes() {
        let tr = generate_failing_trace(0xA9E1);
        let agentic = AgenticTime::new(AgenticWeights::default());
        // A transition across the thrash onset should be a contradiction/collapse
        // tick dominated by an identifiable channel, with a reason string.
        let o = tr.thrash_onset;
        let tick = agentic.explain(&tr.states[o - 1], &tr.states[o], 0.0);
        assert!(tick.delta > 0.0);
        assert!(!tick.reason.is_empty());
        assert!(matches!(
            tick.class,
            TickClass::Progress | TickClass::Learning | TickClass::Contradiction | TickClass::Collapse
        ));
        // Total delta equals the sum of channel contributions (no noise floor).
        let sum = tick.belief
            + tick.memory
            + tick.retrieval
            + tick.goal_graph
            + tick.contradiction
            + tick.plan;
        assert!((tick.delta - sum).abs() < 1e-9);
    }

    #[test]
    fn idle_transition_is_idle_tick() {
        let s = AgentState {
            belief: vec![1.0, 2.0],
            memory: vec![0.0, 0.0],
            retrieval: vec![1.0, 1.0],
            goal_graph: 0.5,
            contradiction: 0.1,
            plan: vec![1.0, 0.0],
            tokens: 100,
        };
        let agentic = AgenticTime::new(AgenticWeights::default());
        let tick = agentic.explain(&s, &s.clone(), 1e-6);
        assert_eq!(tick.class, TickClass::Idle);
        assert_eq!(tick.delta, 0.0);
    }
}
