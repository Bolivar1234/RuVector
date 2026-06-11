//! HRM-backed routing (ADR-199 Layer 3): user intent + context -> local
//! model, tool, vector search, workflow, or remote model. Composes with the
//! 3-tier model routing (ADR-026 lineage): HRM occupies the gap between
//! Agent Booster (Tier 1) and remote frontier models (Tier 3).

use crate::backend::{HrmCycles, LlmBackend};
use crate::prompt::{FewShot, HrmMode, PromptBuilder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Compute-depth class of a task. The router schedules recurrence depth as
/// a **routed resource** (ADR-199): each class maps to a cycle budget via
/// [`choose_cycles`], which callers attach to
/// [`crate::backend::GenerateRequest::hrm_cycles`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeClass {
    /// Strict extraction / formatting — minimum depth.
    ExtractFast,
    /// Standard planning passes.
    PlanNormal,
    /// Verification needs deeper recurrence than planning.
    VerifyDeep,
    /// Math / multi-step reasoning — maximum depth.
    MathDeep,
    /// Unclassified intent — moderate, L-heavy default.
    UnknownAdaptive,
}

impl ComputeClass {
    /// Default class for a prompting mode.
    pub fn for_mode(mode: HrmMode) -> Self {
        match mode {
            HrmMode::Extract | HrmMode::Direct => ComputeClass::ExtractFast,
            HrmMode::Plan => ComputeClass::PlanNormal,
            HrmMode::Verify => ComputeClass::VerifyDeep,
            HrmMode::SynthCot => ComputeClass::MathDeep,
        }
    }
}

/// ADR-199 cycle schedule: compute depth per task class.
pub fn choose_cycles(class: ComputeClass) -> HrmCycles {
    match class {
        ComputeClass::ExtractFast => HrmCycles { h_cycles: 1, l_cycles: 1 },
        ComputeClass::PlanNormal => HrmCycles { h_cycles: 2, l_cycles: 2 },
        ComputeClass::VerifyDeep => HrmCycles { h_cycles: 3, l_cycles: 3 },
        ComputeClass::MathDeep => HrmCycles { h_cycles: 4, l_cycles: 4 },
        ComputeClass::UnknownAdaptive => HrmCycles { h_cycles: 2, l_cycles: 3 },
    }
}

/// Routing destination.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Route {
    LocalModel(String),
    Tool(String),
    VectorSearch,
    Workflow(String),
    Remote(String),
}

impl Route {
    /// Canonical wire form: `tool:calculator`, `vector_search`, ...
    pub fn as_wire(&self) -> String {
        match self {
            Route::LocalModel(m) => format!("local_model:{m}"),
            Route::Tool(t) => format!("tool:{t}"),
            Route::VectorSearch => "vector_search".to_string(),
            Route::Workflow(w) => format!("workflow:{w}"),
            Route::Remote(r) => format!("remote:{r}"),
        }
    }

    /// Parse the canonical wire form. Returns `None` for unknown kinds or
    /// missing arguments.
    pub fn from_wire(s: &str) -> Option<Route> {
        let s = s.trim();
        if s == "vector_search" {
            return Some(Route::VectorSearch);
        }
        let (kind, arg) = s.split_once(':')?;
        let arg = arg.trim();
        if arg.is_empty() {
            return None;
        }
        match kind.trim() {
            "local_model" => Some(Route::LocalModel(arg.to_string())),
            "tool" => Some(Route::Tool(arg.to_string())),
            "workflow" => Some(Route::Workflow(arg.to_string())),
            "remote" => Some(Route::Remote(arg.to_string())),
            _ => None,
        }
    }
}

/// Typed routing decision (ADR-199 `router.rs returns a typed RoutingDecision`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub route: Route,
    pub confidence: f32,
    pub rationale: String,
}

/// Deterministic keyword routing. Shared by the [`HrmRouter`] fallback and
/// by [`crate::mock::MockBackend`]'s default responder so offline benchmarks
/// and the fallback agree on canonical destinations.
pub fn keyword_route(intent: &str) -> Route {
    let t = intent.to_lowercase();
    let any = |keys: &[&str]| keys.iter().any(|k| t.contains(k));
    if any(&["calculate", "compute", "sum of", "multiply", "divide", "average"]) {
        Route::Tool("calculator".to_string())
    } else if any(&["weather", "forecast"]) {
        Route::Tool("weather".to_string())
    } else if any(&["search", "find similar", "look up", "retrieve", "nearest neighbor"]) {
        Route::VectorSearch
    } else if any(&["deploy", "pipeline", "workflow", "backup", "ci ", "trigger"]) {
        Route::Workflow("ops".to_string())
    } else if any(&["architecture", "legal", "formal proof", "theorem", "multi-region"]) {
        Route::Remote("frontier".to_string())
    } else {
        Route::LocalModel("ruvltra".to_string())
    }
}

/// Marker the mock responder and the prompt share to recognize routing asks.
pub(crate) const ROUTING_MARKER: &str = "Route the following request";

/// Router that asks the HRM kernel in Direct mode with a routing few-shot
/// bank, parsing a structured `route=... confidence=... rationale=...` line.
/// Falls back to [`keyword_route`] below the confidence threshold or on
/// parse failure.
pub struct HrmRouter {
    backend: Arc<dyn LlmBackend>,
    builder: PromptBuilder,
    confidence_threshold: f32,
}

/// Confidence reported by the rule-based fallback.
const FALLBACK_CONFIDENCE: f32 = 0.5;

impl HrmRouter {
    pub fn new(backend: Arc<dyn LlmBackend>) -> Self {
        let builder = PromptBuilder::bare(HrmMode::Direct)
            .with_shots(routing_bank())
            .expect("built-in routing bank is within the shot cap");
        Self {
            backend,
            builder,
            confidence_threshold: 0.6,
        }
    }

    /// Override the fallback threshold (clamped to `0.0..=1.0`).
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Route a user intent.
    pub async fn route(&self, intent: &str) -> anyhow::Result<RoutingDecision> {
        let intent = intent.trim();
        if intent.is_empty() {
            anyhow::bail!("cannot route an empty intent");
        }
        let input = format!(
            "{ROUTING_MARKER} to exactly one destination \
             (local_model:<name> | tool:<name> | vector_search | workflow:<name> | remote:<name>). \
             Reply with: route=<destination> confidence=<0..1> rationale=<short reason>.\n\
             Request: {intent}"
        );
        let resp = self.backend.generate(self.builder.to_request(&input)).await?;
        match parse_decision(&resp.text) {
            Some(d) if d.confidence >= self.confidence_threshold => Ok(d),
            Some(d) => {
                tracing::debug!(confidence = d.confidence, "below threshold, using keyword fallback");
                Ok(Self::rule_based(intent))
            }
            None => {
                tracing::debug!("unparseable routing reply, using keyword fallback");
                Ok(Self::rule_based(intent))
            }
        }
    }

    /// Deterministic keyword fallback.
    pub fn rule_based(intent: &str) -> RoutingDecision {
        RoutingDecision {
            route: keyword_route(intent),
            confidence: FALLBACK_CONFIDENCE,
            rationale: "rule-based keyword fallback".to_string(),
        }
    }
}

/// Parse `route=<dest> confidence=<f32> rationale=<text>` from a generation.
fn parse_decision(text: &str) -> Option<RoutingDecision> {
    let line = text.lines().find(|l| l.contains("route="))?;
    let field = |key: &str| -> Option<&str> {
        let pos = line.find(key)?;
        let rest = &line[pos + key.len()..];
        rest.split_whitespace().next()
    };
    let route = Route::from_wire(field("route=")?)?;
    let confidence = field("confidence=")?
        .parse::<f32>()
        .ok()
        .filter(|c| c.is_finite())?;
    // Rationale runs to end of line.
    let rationale = line
        .find("rationale=")
        .map(|p| line[p + "rationale=".len()..].trim().to_string())
        .unwrap_or_default();
    Some(RoutingDecision {
        route,
        confidence: confidence.clamp(0.0, 1.0),
        rationale,
    })
}

/// Built-in routing few-shot bank (Direct mode, compact shots).
fn routing_bank() -> Vec<FewShot> {
    vec![
        FewShot::new(
            "Route: compute 12 * 9",
            "route=tool:calculator confidence=0.95 rationale=arithmetic intent",
        ),
        FewShot::new(
            "Route: find documents similar to this incident report",
            "route=vector_search confidence=0.9 rationale=similarity retrieval",
        ),
        FewShot::new(
            "Route: summarize this paragraph",
            "route=local_model:ruvltra confidence=0.85 rationale=simple NLP task",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockBackend;

    #[test]
    fn wire_roundtrip() {
        for r in [
            Route::LocalModel("ruvltra".into()),
            Route::Tool("calculator".into()),
            Route::VectorSearch,
            Route::Workflow("ops".into()),
            Route::Remote("frontier".into()),
        ] {
            assert_eq!(Route::from_wire(&r.as_wire()), Some(r));
        }
        assert_eq!(Route::from_wire("tool:"), None);
        assert_eq!(Route::from_wire("nonsense"), None);
    }

    #[test]
    fn decision_parsing() {
        let d = parse_decision("route=tool:weather confidence=0.93 rationale=asks about rain")
            .unwrap();
        assert_eq!(d.route, Route::Tool("weather".into()));
        assert!((d.confidence - 0.93).abs() < 1e-6);
        assert_eq!(d.rationale, "asks about rain");
        assert!(parse_decision("no structure at all").is_none());
        assert!(parse_decision("route=bogus confidence=0.9 rationale=x").is_none());
    }

    #[tokio::test]
    async fn router_uses_kernel_then_falls_back() {
        let mock = Arc::new(MockBackend::new());
        let router = HrmRouter::new(mock.clone());
        let d = router.route("Calculate the sum of 19 and 23").await.unwrap();
        assert_eq!(d.route, Route::Tool("calculator".into()));
        assert!(d.confidence >= 0.6);

        // Garbage reply -> keyword fallback.
        mock.push_response("I cannot decide anything today.");
        let d = router.route("Deploy the new release to staging").await.unwrap();
        assert_eq!(d.route, Route::Workflow("ops".into()));
        assert_eq!(d.rationale, "rule-based keyword fallback");

        // Low-confidence reply -> keyword fallback too.
        mock.push_response("route=remote:frontier confidence=0.1 rationale=unsure");
        let d = router.route("What's the weather in Oslo?").await.unwrap();
        assert_eq!(d.route, Route::Tool("weather".into()));

        assert!(router.route("   ").await.is_err());
    }
}
