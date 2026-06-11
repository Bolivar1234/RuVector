//! ADR-199 composite scores — quality and cost published together.
//!
//! The benchmark's credibility rests on never reporting accuracy alone:
//! every run carries `edge_score` (quality per latency x watts x memory)
//! and `agent_score` (the weighted governance composite).

use serde::{Deserialize, Serialize};

/// `edge_score = task_quality / (latency_seconds × watts × memory_gb)`
///
/// The ADR-199 R4 edge benchmark headline number. All cost terms must be
/// measured, strictly positive values; `task_quality` is in `[0, 1]`.
pub fn edge_score(
    task_quality: f64,
    latency_seconds: f64,
    watts: f64,
    memory_gb: f64,
) -> anyhow::Result<f64> {
    if !(0.0..=1.0).contains(&task_quality) || !task_quality.is_finite() {
        anyhow::bail!("task_quality must be in [0, 1], got {task_quality}");
    }
    for (name, v) in [
        ("latency_seconds", latency_seconds),
        ("watts", watts),
        ("memory_gb", memory_gb),
    ] {
        if !v.is_finite() || v <= 0.0 {
            anyhow::bail!("{name} must be a finite positive measurement, got {v}");
        }
    }
    Ok(task_quality / (latency_seconds * watts * memory_gb))
}

/// Inputs to [`agent_score`], each in `[0, 1]`. Values are clamped at the
/// boundary so a slightly-off proxy cannot push the composite out of range.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AgentScoreInputs {
    pub task_success: f64,
    pub verifier_accuracy: f64,
    pub json_validity: f64,
    pub latency_score: f64,
    pub memory_score: f64,
}

/// ADR-199 governance composite:
///
/// ```text
/// agent_score = 0.40 × task_success
///             + 0.20 × verifier_accuracy
///             + 0.15 × json_validity
///             + 0.15 × latency_score
///             + 0.10 × memory_score
/// ```
pub fn agent_score(i: &AgentScoreInputs) -> f64 {
    let c = |v: f64| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
    0.40 * c(i.task_success)
        + 0.20 * c(i.verifier_accuracy)
        + 0.15 * c(i.json_validity)
        + 0.15 * c(i.latency_score)
        + 0.10 * c(i.memory_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_score_formula() {
        // 0.8 quality at 2s, 5W, 2GB -> 0.8 / 20 = 0.04
        let s = edge_score(0.8, 2.0, 5.0, 2.0).unwrap();
        assert!((s - 0.04).abs() < 1e-12);
    }

    #[test]
    fn edge_score_rejects_bad_measurements() {
        assert!(edge_score(1.5, 1.0, 1.0, 1.0).is_err());
        assert!(edge_score(0.5, 0.0, 1.0, 1.0).is_err());
        assert!(edge_score(0.5, 1.0, -2.0, 1.0).is_err());
        assert!(edge_score(0.5, 1.0, 1.0, f64::NAN).is_err());
    }

    #[test]
    fn agent_score_weights_sum_to_one() {
        let perfect = AgentScoreInputs {
            task_success: 1.0,
            verifier_accuracy: 1.0,
            json_validity: 1.0,
            latency_score: 1.0,
            memory_score: 1.0,
        };
        assert!((agent_score(&perfect) - 1.0).abs() < 1e-12);
        let zero = AgentScoreInputs {
            task_success: 0.0,
            verifier_accuracy: 0.0,
            json_validity: 0.0,
            latency_score: 0.0,
            memory_score: 0.0,
        };
        assert_eq!(agent_score(&zero), 0.0);
    }

    #[test]
    fn agent_score_component_weights() {
        let only_task = AgentScoreInputs {
            task_success: 1.0,
            verifier_accuracy: 0.0,
            json_validity: 0.0,
            latency_score: 0.0,
            memory_score: 0.0,
        };
        assert!((agent_score(&only_task) - 0.40).abs() < 1e-12);
        let clamped = AgentScoreInputs {
            task_success: 7.0, // clamped to 1.0
            verifier_accuracy: -3.0,
            json_validity: f64::NAN,
            latency_score: 0.0,
            memory_score: 0.0,
        };
        assert!((agent_score(&clamped) - 0.40).abs() < 1e-12);
    }
}
