//! HRM verifier pass (ADR-199 Layer 3): generated answer + rules + evidence
//! -> pass/fail, contradictions, missing facts.

use crate::backend::LlmBackend;
use crate::parse::parse_verdict;
use crate::prompt::{HrmMode, PromptBuilder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Typed verifier verdict (ADR-199 `Verdict { pass, contradictions,
/// missing_facts }`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    pub pass: bool,
    pub contradictions: Vec<String>,
    pub missing_facts: Vec<String>,
}

impl Verdict {
    /// A passing verdict with no findings.
    pub fn passing() -> Self {
        Self {
            pass: true,
            contradictions: Vec::new(),
            missing_facts: Vec::new(),
        }
    }
}

/// Verifier built on a Verify-mode HRM prompt.
pub struct HrmVerifier {
    backend: Arc<dyn LlmBackend>,
    builder: PromptBuilder,
}

impl HrmVerifier {
    /// New verifier with the built-in Verify few-shot bank.
    pub fn new(backend: Arc<dyn LlmBackend>) -> Self {
        Self {
            backend,
            builder: PromptBuilder::new(HrmMode::Verify),
        }
    }

    /// Replace the prompt builder (must be Verify mode).
    pub fn with_builder(mut self, builder: PromptBuilder) -> anyhow::Result<Self> {
        if builder.mode() != HrmMode::Verify {
            anyhow::bail!("HrmVerifier requires a Verify-mode PromptBuilder");
        }
        self.builder = builder;
        Ok(self)
    }

    /// Check `answer` against `rules` and `evidence`.
    ///
    /// Empty evidence is rendered as `(unverified)` — nothing to contradict —
    /// while an explicit `(none)` evidence item signals that required
    /// evidence is missing.
    pub async fn verify(
        &self,
        answer: &str,
        rules: &[String],
        evidence: &[String],
    ) -> anyhow::Result<Verdict> {
        if answer.trim().is_empty() {
            anyhow::bail!("cannot verify an empty answer");
        }
        let rules_line = if rules.is_empty() {
            "(none)".to_string()
        } else {
            rules.join("; ")
        };
        let evidence_line = if evidence.is_empty() {
            "(unverified)".to_string()
        } else {
            evidence.join("; ")
        };
        let input = format!(
            "Claim: {}\nRules: {}\nEvidence: {}",
            answer.trim(),
            rules_line,
            evidence_line
        );
        let resp = self.backend.generate(self.builder.to_request(&input)).await?;
        Ok(parse_verdict(&resp.text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockBackend;

    #[tokio::test]
    async fn verify_pass_and_fail_via_mock_rules() {
        let mock = Arc::new(MockBackend::new());
        let verifier = HrmVerifier::new(mock.clone());

        let v = verifier
            .verify("The server is online.", &[], &["The server is online.".into()])
            .await
            .unwrap();
        assert!(v.pass, "identical evidence should pass: {v:?}");

        let v = verifier
            .verify(
                "The server is online.",
                &[],
                &["The server is not online.".into()],
            )
            .await
            .unwrap();
        assert!(!v.pass);
        assert!(!v.contradictions.is_empty());

        // The verifier sent Verify-mode prompts through the kernel template.
        let reqs = mock.requests();
        assert!(reqs.iter().all(|r| r.prompt.starts_with("direct\n")));
        assert!(reqs.iter().all(|r| r.prompt.contains("Check for contradictions")));
    }

    #[tokio::test]
    async fn verify_rejects_empty_answer_and_wrong_mode() {
        let mock = Arc::new(MockBackend::new());
        let verifier = HrmVerifier::new(mock.clone());
        assert!(verifier.verify("  ", &[], &[]).await.is_err());
        assert!(HrmVerifier::new(mock)
            .with_builder(PromptBuilder::bare(HrmMode::Plan))
            .is_err());
    }
}
