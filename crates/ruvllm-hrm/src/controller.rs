//! The governed loop (ADR-199 Layer 3 — the actual product).
//!
//! HRM-Text never answers user queries directly. It sits in the loop:
//!
//! ```text
//! User request
//!   -> ruvector retrieval                 (context, memory, patterns)
//!   -> HRM planning pass    (Plan)        (decomposition, next action)
//!   -> model/tool routing   (router)      (local model | tool | vector search | workflow | remote)
//!   -> execution model                    (RuvLTRA / Qwen / remote — the workhorse)
//!   -> HRM verifier pass    (Verify)      (contradictions, missing facts -> retry or escalate)
//!   -> final answer
//! ```

use crate::backend::{GenerateRequest, LlmBackend};
use crate::prompt::{HrmMode, PromptBuilder};
use crate::router::{HrmRouter, RoutingDecision};
use crate::verifier::{HrmVerifier, Verdict};
use std::sync::Arc;

/// Retrieval hook: request -> context strings. This is the ruvector
/// integration point (vector search / memory); defaults to `None`.
pub type RetrievalFn = Box<dyn Fn(&str) -> Vec<String> + Send + Sync>;

/// Result of one governed loop run.
#[derive(Debug, Clone)]
pub struct ControllerResult {
    /// Final answer from the execution model (last attempt).
    pub answer: String,
    /// Plan produced by the HRM kernel (Plan mode).
    pub plan: String,
    /// Routing decision for the request.
    pub route: RoutingDecision,
    /// Verdict on the final answer.
    pub verdict: Verdict,
    /// Number of execution attempts (1 = passed first try or no retry left).
    pub attempts: usize,
}

/// Governed controller: HRM kernel plans/verifies, the executor answers.
pub struct Controller {
    kernel: Arc<dyn LlmBackend>,
    executor: Arc<dyn LlmBackend>,
    router: HrmRouter,
    verifier: HrmVerifier,
    max_retries: usize,
    retrieval: Option<RetrievalFn>,
    plan_builder: PromptBuilder,
}

impl Controller {
    /// New controller with `max_retries = 1` (one verify-fail retry, per
    /// ADR-199) and no retrieval hook.
    pub fn new(kernel: Arc<dyn LlmBackend>, executor: Arc<dyn LlmBackend>) -> Self {
        Self {
            router: HrmRouter::new(kernel.clone()),
            verifier: HrmVerifier::new(kernel.clone()),
            kernel,
            executor,
            max_retries: 1,
            retrieval: None,
            plan_builder: PromptBuilder::bare(HrmMode::Plan),
        }
    }

    /// Install the ruvector retrieval hook.
    pub fn with_retrieval(mut self, retrieval: RetrievalFn) -> Self {
        self.retrieval = Some(retrieval);
        self
    }

    /// Override the verify-fail retry budget.
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Run the governed loop for one request.
    pub async fn run(&self, request: &str) -> anyhow::Result<ControllerResult> {
        let request = request.trim();
        if request.is_empty() {
            anyhow::bail!("cannot run the governed loop on an empty request");
        }

        // 1. Retrieval (ruvector hook).
        let context: Vec<String> = self
            .retrieval
            .as_ref()
            .map(|f| f(request))
            .unwrap_or_default();

        // 2. HRM planning pass (Plan mode).
        let plan_input = if context.is_empty() {
            request.to_string()
        } else {
            format!("{request}\nContext:\n- {}", context.join("\n- "))
        };
        let plan = self
            .kernel
            .generate(self.plan_builder.to_request(&plan_input))
            .await?
            .text
            .trim()
            .to_string();

        // 3. Routing.
        let route = self.router.route(request).await?;
        tracing::debug!(route = %route.route.as_wire(), confidence = route.confidence, "routed");

        // 4 + 5. Execute, verify, retry once on FAIL with the verdict's
        // findings appended to the executor prompt.
        let mut attempts = 0usize;
        let mut feedback: Option<Verdict> = None;
        let (answer, verdict) = loop {
            attempts += 1;
            let prompt = self.executor_prompt(request, &context, &plan, feedback.as_ref());
            let answer = self
                .executor
                .generate(GenerateRequest::new(prompt))
                .await?
                .text
                .trim()
                .to_string();
            let verdict = self.verifier.verify(&answer, &[], &context).await?;
            if verdict.pass || attempts > self.max_retries {
                break (answer, verdict);
            }
            tracing::debug!(
                contradictions = verdict.contradictions.len(),
                missing = verdict.missing_facts.len(),
                "verification failed, retrying execution"
            );
            feedback = Some(verdict);
        };

        Ok(ControllerResult {
            answer,
            plan,
            route,
            verdict,
            attempts,
        })
    }

    /// Plain workhorse prompt — the executor is a normal instruction-tuned
    /// model, not the pre-alignment HRM kernel, so it gets no condition tags.
    fn executor_prompt(
        &self,
        request: &str,
        context: &[String],
        plan: &str,
        feedback: Option<&Verdict>,
    ) -> String {
        let mut prompt = format!("Task: {request}\n");
        if !context.is_empty() {
            prompt.push_str(&format!("Context:\n- {}\n", context.join("\n- ")));
        }
        prompt.push_str(&format!("Plan:\n{plan}\n"));
        if let Some(v) = feedback {
            prompt.push_str("The previous answer failed verification.\n");
            if !v.contradictions.is_empty() {
                prompt.push_str(&format!("Contradictions:\n- {}\n", v.contradictions.join("\n- ")));
            }
            if !v.missing_facts.is_empty() {
                prompt.push_str(&format!("Missing facts:\n- {}\n", v.missing_facts.join("\n- ")));
            }
            prompt.push_str("Provide a corrected answer.\n");
        }
        prompt.push_str("Answer:");
        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockBackend;

    #[tokio::test]
    async fn rejects_empty_request() {
        let kernel = Arc::new(MockBackend::new());
        let executor = Arc::new(MockBackend::new());
        let c = Controller::new(kernel, executor);
        assert!(c.run("  ").await.is_err());
    }

    #[tokio::test]
    async fn retrieval_hook_feeds_plan_and_verifier() {
        let kernel = Arc::new(MockBackend::new());
        let executor = Arc::new(MockBackend::new());
        // Kernel call order: plan, route, verify.
        kernel.push_response("1. step one\n2. step two");
        kernel.push_response("route=local_model:ruvltra confidence=0.9 rationale=simple");
        kernel.push_response("PASS");
        executor.push_response("The cache holds 100 entries.");
        let c = Controller::new(kernel.clone(), executor)
            .with_retrieval(Box::new(|_req| vec!["The cache holds 100 entries.".to_string()]));
        let result = c.run("How many entries does the cache hold?").await.unwrap();
        assert_eq!(result.attempts, 1);
        assert!(result.verdict.pass);
        let reqs = kernel.requests();
        // Plan prompt carried the retrieved context.
        assert!(reqs[0].prompt.contains("Context:"));
        assert!(reqs[0].prompt.starts_with("synth,cot\n"));
    }
}
