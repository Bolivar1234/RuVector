//! Deterministic, programmable mock backend.
//!
//! Powers the unit/integration tests and the offline acceptance benchmarks
//! (`--mock` paths in the examples). Two layers:
//!
//! 1. **Queue**: canned responses/scores pushed by tests are returned first.
//! 2. **Rule-based default responder**: keyed on the condition-prefix mode
//!    markers found in the prompt (`direct\n`, `synth,cot\n` and the
//!    per-mode instruction lines), it computes small deterministic answers
//!    (key=value -> JSON, negation-aware contradiction checks, keyword
//!    routing, numbered plans, two-operand arithmetic).
//!
//! All received requests are recorded for assertions.

use crate::backend::{
    BackendError, GenerateRequest, GenerateResponse, LlmBackend, ScoreRequest, ScoreResponse,
    StateRequest, StateResponse,
};
use crate::router::{keyword_route, ROUTING_MARKER};
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::Mutex;

/// Programmable offline backend. See module docs.
#[derive(Default)]
pub struct MockBackend {
    queued: Mutex<VecDeque<String>>,
    queued_scores: Mutex<VecDeque<f64>>,
    recorded: Mutex<Vec<GenerateRequest>>,
    recorded_scores: Mutex<Vec<ScoreRequest>>,
}

impl MockBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a canned generation; consumed before the rule-based responder.
    pub fn push_response(&self, text: impl Into<String>) {
        self.queued.lock().unwrap().push_back(text.into());
    }

    /// Queue a canned score; consumed before the rule-based scorer.
    pub fn push_score(&self, score: f64) {
        self.queued_scores.lock().unwrap().push_back(score);
    }

    /// All generate requests received so far.
    pub fn requests(&self) -> Vec<GenerateRequest> {
        self.recorded.lock().unwrap().clone()
    }

    /// All score requests received so far.
    pub fn score_requests(&self) -> Vec<ScoreRequest> {
        self.recorded_scores.lock().unwrap().clone()
    }

    /// Rule-based default responder. Public-in-crate for benchmark sanity
    /// tests; behavior is part of the mock's contract.
    fn default_respond(prompt: &str) -> String {
        // Few-shot banks repeat instruction markers, so always key on the
        // LAST occurrence — the real task comes after the shots.
        if prompt.contains(ROUTING_MARKER) {
            return respond_routing(prompt);
        }
        if let Some(input) = last_segment(prompt, "Return strict JSON only.\n", "\nJSON:") {
            return respond_extract(input);
        }
        if prompt.contains("Check for contradictions") {
            return respond_verify(prompt);
        }
        if let Some(input) =
            last_segment(prompt, "Decompose this into executable steps.\n", "\nPlan:")
        {
            return respond_plan(input);
        }
        if let Some(input) = last_segment(prompt, "Problem:\n", "\nAnswer:") {
            return respond_synth_cot(input);
        }
        if let Some(input) = last_segment(prompt, "Input:\n", "\nOutput:") {
            return input.trim().to_string();
        }
        "OK".to_string()
    }
}

/// Slice between the LAST `start` marker and the following `end` marker.
fn last_segment<'a>(prompt: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let s = prompt.rfind(start)? + start.len();
    let rest = &prompt[s..];
    let e = rest.find(end)?;
    Some(&rest[..e])
}

fn respond_routing(prompt: &str) -> String {
    let intent = prompt
        .rfind("Request: ")
        .map(|p| prompt[p + "Request: ".len()..].lines().next().unwrap_or(""))
        .unwrap_or("");
    let route = keyword_route(intent);
    format!(
        "route={} confidence=0.92 rationale=keyword match on intent",
        route.as_wire()
    )
}

/// `fields: k=v; k=v` -> strict JSON object (i64 / f64 / bool / string typing).
fn respond_extract(input: &str) -> String {
    let body = input
        .trim()
        .strip_prefix("fields:")
        .unwrap_or(input.trim())
        .trim();
    let mut map = serde_json::Map::new();
    for pair in body.split(';') {
        if let Some((k, v)) = pair.split_once('=') {
            let (k, v) = (k.trim(), v.trim());
            if k.is_empty() {
                continue;
            }
            let value = if let Ok(i) = v.parse::<i64>() {
                serde_json::json!(i)
            } else if let Ok(f) = v.parse::<f64>() {
                serde_json::json!(f)
            } else if v == "true" || v == "false" {
                serde_json::json!(v == "true")
            } else {
                serde_json::json!(v)
            };
            map.insert(k.to_string(), value);
        }
    }
    serde_json::Value::Object(map).to_string()
}

fn respond_verify(prompt: &str) -> String {
    let Some(claim_pos) = prompt.rfind("Claim:") else {
        return "FAIL\nMissing facts:\n- no claim provided".to_string();
    };
    let segment = &prompt[claim_pos..];
    let line_after = |key: &str| -> Option<&str> {
        let p = segment.find(key)?;
        segment[p + key.len()..].lines().next().map(str::trim)
    };
    let claim = line_after("Claim:").unwrap_or("");
    let evidence = line_after("Evidence:").unwrap_or("(unverified)");
    if evidence.contains("(none)") {
        return "FAIL\nMissing facts:\n- no evidence provided".to_string();
    }
    let mut contradictions = Vec::new();
    for item in evidence.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        if item == "(unverified)" {
            continue;
        }
        if contradicts(claim, item) {
            contradictions.push(format!("- claim conflicts with evidence: {item}"));
        }
    }
    if contradictions.is_empty() {
        "PASS".to_string()
    } else {
        format!("FAIL\nContradictions:\n{}", contradictions.join("\n"))
    }
}

fn normalize(s: &str) -> String {
    s.to_lowercase()
        .trim()
        .trim_end_matches('.')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Tiny deterministic contradiction check: equal statements never contradict;
/// statements differing by a `not` contradict; `X is A` vs `X is B` with the
/// same subject and different predicate contradict.
fn contradicts(a: &str, b: &str) -> bool {
    let (na, nb) = (normalize(a), normalize(b));
    if na == nb {
        return false;
    }
    let strip_not = |s: &str| -> Option<String> {
        if let Some(rest) = s.strip_prefix("not ") {
            return Some(rest.to_string());
        }
        s.find(" not ").map(|p| format!("{}{}", &s[..p], &s[p + 4..]))
    };
    if strip_not(&na).map(|s| s == nb).unwrap_or(false)
        || strip_not(&nb).map(|s| s == na).unwrap_or(false)
    {
        return true;
    }
    if let (Some((sa, pa)), Some((sb, pb))) = (na.split_once(" is "), nb.split_once(" is ")) {
        if sa == sb && pa != pb {
            return true;
        }
    }
    false
}

fn respond_plan(input: &str) -> String {
    let topic: String = input.split_whitespace().take(6).collect::<Vec<_>>().join(" ");
    format!(
        "1. Analyze the task: {topic}\n\
         2. Gather required inputs and constraints\n\
         3. Execute the core action\n\
         4. Verify the result against the requirements"
    )
}

/// Solve `What is A <op> B?` (`+ - * x times plus minus`), else echo `42`.
fn respond_synth_cot(input: &str) -> String {
    match eval_arithmetic(input) {
        Some(n) => n.to_string(),
        None => "42".to_string(),
    }
}

/// Two-operand integer arithmetic over loose text.
fn eval_arithmetic(text: &str) -> Option<i64> {
    let lower = text.to_lowercase();
    let op: fn(i64, i64) -> i64 = if lower.contains('+') || lower.contains("plus") {
        |a, b| a + b
    } else if lower.contains('*') || lower.contains("times") || lower.contains(" x ") {
        |a, b| a * b
    } else if lower.contains('-') || lower.contains("minus") {
        |a, b| a - b
    } else {
        return None;
    };
    let mut nums = Vec::new();
    let mut cur = String::new();
    for c in lower.chars() {
        if c.is_ascii_digit() {
            cur.push(c);
        } else if !cur.is_empty() {
            nums.push(cur.parse::<i64>().ok()?);
            cur.clear();
        }
    }
    if !cur.is_empty() {
        nums.push(cur.parse::<i64>().ok()?);
    }
    if nums.len() < 2 {
        return None;
    }
    Some(op(nums[0], nums[1]))
}

#[async_trait]
impl LlmBackend for MockBackend {
    async fn generate(&self, req: GenerateRequest) -> anyhow::Result<GenerateResponse> {
        req.validate()?;
        self.recorded.lock().unwrap().push(req.clone());
        let text = match self.queued.lock().unwrap().pop_front() {
            Some(t) => t,
            None => Self::default_respond(&req.prompt),
        };
        Ok(GenerateResponse {
            text,
            finish_reason: Some("stop".to_string()),
            latency_ms: 0,
        })
    }

    async fn score(&self, req: ScoreRequest) -> anyhow::Result<ScoreResponse> {
        if req.prompt.trim().is_empty() || req.completion.trim().is_empty() {
            return Err(
                BackendError::InvalidRequest("empty prompt or completion".into()).into(),
            );
        }
        self.recorded_scores.lock().unwrap().push(req.clone());
        if let Some(score) = self.queued_scores.lock().unwrap().pop_front() {
            return Ok(ScoreResponse { score });
        }
        // Deterministic rule: arithmetic prompts score the correct final
        // integer near 0 (logprob-style, higher is better), wrong answers low.
        let score = match (
            eval_arithmetic(&req.prompt),
            crate::parse::extract_final_int(&req.completion),
        ) {
            (Some(expected), Some(got)) if expected == got => -0.5,
            (Some(_), _) => -20.0,
            _ => -(req.completion.len() as f64) * 0.01,
        };
        Ok(ScoreResponse { score })
    }

    async fn embed_state(&self, req: StateRequest) -> anyhow::Result<StateResponse> {
        if req.text.is_empty() {
            return Err(BackendError::InvalidRequest("empty state text".into()).into());
        }
        // Deterministic 8-dim hash embedding, good enough for plumbing tests.
        let mut embedding = vec![0.0f32; 8];
        for (i, b) in req.text.bytes().enumerate() {
            embedding[i % 8] += (b as f32) / 255.0;
        }
        let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        embedding.iter_mut().for_each(|v| *v /= norm);
        Ok(StateResponse { embedding })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::{HrmMode, PromptBuilder};

    #[tokio::test]
    async fn queue_takes_precedence_and_requests_recorded() {
        let mock = MockBackend::new();
        mock.push_response("canned");
        let r = mock.generate(GenerateRequest::new("direct\nanything")).await.unwrap();
        assert_eq!(r.text, "canned");
        let r = mock.generate(GenerateRequest::new("unmatched prompt")).await.unwrap();
        assert_eq!(r.text, "OK");
        assert_eq!(mock.requests().len(), 2);
    }

    #[tokio::test]
    async fn extract_responder_handles_typed_pairs() {
        let mock = MockBackend::new();
        let b = PromptBuilder::new(HrmMode::Extract);
        let r = mock
            .generate(b.to_request("fields: name=Ada; age=36; active=true; score=9.5"))
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&r.text).unwrap();
        assert_eq!(v["name"], "Ada");
        assert_eq!(v["age"], 36);
        assert_eq!(v["active"], true);
        assert_eq!(v["score"], 9.5);
    }

    #[test]
    fn contradiction_rules() {
        assert!(contradicts("The server is online.", "The server is not online."));
        assert!(contradicts("The sky is green.", "The sky is blue."));
        assert!(!contradicts("The sky is blue.", "The sky is blue."));
        assert!(!contradicts("Paris is the capital of France.", "France is in Europe."));
    }

    #[test]
    fn arithmetic_rules() {
        assert_eq!(eval_arithmetic("What is 17 + 25?"), Some(42));
        assert_eq!(eval_arithmetic("what is 6 times 7"), Some(42));
        assert_eq!(eval_arithmetic("What is 50 - 8?"), Some(42));
        assert_eq!(eval_arithmetic("no math"), None);
    }

    #[tokio::test]
    async fn embed_state_is_deterministic_unit_norm() {
        let mock = MockBackend::new();
        let a = mock.embed_state(StateRequest { text: "abc".into() }).await.unwrap();
        let b = mock.embed_state(StateRequest { text: "abc".into() }).await.unwrap();
        assert_eq!(a.embedding, b.embedding);
        let norm: f32 = a.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }
}
