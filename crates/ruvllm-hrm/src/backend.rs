//! `LlmBackend` transport trait + `HrmTextBackend` (vLLM/SGLang HTTP client).
//!
//! ADR-199 Layer 1: a provider trait so the transport can later swap from
//! HTTP to native Rust inference (Phase D) without touching the router,
//! verifier, or controller. The deterministic offline implementation lives
//! in [`crate::mock::MockBackend`].

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// PrefixLM handling mode of the serving path.
///
/// HRM-Text is PrefixLM-trained: prompt tokens attend bidirectionally and
/// inference requires `token_type_ids = ones` to match training. Whether an
/// OpenAI-compatible endpoint applies this internally is the top
/// silent-failure risk called out by ADR-199; record the verified mode here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PrefixMode {
    /// Endpoint verified to apply the PrefixLM mask (parity-checked).
    #[default]
    PrefixLm,
    /// Endpoint runs causal-only attention; quality degrades for HRM-Text.
    CausalOnly,
}

/// Backend errors. Wrapped in `anyhow::Error` at the trait boundary.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("operation unsupported by this backend: {0}")]
    Unsupported(&'static str),
    #[error("http error: {0}")]
    Http(String),
    #[error("malformed response: {0}")]
    MalformedResponse(String),
}

/// Recurrence depth for one request (ADR-199 R1 test-time-compute hook).
///
/// Cycle depth is a **first-class, routable inference parameter**, not
/// hidden backend config — the router schedules it per task class via
/// [`crate::router::choose_cycles`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HrmCycles {
    /// High-level (slow, strategic) recurrence cycles.
    pub h_cycles: usize,
    /// Low-level (fast, execution) cycles per H cycle.
    pub l_cycles: usize,
}

impl HrmCycles {
    pub fn new(h_cycles: usize, l_cycles: usize) -> Self {
        Self { h_cycles, l_cycles }
    }
}

/// A single text-completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// Full prompt text. Must come from [`crate::prompt`] for HRM-Text —
    /// the checkpoint is pre-alignment and condition-token-trained.
    pub prompt: String,
    /// Maximum tokens to generate (default 512, per ADR-199 kernel guidance).
    pub max_tokens: u32,
    /// Sampling temperature (default 0.2 for kernel tasks).
    pub temperature: f32,
    /// Stop sequences.
    pub stop: Vec<String>,
    /// Per-request recurrence depth. Forwarded as extra body params
    /// `h_cycles` / `l_cycles`; servers that ignore them are fine (the R1
    /// sweep then measures a flat line).
    pub hrm_cycles: Option<HrmCycles>,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 512,
            temperature: 0.2,
            stop: Vec::new(),
            hrm_cycles: None,
        }
    }
}

impl GenerateRequest {
    /// New request with kernel-task defaults (max_tokens 512, temperature 0.2).
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            ..Self::default()
        }
    }

    /// Boundary validation: non-empty prompt, sane token budget/temperature.
    pub fn validate(&self) -> Result<(), BackendError> {
        if self.prompt.trim().is_empty() {
            return Err(BackendError::InvalidRequest("empty prompt".into()));
        }
        if self.max_tokens == 0 || self.max_tokens > 8192 {
            return Err(BackendError::InvalidRequest(format!(
                "max_tokens out of range (1..=8192): {}",
                self.max_tokens
            )));
        }
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err(BackendError::InvalidRequest(format!(
                "temperature out of range (0.0..=2.0): {}",
                self.temperature
            )));
        }
        Ok(())
    }
}

/// Completion result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    /// Generated text (`choices[0].text`).
    pub text: String,
    /// Finish reason if reported by the server.
    pub finish_reason: Option<String>,
    /// Wall-clock latency observed by the backend, in milliseconds.
    pub latency_ms: u64,
}

/// Scoring request: how plausible is `completion` given `prompt`?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreRequest {
    pub prompt: String,
    pub completion: String,
    /// R1 hook, same semantics as [`GenerateRequest::hrm_cycles`].
    pub hrm_cycles: Option<HrmCycles>,
}

/// Scoring result. Higher is better; logprob-style scores are <= 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreResponse {
    pub score: f64,
}

/// Request for a compressed recurrence-state embedding (R3 / ruvector memory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateRequest {
    pub text: String,
}

/// Recurrence-state embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateResponse {
    pub embedding: Vec<f32>,
}

/// Transport-agnostic LLM backend (ADR-199 Layer 1).
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// Generate a completion.
    async fn generate(&self, req: GenerateRequest) -> anyhow::Result<GenerateResponse>;
    /// Score a completion under the model (best-of-N reranking, R5).
    async fn score(&self, req: ScoreRequest) -> anyhow::Result<ScoreResponse>;
    /// Embed hidden/recurrence state for ruvector memory (native-path
    /// feature; the HTTP backend returns [`BackendError::Unsupported`]).
    async fn embed_state(&self, req: StateRequest) -> anyhow::Result<StateResponse>;
}

/// HTTP backend for a vLLM/SGLang OpenAI-compatible `/v1/completions` endpoint.
///
/// ```bash
/// pip install vllm
/// vllm serve sapientinc/HRM-Text-1B
/// ```
pub struct HrmTextBackend {
    endpoint: String,
    model: String,
    tokenizer_mode: PrefixMode,
    client: reqwest::Client,
}

/// Backoff schedule for transient HTTP failures (connect errors, 429, 5xx).
const RETRY_BACKOFF_MS: [u64; 3] = [200, 400, 800];

#[derive(Deserialize)]
struct CompletionsResponse {
    choices: Vec<CompletionChoice>,
}

#[derive(Deserialize)]
struct CompletionChoice {
    text: Option<String>,
    finish_reason: Option<String>,
    logprobs: Option<ChoiceLogprobs>,
}

#[derive(Deserialize)]
struct ChoiceLogprobs {
    token_logprobs: Option<Vec<Option<f64>>>,
}

impl HrmTextBackend {
    /// Create a backend for `endpoint` (e.g. `http://localhost:8000/v1`)
    /// serving `model` (e.g. `sapientinc/HRM-Text-1B`).
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> anyhow::Result<Self> {
        let endpoint = endpoint.into();
        let endpoint = endpoint.trim_end_matches('/').to_string();
        if !(endpoint.starts_with("http://") || endpoint.starts_with("https://")) {
            return Err(BackendError::InvalidRequest(format!(
                "endpoint must be an http(s) URL, got {endpoint:?}"
            ))
            .into());
        }
        let model = model.into();
        if model.trim().is_empty() {
            return Err(BackendError::InvalidRequest("empty model name".into()).into());
        }
        Ok(Self {
            endpoint,
            model,
            tokenizer_mode: PrefixMode::default(),
            client: reqwest::Client::new(),
        })
    }

    /// Record the verified PrefixLM serving mode (see [`PrefixMode`]).
    pub fn with_prefix_mode(mut self, mode: PrefixMode) -> Self {
        self.tokenizer_mode = mode;
        self
    }

    /// The verified PrefixLM serving mode.
    pub fn prefix_mode(&self) -> PrefixMode {
        self.tokenizer_mode
    }

    fn completions_url(&self) -> String {
        format!("{}/completions", self.endpoint)
    }

    /// POST with up to 3 retries on transient failures (exponential backoff
    /// 200ms / 400ms / 800ms). 4xx other than 429 fails immediately.
    async fn post_completions(
        &self,
        body: &serde_json::Value,
    ) -> anyhow::Result<CompletionsResponse> {
        let url = self.completions_url();
        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 0..=RETRY_BACKOFF_MS.len() {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(RETRY_BACKOFF_MS[attempt - 1])).await;
            }
            match self.client.post(&url).json(body).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let parsed: CompletionsResponse = resp.json().await.map_err(|e| {
                            BackendError::MalformedResponse(format!("bad completions JSON: {e}"))
                        })?;
                        return Ok(parsed);
                    }
                    let transient = status.as_u16() == 429 || status.is_server_error();
                    let text = resp.text().await.unwrap_or_default();
                    let err = BackendError::Http(format!("{url}: status {status}: {text}"));
                    if !transient {
                        return Err(err.into());
                    }
                    tracing::warn!(attempt, %status, "transient HTTP failure, will retry");
                    last_err = Some(err.into());
                }
                Err(e) => {
                    tracing::warn!(attempt, error = %e, "request failure, will retry");
                    last_err = Some(BackendError::Http(format!("{url}: {e}")).into());
                }
            }
        }
        Err(last_err.unwrap_or_else(|| BackendError::Http("retries exhausted".into()).into()))
    }

    fn build_body(&self, req: &GenerateRequest, echo_score: bool) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": self.model,
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        });
        if !req.stop.is_empty() {
            body["stop"] = serde_json::json!(req.stop);
        }
        if let Some(HrmCycles { h_cycles, l_cycles }) = req.hrm_cycles {
            // R1 hook: forwarded as extra body params; servers that do not
            // understand them ignore unknown fields.
            body["h_cycles"] = serde_json::json!(h_cycles);
            body["l_cycles"] = serde_json::json!(l_cycles);
        }
        if echo_score {
            body["max_tokens"] = serde_json::json!(0);
            body["echo"] = serde_json::json!(true);
            body["logprobs"] = serde_json::json!(0);
        }
        body
    }
}

#[async_trait]
impl LlmBackend for HrmTextBackend {
    async fn generate(&self, req: GenerateRequest) -> anyhow::Result<GenerateResponse> {
        req.validate()?;
        let start = Instant::now();
        let body = self.build_body(&req, false);
        let parsed = self.post_completions(&body).await?;
        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| BackendError::MalformedResponse("empty choices array".into()))?;
        Ok(GenerateResponse {
            text: choice.text.unwrap_or_default(),
            finish_reason: choice.finish_reason,
            latency_ms: start.elapsed().as_millis() as u64,
        })
    }

    async fn score(&self, req: ScoreRequest) -> anyhow::Result<ScoreResponse> {
        if req.prompt.trim().is_empty() || req.completion.trim().is_empty() {
            return Err(
                BackendError::InvalidRequest("empty prompt or completion".into()).into(),
            );
        }
        // Logprob-style scoring: echo the full text back with 0 new tokens
        // and sum token logprobs when the server reports them.
        let full = format!("{}{}", req.prompt, req.completion);
        let gen_req = GenerateRequest {
            prompt: full,
            max_tokens: 1, // validated > 0; build_body(echo) overrides to 0
            temperature: 0.0,
            stop: Vec::new(),
            hrm_cycles: req.hrm_cycles,
        };
        let body = self.build_body(&gen_req, true);
        let parsed = self.post_completions(&body).await?;
        if let Some(choice) = parsed.choices.into_iter().next() {
            if let Some(lp) = choice.logprobs.and_then(|l| l.token_logprobs) {
                let score: f64 = lp.into_iter().flatten().sum();
                return Ok(ScoreResponse { score });
            }
        }
        // Fallback: a Verify-mode generation parsed to a coarse score.
        let verify_input = format!(
            "Claim: {}\nRules: (none)\nEvidence: {}",
            req.completion, req.prompt
        );
        let verify_req = GenerateRequest {
            prompt: crate::prompt::build_hrm_prompt(crate::prompt::HrmMode::Verify, &verify_input),
            max_tokens: 128,
            temperature: 0.0,
            stop: vec!["\n\n".to_string()],
            hrm_cycles: req.hrm_cycles,
        };
        let resp = self.generate(verify_req).await?;
        let verdict = crate::parse::parse_verdict(&resp.text);
        let penalty = (verdict.contradictions.len() + verdict.missing_facts.len()) as f64;
        let score = if verdict.pass { 0.0 } else { -10.0 - penalty };
        Ok(ScoreResponse { score })
    }

    async fn embed_state(&self, _req: StateRequest) -> anyhow::Result<StateResponse> {
        // Recurrence-state extraction (z_H snapshots, R3) requires in-process
        // access to hidden state — a native-path (Phase D) feature. The HTTP
        // completions API cannot expose it.
        Err(BackendError::Unsupported(
            "embed_state requires the native inference path (ADR-199 Phase D); \
             the HTTP completions API does not expose recurrence state",
        )
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults_match_adr() {
        let req = GenerateRequest::new("direct\nhello");
        assert_eq!(req.max_tokens, 512);
        assert!((req.temperature - 0.2).abs() < f32::EPSILON);
        assert!(req.hrm_cycles.is_none());
        assert!(req.validate().is_ok());
    }

    #[test]
    fn request_validation_rejects_bad_inputs() {
        assert!(GenerateRequest::new("  ").validate().is_err());
        let mut req = GenerateRequest::new("x");
        req.max_tokens = 0;
        assert!(req.validate().is_err());
        req.max_tokens = 16;
        req.temperature = f32::NAN;
        assert!(req.validate().is_err());
        req.temperature = 3.0;
        assert!(req.validate().is_err());
    }

    #[test]
    fn backend_endpoint_validation() {
        assert!(HrmTextBackend::new("ftp://x", "m").is_err());
        assert!(HrmTextBackend::new("http://localhost:8000/v1", "").is_err());
        let b = HrmTextBackend::new("http://localhost:8000/v1/", "m").unwrap();
        assert_eq!(b.completions_url(), "http://localhost:8000/v1/completions");
        assert_eq!(b.prefix_mode(), PrefixMode::PrefixLm);
    }

    #[test]
    fn hrm_cycles_forwarded_in_body() {
        let b = HrmTextBackend::new("http://localhost:8000/v1", "m").unwrap();
        let mut req = GenerateRequest::new("p");
        req.hrm_cycles = Some(HrmCycles::new(3, 4));
        req.stop = vec!["\n\n".into()];
        let body = b.build_body(&req, false);
        assert_eq!(body["h_cycles"], 3);
        assert_eq!(body["l_cycles"], 4);
        assert_eq!(body["stop"][0], "\n\n");
        let body = b.build_body(&req, true);
        assert_eq!(body["max_tokens"], 0);
        assert_eq!(body["echo"], true);
    }
}
