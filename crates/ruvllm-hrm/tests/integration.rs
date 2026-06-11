//! End-to-end tests for ruvllm-hrm (ADR-199 Phases A-C) against the
//! deterministic MockBackend, plus a minimal tokio TCP HTTP stub that
//! exercises HrmTextBackend's real reqwest path with no external network.

use ruvllm_hrm::{
    build_hrm_prompt, extract_json_with_retry, parse_strict_json, parse_verdict, run_acceptance,
    run_best_of_n, run_cycle_sweep, Controller, GenerateRequest, HrmCycles, HrmMode, HrmRouter,
    HrmTextBackend, LlmBackend, MockBackend, PromptBuilder, Route, ScoreRequest, StateRequest,
};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

// ---------------------------------------------------------------------------
// Prompt templates (exact ADR-199 strings)
// ---------------------------------------------------------------------------

#[test]
fn adr199_templates_exact() {
    assert_eq!(
        build_hrm_prompt(HrmMode::Direct, "INPUT"),
        "direct\nTask: Extract the answer.\nInput:\nINPUT\nOutput:"
    );
    assert_eq!(
        build_hrm_prompt(HrmMode::SynthCot, "INPUT"),
        "synth,cot\nSolve carefully.\nProblem:\nINPUT\nAnswer:"
    );
    assert_eq!(
        build_hrm_prompt(HrmMode::Verify, "INPUT"),
        "direct\nCheck for contradictions, missing evidence, and invalid assumptions.\nINPUT\nVerdict:"
    );
    assert_eq!(
        build_hrm_prompt(HrmMode::Plan, "INPUT"),
        "synth,cot\nDecompose this into executable steps.\nINPUT\nPlan:"
    );
    assert_eq!(
        build_hrm_prompt(HrmMode::Extract, "INPUT"),
        "direct\nReturn strict JSON only.\nINPUT\nJSON:"
    );
}

// ---------------------------------------------------------------------------
// Parsers (adversarial cases)
// ---------------------------------------------------------------------------

#[test]
fn verdict_parser_adversarial() {
    let cases: &[(&str, bool, usize, usize)] = &[
        ("PASS", true, 0, 0),
        ("Verdict: PASS", true, 0, 0),
        ("  fail  ", false, 0, 0),
        ("FAIL\nContradictions:\n- a\n- b", false, 2, 0),
        ("FAIL\nMissing facts:\n- deployment date", false, 0, 1),
        ("FAIL\nContradictions: x; y\nMissing facts: none", false, 2, 0),
        ("The passage discusses compasses.", false, 0, 0), // no verdict word
        ("", false, 0, 0),
        ("PASS\nContradictions:\n- none", true, 0, 0),
        ("fail.\ncontradictions:\n1. first\n2) second", false, 2, 0),
    ];
    for (text, pass, n_contra, n_missing) in cases {
        let v = parse_verdict(text);
        assert_eq!(v.pass, *pass, "pass mismatch for {text:?}");
        assert_eq!(v.contradictions.len(), *n_contra, "contradictions for {text:?}");
        assert_eq!(v.missing_facts.len(), *n_missing, "missing facts for {text:?}");
    }
}

#[test]
fn json_parser_adversarial() {
    assert!(parse_strict_json(r#"{"ok":1}"#).is_ok());
    assert!(parse_strict_json("```json\n{\"ok\":1}\n```").is_ok());
    assert!(parse_strict_json("prefix {\"a\":{\"b\":[1]}} suffix").is_ok());
    assert!(parse_strict_json(r#"{"s":"quote \" and { brace"}"#).is_ok());
    assert!(parse_strict_json("[]").is_ok());
    assert!(parse_strict_json("nothing here").is_err());
    assert!(parse_strict_json("{\"unterminated\": ").is_err());
}

#[tokio::test]
async fn extract_json_retries_once_then_succeeds() {
    let mock = MockBackend::new();
    mock.push_response("sorry, I cannot");
    mock.push_response(r#"{"fixed": true}"#);
    let builder = PromptBuilder::new(HrmMode::Extract);
    let v = extract_json_with_retry(&mock, &builder, "anything").await.unwrap();
    assert_eq!(v["fixed"], true);
    let reqs = mock.requests();
    assert_eq!(reqs.len(), 2, "exactly one retry");
    assert!(reqs[1].prompt.contains("was not valid JSON"));
}

// ---------------------------------------------------------------------------
// Router fallback
// ---------------------------------------------------------------------------

#[tokio::test]
async fn router_falls_back_to_keywords_on_garbage() {
    let mock = Arc::new(MockBackend::new());
    let router = HrmRouter::new(mock.clone());
    mock.push_response("hmm, tough one");
    let d = router.route("Search the knowledge base for similar incidents").await.unwrap();
    assert_eq!(d.route, Route::VectorSearch);
    assert_eq!(d.rationale, "rule-based keyword fallback");
}

// ---------------------------------------------------------------------------
// Controller: happy path + verify-fail retry
// ---------------------------------------------------------------------------

#[tokio::test]
async fn controller_happy_path() {
    let kernel = Arc::new(MockBackend::new());
    let executor = Arc::new(MockBackend::new());
    // Kernel: plan, routing, verify.
    kernel.push_response("1. read the cache config\n2. report the size");
    kernel.push_response("route=tool:calculator confidence=0.9 rationale=numbers");
    kernel.push_response("PASS");
    executor.push_response("The cache holds 100 entries.");

    let controller = Controller::new(kernel, executor.clone());
    let result = controller.run("How many entries does the cache hold?").await.unwrap();
    assert_eq!(result.answer, "The cache holds 100 entries.");
    assert!(result.plan.starts_with("1."));
    assert_eq!(result.route.route, Route::Tool("calculator".into()));
    assert!(result.verdict.pass);
    assert_eq!(result.attempts, 1);
}

#[tokio::test]
async fn controller_retries_once_on_failed_verification() {
    let kernel = Arc::new(MockBackend::new());
    let executor = Arc::new(MockBackend::new());
    // Kernel: plan, routing, verify #1 (FAIL), verify #2 (PASS).
    kernel.push_response("1. check\n2. answer");
    kernel.push_response("route=local_model:ruvltra confidence=0.95 rationale=simple");
    kernel.push_response("FAIL\nContradictions:\n- the count is wrong");
    kernel.push_response("PASS");
    executor.push_response("The cache holds 99 entries.");
    executor.push_response("The cache holds 100 entries.");

    let controller = Controller::new(kernel, executor.clone());
    let result = controller.run("How many entries does the cache hold?").await.unwrap();
    assert_eq!(result.attempts, 2);
    assert!(result.verdict.pass);
    assert_eq!(result.answer, "The cache holds 100 entries.");
    // The retry prompt carried the verdict's contradictions to the executor.
    let exec_reqs = executor.requests();
    assert_eq!(exec_reqs.len(), 2);
    assert!(exec_reqs[1].prompt.contains("failed verification"));
    assert!(exec_reqs[1].prompt.contains("the count is wrong"));
}

#[tokio::test]
async fn controller_gives_up_after_retry_budget() {
    let kernel = Arc::new(MockBackend::new());
    let executor = Arc::new(MockBackend::new());
    kernel.push_response("1. a\n2. b");
    kernel.push_response("route=local_model:ruvltra confidence=0.95 rationale=x");
    kernel.push_response("FAIL\nContradictions:\n- wrong");
    kernel.push_response("FAIL\nContradictions:\n- still wrong");
    executor.push_response("bad answer");
    executor.push_response("still bad");

    let result = Controller::new(kernel, executor)
        .run("anything at all")
        .await
        .unwrap();
    assert_eq!(result.attempts, 2);
    assert!(!result.verdict.pass);
    assert_eq!(result.verdict.contradictions, vec!["still wrong"]);
}

// ---------------------------------------------------------------------------
// Acceptance harness, cycle sweep, best-of-N (offline)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn acceptance_harness_passes_on_mock() {
    let backend: Arc<dyn LlmBackend> = Arc::new(MockBackend::new());
    let report = run_acceptance(backend).await.unwrap();
    assert!(report.total_tasks >= 100);
    assert_eq!(report.categories.len(), 5);
    for cat in &report.categories {
        assert!(cat.total >= 20, "{:?} below 20 tasks", cat.category);
        assert!(
            cat.passed,
            "{:?} failed: {}/{} (target {})",
            cat.category, cat.passed_tasks, cat.total, cat.target
        );
    }
    assert!(report.all_passed);
    // Serde-serializable report, printed for CI logs (no stray files).
    let json = serde_json::to_string_pretty(&report).unwrap();
    println!("=== mock acceptance report ===\n{json}");
    let round_trip: ruvllm_hrm::AcceptanceReport = serde_json::from_str(&json).unwrap();
    assert!(round_trip.all_passed);
}

#[tokio::test]
async fn cycle_sweep_report_shape() {
    let backend: Arc<dyn LlmBackend> = Arc::new(MockBackend::new());
    let tasks = ruvllm_hrm::benchmark::default_reasoning_tasks();
    let report = run_cycle_sweep(
        backend.clone(),
        &ruvllm_hrm::benchmark::DEFAULT_CYCLE_GRID,
        &tasks,
    )
    .await
    .unwrap();
    assert_eq!(report.entries.len(), 5);
    for entry in &report.entries {
        assert_eq!(entry.tasks, tasks.len());
        assert!((0.0..=1.0).contains(&entry.accuracy));
        // The mock is cycle-invariant and always correct on arithmetic.
        assert_eq!(entry.accuracy, 1.0);
    }
    assert!(serde_json::to_string(&report).is_ok());
    // Empty grids/tasks are rejected at the boundary.
    assert!(run_cycle_sweep(backend.clone(), &[], &tasks).await.is_err());
    assert!(run_cycle_sweep(backend, &[HrmCycles::new(1, 1)], &[]).await.is_err());
}

#[tokio::test]
async fn hrm_cycles_reach_the_backend() {
    let mock = Arc::new(MockBackend::new());
    let tasks = vec![ruvllm_hrm::ReasoningTask {
        problem: "What is 1 + 1?".to_string(),
        expected: 2,
    }];
    run_cycle_sweep(mock.clone(), &[HrmCycles::new(3, 4)], &tasks).await.unwrap();
    let reqs = mock.requests();
    assert_eq!(reqs.len(), 1);
    assert_eq!(reqs[0].hrm_cycles, Some(HrmCycles::new(3, 4)));
    assert_eq!(reqs[0].temperature, 0.0);
}

#[tokio::test]
async fn best_of_n_report_shape() {
    let kernel: Arc<dyn LlmBackend> = Arc::new(MockBackend::new());
    let executor: Arc<dyn LlmBackend> = Arc::new(MockBackend::new());
    let tasks = ruvllm_hrm::benchmark::default_reasoning_tasks();
    let report = run_best_of_n(kernel.clone(), executor.clone(), &[1, 4], &tasks)
        .await
        .unwrap();
    assert_eq!(report.entries.len(), 2);
    assert_eq!(report.entries[0].n, 1);
    assert_eq!(report.entries[1].n, 4);
    for e in &report.entries {
        assert!((0.0..=1.0).contains(&e.single_shot_accuracy));
        assert!((0.0..=1.0).contains(&e.majority_vote_accuracy));
        assert!((0.0..=1.0).contains(&e.kernel_scored_accuracy));
    }
    assert!(serde_json::to_string(&report).is_ok());
    assert!(run_best_of_n(kernel.clone(), executor.clone(), &[0], &tasks).await.is_err());
    assert!(run_best_of_n(kernel, executor, &[], &tasks).await.is_err());
}

#[tokio::test]
async fn best_of_n_kernel_scoring_rescues_bad_first_draft() {
    let kernel: Arc<dyn LlmBackend> = Arc::new(MockBackend::new());
    let executor = Arc::new(MockBackend::new());
    // One task, N=3: first draft wrong, two right. Majority and kernel
    // scoring recover; single-shot does not.
    executor.push_response("Answer: 41");
    executor.push_response("Answer: 42");
    executor.push_response("Answer: 42");
    let tasks = vec![ruvllm_hrm::ReasoningTask {
        problem: "What is 17 + 25?".to_string(),
        expected: 42,
    }];
    let report = run_best_of_n(kernel, executor, &[3], &tasks).await.unwrap();
    let entry = &report.entries[0];
    assert_eq!(entry.single_shot_accuracy, 0.0);
    assert_eq!(entry.majority_vote_accuracy, 1.0);
    assert_eq!(entry.kernel_scored_accuracy, 1.0);
}

// ---------------------------------------------------------------------------
// HrmTextBackend over a real (stub) HTTP server
// ---------------------------------------------------------------------------

/// Minimal HTTP/1.1 stub: serves one canned `(status, body)` per accepted
/// connection and records request bodies.
async fn spawn_http_stub(
    responses: Vec<(u16, String)>,
) -> (std::net::SocketAddr, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let bodies_clone = bodies.clone();
    tokio::spawn(async move {
        for (status, body) in responses {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            // Read headers, then the Content-Length body.
            let (header_end, mut have) = loop {
                let n = stream.read(&mut chunk).await.unwrap_or(0);
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&chunk[..n]);
                if let Some(pos) = find_subsequence(&buf, b"\r\n\r\n") {
                    break (pos + 4, buf.len());
                }
            };
            let headers = String::from_utf8_lossy(&buf[..header_end]).to_lowercase();
            let content_length: usize = headers
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0);
            while have - header_end < content_length {
                let n = stream.read(&mut chunk).await.unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                have = buf.len();
            }
            let body_received =
                String::from_utf8_lossy(&buf[header_end..header_end + content_length]).to_string();
            bodies_clone.lock().unwrap().push(body_received);
            let reason = if status == 200 { "OK" } else { "ERR" };
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.shutdown().await;
        }
    });
    (addr, bodies)
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[tokio::test]
async fn http_backend_generates_against_stub() {
    let canned = serde_json::json!({
        "choices": [{"text": "1. step one\n2. step two", "finish_reason": "stop"}]
    });
    let (addr, bodies) = spawn_http_stub(vec![(200, canned.to_string())]).await;
    let backend =
        HrmTextBackend::new(format!("http://{addr}/v1"), "sapientinc/HRM-Text-1B").unwrap();
    let mut req = GenerateRequest::new(build_hrm_prompt(HrmMode::Plan, "ship the release"));
    req.hrm_cycles = Some(HrmCycles::new(2, 2));
    let resp = backend.generate(req).await.unwrap();
    assert_eq!(resp.text, "1. step one\n2. step two");
    assert_eq!(resp.finish_reason.as_deref(), Some("stop"));

    // The wire body carried the model, prompt, and the R1 cycle override.
    let sent = bodies.lock().unwrap()[0].clone();
    let body: serde_json::Value = serde_json::from_str(&sent).unwrap();
    assert_eq!(body["model"], "sapientinc/HRM-Text-1B");
    assert_eq!(body["h_cycles"], 2);
    assert_eq!(body["l_cycles"], 2);
    assert!(body["prompt"].as_str().unwrap().starts_with("synth,cot\n"));
}

#[tokio::test]
async fn http_backend_retries_transient_500_then_succeeds() {
    let canned = serde_json::json!({"choices": [{"text": "PASS"}]});
    let (addr, _bodies) = spawn_http_stub(vec![
        (500, "{\"error\":\"boom\"}".to_string()),
        (200, canned.to_string()),
    ])
    .await;
    let backend = HrmTextBackend::new(format!("http://{addr}/v1"), "m").unwrap();
    let resp = backend.generate(GenerateRequest::new("direct\nx")).await.unwrap();
    assert_eq!(resp.text, "PASS");
}

#[tokio::test]
async fn http_backend_scores_via_logprobs_echo() {
    let canned = serde_json::json!({
        "choices": [{
            "text": "",
            "logprobs": {"token_logprobs": [null, -0.5, -1.5]}
        }]
    });
    let (addr, bodies) = spawn_http_stub(vec![(200, canned.to_string())]).await;
    let backend = HrmTextBackend::new(format!("http://{addr}/v1"), "m").unwrap();
    let resp = backend
        .score(ScoreRequest {
            prompt: "synth,cot\nProblem:\nx\nAnswer:".to_string(),
            completion: " 42".to_string(),
            hrm_cycles: None,
        })
        .await
        .unwrap();
    assert!((resp.score - (-2.0)).abs() < 1e-9);
    let sent: serde_json::Value =
        serde_json::from_str(&bodies.lock().unwrap()[0]).unwrap();
    assert_eq!(sent["max_tokens"], 0);
    assert_eq!(sent["echo"], true);
    assert_eq!(sent["logprobs"], 0);
}

#[tokio::test]
async fn http_backend_embed_state_is_unsupported() {
    let backend = HrmTextBackend::new("http://127.0.0.1:9/v1", "m").unwrap();
    let err = backend
        .embed_state(StateRequest { text: "z_H".to_string() })
        .await
        .unwrap_err();
    assert!(err.to_string().contains("Phase D"), "got: {err}");
}
