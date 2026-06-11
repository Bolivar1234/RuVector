//! HRM verifier pass (ADR-199 Verify mode): answer + evidence -> verdict.
//!
//! Live: `HRM_ENDPOINT=http://localhost:8000/v1 cargo run -p ruvllm-hrm --example verify`
//! Offline: `cargo run -p ruvllm-hrm --example verify -- --mock`

use ruvllm_hrm::{HrmTextBackend, HrmVerifier, LlmBackend, MockBackend};
use std::sync::Arc;

fn backend_from_args() -> anyhow::Result<Arc<dyn LlmBackend>> {
    if std::env::args().any(|a| a == "--mock") {
        eprintln!("[verify] using MockBackend (offline)");
        return Ok(Arc::new(MockBackend::new()));
    }
    let endpoint =
        std::env::var("HRM_ENDPOINT").unwrap_or_else(|_| "http://localhost:8000/v1".to_string());
    let model =
        std::env::var("HRM_MODEL").unwrap_or_else(|_| "sapientinc/HRM-Text-1B".to_string());
    eprintln!("[verify] using HrmTextBackend at {endpoint} (model {model})");
    Ok(Arc::new(HrmTextBackend::new(endpoint, model)?))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let backend = backend_from_args()?;
    let verifier = HrmVerifier::new(backend);

    let cases = [
        ("The cache is empty.", "The cache is not empty."),
        ("The server is online.", "The server is online."),
    ];
    for (claim, evidence) in cases {
        let verdict = verifier.verify(claim, &[], &[evidence.to_string()]).await?;
        println!("Claim:    {claim}");
        println!("Evidence: {evidence}");
        println!(
            "Verdict:  {} (contradictions: {:?}, missing facts: {:?})\n",
            if verdict.pass { "PASS" } else { "FAIL" },
            verdict.contradictions,
            verdict.missing_facts
        );
    }
    Ok(())
}
