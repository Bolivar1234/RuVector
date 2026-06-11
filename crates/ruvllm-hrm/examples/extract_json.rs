//! HRM strict-JSON extraction (ADR-199 Extract mode) with one corrective
//! retry on parse failure.
//!
//! Live: `HRM_ENDPOINT=http://localhost:8000/v1 cargo run -p ruvllm-hrm --example extract_json`
//! Offline: `cargo run -p ruvllm-hrm --example extract_json -- --mock`

use ruvllm_hrm::{extract_json_with_retry, HrmMode, HrmTextBackend, LlmBackend, MockBackend, PromptBuilder};
use std::sync::Arc;

fn backend_from_args() -> anyhow::Result<Arc<dyn LlmBackend>> {
    if std::env::args().any(|a| a == "--mock") {
        eprintln!("[extract_json] using MockBackend (offline)");
        return Ok(Arc::new(MockBackend::new()));
    }
    let endpoint =
        std::env::var("HRM_ENDPOINT").unwrap_or_else(|_| "http://localhost:8000/v1".to_string());
    let model =
        std::env::var("HRM_MODEL").unwrap_or_else(|_| "sapientinc/HRM-Text-1B".to_string());
    eprintln!("[extract_json] using HrmTextBackend at {endpoint} (model {model})");
    Ok(Arc::new(HrmTextBackend::new(endpoint, model)?))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let backend = backend_from_args()?;
    let input = std::env::args()
        .skip(1)
        .find(|a| a != "--mock")
        .unwrap_or_else(|| "fields: name=Ada; age=36; active=true".to_string());

    // Built-in Extract few-shot bank (model card: direct + 2-8 shots).
    let builder = PromptBuilder::new(HrmMode::Extract);
    let value = extract_json_with_retry(backend.as_ref(), &builder, &input).await?;

    println!("Input: {input}");
    println!("JSON:  {}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
