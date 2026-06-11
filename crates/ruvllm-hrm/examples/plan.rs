//! HRM planning pass (ADR-199 Plan mode).
//!
//! Against a live endpoint (`vllm serve sapientinc/HRM-Text-1B`):
//! ```bash
//! HRM_ENDPOINT=http://localhost:8000/v1 cargo run -p ruvllm-hrm --example plan
//! ```
//! Offline, against the deterministic mock backend:
//! ```bash
//! cargo run -p ruvllm-hrm --example plan -- --mock
//! ```

use ruvllm_hrm::{HrmMode, HrmTextBackend, LlmBackend, MockBackend, PromptBuilder};
use std::sync::Arc;

fn backend_from_args() -> anyhow::Result<Arc<dyn LlmBackend>> {
    if std::env::args().any(|a| a == "--mock") {
        eprintln!("[plan] using MockBackend (offline)");
        return Ok(Arc::new(MockBackend::new()));
    }
    let endpoint =
        std::env::var("HRM_ENDPOINT").unwrap_or_else(|_| "http://localhost:8000/v1".to_string());
    let model =
        std::env::var("HRM_MODEL").unwrap_or_else(|_| "sapientinc/HRM-Text-1B".to_string());
    eprintln!("[plan] using HrmTextBackend at {endpoint} (model {model})");
    Ok(Arc::new(HrmTextBackend::new(endpoint, model)?))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let backend = backend_from_args()?;
    let task = std::env::args()
        .skip(1)
        .find(|a| a != "--mock")
        .unwrap_or_else(|| "Migrate the user database from MySQL to Postgres without downtime".to_string());

    let builder = PromptBuilder::bare(HrmMode::Plan);
    let resp = backend.generate(builder.to_request(&task)).await?;

    println!("Task: {task}\n");
    println!("Plan:\n{}", resp.text.trim());
    println!("\n(latency: {} ms)", resp.latency_ms);
    Ok(())
}
