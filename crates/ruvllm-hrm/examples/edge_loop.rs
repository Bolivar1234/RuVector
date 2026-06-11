//! ADR-199 R4 edge-loop benchmark: the full governed loop
//! (plan -> route -> execute -> verify) under a memory/watt budget, emitting
//! the Pareto-chart data that gates any announcement.
//!
//! Against a live endpoint (`vllm serve sapientinc/HRM-Text-1B`):
//! ```bash
//! HRM_ENDPOINT=http://localhost:8000/v1 cargo run -p ruvllm-hrm --example edge_loop
//! ```
//! Offline, against the deterministic mock backend (kernel + executor),
//! with the default 8 W "pi5-proxy" power profile:
//! ```bash
//! cargo run -p ruvllm-hrm --example edge_loop -- --mock
//! ```
//!
//! Optional environment overrides:
//! - `EDGE_POWER_LABEL` / `EDGE_WATTS` — operator-declared power profile
//!   (no portable power sensor exists; e.g. Pi 5 ~8 W, M4 Pro ~30 W).
//! - `EDGE_MEMORY_GB` — peak-memory override; required off-Linux, and the
//!   right choice when the model serves from a separate vLLM process whose
//!   RSS this process cannot sample.

use ruvllm_hrm::{
    pareto_csv, EdgeLoopBenchmark, EdgePowerProfile, HrmTextBackend, LlmBackend, MockBackend,
};
use std::sync::Arc;

fn backends_from_args() -> anyhow::Result<(Arc<dyn LlmBackend>, Arc<dyn LlmBackend>)> {
    if std::env::args().any(|a| a == "--mock") {
        eprintln!("[edge_loop] using MockBackend for kernel + executor (offline)");
        return Ok((Arc::new(MockBackend::new()), Arc::new(MockBackend::new())));
    }
    let endpoint =
        std::env::var("HRM_ENDPOINT").unwrap_or_else(|_| "http://localhost:8000/v1".to_string());
    let model =
        std::env::var("HRM_MODEL").unwrap_or_else(|_| "sapientinc/HRM-Text-1B".to_string());
    eprintln!("[edge_loop] using HrmTextBackend at {endpoint} (model {model}) for kernel + executor");
    let backend: Arc<dyn LlmBackend> = Arc::new(HrmTextBackend::new(endpoint, model)?);
    // The kernel doubles as the executor here; production routes execution
    // to the workhorse model (RuvLTRA / Qwen / remote) per ADR-199 Layer 3.
    Ok((backend.clone(), backend))
}

fn power_from_env() -> anyhow::Result<EdgePowerProfile> {
    match std::env::var("EDGE_WATTS") {
        Ok(w) => EdgePowerProfile::new(
            std::env::var("EDGE_POWER_LABEL").unwrap_or_else(|_| "custom".to_string()),
            w.parse::<f64>()?,
        ),
        Err(_) => Ok(EdgePowerProfile::pi5_proxy()),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (kernel, executor) = backends_from_args()?;
    let power = power_from_env()?;
    let label = format!("hrm-edge-loop@{}", power.label);

    let mut bench = EdgeLoopBenchmark::new(kernel, executor, power);
    if let Ok(gb) = std::env::var("EDGE_MEMORY_GB") {
        bench = bench.with_memory_gb_override(gb.parse::<f64>()?);
    }

    let report = bench.run().await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    println!();
    println!("{}", pareto_csv(&[report.pareto_point(&label)]).trim_end());
    eprintln!(
        "\n[edge_loop] quality={:.3} edge_score={:.4} agent_score={:.4} \
         peak_memory_gb={:.4} under_2gb={} stable={}",
        report.quality,
        report.edge_score,
        report.agent_score,
        report.peak_memory_gb,
        report.under_2gb,
        report.stable
    );
    Ok(())
}
