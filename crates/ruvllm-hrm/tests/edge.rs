//! Full offline run of the ADR-199 R4 edge-loop benchmark on `MockBackend`:
//! report completeness, product-gate fields, composite scores, and the
//! Pareto point/CSV round trip.

use ruvllm_hrm::{
    pareto_csv, EdgeLoopBenchmark, EdgeLoopReport, EdgePowerProfile, MemorySampler, MockBackend,
    PARETO_CSV_HEADER,
};
use std::sync::Arc;

fn offline_benchmark() -> EdgeLoopBenchmark {
    let kernel = Arc::new(MockBackend::new());
    let executor = Arc::new(MockBackend::new());
    let bench = EdgeLoopBenchmark::new(kernel, executor, EdgePowerProfile::pi5_proxy());
    // Linux samples this process's RSS; elsewhere the report must carry an
    // operator-supplied footprint (documented MemorySampler contract).
    if MemorySampler::rss_bytes().is_none() {
        bench.with_memory_gb_override(1.0)
    } else {
        bench
    }
}

#[tokio::test]
async fn edge_loop_runs_offline_end_to_end() {
    let report = offline_benchmark().run().await.expect("offline run succeeds");

    // Report completeness: every default task produced a record.
    assert_eq!(report.total_tasks, 20);
    assert_eq!(report.records.len(), 20);
    assert!(report.records.iter().all(|r| r.latency_ms >= 0.0));
    assert!(
        report.records.iter().all(|r| r.attempts >= 1),
        "every governed loop run executes at least once"
    );

    // Product gates: stable execution and the strict 2 GB memory gate.
    assert!(report.stable, "no task may error on the mock");
    assert!(report.records.iter().all(|r| r.error.is_none()));
    assert!(report.peak_memory_gb > 0.0);
    assert_eq!(report.under_2gb, report.peak_memory_gb < 2.0);

    // The deterministic mock solves the whole default set.
    assert_eq!(report.quality, 1.0, "mock run must fully pass: {report:#?}");
    assert!(report
        .records
        .iter()
        .filter(|r| r.kind == "routing")
        .all(|r| r.route_correct == Some(true)));
    assert!(report
        .records
        .iter()
        .filter(|r| r.kind == "json_extraction")
        .all(|r| r.json_valid == Some(true)));

    // Composite scores: quality and cost published together.
    assert!(report.edge_score > 0.0);
    assert!((0.0..=1.0).contains(&report.agent_score));

    // Latency stats are ordered and internally consistent.
    assert!(report.latency_p50_ms <= report.latency_p95_ms);
    assert!(report.latency_mean_ms > 0.0);

    // Serde round trip keeps the report a stable CI artifact. Floats use a
    // relative tolerance: serde_json's default float parsing (without the
    // `float_roundtrip` feature) may be off by one ulp.
    let json = serde_json::to_string(&report).unwrap();
    let back: EdgeLoopReport = serde_json::from_str(&json).unwrap();
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs());
    assert_eq!(back.total_tasks, report.total_tasks);
    assert!(close(back.edge_score, report.edge_score));
    assert!(close(back.agent_score, report.agent_score));
    assert_eq!(back.stable, report.stable);
    assert_eq!(back.under_2gb, report.under_2gb);
}

#[tokio::test]
async fn pareto_point_and_csv_round_trip() {
    let report = offline_benchmark().run().await.unwrap();
    let point = report.pareto_point("hrm-mock");

    assert_eq!(point.label, "hrm-mock");
    assert_eq!(point.latency_ms, report.latency_mean_ms);
    assert_eq!(point.watts, report.power_profile.watts);
    assert_eq!(point.quality, report.quality);
    assert_eq!(point.memory_gb, report.peak_memory_gb);
    assert_eq!(point.edge_score, report.edge_score);

    let csv = pareto_csv(std::slice::from_ref(&point));
    let mut lines = csv.lines();
    assert_eq!(lines.next(), Some(PARETO_CSV_HEADER));
    let row = lines.next().expect("one data row");
    assert_eq!(lines.next(), None);

    // f64 Display is shortest-round-trip, so parsed values are bit-exact.
    let cols: Vec<&str> = row.split(',').collect();
    assert_eq!(cols.len(), 6);
    assert_eq!(cols[0], point.label);
    assert_eq!(cols[1].parse::<f64>().unwrap(), point.latency_ms);
    assert_eq!(cols[2].parse::<f64>().unwrap(), point.watts);
    assert_eq!(cols[3].parse::<f64>().unwrap(), point.quality);
    assert_eq!(cols[4].parse::<f64>().unwrap(), point.memory_gb);
    assert_eq!(cols[5].parse::<f64>().unwrap(), point.edge_score);
}
