//! ADR-199 "the one chart that gates any announcement" — Pareto data.
//!
//! A Pareto plot: x-axis latency (or watts), y-axis task quality; one point
//! per configuration (HRM-Text at each cycle setting, a Qwen small baseline,
//! a Llama small baseline, ruvLLM routed mode). The win condition is not
//! highest quality — it is a **new Pareto point**: better quality at the
//! same edge budget, or the same quality at lower cost.
//!
//! [`pareto_csv`] is the plot-ready export; [`pareto_frontier`] computes the
//! non-dominated set on the (latency, quality) axes.

use serde::{Deserialize, Serialize};

/// One point on the edge Pareto chart. Produced per benchmark run via
/// [`crate::edge::EdgeLoopReport::pareto_point`]; baseline points (Qwen,
/// Llama, cycle settings) are built the same way from their own runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoPoint {
    /// Configuration label, e.g. `hrm-2x2`, `qwen2.5-1.5b`, `pi5-routed`.
    pub label: String,
    /// Mean end-to-end governed-loop latency in milliseconds.
    pub latency_ms: f64,
    /// Operator-declared power budget (see [`crate::edge::EdgePowerProfile`]).
    pub watts: f64,
    /// Fraction of tasks fully passing the loop, in `[0, 1]`.
    pub quality: f64,
    /// Peak memory footprint in GB (measured RSS or operator-supplied).
    pub memory_gb: f64,
    /// `quality / (latency_s × watts × memory_gb)` ([`crate::scoring::edge_score`]).
    pub edge_score: f64,
}

/// Column header of [`pareto_csv`].
pub const PARETO_CSV_HEADER: &str = "label,latency_ms,watts,quality,memory_gb,edge_score";

/// Render points as plot-ready CSV (header + one line per point).
///
/// Numeric fields use Rust's shortest-round-trip `f64` formatting, so a
/// parsed-back value is bit-identical. Labels are sanitized: `,` and
/// newlines become `;` / space so a label can never break the row format.
pub fn pareto_csv(points: &[ParetoPoint]) -> String {
    let mut out = String::from(PARETO_CSV_HEADER);
    out.push('\n');
    for p in points {
        let label: String = p
            .label
            .chars()
            .map(|c| match c {
                ',' => ';',
                '\n' | '\r' => ' ',
                c => c,
            })
            .collect();
        out.push_str(&format!(
            "{label},{},{},{},{},{}\n",
            p.latency_ms, p.watts, p.quality, p.memory_gb, p.edge_score
        ));
    }
    out
}

/// `a` dominates `b` iff `a` is no worse on both chart axes (latency lower
/// or equal AND quality higher or equal) and strictly better on at least one.
fn dominates(a: &ParetoPoint, b: &ParetoPoint) -> bool {
    a.latency_ms <= b.latency_ms
        && a.quality >= b.quality
        && (a.latency_ms < b.latency_ms || a.quality > b.quality)
}

/// Non-dominated subset of `points` on the (latency, quality) axes —
/// lower latency AND higher quality win.
///
/// Tie handling: points with identical (latency, quality) do not dominate
/// each other, so *all* of them are retained (distinct configurations that
/// land on the same spot are equally announce-worthy). Input order is
/// preserved. Points with NaN coordinates never compare as dominated and
/// are therefore retained — validate measurements upstream.
pub fn pareto_frontier(points: &[ParetoPoint]) -> Vec<ParetoPoint> {
    points
        .iter()
        .filter(|p| !points.iter().any(|q| dominates(q, p)))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(label: &str, latency_ms: f64, quality: f64) -> ParetoPoint {
        ParetoPoint {
            label: label.to_string(),
            latency_ms,
            watts: 8.0,
            quality,
            memory_gb: 1.0,
            edge_score: 0.0,
        }
    }

    #[test]
    fn frontier_drops_dominated_points() {
        let pts = vec![
            point("fast-good", 100.0, 0.9),   // frontier
            point("slow-worse", 200.0, 0.8),  // dominated by fast-good
            point("slow-best", 300.0, 0.95),  // frontier (highest quality)
            point("fast-bad", 50.0, 0.5),     // frontier (lowest latency)
            point("mid-dominated", 120.0, 0.85), // dominated by fast-good
        ];
        let frontier = pareto_frontier(&pts);
        let labels: Vec<&str> = frontier.iter().map(|p| p.label.as_str()).collect();
        assert_eq!(labels, ["fast-good", "slow-best", "fast-bad"]);
    }

    #[test]
    fn frontier_keeps_ties_and_handles_edges() {
        // Identical (latency, quality): mutually non-dominated, both kept.
        let pts = vec![point("a", 100.0, 0.9), point("b", 100.0, 0.9)];
        assert_eq!(pareto_frontier(&pts).len(), 2);
        // Same latency, different quality: only the better one survives.
        let pts = vec![point("a", 100.0, 0.9), point("b", 100.0, 0.8)];
        let frontier = pareto_frontier(&pts);
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].label, "a");
        // Single point is always the frontier; empty input is empty.
        assert_eq!(pareto_frontier(&[point("solo", 1.0, 0.1)]).len(), 1);
        assert!(pareto_frontier(&[]).is_empty());
    }

    #[test]
    fn csv_format_and_label_sanitization() {
        let mut p = point("hrm,2x2\nmock", 123.5, 0.95);
        p.edge_score = 0.0125;
        let csv = pareto_csv(&[p]);
        let mut lines = csv.lines();
        assert_eq!(lines.next(), Some(PARETO_CSV_HEADER));
        assert_eq!(lines.next(), Some("hrm;2x2 mock,123.5,8,0.95,1,0.0125"));
        assert_eq!(lines.next(), None);
        // Header-only when there are no points.
        assert_eq!(pareto_csv(&[]), format!("{PARETO_CSV_HEADER}\n"));
    }

    #[test]
    fn csv_numbers_round_trip_exactly() {
        let mut p = point("x", 0.1 + 0.2, 1.0 / 3.0);
        p.edge_score = 7.123456789e-3;
        let csv = pareto_csv(&[p.clone()]);
        let row = csv.lines().nth(1).unwrap();
        let cols: Vec<&str> = row.split(',').collect();
        assert_eq!(cols.len(), 6);
        assert_eq!(cols[1].parse::<f64>().unwrap(), p.latency_ms);
        assert_eq!(cols[3].parse::<f64>().unwrap(), p.quality);
        assert_eq!(cols[5].parse::<f64>().unwrap(), p.edge_score);
    }
}
