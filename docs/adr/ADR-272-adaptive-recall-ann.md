# ADR-272: Adaptive Recall-Targeted ANN Search

**Status**: Proposed  
**Date**: 2026-07-23  
**Author**: Nightly Research Agent  
**Branch**: `research/nightly/2026-07-23-adaptive-recall-ann`  
**Crate**: `crates/ruvector-adaptive-ann`  
**Related**: ADR-240 (Coherence-HNSW), ADR-256 (Hybrid Search), ADR-264 (LSM-ANN), ADR-268 (Capability-Gated ANN)

---

## Context

Every HNSW deployment requires tuning the `ef_search` (beam width) parameter. Too
small and recall collapses; too large and latency blows up. Finding the right value
requires offline profiling, and the optimal ef shifts whenever the data distribution
changes — new agent memory, shifted query embeddings, or seasonal corpus drift.

Current state of the ecosystem:

- **All prior RuVector ANN crates** expose `ef` as a caller-controlled parameter.
  There is no mechanism to auto-select ef given a recall target.
- **Qdrant, Milvus, FAISS, pgvector**: all expose ef (or similar) as a raw search
  parameter. None perform automatic calibration to a recall target.
- **Agent memory workloads** care deeply about recall: retrieving only 7 of 10
  relevant memories loses 30% of context quality. But agents cannot tune ef.
- **MCP tool surfaces**: a search tool that targets a latency/recall trade-off while measuring
  recall quality is more useful than one the operator must tune.
- **Edge deployments**: ef tuning requires hardware profiling. Edge devices need
  self-calibrating parameters.

This ADR proposes introducing an adaptive recall-targeted search layer: the caller
specifies `recall_target` (e.g., 0.90) and the system automatically selects the
estimated ef for that calibrated workload and `k`. Three strategies are provided: fixed-ef (baseline),
binary-search calibrated (oracle), and table-calibrated (production).

---

## Decision

Introduce `crates/ruvector-adaptive-ann` with the `RecallTargetedSearch` trait and
three implementors. The core innovation is a `CalibrationTable` that maps ef →
mean_recall@k on a held-out query sample drawn from the same distribution as
production queries, enabling O(1) ef selection at query time.

The key lesson from the PoC: **calibration queries must match the production query
distribution**. Calibrating on graph-member vectors (in-distribution) and testing
on random queries (out-of-distribution) causes the table to under-estimate ef needs.
The PoC makes this explicit and the production API enforces that calibration data
comes from a user-provided held-out sample.

---

## Consequences

### Positive

- Operators specify **recall targets**, not ef values — dramatically simpler API.
- Self-calibrating: re-run calibration after data distribution shifts.
- TableCalibrated provides O(1) ef selection with amortised calibration overhead.
- BinarySearchCalibrated provides a per-query oracle comparison (with ground
  truth access — useful for offline quality audits).
- Directly composable with coherence-hnsw, capability-gated, and hybrid search.

### Negative

- Calibration requires a representative held-out query sample (20–100 queries).
- Calibration overhead is proportional to |ef_candidates| × n_sample × search cost.
- Calibration values are empirical estimates: the table records mean recall,
  not a probabilistic bound or a per-query guarantee.
- Distribution drift invalidates the table: periodic re-calibration is required.

---

## Alternatives Considered

1. **Static ef lookup table** (per graph size): too coarse, ignores query distribution.
2. **Learned ef predictor** (neural): too heavy for edge, requires labelled training data.
3. **Dynamic ef with early-exit** (stopping when distance improvement < ε): does not
   provide a recall bound; was rejected as a separate research direction.
4. **Percentile-based ef selection**: equivalent to TableCalibrated but harder to
   reason about; rejected in favour of direct recall measurement.

---

## Implementation Plan

### Phase 1 (This ADR — PoC)

- `crates/ruvector-adaptive-ann`: flat proximity graph + three search variants.
- `RecallTargetedSearch` trait with `search_with_target(query, k, recall_target)`.
- `CalibrationTable` built from held-out random queries.
- 7 integration tests + benchmark binary with all acceptance tests passing.

### Phase 2 (Production hardening)

- Integrate `RecallTargetedSearch` into `ruvector-coherence-hnsw` (multi-layer HNSW).
- Online recalibration: background thread re-runs calibration after N inserts.
- Percentile recall tracking (p50/p95) not just mean.
- Expose via `ruvector-server` REST API: `ef_search` becomes optional when
  `recall_target` is provided.

### Phase 3 (Research)

- Learned calibration: train a lightweight ridge regression mapping
  (graph_density, query_norm, dataset_n, dims) → optimal_ef.
- Per-cluster calibration: different clusters may need different ef values.
- Distribution drift detection: alert when calibration table is stale.

---

## Benchmark Evidence

Collected 2026-07-23 on x86_64 Linux, release build, `cargo run --release`.  
Dataset: 3,000 vectors × 64 dims, 10 clusters, σ=0.20.  
Graph: M=16, M_longjump=6, fixed entry node 0.  
Calibration: 100 held-out random queries, 10 ef candidates [10, 16, 24, 32, 48, 64, 96, 128, 192, 256].

| Variant | Recall@10 | Mean µs | p50 µs | p95 µs | QPS |
|---------|-----------|---------|--------|--------|-----|
| FixedEf(64) | 0.778 | 105.3 | 102.7 | 130.4 | 9,497 |
| BinarySearchCalibrated | 0.902 | 1,355.0 | 1,258.8 | 2,094.8 | 738 |
| TableCalibrated | 0.940 | 227.8 | 225.4 | 267.7 | 4,390 |

Calibration table (ef → mean_recall@10 on held-out random queries):

```
ef=  10  recall=0.378
ef=  16  recall=0.457
ef=  24  recall=0.550
ef=  32  recall=0.616
ef=  48  recall=0.693
ef=  64  recall=0.751
ef=  96  recall=0.838
ef= 128  recall=0.883
ef= 192  recall=0.927
ef= 256  recall=0.953
```

Table selected ef=192 for recall_target=0.90. TableCalibrated achieved 0.940 recall,
exceeding the target. BinarySearch oracle achieved 0.902 with mean ef=103.8 (per query
calibration finds tighter ef per query vs. the table's conservative worst-case choice).

---

## Failure Modes

1. **Distribution mismatch**: calibration on in-distribution queries (e.g., graph
   members) overfits to easy cases; random queries need 5× wider ef. **Mitigation**:
   require calibration queries from the production distribution.
2. **Stale calibration**: data distribution shifts post-calibration; table becomes
   inaccurate. **Mitigation**: re-calibrate after every K inserts.
3. **Small calibration sample**: high variance in recall estimates → wrong ef chosen.
   **Mitigation**: use n_sample ≥ 50 (100 recommended).
4. **Flat graph entry bias**: with a fixed entry node, recall depends on entry
   distance to query. A multi-layer HNSW upper layer eliminates this.

---

## Security Considerations

- No external input processed in calibration path.
- Calibration data may encode query patterns; avoid logging calibration queries in
  security-sensitive deployments.
- Recall guarantees are statistical; do not use for cryptographic or safety-critical
  retrieval where exact recall must be verified.

---

## Migration Path

- Existing callers using raw `ef` continue to work: `FixedEfSearch` implements the
  same beam search with no change to the call site.
- New callers adopt `RecallTargetedSearch::search_with_target` and provide calibration
  data once at startup (or after significant data changes).
- Server API: `ef_search` parameter remains; `recall_target` is a new optional
  parameter that, when present, overrides `ef_search` via the table.

---

## Open Questions

1. Should calibration be per-collection or per-index-partition?
2. How frequently should online recalibration run in agent memory workloads?
3. Can we use the BinarySearch oracle as a quality monitor in production (sampling
   1% of queries for offline recall audit)?
4. Should the calibration table be persisted to the RVF manifest?
5. For multi-layer HNSW, does calibration need separate tables per layer?
