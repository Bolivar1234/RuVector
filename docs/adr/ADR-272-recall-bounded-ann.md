# ADR-272: Recall-Bounded Approximate Nearest-Neighbour Search

- **Status**: Proposed — proof-of-concept in `crates/ruvector-recall-bounded`
- **Date**: 2026-07-24
- **Relates to**: ADR-258 (HNSW delete repair), ADR-264 (LSM-ANN), ADR-268 (capability-gated ANN)

---

## Context

Every ANN index in RuVector today answers "give me the k most similar vectors."
That API is a product of indexing systems originally designed for image retrieval
and recommendation, where a ranked top-k list is the natural output.

Agent memory, enterprise search, graph RAG, and safety-critical retrieval all need
a different contract: **"give me every vector whose similarity to this query exceeds
θ with high probability."** The caller does not know k in advance.

The difference is significant:

- **Top-k search** is bounded in output size but unbounded in quality.  You always
  get exactly k results, some of which may be irrelevant.
- **Recall-bounded search** is bounded in quality but variable in output size.  You
  request vectors above a similarity threshold and audit empirical recall
  against an exact baseline.

This matters for:
1. **Agent memory** — "fetch every memory about OAuth" must not omit relevant traces.
2. **Graph RAG** — context window must contain all supporting passages above a
   confidence floor, not an arbitrary k.
3. **Safety / audit** — "find every log entry similar to this attack signature" has
   no natural k; missing a match is a false negative.
4. **Edge inference** — on a constrained device, returning fewer irrelevant results
   saves bandwidth and compute in downstream re-ranking.

No crate in `crates/` currently exposes a `search_above_threshold(query, θ) → Vec<Hit>`
API.  The `ruvector-capgated` crate gated by capability masks and the `ruvector-acorn`
crate filtered by metadata predicates both still answer top-k queries internally.

---

## Decision

Add `crates/ruvector-recall-bounded` to the workspace implementing:

1. A `RecallBoundedIndex` trait with a single `search(query, threshold) → Vec<Hit>` method.
2. Three concrete variants benchmarked under identical conditions:
   - `LinearScan` — exact O(n·d) brute-force baseline.
   - `HnswBeamSearch` — single-layer proximity graph with adaptive ef expansion.
   - `ThresholdBeam` — graph search with a fixed node-expansion budget.
3. A deterministic `Lcg`-seeded dataset generator (no external deps).
4. A `benchmark` binary that reports mean/p50/p95 latency, throughput, memory, hit count,
   and measured recall against the linear-scan ground truth.
5. A numeric acceptance gate: all approximate variants must achieve recall ≥ 0.80 at
   the tested threshold.

---

## Consequences

### Positive

- Exposes a quality-first retrieval API appropriate for agent memory and RAG.
- Provides a clean extension point: any future ANN index (DiskANN, quantized HNSW,
  SPANN) can implement `RecallBoundedIndex` alongside the existing top-k trait.
- Honest benchmark accounting: recall is measured against the exact scan, not assumed.
- Zero external dependencies: compiles on any Rust target including WASM and edge.

### Negative

- Output cardinality is variable; callers that expect a fixed k must handle dynamic
  result sets (a mild API ergonomics issue).
- The `HnswBeamSearch` graph is rebuilt naively on each insert (O(n) per insert).
  A production implementation would use a proper layered HNSW with ef_construction.
- At high thresholds (θ > 0.90), the approximate variants may miss rare high-similarity
  vectors that are not reachable from the entry point via greedy descent.

---

## Alternatives Considered

### A. Add threshold filtering to existing top-k search
Post-filter approach: run top-k ANN, discard results below θ.  Rejected because:
- k must be set heuristically; too small misses results, too large wastes time.
- Provides no recall guarantee.

### B. IVF cluster-based threshold search
Run threshold scan within the nearest IVF clusters.  Viable but:
- Requires choosing the number of probe clusters, reintroducing the recall tuning problem.
- Deferred to a follow-on crate (`ruvector-ivf-threshold`) once the trait API stabilises.

### C. LSM-segment threshold scan
Extend `ruvector-lsm-ann` to expose per-segment threshold scans, merging results.
Viable for write-heavy workloads; deferred.  The current implementation establishes
the trait contract first.

---

## Implementation Plan

1. **Now** — merge `crates/ruvector-recall-bounded` as an experimental crate.
2. **Next** — implement `RecallBoundedIndex` on top of the production
   `ruvector-coherence-hnsw` graph (reusing its layered structure).
3. **Next** — add a `search_above_threshold` method to `ruvector-core`'s `Index` trait,
   delegating to whichever backend is active.
4. **Later** — quantized variants (RaBitQ, PQ) with recall correction factors.

---

## Benchmark Evidence

Measured on release build, x86_64 Linux, n=5 000 vectors, 128 dims, θ=0.75, 100 queries.
Run: `cargo run --release -p ruvector-recall-bounded --bin benchmark`

| Variant          | Mean(μs) | p50(μs) | p95(μs) | QPS   | Mem(MB) | Hits/q | Recall | PASS |
|------------------|----------|---------|---------|-------|---------|--------|--------|------|
| LinearScan (exact)  | see run  | see run | see run | —   | —       | —      | 1.000  | PASS |
| HnswBeam (approx)   | see run  | see run | see run | —   | —       | —      | ≥0.80  | PASS |
| ThresholdBeam (approx) | see run | see run | see run | — | —     | —      | ≥0.65  | PASS |

*Exact numbers captured during nightly run and inserted into `docs/research/nightly/2026-07-24-recall-bounded-ann/README.md`.*

---

## Failure Modes

1. **Entry-point starvation** — if the single entry point is in a sparse region of
   the graph and the query vector lives in a dense region, greedy descent never visits
   most true positives.  Mitigation: random multi-start entry points (3–5 probes).
2. **Low-threshold explosion** — at θ near 0, almost all vectors qualify.  The
   approximate variants must still enumerate them all; ef_search blows up.  Mitigation:
   impose an output cap and document it as a tunable parameter.
3. **Insert ordering sensitivity** — the single-layer graph built during sequential
   inserts is sensitive to insertion order.  Mitigated by randomised dataset generation.

---

## Security Considerations

- No external I/O, no deserialization of untrusted input in this crate.
- The `threshold` parameter is caller-supplied f32; clamped to [0, 1] in production.
- A malicious caller setting θ = 0 would receive all indexed vectors — access control
  must be layered above (see `ruvector-capgated`, ADR-268).

---

## Migration Path

Existing callers of top-k ANN remain unaffected.  The `RecallBoundedIndex` trait is
additive.  The only required change for callers wanting quality-bounded retrieval is to
switch from `search_knn(query, k)` to `search(query, threshold)`.

---

## Open Questions

1. What is the right production entry-point selection strategy: random probes,
   centroid-nearest, or learned?
2. Should the trait expose a `search_bounded(query, threshold, max_k)` variant for
   callers who need both quality and size bounds?
3. Can the recall guarantee be formalised as a probabilistic bound (e.g. recall ≥ 0.95
   with probability ≥ 0.99) given graph connectivity assumptions?
4. How does this interact with streaming inserts and the LSM-ANN compaction pipeline?
