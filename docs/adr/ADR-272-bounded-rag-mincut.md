# ADR-272: Bounded Context RAG via MinCut Graph Partitioning

- **Status**: Proposed
- **Date**: 2026-07-25
- **Crate**: `crates/ruvector-bounded-rag`
- **Branch**: `research/nightly/2026-07-25-bounded-rag-mincut`
- **Connects**: vector search, ruvector-mincut, ruvector-graph, ruvector-agent-memory, RAG safety

---

## Context

Standard RAG retrieves the top-k chunks by vector cosine similarity and feeds all of them into a language model's context window. This is correct when the k-nearest neighbours form a coherent set, but breaks in two common failure modes:

1. **Semantic scatter**: chunks from different topics land near the query centroid, so the retrieved set is incoherent even though each chunk individually scores well.
2. **Budget overrun**: a fixed-k policy either wastes tokens (when fewer chunks suffice) or truncates (when the coherent cluster is larger than k).

Agent memory systems have a third failure: accumulated memories from many sessions may share surface-level similarity to a query while being contextually unrelated. Returning them degrades inference quality and increases hallucination risk.

The hypothesis behind this ADR is that graph coherence can improve a
budget-capped context set compared with top-k retrieval alone.

Graph min-cut provides one measurable mechanism. If chunks are nodes and
similarity edges connect them, a min-cut between a query-seeded source
partition and a noise sink separates a candidate cluster from noise. The
implementation then ranks and truncates that partition to the budget, so the
final set is not the solution to a budget-constrained min-cut objective.

---

## Decision

Add `ruvector-bounded-rag` to the workspace as a standalone research crate implementing three retrieval strategies with a shared `BoundedRetriever` trait:

1. **TopK**: cosine rank, no graph, O(n log n) — baseline.
2. **GraphBFS**: rebuild a dense similarity graph in O(n²·d), then run priority-queue traversal bounded by coherence threshold and budget.
3. **MinCutBounded**: Edmonds-Karp max-flow/min-cut on a source-sink flow network built from the chunk graph, followed by relevance ranking and budget truncation; requires pre-filtering at large scale.

All three share `RetrieverConfig { budget, edge_threshold, seed_threshold }`.

The flow network for MinCutBounded:
- Source → chunk edges with capacity = cosine(query, chunk)
- Chunk → sink edges with capacity = 1 − cosine(query, chunk)
- Inter-chunk edges with capacity = cosine(chunk_i, chunk_j) × scale
- BFS residual graph traversal finds the source-side partition

---

## Consequences

### Positive

- MinCutBounded provides the tightest coherence guarantee of the three strategies.
- GraphBFS has a simpler traversal than MinCut, but this PoC still pays an O(n²·d) graph-build cost on every query.
- All three use the same `BoundedRetriever` trait, making them drop-in swappable.
- The budget parameter directly controls context window consumption.
- Proof-gated integration is straightforward: add a pre-filter step using `ruvector-proof-gate` to remove chunks the requester lacks access to before running the retriever.

### Negative

- MinCutBounded builds an O(n²) similarity matrix (naive implementation). At n=3000, the full matrix takes ~78 seconds per query. **This is a PoC limitation**, not a design limitation.
- GraphBFS at n=3000 takes ~112ms per query; acceptable for offline indexing but not live query paths.
- The current implementation computes all pairwise similarities at query time. A production path would pre-build a k-NN graph offline and run retrieval on the stored graph.

---

## Alternatives Considered

1. **Re-ranker chain**: Apply a cross-encoder re-ranker after top-k retrieval to filter incoherent results. Simpler but doesn't provide a principled coherence boundary; doesn't model inter-chunk relationships.

2. **MMR (Maximal Marginal Relevance)**: Classic diversity-aware retrieval. Reduces redundancy but doesn't enforce coherence; tends to over-diversify rather than finding the maximal coherent cluster.

3. **Clustering-then-retrieve**: Cluster the corpus offline, then retrieve only from the cluster closest to the query. Faster at query time but loses fidelity when queries straddle cluster boundaries.

4. **Coherence HNSW search** (ADR-268 predecessor): Adds coherence scoring to HNSW graph traversal. Complementary to this work — coherence-HNSW finds candidates; MinCut post-processes them for budget bounding.

---

## Implementation Plan

### Phase 1 (this ADR): PoC — done
- [x] Trait `BoundedRetriever` with shared config
- [x] `TopKRetriever` baseline
- [x] `GraphBfsRetriever` with priority-queue BFS
- [x] `MinCutRetriever` with Edmonds-Karp max-flow
- [x] 9 unit tests + 1 numeric acceptance test
- [x] Release benchmark binary with 3 dataset sizes
- [x] All tests green, all benchmarks passed acceptance threshold

### Phase 2: Production hardening
- [ ] Pre-build k-NN graph offline (approximate) — O(n log n) build, O(k log n) query
- [ ] Cache normalised chunk vectors
- [ ] Apply MinCut only to HNSW pre-filter output (~50–100 candidates), not full corpus
- [ ] Integrate with `ruvector-proof-gate` for access-controlled chunk pre-filtering
- [ ] Add streaming mode: expand BFS incrementally as the LLM consumes context

### Phase 3: Ecosystem integration
- [ ] MCP tool surface: `bounded_retrieve(query, budget, config)` returning chunk IDs + scores
- [ ] `ruFlo` node for adaptive budget control based on downstream model feedback
- [ ] WASM build for edge deployments (GraphBFS only; MinCut too expensive for edge)
- [ ] `ruvector-agent-memory` integration: session memories as labelled chunk graph

---

## Benchmark Evidence

Hardware: x86_64 Linux, Rust 1.77, `cargo run --release -p ruvector-bounded-rag --bin benchmark`

| Variant | n | Dim | Queries | Mean(μs) | p50(μs) | p95(μs) | QPS | Precision | BudgetUtil |
|---------|---|-----|---------|----------|---------|---------|-----|-----------|------------|
| TopK | 200 | 64 | 50 | 31.6 | 30.0 | 36.0 | 30,941 | 1.000 | 1.000 |
| GraphBFS | 200 | 64 | 50 | 1,256 | 1,238 | 1,367 | 795 | 1.000 | 1.000 |
| MinCutBounded | 200 | 64 | 50 | 2,707 | 2,688 | 3,023 | 369 | 1.000 | 1.000 |
| TopK | 1,000 | 64 | 100 | 175.5 | 156.0 | 245.0 | 5,666 | 1.000 | 1.000 |
| GraphBFS | 1,000 | 64 | 100 | 29,063 | 28,766 | 30,301 | 34 | 1.000 | 1.000 |
| MinCutBounded | 1,000 | 64 | 100 | 78,533 | 78,222 | 82,747 | 13 | 1.000 | 1.000 |
| TopK | 3,000 | 32 | 30 | 302.3 | 294.0 | 349.0 | 3,296 | 1.000 | 1.000 |
| GraphBFS | 3,000 | 32 | 30 | 112,589 | 111,121 | 124,294 | 9 | 1.000 | 1.000 |
| MinCutBounded | 3,000 | 32 | 30 | 1,270,130 | 1,262,693 | 1,379,184 | 1 | 1.000 | 1.000 |

All three variants achieve precision = 1.000 on well-separated synthetic clusters. Acceptance threshold: precision ≥ 0.70. Result: **PASS**.

**Key observation**: The O(n²) full pairwise graph construction dominates both GraphBFS and MinCut at n≥1000. The proposed Phase 2 fix (pre-built k-NN graph + HNSW pre-filter output) reduces the effective n from corpus size (~1000+) to candidate set size (~50–100), making MinCut viable in production.

---

## Failure Modes

1. **Disconnected corpus**: If edge_threshold is too high, the chunk graph is sparse and GraphBFS returns only seeds. Mitigation: lower edge_threshold or fall back to TopK.
2. **Over-cutting**: If query similarity is low for all chunks, the source partition is empty. Mitigation: seed_threshold fallback to top-1 ensures at least one seed.
3. **Budget underrun**: MinCut may return fewer chunks than budget when the coherent partition is small. This is correct behaviour — the budget is a maximum, not a target.
4. **Adversarial insertion**: A malicious chunk with artificially high similarity to many query topics could bridge otherwise disjoint clusters. Mitigation: proof-gate pre-filter from `ruvector-proof-gate`.
5. **Scalability cliff**: Full pairwise similarity at n=3000 takes 1.27 seconds. At n=10,000 this becomes ~14 seconds, unusable. The Phase 2 k-NN graph fix is essential before production.

---

## Security Considerations

- The flow network is built from query-time data only; no persistent secrets are embedded.
- Chunk labels are used for evaluation but not for routing in production mode.
- Proof-gate integration (Phase 2) would filter chunks before graph construction, preventing side-channel leakage through graph structure.
- Budget bounding reduces prompt injection surface: a malicious chunk that grows the context beyond budget is simply cut off at the partition boundary.

---

## Migration Path

This crate is additive. No existing crate is modified. Integration with `ruvector-agent-memory` and `ruvector-proof-gate` is proposed for Phase 2 via optional dependencies.

The `BoundedRetriever` trait is the stable API surface. `RetrieverConfig` fields may gain defaults but will not lose existing fields in a breaking change.

---

## Open Questions

1. What is the optimal k for the pre-built k-NN graph in Phase 2? Expected range: k=15–32 (HNSW-style).
2. Should inter-chunk edges be similarity or dissimilarity? Current implementation uses similarity as capacity, meaning high-similarity edges are harder to cut — this is correct for coherence preservation.
3. Can the flow network be updated incrementally when new chunks are inserted (online indexing)?
4. What is the right integration point with `ruFlo`: should budget be a static config or a dynamic signal from the downstream model?
5. Should MinCut use directed or undirected capacities? Current implementation uses undirected (bi-directional edges). Directed edges from source to sink only would change the partition semantics.
