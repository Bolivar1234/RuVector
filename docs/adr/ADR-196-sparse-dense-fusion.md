# ADR-196: Hybrid Sparse-Dense Vector Search — BM25 + Dense ANN via RRF and Cascade

**Date**: 2026-06-04  
**Status**: Accepted  
**Deciders**: nightly research agent  
**Branch**: `research/nightly/2026-06-04-sparse-dense-fusion`

---

## Context

RuVector provides multiple dense ANN backends (DiskANN, RaBitQ) and a filtered search layer
(ACORN). It does not currently offer sparse retrieval or a hybrid fusion layer. Production
retrieval pipelines at scale (e.g., Bing, Elasticsearch, Pinecone) uniformly combine a
sparse lexical signal (BM25 / learned sparse) with a dense semantic signal. The gap means
RuVector cannot serve keyword-sensitive workloads without an external sparse index.

Two complementary fusion strategies are established in the literature:

- **Reciprocal Rank Fusion (RRF)** — rank-based merge, no score calibration needed, trivial
  to implement, widely deployed (Cormack et al., 2009).
- **Cascade reranking** — sparse first-stage (cheap, fast) → dense reranker (expensive, precise),
  dominant in production pipelines since at least Matveeva et al. (2006).

Both strategies are implementable as pure Rust without changes to existing ANN indexes.

---

## Decision

We implement `ruvector-hybrid-search` as a new workspace crate providing:

1. **`SparseInvertedIndex`** — BM25 scoring with Robertson IDF, inverted posting lists.
2. **`DenseLinearSearcher`** — brute-force cosine NN (ground-truth baseline for benchmarking).
3. **`HybridRrfSearcher`** — runs dense and sparse independently, fuses via RRF (k=60).
4. **`HybridCascadeSearcher`** — sparse candidates (C = min(k·20, N)) → dense rerank.
5. **`SyntheticDataset`** — holdout benchmark generator with cluster-correlated dense + sparse features.
6. **Benchmark binary** — reproducible holdout evaluation producing recall@10 and latency metrics.

The `HybridSearcher` trait provides a common `index` / `search` interface so all four backends
are pluggable. BM25 parameters (`k1`, `b`) are exposed via `Bm25Params`.

---

## Rationale

### Why RRF over score normalisation?

Score normalisation (e.g., min-max scaling) requires knowing the score range of each backend,
which is dataset-dependent. RRF works on ranks only — no calibration, robust to score range
differences, competitive recall in ablation studies. It is the right default for a research
baseline.

### Why cascade over RRF as the primary production recommendation?

In the benchmark, cascade achieves:
- **45.3 % recall@10** at **0.193 ms** (4.4× faster than dense brute-force)
- **9.8× better recall than sparse alone** at **8.7× the sparse latency**

RRF achieves 35.3 % recall but at 0.918 ms — effectively the same cost as full dense scan.
Cascade gives better recall AND better latency by limiting dense computation to the top-C
sparse candidates.

### Why holdout evaluation?

If query sources are also indexed, the ground-truth dense NN is the document itself, giving
trivial 100 % recall for all methods. The holdout design (index N, query from held-out Q)
forces genuine retrieval and produces meaningful recall differences.

### Why no `unsafe`?

An earlier implementation used an unsafe interior-mutability cast to lazily cache IDF values.
The cast (`&T` → `&mut T`) is undefined behaviour. The correct fix is to recompute IDF
on-the-fly from posting list sizes, which is O(|vocab_hit|) per query — negligible at PoC
scale and avoids correctness bugs. Production code should use a `build()` phase to cache IDF.

---

## Consequences

### Positive

- RuVector gains a hybrid retrieval primitive usable from any Rust codebase.
- The `HybridSearcher` trait allows future backends (DiskANN-backed dense, SPLADE sparse) to
  be dropped in without changing client code.
- Benchmark infrastructure enables regression-free iteration on future improvements.
- All 7 acceptance criteria pass; 20 unit tests pass.

### Negative / Trade-offs

- **Linear dense scan** is O(N·D) per query. Integration with DiskANN approximate NN is the
  next required step for production use at N > 100 K.
- **No WAND pruning**: sparse index is O(Σ df(t)) per query, not sub-linear. At N=5000 this
  is fast (0.022 ms); at N=100 M it is not.
- **Static BM25**: no learned sparse expansion (SPLADE). The crate is a platform for plugging
  in learned weights, not a replacement for them.

---

## Alternatives Considered

| Alternative | Reason Rejected |
|---|---|
| Wrap Tantivy (Rust BM25 library) | External dependency; harder to control BM25 formulation and integrate with RuVector's internal types |
| Implement SPLADE directly | Requires a BERT-family model for inference — out of scope for a single nightly |
| Score normalisation instead of RRF | Requires calibration; RRF is a better default |
| HNSW-backed dense | Adds an HNSW implementation; DiskANN integration is the RuVector-native path; out of scope for this nightly |

---

## Implementation Notes

**BM25 IDF**: Recomputed at query time from `posting_list.len()` as document frequency.
`avgdl` is the mean of stored document lengths. This is correct for a static index but
would need incremental maintenance for a live index.

**RRF k=60**: The canonical value from Cormack et al. (2009). Exposed as a constructor
parameter for future tuning.

**Cascade candidate count**: `C = min(k × 20, N)` with `k=10` → `C=200`. This is 4 % of
the 5000-doc index. Tunable via constructor.

**Parallelism**: Rayon is a workspace dependency but not used in search hot paths. Batch
parallel scoring is a straightforward future extension.

---

## Benchmark Summary

Hardware: linux/x86_64 — Rust 1.94.1 — `cargo run --release`  
N=5000 indexed, D=128, Q=200 queries, k=10

| Variant | Mean ms | Recall@10 |
|---|---|---|
| Dense brute-force (ground truth) | 0.847 | 100.0 % |
| Sparse BM25 only | 0.022 | 4.6 % |
| Hybrid RRF | 0.918 | 35.3 % |
| Hybrid Cascade | 0.193 | 45.3 % |

Cascade speedup over dense: **4.4×**  
Cascade recall over sparse: **9.8×**

---

## References

1. Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal rank fusion outperforms
   condorcet and individual rank learning methods.* SIGIR 2009.
2. Robertson, S., & Zaragoza, H. (2009). *The probabilistic relevance framework: BM25 and beyond.*
   Foundations and Trends in IR.
3. Formal, T., Piwowarski, B., Clinchant, S. (2022). *SPLADE v2.* SIGIR 2022.
4. Chen, J. et al. (2024). *BGE M3-Embedding.* arXiv:2402.03216.
5. Broder, A. Z. et al. (2003). *Efficient query evaluation using a two-level retrieval process.*
   CIKM 2003 (WAND algorithm).
