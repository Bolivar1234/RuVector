# ADR-272: Diverse Beam ANN — MMR Post-Reranking and Coherence-Pruned Beam Search

- **Status**: Proposed (implemented and benchmarked — `crates/ruvector-diverse-beam`)
- **Date**: 2026-07-26
- **Research doc**: `docs/research/nightly/2026-07-26-diverse-beam-ann/README.md`

---

## Context

RuVector's retrieval pipeline returns the k nearest neighbours by L2 distance. For agent-memory retrieval and RAG use cases, pure nearest-neighbour results suffer from **diversity collapse**: the top-k often contains near-duplicate vectors representing the same semantic concept from slightly different angles. Downstream LLMs and reasoning agents benefit from diverse context — coverage of multiple distinct relevant aspects rather than multiple copies of one.

Two approaches exist in the literature for adding diversity to ANN:

1. **Post-reranking** (apply diversity selection on the candidate pool after traversal) — used in document IR (MMR, Carbonell & Goldstein 1998) and neural IR (Vinh et al. 2023), but not characterised for graph-based ANN.
2. **In-traversal pruning** (prune directionally-redundant candidates during graph BFS) — appears in conceptual proposals but not benchmarked against real recall/diversity/latency trade-offs.

This ADR documents the decision to implement both approaches, measure them on two synthetic datasets, and characterise their trade-offs — including two important **negative results**.

---

## Decision

Implement `ruvector-diverse-beam` as a standalone Rust crate in the workspace containing three beam-search variants and a benchmark binary:

1. **GreedyBeam** — standard greedy BFS (baseline).
2. **MMRRerank** — greedy BFS to collect `max(beam_width, k×4)` candidates, then iterative MMR selection to pick k final results. Uses normalised relevance and diversity scores in [0, 1].
3. **CoherenceBeam** — greedy BFS with a cosine-similarity gate: a candidate is skipped if its cosine similarity to any of the 8 most-recently-expanded nodes exceeds `coherence_threshold`.

The crate exposes a `BeamSearch` trait for pluggable backends, `recall_at_k` and `mean_pairwise_dist` as evaluation utilities, and a `FlatGraph` for exact kNN graph construction and brute-force ground truth.

---

## Measured Results

All from `cargo run --release -p ruvector-diverse-beam --bin benchmark`, Linux/x86_64.

### Uniform random (n=2500, dim=64, K_NN=16, K=10, beam=50)

| Variant | Recall@10 | Diversity | Mean µs | QPS |
|---------|-----------|-----------|---------|-----|
| GreedyBeam | 0.816 | 5.7410 | 87.1 | 10,975 |
| MMRRerank (λ=0.75) | 0.707 | 5.8364 | 193.6 | 5,069 |
| CoherenceBeam (θ=0.90) | 0.816 | 5.7410 | 687.0 | 1,448 |

### 10-cluster Gaussian (σ=0.14)

| Variant | Recall@10 | Diversity | Mean µs | QPS |
|---------|-----------|-----------|----------|-----|
| GreedyBeam | 0.516 | 1.5417 | 46.0 | 20,098 |
| MMRRerank | 0.502 | 1.5958 | 151.6 | 6,439 |
| CoherenceBeam | **0.002** | 5.3257 | 169.4 | 5,773 |

---

## Consequences

### Positive

- **MMRRerank** provides a composable, backend-agnostic diversity layer (+1.67% diversity, −13.4% recall, −53.8% QPS on uniform data). It can be applied on top of any ANN pool without modifying the graph or index.
- The `BeamSearch` trait establishes a reusable interface for beam-search backends across the RuVector ecosystem.
- `mean_pairwise_dist` is a useful diversity metric that can be added to RuVector's query evaluation toolkit.
- The normalised MMR formulation (both terms in [0, 1] using pool's max distance) is reusable for any distance-based diversity scoring.

### Negative

- **CoherenceBeam is an anti-pattern for clustered data** (recall=0.002, σ=0.14, 10 clusters). Tight cluster cohesion (high intra-cluster cosine similarity) is indistinguishable from the "redundant direction" signal the coherence gate is designed to suppress. Any deployment of CoherenceBeam must verify that `max_intra_cluster_cosine_sim < coherence_threshold` — practically limiting it to near-uniform datasets. **This variant is shipped as a documented negative result, not a production recommendation.**
- CoherenceBeam's QPS (1,448 on uniform) is 7.5× lower than GreedyBeam (10,975) despite producing identical results on uniform data — the cosine computation overhead is not justified.
- MMRRerank at λ=0.75 reduces recall by 13.4%. Applications where recall is the primary metric should use GreedyBeam.

---

## Alternatives Considered

### 1. MMR during graph traversal

An early "MMRBeam" variant picked candidates to expand next based on their MMR score (relevance to query minus redundancy to already-expanded nodes). Result: recall=0.610 uniform, recall=0.034 clustered. **Rejected.** Redirecting the beam away from the query during traversal is fundamentally incompatible with recall-preserving ANN. The correct application point for MMR is after traversal.

### 2. Determinantal Point Process (DPP) reranking

DPP selection maximises the determinant of the result set's kernel matrix — a theoretically principled diversity measure. Cost: O(k³) per query with a full-rank kernel. For k=10 this is tractable, but DPP requires a kernel matrix computation (n²/2 pairwise distances from the pool). Deferred as a follow-up comparison baseline.

### 3. Structural diversity (Diverse-HNSW style)

Modify graph construction to ensure diverse neighbourhood lists — each node's k_nn neighbours are chosen to maximise coverage of angular sectors. This produces diversity structurally without any query-time overhead. **Deferred** — requires a new graph build algorithm. MMRRerank is the practical near-term solution as it works with any existing graph.

---

## Implementation Notes

### Entry point alignment fix

Entry points use stride `((n / n_entry) | 1).max(1)` — the smallest odd integer ≥ n/n_entry. `gcd(odd, any_even_cluster_count) ≤ smallest_prime_factor_of_odd`, so an odd stride cycles through more clusters before any modular repeat. This prevents even-stride period alignment with round-robin cluster assignments.

### MMR normalisation invariant

Both relevance and diversity must be in [0, 1] and on the same absolute scale. Using pool-max distance `max_dist = max(d_q(c))` as the denominator for both terms satisfies this. The `max(·, 1e-6)` floor prevents division by zero when all candidates are at the query vector.

### CoherenceBeam history management

The expansion history is a `Vec<VecId>` of length ≤ 8, maintained as a sliding window (`remove(0)` when full). This is O(8) per expansion — acceptable for small history sizes.

---

## Related ADRs

- ADR-243: ruvector-mincut spectral coherence (graph structural coherence, complementary signal)
- ADR-254: ruvector-acorn filtered HNSW (predicate-based beam pruning)
- ADR-258: ruvector-coherence-hnsw (traversal-direction coherence in HNSW, different from CoherenceBeam)
- ADR-266: metaharness-Darwin ANN optimization (evolutionary hyper-parameter search)
- ADR-270: self-reconstructing graph memory (graph-level diversity in memory consolidation)
