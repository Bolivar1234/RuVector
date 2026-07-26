# Why Your ANN Search Returns Near-Duplicates (And What To Do About It)

**Diverse beam search for approximate nearest neighbours — three Rust implementations, real benchmarks, two negative results**

---

If you've built a RAG pipeline or agent memory system on top of a vector database, you've probably noticed this: the top-k results for a query are often essentially the same document, chunked five different ways. Your embedding model returns nearly identical vectors for nearly identical text, your ANN index finds them all, and your LLM gets five copies of the same paragraph instead of five different relevant perspectives.

This is the **diversity collapse** problem in vector retrieval, and it's more damaging than it looks.

## The Problem

A standard greedy beam search on a kNN graph is a recall-maximising algorithm. It finds the k nearest neighbours with high probability. But "nearest" and "most informative" are not the same thing. When you feed top-k nearest vectors into an LLM context window:

- Duplicate information wastes tokens
- Redundant context anchors the model to one perspective
- Edge-case knowledge that's slightly less similar to the query gets squeezed out

For RAG, this means answers that miss important nuances. For agent memory retrieval, it means the agent's context window is dominated by one memory cluster while other relevant memories are never surfaced.

## Three Solutions, Three Trade-Offs

I implemented three beam-search variants in Rust on a flat kNN graph, measured their recall, diversity, and latency, and found two important negative results along the way.

### Variant 1: GreedyBeam (Baseline)

Standard greedy BFS. Always expand the candidate nearest to the query. Maximises recall, makes no attempt at diversity.

```
Recall@10: 0.816    Diversity: 5.74    QPS: 10,975
```

This is your baseline. Every other variant will trade something against this.

### Variant 2: MMRRerank

Maximum Marginal Relevance, applied as post-reranking over a wider candidate pool.

The key insight is **where** to apply MMR. An early version tried applying MMR during traversal — picking which node to expand next based on a relevance-diversity score. Result: recall dropped to 0.034 on clustered data. MMR during traversal redirects the beam away from the query, which is the opposite of what you want from an ANN algorithm.

The correct approach: run standard greedy beam search to collect a pool of `max(beam_width, k×4)` candidates, then iteratively select k final results using:

```
score(c) = λ · relevance(c) + (1−λ) · diversity(c)
```

where both terms are normalised to [0, 1] using the pool's maximum distance. This normalisation is critical — if relevance and diversity aren't on the same scale, one term dominates.

```
Recall@10: 0.707    Diversity: 5.84    QPS: 5,069
```

At λ=0.75: +1.67% diversity, −13.4% recall, −53.8% QPS. Measurable but modest trade-off.

### Variant 3: CoherenceBeam (Negative Result)

The idea: during BFS traversal, maintain a history of the last 8 expanded nodes. Before adding a candidate to the beam, check its cosine similarity to the history. If it's too similar (cosine > threshold), skip it — you've already explored that direction.

On uniform random data: it works fine, matching GreedyBeam's recall while theoretically preventing redundant exploration.

On 10-cluster Gaussian data (σ=0.14): **recall = 0.002**.

The failure mode is fundamental. Tight cluster members have high cosine similarity to each other. When you expand any node in a cluster, every other cluster member looks "coherent" with the recently-expanded history and gets pruned. The algorithm designed to avoid redundant exploration perfectly avoids exploring the relevant cluster.

**The take-away**: coherence-based pruning is indistinguishable from cluster membership. Don't use it on clustered embeddings. Real-world text embeddings are *very* clustered.

## The Entry Point Problem

There's a subtler bug that cost me two hours: **entry point alignment**.

If you initialise beam search from evenly-spaced entry points with stride `n / n_entry`, and your dataset has clusters assigned round-robin, the entry points can land entirely within certain clusters and completely miss others. With n=300, n_entry=6: stride=50, and with 8 clusters, all 6 entry points land in only 4 clusters (modular arithmetic).

Fix: use the smallest odd stride ≥ `n / n_entry`:

```rust
let step = ((n / n_entry) | 1).max(1);
```

An odd stride's GCD with any even cluster count is bounded by the stride's smallest odd prime factor, so it cycles through more of the modular ring before repeating.

## Real Numbers

Full benchmark on n=2500, dim=64, K_NN=16, K=10, beam=50, 200 queries, Linux/x86_64, `cargo run --release`:

**Uniform random:**

| Variant | Recall@10 | Diversity | Mean µs | QPS |
|---------|-----------|-----------|---------|-----|
| GreedyBeam | 0.816 | 5.74 | 87 | 10,975 |
| MMRRerank (λ=0.75) | 0.707 | 5.84 | 194 | 5,069 |
| CoherenceBeam (θ=0.90) | 0.816 | 5.74 | 687 | 1,448 |

**10-cluster Gaussian (σ=0.14):**

| Variant | Recall@10 | Diversity | Mean µs | QPS |
|---------|-----------|-----------|---------|-----|
| GreedyBeam | 0.516 | 1.54 | 46 | 20,098 |
| MMRRerank | 0.502 | 1.60 | 152 | 6,439 |
| CoherenceBeam | **0.002** | 5.33 | 169 | 5,773 |

The clustered data also exposes that GreedyBeam's recall of 0.516 is a graph connectivity problem, not an algorithm problem: with tight clusters and K_NN=16, there are no cross-cluster edges. Every algorithm is limited by the graph structure. The solution is ensuring `n_entry ≥ n_clusters` — not changing the search algorithm.

## Practical Recommendations

**Use MMRRerank if:**
- Your use case is RAG or agent memory retrieval
- You can tolerate 13% recall reduction for 1.7% diversity improvement
- You can tolerate 2× higher latency (pool construction + MMR selection)

**Use GreedyBeam if:**
- Recall is your primary metric
- Latency is critical
- You're serving similarity search for ranking, not generation

**Don't use CoherenceBeam if:**
- Your embeddings have cluster structure (i.e., always for text)
- You want production-grade recall

**Consider CoherenceBeam only if:**
- Your data is near-uniform in the embedding space
- You've verified `max_intra_cluster_cosine_sim < coherence_threshold`
- Even then, the latency overhead (7.5× vs GreedyBeam) is hard to justify

## Where MMR Doesn't Help Enough

MMR with λ=0.75 delivers +1.67% diversity. For many production RAG systems, that's not enough — you want the diversity to be large enough to be noticeable in answer quality. Options:

1. Lower λ (more diversity weight) — but recall drops further
2. Use a larger pool (wider beam, more candidates to select from)
3. Use structural diversity: build a graph where each node's k_nn list is chosen to cover different angular sectors (Diverse-HNSW approach). This shifts the diversity cost to build time instead of query time.

## The Code

```toml
# Cargo.toml
[dependencies]
ruvector-diverse-beam = { path = "crates/ruvector-diverse-beam" }
```

```rust
use ruvector_diverse_beam::{
    graph::FlatGraph,
    search::{MMRRerank, GreedyBeam},
    BeamSearch,
};

let graph = FlatGraph::build(vectors, k_nn);
let searcher = MMRRerank { graph: &graph, n_entry: 12, lambda: 0.75 };
let results = searcher.search(&query, k, beam_width);
```

The `BeamSearch` trait is backend-agnostic — you can implement it for HNSW or DiskANN without changing the MMR reranking logic.

---

**Repository**: `ruvnet/ruvector`, crate `ruvector-diverse-beam`

**The key lesson**: for diversity in vector retrieval, post-reranking beats in-traversal modification every time. The traversal phase is for coverage; the selection phase is for quality and diversity. Don't mix the two.
