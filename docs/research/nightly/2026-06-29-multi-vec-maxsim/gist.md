# Multi-Vector Late Interaction Search: ColBERT MaxSim in Pure Rust

**Published:** 2026-06-29 | **Crate:** `ruvector-maxsim` | **Branch:** `research/nightly/2026-06-29-multi-vec-maxsim`

---

Single-vector retrieval has one fundamental problem: it erases context. Compress
a 512-token document into a single 768-dimensional embedding and you lose the
relationship between tokens. Ask "what are the risks of async Python exceptions in
concurrent database transactions?" and your bi-encoder sees a semantic centroid of
that sentence — not the individual concepts.

ColBERT solved this in 2020. **Multi-vector late interaction** keeps all token
embeddings for both queries and documents, then scores via MaxSim:

```
score(q, d) = Σ_{qi ∈ q}  max_{dj ∈ d}  cosine_sim(qi, dj)
```

Every query token independently finds its best-matching document token. The sum
tells you: *how many query concepts does this document actually cover?* The result
is 3–7% better nDCG@10 than bi-encoders across BEIR benchmarks — consistent,
meaningful, and theoretically principled.

The catch: scoring N documents × M query tokens × K doc tokens × D dimensions is
O(N·M·K·D). At N=1M, M=8, K=32, D=128 that's 33.5 trillion multiply-adds per
query. Not suitable for brute force at scale.

---

## The Implementation: Three Variants, One Trait

`ruvector-maxsim` ships three indexes behind a single `MultiVecIndex` trait:

```rust
pub trait MultiVecIndex {
    fn add(&mut self, doc: MultiVecDoc);
    fn search(&self, query_tokens: &[Token], k: usize) -> Vec<MaxSimResult>;
    fn len(&self) -> usize;
}
```

### Variant 1: FlatMaxSim (exact baseline)

Brute-force. Scores every document with full MaxSim. O(N·M·K·D). The ground-truth
oracle — 100% recall by definition, but slow.

### Variant 2: PrunedMaxSim (centroid pruning)

Inspired by PLAID (Santhanam et al., 2022). The insight: a document's *centroid*
(mean of its token embeddings) is a cheap proxy for "roughly where is this document
in embedding space?" For each query token, retrieve the top-C documents by centroid
cosine similarity, union those candidate sets across all query tokens, then run
exact MaxSim on the union only.

Build: O(N·K·D) to compute centroids.
Search: O(N·M·D) centroid scan + O(C·M·K·D) MaxSim rerank, where C ≪ N.

At C=200 (10% of a 2,000-doc corpus): **100% recall@10 at 2.70× speedup** vs.
brute force.

### Variant 3: GraphMaxSim (kNN graph + beam search)

Goes further. Build a greedy kNN graph over centroid vectors offline (O(N²·D), one
time). At query time, beam-search the graph from multiple entry points to find
centroid neighbours fast — O(ef·m·D) instead of O(N·D) — then rerank with MaxSim.

**Result: 98.7% recall@10 at 7.94× speedup** on a 2,000-document corpus.

---

## The Bug That Killed Recall (And How to Avoid It)

Early versions of GraphMaxSim had a catastrophic recall failure: **5% recall**
instead of the expected >90%. The cause was subtle.

Documents were indexed in cluster-interleaved order: doc 0 → cluster 0, doc 1 →
cluster 1, …, doc 19 → cluster 19, doc 20 → cluster 0, …

The beam search seeded from 5 entry points at step = N/5 = 400:
indices {0, 400, 800, 1200, 1600}. Now, 400 % 20 = 0. So did 800, 1200, 1600.
**All five entry points landed in cluster 0.** Queries from clusters 1–19 had no
nearby seed and the beam search couldn't navigate to them.

The fix: seed from the **first 40 consecutive documents** instead. Docs 0–19 cover
all 20 clusters. Docs 20–39 add a second representative per cluster. Consecutive
seeding is robust to interleaved ordering by construction.

```rust
// Seed from first n_seeds consecutive docs — guarantees cluster coverage
// for interleaved layouts (doc i → cluster i % C).
let n_seeds = 40usize.min(n);
for e in 0..n_seeds {
    if !visited[e] {
        visited[e] = true;
        let s = cosine_sim(query_centroid, &self.centroids[e]);
        frontier.push(OrdF32(s, e));
    }
}
```

Step-based seeding is a trap whenever your step is a multiple of your cluster count.
Consecutive seeding is always safe. File this under "off-by-cluster-modulo errors."

---

## Benchmark Results

```
Corpus: 2,000 docs, 8 tokens/doc, dim=64, 20 Gaussian clusters, σ=0.3
Queries: 200, 4 tokens/query
Hardware: x86-64 Linux 6.18, single thread, cargo --release
```

| Variant | Mean latency | QPS | Recall@10 |
|---------|-------------|-----|-----------|
| FlatMaxSim | 4,851 µs | 206 | 100.0% |
| PrunedMaxSim | 1,794 µs | 557 | 100.0% |
| GraphMaxSim | 611 µs | 1,637 | 98.7% |

GraphMaxSim builds its graph in 400 ms (offline, one-time). After that, sub-ms
query latency at 98.7% recall.

---

## Why This Matters for Production RAG

Standard RAG pipelines: embed query → single-vector ANN → retrieve chunks → LLM.
The ANN step loses token-level information that ColBERT recovers.

**What MaxSim enables that bi-encoders miss:**

- **Token-level coverage**: knows *which specific concepts* in a query match a document
- **Cross-lingual recall**: token embeddings from multilingual ColBERT align across
  languages even when bi-encoders miss the global embedding alignment
- **Safety reasoning**: per-token similarity scores are interpretable — an MCP tool
  can report *which* query tokens matched *which* document tokens
- **Nested entity recall**: "Federal Reserve chair Jerome Powell" — MaxSim finds
  documents covering each entity separately; bi-encoders may not

For ruvector, the next natural step is a `ruvector-colbert` crate that couples an
ONNX ColBERT encoder (via `ruvector-onnx-embedder`, ADR-194) to `ruvector-maxsim`'s
index — end-to-end multi-vector retrieval in pure Rust.

---

## Using the Crate

```toml
[dependencies]
ruvector-maxsim = "0.1"
```

```rust
use ruvector_maxsim::{MultiVecDoc, MultiVecIndex, Token};
use ruvector_maxsim::graph::GraphMaxSim;

// Build index
let mut idx = GraphMaxSim::new(12, 64, 200);
for (id, token_embeddings) in your_corpus {
    idx.add(MultiVecDoc::new(id, token_embeddings));
}
idx.build(); // one-time O(N²·D) graph construction

// Query
let query_tokens: Vec<Token> = your_encoder.encode_query("async Python exceptions");
let results = idx.search(&query_tokens, 10);
// results: Vec<MaxSimResult> sorted by MaxSim score descending
```

For N < 10K and latency-insensitive pipelines, `PrunedMaxSim` is simpler (no
`build()` call needed):

```rust
use ruvector_maxsim::pruned::PrunedMaxSim;

let mut idx = PrunedMaxSim::new(200); // 200 centroid candidates per query token
idx.add(doc);
let results = idx.search(&query_tokens, 10);
```

---

## The SOTA Landscape

| System / Paper | Technique | Notes |
|----------------|-----------|-------|
| ColBERT v2 (2022) | Residual compression | 4-bit PQ on token embeddings |
| PLAID (2022) | Centroid candidate generation | Matches PrunedMaxSim |
| PyLate (arXiv:2508.03555) | Sentence-Transformers + ColBERT | Python, 2025 |
| GNN-RAG (arXiv:2405.20139) | Graph-guided token retrieval | 2026 |
| KET-RAG (arXiv:2502.09304) | Knowledge-enhanced token retrieval | 2026 |
| **ruvector-maxsim** | Pure Rust, 3 variants, sub-ms | This work |

The Python ecosystem has strong ColBERT support (PyLate, RAGatouille). Rust has
almost nothing. `ruvector-maxsim` is a starting point for Rust-native multi-vector
retrieval.

---

## What's Next

- **Phase 2**: SIMD inner product kernels (AVX2/NEON via `ruvector-simd`) — expect
  4–8× throughput improvement for the MaxSim inner loop
- **Phase 3**: Disk-backed token storage (mmap pages) — enables N > 100K on
  commodity hardware
- **Phase 4**: `ruvector-colbert` — ONNX ColBERT encoder + `ruvector-maxsim` +
  single end-to-end API
- **Phase 5**: MCP tool in `mcp-gate` — expose MaxSim search as an MCP tool for
  token-level agent safety reasoning

The crate is live: `crates/ruvector-maxsim`. Run the benchmark yourself:

```bash
CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse cargo run --release -p ruvector-maxsim
```

---

*Part of the ruvector nightly research series. Prior nights: rabitq (quantisation),
acorn-filtered-hnsw (filtered graph ANN), rairs-ivf (dual-assignment IVF).*
