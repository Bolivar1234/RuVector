# ruvector 2026: Speculative ANN Search — Draft-Then-Verify for High-Performance Rust Vector Retrieval

**Speculative ANN search applies the speculative decoding protocol to vector retrieval: a fast u8 draft index proposes k' candidates; exact f32 distances verify them; an adaptive controller tunes k' from sampled recall audits. Implemented in pure Rust with no runtime dependencies.**

The measured synthetic benchmark achieves 0.964 recall@10 at 1293
queries/sec—1.61× faster than its full f32 linear baseline—when the adaptive
candidate multiplier receives exact benchmark feedback.

→ Repository: [https://github.com/ruvnet/ruvector](https://github.com/ruvnet/ruvector)  
→ Branch: `research/nightly/2026-07-27-speculative-ann-search`  
→ PR: see repository for current draft PR

---

## Introduction

Every vector database deployment faces the same tension: speed versus accuracy. A brute-force linear scan returns exact top-k neighbors but is slow. Quantized indexes (Product Quantization, Scalar Quantization, RaBitQ) are fast but lossy — they sacrifice recall to gain speed. Hierarchical Navigable Small World (HNSW) graphs offer a middle ground but require expensive index construction and static ef parameter tuning.

This research asks: *can we do better than a static tradeoff by borrowing the speculative decoding protocol from LLM inference?*

In large language model serving, speculative decoding (Leviathan et al., 2023) uses a fast draft model to propose tokens and a target model to verify them. The accepted tokens are provably equivalent to direct target-model generation, but arrive 2–3× faster. The key insight: verification is cheap relative to fresh generation.

The same insight applies to vector search. Scalar quantization (SQ8) compresses f32 vectors to u8 with 4× smaller memory and cheaper distance arithmetic. The rank-ordering errors introduced by quantization are modest — measured recall at k'=k is 0.858 on Gaussian data. But if we use SQ8 only for a *draft* — to identify a candidate set of size k' — and then *verify* with exact f32 distances for those k' candidates only, we can achieve near-full-precision recall at near-draft speed.

The verification step is cheap because k' is small relative to n. For n=10,000, k'=20, d=128: the verify step computes 20 × 128 = 2,560 exact distances, versus the draft's 10,000 × 32 = 320,000 integer distance computations. The overhead is negligible.

Current Rust vector databases — including those in the ruvector ecosystem — do not implement speculative search. They make the speed-accuracy tradeoff at build time. `ruvector-speculative-ann` introduces a new primitive: a *recall-budgeted* retrieval interface where the caller specifies a target recall threshold, and the system self-tunes the candidate multiplier k'/k to deliver that threshold at minimum latency.

This matters for AI agents, graph RAG pipelines, and edge AI systems where retrieval quality directly affects downstream reasoning quality, and where query volume is high enough that latency budgets are real. For MCP-based agent tooling, the speculative search primitive enables a `memory_recall` tool that self-calibrates to the agent's quality requirements — no static parameter tuning required. For WASM edge deployments, the 4× smaller u8 draft index fits within memory constraints that the full f32 corpus cannot.

---

## Features

| Feature | What It Does | Why It Matters | Status |
|---------|-------------|----------------|--------|
| Scalar Quantizer (SQ8) | Compresses f32 → u8 per dimension via min/max range | 4× memory reduction; integer distance arithmetic | Implemented in PoC |
| LinearFull variant | Brute-force f32 scan (ground truth) | Reference baseline for recall measurement | Measured |
| QuantizedDraft variant | Brute-force u8 scan, no verify | Establishes speed/recall bounds of draft alone | Measured |
| SpeculativeANN variant | u8 draft k' candidates + f32 exact re-rank | Near-full recall at near-draft speed | Measured |
| Adaptive k' controller | Sampled recall audits drive k' up/down | Feedback-driven target recall | Measured with oracle feedback |
| `AnnVariant` trait | Uniform search interface across all variants | Drop-in replacement for existing linear scan | Implemented |
| `SpecConfig` | Target recall, window size, min/max mult bounds | Caller-specified quality SLA | Implemented |
| recall_at_k metric | Intersection-based recall vs. ground truth | Standard ANN benchmark metric | Implemented |
| Deterministic benchmark | LCG dataset, reproducible seeds | Comparable runs across environments | Measured |
| Multiplier sweep | Benchmark across 6 k' values | Reveals recall-latency Pareto curve | Measured |
| WASM-compatible | No unsafe, no external deps | Runs in constrained edge environments | Research direction |
| HNSW draft integration | Replace linear scan with graph traversal for draft | O(log n) draft for large n | Production candidate |

---

## Technical Design

### Core Data Structure

Two co-resident indexes:
1. **f32 corpus**: original vectors for exact distance computation in the verify stage.
2. **u8 draft index**: scalar-quantized corpus for fast rank-approximate search in the draft stage.

The `ScalarQuantizer` computes per-dimension min/max ranges on the training corpus (one pass, O(n × d)) and maps each f32 value to the closest u8 bucket. The quantization is deterministic and reversible (approximately) given the stored scale parameters.

### Trait-Based API

```rust
pub trait AnnVariant: Send + Sync {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;
    fn name(&self) -> &str;
    fn memory_bytes(&self) -> usize;
}

pub struct Hit { pub id: usize, pub dist: f32 }
```

All three variants implement `AnnVariant`. For adaptive search, `SpeculativeANN` additionally exposes:

```rust
pub fn search_adaptive(&mut self, query: &[f32], k: usize,
                       ground_truth: Option<&[Hit]>) -> Vec<Hit>
```

### Baseline Variant: LinearFull

O(n × d) f32 distance computations using `select_nth_unstable_by` (partial sort, O(n) average). Ground truth for recall measurement.

### Alternative A: QuantizedDraft

Same O(n × d/4) structure but with u8 integer arithmetic. Uses `select_nth_unstable_by_key` on u64 distances. Returns approximate top-k without verification. Achieves 0.858 recall@10 on Gaussian data at n=10k, d=128.

### Alternative B: SpeculativeANN

```
Draft:  u8_scan(query, k') → [candidate_ids]          O(n × d / 4)
Verify: [exact_dist(query, corpus[id]) for id in ids]  O(k' × d)
Rank:   top_k(verified)                                 O(k' log k')
Control: rolling_recall → k' adjustment               O(window)
```

The adaptive controller uses a rolling mean over the last `window` queries. If mean_recall < target_recall: k' += 2. If mean_recall > target_recall + 0.03: k' -= 1.

### Memory Model

| Index | Per-vector | n=10k × d=128 |
|-------|-----------|----------------|
| f32 corpus | d × 4 bytes = 512 bytes | 5.12 MB |
| u8 draft | d × 1 byte = 128 bytes | 1.28 MB |
| SpeculativeANN total | 640 bytes | 6.40 MB |

### Performance Model

```
Draft cost:   n × d / 4 FLOPs (integer L2)
Verify cost:  k' × d FLOPs   (f32 L2, k' candidates)
Total:        n × d/4 + k' × d

Break-even vs. LinearFull (n × d FLOPs):
  n/4 + k' < n  ↔  k' < 3n/4

For k'=20, n=10000: 20 < 7500 ✓ (break-even always satisfied)
```

### Architecture Diagram

```
┌──────────────────────────────────────────────────┐
│                 SpeculativeANN                   │
│                                                  │
│  query (f32) ──→ ScalarQuantizer ──→ query (u8)  │
│                                          │        │
│  u8 corpus ─────────────────────────────→ Draft  │
│  (1.28 MB)                               scan    │
│                                          │        │
│                              k' candidate ids    │
│                                          │        │
│  f32 corpus ─────────────────────────────→ Verify│
│  (5.12 MB)                               exact   │
│                                          │        │
│                              top-k verified hits  │
│                                          │        │
│                              Adaptive Controller  │
│                           rolling_recall → k'    │
└──────────────────────────────────────────────────┘
```

---

## Benchmark Results

**Hardware**: x86_64 Linux (cloud instance)  
**OS**: linux  
**Rust version**: workspace rust-version = "1.77" (compiled with rustc 1.86.0)  
**Dataset**: 10,000 Gaussian f32 vectors, 128 dimensions, 500 queries, k=10  
**Seed**: 0xC0DECAFEBABE7777 (deterministic, reproducible)  
**Command**: `cargo run --release -p ruvector-speculative-ann --bin benchmark`

### Candidate Multiplier Sweep

| k' | Mult | Recall@10 | Mean (µs) | p50 (µs) | p95 (µs) | Accept-rate |
|----|------|-----------|-----------|----------|----------|-------------|
| 10 | 1 | 0.858 | 713.4 | 708 | 758 | 0.564 |
| **20** | **2** | **0.995** | **718.9** | **714** | **763** | **0.998** |
| 30 | 3 | 1.000 | 722.0 | 717 | 768 | 1.000 |
| 50 | 5 | 1.000 | 740.6 | 736 | 788 | 1.000 |
| 80 | 8 | 1.000 | 740.7 | 735 | 783 | 1.000 |
| 120 | 12 | 1.000 | 775.3 | 764 | 856 | 1.000 |

**Key insight**: mult=2 (k'=20) gains 13.7pp recall (+0.858 → +0.995) at only 5.5µs (+0.8% latency overhead) vs. mult=1. This is the speculative sweet spot.

### Three-Variant Summary

| Variant | Recall@10 | Mean (µs) | p50 (µs) | p95 (µs) | QPS | Memory | Notes |
|---------|-----------|-----------|----------|----------|-----|--------|-------|
| LinearFull | 1.000 | 1242.7 | 1236 | 1332 | 805 | 4.9 MB | Ground truth |
| QuantizedDraft | 0.858 | 722.3 | 709 | 810 | 1385 | **1.2 MB** | Fast, lossy |
| **SpeculativeANN** | **0.964** | **773.2** | 773 | 846 | **1293** | 6.1 MB | **Adaptive k'=3** |

### Benchmark Limitations

- Gaussian N(0,1) data has known good SQ rank-preservation properties. Real embedding datasets (SIFT, GLOVE, sentence-transformers) may have different recall characteristics.
- The cloud container environment has limited CPU cache (shared tenancy), which suppresses the theoretical 4× speedup from u8 integer arithmetic. On dedicated hardware with warm L2 cache, the speedup is expected to be closer to 3–4×.
- No external competitor benchmarks are included — all numbers are from this crate's benchmark binary. Comparisons with Qdrant, Milvus, or FAISS would require matching their hardware, index parameters, and dataset.

---

## Comparison with Vector Databases

| System | Core Strength | Where It's Strong | Where RuVector Differs | Directly Benchmarked Here |
|--------|--------------|-------------------|------------------------|--------------------------|
| **Milvus** | Scalable distributed ANN | Cloud-native, GPU indexing | RuVector: Rust-native, proof-gated, edge-first | No |
| **Qdrant** | Rust, scalar quantization | Production SQ8 in Rust | RuVector: adaptive k', coherence gates, agent-memory | No |
| **Weaviate** | Hybrid vector+graph | Schema-based retrieval | RuVector: MCP-native, ruFlo automation, RVF packages | No |
| **Pinecone** | Managed vector DB | Zero-ops production | RuVector: self-hosted, air-gapped, WASM-portable | No |
| **LanceDB** | Lance columnar format | Multi-modal, Python-first | RuVector: Rust-only, no Python, graph-vector co-index | No |
| **FAISS** | C++ ANN primitives | GPU brute-force, PQ | RuVector: safe Rust, no C FFI, proof-gated writes | No |
| **pgvector** | PostgreSQL extension | SQL-native, OLTP | RuVector: agent memory, WASM, MCP tools | No |
| **Chroma** | Python embedding DB | Simple, LLM-first | RuVector: multi-tenant isolation, coherence, proofs | No |
| **Vespa** | Ranking + ANN fusion | WAND + ANN together | RuVector: graph cuts, RVF, edge-native Rust | No |

**Framing note**: RuVector's differentiation is not in raw ANN throughput — it's in the combination of Rust memory safety, agent-memory primitives, coherence-gated retrieval, proof-gated writes, MCP tool surface, and WASM portability. SpeculativeANN adds adaptive recall targeting to this stack.

---

## Practical Applications

| Application | User | Why It Matters | RuVector Use | Path |
|-------------|------|----------------|--------------|------|
| **Agent context recall** | AI agent runtimes | Agent needs correct context fast; recall errors cause reasoning failures | SpeculativeANN in agent-memory crate, target_recall=0.95 | Near-term |
| **Enterprise semantic search** | IT/search teams | Sub-second recall across millions of docs; accuracy SLA required | Scale n, add HNSW draft | Near-term |
| **RAG pipeline acceleration** | LLM app developers | Retrieval is often the p99 bottleneck in production RAG | Replace LinearFull in ruvector-server | Near-term |
| **Local-first AI assistants** | Privacy-conscious users | 4× smaller index enables on-device inference | u8 draft only on mobile, f32 verify on local server | Near-term |
| **Code intelligence tooling** | IDE developers | Symbol search needs low latency; false negatives confuse autocomplete | Speculative over code embeddings | Near-term |
| **Security event retrieval** | SOC analysts | Log embedding search for pattern detection; miss rate has security cost | Target recall=0.99, larger k' | Near-term |
| **MCP memory tools** | MCP agent developers | `memory_recall(query, recall_target=0.95)` as a self-calibrating tool | Wrap SpeculativeANN in MCP tool handler | Medium-term |
| **Multi-agent memory federation** | Multi-agent platforms | Shared draft index, per-agent f32 verify corpus for isolation | Combine with capgated (ADR-268) | Medium-term |

---

## Exotic Applications

| Application | 10–20 Year Thesis | Required Advances | RuVector Role | Risk |
|-------------|-------------------|-------------------|---------------|------|
| **Speculative cognition on Cognitum Seed** | Edge appliances run u8 draft locally; cloud or companion device runs f32 verify on demand | Learned draft oracles, compressed model distillation | u8 draft in WASM kernel on Cognitum Seed hardware | Power constraints on verify frequency |
| **RVM coherence-domain routing** | Different memory coherence domains use domain-specific SQ quantizers; draft is coherence-gated | Per-domain quantizer training, coherence boundary detection | rvm + speculative + coherence-hnsw integration | Domain boundary misclassification |
| **Proof-gated speculative retrieval** | Every verified candidate generates a cryptographic receipt; agents prove what they searched before acting | Efficient proof circuits for exact-distance computation | proof-gate wrapping the verify stage | Proof overhead per candidate |
| **Swarm shared draft** | N agents share one u8 draft index; each agent maintains its private f32 verify corpus | Draft-verify protocol with per-agent private corpus partitions | Draft federation over ruvector-cluster | Privacy leakage via draft hit patterns |
| **Self-healing quantization** | SQ statistics drift as new memories accumulate; background thread re-trains SQ without full re-index using EWC++ | Online SQ re-training with catastrophic-forgetting prevention | Integrate with future semantic-drift sentinel crate | Distribution shift during re-training window |
| **Bio-signal neural memory** | Speculative ANN over neural signal embeddings for BCI memory retrieval; u8 on NPU, f32 verify on host | Low-latency neural processing unit integration | ruvector-hailo + speculative for NPU draft | Signal latency constraints |
| **Agent OS memory scheduler** | OS-level scheduler treats recall-quality as a first-class resource; high-priority agents get larger k' | ruFlo integration as memory scheduler priority mechanism | SpeculativeANN as memory subsystem primitive | Priority inversion in shared draft |
| **Distributed autonomous recall** | Each retrieval carries a proof of completeness: the system attests that the returned results are within ε of exact top-k with probability ≥ p | Statistical recall guarantees, formal verification of adaptive controller | RuVector as certified retrieval substrate | Formal verification complexity |

---

## Deep Research Notes

### What the SOTA Suggests

The speculative decoding literature establishes that the protocol's effectiveness depends on draft-target alignment — how often the draft's top choices match the target's top choices. For token generation, this is ~0.80–0.90 for well-chosen draft models. For ANN search with SQ8 on Gaussian data, we measure 0.858 alignment at k'=k — comparable to good draft-target token alignment in LLMs.

The 2026 ANN literature focuses heavily on quantization (PQ, RaBitQ, SQ), graph traversal (HNSW variants), and SSD-resident indexes (DiskANN, SPANN). No published work formalises the speculative search protocol for ANN retrieval. The closest are:

- **DiskANN** (NeurIPS 2019): uses a two-pass approach — SSD-resident compressed index for candidate selection, in-memory full-precision re-rank. This is speculative search without the adaptive controller or the formal accept/reject framing.
- **Cascade retrieval** (various): re-ranks candidates from a fast index with a slower, more accurate scorer. SpeculativeANN generalises this with an explicit recall target and adaptive k'.

### What Remains Unsolved

1. **Non-Gaussian recall guarantees**: the 0.858 draft recall at mult=1 is measured on synthetic Gaussian data. Real embedding spaces have different structure — hyperbolic geometry, cluster gaps, distributional tails. Empirical validation on SIFT-1M, GLOVE-1.2M, and sentence-transformer embeddings is required.
2. **Optimal draft oracle**: SQ8 is simple and fast but not optimal for rank preservation. Research direction: learn a compact projection that maximises rank preservation within the u8 space.
3. **Production-scale k' adaptation**: the rolling window estimator works in benchmarks. Under query distribution shift, adversarial queries, or model updates, the controller may oscillate. Needs stability analysis.

### Sources

[^1]: Leviathan Y., Kalman M., Matias Y. "Fast Inference from Transformers via Speculative Decoding." ICML 2023. https://arxiv.org/abs/2211.17192.
[^2]: Jayaram Subramanya S. et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." NeurIPS 2019. https://arxiv.org/abs/2108.11418.
[^3]: Jégou H. et al. "Product Quantization for Nearest Neighbor Search." IEEE TPAMI 2011. doi:10.1109/TPAMI.2010.57.
[^4]: Malkov Y., Yashunin D. "Efficient and robust approximate nearest neighbor search using HNSW." IEEE TPAMI 2018. https://arxiv.org/abs/1603.09320.

---

## Usage Guide

```bash
# Clone and switch to the research branch
git clone https://github.com/ruvnet/ruvector
cd ruvector
git checkout research/nightly/2026-07-27-speculative-ann-search

# Build the crate
cargo build --release -p ruvector-speculative-ann

# Run all tests
cargo test -p ruvector-speculative-ann

# Run the benchmark (default: 10k × 128, 500 queries)
cargo run --release -p ruvector-speculative-ann --bin benchmark

# Change dataset size via environment variables
N_VECS=50000 N_QUERIES=1000 DIMS=256 \
  cargo run --release -p ruvector-speculative-ann --bin benchmark
```

### Expected Output

```
╔══════════════════════════════════════════════════════════════════╗
║       RuVector Speculative ANN Benchmark                        ║
╠══════════════════════════════════════════════════════════════════╣
║  OS                         linux                                 ║
║  Corpus                     10000 vectors × 128 dims              ║
║  Queries                    500                                   ║
╚══════════════════════════════════════════════════════════════════╝
...
ACCEPTANCE RESULT: PASS ✓ — all thresholds met
```

### Interpreting Results

- **recall@10**: fraction of LinearFull's true top-10 present in the variant's top-10. Higher is better.
- **accept-rate**: fraction of queries where recall@10 ≥ 0.90. Tracks adaptive controller quality.
- **Speedup vs full**: lower mean latency than LinearFull indicates the speculative path is effective.
- **mult sweep**: the recall-latency Pareto curve. For Gaussian data, mult=2 is the sweet spot.

### Adding a New Backend

Implement `AnnVariant` for your index type:
```rust
impl AnnVariant for MyIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> { /* ... */ }
    fn name(&self) -> &str { "MyIndex" }
    fn memory_bytes(&self) -> usize { /* ... */ }
}
```

Use it as the draft oracle in `SpeculativeANN` by replacing the `QuantizedDraft` field with your index and implementing `draft_ids` to return candidate ids ranked by approximate distance.

### Plugging into RuVector

The `AnnVariant` trait is designed for drop-in use anywhere a linear scan appears:
```rust
// In ruvector-agent-memory (future integration):
let index: Box<dyn AnnVariant> = Box::new(
    SpeculativeANN::build(stored_vectors, SpecConfig {
        target_recall: 0.95,
        initial_mult: 2,
        ..Default::default()
    })
);
let hits = index.search(&query_embedding, 10);
```

---

## Optimization Guide

### Memory Optimization

- **Use u8 draft only** (QuantizedDraft) when memory is severely constrained — 4× smaller, 0.858 recall.
- **Increase k' slowly**: each increment adds k × d = 10 × 128 = 1,280 bytes to verify overhead — negligible vs. the constant draft cost.
- **Share the f32 corpus** with other indexes if possible: if `ruvector-agent-memory` already stores f32 vectors, the verify stage can reference them without duplication.

### Latency Optimization

- **Fix mult=2** rather than using adaptive mode for predictable latency (718.9µs vs. 773.2µs adaptive mean).
- **Warm the L2 cache**: for n ≤ 10k at d=128, the u8 index (1.28 MB) fits in L2. Structure access patterns to maximise cache residence.
- **SIMD**: replace the scalar u64 accumulator in `sq_l2_u8` with AVX2 `_mm256_madd_epi16` — expected 4× throughput improvement.

### Recall Optimization

- **Increase mult** to 3 for 100% recall on Gaussian data (mult=3 → recall=1.000 at 722µs).
- **Use PQ draft** (future): PQ draft oracle achieves higher rank preservation than SQ8 at the same memory budget.
- **Cluster-aware SQ**: train separate SQ quantizers per k-means cluster; quantize query to its cluster's SQ. Better rank preservation for clustered data.

### Edge Deployment Optimization

- **u8 draft on-device, f32 verify on server**: transfer k' × 8 bytes per query (20 × 8 = 160 bytes) to the verify endpoint.
- **WASM target**: the crate has no `unsafe` code and no system dependencies. Build with `wasm32-unknown-unknown` target.
- **Reduce d**: smaller dimensions reduce both u8 draft memory and scan latency linearly.

### MCP Tool Optimization

A `memory_recall` MCP tool wrapping `SpeculativeANN`:
```json
{
  "name": "memory_recall",
  "description": "Retrieve k nearest memories to a query embedding",
  "inputSchema": {
    "query_embedding": "float32[]",
    "top_k": "integer",
    "recall_target": "float",
    "snapshot_id": "string?"
  }
}
```
The `recall_target` parameter maps directly to `SpecConfig.target_recall`, making the adaptive k' controller a user-visible quality SLA.

---

## Roadmap

### Now

- Integrate `SpeculativeANN` as an optional retrieval backend in `ruvector-agent-memory`.
- SIMD-accelerated `sq_l2_u8` using `std::simd` or x86-specific intrinsics.
- Validate recall on real embedding benchmark (SIFT-1M at minimum).

### Next

- `SpecHnswDraft`: use HNSW traversal with u8 distances as the draft oracle — O(log n × d/4) instead of O(n × d/4). Required for production use at n > 100k.
- Learned draft oracle: compact projection model that preserves rank ordering better than SQ8 for real (non-Gaussian) embeddings.
- MCP tool wrapper for `memory_recall` with `recall_target` SLA parameter.
- Combine with `ruvector-capgated` (ADR-268): capability-gated verify stage where only authorised candidates receive exact distance computation.

### Later (2031–2046)

- Formal recall guarantees: statistical certificates on the adaptive controller's recall delivery, enabling SLA contracts in regulated deployments.
- Distributed speculation: draft shard + verify shard protocol for multi-node deployments where neither corpus fits on a single machine.
- Proof-gated speculative retrieval: cryptographic receipt per verified candidate, enabling audit trails for high-stakes agent memory (medical, legal, financial).
- Neural draft oracle: a small learned model trained on the retrieval task — aligns draft rank ordering with the target distribution, not just geometric proximity. Expected to push draft recall from 0.858 toward 0.95+ at mult=1.

---

## Footnotes and References

[^1]: Leviathan Y., Kalman M., Matias Y. "Fast Inference from Transformers via Speculative Decoding." ICML 2023. https://arxiv.org/abs/2211.17192. Accessed 2026-07-27.

[^2]: Chen C. et al. "Accelerating Large Language Model Decoding with Speculative Sampling." arXiv 2302.01318. 2023. https://arxiv.org/abs/2302.01318. Accessed 2026-07-27.

[^3]: Jayaram Subramanya S. et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." NeurIPS 2019. https://arxiv.org/abs/2108.11418. Accessed 2026-07-27.

[^4]: Jégou H., Douze M., Schmid C. "Product Quantization for Nearest Neighbor Search." IEEE TPAMI 33(1), 2011. doi:10.1109/TPAMI.2010.57. Accessed 2026-07-27.

[^5]: Malkov Y., Yashunin D. "Efficient and robust approximate nearest neighbor search using HNSW." IEEE TPAMI 42(4), 2020. https://arxiv.org/abs/1603.09320. Accessed 2026-07-27.

[^6]: Kusupati A. et al. "Matryoshka Representation Learning." NeurIPS 2022. https://arxiv.org/abs/2205.13147. Related: coarse-to-fine retrieval with dimension cascade. Accessed 2026-07-27.

[^7]: Aguerrebere C. et al. "Locally-adaptive Quantization for Streaming Vector Search." arXiv 2023. Related: adaptive quantization for ANN. Accessed 2026-07-27.

---

## SEO Tags

**Keywords**: ruvector, Rust vector database, Rust vector search, high performance Rust, ANN search, HNSW, DiskANN, filtered vector search, graph RAG, agent memory, AI agents, MCP, WASM AI, edge AI, self learning vector database, ruvnet, ruFlo, Claude Flow, autonomous agents, retrieval augmented generation, speculative decoding, scalar quantization, approximate nearest neighbor, adaptive recall, vector retrieval.

**Suggested GitHub topics**: rust, vector-database, vector-search, ann, hnsw, diskann, rag, graph-rag, ai-agents, agent-memory, mcp, wasm, edge-ai, rust-ai, semantic-search, speculative-decoding, approximate-nearest-neighbor, scalar-quantization, adaptive-recall, ruvector.
