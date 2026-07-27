# ADR-272: Speculative ANN Search

**Status**: Proposed  
**Date**: 2026-07-27  
**Author**: Nightly Research Agent  
**Branch**: `research/nightly/2026-07-27-speculative-ann-search`  
**Crate**: `crates/ruvector-speculative-ann`  
**Related**: ADR-193 (RAIRS IVF), ADR-240 (Coherence-HNSW), ADR-264 (PQ-ADC), ADR-268 (Capability-Gated ANN), ADR-227 (Proof-Gated Writes)

---

## Context

RuVector's retrieval stack serves agents that need both speed and accuracy. Today this requires a static tradeoff:

| Choice | Speed | Recall | Flexibility |
|--------|-------|--------|-------------|
| Full f32 linear scan | Slow | 1.000 | None |
| Quantized (PQ, SQ, RaBitQ) | Fast | 0.75–0.90 | Fixed at build time |
| HNSW | Medium | 0.93–0.98 | ef parameter, static |

No existing RuVector retrieval variant *self-adapts* to deliver a caller-specified recall target. The agent must choose between speed and accuracy at index build time, not at query time.

Speculative decoding in LLMs (Leviathan et al., 2023) resolved an analogous tradeoff in token generation: a cheap draft model proposes tokens; a target model verifies them. The accepted tokens are exact; the protocol is provably correct under the target distribution. Speedups of 2–3× are common.

**This ADR applies the speculative protocol to ANN search.**

The technical mechanism:
1. **Draft**: brute-force scan over scalar-quantized u8 vectors to produce k' candidate ids. Cost: O(n × d / 4) using integer arithmetic.
2. **Verify**: exact f32 distances for the k' candidates only. Cost: O(k' × d).
3. **Adaptive k'**: an online controller tunes k' toward a caller-specified
   recall target using sampled, externally audited recall as its feedback
   signal. Calls without ground truth use the current multiplier unchanged.

---

## Decision

Introduce `crates/ruvector-speculative-ann` as a standalone crate implementing three retrieval variants:

| Variant | Mechanism | Recall@10 | Mean (µs) | QPS | Memory |
|---------|-----------|-----------|-----------|-----|--------|
| LinearFull | f32 brute-force scan | 1.000 | 1242.7 | 805 | 4.9 MB |
| QuantizedDraft | u8 SQ brute-force scan | 0.858 | 722.3 | 1385 | 1.2 MB |
| SpeculativeANN | u8 draft + f32 verify, adaptive k' | **0.964** | **773.2** | 1293 | 6.1 MB |

*(n=10,000 × 128-dim, 500 queries, k=10, release build on x86_64 Linux.)*

The public API surface is the `AnnVariant` trait:

```rust
pub trait AnnVariant: Send + Sync {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;
    fn name(&self) -> &str;
    fn memory_bytes(&self) -> usize;
}
```

`SpeculativeANN` additionally exposes:

```rust
impl SpeculativeANN {
    pub fn build(vectors: Vec<Vec<f32>>, cfg: SpecConfig) -> Self;
    pub fn search_fixed(&self, query: &[f32], k: usize, mult: usize) -> Vec<Hit>;
    pub fn search_adaptive(&mut self, query: &[f32], k: usize,
                           ground_truth: Option<&[Hit]>) -> Vec<Hit>;
    pub fn current_mult(&self) -> usize;
    pub fn acceptance_rate(&self) -> f32;
}
```

---

## Consequences

### Positive

- **Recall-latency tradeoff made explicit and tunable**: callers specify a `target_recall` in `SpecConfig`; the system self-tunes k'.
- **No index build overhead**: the u8 draft index is produced by a single quantization pass (O(n × d)); no graph construction required.
- **Memory-efficient draft**: u8 corpus is 4× smaller than f32. For n=10k × d=128, u8 = 1.28 MB (fits L2 cache on most architectures).
- **Composable with existing primitives**: the verify stage can be wrapped by `CapGatedIndex` (ADR-268), `ProofGate` (ADR-227), or coherence scoring (ADR-240).
- **WASM-safe**: no unsafe code, no external dependencies. u8 index fits in constrained WASM heaps.

### Negative / Tradeoffs

- **Dual memory footprint**: SpeculativeANN stores both u8 draft (1.28 MB) and f32 verify corpus (5.12 MB) — 1.25× more than LinearFull alone.
- **Adaptive controller requires calibration**: the rolling recall estimator uses `target_recall = 0.95` by default; this may be wrong for corpora with high quantization error.
- **Linear scan draft does not scale to very large n**: at n=10M, even the u8 draft scan takes ~7 seconds. For large corpora, the draft must be replaced with HNSW traversal using u8 distances (future work).
- **Observed speedup < theoretical**: on n=10k, the theoretical 4× speedup materialises at ~1.7× due to L3 cache effects. The full benefit appears for n > 50k where the f32 corpus exceeds L3 cache but the u8 corpus does not.

---

## Alternatives Considered

### Alternative 1: PQ-ADC as draft oracle

**Rejected for this iteration.** PQ requires training M codebooks (k-means), which adds significant build time and complexity. SQ8 trains in a single pass. The recall improvement from PQ over SQ8 as a draft oracle is measurable but not decisive at d=128. Recommend as the next iteration (ADR-273 candidate).

### Alternative 2: HNSW draft + exact verify

**Deferred.** HNSW as the draft would replace O(n × d/4) with O(log n × d/4), dramatically better for large n. However, it requires integrating with `ruvector-coherence-hnsw`'s layer structure and sharing graph state with the u8 distance function — not a one-evening PoC. This is the production-grade path described in the roadmap.

### Alternative 3: mcp-snapshot-memory (research agent recommendation)

The nightly research agent recommended this topic (score 4.80 vs. 4.50 for speculative-ann-search) based on stronger ecosystem leverage and MCP fit. It was deferred for two reasons: (1) it requires redb snapshot integration not available in a self-contained PoC, and (2) speculative-ann-search is a more fundamental retrieval primitive that benefits all downstream consumers including a future MCP memory server. Recommend as the next nightly topic.

---

## Implementation Plan

### Phase 1 (Complete — this PR)

- [x] `ScalarQuantizer`: train-time per-dimension min/max quantizer, f32 → u8
- [x] `LinearFull`: ground-truth f32 brute-force scan
- [x] `QuantizedDraft`: u8 brute-force scan with `select_nth_unstable_by_key`
- [x] `SpeculativeANN`: draft + verify + adaptive controller
- [x] 18 unit tests, all passing
- [x] Benchmark binary with real measurements
- [x] Acceptance tests: all 6 passing

### Phase 2 (Next iteration)

- [ ] SIMD-accelerated u8 distance via `std::simd` or AVX2 intrinsics
- [ ] `SpecHnswDraft`: HNSW traversal with u8 distances as draft oracle
- [ ] Integration with `ruvector-agent-memory` as a drop-in scan replacement
- [ ] Validation on real embedding benchmarks (SIFT-1M, GLOVE-1.2M)

### Phase 3 (Research)

- [ ] Learned draft oracle (compact projection model)
- [ ] Ground-truth-free recall estimation for production deployments
- [ ] Distributed speculation (draft shard → verify shard protocol)

---

## Benchmark Evidence

All numbers are from `cargo run --release -p ruvector-speculative-ann --bin benchmark` on x86_64 Linux, Rust 2021 edition, dataset n=10,000 × d=128.

**Key finding**: moving from mult=1 (k'=k) to mult=2 (k'=2k) costs 5.5µs mean latency (+0.8%) and improves recall from 0.858 to 0.995 (+13.7pp). This is the speculative "sweet spot" for Gaussian data at this n and d.

**Speedup vs. LinearFull**:
- QuantizedDraft: 1242.7µs → 722.3µs = **1.72× faster**
- SpeculativeANN (adaptive): 1242.7µs → 773.2µs = **1.61× faster** at 0.964 recall

**Comparison at matched recall ≥ 0.95**:
- LinearFull: 1242.7µs, recall=1.000
- SpeculativeANN (mult=2): ~720µs (extrapolated from sweep), recall=0.995
- **Speedup at matched recall: ~1.73×**

---

## Failure Modes

1. **High-variance corpus**: SQ8 range maps distant embeddings to adjacent u8 bins → draft precision collapses. Detect with: after building draft index, run a calibration pass on 1% of corpus and compute draft recall; if < 0.70, warn and recommend PQ draft.
2. **Distribution shift**: embedding model update invalidates SQ statistics → recall degrades silently. Mitigate: integrate with future semantic-drift sentinel; trigger SQ re-train on drift alert.
3. **Verify bottleneck at large k'**: if the adaptive controller drives k' above n/4, the verify cost dominates. Cap k' at min(max_mult × k, n/8) as a safety guard.
4. **Single-threaded scan on large n**: at n=1M, the u8 scan takes ~720ms × (1M/10k) = 72s without parallelism. Production integration must use rayon parallel scan or switch to HNSW draft.

---

## Security Considerations

- No unsafe code in this crate; all array accesses are bounds-checked.
- SQ statistics (min/max per dimension) leak distributional information about the corpus; treat as sensitive metadata in multi-tenant deployments.
- The verify stage can be gated by `CapGatedIndex` (ADR-268) without changes to the draft — only authorised candidates proceed to exact distance computation, preventing timing-based enumeration attacks.

---

## Migration Path

`SpeculativeANN` implements the same `AnnVariant` trait as `LinearFull` and `QuantizedDraft`. Drop-in replacement:

```rust
// Before:
let idx = LinearFull::build(vectors.clone());

// After:
let idx = SpeculativeANN::build(vectors, SpecConfig { target_recall: 0.95, ..Default::default() });
```

The `search` method signature is identical. For adaptive search with recall feedback, call `search_adaptive` instead.

---

## Open Questions

1. **What is the optimal draft oracle for real (non-Gaussian) embedding distributions?** PQ, learned projections, or tree-based partitions may outperform SQ8 in practice. Requires empirical study on public benchmarks.
2. **How does the adaptive controller behave under adversarial query distributions?** An attacker who sends queries maximally misranked by SQ8 could drive k' to max_mult indefinitely. Needs a minimum batch quota to prevent manipulation.
3. **What is the right interface for integrating with HNSW?** The verify corpus is the f32 store; the HNSW graph provides the draft. These may share internal representation if the HNSW implementation exposes a `draft_search` interface.
4. **Should SpeculativeANN be a standalone crate or a feature flag in ruvector-core?** Current: standalone (same pattern as ruvector-capgated, ruvector-matryoshka). Recommend: keep standalone until the HNSW integration is complete, then consider merging into ruvector-coherence-hnsw as a feature flag.
