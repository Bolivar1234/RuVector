# ADR-275: RuVector Runtime Hot-Path Optimization (Typed Arrays, Caches, Batch Inserts)

**Status**: Proposed
**Date**: 2026-07-02
**Authors**: RuVector NPM Deep Review Swarm
**Supersedes**: None
**Related**: ADR-274 (CLI Startup), ADR-276 (Parallel Subsystem Consolidation)

---

## Context

A runtime review of `npm/packages/ruvector/src` found that while the native/WASM layers are fast, the JS layer between them systematically discards their performance through boxing, O(n) cache algorithms, and per-item round-trips. Findings, with evidence:

### 1. Float32Array → number[] boxing across the entire vector path (MED-HIGH)

The WASM/native layers produce `Float32Array`s; the JS layer immediately boxes them, and downstream consumers convert *back*:

- `onnx-embedder.ts:339` — `Array.from(embedding)` per `embed()`; `:400-407` — `slice()` **plus** `Array.from()` per batch row (two copies each).
- `bundled-parallel.mjs:149` — `Array.from(flat.subarray(...))` per row, undoing the transferable-buffer optimization one line after receiving it.
- `embed-worker.mjs:55` — redundant `Float32Array.from(flat)` copy (wasm-bindgen already returns a fresh, transferable-backed array).
- `attention-fallbacks.ts:51-53` — boxes every attention output.
- Consumers re-wrap: `native-worker.ts:353`, `intelligence-engine.ts:584` do `new Float32Array(item.embedding)` before native insert.

For a 10k-document ingest at 384d this is ~4 avoidable full copies per vector plus number[] storage at **5–10× the memory of f32** (8-byte boxed doubles + pointer overhead vs 4-byte floats).

### 2. O(n) cache algorithms on hot paths (MEDIUM → MED-HIGH at scale)

- **`EmbeddingService`** (`src/services/embedding-service.ts:248-270`): at `maxCacheSize` (default 10,000), *every* insert scans all entries to find the least-hit item — O(n) per insert, ~10⁸ comparisons over a 10k ingest. It is labelled LRU but implements LFU-without-aging, stores `number[]` (~30 MB+ at capacity), and keys on a 32-bit `hashText` (`:46-54`) — collision ⇒ **wrong embedding returned**. The correct pattern exists 100 lines away: `onnx-optimized.ts:110-169` has a Map-order O(1) LRU.
- **`FastAgentDB`** (`src/core/agentdb-fast.ts`): `getEpisode` recency update via `episodeOrder.indexOf(id)` + `splice` (`:176-183`) — O(n) with `maxEpisodes` default **100,000**, so at capacity every read scans 100k entries. `storeEpisodes` (`:157-165`) awaits `storeEpisode` per item — one native round-trip each despite the binding exposing `insertBatch` (already used in `index.ts:185-193`). `fallbackSearch` (`:230-243`) full-sorts instead of a k-heap; states as `number[]` ⇒ >100 MB at capacity.
- **`IntelligenceEngine`** (`src/core/intelligence-engine.ts:626-633, 186`): fallback `recall()` shallow-copies **every** memory (spread including content strings) per query just to sort; `memories` Map has no eviction — every `remember()` grows it forever, duplicating embeddings already inserted into VectorDB. `attentionEmbed` (`:445-478`) constructs a fresh native attention object per call and materializes identical K and V as two separate copies.
- **`MemoryPhysics.encode`** (`src/core/neural-embeddings.ts:419-426, 443-470`): O(n) interference scan per insert (O(n²) ingest), brute-force recall — while the package wraps a native HNSW index that could serve both.

### 3. Redundant embedder instantiations (HIGH where used)

- `loader.js` convenience helpers (`src/core/onnx/loader.js:453-460, 473-476`): `embed(text)` / `similarity(a,b)` call `createEmbedder()` **per invocation** — 7 MB WASM instantiate + full ONNX graph re-parse each call (hundreds of ms vs 5–20 ms warm). The `_inMemoryModelCache` only caches downloaded bytes, not the instantiated embedder.
- `OptimizedOnnxEmbedder` (`src/core/onnx-optimized.ts`) loads its **own** WASM instance and model copy rather than sharing with `onnx-embedder.ts` (both are exported from the barrel — using both doubles model memory). Its advertised features are inert: `tokenizerCache` (`:218`) is allocated but never read/written; `QUANTIZED_MODELS` fp16/int8 URLs (`:56-104`) are never passed to the loader (acknowledged `:260-265`); `getOptimizedOnnxEmbedder(config)` ignores `config` after first call (`:498-503`).

### 4. Scattered scalar kernels and misc hot-path waste

- **Eleven duplicated pure-JS cosine implementations**: `index.ts:352`, `onnx-embedder.ts:433`, `embedding-service.ts:275`, `agentdb-fast.ts:246`, `intelligence-engine.ts:635`, `diff-embeddings.ts:353`, `adaptive-embedder.ts:331,542`, `neural-embeddings.ts:558,972,1208`, `parallel-workers.ts:406` — most over `number[]`, on the main thread, while native SIMD and WASM `similarity` sit unused.
- **O(n²) line-number computation** in the security scanner (`src/analysis/security.ts:77`): `fileContent.slice(0, match.index).split('\n').length` per match, with a pattern set including `/\$\{.*\}/g` that matches essentially every template literal — multi-second scans on large files. Duplicated inside the `parallel-workers.ts` eval'd worker.
- `isOnnxAvailable()` / `detectParallelAvailable()` (`onnx-embedder.ts:123-130, 139-148`) re-`fs.existsSync` on every call (native wrappers cache correctly).
- `native-worker.ts:307-308` reads whole files to keep 512 chars; `:350-365` inserts vectors one at a time inside try/catch.
- Per-result `JSON.parse` of metadata on every search hit (`src/index.ts:207, 213-218`) with no opt-out.

Recorded as done-right (patterns to standardize on): lazy memoized `getXModule()` wrappers with cached errors; `bundled-parallel.mjs` SAB model sharing + transferable results + unref'd timeouts; `loader.js` atomic disk model cache; `onnx-optimized`'s Map-order LRU and unrolled cosine.

## Decision

1. **`Float32Array` becomes the canonical vector type end-to-end.** `embed`/`embedBatch`/`embedBulk` return `Float32Array` (or `{flat, dim}` views over one buffer); attention outputs stay typed; `number[]` survives only in explicitly named compat helpers (`embedToArray`) with a deprecation note. Remove the four identified redundant copies.
2. **One shared O(1) LRU** (extracted from `onnx-optimized.ts:110-169`) replaces the `EmbeddingService` scan-evict cache and bounds `IntelligenceEngine.memories`; cache keys must include a collision guard (store full text or a 128-bit hash). `FastAgentDB` LRU switches to Map insertion-order (delete+set, O(1)).
3. **Batch native round-trips**: `FastAgentDB.storeEpisodes` and `native-worker` vector storage use `insertBatch`; `fallbackSearch` and `recall()` use score-only tuples + partial selection (k-heap) instead of clone-and-full-sort; `attentionEmbed` hoists attention-object construction and passes the same array for K and V.
4. **One embedder instance per model, one WASM runtime.** `loader.js` memoizes `createEmbedder()` per model name; `OptimizedOnnxEmbedder` is folded into `onnx-embedder.ts` (its LRU migrates; its dead tokenizer-cache and unwired quantization config are either implemented or deleted — not advertised).
5. **One shared cosine/similarity kernel** over `Float32Array`, delegating to native/WASM when loaded, used by all eleven call sites. `MemoryPhysics` interference/recall route through the native HNSW wrapper.
6. **Mechanical fixes**: precomputed newline-offset table + binary search for scanner line numbers (both copies); memoize availability probes; 512-byte `readSync` for file previews; `rawMetadata` option to skip per-hit JSON.parse.

## Consequences

**Positive**
- Bulk ingest: removes ~4 copies/vector and per-item native round-trips — the dominant JS-side costs; vector memory drops 5–10× where number[] storage is replaced.
- Read paths stop degrading with scale: O(1) cache reads at 100k episodes vs 100k-element scans; queries stop cloning the entire memory store.
- Eliminates a correctness bug class: 32-bit-hash cache collisions returning wrong embeddings; "optimized" embedder silently ignoring its config.
- Halves model memory for users who touch both embedder exports.

**Negative / Risks**
- Returning `Float32Array` where `number[]` was returned is a breaking change for consumers using array methods like `.map` returning numbers (`Float32Array.map` returns Float32Array) or JSON-serializing results — requires a major/minor version gate, compat helpers, and a migration note.
- Folding `OptimizedOnnxEmbedder` removes a public export; keep a deprecated alias for one minor version.

## Implementation Plan

| Phase | Work | Verification |
|---|---|---|
| 1 | Shared LRU + collision-safe keys; FastAgentDB O(1) LRU + insertBatch; scanner newline table; probe memoization | unit tests incl. 100k-episode read benchmark (assert O(1)-ish); existing `npm test` |
| 2 | Typed-array canonicalization behind compat helpers; remove redundant copies; shared cosine kernel | ingest microbenchmark (10k × 384d): copies, RSS, wall time vs baseline |
| 3 | Embedder unification (loader memoization, fold onnx-optimized); IntelligenceEngine recall/eviction; MemoryPhysics → HNSW | model-memory RSS assertion (one WASM instance); recall correctness tests |
| 4 | `rawMetadata` opt-out; native-worker read/insert fixes | search-path benchmark |
