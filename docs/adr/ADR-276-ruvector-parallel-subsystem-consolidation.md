# ADR-276: RuVector Parallel Subsystem Consolidation (Worker Pools, Correctness, Memory)

**Status**: Proposed
**Date**: 2026-07-02
**Authors**: RuVector NPM Deep Review Swarm
**Supersedes**: None
**Related**: ADR-275 (Runtime Hot Paths), ADR-274 (CLI Startup & Hooks Hot Path)

---

## Context

The package contains **four** worker-pool implementations of varying quality: `bundled-parallel.mjs` (correct: id-correlated request map, SAB model sharing, transferable results, unref'd timeouts, dead-worker cleanup), the ONNX embedder pools, `ParallelIntelligence`, and `ExtendedWorkerPool`. The review found the latter three carry correctness bugs and structural waste that matter most in long-lived MCP-server processes:

### 1. Dual ONNX pools — up to 2× workers, each with a full model copy (HIGH)

`src/core/onnx-embedder.ts`: `tryInitParallel()` (`:298-311`, auto-enabled under MCP) populates `parallelEmbedder`, while `embedBulk()`/`initParallelEmbedder()` (`:546-564`) lazily create a **second** pool in `bundledPool` — both `ParallelEmbedder` instances over the same model. Each pool spawns up to `min(cpus-2, 16)` workers, and each worker instantiates its own WASM runtime and parses the ~23 MB MiniLM model into WASM linear memory (`embed-worker.mjs:25-48` — the SharedArrayBuffer only deduplicates *source bytes*; wasm-bindgen copies them per instance). On a 16-core MCP server: 2 pools × 14 workers ≈ 28 WASM instances, plausibly **>1 GB RSS**, plus 2× thread oversubscription.

### 2. `ParallelIntelligence`: stub workers + queued-task promise hang (HIGH)

`src/core/parallel-intelligence.ts`:
- **All six task handlers are stubs** (`:300-384`): `processEpisodes` returns `episodes.length`, `matchPatterns` returns empty, `searchShard`/`analyzeCommits` return `[]`, `analyzeFiles` returns hardcoded `{agent:'coder', confidence:0.5}`. Yet under MCP (`enabled` default true, `:70`) `init()` spawns `cpus−1` threads, each re-evaluating the whole compiled module (`new Worker(__filename)`, `:82`) — pure thread/memory overhead producing **fabricated results**.
- **Queued tasks never resolve** (`:103-144`): `executeInWorker` wires the response handler only when a worker is immediately free; `processQueue()` posts queued tasks without attaching `resolve`/`reject` — queued promises **hang forever**. Each dispatch also stacks a temporary `'message'` listener on the permanent one (`:86-89`) — `MaxListenersExceededWarning` / double-handling risk under churn.
- `recordEpisodesBatch` (`:153-157`) silently drops episodes below the batch threshold ("fall back to sequential" returns void).

### 3. `ExtendedWorkerPool`: same task dispatched to every idle worker (MEDIUM)

`src/core/parallel-workers.ts:658-668`: `processQueue` always reads `taskQueue[0]` and never removes/marks it, so with W idle workers **one task is posted to all W** (entries removed only on result, `:640-655`) — regex scans, `git blame`, and file reads duplicated W×, plus O(n) `findIndex` per completion. Also: an unused `Blob` of worker code allocated in `init()` (`:165-166`); workers built via `eval:true` string concatenation (`:170-175`, re-parse per worker); the eval'd worker repeats the O(n²) line-number anti-pattern (~`:390`) and O(n²) pairwise-Jaccard `deduplicate` (`:612-640`).

### 4. Unbounded, never-invalidated caches in long-lived processes (MEDIUM)

`parallel-workers.ts:147-148, 718-733, 745-762`: `speculativeCache` and `astCache` grow for the process lifetime, keyed by file path with **no mtime check** — long-lived MCP servers accumulate memory and serve **stale ASTs/embeddings after edits**. `clearCaches()` exists but has no callers.

### 5. Eager library-load native binding (MEDIUM — shared with ADR-274)

`src/index.ts:38-77` runs the `@ruvector/core` NAPI dlopen (with rvf fallback + three `console.warn`s) at `require('ruvector')` time — the sole eager loader in an otherwise lazy codebase; relevant here because MCP/worker children that require the module pay it per process.

## Decision

1. **One worker-pool implementation.** `bundled-parallel.mjs`'s protocol (id-correlated `_pending` map, transferables, unref'd timeouts, dead-worker replacement) becomes the single pool primitive; the ONNX embedder unifies `parallelEmbedder` and `bundledPool` onto one instance (`parallelEmbedder ??= bundledPool`), and `ExtendedWorkerPool`/`ParallelIntelligence` are ported onto it or deleted.
2. **`ParallelIntelligence.enabled` defaults to `false` until real handlers exist.** Stub handlers must not spawn threads or return fabricated confidence values. When re-enabled: adopt the id-correlated protocol (fixes the queued-promise hang), a single permanent message listener, and explicit sequential fallback that actually processes (not drops) sub-threshold batches.
3. **Fix `ExtendedWorkerPool` dispatch**: shift the task off the queue into a `pending` map keyed by task id at dispatch time; workers get a real module file instead of `eval:true` string assembly; port the newline-offset table and hash-based dedup from ADR-275 into the worker code.
4. **Bound and invalidate caches**: `speculativeCache`/`astCache` become mtime-validated bounded LRUs (shared LRU from ADR-275); wire `clearCaches()` into the MCP server's session lifecycle.
5. **Worker-count budgeting**: a single process-wide budget (`min(cpus − 2, 16)` total across all pools, env-overridable) so pools can't stack oversubscription; workers hold **one** WASM/model instance per process by construction (single pool) rather than per pool.
6. **Defer native dlopen out of module load** (`src/index.ts` → memoized `getImplementation()`, per ADR-274) so worker children and library consumers don't pay it eagerly.

## Consequences

**Positive**
- MCP-server RSS drops by hundreds of MB on many-core machines (28 → ≤14 WASM instances; stub pool removed entirely).
- Two hang/correctness bug classes eliminated: forever-pending queued promises, and W× duplicate side-effectful task execution (duplicated `git blame`/file reads are also a subtle correctness hazard, not just waste).
- Long-lived processes stop leaking memory through unbounded caches and stop serving stale analysis after file edits.
- One pool implementation to test, instrument, and tune instead of four.

**Negative / Risks**
- Disabling `ParallelIntelligence` changes reported behavior for anything consuming its (currently fabricated) outputs — audit consumers first; the honest results may differ from the hardcoded ones.
- Consolidating pools is invasive in `onnx-embedder.ts`; needs load tests to confirm no throughput regression when one pool serves both `embedBatch` and `embedBulk` traffic classes.
- mtime-based invalidation adds a `stat` per cache hit; acceptable (µs) but should be sampled/batched if profiling shows otherwise.

## Implementation Plan

| Phase | Work | Verification |
|---|---|---|
| 1 | `ParallelIntelligence.enabled=false`; fix ExtendedWorkerPool dispatch (pending map); bound + mtime-validate caches | regression test: queued-task resolution under saturation; duplicate-dispatch unit test; soak RSS profile |
| 2 | Unify ONNX pools; process-wide worker budget | 16-core load test: RSS and thread count vs baseline; embedBatch/embedBulk throughput |
| 3 | Port/delete ExtendedWorkerPool onto the shared primitive; real worker module (drop eval) | full `npm test` + MCP integration tests |
| 4 | Re-enable ParallelIntelligence only with real handlers + id-correlated protocol | handler-level unit tests; no-fabricated-results audit |
