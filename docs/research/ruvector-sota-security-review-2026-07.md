# RuVector SOTA, Security, and Package Review — 2026-07-26

## Scope and method

This review covers the complete tracked release surface at `main`
(`6a6c39e66`): 197 Rust workspace packages, 169 tracked npm manifests, 47
GitHub Actions workflows, and approximately 382,000 lines of Rust,
TypeScript, JavaScript, and native source under `crates/`, `npm/packages/`,
and `packages/`.

The review combined:

- manifest and release-workflow inventory;
- RustSec and `cargo-deny` dependency-graph checks;
- npm production-dependency audit;
- source-wide searches for unsafe Rust, process execution, unchecked panics,
  path handling, protocol stdout, and release bypasses;
- targeted execution of the `ruvector` npm build, package verification, MCP
  initialize handshake, signal cleanup, and policy tests;
- comparison with 2025–2026 work on filtered ANN, dynamic quantization,
  compressed graph/ID storage, hybrid sparse+dense retrieval, and fresh
  disk-backed ANN.

Raw pattern counts are triage signals, not vulnerability counts. Generated
bindings, tests, examples, and deliberately low-level SIMD/FFI code account
for much of the `unsafe`, `unwrap`, and process-execution footprint.

## Executive assessment

RuVector's algorithm portfolio is unusually broad and substantially aligned
with the current research frontier. The repository already contains HNSW,
DiskANN, SPANN, ACORN/filter-aware search, RaBitQ and other quantization,
hybrid BM25+dense fusion, sparse and multivector representations, GNN
reranking, Matryoshka evaluation, RVF persistence, Postgres integration,
WASM/N-API bindings, and dedicated SOTA benchmark runners.

The largest gap is not another ANN algorithm. It is productization across the
many implementations: durable metadata and filter semantics across every
binding, common recall/latency/memory benchmarks, streaming-update evaluation,
safe-by-default agent tool execution, and a release process that can prove
exactly what was built and published.

## Current SOTA alignment

| Area | RuVector coverage | Assessment |
|---|---|---|
| In-memory ANN | HNSW, learned and coherence variants, SIMD kernels | Strong |
| Disk / billion-scale ANN | DiskANN, SPANN, delta/LSM and repair research | Strong portfolio; needs one shared freshness benchmark |
| Filtered ANN | Filtered search and ACORN-family work | Algorithmically current; metadata durability and selectivity-aware planning remain the limiting integration gaps |
| Compression | PQ/scalar/binary paths, RaBitQ, Matryoshka runners | Strong; add graph-edge/vector-ID compression and streaming retraining measurements |
| Hybrid retrieval | BM25, sparse vectors, RRF/RSF/score fusion | Strong in Rust; Node's top-level `ruvector` API remains incomplete |
| Late interaction | Multivector support and reranking components | Present but not yet a single documented, benchmarked MaxSim product path |
| Freshness | Delta indexes, repair, LSM research, snapshot/raft components | Broad building blocks; no unified insert/delete/update SLO gate |
| Portability | Native Rust, N-API, WASM, Postgres, RVF | Excellent breadth; API parity is inconsistent |
| Evaluation | `ruvector-sota-bench`, VDBBench, MTEB and focused benchmarks | Good foundation; results need reproducible hardware/dataset manifests and regression budgets |

### Highest-value SOTA work

1. **Filtered ANN as a query-planning problem.** Choose exact scan, IVF,
   graph traversal, or pre/inline/post-filtering from measured selectivity and
   vector/filter correlation. Add difficult-filter datasets and recall
   stability gates, not only unfiltered ANN recall.
2. **Fresh quantization under updates.** Measure quality drift after inserts,
   deletes, and distribution shifts; trigger local codebook repair before a
   global rebuild.
3. **Compress graph structure, not only vectors.** Vector IDs and HNSW/IVF
   adjacency can dominate memory after aggressive vector quantization.
4. **One multistage retrieval contract.** Standardize dense+sparse candidate
   generation, RRF/RSF fusion, optional MaxSim/GNN reranking, and provenance
   of every score.
5. **Budget-aware serving.** Expose recall target, latency/evaluation budget,
   and freshness target as the stable API; keep index-specific tuning internal.

## Security findings

### Fixed in this change

- **Critical availability — MCP launcher no-op (#715).** The CLI required
  `mcp-server.js`, but that module only started when executed directly. The
  module now exports `main()` and the CLI invokes it explicitly.
- **Protocol integrity — stdout corruption (#710).** All ONNX loader status
  messages now use stderr; stdout remains JSON-RPC-only.
- **MCP command injection.** `workers_create` previously interpolated raw MCP
  fields into `execSync`. It now uses `execFileSync` with an argument vector.
- **Rust soundness advisories.** `anyhow` is updated to 1.0.104 and `memmap2`
  to 0.9.11; their RustSec exceptions are removed.
- **npm dependency exposure.** The MCP SDK is updated to 1.29.x and the unused
  `js-beautify` dependency (and its vulnerable glob stack) is removed.

### Open risks requiring separate changes

1. **MCP defaults remain permissive.** With no environment policy, all tools
   are exposed, including tools that mutate files or launch subprocesses.
   Make `readonly` the default in the next semver-major release and require an
   explicit profile for process-launching tools.
2. **Process execution is widespread.** Replace remaining `execSync` command
   strings with `execFile`/`spawn` argument arrays, then enforce this with a
   lint rule on MCP, CLI, deploy, and installer code.
3. **Release gates can report false success.** Several workflows append
   `|| true` or `|| echo` to `npm publish`; several Cargo paths use
   `--allow-dirty`. Publishing must fail closed and verify the registry
   version, tarball digest, SBOM, provenance, and tag after upload.
4. **Workspace npm install is not reproducible on Linux.** `npm ci` currently
   attempts to install a Darwin/ARM64 workspace package and exits with
   `EBADPLATFORM`. Platform binaries need to be optional dependencies or
   excluded from the root workspace install.
5. **The aggregate npm lock contains known advisories.** The workspace audit
   reported 46 affected dependency nodes before package scoping. Triage by
   reachable production package and update direct owners; do not hide the
   aggregate result behind blanket audit exceptions.
6. **Post-quantum dependency maintenance.** RustSec now marks the PQClean-based
   `pqcrypto-*` ecosystem unmaintained. Select and benchmark a maintained
   ML-KEM/ML-DSA implementation before the next cryptography release.
7. **Residual dependency warnings.** `spin 0.9.8` is yanked but remains
   transitive through `lazy_static`; the MCP SDK also carries a moderate Hono
   Windows static-file advisory. RuVector's MCP package imports only the stdio
   server path, so the Hono static-file handler is unreachable here, but the
   dependency should still be upgraded as soon as the SDK accepts Hono 2.x.
8. **Unsafe/panic budgets are not centralized.** Add crate-level policy:
   `unsafe` only in named FFI/SIMD modules with safety invariants, and no
   `unwrap`/`expect` on production input paths.

## Package and API findings

- Issues #704–#707 are already materially addressed on `main`: RVF rejects
  unsupported metadata instead of dropping it, WASM byte export/open exists,
  SONA N-API buffers preserve residual identity, and Node hybrid-search docs
  disclose the unshipped API.
- RVF metadata still needs an end-to-end schema, durable segment storage,
  reload, result retrieval, and identical filter typing across Rust, N-API,
  WASM, and Postgres.
- `HybridSearch` remains Rust-only in the top-level Node package. Bind the
  production Rust implementation rather than introducing another JavaScript
  scoring implementation.
- The repository has many publishable manifests but release automation covers
  only a subset. Generate a canonical package graph with owner, source,
  registry, version, native platforms, test command, and publish workflow.

## Required release gates

Every published package should pass:

1. clean checkout and locked dependency install;
2. build, unit tests, integration tests, and package/tarball smoke test;
3. RustSec/npm audit with documented reachability for any exception;
4. secret scan, SBOM, license/source policy, and artifact checksums;
5. native/WASM API parity checks where applicable;
6. registry publish with provenance, followed by registry version and digest
   verification;
7. rollback/deprecation instructions recorded before release.

For `ruvector@0.2.36`, the relevant focused gates are the end-to-end MCP
initialize test, stdout JSON purity, clean signal shutdown, full npm package
tests, TypeScript build, distribution verification, and `npm pack` smoke test.

## Primary references

- [DiskANN: fast, fresh, and filtered vector search](https://github.com/microsoft/DiskANN)
- [Survey of Filtered Approximate Nearest Neighbor Search (2025)](https://arxiv.org/abs/2505.06501)
- [Filtered ANN system design and performance analysis (2026)](https://arxiv.org/abs/2602.11443)
- [Quantization for vector search under streaming updates](https://arxiv.org/abs/2512.18335)
- [Lossless compression of vector IDs for ANN](https://arxiv.org/abs/2501.10479)
- [Qdrant data, filter, index, and quantization documentation](https://qdrant.tech/documentation/manage-data/)
