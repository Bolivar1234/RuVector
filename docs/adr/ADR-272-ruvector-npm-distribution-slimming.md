# ADR-272: RuVector NPM Package Distribution Slimming

**Status**: Proposed
**Date**: 2026-07-02
**Authors**: RuVector NPM Deep Review Swarm
**Supersedes**: None
**Related**: ADR-273 (Dependency Diet & Version Coherence), ADR-274 (CLI Startup & Hooks Hot Path)

---

## Context

A deep packaging review of `ruvector` v0.2.33 (`npm/packages/ruvector`) against the published artifact baseline (`ruvector-0.2.26.tgz`: **3.18 MB packed / 12.03 MB unpacked / 146 files**) found that the vast majority of the published payload is optional-feature weight and accidental artifacts:

| Payload | Size | Share of unpacked | Needed by default? |
|---|---|---|---|
| `dist/core/onnx/pkg/ruvector_onnx_embeddings_wasm_bg.wasm` | 7.43 MB | **62%** | No — only for ONNX embedding, and only when native `@ruvector/core` can't serve |
| `dist/core/onnx/pkg/ruvector.db` (leaked runtime SQLite DB) | 1.59 MB | 13% | **Never** — accidental pack-time leak |
| `wasm/ruvector_decompiler_wasm_bg.wasm` | 1.47 MB | 12% | No — `ruvector decompile` nicety with a pure-JS fallback (`src/decompiler/index.js:40` returns `null` on failure and falls back to the Louvain pipeline) |
| 76 `.map`/`.d.ts` files (36 declaration maps referencing unshipped `../src/**/*.ts`) | ~83 KB of maps | <1% | Maps: no |

Additional distribution problems:

1. **The `.db` leak mechanism is still armed.** The build script (`package.json`: `tsc && cpSync('src/core/onnx','dist/core/onnx')`) copies *whatever* is in `src/core/onnx` into `dist/`, and the `files` whitelist ships all of `dist/`. `.npmignore` has `*.db`, but `.npmignore` cannot exclude files pulled in via the `files` array. `scripts/verify-dist.js` (written for issue #399) only checks that required files *exist*, never that forbidden files are *absent*. A dirty working tree at pack time shipped a 1.6 MB runtime database — a size and potential data-leak problem (learned patterns, local paths).
2. **`verify-dist.js` has a blind spot.** It validates the relative `require('../dist/...')` calls in `bin/cli.js` but not the six `require('../src/decompiler/...')` calls in `bin/mcp-server.js` (lines 3758–3925). This is a latent repeat of the v0.2.23 "published without dist" breakage.
3. **A committed 3.1 MB tarball** (`ruvector-0.2.26.tgz`) is tracked in git. It is never published (`files` whitelist + npm's built-in `*.tgz` exclusion) but bloats every clone permanently, and is 7 versions stale.
4. **Generated tsc output committed into `src/`**: 34 artifacts (10 `.d.ts`, 24 `.js.map`/`.d.ts.map`, e.g. `src/analysis/complexity.d.ts`, `src/workers/native-worker.d.ts.map`) sit alongside their `.ts` sources — drift risk and diff noise. (The 19 JS files under `src/decompiler/` and `src/optimizer/` are hand-written, not generated.)
5. **No `exports` map, no `sideEffects` flag.** `package.json` exposes only `main`/`types`. Deep imports into `dist/**` are unguarded API surface; bundlers cannot tree-shake. Sibling `@ruvector/wasm-unified` already demonstrates the full conditional-exports pattern in this repo.
6. **Fragile published layout**: `files` ships `src/decompiler/` and `src/optimizer/` raw JS *outside* `dist/`, reached from `bin/` via `require('../src/...')` — the exact shape that caused issue #399.

The repo already contains the correct architectural template: `@ruvector/router` and `@ruvector/core` publish a tiny JS loader with heavy artifacts in per-platform `optionalDependencies` ("hub + optional heavy satellites").

## Decision

Apply the hub-and-satellites pattern to `ruvector`'s two WASM blobs, and harden the pack pipeline with a deny-list:

1. **Extract ONNX embeddings WASM → `@ruvector/onnx-embeddings-wasm`** (optionalDependency). `src/core/onnx/loader.js` already dynamically loads; change resolution order to `require.resolve('@ruvector/onnx-embeddings-wasm')` → local `pkg/` fallback (dev). Saving: **−7.43 MB** for every user who never calls `embed`.
2. **Extract decompiler WASM → `@ruvector/decompiler-wasm`** (optionalDependency). `wasm/package.json` already exists with its own name/`files` — it is pre-shaped for standalone publishing. `src/decompiler/index.js:40` tries the package first, keeps the local-path fallback. Saving: **−1.47 MB**.
3. **Add a forbidden-file gate to `verify-dist.js`**: fail `prepack` if `dist/` contains `*.db`, `*.log`, `*.tgz`, `*.sqlite`; add a `filter` to the build `cpSync` as defense-in-depth. This is a regression guard worth 1.6 MB and closes the data-leak vector.
4. **Extend `verify-dist.js`** to validate every relative `require()` in **both** `bin/cli.js` and `bin/mcp-server.js`, and move `src/decompiler/` + `src/optimizer/` under `lib/` (or copy into `dist/` at build) so the published layout is exactly `bin/ + dist/ + wasm-shim + README + LICENSE`.
5. **Untrack `ruvector-0.2.26.tgz`** (`git rm --cached`) and gitignore `*.tgz`; delete generated `.d.ts`/`.map` artifacts from `src/` and gitignore them.
6. **Add `exports` + `sideEffects`** to package.json:
   ```json
   "exports": {
     ".": { "types": "./dist/index.d.ts", "require": "./dist/index.js" },
     "./package.json": "./package.json"
   },
   "sideEffects": false
   ```
   Dual ESM/CJS (tsup, as in `@ruvector/wasm-unified`) is a follow-up, not a blocker.
7. **Stop publishing declaration maps**: publish tsconfig with `"declarationMap": false` (keep `.d.ts`). Saving: ~83 KB and no broken-source-path debugger warnings.

## Consequences

**Positive**
- Default install payload drops **12.03 MB → ~3.1 MB unpacked (−74%); 3.18 MB → ~0.9 MB packed**.
- The `.db`-leak class of bug becomes impossible to publish, not merely unlikely.
- Issue-#399-class breakage (bin requiring unshipped paths) is caught for both entry points at pack time.
- Public API surface is explicit; tree-shaking enabled for bundler consumers.

**Negative / Risks**
- Two new packages to version and publish — must be wired into the same release automation as the platform packages (see ADR-273 version-coherence CI, which this depends on).
- Users of `ruvector embed`/`decompile` on machines without the optional packages get a graceful degradation message instead of built-in function; docs must state `npm i @ruvector/onnx-embeddings-wasm` explicitly.
- `sideEffects: false` must be verified against `dist/` (no import-time registration side effects) before enabling.

## Implementation Plan

| Phase | Work | Verification |
|---|---|---|
| 1 (immediate) | verify-dist deny-list + mcp-server.js require scan; untrack tgz; delete generated src/ artifacts; gitignore updates | `npm pack --dry-run` file list golden test |
| 2 | `exports` map + `sideEffects`; declarationMap off | `npm run test` + a smoke `require('ruvector')` / deep-import failure test |
| 3 | Publish `@ruvector/onnx-embeddings-wasm` + `@ruvector/decompiler-wasm`; flip loader resolution order; demote from `files` | tarball size assertion in CI (packed < 1.2 MB); embed/decompile integration tests with and without optional packages |
