# ADR-273: RuVector Dependency Diet and Ecosystem Version Coherence

**Status**: Proposed
**Date**: 2026-07-02
**Authors**: RuVector NPM Deep Review Swarm
**Supersedes**: None
**Related**: ADR-272 (Distribution Slimming), ADR-274 (CLI Startup & Hooks Hot Path)

---

## Context

A single-platform (linux-x64) `npm install ruvector` currently pulls **~43–46 MB**, of which roughly **18–20 MB is avoidable**. The review (verified against the live npm registry via `npm view`) found three cost classes:

### 1. Fat satellite packages that double-ship native binaries

- **`@ruvector/sona@0.1.7`**: `files` includes `"*.node"`, so the published package is **4.71 MB** (≈7 bundled platform binaries) — while *also* declaring 7 per-platform `optionalDependencies` at 600 KB each. A linux-x64 user pays 5.3 MB where 0.6 MB is needed (**~4.7 MB waste**).
- **`@ruvector/rvf-node@0.1.7`**: same pattern (`files: [..., "*.node"]`, 5 committed binaries of 944 KB–1.4 MB) → **5.46 MB fat** plus platform optionalDeps (**~5.5 MB waste**). It is a *hard* dep of `@ruvector/rvf`, which `ruvector` pulls as an optionalDependency — so most installs pay it.
- Contrast with the correct pattern already in the ecosystem: `@ruvector/core` (11 KB loader), `@ruvector/gnn` (61 KB), `@ruvector/tiny-dancer` (17 KB), `@ruvector/router` — loader-only hubs with per-platform satellites.

### 2. Library users pay for CLI/MCP/decompiler dependencies

- `@modelcontextprotocol/sdk` (^1.0.0): **4.27 MB + ~5–10 MB transitives** (express 5, hono, ajv, zod, jose, cors). Sole consumer: `bin/mcp-server.js` (plus one probe in `bin/cli.js`). Programmatic `require('ruvector')` never touches it.
- `js-beautify` (~1 MB + editorconfig/nopt/glob chain): sole usage is `src/decompiler/index.js:87` **inside a try/catch that already treats it as optional** ("js-beautify not installed; return source as-is"). It is nonetheless a hard dependency.
- `commander` (~207 KB), `chalk@4` (~44 KB), `ora@5` (~41 KB + 8 transitive packages): used only by `bin/`.

### 3. Version drift across workspace ↔ registry ↔ platform pins

- **`@ruvector/attention` three-way drift**: workspace source (`crates/ruvector-attention-node`) says **0.1.4**; the registry has 0.1.32 and a **2.2.2** latest; `ruvector`'s `^0.1.3` resolves to **0.1.32** — users get code corresponding to no workspace state, and the 2.x line is unreachable from `ruvector`.
- `@ruvector/core@0.1.31` pins platform packages at **0.1.29** (no 0.1.30/0.1.31 platform builds exist) — loader and native binary versions diverge silently.
- `@ruvector/sona` local package.json pins platform deps at 0.1.5 while published 0.1.7 pins 0.1.7 — repo out of sync with what shipped.
- `@ruvector/rvf` local is 0.2.2 but `ruvector` declares `^0.1.0` → resolves to old 0.1.9. `@ruvector/diskann@0.1.1` pins platform pkgs at 0.1.0; its local dir is missing `index.js`/`index.d.ts` declared in `files` and contains a stray file literally named `false`.
- `@ruvector/wasm-unified@1.0.0` is published with **placeholder implementations** (`src/attention.ts:261-276` returns zero-filled `Float32Array`s; WASM init commented out; declared `dist/` doesn't exist) — silent wrong-results risk for any adopter.
- `crates/ruvector-gnn-node/npm/*/` has cross-platform binaries committed into wrong platform dirs (e.g. `npm/linux-x64-gnu/ruvector-gnn.darwin-arm64.node`) — 12 MB of git noise and a mispublish waiting to happen.

## Decision

1. **De-fat `@ruvector/sona` and `@ruvector/rvf-node`**: remove `"*.node"` from their `files` arrays; the napi loader + per-platform `optionalDependencies` (already declared) become the only distribution path, matching `@ruvector/core`. Add a clear loader error message for unsupported platforms. Saving: **−10 MB** per default install.
2. **Demote `@ruvector/rvf` from optionalDependency to optional peerDependency** of `ruvector` (same treatment as `diskann`/`router`/`pi-brain`/`ruvllm` — all already guarded by lazy try/catch requires in `src/index.ts:29,51`, `src/core/rvf-wrapper.ts:16`). Saving: **−7 MB** on default installs; users who want the RVF fallback install it explicitly.
3. **Move `js-beautify` to `optionalDependencies`** (zero code change — the try/catch fallback already exists). Move `@modelcontextprotocol/sdk` to an optional peer, since `bin/mcp-server.js` is only launched explicitly; `ruvector mcp start` prints an install hint when absent. Longer term, bundle `commander`/`chalk`/`ora` into the already-prebuilt `bin/cli.js` (esbuild) and drop them from `dependencies` (see ADR-274). Saving: **~10 MB** for library-only users.
4. **Version-coherence CI gate** (blocking, runs on every release PR):
   - For every hub package: `optionalDependencies` platform-package versions **must equal** the hub's own version.
   - Workspace package.json version must equal the version being published (no publish from dirty/stale trees).
   - `ruvector`'s declared ranges must resolve to the workspace versions in a clean lockfile simulation.
   - Fail on stray artifacts in platform dirs (`.node` files whose target triple doesn't match the directory).
5. **Resolve the attention drift explicitly**: either bump `ruvector` to `@ruvector/attention@^2.2.2` (after API review) or deprecate the 2.x line on npm; publish the workspace state either way. **Deprecate `@ruvector/wasm-unified@1.0.0`** on npm until real WASM is wired (it currently returns zeros).
6. **Repo hygiene**: remove stray cross-platform `.node` files from `crates/ruvector-gnn-node/npm/*/`; fix `@ruvector/diskann` `files` (missing entries, stray `false` file); adopt a single-source release script (changesets or equivalent) that bumps hub + platform packages atomically.

## Consequences

**Positive**
- Default `npm install ruvector` drops from **~44 MB to ~15–17 MB** (with ADR-272 applied), with all capabilities reachable via explicit optional installs.
- "Loader v0.1.31 / binary v0.1.29"-class silent divergence becomes a CI failure instead of a production surprise.
- No more published packages whose source cannot be located at any workspace commit.

**Negative / Risks**
- Users relying on the implicit RVF fallback must now install `@ruvector/rvf` — needs a loud CHANGELOG entry and a runtime hint message.
- De-fatting sona/rvf-node changes install behavior on platforms without prebuilt satellites (previously the fat package guaranteed *some* binary): the loader must fail with an actionable message, and the platform matrix should be reviewed before flipping.
- MCP SDK as optional peer means `ruvector mcp start` errors on a bare install — acceptable given it is an explicit opt-in command, but must be documented.

## Implementation Plan

| Phase | Work | Verification |
|---|---|---|
| 1 | js-beautify → optional; rvf → optional peer; CHANGELOG + runtime hints | integration tests with/without optional packages |
| 2 | De-fat sona + rvf-node (`files` fix); platform-matrix audit; loader error paths | install-size assertion in CI (`npm pack --dry-run` size budget per package) |
| 3 | Version-coherence CI gate; release script; gnn-node/diskann cleanup; attention-drift resolution; wasm-unified deprecation | CI gate red/green on synthetic drift fixtures |
| 4 | MCP SDK → optional peer; CLI bundling (with ADR-274) | `ruvector mcp start` hint test; library-only install footprint measurement |
