# ADR-274: RuVector CLI Startup and Hooks Hot-Path Optimization

**Status**: Proposed
**Date**: 2026-07-02
**Authors**: RuVector NPM Deep Review Swarm
**Supersedes**: None
**Related**: ADR-272 (Distribution Slimming), ADR-273 (Dependency Diet), ADR-275 (Runtime Hot Paths)

---

## Context

The `ruvector` CLI (`bin/cli.js`: **445 KB, 10,245 lines, single file, ~186 `.command()` registrations** across 26 top-level groups) has genuinely good lazy-load discipline at the top level — only `commander`, `chalk`, `fs`, `path`, and `package.json` are eager (lines 6–10, 126); `ora`, `dist/index.js`, `@ruvector/gnn`, `@ruvector/attention` and ~100 handler-local requires are lazy and memoized (a prior optimization documented in-file: "reduces CLI startup from ~1000ms to ~70ms"). Cold `--help` is estimated at 80–120 ms. But the review found four structural costs that discipline alone can't fix:

### 1. The hooks hot path pays full CLI startup per invocation (HIGH)

`hooks pre-edit` / `post-edit` / `pre-command` / `post-command` (`bin/cli.js:4626-4684`) are designed to be wired into Claude Code hooks — they fire on **every tool use in every session**. Each invocation is a fresh `node bin/cli.js`:

> Node boot (~40–50 ms) + 445 KB parse + 186 command registrations (~1,413 chalk call sites in description strings) + `loadIntelligenceEngine()` (`bin/cli.js:2971-2981`) → `dist/core/intelligence-engine.js` → agentdb-fast + sona-wrapper + onnx-embedder + parallel-intelligence + native `@ruvector/core` dlopen attempt + JSON read of `~/.ruvector/intelligence.json`

Plausibly **150–400 ms per hook fire, multiplied by every edit/command** in a session. This is the single biggest real-world UX cost in the package.

### 2. `dist/core/index.js` is an eager 24-module barrel (MEDIUM)

The first `loadRuvector()` pulls `dist/index.js`, which `__exportStar`s `./core` — eagerly evaluating ~25 modules / ~16.5k lines (~350 KB): intelligence-engine (1,436 lines), neural-embeddings (1,383), adaptive-embedder (1,089), parallel-workers (910), onnx-embedder, router/graph/cluster wrappers, etc. `ruvector search` needs only `VectorDB`. Additionally, `src/index.ts:38-77` performs the native `require('@ruvector/core')` (NAPI dlopen, 10–50 ms) **at module load**, even for consumers that only want `Utils` or the embedding service — it is the sole eager loader in an otherwise lazy codebase.

### 3. Monolithic parse + registration cost on every invocation (MEDIUM)

A typical invocation executes <1% of cli.js (e.g. `ruvector search` runs one ~60-line handler) but pays full parse + registration, including the ~3,500-line `hooks` group. Estimated 20–40 ms pure overhead per invocation.

### 4. The startup-budget guard cannot catch drift (LOW)

`test/startup-budget.js` enforces `ABS_BUDGET_MS = 2000` and a 120 ms delta only for `harness status --json` vs `--help`. A regression from 100 ms → 1,900 ms would pass CI, and the hooks path — the true hot path — has no delta guard at all.

Non-findings worth recording: the MCP server is fully isolated (SDK required only inside `bin/mcp-server.js`; regular CLI never loads it); process lifecycle is clean (no top-level signal handlers, no event-loop retention); there are **no postinstall scripts** — install cost is purely payload size (ADR-272/273).

## Decision

1. **Dedicated micro-entry for hooks: `bin/hooks.js`.** A commander-free, chalk-free entry that dispatches on `process.argv` (hooks need ~5 flags, not a framework), lazily loads only the intelligence modules the specific hook needs, and is registered in `bin` as `ruvector-hooks`. The main CLI keeps `hooks` subcommands for interactive use and delegates to the same handler modules.
   Target: **hook fire < 80 ms** cold (Node boot + minimal parse), measured, not estimated.
2. **Hooks daemon mode (phase 2).** First hook invocation starts a background process holding the warm engine (native bindings + intelligence state); subsequent invocations talk to it over a unix socket with a fire-and-forget fallback to in-process execution if the daemon is unreachable. Target: **hook fire < 15 ms** warm. (The MCP server path already demonstrates the long-lived-process pattern.)
3. **De-barrel `dist/core/index.js`**: convert re-exports to lazy getters (`Object.defineProperty(exports, 'X', { get: () => require('./x').X })`), and move the native `@ruvector/core` selection in `src/index.ts` into a memoized `getImplementation()` invoked from the `VectorDBWrapper` constructor (mirroring the existing `router-wrapper.ts:12-26` pattern). Library `require('ruvector')` becomes O(entry file) instead of O(whole core).
4. **Enable Node's compile cache**: one line at the top of `bin/cli.js` — `require('node:module').enableCompileCache?.()` (Node ≥22, no-op otherwise) — cuts repeat parse cost of the 445 KB file with zero restructuring. Command-group splitting into lazily-required modules is the follow-up once measured wins justify it.
5. **Tighten the budget guard**: `ABS_BUDGET_MS` 2000 → **500 ms**; add a delta check for `hooks pre-edit` (or `hooks stats`) vs `--help`; record medians as CI artifacts to make drift visible over time.
6. **Dependency cleanup tier** (with ADR-273): keep chalk@4/ora@5 for now (last CJS majors — a deliberate, correct choice), but plan chalk → `node:util.styleText` when `engines` moves to ≥20.12, and note ora is already lazy. `mcp start` should `spawn` the server rather than in-process `require` so the long-lived server doesn't retain ~80 KB of registered commander state.

## Consequences

**Positive**
- Hooks stop taxing every Claude Code tool call: 150–400 ms → <80 ms (phase 1) → <15 ms (daemon). Over a 200-tool-call session this is 30–80 seconds of removed latency.
- Library consumers stop paying NAPI dlopen + 350 KB of parse for `require('ruvector')`.
- Startup regressions become CI failures at 500 ms instead of 2,000 ms, on both the harness and hooks paths.

**Negative / Risks**
- A daemon introduces lifecycle concerns (stale state after upgrades, socket permissions, orphan cleanup). Mitigations: version handshake (daemon self-terminates on version mismatch), socket under `~/.ruvector/` with 0600, idle self-shutdown, and the always-available in-process fallback.
- Lazy getters on the core barrel change `Object.keys(require('ruvector'))` enumeration semantics; property descriptors must remain enumerable to avoid breaking consumers that introspect exports.
- Two `bin` entries need docs and Claude Code hook-template updates.

## Implementation Plan

| Phase | Work | Verification |
|---|---|---|
| 1 | `bin/hooks.js` micro-entry; compile-cache one-liner; budget guard tightening (+hooks delta) | `test/startup-budget.js` extended; median timings in CI artifacts |
| 2 | De-barrel core index (lazy getters); memoized `getImplementation()` in `src/index.ts` | library-load microbenchmark (`node -e "require('ruvector')"` timing) + full `npm test` |
| 3 | Hooks daemon (unix socket, version handshake, idle shutdown, fallback) | soak test: 1,000 sequential hook fires, p50/p99 latency + orphan-process check |
| 4 | Command-group lazy splitting; `mcp start` spawn isolation | `--help` timing; MCP RSS comparison |
