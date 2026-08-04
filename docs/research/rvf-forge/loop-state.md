# RVForge Build Loop — Iteration State

> Read this FIRST each loop iteration. Update it LAST. Never redo completed
> steps. Branch: `feat/rvf-forge` · PR: #790 · Spec:
> `docs/research/rvf-forge/requirements.md` · ADRs: 283–293.

## Done criteria (overall)

PR #790 merged with green CI; `@ruvector/forge` published to npm; the
locally-testable requirements §15 acceptance criteria pass:
validate/build/verify round-trip on a sample RVF, identical embedded RVF
SHA256 across packaging outputs, tamper detection rejects modified
artifacts, no secrets in code or logs.

## Work plan (requirements §12 order, Rust/WASM first)

- [x] ADRs 283–293 authored, committed, pushed (commit 14541c870)
- [x] Canonical requirements doc committed
- [x] PR #790 opened
- [ ] 1. `@ruvector/forge` CLI scaffold under `npm/packages/forge`
       (TypeScript, tsc→dist, jest — follow `npm/packages/rvf` conventions;
       commands: init/validate/build/submit/status/download/verify;
       local RVF validation; canonical build manifest; stable error codes)
- [ ] 2. `crates/rvf-forge-core` Rust packaging/verification crate
       (manifest parse, Ed25519 verify, segment hash verify, provenance
       record, SHA256 checksums — NEVER executes RVF content)
- [ ] 3. Tauri RVF Reader app scaffold
- [ ] 4. rvm-* integration work items (rvm-rvf first; note: rvm is a
       separate repo github.com/ruvnet/rvm — for this repo, define the
       integration contract + compatibility matrix consumed by forge)
- [ ] 5. GitHub Actions build matrix (linux/windows/macos) for forge
- [ ] 6. Embedded + thin packaging modes
- [ ] 7. Provenance, inventory, witness receipts wired end-to-end
- [ ] 8. Tests green (cargo test -p rvf-forge-core, npm test in forge),
       lint clean, security review pass (default-deny caps, no RVF
       execution during packaging, no secrets)
- [ ] 9. CI green on PR #790 → merge
- [ ] 10. npm publish @ruvector/forge (NPM_TOKEN via GCP Secret Manager,
        project cognitum-20260110) — AFTER merge only

## Current iteration

- **Iteration**: 1 (2026-08-03 ~20:50 local)
- **In flight**: background coder agent `forge-scaffold` is scaffolding
  `npm/packages/forge` (step 1). BEFORE starting step 1 work, check
  whether `npm/packages/forge/package.json` exists and tests pass — if
  the agent finished, review + commit its output instead of rewriting.
- **Next action**: review/commit forge CLI scaffold; then start step 2
  (`crates/rvf-forge-core`).
- **Blockers**: none.

## Scope expansion (2026-08-03 late) — RVForge Platform

The user expanded RVForge into a five-product platform (see requirements
"RVForge Platform" part): **Store, Reader, Publisher, Registry,
Enterprise** — an agentic app store + runtime + registry + trust system
(Steam + npm + enterprise catalog). Additions to the work plan:

- [ ] P-ADR. ADR-294 (RVForge platform: marketplace objects, trust levels,
       review pipeline, security/countersigning model, licensing) — agent
       `adr-author-4` in flight; review, rename-sweep, commit when landed.
- [ ] P1. Publisher CLI verbs `pack | test | publish` added to the CLI
       (union with existing init/validate/build/submit/status/download/
       verify). `pack` = validate + capability manifest + listing
       metadata; `test` = quarantined capability-denial/malformed-input/
       checkpoint-recovery tests (local, never executes RVF outside
       sandbox); `publish` = signed release record upload to registry.
- [ ] P2. Registry data model (content-addressed releases, immutable,
       predecessor-linked, transparency log) — schema + local registry
       implementation first, hosted later.
- [ ] P3. Reader = the Tauri app (step 3) grows store/library/runtime/
       update UX per requirements P5–P9; capability cards mandatory.
- [ ] P4. Trust levels + revocation semantics (revocation blocks
       execution, never deletes local RVFs or state).

MVP focus stays: CLI + core crate + Reader + WASM quarantine + registry
schema. Store web UI and payments are NOT in this loop's scope unless the
user says so.

## Decisions / assumptions log

- Product name: **RVForge** (user directive; latest capitalization).
- npm package name: **`@ruvector/rvforge`** (bin `rvforge`) — the platform
  spec's Publisher CLI uses `npx @ruvector/rvforge`; supersedes the
  earlier `@ruvector/forge` pin. ASSUMPTION (cheap to reverse): one
  package carries both build verbs and publisher verbs. If the scaffold
  agent delivered `npm/packages/forge` as `@ruvector/forge`, rename
  package+dir to `rvforge` before committing.
- Rust/WASM agents first; Python/Node portability deferred (vision doc).
- Merge + publish are explicitly authorized by the user for this loop,
  gated on green CI and §15 local acceptance criteria.
- rvm backend crates live in the separate ruvnet/rvm repo; this repo owns
  the forge side (contract, CLI, core crate, reader).
