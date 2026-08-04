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
- [x] 1. `@ruvector/rvforge` CLI — npm/packages/rvforge, 73 tests green, committed 4b1cb7551
       (TypeScript, tsc→dist, jest — follow `npm/packages/rvf` conventions;
       commands: init/validate/build/submit/status/download/verify;
       local RVF validation; canonical build manifest; stable error codes)
- [x] 2. `crates/rvf-forge-core` — 103 tests + clippy + fmt green, committed ce25d787c
       (manifest parse, Ed25519 verify, segment hash verify, provenance
       record, SHA256 checksums — NEVER executes RVF content)
- [x] 3. Tauri RVF Reader scaffold — crates/rvforge-reader, 39 tests green, committed 2db655fe1 (inspect stubbed pending rvf-forge-core FFI; dock next per ADR-295)
- [~] 4. rvm-* integration — compatibility-matrix.json v1 done (this repo side); rvm-rvf crates live in ruvnet/rvm (rvm-rvf first; note: rvm is a
       separate repo github.com/ruvnet/rvm — for this repo, define the
       integration contract + compatibility matrix consumed by forge)
- [x] 5. GitHub Actions build matrix — .github/workflows/rvforge-ci.yml (3-OS matrix for CLI npm test + cargo test/clippy/fmt; tolerant of pending package rename)
- [~] 6. Embedded + thin packaging modes — agent `forge-packaging` in flight
- [~] 7. Provenance, inventory, witness receipts — agent `forge-packaging` in flight (CLI side)
- [ ] 8. Tests green (cargo test -p rvf-forge-core, npm test in forge),
       lint clean, security review pass (default-deny caps, no RVF
       execution during packaging, no secrets)
- [ ] 9. CI green on PR #790 → merge
- [ ] 10. npm publish @ruvector/forge (NPM_TOKEN via GCP Secret Manager,
        project cognitum-20260110) — AFTER merge only

## Current iteration

- **Iteration**: 9. Reader scaffold + ADR-295 committed (2db655fe1). In flight: forge-packaging (CLI packaging modes), registry-core (registry crate). Next after those land: dock implementation in reader (P5), publisher verbs pack/test/publish (P1), rvf-forge-core FFI into reader, ADR status flips.
- **In flight**:
  - forge-scaffold: DONE (renamed to rvforge, committed). Was: package.json/
    src/tsconfig exist, tests not yet; still running. Do NOT touch its
    directory until it reports; then review, RENAME to
    `npm/packages/rvforge` + `@ruvector/rvforge` (bin `rvforge`), run
    npm test, commit.
  - forge-core: DONE (committed ce25d787c). Was: spawned this
    iteration. On completion: review, `cargo test -p rvf-forge-core`,
    `cargo clippy -p rvf-forge-core -- -D warnings`, commit.
- **Next action**: when either agent reports, review + test + commit its
  output. If both still running at next fire, start step P2 (registry
  data model schema under npm/packages or docs — content-addressed
  releases, immutable, predecessor-linked, transparency log JSON schema).
- **Blockers**: none.
- ADR-294 committed (a6efab480) and pushed.

## Scope expansion (2026-08-03 late) — RVForge Platform

The user expanded RVForge into a five-product platform (see requirements
"RVForge Platform" part): **Store, Reader, Publisher, Registry,
Enterprise** — an agentic app store + runtime + registry + trust system
(Steam + npm + enterprise catalog). Additions to the work plan:

- [x] P-ADR. ADR-294 (RVForge platform: marketplace objects, trust levels,
       review pipeline, security/countersigning model, licensing) — agent
       `adr-author-4` in flight; review, rename-sweep, commit when landed.
- [ ] P1. Publisher CLI verbs `pack | test | publish` added to the CLI
       (union with existing init/validate/build/submit/status/download/
       verify). `pack` = validate + capability manifest + listing
       metadata; `test` = quarantined capability-denial/malformed-input/
       checkpoint-recovery tests (local, never executes RVF outside
       sandbox); `publish` = signed release record upload to registry.
- [~] P2. Registry data model — wire-format contract done (registry-model.md); local file-backed implementation pending (content-addressed releases, immutable,
       predecessor-linked, transparency log) — schema + local registry
       implementation first, hosted later.
- [ ] P3. Reader = the Tauri app (step 3) grows store/library/runtime/
       update UX per requirements P5–P9; capability cards mandatory.
- [ ] P4. Trust levels + revocation semantics (revocation blocks
       execution, never deletes local RVFs or state).
- [~] P5. RVForge Agent Dock (requirements "RVForge Agent Dock" D1–D8;
       ADR-295 being authored by agent `adr-author-5`). Security/control
       surface: RVForge-owned chrome, agent content strictly separated
       (spoofing defense), 8 states, one-action pause/terminate,
       event-threshold noise control. First implementation target:
       dock window + runtime screen in crates/rvforge-reader AFTER the
       reader scaffold lands (do not collide with reader-scaffold agent;
       dock implementation waits for its completion). Acceptance: identify
       + understand + inspect + terminate within 5 seconds / 2
       interactions.

SCOPE (user directive 2026-08-03 late): fully implement ALL RVForge
ADRs 283–295 until production ready, published, and merged. The earlier
web-UI/payments guardrail is lifted to the extent an ADR requires it —
P15 MVP items (registry, publisher verbs, capability cards, witness
viewer, revocation, private catalogs) are IN scope. Cross-repo rvm-*
items are contract/stub-side only in this repo (blocker: ruvnet/rvm is a
separate repo). As each ADR's scope lands, flip its Status to
Implemented with an Updated date.

Additional in-flight (iteration 8):
- [~] P2-impl. crates/rvforge-registry — agent `registry-core` building
  the file-backed content-addressed registry with ed25519 release rules,
  trust-level raise enforcement, non-destructive revocation, Merkle
  transparency log, witness chains.
- Cron job recreated as e34a0b75 (ADR-283..295 wording); old 79a54ad6
  deleted.

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
