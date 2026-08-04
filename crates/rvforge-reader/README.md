# RVForge Reader

Tauri v2 desktop reader for signed `.rvf` agent packages — the host described
in [ADR-289](../../docs/adr/ADR-289-desktop-host-adapters.md). This is a
**scaffold**: the screens, the capability derivation, the runtime selection
order, and the state layout are real; RVF parsing, signature verification, and
execution are not implemented and are clearly marked as stubs.

## Layout

```text
src/inspect.rs      read a package without executing it (STUB)
src/capability.rs   derive the P6 install-time capability contract
src/runtime.rs      apply the FR004 runtime selection order
src/state.rs        ADR-288 encrypted state-capsule layout (encryption STUB)
src/commands.rs     Tauri command wrappers (feature `desktop`)
src/lib.rs          module root + the security invariants this crate holds
ui/                 three screens, plain HTML/CSS/JS, no build step
assets/             vendored copy of compatibility-matrix.json
capabilities/       Tauri v2 ACL: file dialog only
tests/              runtime selection, capability card, inspection, state
```

## Build and test

The crate is a **standalone workspace** (its `Cargo.toml` carries an empty
`[workspace]` table) and is listed in the repo root's `workspace.exclude`. A
Tauri app pulls a large, app-specific dependency graph — webview bindings,
bundler, plugin build scripts — that every `cargo build --workspace` in the
parent repo would otherwise pay for.

Tauri is behind an optional `desktop` feature, so the core logic builds and
tests with only `serde` and `serde_json`:

```bash
cd crates/rvforge-reader
cargo check          # core only — no webview packages needed
cargo test           # 39 tests, no Tauri dependency
```

This is the CI-testable path. Building the desktop shell additionally needs the
platform webview development packages (`libwebkit2gtk-4.1-dev`,
`libjavascriptcoregtk-4.1-dev`, `libsoup-3.0-dev` on Debian/Ubuntu; WebView2 on
Windows; Xcode command line tools on macOS):

```bash
cargo check --features desktop
cargo run   --features desktop     # launches the window
```

`cargo tauri dev` and `cargo tauri build` need the CLI
(`cargo install tauri-cli --version '^2'`). There is no Node toolchain: the
frontend is static files under `ui/`, referenced by `build.frontendDist`.

Before producing installers, regenerate the icon set with
`cargo tauri icon icons/icon.png` — the committed PNGs are placeholders and
there is no `.ico` or `.icns` yet, which Windows and macOS bundling require.

## What is stubbed, and why

| Stub | Current behavior | Replaced by |
|---|---|---|
| `inspect::inspect` | Reads path metadata only. Reports `identity: null`, `verification: unverified`, `signature: unverified`. | `rvf-forge-core` over the `rvm_inspect` C ABI (ADR-289 §4) |
| `inspect::capability_card` | Reads a `<file>.rvf.manifest.json` development sidecar if present; otherwise returns the everything-denied card. | The signed `CapabilityManifest` segment inside the RVF |
| `state::seal` / `unseal` | Return `EncryptionNotImplemented`. | AEAD sealing bound to the base identity, with customer-held keys (ADR-288) |
| Install / Customize permissions | Disabled buttons. | The install flow, once verification is real |
| Emergency controls | Disabled buttons. | `rvm-ffi` lifecycle calls (pause, terminate, revoke, rollback) |
| Witness status | "no witness chain". | `rvm witness` export (ADR-289 §3) |

The stubs report *absence*, never a benign default. `VerificationStatus` and
`SignatureStatus` have a single variant, `Unverified`, so no code path can
accidentally default to "verified"; the `Verified` and `Failed` variants arrive
with the code that can actually produce them.

## Security invariants

These are enforced in the library and covered by tests. They must survive the
replacement of every stub above.

1. **RVF content is never executed.** No code path loads, links, or interprets
   a segment. `inspect` and `verify` must be safe on an untrusted package
   (ADR-289 §3).
2. **Verification precedes any load**, and an unchecked package is reported as
   unverified rather than assumed good.
3. **Capability rendering is default-deny.** Every one of the fifteen ADR-286
   classes that the manifest does not request appears in the "cannot" list; a
   missing or rejected manifest yields a card that grants nothing.
4. **No vague permission prose.** Broad scopes (`all-files`, `*`,
   `unrestricted`) and banned phrases such as "access your computer" are
   rejected at derivation time, not filtered in the UI (requirements P6).
5. **No network calls.** Nothing here opens a socket. The packaged app's CSP
   has no remote origin in any directive, and the asset protocol is disabled.
6. **The runtime order is not configurable.** It is read from the vendored
   compatibility matrix, which is compiled in via `include_str!` so that
   swapping a file on the installed machine cannot reorder it. ADR-289 permits
   reordering only by signed policy; that path is not implemented, and
   `policy_source` always reports `embedded-default`.
7. **Hosted mode does not claim bare-metal isolation.** The card shows the
   isolation class the matrix records for the selected profile —
   `os-sandbox+wasm`, never `partition` (ADR-285).

Today `HostProfile::detect()` claims no OS confinement, no KVM, and no measured
boot, because the `rvm-host` adapters do not exist yet. Selection therefore
lands on plain `wasm`. That is the honest answer, and it will move up the order
as adapters land — not before.

## Screens

1. **Open** — pick or type a path, see file identity and signature status.
   Unverified renders as a warning, not a neutral state.
2. **Capabilities** — the P6 contract: "This agent requests" beside "This agent
   cannot", with Install / Customize Permissions / Cancel.
3. **Runtime** — selected profile, isolation class, mechanisms engaged, the
   selection order and where it came from, a per-profile eligibility table, and
   the emergency controls (Pause · Terminate · Disconnect Network · Revoke
   Capabilities · Rollback State) as disabled placeholders.

## Keeping the matrix in sync

`assets/compatibility-matrix.json` is a vendored copy of
`docs/research/rvf-forge/compatibility-matrix.json`. A test compares the two
byte-for-byte and fails on drift; update both together.
