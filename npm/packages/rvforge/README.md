# @ruvector/rvforge

**RVForge — one canonical RVF to signed platform installers.**

`forge` turns a single `.rvf` agent into installable packages for Windows,
macOS, Linux, and RVM, without the agent inside each package ever differing.
The RVF identity, contents, policies, and signatures are the same in the
`.dmg` as in the `.msi`.

See [ADR-283](../../../docs/adr/ADR-283-rvf-forge-canonical-installer-pipeline.md)
for the design and `docs/research/rvf-forge/requirements.md` for the full
requirements.

Requires Node.js 20 or later.

## Commands

```bash
npx @ruvector/rvforge init                     # scaffold forge.config.json
npx @ruvector/rvforge validate agent.rvf       # structural check, no execution
npx @ruvector/rvforge build agent.rvf          # local build (--mode embedded|thin)
npx @ruvector/rvforge submit agent.rvf --yes   # hosted build
npx @ruvector/rvforge status BUILD_ID
npx @ruvector/rvforge download BUILD_ID
npx @ruvector/rvforge verify AgentSetup.exe
```

Targets are `windows-x64`, `windows-arm64`, `macos-x64`, `macos-arm64`,
`macos-universal`, `linux-x64`, `linux-arm64`, and `rvm`, with `windows`,
`macos`, and `linux` accepted as aliases. Pass them after the `.rvf`:

```bash
npx @ruvector/rvforge build agent.rvf windows macos linux
```

Omit the `.rvf` and forge uses `rvf` from the config; omit the targets and it
uses `targets`.

### `validate`

Checks that the file is a well-formed RVF: root-manifest magic and CRC32C,
manifest field consistency, a segment-header walk, and signature presence. It
reads bytes and never executes, links, or interprets RVF content.

The default pass touches one 4 KiB page plus one 64-byte header per segment,
which keeps it under two seconds for RVFs below 1 GB. `--deep` additionally
streams the whole file through SHA256 to compute the canonical RVF identity.

```bash
forge validate agent.rvf --deep
forge validate agent.rvf --allow-unsigned   # development builds only
```

An RVF carrying executable segments (kernel, eBPF, WASM) with no root-manifest
signature fails with `FORGE_E_UNSIGNED_SEGMENT` unless `--allow-unsigned` is
passed.

### `build`

Produces the canonical build manifest, one staged bundle per target, a software
inventory, SHA256 checksums, a provenance record, and a witness receipt:

```text
forge-out/
├── build-manifest.json                canonical, deterministic
├── inventory.json                     software inventory + bundle layout
├── provenance.json                    what was built, from what, by whom
├── checksums.txt                      sha256sum-compatible
├── receipts.jsonl                     witness chain, append-only
└── bundles/
    └── <target>/
        ├── rvf/agent.rvf              embedded payload…
        ├── rvf/locator.json           …or the signed locator, in thin mode
        └── reader/reader-slot.json    where the RVF Reader goes
```

The manifest is deterministic: keys, targets, and capability grants are sorted,
and nothing time-, path-, or host-dependent appears in it. The same logical
build description always serialises to the same bytes and the same
`manifestSha256`. Everything that varies per run lives in the provenance record.

Installer generation requires the Tauri packaging layer. Until that lands the
result is labelled `staged` and forge does not claim to have produced an
installer. A failed build leaves no output directory at all, rather than a
half-written one.

### Packaging modes

`--mode` selects how the RVF reaches each bundle, overriding
`packaging.mode` in the config for one build:

```bash
forge build agent.rvf --mode embedded    # the bundle carries the whole RVF
forge build agent.rvf --mode thin        # the bundle carries a signed locator
```

**Embedded** (FR001) stages the complete RVF into every target's bundle, so the
package runs with no network access. The same bytes go into every bundle — that
is core invariant 1, *the embedded RVF hash must be identical across every
platform package* — and forge does not take the copy on trust: it re-hashes each
staged copy and fails the build with `FORGE_E_VERIFY_FAILED` if any of them
diverges, rather than shipping platform packages that disagree about what the
agent is.

**Thin** (FR002) stages a signed RVF locator instead of the payload: the
distribution URL, the RVF identity and size, the capability-policy hash, and a
signature *slot*. The reader resolves the locator, checks the digest it names,
and verifies before executing. `packaging.distributionUrl` is required — a thin
package with nowhere to fetch from is rejected at manifest generation.

The reader slot in each bundle is a JSON descriptor, never an executable.
Nothing in a staged bundle is runnable.

### Compatibility enforcement

Forge refuses any packaging-mode / target / runtime-profile combination absent
from the published RVM compatibility matrix (ADR-291 §2), before the RVF is read
and before anything is uploaded:

```console
$ forge build agent.rvf rvm
error: mode=embedded target=rvm runtime=wasm is absent from the RVM compatibility
       matrix — runtime profile "wasm" has no platform entry for target "rvm"
       (os "rvm", arch x64). Closest supported combination: mode=embedded
       target=linux-arm64 runtime=wasm.
code:  FORGE_E_UNSUPPORTED_TARGET
```

Forge never approximates, downgrades, or substitutes a runtime to make an
unsupported request succeed; it names the nearest supported combination and
lets you decide. The matrix is hash-addressed, and both `provenance.json` and
`inventory.json` record the revision that admitted the build, so a past
admission decision can be reconstructed.

`src/compatibility-matrix.json` is vendored. **The canonical copy is
`docs/research/rvf-forge/compatibility-matrix.json`** — change it there and
re-copy; `tests/compat.test.ts` fails when the two diverge.

### Witness receipts

Every build and every verification appends a receipt to `receipts.jsonl`,
following the registry data model. A receipt's id is the SHA256 of its canonical
JSON (excluding `receiptId` and `signatures`), and receipts hash-chain per
subject through `prevReceipt`:

```json
{"schemaVersion":1,"type":"witness-receipt","receiptId":"sha256:…","subject":"sha256:…",
 "event":"build","outcome":"pass","actor":{"kind":"builder","id":"@ruvector/rvforge@0.1.0"},
 "evidence":{…},"timestamp":"2026-08-03T00:00:00.000Z","prevReceipt":null,"signatures":[]}
```

`verify` checks the chain before it appends to it: an edited receipt fails
because its recomputed id no longer matches, and a removed or reordered one
fails because the link no longer resolves. Either way the result is
`FORGE_E_VERIFY_FAILED`, and forge does not append onto a chain it just refused.

`signatures` is always empty on output — forge holds signing references, never
key material, so the array is a slot for a signing worker to fill.

`receipts.jsonl` is deliberately absent from `provenance.json`: it is
append-only and grows on every verification, so recording its digest would make
a build's own provenance fail the moment the chain was extended.

### `verify`

Recomputes every digest in a provenance record, re-derives the manifest hash
from the manifest's own bytes, and checks the witness chain:

```bash
forge verify forge-out                                  # a whole build directory
forge verify forge-out/bundles/linux-x64/rvf/agent.rvf  # one artifact
forge verify Agent.dmg --provenance path/to/provenance.json
```

Re-deriving the manifest hash from content is what makes the check meaningful:
rewriting the manifest *and* patching its recorded digest still fails.

### Hosted builds

`submit`, `status`, and `download` talk to the hosted build service. The
service is not deployed yet, so these currently fail with `FORGE_E_NETWORK`
against a live endpoint; the request and response shapes are the contract it
will be built to.

```bash
export FORGE_API_URL=https://forge.example
export FORGE_API_TOKEN=...        # never pass a token as a flag
forge submit agent.rvf --yes
```

`submit` validates locally, builds the manifest, and prints the estimated build
time, output size, and cost before anything is uploaded. The upload itself
requires `--yes`, so an unattended run cannot ship a confidential RVF to a
remote worker by accident. Downloads are hashed on arrival and discarded on a
digest mismatch.

## Unattended use

Every command accepts `--json` and writes a single envelope to stdout:

```json
{"ok": true, "command": "validate", "data": { }, "exitCode": 0}
{"ok": false, "command": "verify", "error": {"code": "FORGE_E_VERIFY_FAILED", "message": "..."}, "exitCode": 9}
```

Exit codes are stable:

| Code | Exit | Meaning |
|---|---|---|
| `FORGE_E_USAGE` | 2 | Bad arguments |
| `FORGE_E_INVALID_RVF` | 3 | Missing, truncated, bad magic, failed checksum |
| `FORGE_E_UNSIGNED_SEGMENT` | 4 | Executable segment with no signature |
| `FORGE_E_UNSUPPORTED_TARGET` | 5 | Target outside the supported matrix |
| `FORGE_E_MANIFEST` | 6 | Malformed or inconsistent build manifest |
| `FORGE_E_NETWORK` | 7 | Build service unreachable |
| `FORGE_E_AUTH` | 8 | Credentials missing or rejected |
| `FORGE_E_VERIFY_FAILED` | 9 | A recomputed digest diverged |
| `FORGE_E_IO` | 10 | Filesystem failure |
| `FORGE_E_NOT_FOUND` | 11 | Unknown build, artifact, or record |
| `FORGE_E_CONFIG` | 12 | Bad `forge.config.json` |
| `FORGE_E_TOOLCHAIN` | 13 | Required local build tool unavailable |
| `FORGE_E_INTERNAL` | 20 | A bug in forge |

## Capability policy

`forge init` writes a default-deny policy: every capability class is present
and every allowlist is empty. Grants have to be added explicitly.

```json
{
  "capabilityPolicy": {
    "defaultDeny": true,
    "allow": {
      "network": ["https://api.example.com"],
      "filesystem": ["~/Documents/reports"],
      "devices": [], "memory": [], "models": [], "state": [], "tools": []
    }
  }
}
```

The policy is hashed into `capabilityPolicyHash` and embedded in every package,
so a reviewer can confirm the permissions an installer ships with match the
ones they approved. `defaultDeny` is forced to `true`; a policy that opts out
is rejected rather than silently accepted.

## Security

- **RVF content is never executed.** Validation, packaging, and scanning are
  inspection-only operations. Embedded mode copies bytes, thin mode writes a
  locator, and the reader slot is a descriptor rather than a binary — nothing
  in a staged bundle is runnable.
- **The build path makes no network calls.** A locator records where an RVF
  will be fetched from; forge does not fetch it.
- **Signing references only.** Forge records which key to ask for and which
  service holds it. Private key material is never read, stored, logged, or
  transmitted; signing happens on a worker with HSM or KMS access.
- **Credentials come from the environment.** `FORGE_API_TOKEN` is never
  accepted as a flag, so it stays out of shell history and process listings.
- **Messages are redacted.** Bearer tokens, URL userinfo, and long opaque
  strings are stripped from error messages and detail bags before they are
  printed.

## Library use

```ts
import {
  assertCompatible,
  buildManifest,
  readReceipts,
  runVerify,
  validateRvf,
  verifyReceiptChain,
} from '@ruvector/rvforge';

assertCompatible({ mode: 'embedded', targets: ['linux-x64'], runtimeProfile: 'wasm' });

const result = await validateRvf('agent.rvf', { deep: true });
const manifest = buildManifest({
  app: { name: 'My Agent', version: '1.0.0', publisher: 'Example', identifier: 'com.example.agent' },
  identity: result.identity,
  targets: ['windows', 'macos', 'linux'],
  packaging: { mode: 'embedded' },
  runtime: { profile: 'wasm', rvmVersion: '0.1.0', rvmCommit: 'abc1234' },
});

await runVerify('forge-out');
verifyReceiptChain(await readReceipts('forge-out')); // { ok: true, … }
```

## Development

```bash
npm run build      # tsc → dist/
npm test           # jest
npm run typecheck
```

`tests/fixtures/minimal.rvf` is a 4.5 KB synthetic file generated by
`tests/fixtures/rvf-fixture.ts`; a test asserts the two agree, so the committed
bytes cannot drift from the builder. It contains zero-filled payloads and no
model data.
