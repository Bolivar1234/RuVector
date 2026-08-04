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
npx @ruvector/rvforge build agent.rvf          # local build
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

Produces the canonical build manifest, a staged bundle, a software inventory,
SHA256 checksums, and a provenance record:

```text
forge-out/
├── build-manifest.json   canonical, deterministic
├── inventory.json        software inventory
├── provenance.json       what was built, from what, by whom
├── checksums.txt         sha256sum-compatible
└── rvf/agent.rvf         embedded payload (or locator.json in thin mode)
```

The manifest is deterministic: keys, targets, and capability grants are sorted,
and nothing time-, path-, or host-dependent appears in it. The same logical
build description always serialises to the same bytes and the same
`manifestSha256`. Everything that varies per run lives in the provenance record.

Installer generation requires the Tauri packaging layer. Until that lands the
result is labelled `staged` and forge does not claim to have produced an
installer. A failed build leaves no output directory at all, rather than a
half-written one.

### `verify`

Recomputes every digest in a provenance record and re-derives the manifest hash
from the manifest's own bytes:

```bash
forge verify forge-out                    # a whole build directory
forge verify forge-out/rvf/agent.rvf      # one artifact
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
  inspection-only operations.
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
import { validateRvf, buildManifest, runVerify } from '@ruvector/rvforge';

const result = await validateRvf('agent.rvf', { deep: true });
const manifest = buildManifest({
  app: { name: 'My Agent', version: '1.0.0', publisher: 'Example', identifier: 'com.example.agent' },
  identity: result.identity,
  targets: ['windows', 'macos', 'linux'],
  packaging: { mode: 'embedded' },
  runtime: { profile: 'wasm', rvmVersion: '0.1.0', rvmCommit: 'abc1234' },
});
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
