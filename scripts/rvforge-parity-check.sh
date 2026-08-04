#!/usr/bin/env bash
# CLI ↔ Rust registry parity check.
#
# `npm/packages/rvforge` (TypeScript) writes a local registry; `crates/
# rvforge-registry` (Rust) reads one. Both implement
# `docs/research/rvf-forge/registry-model.md`, and they were built
# independently — so "both pass their own tests" says nothing about whether
# they interoperate.
#
# This script closes that gap end to end: it drives the real CLI through
# init → pack → publish against a synthetic `.rvf`, publishes a *second*
# release so the run exercises lineage rather than just a single object, then
# hands the resulting directory to `rvforge-registry-check`, which re-derives
# every content address, recomputes the Merkle log, re-applies the publication
# rules, and walks the witness chains using nothing but the Rust crate's
# public API.
#
# Prints `PARITY OK` and exits 0 when the Rust reader accepts what the CLI
# wrote; otherwise prints the violation list and exits non-zero.
#
# Usage: scripts/rvforge-parity-check.sh [--keep]
#   --keep   leave the temporary registry on disk and print its path

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI_DIR="$REPO_ROOT/npm/packages/rvforge"
CRATE_DIR="$REPO_ROOT/crates/rvforge-registry"

KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1

for required in "$CLI_DIR/package.json" "$CRATE_DIR/Cargo.toml"; do
  if [[ ! -f "$required" ]]; then
    echo "parity check needs $required; skipping" >&2
    exit 0
  fi
done

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/rvforge-parity.XXXXXX")"
cleanup() {
  if [[ "$KEEP" -eq 1 ]]; then
    echo "kept: $WORKDIR"
  else
    rm -rf "$WORKDIR"
  fi
}
trap cleanup EXIT

PROJECT_DIR="$WORKDIR/project"
REGISTRY_DIR="$WORKDIR/registry"
mkdir -p "$PROJECT_DIR"

echo "==> building the CLI"
# --workspaces=false: the package sits inside the npm/ workspace root, and a
# plain install resolves the whole workspace and fails EBADPLATFORM on
# platform-pinned siblings.
(cd "$CLI_DIR" && npm install --workspaces=false --silent && npm run build --silent)

CLI="$CLI_DIR/dist/cli.js"
KEY_FILE="$PROJECT_DIR/rvforge-publisher.key"

echo "==> rvforge init --keygen"
# init scaffolds rvforge.json too, so it runs first; the fixture generator then
# replaces that scaffold with a project that passes every pack check.
(cd "$PROJECT_DIR" && node "$CLI" init --keygen --quiet >/dev/null)
node "$REPO_ROOT/scripts/rvforge-parity-fixture.cjs" "$CLI_DIR/dist" "$PROJECT_DIR" >/dev/null

echo "==> rvforge pack agent.rvf"
(cd "$PROJECT_DIR" && node "$CLI" pack agent.rvf --quiet >/dev/null)

echo "==> rvforge publish agent.rvf (1.0.0)"
(cd "$PROJECT_DIR" && node "$CLI" publish agent.rvf \
  --registry "$REGISTRY_DIR" --key-file "$KEY_FILE" --quiet >/dev/null)

# A second release is what makes the check meaningful: it exercises the
# predecessor link, the release index as a chain, a two-leaf Merkle tree, and a
# witness receipt that has to name the previous one.
echo "==> rvforge publish agent.rvf (1.1.0)"
node -e '
  const { readFileSync, writeFileSync } = require("node:fs");
  const path = process.argv[1];
  const project = JSON.parse(readFileSync(path, "utf8"));
  project.version = "1.1.0";
  writeFileSync(path, `${JSON.stringify(project, null, 2)}\n`);
' "$PROJECT_DIR/rvforge.json"
(cd "$PROJECT_DIR" && node "$CLI" publish agent.rvf \
  --registry "$REGISTRY_DIR" --key-file "$KEY_FILE" --quiet >/dev/null)

echo "==> rvforge-registry-check (Rust)"
if (cd "$REPO_ROOT" && cargo run --quiet -p rvforge-registry \
      --bin rvforge-registry-check -- "$REGISTRY_DIR"); then
  echo
  echo "PARITY OK — the Rust registry reads everything the CLI wrote"
  exit 0
fi

echo
echo "PARITY FAILED — the violations above are places the CLI and" >&2
echo "crates/rvforge-registry disagree about registry-model.md" >&2
exit 1
