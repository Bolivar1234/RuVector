#!/usr/bin/env python3
"""Inspect candidate changes before any candidate-controlled process executes."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]{12,}", re.I),
]
ADR_RE = re.compile(r"^ADR-(\d+)-.*\.md$")
PACKAGE_RE = re.compile(r'^\s*name\s*=\s*"([^"]+)"\s*$')


def git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def package_names(root: Path) -> set[str]:
    names: set[str] = set()
    for cargo in root.rglob("Cargo.toml"):
        if any(part in {".git", "target"} for part in cargo.parts):
            continue
        for line in cargo.read_text(encoding="utf-8", errors="replace").splitlines():
            match = PACKAGE_RE.match(line)
            if match:
                names.add(match.group(1))
                break
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = Path(args.base).resolve()
    candidate = Path(args.candidate).resolve()
    base_sha = git(base, "rev-parse", "HEAD")
    head_sha = git(candidate, "rev-parse", "HEAD")
    changed = git(candidate, "diff", "--name-only", "--diff-filter=ACMRTUXB", base_sha, head_sha).splitlines()
    if not changed:
        raise SystemExit("candidate has no changes against trusted base")

    base_adrs = {
        int(match.group(1))
        for path in (base / "docs" / "adr").glob("ADR-*.md")
        if (match := ADR_RE.match(path.name))
    }
    seen_new_adrs: set[int] = set()
    base_packages = package_names(base)
    forbidden_lock_changes = {"Cargo.lock", "package-lock.json", "pnpm-lock.yaml", "yarn.lock"}
    for rel_value in changed:
        rel = Path(rel_value)
        if rel.name in forbidden_lock_changes:
            raise SystemExit(
                f"automated candidates may not change dependency lockfiles: {rel}; "
                "new dependencies require a separately reviewed hydration image"
            )
        target = candidate / rel
        if target.is_symlink():
            raise SystemExit(f"candidate changed symlink: {rel}")
        if target.is_file():
            data = target.read_bytes()
            if len(data) <= 10 * 1024 * 1024:
                for pattern in SECRET_PATTERNS:
                    if pattern.search(data):
                        raise SystemExit(f"secret-like material detected in {rel}")
        match = ADR_RE.match(rel.name)
        if match and rel.parts[:2] == ("docs", "adr"):
            number = int(match.group(1))
            if number in base_adrs or number in seen_new_adrs:
                raise SystemExit(f"duplicate ADR number: {number}")
            seen_new_adrs.add(number)
        if rel.name == "Cargo.toml" and target.is_file():
            names = package_names(target.parent)
            duplicate = names & base_packages
            if duplicate and not (base / rel).exists():
                raise SystemExit(f"new crate duplicates package name(s): {sorted(duplicate)}")

    output = Path(args.output)
    output.write_text("\n".join(changed) + "\n", encoding="utf-8")
    print(f"preflight scanned {len(changed)} changed paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
