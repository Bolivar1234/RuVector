#!/usr/bin/env python3
"""Reduce GitHub check/status responses to a deterministic red-base check set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PASSING_CONCLUSIONS = {"success", "neutral", "skipped"}


def failed_base_checks(check_runs: dict, statuses: dict) -> list[str]:
    latest = {}
    for check in check_runs.get("check_runs", []):
        name = str(check.get("name", "")).strip()
        if not name:
            continue
        key = (check.get("completed_at") or check.get("started_at") or "", int(check.get("id", 0)))
        if name not in latest or key > latest[name][0]:
            latest[name] = (key, check)
    failed = []
    for name, (_, check) in latest.items():
        status = check.get("status")
        conclusion = check.get("conclusion")
        if status != "completed" or conclusion not in PASSING_CONCLUSIONS:
            failed.append(f"base/check:{name}:{conclusion or status or 'unknown'}")

    latest_status = {}
    for status in statuses.get("statuses", []):
        context = str(status.get("context", "")).strip()
        if context and context not in latest_status:
            # The combined-status endpoint returns statuses newest first.
            latest_status[context] = status
    for context, status in latest_status.items():
        state = status.get("state")
        if state != "success":
            failed.append(f"base/status:{context}:{state or 'unknown'}")
    return sorted(set(failed))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-runs", required=True)
    parser.add_argument("--statuses", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checks = json.loads(Path(args.check_runs).read_text(encoding="utf-8"))
    statuses = json.loads(Path(args.statuses).read_text(encoding="utf-8"))
    Path(args.output).write_text(
        json.dumps(failed_base_checks(checks, statuses), indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
