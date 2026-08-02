import copy
import datetime as dt
import json
import sys
import tempfile
import unittest
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))

from research_gate import (  # noqa: E402
    GateError,
    canonical_json,
    embedding_space_id,
    evaluate,
    sha256_bytes,
    validate_artifact_index,
    validate_base_gate,
    validate_manifest,
    validate_override,
)
from base_health import failed_base_checks  # noqa: E402
from build_override import build_override  # noqa: E402

HEX = "a" * 64
COMMIT = "b" * 40
COMPONENTS = [
    "encoded_payload_bytes",
    "index_graph_bytes",
    "codebooks_quantizer_bytes",
    "container_bookkeeping_bytes",
    "allocator_overhead_bytes",
    "temporary_build_bytes",
    "temporary_query_bytes",
    "process_peak_rss_bytes",
]


def identity(prompt_hash=HEX, prefix_version=1):
    return {
        "schema_version": 1,
        "provider": "fixture",
        "model_id": "model@revision",
        "model_artifact_sha256": HEX,
        "model_graph_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
        "prompt_template_sha256": prompt_hash,
        "pooling_strategy": "mean",
        "normalize": True,
        "truncation_tokens": 512,
        "output_dimension": 3,
        "output_dtype": "f32",
        "runtime_revision": "fixture-v1",
        "distance_metric": "cosine",
        "role_policy": "asymmetric",
        "prefix_policy": "required",
        "prefix_policy_version": prefix_version,
    }


def manifest(identity_value=None, revision=1, phase="confirmation", seeds=None):
    identity_value = identity_value or identity()
    return {
        "schema_version": 1,
        "commit": COMMIT,
        "revision": revision,
        "phase": phase,
        "claim": "Treatment improves paired recall at an equal resident-memory budget.",
        "independent_variable": "index pruning policy",
        "decision_rule": {
            "primary_metric": "recall_at_10",
            "minimum_meaningful_effect": 0.01,
            "expected_direction": "greater",
            "alpha": 0.05,
            "comparison": "paired",
            "sample_size_or_power_rationale": "Five paired fixtures exercise the gate; production runs require a power analysis.",
            "outcome_rules": {
                "pass": "ci_lower_at_or_above_minimum_meaningful_effect",
                "fail": "ci_upper_at_or_below_zero",
                "inconclusive": "otherwise",
            },
        },
        "datasets": [{
            "name": "public-real-fixture",
            "kind": "real",
            "source": "https://example.invalid/pinned",
            "sha256": "d" * 64,
            "sampling": "first 100 after stable-id sort",
            "held_out": True,
        }],
        "embedding_space": {
            "identity": identity_value,
            "embedding_space_id": embedding_space_id(identity_value),
            "query_api": "embed_query",
            "passage_api": "embed_passage",
        },
        "exploration_seeds": [1, 2, 3, 4, 5],
        "confirmation_seeds": seeds or [101, 102, 103, 104, 105],
        "topology": {"claimed": "hnsw", "implemented": "hnsw", "configuration": {"m": 16}},
        "budget": {
            "primary_resource": "resident_memory_bytes",
            "tolerance": 0.005,
            "indivisible_allocation_unit": 4096,
            "secondary_resources": ["query_latency_ns"],
        },
        "memory_accounting": {
            "components": COMPONENTS,
            "peak_rss_protocol": {
                "version": "rss-v1",
                "isolation": "fresh cgroup process",
                "baseline_subtracted": True,
                "warmup_operations": 10,
                "sampling_interval_ms": 10,
                "allocator": "system",
                "cgroup_limit_bytes": 1073741824,
                "page_size_bytes": 4096,
                "process_count": 1,
                "thread_count": 1,
                "child_process_policy": "forbidden",
                "filesystem_cache_included": False,
                "cache_state": "warm",
                "kernel": "linux-fixture",
                "alternating_order": True,
            },
        },
        "environment": {"os": "linux"},
        "commands": ["fixture"],
        "evaluator_version": "research-gate-v1@sha256:" + HEX,
        "artifact_retention_class": "accepted",
        "selection": {
            "candidate_family": "fixture",
            "selection_rule": "single preregistered candidate",
            "confirmation_error_control": "single-candidate",
        },
    }


def results(manifest_value, treatment=0.73, control=0.70, budget_delta=1000):
    memory = {name: 1 for name in COMPONENTS}
    runs = []
    for seed in manifest_value["confirmation_seeds"]:
        runs.append({
            "seed": seed,
            "control": {
                "metrics": {"recall_at_10": control},
                "resources": {"resident_memory_bytes": 1_000_000},
                "memory": memory,
            },
            "treatment": {
                "metrics": {"recall_at_10": treatment},
                "resources": {"resident_memory_bytes": 1_000_000 + budget_delta},
                "memory": memory,
            },
            "selection": {
                "configured_count": 10,
                "selected_count": 10,
                "deterministic_tie_breaker": "stable vector id",
            },
        })
    return {
        "schema_version": 1,
        "manifest_revision": manifest_value["revision"],
        "manifest_sha256": sha256_bytes(canonical_json(manifest_value)),
        "candidate_commit": COMMIT,
        "embedding_space_id": manifest_value["embedding_space"]["embedding_space_id"],
        "phase": "confirmation",
        "runs": runs,
    }


class ResearchGateTests(unittest.TestCase):
    def test_nightly_dispatch_dedup_key_matches_candidate_run_name(self):
        repository = TEST_ROOT.parents[1]
        candidate_workflow = (repository / ".github/workflows/research-candidate.yml").read_text()
        dispatcher = (repository / ".github/workflows/research-nightly-dispatch.yml").read_text()
        expected = "research-candidate ${{ inputs.candidate_ref }}@${{ inputs.candidate_sha }}"
        self.assertIn(f"run-name: {expected}", candidate_workflow)
        self.assertIn('title="research-candidate $candidate_ref@$candidate_sha"', dispatcher)

    def test_passes_paired_equal_budget_confirmation(self):
        value = manifest()
        summary = evaluate(value, results(value))
        self.assertEqual(summary["outcome"], "pass")
        self.assertAlmostEqual(summary["paired_effect_mean"], 0.03)

    def test_rejects_unequal_primary_budget(self):
        value = manifest()
        with self.assertRaisesRegex(GateError, "exceeds primary-resource budget"):
            evaluate(value, results(value, budget_delta=10_000))

    def test_rejects_percentile_tie_overallocation(self):
        value = manifest()
        raw = results(value)
        raw["runs"][0]["selection"]["selected_count"] = 11
        with self.assertRaisesRegex(GateError, "selection count changed"):
            evaluate(value, raw)

    def test_exploration_cannot_promote(self):
        value = manifest(phase="exploration")
        with self.assertRaisesRegex(GateError, "only confirmation"):
            evaluate(value, results(value))

    def test_query_template_change_requires_new_revision_and_fresh_confirmation(self):
        old = manifest()
        changed_identity = identity(prompt_hash="e" * 64, prefix_version=2)
        unchanged_revision = manifest(changed_identity, revision=1)
        with self.assertRaisesRegex(GateError, "new experimental revision"):
            validate_manifest(unchanged_revision, old)

        reused_seeds = manifest(changed_identity, revision=2)
        with self.assertRaisesRegex(GateError, "fresh confirmation seeds"):
            validate_manifest(reused_seeds, old)

        changed = manifest(changed_identity, revision=2, seeds=[201, 202, 203, 204, 205])
        validate_manifest(changed, old)
        self.assertNotEqual(
            old["embedding_space"]["embedding_space_id"],
            changed["embedding_space"]["embedding_space_id"],
        )

    def test_nfc_is_required_for_cross_runtime_identity(self):
        value = identity()
        value["provider"] = "e\u0301"
        with self.assertRaisesRegex(GateError, "NFC"):
            embedding_space_id(value)

    def test_override_is_exact_sha_and_expires_within_72_hours(self):
        now = dt.datetime(2026, 7, 29, tzinfo=dt.timezone.utc)
        value = {
            "schema_version": 1,
            "scope": "red-base-only",
            "approver": "maintainer",
            "approval_source": "github-environment:research-gate-override",
            "approver_permission": "maintain",
            "approver_in_research_gate_codeowners": True,
            "base_sha": "1" * 40,
            "head_sha": "2" * 40,
            "failed_checks": ["base/check:ci:failure"],
            "rationale": "Differential run proves base-only failure.",
            "approved_at": now.isoformat(),
            "expires_at": (now + dt.timedelta(hours=71)).isoformat(),
        }
        validate_override(value, "1" * 40, "2" * 40, ["base/check:ci:failure"], now)
        stale = copy.deepcopy(value)
        stale["head_sha"] = "3" * 40
        with self.assertRaisesRegex(GateError, "exact base/head"):
            validate_override(stale, "1" * 40, "2" * 40, ["base/check:ci:failure"], now)

    def test_override_cannot_bypass_methodology_or_confirmation(self):
        now = dt.datetime(2026, 7, 29, tzinfo=dt.timezone.utc)
        value = {
            "schema_version": 1,
            "scope": "red-base-only",
            "approver": "maintainer",
            "approval_source": "github-environment:research-gate-override",
            "approver_permission": "admin",
            "approver_in_research_gate_codeowners": True,
            "base_sha": "1" * 40,
            "head_sha": "2" * 40,
            "failed_checks": ["methodology"],
            "rationale": "must be rejected",
            "approved_at": now.isoformat(),
            "expires_at": (now + dt.timedelta(hours=1)).isoformat(),
        }
        with self.assertRaisesRegex(GateError, "red-base checks only"):
            validate_override(value, "1" * 40, "2" * 40, ["methodology"], now)

    def test_base_health_and_authorized_red_gate_are_deterministic(self):
        failed = failed_base_checks(
            {"check_runs": [
                {"id": 1, "name": "ci", "status": "completed", "conclusion": "failure", "completed_at": "2026-01-01"},
                {"id": 2, "name": "docs", "status": "completed", "conclusion": "success", "completed_at": "2026-01-01"},
            ]},
            {"statuses": [{"context": "legacy", "state": "error"}]},
        )
        self.assertEqual(failed, ["base/check:ci:failure", "base/status:legacy:error"])
        gate = {
            "schema_version": 1,
            "base_sha": "1" * 40,
            "head_sha": "2" * 40,
            "state": "authorized-red",
            "failed_checks": failed,
            "override_sha256": "a" * 64,
        }
        validate_base_gate(gate, "1" * 40, "2" * 40)

    def test_override_records_actual_environment_reviewer(self):
        now = dt.datetime(2026, 7, 29, tzinfo=dt.timezone.utc)
        approvals = [{
            "state": "approved",
            "environments": [{"name": "research-gate-override"}],
            "user": {"login": "reviewer"},
        }]
        value = build_override(
            approvals,
            "maintain",
            "someone-else,@reviewer",
            "1" * 40,
            "2" * 40,
            ["base/check:ci:failure"],
            "Known-red main is unchanged in the differential run.",
            (now + dt.timedelta(hours=24)).isoformat(),
            now,
        )
        self.assertEqual(value["approver"], "reviewer")
        with self.assertRaisesRegex(GateError, "not in RESEARCH_GATE_CODEOWNERS"):
            build_override(
                approvals,
                "maintain",
                "someone-else",
                "1" * 40,
                "2" * 40,
                ["base/check:ci:failure"],
                "Known-red main.",
                (now + dt.timedelta(hours=24)).isoformat(),
                now,
            )

    def test_artifact_index_detects_tampering_and_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "raw.json"
            artifact.write_text("{}\n", encoding="utf-8")
            index = {
                "schema_version": 1,
                "retention_class": "candidate",
                "retention_days": 365,
                "artifacts": [{
                    "path": "raw.json",
                    "sha256": __import__("hashlib").sha256(b"{}\n").hexdigest(),
                    "size_bytes": 3,
                }],
            }
            validate_artifact_index(index, root)
            artifact.write_text('{"tampered":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(GateError, "size changed|hash changed"):
                validate_artifact_index(index, root)


if __name__ == "__main__":
    unittest.main()
