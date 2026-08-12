#!/usr/bin/env python3

import importlib.util
import json
import statistics
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_submission_evidence.py")
SPEC = importlib.util.spec_from_file_location("submission_audit", MODULE_PATH)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
VALIDATOR_PATH = Path(__file__).with_name("validate_submission_evidence.py")
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "submission_validator", VALIDATOR_PATH
)
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
V = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(V)


class SubmissionEvidenceTests(unittest.TestCase):
    def test_method_roles(self):
        self.assertEqual(M.role("block4"), "original_newton_muon")
        self.assertEqual(M.role("newton_full"), "original_newton_muon")
        self.assertEqual(M.role("none"), "selective_none")
        self.assertEqual(M.role("down_diag"), "selective_diag")

    def test_metric_gap_contract(self):
        rows = M.build_metric_eligibility(None)
        missing = {row["metric"] for row in rows if row["classification"] == "missing"}
        self.assertEqual(
            missing,
            {"tokens_per_s", "steps_per_s", "four_method_lr_sensitivity"},
        )
        followup = M.followup_contract(rows)
        self.assertEqual(
            {item["id"] for item in followup["experiments"]},
            {"EFF-ISO-R1", "SENS-R1-4WAY"},
        )

    def test_write_csv_rejects_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                M.write_csv(Path(directory) / "empty.csv", [])

    def test_primary_comparison_is_not_diag_none(self):
        rows = M.build_metric_eligibility(None)
        text = " ".join(str(row) for row in rows)
        self.assertNotIn("diag_vs_none", text)

    def test_required_source_can_move_within_registered_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            moved = root / "20_llama_swiglu_1b/capacity_fine"
            moved.mkdir(parents=True)
            source = moved / "capacity_fine_manifest.json"
            source.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
            registry = {
                "required_files": [
                    {
                        "id": "capacity_fine_source_manifest",
                        "path": "20_llama_swiglu_1b/analysis/old/capacity_fine_manifest.json",
                        "fallback_globs": [
                            "20_llama_swiglu_1b/**/capacity_fine_manifest.json"
                        ],
                        "kind": "json",
                    }
                ]
            }
            rows, paths = M.audit_required_sources(root, registry)
            self.assertEqual(paths["capacity_fine_source_manifest"], source.resolve())
            self.assertEqual(
                rows[0]["relative_path"],
                "20_llama_swiglu_1b/capacity_fine/capacity_fine_manifest.json",
            )

    def test_relocated_source_must_be_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                target = root / f"20_llama_swiglu_1b/{name}"
                target.mkdir(parents=True)
                (target / "capacity_fine_manifest.json").write_text(
                    json.dumps({"status": "complete"}), encoding="utf-8"
                )
            registry = {
                "required_files": [
                    {
                        "id": "capacity_fine_source_manifest",
                        "path": "20_llama_swiglu_1b/analysis/old/capacity_fine_manifest.json",
                        "fallback_globs": [
                            "20_llama_swiglu_1b/**/capacity_fine_manifest.json"
                        ],
                        "kind": "json",
                    }
                ]
            }
            with self.assertRaisesRegex(RuntimeError, "ambiguous relocated source"):
                M.audit_required_sources(root, registry)

    def test_portable_snapshot_is_used_when_experiment_source_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_root = base / "experiments"
            portable_root = base / "snapshot"
            input_root.mkdir()
            portable_root.mkdir()
            source = portable_root / "historical_r1_manifest.json"
            source.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
            registry = {
                "required_files": [
                    {
                        "id": "historical_r1_manifest",
                        "path": "15_official_newton_muon_r1/analysis/old/analysis_manifest.json",
                        "portable_path": "historical_r1_manifest.json",
                        "kind": "json",
                    }
                ]
            }
            rows, paths = M.audit_required_sources(
                input_root, registry, portable_root
            )
            self.assertEqual(paths["historical_r1_manifest"], source.resolve())
            self.assertEqual(
                rows[0]["relative_path"],
                "portable_snapshot/historical_r1_manifest.json",
            )

    def test_required_portable_snapshot_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_root = base / "experiments"
            portable_root = base / "snapshot"
            input_root.mkdir()
            portable_root.mkdir()
            (portable_root / "source.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            registry = {
                "portable_snapshot_required": True,
                "required_files": [
                    {
                        "id": "source",
                        "path": "missing/source.json",
                        "portable_path": "source.json",
                        "portable_sha256": "0" * 64,
                        "kind": "json",
                    }
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                M.audit_required_sources(
                    input_root, registry, portable_root
                )

    def test_output_dir_is_optional_only_for_preflight(self):
        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "analyze_submission_evidence.py",
                "--input-root",
                ".",
                "--registry",
                "registry.json",
                "--preflight-only",
            ]
            args = M.parse_args()
            self.assertTrue(args.preflight_only)
            self.assertIsNone(args.output_dir)
        finally:
            sys.argv = original_argv

    def test_final_lr_validation_accepts_gap_and_completed_states(self):
        row = {
            "sensitivity_type": "learning_rate",
            "architecture": "final R1",
            "method_roles": ",".join(sorted(V.ROLES)),
            "grid": "0.8x,1.0x,1.2x",
            "classification": "missing",
        }
        V.validate_final_lr_coverage(
            [row], {"four_method_lr_sensitivity"}
        )
        completed = {**row, "classification": "supporting_only"}
        V.validate_final_lr_coverage([completed], set())

    def test_final_lr_validation_rejects_stale_gap_after_completion(self):
        row = {
            "sensitivity_type": "learning_rate",
            "architecture": "final R1",
            "method_roles": ",".join(sorted(V.ROLES)),
            "grid": "0.8x,1.0x,1.2x",
            "classification": "missing",
        }
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            V.validate_final_lr_coverage([row], set())

    def test_performance_bundle_requires_balanced_recomputed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            batch = run_root / "benchmark"
            batch.mkdir()
            manifest_path = batch / "perf_manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            methods = ["muon", "block4", "none", "diag"]
            source_hashes = {method: f"source-{method}" for method in methods}
            runs = []
            for repeat in range(4):
                order = methods[repeat:] + methods[:repeat]
                for position, method in enumerate(order, start=1):
                    log = (
                        batch
                        / f"repeat{repeat + 1:02d}_{position:02d}_{method}"
                        / "terminal.log"
                    )
                    log.parent.mkdir()
                    log.write_text(
                        f"{repeat}:{position}:{method}\n", encoding="utf-8"
                    )
                    method_index = methods.index(method)
                    elapsed = 10.0 + method_index + repeat / 10
                    runs.append(
                        {
                            "method": method,
                            "repeat": str(repeat + 1),
                            "position": str(position),
                            "seed": "2026",
                            "init_sha256": "same-init",
                            "final_step": "544",
                            "official_train_time_s": str(elapsed),
                            "official_step_avg_ms": str(elapsed * 1000 / 512),
                            "wrapper_wall_elapsed_s": str(elapsed + 1),
                            "source_sha256": source_hashes[method],
                            "log_sha256": M.sha256_file(log),
                        }
                    )
            summary = []
            for method in methods:
                selected = [row for row in runs if row["method"] == method]
                times = [float(row["official_train_time_s"]) for row in selected]
                steps = [float(row["official_step_avg_ms"]) for row in selected]
                summary.append(
                    {
                        "method": method,
                        "runs": "4",
                        "median_train_time_s": str(statistics.median(times)),
                        "median_step_ms": str(statistics.median(steps)),
                        "median_tokens_per_s": str(
                            512 * 512 * 1024 / statistics.median(times)
                        ),
                    }
                )
            manifest = {
                "status": "complete",
                "runtime": {"gpu_name": "NVIDIA H100 80GB HBM3"},
                "methods": methods,
                "source_sha256": source_hashes,
                "timed_steps": 512,
                "repeats": 4,
            }
            now = datetime.now(timezone.utc)
            certificate = {
                "passed": True,
                "created_at": (now - timedelta(seconds=5)).isoformat(),
                "active_compute_processes": [],
                "gpus": [{"index": 0}, {"index": 1}],
            }
            postflight = {
                **certificate,
                "created_at": (now + timedelta(seconds=5)).isoformat(),
            }
            provenance = {
                "passed": True,
                "provenance": {
                    "official_commit": "pinned",
                    "tracked_worktree_clean": True,
                    "canonical_text_sha256": {"train.py": "hash"},
                },
            }
            registry = {
                "performance_required_roles": methods,
                "performance_min_timed_steps": 512,
                "performance_min_repeats": 4,
                "performance_required_gpu_count": 2,
                "official_commit": "pinned",
            }
            self.assertEqual(
                M.performance_bundle_errors(
                    manifest_path=manifest_path,
                    manifest=manifest,
                    summary=summary,
                    runs=runs,
                    preflight=certificate,
                    postflight=postflight,
                    provenance=provenance,
                    registry=registry,
                ),
                [],
            )
            summary[0]["median_tokens_per_s"] = "1"
            errors = M.performance_bundle_errors(
                manifest_path=manifest_path,
                manifest=manifest,
                summary=summary,
                runs=runs,
                preflight=certificate,
                postflight=postflight,
                provenance=provenance,
                registry=registry,
            )
            self.assertIn("muon summary recomputation", errors)


if __name__ == "__main__":
    unittest.main()
