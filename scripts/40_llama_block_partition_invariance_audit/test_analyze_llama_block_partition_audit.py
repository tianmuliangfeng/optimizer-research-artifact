#!/usr/bin/env python3
"""Torch-free integration tests for the frozen audit-40 analyzer."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ANALYZER = HERE / "analyze_llama_block_partition_audit.py"
CONTRACT = HERE / "audit_contract.json"
SPEC = importlib.util.spec_from_file_location(
    "audit40_analyzer_tests", ANALYZER
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {ANALYZER}")
A = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = A
SPEC.loader.exec_module(A)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class AnalyzeLlamaBlockPartitionAuditTests(unittest.TestCase):
    def make_run(self, root: Path, global_drift: float) -> tuple[Path, Path]:
        run = root / "run"
        output = root / "analysis"
        output.mkdir(parents=True)
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        import hashlib

        contract_sha = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
        worker_sha = hashlib.sha256(
            (HERE / "llama_block_partition_worker.py").read_bytes()
        ).hexdigest()
        for label in ("early", "late"):
            directory = run / "formal" / label
            directory.mkdir(parents=True)
            write_json(
                directory / "llama_block_audit_manifest.json",
                {
                    "passed": True,
                    "analysis_tier": "formal",
                    "script_version": "2026-07-29.2",
                    "worker_sha256": worker_sha,
                    "contract_sha256": contract_sha,
                    "checkpoint_sha256": contract["checkpoints"][label]["sha256"],
                },
            )
            write_json(directory / "checks.json", {"synthetic_integrity": True})
            updates = []
            for index in range(24):
                updates.append(
                    {
                        "checkpoint_label": label,
                        "candidate": "block4",
                        "partition_kind": "global_balanced_partition",
                        "update_relative_drift": global_drift,
                    }
                )
            for candidate in ("none", "diag", "dense_full"):
                updates.append(
                    {
                        "checkpoint_label": label,
                        "candidate": candidate,
                        "partition_kind": "global_balanced_partition",
                        "update_relative_drift": 1e-5,
                    }
                )
            updates.append(
                {
                    "checkpoint_label": label,
                    "candidate": "block4",
                    "partition_kind": "within_block_control",
                    "update_relative_drift": 1e-5,
                }
            )
            write_csv(directory / "equivariance_updates.csv", updates)
            partitions = [
                {
                    "checkpoint_label": label,
                    "partition_kind": kind,
                    "off_block_energy_fraction": value,
                }
                for kind, value in (
                    ("identity", 0.4),
                    ("within_block_control", 0.4),
                    ("global_balanced_partition", 0.6),
                )
            ]
            write_csv(directory / "partition_geometry.csv", partitions)
            summaries = [
                {
                    "checkpoint_label": label,
                    "repeat": 0,
                    "direction": "A_to_B",
                    "build_split": "A",
                    "eval_split": "B",
                    "scope": "grouped",
                    "candidate": candidate,
                    "best_relative_loss_delta": value,
                }
                for candidate, value in (
                    ("block4_identity", -0.01),
                    ("block4_global_seed4001", -0.005),
                )
            ]
            write_csv(directory / "line_search_summary.csv", summaries)
        return run, output

    def run_analyzer(self, root: Path, global_drift: float) -> dict[str, Any]:
        run, output = self.make_run(root, global_drift)
        subprocess.run(
            [
                sys.executable,
                str(ANALYZER),
                "--run-dir",
                str(run),
                "--contract",
                str(CONTRACT),
                "--output-dir",
                str(output),
            ],
            check=True,
        )
        return json.loads(
            (output / "llama_block_audit_analysis_manifest.json").read_text(
                encoding="utf-8"
            )
        )

    def test_strong_non_invariance_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.run_analyzer(Path(temporary), 0.1)
        self.assertTrue(manifest["passed"])
        self.assertEqual(manifest["classification"], "strong_non_invariance")

    def test_approximately_invariant_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.run_analyzer(Path(temporary), 0.001)
        self.assertTrue(manifest["passed"])
        self.assertEqual(
            manifest["classification"],
            "approximately_invariant_at_tested_resolution",
        )

    def test_stage_disagreement_is_inconclusive(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        classification, _ = A.classify(
            median_drift=0.06,
            control_maximum=1e-5,
            thresholds=contract["classification_thresholds"],
            stage_medians=[0.1, 0.001],
        )
        self.assertEqual(classification, "inconclusive")


if __name__ == "__main__":
    unittest.main()
