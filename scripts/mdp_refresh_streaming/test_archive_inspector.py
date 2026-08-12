#!/usr/bin/env python3
"""Dependency-free regression tests for the archived-result inspector."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import inspect_stream_replay as INSPECTOR


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ArchiveInspectorTests(unittest.TestCase):
    def make_run(self, root: Path, *, passed: bool = False) -> Path:
        run = root / "fixture_run"
        analysis = run / "analysis"
        analysis.mkdir(parents=True)
        contract_path = (
            run
            / "source_snapshot_v4"
            / "scripts"
            / "mdp_refresh_streaming"
            / "refresh_stream_contract.json"
        )
        write_json(
            contract_path,
            {
                "hard_gates": {
                    "runtime_resolvent_relative_residual_max": 0.01,
                }
            },
        )

        metrics = analysis / "refresh_layer_event_metrics.csv"
        with metrics.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "origin",
                    "data_replica",
                    "event_id",
                    "layer_index",
                    "runtime_resolvent_relative_residual",
                ),
            )
            writer.writeheader()
            for index in range(432):
                writer.writerow(
                    {
                        "origin": "late_muon" if index < 6 else "early_muon",
                        "data_replica": index % 3,
                        "event_id": "production_refresh_32",
                        "layer_index": 3 if index < 6 else index % 18,
                        "runtime_resolvent_relative_residual": (
                            0.02 if index < 6 and not passed else 0.001
                        ),
                    }
                )

        artifacts = {
            "refresh_layer_event_metrics.csv": sha256(metrics),
        }
        numeric_checks = {
            "covariance_refresh_identity": True,
            "inverse_asymmetry": True,
            "k_asymmetry": True,
            "runtime_inverse_backward_residual": True,
            "runtime_resolvent_relative_residual": passed,
        }
        write_json(
            analysis / "formal_stream_manifest.json",
            {
                "artifacts": artifacts,
                "checks": {
                    "accepted_source_hashes": True,
                    "all_units_passed": True,
                    "boolean_integrity_gates": True,
                    "layer_event_rows": True,
                    "no_large_persisted_files": True,
                    "numeric_integrity_gates": passed,
                    "unit_contract_lineage": True,
                    "unit_count": True,
                    "unit_event_rows": True,
                    "unit_outcome_rows": True,
                    "validation_slice_gate": True,
                },
                "numeric_gate_checks": numeric_checks,
                "passed": passed,
                "rows": dict(INSPECTOR.EXPECTED_ROWS),
                "stream_contract_sha256": sha256(contract_path),
                "unit_contract_lineage": [
                    {"origin": f"unit_{index}", "passed": True}
                    for index in range(12)
                ],
                "unit_contract_lineage_counts": {
                    "current_v4": 12,
                    "inherited_stricter_v3": 0,
                },
            },
        )
        write_json(
            run / "status.json",
            {"status": "passed" if passed else "integrity_failed"},
        )
        return run

    def test_expected_numeric_gate_failure_is_clean_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = INSPECTOR.inspect(self.make_run(Path(temporary)))
        self.assertTrue(payload["archive_integrity_passed"])
        self.assertEqual(payload["formal_adjudication"], "numeric_gate_failed")
        self.assertEqual(payload["resolvent_failure"]["bad_row_count"], 6)
        self.assertEqual(payload["resolvent_failure"]["layers"], [3])

    def test_passing_fixture_is_classified_as_passed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = INSPECTOR.inspect(self.make_run(Path(temporary), passed=True))
        self.assertTrue(payload["archive_integrity_passed"])
        self.assertEqual(payload["formal_adjudication"], "passed")
        self.assertEqual(payload["resolvent_failure"]["bad_row_count"], 0)

    def test_artifact_corruption_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = self.make_run(Path(temporary))
            metrics = run / "analysis" / "refresh_layer_event_metrics.csv"
            with metrics.open("a", encoding="utf-8") as handle:
                handle.write("corrupt\n")
            payload = INSPECTOR.inspect(run)
        self.assertFalse(payload["archive_integrity_passed"])
        self.assertEqual(payload["formal_adjudication"], "other_failure")


if __name__ == "__main__":
    unittest.main()
