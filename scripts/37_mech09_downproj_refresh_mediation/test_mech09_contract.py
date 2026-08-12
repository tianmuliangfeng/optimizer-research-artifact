#!/usr/bin/env python3
"""CPU-only frozen-contract tests for MECH-09."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARTIFACT_ROOT = HERE.parents[1]
RESULTS_ROOT = Path(
    os.environ.get("SNM_RESULTS_ROOT", ARTIFACT_ROOT / "runs")
).expanduser().resolve()


def load_analysis():
    path = HERE / "analyze_mech09.py"
    spec = importlib.util.spec_from_file_location("mech09_analysis_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load_analysis()


class Mech09ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract_path = HERE / "refresh_mediation_contract.json"
        cls.reference_path = HERE / "mech08_control_reference.json"
        cls.contract = json.loads(
            cls.contract_path.read_text(encoding="utf-8")
        )
        cls.reference = json.loads(
            cls.reference_path.read_text(encoding="utf-8")
        )

    def test_formal_matrix_is_exactly_24_new_jobs(self) -> None:
        formal = self.contract["formal"]
        jobs = (
            len(formal["origins"])
            * len(formal["interventions"])
            * len(formal["data_replicas"])
        )
        self.assertEqual(jobs, 24)
        self.assertEqual(
            jobs, self.contract["stopping_rule"]["maximum_new_formal_jobs"]
        )
        self.assertEqual(
            jobs,
            self.contract["analysis_contract"]["minimum_complete_formal_jobs"],
        )

    def test_only_two_frozen_interventions_are_new(self) -> None:
        self.assertEqual(
            set(self.contract["interventions"]),
            {"delayed_down_refresh", "frozen_down_refresh"},
        )
        self.assertEqual(
            set(self.contract["formal"]["interventions"]),
            set(self.contract["interventions"]),
        )
        self.assertTrue(
            all(
                row["source_method"] == "newton_full"
                for row in self.contract["interventions"].values()
            )
        )

    def test_refresh_schedules_are_surgical(self) -> None:
        formal = self.contract["formal"]
        production = formal["expected_other_group_refresh_completed_steps"]
        self.assertEqual(production, [32, 64, 96, 128])
        self.assertEqual(
            self.contract["interventions"]["delayed_down_refresh"][
                "formal_down_refresh_completed_steps"
            ],
            [64, 96, 128],
        )
        self.assertEqual(
            self.contract["interventions"]["frozen_down_refresh"][
                "formal_down_refresh_completed_steps"
            ],
            [],
        )
        for row in self.contract["interventions"].values():
            self.assertEqual(row["target_group_suffix"], ".down_input")
            self.assertEqual(
                row["all_non_target_groups"],
                "retain the tier production refresh schedule",
            )

    def test_smoke_exercises_hold_and_refresh_paths(self) -> None:
        smoke = self.contract["smoke"]
        self.assertEqual(smoke["rollout_steps"], 4)
        self.assertEqual(
            smoke["expected_other_group_refresh_completed_steps"], [2, 4]
        )
        self.assertEqual(
            self.contract["interventions"]["delayed_down_refresh"][
                "smoke_down_refresh_completed_steps"
            ],
            [4],
        )
        self.assertEqual(
            self.contract["interventions"]["frozen_down_refresh"][
                "smoke_down_refresh_completed_steps"
            ],
            [],
        )

    def test_primary_comparisons_answer_refresh_causality(self) -> None:
        observed = {
            (row["left"], row["right"])
            for row in self.contract["comparison_contract"]["primary"]
        }
        self.assertEqual(
            observed,
            {
                ("delayed_down_refresh", "original_newton_muon"),
                ("frozen_down_refresh", "original_newton_muon"),
                ("delayed_down_refresh", "frozen_down_refresh"),
            },
        )
        self.assertNotIn(
            ("selective_diag", "selective_none"), observed
        )

    def test_control_reference_is_hash_frozen(self) -> None:
        observed = hashlib.sha256(self.reference_path.read_bytes()).hexdigest()
        expected = self.contract["mech08_control_reference"].get(
            "public_sha256", self.contract["mech08_control_reference"]["sha256"]
        )
        self.assertEqual(observed, expected)
        self.assertTrue(self.reference["passed"])
        self.assertEqual(self.reference["file_count"], 162)
        self.assertEqual(len(self.reference["files"]), 162)
        self.assertEqual(
            self.reference["source_run_id"],
            self.contract["mech08_control_reference"]["source_run_id"],
        )

    def test_reference_files_rebuild_when_mech08_is_local(self) -> None:
        run = (
            RESULTS_ROOT
            / "36_mech08_short_horizon_rollout"
            / self.reference["source_run_id"]
        )
        if not run.is_dir():
            self.skipTest("MECH-08 source run is not available locally")
        audit = A.verify_control_reference(self.reference, run)
        self.assertTrue(audit["passed"])

    def test_replica_training_windows_do_not_overlap(self) -> None:
        formal = self.contract["formal"]
        horizon = int(formal["rollout_steps"])
        intervals = [
            (int(offset), int(offset) + horizon)
            for offset in formal["replica_optimizer_step_offsets"]
        ]
        for left, first in enumerate(intervals):
            for second in intervals[left + 1 :]:
                self.assertGreaterEqual(
                    max(first[0], second[0]), min(first[1], second[1])
                )

    def test_stopping_and_scope_rules_are_frozen(self) -> None:
        stopping = self.contract["stopping_rule"]
        self.assertTrue(stopping["no_additional_interventions"])
        self.assertTrue(stopping["no_hyperparameter_tuning"])
        self.assertTrue(stopping["no_origin_or_replica_selection"])
        self.assertTrue(
            stopping["do_not_rerun_for_unfavorable_scientific_outcome"]
        )
        self.assertTrue(
            self.contract["scope_boundary"]["efficiency_benchmark_excluded"]
        )

    def test_auc_uses_step_widths(self) -> None:
        rows = [
            {"optimizer_step": 0, "normalized_loss": 1.0},
            {"optimizer_step": 1, "normalized_loss": 0.8},
            {"optimizer_step": 3, "normalized_loss": 0.6},
        ]
        self.assertAlmostEqual(A.trapezoid_auc(rows), 2.3 / 3.0)


if __name__ == "__main__":
    unittest.main()
