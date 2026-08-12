#!/usr/bin/env python3
"""CPU-only analysis tests for MECH-09R."""

from __future__ import annotations

import importlib.util
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def load_analysis():
    path = HERE / "analyze_mech09r.py"
    spec = importlib.util.spec_from_file_location(
        "analyze_mech09r_tested", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load_analysis()


class Mech09RAnalysisTests(unittest.TestCase):
    def test_exact_step_audit_requires_bitwise_equal_values(self) -> None:
        arms = (
            "production_newton_muon",
            "delayed_down_refresh",
            "frozen_down_refresh",
        )
        by_arm = {}
        for arm in arms:
            by_arm[arm] = {
                (f"cell_{cell}", replica, 16): {"heldout_loss": "3.0"}
                for cell in range(4)
                for replica in range(3)
            }
        audit = A.exact_step_audit(by_arm, arms, 16)
        self.assertTrue(audit["passed"])
        by_arm["frozen_down_refresh"][("cell_0", 0, 16)][
            "heldout_loss"
        ] = "3.0000001"
        self.assertFalse(A.exact_step_audit(by_arm, arms, 16)["passed"])

    def test_contrast_summary_counts_units_and_origins(self) -> None:
        rows = []
        for cell in range(4):
            for replica in range(3):
                rows.append(
                    {
                        "checkpoint_cell": f"cell_{cell}",
                        "data_replica": replica,
                        "optimizer_step": 48,
                        "contrast": "x",
                        "normalized_loss_delta": -0.01,
                    }
                )
        summary = A.contrast_summary(rows)[0]
        self.assertEqual(summary["paired_units"], 12)
        self.assertEqual(summary["origins"], 4)
        self.assertEqual(summary["left_better_units"], 12)
        self.assertEqual(summary["left_better_origins"], 4)

    def test_auc_is_normalized_by_horizon(self) -> None:
        rows = [
            {"optimizer_step": 0, "normalized_loss": 1.0},
            {"optimizer_step": 64, "normalized_loss": 0.8},
            {"optimizer_step": 128, "normalized_loss": 0.6},
        ]
        self.assertAlmostEqual(A.trapezoid_auc(rows), 0.8)

    def test_end_to_end_synthetic_full_support(self) -> None:
        contract_path = HERE / "refresh_mediation_repair_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract_sha = A.sha256_file(contract_path)
        steps = contract["formal"]["evaluation_steps"]
        arms = list(contract["arms"])
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for cell in contract["formal"]["origins"]:
                for replica in contract["formal"]["data_replicas"]:
                    directory = run / "formal" / cell / f"replica_{replica}"
                    directory.mkdir(parents=True)
                    manifest = {
                        "passed": True,
                        "script_version": A.WORKER_VERSION,
                        "analysis_tier": "formal",
                        "checkpoint_cell": cell,
                        "data_replica": replica,
                        "contract_sha256": contract_sha,
                        "causal_tree": True,
                        "timing_usable_for_paper": False,
                        "legacy_invalid_run_reused": False,
                    }
                    for name, payload in (
                        ("mech09r_manifest.json", manifest),
                        ("status.json", {"status": "passed"}),
                        ("checks.json", {"all": True}),
                        ("branch_audit.json", {"passed": True}),
                        ("refresh_tree_audit.json", {"passed": True}),
                    ):
                        (directory / name).write_text(
                            json.dumps(payload), encoding="utf-8"
                        )
                    evaluation = []
                    for arm in arms:
                        for step in steps:
                            loss = 1.0 - 0.001 * int(step)
                            if int(step) >= 48:
                                if arm in {
                                    "delayed_down_refresh",
                                    "frozen_down_refresh",
                                }:
                                    loss -= 0.01
                            if int(step) >= 80 and arm == "delayed_down_refresh":
                                loss += 0.005
                            evaluation.append(
                                {
                                    "checkpoint_cell": cell,
                                    "data_replica": replica,
                                    "optimizer_step": step,
                                    "arm": arm,
                                    "heldout_loss": loss,
                                    "normalized_loss": loss,
                                }
                            )
                    with (directory / "evaluation.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as handle:
                        writer = csv.DictWriter(
                            handle, fieldnames=list(evaluation[0])
                        )
                        writer.writeheader()
                        writer.writerows(evaluation)
                    training = [
                        {
                            "arm": arm,
                            "optimizer_step": step,
                            "train_loss_mean": 1.0,
                            "timing_usable_for_paper": False,
                        }
                        for arm in arms
                        for step in range(
                            1, int(contract["formal"]["rollout_steps"]) + 1
                        )
                    ]
                    with (directory / "training.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as handle:
                        writer = csv.DictWriter(
                            handle, fieldnames=list(training[0])
                        )
                        writer.writeheader()
                        writer.writerows(training)
            with mock.patch.object(
                sys,
                "argv",
                [
                    "analyze_mech09r.py",
                    "--run-dir",
                    str(run),
                    "--contract",
                    str(contract_path),
                ],
            ):
                A.main()
            decision = json.loads(
                (run / "analysis" / "mediation_decision.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads(
                (
                    run
                    / "analysis"
                    / "mech09r_analysis_manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["passed"])
            self.assertEqual(decision["classification"], "full_support")
            self.assertEqual(decision["directional_predictions_passed"], 3)


if __name__ == "__main__":
    unittest.main()
