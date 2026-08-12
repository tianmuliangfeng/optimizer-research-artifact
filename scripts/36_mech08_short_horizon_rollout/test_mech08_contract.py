#!/usr/bin/env python3
"""CPU-only contract tests for MECH-08."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import statistics
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARTIFACT_ROOT = HERE.parents[1]
RESULTS_ROOT = Path(
    os.environ.get("SNM_RESULTS_ROOT", ARTIFACT_ROOT / "runs")
).expanduser().resolve()


def load_analysis():
    path = HERE / "analyze_mech08.py"
    spec = importlib.util.spec_from_file_location("mech08_analysis_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load_analysis()


class Mech08ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(
            (HERE / "rollout_contract.json").read_text(encoding="utf-8")
        )
        with (HERE / "mech07_prediction_reference.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            cls.predictions = list(csv.DictReader(handle))

    def test_formal_matrix_is_48_jobs(self) -> None:
        formal = self.contract["formal"]
        jobs = (
            len(formal["origins"])
            * len(formal["algorithms"])
            * len(formal["data_replicas"])
        )
        self.assertEqual(jobs, 48)
        self.assertEqual(
            self.contract["analysis_contract"]["minimum_complete_formal_jobs"],
            48,
        )

    def test_primary_comparisons_match_scientific_priority(self) -> None:
        observed = {
            (row["left"], row["right"])
            for row in self.contract["comparison_contract"]["primary"]
        }
        self.assertEqual(
            observed,
            {
                ("selective_diag", "muon"),
                ("selective_none", "muon"),
                ("selective_diag", "original_newton_muon"),
                ("selective_none", "original_newton_muon"),
            },
        )
        self.assertNotIn(
            ("selective_diag", "selective_none"), observed
        )

    def test_algorithm_mapping_is_exact(self) -> None:
        observed = {
            name: row["source_method"]
            for name, row in self.contract["algorithms"].items()
        }
        self.assertEqual(
            observed,
            {
                "muon": "muon",
                "original_newton_muon": "newton_full",
                "selective_diag": "down_diag",
                "selective_none": "down_none",
            },
        )

    def test_replica_training_windows_do_not_overlap(self) -> None:
        formal = self.contract["formal"]
        horizon = int(formal["rollout_steps"])
        offsets = formal["replica_optimizer_step_offsets"]
        intervals = [(value, value + horizon) for value in offsets]
        for left, first in enumerate(intervals):
            for second in intervals[left + 1 :]:
                self.assertGreaterEqual(
                    max(first[0], second[0]), min(first[1], second[1])
                )

    def test_validation_build_and_eval_windows_do_not_overlap(self) -> None:
        for tier in ("smoke", "formal"):
            config = self.contract[tier]
            build_width = (
                config["build_device_batch_size"]
                * config["build_sequence_length"]
                * config["build_batches"]
            )
            eval_width = (
                config["eval_device_batch_size"]
                * config["eval_sequence_length"]
                * config["eval_batches"]
            )
            intervals = [
                (offset, offset + build_width)
                for offset in config["build_token_offsets"]
            ] + [
                (offset, offset + eval_width)
                for offset in config["eval_token_offsets"]
            ]
            for left, first in enumerate(intervals):
                for second in intervals[left + 1 :]:
                    self.assertGreaterEqual(
                        max(first[0], second[0]),
                        min(first[1], second[1]),
                    )

    def test_prediction_reference_is_complete(self) -> None:
        expected = {
            (cell, algorithm)
            for cell in self.contract["formal"]["origins"]
            for algorithm in self.contract["formal"]["algorithms"]
        }
        observed = {
            (row["checkpoint_cell"], row["algorithm"])
            for row in self.predictions
        }
        self.assertEqual(observed, expected)
        self.assertEqual(len(self.predictions), 16)
        self.assertTrue(
            all(float(row["prediction_step_multiplier"]) == 1.0
                for row in self.predictions)
        )
        self.assertEqual(
            self.contract["prediction_reference"]["primary_prediction_column"],
            "predicted_median_relative_loss_delta_at_multiplier_1",
        )

    def test_prediction_reference_rebuilds_from_mech07_when_available(self) -> None:
        source_root = (
            RESULTS_ROOT
            / "35_mech07_llama1b_family_contrast"
            / "20260727T083446+0000"
        )
        if not source_root.is_dir():
            self.skipTest("MECH-07 source artifacts are not present locally")
        reference = {
            (row["checkpoint_cell"], row["algorithm"]): row
            for row in self.predictions
        }
        expected_hashes = self.contract["prediction_reference"][
            "source_shadow_losses_sha256"
        ]
        for cell in self.contract["formal"]["origins"]:
            path = source_root / cell / "formal" / "shadow_losses.csv"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, expected_hashes[cell])
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for algorithm in self.contract["formal"]["algorithms"]:
                values = [
                    float(row["relative_loss_delta"])
                    for row in rows
                    if row["scope"] == "all"
                    and row["algorithm"] == algorithm
                    and float(row["step_multiplier"]) == 1.0
                ]
                frozen = reference[(cell, algorithm)]
                self.assertEqual(len(values), int(frozen["observations"]))
                self.assertAlmostEqual(
                    statistics.mean(values),
                    float(
                        frozen[
                            "predicted_mean_relative_loss_delta_at_multiplier_1"
                        ]
                    ),
                    places=15,
                )
                self.assertAlmostEqual(
                    statistics.median(values),
                    float(
                        frozen[
                            "predicted_median_relative_loss_delta_at_multiplier_1"
                        ]
                    ),
                    places=15,
                )

    def test_checkpoint_certificates_are_frozen(self) -> None:
        checkpoints = self.contract["checkpoints"]
        self.assertEqual(len(checkpoints), 4)
        self.assertEqual(len({row["path"] for row in checkpoints}), 4)
        self.assertEqual(
            len({row["expected_sha256"] for row in checkpoints}), 4
        )
        for row in checkpoints:
            self.assertEqual(len(row["expected_sha256"]), 64)
            self.assertGreater(row["expected_bytes"], 8_000_000_000)

    def test_scope_excludes_efficiency_claims(self) -> None:
        boundary = self.contract["scope_boundary"]
        self.assertTrue(boundary["efficiency_benchmark_excluded"])
        self.assertTrue(
            boundary["formal_throughput_requires_separate_controlled_benchmark"]
        )
        self.assertTrue(boundary["existing_timing_fields_are_descriptive_only"])

    def test_auc_uses_step_widths(self) -> None:
        rows = [
            {"optimizer_step": 0, "normalized_loss": 1.0},
            {"optimizer_step": 1, "normalized_loss": 0.8},
            {"optimizer_step": 3, "normalized_loss": 0.6},
        ]
        self.assertAlmostEqual(A.trapezoid_normalized_auc(rows), 2.3 / 3.0)

    def test_rank_correlations(self) -> None:
        left = [-3.0, -2.0, -1.0, 0.0]
        right = [-6.0, -4.0, -2.0, 0.0]
        self.assertAlmostEqual(A.pearson(left, right), 1.0)
        self.assertAlmostEqual(A.spearman(left, right), 1.0)


if __name__ == "__main__":
    unittest.main()
