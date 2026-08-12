#!/usr/bin/env python3
"""Contract tests for the unified mechanism synthesis."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_unified_mechanism.py")
SPEC = importlib.util.spec_from_file_location("mech38", MODULE_PATH)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class UnifiedMechanismTests(unittest.TestCase):
    def test_primary_comparison_order_is_frozen(self) -> None:
        self.assertEqual(
            M.PRIMARY_CONTRASTS,
            (
                "selective_diag_vs_muon",
                "selective_none_vs_muon",
                "selective_diag_vs_original_newton_muon",
                "selective_none_vs_original_newton_muon",
            ),
        )
        self.assertNotIn("selective_diag_vs_selective_none", M.ALLOWED_CONTRASTS)

    def test_percentile_interpolates(self) -> None:
        self.assertAlmostEqual(M.percentile([0.0, 10.0], 0.25), 2.5)
        self.assertEqual(M.percentile([3.0], 0.9), 3.0)

    def test_cluster_bootstrap_is_deterministic(self) -> None:
        rows = [
            {
                "checkpoint_cell": origin,
                "data_replica": str(replica),
                "delta_left_minus_right": str(value),
            }
            for origin, values in (("a", (-2.0, -1.0)), ("b", (-4.0, -3.0)))
            for replica, value in enumerate(values)
        ]
        first = M.cluster_bootstrap(rows, 200, 19)
        second = M.cluster_bootstrap(rows, 200, 19)
        self.assertEqual(first, second)
        self.assertEqual(first["classification"], "left_better")
        self.assertEqual(first["clusters"], 2)
        self.assertEqual(first["paired_units"], 4)

    def test_cluster_bootstrap_rejects_duplicate_pair(self) -> None:
        rows = [
            {
                "checkpoint_cell": "a",
                "data_replica": "0",
                "delta_left_minus_right": "-1",
            },
            {
                "checkpoint_cell": "a",
                "data_replica": "0",
                "delta_left_minus_right": "-2",
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            M.cluster_bootstrap(rows, 100, 1)

    def test_contrast_validator_rejects_diag_none(self) -> None:
        rows = [{"contrast": name} for name in M.ALLOWED_CONTRASTS]
        M.validate_contrast_set(rows)
        rows.append({"contrast": "selective_diag_vs_selective_none"})
        with self.assertRaisesRegex(RuntimeError, "diag-vs-none"):
            M.validate_contrast_set(rows)

    def test_loss_sign_convention(self) -> None:
        negative = [
            {
                "checkpoint_cell": origin,
                "data_replica": "0",
                "delta_left_minus_right": "-0.1",
            }
            for origin in ("a", "b")
        ]
        self.assertEqual(
            M.cluster_bootstrap(negative, 100, 7)["classification"], "left_better"
        )

    def test_foundational_mode_order_is_frozen(self) -> None:
        self.assertEqual(
            M.FOUNDATIONAL_MODE_ORDER,
            ("diag", "none", "block4", "dense_full", "muon"),
        )

    def test_three_seed_contract_rejects_missing_or_duplicate_seed(self) -> None:
        rows = [{"seed": str(seed)} for seed in (2024, 2025, 2026)]
        indexed = M.require_three_seed_rows(rows, "test")
        self.assertEqual(sorted(indexed), [2024, 2025, 2026])
        with self.assertRaisesRegex(RuntimeError, "duplicate seed"):
            M.require_three_seed_rows(rows + [{"seed": "2024"}], "test")
        with self.assertRaisesRegex(RuntimeError, "expected seeds"):
            M.require_three_seed_rows(rows[:2], "test")

    def test_architecture_transfer_boundary_preserves_claim_limits(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            classification = root / "classification.json"
            checkpoint = root / "checkpoint_summary.csv"
            payload = {
                "classification": "strong_non_invariance",
                "decision_statistics": {
                    "pooled_global_block4_median_update_drift": 0.34,
                    "maximum_equivariant_control_drift": 0.015,
                    "effect_to_control_multiple": 22.0,
                },
                "checkpoint_summaries": [
                    {
                        "checkpoint_label": "early",
                        "checkpoint_step": 1000,
                        "global_block4_update_drift_median": 0.31,
                        "global_block4_update_drift_p95": 0.55,
                    },
                    {
                        "checkpoint_label": "late",
                        "checkpoint_step": 6200,
                        "global_block4_update_drift_median": 0.39,
                        "global_block4_update_drift_p95": 0.58,
                    },
                ],
                "interpretation": {
                    "block4_is_original_newton_muon": False,
                    "block4_is_primary_baseline": False,
                    "official_original_newton_muon_control": "newton_full",
                    "claim_if_supported": "coordinate-partition dependent",
                    "claim_not_authorized": "full-training ordering",
                },
            }
            classification.write_text(json.dumps(payload), encoding="utf-8")
            checkpoint.write_text(
                "checkpoint_label,checkpoint_step,"
                "global_block4_update_drift_median,"
                "global_block4_update_drift_p95\n"
                "early,1000,0.31,0.55\n"
                "late,6200,0.39,0.58\n",
                encoding="utf-8",
            )
            resolved = {
                "llama_block_partition_invariance": {
                    classification.name: classification,
                    checkpoint.name: checkpoint,
                }
            }
            row = M.build_architecture_transfer_boundary(resolved)[0]
            self.assertEqual(row["classification"], "strong_non_invariance")
            self.assertFalse(row["block4_is_primary_baseline"])
            self.assertEqual(
                row["official_original_newton_muon_control"], "newton_full"
            )


if __name__ == "__main__":
    unittest.main()
