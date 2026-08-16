#!/usr/bin/env python3
"""CPU-only tests for Experiment-50 analysis gates."""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_global_diag import (
    EXPECTED_MEMORY,
    T_95_DF2,
    classify_primary_delta,
    collect_controls,
    collect_formal,
    mean_ci,
    main as analyze_main,
)


class GlobalDiagAnalysisTests(unittest.TestCase):
    def test_three_branch_policy_is_frozen(self) -> None:
        self.assertEqual(
            classify_primary_delta(0.0021, 0.002),
            "global_diag_worse_than_selective_diag",
        )
        self.assertEqual(
            classify_primary_delta(0.002, 0.002),
            "descriptively_close_not_formal_equivalence",
        )
        self.assertEqual(
            classify_primary_delta(-0.0021, 0.002),
            "global_diag_better_than_selective_diag",
        )

    def test_mean_ci_uses_frozen_t_critical(self) -> None:
        mean, sd, low, high = mean_ci([-0.003, -0.004, -0.005])
        self.assertAlmostEqual(mean, -0.004)
        self.assertAlmostEqual(sd, 0.001)
        expected_half = T_95_DF2 * 0.001 / math.sqrt(3)
        self.assertAlmostEqual(low, -0.004 - expected_half)
        self.assertAlmostEqual(high, -0.004 + expected_half)

    def test_frozen_controls_are_complete(self) -> None:
        rows = collect_controls(SCRIPT_DIR / "frozen_r1_controls.csv")
        self.assertEqual(len(rows), 12)

    def test_collect_formal_hard_gates_seed_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed, loss in ((2024, 3.26), (2025, 3.25), (2026, 3.27)):
                unit = root / "formal" / f"seed{seed}" / "batch" / "run"
                unit.mkdir(parents=True)
                payload = {
                    "method": "global_diag",
                    "controlled_seed": seed,
                    "final_val_loss": loss,
                    "best_val_loss": loss,
                    "final_val_step": 6200,
                    "init_sha256": "a" * 64,
                    "derived_script_sha256": "b" * 64,
                    "peak_memory_allocated_mib": 1,
                    **EXPECTED_MEMORY,
                }
                (unit / "r1_summary.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                with (unit / "r1_metrics.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=("event", "step", "loss")
                    )
                    writer.writeheader()
                    for step in (5800, 5900, 6000, 6100, 6200):
                        writer.writerow(
                            {"event": "validation", "step": step, "loss": loss}
                        )
            rows = collect_formal(root)
            self.assertEqual([row["seed"] for row in rows], [2024, 2025, 2026])
            self.assertTrue(all(row["validation_points"] == 5 for row in rows))
            output = root / "analysis"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "analyze_global_diag.py",
                    "--run-dir",
                    str(root),
                    "--contract",
                    str(SCRIPT_DIR / "global_diag_contract.json"),
                    "--controls",
                    str(SCRIPT_DIR / "frozen_r1_controls.csv"),
                    "--output-dir",
                    str(output),
                ],
            ):
                analyze_main()
            manifest = json.loads(
                (output / "analysis_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["passed"])
            self.assertEqual(manifest["primary_contrast"]["comparator"], "diag")
            self.assertTrue(manifest["checks"]["formal_seed_grid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
