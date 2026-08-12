#!/usr/bin/env python3
"""Contract tests for the experiment-41 diagonal bridge."""

from __future__ import annotations

import unittest

import analyze_r1_diag_bridge as D


def synthetic_rows(
    diag_losses: tuple[float, float, float] = (3.2613, 3.2599, 3.2621),
) -> list[dict[str, object]]:
    losses = {
        "none": (3.2670, 3.2659, 3.2671),
        "diag": diag_losses,
        "block4": (3.2630, 3.2609, 3.2627),
    }
    states = {"none": 162.0, "diag": 162.28125, "block4": 378.0}
    peaks = {"none": 38304.0, "diag": 38304.0, "block4": 39168.0}
    rows: list[dict[str, object]] = []
    for method in D.METHODS:
        for seed, final in zip(D.SEEDS, losses[method]):
            rows.append(
                {
                    "method": method,
                    "run_name": f"{method}_{seed}",
                    "seed": seed,
                    "initial_val_loss": 10.0 + (seed - 2024) * 0.01,
                    "final_val_loss": final,
                    "tail5_val_loss_mean": final + 0.008,
                    "normalized_val_auc": final + 0.35,
                    "peak_memory_mib": peaks[method],
                    "k_state_mib": states[method],
                    "optimizer_state_mib": 700.0 + states[method],
                }
            )
    return rows


class R1DiagBridgeTests(unittest.TestCase):
    def test_effects_recover_three_seed_paired_deltas(self) -> None:
        rows = synthetic_rows()
        _, summaries = D.build_effects(rows, 0.002)
        diag_none = D.primary_summary(summaries, "diag_minus_none")
        diag_block4 = D.primary_summary(
            summaries, "diag_minus_block4"
        )
        self.assertAlmostEqual(
            diag_none["mean"], -0.005566666666666571
        )
        self.assertEqual(diag_none["negative_seeds"], 3)
        self.assertAlmostEqual(
            diag_block4["mean"], -0.0010999999999999528
        )
        self.assertEqual(diag_block4["negative_seeds"], 3)

    def test_decision_recovers_quality_matching_claim(self) -> None:
        rows = synthetic_rows()
        cells = D.build_cells(rows)
        _, summaries = D.build_effects(rows, 0.002)
        decision = D.build_decision(cells, summaries, 0.002)
        self.assertEqual(
            decision["classification"],
            "diag_recovers_block4_quality_at_near_none_state_cost",
        )
        self.assertTrue(decision["diag_beneficial_over_none"])
        self.assertTrue(decision["diag_quality_matched_to_block4"])
        self.assertFalse(decision["diag_superior_to_block4"])
        self.assertFalse(decision["new_training_recommended"])
        self.assertAlmostEqual(
            decision["diag_extra_k_state_vs_none_mib"], 0.28125
        )
        self.assertAlmostEqual(
            decision["diag_k_state_saved_vs_block4_mib"], 215.71875
        )

    def test_superiority_requires_material_effect_and_interval(self) -> None:
        rows = synthetic_rows((3.2500, 3.2480, 3.2490))
        cells = D.build_cells(rows)
        _, summaries = D.build_effects(rows, 0.002)
        decision = D.build_decision(cells, summaries, 0.002)
        self.assertTrue(decision["diag_superior_to_block4"])

    def test_pareto_rows_use_none_as_quality_baseline(self) -> None:
        rows = synthetic_rows()
        cells = D.build_cells(rows)
        pareto = {
            row["method"]: row for row in D.build_pareto_rows(cells)
        }
        self.assertAlmostEqual(
            pareto["diag"]["loss_improvement_vs_none"],
            0.005566666666666571,
        )
        self.assertAlmostEqual(
            pareto["diag"]["k_state_saved_vs_block4_mib"],
            215.71875,
        )
        self.assertEqual(pareto["none"]["loss_improvement_vs_none"], 0.0)

    def test_report_artifact_has_complete_technical_reading_path(
        self,
    ) -> None:
        rows = synthetic_rows()
        cells = D.build_cells(rows)
        by_seed, summaries = D.build_effects(rows, 0.002)
        decision = D.build_decision(cells, summaries, 0.002)
        pareto = D.build_pareto_rows(cells)
        artifact = D.build_report_artifact(
            cells, by_seed, pareto, decision
        )
        self.assertEqual(artifact["surface"], "report")
        self.assertEqual(artifact["snapshot"]["status"], "ready")
        self.assertEqual(
            artifact["manifest"]["blocks"][0]["body"],
            "# Experiment 41D: R1 diagonal bridge",
        )
        self.assertGreaterEqual(
            sum(
                block["type"] == "chart"
                for block in artifact["manifest"]["blocks"]
            ),
            2,
        )
        self.assertEqual(
            {source["id"] for source in artifact["manifest"]["sources"]},
            {
                "experiment15_formal_r1",
                "experiment15_formal_r1_cells",
                "experiment41_accepted",
            },
        )


if __name__ == "__main__":
    unittest.main()
