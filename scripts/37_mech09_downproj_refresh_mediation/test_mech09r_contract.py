#!/usr/bin/env python3
"""CPU-only contract tests for MECH-09R."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
CONTRACT = json.loads(
    (HERE / "refresh_mediation_repair_contract.json").read_text(
        encoding="utf-8"
    )
)


class Mech09RContractTests(unittest.TestCase):
    def test_protocol_amendment_is_pre_intervention_only(self) -> None:
        amendment = CONTRACT["protocol_amendment"]
        self.assertTrue(amendment["trigger_uses_pre_intervention_data_only"])
        trigger = amendment["trigger"]
        tolerance = float(trigger["frozen_tolerance"])
        self.assertEqual(int(trigger["pre_intervention_step"]), 16)
        self.assertGreater(
            float(trigger["maximum_abs_delayed_vs_frozen_delta"]),
            tolerance,
        )
        self.assertIn("do not relax thresholds", amendment["repair_principle"])

    def test_three_arms_and_primary_comparisons_are_exact(self) -> None:
        self.assertEqual(
            set(CONTRACT["arms"]),
            {
                "production_newton_muon",
                "delayed_down_refresh",
                "frozen_down_refresh",
            },
        )
        observed = {
            (row["left"], row["right"])
            for row in CONTRACT["comparison_contract"]["primary"]
        }
        self.assertEqual(
            observed,
            {
                ("delayed_down_refresh", "production_newton_muon"),
                ("frozen_down_refresh", "production_newton_muon"),
                ("delayed_down_refresh", "frozen_down_refresh"),
            },
        )
        self.assertFalse(
            CONTRACT["comparison_contract"][
                "selective_diag_vs_selective_none_is_primary"
            ]
        )

    def test_causal_tree_covers_every_arm_step_once(self) -> None:
        for tier in ("smoke", "formal"):
            config = CONTRACT[tier]
            tree = config["causal_tree"]
            end = int(config["rollout_steps"])
            shared_end = int(tree["shared_all_end_step"])
            first = int(tree["first_branch_step"])
            common_end = int(tree["shared_no_down_end_step"])
            second = int(tree["second_branch_step"])
            self.assertEqual(first, shared_end + 1)
            self.assertEqual(second, common_end + 1)
            shared = list(range(1, shared_end + 1))
            production = shared + list(range(first, end + 1))
            no_down = list(range(first, common_end + 1))
            delayed = shared + no_down + list(range(second, end + 1))
            frozen = shared + no_down + list(range(second, end + 1))
            expected = list(range(1, end + 1))
            self.assertEqual(production, expected)
            self.assertEqual(delayed, expected)
            self.assertEqual(frozen, expected)

    def test_refresh_schedules_match_tree(self) -> None:
        for tier in ("smoke", "formal"):
            config = CONTRACT[tier]
            interval = int(config["production_refresh_interval"])
            expected = list(
                range(interval, int(config["rollout_steps"]) + 1, interval)
            )
            self.assertEqual(
                config["expected_global_refresh_completed_steps"], expected
            )
            production = CONTRACT["arms"]["production_newton_muon"][
                f"{tier}_down_refresh_completed_steps"
            ]
            delayed = CONTRACT["arms"]["delayed_down_refresh"][
                f"{tier}_down_refresh_completed_steps"
            ]
            frozen = CONTRACT["arms"]["frozen_down_refresh"][
                f"{tier}_down_refresh_completed_steps"
            ]
            self.assertEqual(production, expected)
            self.assertTrue(set(delayed).issubset(expected))
            self.assertEqual(frozen, [])
            self.assertEqual(
                min(delayed),
                int(config["causal_tree"]["second_branch_step"]),
            )

    def test_formal_caps_and_compute_are_frozen(self) -> None:
        formal = CONTRACT["formal"]
        jobs = len(formal["origins"]) * len(formal["data_replicas"])
        self.assertEqual(jobs, 12)
        self.assertEqual(
            jobs, CONTRACT["stopping_rule"]["maximum_new_formal_jobs"]
        )
        self.assertEqual(
            jobs * len(CONTRACT["arms"]),
            CONTRACT["stopping_rule"]["maximum_trajectories"],
        )
        tree = formal["causal_tree"]
        computed_steps = (
            int(tree["shared_all_end_step"])
            + (
                int(formal["rollout_steps"])
                - int(tree["first_branch_step"])
                + 1
            )
            + (
                int(tree["shared_no_down_end_step"])
                - int(tree["first_branch_step"])
                + 1
            )
            + 2
            * (
                int(formal["rollout_steps"])
                - int(tree["second_branch_step"])
                + 1
            )
        )
        self.assertEqual(computed_steps, 290)

    def test_frozen_hyperparameters_match_legacy_design(self) -> None:
        restart = CONTRACT["restart_intervention"]
        self.assertEqual(restart["backup_lr"], 0.0036)
        self.assertEqual(restart["matrix_lr"], 0.01)
        self.assertEqual(restart["matrix_momentum"], 0.95)
        self.assertEqual(restart["ns_steps"], 5)
        self.assertEqual(restart["newton_input_beta"], 0.95)
        self.assertEqual(restart["newton_input_ridge"], 0.2)
        self.assertEqual(restart["production_refresh_steps"], 32)


if __name__ == "__main__":
    unittest.main()
