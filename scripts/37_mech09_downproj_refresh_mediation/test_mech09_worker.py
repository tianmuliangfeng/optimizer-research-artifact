#!/usr/bin/env python3
"""CPU tensor-level tests for the MECH-09 refresh intervention."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent


def load_worker():
    path = HERE / "mech09_worker.py"
    spec = importlib.util.spec_from_file_location("mech09_worker_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mech09_worker_test"] = module
    spec.loader.exec_module(module)
    return module


W = load_worker()


class FakeOptimizer:
    def __init__(self, layers: int = 2) -> None:
        self.global_step = 0
        self._groups = []
        self.state = {}
        for layer in range(layers):
            names = [
                f"layers.{layer}.attn_input",
                f"layers.{layer}.attn_output",
                f"layers.{layer}.mlp_input",
                f"layers.{layer}.down_input",
            ]
            for name in names:
                owner = torch.nn.Parameter(torch.zeros(2, 2))
                group = {
                    "name": name,
                    "members": [owner],
                    "accum": torch.ones(2),
                    "count": torch.tensor(1.0),
                }
                self._groups.append(group)
                self.state[owner] = {
                    "precond_cov": torch.ones(2),
                    "precond_inv_apply": torch.ones(2),
                }

    @torch.no_grad()
    def _refresh_preconditioners(self) -> None:
        for group in self._groups:
            state = self.state[group["members"][0]]
            state["precond_cov"].add_(1.0)
            state["precond_inv_apply"].add_(2.0)
            group["accum"].zero_()
            group["count"].zero_()


def refill_statistics(optimizer: FakeOptimizer) -> None:
    for group in optimizer._groups:
        group["accum"].fill_(1.0)
        group["count"].fill_(1.0)


class Mech09WorkerTests(unittest.TestCase):
    def test_refresh_action_is_exact(self) -> None:
        schedule = (64, 96, 128)
        self.assertEqual(W.refresh_action(32, schedule), "hold")
        self.assertEqual(W.refresh_action(64, schedule), "refresh")
        self.assertEqual(W.refresh_action(96, schedule), "refresh")

    def test_group_partition_targets_only_down_projection(self) -> None:
        optimizer = FakeOptimizer()
        target, other = W.partition_groups(
            optimizer._groups, ".down_input"
        )
        self.assertEqual(len(target), 2)
        self.assertEqual(len(other), 6)
        self.assertTrue(
            all(group["name"].endswith(".down_input") for group in target)
        )
        self.assertTrue(
            all(not group["name"].endswith(".down_input") for group in other)
        )

    def test_delayed_policy_holds_then_refreshes_target(self) -> None:
        optimizer = FakeOptimizer()
        controller = W.RefreshInterventionController(
            optimizer,
            target_suffix=".down_input",
            target_refresh_steps=(4,),
            expected_other_refresh_steps=(2, 4),
            expected_layers=2,
        )
        initial = W.group_state_snapshot(
            optimizer,
            controller.target_groups,
            include_statistics=True,
        )
        optimizer.global_step = 1
        optimizer._refresh_preconditioners()
        refill_statistics(optimizer)
        optimizer.global_step = 3
        optimizer._refresh_preconditioners()
        final = W.group_state_snapshot(
            optimizer,
            controller.target_groups,
            include_statistics=True,
        )
        audit = controller.audit(initial, final)
        self.assertTrue(audit["passed"])
        self.assertEqual(
            [row["target_action"] for row in audit["events"]],
            ["hold", "refresh"],
        )
        self.assertFalse(audit["events"][0]["target_inverse_changed"])
        self.assertTrue(audit["events"][1]["target_inverse_changed"])

    def test_frozen_policy_never_changes_target_state(self) -> None:
        optimizer = FakeOptimizer()
        controller = W.RefreshInterventionController(
            optimizer,
            target_suffix=".down_input",
            target_refresh_steps=(),
            expected_other_refresh_steps=(2, 4),
            expected_layers=2,
        )
        initial = W.group_state_snapshot(
            optimizer,
            controller.target_groups,
            include_statistics=True,
        )
        for global_step in (1, 3):
            optimizer.global_step = global_step
            optimizer._refresh_preconditioners()
            refill_statistics(optimizer)
        for group in controller.target_groups:
            group["accum"].zero_()
            group["count"].zero_()
        final = W.group_state_snapshot(
            optimizer,
            controller.target_groups,
            include_statistics=True,
        )
        audit = controller.audit(initial, final)
        self.assertTrue(audit["passed"])
        self.assertEqual(
            [row["target_action"] for row in audit["events"]],
            ["hold", "hold"],
        )
        self.assertEqual(
            initial["inverse_fingerprint_sha256"],
            final["inverse_fingerprint_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
