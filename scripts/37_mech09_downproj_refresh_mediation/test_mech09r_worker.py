#!/usr/bin/env python3
"""CPU-only state and provenance tests for the MECH-09R worker."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import torch


HERE = Path(__file__).resolve().parent


def load_worker():
    path = HERE / "mech09r_worker.py"
    spec = importlib.util.spec_from_file_location("mech09r_worker_tested", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


W = load_worker()
CONTRACT = json.loads(
    (HERE / "refresh_mediation_repair_contract.json").read_text(
        encoding="utf-8"
    )
)


class Mech09RWorkerTests(unittest.TestCase):
    def test_cpu_clone_tree_is_independent(self) -> None:
        source = {
            "tensor": torch.arange(8, dtype=torch.float32),
            "nested": [torch.ones(2), {"value": 3}],
        }
        cloned = W.cpu_clone_tree(source)
        source["tensor"].add_(10)
        source["nested"][0].zero_()
        self.assertTrue(
            torch.equal(cloned["tensor"], torch.arange(8, dtype=torch.float32))
        )
        self.assertTrue(torch.equal(cloned["nested"][0], torch.ones(2)))

    def test_sampled_bundle_fingerprint_is_stable_and_sensitive(self) -> None:
        kwargs = {
            "model_state": {"weight": torch.arange(32).reshape(4, 8)},
            "backup_state": {"state": {0: {"momentum": torch.ones(9)}}},
            "matrix_state": {"state": {0: {"inverse": torch.eye(5)}}},
            "x": torch.tensor([[1, 2, 3]]),
            "y": torch.tensor([[2, 3, 4]]),
            "loader_state": {"current_shard": 1, "current_position": 20},
            "matrix_global_step": 31,
        }
        first = W.sampled_bundle_fingerprint(**kwargs)
        second = W.sampled_bundle_fingerprint(**kwargs)
        self.assertEqual(first["sha256"], second["sha256"])
        kwargs["matrix_state"]["state"][0]["inverse"][0, 0] = 2
        changed = W.sampled_bundle_fingerprint(**kwargs)
        self.assertNotEqual(first["sha256"], changed["sha256"])

    def test_fingerprint_match_checks_next_batch_and_loader(self) -> None:
        payload = {
            "sha256": "a",
            "structure_sha256": "s",
            "tensor_count": 4,
            "sampled_values_finite": True,
            "next_x_sha256": "x",
            "next_y_sha256": "y",
            "loader_state": {"position": 7},
            "matrix_global_step": 31,
        }
        self.assertTrue(
            W.fingerprint_match(payload, dict(payload), "same")["passed"]
        )
        changed = dict(payload)
        changed["loader_state"] = {"position": 8}
        self.assertFalse(
            W.fingerprint_match(payload, changed, "changed")["passed"]
        )

    def test_exact_optimizer_restore_preserves_custom_fp32_state(self) -> None:
        parameter = torch.nn.Parameter(
            torch.ones(4, dtype=torch.bfloat16)
        )
        optimizer = torch.optim.SGD(
            [parameter], lr=0.2, momentum=0.9
        )
        optimizer.state[parameter]["momentum_buffer"] = torch.arange(
            4, dtype=torch.float32
        )
        optimizer.state[parameter]["custom_inverse"] = torch.eye(
            2, dtype=torch.float32
        )
        live = optimizer.state_dict()
        saved = W.cpu_clone_tree(live)
        devices = W.tensor_device_tree(live)
        optimizer.param_groups[0]["lr"] = 9.0
        optimizer.state[parameter]["momentum_buffer"] = torch.zeros(
            4, dtype=torch.bfloat16
        )
        audit = W.restore_optimizer_state_exact(
            optimizer, saved, devices
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.2)
        self.assertEqual(
            optimizer.state[parameter]["momentum_buffer"].dtype,
            torch.float32,
        )
        self.assertEqual(
            optimizer.state[parameter]["custom_inverse"].dtype,
            torch.float32,
        )
        self.assertTrue(
            torch.equal(
                optimizer.state[parameter]["momentum_buffer"],
                torch.arange(4, dtype=torch.float32),
            )
        )

    def test_assign_arm_preserves_shared_provenance(self) -> None:
        rows = [
            {
                "trajectory_node": "shared_no_down",
                "optimizer_step": 48,
                "heldout_loss": 2.5,
            }
        ]
        delayed = W.assign_arm(rows, "delayed_down_refresh")
        frozen = W.assign_arm(rows, "frozen_down_refresh")
        self.assertEqual(
            delayed[0]["heldout_loss"], frozen[0]["heldout_loss"]
        )
        self.assertEqual(
            delayed[0]["source_trajectory_node"], "shared_no_down"
        )
        self.assertNotIn("arm", rows[0])

    def test_exact_shared_evaluation_audit(self) -> None:
        rows = []
        for arm in CONTRACT["arms"]:
            rows.append(
                {
                    "arm": arm,
                    "optimizer_step": 16,
                    "heldout_loss": 3.0,
                }
            )
        for arm in ("delayed_down_refresh", "frozen_down_refresh"):
            rows.append(
                {
                    "arm": arm,
                    "optimizer_step": 48,
                    "heldout_loss": 2.9,
                }
            )
        audit = W.exact_shared_evaluation_audit(rows, CONTRACT, "formal")
        self.assertTrue(audit["passed"])
        rows[-1]["heldout_loss"] = 2.90001
        self.assertFalse(
            W.exact_shared_evaluation_audit(rows, CONTRACT, "formal")[
                "passed"
            ]
        )


if __name__ == "__main__":
    unittest.main()
