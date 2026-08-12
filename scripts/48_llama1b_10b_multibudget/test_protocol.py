#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ex48_protocol", HERE / "protocol.py")
assert spec is not None and spec.loader is not None
P = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = P
spec.loader.exec_module(P)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads((HERE / "formal_contract.json").read_text(encoding="utf-8"))

    def test_contract(self) -> None:
        checks = P.validate_contract(self.contract)
        self.assertTrue(all(checks.values()), checks)

    def test_single_host_four_gpu_execution_contract(self) -> None:
        self.assertEqual(self.contract["grid"]["host_count"], 1)
        self.assertEqual(self.contract["grid"]["gpus"], 4)
        self.assertEqual(self.contract["runtime"]["gpu_count"], 4)
        self.assertEqual(self.contract["runtime"]["nvidia_driver"], "580.95.05")

    def test_endpoint_steps_and_tokens(self) -> None:
        endpoints = P.endpoint_phases(self.contract)
        self.assertEqual([row["target_step"] for row in endpoints], [6200, 13293, 19073])
        self.assertEqual(
            [row["target_step"] * self.contract["training"]["tokens_per_update"] for row in endpoints],
            [3250585600, 6969360384, 9999745024],
        )

    def test_parent_graph(self) -> None:
        self.assertEqual(P.direct_children(self.contract, "backbone_4400"), ["cooldown_6200", "backbone_11493"])
        self.assertEqual(P.direct_children(self.contract, "backbone_11493"), ["cooldown_13293", "backbone_17273"])
        self.assertEqual(P.direct_children(self.contract, "backbone_17273"), ["cooldown_19073"])

    def test_lr_schedule_matches_equal_cooldown(self) -> None:
        phase = P.phase_map(self.contract)["cooldown_6200"]
        self.assertEqual(P.lr_multiplier(phase, 4400), 1.0)
        self.assertAlmostEqual(P.lr_multiplier(phase, 6199), 1 / 1800)
        self.assertEqual(P.lr_multiplier(phase, 6200), 0.0)

    def test_validation_forces_non_round_nodes(self) -> None:
        phase = P.phase_map(self.contract)["cooldown_13293"]
        self.assertTrue(P.should_validate(phase, 11493, 100))
        self.assertTrue(P.should_validate(phase, 13293, 100))
        self.assertFalse(P.should_validate(phase, 13292, 100))

    def test_cursor_no_wrap_and_overflow(self) -> None:
        row = P.cursor_after_batches([100, 100], 10, 11)
        self.assertEqual(row, {"current_shard": 1, "current_position": 10, "wrap_count": 0, "consumed_batches": 11})
        overflow = P.cursor_after_batches([100, 100], 10, 20)
        self.assertEqual(overflow["wrap_count"], 1)

    def test_atomic_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            P.atomic_json(path, {"b": 2, "a": 1})
            self.assertEqual(P.read_json(path), {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
