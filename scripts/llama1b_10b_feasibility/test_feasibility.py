#!/usr/bin/env python3
"""CPU-only regression tests for the LLaMA-1B 10B feasibility package."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import protocol as P


def write_fake_shard(path: Path, tokens: int) -> None:
    header = bytearray(P.HEADER_BYTES)
    struct.pack_into("<iii", header, 0, P.MAGIC, 1, int(tokens))
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(b"\x00\x00" * int(tokens))


class FeasibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = P.read_json(HERE / "feasibility_contract.json")

    def test_contract_is_explicitly_non_launchable(self) -> None:
        checks = P.validate_contract(self.contract)
        self.assertTrue(all(checks.values()), checks)
        self.assertFalse(self.contract["launch_authorized"])
        self.assertFalse(self.contract["remote_training_command_allowed"])
        self.assertFalse(self.contract["geometry_interface"]["enabled"])

    def test_milestone_grid_is_exact_and_frozen(self) -> None:
        rows = P.build_schedule(self.contract)
        self.assertEqual([row["step"] for row in rows], [6200, 13293, 19073])
        self.assertEqual(rows[0]["actual_tokens"], 3_250_585_600)
        self.assertEqual(rows[1]["actual_tokens"], 6_969_360_384)
        self.assertEqual(rows[2]["actual_tokens"], 9_999_745_024)
        self.assertEqual(rows[2]["stream_tokens_including_prefetch"], 9_999_753_216)

    def test_validation_grid_forces_non_round_milestones(self) -> None:
        steps = P.validation_steps(self.contract)
        self.assertIn(13293, steps)
        self.assertIn(19073, steps)
        self.assertEqual(len(steps), 193)

    def test_budget_exposes_multi_day_cost(self) -> None:
        budget = P.build_budget(self.contract)
        self.assertEqual(budget["aggregate_training_tokens"], 39_998_980_096)
        two_gpu_days = budget["gpu_scenarios"]["2"]["wall_seconds"] / 86400.0
        self.assertGreater(two_gpu_days, 3.5)
        self.assertLess(two_gpu_days, 4.2)
        self.assertEqual(budget["recommended_minimum_free_disk_bytes"], 200_000_000_000)

    def test_cursor_includes_prefetched_batch_and_resume_is_exact(self) -> None:
        consumable = [8192 * 100, 8192 * 100]
        row = P.expected_resume_cursor(2, consumable, self.contract)
        self.assertEqual(row["consumed_batches"], 129)
        self.assertEqual(row["current_shard"], 1)
        self.assertEqual(row["current_position"], 29 * 8192)
        self.assertEqual(row["wrap_count"], 0)
        self.assertEqual(row, P.expected_resume_cursor(2, consumable, self.contract))

    def test_data_audit_accepts_contiguous_no_wrap_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(101):
                write_fake_shard(root / f"fineweb_train_{index:06d}.bin", 16385)
            write_fake_shard(root / "fineweb_val_000000.bin", 16385)
            contract = json.loads(json.dumps(self.contract))
            endpoint = contract["milestones"][-1]
            endpoint["target_tokens"] = 1_000_000
            endpoint["expected_step"] = 2
            contract["milestones"] = [endpoint]
            audit = P.audit_data_dir(root, contract)
            self.assertTrue(audit["passed"], audit["checks"])
            self.assertEqual(audit["train_shard_count"], 101)
            self.assertEqual(audit["first_train_index"], 0)
            self.assertEqual(audit["last_train_index"], 100)

    def test_data_audit_rejects_index_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            indices = list(range(100)) + [101]
            for index in indices:
                write_fake_shard(root / f"fineweb_train_{index:06d}.bin", 16385)
            write_fake_shard(root / "fineweb_val_000000.bin", 16385)
            contract = json.loads(json.dumps(self.contract))
            endpoint = contract["milestones"][-1]
            endpoint["target_tokens"] = 1_000_000
            endpoint["expected_step"] = 2
            contract["milestones"] = [endpoint]
            audit = P.audit_data_dir(root, contract)
            self.assertFalse(audit["passed"])
            self.assertFalse(audit["checks"]["numeric_indices_contiguous"])

    def test_current_sources_are_supportive_but_not_10b_ready(self) -> None:
        audit = P.audit_current_sources(REPO)
        self.assertTrue(audit["resume_payload_supportive"])
        self.assertIn("data_audit_requires_exactly_50_shards", audit["blockers"])
        self.assertIn("loader_uses_modulo_wrap", audit["blockers"])
        self.assertIn("forced_validation_grid_missing", audit["blockers"])
        self.assertFalse(audit["launch_ready"])

    def test_plan_report_cannot_upgrade_launch_status(self) -> None:
        source = P.audit_current_sources(REPO)
        report = P.build_report(self.contract, source, None)
        self.assertFalse(report["launch_authorized"])
        self.assertFalse(report["technical_prerequisites_passed"])
        self.assertIn("long_horizon_lr_schedule_unresolved", report["hard_blockers"])
        self.assertIn("remote_data_inventory_not_audited", report["hard_blockers"])


if __name__ == "__main__":
    unittest.main()
