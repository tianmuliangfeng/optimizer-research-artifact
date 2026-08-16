#!/usr/bin/env python3
"""CPU-only source/contract tests for Experiment 50."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from global_diag_source_builder import (
    EXPECTED_ACTIVATION_STAT_BYTES,
    EXPECTED_K_COV_BYTES,
    EXPECTED_K_INV_BYTES,
    EXPECTED_K_STATE_BYTES,
    build_global_diag_source,
    expected_memory_contract,
    self_test_global_diag_math,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_repo() -> Path | None:
    override = os.environ.get("EX50_OFFICIAL_REPO") or os.environ.get(
        "SNM_OFFICIAL_REPO"
    )
    candidates = [
        Path(override) if override else None,
        SCRIPT_DIR.parents[2] / "Newton-Muon-official",
        SCRIPT_DIR.parents[2] / "Newton-Muon-official-r0",
    ]
    for candidate in candidates:
        if candidate is not None and (
            candidate / "train_gpt_newton_muon_1.py"
        ).is_file():
            return candidate.resolve()
    return None


class GlobalDiagSourceTests(unittest.TestCase):
    def test_memory_contract_is_exact(self) -> None:
        self.assertEqual(EXPECTED_K_COV_BYTES, 258_048)
        self.assertEqual(EXPECTED_K_INV_BYTES, 258_048)
        self.assertEqual(EXPECTED_K_STATE_BYTES, 516_096)
        self.assertEqual(EXPECTED_ACTIVATION_STAT_BYTES, 258_240)
        self.assertEqual(
            expected_memory_contract(),
            {
                "k_cov_bytes": 258_048,
                "k_inv_bytes": 258_048,
                "k_state_bytes": 516_096,
                "activation_stat_bytes": 258_240,
                "precond_workspace_bytes": 0,
            },
        )

    def test_pure_diagonal_math(self) -> None:
        self_test_global_diag_math()

    def test_contract_and_frozen_control_grid(self) -> None:
        contract = json.loads(
            (SCRIPT_DIR / "global_diag_contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(contract["experiment_id"], "50_r1_global_activation_diag")
        self.assertEqual(contract["seeds"], [2024, 2025, 2026])
        self.assertEqual(contract["primary_comparator"], "diag")
        self.assertFalse(contract["formal_equivalence_claim_allowed"])
        self.assertEqual(
            contract["global_diag_route"]["expected_k_state_bytes"], 516_096
        )
        controls_path = SCRIPT_DIR / "frozen_r1_controls.csv"
        self.assertEqual(
            sha256_file(controls_path),
            contract["frozen_controls"]["frozen_reference_sha256"],
        )
        with controls_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            {(row["method"], int(row["seed"])) for row in rows},
            {
                (method, seed)
                for method in ("block4", "diag", "none", "muon")
                for seed in (2024, 2025, 2026)
            },
        )

    @unittest.skipIf(official_repo() is None, "official R0 repository unavailable")
    def test_source_derivation_is_deterministic_and_compiles(self) -> None:
        repo = official_repo()
        assert repo is not None
        first = build_global_diag_source(repo)
        second = build_global_diag_source(repo)
        self.assertEqual(first.derived_sha256, second.derived_sha256)
        self.assertEqual(first.source, second.source)
        compile(first.source, "<global-diag-test>", "exec")
        self.assertEqual(first.method, "global_diag")
        self.assertIn('"kind": "qkv_diag"', first.source)
        self.assertIn('"kind": "o_diag"', first.source)
        self.assertIn('"kind": "c_fc_diag"', first.source)
        self.assertIn('"kind": "c_proj_diag"', first.source)
        self.assertNotIn("self.xtx_tmp", first.source)
        self.assertNotIn("self.fc_xtx_tmp", first.source)
        self.assertIn("model_parameter_keys", first.source)
        self.assertIn(
            "and _r1_storage_key(tensor) not in model_parameter_keys",
            first.source,
        )
        diagonal_inverse = first.source.index(
            'if self._apply_plan["inv_input_diag"] is not None:'
        )
        dense_gate = first.source.index(
            "if self._refresh_K is None or not self._refresh_map:"
        )
        self.assertLess(diagonal_inverse, dense_gate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
