from __future__ import annotations

import ast
import json
import os
import unittest
from pathlib import Path

import torch

from mousse_optimizer import (
    R1Mousse,
    clean_eigenvalues,
    logical_matrix_slices,
    run_small_matrix_reference_audit,
)
from mousse_source_builder import OFFICIAL_R1_CANONICAL_SHA256, build_source, canonical_sha256


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parents[1]
_official_env = os.environ.get("SNM_OFFICIAL_REPO")
_official_candidates = (
    ARTIFACT_ROOT / "third_party" / "Newton-Muon-official-r0",
    ARTIFACT_ROOT / "third_party" / "Newton-Muon-official",
)
OFFICIAL_REPO = (
    Path(_official_env).expanduser().resolve()
    if _official_env
    else next((path for path in _official_candidates if path.is_dir()), _official_candidates[0])
)


class MousseOptimizerTests(unittest.TestCase):
    def test_qkv_is_three_logical_matrices(self) -> None:
        parts = logical_matrix_slices(torch.zeros(12, 4))
        self.assertEqual([tuple(part.shape) for part in parts], [(4, 4)] * 3)
        self.assertEqual(len(logical_matrix_slices(torch.zeros(4, 12))), 1)

    def test_clean_eigenvalues_matches_upstream_shift(self) -> None:
        values = clean_eigenvalues(torch.tensor([-2.0, 1.0]), 1e-5)
        torch.testing.assert_close(values, torch.tensor([1e-5, 3.00001]))

    def test_reference_audit_crosses_second_refresh(self) -> None:
        audit = run_small_matrix_reference_audit("cpu")
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["refresh_steps"], [1, 11])
        self.assertEqual(audit["refresh_counts_per_logical_matrix"], [2, 2, 2])
        self.assertEqual(audit["activation_k_state_routes"], 0)

    def test_state_dict_round_trip(self) -> None:
        parameter = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = R1Mousse([parameter], lr=0.015)
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        saved = optimizer.state_dict()
        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = R1Mousse([restored_parameter], lr=0.015)
        restored.load_state_dict(saved)
        self.assertEqual(restored.param_groups[0]["step"], 1)
        self.assertIn("mousse_factor_L_0", restored.state[restored_parameter])

    @unittest.skipUnless(
        OFFICIAL_REPO.is_dir(), f"pinned upstream repo unavailable: {OFFICIAL_REPO}"
    )
    def test_pinned_r1_source_derivation(self) -> None:
        observed = canonical_sha256((OFFICIAL_REPO / "train_gpt_muon_1.py").read_bytes())
        self.assertEqual(observed, OFFICIAL_R1_CANONICAL_SHA256)
        derived = build_source(OFFICIAL_REPO)
        ast.parse(derived.source)
        for marker in ("R1M_METADATA ", "R1M_ROUTING ", "R1M_HYPERPARAMS ", "R1M_FINAL_MEMORY "):
            self.assertIn(marker, derived.source)
        self.assertIn("optimizer2 = R1Mousse", derived.source)
        self.assertIn("weight_decay=0.0", derived.source)
        self.assertIn("R1M_MATRIX_WEIGHT_DECAY", derived.source)

    def test_contract_freezes_grid_and_formal_budget(self) -> None:
        contract = json.loads((SCRIPT_DIR / "mousse_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["pilot"]["matrix_lrs"], [0.012, 0.015, 0.018])
        self.assertEqual(contract["formal"]["seeds"], [2026, 2024, 2025])
        self.assertEqual(contract["r1_scaffold"]["formal_tokens"], 3_250_585_600)
        self.assertFalse(contract["upstream"].get("unchanged_official_reproduction", False))


if __name__ == "__main__":
    unittest.main()
