#!/usr/bin/env python3
"""CPU contract and algebra tests for audit 40."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
WORKER_PATH = HERE / "llama_block_partition_worker.py"
spec = importlib.util.spec_from_file_location("audit40_worker_tests", WORKER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {WORKER_PATH}")
W = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = W
spec.loader.exec_module(W)


class LlamaBlockPartitionAuditTests(unittest.TestCase):
    def test_contract_forbids_primary_baseline_claim(self) -> None:
        contract = json.loads(
            (HERE / "audit_contract.json").read_text(encoding="utf-8")
        )
        scope = contract["scientific_scope"]
        self.assertFalse(scope["new_training"])
        self.assertFalse(scope["hvp_authorized"])
        self.assertEqual(scope["official_original_newton_muon_control"], "newton_full")
        self.assertIn("never a primary baseline", scope["block4_role"])

    def test_architecture_is_balanced_but_not_semantic(self) -> None:
        architecture = json.loads(
            (HERE / "audit_contract.json").read_text(encoding="utf-8")
        )["architecture"]
        self.assertEqual(
            architecture["intermediate_size"],
            architecture["block_count"] * architecture["block_size"],
        )
        self.assertFalse(architecture["four_semantic_subspaces"])

    def test_map_new_coordinates_back_to_old(self) -> None:
        value = torch.arange(15, dtype=torch.float64).view(3, 5)
        permutation = torch.tensor([2, 4, 0, 1, 3])
        new = value[:, permutation]
        observed = W.map_new_to_old(new, permutation)
        torch.testing.assert_close(observed, value, atol=0, rtol=0)

    def test_within_block_permutation_stays_inside_blocks(self) -> None:
        permutation = W.within_block_permutation(32, 4, 4040)
        labels = torch.arange(32) // 8
        torch.testing.assert_close(labels[permutation], labels, atol=0, rtol=0)
        overlap = W.partition_overlap(permutation, 4)
        self.assertEqual(overlap["same_block_pair_jaccard"], 1.0)

    def test_global_permutation_changes_partition(self) -> None:
        permutation = W.global_permutation(64, 4001)
        overlap = W.partition_overlap(permutation, 4)
        self.assertLess(overlap["same_block_pair_jaccard"], 0.5)
        self.assertLess(overlap["coordinate_retention_fraction"], 0.5)

    def test_block_inverse_is_equivariant_within_blocks(self) -> None:
        generator = torch.Generator().manual_seed(12)
        x = torch.randn((24, 16), generator=generator, dtype=torch.float64)
        gradient = torch.randn((7, 16), generator=generator, dtype=torch.float64)
        base, _ = W.block_inverse_apply(gradient, x, 0.2, 4)
        permutation = W.within_block_permutation(16, 4, 17)
        observed, _ = W.block_inverse_apply(
            gradient[:, permutation], x[:, permutation], 0.2, 4
        )
        observed = W.map_new_to_old(observed, permutation)
        torch.testing.assert_close(observed, base, atol=2e-10, rtol=2e-10)

    def test_block_inverse_changes_under_cross_block_permutation(self) -> None:
        generator = torch.Generator().manual_seed(31)
        x = torch.randn((12, 16), generator=generator, dtype=torch.float64)
        x[:, 4:8] += 0.8 * x[:, :4]
        gradient = torch.randn((5, 16), generator=generator, dtype=torch.float64)
        base, _ = W.block_inverse_apply(gradient, x, 0.15, 4)
        permutation = W.global_permutation(16, 4001)
        observed, _ = W.block_inverse_apply(
            gradient[:, permutation], x[:, permutation], 0.15, 4
        )
        observed = W.map_new_to_old(observed, permutation)
        self.assertGreater(W.relative_drift(observed, base), 1e-3)

    def test_dense_inverse_is_permutation_equivariant(self) -> None:
        generator = torch.Generator().manual_seed(91)
        x = torch.randn((15, 20), generator=generator, dtype=torch.float64)
        gradient = torch.randn((6, 20), generator=generator, dtype=torch.float64)
        base, _ = W.M6.woodbury_apply(gradient, x, torch.tensor(0.25))
        permutation = W.global_permutation(20, 4002)
        observed, _ = W.M6.woodbury_apply(
            gradient[:, permutation],
            x[:, permutation],
            torch.tensor(0.25),
        )
        observed = W.map_new_to_old(observed, permutation)
        torch.testing.assert_close(observed, base, atol=2e-10, rtol=2e-10)

    def test_off_block_energy_within_block_invariant(self) -> None:
        generator = torch.Generator().manual_seed(77)
        x = torch.randn((20, 24), generator=generator, dtype=torch.float64)
        identity = torch.arange(24)
        within = W.within_block_permutation(24, 4, 88)
        base = W.off_block_energy(x, identity, 4)
        observed = W.off_block_energy(x, within, 4)
        self.assertAlmostEqual(
            base["off_block_energy_fraction"],
            observed["off_block_energy_fraction"],
            places=12,
        )

    def test_permutation_hashes_are_deterministic_and_distinct(self) -> None:
        first = W.build_permutations(32, 4, 4040, [4001, 4002])
        second = W.build_permutations(32, 4, 4040, [4001, 4002])
        self.assertEqual(
            [row["sha256"] for row in first],
            [row["sha256"] for row in second],
        )
        self.assertEqual(len({row["sha256"] for row in first}), len(first))


if __name__ == "__main__":
    unittest.main()
