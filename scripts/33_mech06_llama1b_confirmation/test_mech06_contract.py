from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Mech06ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(
            (HERE / "confirmation_contract.json").read_text(encoding="utf-8")
        )

    def test_frozen_checkpoints_and_layers(self) -> None:
        self.assertEqual(self.contract["checkpoints"]["early"]["step"], 1000)
        self.assertEqual(self.contract["checkpoints"]["late"]["step"], 6200)
        self.assertEqual(
            self.contract["formal"]["geometry_layers"], list(range(18))
        )
        self.assertEqual(self.contract["formal"]["shadow_layers"], [0, 6, 12, 17])

    def test_formal_sampling_contract(self) -> None:
        formal = self.contract["formal"]
        self.assertEqual(formal["repeats"], 4)
        self.assertEqual(formal["geometry_batches_per_repeat"], 2)
        self.assertEqual(formal["shadow_batches_per_split"], 8)
        self.assertEqual(formal["candidates"], ["none", "diag", "dense_full"])
        self.assertEqual(formal["step_multipliers"], [0.0, 0.25, 0.5, 1.0])

    def test_mech05_hash_and_retrospective_boundary(self) -> None:
        self.assertEqual(
            self.contract["mech05_contract"]["sha256"],
            "be57755f452eb537d3a4ddc19f82c83cbe87155e39e330085cfcba51bb3681d5",
        )
        interpretation = self.contract["interpretation"]
        self.assertFalse(interpretation["hvp_authorized"])
        self.assertIn("retrospective", interpretation["existing_llama1b_training_rankings"])

    def test_controller_builds_disjoint_formal_offsets(self) -> None:
        controller = load_module("mech06_controller_test", "run_mech06.py")
        formal = self.contract["formal"]
        geometry = [
            index * 4096
            for index in range(
                formal["repeats"] * formal["geometry_batches_per_repeat"]
            )
        ]
        shadow = [
            index * 4096
            for index in range(
                formal["repeats"] * 2 * formal["shadow_batches_per_split"]
            )
        ]
        self.assertEqual(len(geometry), 8)
        self.assertEqual(len(set(geometry)), 8)
        self.assertEqual(len(shadow), 64)
        self.assertEqual(len(set(shadow)), 64)
        self.assertEqual(
            controller.CONTRACT_VERSION, self.contract["contract_version"]
        )
        self.assertEqual(controller.SCRIPT_VERSION, "2026-07-27.4")

    def test_analyzer_primary_inputs_exclude_training_rankings(self) -> None:
        source = (HERE / "analyze_mech06.py").read_text(encoding="utf-8")
        self.assertNotIn("run-summary", source)
        self.assertNotIn("run_summary", source)
        self.assertNotIn("formal_multiseed", source)

    def test_worker_has_no_training_checkpoint_write_or_hvp(self) -> None:
        source = (HERE / "mech06_worker.py").read_text(encoding="utf-8")
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("torch.save(", source)
        self.assertNotIn("wandb.", source)
        self.assertNotIn("autograd.functional.hvp", source)
        self.assertNotIn("backward(create_graph=True", source)

    def test_lowrank_methods_are_frozen(self) -> None:
        formal = self.contract["formal"]
        self.assertEqual(formal["spectrum_method"], "exact_dual_gram_low_rank")
        self.assertEqual(
            formal["dense_inverse_method"], "exact_woodbury_low_rank"
        )

    def test_single_repeat_smoke_allows_header_only_stability(self) -> None:
        source = (HERE / "mech06_worker.py").read_text(encoding="utf-8")
        self.assertEqual(self.contract["smoke"]["repeats"], 1)
        self.assertIn("def write_stability_csv(", source)
        self.assertIn(
            'write_stability_csv(output / "stability.csv", stability_rows)',
            source,
        )

    def test_stabilized_inverse_does_not_use_cancelling_formula(self) -> None:
        source = (HERE / "mech06_worker.py").read_text(encoding="utf-8")
        self.assertIn("torch.linalg.svd(", source)
        self.assertIn("dtype=torch.float64", source)
        self.assertIn("perpendicular / ridge_value", source)
        self.assertNotIn(
            "g / ridge_value - (solved @ u) / (ridge_value * ridge_value)",
            source,
        )

    def test_woodbury_matches_direct_inverse(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is unavailable in the controller environment")

        worker = load_module("mech06_worker_test", "mech06_worker.py")
        generator = torch.Generator().manual_seed(20260727)
        activation = torch.randn(5, 9, generator=generator)
        gradient = torch.randn(4, 9, generator=generator)
        ridge = torch.tensor(0.3)
        observed, residual = worker.woodbury_apply(
            gradient, activation, ridge
        )
        activation64 = activation.double()
        gradient64 = gradient.double()
        ridge64 = ridge.double()
        covariance = activation64.T @ activation64 / activation64.size(0)
        expected = gradient64 @ torch.linalg.inv(
            covariance
            + ridge64
            * torch.eye(covariance.size(0), dtype=torch.float64)
        )
        self.assertTrue(torch.allclose(observed, expected, atol=2e-5, rtol=2e-5))
        self.assertLess(residual, 2e-5)


if __name__ == "__main__":
    unittest.main()
