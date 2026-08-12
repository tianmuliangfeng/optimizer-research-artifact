from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_analysis():
    path = HERE / "analyze_mech05.py"
    spec = importlib.util.spec_from_file_location("mech05_analysis_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Mech05ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_analysis()
        cls.contract = json.loads(
            (HERE / "selection_rule_contract.json").read_text(encoding="utf-8")
        )
        cls.thresholds = cls.contract["thresholds"]

    def test_discovery_scope_and_confirmation_boundary(self) -> None:
        self.assertEqual(
            self.contract["discovery_scope"]["families"], ["r1", "llama124"]
        )
        self.assertEqual(self.contract["discovery_scope"]["seed"], 2026)
        self.assertNotIn(
            "llama1b", " ".join(self.contract["discovery_scope"]["families"]).lower()
        )
        self.assertFalse(
            self.contract["anti_leakage"]["allow_threshold_tuning_from_llama1b"]
        )

    def test_frozen_thresholds(self) -> None:
        self.assertEqual(self.thresholds["relative_shadow_loss_margin"], 1e-6)
        self.assertEqual(self.thresholds["minimum_positive_cell_fraction"], 0.75)
        self.assertEqual(self.thresholds["minimum_positive_material_layers"], 3)
        self.assertEqual(self.thresholds["longrun_practical_loss_margin"], 0.002)

    def base_geometry(self, **updates):
        value = {
            "stable_diagonal_anisotropy": False,
            "stable_non_diagonal_subspace": False,
            "near_scalar_isotropy": False,
            "diagonal_anisotropy_present": True,
            "geometry_stable": False,
        }
        value.update(updates)
        return value

    def contrast(self, candidate, stable):
        return {
            "candidate": candidate,
            "stable_positive_advantage": stable,
        }

    def longrun(self, gains, material):
        best = max(gains, key=gains.get)
        return {
            "candidate_advantage_over_none": gains,
            "best_k_candidate": best,
            "best_k_advantage_over_none": gains[best],
            "material_k_gain_over_none": material,
        }

    def test_synthetic_diag_decision(self) -> None:
        result = self.module.choose_decision(
            "synthetic",
            self.base_geometry(
                stable_diagonal_anisotropy=True, geometry_stable=True
            ),
            {
                "diag_vs_none": self.contrast("diag", True),
                "dense_full_vs_diag": self.contrast("dense_full", False),
            },
            self.longrun({"diag": 0.003, "dense_full": 0.001}, True),
            self.thresholds,
        )
        self.assertEqual(result["decision"], "diag")

    def test_synthetic_full_decision(self) -> None:
        result = self.module.choose_decision(
            "synthetic",
            self.base_geometry(
                stable_diagonal_anisotropy=True,
                stable_non_diagonal_subspace=True,
                geometry_stable=True,
            ),
            {
                "diag_vs_none": self.contrast("diag", True),
                "dense_full_vs_diag": self.contrast("dense_full", True),
            },
            self.longrun({"diag": 0.001, "dense_full": 0.004}, True),
            self.thresholds,
        )
        self.assertEqual(result["decision"], "full_or_block")

    def test_synthetic_none_decision(self) -> None:
        result = self.module.choose_decision(
            "synthetic",
            self.base_geometry(near_scalar_isotropy=True),
            {
                "diag_vs_none": self.contrast("diag", False),
                "dense_full_vs_diag": self.contrast("dense_full", False),
            },
            self.longrun({"diag": 0.0002, "dense_full": -0.0001}, False),
            self.thresholds,
        )
        self.assertEqual(result["decision"], "none_or_muon_sufficient")

    def test_synthetic_conflict_is_uncertain(self) -> None:
        result = self.module.choose_decision(
            "synthetic",
            self.base_geometry(),
            {
                "diag_vs_none": self.contrast("diag", False),
                "dense_full_vs_diag": self.contrast("dense_full", False),
            },
            self.longrun({"diag": 0.005, "dense_full": 0.001}, True),
            self.thresholds,
        )
        self.assertEqual(result["decision"], "uncertain")

    def test_no_training_or_network_side_effects(self) -> None:
        source = (HERE / "analyze_mech05.py").read_text(encoding="utf-8")
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("wandb.", source)
        self.assertNotIn("torch.", source)
        self.assertNotIn("requests.", source)


if __name__ == "__main__":
    unittest.main()
