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


class Mech03ContractTests(unittest.TestCase):
    def test_formal_sampling_contract(self) -> None:
        module = load_module("mech03_controller_test", "run_mech03.py")
        self.assertEqual(module.FORMAL_LAYERS, (0, 4, 8, 11))
        self.assertEqual(module.FORMAL_REPEATS, 4)
        self.assertEqual(module.FORMAL_BATCHES_PER_SPLIT, 8)
        self.assertEqual(len(module.FORMAL_OFFSETS), 64)
        self.assertEqual(len(set(module.FORMAL_OFFSETS)), 64)
        self.assertEqual(module.STEP_MULTIPLIERS, (0.0, 0.25, 0.5, 1.0))

    def test_primary_prediction_is_frozen(self) -> None:
        contract = json.loads(
            (HERE / "prediction_contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            contract["primary_prediction"]["predicted_direction"], "positive"
        )
        self.assertEqual(
            contract["primary_gate"]["minimum_positive_paired_cells"], 24
        )
        self.assertEqual(
            contract["primary_gate"]["minimum_positive_material_layers"], 3
        )
        self.assertTrue(
            contract["authorization_source"]["mech04_never_auto_authorized"]
        )

    def test_worker_has_no_training_or_checkpoint_write(self) -> None:
        source = (HERE / "mech03_worker.py").read_text(encoding="utf-8")
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("wandb.", source)
        self.assertNotIn("torch.save(", source)

    def test_content_invariance_ignores_only_parameter_metadata_digest(self) -> None:
        source = (HERE / "mech03_worker.py").read_text(encoding="utf-8")
        self.assertIn("def model_content_unchanged(", source)
        self.assertNotIn(
            '"model_unchanged": model_before == model_after',
            source,
        )

    def test_analysis_never_auto_authorizes_mech04(self) -> None:
        source = (HERE / "analyze_mech03.py").read_text(encoding="utf-8")
        self.assertIn('"mech04_authorized": False', source)

    def test_synthetic_primary_gate_requires_frozen_direction(self) -> None:
        module = load_module("mech03_analysis_test", "analyze_mech03.py")
        contract = json.loads(
            (HERE / "prediction_contract.json").read_text(encoding="utf-8")
        )
        scores = []
        for family, score in (
            ("r1", -0.01),
            ("gpt_bridge", -0.01),
            ("llama124", 0.0),
        ):
            for repeat in range(4):
                for direction in ("A_to_B", "B_to_A"):
                    for layer in (0, 4, 8, 11):
                        scores.append(
                            {
                                "family": family,
                                "repeat": repeat,
                                "direction": direction,
                                "layer": layer,
                                "diag_minus_none": score,
                            }
                        )
        paired, layers, gate = module.primary_gate(scores, contract)
        self.assertEqual(len(paired), 32)
        self.assertEqual(len(layers), 4)
        self.assertTrue(gate["prediction_gate_passed"])
        self.assertFalse(gate["mech04_authorized"])


if __name__ == "__main__":
    unittest.main()
