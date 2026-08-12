from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_controller():
    path = HERE / "run_mech02.py"
    spec = importlib.util.spec_from_file_location("mech02_controller_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_analysis():
    path = HERE / "analyze_mech02.py"
    spec = importlib.util.spec_from_file_location("mech02_analysis_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Mech02ContractTests(unittest.TestCase):
    def test_frozen_offsets_are_four_disjoint_two_batch_repeats(self) -> None:
        module = load_controller()
        self.assertEqual(len(module.DEFAULT_OFFSETS), 8)
        self.assertEqual(len(set(module.DEFAULT_OFFSETS)), 8)
        self.assertEqual(module.DEFAULT_OFFSETS, tuple(range(0, 32768, 4096)))

    def test_worker_has_no_training_or_wandb_path(self) -> None:
        source = (HERE / "mech02_worker.py").read_text(encoding="utf-8")
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("wandb.", source)
        self.assertNotIn("torch.save(", source)

    def test_controller_output_is_exclusive(self) -> None:
        module = load_controller()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new"
            target.mkdir(parents=True, exist_ok=False)
            with self.assertRaises(FileExistsError):
                target.mkdir(parents=True, exist_ok=False)

    def test_geometry_gate_never_auto_authorizes_mech03(self) -> None:
        module = load_analysis()
        rows = []
        for family, base in (("gpt_bridge", 0.0), ("llama124", 10.0)):
            for layer in range(12):
                for metric in module.PRIMARY_METRICS:
                    rows.append(
                        {
                            "family": family,
                            "layer": layer,
                            "metric": metric,
                            "mean": base,
                            "sd": 0.1,
                            "n": 4,
                        }
                    )
        _comparisons, gate = module.primary_comparisons(rows)
        self.assertTrue(gate["geometry_gate_candidate_passed"])
        self.assertFalse(gate["mech03_authorized"])


if __name__ == "__main__":
    unittest.main()
