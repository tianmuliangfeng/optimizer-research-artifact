from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fine = load("llama_1b_capacity_fine_test", "run_llama_swiglu_1b_capacity_fine.py")
cell = load("llama_1b_capacity_fine_cell_test", "run_llama_swiglu_1b_capacity_fine_cell.py")


class FineCapacityProtocolTests(unittest.TestCase):
    def test_default_grid_is_even_32_through_64(self) -> None:
        args = fine.parse_args(["--official-repo", "/official", "--python-exe", "/python"])
        self.assertEqual(args.device_batches, list(range(32, 65, 2)))
        self.assertEqual(args.methods, ["down_none", "down_diag", "newton_full", "muon"])

    def test_non_divisors_of_512_are_allowed(self) -> None:
        args = fine.parse_args(
            [
                "--official-repo",
                "/official",
                "--python-exe",
                "/python",
                "--device-batches",
                "40",
                "32",
                "36",
                "34",
            ]
        )
        self.assertEqual(args.device_batches, [32, 34, 36, 40])

    def test_grid_requires_anchor_and_stays_in_boundary(self) -> None:
        with self.assertRaises(SystemExit):
            fine.parse_args(
                ["--official-repo", "/official", "--python-exe", "/python", "--device-batches", "34", "36"]
            )
        with self.assertRaises(SystemExit):
            fine.parse_args(
                ["--official-repo", "/official", "--python-exe", "/python", "--device-batches", "32", "66"]
            )

    def test_global_batch_uses_fixed_accumulation(self) -> None:
        self.assertEqual(fine.global_batch(34), 272)
        self.assertEqual(fine.global_batch(40), 320)

    def test_gpu_must_be_idle_before_each_cell(self) -> None:
        accepted = fine.validate_gpu_baseline({"free_bytes": 99, "total_bytes": 100})
        self.assertEqual(accepted["free_bytes"], 99)
        with self.assertRaises(RuntimeError):
            fine.validate_gpu_baseline({"free_bytes": 97, "total_bytes": 100})
        with self.assertRaises(RuntimeError):
            fine.validate_gpu_baseline(None)

    def test_boundary_summary(self) -> None:
        rows = [
            {"method": "down_none", "device_batch_size": 32, "status": "completed", "failure_class": ""},
            {"method": "down_none", "device_batch_size": 34, "status": "completed", "failure_class": ""},
            {"method": "down_none", "device_batch_size": 36, "status": "failed", "failure_class": "oom"},
        ]
        result = fine.boundary_summary(rows, ["down_none"])["down_none"]
        self.assertEqual(result["max_tested_success_batch"], 34)
        self.assertEqual(result["first_tested_oom_batch"], 36)
        self.assertTrue(result["resolved_to_batch_two"])

    def test_internal_worker_rewrites_only_parser_placeholder(self) -> None:
        argv = [
            "worker.py",
            "--stage",
            "smoke",
            "--device-batch-size",
            "38",
            "--capacity-accumulation-steps",
            "8",
        ]
        cleaned, requested, accumulation = cell.split_internal_args(argv)
        self.assertEqual(requested, 38)
        self.assertEqual(accumulation, 8)
        self.assertNotIn("--capacity-accumulation-steps", cleaned)
        self.assertEqual(cleaned[cleaned.index("--device-batch-size") + 1], "8")

    def test_fine_common_config_changes_only_global_batch(self) -> None:
        original = cell.original_common_config
        try:
            cell.original_common_config = lambda args, smoke: {
                "global_batch_size": 512,
                "device_batch_size": args.device_batch_size,
                "sequence_length": 1024,
            }
            args = SimpleNamespace(device_batch_size=38, capacity_accumulation_steps=8)
            config = cell.fine_common_config(args, True)
            self.assertEqual(config["global_batch_size"], 304)
            self.assertEqual(config["device_batch_size"], 38)
            self.assertEqual(config["sequence_length"], 1024)
        finally:
            cell.original_common_config = original


if __name__ == "__main__":
    unittest.main()
