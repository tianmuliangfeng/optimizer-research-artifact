from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent


def load(name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


capacity = load(
    "llama_extended_capacity_test", "run_llama_swiglu_extended_capacity.py"
)


class ExtendedLlamaCapacityTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(
            official_repo=Path("/official"),
            python_exe=Path("/venv/bin/python"),
            seed=2026,
        )

    def test_capacity_command_is_moonlight_high_capacity_only(self) -> None:
        command = capacity.capacity_command(
            self.args(), 96, Path("/results/capacity")
        )
        joined = " ".join(command)
        self.assertEqual(command[1], str(capacity.CAPACITY_TRAINER))
        self.assertIn("--method moonlight_muon", joined)
        self.assertIn("--backup-lr 0.003", joined)
        self.assertIn("--matrix-lr 0.003", joined)
        self.assertIn("--extended-weight-decay 0.1", joined)
        self.assertIn("--device-batch-size 96", joined)
        self.assertIn("--global-batch-size 768", joined)
        self.assertIn("--val-tokens 98304", joined)
        self.assertIn("--num-iterations 34", joined)
        self.assertIn("--checkpoint-every 0", joined)
        self.assertIn("--no-save-final", command)

    def test_boundary_is_exact_only_at_integer_width_one(self) -> None:
        rows = [
            {
                "device_batch_size": 128,
                "status": "completed",
                "failure_class": "",
            },
            {
                "device_batch_size": 130,
                "status": "failed",
                "failure_class": "oom",
            },
        ]
        unresolved = capacity.boundary(rows)
        self.assertEqual(unresolved["resolved_width"], 2)
        self.assertFalse(unresolved["exact_integer_boundary"])
        rows.append(
            {
                "device_batch_size": 129,
                "status": "completed",
                "failure_class": "",
            }
        )
        resolved = capacity.boundary(rows)
        self.assertEqual(resolved["max_success_device_batch"], 129)
        self.assertEqual(resolved["first_oom_device_batch"], 130)
        self.assertTrue(resolved["exact_integer_boundary"])

    def test_failure_classifier_distinguishes_oom(self) -> None:
        self.assertEqual(
            capacity.classify_failure("torch.OutOfMemoryError: CUDA out of memory"),
            "oom",
        )
        self.assertEqual(capacity.classify_failure("ValueError: bad config"), "error")


if __name__ == "__main__":
    unittest.main()
