from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("llama_1b_capacity_test", HERE / "run_llama_swiglu_1b_capacity.py")
assert spec is not None and spec.loader is not None
capacity = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capacity
spec.loader.exec_module(capacity)


class CapacityProtocolTests(unittest.TestCase):
    def test_default_grid_and_methods(self) -> None:
        args = capacity.parse_args(["--official-repo", "/official", "--python-exe", "/python"])
        self.assertEqual(args.methods, ["down_none", "down_diag", "newton_full"])
        self.assertEqual(args.device_batches, [16, 32, 64, 128])

    def test_grid_is_sorted_and_must_divide_512(self) -> None:
        args = capacity.parse_args(["--official-repo", "/official", "--python-exe", "/python", "--device-batches", "64", "16", "32", "16"])
        self.assertEqual(args.device_batches, [16, 32, 64])
        with self.assertRaises(SystemExit):
            capacity.parse_args(["--official-repo", "/official", "--python-exe", "/python", "--device-batches", "24"])

    def test_failure_classification(self) -> None:
        self.assertEqual(capacity.classify_failure("torch.OutOfMemoryError: CUDA out of memory"), "oom")
        self.assertEqual(capacity.classify_failure("dataset checksum mismatch"), "error")


if __name__ == "__main__":
    unittest.main()
