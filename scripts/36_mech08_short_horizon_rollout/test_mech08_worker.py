#!/usr/bin/env python3
"""CPU tensor-level regression tests for the MECH-08 worker."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent


def load_worker():
    path = HERE / "mech08_worker.py"
    spec = importlib.util.spec_from_file_location("mech08_worker_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mech08_worker_test"] = module
    spec.loader.exec_module(module)
    return module


W = load_worker()


class Mech08WorkerTests(unittest.TestCase):
    def test_tensor_transfer_audit_accepts_exact_dtype_conversion(self) -> None:
        source = torch.tensor(
            [[1.0, -2.0], [0.125, 4.0]], dtype=torch.bfloat16
        )
        destination = torch.empty_like(source, dtype=torch.float32)
        destination.copy_(source)
        audit = W.tensor_transfer_audit(source, destination)
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["values_match_exactly"])
        self.assertEqual(audit["max_abs_diff"], 0.0)
        self.assertEqual(
            audit["validation_domain"], "destination_device_and_dtype"
        )

    def test_tensor_transfer_audit_rejects_changed_value(self) -> None:
        source = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        destination = source.clone()
        destination[1] += 0.25
        audit = W.tensor_transfer_audit(source, destination)
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["values_match_exactly"])
        self.assertEqual(audit["max_abs_diff"], 0.25)


if __name__ == "__main__":
    unittest.main()
