from __future__ import annotations

import unittest

import torch
from torch import nn

from malt_optimizer import R1MALT, exact_polar, run_small_matrix_reference_audit, state_schema


class MALTOptimizerTests(unittest.TestCase):
    def test_literal_reference_and_formula_invariants(self) -> None:
        report = run_small_matrix_reference_audit("cpu")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["activation_k_state_routes"], 0)

    def test_qkv_state_shape_and_mass_conservation(self) -> None:
        parameter = nn.Parameter(torch.zeros(6, 2))
        optimizer = R1MALT(
            [parameter],
            lr=1e-3,
            orthogonalize=exact_polar,
            variant="malt",
            weight_decay=0.1,
        )
        torch.manual_seed(49)
        for _ in range(3):
            parameter.grad = torch.randn_like(parameter)
            optimizer.step()
        schema = state_schema(optimizer)
        self.assertEqual(schema["roles"]["malt_momentum"], 1)
        self.assertEqual(schema["roles"]["malt_row_ema"], 3)
        self.assertEqual(schema["roles"]["malt_col_ema"], 3)
        self.assertEqual(schema["optimizer_group_steps"], [3])
        self.assertTrue(schema["numerical_checks_passed"])
        self.assertLessEqual(schema["row_column_mass_max_relative_error"], 1e-5)

    def test_malter_has_one_scalar_per_logical_matrix(self) -> None:
        parameter = nn.Parameter(torch.zeros(6, 2))
        optimizer = R1MALT(
            [parameter],
            lr=1e-3,
            orthogonalize=exact_polar,
            variant="malter",
        )
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        schema = state_schema(optimizer)
        self.assertEqual(schema["roles"]["malt_nu"], 3)


if __name__ == "__main__":
    unittest.main()
