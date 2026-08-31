from pathlib import Path
import json
import unittest

import analyze

HERE = Path(__file__).resolve().parent


class AnalyzeStateTest(unittest.TestCase):
    def test_momentum_only_state(self) -> None:
        contract = json.loads((HERE / "ex54_contract.json").read_text())
        count = contract["profiles"]["1b"]["expected_matrix_tensors"]
        summary = {
            "moonlight_hyperparameters": analyze.expected_moonlight_hyperparameters(contract),
            "moonlight_state_schema": {
                "optimizer": "R1MoonlightMuon",
                "tensor_state_keys": ["momentum_buffer"],
                "logical_matrix_parameters": count,
                "contains_activation_k_state": False,
                "contains_factor_or_eigendecomposition_state": False,
            },
            "momentum_buffer_bytes": 1234,
            "moonlight_matrix_optimizer_state_bytes": 1234,
        }
        self.assertEqual(analyze.state_fields(summary, contract, "1b")["moonlight_momentum_state_bytes"], 1234)


if __name__ == "__main__":
    unittest.main()
