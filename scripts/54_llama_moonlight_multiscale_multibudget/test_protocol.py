from pathlib import Path
import json
import unittest

import protocol as P

HERE = Path(__file__).resolve().parent


class ContractTest(unittest.TestCase):
    def test_fresh_moonlight_contract(self) -> None:
        contract = json.loads((HERE / "ex54_contract.json").read_text())
        checks = P.validate_contract(contract)
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual(contract["method"], "moonlight")
        self.assertEqual(contract["execution"]["physical_gpus"], [0, 1])
        self.assertEqual(contract["training"]["device_batch_size_1b"], 8)
        self.assertEqual(contract["training"]["gradient_accumulation_steps_1b"], 64)
        self.assertEqual(contract["formal"]["accepted_1b_budget_ids"], ["tokens_3p2506b", "tokens_6p9694b"])
        self.assertTrue(contract["formal"]["independent_of_ex57"])


    def test_tuning_seeds_are_disjoint_from_formal(self) -> None:
        contract = json.loads((HERE / "ex54_contract.json").read_text())
        tuning = {int(contract["tuning"][scale]["seed"]) for scale in ("124m", "1b")}
        formal = {int(seed) for seed in contract["formal"]["seeds"]}
        self.assertEqual(tuning, {5401, 5402})
        self.assertTrue(tuning.isdisjoint(formal))
        self.assertTrue(contract["fairness"]["tuning_formal_seed_disjoint"])


if __name__ == "__main__":
    unittest.main()
