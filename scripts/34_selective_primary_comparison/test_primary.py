#!/usr/bin/env python3

import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("analyze_primary.py")
SPEC = importlib.util.spec_from_file_location("analyze_primary", MODULE_PATH)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class PrimaryContractTests(unittest.TestCase):
    def test_primary_contract_excludes_diag_vs_none(self):
        for _, left, right, _ in M.CONTRASTS:
            self.assertNotEqual({left, right}, {"selective_diag", "selective_none"})

    def test_each_proposal_faces_both_baselines(self):
        pairs = {(left, right) for _, left, right, _ in M.CONTRASTS}
        for proposal in ("selective_diag", "selective_none"):
            self.assertIn((proposal, "muon"), pairs)
            self.assertIn((proposal, "original_newton_muon"), pairs)

    def test_baselines_are_distinct(self):
        for spec in M.FAMILIES.values():
            self.assertNotEqual(spec["methods"]["selective_none"], spec["methods"]["muon"])
            self.assertNotEqual(
                spec["methods"]["original_newton_muon"], spec["methods"]["muon"]
            )

    def test_classification(self):
        self.assertEqual(
            M.classify(-0.003, positive_seeds=0, negative_seeds=3),
            "selective_or_left_materially_better",
        )
        self.assertEqual(
            M.classify(0.003, positive_seeds=3, negative_seeds=0),
            "selective_or_left_materially_worse",
        )
        self.assertEqual(
            M.classify(0.001, positive_seeds=2, negative_seeds=1),
            "within_practical_margin",
        )


if __name__ == "__main__":
    unittest.main()
