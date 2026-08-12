#!/usr/bin/env python3

import unittest

from build_mechanism_closure import average_ranks, center_by_group, eta_squared, pearson, spearman


class MechanismClosureMathTests(unittest.TestCase):
    def test_average_ranks_with_tie(self) -> None:
        self.assertEqual(average_ranks([3.0, 1.0, 1.0, 2.0]), [4.0, 1.5, 1.5, 3.0])

    def test_correlations(self) -> None:
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(spearman([3, 1, 2], [30, 10, 20]), 1.0)

    def test_group_centering_and_eta(self) -> None:
        values = [1.0, 3.0, 10.0, 12.0]
        groups = ["a", "a", "b", "b"]
        self.assertEqual(center_by_group(values, groups), [-1.0, 1.0, -1.0, 1.0])
        self.assertGreater(eta_squared(values, groups), 0.9)


if __name__ == "__main__":
    unittest.main()
