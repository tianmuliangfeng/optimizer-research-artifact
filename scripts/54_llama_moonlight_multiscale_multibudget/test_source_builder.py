from pathlib import Path
import unittest

import source_builder as S

HERE = Path(__file__).resolve().parent


class SourceBuilderTest(unittest.TestCase):
    def test_moonlight_route_and_no_mousse_runtime(self) -> None:
        source = (HERE / "source_builder.py").read_text()
        optimizer = (HERE / "moonlight_optimizer.py").read_text()
        self.assertIn('Moonlight optimizer route', source)
        self.assertIn("R1MoonlightMuon", source)
        self.assertIn("0.2 * math.sqrt", optimizer)
        self.assertIn("logical_matrix_slices", optimizer)
        self.assertNotIn("torch.linalg.eigh", optimizer)
        self.assertNotIn("factor_epsilon", optimizer)

    def test_moonlight_algorithm_is_exact_ex19_transfer(self) -> None:
        optimizer = (HERE / "moonlight_optimizer.py").read_text()
        receipt = S.audit_moonlight_transfer(HERE.parents[1], optimizer)
        self.assertTrue(receipt["passed"])
        self.assertEqual(
            receipt["reference_sha256"],
            "bf39d7e1b435ef737833046c564ce8770d858d1aa474c9d7f11a914057253655",
        )
        with self.assertRaises(RuntimeError):
            S.audit_moonlight_transfer(
                HERE.parents[1],
                optimizer.replace(
                    "orthogonal.mul_(0.2 * math.sqrt",
                    "orthogonal.mul_(0.21 * math.sqrt",
                    1,
                ),
            )


if __name__ == "__main__":
    unittest.main()
