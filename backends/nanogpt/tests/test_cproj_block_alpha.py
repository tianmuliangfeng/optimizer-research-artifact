import unittest

import torch

from optimizers import BlockDiagInputCovState


class BlockAlphaInputCovStateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.x = torch.randn(7, 8)

    def make_state(self, alpha):
        return BlockDiagInputCovState(
            n=8,
            blocks=4,
            device=torch.device("cpu"),
            beta=0.0,
            ridge=0.2,
            refresh_interval=1,
            max_samples=None,
            offdiag_alpha=alpha,
        )

    def test_alpha_one_matches_existing_block4_path(self):
        block4 = BlockDiagInputCovState(
            n=8,
            blocks=4,
            device=torch.device("cpu"),
            beta=0.0,
            ridge=0.2,
            refresh_interval=1,
            max_samples=None,
        )
        block_alpha_one = self.make_state(1.0)
        block4.maybe_refresh(self.x, step=0)
        block_alpha_one.maybe_refresh(self.x, step=0)

        self.assertEqual(block4.state_bytes(), block_alpha_one.state_bytes())
        for expected, actual in zip(block4.states, block_alpha_one.states):
            torch.testing.assert_close(actual.K, expected.K, atol=0.0, rtol=0.0)
            torch.testing.assert_close(
                actual.K_inv,
                expected.K_inv,
                atol=0.0,
                rtol=0.0,
            )

    def test_alpha_zero_removes_within_block_offdiagonals(self):
        block_alpha_zero = self.make_state(0.0)
        block_alpha_zero.maybe_refresh(self.x, step=0)

        for state in block_alpha_zero.states:
            inverse_offdiag = state.K_inv - torch.diag_embed(
                state.K_inv.diagonal()
            )
            torch.testing.assert_close(
                inverse_offdiag,
                torch.zeros_like(inverse_offdiag),
                atol=1e-7,
                rtol=0.0,
            )

    def test_intermediate_alpha_changes_inverse_but_not_running_covariance(self):
        alpha_zero = self.make_state(0.0)
        alpha_half = self.make_state(0.5)
        alpha_one = self.make_state(1.0)
        for state in (alpha_zero, alpha_half, alpha_one):
            state.maybe_refresh(self.x, step=0)

        for zero, half, one in zip(
            alpha_zero.states,
            alpha_half.states,
            alpha_one.states,
        ):
            torch.testing.assert_close(zero.K, half.K, atol=0.0, rtol=0.0)
            torch.testing.assert_close(half.K, one.K, atol=0.0, rtol=0.0)
            self.assertFalse(torch.equal(zero.K_inv, half.K_inv))
            self.assertFalse(torch.equal(half.K_inv, one.K_inv))

    def test_efficient_apply_matches_explicit_strict_block_matrix(self):
        state = self.make_state(0.5)
        state.maybe_refresh(self.x, step=0)
        gradient = torch.randn(3, 8)

        explicit_inverse = torch.block_diag(
            *(block.K_inv for block in state.states)
        )
        expected = gradient.float() @ explicit_inverse.float()
        actual = state.apply_right(gradient)

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
