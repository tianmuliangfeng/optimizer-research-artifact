import unittest

import torch

from optimizer_factory import cproj_param_needs_input_hook
from optimizers import CProjKModeNewtonMuon, DiagInputCovState, InputCovState


def matrix(name: str) -> tuple[torch.nn.Parameter, str]:
    return torch.nn.Parameter(torch.ones(4, 8)), name


class CProjLayerModeTest(unittest.TestCase):
    def make_optimizer(self, *, mode: str, layers=()):
        p0, n0 = matrix("transformer.h.0.mlp.c_proj.weight")
        p1, n1 = matrix("transformer.h.1.mlp.c_proj.weight")
        other, other_name = matrix("transformer.h.1.mlp.c_fc.weight")
        params = [p0, p1, other]
        optimizer = CProjKModeNewtonMuon(
            params,
            param_to_module={},
            param_to_name={p0: n0, p1: n1, other: other_name},
            cproj_k_mode=mode,
            cproj_k_layers=layers,
        )
        return optimizer, p0, p1, other

    def test_empty_target_list_preserves_all_cproj_behavior(self):
        optimizer, p0, p1, other = self.make_optimizer(mode="diag")
        group = optimizer.param_groups[0]
        self.assertEqual(optimizer._init_mode_state(p0, group)["k_mode"], "diag")
        self.assertEqual(optimizer._init_mode_state(p1, group)["k_mode"], "diag")
        self.assertEqual(optimizer._init_mode_state(other, group)["k_mode"], "full")

    def test_selected_depth_uses_diag_and_other_depth_stays_full(self):
        optimizer, p0, p1, other = self.make_optimizer(mode="diag", layers=(1,))
        group = optimizer.param_groups[0]
        state0 = optimizer._init_mode_state(p0, group)
        state1 = optimizer._init_mode_state(p1, group)
        other_state = optimizer._init_mode_state(other, group)

        self.assertEqual(state0["k_mode"], "full")
        self.assertIsInstance(state0["input_cov"], InputCovState)
        self.assertEqual(state1["k_mode"], "diag")
        self.assertIsInstance(state1["input_cov"], DiagInputCovState)
        self.assertEqual(other_state["k_mode"], "full")

    def test_selected_depth_uses_none_and_other_depth_stays_full(self):
        optimizer, p0, p1, _ = self.make_optimizer(mode="none", layers=(1,))
        group = optimizer.param_groups[0]
        state0 = optimizer._init_mode_state(p0, group)
        state1 = optimizer._init_mode_state(p1, group)

        self.assertEqual(state0["k_mode"], "full")
        self.assertIsInstance(state0["input_cov"], InputCovState)
        self.assertEqual(state1["k_mode"], "none")
        self.assertIsNone(state1["input_cov"])

    def test_target_layer_validation(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.make_optimizer(mode="diag", layers=(1, 1))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.make_optimizer(mode="diag", layers=(-1,))


class CProjLayerHookTest(unittest.TestCase):
    def test_targeted_none_skips_only_selected_cproj_hook(self):
        kwargs = {
            "cproj_k_mode": "none",
            "cproj_k_layers": (1,),
        }
        self.assertTrue(
            cproj_param_needs_input_hook(
                "transformer.h.0.mlp.c_proj.weight", **kwargs
            )
        )
        self.assertFalse(
            cproj_param_needs_input_hook(
                "transformer.h.1.mlp.c_proj.weight", **kwargs
            )
        )
        self.assertTrue(
            cproj_param_needs_input_hook(
                "transformer.h.1.mlp.c_fc.weight", **kwargs
            )
        )

    def test_shadow_state_restores_hook_for_targeted_none(self):
        self.assertTrue(
            cproj_param_needs_input_hook(
                "transformer.h.1.mlp.c_proj.weight",
                cproj_k_mode="none",
                cproj_k_layers=(1,),
                cproj_shadow_k_modes=("diag",),
                cproj_shadow_k_layers=(1,),
            )
        )


if __name__ == "__main__":
    unittest.main()
