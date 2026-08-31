#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
WORKSPACE = REPO.parent
DEFAULT_R0 = WORKSPACE / "Newton-Muon-official-r0"
DEFAULT_LOCAL_FALLBACK = WORKSPACE / "Newton-Muon-official"
DEFAULT_OFFICIAL = DEFAULT_R0 if DEFAULT_R0.is_dir() else DEFAULT_LOCAL_FALLBACK
OFFICIAL_REPO = Path(
    os.environ.get("EX53_OFFICIAL_REPO", str(DEFAULT_OFFICIAL))
).expanduser().resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from matched_diag_source_builder import (
    ARM_CONFIGS,
    EXPECTED_MEMORY,
    assert_matched_diag_source_contract,
    build_matched_diag_sources,
    expected_memory_contract,
)


class MatchedDiagSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (OFFICIAL_REPO / "train_gpt_newton_muon_1.py").is_file():
            raise unittest.SkipTest(f"local official repository is missing: {OFFICIAL_REPO}")
        cls.built = build_matched_diag_sources(OFFICIAL_REPO)
        cls.source = next(iter(cls.built.values())).source

    def test_exact_five_arms(self) -> None:
        self.assertEqual(
            tuple(ARM_CONFIGS),
            (
                "all_none",
                "c_fc_diag",
                "c_proj_diag",
                "c_fc_c_proj_diag",
                "o_proj_diag",
            ),
        )

    def test_qkv_is_none_in_every_arm(self) -> None:
        self.assertTrue(all(config["qkv"] == "none" for config in ARM_CONFIGS.values()))

    def test_first_four_are_strict_factorial(self) -> None:
        cells = {
            (ARM_CONFIGS[arm]["c_fc"], ARM_CONFIGS[arm]["c_proj"])
            for arm in tuple(ARM_CONFIGS)[:4]
        }
        self.assertEqual(cells, {("none", "none"), ("diag", "none"), ("none", "diag"), ("diag", "diag")})

    def test_one_parameterized_source_hash(self) -> None:
        self.assertEqual(len({item.derived_sha256 for item in self.built.values()}), 1)
        self.assertEqual({item.method for item in self.built.values()}, set(ARM_CONFIGS))

    def test_source_compiles_and_contract_passes(self) -> None:
        compile(self.source, "<test-ex53-source>", "exec")
        assert_matched_diag_source_contract(self.source)

    def test_source_has_no_target_dense_buffers(self) -> None:
        forbidden = (
            "self.qkv_xtx_accum",
            "self.qkv_xtx_count",
            "self.fc_xtx_tmp",
            "self.o_xtx_tmp",
            "self.proj_xtx_tmp",
            "torch.ops.nanogpt.accum_xtx_blocks4(",
        )
        self.assertEqual([item for item in forbidden if item in self.source], [])

    def test_every_module_uses_the_same_coordinate_diagonal_ridge(self) -> None:
        self.assertNotIn(
            "ridge = cov.mean(dim=-1) * self.precond_ridge_mult", self.source
        )
        self.assertGreaterEqual(
            self.source.count(
                "ridge = cov.mean() * self.precond_ridge_mult + self.precond_eps"
            ),
            2,
        )
        for anchor in (
            "precond_init_diag: float = 0.001",
            "do_refresh = (t % 32 == 0)",
            "precond_ewma = 0.950",
        ):
            self.assertIn(anchor, self.source)

    def test_state_free_arm_is_supported(self) -> None:
        self.assertIn('else self.param_groups[0]["params"][0].device', self.source)
        self.assertEqual(expected_memory_contract("all_none")["k_state_bytes"], 0)
        self.assertEqual(expected_memory_contract("all_none")["activation_stat_bytes"], 0)

    def test_expected_memory_is_additive(self) -> None:
        fc = EXPECTED_MEMORY["c_fc_diag"]
        proj = EXPECTED_MEMORY["c_proj_diag"]
        both = EXPECTED_MEMORY["c_fc_c_proj_diag"]
        for key in ("k_cov_bytes", "k_inv_bytes", "k_state_bytes", "activation_stat_bytes"):
            self.assertEqual(both[key], fc[key] + proj[key])
        self.assertEqual(EXPECTED_MEMORY["o_proj_diag"], fc)

    def test_unknown_memory_arm_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            expected_memory_contract("post_hoc_arm")

    def test_json_contract_matches_builder(self) -> None:
        contract = json.loads((SCRIPT_DIR / "matched_diag_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["formal_seeds"], [2024, 2025, 2026])
        self.assertEqual(contract["pilot"]["seed"], 2053)
        self.assertFalse(contract["pilot"]["outcome_eligible"])
        self.assertEqual(contract["expected_memory_bytes"], EXPECTED_MEMORY)
        representation = contract["matched_diagonal_representation"]
        self.assertEqual(representation["initial_diagonal"], 0.001)
        self.assertEqual(representation["ridge_reduction"], "one_mean_over_all_input_coordinates_of_each_matrix")
        self.assertEqual(representation["layer_coverage"], "all_12_transformer_layers")
        self.assertEqual(
            {row["arm"]: {key: row[key] for key in ("c_fc", "c_proj", "o_proj", "qkv")} for row in contract["arms"]},
            ARM_CONFIGS,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
