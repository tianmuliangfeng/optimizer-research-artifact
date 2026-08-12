#!/usr/bin/env python3

import ast
import importlib.util
import json
import pathlib
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parent
CONTRACT = json.loads(
    (HERE / "family_contrast_contract.json").read_text(encoding="utf-8")
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_mech07", HERE / "analyze_mech07.py"
)
assert SPEC and SPEC.loader
ANALYZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZER
SPEC.loader.exec_module(ANALYZER)
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_mech07", HERE / "run_mech07.py"
)
assert RUNNER_SPEC and RUNNER_SPEC.loader
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)


class Mech07ContractTests(unittest.TestCase):
    def test_eight_cells_cover_four_methods_and_two_stages(self):
        rows = CONTRACT["checkpoints"]
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["stage"] for row in rows}, {"early", "late"})
        self.assertEqual(
            {row["method"] for row in rows},
            {"down_diag", "down_none", "newton_full", "muon"},
        )
        self.assertEqual(
            {(row["stage"], row["method"]) for row in rows},
            {
                (stage, method)
                for stage in ("early", "late")
                for method in ("down_diag", "down_none", "newton_full", "muon")
            },
        )

    def test_checkpoint_paths_are_frozen_and_unique(self):
        paths = [row["path"] for row in CONTRACT["checkpoints"]]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(
            all(path.startswith("${SNM_RESULTS_ROOT}/") for path in paths)
        )

    def test_primary_contrasts_face_two_baselines(self):
        primary = set(CONTRACT["comparison_contract"]["primary"])
        self.assertEqual(
            primary,
            {
                "selective_diag_vs_muon",
                "selective_none_vs_muon",
                "selective_diag_vs_original_newton_muon",
                "selective_none_vs_original_newton_muon",
            },
        )
        self.assertNotIn("selective_diag_vs_selective_none", primary)

    def test_algorithm_mapping(self):
        algorithms = CONTRACT["algorithms"]
        self.assertEqual(
            algorithms["muon"], {"family_core": "none", "down": "none"}
        )
        self.assertEqual(
            algorithms["original_newton_muon"],
            {"family_core": "dense_full", "down": "dense_full"},
        )
        self.assertEqual(algorithms["selective_diag"]["down"], "diag")
        self.assertEqual(algorithms["selective_none"]["down"], "none")

    def test_analyzer_contract_excludes_proposal_vs_proposal(self):
        for _, left, right, _ in ANALYZER.CONTRASTS:
            self.assertNotEqual({left, right}, {"selective_diag", "selective_none"})

    def test_worker_never_calls_optimizer_step(self):
        tree = ast.parse((HERE / "mech07_worker.py").read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
        ]
        self.assertEqual(calls, [])

    def test_no_new_training_or_hvp(self):
        interpretation = CONTRACT["interpretation"]
        self.assertIs(interpretation["new_training"], False)
        self.assertIs(interpretation["hvp_authorized"], False)

    def test_smoke_offsets_expand_to_disjoint_a_b_windows(self):
        offsets = RUNNER.expanded_batch_offsets(CONTRACT, "smoke")
        self.assertEqual(offsets, [0, 129])

    def test_formal_offsets_cover_all_repeat_split_batches(self):
        offsets = RUNNER.expanded_batch_offsets(CONTRACT, "formal")
        expected = (
            CONTRACT["formal"]["repeats"]
            * 2
            * CONTRACT["formal"]["batches_per_split"]
        )
        self.assertEqual(len(offsets), expected)
        self.assertEqual(len(offsets), len(set(offsets)))
        self.assertEqual(offsets[:8], [index * 129 for index in range(8)])

    def test_muon_ambiguity_requires_optimizer_state_audit(self):
        source = (HERE / "mech07_worker.py").read_text(encoding="utf-8")
        self.assertIn("def method_identity_audit(", source)
        self.assertIn('entry.get("momentum")', source)
        self.assertIn('"exp_avg" not in state_keys', source)
        self.assertIn('"exp_avg_sq" not in state_keys', source)

    def test_resume_accepts_only_known_computation_compatible_versions(self):
        self.assertEqual(
            RUNNER.COMPATIBLE_WORKER_VERSIONS,
            {"2026-07-27.2", "2026-07-27.3"},
        )


if __name__ == "__main__":
    unittest.main()
