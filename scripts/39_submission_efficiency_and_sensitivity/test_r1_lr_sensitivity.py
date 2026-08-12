#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def load(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WORKER = load("r1_lr_sensitivity_worker.py")
RUNNER = load("run_r1_lr_sensitivity.py")
ANALYZER = load("analyze_r1_lr_sensitivity.py")
VALIDATOR = load("validate_r1_lr_sensitivity.py")


class FakeSpec:
    def __init__(self, name, base, matrix, role):
        self.name = name
        self.base_script = "x.py"
        self.cproj_k_mode = name
        self.base_learning_rate = base
        self.matrix_learning_rate = matrix
        self.role = role


class SensitivityTests(unittest.TestCase):
    def test_multiplier_label(self):
        self.assertEqual(RUNNER.multiplier_label(0.8), "m0p8")
        self.assertEqual(RUNNER.multiplier_label(1.0), "m1")

    def test_contrast_priority(self):
        self.assertEqual(len(ANALYZER.CONTRASTS), 5)
        self.assertFalse(
            any("diag_vs_none" in contrast for contrast, *_ in ANALYZER.CONTRASTS)
        )

    def test_method_roles(self):
        self.assertEqual(
            set(ANALYZER.ROLE.values()),
            {
                "muon",
                "original_newton_muon",
                "selective_none",
                "selective_diag",
            },
        )

    def test_two_gpu_lanes_are_required(self):
        self.assertEqual(ANALYZER.EXPECTED_LANE_COUNT, 2)

    def test_manifest_acceptance_is_bound_to_protocol_seed_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "r1_manifest.json"
            value = {
                "status": "completed_valid",
                "family": RUNNER.FAMILY,
                "protocol": RUNNER.FORMAL_PROTOCOL,
                "methods": ["diag", "none"],
                "seed": 2026,
                "formal_evidence": False,
                "evidence_profile": "shared_recipe_lr_sensitivity_supporting",
                "wandb_complete": True,
                "summaries": [
                    {
                        "method": method,
                        "controlled_seed": 2026,
                        "final_val_step": 3000,
                        "base_learning_rate": RUNNER.BASE_LR[method],
                        "matrix_learning_rate": RUNNER.MATRIX_LR[method],
                    }
                    for method in ("diag", "none")
                ],
            }
            manifest.write_text(json.dumps(value), encoding="utf-8")
            self.assertTrue(
                RUNNER.valid_manifest(
                    manifest,
                    False,
                    ["diag", "none"],
                    seed=2026,
                    budget_steps=3000,
                    multiplier=1.0,
                )
            )
            value["seed"] = 2025
            manifest.write_text(json.dumps(value), encoding="utf-8")
            self.assertFalse(
                RUNNER.valid_manifest(
                    manifest,
                    False,
                    ["diag", "none"],
                    seed=2026,
                    budget_steps=3000,
                    multiplier=1.0,
                )
            )

    def test_new_invalid_directory_does_not_shadow_valid_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            cell = Path(directory)
            valid = cell / "20260729T000000+0000_formal_seed2026"
            invalid = cell / "20260729T010000+0000_formal_seed2026"
            valid.mkdir()
            invalid.mkdir()
            manifest = {
                "status": "completed_valid",
                "family": RUNNER.FAMILY,
                "protocol": RUNNER.FORMAL_PROTOCOL,
                "methods": ["diag", "none"],
                "seed": 2026,
                "formal_evidence": False,
                "evidence_profile": "shared_recipe_lr_sensitivity_supporting",
                "wandb_complete": True,
                "summaries": [
                    {
                        "method": method,
                        "controlled_seed": 2026,
                        "final_val_step": 3000,
                        "base_learning_rate": RUNNER.BASE_LR[method],
                        "matrix_learning_rate": RUNNER.MATRIX_LR[method],
                    }
                    for method in ("diag", "none")
                ],
            }
            (valid / "r1_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (invalid / "r1_plan.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                RUNNER.locate_valid_batch(
                    cell,
                    False,
                    ["diag", "none"],
                    seed=2026,
                    budget_steps=3000,
                    multiplier=1.0,
                ),
                valid,
            )

    def test_plan_without_manifest_uses_parent_resume_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "m1"
            batch = cell / "20260729T000000+0000_formal_seed2026"
            batch.mkdir(parents=True)
            (batch / "r1_plan.json").write_text("{}", encoding="utf-8")
            args = Namespace(
                repo=root,
                official_repo=root,
                training_python="train-python",
                methods=["diag", "none"],
                budget_steps=3000,
                warmdown_steps=871,
                seed=2026,
                wandb_project="project",
                wandb_entity=None,
            )
            with mock.patch.object(
                RUNNER.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=1),
            ) as run:
                with self.assertRaisesRegex(RuntimeError, "formal failed"):
                    RUNNER.run_phase(
                        args,
                        root / "worker.py",
                        cell,
                        1.0,
                        smoke=False,
                        smoke_manifest=root / "smoke.json",
                    )
            command = run.call_args.args[0]
            self.assertIn("--resume-batch", command)
            self.assertIn(str(batch), command)

    def test_analysis_reuse_requires_all_output_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "analysis"
            output.mkdir()
            contract = root / "contract.json"
            contract.write_text(
                json.dumps(
                    {"seed": 2026, "budget_steps": 3000, "warmdown_steps": 871}
                ),
                encoding="utf-8",
            )
            hashes = {}
            for name in VALIDATOR.EXPECTED_ARTIFACTS:
                path = output / name
                path.write_text(f"{name}\n", encoding="utf-8")
                hashes[name] = VALIDATOR.sha256_file(path)
            manifest = {
                "passed": True,
                "script_version": VALIDATOR.EXPECTED_ANALYSIS_VERSION,
                "evidence_class": "supporting_only",
                "tuned_best_claim_allowed": False,
                "diag_vs_none_primary": False,
                "methods": sorted(VALIDATOR.EXPECTED_ROLES),
                "multipliers": sorted(VALIDATOR.EXPECTED_MULTIPLIERS),
                "seed": 2026,
                "budget_steps": 3000,
                "warmdown_steps": 871,
                "run_cells": 12,
                "contract_sha256": VALIDATOR.sha256_file(contract),
                "artifacts": sorted(VALIDATOR.EXPECTED_ARTIFACTS),
                "output_sha256": hashes,
            }
            (output / "lr_sensitivity_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            VALIDATOR.validate_output(output, contract)
            (output / "lr_sensitivity_runs.csv").write_text(
                "corrupt\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                VALIDATOR.validate_output(output, contract)


if __name__ == "__main__":
    unittest.main()
