#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import run_matched_diag_suite as SUITE
import run_matched_diag as WORKER


class MatchedDiagSuiteTests(unittest.TestCase):
    def test_frozen_unit_counts_and_seeds(self) -> None:
        self.assertEqual(SUITE.PILOT_SEED, 2053)
        self.assertEqual(SUITE.FORMAL_SEEDS, (2024, 2025, 2026))
        self.assertEqual(len(SUITE.ARMS), 5)
        self.assertEqual(len(SUITE.ARMS) * len(SUITE.FORMAL_SEEDS), 15)

    def test_resume_is_explicit_stage(self) -> None:
        self.assertIn("resume", SUITE.STAGES)

    def test_accepted_batch_checks_arm_seed_and_formality(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "batch"
            root.mkdir(parents=True)
            manifest = root / "r1_manifest.json"
            summary = {
                "method": "c_fc_diag",
                "controlled_seed": 2053,
                "init_sha256": "a" * 64,
                "derived_script_sha256": "b" * 64,
                "evidence_valid": True,
                "formal_evidence": False,
                "quality_usable": False,
                "memory_usable": True,
                "timing_usable": False,
                "outcome_eligible": False,
                "configuration_selection_allowed": False,
                "run_name": "run",
            }
            run = root / "run"
            run.mkdir()
            (run / "r1_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run / "run_manifest.json").write_text(
                json.dumps({"status": "completed_valid_smoke"}), encoding="utf-8"
            )
            (run / "r1_metrics.csv").write_text("event,step\n", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "family": "53_r1_matched_diag_module_placement",
                        "protocol": "r1_matched_diag_module_placement_engineering_pilot",
                        "batch_kind": "smoke",
                        "status": "completed_valid_smoke",
                        "official_commit": SUITE.OFFICIAL_COMMIT,
                        "methods": ["c_fc_diag"],
                        "seed": 2053,
                        "failures": [],
                        "formal_evidence": False,
                        "evidence_profile": "exact_shape_numerical_smoke",
                        "smoke_steps": 34,
                        "resource_isolation": {
                            "one_process_one_gpu": True,
                            "visible_device_count": 1,
                        },
                        "initialization_audit": {
                            "seed": 2053,
                            "all_methods_identical": True,
                            "init_sha256": "a" * 64,
                        },
                        "derived_source_sha256": {"c_fc_diag": "b" * 64},
                        "summaries": [summary],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                SUITE.accepted_batch(Path(temp), pilot=True, arm="c_fc_diag", seed=2053),
                manifest,
            )
            self.assertIsNone(
                SUITE.accepted_batch(Path(temp), pilot=True, arm="o_proj_diag", seed=2053)
            )

    def test_formal_command_uses_independent_pilot_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(
                run_dir=root / "run",
                official_repo=root / "official",
                python_exe="training-python",
                wandb_mode="disabled",
                wandb_project="project",
                wandb_entity=None,
            )
            pilot = root / "pilot/r1_manifest.json"
            with mock.patch.object(SUITE, "frozen_paths", return_value={"worker": root / "worker.py"}), mock.patch.object(
                SUITE, "resumable_batch", return_value=None
            ):
                command = SUITE.worker_command(
                    args,
                    arm="c_proj_diag",
                    seed=2024,
                    stage="formal",
                    pilot_manifest=pilot,
                )
            self.assertIn("2024", command)
            self.assertIn("c_proj_diag", command)
            self.assertIn(str(pilot), command)
            self.assertNotIn("2053", command)

    def test_pilot_command_is_34_steps_and_wandb_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(
                run_dir=root / "run",
                official_repo=root / "official",
                python_exe="training-python",
                wandb_mode="online",
                wandb_project="project",
                wandb_entity=None,
            )
            with mock.patch.object(SUITE, "frozen_paths", return_value={"worker": root / "worker.py"}), mock.patch.object(
                SUITE, "resumable_batch", return_value=None
            ):
                command = SUITE.worker_command(
                    args, arm="all_none", seed=2053, stage="pilot"
                )
            joined = " ".join(command)
            self.assertIn("--smoke-steps 34", joined)
            self.assertIn("--wandb-mode disabled", joined)

    def test_command_wrapper_exposes_all_stages_and_r0_path(self) -> None:
        wrapper = (REPO / "commands/53_r1_matched_diag_module_placement/20260817_ex53_r1_matched_diag_module_placement.sh").read_text(
            encoding="utf-8"
        )
        for stage in SUITE.STAGES:
            self.assertIn(stage, wrapper)
        self.assertIn("Newton-Muon-official-r0", wrapper)
        self.assertIn("EX53_GPUS:-0", wrapper)

    def test_contract_freezes_local_primary_and_timing_ineligible(self) -> None:
        contract = json.loads((SCRIPT_DIR / "matched_diag_contract.json").read_text(encoding="utf-8"))
        policy = contract["execution_policy"]
        self.assertTrue(policy["local_evidence_primary"])
        self.assertTrue(policy["wandb_secondary"])
        self.assertFalse(policy["timing_usable"])
        self.assertEqual(policy["formal_units"], 15)
        self.assertEqual(policy["physical_gpus"], ["0"])
        self.assertEqual(policy["maximum_concurrent_training_processes"], 1)

    def test_secondary_wandb_readiness_cannot_block_local_training(self) -> None:
        with mock.patch.object(
            WORKER,
            "ORIGINAL_VALIDATE_WANDB_ONLINE_ACCESS",
            side_effect=RuntimeError("offline"),
        ):
            payload = WORKER.validate_secondary_wandb_access(True)
        self.assertFalse(payload["required"])
        self.assertEqual(payload["status"], "unavailable_secondary")
        self.assertFalse(payload["scientific_validity_dependency"])

    def test_engineering_pilot_is_outcome_ineligible(self) -> None:
        payload = WORKER.evidence_eligibility(
            argparse.Namespace(numerical_smoke=True)
        )
        self.assertFalse(payload["quality_usable"])
        self.assertFalse(payload["outcome_eligible"])
        self.assertFalse(payload["configuration_selection_allowed"])
        self.assertTrue(payload["memory_usable"])

    def test_data_inventory_detects_content_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            official = root / "official"
            data = official / "data/fineweb10B"
            data.mkdir(parents=True)
            for name in (*SUITE.REQUIRED_TRAIN_SHARDS, SUITE.REQUIRED_VAL_SHARD):
                (data / name).write_bytes(
                    SUITE.DATA_MAGIC.to_bytes(4, "little", signed=True)
                    + name.encode("utf-8")
                )
            args = argparse.Namespace(official_repo=official, run_dir=root / "run")
            args.run_dir.mkdir()
            payload = SUITE.create_data_inventory(args)
            self.assertEqual(len(payload["entries"]), 51)
            SUITE.verify_data_inventory(args, full_hash=True)
            target = data / SUITE.REQUIRED_TRAIN_SHARDS[0]
            raw = bytearray(target.read_bytes())
            raw[-1] ^= 1
            target.write_bytes(bytes(raw))
            with self.assertRaisesRegex(RuntimeError, "content drift"):
                SUITE.verify_data_inventory(args, full_hash=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
