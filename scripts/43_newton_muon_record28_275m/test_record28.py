#!/usr/bin/env python3
"""CPU-only contract tests for experiment 43."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path

import record28_common as C
import record28_source_builder as B
import run_record28_cell as W
import run_record28_suite as S


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parents[1]
if os.environ.get("RECORD28_OFFICIAL_REPO"):
    OFFICIAL = Path(os.environ["RECORD28_OFFICIAL_REPO"]).expanduser().resolve()
elif os.environ.get("SNM_OFFICIAL_REPO"):
    OFFICIAL = Path(os.environ["SNM_OFFICIAL_REPO"]).expanduser().resolve()
else:
    candidates = (
        ARTIFACT_ROOT / "third_party" / "Newton-Muon-official-r0",
        ARTIFACT_ROOT / "third_party" / "Newton-Muon-official",
    )
    OFFICIAL = next((path for path in candidates if path.is_dir()), candidates[0])


class Record28ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not OFFICIAL.is_dir():
            raise unittest.SkipTest(f"local pinned upstream repo is absent: {OFFICIAL}")
        cls.sources = B.build_all_sources(OFFICIAL)

    def test_upstream_source_hashes_and_two_derived_programs(self) -> None:
        self.assertEqual(set(self.sources), set(C.METHODS))
        self.assertEqual(
            self.sources["muon"].base_canonical_sha256,
            C.EXPECTED_CANONICAL_SHA256["train_gpt_muon_2.py"],
        )
        for method in C.METHODS[1:]:
            self.assertEqual(
                self.sources[method].base_canonical_sha256,
                C.EXPECTED_CANONICAL_SHA256["train_gpt_newton_muon_2.py"],
            )
        hashes = {item.derived_sha256 for item in self.sources.values()}
        self.assertEqual(len(hashes), 2)
        self.assertEqual(
            len(
                {
                    self.sources[method].derived_sha256
                    for method in C.METHODS[1:]
                }
            ),
            1,
        )

    def test_pytorch28_compiled_autograd_remains_disabled(self) -> None:
        for item in self.sources.values():
            self.assertNotIn(
                "torch._dynamo.config.compiled_autograd = True",
                item.source,
            )
            self.assertIn("torch.compile", item.source)
            self.assertIn("flex_attention(", item.source)
        runtime_source = inspect.getsource(W.runtime_probe)
        self.assertIn("compiled_autograd_default_false", runtime_source)
        contract = json.loads(
            (SCRIPT_DIR / "record28_contract.json").read_text(encoding="utf-8")
        )
        gate = contract["runtime_contract"]["training_runtime_gate"]
        self.assertEqual(gate["torch"], "2.8.0+cu126")
        self.assertEqual(gate["triton"], "3.4.0")
        self.assertFalse(gate["fresh_process_compiled_autograd_default"])

    def test_smoke_uses_formal_schedule_prefix(self) -> None:
        for item in self.sources.values():
            self.assertEqual(
                item.source.count(
                    "step / RECORD28_SCHEDULE_ITERATIONS "
                    "# frozen formal schedule, including smoke prefix"
                ),
                2,
            )
            self.assertNotIn(
                "x = step / args.num_iterations # progress in training",
                item.source,
            )
            self.assertIn("RECORD28_SCHEDULE_ITERATIONS = 1695", item.source)

    def test_warmup_cache_is_released_before_peak_memory_reset(self) -> None:
        expected = """del train_loader, initial_state, initial_optimizer_states, record28_rng_state
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()"""
        for item in self.sources.values():
            self.assertIn(expected, item.source)

    def test_k_finiteness_excludes_uninitialized_workspace_slots(self) -> None:
        expected = """for tensor in (k_cov_tensors + k_inv_tensors)"""
        forbidden = (
            "for tensor in "
            "(k_cov_tensors + k_inv_tensors + k_workspace_tensors)"
        )
        for item in self.sources.values():
            self.assertIn(expected, item.source)
            self.assertNotIn(forbidden, item.source)

    def test_cproj_modes_and_state_absence_are_explicit(self) -> None:
        newton = self.sources["selective_diag"].source
        for anchor in (
            'RECORD28_CPROJ_K_MODE == "block4"',
            'RECORD28_CPROJ_K_MODE == "diag"',
            'RECORD28_CPROJ_K_MODE == "none"',
            'kind == "c_proj_diag"',
            "accum_xtx_diag4_v3",
            "self._record28_refresh_count += 1",
        ):
            self.assertIn(anchor, newton)
        none_branch = """elif RECORD28_CPROJ_K_MODE == "none":
                pass"""
        self.assertIn(none_branch, newton)
        self.assertIn("if world_size != 1:", newton)
        self.assertIn("k_tensors_all_finite", newton)

    def test_exact_cproj_state_schema_bytes(self) -> None:
        d, layers, fp32 = 768, 12, 4
        original = W.CPROJ_SCHEMA_EXPECTED["original_newton_muon"]
        diag = W.CPROJ_SCHEMA_EXPECTED["selective_diag"]
        self.assertEqual(original["cov_bytes"], layers * 4 * d * d * fp32)
        self.assertEqual(original["inv_bytes"], layers * 4 * d * d * fp32)
        self.assertEqual(
            original["workspace_bytes"],
            layers * (4 * d * d + d * 4 * d) * fp32,
        )
        self.assertEqual(
            original["activation_stat_bytes"],
            layers * (4 * d * d * fp32 + 4),
        )
        self.assertEqual(diag["cov_bytes"], layers * 4 * d * fp32)
        self.assertEqual(diag["inv_bytes"], layers * 4 * d * fp32)
        self.assertEqual(
            diag["workspace_bytes"], layers * d * 4 * d * fp32
        )
        self.assertEqual(
            diag["activation_stat_bytes"], layers * (4 * d * fp32 + 4)
        )
        for method in ("muon", "selective_none"):
            schema = W.CPROJ_SCHEMA_EXPECTED[method]
            self.assertTrue(
                all(
                    schema[field] == 0
                    for field in (
                        "cov_bytes",
                        "inv_bytes",
                        "workspace_bytes",
                        "activation_stat_bytes",
                        "activation_workspace_bytes",
                    )
                )
            )

    def test_warmup_restore_preserves_custom_optimizer_state(self) -> None:
        for item in self.sources.values():
            source = item.source
            self.assertIn(
                "record28_optimizer_sha256_before = "
                "_record28_optimizer_fingerprint(optimizers)",
                source,
            )
            self.assertIn(
                "record28_optimizer_sha256_after = "
                "_record28_optimizer_fingerprint(optimizers)",
                source,
            )
            self.assertIn(
                "optimizer.state[parameter] = state",
                source,
            )
            self.assertIn('"optimizer_matches_initial"', source)
            self.assertNotIn(
                "opt.load_state_dict(opt_state)",
                source,
            )

    def test_diag_reference_math(self) -> None:
        B.self_test_diag_math()

    def test_full_precision_validation_wins_over_display_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.log"
            exact = 3.278712345678
            path.write_text(
                "\n".join(
                    (
                        "step:1695/1695 val_loss:3.2787 train_time:1ms step_avg:0.00ms",
                        'RECORD28_VAL {"step":1695,"total_steps":1695,'
                        f'"val_loss":{exact},"train_time_ms":1,'
                        '"step_avg_ms":0.0,"tokens":666501120}',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            parsed = C.parse_training_log(path)
            self.assertEqual(len(parsed["validations"]), 1)
            self.assertEqual(parsed["validations"][0]["val_loss"], exact)

    def test_target_crossing_and_exact_budget(self) -> None:
        protocol = W.expected_protocol("formal")
        self.assertEqual(protocol["iterations"] * 393_216, 666_501_120)
        summary = W.compute_curve_summary(
            [
                {
                    "step": 0,
                    "val_loss": 3.4,
                    "train_time_ms": 0,
                },
                {
                    "step": 50,
                    "val_loss": 3.2,
                    "train_time_ms": 1,
                },
            ],
            {"iterations": 50, "train_tokens": 50 * 393_216},
        )
        self.assertAlmostEqual(summary["steps_to_target"], 25.0)
        self.assertEqual(
            summary["first_observed_step_at_or_below_target"], 50
        )

    def test_two_lane_assignment_is_balanced_before_launch(self) -> None:
        args = argparse.Namespace(gpus=["0", "1"])
        counts = {
            method: {"0": 0, "1": 0} for method in C.METHODS
        }
        for seed in C.SEEDS:
            for method in C.METHODS:
                counts[method][S.assigned_gpu(args, seed, method)] += 1
        self.assertTrue(
            all(per_method == {"0": 2, "1": 2} for per_method in counts.values())
        )

    def test_contract_is_internally_consistent(self) -> None:
        contract = json.loads(
            (SCRIPT_DIR / "record28_contract.json").read_text(encoding="utf-8")
        )
        recipe = contract["training_recipe"]
        self.assertEqual(
            recipe["formal_updates"] * recipe["tokens_per_update"],
            recipe["exact_training_tokens"],
        )
        self.assertEqual(len(recipe["validation_steps"]), 35)
        self.assertEqual(contract["paired_design"]["formal_cells"], 16)
        self.assertFalse(contract["claim_boundary"]["timing_usable"])

    def test_snapshot_dependency_list_is_complete(self) -> None:
        for name in S.SNAPSHOT_FILES:
            self.assertTrue((SCRIPT_DIR / name).is_file(), name)

    def test_snapshot_recovery_command_cannot_dirty_sealed_snapshot(self) -> None:
        command_script = S.recovery_command_path().read_text(encoding="utf-8")
        recovery_function = command_script.split(
            "recovery_command() {", maxsplit=1
        )[1]
        snapshot_branch = recovery_function.split(
            'if [[ -f "${SNAPSHOT_SUITE_SCRIPT}" '
            '&& -f "${SNAPSHOT_MANIFEST}" ]]; then',
            maxsplit=1,
        )[1].split("else", maxsplit=1)[0]
        self.assertIn(
            "recovery=(\n"
            "      env\n"
            "      PYTHONDONTWRITEBYTECODE=1\n"
            '      "${CTRL_PY}"',
            snapshot_branch,
        )
        self.assertIn('"${SNAPSHOT_SUITE_SCRIPT}"', snapshot_branch)
        self.assertIn("--snapshot-active", snapshot_branch)
        self.assertIn("--resume", snapshot_branch)

    def test_formal_disabled_wandb_is_never_marked_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            C.atomic_write_json(
                attempt / "scientific_manifest.json",
                {"passed": True},
            )
            args = argparse.Namespace(
                attempt_dir=attempt,
                stage="formal",
                wandb_mode="disabled",
            )
            self.assertFalse(W.upload_wandb(args))
            status = C.read_json(attempt / "wandb.json")
            self.assertFalse(status["complete"])
            self.assertTrue(status["required_for_paper_handoff"])

    def test_scientific_attempt_seals_exact_artifact_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            names = {
                "checks.json",
                "command.json",
                "metrics.csv",
                "runtime.json",
                "stdout.log",
                "summary.json",
                "training.log",
            }
            for name in names:
                (attempt / name).write_text(name, encoding="utf-8")
            hashes = {
                name: C.sha256_file(attempt / name) for name in names
            }
            C.atomic_write_json(attempt / "artifact_hashes.json", hashes)
            C.atomic_write_json(
                attempt / "scientific_manifest.json",
                {
                    "passed": True,
                    "status": "scientifically_complete",
                    "stage": "smoke",
                    "artifacts": sorted(names | {"artifact_hashes.json"}),
                    "artifact_hashes": hashes,
                },
            )
            S.validate_scientific_attempt(
                attempt, {"stage": "smoke"}, verify_checkpoint=False
            )
            C.atomic_write_json(
                attempt / "artifact_hashes.json", {"tampered": "0" * 64}
            )
            with self.assertRaisesRegex(
                RuntimeError, "scientific attempt integrity failed"
            ):
                S.validate_scientific_attempt(
                    attempt, {"stage": "smoke"}, verify_checkpoint=False
                )

    def test_analysis_precedes_network_upload_retry(self) -> None:
        source = inspect.getsource(S.main_snapshot)
        self.assertLess(
            source.index("analysis_manifest = run_analysis(args)"),
            source.index("pending_wandb = retry_pending_uploads(args)"),
        )

    def test_snapshot_controller_forwards_recovery_controls(self) -> None:
        args = argparse.Namespace(
            run_dir=Path("/tmp/record28-test-run"),
            live_repo=Path("/tmp/live"),
            official_repo=Path("/tmp/official"),
            data_repo_root=Path("/tmp/data"),
            training_python=Path("/tmp/train-python"),
            gpus=["0"],
            wandb_mode="online",
            wandb_project="project",
            wandb_entity=None,
            wandb_upload_timeout_seconds=37,
            resume=True,
            dry_run=False,
        )
        command = S.forwarded_arguments(args, snapshot_active=True)
        self.assertIn("--snapshot-active", command)
        self.assertIn("--resume", command)
        timeout_index = command.index("--wandb-upload-timeout-seconds")
        self.assertEqual(command[timeout_index + 1], "37")
        self.assertEqual(
            command[1],
            str(
                args.run_dir
                / "source_snapshot"
                / "controller"
                / "run_record28_suite.py"
            ),
        )


if __name__ == "__main__":
    unittest.main()
