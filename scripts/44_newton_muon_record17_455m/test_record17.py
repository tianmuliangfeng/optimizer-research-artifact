#!/usr/bin/env python3
"""CPU-only contract tests for experiment 44."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path

import record17_common as C
import record17_source_builder as B
import run_record17_cell as W
import run_record17_suite as S


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parents[1]
if os.environ.get("RECORD17_OFFICIAL_REPO"):
    OFFICIAL = Path(os.environ["RECORD17_OFFICIAL_REPO"]).expanduser().resolve()
elif os.environ.get("SNM_OFFICIAL_REPO"):
    OFFICIAL = Path(os.environ["SNM_OFFICIAL_REPO"]).expanduser().resolve()
else:
    candidates = (
        ARTIFACT_ROOT / "third_party" / "Newton-Muon-official-r0",
        ARTIFACT_ROOT / "third_party" / "Newton-Muon-official",
    )
    OFFICIAL = next((path for path in candidates if path.is_dir()), candidates[0])


class Record17ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not OFFICIAL.is_dir():
            raise unittest.SkipTest(f"local pinned upstream repo is absent: {OFFICIAL}")
        cls.sources = B.build_all_sources(OFFICIAL)

    def test_vendored_source_hash_and_one_unified_program(self) -> None:
        self.assertEqual(set(self.sources), set(C.METHODS))
        for method in C.METHODS:
            self.assertEqual(
                self.sources[method].base_canonical_sha256,
                C.RECORD17_UPSTREAM_CANONICAL_SHA256,
            )
            self.assertEqual(
                self.sources[method].base_script,
                "record17_train_gpt_medium.py",
            )
        hashes = {item.derived_sha256 for item in self.sources.values()}
        self.assertEqual(len(hashes), 1)
        self.assertEqual(len({item.source for item in self.sources.values()}), 1)

    def test_pytorch28_compiler_policy_is_uniform_and_keeps_model_compile(self) -> None:
        for item in self.sources.values():
            self.assertEqual(
                item.source.count(
                    "torch._dynamo.config.compiled_autograd = False"
                ),
                1,
            )
            self.assertNotIn(
                "torch._dynamo.config.compiled_autograd = True",
                item.source,
            )
            self.assertEqual(
                item.source.count(
                    "model: nn.Module = torch.compile(model, dynamic=False)"
                ),
                1,
            )
            self.assertIn("flex_attention(", item.source)
            self.assertIn('"compiled_autograd_enabled"', item.source)
            self.assertEqual(
                item.source.count("out_dtype=torch.float32"),
                2,
            )

    def test_smoke_uses_formal_schedule_prefix(self) -> None:
        for item in self.sources.values():
            self.assertEqual(
                item.source.count(
                    "step / RECORD17_SCHEDULE_ITERATIONS "
                    "# frozen formal schedule, including smoke prefix"
                ),
                2,
            )
            self.assertNotIn(
                "x = step / args.num_iterations # progress in training",
                item.source,
            )
            self.assertIn("RECORD17_SCHEDULE_ITERATIONS = 5960", item.source)
            self.assertIn("elif RECORD17_NUM_ITERATIONS < 27:", item.source)

    def test_warmup_fully_restores_and_releases_cache_before_peak_reset(self) -> None:
        expected = """    loss,
)
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()"""
        for item in self.sources.values():
            self.assertIn(expected, item.source)
            for anchor in (
                "RECORD17_WARMUP_UPDATES = 26",
                '"post_refresh_updates": warmup_steps - 24',
                '"optimizer_matches_initial"',
                '"rng_matches_initial"',
                '"model_matches_initial"',
                '"loader_recreated_from_start": True',
                '"warmup_newton_path_exercised"',
                '"peak_memory_scope": (',
                '"counted_run_after_warmup_reset_including_validation"',
                "record17_zero_activation_buffers(model)",
                "optimizer2.reset_precond_step()",
                "optimizer.state.clear()",
                "optimizer.state[parameter] = state",
                "Optimizer\n# load_state_dict may cast arbitrary floating state",
            ):
                self.assertIn(anchor, item.source)
            self.assertNotIn(
                "optimizer.load_state_dict(optimizer_state)", item.source
            )

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
            'RECORD17_CPROJ_K_MODE == "block4"',
            'RECORD17_CPROJ_K_MODE == "diag"',
            'RECORD17_CPROJ_K_MODE == "none"',
            'kind == "proj_w_diag"',
            "accum_xtx_diag4_v1",
            "self._record17_refresh_count += 1",
            '"fp32_precondition_contract_passed"',
        ):
            self.assertIn(anchor, newton)
        none_branch = """elif RECORD17_CPROJ_K_MODE == "none":
                pass"""
        self.assertIn(none_branch, newton)
        self.assertIn("if world_size != 1:", newton)
        self.assertIn("k_tensors_all_finite", newton)

    def test_exact_cproj_state_schema_bytes(self) -> None:
        d, layers, fp32 = 1024, 16, 4
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

    def test_diag_reference_math(self) -> None:
        B.self_test_diag_math()
        B.self_test_right_precondition_math()

    def test_global_batch_is_loaded_once_then_split_into_eight(self) -> None:
        source = self.sources["muon"].source
        for anchor in (
            '"loader_global_batch_tokens": (',
            '"loader_split_microbatches": RECORD17_GRAD_ACCUM_STEPS',
            '"global_batch_loaded_before_split": True',
            '"newton_k_statistics_scope": (',
            '"all_8_sequential_microbatches_per_counted_update_on_single_h100"',
            '"newton_k_statistics_tokens_per_counted_update": (',
            '"strict_hypothetical_8rank_owner_local_k_equivalence_claimed": False',
            "RECORD17_GRAD_ACCUM_STEPS = 8",
            "global_inputs.chunk(RECORD17_GRAD_ACCUM_STEPS)",
            "global_targets.chunk(RECORD17_GRAD_ACCUM_STEPS)",
        ):
            self.assertIn(anchor, source)
        self.assertEqual(
            source.count("global_inputs.chunk(RECORD17_GRAD_ACCUM_STEPS)"),
            2,
        )

    def test_data_path_is_absolute_and_independent_of_attempt_cwd(self) -> None:
        source = self.sources["muon"].source
        for anchor in (
            'RECORD17_DATA_PATH = os.path.abspath(os.environ["DATA_PATH"])',
            'RECORD17_DATA_PATH, "data", "fineweb10B", "fineweb_train_*.bin"',
            'RECORD17_DATA_PATH, "data", "fineweb10B", "fineweb_val_*.bin"',
            "if not os.path.isabs(filename_pattern):",
            "files = sorted(Path(path) for path in glob.glob(filename_pattern))",
            "no data shards matched absolute pattern",
            '"train_file_pattern": args.train_files',
            '"validation_file_pattern": args.val_files',
        ):
            self.assertIn(anchor, source)
        for forbidden in (
            'train_files = "data/fineweb10B/',
            'val_files = "data/fineweb10B/',
            "Path.cwd().glob(filename_pattern)",
        ):
            self.assertNotIn(forbidden, source)

    def test_fp32_inverse_and_precondition_application_are_explicit(self) -> None:
        source = self.sources["original_newton_muon"].source
        for anchor in (
            "grad_fp32 = raw_grad.float()",
            "if raw_grad.dtype != torch.bfloat16:",
            "if inv.dtype != torch.float32 or buf.dtype != torch.float32:",
            "state[\"precond_buf\"] = torch.empty(",
            "p.shape, device=p.device, dtype=torch.float32",
            '"raw_gradient_dtypes_seen_by_preconditioner"',
            '"preconditioned_gradient_dtypes"',
            '"fp32_precondition_application_count"',
            '"raw_gradients_cast_to_fp32"',
            '"fp32_precondition_contract_passed"',
        ):
            self.assertIn(anchor, source)
        self.assertNotIn('state["precond_buf"] = torch.empty_like(p)', source)

    def test_record17_weight_orientations_are_right_preconditioned(self) -> None:
        source = self.sources["original_newton_muon"].source
        for anchor in (
            "torch.bmm(grad_fp32, inv, out=buf)",
            "torch.mm(grad_fp32, inv, out=buf)",
            "# fc_w is (4d,d): activation covariance acts on the right.",
            "# proj_w is (d,4d): apply four independent dxd right blocks.",
        ):
            self.assertIn(anchor, source)

    def test_full_precision_validation_wins_over_display_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.log"
            exact = 3.278712345678
            path.write_text(
                "\n".join(
                    (
                        "step:1695/1695 val_loss:3.2787 train_time:1ms step_avg:0.00ms",
                        'RECORD17_VAL {"step":1695,"total_steps":1695,'
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
        self.assertEqual(
            protocol["iterations"] * 524_288, 3_124_756_480
        )
        summary = W.compute_curve_summary(
            [
                {
                    "step": 0,
                    "val_loss": 3.0,
                    "train_time_ms": 0,
                },
                {
                    "step": 50,
                    "val_loss": 2.9,
                    "train_time_ms": 1,
                },
            ],
            {"iterations": 50, "train_tokens": 50 * 524_288},
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
            all(
                sum(per_method.values()) == len(C.SEEDS)
                and abs(per_method["0"] - per_method["1"]) <= 1
                for per_method in counts.values()
            )
        )

    def test_contract_is_internally_consistent(self) -> None:
        contract = json.loads(
            (SCRIPT_DIR / "record17_contract.json").read_text(encoding="utf-8")
        )
        recipe = contract["training_recipe"]
        self.assertEqual(
            recipe["formal_updates"] * recipe["tokens_per_update"],
            recipe["exact_training_tokens"],
        )
        self.assertEqual(recipe["validation_row_count"], 49)
        self.assertEqual(contract["paired_design"]["formal_cells"], 12)
        self.assertFalse(contract["claim_boundary"]["timing_usable"])
        self.assertFalse(
            recipe["newton_k_recipe"][
                "strict_hypothetical_eight_rank_owner_local_equivalence_claimed"
            ]
        )

    def test_training_runtime_and_compiler_policy_are_pinned(self) -> None:
        self.assertEqual(
            W.EXPECTED_TRAINING_RUNTIME,
            {
                "python": "3.10.12",
                "torch": "2.8.0+cu126",
                "torch_cuda": "12.6",
                "triton": "3.4.0",
                "cuda_available": True,
                "visible_device_count": 1,
            },
        )
        runtime_source = inspect.getsource(W.runtime_probe)
        self.assertIn("compiled_autograd_default_false", runtime_source)
        contract = json.loads(
            (SCRIPT_DIR / "record17_contract.json").read_text(encoding="utf-8")
        )
        runtime_contract = contract["runtime_contract"]
        self.assertEqual(
            runtime_contract["training_python"],
            "${SNM_TRAINING_PYTHON}",
        )
        gate = runtime_contract["training_runtime_gate"]
        self.assertEqual(gate["torch"], "2.8.0+cu126")
        self.assertEqual(gate["triton"], "3.4.0")
        self.assertFalse(gate["generated_source_compiled_autograd_enabled"])
        self.assertTrue(gate["flex_attention_retained"])
        self.assertTrue(gate["torch_compile_model_retained"])

    def test_snapshot_dependency_list_is_complete(self) -> None:
        for name in S.SNAPSHOT_FILES:
            self.assertTrue((SCRIPT_DIR / name).is_file(), name)

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
            run_dir=Path("/tmp/record17-test-run"),
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
                / "run_record17_suite.py"
            ),
        )

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


if __name__ == "__main__":
    unittest.main()
