#!/usr/bin/env python3
"""CPU-only regression tests for the frozen MDP-05 framework."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402
import run_mdp05 as RUNNER  # noqa: E402


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "mdp05_test_analyzer", HERE / "analyze_mdp05.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract_path = HERE / "mdp05_contract.json"
        cls.contract = P.read_json(cls.contract_path)
        cls.base_path = (
            SCRIPTS
            / "37_mech09_downproj_refresh_mediation"
            / "refresh_mediation_repair_contract.json"
        )
        cls.base = P.read_json(cls.base_path)

    def test_frozen_protocol(self) -> None:
        self.assertTrue(all(P.validate_protocol(self.contract).values()))
        self.assertFalse(
            self.contract["hard_gates"][
                "runtime_resolvent_relative_residual_is_hard_gate"
            ]
        )
        self.assertFalse(
            self.contract["hard_gates"][
                "growing_worker_log_in_scientific_hashes"
            ]
        )

    def test_ns5_gate_is_at_full_optimizer_step_boundary(self) -> None:
        source = (HERE / "mdp05_worker.py").read_text(encoding="utf-8")
        patched_apply = source.split("def patched_apply", 1)[1].split(
            "def patched_ns", 1
        )[0]
        step_boundary = source.split("def step_with_mdp05_boundary", 1)[1].split(
            "patched_step =", 1
        )[0]
        self.assertNotIn("finish_event_after_optimizer_step", patched_apply)
        self.assertIn("finish_event_after_optimizer_step", step_boundary)
        self.assertLess(
            step_boundary.index("result = original_step"),
            step_boundary.index("finish_event_after_optimizer_step"),
        )

    def test_known_pre_outcome_failure_seals_repair_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            run = root / "run"
            contract_relative = (
                "scripts/46_mdp05_confirmatory_update_shock/mdp05_contract.json"
            )
            worker_relative = (
                "scripts/46_mdp05_confirmatory_update_shock/mdp05_worker.py"
            )
            controller_relative = (
                "scripts/46_mdp05_confirmatory_update_shock/run_mdp05.py"
            )
            for relative, contents in (
                (contract_relative, "frozen-contract\n"),
                (worker_relative, "premature-gate\n"),
                (controller_relative, "controller-v1\n"),
            ):
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            old_repo = RUNNER.REPO
            old_sources = RUNNER.SOURCE_FILES
            try:
                RUNNER.REPO = repo
                RUNNER.SOURCE_FILES = (
                    contract_relative,
                    worker_relative,
                    controller_relative,
                )
                run.mkdir()
                RUNNER.write_source_snapshot(run / "source_snapshot")
                P.atomic_json(run / "run_identity.json", {"experiment": "MDP-05"})
                status = (
                    run
                    / "formal/early_muon/replica_3/attempt_001/mdp05_status.json"
                )
                status.parent.mkdir(parents=True)
                P.atomic_json(
                    status,
                    {
                        "error": (
                            "RuntimeError: wrong actual NS5 call count for "
                            "production_refresh_32"
                        )
                    },
                )
                (repo / worker_relative).write_text(
                    "full-step-boundary-gate\n", encoding="utf-8"
                )
                snapshot, manifest, repair = RUNNER.snapshot_sources(run)
            finally:
                RUNNER.REPO = old_repo
                RUNNER.SOURCE_FILES = old_sources
            self.assertEqual(snapshot.name, RUNNER.REPAIR_SNAPSHOT_NAME)
            self.assertTrue(manifest.is_file())
            self.assertIsNotNone(repair)
            assert repair is not None
            self.assertTrue(all(repair["repair_checks"].values()))
            self.assertEqual(repair["failure_evidence"], [status.relative_to(run).as_posix()])

    def test_execution_derivation_is_new_and_does_not_mutate_source(self) -> None:
        before = copy.deepcopy(self.base)
        derived = P.derive_execution_contract(
            self.base, self.contract, P.sha256_file(self.contract_path)
        )
        self.assertEqual(self.base, before)
        self.assertEqual(derived["formal"]["data_replicas"], [3, 4, 5])
        self.assertEqual(
            derived["formal"]["replica_optimizer_step_offsets"],
            [768, 1024, 1280],
        )
        self.assertEqual(derived["formal"]["rollout_steps"], 80)
        self.assertEqual(derived["formal"]["evaluation_steps"], [0, 16, 32, 48, 64, 80])
        self.assertEqual(
            derived["arms"]["production_newton_muon"][
                "formal_down_refresh_completed_steps"
            ],
            [32, 64],
        )
        self.assertEqual(
            derived["arms"]["delayed_down_refresh"][
                "formal_down_refresh_completed_steps"
            ],
            [64],
        )
        self.assertEqual(derived["stopping_rule"]["maximum_new_formal_jobs"], 12)
        self.assertEqual(derived["stopping_rule"]["maximum_trajectories"], 36)

    def test_offset_certificate_has_no_old_or_new_collision(self) -> None:
        certificate = P.build_offset_certificate(self.contract)
        self.assertTrue(certificate["passed"])
        self.assertEqual(certificate["training_collisions"], [])
        self.assertEqual(certificate["validation_collisions"], [])
        labels = {row["label"] for row in certificate["training_intervals"]}
        self.assertIn("source_training_0", labels)
        self.assertIn("mdp05_formal_training_3", labels)

    def test_fineweb_mapping_detects_wrap_and_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shard.bin"
            header = (20240520).to_bytes(4, "little", signed=True)
            header += (1).to_bytes(4, "little", signed=True)
            header += (200).to_bytes(4, "little", signed=True)
            path.write_bytes(header)
            self.assertEqual(P.peek_fineweb_tokens(path), 200)
            mapped = P.map_global_interval([path], [200], 20, 30)
            self.assertTrue(mapped["single_shard"])
            wrapped = P.map_global_interval([path], [200], 220, 30)
            self.assertEqual(wrapped["wrapped_start"], 20)

    def test_exact_randomization_and_holm(self) -> None:
        rows = []
        for origin_index, origin in enumerate(P.ORIGINS):
            for replica in (3, 4, 5):
                value = origin_index * 10.0 + replica
                rows.append(
                    {
                        "origin": origin,
                        "data_replica": replica,
                        "x": value,
                        "y": value,
                    }
                )
        result = P.exact_within_origin_randomization_p(rows, "x", "y")
        self.assertEqual(result["permutations"], 1296)
        self.assertAlmostEqual(result["observed_spearman_rho"], 1.0)
        adjusted = P.holm_adjust([0.01, 0.02, 0.03, 0.04])
        self.assertEqual(len(adjusted), 4)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))

    def test_outcome_orientation_and_event_auc(self) -> None:
        analyzer = load_analyzer()
        evaluations = {
            ("production_newton_muon", 32): 1.02,
            ("production_newton_muon", 48): 1.03,
            ("frozen_down_refresh", 32): 1.00,
            ("frozen_down_refresh", 48): 1.00,
        }
        event = self.contract["event_outcomes"][0]
        outcome = analyzer.outcome_for_event(evaluations, event)
        self.assertAlmostEqual(outcome["oriented_endpoint_loss_harm"], 0.03)
        self.assertAlmostEqual(outcome["oriented_auc_harm"], 0.025)

    def test_previous_failure_classes_are_explicitly_closed(self) -> None:
        controller = (HERE / "run_mdp05.py").read_text(encoding="utf-8")
        worker = (HERE / "mdp05_worker.py").read_text(encoding="utf-8")
        self.assertIn("runtime_preflight(\n        args.child_python,", controller)
        self.assertNotIn("runtime_preflight(\n        args.child_python.resolve()", controller)
        self.assertIn("sealed_after_worker_exit", controller)
        self.assertIn("worker.log\" not in manifest", controller)
        self.assertNotIn("accepted_unit", worker)
        self.assertNotIn("unsupported stream contract", worker)
        self.assertIn("runtime_resolvent_relative_residual_is_hard_gate", worker)

    def test_controller_sealed_dry_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source_run"
            (source / "formal").mkdir(parents=True)
            (source / "analysis").mkdir()
            P.atomic_json(
                source / "formal" / "formal_manifest.json",
                {"passed": True, "completed_jobs": 12},
            )
            P.atomic_json(
                source / "analysis" / "mech09r_analysis_manifest.json",
                {"passed": True, "hypothesis_classification": "full_support"},
            )
            shared = root / "shared_input"
            shared.mkdir()
            checkpoint = shared / "checkpoint.pt"
            certificate = shared / "certificate.json"
            source_script = shared / "source.py"
            profile_script = shared / "profile.py"
            triton = shared / "triton.py"
            execution_contract = shared / "contract.json"
            reference = shared / "reference.json"
            train = shared / "train.bin"
            for path in (
                checkpoint,
                certificate,
                source_script,
                profile_script,
                triton,
                execution_contract,
                reference,
                train,
            ):
                path.write_bytes(b"fixture")
            val = shared / "fineweb_val_000000.bin"
            val.write_bytes(
                (20240520).to_bytes(4, "little", signed=True)
                + (1).to_bytes(4, "little", signed=True)
                + (20_000_000).to_bytes(4, "little", signed=True)
            )
            command_rows = []
            for origin in P.ORIGINS:
                worker_args = [
                    "--output-dir",
                    str(shared / "unused"),
                    "--analysis-tier",
                    "formal",
                    "--cell",
                    origin,
                    "--data-replica",
                    "0",
                    "--checkpoint",
                    str(checkpoint),
                    "--checkpoint-hash-certificate",
                    str(certificate),
                    "--source-script",
                    str(source_script),
                    "--profile-script",
                    str(profile_script),
                    "--triton-kernels",
                    str(triton),
                    "--contract",
                    str(execution_contract),
                    "--mech08-control-reference",
                    str(reference),
                    "--train-data-pattern",
                    str(train),
                    "--val-data-pattern",
                    str(shared / "fineweb_val_*.bin"),
                    "--host-id",
                    "fixture",
                    "--execution-domain",
                    "fixture",
                ]
                command_rows.append(
                    {
                        "label": f"formal/{origin}/replica_0",
                        "command": ["/fake/python", "/fake/worker", *worker_args],
                    }
                )
            (source / "commands.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in command_rows),
                encoding="utf-8",
            )
            for origin_index, origin in enumerate(P.ORIGINS):
                for replica in (0, 1, 2):
                    unit = source / "formal" / origin / f"replica_{replica}"
                    unit.mkdir(parents=True)
                    P.atomic_json(
                        unit / "training_stream_contract.json",
                        {
                            "first_x_sha256": f"train-x-{origin_index}-{replica}",
                            "first_y_sha256": f"train-y-{origin_index}-{replica}",
                        },
                    )
                    batches = [
                        {
                            "x_sha256": f"val-x-{origin_index}-{replica}-{index}",
                            "y_sha256": f"val-y-{origin_index}-{replica}-{index}",
                        }
                        for index in range(2)
                    ]
                    P.atomic_json(
                        unit / "heldout_batch_contract.json",
                        {"build": {"hashes": batches[:1]}, "evaluation": {"hashes": batches[1:]}},
                    )
            run_dir = root / "dry_run"
            args = SimpleNamespace(
                run_dir=run_dir,
                source_run=source,
                child_python=Path(sys.executable),
                gpus=["0", "1"],
                max_parallel=2,
                pilot_certificate=None,
                dry_run=True,
                resume=False,
            )
            self.assertEqual(RUNNER.controller(args), 0)
            self.assertEqual(
                P.read_json(run_dir / "status.json")["status"],
                "dry_run_passed",
            )
            self.assertTrue(
                P.read_json(
                    run_dir / "sealed" / "offset_collision_certificate.json"
                )["passed"]
            )
            self.assertEqual(
                P.read_json(run_dir / "sealed" / "formal_job_plan.json")[
                    "formal_units"
                ],
                12,
            )

    def test_complete_analyzer_fixture(self) -> None:
        analyzer = load_analyzer()

        def write_table(path: Path, rows: list[dict[str, object]]) -> None:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "formal_run"
            sealed = run / "sealed"
            sealed.mkdir(parents=True)
            contract_sha = P.sha256_file(self.contract_path)
            execution = sealed / "derived_execution_contract.json"
            P.atomic_json(execution, {"fixture": True})
            execution_sha = P.sha256_file(execution)
            P.atomic_json(
                run / "run_identity.json",
                {"experiment": "MDP-05", "contract_sha256": contract_sha},
            )
            P.atomic_json(
                sealed / "offset_collision_certificate.json", {"passed": True}
            )
            P.atomic_json(
                sealed / "formal_job_plan.json",
                {"passed": True, "formal_units": 12},
            )
            P.atomic_json(
                sealed / "source_data_reference.json",
                {
                    "training_first_batch_hashes": ["old-train"],
                    "validation_batch_hashes": ["old-val"],
                },
            )
            calibration_origin = self.contract["precision_calibration"]["origin"]
            calibration_replica = self.contract["precision_calibration"]["replica"]
            for origin_index, origin in enumerate(P.ORIGINS):
                for replica_index, replica in enumerate((3, 4, 5)):
                    unit = run / "formal" / origin / f"replica_{replica}"
                    attempt = unit / "attempt_001"
                    slices = attempt / "validation_slices"
                    slices.mkdir(parents=True)
                    harm32 = 0.002 + origin_index * 0.0005 + replica_index * 0.0002
                    harm64 = 0.001 + origin_index * 0.0004 + replica_index * 0.00015
                    evaluations = []
                    for arm in (
                        "production_newton_muon",
                        "delayed_down_refresh",
                        "frozen_down_refresh",
                    ):
                        for step in (0, 16, 32, 48, 64, 80):
                            value = 1.0 - step * 0.0001
                            if arm == "production_newton_muon" and step >= 32:
                                value += harm32 * (0.5 if step == 32 else 1.0)
                            if arm == "delayed_down_refresh" and step >= 64:
                                value += harm64 * (0.5 if step == 64 else 1.0)
                            evaluations.append(
                                {
                                    "checkpoint_cell": origin,
                                    "data_replica": replica,
                                    "arm": arm,
                                    "optimizer_step": step,
                                    "normalized_loss": value,
                                }
                            )
                    write_table(attempt / "evaluation.csv", evaluations)
                    layer_rows = []
                    for event_index, event in enumerate(P.EVENTS):
                        shock = (harm32 if event_index == 0 else harm64) * 10.0
                        for layer in range(18):
                            before = 10.0 + layer
                            delta = before * (shock + layer * 1.0e-5)
                            layer_rows.append(
                                {
                                    "origin": origin,
                                    "data_replica": replica,
                                    "event_id": event,
                                    "layer_index": layer,
                                    "relative_k_fro_change": shock * 0.5,
                                    "relative_a_fro_change": shock * 0.6,
                                    "relative_runtime_inverse_fro_change": shock * 0.7,
                                    "condition_proxy_after": 2.0 + shock,
                                    "runtime_resolvent_relative_residual": 0.02 if layer == 3 else 0.005,
                                    "matched_g_preconditioned_relative_change": delta / before,
                                    "runtime_ns5_update_relative_change": delta / before * 1.1,
                                    "matched_g_preconditioned_fro_before": before,
                                    "matched_g_preconditioned_fro_after": before + delta,
                                    "matched_g_preconditioned_delta_fro": delta,
                                    "matched_g_preconditioned_cosine": 0.99,
                                    "runtime_ns5_update_fro_before": before,
                                    "runtime_ns5_update_fro_after": before + delta * 1.1,
                                    "runtime_ns5_update_delta_fro": delta * 1.1,
                                    "runtime_ns5_update_cosine": 0.98,
                                    "all_full_state_values_finite": True,
                                }
                            )
                    write_table(
                        attempt / "mdp05_refresh_layer_metrics.csv", layer_rows
                    )
                    P.atomic_json(
                        attempt / "training_stream_contract.json",
                        {
                            "first_x_sha256": f"new-train-x-{origin_index}-{replica}",
                            "first_y_sha256": f"new-train-y-{origin_index}-{replica}",
                        },
                    )
                    P.atomic_json(
                        attempt / "heldout_batch_contract.json",
                        {
                            "build": {
                                "hashes": [
                                    {
                                        "x_sha256": f"new-build-x-{replica}",
                                        "y_sha256": f"new-build-y-{replica}",
                                    }
                                ]
                            },
                            "evaluation": {
                                "hashes": [
                                    {
                                        "x_sha256": f"new-eval-x-{replica}",
                                        "y_sha256": f"new-eval-y-{replica}",
                                    }
                                ]
                            },
                        },
                    )
                    if origin == calibration_origin and replica == calibration_replica:
                        for event in P.EVENTS:
                            for layer in (0, 3, 8, 17):
                                P.atomic_json(
                                    slices / f"fixture_{event}_layer{layer}_float64.json",
                                    {
                                        "origin": origin,
                                        "data_replica": replica,
                                        "event_id": event,
                                        "layer_index": layer,
                                        "all_values_finite": True,
                                    },
                                )
                    (attempt / "worker.log").write_text("done\n", encoding="utf-8")
                    P.atomic_json(attempt / "mdp05_status.json", {"status": "passed"})
                    scientific = [
                        "evaluation.csv",
                        "mdp05_refresh_layer_metrics.csv",
                        "training_stream_contract.json",
                        "heldout_batch_contract.json",
                    ]
                    scientific.extend(
                        path.relative_to(attempt).as_posix()
                        for path in slices.glob("*_float64.json")
                    )
                    manifest_path = attempt / "mdp05_unit_manifest.json"
                    P.atomic_json(
                        manifest_path,
                        {
                            "passed": True,
                            "origin": origin,
                            "data_replica": replica,
                            "mdp05_contract_sha256": contract_sha,
                            "execution_contract_sha256": execution_sha,
                            "layer_event_rows": 36,
                            "source_experiment_outcomes_read": False,
                            "scientific_artifact_sha256": {
                                name: P.sha256_file(attempt / name)
                                for name in scientific
                            },
                        },
                    )
                    P.atomic_json(
                        attempt / "worker_log_seal.json",
                        {
                            "sealed_after_worker_exit": True,
                            "sha256": P.sha256_file(attempt / "worker.log"),
                        },
                    )
                    P.atomic_json(
                        unit / "unit_selection.json",
                        {
                            "passed": True,
                            "selected_attempt": "attempt_001",
                            "manifest_sha256": P.sha256_file(manifest_path),
                        },
                    )
            manifest = analyzer.analyze(run, self.contract_path)
            self.assertTrue(manifest["integrity_passed"])
            self.assertEqual(manifest["scientific_result"], "confirmatory_success")
            self.assertEqual(len(manifest["primary_tests"]), 4)
            self.assertFalse(
                manifest["resolvent_diagnostics"]["hard_gate"]
            )


if __name__ == "__main__":
    unittest.main()
