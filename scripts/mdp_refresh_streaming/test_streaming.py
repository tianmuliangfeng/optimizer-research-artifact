#!/usr/bin/env python3
"""CPU regression tests for the MDP-04 streaming framework."""

from __future__ import annotations

import importlib.util
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


METRICS = load("test_mdp04_stream_metrics", "stream_metrics.py")
VALIDATOR = load("test_mdp04_stream_validator", "validate_stream_replay.py")
CONTROLLER = load("test_mdp04_stream_controller", "run_stream_replay.py")
STREAM_WORKER = load("test_mdp04_stream_worker", "stream_worker.py")


class StreamingMetricTests(unittest.TestCase):
    def test_stable_seed_is_process_independent(self) -> None:
        self.assertEqual(
            METRICS.stable_seed(123, "origin", 2, "event", 9),
            METRICS.stable_seed(123, "origin", 2, "event", 9),
        )
        self.assertNotEqual(
            METRICS.stable_seed(123, "origin", 2, "event", 9),
            METRICS.stable_seed(123, "origin", 2, "event", 10),
        )

    def test_tensor_fingerprint_matches_legacy_schema(self) -> None:
        tensor = torch.arange(35, dtype=torch.float32).reshape(5, 7)
        fingerprint = METRICS.tensor_fingerprint(tensor)
        payload = {
            "shape": fingerprint["shape"],
            "dtype": fingerprint["dtype"],
            "indices": fingerprint["indices"],
            "values": fingerprint["values"],
        }
        import hashlib

        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.assertEqual(
            fingerprint["fingerprint_sha256"], hashlib.sha256(encoded).hexdigest()
        )

    def test_full_metric_chain_on_small_spd_state(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260803)
        width = 24
        rows = 10
        base = torch.randn(width, width, generator=generator)
        fresh_base = torch.randn(width, width, generator=generator)
        covariance_before = base @ base.T / width + 0.25 * torch.eye(width)
        fresh = fresh_base @ fresh_base.T / width + 0.5 * torch.eye(width)
        covariance_after = 0.95 * covariance_before + 0.05 * fresh

        def inverse(covariance: torch.Tensor) -> torch.Tensor:
            ridge = covariance.diagonal().mean() * 0.2 + 1.0e-8
            return torch.linalg.inv(covariance + ridge * torch.eye(width))

        inverse_before = inverse(covariance_before)
        inverse_after = inverse(covariance_after)
        gradient = torch.randn(rows, width, generator=generator)
        momentum = torch.randn(rows, width, generator=generator)

        def normalized_update(value: torch.Tensor, _: int) -> torch.Tensor:
            return value / torch.linalg.vector_norm(value).clamp_min(1.0e-7)

        metrics, fingerprints, slice_payload = METRICS.compute_layer_metrics(
            covariance_before=covariance_before,
            covariance_after=covariance_after,
            inverse_before=inverse_before,
            inverse_after=inverse_after,
            fresh_covariance=fresh,
            raw_gradient=gradient,
            historical_momentum=momentum,
            input_beta=0.95,
            ridge_scale=0.2,
            ridge_epsilon=1.0e-8,
            momentum_beta=0.95,
            ns_steps=5,
            ns_update=normalized_update,
            probe_count=8,
            probe_iterations=12,
            probe_seed=42,
            slice_coordinate_count=8,
            slice_gradient_row_count=6,
            slice_seed=17,
        )
        self.assertLess(
            metrics["covariance_refresh_identity_relative_residual"], 1.0e-6
        )
        self.assertLess(metrics["k_asymmetry_before"], 1.0e-6)
        self.assertLess(metrics["k_asymmetry_after"], 1.0e-6)
        self.assertLess(
            metrics["runtime_inverse_backward_residual_before"], 1.0e-5
        )
        self.assertLess(
            metrics["runtime_inverse_backward_residual_after"], 1.0e-5
        )
        self.assertLess(metrics["runtime_resolvent_relative_residual"], 1.0e-4)
        self.assertTrue(metrics["all_full_state_values_finite"])
        self.assertIn("gradient_after", fingerprints)
        self.assertIsNotNone(slice_payload)
        assert slice_payload is not None
        self.assertEqual(slice_payload["covariance_before"].shape, (8, 8))
        self.assertEqual(slice_payload["raw_gradient"].shape, (6, 8))


class ReplayAuditTests(unittest.TestCase):
    def test_cross_run_float_values_are_diagnostic_but_reference_is_gated(self) -> None:
        expected = {
            "shape": [3],
            "dtype": "torch.float32",
            "indices": [0, 1, 2],
            "values": [1.0, -0.25, 0.0],
            "fingerprint_sha256": "accepted",
        }
        observed = {
            **expected,
            "values": [1.00005, -0.25001, 0.0000005],
            "fingerprint_sha256": "fresh-process",
        }
        audit = STREAM_WORKER.compare_replay_fingerprint(
            expected, observed, rtol=1.0e-4, atol=1.0e-6
        )
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["reference_integrity_passed"])
        self.assertFalse(audit["fingerprint_sha256_exact"])
        failed = STREAM_WORKER.compare_replay_fingerprint(
            expected,
            {**observed, "values": [1.01, -0.25001, 0.0000005]},
            rtol=1.0e-4,
            atol=1.0e-6,
        )
        self.assertFalse(failed["passed"])
        self.assertTrue(failed["reference_integrity_passed"])
        malformed = STREAM_WORKER.compare_replay_fingerprint(
            expected,
            {**observed, "indices": [0, 2, 1]},
            rtol=1.0e-4,
            atol=1.0e-6,
        )
        self.assertFalse(malformed["reference_integrity_passed"])
        recorder = object.__new__(STREAM_WORKER.StreamRecorder)
        recorder.pending = {"spec": {"event_id": "e32"}, "rows": [{"module_id": "layer"}]}
        recorder.event_audits = []
        recorder.contract = {
            "cross_run_replay_audit": {
                "accepted_refresh_value_tolerance": {
                    "rtol": 1.0e-4,
                    "atol": 1.0e-6,
                }
            }
        }
        recorder.expected_refresh_event = lambda _spec: {
            "target_before": {"rows": [{"name": "layer", "covariance": expected, "inverse": expected}]},
            "target_after": {"rows": [{"name": "layer", "covariance": expected, "inverse": expected}]},
        }
        drifted = {**observed, "values": [1.01, -0.25001, 0.0000005]}
        snapshot = {
            "rows": [{"name": "layer", "covariance": drifted, "inverse": drifted}]
        }
        recorder.attach_accepted_fingerprint_checks(
            target_before=snapshot,
            target_after=snapshot,
        )
        self.assertTrue(
            recorder.pending["rows"][0][
                "accepted_covariance_after_reference_integrity_passed"
            ]
        )
        self.assertFalse(
            recorder.pending["rows"][0]["accepted_covariance_after_numeric_match"]
        )
        self.assertFalse(
            recorder.event_audits[0]["accepted_refresh_numeric_match_diagnostic"]
        )

    def test_branch_anchor_keeps_within_replay_sha_hard(self) -> None:
        fields = [
            "structure_sha256",
            "tensor_count",
            "sampled_values_finite",
            "next_x_sha256",
            "next_y_sha256",
            "loader_state",
            "matrix_global_step",
        ]
        expected = {
            "sha256": "accepted-process",
            "structure_sha256": "structure",
            "tensor_count": 697,
            "sampled_values_finite": True,
            "next_x_sha256": "x",
            "next_y_sha256": "y",
            "loader_state": {"current_shard": 5, "current_position": 10},
            "matrix_global_step": 31,
        }
        recorder = object.__new__(STREAM_WORKER.StreamRecorder)
        recorder.contract = {
            "cross_run_replay_audit": {
                "accepted_branch_hard_exact_fields": fields
            }
        }
        recorder.accepted_branch = {
            "branch_start_audits": [
                {"label": "production_starts_from_first_fork", "expected": expected}
            ]
        }
        recorder.branch_anchors = []
        recorder._bound_optimizer = None
        observed = {**expected, "sha256": "fresh-process"}
        recorder.record_branch_anchor(
            "production_starts_from_first_fork",
            observed,
            {"passed": True, "checks": {"sha256": True}},
        )
        self.assertTrue(recorder.branch_anchors[0]["passed"])
        self.assertFalse(
            recorder.branch_anchors[0]["accepted_sha256_exact_diagnostic"]
        )
        with self.assertRaisesRegex(RuntimeError, "branch anchor mismatch"):
            recorder.record_branch_anchor(
                "production_starts_from_first_fork",
                observed,
                {"passed": False, "checks": {"sha256": False}},
            )

    def test_optimizer_hooks_are_reversible_and_event_scoped(self) -> None:
        module_name = "test_mdp04_event_scoped_optimizer"
        source_module = types.ModuleType(module_name)

        def source_ns(value: torch.Tensor, steps: int = 5) -> torch.Tensor:
            return value + float(steps)

        def source_apply(self) -> None:
            self.apply_calls += 1

        optimizer_type = type(
            "FakeOptimizer",
            (),
            {
                "__module__": module_name,
                "__init__": lambda self: setattr(self, "apply_calls", 0),
                "_apply_preconditioners": source_apply,
            },
        )
        source_module.zeropower_via_newtonschulz5 = source_ns
        sys.modules[module_name] = source_module
        try:
            optimizer = optimizer_type()
            recorder = object.__new__(STREAM_WORKER.StreamRecorder)
            recorder.pending = None
            recorder.original_ns = None
            recorder._bound_optimizer = None
            recorder._source_module = None
            recorder._original_apply_had_instance_override = False
            recorder._original_apply_instance_value = None
            recorder._patched_apply_bound = None
            recorder._patched_ns = None
            recorder._bound_event_id = None
            recorder.hook_lifecycle = []
            recorder.bind_optimizer(optimizer, event_id="event32")
            self.assertIn("_apply_preconditioners", optimizer.__dict__)
            optimizer._apply_preconditioners()
            self.assertEqual(optimizer.apply_calls, 1)
            torch.testing.assert_close(
                source_module.zeropower_via_newtonschulz5(torch.tensor(1.0)),
                torch.tensor(6.0),
            )
            recorder.unbind_optimizer()
            self.assertNotIn("_apply_preconditioners", optimizer.__dict__)
            self.assertIs(source_module.zeropower_via_newtonschulz5, source_ns)
            self.assertEqual(
                recorder.hook_lifecycle,
                [
                    {"action": "bind", "event_id": "event32"},
                    {"action": "unbind", "event_id": "event32"},
                ],
            )
        finally:
            del sys.modules[module_name]

    def test_controller_initialization_does_not_bind_stream_hooks(self) -> None:
        previous = STREAM_WORKER.RECORDER
        recorder = mock.Mock()
        STREAM_WORKER.RECORDER = recorder
        try:
            with mock.patch.object(
                STREAM_WORKER.LEGACY.RefreshInterventionController,
                "__init__",
                return_value=None,
            ):
                STREAM_WORKER.InstrumentedController(object())
            recorder.bind_optimizer.assert_not_called()
        finally:
            STREAM_WORKER.RECORDER = previous


class ContractAndAggregationTests(unittest.TestCase):
    def test_contract_coverage_and_executable_path_are_exact(self) -> None:
        contract = json.loads(
            (HERE / "refresh_stream_contract.json").read_text(encoding="utf-8")
        )
        coverage = contract["coverage"]
        expected = (
            len(coverage["origins"])
            * len(coverage["replicas"])
            * len(coverage["events"])
            * len(coverage["layer_indices"])
        )
        self.assertEqual(expected, coverage["expected_layer_event_rows"])
        self.assertEqual(coverage["expected_unit_event_rows"], 24)
        self.assertEqual(contract["matrix_contract"]["gradient_shape"], [2048, 5504])
        self.assertNotIn(
            "independent_review/independent_audit_manifest.json",
            contract["accepted_source_artifact_sha256"],
        )
        self.assertFalse(
            contract["local_posthoc_audit_reference"]["remote_source_required"]
        )
        self.assertFalse(contract["local_posthoc_audit_reference"]["used_by_worker"])
        self.assertFalse(
            contract["local_posthoc_audit_reference"]["used_by_validator"]
        )
        pinned = contract["pinned_runtime_sources"]["triton_kernels"]
        pinned_path = HERE.parent.parent / pinned["relative_path"]
        self.assertEqual(pinned_path.name, "triton_kernels.py")
        self.assertEqual(pinned_path.stat().st_size, pinned["bytes"])
        self.assertEqual(CONTROLLER.sha256_file(pinned_path), pinned["sha256"])
        repair = json.loads(
            (
                HERE.parent
                / "37_mech09_downproj_refresh_mediation"
                / "refresh_mediation_repair_contract.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            pinned["sha256"], repair["source_constraints"]["triton_sha256"]
        )
        self.assertEqual(
            STREAM_WORKER.validate_stream_contract(contract),
            "mdp04_refresh_stream_contract_v4",
        )
        predecessor_spec = contract["resume_compatibility"]["allowed_predecessors"][0]
        self.assertTrue(
            CONTROLLER.predecessor_contract_allowed(
                contract,
                {"schema_version": predecessor_spec["schema_version"]},
                predecessor_spec["sha256"],
            )
        )
        self.assertFalse(
            CONTROLLER.predecessor_contract_allowed(
                contract,
                {"schema_version": predecessor_spec["schema_version"]},
                "0" * 64,
            )
        )
        unsupported = dict(contract)
        unsupported["schema_version"] = "mdp04_refresh_stream_contract_v5"
        with self.assertRaisesRegex(RuntimeError, "unsupported stream contract"):
            STREAM_WORKER.validate_stream_contract(unsupported)
        invalid_boundary = dict(contract)
        invalid_boundary["local_posthoc_audit_reference"] = dict(
            contract["local_posthoc_audit_reference"],
            remote_source_required=True,
        )
        with self.assertRaisesRegex(RuntimeError, "invalid v2/v3/v4"):
            STREAM_WORKER.validate_stream_contract(invalid_boundary)
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            sealed = snapshot / pinned["relative_path"]
            sealed.parent.mkdir(parents=True)
            sealed.write_bytes(pinned_path.read_bytes())
            manifest = snapshot / "source_snapshot_manifest.json"
            manifest.write_text(
                json.dumps({"files": {pinned["relative_path"]: pinned["sha256"]}}),
                encoding="utf-8",
            )
            self.assertEqual(
                STREAM_WORKER.resolve_pinned_runtime_source(
                    contract, manifest, "triton_kernels"
                ),
                sealed,
            )
        requested = HERE / "fake_venv" / "bin" / "python"
        with mock.patch.object(
            Path,
            "resolve",
            side_effect=AssertionError("virtual-environment symlinks must be preserved"),
        ):
            self.assertEqual(
                CONTROLLER.executable_path(requested),
                os.path.abspath(os.fspath(requested)),
            )

    def test_pooled_ratio_does_not_treat_layers_as_independent(self) -> None:
        rows = [
            {
                "x_fro_before": "3.0",
                "x_delta_fro": "4.0",
                "x_fro_after": "5.0",
                "cos": "0.5",
            },
            {
                "x_fro_before": "4.0",
                "x_delta_fro": "3.0",
                "x_fro_after": "5.0",
                "cos": "0.5",
            },
        ]
        self.assertAlmostEqual(VALIDATOR.pooled_ratio(rows, "x"), 1.0)
        self.assertAlmostEqual(
            VALIDATOR.pooled_cosine(rows, "x", "cos"),
            17.5 / (5.0 * (50.0 ** 0.5)),
        )

    def test_validator_small_end_to_end_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            run = root / "run"
            analysis_source = source / "analysis"
            analysis_source.mkdir(parents=True)

            def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            paired_rows = [
                {
                    "checkpoint_cell": "origin",
                    "data_replica": 0,
                    "optimizer_step": 48,
                    "contrast": "c32",
                    "normalized_loss_delta": -0.1,
                },
                {
                    "checkpoint_cell": "origin",
                    "data_replica": 0,
                    "optimizer_step": 80,
                    "contrast": "c64",
                    "normalized_loss_delta": 0.2,
                },
            ]
            auc_rows = [
                {
                    "checkpoint_cell": "origin",
                    "data_replica": 0,
                    "contrast": "c32",
                    "auc_delta": -0.01,
                },
                {
                    "checkpoint_cell": "origin",
                    "data_replica": 0,
                    "contrast": "c64",
                    "auc_delta": 0.02,
                },
            ]
            write_csv(analysis_source / "paired_contrasts.csv", paired_rows)
            write_csv(analysis_source / "auc_contrasts.csv", auc_rows)

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            contract = {
                "accepted_source_artifact_sha256": {
                    "analysis/paired_contrasts.csv": digest(
                        analysis_source / "paired_contrasts.csv"
                    ),
                    "analysis/auc_contrasts.csv": digest(
                        analysis_source / "auc_contrasts.csv"
                    ),
                },
                "coverage": {
                    "origins": ["origin"],
                    "replicas": [0],
                    "events": [
                        {
                            "event_id": "e32",
                            "completed_step": 32,
                            "accepted_loss_contrast": "c32",
                            "accepted_loss_step": 48,
                            "loss_harm_orientation": "negative_of_normalized_loss_delta",
                        },
                        {
                            "event_id": "e64",
                            "completed_step": 64,
                            "accepted_loss_contrast": "c64",
                            "accepted_loss_step": 80,
                            "loss_harm_orientation": "normalized_loss_delta",
                        },
                    ],
                    "layer_indices": [0, 1],
                    "expected_layer_event_rows": 4,
                    "expected_unit_event_rows": 2,
                },
                "hard_gates": {
                    "covariance_refresh_identity_relative_residual_max": 1.0,
                    "k_asymmetry_relative_max": 1.0,
                    "inverse_asymmetry_relative_max": 1.0,
                    "runtime_inverse_backward_residual_max": 1.0,
                    "runtime_resolvent_relative_residual_max": 1.0,
                },
                "validation_slices": {
                    "events": ["e32", "e64"],
                    "layers": [0],
                },
                "transport": {"single_file_size_limit_bytes": 2147483648},
                "resume_compatibility": {
                    "allowed_predecessors": [
                        {
                            "schema_version": "mdp04_refresh_stream_contract_v3",
                            "sha256": "legacy-v3-contract",
                            "reuse_scope": "selected passed units only",
                        }
                    ]
                },
            }
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            attempt = run / "formal" / "origin" / "replica_0" / "attempt_001"
            attempt.mkdir(parents=True)
            metric_rows = []
            for event in ("e32", "e64"):
                for layer in (0, 1):
                    metric_rows.append(
                        {
                            "origin": "origin",
                            "data_replica": 0,
                            "event_id": event,
                            "layer_index": layer,
                            "all_full_state_values_finite": True,
                            "actual_preconditioned_gradient_fingerprint_match": True,
                            "actual_ns_input_fingerprint_match": True,
                            "actual_ns_output_fingerprint_match": True,
                            "accepted_covariance_before_fingerprint_exact": False,
                            "accepted_inverse_before_fingerprint_exact": False,
                            "accepted_covariance_after_fingerprint_exact": False,
                            "accepted_inverse_after_fingerprint_exact": False,
                            "accepted_covariance_before_numeric_match": False,
                            "accepted_inverse_before_numeric_match": True,
                            "accepted_covariance_after_numeric_match": True,
                            "accepted_inverse_after_numeric_match": True,
                            "accepted_covariance_before_reference_integrity_passed": True,
                            "accepted_inverse_before_reference_integrity_passed": True,
                            "accepted_covariance_after_reference_integrity_passed": True,
                            "accepted_inverse_after_reference_integrity_passed": True,
                            "accepted_covariance_before_max_abs_error": 1.0e-7,
                            "accepted_inverse_before_max_abs_error": 1.0e-7,
                            "accepted_covariance_after_max_abs_error": 1.0e-7,
                            "accepted_inverse_after_max_abs_error": 1.0e-7,
                            "accepted_covariance_before_max_relative_error": 1.0e-7,
                            "accepted_inverse_before_max_relative_error": 1.0e-7,
                            "accepted_covariance_after_max_relative_error": 1.0e-7,
                            "accepted_inverse_after_max_relative_error": 1.0e-7,
                            "covariance_refresh_identity_relative_residual": 0.0,
                            "k_asymmetry_before": 0.0,
                            "k_asymmetry_after": 0.0,
                            "inverse_asymmetry_before": 0.0,
                            "inverse_asymmetry_after": 0.0,
                            "runtime_inverse_backward_residual_before": 0.0,
                            "runtime_inverse_backward_residual_after": 0.0,
                            "runtime_resolvent_relative_residual": 0.0,
                            "relative_k_fro_change": 0.1,
                            "relative_a_fro_change": 0.1,
                            "relative_runtime_inverse_fro_change": 0.1,
                            "matched_g_preconditioned_relative_change": 0.1,
                            "runtime_ns5_update_relative_change": 0.1,
                            "condition_proxy_before": 2.0,
                            "condition_proxy_after": 2.1,
                            "matched_g_preconditioned_fro_before": 2.0,
                            "matched_g_preconditioned_fro_after": 2.1,
                            "matched_g_preconditioned_delta_fro": 0.2,
                            "matched_g_preconditioned_cosine": 0.99,
                            "runtime_ns5_update_fro_before": 3.0,
                            "runtime_ns5_update_fro_after": 3.1,
                            "runtime_ns5_update_delta_fro": 0.3,
                            "runtime_ns5_update_cosine": 0.98,
                            "ridge_before": 0.2,
                            "ridge_after": 0.21,
                        }
                    )
            write_csv(attempt / "refresh_layer_event_metrics.csv", metric_rows)
            slice_dir = attempt / "validation_slices"
            slice_dir.mkdir()
            for event in ("e32", "e64"):
                npz = slice_dir / f"{event}.npz"
                identity = np.eye(4, dtype=np.float32)
                update = np.eye(4, dtype=np.float32)
                np.savez_compressed(
                    npz,
                    covariance_before=identity,
                    covariance_after=identity * 1.01,
                    runtime_inverse_before=identity,
                    runtime_inverse_after=identity,
                    raw_gradient=identity,
                    historical_momentum=identity,
                    matched_gradient_before=identity,
                    matched_gradient_after=identity,
                    runtime_ns5_update_before=update,
                    runtime_ns5_update_after=update,
                )
                (slice_dir / f"{event}.json").write_text(
                    json.dumps(
                        {
                            "origin": "origin",
                            "data_replica": 0,
                            "event_id": event,
                            "layer_index": 0,
                            "npz": npz.name,
                            "npz_sha256": digest(npz),
                            "warning": "slice only",
                        }
                    ),
                    encoding="utf-8",
                )
            unit_manifest_path = attempt / "stream_unit_manifest.json"
            unit_manifest_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "stream_contract_sha256": digest(contract_path),
                    }
                ),
                encoding="utf-8",
            )
            selection = attempt.parent / "unit_selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "selected_attempt": "attempt_001",
                        "manifest_sha256": digest(unit_manifest_path),
                    }
                ),
                encoding="utf-8",
            )
            args = type(
                "Args",
                (),
                {"run_dir": run, "source_run": source, "contract": contract_path},
            )()
            manifest = VALIDATOR.validate(args)
            self.assertTrue(manifest["passed"])
            self.assertFalse(
                manifest["accepted_refresh_diagnostics"][
                    "all_17_point_values_within_original_v3_tolerance"
                ]
            )
            self.assertEqual(manifest["rows"]["layer_event"], 4)
            self.assertEqual(manifest["rows"]["validation_slice"], 2)
            for row in metric_rows:
                row["accepted_covariance_before_numeric_match"] = True
                for prefix in (
                    "accepted_covariance_before",
                    "accepted_inverse_before",
                    "accepted_covariance_after",
                    "accepted_inverse_after",
                ):
                    del row[f"{prefix}_reference_integrity_passed"]
            write_csv(attempt / "refresh_layer_event_metrics.csv", metric_rows)
            unit_manifest_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "stream_contract_sha256": "legacy-v3-contract",
                    }
                ),
                encoding="utf-8",
            )
            selection.write_text(
                json.dumps(
                    {
                        "selected_attempt": "attempt_001",
                        "manifest_sha256": digest(unit_manifest_path),
                    }
                ),
                encoding="utf-8",
            )
            inherited_manifest = VALIDATOR.validate(args)
            self.assertTrue(inherited_manifest["passed"])
            self.assertEqual(
                inherited_manifest["unit_contract_lineage_counts"][
                    "inherited_stricter_v3"
                ],
                1,
            )


if __name__ == "__main__":
    unittest.main()
