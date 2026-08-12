#!/usr/bin/env python3
"""Instrument the accepted MECH-09R worker and stream MDP-04 metrics.

The accepted worker is reused for checkpoint loading, optimizer-state transfer,
preconditioner build, loaders, RNG restoration, compilation, and branch logic.
Only the production step-32 and delayed step-64 paths are retained.  Full
matrices are reduced one layer at a time and are never persisted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback
import types
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SCRIPT_VERSION = "2026-08-03.6"
SUPPORTED_STREAM_CONTRACT_SCHEMAS = {
    "mdp04_refresh_stream_contract_v1",
    "mdp04_refresh_stream_contract_v2",
    "mdp04_refresh_stream_contract_v3",
    "mdp04_refresh_stream_contract_v4",
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


METRICS = load_module("mdp04_stream_metrics", HERE / "stream_metrics.py")
WORKER = load_module(
    "mdp04_accepted_mech09r_worker",
    SCRIPTS / "37_mech09_downproj_refresh_mediation" / "mech09r_worker.py",
)
LEGACY = WORKER.LEGACY


class CaptureComplete(RuntimeError):
    """Internal sentinel raised only after both registered events finish."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def validate_stream_contract(contract: dict[str, Any]) -> str:
    schema = contract.get("schema_version")
    if schema not in SUPPORTED_STREAM_CONTRACT_SCHEMAS:
        raise RuntimeError(
            f"unsupported stream contract schema={schema!r}; "
            f"supported={sorted(SUPPORTED_STREAM_CONTRACT_SCHEMAS)}"
        )
    if schema in {
        "mdp04_refresh_stream_contract_v2",
        "mdp04_refresh_stream_contract_v3",
        "mdp04_refresh_stream_contract_v4",
    }:
        reference = contract.get("local_posthoc_audit_reference")
        if not isinstance(reference, dict) or any(
            reference.get(field) is not False
            for field in ("remote_source_required", "used_by_worker", "used_by_validator")
        ):
            raise RuntimeError("invalid v2/v3/v4 local post-hoc audit boundary")
    if schema in {
        "mdp04_refresh_stream_contract_v3",
        "mdp04_refresh_stream_contract_v4",
    }:
        audit = contract.get("cross_run_replay_audit")
        expected_fields = [
            "structure_sha256",
            "tensor_count",
            "sampled_values_finite",
            "next_x_sha256",
            "next_y_sha256",
            "loader_state",
            "matrix_global_step",
        ]
        if not isinstance(audit, dict) or any(
            audit.get(field) is not expected
            for field, expected in {
                "amended_before_any_mdp04_layer_metric_was_accepted": True,
                "within_replay_branch_sha256_exact": True,
                "accepted_branch_sha256_diagnostic_only": True,
                "stream_hooks_inactive_at_branch_anchors": True,
                "accepted_refresh_fingerprint_metadata_exact": True,
                "accepted_refresh_sha256_diagnostic_only": True,
            }.items()
        ):
            raise RuntimeError("invalid v3/v4 cross-run replay audit boundary")
        if audit.get("accepted_branch_hard_exact_fields") != expected_fields:
            raise RuntimeError("invalid v3/v4 accepted branch hard fields")
        tolerance = audit.get("accepted_refresh_value_tolerance")
        if not isinstance(tolerance, dict) or not (
            tolerance.get("comparison") == "elementwise_abs_le_atol_plus_rtol_abs_expected"
            and 0.0 < float(tolerance.get("rtol", 0.0)) <= 1.0e-4
            and 0.0 < float(tolerance.get("atol", 0.0)) <= 1.0e-6
        ):
            raise RuntimeError("invalid v3/v4 accepted refresh tolerance")
        if schema == "mdp04_refresh_stream_contract_v4" and any(
            audit.get(field) is not expected
            for field, expected in {
                "amended_after_cross_run_numeric_gate_failed_before_any_mdp04_metric_was_accepted": True,
                "accepted_refresh_values_diagnostic_only": True,
                "accepted_refresh_metadata_count_and_finiteness_hard": True,
            }.items()
        ):
            raise RuntimeError("invalid v4 floating replay audit boundary")
    return str(schema)


def resolve_pinned_runtime_source(
    contract: dict[str, Any], source_snapshot_manifest: Path, name: str
) -> Path:
    runtime_sources = contract.get("pinned_runtime_sources")
    if not isinstance(runtime_sources, dict) or name not in runtime_sources:
        raise RuntimeError(f"missing pinned runtime source: {name}")
    spec = runtime_sources[name]
    relative = str(spec["relative_path"])
    expected = str(spec["sha256"])
    expected_bytes = int(spec["bytes"])
    manifest = read_json(source_snapshot_manifest)
    if manifest.get("files", {}).get(relative) != expected:
        raise RuntimeError(f"source snapshot did not seal pinned runtime source: {name}")
    path = source_snapshot_manifest.parent / relative
    if (
        not path.is_file()
        or path.stat().st_size != expected_bytes
        or sha256_file(path) != expected
    ):
        raise RuntimeError(f"pinned runtime source integrity failed: {name}")
    return path


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def replace_option(arguments: list[str], option: str, value: str) -> None:
    try:
        index = arguments.index(option)
    except ValueError as exc:
        raise RuntimeError(f"missing accepted worker option: {option}") from exc
    if index + 1 >= len(arguments):
        raise RuntimeError(f"missing value after accepted worker option: {option}")
    arguments[index + 1] = value


def parse_layer(group_name: str) -> int:
    parts = group_name.split(".")
    if len(parts) != 3 or parts[0] != "layers" or parts[2] != "down_input":
        raise RuntimeError(f"unexpected target group name: {group_name}")
    return int(parts[1])


def compare_replay_fingerprint(
    expected: dict[str, Any],
    observed: dict[str, Any],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Compare saved float32 samples without requiring cross-process bit identity."""

    metadata_checks = {
        field: observed.get(field) == expected.get(field)
        for field in ("shape", "dtype", "indices")
    }
    expected_values = np.asarray(expected.get("values", []), dtype=np.float64)
    observed_values = np.asarray(observed.get("values", []), dtype=np.float64)
    value_count = int(expected_values.size)
    count_match = expected_values.shape == observed_values.shape
    finite = bool(
        count_match
        and np.isfinite(expected_values).all()
        and np.isfinite(observed_values).all()
    )
    if count_match and value_count:
        absolute = np.abs(observed_values - expected_values)
        allowed = float(atol) + float(rtol) * np.abs(expected_values)
        numeric_match = bool(finite and np.all(absolute <= allowed))
        max_abs_error = float(np.max(absolute))
        relative = absolute / np.maximum(np.abs(expected_values), float(atol))
        max_relative_error = float(np.max(relative))
    else:
        numeric_match = bool(finite and value_count == 0)
        max_abs_error = 0.0 if numeric_match else math.inf
        max_relative_error = 0.0 if numeric_match else math.inf
    fingerprint_exact = (
        observed.get("fingerprint_sha256") == expected.get("fingerprint_sha256")
    )
    passed = bool(
        all(metadata_checks.values()) and count_match and finite and numeric_match
    )
    reference_integrity_passed = bool(
        all(metadata_checks.values()) and count_match and finite
    )
    return {
        "metadata_checks": metadata_checks,
        "value_count": value_count,
        "value_count_match": count_match,
        "values_finite": finite,
        "fingerprint_sha256_exact": fingerprint_exact,
        "expected_fingerprint_sha256": expected.get("fingerprint_sha256"),
        "observed_fingerprint_sha256": observed.get("fingerprint_sha256"),
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs_error": max_abs_error,
        "max_relative_error": max_relative_error,
        "numeric_match": numeric_match,
        "reference_integrity_passed": reference_integrity_passed,
        "passed": passed,
    }


class StreamRecorder:
    def __init__(
        self,
        *,
        output: Path,
        contract_path: Path,
        accepted_unit: Path,
        worker_args: argparse.Namespace,
        source_snapshot_manifest: Path,
    ) -> None:
        self.output = output
        self.contract_path = contract_path
        self.contract = read_json(contract_path)
        self.accepted_unit = accepted_unit
        self.worker_args = worker_args
        self.source_snapshot_manifest = source_snapshot_manifest
        self.accepted_refresh = read_json(accepted_unit / "refresh_tree_audit.json")
        self.accepted_branch = read_json(accepted_unit / "branch_audit.json")
        accepted_manifest = read_json(accepted_unit / "mech09r_manifest.json")
        if not (
            self.accepted_refresh.get("passed") is True
            and self.accepted_branch.get("passed") is True
            and accepted_manifest.get("passed") is True
        ):
            raise RuntimeError("accepted MECH-09R unit is not passed")
        if accepted_manifest.get("checkpoint_cell") != worker_args.cell:
            raise RuntimeError("accepted unit cell mismatch")
        if int(accepted_manifest.get("data_replica", -1)) != int(
            worker_args.data_replica
        ):
            raise RuntimeError("accepted unit replica mismatch")
        self.rows: list[dict[str, Any]] = []
        self.event_audits: list[dict[str, Any]] = []
        self.branch_anchors: list[dict[str, Any]] = []
        self.pending: dict[str, Any] | None = None
        self.original_ns: Any = None
        self._bound_optimizer: Any = None
        self._source_module: Any = None
        self._original_apply_had_instance_override = False
        self._original_apply_instance_value: Any = None
        self._patched_apply_bound: Any = None
        self._patched_ns: Any = None
        self._bound_event_id: str | None = None
        self.hook_lifecycle: list[dict[str, Any]] = []
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "validation_slices").mkdir(exist_ok=True)

    def expected_branch_anchor(self, label: str) -> dict[str, Any]:
        matches = [
            row
            for row in self.accepted_branch["branch_start_audits"]
            if row["label"] == label
        ]
        if len(matches) != 1:
            raise RuntimeError(f"accepted branch anchor is not unique: {label}")
        return matches[0]["expected"]

    def record_branch_anchor(
        self,
        label: str,
        observed: dict[str, Any],
        within_replay: dict[str, Any],
    ) -> None:
        expected = self.expected_branch_anchor(label)
        fields = self.contract["cross_run_replay_audit"][
            "accepted_branch_hard_exact_fields"
        ]
        accepted_checks = {
            field: observed.get(field) == expected.get(field) for field in fields
        }
        stream_hooks_inactive = self._bound_optimizer is None
        within_replay_passed = within_replay.get("passed") is True
        payload = {
            "schema_version": "mdp04_branch_anchor_audit_v2",
            "label": label,
            "within_replay_sha256_exact": within_replay_passed,
            "within_replay_checks": within_replay.get("checks"),
            "accepted_hard_checks": accepted_checks,
            "accepted_sha256_exact_diagnostic": observed.get("sha256")
            == expected.get("sha256"),
            "accepted_expected_sha256": expected.get("sha256"),
            "accepted_observed_sha256": observed.get("sha256"),
            "stream_hooks_inactive": stream_hooks_inactive,
            "passed": within_replay_passed
            and all(accepted_checks.values())
            and stream_hooks_inactive,
        }
        self.branch_anchors.append(payload)
        if not payload["passed"]:
            raise RuntimeError(f"accepted branch anchor mismatch: {payload}")

    def bind_optimizer(self, optimizer: Any, *, event_id: str) -> None:
        if self._bound_optimizer is optimizer:
            return
        if self._bound_optimizer is not None:
            raise RuntimeError("recorder cannot bind more than one optimizer")
        self._bound_optimizer = optimizer
        module_name = optimizer.__class__.__module__
        source_module = sys.modules.get(module_name)
        if source_module is None:
            raise RuntimeError(f"source optimizer module is not loaded: {module_name}")
        self._source_module = source_module
        self.original_ns = source_module.zeropower_via_newtonschulz5
        self._original_apply_had_instance_override = (
            "_apply_preconditioners" in optimizer.__dict__
        )
        self._original_apply_instance_value = optimizer.__dict__.get(
            "_apply_preconditioners"
        )

        recorder = self
        original_apply = optimizer._apply_preconditioners

        def patched_apply(_: Any) -> None:
            original_apply()
            recorder.validate_applied_gradients()

        self._patched_apply_bound = types.MethodType(patched_apply, optimizer)
        optimizer._apply_preconditioners = self._patched_apply_bound

        def patched_ns(gradient: torch.Tensor, steps: int = 5) -> torch.Tensor:
            output = recorder.original_ns(gradient, steps=steps)
            recorder.validate_actual_ns_call(gradient, output)
            return output

        source_module.zeropower_via_newtonschulz5 = patched_ns
        self._patched_ns = patched_ns
        self._bound_event_id = event_id
        self.hook_lifecycle.append({"action": "bind", "event_id": event_id})

    def unbind_optimizer(self, *, allow_pending: bool = False) -> None:
        if self._bound_optimizer is None:
            return
        if self.pending is not None and not allow_pending:
            raise RuntimeError("cannot unbind stream hooks while an event is pending")
        optimizer = self._bound_optimizer
        source_module = self._source_module
        if optimizer.__dict__.get("_apply_preconditioners") is not self._patched_apply_bound:
            raise RuntimeError("optimizer apply hook changed before restoration")
        if source_module is None or source_module.zeropower_via_newtonschulz5 is not self._patched_ns:
            raise RuntimeError("NS5 hook changed before restoration")
        if self._original_apply_had_instance_override:
            optimizer.__dict__["_apply_preconditioners"] = (
                self._original_apply_instance_value
            )
        else:
            del optimizer.__dict__["_apply_preconditioners"]
        source_module.zeropower_via_newtonschulz5 = self.original_ns
        self.hook_lifecycle.append(
            {"action": "unbind", "event_id": self._bound_event_id}
        )
        self._bound_optimizer = None
        self._source_module = None
        self.original_ns = None
        self._original_apply_had_instance_override = False
        self._original_apply_instance_value = None
        self._patched_apply_bound = None
        self._patched_ns = None
        self._bound_event_id = None

    def event_spec(self, trajectory: str, completed_step: int) -> dict[str, Any]:
        matches = [
            row
            for row in self.contract["coverage"]["events"]
            if row["trajectory_node"] == trajectory
            and int(row["completed_step"]) == int(completed_step)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"stream event is not registered: {trajectory} step={completed_step}"
            )
        return matches[0]

    def expected_refresh_event(self, event_spec: dict[str, Any]) -> dict[str, Any]:
        trajectory = event_spec["trajectory_node"]
        step = int(event_spec["completed_step"])
        matches = [
            row
            for row in self.accepted_refresh["segment_audits"][trajectory]["events"]
            if int(row["completed_step"]) == step
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"accepted refresh event is not unique: {trajectory} step={step}"
            )
        return matches[0]

    def start_event(self, event_spec: dict[str, Any]) -> None:
        if self.pending is not None:
            raise RuntimeError("previous stream event is still pending")
        self.pending = {
            "spec": event_spec,
            "rows": [],
            "internal": [],
            "slices": [],
            "actual_ns_calls": 0,
        }

    def capture_group_and_refresh(
        self,
        *,
        controller: Any,
        group: dict[str, Any],
    ) -> None:
        if self.pending is None:
            raise RuntimeError("no pending stream event")
        optimizer = controller.optimizer
        members = list(group["members"])
        if len(members) != 1:
            raise RuntimeError(
                f"down-input group must have one member: {group['name']}"
            )
        parameter = members[0]
        if parameter.grad is None:
            raise RuntimeError(f"missing raw gradient: {group['name']}")
        state = optimizer.state[parameter]
        if "momentum" not in state:
            raise RuntimeError(f"missing historical momentum: {group['name']}")
        count = group["count"]
        if float(count.item()) <= 0.0:
            raise RuntimeError(f"empty refresh statistics: {group['name']}")

        covariance_before = state["precond_cov"].detach().clone()
        inverse_before = state["precond_inv_apply"].detach().clone()
        fresh_covariance = (group["accum"] / count).detach().clone()
        optimizer._groups = [group]
        controller.original_refresh()
        covariance_after = state["precond_cov"]
        inverse_after = state["precond_inv_apply"]
        event_spec = self.pending["spec"]
        layer = parse_layer(str(group["name"]))
        probe_contract = self.contract["probe_metrics"]
        slice_contract = self.contract["validation_slices"]
        selected_slice = (
            self.worker_args.cell == slice_contract["origin"]
            and int(self.worker_args.data_replica) == int(slice_contract["replica"])
            and event_spec["event_id"] in slice_contract["events"]
            and layer in slice_contract["layers"]
        )
        probe_seed = METRICS.stable_seed(
            int(probe_contract["base_seed"]),
            self.worker_args.cell,
            int(self.worker_args.data_replica),
            event_spec["event_id"],
            layer,
        )
        slice_seed = METRICS.stable_seed(
            int(slice_contract["seed"]), event_spec["event_id"], layer
        )
        metrics, fingerprints, slice_payload = METRICS.compute_layer_metrics(
            covariance_before=covariance_before,
            covariance_after=covariance_after,
            inverse_before=inverse_before,
            inverse_after=inverse_after,
            fresh_covariance=fresh_covariance,
            raw_gradient=parameter.grad,
            historical_momentum=state["momentum"],
            input_beta=float(optimizer.input_beta),
            ridge_scale=float(optimizer.input_ridge),
            ridge_epsilon=float(self.contract["matrix_contract"]["ridge_epsilon"]),
            momentum_beta=float(optimizer.param_groups[0]["momentum"]),
            ns_steps=int(optimizer.param_groups[0]["ns_steps"]),
            ns_update=lambda value, steps: self.original_ns(value, steps=steps),
            probe_count=int(probe_contract["probe_count"]),
            probe_iterations=int(probe_contract["power_iterations"]),
            probe_seed=probe_seed,
            slice_coordinate_count=(
                int(slice_contract["coordinate_count"]) if selected_slice else 0
            ),
            slice_gradient_row_count=(
                int(slice_contract["gradient_row_count"]) if selected_slice else 0
            ),
            slice_seed=slice_seed,
        )
        row = {
            "schema_version": "mdp04_refresh_layer_event_v1",
            "origin": self.worker_args.cell,
            "data_replica": int(self.worker_args.data_replica),
            "event_id": event_spec["event_id"],
            "trajectory_node": event_spec["trajectory_node"],
            "completed_step": int(event_spec["completed_step"]),
            "layer_index": layer,
            "module_id": str(group["name"]),
            "checkpoint_sha256": read_json(
                self.worker_args.checkpoint_hash_certificate
            )["sha256"],
            "source_script_sha256": sha256_file(self.worker_args.source_script),
            "repair_contract_sha256": sha256_file(self.worker_args.contract),
            "stream_contract_sha256": sha256_file(self.contract_path),
            "input_beta": float(optimizer.input_beta),
            "ridge_scale": float(optimizer.input_ridge),
            "matrix_momentum": float(optimizer.param_groups[0]["momentum"]),
            "ns_steps": int(optimizer.param_groups[0]["ns_steps"]),
            "matched_gradient_semantics": self.contract["matrix_contract"][
                "matched_gradient_semantics"
            ],
            "raw_full_matrices_persisted": False,
            "validation_slice_persisted": bool(selected_slice),
            **metrics,
            "raw_gradient_fingerprint_sha256": fingerprints["raw_gradient"][
                "fingerprint_sha256"
            ],
            "historical_momentum_fingerprint_sha256": fingerprints[
                "historical_momentum"
            ]["fingerprint_sha256"],
            "shadow_gradient_after_fingerprint_sha256": fingerprints[
                "gradient_after"
            ]["fingerprint_sha256"],
            "shadow_ns_input_after_fingerprint_sha256": fingerprints[
                "ns_input_after"
            ]["fingerprint_sha256"],
            "shadow_ns_output_after_fingerprint_sha256": fingerprints[
                "ns_output_after"
            ]["fingerprint_sha256"],
            "actual_preconditioned_gradient_fingerprint_match": False,
            "actual_ns_input_fingerprint_match": False,
            "actual_ns_output_fingerprint_match": False,
        }
        self.pending["rows"].append(row)
        self.pending["internal"].append(
            {
                "parameter": parameter,
                "row": row,
                "gradient_after": fingerprints["gradient_after"],
                "ns_input_after": fingerprints["ns_input_after"],
                "ns_output_after": fingerprints["ns_output_after"],
            }
        )
        if slice_payload is not None:
            self.pending["slices"].append(
                {
                    "event_id": event_spec["event_id"],
                    "layer_index": layer,
                    "payload": slice_payload,
                }
            )
        del covariance_before, inverse_before, fresh_covariance

    def attach_accepted_fingerprint_checks(
        self,
        *,
        target_before: dict[str, Any],
        target_after: dict[str, Any],
    ) -> None:
        if self.pending is None:
            raise RuntimeError("no pending event for fingerprint checks")
        expected = self.expected_refresh_event(self.pending["spec"])
        observed_before = {row["name"]: row for row in target_before["rows"]}
        observed_after = {row["name"]: row for row in target_after["rows"]}
        expected_before = {row["name"]: row for row in expected["target_before"]["rows"]}
        expected_after = {row["name"]: row for row in expected["target_after"]["rows"]}
        tolerance = self.contract["cross_run_replay_audit"][
            "accepted_refresh_value_tolerance"
        ]
        all_exact = True
        all_numeric = True
        for row in self.pending["rows"]:
            name = row["module_id"]
            comparisons = {
                "accepted_covariance_before": compare_replay_fingerprint(
                    expected_before[name]["covariance"],
                    observed_before[name]["covariance"],
                    rtol=float(tolerance["rtol"]),
                    atol=float(tolerance["atol"]),
                ),
                "accepted_inverse_before": compare_replay_fingerprint(
                    expected_before[name]["inverse"],
                    observed_before[name]["inverse"],
                    rtol=float(tolerance["rtol"]),
                    atol=float(tolerance["atol"]),
                ),
                "accepted_covariance_after": compare_replay_fingerprint(
                    expected_after[name]["covariance"],
                    observed_after[name]["covariance"],
                    rtol=float(tolerance["rtol"]),
                    atol=float(tolerance["atol"]),
                ),
                "accepted_inverse_after": compare_replay_fingerprint(
                    expected_after[name]["inverse"],
                    observed_after[name]["inverse"],
                    rtol=float(tolerance["rtol"]),
                    atol=float(tolerance["atol"]),
                ),
            }
            for prefix, comparison in comparisons.items():
                row[f"{prefix}_fingerprint_exact"] = comparison[
                    "fingerprint_sha256_exact"
                ]
                row[f"{prefix}_numeric_match"] = comparison["passed"]
                row[f"{prefix}_reference_integrity_passed"] = comparison[
                    "reference_integrity_passed"
                ]
                row[f"{prefix}_max_abs_error"] = comparison["max_abs_error"]
                row[f"{prefix}_max_relative_error"] = comparison[
                    "max_relative_error"
                ]
            all_exact = all_exact and all(
                comparison["fingerprint_sha256_exact"]
                for comparison in comparisons.values()
            )
            all_numeric = all_numeric and all(
                comparison["passed"] for comparison in comparisons.values()
            )
            if not all(
                comparison["reference_integrity_passed"]
                for comparison in comparisons.values()
            ):
                diagnostic = {
                    prefix: comparison
                    for prefix, comparison in comparisons.items()
                    if not comparison["reference_integrity_passed"]
                }
                raise RuntimeError(
                    "accepted refresh reference integrity mismatch: "
                    f"{name} {diagnostic}"
                )
        self.event_audits.append(
            {
                "event_id": self.pending["spec"]["event_id"],
                "accepted_refresh_reference_integrity_passed": True,
                "accepted_refresh_numeric_match_diagnostic": all_numeric,
                "accepted_refresh_fingerprints_exact_diagnostic": all_exact,
                "accepted_refresh_rtol": float(tolerance["rtol"]),
                "accepted_refresh_atol": float(tolerance["atol"]),
                "layer_count": len(self.pending["rows"]),
            }
        )

    def validate_applied_gradients(self) -> None:
        if self.pending is None:
            return
        for entry in self.pending["internal"]:
            parameter = entry["parameter"]
            if parameter.grad is None:
                raise RuntimeError("gradient disappeared before actual apply audit")
            observed = METRICS.tensor_fingerprint(parameter.grad)
            matched = (
                observed["fingerprint_sha256"]
                == entry["gradient_after"]["fingerprint_sha256"]
            )
            entry["row"][
                "actual_preconditioned_gradient_fingerprint_match"
            ] = matched
            if not matched:
                raise RuntimeError(
                    "shadow/actual preconditioned-gradient fingerprint mismatch "
                    f"for {entry['row']['module_id']}"
                )

    def validate_actual_ns_call(
        self, ns_input: torch.Tensor, ns_output: torch.Tensor
    ) -> None:
        if self.pending is None:
            return
        target_shape = tuple(self.contract["matrix_contract"]["gradient_shape"])
        if tuple(ns_input.shape) != target_shape:
            return
        input_fingerprint = METRICS.tensor_fingerprint(ns_input)
        matches = [
            entry
            for entry in self.pending["internal"]
            if not entry["row"]["actual_ns_input_fingerprint_match"]
            and input_fingerprint["fingerprint_sha256"]
            == entry["ns_input_after"]["fingerprint_sha256"]
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "actual NS5 input does not match exactly one shadow down input"
            )
        entry = matches[0]
        output_fingerprint = METRICS.tensor_fingerprint(ns_output)
        output_match = (
            output_fingerprint["fingerprint_sha256"]
            == entry["ns_output_after"]["fingerprint_sha256"]
        )
        entry["row"]["actual_ns_input_fingerprint_match"] = True
        entry["row"]["actual_ns_output_fingerprint_match"] = output_match
        self.pending["actual_ns_calls"] += 1
        if not output_match:
            raise RuntimeError(
                "shadow/actual NS5 output fingerprint mismatch for "
                f"{entry['row']['module_id']}"
            )

    def finish_event_after_optimizer_step(self, event_id: str) -> None:
        if self.pending is None or self.pending["spec"]["event_id"] != event_id:
            raise RuntimeError(f"event was not captured: {event_id}")
        layer_count = len(self.contract["coverage"]["layer_indices"])
        if len(self.pending["rows"]) != layer_count:
            raise RuntimeError(
                f"wrong layer count for {event_id}: {len(self.pending['rows'])}"
            )
        if int(self.pending["actual_ns_calls"]) != layer_count:
            raise RuntimeError(
                f"wrong actual NS5 call count for {event_id}: "
                f"{self.pending['actual_ns_calls']}"
            )
        required = [
            "actual_preconditioned_gradient_fingerprint_match",
            "actual_ns_input_fingerprint_match",
            "actual_ns_output_fingerprint_match",
            "accepted_covariance_before_reference_integrity_passed",
            "accepted_inverse_before_reference_integrity_passed",
            "accepted_covariance_after_reference_integrity_passed",
            "accepted_inverse_after_reference_integrity_passed",
        ]
        for row in self.pending["rows"]:
            if not all(row.get(field) is True for field in required):
                raise RuntimeError(
                    f"event audit incomplete for {row['module_id']}: "
                    f"{[(field, row.get(field)) for field in required]}"
                )
        self.rows.extend(self.pending["rows"])
        for item in self.pending["slices"]:
            filename = (
                f"{self.worker_args.cell}_replica{self.worker_args.data_replica}_"
                f"{item['event_id']}_layer{item['layer_index']}.npz"
            )
            path = self.output / "validation_slices" / filename
            np.savez_compressed(path, **item["payload"])
            atomic_json(
                path.with_suffix(".json"),
                {
                    "schema_version": "mdp04_validation_slice_v1",
                    "origin": self.worker_args.cell,
                    "data_replica": int(self.worker_args.data_replica),
                    "event_id": item["event_id"],
                    "layer_index": int(item["layer_index"]),
                    "npz": path.name,
                    "npz_sha256": sha256_file(path),
                    "paper_empirical_claim_eligible": False,
                    "warning": self.contract["validation_slices"]["warning"],
                },
            )
        event_rows = [row for row in self.pending["rows"]]
        write_csv(self.output / f"{event_id}_metrics.csv", event_rows)
        atomic_json(
            self.output / f"{event_id}_manifest.json",
            {
                "schema_version": "mdp04_stream_event_manifest_v1",
                "event_id": event_id,
                "rows": len(event_rows),
                "layers": sorted(row["layer_index"] for row in event_rows),
                "accepted_refresh_reference_integrity_passed": True,
                "accepted_refresh_numeric_match_diagnostic": all(
                    row["accepted_covariance_before_numeric_match"]
                    and row["accepted_inverse_before_numeric_match"]
                    and row["accepted_covariance_after_numeric_match"]
                    and row["accepted_inverse_after_numeric_match"]
                    for row in event_rows
                ),
                "accepted_fingerprints_exact_diagnostic": all(
                    row["accepted_covariance_before_fingerprint_exact"]
                    and row["accepted_inverse_before_fingerprint_exact"]
                    and row["accepted_covariance_after_fingerprint_exact"]
                    and row["accepted_inverse_after_fingerprint_exact"]
                    for row in event_rows
                ),
                "shadow_actual_fingerprints_exact": True,
                "all_values_finite": all(
                    row["all_full_state_values_finite"] is True for row in event_rows
                ),
                "passed": True,
            },
        )
        self.pending = None

    def finalize_unit(self) -> dict[str, Any]:
        expected_rows = len(self.contract["coverage"]["events"]) * len(
            self.contract["coverage"]["layer_indices"]
        )
        if self.pending is not None:
            raise RuntimeError("cannot finalize while an event is pending")
        if len(self.rows) != expected_rows:
            raise RuntimeError(
                f"unit has {len(self.rows)} rows, expected {expected_rows}"
            )
        expected_events = sorted(
            row["event_id"] for row in self.contract["coverage"]["events"]
        )
        observed_events = sorted({row["event_id"] for row in self.rows})
        branch_labels = {row["label"] for row in self.branch_anchors}
        checks = {
            "row_count": len(self.rows) == expected_rows,
            "event_set": observed_events == expected_events,
            "branch_anchors": branch_labels
            == {
                "production_starts_from_first_fork",
                "delayed_starts_from_second_fork",
            }
            and all(row["passed"] for row in self.branch_anchors),
            "finite": all(row["all_full_state_values_finite"] for row in self.rows),
            "accepted_refresh_reference_integrity": all(
                row["accepted_covariance_before_reference_integrity_passed"]
                and row["accepted_inverse_before_reference_integrity_passed"]
                and row["accepted_covariance_after_reference_integrity_passed"]
                and row["accepted_inverse_after_reference_integrity_passed"]
                for row in self.rows
            ),
            "shadow_actual": all(
                row["actual_preconditioned_gradient_fingerprint_match"]
                and row["actual_ns_input_fingerprint_match"]
                and row["actual_ns_output_fingerprint_match"]
                for row in self.rows
            ),
            "event_scoped_hooks": self._bound_optimizer is None
            and self.hook_lifecycle
            == [
                {"action": "bind", "event_id": "production_refresh_32"},
                {"action": "unbind", "event_id": "production_refresh_32"},
                {"action": "bind", "event_id": "delayed_refresh_64"},
                {"action": "unbind", "event_id": "delayed_refresh_64"},
            ],
        }
        write_csv(self.output / "refresh_layer_event_metrics.csv", self.rows)
        jsonl = self.output / "refresh_layer_event_metrics.jsonl"
        temporary = jsonl.with_suffix(jsonl.suffix + ".tmp")
        temporary.write_text(
            "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in self.rows
            ),
            encoding="utf-8",
        )
        os.replace(temporary, jsonl)
        artifacts = sorted(
            str(path.relative_to(self.output)).replace("\\", "/")
            for path in self.output.rglob("*")
            if path.is_file() and path.name not in {"status.json", "stream_unit_manifest.json"}
        )
        hashes = {
            name: sha256_file(self.output / name)
            for name in artifacts
        }
        passed = all(checks.values())
        manifest = {
            "schema_version": "mdp04_stream_unit_manifest_v1",
            "script_version": SCRIPT_VERSION,
            "origin": self.worker_args.cell,
            "data_replica": int(self.worker_args.data_replica),
            "accepted_unit": str(self.accepted_unit),
            "accepted_unit_refresh_audit_sha256": sha256_file(
                self.accepted_unit / "refresh_tree_audit.json"
            ),
            "accepted_unit_branch_audit_sha256": sha256_file(
                self.accepted_unit / "branch_audit.json"
            ),
            "stream_contract_sha256": sha256_file(self.contract_path),
            "source_snapshot_manifest": str(self.source_snapshot_manifest),
            "source_snapshot_manifest_sha256": sha256_file(
                self.source_snapshot_manifest
            ),
            "layer_event_rows": len(self.rows),
            "real_optimizer_steps_computed": int(
                self.contract["short_replay"]["real_optimizer_steps_per_unit"]
            ),
            "raw_full_matrices_persisted": False,
            "timing_eligible_for_paper": False,
            "checks": checks,
            "branch_anchors": self.branch_anchors,
            "event_audits": self.event_audits,
            "hook_lifecycle": self.hook_lifecycle,
            "artifacts": artifacts,
            "artifact_sha256": hashes,
            "passed": passed,
        }
        atomic_json(self.output / "stream_unit_manifest.json", manifest)
        atomic_json(
            self.output / "status.json",
            {
                "status": "passed" if passed else "integrity_failed",
                "script_version": SCRIPT_VERSION,
            },
        )
        if not passed:
            raise RuntimeError(f"stream unit gates failed: {checks}")
        return manifest


RECORDER: StreamRecorder | None = None
CURRENT_TRAJECTORY: str | None = None


class InstrumentedController(LEGACY.RefreshInterventionController):
    def __init__(self, optimizer: Any, **kwargs: Any) -> None:
        super().__init__(optimizer, **kwargs)
        if RECORDER is None:
            raise RuntimeError("stream recorder is not configured")

    @torch.no_grad()
    def handle_refresh(self) -> None:
        if RECORDER is None or CURRENT_TRAJECTORY is None:
            raise RuntimeError("stream instrumentation context is missing")
        completed_step = int(self.optimizer.global_step) + 1
        action = LEGACY.refresh_action(completed_step, self.target_refresh_steps)
        selected = (
            (CURRENT_TRAJECTORY == "production" and completed_step == 32)
            or (CURRENT_TRAJECTORY == "delayed" and completed_step == 64)
        ) and action == "refresh"
        if not selected:
            return super().handle_refresh()
        event_spec = RECORDER.event_spec(CURRENT_TRAJECTORY, completed_step)
        RECORDER.bind_optimizer(
            self.optimizer, event_id=str(event_spec["event_id"])
        )
        RECORDER.start_event(event_spec)
        target_before = LEGACY.group_state_snapshot(
            self.optimizer, self.target_groups, include_statistics=False
        )
        other_before = LEGACY.group_state_snapshot(
            self.optimizer, self.other_groups, include_statistics=False
        )
        try:
            self.optimizer._groups = self.other_groups
            self.original_refresh()
            for group in self.target_groups:
                RECORDER.capture_group_and_refresh(controller=self, group=group)
        finally:
            self.optimizer._groups = self.all_groups
        target_after = LEGACY.group_state_snapshot(
            self.optimizer, self.target_groups, include_statistics=True
        )
        other_after = LEGACY.group_state_snapshot(
            self.optimizer, self.other_groups, include_statistics=True
        )
        RECORDER.attach_accepted_fingerprint_checks(
            target_before=target_before, target_after=target_after
        )
        self.events.append(
            {
                "completed_step": completed_step,
                "target_action": action,
                "target_before": target_before,
                "target_after": target_after,
                "target_covariance_changed": target_before[
                    "covariance_fingerprint_sha256"
                ]
                != target_after["covariance_fingerprint_sha256"],
                "target_inverse_changed": target_before[
                    "inverse_fingerprint_sha256"
                ]
                != target_after["inverse_fingerprint_sha256"],
                "target_statistics_zero_after": target_after["accum_nonzero"] == 0
                and target_after["count_sum"] == 0.0,
                "other_before": other_before,
                "other_after": other_after,
                "other_covariance_changed": other_before[
                    "covariance_fingerprint_sha256"
                ]
                != other_after["covariance_fingerprint_sha256"],
                "other_inverse_changed": other_before[
                    "inverse_fingerprint_sha256"
                ]
                != other_after["inverse_fingerprint_sha256"],
                "other_statistics_zero_after": other_after["accum_nonzero"] == 0
                and other_after["count_sum"] == 0.0,
            }
        )


def install_short_replay_hooks() -> None:
    original_run_segment = WORKER.run_segment
    original_fingerprint_match = WORKER.fingerprint_match

    def audited_fingerprint_match(
        expected: dict[str, Any], observed: dict[str, Any], label: str
    ) -> dict[str, Any]:
        result = original_fingerprint_match(expected, observed, label)
        if RECORDER is not None and label in {
            "production_starts_from_first_fork",
            "delayed_starts_from_second_fork",
        }:
            RECORDER.record_branch_anchor(label, observed, result)
        return result

    def short_run_segment(**kwargs: Any) -> Any:
        global CURRENT_TRAJECTORY
        trajectory = str(kwargs["trajectory_node"])
        CURRENT_TRAJECTORY = trajectory
        if trajectory == "production":
            kwargs["end_step"] = 32
            kwargs["target_refresh_steps"] = (32,)
            kwargs["expected_global_events"] = (32,)
        elif trajectory == "delayed":
            kwargs["end_step"] = 64
            kwargs["target_refresh_steps"] = (64,)
            kwargs["expected_global_events"] = (64,)
        try:
            result = original_run_segment(**kwargs)
        except BaseException:
            if RECORDER is not None:
                RECORDER.unbind_optimizer(allow_pending=True)
            raise
        finally:
            CURRENT_TRAJECTORY = None
        if trajectory == "production":
            if RECORDER is None:
                raise RuntimeError("stream recorder disappeared")
            try:
                RECORDER.finish_event_after_optimizer_step("production_refresh_32")
            finally:
                RECORDER.unbind_optimizer(allow_pending=True)
        elif trajectory == "delayed":
            if RECORDER is None:
                raise RuntimeError("stream recorder disappeared")
            try:
                RECORDER.finish_event_after_optimizer_step("delayed_refresh_64")
            finally:
                RECORDER.unbind_optimizer(allow_pending=True)
            raise CaptureComplete("registered refresh boundaries completed")
        return result

    WORKER.fingerprint_match = audited_fingerprint_match
    WORKER.run_segment = short_run_segment
    WORKER.LEGACY.RefreshInterventionController = InstrumentedController


def parse_stream_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-output-dir", required=True, type=Path)
    parser.add_argument("--stream-contract", required=True, type=Path)
    parser.add_argument("--accepted-unit-dir", required=True, type=Path)
    parser.add_argument("--source-snapshot-manifest", required=True, type=Path)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    arguments = list(parsed.worker_args)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if not arguments:
        raise RuntimeError("accepted worker arguments are required after --")
    return parsed, arguments


def main() -> None:
    global RECORDER
    stream_args, worker_arguments = parse_stream_args()
    output = stream_args.stream_output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = stream_args.stream_contract.resolve()
    contract = read_json(contract_path)
    validate_stream_contract(contract)
    pinned_triton = resolve_pinned_runtime_source(
        contract,
        stream_args.source_snapshot_manifest.resolve(),
        "triton_kernels",
    )
    pinned_repair_contract = resolve_pinned_runtime_source(
        contract,
        stream_args.source_snapshot_manifest.resolve(),
        "repair_contract",
    )
    pinned_control_reference = resolve_pinned_runtime_source(
        contract,
        stream_args.source_snapshot_manifest.resolve(),
        "mech08_control_reference",
    )
    replace_option(worker_arguments, "--output-dir", str(output))
    replace_option(worker_arguments, "--triton-kernels", str(pinned_triton))
    replace_option(worker_arguments, "--contract", str(pinned_repair_contract))
    replace_option(
        worker_arguments,
        "--mech08-control-reference",
        str(pinned_control_reference),
    )
    sys.argv = [str(WORKER.__file__), *worker_arguments]
    worker_args = WORKER.parse_args()
    try:
        if sha256_file(worker_args.contract.resolve()) != contract[
            "source_repair_contract_sha256"
        ]:
            raise RuntimeError("MECH-09R repair contract hash mismatch")
        RECORDER = StreamRecorder(
            output=output,
            contract_path=contract_path,
            accepted_unit=stream_args.accepted_unit_dir.resolve(),
            worker_args=worker_args,
            source_snapshot_manifest=stream_args.source_snapshot_manifest.resolve(),
        )
        install_short_replay_hooks()
        try:
            WORKER.run_worker(worker_args)
        except CaptureComplete:
            pass
        manifest = RECORDER.finalize_unit()
        print(
            "MDP-04 stream unit passed: "
            f"origin={manifest['origin']} replica={manifest['data_replica']} "
            f"rows={manifest['layer_event_rows']}",
            flush=True,
        )
    except Exception as exc:
        atomic_json(
            output / "status.json",
            {
                "status": "failed",
                "script_version": SCRIPT_VERSION,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
