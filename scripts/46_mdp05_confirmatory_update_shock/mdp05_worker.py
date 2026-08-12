#!/usr/bin/env python3
"""Run one formal MDP-05 causal-tree unit with streaming matrix metrics."""

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


SCRIPT_VERSION = "2026-08-04.2"
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROTOCOL = load_module("mdp05_protocol", HERE / "protocol.py")
METRICS = load_module(
    "mdp05_stream_metrics",
    SCRIPTS / "mdp_refresh_streaming" / "stream_metrics.py",
)
WORKER = load_module(
    "mdp05_accepted_mech09r_worker",
    SCRIPTS / "37_mech09_downproj_refresh_mediation" / "mech09r_worker.py",
)
LEGACY = WORKER.LEGACY


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def parse_layer(group_name: str) -> int:
    parts = group_name.split(".")
    if len(parts) < 3 or parts[0] != "layers" or parts[-1] != "down_input":
        raise RuntimeError(f"unexpected target group: {group_name}")
    return int(parts[1])


def float64_slice_diagnostics(
    payload: dict[str, Any], ridge_before: float, ridge_after: float
) -> dict[str, Any]:
    covariance_before = payload["covariance_before"].astype(np.float64)
    covariance_after = payload["covariance_after"].astype(np.float64)
    inverse_before = payload["runtime_inverse_before"].astype(np.float64)
    inverse_after = payload["runtime_inverse_after"].astype(np.float64)
    size = covariance_before.shape[0]
    identity = np.eye(size, dtype=np.float64)
    a_before = covariance_before + float(ridge_before) * identity
    a_after = covariance_after + float(ridge_after) * identity
    lhs = inverse_after - inverse_before
    rhs = -inverse_after @ (a_after - a_before) @ inverse_before
    resolvent = np.linalg.norm(lhs - rhs) / max(
        np.linalg.norm(lhs) + np.linalg.norm(rhs), 1.0e-30
    )
    inverse_residual_before = np.linalg.norm(
        a_before @ inverse_before - identity
    ) / max(np.linalg.norm(identity), 1.0e-30)
    inverse_residual_after = np.linalg.norm(
        a_after @ inverse_after - identity
    ) / max(np.linalg.norm(identity), 1.0e-30)
    update_before = payload["runtime_ns5_update_before"].astype(np.float64)
    update_after = payload["runtime_ns5_update_after"].astype(np.float64)

    def polar(value: np.ndarray) -> np.ndarray:
        left, _, right = np.linalg.svd(value, full_matrices=False)
        return left @ right

    polar_before = polar(update_before)
    polar_after = polar(update_after)
    polar_change = np.linalg.norm(polar_after - polar_before) / max(
        np.linalg.norm(polar_before), 1.0e-30
    )
    values = {
        "float64_slice_condition_before": float(np.linalg.cond(a_before)),
        "float64_slice_condition_after": float(np.linalg.cond(a_after)),
        "float64_slice_inverse_residual_before": float(
            inverse_residual_before
        ),
        "float64_slice_inverse_residual_after": float(inverse_residual_after),
        "float64_slice_resolvent_relative_residual": float(resolvent),
        "float64_slice_svd_polar_change_on_ns5_update_slice": float(
            polar_change
        ),
    }
    values["all_values_finite"] = all(
        math.isfinite(value) for value in values.values()
    )
    return values


def replace_option(arguments: list[str], option: str, value: str) -> None:
    if option not in arguments:
        raise RuntimeError(f"worker argument is missing: {option}")
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise RuntimeError(f"worker argument has no value: {option}")
    arguments[index + 1] = value


class Recorder:
    def __init__(
        self,
        *,
        output: Path,
        protocol_path: Path,
        source_snapshot_manifest: Path,
        worker_args: argparse.Namespace,
    ) -> None:
        self.output = output
        self.protocol_path = protocol_path
        self.protocol = PROTOCOL.read_json(protocol_path)
        checks = PROTOCOL.validate_protocol(self.protocol)
        if not all(checks.values()):
            raise RuntimeError(f"protocol validation failed: {checks}")
        self.source_snapshot_manifest = source_snapshot_manifest
        snapshot = PROTOCOL.read_json(source_snapshot_manifest)
        snapshot_files = snapshot.get("files", {})
        sealed_contract = (
            source_snapshot_manifest.parent
            / "scripts/46_mdp05_confirmatory_update_shock/mdp05_contract.json"
        ).resolve()
        snapshot_checks = {
            "manifest_passed": snapshot.get("passed") is True,
            "contract_path": protocol_path.resolve() == sealed_contract,
            "contract_hash": snapshot_files.get(
                "scripts/46_mdp05_confirmatory_update_shock/mdp05_contract.json"
            )
            == sha256_file(protocol_path),
            "worker_hash": snapshot_files.get(
                "scripts/46_mdp05_confirmatory_update_shock/mdp05_worker.py"
            )
            == sha256_file(Path(__file__).resolve()),
            "metrics_hash": snapshot_files.get(
                "scripts/mdp_refresh_streaming/stream_metrics.py"
            )
            == sha256_file(
                SCRIPTS / "mdp_refresh_streaming" / "stream_metrics.py"
            ),
        }
        if not all(snapshot_checks.values()):
            raise RuntimeError(f"source snapshot integrity failed: {snapshot_checks}")
        self.snapshot_checks = snapshot_checks
        self.worker_args = worker_args
        if worker_args.analysis_tier != "formal":
            raise RuntimeError("MDP-05 instrumentation is formal-only")
        self.rows: list[dict[str, Any]] = []
        self.pending: dict[str, Any] | None = None
        self.hook_lifecycle: list[dict[str, Any]] = []
        self._bound_optimizer: Any = None
        self._source_module: Any = None
        self._original_ns: Any = None
        self._original_apply_had_override = False
        self._original_apply_value: Any = None
        self._patched_apply: Any = None
        self._patched_ns: Any = None
        self._bound_event_id: str | None = None
        (output / "validation_slices").mkdir(parents=True, exist_ok=True)

    def event_spec(self, trajectory: str, completed_step: int) -> dict[str, Any]:
        matches = [
            row
            for row in self.protocol["event_outcomes"]
            if row["trajectory_node"] == trajectory
            and int(row["completed_step"]) == int(completed_step)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"MDP-05 event not unique: {trajectory}/{completed_step}"
            )
        return matches[0]

    def start_event(self, event_spec: dict[str, Any]) -> None:
        if self.pending is not None:
            raise RuntimeError("previous refresh event is still pending")
        self.pending = {
            "spec": event_spec,
            "rows": [],
            "internal": [],
            "slices": [],
            "actual_ns_calls": 0,
        }

    def bind_optimizer(self, optimizer: Any, *, event_id: str) -> None:
        if self._bound_optimizer is not None:
            raise RuntimeError("event-scoped hooks are already active")
        self._bound_optimizer = optimizer
        source_module = sys.modules.get(optimizer.__class__.__module__)
        if source_module is None:
            raise RuntimeError("optimizer source module is unavailable")
        self._source_module = source_module
        self._original_ns = source_module.zeropower_via_newtonschulz5
        self._original_apply_had_override = (
            "_apply_preconditioners" in optimizer.__dict__
        )
        self._original_apply_value = optimizer.__dict__.get(
            "_apply_preconditioners"
        )
        original_apply = optimizer._apply_preconditioners
        recorder = self

        def patched_apply(_: Any) -> None:
            original_apply()
            recorder.validate_applied_gradients()

        self._patched_apply = types.MethodType(patched_apply, optimizer)
        optimizer._apply_preconditioners = self._patched_apply

        def patched_ns(gradient: torch.Tensor, steps: int = 5) -> torch.Tensor:
            output = recorder._original_ns(gradient, steps=steps)
            recorder.validate_actual_ns_call(gradient, output)
            return output

        self._patched_ns = patched_ns
        source_module.zeropower_via_newtonschulz5 = patched_ns
        self._bound_event_id = event_id
        self.hook_lifecycle.append({"action": "bind", "event_id": event_id})

    def unbind_optimizer(self, *, allow_pending: bool = False) -> None:
        if self._bound_optimizer is None:
            return
        if self.pending is not None and not allow_pending:
            raise RuntimeError("cannot unbind with a pending event")
        optimizer = self._bound_optimizer
        if optimizer.__dict__.get("_apply_preconditioners") is not self._patched_apply:
            raise RuntimeError("optimizer hook changed before restoration")
        if self._source_module.zeropower_via_newtonschulz5 is not self._patched_ns:
            raise RuntimeError("NS5 hook changed before restoration")
        if self._original_apply_had_override:
            optimizer.__dict__["_apply_preconditioners"] = self._original_apply_value
        else:
            del optimizer.__dict__["_apply_preconditioners"]
        self._source_module.zeropower_via_newtonschulz5 = self._original_ns
        self.hook_lifecycle.append(
            {"action": "unbind", "event_id": self._bound_event_id}
        )
        self._bound_optimizer = None
        self._source_module = None
        self._original_ns = None
        self._original_apply_had_override = False
        self._original_apply_value = None
        self._patched_apply = None
        self._patched_ns = None
        self._bound_event_id = None

    def capture_group_and_refresh(self, controller: Any, group: dict[str, Any]) -> None:
        if self.pending is None:
            raise RuntimeError("no pending event")
        optimizer = controller.optimizer
        members = list(group["members"])
        if len(members) != 1:
            raise RuntimeError(f"unexpected group membership: {group['name']}")
        parameter = members[0]
        if parameter.grad is None:
            raise RuntimeError(f"missing raw gradient: {group['name']}")
        state = optimizer.state[parameter]
        if "momentum" not in state or float(group["count"].item()) <= 0.0:
            raise RuntimeError(f"missing refresh state: {group['name']}")
        covariance_before = state["precond_cov"].detach().clone()
        inverse_before = state["precond_inv_apply"].detach().clone()
        fresh_covariance = (group["accum"] / group["count"]).detach().clone()
        optimizer._groups = [group]
        controller.original_refresh()
        covariance_after = state["precond_cov"]
        inverse_after = state["precond_inv_apply"]
        event = self.pending["spec"]
        layer = parse_layer(str(group["name"]))
        probes = self.protocol["probe_metrics"]
        calibration = self.protocol["precision_calibration"]
        selected_slice = (
            self.worker_args.cell == calibration["origin"]
            and int(self.worker_args.data_replica) == int(calibration["replica"])
            and event["event_id"] in calibration["events"]
            and layer in calibration["layers"]
        )
        probe_seed = METRICS.stable_seed(
            int(probes["base_seed"]),
            self.worker_args.cell,
            int(self.worker_args.data_replica),
            event["event_id"],
            layer,
        )
        slice_seed = METRICS.stable_seed(
            int(calibration["seed"]), event["event_id"], layer
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
            ridge_epsilon=float(
                self.protocol["matrix_contract"]["ridge_epsilon"]
            ),
            momentum_beta=float(optimizer.param_groups[0]["momentum"]),
            ns_steps=int(optimizer.param_groups[0]["ns_steps"]),
            ns_update=lambda value, steps: self._original_ns(value, steps=steps),
            probe_count=int(probes["probe_count"]),
            probe_iterations=int(probes["power_iterations"]),
            probe_seed=probe_seed,
            slice_coordinate_count=(
                int(calibration["coordinate_count"]) if selected_slice else 0
            ),
            slice_gradient_row_count=(
                int(calibration["gradient_row_count"]) if selected_slice else 0
            ),
            slice_seed=slice_seed,
        )
        row = {
            "schema_version": "mdp05_refresh_layer_event_v1",
            "origin": self.worker_args.cell,
            "data_replica": int(self.worker_args.data_replica),
            "event_id": event["event_id"],
            "trajectory_node": event["trajectory_node"],
            "completed_step": int(event["completed_step"]),
            "layer_index": layer,
            "module_id": str(group["name"]),
            "checkpoint_sha256": PROTOCOL.read_json(
                self.worker_args.checkpoint_hash_certificate
            )["sha256"],
            "source_script_sha256": sha256_file(self.worker_args.source_script),
            "execution_contract_sha256": sha256_file(self.worker_args.contract),
            "mdp05_contract_sha256": sha256_file(self.protocol_path),
            "matched_gradient_semantics": self.protocol["matrix_contract"][
                "matched_gradient_semantics"
            ],
            "raw_full_matrices_persisted": False,
            "validation_slice_persisted": selected_slice,
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
                    "event_id": event["event_id"],
                    "layer_index": layer,
                    "ridge_before": metrics["ridge_before"],
                    "ridge_after": metrics["ridge_after"],
                    "payload": slice_payload,
                }
            )
        del covariance_before, inverse_before, fresh_covariance

    def validate_applied_gradients(self) -> None:
        if self.pending is None:
            return
        for entry in self.pending["internal"]:
            parameter = entry["parameter"]
            if parameter.grad is None:
                raise RuntimeError("gradient disappeared before actual apply audit")
            observed = METRICS.tensor_fingerprint(parameter.grad)
            matched = observed["fingerprint_sha256"] == entry["gradient_after"][
                "fingerprint_sha256"
            ]
            entry["row"]["actual_preconditioned_gradient_fingerprint_match"] = matched
            if not matched:
                raise RuntimeError(
                    "shadow/actual preconditioned-gradient mismatch: "
                    f"{entry['row']['module_id']}"
                )

    def validate_actual_ns_call(
        self, ns_input: torch.Tensor, ns_output: torch.Tensor
    ) -> None:
        if self.pending is None:
            return
        if tuple(ns_input.shape) != tuple(
            self.protocol["matrix_contract"]["gradient_shape"]
        ):
            return
        fingerprint = METRICS.tensor_fingerprint(ns_input)
        matches = [
            entry
            for entry in self.pending["internal"]
            if not entry["row"]["actual_ns_input_fingerprint_match"]
            and fingerprint["fingerprint_sha256"]
            == entry["ns_input_after"]["fingerprint_sha256"]
        ]
        if len(matches) != 1:
            raise RuntimeError("actual NS5 input did not match one shadow down input")
        entry = matches[0]
        output = METRICS.tensor_fingerprint(ns_output)
        output_match = output["fingerprint_sha256"] == entry["ns_output_after"][
            "fingerprint_sha256"
        ]
        entry["row"]["actual_ns_input_fingerprint_match"] = True
        entry["row"]["actual_ns_output_fingerprint_match"] = output_match
        self.pending["actual_ns_calls"] += 1
        if not output_match:
            raise RuntimeError(
                f"shadow/actual NS5 output mismatch: {entry['row']['module_id']}"
            )

    def finish_event_after_optimizer_step(self, event_id: str) -> None:
        if self.pending is None or self.pending["spec"]["event_id"] != event_id:
            raise RuntimeError(f"pending event mismatch: {event_id}")
        layers = self.protocol["design"]["layer_indices"]
        rows = self.pending["rows"]
        if sorted(int(row["layer_index"]) for row in rows) != layers:
            raise RuntimeError(f"wrong layer coverage for {event_id}")
        if int(self.pending["actual_ns_calls"]) != len(layers):
            raise RuntimeError(f"wrong actual NS5 call count for {event_id}")
        required = (
            "actual_preconditioned_gradient_fingerprint_match",
            "actual_ns_input_fingerprint_match",
            "actual_ns_output_fingerprint_match",
        )
        if not all(all(row[field] is True for field in required) for row in rows):
            raise RuntimeError(f"actual update audit failed for {event_id}")
        for item in self.pending["slices"]:
            name = (
                f"{self.worker_args.cell}_replica{self.worker_args.data_replica}_"
                f"{item['event_id']}_layer{item['layer_index']}.npz"
            )
            path = self.output / "validation_slices" / name
            np.savez_compressed(path, **item["payload"])
            atomic_json(
                path.with_suffix(".json"),
                {
                    "schema_version": "mdp05_float64_slice_input_v1",
                    "origin": self.worker_args.cell,
                    "data_replica": int(self.worker_args.data_replica),
                    "event_id": item["event_id"],
                    "layer_index": int(item["layer_index"]),
                    "npz": path.name,
                    "npz_sha256": sha256_file(path),
                    "paper_primary_claim_eligible": False,
                    "warning": self.protocol["precision_calibration"]["warning"],
                },
            )
            diagnostics = float64_slice_diagnostics(
                item["payload"], item["ridge_before"], item["ridge_after"]
            )
            atomic_json(
                path.with_name(path.stem + "_float64.json"),
                {
                    "schema_version": "mdp05_float64_slice_diagnostic_v1",
                    "origin": self.worker_args.cell,
                    "data_replica": int(self.worker_args.data_replica),
                    "event_id": item["event_id"],
                    "layer_index": int(item["layer_index"]),
                    "coordinate_count": int(
                        self.protocol["precision_calibration"]["coordinate_count"]
                    ),
                    "paper_primary_claim_eligible": False,
                    "warning": self.protocol["precision_calibration"]["warning"],
                    **diagnostics,
                },
            )
        write_csv(self.output / f"{event_id}_layer_metrics.csv", rows)
        atomic_json(
            self.output / f"{event_id}_manifest.json",
            {
                "schema_version": "mdp05_event_manifest_v1",
                "event_id": event_id,
                "rows": len(rows),
                "layers": sorted(int(row["layer_index"]) for row in rows),
                "actual_update_fingerprints_exact": True,
                "all_values_finite": all(
                    row["all_full_state_values_finite"] is True for row in rows
                ),
                "passed": True,
            },
        )
        self.rows.extend(rows)
        self.pending = None

    @staticmethod
    def reduction_reconciles(row: dict[str, Any]) -> bool:
        pairs = (
            (
                "matched_g_preconditioned_fro_before",
                "matched_g_preconditioned_delta_fro",
                "matched_g_preconditioned_relative_change",
            ),
            (
                "runtime_ns5_update_fro_before",
                "runtime_ns5_update_delta_fro",
                "runtime_ns5_update_relative_change",
            ),
        )
        for before, delta, relative in pairs:
            expected = float(row[delta]) / max(float(row[before]), 1.0e-30)
            if not math.isclose(
                expected, float(row[relative]), rel_tol=2.0e-6, abs_tol=2.0e-8
            ):
                return False
        return True

    def finalize(self) -> dict[str, Any]:
        expected_rows = len(self.protocol["event_outcomes"]) * len(
            self.protocol["design"]["layer_indices"]
        )
        gates = self.protocol["hard_gates"]
        if self.pending is not None or self._bound_optimizer is not None:
            raise RuntimeError("event hooks were not fully restored")
        expected_lifecycle = []
        for event in PROTOCOL.EVENTS:
            expected_lifecycle.extend(
                [{"action": "bind", "event_id": event}, {"action": "unbind", "event_id": event}]
            )
        checks = {
            "row_count": len(self.rows) == expected_rows,
            "coverage": {
                (row["event_id"], int(row["layer_index"])) for row in self.rows
            }
            == {
                (event, layer)
                for event in PROTOCOL.EVENTS
                for layer in self.protocol["design"]["layer_indices"]
            },
            "finite": all(row["all_full_state_values_finite"] for row in self.rows),
            "actual_gradient": all(
                row["actual_preconditioned_gradient_fingerprint_match"]
                for row in self.rows
            ),
            "actual_ns5": all(
                row["actual_ns_input_fingerprint_match"]
                and row["actual_ns_output_fingerprint_match"]
                for row in self.rows
            ),
            "covariance_refresh_identity": max(
                float(row["covariance_refresh_identity_relative_residual"])
                for row in self.rows
            )
            <= float(gates["covariance_refresh_identity_relative_residual_max"]),
            "k_asymmetry": max(
                max(float(row["k_asymmetry_before"]), float(row["k_asymmetry_after"]))
                for row in self.rows
            )
            <= float(gates["k_asymmetry_relative_max"]),
            "inverse_asymmetry": max(
                max(
                    float(row["inverse_asymmetry_before"]),
                    float(row["inverse_asymmetry_after"]),
                )
                for row in self.rows
            )
            <= float(gates["inverse_asymmetry_relative_max"]),
            "inverse_backward_residual": max(
                max(
                    float(row["runtime_inverse_backward_residual_before"]),
                    float(row["runtime_inverse_backward_residual_after"]),
                )
                for row in self.rows
            )
            <= float(gates["runtime_inverse_backward_residual_max"]),
            "full_state_reduction_reconciliation": all(
                self.reduction_reconciles(row) for row in self.rows
            ),
            "event_scoped_hooks": self.hook_lifecycle == expected_lifecycle,
            "resolvent_not_hard_gate": gates[
                "runtime_resolvent_relative_residual_is_hard_gate"
            ]
            is False,
        }
        write_csv(self.output / "mdp05_refresh_layer_metrics.csv", self.rows)
        jsonl = self.output / "mdp05_refresh_layer_metrics.jsonl"
        temporary = jsonl.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in self.rows
            ),
            encoding="utf-8",
        )
        os.replace(temporary, jsonl)
        base_manifest_path = self.output / "mech09r_manifest.json"
        base_manifest = PROTOCOL.read_json(base_manifest_path)
        scientific_artifacts = sorted(
            path.relative_to(self.output).as_posix()
            for path in self.output.rglob("*")
            if path.is_file()
            and path.name
            not in {
                "worker.log",
                "status.json",
                "mdp05_status.json",
                "mdp05_unit_manifest.json",
            }
        )
        artifact_hashes = {
            name: sha256_file(self.output / name) for name in scientific_artifacts
        }
        passed = all(checks.values()) and base_manifest.get("passed") is True
        manifest = {
            "schema_version": "mdp05_unit_manifest_v1",
            "script_version": SCRIPT_VERSION,
            "origin": self.worker_args.cell,
            "data_replica": int(self.worker_args.data_replica),
            "mdp05_contract_sha256": sha256_file(self.protocol_path),
            "execution_contract_sha256": sha256_file(self.worker_args.contract),
            "source_snapshot_manifest": str(self.source_snapshot_manifest),
            "source_snapshot_manifest_sha256": sha256_file(
                self.source_snapshot_manifest
            ),
            "base_outcome_manifest": "mech09r_manifest.json",
            "base_outcome_manifest_sha256": sha256_file(base_manifest_path),
            "base_outcomes_computed_in_same_unit": True,
            "source_experiment_outcomes_read": False,
            "layer_event_rows": len(self.rows),
            "resolvent_diagnostic_max": max(
                float(row["runtime_resolvent_relative_residual"])
                for row in self.rows
            ),
            "resolvent_diagnostic_passed_old_mdp04_threshold": max(
                float(row["runtime_resolvent_relative_residual"])
                for row in self.rows
            )
            <= 0.01,
            "checks": checks,
            "source_snapshot_checks": self.snapshot_checks,
            "hook_lifecycle": self.hook_lifecycle,
            "scientific_artifacts": scientific_artifacts,
            "scientific_artifact_sha256": artifact_hashes,
            "growing_worker_log_excluded": True,
            "passed": passed,
        }
        atomic_json(self.output / "mdp05_unit_manifest.json", manifest)
        atomic_json(
            self.output / "mdp05_status.json",
            {
                "status": "passed" if passed else "integrity_failed",
                "script_version": SCRIPT_VERSION,
            },
        )
        if not passed:
            raise RuntimeError(f"MDP-05 unit integrity gates failed: {checks}")
        return manifest


RECORDER: Recorder | None = None
CURRENT_TRAJECTORY: str | None = None


class InstrumentedController(LEGACY.RefreshInterventionController):
    @torch.no_grad()
    def handle_refresh(self) -> None:
        if RECORDER is None or CURRENT_TRAJECTORY is None:
            raise RuntimeError("MDP-05 instrumentation context is missing")
        completed_step = int(self.optimizer.global_step) + 1
        action = LEGACY.refresh_action(completed_step, self.target_refresh_steps)
        selected = (
            (CURRENT_TRAJECTORY == "production" and completed_step == 32)
            or (CURRENT_TRAJECTORY == "delayed" and completed_step == 64)
        ) and action == "refresh"
        if not selected:
            return super().handle_refresh()
        event = RECORDER.event_spec(CURRENT_TRAJECTORY, completed_step)
        RECORDER.bind_optimizer(self.optimizer, event_id=event["event_id"])
        RECORDER.start_event(event)
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
                RECORDER.capture_group_and_refresh(self, group)
        finally:
            self.optimizer._groups = self.all_groups
        target_after = LEGACY.group_state_snapshot(
            self.optimizer, self.target_groups, include_statistics=True
        )
        other_after = LEGACY.group_state_snapshot(
            self.optimizer, self.other_groups, include_statistics=True
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


def install_hooks() -> None:
    original_run_segment = WORKER.run_segment

    def instrumented_run_segment(**kwargs: Any) -> Any:
        global CURRENT_TRAJECTORY
        if CURRENT_TRAJECTORY is not None:
            raise RuntimeError("nested trajectory instrumentation")
        CURRENT_TRAJECTORY = str(kwargs["trajectory_node"])
        optimizer = kwargs["matrix_optimizer"]
        step_had_override = "step" in optimizer.__dict__
        step_override = optimizer.__dict__.get("step")
        original_step = optimizer.step

        def step_with_mdp05_boundary(_: Any, *args: Any, **step_kwargs: Any) -> Any:
            result = original_step(*args, **step_kwargs)
            if RECORDER is not None and RECORDER.pending is not None:
                event_id = str(RECORDER.pending["spec"]["event_id"])
                try:
                    RECORDER.finish_event_after_optimizer_step(event_id)
                finally:
                    RECORDER.unbind_optimizer(allow_pending=True)
            return result

        patched_step = types.MethodType(step_with_mdp05_boundary, optimizer)
        optimizer.step = patched_step
        try:
            return original_run_segment(**kwargs)
        finally:
            if RECORDER is not None:
                RECORDER.unbind_optimizer(allow_pending=True)
            if optimizer.__dict__.get("step") is not patched_step:
                raise RuntimeError("optimizer step boundary hook changed before restoration")
            if step_had_override:
                optimizer.__dict__["step"] = step_override
            else:
                del optimizer.__dict__["step"]
            CURRENT_TRAJECTORY = None

    WORKER.run_segment = instrumented_run_segment
    WORKER.LEGACY.RefreshInterventionController = InstrumentedController


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mdp05-output-dir", required=True, type=Path)
    parser.add_argument("--mdp05-contract", required=True, type=Path)
    parser.add_argument("--source-snapshot-manifest", required=True, type=Path)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    arguments = list(parsed.worker_args)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if not arguments:
        raise RuntimeError("MECH-09R worker arguments are required after --")
    return parsed, arguments


def main() -> int:
    global RECORDER
    args, worker_arguments = parse_args()
    output = args.mdp05_output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        replace_option(worker_arguments, "--output-dir", str(output))
        sys.argv = [str(WORKER.__file__), *worker_arguments]
        worker_args = WORKER.parse_args()
        if worker_args.analysis_tier != "formal":
            raise RuntimeError("MDP-05 worker accepts formal units only")
        RECORDER = Recorder(
            output=output,
            protocol_path=args.mdp05_contract.resolve(),
            source_snapshot_manifest=args.source_snapshot_manifest.resolve(),
            worker_args=worker_args,
        )
        install_hooks()
        WORKER.run_worker(worker_args)
        manifest = RECORDER.finalize()
        print(
            f"MDP-05 unit passed origin={manifest['origin']} "
            f"replica={manifest['data_replica']} rows={manifest['layer_event_rows']}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        if RECORDER is not None:
            try:
                RECORDER.unbind_optimizer(allow_pending=True)
            except BaseException:
                pass
        atomic_json(
            output / "mdp05_status.json",
            {
                "status": "failed",
                "script_version": SCRIPT_VERSION,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(
            f"MDP-05 unit failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
