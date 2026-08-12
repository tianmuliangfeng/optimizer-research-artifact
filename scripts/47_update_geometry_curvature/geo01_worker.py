#!/usr/bin/env python3
"""Run one source-pinned LLaMA GEO-01 pilot unit on an H100."""

from __future__ import annotations

import argparse
import csv
import gc
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

import torch


SCRIPT_VERSION = "2026-08-04.1"
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


P = load_module("geo01_worker_protocol", HERE / "protocol.py")
G = load_module("geo01_worker_geometry_core", HERE / "geometry_core.py")
METRICS = load_module(
    "geo01_stream_metrics",
    SCRIPTS / "mdp_refresh_streaming" / "stream_metrics.py",
)
WORKER = load_module(
    "geo01_accepted_mech09r_worker",
    SCRIPTS / "37_mech09_downproj_refresh_mediation" / "mech09r_worker.py",
)
LEGACY = WORKER.LEGACY
BASE_CONTROLLER = LEGACY.RefreshInterventionController


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty GEO-01 CSV")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen and not isinstance(row[field], (dict, list)):
                seen.add(field)
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    os.replace(temporary, path)


def parse_layer(group_name: str) -> int:
    parts = group_name.split(".")
    if len(parts) < 3 or parts[0] != "layers" or parts[-1] != "down_input":
        raise RuntimeError(f"unexpected down-projection group: {group_name}")
    return int(parts[1])


def replace_option(arguments: list[str], option: str, value: str) -> None:
    if option not in arguments:
        raise RuntimeError(f"worker argument is missing: {option}")
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise RuntimeError(f"worker argument has no value: {option}")
    arguments[index + 1] = value


class GeometryRecorder:
    def __init__(
        self,
        *,
        output: Path,
        contract_path: Path,
        snapshot_manifest_path: Path,
        worker_args: argparse.Namespace,
    ) -> None:
        self.output = output
        self.contract_path = contract_path
        self.contract = P.read_json(contract_path)
        checks = P.validate_contract(self.contract)
        if not all(checks.values()):
            raise RuntimeError(f"GEO-01 contract validation failed: {checks}")
        self.worker_args = worker_args
        pilot = self.contract["pilot"]
        unit_checks = {
            "formal_tier": worker_args.analysis_tier == "formal",
            "origin": worker_args.cell in pilot["origins"],
            "replica": int(worker_args.data_replica) in pilot["data_replicas"],
        }
        if not all(unit_checks.values()):
            raise RuntimeError(f"GEO-01 pilot unit mismatch: {unit_checks}")
        self.snapshot_manifest_path = snapshot_manifest_path
        snapshot = P.read_json(snapshot_manifest_path)
        files = snapshot.get("files", {})
        expected_paths = {
            "scripts/47_update_geometry_curvature/geo01_contract.json": contract_path,
            "scripts/47_update_geometry_curvature/geometry_core.py": Path(G.__file__),
            "scripts/47_update_geometry_curvature/geo01_worker.py": Path(__file__),
            "scripts/37_mech09_downproj_refresh_mediation/mech09r_worker.py": Path(WORKER.__file__),
            "scripts/mdp_refresh_streaming/stream_metrics.py": Path(METRICS.__file__),
        }
        snapshot_checks = {
            relative: relative in files
            and sha256_file(path.resolve()) == files[relative]
            for relative, path in expected_paths.items()
        }
        snapshot_checks["manifest_passed"] = snapshot.get("passed") is True
        if not all(snapshot_checks.values()):
            raise RuntimeError(f"source snapshot integrity failed: {snapshot_checks}")
        self.snapshot_checks = snapshot_checks
        self.rows: list[dict[str, Any]] = []
        self.pending: dict[str, Any] | None = None
        self._bound_optimizer: Any = None
        self._source_module: Any = None
        self._original_ns: Any = None
        self._patched_ns: Any = None
        self._patched_apply: Any = None
        self._original_apply_had_override = False
        self._original_apply_value: Any = None
        self.hook_lifecycle: list[dict[str, Any]] = []

    def event_selected(self, trajectory: str, completed_step: int) -> bool:
        pilot = self.contract["pilot"]
        return (
            trajectory == "production"
            and int(completed_step) == int(pilot["completed_step"])
        )

    def start_event(self) -> None:
        if self.pending is not None:
            raise RuntimeError("a GEO-01 event is already pending")
        self.pending = {
            "event_id": self.contract["pilot"]["event_id"],
            "entries": [],
            "geometry_measured": False,
            "actual_ns_matches": 0,
        }

    def bind_optimizer(self, optimizer: Any) -> None:
        if self._bound_optimizer is not None:
            raise RuntimeError("GEO-01 optimizer hooks are already active")
        self._bound_optimizer = optimizer
        source_module = sys.modules.get(optimizer.__class__.__module__)
        if source_module is None:
            raise RuntimeError("source optimizer module is unavailable")
        self._source_module = source_module
        self._original_ns = source_module.zeropower_via_newtonschulz5
        self._original_apply_had_override = "_apply_preconditioners" in optimizer.__dict__
        self._original_apply_value = optimizer.__dict__.get("_apply_preconditioners")
        original_apply = optimizer._apply_preconditioners
        recorder = self

        def patched_apply(_: Any) -> None:
            original_apply()
            recorder.validate_applied_gradients()

        self._patched_apply = types.MethodType(patched_apply, optimizer)
        optimizer._apply_preconditioners = self._patched_apply

        def patched_ns(gradient: torch.Tensor, steps: int = 5) -> torch.Tensor:
            result = recorder._original_ns(gradient, steps=steps)
            recorder.validate_actual_ns(gradient, result)
            return result

        self._patched_ns = patched_ns
        source_module.zeropower_via_newtonschulz5 = patched_ns
        self.hook_lifecycle.append({"action": "bind"})

    def unbind_optimizer(self, *, allow_pending: bool = False) -> None:
        if self._bound_optimizer is None:
            return
        if self.pending is not None and not allow_pending:
            raise RuntimeError("cannot unbind while a GEO-01 event is pending")
        optimizer = self._bound_optimizer
        if optimizer.__dict__.get("_apply_preconditioners") is not self._patched_apply:
            raise RuntimeError("optimizer apply hook changed before restoration")
        if self._source_module.zeropower_via_newtonschulz5 is not self._patched_ns:
            raise RuntimeError("source NS5 hook changed before restoration")
        if self._original_apply_had_override:
            optimizer.__dict__["_apply_preconditioners"] = self._original_apply_value
        else:
            del optimizer.__dict__["_apply_preconditioners"]
        self._source_module.zeropower_via_newtonschulz5 = self._original_ns
        self.hook_lifecycle.append({"action": "unbind"})
        self._bound_optimizer = None
        self._source_module = None
        self._original_ns = None
        self._patched_ns = None
        self._patched_apply = None

    def capture_group_and_refresh(self, controller: Any, group: dict[str, Any]) -> None:
        if self.pending is None:
            raise RuntimeError("no pending GEO-01 event")
        members = list(group["members"])
        if len(members) != 1:
            raise RuntimeError(f"unexpected group membership: {group['name']}")
        parameter = members[0]
        layer = parse_layer(str(group["name"]))
        target_layers = set(int(value) for value in self.contract["pilot"]["target_layers"])
        optimizer = controller.optimizer
        state = optimizer.state[parameter]
        if parameter.grad is None or "momentum" not in state:
            raise RuntimeError(f"missing gradient/momentum for {group['name']}")
        selected = layer in target_layers
        inverse_before = (
            state["precond_inv_apply"].detach().clone() if selected else None
        )
        raw_gradient = parameter.grad.detach().clone() if selected else None
        historical_momentum = state["momentum"].detach().clone() if selected else None
        optimizer._groups = [group]
        controller.original_refresh()
        if not selected:
            return
        inverse_after = state["precond_inv_apply"].detach().clone()
        direction, audit = G.counterfactual_update_direction(
            raw_gradient=raw_gradient,
            historical_momentum=historical_momentum,
            inverse_reference=inverse_before,
            inverse_treatment=inverse_after,
            momentum_beta=float(optimizer.param_groups[0]["momentum"]),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            ns_steps=int(optimizer.param_groups[0]["ns_steps"]),
            ns_update=lambda value, steps: self._original_ns(value, steps=steps),
            fingerprint_fn=METRICS.tensor_fingerprint,
        )
        name_matches = [
            name
            for name, candidate in ACTIVE_MODEL.named_parameters()
            if candidate is parameter
        ]
        if len(name_matches) != 1:
            raise RuntimeError(f"cannot identify model parameter for {group['name']}")
        self.pending["entries"].append(
            {
                "layer": layer,
                "group_name": str(group["name"]),
                "parameter_name": name_matches[0],
                "parameter": parameter,
                "direction": direction,
                "audit": audit,
                "expected_gradient": audit["gradient_treatment"]["fingerprint_sha256"],
                "expected_ns_input": audit["ns_input_treatment"]["fingerprint_sha256"],
                "expected_ns_output": audit["update_treatment"]["fingerprint_sha256"],
                "actual_gradient_match": False,
                "actual_ns_input_match": False,
                "actual_ns_output_match": False,
            }
        )
        del inverse_before, inverse_after, raw_gradient, historical_momentum

    def measure_geometry(self) -> None:
        if self.pending is None or self.pending["geometry_measured"]:
            raise RuntimeError("GEO-01 geometry event state is invalid")
        entries = self.pending["entries"]
        expected_layers = sorted(int(value) for value in self.contract["pilot"]["target_layers"])
        if sorted(int(entry["layer"]) for entry in entries) != expected_layers:
            raise RuntimeError("GEO-01 target-layer direction coverage failed")
        geometry = self.contract["geometry"]
        batch_count = int(geometry["heldout_microbatches"])
        batch_size = int(geometry["microbatch_size"])
        sequence_length = int(geometry["sequence_length"])
        if len(ACTIVE_EVAL_BATCHES) < batch_count:
            raise RuntimeError("not enough held-out batches for GEO-01")
        batches = [
            (
                x[:batch_size, :sequence_length].contiguous(),
                y[:batch_size, :sequence_length].contiguous(),
            )
            for x, y in ACTIVE_EVAL_BATCHES[:batch_count]
        ]
        batch_values_before = [
            {
                "x": METRICS.tensor_fingerprint(x),
                "y": METRICS.tensor_fingerprint(y),
            }
            for x, y in batches
        ]
        grad_before = {
            entry["parameter_name"]: METRICS.tensor_fingerprint(entry["parameter"].grad)
            for entry in entries
        }
        state_before = {
            entry["parameter_name"]: {
                "momentum": METRICS.tensor_fingerprint(
                    self._bound_optimizer.state[entry["parameter"]]["momentum"]
                ),
                "inverse": METRICS.tensor_fingerprint(
                    self._bound_optimizer.state[entry["parameter"]]["precond_inv_apply"]
                ),
            }
            for entry in entries
        }
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for scope in self.contract["pilot"]["scopes"]:
                layers = set(int(value) for value in scope["layers"])
                named_direction = {
                    entry["parameter_name"]: entry["direction"]
                    for entry in entries
                    if int(entry["layer"]) in layers
                }
                result = G.measure_directional_geometry(
                    model=ACTIVE_MODEL,
                    batches=batches,
                    named_direction=named_direction,
                    forward_kwargs={"return_logits": False, "precond_flag": False},
                    fd_target_relative_parameter_norm=float(
                        geometry["fd_target_relative_parameter_norm"]
                    ),
                    fd_scale_min=float(geometry["fd_scale_min"]),
                    fd_scale_max=float(geometry["fd_scale_max"]),
                )
                self.rows.append(
                    {
                        "schema_version": "geo01_pilot_scope_row_v1",
                        "phase": "pilot",
                        "origin": self.worker_args.cell,
                        "data_replica": int(self.worker_args.data_replica),
                        "event_id": self.contract["pilot"]["event_id"],
                        "completed_step": int(self.contract["pilot"]["completed_step"]),
                        "scope_id": scope["scope_id"],
                        "layers": list(scope["layers"]),
                        "contract_sha256": sha256_file(self.contract_path),
                        "source_execution_contract_sha256": sha256_file(
                            self.worker_args.contract
                        ),
                        **result,
                    }
                )
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        grad_after = {
            entry["parameter_name"]: METRICS.tensor_fingerprint(entry["parameter"].grad)
            for entry in entries
        }
        state_after = {
            entry["parameter_name"]: {
                "momentum": METRICS.tensor_fingerprint(
                    self._bound_optimizer.state[entry["parameter"]]["momentum"]
                ),
                "inverse": METRICS.tensor_fingerprint(
                    self._bound_optimizer.state[entry["parameter"]]["precond_inv_apply"]
                ),
            }
            for entry in entries
        }
        batch_values_after = [
            {
                "x": METRICS.tensor_fingerprint(x),
                "y": METRICS.tensor_fingerprint(y),
            }
            for x, y in batches
        ]
        self.pending["probe_invariance"] = {
            "raw_gradients_unchanged": grad_before == grad_after,
            "optimizer_state_unchanged": state_before == state_after,
            "heldout_batch_values_unchanged": batch_values_before
            == batch_values_after,
            "loader_iterator_advanced_by_probe": False,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
        self.pending["geometry_measured"] = True
        for entry in entries:
            entry["direction"] = None
        del batches
        gc.collect()

    def validate_applied_gradients(self) -> None:
        if self.pending is None:
            return
        for entry in self.pending["entries"]:
            observed = METRICS.tensor_fingerprint(entry["parameter"].grad)
            matched = observed["fingerprint_sha256"] == entry["expected_gradient"]
            entry["actual_gradient_match"] = matched
            if not matched:
                raise RuntimeError(
                    f"GEO-01 shadow/actual gradient mismatch: {entry['group_name']}"
                )

    def validate_actual_ns(self, ns_input: torch.Tensor, ns_output: torch.Tensor) -> None:
        if self.pending is None:
            return
        input_fp = METRICS.tensor_fingerprint(ns_input)
        matches = [
            entry
            for entry in self.pending["entries"]
            if not entry["actual_ns_input_match"]
            and input_fp["fingerprint_sha256"] == entry["expected_ns_input"]
        ]
        if not matches:
            return
        if len(matches) != 1:
            raise RuntimeError("GEO-01 actual NS5 input matched multiple layers")
        entry = matches[0]
        output_fp = METRICS.tensor_fingerprint(ns_output)
        entry["actual_ns_input_match"] = True
        entry["actual_ns_output_match"] = (
            output_fp["fingerprint_sha256"] == entry["expected_ns_output"]
        )
        self.pending["actual_ns_matches"] += 1
        if not entry["actual_ns_output_match"]:
            raise RuntimeError(
                f"GEO-01 shadow/actual NS5 mismatch: {entry['group_name']}"
            )

    def finish_after_optimizer_step(self) -> None:
        if self.pending is None or not self.pending["geometry_measured"]:
            raise RuntimeError("GEO-01 event did not measure geometry")
        entries = self.pending["entries"]
        invariance = self.pending["probe_invariance"]
        checks = {
            "target_layers": len(entries) == len(self.contract["pilot"]["target_layers"]),
            "actual_gradients": all(entry["actual_gradient_match"] for entry in entries),
            "actual_ns_inputs": all(entry["actual_ns_input_match"] for entry in entries),
            "actual_ns_outputs": all(entry["actual_ns_output_match"] for entry in entries),
            "actual_ns_count": int(self.pending["actual_ns_matches"]) == len(entries),
            "geometry_rows": len(self.rows) == len(self.contract["pilot"]["scopes"]),
            "geometry_finite": all(row["all_values_finite"] for row in self.rows),
            "parameters_unchanged": all(row["parameters_unchanged"] for row in self.rows),
            "gradient_unchanged_by_probe": invariance["raw_gradients_unchanged"],
            "optimizer_unchanged_by_probe": invariance["optimizer_state_unchanged"],
            "heldout_batches_unchanged": invariance[
                "heldout_batch_values_unchanged"
            ],
            "loader_not_advanced": invariance[
                "loader_iterator_advanced_by_probe"
            ]
            is False,
            "memory": int(invariance["peak_allocated_bytes"])
            <= int(self.contract["pilot_hard_gates"]["peak_allocated_bytes_max"]),
            "baseline_graph": max(
                float(row["baseline_graph_relative_error"]) for row in self.rows
            )
            <= float(
                self.contract["pilot_hard_gates"]["baseline_graph_relative_error_max"]
            ),
        }
        audit_rows = [
            {
                "layer": int(entry["layer"]),
                "group_name": entry["group_name"],
                "parameter_name": entry["parameter_name"],
                "direction_audit": entry["audit"],
                "actual_gradient_match": entry["actual_gradient_match"],
                "actual_ns_input_match": entry["actual_ns_input_match"],
                "actual_ns_output_match": entry["actual_ns_output_match"],
            }
            for entry in entries
        ]
        atomic_json(
            self.output / "direction_construction_audit.json",
            {"rows": audit_rows, "passed": all(checks.values())},
        )
        atomic_json(
            self.output / "geometry_event_audit.json",
            {
                "event_id": self.pending["event_id"],
                "probe_invariance": invariance,
                "checks": checks,
                "passed": all(checks.values()),
            },
        )
        if not all(checks.values()):
            raise RuntimeError(f"GEO-01 event integrity failed: {checks}")
        self.pending = None

    def finalize(self) -> dict[str, Any]:
        if self.pending is not None or self._bound_optimizer is not None:
            raise RuntimeError("GEO-01 hooks were not fully closed")
        base_path = self.output / "mech09r_manifest.json"
        base = P.read_json(base_path)
        write_jsonl(self.output / "geo01_geometry_rows.jsonl", self.rows)
        write_csv(self.output / "geo01_geometry_rows.csv", self.rows)
        expected_scopes = sorted(
            row["scope_id"] for row in self.contract["pilot"]["scopes"]
        )
        checks = {
            "base_worker": base.get("passed") is True,
            "row_count": len(self.rows) == len(expected_scopes),
            "scope_coverage": sorted(row["scope_id"] for row in self.rows)
            == expected_scopes,
            "finite": all(row["all_values_finite"] for row in self.rows),
            "parameters_unchanged": all(row["parameters_unchanged"] for row in self.rows),
            "hook_lifecycle": self.hook_lifecycle
            == [{"action": "bind"}, {"action": "unbind"}],
            "pilot_not_claim_eligible": self.contract["claim_boundary"][
                "pilot_claim_eligible"
            ]
            is False,
        }
        artifacts = [
            "direction_construction_audit.json",
            "geo01_geometry_rows.csv",
            "geo01_geometry_rows.jsonl",
            "geometry_event_audit.json",
            "mech09r_manifest.json",
        ]
        manifest = {
            "schema_version": "geo01_pilot_unit_manifest_v1",
            "script_version": SCRIPT_VERSION,
            "experiment": "GEO-01",
            "phase": "pilot",
            "origin": self.worker_args.cell,
            "data_replica": int(self.worker_args.data_replica),
            "contract_sha256": sha256_file(self.contract_path),
            "execution_contract_sha256": sha256_file(self.worker_args.contract),
            "source_snapshot_manifest_sha256": sha256_file(
                self.snapshot_manifest_path
            ),
            "rows": len(self.rows),
            "scopes": sorted(row["scope_id"] for row in self.rows),
            "scientific_outcomes_opened_for_metric_selection": False,
            "claim_eligible": False,
            "full_direction_persisted": False,
            "full_hessian_constructed": False,
            "artifacts": artifacts,
            "artifact_sha256": {
                name: sha256_file(self.output / name) for name in artifacts
            },
            "checks": checks,
            "snapshot_checks": self.snapshot_checks,
            "passed": all(checks.values()),
        }
        atomic_json(self.output / "geo01_unit_manifest.json", manifest)
        atomic_json(
            self.output / "geo01_status.json",
            {"status": "passed" if manifest["passed"] else "failed"},
        )
        if not manifest["passed"]:
            raise RuntimeError(f"GEO-01 final integrity failed: {checks}")
        return manifest


RECORDER: GeometryRecorder | None = None
ACTIVE_TRAJECTORY: str | None = None
ACTIVE_MODEL: Any = None
ACTIVE_EVAL_BATCHES: list[tuple[torch.Tensor, torch.Tensor]] = []


class GeometryController(BASE_CONTROLLER):
    @torch.no_grad()
    def handle_refresh(self) -> None:
        if RECORDER is None or ACTIVE_TRAJECTORY is None:
            raise RuntimeError("GEO-01 instrumentation context is unavailable")
        completed_step = int(self.optimizer.global_step) + 1
        action = LEGACY.refresh_action(completed_step, self.target_refresh_steps)
        if not RECORDER.event_selected(ACTIVE_TRAJECTORY, completed_step) or action != "refresh":
            return super().handle_refresh()
        RECORDER.bind_optimizer(self.optimizer)
        RECORDER.start_event()
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
            RECORDER.measure_geometry()
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
        global ACTIVE_TRAJECTORY, ACTIVE_MODEL, ACTIVE_EVAL_BATCHES
        if ACTIVE_TRAJECTORY is not None:
            raise RuntimeError("nested GEO-01 trajectory instrumentation")
        ACTIVE_TRAJECTORY = str(kwargs["trajectory_node"])
        ACTIVE_MODEL = kwargs["model"]
        ACTIVE_EVAL_BATCHES = list(kwargs["eval_batches"])
        optimizer = kwargs["matrix_optimizer"]
        step_had_override = "step" in optimizer.__dict__
        step_override = optimizer.__dict__.get("step")
        original_step = optimizer.step

        def step_with_geo_boundary(_: Any, *args: Any, **step_kwargs: Any) -> Any:
            result = original_step(*args, **step_kwargs)
            if RECORDER is not None and RECORDER.pending is not None:
                try:
                    RECORDER.finish_after_optimizer_step()
                finally:
                    RECORDER.unbind_optimizer(allow_pending=True)
            return result

        patched_step = types.MethodType(step_with_geo_boundary, optimizer)
        optimizer.step = patched_step
        try:
            return original_run_segment(**kwargs)
        finally:
            if RECORDER is not None:
                RECORDER.unbind_optimizer(allow_pending=True)
            if optimizer.__dict__.get("step") is not patched_step:
                raise RuntimeError("optimizer step hook changed before restoration")
            if step_had_override:
                optimizer.__dict__["step"] = step_override
            else:
                del optimizer.__dict__["step"]
            ACTIVE_TRAJECTORY = None
            ACTIVE_MODEL = None
            ACTIVE_EVAL_BATCHES = []

    WORKER.run_segment = instrumented_run_segment
    WORKER.LEGACY.RefreshInterventionController = GeometryController


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geo01-output-dir", required=True, type=Path)
    parser.add_argument("--geo01-contract", required=True, type=Path)
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
    output = args.geo01_output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        replace_option(worker_arguments, "--output-dir", str(output))
        sys.argv = [str(WORKER.__file__), *worker_arguments]
        worker_args = WORKER.parse_args()
        RECORDER = GeometryRecorder(
            output=output,
            contract_path=args.geo01_contract.resolve(),
            snapshot_manifest_path=args.source_snapshot_manifest.resolve(),
            worker_args=worker_args,
        )
        install_hooks()
        WORKER.run_worker(worker_args)
        manifest = RECORDER.finalize()
        print(
            f"GEO-01 pilot passed origin={manifest['origin']} "
            f"replica={manifest['data_replica']} rows={manifest['rows']}",
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
            output / "geo01_status.json",
            {
                "status": "failed",
                "script_version": SCRIPT_VERSION,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(
            f"GEO-01 pilot failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
