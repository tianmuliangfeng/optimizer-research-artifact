#!/usr/bin/env python3
"""Run one source-pinned GEO-01B origin-by-replica discovery unit."""

from __future__ import annotations

import argparse
import copy
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


P = load_module("geo01b_worker_protocol", HERE / "protocol.py")
PILOT = load_module(
    "geo01b_accepted_geo01a_worker",
    SCRIPTS / "47_update_geometry_curvature" / "geo01_worker.py",
)
G = PILOT.G
METRICS = PILOT.METRICS
WORKER = PILOT.WORKER
LEGACY = PILOT.LEGACY


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
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field, value in row.items():
            if field not in seen and not isinstance(value, (dict, list)):
                seen.add(field)
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def event_filename(prefix: str, event_id: str) -> str:
    if not event_id.replace("_", "").isalnum():
        raise ValueError(f"unsafe event id: {event_id}")
    return f"{prefix}__{event_id}.json"


class DiscoveryRecorder(PILOT.GeometryRecorder):
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
        self.discovery_contract = P.read_json(contract_path)
        checks = P.validate_contract(self.discovery_contract)
        if not all(checks.values()):
            raise RuntimeError(f"GEO-01B contract validation failed: {checks}")
        self.worker_args = worker_args
        discovery = self.discovery_contract["discovery"]
        unit_checks = {
            "formal_tier": worker_args.analysis_tier == "formal",
            "origin": worker_args.cell in discovery["origins"],
            "replica": int(worker_args.data_replica) in discovery["data_replicas"],
        }
        if not all(unit_checks.values()):
            raise RuntimeError(f"GEO-01B discovery unit mismatch: {unit_checks}")

        # Reuse the accepted pilot's source-pinned direction-construction
        # implementation. Its method reads only pilot.target_layers; provide an
        # internal compatibility view without changing the sealed contract.
        self.contract = copy.deepcopy(self.discovery_contract)
        self.contract["pilot"] = {
            "target_layers": list(discovery["target_layers"]),
            "scopes": copy.deepcopy(discovery["scopes"]),
        }
        self.snapshot_manifest_path = snapshot_manifest_path
        snapshot = P.read_json(snapshot_manifest_path)
        files = snapshot.get("files", {})
        expected_paths = {
            "scripts/47b_geo01b_update_geometry_discovery/geo01b_contract.json": contract_path,
            "scripts/47b_geo01b_update_geometry_discovery/protocol.py": Path(P.__file__),
            "scripts/47b_geo01b_update_geometry_discovery/geo01b_worker.py": Path(__file__),
            "scripts/47_update_geometry_curvature/geo01_worker.py": Path(PILOT.__file__),
            "scripts/47_update_geometry_curvature/geometry_core.py": Path(G.__file__),
            "scripts/37_mech09_downproj_refresh_mediation/mech09r_worker.py": Path(WORKER.__file__),
            "scripts/mdp_refresh_streaming/stream_metrics.py": Path(METRICS.__file__),
        }
        self.snapshot_checks = {
            relative: relative in files
            and sha256_file(path.resolve()) == files[relative]
            for relative, path in expected_paths.items()
        }
        self.snapshot_checks["manifest_passed"] = snapshot.get("passed") is True
        if not all(self.snapshot_checks.values()):
            raise RuntimeError(
                f"GEO-01B source snapshot integrity failed: {self.snapshot_checks}"
            )

        self.rows: list[dict[str, Any]] = []
        self.outcome_rows: list[dict[str, Any]] = []
        self.completed_events: list[str] = []
        self.event_artifacts: list[str] = []
        self.pending: dict[str, Any] | None = None
        self._selected_event: dict[str, Any] | None = None
        self._bound_optimizer: Any = None
        self._source_module: Any = None
        self._original_ns: Any = None
        self._patched_ns: Any = None
        self._patched_apply: Any = None
        self._original_apply_had_override = False
        self._original_apply_value: Any = None
        self.hook_lifecycle: list[dict[str, Any]] = []

    def event_selected(self, trajectory: str, completed_step: int) -> bool:
        matches = [
            event
            for event in self.discovery_contract["discovery"]["events"]
            if event["trajectory"] == trajectory
            and int(event["completed_step"]) == int(completed_step)
        ]
        if len(matches) > 1:
            raise RuntimeError("multiple GEO-01B events match one optimizer boundary")
        self._selected_event = matches[0] if matches else None
        return self._selected_event is not None

    def start_event(self) -> None:
        if self.pending is not None or self._selected_event is None:
            raise RuntimeError("GEO-01B event selection state is invalid")
        event = copy.deepcopy(self._selected_event)
        if event["event_id"] in self.completed_events:
            raise RuntimeError(f"duplicate GEO-01B event: {event['event_id']}")
        self.pending = {
            "event": event,
            "event_id": event["event_id"],
            "entries": [],
            "geometry_measured": False,
            "actual_ns_matches": 0,
            "row_start": len(self.rows),
        }

    def measure_geometry(self) -> None:
        if self.pending is None or self.pending["geometry_measured"]:
            raise RuntimeError("GEO-01B geometry event state is invalid")
        entries = self.pending["entries"]
        discovery = self.discovery_contract["discovery"]
        expected_layers = sorted(int(value) for value in discovery["target_layers"])
        if sorted(int(entry["layer"]) for entry in entries) != expected_layers:
            raise RuntimeError("GEO-01B all-layer direction coverage failed")
        geometry = self.discovery_contract["geometry"]
        batch_count = int(geometry["heldout_microbatches"])
        batch_size = int(geometry["microbatch_size"])
        sequence_length = int(geometry["sequence_length"])
        if len(PILOT.ACTIVE_EVAL_BATCHES) < batch_count:
            raise RuntimeError("not enough held-out batches for GEO-01B")
        batches = [
            (
                x[:batch_size, :sequence_length].contiguous(),
                y[:batch_size, :sequence_length].contiguous(),
            )
            for x, y in PILOT.ACTIVE_EVAL_BATCHES[:batch_count]
        ]
        batch_values_before = [
            {"x": METRICS.tensor_fingerprint(x), "y": METRICS.tensor_fingerprint(y)}
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
            for scope in discovery["scopes"]:
                layers = set(int(value) for value in scope["layers"])
                named_direction = {
                    entry["parameter_name"]: entry["direction"]
                    for entry in entries
                    if int(entry["layer"]) in layers
                }
                result = G.measure_directional_geometry(
                    model=PILOT.ACTIVE_MODEL,
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
                        "schema_version": "geo01b_discovery_scope_row_v1",
                        "phase": "discovery",
                        "origin": self.worker_args.cell,
                        "data_replica": int(self.worker_args.data_replica),
                        "event_id": self.pending["event_id"],
                        "completed_step": int(self.pending["event"]["completed_step"]),
                        "endpoint_step": int(self.pending["event"]["endpoint_step"]),
                        "scope_id": scope["scope_id"],
                        "scope_role": scope["role"],
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
            {"x": METRICS.tensor_fingerprint(x), "y": METRICS.tensor_fingerprint(y)}
            for x, y in batches
        ]
        self.pending["probe_invariance"] = {
            "raw_gradients_unchanged": grad_before == grad_after,
            "optimizer_state_unchanged": state_before == state_after,
            "heldout_batch_values_unchanged": batch_values_before == batch_values_after,
            "loader_iterator_advanced_by_probe": False,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
        self.pending["geometry_measured"] = True
        for entry in entries:
            entry["direction"] = None
        del batches
        gc.collect()

    def finish_after_optimizer_step(self) -> None:
        if self.pending is None or not self.pending["geometry_measured"]:
            raise RuntimeError("GEO-01B event did not measure geometry")
        entries = self.pending["entries"]
        invariance = self.pending["probe_invariance"]
        discovery = self.discovery_contract["discovery"]
        gates = self.discovery_contract["integrity_gates"]
        event_rows = self.rows[int(self.pending["row_start"]) :]
        checks = {
            "target_layers": len(entries) == len(discovery["target_layers"]),
            "actual_gradients": all(entry["actual_gradient_match"] for entry in entries),
            "actual_ns_inputs": all(entry["actual_ns_input_match"] for entry in entries),
            "actual_ns_outputs": all(entry["actual_ns_output_match"] for entry in entries),
            "actual_ns_count": int(self.pending["actual_ns_matches"]) == len(entries),
            "geometry_rows": len(event_rows) == len(discovery["scopes"]),
            "scope_coverage": sorted(row["scope_id"] for row in event_rows)
            == sorted(row["scope_id"] for row in discovery["scopes"]),
            "geometry_finite": all(row["all_values_finite"] for row in event_rows),
            "parameters_unchanged": all(row["parameters_unchanged"] for row in event_rows),
            "gradient_unchanged_by_probe": invariance["raw_gradients_unchanged"],
            "optimizer_unchanged_by_probe": invariance["optimizer_state_unchanged"],
            "heldout_batches_unchanged": invariance["heldout_batch_values_unchanged"],
            "loader_not_advanced": invariance["loader_iterator_advanced_by_probe"] is False,
            "memory": int(invariance["peak_allocated_bytes"])
            <= int(gates["peak_allocated_bytes_max"]),
            "baseline_graph": max(
                float(row["baseline_graph_relative_error"]) for row in event_rows
            )
            <= float(gates["baseline_graph_relative_error_max"]),
        }
        event_id = self.pending["event_id"]
        direction_name = event_filename("direction_construction_audit", event_id)
        geometry_name = event_filename("geometry_event_audit", event_id)
        atomic_json(
            self.output / direction_name,
            {
                "event_id": event_id,
                "rows": [
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
                ],
                "passed": all(checks.values()),
            },
        )
        atomic_json(
            self.output / geometry_name,
            {
                "event_id": event_id,
                "probe_invariance": invariance,
                "checks": checks,
                "passed": all(checks.values()),
            },
        )
        if not all(checks.values()):
            raise RuntimeError(f"GEO-01B event integrity failed: {checks}")
        self.completed_events.append(event_id)
        self.event_artifacts.extend([direction_name, geometry_name])
        self.pending = None
        self._selected_event = None

    @staticmethod
    def _one_evaluation(
        rows: list[dict[str, str]], *, arm: str, trajectory: str, step: int
    ) -> dict[str, str]:
        matches = [
            row
            for row in rows
            if row["arm"] == arm
            and row["trajectory_node"] == trajectory
            and int(row["optimizer_step"]) == int(step)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one evaluation row arm={arm} trajectory={trajectory} step={step}; got {len(matches)}"
            )
        return matches[0]

    def build_outcomes(self) -> list[dict[str, Any]]:
        evaluations = read_csv(self.output / "evaluation.csv")
        primary_scope = self.discovery_contract["discovery"]["primary_scope"]
        floor = float(self.discovery_contract["analysis"]["finite_floor"])
        outcomes = []
        for event in self.discovery_contract["discovery"]["events"]:
            treatment_start = self._one_evaluation(
                evaluations,
                arm=event["treatment_arm"],
                trajectory=event["treatment_trajectory"],
                step=int(event["completed_step"]),
            )
            reference_start = self._one_evaluation(
                evaluations,
                arm=event["reference_arm"],
                trajectory=event["reference_trajectory"],
                step=int(event["completed_step"]),
            )
            treatment_end = self._one_evaluation(
                evaluations,
                arm=event["treatment_arm"],
                trajectory=event["treatment_trajectory"],
                step=int(event["endpoint_step"]),
            )
            reference_end = self._one_evaluation(
                evaluations,
                arm=event["reference_arm"],
                trajectory=event["reference_trajectory"],
                step=int(event["endpoint_step"]),
            )
            geometry_rows = [
                row
                for row in self.rows
                if row["event_id"] == event["event_id"]
                and row["scope_id"] == primary_scope
            ]
            if len(geometry_rows) != 1:
                raise RuntimeError(f"primary geometry row missing: {event['event_id']}")
            geometry = geometry_rows[0]
            start_harm = float(treatment_start["normalized_loss"]) - float(
                reference_start["normalized_loss"]
            )
            endpoint_harm = float(treatment_end["normalized_loss"]) - float(
                reference_end["normalized_loss"]
            )
            width = int(event["endpoint_step"]) - int(event["completed_step"])
            local_exact = float(geometry["exact_actual_delta_loss"])
            local_full = float(geometry["taylor_actual_delta_loss"])
            local_first = float(geometry["first_order_alignment"])
            outcomes.append(
                {
                    "schema_version": "geo01b_discovery_unit_event_outcome_v1",
                    "origin": self.worker_args.cell,
                    "data_replica": int(self.worker_args.data_replica),
                    "event_id": event["event_id"],
                    "completed_step": int(event["completed_step"]),
                    "endpoint_step": int(event["endpoint_step"]),
                    "primary_scope": primary_scope,
                    "norm_only_predictor": float(geometry["relative_direction_fro_norm"]),
                    "first_order_predictor": local_first,
                    "full_taylor_predictor": local_full,
                    "local_exact_delta_loss": local_exact,
                    "local_first_relative_error": abs(local_exact - local_first)
                    / max(abs(local_exact), floor),
                    "local_taylor_relative_error": abs(local_exact - local_full)
                    / max(abs(local_exact), floor),
                    "local_first_sign_match": local_exact * local_first > 0.0,
                    "local_taylor_sign_match": local_exact * local_full > 0.0,
                    "start_normalized_loss_harm": start_harm,
                    "endpoint_normalized_loss_harm": endpoint_harm,
                    "endpoint_raw_loss_harm": float(treatment_end["heldout_loss"])
                    - float(reference_end["heldout_loss"]),
                    "trapezoid_normalized_auc_harm": 0.5
                    * (start_harm + endpoint_harm)
                    * width,
                    "outcome_positive_means_refresh_harm": True,
                    "all_values_finite": all(
                        math.isfinite(value)
                        for value in (
                            local_exact,
                            local_full,
                            local_first,
                            endpoint_harm,
                            start_harm,
                        )
                    ),
                }
            )
        return outcomes

    def finalize(self) -> dict[str, Any]:
        if self.pending is not None or self._bound_optimizer is not None:
            raise RuntimeError("GEO-01B hooks were not fully closed")
        base = P.read_json(self.output / "mech09r_manifest.json")
        self.outcome_rows = self.build_outcomes()
        write_jsonl(self.output / "geo01b_geometry_rows.jsonl", self.rows)
        write_csv(self.output / "geo01b_geometry_rows.csv", self.rows)
        write_jsonl(self.output / "geo01b_outcome_rows.jsonl", self.outcome_rows)
        write_csv(self.output / "geo01b_outcome_rows.csv", self.outcome_rows)
        discovery = self.discovery_contract["discovery"]
        expected_events = sorted(P.EVENTS)
        expected_scopes = sorted(P.SCOPES)
        checks = {
            "base_worker": base.get("passed") is True,
            "events": sorted(self.completed_events) == expected_events,
            "geometry_row_count": len(self.rows)
            == len(expected_events) * len(expected_scopes),
            "geometry_grid": sorted(
                (row["event_id"], row["scope_id"]) for row in self.rows
            )
            == sorted(
                (event, scope) for event in expected_events for scope in expected_scopes
            ),
            "outcome_row_count": len(self.outcome_rows) == len(expected_events),
            "outcome_events": sorted(row["event_id"] for row in self.outcome_rows)
            == expected_events,
            "finite": all(row["all_values_finite"] for row in self.rows)
            and all(row["all_values_finite"] for row in self.outcome_rows),
            "parameters_unchanged": all(
                row["parameters_unchanged"] for row in self.rows
            ),
            "hook_lifecycle": self.hook_lifecycle
            == [
                {"action": "bind"},
                {"action": "unbind"},
                {"action": "bind"},
                {"action": "unbind"},
            ],
            "discovery_not_claim_eligible": self.discovery_contract["claim_boundary"][
                "discovery_claim_eligible"
            ]
            is False,
            "all_layers": discovery["target_layers"] == list(range(18)),
        }
        artifacts = [
            *self.event_artifacts,
            "evaluation.csv",
            "geo01b_geometry_rows.csv",
            "geo01b_geometry_rows.jsonl",
            "geo01b_outcome_rows.csv",
            "geo01b_outcome_rows.jsonl",
            "mech09r_manifest.json",
        ]
        manifest = {
            "schema_version": "geo01b_discovery_unit_manifest_v1",
            "script_version": SCRIPT_VERSION,
            "experiment": "GEO-01B",
            "experiment_number": 47,
            "phase": "discovery",
            "origin": self.worker_args.cell,
            "data_replica": int(self.worker_args.data_replica),
            "contract_sha256": sha256_file(self.contract_path),
            "execution_contract_sha256": sha256_file(self.worker_args.contract),
            "source_snapshot_manifest_sha256": sha256_file(
                self.snapshot_manifest_path
            ),
            "events": expected_events,
            "geometry_rows": len(self.rows),
            "outcome_rows": len(self.outcome_rows),
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
        atomic_json(self.output / "geo01b_unit_manifest.json", manifest)
        atomic_json(
            self.output / "geo01b_status.json",
            {"status": "passed" if manifest["passed"] else "failed"},
        )
        if not manifest["passed"]:
            raise RuntimeError(f"GEO-01B final integrity failed: {checks}")
        return manifest


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geo01b-output-dir", required=True, type=Path)
    parser.add_argument("--geo01b-contract", required=True, type=Path)
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
    args, worker_arguments = parse_args()
    output = args.geo01b_output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    recorder: DiscoveryRecorder | None = None
    try:
        PILOT.replace_option(worker_arguments, "--output-dir", str(output))
        sys.argv = [str(WORKER.__file__), *worker_arguments]
        worker_args = WORKER.parse_args()
        recorder = DiscoveryRecorder(
            output=output,
            contract_path=args.geo01b_contract.resolve(),
            snapshot_manifest_path=args.source_snapshot_manifest.resolve(),
            worker_args=worker_args,
        )
        PILOT.RECORDER = recorder
        PILOT.install_hooks()
        WORKER.run_worker(worker_args)
        manifest = recorder.finalize()
        print(
            f"GEO-01B discovery unit passed origin={manifest['origin']} "
            f"replica={manifest['data_replica']} geometry_rows={manifest['geometry_rows']}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        if recorder is not None:
            try:
                recorder.unbind_optimizer(allow_pending=True)
            except BaseException:
                pass
        atomic_json(
            output / "geo01b_status.json",
            {
                "status": "failed",
                "script_version": SCRIPT_VERSION,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(
            f"GEO-01B discovery unit failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
