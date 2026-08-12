#!/usr/bin/env python3
"""Run one MECH-09 down-projection full-K refresh intervention."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import math
import shutil
import sys
import time
import traceback
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn


SCRIPT_VERSION = "2026-07-28.1"
CONTRACT_VERSION = "2026-07-28.1"
HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M8 = load_module(
    "mech09_mech08_worker",
    HERE.parent / "36_mech08_short_horizon_rollout" / "mech08_worker.py",
)
M1 = M8.M1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--analysis-tier", required=True, choices=("smoke", "formal"))
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--intervention", required=True)
    parser.add_argument("--data-replica", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-hash-certificate", required=True, type=Path)
    parser.add_argument("--source-script", required=True, type=Path)
    parser.add_argument("--profile-script", required=True, type=Path)
    parser.add_argument("--triton-kernels", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--mech08-control-reference", required=True, type=Path)
    parser.add_argument("--train-data-pattern", required=True)
    parser.add_argument("--val-data-pattern", required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--execution-domain", required=True)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def checkpoint_spec(contract: dict[str, Any], cell: str) -> dict[str, Any]:
    matches = [row for row in contract["checkpoints"] if row["cell"] == cell]
    if len(matches) != 1:
        raise RuntimeError(f"checkpoint cell is not unique: {cell}")
    return matches[0]


def down_refresh_steps(
    intervention: dict[str, Any], tier: str
) -> tuple[int, ...]:
    key = f"{tier}_down_refresh_completed_steps"
    values = tuple(int(value) for value in intervention[key])
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{key} must be sorted and unique")
    return values


def refresh_action(completed_step: int, target_steps: tuple[int, ...]) -> str:
    return "refresh" if int(completed_step) in target_steps else "hold"


def validate_contract(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = read_json(args.contract.resolve())
    config = contract[args.analysis_tier]
    spec = checkpoint_spec(contract, args.cell)
    certificate = read_json(args.checkpoint_hash_certificate.resolve())
    intervention = contract["interventions"].get(args.intervention)
    reference = read_json(args.mech08_control_reference.resolve())
    reference_spec = contract["mech08_control_reference"]
    target_steps = (
        down_refresh_steps(intervention, args.analysis_tier)
        if isinstance(intervention, dict)
        else ()
    )
    expected_other = tuple(
        int(value)
        for value in config["expected_other_group_refresh_completed_steps"]
    )
    derived_other = tuple(
        range(
            int(config["other_group_refresh_interval"]),
            int(config["rollout_steps"]) + 1,
            int(config["other_group_refresh_interval"]),
        )
    )
    primary = {
        (row["left"], row["right"])
        for row in contract["comparison_contract"]["primary"]
    }
    checks = {
        "contract_version": contract.get("contract_version") == CONTRACT_VERSION,
        "experiment": contract.get("experiment") == "MECH-09",
        "family": contract.get("family") == "llama1b",
        "cell_in_tier": args.cell in config["origins"],
        "intervention_in_tier": args.intervention in config["interventions"],
        "intervention_known": isinstance(intervention, dict),
        "source_method": isinstance(intervention, dict)
        and intervention.get("source_method") == "newton_full",
        "replica_in_tier": args.data_replica in config["data_replicas"],
        "checkpoint_path": str(args.checkpoint.resolve()) == spec["path"],
        "certificate_passed": certificate.get("passed") is True,
        "certificate_cell": certificate.get("cell") == args.cell,
        "certificate_path": certificate.get("path") == spec["path"],
        "certificate_sha": certificate.get("sha256") == spec["expected_sha256"],
        "certificate_bytes": int(certificate.get("bytes", -1))
        == int(spec["expected_bytes"]),
        "source_sha": M1.sha256_file(args.source_script.resolve())
        == contract["source_constraints"]["base_source_sha256"],
        "profile_sha": M1.sha256_file(args.profile_script.resolve())
        == contract["source_constraints"]["profile_script_sha256"],
        "triton_sha": M1.sha256_file(args.triton_kernels.resolve())
        == contract["source_constraints"]["triton_sha256"],
        "reference_sha": M1.sha256_file(
            args.mech08_control_reference.resolve()
        )
        == reference_spec.get("public_sha256", reference_spec["sha256"]),
        "reference_passed": reference.get("passed") is True,
        "reference_run": reference.get("source_run_id")
        == reference_spec["source_run_id"],
        "reference_file_count": int(reference.get("file_count", -1))
        == int(reference_spec["expected_file_count"]),
        "reference_contract": reference.get("mech08_contract_sha256")
        == reference_spec["expected_mech08_contract_sha256"],
        "target_schedule_subset": set(target_steps).issubset(expected_other),
        "other_schedule_consistent": expected_other == derived_other,
        "interventions_exact": set(contract["interventions"])
        == {"delayed_down_refresh", "frozen_down_refresh"},
        "primary_contrasts_exact": primary
        == {
            ("delayed_down_refresh", "original_newton_muon"),
            ("frozen_down_refresh", "original_newton_muon"),
            ("delayed_down_refresh", "frozen_down_refresh"),
        },
        "formal_job_cap": int(
            contract["stopping_rule"]["maximum_new_formal_jobs"]
        )
        == 24,
        "diag_none_not_primary": "selective_diag_vs_selective_none"
        in contract["comparison_contract"]["excluded_from_primary"],
        "efficiency_excluded": contract["scope_boundary"][
            "efficiency_benchmark_excluded"
        ]
        is True,
    }
    return contract, config, {
        "contract_sha256": M1.sha256_file(args.contract.resolve()),
        "checkpoint_spec": spec,
        "intervention_spec": intervention,
        "checkpoint_certificate": certificate,
        "mech08_control_reference": {
            "path": str(args.mech08_control_reference.resolve()),
            "sha256": M1.sha256_file(args.mech08_control_reference.resolve()),
            "source_run_id": reference.get("source_run_id"),
            "file_count": reference.get("file_count"),
        },
        "target_refresh_completed_steps": list(target_steps),
        "other_refresh_completed_steps": list(expected_other),
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_smoke_gate(
    args: argparse.Namespace, contract_sha256: str
) -> dict[str, Any]:
    if args.analysis_tier == "smoke":
        return {"required": False, "passed": True}
    if args.smoke_manifest is None or not args.smoke_manifest.is_file():
        return {"required": True, "passed": False, "reason": "missing smoke manifest"}
    manifest = read_json(args.smoke_manifest.resolve())
    checks = {
        "passed": manifest.get("passed") is True,
        "contract_sha": manifest.get("contract_sha256") == contract_sha256,
        "worker_version": manifest.get("worker_version") == SCRIPT_VERSION,
        "jobs": int(manifest.get("completed_jobs", -1)) == 2,
    }
    return {
        "required": True,
        "manifest": str(args.smoke_manifest.resolve()),
        "manifest_sha256": M1.sha256_file(args.smoke_manifest.resolve()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def partition_groups(
    groups: list[dict[str, Any]], target_suffix: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = [
        group for group in groups if str(group["name"]).endswith(target_suffix)
    ]
    other = [
        group for group in groups if not str(group["name"]).endswith(target_suffix)
    ]
    return target, other


def tensor_fingerprints(
    tensors: list[Tensor], samples: int = 17
) -> list[dict[str, Any]]:
    """Fingerprint many tensors with one bounded device-to-host transfer."""
    pending: list[tuple[dict[str, Any], Tensor]] = []
    for tensor in tensors:
        flat = tensor.detach().reshape(-1)
        indices = (
            []
            if flat.numel() == 0
            else sorted(
                {
                    int(index * (flat.numel() - 1) // max(samples - 1, 1))
                    for index in range(samples)
                }
            )
        )
        if indices:
            index_tensor = torch.tensor(
                indices, device=flat.device, dtype=torch.long
            )
            sampled = flat.index_select(0, index_tensor).float()
        else:
            sampled = torch.empty(
                0, device=flat.device, dtype=torch.float32
            )
        pending.append(
            (
                {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "indices": indices,
                },
                sampled,
            )
        )
    packed = torch.cat([sampled for _, sampled in pending])
    packed_values = [float(value) for value in packed.cpu().tolist()]
    output = []
    cursor = 0
    for metadata, sampled in pending:
        count = sampled.numel()
        values = packed_values[cursor : cursor + count]
        cursor += count
        payload = {**metadata, "values": values}
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        output.append(
            {
                **payload,
                "sampled_values_finite": all(
                    math.isfinite(value) for value in values
                ),
                "fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    return output


def group_state_snapshot(
    optimizer: torch.optim.Optimizer,
    groups: list[dict[str, Any]],
    *,
    include_statistics: bool,
) -> dict[str, Any]:
    state_tensors = []
    for group in groups:
        state = optimizer.state[group["members"][0]]
        state_tensors.extend(
            [state["precond_cov"], state["precond_inv_apply"]]
        )
    fingerprints = tensor_fingerprints(state_tensors)
    rows = [
        {
            "name": str(group["name"]),
            "covariance": fingerprints[index * 2],
            "inverse": fingerprints[index * 2 + 1],
        }
        for index, group in enumerate(groups)
    ]
    covariance_digest = hashlib.sha256()
    inverse_digest = hashlib.sha256()
    for row in rows:
        covariance_digest.update(
            row["covariance"]["fingerprint_sha256"].encode("ascii")
        )
        inverse_digest.update(
            row["inverse"]["fingerprint_sha256"].encode("ascii")
        )
    if include_statistics:
        accum_nonzero = torch.stack(
            [torch.count_nonzero(group["accum"]) for group in groups]
        ).sum()
        count_sum = torch.stack(
            [group["count"].float() for group in groups]
        ).sum()
        accum_value, count_value = [
            float(value)
            for value in torch.stack([accum_nonzero.float(), count_sum])
            .cpu()
            .tolist()
        ]
        statistics = {
            "accum_nonzero": int(accum_value),
            "count_sum": count_value,
        }
    else:
        statistics = {"accum_nonzero": None, "count_sum": None}
    return {
        "groups": len(groups),
        "fingerprint_method": "17 fixed flattened samples per state tensor",
        "rows": rows,
        "covariance_fingerprint_sha256": covariance_digest.hexdigest(),
        "inverse_fingerprint_sha256": inverse_digest.hexdigest(),
        "sampled_state_finite": all(
            row["covariance"]["sampled_values_finite"]
            and row["inverse"]["sampled_values_finite"]
            for row in rows
        ),
        **statistics,
    }


class RefreshInterventionController:
    """Surgically change only the down-projection refresh actions."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        target_suffix: str,
        target_refresh_steps: tuple[int, ...],
        expected_other_refresh_steps: tuple[int, ...],
        expected_layers: int,
    ) -> None:
        self.optimizer = optimizer
        self.target_refresh_steps = target_refresh_steps
        self.expected_other_refresh_steps = expected_other_refresh_steps
        self.all_groups = list(optimizer._groups)
        self.target_groups, self.other_groups = partition_groups(
            self.all_groups, target_suffix
        )
        if len(self.target_groups) != int(expected_layers):
            raise RuntimeError(
                "unexpected down-projection group count: "
                f"{len(self.target_groups)} != {expected_layers}"
            )
        if len(self.other_groups) != int(expected_layers) * 3:
            raise RuntimeError(
                "unexpected non-target group count: "
                f"{len(self.other_groups)} != {expected_layers * 3}"
            )
        self.original_refresh = optimizer._refresh_preconditioners
        self.events: list[dict[str, Any]] = []

        def patched_refresh(_: torch.optim.Optimizer) -> None:
            self.handle_refresh()

        optimizer._refresh_preconditioners = types.MethodType(
            patched_refresh, optimizer
        )

    @torch.no_grad()
    def handle_refresh(self) -> None:
        completed_step = int(self.optimizer.global_step) + 1
        if completed_step not in self.expected_other_refresh_steps:
            raise RuntimeError(
                f"unexpected global refresh event at step {completed_step}"
            )
        action = refresh_action(completed_step, self.target_refresh_steps)
        target_before = group_state_snapshot(
            self.optimizer,
            self.target_groups,
            include_statistics=False,
        )
        other_before = group_state_snapshot(
            self.optimizer,
            self.other_groups,
            include_statistics=False,
        )
        try:
            self.optimizer._groups = self.other_groups
            self.original_refresh()
            if action == "refresh":
                self.optimizer._groups = self.target_groups
                self.original_refresh()
            else:
                for group in self.target_groups:
                    group["accum"].zero_()
                    group["count"].zero_()
        finally:
            self.optimizer._groups = self.all_groups
        target_after = group_state_snapshot(
            self.optimizer,
            self.target_groups,
            include_statistics=True,
        )
        other_after = group_state_snapshot(
            self.optimizer,
            self.other_groups,
            include_statistics=True,
        )
        self.events.append(
            {
                "completed_step": completed_step,
                "target_action": action,
                "target_before": target_before,
                "target_after": target_after,
                "target_covariance_changed": (
                    target_before["covariance_fingerprint_sha256"]
                    != target_after["covariance_fingerprint_sha256"]
                ),
                "target_inverse_changed": (
                    target_before["inverse_fingerprint_sha256"]
                    != target_after["inverse_fingerprint_sha256"]
                ),
                "target_statistics_zero_after": (
                    target_after["accum_nonzero"] == 0
                    and target_after["count_sum"] == 0.0
                ),
                "other_before": other_before,
                "other_after": other_after,
                "other_covariance_changed": (
                    other_before["covariance_fingerprint_sha256"]
                    != other_after["covariance_fingerprint_sha256"]
                ),
                "other_inverse_changed": (
                    other_before["inverse_fingerprint_sha256"]
                    != other_after["inverse_fingerprint_sha256"]
                ),
                "other_statistics_zero_after": (
                    other_after["accum_nonzero"] == 0
                    and other_after["count_sum"] == 0.0
                ),
            }
        )

    def audit(
        self,
        initial_target: dict[str, Any],
        final_target: dict[str, Any],
    ) -> dict[str, Any]:
        observed_steps = [row["completed_step"] for row in self.events]
        observed_target_refresh = [
            row["completed_step"]
            for row in self.events
            if row["target_action"] == "refresh"
        ]
        held_rows = [
            row for row in self.events if row["target_action"] == "hold"
        ]
        refreshed_rows = [
            row for row in self.events if row["target_action"] == "refresh"
        ]
        checks = {
            "target_group_count": len(self.target_groups) * 3
            == len(self.other_groups),
            "event_schedule": observed_steps
            == list(self.expected_other_refresh_steps),
            "target_schedule": observed_target_refresh
            == list(self.target_refresh_steps),
            "all_statistics_zero_after": all(
                row["target_statistics_zero_after"]
                and row["other_statistics_zero_after"]
                for row in self.events
            ),
            "held_target_state_unchanged": all(
                not row["target_covariance_changed"]
                and not row["target_inverse_changed"]
                for row in held_rows
            ),
            "refreshed_target_state_changed": all(
                row["target_covariance_changed"]
                and row["target_inverse_changed"]
                for row in refreshed_rows
            ),
            "other_groups_refreshed": all(
                row["other_covariance_changed"] and row["other_inverse_changed"]
                for row in self.events
            ),
            "target_state_finite": final_target["sampled_state_finite"],
            "frozen_target_state_matches_initial": bool(
                self.target_refresh_steps
            )
            or (
                initial_target["covariance_fingerprint_sha256"]
                == final_target["covariance_fingerprint_sha256"]
                and initial_target["inverse_fingerprint_sha256"]
                == final_target["inverse_fingerprint_sha256"]
            ),
        }
        return {
            "target_group_names": [
                str(group["name"]) for group in self.target_groups
            ],
            "other_group_count": len(self.other_groups),
            "target_refresh_completed_steps": list(self.target_refresh_steps),
            "other_refresh_completed_steps": list(
                self.expected_other_refresh_steps
            ),
            "initial_target_state": initial_target,
            "final_target_state": final_target,
            "events": self.events,
            "checks": checks,
            "passed": all(checks.values()),
        }


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MECH-09 requires CUDA")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    output = args.output_dir.resolve()
    if not output.is_dir():
        raise RuntimeError(f"controller must create output directory: {output}")
    M1.atomic_json(
        output / "status.json",
        {"status": "running", "script_version": SCRIPT_VERSION},
    )

    contract, config, contract_audit = validate_contract(args)
    if not contract_audit["passed"]:
        raise RuntimeError(f"contract audit failed: {contract_audit}")
    smoke_gate = validate_smoke_gate(
        args, contract_audit["contract_sha256"]
    )
    if not smoke_gate["passed"]:
        raise RuntimeError(f"smoke gate failed: {smoke_gate}")
    shutil.copyfile(args.contract, output / "refresh_mediation_contract.json")
    shutil.copyfile(
        args.mech08_control_reference,
        output / "mech08_control_reference.json",
    )

    spec = contract_audit["checkpoint_spec"]
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_stat_before = checkpoint_path.stat()
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except (TypeError, RuntimeError) as exc:
        if "mmap" not in str(exc).lower() and not isinstance(exc, TypeError):
            raise
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    model_state, schema = M1.checkpoint_schema(
        checkpoint_path,
        checkpoint,
        "llama1b",
        args.source_script.resolve(),
        contract_audit["checkpoint_certificate"]["sha256"],
        False,
    )
    M1.attach_profile_provenance(schema, args.profile_script)
    method_audit = M8.method_identity_audit(
        checkpoint, spec["method"], schema["method_inferred"]
    )
    architecture = schema["architecture"]
    architecture_checks = {
        "step": int(schema["step"]) == int(spec["step"]),
        "method": method_audit["passed"],
        "n_layer": architecture["n_layer"]
        == contract["architecture"]["n_layer"],
        "n_embd": architecture["n_embd"]
        == contract["architecture"]["n_embd"],
        "vocab_size": architecture["vocab_size"]
        == contract["architecture"]["vocab_size"],
        "target_input_width": architecture["target_input_width"]
        == contract["architecture"]["target_input_width"],
        "checkpoint_next_batch_shape": list(checkpoint["next_x"].shape)
        == [int(config["device_batch_size"]), int(config["sequence_length"])],
    }
    if not schema["passed"] or not all(architecture_checks.values()):
        raise RuntimeError(
            f"checkpoint schema/architecture failed: {architecture_checks}"
        )

    source_runtime, _, triton_audit = M1.load_source_runtime(
        "llama1b", args.source_script.resolve(), args.triton_kernels.resolve()
    )
    source_method = "newton_full"
    source_config = M1.configure_source_runtime_globals(
        "llama1b", source_runtime, source_method
    )
    model = M1.build_model(
        "llama1b", source_runtime, architecture, source_method
    )
    parameter_audit = M8.copy_checkpoint_parameters(model, model_state)
    model.cuda()
    restart = dict(contract["restart_intervention"])
    restart["newton_refresh_steps"] = int(
        config["other_group_refresh_interval"]
    )
    optimizers, matrix_optimizer = M8.make_candidate_optimizers(
        source_runtime, model, source_method, restart
    )
    if not isinstance(matrix_optimizer, source_runtime.SharedInputNewtonMuon):
        raise RuntimeError("MECH-09 requires SharedInputNewtonMuon")
    matrix_optimizer.refresh = int(config["other_group_refresh_interval"])
    backup_audit, momentum_audit = M8.transfer_optimizer_state(
        checkpoint, optimizers, matrix_optimizer, model
    )
    for group in optimizers[0].param_groups:
        group["lr"] = float(restart["backup_lr"])
    for group in matrix_optimizer.param_groups:
        group["lr"] = float(restart["matrix_lr"])
    optimizer_audit = M8.optimizer_hyperparameter_audit(
        optimizers, matrix_optimizer, source_runtime, restart
    )
    if not optimizer_audit["passed"]:
        raise RuntimeError(
            f"candidate optimizer hyperparameters failed: {optimizer_audit}"
        )

    replicas = [int(value) for value in config["data_replicas"]]
    offsets = [
        int(value) for value in config["replica_optimizer_step_offsets"]
    ]
    replica_index = replicas.index(args.data_replica)
    optimizer_step_offset = offsets[replica_index]
    accumulation = int(config["global_batch_size"]) // int(
        config["device_batch_size"]
    )
    if accumulation * int(config["device_batch_size"]) != int(
        config["global_batch_size"]
    ):
        raise RuntimeError("global batch size is not divisible by device batch size")
    train_loader = source_runtime.SequentialShardLoader(
        args.train_data_pattern,
        int(config["device_batch_size"]),
        int(config["sequence_length"]),
    )
    train_loader.load_state_dict(checkpoint["train_loader"])
    if optimizer_step_offset == 0:
        x = checkpoint["next_x"].cuda(non_blocking=True)
        y = checkpoint["next_y"].cuda(non_blocking=True)
        seek_audit = {
            "requested_batches": 0,
            "start": checkpoint["train_loader"],
            "end": train_loader.state_dict(),
            "materialized_batches": 0,
            "used_checkpoint_next_batch": True,
        }
    else:
        skip_batches = optimizer_step_offset * accumulation - 1
        seek_audit = M8.advance_loader_without_materializing(
            train_loader, source_runtime, skip_batches
        )
        x, y = train_loader.next_batch()
        seek_audit["used_checkpoint_next_batch"] = False
    training_stream_contract = {
        "origin_cell": args.cell,
        "data_replica": args.data_replica,
        "optimizer_step_offset": optimizer_step_offset,
        "rollout_optimizer_steps": int(config["rollout_steps"]),
        "global_batch_size": int(config["global_batch_size"]),
        "device_batch_size": int(config["device_batch_size"]),
        "sequence_length": int(config["sequence_length"]),
        "gradient_accumulation_steps": accumulation,
        "first_x_sha256": M1.tensor_sha256(x),
        "first_y_sha256": M1.tensor_sha256(y),
        "seek": seek_audit,
        "checkpoint_loader_state_is_after_stored_next_batch": True,
        "passed": True,
    }

    build_offset = int(config["build_token_offsets"][replica_index])
    eval_offset = int(config["eval_token_offsets"][replica_index])
    build_batches, build_contract = M8.frozen_batches(
        source_runtime,
        args.val_data_pattern,
        int(config["build_device_batch_size"]),
        int(config["build_sequence_length"]),
        int(config["build_batches"]),
        build_offset,
    )
    eval_batches, eval_contract = M8.frozen_batches(
        source_runtime,
        args.val_data_pattern,
        int(config["eval_device_batch_size"]),
        int(config["eval_sequence_length"]),
        int(config["eval_batches"]),
        eval_offset,
    )
    build_end = build_offset + int(config["build_batches"]) * int(
        config["build_device_batch_size"]
    ) * int(config["build_sequence_length"])
    eval_end = eval_offset + int(config["eval_batches"]) * int(
        config["eval_device_batch_size"]
    ) * int(config["eval_sequence_length"])
    heldout_contract = {
        "build": build_contract,
        "evaluation": eval_contract,
        "build_interval": [build_offset, build_end],
        "evaluation_interval": [eval_offset, eval_end],
        "windows_disjoint": max(build_offset, eval_offset)
        >= min(build_end, eval_end),
    }
    heldout_contract["passed"] = heldout_contract["windows_disjoint"]
    if not heldout_contract["passed"]:
        raise RuntimeError("preconditioner-build and evaluation windows overlap")

    preconditioner_audit = M8.build_fresh_preconditioner(
        model, matrix_optimizer, build_batches, source_runtime
    )
    if not preconditioner_audit["passed"]:
        raise RuntimeError(
            f"initial preconditioner audit failed: {preconditioner_audit}"
        )
    intervention_spec = contract_audit["intervention_spec"]
    target_steps = down_refresh_steps(
        intervention_spec, args.analysis_tier
    )
    expected_other_steps = tuple(
        int(value)
        for value in config["expected_other_group_refresh_completed_steps"]
    )
    controller = RefreshInterventionController(
        matrix_optimizer,
        target_suffix=intervention_spec["target_group_suffix"],
        target_refresh_steps=target_steps,
        expected_other_refresh_steps=expected_other_steps,
        expected_layers=int(contract["architecture"]["n_layer"]),
    )
    initial_target_state = group_state_snapshot(
        matrix_optimizer,
        controller.target_groups,
        include_statistics=True,
    )

    del build_batches
    model.zero_grad(set_to_none=True)
    gc.collect()
    del model_state
    del checkpoint
    gc.collect()

    evaluation_steps = {int(value) for value in config["evaluation_steps"]}
    if 0 not in evaluation_steps or int(config["rollout_steps"]) not in evaluation_steps:
        raise RuntimeError("evaluation schedule must include both endpoints")
    torch.cuda.reset_peak_memory_stats()
    evaluation_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    initial_loss = M8.evaluate(model, eval_batches)
    evaluation_rows.append(
        {
            "checkpoint_cell": args.cell,
            "checkpoint_stage": spec["stage"],
            "checkpoint_method": spec["method"],
            "intervention": args.intervention,
            "source_method": source_method,
            "data_replica": args.data_replica,
            "optimizer_step": 0,
            "heldout_loss": initial_loss,
            "normalized_loss": 1.0,
            "loss_delta_from_step0": 0.0,
            "relative_loss_delta_from_step0": 0.0,
        }
    )

    compiled_model: nn.Module = model if args.no_compile else torch.compile(model)
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    rollout_steps = int(config["rollout_steps"])
    for step in range(rollout_steps):
        matrix_optimizer.global_step = step
        precond_flag = matrix_optimizer.precond_flag_for_step(step)
        completed = step + 1
        target_action = (
            refresh_action(completed, target_steps)
            if precond_flag
            else "none"
        )
        model.train()
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss_sum = 0.0
        for _ in range(accumulation):
            with autocast:
                _, loss = compiled_model(
                    x,
                    y,
                    return_logits=False,
                    precond_flag=precond_flag,
                )
                if loss is None:
                    raise RuntimeError("training loss is missing")
                scaled = loss / accumulation
            loss_sum += float(loss.detach().item())
            x, y = train_loader.next_batch()
            scaled.backward()
        for optimizer in optimizers:
            optimizer.step()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        train_loss = loss_sum / accumulation
        if not math.isfinite(train_loss):
            raise FloatingPointError(
                f"non-finite training loss at optimizer step {completed}"
            )
        training_rows.append(
            {
                "checkpoint_cell": args.cell,
                "checkpoint_stage": spec["stage"],
                "checkpoint_method": spec["method"],
                "intervention": args.intervention,
                "source_method": source_method,
                "data_replica": args.data_replica,
                "optimizer_step": completed,
                "train_loss_mean": train_loss,
                "global_preconditioner_event": precond_flag,
                "down_preconditioner_action": target_action,
                "backup_lr": float(optimizers[0].param_groups[0]["lr"]),
                "matrix_lr": float(matrix_optimizer.param_groups[0]["lr"]),
                "tokens_processed": completed
                * int(config["global_batch_size"])
                * int(config["sequence_length"]),
                "diagnostic_step_seconds": duration,
                "timing_usable_for_paper": False,
            }
        )
        if completed in evaluation_steps:
            heldout_loss = M8.evaluate(model, eval_batches)
            evaluation_rows.append(
                {
                    "checkpoint_cell": args.cell,
                    "checkpoint_stage": spec["stage"],
                    "checkpoint_method": spec["method"],
                    "intervention": args.intervention,
                    "source_method": source_method,
                    "data_replica": args.data_replica,
                    "optimizer_step": completed,
                    "heldout_loss": heldout_loss,
                    "normalized_loss": heldout_loss / initial_loss,
                    "loss_delta_from_step0": heldout_loss - initial_loss,
                    "relative_loss_delta_from_step0": (
                        heldout_loss - initial_loss
                    )
                    / initial_loss,
                }
            )
        if completed in evaluation_steps or precond_flag:
            print(
                "MECH-09 "
                f"cell={args.cell} intervention={args.intervention} "
                f"replica={args.data_replica} step={completed}/{rollout_steps} "
                f"down_action={target_action} train_loss={train_loss:.6f}",
                flush=True,
            )

    final_target_state = group_state_snapshot(
        matrix_optimizer,
        controller.target_groups,
        include_statistics=True,
    )
    intervention_audit = controller.audit(
        initial_target_state, final_target_state
    )
    checkpoint_stat_after = checkpoint_path.stat()
    invariance = {
        "checkpoint_size_unchanged": checkpoint_stat_before.st_size
        == checkpoint_stat_after.st_size,
        "checkpoint_mtime_unchanged": checkpoint_stat_before.st_mtime_ns
        == checkpoint_stat_after.st_mtime_ns,
        "checkpoint_before": {
            "bytes": checkpoint_stat_before.st_size,
            "mtime_ns": checkpoint_stat_before.st_mtime_ns,
        },
        "checkpoint_after": {
            "bytes": checkpoint_stat_after.st_size,
            "mtime_ns": checkpoint_stat_after.st_mtime_ns,
        },
    }
    checks = {
        "contract_audit": contract_audit["passed"],
        "smoke_gate": smoke_gate["passed"],
        "checkpoint_schema": schema["passed"]
        and all(architecture_checks.values()),
        "method_identity": method_audit["passed"],
        "source_runtime": source_config["passed"],
        "triton_provenance": triton_audit["passed"],
        "parameters_copied": parameter_audit["passed"],
        "backup_state_loaded": backup_audit["passed"],
        "historical_momentum_transferred": momentum_audit["passed"],
        "optimizer_hyperparameters": optimizer_audit["passed"],
        "fresh_preconditioner": preconditioner_audit["passed"],
        "refresh_intervention": intervention_audit["passed"],
        "heldout_windows": heldout_contract["passed"],
        "training_rows": len(training_rows) == rollout_steps,
        "evaluation_rows": len(evaluation_rows) == len(evaluation_steps),
        "evaluation_schedule": {
            row["optimizer_step"] for row in evaluation_rows
        }
        == evaluation_steps,
        "finite_training": M1.finite_numbers(training_rows),
        "finite_evaluation": M1.finite_numbers(evaluation_rows),
        "final_parameters_finite": M8.model_parameters_finite(model),
        "checkpoint_file_unchanged": invariance["checkpoint_size_unchanged"]
        and invariance["checkpoint_mtime_unchanged"],
        "real_optimizer_steps": True,
        "timing_not_for_paper": all(
            row["timing_usable_for_paper"] is False for row in training_rows
        ),
    }
    passed = all(checks.values())
    runtime_args = SimpleNamespace(
        host_id=args.host_id,
        execution_domain=args.execution_domain,
    )
    artifacts = {
        "contract_audit.json": contract_audit,
        "smoke_gate.json": smoke_gate,
        "checkpoint_schema.json": schema,
        "method_identity_audit.json": method_audit,
        "architecture_checks.json": architecture_checks,
        "source_runtime_config.json": source_config,
        "triton_audit.json": triton_audit,
        "parameter_transfer_audit.json": parameter_audit,
        "backup_optimizer_audit.json": backup_audit,
        "momentum_transfer_audit.json": momentum_audit,
        "optimizer_hyperparameters.json": optimizer_audit,
        "initial_preconditioner_audit.json": preconditioner_audit,
        "refresh_intervention_audit.json": intervention_audit,
        "training_stream_contract.json": training_stream_contract,
        "heldout_batch_contract.json": heldout_contract,
        "checkpoint_invariance.json": invariance,
        "runtime.json": M1.runtime_metadata(runtime_args),
        "checks.json": checks,
    }
    for name, value in artifacts.items():
        M1.atomic_json(output / name, value)
    M8.write_csv(output / "training.csv", training_rows)
    M8.write_csv(output / "evaluation.csv", evaluation_rows)
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "analysis_tier": args.analysis_tier,
        "contract_sha256": contract_audit["contract_sha256"],
        "checkpoint_cell": args.cell,
        "checkpoint_stage": spec["stage"],
        "checkpoint_method": spec["method"],
        "checkpoint_step": spec["step"],
        "checkpoint_sha256": contract_audit["checkpoint_certificate"]["sha256"],
        "intervention": args.intervention,
        "source_method": source_method,
        "data_replica": args.data_replica,
        "optimizer_step_offset": optimizer_step_offset,
        "rollout_steps": rollout_steps,
        "target_refresh_completed_steps": list(target_steps),
        "other_refresh_completed_steps": list(expected_other_steps),
        "training_rows": len(training_rows),
        "evaluation_rows": len(evaluation_rows),
        "real_optimizer_steps": True,
        "timing_usable_for_paper": False,
        "efficiency_benchmark_run": False,
        "passed": passed,
        "artifacts": sorted(
            [
                *artifacts,
                "refresh_mediation_contract.json",
                "mech08_control_reference.json",
                "training.csv",
                "evaluation.csv",
                "mech09_manifest.json",
                "status.json",
            ]
        ),
    }
    M1.atomic_json(output / "mech09_manifest.json", manifest)
    M1.atomic_json(
        output / "status.json",
        {
            "status": "passed" if passed else "failed_checks",
            "script_version": SCRIPT_VERSION,
            "passed": passed,
        },
    )
    if not passed:
        raise SystemExit(2)
    print(f"MECH-09 manifest: {output / 'mech09_manifest.json'}", flush=True)
    print(f"MECH-09 artifacts: {output}", flush=True)


def main() -> None:
    args = parse_args()
    try:
        run_worker(args)
    except BaseException as exc:
        output = args.output_dir.resolve()
        if isinstance(exc, SystemExit) and (output / "status.json").is_file():
            raise
        output.mkdir(parents=True, exist_ok=True)
        M1.atomic_json(
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
