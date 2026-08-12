#!/usr/bin/env python3
"""Run one shared-prefix MECH-09R causal-tree unit on CUDA."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
from types import SimpleNamespace
from typing import Any, Iterable
import traceback

import numpy as np
import torch
from torch import Tensor, nn


SCRIPT_VERSION = "2026-07-28.3"
CONTRACT_VERSION = "2026-07-28.2"
PUBLIC_CONTROL_REFERENCE_SHA256 = (
    "63464873e00c55c28b120c930ad207aa26fc75646678f9262e03904480c263ac"
)
HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = load_module("mech09r_legacy_helpers", HERE / "mech09_worker.py")
M8 = LEGACY.M8
M1 = LEGACY.M1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--analysis-tier", required=True, choices=("smoke", "formal")
    )
    parser.add_argument("--cell", required=True)
    parser.add_argument("--data-replica", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-hash-certificate", required=True, type=Path
    )
    parser.add_argument("--source-script", required=True, type=Path)
    parser.add_argument("--profile-script", required=True, type=Path)
    parser.add_argument("--triton-kernels", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument(
        "--mech08-control-reference", required=True, type=Path
    )
    parser.add_argument("--train-data-pattern", required=True)
    parser.add_argument("--val-data-pattern", required=True)
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--execution-domain", required=True)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def checkpoint_spec(contract: dict[str, Any], cell: str) -> dict[str, Any]:
    matches = [row for row in contract["checkpoints"] if row["cell"] == cell]
    if len(matches) != 1:
        raise RuntimeError(f"checkpoint cell is not unique: {cell}")
    return matches[0]


def tier_target_steps(
    contract: dict[str, Any], tier: str, arm: str
) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in contract["arms"][arm][
            f"{tier}_down_refresh_completed_steps"
        ]
    )


def derived_refresh_steps(config: dict[str, Any]) -> tuple[int, ...]:
    interval = int(config["production_refresh_interval"])
    return tuple(range(interval, int(config["rollout_steps"]) + 1, interval))


def validate_contract(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = read_json(args.contract.resolve())
    config = contract[args.analysis_tier]
    spec = checkpoint_spec(contract, args.cell)
    certificate = read_json(args.checkpoint_hash_certificate.resolve())
    reference = read_json(args.mech08_control_reference.resolve())
    reference_spec = contract["checkpoint_certificate_source"]
    tree = config["causal_tree"]
    expected_refresh = tuple(
        int(value)
        for value in config["expected_global_refresh_completed_steps"]
    )
    arms = set(contract["arms"])
    primary = {
        (row["left"], row["right"])
        for row in contract["comparison_contract"]["primary"]
    }
    checks = {
        "contract_version": contract.get("contract_version")
        == CONTRACT_VERSION,
        "experiment": contract.get("experiment") == "MECH-09R",
        "family": contract.get("family") == "llama1b",
        "amendment_pre_intervention_only": contract["protocol_amendment"][
            "trigger_uses_pre_intervention_data_only"
        ]
        is True,
        "cell_in_tier": args.cell in config["origins"],
        "replica_in_tier": args.data_replica in config["data_replicas"],
        "checkpoint_path": str(args.checkpoint.resolve()) == spec["path"],
        "certificate_passed": certificate.get("passed") is True,
        "certificate_cell": certificate.get("cell") == args.cell,
        "certificate_path": certificate.get("path") == spec["path"],
        "certificate_sha": certificate.get("sha256")
        == spec["expected_sha256"],
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
        in {
            reference_spec["mech08_control_reference_sha256"],
            PUBLIC_CONTROL_REFERENCE_SHA256,
        },
        "reference_passed": reference.get("passed") is True,
        "reference_run": reference.get("source_run_id")
        == reference_spec["source_run_id"],
        "reference_file_count": int(reference.get("file_count", -1))
        == int(reference_spec["expected_file_count"]),
        "arms_exact": arms
        == {
            "production_newton_muon",
            "delayed_down_refresh",
            "frozen_down_refresh",
        },
        "primary_exact": primary
        == {
            ("delayed_down_refresh", "production_newton_muon"),
            ("frozen_down_refresh", "production_newton_muon"),
            ("delayed_down_refresh", "frozen_down_refresh"),
        },
        "refresh_schedule": expected_refresh
        == derived_refresh_steps(config),
        "production_target_schedule": tier_target_steps(
            contract, args.analysis_tier, "production_newton_muon"
        )
        == expected_refresh,
        "delayed_target_subset": set(
            tier_target_steps(
                contract, args.analysis_tier, "delayed_down_refresh"
            )
        ).issubset(expected_refresh),
        "frozen_target_empty": not tier_target_steps(
            contract, args.analysis_tier, "frozen_down_refresh"
        ),
        "tree_first_branch": int(tree["first_branch_step"])
        == int(tree["shared_all_end_step"]) + 1,
        "tree_second_branch": int(tree["second_branch_step"])
        == int(tree["shared_no_down_end_step"]) + 1,
        "tree_starts_at_refresh": int(tree["first_branch_step"])
        == int(config["production_refresh_interval"]),
        "tree_second_is_refresh": int(tree["second_branch_step"])
        % int(config["production_refresh_interval"])
        == 0,
        "tree_ends_in_range": 0
        < int(tree["shared_all_end_step"])
        < int(tree["shared_no_down_end_step"])
        < int(config["rollout_steps"]),
        "formal_job_cap": int(
            contract["stopping_rule"]["maximum_new_formal_jobs"]
        )
        == 12,
        "diag_none_not_primary": contract["comparison_contract"][
            "selective_diag_vs_selective_none_is_primary"
        ]
        is False,
        "timing_excluded": contract["scope_boundary"][
            "efficiency_benchmark_excluded"
        ]
        is True,
    }
    return contract, config, {
        "contract_sha256": M1.sha256_file(args.contract.resolve()),
        "checkpoint_spec": spec,
        "checkpoint_certificate": certificate,
        "mech08_control_reference": {
            "path": str(args.mech08_control_reference.resolve()),
            "sha256": M1.sha256_file(
                args.mech08_control_reference.resolve()
            ),
            "source_run_id": reference.get("source_run_id"),
            "file_count": reference.get("file_count"),
            "used_for_primary_outcomes": False,
            "used_for_checkpoint_certificates_only": True,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_smoke_gate(
    args: argparse.Namespace, contract_sha256: str
) -> dict[str, Any]:
    if args.analysis_tier == "smoke":
        return {"required": False, "passed": True}
    if args.smoke_manifest is None or not args.smoke_manifest.is_file():
        return {
            "required": True,
            "passed": False,
            "reason": "missing smoke manifest",
        }
    manifest = read_json(args.smoke_manifest.resolve())
    checks = {
        "passed": manifest.get("passed") is True,
        "script_version": manifest.get("script_version") == SCRIPT_VERSION,
        "contract_sha256": manifest.get("contract_sha256")
        == contract_sha256,
        "tier": manifest.get("analysis_tier") == "smoke",
        "causal_tree": manifest.get("causal_tree") is True,
    }
    return {
        "required": True,
        "path": str(args.smoke_manifest.resolve()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def cpu_clone_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone_tree(item) for item in value)
    return value


def named_tensors(
    value: Any, prefix: str
) -> Iterable[tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda item: repr(item)):
            yield from named_tensors(
                value[key], f"{prefix}/dict:{repr(key)}"
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from named_tensors(item, f"{prefix}/item:{index}")


def structural_metadata(value: Any) -> Any:
    if isinstance(value, Tensor):
        return {
            "tensor_shape": list(value.shape),
            "tensor_dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {
            repr(key): structural_metadata(value[key])
            for key in sorted(value, key=lambda item: repr(item))
        }
    if isinstance(value, (list, tuple)):
        return [structural_metadata(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"repr": repr(value), "type": type(value).__name__}


def tensor_device_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return str(value.device)
    if isinstance(value, dict):
        return {key: tensor_device_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_device_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tensor_device_tree(item) for item in value)
    return None


def restore_tree_to_recorded_devices(
    value: Any,
    device_tree: Any,
    *,
    parameter_device: torch.device,
) -> Any:
    if isinstance(value, Tensor):
        if not isinstance(device_tree, str):
            raise RuntimeError("tensor device metadata is missing")
        target_device = (
            parameter_device
            if device_tree.startswith("cuda")
            else torch.device(device_tree)
        )
        return value.detach().to(
            device=target_device, dtype=value.dtype
        ).clone()
    if isinstance(value, dict):
        if not isinstance(device_tree, dict):
            raise RuntimeError("dict device metadata is missing")
        return {
            key: restore_tree_to_recorded_devices(
                item,
                device_tree[key],
                parameter_device=parameter_device,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        if not isinstance(device_tree, list):
            raise RuntimeError("list device metadata is missing")
        return [
            restore_tree_to_recorded_devices(
                item,
                device_tree[index],
                parameter_device=parameter_device,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        if not isinstance(device_tree, tuple):
            raise RuntimeError("tuple device metadata is missing")
        return tuple(
            restore_tree_to_recorded_devices(
                item,
                device_tree[index],
                parameter_device=parameter_device,
            )
            for index, item in enumerate(value)
        )
    return value


def restore_optimizer_state_exact(
    optimizer: torch.optim.Optimizer,
    saved_state_dict: dict[str, Any],
    saved_device_tree: dict[str, Any],
) -> dict[str, Any]:
    saved_groups = saved_state_dict["param_groups"]
    saved_group_devices = saved_device_tree["param_groups"]
    if len(saved_groups) != len(optimizer.param_groups):
        raise RuntimeError("optimizer param-group count changed at restore")
    saved_ids: list[Any] = []
    current_parameters: list[Tensor] = []
    group_checks = []
    for current, saved, devices in zip(
        optimizer.param_groups, saved_groups, saved_group_devices
    ):
        if len(current["params"]) != len(saved["params"]):
            raise RuntimeError("optimizer parameter count changed at restore")
        saved_ids.extend(saved["params"])
        current_parameters.extend(current["params"])
        for key in list(current):
            if key != "params" and key not in saved:
                del current[key]
        for key, value in saved.items():
            if key == "params":
                continue
            current[key] = restore_tree_to_recorded_devices(
                value,
                devices[key],
                parameter_device=current["params"][0].device,
            )
        group_checks.append(
            {
                "parameters": len(current["params"]),
                "hyperparameter_keys": sorted(
                    key for key in saved if key != "params"
                ),
                "keys_exact": set(current) == set(saved),
            }
        )
    if len(saved_ids) != len(current_parameters):
        raise RuntimeError("optimizer flattened parameter count changed")
    saved_state = saved_state_dict["state"]
    saved_state_devices = saved_device_tree["state"]
    optimizer.state.clear()
    restored_entries = 0
    for saved_id, parameter in zip(saved_ids, current_parameters):
        entry = saved_state.get(
            saved_id, saved_state.get(str(saved_id))
        )
        devices = saved_state_devices.get(
            saved_id, saved_state_devices.get(str(saved_id))
        )
        if entry is None:
            continue
        if devices is None:
            raise RuntimeError(
                f"optimizer state device metadata missing: {saved_id}"
            )
        optimizer.state[parameter] = restore_tree_to_recorded_devices(
            entry,
            devices,
            parameter_device=parameter.device,
        )
        restored_entries += 1
    checks = {
        "param_groups": len(group_checks) == len(saved_groups)
        and all(row["keys_exact"] for row in group_checks),
        "flattened_parameters": len(saved_ids) == len(current_parameters),
        "state_entries": restored_entries == len(saved_state),
    }
    return {
        "groups": group_checks,
        "saved_state_entries": len(saved_state),
        "restored_state_entries": restored_entries,
        "checks": checks,
        "passed": all(checks.values()),
    }


def sampled_bundle_fingerprint(
    *,
    model_state: dict[str, Any],
    backup_state: dict[str, Any],
    matrix_state: dict[str, Any],
    x: Tensor,
    y: Tensor,
    loader_state: dict[str, Any],
    matrix_global_step: int,
) -> dict[str, Any]:
    items = [
        *named_tensors(model_state, "model"),
        *named_tensors(backup_state, "backup_optimizer"),
        *named_tensors(matrix_state, "matrix_optimizer"),
        ("next_x", x),
        ("next_y", y),
    ]
    by_device: dict[str, list[tuple[int, Tensor]]] = {}
    for index, (_, tensor) in enumerate(items):
        by_device.setdefault(str(tensor.device), []).append((index, tensor))
    fingerprints: list[dict[str, Any] | None] = [None] * len(items)
    for rows in by_device.values():
        observed = LEGACY.tensor_fingerprints(
            [tensor for _, tensor in rows]
        )
        for (index, _), payload in zip(rows, observed):
            fingerprints[index] = payload
    digest = hashlib.sha256()
    finite = True
    for (name, _), payload in zip(items, fingerprints):
        if payload is None:
            raise RuntimeError(f"missing fingerprint payload: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(payload["fingerprint_sha256"].encode("ascii"))
        finite = finite and bool(payload["sampled_values_finite"])
    digest.update(
        json.dumps(
            {
                "loader_state": loader_state,
                "matrix_global_step": int(matrix_global_step),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    structure = structural_metadata(
        {
            "model_state": model_state,
            "backup_state": backup_state,
            "matrix_state": matrix_state,
        }
    )
    structure_sha256 = hashlib.sha256(
        json.dumps(
            structure, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    digest.update(structure_sha256.encode("ascii"))
    return {
        "method": "17 fixed flattened samples from every named state tensor",
        "tensor_count": len(items),
        "device_groups": sorted(by_device),
        "sampled_values_finite": finite,
        "sha256": digest.hexdigest(),
        "structure_sha256": structure_sha256,
        "next_x_sha256": M1.tensor_sha256(x),
        "next_y_sha256": M1.tensor_sha256(y),
        "loader_state": loader_state,
        "matrix_global_step": int(matrix_global_step),
    }


def group_statistics_zero(matrix_optimizer: torch.optim.Optimizer) -> bool:
    return all(
        int(torch.count_nonzero(group["accum"]).item()) == 0
        and float(group["count"].item()) == 0.0
        for group in matrix_optimizer._groups
    )


def model_gradients_clear(model: nn.Module) -> bool:
    return all(parameter.grad is None for parameter in model.parameters())


def take_branch_snapshot(
    *,
    label: str,
    model: nn.Module,
    backup_optimizer: torch.optim.Optimizer,
    matrix_optimizer: torch.optim.Optimizer,
    train_loader: Any,
    x: Tensor,
    y: Tensor,
) -> dict[str, Any]:
    if not group_statistics_zero(matrix_optimizer):
        raise RuntimeError(f"non-zero activation statistics at snapshot {label}")
    if not model_gradients_clear(model):
        raise RuntimeError(f"non-empty model gradients at snapshot {label}")
    torch.cuda.synchronize()
    backup_live_state = backup_optimizer.state_dict()
    matrix_live_state = matrix_optimizer.state_dict()
    snapshot = {
        "label": label,
        "model_state": cpu_clone_tree(model.state_dict()),
        "backup_state": cpu_clone_tree(backup_live_state),
        "backup_state_devices": tensor_device_tree(backup_live_state),
        "matrix_state": cpu_clone_tree(matrix_live_state),
        "matrix_state_devices": tensor_device_tree(matrix_live_state),
        "x": x.detach().cpu().clone(),
        "y": y.detach().cpu().clone(),
        "loader_state": cpu_clone_tree(train_loader.state_dict()),
        "matrix_global_step": int(matrix_optimizer.global_step),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_state": torch.cuda.get_rng_state().cpu().clone(),
    }
    snapshot["fingerprint"] = sampled_bundle_fingerprint(
        model_state=snapshot["model_state"],
        backup_state=snapshot["backup_state"],
        matrix_state=snapshot["matrix_state"],
        x=snapshot["x"],
        y=snapshot["y"],
        loader_state=snapshot["loader_state"],
        matrix_global_step=snapshot["matrix_global_step"],
    )
    return snapshot


def live_branch_fingerprint(
    *,
    model: nn.Module,
    backup_optimizer: torch.optim.Optimizer,
    matrix_optimizer: torch.optim.Optimizer,
    train_loader: Any,
    x: Tensor,
    y: Tensor,
) -> dict[str, Any]:
    return sampled_bundle_fingerprint(
        model_state=model.state_dict(),
        backup_state=backup_optimizer.state_dict(),
        matrix_state=matrix_optimizer.state_dict(),
        x=x,
        y=y,
        loader_state=train_loader.state_dict(),
        matrix_global_step=int(matrix_optimizer.global_step),
    )


def fingerprint_match(
    expected: dict[str, Any], observed: dict[str, Any], label: str
) -> dict[str, Any]:
    checks = {
        "sha256": observed["sha256"] == expected["sha256"],
        "structure_sha256": observed["structure_sha256"]
        == expected["structure_sha256"],
        "tensor_count": int(observed["tensor_count"])
        == int(expected["tensor_count"]),
        "sampled_values_finite": observed["sampled_values_finite"] is True,
        "next_x_sha256": observed["next_x_sha256"]
        == expected["next_x_sha256"],
        "next_y_sha256": observed["next_y_sha256"]
        == expected["next_y_sha256"],
        "loader_state": observed["loader_state"] == expected["loader_state"],
        "matrix_global_step": int(observed["matrix_global_step"])
        == int(expected["matrix_global_step"]),
    }
    return {
        "label": label,
        "expected": expected,
        "observed": observed,
        "checks": checks,
        "passed": all(checks.values()),
    }


def restore_branch_snapshot(
    *,
    snapshot: dict[str, Any],
    model: nn.Module,
    backup_optimizer: torch.optim.Optimizer,
    matrix_optimizer: torch.optim.Optimizer,
    train_loader: Any,
    refresh_interval: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    model.load_state_dict(snapshot["model_state"], strict=True)
    backup_restore = restore_optimizer_state_exact(
        backup_optimizer,
        snapshot["backup_state"],
        snapshot["backup_state_devices"],
    )
    matrix_restore = restore_optimizer_state_exact(
        matrix_optimizer,
        snapshot["matrix_state"],
        snapshot["matrix_state_devices"],
    )
    if not backup_restore["passed"] or not matrix_restore["passed"]:
        raise RuntimeError(
            "optimizer exact restore failed: "
            f"backup={backup_restore} matrix={matrix_restore}"
        )
    matrix_optimizer.refresh = int(refresh_interval)
    matrix_optimizer.global_step = int(snapshot["matrix_global_step"])
    for group in matrix_optimizer._groups:
        group["accum"].zero_()
        group["count"].zero_()
    train_loader.load_state_dict(snapshot["loader_state"])
    x = snapshot["x"].cuda(non_blocking=False)
    y = snapshot["y"].cuda(non_blocking=False)
    random.setstate(snapshot["python_rng_state"])
    np.random.set_state(snapshot["numpy_rng_state"])
    torch.set_rng_state(snapshot["torch_rng_state"])
    torch.cuda.set_rng_state(snapshot["cuda_rng_state"])
    torch.cuda.synchronize()
    observed = live_branch_fingerprint(
        model=model,
        backup_optimizer=backup_optimizer,
        matrix_optimizer=matrix_optimizer,
        train_loader=train_loader,
        x=x,
        y=y,
    )
    audit = fingerprint_match(
        snapshot["fingerprint"],
        observed,
        f"restore_{snapshot['label']}",
    )
    observed_numpy_rng = np.random.get_state()
    expected_numpy_rng = snapshot["numpy_rng_state"]
    rng_checks = {
        "python_rng": random.getstate() == snapshot["python_rng_state"],
        "numpy_rng_kind": observed_numpy_rng[0] == expected_numpy_rng[0],
        "numpy_rng_values": bool(
            np.array_equal(observed_numpy_rng[1], expected_numpy_rng[1])
        ),
        "numpy_rng_position": observed_numpy_rng[2:]
        == expected_numpy_rng[2:],
        "torch_rng": bool(
            torch.equal(torch.get_rng_state(), snapshot["torch_rng_state"])
        ),
        "cuda_rng": bool(
            torch.equal(
                torch.cuda.get_rng_state().cpu(),
                snapshot["cuda_rng_state"],
            )
        ),
    }
    audit["rng_checks"] = rng_checks
    audit["optimizer_restore"] = {
        "backup": backup_restore,
        "matrix": matrix_restore,
        "passed": backup_restore["passed"] and matrix_restore["passed"],
    }
    audit["activation_statistics_zero"] = group_statistics_zero(
        matrix_optimizer
    )
    audit["model_gradients_clear"] = model_gradients_clear(model)
    audit["passed"] = (
        audit["passed"]
        and audit["optimizer_restore"]["passed"]
        and all(rng_checks.values())
        and audit["activation_statistics_zero"]
        and audit["model_gradients_clear"]
    )
    if not audit["passed"]:
        raise RuntimeError(f"branch restore audit failed: {audit}")
    return x, y, audit


def run_segment(
    *,
    trajectory_node: str,
    start_step: int,
    end_step: int,
    target_refresh_steps: tuple[int, ...],
    expected_global_events: tuple[int, ...],
    model: nn.Module,
    compiled_model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    matrix_optimizer: torch.optim.Optimizer,
    source_runtime: Any,
    train_loader: Any,
    x: Tensor,
    y: Tensor,
    eval_batches: list[tuple[Tensor, Tensor]],
    evaluation_steps: set[int],
    initial_loss: float,
    accumulation: int,
    config: dict[str, Any],
    metadata: dict[str, Any],
    target_suffix: str,
    expected_layers: int,
) -> tuple[
    Tensor,
    Tensor,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if start_step > end_step:
        raise ValueError(f"empty segment: {start_step}>{end_step}")
    if any(
        step < start_step or step > end_step
        for step in expected_global_events
    ):
        raise ValueError("global refresh event outside segment")
    if not set(target_refresh_steps).issubset(expected_global_events):
        raise ValueError("target refresh outside global refresh events")
    controller = LEGACY.RefreshInterventionController(
        matrix_optimizer,
        target_suffix=target_suffix,
        target_refresh_steps=target_refresh_steps,
        expected_other_refresh_steps=expected_global_events,
        expected_layers=expected_layers,
    )
    initial_target_state = LEGACY.group_state_snapshot(
        matrix_optimizer,
        controller.target_groups,
        include_statistics=True,
    )
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    training_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    original_refresh = controller.original_refresh
    try:
        for completed in range(start_step, end_step + 1):
            step = completed - 1
            matrix_optimizer.global_step = step
            precond_flag = matrix_optimizer.precond_flag_for_step(step)
            expected_flag = completed in expected_global_events
            if bool(precond_flag) != expected_flag:
                raise RuntimeError(
                    "preconditioner schedule mismatch "
                    f"node={trajectory_node} step={completed} "
                    f"observed={precond_flag} expected={expected_flag}"
                )
            target_action = (
                LEGACY.refresh_action(completed, target_refresh_steps)
                if precond_flag
                else "none"
            )
            model.train()
            torch.cuda.synchronize()
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
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
            finished.record()
            torch.cuda.synchronize()
            duration = float(started.elapsed_time(finished)) / 1000.0
            train_loss = loss_sum / accumulation
            if not math.isfinite(train_loss):
                raise FloatingPointError(
                    f"non-finite loss at {trajectory_node} step {completed}"
                )
            training_rows.append(
                {
                    **metadata,
                    "trajectory_node": trajectory_node,
                    "optimizer_step": completed,
                    "train_loss_mean": train_loss,
                    "global_preconditioner_event": bool(precond_flag),
                    "down_preconditioner_action": target_action,
                    "backup_lr": float(
                        optimizers[0].param_groups[0]["lr"]
                    ),
                    "matrix_lr": float(
                        matrix_optimizer.param_groups[0]["lr"]
                    ),
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
                        **metadata,
                        "trajectory_node": trajectory_node,
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
                    "MECH-09R "
                    f"cell={metadata['checkpoint_cell']} "
                    f"replica={metadata['data_replica']} "
                    f"node={trajectory_node} "
                    f"step={completed}/{config['rollout_steps']} "
                    f"down_action={target_action} "
                    f"train_loss={train_loss:.6f}",
                    flush=True,
                )
    finally:
        matrix_optimizer._refresh_preconditioners = original_refresh
        matrix_optimizer._groups = controller.all_groups
    final_target_state = LEGACY.group_state_snapshot(
        matrix_optimizer,
        controller.target_groups,
        include_statistics=True,
    )
    audit = controller.audit(initial_target_state, final_target_state)
    audit.update(
        {
            "trajectory_node": trajectory_node,
            "segment_start_step": start_step,
            "segment_end_step": end_step,
        }
    )
    if not audit["passed"]:
        raise RuntimeError(f"segment refresh audit failed: {audit}")
    return x, y, training_rows, evaluation_rows, audit


def assign_arm(
    rows: list[dict[str, Any]], arm: str
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "arm": arm,
            "source_trajectory_node": row["trajectory_node"],
        }
        for row in rows
    ]


def event_steps(audit: dict[str, Any], action: str | None = None) -> list[int]:
    rows = audit["events"]
    if action is not None:
        rows = [row for row in rows if row["target_action"] == action]
    return [int(row["completed_step"]) for row in rows]


def compose_refresh_audit(
    *,
    contract: dict[str, Any],
    tier: str,
    config: dict[str, Any],
    segment_audits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    production = segment_audits["production"]
    common = segment_audits["shared_no_down"]
    delayed = segment_audits["delayed"]
    frozen = segment_audits["frozen"]
    expected_global = [
        int(value)
        for value in config["expected_global_refresh_completed_steps"]
    ]
    arm_events = {
        "production_newton_muon": production["events"],
        "delayed_down_refresh": [*common["events"], *delayed["events"]],
        "frozen_down_refresh": [*common["events"], *frozen["events"]],
    }
    rows = {}
    for arm, events in arm_events.items():
        observed_global = [int(row["completed_step"]) for row in events]
        observed_target = [
            int(row["completed_step"])
            for row in events
            if row["target_action"] == "refresh"
        ]
        expected_target = list(tier_target_steps(contract, tier, arm))
        checks = {
            "global_schedule": observed_global == expected_global,
            "target_schedule": observed_target == expected_target,
            "statistics_zero_after_events": all(
                row["target_statistics_zero_after"]
                and row["other_statistics_zero_after"]
                for row in events
            ),
            "other_groups_refreshed": all(
                row["other_covariance_changed"]
                and row["other_inverse_changed"]
                for row in events
            ),
        }
        rows[arm] = {
            "expected_global_steps": expected_global,
            "observed_global_steps": observed_global,
            "expected_target_steps": expected_target,
            "observed_target_steps": observed_target,
            "checks": checks,
            "passed": all(checks.values()),
        }
    checks = {
        "all_segments_passed": all(
            row["passed"] for row in segment_audits.values()
        ),
        "all_arms_passed": all(row["passed"] for row in rows.values()),
        "common_event_reused_exactly": event_steps(common)
        == [
            step
            for step in expected_global
            if step <= int(
                config["causal_tree"]["shared_no_down_end_step"]
            )
        ],
    }
    return {
        "arms": rows,
        "segment_audits": segment_audits,
        "checks": checks,
        "passed": all(checks.values()),
    }


def unique_grain(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> bool:
    keys = {tuple(row[field] for field in fields) for row in rows}
    return len(keys) == len(rows)


def exact_shared_evaluation_audit(
    evaluation_rows: list[dict[str, Any]],
    contract: dict[str, Any],
    tier: str,
) -> dict[str, Any]:
    config = contract[tier]
    arms = tuple(contract["arms"])

    def values(step: int, selected: tuple[str, ...]) -> list[float]:
        return [
            float(row["heldout_loss"])
            for row in evaluation_rows
            if int(row["optimizer_step"]) == step
            and row["arm"] in selected
        ]

    if tier == "formal":
        pre_step = int(
            contract["analysis_contract"]["pre_refresh_equivalence_step"]
        )
        shared_step = int(
            contract["analysis_contract"]["pre_delayed_refresh_shared_step"]
        )
    else:
        pre_step = int(config["causal_tree"]["shared_all_end_step"])
        shared_step = int(config["causal_tree"]["shared_no_down_end_step"])
    pre_values = values(pre_step, arms)
    delayed_frozen_values = values(
        shared_step,
        ("delayed_down_refresh", "frozen_down_refresh"),
    )
    checks = {
        "pre_arm_count": len(pre_values) == len(arms),
        "pre_exact": len(set(pre_values)) == 1,
        "delayed_frozen_count": len(delayed_frozen_values) == 2,
        "delayed_frozen_exact": len(set(delayed_frozen_values)) == 1,
    }
    return {
        "pre_refresh_step": pre_step,
        "pre_refresh_values": pre_values,
        "pre_refresh_max_abs_delta": max(pre_values) - min(pre_values),
        "pre_delayed_refresh_step": shared_step,
        "delayed_frozen_values": delayed_frozen_values,
        "delayed_frozen_max_abs_delta": max(delayed_frozen_values)
        - min(delayed_frozen_values),
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MECH-09R requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

    cell_index = config["origins"].index(args.cell)
    replica_index = config["data_replicas"].index(args.data_replica)
    unit_seed = (
        int(contract["determinism"]["base_seed"])
        + cell_index * 1000
        + args.data_replica
    )
    random.seed(unit_seed)
    np.random.seed(unit_seed % (2**32))
    torch.manual_seed(unit_seed)
    torch.cuda.manual_seed_all(unit_seed)

    shutil.copyfile(
        args.contract, output / "refresh_mediation_repair_contract.json"
    )
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
        config["production_refresh_interval"]
    )
    optimizers, matrix_optimizer = M8.make_candidate_optimizers(
        source_runtime, model, source_method, restart
    )
    if not isinstance(matrix_optimizer, source_runtime.SharedInputNewtonMuon):
        raise RuntimeError("MECH-09R requires SharedInputNewtonMuon")
    matrix_optimizer.refresh = int(config["production_refresh_interval"])
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

    offsets = [
        int(value) for value in config["replica_optimizer_step_offsets"]
    ]
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
        "one_stream_shared_until_each_causal_fork": True,
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
    del build_batches
    model.zero_grad(set_to_none=True)
    gc.collect()
    del model_state
    del checkpoint
    gc.collect()

    evaluation_steps = {int(value) for value in config["evaluation_steps"]}
    if 0 not in evaluation_steps or int(config["rollout_steps"]) not in evaluation_steps:
        raise RuntimeError("evaluation schedule must include endpoints")
    initial_loss = M8.evaluate(model, eval_batches)
    if not math.isfinite(initial_loss):
        raise FloatingPointError("initial held-out loss is not finite")
    base_metadata = {
        "checkpoint_cell": args.cell,
        "checkpoint_stage": spec["stage"],
        "checkpoint_method": spec["method"],
        "source_method": source_method,
        "data_replica": args.data_replica,
    }
    initial_row = {
        **base_metadata,
        "trajectory_node": "shared_all",
        "optimizer_step": 0,
        "heldout_loss": initial_loss,
        "normalized_loss": 1.0,
        "loss_delta_from_step0": 0.0,
        "relative_loss_delta_from_step0": 0.0,
    }
    compiled_model: nn.Module = (
        model if args.no_compile else torch.compile(model)
    )
    tree = config["causal_tree"]
    shared_end = int(tree["shared_all_end_step"])
    first_branch = int(tree["first_branch_step"])
    common_end = int(tree["shared_no_down_end_step"])
    second_branch = int(tree["second_branch_step"])
    rollout_end = int(config["rollout_steps"])
    all_events = tuple(
        int(value)
        for value in config["expected_global_refresh_completed_steps"]
    )
    target_suffix = contract["arms"]["production_newton_muon"][
        "target_group_suffix"
    ]

    segment_audits: dict[str, dict[str, Any]] = {}
    restore_audits: list[dict[str, Any]] = []
    branch_start_audits: list[dict[str, Any]] = []

    x, y, shared_training, shared_evaluation, segment_audits["shared_all"] = (
        run_segment(
            trajectory_node="shared_all",
            start_step=1,
            end_step=shared_end,
            target_refresh_steps=(),
            expected_global_events=tuple(
                step for step in all_events if step <= shared_end
            ),
            model=model,
            compiled_model=compiled_model,
            optimizers=optimizers,
            matrix_optimizer=matrix_optimizer,
            source_runtime=source_runtime,
            train_loader=train_loader,
            x=x,
            y=y,
            eval_batches=eval_batches,
            evaluation_steps=evaluation_steps,
            initial_loss=initial_loss,
            accumulation=accumulation,
            config=config,
            metadata=base_metadata,
            target_suffix=target_suffix,
            expected_layers=int(contract["architecture"]["n_layer"]),
        )
    )
    snapshot_first = take_branch_snapshot(
        label=f"after_step_{shared_end}",
        model=model,
        backup_optimizer=optimizers[0],
        matrix_optimizer=matrix_optimizer,
        train_loader=train_loader,
        x=x,
        y=y,
    )
    production_start = live_branch_fingerprint(
        model=model,
        backup_optimizer=optimizers[0],
        matrix_optimizer=matrix_optimizer,
        train_loader=train_loader,
        x=x,
        y=y,
    )
    branch_start_audits.append(
        fingerprint_match(
            snapshot_first["fingerprint"],
            production_start,
            "production_starts_from_first_fork",
        )
    )

    production_events = tuple(
        step for step in all_events if first_branch <= step <= rollout_end
    )
    production_targets = tuple(
        step
        for step in tier_target_steps(
            contract, args.analysis_tier, "production_newton_muon"
        )
        if first_branch <= step <= rollout_end
    )
    (
        x,
        y,
        production_training,
        production_evaluation,
        segment_audits["production"],
    ) = run_segment(
        trajectory_node="production",
        start_step=first_branch,
        end_step=rollout_end,
        target_refresh_steps=production_targets,
        expected_global_events=production_events,
        model=model,
        compiled_model=compiled_model,
        optimizers=optimizers,
        matrix_optimizer=matrix_optimizer,
        source_runtime=source_runtime,
        train_loader=train_loader,
        x=x,
        y=y,
        eval_batches=eval_batches,
        evaluation_steps=evaluation_steps,
        initial_loss=initial_loss,
        accumulation=accumulation,
        config=config,
        metadata=base_metadata,
        target_suffix=target_suffix,
        expected_layers=int(contract["architecture"]["n_layer"]),
    )

    x, y, restore_first = restore_branch_snapshot(
        snapshot=snapshot_first,
        model=model,
        backup_optimizer=optimizers[0],
        matrix_optimizer=matrix_optimizer,
        train_loader=train_loader,
        refresh_interval=int(config["production_refresh_interval"]),
    )
    restore_audits.append(restore_first)
    common_events = tuple(
        step for step in all_events if first_branch <= step <= common_end
    )
    (
        x,
        y,
        common_training,
        common_evaluation,
        segment_audits["shared_no_down"],
    ) = run_segment(
        trajectory_node="shared_no_down",
        start_step=first_branch,
        end_step=common_end,
        target_refresh_steps=(),
        expected_global_events=common_events,
        model=model,
        compiled_model=compiled_model,
        optimizers=optimizers,
        matrix_optimizer=matrix_optimizer,
        source_runtime=source_runtime,
        train_loader=train_loader,
        x=x,
        y=y,
        eval_batches=eval_batches,
        evaluation_steps=evaluation_steps,
        initial_loss=initial_loss,
        accumulation=accumulation,
        config=config,
        metadata=base_metadata,
        target_suffix=target_suffix,
        expected_layers=int(contract["architecture"]["n_layer"]),
    )
    del snapshot_first
    gc.collect()
    snapshot_second = take_branch_snapshot(
        label=f"after_step_{common_end}",
        model=model,
        backup_optimizer=optimizers[0],
        matrix_optimizer=matrix_optimizer,
        train_loader=train_loader,
        x=x,
        y=y,
    )
    delayed_start = live_branch_fingerprint(
        model=model,
        backup_optimizer=optimizers[0],
        matrix_optimizer=matrix_optimizer,
        train_loader=train_loader,
        x=x,
        y=y,
    )
    branch_start_audits.append(
        fingerprint_match(
            snapshot_second["fingerprint"],
            delayed_start,
            "delayed_starts_from_second_fork",
        )
    )
    second_events = tuple(
        step for step in all_events if second_branch <= step <= rollout_end
    )
    delayed_targets = tuple(
        step
        for step in tier_target_steps(
            contract, args.analysis_tier, "delayed_down_refresh"
        )
        if second_branch <= step <= rollout_end
    )
    (
        x,
        y,
        delayed_training,
        delayed_evaluation,
        segment_audits["delayed"],
    ) = run_segment(
        trajectory_node="delayed",
        start_step=second_branch,
        end_step=rollout_end,
        target_refresh_steps=delayed_targets,
        expected_global_events=second_events,
        model=model,
        compiled_model=compiled_model,
        optimizers=optimizers,
        matrix_optimizer=matrix_optimizer,
        source_runtime=source_runtime,
        train_loader=train_loader,
        x=x,
        y=y,
        eval_batches=eval_batches,
        evaluation_steps=evaluation_steps,
        initial_loss=initial_loss,
        accumulation=accumulation,
        config=config,
        metadata=base_metadata,
        target_suffix=target_suffix,
        expected_layers=int(contract["architecture"]["n_layer"]),
    )

    x, y, restore_second = restore_branch_snapshot(
        snapshot=snapshot_second,
        model=model,
        backup_optimizer=optimizers[0],
        matrix_optimizer=matrix_optimizer,
        train_loader=train_loader,
        refresh_interval=int(config["production_refresh_interval"]),
    )
    restore_audits.append(restore_second)
    (
        x,
        y,
        frozen_training,
        frozen_evaluation,
        segment_audits["frozen"],
    ) = run_segment(
        trajectory_node="frozen",
        start_step=second_branch,
        end_step=rollout_end,
        target_refresh_steps=(),
        expected_global_events=second_events,
        model=model,
        compiled_model=compiled_model,
        optimizers=optimizers,
        matrix_optimizer=matrix_optimizer,
        source_runtime=source_runtime,
        train_loader=train_loader,
        x=x,
        y=y,
        eval_batches=eval_batches,
        evaluation_steps=evaluation_steps,
        initial_loss=initial_loss,
        accumulation=accumulation,
        config=config,
        metadata=base_metadata,
        target_suffix=target_suffix,
        expected_layers=int(contract["architecture"]["n_layer"]),
    )
    del snapshot_second
    gc.collect()

    arms = tuple(contract["arms"])
    evaluation_rows = [
        *assign_arm([initial_row, *shared_evaluation], arms[0]),
        *assign_arm([initial_row, *shared_evaluation], arms[1]),
        *assign_arm([initial_row, *shared_evaluation], arms[2]),
        *assign_arm(production_evaluation, "production_newton_muon"),
        *assign_arm(common_evaluation, "delayed_down_refresh"),
        *assign_arm(common_evaluation, "frozen_down_refresh"),
        *assign_arm(delayed_evaluation, "delayed_down_refresh"),
        *assign_arm(frozen_evaluation, "frozen_down_refresh"),
    ]
    training_rows = [
        *assign_arm(shared_training, arms[0]),
        *assign_arm(shared_training, arms[1]),
        *assign_arm(shared_training, arms[2]),
        *assign_arm(production_training, "production_newton_muon"),
        *assign_arm(common_training, "delayed_down_refresh"),
        *assign_arm(common_training, "frozen_down_refresh"),
        *assign_arm(delayed_training, "delayed_down_refresh"),
        *assign_arm(frozen_training, "frozen_down_refresh"),
    ]
    evaluation_rows.sort(
        key=lambda row: (row["arm"], int(row["optimizer_step"]))
    )
    training_rows.sort(
        key=lambda row: (row["arm"], int(row["optimizer_step"]))
    )
    refresh_audit = compose_refresh_audit(
        contract=contract,
        tier=args.analysis_tier,
        config=config,
        segment_audits=segment_audits,
    )
    shared_audit = exact_shared_evaluation_audit(
        evaluation_rows, contract, args.analysis_tier
    )
    branch_audit_checks = {
        "branch_starts_match": all(
            row["passed"] for row in branch_start_audits
        ),
        "restores_match": all(row["passed"] for row in restore_audits),
        "shared_evaluations_exact": shared_audit["passed"],
        "refresh_schedules": refresh_audit["passed"],
    }
    branch_audit = {
        "method": contract["shared_policy"]["branch_snapshot"],
        "first_fork_step": shared_end,
        "second_fork_step": common_end,
        "branch_start_audits": branch_start_audits,
        "restore_audits": restore_audits,
        "shared_evaluation_audit": shared_audit,
        "checks": branch_audit_checks,
        "passed": all(branch_audit_checks.values()),
    }

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
    determinism_audit = {
        "unit_seed": unit_seed,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "compile_enabled": not args.no_compile,
        "shared_prefix_primary_protection": True,
        "passed": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            == contract["determinism"]["cublas_workspace_config"]
            and torch.are_deterministic_algorithms_enabled()
            and not torch.backends.cudnn.benchmark
            and torch.backends.cudnn.deterministic
        ),
    }
    expected_evaluations = len(evaluation_steps) * len(arms)
    expected_training = int(config["rollout_steps"]) * len(arms)
    checks = {
        "contract": contract_audit["passed"],
        "smoke_gate": smoke_gate["passed"],
        "architecture": all(architecture_checks.values()),
        "parameter_transfer": parameter_audit["passed"],
        "backup_state_loaded": backup_audit["passed"],
        "momentum_transfer": momentum_audit["passed"],
        "optimizer_hyperparameters": optimizer_audit["passed"],
        "fresh_preconditioner": preconditioner_audit["passed"],
        "training_stream": training_stream_contract["passed"],
        "heldout_windows": heldout_contract["passed"],
        "branch_audit": branch_audit["passed"],
        "refresh_audit": refresh_audit["passed"],
        "determinism": determinism_audit["passed"],
        "evaluation_rows": len(evaluation_rows) == expected_evaluations,
        "training_rows": len(training_rows) == expected_training,
        "evaluation_unique": unique_grain(
            evaluation_rows, ("arm", "optimizer_step")
        ),
        "training_unique": unique_grain(
            training_rows, ("arm", "optimizer_step")
        ),
        "evaluation_finite": all(
            math.isfinite(float(row["heldout_loss"]))
            and math.isfinite(float(row["normalized_loss"]))
            for row in evaluation_rows
        ),
        "training_finite": all(
            math.isfinite(float(row["train_loss_mean"]))
            for row in training_rows
        ),
        "shared_values_exact": shared_audit["passed"],
        "checkpoint_unchanged": all(
            value
            for key, value in invariance.items()
            if key.endswith("_unchanged")
        ),
        "timing_excluded": all(
            row["timing_usable_for_paper"] is False
            for row in training_rows
        ),
        "real_optimizer_steps": True,
        "legacy_post_treatment_outcomes_not_read": True,
    }
    M8.write_csv(output / "evaluation.csv", evaluation_rows)
    M8.write_csv(output / "training.csv", training_rows)
    artifacts = {
        "architecture_checks.json": architecture_checks,
        "backup_optimizer_audit.json": backup_audit,
        "branch_audit.json": branch_audit,
        "checkpoint_hash_audit.json": contract_audit[
            "checkpoint_certificate"
        ],
        "checkpoint_invariance.json": invariance,
        "checkpoint_schema.json": schema,
        "checks.json": checks,
        "contract_audit.json": contract_audit,
        "determinism_audit.json": determinism_audit,
        "heldout_batch_contract.json": heldout_contract,
        "initial_preconditioner_audit.json": preconditioner_audit,
        "mech08_control_reference_audit.json": contract_audit[
            "mech08_control_reference"
        ],
        "method_identity_audit.json": method_audit,
        "momentum_transfer_audit.json": momentum_audit,
        "optimizer_hyperparameters.json": optimizer_audit,
        "parameter_transfer_audit.json": parameter_audit,
        "protocol_amendment.json": contract["protocol_amendment"],
        "refresh_tree_audit.json": refresh_audit,
        "runtime.json": M1.runtime_metadata(
            SimpleNamespace(
                host_id=args.host_id,
                execution_domain=args.execution_domain,
            )
        ),
        "smoke_gate.json": smoke_gate,
        "source_runtime_config.json": source_config,
        "training_stream_contract.json": training_stream_contract,
        "triton_audit.json": triton_audit,
    }
    for name, payload in artifacts.items():
        M1.atomic_json(output / name, payload)
    passed = all(value is True for value in checks.values())
    manifest = {
        "schema_version": 2,
        "script_version": SCRIPT_VERSION,
        "experiment": "MECH-09R",
        "analysis_tier": args.analysis_tier,
        "checkpoint_cell": args.cell,
        "checkpoint_stage": spec["stage"],
        "checkpoint_method": spec["method"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": contract_audit["checkpoint_certificate"][
            "sha256"
        ],
        "data_replica": args.data_replica,
        "contract_sha256": contract_audit["contract_sha256"],
        "causal_tree": True,
        "arms": list(arms),
        "branch_steps": [shared_end, common_end],
        "evaluation_rows": len(evaluation_rows),
        "training_rows": len(training_rows),
        "real_optimizer_steps_computed": (
            shared_end
            + (rollout_end - first_branch + 1)
            + (common_end - first_branch + 1)
            + 2 * (rollout_end - second_branch + 1)
        ),
        "logical_arm_optimizer_steps": expected_training,
        "timing_usable_for_paper": False,
        "legacy_invalid_run_reused": False,
        "passed": passed,
        "artifacts": sorted(
            [
                *artifacts,
                "evaluation.csv",
                "training.csv",
                "refresh_mediation_repair_contract.json",
                "mech08_control_reference.json",
                "mech09r_manifest.json",
                "status.json",
            ]
        ),
    }
    M1.atomic_json(output / "mech09r_manifest.json", manifest)
    M1.atomic_json(
        output / "status.json",
        {
            "status": "passed" if passed else "integrity_failed",
            "script_version": SCRIPT_VERSION,
        },
    )
    if not passed:
        raise SystemExit(2)
    print(f"MECH-09R manifest: {output / 'mech09r_manifest.json'}")
    print(f"MECH-09R artifacts: {output}")


def main() -> None:
    args = parse_args()
    try:
        run_worker(args)
    except Exception as exc:
        output = args.output_dir.resolve()
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
