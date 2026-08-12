#!/usr/bin/env python3
"""Run one matched-start MECH-08 LLaMA-1B optimizer rollout."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn


SCRIPT_VERSION = "2026-07-27.2"
CONTRACT_VERSION = "2026-07-27.1"
HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M1 = load_module(
    "mech08_m1",
    HERE.parent / "27_mech01_unified_k_diagnostics" / "mech01_worker.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--analysis-tier", required=True, choices=("smoke", "formal"))
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--data-replica", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-hash-certificate", required=True, type=Path)
    parser.add_argument("--source-script", required=True, type=Path)
    parser.add_argument("--profile-script", required=True, type=Path)
    parser.add_argument("--triton-kernels", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--prediction-reference", required=True, type=Path)
    parser.add_argument("--train-data-pattern", required=True)
    parser.add_argument("--val-data-pattern", required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--execution-domain", required=True)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Debug only. Formal controller never sets this flag.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_spec(contract: dict[str, Any], cell: str) -> dict[str, Any]:
    matches = [row for row in contract["checkpoints"] if row["cell"] == cell]
    if len(matches) != 1:
        raise RuntimeError(f"contract checkpoint cell is not unique: {cell}")
    return matches[0]


def validate_contract(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = read_json(args.contract.resolve())
    spec = checkpoint_spec(contract, args.cell)
    certificate = read_json(args.checkpoint_hash_certificate.resolve())
    config = contract[args.analysis_tier]
    algorithm = contract["algorithms"].get(args.algorithm)
    primary = contract["comparison_contract"]["primary"]
    checks = {
        "contract_version": contract.get("contract_version") == CONTRACT_VERSION,
        "family": contract.get("family") == "llama1b",
        "cell_in_tier": args.cell in config["origins"],
        "algorithm_in_tier": args.algorithm in config["algorithms"],
        "algorithm_known": isinstance(algorithm, dict),
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
        "prediction_reference_sha": M1.sha256_file(
            args.prediction_reference.resolve()
        )
        == contract["prediction_reference"]["sha256"],
        "algorithms_exact": set(contract["algorithms"])
        == {
            "muon",
            "original_newton_muon",
            "selective_diag",
            "selective_none",
        },
        "primary_contrasts_exact": {
            (row["left"], row["right"]) for row in primary
        }
        == {
            ("selective_diag", "muon"),
            ("selective_none", "muon"),
            ("selective_diag", "original_newton_muon"),
            ("selective_none", "original_newton_muon"),
        },
        "diag_none_excluded": "selective_diag_vs_selective_none"
        in contract["comparison_contract"]["excluded_from_primary"],
        "efficiency_excluded": contract["scope_boundary"][
            "efficiency_benchmark_excluded"
        ]
        is True,
    }
    audit = {
        "contract_sha256": M1.sha256_file(args.contract.resolve()),
        "checkpoint_spec": spec,
        "algorithm_spec": algorithm,
        "checkpoint_certificate": certificate,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return contract, config, audit


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
        "jobs": int(manifest.get("completed_jobs", -1)) == 4,
    }
    return {
        "required": True,
        "manifest": str(args.smoke_manifest.resolve()),
        "manifest_sha256": M1.sha256_file(args.smoke_manifest.resolve()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def method_identity_audit(
    checkpoint: dict[str, Any], expected_method: str, inferred_method: str
) -> dict[str, Any]:
    optimizers = checkpoint.get("optimizers")
    matrix_state = (
        optimizers[1]
        if isinstance(optimizers, list)
        and len(optimizers) >= 2
        and isinstance(optimizers[1], dict)
        else {}
    )
    entries = (
        list(matrix_state.get("state", {}).values())
        if isinstance(matrix_state.get("state"), dict)
        else []
    )
    keys = sorted(
        {
            str(key)
            for entry in entries
            if isinstance(entry, dict)
            for key in entry
        }
    )
    momentum_tensors = sum(
        isinstance(entry, dict) and isinstance(entry.get("momentum"), Tensor)
        for entry in entries
    )
    exact = inferred_method == expected_method
    muon_signature = (
        expected_method == "muon"
        and inferred_method == "muon_or_adamw"
        and bool(entries)
        and momentum_tensors == len(entries)
        and "exp_avg" not in keys
        and "exp_avg_sq" not in keys
    )
    return {
        "expected_method": expected_method,
        "model_state_inferred_method": inferred_method,
        "matrix_optimizer_state_entries": len(entries),
        "matrix_optimizer_state_keys": keys,
        "matrix_optimizer_momentum_tensors": momentum_tensors,
        "muon_not_adamw_state_signature": muon_signature,
        "passed": exact or muon_signature,
    }


def copy_checkpoint_parameters(
    model: nn.Module, model_state: dict[str, Tensor]
) -> dict[str, Any]:
    rows = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            source = model_state.get(name)
            if not isinstance(source, Tensor):
                raise RuntimeError(f"checkpoint parameter is missing: {name}")
            if tuple(source.shape) != tuple(parameter.shape):
                raise RuntimeError(
                    f"checkpoint parameter shape mismatch for {name}: "
                    f"{tuple(source.shape)} != {tuple(parameter.shape)}"
                )
            parameter.copy_(source)
            rows.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "numel": parameter.numel(),
                }
            )
        copied_buffers = []
        buffers = dict(model.named_buffers())
        for name in ("rope_cos", "rope_sin"):
            destination = buffers.get(name)
            source = model_state.get(name)
            if not isinstance(destination, Tensor) or not isinstance(source, Tensor):
                raise RuntimeError(f"checkpoint positional buffer is missing: {name}")
            if tuple(source.shape) != tuple(destination.shape):
                raise RuntimeError(
                    f"checkpoint positional buffer shape mismatch for {name}"
                )
            destination.copy_(source)
            copied_buffers.append(name)
    return {
        "named_parameters": len(rows),
        "parameter_numel": sum(row["numel"] for row in rows),
        "all_parameters_copied": len(rows) == len(list(model.named_parameters())),
        "copied_checkpoint_buffers": copied_buffers,
        "algorithm_specific_activation_buffers_copied": False,
        "rows": rows,
        "passed": bool(rows) and copied_buffers == ["rope_cos", "rope_sin"],
    }


def make_candidate_optimizers(
    source_runtime: Any,
    model: nn.Module,
    source_method: str,
    restart: dict[str, Any],
) -> tuple[list[torch.optim.Optimizer], torch.optim.Optimizer]:
    optimizer_args = SimpleNamespace(
        method=source_method,
        backup_lr=float(restart["backup_lr"]),
        matrix_lr=float(restart["matrix_lr"]),
        adamw_matrix_lr=float(restart["matrix_lr"]),
    )
    optimizers, matrix_optimizer = source_runtime.make_optimizers(
        model, optimizer_args
    )
    if matrix_optimizer is None:
        raise RuntimeError("matrix optimizer is missing")
    optimizers[0].param_groups[0]["lr"] = float(restart["backup_lr"])
    matrix_optimizer.param_groups[0]["lr"] = float(restart["matrix_lr"])
    return optimizers, matrix_optimizer


def tensor_transfer_audit(source: Tensor, destination: Tensor) -> dict[str, Any]:
    """Verify a cross-device/dtype copy in the destination arithmetic domain."""
    with torch.no_grad():
        expected = source.detach().to(
            device=destination.device,
            dtype=destination.dtype,
        )
        values_match_exactly = bool(torch.equal(destination, expected))
        expected_finite = bool(torch.isfinite(expected).all().item())
        destination_finite = bool(torch.isfinite(destination).all().item())
        if destination.numel():
            max_abs_diff = float(
                (destination.float() - expected.float()).abs().max().item()
            )
        else:
            max_abs_diff = 0.0
        expected_norm = float(expected.float().norm().item())
        destination_norm = float(destination.float().norm().item())
    return {
        "validation_domain": "destination_device_and_dtype",
        "values_match_exactly": values_match_exactly,
        "expected_finite": expected_finite,
        "destination_finite": destination_finite,
        "max_abs_diff": max_abs_diff,
        "expected_destination_norm": expected_norm,
        "destination_norm": destination_norm,
        "passed": (
            values_match_exactly
            and expected_finite
            and destination_finite
        ),
    }


def transfer_optimizer_state(
    checkpoint: dict[str, Any],
    optimizers: list[torch.optim.Optimizer],
    matrix_optimizer: torch.optim.Optimizer,
    model: nn.Module,
) -> tuple[dict[str, Any], dict[str, Any]]:
    saved = checkpoint.get("optimizers")
    if not isinstance(saved, list) or len(saved) < 2:
        raise RuntimeError("checkpoint optimizer states are missing")
    optimizers[0].load_state_dict(saved[0])

    saved_matrix = saved[1]
    saved_ids = M1.flatten_param_ids(saved_matrix.get("param_groups"))
    current_named = list(model.matrix_named_parameters())
    current_params = [
        parameter
        for group in matrix_optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(saved_ids) != len(current_named) or len(current_named) != len(current_params):
        raise RuntimeError(
            "matrix optimizer parameter-order mismatch: "
            f"saved={len(saved_ids)} named={len(current_named)} "
            f"optimizer={len(current_params)}"
        )
    saved_state = saved_matrix.get("state", {})
    rows = []
    with torch.no_grad():
        for index, (saved_id, named, parameter) in enumerate(
            zip(saved_ids, current_named, current_params)
        ):
            name, named_parameter = named
            if named_parameter is not parameter:
                raise RuntimeError(f"matrix optimizer order mismatch at {name}")
            entry = saved_state.get(saved_id, saved_state.get(str(saved_id), {}))
            momentum = entry.get("momentum") if isinstance(entry, dict) else None
            if not isinstance(momentum, Tensor):
                raise RuntimeError(f"historical momentum is missing for {name}")
            if tuple(momentum.shape) != tuple(parameter.shape):
                raise RuntimeError(f"historical momentum shape mismatch for {name}")
            destination = matrix_optimizer.state[parameter].get("momentum")
            if destination is None:
                destination = torch.empty_like(parameter, dtype=torch.float32)
                matrix_optimizer.state[parameter]["momentum"] = destination
            destination.copy_(momentum)
            transfer_audit = tensor_transfer_audit(momentum, destination)
            rows.append(
                {
                    "index": index,
                    "name": name,
                    "saved_parameter_id": str(saved_id),
                    "shape": list(momentum.shape),
                    "source_dtype": str(momentum.dtype),
                    "destination_dtype": str(destination.dtype),
                    "source_norm": float(momentum.float().norm().item()),
                    **transfer_audit,
                }
            )
    for optimizer, lr in (
        (optimizers[0], float(optimizers[0].param_groups[0]["lr"])),
        (matrix_optimizer, float(matrix_optimizer.param_groups[0]["lr"])),
    ):
        optimizer.param_groups[0]["lr"] = lr
    backup_audit = {
        "state_entries": len(optimizers[0].state),
        "param_groups": len(optimizers[0].param_groups),
        "loaded_from_origin": True,
        "passed": bool(optimizers[0].state),
    }
    momentum_audit = {
        "policy": "historical momentum only; no origin K state transferred",
        "rows": rows,
        "parameters": len(rows),
        "all_present": len(rows) == len(current_named),
        "all_values_match_exactly": all(
            row["values_match_exactly"] for row in rows
        ),
        "all_values_finite": all(
            row["expected_finite"] and row["destination_finite"]
            for row in rows
        ),
        "all_norms_match": all(
            math.isclose(
                row["expected_destination_norm"],
                row["destination_norm"],
                rel_tol=0.0,
                abs_tol=0.0,
            )
            for row in rows
        ),
    }
    momentum_audit["passed"] = (
        momentum_audit["all_present"]
        and momentum_audit["all_values_match_exactly"]
        and momentum_audit["all_values_finite"]
    )
    return backup_audit, momentum_audit


def optimizer_hyperparameter_audit(
    optimizers: list[torch.optim.Optimizer],
    matrix_optimizer: torch.optim.Optimizer,
    source_runtime: Any,
    restart: dict[str, Any],
) -> dict[str, Any]:
    backup = optimizers[0].param_groups[0]
    matrix = matrix_optimizer.param_groups[0]
    observed: dict[str, Any] = {
        "backup_lr": float(backup["lr"]),
        "backup_betas": [float(value) for value in backup["betas"]],
        "backup_weight_decay": float(backup["weight_decay"]),
        "matrix_lr": float(matrix["lr"]),
        "matrix_momentum": float(matrix["momentum"]),
        "matrix_ns_steps": int(matrix["ns_steps"]),
        "matrix_optimizer_class": type(matrix_optimizer).__name__,
    }
    checks = {
        "backup_lr": observed["backup_lr"] == float(restart["backup_lr"]),
        "backup_betas": observed["backup_betas"] == [0.9, 0.95],
        "backup_weight_decay": observed["backup_weight_decay"]
        == float(restart["weight_decay"]),
        "matrix_lr": observed["matrix_lr"] == float(restart["matrix_lr"]),
        "matrix_momentum": observed["matrix_momentum"]
        == float(restart["matrix_momentum"]),
        "matrix_ns_steps": observed["matrix_ns_steps"]
        == int(restart["ns_steps"]),
    }
    if isinstance(matrix_optimizer, source_runtime.SharedInputNewtonMuon):
        observed.update(
            {
                "newton_input_beta": float(matrix_optimizer.input_beta),
                "newton_input_ridge": float(matrix_optimizer.input_ridge),
                "newton_refresh_steps": int(matrix_optimizer.refresh),
                "newton_init_scale": float(matrix_optimizer.init_scale),
            }
        )
        checks.update(
            {
                "newton_input_beta": observed["newton_input_beta"]
                == float(restart["newton_input_beta"]),
                "newton_input_ridge": observed["newton_input_ridge"]
                == float(restart["newton_input_ridge"]),
                "newton_refresh_steps": observed["newton_refresh_steps"]
                == int(restart["newton_refresh_steps"]),
                "newton_init_scale": observed["newton_init_scale"]
                == float(restart["newton_init_scale"]),
            }
        )
    return {
        "observed": observed,
        "expected": restart,
        "checks": checks,
        "passed": all(checks.values()),
    }


def set_loader_global_offset(
    loader: Any, source_runtime: Any, token_offset: int
) -> dict[str, Any]:
    offset = int(token_offset) % int(loader.total_tokens)
    remaining = offset
    selected = None
    for index, filename in enumerate(loader.files):
        tokens = int(source_runtime.peek_data_shard(filename))
        if remaining < tokens:
            selected = (index, remaining, tokens)
            break
        remaining -= tokens
    if selected is None:
        raise RuntimeError("global token offset mapping failed")
    shard, position, shard_tokens = selected
    count = loader.batch_size * loader.sequence_length
    if position + count + 1 > shard_tokens:
        raise RuntimeError(
            f"frozen token window crosses a shard boundary: offset={token_offset}"
        )
    loader.current_shard = shard
    loader.current_position = position
    loader.tokens = source_runtime.load_data_shard(loader.files[shard])
    return {
        "global_token_offset": int(token_offset),
        "wrapped_global_token_offset": offset,
        "current_shard": shard,
        "current_position": position,
        "batch_tokens": count,
        "file": str(loader.files[shard]),
    }


def advance_loader_without_materializing(
    loader: Any, source_runtime: Any, batches: int
) -> dict[str, Any]:
    remaining = int(batches)
    if remaining < 0:
        raise ValueError("cannot rewind checkpoint loader state")
    start = loader.state_dict()
    count = loader.batch_size * loader.sequence_length
    advances = 0
    while remaining:
        available = (len(loader.tokens) - loader.current_position - 1) // count
        if available <= 0:
            loader.advance()
            advances += 1
            continue
        take = min(remaining, available)
        loader.current_position += take * count
        remaining -= take
        if loader.current_position + count + 1 > len(loader.tokens):
            loader.advance()
            advances += 1
    return {
        "requested_batches": int(batches),
        "start": start,
        "end": loader.state_dict(),
        "shard_advances": advances,
        "materialized_batches": 0,
    }


def frozen_batches(
    source_runtime: Any,
    pattern: str,
    batch_size: int,
    sequence_length: int,
    batches: int,
    token_offset: int,
) -> tuple[list[tuple[Tensor, Tensor]], dict[str, Any]]:
    loader = source_runtime.SequentialShardLoader(
        pattern, int(batch_size), int(sequence_length)
    )
    location = set_loader_global_offset(loader, source_runtime, token_offset)
    values = []
    hashes = []
    for index in range(int(batches)):
        x, y = loader.next_batch()
        values.append((x, y))
        hashes.append(
            {
                "batch": index,
                "x_sha256": M1.tensor_sha256(x),
                "y_sha256": M1.tensor_sha256(y),
                "shape": list(x.shape),
            }
        )
    return values, {
        "pattern": pattern,
        "batch_size": int(batch_size),
        "sequence_length": int(sequence_length),
        "batches": int(batches),
        "location": location,
        "hashes": hashes,
    }


def build_fresh_preconditioner(
    model: nn.Module,
    matrix_optimizer: torch.optim.Optimizer,
    batches: list[tuple[Tensor, Tensor]],
    source_runtime: Any,
) -> dict[str, Any]:
    is_newton = isinstance(matrix_optimizer, source_runtime.SharedInputNewtonMuon)
    if not is_newton:
        return {
            "required": False,
            "origin_preconditioner_transferred": False,
            "passed": True,
        }
    model.train()
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for x, _ in batches:
            with autocast:
                model(
                    x,
                    targets=None,
                    return_logits=False,
                    precond_flag=True,
                )
        matrix_optimizer._refresh_preconditioners()
    rows = []
    for spec in matrix_optimizer._groups:
        owner = spec["members"][0]
        state = matrix_optimizer.state[owner]
        inverse = state["precond_inv_apply"]
        rows.append(
            {
                "name": spec["name"],
                "kind": spec["kind"],
                "members": len(spec["members"]),
                "width": int(owner.shape[1]),
                "inverse_finite": bool(torch.isfinite(inverse).all().item()),
                "inverse_norm": float(inverse.norm().item()),
                "accum_zero_after_refresh": float(spec["accum"].abs().max().item())
                == 0.0,
                "count_zero_after_refresh": float(spec["count"].item()) == 0.0,
            }
        )
    return {
        "required": True,
        "origin_preconditioner_transferred": False,
        "build_batches": len(batches),
        "global_step_after_build": int(matrix_optimizer.global_step),
        "groups": rows,
        "passed": bool(rows)
        and int(matrix_optimizer.global_step) == 0
        and all(
            row["inverse_finite"]
            and row["accum_zero_after_refresh"]
            and row["count_zero_after_refresh"]
            for row in rows
        ),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: list[tuple[Tensor, Tensor]],
) -> float:
    model.eval()
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    total = torch.zeros((), device="cuda", dtype=torch.float32)
    for x, y in batches:
        with autocast:
            _, loss = model(
                x, y, return_logits=False, precond_flag=False
            )
        if loss is None:
            raise RuntimeError("evaluation loss is missing")
        total += loss.float()
    value = float((total / len(batches)).item())
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite held-out loss: {value}")
    return value


def model_parameters_finite(model: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter).all().item())
        for parameter in model.parameters()
    )


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MECH-08 requires CUDA")
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
    shutil.copyfile(args.contract, output / "rollout_contract.json")
    shutil.copyfile(
        args.prediction_reference, output / "mech07_prediction_reference.csv"
    )

    spec = contract_audit["checkpoint_spec"]
    algorithm_spec = contract_audit["algorithm_spec"]
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
    method_audit = method_identity_audit(
        checkpoint, spec["method"], schema["method_inferred"]
    )
    architecture = schema["architecture"]
    architecture_checks = {
        "step": int(schema["step"]) == int(spec["step"]),
        "method": method_audit["passed"],
        "n_layer": architecture["n_layer"] == contract["architecture"]["n_layer"],
        "n_embd": architecture["n_embd"] == contract["architecture"]["n_embd"],
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
    source_method = algorithm_spec["source_method"]
    source_config = M1.configure_source_runtime_globals(
        "llama1b", source_runtime, source_method
    )
    model = M1.build_model(
        "llama1b", source_runtime, architecture, source_method
    )
    parameter_audit = copy_checkpoint_parameters(model, model_state)
    model.cuda()
    restart = contract["restart_intervention"]
    optimizers, matrix_optimizer = make_candidate_optimizers(
        source_runtime, model, source_method, restart
    )
    backup_audit, momentum_audit = transfer_optimizer_state(
        checkpoint, optimizers, matrix_optimizer, model
    )
    for group in optimizers[0].param_groups:
        group["lr"] = float(restart["backup_lr"])
    for group in matrix_optimizer.param_groups:
        group["lr"] = float(restart["matrix_lr"])
    optimizer_audit = optimizer_hyperparameter_audit(
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
        seek_audit = advance_loader_without_materializing(
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
    build_batches, build_contract = frozen_batches(
        source_runtime,
        args.val_data_pattern,
        int(config["build_device_batch_size"]),
        int(config["build_sequence_length"]),
        int(config["build_batches"]),
        build_offset,
    )
    eval_batches, eval_contract = frozen_batches(
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

    preconditioner_audit = build_fresh_preconditioner(
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
    # Optimizer states have been copied onto the candidate parameters.
    del checkpoint
    gc.collect()

    evaluation_steps = {int(value) for value in config["evaluation_steps"]}
    if 0 not in evaluation_steps or int(config["rollout_steps"]) not in evaluation_steps:
        raise RuntimeError("evaluation schedule must include both endpoints")
    torch.cuda.reset_peak_memory_stats()
    evaluation_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    initial_loss = evaluate(model, eval_batches)
    evaluation_rows.append(
        {
            "checkpoint_cell": args.cell,
            "checkpoint_stage": spec["stage"],
            "checkpoint_method": spec["method"],
            "algorithm": args.algorithm,
            "source_method": source_method,
            "data_replica": args.data_replica,
            "optimizer_step": 0,
            "heldout_loss": initial_loss,
            "normalized_loss": 1.0,
            "loss_delta_from_step0": 0.0,
            "relative_loss_delta_from_step0": 0.0,
        }
    )

    compiled_model: nn.Module
    if args.no_compile:
        compiled_model = model
    else:
        compiled_model = torch.compile(model)
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    rollout_steps = int(config["rollout_steps"])
    for step in range(rollout_steps):
        if isinstance(matrix_optimizer, source_runtime.SharedInputNewtonMuon):
            matrix_optimizer.global_step = step
            precond_flag = matrix_optimizer.precond_flag_for_step(step)
        else:
            precond_flag = False
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
        completed = step + 1
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
                "algorithm": args.algorithm,
                "source_method": source_method,
                "data_replica": args.data_replica,
                "optimizer_step": completed,
                "train_loss_mean": train_loss,
                "preconditioner_refresh": precond_flag,
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
            heldout_loss = evaluate(model, eval_batches)
            evaluation_rows.append(
                {
                    "checkpoint_cell": args.cell,
                    "checkpoint_stage": spec["stage"],
                    "checkpoint_method": spec["method"],
                    "algorithm": args.algorithm,
                    "source_method": source_method,
                    "data_replica": args.data_replica,
                    "optimizer_step": completed,
                    "heldout_loss": heldout_loss,
                    "normalized_loss": heldout_loss / initial_loss,
                    "loss_delta_from_step0": heldout_loss - initial_loss,
                    "relative_loss_delta_from_step0": (heldout_loss - initial_loss)
                    / initial_loss,
                }
            )
        print(
            "MECH-08 "
            f"cell={args.cell} algorithm={args.algorithm} "
            f"replica={args.data_replica} step={completed}/{rollout_steps} "
            f"train_loss={train_loss:.6f}",
            flush=True,
        )

    expected_evaluations = len(evaluation_steps)
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
        "checkpoint_schema": schema["passed"] and all(architecture_checks.values()),
        "method_identity": method_audit["passed"],
        "source_runtime": source_config["passed"],
        "triton_provenance": triton_audit["passed"],
        "parameters_copied": parameter_audit["passed"],
        "backup_state_loaded": backup_audit["passed"],
        "historical_momentum_transferred": momentum_audit["passed"],
        "optimizer_hyperparameters": optimizer_audit["passed"],
        "fresh_preconditioner": preconditioner_audit["passed"],
        "heldout_windows": heldout_contract["passed"],
        "training_rows": len(training_rows) == rollout_steps,
        "evaluation_rows": len(evaluation_rows) == expected_evaluations,
        "evaluation_schedule": {row["optimizer_step"] for row in evaluation_rows}
        == evaluation_steps,
        "finite_training": M1.finite_numbers(training_rows),
        "finite_evaluation": M1.finite_numbers(evaluation_rows),
        "final_parameters_finite": model_parameters_finite(model),
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
        "training_stream_contract.json": training_stream_contract,
        "heldout_batch_contract.json": heldout_contract,
        "checkpoint_invariance.json": invariance,
        "runtime.json": M1.runtime_metadata(runtime_args),
        "checks.json": checks,
    }
    for name, value in artifacts.items():
        M1.atomic_json(output / name, value)
    write_csv(output / "training.csv", training_rows)
    write_csv(output / "evaluation.csv", evaluation_rows)
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
        "algorithm": args.algorithm,
        "source_method": source_method,
        "data_replica": args.data_replica,
        "optimizer_step_offset": optimizer_step_offset,
        "rollout_steps": rollout_steps,
        "training_rows": len(training_rows),
        "evaluation_rows": len(evaluation_rows),
        "real_optimizer_steps": True,
        "timing_usable_for_paper": False,
        "efficiency_benchmark_run": False,
        "passed": passed,
        "artifacts": sorted(
            [
                *artifacts,
                "rollout_contract.json",
                "mech07_prediction_reference.csv",
                "training.csv",
                "evaluation.csv",
                "mech08_manifest.json",
                "status.json",
            ]
        ),
    }
    M1.atomic_json(output / "mech08_manifest.json", manifest)
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
    print(f"MECH-08 manifest: {output / 'mech08_manifest.json'}", flush=True)
    print(f"MECH-08 artifacts: {output}", flush=True)


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
