#!/usr/bin/env python3
"""Read-only CUDA worker for the LLaMA block-partition invariance audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn


SCRIPT_VERSION = "2026-07-29.2"
CONTRACT_VERSION = "2026-07-29.2"
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
M1_PATH = SCRIPTS / "27_mech01_unified_k_diagnostics" / "mech01_worker.py"
M2_PATH = SCRIPTS / "30_mech02_k_geometry" / "mech02_worker.py"
M3_PATH = SCRIPTS / "31_mech03_crossfit_shadow" / "mech03_worker.py"
M6_PATH = SCRIPTS / "33_mech06_llama1b_confirmation" / "mech06_worker.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M1 = load_module("llama_block_audit_m1", M1_PATH)
M2 = load_module("llama_block_audit_m2", M2_PATH)
M3 = load_module("llama_block_audit_m3", M3_PATH)
M6 = load_module("llama_block_audit_m6", M6_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--analysis-tier", required=True, choices=("smoke", "formal"))
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--checkpoint-label", required=True, choices=("early", "late"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-hash-certificate", required=True, type=Path)
    parser.add_argument("--source-script", required=True, type=Path)
    parser.add_argument("--profile-script", required=True, type=Path)
    parser.add_argument("--triton-kernels", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--data-pattern", required=True)
    parser.add_argument("--layers", nargs="+", required=True, type=int)
    parser.add_argument("--repeat-offsets", nargs="+", required=True, type=int)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--batches-per-split", required=True, type=int)
    parser.add_argument("--device-batch-size", required=True, type=int)
    parser.add_argument("--sequence-length", required=True, type=int)
    parser.add_argument("--max-activation-rows", required=True, type=int)
    parser.add_argument("--global-permutation-seeds", nargs="+", required=True, type=int)
    parser.add_argument("--within-block-seed", required=True, type=int)
    parser.add_argument("--block-count", required=True, type=int)
    parser.add_argument("--ridge-mult", required=True, type=float)
    parser.add_argument("--ridge-eps", required=True, type=float)
    parser.add_argument("--momentum", required=True, type=float)
    parser.add_argument("--ns-steps", required=True, type=int)
    parser.add_argument("--step-multipliers", nargs="+", required=True, type=float)
    parser.add_argument("--dense-control-permutation-count", required=True, type=int)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--execution-domain", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def relative_drift(left: Tensor, right: Tensor) -> float:
    left64 = left.to(dtype=torch.float64)
    right64 = right.to(device=left64.device, dtype=torch.float64)
    return float(
        torch.linalg.vector_norm(left64 - right64)
        / torch.linalg.vector_norm(right64).clamp_min(1e-30)
    )


def cosine(left: Tensor, right: Tensor) -> float:
    left64 = left.to(dtype=torch.float64).reshape(-1)
    right64 = right.to(device=left64.device, dtype=torch.float64).reshape(-1)
    return float(
        torch.dot(left64, right64)
        / (
            torch.linalg.vector_norm(left64)
            * torch.linalg.vector_norm(right64)
        ).clamp_min(1e-30)
    )


def map_new_to_old(value: Tensor, permutation: Tensor) -> Tensor:
    """Map columns from new coordinates back to their old coordinate indices."""
    result = torch.empty_like(value)
    result[:, permutation.to(value.device)] = value
    return result


def global_permutation(width: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(width, generator=generator)


def within_block_permutation(width: int, blocks: int, seed: int) -> Tensor:
    if width % blocks:
        raise ValueError(f"width {width} is not divisible by blocks {blocks}")
    block_size = width // blocks
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pieces = []
    for block in range(blocks):
        start = block * block_size
        pieces.append(start + torch.randperm(block_size, generator=generator))
    return torch.cat(pieces)


def partition_overlap(permutation: Tensor, blocks: int) -> dict[str, Any]:
    width = int(permutation.numel())
    if width % blocks:
        raise ValueError("balanced partition requires divisibility")
    size = width // blocks
    original_labels = torch.arange(width, dtype=torch.int64) // size
    new_labels_by_new_coordinate = torch.arange(width, dtype=torch.int64) // size
    old_labels_in_new_order = original_labels[permutation.cpu()]
    confusion = torch.zeros((blocks, blocks), dtype=torch.int64)
    for new_block in range(blocks):
        mask = new_labels_by_new_coordinate == new_block
        counts = torch.bincount(
            old_labels_in_new_order[mask], minlength=blocks
        )
        confusion[:, new_block] = counts
    choose2 = lambda value: value * (value - 1) // 2
    intersection = sum(
        choose2(int(confusion[left, right]))
        for left in range(blocks)
        for right in range(blocks)
    )
    same_pairs = blocks * choose2(size)
    union = 2 * same_pairs - intersection
    return {
        "confusion": confusion.tolist(),
        "same_block_pair_intersection": intersection,
        "same_block_pair_jaccard": intersection / max(union, 1),
        "coordinates_remaining_in_original_block": int(confusion.diag().sum()),
        "coordinate_retention_fraction": float(confusion.diag().sum() / width),
    }


def off_block_energy(activation: Tensor, permutation: Tensor, blocks: int) -> dict[str, float]:
    """Compute block/off-block covariance energy through dual Gram matrices."""
    x = activation[:, permutation.to(activation.device)].to(dtype=torch.float64)
    rows, width = x.shape
    if width % blocks:
        raise ValueError("balanced block energy requires divisible width")
    size = width // blocks
    full_dual = x @ x.T
    full_energy = float(full_dual.square().sum() / (rows * rows))
    block_energy = 0.0
    for block in range(blocks):
        xb = x[:, block * size : (block + 1) * size]
        dual = xb @ xb.T
        block_energy += float(dual.square().sum() / (rows * rows))
    off = max(0.0, full_energy - block_energy)
    return {
        "covariance_frobenius_sq": full_energy,
        "block_diagonal_frobenius_sq": block_energy,
        "off_block_frobenius_sq": off,
        "off_block_energy_fraction": off / max(full_energy, 1e-30),
    }


def block_inverse_apply(
    gradient: Tensor,
    activation: Tensor,
    ridge: Tensor | float,
    blocks: int,
) -> tuple[Tensor, float]:
    if activation.size(1) % blocks:
        raise ValueError("block inverse requires divisible activation width")
    size = activation.size(1) // blocks
    outputs = []
    residuals = []
    for block in range(blocks):
        start, stop = block * size, (block + 1) * size
        value, residual = M6.woodbury_apply(
            gradient[:, start:stop],
            activation[:, start:stop],
            torch.as_tensor(ridge, device=activation.device),
        )
        outputs.append(value)
        residuals.append(residual)
    return torch.cat(outputs, dim=1), max(residuals)


def update_from_preconditioned(
    preconditioned: Tensor,
    old_momentum: Tensor,
    production_ns: Any,
    momentum: float,
    ns_steps: int,
) -> Tensor:
    _, lookahead = M1.momentum_lookahead(
        preconditioned.float(), old_momentum.float(), "llama1b", momentum
    )
    return production_ns(lookahead.float(), steps=ns_steps).float()


def build_permutations(
    width: int,
    blocks: int,
    within_seed: int,
    global_seeds: list[int],
) -> list[dict[str, Any]]:
    identity = torch.arange(width, dtype=torch.int64)
    rows = [
        {
            "name": "identity",
            "kind": "identity",
            "seed": -1,
            "permutation": identity,
        },
        {
            "name": f"within_block_seed{within_seed}",
            "kind": "within_block_control",
            "seed": within_seed,
            "permutation": within_block_permutation(width, blocks, within_seed),
        },
    ]
    rows.extend(
        {
            "name": f"global_seed{seed}",
            "kind": "global_balanced_partition",
            "seed": seed,
            "permutation": global_permutation(width, seed),
        }
        for seed in global_seeds
    )
    for row in rows:
        permutation = row["permutation"]
        row["sha256"] = M1.tensor_sha256(permutation)
        row["overlap"] = partition_overlap(permutation, blocks)
    return rows


def validate_contract(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_json(args.contract.resolve())
    tier = contract[args.analysis_tier]
    common = contract["common"]
    architecture = contract["architecture"]
    checks = {
        "contract_version": contract.get("contract_version") == CONTRACT_VERSION,
        "family": contract.get("family") == "llama1b",
        "source_method": contract.get("source_method") == "down_none",
        "block4_not_primary": (
            contract["scientific_scope"]["block4_role"]
            == "diagnostic arbitrary partition approximation; never a primary baseline or original Newton-Muon label"
        ),
        "new_training_false": contract["scientific_scope"]["new_training"] is False,
        "hvp_false": contract["scientific_scope"]["hvp_authorized"] is False,
        "layers": args.layers == tier["layers"],
        "repeats": args.repeats == tier["repeats"],
        "batches_per_split": args.batches_per_split == tier["batches_per_split"],
        "max_activation_rows": args.max_activation_rows == tier["max_activation_rows"],
        "global_seeds": args.global_permutation_seeds
        == tier["global_permutation_seeds"],
        "step_multipliers": args.step_multipliers == tier["step_multipliers"],
        "block_count": args.block_count == architecture["block_count"],
        "within_block_seed": args.within_block_seed == common["within_block_seed"],
        "dense_control_count": args.dense_control_permutation_count
        == common["dense_control_permutation_count"],
        "device_batch_size": args.device_batch_size == common["device_batch_size"],
        "sequence_length": args.sequence_length == common["sequence_length"],
        "ridge_mult": args.ridge_mult == common["ridge_mult"],
        "ridge_eps": args.ridge_eps == common["ridge_eps"],
        "momentum": args.momentum == common["momentum"],
        "ns_steps": args.ns_steps == common["ns_steps"],
        "source_sha256": M1.sha256_file(args.source_script.resolve())
        == contract["source_constraints"]["base_source_sha256"],
        "triton_sha256": M1.sha256_file(args.triton_kernels.resolve())
        == contract["source_constraints"]["triton_sha256"],
    }
    return contract, {
        "contract": str(args.contract.resolve()),
        "contract_sha256": M1.sha256_file(args.contract.resolve()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_hash_certificate(args: argparse.Namespace, contract: dict[str, Any]) -> dict[str, Any]:
    certificate = read_json(args.checkpoint_hash_certificate.resolve())
    checkpoint = args.checkpoint.resolve()
    stat = checkpoint.stat()
    expected = contract["checkpoints"][args.checkpoint_label]["sha256"]
    checks = {
        "certificate_passed": certificate.get("passed") is True,
        "label": certificate.get("label") == args.checkpoint_label,
        "path": certificate.get("path") == str(checkpoint),
        "sha256": certificate.get("sha256") == expected,
        "size": int(certificate.get("bytes", -1)) == stat.st_size,
        "mtime": int(certificate.get("mtime_ns", -1)) == stat.st_mtime_ns,
    }
    return {
        "certificate": str(args.checkpoint_hash_certificate.resolve()),
        "payload": certificate,
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_smoke_gate(
    args: argparse.Namespace, contract_sha256: str, checkpoint_sha256: str
) -> dict[str, Any]:
    if args.analysis_tier == "smoke":
        return {"required": False, "passed": True}
    if args.smoke_manifest is None:
        raise RuntimeError("formal tier requires --smoke-manifest")
    payload = read_json(args.smoke_manifest.resolve())
    checks = {
        "passed": payload.get("passed") is True,
        "tier": payload.get("analysis_tier") == "smoke",
        "checkpoint_label": payload.get("checkpoint_label") == args.checkpoint_label,
        "checkpoint_sha256": payload.get("checkpoint_sha256") == checkpoint_sha256,
        "contract_sha256": payload.get("contract_sha256") == contract_sha256,
        "script_version": payload.get("script_version") == SCRIPT_VERSION,
        "worker_sha256": payload.get("worker_sha256")
        == M1.sha256_file(Path(__file__).resolve()),
    }
    return {
        "required": True,
        "manifest": str(args.smoke_manifest.resolve()),
        "manifest_sha256": M1.sha256_file(args.smoke_manifest.resolve()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def crossfit_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_pattern=args.data_pattern,
        device_batch_size=args.device_batch_size,
        sequence_length=args.sequence_length,
        repeat_offsets=args.repeat_offsets,
        repeats=args.repeats,
        batches_per_split=args.batches_per_split,
    )


def candidate_updates_for_layer(
    activation_cpu: Tensor,
    gradient_cpu: Tensor,
    momentum_cpu: Tensor,
    weight: Tensor,
    permutations: list[dict[str, Any]],
    args: argparse.Namespace,
    production_ns: Any,
    layer: int,
    repeat: int,
    direction: str,
    build_split: str,
) -> tuple[dict[str, Tensor], list[dict[str, Any]], list[dict[str, Any]]]:
    x = activation_cpu.cuda().float()
    gradient = gradient_cpu.cuda().float()
    old_momentum = momentum_cpu.cuda().float()
    diagonal = x.square().mean(dim=0)
    ridge = diagonal.mean() * args.ridge_mult + args.ridge_eps

    base_preconditioned: dict[str, Tensor] = {
        "none": gradient,
        "diag": gradient / (diagonal + ridge).unsqueeze(0),
    }
    dense, dense_residual = M6.woodbury_apply(gradient, x, ridge)
    block, block_residual = block_inverse_apply(
        gradient, x, ridge, args.block_count
    )
    base_preconditioned["dense_full"] = dense
    base_preconditioned["block4_identity"] = block
    base_updates = {
        candidate: update_from_preconditioned(
            value, old_momentum, production_ns, args.momentum, args.ns_steps
        )
        for candidate, value in base_preconditioned.items()
    }
    updates = {
        candidate: value.detach().cpu().contiguous()
        for candidate, value in base_updates.items()
    }

    direction_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    projection_rows = min(32, x.size(0))
    weight_sample = weight.detach()[: min(64, weight.size(0))].double()

    for permutation_index, spec in enumerate(permutations):
        permutation_cpu = spec["permutation"]
        permutation = permutation_cpu.cuda()
        xp = x[:, permutation]
        gp = gradient[:, permutation]
        mp = old_momentum[:, permutation]
        energy = off_block_energy(x, permutation, args.block_count)
        base_projection = x[:projection_rows].double() @ weight_sample.T
        perm_projection = (
            xp[:projection_rows].double() @ weight_sample[:, permutation].T
        )
        projection_relative = relative_drift(perm_projection, base_projection)
        partition_rows.append(
            {
                "checkpoint_label": args.checkpoint_label,
                "repeat": repeat,
                "direction": direction,
                "build_split": build_split,
                "layer": layer,
                "partition": spec["name"],
                "partition_kind": spec["kind"],
                "permutation_seed": spec["seed"],
                "permutation_sha256": spec["sha256"],
                "same_block_pair_jaccard": spec["overlap"][
                    "same_block_pair_jaccard"
                ],
                "coordinate_retention_fraction": spec["overlap"][
                    "coordinate_retention_fraction"
                ],
                "projection_equivalence_relative": projection_relative,
                **energy,
            }
        )
        if spec["kind"] == "identity":
            continue

        perm_diagonal = xp.square().mean(dim=0)
        perm_preconditioned: dict[str, tuple[Tensor, float]] = {
            "none": (gp, 0.0),
            "diag": (
                gp / (perm_diagonal + ridge).unsqueeze(0),
                0.0,
            ),
        }
        if (
            spec["kind"] == "within_block_control"
            or (
                spec["kind"] == "global_balanced_partition"
                and permutation_index
                <= 1 + args.dense_control_permutation_count
            )
        ):
            dense_perm, residual = M6.woodbury_apply(gp, xp, ridge)
            perm_preconditioned["dense_full"] = (dense_perm, residual)
        block_perm, residual = block_inverse_apply(
            gp, xp, ridge, args.block_count
        )
        perm_preconditioned["block4"] = (block_perm, residual)

        for candidate, (preconditioned, residual) in perm_preconditioned.items():
            preconditioned_old = map_new_to_old(preconditioned, permutation)
            update_new = update_from_preconditioned(
                preconditioned,
                mp,
                production_ns,
                args.momentum,
                args.ns_steps,
            )
            update_old = map_new_to_old(update_new, permutation)
            reference_name = (
                "block4_identity" if candidate == "block4" else candidate
            )
            reference = base_updates[reference_name]
            preconditioned_reference = base_preconditioned[reference_name]
            candidate_name = (
                f"block4_{spec['name']}" if candidate == "block4" else reference_name
            )
            if candidate == "block4":
                updates[candidate_name] = update_old.detach().cpu().contiguous()
            direction_rows.append(
                {
                    "checkpoint_label": args.checkpoint_label,
                    "repeat": repeat,
                    "direction": direction,
                    "build_split": build_split,
                    "layer": layer,
                    "partition": spec["name"],
                    "partition_kind": spec["kind"],
                    "permutation_seed": spec["seed"],
                    "permutation_sha256": spec["sha256"],
                    "candidate": candidate,
                    "reference_candidate": reference_name,
                    "activation_rows": int(x.size(0)),
                    "activation_width": int(x.size(1)),
                    "ridge": float(ridge),
                    "inverse_residual_relative": residual,
                    "preconditioned_norm": float(
                        torch.linalg.vector_norm(preconditioned)
                    ),
                    "preconditioned_relative_drift": relative_drift(
                        preconditioned_old, preconditioned_reference
                    ),
                    "preconditioned_cosine": cosine(
                        preconditioned_old, preconditioned_reference
                    ),
                    "update_norm": float(torch.linalg.vector_norm(update_old)),
                    "update_relative_drift": relative_drift(update_old, reference),
                    "update_cosine": cosine(update_old, reference),
                    "update_finite": bool(torch.isfinite(update_old).all()),
                }
            )
        del xp, gp, mp, perm_diagonal, block_perm
        torch.cuda.empty_cache()

    baseline_rows = [
        {
            "checkpoint_label": args.checkpoint_label,
            "repeat": repeat,
            "direction": direction,
            "build_split": build_split,
            "layer": layer,
            "candidate": candidate,
            "preconditioned_norm": float(torch.linalg.vector_norm(value)),
            "update_norm": float(torch.linalg.vector_norm(base_updates[candidate])),
            "inverse_residual_relative": (
                dense_residual
                if candidate == "dense_full"
                else block_residual
                if candidate == "block4_identity"
                else 0.0
            ),
            "update_sha256": M1.tensor_sha256(updates[candidate]),
            "update_finite": bool(torch.isfinite(base_updates[candidate]).all()),
        }
        for candidate, value in base_preconditioned.items()
    ]
    for row in direction_rows:
        row["baseline_dense_inverse_residual"] = dense_residual
        row["baseline_block_inverse_residual"] = block_residual
    del x, gradient, old_momentum, dense, block, base_preconditioned, base_updates
    torch.cuda.empty_cache()
    return updates, baseline_rows + direction_rows, partition_rows


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("LLaMA block-partition audit requires CUDA")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    output = args.output_dir.resolve()
    if not output.is_dir():
        raise RuntimeError(f"controller must create output directory: {output}")
    M1.atomic_json(
        output / "status.json",
        {
            "status": "running",
            "script_version": SCRIPT_VERSION,
            "analysis_tier": args.analysis_tier,
        },
    )

    contract, contract_audit = validate_contract(args)
    if not contract_audit["passed"]:
        raise RuntimeError(f"contract audit failed: {contract_audit}")
    hash_audit = validate_hash_certificate(args, contract)
    if not hash_audit["passed"]:
        raise RuntimeError(f"checkpoint hash audit failed: {hash_audit}")
    checkpoint_sha = hash_audit["payload"]["sha256"]
    smoke_gate = validate_smoke_gate(
        args, contract_audit["contract_sha256"], checkpoint_sha
    )
    if not smoke_gate["passed"]:
        raise RuntimeError(f"smoke gate failed: {smoke_gate}")
    shutil.copyfile(args.contract.resolve(), output / "audit_contract.json")

    checkpoint_path = args.checkpoint.resolve()
    source_path = args.source_script.resolve()
    triton_path = args.triton_kernels.resolve()
    before_stat = checkpoint_path.stat()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state, schema = M1.checkpoint_schema(
        checkpoint_path,
        checkpoint,
        "llama1b",
        source_path,
        checkpoint_sha,
        False,
    )
    M1.attach_profile_provenance(schema, args.profile_script)
    expected_architecture = contract["architecture"]
    architecture_checks = {
        "step": int(schema["step"])
        == int(contract["checkpoints"][args.checkpoint_label]["step"]),
        "method": schema["method_inferred"] == contract["source_method"],
        "n_layer": schema["architecture"]["n_layer"]
        == expected_architecture["n_layer"],
        "n_embd": schema["architecture"]["n_embd"]
        == expected_architecture["n_embd"],
        "intermediate_size": schema["architecture"]["target_input_width"]
        == expected_architecture["intermediate_size"],
        "balanced_four_way_split": (
            schema["architecture"]["target_input_width"] % args.block_count == 0
        ),
        "block_size": (
            schema["architecture"]["target_input_width"] // args.block_count
            == expected_architecture["block_size"]
        ),
        "no_semantic_four_subspaces": expected_architecture[
            "four_semantic_subspaces"
        ]
        is False,
    }
    if not schema["passed"] or not all(architecture_checks.values()):
        raise RuntimeError(
            f"checkpoint schema/architecture failed: {architecture_checks}"
        )
    route_audit = M1.route_audit("llama1b", source_path, schema["architecture"])
    if not route_audit["passed"]:
        raise RuntimeError(f"SwiGLU route audit failed: {route_audit}")

    source_runtime, production_ns, triton_audit = M1.load_source_runtime(
        "llama1b", source_path, triton_path
    )
    source_config = M1.configure_source_runtime_globals(
        "llama1b", source_runtime, "down_none"
    )
    model = M1.build_model(
        "llama1b", source_runtime, schema["architecture"], "down_none"
    )
    incompatible = model.load_state_dict(model_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"model load mismatch: {incompatible}")
    modules, weights, names = M1.target_modules_and_weights(
        model, "llama1b", args.layers
    )
    momenta, momentum_audit = M1.extract_target_momenta(
        checkpoint, model, "llama1b", names
    )
    optimizer_hyperparameters = M3.matrix_optimizer_hyperparameters(
        checkpoint, weights
    )
    model_before = M1.model_state_signature(model, names.values())
    aux_before = M1.checkpoint_aux_signature(checkpoint)
    model.cuda()

    batches, batch_contract = M3.read_crossfit_batches(crossfit_args(args))
    width = int(schema["architecture"]["target_input_width"])
    permutations = build_permutations(
        width,
        args.block_count,
        args.within_block_seed,
        args.global_permutation_seeds,
    )
    permutation_audit = {
        "schema_version": 1,
        "width": width,
        "block_count": args.block_count,
        "block_size": width // args.block_count,
        "generator": "torch.randperm with an explicit CPU torch.Generator seed",
        "torch_version": torch.__version__,
        "permutation_convention": contract["common"]["permutation_convention"],
        "function_preserving_parameter_map": {
            "gate_proj.weight": "permute output rows",
            "up_proj.weight": "permute output rows",
            "down_proj.weight": "permute input columns by the same permutation",
            "numerical_test": (
                "captured down input and actual down_proj weight preserve the "
                "sampled down-projection output"
            ),
        },
        "partitions": [
            {
                key: value
                for key, value in row.items()
                if key != "permutation"
            }
            for row in permutations
        ],
    }
    permutation_index_rows = [
        {
            "partition": row["name"],
            "partition_kind": row["kind"],
            "seed": row["seed"],
            "new_coordinate": new_coordinate,
            "old_coordinate": int(old_coordinate),
        }
        for row in permutations
        for new_coordinate, old_coordinate in enumerate(
            row["permutation"].tolist()
        )
    ]
    permutation_hashes = [row["sha256"] for row in permutations]
    permutation_integrity = {
        "unique_hashes": len(set(permutation_hashes)) == len(permutation_hashes),
        "all_bijections": all(
            torch.equal(torch.sort(row["permutation"]).values, torch.arange(width))
            for row in permutations
        ),
        "identity_overlap_one": permutations[0]["overlap"][
            "same_block_pair_jaccard"
        ]
        == 1.0,
        "within_block_overlap_one": permutations[1]["overlap"][
            "same_block_pair_jaccard"
        ]
        == 1.0,
        "global_partitions_cross_blocks": all(
            row["overlap"]["same_block_pair_jaccard"] < 1.0
            for row in permutations
            if row["kind"] == "global_balanced_partition"
        ),
    }
    permutation_audit["integrity"] = permutation_integrity

    update_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    candidate_names = [
        "none",
        "diag",
        "dense_full",
        "block4_identity",
        *[
            f"block4_{row['name']}"
            for row in permutations
            if row["kind"] != "identity"
        ],
    ]
    for repeat in range(args.repeats):
        builds = {
            split: M3.collect_build_split(
                model,
                modules,
                weights,
                {},
                batches[repeat][split],
                args.max_activation_rows,
            )
            for split in M3.SPLITS
        }
        for direction, build_split, eval_split in M3.DIRECTIONS:
            build = builds[build_split]
            layer_updates: dict[int, dict[str, Tensor]] = {}
            for layer in args.layers:
                updates, rows, partitions = candidate_updates_for_layer(
                    build["activations"][layer],
                    build["gradients"][layer],
                    momenta[layer],
                    weights[layer],
                    permutations,
                    args,
                    production_ns,
                    layer,
                    repeat,
                    direction,
                    build_split,
                )
                layer_updates[layer] = updates
                update_rows.extend(rows)
                partition_rows.extend(partitions)
            losses, summaries = M3.line_search(
                model,
                weights,
                layer_updates,
                batches[repeat][eval_split],
                candidate_names,
                args.step_multipliers,
                optimizer_hyperparameters,
                "llama1b",
                repeat,
                direction,
                build_split,
                eval_split,
                args.batches_per_split,
            )
            for row in losses:
                row["checkpoint_label"] = args.checkpoint_label
            for row in summaries:
                row["checkpoint_label"] = args.checkpoint_label
            loss_rows.extend(losses)
            summary_rows.extend(summaries)
            del layer_updates, losses, summaries
            torch.cuda.empty_cache()

    model_after = M1.model_state_signature(model, names.values())
    aux_after = M1.checkpoint_aux_signature(checkpoint)
    after_stat = checkpoint_path.stat()
    state_invariance = {
        "model_content_unchanged": M3.model_content_unchanged(
            model_before, model_after
        ),
        "optimizer_loader_unchanged": aux_before == aux_after,
        "checkpoint_file_unchanged": (
            before_stat.st_size == after_stat.st_size
            and before_stat.st_mtime_ns == after_stat.st_mtime_ns
        ),
        "model_before": model_before,
        "model_after": model_after,
        "aux_before": aux_before,
        "aux_after": aux_after,
        "checkpoint_stat_before": {
            "size": before_stat.st_size,
            "mtime_ns": before_stat.st_mtime_ns,
        },
        "checkpoint_stat_after": {
            "size": after_stat.st_size,
            "mtime_ns": after_stat.st_mtime_ns,
        },
    }

    thresholds = contract["integrity_thresholds"]
    global_rows = [
        row
        for row in update_rows
        if row.get("partition_kind") == "global_balanced_partition"
    ]
    within_rows = [
        row
        for row in update_rows
        if row.get("partition_kind") == "within_block_control"
    ]
    equivariant_candidates = {"none", "diag", "dense_full"}
    expected_directions = args.repeats * len(M3.DIRECTIONS)
    expected_partitions = expected_directions * len(args.layers) * len(permutations)
    expected_summaries = (
        expected_directions
        * (len(args.layers) + 1)
        * len(candidate_names)
    )
    checks = {
        "contract": contract_audit["passed"],
        "checkpoint_hash": hash_audit["passed"],
        "smoke_gate": smoke_gate["passed"],
        "checkpoint_schema": schema["passed"] and all(architecture_checks.values()),
        "swiglu_route": route_audit["passed"],
        "source_runtime": source_config["passed"],
        "triton": triton_audit["passed"],
        "historical_momentum": momentum_audit["all_present"],
        "optimizer_hyperparameters": optimizer_hyperparameters["passed"],
        "batch_windows_disjoint": batch_contract["all_windows_disjoint"],
        "partition_rows": len(partition_rows) == expected_partitions,
        "permutations": all(permutation_integrity.values()),
        "global_rows_present": bool(global_rows),
        "summary_rows": len(summary_rows) == expected_summaries,
        "loss_rows": len(loss_rows)
        == expected_summaries * len(args.step_multipliers),
        "projection_equivalence": all(
            row["projection_equivalence_relative"]
            <= thresholds["projection_equivalence_relative"]
            for row in partition_rows
        ),
        "preconditioner_coordinate_controls": all(
            row["preconditioned_relative_drift"]
            <= thresholds["preconditioner_equivariance_relative"]
            for row in update_rows
            if row.get("candidate") in equivariant_candidates
            and row.get("partition_kind") is not None
        ),
        "preconditioner_within_block_control": all(
            row["candidate"] != "block4"
            or row["preconditioned_relative_drift"]
            <= thresholds["preconditioner_equivariance_relative"]
            for row in within_rows
        ),
        "production_ns_coordinate_controls": all(
            row["update_relative_drift"]
            <= thresholds["production_ns_equivariance_relative"]
            for row in update_rows
            if row.get("candidate") in equivariant_candidates
            and row.get("partition_kind") is not None
        ),
        "production_ns_within_block_control": all(
            row["candidate"] != "block4"
            or row["update_relative_drift"]
            <= thresholds["production_ns_equivariance_relative"]
            for row in within_rows
        ),
        "inverse_residuals": all(
            row.get("inverse_residual_relative", 0.0)
            <= thresholds["inverse_residual_relative"]
            for row in update_rows
        ),
        "finite": M1.finite_numbers(update_rows)
        and M1.finite_numbers(partition_rows)
        and M1.finite_numbers(loss_rows)
        and M1.finite_numbers(summary_rows)
        and all(row.get("update_finite", True) for row in update_rows),
        "model_content_unchanged": state_invariance["model_content_unchanged"],
        "optimizer_loader_unchanged": state_invariance[
            "optimizer_loader_unchanged"
        ],
        "checkpoint_file_unchanged": state_invariance[
            "checkpoint_file_unchanged"
        ],
        "no_training": True,
        "hvp_not_run": True,
        "global_effect_not_gated": True,
    }
    passed = all(checks.values())
    artifacts = {
        "contract_audit.json": contract_audit,
        "checkpoint_hash_audit.json": hash_audit,
        "smoke_gate.json": smoke_gate,
        "checkpoint_schema.json": schema,
        "architecture_checks.json": architecture_checks,
        "swiglu_route_audit.json": route_audit,
        "source_runtime_config.json": source_config,
        "triton_audit.json": triton_audit,
        "momentum_audit.json": momentum_audit,
        "optimizer_hyperparameters.json": optimizer_hyperparameters,
        "batch_contract.json": batch_contract,
        "permutation_audit.json": permutation_audit,
        "state_invariance.json": state_invariance,
        "runtime.json": M1.runtime_metadata(args),
        "checks.json": checks,
    }
    for name, payload in artifacts.items():
        M1.atomic_json(output / name, payload)
    M2.write_csv(output / "partition_geometry.csv", partition_rows)
    M2.write_csv(output / "permutation_indices.csv", permutation_index_rows)
    M2.write_csv(output / "equivariance_updates.csv", update_rows)
    M2.write_csv(output / "shadow_losses.csv", loss_rows)
    M2.write_csv(output / "line_search_summary.csv", summary_rows)
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "worker_sha256": M1.sha256_file(Path(__file__).resolve()),
        "contract_version": CONTRACT_VERSION,
        "analysis_tier": args.analysis_tier,
        "passed": passed,
        "family": "llama1b",
        "source_method": "down_none",
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_step": int(schema["step"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "contract_sha256": contract_audit["contract_sha256"],
        "layers": args.layers,
        "block_count": args.block_count,
        "block_size": width // args.block_count,
        "partition_count": len(permutations),
        "global_permutation_count": len(args.global_permutation_seeds),
        "candidate_names": candidate_names,
        "partition_rows": len(partition_rows),
        "update_rows": len(update_rows),
        "shadow_loss_rows": len(loss_rows),
        "line_search_summary_rows": len(summary_rows),
        "scientific_result_used_for_pass": False,
        "new_training": False,
        "hvp_run": False,
        "artifacts": sorted(path.name for path in output.iterdir()),
    }
    M1.atomic_json(output / "llama_block_audit_manifest.json", manifest)
    M1.atomic_json(
        output / "status.json",
        {
            "status": "passed" if passed else "failed",
            "script_version": SCRIPT_VERSION,
            "analysis_tier": args.analysis_tier,
        },
    )
    if not passed:
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    try:
        run_worker(args)
    except BaseException as exc:
        try:
            index = sys.argv.index("--output-dir") + 1
            output = Path(sys.argv[index]).resolve()
            if output.is_dir():
                M1.atomic_json(
                    output / "status.json",
                    {
                        "status": "failed",
                        "script_version": SCRIPT_VERSION,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    main()
