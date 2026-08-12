#!/usr/bin/env python3
"""CUDA worker for read-only MECH-06 LLaMA-1B diagnostics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
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


SCRIPT_VERSION = "2026-07-27.4"
CONTRACT_VERSION = "2026-07-27.1"
HERE = Path(__file__).resolve().parent
M1_PATH = HERE.parent / "27_mech01_unified_k_diagnostics" / "mech01_worker.py"
M2_PATH = HERE.parent / "30_mech02_k_geometry" / "mech02_worker.py"
M3_PATH = HERE.parent / "31_mech03_crossfit_shadow" / "mech03_worker.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M1 = load_module("mech06_m1", M1_PATH)
M2 = load_module("mech06_m2", M2_PATH)
M3 = load_module("mech06_m3", M3_PATH)


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
    parser.add_argument("--mech01-reference-smoke-dir", required=True, type=Path)
    parser.add_argument("--confirmation-contract", required=True, type=Path)
    parser.add_argument("--mech05-contract", required=True, type=Path)
    parser.add_argument("--data-pattern", required=True)
    parser.add_argument("--geometry-layers", nargs="+", required=True, type=int)
    parser.add_argument("--shadow-layers", nargs="+", required=True, type=int)
    parser.add_argument("--geometry-offsets", nargs="+", required=True, type=int)
    parser.add_argument("--shadow-offsets", nargs="+", required=True, type=int)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--geometry-batches-per-repeat", required=True, type=int)
    parser.add_argument("--shadow-batches-per-split", required=True, type=int)
    parser.add_argument("--device-batch-size", required=True, type=int)
    parser.add_argument("--sequence-length", required=True, type=int)
    parser.add_argument("--max-geometry-rows", required=True, type=int)
    parser.add_argument("--max-shadow-rows", required=True, type=int)
    parser.add_argument("--ridge-mult", required=True, type=float)
    parser.add_argument("--ridge-eps", required=True, type=float)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--momentum", required=True, type=float)
    parser.add_argument("--ns-steps", required=True, type=int)
    parser.add_argument("--step-multipliers", nargs="+", required=True, type=float)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--execution-domain", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def validate_contracts(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    confirmation = read_json(args.confirmation_contract.resolve())
    mech05 = read_json(args.mech05_contract.resolve())
    checkpoint_spec = confirmation["checkpoints"][args.checkpoint_label]
    source_constraints = confirmation["source_constraints"]
    checks = {
        "contract_version": confirmation.get("contract_version") == CONTRACT_VERSION,
        "family": confirmation.get("family") == "llama1b",
        "method": confirmation.get("method") == "down_none",
        "mech05_sha256": (
            M1.sha256_file(args.mech05_contract.resolve())
            == confirmation["mech05_contract"]["sha256"]
        ),
        "checkpoint_certificate_sha": (
            read_json(args.checkpoint_hash_certificate.resolve()).get("sha256")
            == checkpoint_spec["sha256"]
        ),
        "source_sha256": (
            M1.sha256_file(args.source_script.resolve())
            == source_constraints["base_source_sha256"]
        ),
        "triton_sha256": (
            M1.sha256_file(args.triton_kernels.resolve())
            == source_constraints["triton_sha256"]
        ),
        "hvp_disabled": confirmation["interpretation"]["hvp_authorized"] is False,
        "existing_training_excluded": (
            confirmation["interpretation"]["existing_llama1b_training_rankings"]
            == "retrospective and excluded from prediction generation"
        ),
    }
    audit = {
        "confirmation_contract_sha256": M1.sha256_file(
            args.confirmation_contract.resolve()
        ),
        "mech05_contract_sha256": M1.sha256_file(args.mech05_contract.resolve()),
        "checkpoint_spec": checkpoint_spec,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return confirmation, audit


def validate_hash_certificate(args: argparse.Namespace) -> dict[str, Any]:
    path = args.checkpoint_hash_certificate.resolve()
    payload = read_json(path)
    checkpoint = args.checkpoint.resolve()
    stat = checkpoint.stat()
    checks = {
        "passed": payload.get("passed") is True,
        "path_matches": payload.get("path") == str(checkpoint),
        "label_matches": payload.get("label") == args.checkpoint_label,
        "size_matches": int(payload.get("bytes", -1)) == stat.st_size,
        "mtime_matches": int(payload.get("mtime_ns", -1)) == stat.st_mtime_ns,
    }
    return {
        "path": str(path),
        "sha256": M1.sha256_file(path),
        "payload": payload,
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_reference(args: argparse.Namespace, architecture: dict[str, Any]) -> dict[str, Any]:
    directory = args.mech01_reference_smoke_dir.resolve()
    manifest = read_json(directory / "mech01_manifest.json")
    schema = read_json(directory / "checkpoint_schema.json")
    route = read_json(directory / "route_audit.json")
    production = read_json(directory / "production_path_audit.json")
    invariance = read_json(directory / "state_invariance.json")
    checks = {
        "manifest_passed": manifest.get("passed") is True,
        "reference_family": schema.get("family") == "llama1b",
        "architecture_matches": schema.get("architecture") == architecture,
        "source_matches": schema.get("source_sha256")
        == M1.sha256_file(args.source_script.resolve()),
        "route_passed": route.get("passed") is True,
        "production_passed": production.get("ns5", {}).get("allclose") is True,
        "triton_matches": production.get("triton", {}).get("sha256")
        == M1.sha256_file(args.triton_kernels.resolve()),
        "reference_invariance": (
            invariance.get("model_unchanged") is True
            and invariance.get("optimizer_loader_unchanged") is True
            and invariance.get("checkpoint_file_unchanged") is True
        ),
    }
    return {
        "directory": str(directory),
        "reference_checkpoint_sha256": schema.get("checkpoint_sha256_observed"),
        "current_checkpoint_may_differ": True,
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_smoke_gate(args: argparse.Namespace, contract_sha: str) -> dict[str, Any]:
    if args.analysis_tier == "smoke":
        return {"required": False, "passed": True}
    if args.smoke_manifest is None:
        raise RuntimeError("formal MECH-06 requires --smoke-manifest")
    payload = read_json(args.smoke_manifest.resolve())
    checks = {
        "passed": payload.get("passed") is True,
        "tier": payload.get("analysis_tier") == "smoke",
        "label": payload.get("checkpoint_label") == args.checkpoint_label,
        "checkpoint_sha": payload.get("checkpoint_sha256")
        == read_json(args.checkpoint_hash_certificate)["sha256"],
        "contract_sha": payload.get("confirmation_contract_sha256") == contract_sha,
        "worker_version": payload.get("script_version") == SCRIPT_VERSION,
    }
    return {
        "required": True,
        "manifest": str(args.smoke_manifest.resolve()),
        "manifest_sha256": M1.sha256_file(args.smoke_manifest.resolve()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def geometry_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_pattern=args.data_pattern,
        device_batch_size=args.device_batch_size,
        sequence_length=args.sequence_length,
        repeat_offsets=args.geometry_offsets,
        repeats=args.repeats,
        batches_per_repeat=args.geometry_batches_per_repeat,
        max_activation_rows=args.max_geometry_rows,
    )


def shadow_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_pattern=args.data_pattern,
        device_batch_size=args.device_batch_size,
        sequence_length=args.sequence_length,
        repeat_offsets=args.shadow_offsets,
        repeats=args.repeats,
        batches_per_split=args.shadow_batches_per_split,
    )


def lowrank_geometry(
    activation: Tensor, layer: int, repeat: int, args: argparse.Namespace
) -> tuple[dict[str, Any], Tensor, Tensor, Tensor]:
    x = activation.cuda().float()
    n, width = x.shape
    diagonal = x.square().mean(dim=0)
    gram = x @ x.T / float(n)
    eigenvalues, left_vectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0)
    ridge = diagonal.mean() * args.ridge_mult + args.ridge_eps
    damped_values = torch.cat(
        [
            torch.full(
                (max(0, width - n),),
                float(ridge),
                device=x.device,
                dtype=eigenvalues.dtype,
            ),
            eigenvalues + ridge,
        ]
    )
    probability = damped_values / damped_values.sum().clamp_min(1e-30)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
    top_k = min(args.top_k, n, width)
    top_values = eigenvalues[-top_k:].clamp_min(args.ridge_eps)
    top_left = left_vectors[:, -top_k:]
    top_vectors = x.T @ top_left
    top_vectors /= torch.sqrt(float(n) * top_values).unsqueeze(0)
    top_vectors, _ = torch.linalg.qr(top_vectors, mode="reduced")
    covariance_sq = gram.square().sum().clamp_min(1e-30)
    diagonal_sq = diagonal.square().sum()
    offdiag_sq = (covariance_sq - diagonal_sq).clamp_min(0)
    positive_diag = diagonal.clamp_min(args.ridge_eps)
    row = {
        "family": "llama1b",
        "layer": layer,
        "repeat": repeat,
        "activation_rows": int(n),
        "activation_width": int(width),
        "n_eff_over_d": float(n / width),
        "diag_mean": float(diagonal.mean()),
        "diag_std": float(diagonal.std(unbiased=False)),
        "diag_cv": float(diagonal.std(unbiased=False) / diagonal.mean().clamp_min(args.ridge_eps)),
        "diag_p05": float(torch.quantile(diagonal, 0.05)),
        "diag_p50": float(torch.quantile(diagonal, 0.50)),
        "diag_p95": float(torch.quantile(diagonal, 0.95)),
        "diag_p95_over_p05": float(
            torch.quantile(diagonal, 0.95)
            / torch.quantile(diagonal, 0.05).clamp_min(args.ridge_eps)
        ),
        "log_diag_p05": float(torch.quantile(positive_diag.log(), 0.05)),
        "log_diag_p50": float(torch.quantile(positive_diag.log(), 0.50)),
        "log_diag_p95": float(torch.quantile(positive_diag.log(), 0.95)),
        "offdiag_frobenius": float(torch.sqrt(offdiag_sq)),
        "offdiag_energy_fraction": float(offdiag_sq / covariance_sq),
        "damped_ridge": float(ridge),
        "damped_eigen_min": float(damped_values.min()),
        "damped_eigen_max": float(damped_values.max()),
        "damped_condition_number": float(
            damped_values.max() / damped_values.min().clamp_min(1e-30)
        ),
        "damped_effective_rank": float(torch.exp(entropy)),
        "damped_top1_mass": float(probability.max()),
        "damped_top10_mass": float(
            torch.topk(probability, min(10, probability.numel())).values.sum()
        ),
        "spectrum_method": "exact_dual_gram_low_rank",
        "activation_sha256": M1.tensor_sha256(activation),
        "dual_gram_sha256": M1.tensor_sha256(gram.cpu()),
        "top_eigenspace_sha256": M1.tensor_sha256(top_vectors.cpu()),
    }
    return (
        row,
        activation.contiguous(),
        diagonal.cpu().contiguous(),
        top_vectors.cpu().contiguous(),
    )


def lowrank_stability(
    layer: int,
    activations: list[Tensor],
    diagonals: list[Tensor],
    top_vectors: list[Tensor],
) -> list[dict[str, Any]]:
    rows = []
    for left, right in itertools.combinations(range(len(activations)), 2):
        a = activations[left].cuda().float()
        b = activations[right].cuda().float()
        na, nb = a.size(0), b.size(0)
        norm_a_sq = float((a @ a.T).square().sum() / (na * na))
        norm_b_sq = float((b @ b.T).square().sum() / (nb * nb))
        inner = float((a @ b.T).square().sum() / (na * nb))
        norm_a = math.sqrt(max(norm_a_sq, 1e-30))
        norm_b = math.sqrt(max(norm_b_sq, 1e-30))
        drift_sq = max(0.0, norm_a_sq + norm_b_sq - 2.0 * inner)
        qa = top_vectors[left].cuda().float()
        qb = top_vectors[right].cuda().float()
        overlap = torch.linalg.matrix_norm(qa.T @ qb).square() / qa.size(1)
        rows.append(
            {
                "family": "llama1b",
                "layer": layer,
                "repeat_a": left,
                "repeat_b": right,
                "covariance_relative_drift": math.sqrt(drift_sq)
                / math.sqrt(max(norm_a * norm_b, 1e-30)),
                "covariance_cosine": inner / max(norm_a * norm_b, 1e-30),
                "diagonal_cosine": M1.matrix_cosine(
                    diagonals[left].cuda().float(), diagonals[right].cuda().float()
                ),
                "top_eigenspace_overlap": float(overlap),
                "top_k": int(qa.size(1)),
                "method": "exact_dual_gram_low_rank",
            }
        )
        del a, b, qa, qb
    return rows


def woodbury_apply(gradient: Tensor, activation: Tensor, ridge: Tensor) -> tuple[Tensor, float]:
    """Apply the exact low-rank inverse in a cancellation-resistant basis.

    Direct Woodbury evaluation subtracts two O(1/ridge) terms and is unstable
    in float32 for the 5504-wide LLaMA-1B activation.  The thin SVD expresses
    the same inverse as orthogonal parallel/perpendicular components.
    """
    x = activation.to(dtype=torch.float64)
    g = gradient.to(device=x.device, dtype=torch.float64)
    ridge_value = ridge.to(device=x.device, dtype=torch.float64).clamp_min(1e-12)
    _left, singular_values, right_t = torch.linalg.svd(
        x, full_matrices=False
    )
    right = right_t.T.contiguous()
    eigenvalues = singular_values.square() / float(x.size(0))
    projected_coefficients = g @ right
    parallel = projected_coefficients @ right.T
    perpendicular = g - parallel
    result = perpendicular / ridge_value
    result += (
        projected_coefficients / (eigenvalues + ridge_value).unsqueeze(0)
    ) @ right.T
    u = x / math.sqrt(float(x.size(0)))
    reconstructed = result * ridge_value + (result @ u.T) @ u
    residual = float(
        torch.linalg.vector_norm(reconstructed - g)
        / torch.linalg.vector_norm(g).clamp_min(1e-30)
    )
    return result, residual


def candidate_updates(
    activations: dict[int, Tensor],
    gradients: dict[int, Tensor],
    momenta: dict[int, Tensor],
    args: argparse.Namespace,
    production_ns: Any,
    repeat: int,
    direction: str,
    build_split: str,
) -> tuple[dict[int, dict[str, Tensor]], list[dict[str, Any]]]:
    updates: dict[int, dict[str, Tensor]] = {}
    rows: list[dict[str, Any]] = []
    for layer in sorted(activations):
        x = activations[layer].cuda().float()
        gradient = gradients[layer].cuda().float()
        old_momentum = momenta[layer].cuda().float()
        diagonal = x.square().mean(dim=0)
        ridge = diagonal.mean() * args.ridge_mult + args.ridge_eps
        preconditioned = {
            "none": gradient,
            "diag": gradient / (diagonal + ridge).unsqueeze(0),
        }
        dense, dense_residual = woodbury_apply(gradient, x, ridge)
        preconditioned["dense_full"] = dense
        layer_updates: dict[str, Tensor] = {}
        for candidate in ("none", "diag", "dense_full"):
            value = preconditioned[candidate]
            _, lookahead = M1.momentum_lookahead(
                value, old_momentum, "llama1b", args.momentum
            )
            update = production_ns(lookahead.float(), steps=args.ns_steps).float()
            layer_updates[candidate] = update.cpu().contiguous()
            rows.append(
                {
                    "family": "llama1b",
                    "repeat": repeat,
                    "direction": direction,
                    "build_split": build_split,
                    "layer": layer,
                    "candidate": candidate,
                    "activation_rows": int(x.size(0)),
                    "activation_width": int(x.size(1)),
                    "ridge_mean": float(ridge),
                    "inverse_method": (
                        "exact_woodbury_low_rank_svd_stabilized"
                        if candidate == "dense_full"
                        else candidate
                    ),
                    "inverse_residual_relative": (
                        dense_residual if candidate == "dense_full" else 0.0
                    ),
                    "inverse_compute_dtype": (
                        "float64" if candidate == "dense_full" else "float32"
                    ),
                    "update_sha256": M1.tensor_sha256(layer_updates[candidate]),
                    "gradient_norm": float(torch.linalg.vector_norm(gradient)),
                    "preconditioned_gradient_norm": float(
                        torch.linalg.vector_norm(value)
                    ),
                    "update_norm": float(torch.linalg.vector_norm(update)),
                    "gradient_to_update_cosine": M1.matrix_cosine(gradient, update),
                    "update_finite": bool(torch.isfinite(update).all()),
                }
            )
        reference = layer_updates["none"].float()
        for row in rows[-3:]:
            update = layer_updates[row["candidate"]].float()
            row["update_cosine_to_none"] = M1.matrix_cosine(update, reference)
            row["update_relative_norm_to_none"] = float(
                torch.linalg.vector_norm(update)
                / torch.linalg.vector_norm(reference).clamp_min(1e-30)
            )
        updates[layer] = layer_updates
        del x, gradient, old_momentum, dense, preconditioned
        torch.cuda.empty_cache()
    return updates, rows


def write_stability_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write formal rows or a schema-valid header-only smoke artifact."""
    if rows:
        M2.write_csv(path, rows)
        return
    fields = [
        "family",
        "layer",
        "repeat_a",
        "repeat_b",
        "covariance_relative_drift",
        "covariance_cosine",
        "diagonal_cosine",
        "top_eigenspace_overlap",
        "top_k",
        "method",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MECH-06 requires CUDA")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    output = args.output_dir.resolve()
    if not output.is_dir():
        raise RuntimeError(f"controller must create output directory: {output}")
    M1.atomic_json(output / "status.json", {"status": "running", "script_version": SCRIPT_VERSION})

    confirmation, contract_audit = validate_contracts(args)
    if not contract_audit["passed"]:
        raise RuntimeError(f"contract audit failed: {contract_audit}")
    hash_audit = validate_hash_certificate(args)
    if not hash_audit["passed"]:
        raise RuntimeError(f"checkpoint hash certificate failed: {hash_audit}")
    smoke_gate = validate_smoke_gate(
        args, contract_audit["confirmation_contract_sha256"]
    )
    if not smoke_gate["passed"]:
        raise RuntimeError(f"smoke gate failed: {smoke_gate}")
    shutil.copyfile(args.confirmation_contract, output / "confirmation_contract.json")
    shutil.copyfile(args.mech05_contract, output / "mech05_selection_rule.json")

    checkpoint_path = args.checkpoint.resolve()
    source_path = args.source_script.resolve()
    triton_path = args.triton_kernels.resolve()
    before_stat = checkpoint_path.stat()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_sha = confirmation["checkpoints"][args.checkpoint_label]["sha256"]
    model_state, schema = M1.checkpoint_schema(
        checkpoint_path, checkpoint, "llama1b", source_path, expected_sha, False
    )
    M1.attach_profile_provenance(schema, args.profile_script)
    expected_architecture = confirmation["architecture"]
    architecture_checks = {
        "step": int(schema["step"])
        == int(confirmation["checkpoints"][args.checkpoint_label]["step"]),
        "n_layer": schema["architecture"]["n_layer"] == expected_architecture["n_layer"],
        "n_embd": schema["architecture"]["n_embd"] == expected_architecture["n_embd"],
        "target_input_width": schema["architecture"]["target_input_width"]
        == expected_architecture["intermediate_size"],
        "method": schema["method_inferred"] == "down_none",
    }
    if not schema["passed"] or not all(architecture_checks.values()):
        raise RuntimeError(f"checkpoint schema/architecture failed: {architecture_checks}")
    reference = validate_reference(args, schema["architecture"])
    if not reference["passed"]:
        raise RuntimeError(f"MECH-01 reference rejected: {reference}")

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
    all_modules, _, all_names = M1.target_modules_and_weights(
        model, "llama1b", args.geometry_layers
    )
    shadow_modules, shadow_weights, shadow_names = M1.target_modules_and_weights(
        model, "llama1b", args.shadow_layers
    )
    momenta, momentum_audit = M1.extract_target_momenta(
        checkpoint, model, "llama1b", shadow_names
    )
    optimizer_hyperparameters = M3.matrix_optimizer_hyperparameters(
        checkpoint, shadow_weights
    )
    model_before = M1.model_state_signature(model, all_names.values())
    aux_before = M1.checkpoint_aux_signature(checkpoint)
    model.cuda()

    gargs = geometry_args(args)
    geometry_batches, geometry_batch_contract = M2.read_probe_batches(gargs)
    activations = M2.collect_activations(model, all_modules, geometry_batches, gargs)
    geometry_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for layer in args.geometry_layers:
        layer_activations, diagonals, top_vectors = [], [], []
        for repeat, activation in enumerate(activations[layer]):
            row, retained, diagonal, top = lowrank_geometry(
                activation, layer, repeat, args
            )
            geometry_rows.append(row)
            layer_activations.append(retained)
            diagonals.append(diagonal)
            top_vectors.append(top)
        stability_rows.extend(
            lowrank_stability(layer, layer_activations, diagonals, top_vectors)
        )
        del layer_activations, diagonals, top_vectors
    del activations, geometry_batches
    torch.cuda.empty_cache()

    sargs = shadow_args(args)
    shadow_batches, shadow_batch_contract = M3.read_crossfit_batches(sargs)
    split_builds: dict[int, dict[str, dict[str, Any]]] = {}
    for repeat in range(args.repeats):
        split_builds[repeat] = {}
        for split in M3.SPLITS:
            split_builds[repeat][split] = M3.collect_build_split(
                model,
                shadow_modules,
                shadow_weights,
                {},
                shadow_batches[repeat][split],
                args.max_shadow_rows,
            )
    update_rows, loss_rows, summary_rows = [], [], []
    for repeat in range(args.repeats):
        for direction, build_split, eval_split in M3.DIRECTIONS:
            build = split_builds[repeat][build_split]
            updates, update_geometry = candidate_updates(
                build["activations"],
                build["gradients"],
                momenta,
                args,
                production_ns,
                repeat,
                direction,
                build_split,
            )
            losses, summaries = M3.line_search(
                model,
                shadow_weights,
                updates,
                shadow_batches[repeat][eval_split],
                ["none", "diag", "dense_full"],
                args.step_multipliers,
                optimizer_hyperparameters,
                "llama1b",
                repeat,
                direction,
                build_split,
                eval_split,
                args.shadow_batches_per_split,
            )
            update_rows.extend(update_geometry)
            loss_rows.extend(losses)
            summary_rows.extend(summaries)
            del updates, update_geometry, losses, summaries
            torch.cuda.empty_cache()

    model_after = M1.model_state_signature(model, all_names.values())
    aux_after = M1.checkpoint_aux_signature(checkpoint)
    after_stat = checkpoint_path.stat()
    content_unchanged = M3.model_content_unchanged(model_before, model_after)
    invariance = {
        "model_content_unchanged": content_unchanged,
        "raw_model_signature_equal": model_before == model_after,
        "optimizer_loader_unchanged": aux_before == aux_after,
        "checkpoint_file_unchanged": (
            before_stat.st_size == after_stat.st_size
            and before_stat.st_mtime_ns == after_stat.st_mtime_ns
        ),
        "model_signature_before": model_before,
        "model_signature_after": model_after,
        "optimizer_loader_signature_before": aux_before,
        "optimizer_loader_signature_after": aux_after,
        "checkpoint_stat_before": {"size": before_stat.st_size, "mtime_ns": before_stat.st_mtime_ns},
        "checkpoint_stat_after": {"size": after_stat.st_size, "mtime_ns": after_stat.st_mtime_ns},
    }
    directions = args.repeats * 2
    expected_summary = directions * (len(args.shadow_layers) + 1) * 3
    checks = {
        "contract_audit": contract_audit["passed"],
        "checkpoint_hash_certificate": hash_audit["passed"],
        "smoke_gate": smoke_gate["passed"],
        "checkpoint_schema": schema["passed"] and all(architecture_checks.values()),
        "mech01_reference": reference["passed"],
        "source_runtime": source_config["passed"],
        "triton_provenance": triton_audit["passed"],
        "geometry_batch_contract": geometry_batch_contract["all_windows_disjoint"],
        "shadow_batch_contract": shadow_batch_contract["all_windows_disjoint"],
        "geometry_rows": len(geometry_rows)
        == len(args.geometry_layers) * args.repeats,
        "stability_rows": len(stability_rows)
        == len(args.geometry_layers) * math.comb(args.repeats, 2),
        "update_rows": len(update_rows)
        == directions * len(args.shadow_layers) * 3,
        "summary_rows": len(summary_rows) == expected_summary,
        "loss_rows": len(loss_rows)
        == expected_summary * len(args.step_multipliers),
        "geometry_finite": M1.finite_numbers(geometry_rows),
        "stability_finite": M1.finite_numbers(stability_rows),
        "updates_finite": M1.finite_numbers(update_rows)
        and all(row["update_finite"] for row in update_rows),
        "shadow_finite": M1.finite_numbers(loss_rows)
        and M1.finite_numbers(summary_rows),
        "woodbury_health": all(
            row["candidate"] != "dense_full"
            or row["inverse_residual_relative"] <= 1e-3
            for row in update_rows
        ),
        "historical_momentum": momentum_audit["all_present"],
        "optimizer_hyperparameters": optimizer_hyperparameters["passed"],
        "model_content_unchanged": invariance["model_content_unchanged"],
        "optimizer_loader_unchanged": invariance["optimizer_loader_unchanged"],
        "checkpoint_file_unchanged": invariance["checkpoint_file_unchanged"],
        "hvp_not_run": True,
        "existing_training_rankings_not_read": True,
    }
    passed = all(checks.values())
    artifacts = {
        "contract_audit.json": contract_audit,
        "checkpoint_hash_audit.json": hash_audit,
        "smoke_gate.json": smoke_gate,
        "checkpoint_schema.json": schema,
        "architecture_checks.json": architecture_checks,
        "mech01_reference.json": reference,
        "source_runtime_config.json": source_config,
        "runtime.json": M1.runtime_metadata(args),
        "momentum_audit.json": momentum_audit,
        "matrix_optimizer_hyperparameters.json": optimizer_hyperparameters,
        "geometry_batch_contract.json": geometry_batch_contract,
        "shadow_batch_contract.json": shadow_batch_contract,
        "state_invariance.json": invariance,
        "checks.json": checks,
    }
    for name, value in artifacts.items():
        M1.atomic_json(output / name, value)
    M2.write_csv(output / "geometry.csv", geometry_rows)
    write_stability_csv(output / "stability.csv", stability_rows)
    M2.write_csv(output / "update_geometry.csv", update_rows)
    M2.write_csv(output / "shadow_losses.csv", loss_rows)
    M2.write_csv(output / "line_search_summary.csv", summary_rows)
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "analysis_tier": args.analysis_tier,
        "passed": passed,
        "family": "llama1b",
        "method": "down_none",
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_step": int(schema["step"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hash_audit["payload"]["sha256"],
        "confirmation_contract_sha256": contract_audit["confirmation_contract_sha256"],
        "mech05_contract_sha256": contract_audit["mech05_contract_sha256"],
        "geometry_layers": args.geometry_layers,
        "shadow_layers": args.shadow_layers,
        "repeats": args.repeats,
        "geometry_rows": len(geometry_rows),
        "stability_rows": len(stability_rows),
        "update_rows": len(update_rows),
        "shadow_loss_rows": len(loss_rows),
        "line_search_summary_rows": len(summary_rows),
        "hvp_run": False,
        "existing_training_rankings_read": False,
        "artifacts": sorted(path.name for path in output.iterdir()),
    }
    M1.atomic_json(output / "mech06_manifest.json", manifest)
    M1.atomic_json(output / "status.json", {"status": "passed" if passed else "failed", "script_version": SCRIPT_VERSION})
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        try:
            index = sys.argv.index("--output-dir") + 1
            failure_output = Path(sys.argv[index]).resolve()
            if failure_output.is_dir():
                M1.atomic_json(
                    failure_output / "status.json",
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
