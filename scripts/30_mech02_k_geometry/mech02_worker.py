#!/usr/bin/env python3
"""CUDA worker for read-only MECH-02 checkpoint K-geometry."""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import itertools
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


SCRIPT_VERSION = "2026-07-27.1"
HERE = Path(__file__).resolve().parent
MECH01_WORKER = HERE.parent / "27_mech01_unified_k_diagnostics" / "mech01_worker.py"
FAMILIES = ("r1", "gpt_bridge", "llama124")
R1_FAMILIES = {"r1", "gpt_bridge"}


def load_mech01() -> Any:
    spec = importlib.util.spec_from_file_location("mech01_certified_runtime", MECH01_WORKER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MECH-01 worker: {MECH01_WORKER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1 = load_mech01()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-tier", choices=("smoke", "formal"), required=True)
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--hash-checkpoint", action="store_true")
    parser.add_argument("--source-script", type=Path, required=True)
    parser.add_argument("--triton-kernels", type=Path, required=True)
    parser.add_argument("--mech01-smoke-dir", type=Path, required=True)
    parser.add_argument("--data-pattern", required=True)
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--repeat-offsets", nargs="+", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--batches-per-repeat", type=int, required=True)
    parser.add_argument("--device-batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--max-activation-rows", type=int, required=True)
    parser.add_argument("--ridge-mult", type=float, required=True)
    parser.add_argument("--ridge-eps", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--spectrum-dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--execution-domain", required=True)
    return parser.parse_args()


def validate_smoke_gate(args: argparse.Namespace) -> dict[str, Any]:
    if args.analysis_tier == "smoke":
        return {"required": False, "passed": True}
    if args.smoke_manifest is None:
        raise RuntimeError("formal MECH-02 requires a smoke manifest")
    path = args.smoke_manifest.resolve()
    payload = read_json(path)
    checks = {
        "manifest_passed": payload.get("passed") is True,
        "analysis_tier_smoke": payload.get("analysis_tier") == "smoke",
        "family_matches": payload.get("family") == args.family,
        "method_matches": payload.get("method") == args.method,
        "checkpoint_sha256_matches": (
            payload.get("checkpoint_sha256", "").lower()
            == args.checkpoint_sha256.lower()
        ),
        "script_version_matches": payload.get("script_version") == SCRIPT_VERSION,
    }
    return {
        "required": True,
        "manifest": str(path),
        "manifest_sha256": M1.sha256_file(path),
        "checks": checks,
        "passed": all(checks.values()),
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def validate_mech01_certificate(args: argparse.Namespace) -> dict[str, Any]:
    directory = args.mech01_smoke_dir.resolve()
    required = [
        "mech01_manifest.json",
        "status.json",
        "checks.csv",
        "checkpoint_schema.json",
        "route_audit.json",
        "production_path_audit.json",
        "repeatability.json",
        "state_invariance.json",
        "source_runtime_config.json",
    ]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"MECH-01 certificate missing artifacts: {missing}")
    manifest = read_json(directory / "mech01_manifest.json")
    status = read_json(directory / "status.json")
    schema = read_json(directory / "checkpoint_schema.json")
    route = read_json(directory / "route_audit.json")
    production = read_json(directory / "production_path_audit.json")
    repeatability = read_json(directory / "repeatability.json")
    invariance = read_json(directory / "state_invariance.json")
    source_runtime = read_json(directory / "source_runtime_config.json")
    with (directory / "checks.csv").open("r", encoding="utf-8", newline="") as handle:
        check_rows = list(csv.DictReader(handle))
    failed_checks = [row["check"] for row in check_rows if row.get("passed") != "True"]
    source_sha = M1.sha256_file(args.source_script.resolve())
    triton_sha = M1.sha256_file(args.triton_kernels.resolve())
    checks = {
        "manifest_passed": manifest.get("passed") is True,
        "status_passed": status.get("status") == "passed",
        "worker_version_certified": manifest.get("script_version") == M1.SCRIPT_VERSION,
        "all_checks_passed": bool(check_rows) and not failed_checks,
        "family_matches": schema.get("family") == args.family,
        "checkpoint_sha256_matches": (
            schema.get("checkpoint_sha256_observed", "").lower()
            == args.checkpoint_sha256.lower()
        ),
        "source_sha256_matches": schema.get("source_sha256") == source_sha,
        "route_passed": route.get("passed") is True,
        "production_path_passed": (
            production.get("ns5", {}).get("finite") is True
            and production.get("ns5", {}).get("allclose") is True
        ),
        "triton_sha256_matches": (
            production.get("triton", {}).get("passed") is True
            and production.get("triton", {}).get("sha256") == triton_sha
        ),
        "repeatability_passed": repeatability.get("passed") is True,
        "state_invariance_passed": (
            invariance.get("model_unchanged") is True
            and invariance.get("optimizer_loader_unchanged") is True
            and invariance.get("checkpoint_file_unchanged") is True
        ),
        "source_runtime_passed": source_runtime.get("passed") is True,
    }
    return {
        "directory": str(directory),
        "manifest_sha256": M1.sha256_file(directory / "mech01_manifest.json"),
        "mech01_worker_sha256": M1.sha256_file(MECH01_WORKER),
        "mech01_worker_version": M1.SCRIPT_VERSION,
        "source_sha256": source_sha,
        "triton_sha256": triton_sha,
        "failed_mech01_checks": failed_checks,
        "checks": checks,
        "passed": all(checks.values()),
    }


def read_probe_batches(args: argparse.Namespace) -> tuple[list[tuple[Tensor, Tensor]], dict[str, Any]]:
    paths = sorted(Path(value).resolve() for value in glob.glob(args.data_pattern))
    if not paths:
        raise FileNotFoundError(f"no files match data pattern: {args.data_pattern}")
    path = paths[0]
    with path.open("rb") as handle:
        header = np.frombuffer(handle.read(256 * 4), dtype=np.int32)
    if len(header) != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise RuntimeError(f"unsupported FineWeb shard header: {path}")
    token_count = int(header[2])
    count = args.device_batch_size * args.sequence_length + 1
    tokens = np.memmap(
        path, dtype=np.uint16, mode="r", offset=256 * 4, shape=(token_count,)
    )
    batches: list[tuple[Tensor, Tensor]] = []
    rows: list[dict[str, Any]] = []
    intervals: list[tuple[int, int]] = []
    for index, offset in enumerate(args.repeat_offsets):
        if offset < 0 or offset + count > token_count:
            raise RuntimeError(f"probe window exceeds shard: {offset}:{offset + count}")
        window = np.asarray(tokens[offset : offset + count], dtype=np.int64)
        x = torch.from_numpy(window[:-1].copy()).view(
            args.device_batch_size, args.sequence_length
        )
        y = torch.from_numpy(window[1:].copy()).view(
            args.device_batch_size, args.sequence_length
        )
        repeat = index // args.batches_per_repeat
        intervals.append((offset, offset + count))
        rows.append(
            {
                "batch_index": index,
                "repeat": repeat,
                "batch_within_repeat": index % args.batches_per_repeat,
                "offset": offset,
                "exclusive_end": offset + count,
                "x_sha256": M1.tensor_sha256(x),
                "y_sha256": M1.tensor_sha256(y),
            }
        )
        batches.append((x, y))
    overlaps = []
    for left, right in itertools.combinations(range(len(intervals)), 2):
        a0, a1 = intervals[left]
        b0, b1 = intervals[right]
        if max(a0, b0) < min(a1, b1):
            overlaps.append([left, right])
    contract = {
        "schema_version": 1,
        "data_pattern": args.data_pattern,
        "selected_shard": str(path),
        "shard_size_bytes": path.stat().st_size,
        "shard_token_count": token_count,
        "device_batch_size": args.device_batch_size,
        "sequence_length": args.sequence_length,
        "rows_per_batch": args.device_batch_size * args.sequence_length,
        "repeats": args.repeats,
        "batches_per_repeat": args.batches_per_repeat,
        "all_windows_disjoint": not overlaps,
        "overlapping_pairs": overlaps,
        "batches": rows,
        "contract_sha256": M1.json_sha256(rows),
    }
    if overlaps:
        raise RuntimeError(f"MECH-02 probe windows overlap: {overlaps}")
    return batches, contract


def collect_activations(
    model: nn.Module,
    modules: dict[int, nn.Module],
    batches: list[tuple[Tensor, Tensor]],
    args: argparse.Namespace,
) -> dict[int, list[Tensor]]:
    current: dict[int, Tensor] = {}
    handles = []
    for layer, module in modules.items():
        def capture(
            _module: nn.Module, inputs: tuple[Any, ...], layer_index: int = layer
        ) -> None:
            value = inputs[0]
            if not isinstance(value, Tensor):
                raise TypeError(f"layer {layer_index} target input is not a tensor")
            current[layer_index] = value.detach()
        handles.append(module.register_forward_pre_hook(capture))
    per_layer_batch: dict[int, list[Tensor]] = {layer: [] for layer in modules}
    previous_training = model.training
    try:
        model.train()
        for x_cpu, y_cpu in batches:
            current.clear()
            x = x_cpu.cuda(non_blocking=False)
            y = y_cpu.cuda(non_blocking=False)
            with torch.no_grad(), torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                model(x, y, return_logits=False, precond_flag=False)
            if set(current) != set(modules):
                raise RuntimeError(
                    f"activation coverage mismatch: {sorted(current)} != {sorted(modules)}"
                )
            for layer, activation in current.items():
                per_layer_batch[layer].append(
                    activation.flatten(0, -2).float().cpu()
                )
        result: dict[int, list[Tensor]] = {}
        for layer, values in per_layer_batch.items():
            repeats = []
            for repeat in range(args.repeats):
                start = repeat * args.batches_per_repeat
                stop = start + args.batches_per_repeat
                joined = torch.cat(values[start:stop], dim=0)
                repeats.append(
                    M1.deterministic_subsample_rows(
                        joined, args.max_activation_rows
                    ).contiguous()
                )
            result[layer] = repeats
        return result
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous_training)
        torch.cuda.empty_cache()


def quantile(value: Tensor, probability: float) -> float:
    return float(torch.quantile(value, probability).item())


def geometry_for_repeat(
    activation: Tensor,
    family: str,
    layer: int,
    repeat: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Tensor, Tensor]:
    x = activation.cuda().float()
    covariance = x.T @ x / float(x.size(0))
    diagonal = covariance.diagonal()
    ridge = diagonal.mean() * args.ridge_mult + args.ridge_eps
    spectrum_dtype = torch.float64 if args.spectrum_dtype == "float64" else torch.float32
    damped = covariance.to(spectrum_dtype)
    damped.diagonal().add_(ridge.to(spectrum_dtype))
    eigenvalues, eigenvectors = torch.linalg.eigh(damped)
    eigenvalues = eigenvalues.clamp_min(0)
    total_eigen = eigenvalues.sum().clamp_min(torch.finfo(eigenvalues.dtype).eps)
    probability = eigenvalues / total_eigen
    entropy = -(probability * probability.clamp_min(1e-300).log()).sum()
    top_k = min(args.top_k, covariance.size(0))
    top_vectors = eigenvectors[:, -top_k:].float().cpu().contiguous()
    offdiag = covariance.clone()
    offdiag.diagonal().zero_()
    covariance_sq = covariance.square().sum().clamp_min(1e-30)
    offdiag_sq = offdiag.square().sum()
    work = damped.clone()
    factor, info = torch.linalg.cholesky_ex(work, upper=False, check_errors=False)
    info_value = int(info.item())
    inverse_residual = float("inf")
    if info_value == 0:
        inverse = torch.cholesky_inverse(factor, upper=False)
        identity = torch.eye(work.size(0), device=work.device, dtype=work.dtype)
        inverse_residual = float(
            torch.linalg.vector_norm(work @ inverse - identity)
            / torch.linalg.vector_norm(identity)
        )
        del inverse, identity
    positive_diag = diagonal.clamp_min(args.ridge_eps)
    log_diag = positive_diag.log()
    row: dict[str, Any] = {
        "family": family,
        "layer": layer,
        "repeat": repeat,
        "activation_rows": int(x.size(0)),
        "activation_width": int(x.size(1)),
        "n_eff_over_d": float(x.size(0) / x.size(1)),
        "diag_mean": float(diagonal.mean()),
        "diag_std": float(diagonal.std(unbiased=False)),
        "diag_cv": float(
            diagonal.std(unbiased=False) / diagonal.mean().clamp_min(args.ridge_eps)
        ),
        "diag_p05": quantile(diagonal, 0.05),
        "diag_p50": quantile(diagonal, 0.50),
        "diag_p95": quantile(diagonal, 0.95),
        "diag_p95_over_p05": float(
            torch.quantile(diagonal, 0.95)
            / torch.quantile(diagonal, 0.05).clamp_min(args.ridge_eps)
        ),
        "log_diag_p05": quantile(log_diag, 0.05),
        "log_diag_p50": quantile(log_diag, 0.50),
        "log_diag_p95": quantile(log_diag, 0.95),
        "offdiag_frobenius": float(torch.sqrt(offdiag_sq)),
        "offdiag_energy_fraction": float(offdiag_sq / covariance_sq),
        "damped_ridge": float(ridge),
        "damped_eigen_min": float(eigenvalues[0]),
        "damped_eigen_max": float(eigenvalues[-1]),
        "damped_condition_number": float(
            eigenvalues[-1]
            / eigenvalues[0].clamp_min(torch.finfo(eigenvalues.dtype).eps)
        ),
        "damped_effective_rank": float(torch.exp(entropy)),
        "damped_top1_mass": float(probability[-1]),
        "damped_top10_mass": float(probability[-min(10, len(probability)) :].sum()),
        "cholesky_info": info_value,
        "inverse_residual_relative": inverse_residual,
        "covariance_sha256": M1.tensor_sha256(covariance.cpu()),
        "top_eigenspace_sha256": M1.tensor_sha256(top_vectors),
    }
    if family in R1_FAMILIES:
        width = covariance.size(0) // 4
        within_sq = covariance.new_zeros(())
        for block in range(4):
            start = block * width
            stop = start + width
            within = covariance[start:stop, start:stop].clone()
            within.diagonal().zero_()
            within_sq += within.square().sum()
        cross_sq = (offdiag_sq - within_sq).clamp_min(0)
        denominator = offdiag_sq.clamp_min(1e-30)
        row.update(
            {
                "within_block_offdiag_energy_fraction": float(within_sq / denominator),
                "cross_block_offdiag_energy_fraction": float(cross_sq / denominator),
            }
        )
    del x, diagonal, offdiag, damped, factor, eigenvectors, eigenvalues
    torch.cuda.empty_cache()
    return row, covariance.cpu().contiguous(), top_vectors


def cosine(left: Tensor, right: Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.sum(left * right) / denominator.clamp_min(1e-30))


def stability_rows(
    family: str,
    layer: int,
    covariances: list[Tensor],
    top_vectors: list[Tensor],
) -> list[dict[str, Any]]:
    rows = []
    for left, right in itertools.combinations(range(len(covariances)), 2):
        a = covariances[left].float()
        b = covariances[right].float()
        scale = torch.sqrt(
            torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
        ).clamp_min(1e-30)
        qa = top_vectors[left].float()
        qb = top_vectors[right].float()
        overlap = torch.linalg.matrix_norm(qa.T @ qb).square() / qa.size(1)
        rows.append(
            {
                "family": family,
                "layer": layer,
                "repeat_a": left,
                "repeat_b": right,
                "covariance_relative_drift": float(
                    torch.linalg.vector_norm(a - b) / scale
                ),
                "covariance_cosine": cosine(a, b),
                "diagonal_cosine": cosine(a.diagonal(), b.diagonal()),
                "top_eigenspace_overlap": float(overlap),
                "top_k": int(qa.size(1)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MECH-02 requires CUDA")
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
    smoke_gate = validate_smoke_gate(args)
    M1.atomic_json(output / "smoke_gate.json", smoke_gate)
    if not smoke_gate["passed"]:
        raise RuntimeError(f"MECH-02 smoke gate rejected: {smoke_gate}")
    certificate = validate_mech01_certificate(args)
    M1.atomic_json(output / "mech01_certificate.json", certificate)
    if not certificate["passed"]:
        raise RuntimeError(f"MECH-01 certificate rejected: {certificate}")
    checkpoint_path = args.checkpoint.resolve()
    source_path = args.source_script.resolve()
    triton_path = args.triton_kernels.resolve()
    before_stat = checkpoint_path.stat()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state, schema = M1.checkpoint_schema(
        checkpoint_path,
        checkpoint,
        args.family,
        source_path,
        args.checkpoint_sha256,
        args.hash_checkpoint,
    )
    if not schema["passed"]:
        raise RuntimeError(f"checkpoint schema failed: {schema}")
    if args.hash_checkpoint and (
        schema["checkpoint_sha256_observed"].lower()
        != args.checkpoint_sha256.lower()
    ):
        raise RuntimeError("checkpoint differs from frozen MECH-00/01 SHA-256")
    layers = (
        M1.select_layers(schema["architecture"]["n_layer"], args.layers)
        if args.layers
        else list(range(schema["architecture"]["n_layer"]))
    )
    source_runtime, _production_ns, triton_audit = M1.load_source_runtime(
        args.family, source_path, triton_path
    )
    source_config = M1.configure_source_runtime_globals(
        args.family, source_runtime, args.method
    )
    model = M1.build_model(
        args.family, source_runtime, schema["architecture"], args.method
    )
    incompatible = model.load_state_dict(model_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"model load mismatch: {incompatible}")
    modules, _weights, target_names = M1.target_modules_and_weights(
        model, args.family, layers
    )
    model_before = M1.model_state_signature(model, target_names.values())
    aux_before = M1.checkpoint_aux_signature(checkpoint)
    batches, batch_contract = read_probe_batches(args)
    model.cuda()
    activations = collect_activations(model, modules, batches, args)
    geometry: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    for layer in layers:
        layer_covariances = []
        layer_top_vectors = []
        for repeat, activation in enumerate(activations[layer]):
            row, covariance, top_vectors = geometry_for_repeat(
                activation, args.family, layer, repeat, args
            )
            geometry.append(row)
            layer_covariances.append(covariance)
            layer_top_vectors.append(top_vectors)
        stability.extend(
            stability_rows(
                args.family, layer, layer_covariances, layer_top_vectors
            )
        )
        del layer_covariances, layer_top_vectors
    model_after = M1.model_state_signature(model, target_names.values())
    aux_after = M1.checkpoint_aux_signature(checkpoint)
    after_stat = checkpoint_path.stat()
    invariance = {
        "model_unchanged": model_before == model_after,
        "optimizer_loader_unchanged": aux_before == aux_after,
        "checkpoint_file_unchanged": (
            before_stat.st_size == after_stat.st_size
            and before_stat.st_mtime_ns == after_stat.st_mtime_ns
        ),
        "model_signature_before": model_before,
        "model_signature_after": model_after,
        "optimizer_loader_signature_before": aux_before,
        "optimizer_loader_signature_after": aux_after,
        "checkpoint_stat_before": {
            "size": before_stat.st_size,
            "mtime_ns": before_stat.st_mtime_ns,
        },
        "checkpoint_stat_after": {
            "size": after_stat.st_size,
            "mtime_ns": after_stat.st_mtime_ns,
        },
    }
    checks = {
        "smoke_gate": smoke_gate["passed"],
        "mech01_certificate": certificate["passed"],
        "checkpoint_schema": schema["passed"],
        "source_runtime": source_config["passed"],
        "triton_provenance": triton_audit["passed"],
        "all_probe_windows_disjoint": batch_contract["all_windows_disjoint"],
        "all_layers_covered": sorted(activations) == layers,
        "geometry_row_count": len(geometry) == len(layers) * args.repeats,
        "stability_row_count": len(stability)
        == len(layers) * math.comb(args.repeats, 2),
        "geometry_finite": M1.finite_numbers(geometry),
        "stability_finite": M1.finite_numbers(stability),
        "cholesky_health": all(row["cholesky_info"] == 0 for row in geometry),
        "model_unchanged": invariance["model_unchanged"],
        "optimizer_loader_unchanged": invariance["optimizer_loader_unchanged"],
        "checkpoint_file_unchanged": invariance["checkpoint_file_unchanged"],
        "single_layer_streaming": True,
    }
    passed = all(checks.values())
    M1.atomic_json(output / "checkpoint_schema.json", schema)
    M1.atomic_json(output / "batch_contract.json", batch_contract)
    M1.atomic_json(output / "source_runtime_config.json", source_config)
    M1.atomic_json(output / "runtime.json", M1.runtime_metadata(args))
    M1.atomic_json(output / "state_invariance.json", invariance)
    M1.atomic_json(
        output / "geometry_contract.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "family": args.family,
            "method": args.method,
            "analysis_tier": args.analysis_tier,
            "layers": layers,
            "repeats": args.repeats,
            "batches_per_repeat": args.batches_per_repeat,
            "repeat_offsets": args.repeat_offsets,
            "ridge_mult": args.ridge_mult,
            "ridge_eps": args.ridge_eps,
            "top_k": args.top_k,
            "spectrum_dtype": args.spectrum_dtype,
            "layer_processing": "sequential",
            "covariances_resident_layers": 1,
        },
    )
    M1.atomic_json(output / "geometry.json", {"rows": geometry})
    M1.atomic_json(output / "stability.json", {"rows": stability})
    M1.atomic_json(output / "checks.json", checks)
    write_csv(output / "geometry.csv", geometry)
    write_csv(output / "stability.csv", stability)
    M1.atomic_json(
        output / "mech02_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "stage": "k_geometry",
            "analysis_tier": args.analysis_tier,
            "passed": passed,
            "family": args.family,
            "method": args.method,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": args.checkpoint_sha256.lower(),
            "layers": layers,
            "geometry_rows": len(geometry),
            "stability_rows": len(stability),
            "artifacts": sorted(path.name for path in output.iterdir()),
        },
    )
    M1.atomic_json(
        output / "status.json",
        {
            "status": "passed" if passed else "failed",
            "script_version": SCRIPT_VERSION,
        },
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        try:
            output_index = sys.argv.index("--output-dir") + 1
            failure_output = Path(sys.argv[output_index]).resolve()
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
