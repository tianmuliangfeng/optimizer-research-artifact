#!/usr/bin/env python3
"""CUDA worker for read-only MECH-03 cross-fit shadow updates."""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import itertools
import json
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn


SCRIPT_VERSION = "2026-07-27.2"
PREDICTION_CONTRACT_VERSION = "2026-07-27.1"
HERE = Path(__file__).resolve().parent
MECH02_WORKER = HERE.parent / "30_mech02_k_geometry" / "mech02_worker.py"
FAMILIES = ("r1", "gpt_bridge", "llama124")
R1_FAMILIES = {"r1", "gpt_bridge"}
SPLITS = ("A", "B")
DIRECTIONS = (("A_to_B", "A", "B"), ("B_to_A", "B", "A"))


def load_mech02() -> Any:
    spec = importlib.util.spec_from_file_location("mech02_certified_runtime", MECH02_WORKER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MECH-02 worker: {MECH02_WORKER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M2 = load_mech02()
M1 = M2.M1


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
    parser.add_argument("--mech02-formal-dir", type=Path, required=True)
    parser.add_argument("--prediction-contract", type=Path, required=True)
    parser.add_argument("--data-pattern", required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--repeat-offsets", nargs="+", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--batches-per-split", type=int, required=True)
    parser.add_argument("--device-batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--max-activation-rows", type=int, required=True)
    parser.add_argument("--ridge-mult", type=float, required=True)
    parser.add_argument("--ridge-eps", type=float, required=True)
    parser.add_argument("--momentum", type=float, required=True)
    parser.add_argument("--ns-steps", type=int, required=True)
    parser.add_argument("--step-multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--execution-domain", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def validate_prediction_contract(args: argparse.Namespace) -> dict[str, Any]:
    path = args.prediction_contract.resolve()
    payload = read_json(path)
    design = payload.get("formal_design", {})
    candidate_sets = payload.get("candidate_sets", {})
    expected_candidates = candidate_sets.get(args.family)
    checks = {
        "script_version_matches": (
            payload.get("script_version") == PREDICTION_CONTRACT_VERSION
        ),
        "stage_matches": payload.get("stage") == "MECH-03 cross-fit shadow update",
        "family_candidate_set_present": isinstance(expected_candidates, list),
        "primary_direction_frozen": (
            payload.get("primary_prediction", {}).get("predicted_direction")
            == "positive"
        ),
        "mech04_not_auto_authorized": (
            payload.get("authorization_source", {}).get(
                "mech04_never_auto_authorized"
            )
            is True
        ),
        "step_multipliers_match": (
            [float(value) for value in design.get("step_multipliers", [])]
            == args.step_multipliers
        ),
        "momentum_matches": float(design.get("momentum_beta", -1)) == args.momentum,
        "ns_steps_match": int(design.get("newton_schulz_steps", -1)) == args.ns_steps,
        "ridge_matches": (
            float(design.get("ridge_multiplier", -1)) == args.ridge_mult
            and float(design.get("ridge_epsilon", -1)) == args.ridge_eps
        ),
    }
    if args.analysis_tier == "formal":
        checks.update(
            {
                "formal_layers_match": design.get("layers") == args.layers,
                "formal_repeats_match": design.get("repeats") == args.repeats,
                "formal_batches_per_split_match": (
                    design.get("batches_per_split") == args.batches_per_split
                ),
                "formal_sequence_length_match": (
                    design.get("sequence_length") == args.sequence_length
                ),
                "formal_device_batch_size_match": (
                    design.get("device_batch_size_per_window")
                    == args.device_batch_size
                ),
            }
        )
    return {
        "path": str(path),
        "sha256": M1.sha256_file(path),
        "payload": payload,
        "candidates": expected_candidates,
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
    }


def validate_smoke_gate(
    args: argparse.Namespace, prediction_sha256: str
) -> dict[str, Any]:
    if args.analysis_tier == "smoke":
        return {"required": False, "passed": True}
    if args.smoke_manifest is None:
        raise RuntimeError("formal MECH-03 requires a smoke manifest")
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
        "prediction_contract_matches": (
            payload.get("prediction_contract_sha256") == prediction_sha256
        ),
    }
    return {
        "required": True,
        "manifest": str(path),
        "manifest_sha256": M1.sha256_file(path),
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_mech02_certificate(args: argparse.Namespace) -> dict[str, Any]:
    directory = args.mech02_formal_dir.resolve()
    required = [
        "mech02_manifest.json",
        "checks.json",
        "state_invariance.json",
        "batch_contract.json",
        "geometry_contract.json",
    ]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"MECH-02 certificate missing artifacts: {missing}")
    manifest = read_json(directory / "mech02_manifest.json")
    checks_payload = read_json(directory / "checks.json")
    invariance = read_json(directory / "state_invariance.json")
    checks = {
        "manifest_passed": manifest.get("passed") is True,
        "analysis_tier_formal": manifest.get("analysis_tier") == "formal",
        "family_matches": manifest.get("family") == args.family,
        "method_matches": manifest.get("method") == args.method,
        "checkpoint_sha256_matches": (
            manifest.get("checkpoint_sha256", "").lower()
            == args.checkpoint_sha256.lower()
        ),
        "script_version_certified": manifest.get("script_version") == M2.SCRIPT_VERSION,
        "all_checks_passed": bool(checks_payload) and all(checks_payload.values()),
        "state_invariance_passed": (
            invariance.get("model_unchanged") is True
            and invariance.get("optimizer_loader_unchanged") is True
            and invariance.get("checkpoint_file_unchanged") is True
        ),
    }
    return {
        "directory": str(directory),
        "manifest_sha256": M1.sha256_file(directory / "mech02_manifest.json"),
        "checks": checks,
        "passed": all(checks.values()),
    }


def read_crossfit_batches(
    args: argparse.Namespace,
) -> tuple[dict[int, dict[str, tuple[Tensor, Tensor]]], dict[str, Any]]:
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
    split_values: dict[int, dict[str, list[tuple[Tensor, Tensor]]]] = {
        repeat: {"A": [], "B": []} for repeat in range(args.repeats)
    }
    rows = []
    intervals: list[tuple[int, int]] = []
    per_repeat = 2 * args.batches_per_split
    for index, offset in enumerate(args.repeat_offsets):
        if offset < 0 or offset + count > token_count:
            raise RuntimeError(f"probe window exceeds shard: {offset}:{offset + count}")
        repeat = index // per_repeat
        within_repeat = index % per_repeat
        split = SPLITS[within_repeat % 2]
        batch_within_split = within_repeat // 2
        window = np.asarray(tokens[offset : offset + count], dtype=np.int64)
        x = torch.from_numpy(window[:-1].copy()).view(
            args.device_batch_size, args.sequence_length
        )
        y = torch.from_numpy(window[1:].copy()).view(
            args.device_batch_size, args.sequence_length
        )
        split_values[repeat][split].append((x, y))
        intervals.append((offset, offset + count))
        rows.append(
            {
                "batch_index": index,
                "repeat": repeat,
                "split": split,
                "batch_within_split": batch_within_split,
                "offset": offset,
                "exclusive_end": offset + count,
                "x_sha256": M1.tensor_sha256(x),
                "y_sha256": M1.tensor_sha256(y),
            }
        )
    overlaps = []
    for left, right in itertools.combinations(range(len(intervals)), 2):
        a0, a1 = intervals[left]
        b0, b1 = intervals[right]
        if max(a0, b0) < min(a1, b1):
            overlaps.append([left, right])
    packed: dict[int, dict[str, tuple[Tensor, Tensor]]] = {}
    for repeat, values in split_values.items():
        packed[repeat] = {}
        for split in SPLITS:
            if len(values[split]) != args.batches_per_split:
                raise RuntimeError(
                    f"split batch count mismatch repeat={repeat} split={split}"
                )
            packed[repeat][split] = (
                torch.cat([pair[0] for pair in values[split]], dim=0),
                torch.cat([pair[1] for pair in values[split]], dim=0),
            )
    contract = {
        "schema_version": 1,
        "data_pattern": args.data_pattern,
        "selected_shard": str(path),
        "shard_size_bytes": path.stat().st_size,
        "shard_token_count": token_count,
        "device_batch_size_per_window": args.device_batch_size,
        "sequence_length": args.sequence_length,
        "repeats": args.repeats,
        "splits": list(SPLITS),
        "batches_per_split": args.batches_per_split,
        "packed_split_batch_size": args.batches_per_split * args.device_batch_size,
        "all_windows_disjoint": not overlaps,
        "overlapping_pairs": overlaps,
        "batches": rows,
        "contract_sha256": M1.json_sha256(rows),
    }
    if overlaps:
        raise RuntimeError(f"MECH-03 probe windows overlap: {overlaps}")
    return packed, contract


def collect_build_split(
    model: nn.Module,
    modules: dict[int, nn.Module],
    weights: dict[int, nn.Parameter],
    component_modules: dict[tuple[int, str], nn.Module],
    batch: tuple[Tensor, Tensor],
    max_rows: int,
) -> dict[str, Any]:
    captured: dict[int, Tensor] = {}
    captured_components: dict[tuple[int, str], Tensor] = {}
    handles = []
    for layer, module in modules.items():
        def capture(
            _module: nn.Module, inputs: tuple[Any, ...], layer_index: int = layer
        ) -> None:
            value = inputs[0]
            if not isinstance(value, Tensor):
                raise TypeError(f"target input for layer {layer_index} is not a tensor")
            captured[layer_index] = value.detach()
        handles.append(module.register_forward_pre_hook(capture))
    for key, module in component_modules.items():
        def capture_component(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            component_key: tuple[int, str] = key,
        ) -> None:
            if not isinstance(output, Tensor):
                raise TypeError(f"component output {component_key} is not a tensor")
            captured_components[component_key] = output.detach()
        handles.append(module.register_forward_hook(capture_component))
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    target_ids = {id(parameter) for parameter in weights.values()}
    previous_training = model.training
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(id(parameter) in target_ids)
        model.train()
        model.zero_grad(set_to_none=True)
        x = batch[0].cuda(non_blocking=False)
        y = batch[1].cuda(non_blocking=False)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y, return_logits=False, precond_flag=False)
        if loss is None:
            raise RuntimeError("build forward returned no loss")
        loss.backward()
        if set(captured) != set(modules):
            raise RuntimeError(
                f"activation coverage mismatch: {sorted(captured)} != {sorted(modules)}"
            )
        if set(captured_components) != set(component_modules):
            raise RuntimeError(
                "component coverage mismatch: "
                f"{sorted(captured_components)} != {sorted(component_modules)}"
            )
        activations = {}
        gradients = {}
        for layer in modules:
            activation = captured[layer].flatten(0, -2).float()
            activations[layer] = M1.deterministic_subsample_rows(
                activation, max_rows
            ).cpu().contiguous()
            gradient = weights[layer].grad
            if gradient is None:
                raise RuntimeError(f"no gradient captured for layer {layer}")
            gradients[layer] = gradient.detach().float().cpu().contiguous()
        components = {}
        for key, value in captured_components.items():
            flat = value.flatten(0, -2).float()
            components[key] = M1.deterministic_subsample_rows(
                flat, max_rows
            ).cpu().contiguous()
        for layer, activation in activations.items():
            components[(layer, "down_input")] = activation
        return {
            "loss": float(loss.detach().cpu()),
            "activations": activations,
            "gradients": gradients,
            "components": components,
        }
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_requires_grad[name])
        model.train(previous_training)


def rng_snapshot() -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(snapshot: dict[str, Any]) -> None:
    torch.set_rng_state(snapshot["cpu"])
    torch.cuda.set_rng_state_all(snapshot["cuda"])


def evaluate_loss(
    model: nn.Module,
    batch: tuple[Tensor, Tensor],
    snapshot: dict[str, Any],
) -> float:
    restore_rng(snapshot)
    previous_training = model.training
    try:
        model.train()
        x = batch[0].cuda(non_blocking=False)
        y = batch[1].cuda(non_blocking=False)
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            _, loss = model(x, y, return_logits=False, precond_flag=False)
        if loss is None:
            raise RuntimeError("held-out forward returned no loss")
        return float(loss.detach().cpu())
    finally:
        model.train(previous_training)


def matrix_optimizer_hyperparameters(
    checkpoint: dict[str, Any], weights: dict[int, nn.Parameter]
) -> dict[str, Any]:
    optimizers = checkpoint.get("optimizers")
    if not isinstance(optimizers, list) or len(optimizers) < 2:
        raise RuntimeError("matrix optimizer state is missing")
    groups = optimizers[1].get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("matrix optimizer parameter groups are missing")
    scheduled_learning_rates = {
        float(group["lr"]) for group in groups if "lr" in group
    }
    initial_learning_rates = {
        float(group.get("initial_lr", group["lr"]))
        for group in groups
        if "lr" in group
    }
    weight_decays = {
        float(group.get("weight_decay", 0.0)) for group in groups
    }
    if (
        len(scheduled_learning_rates) != 1
        or len(initial_learning_rates) != 1
        or len(weight_decays) != 1
    ):
        raise RuntimeError(
            f"non-uniform matrix optimizer hyperparameters: "
            f"scheduled_lr={scheduled_learning_rates}, "
            f"initial_lr={initial_learning_rates}, weight_decay={weight_decays}"
        )
    scheduled_learning_rate = next(iter(scheduled_learning_rates))
    initial_learning_rate = next(iter(initial_learning_rates))
    weight_decay = next(iter(weight_decays))
    target_rows = []
    for layer, weight in sorted(weights.items()):
        shape_multiplier = max(1.0, weight.size(-2) / weight.size(-1)) ** 0.5
        lr_multiplier = float(getattr(weight, "lr_mul", 1.0))
        wd_multiplier = float(getattr(weight, "wd_mul", 1.0))
        target_rows.append(
            {
                "layer": layer,
                "shape_multiplier": shape_multiplier,
                "lr_multiplier": lr_multiplier,
                "weight_decay_multiplier": wd_multiplier,
                "effective_update_learning_rate": (
                    initial_learning_rate * shape_multiplier * lr_multiplier
                ),
                "effective_decay_coefficient": (
                    initial_learning_rate * weight_decay * wd_multiplier
                ),
            }
        )
    finite_targets = all(
        math.isfinite(float(row["effective_update_learning_rate"]))
        and float(row["effective_update_learning_rate"]) > 0
        and math.isfinite(float(row["effective_decay_coefficient"]))
        and float(row["effective_decay_coefficient"]) >= 0
        for row in target_rows
    )
    return {
        "scheduled_learning_rate": scheduled_learning_rate,
        "initial_learning_rate": initial_learning_rate,
        "weight_decay": weight_decay,
        "param_groups": len(groups),
        "targets": target_rows,
        "passed": math.isfinite(scheduled_learning_rate)
        and scheduled_learning_rate >= 0
        and math.isfinite(initial_learning_rate)
        and initial_learning_rate > 0
        and math.isfinite(weight_decay)
        and weight_decay >= 0
        and finite_targets,
    }


def candidate_updates(
    activations: dict[int, Tensor],
    gradients: dict[int, Tensor],
    momenta: dict[int, Tensor],
    family: str,
    candidates: list[str],
    ridge_mult: float,
    ridge_eps: float,
    momentum: float,
    ns_steps: int,
    production_ns: Callable[..., Tensor],
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
        covariance = M1.covariance_from_activations(x)
        layer_updates: dict[str, Tensor] = {}
        for candidate in candidates:
            inverse, inverse_metrics = M1.representation_inverse(
                covariance, family, candidate, ridge_mult, ridge_eps
            )
            preconditioned = M1.apply_inverse(gradient, inverse)
            _, lookahead = M1.momentum_lookahead(
                preconditioned, old_momentum, family, momentum
            )
            update = production_ns(lookahead, steps=ns_steps).float()
            layer_updates[candidate] = update.cpu().contiguous()
            rows.append(
                {
                    "family": family,
                    "repeat": repeat,
                    "direction": direction,
                    "build_split": build_split,
                    "layer": layer,
                    "candidate": candidate,
                    "activation_rows": int(x.size(0)),
                    "activation_width": int(x.size(1)),
                    "activation_sha256": M1.tensor_sha256(activations[layer]),
                    "gradient_sha256": M1.tensor_sha256(gradients[layer]),
                    "historical_momentum_sha256": M1.tensor_sha256(momenta[layer]),
                    "update_sha256": M1.tensor_sha256(layer_updates[candidate]),
                    "gradient_norm": float(torch.linalg.vector_norm(gradient)),
                    "preconditioned_gradient_norm": float(
                        torch.linalg.vector_norm(preconditioned)
                    ),
                    "lookahead_norm": float(torch.linalg.vector_norm(lookahead)),
                    "update_norm": float(torch.linalg.vector_norm(update)),
                    "gradient_to_update_cosine": M1.matrix_cosine(gradient, update),
                    "update_finite": bool(torch.isfinite(update).all()),
                    **inverse_metrics,
                }
            )
            del inverse, preconditioned, lookahead, update
        reference = layer_updates["none"].float()
        reference_norm = torch.linalg.vector_norm(reference).clamp_min(1e-30)
        for row in rows[-len(candidates) :]:
            update = layer_updates[str(row["candidate"])].float()
            row["update_cosine_to_none"] = M1.matrix_cosine(update, reference)
            row["update_relative_norm_to_none"] = float(
                torch.linalg.vector_norm(update) / reference_norm
            )
        updates[layer] = layer_updates
        del x, gradient, old_momentum, covariance, reference
        torch.cuda.empty_cache()
    return updates, rows


def component_distribution_rows(
    split_builds: dict[int, dict[str, dict[str, Any]]],
    family: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if family != "llama124":
        return [], []
    rows: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    for repeat, split_values in sorted(split_builds.items()):
        for split in SPLITS:
            for (layer, component), value in sorted(
                split_values[split]["components"].items()
            ):
                flat = value.float()
                absolute = flat.abs()
                flattened_absolute = absolute.flatten()
                tail_stride = max(1, math.ceil(flattened_absolute.numel() / 131072))
                tail_sample = flattened_absolute[::tail_stride]
                second_moment = flat.square().mean(dim=0)
                rows.append(
                    {
                        "family": family,
                        "repeat": repeat,
                        "split": split,
                        "layer": layer,
                        "component": component,
                        "activation_rows": int(flat.size(0)),
                        "activation_width": int(flat.size(1)),
                        "mean": float(flat.mean()),
                        "std": float(flat.std(unbiased=False)),
                        "rms": float(torch.sqrt(flat.square().mean())),
                        "tail_sample_count": int(tail_sample.numel()),
                        "tail_sample_stride": tail_stride,
                        "abs_p95": float(torch.quantile(tail_sample, 0.95)),
                        "abs_p99": float(torch.quantile(tail_sample, 0.99)),
                        "abs_max": float(absolute.max()),
                        "second_moment_sha256": M1.tensor_sha256(second_moment),
                    }
                )
        keys = set(split_values["A"]["components"])
        if keys != set(split_values["B"]["components"]):
            raise RuntimeError(f"SwiGLU component split mismatch repeat={repeat}")
        for layer, component in sorted(keys):
            left = split_values["A"]["components"][(layer, component)].float()
            right = split_values["B"]["components"][(layer, component)].float()
            left_moment = left.square().mean(dim=0)
            right_moment = right.square().mean(dim=0)
            scale = torch.sqrt(
                torch.linalg.vector_norm(left_moment)
                * torch.linalg.vector_norm(right_moment)
            ).clamp_min(1e-30)
            stability.append(
                {
                    "family": family,
                    "repeat": repeat,
                    "layer": layer,
                    "component": component,
                    "second_moment_relative_drift": float(
                        torch.linalg.vector_norm(left_moment - right_moment)
                        / scale
                    ),
                    "second_moment_cosine": M1.matrix_cosine(
                        left_moment, right_moment
                    ),
                }
            )
    return rows, stability


def apply_shadow(
    weights: dict[int, nn.Parameter],
    originals: dict[int, Tensor],
    updates: dict[int, dict[str, Tensor]],
    layers: list[int],
    candidate: str,
    multiplier: float,
    step_hyperparameters: dict[int, dict[str, float]],
) -> None:
    with torch.no_grad():
        for layer in layers:
            weight = weights[layer]
            weight.copy_(originals[layer])
            if multiplier:
                row = step_hyperparameters[layer]
                weight.mul_(
                    1.0 - multiplier * row["effective_decay_coefficient"]
                )
                weight.add_(
                    updates[layer][candidate].to(
                        device=weight.device, dtype=weight.dtype
                    ),
                    alpha=-multiplier * row["effective_update_learning_rate"],
                )


def restore_weights(
    weights: dict[int, nn.Parameter], originals: dict[int, Tensor]
) -> None:
    with torch.no_grad():
        for layer, weight in weights.items():
            weight.copy_(originals[layer])


def line_search(
    model: nn.Module,
    weights: dict[int, nn.Parameter],
    updates: dict[int, dict[str, Tensor]],
    heldout_batch: tuple[Tensor, Tensor],
    candidates: list[str],
    step_multipliers: list[float],
    optimizer_hyperparameters: dict[str, Any],
    family: str,
    repeat: int,
    direction: str,
    build_split: str,
    eval_split: str,
    evaluation_batches: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    originals = {
        layer: weight.detach().clone() for layer, weight in weights.items()
    }
    step_hyperparameters = {
        int(row["layer"]): row
        for row in optimizer_hyperparameters["targets"]
    }
    snapshot = rng_snapshot()
    baseline = evaluate_loss(model, heldout_batch, snapshot)
    loss_rows: list[dict[str, Any]] = []
    try:
        scopes: list[tuple[str, str, list[int]]] = [
            ("layer", str(layer), [layer]) for layer in sorted(weights)
        ]
        grouped_layers = sorted(weights)
        scopes.append(
            (
                "grouped",
                "_".join(f"h{layer}" for layer in grouped_layers),
                grouped_layers,
            )
        )
        for scope, scope_name, scope_layers in scopes:
            for candidate in candidates:
                for multiplier in step_multipliers:
                    if multiplier == 0:
                        loss = baseline
                    else:
                        apply_shadow(
                            weights,
                            originals,
                            updates,
                            scope_layers,
                            candidate,
                            multiplier,
                            step_hyperparameters,
                        )
                        loss = evaluate_loss(model, heldout_batch, snapshot)
                        restore_weights(weights, originals)
                    delta = loss - baseline
                    scope_update_lrs = [
                        step_hyperparameters[layer][
                            "effective_update_learning_rate"
                        ]
                        for layer in scope_layers
                    ]
                    scope_decay = [
                        step_hyperparameters[layer][
                            "effective_decay_coefficient"
                        ]
                        for layer in scope_layers
                    ]
                    loss_rows.append(
                        {
                            "family": family,
                            "repeat": repeat,
                            "direction": direction,
                            "build_split": build_split,
                            "eval_split": eval_split,
                            "scope": scope,
                            "scope_name": scope_name,
                            "layer": scope_layers[0] if scope == "layer" else "",
                            "candidate": candidate,
                            "step_multiplier": multiplier,
                            "matrix_initial_learning_rate": (
                                optimizer_hyperparameters[
                                    "initial_learning_rate"
                                ]
                            ),
                            "matrix_scheduled_learning_rate": (
                                optimizer_hyperparameters[
                                    "scheduled_learning_rate"
                                ]
                            ),
                            "weight_decay": optimizer_hyperparameters[
                                "weight_decay"
                            ],
                            "effective_step_size_min": (
                                multiplier * min(scope_update_lrs)
                            ),
                            "effective_step_size_max": (
                                multiplier * max(scope_update_lrs)
                            ),
                            "effective_decay_coefficient_min": (
                                multiplier * min(scope_decay)
                            ),
                            "effective_decay_coefficient_max": (
                                multiplier * max(scope_decay)
                            ),
                            "evaluation_batches": evaluation_batches,
                            "baseline_loss": baseline,
                            "shadow_loss": loss,
                            "loss_delta": delta,
                            "relative_loss_delta": delta / max(abs(baseline), 1e-30),
                        }
                    )
    finally:
        restore_weights(weights, originals)
    summary: list[dict[str, Any]] = []
    keys = sorted(
        {
            (
                row["scope"],
                row["scope_name"],
                row["layer"],
                row["candidate"],
            )
            for row in loss_rows
        }
    )
    for scope, scope_name, layer, candidate in keys:
        candidates_rows = [
            row
            for row in loss_rows
            if (
                row["scope"],
                row["scope_name"],
                row["layer"],
                row["candidate"],
            )
            == (scope, scope_name, layer, candidate)
        ]
        best = min(
            candidates_rows,
            key=lambda row: (float(row["shadow_loss"]), float(row["step_multiplier"])),
        )
        summary.append(
            {
                "family": family,
                "repeat": repeat,
                "direction": direction,
                "build_split": build_split,
                "eval_split": eval_split,
                "scope": scope,
                "scope_name": scope_name,
                "layer": layer,
                "candidate": candidate,
                "evaluation_batches": evaluation_batches,
                "baseline_loss": baseline,
                "best_loss": best["shadow_loss"],
                "best_loss_delta": best["loss_delta"],
                "best_relative_loss_delta": best["relative_loss_delta"],
                "best_step_multiplier": best["step_multiplier"],
                "best_effective_step_size_min": best[
                    "effective_step_size_min"
                ],
                "best_effective_step_size_max": best[
                    "effective_step_size_max"
                ],
            }
        )
    return loss_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def model_content_unchanged(
    before: dict[str, Any], after: dict[str, Any]
) -> bool:
    """Compare persistent tensor content, excluding autograd version counters.

    Reversible in-place shadow perturbations necessarily increment
    ``Parameter._version`` even after the original bytes are copied back.
    MECH-01 includes that counter in ``parameter_rows_sha256`` for detecting
    accidental in-place work. MECH-03 intentionally performs such work, so its
    invariant is exact tensor/state_dict content, target hashes, and schema.
    """
    keys = (
        "full_parameter_sha256",
        "full_state_dict_sha256",
        "state_rows_sha256",
        "target_tensor_sha256",
        "parameter_count",
        "state_tensor_count",
    )
    return all(before.get(key) == after.get(key) for key in keys)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MECH-03 requires CUDA")
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
    prediction = validate_prediction_contract(args)
    M1.atomic_json(
        output / "prediction_contract_audit.json",
        {key: value for key, value in prediction.items() if key != "payload"},
    )
    if not prediction["passed"]:
        raise RuntimeError(f"prediction contract rejected: {prediction}")
    shutil.copyfile(
        args.prediction_contract.resolve(), output / "prediction_contract.json"
    )
    smoke_gate = validate_smoke_gate(args, prediction["sha256"])
    M1.atomic_json(output / "smoke_gate.json", smoke_gate)
    if not smoke_gate["passed"]:
        raise RuntimeError(f"MECH-03 smoke gate rejected: {smoke_gate}")
    mech02_certificate = validate_mech02_certificate(args)
    M1.atomic_json(output / "mech02_certificate.json", mech02_certificate)
    if not mech02_certificate["passed"]:
        raise RuntimeError(f"MECH-02 certificate rejected: {mech02_certificate}")
    mech01_certificate = M2.validate_mech01_certificate(args)
    M1.atomic_json(output / "mech01_certificate.json", mech01_certificate)
    if not mech01_certificate["passed"]:
        raise RuntimeError(f"MECH-01 certificate rejected: {mech01_certificate}")

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
    layers = M1.select_layers(schema["architecture"]["n_layer"], args.layers)
    source_runtime, production_ns, triton_audit = M1.load_source_runtime(
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
    modules, weights, target_names = M1.target_modules_and_weights(
        model, args.family, layers
    )
    component_modules: dict[tuple[int, str], nn.Module] = {}
    if args.family == "llama124":
        for layer in layers:
            mlp = model.layers[layer].mlp
            component_modules[(layer, "gate_proj_output")] = mlp.gate_proj
            component_modules[(layer, "up_proj_output")] = mlp.up_proj
    momenta, momentum_audit = M1.extract_target_momenta(
        checkpoint, model, args.family, target_names
    )
    optimizer_hyperparameters = matrix_optimizer_hyperparameters(
        checkpoint, weights
    )
    if not optimizer_hyperparameters["passed"]:
        raise RuntimeError(
            f"invalid matrix optimizer hyperparameters: {optimizer_hyperparameters}"
        )
    model_before = M1.model_state_signature(model, target_names.values())
    aux_before = M1.checkpoint_aux_signature(checkpoint)
    batches, batch_contract = read_crossfit_batches(args)
    model.cuda()

    split_builds: dict[int, dict[str, dict[str, Any]]] = {}
    for repeat in range(args.repeats):
        split_builds[repeat] = {}
        for split in SPLITS:
            split_builds[repeat][split] = collect_build_split(
                model,
                modules,
                weights,
                component_modules,
                batches[repeat][split],
                args.max_activation_rows,
            )

    component_rows, component_stability_rows = component_distribution_rows(
        split_builds, args.family
    )
    update_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    candidates = [str(value) for value in prediction["candidates"]]
    for repeat in range(args.repeats):
        for direction, build_split, eval_split in DIRECTIONS:
            build = split_builds[repeat][build_split]
            updates, geometry = candidate_updates(
                build["activations"],
                build["gradients"],
                momenta,
                args.family,
                candidates,
                args.ridge_mult,
                args.ridge_eps,
                args.momentum,
                args.ns_steps,
                production_ns,
                repeat,
                direction,
                build_split,
            )
            update_rows.extend(geometry)
            losses, summaries = line_search(
                model,
                weights,
                updates,
                batches[repeat][eval_split],
                candidates,
                args.step_multipliers,
                optimizer_hyperparameters,
                args.family,
                repeat,
                direction,
                build_split,
                eval_split,
                args.batches_per_split,
            )
            loss_rows.extend(losses)
            summary_rows.extend(summaries)
            del updates, geometry, losses, summaries
            torch.cuda.empty_cache()

    model_after = M1.model_state_signature(model, target_names.values())
    aux_after = M1.checkpoint_aux_signature(checkpoint)
    after_stat = checkpoint_path.stat()
    content_unchanged = model_content_unchanged(model_before, model_after)
    invariance = {
        "model_unchanged": content_unchanged,
        "model_content_unchanged": content_unchanged,
        "raw_model_signature_equal": model_before == model_after,
        "parameter_rows_sha256_equal": (
            model_before.get("parameter_rows_sha256")
            == model_after.get("parameter_rows_sha256")
        ),
        "parameter_version_counter_change_expected": (
            model_before.get("parameter_rows_sha256")
            != model_after.get("parameter_rows_sha256")
        ),
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
    expected_directions = args.repeats * len(DIRECTIONS)
    expected_updates = expected_directions * len(layers) * len(candidates)
    expected_summary = expected_directions * (len(layers) + 1) * len(candidates)
    expected_losses = expected_summary * len(args.step_multipliers)
    checks = {
        "prediction_contract": prediction["passed"],
        "prediction_contract_copy_exact": (
            M1.sha256_file(output / "prediction_contract.json")
            == prediction["sha256"]
        ),
        "smoke_gate": smoke_gate["passed"],
        "mech02_certificate": mech02_certificate["passed"],
        "mech01_certificate": mech01_certificate["passed"],
        "checkpoint_schema": schema["passed"],
        "source_runtime": source_config["passed"],
        "triton_provenance": triton_audit["passed"],
        "all_probe_windows_disjoint": batch_contract["all_windows_disjoint"],
        "all_layers_covered": sorted(modules) == layers,
        "historical_momentum_present": momentum_audit["all_present"],
        "matrix_optimizer_hyperparameters": optimizer_hyperparameters["passed"],
        "update_row_count": len(update_rows) == expected_updates,
        "loss_row_count": len(loss_rows) == expected_losses,
        "summary_row_count": len(summary_rows) == expected_summary,
        "update_geometry_finite": M1.finite_numbers(update_rows),
        "shadow_losses_finite": M1.finite_numbers(loss_rows),
        "line_search_summary_finite": M1.finite_numbers(summary_rows),
        "swiglu_component_coverage": (
            args.family != "llama124"
            or (
                len(component_rows) == args.repeats * len(SPLITS) * len(layers) * 3
                and len(component_stability_rows)
                == args.repeats * len(layers) * 3
            )
        ),
        "swiglu_components_finite": (
            args.family != "llama124"
            or (
                M1.finite_numbers(component_rows)
                and M1.finite_numbers(component_stability_rows)
            )
        ),
        "all_updates_finite": all(row["update_finite"] for row in update_rows),
        "inverse_health": all(
            int(row.get("cholesky_info_max", 0)) == 0 for row in update_rows
        ),
        "model_unchanged": invariance["model_unchanged"],
        "optimizer_loader_unchanged": invariance["optimizer_loader_unchanged"],
        "checkpoint_file_unchanged": invariance["checkpoint_file_unchanged"],
    }
    passed = all(checks.values())
    M1.atomic_json(output / "checkpoint_schema.json", schema)
    M1.atomic_json(output / "batch_contract.json", batch_contract)
    M1.atomic_json(output / "source_runtime_config.json", source_config)
    M1.atomic_json(output / "runtime.json", M1.runtime_metadata(args))
    M1.atomic_json(output / "momentum_audit.json", momentum_audit)
    M1.atomic_json(
        output / "matrix_optimizer_hyperparameters.json",
        optimizer_hyperparameters,
    )
    M1.atomic_json(output / "state_invariance.json", invariance)
    M1.atomic_json(output / "checks.json", checks)
    write_csv(output / "update_geometry.csv", update_rows)
    write_csv(output / "shadow_losses.csv", loss_rows)
    write_csv(output / "line_search_summary.csv", summary_rows)
    if component_rows:
        write_csv(output / "swiglu_components.csv", component_rows)
        write_csv(
            output / "swiglu_component_stability.csv",
            component_stability_rows,
        )
    M1.atomic_json(
        output / "mech03_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "stage": "crossfit_shadow_update",
            "analysis_tier": args.analysis_tier,
            "passed": passed,
            "family": args.family,
            "method": args.method,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": args.checkpoint_sha256.lower(),
            "prediction_contract_sha256": prediction["sha256"],
            "layers": layers,
            "repeats": args.repeats,
            "batches_per_split": args.batches_per_split,
            "candidates": candidates,
            "update_rows": len(update_rows),
            "shadow_loss_rows": len(loss_rows),
            "line_search_summary_rows": len(summary_rows),
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
