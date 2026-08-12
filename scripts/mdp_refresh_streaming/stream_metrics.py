#!/usr/bin/env python3
"""GPU-streamed metrics for one full-K refresh layer.

The module never writes full matrices.  Callers process one layer at a time,
persist scalars and optional pre-registered small slices, then release tensors.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable

import torch
from torch import Tensor


EPS = 1.0e-30


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(int(base)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def tensor_fingerprint(tensor: Tensor, samples: int = 17) -> dict[str, Any]:
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
        selected = flat.index_select(
            0, torch.tensor(indices, device=flat.device, dtype=torch.long)
        ).float()
        values = [float(value) for value in selected.cpu().tolist()]
    else:
        values = []
    payload = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "indices": indices,
        "values": values,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        **payload,
        "sampled_values_finite": all(math.isfinite(value) for value in values),
        "fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _norm(tensor: Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor, dtype=torch.float64).item())


def _dot(left: Tensor, right: Tensor) -> float:
    return float(torch.sum(left.double() * right.double()).item())


def _change_and_cosine(before: Tensor, after: Tensor) -> tuple[float, float]:
    before_norm = _norm(before)
    after_norm = _norm(after)
    change = _norm(after - before) / max(before_norm, EPS)
    cosine = _dot(before, after) / max(before_norm * after_norm, EPS)
    return change, max(-1.0, min(1.0, cosine))


def _asymmetry(matrix: Tensor) -> float:
    return _norm(matrix - matrix.T) / max(_norm(matrix), EPS)


def _rademacher(
    rows: int, columns: int, seed: int, device: torch.device
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values = torch.randint(
        0,
        2,
        (rows, columns),
        generator=generator,
        device="cpu",
        dtype=torch.int8,
    )
    return values.to(device=device, dtype=torch.float32).mul_(2).sub_(1)


def _normalize_columns(matrix: Tensor) -> Tensor:
    denominator = torch.linalg.vector_norm(
        matrix, dim=0, dtype=torch.float64
    ).to(matrix.dtype)
    return matrix / denominator.clamp_min(1.0e-20)


def _power_proxy(
    apply: Callable[[Tensor], Tensor],
    *,
    width: int,
    probes: int,
    iterations: int,
    seed: int,
    device: torch.device,
) -> float:
    vector = _normalize_columns(_rademacher(width, probes, seed, device))
    for _ in range(iterations):
        vector = _normalize_columns(apply(vector))
    applied = apply(vector)
    rayleigh = torch.sum(vector.double() * applied.double(), dim=0).abs()
    return float(torch.max(rayleigh).item())


def _a_apply(covariance: Tensor, ridge: float, value: Tensor) -> Tensor:
    return torch.mm(covariance, value).add(value, alpha=float(ridge))


def _delta_a_apply(
    before: Tensor,
    after: Tensor,
    ridge_delta: float,
    value: Tensor,
) -> Tensor:
    return torch.mm(after - before, value).add(
        value, alpha=float(ridge_delta)
    )


def _matrix_fro_with_ridge(covariance: Tensor, ridge: float) -> float:
    covariance_norm = _norm(covariance)
    trace = float(torch.trace(covariance).double().item())
    width = covariance.shape[0]
    square = (
        covariance_norm * covariance_norm
        + 2.0 * float(ridge) * trace
        + width * float(ridge) * float(ridge)
    )
    return math.sqrt(max(square, 0.0))


def _delta_a_fro(
    before: Tensor, after: Tensor, ridge_before: float, ridge_after: float
) -> float:
    delta = after - before
    delta_norm = _norm(delta)
    delta_trace = float(torch.trace(delta).double().item())
    ridge_delta = float(ridge_after) - float(ridge_before)
    width = before.shape[0]
    square = (
        delta_norm * delta_norm
        + 2.0 * ridge_delta * delta_trace
        + width * ridge_delta * ridge_delta
    )
    return math.sqrt(max(square, 0.0))


def _backward_inverse_residual(
    covariance: Tensor,
    ridge: float,
    inverse: Tensor,
    probes: Tensor,
    a_norm_proxy: float,
) -> float:
    inverse_probe = torch.mm(inverse, probes)
    residual = _a_apply(covariance, ridge, inverse_probe) - probes
    denominator = (
        float(a_norm_proxy) * _norm(inverse_probe) + _norm(probes)
    )
    return _norm(residual) / max(denominator, EPS)


def _slice_indices(width: int, count: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(width, generator=generator)[:count].sort().values


@torch.no_grad()
def compute_layer_metrics(
    *,
    covariance_before: Tensor,
    covariance_after: Tensor,
    inverse_before: Tensor,
    inverse_after: Tensor,
    fresh_covariance: Tensor,
    raw_gradient: Tensor,
    historical_momentum: Tensor,
    input_beta: float,
    ridge_scale: float,
    ridge_epsilon: float,
    momentum_beta: float,
    ns_steps: int,
    ns_update: Callable[[Tensor, int], Tensor],
    probe_count: int,
    probe_iterations: int,
    probe_seed: int,
    slice_coordinate_count: int = 0,
    slice_gradient_row_count: int = 0,
    slice_seed: int = 0,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Return scalars, shadow fingerprints, and an optional small slice."""
    width = covariance_before.shape[0]
    if covariance_before.shape != (width, width):
        raise ValueError("covariance_before must be square")
    expected_shapes = {
        "covariance_after": covariance_before.shape,
        "inverse_before": covariance_before.shape,
        "inverse_after": covariance_before.shape,
        "fresh_covariance": covariance_before.shape,
    }
    local_tensors = {
        "covariance_after": covariance_after,
        "inverse_before": inverse_before,
        "inverse_after": inverse_after,
        "fresh_covariance": fresh_covariance,
    }
    for name, expected in expected_shapes.items():
        if local_tensors[name].shape != expected:
            raise ValueError(f"{name} has shape {local_tensors[name].shape}, expected {expected}")
    if raw_gradient.shape[1] != width:
        raise ValueError("raw gradient width does not match covariance")
    if historical_momentum.shape != raw_gradient.shape:
        raise ValueError("historical momentum shape does not match gradient")

    ridge_before = (
        float(covariance_before.diagonal().mean().item()) * ridge_scale
        + ridge_epsilon
    )
    ridge_after = (
        float(covariance_after.diagonal().mean().item()) * ridge_scale
        + ridge_epsilon
    )
    expected_after = covariance_before.mul(input_beta).add(
        fresh_covariance, alpha=1.0 - input_beta
    )
    refresh_identity = _norm(covariance_after - expected_after) / max(
        _norm(covariance_after), EPS
    )
    relative_k = _norm(covariance_after - covariance_before) / max(
        _norm(covariance_before), EPS
    )
    relative_a = _delta_a_fro(
        covariance_before, covariance_after, ridge_before, ridge_after
    ) / max(_matrix_fro_with_ridge(covariance_before, ridge_before), EPS)
    relative_inverse, inverse_cosine = _change_and_cosine(
        inverse_before, inverse_after
    )

    half_probes = max(1, probe_count // 2)
    probe_bank_seeds = [
        stable_seed(probe_seed, "bank", bank) for bank in range(2)
    ]
    bank_rows: list[dict[str, float]] = []
    for bank, bank_seed in enumerate(probe_bank_seeds):
        probes = _rademacher(width, half_probes, bank_seed, covariance_before.device)
        a_before_proxy = _power_proxy(
            lambda value: _a_apply(covariance_before, ridge_before, value),
            width=width,
            probes=half_probes,
            iterations=probe_iterations,
            seed=stable_seed(bank_seed, "a_before"),
            device=covariance_before.device,
        )
        a_after_proxy = _power_proxy(
            lambda value: _a_apply(covariance_after, ridge_after, value),
            width=width,
            probes=half_probes,
            iterations=probe_iterations,
            seed=stable_seed(bank_seed, "a_after"),
            device=covariance_before.device,
        )
        inverse_before_proxy = _power_proxy(
            lambda value: torch.mm(inverse_before, value),
            width=width,
            probes=half_probes,
            iterations=probe_iterations,
            seed=stable_seed(bank_seed, "inverse_before"),
            device=covariance_before.device,
        )
        inverse_after_proxy = _power_proxy(
            lambda value: torch.mm(inverse_after, value),
            width=width,
            probes=half_probes,
            iterations=probe_iterations,
            seed=stable_seed(bank_seed, "inverse_after"),
            device=covariance_before.device,
        )
        delta_a_proxy = _power_proxy(
            lambda value: _delta_a_apply(
                covariance_before,
                covariance_after,
                ridge_after - ridge_before,
                value,
            ),
            width=width,
            probes=half_probes,
            iterations=probe_iterations,
            seed=stable_seed(bank_seed, "delta_a"),
            device=covariance_before.device,
        )
        delta_inverse_proxy = _power_proxy(
            lambda value: torch.mm(inverse_after - inverse_before, value),
            width=width,
            probes=half_probes,
            iterations=probe_iterations,
            seed=stable_seed(bank_seed, "delta_inverse"),
            device=covariance_before.device,
        )
        before_probe = torch.mm(inverse_before, probes)
        lhs = torch.mm(inverse_after - inverse_before, probes)
        delta_on_before = _delta_a_apply(
            covariance_before,
            covariance_after,
            ridge_after - ridge_before,
            before_probe,
        )
        rhs = -torch.mm(inverse_after, delta_on_before)
        resolvent = _norm(lhs - rhs) / max(_norm(lhs) + _norm(rhs), EPS)
        bank_rows.append(
            {
                "bank": float(bank),
                "a_before_proxy": a_before_proxy,
                "a_after_proxy": a_after_proxy,
                "inverse_before_proxy": inverse_before_proxy,
                "inverse_after_proxy": inverse_after_proxy,
                "delta_a_proxy": delta_a_proxy,
                "delta_inverse_proxy": delta_inverse_proxy,
                "condition_before_proxy": a_before_proxy * inverse_before_proxy,
                "condition_after_proxy": a_after_proxy * inverse_after_proxy,
                "inverse_residual_before": _backward_inverse_residual(
                    covariance_before,
                    ridge_before,
                    inverse_before,
                    probes,
                    a_before_proxy,
                ),
                "inverse_residual_after": _backward_inverse_residual(
                    covariance_after,
                    ridge_after,
                    inverse_after,
                    probes,
                    a_after_proxy,
                ),
                "resolvent_residual": resolvent,
                "bound_ratio_proxy": delta_inverse_proxy
                / max(
                    inverse_after_proxy
                    * delta_a_proxy
                    * inverse_before_proxy,
                    EPS,
                ),
            }
        )
    probe_keys = [key for key in bank_rows[0] if key != "bank"]
    probe_means = {
        key: sum(row[key] for row in bank_rows) / len(bank_rows)
        for key in probe_keys
    }
    condition_disagreements = []
    for key in ("condition_before_proxy", "condition_after_proxy"):
        first, second = bank_rows[0][key], bank_rows[1][key]
        condition_disagreements.append(
            abs(first - second) / max((abs(first) + abs(second)) / 2.0, EPS)
        )

    gradient_before = torch.mm(raw_gradient.float(), inverse_before.float())
    gradient_after = torch.mm(raw_gradient.float(), inverse_after.float())
    gradient_change, gradient_cosine = _change_and_cosine(
        gradient_before, gradient_after
    )

    momentum_before_path = historical_momentum.detach().float().clone()
    momentum_before_path.lerp_(gradient_before, 1.0 - momentum_beta)
    ns_input_before = torch.lerp(
        gradient_before, momentum_before_path, momentum_beta
    )
    update_before = ns_update(ns_input_before, ns_steps)
    momentum_after_path = historical_momentum.detach().float().clone()
    momentum_after_path.lerp_(gradient_after, 1.0 - momentum_beta)
    ns_input_after = torch.lerp(
        gradient_after, momentum_after_path, momentum_beta
    )
    update_after = ns_update(ns_input_after, ns_steps)
    update_change, update_cosine = _change_and_cosine(
        update_before, update_after
    )

    finite_checks = {
        name: bool(torch.isfinite(tensor).all().item())
        for name, tensor in {
            "covariance_before": covariance_before,
            "covariance_after": covariance_after,
            "inverse_before": inverse_before,
            "inverse_after": inverse_after,
            "fresh_covariance": fresh_covariance,
            "raw_gradient": raw_gradient,
            "historical_momentum": historical_momentum,
            "gradient_before": gradient_before,
            "gradient_after": gradient_after,
            "update_before": update_before,
            "update_after": update_after,
        }.items()
    }
    metrics: dict[str, Any] = {
        "k_rows": int(width),
        "k_columns": int(width),
        "gradient_rows": int(raw_gradient.shape[0]),
        "gradient_columns": int(raw_gradient.shape[1]),
        "k_dtype": str(covariance_before.dtype),
        "inverse_dtype": str(inverse_before.dtype),
        "gradient_dtype": str(raw_gradient.dtype),
        "ridge_before": ridge_before,
        "ridge_after": ridge_after,
        "k_trace_before": float(torch.trace(covariance_before).double().item()),
        "k_trace_after": float(torch.trace(covariance_after).double().item()),
        "k_fro_before": _norm(covariance_before),
        "k_fro_after": _norm(covariance_after),
        "inverse_fro_before": _norm(inverse_before),
        "inverse_fro_after": _norm(inverse_after),
        "raw_gradient_fro": _norm(raw_gradient),
        "historical_momentum_fro": _norm(historical_momentum),
        "relative_k_fro_change": relative_k,
        "relative_a_fro_change": relative_a,
        "relative_runtime_inverse_fro_change": relative_inverse,
        "runtime_inverse_cosine": inverse_cosine,
        "k_asymmetry_before": _asymmetry(covariance_before),
        "k_asymmetry_after": _asymmetry(covariance_after),
        "inverse_asymmetry_before": _asymmetry(inverse_before),
        "inverse_asymmetry_after": _asymmetry(inverse_after),
        "covariance_refresh_identity_relative_residual": refresh_identity,
        "matched_g_preconditioned_fro_before": _norm(gradient_before),
        "matched_g_preconditioned_fro_after": _norm(gradient_after),
        "matched_g_preconditioned_delta_fro": _norm(
            gradient_after - gradient_before
        ),
        "matched_g_preconditioned_relative_change": gradient_change,
        "matched_g_preconditioned_cosine": gradient_cosine,
        "runtime_ns5_update_fro_before": _norm(update_before),
        "runtime_ns5_update_fro_after": _norm(update_after),
        "runtime_ns5_update_delta_fro": _norm(update_after - update_before),
        "runtime_ns5_update_relative_change": update_change,
        "runtime_ns5_update_cosine": update_cosine,
        "condition_proxy_before": probe_means["condition_before_proxy"],
        "condition_proxy_after": probe_means["condition_after_proxy"],
        "runtime_inverse_backward_residual_before": probe_means[
            "inverse_residual_before"
        ],
        "runtime_inverse_backward_residual_after": probe_means[
            "inverse_residual_after"
        ],
        "runtime_resolvent_relative_residual": probe_means[
            "resolvent_residual"
        ],
        "spectral_resolvent_bound_ratio_proxy": probe_means[
            "bound_ratio_proxy"
        ],
        "probe_bank_condition_relative_disagreement": max(
            condition_disagreements
        ),
        "probe_count": int(probe_count),
        "probe_iterations": int(probe_iterations),
        "probe_seed": int(probe_seed),
        "all_full_state_values_finite": all(finite_checks.values()),
    }
    for index, bank in enumerate(bank_rows):
        for key, value in bank.items():
            if key != "bank":
                metrics[f"probe_bank_{index}_{key}"] = value

    fingerprints = {
        "raw_gradient": tensor_fingerprint(raw_gradient),
        "historical_momentum": tensor_fingerprint(historical_momentum),
        "gradient_after": tensor_fingerprint(gradient_after),
        "ns_input_after": tensor_fingerprint(ns_input_after),
        "ns_output_after": tensor_fingerprint(update_after),
    }

    slice_payload: dict[str, Any] | None = None
    if slice_coordinate_count > 0 and slice_gradient_row_count > 0:
        column_indices = _slice_indices(
            width, slice_coordinate_count, stable_seed(slice_seed, "columns")
        )
        row_indices = _slice_indices(
            raw_gradient.shape[0],
            slice_gradient_row_count,
            stable_seed(slice_seed, "rows"),
        )
        device_columns = column_indices.to(covariance_before.device)
        device_rows = row_indices.to(raw_gradient.device)

        def square_slice(value: Tensor) -> Tensor:
            return value.index_select(0, device_columns).index_select(
                1, device_columns
            )

        def gradient_slice(value: Tensor) -> Tensor:
            return value.index_select(0, device_rows).index_select(
                1, device_columns
            )

        slice_payload = {
            "column_indices": column_indices.numpy(),
            "gradient_row_indices": row_indices.numpy(),
            "covariance_before": square_slice(covariance_before).float().cpu().numpy(),
            "covariance_after": square_slice(covariance_after).float().cpu().numpy(),
            "runtime_inverse_before": square_slice(inverse_before).float().cpu().numpy(),
            "runtime_inverse_after": square_slice(inverse_after).float().cpu().numpy(),
            "raw_gradient": gradient_slice(raw_gradient).float().cpu().numpy(),
            "historical_momentum": gradient_slice(historical_momentum).float().cpu().numpy(),
            "matched_gradient_before": gradient_slice(gradient_before).float().cpu().numpy(),
            "matched_gradient_after": gradient_slice(gradient_after).float().cpu().numpy(),
            "runtime_ns5_update_before": gradient_slice(update_before).float().cpu().numpy(),
            "runtime_ns5_update_after": gradient_slice(update_after).float().cpu().numpy(),
        }
    return metrics, fingerprints, slice_payload

