#!/usr/bin/env python3
"""Pure directional-geometry kernels for experiment 47 / GEO-01.

The module deliberately separates the scientific calculation from checkpoint,
optimizer and controller plumbing.  Unit tests use a toy model; the remote
worker will provide source-pinned Newton--Muon counterfactual directions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
import hashlib
import math
from typing import Any

import torch
from torch import Tensor, nn


EPS = 1.0e-30


def tensor_sha256(value: Tensor) -> str:
    """Hash logical tensor values without persisting the tensor."""

    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def tensor_fingerprint(value: Tensor) -> dict[str, Any]:
    detached = value.detach()
    finite = bool(torch.isfinite(detached).all().item())
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "fro_norm": float(torch.linalg.vector_norm(detached.float()).item()),
        "finite": finite,
        "sha256": tensor_sha256(detached),
    }


def _dot(left: Sequence[Tensor], right: Sequence[Tensor]) -> Tensor:
    if len(left) != len(right) or not left:
        raise ValueError("directional dot requires non-empty equal-length inputs")
    terms = [(x * y.to(dtype=x.dtype)).sum() for x, y in zip(left, right)]
    return torch.stack(terms).sum()


def _norm(values: Sequence[Tensor]) -> float:
    if not values:
        raise ValueError("norm requires at least one tensor")
    total = sum(float(torch.sum(value.detach().float().square()).item()) for value in values)
    return math.sqrt(max(total, 0.0))


def relative_error(observed: float, expected: float, floor: float = 1.0e-12) -> float:
    return abs(float(observed) - float(expected)) / max(
        abs(float(observed)), abs(float(expected)), float(floor)
    )


def counterfactual_update_direction(
    *,
    raw_gradient: Tensor,
    historical_momentum: Tensor,
    inverse_reference: Tensor,
    inverse_treatment: Tensor,
    momentum_beta: float,
    learning_rate: float,
    ns_steps: int,
    ns_update: Callable[[Tensor, int], Tensor],
    fingerprint_fn: Callable[[Tensor], dict[str, Any]] = tensor_fingerprint,
) -> tuple[Tensor, dict[str, Any]]:
    """Return the exact treatment-minus-reference parameter update.

    This mirrors the source optimizer's EMA momentum, Nesterov lookahead,
    source-pinned NS5 call and rectangular shape scaling.  Both paths use the
    identical raw gradient and historical momentum.
    """

    if raw_gradient.ndim != 2:
        raise ValueError("GEO-01 currently supports matrix parameters only")
    rows, columns = raw_gradient.shape
    expected_inverse = (columns, columns)
    if tuple(inverse_reference.shape) != expected_inverse:
        raise ValueError("reference inverse shape does not match gradient")
    if tuple(inverse_treatment.shape) != expected_inverse:
        raise ValueError("treatment inverse shape does not match gradient")
    if tuple(historical_momentum.shape) != tuple(raw_gradient.shape):
        raise ValueError("historical momentum shape does not match gradient")
    if not 0.0 <= float(momentum_beta) < 1.0:
        raise ValueError("momentum_beta must lie in [0, 1)")
    if learning_rate <= 0.0 or ns_steps <= 0:
        raise ValueError("learning rate and NS steps must be positive")

    gradient_reference = torch.mm(raw_gradient.float(), inverse_reference.float())
    gradient_treatment = torch.mm(raw_gradient.float(), inverse_treatment.float())

    def path(preconditioned: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        momentum = historical_momentum.detach().float().clone()
        momentum.lerp_(preconditioned, 1.0 - float(momentum_beta))
        ns_input = torch.lerp(preconditioned, momentum, float(momentum_beta))
        update = ns_update(ns_input, int(ns_steps)).float()
        return momentum, ns_input, update

    _, ns_input_reference, update_reference = path(gradient_reference)
    _, ns_input_treatment, update_treatment = path(gradient_treatment)
    shape_scale = math.sqrt(max(1.0, float(rows) / float(columns)))
    direction = (
        -float(learning_rate)
        * float(shape_scale)
        * (update_treatment - update_reference)
    )
    audit = {
        "shape_scale": shape_scale,
        "learning_rate": float(learning_rate),
        "momentum_beta": float(momentum_beta),
        "ns_steps": int(ns_steps),
        "gradient_reference": fingerprint_fn(gradient_reference),
        "gradient_treatment": fingerprint_fn(gradient_treatment),
        "ns_input_reference": fingerprint_fn(ns_input_reference),
        "ns_input_treatment": fingerprint_fn(ns_input_treatment),
        "update_reference": fingerprint_fn(update_reference),
        "update_treatment": fingerprint_fn(update_treatment),
        "parameter_direction": fingerprint_fn(direction),
        "finite": bool(torch.isfinite(direction).all().item()),
        "nonzero": bool(torch.count_nonzero(direction).item() > 0),
    }
    return direction, audit


def _extract_loss(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        if output.ndim != 0:
            raise TypeError("model returned a non-scalar tensor instead of loss")
        return output
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        loss = output[1]
        if isinstance(loss, Tensor) and loss.ndim == 0:
            return loss
    if isinstance(output, Mapping):
        loss = output.get("loss")
        if isinstance(loss, Tensor) and loss.ndim == 0:
            return loss
    raise TypeError("unable to extract a scalar loss from model output")


def _functional_mean_loss(
    model: nn.Module,
    batches: Sequence[tuple[Tensor, Tensor]],
    overrides: Mapping[str, Tensor],
    forward_kwargs: Mapping[str, Any],
) -> Tensor:
    state: dict[str, Tensor] = {
        **dict(model.named_parameters()),
        **dict(model.named_buffers()),
    }
    state.update(overrides)
    losses = []
    for x, y in batches:
        output = torch.func.functional_call(
            model,
            state,
            (x, y),
            dict(forward_kwargs),
            strict=False,
        )
        losses.append(_extract_loss(output))
    return torch.stack(losses).mean()


def _ordinary_mean_loss(
    model: nn.Module,
    batches: Sequence[tuple[Tensor, Tensor]],
    forward_kwargs: Mapping[str, Any],
) -> Tensor:
    losses = [
        _extract_loss(model(x, y, **dict(forward_kwargs))) for x, y in batches
    ]
    return torch.stack(losses).mean()


def validate_named_direction(
    model: nn.Module, named_direction: Mapping[str, Tensor]
) -> tuple[list[str], list[nn.Parameter], list[Tensor]]:
    if not named_direction:
        raise ValueError("named_direction must not be empty")
    parameters = dict(model.named_parameters())
    missing = sorted(set(named_direction) - set(parameters))
    if missing:
        raise KeyError(f"direction names are not model parameters: {missing}")
    names = sorted(named_direction)
    selected_parameters: list[nn.Parameter] = []
    selected_directions: list[Tensor] = []
    for name in names:
        parameter = parameters[name]
        direction = named_direction[name]
        if tuple(parameter.shape) != tuple(direction.shape):
            raise ValueError(f"direction shape mismatch for {name}")
        if direction.device != parameter.device:
            raise ValueError(f"direction device mismatch for {name}")
        if not torch.isfinite(direction).all():
            raise FloatingPointError(f"non-finite direction for {name}")
        selected_parameters.append(parameter)
        selected_directions.append(direction.detach().to(parameter.dtype))
    return names, selected_parameters, selected_directions


def measure_directional_geometry(
    *,
    model: nn.Module,
    batches: Sequence[tuple[Tensor, Tensor]],
    named_direction: Mapping[str, Tensor],
    forward_kwargs: Mapping[str, Any] | None = None,
    fd_target_relative_parameter_norm: float = 1.0e-4,
    fd_scale_min: float = 1.0,
    fd_scale_max: float = 64.0,
) -> dict[str, Any]:
    """Measure first order, directional HVP and exact line losses.

    The actual optimizer difference is always evaluated at multiplier 1.  A
    deterministic central-difference multiplier is chosen only for numerical
    calibration and is reported separately.
    """

    if not batches:
        raise ValueError("at least one held-out batch is required")
    if not 0.0 < fd_target_relative_parameter_norm < 1.0:
        raise ValueError("finite-difference target must lie in (0, 1)")
    if not 0.0 < fd_scale_min <= fd_scale_max:
        raise ValueError("invalid finite-difference scale bounds")
    kwargs = dict(forward_kwargs or {})
    names, parameters, directions = validate_named_direction(model, named_direction)
    parameter_hashes_before = {name: tensor_sha256(param) for name, param in zip(names, parameters)}
    parameter_norm = _norm(parameters)
    direction_norm = _norm(directions)
    if direction_norm <= 0.0:
        raise ValueError("counterfactual update direction is zero")
    relative_direction_norm = direction_norm / max(parameter_norm, EPS)
    raw_fd_scale = (
        float(fd_target_relative_parameter_norm)
        / max(relative_direction_norm, EPS)
    )
    fd_scale = min(max(raw_fd_scale, float(fd_scale_min)), float(fd_scale_max))

    attention = getattr(torch.nn, "attention", None)
    attention_context = (
        attention.sdpa_kernel([attention.SDPBackend.MATH])
        if attention is not None
        else nullcontext()
    )
    was_training = model.training
    model.eval()
    try:
        # Fused Flash/memory-efficient SDPA kernels do not provide a portable
        # second derivative.  Force the PyTorch math backend for both the HVP
        # graph and its line-loss calibration.
        with attention_context:
            with torch.enable_grad():
                loss = _ordinary_mean_loss(model, batches, kwargs)
                gradients = torch.autograd.grad(
                    loss,
                    parameters,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=False,
                )
                first_order_tensor = _dot(gradients, directions)
                hvp = torch.autograd.grad(
                    first_order_tensor,
                    parameters,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )
                curvature_tensor = _dot(hvp, directions)
                baseline_from_graph = float(loss.detach().double().item())
                first_order = float(first_order_tensor.detach().double().item())
                directional_curvature = float(curvature_tensor.detach().double().item())
            del gradients, hvp, loss, first_order_tensor, curvature_tensor

            def line_loss(alpha: float) -> float:
                overrides = {
                    name: parameter.detach() + float(alpha) * direction
                    for name, parameter, direction in zip(names, parameters, directions)
                }
                with torch.no_grad():
                    value = _functional_mean_loss(model, batches, overrides, kwargs)
                return float(value.detach().double().item())

            baseline = line_loss(0.0)
            actual_plus = line_loss(1.0)
            fd_plus = line_loss(fd_scale)
            fd_minus = line_loss(-fd_scale)
    finally:
        model.train(was_training)

    parameter_hashes_after = {name: tensor_sha256(param) for name, param in zip(names, parameters)}
    fd_first = (fd_plus - fd_minus) / (2.0 * fd_scale)
    fd_curvature = (fd_plus - 2.0 * baseline + fd_minus) / (fd_scale**2)
    exact_actual_delta = actual_plus - baseline
    second_order_term = 0.5 * directional_curvature
    taylor_actual_delta = first_order + second_order_term
    values = {
        "baseline_loss": baseline,
        "baseline_graph_loss": baseline_from_graph,
        "actual_plus_loss": actual_plus,
        "exact_actual_delta_loss": exact_actual_delta,
        "first_order_alignment": first_order,
        "directional_curvature": directional_curvature,
        "second_order_term": second_order_term,
        "taylor_actual_delta_loss": taylor_actual_delta,
        "taylor_residual": exact_actual_delta - taylor_actual_delta,
        "fd_scale": fd_scale,
        "fd_plus_loss": fd_plus,
        "fd_minus_loss": fd_minus,
        "fd_first_order": fd_first,
        "fd_directional_curvature": fd_curvature,
        "fd_first_relative_error": relative_error(fd_first, first_order),
        "fd_curvature_relative_error": relative_error(
            fd_curvature, directional_curvature
        ),
        "baseline_graph_relative_error": relative_error(
            baseline_from_graph, baseline
        ),
        "parameter_fro_norm": parameter_norm,
        "direction_fro_norm": direction_norm,
        "relative_direction_fro_norm": relative_direction_norm,
    }
    finite = all(math.isfinite(float(value)) for value in values.values())
    return {
        "schema_version": "geo01_directional_geometry_v1",
        "parameter_names": names,
        "parameter_count": len(names),
        "heldout_batches": len(batches),
        "attention_backend": "math_only_for_second_order",
        **values,
        "all_values_finite": finite,
        "parameters_unchanged": parameter_hashes_before == parameter_hashes_after,
        "parameter_sha256_before": parameter_hashes_before,
        "parameter_sha256_after": parameter_hashes_after,
    }
