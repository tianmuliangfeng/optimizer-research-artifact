"""Audited single-GPU Mousse-R1 optimizer adaptation.

This module transcribes the hidden-matrix update from the official Mousse
implementation pinned by experiment 45.  The only layout adaptation is the
controlled-R1 convention that a packed GPT-2 QKV weight is treated as three
logical matrices.  It is not an unchanged reproduction of the authors'
training scaffold.

Upstream: https://github.com/Anti-Entrophic/Mousse
Pinned commit: d00c1bf17790fbe56424ee5567cce80d8e75f4b2
License: MIT (see THIRD_PARTY_NOTICES.md)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import torch
from torch import Tensor, nn


UPSTREAM_COMMIT = "d00c1bf17790fbe56424ee5567cce80d8e75f4b2"
NS_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def logical_matrix_slices(matrix: Tensor) -> tuple[Tensor, ...]:
    """Apply the R1 packed-QKV contract and otherwise keep a matrix intact."""
    if matrix.ndim != 2:
        raise ValueError("Mousse-R1 accepts hidden 2D matrices only")
    rows, cols = matrix.shape
    if rows == 3 * cols:
        return tuple(matrix.split(cols, dim=0))
    return (matrix,)


def clean_eigenvalues(eigenvalues: Tensor, epsilon: float) -> Tensor:
    """Exact Mousse eigenvalue shift: max(0, -lambda_min) + epsilon."""
    shift = torch.clamp(-eigenvalues.min(), min=0.0) + float(epsilon)
    return eigenvalues + shift


def zeropower_via_newtonschulz5(matrix: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Official five-stage Mousse Newton--Schulz polynomial in BF16."""
    if matrix.ndim != 2:
        raise ValueError("Newton--Schulz expects one logical matrix")
    original_dtype = matrix.dtype
    x = matrix.to(dtype=torch.bfloat16)
    transposed = matrix.size(0) > matrix.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + float(epsilon))
    for a, b, c in NS_COEFFICIENTS:
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(dtype=original_dtype)


def _state_key(role: str, logical_index: int) -> str:
    return f"mousse_{role}_{logical_index}"


def _initialize_logical_state(
    state: dict[str, object],
    logical: Tensor,
    logical_index: int,
) -> None:
    rows, cols = logical.shape
    device = logical.device
    specs = {
        "factor_L": torch.zeros((rows, rows), device=device, dtype=torch.float32),
        "factor_R": torch.zeros((cols, cols), device=device, dtype=torch.float32),
        "eigval_L": torch.zeros(rows, device=device, dtype=torch.float32),
        "eigvec_L": torch.eye(rows, device=device, dtype=torch.float32),
        "eigval_R": torch.zeros(cols, device=device, dtype=torch.float32),
        "eigvec_R": torch.eye(cols, device=device, dtype=torch.float32),
    }
    for role, value in specs.items():
        state[_state_key(role, logical_index)] = value
    state[_state_key("refresh_count", logical_index)] = 0


def _logical_mousse_update(
    momentum_update: Tensor,
    gradient: Tensor,
    state: dict[str, object],
    logical_index: int,
    *,
    step: int,
    factor_beta: float,
    factor_epsilon: float,
    factor_alpha: float,
    refresh_interval: int,
    bias_correction: bool,
    grafting: bool,
    ns_epsilon: float,
) -> Tensor:
    """One logical matrix update, ordered exactly like official Mousse."""
    prefix = _state_key
    if prefix("factor_L", logical_index) not in state:
        _initialize_logical_state(state, gradient, logical_index)

    g = gradient.to(dtype=torch.float32)
    update = momentum_update.to(dtype=torch.float32)
    factor_l = state[prefix("factor_L", logical_index)]
    factor_r = state[prefix("factor_R", logical_index)]
    assert isinstance(factor_l, Tensor) and isinstance(factor_r, Tensor)

    # 1. Gradient-space Kronecker statistics, before momentum whitening.
    factor_l.mul_(factor_beta).add_(g @ g.T, alpha=1.0 - factor_beta)
    factor_r.mul_(factor_beta).add_(g.T @ g, alpha=1.0 - factor_beta)

    # The official group counter is incremented before this test: 1, 11, 21, ...
    if step % refresh_interval == 1 or refresh_interval == 1:
        correction = 1.0 - factor_beta**step if bias_correction else 1.0
        corrected_l = factor_l / correction
        corrected_r = factor_r / correction
        corrected_l = corrected_l * (corrected_l.size(0) / corrected_l.trace())
        corrected_r = corrected_r * (corrected_r.size(0) / corrected_r.trace())
        eye_l = torch.eye(corrected_l.size(0), device=g.device, dtype=torch.float32)
        eye_r = torch.eye(corrected_r.size(0), device=g.device, dtype=torch.float32)
        eigval_l, eigvec_l = torch.linalg.eigh(corrected_l + factor_epsilon * eye_l)
        eigval_r, eigvec_r = torch.linalg.eigh(corrected_r + factor_epsilon * eye_r)
        state[prefix("eigval_L", logical_index)] = clean_eigenvalues(eigval_l, factor_epsilon)
        state[prefix("eigvec_L", logical_index)] = eigvec_l
        state[prefix("eigval_R", logical_index)] = clean_eigenvalues(eigval_r, factor_epsilon)
        state[prefix("eigvec_R", logical_index)] = eigvec_r
        state[prefix("refresh_count", logical_index)] = int(
            state[prefix("refresh_count", logical_index)]
        ) + 1

    eigval_l = state[prefix("eigval_L", logical_index)]
    eigvec_l = state[prefix("eigvec_L", logical_index)]
    eigval_r = state[prefix("eigval_R", logical_index)]
    eigvec_r = state[prefix("eigvec_R", logical_index)]
    assert all(isinstance(value, Tensor) for value in (eigval_l, eigvec_l, eigval_r, eigvec_r))

    # 2. Whiten momentum, 3. spectral constraint, 4. unwhiten, 5. graft.
    scale_l = eigval_l.abs().pow(factor_alpha)
    scale_r = eigval_r.abs().pow(factor_alpha)
    update = eigvec_l.T @ update @ eigvec_r
    update = update / scale_l.unsqueeze(1) / scale_r.unsqueeze(0)
    update = zeropower_via_newtonschulz5(update, epsilon=ns_epsilon)
    target_norm = update.norm() if grafting else None
    update = update / scale_l.unsqueeze(1) / scale_r.unsqueeze(0)
    update = eigvec_l @ update @ eigvec_r.T
    if target_norm is not None:
        update = update * (target_norm / update.norm().clamp_min(1e-30))
    return update.to(dtype=momentum_update.dtype)


class R1Mousse(torch.optim.Optimizer):
    """Single-GPU, double-sided Mousse for R1 hidden matrices."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = False,
        factor_beta: float = 0.95,
        factor_epsilon: float = 1e-5,
        factor_alpha: float = 0.125,
        refresh_interval: int = 10,
        bias_correction: bool = True,
        grafting: bool = True,
        adjust_lr: str = "spectral_norm",
        ns_epsilon: float = 1e-8,
    ) -> None:
        if lr < 0 or weight_decay < 0:
            raise ValueError("learning rate and weight decay must be non-negative")
        if not 0 <= momentum < 1 or not 0 <= factor_beta < 1:
            raise ValueError("momentum and factor_beta must be in [0, 1)")
        if factor_epsilon <= 0 or ns_epsilon <= 0 or refresh_interval <= 0:
            raise ValueError("epsilons and refresh_interval must be positive")
        if adjust_lr != "spectral_norm":
            raise ValueError("experiment 45 freezes adjust_lr='spectral_norm'")
        defaults = dict(
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            factor_beta=float(factor_beta),
            factor_epsilon=float(factor_epsilon),
            factor_alpha=float(factor_alpha),
            refresh_interval=int(refresh_interval),
            bias_correction=bool(bias_correction),
            grafting=bool(grafting),
            adjust_lr=adjust_lr,
            ns_epsilon=float(ns_epsilon),
            step=0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            group["step"] += 1
            step = int(group["step"])
            base_lr = float(group["lr"])
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if parameter.ndim != 2:
                    raise ValueError("R1Mousse received a non-matrix parameter")
                state = self.state[parameter]
                momentum_buffer = state.get("momentum")
                if momentum_buffer is None:
                    momentum_buffer = state["momentum"] = torch.zeros_like(parameter)
                assert isinstance(momentum_buffer, Tensor)
                momentum_buffer.mul_(group["momentum"]).add_(gradient)
                if group["nesterov"]:
                    momentum_update = gradient.add(momentum_buffer, alpha=group["momentum"])
                else:
                    momentum_update = momentum_buffer

                update_chunks = []
                gradient_chunks = logical_matrix_slices(gradient)
                momentum_chunks = logical_matrix_slices(momentum_update)
                for logical_index, (logical_momentum, logical_gradient) in enumerate(
                    zip(momentum_chunks, gradient_chunks, strict=True)
                ):
                    update_chunks.append(
                        _logical_mousse_update(
                            logical_momentum,
                            logical_gradient,
                            state,
                            logical_index,
                            step=step,
                            factor_beta=group["factor_beta"],
                            factor_epsilon=group["factor_epsilon"],
                            factor_alpha=group["factor_alpha"],
                            refresh_interval=group["refresh_interval"],
                            bias_correction=group["bias_correction"],
                            grafting=group["grafting"],
                            ns_epsilon=group["ns_epsilon"],
                        )
                    )
                if group["weight_decay"]:
                    parameter.mul_(1.0 - base_lr * group["weight_decay"])
                parameter_chunks = logical_matrix_slices(parameter)
                for target, update in zip(parameter_chunks, update_chunks, strict=True):
                    rows, cols = update.shape
                    adjusted_lr = base_lr * math.sqrt(float(rows) / float(cols))
                    target.add_(update, alpha=-adjusted_lr)
        return loss


def parameter_routing_audit(model: nn.Module) -> dict[str, object]:
    hidden = list(model.transformer.h.named_parameters(prefix="transformer.h"))
    auxiliary = list(model.lm_head.named_parameters(prefix="lm_head"))
    hidden_ids = {id(parameter) for _, parameter in hidden}
    auxiliary_ids = {id(parameter) for _, parameter in auxiliary}
    all_named = list(model.named_parameters())
    all_ids = {id(parameter) for _, parameter in all_named}
    if hidden_ids & auxiliary_ids:
        raise RuntimeError("hidden and auxiliary parameter groups overlap")
    if hidden_ids | auxiliary_ids != all_ids:
        missing = [name for name, value in all_named if id(value) not in hidden_ids | auxiliary_ids]
        raise RuntimeError(f"parameter routing is not exhaustive: {missing}")
    if any(parameter.ndim != 2 for _, parameter in hidden):
        raise RuntimeError("hidden Mousse route includes a non-matrix")
    qkv = [(name, list(parameter.shape)) for name, parameter in hidden if parameter.size(0) == 3 * parameter.size(1)]
    return {
        "all_parameter_tensors": len(all_named),
        "all_parameters": sum(parameter.numel() for _, parameter in all_named),
        "hidden_matrix_tensors": len(hidden),
        "hidden_matrix_parameters": sum(parameter.numel() for _, parameter in hidden),
        "auxiliary_tensors": len(auxiliary),
        "auxiliary_parameters": sum(parameter.numel() for _, parameter in auxiliary),
        "packed_qkv_tensors": len(qkv),
        "packed_qkv": qkv,
        "logical_hidden_matrices": sum(len(logical_matrix_slices(parameter)) for _, parameter in hidden),
        "hidden_names": [name for name, _ in hidden],
        "auxiliary_names": [name for name, _ in auxiliary],
        "embedding_head_tied": model.transformer.wte.weight is model.lm_head.weight,
        "activation_k_state_routes": 0,
    }


def optimizer_state_breakdown(optimizers: Iterable[torch.optim.Optimizer]) -> dict[str, int]:
    storages: dict[tuple[object, ...], int] = {}
    roles: dict[str, dict[tuple[object, ...], int]] = {}
    refresh_counts: list[int] = []
    for optimizer in optimizers:
        for state in optimizer.state.values():
            for key, value in state.items():
                if str(key).startswith("mousse_refresh_count_"):
                    refresh_counts.append(int(value))
                if not isinstance(value, Tensor):
                    continue
                storage = value.untyped_storage()
                identity = (value.device.type, value.device.index or -1, storage.data_ptr(), storage.nbytes())
                storages[identity] = storage.nbytes()
                role = str(key).rsplit("_", 1)[0]
                roles.setdefault(role, {})[identity] = storage.nbytes()
    result = {f"{role}_bytes": sum(values.values()) for role, values in sorted(roles.items())}
    result["optimizer_state_bytes"] = sum(storages.values())
    result["mousse_refresh_count_total"] = sum(refresh_counts)
    result["mousse_refreshed_logical_matrices"] = sum(value > 0 for value in refresh_counts)
    return {key: int(value) for key, value in result.items()}


def state_schema(optimizer: R1Mousse) -> dict[str, object]:
    counts: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    refresh_counts: list[int] = []
    for state in optimizer.state.values():
        for key, value in state.items():
            role = str(key).rsplit("_", 1)[0]
            counts[role] += 1
            if isinstance(value, Tensor):
                shapes[f"{role}:{tuple(value.shape)}:{value.dtype}"] += 1
            elif role == "mousse_refresh_count":
                refresh_counts.append(int(value))
    return {
        "roles": dict(sorted(counts.items())),
        "tensor_shapes": dict(sorted(shapes.items())),
        "refresh_counts": refresh_counts,
        "contains_activation_k_state": any("activation" in key.lower() or "k_state" in key.lower() for key in counts),
    }


def _reference_logical_update(
    momentum_update: Tensor,
    gradient: Tensor,
    state: dict[str, Tensor | int],
    *,
    step: int,
) -> Tensor:
    """Deliberately un-factored audit transcription of the frozen recipe."""
    g = gradient.float()
    update = momentum_update.float()
    rows, cols = g.shape
    if not state:
        state.update(
            {
                "factor_L": torch.zeros((rows, rows), device=g.device),
                "factor_R": torch.zeros((cols, cols), device=g.device),
                "eigval_L": torch.zeros(rows, device=g.device),
                "eigvec_L": torch.eye(rows, device=g.device),
                "eigval_R": torch.zeros(cols, device=g.device),
                "eigvec_R": torch.eye(cols, device=g.device),
                "refresh_count": 0,
            }
        )
    factor_l, factor_r = state["factor_L"], state["factor_R"]
    assert isinstance(factor_l, Tensor) and isinstance(factor_r, Tensor)
    factor_l.mul_(0.95).add_(g @ g.T, alpha=0.05)
    factor_r.mul_(0.95).add_(g.T @ g, alpha=0.05)
    if step % 10 == 1:
        correction = 1.0 - 0.95**step
        normalized_l = (factor_l / correction) * (rows / (factor_l / correction).trace())
        normalized_r = (factor_r / correction) * (cols / (factor_r / correction).trace())
        eigen_l, vectors_l = torch.linalg.eigh(normalized_l + 1e-5 * torch.eye(rows, device=g.device))
        eigen_r, vectors_r = torch.linalg.eigh(normalized_r + 1e-5 * torch.eye(cols, device=g.device))
        state["eigval_L"] = clean_eigenvalues(eigen_l, 1e-5)
        state["eigvec_L"] = vectors_l
        state["eigval_R"] = clean_eigenvalues(eigen_r, 1e-5)
        state["eigvec_R"] = vectors_r
        state["refresh_count"] = int(state["refresh_count"]) + 1
    eigval_l, eigvec_l = state["eigval_L"], state["eigvec_L"]
    eigval_r, eigvec_r = state["eigval_R"], state["eigvec_R"]
    assert all(isinstance(value, Tensor) for value in (eigval_l, eigvec_l, eigval_r, eigvec_r))
    scale_l, scale_r = eigval_l.abs().pow(0.125), eigval_r.abs().pow(0.125)
    update = eigvec_l.T @ update @ eigvec_r
    update = update / scale_l[:, None] / scale_r[None, :]
    update = zeropower_via_newtonschulz5(update, epsilon=1e-8)
    graft_norm = update.norm()
    update = update / scale_l[:, None] / scale_r[None, :]
    update = eigvec_l @ update @ eigvec_r.T
    update = update * (graft_norm / update.norm())
    return update.to(momentum_update.dtype)


def run_small_matrix_reference_audit(device: str = "cpu") -> dict[str, object]:
    """Compare 12 optimizer steps to an independent literal reference rollout."""
    target = torch.device(device)
    torch.manual_seed(451245)
    parameter = nn.Parameter(torch.randn((12, 4), device=target, dtype=torch.float32))
    optimizer = R1Mousse([parameter], lr=0.015)
    initial = parameter.detach().clone()
    gradients = [torch.randn_like(parameter) * (0.25 + 0.01 * step) for step in range(12)]
    reference_parameter = initial.clone()
    reference_momentum = torch.zeros_like(reference_parameter)
    reference_states: list[dict[str, Tensor | int]] = [{}, {}, {}]
    parameter_errors = []
    for zero_based_step, gradient in enumerate(gradients):
        parameter.grad = gradient.clone()
        optimizer.step()
        one_based_step = zero_based_step + 1
        reference_momentum.mul_(0.95).add_(gradient)
        updates = [
            _reference_logical_update(momentum, grad, logical_state, step=one_based_step)
            for momentum, grad, logical_state in zip(
                reference_momentum.split(4), gradient.split(4), reference_states, strict=True
            )
        ]
        reference_parameter.mul_(1.0 - 0.015 * 0.01)
        for logical_parameter, update in zip(reference_parameter.split(4), updates, strict=True):
            logical_parameter.add_(update, alpha=-0.015)
        parameter_errors.append(float((parameter.detach() - reference_parameter).abs().max()))
    schema = state_schema(optimizer)
    state = optimizer.state[parameter]
    observed_refreshes = [int(state[_state_key("refresh_count", index)]) for index in range(3)]
    if observed_refreshes != [2, 2, 2]:
        raise AssertionError(f"refresh schedule mismatch: {observed_refreshes}")
    if schema["contains_activation_k_state"]:
        raise AssertionError("Newton--Muon activation-K state leaked into Mousse")
    if torch.equal(parameter, initial) or not torch.isfinite(parameter).all():
        raise AssertionError("Mousse parameter update is absent or non-finite")
    for value in state.values():
        if isinstance(value, Tensor) and not torch.isfinite(value).all():
            raise AssertionError("Mousse optimizer state became non-finite")
    state_errors = []
    for logical_index, reference_state in enumerate(reference_states):
        for role in ("factor_L", "factor_R", "eigval_L", "eigvec_L", "eigval_R", "eigvec_R"):
            actual = state[_state_key(role, logical_index)]
            expected = reference_state[role]
            assert isinstance(actual, Tensor) and isinstance(expected, Tensor)
            state_errors.append(float((actual - expected).abs().max()))
    max_parameter_error = max(parameter_errors)
    max_state_error = max(state_errors)
    if max_parameter_error > 2e-6 or max_state_error > 2e-6:
        raise AssertionError(
            f"optimizer/reference mismatch: parameter={max_parameter_error}, state={max_state_error}"
        )
    expected_roles = {
        "momentum": 1,
        "mousse_factor_L": 3,
        "mousse_factor_R": 3,
        "mousse_eigval_L": 3,
        "mousse_eigvec_L": 3,
        "mousse_eigval_R": 3,
        "mousse_eigvec_R": 3,
        "mousse_refresh_count": 3,
    }
    if schema["roles"] != expected_roles:
        raise AssertionError(f"unexpected state schema: {schema['roles']}")
    return {
        "status": "passed",
        "device": str(target),
        "steps": 12,
        "refresh_steps": [1, 11],
        "refresh_counts_per_logical_matrix": observed_refreshes,
        "packed_qkv_shape": [12, 4],
        "logical_shapes": [[4, 4], [4, 4], [4, 4]],
        "state_schema": schema,
        "state_breakdown": optimizer_state_breakdown([optimizer]),
        "finite": True,
        "single_step_parameter_max_abs_error": parameter_errors[0],
        "twelve_step_parameter_max_abs_error": parameter_errors[-1],
        "factor_eigensystem_max_abs_error": max_state_error,
        "reference_tolerance": 2e-6,
        "activation_k_state_routes": 0,
        "ns_coefficients": [list(values) for values in NS_COEFFICIENTS],
    }
