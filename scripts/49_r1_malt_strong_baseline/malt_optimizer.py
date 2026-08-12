"""Paper-derived MALT/MALTER adaptation for the controlled 124M R1 environment.

This is an independent transcription of Algorithms 1--2 and Equation (17) in
arXiv:2608.05088v1.  No author implementation was publicly available when
experiment 49 was frozen.  The R1 adaptation keeps the existing packed-QKV
split, but the primary Algorithm-1 arm does not add an unreported shape
multiplier.

MALTER follows Equation (17) and the explanatory prose: ``alpha_t`` does not
contain the base learning rate.  Algorithm 2 line 11 prints an extra ``eta``
and line 12 multiplies by ``eta`` again; without author code this is treated as
an internal paper inconsistency and is recorded in the experiment contract.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Iterable

import torch
from torch import Tensor, nn


ARXIV_ID = "2608.05088v1"
Orthogonalize = Callable[..., Tensor]


def logical_matrix_slices(matrix: Tensor) -> tuple[Tensor, ...]:
    """Apply the accepted R1 packed-QKV convention."""
    if matrix.ndim != 2:
        raise ValueError("MALT-R1 accepts hidden 2D matrices only")
    rows, cols = matrix.shape
    if rows == 3 * cols:
        return tuple(matrix.split(cols, dim=0))
    return (matrix,)


def exact_polar(matrix: Tensor, *, steps: int = 5) -> Tensor:
    """Deterministic CPU/GPU reference polar factor; ``steps`` is ignored."""
    del steps
    u, _, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    return (u @ vh).to(matrix.dtype)


def _state_key(role: str, logical_index: int) -> str:
    return f"malt_{role}_{logical_index}"


def _initialize_logical_state(
    state: dict[str, object], logical: Tensor, logical_index: int, *, variant: str
) -> None:
    rows, cols = logical.shape
    device = logical.device
    state[_state_key("row_ema", logical_index)] = torch.zeros(rows, device=device, dtype=torch.float32)
    state[_state_key("col_ema", logical_index)] = torch.zeros(cols, device=device, dtype=torch.float32)
    if variant == "malter":
        state[_state_key("nu", logical_index)] = torch.zeros((), device=device, dtype=torch.float32)


def _dimension_scale(logical: Tensor) -> float:
    return math.sqrt(max(logical.shape))


class R1MALT(torch.optim.Optimizer):
    """MALT or MALTER on R1 hidden matrices.

    The left/right statistics and MALTER scalar are float32.  Momentum follows
    the paper's EMA convention.  Orthogonalization is injected by the derived
    R1 trainer so the experiment uses the exact accepted R1 Newton--Schulz
    backend rather than silently substituting another polynomial.
    """

    def __init__(
        self,
        params,
        *,
        lr: float,
        orthogonalize: Orthogonalize,
        variant: str = "malt",
        beta1: float = 0.95,
        beta2: float = 0.99,
        epsilon: float = 1e-8,
        weight_decay: float = 0.1,
        orthogonalize_steps: int = 5,
        r1_dimension_scale: bool = False,
    ) -> None:
        if variant not in {"malt", "malter"}:
            raise ValueError(f"unsupported variant: {variant}")
        if lr <= 0 or epsilon <= 0 or weight_decay < 0:
            raise ValueError("lr/epsilon must be positive and weight decay non-negative")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if orthogonalize_steps <= 0:
            raise ValueError("orthogonalize_steps must be positive")
        if not callable(orthogonalize):
            raise TypeError("orthogonalize must be callable")
        defaults = dict(
            lr=float(lr),
            variant=variant,
            beta1=float(beta1),
            beta2=float(beta2),
            epsilon=float(epsilon),
            weight_decay=float(weight_decay),
            orthogonalize_steps=int(orthogonalize_steps),
            r1_dimension_scale=bool(r1_dimension_scale),
            step=0,
        )
        super().__init__(params, defaults)
        self._orthogonalize = orthogonalize

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            group["step"] = int(group.get("step", 0)) + 1
            step = int(group["step"])
            lr = float(group["lr"])
            beta1 = float(group["beta1"])
            beta2 = float(group["beta2"])
            epsilon = float(group["epsilon"])
            variant = str(group["variant"])
            weight_decay = float(group["weight_decay"])
            orth_steps = int(group["orthogonalize_steps"])
            use_dimension_scale = bool(group["r1_dimension_scale"])

            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if parameter.ndim != 2 or gradient.shape != parameter.shape:
                    raise RuntimeError("MALT-R1 parameter/gradient must be matching matrices")
                if not torch.isfinite(gradient).all():
                    raise RuntimeError("MALT-R1 received a non-finite gradient")

                state = self.state[parameter]
                momentum = state.get("malt_momentum")
                if momentum is None:
                    momentum = state["malt_momentum"] = torch.zeros_like(
                        parameter, dtype=torch.float32
                    )
                momentum.mul_(beta1).add_(gradient, alpha=1.0 - beta1)

                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)

                logical_gradients = logical_matrix_slices(gradient)
                logical_momenta = logical_matrix_slices(momentum)
                logical_updates: list[Tensor] = []
                alpha_values: list[float] = []
                for logical_index, (logical_gradient, logical_momentum) in enumerate(
                    zip(logical_gradients, logical_momenta, strict=True)
                ):
                    row_key = _state_key("row_ema", logical_index)
                    if row_key not in state:
                        _initialize_logical_state(
                            state, logical_gradient, logical_index, variant=variant
                        )
                    row_ema = state[row_key]
                    col_ema = state[_state_key("col_ema", logical_index)]
                    if not isinstance(row_ema, Tensor) or not isinstance(col_ema, Tensor):
                        raise RuntimeError("corrupt MALT diagonal state")

                    gradient32 = logical_gradient.float()
                    row_ema.mul_(beta2).add_(
                        gradient32.square().sum(dim=1), alpha=1.0 - beta2
                    )
                    col_ema.mul_(beta2).add_(
                        gradient32.square().sum(dim=0), alpha=1.0 - beta2
                    )
                    left = (row_ema + epsilon).pow(-0.125)
                    right = (col_ema + epsilon).pow(-0.125)

                    preconditioned_momentum = (
                        left[:, None] * logical_momentum.float() * right[None, :]
                    )
                    orthogonal = self._orthogonalize(
                        preconditioned_momentum, steps=orth_steps
                    ).float()
                    mapped = left[:, None] * orthogonal * right[None, :]
                    graft = orthogonal.norm() / (mapped.norm() + epsilon)
                    update = mapped * graft

                    alpha = torch.ones((), device=parameter.device, dtype=torch.float32)
                    if variant == "malter":
                        nu_key = _state_key("nu", logical_index)
                        nu = state[nu_key]
                        if not isinstance(nu, Tensor) or nu.ndim != 0:
                            raise RuntimeError("corrupt MALTER scalar state")
                        preconditioned_gradient = (
                            left[:, None] * gradient32 * right[None, :]
                        )
                        nu.mul_(beta2).add_(
                            preconditioned_gradient.square().sum(), alpha=1.0 - beta2
                        )
                        bias_ratio = math.sqrt(1.0 - beta2**step) / (
                            1.0 - beta1**step + epsilon
                        )
                        alpha = (
                            bias_ratio
                            * preconditioned_momentum.norm()
                            / (nu.sqrt() + epsilon)
                        )
                        update = update * alpha

                    if use_dimension_scale:
                        update = update * _dimension_scale(logical_gradient)
                    if not torch.isfinite(update).all() or not torch.isfinite(alpha):
                        raise RuntimeError("MALT-R1 produced a non-finite update")
                    logical_updates.append(update.to(parameter.dtype))
                    alpha_values.append(float(alpha.detach().cpu()))

                full_update = (
                    torch.cat(logical_updates, dim=0)
                    if len(logical_updates) > 1
                    else logical_updates[0]
                )
                parameter.add_(full_update, alpha=-lr)
                state["malt_last_alpha_min"] = min(alpha_values)
                state["malt_last_alpha_max"] = max(alpha_values)
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
        missing = [
            name
            for name, value in all_named
            if id(value) not in hidden_ids | auxiliary_ids
        ]
        raise RuntimeError(f"parameter routing is not exhaustive: {missing}")
    if any(parameter.ndim != 2 for _, parameter in hidden):
        raise RuntimeError("hidden MALT route includes a non-matrix")
    qkv = [
        (name, list(parameter.shape))
        for name, parameter in hidden
        if parameter.size(0) == 3 * parameter.size(1)
    ]
    return {
        "all_parameter_tensors": len(all_named),
        "all_parameters": sum(parameter.numel() for _, parameter in all_named),
        "hidden_matrix_tensors": len(hidden),
        "hidden_matrix_parameters": sum(parameter.numel() for _, parameter in hidden),
        "auxiliary_tensors": len(auxiliary),
        "auxiliary_parameters": sum(parameter.numel() for _, parameter in auxiliary),
        "packed_qkv_tensors": len(qkv),
        "packed_qkv": qkv,
        "logical_hidden_matrices": sum(
            len(logical_matrix_slices(parameter)) for _, parameter in hidden
        ),
        "hidden_names": [name for name, _ in hidden],
        "auxiliary_names": [name for name, _ in auxiliary],
        "embedding_head_tied": model.transformer.wte.weight is model.lm_head.weight,
        "activation_k_state_routes": 0,
    }


def _role(key: object) -> str:
    text = str(key)
    if text in {"malt_momentum", "malt_last_alpha_min", "malt_last_alpha_max"}:
        return text
    if text.rsplit("_", 1)[-1].isdigit():
        return text.rsplit("_", 1)[0]
    return text


def optimizer_state_breakdown(
    optimizers: Iterable[torch.optim.Optimizer],
) -> dict[str, int | float]:
    storages: dict[tuple[object, ...], int] = {}
    roles: dict[str, dict[tuple[object, ...], int]] = {}
    alpha_min: list[float] = []
    alpha_max: list[float] = []
    for optimizer in optimizers:
        for state in optimizer.state.values():
            for key, value in state.items():
                if key == "malt_last_alpha_min":
                    alpha_min.append(float(value))
                if key == "malt_last_alpha_max":
                    alpha_max.append(float(value))
                if not isinstance(value, Tensor):
                    continue
                storage = value.untyped_storage()
                identity = (
                    value.device.type,
                    value.device.index if value.device.index is not None else -1,
                    storage.data_ptr(),
                    storage.nbytes(),
                )
                storages[identity] = storage.nbytes()
                roles.setdefault(_role(key), {})[identity] = storage.nbytes()
    result: dict[str, int | float] = {
        f"{role}_bytes": sum(values.values())
        for role, values in sorted(roles.items())
    }
    result["optimizer_state_bytes"] = sum(storages.values())
    result["malt_last_alpha_global_min"] = min(alpha_min) if alpha_min else 1.0
    result["malt_last_alpha_global_max"] = max(alpha_max) if alpha_max else 1.0
    return result


def state_schema(optimizer: R1MALT) -> dict[str, object]:
    counts: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    all_state_finite = True
    all_diagonal_state_nonnegative = True
    row_column_mass_max_relative_error = 0.0
    for state in optimizer.state.values():
        for key, value in state.items():
            role = _role(key)
            counts[role] += 1
            if isinstance(value, Tensor):
                shapes[f"{role}:{tuple(value.shape)}:{value.dtype}"] += 1
                all_state_finite = all_state_finite and bool(torch.isfinite(value).all())
                if role in {"malt_row_ema", "malt_col_ema", "malt_nu"}:
                    all_diagonal_state_nonnegative = (
                        all_diagonal_state_nonnegative and bool((value >= 0).all())
                    )
        row_indices = sorted(
            int(str(key).rsplit("_", 1)[-1])
            for key in state
            if str(key).startswith("malt_row_ema_")
        )
        for logical_index in row_indices:
            row = state[_state_key("row_ema", logical_index)]
            col = state[_state_key("col_ema", logical_index)]
            assert isinstance(row, Tensor) and isinstance(col, Tensor)
            row_mass = float(row.double().sum().cpu())
            col_mass = float(col.double().sum().cpu())
            scale = max(abs(row_mass), abs(col_mass), 1e-30)
            row_column_mass_max_relative_error = max(
                row_column_mass_max_relative_error,
                abs(row_mass - col_mass) / scale,
            )
    numerical_checks = {
        "all_state_finite": all_state_finite,
        "all_diagonal_state_nonnegative": all_diagonal_state_nonnegative,
        "row_column_mass_conserved": row_column_mass_max_relative_error <= 1e-5,
    }
    return {
        "roles": dict(sorted(counts.items())),
        "tensor_shapes": dict(sorted(shapes.items())),
        "contains_activation_k_state": any(
            "activation" in key.lower() or "k_state" in key.lower()
            for key in counts
        ),
        "optimizer_group_steps": [int(group.get("step", 0)) for group in optimizer.param_groups],
        "numerical_checks": numerical_checks,
        "row_column_mass_max_relative_error": row_column_mass_max_relative_error,
        "numerical_checks_passed": all(numerical_checks.values()),
    }


def _literal_step(
    parameter: Tensor,
    gradient: Tensor,
    state: dict[str, object],
    *,
    lr: float,
    variant: str,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    step: int,
) -> tuple[Tensor, dict[str, object], list[float]]:
    """Independent literal reference used only by the audit."""
    momentum = state.setdefault("momentum", torch.zeros_like(parameter))
    assert isinstance(momentum, Tensor)
    momentum = beta1 * momentum + (1.0 - beta1) * gradient
    state["momentum"] = momentum
    output = parameter * (1.0 - lr * weight_decay)
    updates: list[Tensor] = []
    alphas: list[float] = []
    for index, (g, m) in enumerate(
        zip(logical_matrix_slices(gradient), logical_matrix_slices(momentum), strict=True)
    ):
        rows = state.setdefault(f"row_{index}", torch.zeros(g.shape[0]))
        cols = state.setdefault(f"col_{index}", torch.zeros(g.shape[1]))
        assert isinstance(rows, Tensor) and isinstance(cols, Tensor)
        rows = beta2 * rows + (1.0 - beta2) * g.float().square().sum(1)
        cols = beta2 * cols + (1.0 - beta2) * g.float().square().sum(0)
        state[f"row_{index}"] = rows
        state[f"col_{index}"] = cols
        left, right = (rows + epsilon).pow(-0.125), (cols + epsilon).pow(-0.125)
        pre_m = left[:, None] * m.float() * right[None, :]
        orth = exact_polar(pre_m).float()
        mapped = left[:, None] * orth * right[None, :]
        update = mapped * orth.norm() / (mapped.norm() + epsilon)
        alpha = torch.ones(())
        if variant == "malter":
            pre_g = left[:, None] * g.float() * right[None, :]
            nu = state.setdefault(f"nu_{index}", torch.zeros(()))
            assert isinstance(nu, Tensor)
            nu = beta2 * nu + (1.0 - beta2) * pre_g.square().sum()
            state[f"nu_{index}"] = nu
            alpha = (
                math.sqrt(1.0 - beta2**step)
                / (1.0 - beta1**step + epsilon)
                * pre_m.norm()
                / (nu.sqrt() + epsilon)
            )
            update = update * alpha
        updates.append(update)
        alphas.append(float(alpha))
    full = torch.cat(updates, 0) if len(updates) > 1 else updates[0]
    return output - lr * full.to(output.dtype), state, alphas


def run_small_matrix_reference_audit(device: str = "cpu") -> dict[str, object]:
    """Compare both variants with a literal rollout and test key invariants."""
    torch.manual_seed(4901)
    reports: dict[str, object] = {}
    for variant in ("malt", "malter"):
        initial = torch.randn(6, 2, device=device) * 0.1
        parameter = nn.Parameter(initial.clone())
        optimizer = R1MALT(
            [parameter],
            lr=3.6e-4 if variant == "malt" else 3.6e-3,
            orthogonalize=exact_polar,
            variant=variant,
            weight_decay=0.1,
        )
        reference = initial.clone()
        reference_state: dict[str, object] = {}
        first_alpha: list[float] = []
        for step in range(1, 6):
            gradient = torch.randn_like(parameter) * (0.2 + 0.01 * step)
            parameter.grad = gradient.clone()
            optimizer.step()
            reference, reference_state, alphas = _literal_step(
                reference,
                gradient.cpu(),
                reference_state,
                lr=3.6e-4 if variant == "malt" else 3.6e-3,
                variant=variant,
                beta1=0.95,
                beta2=0.99,
                epsilon=1e-8,
                weight_decay=0.1,
                step=step,
            )
            if step == 1:
                first_alpha = alphas
        max_error = float((parameter.detach().cpu() - reference).abs().max())
        if max_error > 2e-6:
            raise AssertionError(f"{variant} literal rollout mismatch: {max_error}")
        schema = state_schema(optimizer)
        expected_rows = 3
        expected_cols = 3
        if schema["roles"].get("malt_row_ema") != expected_rows:
            raise AssertionError(f"{variant} row-state schema mismatch: {schema}")
        if schema["roles"].get("malt_col_ema") != expected_cols:
            raise AssertionError(f"{variant} col-state schema mismatch: {schema}")
        if variant == "malter":
            if schema["roles"].get("malt_nu") != 3:
                raise AssertionError(f"MALTER scalar-state schema mismatch: {schema}")
            if max(abs(value - 1.0) for value in first_alpha) > 5e-5:
                raise AssertionError(f"MALTER first-step alpha is not one: {first_alpha}")
        elif "malt_nu" in schema["roles"]:
            raise AssertionError("MALT unexpectedly owns MALTER scalar state")
        if schema["contains_activation_k_state"]:
            raise AssertionError("activation-K state leaked into MALT")
        reports[variant] = {
            "five_step_parameter_max_abs_error": max_error,
            "first_step_alpha": first_alpha,
            "state_schema": schema,
            "state_breakdown": optimizer_state_breakdown([optimizer]),
        }

    # Positive gradient rescaling should not change the one-step MALT update
    # direction/magnitude when epsilon is negligible.
    base = torch.randn(5, 3) * 0.1
    grad = torch.randn_like(base)
    scaled_parameters = [nn.Parameter(base.clone()), nn.Parameter(base.clone())]
    scaled_updates: list[Tensor] = []
    for scale, parameter in zip((1.0, 7.0), scaled_parameters, strict=True):
        optimizer = R1MALT(
            [parameter], lr=1e-3, orthogonalize=exact_polar, epsilon=1e-30,
            weight_decay=0.0, r1_dimension_scale=False,
        )
        parameter.grad = grad * scale
        optimizer.step()
        scaled_updates.append(base - parameter.detach())
    scale_error = float((scaled_updates[0] - scaled_updates[1]).abs().max())
    if scale_error > 2e-6:
        raise AssertionError(f"MALT positive-scale invariance failed: {scale_error}")

    # Transposition swaps left/right statistics and transposes the update.
    transpose_parameters = [nn.Parameter(base.clone()), nn.Parameter(base.T.clone())]
    transpose_updates: list[Tensor] = []
    for parameter, gradient in zip(transpose_parameters, (grad, grad.T), strict=True):
        optimizer = R1MALT(
            [parameter], lr=1e-3, orthogonalize=exact_polar,
            weight_decay=0.0, r1_dimension_scale=False,
        )
        parameter.grad = gradient.clone()
        optimizer.step()
        transpose_updates.append(parameter.detach().clone())
    transpose_error = float((transpose_updates[0].T - transpose_updates[1]).abs().max())
    if transpose_error > 2e-6:
        raise AssertionError(f"MALT transpose equivariance failed: {transpose_error}")

    return {
        "status": "passed",
        "arxiv_id": ARXIV_ID,
        "implementation_label": "paper-derived independent implementation",
        "malter_formula_choice": "equation_17_single_eta",
        "variants": reports,
        "positive_scale_invariance_max_abs_error": scale_error,
        "transpose_equivariance_max_abs_error": transpose_error,
        "activation_k_state_routes": 0,
    }
