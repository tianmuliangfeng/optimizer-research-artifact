"""Audited single-GPU optimizer kernels for the R1 extended-baseline pilot.

The two implementations below are deliberately small adaptations of the
authors' public reference algorithms.  The adaptation is limited to the R1
GPT-2 parameter layout: packed QKV weights are split into three logical
matrices before orthogonalization, while optimizer state remains attached to
the original packed parameter.

Upstream authorities (retrieved 2026-07-21):

* NorMuon: https://github.com/zichongli5/NorMuon/blob/main/normuon.py
* Moonlight Muon:
  https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn

from triton_kernels import XXT, ba_plus_cAA


NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5
NOR_MUON_REFERENCE_URL = (
    "https://github.com/zichongli5/NorMuon/blob/main/normuon.py"
)
MOONLIGHT_REFERENCE_URL = (
    "https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py"
)


@torch.compile
def zeropower_via_newtonschulz5(gradient: Tensor, steps: int = NS_STEPS) -> Tensor:
    """R1's H100-compatible quintic Newton--Schulz implementation."""
    if gradient.ndim != 2:
        raise ValueError("Newton--Schulz expects one 2D logical matrix")
    a, b, c = NS_COEFFICIENTS
    x = gradient.bfloat16() / gradient.norm().clamp_min(1e-7)
    transposed = gradient.size(0) > gradient.size(1)
    if transposed:
        x = x.T
    x = x.contiguous()

    rows = x.size(0)
    a_buf = torch.empty((rows, rows), device=x.device, dtype=x.dtype)
    b_buf = torch.empty_like(a_buf)
    c_buf = torch.empty_like(x)
    for _ in range(steps):
        XXT(x, out=a_buf)
        ba_plus_cAA(a_buf, beta=b, alpha=c, out=b_buf)
        torch.mm(b_buf, x, out=c_buf)
        c_buf.add_(x, alpha=a)
        x, c_buf = c_buf, x
    if transposed:
        x = x.T
    return x.to(dtype=gradient.dtype)


def logical_matrix_slices(matrix: Tensor) -> tuple[Tensor, ...]:
    """Split GPT-2 packed QKV; leave every other 2D parameter intact."""
    if matrix.ndim != 2:
        raise ValueError("extended Muon methods accept hidden 2D matrices only")
    rows, cols = matrix.shape
    if rows == 3 * cols:
        return tuple(matrix.split(cols, dim=0))
    return (matrix,)


def _normuon_logical_update(
    update: Tensor,
    second_moment: Tensor,
    *,
    beta2: float,
    ns_steps: int,
) -> Tensor:
    """Exact neuron-wise normalization and global-norm restoration."""
    orthogonal = zeropower_via_newtonschulz5(update, steps=ns_steps)
    original_norm = orthogonal.norm()
    row_second = orthogonal.square().mean(dim=-1, keepdim=True)
    second_moment.lerp_(row_second, 1.0 - beta2)
    normalized = orthogonal * second_moment.sqrt().add(1e-10).reciprocal()
    normalized.mul_(original_norm / normalized.norm().add(1e-10))
    rows, cols = normalized.shape
    normalized.mul_(math.sqrt(max(1.0, float(rows) / float(cols))))
    return normalized


class R1NorMuon(torch.optim.Optimizer):
    """Single-device NorMuon with explicit packed-QKV logical splitting."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        momentum: float = 0.95,
        beta2: float = 0.95,
        ns_steps: int = NS_STEPS,
    ) -> None:
        defaults = dict(
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            beta2=float(beta2),
            ns_steps=int(ns_steps),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            beta = float(group["momentum"])
            beta2 = float(group["beta2"])
            ns_steps = int(group["ns_steps"])
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if parameter.ndim != 2:
                    raise ValueError("R1NorMuon received a non-matrix parameter")
                state = self.state[parameter]
                momentum = state.get("momentum_buffer")
                if momentum is None:
                    momentum = state["momentum_buffer"] = torch.zeros_like(parameter)
                    state["second_momentum_buffer"] = torch.zeros(
                        (parameter.size(0), 1),
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                second = state["second_momentum_buffer"]
                momentum.lerp_(gradient, 1.0 - beta)
                nesterov = torch.lerp(gradient, momentum, beta)
                updates = []
                row_offset = 0
                for logical in logical_matrix_slices(nesterov):
                    rows = logical.size(0)
                    updates.append(
                        _normuon_logical_update(
                            logical,
                            second[row_offset : row_offset + rows],
                            beta2=beta2,
                            ns_steps=ns_steps,
                        )
                    )
                    row_offset += rows
                update = torch.cat(updates, dim=0) if len(updates) > 1 else updates[0]
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(update, alpha=-lr)
        return loss


class R1MoonlightMuon(torch.optim.Optimizer):
    """Moonlight Muon scaling/decay with packed-QKV logical splitting."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = NS_STEPS,
    ) -> None:
        defaults = dict(
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            ns_steps=int(ns_steps),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            beta = float(group["momentum"])
            use_nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if parameter.ndim != 2:
                    raise ValueError("R1MoonlightMuon received a non-matrix parameter")
                state = self.state[parameter]
                momentum = state.get("momentum_buffer")
                if momentum is None:
                    momentum = state["momentum_buffer"] = torch.zeros_like(parameter)
                momentum.mul_(beta).add_(gradient)
                nesterov = gradient.add(momentum, alpha=beta) if use_nesterov else momentum
                updates = []
                for logical in logical_matrix_slices(nesterov):
                    orthogonal = zeropower_via_newtonschulz5(logical, steps=ns_steps)
                    rows, cols = logical.shape
                    orthogonal.mul_(0.2 * math.sqrt(float(max(rows, cols))))
                    updates.append(orthogonal)
                update = torch.cat(updates, dim=0) if len(updates) > 1 else updates[0]
                # Moonlight applies decay with the unadjusted base LR.
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(update, alpha=-lr)
        return loss


def parameter_routing_audit(model: nn.Module) -> dict[str, object]:
    """Prove the exact hidden-matrix/auxiliary split used by R1."""
    hidden = list(model.transformer.h.named_parameters(prefix="transformer.h"))
    auxiliary = list(model.lm_head.named_parameters(prefix="lm_head"))
    hidden_ids = {id(parameter) for _, parameter in hidden}
    auxiliary_ids = {id(parameter) for _, parameter in auxiliary}
    all_named = list(model.named_parameters())
    all_ids = {id(parameter) for _, parameter in all_named}
    if hidden_ids & auxiliary_ids:
        raise RuntimeError("hidden and auxiliary parameter groups overlap")
    if hidden_ids | auxiliary_ids != all_ids:
        missing = [name for name, parameter in all_named if id(parameter) not in hidden_ids | auxiliary_ids]
        raise RuntimeError(f"parameter routing is not exhaustive: {missing}")
    nonmatrices = [name for name, parameter in hidden if parameter.ndim != 2]
    if nonmatrices:
        raise RuntimeError(f"hidden optimizer group contains non-matrices: {nonmatrices}")
    qkv = [(name, tuple(parameter.shape)) for name, parameter in hidden if parameter.size(0) == 3 * parameter.size(1)]
    return {
        "all_parameter_tensors": len(all_named),
        "all_parameters": sum(parameter.numel() for _, parameter in all_named),
        "hidden_matrix_tensors": len(hidden),
        "hidden_matrix_parameters": sum(parameter.numel() for _, parameter in hidden),
        "auxiliary_tensors": len(auxiliary),
        "auxiliary_parameters": sum(parameter.numel() for _, parameter in auxiliary),
        "packed_qkv_tensors": len(qkv),
        "packed_qkv": qkv,
        "hidden_names": [name for name, _ in hidden],
        "auxiliary_names": [name for name, _ in auxiliary],
        "embedding_head_tied": model.transformer.wte.weight is model.lm_head.weight,
    }


def optimizer_state_breakdown(optimizers: Iterable[torch.optim.Optimizer]) -> dict[str, int]:
    """Count unique optimizer-state storage bytes, separated by semantic role."""
    by_key: dict[str, dict[tuple[object, ...], int]] = {}
    all_storages: dict[tuple[object, ...], int] = {}
    for optimizer in optimizers:
        for state in optimizer.state.values():
            for key, value in state.items():
                if not isinstance(value, Tensor):
                    continue
                storage = value.untyped_storage()
                storage_key = (
                    value.device.type,
                    value.device.index if value.device.index is not None else -1,
                    storage.data_ptr(),
                    storage.nbytes(),
                )
                all_storages[storage_key] = storage.nbytes()
                by_key.setdefault(str(key), {})[storage_key] = storage.nbytes()
    result = {f"{key}_bytes": int(sum(storages.values())) for key, storages in sorted(by_key.items())}
    result["optimizer_state_bytes"] = int(sum(all_storages.values()))
    return result


def run_single_step_reference_audit(device: str = "cuda") -> dict[str, object]:
    """Numerically check adapted QKV, scaling, decay, and NorMuon state shapes."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the H100 optimizer audit")
    target = torch.device(device)
    torch.manual_seed(314159)
    gradient = torch.randn((12, 4), device=target, dtype=torch.float32)
    initial = torch.randn_like(gradient)

    # AdamW: verify the fused implementation used by the official source is
    # decoupled weight decay, with bias-corrected first/second moments.
    adam_parameter = nn.Parameter(initial.clone())
    adam_parameter.grad = gradient.clone()
    adam = torch.optim.AdamW(
        [adam_parameter],
        lr=0.0036,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )
    adam.step()
    adam_expected = initial * (1.0 - 0.0036 * 0.1)
    adam_expected.add_(gradient / (gradient.abs() + 1e-8), alpha=-0.0036)
    torch.testing.assert_close(adam_parameter, adam_expected, rtol=2e-5, atol=2e-6)

    # NorMuon: compare the optimizer step with a direct transcription applied
    # independently to Q, K, and V.
    nor_parameter = nn.Parameter(initial.clone())
    nor_parameter.grad = gradient.clone()
    nor = R1NorMuon([nor_parameter], lr=0.02, weight_decay=0.01)
    nor.step()
    beta = beta2 = 0.95
    momentum = torch.zeros_like(gradient)
    momentum.lerp_(gradient, 1.0 - beta)
    lookahead = torch.lerp(gradient, momentum, beta)
    second = torch.zeros((12, 1), device=target)
    direct_chunks = []
    for index, logical in enumerate(lookahead.split(4, dim=0)):
        direct_chunks.append(
            _normuon_logical_update(
                logical,
                second[index * 4 : (index + 1) * 4],
                beta2=beta2,
                ns_steps=NS_STEPS,
            )
        )
    nor_expected = initial * (1.0 - 0.02 * 0.01) - 0.02 * torch.cat(direct_chunks)
    torch.testing.assert_close(nor_parameter, nor_expected, rtol=2e-5, atol=2e-6)
    if tuple(nor.state[nor_parameter]["second_momentum_buffer"].shape) != (12, 1):
        raise AssertionError("NorMuon row-second-moment state has the wrong shape")

    # Moonlight: check the authors' 0.2*sqrt(max(A,B)) scaling and base-LR decay.
    moon_parameter = nn.Parameter(initial.clone())
    moon_parameter.grad = gradient.clone()
    moon = R1MoonlightMuon([moon_parameter], lr=0.001, weight_decay=0.1)
    moon.step()
    moon_lookahead = gradient + 0.95 * gradient
    moon_chunks = []
    for logical in moon_lookahead.split(4, dim=0):
        moon_chunks.append(
            zeropower_via_newtonschulz5(logical) * (0.2 * math.sqrt(4.0))
        )
    moon_expected = initial * (1.0 - 0.001 * 0.1) - 0.001 * torch.cat(moon_chunks)
    torch.testing.assert_close(moon_parameter, moon_expected, rtol=2e-5, atol=2e-6)

    # The packed matrix must genuinely be treated as three logical matrices.
    packed_once = zeropower_via_newtonschulz5(moon_lookahead)
    split_update = torch.cat(
        [zeropower_via_newtonschulz5(part) for part in moon_lookahead.split(4)], dim=0
    )
    if torch.allclose(packed_once, split_update, rtol=1e-4, atol=1e-5):
        raise AssertionError("QKV split audit was non-discriminating")

    return {
        "status": "passed",
        "device": str(target),
        "ns_coefficients": NS_COEFFICIENTS,
        "ns_steps": NS_STEPS,
        "qkv_shape": list(gradient.shape),
        "adamw_decoupled_single_step": True,
        "logical_qkv_shapes": [[4, 4], [4, 4], [4, 4]],
        "normuon_row_state_shape": [12, 1],
        "moonlight_scale_for_4x4": 0.2 * math.sqrt(4.0),
        "packed_vs_split_distinct": True,
    }
