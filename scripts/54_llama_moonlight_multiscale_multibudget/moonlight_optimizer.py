"""Audited single-GPU Moonlight Muon matrix optimizer for EX54.

The numerical rule is the Moonlight Muon rule already audited in Experiment 19:
Muon momentum + Newton--Schulz orthogonalization, parameter-wise
``0.2 * sqrt(max(rows, cols))`` update scaling, and decoupled weight decay
using the unadjusted base learning rate.  This implementation keeps optimizer
state attached to the original parameter and never constructs curvature/eigen
factor matrices.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn

from triton_kernels import XXT, ba_plus_cAA


NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5
MOONLIGHT_REFERENCE = "MoonshotAI/Moonlight:examples/toy_train.py"


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
                nesterov = (
                    gradient.add(momentum, alpha=beta) if use_nesterov else momentum
                )
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


def optimizer_state_breakdown(optimizers: Iterable[torch.optim.Optimizer]) -> dict[str, int]:
    """Count unique optimizer-state storage bytes by semantic state key."""
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
    result = {
        f"{key}_bytes": int(sum(storages.values()))
        for key, storages in sorted(by_key.items())
    }
    result["optimizer_state_bytes"] = int(sum(all_storages.values()))
    return result


def state_schema(optimizer: R1MoonlightMuon) -> dict[str, object]:
    keys = sorted(
        {
            str(key)
            for state in optimizer.state.values()
            for key, value in state.items()
            if isinstance(value, Tensor)
        }
    )
    matrix_count = sum(1 for group in optimizer.param_groups for _ in group["params"])
    return {
        "optimizer": "R1MoonlightMuon",
        "tensor_state_keys": keys,
        "logical_matrix_parameters": matrix_count,
        "contains_activation_k_state": False,
        "contains_factor_or_eigendecomposition_state": False,
    }


def run_small_matrix_reference_audit(device: str = "cuda") -> dict[str, object]:
    """Reproduce the Experiment-19 packed-QKV Moonlight reference check."""
    target = torch.device(device)
    torch.manual_seed(314159)
    initial = torch.randn((12, 4), device=target, dtype=torch.float32)
    gradient = torch.randn_like(initial)
    parameter = nn.Parameter(initial.clone())
    parameter.grad = gradient.clone()
    optimizer = R1MoonlightMuon([parameter], lr=0.001, weight_decay=0.1)
    optimizer.step()

    lookahead = gradient + 0.95 * gradient
    chunks = []
    for logical in lookahead.split(4, dim=0):
        chunks.append(
            zeropower_via_newtonschulz5(logical) * (0.2 * math.sqrt(4.0))
        )
    direct = torch.cat(chunks, dim=0)
    expected = initial * (1.0 - 0.001 * 0.1) - 0.001 * direct
    torch.testing.assert_close(parameter, expected, rtol=2e-5, atol=2e-6)

    packed_once = zeropower_via_newtonschulz5(lookahead)
    split_once = torch.cat(
        [zeropower_via_newtonschulz5(part) for part in lookahead.split(4, dim=0)],
        dim=0,
    )
    if torch.allclose(packed_once, split_once, rtol=1e-4, atol=1e-5):
        raise AssertionError("packed-QKV split audit was non-discriminating")

    schema = state_schema(optimizer)
    breakdown = optimizer_state_breakdown([optimizer])
    if schema["tensor_state_keys"] != ["momentum_buffer"]:
        raise AssertionError(f"unexpected Moonlight state schema: {schema}")
    return {
        "passed": True,
        "reference": MOONLIGHT_REFERENCE,
        "scale_rule": "0.2*sqrt(max(rows,cols))",
        "base_lr_weight_decay": True,
        "logical_qkv_shapes": [[4, 4], [4, 4], [4, 4]],
        "packed_vs_split_distinct": True,
        "momentum": 0.95,
        "nesterov": True,
        "ns_steps": NS_STEPS,
        "state_schema": schema,
        "state_bytes": breakdown,
    }
