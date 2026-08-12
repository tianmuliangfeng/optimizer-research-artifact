"""Single-GPU LLaMA/SwiGLU architecture-validation trainer.

The model is deliberately parameter-matched to the official R1 GPT model and
the five methods share one model, data loader, initialization, and training
loop.  W&B is intentionally absent from this process; the controller uploads
the validated local scalar evidence after training has completed.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from triton_kernels import XXT, ba_plus_cAA
except Exception as exc:  # audit-only remains usable without the GPU kernels
    XXT = None
    ba_plus_cAA = None
    TRITON_KERNEL_IMPORT_ERROR: Exception | None = exc
else:
    TRITON_KERNEL_IMPORT_ERROR = None


METHODS = ("adamw", "muon", "newton_full", "down_none", "down_diag")
NEWTON_METHODS = ("newton_full", "down_none", "down_diag")
METRIC_FIELDS = (
    "event",
    "step",
    "loss",
    "train_s",
    "steady_train_s",
    "step_avg_ms",
    "lr_backup",
    "lr_matrix",
    "tokens_seen",
)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_named_parameters(module: nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def iter_tensors(value: Any) -> Iterable[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_tensors(child)


def unique_storage_bytes(tensors: Iterable[Tensor]) -> int:
    storages: dict[tuple[str, int, int, int], int] = {}
    for tensor in tensors:
        if not isinstance(tensor, Tensor):
            continue
        storage = tensor.untyped_storage()
        device_index = tensor.device.index if tensor.device.index is not None else -1
        key = (tensor.device.type, device_index, storage.data_ptr(), storage.nbytes())
        storages[key] = storage.nbytes()
    return int(sum(storages.values()))


# -----------------------------------------------------------------------------
# Activation statistics. The dense path uses the pinned Newton-Muon Triton XXT
# kernel so BF16 down-projection activations are not materialized as a giant
# FP32 [B*T, d_ff] temporary.


def _dummy_scalar_like(x: Tensor) -> Tensor:
    return x.new_empty(())


@torch.compile
def _accum_xtx_impl(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor) -> Tensor:
    if XXT is None:
        raise RuntimeError(f"triton_kernels import failed: {TRITON_KERNEL_IMPORT_ERROR!r}")
    transposed = x_2d.transpose(0, 1)
    XXT(transposed, out=tmp)
    tmp.mul_(1.0 / x_2d.size(0))
    accum.add_(tmp)
    count.add_(1.0)
    return _dummy_scalar_like(accum)


@torch.compile
def _accum_diag_impl(x_2d: Tensor, accum: Tensor, count: Tensor) -> Tensor:
    # Inductor fuses square/reduction/cast; it does not retain an FP32 copy of
    # the full activation matrix.
    accum.add_(x_2d.square().mean(dim=0, dtype=torch.float32))
    count.add_(1.0)
    return _dummy_scalar_like(accum)


@torch.library.custom_op(
    "llama_swiglu::accum_xtx", mutates_args=("accum", "count", "tmp")
)
@torch.no_grad()
def accum_xtx_op(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor) -> Tensor:
    return _accum_xtx_impl(x_2d, accum, count, tmp)


@accum_xtx_op.register_fake
def accum_xtx_fake(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor) -> Tensor:
    return accum.new_empty(())


@torch.library.custom_op(
    "llama_swiglu::accum_diag", mutates_args=("accum", "count")
)
@torch.no_grad()
def accum_diag_op(x_2d: Tensor, accum: Tensor, count: Tensor) -> Tensor:
    return _accum_diag_impl(x_2d, accum, count)


@accum_diag_op.register_fake
def accum_diag_fake(x_2d: Tensor, accum: Tensor, count: Tensor) -> Tensor:
    return accum.new_empty(())


# -----------------------------------------------------------------------------
# Model


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    intermediate_size: int = 2048
    sequence_length: int = 1024
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype=x.dtype) * self.weight


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class LlamaAttention(nn.Module):
    def __init__(self, config: ModelConfig, enable_stats: bool) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.enable_stats = bool(enable_stats)
        if self.enable_stats:
            d = config.n_embd
            self.register_buffer("attn_in_accum", torch.zeros(d, d), persistent=True)
            self.register_buffer("attn_in_count", torch.zeros(()), persistent=True)
            self.register_buffer("o_accum", torch.zeros(d, d), persistent=True)
            self.register_buffer("o_count", torch.zeros(()), persistent=True)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        precond_flag: bool,
        xtx_tmp_d: Tensor,
    ) -> Tensor:
        batch, length, width = x.shape
        if precond_flag and self.enable_stats:
            torch.ops.llama_swiglu.accum_xtx(
                x.flatten(0, -2), self.attn_in_accum, self.attn_in_count, xtx_tmp_d
            )
        q = self.q_proj(x).view(batch, length, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch, length, self.n_head, self.head_dim)
        v = self.v_proj(x).view(batch, length, self.n_head, self.head_dim)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        if precond_flag and self.enable_stats:
            torch.ops.llama_swiglu.accum_xtx(
                y.flatten(0, -2), self.o_accum, self.o_count, xtx_tmp_d
            )
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig, down_mode: str, enable_stats: bool) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.enable_stats = bool(enable_stats)
        self.down_mode = down_mode
        if self.enable_stats:
            d = config.n_embd
            ff = config.intermediate_size
            self.register_buffer("mlp_in_accum", torch.zeros(d, d), persistent=True)
            self.register_buffer("mlp_in_count", torch.zeros(()), persistent=True)
            if self.down_mode == "dense":
                self.register_buffer("down_accum", torch.zeros(ff, ff), persistent=True)
                self.register_buffer("down_count", torch.zeros(()), persistent=True)
            elif self.down_mode == "diag":
                self.register_buffer("down_accum", torch.zeros(ff), persistent=True)
                self.register_buffer("down_count", torch.zeros(()), persistent=True)
            elif self.down_mode != "none":
                raise ValueError(f"invalid down mode {self.down_mode!r}")

    def forward(
        self,
        x: Tensor,
        precond_flag: bool,
        xtx_tmp_d: Tensor,
        xtx_tmp_ff: Tensor,
    ) -> Tensor:
        if precond_flag and self.enable_stats:
            torch.ops.llama_swiglu.accum_xtx(
                x.flatten(0, -2), self.mlp_in_accum, self.mlp_in_count, xtx_tmp_d
            )
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        if precond_flag and self.enable_stats and self.down_mode != "none":
            flat = hidden.flatten(0, -2)
            if self.down_mode == "dense":
                torch.ops.llama_swiglu.accum_xtx(
                    flat, self.down_accum, self.down_count, xtx_tmp_ff
                )
            else:
                torch.ops.llama_swiglu.accum_diag(
                    flat, self.down_accum, self.down_count
                )
        return self.down_proj(hidden)


class LlamaBlock(nn.Module):
    def __init__(self, config: ModelConfig, down_mode: str, enable_stats: bool) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.attn = LlamaAttention(config, enable_stats)
        self.mlp_norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.mlp = SwiGLU(config, down_mode, enable_stats)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        precond_flag: bool,
        xtx_tmp_d: Tensor,
        xtx_tmp_ff: Tensor,
    ) -> Tensor:
        x = x + self.attn(
            self.attn_norm(x), cos, sin, precond_flag, xtx_tmp_d
        )
        x = x + self.mlp(
            self.mlp_norm(x), precond_flag, xtx_tmp_d, xtx_tmp_ff
        )
        return x


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig, method: str) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"unsupported method {method!r}")
        self.config = config
        self.method = method
        enable_stats = method in NEWTON_METHODS
        down_mode = {
            "newton_full": "dense",
            "down_diag": "diag",
            "down_none": "none",
        }.get(method, "none")
        self.down_mode = down_mode
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList(
            [LlamaBlock(config, down_mode, enable_stats) for _ in range(config.n_layer)]
        )
        self.norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        inv_freq = 1.0 / (
            config.rope_base
            ** (torch.arange(0, config.n_embd // config.n_head, 2).float()
                / (config.n_embd // config.n_head))
        )
        positions = torch.arange(config.sequence_length).float()
        frequencies = torch.outer(positions, inv_freq)
        self.register_buffer("rope_cos", frequencies.cos(), persistent=True)
        self.register_buffer("rope_sin", frequencies.sin(), persistent=True)
        self.register_buffer(
            "xtx_tmp_d", torch.empty(config.n_embd, config.n_embd), persistent=False
        )
        ff_tmp_size = config.intermediate_size if down_mode == "dense" else 1
        self.register_buffer(
            "xtx_tmp_ff", torch.empty(ff_tmp_size, ff_tmp_size), persistent=False
        )
        self.apply(self._initialize_weights)
        # Tie only after initialization so the shared tensor is initialized
        # exactly once rather than being revisited through two modules.
        self.lm_head.weight = self.tok_embeddings.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        return_logits: bool = True,
        precond_flag: bool = False,
    ) -> tuple[Tensor | None, Tensor | None]:
        if idx.size(1) > self.config.sequence_length:
            raise ValueError("input exceeds configured sequence length")
        precond_flag = bool(precond_flag) and self.training
        length = idx.size(1)
        cos = self.rope_cos[:length][None, :, None, :]
        sin = self.rope_sin[:length][None, :, None, :]
        x = self.tok_embeddings(idx)
        for block in self.layers:
            x = block(
                x,
                cos,
                sin,
                precond_flag,
                self.xtx_tmp_d,
                self.xtx_tmp_ff,
            )
        x = self.norm(x)
        if targets is None:
            logits = self.lm_head(x[:, [-1], :]).float()
            loss = None
        else:
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        if not return_logits:
            logits = None
        return logits, loss

    def matrix_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.ndim == 2 and name != "tok_embeddings.weight"
        ]

    def backup_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        matrix_ids = {id(parameter) for _, parameter in self.matrix_named_parameters()}
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if id(parameter) not in matrix_ids
        ]

    def preconditioner_groups(self) -> list[dict[str, Any]]:
        if self.method not in NEWTON_METHODS:
            return []
        groups: list[dict[str, Any]] = []
        for layer_index, block in enumerate(self.layers):
            groups.extend(
                [
                    {
                        "name": f"layers.{layer_index}.attn_input",
                        "kind": "dense",
                        "members": [
                            block.attn.q_proj.weight,
                            block.attn.k_proj.weight,
                            block.attn.v_proj.weight,
                        ],
                        "accum": block.attn.attn_in_accum,
                        "count": block.attn.attn_in_count,
                    },
                    {
                        "name": f"layers.{layer_index}.attn_output",
                        "kind": "dense",
                        "members": [block.attn.o_proj.weight],
                        "accum": block.attn.o_accum,
                        "count": block.attn.o_count,
                    },
                    {
                        "name": f"layers.{layer_index}.mlp_input",
                        "kind": "dense",
                        "members": [
                            block.mlp.gate_proj.weight,
                            block.mlp.up_proj.weight,
                        ],
                        "accum": block.mlp.mlp_in_accum,
                        "count": block.mlp.mlp_in_count,
                    },
                ]
            )
            if self.down_mode != "none":
                groups.append(
                    {
                        "name": f"layers.{layer_index}.down_input",
                        "kind": self.down_mode,
                        "members": [block.mlp.down_proj.weight],
                        "accum": block.mlp.down_accum,
                        "count": block.mlp.down_count,
                    }
                )
        return groups


def architecture_audit(model: LlamaForCausalLM) -> dict[str, Any]:
    config = model.config
    named = list(model.named_parameters())
    matrix = model.matrix_named_parameters()
    backup = model.backup_named_parameters()
    group_rows = []
    for group in model.preconditioner_groups():
        width = int(group["members"][0].shape[1])
        group_rows.append(
            {
                "name": group["name"],
                "kind": group["kind"],
                "input_width": width,
                "member_shapes": [list(parameter.shape) for parameter in group["members"]],
            }
        )
    return {
        "architecture": "llama_swiglu_parameter_matched_r1",
        "config": asdict(config),
        "parameter_count": int(sum(parameter.numel() for _, parameter in named)),
        "matrix_parameter_count": int(sum(parameter.numel() for _, parameter in matrix)),
        "backup_parameter_count": int(sum(parameter.numel() for _, parameter in backup)),
        "named_parameter_count": len(named),
        "matrix_tensor_count": len(matrix),
        "backup_tensor_count": len(backup),
        "embedding_head_tied": model.lm_head.weight is model.tok_embeddings.weight,
        "bias_parameter_count": int(
            sum(parameter.numel() for name, parameter in named if name.endswith(".bias"))
        ),
        "rmsnorm": "learned_gain_eps_1e-6",
        "rope": "base_10000_precomputed",
        "qkv": "separate_q_k_v_projections",
        "mlp": "swiglu_gate_up_down",
        "down_mode": model.down_mode,
        "preconditioner_groups": group_rows,
        "preconditioner_group_count": len(group_rows),
    }


# -----------------------------------------------------------------------------
# Reference-aligned Muon and generalized shared-input Newton-Muon.


@torch.compile
def zeropower_via_newtonschulz5(gradient: Tensor, steps: int = 5) -> Tensor:
    if XXT is None or ba_plus_cAA is None:
        raise RuntimeError(f"triton_kernels import failed: {TRITON_KERNEL_IMPORT_ERROR!r}")
    a, b, c = 3.4445, -4.7750, 2.0315
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


class ReferenceMuon(torch.optim.Optimizer):
    """Public/reference Muon conventions used by the corrected 60-run line."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _pre_step(self) -> None:
        return None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._pre_step()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta = float(group["momentum"])
            steps = int(group["ns_steps"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2:
                    raise ValueError("ReferenceMuon accepts hidden 2D matrices only")
                gradient = parameter.grad
                state = self.state[parameter]
                momentum = state.get("momentum")
                if momentum is None:
                    momentum = state["momentum"] = torch.zeros_like(
                        parameter, dtype=torch.float32
                    )
                # Public EMA-form momentum and Nesterov lookahead.
                momentum.lerp_(gradient, 1.0 - beta)
                update = torch.lerp(gradient, momentum, beta)
                update = zeropower_via_newtonschulz5(update, steps=steps)
                rows, cols = parameter.shape
                shape_scale = math.sqrt(max(1.0, float(rows) / float(cols)))
                parameter.add_(update, alpha=-lr * shape_scale)
        return loss


class SharedInputNewtonMuon(ReferenceMuon):
    """Muon with one right preconditioner per distinct activation source.

    Q/K/V share one K because their right-hand input is identical. Gate/up do
    the same. This is mathematically equivalent to duplicating identical K
    tensors but avoids dishonest state inflation. O and down each own a group.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        preconditioner_groups: list[dict[str, Any]],
        lr: float,
        momentum: float = 0.95,
        ns_steps: int = 5,
        beta: float = 0.95,
        ridge: float = 0.2,
        refresh: int = 32,
        init_scale: float = 0.001,
    ) -> None:
        super().__init__(params, lr=lr, momentum=momentum, ns_steps=ns_steps)
        self.input_beta = float(beta)
        self.input_ridge = float(ridge)
        self.refresh = int(refresh)
        self.init_scale = float(init_scale)
        self.global_step = 0
        self._groups = preconditioner_groups
        self._grad_workspaces: dict[tuple[int, int, str, int], Tensor] = {}
        for spec in self._groups:
            owner = spec["members"][0]
            width = int(owner.shape[1])
            state = self.state[owner]
            state["precond_kind"] = spec["kind"]
            state["precond_group_name"] = spec["name"]
            if spec["kind"] == "dense":
                covariance = torch.zeros(
                    width, width, device=owner.device, dtype=torch.float32
                )
                covariance.diagonal().fill_(self.init_scale)
                inverse = torch.eye(width, device=owner.device, dtype=torch.float32)
            elif spec["kind"] == "diag":
                covariance = torch.full(
                    (width,), self.init_scale, device=owner.device, dtype=torch.float32
                )
                inverse = torch.ones(width, device=owner.device, dtype=torch.float32)
            else:
                raise ValueError(f"unsupported preconditioner kind {spec['kind']!r}")
            state["precond_cov"] = covariance
            state["precond_inv_apply"] = inverse

    def precond_flag_for_step(self, step: int) -> bool:
        return (int(step) + 1) % self.refresh == 0

    def state_dict(self):
        payload = super().state_dict()
        payload["shared_input_newton"] = {"global_step": int(self.global_step)}
        return payload

    def load_state_dict(self, state_dict):
        extra = state_dict.pop("shared_input_newton", {"global_step": 0})
        result = super().load_state_dict(state_dict)
        self.global_step = int(extra.get("global_step", 0))
        return result

    @torch.no_grad()
    def _refresh_preconditioners(self) -> None:
        update_weight = 1.0 - self.input_beta
        for spec in self._groups:
            count = spec["count"]
            if float(count.item()) <= 0.0:
                raise RuntimeError(
                    f"preconditioner refresh has no activations for {spec['name']}"
                )
            owner = spec["members"][0]
            state = self.state[owner]
            covariance = state["precond_cov"]
            covariance.lerp_(spec["accum"] / count, update_weight)
            if spec["kind"] == "dense":
                work = covariance.clone()
                diagonal = work.diagonal()
                ridge = diagonal.mean() * self.input_ridge + 1e-8
                diagonal.add_(ridge)
                factor, info = torch.linalg.cholesky_ex(
                    work, upper=False, check_errors=False
                )
                if int(info.item()) != 0:
                    raise FloatingPointError(
                        f"Cholesky failed for {spec['name']} with info={int(info.item())}"
                    )
                torch.cholesky_inverse(
                    factor, upper=False, out=state["precond_inv_apply"]
                )
            else:
                ridge = covariance.mean() * self.input_ridge + 1e-8
                state["precond_inv_apply"].copy_(
                    torch.reciprocal(covariance + ridge)
                )
            spec["accum"].zero_()
            spec["count"].zero_()

    def _workspace_for(self, gradient: Tensor) -> Tensor:
        index = gradient.device.index if gradient.device.index is not None else -1
        key = (gradient.size(0), gradient.size(1), gradient.device.type, index)
        workspace = self._grad_workspaces.get(key)
        if workspace is None:
            workspace = torch.empty_like(gradient, dtype=torch.float32)
            self._grad_workspaces[key] = workspace
        return workspace

    @torch.no_grad()
    def _apply_preconditioners(self) -> None:
        for spec in self._groups:
            owner = spec["members"][0]
            inverse = self.state[owner]["precond_inv_apply"]
            for parameter in spec["members"]:
                if parameter.grad is None:
                    continue
                if spec["kind"] == "dense":
                    workspace = self._workspace_for(parameter.grad)
                    torch.mm(parameter.grad, inverse, out=workspace)
                    parameter.grad.copy_(workspace)
                else:
                    parameter.grad.mul_(inverse)

    @torch.no_grad()
    def _pre_step(self) -> None:
        if self.precond_flag_for_step(self.global_step):
            self._refresh_preconditioners()
        self._apply_preconditioners()
        self.global_step += 1

    def memory_audit(self) -> dict[str, int]:
        covariance: list[Tensor] = []
        inverse: list[Tensor] = []
        for spec in self._groups:
            state = self.state[spec["members"][0]]
            covariance.append(state["precond_cov"])
            inverse.append(state["precond_inv_apply"])
        return {
            "k_cov_bytes": unique_storage_bytes(covariance),
            "k_inv_bytes": unique_storage_bytes(inverse),
            "k_state_bytes": unique_storage_bytes([*covariance, *inverse]),
            "preconditioner_workspace_bytes": unique_storage_bytes(
                self._grad_workspaces.values()
            ),
        }


# -----------------------------------------------------------------------------
# FineWeb loader and exact checkpoint/resume.


def peek_data_shard(filename: Path) -> int:
    with filename.open("rb") as handle:
        header = np.frombuffer(handle.read(256 * 4), dtype=np.int32)
    if len(header) != 256 or header[0] != 20240520 or header[1] != 1:
        raise ValueError(f"invalid FineWeb shard header: {filename}")
    return int(header[2])


def load_data_shard(filename: Path) -> np.ndarray:
    with filename.open("rb") as handle:
        header = np.frombuffer(handle.read(256 * 4), dtype=np.int32)
        if len(header) != 256 or header[0] != 20240520 or header[1] != 1:
            raise ValueError(f"invalid FineWeb shard header: {filename}")
        expected = int(header[2])
        tokens = np.frombuffer(handle.read(), dtype=np.uint16)
    if len(tokens) != expected:
        raise ValueError(f"token count mismatch in {filename}: {len(tokens)} != {expected}")
    return tokens


class SequentialShardLoader:
    def __init__(self, pattern: str, batch_size: int, sequence_length: int) -> None:
        import glob

        self.files = [Path(value).resolve() for value in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"no data files match {pattern}")
        self.batch_size = int(batch_size)
        self.sequence_length = int(sequence_length)
        minimum = self.batch_size * self.sequence_length + 1
        self.total_tokens = 0
        for filename in self.files:
            tokens = peek_data_shard(filename)
            if tokens < minimum:
                raise ValueError(f"shard too small for one batch: {filename}")
            self.total_tokens += tokens
        self.current_shard = 0
        self.current_position = 0
        self.tokens = load_data_shard(self.files[0])

    def reset(self) -> None:
        self.current_shard = 0
        self.current_position = 0
        self.tokens = load_data_shard(self.files[0])

    def advance(self) -> None:
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = load_data_shard(self.files[self.current_shard])

    def next_batch(self) -> tuple[Tensor, Tensor]:
        count = self.batch_size * self.sequence_length
        buffer = self.tokens[self.current_position : self.current_position + count + 1]
        if len(buffer) != count + 1:
            raise RuntimeError("loader position invariant failed")
        tensor = torch.tensor(buffer.astype(np.int32), dtype=torch.long)
        x = tensor[:-1].view(self.batch_size, self.sequence_length)
        y = tensor[1:].view(self.batch_size, self.sequence_length)
        self.current_position += count
        if self.current_position + count + 1 > len(self.tokens):
            self.advance()
        return x.cuda(non_blocking=True), y.cuda(non_blocking=True)

    def state_dict(self) -> dict[str, int]:
        return {
            "current_shard": int(self.current_shard),
            "current_position": int(self.current_position),
        }

    def load_state_dict(self, payload: dict[str, int]) -> None:
        shard = int(payload["current_shard"])
        position = int(payload["current_position"])
        if not 0 <= shard < len(self.files):
            raise ValueError("checkpoint loader shard is out of range")
        self.current_shard = shard
        self.current_position = position
        self.tokens = load_data_shard(self.files[shard])


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    torch.cuda.set_rng_state_all(payload["torch_cuda"])


def trim_metrics(path: Path, completed_steps: int) -> None:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # A checkpoint is written immediately after the training update and before
    # the next loop's validation. If validation at the same step reached disk
    # after that checkpoint, it must be recomputed once after resume rather
    # than retained and duplicated.
    kept = [
        row
        for row in rows
        if int(row["step"]) < completed_steps
        or (int(row["step"]) == completed_steps and row["event"] == "train")
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(kept)


def append_metric(path: Path, row: dict[str, Any]) -> None:
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def optimizer_state_bytes(optimizers: Iterable[torch.optim.Optimizer]) -> int:
    tensors: list[Tensor] = []
    for optimizer in optimizers:
        tensors.extend(iter_tensors(optimizer.state))
    return unique_storage_bytes(tensors)


def activation_state_memory(model: LlamaForCausalLM) -> dict[str, int]:
    stats: list[Tensor] = []
    scratch: list[Tensor] = []
    for name, tensor in model.named_buffers():
        if name.endswith("_accum") or name.endswith("_count"):
            stats.append(tensor)
        elif name.startswith("xtx_tmp"):
            scratch.append(tensor)
    return {
        "activation_stat_bytes": unique_storage_bytes(stats),
        "activation_scratch_bytes": unique_storage_bytes(scratch),
    }


# -----------------------------------------------------------------------------
# Training entry point.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-iterations", type=int, default=6200)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--device-batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-tokens", type=int, default=10485760)
    parser.add_argument("--warmdown-iters", type=int, default=1800)
    parser.add_argument("--backup-lr", type=float, default=0.0036)
    parser.add_argument("--matrix-lr", type=float, default=0.01)
    parser.add_argument("--adamw-matrix-lr", type=float, default=0.000576)
    parser.add_argument("--checkpoint-every", type=int, default=128)
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--no-save-final", action="store_true")
    args = parser.parse_args()
    if args.num_iterations <= 0:
        parser.error("--num-iterations must be positive")
    if args.global_batch_size % args.device_batch_size != 0:
        parser.error("global batch must be divisible by device batch")
    batch_tokens = args.device_batch_size * args.sequence_length
    if args.val_tokens % batch_tokens != 0:
        parser.error("val tokens must be divisible by device_batch_size * sequence_length")
    if args.warmdown_iters < 0 or args.warmdown_iters > args.num_iterations:
        parser.error("invalid warmdown length")
    return args


def lr_multiplier(step: int, total: int, warmdown: int) -> float:
    if warmdown == 0 or step < total - warmdown:
        return 1.0
    return max(0.0, float(total - step) / float(warmdown))


def make_optimizers(
    model: LlamaForCausalLM, args: argparse.Namespace
) -> tuple[list[torch.optim.Optimizer], torch.optim.Optimizer | None]:
    backup = [parameter for _, parameter in model.backup_named_parameters()]
    matrix = [parameter for _, parameter in model.matrix_named_parameters()]
    backup_optimizer = torch.optim.AdamW(
        backup,
        lr=args.backup_lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    if args.method == "adamw":
        matrix_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            matrix,
            lr=args.adamw_matrix_lr,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            fused=True,
        )
    elif args.method == "muon":
        matrix_optimizer = ReferenceMuon(matrix, lr=args.matrix_lr)
    else:
        matrix_optimizer = SharedInputNewtonMuon(
            matrix,
            model.preconditioner_groups(),
            lr=args.matrix_lr,
        )
    return [backup_optimizer, matrix_optimizer], matrix_optimizer


def runtime_metadata() -> dict[str, Any]:
    import triton
    import triton_kernels

    gpu = torch.cuda.get_device_properties(0)
    kernel_path = Path(triton_kernels.__file__).resolve()
    return {
        # Preserve the virtualenv entry point instead of dereferencing its
        # common symlink to /usr/bin/python.
        "python_executable": str(Path(sys.executable).absolute()),
        "python_version": list(sys.version_info[:3]),
        "python_full": sys.version.replace("\n", " "),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "triton": str(triton.__version__),
        "triton_module": str(Path(triton.__file__).resolve()),
        "triton_kernels_module": str(kernel_path),
        "triton_kernels_sha256": sha256_file(kernel_path),
        "gpu_name": gpu.name,
        "gpu_total_memory_bytes": int(gpu.total_memory),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
    }


def save_checkpoint(
    path: Path,
    raw_model: LlamaForCausalLM,
    optimizers: list[torch.optim.Optimizer],
    train_loader: SequentialShardLoader,
    x: Tensor,
    y: Tensor,
    completed_steps: int,
    train_s: float,
    steady_train_s: float,
    steady_steps: int,
    peak_allocated_bytes: int,
    resume_count: int,
    init_sha256: str,
) -> None:
    torch.cuda.synchronize()
    payload = {
        "format_version": 1,
        "completed_steps": int(completed_steps),
        "model": raw_model.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "train_loader": train_loader.state_dict(),
        "next_x": x.detach().cpu(),
        "next_y": y.detach().cpu(),
        "rng": capture_rng_state(),
        "train_s": float(train_s),
        "steady_train_s": float(steady_train_s),
        "steady_steps": int(steady_steps),
        "peak_allocated_bytes": int(peak_allocated_bytes),
        "resume_count": int(resume_count),
        "init_sha256": init_sha256,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_metric_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = ModelConfig(sequence_length=args.sequence_length)
    model_cpu = LlamaForCausalLM(config, args.method)
    init_sha256 = hash_named_parameters(model_cpu)
    audit = architecture_audit(model_cpu)
    init_payload = {
        "event": "init_audit",
        "method": args.method,
        "seed": args.seed,
        "init_sha256": init_sha256,
        "architecture": audit,
    }
    print("LLAMA_INIT_AUDIT " + json.dumps(init_payload, sort_keys=True), flush=True)
    if args.init_only:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LLaMA/SwiGLU training")
    if TRITON_KERNEL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "the pinned official triton_kernels.py is required: "
            f"{TRITON_KERNEL_IMPORT_ERROR!r}"
        )
    torch.cuda.set_device(0)
    torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    summary_path = output_dir / "summary.json"
    status_path = output_dir / "status.json"
    atomic_write_json(
        status_path,
        {
            "status": "starting",
            "method": args.method,
            "seed": args.seed,
            "init_sha256": init_sha256,
        },
    )

    data_dir = args.data_dir.resolve()
    train_loader = SequentialShardLoader(
        str(data_dir / "fineweb_train_*.bin"),
        args.device_batch_size,
        args.sequence_length,
    )
    val_loader = SequentialShardLoader(
        str(data_dir / "fineweb_val_*.bin"),
        args.device_batch_size,
        args.sequence_length,
    )
    raw_model = model_cpu.cuda()
    optimizers, matrix_optimizer = make_optimizers(raw_model, args)
    compiled_model = torch.compile(raw_model)
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    completed_steps = 0
    train_s = 0.0
    steady_train_s = 0.0
    steady_steps = 0
    peak_allocated_bytes = 0
    resume_count = 0
    if args.resume == "auto" and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        if checkpoint.get("init_sha256") != init_sha256:
            raise RuntimeError("checkpoint initialization fingerprint does not match")
        raw_model.load_state_dict(checkpoint["model"])
        for optimizer, state in zip(optimizers, checkpoint["optimizers"]):
            optimizer.load_state_dict(state)
        train_loader.load_state_dict(checkpoint["train_loader"])
        x = checkpoint["next_x"].cuda(non_blocking=True)
        y = checkpoint["next_y"].cuda(non_blocking=True)
        completed_steps = int(checkpoint["completed_steps"])
        train_s = float(checkpoint["train_s"])
        steady_train_s = float(checkpoint["steady_train_s"])
        steady_steps = int(checkpoint["steady_steps"])
        peak_allocated_bytes = int(checkpoint["peak_allocated_bytes"])
        resume_count = int(checkpoint.get("resume_count", 0)) + 1
        restore_rng_state(checkpoint["rng"])
        trim_metrics(metrics_path, completed_steps)
        print(f"LLAMA_RESUME completed_steps={completed_steps}", flush=True)
    else:
        x, y = train_loader.next_batch()

    if completed_steps > args.num_iterations:
        raise RuntimeError("checkpoint is beyond requested num_iterations")
    train_accumulation_steps = args.global_batch_size // args.device_batch_size
    val_steps = args.val_tokens // (args.device_batch_size * args.sequence_length)
    torch.cuda.reset_peak_memory_stats()
    atomic_write_json(
        status_path,
        {
            "status": "training",
            "method": args.method,
            "seed": args.seed,
            "completed_steps": completed_steps,
            "resume_count": resume_count,
            "runtime": runtime_metadata(),
        },
    )

    while True:
        if completed_steps % args.val_every == 0 or completed_steps == args.num_iterations:
            raw_model.eval()
            val_loader.reset()
            val_loss = torch.zeros((), device="cuda", dtype=torch.float32)
            for _ in range(val_steps):
                x_val, y_val = val_loader.next_batch()
                with torch.no_grad(), autocast:
                    _, loss = compiled_model(
                        x_val,
                        y_val,
                        return_logits=False,
                        precond_flag=False,
                    )
                assert loss is not None
                val_loss += loss.detach()
            val_value = float((val_loss / val_steps).item())
            if not math.isfinite(val_value):
                raise FloatingPointError(f"non-finite validation loss at {completed_steps}")
            avg_ms = (
                1000.0 * steady_train_s / steady_steps if steady_steps > 0 else float("nan")
            )
            append_metric(
                metrics_path,
                {
                    "event": "val",
                    "step": completed_steps,
                    "loss": f"{val_value:.9f}",
                    "train_s": f"{train_s:.9f}",
                    "steady_train_s": f"{steady_train_s:.9f}",
                    "step_avg_ms": f"{avg_ms:.6f}",
                    "lr_backup": f"{optimizers[0].param_groups[0]['lr']:.12g}",
                    "lr_matrix": f"{optimizers[1].param_groups[0]['lr']:.12g}",
                    "tokens_seen": completed_steps
                    * args.global_batch_size
                    * args.sequence_length,
                },
            )
            print(
                f"step:{completed_steps}/{args.num_iterations} val_loss:{val_value:.4f} "
                f"train_time:{train_s * 1000:.0f}ms step_avg:{avg_ms:.2f}ms",
                flush=True,
            )

        if completed_steps == args.num_iterations:
            break

        multiplier = lr_multiplier(
            completed_steps, args.num_iterations, args.warmdown_iters
        )
        optimizers[0].param_groups[0]["lr"] = args.backup_lr * multiplier
        matrix_base = args.adamw_matrix_lr if args.method == "adamw" else args.matrix_lr
        optimizers[1].param_groups[0]["lr"] = matrix_base * multiplier
        if isinstance(matrix_optimizer, SharedInputNewtonMuon):
            matrix_optimizer.global_step = completed_steps
            precond_flag = matrix_optimizer.precond_flag_for_step(completed_steps)
        else:
            precond_flag = False

        raw_model.train()
        torch.cuda.synchronize()
        update_start = time.perf_counter()
        for _ in range(train_accumulation_steps):
            with autocast:
                _, loss = compiled_model(
                    x,
                    y,
                    return_logits=False,
                    precond_flag=precond_flag,
                )
                assert loss is not None
                train_loss = loss.detach()
                scaled_loss = loss / train_accumulation_steps
            x, y = train_loader.next_batch()
            scaled_loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        raw_model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        update_s = time.perf_counter() - update_start
        train_s += update_s
        completed_steps += 1
        if completed_steps > 32:
            steady_train_s += update_s
            steady_steps += 1
        loss_value = float(train_loss.item())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"non-finite training loss at {completed_steps}")
        avg_ms = (
            1000.0 * steady_train_s / steady_steps if steady_steps > 0 else float("nan")
        )
        append_metric(
            metrics_path,
            {
                "event": "train",
                "step": completed_steps,
                "loss": f"{loss_value:.9f}",
                "train_s": f"{train_s:.9f}",
                "steady_train_s": f"{steady_train_s:.9f}",
                "step_avg_ms": f"{avg_ms:.6f}",
                "lr_backup": f"{optimizers[0].param_groups[0]['lr']:.12g}",
                "lr_matrix": f"{optimizers[1].param_groups[0]['lr']:.12g}",
                "tokens_seen": completed_steps
                * args.global_batch_size
                * args.sequence_length,
            },
        )
        print(
            f"step:{completed_steps}/{args.num_iterations} train_loss:{loss_value:.4f} "
            f"train_time:{train_s * 1000:.0f}ms step_avg:{avg_ms:.2f}ms",
            flush=True,
        )
        peak_allocated_bytes = max(
            peak_allocated_bytes, int(torch.cuda.max_memory_allocated())
        )
        should_checkpoint = args.checkpoint_every > 0 and (
            completed_steps % args.checkpoint_every == 0
        )
        if should_checkpoint:
            save_checkpoint(
                checkpoint_path,
                raw_model,
                optimizers,
                train_loader,
                x,
                y,
                completed_steps,
                train_s,
                steady_train_s,
                steady_steps,
                peak_allocated_bytes,
                resume_count,
                init_sha256,
            )
            atomic_write_json(
                status_path,
                {
                    "status": "training",
                    "method": args.method,
                    "seed": args.seed,
                    "completed_steps": completed_steps,
                    "resume_count": resume_count,
                    "checkpoint": str(checkpoint_path),
                },
            )

    peak_allocated_bytes = max(
        peak_allocated_bytes, int(torch.cuda.max_memory_allocated())
    )
    if not args.no_save_final:
        save_checkpoint(
            checkpoint_path,
            raw_model,
            optimizers,
            train_loader,
            x,
            y,
            completed_steps,
            train_s,
            steady_train_s,
            steady_steps,
            peak_allocated_bytes,
            resume_count,
            init_sha256,
        )

    memory = activation_state_memory(raw_model)
    if isinstance(matrix_optimizer, SharedInputNewtonMuon):
        memory.update(matrix_optimizer.memory_audit())
    else:
        memory.update(
            {
                "k_cov_bytes": 0,
                "k_inv_bytes": 0,
                "k_state_bytes": 0,
                "preconditioner_workspace_bytes": 0,
            }
        )
    rows = load_metric_rows(metrics_path)
    val_rows = [row for row in rows if row["event"] == "val"]
    train_rows = [row for row in rows if row["event"] == "train"]
    summary = {
        "status": "completed",
        "method": args.method,
        "seed": args.seed,
        "completed_steps": completed_steps,
        "tokens_seen": completed_steps * args.global_batch_size * args.sequence_length,
        "init_sha256": init_sha256,
        "architecture": audit,
        "runtime": runtime_metadata(),
        "config": vars(args) | {"data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
        "final_val_loss": float(val_rows[-1]["loss"]),
        "best_val_loss": min(float(row["loss"]) for row in val_rows),
        "final_train_loss": float(train_rows[-1]["loss"]),
        "train_s": train_s,
        "steady_train_s": steady_train_s,
        "steady_steps": steady_steps,
        "step_avg_ms": 1000.0 * steady_train_s / max(1, steady_steps),
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_allocated_mib": peak_allocated_bytes / (1024**2),
        "model_parameter_bytes": int(
            sum(parameter.numel() * parameter.element_size() for parameter in raw_model.parameters())
        ),
        "optimizer_state_bytes": optimizer_state_bytes(optimizers),
        "resume_count": resume_count,
        "timing_comparable": resume_count == 0,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.is_file() else "",
        "metrics_path": str(metrics_path),
        **memory,
    }
    atomic_write_json(summary_path, summary)
    atomic_write_json(
        status_path,
        {
            "status": "completed",
            "method": args.method,
            "seed": args.seed,
            "completed_steps": completed_steps,
            "summary": str(summary_path),
            "resume_count": resume_count,
        },
    )
    print("LLAMA_FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
