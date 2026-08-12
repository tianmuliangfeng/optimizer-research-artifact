"""Local quadratic-score diagnostics for mlp.c_proj update directions.

The probe is deliberately independent from optimizer state. At a fixed model
checkpoint it uses one diagnostic batch to build fresh activation second
moments, constructs several instantaneous Muon/Newton-Muon directions, and
compares their gradient alignment, exact Hessian-vector-product curvature, and
one-dimensional loss changes. None of the candidate directions is applied to
the training trajectory.
"""

from __future__ import annotations

import csv
import json
import math
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable

import torch

from optimizers import matrix_cosine, matrix_sign_ns5, matrix_sign_svd

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - fallback for older PyTorch releases
    SDPBackend = None
    sdpa_kernel = None


VALID_MODES = ("none", "scalar", "diag", "block4", "full")
NORMMATCH_SUFFIX = "_normmatch"
NONE_REPEAT_MODE = "none_repeat"


def parse_int_csv(value: str) -> list[int]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    result = [int(item) for item in items]
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate integer values in {value!r}")
    return result


def parse_float_csv(value: str) -> list[float]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    result = [float(item) for item in items]
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate float values in {value!r}")
    return result


def parse_mode_csv(value: str) -> list[str]:
    modes = [item.strip() for item in str(value).split(",") if item.strip()]
    unknown = [mode for mode in modes if mode not in VALID_MODES]
    if unknown:
        raise ValueError(f"unknown quadratic-probe modes {unknown}; valid modes are {VALID_MODES}")
    if len(modes) != len(set(modes)):
        raise ValueError(f"duplicate quadratic-probe modes in {value!r}")
    if "none" not in modes:
        raise ValueError("quadratic probe requires mode 'none' as its direction reference")
    return modes


def expanded_probe_modes(
    modes: list[str],
    *,
    include_none_repeat: bool,
    normmatch_modes: list[str],
) -> list[str]:
    """Return base modes plus deterministic measurement-control variants."""
    parse_mode_csv(",".join(modes))
    unknown = [mode for mode in normmatch_modes if mode not in modes]
    if unknown:
        raise ValueError(
            "norm-matched modes must also be present in base modes; "
            f"missing base modes: {unknown}"
        )
    invalid = [mode for mode in normmatch_modes if mode in ("none", "scalar")]
    if invalid:
        raise ValueError(
            "norm matching is only informative for non-scalar preconditioners; "
            f"got {invalid}"
        )
    if len(normmatch_modes) != len(set(normmatch_modes)):
        raise ValueError(f"duplicate norm-matched modes: {normmatch_modes}")
    result = list(modes)
    if include_none_repeat:
        result.append(NONE_REPEAT_MODE)
    result.extend(f"{mode}{NORMMATCH_SUFFIX}" for mode in normmatch_modes)
    return result


def _autocast_context(device_type: str, autocast_dtype: torch.dtype):
    if device_type == "cuda" and autocast_dtype in (torch.float16, torch.bfloat16):
        return torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
    return nullcontext()


@contextmanager
def _strict_float32_context(device_type: str, dtype: torch.dtype):
    """Disable TF32 when a probe is explicitly requested in float32."""
    if device_type != "cuda" or dtype != torch.float32:
        yield
        return
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


@contextmanager
def _math_sdp_context(device_type: str):
    """Use the math SDPA kernel because exact HVP needs double backward."""
    if sdpa_kernel is not None and SDPBackend is not None:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    elif device_type == "cuda" and hasattr(torch.backends.cuda, "sdp_kernel"):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        ):
            yield
    else:
        yield


def _fresh_covariance_stats(x2d: torch.Tensor) -> dict[str, float]:
    sample_count = int(x2d.shape[0])
    diag = torch.mean(x2d * x2d, dim=0)
    sample_gram = (x2d @ x2d.T) / sample_count
    k_fro_sq = torch.sum(sample_gram * sample_gram)
    diag_fro_sq = torch.sum(diag * diag)
    offdiag_fro_sq = torch.clamp(k_fro_sq - diag_fro_sq, min=0.0)
    k_fro = torch.sqrt(k_fro_sq).clamp_min(1e-30)
    return {
        "activation_samples": sample_count,
        "activation_features": int(x2d.shape[1]),
        "k_diag_mean": float(diag.mean().item()),
        "k_diag_min": float(diag.min().item()),
        "k_diag_max": float(diag.max().item()),
        "k_diag_condition": float((diag.max() / diag.min().clamp_min(1e-30)).item()),
        "k_fro_norm": float(k_fro.item()),
        "k_offdiag_fro_norm": float(torch.sqrt(offdiag_fro_sq).item()),
        "k_offdiag_fro_fraction": float((torch.sqrt(offdiag_fro_sq) / k_fro).item()),
    }


def _apply_low_rank_cov_inverse(
    g: torch.Tensor,
    x2d: torch.Tensor,
    damping: torch.Tensor,
) -> torch.Tensor:
    """Return g @ (x.T @ x / N + damping * I)^-1 via Woodbury.

    The result is mathematically identical to the dense solve, while the
    Cholesky factor is only N x N. This matters for the 4096-wide c_proj probe.
    """
    sample_count = int(x2d.shape[0])
    damping = damping.float().clamp_min(1e-12)
    sample_gram = (x2d @ x2d.T) / sample_count
    middle = torch.eye(sample_count, device=x2d.device, dtype=torch.float32)
    middle.add_(sample_gram / damping)
    chol = torch.linalg.cholesky(middle)
    solved_x = torch.cholesky_solve(x2d, chol)
    correction = (g @ x2d.T) @ solved_x
    correction.div_(sample_count * damping.square())
    return g / damping - correction


def _preconditioned_gradient(
    g: torch.Tensor,
    x2d: torch.Tensor,
    *,
    mode: str,
    ridge: float,
    blocks: int,
) -> torch.Tensor:
    diag = torch.mean(x2d * x2d, dim=0)
    mean_diag = diag.mean().clamp_min(1e-12)
    ridge_scale = mean_diag * float(ridge)

    if mode == "none":
        return g.clone()
    if mode == "scalar":
        return g / (mean_diag * (1.0 + float(ridge))).clamp_min(1e-12)
    if mode == "diag":
        return g * (diag + ridge_scale).clamp_min(1e-12).reciprocal().unsqueeze(0)
    if mode == "full":
        if ridge <= 0:
            raise ValueError("full fresh-K quadratic probe requires positive ridge")
        return _apply_low_rank_cov_inverse(g, x2d, ridge_scale)
    if mode == "block4":
        if blocks <= 0 or g.shape[1] % blocks != 0:
            raise ValueError(
                f"gradient width {g.shape[1]} must be divisible by blocks={blocks}"
            )
        block_width = g.shape[1] // blocks
        pieces = []
        for block_idx in range(blocks):
            start = block_idx * block_width
            end = start + block_width
            x_block = x2d[:, start:end]
            diag_block = torch.mean(x_block * x_block, dim=0)
            damping = diag_block.mean().clamp_min(1e-12) * float(ridge)
            if ridge <= 0:
                raise ValueError("block4 fresh-K quadratic probe requires positive ridge")
            pieces.append(_apply_low_rank_cov_inverse(g[:, start:end], x_block, damping))
        return torch.cat(pieces, dim=1)
    raise ValueError(f"unsupported quadratic-probe mode: {mode}")


def _row_orthogonality_residual(q: torch.Tensor) -> float:
    rows = int(q.shape[0])
    gram = q @ q.T
    eye = torch.eye(rows, device=q.device, dtype=gram.dtype)
    return float((torch.linalg.vector_norm(gram - eye) / math.sqrt(rows)).item())


def _projector_drift(q: torch.Tensor, reference: torch.Tensor) -> float:
    """Compute ||Q.T Q - R.T R||_F / ||R.T R||_F without n x n matrices."""
    qq = q @ q.T
    rr = reference @ reference.T
    qr = q @ reference.T
    distance_sq = torch.sum(qq * qq) + torch.sum(rr * rr) - 2.0 * torch.sum(qr * qr)
    reference_sq = torch.sum(rr * rr).clamp_min(1e-30)
    return float(torch.sqrt(torch.clamp(distance_sq, min=0.0) / reference_sq).item())


def _loss_value(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    device_type: str,
    autocast_dtype: torch.dtype,
) -> float:
    with torch.no_grad(), _math_sdp_context(device_type), _autocast_context(
        device_type, autocast_dtype
    ):
        _, loss = model(x, y)
    return float(loss.item())


def _line_search(
    model,
    param: torch.Tensor,
    candidates: dict[str, dict[str, torch.Tensor | float]],
    batches: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    step: int,
    build_repeat: int,
    layer: int,
    learning_rate: float,
    multipliers: list[float],
    device_type: str,
    autocast_dtype: torch.dtype,
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    original = param.detach().clone()
    base_losses = {
        split: _loss_value(
            model,
            batch[0],
            batch[1],
            device_type=device_type,
            autocast_dtype=autocast_dtype,
        )
        for split, batch in batches.items()
    }
    rows: list[dict] = []
    best: dict[str, dict[str, float]] = {}
    try:
        for mode, candidate in candidates.items():
            q = candidate["q"]
            if not isinstance(q, torch.Tensor):
                raise TypeError(f"candidate {mode} is missing tensor q")
            for split, (x, y) in batches.items():
                best_delta = 0.0
                best_eta = 0.0
                best_loss = base_losses[split]
                for multiplier in multipliers:
                    eta = float(multiplier) * float(learning_rate)
                    with torch.no_grad():
                        param.copy_(original - eta * q.to(dtype=param.dtype))
                    loss = _loss_value(
                        model,
                        x,
                        y,
                        device_type=device_type,
                        autocast_dtype=autocast_dtype,
                    )
                    delta = loss - base_losses[split]
                    rows.append(
                        {
                            "step": int(step),
                            "build_repeat": int(build_repeat),
                            "layer": int(layer),
                            "mode": mode,
                            "base_mode": candidate["base_mode"],
                            "direction_variant": candidate["direction_variant"],
                            "eval_split": split,
                            "eval_kind": (
                                "same" if split == "same" else "heldout"
                            ),
                            "heldout_index": (
                                -1
                                if split == "same"
                                else int(split.rsplit("_", 1)[1])
                            ),
                            "lr_multiplier": float(multiplier),
                            "eta": eta,
                            "base_loss": base_losses[split],
                            "loss": loss,
                            "loss_delta": delta,
                        }
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_eta = eta
                        best_loss = loss
                best[f"{mode}:{split}"] = {
                    "best_delta": best_delta,
                    "best_eta": best_eta,
                    "best_loss": best_loss,
                    "base_loss": base_losses[split],
                }
    finally:
        with torch.no_grad():
            param.copy_(original)
    return rows, best


def _target_module_and_param(model, layer: int):
    module_name = f"transformer.h.{layer}.mlp.c_proj"
    modules = dict(model.named_modules())
    if module_name not in modules:
        available = sorted(name for name in modules if name.endswith("mlp.c_proj"))
        raise KeyError(f"could not find {module_name}; available c_proj modules: {available}")
    module = modules[module_name]
    if not isinstance(module, torch.nn.Linear):
        raise TypeError(f"{module_name} is {type(module).__name__}, expected nn.Linear")
    return module_name, module, module.weight


def _probe_one_layer(
    model,
    batch: tuple[torch.Tensor, torch.Tensor],
    heldout_batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    step: int,
    build_repeat: int,
    layer: int,
    modes: list[str],
    include_none_repeat: bool,
    normmatch_modes: list[str],
    ridge: float,
    blocks: int,
    ns_steps: int,
    matrix_eps: float,
    matrix_learning_rate: float,
    line_search_multipliers: list[float],
    exact_hvp: bool,
    exact_svd: bool,
    line_search: bool,
    device_type: str,
    autocast_dtype: torch.dtype,
) -> tuple[list[dict], list[dict]]:
    x, y = batch
    module_name, module, param = _target_module_and_param(model, layer)
    activation_cache: dict[str, torch.Tensor] = {}

    def capture_input(_module, inputs):
        activation_cache["x"] = inputs[0].detach()

    handle = module.register_forward_pre_hook(capture_input)
    layer_start = time.perf_counter()
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad(), _math_sdp_context(device_type), _autocast_context(
            device_type, autocast_dtype
        ):
            _, loss = model(x, y)
        probe_loss_value = float(loss.detach().item())
        if "x" not in activation_cache:
            raise RuntimeError(f"activation hook for {module_name} did not fire")

        grad_graph = torch.autograd.grad(loss, param, create_graph=exact_hvp)[0]
        g = grad_graph.detach().float()
        x2d = activation_cache["x"].reshape(-1, param.shape[1]).float()
        cov_stats = _fresh_covariance_stats(x2d)

        candidates: dict[str, dict[str, torch.Tensor | float]] = {}
        candidate_start = time.perf_counter()
        for mode in modes:
            r = _preconditioned_gradient(
                g,
                x2d,
                mode=mode,
                ridge=ridge,
                blocks=blocks,
            )
            q = matrix_sign_ns5(r, steps=ns_steps, eps=matrix_eps).float().detach()
            candidate = {
                "q": q,
                "base_mode": mode,
                "direction_variant": "base",
                "r_fro_norm": float(torch.linalg.vector_norm(r).item()),
                "q_fro_norm": float(torch.linalg.vector_norm(q).item()),
                "preconditioned_cos_vs_g": matrix_cosine(r, g),
                "row_orthogonality_residual": _row_orthogonality_residual(q),
                "ns_svd_cos": math.nan,
                "exact_svd_computed": False,
            }
            if exact_svd:
                q_svd = matrix_sign_svd(r).float()
                candidate["ns_svd_cos"] = matrix_cosine(q, q_svd)
                candidate["exact_svd_computed"] = True
                del q_svd
            candidates[mode] = candidate
        del r
        del x2d
        candidate_seconds = time.perf_counter() - candidate_start

        q_reference = candidates["none"]["q"]
        if not isinstance(q_reference, torch.Tensor):
            raise TypeError("none candidate is missing tensor q")
        reference_norm = torch.linalg.vector_norm(q_reference).clamp_min(1e-30)
        if include_none_repeat:
            reference = candidates["none"]
            candidates[NONE_REPEAT_MODE] = {
                **reference,
                "q": q_reference.clone(),
                "base_mode": "none",
                "direction_variant": "repeat",
            }
        for mode in normmatch_modes:
            candidate = candidates[mode]
            q = candidate["q"]
            if not isinstance(q, torch.Tensor):
                raise TypeError(f"candidate {mode} is missing tensor q")
            q_norm = torch.linalg.vector_norm(q).clamp_min(1e-30)
            q_normmatched = q * (reference_norm / q_norm)
            candidates[f"{mode}{NORMMATCH_SUFFIX}"] = {
                **candidate,
                "q": q_normmatched,
                "base_mode": mode,
                "direction_variant": "normmatch",
                "q_fro_norm": float(
                    torch.linalg.vector_norm(q_normmatched).item()
                ),
                "row_orthogonality_residual": _row_orthogonality_residual(
                    q_normmatched
                ),
            }
        g_norm = torch.linalg.vector_norm(g).clamp_min(1e-30)
        rows: list[dict] = []

        hvp_start = time.perf_counter()
        candidate_items = list(candidates.items())
        for idx, (mode, candidate) in enumerate(candidate_items):
            q = candidate["q"]
            if not isinstance(q, torch.Tensor):
                raise TypeError(f"candidate {mode} is missing tensor q")
            q_norm = torch.linalg.vector_norm(q).clamp_min(1e-30)
            alignment_raw_tensor = torch.sum(g * q)
            alignment_raw = float(alignment_raw_tensor.item())
            alignment_norm = float((alignment_raw_tensor / (g_norm * q_norm)).item())

            curvature = math.nan
            if exact_hvp:
                directional_derivative = torch.sum(grad_graph.float() * q)
                hv = torch.autograd.grad(
                    directional_derivative,
                    param,
                    retain_graph=idx < len(candidate_items) - 1,
                    create_graph=False,
                )[0]
                curvature = float(torch.sum(hv.detach().float() * q).item())

            if (
                math.isfinite(curvature)
                and curvature > 0.0
                and alignment_raw > 0.0
            ):
                score = alignment_raw * alignment_raw / curvature
                eta_star = alignment_raw / curvature
                predicted_best_delta = -0.5 * score
            else:
                score = math.nan
                eta_star = math.nan
                predicted_best_delta = math.nan

            rows.append(
                {
                    "step": int(step),
                    "build_repeat": int(build_repeat),
                    "layer": int(layer),
                    "module_name": module_name,
                    "mode": mode,
                    "base_mode": candidate["base_mode"],
                    "direction_variant": candidate["direction_variant"],
                    "probe_loss": probe_loss_value,
                    "gradient_fro_norm": float(g_norm.item()),
                    "preconditioned_fro_norm": candidate["r_fro_norm"],
                    "direction_fro_norm": candidate["q_fro_norm"],
                    "preconditioned_norm_ratio": float(candidate["r_fro_norm"]) / float(g_norm.item()),
                    "preconditioned_cos_vs_g": candidate["preconditioned_cos_vs_g"],
                    "direction_cos_vs_none": matrix_cosine(q, q_reference),
                    "direction_norm_ratio_vs_none": float(
                        (q_norm / reference_norm).item()
                    ),
                    "projector_drift_vs_none": _projector_drift(q, q_reference),
                    "row_orthogonality_residual": candidate["row_orthogonality_residual"],
                    "ns_svd_cos": candidate["ns_svd_cos"],
                    "exact_svd_computed": candidate["exact_svd_computed"],
                    "alignment_raw": alignment_raw,
                    "alignment_normalized": alignment_norm,
                    "curvature_exact": curvature,
                    "curvature_per_direction_norm2": (
                        curvature / float(q_norm.square().item())
                        if math.isfinite(curvature)
                        else math.nan
                    ),
                    "quadratic_score_exact": score,
                    "eta_star_exact": eta_star,
                    "predicted_best_loss_delta": predicted_best_delta,
                    **cov_stats,
                }
            )
        hvp_seconds = time.perf_counter() - hvp_start

        # Explicitly release the second-order graph before the forward-only line search.
        del grad_graph
        del loss
        activation_cache.clear()
        model.zero_grad(set_to_none=True)

        line_rows: list[dict] = []
        line_best: dict[str, dict[str, float]] = {}
        line_start = time.perf_counter()
        if line_search:
            batches = {"same": batch}
            for heldout_index, heldout_batch in enumerate(heldout_batches):
                batches[f"heldout_{heldout_index}"] = heldout_batch
            line_rows, line_best = _line_search(
                model,
                param,
                candidates,
                batches,
                step=step,
                build_repeat=build_repeat,
                layer=layer,
                learning_rate=matrix_learning_rate,
                multipliers=line_search_multipliers,
                device_type=device_type,
                autocast_dtype=autocast_dtype,
            )
        line_seconds = time.perf_counter() - line_start

        total_seconds = time.perf_counter() - layer_start
        for row in rows:
            mode = row["mode"]
            same = line_best.get(f"{mode}:same", {})
            heldout = [
                line_best.get(f"{mode}:heldout_{index}", {})
                for index in range(len(heldout_batches))
            ]
            heldout_best_deltas = [
                float(item["best_delta"])
                for item in heldout
                if "best_delta" in item
            ]
            heldout_best_etas = [
                float(item["best_eta"])
                for item in heldout
                if "best_eta" in item
            ]
            heldout_eta1_deltas = [
                float(item["loss_delta"])
                for item in line_rows
                if item["mode"] == mode
                and item["eval_kind"] == "heldout"
                and math.isclose(float(item["lr_multiplier"]), 1.0)
            ]
            row.update(
                {
                    "line_search_same_best_delta": same.get("best_delta", math.nan),
                    "line_search_same_best_eta": same.get("best_eta", math.nan),
                    "line_search_heldout_best_delta": (
                        mean(heldout_best_deltas)
                        if heldout_best_deltas
                        else math.nan
                    ),
                    "line_search_heldout_best_eta": (
                        mean(heldout_best_etas)
                        if heldout_best_etas
                        else math.nan
                    ),
                    "line_search_heldout_best_delta_std": (
                        pstdev(heldout_best_deltas)
                        if heldout_best_deltas
                        else math.nan
                    ),
                    "line_search_heldout_eta1_delta_mean": (
                        mean(heldout_eta1_deltas)
                        if heldout_eta1_deltas
                        else math.nan
                    ),
                    "line_search_heldout_eta1_delta_std": (
                        pstdev(heldout_eta1_deltas)
                        if heldout_eta1_deltas
                        else math.nan
                    ),
                    "heldout_eval_batches": len(heldout_batches),
                    "candidate_seconds_layer": candidate_seconds,
                    "hvp_seconds_layer": hvp_seconds,
                    "line_search_seconds_layer": line_seconds,
                    "total_seconds_layer": total_seconds,
                }
            )
        return rows, line_rows
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def run_cproj_quadratic_probe_repeated(
    model,
    build_batches: list[tuple[torch.Tensor, torch.Tensor]],
    heldout_batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    step: int,
    layers: list[int],
    modes: list[str],
    include_none_repeat: bool,
    normmatch_modes: list[str],
    ridge: float,
    blocks: int,
    ns_steps: int,
    matrix_eps: float,
    matrix_learning_rate: float,
    line_search_multipliers: list[float],
    exact_hvp: bool,
    exact_svd: bool,
    exact_svd_repeats: int,
    line_search: bool,
    device_type: str,
    autocast_dtype: torch.dtype,
) -> tuple[list[dict], list[dict], dict]:
    if not build_batches:
        raise ValueError("quadratic probe needs at least one build batch")
    if not layers:
        raise ValueError("quadratic probe needs at least one layer")
    if not modes:
        raise ValueError("quadratic probe needs at least one mode")
    effective_modes = expanded_probe_modes(
        modes,
        include_none_repeat=include_none_repeat,
        normmatch_modes=normmatch_modes,
    )
    if any(multiplier < 0 for multiplier in line_search_multipliers):
        raise ValueError("line-search multipliers must be non-negative")
    if exact_svd_repeats < 0 or exact_svd_repeats > len(build_batches):
        raise ValueError(
            "exact_svd_repeats must be in [0, number of build batches]; "
            f"got {exact_svd_repeats} for {len(build_batches)} build batches"
        )
    if exact_svd and exact_svd_repeats == 0:
        raise ValueError("exact_svd=True requires exact_svd_repeats > 0")
    if not exact_svd and exact_svd_repeats != 0:
        raise ValueError("exact_svd=False requires exact_svd_repeats == 0")

    was_training = bool(model.training)
    if device_type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    probe_start = time.perf_counter()
    all_rows: list[dict] = []
    all_line_rows: list[dict] = []
    try:
        with _strict_float32_context(device_type, autocast_dtype):
            model.eval()
            for build_repeat, batch in enumerate(build_batches):
                for layer in layers:
                    rows, line_rows = _probe_one_layer(
                        model,
                        batch,
                        heldout_batches,
                        step=step,
                        build_repeat=build_repeat,
                        layer=layer,
                        modes=modes,
                        include_none_repeat=include_none_repeat,
                        normmatch_modes=normmatch_modes,
                        ridge=ridge,
                        blocks=blocks,
                        ns_steps=ns_steps,
                        matrix_eps=matrix_eps,
                        matrix_learning_rate=matrix_learning_rate,
                        line_search_multipliers=line_search_multipliers,
                        exact_hvp=exact_hvp,
                        exact_svd=(
                            exact_svd and build_repeat < exact_svd_repeats
                        ),
                        line_search=line_search,
                        device_type=device_type,
                        autocast_dtype=autocast_dtype,
                    )
                    all_rows.extend(rows)
                    all_line_rows.extend(line_rows)
                    if device_type == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
    finally:
        model.train(was_training)

    if device_type == "cuda":
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_mib = 0.0
    metadata = {
        "step": int(step),
        "layers": list(layers),
        "base_modes": list(modes),
        "modes": effective_modes,
        "build_repeats": len(build_batches),
        "heldout_batches": len(heldout_batches),
        "exact_svd_repeats": int(exact_svd_repeats),
        "probe_dtype": str(autocast_dtype).removeprefix("torch."),
        "direction_rows": len(all_rows),
        "line_search_rows": len(all_line_rows),
        "probe_seconds": time.perf_counter() - probe_start,
        "probe_peak_memory_mib": peak_mib,
    }
    return all_rows, all_line_rows, metadata


def run_cproj_quadratic_probe(
    model,
    batch: tuple[torch.Tensor, torch.Tensor],
    heldout_batch: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    step: int,
    layers: list[int],
    modes: list[str],
    ridge: float,
    blocks: int,
    ns_steps: int,
    matrix_eps: float,
    matrix_learning_rate: float,
    line_search_multipliers: list[float],
    exact_hvp: bool,
    exact_svd: bool,
    line_search: bool,
    device_type: str,
    autocast_dtype: torch.dtype,
) -> tuple[list[dict], list[dict], dict]:
    """Backward-compatible single-build/single-heldout P0 entry point."""
    rows, line_rows, metadata = run_cproj_quadratic_probe_repeated(
        model,
        [batch],
        [] if heldout_batch is None else [heldout_batch],
        step=step,
        layers=layers,
        modes=modes,
        include_none_repeat=False,
        normmatch_modes=[],
        ridge=ridge,
        blocks=blocks,
        ns_steps=ns_steps,
        matrix_eps=matrix_eps,
        matrix_learning_rate=matrix_learning_rate,
        line_search_multipliers=line_search_multipliers,
        exact_hvp=exact_hvp,
        exact_svd=exact_svd,
        exact_svd_repeats=1 if exact_svd else 0,
        line_search=line_search,
        device_type=device_type,
        autocast_dtype=autocast_dtype,
    )
    for row in line_rows:
        if row["eval_split"] == "heldout_0":
            row["eval_split"] = "heldout"
    return rows, line_rows, metadata


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _finite_values(rows: Iterable[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, math.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _summary_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((int(row["step"]), str(row["mode"])), []).append(row)
    summary = []
    metric_keys = (
        "alignment_normalized",
        "curvature_exact",
        "quadratic_score_exact",
        "direction_cos_vs_none",
        "projector_drift_vs_none",
        "row_orthogonality_residual",
        "line_search_same_best_delta",
        "line_search_heldout_best_delta",
        "line_search_heldout_eta1_delta_mean",
    )
    for (step, mode), group in sorted(groups.items()):
        item = {
            "step": step,
            "mode": mode,
            "observations": len(group),
            "build_repeats": len({int(row.get("build_repeat", 0)) for row in group}),
            "layers": len({int(row["layer"]) for row in group}),
        }
        for key in metric_keys:
            values = _finite_values(group, key)
            item[f"{key}_mean"] = mean(values) if values else math.nan
        summary.append(item)
    return summary


def _paired_line_rows(line_rows: list[dict]) -> list[dict]:
    none = {
        (
            int(row["step"]),
            int(row.get("build_repeat", 0)),
            int(row["layer"]),
            str(row["eval_split"]),
            float(row["lr_multiplier"]),
        ): float(row["loss_delta"])
        for row in line_rows
        if row["mode"] == "none"
    }
    none_repeat = {
        (
            int(row["step"]),
            int(row.get("build_repeat", 0)),
            int(row["layer"]),
            str(row["eval_split"]),
            float(row["lr_multiplier"]),
        ): float(row["loss_delta"])
        for row in line_rows
        if row["mode"] == NONE_REPEAT_MODE
    }
    result = []
    for row in line_rows:
        key = (
            int(row["step"]),
            int(row.get("build_repeat", 0)),
            int(row["layer"]),
            str(row["eval_split"]),
            float(row["lr_multiplier"]),
        )
        item = dict(row)
        item["loss_delta_vs_none"] = (
            float(row["loss_delta"]) - none[key] if key in none else math.nan
        )
        item["loss_delta_vs_none_repeat"] = (
            float(row["loss_delta"]) - none_repeat[key]
            if key in none_repeat
            else math.nan
        )
        result.append(item)
    return result


def _line_summary_rows(paired_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, str, str, float, float], list[dict]] = {}
    for row in paired_rows:
        key = (
            int(row["step"]),
            str(row["mode"]),
            str(row.get("eval_kind", row["eval_split"])),
            float(row["lr_multiplier"]),
            float(row["eta"]),
        )
        groups.setdefault(key, []).append(row)
    summary = []
    for (step, mode, eval_kind, multiplier, eta), group in sorted(
        groups.items()
    ):
        deltas = [float(row["loss_delta"]) for row in group]
        paired = _finite_values(group, "loss_delta_vs_none")
        summary.append(
            {
                "step": step,
                "mode": mode,
                "eval_kind": eval_kind,
                "lr_multiplier": multiplier,
                "eta": eta,
                "observations": len(group),
                "build_repeats": len(
                    {int(row.get("build_repeat", 0)) for row in group}
                ),
                "layers": len({int(row["layer"]) for row in group}),
                "eval_batches": len(
                    {
                        (
                            str(row["eval_split"]),
                            int(row.get("heldout_index", -1)),
                        )
                        for row in group
                    }
                ),
                "loss_delta_mean": mean(deltas),
                "loss_delta_std": pstdev(deltas),
                "loss_delta_median": median(deltas),
                "delta_vs_none_mean": mean(paired) if paired else math.nan,
                "delta_vs_none_std": pstdev(paired) if paired else math.nan,
                "better_than_none_count": sum(value < 0.0 for value in paired),
                "worse_than_none_count": sum(value > 0.0 for value in paired),
                "tied_with_none_count": sum(value == 0.0 for value in paired),
            }
        )
    return summary


def _quality_checks(
    rows: list[dict],
    line_rows: list[dict],
    *,
    expected_steps: list[int],
    expected_layers: list[int],
    expected_modes: list[str],
    expected_build_repeats: int,
    expected_heldout_batches: int,
    line_search_multipliers: list[float],
    exact_hvp: bool,
    exact_svd_repeats: int,
    include_none_repeat: bool,
    normmatch_modes: list[str],
    probe_precision: str,
    line_search: bool,
) -> list[dict]:
    expected_direction_rows = (
        len(expected_steps)
        * expected_build_repeats
        * len(expected_layers)
        * len(expected_modes)
    )
    expected_splits = 1 + expected_heldout_batches
    expected_line_rows = (
        expected_direction_rows * len(line_search_multipliers) * expected_splits
        if line_search
        else 0
    )
    direction_keys = {
        (
            int(row["step"]),
            int(row.get("build_repeat", 0)),
            int(row["layer"]),
            str(row["mode"]),
        )
        for row in rows
    }
    expected_keys = {
        (step, build_repeat, layer, mode)
        for step in expected_steps
        for build_repeat in range(expected_build_repeats)
        for layer in expected_layers
        for mode in expected_modes
    }
    finite_alignment = len(_finite_values(rows, "alignment_raw"))
    finite_curvature = len(_finite_values(rows, "curvature_exact"))
    scalar_cos = [
        float(row["direction_cos_vs_none"])
        for row in rows
        if row["mode"] == "scalar"
        and math.isfinite(float(row["direction_cos_vs_none"]))
    ]
    none_repeat_cos = [
        float(row["direction_cos_vs_none"])
        for row in rows
        if row["mode"] == NONE_REPEAT_MODE
        and math.isfinite(float(row["direction_cos_vs_none"]))
    ]
    normmatch_norm_errors = [
        abs(float(row["direction_norm_ratio_vs_none"]) - 1.0)
        for row in rows
        if str(row["mode"]).endswith(NORMMATCH_SUFFIX)
    ]
    line_index = {
        (
            int(row["step"]),
            int(row.get("build_repeat", 0)),
            int(row["layer"]),
            str(row["eval_split"]),
            float(row["lr_multiplier"]),
            str(row["mode"]),
        ): float(row["loss_delta"])
        for row in line_rows
    }

    def paired_line_gap(left_mode: str, right_mode: str) -> tuple[float, int]:
        gaps = []
        for key, left in line_index.items():
            if key[-1] != left_mode:
                continue
            right_key = (*key[:-1], right_mode)
            if right_key in line_index:
                gaps.append(abs(left - line_index[right_key]))
        return (max(gaps) if gaps else math.nan, len(gaps))

    none_repeat_gap, none_repeat_pairs = paired_line_gap(
        "none", NONE_REPEAT_MODE
    )
    scalar_none_gap, scalar_none_pairs = paired_line_gap("none", "scalar")
    finite_svd = len(_finite_values(rows, "ns_svd_cos"))
    expected_svd = (
        len(expected_steps)
        * exact_svd_repeats
        * len(expected_layers)
        * len(expected_modes)
    )
    checks = [
        {
            "check": "direction_key_coverage",
            "status": "pass" if direction_keys == expected_keys else "fail",
            "observed": len(direction_keys),
            "expected": len(expected_keys),
        },
        {
            "check": "direction_row_count",
            "status": "pass" if len(rows) == expected_direction_rows else "fail",
            "observed": len(rows),
            "expected": expected_direction_rows,
        },
        {
            "check": "finite_alignment",
            "status": "pass" if finite_alignment == len(rows) else "fail",
            "observed": finite_alignment,
            "expected": len(rows),
        },
        {
            "check": "finite_curvature",
            "status": (
                "pass"
                if (not exact_hvp or finite_curvature == len(rows))
                else "fail"
            ),
            "observed": finite_curvature,
            "expected": len(rows) if exact_hvp else 0,
        },
        {
            "check": "line_search_row_count",
            "status": "pass" if len(line_rows) == expected_line_rows else "fail",
            "observed": len(line_rows),
            "expected": expected_line_rows,
        },
        {
            "check": "scalar_none_instantaneous_direction",
            "status": (
                "pass"
                if (not scalar_cos or min(scalar_cos) >= 0.9999)
                else "fail"
            ),
            "observed": min(scalar_cos) if scalar_cos else "",
            "expected": ">=0.9999 when scalar is enabled",
        },
        {
            "check": "none_repeat_direction_exact",
            "status": (
                "pass"
                if (
                    not include_none_repeat
                    or (
                        len(none_repeat_cos)
                        == len(expected_steps)
                        * expected_build_repeats
                        * len(expected_layers)
                        and min(none_repeat_cos) >= 0.9999999
                    )
                )
                else "fail"
            ),
            "observed": min(none_repeat_cos) if none_repeat_cos else "",
            "expected": ">=0.9999999 when none_repeat is enabled",
        },
        {
            "check": "none_repeat_line_search_exact",
            "status": (
                "pass"
                if (
                    not line_search
                    or not include_none_repeat
                    or (
                        none_repeat_pairs
                        == len(expected_steps)
                        * expected_build_repeats
                        * len(expected_layers)
                        * expected_splits
                        * len(line_search_multipliers)
                        and none_repeat_gap <= 1e-7
                    )
                )
                else "fail"
            ),
            "observed": none_repeat_gap if none_repeat_pairs else "",
            "expected": "<=1e-7 max absolute loss-delta gap",
        },
        {
            "check": "scalar_none_line_search_gap",
            "status": (
                "pass"
                if (
                    not line_search
                    or scalar_none_pairs == 0
                    or probe_precision != "float32"
                    or scalar_none_gap <= 5e-4
                )
                else "fail"
            ),
            "observed": scalar_none_gap if scalar_none_pairs else "",
            "expected": "<=5e-4 for float32 P1",
        },
        {
            "check": "normmatched_direction_norm",
            "status": (
                "pass"
                if (
                    not normmatch_modes
                    or (
                        len(normmatch_norm_errors)
                        == len(expected_steps)
                        * expected_build_repeats
                        * len(expected_layers)
                        * len(normmatch_modes)
                        and max(normmatch_norm_errors) <= 1e-6
                    )
                )
                else "fail"
            ),
            "observed": (
                max(normmatch_norm_errors) if normmatch_norm_errors else ""
            ),
            "expected": "<=1e-6 maximum norm-ratio error",
        },
        {
            "check": "exact_svd_coverage",
            "status": "pass" if finite_svd == expected_svd else "fail",
            "observed": finite_svd,
            "expected": expected_svd,
        },
        {
            "check": "probe_precision",
            "status": "pass" if probe_precision in ("training", "float32") else "fail",
            "observed": probe_precision,
            "expected": "training or float32",
        },
    ]
    return checks


def write_quadratic_probe_artifacts(
    output_dir: str | Path,
    rows: list[dict],
    line_rows: list[dict],
    *,
    config: dict,
    metadata: list[dict],
    expected_steps: list[int],
    expected_layers: list[int],
    expected_modes: list[str],
    expected_build_repeats: int = 1,
    expected_heldout_batches: int = 1,
    line_search_multipliers: list[float],
    exact_hvp: bool,
    exact_svd_repeats: int = 0,
    include_none_repeat: bool = False,
    normmatch_modes: list[str] | None = None,
    probe_precision: str = "training",
    line_search: bool,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    long_path = output / "quadratic_probe_long.csv"
    line_path = output / "line_search_results.csv"
    paired_line_path = output / "line_search_paired_results.csv"
    line_summary_path = output / "line_search_crossfit_summary.csv"
    summary_path = output / "quadratic_probe_summary.csv"
    checks_path = output / "probe_data_quality_checks.csv"
    config_path = output / "probe_config.json"
    metadata_path = output / "probe_metadata.json"

    _write_csv(long_path, rows)
    _write_csv(line_path, line_rows)
    paired_line_rows = _paired_line_rows(line_rows)
    _write_csv(paired_line_path, paired_line_rows)
    _write_csv(line_summary_path, _line_summary_rows(paired_line_rows))
    _write_csv(summary_path, _summary_rows(rows))
    checks = _quality_checks(
        rows,
        line_rows,
        expected_steps=expected_steps,
        expected_layers=expected_layers,
        expected_modes=expected_modes,
        expected_build_repeats=expected_build_repeats,
        expected_heldout_batches=expected_heldout_batches,
        line_search_multipliers=line_search_multipliers,
        exact_hvp=exact_hvp,
        exact_svd_repeats=exact_svd_repeats,
        include_none_repeat=include_none_repeat,
        normmatch_modes=normmatch_modes or [],
        probe_precision=probe_precision,
        line_search=line_search,
    )
    _write_csv(checks_path, checks)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "quadratic_probe_long": str(long_path),
        "line_search_results": str(line_path),
        "line_search_paired_results": str(paired_line_path),
        "line_search_crossfit_summary": str(line_summary_path),
        "quadratic_probe_summary": str(summary_path),
        "probe_data_quality_checks": str(checks_path),
        "probe_config": str(config_path),
        "probe_metadata": str(metadata_path),
    }
