"""Optimizer-state-aware c_proj mechanism probe.

This P2 probe keeps the training trajectory unchanged and compares candidate
directions at one fixed checkpoint along three state axes:

* fresh K with the current probe gradient;
* shadow EMA K with the current probe gradient;
* shadow EMA K with the corresponding accumulated momentum.

NS5 is evaluated for every state condition. Exact-SVD polar directions are
also line-searched for the fresh-gradient condition so finite Newton--Schulz
error is an explicit intervention rather than a cosine-only diagnostic.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable

import torch

from optimizers import matrix_cosine, matrix_sign_ns5, matrix_sign_svd
from quadratic_probe import (
    _autocast_context,
    _fresh_covariance_stats,
    _line_search,
    _math_sdp_context,
    _preconditioned_gradient,
    _projector_drift,
    _row_orthogonality_residual,
    _strict_float32_context,
    _target_module_and_param,
)


VALID_TEMPORAL_MODES = ("none", "diag", "full")
NS5_CONDITIONS = (
    ("fresh", "gradient"),
    ("ema", "gradient"),
    ("ema", "momentum"),
)


def temporal_candidate_names(
    modes: list[str],
    *,
    include_exact_svd: bool,
) -> list[str]:
    validate_temporal_modes(modes)
    names = [
        candidate_name(k_source, buffer_source, mode, "ns5")
        for k_source, buffer_source in NS5_CONDITIONS
        for mode in modes
    ]
    if include_exact_svd:
        names.extend(
            candidate_name("fresh", "gradient", mode, "svd")
            for mode in modes
        )
    return names


def candidate_name(
    k_source: str,
    buffer_source: str,
    mode: str,
    projection: str,
) -> str:
    return f"{k_source}_{buffer_source}_{mode}_{projection}"


def validate_temporal_modes(modes: list[str]) -> None:
    if not modes:
        raise ValueError("temporal probe requires at least one K mode")
    if len(modes) != len(set(modes)):
        raise ValueError(f"duplicate temporal probe modes: {modes}")
    unknown = [mode for mode in modes if mode not in VALID_TEMPORAL_MODES]
    if unknown:
        raise ValueError(
            f"unknown temporal probe modes {unknown}; valid={VALID_TEMPORAL_MODES}"
        )
    if "none" not in modes:
        raise ValueError("temporal probe requires none as the matched reference")


@torch.no_grad()
def _apply_optimizer_covariance(
    gradient: torch.Tensor,
    input_cov,
) -> torch.Tensor:
    if input_cov is None:
        return gradient.clone()
    apply_right = getattr(input_cov, "apply_right", None)
    if callable(apply_right):
        return apply_right(gradient)
    k_inv = getattr(input_cov, "K_inv", None)
    if not isinstance(k_inv, torch.Tensor):
        raise TypeError("optimizer covariance state has no apply_right or K_inv")
    return gradient.float() @ k_inv.float()


def _state_for_mode(probe_state: dict, mode: str) -> tuple[object, torch.Tensor]:
    if mode == "none":
        if probe_state["actual_mode"] != "none":
            raise ValueError(
                "P2 none reference requires a c_proj none training trajectory; "
                f"got actual_mode={probe_state['actual_mode']!r}"
            )
        return probe_state["actual_input_cov"], probe_state["actual_momentum"]
    shadows = probe_state.get("shadows", {})
    if mode not in shadows:
        raise KeyError(
            f"missing shadow optimizer state for mode={mode!r}; "
            f"available={sorted(shadows)}"
        )
    return shadows[mode]["input_cov"], shadows[mode]["momentum"]


def _candidate(
    projection_input: torch.Tensor,
    *,
    gradient: torch.Tensor,
    k_mode: str,
    k_source: str,
    buffer_source: str,
    projection: str,
    current_preconditioned: torch.Tensor,
    prior_momentum: torch.Tensor | None,
    ns_steps: int,
    matrix_eps: float,
    state_updates: int,
    svd_compute_dtype: torch.dtype,
) -> dict:
    if projection == "ns5":
        q = matrix_sign_ns5(
            projection_input,
            steps=ns_steps,
            eps=matrix_eps,
        ).float()
        projection_compute_dtype = "float32"
    elif projection == "svd":
        q = matrix_sign_svd(
            projection_input,
            compute_dtype=svd_compute_dtype,
        ).float()
        projection_compute_dtype = str(svd_compute_dtype).replace("torch.", "")
    else:
        raise ValueError(f"unknown projection: {projection}")
    q = q.detach()
    return {
        "q": q,
        "base_mode": k_mode,
        "direction_variant": f"{k_source}_{buffer_source}_{projection}",
        "k_mode": k_mode,
        "k_source": k_source,
        "buffer_source": buffer_source,
        "projection": projection,
        "projection_compute_dtype": projection_compute_dtype,
        "projection_input_fro_norm": float(
            torch.linalg.vector_norm(projection_input).item()
        ),
        "current_preconditioned_fro_norm": float(
            torch.linalg.vector_norm(current_preconditioned).item()
        ),
        "prior_momentum_fro_norm": (
            float(torch.linalg.vector_norm(prior_momentum).item())
            if isinstance(prior_momentum, torch.Tensor)
            else 0.0
        ),
        "projection_input_cos_vs_gradient": matrix_cosine(
            projection_input,
            gradient,
        ),
        "q_fro_norm": float(torch.linalg.vector_norm(q).item()),
        "row_orthogonality_residual": _row_orthogonality_residual(q),
        "state_updates": int(state_updates),
    }


def _build_candidates(
    gradient: torch.Tensor,
    activations: torch.Tensor,
    probe_state: dict,
    *,
    modes: list[str],
    ridge: float,
    blocks: int,
    ns_steps: int,
    matrix_eps: float,
    include_exact_svd: bool,
    svd_compute_dtype: torch.dtype,
) -> dict[str, dict]:
    del blocks  # P2 intentionally restricts the temporal comparison to none/diag/full.
    validate_temporal_modes(modes)
    momentum_beta = float(probe_state["momentum_beta"])
    fresh_by_mode: dict[str, torch.Tensor] = {}
    ema_by_mode: dict[str, torch.Tensor] = {}
    prior_by_mode: dict[str, torch.Tensor] = {}
    state_updates: dict[str, int] = {}

    for mode in modes:
        fresh_by_mode[mode] = _preconditioned_gradient(
            gradient,
            activations,
            mode=mode,
            ridge=ridge,
            blocks=1,
        )
        input_cov, prior_momentum = _state_for_mode(probe_state, mode)
        ema_by_mode[mode] = _apply_optimizer_covariance(gradient, input_cov)
        prior_by_mode[mode] = prior_momentum.detach().float()
        state_updates[mode] = int(getattr(input_cov, "num_updates", 0))

    candidates: dict[str, dict] = {}
    for mode in modes:
        fresh = fresh_by_mode[mode]
        ema = ema_by_mode[mode]
        temporal_buffer = momentum_beta * prior_by_mode[mode] + ema

        fresh_name = candidate_name("fresh", "gradient", mode, "ns5")
        candidates[fresh_name] = _candidate(
            fresh,
            gradient=gradient,
            k_mode=mode,
            k_source="fresh",
            buffer_source="gradient",
            projection="ns5",
            current_preconditioned=fresh,
            prior_momentum=None,
            ns_steps=ns_steps,
            matrix_eps=matrix_eps,
            state_updates=0,
            svd_compute_dtype=svd_compute_dtype,
        )
        ema_name = candidate_name("ema", "gradient", mode, "ns5")
        candidates[ema_name] = _candidate(
            ema,
            gradient=gradient,
            k_mode=mode,
            k_source="ema",
            buffer_source="gradient",
            projection="ns5",
            current_preconditioned=ema,
            prior_momentum=None,
            ns_steps=ns_steps,
            matrix_eps=matrix_eps,
            state_updates=state_updates[mode],
            svd_compute_dtype=svd_compute_dtype,
        )
        state_name = candidate_name("ema", "momentum", mode, "ns5")
        candidates[state_name] = _candidate(
            temporal_buffer,
            gradient=gradient,
            k_mode=mode,
            k_source="ema",
            buffer_source="momentum",
            projection="ns5",
            current_preconditioned=ema,
            prior_momentum=prior_by_mode[mode],
            ns_steps=ns_steps,
            matrix_eps=matrix_eps,
            state_updates=state_updates[mode],
            svd_compute_dtype=svd_compute_dtype,
        )
        if include_exact_svd:
            svd_name = candidate_name("fresh", "gradient", mode, "svd")
            candidates[svd_name] = _candidate(
                fresh,
                gradient=gradient,
                k_mode=mode,
                k_source="fresh",
                buffer_source="gradient",
                projection="svd",
                current_preconditioned=fresh,
                prior_momentum=None,
                ns_steps=ns_steps,
                matrix_eps=matrix_eps,
                state_updates=0,
                svd_compute_dtype=svd_compute_dtype,
            )

    return candidates


def _add_candidate_pair_geometry(candidates: dict[str, dict]) -> None:
    for name, candidate in candidates.items():
        matched_none_name = candidate_name(
            candidate["k_source"],
            candidate["buffer_source"],
            "none",
            candidate["projection"],
        )
        matched_none = candidates[matched_none_name]["q"]
        candidate["direction_cos_vs_matched_none"] = matrix_cosine(
            candidate["q"],
            matched_none,
        )
        candidate["direction_norm_ratio_vs_matched_none"] = float(
            (
                torch.linalg.vector_norm(candidate["q"])
                / torch.linalg.vector_norm(matched_none).clamp_min(1e-30)
            ).item()
        )
        if candidate["k_source"] == "fresh" and candidate["buffer_source"] == "gradient":
            other_projection = "svd" if candidate["projection"] == "ns5" else "ns5"
            other_name = candidate_name(
                "fresh",
                "gradient",
                candidate["k_mode"],
                other_projection,
            )
            candidate["ns5_svd_cos"] = (
                matrix_cosine(candidate["q"], candidates[other_name]["q"])
                if other_name in candidates
                else math.nan
            )
        else:
            candidate["ns5_svd_cos"] = math.nan


def _probe_one_layer(
    model,
    optimizer_state_owner,
    batch: tuple[torch.Tensor, torch.Tensor],
    heldout_batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    step: int,
    build_repeat: int,
    layer: int,
    modes: list[str],
    ridge: float,
    blocks: int,
    ns_steps: int,
    matrix_eps: float,
    matrix_learning_rate: float,
    line_search_multipliers: list[float],
    exact_hvp: bool,
    include_exact_svd: bool,
    line_search: bool,
    device_type: str,
    autocast_dtype: torch.dtype,
    svd_compute_dtype: torch.dtype,
) -> tuple[list[dict], list[dict], dict]:
    x, y = batch
    module_name, module, param = _target_module_and_param(model, layer)
    state_getter = getattr(
        optimizer_state_owner,
        "get_cproj_temporal_probe_state",
        None,
    )
    if not callable(state_getter):
        raise TypeError(
            "optimizer does not expose get_cproj_temporal_probe_state; "
            "use CProjKModeNewtonMuon with cproj shadow K modes enabled"
        )
    probe_state = state_getter(param)
    activation_cache: dict[str, torch.Tensor] = {}

    def capture_input(_module, inputs):
        activation_cache["x"] = inputs[0].detach()

    handle = module.register_forward_pre_hook(capture_input)
    layer_start = time.perf_counter()
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad(), _math_sdp_context(device_type), _autocast_context(
            device_type,
            autocast_dtype,
        ):
            _, loss = model(x, y)
        probe_loss_value = float(loss.detach().item())
        if "x" not in activation_cache:
            raise RuntimeError(f"activation hook for {module_name} did not fire")

        grad_graph = torch.autograd.grad(loss, param, create_graph=exact_hvp)[0]
        gradient = grad_graph.detach().float()
        activations = activation_cache["x"].reshape(-1, param.shape[1]).float()
        cov_stats = _fresh_covariance_stats(activations)
        candidate_start = time.perf_counter()
        candidates = _build_candidates(
            gradient,
            activations,
            probe_state,
            modes=modes,
            ridge=ridge,
            blocks=blocks,
            ns_steps=ns_steps,
            matrix_eps=matrix_eps,
            include_exact_svd=include_exact_svd,
            svd_compute_dtype=svd_compute_dtype,
        )
        _add_candidate_pair_geometry(candidates)
        candidate_seconds = time.perf_counter() - candidate_start

        global_reference = candidates[
            candidate_name("fresh", "gradient", "none", "ns5")
        ]["q"]
        gradient_norm = torch.linalg.vector_norm(gradient).clamp_min(1e-30)
        global_reference_norm = torch.linalg.vector_norm(global_reference).clamp_min(
            1e-30
        )
        rows: list[dict] = []
        hvp_start = time.perf_counter()
        candidate_items = list(candidates.items())
        for index, (name, candidate) in enumerate(candidate_items):
            q = candidate["q"]
            q_norm = torch.linalg.vector_norm(q).clamp_min(1e-30)
            alignment_tensor = torch.sum(gradient * q)
            alignment_raw = float(alignment_tensor.item())
            alignment_normalized = float(
                (alignment_tensor / (gradient_norm * q_norm)).item()
            )

            curvature = math.nan
            if exact_hvp:
                directional_derivative = torch.sum(grad_graph.float() * q)
                hv = torch.autograd.grad(
                    directional_derivative,
                    param,
                    retain_graph=index < len(candidate_items) - 1,
                    create_graph=False,
                )[0]
                curvature = float(torch.sum(hv.detach().float() * q).item())

            if (
                math.isfinite(curvature)
                and curvature > 0.0
                and alignment_raw > 0.0
            ):
                quadratic_score = alignment_raw * alignment_raw / curvature
                eta_star = alignment_raw / curvature
                predicted_best_delta = -0.5 * quadratic_score
            else:
                quadratic_score = math.nan
                eta_star = math.nan
                predicted_best_delta = math.nan

            rows.append(
                {
                    "step": int(step),
                    "optimizer_step": int(probe_state["optimizer_step"]),
                    "build_repeat": int(build_repeat),
                    "layer": int(layer),
                    "module_name": module_name,
                    "candidate": name,
                    "mode": name,
                    "k_mode": candidate["k_mode"],
                    "k_source": candidate["k_source"],
                    "buffer_source": candidate["buffer_source"],
                    "projection": candidate["projection"],
                    "projection_compute_dtype": candidate[
                        "projection_compute_dtype"
                    ],
                    "probe_loss": probe_loss_value,
                    "gradient_fro_norm": float(gradient_norm.item()),
                    "projection_input_fro_norm": candidate[
                        "projection_input_fro_norm"
                    ],
                    "current_preconditioned_fro_norm": candidate[
                        "current_preconditioned_fro_norm"
                    ],
                    "prior_momentum_fro_norm": candidate[
                        "prior_momentum_fro_norm"
                    ],
                    "projection_input_cos_vs_gradient": candidate[
                        "projection_input_cos_vs_gradient"
                    ],
                    "direction_fro_norm": candidate["q_fro_norm"],
                    "direction_cos_vs_global_reference": matrix_cosine(
                        q,
                        global_reference,
                    ),
                    "direction_norm_ratio_vs_global_reference": float(
                        (q_norm / global_reference_norm).item()
                    ),
                    "direction_cos_vs_matched_none": candidate[
                        "direction_cos_vs_matched_none"
                    ],
                    "direction_norm_ratio_vs_matched_none": candidate[
                        "direction_norm_ratio_vs_matched_none"
                    ],
                    "projector_drift_vs_global_reference": _projector_drift(
                        q,
                        global_reference,
                    ),
                    "row_orthogonality_residual": candidate[
                        "row_orthogonality_residual"
                    ],
                    "ns5_svd_cos": candidate["ns5_svd_cos"],
                    "state_updates": candidate["state_updates"],
                    "momentum_beta": float(probe_state["momentum_beta"]),
                    "alignment_raw": alignment_raw,
                    "alignment_normalized": alignment_normalized,
                    "curvature_exact": curvature,
                    "curvature_per_direction_norm2": (
                        curvature / float(q_norm.square().item())
                        if math.isfinite(curvature)
                        else math.nan
                    ),
                    "quadratic_score_exact": quadratic_score,
                    "eta_star_exact": eta_star,
                    "predicted_best_loss_delta": predicted_best_delta,
                    **cov_stats,
                }
            )
        hvp_seconds = time.perf_counter() - hvp_start

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
            for row in line_rows:
                candidate = candidates[row["mode"]]
                row["candidate"] = row["mode"]
                row["k_mode"] = candidate["k_mode"]
                row["k_source"] = candidate["k_source"]
                row["buffer_source"] = candidate["buffer_source"]
                row["projection"] = candidate["projection"]
                row["projection_compute_dtype"] = candidate[
                    "projection_compute_dtype"
                ]
        line_seconds = time.perf_counter() - line_start

        total_seconds = time.perf_counter() - layer_start
        for row in rows:
            name = row["candidate"]
            same = line_best.get(f"{name}:same", {})
            heldout = [
                line_best.get(f"{name}:heldout_{index}", {})
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
            heldout_eta1 = [
                float(item["loss_delta"])
                for item in line_rows
                if item["candidate"] == name
                and item["eval_kind"] == "heldout"
                and math.isclose(float(item["lr_multiplier"]), 1.0)
            ]
            row.update(
                {
                    "line_search_same_best_delta": same.get(
                        "best_delta",
                        math.nan,
                    ),
                    "line_search_same_best_eta": same.get(
                        "best_eta",
                        math.nan,
                    ),
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
                        mean(heldout_eta1) if heldout_eta1 else math.nan
                    ),
                    "line_search_heldout_eta1_delta_std": (
                        pstdev(heldout_eta1) if heldout_eta1 else math.nan
                    ),
                    "heldout_eval_batches": len(heldout_batches),
                    "candidate_seconds_layer": candidate_seconds,
                    "hvp_seconds_layer": hvp_seconds,
                    "line_search_seconds_layer": line_seconds,
                    "total_seconds_layer": total_seconds,
                }
            )
        state_meta = {
            "optimizer_step": int(probe_state["optimizer_step"]),
            "momentum_beta": float(probe_state["momentum_beta"]),
            "actual_mode": str(probe_state["actual_mode"]),
            "shadow_modes": sorted(probe_state.get("shadows", {})),
            "shadow_state_updates": {
                mode: int(getattr(item["input_cov"], "num_updates", 0))
                for mode, item in probe_state.get("shadows", {}).items()
            },
        }
        return rows, line_rows, state_meta
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def run_cproj_temporal_quadratic_probe(
    model,
    optimizer_state_owner,
    build_batches: list[tuple[torch.Tensor, torch.Tensor]],
    heldout_batches: list[tuple[torch.Tensor, torch.Tensor]],
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
    exact_svd_repeats: int,
    line_search: bool,
    device_type: str,
    autocast_dtype: torch.dtype,
    svd_compute_dtype: torch.dtype = torch.float32,
) -> tuple[list[dict], list[dict], dict]:
    validate_temporal_modes(modes)
    if not build_batches:
        raise ValueError("temporal probe requires at least one build batch")
    if not layers:
        raise ValueError("temporal probe requires at least one layer")
    if exact_svd_repeats < 0 or exact_svd_repeats > len(build_batches):
        raise ValueError(
            "exact_svd_repeats must be in [0, number of build batches]"
        )
    if any(multiplier < 0 for multiplier in line_search_multipliers):
        raise ValueError("line-search multipliers must be non-negative")
    if svd_compute_dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "svd_compute_dtype must be torch.float32 or torch.float64"
        )

    was_training = bool(model.training)
    if device_type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    probe_start = time.perf_counter()
    all_rows: list[dict] = []
    all_line_rows: list[dict] = []
    state_meta_rows: list[dict] = []
    try:
        with _strict_float32_context(device_type, autocast_dtype):
            model.eval()
            for build_repeat, batch in enumerate(build_batches):
                for layer in layers:
                    rows, line_rows, state_meta = _probe_one_layer(
                        model,
                        optimizer_state_owner,
                        batch,
                        heldout_batches,
                        step=step,
                        build_repeat=build_repeat,
                        layer=layer,
                        modes=modes,
                        ridge=ridge,
                        blocks=blocks,
                        ns_steps=ns_steps,
                        matrix_eps=matrix_eps,
                        matrix_learning_rate=matrix_learning_rate,
                        line_search_multipliers=line_search_multipliers,
                        exact_hvp=exact_hvp,
                        include_exact_svd=build_repeat < exact_svd_repeats,
                        line_search=line_search,
                        device_type=device_type,
                        autocast_dtype=autocast_dtype,
                        svd_compute_dtype=svd_compute_dtype,
                    )
                    all_rows.extend(rows)
                    all_line_rows.extend(line_rows)
                    state_meta_rows.append(state_meta)
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
    base_candidates = len(modes) * len(NS5_CONDITIONS)
    exact_candidates = len(modes)
    optimizer_stats = getattr(optimizer_state_owner, "last_stats", {}) or {}
    metadata = {
        "step": int(step),
        "layers": list(layers),
        "modes": list(modes),
        "build_repeats": len(build_batches),
        "heldout_batches": len(heldout_batches),
        "ns5_candidates_per_build_layer": base_candidates,
        "svd_candidates_per_exact_build_layer": exact_candidates,
        "exact_svd_repeats": int(exact_svd_repeats),
        "direction_rows": len(all_rows),
        "line_search_rows": len(all_line_rows),
        "probe_dtype": str(autocast_dtype).replace("torch.", ""),
        "svd_compute_dtype": str(svd_compute_dtype).replace("torch.", ""),
        "probe_seconds": time.perf_counter() - probe_start,
        "probe_peak_memory_mib": peak_mib,
        "diagnostic_shadow_k_state_bytes": int(
            optimizer_stats.get("shadow_k_state_bytes", 0)
        ),
        "diagnostic_shadow_momentum_bytes": int(
            optimizer_stats.get("shadow_momentum_bytes", 0)
        ),
        "optimizer_state": state_meta_rows[0] if state_meta_rows else {},
    }
    return all_rows, all_line_rows, metadata


def _write_csv(path: Path, rows: list[dict]) -> None:
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _finite_values(rows: Iterable[dict], key: str) -> list[float]:
    result = []
    for row in rows:
        try:
            value = float(row.get(key, math.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result.append(value)
    return result


def _paired_line_rows(line_rows: list[dict]) -> list[dict]:
    index = {
        (
            int(row["step"]),
            int(row["build_repeat"]),
            int(row["layer"]),
            str(row["eval_split"]),
            float(row["lr_multiplier"]),
            str(row["k_source"]),
            str(row["buffer_source"]),
            str(row["projection"]),
            str(row["k_mode"]),
        ): float(row["loss_delta"])
        for row in line_rows
    }
    paired: list[dict] = []
    for row in line_rows:
        prefix = (
            int(row["step"]),
            int(row["build_repeat"]),
            int(row["layer"]),
            str(row["eval_split"]),
            float(row["lr_multiplier"]),
        )
        matched_none_key = (
            *prefix,
            str(row["k_source"]),
            str(row["buffer_source"]),
            str(row["projection"]),
            "none",
        )
        ns5_key = (
            *prefix,
            str(row["k_source"]),
            str(row["buffer_source"]),
            "ns5",
            str(row["k_mode"]),
        )
        fresh_key = (
            *prefix,
            "fresh",
            "gradient",
            str(row["projection"]),
            str(row["k_mode"]),
        )
        ema_gradient_key = (
            *prefix,
            "ema",
            "gradient",
            str(row["projection"]),
            str(row["k_mode"]),
        )
        item = dict(row)
        item["loss_delta_vs_matched_none"] = (
            float(row["loss_delta"]) - index[matched_none_key]
            if matched_none_key in index
            else math.nan
        )
        item["loss_delta_vs_matched_ns5"] = (
            float(row["loss_delta"]) - index[ns5_key]
            if ns5_key in index
            else math.nan
        )
        item["loss_delta_vs_fresh_same_mode"] = (
            float(row["loss_delta"]) - index[fresh_key]
            if fresh_key in index
            else math.nan
        )
        item["loss_delta_vs_ema_gradient_same_mode"] = (
            float(row["loss_delta"]) - index[ema_gradient_key]
            if ema_gradient_key in index
            else math.nan
        )
        paired.append(item)
    return paired


def _direction_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            int(row["step"]),
            str(row["candidate"]),
            str(row["k_mode"]),
            str(row["k_source"]),
            str(row["buffer_source"]),
            str(row["projection"]),
        )
        groups.setdefault(key, []).append(row)
    metrics = (
        "alignment_normalized",
        "curvature_exact",
        "quadratic_score_exact",
        "direction_cos_vs_global_reference",
        "direction_cos_vs_matched_none",
        "direction_norm_ratio_vs_matched_none",
        "row_orthogonality_residual",
        "ns5_svd_cos",
        "line_search_same_best_delta",
        "line_search_heldout_best_delta",
        "line_search_heldout_eta1_delta_mean",
    )
    result = []
    for key, group in sorted(groups.items()):
        step, candidate, mode, k_source, buffer_source, projection = key
        item = {
            "step": step,
            "candidate": candidate,
            "k_mode": mode,
            "k_source": k_source,
            "buffer_source": buffer_source,
            "projection": projection,
            "projection_compute_dtype": str(
                group[0].get("projection_compute_dtype", "")
            ),
            "observations": len(group),
            "build_repeats": len({int(row["build_repeat"]) for row in group}),
            "layers": len({int(row["layer"]) for row in group}),
        }
        for metric in metrics:
            values = _finite_values(group, metric)
            item[f"{metric}_mean"] = mean(values) if values else math.nan
        result.append(item)
    return result


def _line_summary(paired_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in paired_rows:
        key = (
            int(row["step"]),
            str(row["candidate"]),
            str(row["k_mode"]),
            str(row["k_source"]),
            str(row["buffer_source"]),
            str(row["projection"]),
            str(row["eval_kind"]),
            float(row["lr_multiplier"]),
            float(row["eta"]),
        )
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(groups.items()):
        (
            step,
            candidate,
            mode,
            k_source,
            buffer_source,
            projection,
            eval_kind,
            multiplier,
            eta,
        ) = key
        deltas = [float(row["loss_delta"]) for row in group]
        item = {
            "step": step,
            "candidate": candidate,
            "k_mode": mode,
            "k_source": k_source,
            "buffer_source": buffer_source,
            "projection": projection,
            "projection_compute_dtype": str(
                group[0].get("projection_compute_dtype", "")
            ),
            "eval_kind": eval_kind,
            "lr_multiplier": multiplier,
            "eta": eta,
            "observations": len(group),
            "build_repeats": len({int(row["build_repeat"]) for row in group}),
            "layers": len({int(row["layer"]) for row in group}),
            "loss_delta_mean": mean(deltas),
            "loss_delta_std": pstdev(deltas),
            "loss_delta_median": median(deltas),
        }
        for metric in (
            "loss_delta_vs_matched_none",
            "loss_delta_vs_matched_ns5",
            "loss_delta_vs_fresh_same_mode",
            "loss_delta_vs_ema_gradient_same_mode",
        ):
            values = _finite_values(group, metric)
            item[f"{metric}_mean"] = mean(values) if values else math.nan
            item[f"{metric}_std"] = pstdev(values) if values else math.nan
        result.append(item)
    return result


def _paired_max_gap(
    paired_rows: list[dict],
    left_candidate: str,
    right_candidate: str,
) -> tuple[float, int]:
    index = {
        (
            int(row["step"]),
            int(row["build_repeat"]),
            int(row["layer"]),
            str(row["eval_split"]),
            float(row["lr_multiplier"]),
            str(row["candidate"]),
        ): float(row["loss_delta"])
        for row in paired_rows
    }
    gaps = []
    for key, left in index.items():
        if key[-1] != left_candidate:
            continue
        right_key = (*key[:-1], right_candidate)
        if right_key in index:
            gaps.append(abs(left - index[right_key]))
    return (max(gaps) if gaps else math.nan, len(gaps))


def _quality_checks(
    rows: list[dict],
    paired_rows: list[dict],
    *,
    expected_steps: list[int],
    expected_layers: list[int],
    modes: list[str],
    expected_build_repeats: int,
    expected_heldout_batches: int,
    line_search_multipliers: list[float],
    exact_hvp: bool,
    exact_svd_repeats: int,
    probe_precision: str,
    line_search: bool,
    svd_compute_dtype: str,
) -> list[dict]:
    ns5_names = temporal_candidate_names(modes, include_exact_svd=False)
    svd_names = [
        candidate_name("fresh", "gradient", mode, "svd") for mode in modes
    ]
    expected_direction_rows = (
        len(expected_steps)
        * len(expected_layers)
        * (
            expected_build_repeats * len(ns5_names)
            + exact_svd_repeats * len(svd_names)
        )
    )
    splits = 1 + expected_heldout_batches
    expected_line_rows = (
        expected_direction_rows * splits * len(line_search_multipliers)
        if line_search
        else 0
    )
    direction_keys = {
        (
            int(row["step"]),
            int(row["build_repeat"]),
            int(row["layer"]),
            str(row["candidate"]),
        )
        for row in rows
    }
    expected_keys = set()
    for step in expected_steps:
        for repeat in range(expected_build_repeats):
            names = list(ns5_names)
            if repeat < exact_svd_repeats:
                names.extend(svd_names)
            for layer in expected_layers:
                for name in names:
                    expected_keys.add((step, repeat, layer, name))

    control_left = candidate_name("fresh", "gradient", "none", "ns5")
    control_right = candidate_name("ema", "gradient", "none", "ns5")
    control_cos = [
        float(row["direction_cos_vs_global_reference"])
        for row in rows
        if row["candidate"] == control_right
    ]
    control_gap, control_pairs = _paired_max_gap(
        paired_rows,
        control_left,
        control_right,
    )
    svd_residuals = [
        float(row["row_orthogonality_residual"])
        for row in rows
        if row["projection"] == "svd"
    ]
    svd_dtype_rows = [
        str(row.get("projection_compute_dtype", ""))
        for row in rows
        if row["projection"] == "svd"
    ]
    shadow_state_rows = [
        row
        for row in rows
        if row["k_mode"] in ("diag", "full")
        and row["k_source"] == "ema"
    ]
    expected_control_pairs = (
        len(expected_steps)
        * expected_build_repeats
        * len(expected_layers)
        * splits
        * len(line_search_multipliers)
        if line_search
        else 0
    )
    return [
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
            "check": "line_search_row_count",
            "status": "pass" if len(paired_rows) == expected_line_rows else "fail",
            "observed": len(paired_rows),
            "expected": expected_line_rows,
        },
        {
            "check": "finite_alignment",
            "status": (
                "pass"
                if len(_finite_values(rows, "alignment_raw")) == len(rows)
                else "fail"
            ),
            "observed": len(_finite_values(rows, "alignment_raw")),
            "expected": len(rows),
        },
        {
            "check": "finite_curvature",
            "status": (
                "pass"
                if (
                    not exact_hvp
                    or len(_finite_values(rows, "curvature_exact")) == len(rows)
                )
                else "fail"
            ),
            "observed": len(_finite_values(rows, "curvature_exact")),
            "expected": len(rows) if exact_hvp else 0,
        },
        {
            "check": "fresh_none_equals_ema_none_direction",
            "status": (
                "pass"
                if control_cos and min(control_cos) >= 0.999999
                else "fail"
            ),
            "observed": min(control_cos) if control_cos else "",
            "expected": ">=0.999999",
        },
        {
            "check": "fresh_none_equals_ema_none_line_search",
            "status": (
                "pass"
                if (
                    not line_search
                    or (
                        control_pairs == expected_control_pairs
                        and control_gap <= 1e-6
                    )
                )
                else "fail"
            ),
            "observed": control_gap if control_pairs else "",
            "expected": "<=1e-6 max absolute paired gap",
        },
        {
            "check": "shadow_state_updated",
            "status": (
                "pass"
                if shadow_state_rows
                and min(int(row["state_updates"]) for row in shadow_state_rows) > 0
                else "fail"
            ),
            "observed": (
                min(int(row["state_updates"]) for row in shadow_state_rows)
                if shadow_state_rows
                else 0
            ),
            "expected": ">0 EMA refreshes for diag/full",
        },
        {
            "check": "exact_svd_row_orthogonality",
            "status": (
                "pass"
                if (
                    exact_svd_repeats == 0
                    or (
                        len(svd_residuals)
                        == len(expected_steps)
                        * exact_svd_repeats
                        * len(expected_layers)
                        * len(modes)
                        and max(svd_residuals) <= 1e-4
                    )
                )
                else "fail"
            ),
            "observed": max(svd_residuals) if svd_residuals else "",
            "expected": "<=1e-4",
        },
        {
            "check": "exact_svd_compute_dtype",
            "status": (
                "pass"
                if (
                    exact_svd_repeats == 0
                    or (
                        len(svd_dtype_rows)
                        == len(expected_steps)
                        * exact_svd_repeats
                        * len(expected_layers)
                        * len(modes)
                        and set(svd_dtype_rows) == {svd_compute_dtype}
                    )
                )
                else "fail"
            ),
            "observed": (
                ",".join(sorted(set(svd_dtype_rows)))
                if svd_dtype_rows
                else ""
            ),
            "expected": svd_compute_dtype,
        },
        {
            "check": "probe_precision",
            "status": (
                "pass"
                if probe_precision in ("training", "float32")
                else "fail"
            ),
            "observed": probe_precision,
            "expected": "training or float32",
        },
    ]


def write_temporal_quadratic_probe_artifacts(
    output_dir: str | Path,
    rows: list[dict],
    line_rows: list[dict],
    *,
    config: dict,
    metadata: list[dict],
    expected_steps: list[int],
    expected_layers: list[int],
    modes: list[str],
    expected_build_repeats: int,
    expected_heldout_batches: int,
    line_search_multipliers: list[float],
    exact_hvp: bool,
    exact_svd_repeats: int,
    probe_precision: str,
    line_search: bool,
    svd_compute_dtype: str = "float32",
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paired_rows = _paired_line_rows(line_rows)
    direction_summary = _direction_summary(rows)
    line_summary = _line_summary(paired_rows)
    quality = _quality_checks(
        rows,
        paired_rows,
        expected_steps=expected_steps,
        expected_layers=expected_layers,
        modes=modes,
        expected_build_repeats=expected_build_repeats,
        expected_heldout_batches=expected_heldout_batches,
        line_search_multipliers=line_search_multipliers,
        exact_hvp=exact_hvp,
        exact_svd_repeats=exact_svd_repeats,
        probe_precision=probe_precision,
        line_search=line_search,
        svd_compute_dtype=svd_compute_dtype,
    )

    paths = {
        "temporal_quadratic_probe_long": output
        / "temporal_quadratic_probe_long.csv",
        "temporal_quadratic_probe_summary": output
        / "temporal_quadratic_probe_summary.csv",
        "temporal_line_search_results": output
        / "temporal_line_search_results.csv",
        "temporal_line_search_paired_results": output
        / "temporal_line_search_paired_results.csv",
        "temporal_line_search_summary": output
        / "temporal_line_search_summary.csv",
        "probe_data_quality_checks": output / "probe_data_quality_checks.csv",
        "probe_config": output / "probe_config.json",
        "probe_metadata": output / "probe_metadata.json",
    }
    _write_csv(paths["temporal_quadratic_probe_long"], rows)
    _write_csv(paths["temporal_quadratic_probe_summary"], direction_summary)
    _write_csv(paths["temporal_line_search_results"], line_rows)
    _write_csv(paths["temporal_line_search_paired_results"], paired_rows)
    _write_csv(paths["temporal_line_search_summary"], line_summary)
    _write_csv(paths["probe_data_quality_checks"], quality)
    paths["probe_config"].write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["probe_metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failed = [row for row in quality if row["status"] != "pass"]
    if failed:
        raise RuntimeError(f"temporal probe quality checks failed: {failed}")
    return {key: str(path) for key, path in paths.items()}
