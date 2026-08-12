#!/usr/bin/env python
"""Reproducible hierarchical analysis for the temporal c_proj probe P2.

The archived probe and W&B exports are immutable inputs.  All derived tables
are written below ``processed/`` in the archived run directory.  The primary
independent unit is one direction-build repeat; layers and shared held-out
batches are averaged within a repeat before uncertainty is estimated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


LAYERS = (0, 11, 23)
BUILD_REPEATS = (0, 1, 2, 3)
CANDIDATES = (
    "fresh_gradient_none_ns5",
    "fresh_gradient_diag_ns5",
    "fresh_gradient_full_ns5",
    "ema_gradient_none_ns5",
    "ema_gradient_diag_ns5",
    "ema_gradient_full_ns5",
    "ema_momentum_none_ns5",
    "ema_momentum_diag_ns5",
    "ema_momentum_full_ns5",
    "fresh_gradient_none_svd",
    "fresh_gradient_diag_svd",
    "fresh_gradient_full_svd",
)
WORKING_ETAS = (0.01, 0.02)
T_CRITICAL_975_DF3 = 3.182446305

CONTRASTS = (
    {
        "contrast": "matched_none",
        "field": "loss_delta_vs_matched_none",
        "candidates": (
            "fresh_gradient_diag_ns5",
            "fresh_gradient_full_ns5",
            "ema_gradient_diag_ns5",
            "ema_gradient_full_ns5",
            "ema_momentum_diag_ns5",
            "ema_momentum_full_ns5",
            "fresh_gradient_diag_svd",
            "fresh_gradient_full_svd",
        ),
        "interpretation": "negative favors the structured-K candidate",
    },
    {
        "contrast": "svd_minus_ns5",
        "field": "loss_delta_vs_matched_ns5",
        "candidates": (
            "fresh_gradient_none_svd",
            "fresh_gradient_diag_svd",
            "fresh_gradient_full_svd",
        ),
        "interpretation": "negative favors exact-SVD over NS5",
    },
    {
        "contrast": "ema_minus_fresh",
        "field": "loss_delta_vs_fresh_same_mode",
        "candidates": (
            "ema_gradient_none_ns5",
            "ema_gradient_diag_ns5",
            "ema_gradient_full_ns5",
        ),
        "interpretation": "negative favors EMA K over fresh K",
    },
    {
        "contrast": "momentum_minus_ema_gradient",
        "field": "loss_delta_vs_ema_gradient_same_mode",
        "candidates": (
            "ema_momentum_none_ns5",
            "ema_momentum_diag_ns5",
            "ema_momentum_full_ns5",
        ),
        "interpretation": "negative favors the historical momentum buffer",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Archived P2 run containing raw_probe/ and raw_wandb_exports/.",
    )
    parser.add_argument(
        "--p1-run-dir",
        type=Path,
        default=None,
        help="Optional archived P1 run with processed/working_point_summary.csv.",
    )
    parser.add_argument(
        "--historical-summary",
        type=Path,
        default=None,
        help="Optional combined_val_curves_all_seeds.csv.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def source_inventory(raw_probe: Path, raw_wandb: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for source_type, folder in (("probe", raw_probe), ("wandb", raw_wandb)):
        for path in sorted(folder.glob("*")):
            if not path.is_file():
                continue
            csv_rows = None
            csv_columns = None
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
                csv_rows = len(frame)
                csv_columns = len(frame.columns)
            rows.append(
                {
                    "source_type": source_type,
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "rows": csv_rows,
                    "columns": csv_columns,
                }
            )
    return pd.DataFrame(rows)


def load_wandb_exports(
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    band_checks: list[dict] = []
    for path in sorted(raw_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        if frame.shape[1] < 2 or frame.columns[0] != "Step":
            raise ValueError(f"Unexpected W&B export schema: {path}")
        value_col = frame.columns[1]
        if " - " not in value_col:
            raise ValueError(f"Cannot recover metric name from {value_col!r}")
        run_name, metric = value_col.split(" - ", 1)
        compact = frame[["Step", value_col]].rename(
            columns={"Step": "step", value_col: "value"}
        )
        compact.insert(0, "metric", metric)
        compact.insert(0, "run_name", run_name)
        compact["source_file"] = path.name
        frames.append(compact)
        non_null = compact["value"].dropna()
        summaries.append(
            {
                "metric": metric,
                "source_file": path.name,
                "rows": len(frame),
                "step_min": int(frame["Step"].min()),
                "step_max": int(frame["Step"].max()),
                "duplicate_steps": int(frame["Step"].duplicated().sum()),
                "null_values": int(frame[value_col].isna().sum()),
                "first_value": float(non_null.iloc[0]),
                "last_value": float(non_null.iloc[-1]),
                "min_value": float(non_null.min()),
                "max_value": float(non_null.max()),
            }
        )
        for suffix in ("__MIN", "__MAX"):
            band_col = value_col + suffix
            if band_col in frame:
                band_checks.append(
                    {
                        "source_file": path.name,
                        "metric": metric,
                        "band": suffix[2:].lower(),
                        "duplicates_raw_series": bool(
                            np.allclose(
                                frame[value_col].to_numpy(dtype=float),
                                frame[band_col].to_numpy(dtype=float),
                                rtol=0.0,
                                atol=0.0,
                                equal_nan=True,
                            )
                        ),
                    }
                )
    return (
        pd.concat(frames, ignore_index=True),
        pd.DataFrame(summaries).sort_values("metric").reset_index(drop=True),
        band_checks,
    )


def candidate_build_effects(paired: pd.DataFrame) -> pd.DataFrame:
    selected = paired[paired["eta"].isin(WORKING_ETAS)].copy()
    return (
        selected.groupby(
            ["candidate", "eval_kind", "eta", "build_repeat"],
            sort=True,
            as_index=False,
        )
        .agg(
            layers=("layer", "nunique"),
            heldout_batches=("heldout_index", "nunique"),
            observations=("loss_delta", "size"),
            loss_delta_mean=("loss_delta", "mean"),
        )
    )


def contrast_build_effects(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    selected = paired[paired["eta"].isin(WORKING_ETAS)].copy()
    for spec in CONTRASTS:
        field = str(spec["field"])
        subset = selected[
            selected["candidate"].isin(spec["candidates"])
            & selected[field].notna()
        ].copy()
        grouped = (
            subset.groupby(
                ["candidate", "eval_kind", "eta", "build_repeat"],
                sort=True,
                as_index=False,
            )
            .agg(
                layers=("layer", "nunique"),
                heldout_batches=("heldout_index", "nunique"),
                observations=(field, "size"),
                contrast_value=(field, "mean"),
            )
        )
        grouped.insert(0, "contrast", spec["contrast"])
        grouped["interpretation"] = spec["interpretation"]
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def summarize_build_contrasts(build_effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in build_effects.groupby(
        ["contrast", "candidate", "eval_kind", "eta"], sort=True
    ):
        contrast, candidate, eval_kind, eta = keys
        values = group["contrast_value"].to_numpy(dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half_width = T_CRITICAL_975_DF3 * sd / math.sqrt(len(values))
        rows.append(
            {
                "contrast": contrast,
                "candidate": candidate,
                "eval_kind": eval_kind,
                "eta": float(eta),
                "independent_build_repeats": len(values),
                "mean": mean,
                "sd": sd,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "negative_repeats": int((values < 0).sum()),
                "positive_repeats": int((values > 0).sum()),
                "zero_repeats": int((values == 0).sum()),
                "interpretation": group["interpretation"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def layer_contrasts(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    selected = paired[paired["eta"].isin(WORKING_ETAS)].copy()
    for spec in CONTRASTS:
        field = str(spec["field"])
        subset = selected[
            selected["candidate"].isin(spec["candidates"])
            & selected[field].notna()
        ].copy()
        grouped = (
            subset.groupby(
                ["candidate", "eval_kind", "eta", "layer", "build_repeat"],
                sort=True,
                as_index=False,
            )
            .agg(
                heldout_batches=("heldout_index", "nunique"),
                observations=(field, "size"),
                contrast_value=(field, "mean"),
            )
        )
        grouped.insert(0, "contrast", spec["contrast"])
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def geometry_summary(directions: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "observations": ("layer", "size"),
        "build_repeats": ("build_repeat", "nunique"),
        "layers": ("layer", "nunique"),
        "alignment_normalized_mean": ("alignment_normalized", "mean"),
        "curvature_per_direction_norm2_mean": (
            "curvature_per_direction_norm2",
            "mean",
        ),
        "quadratic_score_exact_mean": ("quadratic_score_exact", "mean"),
        "eta_star_exact_mean": ("eta_star_exact", "mean"),
        "direction_cos_vs_matched_none_mean": (
            "direction_cos_vs_matched_none",
            "mean",
        ),
        "direction_norm_ratio_vs_matched_none_mean": (
            "direction_norm_ratio_vs_matched_none",
            "mean",
        ),
        "projection_input_cos_vs_gradient_mean": (
            "projection_input_cos_vs_gradient",
            "mean",
        ),
        "row_orthogonality_residual_mean": (
            "row_orthogonality_residual",
            "mean",
        ),
        "row_orthogonality_residual_max": (
            "row_orthogonality_residual",
            "max",
        ),
        "ns5_svd_cos_mean": ("ns5_svd_cos", "mean"),
        "state_updates_mean": ("state_updates", "mean"),
        "line_search_same_best_delta_mean": (
            "line_search_same_best_delta",
            "mean",
        ),
        "line_search_heldout_best_delta_mean": (
            "line_search_heldout_best_delta",
            "mean",
        ),
        "line_search_heldout_eta1_delta_mean": (
            "line_search_heldout_eta1_delta_mean",
            "mean",
        ),
    }
    return (
        directions.groupby(
            ["candidate", "k_mode", "k_source", "buffer_source", "projection"],
            sort=True,
        )
        .agg(**metrics)
        .reset_index()
    )


def p1_p2_comparison(
    p1_run_dir: Path | None,
    contrast_summary: pd.DataFrame,
) -> pd.DataFrame:
    if p1_run_dir is None:
        return pd.DataFrame()
    p1_path = p1_run_dir / "processed" / "working_point_summary.csv"
    if not p1_path.exists():
        raise FileNotFoundError(p1_path)
    p1 = pd.read_csv(p1_path)
    rows: list[dict] = []
    for mode in ("diag", "full"):
        for eval_kind in ("same", "heldout"):
            for eta in WORKING_ETAS:
                p1_row = p1[
                    (p1["mode"] == mode)
                    & (p1["eval_kind"] == eval_kind)
                    & np.isclose(p1["eta"], eta)
                ]
                p2_row = contrast_summary[
                    (contrast_summary["contrast"] == "matched_none")
                    & (
                        contrast_summary["candidate"]
                        == f"fresh_gradient_{mode}_ns5"
                    )
                    & (contrast_summary["eval_kind"] == eval_kind)
                    & np.isclose(contrast_summary["eta"], eta)
                ]
                if len(p1_row) != 1 or len(p2_row) != 1:
                    raise ValueError((mode, eval_kind, eta, len(p1_row), len(p2_row)))
                p1_value = float(p1_row.iloc[0]["repeat_mean_delta_vs_none"])
                p2_value = float(p2_row.iloc[0]["mean"])
                rows.append(
                    {
                        "mode": mode,
                        "eval_kind": eval_kind,
                        "eta": eta,
                        "p1_mean_delta_vs_none": p1_value,
                        "p2_fresh_mean_delta_vs_none": p2_value,
                        "p2_minus_p1": p2_value - p1_value,
                        "same_sign": bool(np.sign(p1_value) == np.sign(p2_value)),
                    }
                )
    return pd.DataFrame(rows)


def historical_at_step(
    path: Path | None, *, step: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    if path is None or not path.exists():
        return pd.DataFrame(), {}
    frame = pd.read_csv(path)
    selected = frame[
        (frame["step"] == step)
        & (frame["mode"].isin(["none", "diag", "full"]))
    ].copy()
    summary = (
        selected.groupby("mode", as_index=False)
        .agg(seeds=("seed", "nunique"), val_loss_mean=("val_loss", "mean"))
        .sort_values("mode")
    )
    lookup = dict(zip(summary["mode"], summary["val_loss_mean"]))
    return selected, {key: float(value) for key, value in lookup.items()}


def lookup_contrast(
    summary: pd.DataFrame,
    *,
    contrast: str,
    candidate: str,
    eval_kind: str,
    eta: float,
) -> pd.Series:
    match = summary[
        (summary["contrast"] == contrast)
        & (summary["candidate"] == candidate)
        & (summary["eval_kind"] == eval_kind)
        & np.isclose(summary["eta"], eta)
    ]
    if len(match) != 1:
        raise ValueError((contrast, candidate, eval_kind, eta, len(match)))
    return match.iloc[0]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    raw_probe = run_dir / "raw_probe"
    raw_wandb = run_dir / "raw_wandb_exports"
    processed = run_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    config = json.loads((raw_probe / "probe_config.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (raw_probe / "probe_metadata.json").read_text(encoding="utf-8")
    )[0]
    emitted_quality = pd.read_csv(raw_probe / "probe_data_quality_checks.csv")
    directions = pd.read_csv(raw_probe / "temporal_quadratic_probe_long.csv")
    paired = pd.read_csv(raw_probe / "temporal_line_search_paired_results.csv")
    line_search = pd.read_csv(raw_probe / "temporal_line_search_results.csv")
    inventory = source_inventory(raw_probe, raw_wandb)
    wandb_series, wandb_summary, band_checks = load_wandb_exports(raw_wandb)

    candidate_build = candidate_build_effects(paired)
    contrast_build = contrast_build_effects(paired)
    contrast_summary = summarize_build_contrasts(contrast_build)
    layer_detail = layer_contrasts(paired)
    geometry = geometry_summary(directions)
    p1_compare = p1_p2_comparison(
        args.p1_run_dir.resolve() if args.p1_run_dir else None,
        contrast_summary,
    )
    historical_rows, historical_lookup = historical_at_step(
        args.historical_summary.resolve() if args.historical_summary else None,
        step=int(config["steps"][0]),
    )

    expected_direction_keys = {
        (repeat, layer, candidate)
        for repeat in BUILD_REPEATS
        for layer in LAYERS
        for candidate in CANDIDATES
    }
    observed_direction_keys = set(
        zip(
            directions["build_repeat"].astype(int),
            directions["layer"].astype(int),
            directions["candidate"].astype(str),
        )
    )
    expected_line_keys = {
        (repeat, layer, candidate, kind, heldout, multiplier)
        for repeat in BUILD_REPEATS
        for layer in LAYERS
        for candidate in CANDIDATES
        for kind, heldouts in (("same", (-1,)), ("heldout", tuple(range(8))))
        for heldout in heldouts
        for multiplier in (0.0, 0.25, 0.5, 1.0, 2.0)
    }
    observed_line_keys = set(
        zip(
            paired["build_repeat"].astype(int),
            paired["layer"].astype(int),
            paired["candidate"].astype(str),
            paired["eval_kind"].astype(str),
            paired["heldout_index"].astype(int),
            paired["lr_multiplier"].astype(float),
        )
    )
    fresh_none_gap = float(
        paired[
            (paired["candidate"] == "ema_gradient_none_ns5")
            & (paired["eta"] > 0)
        ]["loss_delta_vs_fresh_same_mode"]
        .abs()
        .max()
    )
    svd_rows = directions[directions["projection"] == "svd"]
    svd_orth_max = float(svd_rows["row_orthogonality_residual"].max())
    analysis_quality = pd.DataFrame(
        [
            {
                "check": "archived_source_file_count",
                "status": status(len(inventory) == 16),
                "severity_if_failed": "high",
                "evidence": f"observed={len(inventory)}, expected=16",
            },
            {
                "check": "emitted_probe_quality_except_svd_tolerance",
                "status": status(
                    len(emitted_quality) == 10
                    and int(emitted_quality["status"].str.lower().eq("pass").sum())
                    == 9
                    and emitted_quality.loc[
                        emitted_quality["status"].str.lower().eq("fail"), "check"
                    ].tolist()
                    == ["exact_svd_row_orthogonality"]
                ),
                "severity_if_failed": "critical",
                "evidence": (
                    f"passed={emitted_quality['status'].str.lower().eq('pass').sum()}"
                    f"/{len(emitted_quality)}; expected sole failure="
                    "exact_svd_row_orthogonality"
                ),
            },
            {
                "check": "direction_key_coverage",
                "status": status(
                    observed_direction_keys == expected_direction_keys
                    and len(directions) == 144
                ),
                "severity_if_failed": "critical",
                "evidence": (
                    f"keys={len(observed_direction_keys)}/144, rows={len(directions)}"
                ),
            },
            {
                "check": "line_search_key_coverage",
                "status": status(
                    observed_line_keys == expected_line_keys
                    and len(paired) == 6480
                    and len(line_search) == 6480
                ),
                "severity_if_failed": "critical",
                "evidence": (
                    f"keys={len(observed_line_keys)}/6480, "
                    f"paired_rows={len(paired)}, raw_rows={len(line_search)}"
                ),
            },
            {
                "check": "finite_core_direction_metrics",
                "status": status(
                    np.isfinite(
                        directions[
                            [
                                "alignment_raw",
                                "alignment_normalized",
                                "curvature_exact",
                                "direction_fro_norm",
                                "direction_cos_vs_matched_none",
                            ]
                        ].to_numpy(dtype=float)
                    ).all()
                ),
                "severity_if_failed": "critical",
                "evidence": "alignment, curvature, norm and cosine are finite",
            },
            {
                "check": "fresh_none_equals_ema_none_control",
                "status": status(fresh_none_gap <= 1.0e-6),
                "severity_if_failed": "critical",
                "evidence": f"max_abs_line_gap={fresh_none_gap:.12g}",
            },
            {
                "check": "shadow_state_updated",
                "status": status(
                    metadata["optimizer_state"]["shadow_state_updates"]["diag"] > 0
                    and metadata["optimizer_state"]["shadow_state_updates"]["full"] > 0
                ),
                "severity_if_failed": "critical",
                "evidence": json.dumps(
                    metadata["optimizer_state"]["shadow_state_updates"],
                    ensure_ascii=False,
                ),
            },
            {
                "check": "exact_svd_row_orthogonality_predeclared_threshold",
                "status": status(svd_orth_max <= 1.0e-4),
                "severity_if_failed": "high",
                "evidence": (
                    f"max_residual={svd_orth_max:.12g}, threshold=0.0001; "
                    "SVD-arm conclusions are provisional"
                ),
            },
            {
                "check": "wandb_metric_coverage",
                "status": status(
                    len(wandb_summary) == 8
                    and wandb_summary["duplicate_steps"].sum() == 0
                    and wandb_summary["null_values"].sum() == 0
                ),
                "severity_if_failed": "high",
                "evidence": (
                    f"metrics={len(wandb_summary)}/8, "
                    f"duplicate_steps={int(wandb_summary['duplicate_steps'].sum())}, "
                    f"nulls={int(wandb_summary['null_values'].sum())}"
                ),
            },
            {
                "check": "wandb_single_run_bands",
                "status": status(
                    bool(band_checks)
                    and all(row["duplicates_raw_series"] for row in band_checks)
                ),
                "severity_if_failed": "low",
                "evidence": (
                    f"matching={sum(row['duplicates_raw_series'] for row in band_checks)}"
                    f"/{len(band_checks)}"
                ),
            },
        ]
    )

    def metric(name: str) -> pd.Series:
        match = wandb_summary[wandb_summary["metric"] == name]
        if len(match) != 1:
            raise ValueError((name, len(match)))
        return match.iloc[0]

    val_series = wandb_series[wandb_series["metric"] == "val/loss"]
    train_series = wandb_series[wandb_series["metric"] == "train/loss_step"]
    step = int(config["steps"][0])
    val_at_step = float(val_series[val_series["step"] == step]["value"].iloc[0])
    diag_momentum = lookup_contrast(
        contrast_summary,
        contrast="matched_none",
        candidate="ema_momentum_diag_ns5",
        eval_kind="heldout",
        eta=0.01,
    )
    full_momentum = lookup_contrast(
        contrast_summary,
        contrast="matched_none",
        candidate="ema_momentum_full_ns5",
        eval_kind="heldout",
        eta=0.01,
    )
    diag_momentum_history = lookup_contrast(
        contrast_summary,
        contrast="momentum_minus_ema_gradient",
        candidate="ema_momentum_diag_ns5",
        eval_kind="heldout",
        eta=0.01,
    )
    none_momentum_history = lookup_contrast(
        contrast_summary,
        contrast="momentum_minus_ema_gradient",
        candidate="ema_momentum_none_ns5",
        eval_kind="heldout",
        eta=0.01,
    )
    full_momentum_history = lookup_contrast(
        contrast_summary,
        contrast="momentum_minus_ema_gradient",
        candidate="ema_momentum_full_ns5",
        eval_kind="heldout",
        eta=0.01,
    )
    diag_layer_momentum = layer_detail[
        (layer_detail["contrast"] == "matched_none")
        & (layer_detail["candidate"] == "ema_momentum_diag_ns5")
        & (layer_detail["eval_kind"] == "heldout")
        & np.isclose(layer_detail["eta"], 0.01)
    ]
    diag_layer_means = {
        str(int(layer)): float(group["contrast_value"].mean())
        for layer, group in diag_layer_momentum.groupby("layer")
    }
    diag_eta2 = lookup_contrast(
        contrast_summary,
        contrast="matched_none",
        candidate="ema_momentum_diag_ns5",
        eval_kind="heldout",
        eta=0.02,
    )
    full_eta2 = lookup_contrast(
        contrast_summary,
        contrast="matched_none",
        candidate="ema_momentum_full_ns5",
        eval_kind="heldout",
        eta=0.02,
    )
    raw_diag_builds = (
        paired[
            (paired["candidate"] == "ema_momentum_diag_ns5")
            & (paired["eval_kind"] == "heldout")
            & np.isclose(paired["eta"], 0.01)
        ]
        .groupby("build_repeat")["loss_delta_vs_matched_none"]
        .mean()
        .sort_index()
        .to_numpy()
    )
    saved_diag_builds = (
        contrast_build[
            (contrast_build["contrast"] == "matched_none")
            & (contrast_build["candidate"] == "ema_momentum_diag_ns5")
            & (contrast_build["eval_kind"] == "heldout")
            & np.isclose(contrast_build["eta"], 0.01)
        ]
        .sort_values("build_repeat")["contrast_value"]
        .to_numpy()
    )
    validation_checks = pd.DataFrame(
        [
            {
                "check": "raw_to_build_repeat_reconciliation",
                "status": status(
                    np.allclose(
                        raw_diag_builds,
                        saved_diag_builds,
                        rtol=0.0,
                        atol=1.0e-15,
                    )
                ),
                "evidence": (
                    f"max_abs_gap="
                    f"{float(np.max(np.abs(raw_diag_builds-saved_diag_builds))):.12g}"
                ),
            },
            {
                "check": "layer_means_recombine_to_primary_diag_effect",
                "status": status(
                    math.isclose(
                        float(np.mean(list(diag_layer_means.values()))),
                        float(diag_momentum["mean"]),
                        rel_tol=0.0,
                        abs_tol=1.0e-15,
                    )
                ),
                "evidence": (
                    f"layer_mean={np.mean(list(diag_layer_means.values())):.12g}, "
                    f"primary={float(diag_momentum['mean']):.12g}"
                ),
            },
            {
                "check": "eta_doubling_preserves_temporal_order",
                "status": status(
                    float(diag_eta2["mean"]) < 0
                    and float(full_eta2["mean"]) > 0
                    and int(diag_eta2["negative_repeats"]) == 3
                    and int(full_eta2["positive_repeats"]) == 4
                ),
                "evidence": (
                    f"diag_ratio={float(diag_eta2['mean']/diag_momentum['mean']):.6g}, "
                    f"full_ratio={float(full_eta2['mean']/full_momentum['mean']):.6g}"
                ),
            },
            {
                "check": "historical_three_seed_order",
                "status": status(
                    bool(historical_lookup)
                    and historical_lookup["diag"] < historical_lookup["none"]
                    and historical_lookup["none"] < historical_lookup["full"]
                ),
                "evidence": json.dumps(historical_lookup, ensure_ascii=False),
            },
            {
                "check": "optimizer_and_probe_state_separation",
                "status": status(
                    float(
                        metric("matrix/cproj_k_state_bytes")["last_value"]
                    )
                    == 0.0
                    and metadata["diagnostic_shadow_k_state_bytes"] > 0
                    and metadata["diagnostic_shadow_momentum_bytes"] > 0
                ),
                "evidence": (
                    "optimizer_cproj_bytes=0; "
                    f"shadow_k_bytes={metadata['diagnostic_shadow_k_state_bytes']}; "
                    "shadow_momentum_bytes="
                    f"{metadata['diagnostic_shadow_momentum_bytes']}"
                ),
            },
            {
                "check": "svd_failure_reconciles_to_raw_max",
                "status": status(
                    math.isclose(
                        svd_orth_max,
                        float(
                            directions.loc[
                                directions["projection"] == "svd",
                                "row_orthogonality_residual",
                            ].max()
                        ),
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                    and svd_orth_max > 1.0e-4
                ),
                "evidence": (
                    f"raw_max={svd_orth_max:.12g}; threshold=0.0001"
                ),
            },
        ]
    )

    key_results = {
        "run": {
            "name": config["wandb_run_name"],
            "seed": int(config["seed"]),
            "step": step,
            "trajectory_cproj_k_mode": config["trajectory_cproj_k_mode"],
            "build_repeats": int(config["build_repeats"]),
            "layers": list(config["layers"]),
            "heldout_batches_per_build": int(config["heldout_batches"]),
            "probe_precision": config["probe_precision"],
            "shadow_state_updates": metadata["optimizer_state"][
                "shadow_state_updates"
            ],
        },
        "data_quality": {
            "status": "conditional_ready",
            "temporal_ns5_arm_ready": True,
            "exact_svd_arm_ready": False,
            "reason": (
                "All temporal NS5 controls and coverage checks pass. The exact-SVD "
                "arm misses its predeclared row-orthogonality threshold."
            ),
            "archived_source_files": int(len(inventory)),
            "direction_rows": int(len(directions)),
            "line_search_rows": int(len(paired)),
            "emitted_checks_passed": int(
                emitted_quality["status"].str.lower().eq("pass").sum()
            ),
            "emitted_checks_total": int(len(emitted_quality)),
            "analysis_checks_passed": int(
                analysis_quality["status"].eq("PASS").sum()
            ),
            "analysis_checks_total": int(len(analysis_quality)),
            "fresh_none_ema_none_max_abs_line_gap": fresh_none_gap,
            "exact_svd_row_orthogonality_residual_max": svd_orth_max,
            "exact_svd_threshold": 1.0e-4,
        },
        "primary_temporal_result_eta_0.01": {
            "independent_unit": (
                "one build-repeat mean across the three measured layers and "
                "eight shared held-out batches"
            ),
            "diag_momentum_minus_none": {
                "mean": float(diag_momentum["mean"]),
                "sd": float(diag_momentum["sd"]),
                "ci95": [
                    float(diag_momentum["ci95_low"]),
                    float(diag_momentum["ci95_high"]),
                ],
                "better_repeats": int(diag_momentum["negative_repeats"]),
                "worse_repeats": int(diag_momentum["positive_repeats"]),
                "layer_means": diag_layer_means,
            },
            "full_momentum_minus_none": {
                "mean": float(full_momentum["mean"]),
                "sd": float(full_momentum["sd"]),
                "ci95": [
                    float(full_momentum["ci95_low"]),
                    float(full_momentum["ci95_high"]),
                ],
                "better_repeats": int(full_momentum["negative_repeats"]),
                "worse_repeats": int(full_momentum["positive_repeats"]),
            },
            "momentum_minus_ema_gradient": {
                "none": {
                    "mean": float(none_momentum_history["mean"]),
                    "negative_repeats": int(
                        none_momentum_history["negative_repeats"]
                    ),
                },
                "diag": {
                    "mean": float(diag_momentum_history["mean"]),
                    "negative_repeats": int(
                        diag_momentum_history["negative_repeats"]
                    ),
                },
                "full": {
                    "mean": float(full_momentum_history["mean"]),
                    "negative_repeats": int(
                        full_momentum_history["negative_repeats"]
                    ),
                },
            },
        },
        "trajectory_and_resources": {
            "val_loss_step10000": val_at_step,
            "best_val_loss_through_step10000": float(val_series["value"].min()),
            "best_val_step_through_step10000": int(
                val_series.loc[val_series["value"].idxmin(), "step"]
            ),
            "train_loss_last_logged": float(train_series.iloc[-1]["value"]),
            "train_loss_last_step": int(train_series.iloc[-1]["step"]),
            "training_elapsed_last_seconds": float(metric("time_elapsed")["last_value"]),
            "training_elapsed_last_step": int(metric("time_elapsed")["step_max"]),
            "full_run_peak_memory_mib": float(
                metric("cuda/full_run_max_memory_allocated_mib")["last_value"]
            ),
            "probe_seconds": float(metadata["probe_seconds"]),
            "probe_peak_memory_mib": float(metadata["probe_peak_memory_mib"]),
            "optimizer_k_state_mib": float(
                metric("matrix/k_state_bytes")["last_value"] / (1024 * 1024)
            ),
            "optimizer_cproj_k_state_mib": float(
                metric("matrix/cproj_k_state_bytes")["last_value"]
                / (1024 * 1024)
            ),
            "optimizer_k_state_released_fraction": float(
                metric("matrix/k_state_released_fraction")["last_value"]
            ),
            "diagnostic_shadow_k_state_mib": float(
                metadata["diagnostic_shadow_k_state_bytes"] / (1024 * 1024)
            ),
            "diagnostic_shadow_momentum_mib": float(
                metadata["diagnostic_shadow_momentum_bytes"] / (1024 * 1024)
            ),
        },
        "historical_long_run_step10000": {
            "mean_val_loss_by_mode": historical_lookup,
            "diag_minus_none": (
                historical_lookup.get("diag", math.nan)
                - historical_lookup.get("none", math.nan)
            ),
            "full_minus_none": (
                historical_lookup.get("full", math.nan)
                - historical_lookup.get("none", math.nan)
            ),
        },
        "conclusion": {
            "supported": [
                (
                    "Adding the historical momentum buffer exposes the qualitative "
                    "held-out ordering diag approximately none, both better than full."
                ),
                (
                    "The momentum-history benefit is repeat-consistent for none and "
                    "diag, but absent for full, indicating a K-by-history interaction."
                ),
                (
                    "The small diag advantage is localized to layer 0 in this "
                    "three-layer probe; layers 11 and 23 are slightly worse than none."
                ),
                (
                    "The fresh-gradient arm reproduces P1's same-batch structured-K "
                    "penalty and near-zero held-out difference."
                ),
                (
                    "The exact-SVD arm does not suggest NS5 truncation is the source "
                    "of the structured-K penalty, but this statement remains "
                    "provisional because the SVD orthogonality gate failed."
                ),
            ],
            "not_supported": [
                "A seed-general, all-layer claim that diag is better than none.",
                "A causal claim that momentum history alone explains long-run gains.",
                "A publication-ready exact-SVD conclusion before the tolerance failure is fixed.",
                "Treating the 96 held-out layer-batch rows as independent replicates.",
            ],
            "seed_decision": (
                "Do not run unchanged P2 on seeds 2024/2025 yet. The diag-minus-none "
                "four-repeat interval crosses zero and the advantage is layer-0-driven. "
                "First expand layer coverage on seed2026 and repair the SVD arm; expand "
                "seeds only if the temporal ordering survives those checks."
            ),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    inventory.to_csv(processed / "source_inventory.csv", index=False)
    wandb_series.to_csv(processed / "wandb_series_long.csv", index=False)
    wandb_summary.to_csv(processed / "wandb_metric_summary.csv", index=False)
    emitted_quality.to_csv(
        processed / "emitted_probe_data_quality_checks.csv", index=False
    )
    analysis_quality.to_csv(processed / "data_quality_checks.csv", index=False)
    candidate_build.to_csv(
        processed / "candidate_build_repeat_effects.csv", index=False
    )
    contrast_build.to_csv(
        processed / "mechanism_contrasts_by_build_repeat.csv", index=False
    )
    contrast_summary.to_csv(
        processed / "mechanism_contrast_summary.csv", index=False
    )
    layer_detail.to_csv(processed / "mechanism_contrasts_by_layer.csv", index=False)
    geometry.to_csv(processed / "candidate_geometry_summary.csv", index=False)
    p1_compare.to_csv(processed / "p1_p2_fresh_arm_comparison.csv", index=False)
    historical_rows.to_csv(
        processed / "historical_step10000_rows.csv", index=False
    )
    validation_checks.to_csv(processed / "validation_checks.csv", index=False)
    write_json(processed / "key_results.json", key_results)

    print(f"Wrote P2 processed analysis to {processed}")
    print(analysis_quality[["check", "status", "evidence"]].to_string(index=False))
    print(
        "\nSeed gate: FAIL for unchanged P2; expand layer coverage and repair "
        "the SVD arm first."
    )


if __name__ == "__main__":
    main()
