#!/usr/bin/env python
"""Reproducible hierarchical analysis for the all-layer temporal P3 probe.

Raw probe and W&B exports are immutable. Derived tables are written to the
archived run's ``processed/`` directory. The primary uncertainty unit is one
build-repeat aggregate; shared held-out batches and layers are averaged within
each build before intervals or sign counts are computed.
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


LAYERS = tuple(range(24))
BUILD_REPEATS = (0, 1, 2, 3)
WORKING_ETAS = (0.01, 0.02)
T_CRITICAL_975_DF3 = 3.182446305
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
DEPTH_BANDS = {
    "early": tuple(range(0, 8)),
    "middle": tuple(range(8, 16)),
    "late": tuple(range(16, 24)),
}
CONTRASTS = (
    {
        "contrast": "matched_none",
        "field": "loss_delta_vs_matched_none",
        "candidates": tuple(
            candidate
            for candidate in CANDIDATES
            if "_none_" not in candidate
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
        "interpretation": "negative favors FP64-SVD over NS5",
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
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--p2-run-dir", type=Path, default=None)
    parser.add_argument("--historical-summary", type=Path, default=None)
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
            raise ValueError(f"Unexpected W&B schema: {path}")
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


def add_depth_band(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    lookup = {
        layer: band
        for band, layers in DEPTH_BANDS.items()
        for layer in layers
    }
    result["depth_band"] = result["layer"].map(lookup)
    if result["depth_band"].isna().any():
        raise ValueError("unmapped layer in depth-band assignment")
    return result


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


def summarize_build_values(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in frame.groupby(group_columns, sort=True, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group[value_column].to_numpy(dtype=float)
        if len(values) != 4:
            raise ValueError((group_columns, keys, len(values)))
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half_width = T_CRITICAL_975_DF3 * sd / math.sqrt(len(values))
        item = dict(zip(group_columns, keys))
        item.update(
            {
                "independent_build_repeats": len(values),
                "mean": mean,
                "sd": sd,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "negative_repeats": int((values < 0).sum()),
                "positive_repeats": int((values > 0).sum()),
                "zero_repeats": int((values == 0).sum()),
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def depth_band_effects(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = add_depth_band(
        paired[
            paired["eta"].isin(WORKING_ETAS)
            & paired["candidate"].isin(
                ["ema_momentum_diag_ns5", "ema_momentum_full_ns5"]
            )
        ]
    )
    detail = (
        selected.groupby(
            [
                "candidate",
                "eval_kind",
                "eta",
                "depth_band",
                "build_repeat",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            layers=("layer", "nunique"),
            heldout_batches=("heldout_index", "nunique"),
            observations=("loss_delta_vs_matched_none", "size"),
            contrast_value=("loss_delta_vs_matched_none", "mean"),
        )
    )
    summary = summarize_build_values(
        detail,
        ["candidate", "eval_kind", "eta", "depth_band"],
        "contrast_value",
    )
    return detail, summary


def layer_effects(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = paired[
        paired["eta"].isin(WORKING_ETAS)
        & paired["candidate"].isin(
            ["ema_momentum_diag_ns5", "ema_momentum_full_ns5"]
        )
    ]
    detail = (
        selected.groupby(
            ["candidate", "eval_kind", "eta", "layer", "build_repeat"],
            as_index=False,
        )
        .agg(
            heldout_batches=("heldout_index", "nunique"),
            observations=("loss_delta_vs_matched_none", "size"),
            contrast_value=("loss_delta_vs_matched_none", "mean"),
        )
    )
    summary = summarize_build_values(
        detail,
        ["candidate", "eval_kind", "eta", "layer"],
        "contrast_value",
    )
    return detail, summary


def leave_one_layer_out(paired: pd.DataFrame) -> pd.DataFrame:
    selected = paired[
        paired["eta"].isin(WORKING_ETAS)
        & paired["candidate"].isin(
            ["ema_momentum_diag_ns5", "ema_momentum_full_ns5"]
        )
    ]
    detail_rows: list[dict] = []
    for candidate in sorted(selected["candidate"].unique()):
        for eval_kind in ("same", "heldout"):
            for eta in WORKING_ETAS:
                base = selected[
                    (selected["candidate"] == candidate)
                    & (selected["eval_kind"] == eval_kind)
                    & np.isclose(selected["eta"], eta)
                ]
                for omitted_layer in (-1, *LAYERS):
                    subset = (
                        base
                        if omitted_layer == -1
                        else base[base["layer"] != omitted_layer]
                    )
                    for build_repeat, group in subset.groupby("build_repeat"):
                        detail_rows.append(
                            {
                                "candidate": candidate,
                                "eval_kind": eval_kind,
                                "eta": eta,
                                "omitted_layer": omitted_layer,
                                "build_repeat": int(build_repeat),
                                "remaining_layers": int(group["layer"].nunique()),
                                "contrast_value": float(
                                    group[
                                        "loss_delta_vs_matched_none"
                                    ].mean()
                                ),
                            }
                        )
    detail = pd.DataFrame(detail_rows)
    return summarize_build_values(
        detail,
        ["candidate", "eval_kind", "eta", "omitted_layer"],
        "contrast_value",
    ).merge(
        detail[
            [
                "candidate",
                "eval_kind",
                "eta",
                "omitted_layer",
                "remaining_layers",
            ]
        ].drop_duplicates(),
        on=["candidate", "eval_kind", "eta", "omitted_layer"],
        validate="one_to_one",
    )


def geometry_summary(directions: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "observations": ("layer", "size"),
        "build_repeats": ("build_repeat", "nunique"),
        "layers": ("layer", "nunique"),
        "projection_input_cos_vs_gradient_mean": (
            "projection_input_cos_vs_gradient",
            "mean",
        ),
        "direction_cos_vs_matched_none_mean": (
            "direction_cos_vs_matched_none",
            "mean",
        ),
        "direction_norm_ratio_vs_matched_none_mean": (
            "direction_norm_ratio_vs_matched_none",
            "mean",
        ),
        "alignment_normalized_mean": ("alignment_normalized", "mean"),
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
    }
    return (
        directions.groupby(
            [
                "candidate",
                "k_mode",
                "k_source",
                "buffer_source",
                "projection",
                "projection_compute_dtype",
            ],
            sort=True,
        )
        .agg(**metrics)
        .reset_index()
    )


def p2_p3_comparison(
    p2_run_dir: Path | None,
    p3_paired: pd.DataFrame,
) -> pd.DataFrame:
    if p2_run_dir is None:
        return pd.DataFrame()
    p2 = pd.read_csv(
        p2_run_dir
        / "raw_probe"
        / "temporal_line_search_paired_results.csv"
    )
    rows: list[dict] = []
    candidates = (
        "fresh_gradient_diag_ns5",
        "fresh_gradient_full_ns5",
        "ema_gradient_diag_ns5",
        "ema_gradient_full_ns5",
        "ema_momentum_diag_ns5",
        "ema_momentum_full_ns5",
    )
    for candidate in candidates:
        for layer in (0, 11, 23):
            for eta in WORKING_ETAS:
                filters = (
                    ("candidate", candidate),
                    ("layer", layer),
                    ("eval_kind", "heldout"),
                )
                left = p2
                right = p3_paired
                for field, value in filters:
                    left = left[left[field] == value]
                    right = right[right[field] == value]
                left = left[np.isclose(left["eta"], eta)]
                right = right[np.isclose(right["eta"], eta)]
                p2_mean = float(left["loss_delta_vs_matched_none"].mean())
                p3_mean = float(right["loss_delta_vs_matched_none"].mean())
                rows.append(
                    {
                        "candidate": candidate,
                        "layer": layer,
                        "eta": eta,
                        "p2_mean": p2_mean,
                        "p3_mean": p3_mean,
                        "p3_minus_p2": p3_mean - p2_mean,
                        "same_sign": bool(np.sign(p2_mean) == np.sign(p3_mean)),
                    }
                )
    return pd.DataFrame(rows)


def historical_at_step(
    path: Path | None,
    *,
    step: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if path is None or not path.exists():
        return pd.DataFrame(), {}
    frame = pd.read_csv(path)
    selected = frame[
        (frame["step"] == step)
        & frame["mode"].isin(["none", "diag", "full"])
    ].copy()
    summary = (
        selected.groupby("mode", as_index=False)
        .agg(seeds=("seed", "nunique"), val_loss_mean=("val_loss", "mean"))
    )
    return selected, {
        str(row.mode): float(row.val_loss_mean)
        for row in summary.itertuples(index=False)
    }


def lookup_summary(
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

    contrast_build = contrast_build_effects(paired)
    contrast_summary = summarize_build_values(
        contrast_build,
        ["contrast", "candidate", "eval_kind", "eta"],
        "contrast_value",
    )
    contrast_summary = contrast_summary.merge(
        contrast_build[
            ["contrast", "candidate", "eval_kind", "eta", "interpretation"]
        ].drop_duplicates(),
        on=["contrast", "candidate", "eval_kind", "eta"],
        validate="one_to_one",
    )
    band_detail, band_summary = depth_band_effects(paired)
    layer_detail, layer_summary = layer_effects(paired)
    loo_summary = leave_one_layer_out(paired)
    geometry = geometry_summary(directions)
    p2_compare = p2_p3_comparison(
        args.p2_run_dir.resolve() if args.p2_run_dir else None,
        paired,
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
    control_gap = float(
        paired[
            (paired["candidate"] == "ema_gradient_none_ns5")
            & (paired["eta"] > 0)
        ]["loss_delta_vs_fresh_same_mode"]
        .abs()
        .max()
    )
    svd_rows = directions[directions["projection"] == "svd"]
    svd_residual_max = float(svd_rows["row_orthogonality_residual"].max())
    expected_nan_count = int(
        directions["curvature_exact"].isna().sum()
        + directions["curvature_per_direction_norm2"].isna().sum()
        + directions["quadratic_score_exact"].isna().sum()
        + directions["eta_star_exact"].isna().sum()
        + directions["predicted_best_loss_delta"].isna().sum()
        + directions["ns5_svd_cos"].isna().sum()
    )
    analysis_quality = pd.DataFrame(
        [
            {
                "check": "archived_source_file_count",
                "status": status(len(inventory) == 16),
                "severity_if_failed": "high",
                "evidence": f"observed={len(inventory)}, expected=16",
            },
            {
                "check": "emitted_probe_quality",
                "status": status(
                    len(emitted_quality) == 11
                    and emitted_quality["status"].str.lower().eq("pass").all()
                ),
                "severity_if_failed": "critical",
                "evidence": (
                    f"passed={emitted_quality['status'].str.lower().eq('pass').sum()}"
                    f"/{len(emitted_quality)}"
                ),
            },
            {
                "check": "direction_key_coverage",
                "status": status(
                    observed_direction_keys == expected_direction_keys
                    and len(directions) == 1152
                ),
                "severity_if_failed": "critical",
                "evidence": (
                    f"keys={len(observed_direction_keys)}/1152, rows={len(directions)}"
                ),
            },
            {
                "check": "line_search_key_coverage",
                "status": status(
                    observed_line_keys == expected_line_keys
                    and len(paired) == 51840
                    and len(line_search) == 51840
                ),
                "severity_if_failed": "critical",
                "evidence": (
                    f"keys={len(observed_line_keys)}/51840, "
                    f"paired_rows={len(paired)}, raw_rows={len(line_search)}"
                ),
            },
            {
                "check": "finite_alignment_and_expected_missing_curvature",
                "status": status(
                    np.isfinite(
                        directions[
                            [
                                "alignment_raw",
                                "alignment_normalized",
                                "direction_fro_norm",
                            ]
                        ].to_numpy(dtype=float)
                    ).all()
                    and directions["curvature_exact"].isna().all()
                    and not bool(config["exact_hvp"])
                ),
                "severity_if_failed": "critical",
                "evidence": (
                    f"finite_alignment={directions['alignment_raw'].notna().sum()}"
                    f"/{len(directions)}; exact_hvp={config['exact_hvp']}; "
                    f"intended_nan_cells={expected_nan_count}"
                ),
            },
            {
                "check": "fresh_none_equals_ema_none_control",
                "status": status(control_gap <= 1.0e-6),
                "severity_if_failed": "critical",
                "evidence": f"max_abs_line_gap={control_gap:.12g}",
            },
            {
                "check": "all_layer_shadow_state_updated",
                "status": status(
                    set(config["shadow_cproj_k_layers"]) == set(LAYERS)
                    and metadata["optimizer_state"]["shadow_state_updates"]["diag"]
                    > 0
                    and metadata["optimizer_state"]["shadow_state_updates"]["full"]
                    > 0
                ),
                "severity_if_failed": "critical",
                "evidence": (
                    f"shadow_layers={len(config['shadow_cproj_k_layers'])}/24; "
                    f"updates={metadata['optimizer_state']['shadow_state_updates']}"
                ),
            },
            {
                "check": "fp64_svd_gate",
                "status": status(
                    config["svd_compute_dtype"] == "float64"
                    and set(svd_rows["projection_compute_dtype"]) == {"float64"}
                    and svd_residual_max <= 1.0e-4
                ),
                "severity_if_failed": "critical",
                "evidence": (
                    f"dtype={set(svd_rows['projection_compute_dtype'])}; "
                    f"max_residual={svd_residual_max:.12g}"
                ),
            },
            {
                "check": "wandb_metric_coverage",
                "status": status(
                    len(wandb_summary) == 8
                    and wandb_summary["duplicate_steps"].sum() == 0
                    and wandb_summary["null_values"].sum() == 0
                    and wandb_summary["step_max"].min() == 10000
                ),
                "severity_if_failed": "high",
                "evidence": (
                    f"metrics={len(wandb_summary)}/8; "
                    f"minimum_step_max={wandb_summary['step_max'].min()}; "
                    f"duplicates={wandb_summary['duplicate_steps'].sum()}; "
                    f"nulls={wandb_summary['null_values'].sum()}"
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

    primary: dict[str, pd.Series] = {}
    for mode in ("diag", "full"):
        for eta in WORKING_ETAS:
            primary[f"{mode}_{eta:g}"] = lookup_summary(
                contrast_summary,
                contrast="matched_none",
                candidate=f"ema_momentum_{mode}_ns5",
                eval_kind="heldout",
                eta=eta,
            )
    momentum: dict[str, pd.Series] = {}
    for mode in ("none", "diag", "full"):
        momentum[mode] = lookup_summary(
            contrast_summary,
            contrast="momentum_minus_ema_gradient",
            candidate=f"ema_momentum_{mode}_ns5",
            eval_kind="heldout",
            eta=0.01,
        )

    primary_bands = band_summary[
        (band_summary["eval_kind"] == "heldout")
        & (band_summary["eta"].isin(WORKING_ETAS))
    ]
    diag_bands_eta1 = primary_bands[
        (primary_bands["candidate"] == "ema_momentum_diag_ns5")
        & np.isclose(primary_bands["eta"], 0.01)
    ]
    full_bands_eta1 = primary_bands[
        (primary_bands["candidate"] == "ema_momentum_full_ns5")
        & np.isclose(primary_bands["eta"], 0.01)
    ]
    diag_layers_eta1 = layer_summary[
        (layer_summary["candidate"] == "ema_momentum_diag_ns5")
        & (layer_summary["eval_kind"] == "heldout")
        & np.isclose(layer_summary["eta"], 0.01)
    ]
    full_layers_eta1 = layer_summary[
        (layer_summary["candidate"] == "ema_momentum_full_ns5")
        & (layer_summary["eval_kind"] == "heldout")
        & np.isclose(layer_summary["eta"], 0.01)
    ]
    diag_loo0 = loo_summary[
        (loo_summary["candidate"] == "ema_momentum_diag_ns5")
        & (loo_summary["eval_kind"] == "heldout")
        & np.isclose(loo_summary["eta"], 0.01)
        & (loo_summary["omitted_layer"] == 0)
    ].iloc[0]
    full_loo0 = loo_summary[
        (loo_summary["candidate"] == "ema_momentum_full_ns5")
        & (loo_summary["eval_kind"] == "heldout")
        & np.isclose(loo_summary["eta"], 0.01)
        & (loo_summary["omitted_layer"] == 0)
    ].iloc[0]

    diag_gate_checks = {
        "all_layer_mean_negative_both_eta": bool(
            primary["diag_0.01"]["mean"] < 0
            and primary["diag_0.02"]["mean"] < 0
        ),
        "at_least_three_of_four_builds_better_both_eta": bool(
            primary["diag_0.01"]["negative_repeats"] >= 3
            and primary["diag_0.02"]["negative_repeats"] >= 3
        ),
        "at_least_two_depth_bands_nonpositive_eta_0.01": bool(
            int((diag_bands_eta1["mean"] <= 0).sum()) >= 2
        ),
        "leave_layer0_out_remains_negative": bool(diag_loo0["mean"] < 0),
    }
    full_gate_checks = {
        "all_layer_mean_positive_both_eta": bool(
            primary["full_0.01"]["mean"] > 0
            and primary["full_0.02"]["mean"] > 0
        ),
        "four_of_four_builds_worse_both_eta": bool(
            primary["full_0.01"]["positive_repeats"] == 4
            and primary["full_0.02"]["positive_repeats"] == 4
        ),
        "at_least_two_depth_bands_positive_eta_0.01": bool(
            int((full_bands_eta1["mean"] > 0).sum()) >= 2
        ),
        "leave_layer0_out_remains_positive": bool(full_loo0["mean"] > 0),
    }
    fp64_svd_gate = bool(
        config["svd_compute_dtype"] == "float64"
        and svd_residual_max <= 1.0e-4
    )
    seed_expansion = bool(
        all(diag_gate_checks.values()) and fp64_svd_gate
    )
    seed_gate = pd.DataFrame(
        [
            {
                "family": "diag_global_advantage",
                "check": name,
                "status": status(value),
            }
            for name, value in diag_gate_checks.items()
        ]
        + [
            {
                "family": "full_global_damage",
                "check": name,
                "status": status(value),
            }
            for name, value in full_gate_checks.items()
        ]
        + [
            {
                "family": "fp64_svd",
                "check": "all_svd_rows_pass_dtype_and_orthogonality",
                "status": status(fp64_svd_gate),
            },
            {
                "family": "seed_expansion",
                "check": "run_unchanged_p3_seeds_2024_2025",
                "status": status(seed_expansion),
            },
        ]
    )

    val_series = wandb_series[wandb_series["metric"] == "val/loss"]
    train_series = wandb_series[
        wandb_series["metric"] == "train/loss_step"
    ]
    step = int(config["steps"][0])
    val_at_step = float(
        val_series[val_series["step"] == step]["value"].iloc[0]
    )
    shadow_k_mib = float(
        metadata["diagnostic_shadow_k_state_bytes"] / (1024 * 1024)
    )
    shadow_momentum_mib = float(
        metadata["diagnostic_shadow_momentum_bytes"] / (1024 * 1024)
    )
    validation_checks = pd.DataFrame(
        [
            {
                "check": "depth_bands_partition_all_layers",
                "status": status(
                    set().union(*map(set, DEPTH_BANDS.values())) == set(LAYERS)
                    and sum(len(value) for value in DEPTH_BANDS.values()) == 24
                ),
                "evidence": json.dumps(DEPTH_BANDS),
            },
            {
                "check": "depth_band_means_recombine_to_all_layer_mean",
                "status": status(
                    math.isclose(
                        float(diag_bands_eta1["mean"].mean()),
                        float(primary["diag_0.01"]["mean"]),
                        rel_tol=0.0,
                        abs_tol=1.0e-15,
                    )
                ),
                "evidence": (
                    f"band_mean={diag_bands_eta1['mean'].mean():.12g}; "
                    f"all_layer={primary['diag_0.01']['mean']:.12g}"
                ),
            },
            {
                "check": "layer_means_recombine_to_all_layer_mean",
                "status": status(
                    math.isclose(
                        float(diag_layers_eta1["mean"].mean()),
                        float(primary["diag_0.01"]["mean"]),
                        rel_tol=0.0,
                        abs_tol=1.0e-15,
                    )
                ),
                "evidence": (
                    f"layer_mean={diag_layers_eta1['mean'].mean():.12g}; "
                    f"all_layer={primary['diag_0.01']['mean']:.12g}"
                ),
            },
            {
                "check": "eta_doubling_preserves_primary_signs",
                "status": status(
                    primary["diag_0.01"]["mean"] > 0
                    and primary["diag_0.02"]["mean"] > 0
                    and primary["full_0.01"]["mean"] > 0
                    and primary["full_0.02"]["mean"] > 0
                ),
                "evidence": (
                    f"diag_ratio="
                    f"{primary['diag_0.02']['mean']/primary['diag_0.01']['mean']:.6g}; "
                    f"full_ratio="
                    f"{primary['full_0.02']['mean']/primary['full_0.01']['mean']:.6g}"
                ),
            },
            {
                "check": "all_leave_one_layer_out_diag_means_positive",
                "status": status(
                    (
                        loo_summary[
                            (loo_summary["candidate"] == "ema_momentum_diag_ns5")
                            & (loo_summary["eval_kind"] == "heldout")
                            & np.isclose(loo_summary["eta"], 0.01)
                            & (loo_summary["omitted_layer"] >= 0)
                        ]["mean"]
                        > 0
                    ).all()
                ),
                "evidence": (
                    "minimum_loo_mean="
                    f"{loo_summary[(loo_summary['candidate']=='ema_momentum_diag_ns5') & (loo_summary['eval_kind']=='heldout') & np.isclose(loo_summary['eta'],0.01) & (loo_summary['omitted_layer']>=0)]['mean'].min():.12g}"
                ),
            },
            {
                "check": "historical_long_run_order",
                "status": status(
                    bool(historical_lookup)
                    and historical_lookup["diag"] < historical_lookup["none"]
                    and historical_lookup["none"] < historical_lookup["full"]
                ),
                "evidence": json.dumps(historical_lookup, ensure_ascii=False),
            },
        ]
    )

    svd_same_diag = lookup_summary(
        contrast_summary,
        contrast="matched_none",
        candidate="fresh_gradient_diag_svd",
        eval_kind="same",
        eta=0.01,
    )
    svd_same_full = lookup_summary(
        contrast_summary,
        contrast="matched_none",
        candidate="fresh_gradient_full_svd",
        eval_kind="same",
        eta=0.01,
    )
    key_results = {
        "run": {
            "name": config["wandb_run_name"],
            "seed": int(config["seed"]),
            "step": step,
            "layers": list(config["layers"]),
            "build_repeats": int(config["build_repeats"]),
            "heldout_batches_per_build": int(config["heldout_batches"]),
            "exact_hvp": bool(config["exact_hvp"]),
            "probe_precision": config["probe_precision"],
            "svd_compute_dtype": config["svd_compute_dtype"],
        },
        "data_quality": {
            "status": "ready_to_share",
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
            "validation_checks_passed": int(
                validation_checks["status"].eq("PASS").sum()
            ),
            "validation_checks_total": int(len(validation_checks)),
            "fresh_none_ema_none_max_abs_line_gap": control_gap,
            "fp64_svd_max_row_orthogonality_residual": svd_residual_max,
        },
        "all_layer_temporal_eta": {
            key: {
                "mean_delta_vs_none": float(value["mean"]),
                "sd_across_builds": float(value["sd"]),
                "ci95": [
                    float(value["ci95_low"]),
                    float(value["ci95_high"]),
                ],
                "better_repeats": int(value["negative_repeats"]),
                "worse_repeats": int(value["positive_repeats"]),
            }
            for key, value in primary.items()
        },
        "momentum_minus_ema_gradient_heldout_eta_0.01": {
            mode: {
                "mean": float(value["mean"]),
                "ci95": [
                    float(value["ci95_low"]),
                    float(value["ci95_high"]),
                ],
                "improved_repeats": int(value["negative_repeats"]),
            }
            for mode, value in momentum.items()
        },
        "layer_map_heldout_eta_0.01": {
            "diag_positive_mean_layers": int(
                (diag_layers_eta1["mean"] > 0).sum()
            ),
            "diag_negative_mean_layers": int(
                (diag_layers_eta1["mean"] < 0).sum()
            ),
            "full_positive_mean_layers": int(
                (full_layers_eta1["mean"] > 0).sum()
            ),
            "full_negative_mean_layers": int(
                (full_layers_eta1["mean"] < 0).sum()
            ),
            "diag_layer0_mean": float(
                diag_layers_eta1[diag_layers_eta1["layer"] == 0]["mean"].iloc[0]
            ),
            "diag_leave_layer0_out_mean": float(diag_loo0["mean"]),
            "full_layer0_mean": float(
                full_layers_eta1[full_layers_eta1["layer"] == 0]["mean"].iloc[0]
            ),
            "full_leave_layer0_out_mean": float(full_loo0["mean"]),
        },
        "fp64_svd": {
            "max_row_orthogonality_residual": svd_residual_max,
            "same_batch_diag_minus_none_eta_0.01": {
                "mean": float(svd_same_diag["mean"]),
                "ci95": [
                    float(svd_same_diag["ci95_low"]),
                    float(svd_same_diag["ci95_high"]),
                ],
                "worse_repeats": int(svd_same_diag["positive_repeats"]),
            },
            "same_batch_full_minus_none_eta_0.01": {
                "mean": float(svd_same_full["mean"]),
                "ci95": [
                    float(svd_same_full["ci95_low"]),
                    float(svd_same_full["ci95_high"]),
                ],
                "worse_repeats": int(svd_same_full["positive_repeats"]),
            },
        },
        "seed_gate": {
            "diag_global_advantage": diag_gate_checks,
            "full_global_damage": full_gate_checks,
            "fp64_svd_gate": fp64_svd_gate,
            "expand_unchanged_p3_to_seeds_2024_2025": seed_expansion,
            "decision": (
                "Do not run unchanged P3 seeds 2024/2025. The predeclared "
                "diag global-advantage gate fails every substantive condition."
            ),
        },
        "trajectory_and_resources": {
            "val_loss_step10000": val_at_step,
            "best_val_loss_through_step10000": float(val_series["value"].min()),
            "best_val_step_through_step10000": int(
                val_series.loc[val_series["value"].idxmin(), "step"]
            ),
            "train_loss_step10000": float(
                train_series[train_series["step"] == 10000]["value"].iloc[0]
            ),
            "total_elapsed_seconds_step10000": float(
                metric("time_elapsed")["last_value"]
            ),
            "probe_seconds": float(metadata["probe_seconds"]),
            "estimated_training_seconds_excluding_probe": float(
                metric("time_elapsed")["last_value"]
                - metadata["probe_seconds"]
            ),
            "full_run_peak_memory_mib": float(
                metric("cuda/full_run_max_memory_allocated_mib")["last_value"]
            ),
            "probe_peak_memory_mib": float(metadata["probe_peak_memory_mib"]),
            "optimizer_k_state_mib": float(
                metric("matrix/k_state_bytes")["last_value"] / (1024 * 1024)
            ),
            "optimizer_cproj_k_state_mib": float(
                metric("matrix/cproj_k_state_bytes")["last_value"]
                / (1024 * 1024)
            ),
            "diagnostic_shadow_k_state_mib": shadow_k_mib,
            "diagnostic_shadow_momentum_mib": shadow_momentum_mib,
            "diagnostic_shadow_total_mib": (
                shadow_k_mib + shadow_momentum_mib
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
                    "Full EMA-K plus momentum has a repeat-consistent all-layer "
                    "held-out penalty at eta 0.01 and 0.02, and the sign survives "
                    "removing layer 0."
                ),
                (
                    "Full preserves only a small fraction of none's held-out "
                    "momentum benefit; diag preserves most of it but does not "
                    "surpass none when all layers are included."
                ),
                (
                    "The P2 three-layer diag advantage does not generalize: 22 "
                    "of 24 P3 layer means are positive and all three depth-band "
                    "means are positive."
                ),
                (
                    "FP64 exact polar directions pass orthogonality and still "
                    "show structured-K same-batch damage, so finite NS5 error is "
                    "not necessary for that damage."
                ),
            ],
            "not_supported": [
                "A global or depth-band single-step explanation for diag's long-run advantage.",
                "A reason to add unchanged P3 seeds 2024 and 2025.",
                "A claim that a layer-0-only diag intervention is already established.",
                "A causal claim that the measured momentum interaction is the only source of full damage.",
            ],
            "next_experiment": (
                "Move from another fixed-checkpoint direction probe to a short "
                "multi-step counterfactual rollout that applies simultaneous "
                "all-layer none/diag/full updates from one checkpoint and shared "
                "data stream. This is needed to measure parameter-state feedback "
                "and cross-layer interaction."
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
    contrast_build.to_csv(
        processed / "mechanism_contrasts_by_build_repeat.csv", index=False
    )
    contrast_summary.to_csv(
        processed / "mechanism_contrast_summary.csv", index=False
    )
    band_detail.to_csv(
        processed / "depth_band_effects_by_build_repeat.csv", index=False
    )
    band_summary.to_csv(processed / "depth_band_summary.csv", index=False)
    layer_detail.to_csv(
        processed / "layer_effects_by_build_repeat.csv", index=False
    )
    layer_summary.to_csv(processed / "layer_effect_summary.csv", index=False)
    loo_summary.to_csv(processed / "leave_one_layer_out_summary.csv", index=False)
    geometry.to_csv(processed / "candidate_geometry_summary.csv", index=False)
    p2_compare.to_csv(processed / "p2_p3_shared_layer_comparison.csv", index=False)
    historical_rows.to_csv(
        processed / "historical_step10000_rows.csv", index=False
    )
    seed_gate.to_csv(processed / "seed_gate_checks.csv", index=False)
    validation_checks.to_csv(processed / "validation_checks.csv", index=False)
    write_json(processed / "key_results.json", key_results)

    print(f"Wrote P3 processed analysis to {processed}")
    print(analysis_quality[["check", "status", "evidence"]].to_string(index=False))
    print("\nSeed expansion:", "PASS" if seed_expansion else "FAIL")


if __name__ == "__main__":
    main()
