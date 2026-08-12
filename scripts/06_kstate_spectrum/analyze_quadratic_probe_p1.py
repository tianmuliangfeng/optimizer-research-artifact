#!/usr/bin/env python
"""Reproducible hierarchical analysis for the c_proj quadratic probe P1.

Raw probe and W&B exports are treated as immutable. Derived tables are written
to ``processed/`` below one archived run directory.
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


BASE_MODES = ("none", "scalar", "diag", "block4", "full")
ALL_MODES = (
    "none",
    "scalar",
    "diag",
    "block4",
    "full",
    "none_repeat",
    "diag_normmatch",
    "block4_normmatch",
    "full_normmatch",
)
LAYERS = (0, 11, 23)
BUILD_REPEATS = (0, 1, 2, 3)
WORKING_MULTIPLIERS = (1.0, 2.0)
T_CRITICAL_975_DF3 = 3.182446305


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Archived P1 run containing raw_probe/ and raw_wandb_exports/.",
    )
    parser.add_argument(
        "--p0-run-dir",
        type=Path,
        default=None,
        help="Optional archived P0 run for the numerical-control comparison.",
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


def load_wandb_exports(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    long_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
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
        long_frames.append(compact)
        non_null = compact["value"].dropna()
        summary_rows.append(
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
        pd.concat(long_frames, ignore_index=True),
        pd.DataFrame(summary_rows).sort_values("metric").reset_index(drop=True),
        band_checks,
    )


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


def build_repeat_effects(paired: pd.DataFrame) -> pd.DataFrame:
    selected = paired[
        paired["lr_multiplier"].isin(WORKING_MULTIPLIERS)
    ].copy()
    return (
        selected.groupby(
            ["mode", "eval_kind", "lr_multiplier", "eta", "build_repeat"],
            sort=True,
            as_index=False,
        )
        .agg(
            layers=("layer", "nunique"),
            eval_batches=("heldout_index", "nunique"),
            observations=("loss_delta", "size"),
            loss_delta_mean=("loss_delta", "mean"),
            delta_vs_none_mean=("loss_delta_vs_none", "mean"),
            delta_vs_none_min=("loss_delta_vs_none", "min"),
            delta_vs_none_max=("loss_delta_vs_none", "max"),
        )
    )


def working_point_summary(build_effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in build_effects.groupby(
        ["mode", "eval_kind", "lr_multiplier", "eta"], sort=True
    ):
        mode, eval_kind, multiplier, eta = keys
        values = group["delta_vs_none_mean"].to_numpy(dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half_width = T_CRITICAL_975_DF3 * sd / math.sqrt(len(values))
        rows.append(
            {
                "mode": mode,
                "eval_kind": eval_kind,
                "lr_multiplier": float(multiplier),
                "eta": float(eta),
                "independent_build_repeats": len(values),
                "repeat_mean_loss_delta": float(group["loss_delta_mean"].mean()),
                "repeat_mean_delta_vs_none": mean,
                "repeat_sd_delta_vs_none": sd,
                "repeat_ci95_low": mean - half_width,
                "repeat_ci95_high": mean + half_width,
                "better_than_none_repeats": int((values < 0).sum()),
                "worse_than_none_repeats": int((values > 0).sum()),
                "tied_with_none_repeats": int((values == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def layer_effects(paired: pd.DataFrame) -> pd.DataFrame:
    selected = paired[
        paired["lr_multiplier"].isin(WORKING_MULTIPLIERS)
    ].copy()
    return (
        selected.groupby(
            [
                "mode",
                "eval_kind",
                "lr_multiplier",
                "eta",
                "build_repeat",
                "layer",
            ],
            sort=True,
            as_index=False,
        )
        .agg(
            observations=("loss_delta", "size"),
            loss_delta_mean=("loss_delta", "mean"),
            delta_vs_none_mean=("loss_delta_vs_none", "mean"),
            delta_vs_none_std=("loss_delta_vs_none", "std"),
            delta_vs_none_min=("loss_delta_vs_none", "min"),
            delta_vs_none_max=("loss_delta_vs_none", "max"),
        )
    )


def normmatch_effects(build_effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict] = []
    for base in ("diag", "block4", "full"):
        norm = f"{base}_normmatch"
        base_rows = build_effects[build_effects["mode"] == base]
        norm_rows = build_effects[build_effects["mode"] == norm]
        keys = ["eval_kind", "lr_multiplier", "eta", "build_repeat"]
        merged = base_rows.merge(
            norm_rows,
            on=keys,
            suffixes=("_base", "_normmatch"),
            validate="one_to_one",
        )
        for row in merged.itertuples(index=False):
            detail_rows.append(
                {
                    "base_mode": base,
                    "eval_kind": row.eval_kind,
                    "lr_multiplier": row.lr_multiplier,
                    "eta": row.eta,
                    "build_repeat": row.build_repeat,
                    "base_delta_vs_none": row.delta_vs_none_mean_base,
                    "normmatch_delta_vs_none": row.delta_vs_none_mean_normmatch,
                    "normmatch_minus_base": (
                        row.delta_vs_none_mean_normmatch
                        - row.delta_vs_none_mean_base
                    ),
                }
            )
    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(
            ["base_mode", "eval_kind", "lr_multiplier", "eta"],
            sort=True,
            as_index=False,
        )
        .agg(
            independent_build_repeats=("build_repeat", "nunique"),
            base_delta_vs_none_mean=("base_delta_vs_none", "mean"),
            normmatch_delta_vs_none_mean=("normmatch_delta_vs_none", "mean"),
            normmatch_minus_base_mean=("normmatch_minus_base", "mean"),
            normmatch_minus_base_std=("normmatch_minus_base", "std"),
            normmatch_improved_repeats=(
                "normmatch_minus_base",
                lambda x: int((x < 0).sum()),
            ),
        )
    )
    return detail, summary


def geometry_summaries(
    directions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = {
        "observations": ("layer", "size"),
        "alignment_normalized_mean": ("alignment_normalized", "mean"),
        "alignment_normalized_std": ("alignment_normalized", "std"),
        "curvature_exact_mean": ("curvature_exact", "mean"),
        "direction_cos_vs_none_mean": ("direction_cos_vs_none", "mean"),
        "direction_cos_vs_none_std": ("direction_cos_vs_none", "std"),
        "direction_norm_ratio_vs_none_mean": (
            "direction_norm_ratio_vs_none",
            "mean",
        ),
        "projector_drift_vs_none_mean": ("projector_drift_vs_none", "mean"),
        "row_orthogonality_residual_mean": (
            "row_orthogonality_residual",
            "mean",
        ),
        "preconditioned_norm_ratio_mean": ("preconditioned_norm_ratio", "mean"),
        "ns_svd_cos_mean": ("ns_svd_cos", "mean"),
        "k_diag_condition_mean": ("k_diag_condition", "mean"),
        "k_offdiag_fro_fraction_mean": ("k_offdiag_fro_fraction", "mean"),
    }
    aggregate = (
        directions.groupby("mode", sort=False)
        .agg(**metrics)
        .reset_index()
    )

    none = directions[directions["mode"] == "none"][
        [
            "build_repeat",
            "layer",
            "alignment_normalized",
            "curvature_exact",
            "line_search_same_best_delta",
            "line_search_heldout_best_delta",
        ]
    ].rename(
        columns={
            "alignment_normalized": "none_alignment_normalized",
            "curvature_exact": "none_curvature_exact",
            "line_search_same_best_delta": "none_line_search_same_best_delta",
            "line_search_heldout_best_delta": "none_line_search_heldout_best_delta",
        }
    )
    pairwise = directions.merge(
        none, on=["build_repeat", "layer"], validate="many_to_one"
    )
    pairwise["alignment_delta_vs_none"] = (
        pairwise["alignment_normalized"]
        - pairwise["none_alignment_normalized"]
    )
    pairwise["curvature_delta_vs_none"] = (
        pairwise["curvature_exact"] - pairwise["none_curvature_exact"]
    )
    pairwise["same_best_delta_vs_none"] = (
        pairwise["line_search_same_best_delta"]
        - pairwise["none_line_search_same_best_delta"]
    )
    pairwise["heldout_best_delta_vs_none"] = (
        pairwise["line_search_heldout_best_delta"]
        - pairwise["none_line_search_heldout_best_delta"]
    )

    exact = directions[directions["exact_svd_computed"] == True].copy()  # noqa: E712
    exact_summary = (
        exact.groupby("mode", sort=False, as_index=False)
        .agg(
            exact_svd_observations=("layer", "size"),
            layers=("layer", "nunique"),
            ns_svd_cos_mean=("ns_svd_cos", "mean"),
            ns_svd_cos_min=("ns_svd_cos", "min"),
            ns_svd_cos_max=("ns_svd_cos", "max"),
            direction_cos_vs_none_mean=("direction_cos_vs_none", "mean"),
            row_orthogonality_residual_mean=(
                "row_orthogonality_residual",
                "mean",
            ),
        )
    )
    return aggregate, pairwise, exact_summary


def p0_p1_comparison(
    p1_run_dir: Path,
    p0_run_dir: Path | None,
    p1_paired: pd.DataFrame,
    p1_quality: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    p1_scalar_gap = float(
        p1_paired[
            (p1_paired["mode"] == "scalar") & (p1_paired["eta"] > 0)
        ]["loss_delta_vs_none"]
        .abs()
        .max()
    )
    rows.append(
        {
            "metric": "scalar_none_line_search_max_abs_gap",
            "p0": math.nan,
            "p1": p1_scalar_gap,
            "interpretation": "P1 numerical control floor",
        }
    )
    if p0_run_dir is None:
        return pd.DataFrame(rows)

    p0_processed = p0_run_dir / "processed"
    p0_key = json.loads(
        (p0_processed / "key_results.json").read_text(encoding="utf-8")
    )
    rows[0]["p0"] = float(
        p0_key["data_quality"]["scalar_none_line_search_max_abs_gap"]
    )
    for split in ("same", "heldout"):
        for eta in (0.01, 0.02):
            p0_value = float(
                p0_key["fixed_eta_diag_vs_none"][
                    f"{split}_eta_{eta:g}_mean_delta"
                ]
            )
            p1_value = float(
                p1_paired[
                    (p1_paired["mode"] == "diag")
                    & (p1_paired["eval_kind"] == split)
                    & np.isclose(p1_paired["eta"], eta)
                ]["loss_delta_vs_none"].mean()
            )
            rows.append(
                {
                    "metric": f"diag_minus_none_{split}_eta_{eta:g}",
                    "p0": p0_value,
                    "p1": p1_value,
                    "interpretation": (
                        "Negative favors diag; P1 uses four build repeats and "
                        "eight heldout batches per build"
                    ),
                }
            )
    p1_emitted_scalar = p1_quality[
        p1_quality["check"] == "scalar_none_line_search_control"
    ]
    if len(p1_emitted_scalar) == 1:
        rows.append(
            {
                "metric": "p1_emitted_scalar_control_pass",
                "p0": math.nan,
                "p1": float(
                    str(p1_emitted_scalar.iloc[0]["status"]).lower() == "pass"
                ),
                "interpretation": "1 means emitted QA passed",
            }
        )
    return pd.DataFrame(rows)


def historical_val_at_step(
    path: Path | None, *, seed: int, mode: str, step: int
) -> float | None:
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    match = frame[
        (frame["seed"] == seed)
        & (frame["mode"] == mode)
        & (frame["step"] == step)
    ]
    if len(match) != 1:
        raise ValueError(
            f"Expected one historical row for seed={seed}, mode={mode}, "
            f"step={step}; found {len(match)}"
        )
    return float(match.iloc[0]["val_loss"])


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
    directions = pd.read_csv(raw_probe / "quadratic_probe_long.csv")
    paired = pd.read_csv(raw_probe / "line_search_paired_results.csv")
    line_search = pd.read_csv(raw_probe / "line_search_results.csv")
    inventory = source_inventory(raw_probe, raw_wandb)
    wandb_series, wandb_summary, band_checks = load_wandb_exports(raw_wandb)

    build_effects = build_repeat_effects(paired)
    working_summary = working_point_summary(build_effects)
    layer_summary = layer_effects(paired)
    normmatch_detail, normmatch_summary = normmatch_effects(build_effects)
    geometry, geometry_pairwise, exact_svd = geometry_summaries(directions)
    p0_compare = p0_p1_comparison(
        run_dir,
        args.p0_run_dir.resolve() if args.p0_run_dir else None,
        paired,
        emitted_quality,
    )

    expected_direction_keys = {
        (repeat, layer, mode)
        for repeat in BUILD_REPEATS
        for layer in LAYERS
        for mode in ALL_MODES
    }
    observed_direction_keys = set(
        zip(
            directions["build_repeat"].astype(int),
            directions["layer"].astype(int),
            directions["mode"].astype(str),
        )
    )
    expected_line_keys = {
        (repeat, layer, mode, kind, heldout, multiplier)
        for repeat in BUILD_REPEATS
        for layer in LAYERS
        for mode in ALL_MODES
        for kind, heldouts in (("same", (-1,)), ("heldout", tuple(range(8))))
        for heldout in heldouts
        for multiplier in (0.0, 0.25, 0.5, 1.0, 2.0)
    }
    observed_line_keys = set(
        zip(
            paired["build_repeat"].astype(int),
            paired["layer"].astype(int),
            paired["mode"].astype(str),
            paired["eval_kind"].astype(str),
            paired["heldout_index"].astype(int),
            paired["lr_multiplier"].astype(float),
        )
    )
    scalar_gap = float(
        paired[(paired["mode"] == "scalar") & (paired["eta"] > 0)][
            "loss_delta_vs_none"
        ]
        .abs()
        .max()
    )
    none_repeat_gap = float(
        paired[(paired["mode"] == "none_repeat") & (paired["eta"] > 0)][
            "loss_delta_vs_none"
        ]
        .abs()
        .max()
    )
    norm_error = float(
        directions[directions["direction_variant"] == "normmatch"][
            "direction_norm_ratio_vs_none"
        ]
        .sub(1.0)
        .abs()
        .max()
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
                    len(emitted_quality) == 12
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
                    and len(directions) == 108
                ),
                "severity_if_failed": "critical",
                "evidence": (
                    f"keys={len(observed_direction_keys)}/108, rows={len(directions)}"
                ),
            },
            {
                "check": "line_search_key_coverage",
                "status": status(
                    observed_line_keys == expected_line_keys
                    and len(paired) == 4860
                    and len(line_search) == 4860
                ),
                "severity_if_failed": "critical",
                "evidence": (
                    f"keys={len(observed_line_keys)}/4860, "
                    f"paired_rows={len(paired)}, raw_rows={len(line_search)}"
                ),
            },
            {
                "check": "finite_direction_metrics",
                "status": status(
                    np.isfinite(
                        directions[
                            [
                                "alignment_raw",
                                "alignment_normalized",
                                "curvature_exact",
                                "direction_fro_norm",
                                "direction_cos_vs_none",
                            ]
                        ].to_numpy(dtype=float)
                    ).all()
                ),
                "severity_if_failed": "critical",
                "evidence": "alignment, curvature, norm and cosine are finite",
            },
            {
                "check": "scalar_none_fp32_control",
                "status": status(scalar_gap <= 1.0e-6),
                "severity_if_failed": "critical",
                "evidence": f"max_abs_line_gap={scalar_gap:.12g}",
            },
            {
                "check": "none_repeat_control",
                "status": status(none_repeat_gap == 0.0),
                "severity_if_failed": "critical",
                "evidence": f"max_abs_line_gap={none_repeat_gap:.12g}",
            },
            {
                "check": "normmatch_direction_norm",
                "status": status(norm_error <= 1.0e-6),
                "severity_if_failed": "high",
                "evidence": f"max_abs_norm_error={norm_error:.12g}",
            },
            {
                "check": "exact_svd_coverage",
                "status": status(
                    int(directions["exact_svd_computed"].sum()) == 27
                    and np.isfinite(
                        directions.loc[
                            directions["exact_svd_computed"] == True,  # noqa: E712
                            "ns_svd_cos",
                        ].to_numpy(dtype=float)
                    ).all()
                ),
                "severity_if_failed": "high",
                "evidence": (
                    f"finite_exact_svd={int(directions['exact_svd_computed'].sum())}/27"
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

    def working(mode: str, kind: str, multiplier: float) -> pd.Series:
        match = working_summary[
            (working_summary["mode"] == mode)
            & (working_summary["eval_kind"] == kind)
            & np.isclose(working_summary["lr_multiplier"], multiplier)
        ]
        if len(match) != 1:
            raise ValueError((mode, kind, multiplier, len(match)))
        return match.iloc[0]

    val_series = wandb_series[wandb_series["metric"] == "val/loss"]
    train_series = wandb_series[wandb_series["metric"] == "train/loss_step"]
    val_at_step = float(
        val_series[val_series["step"] == config["steps"][0]]["value"].iloc[0]
    )
    historical_val = historical_val_at_step(
        args.historical_summary,
        seed=int(config["seed"]),
        mode=str(config["trajectory_cproj_k_mode"]),
        step=int(config["steps"][0]),
    )
    mode_geometry = {
        row.mode: row for row in geometry.itertuples(index=False)
    }
    exact_lookup = {
        row.mode: row for row in exact_svd.itertuples(index=False)
    }

    diag_same_eta1 = working("diag", "same", 1.0)
    diag_held_eta1 = working("diag", "heldout", 1.0)
    diag_held_eta2 = working("diag", "heldout", 2.0)
    diag_layer_eta1 = layer_summary[
        (layer_summary["mode"] == "diag")
        & (layer_summary["eval_kind"] == "heldout")
        & np.isclose(layer_summary["lr_multiplier"], 1.0)
    ]
    same_diag_rows = paired[
        (paired["mode"] == "diag")
        & (paired["eval_kind"] == "same")
        & np.isclose(paired["lr_multiplier"], 1.0)
    ]

    key_results = {
        "run": {
            "name": config["wandb_run_name"],
            "seed": int(config["seed"]),
            "step": int(config["steps"][0]),
            "trajectory_cproj_k_mode": config["trajectory_cproj_k_mode"],
            "build_repeats": int(config["build_repeats"]),
            "layers": list(config["layers"]),
            "heldout_batches_per_build": int(config["heldout_batches"]),
            "probe_precision": config["probe_precision"],
            "tf32_disabled_for_probe": bool(
                config["tf32_disabled_for_float32_probe"]
            ),
        },
        "data_quality": {
            "archived_source_files": int(len(inventory)),
            "emitted_checks_passed": int(
                emitted_quality["status"].str.lower().eq("pass").sum()
            ),
            "emitted_checks_total": int(len(emitted_quality)),
            "analysis_checks_passed": int(
                analysis_quality["status"].eq("PASS").sum()
            ),
            "analysis_checks_total": int(len(analysis_quality)),
            "direction_rows": int(len(directions)),
            "line_search_rows": int(len(paired)),
            "scalar_none_line_search_max_abs_gap": scalar_gap,
            "none_repeat_line_search_max_abs_gap": none_repeat_gap,
            "normmatch_max_abs_direction_norm_error": norm_error,
        },
        "hierarchical_fixed_eta": {
            "independent_unit": "build_repeat aggregate",
            "diag_same_eta_0.01": {
                "mean_delta_vs_none": float(
                    diag_same_eta1["repeat_mean_delta_vs_none"]
                ),
                "sd_across_4_builds": float(
                    diag_same_eta1["repeat_sd_delta_vs_none"]
                ),
                "ci95": [
                    float(diag_same_eta1["repeat_ci95_low"]),
                    float(diag_same_eta1["repeat_ci95_high"]),
                ],
                "worse_repeats": int(
                    diag_same_eta1["worse_than_none_repeats"]
                ),
                "worse_layer_build_cells": int(
                    (same_diag_rows["loss_delta_vs_none"] > 0).sum()
                ),
                "total_layer_build_cells": int(len(same_diag_rows)),
            },
            "diag_heldout_eta_0.01": {
                "mean_delta_vs_none": float(
                    diag_held_eta1["repeat_mean_delta_vs_none"]
                ),
                "sd_across_4_builds": float(
                    diag_held_eta1["repeat_sd_delta_vs_none"]
                ),
                "ci95": [
                    float(diag_held_eta1["repeat_ci95_low"]),
                    float(diag_held_eta1["repeat_ci95_high"]),
                ],
                "better_repeats": int(
                    diag_held_eta1["better_than_none_repeats"]
                ),
                "worse_repeats": int(
                    diag_held_eta1["worse_than_none_repeats"]
                ),
                "better_layer_build_cells": int(
                    (diag_layer_eta1["delta_vs_none_mean"] < 0).sum()
                ),
                "worse_layer_build_cells": int(
                    (diag_layer_eta1["delta_vs_none_mean"] > 0).sum()
                ),
            },
            "diag_heldout_eta_0.02": {
                "mean_delta_vs_none": float(
                    diag_held_eta2["repeat_mean_delta_vs_none"]
                ),
                "sd_across_4_builds": float(
                    diag_held_eta2["repeat_sd_delta_vs_none"]
                ),
                "ci95": [
                    float(diag_held_eta2["repeat_ci95_low"]),
                    float(diag_held_eta2["repeat_ci95_high"]),
                ],
                "better_repeats": int(
                    diag_held_eta2["better_than_none_repeats"]
                ),
                "worse_repeats": int(
                    diag_held_eta2["worse_than_none_repeats"]
                ),
            },
        },
        "geometry": {
            mode: {
                "alignment_normalized_mean": float(
                    mode_geometry[mode].alignment_normalized_mean
                ),
                "direction_cos_vs_none_mean": float(
                    mode_geometry[mode].direction_cos_vs_none_mean
                ),
                "projector_drift_vs_none_mean": float(
                    mode_geometry[mode].projector_drift_vs_none_mean
                ),
                "direction_norm_ratio_vs_none_mean": float(
                    mode_geometry[mode].direction_norm_ratio_vs_none_mean
                ),
                "row_orthogonality_residual_mean": float(
                    mode_geometry[mode].row_orthogonality_residual_mean
                ),
                "ns_svd_cos_mean_first_build": float(
                    exact_lookup[mode].ns_svd_cos_mean
                ),
            }
            for mode in BASE_MODES
        },
        "normmatch": {
            "interpretation": (
                "Final NS5 direction norms already differ from none by about one "
                "percent or less. Matching those norms changes heldout gaps only "
                "at approximately 1e-7 to 4e-7 scale."
            ),
            "heldout_eta_0.01_normmatch_minus_base": {
                row.base_mode: float(row.normmatch_minus_base_mean)
                for row in normmatch_summary[
                    (normmatch_summary["eval_kind"] == "heldout")
                    & np.isclose(normmatch_summary["lr_multiplier"], 1.0)
                ].itertuples(index=False)
            },
        },
        "trajectory_and_resource": {
            "val_loss_step10000": val_at_step,
            "historical_none_seed2026_val_loss_step10000": historical_val,
            "delta_vs_historical": (
                val_at_step - historical_val
                if historical_val is not None
                else None
            ),
            "best_val_loss_through_step10000": float(val_series["value"].min()),
            "best_val_step_through_step10000": int(
                val_series.loc[val_series["value"].idxmin(), "step"]
            ),
            "train_loss_step10000": float(
                train_series[train_series["step"] == 10000]["value"].iloc[0]
            ),
            "training_elapsed_seconds_step10000": float(
                wandb_summary[wandb_summary["metric"] == "time_elapsed"][
                    "last_value"
                ].iloc[0]
            ),
            "training_peak_memory_mib": float(
                wandb_summary[
                    wandb_summary["metric"]
                    == "cuda/full_run_max_memory_allocated_mib"
                ]["last_value"].iloc[0]
            ),
            "probe_seconds": float(metadata["probe_seconds"]),
            "probe_peak_memory_mib": float(metadata["probe_peak_memory_mib"]),
            "total_k_state_mib": float(
                wandb_summary[wandb_summary["metric"] == "matrix/k_state_bytes"][
                    "last_value"
                ].iloc[0]
                / (1024 * 1024)
            ),
            "cproj_k_state_mib": float(
                wandb_summary[
                    wandb_summary["metric"] == "matrix/cproj_k_state_bytes"
                ]["last_value"].iloc[0]
                / (1024 * 1024)
            ),
            "k_state_released_fraction": float(
                wandb_summary[
                    wandb_summary["metric"]
                    == "matrix/k_state_released_fraction"
                ]["last_value"].iloc[0]
            ),
        },
        "conclusion": {
            "p1_seed_gate_pass": False,
            "supported": [
                (
                    "The strict-FP32 repeatability controls pass: the scalar/none "
                    "line-search gap is below 1e-6 and none_repeat is exact."
                ),
                (
                    "At eta=0.01 every one of the 12 build-by-layer cells has a "
                    "worse same-batch loss change for diag than none."
                ),
                (
                    "All nontrivial K modes reduce normalized gradient alignment "
                    "relative to none in all 12 build-by-layer cells; diag has the "
                    "largest average alignment reduction."
                ),
                (
                    "The heldout diag-minus-none effect is tiny, changes sign "
                    "across build repeats/layers, and its four-repeat interval "
                    "straddles zero."
                ),
                (
                    "Final-direction norm matching does not rescue any K mode, so "
                    "the fixed-state gap is directional rather than a step-norm effect."
                ),
                (
                    "NS5 has low absolute cosine with the exact polar direction "
                    "for every mode, but the cosine is nearly mode-invariant; P1 "
                    "does not evaluate the exact-polar direction's loss."
                ),
            ],
            "not_supported": [
                "The P0 claim that diag has a seed2026 heldout single-step advantage.",
                "A single-step fresh-K explanation for diag's long-run training gain.",
                "A claim that norm inflation causes full/block4 damage.",
                "A claim that P1 rules exact-polar approximation error in or out.",
            ],
            "seed_decision": (
                "Do not spend two full runs on unchanged P1 seeds 2024/2025. "
                "The predeclared seed2026 gate fails. Next make the probe "
                "optimizer-faithful (EMA K plus momentum) and line-search both "
                "NS5 and exact-SVD directions; only expand seeds if that revised "
                "probe produces a repeat-consistent signal."
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
    build_effects.to_csv(processed / "hierarchical_build_repeat.csv", index=False)
    working_summary.to_csv(processed / "working_point_summary.csv", index=False)
    layer_summary.to_csv(processed / "hierarchical_layer.csv", index=False)
    normmatch_detail.to_csv(processed / "normmatch_effects_by_repeat.csv", index=False)
    normmatch_summary.to_csv(processed / "normmatch_effects_summary.csv", index=False)
    geometry.to_csv(processed / "geometry_mode_summary.csv", index=False)
    geometry_pairwise.to_csv(
        processed / "geometry_pairwise_vs_none.csv", index=False
    )
    exact_svd.to_csv(processed / "exact_svd_summary.csv", index=False)
    p0_compare.to_csv(processed / "p0_p1_comparison.csv", index=False)
    write_json(processed / "key_results.json", key_results)

    print(f"Wrote P1 processed analysis to {processed}")
    print(analysis_quality[["check", "status", "evidence"]].to_string(index=False))
    print("\nSeed gate: FAIL — do not add unchanged P1 seeds.")


if __name__ == "__main__":
    main()
