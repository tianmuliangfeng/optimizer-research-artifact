#!/usr/bin/env python
"""Reproducible analysis for the fixed-checkpoint c_proj quadratic probe.

The script keeps raw probe/W&B exports immutable and writes derived tables to
``processed/`` under one archived run directory.
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


EXPECTED_LAYERS = (0, 11, 23)
EXPECTED_MODES = ("none", "scalar", "diag", "block4", "full")
EXPECTED_SPLITS = ("same", "heldout")
EXPECTED_MULTIPLIERS = (0.0, 0.25, 0.5, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Archived run directory containing raw_probe/ and raw_wandb_exports/.",
    )
    parser.add_argument(
        "--historical-summary",
        type=Path,
        default=None,
        help="Optional historical combined_val_curves_all_seeds.csv.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def finite_count(frame: pd.DataFrame, columns: list[str]) -> tuple[int, int]:
    values = frame[columns].to_numpy(dtype=float)
    return int(np.isfinite(values).sum()), int(values.size)


def load_wandb_exports(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    series = []
    summaries = []
    band_checks = []
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
        series.append(compact)

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
                equal = np.allclose(
                    frame[value_col].to_numpy(dtype=float),
                    frame[band_col].to_numpy(dtype=float),
                    equal_nan=True,
                    rtol=0.0,
                    atol=0.0,
                )
                band_checks.append(
                    {
                        "file": path.name,
                        "metric": metric,
                        "band": suffix[2:].lower(),
                        "duplicates_raw_series": bool(equal),
                    }
                )
    return (
        pd.concat(series, ignore_index=True),
        pd.DataFrame(summaries).sort_values("metric"),
        band_checks,
    )


def build_direction_comparison(
    directions: pd.DataFrame, etas: list[float]
) -> pd.DataFrame:
    none = (
        directions[directions["mode"] == "none"]
        .set_index("layer")
        .loc[:, [
            "alignment_raw",
            "curvature_exact",
            "quadratic_score_exact",
            "direction_fro_norm",
        ]]
    )
    result = directions.copy()
    ratio_columns = {
        "alignment_raw": "alignment_ratio_vs_none",
        "curvature_exact": "curvature_ratio_vs_none",
        "quadratic_score_exact": "quadratic_score_ratio_vs_none",
        "direction_fro_norm": "direction_norm_ratio_vs_none",
    }
    for source, target in ratio_columns.items():
        result[target] = result.apply(
            lambda row: float(row[source]) / float(none.loc[row["layer"], source]),
            axis=1,
        )
    result["quadratic_criterion_margin_vs_none"] = (
        result["quadratic_score_ratio_vs_none"] - 1.0
    )
    for eta in etas:
        name = f"predicted_loss_delta_eta_{eta:g}"
        result[name] = (
            -eta * result["alignment_raw"]
            + 0.5 * eta * eta * result["curvature_exact"]
        )
    return result


def build_line_search_comparison(
    line_search: pd.DataFrame, directions: pd.DataFrame
) -> pd.DataFrame:
    result = line_search.copy()
    none = (
        result[result["mode"] == "none"]
        .set_index(["layer", "eval_split", "eta"])["loss_delta"]
    )
    scalar = (
        result[result["mode"] == "scalar"]
        .set_index(["layer", "eval_split", "eta"])["loss_delta"]
    )
    result["loss_delta_vs_none"] = result.apply(
        lambda row: float(row["loss_delta"])
        - float(none.loc[(row["layer"], row["eval_split"], row["eta"])]),
        axis=1,
    )
    result["loss_delta_vs_scalar"] = result.apply(
        lambda row: float(row["loss_delta"])
        - float(scalar.loc[(row["layer"], row["eval_split"], row["eta"])]),
        axis=1,
    )
    direction_index = directions.set_index(["layer", "mode"])
    result["quadratic_predicted_delta_same_batch"] = result.apply(
        lambda row: (
            -float(row["eta"])
            * float(direction_index.loc[(row["layer"], row["mode"]), "alignment_raw"])
            + 0.5
            * float(row["eta"]) ** 2
            * float(
                direction_index.loc[
                    (row["layer"], row["mode"]), "curvature_exact"
                ]
            )
            if row["eval_split"] == "same"
            else math.nan
        ),
        axis=1,
    )
    result["quadratic_residual_same_batch"] = (
        result["loss_delta"]
        - result["quadratic_predicted_delta_same_batch"]
    )
    return result


def build_line_search_summary(line_search: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, eta, mode), group in line_search.groupby(
        ["eval_split", "eta", "mode"], sort=True
    ):
        rows.append(
            {
                "eval_split": split,
                "eta": float(eta),
                "mode": mode,
                "layers": int(len(group)),
                "mean_loss_delta": float(group["loss_delta"].mean()),
                "min_loss_delta": float(group["loss_delta"].min()),
                "max_loss_delta": float(group["loss_delta"].max()),
                "mean_delta_vs_none": float(group["loss_delta_vs_none"].mean()),
                "better_than_none_layers": int(
                    (group["loss_delta_vs_none"] < -1e-12).sum()
                ),
                "tied_with_none_layers": int(
                    np.isclose(
                        group["loss_delta_vs_none"].to_numpy(dtype=float),
                        0.0,
                        rtol=0.0,
                        atol=1e-12,
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def historical_val_at_step(
    historical_path: Path | None, *, seed: int, mode: str, step: int
) -> float | None:
    if historical_path is None or not historical_path.exists():
        return None
    frame = pd.read_csv(historical_path)
    match = frame[
        (frame["seed"] == seed)
        & (frame["mode"] == mode)
        & (frame["step"] == step)
    ]
    if len(match) != 1:
        raise ValueError(
            f"Expected one historical row for seed={seed}, mode={mode}, step={step}; "
            f"found {len(match)}"
        )
    return float(match.iloc[0]["val_loss"])


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
    )
    directions = pd.read_csv(raw_probe / "quadratic_probe_long.csv")
    line_search = pd.read_csv(raw_probe / "line_search_results.csv")
    emitted_quality = pd.read_csv(raw_probe / "probe_data_quality_checks.csv")
    wandb_series, wandb_summary, band_checks = load_wandb_exports(raw_wandb)

    etas = sorted(float(value) for value in line_search["eta"].unique())
    direction_comparison = build_direction_comparison(directions, etas)
    line_comparison = build_line_search_comparison(
        line_search, direction_comparison
    )
    line_summary = build_line_search_summary(line_comparison)

    source_rows = []
    for source_type, directory in (
        ("probe", raw_probe),
        ("wandb", raw_wandb),
    ):
        for path in sorted(directory.glob("*")):
            if not path.is_file():
                continue
            row_count = None
            column_count = None
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
                row_count = len(frame)
                column_count = len(frame.columns)
            source_rows.append(
                {
                    "source_type": source_type,
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "rows": row_count,
                    "columns": column_count,
                }
            )
    source_inventory = pd.DataFrame(source_rows)

    direction_keys = set(
        zip(directions["layer"].astype(int), directions["mode"].astype(str))
    )
    expected_direction_keys = {
        (layer, mode) for layer in EXPECTED_LAYERS for mode in EXPECTED_MODES
    }
    line_keys = set(
        zip(
            line_search["layer"].astype(int),
            line_search["mode"].astype(str),
            line_search["eval_split"].astype(str),
            line_search["lr_multiplier"].astype(float),
        )
    )
    expected_line_keys = {
        (layer, mode, split, multiplier)
        for layer in EXPECTED_LAYERS
        for mode in EXPECTED_MODES
        for split in EXPECTED_SPLITS
        for multiplier in EXPECTED_MULTIPLIERS
    }
    finite_alignment, total_alignment = finite_count(
        directions, ["alignment_raw", "alignment_normalized"]
    )
    finite_curvature, total_curvature = finite_count(
        directions, ["curvature_exact", "quadratic_score_exact", "eta_star_exact"]
    )
    scalar_cos_min = float(
        directions[directions["mode"] == "scalar"]["direction_cos_vs_none"].min()
    )
    scalar_none_line_gap = (
        line_comparison[
            (line_comparison["mode"] == "scalar")
            & (line_comparison["eta"] > 0)
        ]["loss_delta_vs_none"]
        .abs()
        .max()
    )
    same_base = float(
        line_search[line_search["eval_split"] == "same"]["base_loss"].iloc[0]
    )
    heldout_base = float(
        line_search[line_search["eval_split"] == "heldout"]["base_loss"].iloc[0]
    )
    sample_ratio = float(
        directions.iloc[0]["activation_samples"]
        / directions.iloc[0]["activation_features"]
    )

    quality_rows = [
        {
            "check": "archived_source_file_count",
            "status": bool_status(len(source_inventory) == 14),
            "severity_if_failed": "high",
            "evidence": f"observed={len(source_inventory)}, expected=14",
        },
        {
            "check": "direction_key_coverage",
            "status": bool_status(direction_keys == expected_direction_keys),
            "severity_if_failed": "critical",
            "evidence": f"observed={len(direction_keys)}, expected=15",
        },
        {
            "check": "line_search_key_coverage",
            "status": bool_status(line_keys == expected_line_keys),
            "severity_if_failed": "critical",
            "evidence": f"observed={len(line_keys)}, expected=150",
        },
        {
            "check": "finite_alignment",
            "status": bool_status(finite_alignment == total_alignment),
            "severity_if_failed": "critical",
            "evidence": f"finite={finite_alignment}/{total_alignment}",
        },
        {
            "check": "finite_curvature_score_eta",
            "status": bool_status(finite_curvature == total_curvature),
            "severity_if_failed": "critical",
            "evidence": f"finite={finite_curvature}/{total_curvature}",
        },
        {
            "check": "scalar_none_direction_equivalence",
            "status": bool_status(scalar_cos_min >= 0.9999),
            "severity_if_failed": "high",
            "evidence": f"minimum_cosine={scalar_cos_min:.9f}",
        },
        {
            "check": "wandb_metric_coverage",
            "status": bool_status(len(wandb_summary) == 8),
            "severity_if_failed": "high",
            "evidence": f"observed={len(wandb_summary)}, expected=8",
        },
        {
            "check": "wandb_unique_steps",
            "status": bool_status(int(wandb_summary["duplicate_steps"].sum()) == 0),
            "severity_if_failed": "high",
            "evidence": f"duplicate_steps={int(wandb_summary['duplicate_steps'].sum())}",
        },
        {
            "check": "wandb_band_columns_duplicate_single_run",
            "status": bool_status(
                bool(band_checks)
                and all(item["duplicates_raw_series"] for item in band_checks)
            ),
            "severity_if_failed": "low",
            "evidence": f"matching_bands={sum(item['duplicates_raw_series'] for item in band_checks)}/{len(band_checks)}",
        },
        {
            "check": "scalar_none_line_search_control_gap",
            "status": "WARN" if scalar_none_line_gap > 5e-4 else "PASS",
            "severity_if_failed": "medium",
            "evidence": (
                f"max_abs_nonzero_eta_gap={float(scalar_none_line_gap):.9f}; "
                "directions are theoretically equivalent"
            ),
        },
        {
            "check": "single_batch_loss_heterogeneity",
            "status": "WARN",
            "severity_if_failed": "medium",
            "evidence": (
                f"same_base={same_base:.9f}, heldout_base={heldout_base:.9f}, "
                f"absolute_gap={abs(same_base-heldout_base):.9f}"
            ),
        },
        {
            "check": "fresh_covariance_sample_ratio",
            "status": "WARN",
            "severity_if_failed": "high",
            "evidence": (
                f"samples={int(directions.iloc[0]['activation_samples'])}, "
                f"features={int(directions.iloc[0]['activation_features'])}, "
                f"N_over_d={sample_ratio:.5f}"
            ),
        },
        {
            "check": "exact_svd_control",
            "status": "NOT_RUN",
            "severity_if_failed": "medium",
            "evidence": "ns_svd_cos is null for all 15 rows because exact_svd=false",
        },
    ]
    quality = pd.DataFrame(quality_rows)

    val_series = wandb_series[wandb_series["metric"] == "val/loss"].copy()
    train_series = wandb_series[wandb_series["metric"] == "train/loss_step"].copy()
    val_at_10000 = float(val_series[val_series["step"] == 10000]["value"].iloc[0])
    historical_val = historical_val_at_step(
        args.historical_summary,
        seed=int(config["seed"]),
        mode=str(config["trajectory_cproj_k_mode"]),
        step=10000,
    )
    historical_delta = (
        val_at_10000 - historical_val if historical_val is not None else None
    )

    selected_line = line_summary[line_summary["eta"].isin([0.01, 0.02])].copy()
    selected_lookup = {
        (row.eval_split, float(row.eta), row.mode): row
        for row in selected_line.itertuples(index=False)
    }
    direction_aggregate = (
        direction_comparison.groupby("mode", sort=False)
        .agg(
            layers=("layer", "count"),
            alignment_normalized_mean=("alignment_normalized", "mean"),
            curvature_exact_mean=("curvature_exact", "mean"),
            quadratic_score_exact_mean=("quadratic_score_exact", "mean"),
            quadratic_score_ratio_vs_none_mean=(
                "quadratic_score_ratio_vs_none",
                "mean",
            ),
            direction_cos_vs_none_mean=("direction_cos_vs_none", "mean"),
            projector_drift_vs_none_mean=("projector_drift_vs_none", "mean"),
            direction_norm_ratio_vs_none_mean=(
                "direction_norm_ratio_vs_none",
                "mean",
            ),
            row_orthogonality_residual_mean=(
                "row_orthogonality_residual",
                "mean",
            ),
        )
        .reset_index()
    )

    key_results = {
        "run": {
            "name": config["wandb_run_name"],
            "seed": int(config["seed"]),
            "step": int(config["steps"][0]),
            "trajectory_cproj_k_mode": config["trajectory_cproj_k_mode"],
            "probe_layers": list(config["layers"]),
            "probe_modes": list(config["modes"]),
        },
        "data_quality": {
            "source_files": int(len(source_inventory)),
            "direction_rows": int(len(directions)),
            "line_search_rows": int(len(line_search)),
            "emitted_checks_passed": int((emitted_quality["status"] == "pass").sum()),
            "emitted_checks_total": int(len(emitted_quality)),
            "analysis_checks_passed": int((quality["status"] == "PASS").sum()),
            "analysis_warnings": int((quality["status"] == "WARN").sum()),
            "analysis_not_run": int((quality["status"] == "NOT_RUN").sum()),
            "scalar_none_line_search_max_abs_gap": float(scalar_none_line_gap),
            "same_heldout_base_loss_abs_gap": abs(same_base - heldout_base),
        },
        "trajectory_reproduction": {
            "val_loss_step10000": val_at_10000,
            "historical_none_seed2026_val_loss_step10000": historical_val,
            "delta_vs_historical": historical_delta,
            "best_val_loss_through_step10000": float(val_series["value"].min()),
            "best_val_step_through_step10000": int(
                val_series.loc[val_series["value"].idxmin(), "step"]
            ),
            "train_loss_step10000": float(
                train_series[train_series["step"] == 10000]["value"].iloc[0]
            ),
        },
        "resource": {
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
            "probe_seconds": float(metadata[0]["probe_seconds"]),
            "probe_peak_memory_mib": float(metadata[0]["probe_peak_memory_mib"]),
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
        "fixed_eta_layer_mean_loss_delta": {
            f"{split}_eta_{eta:g}_{mode}": float(
                selected_lookup[(split, eta, mode)].mean_loss_delta
            )
            for split in EXPECTED_SPLITS
            for eta in (0.01, 0.02)
            for mode in EXPECTED_MODES
        },
        "fixed_eta_diag_vs_none": {
            f"{split}_eta_{eta:g}_mean_delta": float(
                selected_lookup[(split, eta, "diag")].mean_delta_vs_none
            )
            for split in EXPECTED_SPLITS
            for eta in (0.01, 0.02)
        },
        "geometry": {
            "activation_samples": int(directions.iloc[0]["activation_samples"]),
            "activation_features": int(directions.iloc[0]["activation_features"]),
            "sample_to_feature_ratio": sample_ratio,
            "k_offdiag_fro_fraction_min": float(
                directions["k_offdiag_fro_fraction"].min()
            ),
            "k_offdiag_fro_fraction_max": float(
                directions["k_offdiag_fro_fraction"].max()
            ),
            "scalar_none_direction_cosine_min": scalar_cos_min,
            "exact_svd_run": bool(config["exact_svd"]),
        },
        "interpretation": {
            "verified": [
                "Scalar and none produce the same instantaneous NS5 direction to numerical precision.",
                "At eta=0.01, none has the best same-batch loss delta in all three probed layers.",
                "Diag has the best three-layer mean heldout loss delta at eta=0.01 and eta=0.02.",
                "Block4/full often improve unconstrained A^2/C but do not improve the fixed-eta heldout mean.",
                "Finite NS5 changes direction norms and row-orthogonality substantially across K modes.",
            ],
            "not_established": [
                "A causal explanation of long-run full/block4 damage.",
                "A seed-stable heldout advantage for diag.",
                "The contribution of exact polar geometry versus five-step Newton-Schulz.",
                "The behavior of the optimizer's EMA K state and momentum buffer.",
            ],
            "seed_decision": (
                "Do not run seed2024/2025 with the unchanged P0 probe yet. "
                "First revise the probe to average multiple heldout batches, quantify "
                "float32/repeatability noise, and add exact-SVD or norm-matched controls; "
                "then use seeds 2024/2025 only if the revised seed2026 signal survives."
            ),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    source_inventory.to_csv(processed / "source_inventory.csv", index=False)
    wandb_series.to_csv(processed / "wandb_series_long.csv", index=False)
    wandb_summary.to_csv(processed / "wandb_metric_summary.csv", index=False)
    direction_comparison.to_csv(
        processed / "direction_comparison.csv", index=False
    )
    direction_aggregate.to_csv(
        processed / "direction_mode_aggregate.csv", index=False
    )
    line_comparison.to_csv(
        processed / "line_search_comparison.csv", index=False
    )
    line_summary.to_csv(processed / "line_search_mode_summary.csv", index=False)
    quality.to_csv(processed / "data_quality_checks.csv", index=False)
    write_json(processed / "key_results.json", key_results)

    print(f"Wrote processed analysis to {processed}")
    print(quality[["check", "status", "evidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
