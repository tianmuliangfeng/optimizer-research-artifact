#!/usr/bin/env python
"""Build the canonical portable technical report artifact for probe P2."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def records(frame: pd.DataFrame) -> list[dict]:
    return frame.where(pd.notnull(frame), None).to_dict(orient="records")


def source(
    source_id: str,
    label: str,
    path: str,
    sql: str,
    description: str,
    generated_at: str,
) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": sql,
            "description": description,
            "executed_at": generated_at,
        },
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    processed = run_dir / "processed"
    key = json.loads((processed / "key_results.json").read_text(encoding="utf-8"))
    contrast_summary = pd.read_csv(processed / "mechanism_contrast_summary.csv")
    contrast_build = pd.read_csv(
        processed / "mechanism_contrasts_by_build_repeat.csv"
    )
    layer_detail = pd.read_csv(processed / "mechanism_contrasts_by_layer.csv")
    geometry = pd.read_csv(processed / "candidate_geometry_summary.csv")
    p1p2 = pd.read_csv(processed / "p1_p2_fresh_arm_comparison.csv")
    historical = pd.read_csv(processed / "historical_step10000_rows.csv")
    quality = pd.read_csv(processed / "data_quality_checks.csv")

    mode_labels = {
        "none": "none",
        "diag": "diag",
        "full": "full",
        "ema_momentum_none_ns5": "none",
        "ema_momentum_diag_ns5": "diag",
        "ema_momentum_full_ns5": "full",
    }

    temporal = contrast_summary[
        (contrast_summary["contrast"] == "matched_none")
        & (
            contrast_summary["candidate"].isin(
                ["ema_momentum_diag_ns5", "ema_momentum_full_ns5"]
            )
        )
        & (contrast_summary["eval_kind"] == "heldout")
        & (contrast_summary["eta"].isin([0.01, 0.02]))
    ].copy()
    temporal["mode"] = temporal["candidate"].map(mode_labels)
    temporal["interpretation"] = temporal.apply(
        lambda row: (
            "better than none, unresolved interval"
            if row["mean"] < 0 and row["ci95_high"] >= 0
            else "worse than none"
            if row["ci95_low"] > 0
            else "unresolved"
        ),
        axis=1,
    )
    temporal = temporal[
        [
            "mode",
            "eta",
            "mean",
            "sd",
            "ci95_low",
            "ci95_high",
            "negative_repeats",
            "positive_repeats",
            "interpretation",
        ]
    ].sort_values(["eta", "mode"])

    temporal_chart = contrast_build[
        (contrast_build["contrast"] == "matched_none")
        & (
            contrast_build["candidate"].isin(
                ["ema_momentum_diag_ns5", "ema_momentum_full_ns5"]
            )
        )
        & (contrast_build["eval_kind"] == "heldout")
        & (contrast_build["eta"] == 0.01)
    ].copy()
    temporal_chart["mode"] = temporal_chart["candidate"].map(mode_labels)
    temporal_chart["build_repeat_label"] = temporal_chart["build_repeat"].map(
        lambda value: f"build {int(value)}"
    )
    temporal_chart = temporal_chart[
        [
            "build_repeat_label",
            "build_repeat",
            "mode",
            "contrast_value",
            "layers",
            "heldout_batches",
            "observations",
        ]
    ].sort_values(["build_repeat", "mode"])

    history = contrast_summary[
        (contrast_summary["contrast"] == "momentum_minus_ema_gradient")
        & (contrast_summary["eval_kind"] == "heldout")
        & (contrast_summary["eta"] == 0.01)
    ].copy()
    history["mode"] = history["candidate"].map(mode_labels)
    history = history[
        [
            "mode",
            "mean",
            "sd",
            "ci95_low",
            "ci95_high",
            "negative_repeats",
            "positive_repeats",
        ]
    ].sort_values("mode")

    layer = layer_detail[
        (layer_detail["contrast"] == "matched_none")
        & (
            layer_detail["candidate"].isin(
                ["ema_momentum_diag_ns5", "ema_momentum_full_ns5"]
            )
        )
        & (layer_detail["eval_kind"] == "heldout")
        & (layer_detail["eta"] == 0.01)
    ].copy()
    layer["mode"] = layer["candidate"].map(mode_labels)
    layer = (
        layer.groupby(["mode", "layer"], as_index=False)
        .agg(
            mean_delta_vs_none=("contrast_value", "mean"),
            sd_across_builds=("contrast_value", "std"),
            better_builds=(
                "contrast_value",
                lambda values: int((values < 0).sum()),
            ),
            worse_builds=(
                "contrast_value",
                lambda values: int((values > 0).sum()),
            ),
        )
        .sort_values(["mode", "layer"])
    )

    momentum_geometry = geometry[
        geometry["candidate"].isin(
            [
                "ema_momentum_none_ns5",
                "ema_momentum_diag_ns5",
                "ema_momentum_full_ns5",
            ]
        )
    ].copy()
    momentum_geometry["mode"] = momentum_geometry["candidate"].map(mode_labels)
    momentum_geometry = momentum_geometry[
        [
            "mode",
            "projection_input_cos_vs_gradient_mean",
            "direction_cos_vs_matched_none_mean",
            "direction_norm_ratio_vs_matched_none_mean",
            "alignment_normalized_mean",
            "row_orthogonality_residual_mean",
            "state_updates_mean",
        ]
    ].sort_values("mode")

    fresh_reproduction = p1p2[p1p2["eta"] == 0.01].copy()
    fresh_reproduction = fresh_reproduction[
        [
            "mode",
            "eval_kind",
            "p1_mean_delta_vs_none",
            "p2_fresh_mean_delta_vs_none",
            "p2_minus_p1",
            "same_sign",
        ]
    ].sort_values(["eval_kind", "mode"])

    svd_rows: list[dict] = []
    for contrast, candidates, label in (
        (
            "svd_minus_ns5",
            [
                "fresh_gradient_none_svd",
                "fresh_gradient_diag_svd",
                "fresh_gradient_full_svd",
            ],
            "SVD minus matched NS5",
        ),
        (
            "matched_none",
            ["fresh_gradient_diag_svd", "fresh_gradient_full_svd"],
            "structured SVD minus matched-none SVD",
        ),
    ):
        selected = contrast_summary[
            (contrast_summary["contrast"] == contrast)
            & (contrast_summary["candidate"].isin(candidates))
            & (contrast_summary["eval_kind"] == "heldout")
            & (contrast_summary["eta"] == 0.01)
        ]
        for row in selected.itertuples(index=False):
            mode = row.candidate.split("_")[2]
            svd_rows.append(
                {
                    "comparison": label,
                    "mode": mode,
                    "mean": row.mean,
                    "ci95_low": row.ci95_low,
                    "ci95_high": row.ci95_high,
                    "negative_repeats": row.negative_repeats,
                    "positive_repeats": row.positive_repeats,
                }
            )
    svd = pd.DataFrame(svd_rows).sort_values(["comparison", "mode"])

    svd_orth = geometry[geometry["projection"] == "svd"][
        [
            "candidate",
            "row_orthogonality_residual_mean",
            "row_orthogonality_residual_max",
            "ns5_svd_cos_mean",
        ]
    ].copy()
    svd_orth["mode"] = svd_orth["candidate"].map(
        {
            "fresh_gradient_none_svd": "none",
            "fresh_gradient_diag_svd": "diag",
            "fresh_gradient_full_svd": "full",
        }
    )
    svd_orth = svd_orth[
        [
            "mode",
            "row_orthogonality_residual_mean",
            "row_orthogonality_residual_max",
            "ns5_svd_cos_mean",
        ]
    ].sort_values("mode")

    historical_view = (
        historical.groupby("mode", as_index=False)
        .agg(seeds=("seed", "nunique"), val_loss_mean=("val_loss", "mean"))
        .sort_values("val_loss_mean")
    )
    historical_view["delta_vs_none"] = (
        historical_view["val_loss_mean"]
        - float(
            historical_view.loc[
                historical_view["mode"] == "none", "val_loss_mean"
            ].iloc[0]
        )
    )

    resource = key["trajectory_and_resources"]
    resource_view = pd.DataFrame(
        [
            {
                "val_loss_step10000": resource["val_loss_step10000"],
                "best_val_loss": resource["best_val_loss_through_step10000"],
                "best_val_step": resource["best_val_step_through_step10000"],
                "training_elapsed_seconds_to_9980": resource[
                    "training_elapsed_last_seconds"
                ],
                "full_run_peak_memory_mib": resource[
                    "full_run_peak_memory_mib"
                ],
                "probe_seconds": resource["probe_seconds"],
                "optimizer_k_state_mib": resource["optimizer_k_state_mib"],
                "optimizer_cproj_k_state_mib": resource[
                    "optimizer_cproj_k_state_mib"
                ],
                "diagnostic_shadow_state_mib": (
                    resource["diagnostic_shadow_k_state_mib"]
                    + resource["diagnostic_shadow_momentum_mib"]
                ),
            }
        ]
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    sources = [
        source(
            "temporal_effects",
            "P2 temporal matched-none effects",
            "processed/mechanism_contrast_summary.csv",
            (
                "SELECT * FROM mechanism_contrast_summary WHERE "
                "contrast='matched_none' AND eval_kind='heldout' "
                "AND candidate IN "
                "('ema_momentum_diag_ns5','ema_momentum_full_ns5')"
            ),
            (
                "Means and t intervals use four build-repeat aggregates. Each "
                "aggregate averages three measured layers and eight shared "
                "held-out batches."
            ),
            generated_at,
        ),
        source(
            "temporal_builds",
            "P2 temporal effects by build repeat",
            "processed/mechanism_contrasts_by_build_repeat.csv",
            (
                "SELECT * FROM mechanism_contrasts_by_build_repeat WHERE "
                "contrast='matched_none' AND eval_kind='heldout' AND eta=0.01"
            ),
            (
                "Build-repeat matched-none effects for the EMA-momentum diag "
                "and full directions."
            ),
            generated_at,
        ),
        source(
            "momentum_history",
            "Momentum versus EMA-gradient effects",
            "processed/mechanism_contrast_summary.csv",
            (
                "SELECT * FROM mechanism_contrast_summary WHERE "
                "contrast='momentum_minus_ema_gradient' "
                "AND eval_kind='heldout' AND eta=0.01"
            ),
            (
                "Paired held-out loss difference from replacing the current "
                "EMA-preconditioned gradient buffer with the historical "
                "optimizer momentum buffer."
            ),
            generated_at,
        ),
        source(
            "layer_effects",
            "P2 temporal effects by layer",
            "processed/mechanism_contrasts_by_layer.csv",
            (
                "SELECT * FROM mechanism_contrasts_by_layer WHERE "
                "contrast='matched_none' AND eval_kind='heldout' AND eta=0.01"
            ),
            (
                "Matched-none temporal effects for layers 0, 11, and 23, "
                "preserving build-repeat identity."
            ),
            generated_at,
        ),
        source(
            "momentum_geometry",
            "EMA-momentum direction geometry",
            "processed/candidate_geometry_summary.csv",
            (
                "SELECT * FROM candidate_geometry_summary WHERE candidate IN "
                "('ema_momentum_none_ns5','ema_momentum_diag_ns5',"
                "'ema_momentum_full_ns5')"
            ),
            (
                "Geometry means over four build repeats and layers 0, 11, and "
                "23 at step 10000."
            ),
            generated_at,
        ),
        source(
            "fresh_reproduction",
            "P1/P2 fresh-gradient comparison",
            "processed/p1_p2_fresh_arm_comparison.csv",
            "SELECT * FROM p1_p2_fresh_arm_comparison WHERE eta=0.01",
            (
                "Comparison of the P1 fresh-gradient matched-none arm with the "
                "same P2 control arm."
            ),
            generated_at,
        ),
        source(
            "svd_effects",
            "P2 exact-SVD line-search effects",
            "processed/mechanism_contrast_summary.csv",
            (
                "SELECT * FROM mechanism_contrast_summary WHERE eta=0.01 "
                "AND eval_kind='heldout' AND "
                "contrast IN ('svd_minus_ns5','matched_none')"
            ),
            (
                "SVD versus NS5 and matched-none comparisons. These results are "
                "provisional because the SVD orthogonality gate failed."
            ),
            generated_at,
        ),
        source(
            "svd_quality",
            "P2 exact-SVD orthogonality diagnostics",
            "processed/candidate_geometry_summary.csv",
            "SELECT * FROM candidate_geometry_summary WHERE projection='svd'",
            (
                "Mean and maximum row-orthogonality residuals for the FP32 SVD "
                "directions."
            ),
            generated_at,
        ),
        source(
            "historical_training",
            "Historical three-seed step-10000 validation loss",
            "processed/historical_step10000_rows.csv",
            (
                "SELECT mode, COUNT(DISTINCT seed), AVG(val_loss) "
                "FROM historical_step10000_rows GROUP BY mode"
            ),
            (
                "Matched long-run validation results for none, diag, and full "
                "over seeds 2024, 2025, and 2026."
            ),
            generated_at,
        ),
        source(
            "quality_checks",
            "P2 data-quality checks",
            "processed/data_quality_checks.csv",
            "SELECT * FROM data_quality_checks ORDER BY check",
            (
                "Archive, key coverage, numerical controls, shadow updates, "
                "SVD orthogonality, and W&B integrity checks."
            ),
            generated_at,
        ),
        source(
            "trajectory_resource",
            "P2 trajectory and resource summary",
            "processed/key_results.json",
            "SELECT * FROM trajectory_and_resources",
            (
                "Training endpoint, probe overhead, optimizer state, diagnostic "
                "state, and peak-memory record."
            ),
            generated_at,
        ),
    ]

    tables = [
        {
            "id": "temporal_effect_table",
            "title": "EMA-momentum structured K versus matched none",
            "subtitle": (
                "Held-out paired loss change; negative favors the structured-K "
                "candidate. Intervals use four build-repeat aggregates."
            ),
            "dataset": "temporal_effects",
            "sourceId": "temporal_effects",
            "defaultSort": {"field": "eta", "direction": "asc"},
            "columns": [
                {"field": "mode", "label": "K mode", "type": "text"},
                {"field": "eta", "label": "Eta", "format": "number"},
                {
                    "field": "mean",
                    "label": "Mean delta vs none",
                    "format": "number",
                    "movement": True,
                },
                {"field": "ci95_low", "label": "95% CI low", "format": "number"},
                {"field": "ci95_high", "label": "95% CI high", "format": "number"},
                {
                    "field": "negative_repeats",
                    "label": "Builds better",
                    "type": "number",
                },
                {
                    "field": "positive_repeats",
                    "label": "Builds worse",
                    "type": "number",
                },
                {
                    "field": "interpretation",
                    "label": "Interpretation",
                    "type": "text",
                },
            ],
        },
        {
            "id": "history_table",
            "title": "Held-out effect of replacing EMA-gradient with momentum history",
            "subtitle": (
                "Eta=0.01; negative means the historical momentum buffer improves "
                "over the current EMA-preconditioned gradient."
            ),
            "dataset": "history_effects",
            "sourceId": "momentum_history",
            "defaultSort": {"field": "mean", "direction": "asc"},
            "columns": [
                {"field": "mode", "label": "K mode", "type": "text"},
                {
                    "field": "mean",
                    "label": "Momentum minus EMA-gradient",
                    "format": "number",
                    "movement": True,
                },
                {"field": "ci95_low", "label": "95% CI low", "format": "number"},
                {"field": "ci95_high", "label": "95% CI high", "format": "number"},
                {
                    "field": "negative_repeats",
                    "label": "Builds improved",
                    "type": "number",
                },
            ],
        },
        {
            "id": "layer_table",
            "title": "Temporal matched-none effect by measured layer",
            "subtitle": (
                "Held-out eta=0.01; means and sign counts are across four "
                "build repeats."
            ),
            "dataset": "layer_effects",
            "sourceId": "layer_effects",
            "defaultSort": {"field": "layer", "direction": "asc"},
            "columns": [
                {"field": "mode", "label": "K mode", "type": "text"},
                {"field": "layer", "label": "Layer", "type": "number"},
                {
                    "field": "mean_delta_vs_none",
                    "label": "Mean delta vs none",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "sd_across_builds",
                    "label": "SD across builds",
                    "format": "number",
                },
                {
                    "field": "better_builds",
                    "label": "Builds better",
                    "type": "number",
                },
                {
                    "field": "worse_builds",
                    "label": "Builds worse",
                    "type": "number",
                },
            ],
        },
        {
            "id": "geometry_table",
            "title": "EMA-momentum direction geometry",
            "subtitle": "Means over four builds and layers 0, 11, and 23.",
            "dataset": "momentum_geometry",
            "sourceId": "momentum_geometry",
            "defaultSort": {
                "field": "projection_input_cos_vs_gradient_mean",
                "direction": "desc",
            },
            "columns": [
                {"field": "mode", "label": "K mode", "type": "text"},
                {
                    "field": "projection_input_cos_vs_gradient_mean",
                    "label": "Buffer cosine vs current gradient",
                    "format": "number",
                },
                {
                    "field": "direction_cos_vs_matched_none_mean",
                    "label": "Direction cosine vs none",
                    "format": "number",
                },
                {
                    "field": "direction_norm_ratio_vs_matched_none_mean",
                    "label": "Direction norm ratio",
                    "format": "number",
                },
                {
                    "field": "row_orthogonality_residual_mean",
                    "label": "NS5 row-orth residual",
                    "format": "number",
                },
            ],
        },
        {
            "id": "fresh_table",
            "title": "P2 fresh-gradient control reproduces P1",
            "subtitle": (
                "Eta=0.01 matched-none effects; positive values are worse than none."
            ),
            "dataset": "fresh_reproduction",
            "sourceId": "fresh_reproduction",
            "defaultSort": {"field": "eval_kind", "direction": "asc"},
            "columns": [
                {"field": "mode", "label": "K mode", "type": "text"},
                {"field": "eval_kind", "label": "Evaluation", "type": "text"},
                {
                    "field": "p1_mean_delta_vs_none",
                    "label": "P1",
                    "format": "number",
                },
                {
                    "field": "p2_fresh_mean_delta_vs_none",
                    "label": "P2 fresh",
                    "format": "number",
                },
                {
                    "field": "p2_minus_p1",
                    "label": "P2 minus P1",
                    "format": "number",
                    "movement": True,
                },
                {"field": "same_sign", "label": "Same sign", "type": "boolean"},
            ],
        },
        {
            "id": "svd_effect_table",
            "title": "FP32 SVD line-search comparisons",
            "subtitle": (
                "Held-out eta=0.01. Treat as provisional because the "
                "orthogonality gate failed."
            ),
            "dataset": "svd_effects",
            "sourceId": "svd_effects",
            "defaultSort": {"field": "comparison", "direction": "asc"},
            "columns": [
                {"field": "comparison", "label": "Comparison", "type": "text"},
                {"field": "mode", "label": "K mode", "type": "text"},
                {
                    "field": "mean",
                    "label": "Mean paired delta",
                    "format": "number",
                    "movement": True,
                },
                {"field": "ci95_low", "label": "95% CI low", "format": "number"},
                {"field": "ci95_high", "label": "95% CI high", "format": "number"},
                {
                    "field": "negative_repeats",
                    "label": "Negative builds",
                    "type": "number",
                },
            ],
        },
        {
            "id": "svd_quality_table",
            "title": "FP32 SVD orthogonality misses the predeclared threshold",
            "subtitle": "Required maximum residual ≤ 0.0001.",
            "dataset": "svd_quality",
            "sourceId": "svd_quality",
            "defaultSort": {
                "field": "row_orthogonality_residual_max",
                "direction": "desc",
            },
            "columns": [
                {"field": "mode", "label": "K mode", "type": "text"},
                {
                    "field": "row_orthogonality_residual_mean",
                    "label": "Mean residual",
                    "format": "number",
                },
                {
                    "field": "row_orthogonality_residual_max",
                    "label": "Maximum residual",
                    "format": "number",
                },
                {
                    "field": "ns5_svd_cos_mean",
                    "label": "NS5-SVD cosine",
                    "format": "number",
                },
            ],
        },
        {
            "id": "historical_table",
            "title": "Historical three-seed validation ordering at step 10000",
            "subtitle": "OpenWebText 24L/D1024 matched training configuration.",
            "dataset": "historical_training",
            "sourceId": "historical_training",
            "defaultSort": {"field": "val_loss_mean", "direction": "asc"},
            "columns": [
                {"field": "mode", "label": "K mode", "type": "text"},
                {"field": "seeds", "label": "Seeds", "type": "number"},
                {
                    "field": "val_loss_mean",
                    "label": "Mean val loss",
                    "format": "number",
                },
                {
                    "field": "delta_vs_none",
                    "label": "Delta vs none",
                    "format": "number",
                    "movement": True,
                },
            ],
        },
        {
            "id": "quality_table",
            "title": "P2 integrity checks and the isolated SVD failure",
            "subtitle": (
                "The temporal NS5 arm is usable; the exact-SVD arm is provisional."
            ),
            "dataset": "quality_checks",
            "sourceId": "quality_checks",
            "defaultSort": {"field": "status", "direction": "asc"},
            "columns": [
                {"field": "check", "label": "Check", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {
                    "field": "severity_if_failed",
                    "label": "Severity",
                    "type": "text",
                },
                {"field": "evidence", "label": "Evidence", "type": "text"},
            ],
        },
        {
            "id": "resource_table",
            "title": "Trajectory, optimizer state, and diagnostic overhead",
            "subtitle": (
                "Diagnostic shadow state is probe-only and is not part of the "
                "none optimizer deployment state."
            ),
            "dataset": "trajectory_resource",
            "sourceId": "trajectory_resource",
            "defaultSort": {
                "field": "val_loss_step10000",
                "direction": "asc",
            },
            "columns": [
                {
                    "field": "val_loss_step10000",
                    "label": "Val loss at 10000",
                    "format": "number",
                },
                {
                    "field": "best_val_loss",
                    "label": "Best val loss",
                    "format": "number",
                },
                {
                    "field": "best_val_step",
                    "label": "Best val step",
                    "type": "number",
                },
                {
                    "field": "training_elapsed_seconds_to_9980",
                    "label": "Training seconds to 9980",
                    "format": "number",
                },
                {
                    "field": "full_run_peak_memory_mib",
                    "label": "Full-run peak MiB",
                    "format": "number",
                },
                {
                    "field": "probe_seconds",
                    "label": "Probe seconds",
                    "format": "number",
                },
                {
                    "field": "optimizer_k_state_mib",
                    "label": "Optimizer K-state MiB",
                    "format": "number",
                },
                {
                    "field": "optimizer_cproj_k_state_mib",
                    "label": "c_proj K-state MiB",
                    "format": "number",
                },
                {
                    "field": "diagnostic_shadow_state_mib",
                    "label": "Probe-only shadow MiB",
                    "format": "number",
                },
            ],
        },
    ]

    charts = [
        {
            "id": "temporal_build_chart",
            "title": "EMA-momentum matched-none effect by build repeat",
            "subtitle": (
                "Held-out eta=0.01; each bar averages three layers and eight "
                "shared held-out batches. Negative favors the K-mode candidate."
            ),
            "type": "bar",
            "dataset": "temporal_builds",
            "sourceId": "temporal_builds",
            "valueFormat": "number",
            "encodings": {
                "x": {
                    "field": "build_repeat_label",
                    "type": "nominal",
                    "label": "Build repeat",
                },
                "y": {
                    "field": "contrast_value",
                    "type": "quantitative",
                    "label": "Mean loss delta versus matched none",
                },
                "color": {
                    "field": "mode",
                    "type": "nominal",
                    "label": "K mode",
                },
                "tooltip": [
                    {"field": "mode", "type": "nominal", "label": "K mode"},
                    {
                        "field": "contrast_value",
                        "type": "quantitative",
                        "label": "Delta versus none",
                        "format": "number",
                    },
                    {
                        "field": "observations",
                        "type": "quantitative",
                        "label": "Layer × held-out rows",
                    },
                ],
            },
        }
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# c_proj 时间状态二次探针 P2：seed2026",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## 技术摘要\n\n"
                "**加入真实 momentum 历史后，探针首次恢复了长期训练中的 "
                "`diag ≈ none < full` 定性顺序。** 在 held-out、eta=0.01 "
                "上，diag 相对 none 为 -8.14e-6（4 次 build 中 3 次更好，"
                "95% 区间跨 0），full 相对 none 为 +7.01e-5（4/4 更差，"
                "区间完全大于 0）。这说明 K 与历史 buffer 的交互是关键缺失"
                "变量；但 diag 的优势几乎全部来自 layer 0，暂不支持直接补 "
                "seed。FP32 exact-SVD 还未通过预设正交性门槛。"
            ),
        },
        {
            "id": "primary_result",
            "type": "markdown",
            "sourceId": "temporal_effects",
            "body": (
                "## Momentum 历史恢复了正确顺序，但 diag 仍未越过统计门槛\n\n"
                "Eta=0.01 时，EMA-momentum diag-minus-none 的四-build 均值为 "
                "-8.136e-6，95% 区间 [-1.982e-5, 3.552e-6]；full-minus-none "
                "为 +7.010e-5，区间 [3.862e-5, 1.016e-4]。Eta 加倍到 0.02 "
                "后，两者约按比例加倍且 repeat 符号不变，说明排序并非孤立"
                "学习率点造成。"
            ),
        },
        {"id": "primary_table", "type": "table", "tableId": "temporal_effect_table"},
        {
            "id": "build_chart_intro",
            "type": "markdown",
            "body": (
                "## Full 的损害跨 build 稳定，diag 的小优势仍有一次反向\n\n"
                "图中每根柱先平均三层与八个 held-out batch，避免把 96 个相关"
                "行误当作独立样本。Full 在每次 build 都明显更差；diag 在 "
                "build 1 略差于 none。"
            ),
        },
        {"id": "build_chart", "type": "chart", "chartId": "temporal_build_chart"},
        {
            "id": "history_result",
            "type": "markdown",
            "sourceId": "momentum_history",
            "body": (
                "## 历史 buffer 帮助 none 与 diag，却完全没有帮助 full\n\n"
                "相对 EMA-gradient，momentum 使 none 的 held-out loss change "
                "改善 -3.150e-5，使 diag 改善 -5.395e-5，二者都是 4/4 build "
                "改善；full 的变化只有 +5.46e-8。Diag 比 none 多获得约 "
                "2.245e-5 的历史收益，这使它从 fresh/EMA-gradient 下的略差"
                "转成 momentum 下的略优。"
            ),
        },
        {"id": "history_table_block", "type": "table", "tableId": "history_table"},
        {
            "id": "layer_result",
            "type": "markdown",
            "sourceId": "layer_effects",
            "body": (
                "## Diag 的小优势局限在早层，full 的损害三层一致\n\n"
                "Diag-minus-none 在 layer 0 为 -3.347e-5，但 layer 11 和 23 "
                "分别为 +5.260e-6 与 +3.800e-6。Full 在三层都更差，且 "
                "layer 0 损害最大。当前三层平均不能外推为全 24 层结论。"
            ),
        },
        {"id": "layer_table_block", "type": "table", "tableId": "layer_table"},
        {
            "id": "geometry_result",
            "type": "markdown",
            "sourceId": "momentum_geometry",
            "body": (
                "## Full K 使历史 buffer 与当前梯度严重失配\n\n"
                "EMA-momentum projection input 与当前梯度的平均 cosine 从 "
                "none 的 0.763 降到 diag 的 0.643，再降到 full 的 0.329；"
                "最终方向与 matched-none 的 cosine 分别为 1.000、0.880、"
                "0.807。数据支持这样的机制：diag 主要做坐标重标度，仍保留"
                "历史信息；full 的方向混合会让累积 buffer 失去与当前梯度及"
                "跨 batch 梯度的兼容性。"
            ),
        },
        {"id": "geometry_table_block", "type": "table", "tableId": "geometry_table"},
        {
            "id": "math_result",
            "type": "markdown",
            "body": (
                "## 局部数学解释\n\n"
                "对 mode m，令历史输入为 "
                "`B_t^(m) = beta B_(t-1)^(m) + P_m(K_t) G_t`，最终方向为 "
                "`U_t^(m) = NS5(B_t^(m))`。Held-out 局部变化满足 "
                "`Delta L_h ≈ -eta <G_h,U> + eta^2/2 <U,H_h U>`。Fresh "
                "probe 只检查当前 G_t；P2 表明真正区分 diag 与 full 的是 "
                "P_m(K_t) 对历史 B_(t-1) 的作用。Full 虽可能降低某些局部"
                "曲率量，却同时破坏了动量带来的跨 batch 一阶收益。"
            ),
        },
        {
            "id": "fresh_result",
            "type": "markdown",
            "sourceId": "fresh_reproduction",
            "body": (
                "## Fresh-gradient 控制重复了 P1，而不是制造新现象\n\n"
                "P2 fresh diag 在 same-batch 上比 none 差 +0.001386，在 "
                "held-out 上只差 +7.72e-6；P1 对应为 +0.001244 与 +4.11e-6。"
                "因此 P2 的时间状态结论建立在可重复的 fresh 基线上。"
            ),
        },
        {"id": "fresh_table_block", "type": "table", "tableId": "fresh_table"},
        {
            "id": "svd_result",
            "type": "markdown",
            "sourceId": "svd_effects",
            "body": (
                "## 现有 SVD 结果不支持“NS5 截断导致 K 损害”，但仍属暂定\n\n"
                "Held-out 上，SVD 相对 NS5 只对 none 改善 -8.89e-6；对 diag "
                "和 full 约为 0。使用 SVD 后，diag 与 full 相对 matched-none "
                "仍都约差 +1.9e-5。不过 FP32 SVD 的正交性门槛失败，所以不能"
                "把这一结果写成最终排除性结论。"
            ),
        },
        {"id": "svd_effect_block", "type": "table", "tableId": "svd_effect_table"},
        {
            "id": "svd_quality_result",
            "type": "markdown",
            "sourceId": "svd_quality",
            "body": (
                "## SVD 失败集中在 full 的数值病态输入\n\n"
                "预设最大行正交残差阈值为 1e-4。None、diag、full 的最大残差"
                "约为 1.12e-4、1.08e-4、4.74e-4；full 明显更差。正式引用前"
                "应改用 FP64 SVD 并重新通过门槛。"
            ),
        },
        {"id": "svd_quality_block", "type": "table", "tableId": "svd_quality_table"},
        {
            "id": "historical_result",
            "type": "markdown",
            "sourceId": "historical_training",
            "body": (
                "## P2 的符号顺序与三 seed 长期训练一致\n\n"
                "Step 10000 的三-seed 平均 val loss 为 diag 5.395043、none "
                "5.401933、full 5.457312，即 diag-minus-none=-0.006890，"
                "full-minus-none=+0.055379。P2 只恢复了定性顺序，不能把单步"
                "差值直接累加成长期 loss 差。"
            ),
        },
        {"id": "historical_table_block", "type": "table", "tableId": "historical_table"},
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## 实验范围与统计定义\n\n"
                "这是 OpenWebText 24L/D1024、none 轨迹、seed2026、step10000 "
                "的条件性机制探针。四个 build repeat 是主要独立单位；三层与"
                "八个共用 held-out batch 在 repeat 内先平均。区间因此只反映"
                "方向构造的不确定性，并条件于固定 held-out 集，不能解释为"
                "跨 seed 或全数据分布区间。"
            ),
        },
        {
            "id": "quality_result",
            "type": "markdown",
            "sourceId": "quality_checks",
            "body": (
                "## Temporal NS5 数据完整，SVD 臂单独降级\n\n"
                "归档包含 16 个原始文件、144 个方向与 6480 条 line-search；"
                "fresh-none/EMA-none 控制误差为 0，shadow diag/full 均更新 "
                "313 次。10 个本地检查中只有 SVD 正交性一项失败。"
            ),
        },
        {"id": "quality_table_block", "type": "table", "tableId": "quality_table"},
        {
            "id": "resource_result",
            "type": "markdown",
            "sourceId": "trajectory_resource",
            "body": (
                "## 资源记录区分算法状态与探针状态\n\n"
                "None 算法的 optimizer K-state 为 864 MiB，其中 c_proj 为 "
                "0 MiB；P2 为诊断额外维护 672.094 MiB shadow K/momentum。"
                "训练到 step9980 用时 901.877 s，探针额外 68.283 s，full-run "
                "峰值为 6875.826 MiB。探针 shadow 开销不属于部署时状态。"
            ),
        },
        {"id": "resource_table_block", "type": "table", "tableId": "resource_table"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 证据边界\n\n"
                "- Diag 的四-build 区间仍跨 0，且优势由 layer 0 驱动。\n"
                "- 只有 seed2026；未覆盖全部 24 层或跨层同步更新。\n"
                "- 八个 held-out batch 在 build 间共用，区间条件于该评估集。\n"
                "- Exact-SVD 为 FP32 且未通过预设正交性门槛。\n"
                "- 结果支持 K×history 交互，但尚不能证明它是长期差异的唯一"
                "因果来源。"
            ),
        },
        {
            "id": "next_step",
            "type": "markdown",
            "body": (
                "## 决策与下一步\n\n"
                "**暂不运行 unchanged P2 的 seed2024/2025。** 先在 seed2026 "
                "扩大层覆盖，判断 diag 优势是否仅存在于早层，并确认 full "
                "损害是否跨层稳定；同时把 SVD 改为 FP64 并重新通过正交性"
                "门槛。只有在这两项检查后仍得到 repeat-consistent 的 "
                "`diag ≈ none < full`，再补另外两个 seed。"
            ),
        },
        {
            "id": "open_questions",
            "type": "markdown",
            "body": (
                "## 尚待回答的问题\n\n"
                "- 哪些层贡献了长期 diag 优势，哪些层只增加噪声？\n"
                "- Full 的历史失配来自 K 的 off-diagonal 旋转、状态陈旧，"
                "还是二者的交互？\n"
                "- FP64 exact-polar 是否仍保留同样的 held-out 顺序？\n"
                "- 能否用 buffer-gradient cosine 或跨步方向稳定性预测长期"
                "训练差异？"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "c_proj 时间状态二次探针 P2：seed2026",
            "description": (
                "时间平均 K、历史 momentum、fresh 控制与 exact-SVD "
                "辅助臂的分层机制分析。"
            ),
            "generatedAt": generated_at,
            "filters": [],
            "cards": [],
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "temporal_effects": records(temporal),
                "temporal_builds": records(temporal_chart),
                "history_effects": records(history),
                "layer_effects": records(layer),
                "momentum_geometry": records(momentum_geometry),
                "fresh_reproduction": records(fresh_reproduction),
                "svd_effects": records(svd),
                "svd_quality": records(svd_orth),
                "historical_training": records(historical_view),
                "quality_checks": records(quality),
                "trajectory_resource": records(resource_view),
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    artifact_path = run_dir / "quadratic_probe_p2_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(artifact_path)


if __name__ == "__main__":
    main()
