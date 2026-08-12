#!/usr/bin/env python
"""Build the canonical portable technical report artifact for probe P3."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
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
    contrast = pd.read_csv(processed / "mechanism_contrast_summary.csv")
    layer = pd.read_csv(processed / "layer_effect_summary.csv")
    depth = pd.read_csv(processed / "depth_band_summary.csv")
    geometry = pd.read_csv(processed / "candidate_geometry_summary.csv")
    quality = pd.read_csv(processed / "data_quality_checks.csv")
    validation = pd.read_csv(processed / "validation_checks.csv")
    seed_gate = pd.read_csv(processed / "seed_gate_checks.csv")
    historical = pd.read_csv(processed / "historical_step10000_rows.csv")

    label_map = {
        "ema_momentum_diag_ns5": "diag",
        "ema_momentum_full_ns5": "full",
        "ema_momentum_none_ns5": "none",
    }

    primary = contrast[
        (contrast["contrast"] == "matched_none")
        & contrast["candidate"].isin(
            ["ema_momentum_diag_ns5", "ema_momentum_full_ns5"]
        )
        & (contrast["eval_kind"] == "heldout")
        & contrast["eta"].isin([0.01, 0.02])
    ].copy()
    primary["mode"] = primary["candidate"].map(label_map)
    primary["result"] = primary.apply(
        lambda row: (
            "worse than none"
            if row["ci95_low"] > 0
            else "point estimate worse; interval crosses zero"
        ),
        axis=1,
    )
    primary = primary[
        [
            "mode",
            "eta",
            "mean",
            "sd",
            "ci95_low",
            "ci95_high",
            "negative_repeats",
            "positive_repeats",
            "result",
        ]
    ].sort_values(["eta", "mode"])

    layer_view = layer[
        layer["candidate"].isin(
            ["ema_momentum_diag_ns5", "ema_momentum_full_ns5"]
        )
        & (layer["eval_kind"] == "heldout")
        & (layer["eta"] == 0.01)
    ].copy()
    layer_view["mode"] = layer_view["candidate"].map(label_map)
    layer_view["depth_band"] = layer_view["layer"].map(
        lambda value: (
            "early"
            if value <= 7
            else "middle"
            if value <= 15
            else "late"
        )
    )
    layer_view = layer_view.rename(columns={"mean": "mean_delta_vs_none"})[
        [
            "layer",
            "depth_band",
            "mode",
            "mean_delta_vs_none",
            "sd",
            "ci95_low",
            "ci95_high",
            "negative_repeats",
            "positive_repeats",
        ]
    ].sort_values(["mode", "layer"])

    depth_view = depth[
        depth["candidate"].isin(
            ["ema_momentum_diag_ns5", "ema_momentum_full_ns5"]
        )
        & (depth["eval_kind"] == "heldout")
        & (depth["eta"] == 0.01)
    ].copy()
    depth_view["mode"] = depth_view["candidate"].map(label_map)
    depth_order = {"early": 0, "middle": 1, "late": 2}
    depth_view["depth_order"] = depth_view["depth_band"].map(depth_order)
    depth_view = depth_view[
        [
            "mode",
            "depth_band",
            "depth_order",
            "mean",
            "ci95_low",
            "ci95_high",
            "negative_repeats",
            "positive_repeats",
        ]
    ].sort_values(["depth_order", "mode"])

    momentum = contrast[
        (contrast["contrast"] == "momentum_minus_ema_gradient")
        & contrast["candidate"].isin(
            [
                "ema_momentum_none_ns5",
                "ema_momentum_diag_ns5",
                "ema_momentum_full_ns5",
            ]
        )
        & (contrast["eval_kind"] == "heldout")
        & (contrast["eta"] == 0.01)
    ].copy()
    momentum["mode"] = momentum["candidate"].map(label_map)
    momentum = momentum[
        [
            "mode",
            "mean",
            "sd",
            "ci95_low",
            "ci95_high",
            "negative_repeats",
            "positive_repeats",
        ]
    ].sort_values("mean")

    ema_matched = contrast[
        (contrast["contrast"] == "matched_none")
        & contrast["candidate"].isin(
            ["ema_gradient_diag_ns5", "ema_gradient_full_ns5"]
        )
        & (contrast["eval_kind"] == "heldout")
        & (contrast["eta"] == 0.01)
    ].copy()
    ema_lookup = {
        row.candidate.split("_")[2]: float(row.mean)
        for row in ema_matched.itertuples(index=False)
    }
    momentum_lookup = {
        row.mode: float(row.mean) for row in momentum.itertuples(index=False)
    }
    total_lookup = {
        row.mode: float(row.mean)
        for row in primary[primary["eta"] == 0.01].itertuples(index=False)
    }
    decomposition_rows = []
    for mode in ("diag", "full"):
        lost_history = momentum_lookup[mode] - momentum_lookup["none"]
        total = total_lookup[mode]
        decomposition_rows.append(
            {
                "mode": mode,
                "ema_gradient_delta_vs_none": ema_lookup[mode],
                "lost_momentum_benefit_vs_none": lost_history,
                "total_momentum_delta_vs_none": total,
                "lost_history_share_of_total": lost_history / total,
                "momentum_benefit_preserved_fraction": (
                    abs(momentum_lookup[mode])
                    / abs(momentum_lookup["none"])
                ),
            }
        )
    decomposition = pd.DataFrame(decomposition_rows)

    momentum_geometry = geometry[
        geometry["candidate"].isin(
            [
                "ema_momentum_none_ns5",
                "ema_momentum_diag_ns5",
                "ema_momentum_full_ns5",
            ]
        )
    ].copy()
    momentum_geometry["mode"] = momentum_geometry["candidate"].map(label_map)
    momentum_geometry = momentum_geometry[
        [
            "mode",
            "projection_input_cos_vs_gradient_mean",
            "direction_cos_vs_matched_none_mean",
            "direction_norm_ratio_vs_matched_none_mean",
            "alignment_normalized_mean",
            "row_orthogonality_residual_mean",
        ]
    ].sort_values("projection_input_cos_vs_gradient_mean", ascending=False)

    svd_rows = []
    for comparison, field, candidates, eval_kind in (
        (
            "SVD structured K minus SVD none",
            "matched_none",
            ["fresh_gradient_diag_svd", "fresh_gradient_full_svd"],
            "same",
        ),
        (
            "SVD minus matched NS5",
            "svd_minus_ns5",
            [
                "fresh_gradient_none_svd",
                "fresh_gradient_diag_svd",
                "fresh_gradient_full_svd",
            ],
            "same",
        ),
        (
            "SVD minus matched NS5",
            "svd_minus_ns5",
            [
                "fresh_gradient_none_svd",
                "fresh_gradient_diag_svd",
                "fresh_gradient_full_svd",
            ],
            "heldout",
        ),
    ):
        selected = contrast[
            (contrast["contrast"] == field)
            & contrast["candidate"].isin(candidates)
            & (contrast["eval_kind"] == eval_kind)
            & (contrast["eta"] == 0.01)
        ]
        for row in selected.itertuples(index=False):
            svd_rows.append(
                {
                    "comparison": comparison,
                    "evaluation": eval_kind,
                    "mode": row.candidate.split("_")[2],
                    "mean": row.mean,
                    "ci95_low": row.ci95_low,
                    "ci95_high": row.ci95_high,
                    "negative_repeats": row.negative_repeats,
                    "positive_repeats": row.positive_repeats,
                }
            )
    svd_effects = pd.DataFrame(svd_rows)
    svd_quality = geometry[geometry["projection"] == "svd"][
        [
            "k_mode",
            "projection_compute_dtype",
            "row_orthogonality_residual_mean",
            "row_orthogonality_residual_max",
            "ns5_svd_cos_mean",
        ]
    ].rename(columns={"k_mode": "mode"})

    historical_view = (
        historical.groupby("mode", as_index=False)
        .agg(seeds=("seed", "nunique"), val_loss_mean=("val_loss", "mean"))
        .sort_values("val_loss_mean")
    )
    none_val = float(
        historical_view.loc[
            historical_view["mode"] == "none", "val_loss_mean"
        ].iloc[0]
    )
    historical_view["delta_vs_none"] = (
        historical_view["val_loss_mean"] - none_val
    )

    resources = key["trajectory_and_resources"]
    resource_view = pd.DataFrame([resources])
    all_quality = pd.concat(
        [
            quality.assign(check_family="source_and_protocol"),
            validation.rename(
                columns={"evidence": "evidence"}
            ).assign(
                severity_if_failed="high",
                check_family="independent_recalculation",
            )[
                [
                    "check",
                    "status",
                    "severity_if_failed",
                    "evidence",
                    "check_family",
                ]
            ],
        ],
        ignore_index=True,
    )

    chart_map = pd.DataFrame(
        [
            {
                "section": "all-layer map",
                "question": "Where do diag and full differ from matched none?",
                "family": "trend over ordered architecture depth",
                "type": "line",
                "fields": "layer, mean_delta_vs_none, mode",
                "claim": "diag is positive in 22/24 layers; full is positive in 24/24",
                "palette": "hard two-root cap",
            },
            {
                "section": "momentum mechanism",
                "question": "How much held-out benefit does momentum add by K mode?",
                "family": "category comparison",
                "type": "bar",
                "fields": "mode, mean",
                "claim": "full retains only a small fraction of none's momentum benefit",
                "palette": "single-root preferred",
            },
        ]
    )
    chart_map.to_csv(processed / "report_chart_map.csv", index=False)

    generated_at = datetime.now(timezone.utc).isoformat()
    sources = [
        source(
            "primary_effects",
            "All-layer temporal matched-none effects",
            "processed/mechanism_contrast_summary.csv",
            (
                "SELECT * FROM mechanism_contrast_summary WHERE "
                "contrast='matched_none' AND eval_kind='heldout' AND "
                "candidate IN ('ema_momentum_diag_ns5',"
                "'ema_momentum_full_ns5')"
            ),
            (
                "Four build-repeat aggregates; each averages 24 layers and "
                "eight shared held-out batches."
            ),
            generated_at,
        ),
        source(
            "layer_map",
            "Layer-wise temporal matched-none effects",
            "processed/layer_effect_summary.csv",
            (
                "SELECT * FROM layer_effect_summary WHERE "
                "eval_kind='heldout' AND eta=0.01"
            ),
            (
                "Layer means and intervals across four build repeats for the "
                "EMA-momentum diag and full candidates."
            ),
            generated_at,
        ),
        source(
            "depth_bands",
            "Early, middle, and late depth-band effects",
            "processed/depth_band_summary.csv",
            (
                "SELECT * FROM depth_band_summary WHERE "
                "eval_kind='heldout' AND eta=0.01"
            ),
            (
                "Pre-registered eight-layer bands aggregated within each build."
            ),
            generated_at,
        ),
        source(
            "momentum_effects",
            "Momentum versus EMA-gradient effects",
            "processed/mechanism_contrast_summary.csv",
            (
                "SELECT * FROM mechanism_contrast_summary WHERE "
                "contrast='momentum_minus_ema_gradient' "
                "AND eval_kind='heldout' AND eta=0.01"
            ),
            (
                "Paired held-out change from adding historical momentum to the "
                "current EMA-preconditioned gradient."
            ),
            generated_at,
        ),
        source(
            "momentum_geometry",
            "EMA-momentum direction geometry",
            "processed/candidate_geometry_summary.csv",
            (
                "SELECT * FROM candidate_geometry_summary WHERE "
                "buffer_source='momentum' AND projection='ns5'"
            ),
            "Means over 24 layers and four build repeats.",
            generated_at,
        ),
        source(
            "svd_effects",
            "FP64 exact-SVD line-search effects",
            "processed/mechanism_contrast_summary.csv",
            (
                "SELECT * FROM mechanism_contrast_summary WHERE "
                "candidate LIKE '%_svd' AND eta=0.01"
            ),
            (
                "Same-batch and held-out FP64-SVD comparisons aggregated over "
                "all 24 layers."
            ),
            generated_at,
        ),
        source(
            "svd_quality",
            "FP64 exact-SVD orthogonality",
            "processed/candidate_geometry_summary.csv",
            (
                "SELECT * FROM candidate_geometry_summary "
                "WHERE projection='svd'"
            ),
            (
                "Applied FP32 direction after FP64 SVD; maximum residual is "
                "checked against 1e-4."
            ),
            generated_at,
        ),
        source(
            "seed_gate",
            "Pre-registered P3 decision gates",
            "processed/seed_gate_checks.csv",
            "SELECT * FROM seed_gate_checks",
            (
                "Global diag advantage, full damage, FP64 SVD, and seed "
                "expansion decisions."
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
                "Matched long-run none, diag, and full results for seeds "
                "2024, 2025, and 2026."
            ),
            generated_at,
        ),
        source(
            "quality_checks",
            "P3 source and independent validation checks",
            "processed/data_quality_checks.csv; processed/validation_checks.csv",
            "SELECT * FROM data_quality_checks UNION ALL SELECT * FROM validation_checks",
            (
                "Coverage, numerical controls, all-layer shadow state, FP64 "
                "SVD, W&B integrity, aggregation reconciliation, and gate checks."
            ),
            generated_at,
        ),
        source(
            "trajectory_resource",
            "P3 trajectory and resource summary",
            "processed/key_results.json",
            "SELECT * FROM trajectory_and_resources",
            (
                "Training endpoint, probe time, peak memory, optimizer state, "
                "and diagnostic shadow state."
            ),
            generated_at,
        ),
    ]

    tables = [
        {
            "id": "primary_table",
            "title": "All-layer EMA-momentum effects versus matched none",
            "subtitle": (
                "Held-out paired loss change; negative favors the K-mode "
                "candidate. Intervals use four build-repeat aggregates."
            ),
            "dataset": "primary_effects",
            "sourceId": "primary_effects",
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
                {"field": "result", "label": "Interpretation", "type": "text"},
            ],
        },
        {
            "id": "depth_table",
            "title": "Depth-band matched-none effects",
            "subtitle": "Held-out eta=0.01; each band contains eight layers.",
            "dataset": "depth_bands",
            "sourceId": "depth_bands",
            "defaultSort": {"field": "depth_order", "direction": "asc"},
            "columns": [
                {"field": "depth_band", "label": "Depth band", "type": "text"},
                {"field": "mode", "label": "K mode", "type": "text"},
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
                {"field": "depth_order", "label": "Order", "type": "number"},
            ],
        },
        {
            "id": "decomposition_table",
            "title": "Decomposition of the all-layer momentum effect",
            "subtitle": (
                "Held-out eta=0.01; total equals EMA-gradient difference plus "
                "lost momentum benefit relative to none."
            ),
            "dataset": "momentum_decomposition",
            "sourceId": "momentum_effects",
            "defaultSort": {
                "field": "total_momentum_delta_vs_none",
                "direction": "desc",
            },
            "columns": [
                {"field": "mode", "label": "K mode", "type": "text"},
                {
                    "field": "ema_gradient_delta_vs_none",
                    "label": "EMA-gradient delta",
                    "format": "number",
                },
                {
                    "field": "lost_momentum_benefit_vs_none",
                    "label": "Lost history benefit",
                    "format": "number",
                },
                {
                    "field": "total_momentum_delta_vs_none",
                    "label": "Total delta",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "lost_history_share_of_total",
                    "label": "History share",
                    "format": "percent",
                },
                {
                    "field": "momentum_benefit_preserved_fraction",
                    "label": "Momentum benefit preserved",
                    "format": "percent",
                },
            ],
        },
        {
            "id": "geometry_table",
            "title": "EMA-momentum direction geometry",
            "subtitle": "Means over 24 layers and four build repeats.",
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
                    "label": "Buffer cosine vs gradient",
                    "format": "number",
                },
                {
                    "field": "direction_cos_vs_matched_none_mean",
                    "label": "Direction cosine vs none",
                    "format": "number",
                },
                {
                    "field": "direction_norm_ratio_vs_matched_none_mean",
                    "label": "Norm ratio vs none",
                    "format": "number",
                },
                {
                    "field": "alignment_normalized_mean",
                    "label": "Gradient alignment",
                    "format": "number",
                },
            ],
        },
        {
            "id": "svd_effect_table",
            "title": "FP64 exact-SVD line-search comparisons",
            "subtitle": "Eta=0.01; all values use four all-layer build aggregates.",
            "dataset": "svd_effects",
            "sourceId": "svd_effects",
            "defaultSort": {"field": "evaluation", "direction": "asc"},
            "columns": [
                {"field": "comparison", "label": "Comparison", "type": "text"},
                {"field": "evaluation", "label": "Evaluation", "type": "text"},
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
                {
                    "field": "positive_repeats",
                    "label": "Positive builds",
                    "type": "number",
                },
            ],
        },
        {
            "id": "svd_quality_table",
            "title": "FP64 SVD orthogonality",
            "subtitle": "Applied FP32 directions; required maximum residual ≤0.0001.",
            "dataset": "svd_quality",
            "sourceId": "svd_quality",
            "defaultSort": {
                "field": "row_orthogonality_residual_max",
                "direction": "desc",
            },
            "columns": [
                {"field": "mode", "label": "K mode", "type": "text"},
                {
                    "field": "projection_compute_dtype",
                    "label": "SVD dtype",
                    "type": "text",
                },
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
            "title": "Historical three-seed validation ordering",
            "subtitle": "Step10000, matched OpenWebText 24L/D1024 training runs.",
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
            "id": "seed_gate_table",
            "title": "Pre-registered P3 decision gates",
            "subtitle": "PASS means the stated scientific condition was met.",
            "dataset": "seed_gate",
            "sourceId": "seed_gate",
            "defaultSort": {"field": "family", "direction": "asc"},
            "columns": [
                {"field": "family", "label": "Gate family", "type": "text"},
                {"field": "check", "label": "Condition", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
            ],
        },
        {
            "id": "quality_table",
            "title": "Source, protocol, and independent validation checks",
            "subtitle": "All checks pass; exact HVP missingness is intentional.",
            "dataset": "quality_checks",
            "sourceId": "quality_checks",
            "defaultSort": {"field": "check_family", "direction": "asc"},
            "columns": [
                {"field": "check_family", "label": "Family", "type": "text"},
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
            "title": "Trajectory and resource record",
            "subtitle": "Shadow state is diagnostic overhead, not none optimizer state.",
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
                    "field": "train_loss_step10000",
                    "label": "Train loss at 10000",
                    "format": "number",
                },
                {
                    "field": "total_elapsed_seconds_step10000",
                    "label": "Total seconds",
                    "format": "number",
                },
                {
                    "field": "probe_seconds",
                    "label": "Probe seconds",
                    "format": "number",
                },
                {
                    "field": "full_run_peak_memory_mib",
                    "label": "Peak MiB",
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
                    "field": "diagnostic_shadow_total_mib",
                    "label": "Probe-only shadow MiB",
                    "format": "number",
                },
            ],
        },
    ]

    charts = [
        {
            "id": "layer_map_chart",
            "title": "Held-out matched-none effect across c_proj layers",
            "subtitle": (
                "EMA-momentum, eta=0.01; four-build mean. Negative favors the "
                "K-mode candidate."
            ),
            "type": "line",
            "dataset": "layer_map",
            "sourceId": "layer_map",
            "valueFormat": "number",
            "encodings": {
                "x": {
                    "field": "layer",
                    "type": "quantitative",
                    "label": "c_proj layer",
                },
                "y": {
                    "field": "mean_delta_vs_none",
                    "type": "quantitative",
                    "label": "Mean loss delta versus none",
                },
                "color": {
                    "field": "mode",
                    "type": "nominal",
                    "label": "K mode",
                },
                "tooltip": [
                    {"field": "layer", "type": "quantitative", "label": "Layer"},
                    {
                        "field": "depth_band",
                        "type": "nominal",
                        "label": "Depth band",
                    },
                    {"field": "mode", "type": "nominal", "label": "K mode"},
                    {
                        "field": "mean_delta_vs_none",
                        "type": "quantitative",
                        "label": "Delta vs none",
                        "format": "number",
                    },
                    {
                        "field": "positive_repeats",
                        "type": "quantitative",
                        "label": "Builds worse",
                    },
                ],
            },
        },
        {
            "id": "momentum_chart",
            "title": "Held-out effect of adding momentum history by K mode",
            "subtitle": (
                "Eta=0.01; momentum minus EMA-gradient. More negative is a "
                "larger held-out benefit."
            ),
            "type": "bar",
            "dataset": "momentum_effects",
            "sourceId": "momentum_effects",
            "valueFormat": "number",
            "encodings": {
                "x": {
                    "field": "mode",
                    "type": "nominal",
                    "label": "K mode",
                },
                "y": {
                    "field": "mean",
                    "type": "quantitative",
                    "label": "Momentum minus EMA-gradient loss change",
                },
                "tooltip": [
                    {"field": "mode", "type": "nominal", "label": "K mode"},
                    {
                        "field": "mean",
                        "type": "quantitative",
                        "label": "Mean effect",
                        "format": "number",
                    },
                    {
                        "field": "ci95_low",
                        "type": "quantitative",
                        "label": "95% CI low",
                        "format": "number",
                    },
                    {
                        "field": "ci95_high",
                        "type": "quantitative",
                        "label": "95% CI high",
                        "format": "number",
                    },
                ],
            },
        },
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# c_proj 全层时间状态探针 P3",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## 技术摘要\n\n"
                "**P3 正式支持 full K-state 的全层优化损害，但否定了 P2 "
                "三层探针中的全局 diag 优势。** Held-out、eta=0.01 时，"
                "diag-minus-none 为 +3.30e-6（4/4 build 更差，区间跨 0），"
                "full-minus-none 为 +1.74e-5（4/4 更差，区间完全大于 0）。"
                "Diag 在 22/24 层为正；删除 layer 0 后仍为 +4.12e-6。"
                "FP64 exact-SVD 全部通过正交性门控，并证明有限 NS5 误差不是"
                "structured-K same-batch 损害的必要原因。预注册 seed 扩展失败。"
            ),
        },
        {
            "id": "primary_result",
            "type": "markdown",
            "sourceId": "primary_effects",
            "body": (
                "## 全层平均下 diag 不优于 none，full 稳定更差\n\n"
                "Eta=0.01/0.02 时，diag 的点估计分别为 +3.30e-6/+6.57e-6，"
                "full 为 +1.74e-5/+3.45e-5。四个 build 在两个 eta 上均为正；"
                "full 的区间完全大于 0，diag 的区间略跨 0。近两倍的 eta "
                "产生近两倍效应，支持局部稳定性。"
            ),
        },
        {"id": "primary_table_block", "type": "table", "tableId": "primary_table"},
        {
            "id": "layer_result",
            "type": "markdown",
            "sourceId": "layer_map",
            "body": (
                "## P2 的负号来自 layer 0 在三层平均中被过度加权\n\n"
                "P3 中 diag 只有 layers 0 和 23 的均值为负，其余 22 层为正；"
                "full 在 24/24 层为正。Layer 0 的 diag 效应为 -1.56e-5，"
                "但只占全层平均的 1/24。删除 layer 0 后，diag 剩余均值反而"
                "升到 +4.12e-6，4/4 build 更差。"
            ),
        },
        {"id": "layer_chart_block", "type": "chart", "chartId": "layer_map_chart"},
        {
            "id": "depth_result",
            "type": "markdown",
            "sourceId": "depth_bands",
            "body": (
                "## 没有任何预注册 depth band 支持 diag 优势\n\n"
                "Early、middle、late 的 diag 均值分别为 +5.19e-6、+2.96e-6、"
                "+1.74e-6；full 分别为 +3.56e-5、+1.02e-5、+6.40e-6。"
                "因此信号不是一个可推广的 early-layer diag 效应。"
            ),
        },
        {"id": "depth_table_block", "type": "table", "tableId": "depth_table"},
        {
            "id": "momentum_result",
            "type": "markdown",
            "sourceId": "momentum_effects",
            "body": (
                "## Full 的主要损害是丢失 momentum 的跨 batch 收益\n\n"
                "加入历史 momentum 后，none 的 held-out 改善为 -1.31e-5，"
                "diag 为 -1.05e-5，full 只有 -1.09e-6。Diag 保留约 80% 的 "
                "none 收益，full 只保留约 8%。Full 总损害约 69% 来自这部分"
                "历史收益丢失，其余来自当前 EMA-gradient 方向。"
            ),
        },
        {"id": "momentum_chart_block", "type": "chart", "chartId": "momentum_chart"},
        {
            "id": "decomposition_intro",
            "type": "markdown",
            "body": (
                "## 配对分解将当前方向与历史效应分开\n\n"
                "总 matched-none 差严格等于 EMA-gradient 差，加上该 mode "
                "相对 none 丢失的 momentum 收益。该恒等式避免把两类机制"
                "混在一个总数中。"
            ),
        },
        {
            "id": "decomposition_table_block",
            "type": "table",
            "tableId": "decomposition_table",
        },
        {
            "id": "geometry_result",
            "type": "markdown",
            "sourceId": "momentum_geometry",
            "body": (
                "## 几何量解释为什么 diag 优于 full，却不优于 none\n\n"
                "历史 buffer 与当前梯度 cosine 为 none 0.767、diag 0.641、"
                "full 0.337；最终方向与 matched-none cosine 为 1.000、0.869、"
                "0.806。Diag 的坐标缩放保留了大部分历史结构，full 的特征混合"
                "则严重破坏它，但 none 仍保持最高一致性。"
            ),
        },
        {"id": "geometry_table_block", "type": "table", "tableId": "geometry_table"},
        {
            "id": "svd_result",
            "type": "markdown",
            "sourceId": "svd_effects",
            "body": (
                "## FP64 exact polar 仍显示 structured-K same-batch 损害\n\n"
                "SVD 相对 NS5 在三个 mode 上都改善 same-batch loss，但 exact "
                "diag/full 相对 exact none 仍分别差 +8.86e-4/+3.45e-4，"
                "4/4 build 更差且区间完全大于 0。Held-out 上 SVD 与 NS5 "
                "差异接近 0。有限 NS5 误差存在，但不是 K-state 损害的必要原因。"
            ),
        },
        {"id": "svd_effect_block", "type": "table", "tableId": "svd_effect_table"},
        {
            "id": "svd_quality_result",
            "type": "markdown",
            "sourceId": "svd_quality",
            "body": (
                "## P2 的 SVD 数值问题已被修复\n\n"
                "所有 SVD 均在 float64 中计算，应用方向转回 float32 后的最大"
                "行正交残差为 4.23e-7，低于 1e-4 门槛约 237 倍。"
            ),
        },
        {
            "id": "svd_quality_block",
            "type": "table",
            "tableId": "svd_quality_table",
        },
        {
            "id": "historical_result",
            "type": "markdown",
            "sourceId": "historical_training",
            "body": (
                "## Fixed-checkpoint 探针解释 full，却仍解释不了长期 diag 优势\n\n"
                "历史三-seed step10000 中，diag-minus-none=-0.006890，"
                "full-minus-none=+0.055379。P3 与 full 的长期损害一致；但 "
                "diag 的长期优势与其单层独立方向相反，说明关键机制必须包含"
                "多步参数—K—momentum 反馈、所有层同时更新或轨迹依赖。"
            ),
        },
        {
            "id": "historical_table_block",
            "type": "table",
            "tableId": "historical_table",
        },
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## 实验与统计定义\n\n"
                "P3 在一条 OpenWebText 24L/D1024 none 轨迹的 step10000 上，"
                "对全部 24 个 c_proj 层分别构造反事实方向。四个 build repeat "
                "是独立统计单位；每个 build 内先平均八个共用 held-out batch "
                "和目标层。该设计测量单层瞬时替换，不测所有层同步更新后的"
                "参数反馈。Exact HVP 按预注册协议关闭。"
            ),
        },
        {
            "id": "quality_result",
            "type": "markdown",
            "sourceId": "quality_checks",
            "body": (
                "## 数据与独立复算全部通过\n\n"
                "16 个附件、1152 个方向、51840 条 line-search、24 层 shadow、"
                "FP64 SVD 与八个 W&B 指标均完整。10 个来源/协议检查和 6 个"
                "独立聚合复算检查全部通过。"
            ),
        },
        {"id": "quality_table_block", "type": "table", "tableId": "quality_table"},
        {
            "id": "seed_result",
            "type": "markdown",
            "sourceId": "seed_gate",
            "body": (
                "## 不补 unchanged P3 的 seed2024/2025\n\n"
                "Diag global-advantage 的四项预注册条件全部失败；full damage "
                "四项全部通过；FP64 SVD 门控通过。Seed 扩展条件因此明确失败，"
                "继续复制相同 fixed-checkpoint probe 不具备信息价值。"
            ),
        },
        {"id": "seed_table_block", "type": "table", "tableId": "seed_gate_table"},
        {
            "id": "resource_result",
            "type": "markdown",
            "sourceId": "trajectory_resource",
            "body": (
                "## 全层探针耗时与显存符合设计预期\n\n"
                "探针耗时 532.745 秒，总 elapsed 1527.681 秒，full-run peak "
                "为 11580.641 MiB。None optimizer K-state 仍为 864 MiB、"
                "c_proj 为 0；额外 5376.75 MiB 是全层 shadow 诊断状态。"
            ),
        },
        {"id": "resource_table_block", "type": "table", "tableId": "resource_table"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 证据边界\n\n"
                "- 只有一个 seed2026 fixed-checkpoint 轨迹；build 区间条件于"
                "固定 held-out 集。\n"
                "- 每次只替换一个 layer，不能观察 cross-layer simultaneous "
                "update。\n"
                "- Shadow diag/full state 位于 none 参数轨迹，并非它们各自"
                "训练轨迹上的 optimizer state。\n"
                "- Layer 0 的 diag 均值为负，但只有 3/4 build，区间跨 0，"
                "不足以启动 layer0-only 长期训练。\n"
                "- 结果支持 momentum 收益丢失是 full 损害的重要组成，但不"
                "证明它是唯一因果来源。"
            ),
        },
        {
            "id": "next_step",
            "type": "markdown",
            "body": (
                "## 下一步改做短程多步反事实 rollout\n\n"
                "固定检查点单层方向探针已经足以停止 seed 扩展。下一实验应从"
                "同一 checkpoint 与共享数据流克隆分支，同时应用全层 none、"
                "diag、full 更新并 rollout 16/32/64 步，测量 loss、方向漂移、"
                "K/momentum 演化与跨层联合效应。这是回答长期 diag 优势所缺"
                "少的状态反馈层级。"
            ),
        },
        {
            "id": "open_questions",
            "type": "markdown",
            "body": (
                "## 尚待回答的问题\n\n"
                "- Diag 的长期优势最早在训练的哪个阶段出现？\n"
                "- 优势来自全层同步更新还是参数—K—momentum 的多步反馈？\n"
                "- 在各自真实参数轨迹上，diag/full 的状态几何是否仍与 none "
                "轨迹反事实一致？\n"
                "- 短程 rollout 的哪项统计最能预测三-seed 长期 val loss？"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "c_proj 全层时间状态探针 P3",
            "description": (
                "全层 temporal K-state、momentum 分解、FP64 exact-polar "
                "对照与预注册 seed 判定。"
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
                "primary_effects": records(primary),
                "layer_map": records(layer_view),
                "depth_bands": records(depth_view),
                "momentum_effects": records(momentum),
                "momentum_decomposition": records(decomposition),
                "momentum_geometry": records(momentum_geometry),
                "svd_effects": records(svd_effects),
                "svd_quality": records(svd_quality),
                "historical_training": records(historical_view),
                "seed_gate": records(seed_gate),
                "quality_checks": records(all_quality),
                "trajectory_resource": records(resource_view),
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    artifact_path = run_dir / "quadratic_probe_p3_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(artifact_path)


if __name__ == "__main__":
    main()
