#!/usr/bin/env python
"""Build the canonical portable report artifact for quadratic probe P1."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


MODE_ORDER = {
    "none": 0,
    "scalar": 1,
    "diag": 2,
    "block4": 3,
    "full": 4,
}


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
    working = pd.read_csv(processed / "working_point_summary.csv")
    builds = pd.read_csv(processed / "hierarchical_build_repeat.csv")
    geometry = pd.read_csv(processed / "geometry_mode_summary.csv")
    normmatch = pd.read_csv(processed / "normmatch_effects_summary.csv")
    exact_svd = pd.read_csv(processed / "exact_svd_summary.csv")
    p0p1 = pd.read_csv(processed / "p0_p1_comparison.csv")
    quality = pd.read_csv(processed / "data_quality_checks.csv")

    key_diag = working[
        (working["mode"] == "diag")
        & (
            (
                (working["eval_kind"] == "same")
                & (working["lr_multiplier"] == 1.0)
            )
            | (
                (working["eval_kind"] == "heldout")
                & working["lr_multiplier"].isin([1.0, 2.0])
            )
        )
    ].copy()
    key_diag["result"] = key_diag.apply(
        lambda row: (
            "worse than none"
            if row["repeat_ci95_low"] > 0
            else "not resolved"
        ),
        axis=1,
    )
    key_diag = key_diag[
        [
            "eval_kind",
            "eta",
            "repeat_mean_delta_vs_none",
            "repeat_sd_delta_vs_none",
            "repeat_ci95_low",
            "repeat_ci95_high",
            "better_than_none_repeats",
            "worse_than_none_repeats",
            "result",
        ]
    ].sort_values(["eval_kind", "eta"])

    diag_build_chart = builds[
        (builds["mode"] == "diag")
        & (builds["eval_kind"] == "heldout")
        & (builds["lr_multiplier"] == 1.0)
    ][
        [
            "build_repeat",
            "delta_vs_none_mean",
            "loss_delta_mean",
            "layers",
            "observations",
        ]
    ].copy()
    diag_build_chart["build_repeat_label"] = diag_build_chart[
        "build_repeat"
    ].map(lambda value: f"build {int(value)}")

    geometry_view = geometry[geometry["mode"].isin(MODE_ORDER)].copy()
    geometry_view["mode_order"] = geometry_view["mode"].map(MODE_ORDER)
    geometry_view = geometry_view.sort_values("mode_order")[
        [
            "mode",
            "alignment_normalized_mean",
            "direction_cos_vs_none_mean",
            "direction_norm_ratio_vs_none_mean",
            "projector_drift_vs_none_mean",
            "row_orthogonality_residual_mean",
            "ns_svd_cos_mean",
            "mode_order",
        ]
    ]

    normmatch_view = normmatch[
        (normmatch["eval_kind"] == "heldout")
        & (normmatch["lr_multiplier"] == 1.0)
    ][
        [
            "base_mode",
            "base_delta_vs_none_mean",
            "normmatch_delta_vs_none_mean",
            "normmatch_minus_base_mean",
            "normmatch_minus_base_std",
            "normmatch_improved_repeats",
        ]
    ].sort_values("base_mode")

    exact_view = exact_svd[exact_svd["mode"].isin(MODE_ORDER)].copy()
    exact_view["mode_order"] = exact_view["mode"].map(MODE_ORDER)
    exact_view = exact_view.sort_values("mode_order")[
        [
            "mode",
            "exact_svd_observations",
            "ns_svd_cos_mean",
            "ns_svd_cos_min",
            "ns_svd_cos_max",
            "row_orthogonality_residual_mean",
            "mode_order",
        ]
    ]

    p0p1_view = p0p1[
        p0p1["metric"]
        != "p1_emitted_scalar_control_pass"
    ].copy()
    p0p1_view["change_from_p0_to_p1"] = p0p1_view["p1"] - p0p1_view["p0"]

    resource = key["trajectory_and_resource"]
    resource_view = pd.DataFrame(
        [
            {
                "val_loss_step10000": resource["val_loss_step10000"],
                "historical_none_val_loss_step10000": resource[
                    "historical_none_seed2026_val_loss_step10000"
                ],
                "delta_vs_historical": resource["delta_vs_historical"],
                "training_elapsed_seconds": resource[
                    "training_elapsed_seconds_step10000"
                ],
                "training_peak_memory_mib": resource["training_peak_memory_mib"],
                "total_k_state_mib": resource["total_k_state_mib"],
                "cproj_k_state_mib": resource["cproj_k_state_mib"],
                "probe_seconds": resource["probe_seconds"],
                "probe_peak_memory_mib": resource["probe_peak_memory_mib"],
            }
        ]
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    sources = [
        source(
            "hierarchical_effects",
            "Hierarchical fixed-step effects",
            "processed/working_point_summary.csv",
            (
                "SELECT * FROM working_point_summary "
                "WHERE mode='diag' AND lr_multiplier IN (1, 2)"
            ),
            (
                "Effects are aggregated first within each of four build repeats; "
                "the four build aggregates are the uncertainty units."
            ),
            generated_at,
        ),
        source(
            "build_repeat_effects",
            "Diag heldout effects by build repeat",
            "processed/hierarchical_build_repeat.csv",
            (
                "SELECT build_repeat, delta_vs_none_mean, loss_delta_mean, "
                "layers, observations FROM hierarchical_build_repeat "
                "WHERE mode='diag' AND eval_kind='heldout' AND lr_multiplier=1"
            ),
            (
                "Each plotted row averages three layers and eight heldout batches "
                "for one independently constructed fresh K."
            ),
            generated_at,
        ),
        source(
            "direction_geometry",
            "Direction geometry by K mode",
            "processed/geometry_mode_summary.csv",
            (
                "SELECT * FROM geometry_mode_summary "
                "WHERE mode IN ('none','scalar','diag','block4','full')"
            ),
            (
                "Means over four build repeats and layers 0, 11, and 23 at "
                "the fixed step-10000 checkpoint."
            ),
            generated_at,
        ),
        source(
            "normmatch_effects",
            "Final-direction norm-match control",
            "processed/normmatch_effects_summary.csv",
            (
                "SELECT * FROM normmatch_effects_summary "
                "WHERE eval_kind='heldout' AND lr_multiplier=1"
            ),
            (
                "Paired effect of rescaling each NS5 direction to the none "
                "direction Frobenius norm."
            ),
            generated_at,
        ),
        source(
            "exact_svd",
            "NS5 versus exact polar geometry",
            "processed/exact_svd_summary.csv",
            (
                "SELECT * FROM exact_svd_summary "
                "WHERE mode IN ('none','scalar','diag','block4','full')"
            ),
            (
                "Cosine between NS5 and exact-SVD polar directions in the first "
                "build repeat across three layers."
            ),
            generated_at,
        ),
        source(
            "p0_p1",
            "P0 and P1 numerical comparison",
            "processed/p0_p1_comparison.csv",
            "SELECT * FROM p0_p1_comparison",
            (
                "The P0 single-build/single-heldout effects are compared with "
                "strict-FP32 P1 effects."
            ),
            generated_at,
        ),
        source(
            "quality_checks",
            "P1 data-quality checks",
            "processed/data_quality_checks.csv",
            "SELECT * FROM data_quality_checks ORDER BY check",
            (
                "Source coverage, key coverage, numerical controls, exact-SVD "
                "coverage, and W&B integrity."
            ),
            generated_at,
        ),
        source(
            "trajectory_resource",
            "Trajectory and resource summary",
            "processed/key_results.json",
            "SELECT * FROM trajectory_and_resource",
            (
                "Step-10000 trajectory endpoint, runtime, peak memory, K-state, "
                "and probe overhead."
            ),
            generated_at,
        ),
    ]

    tables = [
        {
            "id": "diag_effect_table",
            "title": "Diag versus none at the working step sizes",
            "subtitle": (
                "Delta is candidate minus none; negative favors diag. "
                "Intervals use four build-repeat aggregates."
            ),
            "dataset": "diag_effects",
            "sourceId": "hierarchical_effects",
            "defaultSort": {"field": "eta", "direction": "asc"},
            "columns": [
                {"field": "eval_kind", "label": "Evaluation", "type": "text"},
                {"field": "eta", "label": "Eta", "format": "number"},
                {
                    "field": "repeat_mean_delta_vs_none",
                    "label": "Mean delta vs none",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "repeat_sd_delta_vs_none",
                    "label": "SD across builds",
                    "format": "number",
                },
                {
                    "field": "repeat_ci95_low",
                    "label": "95% CI low",
                    "format": "number",
                },
                {
                    "field": "repeat_ci95_high",
                    "label": "95% CI high",
                    "format": "number",
                },
                {
                    "field": "better_than_none_repeats",
                    "label": "Builds better",
                    "type": "number",
                },
                {
                    "field": "worse_than_none_repeats",
                    "label": "Builds worse",
                    "type": "number",
                },
                {"field": "result", "label": "Interpretation", "type": "text"},
            ],
        },
        {
            "id": "geometry_table",
            "title": "K changes direction, not merely update norm",
            "subtitle": "Means over four builds and three layers.",
            "dataset": "direction_geometry",
            "sourceId": "direction_geometry",
            "defaultSort": {
                "field": "alignment_normalized_mean",
                "direction": "desc",
            },
            "columns": [
                {"field": "mode", "label": "Mode", "type": "text"},
                {
                    "field": "alignment_normalized_mean",
                    "label": "Gradient alignment",
                    "format": "number",
                },
                {
                    "field": "direction_cos_vs_none_mean",
                    "label": "Direction cosine vs none",
                    "format": "number",
                },
                {
                    "field": "direction_norm_ratio_vs_none_mean",
                    "label": "Norm ratio vs none",
                    "format": "number",
                },
                {
                    "field": "projector_drift_vs_none_mean",
                    "label": "Projector drift",
                    "format": "number",
                },
                {
                    "field": "row_orthogonality_residual_mean",
                    "label": "Row-orth residual",
                    "format": "number",
                },
                {
                    "field": "ns_svd_cos_mean",
                    "label": "NS5–SVD cosine",
                    "format": "number",
                },
            ],
        },
        {
            "id": "normmatch_table",
            "title": "Norm matching does not rescue heldout performance",
            "subtitle": "Eta=0.01; negative normmatch-minus-base is an improvement.",
            "dataset": "normmatch_effects",
            "sourceId": "normmatch_effects",
            "defaultSort": {
                "field": "normmatch_minus_base_mean",
                "direction": "asc",
            },
            "columns": [
                {"field": "base_mode", "label": "Mode", "type": "text"},
                {
                    "field": "base_delta_vs_none_mean",
                    "label": "Base delta vs none",
                    "format": "number",
                },
                {
                    "field": "normmatch_delta_vs_none_mean",
                    "label": "Norm-matched delta vs none",
                    "format": "number",
                },
                {
                    "field": "normmatch_minus_base_mean",
                    "label": "Normmatch minus base",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "normmatch_improved_repeats",
                    "label": "Builds improved",
                    "type": "number",
                },
            ],
        },
        {
            "id": "exact_svd_table",
            "title": "Finite NS5 is far from exact polar for every mode",
            "subtitle": "First build repeat; three layers.",
            "dataset": "exact_svd",
            "sourceId": "exact_svd",
            "defaultSort": {"field": "ns_svd_cos_mean", "direction": "desc"},
            "columns": [
                {"field": "mode", "label": "Mode", "type": "text"},
                {
                    "field": "exact_svd_observations",
                    "label": "Layers",
                    "type": "number",
                },
                {
                    "field": "ns_svd_cos_mean",
                    "label": "Mean NS5–SVD cosine",
                    "format": "number",
                },
                {
                    "field": "ns_svd_cos_min",
                    "label": "Minimum",
                    "format": "number",
                },
                {
                    "field": "ns_svd_cos_max",
                    "label": "Maximum",
                    "format": "number",
                },
                {
                    "field": "row_orthogonality_residual_mean",
                    "label": "Row-orth residual",
                    "format": "number",
                },
            ],
        },
        {
            "id": "p0p1_table",
            "title": "P1 removes the P0 heldout signal and its control error",
            "subtitle": "Negative diag-minus-none favors diag.",
            "dataset": "p0_p1",
            "sourceId": "p0_p1",
            "defaultSort": {"field": "metric", "direction": "asc"},
            "columns": [
                {"field": "metric", "label": "Metric", "type": "text"},
                {"field": "p0", "label": "P0", "format": "number"},
                {"field": "p1", "label": "P1", "format": "number"},
                {
                    "field": "change_from_p0_to_p1",
                    "label": "P1 minus P0",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "interpretation",
                    "label": "Definition",
                    "type": "text",
                },
            ],
        },
        {
            "id": "quality_table",
            "title": "All P1 integrity and numerical controls pass",
            "subtitle": "The failed seed gate is a scientific result, not a data failure.",
            "dataset": "quality_checks",
            "sourceId": "quality_checks",
            "defaultSort": {"field": "check", "direction": "asc"},
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
            "title": "Trajectory and resource record",
            "subtitle": "The probe is isolated from the saved none trajectory.",
            "dataset": "trajectory_resource",
            "sourceId": "trajectory_resource",
            "defaultSort": {
                "field": "val_loss_step10000",
                "direction": "asc",
            },
            "columns": [
                {
                    "field": "val_loss_step10000",
                    "label": "P1 val loss",
                    "format": "number",
                },
                {
                    "field": "historical_none_val_loss_step10000",
                    "label": "Historical none val loss",
                    "format": "number",
                },
                {
                    "field": "delta_vs_historical",
                    "label": "Trajectory delta",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "training_elapsed_seconds",
                    "label": "Training seconds",
                    "format": "number",
                },
                {
                    "field": "training_peak_memory_mib",
                    "label": "Training peak MiB",
                    "format": "number",
                },
                {
                    "field": "total_k_state_mib",
                    "label": "K-state MiB",
                    "format": "number",
                },
                {
                    "field": "cproj_k_state_mib",
                    "label": "c_proj K-state MiB",
                    "format": "number",
                },
                {
                    "field": "probe_seconds",
                    "label": "Probe seconds",
                    "format": "number",
                },
                {
                    "field": "probe_peak_memory_mib",
                    "label": "Probe peak MiB",
                    "format": "number",
                },
            ],
        },
    ]

    charts = [
        {
            "id": "diag_build_chart",
            "title": "Diag heldout delta versus none by build repeat",
            "subtitle": (
                "Eta=0.01; each bar averages three layers and eight heldout "
                "batches. Negative favors diag."
            ),
            "type": "bar",
            "dataset": "diag_build_chart",
            "sourceId": "build_repeat_effects",
            "valueFormat": "number",
            "encodings": {
                "x": {
                    "field": "build_repeat_label",
                    "type": "nominal",
                    "label": "Fresh-K build repeat",
                },
                "y": {
                    "field": "delta_vs_none_mean",
                    "type": "quantitative",
                    "label": "Mean loss delta versus none",
                },
                "tooltip": [
                    {
                        "field": "delta_vs_none_mean",
                        "type": "quantitative",
                        "label": "Delta versus none",
                        "format": "number",
                    },
                    {
                        "field": "loss_delta_mean",
                        "type": "quantitative",
                        "label": "Diag loss delta",
                        "format": "number",
                    },
                    {
                        "field": "observations",
                        "type": "quantitative",
                        "label": "Layer × heldout observations",
                    },
                ],
            },
        }
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# c_proj Quadratic Probe P1 — Seed 2026",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## Technical summary\n\n"
                "**P1 passes every numerical control but fails the predeclared "
                "multi-seed gate.** It confirms that structured K rotates the "
                "fixed-state NS5 direction away from the current gradient. It does "
                "not confirm that diag produces a better fresh-K heldout direction, "
                "and it does not explain the known long-run diag advantage."
            ),
        },
        {
            "id": "primary_result",
            "type": "markdown",
            "sourceId": "hierarchical_effects",
            "body": (
                "## Diag reliably hurts same-batch descent but has no resolved heldout effect\n\n"
                "At eta=0.01, the same-batch diag-minus-none gap is +0.001244 "
                "with a four-build 95% interval of [0.000841, 0.001648]; all "
                "4/4 build aggregates and 12/12 build-by-layer cells are worse. "
                "The heldout gap is only +0.000004, with an interval of "
                "[-0.000006, 0.000015] and just 1/4 builds favoring diag."
            ),
        },
        {"id": "primary_table", "type": "table", "tableId": "diag_effect_table"},
        {
            "id": "repeat_chart_intro",
            "type": "markdown",
            "body": (
                "## The heldout sign is not stable across fresh-K builds\n\n"
                "The chart uses build-repeat aggregates rather than treating the "
                "96 correlated heldout rows as independent evidence."
            ),
        },
        {"id": "repeat_chart", "type": "chart", "chartId": "diag_build_chart"},
        {
            "id": "geometry_result",
            "type": "markdown",
            "sourceId": "direction_geometry",
            "body": (
                "## The stable mechanism is direction rotation\n\n"
                "Normalized gradient alignment falls from 0.610954 for none to "
                "0.551809 for full, 0.521609 for block4, and 0.481905 for diag. "
                "Diag's mean cosine with the none direction is 0.780246 while its "
                "norm ratio is only 1.003479. The same-batch loss ordering therefore "
                "tracks direction alignment, not update magnitude."
            ),
        },
        {"id": "geometry_table_block", "type": "table", "tableId": "geometry_table"},
        {
            "id": "math_result",
            "type": "markdown",
            "body": (
                "## Local mathematical interpretation\n\n"
                "For candidate direction Q_K = NS5(G P_K), the probe measures "
                "L(W − ηQ_K) − L(W) ≈ −η〈G,Q_K〉 + "
                "(η²/2)〈Q_K,HQ_K〉. For the exact polar map and wide c_proj "
                "gradient G = UΣVᵀ, UVᵀ solves max 〈G,Q〉 subject to QQᵀ = I. "
                "Thus none already maximizes the current-batch first-order descent "
                "among equal row-orthogonal candidates; a non-scalar right "
                "preconditioner generally rotates away from that solution. Scalar "
                "P_K = cI disappears under NS5's entrance normalization, which is "
                "why scalar and none are identical controls. P1 is consistent with "
                "the Procrustes result, but finite NS5 is not yet row-orthogonal, so "
                "an actual exact-SVD line search is still required."
            ),
        },
        {
            "id": "normmatch_result",
            "type": "markdown",
            "sourceId": "normmatch_effects",
            "body": (
                "## Norm matching changes heldout gaps only at the 1e-7 scale\n\n"
                "After forcing each structured-K direction to the none Frobenius "
                "norm, the mean eta=0.01 heldout change is −1.44e-7 for diag, "
                "−9.44e-8 for block4, and −1.39e-7 for full. This rejects final "
                "direction-norm inflation as the fixed-state explanation."
            ),
        },
        {"id": "normmatch_table_block", "type": "table", "tableId": "normmatch_table"},
        {
            "id": "svd_result",
            "type": "markdown",
            "sourceId": "exact_svd",
            "body": (
                "## Exact-SVD remains an unresolved intervention\n\n"
                "NS5–exact-polar cosine is low for every base mode: 0.343696 for "
                "none and 0.344–0.347 for the structured modes. The near equality "
                "does not identify a mode-specific approximation failure, but P1 "
                "never applies the exact-SVD direction to the loss. It therefore "
                "cannot rule finite-step NS in or out."
            ),
        },
        {"id": "svd_table_block", "type": "table", "tableId": "exact_svd_table"},
        {
            "id": "p0_result",
            "type": "markdown",
            "sourceId": "p0_p1",
            "body": (
                "## P1 invalidates the heldout claim from P0\n\n"
                "P0's scalar-none line-search discrepancy was 0.002485, while "
                "P1 reduces the control floor to 9.54e-7. The P0 heldout "
                "diag-minus-none gap of −0.001614 at eta=0.01 becomes +0.000004 "
                "under P1. Only the same-batch disadvantage reproduces."
            ),
        },
        {"id": "p0_table_block", "type": "table", "tableId": "p0p1_table"},
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## Scope and methodology\n\n"
                "One OpenWebText 24L/D1024 none trajectory was probed at step "
                "10000. Four 128-token build batches produced fresh K candidates "
                "for layers 0, 11, and 23; each build used eight fixed 128-token "
                "heldout batches. HVP and line search used strict FP32 with TF32 "
                "disabled. This is a fresh-K cross-fit experiment, not the training "
                "optimizer's EMA K plus momentum update."
            ),
        },
        {
            "id": "quality_result",
            "type": "markdown",
            "sourceId": "quality_checks",
            "body": (
                "## The scientific gate fails despite complete, clean data\n\n"
                "All 12 emitted probe checks and all 11 local checks pass. The "
                "archive contains all 108 direction rows, 4860 line-search rows, "
                "27 exact-SVD diagnostics, and eight complete W&B metric series. "
                "None-repeat is exact and the scalar control is below 1e-6."
            ),
        },
        {"id": "quality_table_block", "type": "table", "tableId": "quality_table"},
        {
            "id": "resource_result",
            "type": "markdown",
            "sourceId": "trajectory_resource",
            "body": (
                "## Trajectory and resource record\n\n"
                "The probe trajectory reaches validation loss 5.412502 at step "
                "10000, 0.002376 above the historical matched none export. Training "
                "uses 864 MiB K-state with zero c_proj K-state and peaks at "
                "6203.732 MiB. The isolated P1 probe takes 51.586 seconds."
            ),
        },
        {"id": "resource_table_block", "type": "table", "tableId": "resource_table"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## Limitations\n\n"
                "- Four build repeats are enough to reject a large stable P0-style "
                "heldout effect, but not to estimate tiny effects precisely. The "
                "same eight heldout batches are reused across builds, so the interval "
                "is conditional on that fixed evaluation set.\n"
                "- Fresh covariance uses 128 samples for 4096 features, so it is a "
                "poor proxy for the optimizer's accumulated full K.\n"
                "- Layers are perturbed separately; cross-layer simultaneous updates "
                "and momentum coupling are absent.\n"
                "- Exact-SVD directions are diagnosed but not line-searched."
            ),
        },
        {
            "id": "next_step",
            "type": "markdown",
            "body": (
                "## Decision and next experiment\n\n"
                "**Do not repeat unchanged P1 on seeds 2024 and 2025.** First build "
                "a seed2026 optimizer-faithful probe that compares fresh K with EMA "
                "K, momentum-only, and EMA-K-plus-momentum, while line-searching both "
                "NS5 and exact-SVD directions. Add seeds only if the revised four-build "
                "result is sign-consistent and clearly above the numerical control floor."
            ),
        },
        {
            "id": "open_questions",
            "type": "markdown",
            "body": (
                "## Questions left open\n\n"
                "- Does EMA diag K stabilize momentum across steps rather than "
                "improve an instantaneous fresh-gradient direction?\n"
                "- Does full K damage arise from off-diagonal state drift or from "
                "the interaction between that drift and momentum?\n"
                "- Does the long-run diag gain survive when exact-polar updates "
                "replace finite NS5?\n"
                "- Which temporal statistic predicts the known diag-versus-none "
                "and diag-versus-full training gaps?"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "c_proj Quadratic Probe P1 — Seed 2026",
            "description": (
                "Hierarchical strict-FP32 analysis of fixed-checkpoint c_proj K "
                "directions, norm matching, exact-polar diagnostics, and the P0 gate."
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
                "diag_effects": records(key_diag),
                "diag_build_chart": records(diag_build_chart),
                "direction_geometry": records(geometry_view),
                "normmatch_effects": records(normmatch_view),
                "exact_svd": records(exact_view),
                "p0_p1": records(p0p1_view),
                "quality_checks": records(quality),
                "trajectory_resource": records(resource_view),
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    artifact_path = run_dir / "quadratic_probe_p1_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(artifact_path)


if __name__ == "__main__":
    main()
