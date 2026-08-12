#!/usr/bin/env python
"""Build the canonical report artifact for one quadratic-probe P0 run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


MODE_ORDER = {"none": 0, "scalar": 1, "diag": 2, "block4": 3, "full": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.where(pd.notnull(frame), None)
    return clean.to_dict(orient="records")


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
    directions = pd.read_csv(processed / "direction_mode_aggregate.csv")
    line = pd.read_csv(processed / "line_search_mode_summary.csv")
    quality = pd.read_csv(processed / "data_quality_checks.csv")
    wandb = pd.read_csv(processed / "wandb_metric_summary.csv")

    directions["mode_order"] = directions["mode"].map(MODE_ORDER)
    directions = directions.sort_values("mode_order")
    direction_view = directions[
        [
            "mode",
            "alignment_normalized_mean",
            "quadratic_score_ratio_vs_none_mean",
            "direction_cos_vs_none_mean",
            "projector_drift_vs_none_mean",
            "direction_norm_ratio_vs_none_mean",
            "row_orthogonality_residual_mean",
            "mode_order",
        ]
    ].copy()

    line_view = line[line["eta"].isin([0.01, 0.02])].copy()
    line_view["mode_order"] = line_view["mode"].map(MODE_ORDER)
    line_view["split_order"] = line_view["eval_split"].map(
        {"same": 0, "heldout": 1}
    )
    line_view = line_view.sort_values(
        ["eta", "split_order", "mode_order"]
    )[
        [
            "eta",
            "eval_split",
            "mode",
            "mean_loss_delta",
            "mean_delta_vs_none",
            "better_than_none_layers",
            "mode_order",
            "split_order",
        ]
    ]
    chart_view = line_view[line_view["eta"] == 0.01].copy()

    quality_view = quality.copy()
    quality_view["status_order"] = quality_view["status"].map(
        {"FAIL": 0, "WARN": 1, "NOT_RUN": 2, "PASS": 3}
    )
    quality_view = quality_view.sort_values(["status_order", "check"])

    resource = pd.DataFrame(
        [
            {
                "val_loss_step10000": key["trajectory_reproduction"][
                    "val_loss_step10000"
                ],
                "historical_none_val_loss_step10000": key[
                    "trajectory_reproduction"
                ]["historical_none_seed2026_val_loss_step10000"],
                "trajectory_delta": key["trajectory_reproduction"][
                    "delta_vs_historical"
                ],
                "training_elapsed_seconds": key["resource"][
                    "training_elapsed_seconds_step10000"
                ],
                "training_peak_memory_mib": key["resource"][
                    "training_peak_memory_mib"
                ],
                "total_k_state_mib": key["resource"]["total_k_state_mib"],
                "cproj_k_state_mib": key["resource"]["cproj_k_state_mib"],
                "probe_seconds": key["resource"]["probe_seconds"],
            }
        ]
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    sources = [
        source(
            "direction_geometry",
            "Derived direction geometry by mode",
            "processed/direction_mode_aggregate.csv",
            "SELECT * FROM direction_mode_aggregate ORDER BY mode",
            "Three-layer descriptive means computed from the 15 exact-HVP direction rows.",
            generated_at,
        ),
        source(
            "fixed_eta_line_search",
            "Fixed-step same/heldout line-search summary",
            "processed/line_search_mode_summary.csv",
            "SELECT * FROM line_search_mode_summary WHERE eta IN (0.01, 0.02)",
            "Three-layer descriptive means for the actual and doubled matrix learning rate.",
            generated_at,
        ),
        source(
            "quality_checks",
            "Quadratic-probe data quality checks",
            "processed/data_quality_checks.csv",
            "SELECT check, status, severity_if_failed, evidence FROM data_quality_checks",
            "Coverage, finiteness, W&B integrity, numerical-control, sampling, and exact-SVD checks.",
            generated_at,
        ),
        source(
            "trajectory_resource",
            "Training trajectory and resource summary",
            "processed/key_results.json",
            "SELECT * FROM trajectory_resource",
            "Step-10000 reproduction, elapsed time, peak memory, K-state, and probe runtime.",
            generated_at,
        ),
        source(
            "wandb_metrics",
            "Archived W&B core metric summary",
            "processed/wandb_metric_summary.csv",
            "SELECT * FROM wandb_metric_summary ORDER BY metric",
            "Coverage and endpoints for the eight paper-profile W&B metrics.",
            generated_at,
        ),
    ]

    tables = [
        {
            "id": "direction_table",
            "title": "Direction geometry by K mode",
            "subtitle": "Three probed layers at step 10000; ratios are relative to none.",
            "dataset": "direction_geometry",
            "sourceId": "direction_geometry",
            "defaultSort": {"field": "mode", "direction": "asc"},
            "columns": [
                {"field": "mode", "label": "Mode", "type": "text"},
                {
                    "field": "alignment_normalized_mean",
                    "label": "Normalized alignment",
                    "format": "number",
                },
                {
                    "field": "quadratic_score_ratio_vs_none_mean",
                    "label": "A^2/C ratio vs none",
                    "format": "number",
                },
                {
                    "field": "direction_cos_vs_none_mean",
                    "label": "Direction cosine vs none",
                    "format": "number",
                },
                {
                    "field": "projector_drift_vs_none_mean",
                    "label": "Projector drift",
                    "format": "number",
                },
                {
                    "field": "direction_norm_ratio_vs_none_mean",
                    "label": "Direction norm ratio",
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
            "id": "line_table",
            "title": "Fixed-step line-search results",
            "subtitle": "Mean over layers 0, 11, and 23; negative loss delta is better.",
            "dataset": "fixed_eta_line_search",
            "sourceId": "fixed_eta_line_search",
            "defaultSort": {"field": "eta", "direction": "asc"},
            "columns": [
                {"field": "eta", "label": "Eta", "format": "number"},
                {"field": "eval_split", "label": "Evaluation batch", "type": "text"},
                {"field": "mode", "label": "Mode", "type": "text"},
                {
                    "field": "mean_loss_delta",
                    "label": "Mean loss delta",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "mean_delta_vs_none",
                    "label": "Delta vs none",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "better_than_none_layers",
                    "label": "Layers better than none",
                    "type": "number",
                },
            ],
        },
        {
            "id": "quality_table",
            "title": "Data-quality and robustness checks",
            "subtitle": "Warnings bound interpretation; they are not missing source rows.",
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
            "title": "Trajectory reproduction and resource use",
            "subtitle": "The probe run reproduces the historical none trajectory at step 10000.",
            "dataset": "trajectory_resource",
            "sourceId": "trajectory_resource",
            "defaultSort": {
                "field": "val_loss_step10000",
                "direction": "asc",
            },
            "columns": [
                {
                    "field": "val_loss_step10000",
                    "label": "Probe-run val loss",
                    "format": "number",
                },
                {
                    "field": "historical_none_val_loss_step10000",
                    "label": "Historical none val loss",
                    "format": "number",
                },
                {
                    "field": "trajectory_delta",
                    "label": "Trajectory delta",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "training_elapsed_seconds",
                    "label": "Elapsed seconds",
                    "format": "number",
                },
                {
                    "field": "training_peak_memory_mib",
                    "label": "Peak memory MiB",
                    "format": "number",
                },
                {
                    "field": "total_k_state_mib",
                    "label": "Total K-state MiB",
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
            ],
        },
    ]

    charts = [
        {
            "id": "fixed_eta_chart",
            "title": "Mean loss delta at eta=0.01",
            "subtitle": (
                "Layers 0, 11, and 23; same build batch versus one held-out "
                "128-token batch; negative is better."
            ),
            "type": "bar",
            "dataset": "fixed_eta_chart",
            "sourceId": "fixed_eta_line_search",
            "valueFormat": "number",
            "encodings": {
                "x": {
                    "field": "mode",
                    "type": "nominal",
                    "label": "c_proj K mode",
                },
                "y": {
                    "field": "mean_loss_delta",
                    "type": "quantitative",
                    "label": "Three-layer mean loss delta",
                },
                "color": {
                    "field": "eval_split",
                    "type": "nominal",
                    "label": "Evaluation batch",
                },
                "tooltip": [
                    {
                        "field": "mean_loss_delta",
                        "type": "quantitative",
                        "label": "Mean loss delta",
                        "format": "number",
                    },
                    {
                        "field": "mean_delta_vs_none",
                        "type": "quantitative",
                        "label": "Delta vs none",
                        "format": "number",
                    },
                    {
                        "field": "better_than_none_layers",
                        "type": "quantitative",
                        "label": "Layers better than none",
                    },
                ],
            },
        }
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# c_proj Quadratic Probe P0 — Seed 2026",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## Technical summary\n\n"
                "**The probe narrows the mechanism but does not yet justify a multi-seed claim.** "
                "At the actual matrix learning rate, none gives the strongest same-batch "
                "descent in all three layers, while diag gives the best mean result on the "
                "single held-out batch. Block4/full often improve the unconstrained "
                "quadratic score, so A^2/C alone cannot explain their long-run damage. "
                "Finite five-step Newton–Schulz also changes update norms by up to about "
                "3x, making an exact-SVD or norm-matched control necessary."
            ),
        },
        {
            "id": "fixed_eta_result",
            "type": "markdown",
            "sourceId": "fixed_eta_line_search",
            "body": (
                "## Diag trades same-batch descent for a held-out advantage\n\n"
                "At eta=0.01, diag is worse than none by +0.001759 on the same-batch "
                "three-layer mean but better by -0.001614 on the held-out mean. At "
                "eta=0.02, the corresponding differences are +0.003860 and -0.001562. "
                "This is consistent with a cross-batch stability hypothesis, but one "
                "128-token held-out batch is not an uncertainty estimate."
            ),
        },
        {
            "id": "fixed_eta_visual",
            "type": "chart",
            "chartId": "fixed_eta_chart",
        },
        {"id": "line_evidence", "type": "table", "tableId": "line_table"},
        {
            "id": "geometry_result",
            "type": "markdown",
            "sourceId": "direction_geometry",
            "body": (
                "## Unconstrained quadratic score and finite-NS geometry point in different directions\n\n"
                "Block4 and full average about 1.56x the none A^2/C score, yet they do "
                "not win the fixed-step held-out mean. Their mean direction norms are "
                "1.82x and 3.04x the none norm, and their right-projector drift is much "
                "larger than diag. The optimizer therefore changes both orientation and "
                "effective step magnitude; it is not applying interchangeable exact polar directions."
            ),
        },
        {
            "id": "direction_evidence",
            "type": "table",
            "tableId": "direction_table",
        },
        {
            "id": "scope_methods",
            "type": "markdown",
            "body": (
                "## Scope and measurement definition\n\n"
                "One none trajectory (OpenWebText 24L/D1024, seed2026) was trained to "
                "step 10000. Instantaneous none, scalar, diag, block4, and full c_proj "
                "directions were evaluated at layers 0, 11, and 23 using exact HVPs. "
                "The probe used one 128-token build batch, one 128-token held-out batch, "
                "fresh K estimates, NS5, and eta in {0, 0.0025, 0.005, 0.01, 0.02}. "
                "Each layer was perturbed separately and restored."
            ),
        },
        {
            "id": "trajectory_result",
            "type": "markdown",
            "sourceId": "trajectory_resource",
            "body": (
                "## The main trajectory is reproduced and the probe remains isolated\n\n"
                "The step-10000 validation loss is 5.410046 versus 5.410126 for the "
                "historical matched none/seed2026 run, a difference of -0.000080. "
                "The main run retains 864 MiB total K-state, zero c_proj K-state, and "
                "peaks at 6203.736 MiB. The isolated probe itself takes 2.469 seconds."
            ),
        },
        {
            "id": "resource_evidence",
            "type": "table",
            "tableId": "resource_table",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "quality_checks",
            "body": (
                "## Three warnings prevent a seed-level conclusion\n\n"
                "Scalar and none are direction-equivalent, yet their nonzero-step "
                "line-search deltas differ by as much as 0.002485, setting a practical "
                "numerical-resolution warning. The build and held-out base losses differ "
                "by 0.833020, showing large single-batch variation. Fresh K uses only "
                "128 samples for 4096 features (N/d=0.03125), and exact SVD was not run. "
                "The probe also omits the optimizer's EMA K state and momentum buffer."
            ),
        },
        {
            "id": "quality_evidence",
            "type": "table",
            "tableId": "quality_table",
        },
        {
            "id": "next_step",
            "type": "markdown",
            "body": (
                "## Do not copy the unchanged P0 probe to seeds 2024 and 2025 yet\n\n"
                "First rerun seed2026 with multiple fixed held-out batches, float32 or "
                "repeatability controls, and exact-SVD or norm-matched directions. "
                "The revised probe should also use shadow EMA K and the real momentum "
                "buffer, or explicitly remain a fresh-K cross-fit experiment. If the "
                "diag held-out advantage exceeds the duplicate-direction noise floor "
                "and survives these controls, then seeds 2024 and 2025 are warranted "
                "for the paper's central mechanism claim."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Questions left open\n\n"
                "- Does diag improve expected loss across many independent batches?\n"
                "- Does full/block4 fail because off-diagonal K is under-sampled, because "
                "NS5 changes effective update magnitude, or because EMA momentum and K "
                "define a drifting coordinate system?\n"
                "- After norm matching or exact polar projection, does the observed "
                "ordering persist?"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "c_proj Quadratic Probe P0 — Seed 2026",
            "description": (
                "Exact-HVP, fixed-step, same/heldout analysis of five c_proj K modes "
                "at one step-10000 none trajectory."
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
                "direction_geometry": records(direction_view),
                "fixed_eta_line_search": records(line_view),
                "fixed_eta_chart": records(chart_view),
                "quality_checks": records(quality_view),
                "trajectory_resource": records(resource),
                "wandb_metric_summary": records(wandb),
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    (run_dir / "quadratic_probe_p0_artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    notes = {
        "audience": "technical",
        "required_structure_mapping": {
            "title": "title",
            "technical_summary": "technical_summary",
            "key_findings": [
                "fixed_eta_result",
                "geometry_result",
                "trajectory_result",
            ],
            "scope_and_definitions": "scope_methods",
            "methodology": "scope_methods",
            "limitations_and_robustness": "limitations",
            "recommended_next_steps": "next_step",
            "further_questions": "further_questions",
        },
        "chart_map": [
            {
                "section": "fixed_eta_result",
                "question": (
                    "How do the five candidate directions compare on the build batch "
                    "and one held-out batch at the actual matrix learning rate?"
                ),
                "family": "Comparison and ranking",
                "chart_type": "grouped bar",
                "fields": [
                    "mode",
                    "eval_split",
                    "mean_loss_delta",
                    "mean_delta_vs_none",
                    "better_than_none_layers",
                ],
                "supported_takeaway": (
                    "None leads on the build batch while diag has the best held-out mean."
                ),
                "palette_policy": "hard two-root cap: blue and gold plus neutrals",
                "delivery": "quadratic_probe_p0_report.html",
            }
        ],
        "validation_rating": "Share with caveats",
    }
    (processed / "report_source_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(run_dir / "quadratic_probe_p0_artifact.json")


if __name__ == "__main__":
    main()
