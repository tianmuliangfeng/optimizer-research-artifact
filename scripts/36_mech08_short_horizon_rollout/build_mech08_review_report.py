#!/usr/bin/env python3
"""Build the durable MECH-08 review report from independently audited summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = "2026-07-28.1"
GENERATED_AT = "2026-07-28T02:30:00Z"

CONTRAST_LABELS = {
    "selective_diag_vs_muon": "Selective-diag − Muon",
    "selective_none_vs_muon": "Selective-none − Muon",
    "selective_diag_vs_original_newton_muon": "Selective-diag − original NM",
    "selective_none_vs_original_newton_muon": "Selective-none − original NM",
    "original_newton_muon_vs_muon": "Original NM − Muon",
}
PRIMARY_TRAJECTORY_CONTRASTS = [
    "selective_diag_vs_muon",
    "selective_none_vs_muon",
    "selective_diag_vs_original_newton_muon",
    "selective_none_vs_original_newton_muon",
]
METRIC_LABELS = {
    "normalized_loss_auc": "AUC 0–128",
    "normalized_heldout_loss": "step 128",
}


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def source(
    source_id: str,
    label: str,
    path: str,
    sql: str,
    description: str,
    metric_definitions: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_source = {"id": source_id, "label": label, "path": path}
    query: dict[str, Any] = {
        "engine": "duckdb",
        "language": "sql",
        "sql": sql,
        "description": description,
        "executed_at": GENERATED_AT,
        "tables_used": [path],
    }
    if metric_definitions:
        query["metric_definitions"] = metric_definitions
    canonical_source = {
        "id": source_id,
        "label": label,
        "path": path,
        "query": query,
    }
    return manifest_source, canonical_source


def build_report(review_dir: Path) -> None:
    endpoints = pd.read_csv(review_dir / "primary_endpoints_overall.csv")
    by_origin = pd.read_csv(review_dir / "primary_endpoints_by_origin.csv")
    trajectory = pd.read_csv(review_dir / "trajectory_contrasts.csv")
    alignment = pd.read_csv(review_dir / "prediction_alignment_recomputed.csv")
    important = json.loads((review_dir / "important_results.json").read_text(encoding="utf-8"))
    audit = json.loads((review_dir / "data_quality_audit.json").read_text(encoding="utf-8"))

    endpoint_view = endpoints.copy()
    endpoint_view["contrast_label"] = endpoint_view["contrast"].map(CONTRAST_LABELS)
    endpoint_view["metric_label"] = endpoint_view["metric"].map(METRIC_LABELS)
    endpoint_view["ci_95"] = endpoint_view.apply(
        lambda row: (
            f"[{row['hierarchical_bootstrap_95_low']:+.6f}, "
            f"{row['hierarchical_bootstrap_95_high']:+.6f}]"
        ),
        axis=1,
    )
    endpoint_view = endpoint_view[
        [
            "contrast",
            "contrast_label",
            "metric",
            "metric_label",
            "mean_delta",
            "paired_units",
            "origins",
            "left_better_units",
            "left_better_origins",
            "ci_95",
        ]
    ]

    trajectory_view = trajectory[
        trajectory["contrast"].isin(PRIMARY_TRAJECTORY_CONTRASTS)
    ].copy()
    trajectory_view["contrast_label"] = trajectory_view["contrast"].map(CONTRAST_LABELS)
    trajectory_view = trajectory_view[
        [
            "contrast",
            "contrast_label",
            "optimizer_step",
            "mean_delta",
            "left_better_units",
            "left_better_origins",
        ]
    ].sort_values(["contrast", "optimizer_step"])

    origin_view = by_origin[
        (by_origin["contrast"].isin(PRIMARY_TRAJECTORY_CONTRASTS))
        & (by_origin["optimizer_step"].astype(str) == "128")
    ].copy()
    origin_view["contrast_label"] = origin_view["contrast"].map(CONTRAST_LABELS)
    origin_view["origin"] = origin_view["checkpoint_method"].map(
        {"muon": "Muon-origin", "newton_full": "Newton-full-origin"}
    )
    origin_view = (
        origin_view.groupby(["contrast", "contrast_label", "origin"], as_index=False)[
            "mean_delta"
        ]
        .mean()
        .sort_values(["contrast", "origin"])
    )

    alignment_view = alignment[alignment["contrast_scope"] == "primary"].copy()
    alignment_view["horizon"] = alignment_view.apply(
        lambda row: "AUC 0–128"
        if row["metric"] == "normalized_loss_auc"
        else f"step {row['optimizer_step']}",
        axis=1,
    )
    alignment_view = alignment_view[
        [
            "horizon",
            "origin_contrast_units",
            "pearson",
            "spearman",
            "sign_concordant_units",
            "sign_concordance",
        ]
    ]

    audit_summary = pd.DataFrame(
        [
            {
                "formal_jobs": important["integrity"]["formal_jobs"],
                "independent_checks": len(audit["checks"]),
                "failed_checks": len(audit["failed_checks"]),
                "matched_start_units": important["integrity"]["matched_start_units"],
                "max_step0_spread": important["integrity"]["max_step0_spread"],
            }
        ]
    )
    endpoint_headlines = pd.DataFrame(
        [
            {
                "diag_vs_original_better_units": int(
                    endpoints.loc[
                        (endpoints["contrast"] == "selective_diag_vs_original_newton_muon")
                        & (endpoints["optimizer_step"].astype(str) == "128"),
                        "left_better_units",
                    ].iloc[0]
                ),
                "none_vs_original_better_units": int(
                    endpoints.loc[
                        (endpoints["contrast"] == "selective_none_vs_original_newton_muon")
                        & (endpoints["optimizer_step"].astype(str) == "128"),
                        "left_better_units",
                    ].iloc[0]
                ),
            }
        ]
    )

    manifest_sources: list[dict[str, Any]] = []
    canonical_sources: list[dict[str, Any]] = []
    source_specs = [
        source(
            "audit_source",
            "Independent artifact audit",
            "data_quality_audit.json",
            "SELECT * FROM read_json_auto('data_quality_audit.json')",
            "Loads the independent manifest, status, invariance, transfer, and raw-row audit.",
        ),
        source(
            "important_source",
            "Independently recomputed MECH-08 summary",
            "important_results.json",
            "SELECT * FROM read_json_auto('important_results.json')",
            "Loads the compact integrity and scientific summary recomputed from formal worker outputs.",
        ),
        source(
            "endpoint_source",
            "Primary paired endpoints",
            "primary_endpoints_overall.csv",
            "SELECT * FROM read_csv_auto('primary_endpoints_overall.csv')",
            "Loads the five preregistered paired contrasts at AUC 0–128 and step 128.",
            {
                "mean_delta": (
                    "Mean paired normalized-heldout-loss difference, left algorithm minus "
                    "right algorithm; negative means the left algorithm is better."
                ),
                "hierarchical_bootstrap_95": (
                    "Descriptive 95% interval from hierarchical resampling of four checkpoint "
                    "origins and three rollout replicas per origin; not a training-seed CI."
                ),
            },
        ),
        source(
            "trajectory_source",
            "Paired contrast trajectories",
            "trajectory_contrasts.csv",
            "SELECT * FROM read_csv_auto('trajectory_contrasts.csv')",
            "Loads mean paired contrast deltas at the nine fixed evaluation steps.",
        ),
        source(
            "origin_source",
            "Primary endpoints by checkpoint origin",
            "primary_endpoints_by_origin.csv",
            "SELECT * FROM read_csv_auto('primary_endpoints_by_origin.csv')",
            "Loads endpoint deltas separately for early/late Muon and Newton-full checkpoints.",
        ),
        source(
            "alignment_source",
            "MECH-07 to MECH-08 prediction bridge",
            "prediction_alignment_recomputed.csv",
            "SELECT * FROM read_csv_auto('prediction_alignment_recomputed.csv')",
            "Loads descriptive origin-level correlations and sign concordance between MECH-07 predictions and MECH-08 outcomes.",
        ),
    ]
    for manifest_source, canonical_source in source_specs:
        manifest_sources.append(manifest_source)
        canonical_sources.append(canonical_source)

    cards = [
        {
            "id": "formal_jobs",
            "description": "Formal short-horizon workers included in the audit.",
            "dataset": "audit_summary",
            "sourceId": "audit_source",
            "metrics": [{"label": "Formal workers", "field": "formal_jobs", "format": "number"}],
        },
        {
            "id": "independent_checks",
            "description": "Independently executed artifact and row-level audit checks.",
            "dataset": "audit_summary",
            "sourceId": "audit_source",
            "metrics": [
                {"label": "Audit checks passed", "field": "independent_checks", "format": "number"}
            ],
        },
        {
            "id": "diag_vs_original_units",
            "description": "Step-128 paired units where Selective-diag beats original Newton–Muon.",
            "dataset": "endpoint_headlines",
            "sourceId": "endpoint_source",
            "metrics": [
                {
                    "label": "Diag better than original NM",
                    "field": "diag_vs_original_better_units",
                    "format": "number",
                    "unit": "/12 units",
                }
            ],
        },
        {
            "id": "none_vs_original_units",
            "description": "Step-128 paired units where Selective-none beats original Newton–Muon.",
            "dataset": "endpoint_headlines",
            "sourceId": "endpoint_source",
            "metrics": [
                {
                    "label": "None better than original NM",
                    "field": "none_vs_original_better_units",
                    "format": "number",
                    "unit": "/12 units",
                }
            ],
        },
    ]

    charts = [
        {
            "id": "endpoint_chart",
            "title": "Preregistered short-horizon endpoints",
            "subtitle": "Negative paired delta favors the algorithm named on the left; aggregate wins over Muon are not robust.",
            "type": "bar",
            "dataset": "endpoint_view",
            "sourceId": "endpoint_source",
            "encodings": {
                "x": {
                    "field": "contrast_label",
                    "type": "nominal",
                    "label": "Preregistered contrast",
                },
                "y": {
                    "field": "mean_delta",
                    "type": "quantitative",
                    "label": "Mean normalized-loss delta (left − right)",
                },
                "color": {
                    "field": "metric_label",
                    "type": "nominal",
                    "label": "Endpoint",
                },
                "tooltip": [
                    {
                        "field": "left_better_units",
                        "type": "quantitative",
                        "label": "Left better units",
                    },
                    {
                        "field": "paired_units",
                        "type": "quantitative",
                        "label": "Paired units",
                    },
                ],
            },
        },
        {
            "id": "trajectory_chart",
            "title": "Primary paired contrasts across 128 optimizer steps",
            "subtitle": "Selective-vs-original advantages emerge directly after the first recorded original-NM refresh at step 32.",
            "type": "line",
            "dataset": "trajectory_view",
            "sourceId": "trajectory_source",
            "encodings": {
                "x": {
                    "field": "optimizer_step",
                    "type": "quantitative",
                    "label": "Optimizer step",
                },
                "y": {
                    "field": "mean_delta",
                    "type": "quantitative",
                    "label": "Mean normalized-loss delta (left − right)",
                },
                "color": {
                    "field": "contrast_label",
                    "type": "nominal",
                    "label": "Contrast",
                },
                "tooltip": [
                    {
                        "field": "left_better_units",
                        "type": "quantitative",
                        "label": "Left better units",
                    }
                ],
            },
        },
    ]

    tables = [
        {
            "id": "endpoint_table",
            "title": "Exact endpoint estimates",
            "subtitle": "Intervals are descriptive because the 12 rollout units contain only four checkpoint origins, not 12 independent training seeds.",
            "dataset": "endpoint_view",
            "sourceId": "endpoint_source",
            "defaultSort": {"field": "contrast_label", "direction": "asc"},
            "columns": [
                {"field": "contrast_label", "label": "Contrast", "type": "text"},
                {"field": "metric_label", "label": "Endpoint", "type": "text"},
                {"field": "mean_delta", "label": "Mean delta", "format": "number"},
                {"field": "left_better_units", "label": "Left better", "format": "number"},
                {"field": "paired_units", "label": "Paired units", "format": "number"},
                {"field": "left_better_origins", "label": "Origins left better", "format": "number"},
                {"field": "ci_95", "label": "Descriptive 95% interval", "type": "text"},
            ],
        },
        {
            "id": "origin_table",
            "title": "Step-128 dependence on checkpoint trajectory",
            "subtitle": "Muon-origin and Newton-full-origin starts give materially different rankings against Muon.",
            "dataset": "origin_view",
            "sourceId": "origin_source",
            "defaultSort": {"field": "contrast_label", "direction": "asc"},
            "columns": [
                {"field": "contrast_label", "label": "Contrast", "type": "text"},
                {"field": "origin", "label": "Checkpoint origin", "type": "text"},
                {"field": "mean_delta", "label": "Mean step-128 delta", "format": "number"},
            ],
        },
        {
            "id": "alignment_table",
            "title": "MECH-07 prediction alignment with MECH-08",
            "subtitle": "The one-step diagnostic does not reliably predict AUC or endpoint sign.",
            "dataset": "alignment_view",
            "sourceId": "alignment_source",
            "defaultSort": {"field": "horizon", "direction": "asc"},
            "columns": [
                {"field": "horizon", "label": "Outcome", "type": "text"},
                {"field": "pearson", "label": "Pearson", "format": "number"},
                {"field": "spearman", "label": "Spearman", "format": "number"},
                {"field": "sign_concordance", "label": "Sign agreement", "format": "percent"},
                {
                    "field": "origin_contrast_units",
                    "label": "Origin × contrast units",
                    "format": "number",
                },
            ],
        },
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# MECH-08：LLaMA-1B 短时域因果 rollout 复核",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "important_source",
            "body": (
                "## 技术结论\n\n"
                "**MECH-08 的工程与数据完整性通过，但科学结论不是“Selective 全面胜过 Muon”。** "
                "在同一起点的 128-step rollout 中，Selective-diag 与 Selective-none 对原始 "
                "Newton–Muon 都表现出高度一致的优势；相对 Muon 的结果则受 checkpoint "
                "来源强烈影响，汇总后没有稳定的 Pareto 胜出。最值得继续验证的机制线索是："
                "Selective 相对原始 Newton–Muon 的优势在第一次记录到的 K refresh（step 32）"
                "之后立即出现。"
            ),
        },
        {
            "id": "audit_summary_text",
            "type": "markdown",
            "sourceId": "audit_source",
            "body": (
                "## 数据可信度\n\n"
                "独立审计覆盖 48 个 formal workers、432 条评估记录和 6,144 条训练记录；"
                "所有 1,077 项检查通过。12 个 origin×replica 起点严格配对，step 0 最大差异为 0。"
                "原始 worker CSV 对官方 analysis 的关键汇总可复现。"
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "formal_jobs",
                "independent_checks",
                "diag_vs_original_units",
                "none_vs_original_units",
            ],
        },
        {
            "id": "endpoint_finding",
            "type": "markdown",
            "sourceId": "endpoint_source",
            "body": (
                "## 主终点\n\n"
                "按预先确定的比较优先级，Selective-diag 与 Selective-none 在 step 128 和 "
                "AUC 0–128 上都以 12/12 paired units、4/4 checkpoint origins 胜过原始 "
                "Newton–Muon。对 Muon 而言，两种 Selective 在 AUC 上只有很小的平均优势，"
                "在 step 128 上平均反而更差，而且区间跨过 0 或接近 0。因此当前证据支持"
                "“修正原始 Newton–Muon 的 down-proj K 机制”，不支持“普遍优于 Muon”。"
            ),
        },
        {"id": "endpoint_chart_block", "type": "chart", "chartId": "endpoint_chart"},
        {"id": "endpoint_table_block", "type": "table", "tableId": "endpoint_table"},
        {
            "id": "refresh_finding",
            "type": "markdown",
            "sourceId": "trajectory_source",
            "body": (
                "## 时间定位与 refresh 线索\n\n"
                "原始 Newton–Muon 的训练记录在 optimizer steps 32、64、96、128 标记了 "
                "preconditioner refresh。Selective-diag 相对原始方法的平均差值从 step 32 "
                "的 −0.000199 变为 step 48 的 −0.002823；Selective-none 从 +0.000082 "
                "变为 −0.004575。两项变化都发生在全部 12 个 paired units 中。这是明确的"
                "时间共现证据，但还不是 refresh 的因果证明。"
            ),
        },
        {"id": "trajectory_chart_block", "type": "chart", "chartId": "trajectory_chart"},
        {
            "id": "origin_finding",
            "type": "markdown",
            "sourceId": "origin_source",
            "body": (
                "## checkpoint 路径依赖\n\n"
                "对 Muon 的排序取决于起点来自哪条训练轨迹：在 Muon-origin checkpoints 上，"
                "两种 Selective 的 step-128 结果均一致落后；在 Newton-full-origin checkpoints "
                "上，AUC 均一致优于 Muon，而 step-128 接近持平。MECH-08 因此揭示的是"
                "路径依赖边界，而不是一个与历史状态无关的算法总排序。"
            ),
        },
        {"id": "origin_table_block", "type": "table", "tableId": "origin_table"},
        {
            "id": "prediction_finding",
            "type": "markdown",
            "sourceId": "alignment_source",
            "body": (
                "## MECH-07 → MECH-08 预测桥\n\n"
                "MECH-07 的单步方向诊断不能可靠预测 MECH-08 的 AUC 或 128-step 符号。"
                "primary contrasts 的 AUC Spearman 为 −0.038、符号一致率为 50%；"
                "step 128 的 Spearman 升至 0.579，但符号一致率仍只有 56.25%。后续应把 "
                "MECH-07 定位为局部机制诊断，而不是 rollout 排名预测器。"
            ),
        },
        {"id": "alignment_table_block", "type": "table", "tableId": "alignment_table"},
        {
            "id": "scope_and_method",
            "type": "markdown",
            "body": (
                "## 范围、指标与方法\n\n"
                "- **单位：** 4 个 checkpoint origins（early/late × Muon/Newton-full）× "
                "3 个 rollout replicas，共 12 个严格 matched-start paired units。\n"
                "- **算法：** Muon、原始 Newton–Muon、Selective-diag、Selective-none。\n"
                "- **主指标：** normalized held-out loss；差值定义为 left − right，负值表示"
                "左侧算法更好。终点为 step 128 与梯形积分 AUC 0–128。\n"
                "- **审计方法：** 从 formal worker CSV 独立重算所有配对汇总，并逐项核对"
                "状态、清单、动量迁移、初始 preconditioner、checkpoint/参数不变性及数据粒度。\n"
                "- **不使用：** 本实验计时和显存数据不能作为论文效率证据。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 局限与稳健性边界\n\n"
                "- 三个 replica 是 rollout 数据重复，不是独立训练 seed；可确认的独立起点只有 4 个。\n"
                "- hierarchical bootstrap 区间仅作描述，不应表述为跨 seed 的统计显著性。\n"
                "- 128 steps 是短时域干预，不能替代完整训练曲线或等预算最终性能比较。\n"
                "- refresh 与优势出现的时间一致，但仍可能由 refresh 同期的其他状态演化造成。\n"
                "- diag 与 none 之间的差异不是本实验的首要科学问题，不应成为论文主叙事。"
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 建议的下一步\n\n"
                "启动一个**窄范围、预注册的 MECH-09 down-proj refresh-timing mediation**，"
                "而不是扩大无目标的 prediction-rescue sweep。复用 MECH-08 的 48 个控制工件，"
                "仅新增两种原始 Newton–Muon 干预：① down-proj full-K refresh 从 step 32 "
                "延迟到 step 64；② 128 steps 内冻结 down-proj full-K，其他 K family 保持"
                "生产配置。按 4 origins × 3 replicas × 2 干预，需要 24 个新 workers。若劣化"
                "起点随延迟从 32→48 移到 64→80，且冻结时显著减弱，才可把 refresh 从时间线索"
                "提升为机制证据。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 投稿前仍需回答的问题\n\n"
                "1. refresh mediation 是否在独立训练 seed 上复现，而不只是在四个 checkpoint origins 上？\n"
                "2. 路径依赖来自 momentum/history state、K refresh 本身，还是二者交互？\n"
                "3. 主训练实验中的 Selective-vs-Muon 优势是否与这里的短时域边界一致？\n"
                "4. 公平的 peak memory、tokens/s、steps/s 与等预算超参敏感性应由独立 benchmark 提供。"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "MECH-08：LLaMA-1B 短时域因果 rollout 复核",
            "description": "Independent artifact audit and decision-ready mechanism interpretation.",
            "generatedAt": GENERATED_AT,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": manifest_sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "audit_summary": records(audit_summary),
                "endpoint_headlines": records(endpoint_headlines),
                "endpoint_view": records(endpoint_view),
                "trajectory_view": records(trajectory_view),
                "origin_view": records(origin_view),
                "alignment_view": records(alignment_view),
            },
        },
        "sources": canonical_sources,
    }

    decision = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "run_id": important["run_id"],
        "decision": "targeted_mech09_refresh_timing_mediation",
        "reason": (
            "Selective-vs-original Newton–Muon advantages emerge in all 12 paired units "
            "immediately after the first recorded full-K refresh, while MECH-07 predictions "
            "do not reliably explain the rollout ranking."
        ),
        "reuse_existing_control_workers": 48,
        "new_interventions": [
            "delay_down_proj_full_k_refresh_from_step_32_to_step_64",
            "freeze_down_proj_full_k_for_128_steps",
        ],
        "new_worker_count": 24,
        "design_grain": "4 checkpoint origins x 3 rollout replicas x 2 interventions",
        "success_pattern": {
            "delayed_refresh": "degradation onset shifts from steps 32-48 to steps 64-80",
            "frozen_refresh": "post-step-32 degradation disappears or materially weakens",
        },
        "not_recommended": [
            "broad_prediction_rescue_sweep",
            "claiming_selective_universally_beats_muon",
            "using_mech08_runtime_or_memory_as_paper_efficiency_evidence",
        ],
    }

    markdown = """# MECH-08：LLaMA-1B 短时域因果 rollout 复核

## 一句话结论

MECH-08 工程与数据完整性通过。两种 Selective 方法都稳定修复了原始
Newton–Muon 在短时域内的劣化，但没有稳定击败 Muon；最强的新线索是优势在
原始 Newton–Muon 第一次 K refresh 之后立即出现。

## 关键证据

- 48 个 formal workers、432 条评估记录、6,144 条训练记录。
- 1,077/1,077 项独立检查通过；12 个 matched-start units 的 step-0 最大差异为 0。
- Selective-diag vs original Newton–Muon：step 128 与 AUC 均为 12/12 units、
  4/4 origins 胜出。
- Selective-none vs original Newton–Muon：step 128 与 AUC 均为 12/12 units、
  4/4 origins 胜出。
- 相对 Muon：两种 Selective 的 AUC 仅有微弱平均优势，step 128 平均更差；
  排序显著依赖 checkpoint 来源。
- original Newton–Muon 的第一次记录 refresh 在 step 32；到 step 48，
  diag/none 相对 original 的差值分别突变为 −0.002823/−0.004575，
  且全部 12 个 paired units 同方向。
- MECH-07 对 MECH-08 AUC 的 Spearman 为 −0.038、符号一致率 50%；
  step 128 分别为 0.579、56.25%。它应被视为局部诊断，而不是 rollout 排名预测器。

## 科学表述

当前最稳妥的主张是：Selective down-proj K 设计在 Newton–Muon 家族内部避免了
一次与 K refresh 时间对齐的短时域劣化。相对 Muon 的收益是路径/阶段依赖的，
不能写成普遍胜出。diag 与 none 的相互比较不是主问题。

## 建议

下一项机制实验应是窄范围 MECH-09 refresh-timing mediation：复用现有 48 个控制
workers，仅新增 down-proj full-K 延迟 refresh 与冻结 refresh 两个干预，共
4 origins × 3 replicas × 2 = 24 个新 workers。只有劣化时间随 refresh 延迟而移动、
且冻结时消失或显著减弱，才能把当前时间线索提升为因果机制。

## 局限

三个 replica 不是训练 seed；只有四个 checkpoint origins。128-step rollout 不能代替
完整训练。描述性 bootstrap 不能当作跨 seed 显著性。MECH-08 的运行时间与显存也不能
用于论文效率对比。
"""

    (review_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (review_dir / "MECH08_TECHNICAL_REPORT.md").write_text(markdown, encoding="utf-8")
    (review_dir / "mech09_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"artifact: {review_dir / 'artifact.json'}")
    print(f"markdown: {review_dir / 'MECH08_TECHNICAL_REPORT.md'}")
    print(f"decision: {review_dir / 'mech09_decision.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_report(args.review_dir.resolve())
