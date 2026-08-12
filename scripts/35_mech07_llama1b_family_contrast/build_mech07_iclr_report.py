#!/usr/bin/env python3
"""Build the canonical artifact input for the MECH-07 ICLR decision report."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-27.1"
PRIMARY_LABELS = {
    "selective_diag_vs_muon": "Selective-diag − Muon",
    "selective_none_vs_muon": "Selective-none − Muon",
    "selective_diag_vs_original_newton_muon": (
        "Selective-diag − original Newton–Muon"
    ),
    "selective_none_vs_original_newton_muon": (
        "Selective-none − original Newton–Muon"
    ),
    "original_newton_muon_vs_muon": "original Newton–Muon − Muon",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--primary-analysis-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def bool_text(value: str) -> bool:
    return value.lower() == "true"


def stage_chart_rows(stage_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    order = 0
    for stage in ("early", "late"):
        for contrast in PRIMARY_LABELS:
            if contrast == "original_newton_muon_vs_muon":
                continue
            match = [
                row
                for row in stage_rows
                if row["checkpoint_stage"] == stage
                and row["contrast"] == contrast
            ]
            if len(match) != 1:
                raise RuntimeError(f"missing stage contrast: {stage} {contrast}")
            row = match[0]
            order += 1
            rows.append(
                {
                    "order": order,
                    "stage": "Early (step 1,000)" if stage == "early" else "Late (step 6,200)",
                    "contrast": PRIMARY_LABELS[contrast],
                    "contrast_short": PRIMARY_LABELS[contrast].replace(
                        "original Newton–Muon", "original NM"
                    ),
                    "median_delta_x1e4": float(row["median_delta"]) * 1e4,
                    "mean_delta_x1e4": float(row["mean_delta"]) * 1e4,
                    "left_better_cells": int(row["negative_cells_left_better"]),
                    "cells": int(row["cells"]),
                    "checkpoint_states_left_better": int(
                        row["checkpoint_states_left_better"]
                    ),
                    "checkpoint_states_total": int(row["checkpoint_states_total"]),
                }
            )
    return rows


def early_scope_rows(scope_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    scope_labels = {
        "family_core": "Family core (q/o/gate)",
        "down": "Down projection",
        "all": "All targets",
    }
    for scope in ("family_core", "down", "all"):
        for contrast in PRIMARY_LABELS:
            if contrast == "original_newton_muon_vs_muon":
                continue
            match = [
                row
                for row in scope_rows
                if row["checkpoint_stage"] == "early"
                and row["scope"] == scope
                and row["contrast"] == contrast
            ]
            if len(match) != 1:
                raise RuntimeError(f"missing early scope contrast: {scope} {contrast}")
            row = match[0]
            rows.append(
                {
                    "scope": scope_labels[scope],
                    "contrast": PRIMARY_LABELS[contrast],
                    "median_delta_x1e4": float(row["median_delta"]) * 1e4,
                    "mean_delta_x1e4": float(row["mean_delta"]) * 1e4,
                    "left_better_cells": int(row["negative_cells_left_better"]),
                    "left_worse_cells": int(row["positive_cells_left_worse"]),
                    "near_zero_cells": int(row["near_zero_cells"]),
                    "cells": int(row["cells"]),
                }
            )
    return rows


def long_run_rows(primary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for row in primary_rows:
        if row["family"] != "llama1b":
            continue
        rows.append(
            {
                "priority": row["priority"],
                "contrast": PRIMARY_LABELS[row["contrast"]],
                "final_delta_mean": float(row["final_delta_mean"]),
                "final_delta_sd": float(row["final_delta_sd"]),
                "left_better_seeds": int(row["negative_seeds_left_better"]),
                "left_worse_seeds": int(row["positive_seeds_left_worse"]),
                "seeds": int(row["seeds"]),
                "classification": row["classification"],
            }
        )
    if len(rows) != 5:
        raise RuntimeError(f"expected five LLaMA-1B long-run contrasts, got {len(rows)}")
    return rows


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    primary = args.primary_analysis_dir.resolve()
    analysis = run / "analysis"
    local_audit = analysis / "local_audit"
    key_results = read_json(local_audit / "key_results.json")
    checks = read_json(local_audit / "checks.json")
    if not key_results.get("passed") or not checks or not all(checks.values()):
        raise RuntimeError("local audit must pass before report generation")

    stage_rows = read_csv(analysis / "stage_contrast_summary.csv")
    scope_rows = read_csv(local_audit / "scope_contrast_summary.csv")
    primary_rows = read_csv(primary / "primary_contrasts_summary.csv")
    chart_rows = stage_chart_rows(stage_rows)
    scope_table_rows = early_scope_rows(scope_rows)
    long_rows = long_run_rows(primary_rows)

    output = analysis / "iclr_decision_report"
    output.mkdir(parents=False, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    query_dir = output / "queries"
    query_dir.mkdir(exist_ok=True)
    queries = {
        "mech07_stage.sql": """-- Widget source: MECH-07 official stage contrast summary.
SELECT
  checkpoint_stage,
  contrast,
  median_delta * 10000 AS median_delta_x1e4,
  mean_delta * 10000 AS mean_delta_x1e4,
  negative_cells_left_better AS left_better_cells,
  cells,
  checkpoint_states_left_better,
  checkpoint_states_total
FROM read_csv_auto('../../stage_contrast_summary.csv', header = true)
WHERE priority = 'primary'
ORDER BY checkpoint_stage, contrast;
""",
        "mech07_scope.sql": """-- Widget source: independently recomputed MECH-07 scope contrasts.
SELECT
  checkpoint_stage,
  scope,
  contrast,
  median_delta * 10000 AS median_delta_x1e4,
  mean_delta * 10000 AS mean_delta_x1e4,
  negative_cells_left_better AS left_better_cells,
  positive_cells_left_worse AS left_worse_cells,
  near_zero_cells,
  cells
FROM read_csv_auto('../../local_audit/scope_contrast_summary.csv', header = true)
WHERE checkpoint_stage = 'early' AND priority = 'primary'
ORDER BY scope, contrast;
""",
        "long_run.sql": """-- Widget source: unified three-seed LLaMA-1B primary comparison.
SELECT
  priority,
  contrast,
  final_delta_mean,
  final_delta_sd,
  negative_seeds_left_better AS left_better_seeds,
  positive_seeds_left_worse AS left_worse_seeds,
  seeds,
  classification
FROM read_csv_auto(
  'runs/34_selective_primary_comparison/20260727T083000+0000/primary_contrasts_summary.csv',
  header = true
)
WHERE family = 'llama1b'
ORDER BY priority DESC, contrast;
""",
    }
    for name, sql in queries.items():
        (query_dir / name).write_text(sql, encoding="utf-8")

    source_paths = {
        "mech07_stage": (
            "runs/"
            "35_mech07_llama1b_family_contrast/20260727T083446+0000/"
            "analysis/iclr_decision_report/queries/mech07_stage.sql"
        ),
        "mech07_audit": (
            "runs/"
            "35_mech07_llama1b_family_contrast/20260727T083446+0000/"
            "analysis/iclr_decision_report/queries/mech07_scope.sql"
        ),
        "long_run": (
            "runs/"
            "35_mech07_llama1b_family_contrast/20260727T083446+0000/"
            "analysis/iclr_decision_report/queries/long_run.sql"
        ),
    }

    manifest_sources = [
        {
            "id": "mech07_stage",
            "label": "MECH-07 official stage contrast summary",
            "path": source_paths["mech07_stage"],
        },
        {
            "id": "mech07_audit",
            "label": "MECH-07 independent scope recomputation",
            "path": source_paths["mech07_audit"],
        },
        {
            "id": "long_run",
            "label": "Unified three-seed primary comparison",
            "path": source_paths["long_run"],
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "MECH-07：早期机制成立，晚期机制仍缺一座桥",
            "description": (
                "LLaMA-1B family contrast 的独立审计、机制边界与 ICLR 决策建议。"
            ),
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "stage_primary_medians",
                    "title": "早期信号清晰，晚期局部差异坍缩",
                    "subtitle": (
                        "中位相对 shadow-loss 差值 ×10⁴；负值表示左侧算法更优。"
                    ),
                    "headerMarkdown": (
                        "四个主对比均分别以 Muon 或原始 Newton–Muon 为基线；"
                        "`diag vs none` 不属于主对比。"
                    ),
                    "type": "bar",
                    "dataset": "stage_primary_contrasts",
                    "sourceId": "mech07_stage",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {
                            "field": "contrast_short",
                            "type": "nominal",
                            "label": "主对比",
                        },
                        "y": {
                            "field": "median_delta_x1e4",
                            "type": "quantitative",
                            "label": "Median relative shadow-loss Δ (×10⁴)",
                        },
                        "color": {
                            "field": "stage",
                            "type": "nominal",
                            "label": "Checkpoint stage",
                        },
                        "tooltip": [
                            {
                                "field": "median_delta_x1e4",
                                "type": "quantitative",
                                "label": "Median Δ ×10⁴",
                            },
                            {
                                "field": "left_better_cells",
                                "type": "quantitative",
                                "label": "Left-better cells",
                            },
                            {
                                "field": "cells",
                                "type": "quantitative",
                                "label": "Cells",
                            },
                        ],
                    },
                }
            ],
            "tables": [
                {
                    "id": "stage_exact_table",
                    "title": "主对比的精确 stage-level 结果",
                    "subtitle": (
                        "每个 stage 含 4 checkpoint origins × 4 repeats × 2 directions。"
                    ),
                    "dataset": "stage_primary_contrasts",
                    "sourceId": "mech07_stage",
                    "columns": [
                        {"field": "stage", "label": "Stage", "type": "text"},
                        {"field": "contrast", "label": "Contrast", "type": "text"},
                        {
                            "field": "median_delta_x1e4",
                            "label": "Median Δ ×10⁴",
                            "format": "number",
                        },
                        {
                            "field": "mean_delta_x1e4",
                            "label": "Mean Δ ×10⁴",
                            "format": "number",
                        },
                        {
                            "field": "left_better_cells",
                            "label": "Left-better cells",
                            "format": "number",
                        },
                        {
                            "field": "cells",
                            "label": "Total cells",
                            "format": "number",
                        },
                        {
                            "field": "checkpoint_states_left_better",
                            "label": "Stable origins",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "scope_decomposition_table",
                    "title": "早期机制分解：family core 与 down projection",
                    "subtitle": (
                        "同一批次与 checkpoint momentum 下的 matched local counterfactual。"
                    ),
                    "dataset": "early_scope_decomposition",
                    "sourceId": "mech07_audit",
                    "columns": [
                        {"field": "scope", "label": "Scope", "type": "text"},
                        {"field": "contrast", "label": "Contrast", "type": "text"},
                        {
                            "field": "median_delta_x1e4",
                            "label": "Median Δ ×10⁴",
                            "format": "number",
                        },
                        {
                            "field": "left_better_cells",
                            "label": "Left-better",
                            "format": "number",
                        },
                        {
                            "field": "left_worse_cells",
                            "label": "Left-worse",
                            "format": "number",
                        },
                        {"field": "cells", "label": "Cells", "format": "number"},
                    ],
                },
                {
                    "id": "long_run_table",
                    "title": "三 seed 长程结果：局部机制不能替代最终排名",
                    "subtitle": (
                        "LLaMA-1B paired final validation-loss delta；负值表示左侧方法更优。"
                    ),
                    "dataset": "llama1b_long_run",
                    "sourceId": "long_run",
                    "columns": [
                        {"field": "priority", "label": "Role", "type": "text"},
                        {"field": "contrast", "label": "Contrast", "type": "text"},
                        {
                            "field": "final_delta_mean",
                            "label": "Final Δ mean",
                            "format": "number",
                        },
                        {
                            "field": "final_delta_sd",
                            "label": "SD",
                            "format": "number",
                        },
                        {
                            "field": "left_better_seeds",
                            "label": "Left-better seeds",
                            "format": "number",
                        },
                        {"field": "seeds", "label": "Seeds", "format": "number"},
                    ],
                },
            ],
            "sources": manifest_sources,
            "blocks": [
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## 技术摘要\n\n"
                        "**MECH-07 数据可信、早期机制信号明确，但没有解释晚期最终差距。** "
                        "8 个正式 checkpoint cells 与 8 个 smoke cells 全部通过；"
                        "独立重算覆盖 960 个 scope-level contrast observations，"
                        "官方 stage 汇总逐项复现。\n\n"
                        "最强结果是 early Selective-none 相对原始 Newton–Muon："
                        "all-scope 中位差为 **−1.0664×10⁻³**，32/32 local cells "
                        "均支持 Selective-none；但它相对 Muon 的中位差仍为 "
                        "**+1.2170×10⁻³**，0/32 cells 支持 Selective-none。"
                        "这说明“删除 down 的 K-state”修复了原始 Newton–Muon 的一部分"
                        "早期损害，却没有消除 family-core dense preconditioning 的损害。\n\n"
                        "Late 四个主对比的绝对中位差最多只有 **4.0711×10⁻⁵**，"
                        "且未跨四个 checkpoint origins 稳定。这个负结果必须如实保留："
                        "当前机制链条只解释早期局部更新，不解释三 seed 长程训练中的晚期 "
                        "Muon 优势。"
                    ),
                },
                {"id": "key_chart", "type": "chart", "chartId": "stage_primary_medians"},
                {
                    "id": "scope_and_metrics",
                    "type": "markdown",
                    "body": (
                        "## 范围、数据与指标定义\n\n"
                        "- **Population**：LLaMA-1B seed2026 的 early step 1,000 与 "
                        "late step 6,200；每个 stage 含 Muon、原始 Newton–Muon、"
                        "Selective-diag、Selective-none 四个 checkpoint origins。\n"
                        "- **主对比**：两个 Selective 方案分别对 Muon 与原始 "
                        "Newton–Muon；`diag vs none` 被明确排除。\n"
                        "- **局部指标**：best relative shadow-loss delta 的 paired "
                        "difference（left − right）；负值表示左侧算法更优。\n"
                        "- **单个 stage 样本结构**：4 origins × 4 repeats × 2 "
                        "cross-fit directions = 32 local cells/contrast。\n"
                        "- **Scope**：family core = q/o/gate projections；down = "
                        "down projection；all = 两者合并。"
                    ),
                },
                {"id": "stage_table", "type": "table", "tableId": "stage_exact_table"},
                {
                    "id": "mechanism",
                    "type": "markdown",
                    "body": (
                        "## 方法与机制验证\n\n"
                        "MECH-07 在每个 checkpoint origin 上复用相同历史 momentum，"
                        "用独立 build split 构造新 covariance，再在 held-out split 上"
                        "评估四种候选算法的局部更新；模型参数、checkpoint 文件、"
                        "optimizer/loader 状态均经不变性审计。所有 formal cells 共享"
                        "同一个 batch contract，窗口互不重叠；Muon 两个正式 checkpoint "
                        "共审计 252 个 matrix-optimizer state entries，状态键只有 "
                        "`momentum`，没有 AdamW 一阶/二阶矩状态。\n\n"
                        "早期分解给出一个可解释的链条：dense family-core 相对 Muon "
                        "整体更差；down 上 Selective-none 与 Muon 在构造上相同，而原始 "
                        "Newton–Muon 的 dense down 更新更差。因此 Selective-none 能稳定"
                        "修复原始方法，却仍因保留 family-core dense preconditioning "
                        "而落后 Muon。Selective-diag 的修复更弱且不完全一致。"
                    ),
                },
                {
                    "id": "scope_table",
                    "type": "table",
                    "tableId": "scope_decomposition_table",
                },
                {
                    "id": "long_run_context",
                    "type": "markdown",
                    "body": (
                        "## 与长程结果对齐\n\n"
                        "三 seed 长程主分析显示：LLaMA-1B 的两个 Selective 方案都优于"
                        "原始 Newton–Muon，但最终仍落后 Muon。MECH-07 的 early 结果"
                        "与“Selective 修复原始方法的一部分损害”一致；late 局部结果"
                        "却无法重现最终排名。这更像是**路径积累/状态演化问题**，或当前"
                        "一步 shadow metric 在晚期分辨率不足，而不是已完成的全程机制解释。"
                    ),
                },
                {"id": "long_run", "type": "table", "tableId": "long_run_table"},
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 局限、稳健性与可写边界\n\n"
                        "- 机制实验目前只有 seed2026；cross-fit repeats 与 checkpoint "
                        "origins 提供的是局部稳健性，不等同于独立训练 seed 泛化。\n"
                        "- shadow loss 是一步/局部 counterfactual，不能自动外推到 "
                        "6,200-step 训练轨迹。\n"
                        "- Late 的 near-zero 结果不是“方法打平”的长程证据；三 seed "
                        "最终 loss 明确显示 Muon 更优。\n"
                        "- 可以写：**Selective 的状态删除避免了原始 Newton–Muon 的"
                        "部分早期有害预条件化。**\n"
                        "- 不能写：MECH-07 已解释 late reversal、得到普适 K 选择定律，"
                        "或局部 shadow ranking 已证明最终因果排名。"
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 面向 ICLR 的下一步\n\n"
                        "**最高优先级不是再堆一个只读 endpoint，而是补上局部预测到真实"
                        "优化轨迹之间的因果桥。** 建议预注册一个 matched-start "
                        "short-horizon rollout：从 early/late 的共同 checkpoint origin "
                        "克隆完整模型、loader 与 RNG 状态，分别应用 Muon、原始 "
                        "Newton–Muon、Selective-diag、Selective-none；固定同一 token "
                        "序列，运行 128 steps，每 16 steps 评估一次 held-out loss。\n\n"
                        "建议正式矩阵为 2 stages × 2 origins（Muon / original "
                        "Newton–Muon）× 4 applied algorithms × 3 data-order replicas，"
                        "共 48 个短跑、6,144 optimizer steps，约等于一条 6,200-step "
                        "长跑的计算量。主指标预注册为 held-out loss AUC 与 step-128 "
                        "paired delta；次指标为 MECH-07 一步预测与实际 16/32/64/128-step "
                        "收益的相关性。仍然只做四个主对比，不把 `diag vs none` 升格。\n\n"
                        "若 short rollout 能复现 early 排序并揭示差异何时消失，再只对"
                        "决定性 stage 做 seed2024/2025 复现；若不能，就停止扩张机制实验，"
                        "把 MECH-07 写成“强 early decomposition + 诚实 late negative "
                        "result”，并把资源转向公平调参和真实运行时/峰值显存基准。"
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 仍需回答的问题\n\n"
                        "1. early 的局部优势能否在 16–128 个真实 optimizer steps 中累积？\n"
                        "2. family-core penalty 是随 covariance、momentum 还是参数谱的"
                        "哪一项状态演化而消失？\n"
                        "3. late 局部 near-zero 是真实收敛，还是 shadow metric 的"
                        "分辨率/步长网格不足？\n"
                        "4. 在同等调参预算下，Selective 的质量—状态—吞吐折中是否仍"
                        "优于原始 Newton–Muon，并在哪些架构上优于 Muon？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "stage_primary_contrasts": chart_rows,
                "early_scope_decomposition": scope_table_rows,
                "llama1b_long_run": long_rows,
            },
            "accessIssues": [],
        },
        "sources": manifest_sources,
        "package_info": {
            "originUrl": "artifact://mech07-iclr-decision-report",
            "controls": {"edit": False, "refresh": False},
        },
    }
    write_json(output / "artifact.json", artifact)
    report_manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "generated_at": generated_at,
        "source_run": run.name,
        "source_primary_analysis": primary.name,
        "artifact": "artifact.json",
        "html": "report.html",
    }
    write_json(output / "report_manifest.json", report_manifest)
    print(f"MECH-07 ICLR report artifact: {output / 'artifact.json'}")
    print(f"MECH-07 ICLR report directory: {output}")


if __name__ == "__main__":
    main()
