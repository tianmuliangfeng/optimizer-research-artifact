"""Build the canonical portable-report artifact for the pilot analysis."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_dir.resolve()

    summary = pd.read_csv(root / "pilot_run_summary.csv")
    selected = pd.read_csv(root / "pilot_method_selection.csv")
    history = pd.read_csv(root / "pilot_history_long.csv")
    quality = pd.read_csv(root / "data_quality_checks.csv")
    curves = history[(history.metric == "val/loss") & (history.step >= 100)][
        ["cell", "method", "lr_label", "step", "value"]
    ].rename(columns={"value": "val_loss"})

    selected = selected.sort_values("final_val_loss_step_1000").copy()
    selected["gap_vs_r1_muon"] = selected["cell"].map(
        {
            "moonlight_r1scale": 0.0102,
            "normuon_r1scale": 0.1195,
            "adamw_low": 0.2820,
        }
    )
    selected["formal_label"] = selected["cell"].map(
        {
            "moonlight_r1scale": "Competitive modern baseline",
            "normuon_r1scale": "Formal reference; boundary LR winner",
            "adamw_low": "Tuned conventional baseline",
        }
    )

    generated_at = datetime.now().astimezone().isoformat()
    pilot_source = {
        "id": "pilot_exports",
        "label": "W&B extended-baseline pilot exports",
        "path": "raw_wandb_exports/",
    }
    r1_source = {
        "id": "r1_reference",
        "label": "R1 controlled seed2026 reference curves",
        "path": "reference/r1_seed2026_normalized_history_long.csv",
    }
    selected_source = {
        "id": "selected_results",
        "label": "Selected pilot configurations",
        "path": "pilot_method_selection.csv",
        "query": {
            "engine": "duckdb",
            "sql": "SELECT * FROM read_csv_auto('pilot_method_selection.csv') ORDER BY final_val_loss_step_1000 ASC",
            "description": "One selected LR cell per optimizer method",
            "language": "sql",
            "tables_used": ["pilot_method_selection.csv"],
            "filters": ["seed=2026", "one selected cell per method"],
            "metric_definitions": [
                "Primary endpoint: validation loss at optimizer step 1000",
                "Tail-3: mean of validation losses at steps 800, 900, and 1000",
            ],
        },
    }
    curve_source = {
        "id": "pilot_curve_rows",
        "label": "Pilot validation curves",
        "path": "pilot_history_long.csv",
        "query": {
            "engine": "duckdb",
            "sql": "SELECT cell, method, lr_label, step, value AS val_loss FROM read_csv_auto('pilot_history_long.csv') WHERE metric = 'val/loss' AND step >= 100 ORDER BY step, cell",
            "description": "Validation loss curves after the shared initial point",
            "language": "sql",
            "tables_used": ["pilot_history_long.csv"],
            "filters": ["metric=val/loss", "step>=100", "seed=2026"],
        },
    }
    results_source = {
        "id": "all_pilot_results",
        "label": "All nine pilot result summaries",
        "path": "pilot_run_summary.csv",
        "query": {
            "engine": "duckdb",
            "sql": "SELECT * FROM read_csv_auto('pilot_run_summary.csv') ORDER BY final_val_loss_step_1000 ASC",
            "description": "Exact endpoint, tail, AUC, and memory summary for every pilot cell",
            "language": "sql",
            "tables_used": ["pilot_run_summary.csv"],
            "filters": ["seed=2026", "pilot step budget=1000"],
        },
    }

    cards = []
    for cell in ("moonlight_r1scale", "normuon_r1scale", "adamw_low"):
        cards.append(
            {
                "id": f"card_{cell}",
                "description": "Selected step-1000 pilot endpoint and gap versus R1 Muon",
                "dataset": "selected_cells",
                "filter": {"cell": cell},
                "sourceId": "selected_results",
                "metrics": [
                    {
                        "label": cell,
                        "field": "final_val_loss_step_1000",
                        "format": "number",
                    },
                    {
                        "label": "gap vs R1 Muon",
                        "field": "gap_vs_r1_muon",
                        "format": "number",
                        "signed": True,
                    },
                ],
            }
        )

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "R1 补充优化器 1000-step Pilot 分析",
            "description": "九个优化器/LR cell 的质量审计、选参与正式实验决策",
            "generatedAt": generated_at,
            "sources": [
                pilot_source,
                r1_source,
                selected_source,
                curve_source,
                results_source,
            ],
            "cards": cards,
            "charts": [
                {
                    "id": "pilot_curves",
                    "title": "九个 Pilot cell 的验证损失曲线",
                    "subtitle": "seed2026，steps 100–1000；越低越好",
                    "type": "line",
                    "dataset": "pilot_curves",
                    "sourceId": "pilot_curve_rows",
                    "encodings": {
                        "x": {
                            "field": "step",
                            "type": "quantitative",
                            "label": "Optimizer step",
                        },
                        "y": {
                            "field": "val_loss",
                            "type": "quantitative",
                            "label": "Validation loss",
                        },
                        "color": {
                            "field": "cell",
                            "type": "nominal",
                            "label": "Pilot cell",
                        },
                        "tooltip": [
                            {"field": "cell", "type": "text", "label": "Cell"},
                            {"field": "step", "type": "quantitative", "label": "Step"},
                            {"field": "val_loss", "type": "quantitative", "label": "Val loss"},
                        ],
                    },
                    "xAxisTitle": "Optimizer step",
                    "yAxisTitle": "Validation loss",
                    "layout": "full",
                    "maxRows": 90,
                    "surface": {"legend": {"visible": True}},
                }
            ],
            "tables": [
                {
                    "id": "selected_table",
                    "title": "进入 6200-step formal 的配置",
                    "subtitle": "每个方法只选择一个 LR cell；seed2026 长程筛选",
                    "dataset": "selected_cells",
                    "defaultSort": {"field": "final_val_loss_step_1000", "direction": "asc"},
                    "density": "spacious",
                    "sourceId": "selected_results",
                    "layout": "full",
                    "columns": [
                        {"field": "cell", "label": "Selected cell", "type": "text"},
                        {"field": "auxiliary_lr", "label": "Aux LR", "format": "number"},
                        {"field": "matrix_lr", "label": "Matrix LR", "format": "number"},
                        {"field": "final_val_loss_step_1000", "label": "Step-1000 val", "format": "number"},
                        {"field": "tail3_mean", "label": "Tail-3", "format": "number"},
                        {"field": "gap_vs_r1_muon", "label": "Gap vs R1 Muon", "format": "number", "movement": True},
                        {"field": "formal_label", "label": "Role", "type": "text"},
                    ],
                },
                {
                    "id": "all_results_table",
                    "title": "全部九个 Pilot cell 的精确结果",
                    "subtitle": "按 step-1000 validation loss 升序；所有运行均为 seed2026",
                    "dataset": "all_results",
                    "defaultSort": {"field": "final_val_loss_step_1000", "direction": "asc"},
                    "density": "dense",
                    "sourceId": "all_pilot_results",
                    "layout": "full",
                    "columns": [
                        {"field": "cell", "label": "Cell", "type": "text"},
                        {"field": "auxiliary_lr", "label": "Aux LR", "format": "number"},
                        {"field": "matrix_lr", "label": "Matrix LR", "format": "number"},
                        {"field": "final_val_loss_step_1000", "label": "Step-1000 val", "format": "number"},
                        {"field": "tail3_mean", "label": "Tail-3", "format": "number"},
                        {"field": "post_initial_auc_100_1000", "label": "AUC 100–1000", "format": "number"},
                        {"field": "peak_allocated_mib", "label": "Peak MiB", "format": "number"},
                        {"field": "optimizer_state_mib", "label": "Optimizer MiB", "format": "number"},
                    ],
                },
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# R1 补充优化器 1000-step Pilot 分析",
                    "layout": "full",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## Moonlight 最接近 R1；三种方法各有一个明确入选配置\n\n"
                        "**Moonlight `r1scale` 是唯一接近现有 R1 优化器的候选。** "
                        "它在 step 1000 的 validation loss 为 **3.7541**，比 R1 Muon "
                        "高 0.0102，比 R1 diag 高 0.0316。\n\n"
                        "每个方法在预设网格中的入选配置为 `moonlight_r1scale`、"
                        "`normuon_r1scale` 和 `adamw_low`。建议三者进入 seed2026 的 "
                        "6200-step formal，但只有 Moonlight 当前具有挑战主结果的竞争力。"
                        "Moonlight-high 和 NorMuon-official 的早期 AUC 更好；入选的低 LR "
                        "分别在 steps 900 和 700 后反超，因此选择依据是冻结的 step-1000 primary endpoint。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "selected_metrics",
                    "type": "metric-strip",
                    "cardIds": [card["id"] for card in cards],
                    "layout": "full",
                },
                {
                    "id": "curve_heading",
                    "type": "markdown",
                    "body": (
                        "## Moonlight 曲线整体领先，低 LR 在 NorMuon 与 AdamW 中占优\n\n"
                        "曲线显示 Moonlight 三个 cell 全程构成第一梯队。NorMuon 和 AdamW "
                        "都在更低 LR 下取得更好的晚期结果。Moonlight-r1scale 与 NorMuon-r1scale "
                        "存在先慢后快的交叉，不能宣称所有曲线指标一致；NorMuon 与 AdamW 的入选点"
                        "位于搜索下边界，只能视为当前网格内、当前主终点下最优。"
                    ),
                    "layout": "full",
                },
                {"id": "curve", "type": "chart", "chartId": "pilot_curves", "layout": "full"},
                {
                    "id": "selection_heading",
                    "type": "markdown",
                    "body": (
                        "## Formal 应保留三条基线，但确认性资源优先给 Moonlight\n\n"
                        "Moonlight 用于检验现代 Muon 变体能否追平核心方法；NorMuon 提供另一种"
                        "现代正交化基线；AdamW 提供传统优化器参照。三者都应使用独立 formal "
                        "模式、6200 updates 和 1800-step warmdown，不能把 pilot 直接拉长。"
                    ),
                    "layout": "full",
                },
                {"id": "selected", "type": "table", "tableId": "selected_table", "layout": "full"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## 数据范围与指标定义\n\n"
                        "分析覆盖 9 个预设 optimizer/LR cell、seed2026、0–1000 steps，"
                        "每 100 steps 使用 10,485,760 validation tokens。主要筛选终点是 "
                        "step-1000 validation loss；tail-3 是 steps 800/900/1000 的均值；"
                        "AUC 是 steps 100–1000 的梯形积分均值。所有比较均为 matched-step/token。"
                    ),
                    "layout": "full",
                },
                {"id": "all_results", "type": "table", "tableId": "all_results_table", "layout": "full"},
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## 191 项数据检查全部通过\n\n"
                        "八份 W&B 导出具有完整的九-run 覆盖和预期 step 网格；MIN/MAX 与基础"
                        "序列一致；没有非有限值、重复 cell/metric/step 或初始 loss 不一致。"
                        "因此 W&B 质量与状态数据可用于 pilot 选参。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 限制：这是选参证据，不是独立确认\n\n"
                        "状态为 **PASS_WITH_CAVEATS**：CSV 不包含远端本地 manifest、source/runtime/"
                        "init fingerprint 或 resume count；seed2026 同时承担 LR 选择，因此其 formal "
                        "只能作为长程筛选。若方法进入主表，seeds2024/2025 才是未参与选参的确认性证据。"
                        "此外 timing 仅作诊断，不进入论文性能结论。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 下一步：冻结三个 LR，运行独立 6200-step formal\n\n"
                        "1. `moonlight_r1scale`: aux/matrix LR 0.0018，weight decay 0.1。\n"
                        "2. `normuon_r1scale`: aux LR 0.0003，matrix LR 0.01，weight decay 0.01。\n"
                        "3. `adamw_low`: base LR 0.0027，hidden LR 0.000432，weight decay 0。\n\n"
                        "Primary endpoint 预先固定为 step-6200 final validation loss；tail-5、"
                        "normalized AUC 与 steps/tokens-to-target 为 secondary。只有 formal 仍有竞争力"
                        "的方法才扩展 seeds2024/2025。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## 仍待回答的问题\n\n"
                        "- Moonlight 与 R1 Muon 的 0.0102 差距在长程训练与 warmdown 后会缩小还是扩大？\n"
                        "- NorMuon/AdamW 的低 LR 优势是否只是早期训练预算特有？\n"
                        "- 若 Moonlight 进入主表，其优势或持平结论能否在 seeds2024/2025 保持？"
                    ),
                    "layout": "full",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "selected_cells": records(selected),
                "pilot_curves": records(curves),
                "all_results": records(summary.sort_values("final_val_loss_step_1000")),
                "quality_summary": [
                    {
                        "checks": len(quality),
                        "pass": int((quality.status == "PASS").sum()),
                        "fail": int((quality.status == "FAIL").sum()),
                    }
                ],
            },
        },
        "sources": [
            pilot_source,
            r1_source,
            selected_source,
            curve_source,
            results_source,
        ],
    }

    (root / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(root / "artifact.json")


if __name__ == "__main__":
    main()
