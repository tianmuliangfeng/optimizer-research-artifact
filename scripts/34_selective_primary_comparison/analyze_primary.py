#!/usr/bin/env python3
"""Build the primary Selective Newton-Muon comparison from audited summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-27.1"
SEEDS = (2024, 2025, 2026)
PRACTICAL_MARGIN = 0.002
T_CRIT_DF2_95 = 4.302652729911275

FAMILIES = {
    "r1": {
        "label": "GPT-R1",
        "relative_source": (
            "15_official_newton_muon_r1/analysis/"
            "wandb_20260721_multiseed_factorial/r1_multiseed_run_summary.csv"
        ),
        "methods": {
            "selective_diag": "diag",
            "selective_none": "none",
            "original_newton_muon": "block4",
            "muon": "muon",
        },
        "auc_field": "normalized_val_auc",
        "peak_field": "peak_memory_mib",
        "k_state_field": "k_state_mib",
        "optimizer_state_field": "optimizer_state_mib",
    },
    "llama124": {
        "label": "LLaMA-124M",
        "relative_source": (
            "17_llama_swiglu_validation/analysis/"
            "wandb_20260722_multiseed/llama_multiseed_run_summary.csv"
        ),
        "methods": {
            "selective_diag": "down_diag",
            "selective_none": "down_none",
            "original_newton_muon": "newton_full",
            "muon": "muon",
        },
        "auc_field": "normalized_val_auc",
        "peak_field": "peak_allocated_mib",
        "k_state_field": "k_state_mib",
        "optimizer_state_field": "optimizer_state_mib",
    },
    "llama1b": {
        "label": "LLaMA-1B",
        "relative_source": (
            "20_llama_swiglu_1b/analysis/formal6200_multiseed_20260727/"
            "llama1b_formal_multiseed_run_summary.csv"
        ),
        "methods": {
            "selective_diag": "down_diag",
            "selective_none": "down_none",
            "original_newton_muon": "newton_full",
            "muon": "muon",
        },
        "auc_field": "normalized_val_auc_0_6200",
        "peak_field": None,
        "k_state_field": "expected_k_state_mib_from_preflight",
        "optimizer_state_field": None,
    },
}

# Primary order is deliberate. diag-vs-none is not part of this contract.
CONTRASTS = (
    ("selective_diag_vs_muon", "selective_diag", "muon", "primary"),
    ("selective_none_vs_muon", "selective_none", "muon", "primary"),
    (
        "selective_diag_vs_original_newton_muon",
        "selective_diag",
        "original_newton_muon",
        "primary",
    ),
    (
        "selective_none_vs_original_newton_muon",
        "selective_none",
        "original_newton_muon",
        "primary",
    ),
    (
        "original_newton_muon_vs_muon",
        "original_newton_muon",
        "muon",
        "baseline",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def ci95(mean: float, sd: float, n: int) -> tuple[float, float]:
    radius = T_CRIT_DF2_95 * sd / math.sqrt(n)
    return mean - radius, mean + radius


def classify(mean: float, positive_seeds: int, negative_seeds: int) -> str:
    if negative_seeds == len(SEEDS) and mean < -PRACTICAL_MARGIN:
        return "selective_or_left_materially_better"
    if positive_seeds == len(SEEDS) and mean > PRACTICAL_MARGIN:
        return "selective_or_left_materially_worse"
    if abs(mean) <= PRACTICAL_MARGIN:
        return "within_practical_margin"
    return "mixed_or_uncertain"


def required_methods(spec: dict[str, Any]) -> set[str]:
    return set(spec["methods"].values())


def validate_rows(
    family: str, rows: list[dict[str, str]], spec: dict[str, Any]
) -> dict[tuple[str, int], dict[str, str]]:
    selected = [
        row
        for row in rows
        if row.get("method") in required_methods(spec)
        and int(row.get("seed", -1)) in SEEDS
    ]
    keyed: dict[tuple[str, int], dict[str, str]] = {}
    for row in selected:
        key = (row["method"], int(row["seed"]))
        if key in keyed:
            raise RuntimeError(f"{family}: duplicate method/seed row {key}")
        keyed[key] = row
    expected = {
        (method, seed) for method in required_methods(spec) for seed in SEEDS
    }
    if set(keyed) != expected:
        missing = sorted(expected - set(keyed))
        extra = sorted(set(keyed) - expected)
        raise RuntimeError(f"{family}: coverage mismatch missing={missing} extra={extra}")
    required_fields = {
        "final_val_loss",
        "tail5_val_loss_mean",
        spec["auc_field"],
        spec["k_state_field"],
    }
    for field in required_fields:
        if not field or any(row.get(field, "") == "" for row in keyed.values()):
            raise RuntimeError(f"{family}: required field missing or empty: {field}")
    return keyed


def build_method_rows(
    family: str,
    spec: dict[str, Any],
    keyed: dict[tuple[str, int], dict[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for role, method in spec["methods"].items():
        subset = [keyed[(method, seed)] for seed in SEEDS]
        final_values = [float(row["final_val_loss"]) for row in subset]
        tail_values = [float(row["tail5_val_loss_mean"]) for row in subset]
        auc_values = [float(row[spec["auc_field"]]) for row in subset]
        k_values = [float(row[spec["k_state_field"]]) for row in subset]
        peak_values = (
            [float(row[spec["peak_field"]]) for row in subset]
            if spec["peak_field"]
            and all(row.get(spec["peak_field"], "") != "" for row in subset)
            else []
        )
        optimizer_values = (
            [float(row[spec["optimizer_state_field"]]) for row in subset]
            if spec["optimizer_state_field"]
            and all(
                row.get(spec["optimizer_state_field"], "") != "" for row in subset
            )
            else []
        )
        final_mean, final_sd = mean_sd(final_values)
        output.append(
            {
                "family": family,
                "family_label": spec["label"],
                "role": role,
                "method": method,
                "seeds": len(SEEDS),
                "final_loss_mean": final_mean,
                "final_loss_sd": final_sd,
                "tail5_loss_mean": statistics.mean(tail_values),
                "auc_mean": statistics.mean(auc_values),
                "k_state_mib_mean": statistics.mean(k_values),
                "optimizer_state_mib_mean": (
                    statistics.mean(optimizer_values) if optimizer_values else ""
                ),
                "peak_memory_mib_mean": (
                    statistics.mean(peak_values) if peak_values else ""
                ),
            }
        )
    return output


def build_contrasts(
    family: str,
    spec: dict[str, Any],
    keyed: dict[tuple[str, int], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_seed: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for name, left_role, right_role, priority in CONTRASTS:
        left = spec["methods"][left_role]
        right = spec["methods"][right_role]
        final_deltas: list[float] = []
        tail_deltas: list[float] = []
        auc_deltas: list[float] = []
        for seed in SEEDS:
            left_row = keyed[(left, seed)]
            right_row = keyed[(right, seed)]
            final_delta = float(left_row["final_val_loss"]) - float(
                right_row["final_val_loss"]
            )
            tail_delta = float(left_row["tail5_val_loss_mean"]) - float(
                right_row["tail5_val_loss_mean"]
            )
            auc_delta = float(left_row[spec["auc_field"]]) - float(
                right_row[spec["auc_field"]]
            )
            final_deltas.append(final_delta)
            tail_deltas.append(tail_delta)
            auc_deltas.append(auc_delta)
            by_seed.append(
                {
                    "family": family,
                    "family_label": spec["label"],
                    "priority": priority,
                    "contrast": name,
                    "left_role": left_role,
                    "left_method": left,
                    "right_role": right_role,
                    "right_method": right,
                    "seed": seed,
                    "final_delta_left_minus_right": final_delta,
                    "tail5_delta_left_minus_right": tail_delta,
                    "auc_delta_left_minus_right": auc_delta,
                }
            )
        mean, sd = mean_sd(final_deltas)
        low, high = ci95(mean, sd, len(final_deltas))
        negative = sum(value < 0 for value in final_deltas)
        positive = sum(value > 0 for value in final_deltas)
        summary.append(
            {
                "family": family,
                "family_label": spec["label"],
                "priority": priority,
                "contrast": name,
                "left_role": left_role,
                "left_method": left,
                "right_role": right_role,
                "right_method": right,
                "seeds": len(SEEDS),
                "final_delta_mean": mean,
                "final_delta_sd": sd,
                "final_delta_ci95_low": low,
                "final_delta_ci95_high": high,
                "negative_seeds_left_better": negative,
                "positive_seeds_left_worse": positive,
                "tail5_delta_mean": statistics.mean(tail_deltas),
                "auc_delta_mean": statistics.mean(auc_deltas),
                "practical_margin": PRACTICAL_MARGIN,
                "classification": classify(mean, positive, negative),
            }
        )
    return by_seed, summary


def format_delta(row: dict[str, Any]) -> str:
    return (
        f"{float(row['final_delta_mean']):+.6f} ± "
        f"{float(row['final_delta_sd']):.6f}"
    )


def find_summary(
    rows: list[dict[str, Any]], family: str, contrast: str
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["family"] == family and row["contrast"] == contrast
    )


def build_markdown(
    summary: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    source_audit: dict[str, Any],
) -> str:
    lines = [
        "# Selective Newton–Muon：以两个外部基线为中心的统一主分析",
        "",
        "## 技术摘要",
        "",
        "本报告纠正比较层级：`diag` 与 `none` 都是本文提出的 Selective "
        "Newton–Muon 方案。主比较是每个方案分别对 Muon 和原始 "
        "Newton–Muon；`diag vs none` 不属于主对比合同。",
        "",
        "- GPT-R1：两个 Selective 方案均优于 Muon；diag 基本保留原始 "
        "block4，none 则相对 block4 有可辨认损失。",
        "- LLaMA-124M：两个 Selective 方案、Newton-full 与 Muon 处于紧密核心组，"
        "当前三 seed 不能支持稳定质量排序。",
        "- LLaMA-1B：两个 Selective 方案均稳定优于 Newton-full，但后期均稳定"
        "落后于 Muon。因此 Selective 改善了原始 Newton–Muon，却没有消除"
        "1B 后期的 family-level gap。",
        "",
        "负 delta 表示左侧方法 loss 更低。均值与 SD 均来自 seeds "
        "2024/2025/2026 的配对差；practical margin 为 0.002 loss。",
        "",
        "## 主证据：每个 Selective 方案分别对两个基线",
        "",
        "| 架构 | 主对比 | Final delta mean ± SD | 左侧更优 seeds | Tail-5 delta | AUC delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for contrast in (
            "selective_diag_vs_muon",
            "selective_none_vs_muon",
            "selective_diag_vs_original_newton_muon",
            "selective_none_vs_original_newton_muon",
        ):
            row = find_summary(summary, family, contrast)
            lines.append(
                f"| {row['family_label']} | `{row['left_method']} − "
                f"{row['right_method']}` | {format_delta(row)} | "
                f"{row['negative_seeds_left_better']}/3 | "
                f"{float(row['tail5_delta_mean']):+.6f} | "
                f"{float(row['auc_delta_mean']):+.6f} |"
            )
    lines.extend(
        [
            "",
            "这张表是论文主比较。`diag − none` 被有意排除，因为两者都是本文方案；"
            "它只能出现在补充消融中。",
            "",
            "## 基线关系决定 Selective 方案的解释",
            "",
            "| 架构 | 原始 Newton–Muon − Muon | Final 同方向 seeds | 解释 |",
            "|---|---:|---:|---|",
        ]
    )
    interpretations = {
        "r1": "原始 Newton–Muon 本身优于 Muon；Selective 目标是保留收益并降低状态。",
        "llama124": "两条基线近似打平；Selective 的主要价值是状态效率而非明确质量提升。",
        "llama1b": "原始 Newton–Muon 后期落后于 Muon；Selective 虽改善原始方法，仍未反超 Muon。",
    }
    for family in FAMILIES:
        row = find_summary(summary, family, "original_newton_muon_vs_muon")
        lines.append(
            f"| {row['family_label']} | {format_delta(row)} | "
            f"{row['negative_seeds_left_better']}/3 left-better, "
            f"{row['positive_seeds_left_worse']}/3 left-worse | "
            f"{interpretations[family]} |"
        )
    lines.extend(
        [
            "",
            "## 状态成本：两个方案都比原始 Newton–Muon 更轻",
            "",
            "| 架构 | 方法角色 | 方法 | K-state MiB | Final loss mean ± SD |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in method_rows:
        if row["role"] in {
            "selective_diag",
            "selective_none",
            "original_newton_muon",
            "muon",
        }:
            lines.append(
                f"| {row['family_label']} | `{row['role']}` | `{row['method']}` | "
                f"{float(row['k_state_mib_mean']):.3f} | "
                f"{float(row['final_loss_mean']):.6f} ± "
                f"{float(row['final_loss_sd']):.6f} |"
            )
    lines.extend(
        [
            "",
            "K-state 只表示输入预条件器状态，不等于总 optimizer state 或峰值显存。"
            "不同架构的绝对 loss 也不可直接横向比较。",
            "",
            "## 范围、数据与指标定义",
            "",
            "- 架构：GPT-R1、LLaMA-124M、LLaMA-1B。",
            "- population：每个架构的正式 seeds 2024/2025/2026，固定训练预算的"
            "最终验证点。",
            "- Selective 方案：R1 的 `diag/none`；LLaMA 的 "
            "`down_diag/down_none`。",
            "- 原始 Newton–Muon：R1 的官方 `block4`；LLaMA 的 `newton_full`。",
            "- Muon：独立的 reference Muon baseline；`none` 不等于 Muon。",
            "- Primary metric：paired final validation loss delta；negative 表示"
            "左侧更好。",
            "- Supporting metrics：paired tail-5 delta 与 normalized validation "
            "AUC delta。",
            "",
            "## 方法与验证",
            "",
            "所有 delta 都由逐 run authoritative summary 重新计算，而不是从旧报告"
            "文字或名义排名抄录。每个 family 强制要求 4 methods × 3 seeds 恰好"
            "12 个唯一 method/seed cells；缺失或重复会终止分析。95% t 区间使用"
            "n=3、df=2，仅作为不确定性描述，不以显著性替代 effect size 与三 seed "
            "方向一致性。",
            "",
            "本报告使用精确表格而非跨架构柱状图：三个架构的绝对 loss 尺度与"
            "训练语义不同，主问题又要求精确读取十多个配对 contrast；合并图形容易"
            "把跨架构数值误读成同一量纲的排名。",
            "",
            "## 局限、稳健性与可写边界",
            "",
            "- 每个架构只有 3 seeds，区间较宽；结论依赖配对设计、方向一致性和"
            "预先使用的 0.002 practical margin。",
            "- LLaMA-1B 的 W&B timing 与 memory 不作为性能证据；质量结论来自"
            "固定 token budget 的 validation metrics。",
            "- MECH-05/06 只回答 Newton–Muon 家族内部 K-state 选择，不能替代"
            "本报告的 family-level 主比较。",
            "- 可以写“Selective 改善/压缩原始 Newton–Muon”；只有当其对 Muon "
            "contrast 同时满足证据时，才可以写“相对 Muon 保持或改进”。",
            "- 1B 必须如实写成：Selective 优于 Newton-full，但后期仍落后 Muon。",
            "",
            "## 下一步",
            "",
            "无需新增长训练。唯一建议补充的是 LLaMA-1B seed2026 的 "
            "4 methods × early/late checkpoints 只读 family-level diagnostic，"
            "用于解释 early-to-late family reversal；它不能替代三 seed 长程"
            "因果比较。",
            "",
            "## 进一步问题",
            "",
            "若只读 family diagnostic 能稳定定位某类 update-direction 或"
            "held-out shadow-loss 反转，再条件式决定是否在 GPT-R1 与 "
            "LLaMA-124M endpoint 复现；否则停止扩实验。",
            "",
            "## Source audit",
            "",
        ]
    )
    for family, audit in source_audit["sources"].items():
        lines.append(
            f"- {family}: `{audit['relative_path']}`, SHA-256 "
            f"`{audit['sha256']}`, rows={audit['rows']}."
        )
    return "\n".join(lines) + "\n"


def markdown_to_html(markdown: str) -> str:
    """Render the fixed report shape without an external Markdown dependency."""
    lines = markdown.splitlines()
    body: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            items = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append(f"<li>{html.escape(lines[index][2:])}</li>")
                index += 1
            body.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            headers = [value.strip() for value in table_lines[0].strip("|").split("|")]
            rows = table_lines[2:]
            body.append("<div class='table-wrap'><table><thead><tr>")
            body.extend(f"<th>{html.escape(value)}</th>" for value in headers)
            body.append("</tr></thead><tbody>")
            for row in rows:
                values = [value.strip() for value in row.strip("|").split("|")]
                body.append("<tr>")
                body.extend(f"<td>{html.escape(value)}</td>" for value in values)
                body.append("</tr>")
            body.append("</tbody></table></div>")
            continue
        elif line:
            body.append(f"<p>{html.escape(line)}</p>")
        index += 1
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Selective Newton–Muon unified primary comparison</title>
<style>
:root{color-scheme:light dark;font-family:Inter,"Noto Sans SC",system-ui,sans-serif}
body{max-width:1180px;margin:auto;padding:32px 24px;line-height:1.65}
h1{font-size:2rem;line-height:1.25}h2{margin-top:2.2rem;border-bottom:1px solid #8885;padding-bottom:.3rem}
p,li{max-width:88ch}.table-wrap{overflow-x:auto;margin:1rem 0}
table{border-collapse:collapse;width:100%;font-size:.92rem}th,td{border:1px solid #8886;padding:.55rem;text-align:left}
th{background:#8882}code{font-family:ui-monospace,monospace}
</style></head><body>""" + "\n".join(body) + "</body></html>\n"


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    method_rows: list[dict[str, Any]] = []
    by_seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "comparison_contract": {
            "primary": [
                "each selective proposal vs Muon",
                "each selective proposal vs original Newton-Muon",
            ],
            "baseline": "original Newton-Muon vs Muon",
            "excluded_from_primary": "diag vs none",
        },
        "sources": {},
    }
    for family, spec in FAMILIES.items():
        path = input_root / spec["relative_source"]
        rows = read_csv(path)
        keyed = validate_rows(family, rows, spec)
        method_rows.extend(build_method_rows(family, spec, keyed))
        by_seed, summary = build_contrasts(family, spec, keyed)
        by_seed_rows.extend(by_seed)
        summary_rows.extend(summary)
        source_audit["sources"][family] = {
            "relative_path": spec["relative_source"],
            "sha256": sha256_file(path),
            "rows": len(rows),
            "selected_cells": len(keyed),
            "coverage_passed": True,
        }

    # Contract guard: no proposal-vs-proposal contrast may enter the main tables.
    if any(
        {row["left_role"], row["right_role"]}
        == {"selective_diag", "selective_none"}
        for row in summary_rows
    ):
        raise RuntimeError("diag-vs-none was incorrectly admitted to primary contrasts")

    write_csv(output / "primary_method_summary.csv", method_rows)
    write_csv(output / "primary_contrasts_by_seed.csv", by_seed_rows)
    write_csv(output / "primary_contrasts_summary.csv", summary_rows)
    write_json(output / "source_audit.json", source_audit)
    report = build_markdown(summary_rows, method_rows, source_audit)
    (output / "SELECTIVE_PRIMARY_COMPARISON_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    (output / "report.html").write_text(markdown_to_html(report), encoding="utf-8")

    output_names = [
        "SELECTIVE_PRIMARY_COMPARISON_REPORT.md",
        "primary_contrasts_by_seed.csv",
        "primary_contrasts_summary.csv",
        "primary_method_summary.csv",
        "report.html",
        "source_audit.json",
    ]
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "passed": True,
        "families": list(FAMILIES),
        "seeds": list(SEEDS),
        "primary_contrasts_per_family": 4,
        "baseline_contrasts_per_family": 1,
        "diag_vs_none_primary": False,
        "practical_margin": PRACTICAL_MARGIN,
        "artifacts": output_names,
        "output_sha256": {
            name: sha256_file(output / name) for name in output_names
        },
    }
    write_json(output / "primary_analysis_manifest.json", manifest)
    print(f"Primary comparison manifest: {output / 'primary_analysis_manifest.json'}")
    print(f"Primary comparison artifacts: {output}")


if __name__ == "__main__":
    main()
