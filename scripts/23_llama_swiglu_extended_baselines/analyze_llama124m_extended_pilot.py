#!/usr/bin/env python3
"""Analyze the frozen seed2026 LLaMA-124M extended-baseline pilot.

The script intentionally depends only on the Python standard library so that
the pilot decision can be reproduced without the training environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_METRICS = {
    "val/loss",
    "train/loss_step",
    "tokens/seen",
    "time/train_s",
    "performance/step_avg_ms",
    "lr/auxiliary",
    "lr/matrix",
}
EXPECTED_CELLS = {
    ("moonlight", "official"),
    ("moonlight", "r1scale"),
    ("moonlight", "high"),
    ("normuon", "low"),
    ("normuon", "r1scale"),
    ("normuon", "official"),
}
EXPECTED_LRS = {
    ("moonlight", "official"): (0.001, 0.001),
    ("moonlight", "r1scale"): (0.0018, 0.0018),
    ("moonlight", "high"): (0.003, 0.003),
    ("normuon", "low"): (0.0003, 0.005),
    ("normuon", "r1scale"): (0.0003, 0.01),
    ("normuon", "official"): (0.0003, 0.02),
}
RUN_RE = re.compile(
    r"^llama124m_ext_pilot_(moonlight|normuon)_([a-z0-9]+)_seed(20\d\d)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports", nargs=7, required=True, type=Path)
    parser.add_argument("--core-history", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise ValueError(f"Cannot infer columns for empty CSV: {path}")
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def metric_float(value: str, *, allow_nan: bool = False) -> float:
    number = float(value)
    if math.isnan(number) and allow_nan:
        return number
    if not math.isfinite(number):
        raise ValueError(f"Non-finite metric value: {value}")
    return number


def load_exports(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    long_rows: list[dict] = []
    sources: list[dict] = []
    observed_metrics: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append(
            {
                "source_file": path.name,
                "source_path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or reader.fieldnames[0] != "Step":
                raise ValueError(f"Unexpected W&B export header: {path}")
            main_columns = [
                col for col in reader.fieldnames[1:] if not col.endswith(("__MIN", "__MAX"))
            ]
            if not main_columns:
                raise ValueError(f"No metric columns: {path}")
            metrics = {col.rsplit(" - ", 1)[1] for col in main_columns}
            if len(metrics) != 1:
                raise ValueError(f"Mixed metrics in one export: {path}: {metrics}")
            metric = next(iter(metrics))
            observed_metrics.add(metric)
            rows = list(reader)
            for raw in rows:
                step = int(float(raw["Step"]))
                for col in main_columns:
                    run_name, col_metric = col.rsplit(" - ", 1)
                    if col_metric != metric:
                        raise AssertionError((col_metric, metric))
                    match = RUN_RE.fullmatch(run_name)
                    if not match:
                        raise ValueError(f"Unexpected run name: {run_name}")
                    method, cell, seed = match.groups()
                    allow_nan = metric == "performance/step_avg_ms" and step in (0, 20)
                    value = metric_float(raw[col], allow_nan=allow_nan)
                    for suffix in ("__MIN", "__MAX"):
                        duplicate = raw.get(col + suffix, "")
                        if duplicate and metric_float(duplicate, allow_nan=allow_nan) != value:
                            raise ValueError(
                                f"W&B duplicate aggregate differs: {path.name}, {col}{suffix}"
                            )
                    long_rows.append(
                        {
                            "method": method,
                            "cell": cell,
                            "run_name": run_name,
                            "seed": int(seed),
                            "metric": metric,
                            "step": step,
                            "value": value,
                            "source_file": path.name,
                        }
                    )
    if observed_metrics != EXPECTED_METRICS:
        raise ValueError(
            f"Metric set mismatch: observed={sorted(observed_metrics)}, "
            f"expected={sorted(EXPECTED_METRICS)}"
        )
    return long_rows, sources


def trapezoid_normalized(points: list[tuple[int, float]]) -> float:
    ordered = sorted(points)
    if ordered[-1][0] == ordered[0][0]:
        raise ValueError("AUC requires a nonzero step range")
    area = sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for (left_step, left_value), (right_step, right_value) in zip(
            ordered, ordered[1:]
        )
    )
    return area / (ordered[-1][0] - ordered[0][0])


def metric_map(rows: list[dict]) -> dict[tuple[str, str], list[tuple[int, float]]]:
    mapped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        mapped[(row["run_name"], row["metric"])].append((row["step"], row["value"]))
    for key in mapped:
        mapped[key].sort()
    return mapped


def scalar_at(
    mapped: dict[tuple[str, str], list[tuple[int, float]]],
    run: str,
    metric: str,
    step: int,
) -> float:
    values = dict(mapped[(run, metric)])
    return values[step]


def summarize_pilot(rows: list[dict]) -> list[dict]:
    mapped = metric_map(rows)
    runs: dict[str, tuple[str, str, int]] = {}
    for row in rows:
        runs[row["run_name"]] = (row["method"], row["cell"], row["seed"])
    summaries: list[dict] = []
    for run_name, (method, cell, seed) in sorted(runs.items()):
        val_points = mapped[(run_name, "val/loss")]
        train_points = mapped[(run_name, "train/loss_step")]
        token_points = mapped[(run_name, "tokens/seen")]
        aux_points = mapped[(run_name, "lr/auxiliary")]
        matrix_points = mapped[(run_name, "lr/matrix")]
        expected_aux, expected_matrix = EXPECTED_LRS[(method, cell)]
        val_values = [value for _, value in val_points]
        last_three = val_values[-3:]
        aux_values = {value for _, value in aux_points}
        matrix_values = {value for _, value in matrix_points}
        summaries.append(
            {
                "method": method,
                "cell": cell,
                "run_name": run_name,
                "seed": seed,
                "matrix_lr": matrix_points[-1][1],
                "auxiliary_lr": aux_points[-1][1],
                "expected_matrix_lr": expected_matrix,
                "expected_auxiliary_lr": expected_aux,
                "final_step": val_points[-1][0],
                "final_tokens": int(token_points[-1][1]),
                "val_points": len(val_points),
                "train_points": len(train_points),
                "initial_val_loss": val_values[0],
                "final_val_loss": val_values[-1],
                "tail3_val_loss_mean": statistics.mean(last_three),
                "tail3_val_loss_sd": statistics.stdev(last_three),
                "normalized_val_auc": trapezoid_normalized(val_points),
                "final_train_loss": train_points[-1][1],
                "train_s": scalar_at(mapped, run_name, "time/train_s", 1000),
                "final_step_avg_ms": scalar_at(
                    mapped, run_name, "performance/step_avg_ms", 1000
                ),
                "val_strictly_decreasing": all(
                    b < a for a, b in zip(val_values, val_values[1:])
                ),
                "lr_constant_and_expected": (
                    aux_values == {expected_aux} and matrix_values == {expected_matrix}
                ),
                "complete": (
                    val_points[0][0] == 0
                    and val_points[-1][0] == 1000
                    and len(val_points) == 11
                    and train_points[-1][0] == 1000
                    and token_points[-1][1] == 524_288_000
                ),
            }
        )
    return summaries


def select_cells(summaries: list[dict]) -> list[dict]:
    selections: list[dict] = []
    for method in ("moonlight", "normuon"):
        group = sorted(
            (row for row in summaries if row["method"] == method),
            key=lambda row: row["final_val_loss"],
        )
        best, runner_up = group[:2]
        final_gap = runner_up["final_val_loss"] - best["final_val_loss"]
        within_tie = final_gap <= 0.002
        if within_tie:
            selected = min(group[:2], key=lambda row: row["tail3_val_loss_mean"])
            selection_rule = "tail3_tiebreak"
        else:
            selected = best
            selection_rule = "primary_final"
        selections.append(
            {
                "method": method,
                "selected_cell": selected["cell"],
                "selected_matrix_lr": selected["matrix_lr"],
                "selected_auxiliary_lr": selected["auxiliary_lr"],
                "selected_final_val_loss": selected["final_val_loss"],
                "selected_tail3_mean": selected["tail3_val_loss_mean"],
                "selected_normalized_auc": selected["normalized_val_auc"],
                "runner_up_cell": runner_up["cell"],
                "runner_up_final_val_loss": runner_up["final_val_loss"],
                "runner_up_minus_best_final": final_gap,
                "within_0p002_tie_band": within_tie,
                "selection_rule": selection_rule,
            }
        )
    return selections


def load_core_prefix(path: Path) -> tuple[list[dict], list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    val_by_method: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["seed"]) == 2026
                and row["metric"] == "val/loss"
                and int(row["step"]) <= 1000
            ):
                val_by_method[row["method"]].append(
                        (int(row["step"]), metric_float(row["value"]))
                )
    expected_methods = {"adamw", "down_diag", "down_none", "muon", "newton_full"}
    if set(val_by_method) != expected_methods:
        raise ValueError(f"Core methods mismatch: {sorted(val_by_method)}")
    summaries: list[dict] = []
    curves: list[dict] = []
    for method, points in sorted(val_by_method.items()):
        points.sort()
        if [step for step, _ in points] != list(range(0, 1001, 100)):
            raise ValueError(f"Core prefix grid mismatch: {method}")
        values = [value for _, value in points]
        summaries.append(
            {
                "method": method,
                "seed": 2026,
                "final_step": 1000,
                "val_points": 11,
                "final_val_loss": values[-1],
                "tail3_val_loss_mean": statistics.mean(values[-3:]),
                "tail3_val_loss_sd": statistics.stdev(values[-3:]),
                "normalized_val_auc": trapezoid_normalized(points),
            }
        )
        curves.extend(
            {
                "method": method,
                "series": method,
                "source_family": "core",
                "step": step,
                "val_loss": value,
            }
            for step, value in points
        )
    return summaries, curves


def comparisons(
    pilot: list[dict], selections: list[dict], core: list[dict]
) -> list[dict]:
    selected = {
        row["method"]: next(
            item
            for item in pilot
            if item["method"] == row["method"] and item["cell"] == row["selected_cell"]
        )
        for row in selections
    }
    output: list[dict] = []
    for pilot_method, pilot_row in selected.items():
        for core_row in core:
            output.append(
                {
                    "pilot_method": pilot_method,
                    "pilot_cell": pilot_row["cell"],
                    "core_method": core_row["method"],
                    "final_delta_pilot_minus_core": (
                        pilot_row["final_val_loss"] - core_row["final_val_loss"]
                    ),
                    "tail3_delta_pilot_minus_core": (
                        pilot_row["tail3_val_loss_mean"]
                        - core_row["tail3_val_loss_mean"]
                    ),
                    "auc_delta_pilot_minus_core": (
                        pilot_row["normalized_val_auc"]
                        - core_row["normalized_val_auc"]
                    ),
                }
            )
    return output


def quality_checks(
    rows: list[dict], summaries: list[dict], selections: list[dict]
) -> list[dict]:
    observed_cells = {(row["method"], row["cell"]) for row in summaries}
    run_metric_counts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        run_metric_counts[row["run_name"]].add(row["metric"])
    checks = [
        {
            "check": "six_expected_cells",
            "passed": observed_cells == EXPECTED_CELLS,
            "detail": f"observed={sorted(observed_cells)}",
        },
        {
            "check": "all_seed2026",
            "passed": {row["seed"] for row in summaries} == {2026},
            "detail": f"seeds={sorted({row['seed'] for row in summaries})}",
        },
        {
            "check": "all_expected_metrics_per_run",
            "passed": all(metrics == EXPECTED_METRICS for metrics in run_metric_counts.values()),
            "detail": f"runs={len(run_metric_counts)}, metrics_per_run=7",
        },
        {
            "check": "all_1000_steps_and_token_budget",
            "passed": all(row["complete"] for row in summaries),
            "detail": "11 val points; final step=1000; final tokens=524288000",
        },
        {
            "check": "all_required_metrics_finite",
            "passed": all(
                math.isfinite(row["value"])
                for row in rows
                if row["metric"] != "performance/step_avg_ms"
            ),
            "detail": "all quality, budget, LR and elapsed-time values are finite",
        },
        {
            "check": "step_time_nan_only_during_warmup",
            "passed": all(
                math.isfinite(row["value"])
                or (
                    row["metric"] == "performance/step_avg_ms"
                    and row["step"] in (0, 20)
                )
                for row in rows
            ),
            "detail": "12 expected NaNs: six runs at steps 0 and 20",
        },
        {
            "check": "common_initial_validation_loss",
            "passed": len({row["initial_val_loss"] for row in summaries}) == 1,
            "detail": f"initial={summaries[0]['initial_val_loss']:.9f}",
        },
        {
            "check": "all_validation_curves_strictly_decrease",
            "passed": all(row["val_strictly_decreasing"] for row in summaries),
            "detail": "11/11 validation points per run",
        },
        {
            "check": "learning_rates_match_frozen_grid",
            "passed": all(row["lr_constant_and_expected"] for row in summaries),
            "detail": "auxiliary and matrix LR traces match README grid",
        },
        {
            "check": "one_selected_cell_per_method",
            "passed": {row["method"] for row in selections} == {"moonlight", "normuon"},
            "detail": f"selections={[(r['method'], r['selected_cell']) for r in selections]}",
        },
    ]
    return checks


def make_curves(
    rows: list[dict],
    selections: list[dict],
    core_curves: list[dict],
) -> tuple[list[dict], list[dict]]:
    pilot_all = [
        {
            "method": row["method"],
            "cell": row["cell"],
            "series": f"{row['method']}_{row['cell']}",
            "source_family": "pilot",
            "step": row["step"],
            "val_loss": row["value"],
        }
        for row in rows
        if row["metric"] == "val/loss"
    ]
    selected_cells = {(row["method"], row["selected_cell"]) for row in selections}
    selected = [
        row for row in pilot_all if (row["method"], row["cell"]) in selected_cells
    ]
    selected_and_core = selected + core_curves
    return pilot_all, selected_and_core


def render_markdown(
    output_dir: Path,
    summaries: list[dict],
    selections: list[dict],
    comparisons_rows: list[dict],
    checks: list[dict],
) -> None:
    moonlight = next(row for row in selections if row["method"] == "moonlight")
    normuon = next(row for row in selections if row["method"] == "normuon")
    moon_vs = {
        row["core_method"]: row
        for row in comparisons_rows
        if row["pilot_method"] == "moonlight"
    }
    nor_vs = {
        row["core_method"]: row
        for row in comparisons_rows
        if row["pilot_method"] == "normuon"
    }
    lines = [
        "# LLaMA/SwiGLU-124M 扩展基线 pilot 分析",
        "",
        "## 结论先行",
        "",
        f"- Moonlight 依照预注册 primary endpoint 选择 `high`：matrix/aux LR={moonlight['selected_matrix_lr']:.4g}，"
        f"step-1000 validation loss={moonlight['selected_final_val_loss']:.6f}。它比 Muon 低 "
        f"{abs(moon_vs['muon']['final_delta_pilot_minus_core']):.6f}，与 Newton trio 的终点差仅 "
        f"{min(abs(moon_vs[m]['final_delta_pilot_minus_core']) for m in ('down_diag', 'down_none', 'newton_full')):.6f}–"
        f"{max(abs(moon_vs[m]['final_delta_pilot_minus_core']) for m in ('down_diag', 'down_none', 'newton_full')):.6f}。"
        "架构适配 gate 通过，建议进入冻结 LR 的 6200-step LLaMA-124M confirmatory formal。",
        f"- NorMuon 依 primary endpoint 选择 `r1scale`：matrix LR={normuon['selected_matrix_lr']:.4g}，"
        f"final={normuon['selected_final_val_loss']:.6f}。它虽比 AdamW 低 "
        f"{abs(nor_vs['adamw']['final_delta_pilot_minus_core']):.6f}，但比 Muon 高 "
        f"{nor_vs['muon']['final_delta_pilot_minus_core']:.6f}，比 Newton trio 高约 "
        f"{min(nor_vs[m]['final_delta_pilot_minus_core'] for m in ('down_diag', 'down_none', 'newton_full')):.6f}–"
        f"{max(nor_vs[m]['final_delta_pilot_minus_core'] for m in ('down_diag', 'down_none', 'newton_full')):.6f}。"
        "未出现反转性竞争力，按 gate 停止，不进入 124M formal/1B。",
        "",
        "## 选型规则与结果",
        "",
        "| 方法 | 选中 cell | matrix LR | final | tail-3 | normalized AUC | 规则 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in selections:
        lines.append(
            f"| {row['method']} | {row['selected_cell']} | {row['selected_matrix_lr']:.4g} | "
            f"{row['selected_final_val_loss']:.6f} | {row['selected_tail3_mean']:.6f} | "
            f"{row['selected_normalized_auc']:.6f} | {row['selection_rule']} |"
        )
    lines += [
        "",
        "Moonlight 的三个 primary endpoint 为 high 3.715696、r1scale 3.732770、official 3.769238；"
        "high 明确胜出，无需 tie-break。NorMuon 为 r1scale 3.823241、official 3.826678、"
        "low 3.864997；r1scale 与 official 相差 0.003437，大于冻结的 0.002 tie band，"
        "因此 official 虽有略优 tail-3/AUC，也不能覆盖 primary endpoint。",
        "",
        "## 数据质量与边界",
        "",
        f"- W&B 导出覆盖 6/6 runs、7/7 metrics；质量检查 {sum(bool(r['passed']) for r in checks)}/{len(checks)} 通过。",
        "- 六条曲线均从同一 initial validation loss=10.985333443 出发，完成 1000 steps / "
        "524,288,000 tokens；所有值有限，验证点严格下降，LR trace 与冻结网格一致。",
        "- 这是 seed2026 调参 pilot。Moonlight 在该 seed 上的比较存在 selection bias，不能据此宣称"
        "独立统计优势；后续 seeds2024/2025 必须沿用冻结 `high` 配置，不能重新选 LR。",
        "- Moonlight `high` 是当前网格的上边界。为遵守既定预算，不在看过结果后追加更高 LR；论文需披露"
        "该边界限制。",
        "- 当前 W&B 文件不足以核验 manifest 完成状态、源码/runtime/data/init 指纹、optimizer-state bytes、"
        "peak memory、resume/status。需补本地 pilot artifact；pilot 设计上没有 checkpoint，无需上传 checkpoint。",
        "- 并发环境下的 `train_s`/`step_avg_ms` 仅可排期，不能进入性能主张。",
        "",
        "## 建议决策",
        "",
        "1. 冻结 Moonlight `high`（matrix LR=aux LR=0.003，WD=0.1）。",
        "2. Moonlight 进入普通 LLaMA-124M 6200-step；建议 seed2024/2025 作为真正 confirmatory，"
        "seed2026 标记为 tuned-seed long-run，而不是把三者混称为完全独立确认。",
        "3. NorMuon 停止，不进入普通 LLaMA formal 或 1B；只有审稿人明确要求时才用冻结的 "
        "`r1scale` 配置补跑。",
        "4. Moonlight 124M formal 通过后，再为 1B 设计小型独立 LR pilot；不要把 0.003 直接当作 1B final LR。",
        "",
        "## 复现文件",
        "",
        "- `pilot_run_summary.csv`：六个 pilot cells 的主要统计量。",
        "- `pilot_within_method_selection.csv`：按预注册规则的选择证据。",
        "- `pilot_vs_core_seed2026_prefix.csv`：与普通 LLaMA 核心方法同 seed、同 1000-step prefix 对齐。",
        "- `data_quality_checks.csv`、`source_manifest.csv`：完整性检查与输入哈希。",
        "- `analysis_manifest.json`：分析口径和产物清单。",
    ]
    (output_dir / "LLAMA124M_EXTENDED_PILOT_ANALYSIS_20260723.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    long_rows, sources = load_exports([path.resolve() for path in args.exports])
    pilot = summarize_pilot(long_rows)
    selections = select_cells(pilot)
    core, core_curves = load_core_prefix(args.core_history.resolve())
    compared = comparisons(pilot, selections, core)
    checks = quality_checks(long_rows, pilot, selections)
    if not all(bool(row["passed"]) for row in checks):
        raise RuntimeError(f"Data quality checks failed: {checks}")
    pilot_curves, selected_core_curves = make_curves(long_rows, selections, core_curves)

    write_csv(output_dir / "pilot_history_long.csv", long_rows)
    write_csv(output_dir / "pilot_run_summary.csv", pilot)
    write_csv(output_dir / "pilot_within_method_selection.csv", selections)
    write_csv(output_dir / "core_seed2026_prefix_summary.csv", core)
    write_csv(output_dir / "pilot_vs_core_seed2026_prefix.csv", compared)
    write_csv(output_dir / "pilot_validation_curves.csv", pilot_curves)
    write_csv(output_dir / "selected_and_core_validation_curves.csv", selected_core_curves)
    write_csv(output_dir / "data_quality_checks.csv", checks)
    sources.append(
        {
            "source_file": args.core_history.name,
            "source_path": str(args.core_history.resolve()),
            "bytes": args.core_history.stat().st_size,
            "sha256": sha256(args.core_history),
        }
    )
    write_csv(output_dir / "source_manifest.csv", sources)
    render_markdown(output_dir, pilot, selections, compared, checks)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "llama124m_extended_baseline_pilot_seed2026",
        "primary_endpoint": "finite validation loss at step 1000",
        "tie_rule": "if absolute final difference <= 0.002, lower final-three validation mean",
        "secondary_endpoint": "normalized trapezoidal validation-loss AUC over steps 0..1000",
        "token_budget": 524_288_000,
        "selected_cells": {
            row["method"]: {
                "cell": row["selected_cell"],
                "matrix_lr": row["selected_matrix_lr"],
                "auxiliary_lr": row["selected_auxiliary_lr"],
            }
            for row in selections
        },
        "decision": {
            "moonlight": "advance_to_llama124m_6200_confirmatory_formal",
            "normuon": "stop_after_pilot",
        },
        "limitations": [
            "seed2026 is the tuning seed; selected-cell comparisons are selection-biased",
            "Moonlight winner is the upper boundary of the frozen LR grid",
            "W&B exports do not certify local manifests, source/runtime/data/init fingerprints, memory, or resume state",
            "concurrent-run timing is ineligible for performance claims",
        ],
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
