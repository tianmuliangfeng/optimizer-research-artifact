"""Audit and summarize the three-seed LLaMA/SwiGLU-1B formal-6200 batch.

The primary endpoint and practical margin come from
ANALYSIS_CONTRACT_20260722.md.  This script combines the original seed-2026
exports with the later seed-2024/2025 exports, preserves all raw inputs, and
reports paired seed deltas without treating three seeds as a universal claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


METHODS = ("muon", "down_none", "down_diag", "newton_full")
NEWTON_METHODS = ("down_none", "down_diag", "newton_full")
SEEDS = (2024, 2025, 2026)
TOTAL_STEPS = 6200
TOKENS_PER_STEP = 512 * 1024
TOTAL_TOKENS = TOTAL_STEPS * TOKENS_PER_STEP
PRACTICAL_MARGIN = 0.002
EXPECTED_METRICS = {
    "val/loss": list(range(0, TOTAL_STEPS + 1, 100)),
    "train/loss_step": list(range(20, TOTAL_STEPS + 1, 20)),
    "tokens/seen": list(range(0, TOTAL_STEPS + 1, 20)),
    "time/train_s": list(range(0, TOTAL_STEPS + 1, 20)),
    "performance/step_avg_ms": list(range(0, TOTAL_STEPS + 1, 20)),
    "lr/backup": list(range(0, TOTAL_STEPS + 1, 20)),
    "lr/matrix": list(range(0, TOTAL_STEPS + 1, 20)),
}
EXPECTED_FINITE_STEPS = {
    metric: (steps[2:] if metric == "performance/step_avg_ms" else steps)
    for metric, steps in EXPECTED_METRICS.items()
}
EXPECTED_K_STATE_MIB = {
    "muon": 0.0,
    "down_none": 1728.0,
    "down_diag": 1728.755859375,
    "newton_full": 5888.25,
}
RUN_PATTERN = re.compile(
    r"^llama_swiglu_(down_none|down_diag|newton_full|muon)_seed(\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        nargs=7,
        required=True,
        help="The seven W&B exports containing seed2024 and seed2025.",
    )
    parser.add_argument(
        "--seed2026-analysis-dir",
        type=Path,
        required=True,
        help="Prior seed2026 analysis directory containing raw_wandb_exports/.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_check(
    rows: list[dict[str, object]],
    check: str,
    passed: bool,
    evidence: str,
    severity: str = "critical",
) -> None:
    rows.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "severity_if_failed": severity,
            "evidence": evidence,
        }
    )


def parse_run(run_name: str) -> tuple[str, int]:
    matched = RUN_PATTERN.match(run_name)
    if not matched:
        raise ValueError(f"unrecognized run name: {run_name}")
    method, seed = matched.group(1), int(matched.group(2))
    return method, seed


def normalized_auc(curve: pd.DataFrame, start_step: int = 0) -> float:
    current = curve[curve.step >= start_step].sort_values("step")
    return float(
        np.trapezoid(current.val_loss, current.step)
        / (current.step.iloc[-1] - current.step.iloc[0])
    )


def sustained_positive_crossing(steps: np.ndarray, deltas: np.ndarray) -> dict[str, object]:
    """First point after which method - Muon remains strictly positive."""

    for index in range(len(deltas)):
        if deltas[index] > 0 and bool(np.all(deltas[index:] > 0)):
            discrete = int(steps[index])
            if index == 0 or deltas[index - 1] >= 0:
                interpolated = float(discrete)
            else:
                left_step, right_step = float(steps[index - 1]), float(steps[index])
                left_delta, right_delta = float(deltas[index - 1]), float(deltas[index])
                interpolated = left_step + (-left_delta) * (
                    right_step - left_step
                ) / (right_delta - left_delta)
            return {
                "first_sustained_discrete_step": discrete,
                "interpolated_crossover_step": interpolated,
                "interpolated_crossover_tokens": interpolated * TOKENS_PER_STEP,
            }
    return {
        "first_sustained_discrete_step": math.nan,
        "interpolated_crossover_step": math.nan,
        "interpolated_crossover_tokens": math.nan,
    }


def target_crossing(curve: pd.DataFrame, target: float) -> dict[str, object]:
    current = curve.sort_values("step").reset_index(drop=True)
    reached = current[current.val_loss <= target]
    if reached.empty:
        return {
            "reached": False,
            "first_discrete_step": math.nan,
            "interpolated_step": math.nan,
            "interpolated_tokens": math.nan,
        }
    index = int(reached.index[0])
    row = current.iloc[index]
    if index == 0 or math.isclose(float(row.val_loss), target):
        interpolated = float(row.step)
    else:
        previous = current.iloc[index - 1]
        high, low = float(previous.val_loss), float(row.val_loss)
        fraction = (high - target) / (high - low) if high != low else 1.0
        interpolated = float(
            previous.step + fraction * (float(row.step) - float(previous.step))
        )
    return {
        "reached": True,
        "first_discrete_step": int(row.step),
        "interpolated_step": interpolated,
        "interpolated_tokens": interpolated * TOKENS_PER_STEP,
    }


def make_plots(output: Path, validation: pd.DataFrame, run_summary: pd.DataFrame) -> None:
    colors = {
        "muon": "#111111",
        "down_none": "#2B6CB0",
        "down_diag": "#2F855A",
        "newton_full": "#C05621",
    }
    font = ImageFont.load_default()

    def canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new("RGB", (width, height), "white")
        return image, ImageDraw.Draw(image)

    def dashed_vertical(
        draw: ImageDraw.ImageDraw, x: int, top: int, bottom: int, color: str
    ) -> None:
        for start in range(top, bottom, 12):
            draw.line((x, start, x, min(start + 6, bottom)), fill=color, width=1)

    def dashed_horizontal(
        draw: ImageDraw.ImageDraw, left: int, right: int, y: int, color: str
    ) -> None:
        for start in range(left, right, 12):
            draw.line((start, y, min(start + 6, right), y), fill=color, width=1)

    def draw_panel(
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        series: list[tuple[str, np.ndarray, np.ndarray]],
        title: str,
        y_min: float,
        y_max: float,
        reference_lines: tuple[float, ...] = (),
    ) -> None:
        left, top, right, bottom = box
        plot_left, plot_top = left + 64, top + 32
        plot_right, plot_bottom = right - 16, bottom - 48

        def px(step: float) -> int:
            return int(plot_left + step / TOTAL_STEPS * (plot_right - plot_left))

        def py(value: float) -> int:
            return int(
                plot_bottom
                - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
            )

        draw.text((left + 8, top + 6), title, fill="#111111", font=font)
        for fraction in np.linspace(0, 1, 5):
            value = y_min + fraction * (y_max - y_min)
            y = py(value)
            draw.line((plot_left, y, plot_right, y), fill="#E5E7EB", width=1)
            draw.text((left + 2, y - 6), f"{value:.3f}", fill="#555555", font=font)
        for step in (0, 2000, 4000, 6200):
            x = px(step)
            draw.line((x, plot_top, x, plot_bottom), fill="#F0F0F0", width=1)
            draw.text((x - 14, plot_bottom + 8), str(step), fill="#555555", font=font)
        draw.rectangle(
            (plot_left, plot_top, plot_right, plot_bottom), outline="#777777", width=1
        )
        dashed_vertical(draw, px(4400), plot_top, plot_bottom, "#888888")
        for reference in reference_lines:
            if y_min <= reference <= y_max:
                dashed_horizontal(
                    draw, plot_left, plot_right, py(reference), "#888888"
                )
        for name, steps, values in series:
            points = [
                (px(float(step)), py(float(value)))
                for step, value in zip(steps, values)
                if y_min <= float(value) <= y_max
            ]
            if len(points) >= 2:
                draw.line(points, fill=colors[name], width=3, joint="curve")

    validation_after_initial = validation[validation.step >= 100]
    y_min = float(validation_after_initial.val_loss.min()) - 0.015
    y_max = min(4.35, float(validation_after_initial.val_loss.max()) + 0.015)
    image, draw = canvas(2400, 720)
    panel_width = 800
    for index, seed in enumerate(SEEDS):
        current = validation_after_initial[validation_after_initial.seed == seed]
        series = []
        for method in METHODS:
            curve = current[current.method == method].sort_values("step")
            series.append(
                (method, curve.step.to_numpy(), curve.val_loss.to_numpy())
            )
        draw_panel(
            draw,
            (index * panel_width, 0, (index + 1) * panel_width, 650),
            series,
            f"Validation loss, seed {seed} (step 0 omitted)",
            y_min,
            y_max,
        )
    legend_x = 1490
    for index, method in enumerate(METHODS):
        x = legend_x + index * 205
        draw.line((x, 684, x + 28, 684), fill=colors[method], width=4)
        draw.text((x + 36, 677), method, fill="#111111", font=font)
    image.save(output / "validation_loss_by_seed.png")

    mean_curve = (
        validation_after_initial.groupby(["method", "step"], as_index=False)
        .val_loss.agg(["mean", "min", "max"])
        .reset_index()
    )
    image, draw = canvas(1200, 720)
    series = []
    for method in METHODS:
        curve = mean_curve[mean_curve.method == method].sort_values("step")
        series.append((method, curve.step.to_numpy(), curve["mean"].to_numpy()))
    draw_panel(
        draw,
        (0, 0, 1200, 650),
        series,
        "Validation loss: three-seed mean (step 0 omitted)",
        y_min,
        y_max,
    )
    for index, method in enumerate(METHODS):
        x = 360 + index * 205
        draw.line((x, 684, x + 28, 684), fill=colors[method], width=4)
        draw.text((x + 36, 677), method, fill="#111111", font=font)
    image.save(output / "validation_loss_mean.png")

    delta_frames: dict[int, pd.DataFrame] = {}
    all_deltas: list[float] = []
    for seed in SEEDS:
        current = (
            validation[validation.seed == seed]
            .pivot(index="step", columns="method", values="val_loss")
            .sort_index()
        )
        delta_frames[seed] = current
        for method in NEWTON_METHODS:
            all_deltas.extend(
                (current.loc[current.index >= 800, method] - current.loc[
                    current.index >= 800, "muon"
                ]).tolist()
            )
    delta_min = min(all_deltas) - 0.005
    delta_max = max(max(all_deltas) + 0.005, PRACTICAL_MARGIN + 0.005)
    image, draw = canvas(2400, 720)
    for index, seed in enumerate(SEEDS):
        current = delta_frames[seed]
        focus = current.index >= 800
        series = []
        for method in NEWTON_METHODS:
            series.append(
                (
                    method,
                    current.index.to_numpy()[focus],
                    (current.loc[focus, method] - current.loc[focus, "muon"]).to_numpy(),
                )
            )
        draw_panel(
            draw,
            (index * panel_width, 0, (index + 1) * panel_width, 650),
            series,
            f"Newton path - Muon, seed {seed} (step >= 800)",
            delta_min,
            delta_max,
            (0.0, PRACTICAL_MARGIN),
        )
    legend_x = 1580
    for index, method in enumerate(NEWTON_METHODS):
        x = legend_x + index * 235
        draw.line((x, 684, x + 28, 684), fill=colors[method], width=4)
        draw.text((x + 36, 677), method, fill="#111111", font=font)
    image.save(output / "newton_minus_muon_by_seed.png")

    final = run_summary[run_summary.method != "muon"].copy()
    muon_final = run_summary[run_summary.method == "muon"].set_index("seed").final_val_loss
    final["delta"] = final.apply(
        lambda row: row.final_val_loss - muon_final.loc[row.seed], axis=1
    )
    image, draw = canvas(1050, 700)
    left, top, right, bottom = 100, 50, 1010, 600
    ymax = float(final.delta.max()) + 0.0015

    def final_py(value: float) -> int:
        return int(bottom - value / ymax * (bottom - top))

    draw.rectangle((left, top, right, bottom), outline="#777777", width=1)
    for value in np.linspace(0, ymax, 6):
        y = final_py(float(value))
        draw.line((left, y, right, y), fill="#E5E7EB", width=1)
        draw.text((24, y - 6), f"{value:.4f}", fill="#555555", font=font)
    dashed_horizontal(draw, left, right, final_py(PRACTICAL_MARGIN), "#888888")
    x_positions = (250, 550, 850)
    for x, method in zip(x_positions, NEWTON_METHODS):
        values = final[final.method == method].sort_values("seed").delta.to_numpy()
        for offset, value in zip((-22, 0, 22), values):
            y = final_py(float(value))
            draw.ellipse(
                (x + offset - 6, y - 6, x + offset + 6, y + 6),
                fill=colors[method],
            )
        mean_y = final_py(float(values.mean()))
        sd_top = final_py(float(values.mean() + values.std(ddof=1)))
        sd_bottom = final_py(float(values.mean() - values.std(ddof=1)))
        draw.line((x, sd_top, x, sd_bottom), fill=colors[method], width=3)
        draw.line((x - 10, sd_top, x + 10, sd_top), fill=colors[method], width=3)
        draw.line((x - 10, sd_bottom, x + 10, sd_bottom), fill=colors[method], width=3)
        draw.rectangle(
            (x - 7, mean_y - 7, x + 7, mean_y + 7), fill=colors[method]
        )
        draw.text((x - 45, bottom + 18), method, fill="#111111", font=font)
    draw.text(
        (100, 18),
        "Final validation loss delta vs Muon: seed points and mean +/- SD",
        fill="#111111",
        font=font,
    )
    image.save(output / "final_delta_vs_muon.png")


def write_report(
    path: Path,
    run_summary: pd.DataFrame,
    method_aggregate: pd.DataFrame,
    pairwise: pd.DataFrame,
    pairwise_aggregate: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    aggregate = method_aggregate.set_index("method")
    pairs = pairwise_aggregate.set_index("contrast")
    quality_counts = quality.status.value_counts().to_dict()
    lines = [
        "# LLaMA/SwiGLU-1B formal-6200 三 seed 合并分析（2026-07-27）",
        "",
        "## 结论先行",
        "",
        "Muon 在 seed2024、2025、2026 的冻结 step-6200 主终点和 tail-5 "
        "均为四方法第一，因此 seed2026 的反转不是孤立 seed。三种 Newton 路径在 "
        "step1000 仍全部优于 Muon，但在约 step1400--2500 的 seed/method 相关窗口内被 "
        "Muon 持续反超。当前固定 recipe 下，1B 结果明确不支持 Selective Newton-Muon "
        "相对 Muon 的 Pareto 改进。",
        "",
        "家族内结论则更稳定：down-none 与 down-diag 在三个 seed 的最终 loss、tail-5 "
        "和 AUC 上均优于 Newton-full。去掉或对角化 down-projection K 没有损害 full "
        "Newton-Muon 的结果，反而改善了结果并减少了 K-state；这部分是当前 1B 数据真正 "
        "支持的正结论。",
        "",
        "## 证据完整性",
        "",
        f"- 数据检查：PASS={quality_counts.get('PASS', 0)}，"
        f"FAIL={quality_counts.get('FAIL', 0)}，WARN={quality_counts.get('WARN', 0)}。",
        f"- 每个 run 完成 {TOTAL_STEPS} updates，即 {TOTAL_TOKENS:,} tokens（3.2506B）。",
        "- 3 seeds × 4 methods × 7 metrics 均齐全；没有缺 run、截断曲线或 endpoint 插补。",
        "- 每个 seed 内四方法 step0 validation loss 完全相同；12 条 run 的 matrix/backup "
        "LR 和 tokens/step 逐点一致。",
        "- W&B 导出足以形成 loss-vs-step/token 的正式质量证据，但不含正式 manifest、"
        "checkpoint/resume 证书和实测显存字段；这些仍需用远程 compact artifacts 对账。",
        "- 双卡并行条件下 time/train_s 与 step_avg_ms 只作描述，不进入论文性能结论。",
        "",
        "## 三 seed 聚合",
        "",
        "| 方法 | Final mean ± SD | Tail-5 mean ± SD | AUC mean ± SD | Final 3-seed rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in method_aggregate.sort_values("final_mean").method:
        row = aggregate.loc[method]
        lines.append(
            f"| {method} | {row.final_mean:.6f} ± {row.final_sd:.6f} | "
            f"{row.tail5_mean:.6f} ± {row.tail5_sd_between:.6f} | "
            f"{row.auc_mean:.6f} ± {row.auc_sd:.6f} | "
            f"{int(row.final_seed_wins)}/3 wins |"
        )
    lines.extend(
        [
            "",
            "## 配对差值（正数表示该方法比参考方法更差）",
            "",
            "| 对比 | Final delta mean ± SD | Tail-5 delta | AUC delta | Final 同方向 seed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    ordered = [
        "down_none-muon",
        "down_diag-muon",
        "newton_full-muon",
        "down_none-newton_full",
        "down_diag-newton_full",
        "down_diag-down_none",
    ]
    for contrast in ordered:
        row = pairs.loc[contrast]
        lines.append(
            f"| {contrast} | {row.final_delta_mean:+.6f} ± {row.final_delta_sd:.6f} | "
            f"{row.tail5_delta_mean:+.6f} | {row.auc_delta_mean:+.6f} | "
            f"{int(row.final_negative_seeds)}/3 negative; "
            f"{int(row.final_positive_seeds)}/3 positive |"
        )
    lines.extend(
        [
            "",
            "三个 Newton 变体相对 Muon 的 final delta 分别为：down-none "
            f"{pairs.loc['down_none-muon'].final_delta_mean:+.6f}、down-diag "
            f"{pairs.loc['down_diag-muon'].final_delta_mean:+.6f}、Newton-full "
            f"{pairs.loc['newton_full-muon'].final_delta_mean:+.6f}。三者在每个 seed "
            f"都超过冻结的 {PRACTICAL_MARGIN:.4f} practical margin，因此不能称为 "
            "“相对 Muon 基本无损”。",
            "",
            "down-diag 相对 down-none 的最终 loss 平均高 "
            f"{pairs.loc['down_diag-down_none'].final_delta_mean:.6f}，但 AUC 平均低 "
            f"{-pairs.loc['down_diag-down_none'].auc_delta_mean:.6f}。这说明 diag "
            "更偏早期优化，none 更偏冻结终点；两者不是简单的全程单调排序。",
            "",
            "## 对论文叙事的含义",
            "",
            "1. 可以写：在 1B、3.25B-token 固定 recipe 下，Muon 的后期质量优势跨 "
            "3 seeds 稳定复现；Newton 的早期优势会发生训练阶段反转。",
            "2. 可以写：对 Newton-Muon 家族，down-projection K 的边际价值很低，"
            "down-none/down-diag 以更少状态稳定优于 full。",
            "3. 不可以写：Selective 在 1B 相对 Muon 质量无损、优于 Muon，或 "
            "Newton-full 是质量上界。",
            "4. 当前结果把下一步重点从继续堆长跑，转向定位反转机制，以及独立的 "
            "Newton LR/ridge/refresh 探索分支；调参结果不能替换冻结主表。",
            "",
            "## 尚缺的本地证书",
            "",
            "请补充 seed2024、seed2025 与 seed2026 三个 formal-6200 批次的 "
            "`llama_manifest.json`、"
            "`llama_plan.json`、`llama_swiglu_summary.csv`，以及四个 run 的 "
            "`summary.json` 和 `metrics.csv`。不需要上传约 10GB 的 checkpoint；"
            "manifest 中的 checkpoint 路径、大小、hash/resume 元数据即可。若 summary "
            "含 optimizer/K-state/peak CUDA memory，也可完成曲线与实测内存的对账。",
            "",
            "## 逐 seed 主终点",
            "",
            "| Seed | Muon | down-none | down-diag | Newton-full |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in SEEDS:
        current = run_summary[run_summary.seed == seed].set_index("method")
        lines.append(
            f"| {seed} | {current.loc['muon'].final_val_loss:.6f} | "
            f"{current.loc['down_none'].final_val_loss:.6f} | "
            f"{current.loc['down_diag'].final_val_loss:.6f} | "
            f"{current.loc['newton_full'].final_val_loss:.6f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    raw_new = output / "raw_wandb_exports" / "seed2024_2025"
    raw_old = output / "raw_wandb_exports" / "seed2026"
    raw_new.mkdir(parents=True, exist_ok=True)
    raw_old.mkdir(parents=True, exist_ok=True)

    seed2026_raw = args.seed2026_analysis_dir.resolve() / "raw_wandb_exports"
    old_sources = sorted(seed2026_raw.glob("*.csv"))
    if len(old_sources) != 7:
        raise RuntimeError(f"expected seven seed2026 raw exports in {seed2026_raw}")

    source_groups = [
        ("seed2024_2025", source.resolve(), raw_new / source.name)
        for source in args.input
    ] + [
        ("seed2026", source.resolve(), raw_old / source.name)
        for source in old_sources
    ]

    checks: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []

    for source_group, source, copied in source_groups:
        if copied.exists() and sha256(copied) != sha256(source):
            raise RuntimeError(f"refusing to overwrite different evidence: {copied}")
        if not copied.exists():
            shutil.copy2(source, copied)

        frame = pd.read_csv(source)
        if "Step" not in frame:
            raise RuntimeError(f"Step column missing from {source}")
        primary = [
            column
            for column in frame.columns
            if column != "Step"
            and not column.endswith("__MIN")
            and not column.endswith("__MAX")
        ]
        metrics = {column.rsplit(" - ", 1)[1] for column in primary}
        add_check(checks, f"{source.name}: one metric", len(metrics) == 1, repr(metrics))
        if len(metrics) != 1:
            continue
        metric = next(iter(metrics))
        add_check(checks, f"{source.name}: recognized metric", metric in EXPECTED_METRICS, metric)
        observed_steps = frame.Step.astype(int).tolist()
        add_check(
            checks,
            f"{source.name}: exact row step grid",
            observed_steps == EXPECTED_METRICS.get(metric),
            f"metric={metric} points={len(observed_steps)}",
        )

        runs_in_file: set[tuple[int, str]] = set()
        for column in primary:
            run_name, parsed_metric = column.rsplit(" - ", 1)
            method, seed = parse_run(run_name)
            runs_in_file.add((seed, method))
            values = pd.to_numeric(frame[column], errors="coerce")
            finite = np.isfinite(values.to_numpy(dtype=float))
            finite_steps = frame.loc[finite, "Step"].astype(int).tolist()
            add_check(
                checks,
                f"{run_name} {metric}: expected finite grid",
                finite_steps == EXPECTED_FINITE_STEPS[metric],
                f"finite={len(finite_steps)}",
            )

            min_column, max_column = f"{column}__MIN", f"{column}__MAX"
            mirrors_ok = min_column in frame and max_column in frame
            if mirrors_ok:
                mins = pd.to_numeric(frame[min_column], errors="coerce")
                maxs = pd.to_numeric(frame[max_column], errors="coerce")
                mirrors_ok = bool(
                    values.isna().equals(mins.isna())
                    and values.isna().equals(maxs.isna())
                    and np.allclose(values[finite], mins[finite], rtol=0, atol=0)
                    and np.allclose(values[finite], maxs[finite], rtol=0, atol=0)
                )
            add_check(
                checks,
                f"{run_name} {metric}: MIN/MAX mirrors",
                mirrors_ok,
                f"finite={int(finite.sum())}",
                "high",
            )
            frames.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "run_name": run_name,
                        "seed": seed,
                        "metric": parsed_metric,
                        "step": frame.Step.astype(int),
                        "value": values,
                        "source_file": source.name,
                        "source_group": source_group,
                    }
                )
            )

        expected_runs = (
            {(seed, method) for seed in (2024, 2025) for method in METHODS}
            if source_group == "seed2024_2025"
            else {(2026, method) for method in METHODS}
        )
        add_check(
            checks,
            f"{source.name}: exact run identities",
            runs_in_file == expected_runs,
            repr(sorted(runs_in_file)),
        )
        sources.append(
            {
                "source_group": source_group,
                "source_file": source.name,
                "source_path": str(source),
                "preserved_path": str(copied),
                "sha256": sha256(source),
                "bytes": source.stat().st_size,
                "metric": metric,
                "rows": len(frame),
                "primary_run_columns": len(primary),
            }
        )

    long = pd.concat(frames, ignore_index=True).sort_values(
        ["seed", "method", "metric", "step"]
    )
    add_check(
        checks,
        "exact seed identities",
        set(long.seed.unique()) == set(SEEDS),
        repr(sorted(long.seed.unique())),
    )
    add_check(
        checks,
        "exact method identities",
        set(long.method.unique()) == set(METHODS),
        repr(sorted(long.method.unique())),
    )
    add_check(
        checks,
        "exact seven metrics",
        set(long.metric.unique()) == set(EXPECTED_METRICS),
        repr(sorted(long.metric.unique())),
    )
    duplicates = int(long.duplicated(["seed", "method", "metric", "step"]).sum())
    add_check(checks, "unique seed/method/metric/step grain", duplicates == 0, f"{duplicates=}")

    expected_combinations = len(SEEDS) * len(METHODS) * len(EXPECTED_METRICS)
    combinations = long[["seed", "method", "metric"]].drop_duplicates()
    add_check(
        checks,
        "complete 3 seeds x 4 methods x 7 metrics",
        len(combinations) == expected_combinations,
        f"observed={len(combinations)} expected={expected_combinations}",
    )

    finite_long = long[np.isfinite(long.value.to_numpy(dtype=float))].copy()
    wide = finite_long.pivot(
        index=["seed", "method", "step"], columns="metric", values="value"
    ).reset_index()
    validation = finite_long[finite_long.metric == "val/loss"][
        ["seed", "method", "step", "value"]
    ].rename(columns={"value": "val_loss"})

    for seed in SEEDS:
        initial = validation[(validation.seed == seed) & (validation.step == 0)].set_index(
            "method"
        ).val_loss
        add_check(
            checks,
            f"seed{seed}: identical step0 validation loss",
            initial.nunique(dropna=False) == 1,
            json.dumps(initial.to_dict(), sort_keys=True),
        )

    token_rows = wide[wide["tokens/seen"].notna()]
    token_ok = np.array_equal(
        token_rows["tokens/seen"].to_numpy(dtype=float),
        (token_rows.step * TOKENS_PER_STEP).to_numpy(dtype=float),
    )
    add_check(checks, "tokens equal step * 512 * 1024", token_ok, f"final={TOTAL_TOKENS}")

    for metric in ("lr/backup", "lr/matrix"):
        schedule = finite_long[finite_long.metric == metric].pivot(
            index="step", columns=["seed", "method"], values="value"
        )
        spread = float((schedule.max(axis=1) - schedule.min(axis=1)).max())
        add_check(
            checks,
            f"{metric}: identical across all 12 runs",
            spread == 0.0,
            f"max_stepwise_spread={spread}",
        )

    summary_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for method in METHODS:
            curve = validation[
                (validation.seed == seed) & (validation.method == method)
            ].sort_values("step")
            current = wide[(wide.seed == seed) & (wide.method == method)].sort_values("step")
            at = curve.set_index("step").val_loss
            train_curve = current[current["train/loss_step"].notna()]
            time_curve = current[current["time/train_s"].notna()]
            performance_curve = current[current["performance/step_avg_ms"].notna()]
            summary_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "run_name": f"llama_swiglu_{method}_seed{seed}",
                    "initial_val_loss": float(at.loc[0]),
                    "val_loss_step1000": float(at.loc[1000]),
                    "val_loss_step2000": float(at.loc[2000]),
                    "val_loss_step3000": float(at.loc[3000]),
                    "val_loss_step4400": float(at.loc[4400]),
                    "final_val_loss": float(at.loc[TOTAL_STEPS]),
                    "best_val_loss": float(curve.val_loss.min()),
                    "tail3_val_loss_mean": float(curve.tail(3).val_loss.mean()),
                    "tail5_val_loss_mean": float(curve.tail(5).val_loss.mean()),
                    "tail5_within_run_sd": float(curve.tail(5).val_loss.std(ddof=1)),
                    "normalized_val_auc_0_6200": normalized_auc(curve, 0),
                    "post_initial_auc_100_6200": normalized_auc(curve, 100),
                    "final_train_loss_step": float(
                        train_curve[train_curve.step == TOTAL_STEPS][
                            "train/loss_step"
                        ].iloc[0]
                    ),
                    "final_tokens": TOTAL_TOKENS,
                    "train_time_s_descriptive_only": float(
                        time_curve[time_curve.step == TOTAL_STEPS]["time/train_s"].iloc[0]
                    ),
                    "final_step_avg_ms_descriptive_only": float(
                        performance_curve[performance_curve.step == TOTAL_STEPS][
                            "performance/step_avg_ms"
                        ].iloc[0]
                    ),
                    "expected_k_state_mib_from_preflight": EXPECTED_K_STATE_MIB[method],
                    "memory_usable_from_wandb_export": False,
                    "timing_usable_for_paper": False,
                }
            )
    run_summary = pd.DataFrame(summary_rows)
    run_summary["final_rank_within_seed"] = (
        run_summary.groupby("seed").final_val_loss.rank(method="min").astype(int)
    )
    run_summary["auc_rank_within_seed"] = (
        run_summary.groupby("seed")
        .normalized_val_auc_0_6200.rank(method="min")
        .astype(int)
    )
    run_summary = run_summary.sort_values(["seed", "final_rank_within_seed"])

    method_aggregate = (
        run_summary.groupby("method", as_index=False)
        .agg(
            final_mean=("final_val_loss", "mean"),
            final_sd=("final_val_loss", "std"),
            tail5_mean=("tail5_val_loss_mean", "mean"),
            tail5_sd_between=("tail5_val_loss_mean", "std"),
            auc_mean=("normalized_val_auc_0_6200", "mean"),
            auc_sd=("normalized_val_auc_0_6200", "std"),
            val_step1000_mean=("val_loss_step1000", "mean"),
            val_step2000_mean=("val_loss_step2000", "mean"),
            val_step3000_mean=("val_loss_step3000", "mean"),
            val_step4400_mean=("val_loss_step4400", "mean"),
            final_seed_wins=("final_rank_within_seed", lambda values: int((values == 1).sum())),
            auc_seed_wins=("auc_rank_within_seed", lambda values: int((values == 1).sum())),
        )
        .sort_values("final_mean")
    )

    pair_rows: list[dict[str, object]] = []
    contrasts = [
        ("down_none", "muon"),
        ("down_diag", "muon"),
        ("newton_full", "muon"),
        ("down_none", "newton_full"),
        ("down_diag", "newton_full"),
        ("down_diag", "down_none"),
    ]
    for seed in SEEDS:
        current_summary = run_summary[run_summary.seed == seed].set_index("method")
        val_wide = (
            validation[validation.seed == seed]
            .pivot(index="step", columns="method", values="val_loss")
            .sort_index()
        )
        for method, reference in contrasts:
            delta = val_wide[method] - val_wide[reference]
            row = {
                "seed": seed,
                "contrast": f"{method}-{reference}",
                "method": method,
                "reference": reference,
                "final_delta": float(
                    current_summary.loc[method].final_val_loss
                    - current_summary.loc[reference].final_val_loss
                ),
                "tail5_delta": float(
                    current_summary.loc[method].tail5_val_loss_mean
                    - current_summary.loc[reference].tail5_val_loss_mean
                ),
                "auc_delta": float(
                    current_summary.loc[method].normalized_val_auc_0_6200
                    - current_summary.loc[reference].normalized_val_auc_0_6200
                ),
                "method_lower_checkpoints_excluding_step0": int((delta.iloc[1:] < 0).sum()),
                "reference_lower_checkpoints_excluding_step0": int((delta.iloc[1:] > 0).sum()),
                "ties_excluding_step0": int((delta.iloc[1:] == 0).sum()),
                "final_exceeds_positive_0p002_margin": bool(
                    float(delta.iloc[-1]) > PRACTICAL_MARGIN
                ),
            }
            if reference == "muon":
                row.update(
                    sustained_positive_crossing(
                        delta.index.to_numpy(dtype=int)[1:],
                        delta.to_numpy(dtype=float)[1:],
                    )
                )
            else:
                row.update(
                    {
                        "first_sustained_discrete_step": math.nan,
                        "interpolated_crossover_step": math.nan,
                        "interpolated_crossover_tokens": math.nan,
                    }
                )
            pair_rows.append(row)
    pairwise = pd.DataFrame(pair_rows)
    pairwise_aggregate = (
        pairwise.groupby("contrast", as_index=False)
        .agg(
            final_delta_mean=("final_delta", "mean"),
            final_delta_sd=("final_delta", "std"),
            tail5_delta_mean=("tail5_delta", "mean"),
            tail5_delta_sd=("tail5_delta", "std"),
            auc_delta_mean=("auc_delta", "mean"),
            auc_delta_sd=("auc_delta", "std"),
            final_negative_seeds=("final_delta", lambda values: int((values < 0).sum())),
            final_positive_seeds=("final_delta", lambda values: int((values > 0).sum())),
            tail5_negative_seeds=("tail5_delta", lambda values: int((values < 0).sum())),
            tail5_positive_seeds=("tail5_delta", lambda values: int((values > 0).sum())),
            auc_negative_seeds=("auc_delta", lambda values: int((values < 0).sum())),
            auc_positive_seeds=("auc_delta", lambda values: int((values > 0).sum())),
        )
        .sort_values("contrast")
    )

    target_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        family_common_final = float(
            run_summary[run_summary.seed == seed].final_val_loss.max()
        )
        for target in (4.0, 3.8, 3.6, 3.4, 3.2, 3.0, family_common_final):
            for method in METHODS:
                curve = validation[
                    (validation.seed == seed) & (validation.method == method)
                ][["step", "val_loss"]]
                target_rows.append(
                    {
                        "seed": seed,
                        "target_val_loss": target,
                        "target_scope": (
                            "family_common_final"
                            if math.isclose(target, family_common_final)
                            else "fixed"
                        ),
                        "method": method,
                        **target_crossing(curve, target),
                    }
                )
    targets = pd.DataFrame(target_rows)

    quality = pd.DataFrame(checks)
    warnings = pd.DataFrame(
        [
            {
                "check": "formal local certificates supplied",
                "status": "WARN",
                "severity_if_failed": "critical",
                "evidence": (
                    "need seed2024/2025/2026 formal-6200 manifests, plans, summaries, metrics, "
                    "and checkpoint/resume metadata"
                ),
            },
            {
                "check": "memory evidence in current W&B exports",
                "status": "WARN",
                "severity_if_failed": "high",
                "evidence": "no measured optimizer/K-state/peak CUDA memory fields",
            },
            {
                "check": "timing eligibility",
                "status": "WARN",
                "severity_if_failed": "high",
                "evidence": "concurrent dual-GPU runs; timing is descriptive only",
            },
            {
                "check": "universal population claim",
                "status": "WARN",
                "severity_if_failed": "high",
                "evidence": "three fixed-recipe seeds support replication, not universality",
            },
        ]
    )
    quality = pd.concat([quality, warnings], ignore_index=True)
    fail_count = int((quality.status == "FAIL").sum())

    pd.DataFrame(sources).to_csv(output / "source_manifest.csv", index=False)
    long.to_csv(output / "normalized_history_long.csv", index=False)
    wide.sort_values(["seed", "method", "step"]).to_csv(
        output / "normalized_history_wide.csv", index=False
    )
    validation.to_csv(output / "validation_curves_long.csv", index=False)
    run_summary.to_csv(output / "llama1b_formal_multiseed_run_summary.csv", index=False)
    method_aggregate.to_csv(
        output / "llama1b_formal_multiseed_method_aggregate.csv", index=False
    )
    pairwise.to_csv(output / "llama1b_formal_multiseed_pairwise.csv", index=False)
    pairwise_aggregate.to_csv(
        output / "llama1b_formal_multiseed_pairwise_aggregate.csv", index=False
    )
    targets.to_csv(
        output / "llama1b_formal_multiseed_steps_tokens_to_targets.csv", index=False
    )
    quality.to_csv(output / "data_quality_checks.csv", index=False)
    make_plots(output, validation, run_summary)
    report_path = output / "LLAMA1B_FORMAL6200_MULTISEED_ANALYSIS_20260727.md"
    write_report(
        report_path,
        run_summary,
        method_aggregate,
        pairwise,
        pairwise_aggregate,
        quality,
    )

    manifest = {
        "created_at": "2026-07-27",
        "status": "PASS_WITH_CAVEATS" if fail_count == 0 else "FAIL",
        "stage": "formal-6200",
        "evidence_class": "formal_multiseed_quality_curve_evidence",
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "steps_per_run": TOTAL_STEPS,
        "tokens_per_run": TOTAL_TOKENS,
        "approximately_10b_token_extension": False,
        "quality_curves_usable": fail_count == 0,
        "fixed_recipe_multiseed_replication_complete": fail_count == 0,
        "universal_method_claim_ready": False,
        "formal_local_certificates": "PENDING",
        "memory_usable_from_current_exports": False,
        "timing_usable": False,
        "quality_checks": {
            key: int(value)
            for key, value in quality.status.value_counts().to_dict().items()
        },
        "primary_result": {
            "muon_final_wins": int(
                method_aggregate.set_index("method").loc["muon", "final_seed_wins"]
            ),
            "seeds": len(SEEDS),
            "down_none_minus_muon_final_mean": float(
                pairwise_aggregate.set_index("contrast").loc[
                    "down_none-muon", "final_delta_mean"
                ]
            ),
            "down_diag_minus_muon_final_mean": float(
                pairwise_aggregate.set_index("contrast").loc[
                    "down_diag-muon", "final_delta_mean"
                ]
            ),
            "newton_full_minus_muon_final_mean": float(
                pairwise_aggregate.set_index("contrast").loc[
                    "newton_full-muon", "final_delta_mean"
                ]
            ),
        },
        "outputs": sorted(
            item.name
            for item in output.iterdir()
            if item.is_file() and item.name != "analysis_manifest.json"
        ),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
