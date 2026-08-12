"""Audit the crossed R1 LR cells and combine them with the formal R1 anchor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("diag", "muon")
SEEDS = (2024, 2025, 2026)
EXPECTED_METRICS = {
    "val/loss": list(range(0, 6201, 100)),
    "train/loss_step": list(range(20, 6201, 20)),
    "time/train_s": list(range(0, 6201, 20)),
    "performance/step_avg_ms": list(range(40, 6201, 20)),
    "lr/adamw": list(range(0, 6201, 20)),
    "lr/matrix": list(range(0, 6201, 20)),
    "memory/peak_allocated_mib": [6200],
    "memory/k_state_mib": [6200],
    "memory/optimizer_state_mib": [6200],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formal-r1-analysis", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_run_name(run_name: str) -> tuple[str, int]:
    match = re.search(r"_r1_lr_cross_(diag|muon)_seed(\d+)_", run_name)
    if not match:
        raise ValueError(f"Unrecognized R1 LR-cross run name: {run_name}")
    return match.group(1), int(match.group(2))


def first_crossing(curve: pd.DataFrame, target: float) -> dict[str, float | int | bool]:
    curve = curve.sort_values("step").reset_index(drop=True)
    reached = curve[curve["val_loss"] <= target]
    if reached.empty:
        return {
            "reached": False,
            "first_discrete_step": math.nan,
            "interpolated_step": math.nan,
        }
    index = int(reached.index[0])
    current = curve.iloc[index]
    if index == 0 or float(current["val_loss"]) == target:
        return {
            "reached": True,
            "first_discrete_step": int(current["step"]),
            "interpolated_step": float(current["step"]),
        }
    previous = curve.iloc[index - 1]
    high = float(previous["val_loss"])
    low = float(current["val_loss"])
    fraction = (high - target) / (high - low) if high != low else 1.0
    return {
        "reached": True,
        "first_discrete_step": int(current["step"]),
        "interpolated_step": float(previous["step"] + fraction * (current["step"] - previous["step"])),
    }


def run_summary(wide: pd.DataFrame, run_names: dict[tuple[str, int], str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for seed in SEEDS:
            part = wide[(wide["method"] == method) & (wide["seed"] == seed)].sort_values("step")
            curve = part[part["val/loss"].notna()].copy()
            last = part.iloc[-1]
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "lr_level": "low_0.9x" if method == "diag" else "high_1.0x",
                    "run_name": run_names[(method, seed)],
                    "initial_val_loss": float(curve.iloc[0]["val/loss"]),
                    "final_val_loss": float(curve.iloc[-1]["val/loss"]),
                    "best_val_loss": float(curve["val/loss"].min()),
                    "tail3_val_loss_mean": float(curve.tail(3)["val/loss"].mean()),
                    "tail5_val_loss_mean": float(curve.tail(5)["val/loss"].mean()),
                    "normalized_val_auc": float(np.trapezoid(curve["val/loss"], curve["step"]) / 6200.0),
                    "final_train_loss_step": float(last["train/loss_step"]),
                    "train_time_s_descriptive_only": float(last["time/train_s"]),
                    "final_step_avg_ms_descriptive_only": float(last["performance/step_avg_ms"]),
                    "max_adamw_lr": float(part["lr/adamw"].max()),
                    "max_matrix_lr": float(part["lr/matrix"].max()),
                    "peak_memory_mib": float(last["memory/peak_allocated_mib"]),
                    "k_state_mib": float(last["memory/k_state_mib"]),
                    "optimizer_state_mib": float(last["memory/optimizer_state_mib"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["seed", "method"]).reset_index(drop=True)


def paired_crossed_summary(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["final_val_loss", "tail3_val_loss_mean", "tail5_val_loss_mean", "normalized_val_auc"]
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        by_method = summary[summary["seed"] == seed].set_index("method")
        row: dict[str, object] = {"seed": seed}
        for metric in metrics:
            row[f"diag_low_minus_muon_high_{metric}"] = float(by_method.loc["diag", metric] - by_method.loc["muon", metric])
        row["winner_final_loss"] = "diag_low" if row["diag_low_minus_muon_high_final_val_loss"] < 0 else "muon_high"
        rows.append(row)
    paired = pd.DataFrame(rows)
    aggregate_rows: list[dict[str, object]] = []
    for metric in metrics:
        column = f"diag_low_minus_muon_high_{metric}"
        values = paired[column].to_numpy(dtype=float)
        n = len(values)
        std = float(np.std(values, ddof=1))
        sem = std / math.sqrt(n)
        # Student-t 97.5% critical value for df=2. This interval is deliberately
        # conservative and is reported as a caveat, not as a superiority test.
        t_critical = 4.302652729911275
        mean = float(np.mean(values))
        aggregate_rows.append(
            {
                "metric": metric,
                "contrast": "diag_low_minus_muon_high",
                "n_paired_seeds": n,
                "mean_delta": mean,
                "sample_std": std,
                "min_delta": float(np.min(values)),
                "max_delta": float(np.max(values)),
                "wins_diag": int(np.sum(values < 0)),
                "wins_muon": int(np.sum(values > 0)),
                "two_sided_exact_sign_p": min(1.0, 2.0 * (0.5**n)) if np.all(values < 0) or np.all(values > 0) else 1.0,
                "t95_low": mean - t_critical * sem,
                "t95_high": mean + t_critical * sem,
            }
        )
    return paired, pd.DataFrame(aggregate_rows)


def factorial_seed2026(
    crossed: pd.DataFrame,
    formal_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["final_val_loss", "tail3_val_loss_mean", "tail5_val_loss_mean", "normalized_val_auc"]
    formal = formal_summary[formal_summary["method"].isin(METHODS)].copy()
    formal["lr_level"] = formal["method"].map({"muon": "low_0.9x", "diag": "high_1.0x"})
    formal["evidence_source"] = "formal_r1_seed2026"
    cross = crossed[crossed["seed"] == 2026].copy()
    cross["evidence_source"] = "lr_cross_seed2026"
    keep = ["method", "seed", "lr_level", "evidence_source", *metrics]
    cells = pd.concat([formal[keep], cross[keep]], ignore_index=True).sort_values(["method", "lr_level"])
    if len(cells) != 4 or cells.duplicated(["method", "lr_level"]).any():
        raise RuntimeError("Seed-2026 factorial requires exactly four unique method-LR cells")
    lookup = cells.set_index(["method", "lr_level"])
    rows: list[dict[str, object]] = []
    for metric in metrics:
        dl = float(lookup.loc[("diag", "low_0.9x"), metric])
        dh = float(lookup.loc[("diag", "high_1.0x"), metric])
        ml = float(lookup.loc[("muon", "low_0.9x"), metric])
        mh = float(lookup.loc[("muon", "high_1.0x"), metric])
        method_low = dl - ml
        method_high = dh - mh
        diag_lr = dh - dl
        muon_lr = mh - ml
        method_main = (method_low + method_high) / 2.0
        lr_main = (diag_lr + muon_lr) / 2.0
        rows.append(
            {
                "metric": metric,
                "diag_low": dl,
                "diag_high": dh,
                "muon_low": ml,
                "muon_high": mh,
                "diag_minus_muon_at_low_lr": method_low,
                "diag_minus_muon_at_high_lr": method_high,
                "method_main_effect_diag_minus_muon": method_main,
                "high_minus_low_within_diag": diag_lr,
                "high_minus_low_within_muon": muon_lr,
                "lr_main_effect_high_minus_low": lr_main,
                "method_by_lr_interaction": method_high - method_low,
                "abs_method_to_lr_main_effect_ratio": abs(method_main / lr_main) if lr_main else math.inf,
            }
        )
    return cells.reset_index(drop=True), pd.DataFrame(rows)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    effects: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    final_agg = aggregate.set_index("metric").loc["final_val_loss"]
    final_effect = effects.set_index("metric").loc["final_val_loss"]
    quality_counts = quality["status"].value_counts().to_dict()
    by_seed = summary.pivot(index="seed", columns="method", values="final_val_loss")
    lines = [
        "# R1 学习率交叉实验分析（2026-07-21）",
        "",
        "## 证据范围",
        "",
        "- 本批 6 个 run 是三种子下的两个缺失交叉单元：`diag@0.9x LR` 与 `Muon@1.0x LR`。",
        "- 与既有 formal R1 seed2026 的 `diag@1.0x`、`Muon@0.9x` 合并后，可形成 seed2026 的完整 2×2。",
        "- 目前本地尚无 formal R1 seeds2024/2025 的 W&B 导出，因此三种子的完整 2×2 尚不能最终统计。",
        f"- 数据审计：PASS={quality_counts.get('PASS', 0)}，WARN={quality_counts.get('WARN', 0)}，FAIL={quality_counts.get('FAIL', 0)}。",
        "",
        "## 三种子交叉单元结果",
        "",
        "| seed | diag@0.9x final | Muon@1.0x final | diag - Muon |",
        "|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        delta = float(by_seed.loc[seed, "diag"] - by_seed.loc[seed, "muon"])
        lines.append(f"| {seed} | {by_seed.loc[seed, 'diag']:.4f} | {by_seed.loc[seed, 'muon']:.4f} | {delta:+.4f} |")
    lines.extend(
        [
            "",
            f"三种子配对均由 diag@0.9x 获胜；final-loss 平均差为 {final_agg['mean_delta']:+.6f}，样本标准差 {final_agg['sample_std']:.6f}。",
            "",
            "## seed2026 完整 2×2",
            "",
            f"- 方法主效应（diag - Muon）：{final_effect['method_main_effect_diag_minus_muon']:+.6f}。",
            f"- 学习率主效应（1.0x - 0.9x）：{final_effect['lr_main_effect_high_minus_low']:+.6f}。",
            f"- 方法×学习率交互：{final_effect['method_by_lr_interaction']:+.6f}。",
            f"- 方法主效应绝对值约为学习率主效应的 {final_effect['abs_method_to_lr_main_effect_ratio']:.1f} 倍。",
            "",
            "## 与既有证据的关系",
            "",
            "- 60-run 本地统一参考管线中，Muon 在 OWT/WikiText 12L 的三种子均值最优，而 diag 在 OWT 18L、OWT 24L、WikiText 24L 最优；它说明方法排序存在训练 regime 交互，但使用的是旧本地 Muon，不能与本实验合并统计。",
            "- 官方 R0 中 block4 为 3.2638、Muon 为 3.2770，方向与 R1 一致，但 R0 同时存在不同初始化和不同 LR，只能视为官方 recipe 复现。",
            "- R1 seed2026 固定初始化后，diag@1.0x 为 3.2621、Muon@0.9x 为 3.2771；本次把 LR 反过来仍得到相同方向，因此消除了这组比较中最重要的 LR 替代解释。",
            "",
            "## 结论边界",
            "",
            "seed2026 的完整 2×2 表明，原 R1 中 diag 对 Muon 的优势不是由 10% 学习率差异制造的；把学习率反过来偏向 Muon 后，diag 仍然领先，而且 LR 主效应远小于方法主效应。三种子交叉单元的方向也完全一致。",
            "",
            "但论文中的最终三种子 factorial 主表还需要 formal R1 seeds2024/2025 的相同九项 W&B 导出，并用本地 manifest/summary 核对 runtime、source hash、resume_count 与完成状态。当前 wall-clock 只作运维描述，不进入性能结论。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    raw_dir = out / "raw_wandb_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    long_frames: list[pd.DataFrame] = []
    seen_metrics: set[str] = set()
    run_names: dict[tuple[str, int], str] = {}

    for source_arg in args.input:
        source = source_arg.resolve()
        copied = raw_dir / source.name
        if copied.exists() and sha256(copied) != sha256(source):
            raise RuntimeError(f"Refusing to overwrite different evidence: {copied}")
        if not copied.exists():
            shutil.copy2(source, copied)
        frame = pd.read_csv(source)
        primary = [c for c in frame.columns if c != "Step" and not c.endswith("__MIN") and not c.endswith("__MAX")]
        metrics = {c.rsplit(" - ", 1)[1] for c in primary}
        if len(primary) != 6 or len(metrics) != 1:
            raise RuntimeError(f"Expected six runs and one metric in {source.name}")
        metric = next(iter(metrics))
        seen_metrics.add(metric)
        expected_steps = EXPECTED_METRICS.get(metric)
        manifest_rows.append(
            {
                "source_file": source.name,
                "original_path": str(source),
                "preserved_path": str(copied),
                "metric": metric,
                "rows": len(frame),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )
        quality_rows.extend(
            [
                {
                    "check": f"{metric}: row count",
                    "status": "PASS" if expected_steps is not None and len(frame) == len(expected_steps) else "FAIL",
                    "details": f"observed={len(frame)}, expected={len(expected_steps) if expected_steps else 'unknown'}",
                },
                {
                    "check": f"{metric}: exact step grid",
                    "status": "PASS" if expected_steps is not None and frame["Step"].astype(int).tolist() == expected_steps else "FAIL",
                    "details": f"first={frame['Step'].head(3).tolist()}, last={frame['Step'].tail(3).tolist()}",
                },
            ]
        )
        coverage: set[tuple[str, int]] = set()
        for column in primary:
            run_name, parsed_metric = column.rsplit(" - ", 1)
            method, seed = parse_run_name(run_name)
            coverage.add((method, seed))
            old = run_names.setdefault((method, seed), run_name)
            if old != run_name:
                raise RuntimeError(f"Multiple run names for {(method, seed)}")
            values = pd.to_numeric(frame[column], errors="coerce")
            min_values = pd.to_numeric(frame[f"{column}__MIN"], errors="coerce")
            max_values = pd.to_numeric(frame[f"{column}__MAX"], errors="coerce")
            finite = np.isfinite(values.to_numpy(dtype=float))
            mirrors = np.allclose(values[finite], min_values[finite], rtol=0, atol=0) and np.allclose(values[finite], max_values[finite], rtol=0, atol=0)
            quality_rows.extend(
                [
                    {"check": f"{method} seed{seed} {metric}: finite", "status": "PASS" if finite.all() else "FAIL", "details": f"nonfinite={int((~finite).sum())}"},
                    {"check": f"{method} seed{seed} {metric}: MIN/MAX mirrors", "status": "PASS" if mirrors else "FAIL", "details": f"points={int(finite.sum())}"},
                ]
            )
            long_frames.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "seed": seed,
                        "run_name": run_name,
                        "metric": parsed_metric,
                        "step": frame["Step"].astype(int),
                        "value": values,
                        "source_file": source.name,
                    }
                )
            )
        quality_rows.append(
            {
                "check": f"{metric}: exact method-seed coverage",
                "status": "PASS" if coverage == {(m, s) for m in METHODS for s in SEEDS} else "FAIL",
                "details": json.dumps(sorted(coverage)),
            }
        )

    quality_rows.append(
        {
            "check": "exact nine expected metric exports",
            "status": "PASS" if seen_metrics == set(EXPECTED_METRICS) else "FAIL",
            "details": f"missing={sorted(set(EXPECTED_METRICS)-seen_metrics)}, extra={sorted(seen_metrics-set(EXPECTED_METRICS))}",
        }
    )
    long = pd.concat(long_frames, ignore_index=True).sort_values(["seed", "method", "metric", "step"])
    duplicates = int(long.duplicated(["method", "seed", "metric", "step"]).sum())
    quality_rows.append({"check": "unique method-seed-metric-step keys", "status": "PASS" if duplicates == 0 else "FAIL", "details": f"duplicates={duplicates}"})
    wide = long.pivot(index=["method", "seed", "step"], columns="metric", values="value").reset_index()

    for seed in SEEDS:
        initial = wide[(wide["seed"] == seed) & (wide["step"] == 0)].set_index("method")["val/loss"]
        quality_rows.append(
            {
                "check": f"seed{seed}: identical initial validation loss",
                "status": "PASS" if initial.nunique(dropna=False) == 1 else "FAIL",
                "details": json.dumps(initial.to_dict(), sort_keys=True),
            }
        )

    expected_max = {"diag": (0.0036, 0.00036), "muon": (0.0040, 0.00040)}
    normalized_curves: list[np.ndarray] = []
    lr_ok = True
    ratio_ok = True
    time_ok = True
    for (method, seed), part in wide.groupby(["method", "seed"]):
        part = part.sort_values("step")
        observed = (float(part["lr/adamw"].max()), float(part["lr/matrix"].max()))
        lr_ok = lr_ok and np.allclose(observed, expected_max[method], rtol=0, atol=1e-12)
        positive = part[part["lr/matrix"] > 0]
        ratio_ok = ratio_ok and np.allclose(positive["lr/adamw"], 10 * positive["lr/matrix"], rtol=1e-10, atol=1e-12)
        normalized_curves.append(part["lr/matrix"].to_numpy(dtype=float) / observed[1])
        times = part.loc[part["step"] >= 40, "time/train_s"].dropna().to_numpy(dtype=float)
        time_ok = time_ok and bool(np.all(np.diff(times) >= 0))
    schedule_ok = all(np.allclose(normalized_curves[0], curve, rtol=1e-10, atol=1e-12) for curve in normalized_curves[1:])
    quality_rows.extend(
        [
            {"check": "crossed absolute LR assignment", "status": "PASS" if lr_ok else "FAIL", "details": "diag=0.9x; Muon=1.0x"},
            {"check": "AdamW/matrix LR ratio is 10x", "status": "PASS" if ratio_ok else "FAIL", "details": "all positive-LR points"},
            {"check": "identical normalized LR schedule", "status": "PASS" if schedule_ok else "FAIL", "details": "all six runs"},
            {"check": "train time monotone after reset", "status": "PASS" if time_ok else "FAIL", "details": "timing remains descriptive only"},
            {"check": "local manifests and resume_count", "status": "WARN", "details": "not present in W&B CSV exports; request three LR-cross manifests/summaries"},
            {"check": "formal R1 seeds2024/2025 cells", "status": "WARN", "details": "not present locally; required for full three-seed 2x2"},
        ]
    )

    summary = run_summary(wide, run_names)
    paired, aggregate = paired_crossed_summary(summary)
    method_aggregate = (
        summary.groupby(["method", "lr_level"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            final_val_loss_mean=("final_val_loss", "mean"),
            final_val_loss_std=("final_val_loss", "std"),
            tail5_val_loss_mean=("tail5_val_loss_mean", "mean"),
            tail5_val_loss_std=("tail5_val_loss_mean", "std"),
            normalized_val_auc_mean=("normalized_val_auc", "mean"),
            normalized_val_auc_std=("normalized_val_auc", "std"),
        )
    )
    formal_summary = pd.read_csv(args.formal_r1_analysis / "r1_run_summary.csv")
    cells, effects = factorial_seed2026(summary, formal_summary)

    val = wide[wide["val/loss"].notna()][["method", "seed", "step", "val/loss"]].rename(columns={"val/loss": "val_loss"})
    dominance_rows: list[dict[str, object]] = []
    for seed, part in val.groupby("seed"):
        curves = part.pivot(index="step", columns="method", values="val_loss").sort_index()
        curves = curves[curves.index > 0]
        difference = curves["diag"] - curves["muon"]
        persistent_start = math.nan
        for index in range(len(difference)):
            if np.all(difference.iloc[index:].to_numpy(dtype=float) <= 0):
                persistent_start = int(difference.index[index])
                break
        dominance_rows.append(
            {
                "seed": int(seed),
                "comparison": "diag_low_minus_muon_high",
                "diag_lower_points": int((difference < 0).sum()),
                "muon_lower_points": int((difference > 0).sum()),
                "tied_points": int((difference == 0).sum()),
                "noninitial_validation_points": len(difference),
                "persistent_diag_lead_start_step": persistent_start,
                "mean_curve_delta": float(difference.mean()),
                "median_curve_delta": float(difference.median()),
            }
        )
    dominance = pd.DataFrame(dominance_rows)
    target_rows: list[dict[str, object]] = []
    for target in (3.5, 3.4, 3.3):
        for (method, seed), curve in val.groupby(["method", "seed"]):
            target_rows.append({"target_val_loss": target, "method": method, "seed": seed, **first_crossing(curve, target)})
    targets = pd.DataFrame(target_rows)

    formal_history = pd.read_csv(args.formal_r1_analysis / "normalized_history_long.csv")
    formal_val = formal_history[
        (formal_history["seed"] == 2026)
        & (formal_history["method"].isin(METHODS))
        & (formal_history["metric"] == "val/loss")
    ][["method", "step", "value"]].copy()
    formal_val["lr_level"] = formal_val["method"].map({"diag": "high_1.0x", "muon": "low_0.9x"})
    crossed_val_2026 = val[val["seed"] == 2026][["method", "step", "val_loss"]].rename(columns={"val_loss": "value"})
    crossed_val_2026["lr_level"] = crossed_val_2026["method"].map({"diag": "low_0.9x", "muon": "high_1.0x"})
    factorial_val = pd.concat([formal_val, crossed_val_2026], ignore_index=True)
    factorial_target_rows: list[dict[str, object]] = []
    for target in (3.5, 3.4, 3.3):
        for (method, lr_level), curve in factorial_val.groupby(["method", "lr_level"]):
            renamed = curve.rename(columns={"value": "val_loss"})
            factorial_target_rows.append(
                {
                    "target_val_loss": target,
                    "method": method,
                    "lr_level": lr_level,
                    **first_crossing(renamed, target),
                }
            )
    factorial_targets = pd.DataFrame(factorial_target_rows)

    quality = pd.DataFrame(quality_rows)
    blocking = quality[quality["status"] == "FAIL"]
    pd.DataFrame(manifest_rows).to_csv(out / "source_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    long.to_csv(out / "normalized_history_long.csv", index=False)
    wide.to_csv(out / "normalized_history_wide.csv", index=False)
    summary.to_csv(out / "crossed_run_summary.csv", index=False)
    method_aggregate.to_csv(out / "crossed_method_aggregate.csv", index=False)
    paired.to_csv(out / "crossed_paired_by_seed.csv", index=False)
    aggregate.to_csv(out / "crossed_paired_aggregate.csv", index=False)
    dominance.to_csv(out / "crossed_curve_dominance.csv", index=False)
    cells.to_csv(out / "seed2026_factorial_cells.csv", index=False)
    effects.to_csv(out / "seed2026_factorial_effects.csv", index=False)
    targets.to_csv(out / "steps_to_loss_targets.csv", index=False)
    factorial_targets.to_csv(out / "seed2026_factorial_steps_to_loss.csv", index=False)
    quality.to_csv(out / "data_quality_checks.csv", index=False)
    write_report(out / "R1_LR_CROSS_ANALYSIS_20260721.md", summary, aggregate, effects, quality)
    (out / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS_WITH_CAVEATS" if blocking.empty else "FAIL",
                "scope": "six crossed runs across seeds 2024/2025/2026 plus complete seed2026 2x2 using formal R1 anchor",
                "input_files": [str(path.resolve()) for path in args.input],
                "formal_r1_analysis": str(args.formal_r1_analysis.resolve()),
                "methods": list(METHODS),
                "seeds": list(SEEDS),
                "metrics": sorted(seen_metrics),
                "quality_pass": int((quality["status"] == "PASS").sum()),
                "quality_warn": int((quality["status"] == "WARN").sum()),
                "quality_fail": int((quality["status"] == "FAIL").sum()),
                "quality_usable": blocking.empty,
                "memory_usable": blocking.empty,
                "timing_usable": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if not blocking.empty:
        raise RuntimeError(f"LR-cross audit failed:\n{blocking.to_string(index=False)}")
    print(summary.to_string(index=False))
    print(effects.to_string(index=False))
    print(f"Saved analysis to {out}")


if __name__ == "__main__":
    main()
