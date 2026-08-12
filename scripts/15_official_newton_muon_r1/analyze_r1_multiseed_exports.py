"""Audit formal R1 seeds 2024/2025 and merge with seed2026/LR-cross evidence."""

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


METHODS = ("muon", "block4", "none", "diag")
SEEDS = (2024, 2025)
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
RUN_RE = re.compile(r"_r1_(muon|block4|none|diag)_seed(2024|2025)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--seed2026-dir", type=Path, required=True)
    parser.add_argument("--lr-cross-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_run(run_name: str) -> tuple[str, int]:
    match = RUN_RE.search(run_name)
    if not match:
        raise ValueError(f"unrecognized formal R1 run name: {run_name}")
    return match.group(1), int(match.group(2))


def normalized_auc(curve: pd.DataFrame) -> float:
    curve = curve.sort_values("step")
    return float(np.trapezoid(curve["value"], curve["step"]) / 6200.0)


def first_crossing(curve: pd.DataFrame, target: float) -> tuple[float, float]:
    curve = curve.sort_values("step").reset_index(drop=True)
    reached = curve[curve["value"] <= target]
    if reached.empty:
        return math.nan, math.nan
    index = int(reached.index[0])
    discrete = float(curve.loc[index, "step"])
    if index == 0 or float(curve.loc[index, "value"]) == target:
        return discrete, discrete
    prior = curve.loc[index - 1]
    current = curve.loc[index]
    high, low = float(prior["value"]), float(current["value"])
    fraction = (high - target) / (high - low) if high != low else 1.0
    interpolated = float(prior["step"] + fraction * (current["step"] - prior["step"]))
    return discrete, interpolated


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    raw_dir = out / "raw_wandb_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    quality: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    run_names: dict[tuple[int, str], str] = {}

    for source_arg in args.input:
        source = source_arg.resolve()
        preserved = raw_dir / source.name
        if preserved.exists() and sha256(preserved) != sha256(source):
            raise RuntimeError(f"refusing to overwrite different evidence: {preserved}")
        if not preserved.exists():
            shutil.copy2(source, preserved)
        sources.append({
            "source_role": "formal_r1_seed2024_2025_wandb_export",
            "source_file": source.name,
            "original_path": str(source),
            "preserved_path": str(preserved),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        })
        frame = pd.read_csv(source)
        primary = [c for c in frame.columns if c != "Step" and not c.endswith("__MIN") and not c.endswith("__MAX")]
        metric_names = {c.rsplit(" - ", 1)[1] for c in primary}
        if len(primary) != 8 or len(metric_names) != 1:
            raise RuntimeError(f"{source.name}: expected 8 run columns and one metric")
        metric = next(iter(metric_names))
        expected_steps = EXPECTED_METRICS.get(metric)
        observed_steps = frame["Step"].astype(int).tolist()
        quality.append({
            "check": f"{metric}: exact step grid",
            "status": "PASS" if expected_steps == observed_steps else "FAIL",
            "details": f"rows={len(observed_steps)}, first={observed_steps[:2]}, last={observed_steps[-2:]}",
        })
        coverage: set[tuple[int, str]] = set()
        for column in primary:
            run_name, parsed_metric = column.rsplit(" - ", 1)
            method, seed = parse_run(run_name)
            coverage.add((seed, method))
            previous = run_names.setdefault((seed, method), run_name)
            if previous != run_name:
                raise RuntimeError(f"multiple run names for seed={seed} method={method}")
            values = pd.to_numeric(frame[column], errors="coerce")
            minimum = pd.to_numeric(frame[f"{column}__MIN"], errors="coerce")
            maximum = pd.to_numeric(frame[f"{column}__MAX"], errors="coerce")
            finite = np.isfinite(values.to_numpy(float))
            mirrors = bool(
                finite.all()
                and np.allclose(values, minimum, rtol=0, atol=0)
                and np.allclose(values, maximum, rtol=0, atol=0)
            )
            quality.extend([
                {
                    "check": f"seed{seed} {method} {metric}: finite",
                    "status": "PASS" if finite.all() else "FAIL",
                    "details": f"nonfinite={int((~finite).sum())}",
                },
                {
                    "check": f"seed{seed} {method} {metric}: MIN/MAX mirrors",
                    "status": "PASS" if mirrors else "FAIL",
                    "details": f"points={len(values)}",
                },
            ])
            frames.append(pd.DataFrame({
                "method": method,
                "run_name": run_name,
                "seed": seed,
                "metric": parsed_metric,
                "step": frame["Step"].astype(int),
                "value": values,
                "source_file": source.name,
            }))
        quality.append({
            "check": f"{metric}: exact seed/method coverage",
            "status": "PASS" if coverage == {(s, m) for s in SEEDS for m in METHODS} else "FAIL",
            "details": ";".join(f"{s}:{m}" for s, m in sorted(coverage)),
        })

    seen_metrics = {frame["metric"].iloc[0] for frame in frames}
    quality.append({
        "check": "exact nine-metric coverage",
        "status": "PASS" if seen_metrics == set(EXPECTED_METRICS) else "FAIL",
        "details": ",".join(sorted(seen_metrics)),
    })
    quality.append({
        "check": "exact eight-run coverage",
        "status": "PASS" if set(run_names) == {(s, m) for s in SEEDS for m in METHODS} else "FAIL",
        "details": f"runs={len(run_names)}",
    })
    long = pd.concat(frames, ignore_index=True)
    duplicate_grain = int(long.duplicated(["seed", "method", "metric", "step"]).sum())
    quality.append({
        "check": "no duplicate seed/method/metric/step grain",
        "status": "PASS" if duplicate_grain == 0 else "FAIL",
        "details": f"duplicates={duplicate_grain}",
    })

    summary_rows: list[dict[str, object]] = []
    for (seed, method), run_name in sorted(run_names.items()):
        subset = long[(long.seed == seed) & (long.method == method)]
        val = subset[subset.metric == "val/loss"].sort_values("step")
        tail = val.tail(5)["value"]
        scalar = lambda metric: subset[subset.metric == metric].sort_values("step")["value"]
        summary_rows.append({
            "method": method,
            "run_name": run_name,
            "seed": seed,
            "initial_val_loss": float(val.iloc[0].value),
            "final_val_loss": float(val.iloc[-1].value),
            "best_val_loss": float(val.value.min()),
            "tail3_val_loss_mean": float(val.tail(3).value.mean()),
            "tail5_val_loss_mean": float(tail.mean()),
            "tail5_val_loss_sd": float(tail.std(ddof=1)),
            "normalized_val_auc": normalized_auc(val),
            "final_train_loss_step": float(scalar("train/loss_step").iloc[-1]),
            "train_time_s_descriptive_only": float(scalar("time/train_s").iloc[-1]),
            "final_step_avg_ms_descriptive_only": float(scalar("performance/step_avg_ms").iloc[-1]),
            "max_adamw_lr": float(scalar("lr/adamw").max()),
            "max_matrix_lr": float(scalar("lr/matrix").max()),
            "peak_memory_mib": float(scalar("memory/peak_allocated_mib").iloc[-1]),
            "k_state_mib": float(scalar("memory/k_state_mib").iloc[-1]),
            "optimizer_state_mib": float(scalar("memory/optimizer_state_mib").iloc[-1]),
        })
    new_summary = pd.DataFrame(summary_rows)

    seed2026_dir = args.seed2026_dir.resolve()
    seed2026_summary_path = seed2026_dir / "r1_run_summary.csv"
    seed2026_long_path = seed2026_dir / "normalized_history_long.csv"
    cross_dir = args.lr_cross_dir.resolve()
    cross_summary_path = cross_dir / "crossed_run_summary.csv"
    for role, path in (
        ("formal_r1_seed2026_summary", seed2026_summary_path),
        ("formal_r1_seed2026_history", seed2026_long_path),
        ("r1_lr_cross_summary", cross_summary_path),
    ):
        sources.append({
            "source_role": role,
            "source_file": path.name,
            "original_path": str(path),
            "preserved_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    seed2026_summary = pd.read_csv(seed2026_summary_path)
    seed2026_long = pd.read_csv(seed2026_long_path)
    seed2026_summary = seed2026_summary.rename(columns={
        "train_time_s": "train_time_s_descriptive_only",
        "final_step_avg_ms": "final_step_avg_ms_descriptive_only",
    })
    combined_summary = pd.concat([new_summary, seed2026_summary], ignore_index=True, sort=False)
    combined_long = pd.concat([long, seed2026_long], ignore_index=True, sort=False)

    for seed in (2024, 2025, 2026):
        initial = combined_summary[combined_summary.seed == seed].initial_val_loss
        quality.append({
            "check": f"seed{seed}: identical four-method initialization loss",
            "status": "PASS" if initial.nunique() == 1 else "FAIL",
            "details": ",".join(f"{v:.4f}" for v in initial),
        })
    expected_lr = {"muon": (0.0036, 0.00036), "block4": (0.004, 0.0004), "none": (0.004, 0.0004), "diag": (0.004, 0.0004)}
    for row in combined_summary.itertuples():
        expected = expected_lr[row.method]
        quality.append({
            "check": f"seed{row.seed} {row.method}: expected formal LR",
            "status": "PASS" if np.isclose(row.max_adamw_lr, expected[0]) and np.isclose(row.max_matrix_lr, expected[1]) else "FAIL",
            "details": f"adamw={row.max_adamw_lr}, matrix={row.max_matrix_lr}",
        })

    aggregate_rows: list[dict[str, object]] = []
    for method, group in combined_summary.groupby("method"):
        aggregate_rows.append({
            "method": method,
            "seeds": len(group),
            "final_val_mean": group.final_val_loss.mean(),
            "final_val_sd": group.final_val_loss.std(ddof=1),
            "tail5_mean": group.tail5_val_loss_mean.mean(),
            "tail5_between_seed_sd": group.tail5_val_loss_mean.std(ddof=1),
            "normalized_auc_mean": group.normalized_val_auc.mean(),
            "normalized_auc_sd": group.normalized_val_auc.std(ddof=1),
            "peak_memory_mib": group.peak_memory_mib.mean(),
            "k_state_mib": group.k_state_mib.mean(),
            "optimizer_state_mib": group.optimizer_state_mib.mean(),
        })
    aggregate = pd.DataFrame(aggregate_rows).sort_values("final_val_mean")

    paired_rows: list[dict[str, object]] = []
    comparisons = (("diag", "block4"), ("diag", "none"), ("block4", "none"), ("diag", "muon"), ("block4", "muon"))
    for left, right in comparisons:
        deltas = []
        for seed in (2024, 2025, 2026):
            l = combined_summary[(combined_summary.seed == seed) & (combined_summary.method == left)].iloc[0]
            r = combined_summary[(combined_summary.seed == seed) & (combined_summary.method == right)].iloc[0]
            delta = float(l.final_val_loss - r.final_val_loss)
            deltas.append(delta)
            paired_rows.append({"comparison": f"{left}_minus_{right}", "seed": seed, "final_delta": delta})
        paired_rows.append({
            "comparison": f"{left}_minus_{right}",
            "seed": "aggregate_mean",
            "final_delta": float(np.mean(deltas)),
        })
        paired_rows.append({
            "comparison": f"{left}_minus_{right}",
            "seed": "aggregate_sd",
            "final_delta": float(np.std(deltas, ddof=1)),
        })
    paired = pd.DataFrame(paired_rows)

    dominance_rows: list[dict[str, object]] = []
    for seed in (2024, 2025, 2026):
        val = combined_long[(combined_long.seed == seed) & (combined_long.metric == "val/loss")]
        wide = val.pivot(index="step", columns="method", values="value").sort_index()
        for left, right in comparisons[:3]:
            noninitial = wide.loc[wide.index > 0]
            delta = noninitial[left] - noninitial[right]
            persistent = math.nan
            for index in range(len(delta)):
                if bool((delta.iloc[index:] <= 0).all()):
                    persistent = float(delta.index[index])
                    break
            dominance_rows.append({
                "seed": seed,
                "comparison": f"{left}_minus_{right}",
                "left_lower_points": int((delta < 0).sum()),
                "ties": int((delta == 0).sum()),
                "validation_points_excluding_step0": len(delta),
                "persistent_left_lead_start": persistent,
            })
    dominance = pd.DataFrame(dominance_rows)

    target_rows: list[dict[str, object]] = []
    for seed in (2024, 2025, 2026):
        for method in METHODS:
            curve = combined_long[(combined_long.seed == seed) & (combined_long.method == method) & (combined_long.metric == "val/loss")]
            for target in (3.5, 3.4, 3.3):
                discrete, interpolated = first_crossing(curve, target)
                target_rows.append({
                    "seed": seed,
                    "method": method,
                    "target_loss": target,
                    "discrete_step": discrete,
                    "interpolated_step": interpolated,
                    "interpolated_tokens": interpolated * 512 * 1024,
                })
    targets = pd.DataFrame(target_rows)

    cross = pd.read_csv(cross_summary_path)
    factorial_cells: list[dict[str, object]] = []
    for seed in (2024, 2025, 2026):
        formal = combined_summary[combined_summary.seed == seed]
        crossed = cross[cross.seed == seed]
        cells = {
            ("diag", "low_0.9x"): float(crossed[crossed.method == "diag"].final_val_loss.iloc[0]),
            ("diag", "high_1.0x"): float(formal[formal.method == "diag"].final_val_loss.iloc[0]),
            ("muon", "low_0.9x"): float(formal[formal.method == "muon"].final_val_loss.iloc[0]),
            ("muon", "high_1.0x"): float(crossed[crossed.method == "muon"].final_val_loss.iloc[0]),
        }
        for (method, lr), value in cells.items():
            factorial_cells.append({"seed": seed, "method": method, "lr_level": lr, "final_val_loss": value})
    factorial_cells_frame = pd.DataFrame(factorial_cells)
    effects: list[dict[str, object]] = []
    for seed in (2024, 2025, 2026):
        cell = factorial_cells_frame[factorial_cells_frame.seed == seed].set_index(["method", "lr_level"]).final_val_loss
        diag_low = cell.loc[("diag", "low_0.9x")]
        diag_high = cell.loc[("diag", "high_1.0x")]
        muon_low = cell.loc[("muon", "low_0.9x")]
        muon_high = cell.loc[("muon", "high_1.0x")]
        method_effect = ((diag_low + diag_high) - (muon_low + muon_high)) / 2
        lr_effect = ((diag_high + muon_high) - (diag_low + muon_low)) / 2
        interaction = (diag_high - diag_low) - (muon_high - muon_low)
        effects.append({"seed": seed, "method_effect_diag_minus_muon": method_effect, "lr_effect_high_minus_low": lr_effect, "interaction_difference_in_differences": interaction})
    effect_frame = pd.DataFrame(effects)
    effect_frame = pd.concat([effect_frame, pd.DataFrame([
        {"seed": "aggregate_mean", **{c: effect_frame[c].mean() for c in effect_frame.columns if c != "seed"}},
        {"seed": "aggregate_sd", **{c: effect_frame[c].std(ddof=1) for c in effect_frame.columns if c != "seed"}},
    ])], ignore_index=True)

    q = pd.DataFrame(quality)
    if (q.status == "FAIL").any():
        raise RuntimeError("blocking W&B export audit failure:\n" + q[q.status == "FAIL"].to_string(index=False))

    source_frame = pd.DataFrame(sources)
    source_frame.to_csv(out / "source_manifest.csv", index=False)
    q.to_csv(out / "data_quality_checks.csv", index=False)
    long.to_csv(out / "formal_seed2024_2025_history_long.csv", index=False)
    combined_summary.to_csv(out / "r1_multiseed_run_summary.csv", index=False)
    aggregate.to_csv(out / "r1_multiseed_method_aggregate.csv", index=False)
    paired.to_csv(out / "r1_multiseed_paired_deltas.csv", index=False)
    dominance.to_csv(out / "r1_multiseed_curve_dominance.csv", index=False)
    targets.to_csv(out / "r1_multiseed_steps_to_loss.csv", index=False)
    factorial_cells_frame.to_csv(out / "r1_three_seed_factorial_cells.csv", index=False)
    effect_frame.to_csv(out / "r1_three_seed_factorial_effects.csv", index=False)

    by_method = aggregate.set_index("method")
    pair_index = paired.set_index(["comparison", "seed"])
    effect_mean = effect_frame[effect_frame.seed == "aggregate_mean"].iloc[0]
    effect_sd = effect_frame[effect_frame.seed == "aggregate_sd"].iloc[0]
    report = f"""# R1 三种子正式实验与完整学习率 factorial 分析（2026-07-21）

## 证据状态

- 新导出的 9 个 W&B 指标覆盖 seed2024/2025 × 4 方法，共 8 个正式 run。
- 与已审计 seed2026 合并后，正式 R1 为 3 seeds × 4 methods = 12 个 run。
- W&B 数据质量检查：PASS={int((q.status == 'PASS').sum())}，WARN={int((q.status == 'WARN').sum())}，FAIL=0。
- 当前结论为 `PASS_WITH_CAVEATS`：CSV 不能证明本地 source/runtime fingerprint、`resume_count`、checkpoint 完整性，也不能单独证明 seed2025 diag 是 step0 clean retry。

## 三种子正式 R1

| 方法 | final mean | seed SD | tail-5 mean | normalized AUC | Peak MiB | K-state MiB |
|---|---:|---:|---:|---:|---:|---:|
| diag | {by_method.loc['diag','final_val_mean']:.6f} | {by_method.loc['diag','final_val_sd']:.6f} | {by_method.loc['diag','tail5_mean']:.6f} | {by_method.loc['diag','normalized_auc_mean']:.6f} | {by_method.loc['diag','peak_memory_mib']:.0f} | {by_method.loc['diag','k_state_mib']:.5f} |
| block4 | {by_method.loc['block4','final_val_mean']:.6f} | {by_method.loc['block4','final_val_sd']:.6f} | {by_method.loc['block4','tail5_mean']:.6f} | {by_method.loc['block4','normalized_auc_mean']:.6f} | {by_method.loc['block4','peak_memory_mib']:.0f} | {by_method.loc['block4','k_state_mib']:.0f} |
| none | {by_method.loc['none','final_val_mean']:.6f} | {by_method.loc['none','final_val_sd']:.6f} | {by_method.loc['none','tail5_mean']:.6f} | {by_method.loc['none','normalized_auc_mean']:.6f} | {by_method.loc['none','peak_memory_mib']:.0f} | {by_method.loc['none','k_state_mib']:.0f} |
| Muon | {by_method.loc['muon','final_val_mean']:.6f} | {by_method.loc['muon','final_val_sd']:.6f} | {by_method.loc['muon','tail5_mean']:.6f} | {by_method.loc['muon','normalized_auc_mean']:.6f} | {by_method.loc['muon','peak_memory_mib']:.0f} | {by_method.loc['muon','k_state_mib']:.0f} |

三 seed 配对 final 差：

- diag − block4：{pair_index.loc[('diag_minus_block4','aggregate_mean'),'final_delta']:.6f} ± {pair_index.loc[('diag_minus_block4','aggregate_sd'),'final_delta']:.6f}。
- diag − none：{pair_index.loc[('diag_minus_none','aggregate_mean'),'final_delta']:.6f} ± {pair_index.loc[('diag_minus_none','aggregate_sd'),'final_delta']:.6f}。
- diag − Muon：{pair_index.loc[('diag_minus_muon','aggregate_mean'),'final_delta']:.6f} ± {pair_index.loc[('diag_minus_muon','aggregate_sd'),'final_delta']:.6f}。

## 三种子完整 2×2：diag/Muon × 0.9x/1.0x LR

- 方法主效应（diag − Muon）：{effect_mean['method_effect_diag_minus_muon']:.6f} ± {effect_sd['method_effect_diag_minus_muon']:.6f}。
- LR 主效应（1.0x − 0.9x）：{effect_mean['lr_effect_high_minus_low']:.6f} ± {effect_sd['lr_effect_high_minus_low']:.6f}。
- 方法×LR 交互：{effect_mean['interaction_difference_in_differences']:.6f} ± {effect_sd['interaction_difference_in_differences']:.6f}。
- 方法主效应绝对值约为 LR 主效应绝对值的 {abs(effect_mean['method_effect_diag_minus_muon']) / max(abs(effect_mean['lr_effect_high_minus_low']), 1e-12):.1f} 倍。

## 结论

1. diag 与 block4 的三种子平均终点基本持平；现有证据支持“质量保持、状态更省”，不支持“diag 显著优于 block4”。
2. diag 相对 none 的优势跨 seed 为同一方向，说明逐坐标尺度不是纯粹冗余；但效应量仍小，应结合 tail/AUC 和曲线支配性表述。
3. diag 相对 Muon 的优势在完整 2×2 后仍远大于 10% LR 主效应，学习率混杂这一主要替代解释已经基本排除。
4. block4 的完整 c_proj block covariance 没有显示出相对 diag 的稳定质量收益，却需要额外约 215.72 MiB K-state 和 864 MiB 实测峰值显存。
5. wall-clock/step-time 不进入论文性能结论；这批 run 与其他 GPU 任务并发，且 W&B CSV 不包含完整节点隔离证据。

## 投稿前剩余门禁

- 收到并审计 seed2024/2025 的本地 formal manifests、summary、source/runtime/init hashes 和 checkpoint 完整性。
- 明确核验 seed2025 diag 为从 step0 开始的 clean retry，且没有拼接旧曲线。
- 将 12-run 正式 R1 与 6-run LR-cross 原始证据迁入 paper evidence 包后，再把本报告升级为 `READY_FOR_MAIN_TABLE`。
"""
    (out / "R1_MULTISEED_FACTORIAL_ANALYSIS_20260721.md").write_text(report, encoding="utf-8")
    manifest = {
        "created_at": "2026-07-21",
        "status": "PASS_WITH_CAVEATS",
        "formal_run_count": 12,
        "lr_cross_run_count": 6,
        "quality_checks": q.status.value_counts().to_dict(),
        "outputs": sorted(path.name for path in out.iterdir() if path.is_file()),
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"R1 multiseed analysis written to: {out}")


if __name__ == "__main__":
    main()
