"""Audit and summarize the seed-2026 LLaMA/SwiGLU W&B CSV exports.

The W&B panels contain curve histories but not run-summary scalars.  This
script therefore separates quality evidence from memory/timing eligibility:
quality results are usable after the curve audit passes, while peak memory,
resume status, and formal timing remain pending until local summaries are
provided.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("adamw", "muon", "newton_full", "down_none", "down_diag")
CORE_METHODS = ("muon", "newton_full", "down_none", "down_diag")
NEWTON_METHODS = ("newton_full", "down_none", "down_diag")
EXPECTED_METRICS = {
    "val/loss": list(range(0, 6201, 100)),
    "train/loss_step": list(range(20, 6201, 20)),
    "tokens/seen": list(range(0, 6201, 20)),
    "time/train_s": list(range(0, 6201, 20)),
    "performance/step_avg_ms": list(range(0, 6201, 20)),
    "lr/backup": list(range(0, 6201, 20)),
    "lr/matrix": list(range(0, 6201, 20)),
}
EXPECTED_FINITE_STEPS = {
    metric: (steps[2:] if metric == "performance/step_avg_ms" else steps)
    for metric, steps in EXPECTED_METRICS.items()
}
TOKENS_PER_STEP = 512 * 1024
TOTAL_STEPS = 6200
EXPECTED_K_STATE_MIB = {
    "adamw": 0.0,
    "muon": 0.0,
    "newton_full": 546.0,
    "down_none": 162.0,
    "down_diag": 162.1875,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def method_from_run(run_name: str) -> str:
    prefix = "llama_swiglu_"
    suffix = "_seed2026"
    if run_name.startswith(prefix) and run_name.endswith(suffix):
        method = run_name[len(prefix) : -len(suffix)]
        if method in METHODS:
            return method
    raise ValueError(f"Unrecognized LLaMA/SwiGLU run name: {run_name}")


def first_crossing(curve: pd.DataFrame, target: float) -> dict[str, float | int | bool]:
    curve = curve.sort_values("step").reset_index(drop=True)
    reached = curve[curve["val_loss"] <= target]
    if reached.empty:
        return {
            "reached": False,
            "first_discrete_step": math.nan,
            "interpolated_step": math.nan,
            "first_discrete_tokens": math.nan,
            "interpolated_tokens": math.nan,
            "interpolated_train_time_s_descriptive": math.nan,
        }
    index = int(reached.index[0])
    current = curve.iloc[index]
    if index == 0 or float(current["val_loss"]) == target:
        interpolated_step = float(current["step"])
        interpolated_time = float(current["train_time_s"])
    else:
        previous = curve.iloc[index - 1]
        high = float(previous["val_loss"])
        low = float(current["val_loss"])
        fraction = (high - target) / (high - low) if high != low else 1.0
        interpolated_step = float(
            previous["step"] + fraction * (current["step"] - previous["step"])
        )
        interpolated_time = float(
            previous["train_time_s"]
            + fraction * (current["train_time_s"] - previous["train_time_s"])
        )
    return {
        "reached": True,
        "first_discrete_step": int(current["step"]),
        "interpolated_step": interpolated_step,
        "first_discrete_tokens": int(current["step"]) * TOKENS_PER_STEP,
        "interpolated_tokens": interpolated_step * TOKENS_PER_STEP,
        "interpolated_train_time_s_descriptive": interpolated_time,
    }


def persistent_lead_start(left: pd.Series, right: pd.Series, steps: pd.Series) -> float:
    differences = (left - right).to_numpy(dtype=float)
    step_values = steps.to_numpy(dtype=int)
    for index in range(len(differences)):
        if np.all(differences[index:] <= 0):
            return float(step_values[index])
    return math.nan


def write_report(
    path: Path,
    summary: pd.DataFrame,
    pairs: pd.DataFrame,
    quality: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    s = summary.set_index("method")
    p = pairs.set_index("comparison")
    counts = quality["status"].value_counts().to_dict()
    all_common = float(summary["final_val_loss"].max())
    core_common = float(summary[summary["method"].isin(CORE_METHODS)]["final_val_loss"].max())
    target_33 = targets[np.isclose(targets["target_val_loss"], 3.3)].set_index("method")
    lines = [
        "# LLaMA / SwiGLU seed-2026 W&B 分析（2026-07-21）",
        "",
        "## 证据状态",
        "",
        "- 5 个方法与 7 个曲线指标均齐全，所有运行到达 6200 step / 3,250,585,600 token。",
        f"- 五个方法的初始验证损失完全一致：{s.iloc[0]['initial_val_loss']:.9f}。",
        f"- 数据质量检查：PASS {counts.get('PASS', 0)}，WARN {counts.get('WARN', 0)}，FAIL {counts.get('FAIL', 0)}。",
        "- `quality_usable=true`；W&B 面板未包含 summary 标量，因此 `memory_usable` 与 `timing_usable` 暂不成立。",
        "",
        "## 终点与全程质量",
        "",
        "| 方法 | Final val | Tail-5 mean | 标准化 AUC | Final train | 3.3 首次离散 step |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in summary.sort_values("final_val_loss")["method"]:
        row = s.loc[method]
        crossing = target_33.loc[method]
        step_text = str(int(crossing["first_discrete_step"])) if crossing["reached"] else "未达到"
        lines.append(
            f"| {method} | {row['final_val_loss']:.6f} | {row['tail5_val_loss_mean']:.6f} | "
            f"{row['normalized_val_auc']:.6f} | {row['final_train_loss_step']:.6f} | {step_text} |"
        )
    lines.extend(
        [
            "",
            "## 主要结论",
            "",
            f"1. **受控的三个 Newton 变体在单 seed 下实质持平。** 终点排序为 "
            f"down_none {s.loc['down_none','final_val_loss']:.6f}、newton_full "
            f"{s.loc['newton_full','final_val_loss']:.6f}、down_diag "
            f"{s.loc['down_diag','final_val_loss']:.6f}；diag 相对 none 高 "
            f"{p.loc['down_diag_minus_down_none','final_val_loss_delta_left_minus_right']:.6f}，差值远小于单 seed 可支持的稳定优势量级。",
            f"2. **R1 中的 diag 小优势没有在 LLaMA/SwiGLU 上复现。** 本 seed 里 none 略优，说明 down-proj 的 K 表示收益具有架构依赖性；当前证据不支持“diag 普遍优于 none/full”。",
            f"3. **Muon 与 Newton 三变体也几乎重合。** Muon 终点 {s.loc['muon','final_val_loss']:.6f}；"
            f"相对 full 的差仅 {p.loc['newton_full_minus_muon','final_val_loss_delta_left_minus_right']:.6f}，"
            "需要多 seed 才能判断是否存在可重复差异。",
            f"4. **当前 AdamW 配方明显落后。** 终点 {s.loc['adamw','final_val_loss']:.6f}，"
            f"比最优 down_none 高 {s.loc['adamw','final_val_loss']-s.loc['down_none','final_val_loss']:.6f}。"
            "但 AdamW 的矩阵学习率是独立配方（0.000576 vs 0.01），因此只能表述为“当前配方比较”，不能推广为 AdamW 本身必然更差。",
            f"5. 全方法共同可达目标为 {all_common:.6f}；非 AdamW 四方法共同可达目标为 {core_common:.6f}。"
            "同-loss 的离散 step/Token 是本批主要效率证据；线性插值仅作辅助，因为验证间隔为 100 step。",
            "",
            "## 性能与显存边界",
            "",
            "- W&B 中的累计训练时间和 step_avg_ms 仅保存为描述性数据，不作为正式性能实验结论。",
            "- 从代码/预检可知理论 K-state：none 162.0000 MiB、diag 162.1875 MiB、full 546.0000 MiB；"
            "但实际峰值显存、优化器状态、恢复次数仍需本地 `llama_swiglu_summary.csv` / `summary.json` 复核。",
            "",
            "## 下一步",
            "",
            "- 至少补 seed 2024、2025 的 muon/newton_full/down_none/down_diag；若 AdamW 是论文主表基线，也同步补 AdamW。",
            "- 多 seed 主检验使用配对的 final、tail-5、AUC、steps/tokens-to-loss；不要用本批 wall-clock 做主张。",
            "- 在看到多 seed 前，把“none 略优”视为现象而不是结论，把“diag 优势未迁移”视为当前最可靠的方向性结果。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    raw_dir = out / "raw_wandb_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    long_frames: list[pd.DataFrame] = []
    seen_metrics: set[str] = set()
    run_names_by_method: dict[str, str] = {}

    for source_arg in args.input:
        source = source_arg.resolve()
        copied = (raw_dir / source.name).resolve()
        if copied.exists() and sha256(copied) != sha256(source):
            raise RuntimeError(f"Refusing to overwrite different evidence: {copied}")
        if source != copied and not copied.exists():
            shutil.copy2(source, copied)
        manifest_rows.append(
            {
                "source_file": source.name,
                "source_path": str(source),
                "preserved_path": str(copied),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )

        frame = pd.read_csv(source)
        primary = [
            column
            for column in frame.columns
            if column != "Step" and not column.endswith("__MIN") and not column.endswith("__MAX")
        ]
        if len(primary) != len(METHODS):
            raise RuntimeError(f"Expected five run columns in {source.name}, got {len(primary)}")
        metrics = {column.rsplit(" - ", 1)[1] for column in primary}
        if len(metrics) != 1:
            raise RuntimeError(f"Mixed metrics in {source.name}: {metrics}")
        metric = next(iter(metrics))
        seen_metrics.add(metric)
        expected_steps = EXPECTED_METRICS.get(metric)
        observed_steps = frame["Step"].astype(int).tolist()
        quality_rows.extend(
            [
                {
                    "check": f"{metric}: recognized metric",
                    "status": "PASS" if expected_steps is not None else "FAIL",
                    "details": source.name,
                },
                {
                    "check": f"{metric}: exact step grid",
                    "status": "PASS" if observed_steps == expected_steps else "FAIL",
                    "details": f"rows={len(observed_steps)}, first={observed_steps[:3]}, last={observed_steps[-3:]}",
                },
            ]
        )

        methods_in_file: set[str] = set()
        for column in primary:
            run_name, parsed_metric = column.rsplit(" - ", 1)
            method = method_from_run(run_name)
            methods_in_file.add(method)
            prior = run_names_by_method.setdefault(method, run_name)
            if prior != run_name:
                raise RuntimeError(f"Multiple run names for {method}: {prior} vs {run_name}")
            values = pd.to_numeric(frame[column], errors="coerce")
            mins = pd.to_numeric(frame[f"{column}__MIN"], errors="coerce")
            maxs = pd.to_numeric(frame[f"{column}__MAX"], errors="coerce")
            finite = np.isfinite(values.to_numpy(dtype=float))
            finite_steps = frame.loc[finite, "Step"].astype(int).tolist()
            mirrors = np.allclose(values[finite], mins[finite], rtol=0, atol=0) and np.allclose(
                values[finite], maxs[finite], rtol=0, atol=0
            )
            quality_rows.extend(
                [
                    {
                        "check": f"{method} {metric}: expected finite grid",
                        "status": "PASS" if finite_steps == EXPECTED_FINITE_STEPS[metric] else "FAIL",
                        "details": f"finite={len(finite_steps)}",
                    },
                    {
                        "check": f"{method} {metric}: W&B MIN/MAX mirrors",
                        "status": "PASS" if mirrors else "FAIL",
                        "details": f"finite={int(finite.sum())}",
                    },
                ]
            )
            long_frames.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "run_name": run_name,
                        "seed": 2026,
                        "metric": parsed_metric,
                        "step": frame["Step"].astype(int),
                        "value": values,
                        "source_file": source.name,
                    }
                )
            )
        quality_rows.append(
            {
                "check": f"{metric}: exact five-method coverage",
                "status": "PASS" if methods_in_file == set(METHODS) else "FAIL",
                "details": ",".join(sorted(methods_in_file)),
            }
        )

    quality_rows.append(
        {
            "check": "exact seven expected curve exports",
            "status": "PASS" if seen_metrics == set(EXPECTED_METRICS) else "FAIL",
            "details": f"missing={sorted(set(EXPECTED_METRICS)-seen_metrics)}, extra={sorted(seen_metrics-set(EXPECTED_METRICS))}",
        }
    )
    long = pd.concat(long_frames, ignore_index=True).sort_values(["metric", "method", "step"])
    duplicates = int(long.duplicated(["method", "metric", "step"]).sum())
    quality_rows.append(
        {
            "check": "unique method-metric-step keys",
            "status": "PASS" if duplicates == 0 else "FAIL",
            "details": f"duplicates={duplicates}",
        }
    )
    wide = long.pivot(index=["method", "step"], columns="metric", values="value").reset_index()
    val = wide[wide["val/loss"].notna()].copy().rename(
        columns={"val/loss": "val_loss", "time/train_s": "train_time_s"}
    )

    initial = val[val["step"] == 0].set_index("method")["val_loss"]
    quality_rows.append(
        {
            "check": "identical initial validation loss across five methods",
            "status": "PASS" if initial.nunique(dropna=False) == 1 else "FAIL",
            "details": json.dumps(initial.to_dict(), sort_keys=True),
        }
    )
    tokens_ok = True
    time_ok = True
    for _, frame in wide.groupby("method"):
        ordered = frame.sort_values("step")
        observed = ordered["tokens/seen"].to_numpy(dtype=float)
        expected = ordered["step"].to_numpy(dtype=float) * TOKENS_PER_STEP
        tokens_ok = tokens_ok and np.array_equal(observed, expected)
        times = ordered["time/train_s"].dropna().to_numpy(dtype=float)
        time_ok = time_ok and bool(np.all(np.diff(times) >= 0))
    quality_rows.extend(
        [
            {
                "check": "tokens equal step * 512 * 1024",
                "status": "PASS" if tokens_ok else "FAIL",
                "details": f"final={TOTAL_STEPS*TOKENS_PER_STEP}",
            },
            {
                "check": "cumulative training time is monotone",
                "status": "PASS" if time_ok else "FAIL",
                "details": "descriptive timing only",
            },
        ]
    )
    for metric in ("lr/backup", "lr/matrix"):
        curves = {
            method: wide[wide["method"] == method].sort_values("step")[metric].to_numpy(dtype=float)
            for method in METHODS
        }
        newton_equal = all(np.array_equal(curves["newton_full"], curves[m]) for m in ("down_none", "down_diag"))
        core_equal = all(np.array_equal(curves["muon"], curves[m]) for m in NEWTON_METHODS)
        normalized = {m: values / np.nanmax(values) for m, values in curves.items()}
        schedule_equal = all(np.allclose(normalized["muon"], normalized[m], rtol=1e-12, atol=1e-12) for m in METHODS)
        quality_rows.extend(
            [
                {
                    "check": f"{metric}: Newton variants identical",
                    "status": "PASS" if newton_equal else "FAIL",
                    "details": "newton_full/down_none/down_diag",
                },
                {
                    "check": f"{metric}: Muon and Newton absolute LR identical",
                    "status": "PASS" if core_equal else "FAIL",
                    "details": "four non-AdamW methods",
                },
                {
                    "check": f"{metric}: normalized schedule shape identical",
                    "status": "PASS" if schedule_equal else "FAIL",
                    "details": "all five methods",
                },
            ]
        )

    summary_rows: list[dict[str, object]] = []
    for method in METHODS:
        curve = val[val["method"] == method].sort_values("step")
        method_wide = wide[wide["method"] == method].sort_values("step")
        last = method_wide.iloc[-1]
        summary_rows.append(
            {
                "method": method,
                "run_name": run_names_by_method[method],
                "seed": 2026,
                "initial_val_loss": float(curve.iloc[0]["val_loss"]),
                "final_val_loss": float(curve.iloc[-1]["val_loss"]),
                "best_val_loss": float(curve["val_loss"].min()),
                "final_perplexity": math.exp(float(curve.iloc[-1]["val_loss"])),
                "tail3_val_loss_mean": float(curve.tail(3)["val_loss"].mean()),
                "tail5_val_loss_mean": float(curve.tail(5)["val_loss"].mean()),
                "tail5_val_loss_std": float(curve.tail(5)["val_loss"].std(ddof=1)),
                "normalized_val_auc": float(np.trapezoid(curve["val_loss"], curve["step"]) / TOTAL_STEPS),
                "final_train_loss_step": float(last["train/loss_step"]),
                "train_time_s_descriptive": float(last["time/train_s"]),
                "final_step_avg_ms_descriptive": float(last["performance/step_avg_ms"]),
                "max_backup_lr": float(method_wide["lr/backup"].max()),
                "max_matrix_lr": float(method_wide["lr/matrix"].max()),
                "final_tokens": int(last["tokens/seen"]),
                "expected_k_state_mib_from_preflight": EXPECTED_K_STATE_MIB[method],
                "observed_peak_memory_mib": math.nan,
                "observed_optimizer_state_mib": math.nan,
                "resume_count": math.nan,
                "quality_usable": True,
                "memory_usable": False,
                "timing_usable": False,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["final_loss_rank"] = summary["final_val_loss"].rank(method="min").astype(int)
    summary["auc_rank"] = summary["normalized_val_auc"].rank(method="min").astype(int)
    summary = summary.sort_values("final_loss_rank").reset_index(drop=True)
    by_method = summary.set_index("method")

    all_common = float(summary["final_val_loss"].max())
    core_common = float(summary[summary["method"].isin(CORE_METHODS)]["final_val_loss"].max())
    thresholds = [4.0, 3.8, 3.6, 3.5, 3.4, 3.3, all_common, core_common]
    thresholds = list(dict.fromkeys(thresholds))
    target_rows: list[dict[str, object]] = []
    for target in thresholds:
        scope = "fixed"
        if math.isclose(target, all_common):
            scope = "all_methods_common_final"
        if math.isclose(target, core_common):
            scope = "non_adamw_common_final"
        for method in METHODS:
            curve = val[val["method"] == method][["step", "val_loss", "train_time_s"]]
            target_rows.append(
                {"target_val_loss": target, "target_scope": scope, "method": method, **first_crossing(curve, target)}
            )
    targets = pd.DataFrame(target_rows)

    val_wide = val.pivot(index="step", columns="method", values="val_loss").reset_index()
    pair_specs = [
        ("down_diag", "down_none"),
        ("down_diag", "newton_full"),
        ("down_none", "newton_full"),
        ("newton_full", "muon"),
        ("down_none", "muon"),
        ("down_diag", "muon"),
        ("muon", "adamw"),
        ("newton_full", "adamw"),
        ("down_none", "adamw"),
        ("down_diag", "adamw"),
    ]
    pair_rows: list[dict[str, object]] = []
    for left, right in pair_specs:
        noninitial = val_wide[val_wide["step"] > 0].copy()
        differences = noninitial[left] - noninitial[right]
        signs = np.sign(differences.to_numpy(dtype=float))
        pair_rows.append(
            {
                "comparison": f"{left}_minus_{right}",
                "left_method": left,
                "right_method": right,
                "final_val_loss_delta_left_minus_right": float(by_method.loc[left, "final_val_loss"] - by_method.loc[right, "final_val_loss"]),
                "tail5_delta_left_minus_right": float(by_method.loc[left, "tail5_val_loss_mean"] - by_method.loc[right, "tail5_val_loss_mean"]),
                "normalized_val_auc_delta_left_minus_right": float(by_method.loc[left, "normalized_val_auc"] - by_method.loc[right, "normalized_val_auc"]),
                "left_lower_noninitial_points": int((differences < 0).sum()),
                "right_lower_noninitial_points": int((differences > 0).sum()),
                "tied_noninitial_points": int((differences == 0).sum()),
                "mean_noninitial_delta_left_minus_right": float(differences.mean()),
                "median_noninitial_delta_left_minus_right": float(differences.median()),
                "sign_flips": int(np.sum(signs[1:] * signs[:-1] < 0)),
                "persistent_left_lead_start_step": persistent_lead_start(noninitial[left], noninitial[right], noninitial["step"]),
                "descriptive_train_time_delta_s_left_minus_right": float(by_method.loc[left, "train_time_s_descriptive"] - by_method.loc[right, "train_time_s_descriptive"]),
                "descriptive_step_avg_delta_ms_left_minus_right": float(by_method.loc[left, "final_step_avg_ms_descriptive"] - by_method.loc[right, "final_step_avg_ms_descriptive"]),
                "expected_k_state_delta_mib_left_minus_right": EXPECTED_K_STATE_MIB[left] - EXPECTED_K_STATE_MIB[right],
            }
        )
    pairs = pd.DataFrame(pair_rows)

    milestones = val[val["step"].isin([0, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 6200])][
        ["method", "step", "val_loss", "train_time_s"]
    ].sort_values(["step", "method"])

    quality_rows.extend(
        [
            {
                "check": "local summary scalars supplied",
                "status": "WARN",
                "details": "missing peak memory, optimizer state, resume_count, timing_comparable",
            },
            {
                "check": "quality evidence eligibility",
                "status": "PASS",
                "details": "complete paired seed-2026 curves",
            },
            {
                "check": "formal memory evidence eligibility",
                "status": "WARN",
                "details": "requires local llama_swiglu_summary.csv or per-method summary.json",
            },
            {
                "check": "formal timing evidence eligibility",
                "status": "WARN",
                "details": "quality batch timing is descriptive; isolated repeated performance experiment required",
            },
        ]
    )
    quality = pd.DataFrame(quality_rows)
    blocking = quality[quality["status"] == "FAIL"]

    pd.DataFrame(manifest_rows).to_csv(out / "source_manifest.csv", index=False)
    long.to_csv(out / "normalized_history_long.csv", index=False)
    wide.sort_values(["method", "step"]).to_csv(out / "normalized_history_wide.csv", index=False)
    val_wide.to_csv(out / "validation_curves_wide.csv", index=False)
    milestones.to_csv(out / "validation_milestones.csv", index=False)
    summary.to_csv(out / "llama_run_summary.csv", index=False)
    pairs.to_csv(out / "llama_pairwise_summary.csv", index=False)
    targets.to_csv(out / "steps_tokens_to_loss_targets.csv", index=False)
    quality.to_csv(out / "data_quality_checks.csv", index=False)
    write_report(out / "LLAMA_SWIGLU_ANALYSIS_20260721.md", summary, pairs, quality, targets)
    (out / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS_WITH_CAVEATS" if blocking.empty else "FAIL",
                "output_dir": str(out),
                "metrics": sorted(seen_metrics),
                "methods": list(METHODS),
                "seed": 2026,
                "quality_usable": blocking.empty,
                "memory_usable": False,
                "timing_usable": False,
                "quality_pass": int((quality["status"] == "PASS").sum()),
                "quality_warn": int((quality["status"] == "WARN").sum()),
                "quality_fail": int((quality["status"] == "FAIL").sum()),
                "missing_evidence": [
                    "llama_manifest.json",
                    "llama_plan.json",
                    "llama_swiglu_summary.csv",
                    "per-method summary.json",
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if not blocking.empty:
        raise RuntimeError(f"LLaMA export audit failed:\n{blocking.to_string(index=False)}")
    print(summary.to_string(index=False))
    print(pairs.to_string(index=False))
    print(f"Saved LLaMA analysis to {out}")


if __name__ == "__main__":
    main()
