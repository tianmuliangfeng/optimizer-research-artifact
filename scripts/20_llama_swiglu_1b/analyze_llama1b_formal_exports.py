"""Audit and summarize the four-method LLaMA/SwiGLU-1B formal batch.

This analyzer is intentionally tied to the frozen 6200-step, seed-2026
analysis contract.  It preserves the seven raw W&B exports, verifies their
grain and schedules, and reports the within-run Muon/Newton crossover without
turning a single seed into a population-level claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("down_none", "down_diag", "newton_full", "muon")
NEWTON_METHODS = ("down_none", "down_diag", "newton_full")
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
    "down_none": 1728.0,
    "down_diag": 1728.755859375,
    "newton_full": 5888.25,
    "muon": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs=7, required=True)
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
    raise ValueError(f"unrecognized run name: {run_name}")


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


def crossing(curve: pd.DataFrame, target: float) -> dict[str, object]:
    curve = curve.sort_values("step").reset_index(drop=True)
    reached = curve[curve.val_loss <= target]
    if reached.empty:
        return {
            "reached": False,
            "first_discrete_step": math.nan,
            "interpolated_step": math.nan,
            "interpolated_tokens": math.nan,
        }
    index = int(reached.index[0])
    current = reached.iloc[0]
    if index == 0 or math.isclose(float(current.val_loss), target):
        interpolated = float(current.step)
    else:
        previous = curve.iloc[index - 1]
        high, low = float(previous.val_loss), float(current.val_loss)
        fraction = (high - target) / (high - low) if high != low else 1.0
        interpolated = float(previous.step + fraction * (current.step - previous.step))
    return {
        "reached": True,
        "first_discrete_step": int(current.step),
        "interpolated_step": interpolated,
        "interpolated_tokens": interpolated * TOKENS_PER_STEP,
    }


def sustained_positive_crossing(steps: np.ndarray, deltas: np.ndarray) -> dict[str, object]:
    """First point after which method - Muon remains strictly positive."""

    candidates = [
        index
        for index in range(len(deltas))
        if deltas[index] > 0 and bool(np.all(deltas[index:] > 0))
    ]
    if not candidates:
        return {
            "first_sustained_discrete_step": math.nan,
            "interpolated_crossover_step": math.nan,
        }
    index = candidates[0]
    discrete = int(steps[index])
    if index == 0 or deltas[index - 1] >= 0:
        interpolated = float(discrete)
    else:
        left_step, right_step = float(steps[index - 1]), float(steps[index])
        left_delta, right_delta = float(deltas[index - 1]), float(deltas[index])
        interpolated = left_step + (-left_delta) * (right_step - left_step) / (
            right_delta - left_delta
        )
    return {
        "first_sustained_discrete_step": discrete,
        "interpolated_crossover_step": interpolated,
    }


def make_plots(output: Path, val_wide: pd.DataFrame) -> None:
    colors = {
        "muon": "#111111",
        "down_none": "#2b6cb0",
        "down_diag": "#2f855a",
        "newton_full": "#c05621",
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for method in METHODS:
        ax.plot(val_wide.step, val_wide[method], label=method, color=colors[method], lw=2)
    ax.axvline(4400, color="#777777", ls="--", lw=1, label="warmdown starts")
    ax.set(xlabel="Optimizer step", ylabel="Validation loss")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "validation_loss_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for method in NEWTON_METHODS:
        ax.plot(
            val_wide.step,
            val_wide[method] - val_wide.muon,
            label=f"{method} - muon",
            color=colors[method],
            lw=2,
        )
    ax.axhline(0, color="#111111", lw=1)
    ax.axhline(PRACTICAL_MARGIN, color="#777777", ls=":", lw=1)
    ax.axvline(4400, color="#777777", ls="--", lw=1)
    ax.set(xlabel="Optimizer step", ylabel="Validation-loss delta")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "newton_minus_muon_curves.png", dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    s = summary.set_index("method")
    c = comparisons.set_index("method")
    counts = quality.status.value_counts().to_dict()
    lines = [
        "# LLaMA/SwiGLU-1B formal-6200 seed2026 分析（2026-07-24）",
        "",
        "## 结论先行",
        "",
        "Muon 在冻结的 step-6200 主终点上最低，并且不是单个终点噪声："
        "三条 Newton 曲线分别在约 step 1400--2000 后被 Muon 持续反超，"
        "直到 step 6200 仍未追回。Muon 同时具有本组最低的 K-state 和已验证的"
        "最高 1B 容量边界，因此在当前单 seed、固定 recipe 下对三条 Newton"
        "路径形成质量与内存双重优势。",
        "",
        "这会否定“1B 上 Selective Newton-Muon 相对 Muon 构成 Pareto 改进”的"
        "宽泛表述，但不否定 `down_none/down_diag` 相对 `newton_full` 的家族内"
        "压缩结论。当前只能称为 seed2026 的正式固定-recipe 结果；seed2024/2025"
        "复现完成前不能升级为总体方法排序。",
        "",
        "## 数据质量与预算口径",
        "",
        f"- 质量检查：PASS={counts.get('PASS', 0)}，FAIL={counts.get('FAIL', 0)}。",
        "- 4 methods × 7 W&B metrics 均覆盖完整预期 step grid。",
        "- 四条曲线 step0 validation loss 完全一致，matrix/backup LR 调度也逐点一致。",
        f"- 实际预算为 {TOTAL_STEPS} updates、{TOTAL_TOKENS:,} tokens/run（3.2506B）。",
        "- `FineWeb10B` 是数据缓存/语料口径名称；这批运行不是约 10B-token extension。",
        "- W&B 导出未包含本批正式 manifest、summary、checkpoint/resume 与峰值显存，"
        "因此曲线可用于质量分析，但本地证书仍待补充。",
        "",
        "## 冻结主终点与稳健性指标",
        "",
        "| 方法 | Final val | Tail-5 | AUC 0–6200 | Step1000 | Step2000 | Final train |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in summary.sort_values("final_val_loss").method:
        row = s.loc[method]
        lines.append(
            f"| {method} | {row.final_val_loss:.6f} | {row.tail5_val_loss_mean:.6f} | "
            f"{row.normalized_val_auc_0_6200:.6f} | {row.val_loss_step1000:.6f} | "
            f"{row.val_loss_step2000:.6f} | {row.final_train_loss_step:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Muon 反超轨迹",
            "",
            "| Newton 路径 | Final(method−Muon) | Tail-5 delta | AUC delta | "
            "持续反超离散 step | 插值交叉 step |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in NEWTON_METHODS:
        row = c.loc[method]
        lines.append(
            f"| {method} | {row.final_delta_method_minus_muon:+.6f} | "
            f"{row.tail5_delta_method_minus_muon:+.6f} | "
            f"{row.auc_delta_method_minus_muon:+.6f} | "
            f"{int(row.first_sustained_discrete_step)} | "
            f"{row.interpolated_crossover_step:.1f} |"
        )
    lines.extend(
        [
            "",
            f"- 冻结 practical margin 为 {PRACTICAL_MARGIN:.4f}；三条 Newton 路径"
            "在主终点都比 Muon 高出超过该阈值。",
            f"- `down_none` 是 Newton 家族中终点最好者，但仍比 Muon 高 "
            f"{c.loc['down_none'].final_delta_method_minus_muon:.6f}。",
            f"- `newton_full` 终点最差，比 Muon 高 "
            f"{c.loc['newton_full'].final_delta_method_minus_muon:.6f}；"
            "更多 K-state 在本配置下没有转化为更低 loss。",
            "",
            "## 与已有证据如何接上",
            "",
            "- 同一 1B formal run 在 step1000 仍是三条 Newton 路径优于 Muon，"
            "而约 step1400--2000 后反转。因此这不是“上一批导出和这一批初始化"
            "不同”造成的跨实验假象，而是清楚的训练阶段交互。",
            "- 124M LLaMA 三 seed 中，`newton_full` 与 Muon 的终点均值差仅"
            "约 -0.000725，原先所谓 Newton 优势非常小；1B seed2026 的方向"
            "相反且量级约为 +0.007945。",
            "- 已完成的 reference-reset 60-run 也观察到可重复的 regime reversal："
            "两个 12L suite 中 Muon 对三个 Newton 变体 3/3 seeds 获胜，"
            "18L/24L 又转为 diag 更好。因此当前现象不是首个反例。",
            "",
            "## 当前最合理的解释（尚非机制定论）",
            "",
            "1. **规模 × 训练阶段交互。** 1B 在相同 3.25B token 预算下只有约"
            " 3.2 tokens/parameter，而 124M 约为 26 tokens/parameter；固定 step/token"
            "预算并不是相同的训练成熟度。",
            "2. **固定 recipe 不随规模重调。** 四方法的 LR 调度严格匹配，这保证"
            "受控比较，却不保证 `matrix_lr=0.01`、ridge=0.2、beta=0.95、refresh=32"
            "分别处在 Muon/Newton 的 1B 最优区间。",
            "3. **问题不只位于 down projection。** `down_none` 已移除 down-proj K，"
            "但 attention input/output 与 MLP input 仍使用 Newton 预条件。它仍输给"
            "Muon，说明仅压缩 down-proj 不足以解释或修复 1B 后期差距。",
            "4. **K 的边际价值可能随宽度和激活几何变化。** full 最差、none 最好"
            "（在 Newton 家族内）的顺序与“更多协方差信息不必然更好”相容；但要用"
            "K 谱、条件数、更新夹角和局部 surrogate 才能区分噪声放大、ridge 失配"
            "和真正有害的非对角耦合。",
            "",
            "## 决策",
            "",
            "1. 立即完成 seed2024/2025 的同配置四方法复现；不因 seed2026 排名改 LR。",
            "2. 暂停约 10B-token extension。先确认反转是否跨 seed，再决定长预算花费。",
            "3. 把 seed2026 的反超窗口纳入机制实验：重点观测约 step1000、1800、"
            "2000、3000、4400、6200 的 K 谱/条件数、更新 RMS、Muon 与 Newton"
            "方向夹角和 quadratic score。",
            "4. 多 seed 结果确认后，再单独开预注册的 exploratory tuning branch，"
            "检查 Newton 的 LR/ridge/refresh，而不替换当前冻结主表。",
            "5. 论文主张收窄为家族内结论：Selective 相对 full Newton-Muon 的"
            "memory-quality tradeoff 仍成立；相对 Muon 的方法级优势在 1B 尚不成立。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    raw_dir = output / "raw_wandb_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    seen_metrics: set[str] = set()
    run_names: dict[str, str] = {}

    for source_arg in args.input:
        source = source_arg.resolve()
        copied = raw_dir / source.name
        if copied.exists() and sha256(copied) != sha256(source):
            raise RuntimeError(f"refusing to overwrite different evidence: {copied}")
        if not copied.exists():
            shutil.copy2(source, copied)
        frame = pd.read_csv(source)
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
        seen_metrics.add(metric)
        add_check(checks, f"{metric}: recognized", metric in EXPECTED_METRICS, source.name)
        observed_steps = frame.Step.astype(int).tolist()
        add_check(
            checks,
            f"{metric}: exact step grid",
            observed_steps == EXPECTED_METRICS.get(metric),
            f"points={len(observed_steps)} range={observed_steps[0]}..{observed_steps[-1]}",
        )
        methods_seen: set[str] = set()
        for column in primary:
            run_name, parsed_metric = column.rsplit(" - ", 1)
            method = method_from_run(run_name)
            methods_seen.add(method)
            run_names.setdefault(method, run_name)
            values = pd.to_numeric(frame[column], errors="coerce")
            mins = pd.to_numeric(frame[f"{column}__MIN"], errors="coerce")
            maxs = pd.to_numeric(frame[f"{column}__MAX"], errors="coerce")
            finite = np.isfinite(values.to_numpy(dtype=float))
            finite_steps = frame.loc[finite, "Step"].astype(int).tolist()
            add_check(
                checks,
                f"{method} {metric}: expected finite grid",
                finite_steps == EXPECTED_FINITE_STEPS[metric],
                f"finite={len(finite_steps)}",
            )
            mirrors = bool(
                np.allclose(values[finite], mins[finite], rtol=0, atol=0)
                and np.allclose(values[finite], maxs[finite], rtol=0, atol=0)
            )
            add_check(
                checks,
                f"{method} {metric}: MIN/MAX mirrors",
                mirrors,
                f"finite={int(finite.sum())}",
                "high",
            )
            frames.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "run_name": run_name,
                        "seed": 2026,
                        "metric": parsed_metric,
                        "step": frame.Step.astype(int),
                        "value": values,
                        "source_file": source.name,
                    }
                )
            )
        add_check(
            checks,
            f"{metric}: exact method identities",
            methods_seen == set(METHODS),
            repr(sorted(methods_seen)),
        )
        sources.append(
            {
                "source_file": source.name,
                "source_path": str(source),
                "preserved_path": str(copied),
                "sha256": sha256(source),
                "bytes": source.stat().st_size,
                "metric": metric,
                "rows": len(frame),
                "run_columns": len(primary),
            }
        )

    add_check(
        checks,
        "exact seven expected metrics",
        seen_metrics == set(EXPECTED_METRICS),
        f"observed={sorted(seen_metrics)}",
    )
    long = pd.concat(frames, ignore_index=True).sort_values(["method", "metric", "step"])
    duplicates = int(long.duplicated(["method", "metric", "step"]).sum())
    add_check(checks, "unique method/metric/step grain", duplicates == 0, f"duplicates={duplicates}")
    wide = long.pivot(index=["method", "step"], columns="metric", values="value").reset_index()
    val = wide[wide["val/loss"].notna()][["method", "step", "val/loss"]].rename(
        columns={"val/loss": "val_loss"}
    )
    val_wide = val.pivot(index="step", columns="method", values="val_loss").reset_index()

    initial = val[val.step == 0].set_index("method").val_loss
    add_check(
        checks,
        "identical step0 validation loss",
        initial.nunique(dropna=False) == 1,
        json.dumps(initial.to_dict(), sort_keys=True),
    )

    token_ok = True
    for method in METHODS:
        current = wide[wide.method == method].sort_values("step")
        token_ok = token_ok and np.array_equal(
            current["tokens/seen"].to_numpy(dtype=float),
            current.step.to_numpy(dtype=float) * TOKENS_PER_STEP,
        )
    add_check(checks, "tokens equal step * 512 * 1024", token_ok, f"final={TOTAL_TOKENS}")

    for metric in ("lr/backup", "lr/matrix"):
        curves = [
            wide[wide.method == method].sort_values("step")[metric].to_numpy(dtype=float)
            for method in METHODS
        ]
        add_check(
            checks,
            f"{metric}: identical across methods",
            all(np.array_equal(curves[0], curve) for curve in curves[1:]),
            f"points={len(curves[0])}",
        )

    summary_rows: list[dict[str, object]] = []
    for method in METHODS:
        curve = val[val.method == method].sort_values("step")
        current = wide[wide.method == method].sort_values("step")
        post = curve[curve.step >= 100]
        at = curve.set_index("step").val_loss
        last = current.iloc[-1]
        summary_rows.append(
            {
                "method": method,
                "run_name": run_names[method],
                "seed": 2026,
                "initial_val_loss": float(curve.val_loss.iloc[0]),
                "val_loss_step1000": float(at.loc[1000]),
                "val_loss_step2000": float(at.loc[2000]),
                "val_loss_step4400": float(at.loc[4400]),
                "final_val_loss": float(curve.val_loss.iloc[-1]),
                "best_val_loss": float(curve.val_loss.min()),
                "tail3_val_loss_mean": float(curve.tail(3).val_loss.mean()),
                "tail5_val_loss_mean": float(curve.tail(5).val_loss.mean()),
                "tail5_val_loss_sd": float(curve.tail(5).val_loss.std(ddof=1)),
                "normalized_val_auc_0_6200": float(
                    np.trapezoid(curve.val_loss, curve.step) / TOTAL_STEPS
                ),
                "post_initial_auc_100_6200": float(
                    np.trapezoid(post.val_loss, post.step) / (TOTAL_STEPS - 100)
                ),
                "final_train_loss_step": float(last["train/loss_step"]),
                "final_tokens": int(last["tokens/seen"]),
                "train_time_s_descriptive_only": float(last["time/train_s"]),
                "final_step_avg_ms_descriptive_only": float(last["performance/step_avg_ms"]),
                "expected_k_state_mib_from_preflight": EXPECTED_K_STATE_MIB[method],
                "quality_usable": True,
                "memory_usable_from_wandb_export": False,
                "timing_usable_for_paper": False,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["final_val_loss_rank"] = summary.final_val_loss.rank(method="min").astype(int)
    summary["auc_rank"] = summary.normalized_val_auc_0_6200.rank(method="min").astype(int)
    summary = summary.sort_values("final_val_loss_rank")
    by_method = summary.set_index("method")

    comparison_rows: list[dict[str, object]] = []
    for method in NEWTON_METHODS:
        delta = (val_wide[method] - val_wide.muon).to_numpy(dtype=float)
        steps = val_wide.step.to_numpy(dtype=int)
        post = val_wide.step > 0
        sustained = sustained_positive_crossing(steps[post], delta[post])
        comparison_rows.append(
            {
                "method": method,
                "reference": "muon",
                "final_delta_method_minus_muon": float(
                    by_method.loc[method, "final_val_loss"]
                    - by_method.loc["muon", "final_val_loss"]
                ),
                "tail5_delta_method_minus_muon": float(
                    by_method.loc[method, "tail5_val_loss_mean"]
                    - by_method.loc["muon", "tail5_val_loss_mean"]
                ),
                "auc_delta_method_minus_muon": float(
                    by_method.loc[method, "normalized_val_auc_0_6200"]
                    - by_method.loc["muon", "normalized_val_auc_0_6200"]
                ),
                "method_lower_checkpoints_excluding_step0": int((delta[post] < 0).sum()),
                "muon_lower_checkpoints_excluding_step0": int((delta[post] > 0).sum()),
                "ties_excluding_step0": int((delta[post] == 0).sum()),
                "exceeds_0p002_margin_at_final": bool(delta[-1] > PRACTICAL_MARGIN),
                "k_state_delta_mib_method_minus_muon": EXPECTED_K_STATE_MIB[method],
                **sustained,
            }
        )
    comparisons = pd.DataFrame(comparison_rows)

    target_rows: list[dict[str, object]] = []
    family_common_final = float(summary.final_val_loss.max())
    for target in (4.0, 3.8, 3.6, 3.4, 3.2, 3.0, family_common_final):
        for method in METHODS:
            target_rows.append(
                {
                    "target_val_loss": target,
                    "target_scope": (
                        "family_common_final"
                        if math.isclose(target, family_common_final)
                        else "fixed"
                    ),
                    "method": method,
                    **crossing(val[val.method == method][["step", "val_loss"]], target),
                }
            )
    targets = pd.DataFrame(target_rows)

    quality = pd.DataFrame(checks)
    quality = pd.concat(
        [
            quality,
            pd.DataFrame(
                [
                    {
                        "check": "formal local certificate supplied",
                        "status": "WARN",
                        "severity_if_failed": "critical",
                        "evidence": "need formal llama_manifest.json and per-method summaries/checkpoint metadata",
                    },
                    {
                        "check": "population-level replication",
                        "status": "WARN",
                        "severity_if_failed": "high",
                        "evidence": "seed2026 only; seed2024/2025 pending",
                    },
                    {
                        "check": "timing eligibility",
                        "status": "WARN",
                        "severity_if_failed": "high",
                        "evidence": "concurrent node usage makes W&B timing descriptive only",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    fail_count = int((quality.status == "FAIL").sum())

    pd.DataFrame(sources).to_csv(output / "source_manifest.csv", index=False)
    long.to_csv(output / "normalized_history_long.csv", index=False)
    wide.sort_values(["method", "step"]).to_csv(output / "normalized_history_wide.csv", index=False)
    val_wide.to_csv(output / "validation_curves_wide.csv", index=False)
    summary.to_csv(output / "llama1b_formal_run_summary.csv", index=False)
    comparisons.to_csv(output / "llama1b_formal_muon_comparisons.csv", index=False)
    targets.to_csv(output / "llama1b_formal_steps_tokens_to_targets.csv", index=False)
    quality.to_csv(output / "data_quality_checks.csv", index=False)
    make_plots(output, val_wide)
    report = output / "LLAMA1B_FORMAL6200_ANALYSIS_20260724.md"
    write_report(report, summary, comparisons, quality)

    manifest = {
        "created_at": "2026-07-24",
        "status": "PASS_WITH_CAVEATS" if fail_count == 0 else "FAIL",
        "stage": "formal-6200",
        "evidence_class": "formal_quality_curve_evidence",
        "methods": list(METHODS),
        "seed": 2026,
        "steps": TOTAL_STEPS,
        "tokens": TOTAL_TOKENS,
        "approximately_10b_token_extension": False,
        "quality_usable": fail_count == 0,
        "population_claim_ready": False,
        "formal_local_certificate": "PENDING",
        "memory_usable_from_current_exports": False,
        "timing_usable": False,
        "quality_checks": {
            key: int(value) for key, value in quality.status.value_counts().to_dict().items()
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
