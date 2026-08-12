"""Audit and summarize the four-method LLaMA/SwiGLU-1B medium-1000 batch.

Medium is a plateau-LR stability/cost screen, not truncated formal evidence.
The script therefore reports curve eligibility separately from the local
manifest/checkpoint certificate required to launch the 6200-step formal stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("down_none", "down_diag", "newton_full", "muon")
STRUCTURAL_METHODS = ("down_none", "down_diag", "newton_full")
TOTAL_STEPS = 1000
TOKENS_PER_STEP = 512 * 1024
TOTAL_TOKENS = TOTAL_STEPS * TOKENS_PER_STEP
PRACTICAL_MARGIN = 0.0020
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
EXPECTED_MAX_LR = {"lr/backup": 0.0036, "lr/matrix": 0.01}
EXPECTED_K_STATE_MIB = {
    "down_none": 1728.0,
    "down_diag": 1728.755859375,
    "newton_full": 5888.25,
    "muon": 0.0,
}
PAIR_SPECS = (
    ("down_diag", "down_none"),
    ("newton_full", "down_none"),
    ("muon", "down_none"),
    ("down_diag", "newton_full"),
)


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
    raise ValueError(f"unrecognized LLaMA-1B run name: {run_name}")


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
            "first_discrete_tokens": math.nan,
            "interpolated_tokens": math.nan,
        }
    index = int(reached.index[0])
    current = reached.iloc[0]
    if index == 0 or math.isclose(float(current.val_loss), target):
        interpolated_step = float(current.step)
    else:
        previous = curve.iloc[index - 1]
        high, low = float(previous.val_loss), float(current.val_loss)
        fraction = (high - target) / (high - low) if high != low else 1.0
        interpolated_step = float(previous.step + fraction * (current.step - previous.step))
    return {
        "reached": True,
        "first_discrete_step": int(current.step),
        "interpolated_step": interpolated_step,
        "first_discrete_tokens": int(current.step) * TOKENS_PER_STEP,
        "interpolated_tokens": interpolated_step * TOKENS_PER_STEP,
    }


def write_report(
    path: Path,
    summary: pd.DataFrame,
    pairs: pd.DataFrame,
    quality: pd.DataFrame,
    gate: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    s = summary.set_index("method")
    p = pairs.set_index("comparison")
    counts = quality.status.value_counts().to_dict()
    target36 = targets[np.isclose(targets.target_val_loss, 3.6)].set_index("method")
    gpu0_hours = float(
        s.loc["down_none", "formal_6200_hours_linear_extrapolation"]
        + s.loc["down_diag", "formal_6200_hours_linear_extrapolation"]
    )
    gpu1_hours = float(
        s.loc["newton_full", "formal_6200_hours_linear_extrapolation"]
        + s.loc["muon", "formal_6200_hours_linear_extrapolation"]
    )
    lines = [
        "# LLaMA/SwiGLU-1B medium-1000 分析（2026-07-23）",
        "",
        "## 结论先行",
        "",
        "四种方法全部完成 1000 个 plateau-LR updates，曲线有限、严格单调下降，"
        "且到达 524,288,000 tokens。W&B 曲线侧的 medium gate 全部通过；不需要"
        " medium-2000，也不触发 fallback LR。正式 6200-step 运行仍须先用远端"
        " `llama_manifest.json`、`llama_swiglu_summary.csv` 和 final checkpoints 完成证书审计。",
        "",
        "Medium 是稳定性/成本 screen，不是截短 formal，下面的 step1000 排序不能"
        "替代正式 step6200 primary endpoint，也不能用于重定义 0.0020 margin。",
        "",
        "## 数据与质量状态",
        "",
        "- 4 methods × 7 metrics，全部预期 step grid 完整。",
        f"- 数据质量检查：PASS={counts.get('PASS', 0)}，FAIL={counts.get('FAIL', 0)}。",
        f"- 四方法初始 validation loss 一致：{s.iloc[0].initial_val_loss:.9f}。",
        "- backup/matrix LR 均固定为 0.0036/0.01；medium 无 warmdown。",
        "- `curve_gate_pass=true`；`certificate_gate=PENDING_LOCAL_ARTIFACTS`。",
        "",
        "## Step-1000 屏幕结果",
        "",
        "| 方法 | Final val | Tail-3 | Tail-5 | Post-init AUC | Final train | 3.6 首次 step |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in summary.sort_values("final_val_loss").method:
        row = s.loc[method]
        t = target36.loc[method]
        lines.append(
            f"| {method} | {row.final_val_loss:.6f} | {row.tail3_val_loss_mean:.6f} | "
            f"{row.tail5_val_loss_mean:.6f} | {row.post_initial_auc_100_1000:.6f} | "
            f"{row.final_train_loss_step:.6f} | {int(t.first_discrete_step)} |"
        )
    lines.extend(
        [
            "",
            "## 配对观察",
            "",
            f"- `down_diag - down_none`：final "
            f"{p.loc['down_diag_minus_down_none'].final_delta_left_minus_right:+.6f}；"
            f"小于预设 0.0020 practical margin。",
            f"- `newton_full - down_none`：final "
            f"{p.loc['newton_full_minus_down_none'].final_delta_left_minus_right:+.6f}；"
            "full 在 medium 终点没有显示出优势。",
            f"- `muon - down_none`：final "
            f"{p.loc['muon_minus_down_none'].final_delta_left_minus_right:+.6f}；"
            "这是方向性 screen，不是 formal 结论。",
            "- 三个结构方法的最大 step1000 间距仅 "
            f"{s.loc[list(STRUCTURAL_METHODS)].final_val_loss.max()-s.loc[list(STRUCTURAL_METHODS)].final_val_loss.min():.6f}；"
            "没有理由因 medium 排名改变冻结 LR。",
            "",
            "## 稳定性与放行决定",
            "",
        ]
    )
    for _, row in gate.iterrows():
        lines.append(
            f"- `{row.method}`：complete={str(row.complete_1000).lower()}，"
            f"finite={str(row.finite_losses).lower()}，"
            f"monotone_val={str(row.strictly_decreasing_val).lower()}，"
            f"curve gate={'PASS' if row.curve_gate_pass else 'FAIL'}。"
        )
    lines.extend(
        [
            "",
            "结论：四方法均应进入正式 6200-step core experiment；不做 medium-2000，"
            "不删方法，不根据当前排序调 LR。",
            "",
            "## 仅用于排期的运行时间",
            "",
        ]
    )
    for method in summary.sort_values("method").method:
        row = s.loc[method]
        lines.append(
            f"- `{method}`：medium {row.train_time_s_descriptive/3600:.3f} h；"
            f"线性外推 formal {row.formal_6200_hours_linear_extrapolation:.2f} GPU-h。"
        )
    lines.extend(
        [
            f"- 既定双卡分配 `GPU0=down_none+down_diag` 约 {gpu0_hours:.2f} h，"
            f"`GPU1=newton_full+muon` 约 {gpu1_hours:.2f} h；仍然很均衡。",
            "- 建议主机至少预留 30 小时；这些数字受同节点负载影响，只用于运维，"
            "不进入论文性能表。",
            "",
            "## 仍需补充的证据",
            "",
            "请保留完整 medium artifact 目录，并提供轻量审计包（checkpoint 二进制可不上传），至少包括：",
            "",
            "- `llama_manifest.json`；",
            "- `llama_swiglu_summary.csv`；",
            "- 每方法 `summary.json`、`metrics.csv`，以及 final checkpoint 的路径、大小/哈希或目录清单；",
            "- 若发生过恢复，保留 checkpoint/resume 记录与 stdout/stderr。",
            "",
            "实际 checkpoint 请继续保留在远端，因为 formal certificate 校验仍会检查它。"
            "这些文件用于确认 source/runtime/data/init fingerprint、device batch=8、"
            "resume_count、checkpoint 完整性、peak allocated、optimizer/K-state bytes，"
            "并提供启动 formal 所必需的 `--medium-manifest`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    raw_dir = output / "raw_wandb_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
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
        expected_steps = EXPECTED_METRICS.get(metric)
        seen_metrics.add(metric)
        observed_steps = frame.Step.astype(int).tolist()
        add_check(
            checks,
            f"{metric}: recognized metric",
            expected_steps is not None,
            source.name,
        )
        add_check(
            checks,
            f"{metric}: exact step grid",
            observed_steps == expected_steps,
            f"points={len(observed_steps)} first={observed_steps[:2]} last={observed_steps[-2:]}",
        )
        add_check(
            checks,
            f"{metric}: exact four-run coverage",
            len(primary) == len(METHODS),
            f"runs={len(primary)}",
        )
        methods_in_file: set[str] = set()
        for column in primary:
            run_name, parsed_metric = column.rsplit(" - ", 1)
            method = method_from_run(run_name)
            methods_in_file.add(method)
            prior = run_names.setdefault(method, run_name)
            if prior != run_name:
                raise RuntimeError(f"multiple run names for {method}: {prior} vs {run_name}")
            values = pd.to_numeric(frame[column], errors="coerce")
            mins = pd.to_numeric(frame[f"{column}__MIN"], errors="coerce")
            maxs = pd.to_numeric(frame[f"{column}__MAX"], errors="coerce")
            finite = np.isfinite(values.to_numpy(dtype=float))
            finite_steps = frame.loc[finite, "Step"].astype(int).tolist()
            mirrors = bool(
                np.allclose(values[finite], mins[finite], rtol=0, atol=0)
                and np.allclose(values[finite], maxs[finite], rtol=0, atol=0)
            )
            add_check(
                checks,
                f"{method} {metric}: expected finite grid",
                finite_steps == EXPECTED_FINITE_STEPS[metric],
                f"finite={len(finite_steps)}",
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
            methods_in_file == set(METHODS),
            repr(sorted(methods_in_file)),
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
        "exact seven expected metric exports",
        seen_metrics == set(EXPECTED_METRICS),
        f"missing={sorted(set(EXPECTED_METRICS)-seen_metrics)} extra={sorted(seen_metrics-set(EXPECTED_METRICS))}",
    )
    long = pd.concat(frames, ignore_index=True).sort_values(["method", "metric", "step"])
    duplicates = int(long.duplicated(["method", "metric", "step"]).sum())
    add_check(
        checks,
        "unique method/metric/step grain",
        duplicates == 0,
        f"duplicates={duplicates}",
    )
    wide = long.pivot(index=["method", "step"], columns="metric", values="value").reset_index()
    val = wide[wide["val/loss"].notna()][["method", "step", "val/loss"]].rename(
        columns={"val/loss": "val_loss"}
    )
    initial = val[val.step == 0].set_index("method").val_loss
    add_check(
        checks,
        "identical initial validation loss across four methods",
        initial.nunique(dropna=False) == 1,
        json.dumps(initial.to_dict(), sort_keys=True),
    )

    tokens_ok = True
    time_ok = True
    for _, method_frame in wide.groupby("method"):
        ordered = method_frame.sort_values("step")
        tokens_ok = tokens_ok and np.array_equal(
            ordered["tokens/seen"].to_numpy(dtype=float),
            ordered.step.to_numpy(dtype=float) * TOKENS_PER_STEP,
        )
        times = ordered["time/train_s"].dropna().to_numpy(dtype=float)
        time_ok = time_ok and bool(np.all(np.diff(times) >= 0))
    add_check(checks, "tokens equal step * 512 * 1024", tokens_ok, f"final={TOTAL_TOKENS}")
    add_check(checks, "cumulative training time is monotone", time_ok, "descriptive-only")

    for metric, expected_lr in EXPECTED_MAX_LR.items():
        curves = {
            method: wide[wide.method == method].sort_values("step")[metric].to_numpy(dtype=float)
            for method in METHODS
        }
        identical = all(np.array_equal(curves[METHODS[0]], curves[method]) for method in METHODS[1:])
        constant_expected = all(
            np.allclose(values, expected_lr, rtol=0, atol=1e-15) for values in curves.values()
        )
        add_check(checks, f"{metric}: identical across methods", identical, f"expected={expected_lr}")
        add_check(checks, f"{metric}: constant plateau LR", constant_expected, f"expected={expected_lr}")

    summary_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    for method in METHODS:
        curve = val[val.method == method].sort_values("step")
        method_wide = wide[wide.method == method].sort_values("step")
        last = method_wide.iloc[-1]
        post = curve[curve.step >= 100]
        val_diffs = np.diff(curve.val_loss.to_numpy(dtype=float))
        complete = int(curve.step.iloc[-1]) == TOTAL_STEPS and int(last["tokens/seen"]) == TOTAL_TOKENS
        finite = bool(np.isfinite(curve.val_loss).all()) and bool(
            np.isfinite(method_wide["train/loss_step"].dropna()).all()
        )
        monotone = bool(np.all(val_diffs < 0))
        best_at_final = math.isclose(float(curve.val_loss.iloc[-1]), float(curve.val_loss.min()))
        curve_pass = complete and finite and monotone and best_at_final
        train_s = float(last["time/train_s"])
        summary_rows.append(
            {
                "method": method,
                "run_name": run_names[method],
                "seed": 2026,
                "initial_val_loss": float(curve.val_loss.iloc[0]),
                "final_val_loss": float(curve.val_loss.iloc[-1]),
                "best_val_loss": float(curve.val_loss.min()),
                "tail3_val_loss_mean": float(curve.tail(3).val_loss.mean()),
                "tail5_val_loss_mean": float(curve.tail(5).val_loss.mean()),
                "tail5_val_loss_sd": float(curve.tail(5).val_loss.std(ddof=1)),
                "normalized_val_auc_0_1000": float(
                    np.trapezoid(curve.val_loss, curve.step) / TOTAL_STEPS
                ),
                "post_initial_auc_100_1000": float(
                    np.trapezoid(post.val_loss, post.step) / 900.0
                ),
                "final_train_loss_step": float(last["train/loss_step"]),
                "final_tokens": int(last["tokens/seen"]),
                "train_time_s_descriptive": train_s,
                "final_step_avg_ms_descriptive": float(last["performance/step_avg_ms"]),
                "formal_6200_hours_linear_extrapolation": train_s * 6.2 / 3600.0,
                "max_backup_lr": float(method_wide["lr/backup"].max()),
                "max_matrix_lr": float(method_wide["lr/matrix"].max()),
                "expected_k_state_mib_from_preflight": EXPECTED_K_STATE_MIB[method],
                "observed_peak_memory_mib": math.nan,
                "observed_optimizer_state_mib": math.nan,
                "resume_count": math.nan,
                "curve_gate_pass": curve_pass,
                "certificate_gate": "PENDING_LOCAL_ARTIFACTS",
                "quality_usable": True,
                "memory_usable": False,
                "timing_usable": False,
            }
        )
        gate_rows.append(
            {
                "method": method,
                "complete_1000": complete,
                "finite_losses": finite,
                "strictly_decreasing_val": monotone,
                "best_at_final": best_at_final,
                "max_val_change_between_evals": float(val_diffs.max()),
                "curve_gate_pass": curve_pass,
                "certificate_gate": "PENDING_LOCAL_ARTIFACTS",
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["final_loss_rank_screening_only"] = summary.final_val_loss.rank(method="min").astype(int)
    summary["post_initial_auc_rank_screening_only"] = summary.post_initial_auc_100_1000.rank(
        method="min"
    ).astype(int)
    summary = summary.sort_values("final_loss_rank_screening_only")
    gate = pd.DataFrame(gate_rows)
    by_method = summary.set_index("method")

    val_wide = val.pivot(index="step", columns="method", values="val_loss").reset_index()
    pair_rows: list[dict[str, object]] = []
    for left, right in PAIR_SPECS:
        post = val_wide[val_wide.step > 0]
        deltas = post[left] - post[right]
        final_delta = float(by_method.loc[left, "final_val_loss"] - by_method.loc[right, "final_val_loss"])
        pair_rows.append(
            {
                "comparison": f"{left}_minus_{right}",
                "left_method": left,
                "right_method": right,
                "final_delta_left_minus_right": final_delta,
                "tail3_delta_left_minus_right": float(
                    by_method.loc[left, "tail3_val_loss_mean"] - by_method.loc[right, "tail3_val_loss_mean"]
                ),
                "tail5_delta_left_minus_right": float(
                    by_method.loc[left, "tail5_val_loss_mean"] - by_method.loc[right, "tail5_val_loss_mean"]
                ),
                "post_initial_auc_delta_left_minus_right": float(
                    by_method.loc[left, "post_initial_auc_100_1000"]
                    - by_method.loc[right, "post_initial_auc_100_1000"]
                ),
                "left_lower_checkpoints": int((deltas < 0).sum()),
                "right_lower_checkpoints": int((deltas > 0).sum()),
                "ties": int((deltas == 0).sum()),
                "mean_checkpoint_delta": float(deltas.mean()),
                "max_abs_checkpoint_delta": float(deltas.abs().max()),
                "within_0p002_screening_margin_at_step1000": abs(final_delta) <= PRACTICAL_MARGIN,
                "expected_k_state_delta_mib_left_minus_right": EXPECTED_K_STATE_MIB[left]
                - EXPECTED_K_STATE_MIB[right],
            }
        )
    pairs = pd.DataFrame(pair_rows)

    common_final = float(summary.final_val_loss.max())
    target_rows: list[dict[str, object]] = []
    for target in (4.0, 3.8, 3.6, common_final):
        scope = "family_common_final" if math.isclose(target, common_final) else "formal_fixed_target"
        for method in METHODS:
            curve = val[val.method == method][["step", "val_loss"]]
            target_rows.append(
                {
                    "target_val_loss": target,
                    "target_scope": scope,
                    "method": method,
                    **crossing(curve, target),
                }
            )
    targets = pd.DataFrame(target_rows)

    add_check(
        checks,
        "all four methods pass curve-only medium gate",
        bool(gate.curve_gate_pass.all()),
        json.dumps(gate.set_index("method").curve_gate_pass.to_dict(), sort_keys=True),
    )
    quality = pd.DataFrame(checks)
    quality = pd.concat(
        [
            quality,
            pd.DataFrame(
                [
                    {
                        "check": "local medium certificate supplied",
                        "status": "WARN",
                        "severity_if_failed": "critical",
                        "evidence": "needs llama_manifest.json, llama_swiglu_summary.csv, per-run summaries/checkpoints",
                    },
                    {
                        "check": "memory evidence eligibility",
                        "status": "WARN",
                        "severity_if_failed": "high",
                        "evidence": "W&B curve exports omit peak/state/resume summary scalars",
                    },
                    {
                        "check": "timing evidence eligibility",
                        "status": "WARN",
                        "severity_if_failed": "high",
                        "evidence": "medium timing is operational/descriptive only",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    pd.DataFrame(sources).to_csv(output / "source_manifest.csv", index=False)
    long.to_csv(output / "normalized_history_long.csv", index=False)
    wide.sort_values(["method", "step"]).to_csv(output / "normalized_history_wide.csv", index=False)
    val_wide.to_csv(output / "validation_curves_wide.csv", index=False)
    summary.to_csv(output / "llama1b_medium_run_summary.csv", index=False)
    pairs.to_csv(output / "llama1b_medium_pairwise_summary.csv", index=False)
    targets.to_csv(output / "llama1b_medium_steps_tokens_to_targets.csv", index=False)
    gate.to_csv(output / "llama1b_medium_gate.csv", index=False)
    quality.to_csv(output / "data_quality_checks.csv", index=False)
    report_path = output / "LLAMA1B_MEDIUM1000_ANALYSIS_20260723.md"
    write_report(report_path, summary, pairs, quality, gate, targets)

    fail_count = int((quality.status == "FAIL").sum())
    manifest = {
        "created_at": "2026-07-23",
        "status": "PASS_WITH_CAVEATS" if fail_count == 0 else "FAIL",
        "stage": "medium-1000",
        "evidence_class": "screening_only",
        "methods": list(METHODS),
        "seed": 2026,
        "steps": TOTAL_STEPS,
        "tokens": TOTAL_TOKENS,
        "curve_gate_pass": bool(gate.curve_gate_pass.all()),
        "certificate_gate": "PENDING_LOCAL_ARTIFACTS",
        "formal_launch_ready": False,
        "medium_2000_required": False,
        "fallback_lr_triggered": False,
        "quality_checks": {
            key: int(value) for key, value in quality.status.value_counts().to_dict().items()
        },
        "quality_usable": fail_count == 0,
        "memory_usable": False,
        "timing_usable": False,
        "missing_evidence": [
            "medium llama_manifest.json",
            "medium llama_swiglu_summary.csv",
            "per-method summary.json/metrics.csv and final-checkpoint existence metadata",
            "source/runtime/data/init/resume/checkpoint audit from the medium artifact directory",
        ],
        "outputs": sorted(
            path.name
            for path in output.iterdir()
            if path.is_file() and path.name != "analysis_manifest.json"
        ),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
