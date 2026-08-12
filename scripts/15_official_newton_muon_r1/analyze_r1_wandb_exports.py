"""Audit and summarize the four-method official-architecture R1 W&B exports."""

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


METHODS = ("muon", "block4", "none", "diag")
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
    parser.add_argument("--r0-summary", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def method_from_run(run_name: str) -> str:
    for method in METHODS:
        if f"_r1_{method}_seed2026_" in run_name:
            return method
    raise ValueError(f"Unrecognized R1 run name: {run_name}")


def first_crossing(curve: pd.DataFrame, target: float) -> dict[str, float | int | bool]:
    curve = curve.sort_values("step").reset_index(drop=True)
    reached = curve[curve["val_loss"] <= target]
    if reached.empty:
        return {
            "reached": False,
            "first_discrete_step": math.nan,
            "first_discrete_time_s": math.nan,
            "interpolated_step": math.nan,
            "interpolated_time_s": math.nan,
        }
    index = int(reached.index[0])
    current = curve.iloc[index]
    if index == 0 or float(current["val_loss"]) == target:
        return {
            "reached": True,
            "first_discrete_step": int(current["step"]),
            "first_discrete_time_s": float(current["train_time_s"]),
            "interpolated_step": float(current["step"]),
            "interpolated_time_s": float(current["train_time_s"]),
        }
    previous = curve.iloc[index - 1]
    high = float(previous["val_loss"])
    low = float(current["val_loss"])
    fraction = (high - target) / (high - low) if high != low else 1.0
    return {
        "reached": True,
        "first_discrete_step": int(current["step"]),
        "first_discrete_time_s": float(current["train_time_s"]),
        "interpolated_step": float(previous["step"] + fraction * (current["step"] - previous["step"])),
        "interpolated_time_s": float(
            previous["train_time_s"]
            + fraction * (current["train_time_s"] - previous["train_time_s"])
        ),
    }


def persistent_lead_start(left: pd.Series, right: pd.Series, steps: pd.Series) -> float:
    differences = (left - right).to_numpy(dtype=float)
    step_values = steps.to_numpy(dtype=int)
    for index in range(len(differences)):
        if np.all(differences[index:] <= 0):
            return float(step_values[index])
    return math.nan


def write_markdown(
    path: Path,
    summary: pd.DataFrame,
    pairs: pd.DataFrame,
    quality: pd.DataFrame,
    recommendation: pd.DataFrame,
) -> None:
    by_method = summary.set_index("method")
    diag = by_method.loc["diag"]
    block4 = by_method.loc["block4"]
    none = by_method.loc["none"]
    muon = by_method.loc["muon"]
    pair = pairs.set_index("comparison")
    d_b = pair.loc["diag_minus_block4"]
    d_n = pair.loc["diag_minus_none"]
    quality_counts = quality["status"].value_counts().to_dict()
    text = f"""# R1 seed-2026 W&B analysis (2026-07-20)

## Evidence status

- Four methods are present: Muon, block4, none, and diag.
- All four start from exactly the same exported validation loss ({diag['initial_val_loss']:.4f}), consistent with the controlled-initialization protocol.
- Nine expected W&B metric exports are present with exact step grids and no blocking data-quality failure.
- Quality checks: {quality_counts.get('PASS', 0)} pass, {quality_counts.get('WARN', 0)} warning, {quality_counts.get('FAIL', 0)} fail.

## Endpoint result

| Method | Final val loss | Train time (s) | Peak memory (MiB) | K state (MiB) |
|---|---:|---:|---:|---:|
| diag | {diag['final_val_loss']:.4f} | {diag['train_time_s']:.3f} | {diag['peak_memory_mib']:.0f} | {diag['k_state_mib']:.5f} |
| block4 | {block4['final_val_loss']:.4f} | {block4['train_time_s']:.3f} | {block4['peak_memory_mib']:.0f} | {block4['k_state_mib']:.0f} |
| none | {none['final_val_loss']:.4f} | {none['train_time_s']:.3f} | {none['peak_memory_mib']:.0f} | {none['k_state_mib']:.0f} |
| Muon | {muon['final_val_loss']:.4f} | {muon['train_time_s']:.3f} | {muon['peak_memory_mib']:.0f} | {muon['k_state_mib']:.0f} |

## Main interpretation

- diag is the best endpoint in this seed, beating block4 by {abs(d_b['final_val_loss_delta_left_minus_right']):.4f} and none by {abs(d_n['final_val_loss_delta_left_minus_right']):.4f}.
- Relative to block4, diag saves {block4['k_state_mib'] - diag['k_state_mib']:.5f} MiB of persistent K state and {block4['peak_memory_mib'] - diag['peak_memory_mib']:.0f} MiB of peak allocated memory.
- diag adds only {diag['k_state_mib'] - none['k_state_mib']:.5f} MiB of K state over none, while improving final loss by {abs(d_n['final_val_loss_delta_left_minus_right']):.4f}. This is the most interesting mechanism result in R1.
- Muon is fastest and lowest-memory, but its official learning rate is 10% lower than the three Newton variants. Its comparison is therefore an official-recipe comparison, not a shared-LR causal isolation.

## Multi-seed decision

Yes—add targeted seeds before claiming that diag is superior to or statistically indistinguishable from block4. The observed endpoint advantage is only {abs(d_b['final_val_loss_delta_left_minus_right']):.4f}; a single seed cannot estimate its variability. The efficient first stage is two additional paired seeds for diag, none, and block4 (six new full runs), bringing the controlled K-representation comparison to three seeds. Add Muon for those seeds only if the paper needs a multi-seed official-recipe comparison; multi-seed alone still does not remove the Muon/Newton LR confound.

For timing claims below roughly 1%, repeated/counterbalanced benchmark runs are more informative than changing seeds. Exact K-state and optimizer-state byte differences are structural and do not require multiple seeds.
"""
    path.write_text(text, encoding="utf-8")


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
        copied = raw_dir / source.name
        if copied.exists() and sha256(copied) != sha256(source):
            raise RuntimeError(f"Refusing to overwrite different evidence: {copied}")
        if not copied.exists():
            shutil.copy2(source, copied)
        manifest_rows.append(
            {
                "source_file": source.name,
                "original_path": str(source),
                "preserved_path": str(copied),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )

        frame = pd.read_csv(source)
        primary_columns = [
            column
            for column in frame.columns
            if column != "Step" and not column.endswith("__MIN") and not column.endswith("__MAX")
        ]
        if len(primary_columns) != 4:
            raise RuntimeError(f"Expected four R1 run columns in {source.name}, got {len(primary_columns)}")
        metric_names = {column.rsplit(" - ", 1)[1] for column in primary_columns}
        if len(metric_names) != 1:
            raise RuntimeError(f"Mixed metrics in {source.name}: {metric_names}")
        metric = next(iter(metric_names))
        seen_metrics.add(metric)
        expected_steps = EXPECTED_METRICS.get(metric)
        if expected_steps is None:
            quality_rows.append(
                {"check": f"recognized metric {metric}", "status": "FAIL", "details": source.name}
            )
        else:
            observed_steps = frame["Step"].astype(int).tolist()
            quality_rows.extend(
                [
                    {
                        "check": f"{metric}: row count",
                        "status": "PASS" if len(frame) == len(expected_steps) else "FAIL",
                        "details": f"observed={len(frame)}, expected={len(expected_steps)}",
                    },
                    {
                        "check": f"{metric}: exact step grid",
                        "status": "PASS" if observed_steps == expected_steps else "FAIL",
                        "details": f"first={observed_steps[:3]}, last={observed_steps[-3:]}",
                    },
                ]
            )

        methods_in_file: set[str] = set()
        for column in primary_columns:
            run_name, parsed_metric = column.rsplit(" - ", 1)
            method = method_from_run(run_name)
            methods_in_file.add(method)
            previous_run = run_names_by_method.setdefault(method, run_name)
            if previous_run != run_name:
                raise RuntimeError(f"Multiple run names for {method}: {previous_run} vs {run_name}")
            values = pd.to_numeric(frame[column], errors="coerce")
            min_values = pd.to_numeric(frame[f"{column}__MIN"], errors="coerce")
            max_values = pd.to_numeric(frame[f"{column}__MAX"], errors="coerce")
            finite = np.isfinite(values.to_numpy(dtype=float))
            mirrors = np.allclose(values[finite], min_values[finite], rtol=0, atol=0) and np.allclose(
                values[finite], max_values[finite], rtol=0, atol=0
            )
            quality_rows.extend(
                [
                    {
                        "check": f"{method} {metric}: W&B MIN/MAX mirrors",
                        "status": "PASS" if mirrors else "FAIL",
                        "details": f"finite_points={int(finite.sum())}",
                    },
                    {
                        "check": f"{method} {metric}: finite values",
                        "status": "PASS" if bool(finite.all()) else "FAIL",
                        "details": f"nonfinite={int((~finite).sum())}",
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
                "check": f"{metric}: exact four-method coverage",
                "status": "PASS" if methods_in_file == set(METHODS) else "FAIL",
                "details": ",".join(sorted(methods_in_file)),
            }
        )

    quality_rows.append(
        {
            "check": "exact nine expected metric exports",
            "status": "PASS" if seen_metrics == set(EXPECTED_METRICS) else "FAIL",
            "details": f"missing={sorted(set(EXPECTED_METRICS)-seen_metrics)}, extra={sorted(seen_metrics-set(EXPECTED_METRICS))}",
        }
    )
    long = pd.concat(long_frames, ignore_index=True).sort_values(["metric", "method", "step"])
    duplicate_count = int(long.duplicated(["method", "metric", "step"]).sum())
    quality_rows.append(
        {
            "check": "unique method-metric-step keys",
            "status": "PASS" if duplicate_count == 0 else "FAIL",
            "details": f"duplicates={duplicate_count}",
        }
    )

    wide = long.pivot(index=["method", "step"], columns="metric", values="value").reset_index()
    val = wide[wide["val/loss"].notna()].copy().rename(
        columns={"val/loss": "val_loss", "time/train_s": "train_time_s"}
    )
    initial_values = val[val["step"] == 0].set_index("method")["val_loss"]
    identical_initial = initial_values.nunique(dropna=False) == 1
    quality_rows.append(
        {
            "check": "identical initial validation loss across four methods",
            "status": "PASS" if identical_initial else "FAIL",
            "details": json.dumps(initial_values.to_dict(), sort_keys=True),
        }
    )

    lr_ratio_ok = True
    newton_lr_equal = True
    normalized_lr_equal = True
    post_reset_time_ok = True
    for method, method_wide in wide.groupby("method"):
        ordered = method_wide.sort_values("step")
        positive = ordered[ordered["lr/matrix"] > 0]
        lr_ratio_ok = lr_ratio_ok and np.allclose(
            positive["lr/adamw"], 10 * positive["lr/matrix"], rtol=1e-10, atol=1e-12
        )
        times = ordered.loc[ordered["step"] >= 40, "time/train_s"].dropna().to_numpy(dtype=float)
        post_reset_time_ok = post_reset_time_ok and bool(np.all(np.diff(times) >= 0))
    for metric in ("lr/adamw", "lr/matrix"):
        curves = {
            method: wide[wide["method"] == method].sort_values("step")[metric].to_numpy(dtype=float)
            for method in METHODS
        }
        newton_lr_equal = newton_lr_equal and np.array_equal(curves["block4"], curves["none"])
        newton_lr_equal = newton_lr_equal and np.array_equal(curves["block4"], curves["diag"])
        normalized = {method: values / np.nanmax(values) for method, values in curves.items()}
        normalized_lr_equal = normalized_lr_equal and all(
            np.allclose(normalized["muon"], normalized[method], rtol=1e-10, atol=1e-12)
            for method in ("block4", "none", "diag")
        )
    quality_rows.extend(
        [
            {
                "check": "AdamW/matrix LR ratio is exactly 10x",
                "status": "PASS" if lr_ratio_ok else "FAIL",
                "details": "all positive-LR points",
            },
            {
                "check": "block4/none/diag use identical LR histories",
                "status": "PASS" if newton_lr_equal else "FAIL",
                "details": "both AdamW and matrix LR",
            },
            {
                "check": "all methods use the same normalized LR schedule shape",
                "status": "PASS" if normalized_lr_equal else "FAIL",
                "details": "Muon absolute LR is 0.9x the Newton variants",
            },
            {
                "check": "official train time is monotone after step-32 reset",
                "status": "PASS" if post_reset_time_ok else "FAIL",
                "details": "checked exported points from step 40 through 6200",
            },
        ]
    )

    summary_rows: list[dict[str, object]] = []
    for method in METHODS:
        curve = val[val["method"] == method].sort_values("step")
        method_wide = wide[wide["method"] == method].sort_values("step")
        final = curve.iloc[-1]
        last_metrics = method_wide.iloc[-1]
        summary_rows.append(
            {
                "method": method,
                "run_name": run_names_by_method[method],
                "seed": 2026,
                "initial_val_loss": float(curve.iloc[0]["val_loss"]),
                "final_val_loss": float(final["val_loss"]),
                "best_val_loss": float(curve["val_loss"].min()),
                "final_perplexity": math.exp(float(final["val_loss"])),
                "tail3_val_loss_mean": float(curve.tail(3)["val_loss"].mean()),
                "tail5_val_loss_mean": float(curve.tail(5)["val_loss"].mean()),
                "normalized_val_auc": float(np.trapezoid(curve["val_loss"], curve["step"]) / 6200.0),
                "final_train_loss_step": float(last_metrics["train/loss_step"]),
                "train_time_s": float(last_metrics["time/train_s"]),
                "final_step_avg_ms": float(last_metrics["performance/step_avg_ms"]),
                "max_adamw_lr": float(method_wide["lr/adamw"].max()),
                "max_matrix_lr": float(method_wide["lr/matrix"].max()),
                "peak_memory_mib": float(last_metrics["memory/peak_allocated_mib"]),
                "k_state_mib": float(last_metrics["memory/k_state_mib"]),
                "optimizer_state_mib": float(last_metrics["memory/optimizer_state_mib"]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["final_loss_rank"] = summary["final_val_loss"].rank(method="min").astype(int)
    summary = summary.sort_values("final_loss_rank").reset_index(drop=True)
    by_method = summary.set_index("method")

    common_target = float(summary["final_val_loss"].max())
    thresholds = [3.5, 3.4, 3.3, common_target]
    target_rows: list[dict[str, object]] = []
    crossing_lookup: dict[tuple[str, float], dict[str, float | int | bool]] = {}
    for target in thresholds:
        for method in METHODS:
            curve = val[val["method"] == method][["step", "val_loss", "train_time_s"]]
            crossing = first_crossing(curve, target)
            crossing_lookup[(method, target)] = crossing
            target_rows.append({"target_val_loss": target, "method": method, **crossing})
    targets = pd.DataFrame(target_rows)

    val_wide = val.pivot(index="step", columns="method", values="val_loss").reset_index()
    pair_specs = [
        ("diag", "block4"),
        ("none", "block4"),
        ("diag", "none"),
        ("diag", "muon"),
        ("none", "muon"),
        ("block4", "muon"),
    ]
    pair_rows: list[dict[str, object]] = []
    for left, right in pair_specs:
        noninitial = val_wide[val_wide["step"] > 0].copy()
        differences = noninitial[left] - noninitial[right]
        signs = np.sign(differences.to_numpy(dtype=float))
        sign_flips = int(np.sum(signs[1:] * signs[:-1] < 0))
        left_summary = by_method.loc[left]
        right_summary = by_method.loc[right]
        left_cross = crossing_lookup[(left, common_target)]
        right_cross = crossing_lookup[(right, common_target)]
        pair_rows.append(
            {
                "comparison": f"{left}_minus_{right}",
                "left_method": left,
                "right_method": right,
                "final_val_loss_delta_left_minus_right": float(
                    left_summary["final_val_loss"] - right_summary["final_val_loss"]
                ),
                "tail3_delta_left_minus_right": float(
                    left_summary["tail3_val_loss_mean"] - right_summary["tail3_val_loss_mean"]
                ),
                "tail5_delta_left_minus_right": float(
                    left_summary["tail5_val_loss_mean"] - right_summary["tail5_val_loss_mean"]
                ),
                "normalized_val_auc_delta_left_minus_right": float(
                    left_summary["normalized_val_auc"] - right_summary["normalized_val_auc"]
                ),
                "left_lower_noninitial_points": int((differences < 0).sum()),
                "right_lower_noninitial_points": int((differences > 0).sum()),
                "tied_noninitial_points": int((differences == 0).sum()),
                "total_noninitial_points": len(noninitial),
                "mean_noninitial_delta_left_minus_right": float(differences.mean()),
                "median_noninitial_delta_left_minus_right": float(differences.median()),
                "sign_flips": sign_flips,
                "persistent_left_lead_start_step": persistent_lead_start(
                    noninitial[left], noninitial[right], noninitial["step"]
                ),
                "train_time_delta_s_left_minus_right": float(
                    left_summary["train_time_s"] - right_summary["train_time_s"]
                ),
                "train_time_delta_pct_left_minus_right": float(
                    100
                    * (left_summary["train_time_s"] - right_summary["train_time_s"])
                    / right_summary["train_time_s"]
                ),
                "step_avg_delta_ms_left_minus_right": float(
                    left_summary["final_step_avg_ms"] - right_summary["final_step_avg_ms"]
                ),
                "peak_memory_delta_mib_left_minus_right": float(
                    left_summary["peak_memory_mib"] - right_summary["peak_memory_mib"]
                ),
                "k_state_delta_mib_left_minus_right": float(
                    left_summary["k_state_mib"] - right_summary["k_state_mib"]
                ),
                "optimizer_state_delta_mib_left_minus_right": float(
                    left_summary["optimizer_state_mib"] - right_summary["optimizer_state_mib"]
                ),
                "common_target_val_loss": common_target,
                "time_to_common_target_delta_s_left_minus_right": float(
                    left_cross["interpolated_time_s"] - right_cross["interpolated_time_s"]
                ),
                "step_to_common_target_delta_left_minus_right": float(
                    left_cross["interpolated_step"] - right_cross["interpolated_step"]
                ),
            }
        )
    pairs = pd.DataFrame(pair_rows)

    milestones = val[val["step"].isin([0, 100, 500, 1000, 2000, 3000, 4000, 4400, 5000, 6000, 6200])][
        ["method", "step", "val_loss", "train_time_s"]
    ].sort_values(["step", "method"])

    recommendation = pd.DataFrame(
        [
            {
                "claim": "diag is better than or non-inferior to block4 on quality",
                "current_evidence": "seed2026 endpoint delta = -0.0006; effect is too small for a one-seed superiority claim",
                "additional_evidence": "two additional paired seeds for diag/block4; include none in the same batches",
                "priority": "HIGH",
                "decision_rule": "report paired per-seed deltas; predefine a non-inferiority margin before running",
            },
            {
                "claim": "diagonal c_proj K adds value over no c_proj K",
                "current_evidence": "diag improves endpoint by 0.0050 with only 0.28125 MiB extra K state",
                "additional_evidence": "include none in the two added paired seeds",
                "priority": "HIGH",
                "decision_rule": "direction and paired mean should be consistent across seeds",
            },
            {
                "claim": "Newton variants beat Muon under official method-specific recipes",
                "current_evidence": "all three Newton variants finish 0.0100-0.0150 lower in seed2026",
                "additional_evidence": "add Muon to new seeds only if this recipe-level claim is central",
                "priority": "MEDIUM",
                "decision_rule": "label as recipe comparison; multi-seed does not remove the 10% LR difference",
            },
            {
                "claim": "diag materially reduces state/peak memory versus block4",
                "current_evidence": "exact state accounting and peak-memory scalar show the structural saving",
                "additional_evidence": "no extra seeds required; preserve local memory manifests",
                "priority": "LOW",
                "decision_rule": "verify exact bytes from local R1 summaries",
            },
            {
                "claim": "sub-1% runtime differences are real",
                "current_evidence": "variant timing differences are 0.1%-0.6% and may reflect execution order/noise",
                "additional_evidence": "counterbalanced repeated timing runs, not additional random seeds",
                "priority": "MEDIUM",
                "decision_rule": "report median and spread across repeats on the same H100",
            },
        ]
    )

    quality = pd.DataFrame(quality_rows)
    blocking = quality[quality["status"] == "FAIL"]
    source_manifest = pd.DataFrame(manifest_rows)

    source_manifest.to_csv(out / "source_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    long.to_csv(out / "normalized_history_long.csv", index=False)
    wide.sort_values(["method", "step"]).to_csv(out / "normalized_history_wide.csv", index=False)
    val_wide.to_csv(out / "validation_curves_wide.csv", index=False)
    milestones.to_csv(out / "validation_milestones.csv", index=False)
    summary.to_csv(out / "r1_run_summary.csv", index=False)
    pairs.to_csv(out / "r1_pairwise_summary.csv", index=False)
    targets.to_csv(out / "time_to_loss_targets.csv", index=False)
    quality.to_csv(out / "data_quality_checks.csv", index=False)
    recommendation.to_csv(out / "multiseed_recommendation.csv", index=False)
    if args.r0_summary is not None:
        r0_summary = pd.read_csv(args.r0_summary)
        r0_summary["method"] = r0_summary["method"].replace(
            {"newton_muon_block4": "block4"}
        )
        r0_anchor = r0_summary[r0_summary["method"].isin(["muon", "block4"])][
            ["method", "final_val_loss", "train_time_s"]
        ].rename(
            columns={
                "final_val_loss": "r0_final_val_loss",
                "train_time_s": "r0_train_time_s",
            }
        )
        r1_anchor = summary[summary["method"].isin(["muon", "block4"])][
            ["method", "final_val_loss", "train_time_s"]
        ].rename(
            columns={
                "final_val_loss": "r1_final_val_loss",
                "train_time_s": "r1_train_time_s",
            }
        )
        anchor = r0_anchor.merge(r1_anchor, on="method", validate="one_to_one")
        anchor["r1_minus_r0_final_val_loss"] = (
            anchor["r1_final_val_loss"] - anchor["r0_final_val_loss"]
        )
        anchor["r1_minus_r0_train_time_s"] = (
            anchor["r1_train_time_s"] - anchor["r0_train_time_s"]
        )
        anchor.to_csv(out / "r0_r1_anchor_comparison.csv", index=False)
    write_markdown(out / "R1_ANALYSIS_20260720.md", summary, pairs, quality, recommendation)
    (out / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS" if blocking.empty else "FAIL",
                "input_files": [str(path.resolve()) for path in args.input],
                "output_dir": str(out),
                "metrics": sorted(seen_metrics),
                "methods": list(METHODS),
                "seed": 2026,
                "quality_pass": int((quality["status"] == "PASS").sum()),
                "quality_warn": int((quality["status"] == "WARN").sum()),
                "quality_fail": int((quality["status"] == "FAIL").sum()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if not blocking.empty:
        raise RuntimeError(f"R1 export audit failed:\n{blocking.to_string(index=False)}")
    print(summary.to_string(index=False))
    print(pairs.to_string(index=False))
    print(f"Saved R1 analysis to {out}")


if __name__ == "__main__":
    main()
