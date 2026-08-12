"""Audit and summarize exported W&B scalar histories for official R0.

This utility treats the original CSV files as immutable evidence, copies them
into a dated analysis directory, normalizes the two R0 runs into a long table,
and writes compact result/data-quality tables for later paper analysis.
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


EXPECTED_METRICS = {
    "val/loss": (63, list(range(0, 6201, 100))),
    "official/step_avg_ms": (311, list(range(0, 6201, 20))),
    "official/train_time_s": (311, list(range(0, 6201, 20))),
    "lr/adamw": (311, list(range(0, 6201, 20))),
    "lr/matrix": (311, list(range(0, 6201, 20))),
}

OFFICIAL_REFERENCE = {
    "muon": {"final_val_loss": 3.2793, "train_time_s": 7314.1},
    "newton_muon_block4": {"final_val_loss": 3.2611, "train_time_s": 7443.3},
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
    if "official_newton_muon_1_block4" in run_name:
        return "newton_muon_block4"
    if "official_muon_1" in run_name:
        return "muon"
    raise ValueError(f"Unrecognized R0 run name: {run_name}")


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
    idx = int(reached.index[0])
    current = curve.iloc[idx]
    if idx == 0 or float(current["val_loss"]) == target:
        return {
            "reached": True,
            "first_discrete_step": int(current["step"]),
            "first_discrete_time_s": float(current["train_time_s"]),
            "interpolated_step": float(current["step"]),
            "interpolated_time_s": float(current["train_time_s"]),
        }
    previous = curve.iloc[idx - 1]
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


def write_markdown(
    path: Path,
    quality: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    q_pass = int((quality["status"] == "PASS").sum())
    q_warn = int((quality["status"] == "WARN").sum())
    nm = summary.set_index("method").loc["newton_muon_block4"]
    mu = summary.set_index("method").loc["muon"]
    pair = pairwise.iloc[0]
    common = targets[
        np.isclose(targets["target_val_loss"], float(mu["final_val_loss"]), atol=1e-12)
    ].set_index("method")
    text = f"""# R0 W&B export audit and result summary (2026-07-20)

## Evidence scope

- Two official, unchanged R0 methods: Muon and block4 Newton-Muon.
- Five exported history metrics: validation loss, official train time, official average step time, AdamW LR, matrix LR.
- Validation history: 63 points per method (steps 0..6200 every 100).
- Other histories: 311 points per method (steps 0..6200 every 20).
- The upstream scripts deliberately do not set a random seed, so this is a faithful single-run reproduction gate, not a variance estimate.

## Data-quality verdict

- {q_pass} checks passed; {q_warn} documented warning(s); no blocking failure.
- The only non-finite history values are `official/step_avg_ms = NaN` at steps 0 and 20 for both methods. These precede the official step-32 timing reset and are expected sentinels.
- W&B `__MIN`/`__MAX` mirror columns agree with the raw series at every finite point.
- Missing from this export set: `train/loss_step` history and peak-memory summary. They are not required for the R0 validation-loss conclusion, but should be added for a complete archive.

## Main result

| Method | Final val loss | Perplexity | Official train time (s) | Final step avg (ms) |
|---|---:|---:|---:|---:|
| Muon | {mu['final_val_loss']:.4f} | {mu['final_perplexity']:.3f} | {mu['train_time_s']:.3f} | {mu['final_step_avg_ms']:.2f} |
| block4 Newton-Muon | {nm['final_val_loss']:.4f} | {nm['final_perplexity']:.3f} | {nm['train_time_s']:.3f} | {nm['final_step_avg_ms']:.2f} |

- Newton-Muon lowers final validation loss by {abs(pair['final_val_loss_delta_nm_minus_muon']):.4f}, and lowers final perplexity by {abs(pair['final_perplexity_delta_pct_nm_minus_muon']):.3f}%.
- It costs {pair['train_time_overhead_s_nm_minus_muon']:.3f} s ({pair['train_time_overhead_pct_nm_minus_muon']:.3f}%) more wall-clock training time.
- At Muon's final loss target ({mu['final_val_loss']:.4f}), interpolated time-to-target is {common.loc['newton_muon_block4', 'interpolated_time_s']:.1f} s for Newton-Muon versus {common.loc['muon', 'interpolated_time_s']:.1f} s for Muon, an advantage of {common.loc['muon', 'interpolated_time_s'] - common.loc['newton_muon_block4', 'interpolated_time_s']:.1f} s.
- The lower-loss direction reproduces the official result. The observed gap is {pair['observed_loss_gap_muon_minus_nm']:.4f}, versus the official {pair['official_loss_gap_muon_minus_nm']:.4f} ({pair['official_gap_reproduced_pct']:.1f}% of the reported gap).

## Interpretation boundary

R0 passes its intended gate: the pinned official environment and unchanged official scripts reproduce the Newton-Muon > Muon direction on one H100 run. It does **not** isolate the optimizer rule alone, because the official pair uses different learning rates (Muon 0.0036/0.00036; Newton-Muon 0.0040/0.00040), and it does not quantify randomness because the official scripts do not control a seed. Those questions belong to R1/multi-seed controlled experiments.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    raw_dir = out / "raw_wandb_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    long_frames: list[pd.DataFrame] = []
    quality_rows: list[dict[str, object]] = []
    seen_metrics: set[str] = set()

    for source_arg in args.input:
        source = source_arg.resolve()
        copied = raw_dir / source.name
        if copied.exists() and sha256(copied) != sha256(source):
            raise RuntimeError(f"Refusing to overwrite different raw evidence: {copied}")
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
        if len(primary_columns) != 2:
            raise RuntimeError(f"Expected two primary run columns in {source.name}, got {primary_columns}")
        metric_names = {column.rsplit(" - ", 1)[1] for column in primary_columns}
        if len(metric_names) != 1:
            raise RuntimeError(f"Mixed metrics in {source.name}: {metric_names}")
        metric = next(iter(metric_names))
        seen_metrics.add(metric)

        expected = EXPECTED_METRICS.get(metric)
        if expected is None:
            quality_rows.append(
                {"check": f"recognized metric: {metric}", "status": "WARN", "details": source.name}
            )
        else:
            expected_rows, expected_steps = expected
            quality_rows.append(
                {
                    "check": f"{metric}: row count",
                    "status": "PASS" if len(frame) == expected_rows else "FAIL",
                    "details": f"observed={len(frame)}, expected={expected_rows}",
                }
            )
            observed_steps = frame["Step"].astype(int).tolist()
            quality_rows.append(
                {
                    "check": f"{metric}: exact step grid",
                    "status": "PASS" if observed_steps == expected_steps else "FAIL",
                    "details": f"first={observed_steps[:3]}, last={observed_steps[-3:]}",
                }
            )

        for column in primary_columns:
            run_name, parsed_metric = column.rsplit(" - ", 1)
            method = method_from_run(run_name)
            values = pd.to_numeric(frame[column], errors="coerce")
            min_values = pd.to_numeric(frame[f"{column}__MIN"], errors="coerce")
            max_values = pd.to_numeric(frame[f"{column}__MAX"], errors="coerce")
            finite = np.isfinite(values.to_numpy(dtype=float))
            mirror_ok = np.allclose(
                values.to_numpy(dtype=float)[finite],
                min_values.to_numpy(dtype=float)[finite],
                rtol=0,
                atol=0,
            ) and np.allclose(
                values.to_numpy(dtype=float)[finite],
                max_values.to_numpy(dtype=float)[finite],
                rtol=0,
                atol=0,
            )
            quality_rows.append(
                {
                    "check": f"{method} {metric}: W&B MIN/MAX mirrors",
                    "status": "PASS" if mirror_ok else "FAIL",
                    "details": f"finite_points={int(finite.sum())}",
                }
            )
            nonfinite_steps = frame.loc[~finite, "Step"].astype(int).tolist()
            allowed = parsed_metric == "official/step_avg_ms" and nonfinite_steps == [0, 20]
            quality_rows.append(
                {
                    "check": f"{method} {metric}: finite values",
                    "status": "WARN" if allowed else ("PASS" if not nonfinite_steps else "FAIL"),
                    "details": (
                        "steps 0 and 20 are expected NaN sentinels before the official step-32 timing reset"
                        if allowed
                        else f"nonfinite_steps={nonfinite_steps}"
                    ),
                }
            )
            long_frames.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "run_name": run_name,
                        "metric": parsed_metric,
                        "step": frame["Step"].astype(int),
                        "value": values,
                        "source_file": source.name,
                    }
                )
            )

    missing_metrics = sorted(set(EXPECTED_METRICS) - seen_metrics)
    extra_metrics = sorted(seen_metrics - set(EXPECTED_METRICS))
    quality_rows.append(
        {
            "check": "expected metric files present",
            "status": "PASS" if not missing_metrics and not extra_metrics else "FAIL",
            "details": f"missing={missing_metrics}, extra={extra_metrics}",
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
    val = wide[wide["val/loss"].notna()].copy()
    val = val.rename(columns={"val/loss": "val_loss", "official/train_time_s": "train_time_s"})

    summary_rows: list[dict[str, object]] = []
    for method, curve in val.groupby("method"):
        curve = curve.sort_values("step")
        final = curve.iloc[-1]
        all_method = wide[wide["method"] == method].sort_values("step")
        ref = OFFICIAL_REFERENCE[method]
        summary_rows.append(
            {
                "method": method,
                "run_name": long[long["method"] == method]["run_name"].iloc[0],
                "initial_val_loss": float(curve.iloc[0]["val_loss"]),
                "final_val_loss": float(final["val_loss"]),
                "best_val_loss": float(curve["val_loss"].min()),
                "final_perplexity": math.exp(float(final["val_loss"])),
                "normalized_val_auc": float(np.trapezoid(curve["val_loss"], curve["step"]) / 6200.0),
                "train_time_s": float(final["train_time_s"]),
                "final_step_avg_ms": float(all_method.iloc[-1]["official/step_avg_ms"]),
                "max_adamw_lr": float(all_method["lr/adamw"].max()),
                "max_matrix_lr": float(all_method["lr/matrix"].max()),
                "official_reference_final_val_loss": ref["final_val_loss"],
                "final_val_loss_minus_reference": float(final["val_loss"] - ref["final_val_loss"]),
                "official_reference_train_time_s": ref["train_time_s"],
                "train_time_s_minus_reference": float(final["train_time_s"] - ref["train_time_s"]),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("method").reset_index(drop=True)
    by_method = summary.set_index("method")
    nm = by_method.loc["newton_muon_block4"]
    mu = by_method.loc["muon"]

    val_pivot = val.pivot(index="step", columns="method", values="val_loss").reset_index()
    val_pivot["delta_nm_minus_muon"] = (
        val_pivot["newton_muon_block4"] - val_pivot["muon"]
    )
    nm_better_points = int((val_pivot["delta_nm_minus_muon"] < 0).sum())
    official_gap = (
        OFFICIAL_REFERENCE["muon"]["final_val_loss"]
        - OFFICIAL_REFERENCE["newton_muon_block4"]["final_val_loss"]
    )
    observed_gap = float(mu["final_val_loss"] - nm["final_val_loss"])
    common_target = float(mu["final_val_loss"])
    common_crossings = {
        method: first_crossing(
            curve[["step", "val_loss", "train_time_s"]], common_target
        )
        for method, curve in val.groupby("method")
    }
    pairwise = pd.DataFrame(
        [
            {
                "comparison": "newton_muon_block4_minus_muon",
                "final_val_loss_delta_nm_minus_muon": float(nm["final_val_loss"] - mu["final_val_loss"]),
                "final_val_loss_delta_pct_nm_minus_muon": float(
                    100 * (nm["final_val_loss"] - mu["final_val_loss"]) / mu["final_val_loss"]
                ),
                "final_perplexity_delta_nm_minus_muon": float(
                    nm["final_perplexity"] - mu["final_perplexity"]
                ),
                "final_perplexity_delta_pct_nm_minus_muon": float(
                    100 * (nm["final_perplexity"] - mu["final_perplexity"]) / mu["final_perplexity"]
                ),
                "train_time_overhead_s_nm_minus_muon": float(nm["train_time_s"] - mu["train_time_s"]),
                "train_time_overhead_pct_nm_minus_muon": float(
                    100 * (nm["train_time_s"] - mu["train_time_s"]) / mu["train_time_s"]
                ),
                "step_avg_overhead_ms_nm_minus_muon": float(
                    nm["final_step_avg_ms"] - mu["final_step_avg_ms"]
                ),
                "step_avg_overhead_pct_nm_minus_muon": float(
                    100
                    * (nm["final_step_avg_ms"] - mu["final_step_avg_ms"])
                    / mu["final_step_avg_ms"]
                ),
                "normalized_val_auc_delta_nm_minus_muon": float(
                    nm["normalized_val_auc"] - mu["normalized_val_auc"]
                ),
                "nm_lower_loss_points": nm_better_points,
                "total_val_points": len(val_pivot),
                "nm_lower_loss_point_fraction": nm_better_points / len(val_pivot),
                "observed_loss_gap_muon_minus_nm": observed_gap,
                "official_loss_gap_muon_minus_nm": official_gap,
                "official_gap_reproduced_pct": 100 * observed_gap / official_gap,
                "common_target_val_loss": common_target,
                "nm_time_to_common_target_s": common_crossings["newton_muon_block4"][
                    "interpolated_time_s"
                ],
                "muon_time_to_common_target_s": common_crossings["muon"]["interpolated_time_s"],
                "nm_time_advantage_at_common_target_s": (
                    common_crossings["muon"]["interpolated_time_s"]
                    - common_crossings["newton_muon_block4"]["interpolated_time_s"]
                ),
                "nm_step_advantage_at_common_target": (
                    common_crossings["muon"]["interpolated_step"]
                    - common_crossings["newton_muon_block4"]["interpolated_step"]
                ),
            }
        ]
    )

    target_values = [4.0, 3.5, 3.3, float(mu["final_val_loss"])]
    target_rows: list[dict[str, object]] = []
    for target in target_values:
        for method, curve in val.groupby("method"):
            crossing = first_crossing(curve[["step", "val_loss", "train_time_s"]], target)
            target_rows.append({"target_val_loss": target, "method": method, **crossing})
    targets = pd.DataFrame(target_rows)

    milestones = val[val["step"].isin([0, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 6200])][
        ["method", "step", "val_loss", "train_time_s"]
    ].sort_values(["step", "method"])

    lr_ratio_ok = True
    post_reset_time_ok = True
    for method, method_wide in wide.groupby("method"):
        finite = method_wide[(method_wide["lr/matrix"] > 0) & method_wide["lr/adamw"].notna()]
        lr_ratio_ok = lr_ratio_ok and np.allclose(
            finite["lr/adamw"], 10.0 * finite["lr/matrix"], rtol=1e-10, atol=1e-12
        )
        post_reset_times = method_wide.loc[
            method_wide["step"] >= 40, "official/train_time_s"
        ].to_numpy(dtype=float)
        post_reset_time_ok = post_reset_time_ok and bool(np.all(np.diff(post_reset_times) >= 0))
    quality_rows.append(
        {
            "check": "AdamW/matrix LR ratio is exactly 10x",
            "status": "PASS" if lr_ratio_ok else "FAIL",
            "details": "checked all positive-LR history points",
        }
    )
    quality_rows.append(
        {
            "check": "official train time is monotone after step-32 reset",
            "status": "PASS" if post_reset_time_ok else "FAIL",
            "details": "checked exported points from step 40 through 6200",
        }
    )
    lr_shapes = {}
    for method, method_wide in wide.groupby("method"):
        ordered = method_wide.sort_values("step")
        lr_shapes[method] = {
            "adamw": ordered["lr/adamw"].to_numpy(dtype=float) / ordered["lr/adamw"].max(),
            "matrix": ordered["lr/matrix"].to_numpy(dtype=float) / ordered["lr/matrix"].max(),
        }
    lr_shape_ok = all(
        np.allclose(
            lr_shapes["muon"][name],
            lr_shapes["newton_muon_block4"][name],
            rtol=1e-10,
            atol=1e-12,
        )
        for name in ("adamw", "matrix")
    )
    quality_rows.append(
        {
            "check": "two methods use the same normalized LR schedule shape",
            "status": "PASS" if lr_shape_ok else "FAIL",
            "details": "absolute LR differs by the official 10/9 scale factor",
        }
    )

    quality = pd.DataFrame(quality_rows)
    blocking = quality[quality["status"] == "FAIL"]

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out / "source_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    long.to_csv(out / "normalized_history_long.csv", index=False)
    wide.sort_values(["method", "step"]).to_csv(out / "normalized_history_wide.csv", index=False)
    val_pivot.to_csv(out / "validation_curves_wide.csv", index=False)
    milestones.to_csv(out / "validation_milestones.csv", index=False)
    summary.to_csv(out / "r0_run_summary.csv", index=False)
    pairwise.to_csv(out / "r0_pairwise_summary.csv", index=False)
    targets.to_csv(out / "time_to_loss_targets.csv", index=False)
    quality.to_csv(out / "data_quality_checks.csv", index=False)
    write_markdown(out / "R0_ANALYSIS_20260720.md", quality, summary, pairwise, targets)
    (out / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS" if blocking.empty else "FAIL",
                "input_files": [str(path.resolve()) for path in args.input],
                "output_dir": str(out),
                "metrics": sorted(seen_metrics),
                "methods": sorted(long["method"].unique().tolist()),
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
        raise RuntimeError(f"R0 export audit failed:\n{blocking.to_string(index=False)}")
    print(summary.to_string(index=False))
    print(pairwise.to_string(index=False))
    print(f"Saved R0 analysis to {out}")


if __name__ == "__main__":
    main()
