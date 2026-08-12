"""Analyze W&B chart exports for a reference-LR sanity grid.

The W&B UI exports one CSV per metric.  Each CSV contains one base column per
run plus optional ``__MIN``/``__MAX`` display columns.  This script reconstructs
the run table, validates the grid, preserves the raw exports, and computes loss,
same-loss, storage, memory, and elapsed-time summaries without depending on the
W&B API.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


RUN_RE = re.compile(
    r"^mainconf_reference_lr_sanity_"
    r"(?P<suite>owt(?:12|24)l_3k)_"
    r"(?P<label>.+)_lr(?P<lr>[0-9]+p[0-9]+)_seed(?P<seed>[0-9]+)$"
)

METHOD_BY_LABEL = {
    "muon_blog": "muon",
    "paper_block4": "block4",
    "no_cproj_k": "none",
    "diag_cproj_k": "diag",
    "dense_full_cproj_k_control": "full",
}

EXPECTED_METRICS = {
    "val/loss",
    "train/loss_step",
    "time_elapsed",
    "cuda/memory_allocated_mib",
    "cuda/full_run_max_memory_allocated_mib",
    "lr/matrix",
    "lr/adamw",
    "matrix/non_cproj_k_state_bytes",
    "matrix/k_state_released_fraction",
    "matrix/k_state_bytes",
    "matrix/cproj_k_state_bytes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="W&B per-metric CSV exports belonging to one experiment grid.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-suite",
        choices=("owt12l_3k", "owt24l_3k"),
        default=None,
    )
    parser.add_argument("--expected-seed", type=int, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_from_column(column: str) -> str:
    if " - " not in column:
        raise ValueError(f"Cannot extract metric from column: {column!r}")
    return column.rsplit(" - ", 1)[1]


def run_from_column(column: str) -> str:
    if " - " not in column:
        raise ValueError(f"Cannot extract run from column: {column!r}")
    return column.rsplit(" - ", 1)[0]


def run_metadata(run_name: str) -> dict[str, object]:
    match = RUN_RE.match(run_name)
    if match is None:
        raise ValueError(f"Unexpected run name: {run_name}")
    label = match.group("label")
    if label not in METHOD_BY_LABEL:
        raise ValueError(f"Unexpected method label {label!r} in {run_name}")
    suite = match.group("suite")
    return {
        "run_name": run_name,
        "suite": suite,
        "method": METHOD_BY_LABEL[label],
        "seed": int(match.group("seed")),
        "matrix_lr": float(match.group("lr").replace("p", ".")),
        "n_layer": 12 if suite == "owt12l_3k" else 24,
        "n_embd": 768 if suite == "owt12l_3k" else 1024,
        "max_iters": 5000 if suite == "owt12l_3k" else 3000,
    }


def safe_metric_filename(metric: str) -> str:
    return metric.replace("/", "_").replace(" ", "_") + ".csv"


def last_finite(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.iloc[-1]) if len(values) else math.nan


def value_at_or_before(frame: pd.DataFrame, step: float) -> float:
    subset = frame.loc[frame["step"] <= step]
    if subset.empty:
        return math.nan
    return float(subset.iloc[-1]["value"])


def interpolate_time(time_frame: pd.DataFrame, step: float) -> float:
    clean = time_frame.dropna(subset=["step", "value"]).sort_values("step")
    if clean.empty:
        return math.nan
    return float(np.interp(step, clean["step"], clean["value"]))


def crossing_at_threshold(
    val_frame: pd.DataFrame,
    time_frame: pd.DataFrame,
    threshold: float,
) -> dict[str, float]:
    ordered = val_frame.dropna(subset=["value"]).sort_values("step").reset_index(drop=True)
    hits = np.flatnonzero(ordered["value"].to_numpy() <= threshold)
    if len(hits) == 0:
        return {
            "first_observed_step": math.nan,
            "first_observed_time_s": math.nan,
            "interpolated_step": math.nan,
            "interpolated_time_s": math.nan,
        }

    index = int(hits[0])
    observed_step = float(ordered.loc[index, "step"])
    observed_time = interpolate_time(time_frame, observed_step)
    interpolated_step = observed_step
    if index > 0:
        x0 = float(ordered.loc[index - 1, "step"])
        x1 = observed_step
        y0 = float(ordered.loc[index - 1, "value"])
        y1 = float(ordered.loc[index, "value"])
        if y0 > threshold and y1 < y0:
            fraction = (y0 - threshold) / (y0 - y1)
            interpolated_step = x0 + fraction * (x1 - x0)

    return {
        "first_observed_step": observed_step,
        "first_observed_time_s": observed_time,
        "interpolated_step": interpolated_step,
        "interpolated_time_s": interpolate_time(time_frame, interpolated_step),
    }


def main() -> None:
    args = parse_args()
    inputs = [path.resolve() for path in args.inputs]
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw_wandb_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)

    metric_parts: dict[str, list[pd.DataFrame]] = {}
    inventory_rows: list[dict[str, object]] = []
    duplicate_checks: list[dict[str, object]] = []

    for source in inputs:
        frame = pd.read_csv(source)
        if "Step" not in frame.columns:
            raise ValueError(f"Missing Step column in {source}")
        base_columns = [
            column
            for column in frame.columns
            if column != "Step" and not column.endswith("__MIN") and not column.endswith("__MAX")
        ]
        if not base_columns:
            raise ValueError(f"No base run columns in {source}")
        metrics = {metric_from_column(column) for column in base_columns}
        if len(metrics) != 1:
            raise ValueError(f"Multiple metrics in {source}: {sorted(metrics)}")
        metric = next(iter(metrics))
        part_number = len(metric_parts.get(metric, [])) + 1
        base_destination = Path(safe_metric_filename(metric))
        destination = raw_dir / (
            base_destination.name
            if part_number == 1
            else f"{base_destination.stem}_part{part_number}{base_destination.suffix}"
        )
        # Re-analysis may use the already-preserved raw directory as input.
        # Avoid copying a file onto itself (Windows rejects that operation).
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

        long_parts: list[pd.DataFrame] = []
        for column in base_columns:
            run_name = run_from_column(column)
            metadata = run_metadata(run_name)
            part = pd.DataFrame(
                {
                    "step": pd.to_numeric(frame["Step"], errors="coerce"),
                    "value": pd.to_numeric(frame[column], errors="coerce"),
                }
            )
            for key, value in metadata.items():
                part[key] = value
            long_parts.append(part)

            for suffix in ("__MIN", "__MAX"):
                duplicate_column = column + suffix
                if duplicate_column in frame.columns:
                    left = pd.to_numeric(frame[column], errors="coerce").to_numpy()
                    right = pd.to_numeric(frame[duplicate_column], errors="coerce").to_numpy()
                    equal = bool(np.allclose(left, right, rtol=0, atol=0, equal_nan=True))
                    duplicate_checks.append(
                        {
                            "metric": metric,
                            "run_name": run_name,
                            "display_column": suffix[2:].lower(),
                            "identical_to_base": equal,
                        }
                    )

        long_frame = pd.concat(long_parts, ignore_index=True)
        long_frame["metric"] = metric
        metric_parts.setdefault(metric, []).append(long_frame)
        inventory_rows.append(
            {
                "metric": metric,
                "source_path": str(source),
                "copied_path": str(destination),
                "sha256": sha256(source),
                "csv_rows": int(len(frame)),
                "base_run_columns": int(len(base_columns)),
                "step_min": float(pd.to_numeric(frame["Step"], errors="coerce").min()),
                "step_max": float(pd.to_numeric(frame["Step"], errors="coerce").max()),
            }
        )

    metric_frames: dict[str, pd.DataFrame] = {}
    for metric, parts in metric_parts.items():
        combined_metric = pd.concat(parts, ignore_index=True)
        duplicate_keys = combined_metric.duplicated(["run_name", "step"], keep=False)
        if duplicate_keys.any():
            duplicates = combined_metric.loc[duplicate_keys, ["run_name", "step"]].drop_duplicates()
            raise ValueError(
                f"Duplicate run/step keys across exports for metric {metric}: "
                f"{duplicates.head(10).to_dict(orient='records')}"
            )
        metric_frames[metric] = combined_metric

    missing_metrics = EXPECTED_METRICS.difference(metric_frames)
    unexpected_metrics = set(metric_frames).difference(EXPECTED_METRICS)
    if missing_metrics:
        raise ValueError(f"Missing expected metrics: {sorted(missing_metrics)}")
    if unexpected_metrics:
        raise ValueError(f"Unexpected metrics: {sorted(unexpected_metrics)}")

    val_long = metric_frames["val/loss"].copy()
    train_long = metric_frames["train/loss_step"].copy()
    time_long = metric_frames["time_elapsed"].copy()
    run_meta = (
        val_long[["run_name", "suite", "method", "seed", "matrix_lr", "n_layer", "n_embd", "max_iters"]]
        .drop_duplicates()
        .sort_values(["suite", "matrix_lr", "method"])
        .reset_index(drop=True)
    )

    checks: list[dict[str, object]] = []

    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "details": details})

    expected_methods = set(METHOD_BY_LABEL.values())
    expected_lrs = {0.005, 0.01, 0.02}
    actual_methods = set(run_meta["method"])
    actual_lrs = set(run_meta["matrix_lr"])
    add_check(
        "input_file_count",
        len(inputs) >= 11 and len(inputs) % 11 == 0,
        f"observed={len(inputs)}, expected one or more complete 11-metric export parts",
    )
    add_check("metric_coverage", not missing_metrics and not unexpected_metrics, "all 11 expected metrics present")
    add_check("run_count", len(run_meta) == 15, f"observed={len(run_meta)}, expected=15")
    add_check("method_coverage", actual_methods == expected_methods, f"observed={sorted(actual_methods)}")
    add_check("lr_coverage", actual_lrs == expected_lrs, f"observed={sorted(actual_lrs)}")
    add_check(
        "complete_method_lr_grid",
        len(run_meta[["method", "matrix_lr"]].drop_duplicates()) == 15,
        "expected one run for each of 5 methods x 3 LRs",
    )
    if args.expected_suite is not None:
        suites = set(run_meta["suite"])
        add_check("expected_suite", suites == {args.expected_suite}, f"observed={sorted(suites)}")
    if args.expected_seed is not None:
        seeds = set(run_meta["seed"])
        add_check("expected_seed", seeds == {args.expected_seed}, f"observed={sorted(seeds)}")

    core_metrics = ("val/loss", "train/loss_step", "time_elapsed", "lr/matrix", "lr/adamw")
    expected_runs = set(run_meta["run_name"])
    for metric in core_metrics:
        observed = set(metric_frames[metric]["run_name"])
        add_check(
            f"run_coverage::{metric}",
            observed == expected_runs,
            f"observed={len(observed)}, expected={len(expected_runs)}",
        )

    finite_val = np.isfinite(val_long["value"].dropna()).all()
    finite_train = np.isfinite(train_long["value"].dropna()).all()
    add_check("finite_losses", bool(finite_val and finite_train), "all recorded train/val losses are finite")

    initial_val = val_long.loc[val_long["step"] == val_long["step"].min(), "value"].dropna()
    add_check(
        "identical_initial_val_loss",
        bool(initial_val.nunique(dropna=True) == 1),
        f"unique_initial_values={initial_val.nunique(dropna=True)}",
    )

    matrix_lr_ok = True
    lr_details: list[str] = []
    for run in run_meta.itertuples(index=False):
        observed = metric_frames["lr/matrix"].loc[
            metric_frames["lr/matrix"]["run_name"] == run.run_name, "value"
        ].dropna()
        ok = len(observed) > 0 and bool(np.allclose(observed, run.matrix_lr, rtol=0, atol=1e-12))
        matrix_lr_ok = matrix_lr_ok and ok
        if not ok:
            lr_details.append(run.run_name)
    add_check(
        "matrix_lr_matches_run_name",
        matrix_lr_ok,
        "all exact" if matrix_lr_ok else "mismatch: " + "; ".join(lr_details),
    )

    duplicate_ok = all(row["identical_to_base"] for row in duplicate_checks)
    add_check(
        "wandb_display_min_max_duplicates",
        duplicate_ok,
        f"checked={len(duplicate_checks)} display columns; all identical={duplicate_ok}",
    )

    val_counts = val_long.dropna(subset=["value"]).groupby("run_name").size()
    val_max_steps = val_long.dropna(subset=["value"]).groupby("run_name")["step"].max()
    add_check(
        "uniform_val_coverage",
        val_counts.nunique() == 1 and val_max_steps.nunique() == 1,
        f"points={sorted(val_counts.unique())}, max_steps={sorted(val_max_steps.unique())}",
    )

    k_metrics = {
        name: metric_frames[name]
        for name in (
            "matrix/k_state_bytes",
            "matrix/cproj_k_state_bytes",
            "matrix/non_cproj_k_state_bytes",
        )
    }
    k_identity_errors: list[float] = []
    for run_name in expected_runs:
        rows = {}
        for metric, frame in k_metrics.items():
            subset = frame.loc[frame["run_name"] == run_name, ["step", "value"]].dropna()
            if not subset.empty:
                rows[metric] = subset.set_index("step")["value"]
        if len(rows) == 3:
            joined = pd.concat(rows, axis=1).dropna()
            error = (
                joined["matrix/k_state_bytes"]
                - joined["matrix/cproj_k_state_bytes"]
                - joined["matrix/non_cproj_k_state_bytes"]
            ).abs()
            if len(error):
                k_identity_errors.append(float(error.max()))
    max_k_error = max(k_identity_errors, default=0.0)
    add_check("k_state_additivity", max_k_error == 0.0, f"max_absolute_error_bytes={max_k_error:g}")

    summaries: list[dict[str, object]] = []
    for run in run_meta.itertuples(index=False):
        run_val = val_long.loc[val_long["run_name"] == run.run_name, ["step", "value"]].dropna().sort_values("step")
        run_train = train_long.loc[train_long["run_name"] == run.run_name, ["step", "value"]].dropna().sort_values("step")
        run_time = time_long.loc[time_long["run_name"] == run.run_name, ["step", "value"]].dropna().sort_values("step")

        final_val_step = float(run_val["step"].max())
        final_val_loss = float(run_val.iloc[-1]["value"])
        best_index = run_val["value"].idxmin()
        best_val_loss = float(run_val.loc[best_index, "value"])
        best_val_step = float(run_val.loc[best_index, "step"])
        late_start = final_val_step * 0.8
        late_val = run_val.loc[run_val["step"] >= late_start, "value"]
        val_span = float(run_val["step"].max() - run_val["step"].min())
        val_auc = float(np.trapezoid(run_val["value"], run_val["step"]) / val_span)

        final_train_step = float(run_train["step"].max())
        final_train_loss = float(run_train.iloc[-1]["value"])
        late_train = run_train.loc[run_train["step"] >= final_train_step * 0.8, "value"]

        summary: dict[str, object] = {
            "suite": run.suite,
            "method": run.method,
            "run_name": run.run_name,
            "seed": int(run.seed),
            "n_layer": int(run.n_layer),
            "n_embd": int(run.n_embd),
            "max_iters": int(run.max_iters),
            "matrix_lr": float(run.matrix_lr),
            "final_val_step": final_val_step,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "best_val_step": best_val_step,
            "late_val_mean_last20pct": float(late_val.mean()),
            "normalized_val_auc": val_auc,
            "val_rebound_final_minus_best": final_val_loss - best_val_loss,
            "val_change_mid_to_final": final_val_loss - value_at_or_before(run_val, final_val_step / 2),
            "last_train_logged_step": final_train_step,
            "final_train_loss": final_train_loss,
            "best_train_loss": float(run_train["value"].min()),
            "late_train_mean_last20pct": float(late_train.mean()),
            "last_time_logged_step": float(run_time["step"].max()),
            "time_elapsed_last_s": float(run_time.iloc[-1]["value"]),
        }

        metric_to_field = {
            "matrix/k_state_bytes": "total_k_state_mib",
            "matrix/cproj_k_state_bytes": "cproj_k_state_mib",
            "matrix/non_cproj_k_state_bytes": "non_cproj_k_state_mib",
            "matrix/k_state_released_fraction": "k_state_released_fraction",
            "cuda/memory_allocated_mib": "cuda_allocated_last_mib",
            "cuda/full_run_max_memory_allocated_mib": "cuda_full_run_peak_mib",
            "lr/matrix": "matrix_lr_logged_last",
            "lr/adamw": "adamw_lr_logged_last",
        }
        for metric, field in metric_to_field.items():
            metric_run = metric_frames[metric].loc[
                metric_frames[metric]["run_name"] == run.run_name, ["step", "value"]
            ].dropna().sort_values("step")
            if metric_run.empty:
                summary[field] = math.nan
            elif metric.endswith("_bytes"):
                summary[field] = last_finite(metric_run["value"]) / (1024.0**2)
            elif metric == "cuda/full_run_max_memory_allocated_mib":
                summary[field] = float(metric_run["value"].max())
            else:
                summary[field] = last_finite(metric_run["value"])
        summaries.append(summary)

    summary_frame = pd.DataFrame(summaries).sort_values(["matrix_lr", "final_val_loss", "method"])

    equal_loss_rows: list[dict[str, object]] = []
    for matrix_lr, lr_summary in summary_frame.groupby("matrix_lr"):
        exact_common_threshold = float(lr_summary["best_val_loss"].max())
        rounded_common_threshold = math.ceil((exact_common_threshold - 1e-12) / 0.05) * 0.05
        for row in lr_summary.itertuples(index=False):
            run_val = val_long.loc[val_long["run_name"] == row.run_name, ["step", "value"]]
            run_time = time_long.loc[time_long["run_name"] == row.run_name, ["step", "value"]]
            crossing = crossing_at_threshold(run_val, run_time, rounded_common_threshold)
            equal_loss_rows.append(
                {
                    "matrix_lr": matrix_lr,
                    "common_threshold_rule": "ceil(max_per_method_best_val_loss / 0.05) * 0.05",
                    "exact_deepest_common_threshold": exact_common_threshold,
                    "reported_common_threshold": rounded_common_threshold,
                    "method": row.method,
                    "run_name": row.run_name,
                    **crossing,
                }
            )
    equal_loss_frame = pd.DataFrame(equal_loss_rows).sort_values(
        ["matrix_lr", "interpolated_time_s", "method"], na_position="last"
    )

    comparison_source = summary_frame.merge(
        equal_loss_frame[
            [
                "matrix_lr",
                "method",
                "reported_common_threshold",
                "interpolated_step",
                "interpolated_time_s",
            ]
        ],
        on=["matrix_lr", "method"],
        how="left",
        validate="one_to_one",
    )
    comparisons: list[dict[str, object]] = []
    for matrix_lr, lr_rows in comparison_source.groupby("matrix_lr"):
        baseline = lr_rows.loc[lr_rows["method"] == "none"].iloc[0]
        for row in lr_rows.itertuples(index=False):
            comparisons.append(
                {
                    "matrix_lr": matrix_lr,
                    "method": row.method,
                    "reported_common_threshold": row.reported_common_threshold,
                    "final_val_loss": row.final_val_loss,
                    "final_val_loss_delta_vs_none": row.final_val_loss - baseline["final_val_loss"],
                    "final_val_loss_relative_delta_vs_none_pct": 100.0
                    * (row.final_val_loss - baseline["final_val_loss"])
                    / baseline["final_val_loss"],
                    "late_val_mean_delta_vs_none": row.late_val_mean_last20pct
                    - baseline["late_val_mean_last20pct"],
                    "normalized_val_auc_delta_vs_none": row.normalized_val_auc
                    - baseline["normalized_val_auc"],
                    "interpolated_step": row.interpolated_step,
                    "same_loss_step_delta_vs_none": row.interpolated_step
                    - baseline["interpolated_step"],
                    "interpolated_time_s": row.interpolated_time_s,
                    "same_loss_time_delta_vs_none_s": row.interpolated_time_s
                    - baseline["interpolated_time_s"],
                    "same_loss_time_delta_vs_none_pct": 100.0
                    * (row.interpolated_time_s - baseline["interpolated_time_s"])
                    / baseline["interpolated_time_s"],
                    "time_elapsed_last_s": row.time_elapsed_last_s,
                    "last_time_delta_vs_none_s": row.time_elapsed_last_s
                    - baseline["time_elapsed_last_s"],
                    "last_time_delta_vs_none_pct": 100.0
                    * (row.time_elapsed_last_s - baseline["time_elapsed_last_s"])
                    / baseline["time_elapsed_last_s"],
                    "total_k_state_mib": row.total_k_state_mib,
                    "cproj_k_state_mib": row.cproj_k_state_mib,
                }
            )
    comparison_frame = pd.DataFrame(comparisons).sort_values(["matrix_lr", "final_val_loss"])

    # Preserve the two views needed for a fair optimizer comparison: the
    # common-LR result and the best observed LR for each method.  B is a short
    # sanity grid, so these files are descriptive rather than a claim that the
    # selected LR is globally optimal.
    loss_by_lr = summary_frame.pivot(
        index="method", columns="matrix_lr", values="final_val_loss"
    )
    loss_by_lr.columns = [f"final_val_loss_lr{float(lr):g}" for lr in loss_by_lr.columns]
    loss_by_lr = loss_by_lr.reset_index()
    best_indices = summary_frame.groupby("method")["final_val_loss"].idxmin()
    best_lr_frame = summary_frame.loc[
        best_indices,
        [
            "method",
            "matrix_lr",
            "final_val_loss",
            "best_val_loss",
            "late_val_mean_last20pct",
            "normalized_val_auc",
            "time_elapsed_last_s",
            "total_k_state_mib",
            "cuda_allocated_last_mib",
            "cuda_full_run_peak_mib",
        ],
    ].rename(
        columns={
            "matrix_lr": "best_observed_matrix_lr",
            "final_val_loss": "best_observed_lr_final_val_loss",
        }
    )
    best_lr_frame = best_lr_frame.merge(
        loss_by_lr, on="method", how="left", validate="one_to_one"
    ).sort_values(["best_observed_lr_final_val_loss", "method"])

    muon_block4_rows: list[dict[str, object]] = []
    for matrix_lr, lr_rows in comparison_source.groupby("matrix_lr"):
        muon = lr_rows.loc[lr_rows["method"] == "muon"].iloc[0]
        block4 = lr_rows.loc[lr_rows["method"] == "block4"].iloc[0]
        muon_block4_rows.append(
            {
                "matrix_lr": matrix_lr,
                "reported_common_threshold": muon["reported_common_threshold"],
                "muon_final_val_loss": muon["final_val_loss"],
                "block4_final_val_loss": block4["final_val_loss"],
                "muon_minus_block4_final_val_loss": muon["final_val_loss"]
                - block4["final_val_loss"],
                "muon_late_val_mean": muon["late_val_mean_last20pct"],
                "block4_late_val_mean": block4["late_val_mean_last20pct"],
                "muon_minus_block4_late_val_mean": muon["late_val_mean_last20pct"]
                - block4["late_val_mean_last20pct"],
                "muon_normalized_val_auc": muon["normalized_val_auc"],
                "block4_normalized_val_auc": block4["normalized_val_auc"],
                "muon_minus_block4_normalized_val_auc": muon["normalized_val_auc"]
                - block4["normalized_val_auc"],
                "muon_same_loss_step": muon["interpolated_step"],
                "block4_same_loss_step": block4["interpolated_step"],
                "muon_minus_block4_same_loss_step": muon["interpolated_step"]
                - block4["interpolated_step"],
                "muon_same_loss_time_s": muon["interpolated_time_s"],
                "block4_same_loss_time_s": block4["interpolated_time_s"],
                "muon_minus_block4_same_loss_time_s": muon["interpolated_time_s"]
                - block4["interpolated_time_s"],
                "muon_last_time_s": muon["time_elapsed_last_s"],
                "block4_last_time_s": block4["time_elapsed_last_s"],
                "muon_minus_block4_last_time_s": muon["time_elapsed_last_s"]
                - block4["time_elapsed_last_s"],
            }
        )
    muon_block4_frame = pd.DataFrame(muon_block4_rows).sort_values("matrix_lr")

    resource_frame = (
        summary_frame.groupby("method", as_index=False)
        .agg(
            total_k_state_mib=("total_k_state_mib", "median"),
            cproj_k_state_mib=("cproj_k_state_mib", "median"),
            non_cproj_k_state_mib=("non_cproj_k_state_mib", "median"),
            cuda_allocated_mib=("cuda_allocated_last_mib", "median"),
            cuda_full_run_peak_mib=("cuda_full_run_peak_mib", "max"),
            elapsed_time_mean_s=("time_elapsed_last_s", "mean"),
            elapsed_time_min_s=("time_elapsed_last_s", "min"),
            elapsed_time_max_s=("time_elapsed_last_s", "max"),
        )
        .sort_values(["total_k_state_mib", "method"])
    )

    suite_names = set(run_meta["suite"])
    if len(suite_names) != 1:
        raise ValueError(f"Expected one suite, observed {sorted(suite_names)}")
    suite_name = next(iter(suite_names))
    artifact_prefix = (
        "b12_lr_grid_seed2026" if suite_name == "owt12l_3k" else "b24_lr_grid_seed2026"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory_rows).sort_values("metric").to_csv(
        output_dir / f"{artifact_prefix}_source_inventory.csv", index=False
    )
    pd.DataFrame(duplicate_checks).sort_values(["metric", "run_name", "display_column"]).to_csv(
        output_dir / f"{artifact_prefix}_wandb_duplicate_checks.csv", index=False
    )
    pd.DataFrame(checks).to_csv(output_dir / f"{artifact_prefix}_data_quality_checks.csv", index=False)
    summary_frame.to_csv(output_dir / f"{artifact_prefix}_run_summary.csv", index=False)
    equal_loss_frame.to_csv(output_dir / f"{artifact_prefix}_equal_loss_efficiency.csv", index=False)
    comparison_frame.to_csv(output_dir / f"{artifact_prefix}_pairwise_vs_none.csv", index=False)
    best_lr_frame.to_csv(output_dir / f"{artifact_prefix}_best_observed_lr_by_method.csv", index=False)
    muon_block4_frame.to_csv(output_dir / f"{artifact_prefix}_muon_vs_block4.csv", index=False)
    resource_frame.to_csv(output_dir / f"{artifact_prefix}_resource_summary.csv", index=False)
    val_long.sort_values(["matrix_lr", "method", "step"]).to_csv(
        output_dir / f"{artifact_prefix}_val_loss_series.csv", index=False
    )
    train_long.sort_values(["matrix_lr", "method", "step"]).to_csv(
        output_dir / f"{artifact_prefix}_train_loss_series.csv", index=False
    )

    failed = [row for row in checks if row["status"] != "PASS"]
    print(f"Wrote analysis to {output_dir}")
    print(f"Runs: {len(summary_frame)}; failed data-quality checks: {len(failed)}")
    print(summary_frame[["matrix_lr", "method", "final_val_loss", "best_val_loss", "val_rebound_final_minus_best", "time_elapsed_last_s"]].to_string(index=False))
    print("\nEqual-loss summary:")
    print(equal_loss_frame[["matrix_lr", "reported_common_threshold", "method", "interpolated_step", "interpolated_time_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
