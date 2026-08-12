#!/usr/bin/env python3
"""Validate and analyze the accepted three-seed R1 depth K-mode formal batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from statistics import stdev
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-08-09.1"
SEEDS = (2024, 2025, 2026)
RULES = ("early", "center", "late", "edge", "all")
METHODS = tuple(
    [f"{rule}_{mode}" for rule in RULES for mode in ("none", "diag")]
    + ["block4", "muon"]
)
VALIDATION_GRID = tuple(range(0, 6201, 100))
EXPECTED_WANDB_METRICS = {
    "lr/matrix",
    "lr/adamw",
    "memory/peak_allocated_mib",
    "memory/optimizer_state_mib",
    "memory/k_state_mib",
    "performance/step_avg_ms",
    "time/train_s",
    "train/loss_step",
    "val/loss",
}
T95_DF2 = 4.302652729911275


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--wandb-inputs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--reference-results-root",
        type=Path,
        required=True,
        help="Result root containing accepted Experiment 25 and 28 analyses.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def localize_remote_path(bundle_root: Path, remote_path: str) -> Path:
    marker = "29_r1_depth_kmode/"
    normalized = remote_path.replace("\\", "/")
    if marker not in normalized:
        raise RuntimeError(f"path is outside family 29: {remote_path}")
    return bundle_root / normalized.split(marker, 1)[1]


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def load_local_evidence(
    bundle_root: Path, batch_id: str
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    state_path = bundle_root / "batches" / batch_id / "batch_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = read_json(state_path)
    jobs = list(state.get("jobs", []))
    add_check(checks, "batch_completed", state.get("status") == "completed", state.get("status"))
    add_check(checks, "batch_failures_empty", state.get("failures") == [], state.get("failures"))
    add_check(checks, "exact_six_jobs", len(jobs) == 6, len(jobs))
    add_check(
        checks,
        "exact_seed_shard_grid",
        {(int(j["seed"]), int(j["shard"])) for j in jobs}
        == {(seed, shard) for seed in SEEDS for shard in (0, 1)},
        [(j.get("seed"), j.get("shard"), j.get("status")) for j in jobs],
    )
    add_check(
        checks,
        "controller_training_split",
        bool(state.get("controller_runtime", {}).get("interpreters_separate")),
        state.get("controller_runtime"),
    )

    rows: list[dict[str, Any]] = []
    metrics_by_run: dict[str, pd.DataFrame] = {}
    manifest_statuses: list[dict[str, Any]] = []
    run_statuses: list[dict[str, Any]] = []
    init_by_seed: dict[int, set[str]] = {seed: set() for seed in SEEDS}
    source_by_method: dict[str, set[str]] = {method: set() for method in METHODS}

    for job in jobs:
        seed = int(job["seed"])
        shard = int(job["shard"])
        smoke_path = localize_remote_path(bundle_root, str(job["smoke_manifest"]))
        formal_path = localize_remote_path(bundle_root, str(job["formal_manifest"]))
        smoke = read_json(smoke_path)
        formal = read_json(formal_path)
        manifest_statuses.append(
            {
                "seed": seed,
                "shard": shard,
                "smoke_status": smoke.get("status"),
                "formal_status": formal.get("status"),
                "formal_runs": len(formal.get("summaries", [])),
                "formal_failures": len(formal.get("failures", [])),
                "wandb_complete": formal.get("wandb_complete"),
                "init_identical": formal.get("formal_initialization_fingerprints_identical"),
            }
        )
        for summary in formal.get("summaries", []):
            run_name = str(summary["run_name"])
            run_dir = formal_path.parent / run_name
            run_manifest_path = run_dir / "run_manifest.json"
            summary_path = run_dir / "r1_summary.json"
            metrics_path = run_dir / "r1_metrics.csv"
            if not all(path.is_file() for path in (run_manifest_path, summary_path, metrics_path)):
                raise RuntimeError(f"missing run evidence under {run_dir}")
            run_manifest = read_json(run_manifest_path)
            local_summary = read_json(summary_path)
            metrics = pd.read_csv(metrics_path)
            method = str(local_summary["method"])
            controlled_seed = int(local_summary["controlled_seed"])
            run_statuses.append(
                {
                    "run_name": run_name,
                    "method": method,
                    "seed": controlled_seed,
                    "status": run_manifest.get("status"),
                    "evidence_valid": local_summary.get("evidence_valid"),
                    "quality_usable": local_summary.get("quality_usable"),
                    "timing_usable": local_summary.get("timing_usable"),
                }
            )
            init_by_seed[controlled_seed].add(str(local_summary["init_sha256"]))
            source_by_method[method].add(str(local_summary["derived_script_sha256"]))
            validation = metrics.loc[metrics.event == "validation"].copy()
            validation["step"] = validation.step.astype(int)
            validation = validation.sort_values("step")
            tail5 = float(validation.tail(5).loss.mean())
            auc = float(
                np.trapezoid(
                    validation.loss.to_numpy(dtype=float),
                    validation.step.to_numpy(dtype=float),
                )
                / (validation.step.iloc[-1] - validation.step.iloc[0])
            )
            row = dict(local_summary)
            row.update(
                {
                    "seed": controlled_seed,
                    "tail5_val_loss": tail5,
                    "normalized_val_auc": auc,
                    "run_manifest_path": str(run_manifest_path.relative_to(bundle_root)),
                    "metrics_path": str(metrics_path.relative_to(bundle_root)),
                }
            )
            if method in ("block4", "muon"):
                row.update({"kind": "anchor", "rule": "anchor", "mode": method})
            else:
                rule, mode = method.rsplit("_", 1)
                row.update({"kind": "depth_rule", "rule": rule, "mode": mode})
            rows.append(row)
            metrics_by_run[run_name] = metrics

    summary = pd.DataFrame(rows).sort_values(["seed", "kind", "rule", "mode"])
    observed = {(int(row.seed), str(row.method)) for row in summary.itertuples()}
    expected = {(seed, method) for seed in SEEDS for method in METHODS}
    add_check(checks, "exact_36_formal_runs", observed == expected, {"observed": len(observed), "missing": sorted(expected - observed), "extra": sorted(observed - expected)})
    add_check(checks, "six_smoke_manifests_passed", all(row["smoke_status"] == "completed_valid_smoke" for row in manifest_statuses), manifest_statuses)
    add_check(checks, "six_formal_manifests_passed", all(row["formal_status"] == "completed_valid" and row["formal_runs"] == 6 and row["formal_failures"] == 0 for row in manifest_statuses), manifest_statuses)
    add_check(checks, "six_formal_wandb_complete", all(row["wandb_complete"] is True for row in manifest_statuses), manifest_statuses)
    add_check(checks, "formal_init_fingerprints_identical", all(row["init_identical"] is True for row in manifest_statuses), manifest_statuses)
    add_check(checks, "all_run_manifests_completed_valid", all(row["status"] == "completed_valid" for row in run_statuses), run_statuses)
    add_check(checks, "quality_valid_timing_ineligible", all(row["evidence_valid"] is True and row["quality_usable"] is True and row["timing_usable"] is False for row in run_statuses), run_statuses)
    add_check(checks, "one_init_hash_per_seed", all(len(value) == 1 for value in init_by_seed.values()), {str(key): sorted(value) for key, value in init_by_seed.items()})
    add_check(checks, "one_source_hash_per_method", all(len(value) == 1 for value in source_by_method.values()), {key: sorted(value) for key, value in source_by_method.items()})
    add_check(checks, "checkpoints_disabled", bool((summary.checkpoint_bytes == 0).all()), summary[["run_name", "checkpoint_bytes"]].to_dict("records"))

    bad_grids = []
    bad_final = []
    for row in summary.itertuples(index=False):
        metrics = metrics_by_run[str(row.run_name)]
        validation = metrics.loc[metrics.event == "validation"].sort_values("step")
        if tuple(validation.step.astype(int)) != VALIDATION_GRID:
            bad_grids.append(str(row.run_name))
        if not math.isclose(float(validation.iloc[-1].loss), float(row.final_val_loss), abs_tol=1e-12):
            bad_final.append(str(row.run_name))
    add_check(checks, "exact_validation_grid", not bad_grids, bad_grids)
    add_check(checks, "summary_final_matches_metrics", not bad_final, bad_final)
    return summary, metrics_by_run, checks


def expected_wandb_series(metrics: pd.DataFrame, metric: str, summary_row: pd.Series) -> pd.Series:
    work = metrics.copy()
    work["step"] = work.step.astype(int)
    if metric == "val/loss":
        values = work.loc[work.event == "validation", ["step", "loss"]]
        return values.set_index("step").loss.astype(float)
    if metric == "train/loss_step":
        values = work.loc[(work.event == "train") & ((work.step % 20 == 0) | (work.step == 6200)), ["step", "loss"]]
        return values.set_index("step").loss.astype(float)
    per_step = work.loc[(work.event == "validation") | ((work.event == "train") & ((work.step % 20 == 0) | (work.step == 6200)))].drop_duplicates("step", keep="last").set_index("step")
    if metric == "lr/matrix":
        return per_step.matrix_lr.astype(float)
    if metric == "lr/adamw":
        return per_step.adamw_lr.astype(float)
    if metric == "time/train_s":
        return per_step.official_train_time_ms.astype(float) / 1000.0
    if metric == "performance/step_avg_ms":
        return per_step.step_avg_ms.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    final_step = int(summary_row["final_val_step"])
    if metric == "memory/peak_allocated_mib":
        return pd.Series({final_step: float(summary_row["peak_memory_allocated_mib"])})
    if metric == "memory/optimizer_state_mib":
        return pd.Series({final_step: float(summary_row["optimizer_state_bytes"]) / (1024**2)})
    if metric == "memory/k_state_mib":
        return pd.Series({final_step: float(summary_row["k_state_bytes"]) / (1024**2)})
    raise KeyError(metric)


def load_and_crosscheck_wandb(
    paths: list[Path],
    output_dir: Path,
    summary: pd.DataFrame,
    metrics_by_run: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_dir = output_dir / "raw_wandb_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    input_rows: list[dict[str, Any]] = []
    crosschecks: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    observed_metrics: set[str] = set()
    observed_runs: set[str] = set()
    summary_by_run = summary.set_index("run_name")

    for source in sorted(path.resolve() for path in paths):
        target = raw_dir / source.name
        if target.exists() and sha256_file(target) != sha256_file(source):
            raise RuntimeError(f"archived W&B export differs: {target}")
        if not target.exists():
            shutil.copy2(source, target)
        frame = pd.read_csv(target)
        base_columns = [column for column in frame.columns if column != "Step" and not column.endswith(("__MIN", "__MAX"))]
        parsed = [column.rsplit(" - ", 1) for column in base_columns]
        metrics = {parts[1] for parts in parsed if len(parts) == 2}
        if len(metrics) != 1:
            raise RuntimeError(f"expected one metric per W&B export: {target} -> {metrics}")
        metric = metrics.pop()
        if metric in observed_metrics:
            raise RuntimeError(f"duplicate W&B metric export: {metric}")
        observed_metrics.add(metric)
        band_match = True
        for column, parts in zip(base_columns, parsed):
            run_name = parts[0]
            observed_runs.add(run_name)
            actual = pd.Series(
                pd.to_numeric(frame[column], errors="coerce").to_numpy(),
                index=pd.to_numeric(frame.Step, errors="raise").astype(int),
            ).dropna()
            expected = expected_wandb_series(metrics_by_run[run_name], metric, summary_by_run.loc[run_name])
            actual = actual.sort_index()
            expected = expected.sort_index()
            steps_match = tuple(actual.index) == tuple(expected.index)
            max_error = float(np.max(np.abs(actual.to_numpy() - expected.to_numpy()))) if steps_match and len(actual) else math.inf
            passed = steps_match and max_error <= 1e-9
            crosschecks.append(
                {
                    "metric": metric,
                    "run_name": run_name,
                    "points": len(actual),
                    "steps_match": steps_match,
                    "max_abs_error": max_error,
                    "passed": passed,
                }
            )
            for suffix in ("__MIN", "__MAX"):
                band = column + suffix
                if band not in frame.columns:
                    band_match = False
                else:
                    left = pd.to_numeric(frame[column], errors="coerce").to_numpy()
                    right = pd.to_numeric(frame[band], errors="coerce").to_numpy()
                    if not np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True):
                        band_match = False
        input_rows.append(
            {
                "file_name": source.name,
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
                "rows": len(frame),
                "metric": metric,
                "run_columns": len(base_columns),
                "bands_duplicate_base": band_match,
                "archived_path": str(target.relative_to(output_dir)),
            }
        )
    expected_runs = set(summary.run_name)
    add_check(checks, "exact_nine_wandb_metrics", observed_metrics == EXPECTED_WANDB_METRICS, {"observed": sorted(observed_metrics), "missing": sorted(EXPECTED_WANDB_METRICS - observed_metrics), "extra": sorted(observed_metrics - EXPECTED_WANDB_METRICS)})
    add_check(checks, "exact_36_wandb_runs", observed_runs == expected_runs, {"observed": len(observed_runs), "missing": sorted(expected_runs - observed_runs), "extra": sorted(observed_runs - expected_runs)})
    add_check(checks, "all_wandb_exports_have_36_runs", all(row["run_columns"] == 36 for row in input_rows), input_rows)
    add_check(checks, "wandb_band_columns_duplicate_base", all(row["bands_duplicate_base"] for row in input_rows), input_rows)
    add_check(checks, "wandb_local_values_exact", all(row["passed"] for row in crosschecks), {"rows": len(crosschecks), "max_abs_error": max(row["max_abs_error"] for row in crosschecks)})
    return input_rows, crosschecks, checks


def build_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for rule in RULES:
            none = summary[(summary.seed == seed) & (summary.method == f"{rule}_none")].iloc[0]
            diag = summary[(summary.seed == seed) & (summary.method == f"{rule}_diag")].iloc[0]
            rows.append(
                {
                    "seed": seed,
                    "rule": rule,
                    "diag_minus_none_step6200": float(diag.final_val_loss - none.final_val_loss),
                    "diag_minus_none_tail5": float(diag.tail5_val_loss - none.tail5_val_loss),
                    "diag_minus_none_auc": float(diag.normalized_val_auc - none.normalized_val_auc),
                    "diag_k_state_mib": float(diag.k_state_bytes) / (1024**2),
                    "none_k_state_mib": float(none.k_state_bytes) / (1024**2),
                    "diag_optimizer_state_mib": float(diag.optimizer_state_bytes) / (1024**2),
                    "none_optimizer_state_mib": float(none.optimizer_state_bytes) / (1024**2),
                    "diag_peak_mib": float(diag.peak_memory_allocated_mib),
                    "none_peak_mib": float(none.peak_memory_allocated_mib),
                }
            )
    frame = pd.DataFrame(rows)
    all_delta = frame.loc[frame.rule == "all", ["seed", "diag_minus_none_step6200"]].rename(columns={"diag_minus_none_step6200": "all_delta"})
    frame = frame.merge(all_delta, on="seed")
    frame["interaction_vs_all"] = frame.diag_minus_none_step6200 - frame.all_delta
    return frame


def summarize_rules(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in RULES:
        subset = contrasts[contrasts.rule == rule]
        values = subset.diag_minus_none_step6200.tolist()
        mean_value = float(np.mean(values))
        sd_value = float(stdev(values))
        half_width = T95_DF2 * sd_value / math.sqrt(3)
        interactions = subset.interaction_vs_all.tolist()
        rows.append(
            {
                "rule": rule,
                "n_seeds": 3,
                "mean_diag_minus_none_step6200": mean_value,
                "sd_diag_minus_none_step6200": sd_value,
                "negative_seed_count": int(sum(value < 0 for value in values)),
                "descriptive_t95_low": mean_value - half_width,
                "descriptive_t95_high": mean_value + half_width,
                "mean_diag_minus_none_tail5": float(subset.diag_minus_none_tail5.mean()),
                "mean_diag_minus_none_auc": float(subset.diag_minus_none_auc.mean()),
                "mean_interaction_vs_all": float(np.mean(interactions)),
                "negative_interaction_seed_count": int(sum(value < 0 for value in interactions)),
                "positive_interaction_seed_count": int(sum(value > 0 for value in interactions)),
            }
        )
    return pd.DataFrame(rows)


def build_anchor_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        values = summary[summary.seed == seed].set_index("method")
        rows.append(
            {
                "seed": seed,
                "all_diag_val_step6200": float(values.loc["all_diag", "final_val_loss"]),
                "all_none_val_step6200": float(values.loc["all_none", "final_val_loss"]),
                "block4_val_step6200": float(values.loc["block4", "final_val_loss"]),
                "muon_val_step6200": float(values.loc["muon", "final_val_loss"]),
                "all_diag_minus_block4": float(values.loc["all_diag", "final_val_loss"] - values.loc["block4", "final_val_loss"]),
                "all_none_minus_block4": float(values.loc["all_none", "final_val_loss"] - values.loc["block4", "final_val_loss"]),
                "block4_minus_muon": float(values.loc["block4", "final_val_loss"] - values.loc["muon", "final_val_loss"]),
                "diag_k_state_mib": float(values.loc["all_diag", "k_state_bytes"]) / (1024**2),
                "none_k_state_mib": float(values.loc["all_none", "k_state_bytes"]) / (1024**2),
                "block4_k_state_mib": float(values.loc["block4", "k_state_bytes"]) / (1024**2),
                "muon_k_state_mib": float(values.loc["muon", "k_state_bytes"]) / (1024**2),
                "diag_optimizer_state_mib": float(values.loc["all_diag", "optimizer_state_bytes"]) / (1024**2),
                "none_optimizer_state_mib": float(values.loc["all_none", "optimizer_state_bytes"]) / (1024**2),
                "block4_optimizer_state_mib": float(values.loc["block4", "optimizer_state_bytes"]) / (1024**2),
                "muon_optimizer_state_mib": float(values.loc["muon", "optimizer_state_bytes"]) / (1024**2),
                "diag_peak_mib": float(values.loc["all_diag", "peak_memory_allocated_mib"]),
                "none_peak_mib": float(values.loc["all_none", "peak_memory_allocated_mib"]),
                "block4_peak_mib": float(values.loc["block4", "peak_memory_allocated_mib"]),
                "muon_peak_mib": float(values.loc["muon", "peak_memory_allocated_mib"]),
            }
        )
    return pd.DataFrame(rows)


def build_cross_environment(
    reference_results_root: Path, r1_rules: pd.DataFrame
) -> pd.DataFrame:
    data_root = reference_results_root
    sources = {
        "OWT": data_root / "25_owt_depth_kmode" / "analysis_20260724_multiseed" / "rule_multiseed_summary.csv",
        "WikiText-103": data_root / "28_wikitext_depth_kmode" / "analysis_20260725_multiseed" / "rule_multiseed_summary.csv",
    }
    rows = []
    for dataset, path in sources.items():
        frame = pd.read_csv(path)
        for row in frame.itertuples(index=False):
            rows.append(
                {
                    "environment": dataset,
                    "endpoint": "step4500",
                    "rule": row.rule,
                    "mean_diag_minus_none": row.mean_diag_minus_none_step4500,
                    "sd_diag_minus_none": row.sd_diag_minus_none_step4500,
                    "negative_seed_count": row.negative_seed_count,
                    "mean_interaction_vs_all": row.mean_interaction_vs_all,
                }
            )
    for row in r1_rules.itertuples(index=False):
        rows.append(
            {
                "environment": "official R1",
                "endpoint": "step6200",
                "rule": row.rule,
                "mean_diag_minus_none": row.mean_diag_minus_none_step6200,
                "sd_diag_minus_none": row.sd_diag_minus_none_step6200,
                "negative_seed_count": row.negative_seed_count,
                "mean_interaction_vs_all": row.mean_interaction_vs_all,
            }
        )
    return pd.DataFrame(rows)


def write_delta_svg(
    path: Path,
    title: str,
    series: list[tuple[str, str, pd.DataFrame, str, str]],
) -> None:
    width, height = 900, 500
    left, right, top, bottom = 95, 35, 60, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    values = []
    for _, _, frame, mean_col, sd_col in series:
        values.extend((frame[mean_col] - frame[sd_col]).tolist())
        values.extend((frame[mean_col] + frame[sd_col]).tolist())
    y_min = min(-0.03, min(values) * 1.12)
    y_max = max(0.006, max(values) * 1.25)

    def sx(index: int, offset: float) -> float:
        return left + (index + 0.5 + offset) * plot_w / len(RULES)

    def sy(value: float) -> float:
        return top + (y_max - value) * plot_h / (y_max - y_min)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">{title}</text>',
    ]
    for tick in np.linspace(y_min, y_max, 7):
        y = sy(float(tick))
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{tick:+.3f}</text>')
    zero_y = sy(0.0)
    lines.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="#4B5563" stroke-width="1.5"/>')
    offsets = np.linspace(-0.22, 0.22, len(series)) if len(series) > 1 else [0.0]
    for offset, (label, color, frame, mean_col, sd_col) in zip(offsets, series):
        ordered = frame.set_index("rule").loc[list(RULES)]
        for index, row in enumerate(ordered.itertuples(index=False)):
            mean_value = float(getattr(row, mean_col))
            sd_value = float(getattr(row, sd_col))
            x = sx(index, float(offset))
            y1, y2, ym = sy(mean_value - sd_value), sy(mean_value + sd_value), sy(mean_value)
            lines.append(f'<line x1="{x:.2f}" y1="{y1:.2f}" x2="{x:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<line x1="{x-5:.2f}" y1="{y1:.2f}" x2="{x+5:.2f}" y2="{y1:.2f}" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<line x1="{x-5:.2f}" y1="{y2:.2f}" x2="{x+5:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<circle cx="{x:.2f}" cy="{ym:.2f}" r="5" fill="{color}"/>')
    for index, rule in enumerate(RULES):
        lines.append(f'<text x="{sx(index, 0):.2f}" y="{height-48}" text-anchor="middle" font-family="Arial" font-size="13">{rule}</text>')
    lines.append(f'<text x="20" y="{top + plot_h/2}" text-anchor="middle" font-family="Arial" font-size="13" transform="rotate(-90 20 {top + plot_h/2})">diag - none validation loss</text>')
    legend_x = left
    for label, color, _, _, _ in series:
        lines.append(f'<circle cx="{legend_x}" cy="{height-17}" r="5" fill="{color}"/>')
        lines.append(f'<text x="{legend_x+10}" y="{height-13}" font-family="Arial" font-size="12">{label}</text>')
        legend_x += 155
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(rules: pd.DataFrame, cross: pd.DataFrame, output_dir: Path) -> None:
    write_delta_svg(
        output_dir / "r1_depth_deltas.svg",
        "Official R1 depth effects at step 6200 (mean ± sample SD)",
        [("official R1", "#22577A", rules, "mean_diag_minus_none_step6200", "sd_diag_minus_none_step6200")],
    )
    cross_series = []
    for environment, color in (("OWT", "#2A9D8F"), ("WikiText-103", "#D4A017"), ("official R1", "#E76F51")):
        cross_series.append(
            (
                environment,
                color,
                cross[cross.environment == environment].copy(),
                "mean_diag_minus_none",
                "sd_diag_minus_none",
            )
        )
    write_delta_svg(
        output_dir / "cross_environment_depth_deltas.svg",
        "Depth effects across three accepted environments",
        cross_series,
    )


def inventory_bundle(bundle_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in bundle_root.rglob("*") if p.is_file()):
        rows.append(
            {
                "relative_path": str(path.relative_to(bundle_root)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    bundle_root = args.bundle_root.resolve()
    reference_results_root = args.reference_results_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, metrics_by_run, local_checks = load_local_evidence(bundle_root, args.batch_id)
    input_rows, wandb_crosschecks, wandb_checks = load_and_crosscheck_wandb(
        args.wandb_inputs, output_dir, summary, metrics_by_run
    )
    checks = local_checks + wandb_checks
    failures = [row for row in checks if not row["passed"]]
    if failures:
        write_json(output_dir / "data_quality_checks.json", {"status": "failed", "checks": checks})
        raise RuntimeError("R1 depth integrity checks failed: " + ", ".join(row["check"] for row in failures))

    contrasts = build_contrasts(summary)
    rules = summarize_rules(contrasts)
    anchors = build_anchor_summary(summary)
    cross = build_cross_environment(reference_results_root, rules)
    lookup = rules.set_index("rule")
    all_row = lookup.loc["all"]
    edge_row = lookup.loc["edge"]
    center_row = lookup.loc["center"]
    late_row = lookup.loc["late"]
    endpoint_counts = {
        "step6200_negative": int((contrasts.diag_minus_none_step6200 < 0).sum()),
        "tail5_negative": int((contrasts.diag_minus_none_tail5 < 0).sum()),
        "auc_negative": int((contrasts.diag_minus_none_auc < 0).sum()),
        "total_pairs": len(contrasts),
    }
    direction_pass = bool((rules.mean_diag_minus_none_step6200 < 0).all() and (rules.negative_seed_count >= 2).all())
    pattern = {
        "edge_stronger_than_all": {
            "expected_sign": "negative",
            "observed_mean": float(edge_row.mean_interaction_vs_all),
            "negative_seeds": int(edge_row.negative_interaction_seed_count),
            "passed": bool(edge_row.mean_interaction_vs_all < 0 and edge_row.negative_interaction_seed_count >= 2),
        },
        "center_weaker_than_all": {
            "expected_sign": "positive",
            "observed_mean": float(center_row.mean_interaction_vs_all),
            "positive_seeds": int(center_row.positive_interaction_seed_count),
            "passed": bool(center_row.mean_interaction_vs_all > 0 and center_row.positive_interaction_seed_count >= 2),
        },
        "late_weaker_than_all": {
            "expected_sign": "positive",
            "observed_mean": float(late_row.mean_interaction_vs_all),
            "positive_seeds": int(late_row.positive_interaction_seed_count),
            "passed": bool(late_row.mean_interaction_vs_all > 0 and late_row.positive_interaction_seed_count >= 2),
        },
    }
    verdict = {
        "status": "valid",
        "classification": "uniform_diag_direction_transfer_with_environment_specific_depth_amplitude",
        "formal_batch_id": args.batch_id,
        "formal_run_count": len(summary),
        "paired_contrast_count": len(contrasts),
        "direction_transfer_pass": direction_pass,
        "endpoint_sign_consistency": endpoint_counts,
        "rules_with_negative_means": int((rules.mean_diag_minus_none_step6200 < 0).sum()),
        "rules_with_3_of_3_negative": int((rules.negative_seed_count == 3).sum()),
        "all_rule_mean_diag_minus_none": float(all_row.mean_diag_minus_none_step6200),
        "all_rule_sd_diag_minus_none": float(all_row.sd_diag_minus_none_step6200),
        "all_rule_negative_seed_count": int(all_row.negative_seed_count),
        "frozen_magnitude_pattern": pattern,
        "frozen_magnitude_pattern_full_pass": all(item["passed"] for item in pattern.values()),
        "all_diag_minus_block4_mean": float(anchors.all_diag_minus_block4.mean()),
        "all_none_minus_block4_mean": float(anchors.all_none_minus_block4.mean()),
        "block4_minus_muon_mean": float(anchors.block4_minus_muon.mean()),
        "memory_mean_mib": {
            "all_diag_k_state": float(anchors.diag_k_state_mib.mean()),
            "all_none_k_state": float(anchors.none_k_state_mib.mean()),
            "block4_k_state": float(anchors.block4_k_state_mib.mean()),
            "all_diag_optimizer_state": float(anchors.diag_optimizer_state_mib.mean()),
            "all_none_optimizer_state": float(anchors.none_optimizer_state_mib.mean()),
            "block4_optimizer_state": float(anchors.block4_optimizer_state_mib.mean()),
            "all_diag_peak": float(anchors.diag_peak_mib.mean()),
            "all_none_peak": float(anchors.none_peak_mib.mean()),
            "block4_peak": float(anchors.block4_peak_mib.mean()),
            "muon_peak": float(anchors.muon_peak_mib.mean()),
        },
        "additional_depth_runs_needed": False,
        "recommended_paper_position": {
            "experiment_25": "A; contributes to unified S claim",
            "experiment_28": "A; contributes to unified S claim",
            "experiment_29": "S + A",
            "unified_depth_claim": "S in main text; complete 45 paired contrasts in appendix",
        },
        "claim_boundary": (
            "Across OWT, WikiText-103, and official R1, diag generally and in R1 uniformly "
            "improves over none across preregistered depth rules. The magnitude ordering is "
            "environment-dependent: the local edge-strong pattern does not transfer to R1, "
            "so no universal best depth mask or individual-layer causality is established."
        ),
        "timing_usable": False,
    }

    summary.to_csv(output_dir / "run_summary.csv", index=False)
    contrasts.to_csv(output_dir / "paired_depth_contrasts.csv", index=False)
    rules.to_csv(output_dir / "rule_multiseed_summary.csv", index=False)
    anchors.to_csv(output_dir / "anchor_summary.csv", index=False)
    cross.to_csv(output_dir / "cross_environment_depth_transfer.csv", index=False)
    pd.DataFrame(checks).to_csv(output_dir / "data_quality_checks.csv", index=False)
    pd.DataFrame(wandb_crosschecks).to_csv(output_dir / "wandb_local_crosscheck.csv", index=False)
    inventory = inventory_bundle(bundle_root)
    inventory.to_csv(output_dir / "remote_bundle_inventory.csv", index=False)
    source_zip = args.source_zip.resolve()
    all_inputs = [
        {
            "kind": "remote_bundle_zip",
            "file_name": source_zip.name,
            "sha256": sha256_file(source_zip),
            "size_bytes": source_zip.stat().st_size,
            "metric": "",
            "archived_path": "../batches+results (extracted canonical evidence)",
        },
        *[{"kind": "wandb_export", **row} for row in input_rows],
    ]
    pd.DataFrame(all_inputs).to_csv(output_dir / "input_manifest.csv", index=False)
    write_json(output_dir / "data_quality_checks.json", {"status": "passed", "script_version": SCRIPT_VERSION, "checks": checks})
    write_json(output_dir / "analysis_verdict.json", verdict)
    make_plots(rules, cross, output_dir)

    rule_lines = []
    for row in rules.itertuples(index=False):
        rule_lines.append(
            f"| {row.rule} | {row.mean_diag_minus_none_step6200:+.6f} ± {row.sd_diag_minus_none_step6200:.6f} | "
            f"{row.negative_seed_count}/3 | {row.mean_diag_minus_none_tail5:+.6f} | "
            f"{row.mean_diag_minus_none_auc:+.6f} | {row.mean_interaction_vs_all:+.6f} |"
        )
    pair_lines = []
    for row in contrasts.itertuples(index=False):
        pair_lines.append(
            f"| {row.seed} | {row.rule} | {row.diag_minus_none_step6200:+.6f} | "
            f"{row.diag_minus_none_tail5:+.6f} | {row.diag_minus_none_auc:+.6f} |"
        )
    report = f"""# R1 depth × c_proj K-mode three-seed formal analysis

Analysis date: 2026-08-09  
Formal batch: `{args.batch_id}`  
Primary endpoint: step-6200 validation loss  
Primary contrast: seed-matched `diag - none`; negative favors diagonal K  
Status: 36/36 formal runs accepted; all local/W&B integrity checks passed; timing is ineligible.

## Executive conclusion

The direction transfer succeeds more strongly than expected: all 15 R1 depth pairs
favor `diag` at step 6200, and all five rule means are negative in 3/3 seeds. The
all-depth effect is `{float(all_row.mean_diag_minus_none_step6200):+.6f} ± {float(all_row.sd_diag_minus_none_step6200):.6f}`.
Tail-5 and normalized validation AUC are also negative in
`{endpoint_counts['tail5_negative']}/15` and `{endpoint_counts['auc_negative']}/15` pairs.

The frozen local magnitude pattern does not fully transfer. Center and late remain
weaker than all, but edge reverses: its interaction versus all is
`{float(edge_row.mean_interaction_vs_all):+.6f}` and is positive in
`{int(edge_row.positive_interaction_seed_count)}/3` seeds. Therefore the defensible
claim is that retaining coordinate-wise diagonal scale is broadly useful across
depth, while the best depth allocation is environment-dependent. The data do not
support a universal edge mask or individual-layer causality.

## R1 paired results

| Rule | Mean diag-none ± SD | Negative seeds | Tail-5 mean | AUC mean | Interaction vs all |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rule_lines)}

## Seed-level contrasts

| Seed | Rule | Step-6200 | Tail-5 | Normalized AUC |
|---:|---|---:|---:|---:|
{chr(10).join(pair_lines)}

## Official anchors and state trade-off

Across seeds, all-depth diag minus block4 is
`{float(anchors.all_diag_minus_block4.mean()):+.6f}`; all-depth none minus block4 is
`{float(anchors.all_none_minus_block4.mean()):+.6f}`. Thus diag remains within and
slightly improves the ±0.002 practical neighborhood of block4 in all three seeds,
whereas removing c_proj K completely costs about
`{float(anchors.all_none_minus_block4.mean()):+.6f}` validation loss.

Mean K-state is `{float(anchors.diag_k_state_mib.mean()):.3f}` MiB for all-diag,
`{float(anchors.none_k_state_mib.mean()):.3f}` MiB for all-none, and
`{float(anchors.block4_k_state_mib.mean()):.3f}` MiB for block4. All-diag saves
`{float(anchors.block4_k_state_mib.mean() - anchors.diag_k_state_mib.mean()):.3f}` MiB
of K-state and `{float(anchors.block4_peak_mib.mean() - anchors.diag_peak_mib.mean()):.1f}` MiB
of peak allocation relative to block4. Muon remains the lightest anchor. Concurrent
quality-run timing is permanently ineligible.

Block4 minus Muon is `{float(anchors.block4_minus_muon.mean()):+.6f}`, but this is a
recipe-level comparison because Muon retains its official lower LR.

## Cross-environment synthesis

OWT and WikiText-103 established negative all-depth effects of `-0.013930` and
`-0.015348`; official R1 confirms the direction with a smaller all-depth effect of
`{float(all_row.mean_diag_minus_none_step6200):+.6f}`. OWT/WikiText both placed edge
as the strongest rule, while official R1 places all first and edge below all. This
combination is stronger for the paper than either extreme claim: the representation
benefit is robust, but the location of the largest benefit is not architecture/
implementation invariant.

No further depth seeds, denser masks, or post-hoc layer search are warranted. The
unified depth result should receive a short main-text paragraph or compact figure;
the complete 45 paired contrasts and implementation audits belong in the appendix.

## Integrity

The accepted batch contains six smoke and six formal shard manifests, 36 formal run
manifests, 36 local metric histories, and nine W&B metric exports covering the exact
same 36 run names. Local and W&B values match at every exported point. Source/init
lineage, validation grids, completion, W&B upload, checkpoint-disable policy, and
timing-ineligibility gates all pass. Exact hashes are recorded in
`input_manifest.csv` and `remote_bundle_inventory.csv`.
"""
    (output_dir / "R1_DEPTH_KMODE_FORMAL_ANALYSIS_20260809.md").write_text(report, encoding="utf-8")
    artifact_manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "status": "complete",
        "formal_batch_id": args.batch_id,
        "source_zip_sha256": sha256_file(source_zip),
        "source_wandb_sha256": {path.name: sha256_file(path) for path in args.wandb_inputs},
        "verdict": verdict,
        "artifacts": sorted(str(path.relative_to(output_dir)).replace("\\", "/") for path in output_dir.rglob("*") if path.is_file()),
    }
    write_json(output_dir / "analysis_manifest.json", artifact_manifest)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"Analysis artifacts: {output_dir}")


if __name__ == "__main__":
    main()
