#!/usr/bin/env python3
"""Audit the R1 dense-full alpha confirmatory W&B exports.

The primary confirmatory contrast is frozen in CONFIRMATORY_CONTRACT_20260727.md:

    C_seed = L(alpha=0.5) - 0.5 * (L(alpha=0) + L(alpha=1))

The script treats timing as descriptive-only, retains hashes of every supplied
export, and keeps local-artifact verification separate from W&B curve quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-07-29.1"

METHOD_TO_ALPHA = {
    "fullalpha0": 0.0,
    "fullalpha0p25": 0.25,
    "fullalpha0p50": 0.50,
    "fullalpha0p75": 0.75,
    "fullalpha1": 1.0,
}
ALPHA_TO_METHOD = {value: key for key, value in METHOD_TO_ALPHA.items()}
EXPECTED_SEEDS = (2024, 2025)
EXPECTED_METRICS = {
    "val/loss": np.arange(0, 6201, 100, dtype=int),
    "train/loss_step": np.arange(20, 6201, 20, dtype=int),
    "time/train_s": np.arange(0, 6201, 20, dtype=int),
    "performance/step_avg_ms": np.arange(40, 6201, 20, dtype=int),
    "memory/k_state_mib": np.asarray([6200], dtype=int),
    "memory/optimizer_state_mib": np.asarray([6200], dtype=int),
    "memory/peak_allocated_mib": np.asarray([6200], dtype=int),
    "lr/adamw": np.arange(0, 6201, 20, dtype=int),
    "lr/matrix": np.arange(0, 6201, 20, dtype=int),
}
RUN_RE = re.compile(
    r"^(?P<run>mainconf_r1_dense_full_alpha_confirmatory_"
    r"(?P<method>fullalpha(?:0(?:p25|p50|p75)?|1))_"
    r"seed(?P<seed>2024|2025)_(?P<stamp>\d{8}T\d{6}\+0000))"
    r" - (?P<metric>.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports", type=Path, nargs="+", required=True)
    parser.add_argument("--seed2026-summary", type=Path, required=True)
    parser.add_argument("--seed2026-checks", type=Path, required=True)
    parser.add_argument("--block-alpha-curve", type=Path, required=True)
    parser.add_argument("--official-diag-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=json_value,
        )
        + "\n",
        encoding="utf-8",
    )


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def close(a: Any, b: Any, atol: float = 1e-12) -> bool:
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=0.0))


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    evidence: str,
    severity: str = "required",
) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "severity": severity,
            "evidence": evidence,
        }
    )


def discover_export(path: Path) -> tuple[str, pd.DataFrame, dict[str, dict[str, Any]]]:
    frame = pd.read_csv(path)
    if "Step" not in frame.columns:
        raise RuntimeError(f"Step column missing: {path}")
    base_columns = [
        column
        for column in frame.columns
        if column != "Step"
        and not column.endswith("__MIN")
        and not column.endswith("__MAX")
    ]
    if not base_columns:
        raise RuntimeError(f"no base run columns: {path}")

    identities: dict[str, dict[str, Any]] = {}
    metrics: set[str] = set()
    for column in base_columns:
        match = RUN_RE.fullmatch(column)
        if match is None:
            raise RuntimeError(f"unexpected W&B column: {column}")
        info = match.groupdict()
        info["seed"] = int(info["seed"])
        info["alpha"] = METHOD_TO_ALPHA[info["method"]]
        info["column"] = column
        identities[info["run"]] = info
        metrics.add(info["metric"])
    if len(metrics) != 1:
        raise RuntimeError(f"mixed metrics in {path}: {sorted(metrics)}")
    metric = next(iter(metrics))
    if metric not in EXPECTED_METRICS:
        raise RuntimeError(f"unexpected metric {metric!r}: {path}")

    numeric = frame.copy()
    numeric["Step"] = pd.to_numeric(numeric["Step"], errors="raise").astype(int)
    for column in numeric.columns[1:]:
        numeric[column] = pd.to_numeric(numeric[column], errors="raise")
    return metric, numeric, identities


def normalized_auc(steps: np.ndarray, values: np.ndarray) -> float:
    return float(np.trapezoid(values, steps) / float(steps[-1] - steps[0]))


def build_new_summary(
    metric_frames: dict[str, pd.DataFrame],
    identities: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    val = metric_frames["val/loss"]
    train = metric_frames["train/loss_step"]
    time_frame = metric_frames["time/train_s"]
    perf = metric_frames["performance/step_avg_ms"]
    adamw = metric_frames["lr/adamw"]
    matrix = metric_frames["lr/matrix"]
    peak = metric_frames["memory/peak_allocated_mib"]
    k_state = metric_frames["memory/k_state_mib"]
    optimizer = metric_frames["memory/optimizer_state_mib"]

    rows: list[dict[str, Any]] = []
    for run_name, info in sorted(
        identities.items(), key=lambda item: (item[1]["seed"], item[1]["alpha"])
    ):
        def column(metric: str) -> str:
            return f"{run_name} - {metric}"

        val_values = val[column("val/loss")].to_numpy(dtype=float)
        val_steps = val["Step"].to_numpy(dtype=int)
        best_index = int(np.argmin(val_values))
        rows.append(
            {
                "method": info["method"],
                "alpha": info["alpha"],
                "seed": info["seed"],
                "run_name": run_name,
                "run_stamp": info["stamp"],
                "initial_val_loss": val_values[0],
                "final_val_loss": val_values[-1],
                "best_val_loss": val_values[best_index],
                "best_val_step": int(val_steps[best_index]),
                "tail5_val_loss_mean": float(np.mean(val_values[-5:])),
                "normalized_val_auc": normalized_auc(val_steps, val_values),
                "final_train_loss_step": float(
                    train[column("train/loss_step")].iloc[-1]
                ),
                "train_time_s_descriptive_only": float(
                    time_frame[column("time/train_s")].iloc[-1]
                ),
                "final_step_avg_ms_descriptive_only": float(
                    perf[column("performance/step_avg_ms")].iloc[-1]
                ),
                "max_adamw_lr": float(adamw[column("lr/adamw")].max()),
                "max_matrix_lr": float(matrix[column("lr/matrix")].max()),
                "peak_memory_mib": float(
                    peak[column("memory/peak_allocated_mib")].iloc[-1]
                ),
                "k_state_mib": float(
                    k_state[column("memory/k_state_mib")].iloc[-1]
                ),
                "optimizer_state_mib": float(
                    optimizer[column("memory/optimizer_state_mib")].iloc[-1]
                ),
                "quality_eligible": True,
                "memory_eligible": True,
                "timing_eligible": False,
                "local_manifest_verified": False,
                "source": "W&B confirmation export 2026-07-29",
            }
        )
    return pd.DataFrame(rows)


def normalize_seed2026_summary(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    expected = set(METHOD_TO_ALPHA)
    if set(source["method"]) != expected or set(source["seed"].astype(int)) != {2026}:
        raise RuntimeError("seed2026 summary is not the expected five-cell pilot")
    rows: list[dict[str, Any]] = []
    for row in source.to_dict(orient="records"):
        rows.append(
            {
                "method": row["method"],
                "alpha": float(row["alpha"]),
                "seed": 2026,
                "run_name": row["run_name"],
                "run_stamp": re.search(
                    r"_(\d{8}T\d{6}\+0000)$", row["run_name"]
                ).group(1),
                "initial_val_loss": float(row["initial_val_loss"]),
                "final_val_loss": float(row["final_val_loss"]),
                "best_val_loss": float(row["best_val_loss"]),
                "best_val_step": int(row["best_val_step"]),
                "tail5_val_loss_mean": float(row["tail5_val_loss_mean"]),
                "normalized_val_auc": float(row["normalized_val_auc"]),
                "final_train_loss_step": float(row["final_train_loss_step"]),
                "train_time_s_descriptive_only": float(
                    row["train_time_s_descriptive_only"]
                ),
                "final_step_avg_ms_descriptive_only": float(
                    row["final_step_avg_ms_descriptive_only"]
                ),
                "max_adamw_lr": float(row["max_adamw_lr"]),
                "max_matrix_lr": float(row["max_matrix_lr"]),
                "peak_memory_mib": float(row["peak_memory_mib"]),
                "k_state_mib": float(row["k_state_mib"]),
                "optimizer_state_mib": float(row["optimizer_state_mib"]),
                "quality_eligible": True,
                "memory_eligible": bool_value(row["memory_eligible"]),
                "timing_eligible": bool_value(row["timing_eligible"]),
                "local_manifest_verified": False,
                "source": "Previously audited seed2026 W&B export",
            }
        )
    return pd.DataFrame(rows)


def build_history(
    metric_frames: dict[str, pd.DataFrame],
    identities: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, frame in sorted(metric_frames.items()):
        for run_name, info in identities.items():
            column = f"{run_name} - {metric}"
            for step, value in zip(frame["Step"], frame[column]):
                rows.append(
                    {
                        "metric": metric,
                        "step": int(step),
                        "seed": info["seed"],
                        "method": info["method"],
                        "alpha": info["alpha"],
                        "run_name": run_name,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def per_seed_curvature(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, group in summary.groupby("seed", sort=True):
        by_alpha = group.set_index("alpha")
        if set(by_alpha.index.astype(float)) != set(METHOD_TO_ALPHA.values()):
            raise RuntimeError(f"incomplete alpha grid for seed {seed}")
        final0 = float(by_alpha.loc[0.0, "final_val_loss"])
        final05 = float(by_alpha.loc[0.5, "final_val_loss"])
        final1 = float(by_alpha.loc[1.0, "final_val_loss"])
        tail0 = float(by_alpha.loc[0.0, "tail5_val_loss_mean"])
        tail05 = float(by_alpha.loc[0.5, "tail5_val_loss_mean"])
        tail1 = float(by_alpha.loc[1.0, "tail5_val_loss_mean"])
        auc0 = float(by_alpha.loc[0.0, "normalized_val_auc"])
        auc05 = float(by_alpha.loc[0.5, "normalized_val_auc"])
        auc1 = float(by_alpha.loc[1.0, "normalized_val_auc"])
        alphas = group["alpha"].to_numpy(dtype=float)
        final_values = group["final_val_loss"].to_numpy(dtype=float)
        alpha_ranks = pd.Series(alphas).rank(method="average").to_numpy()
        loss_ranks = pd.Series(final_values).rank(method="average").to_numpy()
        rows.append(
            {
                "seed": int(seed),
                "final_curvature_c": final05 - 0.5 * (final0 + final1),
                "tail5_curvature_c": tail05 - 0.5 * (tail0 + tail1),
                "auc_curvature_c": auc05 - 0.5 * (auc0 + auc1),
                "alpha0p50_minus_alpha0_final": final05 - final0,
                "alpha0p50_minus_alpha1_final": final05 - final1,
                "alpha1_minus_alpha0_final": final1 - final0,
                "alpha1_minus_alpha0_tail5": tail1 - tail0,
                "alpha1_minus_alpha0_auc": auc1 - auc0,
                "alpha0p50_beats_alpha0_final": final05 < final0,
                "alpha0p50_beats_alpha1_final": final05 < final1,
                "alpha0p50_beats_both_endpoints_final": final05
                < min(final0, final1),
                "best_alpha_final_descriptive": float(
                    group.loc[group["final_val_loss"].idxmin(), "alpha"]
                ),
                "spearman_alpha_vs_final_loss_descriptive": float(
                    np.corrcoef(alpha_ranks, loss_ranks)[0, 1]
                ),
            }
        )
    return pd.DataFrame(rows)


def alpha_aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_val_loss",
        "tail5_val_loss_mean",
        "normalized_val_auc",
        "peak_memory_mib",
        "k_state_mib",
        "optimizer_state_mib",
    ]
    rows: list[dict[str, Any]] = []
    for alpha, group in summary.groupby("alpha", sort=True):
        row: dict[str, Any] = {
            "method": ALPHA_TO_METHOD[float(alpha)],
            "alpha": float(alpha),
            "seeds": ",".join(str(value) for value in sorted(group["seed"])),
            "n_seeds": int(len(group)),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def dense_vs_diag(summary: pd.DataFrame, path: Path) -> pd.DataFrame:
    official = pd.read_csv(path)
    official = official.loc[
        (official["method"] == "diag")
        & (official["seed"].astype(int).isin([2024, 2025, 2026]))
    ].copy()
    dense = summary.loc[summary["alpha"] == 0.0].copy()
    merged = dense.merge(
        official,
        on="seed",
        suffixes=("_dense_alpha0", "_efficient_diag"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        rows.append(
            {
                "seed": int(row["seed"]),
                "dense_alpha0_run": row["run_name_dense_alpha0"],
                "efficient_diag_run": row["run_name_efficient_diag"],
                "dense_minus_diag_final": float(row["final_val_loss_dense_alpha0"])
                - float(row["final_val_loss_efficient_diag"]),
                "dense_minus_diag_tail5": float(
                    row["tail5_val_loss_mean_dense_alpha0"]
                )
                - float(row["tail5_val_loss_mean_efficient_diag"]),
                "dense_minus_diag_auc": float(
                    row["normalized_val_auc_dense_alpha0"]
                )
                - float(row["normalized_val_auc_efficient_diag"]),
                "absolute_final_delta_within_0p001": abs(
                    float(row["final_val_loss_dense_alpha0"])
                    - float(row["final_val_loss_efficient_diag"])
                )
                <= 0.001,
                "absolute_tail5_delta_within_0p001": abs(
                    float(row["tail5_val_loss_mean_dense_alpha0"])
                    - float(row["tail5_val_loss_mean_efficient_diag"])
                )
                <= 0.001,
            }
        )
    if len(rows) != 3:
        raise RuntimeError("official diag comparison did not produce three seeds")
    return pd.DataFrame(rows)


def topology_contrasts(summary: pd.DataFrame, path: Path) -> pd.DataFrame:
    block = pd.read_csv(path)
    eligible_alpha = {0.0, 0.25, 0.5, 0.75}
    block = block.loc[
        block["alpha"].astype(float).isin(eligible_alpha)
        & block["seed"].astype(int).isin([2024, 2025, 2026])
    ].copy()
    dense = summary.loc[summary["alpha"].isin(eligible_alpha)].copy()
    merged = dense.merge(
        block,
        on=["seed", "alpha"],
        suffixes=("_dense_full", "_block"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        final = float(row["final_val_loss_dense_full"]) - float(
            row["final_val_loss_block"]
        )
        tail = float(row["tail5_val_loss_mean_dense_full"]) - float(
            row["tail5_val_loss_mean_block"]
        )
        auc = float(row["normalized_val_auc_dense_full"]) - float(
            row["normalized_val_auc_block"]
        )
        nonzero = [value for value in (final, tail, auc) if value != 0.0]
        sign_concordant = bool(
            nonzero
            and all(math.copysign(1.0, value) == math.copysign(1.0, nonzero[0])
                    for value in nonzero)
        )
        rows.append(
            {
                "seed": int(row["seed"]),
                "alpha": float(row["alpha"]),
                "dense_full_method": row["method_dense_full"],
                "block_method": row["method_block"],
                "dense_full_minus_block_final": final,
                "dense_full_minus_block_tail5": tail,
                "dense_full_minus_block_auc": auc,
                "sign_concordant_across_final_tail5_auc": sign_concordant,
                "absolute_final_delta_ge_0p002": abs(final) >= 0.002,
                "material_topology_effect": abs(final) >= 0.002
                and sign_concordant,
            }
        )
    if len(rows) != 12:
        raise RuntimeError(
            f"matched topology grid incomplete: expected 12, observed {len(rows)}"
        )
    return pd.DataFrame(rows).sort_values(["seed", "alpha"])


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"output already exists: {args.output_dir}")
    if len(args.exports) != len(EXPECTED_METRICS):
        raise RuntimeError(
            f"expected {len(EXPECTED_METRICS)} exports, got {len(args.exports)}"
        )
    inputs = [
        *args.exports,
        args.seed2026_summary,
        args.seed2026_checks,
        args.block_alpha_curve,
        args.official_diag_summary,
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    checks: list[dict[str, Any]] = []
    metric_frames: dict[str, pd.DataFrame] = {}
    metric_sources: dict[str, Path] = {}
    identity_by_metric: dict[str, dict[str, dict[str, Any]]] = {}
    for path in args.exports:
        metric, frame, identities = discover_export(path)
        if metric in metric_frames:
            raise RuntimeError(f"duplicate metric export: {metric}")
        metric_frames[metric] = frame
        metric_sources[metric] = path
        identity_by_metric[metric] = identities

    add_check(
        checks,
        "expected_metric_exports",
        set(metric_frames) == set(EXPECTED_METRICS),
        f"observed={sorted(metric_frames)}",
    )
    reference_runs = set(identity_by_metric["val/loss"])
    expected_runs = len(EXPECTED_SEEDS) * len(METHOD_TO_ALPHA)
    add_check(
        checks,
        "expected_run_count",
        len(reference_runs) == expected_runs,
        f"observed={len(reference_runs)} expected={expected_runs}",
    )
    identities_consistent = all(
        set(identities) == reference_runs for identities in identity_by_metric.values()
    )
    add_check(
        checks,
        "run_identity_consistent_across_metrics",
        identities_consistent,
        f"metrics={len(identity_by_metric)} runs_per_metric={len(reference_runs)}",
    )
    identities = identity_by_metric["val/loss"]
    observed_cells = {
        (info["seed"], info["method"]) for info in identities.values()
    }
    required_cells = {
        (seed, method) for seed in EXPECTED_SEEDS for method in METHOD_TO_ALPHA
    }
    add_check(
        checks,
        "complete_seed_method_grid",
        observed_cells == required_cells,
        f"observed={sorted(observed_cells)}",
    )

    grid_failures: list[str] = []
    nonfinite: list[str] = []
    band_failures: list[str] = []
    for metric, expected_steps in EXPECTED_METRICS.items():
        frame = metric_frames[metric]
        observed_steps = frame["Step"].to_numpy(dtype=int)
        if not np.array_equal(observed_steps, expected_steps):
            grid_failures.append(metric)
        for run_name in reference_runs:
            base = f"{run_name} - {metric}"
            values = frame[base].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                nonfinite.append(base)
            for suffix in ("__MIN", "__MAX"):
                band = base + suffix
                if band not in frame.columns or not np.array_equal(
                    values, frame[band].to_numpy(dtype=float), equal_nan=True
                ):
                    band_failures.append(band)
    add_check(
        checks,
        "exact_step_grids",
        not grid_failures,
        f"failed={grid_failures or 'none'}",
    )
    add_check(
        checks,
        "all_values_finite",
        not nonfinite,
        f"failed={nonfinite or 'none'}",
    )
    add_check(
        checks,
        "wandb_min_max_bands_equal_base",
        not band_failures,
        f"failed_count={len(band_failures)}",
    )

    stamps_ok = True
    stamp_evidence: dict[int, list[str]] = {}
    for seed in EXPECTED_SEEDS:
        stamps = sorted(
            {info["stamp"] for info in identities.values() if info["seed"] == seed}
        )
        stamp_evidence[seed] = stamps
        stamps_ok &= len(stamps) == 1
    add_check(
        checks,
        "one_run_stamp_per_seed",
        stamps_ok,
        json.dumps(stamp_evidence, sort_keys=True),
    )

    initial_ok = True
    initial_evidence: dict[int, list[float]] = {}
    val = metric_frames["val/loss"]
    for seed in EXPECTED_SEEDS:
        values = sorted(
            {
                float(val[f"{run_name} - val/loss"].iloc[0])
                for run_name, info in identities.items()
                if info["seed"] == seed
            }
        )
        initial_evidence[seed] = values
        initial_ok &= len(values) == 1
    add_check(
        checks,
        "shared_initial_validation_within_seed",
        initial_ok,
        json.dumps(initial_evidence, sort_keys=True),
    )

    lr_failures: list[str] = []
    for metric in ("lr/adamw", "lr/matrix"):
        frame = metric_frames[metric]
        for seed in EXPECTED_SEEDS:
            run_names = sorted(
                run_name
                for run_name, info in identities.items()
                if info["seed"] == seed
            )
            reference = frame[f"{run_names[0]} - {metric}"].to_numpy(dtype=float)
            for run_name in run_names[1:]:
                observed = frame[f"{run_name} - {metric}"].to_numpy(dtype=float)
                if not np.allclose(reference, observed, atol=1e-15, rtol=0.0):
                    lr_failures.append(f"{seed}:{metric}:{run_name}")
    add_check(
        checks,
        "learning_rate_curves_identical_within_seed",
        not lr_failures,
        f"failed={lr_failures or 'none'}",
    )

    memory_expected = {
        "memory/k_state_mib": 1026.0,
        "memory/optimizer_state_mib": 1644.4746131896973,
        "memory/peak_allocated_mib": 40788.0,
    }
    memory_failures: list[str] = []
    for metric, expected in memory_expected.items():
        frame = metric_frames[metric]
        for run_name in reference_runs:
            observed = float(frame[f"{run_name} - {metric}"].iloc[-1])
            if not close(observed, expected):
                memory_failures.append(f"{metric}:{run_name}:{observed}")
    add_check(
        checks,
        "dense_full_memory_accounting",
        not memory_failures,
        f"expected={memory_expected} failures={memory_failures or 'none'}",
    )

    new_summary = build_new_summary(metric_frames, identities)
    seed2026 = normalize_seed2026_summary(args.seed2026_summary)
    summary = (
        pd.concat([new_summary, seed2026], ignore_index=True)
        .sort_values(["seed", "alpha"])
        .reset_index(drop=True)
    )
    add_check(
        checks,
        "canonical_three_seed_grid",
        len(summary) == 15
        and set(summary["seed"].astype(int)) == {2024, 2025, 2026}
        and summary.groupby("seed")["alpha"].nunique().eq(5).all(),
        f"rows={len(summary)} seeds={sorted(summary['seed'].unique())}",
    )

    previous_checks = pd.read_csv(args.seed2026_checks)
    previous_statuses = set(previous_checks["status"].astype(str))
    previous_wandb_ok = "FAIL" not in previous_statuses
    add_check(
        checks,
        "seed2026_previous_wandb_audit_usable",
        previous_wandb_ok,
        f"statuses={sorted(previous_statuses)}",
    )

    curvature = per_seed_curvature(summary)
    new_curvature = curvature.loc[curvature["seed"].isin(EXPECTED_SEEDS)]
    strict_confirmation = bool(
        new_curvature["alpha0p50_beats_both_endpoints_final"].all()
    )
    all_three_negative = bool((curvature["final_curvature_c"] < 0.0).all())
    add_check(
        checks,
        "frozen_primary_confirmation",
        strict_confirmation,
        "alpha=0.5 beats alpha=0 and alpha=1 at step6200 in both new seeds",
        severity="scientific_gate",
    )

    aggregate = alpha_aggregate(summary)
    diag = dense_vs_diag(summary, args.official_diag_summary)
    diag_gate = bool(
        diag["absolute_final_delta_within_0p001"].all()
        and diag["absolute_tail5_delta_within_0p001"].all()
    )
    add_check(
        checks,
        "dense_alpha0_matches_efficient_diag_gate",
        diag_gate,
        "absolute final and tail5 loss deltas <=0.001 in all three seeds",
        severity="scientific_gate",
    )
    topology = topology_contrasts(summary, args.block_alpha_curve)

    required_failures = [
        row["check"]
        for row in checks
        if row["severity"] in {"required", "scientific_gate"} and not row["passed"]
    ]
    wandb_audit_passed = not required_failures
    if not wandb_audit_passed:
        classification = "wandb_audit_or_scientific_gate_failed"
    elif strict_confirmation:
        classification = "strong_confirmatory_support"
    else:
        classification = "confirmatory_hypothesis_not_supported"

    output = args.output_dir
    raw_dir = output / "raw_wandb_exports"
    matched_dir = output / "matched_sources"
    raw_dir.mkdir(parents=True)
    matched_dir.mkdir()

    source_rows: list[dict[str, Any]] = []
    for metric, source in sorted(metric_sources.items()):
        destination = raw_dir / source.name
        shutil.copy2(source, destination)
        source_rows.append(
            {
                "role": f"wandb_export:{metric}",
                "original_path": str(source.resolve()),
                "retained_path": str(destination.relative_to(output)),
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
            }
        )
    matched_sources = {
        "seed2026_summary": args.seed2026_summary,
        "seed2026_checks": args.seed2026_checks,
        "block_alpha_curve": args.block_alpha_curve,
        "official_diag_summary": args.official_diag_summary,
    }
    for role, source in matched_sources.items():
        destination = matched_dir / f"{role}{source.suffix}"
        shutil.copy2(source, destination)
        source_rows.append(
            {
                "role": role,
                "original_path": str(source.resolve()),
                "retained_path": str(destination.relative_to(output)),
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
            }
        )

    history = build_history(metric_frames, identities)
    checks_frame = pd.DataFrame(checks)
    source_manifest = pd.DataFrame(source_rows)
    summary.to_csv(output / "dense_full_alpha_run_summary.csv", index=False)
    summary.to_csv(output / "canonical_alpha_curve.csv", index=False)
    history.to_csv(output / "confirmatory_history_long.csv", index=False)
    curvature.to_csv(output / "seed_curvature.csv", index=False)
    aggregate.to_csv(output / "alpha_aggregate.csv", index=False)
    diag.to_csv(output / "dense_alpha0_vs_efficient_diag.csv", index=False)
    topology.to_csv(output / "matched_topology_contrasts.csv", index=False)
    checks_frame.to_csv(output / "data_quality_checks.csv", index=False)
    source_manifest.to_csv(output / "source_manifest.csv", index=False)

    best_mean = aggregate.loc[
        aggregate["final_val_loss_mean"].idxmin()
    ].to_dict()
    material_topology = topology.loc[topology["material_topology_effect"]]
    important = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "classification": classification,
        "wandb_audit_passed": wandb_audit_passed,
        "delivery_status": "wandb_complete_local_artifacts_pending",
        "primary_test": {
            "definition": (
                "L(alpha=0.5)-0.5*(L(alpha=0)+L(alpha=1)) at validation step6200"
            ),
            "strict_new_seed_confirmation_passed": strict_confirmation,
            "all_three_seed_curvatures_negative": all_three_negative,
            "per_seed": curvature[
                [
                    "seed",
                    "final_curvature_c",
                    "alpha0p50_minus_alpha0_final",
                    "alpha0p50_minus_alpha1_final",
                    "alpha0p50_beats_both_endpoints_final",
                ]
            ].to_dict(orient="records"),
        },
        "descriptive_three_seed_aggregate": {
            "lowest_mean_final_alpha": float(best_mean["alpha"]),
            "lowest_mean_final_loss": float(best_mean["final_val_loss_mean"]),
            "warning": "Do not claim universal alpha=0.5 optimality.",
        },
        "dense_alpha0_vs_efficient_diag_gate_passed": diag_gate,
        "topology": {
            "eligible_cells": int(len(topology)),
            "material_effect_cells": int(len(material_topology)),
            "material_effect_records": material_topology.to_dict(orient="records"),
        },
        "memory": {
            "peak_allocated_mib_per_process": 40788.0,
            "k_state_mib": 1026.0,
            "optimizer_state_mib": 1644.4746131896973,
            "alpha_dependent_difference_observed": False,
        },
        "timing_eligible": False,
        "local_artifacts_verified": False,
        "local_artifacts_needed": (
            "run/smoke manifests and dense diagnostic refresh audits for the "
            "seed2024/2025 confirmatory batches"
        ),
        "failed_checks": required_failures,
    }
    write_json(output / "important_results.json", important)
    write_json(output / "data_quality_checks.json", checks)

    artifacts = sorted(
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file()
    )
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "passed": wandb_audit_passed,
        "classification": classification,
        "delivery_status": "wandb_complete_local_artifacts_pending",
        "input_export_count": len(args.exports),
        "new_run_count": len(new_summary),
        "canonical_run_count": len(summary),
        "failed_checks": required_failures,
        "artifacts": artifacts + ["audit_manifest.json"],
    }
    write_json(output / "audit_manifest.json", manifest)
    print(f"audit manifest: {output / 'audit_manifest.json'}")
    print(f"classification: {classification}")
    print(f"strict confirmation: {strict_confirmation}")
    print(f"local artifacts verified: False")
    if not wandb_audit_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
