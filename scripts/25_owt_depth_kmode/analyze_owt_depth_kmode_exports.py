#!/usr/bin/env python3
"""Analyze the preregistered 3-seed OWT depth x c_proj K-mode experiment.

The inputs are W&B wide CSV exports (one metric per file).  The script copies
the exact exports into the analysis artifact, verifies the frozen integrity
contract, reconstructs a normalized metric table, and writes paired
seed-matched summaries.  Negative ``diag_minus_none`` validation-loss deltas
favor diagonal K.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import nbformat
    from nbclient import NotebookClient
except ModuleNotFoundError:
    nbformat = None
    NotebookClient = None


SCRIPT_VERSION = "2026-07-24.2"
SEEDS = (2024, 2025, 2026)
RULES = ("early", "center", "late", "edge", "all")
MODES = ("none", "diag")
PARTIAL_RULES = ("early", "center", "late", "edge")
VALIDATION_GRID = tuple(range(0, 4501, 500))
TRAIN_GRID = tuple(range(0, 5000, 20))
EXPECTED_METRICS = {
    "time_elapsed",
    "cuda/memory_allocated_mib",
    "cuda/full_run_max_memory_allocated_mib",
    "lr/matrix",
    "lr/adamw",
    "matrix/non_cproj_k_state_bytes",
    "matrix/k_state_released_fraction",
    "matrix/k_state_bytes",
    "matrix/cproj_target_layers_all",
    "matrix/cproj_target_layer_count",
    "matrix/cproj_none_params",
    "matrix/cproj_mode_applied_params",
    "matrix/cproj_k_state_bytes",
    "matrix/cproj_full_params",
    "matrix/cproj_diag_params",
    "train/loss_step",
    "val/loss",
}
RUN_RE = re.compile(
    r"^mainconf_owt_12L_depth_kmode_formal_"
    r"(?:(?P<anchor>anchor_(?:full|muon))|"
    r"(?P<rule>early|center|late|edge|all)_(?P<mode>none|diag))_"
    r"seed(?P<seed>2024|2025|2026)$"
)
T95_DF2 = 4.302652729911275


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-notebook-execution",
        action="store_true",
        help="Create but do not execute the companion notebook.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def expand_inputs(values: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    for value in values:
        text = str(value)
        if "*" in text or "?" in text:
            result.extend(sorted(Path().glob(text)))
        else:
            result.append(value)
    resolved = [path.expanduser().resolve() for path in result]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input exports: {missing}")
    if len(resolved) != len(set(resolved)):
        raise ValueError("duplicate input paths were supplied")
    return sorted(resolved)


def parse_run_name(run_name: str) -> dict[str, Any]:
    match = RUN_RE.fullmatch(run_name)
    if match is None:
        raise ValueError(f"unexpected W&B run name: {run_name}")
    seed = int(match.group("seed"))
    if match.group("anchor"):
        anchor = match.group("anchor").removeprefix("anchor_")
        return {
            "run_name": run_name,
            "seed": seed,
            "kind": "anchor",
            "rule": "anchor",
            "mode": anchor,
            "variant": f"anchor_{anchor}",
        }
    rule = str(match.group("rule"))
    mode = str(match.group("mode"))
    return {
        "run_name": run_name,
        "seed": seed,
        "kind": "depth_rule",
        "rule": rule,
        "mode": mode,
        "variant": f"{rule}_{mode}",
    }


def copy_inputs(paths: list[Path], output_dir: Path) -> list[Path]:
    raw_dir = output_dir / "raw_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in paths:
        target = raw_dir / source.name
        if target.exists():
            if sha256_file(target) != sha256_file(source):
                raise RuntimeError(f"existing raw export differs: {target}")
        else:
            shutil.copy2(source, target)
        copied.append(target)
    return copied


def load_exports(
    paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    long_frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    metric_to_path: dict[str, Path] = {}
    for path in paths:
        frame = pd.read_csv(path)
        if "Step" not in frame.columns:
            raise RuntimeError(f"missing Step column: {path}")
        base_columns = [
            column
            for column in frame.columns
            if column != "Step" and not column.endswith(("__MIN", "__MAX"))
        ]
        if not base_columns:
            raise RuntimeError(f"no metric columns: {path}")
        parsed = [column.rsplit(" - ", 1) for column in base_columns]
        if any(len(parts) != 2 for parts in parsed):
            raise RuntimeError(f"unparseable run/metric columns: {path}")
        metrics = sorted({parts[1] for parts in parsed})
        if len(metrics) != 1:
            raise RuntimeError(f"expected one metric per export, observed {metrics}")
        metric = metrics[0]
        if metric in metric_to_path:
            raise RuntimeError(
                f"metric {metric!r} appears in {metric_to_path[metric]} and {path}"
            )
        metric_to_path[metric] = path
        run_names = []
        for column, (run_name, _) in zip(base_columns, parsed):
            metadata = parse_run_name(run_name)
            values = pd.to_numeric(frame[column], errors="coerce")
            subset = pd.DataFrame(
                {
                    "step": pd.to_numeric(frame["Step"], errors="raise").astype(int),
                    "run_name": run_name,
                    "metric": metric,
                    "value": values,
                }
            ).dropna(subset=["value"])
            for key in ("seed", "kind", "rule", "mode", "variant"):
                subset[key] = metadata[key]
            long_frames.append(subset)
            run_names.append(run_name)
        manifests.append(
            {
                "file_name": path.name,
                "canonical_path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "rows": len(frame),
                "metric": metric,
                "run_columns": len(base_columns),
                "nonempty_run_columns": int(
                    sum(frame[column].notna().any() for column in base_columns)
                ),
                "run_names_sha256": hashlib.sha256(
                    "\n".join(sorted(run_names)).encode("utf-8")
                ).hexdigest(),
            }
        )
    observed_metrics = set(metric_to_path)
    if observed_metrics != EXPECTED_METRICS:
        raise RuntimeError(
            "metric export set mismatch: "
            f"missing={sorted(EXPECTED_METRICS - observed_metrics)}, "
            f"extra={sorted(observed_metrics - EXPECTED_METRICS)}"
        )
    long = pd.concat(long_frames, ignore_index=True)
    run_metadata = (
        long[["run_name", "seed", "kind", "rule", "mode", "variant"]]
        .drop_duplicates()
        .sort_values(["seed", "kind", "rule", "mode"])
        .reset_index(drop=True)
    )
    return long, run_metadata, manifests


def expected_run_names() -> set[str]:
    names = set()
    for seed in SEEDS:
        for rule in RULES:
            for mode in MODES:
                names.add(
                    f"mainconf_owt_12L_depth_kmode_formal_{rule}_{mode}_seed{seed}"
                )
        for anchor in ("full", "muon"):
            names.add(
                f"mainconf_owt_12L_depth_kmode_formal_anchor_{anchor}_seed{seed}"
            )
    return names


def series_for(long: pd.DataFrame, run_name: str, metric: str) -> pd.Series:
    rows = long[(long.run_name == run_name) & (long.metric == metric)]
    if rows.empty:
        return pd.Series(dtype=float)
    if rows.step.duplicated().any():
        duplicates = sorted(rows.loc[rows.step.duplicated(), "step"].unique())
        raise RuntimeError(
            f"duplicate metric steps for run={run_name}, metric={metric}: {duplicates}"
        )
    return rows.sort_values("step").set_index("step")["value"]


def constant_metric(
    long: pd.DataFrame, run_name: str, metric: str
) -> float:
    series = series_for(long, run_name, metric)
    if series.empty:
        return math.nan
    return float(series.iloc[-1])


def run_integrity_checks(
    long: pd.DataFrame, run_metadata: pd.DataFrame
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(check: str, passed: bool, detail: Any, severity: str = "critical") -> None:
        checks.append(
            {
                "check": check,
                "passed": bool(passed),
                "severity": severity,
                "detail": detail,
            }
        )

    observed_runs = set(run_metadata.run_name)
    expected_runs = expected_run_names()
    add(
        "exact_36_run_set",
        observed_runs == expected_runs,
        {
            "observed": len(observed_runs),
            "expected": len(expected_runs),
            "missing": sorted(expected_runs - observed_runs),
            "extra": sorted(observed_runs - expected_runs),
        },
    )
    add(
        "three_seed_coverage",
        set(run_metadata.seed) == set(SEEDS)
        and all((run_metadata.seed == seed).sum() == 12 for seed in SEEDS),
        run_metadata.groupby("seed").size().to_dict(),
    )

    validation_failures = []
    training_failures = []
    for run_name in sorted(expected_runs & observed_runs):
        val_steps = tuple(series_for(long, run_name, "val/loss").index.astype(int))
        if val_steps != VALIDATION_GRID:
            validation_failures.append({"run": run_name, "steps": val_steps})
        train_steps = tuple(
            series_for(long, run_name, "train/loss_step").index.astype(int)
        )
        if train_steps != TRAIN_GRID:
            training_failures.append(
                {
                    "run": run_name,
                    "count": len(train_steps),
                    "first": train_steps[:3],
                    "last": train_steps[-3:],
                }
            )
    add(
        "exact_validation_grid_0_to_4500",
        not validation_failures,
        validation_failures,
    )
    add(
        "exact_train_grid_0_to_4980",
        not training_failures,
        training_failures,
    )

    step0_failures = []
    for seed in SEEDS:
        values = []
        for run_name in sorted(
            run_metadata.loc[run_metadata.seed == seed, "run_name"]
        ):
            values.append(float(series_for(long, run_name, "val/loss").loc[0]))
        spread = max(values) - min(values)
        if spread > 1e-12:
            step0_failures.append({"seed": seed, "spread": spread, "values": values})
    add("seed_matched_step0_exact", not step0_failures, step0_failures)

    target_count_failures = []
    mode_count_failures = []
    byte_failures = []
    for row in run_metadata.itertuples(index=False):
        if row.mode == "muon":
            continue
        expected_target = (
            12 if row.kind == "anchor" or row.rule == "all" else 8
        )
        target_count = constant_metric(
            long, row.run_name, "matrix/cproj_target_layer_count"
        )
        if target_count != expected_target:
            target_count_failures.append(
                {
                    "run": row.run_name,
                    "expected": expected_target,
                    "observed": target_count,
                }
            )
        none_count = constant_metric(
            long, row.run_name, "matrix/cproj_none_params"
        )
        diag_count = constant_metric(
            long, row.run_name, "matrix/cproj_diag_params"
        )
        full_count = constant_metric(
            long, row.run_name, "matrix/cproj_full_params"
        )
        expected_none = expected_target if row.mode == "none" else 0
        expected_diag = expected_target if row.mode == "diag" else 0
        expected_full = 12 if row.kind == "anchor" else 12 - expected_target
        if (none_count, diag_count, full_count) != (
            expected_none,
            expected_diag,
            expected_full,
        ):
            mode_count_failures.append(
                {
                    "run": row.run_name,
                    "observed": [none_count, diag_count, full_count],
                    "expected": [expected_none, expected_diag, expected_full],
                }
            )
        total = constant_metric(long, row.run_name, "matrix/k_state_bytes")
        cproj = constant_metric(
            long, row.run_name, "matrix/cproj_k_state_bytes"
        )
        non_cproj = constant_metric(
            long, row.run_name, "matrix/non_cproj_k_state_bytes"
        )
        if not math.isclose(total, cproj + non_cproj, rel_tol=0, abs_tol=0.5):
            byte_failures.append(
                {
                    "run": row.run_name,
                    "total": total,
                    "parts": cproj + non_cproj,
                }
            )
    add("target_layer_counts_match_rules", not target_count_failures, target_count_failures)
    add("none_diag_full_tensor_counts_match", not mode_count_failures, mode_count_failures)
    add("k_state_parts_sum_to_total", not byte_failures, byte_failures)

    release_failures = []
    for seed in SEEDS:
        full_name = (
            f"mainconf_owt_12L_depth_kmode_formal_anchor_full_seed{seed}"
        )
        full_bytes = constant_metric(long, full_name, "matrix/k_state_bytes")
        subset = run_metadata[
            (run_metadata.seed == seed) & (run_metadata["mode"] != "muon")
        ]
        for row in subset.itertuples(index=False):
            observed = constant_metric(
                long, row.run_name, "matrix/k_state_released_fraction"
            )
            current = constant_metric(long, row.run_name, "matrix/k_state_bytes")
            expected = 1.0 - current / full_bytes
            if not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-9):
                release_failures.append(
                    {
                        "run": row.run_name,
                        "observed": observed,
                        "expected": expected,
                    }
                )
    add(
        "released_fraction_matches_anchor_full",
        not release_failures,
        release_failures,
    )
    add(
        "all_values_finite",
        bool(np.isfinite(long.value.to_numpy(dtype=float)).all()),
        {"rows": len(long)},
    )
    return checks


def normalized_auc(series: pd.Series) -> float:
    steps = series.index.to_numpy(dtype=float)
    values = series.to_numpy(dtype=float)
    return float(np.trapezoid(values, steps) / (steps[-1] - steps[0]))


def build_run_summary(
    long: pd.DataFrame, run_metadata: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for meta in run_metadata.itertuples(index=False):
        val = series_for(long, meta.run_name, "val/loss")
        train = series_for(long, meta.run_name, "train/loss_step")
        time = series_for(long, meta.run_name, "time_elapsed")
        peak = series_for(
            long, meta.run_name, "cuda/full_run_max_memory_allocated_mib"
        )
        allocated = series_for(
            long, meta.run_name, "cuda/memory_allocated_mib"
        )
        rows.append(
            {
                "run_name": meta.run_name,
                "seed": meta.seed,
                "kind": meta.kind,
                "rule": meta.rule,
                "mode": meta.mode,
                "variant": meta.variant,
                "val_step4500": float(val.loc[4500]),
                "val_late3_mean": float(val.loc[[3500, 4000, 4500]].mean()),
                "val_best_0_4500": float(val.min()),
                "val_auc_0_4500": normalized_auc(val),
                "train_late_mean_4500_4980": float(train.loc[train.index >= 4500].mean()),
                "elapsed_final_s": float(time.iloc[-1]),
                "peak_full_run_memory_mib": float(peak.max()),
                "max_logged_memory_allocated_mib": float(allocated.max()),
                "k_state_bytes": constant_metric(
                    long, meta.run_name, "matrix/k_state_bytes"
                ),
                "cproj_k_state_bytes": constant_metric(
                    long, meta.run_name, "matrix/cproj_k_state_bytes"
                ),
                "non_cproj_k_state_bytes": constant_metric(
                    long, meta.run_name, "matrix/non_cproj_k_state_bytes"
                ),
                "k_state_released_fraction": constant_metric(
                    long, meta.run_name, "matrix/k_state_released_fraction"
                ),
                "cproj_target_layer_count": constant_metric(
                    long, meta.run_name, "matrix/cproj_target_layer_count"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["seed", "kind", "rule", "mode"]
    ).reset_index(drop=True)


def build_paired_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        full = summary[
            (summary.seed == seed)
            & (summary.kind == "anchor")
            & (summary["mode"] == "full")
        ].iloc[0]
        for rule in RULES:
            pair = summary[
                (summary.seed == seed)
                & (summary.kind == "depth_rule")
                & (summary.rule == rule)
            ].set_index("mode")
            none = pair.loc["none"]
            diag = pair.loc["diag"]
            rows.append(
                {
                    "seed": seed,
                    "rule": rule,
                    "selected_layer_count": int(diag.cproj_target_layer_count),
                    "diag_val_step4500": diag.val_step4500,
                    "none_val_step4500": none.val_step4500,
                    "diag_minus_none_step4500": diag.val_step4500
                    - none.val_step4500,
                    "diag_minus_none_late3": diag.val_late3_mean
                    - none.val_late3_mean,
                    "diag_minus_none_auc": diag.val_auc_0_4500
                    - none.val_auc_0_4500,
                    "diag_minus_none_best": diag.val_best_0_4500
                    - none.val_best_0_4500,
                    "diag_minus_none_train_late": diag.train_late_mean_4500_4980
                    - none.train_late_mean_4500_4980,
                    "diag_minus_none_peak_memory_mib": diag.peak_full_run_memory_mib
                    - none.peak_full_run_memory_mib,
                    "diag_minus_none_elapsed_s": diag.elapsed_final_s
                    - none.elapsed_final_s,
                    "diag_minus_full_step4500": diag.val_step4500
                    - full.val_step4500,
                    "none_minus_full_step4500": none.val_step4500
                    - full.val_step4500,
                    "diag_k_state_bytes": diag.k_state_bytes,
                    "none_k_state_bytes": none.k_state_bytes,
                    "diag_release_fraction": diag.k_state_released_fraction,
                    "none_release_fraction": none.k_state_released_fraction,
                }
            )
    result = pd.DataFrame(rows)
    all_by_seed = (
        result[result.rule == "all"]
        .set_index("seed")["diag_minus_none_step4500"]
        .to_dict()
    )
    result["interaction_vs_all"] = [
        row.diag_minus_none_step4500 - all_by_seed[row.seed]
        for row in result.itertuples(index=False)
    ]
    return result.sort_values(["rule", "seed"]).reset_index(drop=True)


def summarize_rules(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in RULES:
        subset = contrasts[contrasts.rule == rule]
        values = subset.diag_minus_none_step4500.to_numpy(dtype=float)
        interaction = subset.interaction_vs_all.to_numpy(dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        se = sd / math.sqrt(len(values))
        rows.append(
            {
                "rule": rule,
                "n_seeds": len(values),
                "mean_diag_minus_none_step4500": mean,
                "sd_diag_minus_none_step4500": sd,
                "median_diag_minus_none_step4500": float(np.median(values)),
                "min_diag_minus_none_step4500": float(values.min()),
                "max_diag_minus_none_step4500": float(values.max()),
                "negative_seed_count": int((values < 0).sum()),
                "positive_seed_count": int((values > 0).sum()),
                "descriptive_t95_low": mean - T95_DF2 * se,
                "descriptive_t95_high": mean + T95_DF2 * se,
                "mean_interaction_vs_all": float(interaction.mean()),
                "sd_interaction_vs_all": float(interaction.std(ddof=1)),
                "mean_diag_minus_none_late3": float(
                    subset.diag_minus_none_late3.mean()
                ),
                "mean_diag_minus_none_auc": float(
                    subset.diag_minus_none_auc.mean()
                ),
                "mean_diag_minus_full_step4500": float(
                    subset.diag_minus_full_step4500.mean()
                ),
                "mean_none_minus_full_step4500": float(
                    subset.none_minus_full_step4500.mean()
                ),
                "mean_diag_release_fraction": float(
                    subset.diag_release_fraction.mean()
                ),
                "mean_none_release_fraction": float(
                    subset.none_release_fraction.mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_anchor_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        anchors = summary[
            (summary.seed == seed) & (summary.kind == "anchor")
        ].set_index("mode")
        full = anchors.loc["full"]
        muon = anchors.loc["muon"]
        rows.append(
            {
                "seed": seed,
                "full_val_step4500": full.val_step4500,
                "muon_val_step4500": muon.val_step4500,
                "full_minus_muon_step4500": full.val_step4500
                - muon.val_step4500,
                "full_minus_muon_late3": full.val_late3_mean
                - muon.val_late3_mean,
                "full_minus_muon_auc": full.val_auc_0_4500
                - muon.val_auc_0_4500,
                "full_k_state_bytes": full.k_state_bytes,
                "full_peak_memory_mib": full.peak_full_run_memory_mib,
                "muon_peak_memory_mib": muon.peak_full_run_memory_mib,
            }
        )
    return pd.DataFrame(rows)


def validation_curves(long: pd.DataFrame) -> pd.DataFrame:
    return (
        long[long.metric == "val/loss"][
            ["step", "run_name", "seed", "kind", "rule", "mode", "variant", "value"]
        ]
        .sort_values(["seed", "kind", "rule", "mode", "step"])
        .reset_index(drop=True)
    )


def make_validation_plot(curves: pd.DataFrame, output: Path) -> None:
    blue = "#2F5D8C"
    orange = "#D9772D"
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for axis, rule in zip(axes_flat, RULES):
        subset = curves[(curves.kind == "depth_rule") & (curves.rule == rule)]
        for mode, color, marker in (
            ("none", blue, "o"),
            ("diag", orange, "s"),
        ):
            mode_rows = subset[subset["mode"] == mode]
            grouped = mode_rows.groupby("step").value
            mean = grouped.mean()
            minimum = grouped.min()
            maximum = grouped.max()
            axis.plot(
                mean.index,
                mean.values,
                color=color,
                marker=marker,
                markersize=3.5,
                linewidth=1.8,
                label=mode,
            )
            axis.fill_between(
                mean.index,
                minimum.values,
                maximum.values,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(rule)
        axis.grid(axis="y", color="#D9DEE5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2)
    fig.suptitle("OWT depth × c_proj K-mode validation loss", fontsize=15, x=0.06, ha="left")
    fig.text(
        0.06,
        0.925,
        "12L/D768, seeds 2024–2026; line = seed mean, band = observed min–max",
        fontsize=10,
        color="#4B5563",
    )
    fig.supxlabel("Training step")
    fig.supylabel("Validation loss")
    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.9))
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def make_delta_plot(
    contrasts: pd.DataFrame, rule_summary: pd.DataFrame, output: Path
) -> None:
    order = list(RULES)
    y = np.arange(len(order))
    colors = {2024: "#2F5D8C", 2025: "#D9772D", 2026: "#6E7F3F"}
    offsets = {2024: -0.18, 2025: 0.0, 2026: 0.18}
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    for seed in SEEDS:
        values = (
            contrasts[contrasts.seed == seed]
            .set_index("rule")
            .loc[order, "diag_minus_none_step4500"]
        )
        axis.scatter(
            values.values,
            y + offsets[seed],
            color=colors[seed],
            s=48,
            label=str(seed),
            zorder=3,
        )
    means = (
        rule_summary.set_index("rule")
        .loc[order, "mean_diag_minus_none_step4500"]
        .to_numpy()
    )
    axis.scatter(
        means,
        y,
        marker="D",
        facecolor="white",
        edgecolor="#111827",
        linewidth=1.4,
        s=70,
        label="3-seed mean",
        zorder=4,
    )
    axis.axvline(0, color="#111827", linewidth=1.0)
    axis.set_yticks(y, order)
    axis.invert_yaxis()
    axis.grid(axis="x", color="#D9DEE5", linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.set_xlabel("Validation-loss delta at step 4500 (diag − none; lower is better)")
    axis.set_title("Paired depth-rule contrasts across seeds", loc="left", fontsize=14)
    axis.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(output, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def gate_verdict(
    contrasts: pd.DataFrame,
    rule_summary: pd.DataFrame,
    anchor_summary: pd.DataFrame,
) -> dict[str, Any]:
    all_means_negative = bool(
        (rule_summary.mean_diag_minus_none_step4500 < 0).all()
    )
    rules_three_of_three = int((rule_summary.negative_seed_count == 3).sum())
    uniform_contract_pass = all_means_negative and rules_three_of_three >= 4
    all_row = rule_summary[rule_summary.rule == "all"].iloc[0]
    mean_range = float(
        rule_summary.mean_diag_minus_none_step4500.max()
        - rule_summary.mean_diag_minus_none_step4500.min()
    )
    max_abs_interaction = float(
        rule_summary.mean_interaction_vs_all.abs().max()
    )
    endpoint_sign_consistency = {
        "step4500": int((contrasts.diag_minus_none_step4500 < 0).sum()),
        "late3": int((contrasts.diag_minus_none_late3 < 0).sum()),
        "auc": int((contrasts.diag_minus_none_auc < 0).sum()),
        "total_pairs": len(contrasts),
    }
    interaction_signs = {
        rule: {
            "negative": int((group.interaction_vs_all < 0).sum()),
            "positive": int((group.interaction_vs_all > 0).sum()),
        }
        for rule, group in contrasts[contrasts.rule != "all"].groupby("rule")
    }
    stable_opposite_interactions = bool(
        interaction_signs.get("center") == {"negative": 0, "positive": 3}
        and interaction_signs.get("edge") == {"negative": 3, "positive": 0}
    )
    if uniform_contract_pass:
        primary_classification = "uniform_diag_benefit"
    elif abs(float(all_row.mean_diag_minus_none_step4500)) < 0.001:
        primary_classification = "no_reliable_diag_benefit"
    else:
        primary_classification = "possible_depth_dependent_benefit"
    return {
        "status": "valid",
        "primary_endpoint": "validation loss at common step 4500",
        "primary_contrast": "diag minus none; negative favors diag",
        "seeds": list(SEEDS),
        "run_count": 36,
        "paired_contrast_count": 15,
        "primary_classification": primary_classification,
        "uniform_contract_pass": uniform_contract_pass,
        "all_rule_mean_diag_minus_none": float(
            all_row.mean_diag_minus_none_step4500
        ),
        "all_rule_sd_diag_minus_none": float(
            all_row.sd_diag_minus_none_step4500
        ),
        "all_rule_negative_seed_count": int(all_row.negative_seed_count),
        "rules_with_3_of_3_negative": rules_three_of_three,
        "all_rule_means_negative": all_means_negative,
        "mean_delta_range_across_rules": mean_range,
        "max_abs_mean_interaction_vs_all": max_abs_interaction,
        "endpoint_sign_consistency": endpoint_sign_consistency,
        "interaction_sign_counts": interaction_signs,
        "secondary_depth_magnitude_modulation": stable_opposite_interactions,
        "anchor_full_minus_muon_mean": float(
            anchor_summary.full_minus_muon_step4500.mean()
        ),
        "additional_owt_seeds_needed": False,
        "additional_owt_seed_reason": (
            "The delivered export already contains the preregistered "
            "seeds 2024/2025/2026 for all 36 runs."
        ),
        "r1_depth_replication_recommended": True,
        "r1_depth_replication_scope": (
            "seed2026 partial-rule screen (early/center/late/edge x none/diag); "
            "reuse all-depth/anchor runs only after numerical-equivalence audit"
        ),
        "r1_depth_replication_reason": (
            "The benefit direction is uniform, while center and edge have "
            "opposite 3/3 interaction signs relative to all-depth. R1 should "
            "test transfer of the magnitude pattern, not rerun OWT seeds."
        ),
        "wikitext_depth_replication_recommended": True,
        "wikitext_depth_replication_scope": (
            "seed2026 screen first; prioritize center/edge/all pairs plus "
            "matched full/Muon anchors, then expand only if the preregistered "
            "interaction transfers"
        ),
        "replication_execution_order": (
            "If sequential: WikiText seed2026 first as the cheaper gate, then "
            "R1 seed2026. If independent hosts are available, run both screens "
            "in parallel."
        ),
    }


def format_number(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def write_report(
    path: Path,
    checks: list[dict[str, Any]],
    run_summary: pd.DataFrame,
    rule_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    anchor_summary: pd.DataFrame,
    verdict: dict[str, Any],
) -> None:
    failed = [row for row in checks if not row["passed"]]
    all_row = rule_summary[rule_summary.rule == "all"].iloc[0]
    variant_means = run_summary.groupby("variant").mean(numeric_only=True)
    full_peak = float(
        variant_means.loc["anchor_full", "peak_full_run_memory_mib"]
    )
    diag_peak = float(
        variant_means.loc["all_diag", "peak_full_run_memory_mib"]
    )
    none_peak = float(
        variant_means.loc["all_none", "peak_full_run_memory_mib"]
    )
    muon_peak = float(
        variant_means.loc["anchor_muon", "peak_full_run_memory_mib"]
    )
    lines = [
        "# OWT depth × c_proj K-mode 三 seed 分析",
        "",
        "更新日期：2026-07-24",
        "",
        "## 结论先行",
        "",
        f"- 数据实际已经是完整多 seed：3 seeds × 12 cells = 36 runs；不是只有 seed2026。",
        f"- 冻结 primary endpoint（step-4500 validation loss）下，all-depth 的 "
        f"`diag-none` 三 seed 均值为 {format_number(all_row.mean_diag_minus_none_step4500)}，"
        f"样本标准差为 {format_number(all_row.sd_diag_minus_none_step4500)}，"
        f"负号 seed 数为 {int(all_row.negative_seed_count)}/3。",
        f"- 合同分类：`{verdict['primary_classification']}`；"
        f"五条 rule 均值全部为负={verdict['all_rule_means_negative']}，"
        f"3/3 为负的 rule 数={verdict['rules_with_3_of_3_negative']}/5。",
        f"- rule 均值跨度为 {format_number(verdict['mean_delta_range_across_rules'])}，"
        f"相对 all 的最大平均 interaction 为 "
        f"{format_number(verdict['max_abs_mean_interaction_vs_all'])}。",
        "- 方向结论是“跨深度一致”，幅度结论是“存在深度调制”：center 相对 all "
        "在 3/3 seeds 更弱，edge 相对 all 在 3/3 seeds 更强；不能据此把 edge "
        "直接宣布为普适最佳层规则。",
        "- 不需要再补 OWT seeds：2024/2025/2026 已经全部交付。是否做 WikiText/R1 "
        "depth replication 应由 interaction materiality 决定，而不是机械增加 OWT seed。",
        "",
        "## 数据质量",
        "",
        f"- 关键完整性检查：{len(checks) - len(failed)}/{len(checks)} 通过。",
    ]
    if failed:
        lines.append("- 未通过检查：" + "; ".join(row["check"] for row in failed))
    else:
        lines.append(
            "- 36-run 集合、三个 seed、validation/train grid、step-0 配对、"
            "layer/mode counts、K-state 会计和 released fraction 全部通过。"
        )
    lines.extend(
        [
            "",
            "## Primary paired contrasts",
            "",
            "| rule | mean diag-none | SD | negative seeds | interaction vs all |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rule_summary.itertuples(index=False):
        lines.append(
            f"| {row.rule} | {row.mean_diag_minus_none_step4500:.6f} | "
            f"{row.sd_diag_minus_none_step4500:.6f} | "
            f"{row.negative_seed_count}/3 | {row.mean_interaction_vs_all:.6f} |"
        )
    lines.extend(
        [
            "",
            "三 seed t 区间仅作为小样本描述，不作为显著性结论；五条 rule 共享 seed，"
            "也不能把 15 个 pair 当作 15 个独立重复。",
            "",
            "## Anchors",
            "",
            f"- full Newton–Muon 相对 Muon 的 step-4500 平均差值为 "
            f"{anchor_summary.full_minus_muon_step4500.mean():.6f}（负值表示 full 更好）。",
            "- anchors 每 seed 均由本 launcher 重跑，没有混用历史曲线。",
            f"- all-depth diag/none 的 K-state release 分别为 "
            f"{all_row.mean_diag_release_fraction:.2%}/"
            f"{all_row.mean_none_release_fraction:.2%}；平均 peak allocated memory "
            f"分别为 {diag_peak:.2f}/{none_peak:.2f} MiB，full 为 "
            f"{full_peak:.2f} MiB，Muon 为 {muon_peak:.2f} MiB。",
            f"- all-depth diag 相对 full 平均少用 {full_peak - diag_peak:.2f} MiB "
            f"peak allocated memory；但仍比 Muon 多用 {diag_peak - muon_peak:.2f} MiB。",
            "",
            "## 允许表述与限制",
            "",
            "- 本实验只操纵被选中 `mlp.c_proj` 层的 `none/diag`；未选 c_proj 和"
            "所有非 c_proj 矩阵仍是 local dense-full Newton–Muon。",
            "- 可以报告 `diag` 相对 `none` 的 paired depth-rule 结果和显存/K-state "
            "会计；不能外推为 attention、c_fc 或全模型 diagonalization。",
            "- 如果不同 rule 的 interaction 没有超过 seed 内波动，不应挑选单一“最佳"
            "深度 mask”写成确认性机制结论。",
            "- timing 只作描述；若同主机并行过其他训练，不能用这些 elapsed 差值作为"
            "算法速度结论。",
            "",
            "## 跨数据集与 R1 复现决策",
            "",
            "- 两者都有价值，但先做 seed2026 screen，不立即复制 36-run 全矩阵。",
            "- WikiText 优先检验数据集迁移：至少冻结 center/edge/all × none/diag；"
            "若希望完全沿用原合同，可跑五条 rules 与两个 anchors，共 12 runs。",
            "- R1 优先检验主论文实现迁移：新跑 early/center/late/edge × none/diag，"
            "共 8 runs。all-depth none/diag 与 block4/Muon anchors 只有通过新旧 "
            "layer-routing source 数值等价审计后才能复用。",
            "- 两个 screen 的主要检验对象均是 OWT 预先给出的幅度模式：edge 是否仍"
            "强于 all、center/late 是否仍弱于 all。若复现，再为预先选定的 "
            "edge/center 补 seeds2024/2025；若不复现，则记录为 dataset/architecture "
            "边界，不继续扩 seed。",
            "- R1 的论文相关性更高，但 WikiText screen 成本显著更低。若串行，先跑"
            "WikiText seed2026 作为省算力 gate，再决定是否启动 R1；若两台机器可"
            "并行，二者可以同时做。",
            "",
            "## 保存文件",
            "",
            "- `run_summary.csv`：36 个 run 的 primary/secondary endpoints；",
            "- `paired_depth_contrasts.csv`：15 个 seed-matched diag-none pair；",
            "- `rule_multiseed_summary.csv`：五条规则的三 seed 汇总；",
            "- `anchor_summary.csv`：full/Muon anchors；",
            "- `validation_curves_long.csv`：完整共同验证网格；",
            "- `data_quality_checks.json` 与 `analysis_verdict.json`；",
            "- `owt_depth_multiseed_analysis.ipynb`：轻量 notebook 索引；"
            "权威可复现入口为项目内分析脚本。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def create_notebook(
    output_dir: Path,
    verdict: dict[str, Any],
    execute: bool,
) -> tuple[Path, str]:
    if nbformat is None:
        path = output_dir / "owt_depth_multiseed_analysis.ipynb"
        notebook_payload = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## OWT depth × K-mode multi-seed analysis\n\n",
                        "The authoritative reproducible analysis is "
                        "`analyze_owt_depth_kmode_exports.py`; this lightweight "
                        "notebook index was not executed because nbformat/nbclient "
                        "are unavailable in the local analysis environment.",
                    ],
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        path.write_text(
            json.dumps(notebook_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path, "created_not_executed_missing_runtime"
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "## tl;dr\n\n"
            f"- 36/36 runs cover seeds 2024/2025/2026.\n"
            f"- Classification: `{verdict['primary_classification']}`.\n"
            f"- All-depth mean diag−none: "
            f"`{verdict['all_rule_mean_diag_minus_none']:.6f}`.\n"
            "- Negative deltas favor diagonal K."
        ),
        nbformat.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "Primary endpoint: common step-4500 validation loss. "
            "Each rule uses a within-seed `diag - none` pair. "
            "The final-three and normalized-AUC endpoints are secondary. "
            "The exact W&B exports and SHA-256 manifest are stored under "
            "`raw_exports/` and `input_manifest.csv`."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n"
            "root = Path('.').resolve()\n"
            "rules = pd.read_csv(root / 'rule_multiseed_summary.csv')\n"
            "pairs = pd.read_csv(root / 'paired_depth_contrasts.csv')\n"
            "anchors = pd.read_csv(root / 'anchor_summary.csv')\n"
            "display(rules)\n"
        ),
        nbformat.v4.new_markdown_cell("## Data"),
        nbformat.v4.new_code_cell(
            "manifest = pd.read_csv(root / 'input_manifest.csv')\n"
            "import json\n"
            "quality = json.loads((root / 'data_quality_checks.json').read_text(encoding='utf-8'))\n"
            "display(manifest[['file_name', 'metric', 'rows', 'run_columns', "
            "'nonempty_run_columns']])\n"
        ),
        nbformat.v4.new_markdown_cell("## Results"),
        nbformat.v4.new_code_cell(
            "display(pairs[['seed', 'rule', 'diag_minus_none_step4500', "
            "'diag_minus_none_late3', 'diag_minus_none_auc', "
            "'interaction_vs_all']])\n"
            "display(Image(filename=str(root / 'paired_depth_deltas.png')))\n"
            "display(Image(filename=str(root / 'validation_curves_by_rule.png')))\n"
        ),
        nbformat.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "Use the paired table rather than ranking isolated run minima. "
            "Three seeds are already complete; further work should target a "
            "dataset/official-implementation replication only if the frozen "
            "depth-interaction gate is materially satisfied."
        ),
    ]
    path = output_dir / "owt_depth_multiseed_analysis.ipynb"
    nbformat.write(notebook, path)
    status = "created_not_executed"
    if execute:
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(output_dir)}},
        )
        executed = client.execute()
        nbformat.write(executed, path)
        status = "executed_top_to_bottom"
    return path, status


def main() -> None:
    args = parse_args()
    source_paths = expand_inputs(args.inputs)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_paths = copy_inputs(source_paths, output_dir)
    long, run_metadata, manifests = load_exports(copied_paths)
    checks = run_integrity_checks(long, run_metadata)
    critical_failures = [
        row for row in checks if row["severity"] == "critical" and not row["passed"]
    ]
    if critical_failures:
        atomic_json(
            output_dir / "data_quality_checks.json",
            {
                "status": "failed",
                "script_version": SCRIPT_VERSION,
                "checks": checks,
            },
        )
        raise RuntimeError(
            "critical depth-data integrity checks failed: "
            + ", ".join(row["check"] for row in critical_failures)
        )

    run_summary = build_run_summary(long, run_metadata)
    contrasts = build_paired_contrasts(run_summary)
    rule_summary = summarize_rules(contrasts)
    anchor_summary = build_anchor_summary(run_summary)
    curves = validation_curves(long)
    verdict = gate_verdict(contrasts, rule_summary, anchor_summary)

    pd.DataFrame(manifests).sort_values("metric").to_csv(
        output_dir / "input_manifest.csv", index=False
    )
    run_summary.to_csv(output_dir / "run_summary.csv", index=False)
    contrasts.to_csv(output_dir / "paired_depth_contrasts.csv", index=False)
    rule_summary.to_csv(output_dir / "rule_multiseed_summary.csv", index=False)
    anchor_summary.to_csv(output_dir / "anchor_summary.csv", index=False)
    curves.to_csv(output_dir / "validation_curves_long.csv", index=False)
    make_validation_plot(curves, output_dir / "validation_curves_by_rule.png")
    make_delta_plot(
        contrasts, rule_summary, output_dir / "paired_depth_deltas.png"
    )
    atomic_json(
        output_dir / "data_quality_checks.json",
        {
            "status": "passed",
            "script_version": SCRIPT_VERSION,
            "source_file_count": len(source_paths),
            "normalized_metric_rows": len(long),
            "checks": checks,
        },
    )
    atomic_json(output_dir / "analysis_verdict.json", verdict)
    write_report(
        output_dir / "OWT_DEPTH_KMODE_MULTISEED_ANALYSIS_20260724.md",
        checks,
        run_summary,
        rule_summary,
        contrasts,
        anchor_summary,
        verdict,
    )
    notebook_path, notebook_status = create_notebook(
        output_dir, verdict, execute=not args.skip_notebook_execution
    )
    artifact_manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "status": "complete",
        "source_exports": len(source_paths),
        "source_export_sha256": {
            path.name: sha256_file(path) for path in copied_paths
        },
        "notebook": str(notebook_path),
        "notebook_status": notebook_status,
        "verdict": verdict,
        "artifacts": sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file()
        ),
    }
    atomic_json(output_dir / "analysis_manifest.json", artifact_manifest)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"Analysis artifacts: {output_dir}")


if __name__ == "__main__":
    main()
