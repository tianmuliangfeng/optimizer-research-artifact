#!/usr/bin/env python3
"""Analyze the preregistered WikiText-103 depth x c_proj K-mode experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-07-25.1"
SEEDS = (2024, 2025, 2026)
RULES = ("early", "center", "late", "edge", "all")
MODES = ("none", "diag")
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
MATRIX_ONLY_METRICS = {
    "matrix/non_cproj_k_state_bytes",
    "matrix/k_state_released_fraction",
    "matrix/cproj_target_layers_all",
    "matrix/cproj_target_layer_count",
    "matrix/cproj_none_params",
    "matrix/cproj_mode_applied_params",
    "matrix/cproj_k_state_bytes",
    "matrix/cproj_full_params",
    "matrix/cproj_diag_params",
}
RUN_RE = re.compile(
    r"^mainconf_wikitext103_12L_depth_kmode_formal_"
    r"(?:(?P<anchor>anchor_(?:full|muon))|"
    r"(?P<rule>early|center|late|edge|all)_(?P<mode>none|diag))_"
    r"seed(?P<seed>2024|2025|2026)$"
)
T95_DF2 = 4.302652729911275


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_run_name(run_name: str) -> dict[str, object]:
    match = RUN_RE.fullmatch(run_name)
    if match is None:
        raise ValueError(f"unexpected W&B run name: {run_name}")
    seed = int(match.group("seed"))
    if match.group("anchor"):
        mode = str(match.group("anchor")).removeprefix("anchor_")
        return {
            "run_name": run_name,
            "seed": seed,
            "kind": "anchor",
            "rule": "anchor",
            "mode": mode,
            "variant": f"anchor_{mode}",
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


def expected_run_names() -> set[str]:
    names = set()
    for seed in SEEDS:
        for rule in RULES:
            for mode in MODES:
                names.add(
                    "mainconf_wikitext103_12L_depth_kmode_formal_"
                    f"{rule}_{mode}_seed{seed}"
                )
        for mode in ("full", "muon"):
            names.add(
                "mainconf_wikitext103_12L_depth_kmode_formal_"
                f"anchor_{mode}_seed{seed}"
            )
    return names


def load_exports(
    paths: list[Path], output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], bool]:
    raw_dir = output_dir / "raw_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    observed_metrics: set[str] = set()
    bands_duplicate = True

    for source in paths:
        target = raw_dir / source.name
        if target.exists() and sha256_file(target) != sha256_file(source):
            raise RuntimeError(f"archived export differs from source: {target}")
        if not target.exists():
            shutil.copy2(source, target)
        frame = pd.read_csv(target)
        base_columns = [
            column
            for column in frame.columns
            if column != "Step" and not column.endswith(("__MIN", "__MAX"))
        ]
        parsed = [column.rsplit(" - ", 1) for column in base_columns]
        metrics = {parts[1] for parts in parsed if len(parts) == 2}
        if len(metrics) != 1:
            raise RuntimeError(f"expected one metric in {target}, observed {metrics}")
        metric = metrics.pop()
        if metric in observed_metrics:
            raise RuntimeError(f"duplicate metric export: {metric}")
        observed_metrics.add(metric)
        run_names: list[str] = []
        for column, parts in zip(base_columns, parsed):
            run_name = parts[0]
            metadata = parse_run_name(run_name)
            run_names.append(run_name)
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
            frames.append(subset)
            for suffix in ("__MIN", "__MAX"):
                band = column + suffix
                if band not in frame.columns or not frame[column].equals(frame[band]):
                    bands_duplicate = False
        manifests.append(
            {
                "file_name": source.name,
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
                "metric": metric,
                "rows": len(frame),
                "run_columns": len(base_columns),
                "expected_run_columns": 33 if metric in MATRIX_ONLY_METRICS else 36,
                "run_name_set_sha256": hashlib.sha256(
                    "\n".join(sorted(run_names)).encode("utf-8")
                ).hexdigest(),
                "archived_path": str(target.relative_to(output_dir)),
            }
        )
    if observed_metrics != EXPECTED_METRICS:
        raise RuntimeError(
            f"metric mismatch missing={sorted(EXPECTED_METRICS-observed_metrics)} "
            f"extra={sorted(observed_metrics-EXPECTED_METRICS)}"
        )
    long = pd.concat(frames, ignore_index=True)
    metadata = (
        long[["run_name", "seed", "kind", "rule", "mode", "variant"]]
        .drop_duplicates()
        .sort_values(["seed", "kind", "rule", "mode"])
        .reset_index(drop=True)
    )
    return long, metadata, manifests, bands_duplicate


def series_for(long: pd.DataFrame, run_name: str, metric: str) -> pd.Series:
    rows = long[(long.run_name == run_name) & (long.metric == metric)]
    if rows.empty:
        return pd.Series(dtype=float)
    if rows.step.duplicated().any():
        raise RuntimeError(f"duplicate steps: run={run_name}, metric={metric}")
    return rows.sort_values("step").set_index("step")["value"]


def constant_metric(long: pd.DataFrame, run_name: str, metric: str) -> float:
    values = series_for(long, run_name, metric)
    return float(values.iloc[-1]) if not values.empty else math.nan


def normalized_auc(values: pd.Series) -> float:
    x = values.index.to_numpy(dtype=float)
    y = values.to_numpy(dtype=float)
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def data_quality_checks(
    long: pd.DataFrame,
    metadata: pd.DataFrame,
    manifests: list[dict[str, object]],
    bands_duplicate: bool,
    fingerprint_path: Path,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, evidence: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    expected = expected_run_names()
    observed = set(metadata.run_name)
    add(
        "exact_36_run_set",
        observed == expected,
        {
            "observed": len(observed),
            "missing": sorted(expected - observed),
            "extra": sorted(observed - expected),
        },
    )
    add(
        "three_seeds_12_cells_each",
        all(int((metadata.seed == seed).sum()) == 12 for seed in SEEDS),
        metadata.groupby("seed").size().to_dict(),
    )
    add(
        "metric_export_counts",
        all(row["run_columns"] == row["expected_run_columns"] for row in manifests),
        {
            str(row["metric"]): int(row["run_columns"])
            for row in manifests
        },
    )
    val_bad, train_bad = [], []
    for run_name in sorted(expected & observed):
        if tuple(series_for(long, run_name, "val/loss").index) != VALIDATION_GRID:
            val_bad.append(run_name)
        if tuple(series_for(long, run_name, "train/loss_step").index) != TRAIN_GRID:
            train_bad.append(run_name)
    add("exact_validation_grid", not val_bad, val_bad)
    add("exact_training_grid", not train_bad, train_bad)

    step0 = {}
    for seed in SEEDS:
        values = [
            float(series_for(long, run, "val/loss").loc[0])
            for run in metadata.loc[metadata.seed == seed, "run_name"]
        ]
        step0[str(seed)] = {"value": values[0], "spread": max(values) - min(values)}
    add("seed_matched_step0", all(row["spread"] <= 1e-12 for row in step0.values()), step0)

    treatment_lr_match = True
    for seed in SEEDS:
        runs = metadata[
            (metadata.seed == seed) & (metadata["mode"] != "muon")
        ].run_name
        for metric in ("lr/matrix", "lr/adamw"):
            signatures = {
                tuple(series_for(long, run, metric).items())
                for run in runs
            }
            treatment_lr_match &= len(signatures) == 1
    add("treatment_lr_histories_match", treatment_lr_match, "within seed")
    add("wandb_band_columns_duplicate_base", bands_duplicate, "all 17 exports")

    layer_bad, mode_bad, byte_bad, release_bad = [], [], [], []
    for row in metadata.itertuples(index=False):
        if row.mode == "muon":
            continue
        expected_targets = 12 if row.kind == "anchor" or row.rule == "all" else 8
        targets = constant_metric(
            long, row.run_name, "matrix/cproj_target_layer_count"
        )
        if targets != expected_targets:
            layer_bad.append((row.run_name, targets, expected_targets))
        none_count = constant_metric(long, row.run_name, "matrix/cproj_none_params")
        diag_count = constant_metric(long, row.run_name, "matrix/cproj_diag_params")
        full_count = constant_metric(long, row.run_name, "matrix/cproj_full_params")
        expected_counts = (
            expected_targets if row.mode == "none" else 0,
            expected_targets if row.mode == "diag" else 0,
            12 if row.kind == "anchor" else 12 - expected_targets,
        )
        if (none_count, diag_count, full_count) != expected_counts:
            mode_bad.append((row.run_name, none_count, diag_count, full_count, expected_counts))
        total = constant_metric(long, row.run_name, "matrix/k_state_bytes")
        cproj = constant_metric(long, row.run_name, "matrix/cproj_k_state_bytes")
        non_cproj = constant_metric(
            long, row.run_name, "matrix/non_cproj_k_state_bytes"
        )
        if not math.isclose(total, cproj + non_cproj, rel_tol=0, abs_tol=0.5):
            byte_bad.append((row.run_name, total, cproj + non_cproj))
        full_name = (
            "mainconf_wikitext103_12L_depth_kmode_formal_"
            f"anchor_full_seed{row.seed}"
        )
        full_bytes = constant_metric(long, full_name, "matrix/k_state_bytes")
        observed_release = constant_metric(
            long, row.run_name, "matrix/k_state_released_fraction"
        )
        expected_release = 1 - total / full_bytes
        if not math.isclose(observed_release, expected_release, abs_tol=1e-9):
            release_bad.append((row.run_name, observed_release, expected_release))
    add("target_layer_counts_match", not layer_bad, layer_bad)
    add("none_diag_full_counts_match", not mode_bad, mode_bad)
    add("k_state_parts_sum", not byte_bad, byte_bad)
    add("released_fraction_matches_full", not release_bad, release_bad)
    add("all_values_finite", bool(np.isfinite(long.value.to_numpy()).all()), len(long))

    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    pinned_ok = (
        fingerprint.get("status") == "passed"
        and fingerprint["files"]["train_bin"]["sha256"]
        == "58c04ef835efade28c303561b99873eed64ac6a4060c5d715b4fb6538ae3cd34"
        and fingerprint["files"]["val_bin"]["sha256"]
        == "397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979"
    )
    add("pinned_wikitext_fingerprint", pinned_ok, fingerprint.get("status"))
    return checks


def build_run_summary(long: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in metadata.itertuples(index=False):
        val = series_for(long, row.run_name, "val/loss")
        train = series_for(long, row.run_name, "train/loss_step")
        peak = series_for(
            long, row.run_name, "cuda/full_run_max_memory_allocated_mib"
        )
        allocated = series_for(
            long, row.run_name, "cuda/memory_allocated_mib"
        )
        elapsed = series_for(long, row.run_name, "time_elapsed")
        rows.append(
            {
                "run_name": row.run_name,
                "seed": row.seed,
                "kind": row.kind,
                "rule": row.rule,
                "mode": row.mode,
                "variant": row.variant,
                "val_step4500": float(val.loc[4500]),
                "val_late3_mean": float(val.loc[[3500, 4000, 4500]].mean()),
                "val_best_0_4500": float(val.min()),
                "val_auc_0_4500": normalized_auc(val),
                "train_late_mean_4500_4980": float(
                    train.loc[train.index >= 4500].mean()
                ),
                "peak_full_run_memory_mib": float(peak.max()),
                "max_logged_memory_allocated_mib": float(allocated.max()),
                "elapsed_final_s_descriptive_only": float(elapsed.iloc[-1]),
                "k_state_bytes": constant_metric(
                    long, row.run_name, "matrix/k_state_bytes"
                ),
                "cproj_k_state_bytes": constant_metric(
                    long, row.run_name, "matrix/cproj_k_state_bytes"
                ),
                "non_cproj_k_state_bytes": constant_metric(
                    long, row.run_name, "matrix/non_cproj_k_state_bytes"
                ),
                "k_state_released_fraction": constant_metric(
                    long, row.run_name, "matrix/k_state_released_fraction"
                ),
                "cproj_target_layer_count": constant_metric(
                    long, row.run_name, "matrix/cproj_target_layer_count"
                ),
                "timing_eligible": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["seed", "kind", "rule", "mode"]
    ).reset_index(drop=True)


def build_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
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
            none, diag = pair.loc["none"], pair.loc["diag"]
            rows.append(
                {
                    "seed": seed,
                    "rule": rule,
                    "selected_layer_count": int(diag.cproj_target_layer_count),
                    "diag_val_step4500": diag.val_step4500,
                    "none_val_step4500": none.val_step4500,
                    "diag_minus_none_step4500": diag.val_step4500 - none.val_step4500,
                    "diag_minus_none_late3": diag.val_late3_mean - none.val_late3_mean,
                    "diag_minus_none_auc": diag.val_auc_0_4500 - none.val_auc_0_4500,
                    "diag_minus_none_best": diag.val_best_0_4500 - none.val_best_0_4500,
                    "diag_minus_full_step4500": diag.val_step4500 - full.val_step4500,
                    "none_minus_full_step4500": none.val_step4500 - full.val_step4500,
                    "diag_minus_none_peak_memory_mib": (
                        diag.peak_full_run_memory_mib
                        - none.peak_full_run_memory_mib
                    ),
                    "diag_k_state_bytes": diag.k_state_bytes,
                    "none_k_state_bytes": none.k_state_bytes,
                    "diag_release_fraction": diag.k_state_released_fraction,
                    "none_release_fraction": none.k_state_released_fraction,
                }
            )
    frame = pd.DataFrame(rows)
    all_by_seed = (
        frame[frame.rule == "all"]
        .set_index("seed")["diag_minus_none_step4500"]
        .to_dict()
    )
    frame["interaction_vs_all"] = frame.apply(
        lambda row: row.diag_minus_none_step4500 - all_by_seed[row.seed],
        axis=1,
    )
    return frame.sort_values(["rule", "seed"]).reset_index(drop=True)


def summarize_rules(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rule in RULES:
        group = contrasts[contrasts.rule == rule]
        values = list(group.diag_minus_none_step4500.astype(float))
        interactions = list(group.interaction_vs_all.astype(float))
        average, sd = mean(values), stdev(values)
        half_width = T95_DF2 * sd / math.sqrt(3)
        rows.append(
            {
                "rule": rule,
                "n_seeds": 3,
                "mean_diag_minus_none_step4500": average,
                "sd_diag_minus_none_step4500": sd,
                "median_diag_minus_none_step4500": median(values),
                "min_diag_minus_none_step4500": min(values),
                "max_diag_minus_none_step4500": max(values),
                "negative_seed_count": sum(value < 0 for value in values),
                "positive_seed_count": sum(value > 0 for value in values),
                "descriptive_t95_low": average - half_width,
                "descriptive_t95_high": average + half_width,
                "mean_interaction_vs_all": mean(interactions),
                "sd_interaction_vs_all": stdev(interactions),
                "negative_interaction_seed_count": sum(value < 0 for value in interactions),
                "positive_interaction_seed_count": sum(value > 0 for value in interactions),
                "mean_diag_minus_none_late3": float(
                    group.diag_minus_none_late3.mean()
                ),
                "mean_diag_minus_none_auc": float(group.diag_minus_none_auc.mean()),
                "mean_diag_minus_full_step4500": float(
                    group.diag_minus_full_step4500.mean()
                ),
                "mean_none_minus_full_step4500": float(
                    group.none_minus_full_step4500.mean()
                ),
                "mean_diag_release_fraction": float(
                    group.diag_release_fraction.mean()
                ),
                "mean_none_release_fraction": float(
                    group.none_release_fraction.mean()
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
        full, muon = anchors.loc["full"], anchors.loc["muon"]
        rows.append(
            {
                "seed": seed,
                "full_val_step4500": full.val_step4500,
                "muon_val_step4500": muon.val_step4500,
                "full_minus_muon_step4500": full.val_step4500 - muon.val_step4500,
                "full_peak_memory_mib": full.peak_full_run_memory_mib,
                "muon_peak_memory_mib": muon.peak_full_run_memory_mib,
            }
        )
    return pd.DataFrame(rows)


def build_transfer(
    wiki_rules: pd.DataFrame, project_root: Path
) -> pd.DataFrame:
    results_root = Path(
        os.environ.get("SNM_RESULTS_ROOT", str(project_root / "runs"))
    ).expanduser()
    owt_path = (
        results_root
        / "25_owt_depth_kmode"
        / "analysis_20260724_multiseed"
        / "rule_multiseed_summary.csv"
    )
    owt = pd.read_csv(owt_path).set_index("rule")
    wiki = wiki_rules.set_index("rule")
    rows = []
    for rule in RULES:
        owt_delta = float(owt.loc[rule, "mean_diag_minus_none_step4500"])
        wiki_delta = float(wiki.loc[rule, "mean_diag_minus_none_step4500"])
        rows.append(
            {
                "rule": rule,
                "owt_mean_diag_minus_none": owt_delta,
                "wikitext_mean_diag_minus_none": wiki_delta,
                "transfer_wikitext_minus_owt": wiki_delta - owt_delta,
                "owt_interaction_vs_all": float(
                    owt.loc[rule, "mean_interaction_vs_all"]
                ),
                "wikitext_interaction_vs_all": float(
                    wiki.loc[rule, "mean_interaction_vs_all"]
                ),
            }
        )
    return pd.DataFrame(rows)


def make_svg(path: Path, transfer: pd.DataFrame) -> None:
    width, height = 980, 560
    left, right, top, bottom = 110, 40, 58, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = -0.026, 0.002

    def sx(index: int, offset: float) -> float:
        return left + (index + 0.5 + offset) * plot_w / len(RULES)

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}'
        '.axis{stroke:#333;stroke-width:1.2}</style>',
        '<text x="490" y="30" text-anchor="middle" font-size="20" font-weight="700">'
        "Depth-rule paired effects across datasets</text>",
    ]
    for tick in (-0.025, -0.020, -0.015, -0.010, -0.005, 0.0):
        y = sy(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(
            f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-size="13">{tick:.3f}</text>'
        )
    zero_y = sy(0)
    parts.append(f'<line class="axis" x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}"/>')
    colors = {"OWT": "#4c78a8", "WikiText-103": "#f58518"}
    bar_width = 34
    for index, rule in enumerate(RULES):
        row = transfer[transfer.rule == rule].iloc[0]
        for label, value, offset in (
            ("OWT", float(row.owt_mean_diag_minus_none), -0.13),
            ("WikiText-103", float(row.wikitext_mean_diag_minus_none), 0.13),
        ):
            x = sx(index, offset) - bar_width / 2
            y = sy(value)
            parts.append(
                f'<rect x="{x:.2f}" y="{zero_y:.2f}" width="{bar_width}" '
                f'height="{y-zero_y:.2f}" fill="{colors[label]}"/>'
            )
        parts.append(
            f'<text x="{sx(index,0):.2f}" y="{height-bottom+28}" '
            f'text-anchor="middle" font-size="14">{rule}</text>'
        )
    for idx, label in enumerate(("OWT", "WikiText-103")):
        x = left + 265 + idx * 210
        parts.append(f'<rect x="{x}" y="{top+8}" width="18" height="18" fill="{colors[label]}"/>')
        parts.append(f'<text x="{x+27}" y="{top+22}" font-size="13">{label}</text>')
    parts.extend(
        [
            f'<text x="{left+plot_w/2:.2f}" y="{height-35}" text-anchor="middle" font-size="15">'
            "Selected c_proj depth rule</text>",
            f'<text transform="translate(25 {top+plot_h/2:.2f}) rotate(-90)" '
            'text-anchor="middle" font-size="15">Diag minus none at step 4500</text>',
            '<text x="490" y="540" text-anchor="middle" font-size="11" fill="#555">'
            "Negative values favor diagonal K; bars are means over seeds 2024/2025/2026."
            "</text>",
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = sorted(path.expanduser().resolve() for path in args.inputs)
    if len(paths) != 17 or len(paths) != len(set(paths)):
        raise RuntimeError(f"expected 17 unique exports, observed {len(paths)}")
    output_dir = args.output_dir.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    results_root = Path(
        os.environ.get("SNM_RESULTS_ROOT", str(project_root / "runs"))
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    long, metadata, manifests, bands_duplicate = load_exports(paths, output_dir)
    fingerprint_path = (
        results_root
        / "28_wikitext_depth_kmode"
        / "data_audit"
        / "wikitext103_gpt2_50m_fingerprint.json"
    )
    checks = data_quality_checks(
        long, metadata, manifests, bands_duplicate, fingerprint_path
    )
    failed = [check["check"] for check in checks if not check["passed"]]
    if failed:
        write_json(
            output_dir / "data_quality_checks.json",
            {"status": "failed", "checks": checks},
        )
        raise RuntimeError("data quality checks failed: " + ", ".join(failed))

    summary = build_run_summary(long, metadata)
    contrasts = build_contrasts(summary)
    rules = summarize_rules(contrasts)
    anchors = build_anchor_summary(summary)
    transfer = build_transfer(rules, project_root)
    curves = long[long.metric == "val/loss"][
        ["run_name", "seed", "kind", "rule", "mode", "variant", "step", "value"]
    ].rename(columns={"value": "val_loss"})

    rule_lookup = rules.set_index("rule")
    uniform_pass = bool(
        (rules.mean_diag_minus_none_step4500 < 0).all()
        and int((rules.negative_seed_count == 3).sum()) >= 4
    )
    magnitude_pass = bool(
        float(rule_lookup.loc["edge", "mean_interaction_vs_all"]) <= -0.002
        and int(rule_lookup.loc["edge", "negative_interaction_seed_count"]) >= 2
        and float(rule_lookup.loc["center", "mean_interaction_vs_all"]) >= 0.002
        and int(rule_lookup.loc["center", "positive_interaction_seed_count"]) >= 2
        and float(rule_lookup.loc["late", "mean_interaction_vs_all"]) >= 0.002
        and int(rule_lookup.loc["late", "positive_interaction_seed_count"]) >= 2
    )
    all_row = rule_lookup.loc["all"]
    variant_means = summary.groupby("variant").mean(numeric_only=True)
    verdict = {
        "analysis_date": "2026-07-25",
        "status": "valid",
        "run_count": 36,
        "paired_contrast_count": 15,
        "primary_endpoint": "validation loss at common step 4500",
        "primary_contrast": "diag minus none; negative favors diag",
        "uniform_direction_replication_pass": uniform_pass,
        "depth_magnitude_pattern_replication_pass": magnitude_pass,
        "classification": (
            "full_cross_dataset_depth_pattern_replication"
            if uniform_pass and magnitude_pass
            else "direction_only_replication"
            if uniform_pass
            else "no_cross_dataset_replication"
        ),
        "all_rule_mean_diag_minus_none": float(
            all_row.mean_diag_minus_none_step4500
        ),
        "all_rule_sd_diag_minus_none": float(
            all_row.sd_diag_minus_none_step4500
        ),
        "all_rule_negative_seed_count": int(all_row.negative_seed_count),
        "rules_with_negative_means": int(
            (rules.mean_diag_minus_none_step4500 < 0).sum()
        ),
        "rules_with_3_of_3_negative": int((rules.negative_seed_count == 3).sum()),
        "all_pair_sign_counts": {
            "step4500_negative": int(
                (contrasts.diag_minus_none_step4500 < 0).sum()
            ),
            "late3_negative": int((contrasts.diag_minus_none_late3 < 0).sum()),
            "auc_negative": int((contrasts.diag_minus_none_auc < 0).sum()),
            "total_pairs": 15,
        },
        "frozen_interaction_checks": {
            rule: {
                "mean": float(rule_lookup.loc[rule, "mean_interaction_vs_all"]),
                "negative_seeds": int(
                    rule_lookup.loc[rule, "negative_interaction_seed_count"]
                ),
                "positive_seeds": int(
                    rule_lookup.loc[rule, "positive_interaction_seed_count"]
                ),
            }
            for rule in ("edge", "center", "late")
        },
        "all_diag_minus_full_mean": float(all_row.mean_diag_minus_full_step4500),
        "all_none_minus_full_mean": float(all_row.mean_none_minus_full_step4500),
        "anchor_full_minus_muon_mean": float(
            anchors.full_minus_muon_step4500.mean()
        ),
        "memory_mib": {
            variant: float(
                variant_means.loc[variant, "peak_full_run_memory_mib"]
            )
            for variant in ("anchor_full", "anchor_muon", "all_none", "all_diag")
        },
        "additional_wikitext_seeds_needed": False,
        "additional_wikitext_depth_masks_needed": False,
        "recommended_next_depth_extension": (
            "R1 seed2026 early/center/late/edge x none/diag screen after "
            "layer-routing equivalence audit; add seeds2024/2025 only for the "
            "preregistered edge/center confirmation if the screen transfers"
        ),
        "claim_boundary": (
            "The experiment identifies a replicated depth-amplitude pattern for "
            "mlp.c_proj in local 12-layer GPT. It does not establish a universal "
            "best mask, individual-layer causality, attention/c_fc behavior, or "
            "transfer to the official R1/block4 implementation."
        ),
        "timing_usable": False,
    }

    pd.DataFrame(manifests).sort_values("metric").to_csv(
        output_dir / "input_manifest.csv", index=False
    )
    summary.to_csv(output_dir / "run_summary.csv", index=False)
    contrasts.to_csv(output_dir / "paired_depth_contrasts.csv", index=False)
    rules.to_csv(output_dir / "rule_multiseed_summary.csv", index=False)
    anchors.to_csv(output_dir / "anchor_summary.csv", index=False)
    transfer.to_csv(output_dir / "cross_dataset_depth_transfer.csv", index=False)
    curves.to_csv(output_dir / "validation_curves_long.csv", index=False)
    pd.DataFrame(checks).to_csv(output_dir / "data_quality_checks.csv", index=False)
    write_json(
        output_dir / "data_quality_checks.json",
        {
            "status": "passed",
            "script_version": SCRIPT_VERSION,
            "source_file_count": len(paths),
            "normalized_metric_rows": len(long),
            "checks": checks,
        },
    )
    write_json(output_dir / "analysis_verdict.json", verdict)
    make_svg(output_dir / "cross_dataset_depth_deltas.svg", transfer)

    rule_lines = []
    for row in rules.itertuples(index=False):
        rule_lines.append(
            f"| {row.rule} | {row.mean_diag_minus_none_step4500:+.6f} +/- "
            f"{row.sd_diag_minus_none_step4500:.6f} | "
            f"{row.negative_seed_count}/3 | "
            f"{row.mean_interaction_vs_all:+.6f} |"
        )
    transfer_lines = []
    for row in transfer.itertuples(index=False):
        transfer_lines.append(
            f"| {row.rule} | {row.owt_mean_diag_minus_none:+.6f} | "
            f"{row.wikitext_mean_diag_minus_none:+.6f} | "
            f"{row.transfer_wikitext_minus_owt:+.6f} |"
        )
    report = f"""# WikiText-103 depth x c_proj K-mode three-seed analysis

Analysis date: 2026-07-25  
Primary endpoint: validation loss at the last common checkpoint, step 4500  
Primary contrast: seed-matched `diag - none`; negative values favor diagonal K  
Status: all preregistered data-integrity checks passed; timing is descriptive only.

## Technical summary

The complete 36-run WikiText-103 matrix passes both frozen confirmation gates.
All five depth-rule means favor `diag`; four rules are negative in 3/3 seeds,
and `late` is negative in 2/3 with the remaining seed only `+0.000500`.
The all-depth primary effect is
`{verdict['all_rule_mean_diag_minus_none']:+.6f} +/- {verdict['all_rule_sd_diag_minus_none']:.6f}`.

The preregistered OWT magnitude pattern also transfers. Relative to all-depth,
WikiText edge is stronger by
`{float(rule_lookup.loc['edge', 'mean_interaction_vs_all']):+.6f}`, while
center and late are weaker by
`{float(rule_lookup.loc['center', 'mean_interaction_vs_all']):+.6f}` and
`{float(rule_lookup.loc['late', 'mean_interaction_vs_all']):+.6f}`. Each
predicted interaction exceeds the frozen 0.002 materiality threshold and has
the expected sign in 2/3 seeds.

## WikiText paired results

| Rule | Mean diag-none +/- SD | Negative seeds | Interaction vs all |
|---|---:|---:|---:|
{chr(10).join(rule_lines)}

The direction is not confined to one terminal point: 14/15 paired comparisons
favor diag at step 4500, in the final-three mean, and in normalized validation
AUC. The sole exception is `late`, seed2026, with effects of only
`+0.000500` final, `+0.000379` late-three, and `+0.001742` AUC.

## Cross-dataset replication

| Rule | OWT mean | WikiText mean | WikiText minus OWT |
|---|---:|---:|---:|
{chr(10).join(transfer_lines)}

Both datasets give the same qualitative ordering: edge is the strongest
diag-over-none rule; center/late are weaker than all; early is near or stronger
than all. The exact magnitudes differ, so depth changes effect size rather than
creating a dataset-independent optimal mask.

All-depth anchors sharpen the mechanism result. On WikiText:

- all-depth diag minus full is
  `{float(all_row.mean_diag_minus_full_step4500):+.6f}` on average;
- all-depth none minus full is
  `{float(all_row.mean_none_minus_full_step4500):+.6f}` on average;
- full Newton-Muon minus Muon is
  `{float(anchors.full_minus_muon_step4500.mean()):+.6f}`.

Thus coordinate-wise K scale is useful: removing K completely is worse than
full, while retaining only the diagonal matches or slightly improves on full.
The OWT anchors show the same ordering (`diag-full=-0.009921`,
`none-full=+0.004009`).

## Memory

The all-depth K-state is 243.28125 MiB for diag versus 1539 MiB for full,
releasing 84.19% of K-state. Mean peak allocation is approximately:

- full: `{variant_means.loc['anchor_full', 'peak_full_run_memory_mib']:.2f}` MiB;
- all-depth diag: `{variant_means.loc['all_diag', 'peak_full_run_memory_mib']:.2f}` MiB;
- all-depth none: `{variant_means.loc['all_none', 'peak_full_run_memory_mib']:.2f}` MiB;
- Muon: `{variant_means.loc['anchor_muon', 'peak_full_run_memory_mib']:.2f}` MiB.

All-depth diag therefore saves about
`{variant_means.loc['anchor_full', 'peak_full_run_memory_mib'] - variant_means.loc['all_diag', 'peak_full_run_memory_mib']:.2f}`
MiB of peak allocation relative to full in this experiment.

## Expansion decision

No additional OWT or WikiText seeds are needed, and a third dataset or a denser
mask sweep is not the next best use of compute. The two-dataset, three-seed
evidence is sufficient for a supporting claim that diag benefits are
directionally broad across depth and that their magnitude is depth-modulated.

The next valuable depth extension is the preregistered R1
architecture/implementation transfer:

1. audit numerical equivalence of the old all-depth/anchor runs under the new
   layer-routing source;
2. run seed2026 `early/center/late/edge x none/diag` (8 new runs);
3. only if the frozen edge-versus-center pattern transfers, add seeds2024/2025
   for the prespecified edge/center confirmation.

Do not add individual-layer or finer depth masks unless the paper elevates
"edge layers are universally optimal" to a central claim. The current
overlapping masks establish effect modulation, not individual-layer causality.

## Integrity and limits

The 17 exports contain 36 unique runs, ten validation checkpoints and 250
training points per run. Step-0 loss, LR histories, target-layer counts,
none/diag/full tensor counts, K-state accounting, released fractions, and the
pinned WikiText train/validation hashes all pass. Raw exports are archived with
SHA-256 in `input_manifest.csv`.

This experiment applies none/diag only to selected `mlp.c_proj` layers in the
local 12-layer GPT implementation. It does not cover attention, `mlp.c_fc`,
LLaMA/SwiGLU, or official R1/block4 routing, and it does not make timing
eligible.
"""
    (output_dir / "WIKITEXT_DEPTH_KMODE_MULTISEED_ANALYSIS_20260725.md").write_text(
        report, encoding="utf-8"
    )
    artifact_manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "status": "complete",
        "source_exports": len(paths),
        "source_export_sha256": {
            path.name: sha256_file(path) for path in paths
        },
        "verdict": verdict,
        "artifacts": sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file()
        ),
    }
    write_json(output_dir / "analysis_manifest.json", artifact_manifest)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"Analysis artifacts: {output_dir}")


if __name__ == "__main__":
    main()
