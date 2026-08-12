#!/usr/bin/env python3
"""Independently audit and summarize a completed MECH-08 handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-07-28.1"
EXPECTED_WORKER_VERSION = "2026-07-27.2"
EVALUATION_STEPS = [0, 16, 32, 48, 64, 80, 96, 112, 128]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_frame(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy(dtype=float)).all())


def frames_close(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
    numeric: list[str],
    *,
    atol: float = 1e-12,
) -> bool:
    left = observed.sort_values(keys).reset_index(drop=True)
    right = expected.sort_values(keys).reset_index(drop=True)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    for column in left.columns:
        if column in numeric:
            if not np.allclose(
                left[column].astype(float),
                right[column].astype(float),
                rtol=0.0,
                atol=atol,
                equal_nan=False,
            ):
                return False
        elif left[column].astype(str).tolist() != right[column].astype(str).tolist():
            return False
    return True


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy()
    right_rank = pd.Series(right).rank(method="average").to_numpy()
    return pearson(left_rank, right_rank)


def hierarchical_bootstrap_ci(
    frame: pd.DataFrame,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    grouped = {
        str(origin): group["delta_left_minus_right"].to_numpy(dtype=float)
        for origin, group in frame.groupby("checkpoint_cell", sort=True)
    }
    origins = np.array(sorted(grouped))
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled_origins = rng.choice(origins, size=len(origins), replace=True)
        origin_means = []
        for origin in sampled_origins:
            values = grouped[str(origin)]
            origin_means.append(
                float(rng.choice(values, size=len(values), replace=True).mean())
            )
        draws[index] = float(np.mean(origin_means))
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    analysis = run / "analysis"
    top_status = read_json(run / "status.json")
    preflight = read_json(run / "preflight.json")
    inventory = read_json(run / "checkpoint_inventory.json")
    smoke_manifest = read_json(run / "smoke" / "smoke_manifest.json")
    analysis_manifest = read_json(analysis / "mech08_analysis_manifest.json")
    analysis_checks = read_json(analysis / "checks.json")

    exemplar = (
        run
        / "formal"
        / "early_muon"
        / "muon"
        / "replica_0"
    )
    contract_path = exemplar / "rollout_contract.json"
    prediction_path = exemplar / "mech07_prediction_reference.csv"
    contract = read_json(contract_path)
    contract_sha = sha256_file(contract_path)
    prediction_sha = sha256_file(prediction_path)

    origins = [str(row["cell"]) for row in contract["checkpoints"]]
    spec_by_origin = {
        str(row["cell"]): row for row in contract["checkpoints"]
    }
    algorithms = [str(value) for value in contract["formal"]["algorithms"]]
    replicas = [int(value) for value in contract["formal"]["data_replicas"]]
    expected_jobs = len(origins) * len(algorithms) * len(replicas)

    worker_checks: dict[str, bool] = {
        "expected_worker_grid": expected_jobs == 48,
    }
    evaluation_frames: list[pd.DataFrame] = []
    training_frames: list[pd.DataFrame] = []
    formal_contract_hashes: set[str] = set()
    formal_prediction_hashes: set[str] = set()
    worker_versions: set[str] = set()
    worker_count = 0

    for origin in origins:
        spec = spec_by_origin[origin]
        for algorithm in algorithms:
            for replica in replicas:
                directory = (
                    run
                    / "formal"
                    / origin
                    / algorithm
                    / f"replica_{replica}"
                )
                manifest = read_json(directory / "mech08_manifest.json")
                checks = read_json(directory / "checks.json")
                status = read_json(directory / "status.json")
                momentum = read_json(directory / "momentum_transfer_audit.json")
                preconditioner = read_json(
                    directory / "initial_preconditioner_audit.json"
                )
                invariance = read_json(directory / "checkpoint_invariance.json")
                local_contract_sha = sha256_file(directory / "rollout_contract.json")
                local_prediction_sha = sha256_file(
                    directory / "mech07_prediction_reference.csv"
                )

                label = f"{origin}/{algorithm}/replica_{replica}"
                local_checks = {
                    f"{label}:manifest_passed": manifest.get("passed") is True,
                    f"{label}:tier": manifest.get("analysis_tier") == "formal",
                    f"{label}:origin": manifest.get("checkpoint_cell") == origin,
                    f"{label}:algorithm": manifest.get("algorithm") == algorithm,
                    f"{label}:replica": int(manifest.get("data_replica", -1))
                    == replica,
                    f"{label}:worker_version": manifest.get("script_version")
                    == EXPECTED_WORKER_VERSION,
                    f"{label}:status": status.get("status") == "passed",
                    f"{label}:checks": bool(checks) and all(checks.values()),
                    f"{label}:momentum": momentum.get("passed") is True
                    and momentum.get("all_values_match_exactly") is True
                    and momentum.get("all_values_finite") is True,
                    f"{label}:preconditioner": preconditioner.get("passed") is True,
                    f"{label}:checkpoint_invariance": invariance.get(
                        "checkpoint_size_unchanged"
                    )
                    is True
                    and invariance.get("checkpoint_mtime_unchanged") is True,
                    f"{label}:contract_hash": local_contract_sha == contract_sha,
                    f"{label}:prediction_hash": local_prediction_sha
                    == prediction_sha,
                    f"{label}:manifest_contract_hash": manifest.get(
                        "contract_sha256"
                    )
                    == contract_sha,
                    f"{label}:checkpoint_hash": manifest.get("checkpoint_sha256")
                    == spec["expected_sha256"],
                }
                worker_checks.update(local_checks)
                formal_contract_hashes.add(local_contract_sha)
                formal_prediction_hashes.add(local_prediction_sha)
                worker_versions.add(str(manifest.get("script_version")))

                evaluation = pd.read_csv(directory / "evaluation.csv")
                training = pd.read_csv(directory / "training.csv")
                worker_checks[f"{label}:evaluation_rows"] = len(evaluation) == 9
                worker_checks[f"{label}:evaluation_steps"] = (
                    evaluation["optimizer_step"].astype(int).tolist()
                    == EVALUATION_STEPS
                )
                worker_checks[f"{label}:training_rows"] = len(training) == 128
                worker_checks[f"{label}:training_steps"] = (
                    training["optimizer_step"].astype(int).tolist()
                    == list(range(1, 129))
                )
                worker_checks[f"{label}:evaluation_finite"] = finite_frame(evaluation)
                worker_checks[f"{label}:training_finite"] = finite_frame(training)
                worker_checks[f"{label}:csv_identity"] = (
                    set(evaluation["checkpoint_cell"]) == {origin}
                    and set(evaluation["algorithm"]) == {algorithm}
                    and set(evaluation["data_replica"].astype(int)) == {replica}
                    and set(training["checkpoint_cell"]) == {origin}
                    and set(training["algorithm"]) == {algorithm}
                    and set(training["data_replica"].astype(int)) == {replica}
                )

                evaluation_frames.append(evaluation)
                training_frames.append(training)
                worker_count += 1

    evaluation_raw = pd.concat(evaluation_frames, ignore_index=True)
    training_raw = pd.concat(training_frames, ignore_index=True)
    evaluation_all = pd.read_csv(analysis / "evaluation_all.csv")
    run_summary_saved = pd.read_csv(analysis / "run_summary.csv")
    paired_saved = pd.read_csv(
        analysis / "paired_contrasts.csv",
        dtype={"optimizer_step": str},
    )
    prediction_alignment_saved = pd.read_csv(
        analysis / "prediction_alignment.csv",
        dtype={"optimizer_step": str},
    )

    evaluation_keys = [
        "checkpoint_cell",
        "algorithm",
        "data_replica",
        "optimizer_step",
    ]
    worker_checks["raw_evaluation_rows"] = len(evaluation_raw) == 432
    worker_checks["raw_training_rows"] = len(training_raw) == 6144
    worker_checks["evaluation_unique_grain"] = not evaluation_raw.duplicated(
        evaluation_keys
    ).any()
    worker_checks["training_unique_grain"] = not training_raw.duplicated(
        ["checkpoint_cell", "algorithm", "data_replica", "optimizer_step"]
    ).any()
    worker_checks["evaluation_all_matches_workers"] = frames_close(
        evaluation_all,
        evaluation_raw[evaluation_all.columns],
        evaluation_keys,
        [
            "heldout_loss",
            "normalized_loss",
            "loss_delta_from_step0",
            "relative_loss_delta_from_step0",
        ],
    )

    summary_rows: list[dict[str, Any]] = []
    for keys, frame in evaluation_raw.groupby(
        [
            "checkpoint_cell",
            "checkpoint_stage",
            "checkpoint_method",
            "algorithm",
            "data_replica",
        ],
        sort=True,
    ):
        ordered = frame.sort_values("optimizer_step")
        steps = ordered["optimizer_step"].to_numpy(dtype=float)
        normalized = ordered["normalized_loss"].to_numpy(dtype=float)
        auc = float(np.trapezoid(normalized, steps) / (steps[-1] - steps[0]))
        step0 = ordered.iloc[0]
        step128 = ordered.iloc[-1]
        summary_rows.append(
            {
                "checkpoint_cell": keys[0],
                "checkpoint_stage": keys[1],
                "checkpoint_method": keys[2],
                "algorithm": keys[3],
                "data_replica": int(keys[4]),
                "step0_loss": float(step0["heldout_loss"]),
                "step128_loss": float(step128["heldout_loss"]),
                "step128_normalized_loss": float(step128["normalized_loss"]),
                "step128_relative_loss_delta": float(
                    step128["relative_loss_delta_from_step0"]
                ),
                "normalized_loss_auc": auc,
                "normalized_loss_auc_delta_from_one": auc - 1.0,
            }
        )
    run_summary = pd.DataFrame(summary_rows)[run_summary_saved.columns]
    worker_checks["run_summary_recomputed"] = frames_close(
        run_summary_saved,
        run_summary,
        ["checkpoint_cell", "algorithm", "data_replica"],
        [
            "step0_loss",
            "step128_loss",
            "step128_normalized_loss",
            "step128_relative_loss_delta",
            "normalized_loss_auc",
            "normalized_loss_auc_delta_from_one",
        ],
    )

    step0_spread = (
        evaluation_raw[evaluation_raw["optimizer_step"] == 0]
        .groupby(["checkpoint_cell", "data_replica"])["heldout_loss"]
        .agg(lambda values: float(values.max() - values.min()))
    )
    worker_checks["matched_starts_recomputed"] = bool((step0_spread <= 1e-7).all())

    summary_index = run_summary.set_index(
        ["checkpoint_cell", "data_replica", "algorithm"]
    )
    evaluation_index = evaluation_raw.set_index(
        ["checkpoint_cell", "data_replica", "algorithm", "optimizer_step"]
    )
    contrasts = [
        {
            **row,
            "contrast_class": "primary",
        }
        for row in contract["comparison_contract"]["primary"]
    ] + [
        {
            **row,
            "contrast_class": "baseline",
        }
        for row in contract["comparison_contract"]["baseline"]
    ]
    paired_rows: list[dict[str, Any]] = []
    for origin in origins:
        spec = spec_by_origin[origin]
        for replica in replicas:
            for contrast in contrasts:
                left = str(contrast["left"])
                right = str(contrast["right"])
                left_summary = summary_index.loc[(origin, replica, left)]
                right_summary = summary_index.loc[(origin, replica, right)]
                paired_rows.append(
                    {
                        "checkpoint_cell": origin,
                        "checkpoint_stage": spec["stage"],
                        "checkpoint_method": spec["method"],
                        "data_replica": replica,
                        "contrast": contrast["name"],
                        "contrast_class": contrast["contrast_class"],
                        "left": left,
                        "right": right,
                        "metric": "normalized_loss_auc",
                        "optimizer_step": "AUC_0_128",
                        "left_value": float(left_summary["normalized_loss_auc"]),
                        "right_value": float(right_summary["normalized_loss_auc"]),
                        "delta_left_minus_right": float(
                            left_summary["normalized_loss_auc"]
                            - right_summary["normalized_loss_auc"]
                        ),
                    }
                )
                for step in EVALUATION_STEPS:
                    left_row = evaluation_index.loc[
                        (origin, replica, left, step)
                    ]
                    right_row = evaluation_index.loc[
                        (origin, replica, right, step)
                    ]
                    paired_rows.append(
                        {
                            "checkpoint_cell": origin,
                            "checkpoint_stage": spec["stage"],
                            "checkpoint_method": spec["method"],
                            "data_replica": replica,
                            "contrast": contrast["name"],
                            "contrast_class": contrast["contrast_class"],
                            "left": left,
                            "right": right,
                            "metric": "normalized_heldout_loss",
                            "optimizer_step": str(step),
                            "left_value": float(left_row["normalized_loss"]),
                            "right_value": float(right_row["normalized_loss"]),
                            "delta_left_minus_right": float(
                                left_row["normalized_loss"]
                                - right_row["normalized_loss"]
                            ),
                        }
                    )
    paired = pd.DataFrame(paired_rows)[paired_saved.columns]
    worker_checks["paired_contrasts_recomputed"] = frames_close(
        paired_saved,
        paired,
        [
            "checkpoint_cell",
            "data_replica",
            "contrast",
            "metric",
            "optimizer_step",
        ],
        ["left_value", "right_value", "delta_left_minus_right"],
    )

    rng = np.random.default_rng(20260728)
    endpoint_rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    for contrast in contrasts:
        for metric, step in (
            ("normalized_loss_auc", "AUC_0_128"),
            ("normalized_heldout_loss", "128"),
        ):
            subset = paired[
                (paired["contrast"] == contrast["name"])
                & (paired["metric"] == metric)
                & (paired["optimizer_step"].astype(str) == step)
            ].copy()
            origin_means = (
                subset.groupby(
                    [
                        "checkpoint_cell",
                        "checkpoint_stage",
                        "checkpoint_method",
                    ],
                    as_index=False,
                )["delta_left_minus_right"]
                .mean()
                .rename(columns={"delta_left_minus_right": "mean_delta"})
            )
            origin_means.insert(3, "contrast", contrast["name"])
            origin_means.insert(4, "contrast_class", contrast["contrast_class"])
            origin_means["metric"] = metric
            origin_means["optimizer_step"] = step
            origin_rows.extend(origin_means.to_dict(orient="records"))
            low, high = hierarchical_bootstrap_ci(
                subset,
                args.bootstrap_samples,
                rng,
            )
            endpoint_rows.append(
                {
                    "contrast": contrast["name"],
                    "contrast_class": contrast["contrast_class"],
                    "left": contrast["left"],
                    "right": contrast["right"],
                    "metric": metric,
                    "optimizer_step": step,
                    "paired_units": len(subset),
                    "origins": subset["checkpoint_cell"].nunique(),
                    "mean_delta": float(subset["delta_left_minus_right"].mean()),
                    "sd_across_paired_units": float(
                        subset["delta_left_minus_right"].std(ddof=1)
                    ),
                    "left_better_units": int(
                        (subset["delta_left_minus_right"] < 0).sum()
                    ),
                    "left_worse_units": int(
                        (subset["delta_left_minus_right"] > 0).sum()
                    ),
                    "left_better_origins": int((origin_means["mean_delta"] < 0).sum()),
                    "left_worse_origins": int((origin_means["mean_delta"] > 0).sum()),
                    "hierarchical_bootstrap_95_low": low,
                    "hierarchical_bootstrap_95_high": high,
                    "negative_delta_means": "left algorithm is better",
                }
            )
    endpoints = pd.DataFrame(endpoint_rows)
    endpoint_origins = pd.DataFrame(origin_rows)

    trajectory_rows: list[dict[str, Any]] = []
    for (contrast, step), subset in paired[
        paired["metric"] == "normalized_heldout_loss"
    ].groupby(["contrast", "optimizer_step"], sort=False):
        origin_means = subset.groupby("checkpoint_cell")[
            "delta_left_minus_right"
        ].mean()
        trajectory_rows.append(
            {
                "contrast": contrast,
                "optimizer_step": int(step),
                "mean_delta": float(subset["delta_left_minus_right"].mean()),
                "sd_across_paired_units": float(
                    subset["delta_left_minus_right"].std(ddof=1)
                ),
                "left_better_units": int(
                    (subset["delta_left_minus_right"] < 0).sum()
                ),
                "left_worse_units": int(
                    (subset["delta_left_minus_right"] > 0).sum()
                ),
                "left_better_origins": int((origin_means < 0).sum()),
                "left_worse_origins": int((origin_means > 0).sum()),
            }
        )
    trajectories = pd.DataFrame(trajectory_rows).sort_values(
        ["contrast", "optimizer_step"]
    )

    prediction_reference = pd.read_csv(prediction_path)
    prediction_index = prediction_reference.set_index(
        ["checkpoint_cell", "algorithm"]
    )["predicted_median_relative_loss_delta_at_multiplier_1"]
    bridge_rows: list[dict[str, Any]] = []
    for row in paired.to_dict(orient="records"):
        if row["metric"] == "normalized_heldout_loss" and str(
            row["optimizer_step"]
        ) not in {"16", "32", "64", "128"}:
            continue
        predicted = float(
            prediction_index.loc[(row["checkpoint_cell"], row["left"])]
            - prediction_index.loc[(row["checkpoint_cell"], row["right"])]
        )
        bridge_rows.append(
            {
                **row,
                "predicted_delta_left_minus_right": predicted,
            }
        )
    bridge_replica = pd.DataFrame(bridge_rows)
    bridge = (
        bridge_replica.groupby(
            [
                "checkpoint_cell",
                "checkpoint_stage",
                "checkpoint_method",
                "contrast",
                "contrast_class",
                "left",
                "right",
                "metric",
                "optimizer_step",
                "predicted_delta_left_minus_right",
            ],
            as_index=False,
        )["delta_left_minus_right"]
        .median()
        .rename(columns={"delta_left_minus_right": "realized_delta_left_minus_right"})
    )
    alignment_rows: list[dict[str, Any]] = []
    for scope in ("primary", "all"):
        for metric, step in (
            ("normalized_loss_auc", "AUC_0_128"),
            ("normalized_heldout_loss", "16"),
            ("normalized_heldout_loss", "32"),
            ("normalized_heldout_loss", "64"),
            ("normalized_heldout_loss", "128"),
        ):
            subset = bridge[
                (bridge["metric"] == metric)
                & (bridge["optimizer_step"].astype(str) == step)
                & (
                    True
                    if scope == "all"
                    else bridge["contrast_class"].eq("primary")
                )
            ]
            predicted = subset["predicted_delta_left_minus_right"].to_numpy(
                dtype=float
            )
            realized = subset["realized_delta_left_minus_right"].to_numpy(
                dtype=float
            )
            sign_match = (
                ((predicted < 0) & (realized < 0))
                | ((predicted > 0) & (realized > 0))
                | ((predicted == 0) & (realized == 0))
            )
            alignment_rows.append(
                {
                    "contrast_scope": scope,
                    "metric": metric,
                    "optimizer_step": step,
                    "origin_contrast_units": len(subset),
                    "pearson": pearson(predicted, realized),
                    "spearman": spearman(predicted, realized),
                    "sign_concordant_units": int(sign_match.sum()),
                    "sign_concordance": float(sign_match.mean()),
                    "descriptive_not_confirmatory": True,
                }
            )
    prediction_alignment = pd.DataFrame(alignment_rows)
    worker_checks["prediction_alignment_recomputed"] = frames_close(
        prediction_alignment_saved,
        prediction_alignment[prediction_alignment_saved.columns],
        ["contrast_scope", "metric", "optimizer_step"],
        [
            "pearson",
            "spearman",
            "sign_concordance",
        ],
    )

    commands = [
        json.loads(line)
        for line in (run / "commands.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    command_labels = [str(row["label"]) for row in commands]
    duplicate_commands = sorted(
        {
            label: command_labels.count(label)
            for label in set(command_labels)
            if command_labels.count(label) > 1
        }.items()
    )
    worker_checks.update(
        {
            "top_status_passed": top_status.get("status") == "passed",
            "preflight_passed": preflight.get("passed") is True
            and all(preflight.get("checks", {}).values()),
            "checkpoint_inventory_passed": inventory.get("passed") is True
            and len(inventory.get("cells", [])) == 4,
            "smoke_manifest_passed": smoke_manifest.get("passed") is True
            and int(smoke_manifest.get("completed_jobs", -1)) == 4,
            "analysis_manifest_passed": analysis_manifest.get("passed") is True
            and int(analysis_manifest.get("formal_jobs", -1)) == 48,
            "analysis_checks_passed": bool(analysis_checks)
            and all(analysis_checks.values()),
            "formal_worker_count": worker_count == 48,
            "single_contract_hash": formal_contract_hashes == {contract_sha},
            "single_prediction_hash": formal_prediction_hashes == {prediction_sha},
            "worker_versions_exact": worker_versions == {EXPECTED_WORKER_VERSION},
            "command_history_expected": len(commands) == 54
            and len(set(command_labels)) == 52
            and duplicate_commands
            == [
                ("smoke/early_muon/muon/replica_0", 2),
                ("smoke/early_muon/original_newton_muon/replica_0", 2),
            ],
        }
    )

    diag_vs_full = trajectories[
        trajectories["contrast"]
        == "selective_diag_vs_original_newton_muon"
    ].set_index("optimizer_step")
    none_vs_full = trajectories[
        trajectories["contrast"]
        == "selective_none_vs_original_newton_muon"
    ].set_index("optimizer_step")
    refresh_transition = {
        "first_recorded_refresh_step": 32,
        "diag_mean_delta_step32": float(diag_vs_full.loc[32, "mean_delta"]),
        "diag_mean_delta_step48": float(diag_vs_full.loc[48, "mean_delta"]),
        "diag_change_32_to_48": float(
            diag_vs_full.loc[48, "mean_delta"]
            - diag_vs_full.loc[32, "mean_delta"]
        ),
        "diag_left_better_units_at_48": int(
            diag_vs_full.loc[48, "left_better_units"]
        ),
        "none_mean_delta_step32": float(none_vs_full.loc[32, "mean_delta"]),
        "none_mean_delta_step48": float(none_vs_full.loc[48, "mean_delta"]),
        "none_change_32_to_48": float(
            none_vs_full.loc[48, "mean_delta"]
            - none_vs_full.loc[32, "mean_delta"]
        ),
        "none_left_better_units_at_48": int(
            none_vs_full.loc[48, "left_better_units"]
        ),
        "paired_units": 12,
    }

    endpoint_lookup = endpoints.set_index(["contrast", "metric"])
    alignment_lookup = prediction_alignment.set_index(
        ["contrast_scope", "metric", "optimizer_step"]
    )
    important_results = {
        "script_version": SCRIPT_VERSION,
        "run_id": run.name,
        "integrity": {
            "passed": all(worker_checks.values()),
            "formal_jobs": worker_count,
            "evaluation_rows": len(evaluation_raw),
            "training_rows": len(training_raw),
            "matched_start_units": len(step0_spread),
            "max_step0_spread": float(step0_spread.max()),
        },
        "primary_endpoint_step128": {
            contrast: float(
                endpoint_lookup.loc[
                    (contrast, "normalized_heldout_loss"), "mean_delta"
                ]
            )
            for contrast in [
                "selective_diag_vs_muon",
                "selective_none_vs_muon",
                "selective_diag_vs_original_newton_muon",
                "selective_none_vs_original_newton_muon",
                "original_newton_muon_vs_muon",
            ]
        },
        "primary_endpoint_auc": {
            contrast: float(
                endpoint_lookup.loc[
                    (contrast, "normalized_loss_auc"), "mean_delta"
                ]
            )
            for contrast in [
                "selective_diag_vs_muon",
                "selective_none_vs_muon",
                "selective_diag_vs_original_newton_muon",
                "selective_none_vs_original_newton_muon",
                "original_newton_muon_vs_muon",
            ]
        },
        "prediction_bridge_primary": {
            "auc": alignment_lookup.loc[
                ("primary", "normalized_loss_auc", "AUC_0_128")
            ].to_dict(),
            "step128": alignment_lookup.loc[
                ("primary", "normalized_heldout_loss", "128")
            ].to_dict(),
        },
        "refresh_transition": refresh_transition,
        "interpretation_contract": {
            "negative_delta_means": "left algorithm is better",
            "data_replicas_are_not_independent_training_seeds": True,
            "timing_usable_for_paper": False,
            "integrity_pass_is_not_hypothesis_success": True,
        },
    }

    archive_payload: dict[str, Any] | None = None
    if args.archive is not None:
        archive = args.archive.resolve()
        archive_payload = {
            "path": archive.name,
            "source": "user-provided local archive",
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        }
    source_inventory = {
        "script_version": SCRIPT_VERSION,
        "run_dir": str(run),
        "archive": archive_payload,
        "contract": {
            "path": str(contract_path),
            "sha256": contract_sha,
        },
        "prediction_reference": {
            "path": str(prediction_path),
            "sha256": prediction_sha,
        },
        "controller_command_rows": len(commands),
        "controller_unique_job_labels": len(set(command_labels)),
        "controller_duplicate_smoke_attempts": dict(duplicate_commands),
        "formal_workers": worker_count,
        "formal_worker_version": sorted(worker_versions),
    }
    quality_audit = {
        "script_version": SCRIPT_VERSION,
        "passed": all(worker_checks.values()),
        "checks": worker_checks,
        "failed_checks": [
            name for name, passed in worker_checks.items() if not passed
        ],
        "notes": [
            "The two duplicate smoke command labels are the documented "
            "2026-07-27.1 norm-audit false-negative attempts; the final smoke "
            "and all formal workers use 2026-07-27.2.",
            "Replica rows are matched data-order repetitions within four "
            "checkpoint origins, not independent training seeds.",
            "MECH-08 timing and peak allocation are excluded from paper "
            "efficiency claims by contract.",
        ],
    }

    endpoints.to_csv(output / "primary_endpoints_overall.csv", index=False)
    endpoint_origins.to_csv(output / "primary_endpoints_by_origin.csv", index=False)
    trajectories.to_csv(output / "trajectory_contrasts.csv", index=False)
    prediction_alignment.to_csv(
        output / "prediction_alignment_recomputed.csv",
        index=False,
    )
    bridge.to_csv(output / "prediction_bridge_recomputed.csv", index=False)
    write_json(output / "data_quality_audit.json", quality_audit)
    write_json(output / "important_results.json", important_results)
    write_json(output / "source_inventory.json", source_inventory)
    print(json.dumps(important_results, indent=2, ensure_ascii=False))
    if not quality_audit["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
