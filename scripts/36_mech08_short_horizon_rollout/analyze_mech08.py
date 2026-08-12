#!/usr/bin/env python3
"""Aggregate MECH-08 paired rollout endpoints and prediction alignment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "2026-07-27.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--prediction-reference", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trapezoid_normalized_auc(rows: list[dict[str, Any]]) -> float:
    ordered = sorted(rows, key=lambda row: int(row["optimizer_step"]))
    if len(ordered) < 2:
        raise RuntimeError("AUC requires at least two evaluation points")
    area = 0.0
    for left, right in zip(ordered, ordered[1:]):
        width = int(right["optimizer_step"]) - int(left["optimizer_step"])
        if width <= 0:
            raise RuntimeError("evaluation steps are not strictly increasing")
        area += 0.5 * width * (
            float(left["normalized_loss"]) + float(right["normalized_loss"])
        )
    horizon = int(ordered[-1]["optimizer_step"]) - int(
        ordered[0]["optimizer_step"]
    )
    if horizon <= 0:
        raise RuntimeError("invalid AUC horizon")
    return area / horizon


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        raise RuntimeError("cannot summarize an empty vector")
    margin = 1e-9
    better = sum(value < -margin for value in values)
    worse = sum(value > margin for value in values)
    return {
        "paired_cells": len(values),
        "mean_delta": statistics.mean(values),
        "median_delta": statistics.median(values),
        "sd_delta": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_delta": min(values),
        "max_delta": max(values),
        "left_better_cells": better,
        "left_worse_cells": worse,
        "near_zero_cells": len(values) - better - worse,
        "left_better_fraction": better / len(values),
        "negative_delta_means": "left algorithm is better",
    }


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.mean(left)
    mean_right = statistics.mean(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0.0:
        return None
    return sum(
        value_left * value_right
        for value_left, value_right in zip(centered_left, centered_right)
    ) / denominator


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = 0.5 * ((index + 1) + end)
        for cursor in range(index, end):
            result[ordered[cursor][0]] = average
        index = end
    return result


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(ranks(left), ranks(right))


def median_by_key(
    rows: Iterable[dict[str, Any]],
    keys: tuple[str, ...],
    value: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(float(row[value]))
    return [
        dict(zip(keys, key))
        | {
            value: statistics.median(values),
            "replicas": len(values),
        }
        for key, values in sorted(groups.items())
    ]


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    contract = read_json(args.contract.resolve())
    expected_contract_sha = sha256_file(args.contract.resolve())
    output = run / "analysis"
    if output.exists():
        manifest_path = output / "mech08_analysis_manifest.json"
        if manifest_path.is_file():
            existing = read_json(manifest_path)
            if (
                existing.get("passed") is True
                and existing.get("script_version") == SCRIPT_VERSION
                and existing.get("contract_sha256") == expected_contract_sha
            ):
                print(f"MECH-08 analysis already passed: {manifest_path}")
                return
    else:
        output.mkdir()

    formal = contract["formal"]
    algorithms = list(formal["algorithms"])
    replicas = [int(value) for value in formal["data_replicas"]]
    evaluation_steps = [int(value) for value in formal["evaluation_steps"]]
    expected_prediction_sha = contract["prediction_reference"]["sha256"]
    if sha256_file(args.prediction_reference.resolve()) != expected_prediction_sha:
        raise RuntimeError("prediction reference hash mismatch")

    evaluation_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    integrity_errors: list[str] = []
    for spec in contract["checkpoints"]:
        cell = spec["cell"]
        for algorithm in algorithms:
            for replica in replicas:
                directory = (
                    run
                    / "formal"
                    / cell
                    / algorithm
                    / f"replica_{replica}"
                )
                manifest_path = directory / "mech08_manifest.json"
                checks_path = directory / "checks.json"
                if not manifest_path.is_file() or not checks_path.is_file():
                    integrity_errors.append(
                        f"missing formal artifacts: {cell}/{algorithm}/{replica}"
                    )
                    continue
                manifest = read_json(manifest_path)
                checks = read_json(checks_path)
                identity = {
                    "manifest_passed": manifest.get("passed") is True,
                    "checks_passed": bool(checks) and all(checks.values()),
                    "tier": manifest.get("analysis_tier") == "formal",
                    "cell": manifest.get("checkpoint_cell") == cell,
                    "algorithm": manifest.get("algorithm") == algorithm,
                    "replica": int(manifest.get("data_replica", -1)) == replica,
                    "contract": manifest.get("contract_sha256")
                    == expected_contract_sha,
                    "timing_excluded": manifest.get("timing_usable_for_paper")
                    is False,
                }
                if not all(identity.values()):
                    integrity_errors.append(
                        f"formal identity failed {cell}/{algorithm}/{replica}: "
                        f"{identity}"
                    )
                    continue
                rows = read_csv(directory / "evaluation.csv")
                if [int(row["optimizer_step"]) for row in rows] != evaluation_steps:
                    integrity_errors.append(
                        f"evaluation schedule failed {cell}/{algorithm}/{replica}"
                    )
                    continue
                typed = [
                    {
                        "checkpoint_cell": cell,
                        "checkpoint_stage": spec["stage"],
                        "checkpoint_method": spec["method"],
                        "algorithm": algorithm,
                        "data_replica": replica,
                        "optimizer_step": int(row["optimizer_step"]),
                        "heldout_loss": float(row["heldout_loss"]),
                        "normalized_loss": float(row["normalized_loss"]),
                        "loss_delta_from_step0": float(
                            row["loss_delta_from_step0"]
                        ),
                        "relative_loss_delta_from_step0": float(
                            row["relative_loss_delta_from_step0"]
                        ),
                    }
                    for row in rows
                ]
                evaluation_rows.extend(typed)
                by_step = {
                    int(row["optimizer_step"]): row for row in typed
                }
                auc = trapezoid_normalized_auc(typed)
                run_summaries.append(
                    {
                        "checkpoint_cell": cell,
                        "checkpoint_stage": spec["stage"],
                        "checkpoint_method": spec["method"],
                        "algorithm": algorithm,
                        "data_replica": replica,
                        "step0_loss": by_step[0]["heldout_loss"],
                        "step128_loss": by_step[128]["heldout_loss"],
                        "step128_normalized_loss": by_step[128][
                            "normalized_loss"
                        ],
                        "step128_relative_loss_delta": by_step[128][
                            "relative_loss_delta_from_step0"
                        ],
                        "normalized_loss_auc": auc,
                        "normalized_loss_auc_delta_from_one": auc - 1.0,
                    }
                )
                source_rows.append(
                    {
                        "checkpoint_cell": cell,
                        "algorithm": algorithm,
                        "data_replica": replica,
                        "manifest": str(manifest_path),
                        "manifest_sha256": sha256_file(manifest_path),
                        "evaluation_sha256": sha256_file(
                            directory / "evaluation.csv"
                        ),
                        "training_sha256": sha256_file(
                            directory / "training.csv"
                        ),
                    }
                )
    if integrity_errors:
        write_json(
            output / "integrity_audit.json",
            {"passed": False, "errors": integrity_errors},
        )
        raise RuntimeError(
            f"MECH-08 formal integrity failed with {len(integrity_errors)} errors"
        )

    expected_jobs = (
        len(contract["checkpoints"]) * len(algorithms) * len(replicas)
    )
    if len(run_summaries) != expected_jobs:
        raise RuntimeError(
            f"formal job count mismatch: {len(run_summaries)} != {expected_jobs}"
        )
    write_json(
        output / "integrity_audit.json",
        {
            "passed": True,
            "expected_formal_jobs": expected_jobs,
            "observed_formal_jobs": len(run_summaries),
            "errors": [],
        },
    )
    starts: dict[tuple[str, int], list[float]] = {}
    for row in run_summaries:
        starts.setdefault(
            (row["checkpoint_cell"], row["data_replica"]), []
        ).append(row["step0_loss"])
    start_audit = []
    for (cell, replica), values in sorted(starts.items()):
        spread = max(values) - min(values)
        start_audit.append(
            {
                "checkpoint_cell": cell,
                "data_replica": replica,
                "algorithms": len(values),
                "min_step0_loss": min(values),
                "max_step0_loss": max(values),
                "absolute_spread": spread,
                "matched": spread <= 1e-7,
            }
        )
    if not all(row["matched"] for row in start_audit):
        raise RuntimeError("matched-start held-out losses diverged across algorithms")

    summary_index = {
        (row["checkpoint_cell"], row["data_replica"], row["algorithm"]): row
        for row in run_summaries
    }
    evaluation_index = {
        (
            row["checkpoint_cell"],
            row["data_replica"],
            row["algorithm"],
            row["optimizer_step"],
        ): row
        for row in evaluation_rows
    }
    contrast_specs = [
        *(row | {"contrast_class": "primary"}
          for row in contract["comparison_contract"]["primary"]),
        *(row | {"contrast_class": "baseline"}
          for row in contract["comparison_contract"]["baseline"]),
    ]
    paired_rows: list[dict[str, Any]] = []
    for spec in contract["checkpoints"]:
        cell = spec["cell"]
        for replica in replicas:
            for contrast in contrast_specs:
                left = contrast["left"]
                right = contrast["right"]
                left_summary = summary_index[(cell, replica, left)]
                right_summary = summary_index[(cell, replica, right)]
                paired_rows.append(
                    {
                        "checkpoint_cell": cell,
                        "checkpoint_stage": spec["stage"],
                        "checkpoint_method": spec["method"],
                        "data_replica": replica,
                        "contrast": contrast["name"],
                        "contrast_class": contrast["contrast_class"],
                        "left": left,
                        "right": right,
                        "metric": "normalized_loss_auc",
                        "optimizer_step": "AUC_0_128",
                        "left_value": left_summary["normalized_loss_auc"],
                        "right_value": right_summary["normalized_loss_auc"],
                        "delta_left_minus_right": left_summary[
                            "normalized_loss_auc"
                        ]
                        - right_summary["normalized_loss_auc"],
                    }
                )
                for step in evaluation_steps:
                    left_row = evaluation_index[(cell, replica, left, step)]
                    right_row = evaluation_index[(cell, replica, right, step)]
                    paired_rows.append(
                        {
                            "checkpoint_cell": cell,
                            "checkpoint_stage": spec["stage"],
                            "checkpoint_method": spec["method"],
                            "data_replica": replica,
                            "contrast": contrast["name"],
                            "contrast_class": contrast["contrast_class"],
                            "left": left,
                            "right": right,
                            "metric": "normalized_heldout_loss",
                            "optimizer_step": step,
                            "left_value": left_row["normalized_loss"],
                            "right_value": right_row["normalized_loss"],
                            "delta_left_minus_right": left_row["normalized_loss"]
                            - right_row["normalized_loss"],
                        }
                    )

    aggregate_rows = []
    for stage in ("early", "late", "all"):
        for contrast in contrast_specs:
            for metric, step in (
                ("normalized_loss_auc", "AUC_0_128"),
                ("normalized_heldout_loss", 16),
                ("normalized_heldout_loss", 32),
                ("normalized_heldout_loss", 64),
                ("normalized_heldout_loss", 128),
            ):
                values = [
                    float(row["delta_left_minus_right"])
                    for row in paired_rows
                    if row["contrast"] == contrast["name"]
                    and row["metric"] == metric
                    and str(row["optimizer_step"]) == str(step)
                    and (
                        stage == "all" or row["checkpoint_stage"] == stage
                    )
                ]
                aggregate_rows.append(
                    {
                        "checkpoint_stage": stage,
                        "contrast": contrast["name"],
                        "contrast_class": contrast["contrast_class"],
                        "left": contrast["left"],
                        "right": contrast["right"],
                        "metric": metric,
                        "optimizer_step": step,
                    }
                    | summarize(values)
                )

    reference_rows = read_csv(args.prediction_reference.resolve())
    prediction_index = {
        (row["checkpoint_cell"], row["algorithm"]): float(
            row["predicted_median_relative_loss_delta_at_multiplier_1"]
        )
        for row in reference_rows
    }
    bridge_replica_rows: list[dict[str, Any]] = []
    for row in paired_rows:
        if row["metric"] == "normalized_heldout_loss" and int(
            row["optimizer_step"]
        ) not in (16, 32, 64, 128):
            continue
        if row["metric"] == "normalized_loss_auc" or row[
            "metric"
        ] == "normalized_heldout_loss":
            predicted = prediction_index[
                (row["checkpoint_cell"], row["left"])
            ] - prediction_index[(row["checkpoint_cell"], row["right"])]
            bridge_replica_rows.append(
                {
                    "checkpoint_cell": row["checkpoint_cell"],
                    "checkpoint_stage": row["checkpoint_stage"],
                    "checkpoint_method": row["checkpoint_method"],
                    "data_replica": row["data_replica"],
                    "contrast": row["contrast"],
                    "contrast_class": row["contrast_class"],
                    "left": row["left"],
                    "right": row["right"],
                    "metric": row["metric"],
                    "optimizer_step": row["optimizer_step"],
                    "predicted_delta_left_minus_right": predicted,
                    "prediction_step_multiplier": 1.0,
                    "realized_delta_left_minus_right": row[
                        "delta_left_minus_right"
                    ],
                }
            )
    bridge_rows = median_by_key(
        bridge_replica_rows,
        (
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
            "prediction_step_multiplier",
        ),
        "realized_delta_left_minus_right",
    )
    alignment_rows = []
    for contrast_class in ("primary", "all"):
        for metric, step in (
            ("normalized_loss_auc", "AUC_0_128"),
            ("normalized_heldout_loss", 16),
            ("normalized_heldout_loss", 32),
            ("normalized_heldout_loss", 64),
            ("normalized_heldout_loss", 128),
        ):
            subset = [
                row
                for row in bridge_rows
                if row["metric"] == metric
                and str(row["optimizer_step"]) == str(step)
                and (
                    contrast_class == "all"
                    or row["contrast_class"] == contrast_class
                )
            ]
            predicted = [
                float(row["predicted_delta_left_minus_right"])
                for row in subset
            ]
            realized = [
                float(row["realized_delta_left_minus_right"])
                for row in subset
            ]
            same_sign = sum(
                (left < 0.0 and right < 0.0)
                or (left > 0.0 and right > 0.0)
                or (left == 0.0 and right == 0.0)
                for left, right in zip(predicted, realized)
            )
            alignment_rows.append(
                {
                    "contrast_scope": contrast_class,
                    "metric": metric,
                    "optimizer_step": step,
                    "origin_contrast_units": len(subset),
                    "pearson": pearson(predicted, realized),
                    "spearman": spearman(predicted, realized),
                    "sign_concordant_units": same_sign,
                    "sign_concordance": same_sign / len(subset),
                    "descriptive_not_confirmatory": True,
                }
            )

    write_csv(output / "evaluation_all.csv", evaluation_rows)
    write_csv(output / "run_summary.csv", run_summaries)
    write_csv(output / "matched_start_audit.csv", start_audit)
    write_csv(output / "paired_contrasts.csv", paired_rows)
    write_csv(output / "paired_contrast_summary.csv", aggregate_rows)
    write_csv(output / "prediction_bridge_by_replica.csv", bridge_replica_rows)
    write_csv(output / "prediction_bridge.csv", bridge_rows)
    write_csv(output / "prediction_alignment.csv", alignment_rows)
    write_csv(output / "source_artifacts.csv", source_rows)

    checks = {
        "formal_jobs": len(run_summaries)
        == int(contract["analysis_contract"]["minimum_complete_formal_jobs"]),
        "evaluation_rows": len(evaluation_rows)
        == expected_jobs * len(evaluation_steps),
        "matched_starts": all(row["matched"] for row in start_audit),
        "four_primary_contrasts": len(
            contract["comparison_contract"]["primary"]
        )
        == 4,
        "diag_none_absent": all(
            {row["left"], row["right"]}
            != {"selective_diag", "selective_none"}
            for row in paired_rows
        ),
        "prediction_reference": sha256_file(
            args.prediction_reference.resolve()
        )
        == expected_prediction_sha,
        "efficiency_excluded": contract["scope_boundary"][
            "efficiency_benchmark_excluded"
        ]
        is True,
        "integrity_not_hypothesis": contract["analysis_contract"][
            "integrity_pass_is_not_hypothesis_success"
        ]
        is True,
    }
    write_json(output / "checks.json", checks)
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "contract_sha256": expected_contract_sha,
        "prediction_reference_sha256": expected_prediction_sha,
        "formal_jobs": len(run_summaries),
        "evaluation_rows": len(evaluation_rows),
        "paired_contrast_rows": len(paired_rows),
        "primary_contrasts": [
            row["name"] for row in contract["comparison_contract"]["primary"]
        ],
        "diag_none_primary": False,
        "integrity_pass_is_hypothesis_success": False,
        "timing_usable_for_paper": False,
        "efficiency_benchmark_run": False,
        "checks": checks,
        "passed": all(checks.values()),
        "artifacts": sorted(
            [
                "checks.json",
                "evaluation_all.csv",
                "integrity_audit.json",
                "matched_start_audit.csv",
                "mech08_analysis_manifest.json",
                "paired_contrasts.csv",
                "paired_contrast_summary.csv",
                "prediction_alignment.csv",
                "prediction_bridge.csv",
                "prediction_bridge_by_replica.csv",
                "run_summary.csv",
                "source_artifacts.csv",
            ]
        ),
    }
    write_json(output / "mech08_analysis_manifest.json", manifest)
    if not manifest["passed"]:
        raise SystemExit(2)
    print(
        f"MECH-08 analysis manifest: "
        f"{output / 'mech08_analysis_manifest.json'}"
    )
    print(f"MECH-08 analysis artifacts: {output}")


if __name__ == "__main__":
    main()
