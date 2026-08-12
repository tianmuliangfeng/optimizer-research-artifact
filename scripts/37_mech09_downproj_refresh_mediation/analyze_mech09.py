#!/usr/bin/env python3
"""Analyze MECH-09 against the hash-frozen MECH-08 controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-28.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--mech08-control-reference", required=True, type=Path)
    parser.add_argument("--mech08-run-dir", required=True, type=Path)
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
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_rows(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> bool:
    return all(
        math.isfinite(float(row[field]))
        for row in rows
        for field in fields
    )


def trapezoid_auc(rows: list[dict[str, Any]]) -> float:
    ordered = sorted(rows, key=lambda row: int(row["optimizer_step"]))
    area = 0.0
    for left, right in zip(ordered, ordered[1:]):
        width = int(right["optimizer_step"]) - int(left["optimizer_step"])
        area += width * (
            float(left["normalized_loss"]) + float(right["normalized_loss"])
        ) / 2.0
    horizon = int(ordered[-1]["optimizer_step"]) - int(
        ordered[0]["optimizer_step"]
    )
    if horizon <= 0:
        raise ValueError("AUC horizon must be positive")
    return area / horizon


def verify_control_reference(
    reference: dict[str, Any], mech08_run: Path
) -> dict[str, Any]:
    rows = []
    for spec in reference["files"]:
        path = mech08_run / spec["relative_path"]
        exists = path.is_file()
        observed_bytes = path.stat().st_size if exists else -1
        observed_sha = sha256_file(path) if exists else None
        rows.append(
            {
                "relative_path": spec["relative_path"],
                "role": spec["role"],
                "exists": exists,
                "expected_bytes": int(spec["bytes"]),
                "observed_bytes": observed_bytes,
                "expected_sha256": spec["sha256"],
                "observed_sha256": observed_sha,
                "passed": exists
                and observed_bytes == int(spec["bytes"])
                and observed_sha == spec["sha256"],
            }
        )
    return {
        "source_run": str(mech08_run),
        "expected_files": int(reference["file_count"]),
        "observed_files": len(rows),
        "files": rows,
        "passed": len(rows) == int(reference["file_count"])
        and all(row["passed"] for row in rows),
    }


def evaluation_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["checkpoint_cell"]),
        int(row["data_replica"]),
        int(row["optimizer_step"]),
    )


def contrast_summary(
    paired_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        grouped[(row["contrast"], int(row["optimizer_step"]))].append(row)
    output = []
    for (contrast, step), rows in sorted(grouped.items()):
        values = [float(row["normalized_loss_delta"]) for row in rows]
        origin_values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            origin_values[str(row["checkpoint_cell"])].append(
                float(row["normalized_loss_delta"])
            )
        origin_means = [
            statistics.mean(values) for values in origin_values.values()
        ]
        output.append(
            {
                "contrast": contrast,
                "optimizer_step": step,
                "paired_units": len(values),
                "origins": len(origin_means),
                "mean_delta": statistics.mean(values),
                "sd_across_paired_units": (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                ),
                "left_better_units": sum(value < 0.0 for value in values),
                "left_worse_units": sum(value > 0.0 for value in values),
                "left_better_origins": sum(
                    value < 0.0 for value in origin_means
                ),
                "left_worse_origins": sum(
                    value > 0.0 for value in origin_means
                ),
                "negative_delta_means": "left intervention is better",
            }
        )
    return output


def summary_row(
    summaries: list[dict[str, Any]], contrast: str, step: int
) -> dict[str, Any]:
    matches = [
        row
        for row in summaries
        if row["contrast"] == contrast
        and int(row["optimizer_step"]) == int(step)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"summary row is not unique: {contrast} step={step}")
    return matches[0]


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    output = run / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "mech09_analysis_manifest.json"
    contract = read_json(args.contract.resolve())
    reference = read_json(args.mech08_control_reference.resolve())
    contract_sha = sha256_file(args.contract.resolve())
    reference_sha = sha256_file(args.mech08_control_reference.resolve())
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if (
            existing.get("passed") is True
            and existing.get("script_version") == SCRIPT_VERSION
            and existing.get("contract_sha256") == contract_sha
            and existing.get("control_reference_sha256") == reference_sha
        ):
            print(f"MECH-09 analysis already passed: {manifest_path}")
            return

    control_audit = verify_control_reference(
        reference, args.mech08_run_dir.resolve()
    )
    write_json(output / "mech08_control_audit.json", control_audit)
    if not control_audit["passed"]:
        raise RuntimeError("hash-frozen MECH-08 control audit failed")

    formal = contract["formal"]
    control_evaluations: list[dict[str, Any]] = []
    for cell in formal["origins"]:
        for algorithm in reference["algorithms"]:
            for replica in formal["data_replicas"]:
                path = (
                    args.mech08_run_dir.resolve()
                    / "formal"
                    / cell
                    / algorithm
                    / f"replica_{int(replica)}"
                    / "evaluation.csv"
                )
                for row in read_csv(path):
                    control_evaluations.append(
                        {
                            "checkpoint_cell": cell,
                            "checkpoint_stage": row["checkpoint_stage"],
                            "checkpoint_method": row["checkpoint_method"],
                            "algorithm": algorithm,
                            "data_replica": int(replica),
                            "optimizer_step": int(row["optimizer_step"]),
                            "heldout_loss": float(row["heldout_loss"]),
                            "normalized_loss": float(row["normalized_loss"]),
                            "source": "MECH-08 hash-frozen control",
                        }
                    )

    intervention_evaluations: list[dict[str, Any]] = []
    worker_inventory: list[dict[str, Any]] = []
    refresh_rows: list[dict[str, Any]] = []
    for cell in formal["origins"]:
        for intervention in formal["interventions"]:
            for replica in formal["data_replicas"]:
                directory = (
                    run
                    / "formal"
                    / cell
                    / intervention
                    / f"replica_{int(replica)}"
                )
                manifest_file = directory / "mech09_manifest.json"
                checks_file = directory / "checks.json"
                status_file = directory / "status.json"
                refresh_file = directory / "refresh_intervention_audit.json"
                manifest = read_json(manifest_file)
                checks = read_json(checks_file)
                status = read_json(status_file)
                refresh = read_json(refresh_file)
                local_checks = {
                    "manifest_passed": manifest.get("passed") is True,
                    "checks_passed": all(checks.values()),
                    "status_passed": status.get("status") == "passed",
                    "tier": manifest.get("analysis_tier") == "formal",
                    "cell": manifest.get("checkpoint_cell") == cell,
                    "intervention": manifest.get("intervention")
                    == intervention,
                    "replica": int(manifest.get("data_replica", -1))
                    == int(replica),
                    "contract": manifest.get("contract_sha256")
                    == contract_sha,
                    "worker_version": manifest.get("script_version")
                    == SCRIPT_VERSION,
                    "timing_excluded": manifest.get(
                        "timing_usable_for_paper"
                    )
                    is False,
                    "refresh_audit": refresh.get("passed") is True,
                }
                worker_inventory.append(
                    {
                        "checkpoint_cell": cell,
                        "intervention": intervention,
                        "data_replica": int(replica),
                        **local_checks,
                        "passed": all(local_checks.values()),
                        "manifest": str(manifest_file),
                        "manifest_sha256": sha256_file(manifest_file),
                    }
                )
                for row in read_csv(directory / "evaluation.csv"):
                    intervention_evaluations.append(
                        {
                            "checkpoint_cell": cell,
                            "checkpoint_stage": row["checkpoint_stage"],
                            "checkpoint_method": row["checkpoint_method"],
                            "algorithm": intervention,
                            "data_replica": int(replica),
                            "optimizer_step": int(row["optimizer_step"]),
                            "heldout_loss": float(row["heldout_loss"]),
                            "normalized_loss": float(row["normalized_loss"]),
                            "source": "MECH-09 intervention",
                        }
                    )
                for event in refresh["events"]:
                    refresh_rows.append(
                        {
                            "checkpoint_cell": cell,
                            "intervention": intervention,
                            "data_replica": int(replica),
                            "completed_step": int(event["completed_step"]),
                            "target_action": event["target_action"],
                            "target_covariance_changed": event[
                                "target_covariance_changed"
                            ],
                            "target_inverse_changed": event[
                                "target_inverse_changed"
                            ],
                            "target_statistics_zero_after": event[
                                "target_statistics_zero_after"
                            ],
                            "other_covariance_changed": event[
                                "other_covariance_changed"
                            ],
                            "other_inverse_changed": event[
                                "other_inverse_changed"
                            ],
                            "other_statistics_zero_after": event[
                                "other_statistics_zero_after"
                            ],
                        }
                    )

    evaluation_all = control_evaluations + intervention_evaluations
    write_csv(output / "evaluation_all.csv", evaluation_all)
    write_csv(output / "worker_inventory.csv", worker_inventory)
    write_csv(output / "refresh_event_summary.csv", refresh_rows)

    by_algorithm: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = {}
    for algorithm in [
        *reference["algorithms"],
        *formal["interventions"],
    ]:
        rows = [row for row in evaluation_all if row["algorithm"] == algorithm]
        by_algorithm[algorithm] = {
            evaluation_key(row): row for row in rows
        }
        if len(by_algorithm[algorithm]) != len(rows):
            raise RuntimeError(f"duplicate evaluation grain for {algorithm}")

    contrast_specs = contract["comparison_contract"]["primary"]
    paired_rows: list[dict[str, Any]] = []
    for contrast in contrast_specs:
        left = contrast["left"]
        right = contrast["right"]
        left_rows = by_algorithm[left]
        right_rows = by_algorithm[right]
        if set(left_rows) != set(right_rows):
            raise RuntimeError(f"paired keys differ for {contrast['name']}")
        for key in sorted(left_rows):
            left_row = left_rows[key]
            right_row = right_rows[key]
            paired_rows.append(
                {
                    "checkpoint_cell": key[0],
                    "data_replica": key[1],
                    "optimizer_step": key[2],
                    "contrast": contrast["name"],
                    "left": left,
                    "right": right,
                    "left_normalized_loss": left_row["normalized_loss"],
                    "right_normalized_loss": right_row["normalized_loss"],
                    "normalized_loss_delta": (
                        float(left_row["normalized_loss"])
                        - float(right_row["normalized_loss"])
                    ),
                    "negative_delta_means": "left intervention is better",
                }
            )
    summaries = contrast_summary(paired_rows)
    write_csv(output / "paired_contrasts.csv", paired_rows)
    write_csv(output / "paired_contrast_summary.csv", summaries)

    auc_by_algorithm: dict[str, dict[tuple[str, int], float]] = {}
    for algorithm, rows_by_key in by_algorithm.items():
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows_by_key.values():
            grouped[
                (row["checkpoint_cell"], int(row["data_replica"]))
            ].append(row)
        auc_by_algorithm[algorithm] = {
            key: trapezoid_auc(rows) for key, rows in grouped.items()
        }
    auc_rows = []
    for contrast in contrast_specs:
        left_values = auc_by_algorithm[contrast["left"]]
        right_values = auc_by_algorithm[contrast["right"]]
        for key in sorted(left_values):
            auc_rows.append(
                {
                    "checkpoint_cell": key[0],
                    "data_replica": key[1],
                    "contrast": contrast["name"],
                    "left": contrast["left"],
                    "right": contrast["right"],
                    "left_auc": left_values[key],
                    "right_auc": right_values[key],
                    "auc_delta": left_values[key] - right_values[key],
                }
            )
    write_csv(output / "auc_contrasts.csv", auc_rows)

    matched_start_rows = []
    for cell in formal["origins"]:
        for replica in formal["data_replicas"]:
            rows = [
                row
                for row in evaluation_all
                if row["checkpoint_cell"] == cell
                and int(row["data_replica"]) == int(replica)
                and int(row["optimizer_step"]) == 0
            ]
            values = [float(row["heldout_loss"]) for row in rows]
            matched_start_rows.append(
                {
                    "checkpoint_cell": cell,
                    "data_replica": int(replica),
                    "algorithms": len(rows),
                    "minimum_heldout_loss": min(values),
                    "maximum_heldout_loss": max(values),
                    "spread": max(values) - min(values),
                }
            )
    write_csv(output / "matched_starts.csv", matched_start_rows)

    rules = contract["analysis_contract"]
    tolerance = float(rules["pre_refresh_max_abs_normalized_loss_delta"])
    pre_step = int(rules["pre_refresh_equivalence_step"])
    step48 = int(rules["first_post_production_refresh_evaluation_step"])
    step80 = int(rules["first_post_delayed_refresh_evaluation_step"])
    minimum_units = int(rules["supermajority_paired_units"])
    minimum_origins = int(rules["supermajority_origins"])

    pre_rows = [
        row
        for row in paired_rows
        if row["contrast"]
        in {
            "delayed_down_refresh_vs_production",
            "frozen_down_refresh_vs_production",
        }
        and int(row["optimizer_step"]) == pre_step
    ]
    pre_refresh_equivalence = (
        len(pre_rows) == 24
        and max(abs(float(row["normalized_loss_delta"])) for row in pre_rows)
        <= tolerance
    )
    delayed_protection = summary_row(
        summaries, "delayed_down_refresh_vs_production", step48
    )
    frozen_protection = summary_row(
        summaries, "frozen_down_refresh_vs_production", step48
    )
    delayed_vs_frozen_48 = [
        row
        for row in paired_rows
        if row["contrast"]
        == "delayed_down_refresh_vs_frozen_down_refresh"
        and int(row["optimizer_step"]) == step48
    ]
    timing_specificity = (
        len(delayed_vs_frozen_48) == 12
        and max(
            abs(float(row["normalized_loss_delta"]))
            for row in delayed_vs_frozen_48
        )
        <= tolerance
    )
    delayed_post_refresh = summary_row(
        summaries, "delayed_down_refresh_vs_frozen_down_refresh", step80
    )
    directional = {
        "delayed_protects_after_production_refresh": (
            float(delayed_protection["mean_delta"]) < 0.0
            and int(delayed_protection["left_better_units"]) >= minimum_units
            and int(delayed_protection["left_better_origins"])
            >= minimum_origins
        ),
        "frozen_protects_after_production_refresh": (
            float(frozen_protection["mean_delta"]) < 0.0
            and int(frozen_protection["left_better_units"]) >= minimum_units
            and int(frozen_protection["left_better_origins"])
            >= minimum_origins
        ),
        "delayed_worsens_after_its_refresh": (
            float(delayed_post_refresh["mean_delta"]) > 0.0
            and int(delayed_post_refresh["left_worse_units"]) >= minimum_units
            and int(delayed_post_refresh["left_worse_origins"])
            >= minimum_origins
        ),
    }
    directional_passes = sum(directional.values())

    integrity_checks = {
        "control_reference": control_audit["passed"],
        "control_reference_sha": reference_sha
        == contract["mech08_control_reference"].get(
            "public_sha256", contract["mech08_control_reference"]["sha256"]
        ),
        "control_run_path": str(args.mech08_run_dir.resolve())
        == contract["mech08_control_reference"]["source_run_path"],
        "formal_worker_count": len(worker_inventory)
        == int(rules["minimum_complete_formal_jobs"]),
        "formal_workers_passed": all(
            row["passed"] for row in worker_inventory
        ),
        "intervention_evaluation_rows": len(intervention_evaluations)
        == 24 * len(formal["evaluation_steps"]),
        "control_evaluation_rows": len(control_evaluations)
        == 48 * len(formal["evaluation_steps"]),
        "refresh_event_rows": len(refresh_rows)
        == 24
        * len(formal["expected_other_group_refresh_completed_steps"]),
        "finite_evaluation": finite_rows(
            evaluation_all, ("heldout_loss", "normalized_loss")
        ),
        "matched_start_units": len(matched_start_rows) == 12,
        "matched_start_algorithm_count": all(
            int(row["algorithms"]) == 6 for row in matched_start_rows
        ),
        "matched_start_spread": max(
            float(row["spread"]) for row in matched_start_rows
        )
        == 0.0,
        "pre_refresh_equivalence": pre_refresh_equivalence,
        "formal_job_cap_respected": len(worker_inventory)
        <= int(contract["stopping_rule"]["maximum_new_formal_jobs"]),
        "timing_excluded": all(
            row["timing_excluded"] for row in worker_inventory
        ),
    }
    integrity_passed = all(integrity_checks.values())
    if not integrity_passed:
        classification = "invalid"
    elif directional_passes == 3 and timing_specificity:
        classification = "full_support"
    elif directional_passes >= 2:
        classification = "partial_support"
    else:
        classification = "not_supported"
    decision = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "integrity_passed": integrity_passed,
        "pre_refresh_equivalence": {
            "step": pre_step,
            "tolerance": tolerance,
            "maximum_abs_delta": max(
                abs(float(row["normalized_loss_delta"]))
                for row in pre_rows
            ),
            "passed": pre_refresh_equivalence,
        },
        "timing_specificity_before_delayed_refresh": {
            "step": step48,
            "maximum_abs_delayed_vs_frozen_delta": max(
                abs(float(row["normalized_loss_delta"]))
                for row in delayed_vs_frozen_48
            ),
            "tolerance": tolerance,
            "passed": timing_specificity,
        },
        "directional_predictions": directional,
        "directional_predictions_passed": directional_passes,
        "classification": classification,
        "classification_is_scientific_not_execution_status": True,
        "do_not_rerun_for_unfavorable_outcome": contract["stopping_rule"][
            "do_not_rerun_for_unfavorable_scientific_outcome"
        ],
    }
    write_json(output / "mediation_decision.json", decision)
    write_json(
        output / "integrity_checks.json",
        {"checks": integrity_checks, "passed": integrity_passed},
    )
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "contract_sha256": contract_sha,
        "control_reference_sha256": reference_sha,
        "mech08_source_run_id": reference["source_run_id"],
        "new_formal_workers": len(worker_inventory),
        "reused_control_workers": int(
            contract["mech08_control_reference"]["reused_control_workers"]
        ),
        "evaluation_rows": len(evaluation_all),
        "paired_contrast_rows": len(paired_rows),
        "refresh_event_rows": len(refresh_rows),
        "integrity_passed": integrity_passed,
        "hypothesis_classification": classification,
        "passed": integrity_passed,
        "timing_usable_for_paper": False,
        "artifacts": sorted(
            [
                "auc_contrasts.csv",
                "evaluation_all.csv",
                "integrity_checks.json",
                "matched_starts.csv",
                "mech08_control_audit.json",
                "mech09_analysis_manifest.json",
                "mediation_decision.json",
                "paired_contrast_summary.csv",
                "paired_contrasts.csv",
                "refresh_event_summary.csv",
                "worker_inventory.csv",
            ]
        ),
    }
    write_json(manifest_path, manifest)
    if not integrity_passed:
        raise SystemExit(2)
    print(f"MECH-09 analysis manifest: {manifest_path}")
    print(f"MECH-09 hypothesis classification: {classification}")


if __name__ == "__main__":
    main()
