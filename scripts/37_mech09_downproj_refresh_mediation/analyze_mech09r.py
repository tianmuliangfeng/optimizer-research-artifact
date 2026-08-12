#!/usr/bin/env python3
"""Analyze the shared-prefix MECH-09R causal-tree experiment."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
import os
import math
from pathlib import Path
import statistics
from typing import Any


SCRIPT_VERSION = "2026-07-28.3"
WORKER_VERSION = "2026-07-28.3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
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


def evaluation_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["checkpoint_cell"]),
        int(row["data_replica"]),
        int(row["optimizer_step"]),
    )


def finite_rows(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> bool:
    return all(
        math.isfinite(float(row[field]))
        for row in rows
        for field in fields
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
        by_origin: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_origin[str(row["checkpoint_cell"])].append(
                float(row["normalized_loss_delta"])
            )
        origin_means = [
            statistics.mean(values) for values in by_origin.values()
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
                "negative_delta_means": "left arm is better",
            }
        )
    return output


def summary_row(
    rows: list[dict[str, Any]], contrast: str, step: int
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["contrast"] == contrast
        and int(row["optimizer_step"]) == int(step)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"summary row is not unique: {contrast} step={step}"
        )
    return matches[0]


def trapezoid_auc(rows: list[dict[str, Any]]) -> float:
    ordered = sorted(rows, key=lambda row: int(row["optimizer_step"]))
    area = 0.0
    for left, right in zip(ordered, ordered[1:]):
        width = int(right["optimizer_step"]) - int(left["optimizer_step"])
        area += width * (
            float(left["normalized_loss"])
            + float(right["normalized_loss"])
        ) / 2.0
    horizon = int(ordered[-1]["optimizer_step"]) - int(
        ordered[0]["optimizer_step"]
    )
    if horizon <= 0:
        raise ValueError("AUC horizon must be positive")
    return area / horizon


def exact_step_audit(
    by_arm: dict[str, dict[tuple[str, int, int], dict[str, Any]]],
    arms: tuple[str, ...],
    step: int,
) -> dict[str, Any]:
    unit_keys = {
        (cell, replica)
        for cell, replica, observed_step in by_arm[arms[0]]
        if observed_step == step
    }
    rows = []
    for cell, replica in sorted(unit_keys):
        values = [
            float(by_arm[arm][(cell, replica, step)]["heldout_loss"])
            for arm in arms
        ]
        rows.append(
            {
                "checkpoint_cell": cell,
                "data_replica": replica,
                "optimizer_step": step,
                "arms": list(arms),
                "values": values,
                "spread": max(values) - min(values),
                "passed": len(set(values)) == 1,
            }
        )
    return {
        "step": step,
        "arms": list(arms),
        "units": len(rows),
        "maximum_spread": max(float(row["spread"]) for row in rows),
        "rows": rows,
        "passed": len(rows) == 12 and all(row["passed"] for row in rows),
    }


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    output = run / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "mech09r_analysis_manifest.json"
    contract = read_json(args.contract.resolve())
    contract_sha = sha256_file(args.contract.resolve())
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if (
            existing.get("passed") is True
            and existing.get("script_version") == SCRIPT_VERSION
            and existing.get("contract_sha256") == contract_sha
        ):
            print(f"MECH-09R analysis already passed: {manifest_path}")
            return

    formal = contract["formal"]
    arms = tuple(contract["arms"])
    worker_inventory: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for cell in formal["origins"]:
        for replica in formal["data_replicas"]:
            directory = run / "formal" / cell / f"replica_{int(replica)}"
            manifest_file = directory / "mech09r_manifest.json"
            status_file = directory / "status.json"
            checks_file = directory / "checks.json"
            branch_file = directory / "branch_audit.json"
            refresh_file = directory / "refresh_tree_audit.json"
            manifest = read_json(manifest_file)
            status = read_json(status_file)
            checks = read_json(checks_file)
            branch = read_json(branch_file)
            refresh = read_json(refresh_file)
            local_checks = {
                "manifest_passed": manifest.get("passed") is True,
                "status_passed": status.get("status") == "passed",
                "checks_passed": all(value is True for value in checks.values()),
                "branch_passed": branch.get("passed") is True,
                "refresh_passed": refresh.get("passed") is True,
                "tier": manifest.get("analysis_tier") == "formal",
                "cell": manifest.get("checkpoint_cell") == cell,
                "replica": int(manifest.get("data_replica", -1))
                == int(replica),
                "worker_version": manifest.get("script_version")
                == WORKER_VERSION,
                "contract_sha256": manifest.get("contract_sha256")
                == contract_sha,
                "causal_tree": manifest.get("causal_tree") is True,
                "timing_excluded": manifest.get(
                    "timing_usable_for_paper"
                )
                is False,
                "legacy_not_reused": manifest.get(
                    "legacy_invalid_run_reused"
                )
                is False,
            }
            worker_inventory.append(
                {
                    "checkpoint_cell": cell,
                    "data_replica": int(replica),
                    **local_checks,
                    "passed": all(local_checks.values()),
                    "manifest": str(manifest_file),
                    "manifest_sha256": sha256_file(manifest_file),
                }
            )
            evaluation_rows.extend(read_csv(directory / "evaluation.csv"))
            training_rows.extend(read_csv(directory / "training.csv"))

    write_csv(output / "worker_inventory.csv", worker_inventory)
    write_csv(output / "evaluation_all.csv", evaluation_rows)
    write_csv(output / "training_all.csv", training_rows)
    by_arm: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = {}
    for arm in arms:
        rows = [row for row in evaluation_rows if row["arm"] == arm]
        by_arm[arm] = {evaluation_key(row): row for row in rows}
        if len(by_arm[arm]) != len(rows):
            raise RuntimeError(f"duplicate evaluation grain for {arm}")
    expected_keys = set(by_arm[arms[0]])
    if any(set(by_arm[arm]) != expected_keys for arm in arms[1:]):
        raise RuntimeError("arm evaluation keys differ")

    paired_rows: list[dict[str, Any]] = []
    for contrast in contract["comparison_contract"]["primary"]:
        left = contrast["left"]
        right = contrast["right"]
        for key in sorted(expected_keys):
            left_row = by_arm[left][key]
            right_row = by_arm[right][key]
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
                    "negative_delta_means": "left arm is better",
                }
            )
    summaries = contrast_summary(paired_rows)
    write_csv(output / "paired_contrasts.csv", paired_rows)
    write_csv(output / "paired_contrast_summary.csv", summaries)

    auc_by_arm: dict[str, dict[tuple[str, int], float]] = {}
    for arm, rows_by_key in by_arm.items():
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows_by_key.values():
            grouped[
                (str(row["checkpoint_cell"]), int(row["data_replica"]))
            ].append(row)
        auc_by_arm[arm] = {
            key: trapezoid_auc(rows) for key, rows in grouped.items()
        }
    auc_rows = []
    for contrast in contract["comparison_contract"]["primary"]:
        for key in sorted(auc_by_arm[contrast["left"]]):
            left_auc = auc_by_arm[contrast["left"]][key]
            right_auc = auc_by_arm[contrast["right"]][key]
            auc_rows.append(
                {
                    "checkpoint_cell": key[0],
                    "data_replica": key[1],
                    "contrast": contrast["name"],
                    "left": contrast["left"],
                    "right": contrast["right"],
                    "left_auc": left_auc,
                    "right_auc": right_auc,
                    "auc_delta": left_auc - right_auc,
                }
            )
    write_csv(output / "auc_contrasts.csv", auc_rows)

    rules = contract["analysis_contract"]
    pre_audit = exact_step_audit(
        by_arm,
        arms,
        int(rules["pre_refresh_equivalence_step"]),
    )
    delayed_frozen_audit = exact_step_audit(
        by_arm,
        ("delayed_down_refresh", "frozen_down_refresh"),
        int(rules["pre_delayed_refresh_shared_step"]),
    )
    shared_prefix_audit = {
        "pre_refresh_all_arms": pre_audit,
        "pre_delayed_refresh_delayed_frozen": delayed_frozen_audit,
        "passed": pre_audit["passed"] and delayed_frozen_audit["passed"],
    }
    write_json(output / "shared_prefix_audit.json", shared_prefix_audit)

    step48 = int(rules["first_post_production_refresh_evaluation_step"])
    step80 = int(rules["first_post_delayed_refresh_evaluation_step"])
    minimum_units = int(rules["supermajority_paired_units"])
    minimum_origins = int(rules["supermajority_origins"])
    delayed_protection = summary_row(
        summaries, "delayed_down_refresh_vs_production", step48
    )
    frozen_protection = summary_row(
        summaries, "frozen_down_refresh_vs_production", step48
    )
    delayed_post_refresh = summary_row(
        summaries, "delayed_down_refresh_vs_frozen_down_refresh", step80
    )
    directional = {
        "delayed_protects_after_production_refresh": (
            float(delayed_protection["mean_delta"]) < 0.0
            and int(delayed_protection["left_better_units"])
            >= minimum_units
            and int(delayed_protection["left_better_origins"])
            >= minimum_origins
        ),
        "frozen_protects_after_production_refresh": (
            float(frozen_protection["mean_delta"]) < 0.0
            and int(frozen_protection["left_better_units"])
            >= minimum_units
            and int(frozen_protection["left_better_origins"])
            >= minimum_origins
        ),
        "delayed_worsens_after_its_refresh": (
            float(delayed_post_refresh["mean_delta"]) > 0.0
            and int(delayed_post_refresh["left_worse_units"])
            >= minimum_units
            and int(delayed_post_refresh["left_worse_origins"])
            >= minimum_origins
        ),
    }
    directional_passes = sum(directional.values())
    expected_units = int(rules["minimum_complete_formal_jobs"])
    expected_evaluation_rows = (
        expected_units * len(arms) * len(formal["evaluation_steps"])
    )
    expected_training_rows = (
        expected_units * len(arms) * int(formal["rollout_steps"])
    )
    integrity_checks = {
        "contract_version": contract.get("contract_version")
        == "2026-07-28.2",
        "protocol_amendment_pre_intervention_only": contract[
            "protocol_amendment"
        ]["trigger_uses_pre_intervention_data_only"]
        is True,
        "formal_worker_count": len(worker_inventory) == expected_units,
        "formal_workers_passed": all(
            row["passed"] for row in worker_inventory
        ),
        "evaluation_rows": len(evaluation_rows)
        == expected_evaluation_rows,
        "training_rows": len(training_rows) == expected_training_rows,
        "finite_evaluation": finite_rows(
            evaluation_rows, ("heldout_loss", "normalized_loss")
        ),
        "finite_training": finite_rows(
            training_rows, ("train_loss_mean",)
        ),
        "arm_keys_exact": all(
            set(by_arm[arm]) == expected_keys for arm in arms
        ),
        "pre_refresh_exact": pre_audit["passed"],
        "pre_delayed_refresh_exact": delayed_frozen_audit["passed"],
        "formal_job_cap": len(worker_inventory)
        <= int(contract["stopping_rule"]["maximum_new_formal_jobs"]),
        "trajectory_cap": len(worker_inventory) * len(arms)
        <= int(contract["stopping_rule"]["maximum_trajectories"]),
        "timing_excluded": all(
            str(row["timing_usable_for_paper"]).lower() == "false"
            for row in training_rows
        ),
    }
    integrity_passed = all(integrity_checks.values())
    if not integrity_passed:
        classification = "invalid"
    elif directional_passes == 3:
        classification = "full_support"
    elif directional_passes >= 2:
        classification = "partial_support"
    else:
        classification = "not_supported"
    decision = {
        "schema_version": 2,
        "script_version": SCRIPT_VERSION,
        "integrity_passed": integrity_passed,
        "shared_prefix_audit": shared_prefix_audit,
        "directional_predictions": directional,
        "directional_predictions_passed": directional_passes,
        "classification": classification,
        "classification_is_scientific_not_execution_status": True,
        "legacy_independent_worker_run_used_for_claim": False,
        "do_not_rerun_for_unfavorable_outcome": contract["stopping_rule"][
            "do_not_rerun_for_unfavorable_scientific_outcome"
        ],
    }
    write_json(output / "mediation_decision.json", decision)
    write_json(
        output / "integrity_checks.json",
        {"checks": integrity_checks, "passed": integrity_passed},
    )
    write_json(
        output / "protocol_amendment.json",
        contract["protocol_amendment"],
    )
    manifest = {
        "schema_version": 2,
        "script_version": SCRIPT_VERSION,
        "experiment": "MECH-09R",
        "contract_sha256": contract_sha,
        "formal_workers": len(worker_inventory),
        "arms": list(arms),
        "evaluation_rows": len(evaluation_rows),
        "training_rows": len(training_rows),
        "paired_contrast_rows": len(paired_rows),
        "integrity_passed": integrity_passed,
        "hypothesis_classification": classification,
        "passed": integrity_passed,
        "timing_usable_for_paper": False,
        "legacy_invalid_run_reused": False,
        "artifacts": sorted(
            [
                "auc_contrasts.csv",
                "evaluation_all.csv",
                "integrity_checks.json",
                "mech09r_analysis_manifest.json",
                "mediation_decision.json",
                "paired_contrast_summary.csv",
                "paired_contrasts.csv",
                "protocol_amendment.json",
                "shared_prefix_audit.json",
                "training_all.csv",
                "worker_inventory.csv",
            ]
        ),
    }
    write_json(manifest_path, manifest)
    if not integrity_passed:
        raise SystemExit(2)
    print(f"MECH-09R analysis manifest: {manifest_path}")
    print(f"MECH-09R hypothesis classification: {classification}")


if __name__ == "__main__":
    main()
