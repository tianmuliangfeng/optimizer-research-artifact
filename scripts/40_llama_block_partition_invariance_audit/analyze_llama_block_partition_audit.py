#!/usr/bin/env python3
"""Aggregate formal early/late block-partition invariance evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "2026-07-29.2"
WORKER_VERSION = "2026-07-29.2"
CONTRACT_VERSION = "2026-07-29.2"
HERE = Path(__file__).resolve().parent
WORKER = HERE / "llama_block_partition_worker.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": quantile(values, 0.95),
        "maximum": max(values),
    }


def classify(
    median_drift: float,
    control_maximum: float,
    thresholds: dict[str, Any],
    stage_medians: list[float],
) -> tuple[str, dict[str, float]]:
    denominator = max(control_maximum, 1e-8)
    multiple = median_drift / denominator
    strong_each_stage = all(
        value >= thresholds["strong_median_block4_update_drift"]
        for value in stage_medians
    )
    detectable_each_stage = all(
        value >= thresholds["detectable_median_block4_update_drift"]
        for value in stage_medians
    )
    negligible_each_stage = all(
        value <= thresholds["negligible_median_block4_update_drift"]
        for value in stage_medians
    )
    require_consensus = thresholds[
        "stage_consensus_required_for_non_invariance"
    ]
    if (
        median_drift >= thresholds["strong_median_block4_update_drift"]
        and multiple >= thresholds["strong_control_multiple"]
        and (strong_each_stage or not require_consensus)
    ):
        classification = "strong_non_invariance"
    elif (
        median_drift >= thresholds["detectable_median_block4_update_drift"]
        and multiple >= thresholds["detectable_control_multiple"]
        and (detectable_each_stage or not require_consensus)
    ):
        classification = "detectable_non_invariance"
    elif (
        median_drift <= thresholds["negligible_median_block4_update_drift"]
        and negligible_each_stage
    ):
        classification = "approximately_invariant_at_tested_resolution"
    else:
        classification = "inconclusive"
    return classification, {
        "pooled_global_block4_median_update_drift": median_drift,
        "maximum_equivariant_control_drift": control_maximum,
        "effect_to_control_multiple": multiple,
        "minimum_stage_median_update_drift": min(stage_medians),
        "maximum_stage_median_update_drift": max(stage_medians),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output_dir.resolve()
    if not output.is_dir():
        raise RuntimeError(f"controller must create output directory: {output}")
    contract = read_json(args.contract.resolve())
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError("analysis contract version mismatch")
    contract_sha = sha256_file(args.contract.resolve())

    manifests = {}
    checks_payloads = {}
    all_updates: list[dict[str, str]] = []
    all_partitions: list[dict[str, str]] = []
    all_summaries: list[dict[str, str]] = []
    input_checks: dict[str, bool] = {}
    for label in ("early", "late"):
        directory = run_dir / "formal" / label
        manifest = read_json(directory / "llama_block_audit_manifest.json")
        checks = read_json(directory / "checks.json")
        manifests[label] = manifest
        checks_payloads[label] = checks
        input_checks[f"{label}_manifest_passed"] = manifest.get("passed") is True
        input_checks[f"{label}_tier"] = manifest.get("analysis_tier") == "formal"
        input_checks[f"{label}_worker_version"] = (
            manifest.get("script_version") == WORKER_VERSION
        )
        input_checks[f"{label}_worker_sha256"] = (
            manifest.get("worker_sha256") == sha256_file(WORKER)
        )
        input_checks[f"{label}_contract_sha"] = (
            manifest.get("contract_sha256") == contract_sha
        )
        input_checks[f"{label}_checkpoint_sha"] = (
            manifest.get("checkpoint_sha256")
            == contract["checkpoints"][label]["sha256"]
        )
        input_checks[f"{label}_integrity_checks"] = all(checks.values())
        all_updates.extend(read_csv(directory / "equivariance_updates.csv"))
        all_partitions.extend(read_csv(directory / "partition_geometry.csv"))
        all_summaries.extend(read_csv(directory / "line_search_summary.csv"))

    global_block_rows = [
        row
        for row in all_updates
        if row.get("candidate") == "block4"
        and row.get("partition_kind") == "global_balanced_partition"
    ]
    control_rows = [
        row
        for row in all_updates
        if row.get("candidate") in {"none", "diag", "dense_full"}
        and row.get("partition_kind")
    ]
    within_rows = [
        row
        for row in all_updates
        if row.get("candidate") == "block4"
        and row.get("partition_kind") == "within_block_control"
    ]
    if not global_block_rows or not control_rows or not within_rows:
        raise RuntimeError("required invariance evidence is absent")

    global_drifts = [float(row["update_relative_drift"]) for row in global_block_rows]
    control_drifts = [float(row["update_relative_drift"]) for row in control_rows]
    within_drifts = [float(row["update_relative_drift"]) for row in within_rows]
    checkpoint_rows = []
    stage_medians = []
    minimum_rows = contract["classification_thresholds"][
        "minimum_global_rows_per_checkpoint"
    ]
    for label in ("early", "late"):
        selected = [
            float(row["update_relative_drift"])
            for row in global_block_rows
            if row["checkpoint_label"] == label
        ]
        input_checks[f"{label}_minimum_global_rows"] = len(selected) >= minimum_rows
        stage_medians.append(statistics.median(selected))
        checkpoint_rows.append(
            {
                "checkpoint_label": label,
                "checkpoint_step": contract["checkpoints"][label]["step"],
                **{
                    f"global_block4_update_drift_{key}": value
                    for key, value in summarize(selected).items()
                },
            }
        )

    classification, decision_statistics = classify(
        statistics.median(global_drifts),
        max(control_drifts + within_drifts),
        contract["classification_thresholds"],
        stage_medians,
    )

    partition_summary_rows = []
    for label in ("early", "late"):
        for kind in ("identity", "within_block_control", "global_balanced_partition"):
            selected = [
                float(row["off_block_energy_fraction"])
                for row in all_partitions
                if row["checkpoint_label"] == label
                and row["partition_kind"] == kind
            ]
            partition_summary_rows.append(
                {
                    "checkpoint_label": label,
                    "partition_kind": kind,
                    **{
                        f"off_block_energy_fraction_{key}": value
                        for key, value in summarize(selected).items()
                    },
                }
            )

    shadow_spread_rows = []
    grouped = {}
    for row in all_summaries:
        if (
            row["scope"] != "grouped"
            or not row["candidate"].startswith("block4_")
        ):
            continue
        key = (
            row["checkpoint_label"],
            row["repeat"],
            row["direction"],
            row["build_split"],
            row["eval_split"],
        )
        grouped.setdefault(key, []).append(float(row["best_relative_loss_delta"]))
    for key, values in sorted(grouped.items()):
        shadow_spread_rows.append(
            {
                "checkpoint_label": key[0],
                "repeat": key[1],
                "direction": key[2],
                "build_split": key[3],
                "eval_split": key[4],
                "partition_candidates": len(values),
                "best_relative_loss_delta_minimum": min(values),
                "best_relative_loss_delta_maximum": max(values),
                "best_relative_loss_delta_range": max(values) - min(values),
            }
        )

    passed = all(input_checks.values())
    if not passed:
        classification = "invalid"
    evidence = {
        "classification": classification,
        "decision_statistics": decision_statistics,
        "global_block4_update_drift": summarize(global_drifts),
        "equivariant_control_update_drift": summarize(control_drifts),
        "within_block_control_update_drift": summarize(within_drifts),
        "checkpoint_summaries": checkpoint_rows,
        "interpretation": {
            "block4_is_primary_baseline": False,
            "block4_is_original_newton_muon": False,
            "official_original_newton_muon_control": "newton_full",
            "claim_if_supported": (
                "A contiguous four-block LLaMA down-projection approximation "
                "depends on an arbitrary hidden-neuron coordinate partition."
            ),
            "claim_not_authorized": (
                "block4 underperforms or outperforms primary optimizers in "
                "full training"
            ),
        },
    }
    write_json(output / "classification.json", evidence)
    write_json(
        output / "input_audit.json",
        {
            "run_dir": str(run_dir),
            "contract": str(args.contract.resolve()),
            "contract_sha256": contract_sha,
            "checks": input_checks,
            "passed": passed,
            "worker_checks": checks_payloads,
        },
    )
    write_csv(output / "checkpoint_summary.csv", checkpoint_rows)
    write_csv(output / "partition_energy_summary.csv", partition_summary_rows)
    write_csv(output / "shadow_partition_spread.csv", shadow_spread_rows)

    report = [
        "# LLaMA block-partition invariance audit",
        "",
        f"- Integrity passed: `{str(passed).lower()}`",
        f"- Classification: `{classification}`",
        (
            "- Pooled median mapped-back block4 update drift: "
            f"`{statistics.median(global_drifts):.6g}`"
        ),
        (
            "- Maximum equivariant-control drift: "
            f"`{max(control_drifts + within_drifts):.6g}`"
        ),
        (
            "- Effect/control multiple: "
            f"`{decision_statistics['effect_to_control_multiple']:.6g}`"
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "This audit tests whether a contiguous four-way partition of the "
            "5504-dimensional SwiGLU hidden coordinate is invariant to "
            "function-preserving neuron permutations. It is not a training "
            "comparison and does not promote block4 to a primary baseline."
        ),
        "",
        (
            "For LLaMA, `newton_full` remains the original Newton–Muon-family "
            "control; `muon` remains the optimizer baseline. Selective `none` "
            "and `diag` are each compared with those controls."
        ),
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    artifacts = sorted(path.name for path in output.iterdir())
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "worker_version": WORKER_VERSION,
        "worker_sha256": sha256_file(WORKER),
        "contract_version": CONTRACT_VERSION,
        "passed": passed,
        "classification": classification,
        "contract_sha256": contract_sha,
        "input_manifests": manifests,
        "global_block4_rows": len(global_block_rows),
        "control_rows": len(control_rows),
        "within_block_rows": len(within_rows),
        "scientific_result_used_for_integrity_pass": False,
        "artifacts": artifacts,
    }
    write_json(output / "llama_block_audit_analysis_manifest.json", manifest)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
