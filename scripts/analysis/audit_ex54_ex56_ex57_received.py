#!/usr/bin/env python3
"""Independently audit received EX54/EX56/EX57 result trees.

The script is intentionally standard-library only.  It checks the sealed artifact
hashes that remain available in the no-checkpoint handoff, recomputes every paired
mean/SD/95% t interval, and emits a compact JSON summary for local acceptance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


T_CRIT_DF2_975 = 4.302652729911275
# EX54/57 preserve full-precision aggregates; EX56 publishes nine decimal places.
# A 5e-9 absolute tolerance is therefore tighter than the last published digit.
ABS_TOL = 5e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=ABS_TOL):
        raise RuntimeError(f"{label}: observed={observed!r} expected={expected!r}")


def sample_stats(values: list[float]) -> dict[str, float]:
    require(len(values) == 3, f"expected three paired seeds, got {len(values)}")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half_width = T_CRIT_DF2_975 * sd / math.sqrt(len(values))
    return {
        "mean": mean,
        "sample_sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def verify_source_snapshot(run_dir: Path) -> int:
    manifest = load_json(run_dir / "source_snapshot" / "source_snapshot_manifest.json")
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "source snapshot file map missing")
    for relative, record in files.items():
        path = run_dir / "source_snapshot" / relative
        require(path.is_file(), f"source snapshot file missing: {relative}")
        require(path.stat().st_size == int(record["bytes"]), f"source bytes mismatch: {relative}")
        require(sha256(path) == record["sha256"], f"source hash mismatch: {relative}")
    return len(files)


def verify_artifact_hashes(run_dir: Path, manifest: dict[str, Any], key: str) -> int:
    records = manifest[key]
    require(isinstance(records, dict) and records, f"{key} hash map missing")
    for name, expected in records.items():
        if name == "frozen_controls.csv":
            path = run_dir / "source_snapshot" / "scripts" / run_dir.parent.name / name
        else:
            path = run_dir / "analysis" / name
        require(path.is_file(), f"sealed artifact missing: {path}")
        require(sha256(path) == expected, f"sealed artifact hash mismatch: {name}")
    return len(records)


def audit_moonlight(run_dir: Path, experiment_id: str, expected_scope: str) -> dict[str, Any]:
    completion = load_json(run_dir / "completion_manifest.json")
    analysis = load_json(run_dir / "analysis" / "analysis_manifest.json")
    verification = load_json(run_dir / "analysis" / "verification_manifest.json")
    selection = load_json(run_dir / "tuning" / "selection.json")
    tuning = load_json(run_dir / "tuning" / "tuning_manifest.json")
    formal_name = "formal_non10b_manifest.json" if experiment_id.startswith("54_") else "formal_manifest.json"
    formal_path = run_dir / "formal" / formal_name
    formal = load_json(formal_path)

    require(completion.get("passed") is True and completion.get("status") == "completed", "completion failed")
    require(completion.get("wandb_required") is False, "unexpected W&B hard requirement")
    require(completion.get("timing_usable") is False, "timing unexpectedly eligible")
    require(analysis.get("passed") is True and analysis.get("scope", expected_scope) == expected_scope, "analysis failed")
    require(verification.get("passed") is True and verification.get("full_checkpoint_hash") is True, "full checkpoint verification failed")
    require(all(value is True for value in verification.get("checks", {}).values()), "verification contains failed checks")
    require(tuning.get("passed") is True and tuning.get("formal_seed_overlap") is False, "tuning/formal isolation failed")
    require(formal.get("passed") is True and all(unit.get("passed") is True for unit in formal.get("units", [])), "formal manifest failed")

    require(sha256(run_dir / "analysis" / "analysis_manifest.json") == completion["analysis_manifest_sha256"], "completion analysis hash mismatch")
    require(sha256(run_dir / "analysis" / "verification_manifest.json") == completion["verification_manifest_sha256"], "completion verification hash mismatch")
    require(sha256(formal_path) == completion["formal_manifest_sha256"], "completion formal hash mismatch")
    require(sha256(run_dir / "tuning" / "selection.json") == analysis["selection_sha256"], "analysis selection hash mismatch")

    sealed_files = verify_artifact_hashes(run_dir, analysis, "files")
    source_files = verify_source_snapshot(run_dir)
    endpoints = load_csv(run_dir / "analysis" / "endpoint_results.csv")
    deltas = load_csv(run_dir / "analysis" / "paired_seed_deltas.csv")
    aggregates = load_csv(run_dir / "analysis" / "paired_contrasts.csv")
    require(len(endpoints) == int(analysis["endpoint_rows"]), "endpoint row count mismatch")
    require(len(deltas) == int(analysis["paired_seed_rows"]), "paired seed row count mismatch")
    require(len(aggregates) == int(analysis["aggregate_contrasts"]), "aggregate contrast count mismatch")

    endpoint_keys: set[tuple[str, str, int]] = set()
    endpoint_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in endpoints:
        key = (row["scale"], row["budget_id"], int(row["seed"]))
        require(key not in endpoint_keys, f"duplicate endpoint: {key}")
        endpoint_keys.add(key)
        endpoint_groups[(row["scale"], row["budget_id"])].append(float(row["final_val_loss"]))
        require(math.isfinite(float(row["final_val_loss"])), f"nonfinite endpoint: {key}")

    delta_groups: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in deltas:
        scale = row.get("scale", "1b")
        key = (scale, row["budget_id"], row["comparator"])
        delta_groups[key].append((int(row["seed"]), float(row["delta_moonlight_minus_comparator"])))

    checked_fields = 0
    contrast_summary: list[dict[str, Any]] = []
    for row in aggregates:
        scale = row.get("scale", "1b")
        key = (scale, row["budget_id"], row["comparator"])
        values = [value for _, value in sorted(delta_groups[key])]
        stats = sample_stats(values)
        for field, source_field in (
            ("mean", "mean_delta_moonlight_minus_comparator"),
            ("sample_sd", "sample_sd"),
            ("ci95_low", "ci95_low"),
            ("ci95_high", "ci95_high"),
        ):
            require_close(stats[field], float(row[source_field]), f"{experiment_id}:{key}:{field}")
            checked_fields += 1
        better = sum(value < 0 for value in values)
        worse = sum(value > 0 for value in values)
        require(better == int(row["moonlight_better_seed_count"]), f"better count mismatch: {key}")
        require(worse == int(row["moonlight_worse_seed_count"]), f"worse count mismatch: {key}")
        checked_fields += 2
        contrast_summary.append({
            "scale": scale,
            "budget_id": row["budget_id"],
            "comparator": row["comparator"],
            **stats,
            "better_seed_count": better,
            "worse_seed_count": worse,
        })

    endpoint_summary = []
    for (scale, budget), values in sorted(endpoint_groups.items()):
        require(len(values) == 3, f"endpoint seed coverage mismatch: {(scale, budget)}")
        endpoint_summary.append({"scale": scale, "budget_id": budget, **sample_stats(values)})

    selected: dict[str, Any]
    if experiment_id.startswith("54_"):
        selected = {
            scale: {
                "cell": payload["selected_cell"],
                "best_observed_cell": payload["best_observed_cell"],
                "center_retained": payload["center_retained"],
            }
            for scale, payload in selection["scales"].items()
        }
    else:
        payload = selection["selected"]
        selected = {
            "1b": {
                "cell": payload["selected_cell"],
                "best_observed_cell": payload["best_observed_cell"],
                "center_retained": payload["center_retained"],
            }
        }

    return {
        "experiment_id": experiment_id,
        "run_id": run_dir.name,
        "passed": True,
        "timing_usable": False,
        "wandb_required": False,
        "formal_units": len(formal["units"]),
        "endpoint_rows": len(endpoints),
        "paired_seed_rows": len(deltas),
        "aggregate_contrasts": len(aggregates),
        "sealed_analysis_files_verified": sealed_files,
        "source_snapshot_files_verified": source_files,
        "independently_recomputed_fields": checked_fields,
        "selection": selected,
        "endpoint_summary": endpoint_summary,
        "paired_contrasts": contrast_summary,
        "analysis_manifest_sha256": sha256(run_dir / "analysis" / "analysis_manifest.json"),
        "verification_manifest_sha256": sha256(run_dir / "analysis" / "verification_manifest.json"),
        "formal_manifest_sha256": sha256(formal_path),
    }


def audit_ex56(run_dir: Path) -> dict[str, Any]:
    analysis = load_json(run_dir / "analysis" / "analysis_manifest.json")
    classification = load_json(run_dir / "analysis" / "classification.json")
    native = load_json(run_dir / "analysis" / "native_verify_full.json")
    handoff = load_json(run_dir / "handoff_manifest.json")
    suite = load_json(run_dir / "suite_status.json")
    require(analysis.get("passed") is True and analysis.get("status") == "completed_valid", "EX56 analysis failed")
    require(all(value is True for value in analysis.get("integrity_checks", {}).values()), "EX56 integrity check failed")
    require(native.get("passed") is True and native.get("full_checkpoint_hash") is True, "EX56 native verify failed")
    require(all(value is True for value in native.get("checks", {}).values()), "EX56 native verify check failed")
    require(handoff.get("passed") is True and handoff.get("status") == "completed", "EX56 handoff failed")
    require(handoff.get("timing_usable") is False and analysis.get("timing_usable") is False, "EX56 timing unexpectedly eligible")
    require(suite.get("passed") is True, "EX56 suite failed")

    for name, expected in analysis["artifacts"].items():
        path = run_dir / "analysis" / name
        require(path.is_file() and sha256(path) == expected, f"EX56 analysis artifact mismatch: {name}")
    source_files = verify_source_snapshot(run_dir)

    endpoints = load_csv(run_dir / "analysis" / "endpoint_results.csv")
    unified = load_csv(run_dir / "analysis" / "unified_endpoint_results.csv")
    contrasts = load_csv(run_dir / "analysis" / "paired_contrasts.csv")
    require(len(endpoints) == int(analysis["primary_endpoints"]) == 9, "EX56 endpoint count mismatch")
    require(len(contrasts) == 12, "EX56 contrast count mismatch")
    require(len(unified) == int(analysis["control_endpoints"]) + len(endpoints), "EX56 unified endpoint count mismatch")

    endpoint_groups: dict[str, list[float]] = defaultdict(list)
    endpoint_keys: set[tuple[str, int]] = set()
    for row in endpoints:
        key = (row["budget_id"], int(row["seed"]))
        require(key not in endpoint_keys, f"EX56 duplicate endpoint: {key}")
        endpoint_keys.add(key)
        endpoint_groups[row["budget_id"]].append(float(row["final_val_loss"]))

    checked_fields = 0
    contrast_summary = []
    for row in contrasts:
        values = [float(row[f"seed{seed}"]) for seed in (2024, 2025, 2026)]
        stats = sample_stats(values)
        for field, source_field in (
            ("mean", "mean_difference"),
            ("sample_sd", "sample_sd"),
            ("ci95_low", "ci95_low"),
            ("ci95_high", "ci95_high"),
        ):
            require_close(stats[field], float(row[source_field]), f"EX56:{row['contrast']}:{field}")
            checked_fields += 1
        better = sum(value < 0 for value in values)
        worse = sum(value > 0 for value in values)
        require(better == int(row["global_diag_better_seeds"]), f"EX56 better count mismatch: {row['contrast']}")
        require(worse == int(row["global_diag_worse_seeds"]), f"EX56 worse count mismatch: {row['contrast']}")
        checked_fields += 2
        contrast_summary.append({
            "budget_id": row["budget_id"],
            "comparator": row["comparator"],
            **stats,
            "better_seed_count": better,
            "worse_seed_count": worse,
        })

    endpoint_summary = []
    for budget, values in sorted(endpoint_groups.items()):
        require(len(values) == 3, f"EX56 endpoint seed coverage mismatch: {budget}")
        endpoint_summary.append({"budget_id": budget, **sample_stats(values)})

    return {
        "experiment_id": "56_llama1b_10b_global_diag",
        "run_id": run_dir.name,
        "passed": True,
        "scientific_classification": classification["classification"],
        "timing_usable": False,
        "wandb_required": False,
        "formal_units": int(analysis["formal_units"]),
        "endpoint_rows": len(endpoints),
        "aggregate_contrasts": len(contrasts),
        "sealed_analysis_files_verified": len(analysis["artifacts"]),
        "source_snapshot_files_verified": source_files,
        "analysis_integrity_checks_verified": len(analysis["integrity_checks"]),
        "independently_recomputed_fields": checked_fields,
        "endpoint_summary": endpoint_summary,
        "paired_contrasts": contrast_summary,
        "analysis_manifest_sha256": sha256(run_dir / "analysis" / "analysis_manifest.json"),
        "native_verify_sha256": sha256(run_dir / "analysis" / "native_verify_full.json"),
        "handoff_manifest_sha256": sha256(run_dir / "handoff_manifest.json"),
    }


def find_contrast(result: dict[str, Any], scale: str, budget: str, comparator: str) -> dict[str, Any]:
    for row in result["paired_contrasts"]:
        if row.get("scale", "1b") == scale and row["budget_id"] == budget and row["comparator"] == comparator:
            return row
    raise RuntimeError(f"contrast not found: {(scale, budget, comparator)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    ex54 = audit_moonlight(
        root / "54_llama_moonlight_multiscale_multibudget" / "20260820T043013+0000",
        "54_llama_moonlight_multiscale_multibudget",
        "non10b",
    )
    ex56 = audit_ex56(root / "56_llama1b_10b_global_diag" / "20260817T065720+0000")
    ex57 = audit_moonlight(
        root / "57_llama1b_10b_moonlight" / "20260820T043038+0000",
        "57_llama1b_10b_moonlight",
        "long_budget",
    )

    replication = []
    for budget in ("tokens_3p2506b", "tokens_6p9694b"):
        left = find_contrast(ex54, "1b", budget, "muon")
        right = find_contrast(ex57, "1b", budget, "muon")
        replication.append({
            "budget_id": budget,
            "ex54_mean_moonlight_minus_muon": left["mean"],
            "ex57_mean_moonlight_minus_muon": right["mean"],
            "same_direction": (left["mean"] > 0) == (right["mean"] > 0),
            "pooling_allowed": False,
        })

    payload = {
        "schema_version": 1,
        "audit_status": "passed",
        "source_root": str(root),
        "experiments": {"54": ex54, "56": ex56, "57": ex57},
        "independent_replication_only": replication,
        "global_boundaries": {
            "wandb_absence_invalidates_results": False,
            "timing_usable": False,
            "ex54_ex57_pooling_allowed": False,
            "checkpoint_bytes_present_locally": False,
            "remote_full_checkpoint_hash_receipts_present": True,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
