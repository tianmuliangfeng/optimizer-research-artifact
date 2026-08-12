#!/usr/bin/env python3
"""Validate and aggregate all MDP-04 streaming replay units."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

import numpy as np


SCRIPT_VERSION = "2026-08-03.4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def as_float(row: dict[str, Any], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite {field}: {value}")
    return value


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def pooled_ratio(rows: list[dict[str, Any]], prefix: str) -> float:
    before = [as_float(row, f"{prefix}_fro_before") for row in rows]
    delta = [as_float(row, f"{prefix}_delta_fro") for row in rows]
    return math.sqrt(sum(value * value for value in delta)) / max(
        math.sqrt(sum(value * value for value in before)), 1.0e-30
    )


def pooled_cosine(rows: list[dict[str, Any]], prefix: str, cosine_field: str) -> float:
    before = [as_float(row, f"{prefix}_fro_before") for row in rows]
    after = [as_float(row, f"{prefix}_fro_after") for row in rows]
    dots = [
        as_float(row, cosine_field) * before[index] * after[index]
        for index, row in enumerate(rows)
    ]
    denominator = math.sqrt(sum(value * value for value in before)) * math.sqrt(
        sum(value * value for value in after)
    )
    return sum(dots) / max(denominator, 1.0e-30)


def unit_attempt(
    run_dir: Path, origin: str, replica: int
) -> tuple[Path, dict[str, Any]]:
    selection_path = run_dir / "formal" / origin / f"replica_{replica}" / "unit_selection.json"
    selection = read_json(selection_path)
    attempt = (selection_path.parent / selection["selected_attempt"]).resolve()
    manifest_path = attempt / "stream_unit_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("passed") is not True:
        raise RuntimeError(f"selected unit is not passed: {manifest_path}")
    if sha256_file(manifest_path) != selection["manifest_sha256"]:
        raise RuntimeError(f"selected unit manifest hash mismatch: {manifest_path}")
    return attempt, manifest


def validate_slice(
    npz_path: Path,
    metadata: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    if sha256_file(npz_path) != metadata["npz_sha256"]:
        raise RuntimeError(f"validation slice hash mismatch: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        covariance_before = data["covariance_before"].astype(np.float64)
        covariance_after = data["covariance_after"].astype(np.float64)
        ridge_before = as_float(row, "ridge_before")
        ridge_after = as_float(row, "ridge_after")
        identity = np.eye(covariance_before.shape[0], dtype=np.float64)
        a_before = covariance_before + ridge_before * identity
        a_after = covariance_after + ridge_after * identity
        inverse_before = np.linalg.inv(a_before)
        inverse_after = np.linalg.inv(a_after)
        delta_a = a_after - a_before
        lhs = inverse_after - inverse_before
        rhs = -inverse_after @ delta_a @ inverse_before
        resolvent = np.linalg.norm(lhs - rhs) / max(
            np.linalg.norm(lhs) + np.linalg.norm(rhs), 1.0e-30
        )
        inverse_residual_before = np.linalg.norm(a_before @ inverse_before - identity) / max(
            np.linalg.norm(identity), 1.0e-30
        )
        inverse_residual_after = np.linalg.norm(a_after @ inverse_after - identity) / max(
            np.linalg.norm(identity), 1.0e-30
        )
        update_before = data["runtime_ns5_update_before"].astype(np.float64)
        update_after = data["runtime_ns5_update_after"].astype(np.float64)

        def polar(value: np.ndarray) -> np.ndarray:
            left, _, right = np.linalg.svd(value, full_matrices=False)
            return left @ right

        polar_before = polar(update_before)
        polar_after = polar(update_after)
        polar_change = np.linalg.norm(polar_after - polar_before) / max(
            np.linalg.norm(polar_before), 1.0e-30
        )
    return {
        "schema_version": "mdp04_validation_slice_result_v1",
        "origin": metadata["origin"],
        "data_replica": int(metadata["data_replica"]),
        "event_id": metadata["event_id"],
        "layer_index": int(metadata["layer_index"]),
        "coordinate_count": int(covariance_before.shape[0]),
        "float64_slice_condition_before": float(np.linalg.cond(a_before)),
        "float64_slice_condition_after": float(np.linalg.cond(a_after)),
        "float64_slice_inverse_residual_before": float(inverse_residual_before),
        "float64_slice_inverse_residual_after": float(inverse_residual_after),
        "float64_slice_resolvent_relative_residual": float(resolvent),
        "float64_slice_svd_polar_change_on_ns5_update_slice": float(polar_change),
        "paper_empirical_claim_eligible": False,
        "warning": metadata["warning"],
    }


def outcome_tables(
    source_run: Path,
    origins: list[str],
    replicas: list[int],
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str], dict[str, Any]]]:
    paired = read_csv(source_run / "analysis" / "paired_contrasts.csv")
    auc = read_csv(source_run / "analysis" / "auc_contrasts.csv")
    by_event: dict[tuple[str, int, str], dict[str, Any]] = {}
    wide_rows: list[dict[str, Any]] = []
    for origin in origins:
        for replica in replicas:
            wide: dict[str, Any] = {"origin": origin, "data_replica": replica}
            for event in events:
                contrast = event["accepted_loss_contrast"]
                step = int(event["accepted_loss_step"])
                paired_matches = [
                    row
                    for row in paired
                    if row["checkpoint_cell"] == origin
                    and int(row["data_replica"]) == replica
                    and row["contrast"] == contrast
                    and int(row["optimizer_step"]) == step
                ]
                auc_matches = [
                    row
                    for row in auc
                    if row["checkpoint_cell"] == origin
                    and int(row["data_replica"]) == replica
                    and row["contrast"] == contrast
                ]
                if len(paired_matches) != 1 or len(auc_matches) != 1:
                    raise RuntimeError(
                        f"accepted outcome is not unique: {origin}/{replica}/{event['event_id']}"
                    )
                raw_delta = float(paired_matches[0]["normalized_loss_delta"])
                raw_auc = float(auc_matches[0]["auc_delta"])
                sign = (
                    -1.0
                    if event["loss_harm_orientation"]
                    == "negative_of_normalized_loss_delta"
                    else 1.0
                )
                payload = {
                    "accepted_loss_contrast": contrast,
                    "accepted_loss_step": step,
                    "accepted_normalized_loss_delta": raw_delta,
                    "oriented_loss_harm": sign * raw_delta,
                    "accepted_auc_delta": raw_auc,
                    "oriented_auc_harm": sign * raw_auc,
                }
                by_event[(origin, replica, event["event_id"])] = payload
                prefix = event["event_id"]
                for key, value in payload.items():
                    wide[f"{prefix}_{key}"] = value
            wide_rows.append(wide)
    return wide_rows, by_event


def validate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    source_run = args.source_run.resolve()
    contract_path = args.contract.resolve()
    contract = read_json(contract_path)
    current_contract_sha256 = sha256_file(contract_path)
    allowed_predecessors = {
        str(row["sha256"]): row
        for row in contract.get("resume_compatibility", {}).get(
            "allowed_predecessors", []
        )
        if isinstance(row, dict) and "sha256" in row
    }
    analysis = run_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    source_hash_checks = {
        relative: sha256_file(source_run / relative) == expected
        for relative, expected in contract["accepted_source_artifact_sha256"].items()
    }
    if not all(source_hash_checks.values()):
        raise RuntimeError(f"accepted source hashes failed: {source_hash_checks}")
    origins = list(contract["coverage"]["origins"])
    replicas = [int(value) for value in contract["coverage"]["replicas"]]
    events = list(contract["coverage"]["events"])
    layer_indices = [int(value) for value in contract["coverage"]["layer_indices"]]
    combined: list[dict[str, Any]] = []
    unit_manifests: list[dict[str, Any]] = []
    selected_attempts: list[str] = []
    selected_attempt_paths: list[Path] = []
    unit_contract_lineage: list[dict[str, Any]] = []
    for origin in origins:
        for replica in replicas:
            attempt, manifest = unit_attempt(run_dir, origin, replica)
            unit_manifests.append(manifest)
            selected_attempts.append(str(attempt))
            selected_attempt_paths.append(attempt)
            rows = read_csv(attempt / "refresh_layer_event_metrics.csv")
            if len(rows) != len(events) * len(layer_indices):
                raise RuntimeError(f"wrong row count in {attempt}: {len(rows)}")
            unit_contract_sha256 = str(manifest.get("stream_contract_sha256", ""))
            if unit_contract_sha256 == current_contract_sha256:
                lineage = "current_v4"
            elif unit_contract_sha256 in allowed_predecessors:
                predecessor = allowed_predecessors[unit_contract_sha256]
                if predecessor.get("reuse_scope") != "selected passed units only":
                    raise RuntimeError(
                        f"invalid predecessor reuse scope: {unit_contract_sha256}"
                    )
                legacy_numeric_fields = (
                    "accepted_covariance_before_numeric_match",
                    "accepted_inverse_before_numeric_match",
                    "accepted_covariance_after_numeric_match",
                    "accepted_inverse_after_numeric_match",
                )
                if not all(
                    all(as_bool(row[field]) for field in legacy_numeric_fields)
                    for row in rows
                ):
                    raise RuntimeError(
                        "selected predecessor unit did not pass the stricter v3 "
                        f"numeric gate: {attempt}"
                    )
                for row in rows:
                    for prefix in (
                        "accepted_covariance_before",
                        "accepted_inverse_before",
                        "accepted_covariance_after",
                        "accepted_inverse_after",
                    ):
                        row[f"{prefix}_reference_integrity_passed"] = row[
                            f"{prefix}_numeric_match"
                        ]
                lineage = "inherited_stricter_v3"
            else:
                raise RuntimeError(
                    "selected unit contract is neither current nor an allowed "
                    f"predecessor: {attempt} {unit_contract_sha256}"
                )
            unit_contract_lineage.append(
                {
                    "origin": origin,
                    "data_replica": replica,
                    "selected_attempt": str(attempt),
                    "stream_contract_sha256": unit_contract_sha256,
                    "lineage": lineage,
                    "passed": True,
                }
            )
            combined.extend(rows)
    keys = [
        (row["origin"], int(row["data_replica"]), row["event_id"], int(row["layer_index"]))
        for row in combined
    ]
    expected_keys = [
        (origin, replica, event["event_id"], layer)
        for origin in origins
        for replica in replicas
        for event in events
        for layer in layer_indices
    ]
    if sorted(keys) != sorted(expected_keys) or len(set(keys)) != len(keys):
        raise RuntimeError("combined layer-event key coverage is not exact")
    hard = contract["hard_gates"]
    boolean_fields = [
        "all_full_state_values_finite",
        "actual_preconditioned_gradient_fingerprint_match",
        "actual_ns_input_fingerprint_match",
        "actual_ns_output_fingerprint_match",
        "accepted_covariance_before_reference_integrity_passed",
        "accepted_inverse_before_reference_integrity_passed",
        "accepted_covariance_after_reference_integrity_passed",
        "accepted_inverse_after_reference_integrity_passed",
    ]
    boolean_gate = all(
        all(as_bool(row[field]) for field in boolean_fields) for row in combined
    )
    maxima = {
        "covariance_refresh_identity_relative_residual": max(
            as_float(row, "covariance_refresh_identity_relative_residual")
            for row in combined
        ),
        "k_asymmetry": max(
            max(
                as_float(row, "k_asymmetry_before"),
                as_float(row, "k_asymmetry_after"),
            )
            for row in combined
        ),
        "inverse_asymmetry": max(
            max(
                as_float(row, "inverse_asymmetry_before"),
                as_float(row, "inverse_asymmetry_after"),
            )
            for row in combined
        ),
        "runtime_inverse_backward_residual": max(
            max(
                as_float(row, "runtime_inverse_backward_residual_before"),
                as_float(row, "runtime_inverse_backward_residual_after"),
            )
            for row in combined
        ),
        "runtime_resolvent_relative_residual": max(
            as_float(row, "runtime_resolvent_relative_residual")
            for row in combined
        ),
        "accepted_refresh_sample_max_abs_error": max(
            as_float(row, field)
            for row in combined
            for field in (
                "accepted_covariance_before_max_abs_error",
                "accepted_inverse_before_max_abs_error",
                "accepted_covariance_after_max_abs_error",
                "accepted_inverse_after_max_abs_error",
            )
        ),
        "accepted_refresh_sample_max_relative_error": max(
            as_float(row, field)
            for row in combined
            for field in (
                "accepted_covariance_before_max_relative_error",
                "accepted_inverse_before_max_relative_error",
                "accepted_covariance_after_max_relative_error",
                "accepted_inverse_after_max_relative_error",
            )
        ),
    }
    accepted_refresh_diagnostics = {
        "all_17_point_values_within_original_v3_tolerance": all(
            all(
                as_bool(row[field])
                for field in (
                    "accepted_covariance_before_numeric_match",
                    "accepted_inverse_before_numeric_match",
                    "accepted_covariance_after_numeric_match",
                    "accepted_inverse_after_numeric_match",
                )
            )
            for row in combined
        ),
        "all_17_point_fingerprints_exact": all(
            all(
                as_bool(row[field])
                for field in (
                    "accepted_covariance_before_fingerprint_exact",
                    "accepted_inverse_before_fingerprint_exact",
                    "accepted_covariance_after_fingerprint_exact",
                    "accepted_inverse_after_fingerprint_exact",
                )
            )
            for row in combined
        ),
        "hard_gate": False,
        "reason": (
            "fresh torch.compile/Triton processes are not elementwise portable; "
            "metadata, sample count and finiteness remain hard-gated"
        ),
    }
    numeric_gate_checks = {
        "covariance_refresh_identity": maxima[
            "covariance_refresh_identity_relative_residual"
        ]
        <= float(hard["covariance_refresh_identity_relative_residual_max"]),
        "k_asymmetry": maxima["k_asymmetry"]
        <= float(hard["k_asymmetry_relative_max"]),
        "inverse_asymmetry": maxima["inverse_asymmetry"]
        <= float(hard["inverse_asymmetry_relative_max"]),
        "runtime_inverse_backward_residual": maxima[
            "runtime_inverse_backward_residual"
        ]
        <= float(hard["runtime_inverse_backward_residual_max"]),
        "runtime_resolvent_relative_residual": maxima[
            "runtime_resolvent_relative_residual"
        ]
        <= float(hard["runtime_resolvent_relative_residual_max"]),
    }

    primary_metrics = [
        "relative_k_fro_change",
        "relative_a_fro_change",
        "relative_runtime_inverse_fro_change",
        "matched_g_preconditioned_relative_change",
        "runtime_ns5_update_relative_change",
        "condition_proxy_before",
        "condition_proxy_after",
        "runtime_resolvent_relative_residual",
    ]
    summary_rows: list[dict[str, Any]] = []
    for origin in origins:
        for replica in replicas:
            for event in events:
                rows = [
                    row
                    for row in combined
                    if row["origin"] == origin
                    and int(row["data_replica"]) == replica
                    and row["event_id"] == event["event_id"]
                ]
                if len(rows) != len(layer_indices):
                    raise RuntimeError("unit-event does not have 18 layers")
                summary: dict[str, Any] = {
                    "schema_version": "mdp04_refresh_unit_event_summary_v1",
                    "origin": origin,
                    "data_replica": replica,
                    "event_id": event["event_id"],
                    "completed_step": int(event["completed_step"]),
                    "layer_count": len(rows),
                    "statistical_unit": "origin_replica; layers are nested repeated measurements",
                }
                for metric in primary_metrics:
                    values = [as_float(row, metric) for row in rows]
                    summary[f"{metric}_layer_median"] = statistics.median(values)
                    summary[f"{metric}_layer_mean"] = statistics.fmean(values)
                    summary[f"{metric}_layer_p90"] = percentile(values, 0.9)
                    summary[f"{metric}_layer_max"] = max(values)
                summary["matched_g_preconditioned_pooled_fro_ratio"] = pooled_ratio(
                    rows, "matched_g_preconditioned"
                )
                summary["matched_g_preconditioned_pooled_cosine"] = pooled_cosine(
                    rows,
                    "matched_g_preconditioned",
                    "matched_g_preconditioned_cosine",
                )
                summary["runtime_ns5_update_pooled_fro_ratio"] = pooled_ratio(
                    rows, "runtime_ns5_update"
                )
                summary["runtime_ns5_update_pooled_cosine"] = pooled_cosine(
                    rows, "runtime_ns5_update", "runtime_ns5_update_cosine"
                )
                summary_rows.append(summary)

    outcome_rows, outcomes = outcome_tables(source_run, origins, replicas, events)
    joined_rows: list[dict[str, Any]] = []
    for summary in summary_rows:
        outcome = outcomes[
            (
                summary["origin"],
                int(summary["data_replica"]),
                summary["event_id"],
            )
        ]
        joined_rows.append({**summary, **outcome})

    slice_rows: list[dict[str, Any]] = []
    by_key = {
        (row["origin"], int(row["data_replica"]), row["event_id"], int(row["layer_index"])): row
        for row in combined
    }
    for attempt in selected_attempt_paths:
        for manifest_path in (attempt / "validation_slices").glob("*.json"):
            metadata = read_json(manifest_path)
            key = (
                metadata["origin"],
                int(metadata["data_replica"]),
                metadata["event_id"],
                int(metadata["layer_index"]),
            )
            slice_rows.append(
                validate_slice(
                    manifest_path.with_name(metadata["npz"]), metadata, by_key[key]
                )
            )
    expected_slices = (
        len(contract["validation_slices"]["events"])
        * len(contract["validation_slices"]["layers"])
    )
    slice_gate = len(slice_rows) == expected_slices and all(
        row["float64_slice_inverse_residual_before"] <= 1.0e-10
        and row["float64_slice_inverse_residual_after"] <= 1.0e-10
        and row["float64_slice_resolvent_relative_residual"] <= 1.0e-10
        for row in slice_rows
    )

    write_csv(analysis / "refresh_layer_event_metrics.csv", combined)
    write_csv(analysis / "refresh_unit_event_summary.csv", summary_rows)
    write_csv(analysis / "refresh_unit_outcomes.csv", outcome_rows)
    write_csv(analysis / "refresh_unit_event_joined.csv", joined_rows)
    write_csv(analysis / "refresh_validation_slices.csv", slice_rows)
    large_files = [
        str(path)
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.stat().st_size
        > int(contract["transport"]["single_file_size_limit_bytes"])
    ]
    checks = {
        "accepted_source_hashes": all(source_hash_checks.values()),
        "unit_count": len(unit_manifests) == len(origins) * len(replicas),
        "all_units_passed": all(manifest["passed"] is True for manifest in unit_manifests),
        "layer_event_rows": len(combined)
        == int(contract["coverage"]["expected_layer_event_rows"]),
        "unit_event_rows": len(summary_rows)
        == int(contract["coverage"]["expected_unit_event_rows"]),
        "unit_outcome_rows": len(outcome_rows) == len(origins) * len(replicas),
        "boolean_integrity_gates": boolean_gate,
        "numeric_integrity_gates": all(numeric_gate_checks.values()),
        "validation_slice_gate": slice_gate,
        "no_large_persisted_files": not large_files,
        "unit_contract_lineage": len(unit_contract_lineage)
        == len(origins) * len(replicas)
        and all(row["passed"] is True for row in unit_contract_lineage),
    }
    passed = all(checks.values())
    artifacts = [
        "refresh_layer_event_metrics.csv",
        "refresh_unit_event_summary.csv",
        "refresh_unit_outcomes.csv",
        "refresh_unit_event_joined.csv",
        "refresh_validation_slices.csv",
    ]
    manifest = {
        "schema_version": "mdp04_formal_stream_manifest_v1",
        "script_version": SCRIPT_VERSION,
        "stream_contract": str(contract_path),
        "stream_contract_sha256": sha256_file(contract_path),
        "source_run": str(source_run),
        "source_hash_checks": source_hash_checks,
        "selected_attempts": selected_attempts,
        "rows": {
            "layer_event": len(combined),
            "unit_event": len(summary_rows),
            "unit_outcome": len(outcome_rows),
            "validation_slice": len(slice_rows),
        },
        "maxima": maxima,
        "accepted_refresh_diagnostics": accepted_refresh_diagnostics,
        "unit_contract_lineage": unit_contract_lineage,
        "unit_contract_lineage_counts": {
            label: sum(1 for row in unit_contract_lineage if row["lineage"] == label)
            for label in ("current_v4", "inherited_stricter_v3")
        },
        "numeric_gate_checks": numeric_gate_checks,
        "large_files": large_files,
        "statistical_unit": "12 nested origin-replica units; 18 layers are aggregated within unit-event",
        "accepted_loss_outcomes_reused_not_recomputed": True,
        "raw_full_matrices_persisted": False,
        "timing_eligible_for_paper": False,
        "artifacts": {
            name: sha256_file(analysis / name) for name in artifacts
        },
        "checks": checks,
        "passed": passed,
    }
    atomic_json(analysis / "formal_stream_manifest.json", manifest)
    atomic_json(
        run_dir / "status.json",
        {
            "status": "passed" if passed else "integrity_failed",
            "script_version": SCRIPT_VERSION,
        },
    )
    if not passed:
        raise SystemExit(2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    manifest = validate(parse_args())
    print(
        "MDP-04 formal stream validation passed: "
        f"rows={manifest['rows']['layer_event']} "
        f"units={len(manifest['selected_attempts'])}"
    )


if __name__ == "__main__":
    main()
