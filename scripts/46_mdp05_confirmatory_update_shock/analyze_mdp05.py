#!/usr/bin/env python3
"""Validate, aggregate, and run the frozen MDP-05 confirmatory analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import protocol as P


SCRIPT_VERSION = "2026-08-04.2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
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
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f(row: dict[str, Any], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}: {value}")
    return value


def aggregate_layers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_metrics = (
        "relative_k_fro_change",
        "relative_a_fro_change",
        "relative_runtime_inverse_fro_change",
        "condition_proxy_after",
        "runtime_resolvent_relative_residual",
        "matched_g_preconditioned_relative_change",
        "runtime_ns5_update_relative_change",
    )
    result: dict[str, Any] = {}
    for metric in scalar_metrics:
        values = [f(row, metric) for row in rows]
        result[f"{metric}_layer_median"] = P.quantile(values, 0.5)
        result[f"{metric}_layer_mean"] = sum(values) / len(values)
        result[f"{metric}_layer_p90"] = P.quantile(values, 0.9)
        result[f"{metric}_layer_max"] = max(values)
    pooled_specs = (
        (
            "matched_g_preconditioned",
            "matched_g_preconditioned_fro_before",
            "matched_g_preconditioned_fro_after",
            "matched_g_preconditioned_delta_fro",
            "matched_g_preconditioned_cosine",
        ),
        (
            "runtime_ns5_update",
            "runtime_ns5_update_fro_before",
            "runtime_ns5_update_fro_after",
            "runtime_ns5_update_delta_fro",
            "runtime_ns5_update_cosine",
        ),
    )
    for prefix, before, after, delta, cosine in pooled_specs:
        before_sq = sum(f(row, before) ** 2 for row in rows)
        after_sq = sum(f(row, after) ** 2 for row in rows)
        delta_sq = sum(f(row, delta) ** 2 for row in rows)
        dot = sum(
            f(row, cosine) * f(row, before) * f(row, after) for row in rows
        )
        result[f"{prefix}_pooled_fro_ratio"] = math.sqrt(delta_sq) / max(
            math.sqrt(before_sq), 1.0e-30
        )
        result[f"{prefix}_pooled_cosine"] = dot / max(
            math.sqrt(before_sq * after_sq), 1.0e-30
        )
    return result


def evaluation_map(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int], float]:
    result: dict[tuple[str, int], float] = {}
    for row in rows:
        key = (str(row["arm"]), int(row["optimizer_step"]))
        if key in result:
            raise RuntimeError(f"duplicate evaluation row: {key}")
        result[key] = f(row, "normalized_loss")
    return result


def trapezoid_mean(values: dict[int, float], steps: list[int]) -> float:
    ordered = sorted(int(step) for step in steps)
    if len(ordered) < 2:
        raise ValueError("AUC requires at least two steps")
    area = 0.0
    for left, right in zip(ordered, ordered[1:]):
        area += (right - left) * (values[left] + values[right]) / 2.0
    return area / (ordered[-1] - ordered[0])


def outcome_for_event(
    evaluations: dict[tuple[str, int], float], event: dict[str, Any]
) -> dict[str, float]:
    left = str(event["left_arm"])
    right = str(event["right_arm"])
    endpoint = int(event["endpoint_step"])
    steps = [int(value) for value in event["auc_steps"]]
    left_values = {step: evaluations[(left, step)] for step in steps}
    right_values = {step: evaluations[(right, step)] for step in steps}
    return {
        "oriented_endpoint_loss_harm": evaluations[(left, endpoint)]
        - evaluations[(right, endpoint)],
        "oriented_auc_harm": trapezoid_mean(left_values, steps)
        - trapezoid_mean(right_values, steps),
    }


def loo_spearman(
    rows: list[dict[str, Any]], x_field: str, y_field: str
) -> list[float]:
    values = []
    for omitted in P.ORIGINS:
        selected = [row for row in rows if row["origin"] != omitted]
        values.append(
            P.spearman(
                [f(row, x_field) for row in selected],
                [f(row, y_field) for row in selected],
            )
        )
    return values


def verify_unit(
    attempt: Path,
    origin: str,
    replica: int,
    contract_sha: str,
    execution_sha: str,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    selection = P.read_json(attempt.parent / "unit_selection.json")
    manifest_path = attempt / "mdp05_unit_manifest.json"
    manifest = P.read_json(manifest_path)
    status = P.read_json(attempt / "mdp05_status.json")
    log_seal = P.read_json(attempt / "worker_log_seal.json")
    scientific_hashes = {
        name: sha256_file(attempt / name) == expected
        for name, expected in manifest["scientific_artifact_sha256"].items()
    }
    checks = {
        "selection_passed": selection.get("passed") is True,
        "selection_attempt": selection.get("selected_attempt") == attempt.name,
        "selection_manifest": selection.get("manifest_sha256")
        == sha256_file(manifest_path),
        "manifest_passed": manifest.get("passed") is True,
        "identity": manifest.get("origin") == origin
        and int(manifest.get("data_replica", -1)) == replica,
        "contract": manifest.get("mdp05_contract_sha256") == contract_sha,
        "execution_contract": manifest.get("execution_contract_sha256")
        == execution_sha,
        "rows": int(manifest.get("layer_event_rows", -1)) == 36,
        "source_outcomes_not_read": manifest.get("source_experiment_outcomes_read")
        is False,
        "status": status.get("status") == "passed",
        "scientific_hashes": all(scientific_hashes.values()),
        "log_excluded": "worker.log"
        not in manifest.get("scientific_artifact_sha256", {}),
        "log_sealed_after_exit": log_seal.get("sealed_after_worker_exit") is True,
        "log_hash": log_seal.get("sha256") == sha256_file(attempt / "worker.log"),
    }
    layer_rows = read_csv(attempt / "mdp05_refresh_layer_metrics.csv")
    evaluation_rows = read_csv(attempt / "evaluation.csv")
    observed_layer_keys = {
        (row["event_id"], int(row["layer_index"])) for row in layer_rows
    }
    checks["layer_coverage"] = observed_layer_keys == {
        (event, layer) for event in P.EVENTS for layer in range(18)
    }
    checks["evaluation_rows"] = len(evaluation_rows) == 18
    return (
        {
            "origin": origin,
            "data_replica": replica,
            "attempt": str(attempt),
            "checks": checks,
            "scientific_hashes": scientific_hashes,
            "passed": all(checks.values()),
        },
        layer_rows,
        evaluation_rows,
    )


def analyze(run_dir: Path, contract_path: Path) -> dict[str, Any]:
    protocol = P.read_json(contract_path)
    protocol_checks = P.validate_protocol(protocol)
    if not all(protocol_checks.values()):
        raise RuntimeError(f"protocol failed: {protocol_checks}")
    contract_sha = sha256_file(contract_path)
    run_identity = P.read_json(run_dir / "run_identity.json")
    execution_path = run_dir / "sealed" / "derived_execution_contract.json"
    execution_sha = sha256_file(execution_path)
    offset_certificate = P.read_json(run_dir / "sealed" / "offset_collision_certificate.json")
    sealed = run_dir / "sealed"
    repair_activation_path = sealed / "source_repair_activation.json"
    repair_activation: dict[str, Any] | None = None
    plan_path = sealed / "formal_job_plan.json"
    if repair_activation_path.is_file():
        repair_activation = P.read_json(repair_activation_path)
        if repair_activation.get("passed") is not True:
            raise RuntimeError("source repair activation did not pass")
        plan_path = sealed / str(repair_activation["active_plan"])
        if sha256_file(plan_path) != repair_activation["active_plan_sha256"]:
            raise RuntimeError("active repair formal plan hash mismatch")
    plan = P.read_json(plan_path)
    analysis = run_dir / "analysis"
    analysis.mkdir(exist_ok=True)

    inventory: list[dict[str, Any]] = []
    all_layer_rows: list[dict[str, Any]] = []
    unit_event_rows: list[dict[str, Any]] = []
    source_reference = P.read_json(run_dir / "sealed" / "source_data_reference.json")
    source_train_hashes = set(source_reference["training_first_batch_hashes"])
    source_val_hashes = set(source_reference["validation_batch_hashes"])
    observed_train_hashes: set[str] = set()
    observed_val_hashes: set[str] = set()
    float64_rows: list[dict[str, Any]] = []

    for origin in P.ORIGINS:
        for replica in protocol["design"]["formal_data_replicas"]:
            unit_dir = run_dir / "formal" / origin / f"replica_{replica}"
            selection = P.read_json(unit_dir / "unit_selection.json")
            attempt = unit_dir / selection["selected_attempt"]
            audit, layer_rows, evaluation_rows = verify_unit(
                attempt, origin, int(replica), contract_sha, execution_sha
            )
            inventory.append(audit)
            all_layer_rows.extend(layer_rows)
            stream = P.read_json(attempt / "training_stream_contract.json")
            heldout = P.read_json(attempt / "heldout_batch_contract.json")
            observed_train_hashes.update(
                {stream["first_x_sha256"], stream["first_y_sha256"]}
            )
            for section in ("build", "evaluation"):
                for batch in heldout[section]["hashes"]:
                    observed_val_hashes.update(
                        {batch["x_sha256"], batch["y_sha256"]}
                    )
            for diagnostic_path in sorted(
                (attempt / "validation_slices").glob("*_float64.json")
            ):
                float64_rows.append(P.read_json(diagnostic_path))
            evaluations = evaluation_map(evaluation_rows)
            for event in protocol["event_outcomes"]:
                event_rows = [
                    row for row in layer_rows if row["event_id"] == event["event_id"]
                ]
                unit_event_rows.append(
                    {
                        "origin": origin,
                        "data_replica": int(replica),
                        "event_id": event["event_id"],
                        **aggregate_layers(event_rows),
                        **outcome_for_event(evaluations, event),
                    }
                )

    expected_units = {
        (origin, int(replica))
        for origin in P.ORIGINS
        for replica in protocol["design"]["formal_data_replicas"]
    }
    expected_events = {
        (origin, int(replica), event)
        for origin, replica in expected_units
        for event in P.EVENTS
    }
    observed_events = {
        (row["origin"], int(row["data_replica"]), row["event_id"])
        for row in unit_event_rows
    }
    integrity_checks = {
        "run_identity": run_identity.get("experiment") == "MDP-05"
        and run_identity.get("contract_sha256") == contract_sha,
        "protocol": all(protocol_checks.values()),
        "offset_certificate": offset_certificate.get("passed") is True,
        "sealed_plan": plan.get("passed") is True
        and int(plan.get("formal_units", -1)) == 12,
        "units_12": len(inventory) == 12
        and {(row["origin"], row["data_replica"]) for row in inventory}
        == expected_units,
        "units_passed": all(row["passed"] for row in inventory),
        "layer_rows_432": len(all_layer_rows) == 432,
        "unit_event_rows_24": len(unit_event_rows) == 24,
        "unit_event_coverage": observed_events == expected_events,
        "new_training_hashes": not observed_train_hashes.intersection(
            source_train_hashes
        ),
        "new_validation_hashes": not observed_val_hashes.intersection(
            source_val_hashes
        ),
        "float64_slice_diagnostics_8": len(float64_rows) == 8,
        "float64_slice_diagnostics_finite": len(float64_rows) == 8
        and all(row.get("all_values_finite") is True for row in float64_rows),
        "all_values_finite": all(
            str(row["all_full_state_values_finite"]).lower() == "true"
            for row in all_layer_rows
        ),
        "resolvent_not_used_as_gate": protocol["hard_gates"][
            "runtime_resolvent_relative_residual_is_hard_gate"
        ]
        is False,
    }

    primary_rows: list[dict[str, Any]] = []
    for event in P.EVENTS:
        selected = [row for row in unit_event_rows if row["event_id"] == event]
        for mediator in P.PRIMARY_MEDIATORS:
            permutation = P.exact_within_origin_randomization_p(
                selected, mediator, "oriented_endpoint_loss_harm"
            )
            loo = loo_spearman(
                selected, mediator, "oriented_endpoint_loss_harm"
            )
            primary_rows.append(
                {
                    "event_id": event,
                    "mediator": mediator,
                    "unit_count": len(selected),
                    "spearman_rho": permutation["observed_spearman_rho"],
                    "within_origin_centered_pearson_r": P.within_origin_centered(
                        selected, mediator, "oriented_endpoint_loss_harm"
                    ),
                    "leave_one_origin_out_spearman_min": min(loo),
                    "leave_one_origin_out_spearman_max": max(loo),
                    "leave_one_origin_out_values": json.dumps(loo),
                    "randomization_permutations": permutation["permutations"],
                    "one_sided_exact_p": permutation["one_sided_exact_p"],
                }
            )
    adjusted = P.holm_adjust([row["one_sided_exact_p"] for row in primary_rows])
    alpha = float(protocol["statistics"]["alpha"])
    for row, adjusted_p in zip(primary_rows, adjusted):
        row["holm_adjusted_p"] = adjusted_p
        row["direction_passed"] = float(row["spearman_rho"]) > 0.0
        row["within_origin_passed"] = (
            float(row["within_origin_centered_pearson_r"]) > 0.0
        )
        row["leave_one_origin_out_passed"] = (
            float(row["leave_one_origin_out_spearman_min"]) > 0.0
        )
        row["multiplicity_passed"] = adjusted_p <= alpha
        row["passed"] = all(
            row[field]
            for field in (
                "direction_passed",
                "within_origin_passed",
                "leave_one_origin_out_passed",
                "multiplicity_passed",
            )
        )

    supportive_rows: list[dict[str, Any]] = []
    for event in P.EVENTS:
        selected = [row for row in unit_event_rows if row["event_id"] == event]
        for mediator in P.PRIMARY_MEDIATORS:
            supportive_rows.append(
                {
                    "event_id": event,
                    "mediator": mediator,
                    "outcome": "oriented_auc_harm",
                    "spearman_rho": P.spearman(
                        [f(row, mediator) for row in selected],
                        [f(row, "oriented_auc_harm") for row in selected],
                    ),
                    "within_origin_centered_pearson_r": P.within_origin_centered(
                        selected, mediator, "oriented_auc_harm"
                    ),
                    "success_gate": False,
                }
            )

    integrity_passed = all(integrity_checks.values())
    claim_success = integrity_passed and all(row["passed"] for row in primary_rows)
    scientific_result = (
        "confirmatory_success"
        if claim_success
        else "partial_or_null"
        if integrity_passed
        else "integrity_failed"
    )
    write_csv(analysis / "unit_inventory.csv", inventory)
    write_csv(analysis / "layer_metrics_all.csv", all_layer_rows)
    write_csv(analysis / "unit_event_joined.csv", unit_event_rows)
    write_csv(analysis / "primary_confirmatory_tests.csv", primary_rows)
    write_csv(analysis / "supportive_auc_associations.csv", supportive_rows)
    write_csv(analysis / "float64_slice_calibration.csv", float64_rows)
    diagnostic = {
        "runtime_resolvent_relative_residual_max": max(
            f(row, "runtime_resolvent_relative_residual")
            for row in all_layer_rows
        ),
        "rows_above_old_mdp04_0p01_threshold": sum(
            f(row, "runtime_resolvent_relative_residual") > 0.01
            for row in all_layer_rows
        ),
        "hard_gate": False,
    }
    P.atomic_json(analysis / "resolvent_diagnostics.json", diagnostic)
    artifacts = sorted(
        path.name
        for path in analysis.iterdir()
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    manifest = {
        "schema_version": "mdp05_analysis_manifest_v1",
        "script_version": SCRIPT_VERSION,
        "contract_sha256": contract_sha,
        "execution_contract_sha256": execution_sha,
        "active_formal_job_plan": plan_path.name,
        "active_formal_job_plan_sha256": sha256_file(plan_path),
        "pre_outcome_source_repair": repair_activation,
        "formal_outcomes_opened_only_after_12_selected_units": True,
        "integrity_checks": integrity_checks,
        "integrity_passed": integrity_passed,
        "claim_success": claim_success,
        "scientific_result": scientific_result,
        "primary_tests": primary_rows,
        "supportive_auc": supportive_rows,
        "resolvent_diagnostics": diagnostic,
        "artifacts": {name: sha256_file(analysis / name) for name in artifacts},
        "passed": integrity_passed,
    }
    P.atomic_json(analysis / "analysis_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = analyze(args.run_dir.resolve(), args.contract.resolve())
        print(
            "MDP-05 analysis "
            f"integrity={manifest['integrity_passed']} "
            f"result={manifest['scientific_result']}",
            flush=True,
        )
        return 0 if manifest["integrity_passed"] else 2
    except Exception as exc:
        print(
            f"MDP-05 analysis failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
