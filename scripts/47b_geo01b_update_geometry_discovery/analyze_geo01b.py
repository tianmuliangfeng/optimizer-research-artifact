#!/usr/bin/env python3
"""Analyze the frozen GEO-01B discovery grid without upgrading its evidence tier."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P


PREDICTORS = {
    "norm_only": "norm_only_predictor",
    "first_order": "first_order_predictor",
    "full_taylor": "full_taylor_predictor",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"row {line_number} is not an object: {path}")
            rows.append(value)
    if not rows:
        raise ValueError(f"input has no rows: {path}")
    return rows


def sign_accuracy(predictor: list[float], target: list[float]) -> float:
    return sum(left * right > 0.0 for left, right in zip(predictor, target)) / len(
        target
    )


def predictor_metrics(
    values: list[float], target: list[float], origins: list[str]
) -> dict[str, Any]:
    centered_values = P.centered(values, origins)
    centered_target = P.centered(target, origins)
    loo = {
        omitted: P.spearman(
            [value for value, origin in zip(values, origins) if origin != omitted],
            [value for value, origin in zip(target, origins) if origin != omitted],
        )
        for omitted in sorted(set(origins))
    }
    return {
        "sign_accuracy": sign_accuracy(values, target),
        "pooled_spearman": P.spearman(values, target),
        "origin_centered_spearman": P.spearman(
            centered_values, centered_target
        ),
        "leave_one_origin_out_spearman": loo,
        "minimum_leave_one_origin_out_spearman": min(loo.values()),
    }


def summarize_event(
    rows: list[dict[str, Any]], contract: dict[str, Any], event_id: str
) -> dict[str, Any]:
    selected = [row for row in rows if row["event_id"] == event_id]
    expected = len(P.ORIGINS) * len(P.REPLICAS)
    if len(selected) != expected:
        raise RuntimeError(f"event {event_id} has {len(selected)} rows, expected {expected}")
    selected.sort(key=lambda row: (P.ORIGINS.index(row["origin"]), int(row["data_replica"])))
    origins = [str(row["origin"]) for row in selected]
    target = [float(row["endpoint_normalized_loss_harm"]) for row in selected]
    metrics = {
        name: predictor_metrics(
            [float(row[field]) for row in selected], target, origins
        )
        for name, field in PREDICTORS.items()
    }
    first_errors = [float(row["local_first_relative_error"]) for row in selected]
    full_errors = [float(row["local_taylor_relative_error"]) for row in selected]
    first_median = P.median(first_errors)
    full_median = P.median(full_errors)
    floor = float(contract["analysis"]["finite_floor"])
    local = {
        "first_order_median_relative_error": first_median,
        "full_taylor_median_relative_error": full_median,
        "full_taylor_max_relative_error": max(full_errors),
        "first_order_sign_accuracy": sum(
            row["local_first_sign_match"] is True for row in selected
        )
        / len(selected),
        "full_taylor_sign_accuracy": sum(
            row["local_taylor_sign_match"] is True for row in selected
        )
        / len(selected),
        "curvature_local_error_reduction_fraction": (first_median - full_median)
        / max(first_median, floor),
    }
    gate = contract["discovery_gate"]
    direction_checks = {
        "local_taylor_error": local["full_taylor_median_relative_error"]
        <= float(gate["local_taylor_median_relative_error_max"]),
        "local_taylor_sign": local["full_taylor_sign_accuracy"]
        >= float(gate["local_taylor_sign_accuracy_min"]),
        "short_horizon_sign": metrics["full_taylor"]["sign_accuracy"]
        >= float(gate["short_horizon_full_sign_accuracy_min"]),
        "short_horizon_pooled": metrics["full_taylor"]["pooled_spearman"]
        >= float(gate["short_horizon_full_pooled_spearman_min"]),
        "short_horizon_origin_centered": metrics["full_taylor"][
            "origin_centered_spearman"
        ]
        >= float(gate["short_horizon_full_origin_centered_spearman_min"]),
        "short_horizon_loo": metrics["full_taylor"][
            "minimum_leave_one_origin_out_spearman"
        ]
        >= float(gate["short_horizon_full_min_leave_one_origin_out_spearman"]),
        "beats_norm_origin_centered": metrics["full_taylor"][
            "origin_centered_spearman"
        ]
        - metrics["norm_only"]["origin_centered_spearman"]
        >= float(gate["full_minus_norm_origin_centered_spearman_min"]),
    }
    curvature_checks = {
        "local_error_reduction": local[
            "curvature_local_error_reduction_fraction"
        ]
        >= float(gate["curvature_local_error_reduction_fraction_min"]),
        "short_horizon_increment": metrics["full_taylor"][
            "origin_centered_spearman"
        ]
        - metrics["first_order"]["origin_centered_spearman"]
        >= float(
            gate["curvature_full_minus_first_origin_centered_spearman_min"]
        ),
    }
    return {
        "event_id": event_id,
        "unit_count": len(selected),
        "endpoint_harm_positive_count": sum(value > 0.0 for value in target),
        "endpoint_harm_mean": sum(target) / len(target),
        "endpoint_harm_min": min(target),
        "endpoint_harm_max": max(target),
        "predictors": metrics,
        "local_closure": local,
        "directional_geometry_checks": direction_checks,
        "directional_geometry_supported": all(direction_checks.values()),
        "curvature_increment_checks": curvature_checks,
        "curvature_increment_supported": all(curvature_checks.values()),
    }


def analyze(
    geometry_rows: list[dict[str, Any]],
    outcome_rows: list[dict[str, Any]],
    contract: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    expected_geometry = {
        (origin, replica, event, scope)
        for origin in P.ORIGINS
        for replica in P.REPLICAS
        for event in P.EVENTS
        for scope in P.SCOPES
    }
    observed_geometry = {
        (
            str(row.get("origin")),
            int(row.get("data_replica", -1)),
            str(row.get("event_id")),
            str(row.get("scope_id")),
        )
        for row in geometry_rows
    }
    expected_outcomes = {
        (origin, replica, event)
        for origin in P.ORIGINS
        for replica in P.REPLICAS
        for event in P.EVENTS
    }
    observed_outcomes = {
        (
            str(row.get("origin")),
            int(row.get("data_replica", -1)),
            str(row.get("event_id")),
        )
        for row in outcome_rows
    }
    primary = {
        (str(row["origin"]), int(row["data_replica"]), str(row["event_id"])): row
        for row in geometry_rows
        if row.get("scope_id") == contract["discovery"]["primary_scope"]
    }
    lineage = []
    for row in outcome_rows:
        key = (str(row["origin"]), int(row["data_replica"]), str(row["event_id"]))
        geometry = primary.get(key, {})
        lineage.append(
            math.isclose(
                float(row["norm_only_predictor"]),
                float(geometry.get("relative_direction_fro_norm", math.nan)),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                float(row["first_order_predictor"]),
                float(geometry.get("first_order_alignment", math.nan)),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                float(row["full_taylor_predictor"]),
                float(geometry.get("taylor_actual_delta_loss", math.nan)),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )
    numeric_fields = (
        "norm_only_predictor",
        "first_order_predictor",
        "full_taylor_predictor",
        "local_exact_delta_loss",
        "local_first_relative_error",
        "local_taylor_relative_error",
        "endpoint_normalized_loss_harm",
        "endpoint_raw_loss_harm",
        "trapezoid_normalized_auc_harm",
    )
    checks = {
        "contract": all(P.validate_contract(contract).values()),
        "geometry_row_count": len(geometry_rows) == 96,
        "geometry_grid": observed_geometry == expected_geometry
        and len(observed_geometry) == len(geometry_rows),
        "outcome_row_count": len(outcome_rows) == 24,
        "outcome_grid": observed_outcomes == expected_outcomes
        and len(observed_outcomes) == len(outcome_rows),
        "contract_lineage": all(
            row.get("contract_sha256") == contract_sha256
            for row in geometry_rows
        ),
        "predictor_lineage": all(lineage),
        "numeric_finite": all(
            field in row and math.isfinite(float(row[field]))
            for row in outcome_rows
            for field in numeric_fields
        ),
        "row_integrity": all(
            row.get("all_values_finite") is True
            and row.get("parameters_unchanged") is True
            for row in geometry_rows
        )
        and all(row.get("all_values_finite") is True for row in outcome_rows),
        "discovery_not_claim_eligible": contract["claim_boundary"][
            "discovery_claim_eligible"
        ]
        is False,
    }
    event_results = {
        event: summarize_event(outcome_rows, contract, event) for event in P.EVENTS
    }
    integrity = all(checks.values())
    supported_count = sum(
        row["directional_geometry_supported"] for row in event_results.values()
    )
    curvature_count = sum(
        row["curvature_increment_supported"] for row in event_results.values()
    )
    if not integrity:
        scientific_result = "integrity_failed"
    elif supported_count == len(P.EVENTS):
        scientific_result = "directional_geometry_supported"
    elif supported_count:
        scientific_result = "directional_geometry_partial"
    else:
        scientific_result = "directional_geometry_not_supported"
    curvature_result = (
        "curvature_increment_supported"
        if curvature_count == len(P.EVENTS)
        else "curvature_increment_partial"
        if curvature_count
        else "curvature_increment_not_supported"
    )
    return {
        "schema_version": "geo01b_discovery_analysis_v1",
        "phase": "discovery",
        "checks": checks,
        "integrity_passed": integrity,
        "event_results": event_results,
        "scientific_result": scientific_result,
        "curvature_increment_result": curvature_result,
        "confirmation_candidate": integrity
        and supported_count == len(P.EVENTS)
        and contract["discovery_gate"][
            "both_events_required_for_confirmation_candidate"
        ]
        is True,
        "confirmation_authorized": False,
        "claim_eligible": False,
        "no_confirmatory_p_values": True,
    }


def write_event_csv(path: Path, analysis: dict[str, Any]) -> None:
    fields = [
        "event_id",
        "unit_count",
        "endpoint_harm_positive_count",
        "endpoint_harm_mean",
        "full_pooled_spearman",
        "full_origin_centered_spearman",
        "norm_origin_centered_spearman",
        "first_origin_centered_spearman",
        "full_sign_accuracy",
        "local_taylor_median_relative_error",
        "directional_geometry_supported",
        "curvature_increment_supported",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in P.EVENTS:
            row = analysis["event_results"][event]
            writer.writerow(
                {
                    "event_id": event,
                    "unit_count": row["unit_count"],
                    "endpoint_harm_positive_count": row[
                        "endpoint_harm_positive_count"
                    ],
                    "endpoint_harm_mean": row["endpoint_harm_mean"],
                    "full_pooled_spearman": row["predictors"]["full_taylor"][
                        "pooled_spearman"
                    ],
                    "full_origin_centered_spearman": row["predictors"][
                        "full_taylor"
                    ]["origin_centered_spearman"],
                    "norm_origin_centered_spearman": row["predictors"][
                        "norm_only"
                    ]["origin_centered_spearman"],
                    "first_origin_centered_spearman": row["predictors"][
                        "first_order"
                    ]["origin_centered_spearman"],
                    "full_sign_accuracy": row["predictors"]["full_taylor"][
                        "sign_accuracy"
                    ],
                    "local_taylor_median_relative_error": row["local_closure"][
                        "full_taylor_median_relative_error"
                    ],
                    "directional_geometry_supported": row[
                        "directional_geometry_supported"
                    ],
                    "curvature_increment_supported": row[
                        "curvature_increment_supported"
                    ],
                }
            )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-jsonl", required=True, type=Path)
    parser.add_argument("--outcomes-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--contract", type=Path, default=HERE / "geo01b_contract.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract_path = args.contract.resolve()
    contract = P.read_json(contract_path)
    contract_checks = P.validate_contract(contract)
    if not all(contract_checks.values()):
        raise RuntimeError(f"contract validation failed: {contract_checks}")
    geometry_path = args.geometry_jsonl.resolve()
    outcomes_path = args.outcomes_jsonl.resolve()
    analysis = analyze(
        read_jsonl(geometry_path),
        read_jsonl(outcomes_path),
        contract,
        P.sha256_file(contract_path),
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "contract_sha256": P.sha256_file(contract_path),
        "geometry_input_sha256": P.sha256_file(geometry_path),
        "outcomes_input_sha256": P.sha256_file(outcomes_path),
        **analysis,
    }
    P.atomic_json(output / "analysis_manifest.json", manifest)
    write_event_csv(output / "event_summary.csv", manifest)
    print(output / "analysis_manifest.json")
    return 0 if manifest["integrity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
