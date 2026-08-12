#!/usr/bin/env python3
"""Analyze scalar GEO-01 geometry rows without opening disabled phases."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P


REQUIRED_NUMERIC = (
    "first_order_alignment",
    "directional_curvature",
    "second_order_term",
    "taylor_actual_delta_loss",
    "exact_actual_delta_loss",
    "taylor_residual",
    "direction_fro_norm",
    "relative_direction_fro_norm",
    "fd_first_relative_error",
    "fd_curvature_relative_error",
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"row {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError("GEO-01 input contains no rows")
    return rows


def summarize(rows: list[dict[str, Any]], phase: str, contract: dict[str, Any]) -> dict[str, Any]:
    if phase not in ("pilot", "discovery", "confirmation"):
        raise ValueError(f"unknown phase: {phase}")
    execution = contract["execution"]
    if phase == "discovery" and execution["discovery_enabled"] is not True:
        raise RuntimeError("discovery analysis is disabled by the current contract")
    if phase == "confirmation" and execution["confirmation_enabled"] is not True:
        raise RuntimeError("confirmation analysis is disabled by the current contract")
    numeric_ok = all(
        field in row and math.isfinite(float(row[field]))
        for row in rows
        for field in REQUIRED_NUMERIC
    )
    invariance_ok = all(row.get("parameters_unchanged") is True for row in rows)
    exact_scopes = sorted(str(row.get("scope_id")) for row in rows)
    if phase == "pilot":
        expected = sorted(row["scope_id"] for row in contract["pilot"]["scopes"])
        scope_ok = exact_scopes == expected
    else:
        scope_ok = len(exact_scopes) == len(set(exact_scopes))
    summary = {
        "row_count": len(rows),
        "scopes": exact_scopes,
        "mean_first_order_alignment": statistics.fmean(
            float(row["first_order_alignment"]) for row in rows
        ),
        "mean_directional_curvature": statistics.fmean(
            float(row["directional_curvature"]) for row in rows
        ),
        "mean_exact_actual_delta_loss": statistics.fmean(
            float(row["exact_actual_delta_loss"]) for row in rows
        ),
        "mean_taylor_residual": statistics.fmean(
            float(row["taylor_residual"]) for row in rows
        ),
        "max_fd_first_relative_error": max(
            float(row["fd_first_relative_error"]) for row in rows
        ),
        "max_fd_curvature_relative_error": max(
            float(row["fd_curvature_relative_error"]) for row in rows
        ),
        "checks": {
            "numeric_fields_finite": numeric_ok,
            "parameters_unchanged": invariance_ok,
            "scope_contract": scope_ok,
            "pilot_not_claim_eligible": phase != "pilot"
            or contract["claim_boundary"]["pilot_claim_eligible"] is False,
        },
    }
    summary["integrity_passed"] = all(summary["checks"].values())
    summary["scientific_result"] = (
        "engineering_pilot_only_no_scientific_claim"
        if phase == "pilot"
        else "analysis_requires_phase_specific_frozen_gate"
    )
    summary["claim_eligible"] = False
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["scope_id", *REQUIRED_NUMERIC, "parameters_unchanged"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=HERE / "geo01_contract.json")
    parser.add_argument("--phase", choices=("pilot", "discovery", "confirmation"), default="pilot")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = P.read_json(args.contract.resolve())
    checks = P.validate_contract(contract)
    if not all(checks.values()):
        raise RuntimeError(f"contract validation failed: {checks}")
    rows = read_rows(args.input_jsonl.resolve())
    summary = summarize(rows, args.phase, contract)
    output = args.output_dir.resolve()
    # Analysis is a deterministic derivative of a selected immutable attempt.
    # Allow a same-contract resume to reconstruct it after an interrupted write.
    output.mkdir(parents=True, exist_ok=True)
    write_summary_csv(output / "geometry_rows.csv", rows)
    manifest = {
        "schema_version": "geo01_analysis_manifest_v1",
        "phase": args.phase,
        "contract_sha256": P.sha256_file(args.contract.resolve()),
        "input_sha256": P.sha256_file(args.input_jsonl.resolve()),
        **summary,
    }
    P.atomic_json(output / "analysis_manifest.json", manifest)
    print(output / "analysis_manifest.json")
    return 0 if manifest["integrity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
