#!/usr/bin/env python3
"""Read-only, dependency-free adjudication of an MDP-04 result directory.

This is intentionally separate from the frozen scientific validator.  It does
not recompute or change a gate.  It verifies the persisted manifest, hashes and
coverage, localizes the recorded resolvent failure, and exits cleanly for an
expected archived negative result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-08-03.1"
EXPECTED_ROWS = {
    "layer_event": 432,
    "unit_event": 24,
    "unit_outcome": 12,
    "validation_slice": 6,
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_contract(run_dir: Path) -> Path:
    candidates = (
        run_dir
        / "source_snapshot_v4"
        / "scripts"
        / "mdp_refresh_streaming"
        / "refresh_stream_contract.json",
        run_dir
        / "source_snapshot"
        / "scripts"
        / "mdp_refresh_streaming"
        / "refresh_stream_contract.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("no sealed stream contract found in source snapshot")


def inspect(run_dir: Path) -> dict[str, Any]:
    analysis_dir = run_dir / "analysis"
    manifest_path = analysis_dir / "formal_stream_manifest.json"
    status_path = run_dir / "status.json"
    metrics_path = analysis_dir / "refresh_layer_event_metrics.csv"
    manifest = read_json(manifest_path)
    status = read_json(status_path)
    contract_path = find_contract(run_dir)
    contract = read_json(contract_path)

    artifact_hash_checks = {
        name: (analysis_dir / name).is_file()
        and sha256(analysis_dir / name) == expected
        for name, expected in sorted(manifest["artifacts"].items())
    }
    row_checks = {
        name: int(manifest["rows"].get(name, -1)) == expected
        for name, expected in EXPECTED_ROWS.items()
    }
    non_numeric_checks = {
        name: bool(value)
        for name, value in manifest["checks"].items()
        if name != "numeric_integrity_gates"
    }
    numeric_checks = {
        name: bool(value)
        for name, value in manifest["numeric_gate_checks"].items()
    }

    threshold = float(
        contract["hard_gates"]["runtime_resolvent_relative_residual_max"]
    )
    bad_rows: list[dict[str, Any]] = []
    metrics_parse_error = None
    try:
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            residual = float(row["runtime_resolvent_relative_residual"])
            if residual > threshold:
                bad_rows.append(
                    {
                        "origin": row["origin"],
                        "data_replica": int(row["data_replica"]),
                        "event_id": row["event_id"],
                        "layer_index": int(row["layer_index"]),
                        "runtime_resolvent_relative_residual": residual,
                    }
                )
    except (KeyError, TypeError, ValueError, csv.Error) as exc:
        rows = []
        bad_rows = []
        metrics_parse_error = f"{type(exc).__name__}: {exc}"

    failed_numeric = sorted(name for name, passed in numeric_checks.items() if not passed)
    structural_checks = {
        "analysis_artifact_hashes": all(artifact_hash_checks.values()),
        "manifest_rows": all(row_checks.values()),
        "csv_layer_rows": len(rows) == EXPECTED_ROWS["layer_event"],
        "csv_metrics_parse": metrics_parse_error is None,
        "all_non_numeric_manifest_checks": all(non_numeric_checks.values()),
        "selected_unit_count": len(manifest["unit_contract_lineage"]) == 12,
        "selected_units_passed": all(
            bool(row["passed"]) for row in manifest["unit_contract_lineage"]
        ),
        "contract_hash": sha256(contract_path) == manifest["stream_contract_sha256"],
    }
    archive_integrity_passed = all(structural_checks.values())

    if manifest["passed"] is True and status.get("status") == "passed":
        adjudication = "passed"
    elif (
        manifest["passed"] is False
        and status.get("status") == "integrity_failed"
        and failed_numeric == ["runtime_resolvent_relative_residual"]
        and archive_integrity_passed
    ):
        adjudication = "numeric_gate_failed"
    else:
        adjudication = "other_failure"

    return {
        "schema_version": "mdp04_archive_inspection_v1",
        "script_version": SCRIPT_VERSION,
        "run_id": run_dir.name,
        "formal_adjudication": adjudication,
        "archive_integrity_passed": archive_integrity_passed,
        "manifest_passed": bool(manifest["passed"]),
        "status": status.get("status"),
        "structural_checks": structural_checks,
        "row_checks": row_checks,
        "artifact_hash_checks": artifact_hash_checks,
        "numeric_gate_checks": numeric_checks,
        "failed_numeric_gates": failed_numeric,
        "resolvent_failure": {
            "threshold": threshold,
            "bad_row_count": len(bad_rows),
            "maximum": max(
                (row["runtime_resolvent_relative_residual"] for row in bad_rows),
                default=0.0,
            ),
            "origins": sorted({row["origin"] for row in bad_rows}),
            "layers": sorted({row["layer_index"] for row in bad_rows}),
            "rows": sorted(
                bad_rows,
                key=lambda row: row["runtime_resolvent_relative_residual"],
                reverse=True,
            ),
            "metrics_parse_error": metrics_parse_error,
        },
        "lineage_counts": manifest["unit_contract_lineage_counts"],
        "scientific_interpretation": (
            "computation_complete_but_formal_numeric_gate_failed"
            if adjudication == "numeric_gate_failed"
            else adjudication
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--expected",
        choices=("any", "passed", "numeric_gate_failed"),
        default="any",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = inspect(args.run_dir.resolve())
    except Exception as exc:  # concise CLI boundary; callers still get exit 2
        print(
            json.dumps(
                {
                    "archive_integrity_passed": False,
                    "formal_adjudication": "inspection_error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(2) from None

    print(json.dumps(payload, indent=2, sort_keys=True))
    expected_matches = args.expected == "any" or payload["formal_adjudication"] == args.expected
    if not payload["archive_integrity_passed"] or not expected_matches:
        print(
            "MDP-04 inspection did not match the requested adjudication; "
            "see the JSON summary above.",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
