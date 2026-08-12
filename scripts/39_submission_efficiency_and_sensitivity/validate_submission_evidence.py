#!/usr/bin/env python3
"""Independent validation for a completed 39 submission evidence audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROLES = {
    "muon",
    "original_newton_muon",
    "selective_none",
    "selective_diag",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_final_lr_coverage(
    sensitivity: list[dict[str, str]], missing_metrics: set[str]
) -> None:
    final_rows = [
        row
        for row in sensitivity
        if row["sensitivity_type"] == "learning_rate"
        and row["architecture"] == "final R1"
        and row["grid"] == "0.8x,1.0x,1.2x"
        and set(row["method_roles"].split(",")) == ROLES
    ]
    if len(final_rows) != 1:
        raise RuntimeError("final four-role LR coverage row is not unique")
    expected = (
        "missing"
        if "four_method_lr_sensitivity" in missing_metrics
        else "supporting_only"
    )
    if final_rows[0]["classification"] != expected:
        raise RuntimeError(
            "four-role LR state disagrees with metric eligibility: "
            f"{final_rows[0]['classification']!r} != {expected!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--require-submission-ready", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(
        (args.run_dir / "submission_evidence_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if not manifest["audit_passed"] or manifest["diag_vs_none_primary"]:
        raise RuntimeError("manifest acceptance failed")
    for name, expected in manifest["output_sha256"].items():
        if sha256_file(args.run_dir / name) != expected:
            raise RuntimeError(f"hash mismatch: {name}")

    capacity = read_csv(args.run_dir / "capacity_boundary.csv")
    memory = read_csv(args.run_dir / "fixed_batch_memory.csv")
    if {row["method_role"] for row in capacity} != ROLES:
        raise RuntimeError("capacity role coverage failed")
    if {row["method_role"] for row in memory} != ROLES:
        raise RuntimeError("memory role coverage failed")
    if any(
        int(row["first_oom_device_batch"])
        - int(row["max_success_device_batch"])
        != 1
        for row in capacity
    ):
        raise RuntimeError("capacity boundary is not exact")
    by_role = {row["method_role"]: row for row in memory}
    if not (
        float(by_role["selective_none"]["peak_allocated_mib"])
        < float(by_role["original_newton_muon"]["peak_allocated_mib"])
        and float(by_role["selective_diag"]["peak_allocated_mib"])
        < float(by_role["original_newton_muon"]["peak_allocated_mib"])
    ):
        raise RuntimeError("memory contrast recomputation failed")

    eligibility = read_csv(args.run_dir / "metric_eligibility.csv")
    missing = {
        row["metric"] for row in eligibility if row["classification"] == "missing"
    }
    if set(manifest["missing_metrics"]) != missing:
        raise RuntimeError("missing metric list mismatch")
    if args.require_submission_ready:
        if (
            manifest.get("submission_ready") is not True
            or manifest.get("isolated_performance_found") is not True
            or missing
            or manifest.get("blocking_followups")
        ):
            raise RuntimeError(
                "final experiment-39 audit is not submission-ready: "
                f"missing={sorted(missing)}, "
                f"blocking={manifest.get('blocking_followups')}"
            )
    followup = json.loads(
        (args.run_dir / "minimal_followup_contract.json").read_text(encoding="utf-8")
    )
    followup_ids = {item["id"] for item in followup["experiments"]}
    expected_ids = {
        row["followup_id"] for row in eligibility if row["followup_id"]
    }
    if followup_ids != expected_ids:
        raise RuntimeError("follow-up coverage mismatch")
    sensitivity = read_csv(args.run_dir / "sensitivity_coverage.csv")
    alpha = next(row for row in sensitivity if row["sensitivity_type"] == "alpha_response")
    if alpha["classification"] != "paper_ready":
        raise RuntimeError("alpha classification changed")
    validate_final_lr_coverage(sensitivity, missing)
    print(
        json.dumps(
            {
                "passed": True,
                "submission_ready": manifest["submission_ready"],
                "roles": len(ROLES),
                "missing_metrics": sorted(missing),
                "followups": sorted(followup_ids),
                "hashes_checked": len(manifest["output_sha256"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
