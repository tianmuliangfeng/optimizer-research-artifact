#!/usr/bin/env python3
"""Generate a read-only LLaMA-1B 10B feasibility report; never launch training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import protocol as P


def markdown_report(report: dict[str, Any]) -> str:
    budget = report["budget"]
    schedule = report["schedule"]
    lines = [
        "# LLaMA-1B 10B feasibility report",
        "",
        "**Status: planning only; launch is not authorized.**",
        "",
        "## Frozen milestone grid",
        "",
        "| Milestone | Step | Actual train tokens | Tokens/parameter | Token error |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in schedule:
        lines.append(
            f"| `{row['id']}` | {row['step']} | {row['actual_tokens']} | "
            f"{row['tokens_per_parameter']:.6f} | {row['token_error']:+d} |"
        )
    lines.extend(
        [
            "",
            "## Resource estimate",
            "",
            f"- Aggregate training tokens: `{budget['aggregate_training_tokens']}`.",
            f"- Raw training GPU-hours: `{budget['raw_training_gpu_hours']:.2f}`.",
            f"- Two-H100 wall estimate including frozen overhead: "
            f"`{budget['gpu_scenarios']['2']['wall_seconds'] / 86400.0:.2f} days`.",
            f"- Four-H100 wall estimate including frozen overhead: "
            f"`{budget['gpu_scenarios']['4']['wall_seconds'] / 3600.0:.2f} hours`.",
            f"- Retained three-milestone checkpoints: "
            f"`{budget['retained_checkpoint_bytes'] / 1e9:.2f} GB`.",
            f"- Checkpoint-only floor after 20% headroom: "
            f"`{budget['checkpoint_headroom_bytes'] / 1e9:.2f} GB`.",
            f"- Recommended operational free disk target: "
            f"`{budget['recommended_minimum_free_disk_bytes'] / 1e9:.2f} GB`.",
            f"- Validation events per method: "
            f"`{budget['validation_events_per_method']}`.",
            "",
            "## Hard blockers",
            "",
        ]
    )
    lines.extend(f"- `{value}`" for value in report["hard_blockers"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This report may guide a later preregistration. It is not an experiment "
            "manifest, remote command, scientific result, or authorization to train.",
            "",
        ]
    )
    return "\n".join(lines)


def write_schedule(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "step_rule",
        "target_tokens",
        "step",
        "actual_tokens",
        "token_error",
        "relative_token_error",
        "tokens_per_parameter",
        "train_microbatches",
        "stream_microbatches_including_prefetch",
        "stream_tokens_including_prefetch",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "plan", "audit-data"))
    parser.add_argument("--contract", type=Path, default=HERE / "feasibility_contract.json")
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "audit-data" and args.data_dir is None:
        parser.error("audit-data requires --data-dir")
    if args.mode != "check" and args.output_dir is None:
        parser.error(f"{args.mode} requires --output-dir")
    return args


def main() -> int:
    args = parse_args()
    contract = P.read_json(args.contract.absolute())
    checks = P.validate_contract(contract)
    source = P.audit_current_sources(args.repo.absolute())
    if args.mode == "check":
        report = P.build_report(contract, source, None)
        payload = {
            "schema_version": "llama1b_10b_feasibility_check_v1",
            "contract_sha256": P.canonical_sha256(contract),
            "contract_checks": checks,
            "source_audit": source,
            "launch_authorized": False,
            "contract_integrity_passed": all(checks.values()),
            "technical_prerequisites_passed": report["technical_prerequisites_passed"],
            "data_audit_passed": None,
            "passed": all(checks.values()),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["passed"] else 2
    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=False)
    data = P.audit_data_dir(args.data_dir, contract) if args.mode == "audit-data" else None
    report = P.build_report(contract, source, data)
    P.atomic_json(output / "feasibility_report.json", report)
    if data is not None:
        P.atomic_json(output / "data_inventory_audit.json", data)
    write_schedule(output / "milestone_schedule.csv", report["schedule"])
    (output / "FEASIBILITY_REPORT.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    artifact_hashes = {
        path.name: P.sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    P.atomic_json(
        output / "feasibility_manifest.json",
        {
            "schema_version": "llama1b_10b_feasibility_manifest_v1",
            "mode": args.mode,
            "contract_sha256": P.canonical_sha256(contract),
            "artifact_sha256": artifact_hashes,
            "launch_authorized": False,
            "scientific_evidence_class": "none_planning_only",
            "contract_integrity_passed": all(checks.values()),
            "technical_prerequisites_passed": report["technical_prerequisites_passed"],
            "data_audit_passed": None if data is None else data["passed"],
            "passed": all(checks.values()) and (data is None or data["passed"]),
        },
    )
    print(output / "FEASIBILITY_REPORT.md")
    print("Launch authorized: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
