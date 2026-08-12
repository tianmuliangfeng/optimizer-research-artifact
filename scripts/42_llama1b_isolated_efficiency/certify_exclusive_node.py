#!/usr/bin/env python3
"""Write a before/after certificate proving that the whole GPU node is idle."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path


SCRIPT_VERSION = "2026-07-29.1"


def query(arguments: list[str]) -> str:
    completed = subprocess.run(
        ["nvidia-smi", *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"nvidia-smi failed ({completed.returncode}): {completed.stdout}"
        )
    return completed.stdout.strip()


def csv_rows(text: str) -> list[list[str]]:
    if not text.strip():
        return []
    return [
        [value.strip() for value in row]
        for row in csv.reader(StringIO(text))
        if any(value.strip() for value in row)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-gpus", nargs="+", type=int, required=True)
    args = parser.parse_args()
    gpu_rows = csv_rows(
        query(
            [
                "--query-gpu=index,uuid,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ]
        )
    )
    process_rows = csv_rows(
        query(
            [
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    )
    gpus = [
        {
            "index": int(row[0]),
            "uuid": row[1],
            "name": row[2],
            "memory_total_mib": int(row[3]),
            "memory_free_mib": int(row[4]),
            "free_fraction": int(row[4]) / int(row[3]),
        }
        for row in gpu_rows
    ]
    processes = [
        {
            "gpu_uuid": row[0],
            "pid": int(row[1]),
            "process_name": row[2],
            "used_memory_mib": int(row[3]),
        }
        for row in process_rows
    ]
    required = set(args.required_gpus)
    observed = {gpu["index"] for gpu in gpus}
    checks = {
        "exact_gpu_inventory": observed == required,
        "gpu_names": all(
            gpu["name"] == "NVIDIA H100 80GB HBM3" for gpu in gpus
        ),
        "minimum_free_fraction": all(
            gpu["free_fraction"] >= 0.98 for gpu in gpus
        ),
        "active_compute_processes_absent": not processes,
    }
    payload = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "minimum_free_fraction": 0.98,
        "required_gpus": args.required_gpus,
        "gpus": gpus,
        "active_compute_processes": processes,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not payload["passed"]:
        raise SystemExit(
            "exclusive-node certificate failed; inspect " f"{args.output}"
        )
    print(f"Exclusive-node certificate: {args.output}")


if __name__ == "__main__":
    main()
