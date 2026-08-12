#!/usr/bin/env python3
"""Continuously audit GPU process isolation during one timed cell."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-29.1"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def query(arguments: list[str]) -> str:
    result = subprocess.run(
        ["nvidia-smi", *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"nvidia-smi failed: {result.stdout}")
    return result.stdout.strip()


def rows(text: str) -> list[list[str]]:
    if not text.strip():
        return []
    return [
        [value.strip() for value in row]
        for row in csv.reader(StringIO(text))
        if any(value.strip() for value in row)
    ]


def sample() -> dict[str, Any]:
    gpu_rows = rows(
        query(
            [
                "--query-gpu=index,uuid,name,memory.total,memory.free,"
                "utilization.gpu,temperature.gpu,clocks.sm",
                "--format=csv,noheader,nounits",
            ]
        )
    )
    process_rows = rows(
        query(
            [
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    )
    return {
        "observed_at": now_iso(),
        "gpus": [
            {
                "index": int(row[0]),
                "uuid": row[1],
                "name": row[2],
                "memory_total_mib": int(row[3]),
                "memory_free_mib": int(row[4]),
                "utilization_gpu_pct": int(row[5]),
                "temperature_c": int(row[6]),
                "sm_clock_mhz": int(row[7]),
            }
            for row in gpu_rows
        ],
        "processes": [
            {
                "gpu_uuid": row[0],
                "pid": int(row[1]),
                "process_name": row[2],
                "used_memory_mib": int(row[3]),
            }
            for row in process_rows
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--timing-gpu", type=int, required=True)
    parser.add_argument("--idle-gpus", nargs="+", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_path = output / "gpu_isolation_samples.jsonl"
    errors: list[str] = []
    observed_samples: list[dict[str, Any]] = []
    while True:
        try:
            observed = sample()
            observed_samples.append(observed)
            with sample_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(observed, sort_keys=True) + "\n")
        except Exception as exc:
            errors.append(repr(exc))
        if args.stop_file.exists():
            break
        time.sleep(max(1.0, args.interval_seconds))

    gpu_index_to_uuid: dict[int, str] = {}
    inventory_stable = True
    expected_gpu_indices: set[int] | None = None
    timing_process_pids: set[int] = set()
    maximum_timing_processes = 0
    idle_process_events: list[dict[str, Any]] = []
    unexpected_names: list[str] = []
    for observed in observed_samples:
        current = {gpu["index"] for gpu in observed["gpus"]}
        if expected_gpu_indices is None:
            expected_gpu_indices = current
        elif current != expected_gpu_indices:
            inventory_stable = False
        for gpu in observed["gpus"]:
            previous = gpu_index_to_uuid.setdefault(gpu["index"], gpu["uuid"])
            if previous != gpu["uuid"]:
                inventory_stable = False
            if gpu["name"] != "NVIDIA H100 80GB HBM3":
                unexpected_names.append(gpu["name"])
        timing_uuid = gpu_index_to_uuid.get(args.timing_gpu)
        idle_uuids = {
            gpu_index_to_uuid[index]
            for index in args.idle_gpus
            if index in gpu_index_to_uuid
        }
        timing_processes = [
            process
            for process in observed["processes"]
            if process["gpu_uuid"] == timing_uuid
        ]
        maximum_timing_processes = max(
            maximum_timing_processes, len(timing_processes)
        )
        timing_process_pids.update(process["pid"] for process in timing_processes)
        idle_processes = [
            process
            for process in observed["processes"]
            if process["gpu_uuid"] in idle_uuids
        ]
        if idle_processes:
            idle_process_events.append(
                {
                    "observed_at": observed["observed_at"],
                    "processes": idle_processes,
                }
            )
    checks = {
        "samples_present": len(observed_samples) >= 2,
        "query_errors_absent": not errors,
        "inventory_stable": inventory_stable,
        "timing_gpu_present": args.timing_gpu in gpu_index_to_uuid,
        "idle_gpus_present": set(args.idle_gpus).issubset(gpu_index_to_uuid),
        "gpu_names": not unexpected_names,
        "idle_gpu_processes_absent": not idle_process_events,
        "at_most_one_timing_process_per_sample": maximum_timing_processes <= 1,
        "single_timing_process_identity": len(timing_process_pids) == 1,
    }
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at": now_iso(),
        "passed": all(checks.values()),
        "timing_gpu": args.timing_gpu,
        "idle_gpus": args.idle_gpus,
        "sample_interval_seconds": args.interval_seconds,
        "sample_count": len(observed_samples),
        "gpu_index_to_uuid": gpu_index_to_uuid,
        "timing_process_pids": sorted(timing_process_pids),
        "maximum_timing_processes": maximum_timing_processes,
        "idle_process_events": idle_process_events,
        "query_errors": errors,
        "unexpected_gpu_names": sorted(set(unexpected_names)),
        "checks": checks,
    }
    (output / "gpu_isolation_monitor.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not manifest["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
