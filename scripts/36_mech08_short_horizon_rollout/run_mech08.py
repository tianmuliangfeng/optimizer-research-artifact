#!/usr/bin/env python3
"""Certify, smoke-test, schedule, resume, and analyze MECH-08."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-27.2"
WORKER_VERSION = "2026-07-27.2"
ANALYSIS_VERSION = "2026-07-27.1"
HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--contract", default=HERE / "rollout_contract.json", type=Path
    )
    parser.add_argument(
        "--prediction-reference",
        default=HERE / "mech07_prediction_reference.csv",
        type=Path,
    )
    parser.add_argument("--source-script", required=True, type=Path)
    parser.add_argument("--profile-script", required=True, type=Path)
    parser.add_argument("--triton-kernels", required=True, type=Path)
    parser.add_argument("--train-data-pattern", required=True)
    parser.add_argument("--val-data-pattern", required=True)
    parser.add_argument("--child-python", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--host-id", default="llama-host-h100")
    parser.add_argument(
        "--execution-domain", default="llama-host-llama1b-mech08"
    )
    parser.add_argument("--stamp")
    parser.add_argument(
        "--resume-stamp",
        help="Resume an existing timestamp and reuse valid checkpoint certificates.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_windows_disjoint(
    contract: dict[str, Any], tier: str
) -> bool:
    config = contract[tier]
    build_width = (
        int(config["build_device_batch_size"])
        * int(config["build_sequence_length"])
        * int(config["build_batches"])
    )
    eval_width = (
        int(config["eval_device_batch_size"])
        * int(config["eval_sequence_length"])
        * int(config["eval_batches"])
    )
    intervals = [
        ("build", index, int(offset), int(offset) + build_width)
        for index, offset in enumerate(config["build_token_offsets"])
    ] + [
        ("evaluation", index, int(offset), int(offset) + eval_width)
        for index, offset in enumerate(config["eval_token_offsets"])
    ]
    for left, first in enumerate(intervals):
        for second in intervals[left + 1 :]:
            if max(first[2], second[2]) < min(first[3], second[3]):
                return False
    return True


def training_windows_disjoint(contract: dict[str, Any], tier: str) -> bool:
    config = contract[tier]
    horizon = int(config["rollout_steps"])
    intervals = [
        (int(offset), int(offset) + horizon)
        for offset in config["replica_optimizer_step_offsets"]
    ]
    return all(
        max(first[0], second[0]) >= min(first[1], second[1])
        for left, first in enumerate(intervals)
        for second in intervals[left + 1 :]
    )


def hash_checkpoint(spec: dict[str, Any], output: Path) -> dict[str, Any]:
    path = Path(spec["path"])
    before = path.stat()
    started = time.time()
    observed = sha256_file(path)
    after = path.stat()
    checks = {
        "stable_size": before.st_size == after.st_size,
        "stable_mtime": before.st_mtime_ns == after.st_mtime_ns,
        "expected_bytes": before.st_size == int(spec["expected_bytes"]),
        "expected_sha256": observed == spec["expected_sha256"],
    }
    payload = {
        "schema_version": 1,
        "cell": spec["cell"],
        "stage": spec["stage"],
        "method": spec["method"],
        "step": spec["step"],
        "path": str(path),
        "bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "sha256": observed,
        "expected_bytes": spec["expected_bytes"],
        "expected_sha256": spec["expected_sha256"],
        "elapsed_seconds": time.time() - started,
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_json(output, payload)
    if not payload["passed"]:
        raise RuntimeError(f"checkpoint hash audit failed: {payload}")
    return payload


def reusable_hash_certificate(
    spec: dict[str, Any], certificate: Path
) -> dict[str, Any] | None:
    if not certificate.is_file():
        return None
    payload = read_json(certificate)
    path = Path(spec["path"])
    stat = path.stat()
    checks = {
        "passed": payload.get("passed") is True,
        "cell": payload.get("cell") == spec["cell"],
        "path": payload.get("path") == spec["path"],
        "size": int(payload.get("bytes", -1)) == stat.st_size,
        "mtime": int(payload.get("mtime_ns", -1)) == stat.st_mtime_ns,
        "expected_size": stat.st_size == int(spec["expected_bytes"]),
        "known_hash": payload.get("sha256") == spec["expected_sha256"],
    }
    return payload if all(checks.values()) else None


def completed_manifest(
    path: Path,
    *,
    tier: str,
    cell: str,
    algorithm: str,
    replica: int,
    contract_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    manifest = read_json(path)
    return (
        manifest.get("passed") is True
        and manifest.get("analysis_tier") == tier
        and manifest.get("checkpoint_cell") == cell
        and manifest.get("algorithm") == algorithm
        and int(manifest.get("data_replica", -1)) == replica
        and manifest.get("contract_sha256") == contract_sha256
        and manifest.get("script_version") == WORKER_VERSION
    )


def worker_command(
    args: argparse.Namespace,
    spec: dict[str, Any],
    certificate: Path,
    output: Path,
    tier: str,
    algorithm: str,
    replica: int,
    smoke_manifest: Path | None,
) -> list[str]:
    command = [
        args.child_python,
        str(HERE / "mech08_worker.py"),
        "--output-dir",
        str(output),
        "--analysis-tier",
        tier,
        "--cell",
        spec["cell"],
        "--algorithm",
        algorithm,
        "--data-replica",
        str(replica),
        "--checkpoint",
        spec["path"],
        "--checkpoint-hash-certificate",
        str(certificate),
        "--source-script",
        str(args.source_script),
        "--profile-script",
        str(args.profile_script),
        "--triton-kernels",
        str(args.triton_kernels),
        "--contract",
        str(args.contract),
        "--prediction-reference",
        str(args.prediction_reference),
        "--train-data-pattern",
        args.train_data_pattern,
        "--val-data-pattern",
        args.val_data_pattern,
        "--host-id",
        args.host_id,
        "--execution-domain",
        args.execution_domain,
    ]
    if smoke_manifest is not None:
        command.extend(["--smoke-manifest", str(smoke_manifest)])
    return command


def run_jobs(
    jobs: list[dict[str, Any]],
    gpus: list[str],
    max_parallel: int,
    command_log: Path,
) -> None:
    if not jobs:
        return
    if not gpus:
        raise RuntimeError("at least one GPU id is required")
    capacity = min(int(max_parallel), len(gpus))
    if capacity <= 0:
        raise RuntimeError("max_parallel must be positive")
    pending = list(jobs)
    available = list(gpus[:capacity])
    running: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    with command_log.open("a", encoding="utf-8") as log:
        while pending or running:
            while pending and available and not failed:
                job = pending.pop(0)
                gpu = available.pop(0)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                payload = {
                    "label": job["label"],
                    "gpu": gpu,
                    "command": job["command"],
                }
                log.write(json.dumps(payload, ensure_ascii=False) + "\n")
                log.flush()
                print(
                    f"MECH-08 launch gpu={gpu} label={job['label']}",
                    flush=True,
                )
                process = subprocess.Popen(job["command"], env=env)
                running.append(job | {"gpu": gpu, "process": process})
            if not running:
                break
            time.sleep(1.0)
            still_running = []
            for job in running:
                return_code = job["process"].poll()
                if return_code is None:
                    still_running.append(job)
                    continue
                available.append(job["gpu"])
                if return_code != 0:
                    failed.append(
                        {
                            "label": job["label"],
                            "gpu": job["gpu"],
                            "return_code": return_code,
                            "output": str(job["output"]),
                        }
                    )
                else:
                    print(
                        f"MECH-08 passed gpu={job['gpu']} "
                        f"label={job['label']}",
                        flush=True,
                    )
            running = still_running
        if running:
            for job in running:
                job["process"].wait()
    if failed:
        raise RuntimeError(f"MECH-08 worker failures: {failed}")


def build_jobs(
    args: argparse.Namespace,
    contract: dict[str, Any],
    run: Path,
    certificates: dict[str, Path],
    tier: str,
    contract_sha256: str,
    smoke_manifest: Path | None,
) -> list[dict[str, Any]]:
    config = contract[tier]
    spec_by_cell = {row["cell"]: row for row in contract["checkpoints"]}
    jobs = []
    for cell in config["origins"]:
        spec = spec_by_cell[cell]
        for algorithm in config["algorithms"]:
            for replica in config["data_replicas"]:
                output = (
                    run
                    / tier
                    / cell
                    / algorithm
                    / f"replica_{int(replica)}"
                )
                manifest = output / "mech08_manifest.json"
                if completed_manifest(
                    manifest,
                    tier=tier,
                    cell=cell,
                    algorithm=algorithm,
                    replica=int(replica),
                    contract_sha256=contract_sha256,
                ):
                    print(
                        "MECH-08 resume: already passed "
                        f"{tier}/{cell}/{algorithm}/replica_{replica}",
                        flush=True,
                    )
                    continue
                output.mkdir(parents=True, exist_ok=True)
                jobs.append(
                    {
                        "label": (
                            f"{tier}/{cell}/{algorithm}/replica_{int(replica)}"
                        ),
                        "output": output,
                        "command": worker_command(
                            args,
                            spec,
                            certificates[cell],
                            output,
                            tier,
                            algorithm,
                            int(replica),
                            smoke_manifest,
                        ),
                    }
                )
    return jobs


def smoke_aggregate(
    run: Path, contract: dict[str, Any], contract_sha256: str
) -> dict[str, Any]:
    config = contract["smoke"]
    rows = []
    for cell in config["origins"]:
        for algorithm in config["algorithms"]:
            for replica in config["data_replicas"]:
                manifest_path = (
                    run
                    / "smoke"
                    / cell
                    / algorithm
                    / f"replica_{int(replica)}"
                    / "mech08_manifest.json"
                )
                manifest = read_json(manifest_path)
                rows.append(
                    {
                        "cell": cell,
                        "algorithm": algorithm,
                        "data_replica": int(replica),
                        "manifest": str(manifest_path),
                        "manifest_sha256": sha256_file(manifest_path),
                        "passed": manifest.get("passed") is True,
                        "worker_version": manifest.get("script_version"),
                        "contract_sha256": manifest.get("contract_sha256"),
                    }
                )
    payload = {
        "schema_version": 1,
        "controller_version": SCRIPT_VERSION,
        "worker_version": WORKER_VERSION,
        "contract_sha256": contract_sha256,
        "expected_jobs": 4,
        "completed_jobs": len(rows),
        "jobs": rows,
        "passed": len(rows) == 4
        and all(
            row["passed"]
            and row["worker_version"] == WORKER_VERSION
            and row["contract_sha256"] == contract_sha256
            for row in rows
        ),
    }
    return payload


def main() -> None:
    args = parse_args()
    if args.stamp and args.resume_stamp:
        raise RuntimeError("--stamp and --resume-stamp are mutually exclusive")
    contract = read_json(args.contract.resolve())
    stamp = (
        args.resume_stamp
        or args.stamp
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S+0000")
    )
    run = args.output_root.resolve() / stamp
    if args.resume_stamp:
        if not run.is_dir():
            raise RuntimeError(f"resume directory does not exist: {run}")
    else:
        run.mkdir(parents=True, exist_ok=False)
    hashes = run / "checkpoint_hashes"
    hashes.mkdir(exist_ok=args.resume_stamp is not None)
    write_json(
        run / "status.json",
        {"status": "running", "script_version": SCRIPT_VERSION},
    )
    try:
        contract_sha256 = sha256_file(args.contract.resolve())
        preflight_checks = {
            "contract_version": contract.get("contract_version")
            == "2026-07-27.1",
            "source_exists": args.source_script.is_file(),
            "profile_exists": args.profile_script.is_file(),
            "triton_exists": args.triton_kernels.is_file(),
            "prediction_reference_exists": args.prediction_reference.is_file(),
            "child_python_exists": Path(args.child_python).is_file(),
            "train_data_exists": bool(glob.glob(args.train_data_pattern)),
            "val_data_exists": bool(glob.glob(args.val_data_pattern)),
            "all_checkpoints_exist": all(
                Path(spec["path"]).is_file()
                for spec in contract["checkpoints"]
            ),
            "source_sha256": args.source_script.is_file()
            and sha256_file(args.source_script)
            == contract["source_constraints"]["base_source_sha256"],
            "profile_sha256": args.profile_script.is_file()
            and sha256_file(args.profile_script)
            == contract["source_constraints"]["profile_script_sha256"],
            "triton_sha256": args.triton_kernels.is_file()
            and sha256_file(args.triton_kernels)
            == contract["source_constraints"]["triton_sha256"],
            "prediction_reference_sha256": args.prediction_reference.is_file()
            and sha256_file(args.prediction_reference)
            == contract["prediction_reference"]["sha256"],
            "formal_jobs": len(contract["formal"]["origins"])
            * len(contract["formal"]["algorithms"])
            * len(contract["formal"]["data_replicas"])
            == 48,
            "gpu_ids_unique": len(args.gpus) == len(set(args.gpus)),
            "smoke_training_windows_disjoint": training_windows_disjoint(
                contract, "smoke"
            ),
            "formal_training_windows_disjoint": training_windows_disjoint(
                contract, "formal"
            ),
            "smoke_validation_windows_disjoint": validation_windows_disjoint(
                contract, "smoke"
            ),
            "formal_validation_windows_disjoint": validation_windows_disjoint(
                contract, "formal"
            ),
            "efficiency_excluded": contract["scope_boundary"][
                "efficiency_benchmark_excluded"
            ]
            is True,
        }
        write_json(
            run / "preflight.json",
            {
                "checks": preflight_checks,
                "passed": all(preflight_checks.values()),
            },
        )
        if not all(preflight_checks.values()):
            raise RuntimeError(f"MECH-08 preflight failed: {preflight_checks}")

        inventory = []
        certificates: dict[str, Path] = {}
        for spec in contract["checkpoints"]:
            certificate = hashes / f"{spec['cell']}.json"
            reused = (
                reusable_hash_certificate(spec, certificate)
                if args.resume_stamp
                else None
            )
            inventory.append(reused or hash_checkpoint(spec, certificate))
            certificates[spec["cell"]] = certificate
        write_json(
            run / "checkpoint_inventory.json",
            {
                "schema_version": 1,
                "contract_sha256": contract_sha256,
                "cells": inventory,
                "passed": all(row["passed"] for row in inventory),
            },
        )

        max_parallel = (
            int(args.max_parallel)
            if args.max_parallel is not None
            else len(args.gpus)
        )
        smoke_jobs = build_jobs(
            args,
            contract,
            run,
            certificates,
            "smoke",
            contract_sha256,
            None,
        )
        run_jobs(
            smoke_jobs,
            [str(value) for value in args.gpus],
            max_parallel,
            run / "commands.jsonl",
        )
        smoke_manifest = run / "smoke" / "smoke_manifest.json"
        smoke_manifest.parent.mkdir(exist_ok=True)
        smoke_payload = smoke_aggregate(run, contract, contract_sha256)
        write_json(smoke_manifest, smoke_payload)
        if not smoke_payload["passed"]:
            raise RuntimeError(f"MECH-08 aggregate smoke failed: {smoke_payload}")

        formal_jobs = build_jobs(
            args,
            contract,
            run,
            certificates,
            "formal",
            contract_sha256,
            smoke_manifest,
        )
        run_jobs(
            formal_jobs,
            [str(value) for value in args.gpus],
            max_parallel,
            run / "commands.jsonl",
        )

        analysis_manifest = run / "analysis" / "mech08_analysis_manifest.json"
        analysis_complete = False
        if analysis_manifest.is_file():
            analysis_payload = read_json(analysis_manifest)
            analysis_complete = (
                analysis_payload.get("passed") is True
                and analysis_payload.get("script_version") == ANALYSIS_VERSION
                and analysis_payload.get("contract_sha256")
                == contract_sha256
            )
        if not analysis_complete:
            analysis_command = [
                sys.executable,
                str(HERE / "analyze_mech08.py"),
                "--run-dir",
                str(run),
                "--contract",
                str(args.contract.resolve()),
                "--prediction-reference",
                str(args.prediction_reference.resolve()),
            ]
            subprocess.run(analysis_command, check=True)
        write_json(
            run / "status.json",
            {
                "status": "passed",
                "script_version": SCRIPT_VERSION,
                "analysis_manifest": str(analysis_manifest),
            },
        )
    except BaseException as exc:
        write_json(
            run / "status.json",
            {
                "status": "failed",
                "script_version": SCRIPT_VERSION,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(f"MECH-08 artifacts: {run}")
    print(
        f"MECH-08 manifest: "
        f"{run / 'analysis' / 'mech08_analysis_manifest.json'}"
    )


if __name__ == "__main__":
    main()
