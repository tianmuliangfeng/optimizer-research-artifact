#!/usr/bin/env python3
"""Preflight, schedule, resume, and analyze MECH-09R."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any


SCRIPT_VERSION = "2026-07-28.4"
WORKER_VERSION = "2026-07-28.3"
ANALYSIS_VERSION = "2026-07-28.3"
PUBLIC_CONTROL_REFERENCE_SHA256 = (
    "63464873e00c55c28b120c930ad207aa26fc75646678f9262e03904480c263ac"
)
COMPATIBLE_RESUME_CONTROLLER_VERSIONS = {
    "2026-07-28.3",
    SCRIPT_VERSION,
}
HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = load_module("mech09r_controller_helpers", HERE / "run_mech09.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--contract",
        default=HERE / "refresh_mediation_repair_contract.json",
        type=Path,
    )
    parser.add_argument(
        "--mech08-control-reference",
        default=HERE / "mech08_control_reference.json",
        type=Path,
    )
    parser.add_argument("--mech08-run-dir", required=True, type=Path)
    parser.add_argument("--source-script", required=True, type=Path)
    parser.add_argument("--profile-script", required=True, type=Path)
    parser.add_argument("--triton-kernels", required=True, type=Path)
    parser.add_argument("--train-data-pattern", required=True)
    parser.add_argument("--val-data-pattern", required=True)
    parser.add_argument("--child-python", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--execution-domain", required=True)
    parser.add_argument("--stamp")
    parser.add_argument("--resume-stamp")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return LEGACY.sha256_file(path)


def completed_manifest(
    path: Path,
    *,
    tier: str,
    cell: str,
    replica: int,
    contract_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    manifest = read_json(path)
    return (
        manifest.get("passed") is True
        and manifest.get("script_version") == WORKER_VERSION
        and manifest.get("analysis_tier") == tier
        and manifest.get("checkpoint_cell") == cell
        and int(manifest.get("data_replica", -1)) == int(replica)
        and manifest.get("contract_sha256") == contract_sha256
        and manifest.get("causal_tree") is True
    )


def worker_command(
    args: argparse.Namespace,
    spec: dict[str, Any],
    certificate: Path,
    output: Path,
    tier: str,
    replica: int,
    smoke_manifest: Path | None,
) -> list[str]:
    command = [
        args.child_python,
        str(HERE / "mech09r_worker.py"),
        "--output-dir",
        str(output),
        "--analysis-tier",
        tier,
        "--cell",
        spec["cell"],
        "--data-replica",
        str(replica),
        "--checkpoint",
        spec["path"],
        "--checkpoint-hash-certificate",
        str(certificate),
        "--source-script",
        str(args.source_script.resolve()),
        "--profile-script",
        str(args.profile_script.resolve()),
        "--triton-kernels",
        str(args.triton_kernels.resolve()),
        "--contract",
        str(args.contract.resolve()),
        "--mech08-control-reference",
        str(args.mech08_control_reference.resolve()),
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
        command.extend(["--smoke-manifest", str(smoke_manifest.resolve())])
    return command


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
    specs = {row["cell"]: row for row in contract["checkpoints"]}
    jobs = []
    for cell in config["origins"]:
        for replica in config["data_replicas"]:
            output = run / tier / cell / f"replica_{int(replica)}"
            manifest = output / "mech09r_manifest.json"
            if completed_manifest(
                manifest,
                tier=tier,
                cell=cell,
                replica=int(replica),
                contract_sha256=contract_sha256,
            ):
                print(
                    "MECH-09R resume: already passed "
                    f"{tier}/{cell}/replica_{int(replica)}",
                    flush=True,
                )
                continue
            output.mkdir(parents=True, exist_ok=True)
            jobs.append(
                {
                    "label": f"{tier}/{cell}/replica_{int(replica)}",
                    "output": output,
                    "command": worker_command(
                        args,
                        specs[cell],
                        certificates[cell],
                        output,
                        tier,
                        int(replica),
                        smoke_manifest,
                    ),
                }
            )
    return jobs


def run_jobs(
    jobs: list[dict[str, Any]],
    gpus: list[str],
    max_parallel: int,
    command_log: Path,
    cublas_workspace_config: str,
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
    failures: list[dict[str, Any]] = []
    with command_log.open("a", encoding="utf-8") as log:
        while pending or running:
            while pending and available and not failures:
                job = pending.pop(0)
                gpu = available.pop(0)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                env["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
                log.write(
                    json.dumps(
                        {
                            "label": job["label"],
                            "gpu": gpu,
                            "command": job["command"],
                            "CUBLAS_WORKSPACE_CONFIG": cublas_workspace_config,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log.flush()
                print(
                    f"MECH-09R launch gpu={gpu} label={job['label']}",
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
                    failures.append(
                        {
                            "label": job["label"],
                            "gpu": job["gpu"],
                            "return_code": return_code,
                            "output": str(job["output"]),
                        }
                    )
                else:
                    print(
                        f"MECH-09R passed gpu={job['gpu']} "
                        f"label={job['label']}",
                        flush=True,
                    )
            running = still_running
    if failures:
        raise RuntimeError(f"MECH-09R worker failures: {failures}")


def aggregate_tier(
    run: Path,
    contract: dict[str, Any],
    tier: str,
    contract_sha256: str,
) -> dict[str, Any]:
    config = contract[tier]
    rows = []
    for cell in config["origins"]:
        for replica in config["data_replicas"]:
            manifest_path = (
                run
                / tier
                / cell
                / f"replica_{int(replica)}"
                / "mech09r_manifest.json"
            )
            manifest = read_json(manifest_path)
            rows.append(
                {
                    "cell": cell,
                    "data_replica": int(replica),
                    "manifest": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "passed": manifest.get("passed") is True,
                    "worker_version": manifest.get("script_version"),
                    "contract_sha256": manifest.get("contract_sha256"),
                    "causal_tree": manifest.get("causal_tree") is True,
                }
            )
    expected = len(config["origins"]) * len(config["data_replicas"])
    return {
        "schema_version": 2,
        "controller_version": SCRIPT_VERSION,
        "worker_version": WORKER_VERSION,
        # These compatibility fields are the formal worker's smoke gate
        # contract.  Keep the controller-specific fields above for provenance.
        "script_version": WORKER_VERSION,
        "analysis_tier": tier,
        "causal_tree": True,
        "tier": tier,
        "contract_sha256": contract_sha256,
        "expected_jobs": expected,
        "completed_jobs": len(rows),
        "jobs": rows,
        "passed": len(rows) == expected
        and all(
            row["passed"]
            and row["worker_version"] == WORKER_VERSION
            and row["contract_sha256"] == contract_sha256
            and row["causal_tree"]
            for row in rows
        ),
    }


def main() -> None:
    args = parse_args()
    if args.stamp and args.resume_stamp:
        raise RuntimeError("--stamp and --resume-stamp are mutually exclusive")
    contract = read_json(args.contract.resolve())
    reference = read_json(args.mech08_control_reference.resolve())
    contract_sha = sha256_file(args.contract.resolve())
    stamp = (
        args.resume_stamp
        or args.stamp
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S+0000")
    )
    run = args.output_root.resolve() / stamp
    if args.resume_stamp:
        if not run.is_dir():
            raise RuntimeError(f"resume directory does not exist: {run}")
        identity_path = run / "run_identity.json"
        if not identity_path.is_file():
            raise RuntimeError(
                "refusing to resume a directory without MECH-09R identity: "
                f"{run}"
            )
        identity = read_json(identity_path)
        identity_checks = {
            "experiment": identity.get("experiment") == "MECH-09R",
            "script_version": identity.get("script_version")
            in COMPATIBLE_RESUME_CONTROLLER_VERSIONS,
            "contract_sha256": identity.get("contract_sha256")
            == contract_sha,
        }
        if not all(identity_checks.values()):
            raise RuntimeError(
                f"resume identity mismatch: {identity_checks}"
            )
        identity.setdefault(
            "created_by_script_version", identity["script_version"]
        )
        identity["script_version"] = SCRIPT_VERSION
        identity["last_resume_controller_version"] = SCRIPT_VERSION
        write_json(identity_path, identity)
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(
            run / "run_identity.json",
            {
                "experiment": "MECH-09R",
                "script_version": SCRIPT_VERSION,
                "contract_sha256": contract_sha,
                "legacy_invalid_run_reused": False,
            },
        )
    hashes = run / "checkpoint_hashes"
    hashes.mkdir(exist_ok=args.resume_stamp is not None)
    write_json(
        run / "status.json",
        {"status": "running", "script_version": SCRIPT_VERSION},
    )
    try:
        reference_sha = sha256_file(args.mech08_control_reference.resolve())
        certificate_source = contract["checkpoint_certificate_source"]
        expected_formal_jobs = (
            len(contract["formal"]["origins"])
            * len(contract["formal"]["data_replicas"])
        )
        preflight_checks = {
            "contract_version": contract.get("contract_version")
            == "2026-07-28.2",
            "experiment": contract.get("experiment") == "MECH-09R",
            "amendment_pre_intervention_only": contract[
                "protocol_amendment"
            ]["trigger_uses_pre_intervention_data_only"]
            is True,
            "reference_sha": reference_sha
            in {
                certificate_source["mech08_control_reference_sha256"],
                PUBLIC_CONTROL_REFERENCE_SHA256,
            },
            "reference_passed": reference.get("passed") is True,
            "reference_run_id": reference.get("source_run_id")
            == certificate_source["source_run_id"],
            "reference_run_path": str(args.mech08_run_dir.resolve())
            == certificate_source["source_run_path"],
            "source_exists": args.source_script.is_file(),
            "profile_exists": args.profile_script.is_file(),
            "triton_exists": args.triton_kernels.is_file(),
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
            "formal_jobs": expected_formal_jobs == 12,
            "formal_job_cap": expected_formal_jobs
            == int(contract["stopping_rule"]["maximum_new_formal_jobs"]),
            "trajectory_cap": expected_formal_jobs * len(contract["arms"])
            == int(contract["stopping_rule"]["maximum_trajectories"]),
            "gpu_ids_unique": len(args.gpus) == len(set(args.gpus)),
            "smoke_training_windows_disjoint": LEGACY.training_windows_disjoint(
                contract, "smoke"
            ),
            "formal_training_windows_disjoint": LEGACY.training_windows_disjoint(
                contract, "formal"
            ),
            "smoke_validation_windows_disjoint": LEGACY.validation_windows_disjoint(
                contract, "smoke"
            ),
            "formal_validation_windows_disjoint": LEGACY.validation_windows_disjoint(
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
            raise RuntimeError(
                f"MECH-09R preflight failed: {preflight_checks}"
            )

        control_audit = LEGACY.audit_control_reference(
            reference, args.mech08_run_dir.resolve()
        )
        control_audit["used_for_primary_outcomes"] = False
        control_audit["used_for_checkpoint_certificates_only"] = True
        write_json(run / "mech08_certificate_source_audit.json", control_audit)
        if not control_audit["passed"]:
            raise RuntimeError("MECH-08 certificate-source audit failed")

        inventory = []
        certificates: dict[str, Path] = {}
        for spec in contract["checkpoints"]:
            certificate = hashes / f"{spec['cell']}.json"
            reused = (
                LEGACY.reusable_hash_certificate(spec, certificate)
                if args.resume_stamp
                else None
            )
            if reused is None:
                reused = LEGACY.reuse_mech08_hash_certificate(
                    spec,
                    args.mech08_run_dir.resolve(),
                    certificate,
                )
            inventory.append(
                reused or LEGACY.hash_checkpoint(spec, certificate)
            )
            certificates[spec["cell"]] = certificate
        write_json(
            run / "checkpoint_inventory.json",
            {
                "schema_version": 2,
                "contract_sha256": contract_sha,
                "cells": inventory,
                "passed": all(row["passed"] for row in inventory),
            },
        )
        if not all(row["passed"] for row in inventory):
            raise RuntimeError("checkpoint inventory failed")

        max_parallel = (
            int(args.max_parallel)
            if args.max_parallel is not None
            else len(args.gpus)
        )
        cublas_config = contract["determinism"]["cublas_workspace_config"]
        smoke_jobs = build_jobs(
            args,
            contract,
            run,
            certificates,
            "smoke",
            contract_sha,
            None,
        )
        run_jobs(
            smoke_jobs,
            [str(value) for value in args.gpus],
            max_parallel,
            run / "commands.jsonl",
            cublas_config,
        )
        smoke_manifest = run / "smoke" / "smoke_manifest.json"
        smoke_manifest.parent.mkdir(exist_ok=True)
        smoke_payload = aggregate_tier(
            run, contract, "smoke", contract_sha
        )
        write_json(smoke_manifest, smoke_payload)
        if not smoke_payload["passed"]:
            raise RuntimeError(
                f"MECH-09R aggregate smoke failed: {smoke_payload}"
            )

        formal_jobs = build_jobs(
            args,
            contract,
            run,
            certificates,
            "formal",
            contract_sha,
            smoke_manifest,
        )
        run_jobs(
            formal_jobs,
            [str(value) for value in args.gpus],
            max_parallel,
            run / "commands.jsonl",
            cublas_config,
        )
        formal_payload = aggregate_tier(
            run, contract, "formal", contract_sha
        )
        write_json(run / "formal" / "formal_manifest.json", formal_payload)
        if not formal_payload["passed"]:
            raise RuntimeError("MECH-09R aggregate formal failed")

        analysis_manifest = (
            run / "analysis" / "mech09r_analysis_manifest.json"
        )
        analysis_complete = False
        if analysis_manifest.is_file():
            payload = read_json(analysis_manifest)
            analysis_complete = (
                payload.get("passed") is True
                and payload.get("script_version") == ANALYSIS_VERSION
                and payload.get("contract_sha256") == contract_sha
            )
        if not analysis_complete:
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "analyze_mech09r.py"),
                    "--run-dir",
                    str(run),
                    "--contract",
                    str(args.contract.resolve()),
                ],
                check=True,
            )
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
    print(f"MECH-09R artifacts: {run}")
    print(f"MECH-09R manifest: {analysis_manifest}")


if __name__ == "__main__":
    main()
