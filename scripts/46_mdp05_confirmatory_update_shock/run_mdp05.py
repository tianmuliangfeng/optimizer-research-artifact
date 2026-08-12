#!/usr/bin/env python3
"""Seal, schedule, resume, and analyze the independent MDP-05 experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import protocol as P


SCRIPT_VERSION = "2026-08-04.2"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REPAIR_FAILURE_SIGNATURE = "wrong actual NS5 call count for production_refresh_32"
REPAIR_SNAPSHOT_NAME = "source_snapshot_step_boundary_v2"
REPAIR_PLAN_NAME = "formal_job_plan_step_boundary_v2.json"
REPAIR_ACTIVATION_NAME = "source_repair_activation.json"
SOURCE_FILES = (
    "commands/46_mdp05_confirmatory_update_shock/20260804_ex46_mdp05_confirmatory_update_shock.sh",
    "scripts/46_mdp05_confirmatory_update_shock/README.md",
    "scripts/46_mdp05_confirmatory_update_shock/SHA256SUMS",
    "scripts/46_mdp05_confirmatory_update_shock/mdp05_contract.json",
    "scripts/46_mdp05_confirmatory_update_shock/protocol.py",
    "scripts/46_mdp05_confirmatory_update_shock/mdp05_worker.py",
    "scripts/46_mdp05_confirmatory_update_shock/analyze_mdp05.py",
    "scripts/46_mdp05_confirmatory_update_shock/inspect_mdp05.py",
    "scripts/46_mdp05_confirmatory_update_shock/pilot_precision.py",
    "scripts/46_mdp05_confirmatory_update_shock/run_mdp05.py",
    "scripts/46_mdp05_confirmatory_update_shock/smoke_worker.py",
    "scripts/46_mdp05_confirmatory_update_shock/test_mdp05.py",
    "scripts/mdp_refresh_streaming/stream_metrics.py",
    "scripts/mdp_refresh_streaming/pinned_ex37_runtime/triton_kernels.py",
    "scripts/37_mech09_downproj_refresh_mediation/mech09r_worker.py",
    "scripts/37_mech09_downproj_refresh_mediation/mech09_worker.py",
    "scripts/37_mech09_downproj_refresh_mediation/refresh_mediation_repair_contract.json",
    "scripts/37_mech09_downproj_refresh_mediation/mech08_control_reference.json",
    "scripts/36_mech08_short_horizon_rollout/mech08_worker.py",
    "scripts/27_mech01_unified_k_diagnostics/mech01_worker.py",
)


def executable_path(path: Path) -> str:
    return os.path.abspath(os.fspath(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_commands(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"command {number} is not an object")
        rows.append(value)
    return rows


def accepted_worker_args(command: dict[str, Any]) -> list[str]:
    values = [str(value) for value in command["command"]]
    if len(values) < 3:
        raise RuntimeError(f"accepted command is too short: {command['label']}")
    return values[2:]


def option_value(arguments: list[str], option: str) -> str:
    if option not in arguments:
        raise RuntimeError(f"missing worker option: {option}")
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise RuntimeError(f"worker option has no value: {option}")
    return arguments[index + 1]


def replace_option(arguments: list[str], option: str, value: str) -> None:
    index = arguments.index(option)
    arguments[index + 1] = value


def remove_option(arguments: list[str], option: str) -> None:
    if option in arguments:
        index = arguments.index(option)
        del arguments[index : index + 2]


def source_templates(source_run: Path) -> dict[str, list[str]]:
    rows = [
        row
        for row in read_commands(source_run / "commands.jsonl")
        if str(row.get("label", "")).startswith("formal/")
    ]
    templates: dict[str, list[str]] = {}
    for row in rows:
        _, origin, replica = str(row["label"]).split("/")
        if replica == "replica_0":
            templates[origin] = accepted_worker_args(row)
    if set(templates) != set(P.ORIGINS):
        raise RuntimeError(f"accepted command templates incomplete: {templates.keys()}")
    return templates


def write_source_snapshot(snapshot: Path) -> tuple[Path, Path]:
    manifest_path = snapshot / "source_snapshot_manifest.json"
    files: dict[str, str] = {}
    for relative in SOURCE_FILES:
        source = REPO / relative
        if not source.is_file():
            raise FileNotFoundError(f"required source is missing: {source}")
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[relative] = sha256_file(target)
    P.atomic_json(
        manifest_path,
        {
            "schema_version": "mdp05_source_snapshot_manifest_v1",
            "controller_version": SCRIPT_VERSION,
            "files": files,
            "passed": True,
        },
    )
    return snapshot, manifest_path


def verify_source_snapshot(snapshot: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = snapshot / "source_snapshot_manifest.json"
    manifest = P.read_json(manifest_path)
    checks = {
        relative: (snapshot / relative).is_file()
        and sha256_file(snapshot / relative) == expected
        for relative, expected in manifest["files"].items()
    }
    if not all(checks.values()):
        raise RuntimeError(f"sealed source snapshot changed: {snapshot}")
    return manifest_path, manifest


def live_source_hashes() -> dict[str, str]:
    hashes = {}
    for relative in SOURCE_FILES:
        source = REPO / relative
        if not source.is_file():
            raise FileNotFoundError(f"required source is missing: {source}")
        hashes[relative] = sha256_file(source)
    return hashes


def accepted_formal_selection_count(run_dir: Path) -> int:
    return sum(
        1
        for path in run_dir.glob("formal/*/replica_*/unit_selection.json")
        if P.read_json(path).get("passed") is True
    )


def repair_failure_evidence(run_dir: Path) -> list[str]:
    evidence = []
    for path in sorted(run_dir.glob("formal/*/replica_*/attempt_*/mdp05_status.json")):
        payload = P.read_json(path)
        if REPAIR_FAILURE_SIGNATURE in str(payload.get("error", "")):
            evidence.append(path.relative_to(run_dir).as_posix())
    return evidence


def snapshot_sources(run_dir: Path) -> tuple[Path, Path, dict[str, Any] | None]:
    primary = run_dir / "source_snapshot"
    if not primary.exists():
        snapshot, manifest = write_source_snapshot(primary)
        return snapshot, manifest, None

    primary_manifest_path, primary_manifest = verify_source_snapshot(primary)
    repair = run_dir / REPAIR_SNAPSHOT_NAME
    if repair.exists():
        repair_manifest_path, repair_manifest = verify_source_snapshot(repair)
        live_hashes = live_source_hashes()
        if repair_manifest["files"] != live_hashes:
            raise RuntimeError("live sources differ from the sealed MDP-05 repair snapshot")
        return (
            repair,
            repair_manifest_path,
            {
                "predecessor_snapshot": primary.name,
                "predecessor_snapshot_manifest_sha256": sha256_file(primary_manifest_path),
                "active_snapshot": repair.name,
                "active_snapshot_manifest_sha256": sha256_file(repair_manifest_path),
            },
        )

    live_hashes = live_source_hashes()
    if primary_manifest["files"] == live_hashes:
        return primary, primary_manifest_path, None

    failure_evidence = repair_failure_evidence(run_dir)
    repair_checks = {
        "resume_of_existing_run": (run_dir / "run_identity.json").is_file(),
        "zero_accepted_formal_units": accepted_formal_selection_count(run_dir) == 0,
        "analysis_not_opened": not (run_dir / "analysis" / "analysis_manifest.json").exists(),
        "known_pre_outcome_failure": bool(failure_evidence),
        "contract_unchanged": primary_manifest["files"].get(
            "scripts/46_mdp05_confirmatory_update_shock/mdp05_contract.json"
        )
        == live_hashes.get(
            "scripts/46_mdp05_confirmatory_update_shock/mdp05_contract.json"
        ),
        "base_worker_unchanged": all(
            primary_manifest["files"].get(relative) == live_hashes.get(relative)
            for relative in (
                "scripts/37_mech09_downproj_refresh_mediation/mech09r_worker.py",
                "scripts/37_mech09_downproj_refresh_mediation/mech09_worker.py",
                "scripts/36_mech08_short_horizon_rollout/mech08_worker.py",
                "scripts/27_mech01_unified_k_diagnostics/mech01_worker.py",
                "scripts/mdp_refresh_streaming/stream_metrics.py",
                "scripts/mdp_refresh_streaming/pinned_ex37_runtime/triton_kernels.py",
            )
        ),
    }
    if not all(repair_checks.values()):
        raise RuntimeError(
            "source snapshot differs and pre-outcome step-boundary repair is not allowed: "
            f"{repair_checks}"
        )
    repair_snapshot, repair_manifest_path = write_source_snapshot(repair)
    return (
        repair_snapshot,
        repair_manifest_path,
        {
            "predecessor_snapshot": primary.name,
            "predecessor_snapshot_manifest_sha256": sha256_file(primary_manifest_path),
            "active_snapshot": repair.name,
            "active_snapshot_manifest_sha256": sha256_file(repair_manifest_path),
            "repair_checks": repair_checks,
            "failure_evidence": failure_evidence,
        },
    )


def source_data_reference(source_run: Path) -> dict[str, Any]:
    train_hashes: set[str] = set()
    val_hashes: set[str] = set()
    units = []
    for origin in P.ORIGINS:
        for replica in (0, 1, 2):
            unit = source_run / "formal" / origin / f"replica_{replica}"
            stream_path = unit / "training_stream_contract.json"
            heldout_path = unit / "heldout_batch_contract.json"
            stream = P.read_json(stream_path)
            heldout = P.read_json(heldout_path)
            train_hashes.update(
                {stream["first_x_sha256"], stream["first_y_sha256"]}
            )
            for section in ("build", "evaluation"):
                for batch in heldout[section]["hashes"]:
                    val_hashes.update({batch["x_sha256"], batch["y_sha256"]})
            units.append(
                {
                    "origin": origin,
                    "data_replica": replica,
                    "training_contract_sha256": sha256_file(stream_path),
                    "heldout_contract_sha256": sha256_file(heldout_path),
                }
            )
    return {
        "schema_version": "mdp05_source_data_reference_v1",
        "source_run": str(source_run),
        "source_outcomes_read": False,
        "training_first_batch_hashes": sorted(train_hashes),
        "validation_batch_hashes": sorted(val_hashes),
        "units": units,
        "passed": len(units) == 12,
    }


def source_preflight(
    source_run: Path,
    protocol_path: Path,
    base_contract_path: Path,
    templates: dict[str, list[str]],
) -> dict[str, Any]:
    protocol = P.read_json(protocol_path)
    base = P.read_json(base_contract_path)
    formal_manifest = P.read_json(source_run / "formal" / "formal_manifest.json")
    source_paths = []
    for arguments in templates.values():
        for option in (
            "--checkpoint",
            "--checkpoint-hash-certificate",
            "--source-script",
            "--profile-script",
        ):
            source_paths.append(Path(option_value(arguments, option)))
    checks = {
        "protocol": all(P.validate_protocol(protocol).values()),
        "base_contract_hash": sha256_file(base_contract_path)
        == protocol["source_execution_contract"].get(
            "public_sha256", protocol["source_execution_contract"]["sha256"]
        ),
        "base_contract_identity": base.get("experiment") == "MECH-09R",
        "source_formal_passed": formal_manifest.get("passed") is True,
        "source_formal_units": int(formal_manifest.get("completed_jobs", -1)) == 12,
        "source_commands": (source_run / "commands.jsonl").is_file(),
        "source_paths": all(path.is_file() for path in source_paths),
        "source_outcomes_not_authorized": protocol["source_experiment"][
            "outcomes_used"
        ]
        is False,
    }
    return {"checks": checks, "passed": all(checks.values())}


def runtime_preflight(child_python: Path, gpus: list[str], expected: dict[str, Any]) -> dict[str, Any]:
    script = r'''
import json
import sys
import numpy
import torch
import triton
expected = json.loads(sys.argv[1])
checks = {
    "python": sys.version.split()[0] == expected["python"],
    "torch": torch.__version__ == expected["torch"],
    "torch_cuda": torch.version.cuda == expected["torch_cuda"],
    "triton": triton.__version__ == expected["triton"],
    "numpy": numpy.__version__ == expected["numpy"],
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count() == int(expected["gpu_count"]),
}
devices = []
if checks["cuda_available"]:
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        row = {
            "index": index,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
        }
        row["passed"] = (
            expected["gpu_name_contains"] in properties.name
            and row["compute_capability"] == expected["compute_capability"]
            and properties.total_memory >= int(expected["minimum_gpu_memory_bytes"])
        )
        devices.append(row)
checks["devices"] = len(devices) == int(expected["gpu_count"]) and all(
    row["passed"] for row in devices
)
print(json.dumps({"checks": checks, "devices": devices, "passed": all(checks.values())}, sort_keys=True))
'''
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    completed = subprocess.run(
        [executable_path(child_python), "-c", script, json.dumps(expected)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if not completed.stdout.strip():
        raise RuntimeError(
            f"runtime preflight produced no JSON rc={completed.returncode}: {completed.stderr}"
        )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    payload["child_python"] = executable_path(child_python)
    payload["return_code"] = completed.returncode
    payload["passed"] = payload.get("passed") is True and completed.returncode == 0
    return payload


def validate_pilot(path: Path, protocol_sha: str) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "passed": False, "reason": "missing"}
    payload = P.read_json(path)
    checks = {
        "schema": payload.get("schema_version")
        == "mdp05_precision_pilot_certificate_v1",
        "contract": payload.get("contract_sha256") == protocol_sha,
        "outcome_blind": payload.get("outcome_blind") is True,
        "no_data": payload.get("checkpoint_or_dataset_opened") is False,
        "mode": payload.get("selected_mode") == "fixed_float64_slice_diagnostic",
        "passed": payload.get("passed") is True,
    }
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "checks": checks,
        "passed": all(checks.values()),
    }


def next_attempt(unit_dir: Path) -> Path:
    existing = [
        int(path.name.split("_")[-1])
        for path in unit_dir.glob("attempt_[0-9][0-9][0-9]")
        if path.is_dir()
    ]
    number = max(existing, default=0) + 1
    attempt = unit_dir / f"attempt_{number:03d}"
    attempt.mkdir(parents=False, exist_ok=False)
    return attempt


def selected_passed(
    unit_dir: Path, protocol_sha: str, execution_sha: str
) -> bool:
    selection_path = unit_dir / "unit_selection.json"
    if not selection_path.is_file():
        return False
    selection = P.read_json(selection_path)
    attempt = unit_dir / str(selection.get("selected_attempt", ""))
    manifest_path = attempt / "mdp05_unit_manifest.json"
    log_seal_path = attempt / "worker_log_seal.json"
    if not manifest_path.is_file() or not log_seal_path.is_file():
        return False
    manifest = P.read_json(manifest_path)
    seal = P.read_json(log_seal_path)
    return (
        selection.get("passed") is True
        and selection.get("manifest_sha256") == sha256_file(manifest_path)
        and manifest.get("passed") is True
        and manifest.get("mdp05_contract_sha256") == protocol_sha
        and manifest.get("execution_contract_sha256") == execution_sha
        and seal.get("sealed_after_worker_exit") is True
        and seal.get("sha256") == sha256_file(attempt / "worker.log")
    )


def seal_worker_log(attempt: Path, return_code: int) -> dict[str, Any]:
    log = attempt / "worker.log"
    payload = {
        "schema_version": "mdp05_worker_log_seal_v1",
        "return_code": int(return_code),
        "bytes": log.stat().st_size,
        "sha256": sha256_file(log),
        "sealed_after_worker_exit": True,
    }
    P.atomic_json(attempt / "worker_log_seal.json", payload)
    return payload


def validate_and_select(job: dict[str, Any], return_code: int) -> dict[str, Any]:
    attempt = job["attempt"]
    seal = seal_worker_log(attempt, return_code)
    manifest_path = attempt / job["manifest_name"]
    checks = {
        "return_code": return_code == 0,
        "manifest_exists": manifest_path.is_file(),
        "status_exists": (attempt / job["status_name"]).is_file(),
        "log_sealed": seal["sealed_after_worker_exit"] is True,
    }
    manifest: dict[str, Any] = {}
    if checks["manifest_exists"]:
        manifest = P.read_json(manifest_path)
        checks["manifest_passed"] = manifest.get("passed") is True
        checks["identity"] = (
            manifest.get(job["origin_field"]) == job["origin"]
            and int(manifest.get("data_replica", -1)) == int(job["replica"])
        )
        if job["manifest_name"] == "mdp05_unit_manifest.json":
            checks["contract"] = manifest.get("mdp05_contract_sha256") == job[
                "protocol_sha"
            ]
            checks["execution_contract"] = manifest.get(
                "execution_contract_sha256"
            ) == job["execution_sha"]
            checks["worker_log_excluded"] = "worker.log" not in manifest.get(
                "scientific_artifact_sha256", {}
            )
            checks["scientific_hashes"] = all(
                (attempt / name).is_file()
                and sha256_file(attempt / name) == expected
                for name, expected in manifest.get(
                    "scientific_artifact_sha256", {}
                ).items()
            )
        else:
            checks["execution_contract"] = manifest.get("contract_sha256") == job[
                "execution_sha"
            ]
    passed = all(checks.values())
    selection = {
        "schema_version": "mdp05_unit_selection_v1",
        "selected_attempt": attempt.name if passed else None,
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "worker_log_seal_sha256": sha256_file(attempt / "worker_log_seal.json"),
        "checks": checks,
        "passed": passed,
    }
    if passed:
        P.atomic_json(job["unit_dir"] / "unit_selection.json", selection)
    return selection


def base_arguments(
    template: list[str],
    *,
    output: Path,
    tier: str,
    origin: str,
    replica: int,
    execution_contract: Path,
    snapshot: Path,
    smoke_manifest: Path | None,
) -> list[str]:
    arguments = list(template)
    for option, value in (
        ("--output-dir", str(output)),
        ("--analysis-tier", tier),
        ("--cell", origin),
        ("--data-replica", str(replica)),
        ("--contract", str(execution_contract)),
        (
            "--triton-kernels",
            str(
                snapshot
                / "scripts/mdp_refresh_streaming/pinned_ex37_runtime/triton_kernels.py"
            ),
        ),
        (
            "--mech08-control-reference",
            str(
                snapshot
                / "scripts/37_mech09_downproj_refresh_mediation/mech08_control_reference.json"
            ),
        ),
    ):
        replace_option(arguments, option, value)
    if smoke_manifest is None:
        remove_option(arguments, "--smoke-manifest")
    elif "--smoke-manifest" in arguments:
        replace_option(arguments, "--smoke-manifest", str(smoke_manifest))
    else:
        arguments.extend(["--smoke-manifest", str(smoke_manifest)])
    return arguments


def smoke_job(
    args: argparse.Namespace,
    run_dir: Path,
    snapshot: Path,
    execution_contract: Path,
    template: list[str],
) -> dict[str, Any] | None:
    protocol = P.read_json(snapshot / "scripts/46_mdp05_confirmatory_update_shock/mdp05_contract.json")
    origin = protocol["design"]["smoke_origin"]
    replica = int(protocol["design"]["smoke_data_replica"])
    unit_dir = run_dir / "smoke" / origin / f"replica_{replica}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    selection_path = unit_dir / "unit_selection.json"
    if selection_path.is_file():
        selection = P.read_json(selection_path)
        attempt = unit_dir / selection["selected_attempt"]
        manifest = attempt / "mech09r_manifest.json"
        if (
            selection.get("passed") is True
            and manifest.is_file()
            and P.read_json(manifest).get("passed") is True
            and P.read_json(manifest).get("contract_sha256")
            == sha256_file(execution_contract)
        ):
            print("skip passed MDP-05 smoke", flush=True)
            return None
    attempt = next_attempt(unit_dir)
    worker = snapshot / "scripts/46_mdp05_confirmatory_update_shock/smoke_worker.py"
    arguments = base_arguments(
        template,
        output=attempt,
        tier="smoke",
        origin=origin,
        replica=replica,
        execution_contract=execution_contract,
        snapshot=snapshot,
        smoke_manifest=None,
    )
    return {
        "label": f"smoke/{origin}/replica_{replica}",
        "origin": origin,
        "replica": replica,
        "gpu": args.gpus[0],
        "attempt": attempt,
        "unit_dir": unit_dir,
        "command": [executable_path(args.child_python), str(worker), *arguments],
        "manifest_name": "mech09r_manifest.json",
        "status_name": "status.json",
        "origin_field": "checkpoint_cell",
        "protocol_sha": sha256_file(
            snapshot / "scripts/46_mdp05_confirmatory_update_shock/mdp05_contract.json"
        ),
        "execution_sha": sha256_file(execution_contract),
    }


def formal_jobs(
    args: argparse.Namespace,
    run_dir: Path,
    snapshot: Path,
    snapshot_manifest: Path,
    execution_contract: Path,
    templates: dict[str, list[str]],
    smoke_manifest: Path,
) -> list[dict[str, Any]]:
    contract_path = snapshot / "scripts/46_mdp05_confirmatory_update_shock/mdp05_contract.json"
    protocol = P.read_json(contract_path)
    protocol_sha = sha256_file(contract_path)
    execution_sha = sha256_file(execution_contract)
    worker = snapshot / "scripts/46_mdp05_confirmatory_update_shock/mdp05_worker.py"
    jobs = []
    index = 0
    for origin in P.ORIGINS:
        for replica in protocol["design"]["formal_data_replicas"]:
            unit_dir = run_dir / "formal" / origin / f"replica_{replica}"
            unit_dir.mkdir(parents=True, exist_ok=True)
            if selected_passed(unit_dir, protocol_sha, execution_sha):
                print(f"skip passed unit: {origin}/replica_{replica}", flush=True)
                index += 1
                continue
            attempt = next_attempt(unit_dir)
            arguments = base_arguments(
                templates[origin],
                output=attempt,
                tier="formal",
                origin=origin,
                replica=int(replica),
                execution_contract=execution_contract,
                snapshot=snapshot,
                smoke_manifest=smoke_manifest,
            )
            command = [
                executable_path(args.child_python),
                str(worker),
                "--mdp05-output-dir",
                str(attempt),
                "--mdp05-contract",
                str(contract_path),
                "--source-snapshot-manifest",
                str(snapshot_manifest),
                "--",
                *arguments,
            ]
            jobs.append(
                {
                    "label": f"formal/{origin}/replica_{replica}",
                    "origin": origin,
                    "replica": int(replica),
                    "gpu": args.gpus[index % len(args.gpus)],
                    "attempt": attempt,
                    "unit_dir": unit_dir,
                    "command": command,
                    "manifest_name": "mdp05_unit_manifest.json",
                    "status_name": "mdp05_status.json",
                    "origin_field": "origin",
                    "protocol_sha": protocol_sha,
                    "execution_sha": execution_sha,
                }
            )
            index += 1
    return jobs


def run_jobs(jobs: list[dict[str, Any]], max_parallel: int) -> list[dict[str, Any]]:
    pending = list(jobs)
    active: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    failure_seen = False
    while pending or active:
        while pending and len(active) < max_parallel and not failure_seen:
            job = pending.pop(0)
            log = (job["attempt"] / "worker.log").open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            P.atomic_json(
                job["attempt"] / "launch.json",
                {
                    "schema_version": "mdp05_attempt_launch_v1",
                    "label": job["label"],
                    "gpu": str(job["gpu"]),
                    "command": job["command"],
                    "environment": {
                        "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                },
            )
            print(f"launch gpu={job['gpu']} {job['label']}", flush=True)
            process = subprocess.Popen(
                job["command"],
                cwd=str(job["attempt"]),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active.append(job | {"process": process, "log_handle": log})
        if not active:
            break
        time.sleep(1.0)
        still_active = []
        for job in active:
            return_code = job["process"].poll()
            if return_code is None:
                still_active.append(job)
                continue
            job["log_handle"].flush()
            job["log_handle"].close()
            selection = validate_and_select(job, int(return_code))
            result = {
                "label": job["label"],
                "gpu": job["gpu"],
                "return_code": int(return_code),
                "attempt": str(job["attempt"]),
                "selection_passed": selection["passed"],
            }
            results.append(result)
            if selection["passed"]:
                print(f"passed {job['label']}", flush=True)
            else:
                print(
                    f"failed {job['label']} rc={return_code}; "
                    f"see {job['attempt'] / 'worker.log'}",
                    file=sys.stderr,
                    flush=True,
                )
                failure_seen = True
        active = still_active
    for job in pending:
        results.append(
            {
                "label": job["label"],
                "gpu": job["gpu"],
                "return_code": None,
                "attempt": str(job["attempt"]),
                "selection_passed": False,
                "not_launched_after_failure": True,
            }
        )
    return results


def all_formal_selected(
    run_dir: Path, protocol: dict[str, Any], protocol_sha: str, execution_sha: str
) -> bool:
    return all(
        selected_passed(
            run_dir / "formal" / origin / f"replica_{replica}",
            protocol_sha,
            execution_sha,
        )
        for origin in P.ORIGINS
        for replica in protocol["design"]["formal_data_replicas"]
    )


def create_handoff(run_dir: Path) -> dict[str, Any]:
    selected = []
    for selection_path in sorted(run_dir.glob("formal/*/replica_*/unit_selection.json")):
        selection = P.read_json(selection_path)
        attempt = selection_path.parent / selection["selected_attempt"]
        for name in (
            "mdp05_unit_manifest.json",
            "worker_log_seal.json",
            "mdp05_refresh_layer_metrics.csv",
            "evaluation.csv",
            "training_stream_contract.json",
            "heldout_batch_contract.json",
        ):
            path = attempt / name
            selected.append(
                {
                    "path": path.relative_to(run_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    analysis = []
    for path in sorted((run_dir / "analysis").iterdir()):
        if path.is_file():
            analysis.append(
                {
                    "path": path.relative_to(run_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    provenance = []
    for name in (
        REPAIR_ACTIVATION_NAME,
        REPAIR_PLAN_NAME,
        "formal_job_plan.json",
    ):
        path = run_dir / "sealed" / name
        if path.is_file():
            provenance.append(
                {
                    "path": path.relative_to(run_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    large = [
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file() and path.stat().st_size > 250_000_000
    ]
    payload = {
        "schema_version": "mdp05_handoff_manifest_v1",
        "run_dir": str(run_dir),
        "selected_artifacts": selected,
        "analysis_artifacts": analysis,
        "provenance_artifacts": provenance,
        "files_over_250mb": large,
        "raw_full_matrices_persisted": False,
        "passed": not large,
    }
    P.atomic_json(run_dir / "handoff_manifest.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--child-python", required=True, type=Path)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--pilot-certificate", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def controller(args: argparse.Namespace) -> int:
    if args.dry_run and args.resume:
        raise RuntimeError("--dry-run and --resume are mutually exclusive")
    run_dir = args.run_dir.resolve()
    source_run = args.source_run.resolve()
    live_contract = HERE / "mdp05_contract.json"
    protocol_sha = sha256_file(live_contract)
    if args.resume:
        identity = P.read_json(run_dir / "run_identity.json")
        checks = {
            "experiment": identity.get("experiment") == "MDP-05",
            "contract": identity.get("contract_sha256") == protocol_sha,
            "not_dry_run": identity.get("dry_run") is False,
        }
        if not all(checks.values()):
            raise RuntimeError(f"resume identity mismatch: {checks}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        P.atomic_json(
            run_dir / "run_identity.json",
            {
                "schema_version": "mdp05_run_identity_v1",
                "experiment": "MDP-05",
                "experiment_number": 46,
                "controller_version": SCRIPT_VERSION,
                "contract_sha256": protocol_sha,
                "dry_run": bool(args.dry_run),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    P.atomic_json(
        run_dir / "status.json",
        {"status": "preflight", "controller_version": SCRIPT_VERSION},
    )
    snapshot, snapshot_manifest, source_repair = snapshot_sources(run_dir)
    contract_path = snapshot / "scripts/46_mdp05_confirmatory_update_shock/mdp05_contract.json"
    base_contract_path = snapshot / "scripts/37_mech09_downproj_refresh_mediation/refresh_mediation_repair_contract.json"
    protocol = P.read_json(contract_path)
    templates = source_templates(source_run)
    preflight = source_preflight(
        source_run, contract_path, base_contract_path, templates
    )
    preflight["checks"].update(
        {
            "child_python_exists": args.child_python.is_file(),
            "two_gpu_ids": len(args.gpus) == 2,
            "gpu_ids_unique": len(set(args.gpus)) == 2,
            "max_parallel": 1 <= int(args.max_parallel) <= 2,
        }
    )
    preflight["passed"] = all(preflight["checks"].values())
    P.atomic_json(run_dir / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError(f"source preflight failed: {preflight['checks']}")

    sealed = run_dir / "sealed"
    sealed.mkdir(exist_ok=args.resume)
    execution_contract = sealed / "derived_execution_contract.json"
    derived = P.derive_execution_contract(
        P.read_json(base_contract_path), protocol, protocol_sha
    )
    if execution_contract.is_file():
        existing = P.read_json(execution_contract)
        if existing != derived:
            raise RuntimeError("sealed derived execution contract changed")
    else:
        P.atomic_json(execution_contract, derived)
    execution_sha = sha256_file(execution_contract)

    val_pattern = option_value(templates[P.ORIGINS[0]], "--val-data-pattern")
    val_files = [Path(path) for path in sorted(glob.glob(val_pattern))]
    offset_certificate = P.build_offset_certificate(protocol, val_files)
    offset_path = sealed / "offset_collision_certificate.json"
    if offset_path.is_file() and P.read_json(offset_path) != offset_certificate:
        raise RuntimeError("sealed offset collision certificate changed")
    P.atomic_json(offset_path, offset_certificate)
    if not offset_certificate["passed"]:
        raise RuntimeError(
            f"offset collision audit failed: {offset_certificate['checks']}"
        )
    reference = source_data_reference(source_run)
    reference_path = sealed / "source_data_reference.json"
    if reference_path.is_file() and P.read_json(reference_path) != reference:
        raise RuntimeError("sealed source data reference changed")
    P.atomic_json(reference_path, reference)

    plan = {
        "schema_version": "mdp05_formal_job_plan_v1",
        "contract_sha256": protocol_sha,
        "execution_contract_sha256": execution_sha,
        "source_snapshot_manifest_sha256": sha256_file(snapshot_manifest),
        "offset_certificate_sha256": sha256_file(offset_path),
        "source_data_reference_sha256": sha256_file(reference_path),
        "formal_units": 12,
        "logical_units": [
            {"origin": origin, "data_replica": int(replica)}
            for origin in P.ORIGINS
            for replica in protocol["design"]["formal_data_replicas"]
        ],
        "outcomes_may_be_opened_after_selected_units": 12,
        "passed": True,
    }
    original_plan_path = sealed / "formal_job_plan.json"
    if source_repair is None:
        plan_path = original_plan_path
        if plan_path.is_file() and P.read_json(plan_path) != plan:
            raise RuntimeError("sealed formal job plan changed")
        P.atomic_json(plan_path, plan)
    else:
        if not original_plan_path.is_file():
            raise RuntimeError("pre-outcome repair requires the predecessor formal plan")
        predecessor_plan = P.read_json(original_plan_path)
        invariant_fields = (
            "contract_sha256",
            "execution_contract_sha256",
            "offset_certificate_sha256",
            "source_data_reference_sha256",
            "formal_units",
            "logical_units",
            "outcomes_may_be_opened_after_selected_units",
        )
        invariant_checks = {
            field: predecessor_plan.get(field) == plan.get(field)
            for field in invariant_fields
        }
        if not all(invariant_checks.values()):
            raise RuntimeError(
                f"scientific job plan changed during source repair: {invariant_checks}"
            )
        plan["pre_outcome_source_repair"] = {
            "failure_signature": REPAIR_FAILURE_SIGNATURE,
            "predecessor_plan": original_plan_path.name,
            "predecessor_plan_sha256": sha256_file(original_plan_path),
            "repair_scope": "move the NS5 count gate from _apply_preconditioners return to full optimizer.step return",
            "scientific_design_changed": False,
            "failed_attempts_reused": False,
        }
        plan_path = sealed / REPAIR_PLAN_NAME
        if plan_path.is_file() and P.read_json(plan_path) != plan:
            raise RuntimeError("sealed repair formal job plan changed")
        P.atomic_json(plan_path, plan)
        activation_path = sealed / REPAIR_ACTIVATION_NAME
        if activation_path.is_file():
            activation = P.read_json(activation_path)
            activation_checks = {
                "passed": activation.get("passed") is True,
                "active_plan": activation.get("active_plan") == plan_path.name,
                "active_plan_sha256": activation.get("active_plan_sha256")
                == sha256_file(plan_path),
                "active_snapshot": activation.get("active_snapshot") == snapshot.name,
                "active_snapshot_manifest_sha256": activation.get(
                    "active_snapshot_manifest_sha256"
                )
                == sha256_file(snapshot_manifest),
            }
            if not all(activation_checks.values()):
                raise RuntimeError(
                    f"sealed source repair activation changed: {activation_checks}"
                )
        else:
            if "repair_checks" not in source_repair:
                raise RuntimeError("source repair activation evidence is unavailable")
            activation = {
                "schema_version": "mdp05_pre_outcome_source_repair_v1",
                "failure_signature": REPAIR_FAILURE_SIGNATURE,
                "failure_evidence": source_repair["failure_evidence"],
                "repair_checks": source_repair["repair_checks"],
                "predecessor_snapshot": source_repair["predecessor_snapshot"],
                "predecessor_snapshot_manifest_sha256": source_repair[
                    "predecessor_snapshot_manifest_sha256"
                ],
                "active_snapshot": snapshot.name,
                "active_snapshot_manifest_sha256": sha256_file(snapshot_manifest),
                "predecessor_plan": original_plan_path.name,
                "predecessor_plan_sha256": sha256_file(original_plan_path),
                "active_plan": plan_path.name,
                "active_plan_sha256": sha256_file(plan_path),
                "accepted_formal_units_before_repair": 0,
                "analysis_opened_before_repair": False,
                "failed_attempts_reused": False,
                "scientific_contract_changed": False,
                "passed": True,
            }
            P.atomic_json(activation_path, activation)
        print(
            "MDP-05 pre-outcome source repair active: "
            f"snapshot={snapshot.name} plan={plan_path.name}",
            flush=True,
        )

    if args.dry_run:
        P.atomic_json(
            run_dir / "status.json",
            {
                "status": "dry_run_passed",
                "controller_version": SCRIPT_VERSION,
                "formal_units_planned": 12,
            },
        )
        print(f"MDP-05 sealed dry-run passed: {run_dir}", flush=True)
        return 0

    if args.pilot_certificate is None:
        raise RuntimeError("formal/resume requires --pilot-certificate")
    pilot = validate_pilot(args.pilot_certificate.resolve(), protocol_sha)
    pilot_audit_path = sealed / "pilot_certificate_audit.json"
    if pilot_audit_path.is_file() and P.read_json(pilot_audit_path) != pilot:
        raise RuntimeError("resume pilot certificate differs from the sealed pilot")
    P.atomic_json(pilot_audit_path, pilot)
    if not pilot["passed"]:
        raise RuntimeError(f"pilot certificate failed: {pilot}")
    runtime = runtime_preflight(
        args.child_python, args.gpus, protocol["runtime_contract"]
    )
    P.atomic_json(run_dir / "runtime_preflight.json", runtime)
    if not runtime["passed"]:
        raise RuntimeError(f"runtime preflight failed: {runtime}")
    P.atomic_json(
        run_dir / "status.json",
        {"status": "running_smoke", "controller_version": SCRIPT_VERSION},
    )
    smoke = smoke_job(
        args,
        run_dir,
        snapshot,
        execution_contract,
        templates[protocol["design"]["smoke_origin"]],
    )
    if smoke is not None:
        smoke_results = run_jobs([smoke], 1)
        P.atomic_json(run_dir / "smoke_jobs.json", {"jobs": smoke_results})
        if not all(row["selection_passed"] for row in smoke_results):
            raise RuntimeError("MDP-05 smoke failed")
    smoke_unit = (
        run_dir
        / "smoke"
        / protocol["design"]["smoke_origin"]
        / f"replica_{protocol['design']['smoke_data_replica']}"
    )
    smoke_selection = P.read_json(smoke_unit / "unit_selection.json")
    smoke_manifest = (
        smoke_unit
        / smoke_selection["selected_attempt"]
        / "mech09r_manifest.json"
    )

    P.atomic_json(
        run_dir / "status.json",
        {"status": "running_formal", "controller_version": SCRIPT_VERSION},
    )
    jobs = formal_jobs(
        args,
        run_dir,
        snapshot,
        snapshot_manifest,
        execution_contract,
        templates,
        smoke_manifest,
    )
    results = run_jobs(jobs, min(int(args.max_parallel), len(args.gpus)))
    attempt_history_path = run_dir / "formal_job_attempts.json"
    attempt_history = (
        P.read_json(attempt_history_path)
        if attempt_history_path.is_file()
        else {"schema_version": "mdp05_job_attempts_v1", "waves": []}
    )
    attempt_history["waves"].append(
        {
            "controller_version": SCRIPT_VERSION,
            "resume": bool(args.resume),
            "jobs": results,
        }
    )
    P.atomic_json(attempt_history_path, attempt_history)
    if not all_formal_selected(
        run_dir, protocol, protocol_sha, execution_sha
    ):
        raise RuntimeError(
            "formal incomplete; use resume with the same run directory and contract"
        )

    analyzer = snapshot / "scripts/46_mdp05_confirmatory_update_shock/analyze_mdp05.py"
    completed = subprocess.run(
        [
            executable_path(Path(sys.executable)),
            str(analyzer),
            "--run-dir",
            str(run_dir),
            "--contract",
            str(contract_path),
        ],
        check=False,
        cwd=str(run_dir),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"MDP-05 final analysis integrity failed rc={completed.returncode}"
        )
    analysis_manifest = P.read_json(run_dir / "analysis" / "analysis_manifest.json")
    handoff = create_handoff(run_dir)
    if not handoff["passed"]:
        raise RuntimeError("handoff contains an unexpected file over 250MB")
    P.atomic_json(
        run_dir / "status.json",
        {
            "status": "completed",
            "controller_version": SCRIPT_VERSION,
            "integrity_passed": analysis_manifest["integrity_passed"],
            "scientific_result": analysis_manifest["scientific_result"],
            "claim_success": analysis_manifest["claim_success"],
        },
    )
    print("MDP-05 completed.", flush=True)
    print(f"Artifacts: {run_dir}", flush=True)
    print(f"Analysis: {run_dir / 'analysis' / 'analysis_manifest.json'}", flush=True)
    print(f"Scientific result: {analysis_manifest['scientific_result']}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    try:
        return controller(args)
    except BaseException as exc:
        run_dir = args.run_dir.resolve()
        if run_dir.is_dir():
            P.atomic_json(
                run_dir / "status.json",
                {
                    "status": "failed_or_incomplete",
                    "controller_version": SCRIPT_VERSION,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "resume_supported": not args.dry_run,
                },
            )
        print(
            f"MDP-05 stopped cleanly: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
