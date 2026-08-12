#!/usr/bin/env python3
"""Seal, schedule, resume, and validate the MDP-04 refresh stream replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


SCRIPT_VERSION = "2026-08-03.10"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def executable_path(path: Path) -> str:
    """Return an absolute executable path without dereferencing venv symlinks."""

    return os.path.abspath(os.fspath(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def predecessor_contract_allowed(
    current: dict[str, Any], predecessor: dict[str, Any], predecessor_sha256: str
) -> bool:
    compatibility = current.get("resume_compatibility", {})
    if not isinstance(compatibility, dict):
        return False
    if any(
        compatibility.get(field) is not True
        for field in (
            "failed_or_incomplete_predecessor_attempts_are_never_selected",
            "predecessor_source_snapshot_remains_immutable",
            "v4_workers_use_a_new_sealed_source_snapshot",
            "mixed_contract_lineage_must_be_reported_by_final_validator",
        )
    ):
        return False
    return any(
        isinstance(row, dict)
        and row.get("schema_version") == predecessor.get("schema_version")
        and row.get("sha256") == predecessor_sha256
        and row.get("reuse_scope") == "selected passed units only"
        for row in compatibility.get("allowed_predecessors", [])
    )


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_commands(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"command line {line_number} is not an object")
        rows.append(value)
    return rows


def snapshot_sources(run_dir: Path, contract_path: Path) -> tuple[Path, Path]:
    snapshot = run_dir / "source_snapshot"
    migration: dict[str, Any] | None = None
    source_files = [
        "scripts/mdp_refresh_streaming/README.md",
        "scripts/mdp_refresh_streaming/SHA256SUMS",
        "scripts/mdp_refresh_streaming/run_stream_replay.py",
        "scripts/mdp_refresh_streaming/stream_worker.py",
        "scripts/mdp_refresh_streaming/stream_metrics.py",
        "scripts/mdp_refresh_streaming/validate_stream_replay.py",
        "scripts/mdp_refresh_streaming/test_streaming.py",
        "scripts/mdp_refresh_streaming/refresh_stream_contract.json",
        "scripts/mdp_refresh_streaming/pinned_ex37_runtime/triton_kernels.py",
        "scripts/37_mech09_downproj_refresh_mediation/mech09r_worker.py",
        "scripts/37_mech09_downproj_refresh_mediation/mech09_worker.py",
        "scripts/37_mech09_downproj_refresh_mediation/refresh_mediation_repair_contract.json",
        "scripts/37_mech09_downproj_refresh_mediation/mech08_control_reference.json",
        "scripts/36_mech08_short_horizon_rollout/mech08_worker.py",
        "scripts/27_mech01_unified_k_diagnostics/mech01_worker.py",
    ]
    if snapshot.exists():
        manifest_path = snapshot / "source_snapshot_manifest.json"
        manifest = read_json(manifest_path)
        for relative, expected in manifest["files"].items():
            observed = sha256_file(snapshot / relative)
            if observed != expected:
                raise RuntimeError(
                    f"sealed source snapshot changed: {relative} {observed} != {expected}"
                )
        sealed_contract = (
            snapshot
            / "scripts"
            / "mdp_refresh_streaming"
            / "refresh_stream_contract.json"
        )
        sealed_sha256 = sha256_file(sealed_contract)
        live_sha256 = sha256_file(contract_path)
        if sealed_sha256 == live_sha256:
            return snapshot, manifest_path
        predecessor = read_json(sealed_contract)
        current = read_json(contract_path)
        if not predecessor_contract_allowed(current, predecessor, sealed_sha256):
            raise RuntimeError("live stream contract differs from the sealed run")
        schema_suffix = str(current["schema_version"]).rsplit("_", 1)[-1]
        migration = {
            "schema_version": "mdp04_resume_contract_migration_v1",
            "predecessor_schema_version": predecessor.get("schema_version"),
            "predecessor_contract_sha256": sealed_sha256,
            "predecessor_source_snapshot": str(snapshot),
            "predecessor_source_snapshot_manifest_sha256": sha256_file(manifest_path),
            "current_schema_version": current.get("schema_version"),
            "current_contract_sha256": live_sha256,
            "reuse_scope": "selected passed units only",
            "predecessor_snapshot_immutable": True,
            "passed": True,
        }
        snapshot = run_dir / f"source_snapshot_{schema_suffix}"
        if snapshot.exists():
            manifest_path = snapshot / "source_snapshot_manifest.json"
            manifest = read_json(manifest_path)
            for relative, expected in manifest["files"].items():
                observed = sha256_file(snapshot / relative)
                if observed != expected:
                    raise RuntimeError(
                        f"sealed migrated source snapshot changed: {relative} "
                        f"{observed} != {expected}"
                    )
            migrated_contract = (
                snapshot
                / "scripts"
                / "mdp_refresh_streaming"
                / "refresh_stream_contract.json"
            )
            if sha256_file(migrated_contract) != live_sha256:
                raise RuntimeError("migrated stream contract differs from live v4")
            migration_path = run_dir / "resume_contract_migration_v3_to_v4.json"
            if not migration_path.is_file() or read_json(migration_path) != migration:
                raise RuntimeError("resume contract migration audit is missing or changed")
            return snapshot, manifest_path
    snapshot.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for relative in source_files:
        source = REPO / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[relative] = sha256_file(destination)
    contract_destination = (
        snapshot
        / "scripts"
        / "mdp_refresh_streaming"
        / "refresh_stream_contract.json"
    )
    if sha256_file(contract_destination) != sha256_file(contract_path):
        raise RuntimeError("snapshotted stream contract does not match requested contract")
    manifest_path = snapshot / "source_snapshot_manifest.json"
    atomic_json(
        manifest_path,
        {
            "schema_version": "mdp04_source_snapshot_manifest_v1",
            "controller_version": SCRIPT_VERSION,
            "repo": str(REPO),
            "files": hashes,
            "passed": True,
        },
    )
    if migration is not None:
        migration_path = run_dir / "resume_contract_migration_v3_to_v4.json"
        if migration_path.exists() and read_json(migration_path) != migration:
            raise RuntimeError("resume contract migration audit changed")
        atomic_json(migration_path, migration)
    return snapshot, manifest_path


def next_attempt(unit_dir: Path) -> Path:
    existing = sorted(
        int(path.name.split("_")[-1])
        for path in unit_dir.glob("attempt_*")
        if path.is_dir() and path.name.split("_")[-1].isdigit()
    )
    number = (existing[-1] + 1) if existing else 1
    attempt = unit_dir / f"attempt_{number:03d}"
    attempt.mkdir(parents=True)
    return attempt


def selected_unit_passed(unit_dir: Path) -> bool:
    selection_path = unit_dir / "unit_selection.json"
    if not selection_path.is_file():
        return False
    selection = read_json(selection_path)
    manifest_path = unit_dir / selection["selected_attempt"] / "stream_unit_manifest.json"
    return (
        manifest_path.is_file()
        and sha256_file(manifest_path) == selection.get("manifest_sha256")
        and read_json(manifest_path).get("passed") is True
    )


def accepted_worker_args(command: dict[str, Any]) -> list[str]:
    values = list(command["command"])
    if len(values) < 3:
        raise RuntimeError(f"accepted command is too short: {command['label']}")
    return [str(value) for value in values[2:]]


def build_jobs(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    source_run: Path,
    snapshot: Path,
    snapshot_manifest: Path,
    contract: Path,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_commands(source_run / "commands.jsonl")
        if str(row["label"]).startswith("formal/")
    ]
    if len(rows) != 12:
        raise RuntimeError(f"expected 12 accepted formal commands, got {len(rows)}")
    jobs: list[dict[str, Any]] = []
    gpu_map = {str(index): args.gpus[index] for index in range(len(args.gpus))}
    for row in rows:
        _, origin, replica_name = str(row["label"]).split("/")
        replica = int(replica_name.split("_")[-1])
        unit_dir = run_dir / "formal" / origin / f"replica_{replica}"
        unit_dir.mkdir(parents=True, exist_ok=True)
        if args.resume and selected_unit_passed(unit_dir):
            print(f"skip passed unit: {origin}/replica_{replica}", flush=True)
            continue
        if not args.resume and any(unit_dir.iterdir()):
            raise RuntimeError(
                f"unit directory is not empty; use --resume: {unit_dir}"
            )
        attempt = next_attempt(unit_dir)
        recorded_gpu = str(row["gpu"])
        if recorded_gpu not in gpu_map:
            raise RuntimeError(
                f"recorded GPU {recorded_gpu} has no mapping in --gpus {args.gpus}"
            )
        worker = (
            snapshot
            / "scripts"
            / "mdp_refresh_streaming"
            / "stream_worker.py"
        )
        accepted_unit = source_run / "formal" / origin / f"replica_{replica}"
        command = [
            executable_path(args.child_python),
            str(worker),
            "--stream-output-dir",
            str(attempt),
            "--stream-contract",
            str(contract),
            "--accepted-unit-dir",
            str(accepted_unit),
            "--source-snapshot-manifest",
            str(snapshot_manifest),
            "--",
            *accepted_worker_args(row),
        ]
        jobs.append(
            {
                "origin": origin,
                "replica": replica,
                "gpu": gpu_map[recorded_gpu],
                "attempt": attempt,
                "unit_dir": unit_dir,
                "command": command,
            }
        )
    return jobs


def run_jobs(jobs: list[dict[str, Any]], max_parallel: int) -> None:
    pending = list(jobs)
    active: list[dict[str, Any]] = []
    while pending or active:
        while pending and len(active) < max_parallel:
            job = pending.pop(0)
            log_path = job["attempt"] / "worker.log"
            log_handle = log_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            process = subprocess.Popen(
                job["command"],
                cwd=str(job["attempt"]),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            job.update(
                {
                    "process": process,
                    "log_handle": log_handle,
                    "log_path": log_path,
                }
            )
            active.append(job)
            print(
                f"started {job['origin']}/replica_{job['replica']} "
                f"on GPU {job['gpu']} pid={process.pid}",
                flush=True,
            )
        time.sleep(1.0)
        for job in list(active):
            return_code = job["process"].poll()
            if return_code is None:
                continue
            job["log_handle"].close()
            active.remove(job)
            if return_code != 0:
                for survivor in active:
                    survivor["process"].terminate()
                    survivor["log_handle"].close()
                raise RuntimeError(
                    f"worker failed rc={return_code}: {job['origin']}/"
                    f"replica_{job['replica']} log={job['log_path']}"
                )
            manifest_path = job["attempt"] / "stream_unit_manifest.json"
            manifest = read_json(manifest_path)
            if manifest.get("passed") is not True:
                raise RuntimeError(f"worker returned without passed manifest: {manifest_path}")
            atomic_json(
                job["unit_dir"] / "unit_selection.json",
                {
                    "schema_version": "mdp04_unit_selection_v1",
                    "selected_attempt": job["attempt"].name,
                    "manifest_sha256": sha256_file(manifest_path),
                    "passed": True,
                },
            )
            print(
                f"passed {job['origin']}/replica_{job['replica']}", flush=True
            )


def preflight(source_run: Path, contract_path: Path) -> dict[str, Any]:
    contract = read_json(contract_path)
    checks: dict[str, bool] = {}
    source_diagnostics: dict[str, Any] = {}
    pinned_diagnostics: dict[str, Any] = {}
    for relative, expected in contract["accepted_source_artifact_sha256"].items():
        path = source_run / relative
        observed = sha256_file(path) if path.is_file() else None
        exact_hash_match = observed == expected
        checks[f"accepted:{relative}"] = exact_hash_match
        source_diagnostics[relative] = {
            "exists": path.is_file(),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "exact_hash_match": exact_hash_match,
        }
    for name, spec in contract["pinned_runtime_sources"].items():
        path = REPO / spec["relative_path"]
        observed = sha256_file(path) if path.is_file() else None
        exact_hash_match = observed == spec["sha256"]
        exact_size_match = path.is_file() and path.stat().st_size == int(spec["bytes"])
        checks[f"pinned_runtime:{name}"] = exact_hash_match and exact_size_match
        pinned_diagnostics[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "expected_sha256": spec["sha256"],
            "observed_sha256": observed,
            "exact_hash_match": exact_hash_match,
            "exact_size_match": exact_size_match,
        }
    for relative, expected in contract["accepted_worker_source_sha256"].items():
        path = REPO / relative
        checks[f"worker:{relative}"] = path.is_file() and sha256_file(path) == expected
    repair = REPO / "scripts" / "37_mech09_downproj_refresh_mediation" / "refresh_mediation_repair_contract.json"
    checks["repair_contract"] = repair.is_file() and sha256_file(repair) == contract[
        "source_repair_contract_sha256"
    ]
    repair_payload = read_json(repair) if checks["repair_contract"] else {}
    checks["pinned_triton_matches_repair_contract"] = (
        contract["pinned_runtime_sources"]["triton_kernels"]["sha256"]
        == repair_payload.get("source_constraints", {}).get("triton_sha256")
    )
    checks["formal_manifest_passed"] = read_json(
        source_run / "formal" / "formal_manifest.json"
    ).get("passed") is True
    analysis_manifest = read_json(
        source_run / "analysis" / "mech09r_analysis_manifest.json"
    )
    checks["analysis_manifest_full_support"] = (
        analysis_manifest.get("passed") is True
        and analysis_manifest.get("hypothesis_classification") == "full_support"
    )
    if not all(checks.values()):
        raise RuntimeError(
            "MDP-04 preflight failed: "
            f"checks={checks}, source_diagnostics={source_diagnostics}, "
            f"pinned_diagnostics={pinned_diagnostics}"
        )
    return {
        "checks": checks,
        "source_artifact_diagnostics": source_diagnostics,
        "pinned_runtime_diagnostics": pinned_diagnostics,
        "local_posthoc_audit_reference": contract["local_posthoc_audit_reference"],
        "passed": True,
    }


def runtime_preflight(
    child_python: Path, gpus: list[str], contract: dict[str, Any]
) -> dict[str, Any]:
    runtime = contract["runtime_contract"]
    program = r'''
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
            "major": properties.major,
            "minor": properties.minor,
            "total_memory": properties.total_memory,
        }
        row["passed"] = (
            expected["gpu_name_contains"] in properties.name
            and [properties.major, properties.minor] == expected["compute_capability"]
            and properties.total_memory >= int(expected["minimum_gpu_memory_bytes"])
        )
        devices.append(row)
checks["devices"] = len(devices) == int(expected["gpu_count"]) and all(
    row["passed"] for row in devices
)
payload = {"checks": checks, "devices": devices, "passed": all(checks.values())}
payload["environment"] = {
    "sys_executable": sys.executable,
    "sys_prefix": sys.prefix,
    "sys_base_prefix": sys.base_prefix,
}
print(json.dumps(payload, sort_keys=True))
if not payload["passed"]:
    raise SystemExit(2)
'''
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    requested_python = executable_path(child_python)
    try:
        completed = subprocess.run(
            [
                requested_python,
                "-c",
                program,
                json.dumps(runtime, sort_keys=True),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except subprocess.CalledProcessError as error:
        stdout = (error.stdout or "").strip()
        stderr = (error.stderr or "").strip()
        raise RuntimeError(
            "MDP-04 runtime contract failed.\n"
            f"requested_python={requested_python}\n"
            f"stdout={stdout or '<empty>'}\n"
            f"stderr={stderr or '<empty>'}"
        ) from error
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    payload["requested_python"] = requested_python
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--child-python", required=True, type=Path)
    parser.add_argument("--contract", default=HERE / "refresh_stream_contract.json", type=Path)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-runtime-preflight",
        action="store_true",
        help="Local CPU framework tests only; forbidden for a formal run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    source_run = args.source_run.resolve()
    contract_path = args.contract.resolve()
    if args.max_parallel < 1 or args.max_parallel > len(args.gpus):
        raise ValueError("max-parallel must be between 1 and the GPU count")
    if len(args.gpus) != 2:
        raise ValueError("the sealed lane mapping requires exactly two GPU ids")
    if args.skip_runtime_preflight and not args.dry_run:
        raise ValueError("--skip-runtime-preflight is forbidden for a formal run")
    if not args.child_python.is_file():
        raise FileNotFoundError(args.child_python)
    if run_dir.exists() and not args.resume:
        raise RuntimeError(f"run directory exists; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_payload = read_json(contract_path)
    runtime_path = run_dir / "runtime_preflight.json"
    runtime_payload = (
        {
            "skipped": True,
            "reason": "local CPU-only controller regression",
            "formal_eligible": False,
        }
        if args.skip_runtime_preflight
        else runtime_preflight(args.child_python, args.gpus, contract_payload)
    )
    if runtime_path.exists():
        if read_json(runtime_path) != runtime_payload:
            raise RuntimeError("resume runtime preflight does not match the sealed run")
    else:
        atomic_json(runtime_path, runtime_payload)
    preflight_payload = preflight(source_run, contract_path)
    preflight_path = run_dir / "preflight.json"
    if preflight_path.exists():
        if read_json(preflight_path) != preflight_payload:
            raise RuntimeError("resume preflight does not match the sealed run")
    else:
        atomic_json(preflight_path, preflight_payload)
    snapshot, snapshot_manifest = snapshot_sources(run_dir, contract_path)
    sealed_contract = (
        snapshot / "scripts" / "mdp_refresh_streaming" / "refresh_stream_contract.json"
    )
    jobs = build_jobs(
        args=args,
        run_dir=run_dir,
        source_run=source_run,
        snapshot=snapshot,
        snapshot_manifest=snapshot_manifest,
        contract=sealed_contract,
    )
    test_script = (
        snapshot / "scripts" / "mdp_refresh_streaming" / "test_streaming.py"
    )
    test_environment = os.environ.copy()
    test_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [executable_path(args.child_python), str(test_script)],
        check=True,
        cwd=str(run_dir),
        env=test_environment,
    )
    test_payload = {"tests": 10, "passed": True, "script": str(test_script)}
    snapshot_suffix = snapshot.name.removeprefix("source_snapshot").strip("_")
    test_record = run_dir / (
        f"cpu_regression_tests_{snapshot_suffix}.json"
        if snapshot_suffix
        else "cpu_regression_tests.json"
    )
    if test_record.exists():
        if read_json(test_record) != test_payload:
            raise RuntimeError("resume CPU-test record does not match the sealed run")
    else:
        atomic_json(test_record, test_payload)
    commands_payload = "".join(
            json.dumps(
                {
                    "origin": job["origin"],
                    "data_replica": job["replica"],
                    "gpu": job["gpu"],
                    "attempt": str(job["attempt"]),
                    "command": job["command"],
                },
                sort_keys=True,
            )
            + "\n"
            for job in jobs
        )
    commands_path = run_dir / "commands.jsonl"
    identity_payload = {
            "schema_version": "mdp04_stream_run_identity_v1",
            "controller_version": SCRIPT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_run": str(source_run),
            "stream_contract_sha256": sha256_file(sealed_contract),
            "source_snapshot_manifest_sha256": sha256_file(snapshot_manifest),
            "gpus": args.gpus,
            "max_parallel": args.max_parallel,
            "resume": args.resume,
            "dry_run": args.dry_run,
            "runtime_preflight_skipped": args.skip_runtime_preflight,
        }
    if commands_path.exists():
        resume_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S+0000")
        (run_dir / f"resume_commands_{resume_stamp}.jsonl").write_text(
            commands_payload, encoding="utf-8"
        )
        atomic_json(
            run_dir / f"resume_identity_{resume_stamp}.json", identity_payload
        )
    else:
        commands_path.write_text(commands_payload, encoding="utf-8")
        atomic_json(run_dir / "run_identity.json", identity_payload)
    if args.dry_run:
        atomic_json(
            run_dir / "status.json",
            {"status": "dry_run_passed", "scheduled_jobs": len(jobs)},
        )
        print(f"MDP-04 dry run passed; scheduled jobs: {len(jobs)}")
        return
    canary = [
        job
        for job in jobs
        if job["origin"] == "early_muon" and int(job["replica"]) == 0
    ]
    remaining = [job for job in jobs if job not in canary]
    if canary:
        print("running the registered formal canary unit first", flush=True)
        run_jobs(canary, 1)
    if remaining:
        print("canary passed; scheduling remaining formal units", flush=True)
        run_jobs(remaining, args.max_parallel)
    validator = (
        snapshot / "scripts" / "mdp_refresh_streaming" / "validate_stream_replay.py"
    )
    command = [
        sys.executable,
        str(validator),
        "--run-dir",
        str(run_dir),
        "--source-run",
        str(source_run),
        "--contract",
        str(sealed_contract),
    ]
    completed = subprocess.run(command, check=False, cwd=str(run_dir))
    manifest_path = run_dir / "analysis" / "formal_stream_manifest.json"
    if not manifest_path.is_file():
        print(
            "MDP-04 final validator stopped without producing a formal manifest; "
            f"validator_rc={completed.returncode}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    manifest = read_json(manifest_path)
    if completed.returncode != 0 or manifest.get("passed") is not True:
        failed_checks = sorted(
            name for name, passed in manifest.get("checks", {}).items() if not passed
        )
        failed_numeric = sorted(
            name
            for name, passed in manifest.get("numeric_gate_checks", {}).items()
            if not passed
        )
        print("MDP-04 computation completed, but the frozen formal gate did not pass.")
        print(f"Artifacts: {run_dir}")
        print(f"Manifest: {manifest_path}")
        print(f"Failed checks: {failed_checks}")
        print(f"Failed numeric gates: {failed_numeric}")
        print(
            "This is a scientific adjudication, not an invitation to relax a "
            "threshold or drop a layer."
        )
        raise SystemExit(2)
    print("MDP-04 refresh streaming replay completed.")
    print(f"Artifacts: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Accepted rows: {manifest['rows']['layer_event']}")


if __name__ == "__main__":
    main()
