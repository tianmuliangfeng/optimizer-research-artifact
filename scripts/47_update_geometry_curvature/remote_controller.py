#!/usr/bin/env python3
"""Seal, run, resume and verify the GEO-01 H100 engineering pilot."""

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
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import protocol as P


SCRIPT_VERSION = "2026-08-04.3"
SOURCE_FILES = (
    "commands/47_update_geometry_curvature/20260804_ex47_update_geometry_curvature.sh",
    "scripts/47_update_geometry_curvature/README.md",
    "scripts/47_update_geometry_curvature/geo01_contract.json",
    "scripts/47_update_geometry_curvature/protocol.py",
    "scripts/47_update_geometry_curvature/geometry_core.py",
    "scripts/47_update_geometry_curvature/geo01_worker.py",
    "scripts/47_update_geometry_curvature/analyze_geo01.py",
    "scripts/47_update_geometry_curvature/remote_controller.py",
    "scripts/47_update_geometry_curvature/run_geo01.py",
    "scripts/47_update_geometry_curvature/test_geo01.py",
    "scripts/mdp_refresh_streaming/stream_metrics.py",
    "scripts/mdp_refresh_streaming/pinned_ex37_runtime/triton_kernels.py",
    "scripts/46_mdp05_confirmatory_update_shock/smoke_worker.py",
    "scripts/37_mech09_downproj_refresh_mediation/mech09r_worker.py",
    "scripts/37_mech09_downproj_refresh_mediation/mech09_worker.py",
    "scripts/37_mech09_downproj_refresh_mediation/refresh_mediation_repair_contract.json",
    "scripts/37_mech09_downproj_refresh_mediation/mech08_control_reference.json",
    "scripts/36_mech08_short_horizon_rollout/mech08_worker.py",
    "scripts/27_mech01_unified_k_diagnostics/mech01_worker.py",
)


def executable(path: Path) -> str:
    return os.path.abspath(os.fspath(path))


def absolute_without_resolving(path: Path) -> Path:
    """Return an absolute spelling while preserving virtualenv symlinks.

    Resolving ``venv/bin/python`` to the base interpreter changes Python's
    virtualenv discovery semantics and silently imports the system packages.
    """

    return Path(os.path.abspath(os.fspath(path.expanduser())))


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
            raise TypeError(f"command row {number} is not an object")
        rows.append(value)
    return rows


def accepted_worker_args(row: dict[str, Any]) -> list[str]:
    values = [str(value) for value in row["command"]]
    if len(values) < 3:
        raise RuntimeError(f"accepted source command is too short: {row['label']}")
    return values[2:]


def option_value(arguments: list[str], option: str) -> str:
    if option not in arguments:
        raise RuntimeError(f"accepted worker option is missing: {option}")
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise RuntimeError(f"accepted worker option has no value: {option}")
    return arguments[index + 1]


def replace_option(arguments: list[str], option: str, value: str) -> None:
    if option not in arguments:
        raise RuntimeError(f"accepted worker option is missing: {option}")
    arguments[arguments.index(option) + 1] = value


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
    required = {
        "early_muon",
        "early_newton_full",
        "late_muon",
        "late_newton_full",
    }
    if set(templates) != required:
        raise RuntimeError(f"accepted source templates incomplete: {templates.keys()}")
    return templates


def live_source_hashes() -> dict[str, str]:
    hashes = {}
    for relative in SOURCE_FILES:
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(f"required GEO-01 source is missing: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def snapshot_sources(run_dir: Path) -> tuple[Path, Path]:
    snapshot = run_dir / "source_snapshot"
    manifest_path = snapshot / "source_snapshot_manifest.json"
    live = live_source_hashes()
    if snapshot.exists():
        manifest = P.read_json(manifest_path)
        checks = {
            relative: (snapshot / relative).is_file()
            and sha256_file(snapshot / relative) == expected
            for relative, expected in manifest.get("files", {}).items()
        }
        if manifest.get("files") != live or not all(checks.values()):
            raise RuntimeError("sealed GEO-01 source snapshot differs from live sources")
        return snapshot, manifest_path
    snapshot.mkdir(parents=True, exist_ok=False)
    for relative in SOURCE_FILES:
        source = REPO / relative
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {
        "schema_version": "geo01_remote_source_snapshot_v1",
        "controller_version": SCRIPT_VERSION,
        "files": {
            relative: sha256_file(snapshot / relative) for relative in SOURCE_FILES
        },
        "passed": True,
    }
    P.atomic_json(manifest_path, manifest)
    return snapshot, manifest_path


def source_preflight(
    source_run: Path,
    templates: dict[str, list[str]],
    contract_path: Path,
    base_contract_path: Path,
) -> dict[str, Any]:
    contract = P.read_json(contract_path)
    source_manifest = P.read_json(source_run / "formal" / "formal_manifest.json")
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
        "contract": all(P.validate_contract(contract).values()),
        "source_contract_hash": sha256_file(base_contract_path)
        == contract["source_lineage"].get(
            "public_execution_contract_sha256",
            contract["source_lineage"]["accepted_execution_contract_sha256"],
        ),
        "source_manifest": source_manifest.get("passed") is True
        and int(source_manifest.get("completed_jobs", -1)) == 12,
        "source_commands": (source_run / "commands.jsonl").is_file(),
        "source_paths": all(path.is_file() for path in source_paths),
        "source_outcomes_unused": contract["source_lineage"][
            "mdp05_outcomes_used_to_select_formula"
        ]
        is False,
    }
    return {"checks": checks, "passed": all(checks.values())}


def runtime_preflight(
    child_python: Path, gpus: list[str], expected: dict[str, Any]
) -> dict[str, Any]:
    script = r'''
import json
import sys
import numpy
import torch
import triton
expected = json.loads(sys.argv[1])
observed = {
    "python": str(sys.version.split()[0]),
    "executable": str(sys.executable),
    "prefix": str(sys.prefix),
    "base_prefix": str(sys.base_prefix),
    "torch": str(torch.__version__),
    "torch_cuda": str(torch.version.cuda),
    "triton": str(triton.__version__),
    "numpy": str(numpy.__version__),
}
checks = {
    key: observed[key] == str(expected[key])
    for key in ("python", "torch", "torch_cuda", "triton", "numpy")
}
checks.update({
    "virtualenv_active": observed["prefix"] != observed["base_prefix"],
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count() == int(expected["gpu_count"]),
})
devices = []
if checks["cuda_available"]:
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        row = {
            "index": index,
            "name": prop.name,
            "compute_capability": [prop.major, prop.minor],
            "total_memory": prop.total_memory,
        }
        row["passed"] = (
            expected["gpu_name_contains"] in prop.name
            and row["compute_capability"] == expected["compute_capability"]
            and prop.total_memory >= int(expected["minimum_gpu_memory_bytes"])
        )
        devices.append(row)
checks["devices"] = len(devices) == int(expected["gpu_count"]) and all(
    row["passed"] for row in devices
)
payload = {
    "checks": checks,
    "observed": observed,
    "expected": expected,
    "devices": devices,
    "passed": all(checks.values()),
}
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] else 2)
'''
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    completed = subprocess.run(
        [executable(child_python), "-c", script, json.dumps(expected, sort_keys=True)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1]) if lines else {
        "passed": False,
        "checks": {},
        "stderr": completed.stderr,
    }
    observed_executable = payload.get("observed", {}).get("executable")
    requested_executable = (
        os.path.normcase(os.path.abspath(str(observed_executable)))
        == os.path.normcase(os.path.abspath(os.fspath(child_python)))
        if observed_executable
        else False
    )
    payload.setdefault("checks", {})["requested_executable"] = requested_executable
    payload["requested_python"] = str(child_python)
    payload["passed"] = (
        payload.get("passed") is True
        and requested_executable
        and completed.returncode == 0
    )
    payload["return_code"] = completed.returncode
    return payload


def next_attempt(unit_dir: Path) -> Path:
    indices = [
        int(path.name.split("_")[-1])
        for path in unit_dir.glob("attempt_*")
        if path.is_dir() and path.name.split("_")[-1].isdigit()
    ]
    attempt = unit_dir / f"attempt_{max(indices, default=0) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def selected_attempt(
    unit_dir: Path,
    manifest_name: str,
    expected: dict[str, Any] | None = None,
) -> Path | None:
    selection_path = unit_dir / "unit_selection.json"
    if not selection_path.is_file():
        return None
    selection = P.read_json(selection_path)
    attempt = unit_dir / str(selection.get("selected_attempt", ""))
    manifest_path = attempt / manifest_name
    if selection.get("passed") is not True or not manifest_path.is_file():
        return None
    manifest = P.read_json(manifest_path)
    if manifest.get("passed") is not True:
        return None
    if expected is not None and any(
        manifest.get(field) != value for field, value in expected.items()
    ):
        raise RuntimeError(
            f"selected attempt metadata mismatch: {unit_dir / selection['selected_attempt']}"
        )
    artifact_hashes = manifest.get("artifact_sha256", {})
    if artifact_hashes and any(
        not (attempt / name).is_file()
        or sha256_file(attempt / name) != expected_hash
        for name, expected_hash in artifact_hashes.items()
    ):
        raise RuntimeError(f"selected attempt artifact hash mismatch: {attempt}")
    return attempt


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
            str(snapshot / "scripts/mdp_refresh_streaming/pinned_ex37_runtime/triton_kernels.py"),
        ),
        (
            "--mech08-control-reference",
            str(snapshot / "scripts/37_mech09_downproj_refresh_mediation/mech08_control_reference.json"),
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


def run_job(
    *,
    label: str,
    command: list[str],
    attempt: Path,
    gpu: str,
    manifest_name: str,
    status_name: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    P.atomic_json(
        attempt / "launch.json",
        {
            "schema_version": "geo01_attempt_launch_v1",
            "label": label,
            "gpu": str(gpu),
            "command": command,
            "environment": {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        },
    )
    log_path = attempt / "worker.log"
    print(f"launch gpu={gpu} {label}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            check=False,
            cwd=str(attempt),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    log_seal = {
        "return_code": int(completed.returncode),
        "bytes": log_path.stat().st_size,
        "sha256": sha256_file(log_path),
    }
    P.atomic_json(attempt / "worker_log_seal.json", log_seal)
    manifest_path = attempt / manifest_name
    status_path = attempt / status_name
    manifest = P.read_json(manifest_path) if manifest_path.is_file() else {}
    status = P.read_json(status_path) if status_path.is_file() else {}
    checks = {
        "return_code": completed.returncode == 0,
        "manifest": manifest.get("passed") is True,
        "status": status.get("status") == "passed",
        **{
            key: manifest.get(field) == value
            for key, (field, value) in expected.items()
        },
    }
    return {
        "label": label,
        "attempt": str(attempt),
        "manifest": manifest_name,
        "checks": checks,
        "passed": all(checks.values()),
    }


def select(unit_dir: Path, attempt: Path, result: dict[str, Any]) -> None:
    selection = {
        "schema_version": "geo01_unit_selection_v1",
        "selected_attempt": attempt.name,
        "result": result,
        "passed": result["passed"],
    }
    if result["passed"]:
        P.atomic_json(unit_dir / "unit_selection.json", selection)


def create_handoff(
    run_dir: Path, smoke_attempt: Path, pilot_attempt: Path
) -> dict[str, Any]:
    paths = [
        run_dir / "run_identity.json",
        run_dir / "preflight.json",
        run_dir / "runtime_preflight.json",
        run_dir / "source_snapshot" / "source_snapshot_manifest.json",
        run_dir / "sealed" / "derived_execution_contract.json",
        run_dir / "sealed" / "offset_collision_certificate.json",
        run_dir / "sealed" / "pilot_plan.json",
        smoke_attempt / "mech09r_manifest.json",
        smoke_attempt / "worker.log",
        smoke_attempt / "worker_log_seal.json",
        pilot_attempt / "geo01_unit_manifest.json",
        pilot_attempt / "mech09r_manifest.json",
        pilot_attempt / "direction_construction_audit.json",
        pilot_attempt / "geometry_event_audit.json",
        pilot_attempt / "geo01_geometry_rows.csv",
        pilot_attempt / "geo01_geometry_rows.jsonl",
        pilot_attempt / "worker.log",
        pilot_attempt / "worker_log_seal.json",
        run_dir / "analysis" / "analysis_manifest.json",
        run_dir / "analysis" / "geometry_rows.csv",
    ]
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"handoff artifact is missing: {path}")
        rows.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema_version": "geo01_handoff_manifest_v1",
        "files": rows,
        "no_large_files": all(int(row["bytes"]) < 250_000_000 for row in rows),
        "claim_eligible": False,
    }
    payload["passed"] = payload["no_large_files"]
    P.atomic_json(run_dir / "handoff_manifest.json", payload)
    return payload


def audit_handoff(run_dir: Path, handoff: dict[str, Any]) -> dict[str, bool]:
    rows = handoff.get("files", [])
    paths = [str(row.get("path", "")) for row in rows]
    file_checks = []
    for row in rows:
        path = run_dir / str(row.get("path", ""))
        file_checks.append(
            path.is_file()
            and path.stat().st_size == int(row.get("bytes", -1))
            and sha256_file(path) == row.get("sha256")
        )
    return {
        "schema": handoff.get("schema_version") == "geo01_handoff_manifest_v1",
        "manifest_passed": handoff.get("passed") is True,
        "claim_boundary": handoff.get("claim_eligible") is False,
        "nonempty_unique_paths": bool(paths) and len(paths) == len(set(paths)),
        "files": all(file_checks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "pilot", "resume", "verify"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--child-python", type=Path)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    return parser.parse_args()


def controller(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    contract_live = HERE / "geo01_contract.json"
    contract_sha = sha256_file(contract_live)
    if args.mode == "verify":
        identity = P.read_json(run_dir / "run_identity.json")
        status = P.read_json(run_dir / "status.json")
        handoff = P.read_json(run_dir / "handoff_manifest.json")
        handoff_checks = audit_handoff(run_dir, handoff)
        checks = {
            "identity": identity.get("experiment") == "GEO-01",
            "contract": identity.get("contract_sha256") == contract_sha,
            "completed": status.get("status") == "completed",
            "handoff": all(handoff_checks.values()),
            "not_claim_eligible": status.get("claim_eligible") is False,
        }
        print(json.dumps({"checks": checks, "passed": all(checks.values())}, indent=2))
        return 0 if all(checks.values()) else 2
    if args.source_run is None or args.child_python is None:
        raise ValueError(f"{args.mode} requires --source-run and --child-python")
    source_run = args.source_run.resolve()
    # Do not call Path.resolve(): the frozen venv executable is a symlink to
    # the system interpreter, and resolving it disables virtualenv discovery.
    child_python = absolute_without_resolving(args.child_python)
    resume = args.mode == "resume"
    if resume:
        identity = P.read_json(run_dir / "run_identity.json")
        identity_checks = {
            "experiment": identity.get("experiment") == "GEO-01",
            "contract": identity.get("contract_sha256") == contract_sha,
            "not_dry_run": identity.get("dry_run") is False,
        }
        if not all(identity_checks.values()):
            raise RuntimeError(f"resume identity mismatch: {identity_checks}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        P.atomic_json(
            run_dir / "run_identity.json",
            {
                "schema_version": "geo01_remote_run_identity_v1",
                "experiment": "GEO-01",
                "experiment_number": 47,
                "phase": "pilot",
                "contract_sha256": contract_sha,
                "controller_version": SCRIPT_VERSION,
                "dry_run": args.mode == "dry-run",
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    P.atomic_json(run_dir / "status.json", {"status": "preflight"})
    snapshot, snapshot_manifest = snapshot_sources(run_dir)
    contract_path = snapshot / "scripts/47_update_geometry_curvature/geo01_contract.json"
    base_contract = snapshot / "scripts/37_mech09_downproj_refresh_mediation/refresh_mediation_repair_contract.json"
    contract = P.read_json(contract_path)
    templates = source_templates(source_run)
    preflight = source_preflight(source_run, templates, contract_path, base_contract)
    preflight["checks"].update(
        {
            "child_python": child_python.is_file(),
            "two_unique_gpus": len(args.gpus) == 2 and len(set(args.gpus)) == 2,
        }
    )
    preflight["passed"] = all(preflight["checks"].values())
    P.atomic_json(run_dir / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError(f"GEO-01 source preflight failed: {preflight['checks']}")
    sealed = run_dir / "sealed"
    sealed.mkdir(exist_ok=resume)
    execution_path = sealed / "derived_execution_contract.json"
    derived = P.derive_execution_contract(
        P.read_json(base_contract), contract, contract_sha
    )
    if execution_path.is_file() and P.read_json(execution_path) != derived:
        raise RuntimeError("sealed GEO-01 execution contract changed")
    P.atomic_json(execution_path, derived)
    execution_sha = sha256_file(execution_path)
    offsets = P.build_offset_certificate(contract)
    P.atomic_json(sealed / "offset_collision_certificate.json", offsets)
    if not offsets["passed"]:
        raise RuntimeError(f"GEO-01 offset collision: {offsets['checks']}")
    plan = {
        "schema_version": "geo01_remote_pilot_plan_v1",
        "contract_sha256": contract_sha,
        "execution_contract_sha256": execution_sha,
        "source_snapshot_manifest_sha256": sha256_file(snapshot_manifest),
        "smoke": contract["remote_smoke"],
        "pilot": contract["pilot"],
        "formal_units": 1,
        "scientific_outcome_selection_authorized": False,
        "passed": True,
    }
    plan_path = sealed / "pilot_plan.json"
    if plan_path.is_file() and P.read_json(plan_path) != plan:
        raise RuntimeError("sealed GEO-01 pilot plan changed")
    P.atomic_json(plan_path, plan)
    if args.mode == "dry-run":
        P.atomic_json(
            run_dir / "status.json",
            {"status": "dry_run_passed", "claim_eligible": False},
        )
        print(f"GEO-01 remote dry-run passed: {run_dir}")
        return 0
    runtime = runtime_preflight(child_python, args.gpus, contract["runtime_contract"])
    P.atomic_json(run_dir / "runtime_preflight.json", runtime)
    if not runtime.get("passed"):
        raise RuntimeError(f"GEO-01 runtime preflight failed: {runtime}")

    smoke = contract["remote_smoke"]
    smoke_dir = run_dir / "smoke" / smoke["origin"] / f"replica_{smoke['data_replica']}"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_attempt = selected_attempt(
        smoke_dir,
        "mech09r_manifest.json",
        {
            "checkpoint_cell": smoke["origin"],
            "data_replica": int(smoke["data_replica"]),
            "contract_sha256": execution_sha,
        },
    )
    if smoke_attempt is None:
        smoke_attempt = next_attempt(smoke_dir)
        smoke_args = base_arguments(
            templates[smoke["origin"]],
            output=smoke_attempt,
            tier="smoke",
            origin=smoke["origin"],
            replica=int(smoke["data_replica"]),
            execution_contract=execution_path,
            snapshot=snapshot,
            smoke_manifest=None,
        )
        smoke_worker = snapshot / "scripts/46_mdp05_confirmatory_update_shock/smoke_worker.py"
        smoke_result = run_job(
            label=f"smoke/{smoke['origin']}/replica_{smoke['data_replica']}",
            command=[executable(child_python), str(smoke_worker), *smoke_args],
            attempt=smoke_attempt,
            gpu=args.gpus[0],
            manifest_name="mech09r_manifest.json",
            status_name="status.json",
            expected={
                "origin": ("checkpoint_cell", smoke["origin"]),
                "replica": ("data_replica", int(smoke["data_replica"])),
                "contract": ("contract_sha256", execution_sha),
            },
        )
        select(smoke_dir, smoke_attempt, smoke_result)
        if not smoke_result["passed"]:
            raise RuntimeError(f"GEO-01 smoke failed; see {smoke_attempt / 'worker.log'}")
    smoke_manifest = smoke_attempt / "mech09r_manifest.json"

    pilot = contract["pilot"]
    origin = pilot["origins"][0]
    replica = int(pilot["data_replicas"][0])
    pilot_dir = run_dir / "pilot" / origin / f"replica_{replica}"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    pilot_attempt = selected_attempt(
        pilot_dir,
        "geo01_unit_manifest.json",
        {
            "origin": origin,
            "data_replica": replica,
            "contract_sha256": contract_sha,
            "execution_contract_sha256": execution_sha,
            "claim_eligible": False,
            "full_direction_persisted": False,
        },
    )
    if pilot_attempt is None:
        pilot_attempt = next_attempt(pilot_dir)
        pilot_args = base_arguments(
            templates[origin],
            output=pilot_attempt,
            tier="formal",
            origin=origin,
            replica=replica,
            execution_contract=execution_path,
            snapshot=snapshot,
            smoke_manifest=smoke_manifest,
        )
        geo_worker = snapshot / "scripts/47_update_geometry_curvature/geo01_worker.py"
        command = [
            executable(child_python),
            str(geo_worker),
            "--geo01-output-dir",
            str(pilot_attempt),
            "--geo01-contract",
            str(contract_path),
            "--source-snapshot-manifest",
            str(snapshot_manifest),
            "--",
            *pilot_args,
        ]
        pilot_result = run_job(
            label=f"pilot/{origin}/replica_{replica}",
            command=command,
            attempt=pilot_attempt,
            gpu=args.gpus[0],
            manifest_name="geo01_unit_manifest.json",
            status_name="geo01_status.json",
            expected={
                "origin": ("origin", origin),
                "replica": ("data_replica", replica),
                "contract": ("contract_sha256", contract_sha),
                "execution": ("execution_contract_sha256", execution_sha),
                "claim": ("claim_eligible", False),
                "direction": ("full_direction_persisted", False),
            },
        )
        select(pilot_dir, pilot_attempt, pilot_result)
        if not pilot_result["passed"]:
            raise RuntimeError(f"GEO-01 pilot failed; see {pilot_attempt / 'worker.log'}")

    analysis_dir = run_dir / "analysis"
    analyzer = snapshot / "scripts/47_update_geometry_curvature/analyze_geo01.py"
    completed = subprocess.run(
        [
            executable(Path(sys.executable)),
            str(analyzer),
            "--input-jsonl",
            str(pilot_attempt / "geo01_geometry_rows.jsonl"),
            "--output-dir",
            str(analysis_dir),
            "--contract",
            str(contract_path),
            "--phase",
            "pilot",
        ],
        check=False,
        cwd=str(run_dir),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"GEO-01 pilot analyzer failed rc={completed.returncode}")
    analysis = P.read_json(analysis_dir / "analysis_manifest.json")
    if analysis.get("integrity_passed") is not True or analysis.get("claim_eligible") is not False:
        raise RuntimeError("GEO-01 pilot analysis boundary failed")
    handoff = create_handoff(run_dir, smoke_attempt, pilot_attempt)
    if not handoff["passed"]:
        raise RuntimeError("GEO-01 handoff failed")
    P.atomic_json(
        run_dir / "status.json",
        {
            "status": "completed",
            "phase": "pilot",
            "engineering_integrity_passed": True,
            "scientific_result": "engineering_pilot_only_no_scientific_claim",
            "claim_eligible": False,
            "discovery_authorized": False,
            "llama_10b_triggered": False,
        },
    )
    print("GEO-01 H100 pilot completed.")
    print(f"Artifacts: {run_dir}")
    print(f"Analysis: {analysis_dir / 'analysis_manifest.json'}")
    print("Scientific result: engineering_pilot_only_no_scientific_claim")
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
                    "resume_allowed_same_contract": args.mode != "dry-run",
                },
            )
        print(
            f"GEO-01 stopped cleanly: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
