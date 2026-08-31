#!/usr/bin/env python3
"""Seal, preflight, pilot, run, resume, and verify experiment 56."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import protocol as P


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGE_REL = Path("scripts/56_llama1b_10b_global_diag")
CONTRACT_NAME = "formal_contract.json"
SNAPSHOT_FILES = (
    f"{PACKAGE_REL.as_posix()}/formal_contract.json",
    f"{PACKAGE_REL.as_posix()}/protocol.py",
    f"{PACKAGE_REL.as_posix()}/train_segment.py",
    f"{PACKAGE_REL.as_posix()}/run_formal.py",
    f"{PACKAGE_REL.as_posix()}/analyze_formal.py",
    f"{PACKAGE_REL.as_posix()}/upload_wandb.py",
    f"{PACKAGE_REL.as_posix()}/README.md",
    f"{PACKAGE_REL.as_posix()}/llama_global_diag_source_builder.py",
    f"{PACKAGE_REL.as_posix()}/frozen_ex48_controls.csv",
)


ACTIVE: dict[int, subprocess.Popen[str]] = {}
ACTIVE_LOCK = threading.Lock()
STOP_EVENT = threading.Event()
REQUIRED_PHYSICAL_GPUS = [3]
STEP_SECONDS = {
    "global_diag": 7.55,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("check", "preflight", "pilot", "formal", "resume", "verify", "upload"),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--live-repo", type=Path, default=REPO)
    parser.add_argument("--official-repo", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--training-python", type=Path)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(REQUIRED_PHYSICAL_GPUS))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-init-timeout", type=int, default=120)
    args = parser.parse_args()
    if args.mode != "check" and args.run_dir is None:
        parser.error(f"{args.mode} requires --run-dir")
    if args.mode in ("preflight", "pilot", "formal", "resume"):
        for name in ("official_repo", "data_dir", "training_python"):
            if getattr(args, name) is None:
                parser.error(f"{args.mode} requires --{name.replace('_', '-')}")
    if args.gpus != REQUIRED_PHYSICAL_GPUS:
        parser.error("experiment 56 is frozen to physical GPU 3")
    return args


def live_contract(repo: Path) -> tuple[Path, dict[str, Any]]:
    path = repo.resolve() / PACKAGE_REL / CONTRACT_NAME
    contract = P.read_json(path)
    P.assert_contract(contract)
    return path, contract


def derived_sources(repo: Path) -> tuple[dict[str, str], dict[str, str]]:
    root = repo.resolve()
    builder_path = root / PACKAGE_REL / "llama_global_diag_source_builder.py"
    spec = importlib.util.spec_from_file_location("ex56_source_builder", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import EX56 source builder: {builder_path}")
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    built = builder.build(root)
    generated = {
        "scripts/17_llama_swiglu_validation/train_llama_swiglu.py": built.trainer,
        "scripts/20_llama_swiglu_1b/train_llama_swiglu_1b.py": built.wrapper1b,
    }
    return generated, dict(builder.PARENT_SOURCE_SHA256)


def check_live_sources(repo: Path, contract: dict[str, Any]) -> dict[str, Any]:
    root = repo.resolve()
    checks: dict[str, bool] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative in SNAPSHOT_FILES:
        path = root / relative
        exists = path.is_file()
        checks[f"exists:{relative}"] = exists
        if exists:
            files[relative] = {"bytes": path.stat().st_size, "sha256": P.sha256_file(path)}
    generated, parent_hashes = derived_sources(root)
    for relative, source in generated.items():
        encoded = source.encode("utf-8")
        row = {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}
        files[relative] = row
        checks[f"derived:{relative}"] = row["sha256"] == contract["accepted_sources"][relative]
    checks["derived_global_diag_route"] = "global_diag" in generated[
        "scripts/17_llama_swiglu_validation/train_llama_swiglu.py"
    ]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "files": files,
        "parent_source_sha256": parent_hashes,
    }


def snapshot_sources(run_dir: Path, repo: Path, contract: dict[str, Any]) -> tuple[Path, Path]:
    snapshot = run_dir / "source_snapshot"
    manifest_path = snapshot / "source_snapshot_manifest.json"
    live = check_live_sources(repo, contract)
    if not live["passed"]:
        raise RuntimeError(f"live EX56 source check failed: {live['checks']}")
    if snapshot.exists():
        manifest = P.read_json(manifest_path)
        checks = {
            relative: (snapshot / relative).is_file()
            and P.sha256_file(snapshot / relative) == item["sha256"]
            for relative, item in manifest.get("files", {}).items()
        }
        if manifest.get("files") != live["files"] or not checks or not all(checks.values()):
            raise RuntimeError("existing EX56 source snapshot differs from live frozen sources")
        return snapshot, manifest_path
    snapshot.mkdir(parents=True, exist_ok=False)
    for relative in SNAPSHOT_FILES:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo.resolve() / relative, target)
    generated, _ = derived_sources(repo)
    for relative, source in generated.items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8", newline="\n")
    payload = {
        "schema_version": "ex56_source_snapshot_v1",
        "created_at": now_iso(),
        "parent_source_sha256": live["parent_source_sha256"],
        "files": live["files"],
    }
    P.atomic_json(manifest_path, payload)
    return snapshot, manifest_path


def worker_env(official_repo: Path, gpu: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(official_repo.resolve()) + (os.pathsep + prior if prior else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def runtime_preflight(
    training_python: Path, official_repo: Path, gpus: list[int], contract: dict[str, Any]
) -> dict[str, Any]:
    expected = dict(contract["runtime"])
    expected["requested_executable"] = str(training_python.absolute())
    expected["physical_gpus"] = list(gpus)
    expected["selected_gpus"] = list(range(len(gpus)))
    code = r'''
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import numpy
import torch
import triton

expected = json.loads(sys.argv[1])
kernel = Path(sys.argv[2]).resolve()
def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

driver_process = subprocess.run(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    text=True,
    capture_output=True,
)
driver_versions = sorted(
    {line.strip() for line in driver_process.stdout.splitlines() if line.strip()}
)
observed = {
    "executable": str(Path(sys.executable).absolute()),
    "hostname": socket.gethostname(),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "triton": triton.__version__,
    "numpy": numpy.__version__,
    "device_count": torch.cuda.device_count(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "nvidia_driver_versions": driver_versions,
    "triton_kernels_sha256": sha256_file(kernel) if kernel.is_file() else None,
}
checks = {
    "requested_executable": observed["executable"] == expected["requested_executable"],
    "python": observed["python"] == expected["python"],
    "torch": observed["torch"] == expected["torch"],
    "torch_cuda": observed["torch_cuda"] == expected["torch_cuda"],
    "triton": observed["triton"] == expected["triton"],
    "numpy": observed["numpy"] == expected["numpy"],
    "nvidia_driver": observed["nvidia_driver_versions"] == [expected["nvidia_driver"]],
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": observed["device_count"] == expected["gpu_count"],
    "selected_gpu_count": len(expected["selected_gpus"]) == expected["gpu_count"],
    "physical_gpu_selection": observed["cuda_visible_devices"]
    == ",".join(str(value) for value in expected["physical_gpus"]),
    "triton_kernels": observed["triton_kernels_sha256"] == expected["accepted_triton_kernels_sha256"],
}
devices = []
for index in expected["selected_gpus"]:
    if 0 <= index < torch.cuda.device_count():
        props = torch.cuda.get_device_properties(index)
        row = {
            "index": index,
            "name": props.name,
            "compute_capability": [props.major, props.minor],
            "total_memory": props.total_memory,
        }
        row["passed"] = (
            expected["gpu_name_contains"] in props.name
            and row["compute_capability"] == expected["compute_capability"]
            and props.total_memory >= expected["minimum_gpu_memory_bytes"]
        )
        devices.append(row)
checks["devices"] = len(devices) == len(expected["selected_gpus"]) and all(row["passed"] for row in devices)
payload = {"passed": all(checks.values()), "checks": checks, "observed": observed, "devices": devices}
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] else 2)
'''
    kernel_path = official_repo.resolve() / "triton_kernels.py"
    completed = subprocess.run(
        [str(training_python.absolute()), "-c", code, json.dumps(expected), str(kernel_path)],
        env={
            **worker_env(official_repo),
            "CUDA_VISIBLE_DEVICES": ",".join(str(value) for value in gpus),
        },
        text=True,
        capture_output=True,
    )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception:
        payload = {
            "passed": False,
            "return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    payload["return_code"] = completed.returncode
    return payload


def run_capture(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, env=env, text=True, capture_output=True)


def init_audit(
    snapshot: Path,
    training_python: Path,
    official_repo: Path,
    data_audit: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    contract_path = snapshot / PACKAGE_REL / CONTRACT_NAME
    worker = snapshot / PACKAGE_REL / "train_segment.py"
    base_trainer = snapshot / "scripts/17_llama_swiglu_validation/train_llama_swiglu.py"
    observed: dict[str, Any] = {}
    method = contract["grid"]["methods"][0]
    for seed in contract["grid"]["seeds"]:
        command = [
            str(training_python.absolute()),
            str(worker),
            "--contract",
            str(contract_path),
            "--base-trainer",
            str(base_trainer),
            "--data-audit",
            str(data_audit),
            "--method",
            method,
            "--seed",
            str(seed),
            "--phase-id",
            "backbone_4400",
            "--output-dir",
            str(data_audit.parent / "init_audit" / f"seed{seed}"),
            "--init-only",
        ]
        completed = run_capture(command, worker_env(official_repo))
        lines = [line for line in completed.stdout.splitlines() if line.startswith("EX56_INIT_AUDIT ")]
        if completed.returncode != 0 or len(lines) != 1:
            raise RuntimeError(
                f"EX56 init audit failed for seed {seed}: rc={completed.returncode}\n{completed.stdout}\n{completed.stderr}"
            )
        observed[str(seed)] = json.loads(lines[0].split(" ", 1)[1])
    expected = contract["accepted_ex48_initialization_sha256"]
    checks = {
        "all_formal_seeds": set(observed) == {str(seed) for seed in contract["grid"]["seeds"]},
        "accepted_ex48_seed_hashes": {
            seed: row["init_sha256"] for seed, row in observed.items()
        } == expected,
        "parameter_count": all(
            int(row["architecture"]["parameter_count"]) == int(contract["profile"]["parameters"])
            for row in observed.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "accepted_ex48_initialization_sha256": expected,
        "seeds": observed,
    }


def load_identity(run_dir: Path) -> dict[str, Any]:
    return P.read_json(run_dir / "run_identity.json")


def validate_identity(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    identity = load_identity(run_dir)
    current = {
        "run_dir": str(run_dir),
        "live_repo": str(args.live_repo.resolve()),
        "official_repo": str(args.official_repo.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "training_python": str(args.training_python.absolute()),
        "gpus": args.gpus,
    }
    checks = {key: identity.get(key) == value for key, value in current.items()}
    if not all(checks.values()):
        raise RuntimeError(f"EX56 same-run resume identity mismatch: {checks}")
    return identity


def preflight(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("EX56 preflight requires a new empty run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    _, contract = live_contract(args.live_repo)
    P.atomic_json(run_dir / "status.json", {"status": "preflight", "updated_at": now_iso()})
    snapshot, snapshot_manifest = snapshot_sources(run_dir, args.live_repo, contract)
    sealed_contract_path = snapshot / PACKAGE_REL / CONTRACT_NAME
    sealed_contract = P.read_json(sealed_contract_path)
    P.assert_contract(sealed_contract)

    runtime = runtime_preflight(args.training_python, args.official_repo, args.gpus, sealed_contract)
    P.atomic_json(run_dir / "runtime_preflight.json", runtime)
    disk = shutil.disk_usage(run_dir)
    disk_check = int(disk.free) >= int(sealed_contract["checkpoint_retention"]["minimum_free_disk_bytes_at_preflight"])
    if not runtime.get("passed") or not disk_check:
        raise RuntimeError(
            f"EX56 runtime/disk preflight failed: runtime={runtime.get('checks')} free_bytes={disk.free} disk_pass={disk_check}"
        )

    print("EX56 hashing the frozen FineWeb inventory; this is a one-time full-content audit.", flush=True)
    data = P.audit_data_dir(args.data_dir.resolve(), sealed_contract, full_hash=True)
    data["checks"]["accepted_ex48_content_projection"] = (
        data["content_projection_sha256"]
        == sealed_contract["data"]["accepted_ex48_content_projection_sha256"]
    )
    data["passed"] = all(data["checks"].values())
    P.atomic_json(run_dir / "data_audit.json", data)
    if not data["passed"]:
        raise RuntimeError(f"EX56 data preflight failed: {data['checks']}")
    initialization = init_audit(
        snapshot,
        args.training_python,
        args.official_repo,
        run_dir / "data_audit.json",
        sealed_contract,
    )
    P.atomic_json(run_dir / "init_audit.json", initialization)
    if not initialization["passed"]:
        raise RuntimeError(f"EX56 initialization preflight failed: {initialization['checks']}")

    identity = {
        "schema_version": "ex56_run_identity_v1",
        "created_at": now_iso(),
        "run_dir": str(run_dir),
        "live_repo": str(args.live_repo.resolve()),
        "official_repo": str(args.official_repo.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "training_python": str(args.training_python.absolute()),
        "gpus": args.gpus,
        "contract_sha256": P.sha256_file(sealed_contract_path),
        "source_snapshot_manifest_sha256": P.sha256_file(snapshot_manifest),
        "data_inventory_sha256": data["inventory_sha256"],
    }
    P.atomic_json(run_dir / "run_identity.json", identity)
    payload = {
        "schema_version": "ex56_preflight_v1",
        "passed": True,
        "created_at": now_iso(),
        "runtime": runtime,
        "disk": {
            "free_bytes": disk.free,
            "required_bytes": sealed_contract["checkpoint_retention"]["minimum_free_disk_bytes_at_preflight"],
            "passed": disk_check,
        },
        "data_audit_sha256": P.sha256_file(run_dir / "data_audit.json"),
        "init_audit_sha256": P.sha256_file(run_dir / "init_audit.json"),
        "identity_sha256": P.sha256_file(run_dir / "run_identity.json"),
    }
    P.atomic_json(run_dir / "preflight_manifest.json", payload)
    P.atomic_json(run_dir / "status.json", {"status": "preflight_passed", "updated_at": now_iso()})
    print(f"EX56 preflight passed. Artifacts: {run_dir}")


def phase_manifest_valid(phase_dir: Path, expected: dict[str, Any], full_checkpoint_hash: bool = False) -> bool:
    manifest_path = phase_dir / "phase_manifest.json"
    summary_path = phase_dir / "summary.json"
    metrics_path = phase_dir / "metrics.csv"
    if not manifest_path.is_file() or not summary_path.is_file() or not metrics_path.is_file():
        return False
    try:
        manifest = P.read_json(manifest_path)
        summary = P.read_json(summary_path)
    except Exception:
        return False
    checks = [
        manifest.get("schema_version") == P.PHASE_MANIFEST_SCHEMA,
        manifest.get("passed") is True,
        manifest.get("method") == expected["method"],
        int(manifest.get("seed", -1)) == int(expected["seed"]),
        manifest.get("phase_id") == expected["phase_id"],
        P.sha256_file(summary_path) == manifest.get("summary_sha256"),
        P.sha256_file(metrics_path) == manifest.get("metrics_sha256"),
        summary.get("status") == "completed",
    ]
    checkpoint = manifest.get("checkpoint", {})
    checkpoint_path = Path(checkpoint.get("path", ""))
    retirement = phase_dir / "checkpoint_retirement.json"
    if checkpoint_path.is_file():
        checks.append(checkpoint_path.stat().st_size == int(checkpoint.get("bytes", -1)))
        if full_checkpoint_hash:
            checks.append(P.sha256_file(checkpoint_path) == checkpoint.get("sha256"))
    elif retirement.is_file() and not bool(checkpoint.get("retained")):
        cert = P.read_json(retirement)
        checks.extend(
            [
                cert.get("passed") is True,
                cert.get("checkpoint_sha256") == checkpoint.get("sha256"),
                int(cert.get("checkpoint_bytes", -1)) == int(checkpoint.get("bytes", -2)),
            ]
        )
    else:
        checks.append(False)
    return all(checks)


def unit_manifest_valid(
    unit_dir: Path, method: str, seed: int, contract: dict[str, Any]
) -> bool:
    path = unit_dir / "unit_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = P.read_json(path)
    except Exception:
        return False
    if (
        manifest.get("schema_version") != P.UNIT_MANIFEST_SCHEMA
        or manifest.get("passed") is not True
        or manifest.get("method") != method
        or int(manifest.get("seed", -1)) != seed
        or manifest.get("completed_phases") != [row["id"] for row in contract["phases"]]
    ):
        return False
    for phase in contract["phases"]:
        phase_dir = unit_dir / phase["id"]
        expected = {"method": method, "seed": seed, "phase_id": phase["id"]}
        if not phase_manifest_valid(phase_dir, expected):
            return False
        frozen = manifest.get("phases", {}).get(phase["id"], {})
        if (
            frozen.get("manifest_sha256") != P.sha256_file(phase_dir / "phase_manifest.json")
            or frozen.get("summary_sha256") != P.sha256_file(phase_dir / "summary.json")
            or frozen.get("metrics_sha256") != P.sha256_file(phase_dir / "metrics.csv")
        ):
            return False
    return True


def worker_command(
    *,
    args: argparse.Namespace,
    snapshot: Path,
    phase_id: str,
    method: str,
    seed: int,
    output_dir: Path,
    source: dict[str, Any] | None,
    max_new_steps: int | None = None,
) -> list[str]:
    command = [
        str(args.training_python.absolute()),
        str(snapshot / PACKAGE_REL / "train_segment.py"),
        "--contract",
        str(snapshot / PACKAGE_REL / CONTRACT_NAME),
        "--base-trainer",
        str(snapshot / "scripts/17_llama_swiglu_validation/train_llama_swiglu.py"),
        "--data-audit",
        str(args.run_dir.resolve() / "data_audit.json"),
        "--method",
        method,
        "--seed",
        str(seed),
        "--phase-id",
        phase_id,
        "--output-dir",
        str(output_dir),
        "--resume",
        "auto",
    ]
    if source is not None:
        command.extend(
            [
                "--source-checkpoint",
                str(source["path"]),
                "--source-checkpoint-sha256",
                str(source["sha256"]),
            ]
        )
    if max_new_steps is not None:
        command.extend(["--max-new-steps", str(max_new_steps)])
    return command


def tee_worker(command: list[str], env: dict[str, str], log_path: Path, gpu: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with ACTIVE_LOCK:
            ACTIVE[gpu] = process
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"[gpu{gpu}] {line}", end="", flush=True)
            return process.wait()
        finally:
            with ACTIVE_LOCK:
                ACTIVE.pop(gpu, None)


def stop_active(except_gpu: int | None = None) -> None:
    STOP_EVENT.set()
    with ACTIVE_LOCK:
        processes = [(gpu, process) for gpu, process in ACTIVE.items() if gpu != except_gpu]
    for _, process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 120
    for _, process in processes:
        remaining = max(1.0, deadline - time.time())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()


def parent_checkpoint(unit_dir: Path, phase: dict[str, Any]) -> dict[str, Any] | None:
    parent = phase.get("parent")
    if parent is None:
        return None
    manifest = P.read_json(unit_dir / str(parent) / "phase_manifest.json")
    checkpoint = manifest["checkpoint"]
    path = Path(checkpoint["path"])
    if not path.is_file():
        raise RuntimeError(f"required fork checkpoint was retired too early: {path}")
    return {"path": path, "sha256": checkpoint["sha256"], "bytes": checkpoint["bytes"]}


def retire_fork(unit_dir: Path, phase_id: str, contract: dict[str, Any]) -> None:
    phase_dir = unit_dir / phase_id
    certificate = phase_dir / "checkpoint_retirement.json"
    children = P.direct_children(contract, phase_id)
    if not children:
        raise RuntimeError(f"cannot retire phase without children: {phase_id}")
    for child in children:
        expected = {"method": unit_dir.parent.name, "seed": int(unit_dir.name.removeprefix("seed")), "phase_id": child}
        if not phase_manifest_valid(unit_dir / child, expected):
            return
    manifest = P.read_json(phase_dir / "phase_manifest.json")
    checkpoint = manifest["checkpoint"]
    path = Path(checkpoint["path"])
    if certificate.is_file():
        payload = P.read_json(certificate)
        checks = {
            "passed": payload.get("passed") is True,
            "phase": payload.get("phase_id") == phase_id,
            "sha256": payload.get("checkpoint_sha256") == checkpoint["sha256"],
            "bytes": int(payload.get("checkpoint_bytes", -1)) == int(checkpoint["bytes"]),
            "children": payload.get("direct_children") == children,
        }
        if not all(checks.values()):
            raise RuntimeError(f"fork retirement certificate drift: {checks}")
        if path.is_file():
            if (
                P.sha256_file(path) != checkpoint["sha256"]
                or path.stat().st_size != int(checkpoint["bytes"])
            ):
                raise RuntimeError("certified fork checkpoint changed before cleanup")
            path.unlink()
        return
    if not path.is_file():
        raise RuntimeError(f"fork checkpoint missing before retirement: {path}")
    observed_sha = P.sha256_file(path)
    if observed_sha != checkpoint["sha256"] or path.stat().st_size != int(checkpoint["bytes"]):
        raise RuntimeError("fork checkpoint failed retirement integrity check")
    payload = {
        "schema_version": "ex56_checkpoint_retirement_v1",
        "passed": True,
        "retired_at": now_iso(),
        "phase_id": phase_id,
        "checkpoint_path": str(path),
        "checkpoint_sha256": observed_sha,
        "checkpoint_bytes": path.stat().st_size,
        "direct_children": children,
        "reason": "all direct child phases passed; formal contract retains endpoint checkpoints only",
    }
    # Seal the evidence before deletion.  If interruption occurs between these
    # two operations, resume validates the certificate and idempotently removes
    # the still-present checkpoint.
    P.atomic_json(certificate, payload)
    path.unlink()


def maybe_upload_endpoint(
    args: argparse.Namespace, snapshot: Path, unit_dir: Path, phase: dict[str, Any]
) -> bool:
    if phase["role"] != "primary_endpoint" or args.wandb_mode == "disabled":
        return True
    command = [
        sys.executable,
        str(snapshot / PACKAGE_REL / "upload_wandb.py"),
        "--contract",
        str(snapshot / PACKAGE_REL / CONTRACT_NAME),
        "--unit-dir",
        str(unit_dir),
        "--endpoint-phase",
        phase["id"],
        "--mode",
        args.wandb_mode,
        "--project",
        args.wandb_project or P.read_json(snapshot / PACKAGE_REL / CONTRACT_NAME)["wandb"]["project"],
        "--init-timeout",
        str(args.wandb_init_timeout),
    ]
    if args.wandb_entity:
        command.extend(["--entity", args.wandb_entity])
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        print(
            f"EX56 W&B upload pending for {unit_dir.name}/{phase['id']}: {completed.stderr or completed.stdout}",
            flush=True,
        )
        return False
    return True


def write_unit_manifest(unit_dir: Path, method: str, seed: int, contract: dict[str, Any]) -> None:
    phase_rows: dict[str, Any] = {}
    for phase in contract["phases"]:
        phase_dir = unit_dir / phase["id"]
        expected = {"method": method, "seed": seed, "phase_id": phase["id"]}
        if not phase_manifest_valid(phase_dir, expected):
            raise RuntimeError(f"cannot seal incomplete phase: {method}/seed{seed}/{phase['id']}")
        phase_rows[phase["id"]] = {
            "manifest_sha256": P.sha256_file(phase_dir / "phase_manifest.json"),
            "summary_sha256": P.sha256_file(phase_dir / "summary.json"),
            "metrics_sha256": P.sha256_file(phase_dir / "metrics.csv"),
        }
    endpoints = {}
    for phase in P.endpoint_phases(contract):
        manifest = P.read_json(unit_dir / phase["id"] / "phase_manifest.json")
        endpoints[phase["budget_id"]] = manifest["checkpoint"]
    payload = {
        "schema_version": P.UNIT_MANIFEST_SCHEMA,
        "passed": True,
        "method": method,
        "seed": seed,
        "completed_phases": [row["id"] for row in contract["phases"]],
        "phases": phase_rows,
        "retained_endpoints": endpoints,
        "sealed_at": now_iso(),
    }
    P.atomic_json(unit_dir / "unit_manifest.json", payload)


def run_unit(
    args: argparse.Namespace,
    snapshot: Path,
    contract: dict[str, Any],
    method: str,
    seed: int,
    gpu: int,
) -> dict[str, Any]:
    unit_dir = args.run_dir.resolve() / "formal" / method / f"seed{seed}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    if unit_manifest_valid(unit_dir, method, seed, contract):
        print(f"skip passed unit: {method}/seed{seed}", flush=True)
        return {"method": method, "seed": seed, "gpu": gpu, "status": "skipped_passed"}
    for phase in contract["phases"]:
        if STOP_EVENT.is_set():
            raise RuntimeError("suite stop requested")
        phase_dir = unit_dir / phase["id"]
        expected = {"method": method, "seed": seed, "phase_id": phase["id"]}
        if phase_manifest_valid(phase_dir, expected):
            print(f"skip passed phase: {method}/seed{seed}/{phase['id']}", flush=True)
        else:
            source = parent_checkpoint(unit_dir, phase)
            command = worker_command(
                args=args,
                snapshot=snapshot,
                phase_id=phase["id"],
                method=method,
                seed=seed,
                output_dir=phase_dir,
                source=source,
            )
            rc = tee_worker(
                command,
                worker_env(args.official_repo, gpu),
                phase_dir / "worker.log",
                gpu,
            )
            if rc != 0 or not phase_manifest_valid(phase_dir, expected):
                stop_active(except_gpu=gpu)
                raise RuntimeError(
                    f"phase failed rc={rc}: {method}/seed{seed}/{phase['id']} log={phase_dir / 'worker.log'}"
                )
        maybe_upload_endpoint(args, snapshot, unit_dir, phase)
        if phase["id"] == "backbone_11493":
            retire_fork(unit_dir, "backbone_4400", contract)
        elif phase["id"] == "backbone_17273":
            retire_fork(unit_dir, "backbone_11493", contract)
        elif phase["id"] == "cooldown_19073":
            retire_fork(unit_dir, "backbone_17273", contract)
    write_unit_manifest(unit_dir, method, seed, contract)
    return {"method": method, "seed": seed, "gpu": gpu, "status": "passed"}


def complete_pilot_retirement(manifest_path: Path) -> dict[str, Any]:
    payload = P.read_json(manifest_path)
    rows = payload.get("retired_pilot_checkpoints", [])
    if payload.get("passed") is not True or len(rows) != 2:
        raise RuntimeError("EX56 pilot retirement manifest is incomplete")
    for row in rows:
        checkpoint = Path(row["path"])
        if checkpoint.is_file():
            if (
                checkpoint.stat().st_size != int(row["bytes"])
                or P.sha256_file(checkpoint) != row["sha256"]
            ):
                raise RuntimeError("pilot checkpoint changed before idempotent retirement")
            checkpoint.unlink()
    payload["retirement_completed_at"] = now_iso()
    P.atomic_json(manifest_path, payload)
    return payload


def pilot(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    validate_identity(args, run_dir)
    if not P.read_json(run_dir / "preflight_manifest.json").get("passed"):
        raise RuntimeError("EX56 pilot requires a passed preflight")
    pilot_manifest_path = run_dir / "pilot_manifest.json"
    if pilot_manifest_path.is_file() and P.read_json(pilot_manifest_path).get("passed") is True:
        complete_pilot_retirement(pilot_manifest_path)
        print(f"skip passed EX56 engineering pilot: {run_dir}")
        return
    snapshot = run_dir / "source_snapshot"
    contract = P.read_json(snapshot / PACKAGE_REL / CONTRACT_NAME)
    pilot_dir = run_dir / "pilot"
    pilot_dir.mkdir(exist_ok=True)
    base_dir = pilot_dir / "pilot_base_2"
    method = contract["pilot"]["method"]
    seed = int(contract["pilot"]["seed"])
    expected_base = {"method": method, "seed": seed, "phase_id": "pilot_base_2"}
    if not phase_manifest_valid(base_dir, expected_base):
        command = worker_command(
            args=args,
            snapshot=snapshot,
            phase_id="pilot_base_2",
            method=method,
            seed=seed,
            output_dir=base_dir,
            source=None,
            max_new_steps=int(contract["pilot"]["planned_interrupt_after_new_steps"]),
        )
        rc = tee_worker(command, worker_env(args.official_repo, args.gpus[0]), base_dir / "worker.log", args.gpus[0])
        if rc != 75:
            raise RuntimeError(f"EX56 pilot did not stop at the planned resume boundary: rc={rc}")
        command = worker_command(
            args=args,
            snapshot=snapshot,
            phase_id="pilot_base_2",
            method=method,
            seed=seed,
            output_dir=base_dir,
            source=None,
        )
        rc = tee_worker(command, worker_env(args.official_repo, args.gpus[0]), base_dir / "worker.log", args.gpus[0])
        if rc != 0 or not phase_manifest_valid(base_dir, expected_base, full_checkpoint_hash=True):
            raise RuntimeError("EX56 in-place resume pilot failed")
    base_manifest = P.read_json(base_dir / "phase_manifest.json")
    branch_dir = pilot_dir / "pilot_branch_4"
    expected_branch = {"method": method, "seed": seed, "phase_id": "pilot_branch_4"}
    if not phase_manifest_valid(branch_dir, expected_branch):
        command = worker_command(
            args=args,
            snapshot=snapshot,
            phase_id="pilot_branch_4",
            method=method,
            seed=seed,
            output_dir=branch_dir,
            source={
                "path": Path(base_manifest["checkpoint"]["path"]),
                "sha256": base_manifest["checkpoint"]["sha256"],
                "bytes": base_manifest["checkpoint"]["bytes"],
            },
        )
        rc = tee_worker(command, worker_env(args.official_repo, args.gpus[0]), branch_dir / "worker.log", args.gpus[0])
        if rc != 0 or not phase_manifest_valid(branch_dir, expected_branch, full_checkpoint_hash=True):
            raise RuntimeError("EX56 source-branch pilot failed")
    retired = []
    for phase_dir in (base_dir, branch_dir):
        manifest = P.read_json(phase_dir / "phase_manifest.json")
        checkpoint = Path(manifest["checkpoint"]["path"])
        if not checkpoint.is_file():
            raise RuntimeError("pilot checkpoint disappeared before retirement was prepared")
        if (
            P.sha256_file(checkpoint) != manifest["checkpoint"]["sha256"]
            or checkpoint.stat().st_size != int(manifest["checkpoint"]["bytes"])
        ):
            raise RuntimeError("pilot checkpoint changed before retirement")
        retired.append(
            {
                "path": str(checkpoint),
                "sha256": manifest["checkpoint"]["sha256"],
                "bytes": checkpoint.stat().st_size,
            }
        )
    payload = {
        "schema_version": "ex56_engineering_pilot_v1",
        "passed": True,
        "planned_interrupt_return_code": 75,
        "in_place_resume": True,
        "source_checkpoint_branch": True,
        "no_wrap": True,
        "retired_pilot_checkpoints": retired,
        "retirement_prepared_at": now_iso(),
        "completed_at": now_iso(),
    }
    # Seal the expected identities before deleting either checkpoint.  A crash
    # after this write is recovered idempotently on the next pilot/all call.
    P.atomic_json(pilot_manifest_path, payload)
    complete_pilot_retirement(pilot_manifest_path)
    P.atomic_json(run_dir / "status.json", {"status": "pilot_passed", "updated_at": now_iso()})
    print(f"EX56 engineering pilot passed. Artifacts: {run_dir}")


def suite_plan(args: argparse.Namespace, contract: dict[str, Any]) -> dict[str, Any]:
    units = [
        {"method": method, "seed": int(seed)}
        for method in contract["grid"]["methods"]
        for seed in contract["grid"]["seeds"]
    ]
    units.sort(key=lambda row: (-STEP_SECONDS[row["method"]], row["seed"], row["method"]))
    return {
        "schema_version": "ex56_suite_plan_v1",
        "contract_sha256": P.sha256_file(args.run_dir.resolve() / "source_snapshot" / PACKAGE_REL / CONTRACT_NAME),
        "data_inventory_sha256": P.read_json(args.run_dir.resolve() / "data_audit.json")["inventory_sha256"],
        "gpus": args.gpus,
        "units": units,
        "phase_order": [row["id"] for row in contract["phases"]],
    }


def build_gpu_queues(
    units: list[dict[str, Any]], gpus: list[int]
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, float]]:
    if gpus != REQUIRED_PHYSICAL_GPUS:
        raise RuntimeError("EX56 queue construction requires physical GPU 3")
    queues: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in gpus}
    predicted_loads = {gpu: 0.0 for gpu in gpus}
    for unit in units:
        gpu = min(gpus, key=lambda value: (predicted_loads[value], value))
        queues[gpu].append(unit)
        predicted_loads[gpu] += STEP_SECONDS[unit["method"]]
    return queues, predicted_loads


def run_formal(args: argparse.Namespace, resume: bool) -> None:
    global STOP_EVENT
    STOP_EVENT.clear()
    run_dir = args.run_dir.resolve()
    validate_identity(args, run_dir)
    pilot_receipt = P.read_json(run_dir / "pilot_manifest.json")
    if not pilot_receipt.get("passed") or not pilot_receipt.get("retirement_completed_at"):
        raise RuntimeError("EX56 formal requires the exact-resume engineering pilot")
    snapshot = run_dir / "source_snapshot"
    contract = P.read_json(snapshot / PACKAGE_REL / CONTRACT_NAME)
    P.assert_contract(contract)
    plan_path = run_dir / "suite_plan.json"
    planned = suite_plan(args, contract)
    if plan_path.exists():
        existing = P.read_json(plan_path)
        if existing != planned:
            raise RuntimeError("EX56 suite plan changed; same-contract resume refused")
        if not resume and (run_dir / "formal").exists():
            raise RuntimeError("formal output already exists; use resume")
    else:
        if resume:
            raise RuntimeError("resume requested before a formal suite plan exists")
        P.atomic_json(plan_path, planned)
    P.atomic_json(
        run_dir / "status.json",
        {"status": "formal_running", "resume": resume, "updated_at": now_iso()},
    )

    units = planned["units"]
    results: list[dict[str, Any]] = []
    failures: list[str] = []

    queues, predicted_loads = build_gpu_queues(units, args.gpus)

    def run_gpu_queue(gpu: int) -> list[dict[str, Any]]:
        rows = []
        for unit in queues[gpu]:
            if STOP_EVENT.is_set():
                raise RuntimeError("suite stop requested")
            rows.append(
                run_unit(
                    args,
                    snapshot,
                    contract,
                    unit["method"],
                    int(unit["seed"]),
                    gpu,
                )
            )
        return rows

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        future_to_gpu = {pool.submit(run_gpu_queue, gpu): gpu for gpu in args.gpus}
        for future in as_completed(future_to_gpu):
            gpu = future_to_gpu[future]
            try:
                results.extend(future.result())
            except Exception as exc:
                failures.append(f"gpu{gpu}: {type(exc).__name__}: {exc}")
                stop_active()
    status = {
        "schema_version": "ex56_suite_status_v1",
        "passed": not failures and len(results) == len(units),
        "completed_units": len(results),
        "expected_units": len(units),
        "results": results,
        "failures": failures,
        "updated_at": now_iso(),
    }
    P.atomic_json(run_dir / "suite_status.json", status)
    if failures:
        P.atomic_json(
            run_dir / "status.json",
            {"status": "formal_incomplete_resume_required", "failures": failures, "updated_at": now_iso()},
        )
        raise RuntimeError("formal incomplete; use resume with the same run directory: " + " | ".join(failures))

    analyzer = snapshot / PACKAGE_REL / "analyze_formal.py"
    completed = subprocess.run(
        [sys.executable, str(analyzer), "build", "--run-dir", str(run_dir)],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"EX56 analysis failed rc={completed.returncode}: {completed.stdout}\n{completed.stderr}")
    P.atomic_json(run_dir / "status.json", {"status": "completed", "updated_at": now_iso()})
    print("Experiment 56 completed.")
    print(f"Artifacts: {run_dir}")
    print(f"Analysis: {run_dir / 'analysis' / 'analysis_manifest.json'}")


def verify(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    snapshot = run_dir / "source_snapshot"
    analyzer = snapshot / PACKAGE_REL / "analyze_formal.py"
    completed = subprocess.run(
        [sys.executable, str(analyzer), "verify", "--run-dir", str(run_dir), "--full-checkpoint-hash"],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"EX56 verification failed: {completed.stdout}\n{completed.stderr}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"EX56 verifier did not emit one JSON receipt: {exc}") from exc
    if payload.get("passed") is not True or payload.get("full_checkpoint_hash") is not True:
        raise RuntimeError(f"EX56 full-checkpoint receipt is not accepted: {payload}")
    payload["persisted_at"] = now_iso()
    payload["run_id"] = run_dir.name
    receipt_path = run_dir / "analysis" / "native_verify_full.json"
    P.atomic_json(receipt_path, payload)
    handoff_path = run_dir / "handoff_manifest.json"
    handoff = P.read_json(handoff_path)
    handoff["native_full_checkpoint_verify"] = {
        "path": str(receipt_path),
        "sha256": P.sha256_file(receipt_path),
        "retained_checkpoint_count": 9,
        "full_checkpoint_hash": True,
        "passed": True,
    }
    P.atomic_json(handoff_path, handoff)
    print(completed.stdout, end="")


def upload_all(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    snapshot = run_dir / "source_snapshot"
    contract = P.read_json(snapshot / PACKAGE_REL / CONTRACT_NAME)
    failures = []
    for method in contract["grid"]["methods"]:
        for seed in contract["grid"]["seeds"]:
            unit_dir = run_dir / "formal" / method / f"seed{seed}"
            for phase in P.endpoint_phases(contract):
                try:
                    if not maybe_upload_endpoint(args, snapshot, unit_dir, phase):
                        failures.append(f"{method}/seed{seed}/{phase['id']}: upload pending")
                except Exception as exc:
                    failures.append(f"{method}/seed{seed}/{phase['id']}: {exc}")
    if failures:
        raise RuntimeError("some W&B uploads failed: " + " | ".join(failures))


def main() -> int:
    args = parse_args()
    if args.mode == "check":
        _, contract = live_contract(args.live_repo)
        sources = check_live_sources(args.live_repo, contract)
        payload = {"passed": sources["passed"], "contract_checks": P.validate_contract(contract), "sources": sources}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["passed"] and all(payload["contract_checks"].values()) else 2
    try:
        if args.mode == "preflight":
            preflight(args)
        elif args.mode == "pilot":
            pilot(args)
        elif args.mode == "formal":
            run_formal(args, resume=False)
        elif args.mode == "resume":
            run_formal(args, resume=True)
        elif args.mode == "verify":
            verify(args)
        elif args.mode == "upload":
            upload_all(args)
        return 0
    except KeyboardInterrupt:
        stop_active()
        print("EX56 stopped cleanly by user; resume the same run directory.", file=sys.stderr)
        return 130
    except Exception as exc:
        stop_active()
        print(f"EX56 stopped cleanly: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
