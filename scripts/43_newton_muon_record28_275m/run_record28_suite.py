#!/usr/bin/env python3
"""Bootstrap, preflight, schedule, recover, and analyze experiment 43."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import analyze_record28 as record28_analysis
import record28_common as common
from record28_source_builder import build_all_sources, self_test_diag_math
from run_record28_cell import runtime_probe


SCRIPT_VERSION = "2026-07-31.1"
SCRIPT_DIR = Path(__file__).resolve().parent
RECOVERY_COMMAND_RELATIVE = (
    Path("commands")
    / "43_newton_muon_record28_275m"
    / "20260730_newton_muon_record28_275m.sh"
)
SNAPSHOT_FILES = (
    "record28_common.py",
    "record28_source_builder.py",
    "run_record28_cell.py",
    "run_record28_suite.py",
    "analyze_record28.py",
    "test_analyze_record28.py",
    "test_record28.py",
    "record28_contract.json",
    "RECORD28_275M_CONTRACT.md",
    "README.md",
)
ACTIVE_PROCESSES: dict[str, subprocess.Popen[str]] = {}
INTERRUPTED_SIGNAL: int | None = None


def recovery_command_path() -> Path:
    """Resolve the launcher from either the live tree or a sealed snapshot."""
    candidates = (
        SCRIPT_DIR.parent / RECOVERY_COMMAND_RELATIVE,
        SCRIPT_DIR.parents[1] / RECOVERY_COMMAND_RELATIVE,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "experiment-43 recovery launcher is missing; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--live-repo", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--data-repo-root", type=Path, required=True)
    parser.add_argument("--training-python", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument(
        "--wandb-project", default="selective-newton-muon"
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-upload-timeout-seconds", type=int, default=120
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--snapshot-active", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.gpus or len(args.gpus) != len(set(args.gpus)):
        parser.error("--gpus must contain one or more unique physical GPU IDs")
    if any(not gpu.isdigit() or str(int(gpu)) != gpu for gpu in args.gpus):
        parser.error(
            "--gpus must use canonical non-negative physical indices "
            "(for example: --gpus 0 1), not UUID aliases"
        )
    if len(args.gpus) > 2:
        parser.error("experiment 43 supports at most two quality lanes")
    if args.wandb_upload_timeout_seconds < 1:
        parser.error("--wandb-upload-timeout-seconds must be positive")
    return args


def controller_python() -> str:
    """Keep the controller venv entrypoint instead of resolving its symlink."""

    return sys.executable


def terminate_group(process: subprocess.Popen[str], grace_seconds: float = 20) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "nt":
        process.kill()
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


def suite_signal_handler(signum: int, _frame: Any) -> None:
    global INTERRUPTED_SIGNAL
    INTERRUPTED_SIGNAL = signum
    for process in list(ACTIVE_PROCESSES.values()):
        terminate_group(process)


def install_signal_handlers() -> None:
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), suite_signal_handler)


def snapshot_root(run_dir: Path) -> Path:
    return run_dir / "source_snapshot"


def snapshot_manifest_path(run_dir: Path) -> Path:
    return snapshot_root(run_dir) / "source_snapshot_manifest.json"


def snapshot_controller(run_dir: Path) -> Path:
    return snapshot_root(run_dir) / "controller"


def snapshot_training_source(run_dir: Path, method: str) -> Path:
    return snapshot_root(run_dir) / "training" / f"train_{method}.py"


def verify_snapshot(run_dir: Path) -> dict[str, Any]:
    manifest_path = snapshot_manifest_path(run_dir)
    if not manifest_path.is_file():
        raise RuntimeError(f"source snapshot manifest is missing: {manifest_path}")
    manifest = common.read_json(manifest_path)
    failures = []
    for relative, expected in manifest.get("file_sha256", {}).items():
        path = snapshot_root(run_dir) / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif common.sha256_file(path) != expected:
            failures.append(f"hash:{relative}")
    observed_files = {
        path.relative_to(snapshot_root(run_dir)).as_posix()
        for path in snapshot_root(run_dir).rglob("*")
        if path.is_file() and path != manifest_path
    }
    expected_files = set(manifest.get("file_sha256", {}))
    if observed_files != expected_files:
        failures.append(
            "inventory:"
            f"missing={sorted(expected_files - observed_files)},"
            f"extra={sorted(observed_files - expected_files)}"
        )
    if failures:
        raise RuntimeError(f"sealed source snapshot failed integrity: {failures}")
    if manifest.get("passed") is not True:
        raise RuntimeError("source snapshot manifest did not pass")
    return manifest


def bootstrap_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    root = snapshot_root(args.run_dir)
    manifest_path = snapshot_manifest_path(args.run_dir)
    if manifest_path.is_file():
        return verify_snapshot(args.run_dir)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"partial source snapshot exists without manifest: {root}")
    controller_dir = root / "controller"
    upstream_dir = root / "upstream"
    training_dir = root / "training"
    diff_dir = root / "diffs"
    for directory in (controller_dir, upstream_dir, training_dir, diff_dir):
        directory.mkdir(parents=True, exist_ok=True)

    provenance = common.audit_official_repo(args.official_repo)
    common.atomic_write_json(
        args.run_dir / "preflight" / "official_repo.json", provenance
    )
    for name in SNAPSHOT_FILES:
        source = SCRIPT_DIR / name
        if not source.is_file():
            raise RuntimeError(f"snapshot dependency is missing: {source}")
        destination = controller_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    command_destination = root / RECOVERY_COMMAND_RELATIVE
    command_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(recovery_command_path(), command_destination)

    for relative_path in common.EXPECTED_CANONICAL_SHA256:
        raw = common.git_blob(
            args.official_repo, relative_path, common.EXPECTED_COMMIT
        )
        destination = upstream_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        common.atomic_write_bytes(destination, raw)

    self_test_diag_math()
    derived = build_all_sources(args.official_repo)
    source_records: dict[str, Any] = {}
    for method, item in derived.items():
        source_path = training_dir / f"train_{method}.py"
        diff_path = diff_dir / f"train_{method}.diff"
        common.atomic_write_text(source_path, item.source)
        common.atomic_write_text(diff_path, item.unified_diff)
        source_records[method] = {
            "method": method,
            "cproj_k_mode": item.cproj_k_mode,
            "base_script": item.base_script,
            "base_canonical_sha256": item.base_canonical_sha256,
            "derived_sha256": item.derived_sha256,
            "source": str(source_path),
            "diff": str(diff_path),
        }
    if len({record["derived_sha256"] for record in source_records.values()}) != 2:
        raise RuntimeError("snapshot must contain one Muon and one shared Newton source")

    file_hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest_path:
            file_hashes[path.relative_to(root).as_posix()] = common.sha256_file(path)
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "status": "sealed",
        "passed": True,
        "created_at": common.utc_now(),
        "created_by_live_script": str(Path(__file__).resolve()),
        "official_provenance": provenance,
        "controller_python": controller_python(),
        "training_python_entrypoint": str(args.training_python),
        "sources": source_records,
        "file_sha256": file_hashes,
        "execution_authority": str(controller_dir),
        "recovery_ignores_later_live_source_drift": True,
    }
    common.atomic_write_json(manifest_path, manifest)
    return verify_snapshot(args.run_dir)


def forwarded_arguments(args: argparse.Namespace, *, snapshot_active: bool) -> list[str]:
    command = [
        controller_python(),
        str(snapshot_controller(args.run_dir) / "run_record28_suite.py"),
        "--run-dir",
        str(args.run_dir),
        "--live-repo",
        str(args.live_repo),
        "--official-repo",
        str(args.official_repo),
        "--data-repo-root",
        str(args.data_repo_root),
        "--training-python",
        str(args.training_python),
        "--gpus",
        *args.gpus,
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-project",
        args.wandb_project,
        "--wandb-upload-timeout-seconds",
        str(args.wandb_upload_timeout_seconds),
    ]
    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    if args.resume:
        command.append("--resume")
    if snapshot_active:
        command.append("--snapshot-active")
    if args.dry_run:
        command.append("--dry-run")
    return command


def run_snapshot_controller(args: argparse.Namespace) -> None:
    command = forwarded_arguments(args, snapshot_active=True)
    common.append_jsonl(
        args.run_dir / "commands.jsonl",
        {
            "label": "snapshot_controller",
            "command": command,
            "command_text": shlex.join(command),
            "started_at": common.utc_now(),
        },
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command, cwd=snapshot_controller(args.run_dir), env=env
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def source_drift_annotation(args: argparse.Namespace) -> dict[str, Any]:
    records = []
    for name in SNAPSHOT_FILES:
        live = args.live_repo / "scripts" / "43_newton_muon_record28_275m" / name
        frozen = snapshot_controller(args.run_dir) / name
        records.append(
            {
                "file": name,
                "live_exists": live.is_file(),
                "frozen_sha256": common.sha256_file(frozen),
                "live_sha256": common.sha256_file(live) if live.is_file() else None,
                "matches": live.is_file()
                and common.sha256_file(live) == common.sha256_file(frozen),
            }
        )
    return {
        "schema_version": 1,
        "informational_only": True,
        "snapshot_remains_execution_authority": True,
        "records": records,
        "checked_at": common.utc_now(),
    }


def preflight(args: argparse.Namespace, snapshot: dict[str, Any]) -> dict[str, Any]:
    preflight_dir = args.run_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    live_incomplete_processes = []
    if os.name != "nt":
        for command_path in args.run_dir.glob(
            "*/seed*/*/attempt_*/command.json"
        ):
            attempt = command_path.parent
            if (attempt / "scientific_manifest.json").is_file():
                continue
            command_record = common.read_json(command_path)
            pid = command_record.get("training_pid")
            if not isinstance(pid, int):
                continue
            cmdline_path = Path("/proc") / str(pid) / "cmdline"
            if not cmdline_path.is_file():
                continue
            cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
            recorded_command = command_record.get("command", [])
            training_source = (
                str(recorded_command[-1])
                if isinstance(recorded_command, list) and recorded_command
                else ""
            )
            if training_source and training_source in cmdline:
                live_incomplete_processes.append(
                    {
                        "pid": pid,
                        "attempt": str(attempt),
                        "cmdline": cmdline,
                    }
                )
    if live_incomplete_processes:
        raise RuntimeError(
            "an earlier experiment-43 training process is still alive; "
            f"do not start a duplicate: {live_incomplete_processes}"
        )
    test_result_path = preflight_dir / "controller_tests.json"
    test_log = preflight_dir / "controller_tests.log"
    reusable_test_result = (
        test_result_path.is_file()
        and common.read_json(test_result_path).get("passed") is True
        and common.read_json(test_result_path).get(
            "snapshot_manifest_sha256"
        )
        == common.sha256_file(snapshot_manifest_path(args.run_dir))
    )
    if reusable_test_result:
        controller_tests = common.read_json(test_result_path)
        controller_tests["reused"] = True
    else:
        test_env = os.environ.copy()
        test_env["RECORD28_OFFICIAL_REPO"] = str(args.official_repo)
        test_env["PYTHONDONTWRITEBYTECODE"] = "1"
        test_command = [
            controller_python(),
            "-m",
            "unittest",
            "-v",
            "test_record28.py",
            "test_analyze_record28.py",
        ]
        with test_log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                test_command,
                cwd=snapshot_controller(args.run_dir),
                env=test_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        controller_tests = {
            "schema_version": 1,
            "passed": completed.returncode == 0,
            "return_code": completed.returncode,
            "command": test_command,
            "command_text": shlex.join(test_command),
            "log": str(test_log),
            "official_repo": str(args.official_repo),
            "snapshot_manifest_sha256": common.sha256_file(
                snapshot_manifest_path(args.run_dir)
            ),
            "completed_at": common.utc_now(),
            "reused": False,
        }
        common.atomic_write_json(test_result_path, controller_tests)
        if not controller_tests["passed"]:
            raise RuntimeError(
                f"experiment-43 controller tests failed; inspect {test_log}"
            )
    data_path = preflight_dir / "data_certificate.json"
    data_seal_path = preflight_dir / "data_certificate_seal.json"
    previous = None
    if data_path.is_file() and data_seal_path.is_file():
        prior_seal = common.read_json(data_seal_path)
        if prior_seal.get("certificate_sha256") == common.sha256_file(data_path):
            previous = data_path
    data_certificate = common.audit_fineweb_data(
        args.data_repo_root,
        previous_certificate=previous,
    )
    common.atomic_write_json(data_path, data_certificate)
    common.atomic_write_json(
        data_seal_path,
        {
            "schema_version": 1,
            "certificate": str(data_path),
            "certificate_sha256": common.sha256_file(data_path),
            "data_fingerprint_sha256": data_certificate[
                "fingerprint_sha256"
            ],
            "sealed_at": common.utc_now(),
        },
    )

    gpus = common.query_gpus()
    selected = [common.resolve_gpu(gpus, identifier) for identifier in args.gpus]
    selected_uuids = {row["uuid"] for row in selected}
    if len(selected_uuids) != len(args.gpus):
        raise RuntimeError(
            "multiple experiment-43 GPU lane arguments resolve to the same "
            f"physical GPU: requests={args.gpus}, resolved={selected}"
        )
    active = [
        row
        for row in common.query_compute_processes()
        if row["gpu_uuid"] in selected_uuids
    ]
    if active:
        raise RuntimeError(f"selected experiment-43 GPU lanes are not idle: {active}")
    runtime = runtime_probe(args.training_python, args.gpus[0])
    if args.wandb_mode != "disabled":
        completed = subprocess.run(
            [controller_python(), "-c", "import wandb; print(wandb.__version__)"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        wandb_audit = {
            "passed": completed.returncode == 0,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        if not wandb_audit["passed"]:
            raise RuntimeError(
                "controller Python cannot import wandb; training Python is "
                "intentionally not used for upload"
            )
    else:
        wandb_audit = {"passed": True, "mode": "disabled"}
    disk = shutil.disk_usage(args.run_dir)
    minimum_free = 30 * 1024**3
    if disk.free < minimum_free:
        raise RuntimeError(
            f"insufficient free disk for 16 model-only checkpoints: "
            f"{disk.free / 1024**3:.1f} GiB < 30 GiB"
        )
    payload = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "passed": True,
        "snapshot_manifest_sha256": common.sha256_file(
            snapshot_manifest_path(args.run_dir)
        ),
        "contract_sha256": common.sha256_file(
            snapshot_controller(args.run_dir) / "record28_contract.json"
        ),
        "data_certificate": str(data_path),
        "data_fingerprint_sha256": data_certificate["fingerprint_sha256"],
        "selected_physical_gpus": selected,
        "training_runtime": runtime,
        "live_incomplete_processes": live_incomplete_processes,
        "controller_tests": controller_tests,
        "controller_python": controller_python(),
        "training_python_entrypoint": str(args.training_python),
        "wandb": wandb_audit,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "minimum_required_free_bytes": minimum_free,
        },
        "inherited_disable_fp8": os.environ.get("DISABLE_FP8"),
        "fp8_forced_enabled_by_cell_environment": True,
        "source_drift": source_drift_annotation(args),
        "timing_eligible": False,
        "completed_at": common.utc_now(),
    }
    common.atomic_write_json(preflight_dir / "preflight.json", payload)
    return payload


def method_order(seed: int) -> list[str]:
    methods = list(common.METHODS)
    shift = common.SEEDS.index(seed)
    return methods[shift:] + methods[:shift]


def assigned_gpu(args: argparse.Namespace, seed: int, method: str) -> str:
    """Pre-register a lane instead of assigning whichever GPU finishes first."""

    if len(args.gpus) == 1:
        return args.gpus[0]
    seed_index = common.SEEDS.index(seed)
    method_index = common.METHODS.index(method)
    return args.gpus[(seed_index + method_index) % 2]


def expected_cell_values(
    args: argparse.Namespace,
    snapshot: dict[str, Any],
    preflight_payload: dict[str, Any],
    *,
    stage: str,
    seed: int,
    method: str,
) -> dict[str, Any]:
    protocol = {
        "smoke": (18, 18 * 393_216),
        "formal": (1695, 666_501_120),
    }[stage]
    return {
        "stage": stage,
        "seed": seed,
        "method": method,
        "cell_key": common.cell_key(stage, seed, method),
        "contract_sha256": preflight_payload["contract_sha256"],
        "source_snapshot_sha256": preflight_payload[
            "snapshot_manifest_sha256"
        ],
        "derived_source_sha256": snapshot["sources"][method][
            "derived_sha256"
        ],
        "data_fingerprint_sha256": preflight_payload[
            "data_fingerprint_sha256"
        ],
        "total_steps": protocol[0],
        "train_tokens": protocol[1],
        "tokens_per_update": 393_216,
        "timing_eligible": False,
        "assigned_physical_gpu_request": assigned_gpu(
            args, seed, method
        ),
    }


def validate_scientific_attempt(
    attempt_dir: Path,
    expected: dict[str, Any],
    *,
    verify_checkpoint: bool,
) -> dict[str, Any]:
    manifest_path = attempt_dir / "scientific_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"scientific manifest is missing: {manifest_path}")
    manifest = common.read_json(manifest_path)
    checks = {
        "passed": manifest.get("passed") is True,
        "status": manifest.get("status") == "scientifically_complete",
        **{
            key: manifest.get(key) == value for key, value in expected.items()
        },
    }
    hashes = manifest.get("artifact_hashes", {})
    declared_artifacts = manifest.get("artifacts", [])
    required_artifacts = {
        "checks.json",
        "command.json",
        "metrics.csv",
        "runtime.json",
        "stdout.log",
        "summary.json",
        "training.log",
        "artifact_hashes.json",
    }
    if expected["stage"] == "formal":
        required_artifacts.add("checkpoint_hash.json")
    checks["artifact_hashes"] = bool(hashes) and all(
        (attempt_dir / name).is_file()
        and common.sha256_file(attempt_dir / name) == expected_hash
        for name, expected_hash in hashes.items()
    )
    checks["artifact_inventory"] = (
        isinstance(declared_artifacts, list)
        and set(declared_artifacts) == required_artifacts
        and set(hashes) == required_artifacts - {"artifact_hashes.json"}
    )
    checks["artifact_hash_map_sealed"] = (
        (attempt_dir / "artifact_hashes.json").is_file()
        and common.read_json(attempt_dir / "artifact_hashes.json") == hashes
    )
    if expected["stage"] == "formal":
        checkpoint_record = manifest.get("checkpoint") or {}
        checkpoint = Path(checkpoint_record.get("path", ""))
        checks["checkpoint_exists"] = checkpoint.is_file()
        checks["checkpoint_size"] = (
            checkpoint.is_file()
            and checkpoint.stat().st_size == checkpoint_record.get("bytes")
        )
        checks["checkpoint_sha256"] = (
            not verify_checkpoint
            or (
                checkpoint.is_file()
                and common.sha256_file(checkpoint)
                == checkpoint_record.get("sha256")
            )
        )
        checks["checkpoint_scope"] = (
            checkpoint_record.get("checkpoint_scope") == "model_only"
        )
    if not all(checks.values()):
        raise RuntimeError(
            f"scientific attempt integrity failed at {attempt_dir}: {checks}"
        )
    return manifest


def accepted_attempt(
    method_dir: Path,
    expected: dict[str, Any],
    *,
    recover_orphan: bool = True,
) -> Path | None:
    pointer = method_dir / "accepted.json"
    if pointer.is_file():
        payload = common.read_json(pointer)
        attempt = Path(payload["attempt_dir"])
        if not attempt.is_absolute():
            attempt = (method_dir / attempt).resolve()
        if common.sha256_file(
            attempt / "scientific_manifest.json"
        ) != payload.get("scientific_manifest_sha256"):
            raise RuntimeError(f"accepted pointer hash mismatch: {pointer}")
        validate_scientific_attempt(attempt, expected, verify_checkpoint=True)
        return attempt
    if not recover_orphan:
        return None
    candidates = []
    for attempt in sorted(method_dir.glob("attempt_*")):
        if (attempt / "scientific_manifest.json").is_file():
            validate_scientific_attempt(attempt, expected, verify_checkpoint=True)
            candidates.append(attempt.resolve())
    if len(candidates) > 1:
        raise RuntimeError(
            f"multiple complete attempts without accepted pointer: {candidates}"
        )
    if candidates:
        seal_accepted(method_dir, candidates[0])
        return candidates[0]
    return None


def seal_accepted(method_dir: Path, attempt: Path) -> None:
    manifest_path = attempt / "scientific_manifest.json"
    manifest = common.read_json(manifest_path)
    common.atomic_write_json(
        method_dir / "accepted.json",
        {
            "schema_version": 1,
            "cell_key": manifest["cell_key"],
            "attempt_dir": attempt.name,
            "scientific_manifest": f"{attempt.name}/scientific_manifest.json",
            "remote_attempt_dir": str(attempt.resolve()),
            "scientific_manifest_sha256": common.sha256_file(manifest_path),
            "accepted_at": common.utc_now(),
            "wandb_is_separate_from_scientific_acceptance": True,
        },
    )


def next_attempt(method_dir: Path) -> Path:
    method_dir.mkdir(parents=True, exist_ok=True)
    index = 1
    while (method_dir / f"attempt_{index:03d}").exists():
        index += 1
    return method_dir / f"attempt_{index:03d}"


def cell_command(
    args: argparse.Namespace,
    *,
    stage: str,
    seed: int,
    method: str,
    gpu: str,
    attempt: Path,
    upload_only: bool = False,
) -> list[str]:
    command = [
        controller_python(),
        str(snapshot_controller(args.run_dir) / "run_record28_cell.py"),
        "--attempt-dir",
        str(attempt),
        "--stage",
        stage,
        "--seed",
        str(seed),
        "--method",
        method,
        "--physical-gpu",
        gpu,
        "--training-python",
        str(args.training_python),
        "--training-source",
        str(snapshot_training_source(args.run_dir, method)),
        "--contract",
        str(snapshot_controller(args.run_dir) / "record28_contract.json"),
        "--source-snapshot-manifest",
        str(snapshot_manifest_path(args.run_dir)),
        "--data-certificate",
        str(args.run_dir / "preflight" / "data_certificate.json"),
        "--data-repo-root",
        str(args.data_repo_root),
        "--result-root",
        str(args.run_dir.parent.parent),
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-project",
        args.wandb_project,
    ]
    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    if upload_only:
        command.append("--upload-only")
    return command


def launch_jobs(
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
    expected_by_label: dict[str, dict[str, Any]],
) -> None:
    pending = list(jobs)
    active: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    logs = args.run_dir / "controller_logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            pending_index = next(
                (
                    index
                    for index, candidate in enumerate(pending)
                    if candidate["assigned_gpu"] == gpu
                ),
                None,
            )
            if pending_index is None:
                continue
            job = pending.pop(pending_index)
            label = job["label"]
            attempt = Path(job["attempt"])
            attempt.mkdir(parents=True, exist_ok=False)
            command = cell_command(
                args,
                stage=job["stage"],
                seed=job["seed"],
                method=job["method"],
                gpu=gpu,
                attempt=attempt,
            )
            log_path = logs / f"{label.replace('/', '_')}_{attempt.name}.log"
            log_handle = log_path.open("w", encoding="utf-8", buffering=1)
            common.append_jsonl(
                args.run_dir / "commands.jsonl",
                {
                    "label": label,
                    "attempt": str(attempt),
                    "physical_gpu": gpu,
                    "command": command,
                    "command_text": shlex.join(command),
                    "log": str(log_path),
                    "started_at": common.utc_now(),
                },
            )
            print(f"START gpu={gpu} label={label} attempt={attempt.name}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=snapshot_controller(args.run_dir),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            ACTIVE_PROCESSES[gpu] = process
            active[gpu] = {
                **job,
                "process": process,
                "log_handle": log_handle,
                "log_path": log_path,
                "attempt": attempt,
            }
        finished = []
        for gpu, job in list(active.items()):
            return_code = job["process"].poll()
            if return_code is None:
                continue
            job["log_handle"].close()
            ACTIVE_PROCESSES.pop(gpu, None)
            label = job["label"]
            print(
                f"END gpu={gpu} label={label} return_code={return_code}",
                flush=True,
            )
            if return_code == 0:
                try:
                    validate_scientific_attempt(
                        job["attempt"],
                        expected_by_label[label],
                        verify_checkpoint=True,
                    )
                    method_dir = job["attempt"].parent
                    seal_accepted(method_dir, job["attempt"])
                except Exception as error:
                    return_code = 99
                    failures.append(
                        {
                            "label": label,
                            "gpu": gpu,
                            "return_code": return_code,
                            "attempt": str(job["attempt"]),
                            "log": str(job["log_path"]),
                            "validation_error": repr(error),
                        }
                    )
            else:
                failures.append(
                    {
                        "label": label,
                        "gpu": gpu,
                        "return_code": return_code,
                        "attempt": str(job["attempt"]),
                        "log": str(job["log_path"]),
                    }
                )
            finished.append(gpu)
        for gpu in finished:
            active.pop(gpu, None)
        if failures or INTERRUPTED_SIGNAL is not None:
            for process in list(ACTIVE_PROCESSES.values()):
                terminate_group(process)
            for job in active.values():
                job["log_handle"].close()
            ACTIVE_PROCESSES.clear()
            common.atomic_write_json(
                args.run_dir / "worker_failures.json", failures
            )
            if INTERRUPTED_SIGNAL is not None:
                raise KeyboardInterrupt
            raise RuntimeError(f"experiment-43 worker failures: {failures}")
        if pending or active:
            time.sleep(5)


def build_jobs(
    args: argparse.Namespace,
    snapshot: dict[str, Any],
    preflight_payload: dict[str, Any],
    *,
    stage: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    expected_by_label: dict[str, dict[str, Any]] = {}
    seeds = (2026,) if stage == "smoke" else common.SEEDS
    for seed in seeds:
        methods = list(common.METHODS) if stage == "smoke" else method_order(seed)
        for method in methods:
            label = f"{stage}/seed{seed}/{method}"
            expected = expected_cell_values(
                args,
                snapshot,
                preflight_payload,
                stage=stage,
                seed=seed,
                method=method,
            )
            expected_by_label[label] = expected
            method_dir = args.run_dir / stage / f"seed{seed}" / method
            if accepted_attempt(method_dir, expected) is None:
                jobs.append(
                    {
                        "label": label,
                        "stage": stage,
                        "seed": seed,
                        "method": method,
                        "assigned_gpu": assigned_gpu(args, seed, method),
                        "attempt": str(next_attempt(method_dir)),
                    }
                )
    return jobs, expected_by_label


def validate_pairing(
    args: argparse.Namespace,
    snapshot: dict[str, Any],
    preflight_payload: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    seeds = (2026,) if stage == "smoke" else common.SEEDS
    cells = []
    for seed in seeds:
        seed_cells = []
        for method in common.METHODS:
            expected = expected_cell_values(
                args,
                snapshot,
                preflight_payload,
                stage=stage,
                seed=seed,
                method=method,
            )
            attempt = accepted_attempt(
                args.run_dir / stage / f"seed{seed}" / method, expected
            )
            if attempt is None:
                raise RuntimeError(f"accepted {stage} cell missing: seed={seed} {method}")
            manifest = common.read_json(attempt / "scientific_manifest.json")
            metrics = list(
                __import__("csv").DictReader(
                    (attempt / "metrics.csv").open(
                        "r", encoding="utf-8", newline=""
                    )
                )
            )
            seed_cells.append(
                {
                    "method": method,
                    "attempt": str(attempt),
                    "init_sha256": manifest["init_sha256"],
                    "step0_val_loss": float(metrics[0]["val_loss"]),
                    "physical_gpu_index": str(
                        manifest["physical_gpu"]["index"]
                    ),
                    "physical_gpu_uuid": manifest["physical_gpu"]["uuid"],
                    "scientific_manifest_sha256": common.sha256_file(
                        attempt / "scientific_manifest.json"
                    ),
                }
            )
        step0_values = [cell["step0_val_loss"] for cell in seed_cells]
        checks = {
            "same_initialization": len(
                {cell["init_sha256"] for cell in seed_cells}
            )
            == 1,
            "same_step0_validation": max(step0_values) - min(step0_values)
            <= 1e-6,
            "four_methods": {cell["method"] for cell in seed_cells}
            == set(common.METHODS),
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"{stage} seed {seed} pairing gate failed: {checks}"
            )
        cells.extend(seed_cells)
    lane_balance: dict[str, dict[str, int]] = {}
    for method in common.METHODS:
        lane_balance[method] = {}
        for cell in cells:
            if cell["method"] != method:
                continue
            lane = cell["physical_gpu_index"]
            lane_balance[method][lane] = lane_balance[method].get(lane, 0) + 1
    lane_balance_passed = True
    if stage == "formal" and len(args.gpus) == 2:
        expected_indices = {
            common.resolve_gpu(common.query_gpus(), gpu)["index"]
            for gpu in args.gpus
        }
        lane_balance_passed = all(
            set(counts) == expected_indices
            and all(count == 2 for count in counts.values())
            for counts in lane_balance.values()
        )
        if not lane_balance_passed:
            raise RuntimeError(
                f"formal method/GPU lane balance failed: {lane_balance}"
            )
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "passed": True,
        "checks": {
            "all_cells": len(cells)
            == (4 if stage == "smoke" else 16),
            "snapshot_hash": True,
            "data_fingerprint": True,
            "paired_initialization": True,
            "paired_step0_validation": True,
            "method_gpu_lane_balance": lane_balance_passed,
        },
        "method_gpu_lane_counts": lane_balance,
        "cells": cells,
        "timing_eligible": False,
        "completed_at": common.utc_now(),
    }
    output = (
        args.run_dir / "smoke" / "smoke_manifest.json"
        if stage == "smoke"
        else args.run_dir / "formal" / "formal_manifest.json"
    )
    common.atomic_write_json(output, manifest)
    return manifest


def retry_pending_uploads(args: argparse.Namespace) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for seed in common.SEEDS:
        for method in common.METHODS:
            method_dir = args.run_dir / "formal" / f"seed{seed}" / method
            pointer = common.read_json(method_dir / "accepted.json")
            attempt = Path(pointer["attempt_dir"])
            if not attempt.is_absolute():
                attempt = (method_dir / attempt).resolve()
            wandb_path = attempt / "wandb.json"
            existing = (
                common.read_json(wandb_path) if wandb_path.is_file() else {}
            )
            if existing.get("complete") is True:
                continue
            pending_row = {
                "seed": seed,
                "method": method,
                "attempt": str(attempt),
                "wandb": str(wandb_path),
            }
            if args.wandb_mode == "disabled":
                pending.append(
                    {
                        **pending_row,
                        "reason": "formal_wandb_upload_disabled",
                    }
                )
                continue
            if (
                args.wandb_mode == "offline"
                and existing.get("status")
                == "offline_created_pending_sync"
            ):
                pending.append(
                    {
                        **pending_row,
                        "reason": "offline_run_requires_wandb_sync",
                    }
                )
                continue
            command = cell_command(
                args,
                stage="formal",
                seed=seed,
                method=method,
                gpu=args.gpus[0],
                attempt=attempt,
                upload_only=True,
            )
            log = (
                args.run_dir
                / "controller_logs"
                / f"upload_seed{seed}_{method}.log"
            )
            common.append_jsonl(
                args.run_dir / "commands.jsonl",
                {
                    "label": f"upload/seed{seed}/{method}",
                    "command": command,
                    "command_text": shlex.join(command),
                    "log": str(log),
                    "started_at": common.utc_now(),
                    "upload_only": True,
                },
            )
            with log.open("a", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    command,
                    cwd=snapshot_controller(args.run_dir),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=(os.name != "nt"),
                )
                timed_out = False
                try:
                    process.wait(timeout=args.wandb_upload_timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    terminate_group(process, grace_seconds=5)
                    handle.write(
                        "\nRECORD28_WANDB_UPLOAD_TIMEOUT "
                        f"seconds={args.wandb_upload_timeout_seconds}\n"
                    )
                    handle.flush()
            if timed_out:
                common.atomic_write_json(
                    wandb_path,
                    {
                        **existing,
                        "schema_version": 1,
                        "status": "upload_timeout_pending_retry",
                        "complete": False,
                        "mode": args.wandb_mode,
                        "required_for_paper_handoff": True,
                        "timeout_seconds": args.wandb_upload_timeout_seconds,
                        "updated_at": common.utc_now(),
                    },
                )
            complete = wandb_path.is_file() and common.read_json(
                wandb_path
            ).get("complete") is True
            if not complete:
                pending.append(
                    {
                        **pending_row,
                        "reason": (
                            "upload_timeout"
                            if timed_out
                            else "upload_incomplete"
                        ),
                    }
                )
    return pending


def validate_reusable_analysis(
    manifest_path: Path, run_dir: Path
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Validate both analysis outputs and the exact 16 accepted inputs."""

    checks: dict[str, Any] = {
        "manifest_exists": manifest_path.is_file(),
    }
    if not checks["manifest_exists"]:
        return None, checks
    try:
        payload = common.read_json(manifest_path)
        checks.update(
            {
                "manifest_object": isinstance(payload, dict),
                "passed": isinstance(payload, dict)
                and payload.get("passed") is True,
                "status": isinstance(payload, dict)
                and payload.get("status") == "passed",
                "run_dir": isinstance(payload, dict)
                and Path(payload.get("run_dir", "")).resolve()
                == run_dir.resolve(),
                "decision": isinstance(payload, dict)
                and isinstance(payload.get("decision"), dict),
                "primary_classifications": isinstance(payload, dict)
                and isinstance(payload.get("primary_classifications"), dict)
                and payload.get("primary_classifications")
                == (payload.get("decision") or {}).get(
                    "primary_classifications"
                ),
            }
        )
        hashes = (
            payload.get("artifact_hashes", payload.get("artifact_sha256"))
            if isinstance(payload, dict)
            else None
        )
        artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
        checks["artifact_hash_map"] = isinstance(hashes, dict) and bool(hashes)
        checks["artifact_inventory"] = (
            isinstance(artifacts, list)
            and isinstance(hashes, dict)
            and set(artifacts) == set(hashes)
            and all(
                isinstance(name, str)
                and Path(name).name == name
                and name not in {"", ".", ".."}
                for name in artifacts
            )
        )
        checks["artifact_hashes"] = bool(checks["artifact_inventory"]) and all(
            (manifest_path.parent / name).is_file()
            and common.sha256_file(manifest_path.parent / name)
            == expected_hash
            for name, expected_hash in hashes.items()
        )
        current_fingerprints = (
            record28_analysis.collect_accepted_cell_fingerprints(run_dir)
        )
        current_digest = (
            record28_analysis.accepted_cells_fingerprint_sha256(
                current_fingerprints
            )
        )
        checks["accepted_cell_fingerprints"] = (
            payload.get("accepted_cell_fingerprints")
            == current_fingerprints
        )
        checks["accepted_cells_fingerprint_sha256"] = (
            payload.get("accepted_cells_fingerprint_sha256")
            == current_digest
        )
    except Exception as error:
        checks["validation_error"] = repr(error)
        return None, checks
    checks["passed_all"] = all(
        value is True
        for key, value in checks.items()
        if key != "validation_error"
    )
    if not checks["passed_all"]:
        return None, checks
    return payload, checks


def select_analysis_output(
    run_dir: Path,
) -> tuple[Path | None, Path, list[dict[str, Any]]]:
    """Return a reusable manifest or a never-before-used output directory."""

    candidates = [run_dir / "analysis"]
    candidates.extend(
        sorted(
            (
                path
                for path in run_dir.glob("analysis_retry_*")
                if path.is_dir()
            ),
            key=lambda path: path.name,
        )
    )
    audits: list[dict[str, Any]] = []
    for output in candidates:
        manifest = output / "record28_analysis_manifest.json"
        payload, checks = validate_reusable_analysis(manifest, run_dir)
        audits.append(
            {
                "output_dir": str(output),
                "manifest": str(manifest),
                "checks": checks,
            }
        )
        if payload is not None:
            return manifest, output, audits

    base = run_dir / "analysis"
    if not base.exists():
        return None, base, audits
    retry_index = 1
    while (run_dir / f"analysis_retry_{retry_index:03d}").exists():
        retry_index += 1
    return None, run_dir / f"analysis_retry_{retry_index:03d}", audits


def run_analysis(args: argparse.Namespace) -> Path:
    reusable, output, audits = select_analysis_output(args.run_dir)
    common.append_jsonl(
        args.run_dir / "analysis_reuse_audit.jsonl",
        {
            "checked_at": common.utc_now(),
            "reusable_manifest": str(reusable) if reusable else None,
            "selected_output": str(output),
            "candidates": audits,
        },
    )
    if reusable is not None:
        return reusable
    manifest = output / "record28_analysis_manifest.json"
    command = [
        controller_python(),
        str(snapshot_controller(args.run_dir) / "analyze_record28.py"),
        "--run-dir",
        str(args.run_dir),
        "--contract",
        str(snapshot_controller(args.run_dir) / "record28_contract.json"),
        "--output-dir",
        str(output),
    ]
    common.append_jsonl(
        args.run_dir / "commands.jsonl",
        {
            "label": "analysis",
            "command": command,
            "command_text": shlex.join(command),
            "started_at": common.utc_now(),
        },
    )
    subprocess.run(command, cwd=snapshot_controller(args.run_dir), check=True)
    payload, checks = validate_reusable_analysis(manifest, args.run_dir)
    if payload is None:
        raise RuntimeError(
            f"experiment-43 analysis did not pass sealed validation: "
            f"{manifest}: {checks}"
        )
    return manifest


def final_analysis_fields(analysis: dict[str, Any]) -> dict[str, Any]:
    decision = analysis.get("decision")
    classifications = analysis.get("primary_classifications")
    checks = {
        "decision": isinstance(decision, dict),
        "primary_classifications": isinstance(classifications, dict),
        "classification_consistency": isinstance(decision, dict)
        and classifications == decision.get("primary_classifications"),
        "accepted_fingerprint": isinstance(
            analysis.get("accepted_cells_fingerprint_sha256"), str
        )
        and len(analysis["accepted_cells_fingerprint_sha256"]) == 64,
    }
    if not all(checks.values()):
        raise RuntimeError(f"analysis summary contract failed: {checks}")
    return {
        "analysis_decision": decision,
        "analysis_primary_classifications": classifications,
        "analysis_statistical_seed_append_gate_triggered": analysis[
            "statistical_seed_append_gate_triggered"
        ],
        "analysis_accepted_cells_fingerprint_sha256": analysis[
            "accepted_cells_fingerprint_sha256"
        ],
    }


def write_handoff_manifest(run_dir: Path) -> Path:
    output = run_dir / "handoff_manifest.json"
    files = []
    excluded_checkpoints = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path == output:
            continue
        relative = path.relative_to(run_dir)
        if path.suffix == ".pt":
            checkpoint_hash_path = path.parent / "checkpoint_hash.json"
            checkpoint_sha256 = None
            if checkpoint_hash_path.is_file():
                checkpoint_record = common.read_json(checkpoint_hash_path)
                if (
                    Path(checkpoint_record.get("path", "")).resolve()
                    == path.resolve()
                    and checkpoint_record.get("bytes") == path.stat().st_size
                ):
                    checkpoint_sha256 = checkpoint_record.get("sha256")
            if not isinstance(checkpoint_sha256, str):
                checkpoint_sha256 = common.sha256_file(path)
            excluded_checkpoints.append(
                {
                    "path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": checkpoint_sha256,
                    "reason": "remote-retained model-only checkpoint",
                }
            )
            continue
        if (
            path.name.endswith(".lock")
            or "wandb_local" in relative.parts
            or "runtime_cache" in relative.parts
        ):
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": common.sha256_file(path),
            }
        )
    payload = {
        "schema_version": 1,
        "passed": True,
        "run_dir": str(run_dir),
        "transfer_policy": (
            "copy the listed files/directories normally; no archive is "
            "required and model-only .pt checkpoints stay remote"
        ),
        "included_file_count": len(files),
        "included_total_bytes": sum(item["bytes"] for item in files),
        "files": files,
        "excluded_checkpoints": excluded_checkpoints,
        "created_at": common.utc_now(),
    }
    common.atomic_write_json(output, payload)
    return output


def dry_run_plan(
    args: argparse.Namespace, snapshot: dict[str, Any]
) -> None:
    plan = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "dry_run": True,
        "methods": list(common.METHODS),
        "seeds": list(common.SEEDS),
        "smoke_cells": 4,
        "formal_cells": 16,
        "gpus": args.gpus,
        "method_orders": {
            str(seed): method_order(seed) for seed in common.SEEDS
        },
        "formal_gpu_assignment": {
            f"seed{seed}/{method}": assigned_gpu(args, seed, method)
            for seed in common.SEEDS
            for method in common.METHODS
        },
        "source_hashes": {
            method: snapshot["sources"][method]["derived_sha256"]
            for method in common.METHODS
        },
        "timing_eligible": False,
        "created_at": common.utc_now(),
    }
    common.atomic_write_json(args.run_dir / "dry_run_plan.json", plan)
    print(f"RECORD28_DRY_RUN_PLAN={args.run_dir / 'dry_run_plan.json'}")


def freeze_or_validate_suite_plan(
    args: argparse.Namespace, snapshot: dict[str, Any]
) -> dict[str, Any]:
    path = args.run_dir / "suite_plan.json"
    visible_gpus = common.query_gpus()
    gpu_uuid_by_request = {
        request: common.resolve_gpu(visible_gpus, request)["uuid"]
        for request in args.gpus
    }
    if len(set(gpu_uuid_by_request.values())) != len(args.gpus):
        raise RuntimeError(
            "requested GPU lanes are not distinct physical devices: "
            f"{gpu_uuid_by_request}"
        )
    proposed = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": 43,
        "recipe": "Newton-Muon-2 upstream near Record #28",
        "gpus": list(args.gpus),
        "gpu_uuid_by_request": gpu_uuid_by_request,
        "formal_gpu_assignment": {
            f"seed{seed}/{method}": assigned_gpu(args, seed, method)
            for seed in common.SEEDS
            for method in common.METHODS
        },
        "methods": list(common.METHODS),
        "seeds": list(common.SEEDS),
        "official_repo_at_bootstrap": str(args.official_repo),
        "data_repo_root": str(args.data_repo_root),
        "training_python_entrypoint": str(args.training_python),
        "source_snapshot_manifest_sha256": common.sha256_file(
            snapshot_manifest_path(args.run_dir)
        ),
        "contract_sha256": common.sha256_file(
            snapshot_controller(args.run_dir) / "record28_contract.json"
        ),
        "smoke_updates": 18,
        "formal_updates": 1695,
        "tokens_per_update": 393_216,
        "formal_train_tokens": 666_501_120,
        "timing_eligible": False,
    }
    if path.is_file():
        frozen = common.read_json(path)
        checks = {
            key: frozen.get(key) == value
            for key, value in proposed.items()
            if key not in {"script_version"}
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"resume arguments drift from frozen suite plan: {checks}"
            )
        return frozen
    proposed["frozen_at"] = common.utc_now()
    common.atomic_write_json(path, proposed)
    return proposed


def main_snapshot(args: argparse.Namespace) -> None:
    install_signal_handlers()
    snapshot = verify_snapshot(args.run_dir)
    common.atomic_write_json(
        args.run_dir / "live_source_drift_annotation.json",
        source_drift_annotation(args),
    )
    if args.dry_run:
        dry_run_plan(args, snapshot)
        return

    run_lock = args.run_dir / ".run.lock"
    with common.exclusive_file_lock(
        run_lock,
        timeout_seconds=1,
        metadata={
            "experiment": 43,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": common.utc_now(),
        },
    ):
        common.atomic_write_json(
            args.run_dir / "status.json",
            {
                "status": "running",
                "script_version": SCRIPT_VERSION,
                "snapshot_active": True,
                "resume": args.resume,
                "gpus": args.gpus,
                "updated_at": common.utc_now(),
            },
        )
        freeze_or_validate_suite_plan(args, snapshot)
        preflight_payload = preflight(args, snapshot)
        experiment_root = args.run_dir.parent
        common.atomic_write_json(
            experiment_root / "LATEST.json",
            {
                "run_dir": str(args.run_dir),
                "source_snapshot_manifest": str(
                    snapshot_manifest_path(args.run_dir)
                ),
                "updated_at": common.utc_now(),
            },
        )

        smoke_jobs, smoke_expected = build_jobs(
            args, snapshot, preflight_payload, stage="smoke"
        )
        launch_jobs(args, smoke_jobs, smoke_expected)
        validate_pairing(
            args, snapshot, preflight_payload, stage="smoke"
        )

        formal_jobs, formal_expected = build_jobs(
            args, snapshot, preflight_payload, stage="formal"
        )
        launch_jobs(args, formal_jobs, formal_expected)
        validate_pairing(
            args, snapshot, preflight_payload, stage="formal"
        )
        # Commit the scientific analysis before starting any network-bound
        # logging.  A W&B outage can therefore leave only an upload backlog,
        # never an unanalyzed or unsealed experiment.
        analysis_manifest = run_analysis(args)
        analysis = common.read_json(analysis_manifest)
        pending_wandb = retry_pending_uploads(args)
        final = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "status": (
                "completed"
                if not pending_wandb
                else "scientifically_completed_wandb_pending"
            ),
            "passed": True,
            "scientific_integrity_passed": True,
            "wandb_complete": not pending_wandb,
            "wandb_pending": pending_wandb,
            "run_dir": str(args.run_dir),
            "source_snapshot_manifest": str(
                snapshot_manifest_path(args.run_dir)
            ),
            "source_snapshot_manifest_sha256": common.sha256_file(
                snapshot_manifest_path(args.run_dir)
            ),
            "preflight": str(args.run_dir / "preflight" / "preflight.json"),
            "smoke_manifest": str(
                args.run_dir / "smoke" / "smoke_manifest.json"
            ),
            "formal_manifest": str(
                args.run_dir / "formal" / "formal_manifest.json"
            ),
            "analysis_manifest": str(analysis_manifest),
            **final_analysis_fields(analysis),
            "formal_cells": 16,
            "timing_eligible": False,
            "completed_at": common.utc_now(),
        }
        common.atomic_write_json(
            args.run_dir / "record28_suite_manifest.json", final
        )
        common.atomic_write_json(
            args.run_dir / "status.json",
            {
                "status": final["status"],
                "passed": True,
                "manifest": str(
                    args.run_dir / "record28_suite_manifest.json"
                ),
                "completed_at": common.utc_now(),
            },
        )
        handoff_manifest = write_handoff_manifest(args.run_dir)
        print(f"RECORD28_ARTIFACTS={args.run_dir}")
        print(
            f"RECORD28_MANIFEST="
            f"{args.run_dir / 'record28_suite_manifest.json'}"
        )
        print(f"RECORD28_HANDOFF_MANIFEST={handoff_manifest}")
        if pending_wandb:
            print(
                "RECORD28_WANDB_PENDING=1; rerun the printed recovery command. "
                "Accepted training cells will not be repeated."
            )


def normalize_args(args: argparse.Namespace) -> None:
    args.run_dir = args.run_dir.expanduser().resolve()
    args.live_repo = args.live_repo.expanduser().resolve()
    args.official_repo = args.official_repo.expanduser().resolve()
    args.data_repo_root = args.data_repo_root.expanduser().resolve()
    # Do not resolve the venv executable symlink.
    args.training_python = args.training_python.expanduser().absolute()


def main() -> None:
    args = parse_args()
    normalize_args(args)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.snapshot_active:
        main_snapshot(args)
        return
    bootstrap_lock = args.run_dir / ".bootstrap.lock"
    with common.exclusive_file_lock(
        bootstrap_lock,
        timeout_seconds=1,
        metadata={
            "experiment": 43,
            "pid": os.getpid(),
            "created_at": common.utc_now(),
        },
    ):
        bootstrap_snapshot(args)
    args.resume = args.resume or any(
        path.name not in {".bootstrap.lock", "source_snapshot", "preflight"}
        for path in args.run_dir.iterdir()
    )
    run_snapshot_controller(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
