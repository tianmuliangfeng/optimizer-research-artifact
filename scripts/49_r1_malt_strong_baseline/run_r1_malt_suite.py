"""Resumable pilot and dual-method three-seed Experiment-49 suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_r1_malt as runner


ACCEPTED = {"completed_valid", "completed_valid_local_wandb_incomplete"}
FORMAL_SEEDS = (2024, 2025, 2026)
FORMAL_METHODS = runner.FORMAL_METHODS
EX45_SUMMARY_SHA256 = "eda9ba780c934230c641e74c295c59c5105c384d8b5decff4623926fbec5627b"
EX45_MANIFEST_SHA256 = "7e7893d37c4049f327f2da2d4b8d30c8be012f02de614a7504e7b3f384761b59"
EXPECTED_TRAIN_SHARDS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "pilot", "formal", "verify", "all"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--training-python", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--experiment45-summary", type=Path)
    parser.add_argument("--experiment45-analysis-manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus contains duplicates")
    if not args.gpus:
        parser.error("at least one GPU is required")
    if any(not token.isdigit() or str(int(token)) != token for token in args.gpus):
        parser.error("every GPU token must be one canonical non-negative integer")
    if args.stage in {"verify", "all"} and (
        args.experiment45_summary is None or args.experiment45_analysis_manifest is None
    ):
        parser.error("verify/all requires the accepted Experiment-45 summary and analysis manifest")
    return args


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty CSV")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_analysis_bundle(manifest_path: Path) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
        if manifest.get("status") != "completed_valid":
            return False
        seal = manifest_path.with_suffix(".sha256")
        if not seal.is_file() or seal.read_text(encoding="ascii").split()[0] != sha256_file(manifest_path):
            return False
        for record in manifest.get("input_files", []):
            path = Path(str(record["path"])).expanduser().resolve()
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                return False
        outputs = manifest.get("outputs", [])
        if outputs and isinstance(outputs[0], dict):
            for record in outputs:
                path = manifest_path.parent / str(record["path"])
                if not path.is_file() or sha256_file(path) != record["sha256"]:
                    return False
        else:
            output_hashes = manifest.get("output_sha256", {})
            if set(outputs) != set(output_hashes):
                return False
            for name in outputs:
                path = manifest_path.parent / str(name)
                if not path.is_file() or sha256_file(path) != output_hashes[name]:
                    return False
        return True
    except (IndexError, KeyError, TypeError, ValueError, OSError, UnicodeError, json.JSONDecodeError):
        return False


def accepted_analysis_dir(parent: Path, manifest_name: str) -> Path | None:
    candidates = [
        path
        for path in sorted(parent.glob("analysis*"))
        if path.is_dir() and validate_analysis_bundle(path / manifest_name)
    ]
    if len(candidates) > 1:
        raise RuntimeError(f"multiple accepted analysis bundles under {parent}: {candidates}")
    return candidates[0] if candidates else None


def next_analysis_dir(parent: Path) -> Path:
    first = parent / "analysis"
    if not first.exists():
        return first
    index = 1
    while (parent / f"analysis_retry_{index:03d}").exists():
        index += 1
    return parent / f"analysis_retry_{index:03d}"


def append_command(run_dir: Path, payload: dict[str, object]) -> None:
    path = run_dir / "commands.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def requested_gpu_inventory(gpus: list[str]) -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi inventory failed: {completed.stderr.strip()}")
    inventory: dict[str, dict[str, object]] = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 4)]
        if len(parts) != 5:
            raise RuntimeError(f"unexpected nvidia-smi inventory row: {line!r}")
        index, uuid, name, memory_mib, driver_version = parts
        inventory[index] = {
            "physical_index": index,
            "uuid": uuid,
            "name": name,
            "memory_mib": int(memory_mib),
            "driver_version": driver_version,
        }
    missing = [gpu for gpu in gpus if gpu not in inventory]
    if missing:
        raise RuntimeError(f"requested physical GPUs are absent: {missing}")
    selected = [inventory[gpu] for gpu in gpus]
    if any("H100" not in str(row["name"]).upper() or int(row["memory_mib"]) < 80_000 for row in selected):
        raise RuntimeError(f"Experiment-49 requires H100 80GB GPUs: {selected}")
    return selected


def freeze_or_validate_data_inventory(args: argparse.Namespace) -> Path:
    data_dir = (args.official_repo / "data" / "fineweb10B").resolve()
    train_paths = [
        data_dir / f"fineweb_train_{index:06d}.bin"
        for index in range(1, EXPECTED_TRAIN_SHARDS + 1)
    ]
    validation_path = data_dir / "fineweb_val_000000.bin"
    missing = [path.name for path in [*train_paths, validation_path] if not path.is_file()]
    if missing:
        raise RuntimeError(f"frozen Experiment-49 data files are missing: {missing[:5]}")
    print(
        "EX49 hashing the frozen R1 FineWeb inventory "
        "(train 000001--000050 plus validation 000000).",
        flush=True,
    )
    records: list[dict[str, object]] = []
    for index, path in enumerate(train_paths, start=1):
        records.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        if index % 10 == 0:
            print(f"EX49 data hash progress: {index}/{EXPECTED_TRAIN_SHARDS}", flush=True)
    validation = {
        "name": validation_path.name,
        "bytes": validation_path.stat().st_size,
        "sha256": sha256_file(validation_path),
    }
    payload = {
        "schema_version": 2,
        "status": "passed",
        "selection_policy": "exact_train_000001_through_000050_and_val_000000",
        "data_dir": str(data_dir),
        "ordered_train_shards": records,
        "validation_shard": validation,
        "selected_total_bytes": sum(int(record["bytes"]) for record in records)
        + int(validation["bytes"]),
        "extra_train_shards_are_ignored": True,
    }
    certificate = args.run_dir / "frozen_data_inventory.json"
    if certificate.is_file():
        accepted = read_json(certificate)
        if accepted != payload:
            accepted_records = {
                str(record.get("name")): record
                for record in accepted.get("ordered_train_shards", [])
                if isinstance(record, dict)
            }
            changed = [
                record["name"]
                for record in records
                if accepted_records.get(str(record["name"])) != record
            ]
            if accepted.get("validation_shard") != validation:
                changed.append(validation_path.name)
            raise RuntimeError(
                "frozen Experiment-49 FineWeb content changed; start no training and inspect: "
                + ", ".join(str(name) for name in changed[:8])
            )
    else:
        write_json(certificate, payload)
    return certificate


def acquire_gpu_locks(args: argparse.Namespace) -> list[Any]:
    if os.name != "posix":
        return []
    import fcntl

    host = "".join(
        character if character.isalnum() or character in "-._" else "_"
        for character in socket.gethostname()
    )
    # Match the project-wide 43/44 convention: locks are shared by every
    # experiment under one result root, but namespaced by host so GPU 0 on two
    # independent machines does not collide on the shared filesystem.
    lock_dir = args.run_dir.parent.parent / ".physical_gpu_locks" / host
    lock_dir.mkdir(parents=True, exist_ok=True)
    handles: list[Any] = []
    try:
        for gpu in args.gpus:
            handle = (lock_dir / f"gpu_{gpu}.lock").open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise RuntimeError(f"physical GPU {gpu} lock is busy") from exc
            handle.seek(0)
            handle.truncate()
            handle.write(
                json.dumps(
                    {
                        "experiment": 49,
                        "host": socket.gethostname(),
                        "gpu": gpu,
                        "pid": os.getpid(),
                        "run_dir": str(args.run_dir),
                        "acquired_at": datetime.now().astimezone().isoformat(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
            handles.append(handle)
    except Exception:
        for handle in handles:
            handle.close()
        raise
    return handles


def suite_plan_payload(args: argparse.Namespace) -> dict[str, object]:
    code_files = (
        "malt_optimizer.py",
        "malt_source_builder.py",
        "run_r1_malt.py",
        "run_r1_malt_suite.py",
        "analyze_malt_pilot.py",
        "analyze_malt_formal.py",
        "malt_contract.json",
        "PAPER_DERIVATION.md",
        "historical_control_snapshot/r1_unified_eight_method_run_summary.csv",
        "historical_control_snapshot/analysis_manifest.json",
    )
    command_wrapper = (
        SCRIPT_DIR.parent.parent
        / "commands"
        / "49_r1_malt_strong_baseline"
        / "20260807_ex49_r1_malt_strong_baseline.sh"
    )
    return {
        "schema_version": 3,
        "family": runner.FAMILY,
        "protocols": {
            "preflight": runner.PREFLIGHT_PROTOCOL,
            "pilot": runner.PILOT_PROTOCOL,
            "selection": runner.SELECTION_PROTOCOL,
            "formal_smoke": runner.SMOKE_PROTOCOL,
            "formal": runner.FORMAL_PROTOCOL,
            "formal_analysis": "malt_r1_ten_method_analysis_v4",
        },
        "pilot_cells": [asdict(cell) for cell in runner.PILOT_CELLS],
        "formal_methods": list(FORMAL_METHODS),
        "formal_seeds": list(FORMAL_SEEDS),
        "repo": str(args.repo),
        "official_repo": str(args.official_repo),
        # Preserve the requested venv entrypoints exactly.  Path.resolve() follows
        # the common venv/bin/python symlink to /usr/bin/python and would erase the
        # controller/training-environment distinction from the frozen plan.
        "controller_python": os.path.abspath(os.path.expanduser(sys.executable)),
        "training_python": os.path.abspath(os.path.expanduser(args.training_python)),
        "gpus": list(args.gpus),
        "gpu_inventory": requested_gpu_inventory(list(args.gpus)),
        "wandb_mode": args.wandb_mode,
        "wandb_entity": args.wandb_entity,
        "wandb_base_url": os.environ.get("WANDB_BASE_URL", "default"),
        "code_sha256": {name: sha256_file(SCRIPT_DIR / name) for name in code_files},
        "shared_r0_controller": {
            "path": str(runner.R0_DIR / "run_official_newton_muon_r0.py"),
            "sha256": sha256_file(
                runner.R0_DIR / "run_official_newton_muon_r0.py"
            ),
        },
        "command_wrapper": {
            "path": str(command_wrapper),
            "sha256": sha256_file(command_wrapper),
        },
    }


def freeze_or_validate_suite_plan(args: argparse.Namespace) -> None:
    path = args.run_dir / "suite_plan.json"
    current = suite_plan_payload(args)
    if path.is_file():
        if read_json(path) != current:
            raise RuntimeError("suite plan/code/runtime/GPU/W&B contract changed; start a new run")
    else:
        if any(args.run_dir.iterdir()):
            raise RuntimeError("nonempty Experiment-49 run lacks suite_plan.json")
        write_json(path, current)


def freeze_or_validate_historical_inputs(args: argparse.Namespace) -> dict[str, object]:
    if args.experiment45_summary is None or args.experiment45_analysis_manifest is None:
        raise RuntimeError("Experiment-45 historical inputs were not provided")
    summary = args.experiment45_summary.expanduser().resolve()
    manifest = args.experiment45_analysis_manifest.expanduser().resolve()
    if not summary.is_file() or not manifest.is_file():
        raise RuntimeError("accepted Experiment-45 historical inputs are missing")
    payload = {
        "summary": str(summary),
        "summary_sha256": sha256_file(summary),
        "analysis_manifest": str(manifest),
        "analysis_manifest_sha256": sha256_file(manifest),
    }
    if payload["summary_sha256"] != EX45_SUMMARY_SHA256:
        raise RuntimeError("Experiment-45 eight-method summary SHA-256 is not authoritative")
    if payload["analysis_manifest_sha256"] != EX45_MANIFEST_SHA256:
        raise RuntimeError("Experiment-45 analysis manifest SHA-256 is not authoritative")
    path = args.run_dir / "historical_control_inputs.json"
    if path.is_file() and read_json(path) != payload:
        raise RuntimeError("Experiment-45 historical control inputs changed within the suite")
    write_json(path, payload)
    return payload


def mode_manifest_name(mode: str) -> str:
    return f"{mode}_manifest.json"


def accepted_batch(
    stage_dir: Path,
    mode: str,
    *,
    require_wandb: bool,
    expected_wandb_mode: str | None = None,
) -> Path | None:
    accepted: list[Path] = []
    for manifest_path in sorted(stage_dir.glob(f"*/{mode_manifest_name(mode)}")):
        payload = read_json(manifest_path)
        if payload.get("status") not in ACCEPTED:
            continue
        if payload.get("failures") not in (None, []):
            continue
        summaries = payload.get("summaries")
        if not isinstance(summaries, list) or len(summaries) != 1:
            continue
        if summaries[0].get("evidence_valid") is not True:
            continue
        if expected_wandb_mode is not None:
            if payload.get("wandb_mode") != expected_wandb_mode:
                continue
            expected_status = {
                "online": "uploaded_online",
                "offline": "saved_offline",
                "disabled": "disabled",
            }[expected_wandb_mode]
            if summaries[0].get("wandb_status") != expected_status:
                continue
        if require_wandb and payload.get("wandb_complete") is not True:
            continue
        accepted.append(manifest_path.parent)
    if len(accepted) > 1:
        raise RuntimeError(f"multiple accepted {mode} batches under {stage_dir}: {accepted}")
    return accepted[0] if accepted else None


def resumable_batch(stage_dir: Path, mode: str) -> Path | None:
    candidates = sorted(
        (path.parent for path in stage_dir.glob(f"*/{mode}_plan.json")),
        key=lambda path: path.name,
    )
    if len(candidates) > 1:
        raise RuntimeError(f"multiple incomplete {mode} batches under {stage_dir}: {candidates}")
    return candidates[0] if candidates else None


def runner_command(
    args: argparse.Namespace,
    *,
    mode: str,
    stage_dir: Path,
    seed: int,
    cell_id: str | None = None,
    selection: Path | None = None,
    selected_method: str | None = None,
    smoke_manifest: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_r1_malt.py"),
        "--official-repo",
        str(args.official_repo),
        "--python-exe",
        args.training_python,
        "--data-inventory-certificate",
        str(args.run_dir / "frozen_data_inventory.json"),
        "--seed",
        str(seed),
        "--results-dir",
        str(stage_dir),
        "--wandb-mode",
        "disabled" if mode == "formal_smoke" else args.wandb_mode,
    ]
    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    flag = {
        "preflight": "--preflight",
        "pilot": "--pilot",
        "formal_smoke": "--formal-smoke",
        "formal": "--formal",
    }[mode]
    command.append(flag)
    if cell_id is not None:
        command.extend(["--cells", cell_id])
    if selection is not None:
        command.extend(["--selection-certificate", str(selection)])
    if selected_method is not None:
        command.extend(["--selected-method", selected_method])
    if smoke_manifest is not None:
        command.extend(["--smoke-manifest", str(smoke_manifest)])
    accepted = accepted_batch(
        stage_dir,
        mode,
        require_wandb=mode in {"pilot", "formal"} and args.wandb_mode != "disabled",
        expected_wandb_mode="disabled" if mode == "formal_smoke" else args.wandb_mode,
    )
    if accepted is not None:
        return []
    previous = resumable_batch(stage_dir, mode)
    if previous is not None:
        command.extend(["--resume-batch", str(previous)])
    return command


def terminate_job_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def execute_jobs(args: argparse.Namespace, jobs: list[dict[str, object]]) -> None:
    pending = [job for job in jobs if job["command"]]
    if not pending:
        return
    logs = args.run_dir / "controller_logs"
    logs.mkdir(parents=True, exist_ok=True)
    active: dict[subprocess.Popen[str], tuple[dict[str, object], Any]] = {}
    available = list(args.gpus)
    recoverable_upload_failures: list[str] = []
    try:
        while pending or active:
            while pending and available:
                job = pending.pop(0)
                gpu = available.pop(0)
                label = str(job["label"])
                command = [str(value) for value in job["command"]]
                safe_label = label.replace("/", "__")
                attempt = 1 + len(list(logs.glob(f"{safe_label}__attempt_*.log")))
                log_path = logs / f"{safe_label}__attempt_{attempt:03d}.log"
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                record = {
                    "label": label,
                    "attempt": attempt,
                    "gpu": gpu,
                    "command": command,
                    "command_text": shlex.join(command),
                    "started_at": datetime.now().astimezone().isoformat(),
                    "log": str(log_path),
                }
                append_command(args.run_dir, record)
                handle = log_path.open("x", encoding="utf-8", buffering=1)
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=args.repo,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=os.name == "posix",
                    )
                except BaseException:
                    handle.close()
                    raise
                active[process] = (record, handle)
                print(f"started {label} on physical GPU {gpu}; log={log_path}", flush=True)
            finished = [process for process in active if process.poll() is not None]
            if not finished:
                time.sleep(1.0)
                continue
            for process in finished:
                record, handle = active.pop(process)
                handle.close()
                available.append(str(record["gpu"]))
                available.sort(key=args.gpus.index)
                completed = {
                    **record,
                    "returncode": process.returncode,
                    "finished_at": datetime.now().astimezone().isoformat(),
                }
                append_command(args.run_dir, completed)
                if process.returncode == 3:
                    recoverable_upload_failures.append(str(record["label"]))
                    print(
                        f"local evidence passed but upload is incomplete for {record['label']}; other jobs continue",
                        flush=True,
                    )
                    continue
                if process.returncode != 0:
                    raise RuntimeError(
                        f"{record['label']} failed rc={process.returncode}; see {record['log']}"
                    )
                print(f"passed {record['label']}", flush=True)
        if recoverable_upload_failures:
            raise RuntimeError(
                "local evidence is valid but W&B upload must be resumed for: "
                + ", ".join(recoverable_upload_failures)
            )
    finally:
        for process, (_, handle) in list(active.items()):
            terminate_job_process(process)
            handle.close()
        active.clear()


def run_preflight(args: argparse.Namespace) -> None:
    marker = args.run_dir / "preflight" / "preflight_passed.json"
    if marker.is_file():
        marker_payload = read_json(marker)
        artifact = Path(str(marker_payload.get("artifact", "")))
        if (
            marker_payload.get("status") == "passed"
            and artifact.is_file()
            and sha256_file(artifact) == marker_payload.get("sha256")
        ):
            print("skip accepted preflight")
            return
        raise RuntimeError("preflight marker exists but its artifact certificate is invalid")
    result_dir = args.run_dir / "preflight" / "runner_artifacts"
    accepted_existing = [
        path
        for path in sorted(result_dir.glob("*_preflight_seed2026.json"))
        if read_json(path).get("status") == "passed"
    ]
    if len(accepted_existing) > 1:
        raise RuntimeError(f"multiple accepted preflight artifacts: {accepted_existing}")
    if len(accepted_existing) == 1:
        write_json(
            marker,
            {
                "status": "passed",
                "artifact": str(accepted_existing[0]),
                "sha256": sha256_file(accepted_existing[0]),
            },
        )
        print("recovered accepted preflight artifact")
        return
    command = runner_command(args, mode="preflight", stage_dir=result_dir, seed=2026)
    # Preflight artifacts are not batch manifests, so the command is never empty.
    log_path = args.run_dir / "controller_logs" / "preflight.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus[0]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    append_command(args.run_dir, {"label": "preflight", "gpu": args.gpus[0], "command": command})
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=args.repo, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    artifacts = [
        path
        for path in sorted(result_dir.glob("*_preflight_seed2026.json"))
        if read_json(path).get("status") == "passed"
    ]
    if completed.returncode != 0 or len(artifacts) != 1:
        raise RuntimeError(f"Experiment-49 preflight failed; inspect {log_path}")
    write_json(marker, {"status": "passed", "artifact": str(artifacts[0]), "sha256": sha256_file(artifacts[0])})


def aggregate_pilot(args: argparse.Namespace) -> Path:
    pilot_root = args.run_dir / "pilot"
    summaries: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    source_audits: list[dict[str, object]] = []
    runtime_fingerprints: list[dict[str, object]] = []
    exact_runtime_contracts: list[dict[str, object]] = []
    data_inventories: list[dict[str, object]] = []
    init_hashes: list[str] = []
    for cell in runner.PILOT_CELLS:
        batch = accepted_batch(
            pilot_root / cell.cell_id,
            "pilot",
            require_wandb=args.wandb_mode != "disabled",
            expected_wandb_mode=args.wandb_mode,
        )
        if batch is None:
            raise RuntimeError(f"accepted pilot cell missing: {cell.cell_id}")
        manifest_path = batch / "pilot_manifest.json"
        manifest = read_json(manifest_path)
        summary = manifest["summaries"][0]
        expected = {
            "family": runner.FAMILY,
            "protocol": runner.PILOT_PROTOCOL,
            "seed": 2026,
            "total_steps": runner.PILOT_STEPS,
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches or summary.get("cell_id") != cell.cell_id:
            raise RuntimeError(f"pilot cell lineage mismatch for {cell.cell_id}: {mismatches}")
        source_audit = manifest.get("source_audit")
        runtime = manifest.get("training_runtime_fingerprint")
        exact_runtime = manifest.get("exact_runtime_contract")
        data_inventory = manifest.get("data_inventory")
        init_audit = manifest.get("initialization_audit", {})
        if (
            not isinstance(source_audit, dict)
            or not isinstance(runtime, dict)
            or not isinstance(exact_runtime, dict)
            or exact_runtime.get("status") != "passed"
            or not isinstance(data_inventory, dict)
            or data_inventory.get("status") != "passed"
        ):
            raise RuntimeError(f"pilot source/runtime/data audit missing for {cell.cell_id}")
        init_hash = str(init_audit.get("init_sha256", ""))
        if not init_hash or summary.get("init_sha256") != init_hash:
            raise RuntimeError(f"pilot initialization lineage failed for {cell.cell_id}")
        source_audits.append(source_audit)
        runtime_fingerprints.append(runtime)
        exact_runtime_contracts.append(exact_runtime)
        data_inventories.append(data_inventory)
        init_hashes.append(init_hash)
        summaries.append(summary)
        manifests.append({"cell_id": cell.cell_id, "path": str(manifest_path), "sha256": sha256_file(manifest_path)})
    summaries.sort(key=lambda row: str(row["cell_id"]))
    if len({canonical_json(value) for value in source_audits}) != 1:
        raise RuntimeError("the twelve V4 pilot cells do not share one frozen source audit")
    if len({canonical_json(value) for value in runtime_fingerprints}) != 1:
        raise RuntimeError("the twelve V4 pilot cells do not share one runtime fingerprint")
    if len({canonical_json(value) for value in exact_runtime_contracts}) != 1:
        raise RuntimeError("the twelve V4 pilot cells do not share one exact runtime contract")
    if len({canonical_json(value) for value in data_inventories}) != 1:
        raise RuntimeError("the twelve V4 pilot cells do not share one frozen data inventory")
    if len(set(init_hashes)) != 1:
        raise RuntimeError("the twelve V4 pilot cells do not share one seed-2026 initialization")
    aggregate_dir = pilot_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = aggregate_dir / "pilot_manifest.json"
    csv_path = aggregate_dir / "pilot_summary.csv"
    payload = {
        "family": runner.FAMILY,
        "protocol": runner.PILOT_PROTOCOL,
        "status": "completed_valid",
        "seed": 2026,
        "total_steps": runner.PILOT_STEPS,
        "total_tokens": runner.PILOT_STEPS * runner.TOKENS_PER_STEP,
        "grid_design": "fresh_v4_focused_malt_upper_grid_dual_method",
        "malt_execution_order": [
            cell.matrix_lr for cell in runner.PILOT_CELLS if cell.method == "malt"
        ],
        "malter_execution_order": [
            cell.matrix_lr for cell in runner.PILOT_CELLS if cell.method == "malter_eq17"
        ],
        "failures": [],
        "summaries": summaries,
        "source_manifests": manifests,
        "source_audit": source_audits[0],
        "training_runtime_fingerprint": runtime_fingerprints[0],
        "exact_runtime_contract": exact_runtime_contracts[0],
        "data_inventory": data_inventories[0],
        "init_sha256": init_hashes[0],
    }
    write_json(manifest_path, payload)
    write_csv(csv_path, summaries)
    runner.make_selection(aggregate_dir, summaries, manifest_path)
    analysis_dir = accepted_analysis_dir(pilot_root, "pilot_analysis_manifest.json")
    if analysis_dir is None:
        analysis_dir = next_analysis_dir(pilot_root)
        command = [
            sys.executable,
            str(SCRIPT_DIR / "analyze_malt_pilot.py"),
            str(aggregate_dir),
            "--output-dir",
            str(analysis_dir),
        ]
        subprocess.run(command, cwd=args.repo, check=True)
    analysis_manifest = analysis_dir / "pilot_analysis_manifest.json"
    if not validate_analysis_bundle(analysis_manifest):
        raise RuntimeError(f"pilot analysis integrity failed: {analysis_manifest}")
    selection = analysis_dir / "pilot_selection_verified.json"
    if not selection.is_file():
        raise RuntimeError("pilot analyzer did not emit its selection certificate")
    return selection


def run_pilot(args: argparse.Namespace) -> Path:
    run_preflight(args)
    jobs: list[dict[str, object]] = []
    for cell in runner.PILOT_CELLS:
        stage_dir = args.run_dir / "pilot" / cell.cell_id
        command = runner_command(
            args,
            mode="pilot",
            stage_dir=stage_dir,
            seed=2026,
            cell_id=cell.cell_id,
        )
        jobs.append({"label": f"pilot/{cell.cell_id}", "command": command})
    execute_jobs(args, jobs)
    return aggregate_pilot(args)


def validated_selection(args: argparse.Namespace) -> Path:
    analysis_dir = accepted_analysis_dir(
        args.run_dir / "pilot", "pilot_analysis_manifest.json"
    )
    selection = (
        analysis_dir / "pilot_selection_verified.json"
        if analysis_dir is not None
        else aggregate_pilot(args)
    )
    payload = read_json(selection)
    if payload.get("status") != "selected" or payload.get("formal_allowed") is not True:
        raise RuntimeError(
            f"formal is blocked by pilot outcome {payload.get('status')!r}; inspect {selection}"
        )
    for method in FORMAL_METHODS:
        runner.validate_selection(selection, method)
    return selection


def run_formal(args: argparse.Namespace) -> list[tuple[str, int, Path, Path]]:
    selection = validated_selection(args)
    smoke_jobs: list[dict[str, object]] = []
    for method in FORMAL_METHODS:
        for seed in FORMAL_SEEDS:
            stage_dir = args.run_dir / "formal_smoke" / method / f"seed{seed}"
            smoke_jobs.append(
                {
                    "label": f"formal_smoke/{method}/seed{seed}",
                    "command": runner_command(
                        args,
                        mode="formal_smoke",
                        stage_dir=stage_dir,
                        seed=seed,
                        selection=selection,
                        selected_method=method,
                    ),
                }
            )
    execute_jobs(args, smoke_jobs)

    formal_jobs: list[dict[str, object]] = []
    for method in FORMAL_METHODS:
        for seed in FORMAL_SEEDS:
            smoke_batch = accepted_batch(
                args.run_dir / "formal_smoke" / method / f"seed{seed}",
                "formal_smoke",
                require_wandb=False,
                expected_wandb_mode="disabled",
            )
            if smoke_batch is None:
                raise RuntimeError(
                    f"accepted exact-shape smoke missing for {method}/seed{seed}"
                )
            smoke_manifest = smoke_batch / "formal_smoke_manifest.json"
            stage_dir = args.run_dir / "formal" / method / f"seed{seed}"
            formal_jobs.append(
                {
                    "label": f"formal/{method}/seed{seed}",
                    "command": runner_command(
                        args,
                        mode="formal",
                        stage_dir=stage_dir,
                        seed=seed,
                        selection=selection,
                        selected_method=method,
                        smoke_manifest=smoke_manifest,
                    ),
                }
            )
    execute_jobs(args, formal_jobs)

    accepted: list[tuple[str, int, Path, Path]] = []
    for method in FORMAL_METHODS:
        for seed in FORMAL_SEEDS:
            batch = accepted_batch(
                args.run_dir / "formal" / method / f"seed{seed}",
                "formal",
                require_wandb=args.wandb_mode != "disabled",
                expected_wandb_mode=args.wandb_mode,
            )
            if batch is None:
                raise RuntimeError(f"accepted formal batch missing for {method}/seed{seed}")
            manifest = batch / "formal_manifest.json"
            payload = read_json(manifest)
            summary_path = batch / "formal_summary.csv"
            summary = payload["summaries"][0]
            if (
                not summary_path.is_file()
                or summary.get("controlled_seed") != seed
                or summary.get("method") != method
            ):
                raise RuntimeError(f"formal summary lineage failed for {method}/seed{seed}")
            accepted.append((method, seed, summary_path, manifest))
    return accepted


def verify_and_analyze(args: argparse.Namespace) -> Path:
    selection_path = validated_selection(args)
    selection_payload = read_json(selection_path)
    selection_sha256 = sha256_file(selection_path)
    accepted: list[dict[str, object]] = []
    method_specs = {
        "malt": {
            "adaptation_label": "MALT-R1 adaptation",
            "hidden_state_bytes": runner.MALT_EXPECTED_HIDDEN_STATE_BYTES,
            "malt_nu_bytes": 0,
        },
        "malter_eq17": {
            "adaptation_label": "MALTER-Eq17-R1 adaptation",
            "hidden_state_bytes": runner.MALTER_EXPECTED_HIDDEN_STATE_BYTES,
            "malt_nu_bytes": 288,
        },
    }
    for method in FORMAL_METHODS:
        selected = selection_payload["selections"][method]
        for seed in FORMAL_SEEDS:
            batch = accepted_batch(
                args.run_dir / "formal" / method / f"seed{seed}",
                "formal",
                require_wandb=args.wandb_mode != "disabled",
                expected_wandb_mode=args.wandb_mode,
            )
            if batch is None:
                raise RuntimeError(f"formal {method}/seed{seed} is not accepted")
            summary_path = batch / "formal_summary.csv"
            if not summary_path.is_file():
                raise RuntimeError(f"formal {method}/seed{seed} summary is missing")
            manifest_path = batch / "formal_manifest.json"
            payload = read_json(manifest_path)
            embedded = payload.get("summary", {})
            selection_record = payload.get("selection_certificate", {})
            spec = method_specs[method]
            roles = embedded.get("state_schema", {}).get("roles", {})
            expected_roles = {
                "malt_momentum": 48,
                "malt_row_ema": 72,
                "malt_col_ema": 72,
                "malt_last_alpha_min": 48,
                "malt_last_alpha_max": 48,
            }
            if method == "malter_eq17":
                expected_roles["malt_nu"] = 72
            state_schema = embedded.get("state_schema", {})
            checks = {
                "family": payload.get("family") == runner.FAMILY,
                "protocol": payload.get("protocol") == runner.FORMAL_PROTOCOL,
                "seed": payload.get("seed") == seed and embedded.get("controlled_seed") == seed,
                "steps": payload.get("total_steps") == runner.FULL_STEPS and embedded.get("total_steps") == runner.FULL_STEPS,
                "method": embedded.get("method") == method,
                "cell": embedded.get("cell_id") == selected.get("selected_cell_id"),
                "lr": float(embedded.get("matrix_lr", -1)) == float(selected.get("selected_matrix_lr", -2)),
                "adaptation": embedded.get("adaptation_label") == spec["adaptation_label"],
                "state_bytes": int(embedded.get("hidden_optimizer_state_bytes", -1)) == spec["hidden_state_bytes"],
                "nu_bytes": int(embedded.get("malt_nu_bytes", -1)) == spec["malt_nu_bytes"],
                "state_roles": roles == expected_roles,
                "no_activation_k": state_schema.get("contains_activation_k_state") is False,
                "optimizer_steps": state_schema.get("optimizer_group_steps") == [runner.FULL_STEPS],
                "state_numerics": state_schema.get("numerical_checks_passed") is True,
                "selection_path": selection_record.get("path") == str(selection_path.resolve()),
                "selection_sha": selection_record.get("sha256") == selection_sha256,
                "selection_method": selection_record.get("validated_selected_method") == method,
            }
            if not all(checks.values()):
                raise RuntimeError(
                    f"formal lineage/state audit failed for {method}/seed{seed}: {checks}"
                )
            checkpoint = Path(str(embedded.get("checkpoint_path", "")))
            if (
                not checkpoint.is_file()
                or checkpoint.stat().st_size != int(embedded.get("checkpoint_bytes", -1))
                or sha256_file(checkpoint) != embedded.get("checkpoint_sha256")
            ):
                raise RuntimeError(
                    f"formal {method}/seed{seed} checkpoint certificate failed"
                )
            accepted.append(
                {
                    "method": method,
                    "adaptation_label": spec["adaptation_label"],
                    "selected_cell_id": selected["selected_cell_id"],
                    "selected_matrix_lr": selected["selected_matrix_lr"],
                    "seed": seed,
                    "summary": summary_path,
                    "manifest": manifest_path,
                    "checkpoint": str(checkpoint),
                    "checkpoint_bytes": embedded["checkpoint_bytes"],
                    "checkpoint_sha256": embedded["checkpoint_sha256"],
                }
            )
    output_dir = accepted_analysis_dir(args.run_dir, "analysis_manifest.json")
    if output_dir is None:
        output_dir = next_analysis_dir(args.run_dir)
        malt_units = [record for record in accepted if record["method"] == "malt"]
        malter_units = [record for record in accepted if record["method"] == "malter_eq17"]
        command = [
            sys.executable,
            str(SCRIPT_DIR / "analyze_malt_formal.py"),
            "--malt-summaries",
            *[str(item["summary"]) for item in malt_units],
            "--malt-manifests",
            *[str(item["manifest"]) for item in malt_units],
            "--malter-summaries",
            *[str(item["summary"]) for item in malter_units],
            "--malter-manifests",
            *[str(item["manifest"]) for item in malter_units],
            "--selection-certificate",
            str(selection_path),
            "--experiment45-summary",
            str(args.experiment45_summary),
            "--experiment45-analysis-manifest",
            str(args.experiment45_analysis_manifest),
            "--output-dir",
            str(output_dir),
        ]
        subprocess.run(command, cwd=args.repo, check=True)
    manifest_path = output_dir / "analysis_manifest.json"
    if not validate_analysis_bundle(manifest_path):
        raise RuntimeError(f"Experiment-49 formal analysis failed: {manifest_path}")
    final = {
        "status": "completed",
        "experiment": 49,
        "family": runner.FAMILY,
        "run_dir": str(args.run_dir),
        "suite_plan": str(args.run_dir / "suite_plan.json"),
        "suite_plan_sha256": sha256_file(args.run_dir / "suite_plan.json"),
        "pilot_selection": str(selection_path),
        "pilot_selection_sha256": selection_sha256,
        "formal_methods": list(FORMAL_METHODS),
        "n_formal_units": len(accepted),
        "formal_units": [
            {
                **{key: value for key, value in record.items() if key not in {"summary", "manifest"}},
                "summary": str(record["summary"]),
                "summary_sha256": sha256_file(record["summary"]),
                "manifest": str(record["manifest"]),
                "manifest_sha256": sha256_file(record["manifest"]),
            }
            for record in accepted
        ],
        "analysis_manifest": str(manifest_path),
        "analysis_manifest_sha256": sha256_file(manifest_path),
        "data_inventory": {
            "path": str(args.run_dir / "frozen_data_inventory.json"),
            "sha256": sha256_file(args.run_dir / "frozen_data_inventory.json"),
        },
        "exact_runtime_contract": read_json(accepted[0]["manifest"])["exact_runtime_contract"],
        "claim_labels": ["MALT-R1 adaptation", "MALTER-Eq17-R1 adaptation"],
        "official_reproduction": False,
        "malter_formula_choice": "Equation (17), exactly one outer eta",
        "timing_eligible": False,
        "contract": str(SCRIPT_DIR / "malt_contract.json"),
        "contract_sha256": sha256_file(SCRIPT_DIR / "malt_contract.json"),
    }
    handoff = args.run_dir / "handoff_manifest.json"
    write_json(handoff, final)
    return handoff


def install_shutdown_handlers() -> None:
    if os.name != "posix":
        return

    def stop_controller(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"controller received signal {signum}")

    signal.signal(signal.SIGTERM, stop_controller)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, stop_controller)


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    args.official_repo = args.official_repo.expanduser().resolve()
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.resume:
        raise RuntimeError(f"run directory is nonempty; pass --resume: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    install_shutdown_handlers()
    locks = [] if args.stage == "verify" else acquire_gpu_locks(args)
    try:
        freeze_or_validate_suite_plan(args)
        freeze_or_validate_data_inventory(args)
        if args.stage in {"verify", "all"}:
            freeze_or_validate_historical_inputs(args)
        if args.stage == "preflight":
            run_preflight(args)
        elif args.stage == "pilot":
            selection = run_pilot(args)
            print(f"EX49_SELECTION={selection}")
        elif args.stage == "formal":
            run_formal(args)
        elif args.stage == "verify":
            print(f"EX49_HANDOFF={verify_and_analyze(args)}")
        else:
            run_pilot(args)
            run_formal(args)
            print(f"EX49_HANDOFF={verify_and_analyze(args)}")
        print(f"EX49_ARTIFACTS={args.run_dir}")
    finally:
        for handle in locks:
            handle.close()


if __name__ == "__main__":
    main()
