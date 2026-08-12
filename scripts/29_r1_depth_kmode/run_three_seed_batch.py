#!/usr/bin/env python3
"""Run the complete three-seed R1 depth experiment, sharded across GPUs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_r1_depth_kmode.py"
METHOD_SHARDS = (
    (
        "early_none",
        "early_diag",
        "late_none",
        "late_diag",
        "edge_none",
        "edge_diag",
    ),
    (
        "center_none",
        "center_diag",
        "all_none",
        "all_diag",
        "block4",
        "muon",
    ),
)


def scheduled_jobs(
    seeds: list[int], devices: list[str]
) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for seed_index, seed in enumerate(seeds):
        for shard_index, methods in enumerate(METHOD_SHARDS):
            # Cross over physical devices between seeds. Every none/diag pair
            # remains within one shard/GPU, while neither shard is completely
            # confounded with a single physical H100.
            device = devices[(seed_index + shard_index) % len(devices)]
            jobs.append(
                {
                    "seed": seed,
                    "shard": shard_index,
                    "device": device,
                    "methods": list(methods),
                    "status": "pending",
                }
            )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-command three-seed R1 depth batch. Each seed is split into two "
            "six-method shards so two H100s remain balanced."
        )
    )
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument("--devices", nargs="+", default=["0", "1"])
    parser.add_argument("--smoke-steps", type=int, default=34)
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--wandb-project",
        default="Selective-Newton-Muon-MainConf-R1-Depth-KMode-20260725",
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--run-prefix", default="mainconf_r1_depth_kmode")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_state(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def controller_runtime_preflight(
    training_python: str,
    wandb_mode: str,
) -> dict[str, object]:
    """Fail before any smoke work if the controller/runtime split is invalid."""

    # Do not call Path.resolve() here. Virtual-environment launchers commonly
    # symlink to the same system Python binary, while sys.prefix/package state
    # remains correctly isolated by the distinct venv entry paths.
    controller = Path(os.path.abspath(os.path.expanduser(sys.executable)))
    training = Path(os.path.abspath(os.path.expanduser(training_python)))
    if not training.is_file():
        raise RuntimeError(f"training Python does not exist: {training}")
    if controller == training:
        raise RuntimeError(
            "controller Python and training Python must be separate for the accepted "
            "R1 runtime contract; set SNM_CONTROLLER_PYTHON to the controller "
            "environment and pass SNM_TRAINING_PYTHON via --python-exe"
        )

    payload: dict[str, object] = {
        "controller_python": str(controller),
        "training_python": str(training),
        "interpreters_separate": True,
        "wandb_mode": wandb_mode,
    }
    try:
        wandb = importlib.import_module("wandb")
    except Exception as exc:
        raise RuntimeError(
            "controller Python cannot import wandb before the batch starts: "
            f"controller={controller}, error={exc!r}. Use a controller "
            "environment with wandb installed."
        ) from exc
    payload["wandb_version"] = str(getattr(wandb, "__version__", "unknown"))
    return payload


def command_for(
    args: argparse.Namespace,
    seed: int,
    methods: tuple[str, ...],
    results_dir: Path,
    *,
    smoke: bool,
    smoke_manifest: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--official-repo",
        str(args.official_repo),
        "--python-exe",
        args.python_exe,
        "--seed",
        str(seed),
        "--methods",
        *methods,
        "--results-dir",
        str(results_dir),
        "--run-prefix",
        args.run_prefix,
        "--wandb-project",
        args.wandb_project,
        "--continue-on-error",
    ]
    if smoke:
        command.extend(
            [
                "--numerical-smoke",
                "--smoke-steps",
                str(args.smoke_steps),
                "--wandb-mode",
                "disabled",
            ]
        )
    else:
        assert smoke_manifest is not None
        command.extend(
            [
                "--smoke-manifest",
                str(smoke_manifest),
                "--wandb-mode",
                args.wandb_mode,
            ]
        )
        if args.wandb_entity:
            command.extend(["--wandb-entity", args.wandb_entity])
    return command


def new_manifest(
    results_dir: Path,
    seed: int,
    batch_kind: str,
    before: set[Path],
) -> Path:
    after = set(results_dir.glob(f"*_{batch_kind}_seed{seed}/r1_manifest.json"))
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if len(created) != 1:
        raise RuntimeError(
            f"expected one new {batch_kind} manifest for seed{seed}, observed {created}"
        )
    payload = json.loads(created[0].read_text(encoding="utf-8"))
    expected_status = (
        "completed_valid_smoke" if batch_kind == "smoke" else "completed_valid"
    )
    if payload.get("status") != expected_status:
        raise RuntimeError(
            f"{created[0]} status={payload.get('status')!r}, expected={expected_status!r}"
        )
    return created[0]


def run_phase(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
) -> None:
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("\nCOMMAND " + subprocess.list2cmdline(command) + "\n")
        result = subprocess.run(
            command,
            cwd=SCRIPT_DIR.parent.parent,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def main() -> None:
    args = parse_args()
    if not args.devices:
        raise ValueError("--devices must not be empty")
    if len(args.devices) != len(set(args.devices)):
        raise ValueError("--devices contains duplicates")
    if args.smoke_steps < 34:
        raise ValueError("--smoke-steps must be at least 34 to cross refresh step 32")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds contains duplicates")

    controller_runtime = (
        {
            "controller_python": str(Path(sys.executable).resolve()),
            "training_python": str(Path(args.python_exe).expanduser()),
            "dry_run_not_validated": True,
            "wandb_mode": args.wandb_mode,
        }
        if args.dry_run
        else controller_runtime_preflight(args.python_exe, args.wandb_mode)
    )
    print(
        "R1 depth controller preflight: "
        f"controller={controller_runtime['controller_python']} "
        f"training={controller_runtime['training_python']} "
        f"wandb={controller_runtime.get('wandb_version', 'not_checked_for_dry_run')}",
        flush=True,
    )

    args.results_dir = args.results_dir.expanduser().resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    batch_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    batch_dir = args.results_dir.parent / "batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    jobs = scheduled_jobs(args.seeds, args.devices)
    state: dict[str, object] = {
        "batch_id": batch_id,
        "status": "dry_run" if args.dry_run else "running",
        "seeds": args.seeds,
        "devices": args.devices,
        "method_shards": [list(methods) for methods in METHOD_SHARDS],
        "jobs": jobs,
        "checkpoint_policy": "disabled",
        "timing_usable": False,
        "controller_runtime": controller_runtime,
    }
    state_path = batch_dir / "batch_state.json"
    write_state(state_path, state)
    print(f"R1 depth batch state: {state_path}", flush=True)
    lock = threading.Lock()
    failures: list[dict[str, object]] = []

    job_queues: dict[str, list[dict[str, object]]] = {
        device: [] for device in args.devices
    }
    for job in jobs:
        job_queues[str(job["device"])].append(job)

    def worker(device: str) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        for job in job_queues[device]:
            seed = int(job["seed"])
            shard = int(job["shard"])
            methods = METHOD_SHARDS[shard]
            log_path = batch_dir / f"seed{seed}_shard{shard}_gpu{device}.log"
            # Each concurrently running shard owns its result namespace. This
            # prevents two same-seed smoke runs from discovering one another's
            # manifest and makes retries unambiguous.
            job_results_dir = args.results_dir / f"seed{seed}_shard{shard}"
            job_results_dir.mkdir(parents=True, exist_ok=True)
            smoke_command = command_for(
                args, seed, methods, job_results_dir, smoke=True
            )
            if args.dry_run:
                with lock:
                    job["status"] = "dry_run"
                    job["results_dir"] = str(job_results_dir)
                    job["smoke_command"] = smoke_command
                    job["formal_command_template"] = command_for(
                        args,
                        seed,
                        methods,
                        job_results_dir,
                        smoke=False,
                        smoke_manifest=Path("<SMOKE_MANIFEST>"),
                    )
                    write_state(state_path, state)
                continue
            try:
                with lock:
                    job["status"] = "smoke_running"
                    job["log"] = str(log_path)
                    job["results_dir"] = str(job_results_dir)
                    write_state(state_path, state)
                    print(
                        f"START seed={seed} shard={shard} gpu={device} "
                        f"methods={','.join(methods)}",
                        flush=True,
                    )
                before_smoke = set(
                    job_results_dir.glob(f"*_smoke_seed{seed}/r1_manifest.json")
                )
                run_phase(smoke_command, env=env, log_path=log_path)
                smoke_manifest = new_manifest(
                    job_results_dir, seed, "smoke", before_smoke
                )
                formal_command = command_for(
                    args,
                    seed,
                    methods,
                    job_results_dir,
                    smoke=False,
                    smoke_manifest=smoke_manifest,
                )
                with lock:
                    job["status"] = "formal_running"
                    job["smoke_manifest"] = str(smoke_manifest)
                    write_state(state_path, state)
                before_formal = set(
                    job_results_dir.glob(f"*_formal_seed{seed}/r1_manifest.json")
                )
                run_phase(formal_command, env=env, log_path=log_path)
                formal_manifest = new_manifest(
                    job_results_dir, seed, "formal", before_formal
                )
                with lock:
                    job["status"] = "completed"
                    job["formal_manifest"] = str(formal_manifest)
                    write_state(state_path, state)
                    print(
                        f"DONE seed={seed} shard={shard} gpu={device}",
                        flush=True,
                    )
            except BaseException as exc:
                with lock:
                    job["status"] = "failed"
                    job["error"] = repr(exc)
                    failures.append(
                        {
                            "seed": seed,
                            "shard": shard,
                            "device": device,
                            "error": repr(exc),
                        }
                    )
                    write_state(state_path, state)
                    print(
                        f"FAILED seed={seed} shard={shard} gpu={device}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                return

    threads = [
        threading.Thread(target=worker, args=(device,), name=f"gpu-{device}")
        for device in args.devices
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    state["status"] = (
        "dry_run"
        if args.dry_run
        else "failed"
        if failures
        else "completed"
    )
    state["failures"] = failures
    write_state(state_path, state)
    print(f"R1 depth batch state: {state_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
