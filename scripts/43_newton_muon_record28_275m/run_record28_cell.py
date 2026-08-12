#!/usr/bin/env python3
"""Run, validate, and optionally upload one experiment-43 quality cell."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import record28_common as common


SCRIPT_VERSION = "2026-07-31.1"
CPROJ_MODES = {
    "muon": "not_applicable",
    "original_newton_muon": "block4",
    "selective_none": "none",
    "selective_diag": "diag",
}
CPROJ_SCHEMA_EXPECTED = {
    "muon": {
        "count": 0,
        "kind": [],
        "shape": None,
        "cov_bytes": 0,
        "inv_bytes": 0,
        "workspace_bytes": 0,
        "activation_stat_bytes": 0,
        "activation_workspace_bytes": 0,
    },
    "original_newton_muon": {
        "count": 12,
        "kind": ["c_proj"],
        "shape": [4, 768, 768],
        "cov_bytes": 113_246_208,
        "inv_bytes": 113_246_208,
        "workspace_bytes": 226_492_416,
        "activation_stat_bytes": 113_246_256,
        "activation_workspace_bytes": 113_246_208,
    },
    "selective_none": {
        "count": 0,
        "kind": [],
        "shape": None,
        "cov_bytes": 0,
        "inv_bytes": 0,
        "workspace_bytes": 0,
        "activation_stat_bytes": 0,
        "activation_workspace_bytes": 0,
    },
    "selective_diag": {
        "count": 12,
        "kind": ["c_proj_diag"],
        "shape": [4, 768],
        "cov_bytes": 147_456,
        "inv_bytes": 147_456,
        "workspace_bytes": 113_246_208,
        "activation_stat_bytes": 147_504,
        "activation_workspace_bytes": 0,
    },
}
EXPECTED_FORMAL_STEPS = [*range(0, 1651, 50), 1695]
ACTIVE_TRAINING_PROCESS: subprocess.Popen[str] | None = None
INTERRUPTED_SIGNAL: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", choices=common.METHODS, required=True)
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--training-python", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--data-certificate", type=Path, required=True)
    parser.add_argument("--data-repo-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb-project", default="selective-newton-muon")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--upload-only", action="store_true")
    return parser.parse_args()


def terminate_process_group(process: subprocess.Popen[str], grace_seconds: float = 20) -> None:
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


def signal_handler(signum: int, _frame: Any) -> None:
    global INTERRUPTED_SIGNAL
    INTERRUPTED_SIGNAL = signum
    if ACTIVE_TRAINING_PROCESS is not None:
        terminate_process_group(ACTIVE_TRAINING_PROCESS)


def install_signal_handlers() -> None:
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal_handler)


def runtime_probe(training_python: Path, physical_gpu: str) -> dict[str, Any]:
    script = r"""
import json, os, platform
import numpy, torch, triton
payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "triton": triton.__version__,
    "numpy": numpy.__version__,
    "compiled_autograd": torch._dynamo.config.compiled_autograd,
    "cuda_available": torch.cuda.is_available(),
    "visible_device_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    payload.update({
        "logical_device": 0,
        "gpu_name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    })
print(json.dumps(payload, sort_keys=True))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = physical_gpu
    completed = subprocess.run(
        [str(training_python), "-c", script],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    expected = {
        "torch": "2.8.0+cu126",
        "torch_cuda": "12.6",
        "triton": "3.4.0",
        "cuda_available": True,
        "visible_device_count": 1,
    }
    checks = {key: payload.get(key) == value for key, value in expected.items()}
    checks["compiled_autograd_default_false"] = (
        payload.get("compiled_autograd") is False
    )
    checks["h100"] = "H100" in str(payload.get("gpu_name"))
    checks["capability"] = payload.get("capability") == [9, 0]
    if not all(checks.values()):
        raise RuntimeError(
            f"training-runtime drift: checks={checks}, observed={payload}"
        )
    return {
        "schema_version": 1,
        "passed": True,
        "training_python": str(training_python),
        "physical_gpu_request": physical_gpu,
        "payload": payload,
        "checks": checks,
        "audited_at": common.utc_now(),
    }


def expected_protocol(stage: str) -> dict[str, Any]:
    if stage == "formal":
        return {
            "iterations": 1695,
            "val_loss_every": 50,
            "val_tokens": 10_485_760,
            "validation_steps": EXPECTED_FORMAL_STEPS,
            "save_checkpoint": True,
            "train_tokens": 666_501_120,
        }
    return {
        "iterations": 18,
        "val_loss_every": 18,
        "val_tokens": 262_144,
        "validation_steps": [0, 18],
        "save_checkpoint": False,
        "train_tokens": 18 * 393_216,
    }


def compute_curve_summary(
    validations: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    losses = [float(row["val_loss"]) for row in validations]
    steps = [int(row["step"]) for row in validations]
    total_steps = int(protocol["iterations"])
    area = 0.0
    for left, right in zip(validations, validations[1:]):
        width = int(right["step"]) - int(left["step"])
        area += width * (float(left["val_loss"]) + float(right["val_loss"])) / 2
    normalized_auc = area / total_steps
    target = 3.30
    target_step: float | None = None
    first_observed_target_step: int | None = None
    for index, row in enumerate(validations):
        if float(row["val_loss"]) <= target:
            first_observed_target_step = int(row["step"])
            if index == 0:
                target_step = float(row["step"])
            else:
                left = validations[index - 1]
                x0, y0 = float(left["step"]), float(left["val_loss"])
                x1, y1 = float(row["step"]), float(row["val_loss"])
                if math.isclose(y0, y1):
                    target_step = x1
                else:
                    fraction = (target - y0) / (y1 - y0)
                    target_step = x0 + max(0.0, min(1.0, fraction)) * (x1 - x0)
            break
    tail = losses[-min(5, len(losses)) :]
    final = validations[-1]
    return {
        "final_val_loss": losses[-1],
        "best_val_loss": min(losses),
        "tail5_mean": statistics.fmean(tail),
        "normalized_auc": normalized_auc,
        "steps_to_target": target_step,
        "tokens_to_target": (
            target_step * 393_216 if target_step is not None else None
        ),
        "first_observed_step_at_or_below_target": first_observed_target_step,
        "common_target_loss": target,
        "final_step": int(final["step"]),
        "train_tokens": int(protocol["train_tokens"]),
        "tokens_per_update": 393_216,
        "diagnostic_train_time_ms": int(final["train_time_ms"]),
        "timing_eligible": False,
    }


def upload_wandb(args: argparse.Namespace) -> bool:
    manifest_path = args.attempt_dir / "scientific_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("upload-only requires a completed scientific manifest")
    manifest = common.read_json(manifest_path)
    if manifest.get("passed") is not True:
        raise RuntimeError("refusing W&B upload for an invalid scientific cell")
    wandb_path = args.attempt_dir / "wandb.json"
    if args.stage == "smoke":
        common.atomic_write_json(
            wandb_path,
            {
                "schema_version": 1,
                "status": "not_required_smoke",
                "complete": True,
                "mode": args.wandb_mode,
                "required_for_paper_handoff": False,
                "updated_at": common.utc_now(),
            },
        )
        return True
    if args.wandb_mode == "disabled":
        common.atomic_write_json(
            wandb_path,
            {
                "schema_version": 1,
                "status": "disabled_formal_upload_pending",
                "complete": False,
                "mode": "disabled",
                "required_for_paper_handoff": True,
                "scientific_manifest_sha256": common.sha256_file(
                    manifest_path
                ),
                "updated_at": common.utc_now(),
            },
        )
        return False
    contract_sha256 = manifest["contract_sha256"]
    run_id = common.stable_wandb_id(
        args.stage, args.seed, args.method, contract_sha256
    )
    status = {
        "schema_version": 1,
        "status": "uploading",
        "complete": False,
        "mode": args.wandb_mode,
        "run_id": run_id,
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "updated_at": common.utc_now(),
    }
    common.atomic_write_json(wandb_path, status)
    try:
        import wandb

        (args.attempt_dir / "wandb_local").mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=run_id,
            resume="allow",
            name=f"record28_{args.method}_seed{args.seed}",
            mode=args.wandb_mode,
            dir=str(args.attempt_dir / "wandb_local"),
            config={
                "experiment": 43,
                "recipe": "Newton-Muon-2 upstream near Record #28",
                "stage": args.stage,
                "seed": args.seed,
                "method": args.method,
                "cproj_k_mode": CPROJ_MODES[args.method],
                "contract_sha256": contract_sha256,
                "source_snapshot_sha256": manifest["source_snapshot_sha256"],
                "derived_source_sha256": manifest["derived_source_sha256"],
                "data_fingerprint_sha256": manifest["data_fingerprint_sha256"],
                "init_sha256": manifest["init_sha256"],
                "parameter_count": 275_743_572,
                "train_tokens": manifest["train_tokens"],
                "timing_eligible": False,
            },
            tags=["experiment43", "record28-near", args.method, f"seed{args.seed}"],
        )
        with (args.attempt_dir / "metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                run.log(
                    {
                        "validation/loss": float(row["val_loss"]),
                        "train/tokens": int(float(row["tokens"])),
                        "diagnostic/train_time_ms": int(float(row["train_time_ms"])),
                    },
                    step=int(row["step"]),
                )
        summary = common.read_json(args.attempt_dir / "summary.json")
        for key, value in summary.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                run.summary[key] = value
        run.summary["scientific_manifest_sha256"] = common.sha256_file(
            manifest_path
        )
        run.finish()
        if args.wandb_mode == "offline":
            status.update(
                {
                    "status": "offline_created_pending_sync",
                    "complete": False,
                    "required_for_paper_handoff": True,
                    "scientific_manifest_sha256": common.sha256_file(
                        manifest_path
                    ),
                    "updated_at": common.utc_now(),
                }
            )
            common.atomic_write_json(wandb_path, status)
            return False
        status.update(
            {
                "status": "completed",
                "complete": True,
                "required_for_paper_handoff": True,
                "url": getattr(run, "url", None),
                "scientific_manifest_sha256": common.sha256_file(
                    manifest_path
                ),
                "updated_at": common.utc_now(),
            }
        )
        common.atomic_write_json(wandb_path, status)
        return True
    except Exception as error:  # network/upload failure must never retrain
        status.update(
            {
                "status": "pending_retry",
                "complete": False,
                "error": repr(error),
                "updated_at": common.utc_now(),
            }
        )
        common.atomic_write_json(wandb_path, status)
        return False


def run_training(args: argparse.Namespace) -> None:
    global ACTIVE_TRAINING_PROCESS
    protocol = expected_protocol(args.stage)
    args.attempt_dir.mkdir(parents=True, exist_ok=True)
    contract = common.read_json(args.contract)
    data_certificate = common.read_json(args.data_certificate)
    if data_certificate.get("passed") is not True:
        raise RuntimeError("data certificate did not pass")
    source_sha256 = common.sha256_file(args.training_source)
    snapshot_sha256 = common.sha256_file(args.source_snapshot_manifest)
    contract_sha256 = common.sha256_file(args.contract)

    gpus = common.query_gpus()
    selected_gpu = common.resolve_gpu(gpus, args.physical_gpu)
    lock_path = common.shared_gpu_lock_path(args.result_root, args.physical_gpu)
    with common.exclusive_file_lock(
        lock_path,
        timeout_seconds=5,
        metadata={
            "experiment": 43,
            "cell": common.cell_key(args.stage, args.seed, args.method),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "physical_gpu": selected_gpu,
            "acquired_at": common.utc_now(),
        },
    ):
        active = [
            row
            for row in common.query_compute_processes()
            if row["gpu_uuid"] == selected_gpu["uuid"]
        ]
        if active:
            raise RuntimeError(
                f"physical GPU {args.physical_gpu} is not idle: {active}"
            )
        runtime = runtime_probe(args.training_python, args.physical_gpu)
        runtime["physical_gpu"] = selected_gpu
        runtime["all_visible_gpus_before_launch"] = gpus
        common.atomic_write_json(args.attempt_dir / "runtime.json", runtime)

        command = [
            str(args.training_python),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            str(args.training_source),
        ]
        env = os.environ.copy()
        # Upstream interprets any nonempty DISABLE_FP8 string (including "0")
        # as disabling FP8, so remove inherited values for this frozen recipe.
        env.pop("DISABLE_FP8", None)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": args.physical_gpu,
                "DATA_PATH": str(args.data_repo_root),
                "PYTHONHASHSEED": str(args.seed),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "RECORD28_METHOD": args.method,
                "RECORD28_CPROJ_K_MODE": CPROJ_MODES[args.method],
                "RECORD28_SEED": str(args.seed),
                "RECORD28_STAGE": args.stage,
                "RECORD28_OUTPUT_DIR": str(args.attempt_dir),
                "RECORD28_CELL_ID": (
                    f"exp43_{args.stage}_{args.method}_seed{args.seed}"
                ),
                "RECORD28_NUM_ITERATIONS": str(protocol["iterations"]),
                "RECORD28_VAL_LOSS_EVERY": str(protocol["val_loss_every"]),
                "RECORD28_VAL_TOKENS": str(protocol["val_tokens"]),
                "RECORD28_SAVE_CHECKPOINT": (
                    "1" if protocol["save_checkpoint"] else "0"
                ),
            }
        )
        runtime_cache_root = args.attempt_dir / "runtime_cache"
        env.update(
            {
                "TORCHINDUCTOR_CACHE_DIR": str(
                    runtime_cache_root / "torchinductor"
                ),
                "TRITON_CACHE_DIR": str(runtime_cache_root / "triton"),
                "CUDA_CACHE_PATH": str(runtime_cache_root / "cuda"),
            }
        )
        command_record = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "command": command,
            "command_text": shlex.join(command),
            "cwd": str(args.attempt_dir),
            "environment": {
                key: env[key]
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "DATA_PATH",
                    "PYTHONHASHSEED",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "RECORD28_METHOD",
                    "RECORD28_CPROJ_K_MODE",
                    "RECORD28_SEED",
                    "RECORD28_STAGE",
                    "RECORD28_OUTPUT_DIR",
                    "RECORD28_CELL_ID",
                    "RECORD28_NUM_ITERATIONS",
                    "RECORD28_VAL_LOSS_EVERY",
                    "RECORD28_VAL_TOKENS",
                    "RECORD28_SAVE_CHECKPOINT",
                    "TORCHINDUCTOR_CACHE_DIR",
                    "TRITON_CACHE_DIR",
                    "CUDA_CACHE_PATH",
                )
            },
            "physical_gpu": selected_gpu,
            "timing_eligible": False,
            "started_at": common.utc_now(),
        }
        common.atomic_write_json(args.attempt_dir / "command.json", command_record)
        common.atomic_write_json(
            args.attempt_dir / "status.json",
            {
                "status": "running",
                "script_version": SCRIPT_VERSION,
                "pid": os.getpid(),
                "cell_key": common.cell_key(args.stage, args.seed, args.method),
                "started_at": command_record["started_at"],
            },
        )
        stdout_path = args.attempt_dir / "stdout.log"
        with stdout_path.open("w", encoding="utf-8", buffering=1) as output:
            ACTIVE_TRAINING_PROCESS = subprocess.Popen(
                command,
                cwd=args.attempt_dir,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            command_record["training_pid"] = ACTIVE_TRAINING_PROCESS.pid
            command_record["training_pgid"] = (
                ACTIVE_TRAINING_PROCESS.pid if os.name != "nt" else None
            )
            common.atomic_write_json(
                args.attempt_dir / "command.json", command_record
            )
            return_code = ACTIVE_TRAINING_PROCESS.wait()
            ACTIVE_TRAINING_PROCESS = None
        if INTERRUPTED_SIGNAL is not None:
            common.atomic_write_json(
                args.attempt_dir / "status.json",
                {
                    "status": "interrupted",
                    "signal": INTERRUPTED_SIGNAL,
                    "return_code": return_code,
                    "restart_policy": "new attempt from step 0",
                    "updated_at": common.utc_now(),
                },
            )
            raise KeyboardInterrupt
        if return_code != 0:
            common.atomic_write_json(
                args.attempt_dir / "status.json",
                {
                    "status": "training_failed",
                    "return_code": return_code,
                    "restart_policy": "new attempt from step 0",
                    "stdout": str(stdout_path),
                    "updated_at": common.utc_now(),
                },
            )
            raise RuntimeError(
                f"training returned {return_code}; inspect {stdout_path}"
            )
        resolved_cache = runtime_cache_root.resolve()
        resolved_attempt = args.attempt_dir.resolve()
        if resolved_cache.parent != resolved_attempt:
            raise RuntimeError(f"unsafe runtime cache path: {resolved_cache}")
        if resolved_cache.exists():
            shutil.rmtree(resolved_cache)

    parsed = common.parse_training_log(args.attempt_dir / "training.log")
    validations = parsed["validations"]
    metadata = parsed["metadata"]
    final_audit = parsed["final_audit"]
    warmup_reset = parsed["warmup_reset"]
    validation_steps = [row["step"] for row in validations]
    checks = {
        "validation_steps": validation_steps == protocol["validation_steps"],
        "metadata_present": isinstance(metadata, dict),
        "final_audit_present": isinstance(final_audit, dict),
        "warmup_reset_present": isinstance(warmup_reset, dict),
        "warmup_model_restored": warmup_reset
        and warmup_reset.get("model_matches_initial") is True,
        "warmup_optimizer_restored": warmup_reset
        and warmup_reset.get("optimizer_matches_initial") is True,
        "warmup_preconditioner_reset": warmup_reset
        and warmup_reset.get("preconditioner_step_zero") is True,
        "warmup_activation_statistics_reset": warmup_reset
        and warmup_reset.get("activation_accumulators_zero") is True,
        "warmup_gradients_clear": warmup_reset
        and warmup_reset.get("gradients_clear") is True,
        "method": metadata and metadata.get("method") == args.method,
        "cproj_k_mode": metadata
        and metadata.get("cproj_k_mode") == CPROJ_MODES[args.method],
        "seed": metadata and metadata.get("seed") == args.seed,
        "stage": metadata and metadata.get("stage") == args.stage,
        "parameter_count": metadata
        and metadata.get("parameter_count") == 275_743_572,
        "tokens_per_update": metadata
        and metadata.get("tokens_per_update") == 393_216,
        "world_size": metadata and metadata.get("world_size") == 1,
        "formal_schedule_denominator": metadata
        and metadata.get("schedule_iterations") == 1695,
        "losses_finite": all(
            math.isfinite(float(row["val_loss"])) for row in validations
        ),
        "parameters_finite": final_audit
        and final_audit.get("all_finite") is True,
        "k_tensors_finite": final_audit
        and final_audit.get("k_tensors_all_finite") is True,
        "preconditioner_refresh_crossed": (
            args.method == "muon"
            or (
                final_audit
                and int(final_audit.get("preconditioner_step", -1))
                == protocol["iterations"]
            )
        ),
        "refresh_count": (
            bool(final_audit)
            and int(final_audit.get("refresh_count", -1))
            == protocol["iterations"] // 16
            if args.method != "muon"
            else bool(final_audit)
            and int(final_audit.get("refresh_count", -1)) == 0
        ),
        "checkpoint_policy": (
            (args.attempt_dir / "state_step001695.pt").is_file()
            if args.stage == "formal"
            else not any(args.attempt_dir.glob("state_step*.pt"))
        ),
    }
    cproj_expected = CPROJ_SCHEMA_EXPECTED[args.method]
    checks["cproj_k_parameter_count"] = (
        final_audit
        and final_audit.get("cproj_k_parameter_count") == cproj_expected["count"]
    )
    checks["cproj_k_kind"] = (
        final_audit and final_audit.get("cproj_k_kind") == cproj_expected["kind"]
    )
    for field in (
        "cov_bytes",
        "inv_bytes",
        "workspace_bytes",
        "activation_stat_bytes",
        "activation_workspace_bytes",
    ):
        checks[f"cproj_{field}"] = (
            final_audit
            and final_audit.get(f"cproj_{field}") == cproj_expected[field]
        )
    if cproj_expected["shape"] is None:
        checks["cproj_cov_shapes"] = (
            final_audit and final_audit.get("cproj_cov_shapes") == []
        )
        checks["cproj_inv_shapes"] = (
            final_audit and final_audit.get("cproj_inv_shapes") == []
        )
        checks["cproj_activation_state_absent"] = (
            final_audit
            and final_audit.get("cproj_activation_stat_bytes") == 0
            and final_audit.get("cproj_activation_workspace_bytes") == 0
        )
    else:
        expected_shapes = [cproj_expected["shape"]] * 12
        checks["cproj_cov_shapes"] = (
            final_audit
            and final_audit.get("cproj_cov_shapes") == expected_shapes
        )
        checks["cproj_inv_shapes"] = (
            final_audit
            and final_audit.get("cproj_inv_shapes") == expected_shapes
        )
        checks["cproj_activation_state_present"] = (
            final_audit
            and final_audit.get("cproj_activation_stat_bytes", 0) > 0
        )
    if args.method == "muon":
        checks["muon_all_k_state_absent"] = (
            final_audit
            and final_audit.get("k_state_bytes") == 0
            and final_audit.get("k_workspace_bytes") == 0
            and final_audit.get("activation_stat_bytes") == 0
            and final_audit.get("activation_workspace_bytes") == 0
            and final_audit.get("total_preconditioner_bytes") == 0
        )
    if not all(checks.values()):
        common.atomic_write_json(args.attempt_dir / "checks.json", checks)
        raise RuntimeError(f"scientific cell checks failed: {checks}")
    common.atomic_write_json(args.attempt_dir / "checks.json", checks)

    metric_rows = [
        {
            **row,
            "tokens": int(row["step"]) * 393_216,
        }
        for row in validations
    ]
    common.write_csv(
        args.attempt_dir / "metrics.csv",
        metric_rows,
        (
            "step",
            "total_steps",
            "val_loss",
            "train_time_ms",
            "step_avg_ms",
            "tokens",
        ),
    )
    summary = compute_curve_summary(validations, protocol)
    summary.update(
        {
            "schema_version": 1,
            "stage": args.stage,
            "seed": args.seed,
            "method": args.method,
            "init_sha256": metadata["init_sha256"],
            "parameter_structure_sha256": metadata[
                "parameter_structure_sha256"
            ],
            "peak_memory_allocated_bytes": final_audit[
                "peak_memory_allocated_bytes"
            ],
            "peak_memory_reserved_bytes": final_audit[
                "peak_memory_reserved_bytes"
            ],
            "k_state_bytes": final_audit["k_state_bytes"],
            "optimizer_state_bytes": final_audit["optimizer_state_bytes"],
            "optimizer_runtime_cache_bytes": final_audit[
                "optimizer_runtime_cache_bytes"
            ],
            "optimizer_runtime_total_bytes": final_audit[
                "optimizer_runtime_total_bytes"
            ],
            "total_preconditioner_bytes": final_audit[
                "total_preconditioner_bytes"
            ],
            "cproj_cov_bytes": final_audit["cproj_cov_bytes"],
            "cproj_inv_bytes": final_audit["cproj_inv_bytes"],
            "cproj_workspace_bytes": final_audit["cproj_workspace_bytes"],
            "cproj_activation_stat_bytes": final_audit[
                "cproj_activation_stat_bytes"
            ],
            "cproj_activation_workspace_bytes": final_audit[
                "cproj_activation_workspace_bytes"
            ],
        }
    )
    common.atomic_write_json(args.attempt_dir / "summary.json", summary)

    checkpoint_record: dict[str, Any] | None = None
    if args.stage == "formal":
        checkpoint = args.attempt_dir / "state_step001695.pt"
        checkpoint_record = {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": common.sha256_file(checkpoint),
            "step": 1695,
            "checkpoint_scope": "model_only",
            "remote_retention_only": True,
            "handoff_transfer_required": False,
        }
        common.atomic_write_json(
            args.attempt_dir / "checkpoint_hash.json", checkpoint_record
        )

    required_artifacts = [
        "checks.json",
        "command.json",
        "metrics.csv",
        "runtime.json",
        "stdout.log",
        "summary.json",
        "training.log",
    ]
    if checkpoint_record is not None:
        required_artifacts.append("checkpoint_hash.json")
    hashes = common.artifact_hashes(args.attempt_dir, required_artifacts)
    common.atomic_write_json(args.attempt_dir / "artifact_hashes.json", hashes)
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "status": "scientifically_complete",
        "passed": True,
        "stage": args.stage,
        "seed": args.seed,
        "method": args.method,
        "cproj_k_mode": CPROJ_MODES[args.method],
        "cell_key": common.cell_key(args.stage, args.seed, args.method),
        "contract_sha256": contract_sha256,
        "source_snapshot_sha256": snapshot_sha256,
        "derived_source_sha256": source_sha256,
        "data_fingerprint_sha256": data_certificate["fingerprint_sha256"],
        "init_sha256": metadata["init_sha256"],
        "parameter_structure_sha256": metadata["parameter_structure_sha256"],
        "parameter_count": metadata["parameter_count"],
        "total_steps": protocol["iterations"],
        "train_tokens": protocol["train_tokens"],
        "tokens_per_update": 393_216,
        "timing_eligible": False,
        "timing_exclusion_reason": (
            "quality runs may share the host with experiment 44; "
            "use experiments 39/42 for isolated efficiency"
        ),
        "physical_gpu": runtime["physical_gpu"],
        "assigned_physical_gpu_request": args.physical_gpu,
        "training_python": str(args.training_python),
        "training_source": str(args.training_source),
        "contract": str(args.contract),
        "source_snapshot_manifest": str(args.source_snapshot_manifest),
        "data_certificate": str(args.data_certificate),
        "checkpoint": checkpoint_record,
        "artifacts": required_artifacts + ["artifact_hashes.json"],
        "artifact_hashes": hashes,
        "completed_at": common.utc_now(),
    }
    common.atomic_write_json(
        args.attempt_dir / "scientific_manifest.json", manifest
    )
    wandb_required = args.stage == "formal"
    common.atomic_write_json(
        args.attempt_dir / "wandb.json",
        {
            "schema_version": 1,
            "status": (
                "pending_upload"
                if wandb_required and args.wandb_mode != "disabled"
                else (
                    "disabled_formal_upload_pending"
                    if wandb_required
                    else "not_required_smoke"
                )
            ),
            "complete": not wandb_required,
            "mode": args.wandb_mode,
            "required_for_paper_handoff": wandb_required,
            "scientific_manifest_sha256": common.sha256_file(
                args.attempt_dir / "scientific_manifest.json"
            ),
            "updated_at": common.utc_now(),
        },
    )
    common.atomic_write_json(
        args.attempt_dir / "status.json",
        {
            "status": "scientifically_complete",
            "passed": True,
            "wandb_pending": wandb_required,
            "scientific_manifest": str(
                args.attempt_dir / "scientific_manifest.json"
            ),
            "completed_at": common.utc_now(),
        },
    )


def main() -> None:
    args = parse_args()
    install_signal_handlers()
    for field in (
        "attempt_dir",
        "training_source",
        "contract",
        "source_snapshot_manifest",
        "data_certificate",
        "data_repo_root",
        "result_root",
    ):
        value = getattr(args, field)
        setattr(args, field, value.expanduser().resolve())
    # Preserve the venv entrypoint; resolving a symlink can silently turn it
    # into /usr/bin/python and lose the pinned Torch environment.
    args.training_python = args.training_python.expanduser().absolute()
    if args.seed not in common.SEEDS:
        raise RuntimeError(f"formal experiment-43 seed is not frozen: {args.seed}")
    if args.upload_only:
        complete = upload_wandb(args)
    else:
        # Scientific acceptance is deliberately completed before any network
        # operation.  The suite seals the accepted pointer and runs the
        # analysis before invoking this entry point again with --upload-only.
        run_training(args)
        complete = args.stage == "smoke"
    print(
        f"RECORD28_CELL scientific_manifest="
        f"{args.attempt_dir / 'scientific_manifest.json'}"
    )
    if args.upload_only and not complete:
        print(
            "RECORD28_WANDB_PENDING upload-only recovery is required; "
            "scientific training will not be repeated",
            file=sys.stderr,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
