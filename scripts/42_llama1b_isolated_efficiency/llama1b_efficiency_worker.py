#!/usr/bin/env python3
"""Run and validate one isolated LLaMA-1B efficiency cell."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import efficiency_common as common


SCRIPT_VERSION = "2026-07-29.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-tier", choices=("smoke", "formal"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--position-index", type=int, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--derived-base", type=Path, required=True)
    parser.add_argument("--profile-wrapper", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--physical-gpu", default="0")
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--execution-domain", required=True)
    return parser.parse_args()


def tier_configuration(
    contract: dict[str, Any], tier: str
) -> dict[str, int | float]:
    frozen = contract["frozen_configuration"]
    if tier == "formal":
        return {
            "num_iterations": int(frozen["total_updates"]),
            "global_batch_size": int(frozen["global_batch_size"]),
            "device_batch_size": int(frozen["device_batch_size"]),
            "sequence_length": int(frozen["sequence_length"]),
            "val_every": int(frozen["validation_every"]),
            "val_tokens": int(frozen["validation_tokens"]),
            "warmdown_iters": int(frozen["warmdown_updates"]),
            "backup_lr": float(frozen["backup_lr"]),
            "matrix_lr": float(frozen["matrix_lr"]),
            "adamw_matrix_lr": float(frozen["adamw_matrix_lr"]),
            "checkpoint_every": 0,
            "steady_steps": int(frozen["timed_updates"]),
        }
    # Smoke reaches the first Newton refresh at update 32 and proves that the
    # post-warmup measurement boundary contains exactly two updates.
    return {
        "num_iterations": 34,
        "global_batch_size": int(frozen["global_batch_size"]),
        "device_batch_size": int(frozen["device_batch_size"]),
        "sequence_length": int(frozen["sequence_length"]),
        "val_every": 34,
        "val_tokens": int(frozen["device_batch_size"])
        * int(frozen["sequence_length"]),
        "warmdown_iters": 0,
        "backup_lr": float(frozen["backup_lr"]),
        "matrix_lr": float(frozen["matrix_lr"]),
        "adamw_matrix_lr": float(frozen["adamw_matrix_lr"]),
        "checkpoint_every": 0,
        "steady_steps": 2,
    }


def build_training_command(
    args: argparse.Namespace, config: dict[str, int | float], trainer_dir: Path
) -> list[str]:
    command = [
        str(common.lexical_absolute(args.python_exe)),
        str(args.profile_wrapper.resolve()),
        "--method",
        args.method,
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output-dir",
        str(trainer_dir.resolve()),
        "--seed",
        "2026",
        "--num-iterations",
        str(config["num_iterations"]),
        "--global-batch-size",
        str(config["global_batch_size"]),
        "--device-batch-size",
        str(config["device_batch_size"]),
        "--sequence-length",
        str(config["sequence_length"]),
        "--val-every",
        str(config["val_every"]),
        "--val-tokens",
        str(config["val_tokens"]),
        "--warmdown-iters",
        str(config["warmdown_iters"]),
        "--backup-lr",
        str(config["backup_lr"]),
        "--matrix-lr",
        str(config["matrix_lr"]),
        "--adamw-matrix-lr",
        str(config["adamw_matrix_lr"]),
        "--checkpoint-every",
        "0",
        "--resume",
        "never",
        "--no-save-final",
    ]
    return command


def tee_process(command: list[str], env: dict[str, str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            if line.startswith("step:") and " train_loss:" in line:
                try:
                    step = int(line.split(":", 2)[1].split("/", 1)[0])
                except (IndexError, ValueError):
                    step = -1
                if step > 0 and step % 64 == 0:
                    print(f"MECH-42 worker progress step={step}", flush=True)
        return process.wait()


def validate_metrics(
    path: Path,
    *,
    total_steps: int,
    timed_steps: int,
    tokens_per_update: int,
    steady_train_s: float,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row.get("event") == "train"]
    validation = [row for row in rows if row.get("event") == "val"]
    other = [row for row in rows if row.get("event") not in ("train", "val")]
    failures: list[str] = []
    train_steps = [int(row["step"]) for row in train]
    if train_steps != list(range(1, total_steps + 1)):
        failures.append("training steps are missing, duplicated, or out of order")
    validation_steps = [int(row["step"]) for row in validation]
    if validation_steps != [0, total_steps]:
        failures.append(f"validation steps differ: {validation_steps}")
    if other:
        failures.append("unknown metric event")
    prior_train_s = -1.0
    by_step: dict[int, float] = {}
    for row in rows:
        step = int(row["step"])
        loss = float(row["loss"])
        train_s = float(row["train_s"])
        tokens = int(row["tokens_seen"])
        if not math.isfinite(loss) or not math.isfinite(train_s):
            failures.append(f"non-finite metric at {row.get('event')} step {step}")
        if train_s + 1e-9 < prior_train_s:
            failures.append("cumulative training time decreased")
        prior_train_s = train_s
        if tokens != step * tokens_per_update:
            failures.append(f"token count mismatch at step {step}")
        if row.get("event") == "train":
            steady = float(row["steady_train_s"])
            if not math.isfinite(steady):
                failures.append(f"non-finite steady time at step {step}")
            by_step[step] = steady
    expected_timed = list(range(33, total_steps + 1))
    if len(expected_timed) != timed_steps:
        failures.append("timed-step arithmetic differs from the contract")
    if total_steps >= 32 and abs(by_step.get(32, math.nan)) > 1e-12:
        failures.append("steady timer was not zero through warmup update 32")
    timed_deltas: list[float] = []
    previous = by_step.get(32, 0.0)
    for step in expected_timed:
        current = by_step.get(step, math.nan)
        delta = current - previous
        if not math.isfinite(delta) or delta <= 0:
            failures.append(f"invalid timed interval at step {step}")
        timed_deltas.append(delta)
        previous = current
    if not math.isclose(
        sum(timed_deltas), steady_train_s, rel_tol=1e-8, abs_tol=1e-6
    ):
        failures.append("timed interval sum does not equal steady_train_s")
    if failures:
        raise RuntimeError("metric validation failed:\n- " + "\n- ".join(failures))
    return {
        "row_count": len(rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "timed_rows": len(timed_deltas),
        "timed_step_first": expected_timed[0],
        "timed_step_last": expected_timed[-1],
        "timed_interval_min_s": min(timed_deltas),
        "timed_interval_max_s": max(timed_deltas),
        "timed_interval_sum_s": sum(timed_deltas),
    }


def validate_summary(
    args: argparse.Namespace,
    contract: dict[str, Any],
    preflight: dict[str, Any],
    config: dict[str, int | float],
    trainer_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = trainer_dir / "summary.json"
    metrics_path = trainer_dir / "metrics.csv"
    if not summary_path.is_file() or not metrics_path.is_file():
        raise RuntimeError("trainer did not produce summary.json and metrics.csv")
    summary = common.read_json(summary_path)
    expected_steps = int(config["num_iterations"])
    expected_steady = int(config["steady_steps"])
    frozen = contract["frozen_configuration"]
    expected_tokens_per_update = int(frozen["tokens_per_update"])
    failures: list[str] = []
    exact = {
        "status": "completed",
        "method": args.method,
        "seed": int(frozen["seed"]),
        "completed_steps": expected_steps,
        "tokens_seen": expected_steps * expected_tokens_per_update,
        "init_sha256": preflight["init_audit"]["common_init_sha256"],
        "steady_steps": expected_steady,
        "resume_count": 0,
        "timing_comparable": True,
        "checkpoint_path": "",
        "peak_memory_stats_reset": True,
        "peak_reset_after_completed_step": 32,
        "timed_step_first": 33,
        "timed_step_last": expected_steps,
        "peak_measurement_scope": (
            "CUDA peak statistics reset after completed update 32; "
            "training updates 33 through the final update; final validation excluded"
        ),
        "timing_measurement_scope": (
            "CUDA-synchronized optimizer updates only; first 32 updates excluded"
        ),
    }
    for key, value in exact.items():
        if summary.get(key) != value:
            failures.append(f"{key}={summary.get(key)!r}, expected {value!r}")
    architecture = summary.get("architecture", {})
    if architecture.get("parameter_count") != int(frozen["parameter_count"]):
        failures.append("parameter count mismatch")
    if (
        architecture.get("base_trainer_sha256")
        != preflight["source_audit"]["derived_base_sha256"]
    ):
        failures.append("derived base trainer hash mismatch in architecture audit")
    summary_config = summary.get("config", {})
    config_pairs = {
        "num_iterations": expected_steps,
        "global_batch_size": int(config["global_batch_size"]),
        "device_batch_size": int(config["device_batch_size"]),
        "sequence_length": int(config["sequence_length"]),
        "val_every": int(config["val_every"]),
        "val_tokens": int(config["val_tokens"]),
        "warmdown_iters": int(config["warmdown_iters"]),
        "backup_lr": float(config["backup_lr"]),
        "matrix_lr": float(config["matrix_lr"]),
        "adamw_matrix_lr": float(config["adamw_matrix_lr"]),
        "checkpoint_every": 0,
        "resume": "never",
        "no_save_final": True,
    }
    for key, value in config_pairs.items():
        if summary_config.get(key) != value:
            failures.append(
                f"trainer config {key}={summary_config.get(key)!r}, expected {value!r}"
            )
    runtime = common.stable_runtime(summary.get("runtime", {}))
    runtime_fingerprint = common.canonical_json_sha256(runtime)
    if runtime_fingerprint != preflight["runtime_fingerprint"]:
        failures.append("stable runtime fingerprint differs from preflight")
    expected_k = int(contract["expected_k_state_bytes"][args.method])
    if summary.get("k_state_bytes") != expected_k:
        failures.append(
            f"K-state bytes={summary.get('k_state_bytes')} expected={expected_k}"
        )
    expected_state = contract["expected_state_bytes"]
    if summary.get("model_parameter_bytes") != int(
        expected_state["model_parameter_bytes"]
    ):
        failures.append("model parameter byte count mismatch")
    if summary.get("optimizer_state_bytes") != int(
        expected_state["optimizer_state_bytes"][args.method]
    ):
        failures.append("optimizer state byte count mismatch")
    finite_fields = (
        "final_val_loss",
        "best_val_loss",
        "final_train_loss",
        "train_s",
        "steady_train_s",
        "step_avg_ms",
        "timed_training_peak_allocated_bytes",
        "timed_training_peak_reserved_bytes",
        "allocated_bytes_at_timing_reset",
        "reserved_bytes_at_timing_reset",
        "allocated_bytes_at_timed_end",
        "reserved_bytes_at_timed_end",
    )
    for key in finite_fields:
        if not common.finite_number(summary.get(key)):
            failures.append(f"{key} is missing or non-finite")
    allocated_peak = int(summary.get("timed_training_peak_allocated_bytes", 0))
    reserved_peak = int(summary.get("timed_training_peak_reserved_bytes", 0))
    if allocated_peak <= 0 or reserved_peak < allocated_peak:
        failures.append("timed training CUDA peak relation is invalid")
    if int(summary.get("allocated_bytes_at_timing_reset", 0)) > allocated_peak:
        failures.append("allocated bytes at reset exceed timed peak")
    if int(summary.get("reserved_bytes_at_timing_reset", 0)) > reserved_peak:
        failures.append("reserved bytes at reset exceed timed peak")
    if int(summary.get("allocated_bytes_at_timed_end", 0)) > allocated_peak:
        failures.append("allocated bytes at timed end exceed timed peak")
    if int(summary.get("reserved_bytes_at_timed_end", 0)) > reserved_peak:
        failures.append("reserved bytes at timed end exceed timed peak")
    copied_base = trainer_dir / "train_llama_swiglu_base.py"
    if (
        not copied_base.is_file()
        or common.sha256_file(copied_base)
        != preflight["source_audit"]["derived_base_sha256"]
    ):
        failures.append("trainer-local source copy is missing or changed")
    if (trainer_dir / "checkpoint_latest.pt").exists():
        failures.append("efficiency cell wrote a checkpoint")
    if failures:
        raise RuntimeError("summary validation failed:\n- " + "\n- ".join(failures))
    metrics_audit = validate_metrics(
        metrics_path,
        total_steps=expected_steps,
        timed_steps=expected_steady,
        tokens_per_update=expected_tokens_per_update,
        steady_train_s=float(summary["steady_train_s"]),
    )
    return summary, metrics_audit


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    manifest_path = output / "worker_manifest.json"
    if manifest_path.exists() or (output / "trainer").exists():
        raise RuntimeError(
            "worker attempt is not empty; controller must create a new attempt"
        )
    contract = common.read_json(args.contract.resolve())
    preflight = common.read_json(args.preflight.resolve())
    contract_sha = common.sha256_file(args.contract.resolve())
    checks = {
        "preflight_passed": preflight.get("passed") is True,
        "contract_sha256": preflight.get("contract_sha256") == contract_sha,
        "method_registered": args.method in contract["method_order"],
        "repeat_index": 0 <= args.repeat_index < int(
            contract["execution_policy"]["repeats"]
        ),
        "position_index": 0 <= args.position_index < len(contract["method_order"]),
        "derived_base_sha256": common.sha256_file(args.derived_base.resolve())
        == preflight["source_audit"]["derived_base_sha256"],
        "profile_wrapper_sha256": common.sha256_file(args.profile_wrapper.resolve())
        == preflight["source_audit"]["profile_wrapper_sha256"],
        "data_dir": args.data_dir.resolve()
        == Path(preflight["data_audit"]["data_dir"]).resolve(),
        "physical_gpu": args.physical_gpu
        == contract["execution_policy"]["physical_timing_gpu"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"worker preflight checks failed: {checks}")
    config = tier_configuration(contract, args.analysis_tier)
    trainer_dir = output / "trainer"
    command = build_training_command(args, config, trainer_dir)
    common.atomic_write_json(
        status_path,
        {
            "status": "running",
            "script_version": SCRIPT_VERSION,
            "tier": args.analysis_tier,
            "method": args.method,
            "repeat_index": args.repeat_index,
            "position_index": args.position_index,
            "command": command,
            "created_at": common.now_iso(),
        },
    )
    env = common.subprocess_environment(
        args.official_repo.resolve(),
        derived_base=args.derived_base.resolve(),
        derived_base_sha256=preflight["source_audit"]["derived_base_sha256"],
        physical_gpu=args.physical_gpu,
    )
    return_code = tee_process(command, env, output / "terminal.log")
    if return_code:
        raise RuntimeError(f"trainer returned non-zero exit status {return_code}")
    summary, metrics_audit = validate_summary(
        args, contract, preflight, config, trainer_dir
    )
    summary_path = trainer_dir / "summary.json"
    metrics_path = trainer_dir / "metrics.csv"
    observed = {
        "steady_train_s": float(summary["steady_train_s"]),
        "steady_steps": int(summary["steady_steps"]),
        "step_avg_ms": float(summary["step_avg_ms"]),
        "steps_per_s": 1000.0 / float(summary["step_avg_ms"]),
        "tokens_per_s": int(contract["frozen_configuration"]["tokens_per_update"])
        * 1000.0
        / float(summary["step_avg_ms"]),
        "timed_training_peak_allocated_bytes": int(
            summary["timed_training_peak_allocated_bytes"]
        ),
        "timed_training_peak_reserved_bytes": int(
            summary["timed_training_peak_reserved_bytes"]
        ),
        "k_state_bytes": int(summary["k_state_bytes"]),
        "optimizer_state_bytes": int(summary["optimizer_state_bytes"]),
        "model_parameter_bytes": int(summary["model_parameter_bytes"]),
        "init_sha256": summary["init_sha256"],
        "runtime_fingerprint": preflight["runtime_fingerprint"],
        "data_fingerprint": preflight["data_audit"]["fingerprint"],
        "resume_count": int(summary["resume_count"]),
        "timing_comparable": bool(summary["timing_comparable"]),
        "completed_steps": int(summary["completed_steps"]),
        "tokens_seen": int(summary["tokens_seen"]),
        "timed_refresh_updates_expected": (
            0
            if args.method == "muon"
            else sum(
                step % 32 == 0
                for step in range(33, int(config["num_iterations"]) + 1)
            )
        ),
    }
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "passed": True,
        "tier": args.analysis_tier,
        "method": args.method,
        "repeat_index": args.repeat_index,
        "position_index": args.position_index,
        "host_id": args.host_id,
        "execution_domain": args.execution_domain,
        "contract_sha256": contract_sha,
        "preflight_sha256": common.sha256_file(args.preflight.resolve()),
        "summary_sha256": common.sha256_file(summary_path),
        "metrics_sha256": common.sha256_file(metrics_path),
        "terminal_log_sha256": common.sha256_file(output / "terminal.log"),
        "derived_base_sha256": preflight["source_audit"]["derived_base_sha256"],
        "profile_wrapper_sha256": preflight["source_audit"][
            "profile_wrapper_sha256"
        ],
        "trainer_local_base_sha256": common.sha256_file(
            trainer_dir / "train_llama_swiglu_base.py"
        ),
        "metric_audit": metrics_audit,
        "observed": observed,
        "completed_at": common.now_iso(),
    }
    common.atomic_write_json(manifest_path, manifest)
    common.atomic_write_json(
        status_path,
        {
            "status": "completed",
            "script_version": SCRIPT_VERSION,
            "passed": True,
            "worker_manifest": str(manifest_path),
            "completed_at": common.now_iso(),
        },
    )
    print(f"LLaMA-1B efficiency manifest: {manifest_path}", flush=True)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except BaseException as exc:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        common.atomic_write_json(
            output / "status.json",
            {
                "status": "failed",
                "script_version": SCRIPT_VERSION,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "failed_at": common.now_iso(),
            },
        )
        traceback.print_exc()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
