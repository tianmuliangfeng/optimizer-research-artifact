"""Audit, run, resume, validate, and upload LLaMA/SwiGLU experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METHOD_ORDER = ("down_diag", "down_none", "newton_full", "muon", "adamw")
METHOD_SET = frozenset(METHOD_ORDER)
DEFAULT_PROJECT = "Selective-Newton-Muon-MainConf-LLaMA-SwiGLU-20260720"
EXPECTED_PARAMETER_COUNT = 123_551_232
EXPECTED_MATRIX_TENSORS = 84
EXPECTED_BACKUP_TENSORS = 26
STABLE_RUNTIME_FIELDS = (
    "python_executable",
    "python_version",
    "numpy",
    "torch",
    "torch_cuda",
    "triton",
    "triton_kernels_sha256",
    "gpu_name",
    "gpu_total_memory_bytes",
    "gpu_capability",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S+0000")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def lexical_absolute(path: Path) -> Path:
    """Return an absolute path without dereferencing a virtualenv symlink.

    On Linux, resolving ``venv/bin/python`` can yield ``/usr/bin/python`` and
    silently discard the virtual environment. ``abspath`` normalizes the
    spelling while preserving the interpreter entry point that was requested.
    """

    return Path(os.path.abspath(os.path.expanduser(str(path))))


def default_output_root() -> Path:
    artifact_root = Path(__file__).resolve().parents[2]
    results_root = Path(
        os.environ.get("SNM_RESULTS_ROOT", str(artifact_root / "runs"))
    ).expanduser()
    return results_root / "17_llama_swiglu_validation" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled parameter-matched LLaMA/SwiGLU architecture validation"
    )
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_SET), default=list(METHOD_ORDER))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--numerical-smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=34)
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--resume-batch", type=Path)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-train-log-every", type=int, default=20)
    parser.add_argument("--wandb-init-timeout", type=int, default=120)
    parser.add_argument("--checkpoint-every", type=int, default=128)
    parser.add_argument("--device-batch-size", type=int, default=64)
    parser.add_argument("--backup-lr", type=float, default=0.0036)
    parser.add_argument("--matrix-lr", type=float, default=0.01)
    parser.add_argument("--adamw-matrix-lr", type=float, default=0.000576)
    args = parser.parse_args()
    if len(args.methods) != len(set(args.methods)):
        parser.error("--methods contains duplicates")
    if args.smoke_steps < 32:
        parser.error("--smoke-steps must be at least 32 to exercise the first K refresh")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    modes = sum(bool(value) for value in (args.dry_run, args.preflight, args.numerical_smoke))
    if modes > 1:
        parser.error("choose only one of --dry-run, --preflight, or --numerical-smoke")
    if args.resume_batch and (modes or args.smoke_manifest):
        parser.error("--resume-batch cannot be combined with planning/smoke options")
    if not (args.dry_run or args.preflight or args.numerical_smoke or args.resume_batch):
        if args.smoke_manifest is None:
            parser.error("formal training requires --smoke-manifest")
    if args.numerical_smoke and args.wandb_mode != "disabled":
        parser.error("numerical smoke must use --wandb-mode disabled")
    return args


def training_script_path() -> Path:
    return Path(__file__).with_name("train_llama_swiglu.py").resolve()


def stable_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    return {field: runtime.get(field) for field in STABLE_RUNTIME_FIELDS}


def subprocess_env(official_repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(official_repo) + (os.pathsep + current if current else "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def validate_runtime(python_exe: Path, official_repo: Path) -> dict[str, Any]:
    kernel_path = official_repo / "triton_kernels.py"
    if not kernel_path.is_file():
        raise FileNotFoundError(f"missing pinned triton_kernels.py: {kernel_path}")
    probe = r'''
import json, pathlib, sys
import numpy, torch, triton, triton_kernels
if not torch.cuda.is_available():
    raise RuntimeError("training interpreter cannot access CUDA")
gpu = torch.cuda.get_device_properties(0)
payload = {
    "python_executable": str(pathlib.Path(sys.executable).absolute()),
    "python_version": list(sys.version_info[:3]),
    "python_full": sys.version.replace("\n", " "),
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "triton": triton.__version__,
    "triton_module": str(pathlib.Path(triton.__file__).resolve()),
    "triton_kernels_module": str(pathlib.Path(triton_kernels.__file__).resolve()),
    "gpu_name": gpu.name,
    "gpu_total_memory_bytes": int(gpu.total_memory),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
}
print("RUNTIME_JSON " + json.dumps(payload, sort_keys=True))
'''
    result = subprocess.run(
        [str(python_exe), "-c", probe],
        env=subprocess_env(official_repo),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"training runtime validation failed for {python_exe}:\n{result.stdout}\n{result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.startswith("RUNTIME_JSON ")]
    if len(lines) != 1:
        raise RuntimeError(f"runtime probe produced no unique JSON payload:\n{result.stdout}")
    payload = json.loads(lines[0].split(" ", 1)[1])
    payload["triton_kernels_sha256"] = sha256_file(kernel_path)
    return payload


def audit_data(data_dir: Path) -> dict[str, Any]:
    train = sorted(data_dir.glob("fineweb_train_*.bin"))
    val = sorted(data_dir.glob("fineweb_val_*.bin"))
    if len(train) != 50 or len(val) != 1:
        raise RuntimeError(
            f"FineWeb audit requires 50 train shards and 1 val shard; observed {len(train)} and {len(val)} in {data_dir}"
        )

    def header(path: Path) -> dict[str, Any]:
        import struct

        with path.open("rb") as handle:
            raw = handle.read(12)
        if len(raw) != 12:
            raise RuntimeError(f"short shard header: {path}")
        magic, version, tokens = struct.unpack("<iii", raw)
        if magic != 20240520 or version != 1:
            raise RuntimeError(f"invalid shard header: {path}")
        return {"name": path.name, "bytes": path.stat().st_size, "tokens": tokens}

    payload = {
        "data_dir": str(data_dir.resolve()),
        "train_shard_count": len(train),
        "val_shard_count": len(val),
        "first_train": header(train[0]),
        "last_train": header(train[-1]),
        "validation": header(val[0]),
        "total_bytes": sum(path.stat().st_size for path in [*train, *val]),
        "ordered_names": [path.name for path in [*train, *val]],
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def validate_wandb_access(args: argparse.Namespace) -> dict[str, Any]:
    if args.wandb_mode != "online" or args.numerical_smoke:
        return {"status": "not_required", "mode": args.wandb_mode}
    try:
        import wandb

        api = wandb.Api(timeout=30)
        viewer = api.viewer
        return {"status": "ready", "viewer": getattr(viewer, "username", None)}
    except Exception as exc:
        raise RuntimeError(f"W&B online readiness check failed: {exc!r}") from exc


def init_command(
    args: argparse.Namespace, method: str, data_dir: Path, script: Path
) -> list[str]:
    return [
        str(args.python_exe),
        str(script),
        "--method",
        method,
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(args.output_root / "_init_only_unused"),
        "--seed",
        str(args.seed),
        "--init-only",
    ]


def run_init_audit(
    args: argparse.Namespace, official_repo: Path, data_dir: Path, script: Path
) -> dict[str, Any]:
    observed: dict[str, dict[str, Any]] = {}
    for method in args.methods:
        result = subprocess.run(
            init_command(args, method, data_dir, script),
            env=subprocess_env(official_repo),
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"initialization audit failed for {method}:\n{result.stdout}\n{result.stderr}"
            )
        lines = [
            line for line in result.stdout.splitlines() if line.startswith("LLAMA_INIT_AUDIT ")
        ]
        if len(lines) != 1:
            raise RuntimeError(f"no unique init payload for {method}:\n{result.stdout}")
        observed[method] = json.loads(lines[0].split(" ", 1)[1])

    fingerprints = {payload["init_sha256"] for payload in observed.values()}
    if len(fingerprints) != 1:
        raise RuntimeError(f"method initialization fingerprints differ: {fingerprints}")
    for method, payload in observed.items():
        architecture = payload["architecture"]
        failures: list[str] = []
        if architecture["parameter_count"] != EXPECTED_PARAMETER_COUNT:
            failures.append(f"parameter_count={architecture['parameter_count']}")
        if architecture["matrix_tensor_count"] != EXPECTED_MATRIX_TENSORS:
            failures.append(f"matrix_tensor_count={architecture['matrix_tensor_count']}")
        if architecture["backup_tensor_count"] != EXPECTED_BACKUP_TENSORS:
            failures.append(f"backup_tensor_count={architecture['backup_tensor_count']}")
        if not architecture["embedding_head_tied"]:
            failures.append("embedding/head are not tied")
        if architecture["bias_parameter_count"] != 0:
            failures.append("bias parameters are present")
        expected_groups = {
            "newton_full": 48,
            "down_diag": 48,
            "down_none": 36,
            "muon": 0,
            "adamw": 0,
        }[method]
        if architecture["preconditioner_group_count"] != expected_groups:
            failures.append(
                f"preconditioner_group_count={architecture['preconditioner_group_count']}"
            )
        if failures:
            raise RuntimeError(f"architecture audit failed for {method}: " + "; ".join(failures))

    k_bytes: dict[str, int] = {}
    for method, payload in observed.items():
        total = 0
        for group in payload["architecture"]["preconditioner_groups"]:
            width = int(group["input_width"])
            elements = width * width if group["kind"] == "dense" else width
            total += elements * 4 * 2  # FP32 covariance plus applied inverse
        k_bytes[method] = total
    return {
        "common_init_sha256": next(iter(fingerprints)),
        "methods": observed,
        "expected_k_state_bytes": k_bytes,
    }


def common_config(args: argparse.Namespace, smoke: bool) -> dict[str, Any]:
    steps = args.smoke_steps if smoke else 6200
    return {
        "num_iterations": steps,
        "global_batch_size": 512,
        "device_batch_size": args.device_batch_size,
        "sequence_length": 1024,
        "val_every": steps if smoke else 100,
        "val_tokens": args.device_batch_size * 1024 if smoke else 10485760,
        "warmdown_iters": 1 if smoke else 1800,
        "backup_lr": args.backup_lr,
        "matrix_lr": args.matrix_lr,
        "adamw_matrix_lr": args.adamw_matrix_lr,
        "checkpoint_every": 0 if smoke else args.checkpoint_every,
    }


def train_command(
    args: argparse.Namespace,
    method: str,
    data_dir: Path,
    output_dir: Path,
    script: Path,
    smoke: bool,
) -> list[str]:
    config = common_config(args, smoke)
    command = [
        str(args.python_exe),
        str(script),
        "--method",
        method,
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
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
        str(config["checkpoint_every"]),
        "--resume",
        "never" if smoke else "auto",
    ]
    if smoke:
        command.append("--no-save-final")
    return command


def tee_process(command: list[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND " + json.dumps(command) + "\n")
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
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return process.wait()


def validate_summary(
    summary_path: Path,
    method: str,
    args: argparse.Namespace,
    runtime: dict[str, Any],
    init_audit: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    if not summary_path.is_file():
        raise RuntimeError(f"missing summary for {method}: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    expected_steps = args.smoke_steps if smoke else 6200
    if payload.get("status") != "completed":
        failures.append("status is not completed")
    if payload.get("method") != method or payload.get("seed") != args.seed:
        failures.append("method/seed mismatch")
    if payload.get("completed_steps") != expected_steps:
        failures.append(f"completed_steps != {expected_steps}")
    if payload.get("init_sha256") != init_audit["common_init_sha256"]:
        failures.append("initialization fingerprint mismatch")
    if payload.get("architecture", {}).get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        failures.append("parameter count mismatch")
    for key in ("final_val_loss", "best_val_loss", "final_train_loss"):
        value = payload.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"{key} is non-finite")
    expected_k = init_audit["expected_k_state_bytes"][method]
    if payload.get("k_state_bytes") != expected_k:
        failures.append(
            f"k_state_bytes={payload.get('k_state_bytes')} expected={expected_k}"
        )
    if stable_runtime(payload.get("runtime", {})) != stable_runtime(runtime):
        failures.append("training runtime differs from controller preflight")
    metrics_path = summary_path.with_name("metrics.csv")
    try:
        validate_metric_evidence(
            metrics_path,
            total_steps=expected_steps,
            val_every=expected_steps if smoke else 100,
            global_batch_size=512,
            sequence_length=1024,
        )
    except Exception as exc:
        failures.append(f"metric evidence invalid: {exc}")
    checkpoint = str(payload.get("checkpoint_path", ""))
    if smoke and checkpoint:
        failures.append("smoke unexpectedly saved a checkpoint")
    if not smoke and (not checkpoint or not Path(checkpoint).is_file()):
        failures.append("formal run has no final resumable checkpoint")
    if failures:
        raise RuntimeError(f"summary validation failed for {method}:\n- " + "\n- ".join(failures))
    return payload


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty metrics file: {path}")
    return rows


def validate_metric_evidence(
    path: Path,
    *,
    total_steps: int,
    val_every: int,
    global_batch_size: int,
    sequence_length: int,
) -> None:
    rows = read_metrics(path)
    train_rows = [row for row in rows if row.get("event") == "train"]
    val_rows = [row for row in rows if row.get("event") == "val"]
    other = [row for row in rows if row.get("event") not in ("train", "val")]
    if other:
        raise ValueError(f"unknown metric events: {sorted({row.get('event') for row in other})}")
    train_steps = [int(row["step"]) for row in train_rows]
    expected_train = list(range(1, total_steps + 1))
    if train_steps != expected_train:
        raise ValueError(
            f"train steps are incomplete or duplicated: observed={len(train_steps)} expected={total_steps}"
        )
    expected_val = list(range(0, total_steps + 1, val_every))
    if expected_val[-1] != total_steps:
        expected_val.append(total_steps)
    val_steps = [int(row["step"]) for row in val_rows]
    if val_steps != expected_val:
        raise ValueError(f"validation steps differ: {val_steps} != {expected_val}")
    prior_time = -1.0
    for row in rows:
        loss = float(row["loss"])
        train_s = float(row["train_s"])
        tokens = int(row["tokens_seen"])
        step = int(row["step"])
        if not math.isfinite(loss) or not math.isfinite(train_s):
            raise ValueError(f"non-finite metric at {row['event']} step {step}")
        if train_s + 1e-9 < prior_time:
            raise ValueError("cumulative training time decreased")
        prior_time = train_s
        expected_tokens = step * global_batch_size * sequence_length
        if tokens != expected_tokens:
            raise ValueError(
                f"tokens_seen mismatch at step {step}: {tokens} != {expected_tokens}"
            )


def upload_to_wandb(
    args: argparse.Namespace,
    batch_id: str,
    method: str,
    method_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    if args.wandb_mode == "disabled":
        return {"status": "disabled", "mode": "disabled"}
    import wandb

    metrics = read_metrics(method_dir / "metrics.csv")
    identity = {
        "family": "llama_swiglu_parameter_matched_r1",
        "seed": args.seed,
        "method": method,
        "architecture": summary["architecture"]["config"],
        "backup_lr": args.backup_lr,
        "matrix_lr": args.matrix_lr,
        "adamw_matrix_lr": args.adamw_matrix_lr,
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    run_name = f"llama_swiglu_{method}_seed{args.seed}"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=run_id,
        resume="allow",
        name=run_name,
        group=f"llama_swiglu_seed{args.seed}",
        mode=args.wandb_mode,
        config=identity
        | {
            "batch_id": batch_id,
            "wandb_upload_timing": "after_training_completed",
            "wandb_tables_enabled": False,
            "init_sha256": summary["init_sha256"],
            "timing_comparable": summary["timing_comparable"],
        },
        tags=["llama", "swiglu", "fineweb10B", "architecture-validation"],
        settings=wandb.Settings(init_timeout=args.wandb_init_timeout),
    )
    try:
        per_step: dict[int, dict[str, float]] = {}
        final_train_step = max(int(row["step"]) for row in metrics if row["event"] == "train")
        for row in metrics:
            step = int(row["step"])
            event = row["event"]
            if event == "train" and step % args.wandb_train_log_every != 0 and step != final_train_step:
                continue
            values = per_step.setdefault(step, {})
            if event == "train":
                values["train/loss_step"] = float(row["loss"])
            else:
                values["val/loss"] = float(row["loss"])
            values.update(
                {
                    "time/train_s": float(row["train_s"]),
                    "performance/step_avg_ms": float(row["step_avg_ms"]),
                    "lr/backup": float(row["lr_backup"]),
                    "lr/matrix": float(row["lr_matrix"]),
                    "tokens/seen": float(row["tokens_seen"]),
                }
            )
        for step in sorted(per_step):
            wandb.log(per_step[step], step=step)
        run.summary.update(
            {
                "final_val_loss": summary["final_val_loss"],
                "best_val_loss": summary["best_val_loss"],
                "final_train_loss": summary["final_train_loss"],
                "peak_allocated_mib": summary["peak_allocated_mib"],
                "k_state_mib": summary["k_state_bytes"] / (1024**2),
                "k_cov_mib": summary["k_cov_bytes"] / (1024**2),
                "k_inv_mib": summary["k_inv_bytes"] / (1024**2),
                "activation_stat_mib": summary["activation_stat_bytes"] / (1024**2),
                "preconditioner_workspace_mib": summary["preconditioner_workspace_bytes"] / (1024**2),
                "optimizer_state_mib": summary["optimizer_state_bytes"] / (1024**2),
                "model_parameter_count": summary["architecture"]["parameter_count"],
                "train_s": summary["train_s"],
                "step_avg_ms": summary["step_avg_ms"],
                "resume_count": summary["resume_count"],
                "timing_comparable": summary["timing_comparable"],
            }
        )
    finally:
        run.finish()
    return {
        "status": "uploaded",
        "mode": args.wandb_mode,
        "project": args.wandb_project,
        "run_id": run_id,
        "run_name": run_name,
        "uploaded_at": now_iso(),
    }


def validate_smoke_certificate(
    path: Path,
    args: argparse.Namespace,
    runtime: dict[str, Any],
    data_audit: dict[str, Any],
    script_sha256: str,
    init_audit: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    failures: list[str] = []
    if payload.get("batch_kind") != "smoke" or payload.get("status") != "completed":
        failures.append("manifest is not a completed smoke")
    if payload.get("script_sha256") != script_sha256:
        failures.append("training source differs from smoke")
    if payload.get("data_audit", {}).get("fingerprint") != data_audit["fingerprint"]:
        failures.append("FineWeb data differs from smoke")
    if stable_runtime(payload.get("runtime", {})) != stable_runtime(runtime):
        failures.append("stable training runtime differs from smoke")
    if payload.get("init_audit", {}).get("common_init_sha256") != init_audit["common_init_sha256"]:
        failures.append("initialization fingerprint differs from smoke")
    completed = set(payload.get("completed_methods", []))
    missing = set(args.methods) - completed
    if missing:
        failures.append(f"smoke did not complete requested methods: {sorted(missing)}")
    if failures:
        raise RuntimeError("smoke certificate validation failed:\n- " + "\n- ".join(failures))
    return payload


def plan_payload(
    args: argparse.Namespace,
    batch_id: str,
    kind: str,
    runtime: dict[str, Any],
    data_audit: dict[str, Any],
    init_audit: dict[str, Any],
    script_sha256: str,
    smoke_certificate: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "family": "llama_swiglu_parameter_matched_r1",
        "batch_id": batch_id,
        "batch_kind": kind,
        "created_at": now_iso(),
        "seed": args.seed,
        "methods": args.methods,
        "config": common_config(args, kind == "smoke"),
        "official_repo": str(args.official_repo.resolve()),
        "python_exe": str(args.python_exe),
        "runtime": runtime,
        "data_audit": data_audit,
        "init_audit": init_audit,
        "script_sha256": script_sha256,
        "smoke_certificate_sha256": (
            canonical_json_sha256(smoke_certificate) if smoke_certificate else None
        ),
        "wandb_project": args.wandb_project,
        "wandb_mode": "disabled" if kind == "smoke" else args.wandb_mode,
    }


def validate_resume_plan(
    plan: dict[str, Any],
    args: argparse.Namespace,
    runtime: dict[str, Any],
    data_audit: dict[str, Any],
    init_audit: dict[str, Any],
    script_sha256: str,
) -> None:
    failures: list[str] = []
    if plan.get("batch_kind") != "formal":
        failures.append("only formal batches can resume")
    if plan.get("seed") != args.seed:
        failures.append("seed differs")
    if plan.get("methods") != args.methods:
        failures.append("method list/order differs")
    if plan.get("config") != common_config(args, False):
        failures.append("training configuration differs")
    if plan.get("script_sha256") != script_sha256:
        failures.append("training source differs")
    if plan.get("data_audit", {}).get("fingerprint") != data_audit["fingerprint"]:
        failures.append("data fingerprint differs")
    if stable_runtime(plan.get("runtime", {})) != stable_runtime(runtime):
        failures.append("stable training runtime differs")
    if plan.get("init_audit", {}).get("common_init_sha256") != init_audit["common_init_sha256"]:
        failures.append("initialization fingerprint differs")
    if failures:
        raise RuntimeError("resume validation failed:\n- " + "\n- ".join(failures))


def write_batch_csv(batch_dir: Path, summaries: dict[str, dict[str, Any]]) -> None:
    path = batch_dir / "llama_swiglu_summary.csv"
    fields = [
        "method",
        "seed",
        "final_val_loss",
        "best_val_loss",
        "final_train_loss",
        "peak_allocated_mib",
        "k_state_mib",
        "optimizer_state_mib",
        "step_avg_ms",
        "train_s",
        "resume_count",
        "timing_comparable",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHOD_ORDER:
            if method not in summaries:
                continue
            item = summaries[method]
            writer.writerow(
                {
                    "method": method,
                    "seed": item["seed"],
                    "final_val_loss": item["final_val_loss"],
                    "best_val_loss": item["best_val_loss"],
                    "final_train_loss": item["final_train_loss"],
                    "peak_allocated_mib": item["peak_allocated_mib"],
                    "k_state_mib": item["k_state_bytes"] / (1024**2),
                    "optimizer_state_mib": item["optimizer_state_bytes"] / (1024**2),
                    "step_avg_ms": item["step_avg_ms"],
                    "train_s": item["train_s"],
                    "resume_count": item["resume_count"],
                    "timing_comparable": item["timing_comparable"],
                }
            )


def print_plan(args: argparse.Namespace, data_dir: Path) -> None:
    kind = "smoke" if args.numerical_smoke else "formal"
    print("LLaMA/SwiGLU architecture-validation plan")
    print(f"training interpreter: {args.python_exe}")
    print(f"official support repo: {args.official_repo.resolve()}")
    print(f"FineWeb data:         {data_dir}")
    print(f"kind:                 {kind}")
    print(f"seed:                 {args.seed}")
    print(f"methods:              {' -> '.join(args.methods)}")
    print(f"config:               {json.dumps(common_config(args, args.numerical_smoke), sort_keys=True)}")
    print(f"W&B:                  {args.wandb_mode} / {args.wandb_project}")
    print("block4:               excluded")


def main() -> None:
    args = parse_args()
    args.official_repo = args.official_repo.resolve()
    # Do not call Path.resolve() here: a venv's python is commonly a symlink to
    # the system executable, and dereferencing it bypasses the venv site-packages.
    args.python_exe = lexical_absolute(args.python_exe)
    args.output_root = args.output_root.resolve()
    script = training_script_path()
    data_dir = (args.official_repo / "data" / "fineweb10B").resolve()
    print_plan(args, data_dir)
    if args.dry_run:
        return
    if not args.python_exe.is_file():
        raise FileNotFoundError(f"training interpreter does not exist: {args.python_exe}")
    if not script.is_file():
        raise FileNotFoundError(script)
    runtime = validate_runtime(args.python_exe, args.official_repo)
    data_audit = audit_data(data_dir)
    script_sha256 = sha256_file(script)
    init_audit = run_init_audit(args, args.official_repo, data_dir, script)
    wandb_readiness = validate_wandb_access(args)
    print(f"Initialization audit: {init_audit['common_init_sha256']}")
    print(
        "Expected K state MiB: "
        + json.dumps(
            {
                method: value / (1024**2)
                for method, value in init_audit["expected_k_state_bytes"].items()
            },
            sort_keys=True,
        )
    )
    if args.preflight:
        args.output_root.mkdir(parents=True, exist_ok=True)
        artifact = args.output_root / f"{timestamp_id()}_preflight_seed{args.seed}.json"
        atomic_write_json(
            artifact,
            {
                "status": "passed",
                "created_at": now_iso(),
                "runtime": runtime,
                "data_audit": data_audit,
                "init_audit": init_audit,
                "script_sha256": script_sha256,
                "wandb_readiness": wandb_readiness,
            },
        )
        print(f"LLaMA preflight artifact: {artifact}")
        return

    smoke_certificate = None
    if args.smoke_manifest is not None:
        smoke_certificate = validate_smoke_certificate(
            args.smoke_manifest,
            args,
            runtime,
            data_audit,
            script_sha256,
            init_audit,
        )

    if args.resume_batch:
        batch_dir = args.resume_batch.resolve()
        plan_path = batch_dir / "llama_plan.json"
        if not plan_path.is_file():
            raise FileNotFoundError(f"resume batch has no llama_plan.json: {batch_dir}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        validate_resume_plan(
            plan, args, runtime, data_audit, init_audit, script_sha256
        )
        batch_id = str(plan["batch_id"])
        kind = "formal"
    else:
        kind = "smoke" if args.numerical_smoke else "formal"
        batch_id = f"{timestamp_id()}_{kind}_seed{args.seed}"
        batch_dir = args.output_root / batch_id
        batch_dir.mkdir(parents=True, exist_ok=False)
        plan = plan_payload(
            args,
            batch_id,
            kind,
            runtime,
            data_audit,
            init_audit,
            script_sha256,
            smoke_certificate,
        )
        atomic_write_json(batch_dir / "llama_plan.json", plan)

    manifest_path = batch_dir / "llama_manifest.json"
    existing_manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {
        **plan,
        "status": "running",
        "last_updated_at": now_iso(),
        "wandb_readiness": wandb_readiness,
        "completed_methods": [],
        "failed_methods": {},
        "method_results": existing_manifest.get("method_results", {}),
    }
    atomic_write_json(manifest_path, manifest)
    summaries: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    smoke = kind == "smoke"

    for index, method in enumerate(args.methods, start=1):
        method_dir = batch_dir / f"{index:02d}_{method}"
        method_dir.mkdir(parents=True, exist_ok=True)
        run_script = method_dir / "train_llama_swiglu.py"
        if not run_script.is_file():
            shutil.copy2(script, run_script)
        if sha256_file(run_script) != script_sha256:
            raise RuntimeError(f"saved training source changed in {method_dir}")
        summary_path = method_dir / "summary.json"
        try:
            if summary_path.is_file():
                summary = validate_summary(
                    summary_path, method, args, runtime, init_audit, smoke
                )
                print(f"Skipping completed local method: {method}")
            else:
                command = train_command(
                    args, method, data_dir, method_dir, run_script, smoke
                )
                return_code = tee_process(
                    command,
                    subprocess_env(args.official_repo),
                    method_dir / "terminal.log",
                )
                if return_code != 0:
                    raise RuntimeError(f"training exited with code {return_code}")
                summary = validate_summary(
                    summary_path, method, args, runtime, init_audit, smoke
                )
            summaries[method] = summary
            wandb_path = method_dir / "wandb_upload.json"
            if smoke:
                upload = {"status": "disabled_for_smoke"}
            elif wandb_path.is_file():
                upload = json.loads(wandb_path.read_text(encoding="utf-8"))
                if upload.get("status") != "uploaded" and args.wandb_mode != "disabled":
                    upload = upload_to_wandb(
                        args, batch_id, method, method_dir, summary
                    )
            else:
                upload = upload_to_wandb(args, batch_id, method, method_dir, summary)
            atomic_write_json(wandb_path, upload)
            manifest["method_results"][method] = {
                "status": "completed",
                "summary_path": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "wandb": upload,
            }
        except Exception as exc:
            failures[method] = repr(exc)
            manifest["method_results"][method] = {
                "status": "failed_or_interrupted",
                "error": repr(exc),
            }
            manifest["failed_methods"] = failures
            manifest["completed_methods"] = list(summaries)
            manifest["last_updated_at"] = now_iso()
            atomic_write_json(manifest_path, manifest)
            if not args.continue_on_error:
                print(f"LLaMA artifacts: {batch_dir}")
                raise
        manifest["completed_methods"] = list(summaries)
        manifest["failed_methods"] = failures
        manifest["last_updated_at"] = now_iso()
        atomic_write_json(manifest_path, manifest)

    write_batch_csv(batch_dir, summaries)
    all_complete = set(args.methods) == set(summaries) and not failures
    wandb_complete = smoke or args.wandb_mode == "disabled" or all(
        manifest["method_results"][method]["wandb"].get("status") == "uploaded"
        for method in summaries
    )
    manifest.update(
        {
            "status": (
                "completed"
                if all_complete and wandb_complete
                else "completed_valid_local_wandb_incomplete"
                if all_complete
                else "incomplete"
            ),
            "completed_methods": list(summaries),
            "failed_methods": failures,
            "wandb_complete": wandb_complete,
            "completed_at": now_iso(),
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(f"LLaMA artifacts: {batch_dir}")
    print(f"LLaMA manifest:  {manifest_path}")
    if not all_complete:
        raise RuntimeError(f"batch incomplete: {failures}")


if __name__ == "__main__":
    main()
