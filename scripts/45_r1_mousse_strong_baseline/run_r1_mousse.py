"""Audit, tune, and formally run the controlled 124M Mousse-R1 baseline."""

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
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
R0_DIR = SCRIPT_DIR.parent / "14_official_newton_muon_r0"
SHARED_DIR = SCRIPT_DIR.parent / "_shared"
sys.path[:0] = [str(R0_DIR), str(SHARED_DIR), str(SCRIPT_DIR)]

import run_official_newton_muon_r0 as r0
from mousse_source_builder import DerivedSource, build_source
from project_paths import EXPERIMENT_RESULTS_ROOT


FAMILY = "45_r1_mousse_strong_baseline"
PREFLIGHT_PROTOCOL = "mousse_r1_implementation_audit_v1"
PILOT_PROTOCOL = "mousse_r1_three_point_pilot_v1"
SMOKE_PROTOCOL = "mousse_r1_selected_exact_shape_smoke_v1"
FORMAL_PROTOCOL = "mousse_r1_selected_6200step_v1"
SELECTION_PROTOCOL = "mousse_r1_pilot_selection_v1"
PILOT_PROJECT = "Selective-Newton-Muon-MainConf-R1-MoussePilot-20260731"
FORMAL_PROJECT = "Selective-Newton-Muon-MainConf-R1-MousseFormal-20260731"
TOKENS_PER_STEP = 512 * 1024
FULL_STEPS = 6200
FULL_WARMDOWN = 1800
PILOT_STEPS = 1000
SMOKE_STEPS = 34
VAL_EVERY = 100
VAL_TOKENS = 10_485_760
AUX_LR = 0.0036
MATRIX_WEIGHT_DECAY = 0.01
CENTER_LR = 0.015
TIE_MARGIN = 0.002
JSON_PREFIXES = ("R1M_METADATA ", "R1M_ROUTING ", "R1M_HYPERPARAMS ", "R1M_FINAL_MEMORY ")


@dataclass(frozen=True)
class MousseCell:
    cell_id: str
    lr_label: str
    matrix_lr: float
    auxiliary_lr: float = AUX_LR
    matrix_weight_decay: float = MATRIX_WEIGHT_DECAY


PILOT_CELLS = (
    MousseCell("mousse_lr080", "0.8x_official_mapped", 0.012),
    MousseCell("mousse_lr100", "1.0x_official_mapped", CENTER_LR),
    MousseCell("mousse_lr120", "1.2x_official_mapped", 0.018),
)
CENTER_CELL_ID = "mousse_lr100"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_wandb_readiness(args: argparse.Namespace) -> dict[str, object]:
    if not (args.pilot or args.formal) or args.wandb_mode != "online":
        return {"required": False, "status": "not_checked"}
    try:
        import wandb
        viewer = wandb.Api(timeout=30).viewer
        viewer = viewer() if callable(viewer) else viewer
        if not viewer:
            raise RuntimeError("W&B returned no authenticated viewer")
        return {"required": True, "status": "authenticated_online", "base_url": os.environ.get("WANDB_BASE_URL", "default")}
    except Exception as exc:
        raise RuntimeError(f"W&B online readiness failed before training: {exc!r}") from exc


def mode_name(args: argparse.Namespace) -> str:
    if args.formal:
        return "formal"
    if args.formal_smoke:
        return "formal_smoke"
    if args.pilot:
        return "pilot"
    if args.numerical_smoke:
        return "pilot_smoke"
    return "preflight"


def protocol(args: argparse.Namespace) -> str:
    return {
        "preflight": PREFLIGHT_PROTOCOL,
        "pilot": PILOT_PROTOCOL,
        "pilot_smoke": PILOT_PROTOCOL,
        "formal_smoke": SMOKE_PROTOCOL,
        "formal": FORMAL_PROTOCOL,
    }[mode_name(args)]


def total_steps(args: argparse.Namespace) -> int:
    if args.formal:
        return FULL_STEPS
    if args.formal_smoke:
        return args.smoke_steps
    if args.numerical_smoke:
        return args.smoke_steps
    return args.pilot_steps


def warmdown_steps(args: argparse.Namespace) -> int:
    return FULL_WARMDOWN if args.formal else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, default=r0.default_official_repo())
    parser.add_argument("--python-exe", default="python")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT_RESULTS_ROOT / FAMILY / "results")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--numerical-smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--formal-smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--pilot-steps", type=int, default=PILOT_STEPS)
    parser.add_argument("--smoke-steps", type=int, default=SMOKE_STEPS)
    parser.add_argument("--val-every", type=int, default=VAL_EVERY)
    parser.add_argument("--val-tokens", type=int, default=VAL_TOKENS)
    parser.add_argument("--cells", nargs="+", choices=[cell.cell_id for cell in PILOT_CELLS])
    parser.add_argument("--selection-certificate", type=Path)
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--resume-batch", type=Path)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-init-timeout", type=int, default=120)
    parser.add_argument("--wandb-train-log-every", type=int, default=20)
    args = parser.parse_args()
    if args.seed < 0 or args.pilot_steps < 2 or args.smoke_steps < 2:
        parser.error("seed must be non-negative and run lengths must be at least two")
    if args.formal_smoke and args.smoke_steps < SMOKE_STEPS:
        parser.error(f"formal smoke must run at least {SMOKE_STEPS} updates")
    if args.val_every <= 0 or args.val_tokens <= 0 or args.val_tokens % (64 * 1024):
        parser.error("validation tokens must be a positive multiple of 65536")
    if args.cells and not (args.pilot or args.numerical_smoke):
        parser.error("--cells is only valid for pilot/pilot-smoke")
    if (args.formal or args.formal_smoke) and args.selection_certificate is None and args.resume_batch is None:
        parser.error("formal profiles require --selection-certificate")
    if args.formal and args.smoke_manifest is None and args.resume_batch is None:
        parser.error("a new formal batch requires --smoke-manifest")
    if args.resume_batch is not None and args.preflight:
        parser.error("preflight cannot resume a batch")
    if args.wandb_project is None:
        args.wandb_project = FORMAL_PROJECT if args.formal else PILOT_PROJECT
    return args


def parse_prefixed_json(stdout: str, prefix: str) -> dict[str, object]:
    matches = [line[len(prefix) :] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {prefix.strip()} line, observed {len(matches)}")
    payload = json.loads(matches[0])
    if not isinstance(payload, dict):
        raise RuntimeError(f"{prefix.strip()} payload is not an object")
    return payload


def selected_pilot_cells(args: argparse.Namespace) -> list[MousseCell]:
    wanted = set(args.cells or [cell.cell_id for cell in PILOT_CELLS])
    cells = [cell for cell in PILOT_CELLS if cell.cell_id in wanted]
    if not cells:
        raise RuntimeError("empty pilot cell selection")
    return cells


def controlled_env(
    args: argparse.Namespace,
    data_dir: Path,
    cell: MousseCell,
    *,
    init_only: bool = False,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": str(args.seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "R1M_SEED": str(args.seed),
            "R1M_DATA_DIR": str(data_dir.resolve()),
            "R1M_TOTAL_STEPS": str(total_steps(args)),
            "R1M_WARMDOWN_STEPS": str(warmdown_steps(args)),
            "R1M_VAL_EVERY": str(args.val_every),
            "R1M_VAL_TOKENS": str(args.val_tokens),
            "R1M_AUX_LR": repr(cell.auxiliary_lr),
            "R1M_MATRIX_LR": repr(cell.matrix_lr),
            "R1M_MATRIX_WEIGHT_DECAY": repr(cell.matrix_weight_decay),
            "R1M_INIT_ONLY": "1" if init_only else "0",
            "R1M_DISABLE_CHECKPOINT": "0" if args.formal else "1",
        }
    )
    return env


def materialize_source(directory: Path, official_repo: Path, derived: DerivedSource) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "train_r1_mousse.py"
    script.write_text(derived.source, encoding="utf-8", newline="\n")
    shutil.copy2(SCRIPT_DIR / "mousse_optimizer.py", directory / "mousse_optimizer.py")
    shutil.copy2(SCRIPT_DIR / "mousse_contract.json", directory / "mousse_contract.json")
    shutil.copy2(SCRIPT_DIR / "THIRD_PARTY_NOTICES.md", directory / "THIRD_PARTY_NOTICES.md")
    shutil.copytree(SCRIPT_DIR / "upstream_snapshot", directory / "upstream_snapshot", dirs_exist_ok=True)
    shutil.copy2(official_repo / "triton_kernels.py", directory / "triton_kernels.py")
    return script


def initialization_audit(
    args: argparse.Namespace, repo: Path, data_dir: Path, derived: DerivedSource, cell: MousseCell
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="r1m_init_") as raw:
        workspace = Path(raw)
        script = materialize_source(workspace, repo, derived)
        result = subprocess.run(
            [args.python_exe, script.name],
            cwd=workspace,
            env=controlled_env(args, data_dir, cell, init_only=True),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=900,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"initialization audit failed:\n{result.stdout[-8000:]}")
    metadata = parse_prefixed_json(result.stdout, "R1M_METADATA ")
    routing = parse_prefixed_json(result.stdout, "R1M_ROUTING ")
    if metadata.get("method") != "mousse" or metadata.get("seed") != args.seed:
        raise RuntimeError(f"initialization metadata mismatch: {metadata}")
    expected = {"hidden_matrix_tensors": 48, "auxiliary_tensors": 1, "packed_qkv_tensors": 12, "logical_hidden_matrices": 72, "activation_k_state_routes": 0}
    mismatches = {key: (routing.get(key), value) for key, value in expected.items() if routing.get(key) != value}
    if mismatches:
        raise RuntimeError(f"R1 parameter routing mismatch: {mismatches}")
    return {"status": "passed", **metadata, "routing": routing}


def reference_audit(args: argparse.Namespace) -> dict[str, object]:
    code = (
        "import json,sys;sys.path.insert(0," + repr(str(SCRIPT_DIR)) + ");"
        "from mousse_optimizer import run_small_matrix_reference_audit;"
        "print(json.dumps(run_small_matrix_reference_audit('cpu'),sort_keys=True))"
    )
    result = subprocess.run(
        [args.python_exe, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=900, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"small-matrix reference audit failed:\n{result.stdout[-8000:]}")
    payload = json.loads(result.stdout.splitlines()[-1])
    if payload.get("status") != "passed" or payload.get("activation_k_state_routes") != 0:
        raise RuntimeError(f"small-matrix reference audit rejected: {payload}")
    return payload


def validation_steps(steps: int, every: int) -> list[int]:
    return sorted(set(range(0, steps + 1, every)) | {steps})


def curve_mean(rows: list[dict[str, object]]) -> float:
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    if len(ordered) < 2:
        return math.nan
    area = sum(
        (int(right["step"]) - int(left["step"]))
        * (float(left["loss"]) + float(right["loss"])) / 2.0
        for left, right in zip(ordered, ordered[1:])
    )
    span = int(ordered[-1]["step"]) - int(ordered[0]["step"])
    return area / span


def find_log(workspace: Path) -> Path:
    logs = list((workspace / "logs").glob("*.txt"))
    if len(logs) != 1:
        raise RuntimeError(f"expected one official log, observed {len(logs)}")
    return logs[0]


def find_checkpoint(workspace: Path) -> Path | None:
    paths = list((workspace / "logs").glob("*/state_step*.pt"))
    if len(paths) > 1:
        raise RuntimeError(f"expected at most one checkpoint, observed {len(paths)}")
    return paths[0] if paths else None


def parse_metrics(
    log_path: Path,
    stdout_path: Path,
    args: argparse.Namespace,
    cell: MousseCell,
    expected_init: str,
    checkpoint: Path | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    steps, warmdown = total_steps(args), warmdown_steps(args)
    rows: list[dict[str, object]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = r0.VAL_RE.match(line) or r0.TRAIN_RE.match(line)
        if match is None:
            continue
        step = int(match.group("step"))
        event = "validation" if "val_loss:" in line else "train"
        multiplier = 1.0 if step < steps - warmdown else max(0.0, (steps - step) / warmdown)
        rows.append(
            {
                "cell_id": cell.cell_id, "method": "mousse", "event": event,
                "step": step, "total_steps": int(match.group("total")),
                "tokens_seen": step * TOKENS_PER_STEP, "loss": float(match.group("loss")),
                "official_train_time_ms": int(match.group("time")), "step_avg_ms": float(match.group("avg")),
                "lr_multiplier": multiplier, "auxiliary_lr": cell.auxiliary_lr * multiplier,
                "matrix_lr": cell.matrix_lr * multiplier,
            }
        )
    train = sorted((row for row in rows if row["event"] == "train"), key=lambda row: int(row["step"]))
    val = sorted((row for row in rows if row["event"] == "validation"), key=lambda row: int(row["step"]))
    if len(train) != steps:
        raise RuntimeError(f"{cell.cell_id}: expected {steps} train rows, observed {len(train)}")
    if [int(row["step"]) for row in val] != validation_steps(steps, args.val_every):
        raise RuntimeError(f"{cell.cell_id}: validation grid mismatch")
    if any(not math.isfinite(float(row["loss"])) for row in rows):
        raise RuntimeError(f"{cell.cell_id}: non-finite loss")
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    metadata = parse_prefixed_json(stdout, "R1M_METADATA ")
    routing = parse_prefixed_json(stdout, "R1M_ROUTING ")
    hyper = parse_prefixed_json(stdout, "R1M_HYPERPARAMS ")
    memory = parse_prefixed_json(stdout, "R1M_FINAL_MEMORY ")
    peaks = [match for line in stdout.splitlines() if (match := r0.PEAK_RE.match(line))]
    if metadata != {"method": "mousse", "seed": args.seed, "init_sha256": expected_init}:
        raise RuntimeError(f"metadata mismatch: {metadata}")
    expected_hyper = {
        "method": "mousse", "aux_lr": cell.auxiliary_lr, "matrix_lr": cell.matrix_lr,
        "matrix_weight_decay": cell.matrix_weight_decay, "momentum": 0.95, "nesterov": False,
        "factor_beta": 0.95, "factor_epsilon": 1e-5, "factor_alpha": 0.125,
        "refresh_interval": 10, "bias_correction": True, "grafting": True,
        "adjust_lr": "spectral_norm", "ns_epsilon": 1e-8, "total_steps": steps,
        "warmdown_steps": warmdown, "val_every": args.val_every, "val_tokens": args.val_tokens,
    }
    if hyper != expected_hyper:
        raise RuntimeError(f"hyperparameter echo mismatch: {hyper} != {expected_hyper}")
    expected_refreshes = 1 + (steps - 1) // 10
    if int(memory.get("mousse_refreshed_logical_matrices", 0)) != 72:
        raise RuntimeError("not all 72 logical matrices refreshed")
    if int(memory.get("mousse_refresh_count_total", 0)) != 72 * expected_refreshes:
        raise RuntimeError("Mousse refresh count mismatch")
    schema = memory.get("state_schema", {})
    if schema.get("contains_activation_k_state") is not False or routing.get("activation_k_state_routes") != 0:
        raise RuntimeError("activation-K state leakage detected")
    if len(peaks) != 1 or int(memory.get("optimizer_state_bytes", 0)) <= 0:
        raise RuntimeError("invalid state/memory report")
    if args.formal and (checkpoint is None or not checkpoint.is_file() or checkpoint.stat().st_size <= 0):
        raise RuntimeError("formal run is missing its final checkpoint")
    if not args.formal and checkpoint is not None:
        raise RuntimeError("non-formal run unexpectedly produced a checkpoint")
    summary: dict[str, object] = {
        **asdict(cell), "method": "mousse", "controlled_seed": args.seed,
        "init_sha256": expected_init, "total_steps": steps, "warmdown_steps": warmdown,
        "tokens_per_step": TOKENS_PER_STEP, "total_tokens": steps * TOKENS_PER_STEP,
        "initial_val_loss": float(val[0]["loss"]), "final_val_loss": float(val[-1]["loss"]),
        "best_val_loss": min(float(row["loss"]) for row in val),
        "tail5_val_loss_mean": sum(float(row["loss"]) for row in val[-5:]) / min(5, len(val)),
        "normalized_val_auc": curve_mean(val), "final_train_loss": float(train[-1]["loss"]),
        "official_train_time_s_diagnostic": float(val[-1]["official_train_time_ms"]) / 1000.0,
        "peak_memory_allocated_mib": int(peaks[0].group("mib")),
        "optimizer_state_bytes": int(memory["optimizer_state_bytes"]),
        "model_parameter_bytes": int(memory["model_parameter_bytes"]),
        "mousse_refresh_count_total": int(memory["mousse_refresh_count_total"]),
        "mousse_refreshed_logical_matrices": int(memory["mousse_refreshed_logical_matrices"]),
        "checkpoint_path": str(checkpoint.resolve()) if checkpoint else "",
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint else 0,
        "evidence_profile": protocol(args), "formal_evidence": bool(args.formal),
        "timing_eligible": False, "evidence_valid": True,
        "seed_role": "tuned_seed_long_horizon_screen" if args.seed == 2026 else "independent_confirmatory_seed",
    }
    for row in val:
        summary[f"val_loss_step_{row['step']}"] = float(row["loss"])
    return rows, summary


def upload_wandb(
    args: argparse.Namespace, run_dir: Path, run_name: str, batch_id: str,
    cell: MousseCell, rows: list[dict[str, object]], summary: dict[str, object],
    runtime: dict[str, object], derived: DerivedSource,
) -> dict[str, object]:
    if args.formal_smoke or args.numerical_smoke or args.wandb_mode == "disabled":
        return {"status": "disabled"}
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            id=hashlib.sha256(run_name.encode()).hexdigest()[:12], resume="allow",
            name=run_name, group=f"r1_mousse_{mode_name(args)}_seed{args.seed}_{batch_id}",
            mode=args.wandb_mode, dir=str(run_dir), reinit=True,
            tags=["publication", "experiment45", "r1", "mousse", mode_name(args)],
            config={
                "experiment_family": FAMILY, "protocol": protocol(args), "method": "mousse",
                "cell_id": cell.cell_id, "seed": args.seed, "matrix_lr": cell.matrix_lr,
                "auxiliary_lr": cell.auxiliary_lr, "matrix_weight_decay": cell.matrix_weight_decay,
                "official_r1_commit": r0.OFFICIAL_COMMIT, "derived_script_sha256": derived.derived_sha256,
                "num_iterations": total_steps(args), "warmdown_iters": warmdown_steps(args),
                "batch_size_sequences": 512, "sequence_length": 1024,
                "tokens_per_step": TOKENS_PER_STEP, "timing_eligible": False,
                "gpu_name": runtime.get("gpu_name"), "adaptation_label": "Mousse-R1 adaptation",
            },
            settings=wandb.Settings(init_timeout=args.wandb_init_timeout),
        )
        per_step: dict[int, dict[str, float]] = {}
        for row in rows:
            step = int(row["step"])
            if row["event"] == "train" and step % args.wandb_train_log_every and step != total_steps(args):
                continue
            values = per_step.setdefault(step, {})
            values["val/loss" if row["event"] == "validation" else "train/loss_step"] = float(row["loss"])
            values["lr/auxiliary"] = float(row["auxiliary_lr"])
            values["lr/matrix"] = float(row["matrix_lr"])
            values["tokens/seen"] = float(row["tokens_seen"])
        end = per_step.setdefault(total_steps(args), {})
        end["memory/peak_allocated_mib"] = float(summary["peak_memory_allocated_mib"])
        end["memory/optimizer_state_mib"] = float(summary["optimizer_state_bytes"]) / 1024**2
        for step in sorted(per_step):
            wandb.log(per_step[step], step=step)
        for key, value in summary.items():
            if isinstance(value, (str, int, float, bool)):
                run.summary[key] = value
        run.finish()
        return {"status": "uploaded", "run_id": getattr(run, "id", None), "run_url": getattr(run, "url", None)}
    except Exception as exc:
        return {"status": "failed", "error": repr(exc)}


def run_cell(
    args: argparse.Namespace, official_repo: Path, data_dir: Path, batch_dir: Path, batch_id: str,
    cell: MousseCell, derived: DerivedSource, expected_init: str, runtime: dict[str, object],
) -> dict[str, object]:
    run_name = f"mainconf_r1_mousse_{mode_name(args)}_{cell.cell_id}_seed{args.seed}_{batch_id}"
    run_dir = batch_dir / run_name
    summary_path, manifest_path = run_dir / "summary.json", run_dir / "run_manifest.json"
    if summary_path.is_file() and manifest_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics_path = run_dir / "metrics.csv"
        checkpoint_path = Path(str(summary.get("checkpoint_path", "")))
        checkpoint_ok = not args.formal or (
            checkpoint_path.is_file()
            and checkpoint_path.stat().st_size == int(summary.get("checkpoint_bytes", -1))
            and checkpoint_path.stat().st_size > 0
        )
        if (
            manifest.get("status", "").startswith("completed_valid")
            and summary.get("evidence_valid") is True
            and summary.get("cell_id") == cell.cell_id
            and summary.get("controlled_seed") == args.seed
            and summary.get("init_sha256") == expected_init
            and summary.get("total_steps") == total_steps(args)
            and manifest.get("derived_source_sha256") == derived.derived_sha256
            and metrics_path.is_file()
            and checkpoint_ok
        ):
            rows = list(csv.DictReader(metrics_path.open(encoding="utf-8", newline="")))
            if not (args.formal_smoke or args.numerical_smoke) and args.wandb_mode != "disabled" and summary.get("wandb_status") != "uploaded":
                upload = upload_wandb(args, run_dir, run_name, batch_id, cell, rows, summary, runtime, derived)
                summary["wandb_status"] = upload["status"]
                manifest["wandb"] = upload
                manifest["status"] = "completed_valid" if upload["status"] == "uploaded" else "completed_valid_local_wandb_incomplete"
                write_json(summary_path, summary)
                write_json(manifest_path, manifest)
            print(f"Resume: reusing completed Mousse cell {cell.cell_id}")
            return summary
        raise RuntimeError(f"incompatible existing run directory: {run_dir}")
    workspaces = run_dir / "workspaces"
    existing_attempts = sorted(workspaces.glob("attempt_*")) if workspaces.is_dir() else []
    workspace = workspaces / f"attempt_{len(existing_attempts) + 1:03d}"
    script = materialize_source(workspace, official_repo, derived)
    (run_dir / "official_r1_to_mousse.patch").write_text(derived.unified_diff, encoding="utf-8")
    write_json(run_dir / "cell_spec.json", asdict(cell))
    stdout_path = run_dir / "training_stdout.log"
    manifest: dict[str, object] = {
        "status": "running", "run_name": run_name, "seed": args.seed, "cell": asdict(cell),
        "command": [args.python_exe, script.name], "cwd": str(workspace.resolve()),
        "started_at": datetime.now().astimezone().isoformat(), "derived_source_sha256": derived.derived_sha256,
    }
    write_json(manifest_path, manifest)
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8", buffering=1) as output:
        process = subprocess.Popen(
            [args.python_exe, script.name], cwd=workspace,
            env=controlled_env(args, data_dir, cell), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        first_nonfinite = r0.stream_process_with_finite_gate(process, output)
        returncode = process.wait()
    manifest.update({"returncode": returncode, "wall_elapsed_s": time.monotonic() - started, "finished_at": datetime.now().astimezone().isoformat()})
    if first_nonfinite is not None or returncode != 0:
        manifest.update({"status": "invalid_nonfinite" if first_nonfinite else "training_failed", "first_nonfinite": first_nonfinite})
        write_json(manifest_path, manifest)
        raise RuntimeError(f"{cell.cell_id} failed; see {stdout_path}")
    copied_log = run_dir / "training_log_with_source.txt"
    shutil.copy2(find_log(workspace), copied_log)
    checkpoint = find_checkpoint(workspace)
    rows, summary = parse_metrics(copied_log, stdout_path, args, cell, expected_init, checkpoint)
    summary["run_name"] = run_name
    summary["gpu_name"] = runtime.get("gpu_name", "")
    summary["torch_version"] = runtime.get("torch", "")
    summary["torch_cuda_version"] = runtime.get("torch_cuda", "")
    summary["training_runtime_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(r0.runtime_fingerprint(runtime), sort_keys=True).encode()
    ).hexdigest()
    write_csv(run_dir / "metrics.csv", rows)
    write_json(summary_path, summary)
    manifest.update({"status": "completed_valid_local", "summary": summary, "wandb": {"status": "pending"}})
    write_json(manifest_path, manifest)
    upload = upload_wandb(args, run_dir, run_name, batch_id, cell, rows, summary, runtime, derived)
    summary["wandb_status"] = upload["status"]
    manifest["wandb"] = upload
    manifest["status"] = "completed_valid" if upload["status"] in ("uploaded", "disabled") else "completed_valid_local_wandb_incomplete"
    write_json(summary_path, summary)
    write_json(manifest_path, manifest)
    return summary


def validate_selection(path: Path) -> tuple[MousseCell, dict[str, object]]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    failures = []
    if payload.get("status") != "selected" or payload.get("protocol") != SELECTION_PROTOCOL:
        failures.append("selection status/protocol mismatch")
    if payload.get("seed") != 2026 or payload.get("pilot_steps") != PILOT_STEPS:
        failures.append("selection was not made from the frozen seed-2026 1000-step pilot")
    selected = str(payload.get("selected_cell_id"))
    matches = [cell for cell in PILOT_CELLS if cell.cell_id == selected]
    if len(matches) != 1 or float(payload.get("selected_matrix_lr", -1)) != matches[0].matrix_lr:
        failures.append("selected cell/LR is outside the frozen grid")
    if failures:
        raise RuntimeError("selection certificate rejected:\n- " + "\n- ".join(failures))
    return matches[0], {"path": str(resolved), "sha256": sha256_file(resolved), **payload}


def validate_smoke(
    path: Path, args: argparse.Namespace, cell: MousseCell, runtime: dict[str, object],
    init_audit: dict[str, object], derived: DerivedSource,
) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    failures = []
    if payload.get("status") != "completed_valid" or payload.get("protocol") != SMOKE_PROTOCOL:
        failures.append("smoke status/protocol mismatch")
    if payload.get("seed") != args.seed or int(payload.get("total_steps", 0)) < SMOKE_STEPS:
        failures.append("smoke seed/length mismatch")
    if payload.get("cell", {}).get("cell_id") != cell.cell_id:
        failures.append("smoke selected cell mismatch")
    if payload.get("initialization_audit", {}).get("init_sha256") != init_audit.get("init_sha256"):
        failures.append("smoke initialization hash mismatch")
    if payload.get("source_audit", {}).get("derived_source_sha256") != derived.derived_sha256:
        failures.append("smoke derived source mismatch")
    observed = r0.normalize_runtime_fingerprint(payload.get("training_runtime_fingerprint"))
    expected = r0.normalize_runtime_fingerprint(r0.runtime_fingerprint(runtime))
    if observed != expected:
        failures.append("smoke runtime mismatch")
    summary = payload.get("summary", {})
    if summary.get("evidence_valid") is not True or int(summary.get("mousse_refresh_count_total", 0)) < 72 * 4:
        failures.append("smoke did not cross the required refreshes")
    if failures:
        raise RuntimeError("formal smoke certificate rejected:\n- " + "\n- ".join(failures))
    return {"path": str(resolved), "sha256": sha256_file(resolved), "status": "accepted"}


def make_selection(batch_dir: Path, summaries: list[dict[str, object]], manifest_path: Path) -> Path:
    if len(summaries) != 3 or {row["cell_id"] for row in summaries} != {cell.cell_id for cell in PILOT_CELLS}:
        raise RuntimeError("a selection certificate requires all three frozen pilot cells")
    ranked = sorted(summaries, key=lambda row: float(row["final_val_loss"]))
    center = next(row for row in summaries if row["cell_id"] == CENTER_CELL_ID)
    chosen = center if float(center["final_val_loss"]) <= float(ranked[0]["final_val_loss"]) + TIE_MARGIN else ranked[0]
    path = batch_dir / "pilot_selection.json"
    write_json(
        path,
        {
            "status": "selected", "protocol": SELECTION_PROTOCOL, "seed": 2026,
            "pilot_steps": PILOT_STEPS, "selection_endpoint": "step-1000 validation loss",
            "center_tie_margin": TIE_MARGIN, "center_preferred_if_within_margin_of_best": True,
            "selected_cell_id": chosen["cell_id"], "selected_matrix_lr": chosen["matrix_lr"],
            "pilot_manifest": str(manifest_path.resolve()), "pilot_manifest_sha256": sha256_file(manifest_path),
            "ranked_cells": [{"cell_id": row["cell_id"], "matrix_lr": row["matrix_lr"], "final_val_loss": row["final_val_loss"]} for row in ranked],
        },
    )
    return path


def main() -> None:
    args = parse_args()
    repo = args.official_repo.expanduser().resolve()
    provenance = r0.validate_official_repo(repo)
    data = r0.validate_data(repo)
    data_dir = Path(str(data["data_dir"]))
    runtime = r0.validate_runtime(repo, args.python_exe)
    controller = r0.validate_controller_runtime(require_wandb=(args.pilot or args.formal) and args.wandb_mode != "disabled")
    wandb_readiness = validate_wandb_readiness(args)
    if runtime.get("runtime_rejection_reason"):
        raise RuntimeError(str(runtime["runtime_rejection_reason"]))
    derived = build_source(repo)
    resume_plan: dict[str, object] | None = None
    if args.resume_batch is not None:
        resume_plan_path = args.resume_batch.expanduser().resolve() / f"{mode_name(args)}_plan.json"
        if not resume_plan_path.is_file():
            raise RuntimeError(f"resume plan not found: {resume_plan_path}")
        resume_plan = json.loads(resume_plan_path.read_text(encoding="utf-8"))
    selection: dict[str, object] | None = None
    if args.formal or args.formal_smoke:
        selection_path = args.selection_certificate
        if selection_path is None and resume_plan is not None:
            saved_selection = resume_plan.get("selection_certificate", {})
            if isinstance(saved_selection, dict) and saved_selection.get("path"):
                selection_path = Path(str(saved_selection["path"]))
        if selection_path is None:
            raise RuntimeError("formal resume cannot recover its selection certificate path")
        cell, selection = validate_selection(selection_path)
        cells = [cell]
    else:
        cells = selected_pilot_cells(args) if not args.preflight else [PILOT_CELLS[1]]
    init_audit = initialization_audit(args, repo, data_dir, derived, cells[0])
    small_audit = reference_audit(args)
    source_audit = {
        "official_r1_base_sha256": derived.base_canonical_sha256,
        "derived_source_sha256": derived.derived_sha256,
        "mousse_optimizer_sha256": sha256_file(SCRIPT_DIR / "mousse_optimizer.py"),
        "mousse_source_builder_sha256": sha256_file(SCRIPT_DIR / "mousse_source_builder.py"),
        "contract_sha256": sha256_file(SCRIPT_DIR / "mousse_contract.json"),
        "third_party_notice_sha256": sha256_file(SCRIPT_DIR / "THIRD_PARTY_NOTICES.md"),
        "snapshot_manifest_sha256": sha256_file(SCRIPT_DIR / "upstream_snapshot" / "SNAPSHOT_MANIFEST.json"),
        "upstream_commit": "d00c1bf17790fbe56424ee5567cce80d8e75f4b2",
        "upstream_mousse_py_sha256": "29cddc3b76e8beeacb973511f71b43e4152f1f203d4eb5c66ba3012002e6d149",
    }
    if args.preflight:
        stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        artifact = args.results_dir.expanduser().resolve() / f"{stamp}_preflight_seed{args.seed}.json"
        write_json(artifact, {"status": "passed", "protocol": PREFLIGHT_PROTOCOL, "official_provenance": provenance, "data": data, "runtime": runtime, "controller_runtime": controller, "wandb_readiness": wandb_readiness, "initialization_audit": init_audit, "small_matrix_reference_audit": small_audit, "source_audit": source_audit})
        print(f"R1 Mousse preflight artifact: {artifact}")
        return

    smoke_certificate = None
    if args.formal and args.resume_batch is None:
        smoke_certificate = validate_smoke(args.smoke_manifest, args, cells[0], runtime, init_audit, derived)
    stem = mode_name(args)
    if args.resume_batch:
        batch_dir = args.resume_batch.expanduser().resolve()
        plan_path = batch_dir / f"{stem}_plan.json"
        plan = resume_plan if resume_plan is not None else json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("protocol") != protocol(args) or plan.get("seed") != args.seed or plan.get("source_audit") != source_audit:
            raise RuntimeError("resume plan protocol/seed/source mismatch")
        if plan.get("cell", {}).get("cell_id") not in {cell.cell_id for cell in cells} and len(cells) == 1:
            raise RuntimeError("resume selected cell mismatch")
        batch_id = str(plan["batch_id"])
        smoke_certificate = plan.get("smoke_certificate")
        selection = plan.get("selection_certificate")
    else:
        batch_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        batch_dir = args.results_dir.expanduser().resolve() / f"{batch_id}_{stem}_seed{args.seed}"
        batch_dir.mkdir(parents=True, exist_ok=False)
        plan = {
            "family": FAMILY, "protocol": protocol(args), "batch_id": batch_id, "batch_kind": stem,
            "seed": args.seed, "total_steps": total_steps(args), "warmdown_steps": warmdown_steps(args),
            "tokens_per_step": TOKENS_PER_STEP, "total_tokens": total_steps(args) * TOKENS_PER_STEP,
            "val_every": args.val_every, "val_tokens": args.val_tokens,
            "cells": [asdict(cell) for cell in cells], "cell": asdict(cells[0]) if len(cells) == 1 else None,
            "formal_evidence": bool(args.formal), "timing_eligible": False,
            "official_provenance": provenance, "data": data, "runtime": runtime,
            "training_runtime_fingerprint": r0.runtime_fingerprint(runtime), "controller_runtime": controller,
            "wandb_readiness": wandb_readiness,
            "initialization_audit": init_audit, "small_matrix_reference_audit": small_audit,
            "source_audit": source_audit, "selection_certificate": selection,
            "smoke_certificate": smoke_certificate, "wandb_project": args.wandb_project,
        }
        write_json(batch_dir / f"{stem}_plan.json", plan)

    summaries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for cell in cells:
        try:
            summaries.append(run_cell(args, repo, data_dir, batch_dir, batch_id, cell, derived, str(init_audit["init_sha256"]), runtime))
        except Exception as exc:
            failures.append({"cell_id": cell.cell_id, "error": repr(exc)})
            break
    ranked = sorted(summaries, key=lambda row: float(row["final_val_loss"]))
    write_csv(batch_dir / f"{stem}_summary.csv", ranked)
    wandb_complete = args.formal_smoke or args.numerical_smoke or args.wandb_mode == "disabled" or all(row.get("wandb_status") == "uploaded" for row in summaries)
    final = {
        **plan,
        "status": "completed_valid" if len(summaries) == len(cells) and not failures and wandb_complete else "completed_valid_local_wandb_incomplete" if len(summaries) == len(cells) and not failures else "failed",
        "summaries": ranked, "summary": ranked[0] if len(ranked) == 1 else None,
        "failures": failures, "wandb_complete": wandb_complete,
    }
    manifest_path = batch_dir / f"{stem}_manifest.json"
    write_json(manifest_path, final)
    if args.pilot and args.seed == 2026 and args.pilot_steps == PILOT_STEPS and not failures and len(cells) == 3:
        selection_path = make_selection(batch_dir, ranked, manifest_path)
        print(f"R1 Mousse selection certificate: {selection_path}")
    print(f"R1 Mousse artifacts: {batch_dir}")
    if failures:
        raise SystemExit(1)
    if (args.pilot or args.formal) and not wandb_complete:
        raise SystemExit("local evidence is valid but W&B upload is incomplete; resume the batch")


if __name__ == "__main__":
    main()
