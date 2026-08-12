"""Run audited Moonlight/NorMuon pilots and confirmations on LLaMA/SwiGLU-124M."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
BASE_DIR = SCRIPTS_DIR / "17_llama_swiglu_validation"
EXT_DIR = SCRIPTS_DIR / "19_r1_extended_baselines"
SHARED_DIR = SCRIPTS_DIR / "_shared"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SHARED_DIR))

import run_llama_swiglu_validation as base
from project_paths import EXPERIMENT_RESULTS_ROOT


FAMILY = "23_llama_swiglu_extended_baselines"
PROTOCOL = "llama_swiglu_124m_extended_progressive_v2"
PILOT_PROJECT = "Selective-Newton-Muon-MainConf-LLaMA124M-ExtendedPilot-20260722"
FORMAL_PROJECT = "Selective-Newton-Muon-MainConf-LLaMA124M-ExtendedFormal-20260722"
TRAINER = SCRIPT_DIR / "train_llama_swiglu_extended.py"
BASE_TRAINER = BASE_DIR / "train_llama_swiglu.py"
OPTIMIZERS = EXT_DIR / "extended_optimizers.py"
FULL_STEPS = 6200
PILOT_STEPS = 1000
FULL_WARMDOWN = 1800
SMOKE_STEPS = 34
VAL_EVERY = 100
VAL_TOKENS = 10_485_760
GLOBAL_BATCH = 512
SEQUENCE_LENGTH = 1024
EXPECTED_PARAMETER_COUNT = 123_551_232
EXPECTED_MATRIX_TENSORS = 84
EXPECTED_BACKUP_TENSORS = 26


@dataclass(frozen=True)
class Cell:
    cell_id: str
    method: str
    lr_label: str
    auxiliary_lr: float
    matrix_lr: float
    weight_decay: float
    rationale: str


CELLS = (
    Cell("normuon_low", "normuon", "0.5x_r1_selected", 0.0003, 0.0050, 0.01,
         "lower LLaMA transfer cell around the R1-selected boundary"),
    Cell("normuon_r1scale", "normuon", "r1_selected", 0.0003, 0.0100, 0.01,
         "frozen R1-selected recipe transferred as the center cell"),
    Cell("normuon_official", "normuon", "official_default", 0.0003, 0.0200, 0.01,
         "public NorMuon default matrix learning rate"),
    Cell("moonlight_official", "moonlight_muon", "official_toy_default", 0.0010, 0.0010, 0.10,
         "public Moonlight toy recipe"),
    Cell("moonlight_r1scale", "moonlight_muon", "r1_selected", 0.0018, 0.0018, 0.10,
         "frozen R1-selected recipe transferred as the center cell"),
    Cell("moonlight_high", "moonlight_muon", "upper_screen", 0.0030, 0.0030, 0.10,
         "predeclared upper stability and ranking cell"),
)
CELL_BY_ID = {cell.cell_id: cell for cell in CELLS}
CENTER_CELLS = ("normuon_r1scale", "moonlight_r1scale")
FROZEN_FORMAL_CELLS = ("moonlight_high", "normuon_r1scale")


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S+0000")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def default_results_dir() -> Path:
    return EXPERIMENT_RESULTS_ROOT / FAMILY / "results"


def validate_formal_cell_ids(cell_ids: list[str] | None) -> list[str]:
    chosen = list(cell_ids or [])
    if not chosen:
        raise ValueError("formal modes require at least one explicit --cells choice")
    if len(chosen) != len(set(chosen)):
        raise ValueError("formal --cells choices must be unique")
    unknown = set(chosen) - set(CELL_BY_ID)
    if unknown:
        raise ValueError(f"unknown formal cells: {sorted(unknown)}")
    unfrozen = set(chosen) - set(FROZEN_FORMAL_CELLS)
    if unfrozen:
        raise ValueError(
            "formal modes accept only pilot-frozen cells; "
            f"unfrozen={sorted(unfrozen)}, allowed={list(FROZEN_FORMAL_CELLS)}"
        )
    methods = [CELL_BY_ID[cell_id].method for cell_id in chosen]
    if len(methods) != len(set(methods)):
        raise ValueError("formal modes accept at most one frozen cell per method")
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Progressive LLaMA-124M pilot/formal runner for frozen extended baselines"
    )
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--numerical-smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--formal-smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--cells", nargs="+", choices=tuple(CELL_BY_ID))
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        help="completed v1 six-cell pilot manifest that froze the formal cell",
    )
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--resume-batch", type=Path)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--device-batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=128)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-train-log-every", type=int, default=20)
    parser.add_argument("--wandb-init-timeout", type=int, default=120)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain unique non-negative integers")
    if args.device_batch_size <= 0 or GLOBAL_BATCH % args.device_batch_size:
        parser.error("--device-batch-size must be a positive divisor of 512")
    if VAL_TOKENS % (args.device_batch_size * SEQUENCE_LENGTH):
        parser.error("validation tokens must divide into exact device batches")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    if args.pilot and args.seeds != [2026]:
        parser.error("the LR pilot is frozen to seed2026")
    if (args.numerical_smoke or args.formal_smoke) and args.wandb_mode != "disabled":
        parser.error("smoke modes require --wandb-mode disabled")
    if (args.pilot or args.formal) and args.smoke_manifest is None and args.resume_batch is None:
        parser.error("pilot/formal requires --smoke-manifest, or --resume-batch")
    if args.formal or args.formal_smoke:
        try:
            args.cells = validate_formal_cell_ids(args.cells)
        except ValueError as exc:
            parser.error(str(exc))
        if args.pilot_manifest is None:
            parser.error("formal modes require --pilot-manifest")
    if args.resume_batch is not None and (args.dry_run or args.preflight):
        parser.error("--resume-batch is not valid with dry-run/preflight")
    if args.wandb_project is None:
        args.wandb_project = FORMAL_PROJECT if args.formal else PILOT_PROJECT
    return args


def selected_cells(args: argparse.Namespace) -> list[Cell]:
    if args.cells:
        return [CELL_BY_ID[cell] for cell in args.cells]
    if args.dry_run:
        return list(CELLS)
    if args.numerical_smoke:
        return [CELL_BY_ID[cell] for cell in CENTER_CELLS]
    if args.pilot:
        return list(CELLS)
    return [CELL_BY_ID[cell] for cell in CENTER_CELLS]


def kind(args: argparse.Namespace) -> str:
    for value in ("dry_run", "preflight", "numerical_smoke", "pilot", "formal_smoke", "formal"):
        if getattr(args, value):
            return value
    raise AssertionError("mode missing")


def source_bundle() -> dict[str, Any]:
    files = {
        "runner": Path(__file__).resolve(),
        "adapter": TRAINER.resolve(),
        "base_trainer": BASE_TRAINER.resolve(),
        "extended_optimizers": OPTIMIZERS.resolve(),
    }
    hashes = {name: sha256_file(path) for name, path in files.items()}
    return {"files": {name: str(path) for name, path in files.items()}, "sha256": hashes,
            "bundle_sha256": canonical_hash(hashes)}


def subprocess_env(official_repo: Path) -> dict[str, str]:
    return base.subprocess_env(official_repo)


def runtime_and_data(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = base.validate_runtime(args.python_exe, args.official_repo)
    data = base.audit_data(args.official_repo / "data" / "fineweb10B")
    return runtime, data


def validate_pilot_selection(
    path: Path,
    args: argparse.Namespace,
    runtime: dict[str, Any],
    data: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    failures: list[str] = []
    if payload.get("status") != "completed" or payload.get("kind") != "pilot":
        failures.append("selection certificate is not a completed pilot")
    if payload.get("protocol") != "llama_swiglu_124m_extended_progressive_v1":
        failures.append("selection certificate is not the frozen v1 six-cell pilot")
    if payload.get("seeds") != [2026] or payload.get("failed_tasks") != {}:
        failures.append("pilot seed/failure state differs from the frozen selection run")
    if payload.get("data", {}).get("fingerprint") != data["fingerprint"]:
        failures.append("FineWeb data differs from the selection pilot")
    if base.stable_runtime(payload.get("runtime", {})) != base.stable_runtime(runtime):
        failures.append("runtime differs from the selection pilot")
    historical_hashes = payload.get("source_bundle", {}).get("sha256", {})
    for implementation in ("adapter", "base_trainer", "extended_optimizers"):
        if historical_hashes.get(implementation) != bundle["sha256"].get(implementation):
            failures.append(f"{implementation} differs from the selection pilot")
    pilot_cells = {row.get("cell_id"): row for row in payload.get("cells", [])}
    completed = {row.get("cell_id"): row for row in payload.get("completed_tasks", [])}
    for cell in selected_cells(args):
        observed = pilot_cells.get(cell.cell_id)
        if observed != asdict(cell):
            failures.append(f"{cell.cell_id} recipe differs from the selection pilot")
            continue
        task = completed.get(cell.cell_id)
        if not isinstance(task, dict):
            failures.append(f"{cell.cell_id} has no completed pilot task")
            continue
        summary_path = Path(str(task.get("summary_path", "")))
        if not summary_path.is_file():
            failures.append(f"{cell.cell_id} pilot summary is missing: {summary_path}")
            continue
        if sha256_file(summary_path) != task.get("summary_sha256"):
            failures.append(f"{cell.cell_id} pilot summary hash differs")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        extended = summary.get("extended_optimizer", {})
        recipe = (
            extended.get("method"),
            extended.get("auxiliary_lr"),
            extended.get("matrix_lr"),
            extended.get("weight_decay"),
        )
        expected = (cell.method, cell.auxiliary_lr, cell.matrix_lr, cell.weight_decay)
        if (
            summary.get("status") != "completed"
            or summary.get("completed_steps") != PILOT_STEPS
            or summary.get("seed") != 2026
            or summary.get("resume_count") != 0
            or summary.get("checkpoint_path") != ""
            or recipe != expected
        ):
            failures.append(f"{cell.cell_id} pilot summary contract differs")
    if failures:
        raise RuntimeError(
            "extended LLaMA pilot-selection certificate rejected:\n- "
            + "\n- ".join(failures)
        )
    return payload


def train_command(
    args: argparse.Namespace,
    cell: Cell,
    seed: int,
    output_dir: Path,
    run_kind: str,
) -> list[str]:
    smoke = run_kind in {"numerical_smoke", "formal_smoke"}
    pilot = run_kind == "pilot"
    steps = SMOKE_STEPS if smoke else PILOT_STEPS if pilot else FULL_STEPS
    warmdown = 1 if smoke or pilot else FULL_WARMDOWN
    val_every = steps if smoke else VAL_EVERY
    val_tokens = args.device_batch_size * SEQUENCE_LENGTH if smoke else VAL_TOKENS
    command = [
        str(args.python_exe), str(TRAINER),
        "--method", cell.method,
        "--data-dir", str(args.official_repo / "data" / "fineweb10B"),
        "--output-dir", str(output_dir),
        "--seed", str(seed),
        "--num-iterations", str(steps),
        "--global-batch-size", str(GLOBAL_BATCH),
        "--device-batch-size", str(args.device_batch_size),
        "--sequence-length", str(SEQUENCE_LENGTH),
        "--val-every", str(val_every),
        "--val-tokens", str(val_tokens),
        "--warmdown-iters", str(warmdown),
        "--backup-lr", repr(cell.auxiliary_lr),
        "--matrix-lr", repr(cell.matrix_lr),
        "--adamw-matrix-lr", repr(cell.matrix_lr),
        "--extended-weight-decay", repr(cell.weight_decay),
        "--checkpoint-every", "0" if smoke or pilot else str(args.checkpoint_every),
        "--resume", "never" if smoke or pilot else "auto",
    ]
    if smoke or pilot:
        command.append("--no-save-final")
    return command


def run_init_audit(args: argparse.Namespace, cells: list[Cell]) -> dict[str, Any]:
    observed: dict[str, dict[str, Any]] = {}
    representatives: dict[str, Cell] = {}
    for cell in cells:
        representatives.setdefault(cell.method, cell)
    for seed in args.seeds:
        for method, cell in representatives.items():
            command = train_command(args, cell, seed, args.results_dir / "_init_unused", "numerical_smoke")
            command.extend(["--init-only"])
            result = subprocess.run(command, env=subprocess_env(args.official_repo), text=True, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"init audit failed for {method}/seed{seed}:\n{result.stdout}\n{result.stderr}")
            lines = [line for line in result.stdout.splitlines() if line.startswith("LLAMA_INIT_AUDIT ")]
            if len(lines) != 1:
                raise RuntimeError(f"no unique init payload for {method}/seed{seed}:\n{result.stdout}")
            observed[f"{method}:seed{seed}"] = json.loads(lines[0].split(" ", 1)[1])
    for seed in args.seeds:
        hashes = {observed[f"{method}:seed{seed}"]["init_sha256"] for method in representatives}
        if len(hashes) != 1:
            raise RuntimeError(f"method initializations differ for seed{seed}: {hashes}")
    for key, payload in observed.items():
        audit = payload["architecture"]
        expected = (EXPECTED_PARAMETER_COUNT, EXPECTED_MATRIX_TENSORS, EXPECTED_BACKUP_TENSORS)
        actual = (audit["parameter_count"], audit["matrix_tensor_count"], audit["backup_tensor_count"])
        if actual != expected or not audit["embedding_head_tied"] or audit["preconditioner_group_count"] != 0:
            raise RuntimeError(f"architecture/routing audit failed for {key}: {audit}")
    by_seed = {
        str(seed): observed[f"{next(iter(representatives))}:seed{seed}"]["init_sha256"]
        for seed in args.seeds
    }
    return {
        "common_init_sha256_by_seed": by_seed,
        "observed": observed,
        "routing": {
            "matrix_tensor_count": EXPECTED_MATRIX_TENSORS,
            "backup_tensor_count": EXPECTED_BACKUP_TENSORS,
            "matrix_rule": "all 2D parameters except tied token embedding",
            "backup_rule": "tied token embedding plus 25 RMSNorm gains",
            "qkv_layout": "separate; packed-QKV split is not applied",
        },
    }


def validate_smoke(
    path: Path,
    args: argparse.Namespace,
    runtime: dict[str, Any],
    data: dict[str, Any],
    bundle: dict[str, Any],
    pilot_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    failures: list[str] = []
    if payload.get("status") != "completed" or payload.get("kind") not in {"numerical_smoke", "formal_smoke"}:
        failures.append("certificate is not a completed smoke batch")
    if payload.get("source_bundle", {}).get("bundle_sha256") != bundle["bundle_sha256"]:
        failures.append("source bundle differs")
    if payload.get("data", {}).get("fingerprint") != data["fingerprint"]:
        failures.append("FineWeb data differs")
    if base.stable_runtime(payload.get("runtime", {})) != base.stable_runtime(runtime):
        failures.append("runtime differs")
    completed_methods = {row["method"] for row in payload.get("completed_tasks", [])}
    if args.pilot:
        if payload.get("kind") != "numerical_smoke":
            failures.append("pilot requires a numerical-smoke certificate")
        if completed_methods != {"normuon", "moonlight_muon"}:
            failures.append("pilot smoke did not cover both optimizer implementations")
    if args.formal:
        if payload.get("kind") != "formal_smoke":
            failures.append("formal requires a formal-smoke certificate")
        if payload.get("pilot_selection_certificate_sha256") != canonical_hash(
            pilot_selection
        ):
            failures.append("formal smoke is not bound to the requested selection pilot")
        expected_tasks = {(seed, cell.cell_id) for seed in args.seeds for cell in selected_cells(args)}
        observed_tasks = {(row["seed"], row["cell_id"]) for row in payload.get("completed_tasks", [])}
        if expected_tasks != observed_tasks:
            failures.append("formal-smoke seeds/cells differ from formal plan")
    if failures:
        raise RuntimeError("extended LLaMA smoke certificate rejected:\n- " + "\n- ".join(failures))
    return payload


def tee_process(command: list[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND " + json.dumps(command) + "\n")
        handle.flush()
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return process.wait()


def validate_summary(
    path: Path, cell: Cell, seed: int, steps: int, val_every: int,
    expected_init: str, runtime: dict[str, Any], require_checkpoint: bool,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if payload.get("status") != "completed" or payload.get("method") != cell.method:
        failures.append("status/method mismatch")
    if payload.get("seed") != seed or payload.get("completed_steps") != steps:
        failures.append("seed/step mismatch")
    if payload.get("init_sha256") != expected_init:
        failures.append("initialization fingerprint mismatch")
    if payload.get("architecture", {}).get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        failures.append("parameter count mismatch")
    extended = payload.get("extended_optimizer", {})
    expected_recipe = (cell.method, cell.auxiliary_lr, cell.matrix_lr, cell.weight_decay)
    actual_recipe = (extended.get("method"), extended.get("auxiliary_lr"),
                     extended.get("matrix_lr"), extended.get("weight_decay"))
    if actual_recipe != expected_recipe:
        failures.append(f"optimizer recipe mismatch: {actual_recipe} != {expected_recipe}")
    if payload.get("k_state_bytes") != 0:
        failures.append("extended optimizer unexpectedly allocated Newton K state")
    if base.stable_runtime(payload.get("runtime", {})) != base.stable_runtime(runtime):
        failures.append("runtime differs from controller")
    for field in ("final_val_loss", "best_val_loss", "final_train_loss"):
        value = payload.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"{field} is non-finite")
    try:
        base.validate_metric_evidence(path.with_name("metrics.csv"), total_steps=steps,
                                      val_every=val_every, global_batch_size=GLOBAL_BATCH,
                                      sequence_length=SEQUENCE_LENGTH)
    except Exception as exc:
        failures.append(f"metric evidence invalid: {exc}")
    checkpoint = Path(str(payload.get("checkpoint_path", "")))
    if require_checkpoint and (not str(payload.get("checkpoint_path", "")) or not checkpoint.is_file()):
        failures.append("formal run has no checkpoint")
    if not require_checkpoint and str(payload.get("checkpoint_path", "")):
        failures.append("smoke/pilot unexpectedly retained a checkpoint")
    if failures:
        raise RuntimeError(f"summary validation failed for {cell.cell_id}/seed{seed}:\n- " + "\n- ".join(failures))
    return payload


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def upload_wandb(args: argparse.Namespace, batch_id: str, cell: Cell, seed: int,
                 run_dir: Path, summary: dict[str, Any], run_kind: str) -> dict[str, Any]:
    if args.wandb_mode == "disabled":
        return {"status": "disabled", "mode": "disabled"}
    import wandb

    identity = {
        "family": "llama_swiglu_124m_extended_baselines",
        "protocol": PROTOCOL,
        "kind": run_kind,
        "cell_id": cell.cell_id,
        "method": cell.method,
        "seed": seed,
        "auxiliary_lr": cell.auxiliary_lr,
        "matrix_lr": cell.matrix_lr,
        "weight_decay": cell.weight_decay,
        "selection_role": "lr_screen" if run_kind == "pilot" else "frozen_config_confirmation",
        "tuning_seed": seed == 2026,
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    run_name = f"llama124m_ext_{run_kind}_{cell.cell_id}_seed{seed}"
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=run_id,
                     resume="allow", name=run_name, group=f"llama124m_ext_{run_kind}_seed{seed}",
                     mode=args.wandb_mode, config=identity | {"batch_id": batch_id},
                     tags=["llama", "swiglu", "124m", "extended-baseline", run_kind],
                     settings=wandb.Settings(init_timeout=args.wandb_init_timeout))
    try:
        metrics = read_metrics(run_dir / "metrics.csv")
        last_train = max(int(row["step"]) for row in metrics if row["event"] == "train")
        per_step: dict[int, dict[str, float]] = {}
        for row in metrics:
            step = int(row["step"])
            if row["event"] == "train" and step % args.wandb_train_log_every and step != last_train:
                continue
            values = per_step.setdefault(step, {})
            values["train/loss_step" if row["event"] == "train" else "val/loss"] = float(row["loss"])
            values.update({"time/train_s": float(row["train_s"]),
                           "performance/step_avg_ms": float(row["step_avg_ms"]),
                           "lr/auxiliary": float(row["lr_backup"]),
                           "lr/matrix": float(row["lr_matrix"]),
                           "tokens/seen": float(row["tokens_seen"])})
        for step in sorted(per_step):
            wandb.log(per_step[step], step=step)
        run.summary.update({key: summary[key] for key in (
            "final_val_loss", "best_val_loss", "final_train_loss", "peak_allocated_mib",
            "optimizer_state_bytes", "train_s", "step_avg_ms", "resume_count", "timing_comparable"
        )})
    finally:
        run.finish()
    return {"status": "uploaded", "project": args.wandb_project, "run_id": run_id,
            "run_name": run_name, "uploaded_at": now_iso()}


def plan_payload(args: argparse.Namespace, run_kind: str, cells: list[Cell],
                 runtime: dict[str, Any], data: dict[str, Any], init: dict[str, Any],
                 bundle: dict[str, Any], smoke: dict[str, Any] | None,
                 pilot_selection: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "family": FAMILY, "protocol": PROTOCOL, "kind": run_kind,
        "created_at": now_iso(), "seeds": args.seeds,
        "cells": [asdict(cell) for cell in cells],
        "config": {"global_batch_size": GLOBAL_BATCH, "device_batch_size": args.device_batch_size,
                   "sequence_length": SEQUENCE_LENGTH, "steps": SMOKE_STEPS if "smoke" in run_kind else PILOT_STEPS if run_kind == "pilot" else FULL_STEPS,
                   "warmdown_steps": 1 if run_kind in {"numerical_smoke", "formal_smoke", "pilot"} else FULL_WARMDOWN,
                   "val_every": SMOKE_STEPS if "smoke" in run_kind else VAL_EVERY,
                   "val_tokens": args.device_batch_size * SEQUENCE_LENGTH if "smoke" in run_kind else VAL_TOKENS},
        "selection_contract": {
            "pilot_seed": 2026,
            "primary": "lowest finite validation loss at step 1000 within each method",
            "tie_break": "if final losses differ by <=0.002, prefer lower mean of the last three validation points; normalized AUC is secondary",
            "pilot_decision": {
                "moonlight_high": "advance to 6200-step architecture gate",
                "normuon_r1scale": "archival only; stopped unless explicitly reopened for reviewer request",
            },
            "formal_rule": "one or more explicitly requested pilot-frozen cells, at most one per method; no LR reselection on seeds2024/2025",
            "fairness": "core Newton trio remains frozen at shared LR; tuning budgets and recipe sources are disclosed",
        },
        "official_repo": str(args.official_repo.resolve()), "python_exe": str(args.python_exe),
        "runtime": runtime, "data": data, "init_audit": init, "source_bundle": bundle,
        "pilot_selection_manifest": str(args.pilot_manifest.resolve()) if args.pilot_manifest else None,
        "pilot_selection_certificate_sha256": canonical_hash(pilot_selection) if pilot_selection else None,
        "smoke_certificate_sha256": canonical_hash(smoke) if smoke else None,
        "wandb_project": args.wandb_project, "wandb_mode": args.wandb_mode,
    }


def main() -> None:
    args = parse_args()
    cells = selected_cells(args)
    run_kind = kind(args)
    bundle = source_bundle()
    if args.dry_run:
        print(json.dumps({"protocol": PROTOCOL, "kind": run_kind, "seeds": args.seeds,
                          "cells": [asdict(cell) for cell in cells], "source_bundle": bundle,
                          "note": "dry-run does not access CUDA or FineWeb"}, indent=2))
        return

    runtime, data = runtime_and_data(args)
    init = run_init_audit(args, cells)
    if args.preflight:
        artifact = args.results_dir.resolve() / f"{timestamp_id()}_preflight.json"
        atomic_json(artifact, {"status": "passed", "created_at": now_iso(), "runtime": runtime,
                               "data": data, "init_audit": init, "source_bundle": bundle,
                               "cells": [asdict(cell) for cell in cells]})
        print(f"LLaMA extended preflight: {artifact}")
        return

    pilot_selection = None
    if args.formal or args.formal_smoke:
        pilot_selection = validate_pilot_selection(
            args.pilot_manifest, args, runtime, data, bundle
        )
    smoke_certificate = None
    if args.pilot or args.formal:
        smoke_certificate = validate_smoke(
            args.smoke_manifest,
            args,
            runtime,
            data,
            bundle,
            pilot_selection,
        )
    if args.wandb_mode == "online" and (args.pilot or args.formal):
        try:
            import wandb
            viewer = wandb.Api(timeout=30).viewer
            if callable(viewer):
                viewer = viewer()
            if not viewer:
                raise RuntimeError("W&B returned no authenticated viewer")
        except Exception as exc:
            raise RuntimeError(f"W&B online readiness failed before training: {exc!r}") from exc

    if args.resume_batch:
        batch_dir = args.resume_batch.resolve()
        manifest_path = batch_dir / "llama_extended_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = plan_payload(
            args,
            run_kind,
            cells,
            runtime,
            data,
            init,
            bundle,
            smoke_certificate,
            pilot_selection,
        )
        for key in ("kind", "seeds", "cells", "config", "runtime", "data", "source_bundle"):
            if manifest.get(key) != expected.get(key):
                raise RuntimeError(f"resume plan differs in {key}")
        batch_id = manifest["batch_id"]
    else:
        batch_id = f"{timestamp_id()}_{run_kind}"
        batch_dir = args.results_dir.resolve() / batch_id
        batch_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = batch_dir / "llama_extended_manifest.json"
        manifest = plan_payload(
            args,
            run_kind,
            cells,
            runtime,
            data,
            init,
            bundle,
            smoke_certificate,
            pilot_selection,
        )
        manifest.update({"batch_id": batch_id, "status": "running", "completed_tasks": [], "failed_tasks": {}})
        atomic_json(manifest_path, manifest)

    smoke_mode = run_kind in {"numerical_smoke", "formal_smoke"}
    steps = SMOKE_STEPS if smoke_mode else PILOT_STEPS if run_kind == "pilot" else FULL_STEPS
    val_every = steps if smoke_mode else VAL_EVERY
    completed = {(row["seed"], row["cell_id"]): row for row in manifest.get("completed_tasks", [])}
    failures = dict(manifest.get("failed_tasks", {}))
    tasks = [(seed, cell) for seed in args.seeds for cell in cells]
    for index, (seed, cell) in enumerate(tasks, start=1):
        key = (seed, cell.cell_id)
        run_dir = batch_dir / f"{index:02d}_{cell.cell_id}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            summary_path = run_dir / "summary.json"
            if summary_path.is_file():
                summary = validate_summary(summary_path, cell, seed, steps, val_every,
                                           init["common_init_sha256_by_seed"][str(seed)], runtime,
                                           require_checkpoint=run_kind == "formal")
                print(f"Skipping completed local task: {cell.cell_id}/seed{seed}")
            else:
                command = train_command(args, cell, seed, run_dir, run_kind)
                code = tee_process(command, subprocess_env(args.official_repo), run_dir / "terminal.log")
                if code:
                    raise RuntimeError(f"training exited with code {code}")
                summary = validate_summary(summary_path, cell, seed, steps, val_every,
                                           init["common_init_sha256_by_seed"][str(seed)], runtime,
                                           require_checkpoint=run_kind == "formal")
            summary["cell_id"] = cell.cell_id
            summary["selection_role"] = "lr_screen" if run_kind == "pilot" else "frozen_config_confirmation" if run_kind == "formal" else "smoke"
            atomic_json(summary_path, summary)
            wandb_path = run_dir / "wandb_upload.json"
            if smoke_mode:
                upload = {"status": "disabled_for_smoke"}
            elif wandb_path.is_file() and json.loads(wandb_path.read_text(encoding="utf-8")).get("status") == "uploaded":
                upload = json.loads(wandb_path.read_text(encoding="utf-8"))
            else:
                upload = upload_wandb(args, batch_id, cell, seed, run_dir, summary, run_kind)
                atomic_json(wandb_path, upload)
            completed[key] = {"seed": seed, "cell_id": cell.cell_id, "method": cell.method,
                              "summary_path": str(summary_path), "summary_sha256": sha256_file(summary_path),
                              "wandb": upload}
            failures.pop(f"{cell.cell_id}:seed{seed}", None)
        except Exception as exc:
            failures[f"{cell.cell_id}:seed{seed}"] = repr(exc)
            if not args.continue_on_error:
                manifest.update({"completed_tasks": list(completed.values()), "failed_tasks": failures,
                                 "status": "incomplete", "last_updated_at": now_iso()})
                atomic_json(manifest_path, manifest)
                print(f"LLaMA extended artifacts: {batch_dir}")
                raise
        manifest.update({"completed_tasks": list(completed.values()), "failed_tasks": failures,
                         "last_updated_at": now_iso()})
        atomic_json(manifest_path, manifest)

    all_complete = len(completed) == len(tasks) and not failures
    manifest.update({"status": "completed" if all_complete else "incomplete",
                     "completed_tasks": list(completed.values()), "failed_tasks": failures,
                     "completed_at": now_iso()})
    atomic_json(manifest_path, manifest)
    print(f"LLaMA extended artifacts: {batch_dir}")
    print(f"LLaMA extended manifest:  {manifest_path}")
    if not all_complete:
        raise RuntimeError(f"extended batch incomplete: {failures}")


if __name__ == "__main__":
    main()
