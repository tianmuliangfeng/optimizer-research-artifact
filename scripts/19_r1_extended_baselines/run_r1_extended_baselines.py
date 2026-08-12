"""Audit, screen, and formally run R1 AdamW/NorMuon/Moonlight-Muon baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
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
sys.path.insert(0, str(R0_DIR))
sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import run_official_newton_muon_r0 as r0
from extended_source_builder import ALLOWED_METHODS, DerivedSource, build_source
from project_paths import EXPERIMENT_RESULTS_ROOT


FAMILY = "19_r1_extended_baselines"
PILOT_PROTOCOL = "r1_extended_baselines_short_lr_pilot_v1"
FORMAL_SMOKE_PROTOCOL = "r1_extended_baselines_selected_exact_shape_smoke_v1"
FORMAL_PROTOCOL = "r1_extended_baselines_selected_6200step_v1"
AUDIT_PROTOCOL = "r1_extended_baselines_implementation_audit_v1"
PILOT_PROJECT = "Selective-Newton-Muon-MainConf-R1-ExtendedPilot-20260721"
FORMAL_PROJECT = "Selective-Newton-Muon-MainConf-R1-ExtendedFormal-20260721"
PILOT_RUN_PREFIX = "mainconf_r1_extended_pilot"
FORMAL_RUN_PREFIX = "mainconf_r1_extended_formal"
TOKENS_PER_STEP = 512 * 1024
FULL_STEPS = 6200
FULL_WARMDOWN = 1800
DEFAULT_PILOT_STEPS = 1000
DEFAULT_FORMAL_SMOKE_STEPS = 34
DEFAULT_VAL_EVERY = 100
DEFAULT_VAL_TOKENS = 10_485_760
JSON_PREFIXES = (
    "R1X_METADATA ",
    "R1X_ROUTING ",
    "R1X_HYPERPARAMS ",
    "R1X_FINAL_MEMORY ",
)


@dataclass(frozen=True)
class PilotCell:
    cell_id: str
    method: str
    lr_label: str
    auxiliary_lr: float
    matrix_lr: float
    weight_decay: float
    source_authority: str
    interpretation: str


PILOT_CELLS = (
    PilotCell(
        "adamw_low",
        "adamw",
        "0.75x_official",
        0.0027,
        0.000432,
        0.0,
        "pinned train_gpt_adam_1.py; hidden LR is 0.16x head LR",
        "official AdamW recipe with a lower base LR",
    ),
    PilotCell(
        "adamw_official",
        "adamw",
        "official",
        0.0036,
        0.000576,
        0.0,
        "pinned train_gpt_adam_1.py; hidden LR is 0.16x head LR",
        "exact official AdamW learning rates and zero weight decay",
    ),
    PilotCell(
        "adamw_high",
        "adamw",
        "1.25x_official",
        0.0045,
        0.000720,
        0.0,
        "pinned train_gpt_adam_1.py; hidden LR is 0.16x head LR",
        "official AdamW recipe with a higher base LR",
    ),
    PilotCell(
        "normuon_r1scale",
        "normuon",
        "r1_muon_effective_scale",
        0.0003,
        0.010,
        0.01,
        "zichongli5/NorMuon official SingleDeviceNorMuonWithAuxAdam",
        "matrix LR near R1 Muon's square-matrix effective update scale",
    ),
    PilotCell(
        "normuon_official",
        "normuon",
        "official_default",
        0.0003,
        0.020,
        0.01,
        "zichongli5/NorMuon official SingleDeviceNorMuonWithAuxAdam",
        "official NorMuon default LR and recommended auxiliary AdamW LR",
    ),
    PilotCell(
        "normuon_high",
        "normuon",
        "1.5x_official",
        0.0003,
        0.030,
        0.01,
        "zichongli5/NorMuon official SingleDeviceNorMuonWithAuxAdam",
        "upper stability/ranking cell",
    ),
    PilotCell(
        "moonlight_official",
        "moonlight_muon",
        "official_toy_default",
        0.0010,
        0.0010,
        0.1,
        "MoonshotAI/Moonlight examples/toy_train.py",
        "official public toy LR/decay with Moonlight RMS scaling",
    ),
    PilotCell(
        "moonlight_r1scale",
        "moonlight_muon",
        "r1_muon_effective_scale",
        0.0018,
        0.0018,
        0.1,
        "MoonshotAI/Moonlight examples/toy_train.py",
        "base LR chosen to approximately match R1 Muon on 768x768 matrices",
    ),
    PilotCell(
        "moonlight_high",
        "moonlight_muon",
        "upper_screen",
        0.0030,
        0.0030,
        0.1,
        "MoonshotAI/Moonlight examples/toy_train.py",
        "upper stability/ranking cell",
    ),
)


CENTER_CELL_IDS = ("adamw_official", "normuon_official", "moonlight_official")
FORMAL_CELL_IDS = ("adamw_low", "normuon_r1scale", "moonlight_r1scale")


def is_formal_profile(args: argparse.Namespace) -> bool:
    return bool(args.formal or args.formal_smoke)


def is_smoke(args: argparse.Namespace) -> bool:
    return bool(args.numerical_smoke or args.formal_smoke)


def protocol(args: argparse.Namespace) -> str:
    if args.formal:
        return FORMAL_PROTOCOL
    if args.formal_smoke:
        return FORMAL_SMOKE_PROTOCOL
    return PILOT_PROTOCOL


def batch_kind(args: argparse.Namespace) -> str:
    if args.formal:
        return "formal"
    if args.formal_smoke:
        return "formal_smoke"
    if args.numerical_smoke:
        return "pilot_smoke"
    return "pilot"


def artifact_stem(args: argparse.Namespace) -> str:
    return batch_kind(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit, short-pilot, and formally run extended optimizers on R1."
    )
    parser.add_argument("--official-repo", type=Path, default=r0.default_official_repo())
    parser.add_argument("--python-exe", default="python")
    parser.add_argument("--seed", type=int, default=2026)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--numerical-smoke", action="store_true")
    mode.add_argument("--formal-smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--pilot-steps", type=int, default=DEFAULT_PILOT_STEPS)
    parser.add_argument("--val-every", type=int, default=DEFAULT_VAL_EVERY)
    parser.add_argument("--val-tokens", type=int, default=DEFAULT_VAL_TOKENS)
    parser.add_argument("--methods", nargs="+", choices=ALLOWED_METHODS, default=list(ALLOWED_METHODS))
    parser.add_argument("--cells", nargs="+", choices=tuple(cell.cell_id for cell in PILOT_CELLS))
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT_RESULTS_ROOT / FAMILY / "results")
    parser.add_argument("--resume-batch", type=Path)
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-init-timeout", type=int, default=120)
    parser.add_argument("--wandb-train-log-every", type=int, default=20)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.smoke_steps is None:
        args.smoke_steps = DEFAULT_FORMAL_SMOKE_STEPS if args.formal_smoke else 10
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.smoke_steps < 2 or args.pilot_steps < 2:
        parser.error("smoke/pilot steps must be at least 2")
    if args.formal_smoke and args.smoke_steps < DEFAULT_FORMAL_SMOKE_STEPS:
        parser.error(f"--formal-smoke requires at least {DEFAULT_FORMAL_SMOKE_STEPS} steps")
    if args.val_every <= 0:
        parser.error("--val-every must be positive")
    if args.val_tokens <= 0 or args.val_tokens % (64 * 1024) != 0:
        parser.error("--val-tokens must be a positive multiple of 65536")
    if len(args.methods) != len(set(args.methods)):
        parser.error("--methods contains duplicates")
    if is_formal_profile(args) and set(args.methods) != set(ALLOWED_METHODS):
        parser.error("formal and formal-smoke profiles require all three frozen methods")
    if args.cells and not (args.pilot or args.numerical_smoke):
        parser.error("--cells is only valid with --pilot or --numerical-smoke")
    if args.resume_batch is not None and args.preflight:
        parser.error("--resume-batch is not valid with --preflight")
    if args.smoke_manifest is not None and not args.formal:
        parser.error("--smoke-manifest is only valid with --formal")
    if args.formal and args.smoke_manifest is None and args.resume_batch is None:
        parser.error("a new --formal batch requires --smoke-manifest from the same seed")
    if args.wandb_project is None:
        args.wandb_project = FORMAL_PROJECT if is_formal_profile(args) else PILOT_PROJECT
    return args


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

        api = wandb.Api(timeout=30)
        viewer = api.viewer
        if callable(viewer):
            viewer = viewer()
        if not viewer:
            raise RuntimeError("W&B returned no authenticated viewer")
        return {
            "required": True,
            "status": "authenticated_online",
            "base_url": os.environ.get("WANDB_BASE_URL", "default"),
        }
    except Exception as exc:
        raise RuntimeError(
            "W&B online readiness failed before training; fix login/network first: "
            f"{exc!r}"
        ) from exc


def selected_cells(args: argparse.Namespace) -> list[PilotCell]:
    methods = set(args.methods)
    if is_formal_profile(args):
        wanted_ids = set(FORMAL_CELL_IDS)
    elif args.numerical_smoke:
        wanted_ids = set(CENTER_CELL_IDS)
    elif args.cells:
        wanted_ids = set(args.cells)
    else:
        wanted_ids = {cell.cell_id for cell in PILOT_CELLS}
    cells = [cell for cell in PILOT_CELLS if cell.method in methods and cell.cell_id in wanted_ids]
    if not cells:
        raise RuntimeError("the method/cell selection is empty")
    return cells


def warmdown_steps(args: argparse.Namespace | int) -> int:
    # A short pilot is a prefix of the formal schedule: every actual pilot
    # update uses the plateau LR, exactly as steps 1..1000 do in formal R1.
    # The one-step terminal warmdown only changes the scheduler after the last
    # update and therefore does not alter that prefix.
    # Retain the integer form for the original pilot tests/callers.
    return FULL_WARMDOWN if not isinstance(args, int) and args.formal else 1


def total_steps(args: argparse.Namespace) -> int:
    if args.formal:
        return FULL_STEPS
    if is_smoke(args):
        return args.smoke_steps
    return args.pilot_steps


def build_all_sources(repo: Path) -> dict[str, DerivedSource]:
    return {method: build_source(repo, method) for method in ALLOWED_METHODS}


def controlled_env(
    args: argparse.Namespace,
    data_dir: Path,
    cell: PilotCell,
    *,
    init_only: bool = False,
) -> dict[str, str]:
    steps = total_steps(args)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": str(args.seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "R1X_METHOD": cell.method,
            "R1X_SEED": str(args.seed),
            "R1X_DATA_DIR": str(data_dir.resolve()),
            "R1X_TOTAL_STEPS": str(steps),
            "R1X_WARMDOWN_STEPS": str(warmdown_steps(args)),
            "R1X_VAL_EVERY": str(args.val_every),
            "R1X_VAL_TOKENS": str(args.val_tokens),
            "R1X_AUX_LR": repr(cell.auxiliary_lr),
            "R1X_MATRIX_LR": repr(cell.matrix_lr),
            "R1X_WEIGHT_DECAY": repr(cell.weight_decay),
            "R1X_INIT_ONLY": "1" if init_only else "0",
            "R1X_DISABLE_CHECKPOINT": "0" if args.formal else "1",
        }
    )
    return env


def materialize_source(
    directory: Path,
    repo: Path,
    derived: DerivedSource,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    script_path = directory / f"train_r1x_{derived.method}.py"
    script_path.write_text(derived.source, encoding="utf-8", newline="\n")
    shutil.copy2(repo / "triton_kernels.py", directory / "triton_kernels.py")
    shutil.copy2(SCRIPT_DIR / "extended_optimizers.py", directory / "extended_optimizers.py")
    return script_path


def parse_prefixed_json(stdout: str, prefix: str) -> dict[str, object]:
    matching = [line[len(prefix) :] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(matching) != 1:
        raise RuntimeError(f"expected one {prefix.strip()} line, observed {len(matching)}")
    payload = json.loads(matching[0])
    if not isinstance(payload, dict):
        raise RuntimeError(f"{prefix.strip()} payload is not an object")
    return payload


def initialization_audit(
    args: argparse.Namespace,
    repo: Path,
    data_dir: Path,
    built: dict[str, DerivedSource],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    routing_records: list[dict[str, object]] = []
    center = {cell.method: cell for cell in PILOT_CELLS if cell.cell_id in CENTER_CELL_IDS}
    with tempfile.TemporaryDirectory(prefix="r1x_init_audit_") as temp:
        root = Path(temp)
        for method in ALLOWED_METHODS:
            workspace = root / method
            script = materialize_source(workspace, repo, built[method])
            result = subprocess.run(
                [args.python_exe, script.name],
                cwd=workspace,
                env=controlled_env(args, data_dir, center[method], init_only=True),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=900,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"initialization audit failed for {method}:\n{result.stdout[-8000:]}")
            metadata = parse_prefixed_json(result.stdout, "R1X_METADATA ")
            routing = parse_prefixed_json(result.stdout, "R1X_ROUTING ")
            if metadata.get("method") != method or metadata.get("seed") != args.seed:
                raise RuntimeError(f"bad initialization metadata for {method}: {metadata}")
            records.append(metadata)
            routing_records.append(routing)
            print(f"Initialization audit {method}: {metadata['init_sha256']}")
    fingerprints = {str(record["init_sha256"]) for record in records}
    if len(fingerprints) != 1:
        raise RuntimeError(f"initial model fingerprints differ: {records}")
    routing_fingerprints = {
        hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        for record in routing_records
    }
    if len(routing_fingerprints) != 1:
        raise RuntimeError("parameter routing differs across derived sources")
    routing = routing_records[0]
    if routing.get("hidden_matrix_tensors") != 48 or routing.get("packed_qkv_tensors") != 12:
        raise RuntimeError(f"unexpected R1 hidden/QKV routing: {routing}")
    return {
        "seed": args.seed,
        "all_methods_identical": True,
        "init_sha256": next(iter(fingerprints)),
        "routing_identical": True,
        "routing": routing,
        "records": records,
    }


def single_step_audit(args: argparse.Namespace, repo: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="r1x_step_audit_") as temp:
        workspace = Path(temp)
        shutil.copy2(repo / "triton_kernels.py", workspace / "triton_kernels.py")
        shutil.copy2(SCRIPT_DIR / "extended_optimizers.py", workspace / "extended_optimizers.py")
        code = (
            "import json; "
            "from extended_optimizers import run_single_step_reference_audit; "
            "print('R1X_STEP_AUDIT ' + json.dumps(run_single_step_reference_audit('cuda'), sort_keys=True))"
        )
        result = subprocess.run(
            [args.python_exe, "-c", code],
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=900,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"single-step optimizer audit failed:\n{result.stdout[-12000:]}")
        return parse_prefixed_json(result.stdout, "R1X_STEP_AUDIT ")


def validation_steps(steps: int, every: int) -> list[int]:
    values = list(range(0, steps + 1, every))
    if values[-1] != steps:
        values.append(steps)
    return values


def curve_mean(rows: list[dict[str, object]]) -> float:
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    if len(ordered) < 2:
        return math.nan
    area = sum(
        (int(right["step"]) - int(left["step"]))
        * (float(left["loss"]) + float(right["loss"]))
        / 2.0
        for left, right in zip(ordered, ordered[1:])
    )
    span = int(ordered[-1]["step"]) - int(ordered[0]["step"])
    return area / span if span else math.nan


def parse_metrics(
    official_log: Path,
    stdout_path: Path,
    cell: PilotCell,
    args: argparse.Namespace,
    expected_init: str,
    checkpoint_path: Path | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    steps = total_steps(args)
    warmdown = warmdown_steps(args)
    rows: list[dict[str, object]] = []
    for line in official_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = r0.VAL_RE.match(line) or r0.TRAIN_RE.match(line)
        if match is None:
            continue
        step = int(match.group("step"))
        event = "validation" if "val_loss:" in line else "train"
        lr_multiplier = 1.0 if step < steps - warmdown else max(0.0, (steps - step) / warmdown)
        rows.append(
            {
                "cell_id": cell.cell_id,
                "method": cell.method,
                "event": event,
                "step": step,
                "total_steps": int(match.group("total")),
                "tokens_seen": step * TOKENS_PER_STEP,
                "loss": float(match.group("loss")),
                "official_train_time_ms": int(match.group("time")),
                "step_avg_ms": float(match.group("avg")),
                "lr_multiplier": lr_multiplier,
                "auxiliary_lr": cell.auxiliary_lr * lr_multiplier,
                "matrix_lr": cell.matrix_lr * lr_multiplier,
            }
        )
    train_rows = sorted((row for row in rows if row["event"] == "train"), key=lambda row: int(row["step"]))
    val_rows = sorted((row for row in rows if row["event"] == "validation"), key=lambda row: int(row["step"]))
    if len(train_rows) != steps:
        raise RuntimeError(f"{cell.cell_id}: expected {steps} train rows, observed {len(train_rows)}")
    observed_val = [int(row["step"]) for row in val_rows]
    expected_val = validation_steps(steps, args.val_every)
    if observed_val != expected_val:
        raise RuntimeError(f"{cell.cell_id}: validation steps {observed_val} != {expected_val}")
    if any(not math.isfinite(float(row["loss"])) for row in rows):
        raise RuntimeError(f"{cell.cell_id}: non-finite loss in parsed evidence")

    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    metadata = parse_prefixed_json(stdout, "R1X_METADATA ")
    routing = parse_prefixed_json(stdout, "R1X_ROUTING ")
    hyper = parse_prefixed_json(stdout, "R1X_HYPERPARAMS ")
    memory = parse_prefixed_json(stdout, "R1X_FINAL_MEMORY ")
    peak_matches = [r0.PEAK_RE.match(line) for line in stdout.splitlines()]
    peak_matches = [match for match in peak_matches if match is not None]
    if len(peak_matches) != 1:
        raise RuntimeError(f"{cell.cell_id}: expected one peak-memory line")
    if metadata != {"method": cell.method, "seed": args.seed, "init_sha256": expected_init}:
        raise RuntimeError(f"{cell.cell_id}: metadata mismatch: {metadata}")
    expected_hyper = {
        "method": cell.method,
        "aux_lr": cell.auxiliary_lr,
        "matrix_lr": cell.matrix_lr,
        "weight_decay": cell.weight_decay,
        "total_steps": steps,
        "warmdown_steps": warmdown,
        "val_every": args.val_every,
        "val_tokens": args.val_tokens,
    }
    if hyper != expected_hyper:
        raise RuntimeError(f"{cell.cell_id}: hyperparameter echo mismatch: {hyper} != {expected_hyper}")
    if int(memory.get("optimizer_state_bytes", 0)) <= 0 or int(memory.get("model_parameter_bytes", 0)) <= 0:
        raise RuntimeError(f"{cell.cell_id}: invalid memory report: {memory}")
    if args.formal and (
        checkpoint_path is None or not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0
    ):
        raise RuntimeError(f"{cell.cell_id}: formal run is missing its final checkpoint")
    if not args.formal and checkpoint_path is not None:
        raise RuntimeError(f"{cell.cell_id}: non-formal run unexpectedly produced a checkpoint")

    final_val = val_rows[-1]
    best_val = min(val_rows, key=lambda row: float(row["loss"]))
    final_train = train_rows[-1]
    summary: dict[str, object] = {
        **asdict(cell),
        "controlled_seed": args.seed,
        "init_sha256": expected_init,
        "total_steps": steps,
        "warmdown_steps": warmdown,
        "val_tokens": args.val_tokens,
        "final_val_loss": float(final_val["loss"]),
        "best_val_loss": float(best_val["loss"]),
        "best_val_step": int(best_val["step"]),
        "val_curve_mean": curve_mean(val_rows),
        "tail5_val_loss_mean": sum(float(row["loss"]) for row in val_rows[-5:]) / min(5, len(val_rows)),
        "final_train_loss": float(final_train["loss"]),
        "official_train_time_s": float(final_val["official_train_time_ms"]) / 1000.0,
        "peak_memory_allocated_mib": int(peak_matches[0].group("mib")),
        "optimizer_state_bytes": int(memory["optimizer_state_bytes"]),
        "model_parameter_bytes": int(memory["model_parameter_bytes"]),
        "checkpoint_bytes": checkpoint_path.stat().st_size if checkpoint_path else 0,
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path else "",
        "momentum_buffer_bytes": int(memory.get("momentum_buffer_bytes", 0)),
        "second_momentum_buffer_bytes": int(memory.get("second_momentum_buffer_bytes", 0)),
        "exp_avg_bytes": int(memory.get("exp_avg_bytes", 0)),
        "exp_avg_sq_bytes": int(memory.get("exp_avg_sq_bytes", 0)),
        "routing_hidden_matrix_tensors": int(routing["hidden_matrix_tensors"]),
        "routing_auxiliary_tensors": int(routing["auxiliary_tensors"]),
        "routing_packed_qkv_tensors": int(routing["packed_qkv_tensors"]),
        "evidence_profile": protocol(args),
        "formal_evidence": bool(args.formal),
        "seed_role": "tuned_seed_long_horizon_screen" if args.seed == 2026 else "independent_confirmatory_seed",
        "evidence_valid": True,
    }
    for row in val_rows:
        summary[f"val_loss_step_{row['step']}"] = float(row["loss"])
    return rows, summary


def upload_to_wandb(
    args: argparse.Namespace,
    run_dir: Path,
    run_name: str,
    batch_id: str,
    cell: PilotCell,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    runtime: dict[str, object],
    derived: DerivedSource,
) -> dict[str, object]:
    if is_smoke(args) or args.wandb_mode == "disabled":
        return {"status": "disabled"}
    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=hashlib.sha256(run_name.encode()).hexdigest()[:12],
            resume="allow",
            name=run_name,
            group=f"{FORMAL_RUN_PREFIX if args.formal else PILOT_RUN_PREFIX}_seed{args.seed}_{batch_id}",
            mode=args.wandb_mode,
            dir=str(run_dir),
            tags=[
                "publication", "r1", "extended_baseline",
                "formal" if args.formal else "short_pilot", cell.method, cell.lr_label,
            ],
            config={
                "experiment_family": FAMILY,
                "protocol": protocol(args),
                "cell_id": cell.cell_id,
                "method": cell.method,
                "lr_label": cell.lr_label,
                "seed": args.seed,
                "official_commit": r0.OFFICIAL_COMMIT,
                "official_base_script": derived.base_script,
                "official_base_canonical_sha256": derived.base_canonical_sha256,
                "derived_script_sha256": derived.derived_sha256,
                "auxiliary_lr": cell.auxiliary_lr,
                "matrix_lr": cell.matrix_lr,
                "weight_decay": cell.weight_decay,
                "num_iterations": total_steps(args),
                "warmdown_iters": warmdown_steps(args),
                "val_loss_every": args.val_every,
                "val_tokens": args.val_tokens,
                "batch_size_sequences": 512,
                "device_batch_size_sequences": 64,
                "sequence_length": 1024,
                "tokens_per_step": TOKENS_PER_STEP,
                "quality_role": (
                    "formal_primary_endpoint_final_validation_loss_at_step_6200"
                    if args.formal else "hyperparameter_screen_only_not_formal_evidence"
                ),
                "timing_role": "diagnostic_only",
                "source_authority": cell.source_authority,
                "gpu_name": runtime.get("gpu_name"),
            },
            reinit=True,
            settings=wandb.Settings(init_timeout=args.wandb_init_timeout),
        )
        per_step: dict[int, dict[str, float]] = {}
        for row in rows:
            step = int(row["step"])
            if row["event"] == "train" and step % args.wandb_train_log_every and step != total_steps(args):
                continue
            values = per_step.setdefault(step, {})
            values["time/train_s"] = float(row["official_train_time_ms"]) / 1000.0
            avg = float(row["step_avg_ms"])
            if math.isfinite(avg):
                values["performance/step_avg_ms"] = avg
            values["lr/auxiliary"] = float(row["auxiliary_lr"])
            values["lr/matrix"] = float(row["matrix_lr"])
            values["val/loss" if row["event"] == "validation" else "train/loss_step"] = float(row["loss"])
        final_values = per_step.setdefault(total_steps(args), {})
        final_values["memory/peak_allocated_mib"] = float(summary["peak_memory_allocated_mib"])
        final_values["memory/optimizer_state_mib"] = float(summary["optimizer_state_bytes"]) / 1024**2
        for step in sorted(per_step):
            wandb.log(per_step[step], step=step)
        for key, value in summary.items():
            if isinstance(value, (str, int, float, bool)):
                run.summary[key] = value
        run.finish()
        return {"status": "uploaded", "run_id": getattr(run, "id", None), "run_url": getattr(run, "url", None)}
    except Exception as exc:
        return {"status": "failed", "error": repr(exc)}


def find_official_log(workspace: Path) -> Path:
    logs = list((workspace / "logs").glob("*.txt"))
    if len(logs) != 1:
        raise RuntimeError(f"expected one official log, observed {len(logs)} in {workspace}")
    return logs[0]


def find_checkpoint(workspace: Path) -> Path | None:
    checkpoints = list((workspace / "logs").glob("*/state_step*.pt"))
    if not checkpoints:
        return None
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected at most one final checkpoint, observed {len(checkpoints)}")
    return checkpoints[0]


def run_cell(
    args: argparse.Namespace,
    repo: Path,
    data_dir: Path,
    batch_dir: Path,
    batch_id: str,
    cell: PilotCell,
    derived: DerivedSource,
    expected_init: str,
    runtime: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    run_prefix = FORMAL_RUN_PREFIX if is_formal_profile(args) else PILOT_RUN_PREFIX
    run_name = f"{run_prefix}_{cell.cell_id}_seed{args.seed}_{batch_id}"
    run_dir = batch_dir / run_name
    existing_summary = run_dir / "summary.json"
    existing_manifest = run_dir / "run_manifest.json"
    if existing_summary.is_file() and existing_manifest.is_file():
        summary = json.loads(existing_summary.read_text(encoding="utf-8"))
        manifest = json.loads(existing_manifest.read_text(encoding="utf-8"))
        metrics_path = run_dir / "metrics.csv"
        saved_checkpoint = Path(str(summary.get("checkpoint_path", "")))
        checkpoint_ok = not args.formal or (
            saved_checkpoint.is_file()
            and saved_checkpoint.stat().st_size == int(summary.get("checkpoint_bytes", -1))
            and saved_checkpoint.stat().st_size > 0
        )
        if (
            manifest.get("status") in (
                "completed_valid", "completed_valid_local", "completed_valid_local_wandb_incomplete"
            )
            and summary.get("evidence_valid") is True
            and summary.get("cell_id") == cell.cell_id
            and summary.get("controlled_seed") == args.seed
            and summary.get("init_sha256") == expected_init
            and summary.get("total_steps") == total_steps(args)
            and metrics_path.is_file()
            and checkpoint_ok
        ):
            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if (
                (args.pilot or args.formal)
                and args.wandb_mode != "disabled"
                and summary.get("wandb_status") != "uploaded"
            ):
                print(f"Resume: retrying W&B upload for completed cell {cell.cell_id}")
                upload = upload_to_wandb(
                    args,
                    run_dir,
                    run_name,
                    batch_id,
                    cell,
                    rows,
                    summary,
                    runtime,
                    derived,
                )
                summary["wandb_status"] = str(upload.get("status", "unknown"))
                manifest["wandb"] = upload
                manifest["status"] = (
                    "completed_valid"
                    if upload.get("status") == "uploaded"
                    else "completed_valid_local_wandb_incomplete"
                )
                manifest["summary"] = summary
                write_json(existing_summary, summary)
                write_json(existing_manifest, manifest)
            print(f"Resume: reusing completed cell {cell.cell_id}")
            return summary, rows
        raise RuntimeError(f"resume found incompatible existing run directory: {run_dir}")

    workspace = run_dir / "workspace"
    script = materialize_source(workspace, repo, derived)
    (run_dir / "official_to_extended.patch").write_text(derived.unified_diff, encoding="utf-8")
    write_json(run_dir / "cell_spec.json", asdict(cell))
    stdout_path = run_dir / "training_stdout.log"
    command = [args.python_exe, script.name]
    manifest: dict[str, object] = {
        "status": "running",
        "run_name": run_name,
        "cell": asdict(cell),
        "seed": args.seed,
        "command": command,
        "cwd": str(workspace.resolve()),
        "started_at": datetime.now().astimezone().isoformat(),
        "derived_source_sha256": derived.derived_sha256,
    }
    write_json(run_dir / "run_manifest.json", manifest)
    wall_start = time.monotonic()
    with stdout_path.open("w", encoding="utf-8", buffering=1) as output:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=controlled_env(args, data_dir, cell),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        first_nonfinite = r0.stream_process_with_finite_gate(process, output)
        returncode = process.wait()
    manifest.update(
        {
            "returncode": returncode,
            "wall_elapsed_s": time.monotonic() - wall_start,
            "finished_at": datetime.now().astimezone().isoformat(),
        }
    )
    if first_nonfinite is not None:
        manifest.update({"status": "invalid_nonfinite", "first_nonfinite": first_nonfinite})
        write_json(run_dir / "run_manifest.json", manifest)
        raise RuntimeError(f"{cell.cell_id} produced non-finite loss: {first_nonfinite}")
    if returncode != 0:
        manifest["status"] = "training_failed"
        write_json(run_dir / "run_manifest.json", manifest)
        raise RuntimeError(f"{cell.cell_id} failed with code {returncode}; see {stdout_path}")
    official_log = find_official_log(workspace)
    copied_log = run_dir / "training_log_with_source.txt"
    shutil.copy2(official_log, copied_log)
    checkpoint = find_checkpoint(workspace)
    rows, summary = parse_metrics(copied_log, stdout_path, cell, args, expected_init, checkpoint)
    write_csv(run_dir / "metrics.csv", rows)
    write_json(run_dir / "summary.json", summary)
    manifest["status"] = "completed_valid_local"
    manifest["summary"] = summary
    manifest["checkpoint_path"] = str(checkpoint.resolve()) if checkpoint else ""
    manifest["wandb"] = {"status": "pending" if not is_smoke(args) else "disabled"}
    write_json(run_dir / "run_manifest.json", manifest)
    upload = upload_to_wandb(
        args, run_dir, run_name, batch_id, cell, rows, summary, runtime, derived
    )
    summary["wandb_status"] = str(upload.get("status", "unknown"))
    write_json(run_dir / "summary.json", summary)
    manifest["wandb"] = upload
    manifest["status"] = (
        "completed_valid"
        if upload.get("status") in ("uploaded", "disabled")
        else "completed_valid_local_wandb_incomplete"
    )
    manifest["summary"] = summary
    write_json(run_dir / "run_manifest.json", manifest)
    return summary, rows


def validate_formal_smoke_certificate(
    path: Path,
    args: argparse.Namespace,
    cells: list[PilotCell],
    runtime: dict[str, object],
    init_audit: dict[str, object],
    source_audit: dict[str, object],
) -> dict[str, object]:
    certificate_path = path.expanduser().resolve()
    if not certificate_path.is_file():
        raise RuntimeError(f"formal smoke manifest does not exist: {certificate_path}")
    payload = json.loads(certificate_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if payload.get("status") != "completed_valid":
        failures.append(f"status is {payload.get('status')!r}, expected 'completed_valid'")
    if payload.get("protocol") != FORMAL_SMOKE_PROTOCOL:
        failures.append(f"protocol is {payload.get('protocol')!r}")
    if payload.get("batch_kind") != "formal_smoke":
        failures.append(f"batch_kind is {payload.get('batch_kind')!r}")
    if payload.get("seed") != args.seed:
        failures.append(f"seed is {payload.get('seed')!r}, expected {args.seed}")
    if int(payload.get("total_steps", 0)) < DEFAULT_FORMAL_SMOKE_STEPS:
        failures.append(f"smoke has fewer than {DEFAULT_FORMAL_SMOKE_STEPS} steps")
    expected_cells = [cell.cell_id for cell in cells]
    observed_cells = [str(item.get("cell_id")) for item in payload.get("cells", [])]
    if observed_cells != expected_cells:
        failures.append(f"cells differ: observed={observed_cells}, expected={expected_cells}")
    if payload.get("initialization_audit", {}).get("init_sha256") != init_audit.get("init_sha256"):
        failures.append("initialization SHA-256 differs")
    observed_sources = payload.get("source_audit", {}).get("derived_source_sha256")
    if observed_sources != source_audit.get("derived_source_sha256"):
        failures.append("derived-source SHA-256 map differs")
    observed_runtime = r0.normalize_runtime_fingerprint(payload.get("training_runtime_fingerprint"))
    expected_runtime = r0.normalize_runtime_fingerprint(r0.runtime_fingerprint(runtime))
    if observed_runtime != expected_runtime:
        failures.append(f"training runtime differs: observed={observed_runtime}, expected={expected_runtime}")
    summaries = payload.get("summaries", [])
    if len(summaries) != len(cells) or any(item.get("evidence_valid") is not True for item in summaries):
        failures.append("smoke summaries are incomplete or invalid")
    if failures:
        raise RuntimeError("formal smoke certificate rejected:\n- " + "\n- ".join(failures))
    return {
        "path": str(certificate_path),
        "sha256": sha256_file(certificate_path),
        "batch_id": payload.get("batch_id"),
        "seed": payload.get("seed"),
        "status": payload.get("status"),
    }


def main() -> None:
    args = parse_args()
    repo = args.official_repo.expanduser().resolve()
    provenance = r0.validate_official_repo(repo)
    built = build_all_sources(repo)
    data = r0.validate_data(repo)
    data_dir = Path(str(data["data_dir"]))
    runtime = r0.validate_runtime(repo, args.python_exe)
    controller = r0.validate_controller_runtime(
        require_wandb=(args.pilot or args.formal) and args.wandb_mode != "disabled"
    )
    wandb_readiness = validate_wandb_readiness(args)
    if runtime.get("runtime_rejection_reason"):
        raise RuntimeError(str(runtime["runtime_rejection_reason"]))
    print(
        f"Runtime: {runtime['gpu_name']}, {runtime['gpu_total_memory_gib']:.2f} GiB; "
        f"official commit {r0.OFFICIAL_COMMIT[:12]}"
    )
    init_audit = initialization_audit(args, repo, data_dir, built)
    step_audit = single_step_audit(args, repo)
    source_audit = {
        "official_adam_base_sha256": next(iter(built.values())).base_canonical_sha256,
        "derived_source_sha256": {method: item.derived_sha256 for method, item in built.items()},
        "extended_optimizers_sha256": sha256_file(SCRIPT_DIR / "extended_optimizers.py"),
        "normuon_reference": "https://github.com/zichongli5/NorMuon/blob/main/normuon.py",
        "moonlight_reference": "https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py",
        "retrieved_on": "2026-07-21",
    }

    if args.preflight:
        stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        artifact = args.results_dir.expanduser().resolve() / f"{stamp}_preflight_seed{args.seed}.json"
        payload = {
            "status": "passed",
            "protocol": AUDIT_PROTOCOL,
            "official_provenance": provenance,
            "data": data,
            "runtime": runtime,
            "controller_runtime": controller,
            "wandb_readiness": wandb_readiness,
            "initialization_audit": init_audit,
            "single_step_reference_audit": step_audit,
            "source_audit": source_audit,
        }
        write_json(artifact, payload)
        print(f"R1 extended preflight artifact: {artifact}")
        return

    cells = selected_cells(args)
    stem = artifact_stem(args)
    smoke_certificate: dict[str, object] | None = None
    if args.formal and args.resume_batch is None:
        smoke_certificate = validate_formal_smoke_certificate(
            args.smoke_manifest, args, cells, runtime, init_audit, source_audit
        )
    if args.resume_batch is not None:
        batch_dir = args.resume_batch.expanduser().resolve()
        plan_path = batch_dir / f"{stem}_plan.json"
        if not plan_path.is_file():
            raise RuntimeError(f"resume batch has no {stem}_plan.json: {batch_dir}")
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        if saved.get("protocol") != protocol(args) or saved.get("batch_kind") != batch_kind(args):
            raise RuntimeError("resume mode/protocol differs from the saved plan")
        if saved.get("seed") != args.seed or saved.get("total_steps") != total_steps(args):
            raise RuntimeError("resume seed/step count differs from the saved plan")
        saved_cells = [str(item.get("cell_id")) for item in saved.get("cells", [])]
        current_cells = [cell.cell_id for cell in cells]
        if saved_cells != current_cells:
            raise RuntimeError(f"resume cells differ: saved={saved_cells}, current={current_cells}")
        saved_sources = saved.get("source_audit", {}).get("derived_source_sha256")
        if saved_sources != source_audit["derived_source_sha256"]:
            raise RuntimeError("resume derived-source fingerprints differ from the saved plan")
        if saved.get("initialization_audit", {}).get("init_sha256") != init_audit.get("init_sha256"):
            raise RuntimeError("resume initialization fingerprint differs from the saved plan")
        saved_runtime = r0.normalize_runtime_fingerprint(saved.get("training_runtime_fingerprint"))
        current_runtime = r0.normalize_runtime_fingerprint(r0.runtime_fingerprint(runtime))
        if saved_runtime != current_runtime:
            raise RuntimeError(
                f"resume training runtime differs: saved={saved_runtime}, current={current_runtime}"
            )
        batch_id = str(saved["batch_id"])
        smoke_certificate = saved.get("smoke_certificate")
        if args.formal and not smoke_certificate:
            raise RuntimeError("formal resume plan has no accepted smoke certificate")
        plan = saved
    else:
        batch_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        kind = batch_kind(args)
        batch_dir = args.results_dir.expanduser().resolve() / f"{batch_id}_{kind}_seed{args.seed}"
        batch_dir.mkdir(parents=True, exist_ok=False)
        plan = {
            "family": FAMILY,
            "protocol": protocol(args),
            "batch_id": batch_id,
            "batch_kind": kind,
            "seed": args.seed,
            "total_steps": total_steps(args),
            "warmdown_steps": warmdown_steps(args),
            "val_every": args.val_every,
            "val_tokens": args.val_tokens,
            "cells": [asdict(cell) for cell in cells],
            "formal_evidence": bool(args.formal),
            "primary_metric": (
                "final validation loss at step 6200 (3,250,585,600 tokens)"
                if args.formal else "validation loss at matched optimizer steps/tokens"
            ),
            "secondary_metrics": ["tail5_val_loss_mean", "val_curve_mean", "best_val_loss"],
            "seed_role": "tuned_seed_long_horizon_screen" if args.seed == 2026 else "independent_confirmatory_seed",
            "timing_role": "diagnostic only; not the dedicated performance experiment",
            "official_provenance": provenance,
            "data": data,
            "runtime": runtime,
            "training_runtime_fingerprint": r0.runtime_fingerprint(runtime),
            "controller_runtime": controller,
            "wandb_readiness": wandb_readiness,
            "initialization_audit": init_audit,
            "single_step_reference_audit": step_audit,
            "source_audit": source_audit,
            "wandb_project": args.wandb_project,
            "wandb_mode": "disabled" if is_smoke(args) else args.wandb_mode,
            "smoke_certificate": smoke_certificate,
        }
        write_json(batch_dir / f"{stem}_plan.json", plan)

    summaries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for index, cell in enumerate(cells, start=1):
        print(
            f"\n=== R1 extended {index}/{len(cells)}: {cell.cell_id} "
            f"aux_lr={cell.auxiliary_lr:g} matrix_lr={cell.matrix_lr:g} wd={cell.weight_decay:g} ==="
        )
        try:
            summary, _ = run_cell(
                args,
                repo,
                data_dir,
                batch_dir,
                batch_id,
                cell,
                built[cell.method],
                str(init_audit["init_sha256"]),
                runtime,
            )
            summaries.append(summary)
        except Exception as exc:
            failures.append({"cell_id": cell.cell_id, "error": repr(exc)})
            print(f"R1 extended cell failed: {cell.cell_id}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break
        finally:
            write_json(
                batch_dir / f"{stem}_manifest.json",
                {
                    **plan,
                    "status": "running" if not failures else "failed",
                    "batch_id": batch_id,
                    "summaries": summaries,
                    "failures": failures,
                },
            )
    ranked = sorted(summaries, key=lambda row: (float(row["final_val_loss"]), float(row["best_val_loss"])))
    for rank, summary in enumerate(ranked, start=1):
        summary["rank_by_final_val"] = rank
    write_csv(batch_dir / f"{stem}_summary.csv", ranked)
    all_completed = len(summaries) == len(cells) and not failures
    wandb_complete = is_smoke(args) or args.wandb_mode == "disabled" or all(
        summary.get("wandb_status") == "uploaded" for summary in summaries
    )
    final = {
        **plan,
        "status": (
            "completed_valid"
            if all_completed and wandb_complete
            else "completed_valid_local_wandb_incomplete"
            if all_completed
            else "failed"
        ),
        "batch_id": batch_id,
        "formal_evidence": bool(args.formal),
        "ranking_is_screening_only": not args.formal,
        "wandb_complete": wandb_complete,
        "summaries": ranked,
        "failures": failures,
    }
    write_json(batch_dir / f"{stem}_manifest.json", final)
    print(f"R1 extended artifacts: {batch_dir}")
    if failures:
        raise SystemExit(1)
    if args.formal and not wandb_complete:
        raise SystemExit("formal local evidence is valid, but W&B upload is incomplete; resume the batch")


if __name__ == "__main__":
    main()
