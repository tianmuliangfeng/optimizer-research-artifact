"""Audit, tune, and formally run controlled 124M MALT-family R1 baselines."""

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
from malt_source_builder import DerivedSource, build_source
from project_paths import EXPERIMENT_RESULTS_ROOT


FAMILY = "49_r1_malt_strong_baseline"
PREFLIGHT_PROTOCOL = "malt_r1_implementation_audit_v4"
PILOT_PROTOCOL = "malt_r1_focused_grid_pilot_v4"
SMOKE_PROTOCOL = "malt_r1_selected_exact_shape_smoke_v4"
FORMAL_PROTOCOL = "malt_r1_selected_6200step_v4"
SELECTION_PROTOCOL = "malt_r1_focused_grid_selection_v4"
PILOT_PROJECT = "Selective-Newton-Muon-MainConf-R1-MALTFamilyPilotV4-20260809"
FORMAL_PROJECT = "Selective-Newton-Muon-MainConf-R1-MALTFamilyFormalV4-20260809"
TOKENS_PER_STEP = 512 * 1024
FULL_STEPS = 6200
FULL_WARMDOWN = 1800
PILOT_STEPS = 1000
SMOKE_STEPS = 34
VAL_EVERY = 100
VAL_TOKENS = 10_485_760
AUX_LR = 0.0036
MATRIX_WEIGHT_DECAY = 0.1
MALTER_CENTER_LR = 0.012
TIE_MARGIN = 0.002
MALT_EXPECTED_HIDDEN_STATE_BYTES = 340_402_176
MALTER_EXPECTED_HIDDEN_STATE_BYTES = 340_402_464
EXPECTED_TRAIN_SHARDS = 50
EXPECTED_TRAINING_RUNTIME = {
    "python": "3.10.12",
    "torch": "2.8.0+cu126",
    "torch_cuda": "12.6",
    "triton": "3.4.0",
    "numpy": "2.2.6",
}
JSON_PREFIXES = ("R1T_METADATA ", "R1T_ROUTING ", "R1T_HYPERPARAMS ", "R1T_FINAL_MEMORY ")


@dataclass(frozen=True)
class MALTCell:
    cell_id: str
    lr_label: str
    matrix_lr: float
    formal_eligible: bool
    method: str = "malt"
    auxiliary_lr: float = AUX_LR
    matrix_weight_decay: float = MATRIX_WEIGHT_DECAY


PILOT_CELLS = (
    # V4 is a fresh, focused rerun.  It retains the two highest V3 MALT points
    # as lower-side anchors and scans the frozen extension from high to low.
    MALTCell("malt_lr0160", "v4_upper_boundary_0.0160", 0.0160, True),
    MALTCell("malt_lr0125", "v4_grid_0.0125", 0.0125, True),
    MALTCell("malt_lr0100", "v4_grid_0.0100", 0.0100, True),
    MALTCell("malt_lr0090", "v4_grid_0.0090", 0.0090, True),
    MALTCell("malt_lr0080", "v3_upper_anchor_0.0080", 0.0080, True),
    MALTCell("malt_lr0064", "v3_preceding_lower_boundary_0.0064", 0.0064, True),
    MALTCell("malter_eq17_lr007", "paper_lower_boundary_0.007", 0.007, True, method="malter_eq17"),
    MALTCell("malter_eq17_lr009", "paper_grid_0.009", 0.009, True, method="malter_eq17"),
    MALTCell("malter_eq17_lr012", "paper_center_0.012", MALTER_CENTER_LR, True, method="malter_eq17"),
    MALTCell("malter_eq17_lr015", "paper_grid_0.015", 0.015, True, method="malter_eq17"),
    MALTCell("malter_eq17_lr018", "paper_grid_0.018", 0.018, True, method="malter_eq17"),
    MALTCell("malter_eq17_lr025", "paper_upper_boundary_0.025", 0.025, True, method="malter_eq17"),
)
MALT_REFERENCE_CELL_ID = "malt_lr0080"
MALTER_CENTER_CELL_ID = "malter_eq17_lr012"
MALT_LOWER_BOUNDARY_CELL_ID = "malt_lr0064"
MALT_UPPER_BOUNDARY_CELL_ID = "malt_lr0160"
MALTER_LOWER_BOUNDARY_CELL_ID = "malter_eq17_lr007"
MALTER_UPPER_BOUNDARY_CELL_ID = "malter_eq17_lr025"
FORMAL_METHODS = ("malt", "malter_eq17")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
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


def wandb_status_satisfies(mode: str, status: object) -> bool:
    return {
        "online": status == "uploaded_online",
        "offline": status == "saved_offline",
        "disabled": status == "disabled",
    }[mode]


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
    parser.add_argument("--data-inventory-certificate", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=EXPERIMENT_RESULTS_ROOT / FAMILY / "results",
    )
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
    parser.add_argument("--selected-method", choices=FORMAL_METHODS)
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
    if (args.formal or args.formal_smoke) and args.selected_method is None and args.resume_batch is None:
        parser.error("a new formal profile requires --selected-method")
    if not (args.formal or args.formal_smoke) and args.selected_method is not None:
        parser.error("--selected-method is only valid for formal profiles")
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


def selected_pilot_cells(args: argparse.Namespace) -> list[MALTCell]:
    wanted = set(args.cells or [cell.cell_id for cell in PILOT_CELLS])
    cells = [cell for cell in PILOT_CELLS if cell.cell_id in wanted]
    if not cells:
        raise RuntimeError("empty pilot cell selection")
    return cells


def controlled_env(
    args: argparse.Namespace,
    data_dir: Path,
    cell: MALTCell,
    *,
    init_only: bool = False,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": str(args.seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "R1T_SEED": str(args.seed),
            "R1T_METHOD": cell.method,
            "R1T_DATA_DIR": str(data_dir.resolve()),
            "R1T_TOTAL_STEPS": str(total_steps(args)),
            "R1T_WARMDOWN_STEPS": str(warmdown_steps(args)),
            "R1T_VAL_EVERY": str(args.val_every),
            "R1T_VAL_TOKENS": str(args.val_tokens),
            "R1T_TRAIN_SHARD_COUNT": str(EXPECTED_TRAIN_SHARDS),
            "R1T_AUX_LR": repr(cell.auxiliary_lr),
            "R1T_MATRIX_LR": repr(cell.matrix_lr),
            "R1T_MATRIX_WEIGHT_DECAY": repr(cell.matrix_weight_decay),
            "R1T_INIT_ONLY": "1" if init_only else "0",
            "R1T_DISABLE_CHECKPOINT": "0" if args.formal else "1",
        }
    )
    return env


def validate_exact_training_runtime(runtime: dict[str, object]) -> dict[str, object]:
    observed = {
        "python": str(runtime.get("python", "")).split()[0],
        "torch": str(runtime.get("torch", "")),
        "torch_cuda": str(runtime.get("torch_cuda", "")),
        "triton": str(runtime.get("triton", "")),
        "numpy": str(runtime.get("numpy", "")),
    }
    checks = {
        key: observed.get(key) == expected
        for key, expected in EXPECTED_TRAINING_RUNTIME.items()
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Experiment-49 exact training runtime mismatch: "
            f"expected={EXPECTED_TRAINING_RUNTIME}, observed={observed}"
        )
    return {
        "status": "passed",
        "expected": EXPECTED_TRAINING_RUNTIME,
        "observed": observed,
        "checks": checks,
        "python_executable": runtime.get("python_executable"),
    }


def validate_data_inventory_certificate(
    certificate_path: Path, data_dir: Path
) -> dict[str, object]:
    path = certificate_path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"frozen data inventory certificate is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_names = [
        f"fineweb_train_{index:06d}.bin"
        for index in range(1, EXPECTED_TRAIN_SHARDS + 1)
    ]
    records = payload.get("ordered_train_shards")
    failures: list[str] = []
    if payload.get("status") != "passed":
        failures.append("certificate status")
    if Path(str(payload.get("data_dir", ""))).resolve() != data_dir.resolve():
        failures.append("data directory")
    if not isinstance(records, list) or [record.get("name") for record in records] != expected_names:
        failures.append("ordered train shard names")
        records = []
    validation = payload.get("validation_shard")
    if not isinstance(validation, dict) or validation.get("name") != "fineweb_val_000000.bin":
        failures.append("validation shard name")
        validation = {}
    for record in [*records, validation]:
        if not record:
            continue
        shard = data_dir / str(record["name"])
        if not shard.is_file() or shard.stat().st_size != int(record.get("bytes", -1)):
            failures.append(f"current file/size: {record.get('name')}")
    if failures:
        raise RuntimeError(f"frozen R1 data inventory validation failed: {failures}")
    return {
        "status": "passed",
        "path": str(path),
        "sha256": sha256_file(path),
        "data_dir": str(data_dir.resolve()),
        "train_shard_count": EXPECTED_TRAIN_SHARDS,
        "validation_shard_count": 1,
        "selected_total_bytes": int(payload.get("selected_total_bytes", 0)),
    }


def materialize_source(directory: Path, official_repo: Path, derived: DerivedSource) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "train_r1_malt.py"
    script.write_text(derived.source, encoding="utf-8", newline="\n")
    shutil.copy2(SCRIPT_DIR / "malt_optimizer.py", directory / "malt_optimizer.py")
    shutil.copy2(SCRIPT_DIR / "malt_contract.json", directory / "malt_contract.json")
    shutil.copy2(SCRIPT_DIR / "PAPER_DERIVATION.md", directory / "PAPER_DERIVATION.md")
    shutil.copy2(official_repo / "triton_kernels.py", directory / "triton_kernels.py")
    return script


def initialization_audit(
    args: argparse.Namespace, repo: Path, data_dir: Path, derived: DerivedSource, cell: MALTCell
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
    metadata = parse_prefixed_json(result.stdout, "R1T_METADATA ")
    routing = parse_prefixed_json(result.stdout, "R1T_ROUTING ")
    if metadata.get("method") != cell.method or metadata.get("seed") != args.seed:
        raise RuntimeError(f"initialization metadata mismatch: {metadata}")
    expected = {"hidden_matrix_tensors": 48, "auxiliary_tensors": 1, "packed_qkv_tensors": 12, "logical_hidden_matrices": 72, "activation_k_state_routes": 0}
    mismatches = {key: (routing.get(key), value) for key, value in expected.items() if routing.get(key) != value}
    if mismatches:
        raise RuntimeError(f"R1 parameter routing mismatch: {mismatches}")
    return {"status": "passed", **metadata, "routing": routing}


def reference_audit(args: argparse.Namespace) -> dict[str, object]:
    code = (
        "import json,sys;sys.path.insert(0," + repr(str(SCRIPT_DIR)) + ");"
        "from malt_optimizer import run_small_matrix_reference_audit;"
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
    cell: MALTCell,
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
                "cell_id": cell.cell_id, "method": cell.method, "event": event,
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
    metadata = parse_prefixed_json(stdout, "R1T_METADATA ")
    routing = parse_prefixed_json(stdout, "R1T_ROUTING ")
    hyper = parse_prefixed_json(stdout, "R1T_HYPERPARAMS ")
    memory = parse_prefixed_json(stdout, "R1T_FINAL_MEMORY ")
    peaks = [match for line in stdout.splitlines() if (match := r0.PEAK_RE.match(line))]
    if metadata != {"method": cell.method, "seed": args.seed, "init_sha256": expected_init}:
        raise RuntimeError(f"metadata mismatch: {metadata}")
    expected_hyper = {
        "method": cell.method,
        "implementation_label": "paper-derived independent implementation",
        "aux_lr": cell.auxiliary_lr,
        "matrix_lr": cell.matrix_lr,
        "matrix_weight_decay": cell.matrix_weight_decay,
        "beta1": 0.95,
        "beta2": 0.99,
        "epsilon": 1e-8,
        "nesterov": False,
        "row_column_bias_correction": False,
        "orthogonalize_backend": "pinned_r1_newtonschulz5",
        "orthogonalize_steps": 5,
        "r1_dimension_scale": False,
        "packed_qkv_policy": "split_three_logical_matrices",
        "malter_formula_choice": "equation_17_single_eta" if cell.method == "malter_eq17" else "not_applicable",
        "total_steps": steps,
        "warmdown_steps": warmdown, "val_every": args.val_every, "val_tokens": args.val_tokens,
    }
    if hyper != expected_hyper:
        raise RuntimeError(f"hyperparameter echo mismatch: {hyper} != {expected_hyper}")
    schema = memory.get("state_schema", {})
    if schema.get("contains_activation_k_state") is not False or routing.get("activation_k_state_routes") != 0:
        raise RuntimeError("activation-K state leakage detected")
    expected_roles = {
        "malt_momentum": 48,
        "malt_row_ema": 72,
        "malt_col_ema": 72,
        "malt_last_alpha_min": 48,
        "malt_last_alpha_max": 48,
    }
    if cell.method == "malter_eq17":
        expected_roles["malt_nu"] = 72
    if schema.get("roles") != expected_roles:
        raise RuntimeError(f"MALT state schema mismatch: {schema.get('roles')} != {expected_roles}")
    if schema.get("optimizer_group_steps") != [steps]:
        raise RuntimeError(f"MALT optimizer-step count mismatch: {schema.get('optimizer_group_steps')} != {[steps]}")
    if schema.get("numerical_checks_passed") is not True:
        raise RuntimeError(f"MALT state numerical audit failed: {schema}")
    expected_hidden_bytes = (
        MALTER_EXPECTED_HIDDEN_STATE_BYTES
        if cell.method == "malter_eq17"
        else MALT_EXPECTED_HIDDEN_STATE_BYTES
    )
    if int(memory.get("hidden_optimizer_state_bytes", -1)) != expected_hidden_bytes:
        raise RuntimeError(
            "hidden optimizer state bytes mismatch: "
            f"{memory.get('hidden_optimizer_state_bytes')} != {expected_hidden_bytes}"
        )
    if int(memory.get("malt_momentum_bytes", -1)) != 339_738_624:
        raise RuntimeError("MALT momentum byte audit failed")
    if int(memory.get("malt_row_ema_bytes", -1)) + int(memory.get("malt_col_ema_bytes", -1)) != 663_552:
        raise RuntimeError("MALT row/column state byte audit failed")
    if cell.method == "malter_eq17" and int(memory.get("malt_nu_bytes", -1)) != 288:
        raise RuntimeError("MALTER scalar state byte audit failed")
    if len(peaks) != 1 or int(memory.get("total_optimizer_state_bytes", 0)) <= expected_hidden_bytes:
        raise RuntimeError("invalid state/memory report")
    if args.formal and (checkpoint is None or not checkpoint.is_file() or checkpoint.stat().st_size <= 0):
        raise RuntimeError("formal run is missing its final checkpoint")
    if not args.formal and checkpoint is not None:
        raise RuntimeError("non-formal run unexpectedly produced a checkpoint")
    summary: dict[str, object] = {
        **asdict(cell), "method": cell.method, "controlled_seed": args.seed,
        "implementation_label": "paper-derived independent implementation",
        "adaptation_label": "MALT-R1 adaptation" if cell.method == "malt" else "MALTER-Eq17-R1 adaptation",
        "init_sha256": expected_init, "total_steps": steps, "warmdown_steps": warmdown,
        "tokens_per_step": TOKENS_PER_STEP, "total_tokens": steps * TOKENS_PER_STEP,
        "initial_val_loss": float(val[0]["loss"]), "final_val_loss": float(val[-1]["loss"]),
        "best_val_loss": min(float(row["loss"]) for row in val),
        "tail5_val_loss_mean": sum(float(row["loss"]) for row in val[-5:]) / min(5, len(val)),
        "normalized_val_auc": curve_mean(val), "final_train_loss": float(train[-1]["loss"]),
        "official_train_time_s_diagnostic": float(val[-1]["official_train_time_ms"]) / 1000.0,
        "peak_memory_allocated_mib": int(peaks[0].group("mib")),
        "hidden_optimizer_state_bytes": int(memory["hidden_optimizer_state_bytes"]),
        "total_optimizer_state_bytes": int(memory["total_optimizer_state_bytes"]),
        "auxiliary_optimizer_state_bytes": int(memory["auxiliary_optimizer_state_bytes"]),
        "optimizer_state_bytes": int(memory["total_optimizer_state_bytes"]),
        "model_parameter_bytes": int(memory["model_parameter_bytes"]),
        "malt_momentum_bytes": int(memory["malt_momentum_bytes"]),
        "malt_row_ema_bytes": int(memory["malt_row_ema_bytes"]),
        "malt_col_ema_bytes": int(memory["malt_col_ema_bytes"]),
        "malt_nu_bytes": int(memory.get("malt_nu_bytes", 0)),
        "state_schema": schema,
        "checkpoint_path": str(checkpoint.resolve()) if checkpoint else "",
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint else 0,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else "",
        "evidence_profile": protocol(args), "formal_evidence": bool(args.formal),
        "timing_eligible": False, "evidence_valid": True,
        "seed_role": "tuned_seed_long_horizon_screen" if args.seed == 2026 else "independent_confirmatory_seed",
    }
    for row in val:
        summary[f"val_loss_step_{row['step']}"] = float(row["loss"])
    return rows, summary


def upload_wandb(
    args: argparse.Namespace, run_dir: Path, run_name: str, batch_id: str,
    cell: MALTCell, rows: list[dict[str, object]], summary: dict[str, object],
    runtime: dict[str, object], derived: DerivedSource,
) -> dict[str, object]:
    if args.formal_smoke or args.numerical_smoke or args.wandb_mode == "disabled":
        return {"status": "disabled"}
    ledger_path = run_dir / "wandb_upload_attempts.jsonl"
    previous_attempts: list[int] = []
    if ledger_path.is_file():
        raw_ledger = ledger_path.read_bytes()
        valid_lines: list[bytes] = []
        lines = raw_ledger.splitlines()
        for index, raw_line in enumerate(lines):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line.decode("utf-8"))
                previous_attempts.append(int(row["attempt"]))
                valid_lines.append(raw_line)
            except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
                is_torn_tail = index == len(lines) - 1 and not raw_ledger.endswith(b"\n")
                if is_torn_tail:
                    quarantine = ledger_path.with_name(
                        f"{ledger_path.stem}.torn_{time.time_ns()}.bin"
                    )
                    quarantine.write_bytes(raw_line)
                    normalized = b"\n".join(valid_lines)
                    if normalized:
                        normalized += b"\n"
                    temporary = ledger_path.with_name(
                        f".{ledger_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
                    )
                    try:
                        with temporary.open("xb") as handle:
                            handle.write(normalized)
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(temporary, ledger_path)
                    finally:
                        temporary.unlink(missing_ok=True)
                    break
                raise RuntimeError(f"invalid W&B attempt ledger: {ledger_path}") from exc
        else:
            if raw_ledger and not raw_ledger.endswith(b"\n"):
                with ledger_path.open("ab") as handle:
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
    attempt = max(previous_attempts, default=0) + 1
    upload_id = hashlib.sha256(f"{run_name}:upload:{attempt}".encode()).hexdigest()[:12]
    def record_attempt(status: str, **extra: object) -> None:
        payload: dict[str, object] = {
            "attempt": attempt,
            "run_id": upload_id,
            "mode": args.wandb_mode,
            "status": status,
            "recorded_at": datetime.now().astimezone().isoformat(),
        }
        payload.update(extra)
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    record_attempt("started")
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            id=upload_id, resume="never",
            name=f"{run_name}_upload{attempt:02d}", group=f"r1_{cell.method}_{mode_name(args)}_seed{args.seed}_{batch_id}",
            mode=args.wandb_mode, dir=str(run_dir), reinit=True,
            tags=["main_conference", "experiment49", "r1", cell.method, mode_name(args), f"upload_attempt_{attempt}"],
            config={
                "experiment_family": FAMILY, "protocol": protocol(args), "method": cell.method,
                "cell_id": cell.cell_id, "seed": args.seed, "matrix_lr": cell.matrix_lr,
                "auxiliary_lr": cell.auxiliary_lr, "matrix_weight_decay": cell.matrix_weight_decay,
                "official_r1_commit": r0.OFFICIAL_COMMIT, "derived_script_sha256": derived.derived_sha256,
                "num_iterations": total_steps(args), "warmdown_iters": warmdown_steps(args),
                "batch_size_sequences": 512, "sequence_length": 1024,
                "tokens_per_step": TOKENS_PER_STEP, "timing_eligible": False,
                "gpu_name": runtime.get("gpu_name"),
                "adaptation_label": "MALT-R1 adaptation" if cell.method == "malt" else "MALTER-Eq17-R1 adaptation",
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
        result = {
            "status": "uploaded_online" if args.wandb_mode == "online" else "saved_offline",
            "run_id": getattr(run, "id", None),
            "run_url": getattr(run, "url", None),
            "upload_attempt": attempt,
        }
        record_attempt(
            "completed",
            completion_status=result["status"],
            run_url=result["run_url"],
        )
        return result
    except Exception as exc:
        record_attempt("failed", error=repr(exc))
        return {"status": "failed", "error": repr(exc), "run_id": upload_id, "upload_attempt": attempt}


def run_cell(
    args: argparse.Namespace, official_repo: Path, data_dir: Path, batch_dir: Path, batch_id: str,
    cell: MALTCell, derived: DerivedSource, expected_init: str, runtime: dict[str, object],
) -> dict[str, object]:
    run_name = f"mainconf_r1_{cell.method}_{mode_name(args)}_{cell.cell_id}_seed{args.seed}_{batch_id}"
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
            and sha256_file(checkpoint_path) == summary.get("checkpoint_sha256")
        )
        if (
            summary.get("evidence_valid") is True
            and summary.get("cell_id") == cell.cell_id
            and summary.get("controlled_seed") == args.seed
            and summary.get("init_sha256") == expected_init
            and summary.get("total_steps") == total_steps(args)
            and manifest.get("derived_source_sha256") == derived.derived_sha256
            and metrics_path.is_file()
            and checkpoint_ok
        ):
            rows = list(csv.DictReader(metrics_path.open(encoding="utf-8", newline="")))
            if not (args.formal_smoke or args.numerical_smoke) and not wandb_status_satisfies(
                args.wandb_mode, summary.get("wandb_status")
            ):
                upload = upload_wandb(args, run_dir, run_name, batch_id, cell, rows, summary, runtime, derived)
                summary["wandb_status"] = upload["status"]
                manifest["wandb"] = upload
                manifest["status"] = "completed_valid" if wandb_status_satisfies(args.wandb_mode, upload["status"]) else "completed_valid_local_wandb_incomplete"
                write_json(summary_path, summary)
                write_json(manifest_path, manifest)
            if wandb_status_satisfies(args.wandb_mode, summary.get("wandb_status")):
                manifest["status"] = "completed_valid"
                manifest["summary"] = summary
                write_json(manifest_path, manifest)
            print(f"Resume: reusing completed MALT cell {cell.cell_id}")
            return summary
        raise RuntimeError(f"incompatible existing run directory: {run_dir}")
    workspaces = run_dir / "workspaces"
    existing_attempts = sorted(workspaces.glob("attempt_*")) if workspaces.is_dir() else []
    workspace = workspaces / f"attempt_{len(existing_attempts) + 1:03d}"
    script = materialize_source(workspace, official_repo, derived)
    (run_dir / "official_r1_to_malt.patch").write_text(derived.unified_diff, encoding="utf-8")
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
    manifest["status"] = "completed_valid" if wandb_status_satisfies(args.wandb_mode, upload["status"]) else "completed_valid_local_wandb_incomplete"
    write_json(summary_path, summary)
    write_json(manifest_path, manifest)
    return summary


def validate_selection(
    path: Path, selected_method: str
) -> tuple[MALTCell, dict[str, object]]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    failures = []
    if payload.get("status") != "selected" or payload.get("protocol") != SELECTION_PROTOCOL:
        failures.append("selection status/protocol mismatch")
    if (
        payload.get("formal_allowed") is not True
        or payload.get("scientific_result") != "dual_methods_selected"
        or payload.get("certificate_role") != "independent_pilot_analysis_selection"
    ):
        failures.append("selection was not issued by the independent pilot analyzer")
    if payload.get("seed") != 2026 or payload.get("pilot_steps") != PILOT_STEPS:
        failures.append("selection was not made from the frozen seed-2026 1000-step pilot")
    if payload.get("required_formal_methods") != list(FORMAL_METHODS):
        failures.append("selection does not require the complete dual-method formal panel")
    selections = payload.get("selections")
    if not isinstance(selections, dict) or set(selections) != set(FORMAL_METHODS):
        failures.append("selection does not contain exactly MALT and MALTER-Eq17")
        selections = {}
    for method in FORMAL_METHODS:
        entry = selections.get(method)
        if not isinstance(entry, dict):
            failures.append(f"selection entry missing for {method}")
            continue
        if (
            entry.get("method") != method
            or entry.get("status") != "selected"
            or entry.get("formal_allowed") is not True
            or entry.get("formal_eligible") is not True
            or entry.get("boundary_rule_triggered") is not False
        ):
            failures.append(f"selection entry is not formal-eligible for {method}")
    entry = selections.get(selected_method, {})
    selected = str(entry.get("selected_cell_id"))
    matches = [cell for cell in PILOT_CELLS if cell.cell_id == selected]
    if (
        len(matches) != 1
        or matches[0].method != selected_method
        or not matches[0].formal_eligible
        or float(entry.get("selected_matrix_lr", -1)) != matches[0].matrix_lr
    ):
        failures.append(f"selected {selected_method} cell/LR is outside the frozen grid")
    if failures:
        raise RuntimeError("selection certificate rejected:\n- " + "\n- ".join(failures))
    return matches[0], {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "validated_selected_method": selected_method,
        **payload,
    }


def validate_pilot_source_lineage(
    selection: dict[str, object], current_source_audit: dict[str, object]
) -> dict[str, object]:
    manifest_path = Path(str(selection.get("pilot_manifest", ""))).expanduser().resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"selection pilot manifest is missing: {manifest_path}")
    expected_hash = str(selection.get("pilot_manifest_sha256", ""))
    if expected_hash != sha256_file(manifest_path):
        raise RuntimeError("selection pilot-manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_audits: list[dict[str, object]] = []
    if isinstance(manifest.get("source_audit"), dict):
        observed_audits.append(manifest["source_audit"])
    source_manifests = manifest.get("source_manifests", [])
    if source_manifests:
        if not isinstance(source_manifests, list) or len(source_manifests) != len(PILOT_CELLS):
            raise RuntimeError("aggregate pilot source-manifest coverage mismatch")
        for record in source_manifests:
            if not isinstance(record, dict):
                raise RuntimeError("invalid aggregate pilot source-manifest record")
            path = Path(str(record.get("path", ""))).expanduser().resolve()
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise RuntimeError(f"pilot source-manifest lineage failed: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            audit = payload.get("source_audit")
            if not isinstance(audit, dict):
                raise RuntimeError(f"pilot source audit missing: {path}")
            observed_audits.append(audit)
    if not observed_audits:
        raise RuntimeError("selection pilot manifest does not expose source lineage")
    if any(audit != current_source_audit for audit in observed_audits):
        raise RuntimeError("pilot/formal source audit mismatch; a new pilot is required")
    return {
        "status": "accepted",
        "pilot_manifest": str(manifest_path),
        "pilot_manifest_sha256": expected_hash,
        "source_audit_count": len(observed_audits),
    }


def validate_smoke(
    path: Path, args: argparse.Namespace, cell: MALTCell, runtime: dict[str, object],
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
    if summary.get("evidence_valid") is not True:
        failures.append("smoke summary was not locally accepted")
    expected_label = {
        "malt": "MALT-R1 adaptation",
        "malter_eq17": "MALTER-Eq17-R1 adaptation",
    }[cell.method]
    expected_bytes = {
        "malt": MALT_EXPECTED_HIDDEN_STATE_BYTES,
        "malter_eq17": MALTER_EXPECTED_HIDDEN_STATE_BYTES,
    }[cell.method]
    if summary.get("method") != cell.method or summary.get("adaptation_label") != expected_label:
        failures.append("smoke method/adaptation label mismatch")
    if int(summary.get("hidden_optimizer_state_bytes", -1)) != expected_bytes:
        failures.append("smoke hidden-state byte audit mismatch")
    roles = summary.get("state_schema", {}).get("roles", {})
    if roles.get("malt_momentum") != 48 or roles.get("malt_row_ema") != 72 or roles.get("malt_col_ema") != 72:
        failures.append("smoke did not instantiate the complete 48/72/72 MALT state")
    if cell.method == "malter_eq17":
        if roles.get("malt_nu") != 72 or int(summary.get("malt_nu_bytes", -1)) != 288:
            failures.append("smoke did not instantiate the 72-scalar MALTER Eq.(17) state")
    elif roles.get("malt_nu") is not None or int(summary.get("malt_nu_bytes", -1)) != 0:
        failures.append("MALT smoke unexpectedly contains MALTER scalar state")
    if failures:
        raise RuntimeError("formal smoke certificate rejected:\n- " + "\n- ".join(failures))
    return {"path": str(resolved), "sha256": sha256_file(resolved), "status": "accepted"}


def method_selection_payload(
    rows: list[dict[str, object]],
    *,
    method: str,
    center_cell_id: str | None,
    lower_boundary_cell_id: str,
    upper_boundary_cell_id: str,
) -> dict[str, object]:
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["final_val_loss"]),
            float(row["matrix_lr"]),
            str(row["cell_id"]),
        ),
    )
    center = (
        next(row for row in rows if row["cell_id"] == center_cell_id)
        if center_cell_id is not None
        else None
    )
    minimum_loss = float(ranked[0]["final_val_loss"])
    minimum_rows = [
        row for row in ranked if float(row["final_val_loss"]) == minimum_loss
    ]
    minimum_cell_ids = [str(row["cell_id"]) for row in minimum_rows]
    minimum_includes_lower_boundary = lower_boundary_cell_id in minimum_cell_ids
    minimum_includes_upper_boundary = upper_boundary_cell_id in minimum_cell_ids
    center_within_margin = bool(
        center is not None
        and float(center["final_val_loss"])
        <= float(ranked[0]["final_val_loss"]) + TIE_MARGIN
    )
    chosen = center if center_within_margin else ranked[0]
    assert chosen is not None
    raw_best = ranked[0]
    boundary = minimum_includes_lower_boundary or minimum_includes_upper_boundary
    boundary_side = (
        "both"
        if minimum_includes_lower_boundary and minimum_includes_upper_boundary
        else "lower"
        if minimum_includes_lower_boundary
        else "upper"
        if minimum_includes_upper_boundary
        else None
    )
    return {
        "method": method,
        "selection_policy": (
            "paper_center_within_best_plus_0.002"
            if center_cell_id is not None
            else "raw_endpoint_best"
        ),
        "center_cell_id": center_cell_id,
        "center_tie_margin": TIE_MARGIN if center_cell_id is not None else None,
        "status": "boundary_inconclusive" if boundary else "selected",
        "formal_allowed": not boundary,
        "formal_eligible": not boundary,
        "selected_cell_id": None if boundary else chosen["cell_id"],
        "selected_matrix_lr": None if boundary else chosen["matrix_lr"],
        "selection_reason": (
            "boundary_inconclusive"
            if boundary
            else "paper_center_within_best_plus_0.002"
            if center_within_margin
            else "raw_endpoint_best"
        ),
        "raw_best_cell_id": raw_best["cell_id"],
        "raw_best_matrix_lr": raw_best["matrix_lr"],
        "raw_best_final_val_loss": raw_best["final_val_loss"],
        "minimum_tied_cell_ids": minimum_cell_ids,
        "minimum_includes_lower_boundary": minimum_includes_lower_boundary,
        "minimum_includes_upper_boundary": minimum_includes_upper_boundary,
        "boundary_tie_policy": "any_boundary_at_reported_minimum_blocks_formal",
        "boundary_rule_triggered": boundary,
        "boundary_side": boundary_side,
        "ranked_cells": [
            {
                "cell_id": row["cell_id"],
                "matrix_lr": row["matrix_lr"],
                "final_val_loss": row["final_val_loss"],
            }
            for row in ranked
        ],
    }


def make_selection(batch_dir: Path, summaries: list[dict[str, object]], manifest_path: Path) -> Path:
    if len(summaries) != len(PILOT_CELLS) or {row["cell_id"] for row in summaries} != {cell.cell_id for cell in PILOT_CELLS}:
        raise RuntimeError("a selection certificate requires all twelve frozen V4 pilot cells")
    if any(row.get("evidence_valid") is not True for row in summaries):
        raise RuntimeError("a selection certificate requires twelve locally accepted pilot cells")
    malt_rows = [row for row in summaries if row.get("method") == "malt"]
    malter_rows = [row for row in summaries if row.get("method") == "malter_eq17"]
    if len(malt_rows) != 6 or len(malter_rows) != 6:
        raise RuntimeError("pilot method coverage is not 6 MALT + 6 MALTER-Eq17")
    selections = {
        "malt": method_selection_payload(
            malt_rows,
            method="malt",
            center_cell_id=None,
            lower_boundary_cell_id=MALT_LOWER_BOUNDARY_CELL_ID,
            upper_boundary_cell_id=MALT_UPPER_BOUNDARY_CELL_ID,
        ),
        "malter_eq17": method_selection_payload(
            malter_rows,
            method="malter_eq17",
            center_cell_id=MALTER_CENTER_CELL_ID,
            lower_boundary_cell_id=MALTER_LOWER_BOUNDARY_CELL_ID,
            upper_boundary_cell_id=MALTER_UPPER_BOUNDARY_CELL_ID,
        ),
    }
    blocking_methods = [
        method
        for method in FORMAL_METHODS
        if selections[method]["boundary_rule_triggered"] is True
    ]
    formal_allowed = not blocking_methods
    path = batch_dir / "pilot_selection.json"
    write_json(
        path,
        {
            "status": "selected" if formal_allowed else "boundary_inconclusive",
            "protocol": SELECTION_PROTOCOL, "seed": 2026,
            "certificate_role": "runner_preselection_crosscheck",
            "scientific_result": "dual_methods_selected" if formal_allowed else "boundary_inconclusive",
            "formal_allowed": formal_allowed,
            "required_formal_methods": list(FORMAL_METHODS),
            "blocking_methods": blocking_methods,
            "pilot_steps": PILOT_STEPS, "selection_endpoint": "step-1000 validation loss",
            "grid_design": "fresh_v4_focused_malt_upper_grid_dual_method",
            "malt_execution_order": [
                cell.matrix_lr for cell in PILOT_CELLS if cell.method == "malt"
            ],
            "malt_selection_policy": "raw_endpoint_best",
            "malter_center_tie_margin": TIE_MARGIN,
            "malter_center_preferred_if_within_margin_of_best": True,
            "pilot_manifest": str(manifest_path.resolve()), "pilot_manifest_sha256": sha256_file(manifest_path),
            "selections": selections,
            # Compatibility aliases are diagnostics only; formal consumers must
            # validate the complete nested dual-method envelope above.
            "selected_cell_id": selections["malt"]["selected_cell_id"],
            "selected_matrix_lr": selections["malt"]["selected_matrix_lr"],
            "formal_eligible": formal_allowed,
            "boundary_rule_triggered": bool(blocking_methods),
            "boundary_side": selections["malt"]["boundary_side"],
            "raw_best_cell_id": selections["malt"]["raw_best_cell_id"],
            "malt_ranked_cells": selections["malt"]["ranked_cells"],
            "malter_eq17_role": "formal_candidate",
            "malter_eq17_selected_cell_id": selections["malter_eq17"]["selected_cell_id"],
            "malter_eq17_selected_matrix_lr": selections["malter_eq17"]["selected_matrix_lr"],
            "malter_eq17_ranked_cells": selections["malter_eq17"]["ranked_cells"],
        },
    )
    return path


def main() -> None:
    args = parse_args()
    repo = args.official_repo.expanduser().resolve()
    provenance = r0.validate_official_repo(repo)
    data = r0.validate_data(repo)
    data_dir = Path(str(data["data_dir"]))
    data_inventory = validate_data_inventory_certificate(
        args.data_inventory_certificate, data_dir
    )
    runtime = r0.validate_runtime(repo, args.python_exe)
    exact_runtime = validate_exact_training_runtime(runtime)
    training_runtime_fingerprint = r0.runtime_fingerprint(runtime)
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
        selected_method = args.selected_method
        if selection_path is None and resume_plan is not None:
            saved_selection = resume_plan.get("selection_certificate", {})
            if isinstance(saved_selection, dict) and saved_selection.get("path"):
                selection_path = Path(str(saved_selection["path"]))
        if selected_method is None and resume_plan is not None:
            saved_method = resume_plan.get("selected_method")
            if isinstance(saved_method, str):
                selected_method = saved_method
        if selection_path is None:
            raise RuntimeError("formal resume cannot recover its selection certificate path")
        if selected_method not in FORMAL_METHODS:
            raise RuntimeError("formal resume cannot recover its selected method")
        cell, selection = validate_selection(selection_path, selected_method)
        cells = [cell]
    else:
        cells = selected_pilot_cells(args) if not args.preflight else [next(cell for cell in PILOT_CELLS if cell.cell_id == MALT_REFERENCE_CELL_ID)]
    if args.preflight:
        initialization_audits = {
            method: initialization_audit(
                args,
                repo,
                data_dir,
                derived,
                next(
                    cell
                    for cell in PILOT_CELLS
                    if cell.cell_id
                    == (MALT_REFERENCE_CELL_ID if method == "malt" else MALTER_CENTER_CELL_ID)
                ),
            )
            for method in FORMAL_METHODS
        }
        if len(
            {
                str(audit.get("init_sha256", ""))
                for audit in initialization_audits.values()
            }
        ) != 1:
            raise RuntimeError("MALT and MALTER preflight initializations do not match")
        init_audit = initialization_audits["malt"]
    else:
        initialization_audits = None
        init_audit = initialization_audit(args, repo, data_dir, derived, cells[0])
    small_audit = reference_audit(args)
    source_audit = {
        "official_r1_base_sha256": derived.base_canonical_sha256,
        "derived_source_sha256": derived.derived_sha256,
        "malt_optimizer_sha256": sha256_file(SCRIPT_DIR / "malt_optimizer.py"),
        "malt_source_builder_sha256": sha256_file(SCRIPT_DIR / "malt_source_builder.py"),
        "contract_sha256": sha256_file(SCRIPT_DIR / "malt_contract.json"),
        "paper_derivation_sha256": sha256_file(SCRIPT_DIR / "PAPER_DERIVATION.md"),
        "r0_controller_sha256": sha256_file(
            R0_DIR / "run_official_newton_muon_r0.py"
        ),
        "paper_arxiv_id": "2608.05088v1",
        "implementation_label": "paper-derived independent implementation",
        "official_code_public_at_freeze": False,
        "data_inventory_certificate_sha256": data_inventory["sha256"],
    }
    selection_source_lineage = (
        validate_pilot_source_lineage(selection, source_audit)
        if selection is not None
        else None
    )
    if args.preflight:
        stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        artifact = args.results_dir.expanduser().resolve() / f"{stamp}_preflight_seed{args.seed}.json"
        write_json(artifact, {"status": "passed", "protocol": PREFLIGHT_PROTOCOL, "official_provenance": provenance, "data": data, "data_inventory": data_inventory, "runtime": runtime, "exact_runtime_contract": exact_runtime, "controller_runtime": controller, "wandb_readiness": wandb_readiness, "initialization_audit": init_audit, "initialization_audits": initialization_audits, "small_matrix_reference_audit": small_audit, "source_audit": source_audit})
        print(f"R1 MALT-family preflight artifact: {artifact}")
        return

    smoke_certificate = None
    if args.formal:
        smoke_path = args.smoke_manifest
        if smoke_path is None and resume_plan is not None:
            saved_smoke = resume_plan.get("smoke_certificate", {})
            if isinstance(saved_smoke, dict) and saved_smoke.get("path"):
                smoke_path = Path(str(saved_smoke["path"]))
        if smoke_path is None:
            raise RuntimeError("formal resume cannot recover its smoke manifest path")
        smoke_certificate = validate_smoke(
            smoke_path, args, cells[0], runtime, init_audit, derived
        )
    stem = mode_name(args)
    if args.resume_batch:
        batch_dir = args.resume_batch.expanduser().resolve()
        plan_path = batch_dir / f"{stem}_plan.json"
        plan = resume_plan if resume_plan is not None else json.loads(plan_path.read_text(encoding="utf-8"))
        if (
            plan.get("protocol") != protocol(args)
            or plan.get("seed") != args.seed
            or plan.get("source_audit") != source_audit
            or plan.get("data_inventory") != data_inventory
            or plan.get("training_runtime_fingerprint") != training_runtime_fingerprint
            or plan.get("wandb_mode") != args.wandb_mode
            or plan.get("wandb_entity") != args.wandb_entity
            or plan.get("wandb_base_url") != os.environ.get("WANDB_BASE_URL", "default")
        ):
            raise RuntimeError("resume plan protocol/seed/source mismatch")
        if plan.get("cell", {}).get("cell_id") not in {cell.cell_id for cell in cells} and len(cells) == 1:
            raise RuntimeError("resume selected cell mismatch")
        if (args.formal or args.formal_smoke) and plan.get("selected_method") != cells[0].method:
            raise RuntimeError("resume selected method mismatch")
        if args.formal and plan.get("smoke_certificate") != smoke_certificate:
            raise RuntimeError("resume smoke certificate changed or failed revalidation")
        if plan.get("selection_certificate") != selection:
            raise RuntimeError("resume selection certificate changed or failed revalidation")
        batch_id = str(plan["batch_id"])
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
            "selected_method": cells[0].method if (args.formal or args.formal_smoke) else None,
            "formal_evidence": bool(args.formal), "timing_eligible": False,
            "official_provenance": provenance, "data": data,
            "data_inventory": data_inventory, "runtime": runtime,
            "exact_runtime_contract": exact_runtime,
            "training_runtime_fingerprint": training_runtime_fingerprint, "controller_runtime": controller,
            "wandb_readiness": wandb_readiness,
            "initialization_audit": init_audit, "small_matrix_reference_audit": small_audit,
            "source_audit": source_audit, "selection_certificate": selection,
            "selection_source_lineage": selection_source_lineage,
            "smoke_certificate": smoke_certificate, "wandb_project": args.wandb_project,
            "wandb_mode": args.wandb_mode, "wandb_entity": args.wandb_entity,
            "wandb_base_url": os.environ.get("WANDB_BASE_URL", "default"),
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
    wandb_complete = all(
        wandb_status_satisfies(args.wandb_mode, row.get("wandb_status"))
        for row in summaries
    )
    final = {
        **plan,
        "status": "completed_valid" if len(summaries) == len(cells) and not failures and wandb_complete else "completed_valid_local_wandb_incomplete" if len(summaries) == len(cells) and not failures else "failed",
        "summaries": ranked, "summary": ranked[0] if len(ranked) == 1 else None,
        "failures": failures, "wandb_complete": wandb_complete,
    }
    manifest_path = batch_dir / f"{stem}_manifest.json"
    write_json(manifest_path, final)
    if args.pilot and args.seed == 2026 and args.pilot_steps == PILOT_STEPS and not failures and len(cells) == len(PILOT_CELLS):
        selection_path = make_selection(batch_dir, ranked, manifest_path)
        print(f"R1 MALT-family selection certificate: {selection_path}")
    print(f"R1 MALT-family artifacts: {batch_dir}")
    if failures:
        raise SystemExit(1)
    if (args.pilot or args.formal) and not wandb_complete:
        print("local evidence is valid but W&B upload is incomplete; resume the batch", file=sys.stderr)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
