"""Capacity-only Moonlight-Muon OOM boundary on LLaMA/SwiGLU-124M.

The controller first tests a predeclared coarse device-batch grid, stops at
the first CUDA OOM, and then resolves the success/OOM interval to one integer
with a predeclared binary search.  Global batch is device batch times eight,
so quality and timing results are intentionally ineligible for comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import run_llama_swiglu_extended as quality


HERE = Path(__file__).resolve().parent
CAPACITY_TRAINER = HERE / "train_llama_swiglu_extended_capacity.py"
CELL = quality.CELL_BY_ID["moonlight_high"]
PROTOCOL = "llama124m_moonlight_capacity_binary_v1"
DEFAULT_DEVICE_BATCHES = (64, 96, 128, 160, 192, 224, 256)
ACCUMULATION_STEPS = 8
STEPS_PER_CELL = 34
MIN_BASELINE_FREE_FRACTION = 0.98
OOM_RE = re.compile(
    r"out of memory|OutOfMemoryError|CUDA error:\s*out of memory", re.IGNORECASE
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Moonlight-only LLaMA-124M fixed-accumulation OOM boundary"
    )
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument(
        "--device-batches",
        nargs="+",
        type=int,
        default=list(DEFAULT_DEVICE_BATCHES),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    batches = sorted(set(args.device_batches))
    if not batches or batches[0] != 64:
        parser.error("capacity grid must start with the formal device-batch anchor 64")
    if any(batch <= 0 or batch > 512 for batch in batches):
        parser.error("capacity device batches must be in [1, 512]")
    args.device_batches = batches
    if args.seed != 2026:
        parser.error("the capacity-only boundary is frozen to seed2026")
    args.cells = [CELL.cell_id]
    if args.output_root is None:
        args.output_root = quality.default_results_dir().parent / "capacity_moonlight"
    return args


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_memory(python_exe: Path) -> dict[str, int] | None:
    code = (
        "import json,torch; f,t=torch.cuda.mem_get_info(); "
        "print(json.dumps({'free_bytes':int(f),'total_bytes':int(t)}))"
    )
    result = subprocess.run(
        [str(python_exe), "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        return None
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return None


def validate_gpu_baseline(baseline: dict[str, int] | None) -> dict[str, int]:
    if baseline is None:
        raise RuntimeError("could not query target GPU memory before capacity cell")
    free_bytes = int(baseline["free_bytes"])
    total_bytes = int(baseline["total_bytes"])
    fraction = free_bytes / total_bytes
    if fraction < MIN_BASELINE_FREE_FRACTION:
        raise RuntimeError(
            "target GPU is not idle enough for an OOM boundary: "
            f"free={free_bytes} total={total_bytes} fraction={fraction:.4f}; "
            f"required>={MIN_BASELINE_FREE_FRACTION:.2f}"
        )
    return baseline


def classify_failure(text: str) -> str:
    return "oom" if OOM_RE.search(text) else "error"


def replace_argument(command: list[str], flag: str, value: str) -> None:
    try:
        command[command.index(flag) + 1] = value
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"generated command has no {flag}") from exc


def capacity_command(
    args: argparse.Namespace, batch: int, output_dir: Path
) -> list[str]:
    command_args = SimpleNamespace(
        python_exe=args.python_exe,
        official_repo=args.official_repo,
        device_batch_size=batch,
        checkpoint_every=128,
    )
    command = quality.train_command(
        command_args, CELL, args.seed, output_dir, "formal_smoke"
    )
    command[1] = str(CAPACITY_TRAINER)
    replace_argument(
        command, "--global-batch-size", str(batch * ACCUMULATION_STEPS)
    )
    replace_argument(command, "--device-batch-size", str(batch))
    replace_argument(
        command, "--val-tokens", str(batch * quality.SEQUENCE_LENGTH)
    )
    return command


def validate_summary(
    path: Path,
    batch: int,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    config = summary.get("config", {})
    extended = summary.get("extended_optimizer", {})
    expected_recipe = (
        CELL.method,
        CELL.auxiliary_lr,
        CELL.matrix_lr,
        CELL.weight_decay,
    )
    observed_recipe = (
        extended.get("method"),
        extended.get("auxiliary_lr"),
        extended.get("matrix_lr"),
        extended.get("weight_decay"),
    )
    if (
        summary.get("status") != "completed"
        or summary.get("method") != CELL.method
        or summary.get("seed") != 2026
        or summary.get("completed_steps") != STEPS_PER_CELL
    ):
        failures.append("status/method/seed/step mismatch")
    if (
        config.get("device_batch_size") != batch
        or config.get("global_batch_size") != batch * ACCUMULATION_STEPS
        or config.get("sequence_length") != quality.SEQUENCE_LENGTH
        or config.get("val_tokens") != batch * quality.SEQUENCE_LENGTH
        or config.get("checkpoint_every") != 0
        or config.get("resume") != "never"
        or not config.get("no_save_final")
    ):
        failures.append("capacity config mismatch")
    if observed_recipe != expected_recipe:
        failures.append(
            f"Moonlight recipe mismatch: {observed_recipe} != {expected_recipe}"
        )
    if (
        summary.get("model_parameter_bytes")
        != quality.EXPECTED_PARAMETER_COUNT * 4
        or summary.get("optimizer_state_bytes") != 648_671_336
        or summary.get("k_state_bytes") != 0
        or summary.get("preconditioner_workspace_bytes") != 0
    ):
        failures.append("model/optimizer-state byte contract mismatch")
    if (
        summary.get("evidence_class") != "capacity_only"
        or summary.get("quality_comparable") is not False
        or summary.get("timing_comparable") is not False
    ):
        failures.append("capacity evidence labels are missing")
    if quality.base.stable_runtime(summary.get("runtime", {})) != quality.base.stable_runtime(
        runtime
    ):
        failures.append("runtime differs from controller")
    for field in (
        "final_val_loss",
        "final_train_loss",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "optimizer_state_bytes",
    ):
        value = summary.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            failures.append(f"{field} is non-finite")
    try:
        quality.base.validate_metric_evidence(
            path.with_name("metrics.csv"),
            total_steps=STEPS_PER_CELL,
            val_every=STEPS_PER_CELL,
            global_batch_size=batch * ACCUMULATION_STEPS,
            sequence_length=quality.SEQUENCE_LENGTH,
        )
    except Exception as exc:
        failures.append(f"metric evidence invalid: {exc}")
    if summary.get("checkpoint_path") != "":
        failures.append("capacity cell unexpectedly retained a checkpoint")
    if failures:
        raise RuntimeError(
            f"capacity summary rejected for device batch {batch}:\n- "
            + "\n- ".join(failures)
        )
    return summary


def completed_row(
    batch: int,
    phase: str,
    summary: dict[str, Any],
    baseline: dict[str, int],
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "cell_id": CELL.cell_id,
        "method": CELL.method,
        "phase": phase,
        "device_batch_size": batch,
        "global_batch_size": batch * ACCUMULATION_STEPS,
        "sequence_length": quality.SEQUENCE_LENGTH,
        "accumulation_steps": ACCUMULATION_STEPS,
        "status": "completed",
        "failure_class": "",
        "completed_steps": summary["completed_steps"],
        "peak_allocated_bytes": summary["peak_allocated_bytes"],
        "peak_allocated_mib": summary["peak_allocated_mib"],
        "peak_reserved_bytes": summary["peak_reserved_bytes"],
        "peak_reserved_mib": summary["peak_reserved_mib"],
        "model_parameter_bytes": summary["model_parameter_bytes"],
        "optimizer_state_bytes": summary["optimizer_state_bytes"],
        "k_state_bytes": summary["k_state_bytes"],
        "preconditioner_workspace_bytes": summary[
            "preconditioner_workspace_bytes"
        ],
        "baseline_free_bytes": baseline["free_bytes"],
        "gpu_total_memory_bytes": baseline["total_bytes"],
        "run_dir": str(run_dir.resolve()),
        "error": "",
    }


def failed_row(
    batch: int,
    phase: str,
    failure_class: str,
    return_code: int,
    baseline: dict[str, int],
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "cell_id": CELL.cell_id,
        "method": CELL.method,
        "phase": phase,
        "device_batch_size": batch,
        "global_batch_size": batch * ACCUMULATION_STEPS,
        "sequence_length": quality.SEQUENCE_LENGTH,
        "accumulation_steps": ACCUMULATION_STEPS,
        "status": "failed",
        "failure_class": failure_class,
        "completed_steps": "",
        "peak_allocated_bytes": "",
        "peak_allocated_mib": "",
        "peak_reserved_bytes": "",
        "peak_reserved_mib": "",
        "model_parameter_bytes": "",
        "optimizer_state_bytes": "",
        "k_state_bytes": "",
        "preconditioner_workspace_bytes": "",
        "baseline_free_bytes": baseline["free_bytes"],
        "gpu_total_memory_bytes": baseline["total_bytes"],
        "run_dir": str(run_dir.resolve()),
        "error": f"return_code={return_code}; see {run_dir / 'terminal.log'}",
    }


def run_cell(
    args: argparse.Namespace,
    batch_root: Path,
    batch: int,
    phase: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    baseline = validate_gpu_baseline(gpu_memory(args.python_exe))
    run_dir = batch_root / "cells" / f"{phase}_batch_{batch}"
    run_dir.mkdir(parents=True, exist_ok=False)
    command = capacity_command(args, batch, run_dir)
    chunks: list[str] = []
    with (run_dir / "terminal.log").open("w", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(command) + "\n")
        process = subprocess.Popen(
            command,
            env=quality.subprocess_env(args.official_repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            chunks.append(line)
        return_code = process.wait()
    summary_path = run_dir / "summary.json"
    if return_code == 0 and summary_path.is_file():
        summary = validate_summary(summary_path, batch, runtime)
        return completed_row(batch, phase, summary, baseline, run_dir)
    return failed_row(
        batch,
        phase,
        classify_failure("".join(chunks)),
        return_code,
        baseline,
        run_dir,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boundary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [
        int(row["device_batch_size"])
        for row in rows
        if row["status"] == "completed"
    ]
    oom = [
        int(row["device_batch_size"])
        for row in rows
        if row["failure_class"] == "oom"
    ]
    lower = max(successes) if successes else None
    upper = min((value for value in oom if lower is None or value > lower), default=None)
    return {
        "max_success_device_batch": lower,
        "first_oom_device_batch": upper,
        "resolved_width": upper - lower
        if isinstance(lower, int) and isinstance(upper, int)
        else None,
        "exact_integer_boundary": bool(
            isinstance(lower, int) and isinstance(upper, int) and upper - lower == 1
        ),
        "max_success_global_batch": lower * ACCUMULATION_STEPS
        if isinstance(lower, int)
        else None,
        "quality_comparable": False,
        "timing_eligible": False,
    }


def main() -> None:
    args = parse_args()
    plan = {
        "protocol": PROTOCOL,
        "evidence_class": "capacity_only",
        "cell": {
            "cell_id": CELL.cell_id,
            "method": CELL.method,
            "auxiliary_lr": CELL.auxiliary_lr,
            "matrix_lr": CELL.matrix_lr,
            "weight_decay": CELL.weight_decay,
        },
        "coarse_device_batches": args.device_batches,
        "refinement": "integer binary search between max success and first OOM",
        "seed": args.seed,
        "steps_per_cell": STEPS_PER_CELL,
        "sequence_length": quality.SEQUENCE_LENGTH,
        "accumulation_steps": ACCUMULATION_STEPS,
        "global_batch_rule": "device_batch_size * accumulation_steps",
        "minimum_pre_cell_free_gpu_fraction": MIN_BASELINE_FREE_FRACTION,
        "quality_comparable": False,
        "timing_eligible": False,
    }
    print("Moonlight LLaMA-124M capacity plan")
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    runtime = quality.base.validate_runtime(args.python_exe, args.official_repo)
    data = quality.base.audit_data(args.official_repo / "data" / "fineweb10B")
    bundle = quality.source_bundle()
    pilot = quality.validate_pilot_selection(
        args.pilot_manifest, args, runtime, data, bundle
    )
    batch_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S+0000")
        + "_capacity_moonlight_seed2026"
    )
    batch_root = args.output_root.resolve() / batch_id
    batch_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    manifest_path = batch_root / "capacity_manifest.json"
    results_path = batch_root / "capacity_results.csv"
    boundary_path = batch_root / "capacity_boundary.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "created_at": now_iso(),
        "batch_id": batch_id,
        "plan": plan,
        "official_repo": str(args.official_repo.resolve()),
        "python_exe": str(args.python_exe.resolve()),
        "runtime": runtime,
        "data_fingerprint": data["fingerprint"],
        "pilot_selection_manifest": str(args.pilot_manifest.resolve()),
        "pilot_selection_manifest_sha256": quality.sha256_file(
            args.pilot_manifest.resolve()
        ),
        "pilot_selection_certificate_sha256": quality.canonical_hash(pilot),
        "source_sha256": {
            "capacity_controller": quality.sha256_file(Path(__file__)),
            "capacity_trainer": quality.sha256_file(CAPACITY_TRAINER),
            "quality_runner": quality.sha256_file(Path(quality.__file__)),
            "adapter": bundle["sha256"]["adapter"],
            "base_trainer": bundle["sha256"]["base_trainer"],
            "extended_optimizers": bundle["sha256"]["extended_optimizers"],
        },
        "rows": rows,
    }
    quality.atomic_json(manifest_path, manifest)

    def execute(batch: int, phase: str) -> dict[str, Any]:
        row = run_cell(args, batch_root, batch, phase, runtime)
        rows.append(row)
        manifest["rows"] = rows
        manifest["boundary"] = boundary(rows)
        manifest["last_updated_at"] = now_iso()
        quality.atomic_json(manifest_path, manifest)
        write_csv(results_path, rows)
        if row["status"] != "completed" and row["failure_class"] != "oom":
            manifest["status"] = "failed"
            manifest["fatal_cell"] = row
            manifest["failed_at"] = now_iso()
            quality.atomic_json(manifest_path, manifest)
            raise RuntimeError(
                f"non-OOM capacity failure at batch {batch}; see {row['error']}"
            )
        return row

    first_oom: int | None = None
    last_success: int | None = None
    for batch in args.device_batches:
        row = execute(batch, "coarse")
        if row["status"] == "completed":
            last_success = batch
            continue
        first_oom = batch
        break
    if last_success is not None and first_oom is not None:
        lower, upper = last_success, first_oom
        while upper - lower > 1:
            middle = (lower + upper) // 2
            row = execute(middle, "binary")
            if row["status"] == "completed":
                lower = middle
            else:
                upper = middle

    final_boundary = boundary(rows)
    manifest["status"] = "completed"
    manifest["completed_at"] = now_iso()
    manifest["boundary"] = final_boundary
    quality.atomic_json(manifest_path, manifest)
    quality.atomic_json(boundary_path, final_boundary)
    write_csv(results_path, rows)
    print(f"Moonlight capacity artifacts: {batch_root}")
    print(f"Moonlight capacity manifest:  {manifest_path}")
    print(f"Moonlight capacity boundary:  {boundary_path}")


if __name__ == "__main__":
    main()
