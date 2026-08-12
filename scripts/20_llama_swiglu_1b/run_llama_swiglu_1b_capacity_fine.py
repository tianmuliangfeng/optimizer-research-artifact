"""Sequential fixed-accumulation fine sweep near the 1B H100 OOM boundary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_llama_swiglu_1b_capacity as coarse


HERE = Path(__file__).resolve().parent
WORKER = HERE / "run_llama_swiglu_1b_capacity_fine_cell.py"
CAPACITY_TRAINER = HERE / "train_llama_swiglu_1b_capacity.py"
METHODS = ("down_none", "down_diag", "newton_full", "muon")
ACCUMULATION_STEPS = 8
DEFAULT_BATCHES = list(range(32, 65, 2))
MIN_BASELINE_FREE_FRACTION = 0.98


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1B fine OOM sweep with fixed accumulation=8")
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", type=Path, required=True, help="CUDA training interpreter")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device-batches", nargs="+", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(args.methods) != len(set(args.methods)):
        parser.error("--methods contains duplicates")
    batches = sorted(set(args.device_batches))
    if not batches or any(batch < 32 or batch > 64 for batch in batches):
        parser.error("fine-capacity device batches must be within [32, 64]")
    if 32 not in batches:
        parser.error("fine-capacity grid must include batch 32 as the protocol anchor")
    args.device_batches = batches
    if args.output_root is None:
        repo_root = HERE.parents[1]
        results_root = Path(
            os.environ.get("SNM_RESULTS_ROOT", str(repo_root / "runs"))
        ).expanduser()
        args.output_root = results_root / "20_llama_swiglu_1b" / "capacity_fine"
    return args


def global_batch(batch: int) -> int:
    return batch * ACCUMULATION_STEPS


def validate_gpu_baseline(baseline: dict[str, int] | None) -> dict[str, int]:
    if baseline is None:
        raise RuntimeError("could not query target GPU memory before capacity cell")
    free_bytes = int(baseline["free_bytes"])
    total_bytes = int(baseline["total_bytes"])
    free_fraction = free_bytes / total_bytes
    if free_fraction < MIN_BASELINE_FREE_FRACTION:
        raise RuntimeError(
            "target GPU is not sufficiently idle for an OOM boundary test: "
            f"free={free_bytes} total={total_bytes} fraction={free_fraction:.4f}; "
            f"required>={MIN_BASELINE_FREE_FRACTION:.2f}"
        )
    return baseline


def row_from_summary(
    method: str,
    batch: int,
    summary: dict[str, Any],
    manifest: Path,
    baseline: dict[str, int] | None,
) -> dict[str, Any]:
    return {
        "method": method,
        "device_batch_size": batch,
        "global_batch_size": global_batch(batch),
        "sequence_length": 1024,
        "accumulation_steps": ACCUMULATION_STEPS,
        "status": "completed",
        "failure_class": "",
        "completed_steps": summary.get("completed_steps"),
        "peak_allocated_bytes": summary.get("peak_allocated_bytes"),
        "peak_allocated_mib": summary.get("peak_allocated_mib"),
        "peak_reserved_bytes": summary.get("peak_reserved_bytes"),
        "peak_reserved_mib": summary.get("peak_reserved_mib"),
        "model_parameter_bytes": summary.get("model_parameter_bytes"),
        "optimizer_state_bytes": summary.get("optimizer_state_bytes"),
        "k_state_bytes": summary.get("k_state_bytes"),
        "preconditioner_workspace_bytes": summary.get("preconditioner_workspace_bytes"),
        "baseline_free_bytes": baseline.get("free_bytes") if baseline else None,
        "gpu_total_memory_bytes": baseline.get("total_bytes") if baseline else summary.get("runtime", {}).get("gpu_total_memory_bytes"),
        "source": "capacity_fine_grid",
        "cell_manifest": str(manifest.resolve()),
        "error": "",
    }


def failed_row(
    method: str,
    batch: int,
    failure: str,
    return_code: int,
    log_path: Path,
    manifest_path: Path | None,
    baseline: dict[str, int] | None,
) -> dict[str, Any]:
    return {
        "method": method,
        "device_batch_size": batch,
        "global_batch_size": global_batch(batch),
        "sequence_length": 1024,
        "accumulation_steps": ACCUMULATION_STEPS,
        "status": "failed",
        "failure_class": failure,
        "completed_steps": None,
        "peak_allocated_bytes": None,
        "peak_allocated_mib": None,
        "peak_reserved_bytes": None,
        "peak_reserved_mib": None,
        "model_parameter_bytes": None,
        "optimizer_state_bytes": None,
        "k_state_bytes": None,
        "preconditioner_workspace_bytes": None,
        "baseline_free_bytes": baseline.get("free_bytes") if baseline else None,
        "gpu_total_memory_bytes": baseline.get("total_bytes") if baseline else None,
        "source": "capacity_fine_grid",
        "cell_manifest": str(manifest_path.resolve()) if manifest_path else "",
        "error": f"return_code={return_code}; see {log_path}",
    }


def run_cell(args: argparse.Namespace, batch_root: Path, method: str, batch: int) -> dict[str, Any]:
    baseline = validate_gpu_baseline(coarse.gpu_memory(args.python_exe))
    cell_root = batch_root / "cells" / method / f"batch_{batch}"
    cell_root.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(WORKER),
        "--stage",
        "smoke",
        "--official-repo",
        str(args.official_repo),
        "--python-exe",
        str(args.python_exe),
        "--output-root",
        str(cell_root),
        "--methods",
        method,
        "--seed",
        str(args.seed),
        "--device-batch-size",
        str(batch),
        "--capacity-accumulation-steps",
        str(ACCUMULATION_STEPS),
        "--wandb-mode",
        "disabled",
    ]
    log_path = cell_root / "capacity_fine_cell.log"
    chunks: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
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
    manifests = sorted(cell_root.glob("*/llama_manifest.json"))
    manifest_path = manifests[-1] if manifests else None
    if return_code == 0 and manifest_path:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = manifest.get("method_results", {}).get(method, {})
        if result.get("status") == "completed":
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            if summary.get("completed_steps") != 34:
                raise RuntimeError(f"fine-capacity cell did not complete 34 steps: {manifest_path}")
            return row_from_summary(method, batch, summary, manifest_path, baseline)
    return failed_row(
        method,
        batch,
        coarse.classify_failure("".join(chunks)),
        return_code,
        log_path,
        manifest_path,
        baseline,
    )


def boundary_summary(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    boundaries: dict[str, dict[str, Any]] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        success = [row["device_batch_size"] for row in method_rows if row["status"] == "completed"]
        oom = [row["device_batch_size"] for row in method_rows if row["failure_class"] == "oom"]
        boundaries[method] = {
            "max_tested_success_batch": max(success) if success else None,
            "first_tested_oom_batch": min(oom) if oom else None,
            "resolved_to_batch_two": bool(success and oom and min(oom) - max(success) == 2),
        }
    return boundaries


def main() -> None:
    args = parse_args()
    plan = {
        "protocol": "microbatch_capacity_fine_fixed_accumulation_v1",
        "evidence_class": "capacity_only",
        "methods": args.methods,
        "device_batches": args.device_batches,
        "seed": args.seed,
        "steps_per_cell": 34,
        "sequence_length": 1024,
        "accumulation_steps": ACCUMULATION_STEPS,
        "global_batch_rule": "device_batch_size * accumulation_steps",
        "stop_after_first_failure_per_method": True,
        "minimum_pre_cell_free_gpu_fraction": MIN_BASELINE_FREE_FRACTION,
        "quality_comparable": False,
        "timing_eligible": False,
    }
    print("1B fine fixed-accumulation capacity plan")
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S+0000") + f"_capacity_fine_seed{args.seed}"
    batch_root = args.output_root.resolve() / batch_id
    batch_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    manifest = {
        "status": "running",
        "evidence_class": "capacity_only",
        "created_at": coarse.now_iso(),
        "batch_id": batch_id,
        "plan": plan,
        "official_repo": str(args.official_repo.resolve()),
        "python_exe": str(args.python_exe),
        "fine_controller_sha256": coarse.sha256_file(Path(__file__)),
        "fine_worker_sha256": coarse.sha256_file(WORKER),
        "capacity_trainer_sha256": coarse.sha256_file(CAPACITY_TRAINER),
        "rows": rows,
    }
    coarse.atomic_json(batch_root / "capacity_fine_manifest.json", manifest)
    for method in args.methods:
        for batch in args.device_batches:
            row = run_cell(args, batch_root, method, batch)
            rows.append(row)
            manifest["rows"] = rows
            manifest["boundaries"] = boundary_summary(rows, args.methods)
            manifest["last_updated_at"] = coarse.now_iso()
            coarse.atomic_json(batch_root / "capacity_fine_manifest.json", manifest)
            coarse.write_csv(batch_root / "capacity_fine_results.csv", rows)
            if row["status"] != "completed":
                if row["failure_class"] != "oom":
                    manifest["status"] = "failed"
                    manifest["failed_at"] = coarse.now_iso()
                    manifest["fatal_cell"] = {
                        "method": method,
                        "device_batch_size": batch,
                        "failure_class": row["failure_class"],
                        "error": row["error"],
                    }
                    coarse.atomic_json(batch_root / "capacity_fine_manifest.json", manifest)
                    raise RuntimeError(
                        f"non-OOM fine-capacity failure for {method} batch {batch}; "
                        f"see {row['error']}"
                    )
                break
    manifest["status"] = "completed"
    manifest["completed_at"] = coarse.now_iso()
    manifest["rows"] = rows
    manifest["boundaries"] = boundary_summary(rows, args.methods)
    coarse.atomic_json(batch_root / "capacity_fine_manifest.json", manifest)
    coarse.atomic_json(batch_root / "capacity_fine_boundaries.json", manifest["boundaries"])
    coarse.write_csv(batch_root / "capacity_fine_results.csv", rows)
    print(f"Fine-capacity artifacts:  {batch_root}")
    print(f"Fine-capacity manifest:   {batch_root / 'capacity_fine_manifest.json'}")
    print(f"Fine-capacity boundaries: {batch_root / 'capacity_fine_boundaries.json'}")


if __name__ == "__main__":
    main()
