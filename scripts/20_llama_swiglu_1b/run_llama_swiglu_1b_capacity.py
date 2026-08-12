"""Sequential fixed-memory/OOM grid for the pinned 1.014B profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKER = HERE / "run_llama_swiglu_1b_capacity_cell.py"
CAPACITY_TRAINER = HERE / "train_llama_swiglu_1b_capacity.py"
METHODS = ("down_none", "down_diag", "newton_full")
OOM_RE = re.compile(r"out of memory|OutOfMemoryError|CUDA error:\s*out of memory", re.IGNORECASE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1B fixed-memory/OOM capacity grid")
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", type=Path, required=True, help="CUDA training interpreter")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device-batches", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--baseline-smoke-manifests", nargs="*", type=Path, default=[])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(args.methods) != len(set(args.methods)):
        parser.error("--methods contains duplicates")
    batches = sorted(set(args.device_batches))
    if any(batch <= 8 or 512 % batch for batch in batches):
        parser.error("capacity device batches must be >8 and divide global batch 512")
    args.device_batches = batches
    if args.output_root is None:
        repo_root = HERE.parents[1]
        results_root = Path(
            os.environ.get("SNM_RESULTS_ROOT", str(repo_root / "runs"))
        ).expanduser()
        args.output_root = results_root / "20_llama_swiglu_1b" / "capacity"
    return args


def classify_failure(text: str) -> str:
    return "oom" if OOM_RE.search(text) else "error"


def gpu_memory(python_exe: Path) -> dict[str, int] | None:
    code = "import json,torch; f,t=torch.cuda.mem_get_info(); print(json.dumps({'free_bytes':int(f),'total_bytes':int(t)}))"
    result = subprocess.run([str(python_exe), "-c", code], text=True, capture_output=True, check=False)
    if result.returncode:
        return None
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return None


def baseline_rows(paths: list[Path], methods: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or payload.get("execution_stage") != "smoke":
            raise RuntimeError(f"not a completed 1B smoke manifest: {path}")
        if payload.get("config", {}).get("device_batch_size") != 8:
            raise RuntimeError(f"baseline smoke is not device batch 8: {path}")
        for method, result in payload.get("method_results", {}).items():
            if method not in methods or result.get("status") != "completed":
                continue
            summary_path = Path(result["summary_path"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(row_from_summary(method, 8, summary, "existing_smoke", path, None))
            seen.add(method)
    missing = set(methods) - seen
    if missing:
        raise RuntimeError(f"baseline smoke manifests do not cover {sorted(missing)}")
    return rows


def row_from_summary(method: str, batch: int, summary: dict[str, Any], source: str, manifest: Path, baseline: dict[str, int] | None) -> dict[str, Any]:
    return {
        "method": method,
        "device_batch_size": batch,
        "global_batch_size": 512,
        "sequence_length": 1024,
        "accumulation_steps": 512 // batch,
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
        "source": source,
        "cell_manifest": str(manifest.resolve()),
        "error": "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_cell(args: argparse.Namespace, batch_root: Path, method: str, batch: int) -> dict[str, Any]:
    cell_root = batch_root / "cells" / method / f"batch_{batch}"
    cell_root.mkdir(parents=True, exist_ok=False)
    baseline = gpu_memory(args.python_exe)
    command = [
        sys.executable, str(WORKER), "--stage", "smoke",
        "--official-repo", str(args.official_repo), "--python-exe", str(args.python_exe),
        "--output-root", str(cell_root), "--methods", method, "--seed", str(args.seed),
        "--device-batch-size", str(batch), "--wandb-mode", "disabled",
    ]
    log_path = cell_root / "capacity_cell.log"
    chunks: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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
            return row_from_summary(method, batch, summary, "capacity_grid", manifest_path, baseline)
    text = "".join(chunks)
    failure = classify_failure(text)
    return {
        "method": method, "device_batch_size": batch, "global_batch_size": 512,
        "sequence_length": 1024, "accumulation_steps": 512 // batch,
        "status": "failed", "failure_class": failure, "completed_steps": None,
        "peak_allocated_bytes": None, "peak_allocated_mib": None,
        "peak_reserved_bytes": None, "peak_reserved_mib": None,
        "model_parameter_bytes": None, "optimizer_state_bytes": None, "k_state_bytes": None,
        "preconditioner_workspace_bytes": None,
        "baseline_free_bytes": baseline.get("free_bytes") if baseline else None,
        "gpu_total_memory_bytes": baseline.get("total_bytes") if baseline else None,
        "source": "capacity_grid", "cell_manifest": str(manifest_path.resolve()) if manifest_path else "",
        "error": f"return_code={return_code}; see {log_path}",
    }


def main() -> None:
    args = parse_args()
    plan = {"methods": args.methods, "device_batches": args.device_batches, "seed": args.seed, "steps_per_cell": 34, "sequence_length": 1024, "global_batch_size": 512, "stop_after_first_failure_per_method": True}
    print("1B fixed-memory/OOM capacity plan")
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S+0000") + f"_capacity_seed{args.seed}"
    batch_root = args.output_root.resolve() / batch_id
    batch_root.mkdir(parents=True, exist_ok=False)
    rows = baseline_rows(args.baseline_smoke_manifests, args.methods)
    manifest = {
        "status": "running", "evidence_class": "capacity_only", "created_at": now_iso(),
        "batch_id": batch_id, "plan": plan,
        "official_repo": str(args.official_repo.resolve()), "python_exe": str(args.python_exe),
        "capacity_controller_sha256": sha256_file(Path(__file__)),
        "capacity_worker_sha256": sha256_file(WORKER),
        "capacity_trainer_sha256": sha256_file(CAPACITY_TRAINER),
        "baseline_smoke_manifests": [{"path": str(p.resolve()), "sha256": sha256_file(p.resolve())} for p in args.baseline_smoke_manifests],
        "rows": rows,
    }
    atomic_json(batch_root / "capacity_manifest.json", manifest)
    for method in args.methods:
        for batch in args.device_batches:
            row = run_cell(args, batch_root, method, batch)
            rows.append(row)
            manifest["rows"] = rows
            manifest["last_updated_at"] = now_iso()
            atomic_json(batch_root / "capacity_manifest.json", manifest)
            write_csv(batch_root / "capacity_results.csv", rows)
            if row["status"] != "completed":
                break
    manifest["status"] = "completed"
    manifest["completed_at"] = now_iso()
    manifest["rows"] = rows
    atomic_json(batch_root / "capacity_manifest.json", manifest)
    write_csv(batch_root / "capacity_results.csv", rows)
    print(f"Capacity artifacts: {batch_root}")
    print(f"Capacity manifest:  {batch_root / 'capacity_manifest.json'}")


if __name__ == "__main__":
    main()
