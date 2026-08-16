#!/usr/bin/env python3
"""One/two-GPU, staged and accepted-unit-resumable controller for Experiment 51."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGES = ("preflight", "pilot", "formal", "upload", "verify", "all")
SCALES = ("275m", "455m")
FORMAL_SEEDS = {"275m": (2024, 2025, 2026, 2027), "455m": (2024, 2025, 2026)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--data-repo-root", type=Path, required=True)
    parser.add_argument("--training-python", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--wandb-project", default="anonymous-optimizer-artifact-ex51")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(args.gpus) not in (1, 2) or len(set(args.gpus)) != len(args.gpus):
        parser.error("Experiment 51 requires one or two distinct physical GPU ids")
    return args


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(args: argparse.Namespace) -> Path:
    root = args.run_dir / "source_snapshot"
    manifest = root / "source_snapshot_manifest.json"
    if manifest.is_file():
        payload = read_json(manifest)
        if payload.get("passed") is not True:
            raise RuntimeError(f"invalid EX51 source snapshot: {manifest}")
        if payload.get("official_repo") != str(args.official_repo):
            raise RuntimeError(
                "EX51 source snapshot was built from a different official-r0 path: "
                f"{payload.get('official_repo')} != {args.official_repo}"
            )
        for row in payload.get("files", []):
            path = root / row["path"]
            if not path.is_file() or sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"source snapshot drift: {path}")
        return root
    scripts = args.repo / "scripts"
    for name in ("43_newton_muon_record28_275m", "44_newton_muon_record17_455m", "51_moddedgpt_global_diag_scale"):
        shutil.copytree(scripts / name, root / "scripts" / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    wrapper = (
        args.repo
        / "commands/51_moddedgpt_global_diag_scale/20260814_ex51_moddedgpt_global_diag_scale.sh"
    )
    (root / "commands").mkdir(parents=True, exist_ok=True)
    shutil.copy2(wrapper, root / "commands" / wrapper.name)
    builder = load_module("ex51_frozen_builder", root / "scripts/51_moddedgpt_global_diag_scale/global_diag_scale_source_builder.py")
    training = root / "training"
    training.mkdir(parents=True, exist_ok=True)
    derived = {}
    for scale in SCALES:
        built = builder.build_source(args.official_repo, scale)
        target = training / f"train_global_diag_{scale}.py"
        target.write_text(built.source, encoding="utf-8", newline="\n")
        (training / f"train_global_diag_{scale}.diff").write_text(built.unified_diff, encoding="utf-8", newline="\n")
        derived[scale] = {"path": target.relative_to(root).as_posix(), "sha256": sha256_file(target), "parent_sha256": built.parent_sha256}
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest:
            files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(manifest, {"schema_version": 1, "experiment_id": "51_moddedgpt_global_diag_scale", "passed": True, "created_at": datetime.now(timezone.utc).isoformat(), "official_repo": str(args.official_repo), "derived_training_sources": derived, "file_count": len(files), "files": files})
    return root


def preflight(args: argparse.Namespace) -> None:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    status = args.run_dir / "preflight/preflight_manifest.json"
    if status.is_file():
        prior = read_json(status)
        if prior.get("passed") is True and prior.get("requested_gpus") != args.gpus:
            raise RuntimeError(
                "EX51 GPU allocation is frozen by preflight: "
                f"{prior.get('requested_gpus')} != {args.gpus}. "
                "Use the original EX51_GPUS value or start a fresh run directory."
            )
        if (
            prior.get("passed") is True
            and prior.get("official_repo") == str(args.official_repo)
            and prior.get("data_repo_root") == str(args.data_repo_root)
        ):
            snapshot(args)
            print("skip passed EX51 preflight", flush=True)
            return
    snap = snapshot(args)
    contract = read_json(
        snap / "scripts/51_moddedgpt_global_diag_scale/global_diag_scale_contract.json"
    )
    common = load_module("ex51_preflight_common", snap / "scripts/43_newton_muon_record28_275m/record28_common.py")
    official = common.audit_official_repo(args.official_repo)
    data_path = args.run_dir / "preflight/data_certificate.json"
    cached_data_path = args.data_repo_root / "data/fineweb10B/ex51_data_certificate.json"
    previous = data_path if data_path.is_file() else cached_data_path
    data = common.audit_fineweb_data(
        args.data_repo_root,
        previous_certificate=previous if previous.is_file() else None,
        hash_workers=2,
    )
    write_json(data_path, data)
    write_json(cached_data_path, data)
    runtime_module = load_module("ex51_runtime_cell", snap / "scripts/43_newton_muon_record28_275m/run_record28_cell.py")
    runtimes = {gpu: runtime_module.runtime_probe(args.training_python, gpu) for gpu in args.gpus}
    checks = {
        "source_snapshot": True,
        "official_repo": official.get("passed") is True,
        "data_inventory": data.get("passed") is True,
        "data_file_count_51": len(data.get("files", [])) == 51,
        "data_fingerprint_exact": data.get("fingerprint_sha256")
        == contract["data"]["accepted_fingerprint_sha256"],
        "data_view_within_official_r0": args.data_repo_root != args.official_repo
        and args.data_repo_root.is_relative_to(args.official_repo),
        "runtime_requested_gpu_count": len(runtimes) in (1, 2)
        and len(runtimes) == len(args.gpus),
        "runtime_all_passed": all(row.get("passed") is True for row in runtimes.values()),
    }
    payload = {"schema_version": 1, "experiment_id": "51_moddedgpt_global_diag_scale", "passed": all(checks.values()), "checks": checks, "requested_gpus": list(args.gpus), "requested_gpu_count": len(args.gpus), "gpu_allocation_frozen": True, "official_repo": str(args.official_repo), "data_repo_root": str(args.data_repo_root), "data_fingerprint_sha256": data.get("fingerprint_sha256"), "official": official, "runtime": runtimes, "data_certificate": str(data_path), "data_certificate_sha256": sha256_file(data_path), "source_snapshot_manifest_sha256": sha256_file(snap / "source_snapshot_manifest.json")}
    write_json(status, payload)
    if not payload["passed"]:
        raise RuntimeError(f"Experiment-51 preflight failed: {checks}")


def accepted_attempt(unit: Path, stage: str) -> Path | None:
    accepted = []
    for cert in sorted(unit.glob("attempt_*/ex51_unit_manifest.json")):
        scientific = cert.with_name("scientific_manifest.json")
        if cert.is_file() and scientific.is_file():
            a, b = read_json(cert), read_json(scientific)
            if a.get("passed") is True and b.get("passed") is True and b.get("stage") == stage:
                accepted.append(cert.parent)
    if len(accepted) > 1:
        raise RuntimeError(f"multiple accepted attempts: {unit}")
    return accepted[0] if accepted else None


def require_frozen_gpu_allocation(args: argparse.Namespace) -> None:
    path = args.run_dir / "preflight/preflight_manifest.json"
    if not path.is_file():
        raise RuntimeError("passed Experiment-51 preflight is required")
    payload = read_json(path)
    if payload.get("passed") is not True:
        raise RuntimeError("passed Experiment-51 preflight is required")
    if payload.get("requested_gpus") != args.gpus:
        raise RuntimeError(
            "EX51 does not support changing GPU allocation after preflight: "
            f"{payload.get('requested_gpus')} != {args.gpus}"
        )


def next_attempt(unit: Path) -> Path:
    indexes = [int(path.name.split("_")[-1]) for path in unit.glob("attempt_*") if path.name.split("_")[-1].isdigit()]
    return unit / f"attempt_{max(indexes, default=0)+1:03d}"


def command(
    args: argparse.Namespace,
    scale: str,
    seed: int,
    stage: str,
    attempt: Path,
    gpu: str,
    *,
    upload_only: bool = False,
) -> list[str]:
    snap = args.run_dir / "source_snapshot"
    cmd = [str(sys.executable), "-B", str(snap / "scripts/51_moddedgpt_global_diag_scale/run_global_diag_scale_cell.py"), "--attempt-dir", str(attempt), "--stage", stage, "--seed", str(seed), "--method", "global_diag", "--physical-gpu", gpu, "--training-python", str(args.training_python), "--training-source", str(snap / f"training/train_global_diag_{scale}.py"), "--contract", str(snap / "scripts/51_moddedgpt_global_diag_scale/global_diag_scale_contract.json"), "--source-snapshot-manifest", str(snap / "source_snapshot_manifest.json"), "--data-certificate", str(args.run_dir / "preflight/data_certificate.json"), "--data-repo-root", str(args.data_repo_root), "--result-root", str(args.run_dir), "--wandb-mode", "disabled" if stage == "smoke" else args.wandb_mode, "--wandb-project", args.wandb_project]
    if args.wandb_entity:
        cmd.extend(["--wandb-entity", args.wandb_entity])
    if upload_only:
        cmd.append("--upload-only")
    return cmd


def run_units(args: argparse.Namespace, formal: bool) -> None:
    stage_name = "formal" if formal else "pilot"
    cell_stage = "formal" if formal else "smoke"
    require_frozen_gpu_allocation(args)
    pilot_path = args.run_dir / "pilot/pilot_manifest.json"
    if formal and (
        not pilot_path.is_file() or read_json(pilot_path).get("passed") is not True
    ):
        raise RuntimeError("accepted Experiment-51 pilot is required")
    # Validate the newly derived 275M route first in pilot.  Formal still puts
    # long 455M units first to minimize the one/two-GPU makespan.
    scale_order = ("455m", "275m") if formal else ("275m", "455m")
    pending = [
        (scale, seed)
        for scale in scale_order
        for seed in (FORMAL_SEEDS[scale] if formal else (2026,))
    ]
    pending = [(s, seed) for s, seed in pending if accepted_attempt(args.run_dir / stage_name / s / f"seed{seed}", cell_stage) is None]
    active: dict[str, tuple[subprocess.Popen[str], Any, str, int, Path, Path]] = {}
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            scale, seed = pending.pop(0)
            unit = args.run_dir / stage_name / scale / f"seed{seed}"
            attempt = next_attempt(unit)
            attempt.mkdir(parents=True, exist_ok=False)
            log = attempt / "controller.log"
            handle = log.open("a", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env["EX51_SCALE"] = scale
            env["CUDA_VISIBLE_DEVICES"] = gpu
            proc = subprocess.Popen(command(args, scale, seed, cell_stage, attempt, gpu), cwd=args.run_dir, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            active[gpu] = (proc, handle, scale, seed, attempt, log)
            print(f"started {stage_name}/{scale}/seed{seed} on gpu {gpu}", flush=True)
        time.sleep(1)
        for gpu, item in list(active.items()):
            proc, handle, scale, seed, attempt, log = item
            rc = proc.poll()
            if rc is None:
                continue
            handle.close()
            del active[gpu]
            if rc != 0 or accepted_attempt(attempt.parent, cell_stage) != attempt:
                survivors = list(active.values())
                for other in survivors:
                    other[0].terminate()
                for other in survivors:
                    try:
                        other[0].wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        other[0].kill()
                        other[0].wait(timeout=30)
                    other[1].close()
                active.clear()
                raise RuntimeError(f"failed {stage_name}/{scale}/seed{seed} rc={rc}; see {log}")
            print(f"passed {stage_name}/{scale}/seed{seed}", flush=True)
    units = [
        {
            "scale": scale,
            "seed": seed,
            "attempt": str(
                accepted_attempt(
                    args.run_dir / stage_name / scale / f"seed{seed}", cell_stage
                )
            ),
        }
        for scale in SCALES
        for seed in (FORMAL_SEEDS[scale] if formal else (2026,))
    ]
    manifest = {"schema_version": 1, "experiment_id": "51_moddedgpt_global_diag_scale", "stage": stage_name, "passed": all(row["attempt"] != "None" for row in units), "unit_count": len(units), "units": units}
    write_json(args.run_dir / stage_name / f"{stage_name}_manifest.json", manifest)


def run_upload(args: argparse.Namespace) -> None:
    require_frozen_gpu_allocation(args)
    formal_manifest_path = args.run_dir / "formal/formal_manifest.json"
    if (
        not formal_manifest_path.is_file()
        or read_json(formal_manifest_path).get("passed") is not True
    ):
        raise RuntimeError("accepted Experiment-51 formal results are required for upload")
    if args.wandb_mode == "disabled":
        raise RuntimeError(
            "upload requires --wandb-mode online or offline; accepted formal results remain intact"
        )

    rows: list[dict[str, Any]] = []
    units = [
        (scale, seed) for scale in SCALES for seed in FORMAL_SEEDS[scale]
    ]
    for index, (scale, seed) in enumerate(units):
        unit = args.run_dir / "formal" / scale / f"seed{seed}"
        attempt = accepted_attempt(unit, "formal")
        if attempt is None:
            raise RuntimeError(f"missing accepted formal unit: {scale}/seed{seed}")
        receipt_path = attempt / "wandb.json"
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            if receipt.get("complete") is True and receipt.get("status") == "completed":
                rows.append(
                    {
                        "scale": scale,
                        "seed": seed,
                        "attempt": str(attempt),
                        "receipt": str(receipt_path),
                        "status": "already_completed",
                    }
                )
                continue

        gpu = args.gpus[index % len(args.gpus)]
        log_path = attempt / "wandb_upload_controller.log"
        env = os.environ.copy()
        env["EX51_SCALE"] = scale
        env["CUDA_VISIBLE_DEVICES"] = gpu
        with log_path.open("a", encoding="utf-8", buffering=1) as handle:
            completed = subprocess.run(
                command(
                    args,
                    scale,
                    seed,
                    "formal",
                    attempt,
                    gpu,
                    upload_only=True,
                ),
                cwd=args.run_dir,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if not receipt_path.is_file():
            raise RuntimeError(
                f"W&B upload produced no receipt for {scale}/seed{seed}; see {log_path}"
            )
        receipt = read_json(receipt_path)
        if (
            completed.returncode != 0
            or receipt.get("complete") is not True
            or receipt.get("status") != "completed"
        ):
            raise RuntimeError(
                f"W&B upload failed for {scale}/seed{seed} rc={completed.returncode}; "
                f"accepted scientific result is preserved; see {log_path}"
            )
        rows.append(
            {
                "scale": scale,
                "seed": seed,
                "attempt": str(attempt),
                "receipt": str(receipt_path),
                "status": "completed",
            }
        )
        print(f"uploaded formal/{scale}/seed{seed}", flush=True)

    write_json(
        args.run_dir / "upload/upload_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": "51_moddedgpt_global_diag_scale",
            "stage": "upload",
            "passed": len(rows) == sum(len(seeds) for seeds in FORMAL_SEEDS.values()),
            "unit_count": len(rows),
            "wandb_mode": args.wandb_mode,
            "wandb_project": args.wandb_project,
            "units": rows,
        },
    )


def verify(args: argparse.Namespace) -> None:
    snap = snapshot(args)
    subprocess.run([str(sys.executable), "-B", str(snap / "scripts/51_moddedgpt_global_diag_scale/analyze_global_diag_scale.py"), "--run-dir", str(args.run_dir), "--contract", str(snap / "scripts/51_moddedgpt_global_diag_scale/global_diag_scale_contract.json"), "--controls", str(snap / "scripts/51_moddedgpt_global_diag_scale/frozen_scale_controls.csv")], check=True)
    analysis = read_json(args.run_dir / "analysis/analysis_manifest.json")
    checks = {"preflight": read_json(args.run_dir / "preflight/preflight_manifest.json").get("passed") is True, "pilot": read_json(args.run_dir / "pilot/pilot_manifest.json").get("passed") is True, "formal": read_json(args.run_dir / "formal/formal_manifest.json").get("passed") is True, "analysis": analysis.get("passed") is True}
    upload_manifest_path = args.run_dir / "upload/upload_manifest.json"
    upload_status: dict[str, Any] = {"requested": False, "passed": None}
    if upload_manifest_path.is_file():
        upload_manifest = read_json(upload_manifest_path)
        upload_status = {
            "requested": True,
            "passed": upload_manifest.get("passed") is True,
            "manifest": str(upload_manifest_path),
        }
    write_json(args.run_dir / "handoff_manifest.json", {"schema_version": 1, "experiment_id": "51_moddedgpt_global_diag_scale", "status": "completed" if all(checks.values()) else "failed", "passed": all(checks.values()), "checks": checks, "analysis": str(args.run_dir / "analysis/analysis_manifest.json"), "wandb_upload": upload_status})
    if not all(checks.values()):
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    for field in ("run_dir", "repo", "official_repo", "data_repo_root"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.data_repo_root == args.official_repo or not args.data_repo_root.is_relative_to(
        args.official_repo
    ):
        raise RuntimeError(
            "EX51_DATA_REPO_ROOT must be a dedicated frozen view inside Newton-Muon-official-r0"
        )
    args.training_python = args.training_python.expanduser().absolute()
    if args.stage not in ("preflight", "all"):
        require_frozen_gpu_allocation(args)
    if args.stage in ("preflight", "all"):
        preflight(args)
    if args.stage in ("pilot", "all"):
        run_units(args, False)
    if args.stage in ("formal", "all"):
        run_units(args, True)
    if args.stage == "upload" or (args.stage == "all" and args.wandb_mode != "disabled"):
        run_upload(args)
    if args.stage in ("verify", "all"):
        verify(args)
    print(f"EX51 stage {args.stage} completed. Artifacts: {args.run_dir}")


if __name__ == "__main__":
    main()
