#!/usr/bin/env python3
"""Fail-closed two-GPU staged controller for EX52 LLaMA global-diagonal runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGES = ("preflight", "pilot", "screen", "formal", "verify", "all")
SCALES = ("124m", "1b")
SEEDS = (2024, 2025, 2026)
CURRENT_CONTRACT_VERSION = "2026-08-14.4"
AMENDABLE_SNAPSHOT_VERSIONS = ("2026-08-14.3",)
EXPECTED_DATA_NAMES = [
    "fineweb_val_000000.bin",
    *[f"fineweb_train_{index:06d}.bin" for index in range(1, 51)],
]
EXPECTED_SHARD_BYTES = 200_001_024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-python", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--wandb-project", default="anonymous-optimizer-artifact-ex52")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(args.gpus) != 2 or len(set(args.gpus)) != 2:
        parser.error("Experiment 52 requires exactly two distinct GPU ids")
    return args


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def contract_path(snapshot_root: Path) -> Path:
    return snapshot_root / "scripts/52_llama_global_diag_scale/llama_global_diag_contract.json"


def source_snapshot(args: argparse.Namespace) -> Path:
    root = args.run_dir / "source_snapshot"
    manifest = root / "source_snapshot_manifest.json"
    if manifest.is_file():
        payload = read_json(manifest)
        checks = {
            "passed": payload.get("passed") is True,
            "experiment": payload.get("experiment_id") == "52_llama_global_diag_scale",
            "repo": payload.get("repo") == str(args.repo),
            "official_repo": payload.get("official_repo") == str(args.official_repo),
            "contract_version": payload.get("contract_version") in (*AMENDABLE_SNAPSHOT_VERSIONS, CURRENT_CONTRACT_VERSION),
            "file_count": int(payload.get("file_count", -1)) == len(payload.get("files", [])),
        }
        frozen_contract = read_json(contract_path(root))
        for relative, expected in frozen_contract.get("official_r0_source_sha256", {}).items():
            live = args.official_repo / relative
            checks[f"official:{relative}"] = live.is_file() and sha256_file(live) == expected
        for row in payload.get("files", []):
            path = root / row["path"]
            checks[f"file:{row['path']}"] = (
                path.is_file()
                and path.stat().st_size == int(row["bytes"])
                and sha256_file(path) == row["sha256"]
            )
        if not all(checks.values()):
            raise RuntimeError(f"EX52 source snapshot drift: {checks}")
        return root

    ex52 = args.repo / "scripts/52_llama_global_diag_scale"
    contract = read_json(ex52 / "llama_global_diag_contract.json")
    if contract.get("contract_version") != CURRENT_CONTRACT_VERSION:
        raise RuntimeError(f"EX52 requires the {CURRENT_CONTRACT_VERSION} contract")
    parent_hashes = {}
    for relative, expected in contract["parent_source_sha256"].items():
        path = args.repo / relative
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"accepted parent source drift: {relative}: {observed} != {expected}")
        parent_hashes[relative] = observed
    official_hashes = {}
    for relative, expected in contract["official_r0_source_sha256"].items():
        path = args.official_repo / relative
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"official-r0 source drift: {relative}: {observed} != {expected}")
        official_hashes[relative] = observed

    shutil.copytree(
        ex52,
        root / "scripts/52_llama_global_diag_scale",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    wrapper = (
        args.repo
        / "commands/52_llama_global_diag_scale/20260814_ex52_llama_global_diag_scale.sh"
    )
    (root / "commands").mkdir(parents=True, exist_ok=True)
    shutil.copy2(wrapper, root / "commands" / wrapper.name)
    builder = load_module(
        "ex52_frozen_builder",
        root / "scripts/52_llama_global_diag_scale/llama_global_diag_source_builder.py",
    )
    built = builder.build(args.repo)
    generated = {
        "scripts/17_llama_swiglu_validation/train_llama_swiglu.py": built.trainer,
        "scripts/17_llama_swiglu_validation/run_llama_swiglu_validation.py": built.runner124,
        "scripts/20_llama_swiglu_1b/run_llama_swiglu_1b.py": built.runner1b,
        "scripts/20_llama_swiglu_1b/train_llama_swiglu_1b.py": built.wrapper1b,
    }
    derived = {}
    for relative, text in generated.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        derived[relative] = sha256_file(path)
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest
    ]
    write_json(
        manifest,
        {
            "schema_version": 2,
            "experiment_id": "52_llama_global_diag_scale",
            "passed": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repo": str(args.repo),
            "official_repo": str(args.official_repo),
            "contract_version": contract["contract_version"],
            "contract_sha256": sha256_file(contract_path(root)),
            "parent_source_sha256": parent_hashes,
            "official_r0_source_sha256": official_hashes,
            "derived_source_sha256": derived,
            "expected_k_state_bytes": {
                "124m": builder.expected_k_state_bytes("124m"),
                "1b": builder.expected_k_state_bytes("1b"),
            },
            "file_count": len(files),
            "files": files,
        },
    )
    return root


def runner(args: argparse.Namespace, scale: str) -> Path:
    snap = source_snapshot(args)
    relative = (
        "scripts/17_llama_swiglu_validation/run_llama_swiglu_validation.py"
        if scale == "124m"
        else "scripts/20_llama_swiglu_1b/run_llama_swiglu_1b.py"
    )
    return snap / relative


def data_header(path: Path) -> dict[str, Any]:
    stat = path.stat()
    with path.open("rb") as handle:
        header = handle.read(1024)
    if len(header) != 1024:
        raise RuntimeError(f"truncated FineWeb header: {path}")
    magic, version, num_tokens = struct.unpack_from("<iii", header, 0)
    checks = {
        "magic": magic == 20240520,
        "version": version == 1,
        "tokens": num_tokens == 100_000_000,
        "bytes": stat.st_size == 1024 + 2 * num_tokens == EXPECTED_SHARD_BYTES,
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid FineWeb shard {path}: {checks}")
    return {
        "path": str(path.absolute()),
        "resolved_path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "magic": magic,
        "version": version,
        "num_tokens": num_tokens,
    }


def data_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(Path(row["path"]).name.encode("utf-8"))
        digest.update(str(row["bytes"]).encode("ascii"))
        digest.update(str(row["num_tokens"]).encode("ascii"))
        digest.update(row["sha256"].encode("ascii"))
    return digest.hexdigest()


def controller_inventory_payload(data_dir: str, data_certificate: dict[str, Any]) -> dict[str, Any]:
    """Reproduce the accepted parent controllers' path-bound lightweight audit.

    The parent audit intentionally records ``data_dir``.  Consequently its
    fingerprint is not portable across two directory names even when every
    shard byte is identical.  EX52 therefore derives the expected lightweight
    fingerprint from the already full-hash-certified view instead of pinning
    the fingerprint observed at the historical parent path.
    """
    rows = {Path(row["path"]).name: row for row in data_certificate.get("files", [])}
    train_names = [f"fineweb_train_{index:06d}.bin" for index in range(1, 51)]
    val_name = "fineweb_val_000000.bin"
    expected_names = [*train_names, val_name]
    if set(rows) != set(expected_names):
        raise RuntimeError("cannot derive parent inventory from a non-canonical EX52 data certificate")

    def header(name: str) -> dict[str, Any]:
        row = rows[name]
        return {"name": name, "bytes": int(row["bytes"]), "tokens": int(row["num_tokens"])}

    payload = {
        "data_dir": data_dir,
        "train_shard_count": 50,
        "val_shard_count": 1,
        "first_train": header(train_names[0]),
        "last_train": header(train_names[-1]),
        "validation": header(val_name),
        "total_bytes": sum(int(rows[name]["bytes"]) for name in expected_names),
        "ordered_names": expected_names,
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def audit_data(args: argparse.Namespace, contract: dict[str, Any]) -> dict[str, Any]:
    expected_source = args.official_repo / "data/fineweb10B"
    if args.data_dir == args.official_repo or not args.data_dir.is_relative_to(args.official_repo):
        raise RuntimeError("EX52 data view must be a dedicated directory inside official-r0")
    observed = sorted(path.name for path in args.data_dir.glob("fineweb_*.bin"))
    if observed != sorted(EXPECTED_DATA_NAMES):
        raise RuntimeError(
            f"EX52 FineWeb inventory mismatch: missing={sorted(set(EXPECTED_DATA_NAMES)-set(observed))}, "
            f"extra={sorted(set(observed)-set(EXPECTED_DATA_NAMES))}"
        )
    rows = []
    for name in EXPECTED_DATA_NAMES:
        path = args.data_dir / name
        source_entry = expected_source / name
        if not path.is_symlink() or path.resolve() != source_entry.resolve():
            raise RuntimeError(f"EX52 view entry is not the expected r0 symlink: {path}")
        rows.append(data_header(path))

    cache = args.data_dir / "ex52_data_certificate.json"
    prior = read_json(cache) if cache.is_file() else None
    stable = ("path", "resolved_path", "bytes", "mtime_ns", "device", "inode")
    reused = bool(
        prior
        and prior.get("passed") is True
        and len(prior.get("files", [])) == len(rows)
        and all(
            all(old.get(key) == new.get(key) for key in stable)
            and isinstance(old.get("sha256"), str)
            and len(old["sha256"]) == 64
            for old, new in zip(prior["files"], rows)
        )
    )
    if reused:
        rows = prior["files"]
    else:
        def hash_row(row: dict[str, Any]) -> dict[str, Any]:
            return {**row, "sha256": sha256_file(Path(row["path"]))}
        with ThreadPoolExecutor(max_workers=2) as executor:
            rows = list(executor.map(hash_row, rows))

    name_size_rows = sorted(rows, key=lambda row: (Path(row["path"]).name.startswith("fineweb_val_"), Path(row["path"]).name))
    names_and_sizes = "\n".join(
        f"{Path(row['path']).name}\t{row['bytes']}" for row in name_size_rows
    ).encode("utf-8")
    payload = {
        "schema_version": 2,
        "experiment_id": "52_llama_global_diag_scale",
        "passed": True,
        "official_repo": str(args.official_repo),
        "data_dir": str(args.data_dir),
        "train_shards": 50,
        "validation_shards": 1,
        "total_files": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "total_tokens_on_disk": sum(int(row["num_tokens"]) for row in rows),
        "inventory_name_size_sha256": hashlib.sha256(names_and_sizes).hexdigest(),
        "fingerprint_sha256": data_fingerprint(rows),
        "full_sha256_verified": True,
        "full_sha256_recomputed_this_audit": not reused,
        "reused_unchanged_certificate": reused,
        "files": rows,
        "audited_at": datetime.now(timezone.utc).isoformat(),
    }
    checks = {
        "files": payload["total_files"] == int(contract["data"]["total_files"]),
        "bytes": payload["total_bytes"] == int(contract["data"]["total_bytes"]),
        "name_size": payload["inventory_name_size_sha256"] == contract["data"]["accepted_name_size_sha256"],
        "full_content": payload["fingerprint_sha256"] == contract["data"]["accepted_full_content_fingerprint_sha256"],
    }
    payload["checks"] = checks
    payload["passed"] = all(checks.values())
    write_json(cache, payload)
    if not payload["passed"]:
        raise RuntimeError(f"EX52 full data audit failed: {checks}")
    return payload


def validate_preflight(args: argparse.Namespace) -> dict[str, Any]:
    path = args.run_dir / "preflight/preflight_manifest.json"
    payload = read_json(path)
    snap = source_snapshot(args)
    data_path = args.run_dir / "preflight/data_certificate.json"
    checks = {
        "passed": payload.get("passed") is True,
        "experiment": payload.get("experiment_id") == "52_llama_global_diag_scale",
        "official_repo": payload.get("official_repo") == str(args.official_repo),
        "data_dir": payload.get("data_dir") == str(args.data_dir),
        "contract": payload.get("contract_sha256") == sha256_file(contract_path(snap)),
        "snapshot": payload.get("source_snapshot_manifest_sha256") == sha256_file(snap / "source_snapshot_manifest.json"),
        "data_certificate": data_path.is_file() and payload.get("data_certificate_sha256") == sha256_file(data_path),
    }
    if data_path.is_file():
        data = read_json(data_path)
        expected_controller = controller_inventory_payload(str(args.data_dir), data)
        checks["data_passed"] = data.get("passed") is True
        checks["data_fingerprint"] = data.get("fingerprint_sha256") == read_json(contract_path(snap))["data"]["accepted_full_content_fingerprint_sha256"]
        checks["controller_inventory"] = payload.get("controller_inventory") == expected_controller
        checks["controller_inventory_fingerprint"] = payload.get("controller_inventory_fingerprint") == expected_controller["fingerprint"]
    if not all(checks.values()):
        raise RuntimeError(f"invalid EX52 preflight lineage: {checks}")
    return payload


def parent_preflight_artifact(root: Path) -> Path:
    paths = sorted(root.glob("*_preflight_seed2026.json"), key=lambda path: path.stat().st_mtime_ns)
    if not paths:
        raise RuntimeError(f"missing parent preflight artifact below {root}")
    # A retry after the other scale failed may leave more than one immutable
    # engineering receipt.  The just-completed receipt is the newest one and
    # is content-bound in the aggregate preflight manifest below.
    return paths[-1]


def preflight(args: argparse.Namespace) -> None:
    status = args.run_dir / "preflight/preflight_manifest.json"
    if status.is_file() and read_json(status).get("passed") is True:
        validate_preflight(args)
        print("skip passed EX52 preflight", flush=True)
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    snap = source_snapshot(args)
    contract = read_json(contract_path(snap))
    data = audit_data(args, contract)
    expected_controller = controller_inventory_payload(str(args.data_dir), data)
    data_path = args.run_dir / "preflight/data_certificate.json"
    write_json(data_path, data)
    results = {}
    for gpu_index, scale in enumerate(SCALES):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpus[gpu_index]
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["EX52_DATA_DIR"] = str(args.data_dir)
        output = args.run_dir / "preflight" / scale
        command = [str(sys.executable), "-B", str(runner(args, scale))]
        if scale == "124m":
            command.extend(["--official-repo", str(args.official_repo), "--python-exe", str(args.training_python), "--output-root", str(output), "--methods", "global_diag", "--seed", "2026", "--preflight", "--wandb-mode", "disabled"])
        else:
            command.extend(["--stage", "preflight", "--official-repo", str(args.official_repo), "--python-exe", str(args.training_python), "--output-root", str(output), "--methods", "global_diag", "--seed", "2026", "--wandb-mode", "disabled"])
        completed = subprocess.run(command, cwd=args.run_dir, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = args.run_dir / f"preflight/{scale}_preflight.log"
        log.write_text(completed.stdout, encoding="utf-8")
        row = {"return_code": completed.returncode, "passed": False, "log": str(log), "log_sha256": sha256_file(log)}
        if completed.returncode == 0:
            artifact = parent_preflight_artifact(output)
            parent = read_json(artifact)
            derived = read_json(snap / "source_snapshot_manifest.json")["derived_source_sha256"]
            expected_script = derived["scripts/17_llama_swiglu_validation/train_llama_swiglu.py"] if scale == "124m" else derived["scripts/20_llama_swiglu_1b/train_llama_swiglu_1b.py"]
            checks = {
                "status": parent.get("status") == "passed",
                "data_dir": Path(parent.get("data_audit", {}).get("data_dir", "")).resolve() == args.data_dir,
                "controller_data": parent.get("data_audit", {}) == expected_controller,
                "script": parent.get("script_sha256") == expected_script,
                "init": parent.get("init_audit", {}).get("common_init_sha256") == contract["accepted_init_sha256"][scale]["2026"],
                "method": set(parent.get("init_audit", {}).get("methods", {})) == {"global_diag"},
                "runtime_python": parent.get("runtime", {}).get("python_version") == [int(part) for part in contract["runtime"]["python"].split(".")],
                "runtime_torch": parent.get("runtime", {}).get("torch") == contract["runtime"]["torch"],
                "runtime_cuda": parent.get("runtime", {}).get("torch_cuda") == contract["runtime"]["torch_cuda"],
                "runtime_triton": parent.get("runtime", {}).get("triton") == contract["runtime"]["triton"],
                "runtime_numpy": parent.get("runtime", {}).get("numpy") == contract["runtime"]["numpy"],
                "runtime_gpu": contract["runtime"]["gpu_name_contains"] in parent.get("runtime", {}).get("gpu_name", ""),
                "runtime_capability": parent.get("runtime", {}).get("gpu_capability") == contract["runtime"]["compute_capability"],
                "triton_kernel": parent.get("runtime", {}).get("triton_kernels_sha256") == contract["official_r0_source_sha256"]["triton_kernels.py"],
            }
            row.update({"gpu": args.gpus[gpu_index], "passed": all(checks.values()), "checks": checks, "artifact": str(artifact), "artifact_sha256": sha256_file(artifact)})
        results[scale] = row
    checks = {
        "source_snapshot": True,
        "full_data_certificate": data.get("passed") is True,
        "data_fingerprint_exact": data["fingerprint_sha256"] == contract["data"]["accepted_full_content_fingerprint_sha256"],
        "parent_preflights": all(row["passed"] for row in results.values()),
    }
    payload = {
        "schema_version": 2,
        "experiment_id": "52_llama_global_diag_scale",
        "passed": all(checks.values()),
        "checks": checks,
        "results": results,
        "official_repo": str(args.official_repo),
        "data_dir": str(args.data_dir),
        "contract_sha256": sha256_file(contract_path(snap)),
        "data_fingerprint_sha256": data["fingerprint_sha256"],
        "controller_inventory": expected_controller,
        "controller_inventory_fingerprint": expected_controller["fingerprint"],
        "data_certificate": str(data_path),
        "data_certificate_sha256": sha256_file(data_path),
        "source_snapshot_manifest_sha256": sha256_file(snap / "source_snapshot_manifest.json"),
    }
    write_json(status, payload)
    if not payload["passed"]:
        raise RuntimeError(f"EX52 preflight failed: {checks}; results={results}")


def raw_manifest(root: Path, scale: str, controller_stage: str) -> Path | None:
    accepted = []
    for path in sorted(root.glob("**/llama_manifest.json")):
        payload = read_json(path)
        stage_ok = payload.get("execution_stage") == controller_stage if scale == "1b" else payload.get("batch_kind") == controller_stage
        if stage_ok and payload.get("status") == "completed" and payload.get("completed_methods") == ["global_diag"] and not payload.get("failed_methods"):
            accepted.append(path)
    if len(accepted) > 1:
        raise RuntimeError(f"multiple completed EX52 parent batches under {root}")
    return accepted[0] if accepted else None


def latest_plan(root: Path) -> Path | None:
    paths = sorted(root.glob("**/llama_plan.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[0].parent if paths else None


def unit_certificate_path(root: Path) -> Path | None:
    paths = sorted(root.glob("**/ex52_unit_manifest.json"))
    if len(paths) > 1:
        raise RuntimeError(f"multiple EX52 unit certificates below {root}")
    return paths[0] if paths else None


def dependency_certificate(args: argparse.Namespace, stage_name: str, scale: str, seed: int) -> Path | None:
    if stage_name == "pilot":
        return None
    if stage_name == "screen":
        return unit_certificate_path(args.run_dir / "pilot/1b" / f"seed{seed}")
    if scale == "124m":
        return unit_certificate_path(args.run_dir / "pilot/124m" / f"seed{seed}")
    return unit_certificate_path(args.run_dir / "screen/1b" / f"seed{seed}")


def expected_controller_config(
    contract: dict[str, Any], scale: str, stage_name: str
) -> dict[str, Any]:
    recipe = contract[scale]
    if stage_name not in ("pilot", "screen", "formal"):
        raise ValueError(f"unsupported EX52 unit stage: {stage_name}")
    steps = 34 if stage_name == "pilot" else 1000 if stage_name == "screen" else 6200
    short = stage_name == "pilot"
    return {
        "num_iterations": steps,
        "global_batch_size": int(recipe["global_batch_size"]),
        "device_batch_size": int(recipe["device_batch_size"]),
        "sequence_length": int(recipe["sequence_length"]),
        "val_every": steps if short else int(recipe["val_every"]),
        "val_tokens": (
            int(recipe["device_batch_size"]) * int(recipe["sequence_length"])
            if short
            else int(recipe["val_tokens"])
        ),
        "warmdown_iters": 1 if short else 0 if stage_name == "screen" else 1800,
        "backup_lr": float(recipe["backup_lr"]),
        "matrix_lr": float(recipe["matrix_lr"]),
        # The accepted .3 snapshot inherited this frozen parent default but
        # did not spell it out in the EX52 contract.  .4 makes it explicit.
        "adamw_matrix_lr": float(recipe.get("adamw_matrix_lr", 0.000576)),
        "checkpoint_every": 0 if short else int(contract["resume"]["checkpoint_every"]),
    }


def controller_implementation(args: argparse.Namespace, snap: Path) -> dict[str, Any]:
    snapshot_manifest = read_json(snap / "source_snapshot_manifest.json")
    snapshot_version = snapshot_manifest["contract_version"]
    snapshot_suite = snap / "scripts/52_llama_global_diag_scale/run_llama_global_diag_suite.py"
    if snapshot_version == CURRENT_CONTRACT_VERSION:
        return {
            "kind": "source_snapshot",
            "contract_version": snapshot_version,
            "path": str(snapshot_suite),
            "sha256": sha256_file(snapshot_suite),
            "scientific_contract_changed": False,
        }

    if snapshot_version not in AMENDABLE_SNAPSHOT_VERSIONS:
        raise RuntimeError(f"unsupported EX52 certificate snapshot: {snapshot_version}")
    root = args.run_dir / f"controller_amendments/{CURRENT_CONTRACT_VERSION}"
    target = root / "run_llama_global_diag_suite.py"
    manifest = root / "amendment_manifest.json"
    current = Path(__file__).resolve()
    if not target.is_file():
        root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current, target)
    if not manifest.is_file():
        root.mkdir(parents=True, exist_ok=True)
        write_json(
            manifest,
            {
                "schema_version": 1,
                "experiment_id": "52_llama_global_diag_scale",
                "passed": True,
                "amendment_version": CURRENT_CONTRACT_VERSION,
                "source_snapshot_contract_version": snapshot_version,
                "reason": "pilot parent controllers use short-run validation and checkpoint settings",
                "scope": "certificate_validation_only",
                "scientific_contract_changed": False,
                "training_artifacts_changed": False,
                "controller_path": str(target),
                "controller_sha256": sha256_file(target),
                "frozen_source_snapshot_manifest_sha256": sha256_file(
                    snap / "source_snapshot_manifest.json"
                ),
            },
        )
    payload = read_json(manifest)
    checks = {
        "passed": payload.get("passed") is True,
        "version": payload.get("amendment_version") == CURRENT_CONTRACT_VERSION,
        "source_version": payload.get("source_snapshot_contract_version") == snapshot_version,
        "scope": payload.get("scope") == "certificate_validation_only",
        "scientific_contract": payload.get("scientific_contract_changed") is False,
        "training_artifacts": payload.get("training_artifacts_changed") is False,
        "controller": target.is_file() and payload.get("controller_sha256") == sha256_file(target),
        "snapshot": payload.get("frozen_source_snapshot_manifest_sha256")
        == sha256_file(snap / "source_snapshot_manifest.json"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid EX52 controller amendment: {checks}")
    return {
        "kind": "certificate_only_amendment",
        "contract_version": snapshot_version,
        "path": str(target),
        "sha256": sha256_file(target),
        "amendment_manifest": str(manifest),
        "amendment_manifest_sha256": sha256_file(manifest),
        "scientific_contract_changed": False,
    }


def certify_unit(args: argparse.Namespace, stage_name: str, scale: str, seed: int, controller_stage: str, root: Path) -> Path:
    snap = source_snapshot(args)
    contract = read_json(contract_path(snap))
    pre = validate_preflight(args)
    manifest = raw_manifest(root, scale, controller_stage)
    if manifest is None:
        raise RuntimeError(f"missing completed parent manifest for {stage_name}/{scale}/seed{seed}")
    plan_path = manifest.with_name("llama_plan.json")
    summary_path = manifest.parent / "01_global_diag/summary.json"
    metrics_path = manifest.parent / "01_global_diag/metrics.csv"
    for path in (plan_path, summary_path, metrics_path):
        if not path.is_file():
            raise RuntimeError(f"missing EX52 unit artifact: {path}")
    parent, plan, summary = read_json(manifest), read_json(plan_path), read_json(summary_path)
    derived = read_json(snap / "source_snapshot_manifest.json")["derived_source_sha256"]
    expected_script = derived["scripts/17_llama_swiglu_validation/train_llama_swiglu.py"] if scale == "124m" else derived["scripts/20_llama_swiglu_1b/train_llama_swiglu_1b.py"]
    expected_base = derived["scripts/17_llama_swiglu_validation/train_llama_swiglu.py"]
    expected_steps = 34 if stage_name == "pilot" else 1000 if stage_name == "screen" else 6200
    recipe = contract[scale]
    config = parent.get("config", {})
    expected_config = expected_controller_config(contract, scale, stage_name)
    architecture = summary.get("architecture", {})
    expected_groups = 48 if scale == "124m" else 72
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        metrics_rows = list(csv.DictReader(handle))
    val_rows = [row for row in metrics_rows if row.get("event") == "val"]
    expected_val_points = 2 if expected_steps == 34 else expected_steps // 100 + 1
    checks = {
        "parent_status": parent.get("status") == "completed",
        "parent_method": parent.get("completed_methods") == ["global_diag"] and not parent.get("failed_methods"),
        "parent_seed": int(parent.get("seed", -1)) == seed,
        "plan_seed": int(plan.get("seed", -1)) == seed,
        "stage": (parent.get("execution_stage") == controller_stage if scale == "1b" else parent.get("batch_kind") == controller_stage),
        "script": parent.get("script_sha256") == plan.get("script_sha256") == expected_script,
        "base_trainer": scale == "124m" or (parent.get("base_trainer_sha256") == plan.get("base_trainer_sha256") == expected_base),
        "controller_data": parent.get("data_audit", {}) == plan.get("data_audit", {}) == pre["controller_inventory"],
        "data_dir": Path(parent.get("data_audit", {}).get("data_dir", "")).resolve() == args.data_dir,
        "init": parent.get("init_audit", {}).get("common_init_sha256") == plan.get("init_audit", {}).get("common_init_sha256") == summary.get("init_sha256") == contract["accepted_init_sha256"][scale][str(seed)],
        "summary_status": summary.get("status") == "completed",
        "summary_method": summary.get("method") == "global_diag",
        "summary_seed": int(summary.get("seed", -1)) == seed,
        "steps": int(summary.get("completed_steps", -1)) == expected_steps,
        "tokens": int(summary.get("tokens_seen", -1)) == expected_steps * int(recipe["global_batch_size"]) * int(recipe["sequence_length"]),
        "parameter_count": int(architecture.get("parameter_count", -1)) == int(recipe["parameter_count"]),
        "global_diag_route": architecture.get("global_diag_route") is True,
        "group_count": int(architecture.get("preconditioner_group_count", -1)) == expected_groups and len(architecture.get("preconditioner_groups", [])) == expected_groups,
        "all_groups_diag": {group.get("kind") for group in architecture.get("preconditioner_groups", [])} == {"diag"},
        "no_dense_scratch": int(summary.get("activation_scratch_bytes", -1)) == 8,
        "k_state": int(summary.get("k_state_bytes", -1)) == int(read_json(snap / "source_snapshot_manifest.json")["expected_k_state_bytes"][scale]),
        "config": config == plan.get("config", {}) == expected_config,
        "checkpoint_every": int(config.get("checkpoint_every", -1)) == int(expected_config["checkpoint_every"]),
        "val_grid": len(val_rows) == expected_val_points and int(val_rows[-1]["step"]) == expected_steps,
        "finite_losses": all(math.isfinite(float(summary[key])) for key in ("final_val_loss", "best_val_loss", "final_train_loss")),
    }
    dependency = dependency_certificate(args, stage_name, scale, seed)
    if stage_name != "pilot":
        checks["dependency"] = dependency is not None and read_json(dependency).get("passed") is True
    if not all(checks.values()):
        raise RuntimeError(f"EX52 unit certificate failed for {stage_name}/{scale}/seed{seed}: {checks}")
    implementation = controller_implementation(args, snap)
    certificate = {
        "schema_version": 2,
        "experiment_id": "52_llama_global_diag_scale",
        "passed": True,
        "stage": stage_name,
        "controller_stage": controller_stage,
        "scale": scale,
        "seed": seed,
        "method": "global_diag",
        "checks": checks,
        "expected_steps": expected_steps,
        "expected_train_tokens": expected_steps * int(recipe["global_batch_size"]) * int(recipe["sequence_length"]),
        "init_sha256": summary["init_sha256"],
        "data_fingerprint_sha256": pre["data_fingerprint_sha256"],
        "contract_sha256": sha256_file(contract_path(snap)),
        "source_snapshot_manifest_sha256": sha256_file(snap / "source_snapshot_manifest.json"),
        "certificate_controller": implementation,
        "training_source_sha256": expected_script,
        "base_trainer_sha256": expected_base,
        "dependency_certificate": str(dependency) if dependency else None,
        "dependency_certificate_sha256": sha256_file(dependency) if dependency else None,
        "artifacts": {
            "llama_manifest.json": sha256_file(manifest),
            "llama_plan.json": sha256_file(plan_path),
            "summary.json": sha256_file(summary_path),
            "metrics.csv": sha256_file(metrics_path),
        },
        "paths": {
            "llama_manifest": str(manifest),
            "llama_plan": str(plan_path),
            "summary": str(summary_path),
            "metrics": str(metrics_path),
        },
    }
    path = manifest.with_name("ex52_unit_manifest.json")
    write_json(path, certificate)
    return path


def validate_unit_certificate(args: argparse.Namespace, path: Path, stage_name: str, scale: str, seed: int, controller_stage: str) -> bool:
    if not path.is_file():
        return False
    payload = read_json(path)
    snap = source_snapshot(args)
    checks = {
        "passed": payload.get("passed") is True,
        "identity": payload.get("experiment_id") == "52_llama_global_diag_scale",
        "cell": payload.get("stage") == stage_name and payload.get("scale") == scale and int(payload.get("seed", -1)) == seed,
        "controller_stage": payload.get("controller_stage") == controller_stage,
        "contract": payload.get("contract_sha256") == sha256_file(contract_path(snap)),
        "snapshot": payload.get("source_snapshot_manifest_sha256") == sha256_file(snap / "source_snapshot_manifest.json"),
        "data": payload.get("data_fingerprint_sha256") == validate_preflight(args)["data_fingerprint_sha256"],
    }
    for name, expected in payload.get("artifacts", {}).items():
        key = {"llama_manifest.json": "llama_manifest", "llama_plan.json": "llama_plan", "summary.json": "summary", "metrics.csv": "metrics"}[name]
        artifact = Path(payload["paths"][key])
        checks[f"artifact:{name}"] = artifact.is_file() and sha256_file(artifact) == expected
    implementation = payload.get("certificate_controller", {})
    implementation_path = Path(implementation.get("path", ""))
    checks["certificate_controller"] = (
        implementation_path.is_file()
        and implementation.get("sha256") == sha256_file(implementation_path)
        and implementation.get("scientific_contract_changed") is False
    )
    amendment = implementation.get("amendment_manifest")
    checks["certificate_amendment"] = amendment is None or (
        Path(amendment).is_file()
        and implementation.get("amendment_manifest_sha256") == sha256_file(Path(amendment))
        and read_json(Path(amendment)).get("passed") is True
    )
    dependency = payload.get("dependency_certificate")
    checks["dependency"] = (
        dependency is None and stage_name == "pilot"
    ) or (
        dependency is not None
        and Path(dependency).is_file()
        and payload.get("dependency_certificate_sha256") == sha256_file(Path(dependency))
        and read_json(Path(dependency)).get("passed") is True
    )
    return all(checks.values())


def accepted_unit(args: argparse.Namespace, root: Path, stage_name: str, scale: str, seed: int, controller_stage: str) -> Path | None:
    path = unit_certificate_path(root)
    return path if path and validate_unit_certificate(args, path, stage_name, scale, seed, controller_stage) else None


def make_command(args: argparse.Namespace, scale: str, seed: int, stage: str, root: Path) -> list[str]:
    base = [str(sys.executable), "-B", str(runner(args, scale))]
    resume = latest_plan(root) if stage in ("medium", "formal") else None
    if scale == "124m":
        base.extend(["--official-repo", str(args.official_repo), "--python-exe", str(args.training_python), "--output-root", str(root), "--methods", "global_diag", "--seed", str(seed)])
        if stage == "smoke":
            base.extend(["--numerical-smoke", "--smoke-steps", "34", "--wandb-mode", "disabled"])
        elif resume:
            base.extend(["--resume-batch", str(resume)])
        else:
            smoke = raw_manifest(args.run_dir / "pilot/124m" / f"seed{seed}", "124m", "smoke")
            if smoke is None:
                raise RuntimeError(f"missing 124M smoke seed {seed}")
            base.extend(["--smoke-manifest", str(smoke), "--wandb-mode", args.wandb_mode, "--wandb-project", args.wandb_project])
    else:
        base.extend(["--stage", stage, "--official-repo", str(args.official_repo), "--python-exe", str(args.training_python), "--output-root", str(root), "--methods", "global_diag", "--seed", str(seed)])
        if resume:
            base.extend(["--resume-batch", str(resume)])
        elif stage == "smoke":
            base.extend(["--wandb-mode", "disabled"])
        else:
            smoke = raw_manifest(args.run_dir / "pilot/1b" / f"seed{seed}", "1b", "smoke")
            if smoke is None:
                raise RuntimeError(f"missing 1B smoke seed {seed}")
            base.extend(["--smoke-manifest", str(smoke), "--wandb-mode", args.wandb_mode, "--wandb-project", args.wandb_project])
            if stage == "formal":
                medium = raw_manifest(args.run_dir / "screen/1b" / f"seed{seed}", "1b", "medium")
                if medium is None:
                    raise RuntimeError(f"missing 1B medium seed {seed}")
                base.extend(["--medium-manifest", str(medium)])
        if stage == "medium":
            base.extend(["--medium-steps", "1000"])
    if args.wandb_entity and stage != "smoke":
        base.extend(["--wandb-entity", args.wandb_entity])
    return base


def require_stage(args: argparse.Namespace, stage: str, expected_units: int) -> dict[str, Any]:
    path = args.run_dir / stage / f"{stage}_manifest.json"
    payload = read_json(path)
    snap = source_snapshot(args)
    checks = {
        "passed": payload.get("passed") is True,
        "experiment": payload.get("experiment_id") == "52_llama_global_diag_scale",
        "stage": payload.get("stage") == stage,
        "unit_count": int(payload.get("unit_count", -1)) == expected_units == len(payload.get("units", [])),
        "contract": payload.get("contract_sha256") == sha256_file(contract_path(snap)),
        "snapshot": payload.get("source_snapshot_manifest_sha256") == sha256_file(snap / "source_snapshot_manifest.json"),
        "data": payload.get("data_fingerprint_sha256") == validate_preflight(args)["data_fingerprint_sha256"],
    }
    for row in payload.get("units", []):
        cert = Path(row["certificate"])
        checks[f"unit:{row.get('scale')}:{row.get('seed')}"] = cert.is_file() and sha256_file(cert) == row.get("certificate_sha256") and read_json(cert).get("passed") is True
    if not all(checks.values()):
        raise RuntimeError(f"invalid EX52 {stage} gate: {checks}")
    return payload


def run_jobs(args: argparse.Namespace, stage_name: str, jobs: list[tuple[str, int, str]]) -> None:
    pending = []
    for scale, seed, controller_stage in jobs:
        root = args.run_dir / stage_name / scale / f"seed{seed}"
        if accepted_unit(args, root, stage_name, scale, seed, controller_stage) is not None:
            print(f"skip accepted {stage_name}/{scale}/seed{seed}", flush=True)
            continue
        if raw_manifest(root, scale, controller_stage) is not None:
            cert = certify_unit(args, stage_name, scale, seed, controller_stage, root)
            if not validate_unit_certificate(args, cert, stage_name, scale, seed, controller_stage):
                raise RuntimeError(f"invalid recovered EX52 unit certificate: {cert}")
            print(f"certified completed {stage_name}/{scale}/seed{seed}", flush=True)
            continue
        pending.append((scale, seed, controller_stage, root))
    active: dict[str, tuple[subprocess.Popen[str], Any, str, int, str, Path, Path]] = {}
    logs = args.run_dir / "controller_logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        while pending or active:
            for gpu in args.gpus:
                if gpu in active or not pending:
                    continue
                scale, seed, controller_stage, root = pending.pop(0)
                root.mkdir(parents=True, exist_ok=True)
                log_path = logs / f"{stage_name}_{scale}_seed{seed}.log"
                handle = log_path.open("a", encoding="utf-8", buffering=1)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                env["EX52_DATA_DIR"] = str(args.data_dir)
                proc = subprocess.Popen(make_command(args, scale, seed, controller_stage, root), cwd=args.run_dir, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
                active[gpu] = (proc, handle, scale, seed, controller_stage, log_path, root)
                print(f"started {stage_name}/{scale}/seed{seed} on gpu {gpu}", flush=True)
            time.sleep(1)
            for gpu, value in list(active.items()):
                proc, handle, scale, seed, controller_stage, log_path, root = value
                rc = proc.poll()
                if rc is None:
                    continue
                handle.close()
                del active[gpu]
                if rc != 0:
                    raise RuntimeError(f"failed {stage_name}/{scale}/seed{seed} rc={rc}; see {log_path}")
                cert = certify_unit(args, stage_name, scale, seed, controller_stage, root)
                if not validate_unit_certificate(args, cert, stage_name, scale, seed, controller_stage):
                    raise RuntimeError(f"invalid EX52 unit certificate: {cert}")
                print(f"passed {stage_name}/{scale}/seed{seed}", flush=True)
    except BaseException:
        for proc, _, _, _, _, _, _ in active.values():
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 20
        for proc, handle, _, _, _, _, _ in active.values():
            try:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            handle.close()
        raise

    units = []
    for scale, seed, controller_stage in jobs:
        root = args.run_dir / stage_name / scale / f"seed{seed}"
        cert = accepted_unit(args, root, stage_name, scale, seed, controller_stage)
        if cert is None:
            raise RuntimeError(f"EX52 {stage_name} is incomplete: {scale}/seed{seed}")
        units.append({"scale": scale, "seed": seed, "controller_stage": controller_stage, "certificate": str(cert), "certificate_sha256": sha256_file(cert)})
    snap = source_snapshot(args)
    pre = validate_preflight(args)
    write_json(
        args.run_dir / stage_name / f"{stage_name}_manifest.json",
        {
            "schema_version": 2,
            "experiment_id": "52_llama_global_diag_scale",
            "stage": stage_name,
            "passed": True,
            "unit_count": len(units),
            "contract_sha256": sha256_file(contract_path(snap)),
            "source_snapshot_manifest_sha256": sha256_file(snap / "source_snapshot_manifest.json"),
            "data_fingerprint_sha256": pre["data_fingerprint_sha256"],
            "units": units,
        },
    )


def verify(args: argparse.Namespace) -> None:
    validate_preflight(args)
    require_stage(args, "pilot", 6)
    require_stage(args, "screen", 3)
    require_stage(args, "formal", 6)
    snap = source_snapshot(args)
    analyzer = snap / "scripts/52_llama_global_diag_scale/analyze_llama_global_diag_scale.py"
    subprocess.run(
        [str(sys.executable), "-B", str(analyzer), "--run-dir", str(args.run_dir), "--contract", str(contract_path(snap)), "--controls", str(snap / "scripts/52_llama_global_diag_scale/frozen_llama_controls.csv")],
        check=True,
    )
    analysis = read_json(args.run_dir / "analysis/analysis_manifest.json")
    checks = {
        "preflight": True,
        "pilot": True,
        "screen": True,
        "formal": True,
        "analysis": analysis.get("passed") is True,
        "analysis_contract": analysis.get("contract_sha256") == sha256_file(contract_path(snap)),
        "analysis_snapshot": analysis.get("source_snapshot_manifest_sha256") == sha256_file(snap / "source_snapshot_manifest.json"),
    }
    write_json(
        args.run_dir / "handoff_manifest.json",
        {
            "schema_version": 2,
            "experiment_id": "52_llama_global_diag_scale",
            "passed": all(checks.values()),
            "status": "completed" if all(checks.values()) else "failed",
            "checks": checks,
            "contract_sha256": sha256_file(contract_path(snap)),
            "source_snapshot_manifest_sha256": sha256_file(snap / "source_snapshot_manifest.json"),
            "analysis_manifest_sha256": sha256_file(args.run_dir / "analysis/analysis_manifest.json"),
        },
    )
    if not all(checks.values()):
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    for field in ("run_dir", "repo", "official_repo", "data_dir"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    args.training_python = args.training_python.expanduser().absolute()
    # Freeze a certificate-only amendment when resuming the .3 run whose
    # training source is already immutable.  New .4 runs use their source
    # snapshot directly.
    controller_implementation(args, source_snapshot(args))
    if args.stage in ("preflight", "all"):
        preflight(args)
    if args.stage in ("pilot", "all"):
        validate_preflight(args)
        run_jobs(args, "pilot", [(scale, seed, "smoke") for scale in ("1b", "124m") for seed in SEEDS])
    if args.stage in ("screen", "all"):
        require_stage(args, "pilot", 6)
        run_jobs(args, "screen", [("1b", seed, "medium") for seed in SEEDS])
    if args.stage in ("formal", "all"):
        require_stage(args, "pilot", 6)
        require_stage(args, "screen", 3)
        run_jobs(args, "formal", [(scale, seed, "formal") for scale in ("1b", "124m") for seed in SEEDS])
    if args.stage in ("verify", "all"):
        verify(args)
    print(f"EX52 stage {args.stage} completed. Artifacts: {args.run_dir}")


if __name__ == "__main__":
    main()
