#!/usr/bin/env python3
"""Shared, dependency-light helpers for experiment 42."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def lexical_absolute(path: Path) -> Path:
    """Make a path absolute without resolving a virtualenv interpreter symlink."""

    return Path(os.path.abspath(os.path.expanduser(str(path))))


def stable_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    return {field: runtime.get(field) for field in STABLE_RUNTIME_FIELDS}


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def subprocess_environment(
    official_repo: Path,
    *,
    derived_base: Path | None = None,
    derived_base_sha256: str | None = None,
    physical_gpu: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(official_repo) + (
        os.pathsep + current if current else ""
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = "2026"
    env["WANDB_MODE"] = "disabled"
    if physical_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = physical_gpu
    if derived_base is not None:
        if not derived_base_sha256:
            raise ValueError("derived trainer SHA is required with derived trainer path")
        env["LLAMA_1B_BASE_TRAINER"] = str(derived_base)
        env["LLAMA_1B_BASE_TRAINER_SHA256"] = derived_base_sha256
    return env


def git_output(repo: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = [
        "git",
        "-c",
        f"safe.directory={repo}",
        "-C",
        str(repo),
        *arguments,
    ]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and result.returncode:
        output = result.stdout.strip() or "<no git diagnostic output>"
        raise RuntimeError(
            "Git provenance command failed "
            f"(exit={result.returncode}, repo={repo}, args={list(arguments)}):\n"
            f"{output}"
        )
    return result


def audit_official_repo(repo: Path, contract: dict[str, Any]) -> dict[str, Any]:
    repo = repo.resolve()
    source_contract = contract["source_contract"]
    commit = git_output(repo, "rev-parse", "HEAD").stdout.strip()
    raw_status = git_output(
        repo, "status", "--porcelain", "--untracked-files=no"
    ).stdout.splitlines()
    worktree_semantic = git_output(
        repo, "diff", "--ignore-space-at-eol", "--quiet", check=False
    )
    index_semantic = git_output(
        repo,
        "diff",
        "--cached",
        "--ignore-space-at-eol",
        "--quiet",
        check=False,
    )
    kernel = repo / "triton_kernels.py"
    if not kernel.is_file():
        raise FileNotFoundError(kernel)
    kernel_sha = sha256_file(kernel)
    checks = {
        "commit": commit == source_contract["official_repo_commit"],
        "worktree_changes_are_eol_only": worktree_semantic.returncode == 0,
        "index_changes_are_eol_only": index_semantic.returncode == 0,
        "triton_kernels_sha256": (
            kernel_sha == source_contract["triton_kernels_sha256"]
        ),
    }
    payload = {
        "repo": str(repo),
        "commit": commit,
        "raw_tracked_status": raw_status,
        "worktree_semantic_diff_return_code": worktree_semantic.returncode,
        "index_semantic_diff_return_code": index_semantic.returncode,
        "triton_kernels": str(kernel),
        "triton_kernels_sha256": kernel_sha,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not payload["passed"]:
        raise RuntimeError(f"official repository audit failed: {payload}")
    return payload


def _shard_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(12)
    if len(raw) != 12:
        raise RuntimeError(f"short shard header: {path}")
    magic, version, tokens = struct.unpack("<iii", raw)
    if magic != 20240520 or version != 1 or tokens <= 0:
        raise RuntimeError(
            f"invalid shard header: {path} magic={magic} version={version} tokens={tokens}"
        )
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "tokens": tokens,
        "sha256": sha256_file(path),
    }


def audit_data(data_dir: Path) -> dict[str, Any]:
    """Hash the full frozen 50-shard cache once before the timed cells."""

    data_dir = data_dir.resolve()
    train = sorted(data_dir.glob("fineweb_train_*.bin"))
    validation = sorted(data_dir.glob("fineweb_val_*.bin"))
    if len(train) != 50 or len(validation) != 1:
        raise RuntimeError(
            "FineWeb audit requires exactly 50 train shards and one validation "
            f"shard; observed train={len(train)} validation={len(validation)}"
        )
    files = [_shard_header(path) for path in [*train, *validation]]
    payload = {
        "data_dir": str(data_dir),
        "train_shard_count": len(train),
        "validation_shard_count": len(validation),
        "files": files,
        "total_tokens_in_train_headers": sum(row["tokens"] for row in files[:-1]),
        "total_bytes": sum(row["bytes"] for row in files),
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def relative_to_run(path: Path, run_dir: Path) -> str:
    return str(path.resolve().relative_to(run_dir.resolve())).replace("\\", "/")


def resolve_run_path(value: str, run_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_dir / path
