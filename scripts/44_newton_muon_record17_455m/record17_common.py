#!/usr/bin/env python3
"""Shared, dependency-light utilities for experiment 44.

This module intentionally avoids importing torch or wandb.  It is used by the
controller Python environment, while training is launched with the pinned
Torch/CUDA environment.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import re
import socket
import struct
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


SCRIPT_VERSION = "2026-07-30.1"
METHODS = (
    "muon",
    "original_newton_muon",
    "selective_none",
    "selective_diag",
)
# The three paired seeds are frozen before any experiment-44 outcome is read.
SEEDS = (2024, 2025, 2026)
EXPECTED_COMMIT = "df78af0db523d8bceb25af4919a3e3e7082b80f3"
EXPECTED_CANONICAL_SHA256 = {
    "README.md": "c0de75168a71988240408cace7a973ec666f6006d07466db9f0d497c7f3fdfa6",
    "data/cached_fineweb10B.py": "adcc9f7d81ed1ac115a66d08d94d8d3e5c7425cabaf856da1f1fb106af87d09b",
    "train_gpt_newton_muon_2.py": "d30d31e3a01a18ea19050ea8aba04609d4825af2d26594955c148af83b07c4b6",
    "triton_kernels.py": "b51ac50c699b05306619d92cb9ec6edadd266d8118c53f5b9726db76480ea16d",
}

# Modded-NanoGPT Medium Track Record #17.  The full trainer is vendored in
# this experiment because the Newton-Muon release does not contain its 455M
# reproduction source.  These identifiers are the immutable upstream record,
# not a moving master-branch reference.
RECORD17_UPSTREAM_REPOSITORY = "https://github.com/KellerJordan/modded-nanogpt"
RECORD17_UPSTREAM_COMMIT = "9e7218468ea864a33053142c196d90bbf3ed48e1"
RECORD17_UPSTREAM_PATH = (
    "records/track_2_medium/2025-11-12_BlockMaskRedundantOp/"
    "train_gpt_medium.py"
)
RECORD17_UPSTREAM_GIT_BLOB = "8504813a5ba0b1bf981fd6ad9d6348bfa1754b0f"
RECORD17_UPSTREAM_CANONICAL_SHA256 = (
    "03d91174eed5e8cbf57063a1e997eb98570dde7a09ba9b2c94aa36e9d5eb94cb"
)

VALIDATION_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+"
    r"val_loss:(?P<loss>[0-9]+(?:\.[0-9]+)?)\s+"
    r"train_time:(?P<time_ms>\d+)ms\s+"
    r"step_avg:(?P<step_ms>[0-9]+(?:\.[0-9]+)?)ms"
)
TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+"
    r"train_time:(?P<time_ms>\d+)ms\s+"
    r"step_avg:(?P<step_ms>[0-9]+(?:\.[0-9]+)?)ms"
)
METADATA_PREFIX = "RECORD17_METADATA "
MEMORY_PREFIX = "RECORD17_FINAL_AUDIT "
VALIDATION_PREFIX = "RECORD17_VAL "
WARMUP_RESET_PREFIX = "RECORD17_WARMUP_RESET "


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append and fsync a single command/status record.

    JSONL is intentionally append-only; all authoritative manifests use atomic
    replacement through :func:`atomic_write_json`.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_command(repo: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo}",
            "-C",
            str(repo),
            *arguments,
        ],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def git_blob(repo: Path, relative_path: str, revision: str = "HEAD") -> bytes:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo}",
            "-C",
            str(repo),
            "show",
            f"{revision}:{relative_path}",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return canonical_bytes(completed.stdout)


def audit_official_repo(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    if not repo.is_dir():
        raise RuntimeError(f"official repository does not exist: {repo}")
    commit = git_command(repo, "rev-parse", "HEAD").stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(
            f"official commit mismatch: expected {EXPECTED_COMMIT}, observed {commit}"
        )
    status = git_command(
        repo, "status", "--porcelain", "--untracked-files=no"
    ).stdout.splitlines()
    ignore_eol = git_command(
        repo, "diff", "--ignore-space-at-eol", "--quiet", check=False
    )
    cached_ignore_eol = git_command(
        repo,
        "diff",
        "--cached",
        "--ignore-space-at-eol",
        "--quiet",
        check=False,
    )
    if ignore_eol.returncode not in (0, 1) or cached_ignore_eol.returncode not in (
        0,
        1,
    ):
        raise RuntimeError(ignore_eol.stderr.strip() or "git diff failed")
    if ignore_eol.returncode != 0 or cached_ignore_eol.returncode != 0:
        raise RuntimeError(
            "official repository contains tracked changes beyond line endings:\n"
            + "\n".join(status)
        )
    blobs: dict[str, Any] = {}
    for relative_path, expected in EXPECTED_CANONICAL_SHA256.items():
        raw = git_blob(repo, relative_path, commit)
        observed = sha256_bytes(raw)
        if observed != expected:
            raise RuntimeError(
                f"upstream blob hash mismatch for {relative_path}: "
                f"expected {expected}, observed {observed}"
            )
        blobs[relative_path] = {
            "canonical_sha256": observed,
            "bytes": len(raw),
        }
    return {
        "schema_version": 1,
        "passed": True,
        "repo": str(repo),
        "commit": commit,
        "remote": git_command(repo, "remote", "get-url", "origin", check=False).stdout.strip(),
        "tracked_status": status,
        "tracked_changes_are_eol_only": bool(status),
        "source_from_git_blobs": True,
        "blobs": blobs,
        "audited_at": utc_now(),
    }


def audit_vendored_record17(path: Path) -> dict[str, Any]:
    """Verify the frozen Medium Track Record #17 trainer byte-for-byte.

    Git may check the file out with CRLF on Windows, so provenance is defined
    over canonical LF bytes, matching the upstream Git blob content.
    """

    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"vendored Record #17 trainer does not exist: {path}")
    raw = canonical_bytes(path.read_bytes())
    observed = sha256_bytes(raw)
    if observed != RECORD17_UPSTREAM_CANONICAL_SHA256:
        raise RuntimeError(
            "vendored Record #17 trainer hash mismatch: "
            f"expected {RECORD17_UPSTREAM_CANONICAL_SHA256}, observed {observed}"
        )
    return {
        "schema_version": 1,
        "passed": True,
        "path": str(path),
        "repository": RECORD17_UPSTREAM_REPOSITORY,
        "commit": RECORD17_UPSTREAM_COMMIT,
        "repository_path": RECORD17_UPSTREAM_PATH,
        "git_blob": RECORD17_UPSTREAM_GIT_BLOB,
        "canonical_sha256": observed,
        "canonical_bytes": len(raw),
        "audited_at": utc_now(),
    }


def expected_data_names() -> list[str]:
    return [
        "fineweb_val_000000.bin",
        *[f"fineweb_train_{index:06d}.bin" for index in range(1, 51)],
    ]


def _file_header(path: Path) -> dict[str, Any]:
    stat = path.stat()
    with path.open("rb") as handle:
        header = handle.read(1024)
    if len(header) != 1024:
        raise RuntimeError(f"truncated 1024-byte header: {path}")
    magic, version, num_tokens = struct.unpack_from("<iii", header, 0)
    expected_bytes = 1024 + 2 * num_tokens
    checks = {
        "magic": magic == 20240520,
        "version": version == 1,
        "num_tokens": num_tokens == 100_000_000,
        "bytes": stat.st_size == expected_bytes,
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid FineWeb shard {path}: {checks}")
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "magic": magic,
        "version": version,
        "num_tokens": num_tokens,
    }


def _hash_data_row(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "sha256": sha256_file(Path(row["path"]))}


def audit_fineweb_data(
    data_repo_root: Path,
    *,
    previous_certificate: Path | None = None,
    hash_workers: int = 2,
) -> dict[str, Any]:
    """Audit the exact 50-train/1-validation upstream data contract.

    A previous certificate is reused only when its own content is intact and
    every file identity, size, and mtime still matches.  Otherwise all 10.2 GB
    are hashed again.
    """

    data_repo_root = data_repo_root.expanduser().resolve()
    shard_dir = data_repo_root / "data" / "fineweb10B"
    if not shard_dir.exists():
        raise RuntimeError(f"FineWeb shard directory is missing: {shard_dir}")
    expected = expected_data_names()
    observed = sorted(path.name for path in shard_dir.glob("fineweb_*.bin"))
    if observed != sorted(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise RuntimeError(
            f"FineWeb shard inventory mismatch: missing={missing}, extra={extra}"
        )
    rows = [_file_header(shard_dir / name) for name in expected]

    reused = False
    if previous_certificate and previous_certificate.is_file():
        prior = read_json(previous_certificate)
        prior_rows = prior.get("files", [])
        stable_fields = ("path", "resolved_path", "bytes", "mtime_ns", "device", "inode")
        if (
            prior.get("passed") is True
            and len(prior_rows) == len(rows)
            and all(
                all(old.get(key) == new.get(key) for key in stable_fields)
                and isinstance(old.get("sha256"), str)
                and len(old["sha256"]) == 64
                for old, new in zip(prior_rows, rows)
            )
        ):
            rows = prior_rows
            reused = True
    if not reused:
        with ThreadPoolExecutor(max_workers=max(1, hash_workers)) as executor:
            rows = list(executor.map(_hash_data_row, rows))

    digest = hashlib.sha256()
    for row in rows:
        digest.update(Path(row["path"]).name.encode("utf-8"))
        digest.update(str(row["bytes"]).encode("ascii"))
        digest.update(str(row["num_tokens"]).encode("ascii"))
        digest.update(row["sha256"].encode("ascii"))
    return {
        "schema_version": 1,
        "passed": True,
        "data_repo_root": str(data_repo_root),
        "shard_dir": str(shard_dir),
        "train_shards": 50,
        "validation_shards": 1,
        "total_files": 51,
        "total_tokens_on_disk": sum(row["num_tokens"] for row in rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "fingerprint_sha256": digest.hexdigest(),
        "full_sha256_verified": True,
        "full_sha256_recomputed_this_audit": not reused,
        "reused_unchanged_certificate": reused,
        "files": rows,
        "audited_at": utc_now(),
    }


def sanitize_lock_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


@contextlib.contextmanager
def exclusive_file_lock(
    path: Path,
    *,
    timeout_seconds: float = 0,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Path]:
    """Acquire an advisory cross-process lock on Linux (and a local fallback on Windows)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    started = time.monotonic()
    try:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    if handle.tell() == 0:
                        handle.write(b"\0")
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if timeout_seconds <= 0 or time.monotonic() - started >= timeout_seconds:
                    raise RuntimeError(f"resource lock is already held: {path}")
                time.sleep(1)
        if metadata:
            handle.seek(0)
            handle.truncate()
            handle.write(
                (json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())
        yield path
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def shared_gpu_lock_path(result_root: Path, physical_gpu: str) -> Path:
    host = sanitize_lock_component(socket.gethostname())
    gpu = sanitize_lock_component(physical_gpu)
    return result_root / ".physical_gpu_locks" / host / f"gpu_{gpu}.lock"


def query_gpus() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line!r}")
        rows.append(
            {
                "index": parts[0],
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mib": int(parts[3]),
                "memory_used_mib": int(parts[4]),
                "utilization_percent": int(parts[5]),
            }
        )
    return rows


def query_compute_processes() -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        return []
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip() or "No running processes" in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": parts[1],
                    "process_name": parts[2],
                    "used_memory_mib": parts[3],
                }
            )
    return rows


def resolve_gpu(gpus: Sequence[dict[str, Any]], identifier: str) -> dict[str, Any]:
    matches = [
        row
        for row in gpus
        if row["index"] == identifier or row["uuid"] == identifier
    ]
    if len(matches) != 1:
        raise RuntimeError(f"GPU identifier {identifier!r} did not resolve uniquely")
    return matches[0]


def parse_training_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"training log is missing: {path}")
    validations: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    final_audit: dict[str, Any] | None = None
    warmup_reset: dict[str, Any] | None = None
    structured_validations: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(METADATA_PREFIX):
            metadata = json.loads(line[len(METADATA_PREFIX) :])
            continue
        if line.startswith(MEMORY_PREFIX):
            final_audit = json.loads(line[len(MEMORY_PREFIX) :])
            continue
        if line.startswith(WARMUP_RESET_PREFIX):
            warmup_reset = json.loads(line[len(WARMUP_RESET_PREFIX) :])
            continue
        if line.startswith(VALIDATION_PREFIX):
            structured_validations.append(
                json.loads(line[len(VALIDATION_PREFIX) :])
            )
            continue
        match = VALIDATION_RE.fullmatch(line.strip())
        if match:
            validations.append(
                {
                    "step": int(match["step"]),
                    "total_steps": int(match["total"]),
                    "val_loss": float(match["loss"]),
                    "train_time_ms": int(match["time_ms"]),
                    "step_avg_ms": float(match["step_ms"]),
                }
            )
            continue
        match = TRAIN_RE.fullmatch(line.strip())
        if match:
            train_rows.append(
                {
                    "step": int(match["step"]),
                    "total_steps": int(match["total"]),
                    "train_time_ms": int(match["time_ms"]),
                    "step_avg_ms": float(match["step_ms"]),
                }
            )
    if structured_validations:
        validations = structured_validations
    return {
        "validations": validations,
        "train_rows": train_rows,
        "metadata": metadata,
        "final_audit": final_audit,
        "warmup_reset": warmup_reset,
    }


def artifact_hashes(directory: Path, names: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"required artifact is missing: {path}")
        result[name] = sha256_file(path)
    return result


def cell_key(stage: str, seed: int, method: str) -> str:
    if method not in METHODS:
        raise ValueError(method)
    return f"experiment44:{stage}:seed{seed}:{method}"


def stable_wandb_id(stage: str, seed: int, method: str, contract_sha256: str) -> str:
    digest = hashlib.sha256(
        f"{cell_key(stage, seed, method)}:{contract_sha256}".encode("utf-8")
    ).hexdigest()[:20]
    return f"rec17-{seed}-{method[:12]}-{digest}"
