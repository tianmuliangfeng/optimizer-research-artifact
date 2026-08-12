#!/usr/bin/env python3
"""MECH-00: read-only artifact, checkpoint, runtime, and source inventory.

This script deliberately does not import torch and never loads a checkpoint.
It may run on a training host without reserving GPU memory.  The only writes
are the small CSV/JSON files below ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "2026-07-24.2"
SUMMARY_NAMES = ("summary.json", "r1_summary.json")
RUN_MANIFEST_NAME = "run_manifest.json"
BATCH_MANIFEST_NAMES = ("llama_manifest.json", "r1_manifest.json")
CHECKPOINT_GLOBS = ("checkpoint_latest.pt", "state_step*.pt", "checkpoint*.pt", "*.pt")
DEFAULT_TARGET_STEPS = (0, 500, 1000, 3000, 6200)
FORMAL_KINDS = {"formal", "formal_evidence", "medium"}


@dataclass(frozen=True)
class InputSpec:
    label: str
    path: Path
    family_hint: str


@dataclass(frozen=True)
class RepoSpec:
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory R1/LLaMA/bridge artifacts without importing torch or "
            "allocating GPU memory."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Artifact root to scan. Repeat for multiple experiment families.",
    )
    parser.add_argument(
        "--family-hint",
        action="append",
        default=[],
        metavar="LABEL=FAMILY",
        help=(
            "Optional family override for one --input label, e.g. "
            "r1=r1_native or bridge=gpt_bridge."
        ),
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Source repository to audit with read-only git commands.",
    )
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--execution-domain", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[],
        help="Optional method allow-list. Empty means all discovered methods.",
    )
    parser.add_argument(
        "--target-steps",
        nargs="+",
        type=int,
        default=list(DEFAULT_TARGET_STEPS),
    )
    parser.add_argument(
        "--hash-mode",
        choices=("none", "full"),
        default="none",
        help=(
            "Checkpoint hashing policy. Use none during active training to avoid "
            "large disk reads; use full after training has stopped."
        ),
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include non-completed summary rows in the checkpoint step map.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if no usable checkpoint is found or a hash is unstable.",
    )
    return parser.parse_args()


def parse_assignments(values: list[str], option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"{option} expects LABEL=VALUE, observed: {raw!r}")
        label, value = raw.split("=", 1)
        label = label.strip()
        value = value.strip()
        if not label or not value:
            raise ValueError(f"{option} expects non-empty LABEL=VALUE: {raw!r}")
        if label in parsed:
            raise ValueError(f"duplicate {option} label: {label}")
        parsed[label] = value
    return parsed


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def json_load(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def canonical_json_sha256(payload: Any) -> str:
    data = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file_stable(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    """Hash a file and reject a digest if the pathname changed during the read."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    after = path.stat()
    stable = (
        before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and getattr(before, "st_ino", None) == getattr(after, "st_ino", None)
    )
    return {
        "sha256": digest.hexdigest() if stable else "",
        "hash_status": "verified_stable" if stable else "changed_during_hash",
        "size_before": before.st_size,
        "size_after": after.st_size,
        "mtime_ns_before": before.st_mtime_ns,
        "mtime_ns_after": after.st_mtime_ns,
    }


def run_probe(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": command,
            "status": "unavailable",
            "stdout": "",
            "stderr": f"{command[0]} not found",
        }
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "command": command,
            "status": "failed",
            "stdout": "",
            "stderr": repr(exc),
        }
    return {
        "command": command,
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def collect_host_runtime(host_id: str, execution_domain: str) -> dict[str, Any]:
    smi_query = run_probe(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    smi_header = run_probe(["nvidia-smi"])
    return {
        "schema_version": 1,
        "collected_at": now_iso(),
        "host_id": host_id,
        "execution_domain": execution_domain,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi_query": smi_query,
        "nvidia_smi_header": smi_header,
        "note": (
            "MECH-00 does not import torch. Training-runtime versions are taken "
            "from experiment summaries/manifests when available."
        ),
    }


def audit_repo(spec: RepoSpec) -> dict[str, Any]:
    row: dict[str, Any] = {
        "repo_label": spec.label,
        "repo_path": str(spec.path.resolve()),
        "exists": spec.path.is_dir(),
    }
    if not spec.path.is_dir():
        row.update(
            {
                "git_commit": "",
                "git_status_porcelain": "",
                "git_remote": "",
                "audit_status": "missing",
            }
        )
        return row
    git_env = dict(os.environ)
    # Prevent optional index refreshes/locks so this remains a read-only audit.
    git_env["GIT_OPTIONAL_LOCKS"] = "0"
    commit = run_probe(["git", "rev-parse", "HEAD"], cwd=spec.path, env=git_env)
    status = run_probe(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=spec.path,
        env=git_env,
    )
    remote = run_probe(
        ["git", "remote", "get-url", "origin"], cwd=spec.path, env=git_env
    )
    row.update(
        {
            "git_commit": commit.get("stdout", ""),
            "git_status_porcelain": status.get("stdout", ""),
            "git_remote": remote.get("stdout", ""),
            "commit_probe_status": commit["status"],
            "status_probe_status": status["status"],
            "remote_probe_status": remote["status"],
            "commit_probe_error": commit.get("stderr", ""),
            "status_probe_error": status.get("stderr", ""),
            "audit_status": (
                "ok_dirty"
                if commit["status"] == "ok"
                and status["status"] == "ok"
                and bool(status.get("stdout", ""))
                else "ok_clean"
                if commit["status"] == "ok" and status["status"] == "ok"
                else "git_probe_failed"
            ),
        }
    )
    return row


def find_nearest_manifest(summary_path: Path, input_root: Path) -> tuple[Path | None, dict[str, Any]]:
    current = summary_path.parent
    root = input_root.resolve()
    while True:
        for name in BATCH_MANIFEST_NAMES:
            candidate = current / name
            payload = json_load(candidate)
            if payload is not None:
                return candidate, payload
        if current == root or current.parent == current:
            break
        try:
            current.relative_to(root)
        except ValueError:
            break
        current = current.parent
    return None, {}


def candidate_checkpoint_files(directory: Path) -> list[Path]:
    observed: dict[str, Path] = {}
    for pattern in CHECKPOINT_GLOBS:
        for path in directory.rglob(pattern):
            if path.is_file():
                observed[str(path.resolve())] = path.resolve()
        if observed and pattern != "*.pt":
            # Prefer known checkpoint names over arbitrary torch files.
            break
    return sorted(observed.values(), key=lambda item: str(item))


def resolve_checkpoint(reference: Any, summary_path: Path) -> tuple[Path | None, str]:
    ref = str(reference or "").strip()
    if ref:
        candidate = Path(ref).expanduser()
        if candidate.is_file():
            return candidate.resolve(), "manifest_reference"
        if not candidate.is_absolute():
            relative = (summary_path.parent / candidate).resolve()
            if relative.is_file():
                return relative, "summary_relative_reference"
        basename = candidate.name
        if basename:
            matches = [
                path.resolve()
                for path in summary_path.parent.rglob(basename)
                if path.is_file()
            ]
            if len(matches) == 1:
                return matches[0], "recovered_by_basename"
            if len(matches) > 1:
                return None, "ambiguous_basename"
    matches = candidate_checkpoint_files(summary_path.parent)
    if len(matches) == 1:
        return matches[0], "single_checkpoint_fallback"
    if len(matches) > 1:
        return None, "ambiguous_checkpoint_fallback"
    return None, "missing"


def number(payload: dict[str, Any], keys: Iterable[str]) -> int | float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def text_value(payload: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def infer_family(
    payload: dict[str, Any],
    batch: dict[str, Any],
    spec: InputSpec,
    summary_path: Path,
) -> str:
    if spec.family_hint and spec.family_hint != "auto":
        return spec.family_hint
    joined = " ".join(
        [
            spec.label,
            str(spec.path),
            str(summary_path),
            text_value(payload, ("family", "experiment_family")),
            text_value(batch, ("family", "experiment_family")),
        ]
    ).lower()
    architecture = payload.get("architecture", {})
    if not isinstance(architecture, dict):
        architecture = {}
    parameter_count = number(architecture, ("parameter_count",))
    layers = number(architecture, ("n_layer", "num_layers"))
    if "bridge" in joined:
        return "gpt_bridge"
    if "llama" in joined or "swiglu" in joined:
        if (isinstance(parameter_count, (int, float)) and parameter_count >= 500_000_000) or (
            isinstance(layers, (int, float)) and layers >= 18
        ):
            return "llama_1b"
        return "llama_124m"
    if "r1" in joined or summary_path.name == "r1_summary.json":
        return "r1_native"
    return "unknown"


def infer_evidence_kind(payload: dict[str, Any], batch: dict[str, Any], summary_path: Path) -> str:
    # The 1B wrapper intentionally reuses the base runner, whose batch_kind is
    # "formal" for every non-smoke run.  Its execution_stage is the authoritative
    # distinction between the medium screen and the 6200-step formal run.
    for source in (batch, payload):
        value = text_value(source, ("execution_stage",))
        if value:
            return value
    for source in (batch, payload):
        value = text_value(source, ("batch_kind", "evidence_profile"))
        if value:
            return value
        if source.get("formal_evidence") is True:
            return "formal_evidence"
    joined = str(summary_path).lower()
    for kind in ("formal", "medium", "smoke", "pilot", "capacity"):
        if kind in joined:
            return kind
    return "unknown"


def normalize_status(payload: dict[str, Any], run_manifest: dict[str, Any]) -> str:
    value = text_value(payload, ("status",))
    if value:
        return value
    return text_value(run_manifest, ("status",)) or "unknown"


def load_sibling_run_manifest(summary_path: Path) -> tuple[Path | None, dict[str, Any]]:
    candidate = summary_path.parent / RUN_MANIFEST_NAME
    payload = json_load(candidate)
    return (candidate, payload) if payload is not None else (None, {})


def runtime_fields(payload: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    runtime: dict[str, Any] = {}
    for source in (batch.get("runtime"), payload.get("runtime")):
        if isinstance(source, dict):
            runtime.update(source)
    for source in (
        batch.get("training_runtime_fingerprint"),
        payload.get("training_runtime_fingerprint"),
    ):
        if isinstance(source, dict):
            runtime.update(source)
    return runtime


def inventory_summary(spec: InputSpec, summary_path: Path) -> dict[str, Any] | None:
    payload = json_load(summary_path)
    if payload is None:
        return None
    manifest_path, batch = find_nearest_manifest(summary_path, spec.path)
    run_manifest_path, run_manifest = load_sibling_run_manifest(summary_path)
    config: dict[str, Any] = {}
    for source in (batch.get("config"), payload.get("config")):
        if isinstance(source, dict):
            config.update(source)
    method = text_value(payload, ("method", "name")) or text_value(config, ("method",))
    if not method:
        return None
    checkpoint, resolution = resolve_checkpoint(payload.get("checkpoint_path"), summary_path)
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        architecture = {}
    runtime = runtime_fields(payload, batch)
    family = infer_family(payload, batch, spec, summary_path)
    status = normalize_status(payload, run_manifest)
    completed = status in {
        "completed",
        "completed_valid",
        "completed_valid_local",
        "completed_valid_smoke",
    }
    completed_steps = number(payload, ("completed_steps", "final_train_step", "total_steps"))
    if completed_steps is None:
        completed_steps = number(config, ("num_iterations", "total_steps", "steps"))
    tokens = number(payload, ("tokens_seen", "total_tokens"))
    seed = number(payload, ("seed", "controlled_seed"))
    if seed is None:
        seed = number(batch, ("seed", "controlled_seed"))
    if seed is None:
        seed = number(config, ("seed",))
    if family.startswith("llama"):
        checkpoint_contract = "llama_resumable_expected"
        exact_resume_expected = bool(checkpoint and completed)
    elif family in {"r1_native", "gpt_bridge"}:
        checkpoint_contract = "r1_model_checkpoint_expected"
        exact_resume_expected = False
    else:
        checkpoint_contract = "unknown_no_load_validation"
        exact_resume_expected = False
    file_stat = checkpoint.stat() if checkpoint else None
    return {
        "input_label": spec.label,
        "input_root": str(spec.path.resolve()),
        "family": family,
        "method": method,
        "seed": seed if seed is not None else "",
        "status": status,
        "completed": completed,
        "evidence_kind": infer_evidence_kind(payload, batch, summary_path),
        "completed_steps": completed_steps if completed_steps is not None else "",
        "tokens_seen": tokens if tokens is not None else "",
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": canonical_json_sha256(payload),
        "batch_manifest_path": str(manifest_path.resolve()) if manifest_path else "",
        "batch_manifest_sha256": canonical_json_sha256(batch) if batch else "",
        "run_manifest_path": str(run_manifest_path.resolve()) if run_manifest_path else "",
        "checkpoint_reference": str(payload.get("checkpoint_path", "") or ""),
        "checkpoint_path": str(checkpoint) if checkpoint else "",
        "checkpoint_resolution": resolution,
        "checkpoint_exists": checkpoint is not None,
        "checkpoint_bytes": file_stat.st_size if file_stat else 0,
        "checkpoint_mtime_ns": file_stat.st_mtime_ns if file_stat else "",
        "checkpoint_contract": checkpoint_contract,
        "checkpoint_schema_verified": False,
        "fresh_geometry_ready": checkpoint is not None,
        "exact_resume_expected": exact_resume_expected,
        "temporal_replay_ready": exact_resume_expected,
        "optimizer_state_reported": number(
            payload, ("optimizer_state_bytes", "optimizer_bytes")
        )
        or "",
        "loader_rng_state_reported": (
            "expected_unverified"
            if family.startswith("llama") and checkpoint is not None
            else "not_claimed"
        ),
        "init_sha256": text_value(payload, ("init_sha256",)),
        "source_sha256": text_value(
            payload, ("derived_script_sha256", "script_sha256", "source_sha256")
        )
        or text_value(batch, ("script_sha256",)),
        "official_repo": text_value(batch, ("official_repo",))
        or text_value(run_manifest, ("official_repo",)),
        "architecture_sha256": canonical_json_sha256(architecture)
        if architecture
        else "",
        "config_sha256": canonical_json_sha256(config) if config else "",
        "gpu_name": text_value(runtime, ("gpu_name", "gpu")),
        "gpu_total_memory_bytes": number(
            runtime, ("gpu_total_memory_bytes", "total_memory_bytes")
        )
        or "",
        "driver": text_value(runtime, ("driver", "driver_version")),
        "torch": text_value(runtime, ("torch", "torch_version")),
        "torch_cuda": text_value(runtime, ("torch_cuda", "cuda")),
        "triton": text_value(runtime, ("triton",)),
        "python_executable": text_value(runtime, ("python_executable",)),
        "python_version": text_value(runtime, ("python_version",)),
    }


def discover_summaries(spec: InputSpec) -> list[Path]:
    if not spec.path.is_dir():
        return []
    paths: dict[str, Path] = {}
    for name in SUMMARY_NAMES:
        for path in spec.path.rglob(name):
            if path.is_file():
                paths[str(path.resolve())] = path
    return sorted(paths.values(), key=lambda item: str(item))


def hash_rows(inventory: list[dict[str, Any]], hash_mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_path: dict[str, dict[str, Any]] = {}
    for item in inventory:
        path = str(item["checkpoint_path"])
        if not path or path in by_path:
            continue
        by_path[path] = item
    for path_text, item in sorted(by_path.items()):
        path = Path(path_text)
        base = {
            "checkpoint_path": path_text,
            "family": item["family"],
            "method": item["method"],
            "seed": item["seed"],
            "completed_steps": item["completed_steps"],
            "checkpoint_bytes": item["checkpoint_bytes"],
        }
        if hash_mode == "none":
            base.update(
                {
                    "sha256": "",
                    "hash_status": "deferred",
                    "size_before": item["checkpoint_bytes"],
                    "size_after": item["checkpoint_bytes"],
                    "mtime_ns_before": item["checkpoint_mtime_ns"],
                    "mtime_ns_after": item["checkpoint_mtime_ns"],
                }
            )
        else:
            try:
                base.update(sha256_file_stable(path))
            except OSError as exc:
                base.update(
                    {
                        "sha256": "",
                        "hash_status": f"failed:{type(exc).__name__}",
                        "size_before": "",
                        "size_after": "",
                        "mtime_ns_before": "",
                        "mtime_ns_after": "",
                    }
                )
        rows.append(base)
    return rows


def eligible_for_step_map(row: dict[str, Any], include_incomplete: bool) -> bool:
    if not row["checkpoint_exists"]:
        return False
    if not isinstance(row["completed_steps"], (int, float)):
        return False
    return include_incomplete or bool(row["completed"])


def kind_priority(kind: str) -> int:
    normalized = kind.lower()
    if normalized in FORMAL_KINDS:
        return 0
    if normalized == "unknown":
        return 1
    if normalized == "pilot":
        return 2
    if normalized == "smoke":
        return 3
    return 4


def build_step_map(
    inventory: list[dict[str, Any]],
    target_steps: list[int],
    include_incomplete: bool,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in inventory:
        if not eligible_for_step_map(row, include_incomplete):
            continue
        key = (str(row["family"]), str(row["method"]), str(row["seed"]))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (family, method, seed), rows in sorted(groups.items()):
        for target in sorted(set(target_steps)):
            nearest = min(
                rows,
                key=lambda row: (
                    abs(int(row["completed_steps"]) - target),
                    kind_priority(str(row["evidence_kind"])),
                    -int(row["completed_steps"]),
                ),
            )
            available = int(nearest["completed_steps"])
            output.append(
                {
                    "family": family,
                    "method": method,
                    "seed": seed,
                    "target_step": target,
                    "available_step": available,
                    "distance_steps": abs(available - target),
                    "exact_match": available == target,
                    "evidence_kind": nearest["evidence_kind"],
                    "status": nearest["status"],
                    "checkpoint_path": nearest["checkpoint_path"],
                    "summary_path": nearest["summary_path"],
                    "mapping_policy": "nearest_real_checkpoint_no_interpolation",
                }
            )
    return output


def diagnostic_contract(
    args: argparse.Namespace,
    inventory: list[dict[str, Any]],
    step_map: list[dict[str, Any]],
) -> dict[str, Any]:
    exact = sum(bool(row["exact_match"]) for row in step_map)
    return {
        "schema_version": 1,
        "created_at": now_iso(),
        "mech_id": "MECH-00",
        "host_id": args.host_id,
        "execution_domain": args.execution_domain,
        "primary_trajectories": {
            "r1_native": "none",
            "gpt_bridge": "none",
            "llama_124m": "down_none",
            "llama_1b": "down_none",
        },
        "secondary_trajectory": "muon",
        "target_steps": sorted(set(args.target_steps)),
        "step_mapping_policy": (
            "Map each target to the nearest checkpoint that really exists. "
            "Never interpolate or splice checkpoints. Preserve family, seed, "
            "host, and execution-domain boundaries."
        ),
        "checkpoint_loading_policy": (
            "MECH-00 does not torch.load checkpoints. checkpoint_schema_verified "
            "therefore remains false until MECH-01 performs a read-only load audit."
        ),
        "r1_resume_policy": (
            "R1 checkpoints are accepted for fresh-batch geometry diagnostics, "
            "but exact loader/RNG/scheduler replay is not claimed."
        ),
        "llama_resume_policy": (
            "Completed LLaMA formal checkpoints are marked resumable_expected "
            "from the training contract, pending MECH-01 schema verification."
        ),
        "cross_host_policy": (
            "Do not average raw timing, peak memory, or tiny numerical differences "
            "across execution domains. Prefer GPT-bridge versus LLaMA-124M on the "
            "same LLaMA host; use R1-native as robustness evidence."
        ),
        "frozen_data_policy": (
            "MECH-01 must record tokenizer/data fingerprints and prove that probe "
            "token IDs are identical before treating GPT/LLaMA batches as shared."
        ),
        "inventory_rows": len(inventory),
        "step_map_rows": len(step_map),
        "exact_target_matches": exact,
        "checkpoint_hash_mode": args.hash_mode,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        observed: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    observed.append(key)
                    seen.add(key)
        fieldnames = observed
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_checks(
    inputs: list[InputSpec],
    inventory: list[dict[str, Any]],
    hashes: list[dict[str, Any]],
    repo_inventory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for spec in inputs:
        count = sum(row["input_label"] == spec.label for row in inventory)
        checks.append(
            {
                "check": f"input:{spec.label}",
                "status": "pass" if spec.path.is_dir() else "fail",
                "detail": f"path={spec.path.resolve()} inventory_rows={count}",
            }
        )
    usable = sum(bool(row["fresh_geometry_ready"]) for row in inventory)
    checks.append(
        {
            "check": "usable_checkpoint_count",
            "status": "pass" if usable else "fail",
            "detail": str(usable),
        }
    )
    unstable = [
        row for row in hashes if row["hash_status"] not in {"deferred", "verified_stable"}
    ]
    checks.append(
        {
            "check": "checkpoint_hash_stability",
            "status": "pass" if not unstable else "fail",
            "detail": (
                "hashing deferred" if hashes and hashes[0]["hash_status"] == "deferred"
                else f"unstable_or_failed={len(unstable)}"
            ),
        }
    )
    for repo in repo_inventory:
        repo_status = str(repo["audit_status"])
        checks.append(
            {
                "check": f"repo:{repo['repo_label']}",
                "status": (
                    "pass"
                    if repo_status == "ok_clean"
                    else "warn"
                    if repo_status == "ok_dirty"
                    else "fail"
                ),
                "detail": (
                    f"path={repo['repo_path']} audit_status={repo_status} "
                    f"commit={repo.get('git_commit', '')} "
                    f"error={repo.get('commit_probe_error', '') or repo.get('status_probe_error', '')}"
                ),
            }
        )
    duplicate_key_count = len(inventory) - len(
        {
            (
                row["family"],
                row["method"],
                str(row["seed"]),
                str(row["completed_steps"]),
                row["checkpoint_path"],
            )
            for row in inventory
        }
    )
    checks.append(
        {
            "check": "duplicate_inventory_keys",
            "status": "warn" if duplicate_key_count else "pass",
            "detail": str(duplicate_key_count),
        }
    )
    return checks


def main() -> int:
    args = parse_args()
    raw_inputs = parse_assignments(args.input, "--input")
    if not raw_inputs:
        raise SystemExit("at least one --input LABEL=PATH is required")
    family_hints = parse_assignments(args.family_hint, "--family-hint")
    unknown_hints = sorted(set(family_hints) - set(raw_inputs))
    if unknown_hints:
        raise SystemExit(f"--family-hint labels without matching --input: {unknown_hints}")
    inputs = [
        InputSpec(label, Path(value).expanduser(), family_hints.get(label, "auto"))
        for label, value in raw_inputs.items()
    ]
    repos = [
        RepoSpec(label, Path(value).expanduser())
        for label, value in parse_assignments(args.repo, "--repo").items()
    ]
    methods = set(args.methods)

    inventory: list[dict[str, Any]] = []
    discovery: list[dict[str, Any]] = []
    for spec in inputs:
        summary_paths = discover_summaries(spec)
        accepted = 0
        for path in summary_paths:
            row = inventory_summary(spec, path)
            if row is None:
                continue
            if methods and row["method"] not in methods:
                continue
            row["host_id"] = args.host_id
            row["execution_domain"] = args.execution_domain
            inventory.append(row)
            accepted += 1
        discovery.append(
            {
                "input_label": spec.label,
                "input_root": str(spec.path.resolve()),
                "family_hint": spec.family_hint,
                "path_exists": spec.path.is_dir(),
                "summary_files_discovered": len(summary_paths),
                "inventory_rows_accepted": accepted,
            }
        )
    inventory.sort(
        key=lambda row: (
            str(row["family"]),
            str(row["method"]),
            str(row["seed"]),
            str(row["completed_steps"]),
            str(row["summary_path"]),
        )
    )
    hashes = hash_rows(inventory, args.hash_mode)
    step_map = build_step_map(inventory, args.target_steps, args.include_incomplete)
    host_runtime = collect_host_runtime(args.host_id, args.execution_domain)
    repo_inventory = [audit_repo(spec) for spec in repos]
    checks = audit_checks(inputs, inventory, hashes, repo_inventory)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "checkpoint_inventory.csv", inventory)
    write_csv(output / "checkpoint_hashes.csv", hashes)
    write_csv(output / "available_step_map.csv", step_map)
    write_csv(output / "input_discovery.csv", discovery)
    write_csv(output / "source_inventory.csv", repo_inventory)
    write_csv(output / "audit_checks.csv", checks)
    write_json(output / "runtime_inventory.json", host_runtime)
    contract = diagnostic_contract(args, inventory, step_map)
    write_json(output / "diagnostic_data_contract.json", contract)
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at": now_iso(),
        "command": sys.argv,
        "host_id": args.host_id,
        "execution_domain": args.execution_domain,
        "hash_mode": args.hash_mode,
        "methods_filter": args.methods,
        "target_steps": args.target_steps,
        "inputs": [asdict(spec) | {"path": str(spec.path.resolve())} for spec in inputs],
        "repos": [asdict(spec) | {"path": str(spec.path.resolve())} for spec in repos],
        "outputs": {
            "checkpoint_inventory": "checkpoint_inventory.csv",
            "checkpoint_hashes": "checkpoint_hashes.csv",
            "available_step_map": "available_step_map.csv",
            "input_discovery": "input_discovery.csv",
            "source_inventory": "source_inventory.csv",
            "audit_checks": "audit_checks.csv",
            "runtime_inventory": "runtime_inventory.json",
            "diagnostic_data_contract": "diagnostic_data_contract.json",
        },
        "counts": {
            "inventory_rows": len(inventory),
            "checkpoint_files": len(hashes),
            "step_map_rows": len(step_map),
            "failed_checks": sum(row["status"] == "fail" for row in checks),
        },
    }
    write_json(output / "mech00_manifest.json", manifest)

    print(f"MECH-00 output: {output}")
    print(
        "MECH-00 counts: "
        f"inventory_rows={len(inventory)} checkpoints={len(hashes)} "
        f"step_map_rows={len(step_map)}"
    )
    print(f"Checkpoint hash mode: {args.hash_mode}")
    failed = [row for row in checks if row["status"] == "fail"]
    if failed:
        print("MECH-00 failed checks:")
        for row in failed:
            print(f"- {row['check']}: {row['detail']}")
    if args.strict and failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
