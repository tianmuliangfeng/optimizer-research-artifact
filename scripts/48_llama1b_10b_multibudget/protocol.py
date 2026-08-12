#!/usr/bin/env python3
"""Pure-stdlib contract, data, cursor, and artifact helpers for experiment 48."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any, Iterable


SCHEMA = "ex48_llama1b_10b_multibudget_formal_v1"
CHECKPOINT_SCHEMA = "ex48_llama1b_segment_checkpoint_v1"
PHASE_MANIFEST_SCHEMA = "ex48_llama1b_phase_manifest_v1"
UNIT_MANIFEST_SCHEMA = "ex48_llama1b_unit_manifest_v1"
DATA_AUDIT_SCHEMA = "ex48_llama1b_data_audit_v1"
METRIC_FIELDS = (
    "event",
    "phase_id",
    "schedule",
    "step",
    "segment_step",
    "loss",
    "train_s",
    "steady_train_s",
    "step_avg_ms",
    "lr_backup",
    "lr_matrix",
    "tokens_seen",
    "tokens_per_parameter",
    "loader_consumed_batches",
    "wrap_count",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def shard_index(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.bin$)", path.name)
    if match is None:
        raise ValueError(f"shard name has no numeric suffix: {path.name}")
    return int(match.group(1))


def phase_map(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in contract["phases"]}


def endpoint_phases(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in contract["phases"] if row["role"] == "primary_endpoint"]


def direct_children(contract: dict[str, Any], phase_id: str) -> list[str]:
    return [str(row["id"]) for row in contract["phases"] if row.get("parent") == phase_id]


def validate_contract(contract: dict[str, Any]) -> dict[str, bool]:
    phases = phase_map(contract)
    training = contract["training"]
    profile = contract["profile"]
    endpoints = endpoint_phases(contract)
    phase_shape = [
        (row["id"], row.get("parent"), row["start_step"], row["target_step"], row["schedule"])
        for row in contract["phases"]
    ]
    expected_shape = [
        ("backbone_4400", None, 0, 4400, "plateau"),
        ("cooldown_6200", "backbone_4400", 4400, 6200, "linear_cooldown"),
        ("backbone_11493", "backbone_4400", 4400, 11493, "plateau"),
        ("cooldown_13293", "backbone_11493", 11493, 13293, "linear_cooldown"),
        ("backbone_17273", "backbone_11493", 11493, 17273, "plateau"),
        ("cooldown_19073", "backbone_17273", 17273, 19073, "linear_cooldown"),
    ]
    parent_steps = all(
        row.get("parent") is None
        or phases[str(row["parent"])]["target_step"] == row["start_step"]
        for row in contract["phases"]
    )
    tokens = [int(row["target_step"]) * int(training["tokens_per_update"]) for row in endpoints]
    tpp = [value / int(profile["parameters"]) for value in tokens]
    amendments = contract.get("engineering_amendments_before_formal", [])
    amendments_by_field = {
        str(row.get("field")): row for row in amendments if isinstance(row, dict)
    }
    kernel_amendment = amendments_by_field.get(
        "runtime.accepted_triton_kernels_sha256", {}
    )
    rng_amendment = amendments_by_field.get(
        "checkpoint_rng_restore_device_normalization", {}
    )
    host_amendment = amendments_by_field.get("execution.single_host_gpu_count", {})
    return {
        "schema": contract.get("schema_version") == SCHEMA,
        "experiment": int(contract.get("experiment_number", -1)) == 48,
        "formal_authorized": contract.get("status")
        == "preregistered_formal_authorized_by_user_20260805",
        "grid": contract["grid"]["methods"]
        == ["down_none", "down_diag", "newton_full", "muon"]
        and contract["grid"]["seeds"] == [2024, 2025, 2026]
        and int(contract["grid"]["formal_units"]) == 12
        and int(contract["grid"]["host_count"]) == 1
        and int(contract["grid"]["gpus"]) == 4,
        "batch_geometry": int(training["global_batch_size"]) == 512
        and int(training["device_batch_size"]) == 8
        and int(training["sequence_length"]) == 1024
        and int(training["gradient_accumulation_steps"]) == 64
        and int(training["tokens_per_update"]) == 524288,
        "phase_shape": phase_shape == expected_shape,
        "parent_steps": parent_steps,
        "equal_cooldowns": all(
            int(row["target_step"]) - int(row["start_step"]) == 1800
            for row in endpoints
        ),
        "endpoint_tokens": tokens == [3250585600, 6969360384, 9999745024]
        and all(int(row["actual_tokens"]) == value for row, value in zip(endpoints, tokens)),
        "endpoint_tpp": all(
            math.isclose(float(row["tokens_per_parameter"]), value, rel_tol=0, abs_tol=1e-15)
            for row, value in zip(endpoints, tpp)
        ),
        "no_wrap": int(contract["data"]["minimum_contiguous_train_shards"]) >= 101
        and int(contract["data"]["wrap_count_required"]) == 0,
        "full_data_hash": contract["data"]["full_content_sha256_required"] is True,
        "runtime_kernel_amendment": contract["runtime"]["accepted_triton_kernels_sha256"]
        == "b51ac50c699b05306619d92cb9ec6edadd266d8118c53f5b9726db76480ea16d"
        and kernel_amendment.get("field") == "runtime.accepted_triton_kernels_sha256"
        and kernel_amendment.get("old_value")
        == "f092ae994f5a5c1ebacf3938e2bb8d610dc537b928e2a5039438afe1e46a271f"
        and kernel_amendment.get("new_value")
        == "b51ac50c699b05306619d92cb9ec6edadd266d8118c53f5b9726db76480ea16d"
        and kernel_amendment.get("scientific_configuration_changed") is False
        and kernel_amendment.get("outcome_observed") is False,
        "pilot_rng_amendment": rng_amendment.get("stage")
        == "outcome_free_exact_resume_pilot_before_formal"
        and rng_amendment.get("field") == "checkpoint_rng_restore_device_normalization"
        and rng_amendment.get("scientific_configuration_changed") is False
        and rng_amendment.get("outcome_observed") is False,
        "single_host_four_gpu_amendment": host_amendment.get("stage")
        == "after_aborted_unaccepted_formal_before_replacement_preflight"
        and host_amendment.get("old_value") == 2
        and host_amendment.get("new_value") == 4
        and host_amendment.get("scientific_configuration_changed") is False
        and host_amendment.get("analysis_rules_changed") is False
        and host_amendment.get("prior_run_accepted_as_evidence") is False
        and host_amendment.get("prior_run_outcome_used_for_amendment") is False
        and host_amendment.get("old_run_resumable") is False
        and host_amendment.get("timing_claim_eligible") is False,
        "runtime_gpu_geometry": int(contract["runtime"]["gpu_count"]) == 4
        and contract["runtime"]["nvidia_driver"] == "580.95.05",
        "public_packaging_lineage": contract.get("public_source_hashes", {}).get(
            "scripts/20_llama_swiglu_1b/run_llama_swiglu_1b.py"
        )
        == "8ecb634b751017397bc9bd60f032fdd890b2d5c1a4462de7b697e1bdf97ef8c4"
        and contract.get("public_packaging", {}).get("change_scope")
        == "default_result_root_only"
        and contract.get("public_packaging", {}).get(
            "historical_source_hash_retained"
        )
        is True
        and contract.get("public_packaging", {}).get(
            "scientific_configuration_changed"
        )
        is False,
        "resume_lineage": contract["resume"]["source_checkpoint_sha256_required"] is True
        and contract["resume"]["loader_cursor_must_match_completed_steps"] is True,
        "endpoint_retention": contract["checkpoint_retention"]["retain_primary_endpoints_only"] is True
        and int(contract["checkpoint_retention"]["retained_endpoint_checkpoints_per_unit"]) == 3,
    }


def assert_contract(contract: dict[str, Any]) -> None:
    checks = validate_contract(contract)
    if not all(checks.values()):
        raise RuntimeError(f"experiment 48 contract failed: {checks}")


def read_shard_header(path: Path, contract: dict[str, Any], full_hash: bool) -> dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(int(contract["data"]["header_bytes"]))
    if len(header) != int(contract["data"]["header_bytes"]):
        raise RuntimeError(f"short FineWeb header: {path}")
    magic, version, tokens = struct.unpack("<iii", header[:12])
    if magic != int(contract["data"]["header_magic"]):
        raise RuntimeError(f"FineWeb magic mismatch: {path}")
    if version != int(contract["data"]["header_version"]):
        raise RuntimeError(f"FineWeb version mismatch: {path}")
    stat = path.stat()
    expected_bytes = int(contract["data"]["header_bytes"]) + int(tokens) * int(
        contract["data"]["token_dtype_bytes"]
    )
    microbatch = int(contract["training"]["microbatch_tokens"])
    return {
        "name": path.name,
        "index": shard_index(path),
        "tokens": int(tokens),
        "consumable_tokens": ((int(tokens) - 1) // microbatch) * microbatch,
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "expected_bytes": expected_bytes,
        "size_exact": int(stat.st_size) == expected_bytes,
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "sha256": sha256_file(path) if full_hash else None,
    }


def audit_data_dir(data_dir: Path, contract: dict[str, Any], full_hash: bool = True) -> dict[str, Any]:
    root = data_dir.absolute()
    train = sorted(root.glob(contract["data"]["train_pattern"]), key=shard_index)
    validation = sorted(root.glob(contract["data"]["validation_pattern"]), key=shard_index)
    train_rows = [read_shard_header(path, contract, full_hash) for path in train]
    val_rows = [read_shard_header(path, contract, full_hash) for path in validation]
    indices = [row["index"] for row in train_rows]
    contiguous = bool(indices) and indices == list(range(indices[0], indices[0] + len(indices)))
    maximum_tokens = max(int(row["target_step"]) for row in endpoint_phases(contract)) * int(
        contract["training"]["tokens_per_update"]
    )
    prefetch = int(contract["data"]["prefetched_train_microbatches"]) * int(
        contract["training"]["microbatch_tokens"]
    )
    total_consumable = sum(int(row["consumable_tokens"]) for row in train_rows)
    checks = {
        "minimum_train_shards": len(train_rows)
        >= int(contract["data"]["minimum_contiguous_train_shards"]),
        "first_train_index": bool(indices)
        and indices[0] == int(contract["data"]["expected_first_train_index"]),
        "train_indices_unique": len(indices) == len(set(indices)),
        "train_indices_contiguous": contiguous,
        "validation_shards": len(val_rows) == int(contract["data"]["required_validation_shards"]),
        "file_sizes_exact": all(row["size_exact"] for row in [*train_rows, *val_rows]),
        "no_wrap_capacity": total_consumable >= maximum_tokens + prefetch,
        "full_hashes": (not contract["data"]["full_content_sha256_required"])
        or (full_hash and all(row["sha256"] for row in [*train_rows, *val_rows])),
    }
    identity = {"train": train_rows, "validation": val_rows}
    return {
        "schema_version": DATA_AUDIT_SCHEMA,
        "data_dir": str(root),
        "full_hash": bool(full_hash),
        "required_stream_tokens": maximum_tokens + prefetch,
        "total_consumable_train_tokens": total_consumable,
        "unused_consumable_tokens": total_consumable - maximum_tokens - prefetch,
        "train_shard_count": len(train_rows),
        "validation_shard_count": len(val_rows),
        "inventory_sha256": canonical_sha256(identity),
        "inventory": identity,
        "checks": checks,
        "passed": all(checks.values()),
    }


def verify_data_metadata(audit: dict[str, Any]) -> dict[str, bool]:
    root = Path(audit["data_dir"])
    checks: dict[str, bool] = {}
    for split in ("train", "validation"):
        for row in audit["inventory"][split]:
            path = root / row["name"]
            key = f"{split}:{row['name']}"
            if not path.is_file():
                checks[key] = False
                continue
            stat = path.stat()
            checks[key] = int(stat.st_size) == int(row["bytes"]) and int(stat.st_mtime_ns) == int(
                row["mtime_ns"]
            )
    return checks


def cursor_after_batches(
    consumable_tokens: Iterable[int], microbatch_tokens: int, consumed_batches: int
) -> dict[str, int]:
    capacities = [int(value) // int(microbatch_tokens) for value in consumable_tokens]
    if not capacities or any(value <= 0 for value in capacities):
        raise ValueError("every shard must contain a consumable microbatch")
    remaining = int(consumed_batches)
    if remaining < 0:
        raise ValueError("consumed_batches must be non-negative")
    for shard, capacity in enumerate(capacities):
        if remaining < capacity:
            return {
                "current_shard": shard,
                "current_position": remaining * int(microbatch_tokens),
                "wrap_count": 0,
                "consumed_batches": int(consumed_batches),
            }
        remaining -= capacity
    return {
        "current_shard": len(capacities),
        "current_position": 0,
        "wrap_count": 1,
        "consumed_batches": int(consumed_batches),
    }


def expected_cursor(completed_steps: int, audit: dict[str, Any], contract: dict[str, Any]) -> dict[str, int]:
    batches = int(contract["data"]["prefetched_train_microbatches"]) + int(
        completed_steps
    ) * int(contract["training"]["gradient_accumulation_steps"])
    return cursor_after_batches(
        [row["consumable_tokens"] for row in audit["inventory"]["train"]],
        int(contract["training"]["microbatch_tokens"]),
        batches,
    )


def cursor_matches(observed: dict[str, Any], expected: dict[str, int]) -> bool:
    keys = ("current_shard", "current_position", "wrap_count", "consumed_batches")
    return all(int(observed.get(key, -1)) == int(expected[key]) for key in keys)


def lr_multiplier(phase: dict[str, Any], completed_steps: int) -> float:
    if phase["schedule"] == "plateau":
        return 1.0
    if phase["schedule"] != "linear_cooldown":
        raise ValueError(f"unsupported phase schedule: {phase['schedule']}")
    start = int(phase["start_step"])
    target = int(phase["target_step"])
    return max(0.0, float(target - int(completed_steps)) / float(target - start))


def should_validate(phase: dict[str, Any], step: int, every: int) -> bool:
    return int(step) in (int(phase["start_step"]), int(phase["target_step"])) or int(step) % int(every) == 0


def read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def trim_metrics(path: Path, completed_steps: int) -> None:
    rows = read_metrics(path)
    if not rows:
        return
    kept = [
        row
        for row in rows
        if int(row["step"]) < int(completed_steps)
        or (int(row["step"]) == int(completed_steps) and row["event"] == "train")
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(kept)


def append_metric(path: Path, row: dict[str, Any]) -> None:
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())
