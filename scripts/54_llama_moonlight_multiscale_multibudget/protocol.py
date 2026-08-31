#!/usr/bin/env python3
"""Pure-stdlib EX54 contract, data, cursor, and metric helpers."""

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


SCHEMA = "ex54_llama_moonlight_multiscale_multibudget_v2"
CHECKPOINT_SCHEMA = "ex48_llama1b_segment_checkpoint_v1"
PHASE_MANIFEST_SCHEMA = "ex48_llama1b_phase_manifest_v1"
UNIT_MANIFEST_SCHEMA = "ex54_llama_moonlight_unit_v1"
DATA_AUDIT_SCHEMA = "ex54_fineweb_full_hash_audit_v1"
ACCEPTED_DATA_PROJECTION_SCHEMA = "ex54_accepted_ex48_data_projection_v1"
ACCEPTED_DATA_PROJECTION_FIELDS = (
    "split", "ordinal", "name", "index", "tokens", "consumable_tokens",
    "bytes", "header_sha256", "sha256",
)
METRIC_FIELDS = (
    "event", "phase_id", "schedule", "step", "segment_step", "loss",
    "train_s", "steady_train_s", "step_avg_ms", "lr_backup", "lr_matrix",
    "tokens_seen", "tokens_per_parameter", "loader_consumed_batches", "wrap_count",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_text(path: Path, text: str) -> None:
    """Atomically replace *path* using a unique same-directory temp file.

    The old implementation used a fixed ``<name>.tmp`` path.  That is safe
    only for serial writers.  EX54/EX57 tuning and formal scheduling run
    independent single-GPU jobs concurrently, so shared compatibility
    receipts can be materialized by multiple controller threads at once.
    A unique temp file per writer removes that race while preserving the
    exact final bytes and atomic rename semantics.
    """
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


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
    expected_shape = [
        ("backbone_4400", None, 0, 4400, "plateau"),
        ("cooldown_6200", "backbone_4400", 4400, 6200, "linear_cooldown"),
        ("backbone_11493", "backbone_4400", 4400, 11493, "plateau"),
        ("cooldown_13293", "backbone_11493", 11493, 13293, "linear_cooldown"),
    ]
    shape = [
        (row["id"], row.get("parent"), row["start_step"], row["target_step"], row["schedule"])
        for row in contract.get("phases", [])
    ]
    endpoints = endpoint_phases(contract)
    training = contract.get("training", {})
    tokens = [
        int(row["target_step"]) * int(training.get("tokens_per_update", -1))
        for row in endpoints
    ]
    tuning = contract.get("tuning", {})
    formal = contract.get("formal", {})
    execution = contract.get("execution", {})
    moonlight = contract.get("moonlight", {})
    expected_cells = [
        ("lr0010", 0.001), ("lr0018", 0.0018), ("lr0030", 0.003)
    ]

    def tuning_track_ok(scale: str, seed: int) -> bool:
        track = tuning.get(scale, {})
        cells = track.get("cells", [])
        return (
            track.get("seed") == seed
            and track.get("updates") == 1000
            and track.get("center_cell") == "lr0018"
            and len(cells) == 3
            and [(row.get("id"), float(row.get("matrix_lr", -1))) for row in cells]
            == expected_cells
            and all(float(row.get("backup_lr", -1)) == float(row.get("matrix_lr", -2)) for row in cells)
            and all(float(row.get("weight_decay", -1)) == 0.1 for row in cells)
        )

    checks = {
        "schema": contract.get("schema_version") == SCHEMA,
        "experiment": int(contract.get("experiment_number", -1)) == 54
        and contract.get("experiment_id") == "54_llama_moonlight_multiscale_multibudget",
        "frozen": contract.get("status") == "preregistered_authorized_20260819",
        "method": contract.get("method") == "moonlight",
        "moonlight_recipe": moonlight == {
            "transfer_source_experiment": 19,
            "upstream_authority": "MoonshotAI/Moonlight examples/toy_train.py",
            "momentum": 0.95,
            "nesterov": True,
            "newton_schulz_steps": 5,
            "newton_schulz_coefficients": [3.4445, -4.775, 2.0315],
            "adjust_lr": "0.2*sqrt(max(rows,cols))",
            "weight_decay": 0.1,
            "weight_decay_uses_unadjusted_base_lr": True,
            "backup_route": "accepted_llama_adamw",
            "backup_route_scope": "tied_embedding_head_norm_and_nonmatrix_parameters",
            "backup_lr_policy": "same_base_lr_as_matrix",
            "activation_k_state_routes": 0,
            "factor_or_eigendecomposition_state": False,
            "all_hidden_2d_matrices": True,
            "transfer_source_sha256": "bf39d7e1b435ef737833046c564ce8770d858d1aa474c9d7f11a914057253655",
        },
        "tuning": tuning_track_ok("124m", 5401) and tuning_track_ok("1b", 5402)
        and {int(tuning[scale]["seed"]) for scale in ("124m", "1b")}.isdisjoint(
            {int(seed) for seed in formal.get("seeds", [])}
        ),
        "formal": formal.get("seeds") == [2024, 2025, 2026]
        and int(formal.get("units", -1)) == 6
        and formal.get("tracks") == ["124m_3p25b", "1b_multibudget_non10b"]
        and formal.get("accepted_1b_budget_ids") == ["tokens_3p2506b", "tokens_6p9694b"]
        and formal.get("independent_of_ex57") is True
        and "ten_b_continuation_experiment" not in formal,
        "execution": int(execution.get("host_count", -1)) == 1
        and execution.get("physical_gpus") == [0, 1]
        and int(execution.get("maximum_concurrent_training_processes", -1)) == 2
        and int(execution.get("tuning_parallel_workers", -1)) == 2
        and int(execution.get("formal_parallel_workers", -1)) == 2
        and execution.get("parallelism") == "independent_single_gpu_jobs"
        and execution.get("ddp") is False
        and execution.get("quality_timing_eligible") is False,
        "batch": int(training.get("global_batch_size", -1)) == 512
        and int(training.get("sequence_length", -1)) == 1024
        and int(training.get("tokens_per_update", -1)) == 524288
        and int(training.get("device_batch_size_1b", -1)) == 8
        and int(training.get("microbatch_tokens", -1)) == 8192
        and int(training.get("gradient_accumulation_steps_1b", -1)) == 64
        and int(training.get("device_batch_size_1b", -1))
        * int(training.get("gradient_accumulation_steps_1b", -1))
        == int(training.get("global_batch_size", -2)),
        "fairness": contract.get("fairness", {}) == {
            "reference_experiment": 48,
            "backup_adamw_betas": [0.9, 0.95],
            "backup_adamw_eps": 1e-8,
            "backup_adamw_weight_decay": 0.0,
            "device_batch_size_124m": 64,
            "device_batch_size_1b": 8,
            "gradient_accumulation_steps_1b": 64,
            "global_batch_size": 512,
            "gradient_accumulation_steps_124m": 8,
            "tokens_per_update": 524288,
            "same_microbatch_geometry_as_ex48": True,
            "same_data_projection_as_ex48": True,
            "timing_eligible": False,
            "tuning_formal_seed_disjoint": True,
            "tuning_seeds": {"124m": 5401, "1b": 5402},
            "formal_seeds": [2024, 2025, 2026],
            "tuning_cells_per_scale": 3,
            "tuning_updates_per_cell": 1000,
        },
        "moonlight_transfer_lineage": contract.get("accepted_sources", {}).get(
            "parent/scripts/19_r1_extended_baselines/extended_optimizers.py"
        ) == "bf39d7e1b435ef737833046c564ce8770d858d1aa474c9d7f11a914057253655"
        and contract.get("source_derivation", {}).get("algorithm_subtree_exact_match_required") is True
        and contract.get("source_derivation", {}).get("pinned_ex19_optimizer_source_sha256")
        == "bf39d7e1b435ef737833046c564ce8770d858d1aa474c9d7f11a914057253655",
        "phases": shape == expected_shape,
        "parent_steps": all(
            row.get("parent") is None
            or phases[str(row["parent"])]["target_step"] == row["start_step"]
            for row in contract.get("phases", [])
        ),
        "equal_cooldowns": len(endpoints) == 2 and all(
            int(row["target_step"]) - int(row["start_step"]) == 1800 for row in endpoints
        ),
        "endpoint_tokens": tokens == [3250585600, 6969360384],
        "no_retune": formal.get("retune_after_formal_forbidden") is True,
        "timing": formal.get("timing_eligible") is False,
        "long_data": int(contract.get("data", {}).get("1b", {}).get("exact_train_shards", 0)) == 103
        and contract.get("data", {}).get("1b", {}).get("accepted_projection_path")
        == "accepted_ex48_data_projection.json",
        "checkpoint": contract.get("resume", {}).get("full_checkpoint_sha256_required") is True,
        "independence": all("57" not in str(value) for value in contract.get("accepted_sources", {}).keys())
        and contract.get("formal", {}).get("independent_of_ex57") is True,
    }
    return checks

def assert_contract(contract: dict[str, Any]) -> None:
    checks = validate_contract(contract)
    if not all(checks.values()):
        raise RuntimeError(f"experiment 54 contract failed: {checks}")


def _read_shard(path: Path, microbatch_tokens: int, full_hash: bool) -> dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(1024)
    if len(header) != 1024:
        raise RuntimeError(f"short FineWeb header: {path}")
    magic, version, tokens = struct.unpack("<iii", header[:12])
    stat = path.stat()
    expected_bytes = 1024 + 2 * int(tokens)
    if magic != 20240520 or version != 1 or stat.st_size != expected_bytes:
        raise RuntimeError(
            f"invalid FineWeb shard {path}: magic={magic} version={version} "
            f"bytes={stat.st_size} expected={expected_bytes}"
        )
    return {
        "name": path.name,
        "index": shard_index(path),
        "tokens": int(tokens),
        "num_tokens": int(tokens),
        "consumable_tokens": ((int(tokens) - 1) // int(microbatch_tokens)) * int(microbatch_tokens),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "sha256": sha256_file(path) if full_hash else None,
    }


def ex52_content_fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["name"]).encode("utf-8"))
        digest.update(str(row["bytes"]).encode("ascii"))
        digest.update(str(row["tokens"]).encode("ascii"))
        digest.update(str(row["sha256"]).encode("ascii"))
    return digest.hexdigest()


def project_data_inventory(identity: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return EX48's path/mtime-independent, order-sensitive data identity."""
    projected: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation"):
        projected[split] = [
            {
                "split": split,
                "ordinal": ordinal,
                "name": str(row["name"]),
                "index": int(row["index"]),
                "tokens": int(row["tokens"]),
                "consumable_tokens": int(row["consumable_tokens"]),
                "bytes": int(row["bytes"]),
                "header_sha256": str(row["header_sha256"]),
                "sha256": str(row["sha256"]),
            }
            for ordinal, row in enumerate(identity[split])
        ]
    return projected


def validate_accepted_data_projection(
    projection: dict[str, Any], contract: dict[str, Any]
) -> dict[str, bool]:
    """Validate the committed EX48 projection independently of its path."""
    spec = contract["data"]["1b"]
    inventory = projection.get("inventory", {})
    train = inventory.get("train", []) if isinstance(inventory, dict) else []
    validation = inventory.get("validation", []) if isinstance(inventory, dict) else []
    structurally_projectable = (
        isinstance(train, list)
        and isinstance(validation, list)
        and all(isinstance(row, dict) for row in [*train, *validation])
    )
    try:
        normalized = project_data_inventory({"train": train, "validation": validation})
    except (KeyError, TypeError, ValueError):
        normalized = {"train": [], "validation": []}
        structurally_projectable = False
    checks = {
        "schema": projection.get("schema_version") == ACCEPTED_DATA_PROJECTION_SCHEMA,
        "source_experiment": int(projection.get("source_experiment", -1)) == 48,
        "source_run": projection.get("source_run_id") == "20260805T061608+0000",
        "source_data_audit": projection.get("source_data_audit_sha256")
        == spec["source_ex48_data_audit_sha256"],
        "source_inventory": projection.get("source_inventory_sha256")
        == spec["source_ex48_inventory_sha256"],
        "fields": projection.get("fields") == list(ACCEPTED_DATA_PROJECTION_FIELDS),
        "counts": int(projection.get("train_shard_count", -1)) == 103
        and int(projection.get("validation_shard_count", -1)) == 1
        and len(train) == 103
        and len(validation) == 1,
        "normalized": structurally_projectable and normalized == inventory,
        "inventory_sha256": structurally_projectable
        and canonical_sha256(inventory) == spec["accepted_projection_inventory_sha256"],
    }
    return checks


def audit_data_dir(
    data_dir: Path,
    contract: dict[str, Any],
    scale: str,
    *,
    full_hash: bool = True,
    accepted_projection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert_contract(contract)
    if scale not in ("124m", "1b"):
        raise ValueError(scale)
    spec = contract["data"][scale]
    root = data_dir.absolute()
    train = sorted(root.glob("fineweb_train_*.bin"), key=shard_index)
    validation = sorted(root.glob("fineweb_val_*.bin"), key=shard_index)
    microbatch = int(contract["training"]["device_batch_size_1b"] if scale == "1b" else contract["training"]["device_batch_size_124m"]) * int(contract["training"]["sequence_length"])
    train_rows = [_read_shard(path, microbatch, full_hash) for path in train]
    val_rows = [_read_shard(path, microbatch, full_hash) for path in validation]
    indices = [row["index"] for row in train_rows]
    exact_count = spec.get("exact_train_shards")
    count_pass = len(train_rows) == int(exact_count) if exact_count is not None else len(train_rows) >= int(spec["minimum_train_shards"])
    total_consumable = sum(int(row["consumable_tokens"]) for row in train_rows)
    required_tokens = 3250585600 if scale == "124m" else 9999745024
    identity = {"train": train_rows, "validation": val_rows}
    checks = {
        "train_count": count_pass,
        "validation_count": len(val_rows) == 1,
        "first_index": bool(indices) and indices[0] == 1,
        "contiguous": indices == list(range(1, len(indices) + 1)),
        "unique": len(indices) == len(set(indices)),
        "capacity": total_consumable >= required_tokens + microbatch,
        "full_hash": (not full_hash) or all(row.get("sha256") for row in [*train_rows, *val_rows]),
    }
    accepted = spec.get("accepted_full_content_fingerprint_sha256")
    # EX52's accepted full-content certificate intentionally hashed the
    # validation shard first, followed by train shards 1..50.  Preserve that
    # byte-order for the 124M identity gate; the 1B fingerprint is run-local.
    fingerprint_rows = [*val_rows, *train_rows] if scale == "124m" else [*train_rows, *val_rows]
    content_fingerprint = ex52_content_fingerprint(fingerprint_rows)
    if accepted:
        checks["accepted_content"] = content_fingerprint == accepted
    accepted_projection_inventory_sha256 = None
    if scale == "1b":
        projection_checks = (
            validate_accepted_data_projection(accepted_projection, contract)
            if isinstance(accepted_projection, dict)
            else {"provided": False}
        )
        projected_identity = project_data_inventory(identity)
        accepted_inventory = (
            accepted_projection.get("inventory")
            if isinstance(accepted_projection, dict)
            else None
        )
        accepted_projection_inventory_sha256 = canonical_sha256(projected_identity)
        checks["accepted_projection_contract"] = bool(projection_checks) and all(
            projection_checks.values()
        )
        checks["accepted_projection_exact"] = (
            bool(full_hash)
            and projected_identity == accepted_inventory
            and accepted_projection_inventory_sha256
            == spec["accepted_projection_inventory_sha256"]
        )
    payload = {
        "schema_version": DATA_AUDIT_SCHEMA,
        "scale": scale,
        "data_dir": str(root),
        "full_hash": bool(full_hash),
        "inventory": identity,
        "inventory_sha256": canonical_sha256(identity),
        "content_fingerprint_sha256": content_fingerprint,
        "accepted_projection_inventory_sha256": accepted_projection_inventory_sha256,
        "train_shard_count": len(train_rows),
        "validation_shard_count": len(val_rows),
        "total_consumable_train_tokens": total_consumable,
        "required_stream_tokens": required_tokens + microbatch,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return payload


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
            checks[key] = (
                int(stat.st_size) == int(row["bytes"])
                and int(stat.st_mtime_ns) == int(row["mtime_ns"])
            )
    return checks


def cursor_after_batches(
    consumable_tokens: Iterable[int], microbatch_tokens: int, consumed_batches: int
) -> dict[str, int]:
    capacities = [int(value) // int(microbatch_tokens) for value in consumable_tokens]
    remaining = int(consumed_batches)
    if not capacities or remaining < 0:
        raise ValueError("invalid cursor inputs")
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
        "current_shard": len(capacities), "current_position": 0,
        "wrap_count": 1, "consumed_batches": int(consumed_batches),
    }


def expected_cursor(completed_steps: int, audit: dict[str, Any], contract: dict[str, Any]) -> dict[str, int]:
    batches = int(contract["data"]["prefetched_train_microbatches"]) + int(completed_steps) * int(contract["training"]["gradient_accumulation_steps_1b"])
    return cursor_after_batches(
        [row["consumable_tokens"] for row in audit["inventory"]["train"]],
        int(contract["training"]["device_batch_size_1b"]) * int(contract["training"]["sequence_length"]),
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
    start, target = int(phase["start_step"]), int(phase["target_step"])
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
        row for row in rows
        if int(row["step"]) < int(completed_steps)
        or (int(row["step"]) == int(completed_steps) and row["event"] == "train")
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(kept)


def append_metric(path: Path, row: dict[str, Any]) -> None:
    exists = path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if not rows:
        raise ValueError("mean of empty sequence")
    return sum(rows) / len(rows)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))
