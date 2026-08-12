#!/usr/bin/env python3
"""Pure-stdlib planning kernels for the LLaMA-1B 10B feasibility package."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any, Iterable


SCHEMA = "llama1b_10b_feasibility_v1"
MAGIC = 20240520
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4


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


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nearest_update_half_up(target_tokens: int, tokens_per_update: int) -> int:
    if target_tokens <= 0 or tokens_per_update <= 0:
        raise ValueError("token counts must be positive")
    return (int(target_tokens) + int(tokens_per_update) // 2) // int(
        tokens_per_update
    )


def validate_contract(contract: dict[str, Any]) -> dict[str, bool]:
    profile = contract.get("profile", {})
    data = contract.get("data", {})
    lr = contract.get("lr_schedule", {})
    geometry = contract.get("geometry_interface", {})
    gate = contract.get("gate", {})
    methods = profile.get("methods", [])
    milestones = contract.get("milestones", [])
    schedule = build_schedule(contract) if len(milestones) == 3 else []
    return {
        "schema": contract.get("schema_version") == SCHEMA,
        "planning_only": contract.get("status") == "planning_only_not_launchable",
        "launch_disabled": contract.get("launch_authorized") is False
        and contract.get("remote_training_command_allowed") is False,
        "no_experiment_number": contract.get("experiment_number_assigned") is False,
        "methods": methods
        == ["down_none", "down_diag", "newton_full", "muon"],
        "single_seed_screen": int(profile.get("seed", -1)) == 2026,
        "batch_geometry": int(profile.get("global_batch_size", -1)) == 512
        and int(profile.get("device_batch_size", -1)) == 8
        and int(profile.get("sequence_length", -1)) == 1024
        and int(profile.get("tokens_per_update", -1)) == 524288
        and int(profile.get("gradient_accumulation_steps", -1)) == 64,
        "milestone_steps": [row["step"] for row in schedule]
        == [6200, 13293, 19073],
        "approximately_10b": schedule
        and abs(schedule[-1]["actual_tokens"] - 10_000_000_000) < 524288,
        "data_no_wrap": int(data.get("minimum_contiguous_train_shards", -1))
        >= 101
        and int(data.get("wrap_count_required", -1)) == 0,
        "lr_unresolved": lr.get("status") == "unresolved_hard_blocker"
        and lr.get("long_horizon_policy_selected") is False,
        "no_checkpoint_continuation": lr.get("forbidden_shortcut")
        == "continue_from_the_zero_lr_6200_checkpoint",
        "geometry_disabled": geometry.get("enabled") is False
        and geometry.get("metric_formula_selected") is False
        and geometry.get("layers_selected") is False,
        "new_contract_required": gate.get("requires_new_launch_contract") is True,
        "idle_gpu_not_trigger": gate.get("gpu_idleness_is_not_a_trigger") is True,
    }


def build_schedule(contract: dict[str, Any]) -> list[dict[str, Any]]:
    profile = contract["profile"]
    unit = int(profile["tokens_per_update"])
    parameters = int(profile["parameters"])
    rows = []
    for spec in contract["milestones"]:
        target = int(spec["target_tokens"])
        if spec["step_rule"] == "exact":
            if target % unit:
                raise ValueError(f"exact milestone is off update grid: {spec['id']}")
            step = target // unit
        elif spec["step_rule"] == "nearest_integer_update_half_up":
            step = nearest_update_half_up(target, unit)
        else:
            raise ValueError(f"unsupported step rule: {spec['step_rule']}")
        if step != int(spec["expected_step"]):
            raise RuntimeError(f"frozen milestone step drift: {spec['id']}")
        actual = step * unit
        rows.append(
            {
                "id": spec["id"],
                "step_rule": spec["step_rule"],
                "target_tokens": target,
                "step": step,
                "actual_tokens": actual,
                "token_error": actual - target,
                "relative_token_error": (actual - target) / target,
                "tokens_per_parameter": actual / parameters,
                "train_microbatches": step
                * int(profile["gradient_accumulation_steps"]),
                "stream_microbatches_including_prefetch": step
                * int(profile["gradient_accumulation_steps"])
                + int(profile["prefetched_train_microbatches"]),
                "stream_tokens_including_prefetch": actual
                + int(profile["prefetched_train_microbatches"])
                * int(profile["microbatch_tokens"]),
                "interpretation": spec["interpretation"],
            }
        )
    return rows


def validation_steps(contract: dict[str, Any]) -> list[int]:
    final = build_schedule(contract)[-1]["step"]
    every = int(contract["validation"]["regular_every_steps"])
    regular = set(range(0, final + 1, every))
    regular.add(final)
    regular.update(int(value) for value in contract["validation"]["forced_milestone_steps"])
    return sorted(regular)


def lpt_wall_seconds(job_seconds: dict[str, float], gpu_count: int) -> dict[str, Any]:
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    loads = [0.0 for _ in range(gpu_count)]
    assignments: list[list[str]] = [[] for _ in range(gpu_count)]
    for method, seconds in sorted(job_seconds.items(), key=lambda row: (-row[1], row[0])):
        index = min(range(gpu_count), key=lambda item: (loads[item], item))
        loads[index] += float(seconds)
        assignments[index].append(method)
    return {
        "gpu_count": gpu_count,
        "assignments": assignments,
        "gpu_load_seconds": loads,
        "wall_seconds": max(loads, default=0.0),
    }


def build_budget(contract: dict[str, Any]) -> dict[str, Any]:
    steps = build_schedule(contract)[-1]["step"]
    overhead = float(contract["budget"]["wall_overhead_factor"])
    raw_jobs = {
        method: steps * float(seconds)
        for method, seconds in contract["budget"]["empirical_step_seconds"].items()
    }
    adjusted_jobs = {method: seconds * overhead for method, seconds in raw_jobs.items()}
    gpu_scenarios = {
        str(count): lpt_wall_seconds(adjusted_jobs, count) for count in (1, 2, 4)
    }
    checkpoints = {
        key: int(value) for key, value in contract["budget"]["checkpoint_bytes"].items()
    }
    copies = int(contract["budget"]["retained_milestone_checkpoints_per_method"])
    retained = sum(checkpoints.values()) * copies
    headroom = math.ceil(retained * float(contract["budget"]["disk_headroom_factor"]))
    operational_target = int(
        contract["budget"].get("operational_free_disk_target_bytes", headroom)
    )
    methods = len(contract["profile"]["methods"])
    return {
        "endpoint_steps": steps,
        "training_tokens_per_method": steps
        * int(contract["profile"]["tokens_per_update"]),
        "aggregate_training_tokens": steps
        * int(contract["profile"]["tokens_per_update"])
        * methods,
        "validation_events_per_method": len(validation_steps(contract)),
        "validation_tokens_per_method": len(validation_steps(contract))
        * int(contract["validation"]["tokens_per_evaluation"]),
        "raw_training_gpu_hours": sum(raw_jobs.values()) / 3600.0,
        "adjusted_job_seconds": adjusted_jobs,
        "gpu_scenarios": gpu_scenarios,
        "retained_checkpoint_bytes": retained,
        "checkpoint_headroom_bytes": headroom,
        "recommended_minimum_free_disk_bytes": max(headroom, operational_target),
        "checkpoint_bytes_by_method": checkpoints,
    }


def shard_index(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.bin$)", path.name)
    if match is None:
        raise ValueError(f"shard name has no numeric suffix: {path.name}")
    return int(match.group(1))


def read_shard_header(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(int(contract["data"]["header_bytes"]))
    if len(header) != int(contract["data"]["header_bytes"]):
        raise RuntimeError(f"short FineWeb header: {path}")
    magic, version, tokens = struct.unpack("<iii", header[:12])
    if magic != int(contract["data"]["header_magic"]):
        raise RuntimeError(f"FineWeb magic mismatch: {path}")
    if version != int(contract["data"]["header_version"]):
        raise RuntimeError(f"FineWeb version mismatch: {path}")
    expected_bytes = int(contract["data"]["header_bytes"]) + int(tokens) * int(
        contract["data"]["token_dtype_bytes"]
    )
    return {
        "name": path.name,
        "index": shard_index(path),
        "tokens": int(tokens),
        "bytes": path.stat().st_size,
        "expected_bytes": expected_bytes,
        "size_exact": path.stat().st_size == expected_bytes,
        "header_sha256": hashlib.sha256(header).hexdigest(),
    }


def audit_data_dir(data_dir: Path, contract: dict[str, Any]) -> dict[str, Any]:
    root = data_dir.absolute()
    train = sorted(root.glob(contract["data"]["train_pattern"]), key=shard_index)
    validation = sorted(root.glob(contract["data"]["validation_pattern"]), key=shard_index)
    train_rows = [read_shard_header(path, contract) for path in train]
    validation_rows = [read_shard_header(path, contract) for path in validation]
    indices = [row["index"] for row in train_rows]
    contiguous = bool(indices) and indices == list(range(indices[0], indices[0] + len(indices)))
    microbatch = int(contract["profile"]["microbatch_tokens"])
    for row in train_rows:
        row["consumable_tokens"] = ((row["tokens"] - 1) // microbatch) * microbatch
    required_training = build_schedule(contract)[-1]["actual_tokens"]
    required_stream = required_training + int(
        contract["profile"]["prefetched_train_microbatches"]
    ) * microbatch
    total_consumable = sum(row["consumable_tokens"] for row in train_rows)
    checks = {
        "minimum_train_shards": len(train_rows)
        >= int(contract["data"]["minimum_contiguous_train_shards"]),
        "validation_shards": len(validation_rows)
        == int(contract["data"]["required_validation_shards"]),
        "numeric_indices_unique": len(indices) == len(set(indices)),
        "numeric_indices_contiguous": contiguous,
        "file_sizes_exact": all(row["size_exact"] for row in [*train_rows, *validation_rows]),
        "no_wrap_capacity": total_consumable >= required_stream,
    }
    inventory = {
        "train": train_rows,
        "validation": validation_rows,
    }
    return {
        "schema_version": "llama1b_10b_data_feasibility_v1",
        "data_dir": str(root),
        "required_training_tokens": required_training,
        "prefetch_tokens": required_stream - required_training,
        "required_stream_tokens": required_stream,
        "total_consumable_train_tokens": total_consumable,
        "unused_consumable_tokens": total_consumable - required_stream,
        "train_shard_count": len(train_rows),
        "validation_shard_count": len(validation_rows),
        "first_train_index": indices[0] if indices else None,
        "last_train_index": indices[-1] if indices else None,
        "inventory_sha256": canonical_sha256(inventory),
        "inventory": inventory,
        "checks": checks,
        "passed": all(checks.values()),
        "launch_authorized": False,
    }


def cursor_after_batches(
    consumable_tokens: Iterable[int], microbatch_tokens: int, consumed_batches: int
) -> dict[str, int]:
    capacities = [int(value) // int(microbatch_tokens) for value in consumable_tokens]
    if not capacities or any(value <= 0 for value in capacities):
        raise ValueError("every shard must contain a consumable microbatch")
    remaining = int(consumed_batches)
    if remaining < 0:
        raise ValueError("consumed_batches must be non-negative")
    wraps = 0
    while True:
        for shard, capacity in enumerate(capacities):
            if remaining < capacity:
                return {
                    "current_shard": shard,
                    "current_position": remaining * int(microbatch_tokens),
                    "wrap_count": wraps,
                    "consumed_batches": int(consumed_batches),
                }
            remaining -= capacity
        wraps += 1


def expected_resume_cursor(
    completed_steps: int, consumable_tokens: Iterable[int], contract: dict[str, Any]
) -> dict[str, int]:
    profile = contract["profile"]
    batches = int(profile["prefetched_train_microbatches"]) + int(
        completed_steps
    ) * int(profile["gradient_accumulation_steps"])
    row = cursor_after_batches(
        consumable_tokens, int(profile["microbatch_tokens"]), batches
    )
    row["completed_steps"] = int(completed_steps)
    row["logical_training_tokens"] = int(completed_steps) * int(
        profile["tokens_per_update"]
    )
    row["stream_tokens_including_prefetch"] = batches * int(
        profile["microbatch_tokens"]
    )
    return row


def audit_current_sources(repo: Path) -> dict[str, Any]:
    runner = repo / "scripts/20_llama_swiglu_1b/run_llama_swiglu_1b.py"
    base_runner = repo / "scripts/17_llama_swiglu_validation/run_llama_swiglu_validation.py"
    trainer = repo / "scripts/17_llama_swiglu_validation/train_llama_swiglu.py"
    paths = {"runner": runner, "base_runner": base_runner, "trainer": trainer}
    missing = {key: not path.is_file() for key, path in paths.items()}
    if any(missing.values()):
        raise FileNotFoundError(f"LLaMA source inventory incomplete: {missing}")
    runner_text = runner.read_text(encoding="utf-8")
    base_text = base_runner.read_text(encoding="utf-8")
    trainer_text = trainer.read_text(encoding="utf-8")
    checks = {
        "profile_1b": '"expected_parameter_count": 1_013_690_368' in runner_text,
        "current_formal_fixed_6200": '"formal": 6200' in runner_text,
        "current_data_audit_exact_50": "len(train) != 50" in base_text,
        "current_loader_modulo_wrap": "% len(self.files)" in trainer_text,
        "current_loader_has_wrap_count": "wrap_count" in trainer_text,
        "forced_validation_grid_supported": "forced_validation_steps" in trainer_text,
        "checkpoint_train_loader": '"train_loader": train_loader.state_dict()' in trainer_text,
        "checkpoint_prefetched_batch": '"next_x": x.detach().cpu()' in trainer_text
        and '"next_y": y.detach().cpu()' in trainer_text,
        "checkpoint_rng": '"rng": capture_rng_state()' in trainer_text,
    }
    blockers = [
        name
        for name, blocked in {
            "formal_steps_fixed_to_6200": checks["current_formal_fixed_6200"],
            "data_audit_requires_exactly_50_shards": checks["current_data_audit_exact_50"],
            "loader_uses_modulo_wrap": checks["current_loader_modulo_wrap"],
            "loader_has_no_wrap_count": not checks["current_loader_has_wrap_count"],
            "forced_validation_grid_missing": not checks["forced_validation_grid_supported"],
        }.items()
        if blocked
    ]
    return {
        "schema_version": "llama1b_10b_source_feasibility_v1",
        "files": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "checks": checks,
        "blockers": blockers,
        "resume_payload_supportive": checks["checkpoint_train_loader"]
        and checks["checkpoint_prefetched_batch"]
        and checks["checkpoint_rng"],
        "launch_ready": False,
    }


def build_report(
    contract: dict[str, Any], source_audit: dict[str, Any], data_audit: dict[str, Any] | None
) -> dict[str, Any]:
    contract_checks = validate_contract(contract)
    hard_blockers = list(source_audit["blockers"])
    if contract["lr_schedule"]["status"] != "resolved_frozen":
        hard_blockers.append("long_horizon_lr_schedule_unresolved")
    if data_audit is None:
        hard_blockers.append("remote_data_inventory_not_audited")
    elif not data_audit["passed"]:
        hard_blockers.append("remote_data_inventory_failed")
    hard_blockers = sorted(set(hard_blockers))
    return {
        "schema_version": "llama1b_10b_feasibility_report_v1",
        "contract_sha256": canonical_sha256(contract),
        "contract_checks": contract_checks,
        "schedule": build_schedule(contract),
        "validation_steps": validation_steps(contract),
        "budget": build_budget(contract),
        "source_audit": source_audit,
        "data_audit": data_audit,
        "hard_blockers": hard_blockers,
        "technical_prerequisites_passed": not hard_blockers,
        "launch_authorized": False,
        "scientific_evidence_class": "none_planning_only",
        "next_decision": "Resolve blockers, review GEO-01 pilot, run paper red-team gate, then write a new launch contract only if explicitly authorized.",
    }
