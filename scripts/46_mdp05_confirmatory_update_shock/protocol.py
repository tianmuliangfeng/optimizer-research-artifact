#!/usr/bin/env python3
"""Pure-standard-library contract helpers for MDP-05.

This module is deliberately importable on a CPU-only login node.  CUDA and
scientific dependencies belong to the worker, never to controller preflight.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable, Sequence


SCHEMA = "mdp05_confirmatory_update_shock_v1"
EXECUTION_CONTRACT_VERSION = "2026-07-28.2"
ORIGINS = (
    "early_muon",
    "early_newton_full",
    "late_muon",
    "late_newton_full",
)
EVENTS = ("production_refresh_32", "delayed_refresh_64")
PRIMARY_MEDIATORS = (
    "matched_g_preconditioned_relative_change_layer_median",
    "runtime_ns5_update_relative_change_layer_median",
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
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(contract: dict[str, Any]) -> dict[str, bool]:
    design = contract.get("design", {})
    statistics = contract.get("statistics", {})
    gates = contract.get("hard_gates", {})
    checks = {
        "schema": contract.get("schema_version") == SCHEMA,
        "experiment": contract.get("experiment") == "MDP-05",
        "origins": tuple(design.get("origins", ())) == ORIGINS,
        "replicas": tuple(design.get("formal_data_replicas", ())) == (3, 4, 5),
        "new_training_offsets": tuple(
            design.get("formal_optimizer_step_offsets", ())
        )
        == (768, 1024, 1280),
        "rollout": int(design.get("rollout_steps", -1)) == 80,
        "events": tuple(
            row.get("event_id") for row in contract.get("event_outcomes", ())
        )
        == EVENTS,
        "layers": tuple(design.get("layer_indices", ())) == tuple(range(18)),
        "replica_unit": statistics.get("independent_unit")
        == "origin x held-out data replica",
        "nested_layers": statistics.get("layer_rows_are_nested") is True,
        "holm": str(statistics.get("multiplicity", "")).startswith("Holm"),
        "resolvent_not_gate": gates.get(
            "runtime_resolvent_relative_residual_is_hard_gate"
        )
        is False,
        "logs_excluded": gates.get("growing_worker_log_in_scientific_hashes")
        is False,
        "single_formal": contract.get("execution", {}).get(
            "one_formal_attempt"
        )
        is True,
        "no_old_outcomes": contract.get("source_experiment", {}).get(
            "outcomes_used"
        )
        is False,
    }
    return checks


def derive_execution_contract(
    source: dict[str, Any], protocol: dict[str, Any], protocol_sha256: str
) -> dict[str, Any]:
    """Create the worker-schema contract without mutating the accepted source."""
    checks = validate_protocol(protocol)
    if not all(checks.values()):
        raise RuntimeError(f"MDP-05 protocol validation failed: {checks}")
    if source.get("contract_version") != EXECUTION_CONTRACT_VERSION:
        raise RuntimeError("unexpected accepted MECH-09R contract version")
    if source.get("experiment") != "MECH-09R":
        raise RuntimeError("unexpected accepted execution contract identity")
    result = copy.deepcopy(source)
    design = protocol["design"]
    result["mdp05_derivation"] = {
        "outer_contract_schema": SCHEMA,
        "outer_contract_sha256": protocol_sha256,
        "source_outcomes_used": False,
        "source_checkpoint_certificates_only": True,
        "derived_before_formal_outcomes": True,
    }
    formal = result["formal"]
    formal.update(
        {
            "origins": list(design["origins"]),
            "data_replicas": list(design["formal_data_replicas"]),
            "replica_optimizer_step_offsets": list(
                design["formal_optimizer_step_offsets"]
            ),
            "rollout_steps": int(design["rollout_steps"]),
            "evaluation_steps": list(design["evaluation_steps"]),
            "production_refresh_interval": int(
                design["production_refresh_interval"]
            ),
            "expected_global_refresh_completed_steps": list(
                design["expected_global_refresh_completed_steps"]
            ),
            "causal_tree": copy.deepcopy(design["causal_tree"]),
            "build_token_offsets": list(design["formal_build_token_offsets"]),
            "eval_token_offsets": list(design["formal_eval_token_offsets"]),
        }
    )
    smoke = result["smoke"]
    smoke.update(
        {
            "origins": [design["smoke_origin"]],
            "data_replicas": [int(design["smoke_data_replica"])],
            "replica_optimizer_step_offsets": [
                int(design["smoke_optimizer_step_offset"])
            ],
            "build_token_offsets": [int(design["smoke_build_token_offset"])],
            "eval_token_offsets": [int(design["smoke_eval_token_offset"])],
        }
    )
    for arm, steps in design["arms"].items():
        result["arms"][arm]["formal_down_refresh_completed_steps"] = list(
            steps
        )
    result["stopping_rule"]["maximum_new_formal_jobs"] = 12
    result["stopping_rule"]["maximum_trajectories"] = 36
    return result


def interval(start: int, length: int, label: str) -> dict[str, Any]:
    return {"label": label, "start": int(start), "end": int(start) + int(length)}


def overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return max(int(left["start"]), int(right["start"])) < min(
        int(left["end"]), int(right["end"])
    )


def collision_pairs(rows: Sequence[dict[str, Any]]) -> list[list[str]]:
    return [
        [str(left["label"]), str(right["label"])]
        for left, right in itertools.combinations(rows, 2)
        if overlaps(left, right)
    ]


def peek_fineweb_tokens(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(12)
    if len(header) != 12:
        raise RuntimeError(f"short FineWeb header: {path}")
    magic, version, tokens = struct.unpack("<iii", header)
    if magic != 20240520 or version != 1 or tokens <= 0:
        raise RuntimeError(f"invalid FineWeb header: {path}")
    return int(tokens)


def map_global_interval(
    files: Sequence[Path], token_counts: Sequence[int], start: int, length: int
) -> dict[str, Any]:
    total = sum(int(value) for value in token_counts)
    if total <= 0:
        raise RuntimeError("empty validation corpus")
    wrapped = int(start) % total
    remaining = wrapped
    for index, (path, count) in enumerate(zip(files, token_counts)):
        if remaining < int(count):
            return {
                "global_start": int(start),
                "wrapped_start": wrapped,
                "length": int(length),
                "total_tokens": total,
                "shard_index": index,
                "shard": str(path.resolve()),
                "shard_position": remaining,
                "single_shard": remaining + int(length) + 1 <= int(count),
            }
        remaining -= int(count)
    raise RuntimeError("global interval mapping failed")


def build_offset_certificate(
    protocol: dict[str, Any], val_files: Sequence[Path] | None = None
) -> dict[str, Any]:
    design = protocol["design"]
    old_training = [
        interval(start, end - start, f"source_training_{index}")
        for index, (start, end) in enumerate(design["old_training_windows"])
    ]
    formal_training = [
        interval(start, design["rollout_steps"], f"mdp05_formal_training_{replica}")
        for replica, start in zip(
            design["formal_data_replicas"],
            design["formal_optimizer_step_offsets"],
        )
    ]
    smoke_training = [
        interval(
            design["smoke_optimizer_step_offset"],
            4,
            "mdp05_smoke_training",
        )
    ]
    build_length = 4 * 1 * 128
    eval_length = 16 * 8 * 1024
    old_validation = [
        *[
            interval(start, end - start, f"source_build_{index}")
            for index, (start, end) in enumerate(design["old_build_intervals"])
        ],
        *[
            interval(start, end - start, f"source_eval_{index}")
            for index, (start, end) in enumerate(design["old_eval_intervals"])
        ],
    ]
    new_validation = [
        *[
            interval(start, build_length, f"mdp05_formal_build_{replica}")
            for replica, start in zip(
                design["formal_data_replicas"],
                design["formal_build_token_offsets"],
            )
        ],
        *[
            interval(start, eval_length, f"mdp05_formal_eval_{replica}")
            for replica, start in zip(
                design["formal_data_replicas"],
                design["formal_eval_token_offsets"],
            )
        ],
        interval(
            design["smoke_build_token_offset"],
            128,
            "mdp05_smoke_build",
        ),
        interval(
            design["smoke_eval_token_offset"],
            128,
            "mdp05_smoke_eval",
        ),
    ]
    training_rows = [*old_training, *formal_training, *smoke_training]
    validation_rows = [*old_validation, *new_validation]
    mappings: list[dict[str, Any]] = []
    corpus_checks = {"provided": val_files is not None, "passed": True}
    if val_files is not None:
        paths = sorted(Path(path).resolve() for path in val_files)
        counts = [peek_fineweb_tokens(path) for path in paths]
        mappings = [
            {**row, **map_global_interval(paths, counts, row["start"], row["end"] - row["start"])}
            for row in validation_rows
        ]
        corpus_checks = {
            "provided": True,
            "file_count": len(paths),
            "total_tokens": sum(counts),
            "all_single_shard": all(row["single_shard"] for row in mappings),
            "no_wrapping": all(
                row["global_start"] == row["wrapped_start"] for row in mappings
            ),
        }
        corpus_checks["passed"] = all(
            value for key, value in corpus_checks.items() if key in {"all_single_shard", "no_wrapping"}
        )
    training_collisions = collision_pairs(training_rows)
    validation_collisions = collision_pairs(validation_rows)
    checks = {
        "training_windows_disjoint": not training_collisions,
        "validation_windows_disjoint": not validation_collisions,
        "formal_offsets_are_new": not set(
            design["formal_optimizer_step_offsets"]
        ).intersection({0, 256, 512}),
        "validation_corpus_mapping": corpus_checks["passed"],
    }
    return {
        "schema_version": "mdp05_offset_collision_certificate_v1",
        "training_intervals": training_rows,
        "validation_intervals": validation_rows,
        "training_collisions": training_collisions,
        "validation_collisions": validation_collisions,
        "validation_corpus": corpus_checks,
        "mapped_validation_intervals": mappings,
        "checks": checks,
        "passed": all(checks.values()),
    }


def rank_average(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation vectors must have equal length >=2")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    if left_ss <= 0.0 or right_ss <= 0.0:
        return float("nan")
    return numerator / math.sqrt(left_ss * right_ss)


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return pearson(rank_average(left), rank_average(right))


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty quantile")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def within_origin_centered(
    rows: Sequence[dict[str, Any]], x_field: str, y_field: str
) -> float:
    x_values: list[float] = []
    y_values: list[float] = []
    for origin in ORIGINS:
        group = [row for row in rows if row["origin"] == origin]
        x_mean = sum(float(row[x_field]) for row in group) / len(group)
        y_mean = sum(float(row[y_field]) for row in group) / len(group)
        x_values.extend(float(row[x_field]) - x_mean for row in group)
        y_values.extend(float(row[y_field]) - y_mean for row in group)
    return pearson(x_values, y_values)


def exact_within_origin_randomization_p(
    rows: Sequence[dict[str, Any]], x_field: str, y_field: str
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (ORIGINS.index(row["origin"]), int(row["data_replica"])))
    observed = spearman(
        [float(row[x_field]) for row in ordered],
        [float(row[y_field]) for row in ordered],
    )
    groups = [
        [row for row in ordered if row["origin"] == origin] for origin in ORIGINS
    ]
    if any(len(group) != 3 for group in groups):
        raise RuntimeError("exact randomization requires three replicas per origin")
    x = [float(row[x_field]) for row in ordered]
    exceed = 0
    total = 0
    for permutations in itertools.product(itertools.permutations(range(3)), repeat=4):
        permuted: list[float] = []
        for group, permutation in zip(groups, permutations):
            values = [float(row[y_field]) for row in group]
            permuted.extend(values[index] for index in permutation)
        statistic = spearman(x, permuted)
        if not math.isfinite(observed) or (
            math.isfinite(statistic) and statistic >= observed - 1.0e-15
        ):
            exceed += 1
        total += 1
    return {
        "observed_spearman_rho": observed,
        "permutations": total,
        "greater_or_equal": exceed,
        "one_sided_exact_p": exceed / total,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    ordered = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(ordered):
        candidate = (count - rank) * float(p_values[index])
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def all_finite(values: Iterable[Any]) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False
