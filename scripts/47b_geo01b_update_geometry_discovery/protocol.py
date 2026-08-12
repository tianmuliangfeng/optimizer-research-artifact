#!/usr/bin/env python3
"""CPU-only frozen-contract helpers for experiment 47 / GEO-01B."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "geo01b_directional_geometry_discovery_v1"
SOURCE_EXECUTION_CONTRACT_VERSION = "2026-07-28.2"
ORIGINS = ("early_muon", "early_newton_full", "late_muon", "late_newton_full")
REPLICAS = (9, 10, 11)
EVENTS = ("production_refresh_32", "delayed_refresh_64")
SCOPES = ("all_down", "layers_0_5", "layers_6_11", "layers_12_17")


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract(contract: dict[str, Any]) -> dict[str, bool]:
    discovery = contract.get("discovery", {})
    geometry = contract.get("geometry", {})
    predictors = contract.get("predictors", {})
    boundary = contract.get("claim_boundary", {})
    execution = contract.get("execution", {})
    smoke = contract.get("remote_smoke", {})
    scopes = discovery.get("scopes", [])
    events = discovery.get("events", [])
    target_layers = tuple(int(value) for value in discovery.get("target_layers", []))
    unit_count = len(discovery.get("origins", [])) * len(discovery.get("data_replicas", []))
    return {
        "schema": contract.get("schema_version") == SCHEMA,
        "status": contract.get("status")
        == "frozen_after_accepted_geo01a_before_gpu_discovery",
        "experiment": contract.get("experiment") == "GEO-01B",
        "experiment_number": int(contract.get("experiment_number", -1)) == 47,
        "phase": contract.get("current_phase") == "discovery",
        "exploratory": contract.get("evidence_tier")
        == "exploratory_hypothesis_generating",
        "accepted_pilot": contract.get("geo01a_pilot_lineage", {}).get(
            "engineering_integrity_passed"
        )
        is True
        and contract.get("geo01a_pilot_lineage", {}).get("scientific_claim_eligible")
        is False,
        "outcome_blind_formula": contract.get("source_lineage", {}).get(
            "mdp05_outcomes_used_to_select_formula"
        )
        is False,
        "source_contract_hash": contract.get("source_lineage", {}).get(
            "accepted_execution_contract_sha256"
        )
        == "f7917b229901383b9d45891ce67709d6c5f069c0f51b123498b9ef515914c700",
        "origins": tuple(discovery.get("origins", [])) == ORIGINS,
        "replicas": tuple(int(value) for value in discovery.get("data_replicas", []))
        == REPLICAS,
        "unit_count": unit_count == 12
        and int(discovery.get("expected_units", -1)) == 12,
        "offset_lengths": all(
            len(discovery.get(field, [])) == len(REPLICAS)
            for field in (
                "optimizer_step_offsets",
                "build_token_offsets",
                "eval_token_offsets",
            )
        ),
        "events": tuple(row.get("event_id") for row in events) == EVENTS
        and tuple(int(row.get("completed_step", -1)) for row in events) == (32, 64)
        and tuple(int(row.get("endpoint_step", -1)) for row in events) == (48, 80),
        "all_layers": target_layers == tuple(range(18)),
        "scopes": tuple(row.get("scope_id") for row in scopes) == SCOPES
        and next(
            (tuple(row.get("layers", [])) for row in scopes if row.get("scope_id") == "all_down"),
            (),
        )
        == tuple(range(18)),
        "primary_scope": discovery.get("primary_scope") == "all_down",
        "formula": geometry.get("predictor_formula")
        == "<g_val,d> + 0.5*<d,H_val d>",
        "actual_direction": geometry.get("direction")
        == "source_pinned_refresh_minus_frozen_parameter_update"
        and geometry.get("same_raw_gradient") is True
        and geometry.get("same_historical_momentum") is True,
        "exact_hvp": geometry.get("exact_directional_hvp") is True
        and geometry.get("construct_full_hessian") is False
        and geometry.get("hvp_attention_backend") == "math_only",
        "geometry_batches": int(geometry.get("microbatch_size", -1)) == 1
        and int(geometry.get("sequence_length", -1)) == 128
        and int(geometry.get("heldout_microbatches", -1)) == 4,
        "predictor_family": predictors.get("norm_only")
        == "relative_direction_fro_norm"
        and predictors.get("first_order") == "first_order_alignment"
        and predictors.get("full_taylor") == "taylor_actual_delta_loss",
        "new_smoke": int(smoke.get("data_replica", -1)) == 12
        and smoke.get("outcomes_opened") is False,
        "claim_boundary": boundary.get("discovery_claim_eligible") is False
        and boundary.get("mdp05_result_remains_closed") is True
        and boundary.get("confirmation_requires_new_contract") is True
        and boundary.get("llama_10b_triggered") is False,
        "execution": execution.get("discovery_enabled") is True
        and execution.get("confirmation_enabled") is False
        and execution.get("persist_full_direction") is False
        and execution.get("persist_full_hessian") is False
        and int(execution.get("maximum_parallel_jobs", -1)) == 2,
    }


def derive_execution_contract(
    source: dict[str, Any], contract: dict[str, Any], contract_sha256: str
) -> dict[str, Any]:
    checks = validate_contract(contract)
    if not all(checks.values()):
        raise RuntimeError(f"GEO-01B contract validation failed: {checks}")
    if source.get("contract_version") != SOURCE_EXECUTION_CONTRACT_VERSION:
        raise RuntimeError("unexpected accepted MECH-09R contract version")
    if source.get("experiment") != "MECH-09R":
        raise RuntimeError("unexpected accepted source experiment")
    result = copy.deepcopy(source)
    discovery = contract["discovery"]
    smoke = contract["remote_smoke"]
    result["geo01b_derivation"] = {
        "outer_contract_schema": SCHEMA,
        "outer_contract_sha256": contract_sha256,
        "phase": "discovery",
        "source_outcomes_used": False,
        "discovery_claim_eligible": False,
    }
    result["formal"].update(
        {
            "origins": list(discovery["origins"]),
            "data_replicas": list(discovery["data_replicas"]),
            "replica_optimizer_step_offsets": list(
                discovery["optimizer_step_offsets"]
            ),
            "build_token_offsets": list(discovery["build_token_offsets"]),
            "eval_token_offsets": list(discovery["eval_token_offsets"]),
        }
    )
    result["smoke"].update(
        {
            "origins": [smoke["origin"]],
            "data_replicas": [int(smoke["data_replica"])],
            "replica_optimizer_step_offsets": [int(smoke["optimizer_step_offset"])],
            "build_token_offsets": [int(smoke["build_token_offset"])],
            "eval_token_offsets": [int(smoke["eval_token_offset"])],
        }
    )
    # The accepted worker hard-audits these values. The outer controller seals
    # exactly 12 origin-by-replica jobs and never treats event/layer rows as jobs.
    result["stopping_rule"]["maximum_new_formal_jobs"] = 12
    result["stopping_rule"]["maximum_trajectories"] = 3
    compatibility = validate_derived_execution_contract(result, contract)
    if not all(compatibility.values()):
        raise RuntimeError(f"derived execution contract failed: {compatibility}")
    return result


def validate_derived_execution_contract(
    derived: dict[str, Any], outer: dict[str, Any]
) -> dict[str, bool]:
    discovery = outer["discovery"]
    smoke = outer["remote_smoke"]
    return {
        "source_contract_version": derived.get("contract_version")
        == SOURCE_EXECUTION_CONTRACT_VERSION,
        "source_experiment": derived.get("experiment") == "MECH-09R",
        "formal_origins": derived.get("formal", {}).get("origins")
        == list(discovery["origins"]),
        "formal_replicas": derived.get("formal", {}).get("data_replicas")
        == list(discovery["data_replicas"]),
        "formal_optimizer_offsets": derived.get("formal", {}).get(
            "replica_optimizer_step_offsets"
        )
        == list(discovery["optimizer_step_offsets"]),
        "smoke_origin": derived.get("smoke", {}).get("origins")
        == [smoke["origin"]],
        "smoke_replica": derived.get("smoke", {}).get("data_replicas")
        == [int(smoke["data_replica"])],
        "source_formal_cap": int(
            derived.get("stopping_rule", {}).get("maximum_new_formal_jobs", -1)
        )
        == 12,
        "outer_units": len(discovery["origins"])
        * len(discovery["data_replicas"])
        == 12,
        "trajectory_cap": int(
            derived.get("stopping_rule", {}).get("maximum_trajectories", -1)
        )
        == 3,
    }


def _interval(start: int, length: int, label: str) -> dict[str, Any]:
    return {"label": label, "start": int(start), "end": int(start) + int(length)}


def _collisions(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [str(left["label"]), str(right["label"])]
        for left, right in itertools.combinations(rows, 2)
        if max(int(left["start"]), int(right["start"]))
        < min(int(left["end"]), int(right["end"]))
    ]


def build_offset_certificate(contract: dict[str, Any]) -> dict[str, Any]:
    discovery = contract["discovery"]
    smoke = contract["remote_smoke"]
    training = [
        *[_interval(value, 128, f"accepted_mech09r_training_{index}") for index, value in enumerate((0, 256, 512))],
        *[_interval(value, 80, f"accepted_mdp05_training_{index}") for index, value in enumerate((768, 1024, 1280))],
        _interval(1536, 4, "accepted_mdp05_smoke"),
        _interval(1792, 128, "accepted_geo01a_pilot"),
        _interval(2048, 4, "accepted_geo01a_smoke"),
        *[
            _interval(value, 128, f"geo01b_discovery_replica_{replica}")
            for replica, value in zip(
                discovery["data_replicas"], discovery["optimizer_step_offsets"]
            )
        ],
        _interval(int(smoke["optimizer_step_offset"]), int(smoke["rollout_steps"]), "geo01b_smoke"),
    ]
    build_length = 4 * 1 * 128
    eval_length = 16 * 8 * 1024
    validation = [
        *[_interval(value, build_length, f"accepted_source_build_{index}") for index, value in enumerate((0, 2_000_000, 4_000_000, 6_000_000, 8_000_000, 10_000_000, 12_000_000))],
        *[_interval(value, eval_length, f"accepted_source_eval_{index}") for index, value in enumerate((1_048_576, 3_048_576, 5_048_576, 7_048_576, 9_048_576, 11_048_576, 13_048_576))],
        _interval(14_000_000, build_length, "accepted_geo01a_build"),
        _interval(15_048_576, eval_length, "accepted_geo01a_eval"),
        _interval(16_000_000, 128, "accepted_geo01a_smoke_build"),
        _interval(17_048_576, 128, "accepted_geo01a_smoke_eval"),
        *[
            _interval(value, build_length, f"geo01b_build_replica_{replica}")
            for replica, value in zip(
                discovery["data_replicas"], discovery["build_token_offsets"]
            )
        ],
        *[
            _interval(value, eval_length, f"geo01b_eval_replica_{replica}")
            for replica, value in zip(
                discovery["data_replicas"], discovery["eval_token_offsets"]
            )
        ],
        _interval(int(smoke["build_token_offset"]), 128, "geo01b_smoke_build"),
        _interval(int(smoke["eval_token_offset"]), 128, "geo01b_smoke_eval"),
    ]
    training_collisions = _collisions(training)
    validation_collisions = _collisions(validation)
    checks = {
        "training_disjoint": not training_collisions,
        "validation_disjoint": not validation_collisions,
        "discovery_replicas_new": not set(discovery["data_replicas"]).intersection(
            range(0, 9)
        ),
        "smoke_replica_new": int(smoke["data_replica"]) not in range(0, 12),
        "discovery_units": len(discovery["origins"])
        * len(discovery["data_replicas"])
        == 12,
    }
    return {
        "schema_version": "geo01b_offset_collision_certificate_v1",
        "training_intervals": training,
        "validation_intervals": validation,
        "training_collisions": training_collisions,
        "validation_collisions": validation_collisions,
        "checks": checks,
        "passed": all(checks.values()),
    }


def job_matrix(contract: dict[str, Any]) -> list[dict[str, Any]]:
    discovery = contract["discovery"]
    return [
        {
            "origin": origin,
            "data_replica": int(replica),
            "label": f"discovery/{origin}/replica_{replica}",
        }
        for origin in discovery["origins"]
        for replica in discovery["data_replicas"]
    ]


def ranks(values: Iterable[float]) -> list[float]:
    rows = sorted(enumerate(float(value) for value in values), key=lambda row: row[1])
    result = [0.0] * len(rows)
    index = 0
    while index < len(rows):
        end = index + 1
        while end < len(rows) and rows[end][1] == rows[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for offset in range(index, end):
            result[rows[offset][0]] = rank
        index = end
    return result


def pearson(left: Iterable[float], right: Iterable[float]) -> float:
    x = [float(value) for value in left]
    y = [float(value) for value in right]
    if len(x) != len(y) or len(x) < 2:
        return math.nan
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denominator == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def spearman(left: Iterable[float], right: Iterable[float]) -> float:
    return pearson(ranks(left), ranks(right))


def centered(values: list[float], groups: list[str]) -> list[float]:
    means = {
        group: sum(value for value, observed in zip(values, groups) if observed == group)
        / sum(1 for observed in groups if observed == group)
        for group in set(groups)
    }
    return [value - means[group] for value, group in zip(values, groups)]


def median(values: Iterable[float]) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        return math.nan
    middle = len(rows) // 2
    return rows[middle] if len(rows) % 2 else (rows[middle - 1] + rows[middle]) / 2.0


def validation_payload(contract_path: Path) -> dict[str, Any]:
    contract = read_json(contract_path)
    checks = validate_contract(contract)
    offsets = build_offset_certificate(contract)
    return {
        "schema_version": "geo01b_contract_validation_v1",
        "contract": str(contract_path.absolute()),
        "contract_sha256": sha256_file(contract_path),
        "checks": checks,
        "offset_certificate": offsets,
        "passed": all(checks.values()) and offsets["passed"],
    }
