#!/usr/bin/env python3
"""CPU-only protocol validation for experiment 47 / GEO-01."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


SCHEMA = "geo01_actual_update_direction_curvature_v1"
PHASES = ("pilot", "discovery", "confirmation")
SOURCE_EXECUTION_CONTRACT_VERSION = "2026-07-28.2"


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


def validate_contract(contract: dict[str, Any]) -> dict[str, bool]:
    pilot = contract.get("pilot", {})
    smoke = contract.get("remote_smoke", {})
    geometry = contract.get("geometry", {})
    boundaries = contract.get("claim_boundary", {})
    execution = contract.get("execution", {})
    target_layers = tuple(int(value) for value in pilot.get("target_layers", ()))
    scopes = pilot.get("scopes", ())
    scope_layers = sorted(
        {
            int(layer)
            for row in scopes
            for layer in row.get("layers", ())
        }
    )
    checks = {
        "schema": contract.get("schema_version") == SCHEMA,
        "experiment": contract.get("experiment") == "GEO-01",
        "experiment_number": int(contract.get("experiment_number", -1)) == 47,
        "exploratory": contract.get("evidence_tier") == "exploratory",
        "pilot_only": contract.get("current_phase") == "pilot",
        "pilot_single_origin": len(pilot.get("origins", ())) == 1,
        "pilot_single_replica": len(pilot.get("data_replicas", ())) == 1,
        "pilot_event": pilot.get("event_id") == "production_refresh_32",
        "source_contract_hash": contract.get("source_lineage", {}).get(
            "accepted_execution_contract_sha256"
        )
        == "f7917b229901383b9d45891ce67709d6c5f069c0f51b123498b9ef515914c700",
        "smoke_new_replica": int(smoke.get("data_replica", -1)) == 8,
        "smoke_outcome_blind": smoke.get(
            "checkpoint_or_geometry_outcomes_opened"
        )
        is False,
        "target_layers": target_layers == (0, 8, 17),
        "scope_coverage": scope_layers == list(target_layers),
        "joint_scope": any(
            row.get("scope_id") == "joint_0_8_17"
            and tuple(row.get("layers", ())) == target_layers
            for row in scopes
        ),
        "heldout_microbatch": int(geometry.get("microbatch_size", -1)) == 1
        and int(geometry.get("sequence_length", -1)) == 128,
        "direction": geometry.get("direction")
        == "source_pinned_refresh_minus_frozen_parameter_update",
        "exact_hvp": geometry.get("exact_directional_hvp") is True,
        "math_attention_hvp": geometry.get("hvp_attention_backend")
        == "math_only",
        "actual_multiplier": float(geometry.get("actual_update_multiplier", -1))
        == 1.0,
        "no_full_hessian": geometry.get("construct_full_hessian") is False,
        "no_tensor_persistence": execution.get("persist_full_direction") is False
        and execution.get("persist_full_hessian") is False,
        "mdp05_closed": boundaries.get("mdp05_result_remains_closed") is True,
        "not_confirmatory": boundaries.get("pilot_claim_eligible") is False,
        "ten_b_deferred": boundaries.get("llama_10b_triggered") is False,
        "confirmation_disabled": execution.get("confirmation_enabled") is False,
    }
    return checks


def derive_execution_contract(
    source: dict[str, Any], contract: dict[str, Any], contract_sha256: str
) -> dict[str, Any]:
    checks = validate_contract(contract)
    if not all(checks.values()):
        raise RuntimeError(f"GEO-01 contract validation failed: {checks}")
    if source.get("contract_version") != SOURCE_EXECUTION_CONTRACT_VERSION:
        raise RuntimeError("unexpected accepted MECH-09R contract version")
    if source.get("experiment") != "MECH-09R":
        raise RuntimeError("unexpected accepted source experiment")
    result = copy.deepcopy(source)
    pilot = contract["pilot"]
    smoke = contract["remote_smoke"]
    result["geo01_derivation"] = {
        "outer_contract_schema": SCHEMA,
        "outer_contract_sha256": contract_sha256,
        "phase": "pilot",
        "source_outcomes_used": False,
        "scientific_outcomes_may_be_opened_for_selection": False,
    }
    formal = result["formal"]
    formal.update(
        {
            "origins": list(pilot["origins"]),
            "data_replicas": list(pilot["data_replicas"]),
            "replica_optimizer_step_offsets": list(
                pilot["optimizer_step_offsets"]
            ),
            "build_token_offsets": list(pilot["build_token_offsets"]),
            "eval_token_offsets": list(pilot["eval_token_offsets"]),
        }
    )
    smoke_config = result["smoke"]
    smoke_config.update(
        {
            "origins": [smoke["origin"]],
            "data_replicas": [int(smoke["data_replica"])],
            "replica_optimizer_step_offsets": [
                int(smoke["optimizer_step_offset"])
            ],
            "build_token_offsets": [int(smoke["build_token_offset"])],
            "eval_token_offsets": [int(smoke["eval_token_offset"])],
        }
    )
    # MECH-09R's accepted worker hard-audits this source-contract safety cap.
    # GEO-01 schedules only one pilot unit in its outer pilot plan; changing the
    # inherited cap would invalidate the source worker before any computation.
    result["stopping_rule"]["maximum_new_formal_jobs"] = 12
    result["stopping_rule"]["maximum_trajectories"] = 3
    compatibility = validate_derived_execution_contract(result, contract)
    if not all(compatibility.values()):
        raise RuntimeError(
            f"derived MECH-09R execution contract is incompatible: {compatibility}"
        )
    return result


def validate_derived_execution_contract(
    derived: dict[str, Any], outer: dict[str, Any]
) -> dict[str, bool]:
    """Mirror source-worker invariants that GEO-01 is allowed to specialize."""

    pilot = outer["pilot"]
    smoke = outer["remote_smoke"]
    return {
        "source_contract_version": derived.get("contract_version")
        == SOURCE_EXECUTION_CONTRACT_VERSION,
        "source_experiment": derived.get("experiment") == "MECH-09R",
        "formal_origins": derived.get("formal", {}).get("origins")
        == list(pilot["origins"]),
        "formal_replicas": derived.get("formal", {}).get("data_replicas")
        == list(pilot["data_replicas"]),
        "smoke_origins": derived.get("smoke", {}).get("origins")
        == [smoke["origin"]],
        "smoke_replicas": derived.get("smoke", {}).get("data_replicas")
        == [int(smoke["data_replica"])],
        "source_formal_cap_preserved": int(
            derived.get("stopping_rule", {}).get("maximum_new_formal_jobs", -1)
        )
        == 12,
        "outer_scheduler_is_single_unit": len(pilot["origins"])
        * len(pilot["data_replicas"])
        == 1,
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
    pilot = contract["pilot"]
    smoke = contract["remote_smoke"]
    training = [
        *[
            _interval(start, 128, f"accepted_mech09r_training_{index}")
            for index, start in enumerate((0, 256, 512))
        ],
        *[
            _interval(start, 80, f"accepted_mdp05_training_{index}")
            for index, start in enumerate((768, 1024, 1280))
        ],
        _interval(1536, 4, "accepted_mdp05_smoke"),
        _interval(
            int(pilot["optimizer_step_offsets"][0]),
            80,
            "geo01_pilot_training",
        ),
        _interval(
            int(smoke["optimizer_step_offset"]),
            int(smoke["rollout_steps"]),
            "geo01_smoke_training",
        ),
    ]
    build_length = 4 * 1 * 128
    eval_length = 16 * 8 * 1024
    validation = [
        *[
            _interval(start, build_length, f"accepted_source_build_{index}")
            for index, start in enumerate(
                (0, 2_000_000, 4_000_000, 6_000_000, 8_000_000, 10_000_000, 12_000_000)
            )
        ],
        *[
            _interval(start, eval_length, f"accepted_source_eval_{index}")
            for index, start in enumerate(
                (
                    1_048_576,
                    3_048_576,
                    5_048_576,
                    7_048_576,
                    9_048_576,
                    11_048_576,
                    13_048_576,
                )
            )
        ],
        _interval(int(pilot["build_token_offsets"][0]), build_length, "geo01_build"),
        _interval(int(pilot["eval_token_offsets"][0]), eval_length, "geo01_eval"),
        _interval(int(smoke["build_token_offset"]), 128, "geo01_smoke_build"),
        _interval(int(smoke["eval_token_offset"]), 128, "geo01_smoke_eval"),
    ]
    training_collisions = _collisions(training)
    validation_collisions = _collisions(validation)
    checks = {
        "training_disjoint": not training_collisions,
        "validation_disjoint": not validation_collisions,
        "pilot_replica_new": int(pilot["data_replicas"][0]) not in (0, 1, 2, 3, 4, 5, 6),
        "smoke_replica_new": int(smoke["data_replica"]) not in (0, 1, 2, 3, 4, 5, 6, 7),
    }
    return {
        "schema_version": "geo01_offset_collision_certificate_v1",
        "training_intervals": training,
        "validation_intervals": validation,
        "training_collisions": training_collisions,
        "validation_collisions": validation_collisions,
        "checks": checks,
        "passed": all(checks.values()),
    }


def validation_payload(contract_path: Path) -> dict[str, Any]:
    contract = read_json(contract_path)
    checks = validate_contract(contract)
    return {
        "schema_version": "geo01_contract_validation_v1",
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256_file(contract_path),
        "checks": checks,
        "passed": all(checks.values()),
    }
