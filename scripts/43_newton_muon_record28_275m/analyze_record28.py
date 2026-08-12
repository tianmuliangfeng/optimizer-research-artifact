#!/usr/bin/env python3
"""Independent, paired-seed analysis for experiment 43 (Record #28 scale).

The local run directory is authoritative.  This analyzer never reads W&B and
never treats validation points as independent replicates.  It recomputes every
quality endpoint from each accepted attempt's ``metrics.csv`` before forming
the pre-registered paired-seed contrasts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

import record28_common as C


SCRIPT_VERSION = "2026-07-30.2"
SCHEMA_VERSION = 1
EXPECTED_SEEDS = (2024, 2025, 2026, 2027)
EXPECTED_METHODS = (
    "muon",
    "original_newton_muon",
    "selective_none",
    "selective_diag",
)
EXPECTED_TOTAL_STEPS = 1695
EXPECTED_TOKENS_PER_UPDATE = 393_216
EXPECTED_TRAIN_TOKENS = 666_501_120
EXPECTED_VALIDATION_STEPS = tuple(range(0, 1651, 50)) + (1695,)
COMMON_TARGET_LOSS = 3.30
PRACTICAL_MARGIN = 0.002
T_CRITICAL_95_DF3 = 3.182446305284263
FLOAT_TOLERANCE = 1e-10
INITIAL_VALIDATION_TOLERANCE = 1e-6

QUALITY_METRICS = (
    "final_val_loss",
    "normalized_auc",
    "tail5_mean",
    "best_val_loss",
    "steps_to_target",
    "tokens_to_target",
)

# Delta is always candidate minus comparator.  Lower is better for every
# metric in this table.
CONTRASTS = (
    ("primary", "selective_none_vs_muon", "selective_none", "muon"),
    (
        "primary",
        "selective_none_vs_original",
        "selective_none",
        "original_newton_muon",
    ),
    ("primary", "selective_diag_vs_muon", "selective_diag", "muon"),
    (
        "primary",
        "selective_diag_vs_original",
        "selective_diag",
        "original_newton_muon",
    ),
    (
        "benchmark_anchor",
        "original_vs_muon",
        "original_newton_muon",
        "muon",
    ),
    (
        "secondary_selective",
        "diag_vs_none",
        "selective_diag",
        "selective_none",
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the 16 accepted experiment-43 formal cells."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to RUN_DIR/analysis.",
    )
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_exact_int(value: Any, expected: int, label: str) -> int:
    try:
        observed = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not an integer: {value!r}") from error
    if observed != expected:
        raise RuntimeError(f"{label} mismatch: expected {expected}, got {observed}")
    return observed


def _parse_nonnegative_int(value: Any, label: str) -> int:
    try:
        observed = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not an integer: {value!r}") from error
    if observed < 0:
        raise RuntimeError(f"{label} must be non-negative: {observed}")
    return observed


def _require_finite(value: Any, label: str) -> float:
    try:
        observed = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(observed):
        raise RuntimeError(f"{label} is not finite: {observed!r}")
    return observed


def _contract_value(
    contract: dict[str, Any], paths: Iterable[tuple[str, ...]], default: Any
) -> Any:
    for path in paths:
        current: Any = contract
        for key in path:
            if not isinstance(current, dict) or key not in current:
                break
            current = current[key]
        else:
            return current
    return default


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema_version") != 1:
        raise RuntimeError("record28 contract schema_version must be 1")
    seeds = tuple(
        int(value)
        for value in _contract_value(
            contract,
            (
                ("seeds",),
                ("formal", "seeds"),
                ("design", "seeds"),
                ("paired_design", "formal_seeds"),
            ),
            EXPECTED_SEEDS,
        )
    )
    methods = tuple(
        str(value)
        for value in _contract_value(
            contract,
            (
                ("methods", "order"),
                ("formal", "methods"),
                ("design", "methods"),
                ("methods",),
            ),
            EXPECTED_METHODS,
        )
    )
    total_steps = int(
        _contract_value(
            contract,
            (
                ("total_steps",),
                ("training", "total_steps"),
                ("recipe", "iterations"),
                ("training_recipe", "formal_updates"),
            ),
            EXPECTED_TOTAL_STEPS,
        )
    )
    tokens_per_update = int(
        _contract_value(
            contract,
            (
                ("tokens_per_update",),
                ("training", "tokens_per_update"),
                ("recipe", "tokens_per_update"),
                ("training_recipe", "tokens_per_update"),
            ),
            EXPECTED_TOKENS_PER_UPDATE,
        )
    )
    train_tokens = int(
        _contract_value(
            contract,
            (
                ("train_tokens",),
                ("training", "train_tokens"),
                ("training_recipe", "exact_training_tokens"),
            ),
            total_steps * tokens_per_update,
        )
    )
    target = float(
        _contract_value(
            contract,
            (
                ("common_target_loss",),
                ("analysis", "common_target_loss"),
                ("statistics", "common_target_loss"),
                ("analysis_contract", "common_target_validation_loss"),
            ),
            COMMON_TARGET_LOSS,
        )
    )
    margin = float(
        _contract_value(
            contract,
            (
                ("practical_margin",),
                ("practical_loss_margin",),
                ("analysis", "practical_margin"),
                ("statistics", "practical_margin"),
                ("analysis_contract", "practical_final_loss_margin"),
            ),
            PRACTICAL_MARGIN,
        )
    )
    checks = {
        "seeds": seeds == EXPECTED_SEEDS,
        "methods": methods == EXPECTED_METHODS,
        "total_steps": total_steps == EXPECTED_TOTAL_STEPS,
        "tokens_per_update": tokens_per_update == EXPECTED_TOKENS_PER_UPDATE,
        "train_tokens": train_tokens == EXPECTED_TRAIN_TOKENS,
        "common_target_loss": math.isclose(
            target, COMMON_TARGET_LOSS, rel_tol=0.0, abs_tol=1e-12
        ),
        "practical_margin": math.isclose(
            margin, PRACTICAL_MARGIN, rel_tol=0.0, abs_tol=1e-12
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen experiment-43 contract mismatch: {checks}")
    return {
        "checks": checks,
        "seeds": seeds,
        "methods": methods,
        "total_steps": total_steps,
        "tokens_per_update": tokens_per_update,
        "train_tokens": train_tokens,
        "common_target_loss": target,
        "practical_margin": margin,
    }


def resolve_accepted_attempt(cell_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    accepted_path = cell_dir / "accepted.json"
    if not accepted_path.is_file():
        raise RuntimeError(f"accepted pointer is missing: {accepted_path}")
    accepted = C.read_json(accepted_path)
    if not isinstance(accepted, dict):
        raise RuntimeError(f"accepted pointer must be an object: {accepted_path}")
    raw_attempt = accepted.get("attempt_dir")
    if not isinstance(raw_attempt, str) or not raw_attempt.strip():
        raise RuntimeError(f"accepted.json lacks a non-empty attempt_dir: {accepted_path}")
    attempt = Path(raw_attempt)
    if not attempt.is_absolute():
        attempt = cell_dir / attempt
    attempt = attempt.resolve()
    cell_resolved = cell_dir.resolve()
    try:
        attempt.relative_to(cell_resolved)
    except ValueError as error:
        raise RuntimeError(
            f"accepted attempt escapes its formal cell: {attempt} not under {cell_resolved}"
        ) from error
    if not attempt.is_dir():
        raise RuntimeError(f"accepted attempt directory is missing: {attempt}")
    return accepted_path.resolve(), attempt, accepted


def collect_accepted_cell_fingerprints(
    run_dir: Path,
) -> list[dict[str, Any]]:
    """Return the canonical 16-cell input identity used by an analysis.

    The digest deliberately binds both the accepted pointer and the pointed-to
    scientific manifest.  A recovery controller can therefore distinguish a
    reusable analysis from a stale analysis without reading any statistical
    output.
    """

    run_dir = run_dir.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        for method in EXPECTED_METHODS:
            cell_dir = run_dir / "formal" / f"seed{seed}" / method
            accepted_path, attempt, accepted = resolve_accepted_attempt(cell_dir)
            manifest_path = attempt / "scientific_manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"accepted scientific manifest is missing: {manifest_path}"
                )
            manifest_sha256 = C.sha256_file(manifest_path)
            cell_key = C.cell_key("formal", seed, method)
            checks = {
                "pointer_cell_key": accepted.get("cell_key") == cell_key,
                "pointer_manifest_sha256": accepted.get(
                    "scientific_manifest_sha256"
                )
                == manifest_sha256,
            }
            if not all(checks.values()):
                raise RuntimeError(
                    f"accepted fingerprint integrity failed at {accepted_path}: "
                    f"{checks}"
                )
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "cell_key": cell_key,
                    "accepted_pointer_sha256": C.sha256_file(accepted_path),
                    "scientific_manifest_sha256": manifest_sha256,
                }
            )
    if len(rows) != 16:
        raise RuntimeError(f"expected 16 accepted fingerprints, got {len(rows)}")
    return rows


def accepted_cells_fingerprint_sha256(
    rows: Sequence[dict[str, Any]],
) -> str:
    payload = json.dumps(
        list(rows),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return C.sha256_bytes(payload)


def validation_rows(path: Path) -> list[dict[str, Any]]:
    raw_rows = _read_csv(path)
    if not raw_rows:
        raise RuntimeError(f"metrics CSV is empty: {path}")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows, start=2):
        if raw.get("val_loss", "").strip() == "":
            continue
        step = _parse_nonnegative_int(raw.get("step"), f"{path}:{index}:step")
        total_steps = _require_exact_int(
            raw.get("total_steps"), EXPECTED_TOTAL_STEPS, f"{path}:{index}:total_steps"
        )
        loss = _require_finite(raw.get("val_loss"), f"{path}:{index}:val_loss")
        if loss <= 0:
            raise RuntimeError(f"{path}:{index}:val_loss must be positive")
        tokens = _require_exact_int(
            raw.get("tokens"), step * EXPECTED_TOKENS_PER_UPDATE, f"{path}:{index}:tokens"
        )
        rows.append(
            {
                "step": step,
                "total_steps": total_steps,
                "val_loss": loss,
                "tokens": tokens,
            }
        )
    steps = tuple(row["step"] for row in rows)
    if steps != EXPECTED_VALIDATION_STEPS:
        raise RuntimeError(
            f"validation-step grid mismatch in {path}: "
            f"expected {EXPECTED_VALIDATION_STEPS}, got {steps}"
        )
    if len(set(steps)) != len(steps):
        raise RuntimeError(f"duplicate validation steps in {path}")
    return rows


def target_crossing(
    rows: Sequence[dict[str, Any]], target: float, tokens_per_update: int
) -> tuple[float | None, float | None, int | None]:
    for index, row in enumerate(rows):
        current_loss = float(row["val_loss"])
        if current_loss > target:
            continue
        current_step = int(row["step"])
        if index == 0:
            interpolated_step = float(current_step)
        else:
            previous = rows[index - 1]
            previous_loss = float(previous["val_loss"])
            previous_step = int(previous["step"])
            if previous_loss <= target or math.isclose(
                previous_loss, current_loss, rel_tol=0.0, abs_tol=1e-15
            ):
                interpolated_step = float(current_step)
            else:
                fraction = (previous_loss - target) / (previous_loss - current_loss)
                fraction = min(1.0, max(0.0, fraction))
                interpolated_step = previous_step + fraction * (
                    current_step - previous_step
                )
        return (
            interpolated_step,
            interpolated_step * tokens_per_update,
            current_step,
        )
    return None, None, None


def recompute_metrics(
    rows: Sequence[dict[str, Any]], target: float, tokens_per_update: int
) -> dict[str, Any]:
    if len(rows) < 5:
        raise RuntimeError("at least five validation rows are required")
    start = int(rows[0]["step"])
    end = int(rows[-1]["step"])
    if start != 0 or end != EXPECTED_TOTAL_STEPS:
        raise RuntimeError(f"invalid AUC range: {start}..{end}")
    area = 0.0
    for left, right in zip(rows, rows[1:]):
        width = int(right["step"]) - int(left["step"])
        if width <= 0:
            raise RuntimeError("validation steps must be strictly increasing")
        area += width * (float(left["val_loss"]) + float(right["val_loss"])) / 2.0
    steps_to_target, tokens_to_target, observed_step = target_crossing(
        rows, target, tokens_per_update
    )
    return {
        "initial_val_loss": float(rows[0]["val_loss"]),
        "final_val_loss": float(rows[-1]["val_loss"]),
        "best_val_loss": min(float(row["val_loss"]) for row in rows),
        "tail5_mean": statistics.mean(
            float(row["val_loss"]) for row in rows[-5:]
        ),
        "normalized_auc": area / (end - start),
        "steps_to_target": steps_to_target,
        "tokens_to_target": tokens_to_target,
        "first_observed_step_at_or_below_target": observed_step,
        "validation_rows": len(rows),
    }


def _compare_optional_number(
    observed: Any, expected: float | None, label: str
) -> bool:
    if expected is None:
        if observed is not None:
            raise RuntimeError(f"{label} must be null because target was not reached")
        return True
    value = _require_finite(observed, label)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE):
        raise RuntimeError(f"{label} mismatch: expected {expected}, got {value}")
    return True


def validate_summary(
    summary: dict[str, Any],
    computed: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    for key in (
        "final_val_loss",
        "best_val_loss",
        "tail5_mean",
        "normalized_auc",
    ):
        observed = _require_finite(summary.get(key), f"{label}:{key}")
        if not math.isclose(
            observed, float(computed[key]), rel_tol=0.0, abs_tol=FLOAT_TOLERANCE
        ):
            raise RuntimeError(
                f"{label}:{key} mismatch: recomputed {computed[key]}, summary {observed}"
            )
    _compare_optional_number(
        summary.get("steps_to_target"), computed["steps_to_target"], f"{label}:steps_to_target"
    )
    _compare_optional_number(
        summary.get("tokens_to_target"),
        computed["tokens_to_target"],
        f"{label}:tokens_to_target",
    )
    _require_exact_int(summary.get("final_step"), EXPECTED_TOTAL_STEPS, f"{label}:final_step")
    _require_exact_int(
        summary.get("tokens_per_update"),
        EXPECTED_TOKENS_PER_UPDATE,
        f"{label}:tokens_per_update",
    )
    _require_exact_int(
        summary.get("train_tokens"), EXPECTED_TRAIN_TOKENS, f"{label}:train_tokens"
    )
    state: dict[str, int] = {}
    for key in (
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
        "k_state_bytes",
        "optimizer_state_bytes",
    ):
        value = int(summary.get(key, -1))
        if value < 0:
            raise RuntimeError(f"{label}:{key} must be a non-negative integer")
        state[key] = value
    if state["peak_memory_reserved_bytes"] < state["peak_memory_allocated_bytes"]:
        raise RuntimeError(f"{label}: reserved peak is below allocated peak")
    return state


def load_cell(
    run_dir: Path,
    contract_sha256: str,
    seed: int,
    method: str,
    target: float,
    tokens_per_update: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cell_dir = run_dir / "formal" / f"seed{seed}" / method
    accepted_path, attempt, accepted = resolve_accepted_attempt(cell_dir)
    manifest_path = attempt / "scientific_manifest.json"
    metrics_path = attempt / "metrics.csv"
    summary_path = attempt / "summary.json"
    command_path = attempt / "command.json"
    for path in (manifest_path, metrics_path, summary_path, command_path):
        if not path.is_file():
            raise RuntimeError(f"required accepted artifact is missing: {path}")
    manifest = C.read_json(manifest_path)
    summary = C.read_json(summary_path)
    command = C.read_json(command_path)
    if not all(isinstance(value, dict) for value in (manifest, summary, command)):
        raise RuntimeError(
            f"{attempt}: manifest, summary, and command must be JSON objects"
        )
    accepted_integrity_checks = {
        "cell_key": accepted.get("cell_key")
        == C.cell_key("formal", seed, method),
        "scientific_manifest_sha256": accepted.get(
            "scientific_manifest_sha256"
        )
        == C.sha256_file(manifest_path),
    }
    if not all(accepted_integrity_checks.values()):
        raise RuntimeError(
            f"{accepted_path}: accepted pointer integrity failed: "
            f"{accepted_integrity_checks}"
        )

    expected_cproj_mode = {
        "muon": "not_applicable",
        "original_newton_muon": "block4",
        "selective_none": "none",
        "selective_diag": "diag",
    }[method]
    command_environment = command.get("environment")
    if not isinstance(command_environment, dict):
        raise RuntimeError(f"{command_path}: environment object is missing")
    command_environment_checks = {
        "method": command_environment.get("RECORD28_METHOD") == method,
        "cproj_k_mode": (
            command_environment.get("RECORD28_CPROJ_K_MODE")
            == expected_cproj_mode
        ),
    }
    if not all(command_environment_checks.values()):
        raise RuntimeError(
            f"{command_path}: method environment mismatch: "
            f"{command_environment_checks}"
        )

    identity_checks = {
        "passed": manifest.get("passed") is True,
        "status": manifest.get("status") == "scientifically_complete",
        "stage": manifest.get("stage") == "formal",
        "seed": manifest.get("seed") == seed,
        "method": manifest.get("method") == method,
        "cproj_k_mode": manifest.get("cproj_k_mode") == expected_cproj_mode,
        "cell_key": manifest.get("cell_key") == C.cell_key("formal", seed, method),
        "contract_sha256": manifest.get("contract_sha256") == contract_sha256,
        "timing_ineligible": manifest.get("timing_eligible") is False,
        "total_steps": manifest.get("total_steps") == EXPECTED_TOTAL_STEPS,
        "train_tokens": manifest.get("train_tokens") == EXPECTED_TRAIN_TOKENS,
    }
    if not all(identity_checks.values()):
        raise RuntimeError(
            f"accepted manifest identity failed for seed={seed} method={method}: "
            f"{identity_checks}"
        )
    required_hash_fields = (
        "source_snapshot_sha256",
        "derived_source_sha256",
        "data_fingerprint_sha256",
        "init_sha256",
    )
    for key in required_hash_fields:
        value = manifest.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"{manifest_path}:{key} is not a SHA-256 hex digest")
        try:
            bytes.fromhex(value)
        except ValueError as error:
            raise RuntimeError(f"{manifest_path}:{key} is not hexadecimal") from error

    rows = validation_rows(metrics_path)
    computed = recompute_metrics(rows, target, tokens_per_update)
    state = validate_summary(
        summary, computed, label=f"seed{seed}/{method}/summary.json"
    )
    artifact_hashes = manifest.get(
        "artifact_hashes",
        manifest.get("artifact_sha256", manifest.get("hashes")),
    )
    if not isinstance(artifact_hashes, dict):
        raise RuntimeError(f"{manifest_path}: artifact hashes are missing")
    # The scientific manifest cannot self-hash, but it must bind the two inputs
    # used by this independent recomputation.
    for name, path in (
        ("command.json", command_path),
        ("metrics.csv", metrics_path),
        ("summary.json", summary_path),
    ):
        expected_hash = artifact_hashes.get(name)
        observed_hash = C.sha256_file(path)
        if expected_hash != observed_hash:
            raise RuntimeError(
                f"{manifest_path}: hash mismatch for {name}: "
                f"expected {expected_hash}, observed {observed_hash}"
            )

    cell_row = {
        "seed": seed,
        "method": method,
        "cproj_k_mode": expected_cproj_mode,
        **computed,
        **state,
        "timing_eligible": False,
        "contract_sha256": contract_sha256,
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "derived_source_sha256": manifest["derived_source_sha256"],
        "data_fingerprint_sha256": manifest["data_fingerprint_sha256"],
        "init_sha256": manifest["init_sha256"],
        "accepted_pointer": str(accepted_path),
        "accepted_attempt": str(attempt),
        "scientific_manifest": str(manifest_path),
    }
    audit = {
        "seed": seed,
        "method": method,
        "accepted_pointer": str(accepted_path),
        "accepted_pointer_sha256": C.sha256_file(accepted_path),
        "accepted_payload": accepted,
        "attempt_dir": str(attempt),
        "scientific_manifest_sha256": C.sha256_file(manifest_path),
        "metrics_sha256": C.sha256_file(metrics_path),
        "summary_sha256": C.sha256_file(summary_path),
        "command_sha256": C.sha256_file(command_path),
        "command_environment_checks": command_environment_checks,
        "identity_checks": identity_checks,
        "summary_recomputed": True,
    }
    return cell_row, audit


def validate_pairing(
    rows: Sequence[dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    by_key = {(int(row["seed"]), str(row["method"])): row for row in rows}
    expected = {
        (seed, method) for seed in EXPECTED_SEEDS for method in EXPECTED_METHODS
    }
    observed_pointers = {
        path.resolve()
        for path in (run_dir / "formal").glob("seed*/*/accepted.json")
        if path.is_file()
    }
    expected_pointers = {
        (run_dir / "formal" / f"seed{seed}" / method / "accepted.json").resolve()
        for seed, method in expected
    }
    checks: dict[str, Any] = {
        "formal_cell_coverage_16_of_16": set(by_key) == expected and len(rows) == 16,
        "no_unregistered_accepted_pointers": observed_pointers == expected_pointers,
        "one_source_snapshot": len(
            {row["source_snapshot_sha256"] for row in rows}
        )
        == 1,
        "one_data_fingerprint": len(
            {row["data_fingerprint_sha256"] for row in rows}
        )
        == 1,
        "init_paired_within_seed": {},
        "derived_source_constant_within_method": {},
        "derived_source_routing": False,
        "init_distinct_across_seeds": False,
        "initial_validation_paired_within_seed": {},
        "initial_validation_max_abs_delta_by_seed": {},
        "k_state_ordering_within_seed": {},
        "attempts_unique": len({row["accepted_attempt"] for row in rows}) == 16,
        "timing_ineligible_all_cells": all(
            row["timing_eligible"] is False for row in rows
        ),
    }
    for seed in EXPECTED_SEEDS:
        hashes = {by_key[(seed, method)]["init_sha256"] for method in EXPECTED_METHODS}
        checks["init_paired_within_seed"][str(seed)] = len(hashes) == 1
        initial_values = [
            float(by_key[(seed, method)]["initial_val_loss"])
            for method in EXPECTED_METHODS
        ]
        initial_delta = max(initial_values) - min(initial_values)
        checks["initial_validation_max_abs_delta_by_seed"][str(seed)] = (
            initial_delta
        )
        checks["initial_validation_paired_within_seed"][str(seed)] = (
            initial_delta <= INITIAL_VALIDATION_TOLERANCE
        )
        k_values = {
            method: int(by_key[(seed, method)]["k_state_bytes"])
            for method in EXPECTED_METHODS
        }
        checks["k_state_ordering_within_seed"][str(seed)] = (
            k_values["muon"] == 0
            and k_values["original_newton_muon"] > k_values["selective_diag"]
            and k_values["selective_diag"] > k_values["selective_none"]
            and k_values["selective_none"] > 0
        )
    for method in EXPECTED_METHODS:
        hashes = {
            by_key[(seed, method)]["derived_source_sha256"]
            for seed in EXPECTED_SEEDS
        }
        checks["derived_source_constant_within_method"][method] = len(hashes) == 1
    representative_sources = {
        method: by_key[(EXPECTED_SEEDS[0], method)]["derived_source_sha256"]
        for method in EXPECTED_METHODS
    }
    newton_source_hashes = {
        representative_sources[method]
        for method in (
            "original_newton_muon",
            "selective_none",
            "selective_diag",
        )
    }
    checks["derived_source_routing"] = (
        len(newton_source_hashes) == 1
        and representative_sources["muon"] not in newton_source_hashes
        and len(set(representative_sources.values())) == 2
    )
    checks["init_distinct_across_seeds"] = (
        len(
            {
                by_key[(seed, EXPECTED_METHODS[0])]["init_sha256"]
                for seed in EXPECTED_SEEDS
            }
        )
        == len(EXPECTED_SEEDS)
    )
    flat = [
        checks["formal_cell_coverage_16_of_16"],
        checks["no_unregistered_accepted_pointers"],
        checks["one_source_snapshot"],
        checks["one_data_fingerprint"],
        checks["attempts_unique"],
        checks["timing_ineligible_all_cells"],
        checks["derived_source_routing"],
        checks["init_distinct_across_seeds"],
        *checks["init_paired_within_seed"].values(),
        *checks["initial_validation_paired_within_seed"].values(),
        *checks["k_state_ordering_within_seed"].values(),
        *checks["derived_source_constant_within_method"].values(),
    ]
    checks["passed"] = all(flat)
    if not checks["passed"]:
        raise RuntimeError(f"formal pairing/source/data audit failed: {checks}")
    return checks


def t_summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != 4:
        raise RuntimeError(f"four paired seeds are required, got {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("paired values must all be finite")
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    half_width = T_CRITICAL_95_DF3 * sample_sd / math.sqrt(4)
    return {
        "n_seeds": 4,
        "mean": mean,
        "sample_sd": sample_sd,
        "ci95_low_t_df3": mean - half_width,
        "ci95_high_t_df3": mean + half_width,
        "negative_seeds": sum(value < 0 for value in values),
        "positive_seeds": sum(value > 0 for value in values),
        "zero_seeds": sum(value == 0 for value in values),
    }


def method_summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in EXPECTED_METHODS:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 4:
            raise RuntimeError(f"{method}: expected four rows")
        for metric in QUALITY_METRICS:
            values = [row[metric] for row in selected]
            if any(value is None for value in values):
                output.append(
                    {
                        "method": method,
                        "metric": metric,
                        "n_seeds": 4,
                        "eligible_seeds": sum(value is not None for value in values),
                        "mean": None,
                        "sample_sd": None,
                        "ci95_low_t_df3": None,
                        "ci95_high_t_df3": None,
                        "complete": False,
                    }
                )
                continue
            stats = t_summary([float(value) for value in values])
            output.append(
                {
                    "method": method,
                    "metric": metric,
                    "eligible_seeds": 4,
                    **stats,
                    "complete": True,
                }
            )
        for metric in (
            "peak_memory_allocated_bytes",
            "peak_memory_reserved_bytes",
            "k_state_bytes",
            "optimizer_state_bytes",
        ):
            values = [float(row[metric]) for row in selected]
            stats = t_summary(values)
            output.append(
                {
                    "method": method,
                    "metric": metric,
                    "eligible_seeds": 4,
                    **stats,
                    "complete": True,
                }
            )
    return output


def practical_classification(
    low: float, high: float, mean: float, margin: float
) -> str:
    if high < -margin:
        return "candidate_better_beyond_margin"
    if low > margin:
        return "candidate_worse_beyond_margin"
    if low >= -margin and high <= margin:
        return "quality_equivalent_within_margin"
    if low < -margin and high > margin:
        return "ci_spans_both_practical_boundaries"
    if mean < 0:
        return "direction_candidate_better_but_practically_unresolved"
    if mean > 0:
        return "direction_candidate_worse_but_practically_unresolved"
    return "centered_zero_but_practically_unresolved"


def paired_contrasts(
    rows: Sequence[dict[str, Any]], margin: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {(int(row["seed"]), str(row["method"])): row for row in rows}
    raw: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for family, name, candidate, comparator in CONTRASTS:
        for metric in QUALITY_METRICS:
            deltas: list[float] = []
            metric_rows: list[dict[str, Any]] = []
            for seed in EXPECTED_SEEDS:
                candidate_value = by_key[(seed, candidate)][metric]
                comparator_value = by_key[(seed, comparator)][metric]
                eligible = candidate_value is not None and comparator_value is not None
                delta = (
                    float(candidate_value) - float(comparator_value)
                    if eligible
                    else None
                )
                metric_rows.append(
                    {
                        "family": family,
                        "contrast": name,
                        "metric": metric,
                        "seed": seed,
                        "candidate": candidate,
                        "comparator": comparator,
                        "candidate_value": candidate_value,
                        "comparator_value": comparator_value,
                        "delta_candidate_minus_comparator": delta,
                        "eligible": eligible,
                    }
                )
                if delta is not None:
                    deltas.append(delta)
            raw.extend(metric_rows)
            if len(deltas) != 4:
                summaries.append(
                    {
                        "family": family,
                        "contrast": name,
                        "metric": metric,
                        "candidate": candidate,
                        "comparator": comparator,
                        "n_seeds": 4,
                        "eligible_seeds": len(deltas),
                        "mean": None,
                        "sample_sd": None,
                        "ci95_low_t_df3": None,
                        "ci95_high_t_df3": None,
                        "negative_seeds": None,
                        "positive_seeds": None,
                        "zero_seeds": None,
                        "practical_margin": margin if metric == "final_val_loss" else None,
                        "classification": "target_not_reached_by_all_paired_cells",
                        "lower_is_better": True,
                    }
                )
                continue
            stats = t_summary(deltas)
            classification = (
                practical_classification(
                    float(stats["ci95_low_t_df3"]),
                    float(stats["ci95_high_t_df3"]),
                    float(stats["mean"]),
                    margin,
                )
                if metric == "final_val_loss"
                else "secondary_metric_no_practical_margin"
            )
            summaries.append(
                {
                    "family": family,
                    "contrast": name,
                    "metric": metric,
                    "candidate": candidate,
                    "comparator": comparator,
                    "eligible_seeds": 4,
                    **stats,
                    "practical_margin": (
                        margin if metric == "final_val_loss" else None
                    ),
                    "classification": classification,
                    "lower_is_better": True,
                }
            )
    return raw, summaries


def build_decision(
    contrast_rows: Sequence[dict[str, Any]], margin: float
) -> dict[str, Any]:
    primary = [
        row
        for row in contrast_rows
        if row["family"] == "primary" and row["metric"] == "final_val_loss"
    ]
    if len(primary) != 4:
        raise RuntimeError("four primary final-loss contrasts are required")
    ambiguity = [
        row["contrast"]
        for row in primary
        if row["classification"] == "ci_spans_both_practical_boundaries"
    ]
    classifications = {
        row["contrast"]: row["classification"] for row in primary
    }
    return {
        "primary_endpoint": "final_val_loss",
        "delta_convention": "candidate_minus_comparator; negative is better",
        "common_target_crossing_definition": (
            "first downward crossing of validation loss 3.30, linearly "
            "interpolated between adjacent validation checkpoints"
        ),
        "practical_margin": margin,
        "primary_classifications": classifications,
        "statistical_seed_append_gate_triggered": bool(ambiguity),
        "ambiguous_primary_contrasts": ambiguity,
        "seed_append_allowed_automatically": False,
        "seed_append_rule": (
            "A statistical trigger is necessary but not sufficient.  Additional "
            "seeds are allowed only if this uncertainty would change the frozen "
            "cross-scale paper claim; if allowed, run all four methods together "
            "on the next unused seeds (2028 then 2029), at most two."
        ),
        "diag_vs_none_is_primary": False,
        "original_vs_muon_is_mandatory_benchmark_anchor": True,
        "timing_claim_eligible": False,
    }


def build_report(
    cells: Sequence[dict[str, Any]],
    contrasts: Sequence[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    final_means = {
        method: statistics.mean(
            float(row["final_val_loss"]) for row in cells if row["method"] == method
        )
        for method in EXPECTED_METHODS
    }
    primary = [
        row
        for row in contrasts
        if row["metric"] == "final_val_loss"
        and row["family"] in {"primary", "benchmark_anchor", "secondary_selective"}
    ]
    lines = [
        "# Experiment 43: Record #28-scale paired analysis",
        "",
        "All 16 formal cells passed local integrity checks. The seed is the "
        "inferential unit; validation checkpoints are not treated as replicates.",
        "",
        "## Mean final validation loss",
        "",
    ]
    for method in EXPECTED_METHODS:
        lines.append(f"- `{method}`: {final_means[method]:.6f}")
    lines.extend(
        [
            "",
            "## Paired final-loss contrasts",
            "",
            "Deltas are candidate minus comparator, so negative is better.",
            "",
        ]
    )
    for row in primary:
        lines.append(
            f"- `{row['contrast']}`: {row['mean']:+.6f} "
            f"(95% t CI {row['ci95_low_t_df3']:+.6f}, "
            f"{row['ci95_high_t_df3']:+.6f}); {row['classification']}."
        )
    lines.extend(
        [
            "",
            "## Statistical boundary",
            "",
            f"- Frozen practical margin: ±{decision['practical_margin']:.3f}.",
            "- `diag_vs_none` is secondary and cannot drive seed expansion.",
            "- Original Newton–Muon versus Muon is a mandatory benchmark anchor.",
            "- Concurrent quality runs are timing-ineligible; no throughput or "
            "wall-clock claim is derived here.",
            f"- Statistical append trigger: "
            f"{decision['statistical_seed_append_gate_triggered']}.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    run_dir: Path, contract_path: Path, output_dir: Path
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    contract_path = contract_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise RuntimeError(f"run directory does not exist: {run_dir}")
    if not contract_path.is_file():
        raise RuntimeError(f"contract does not exist: {contract_path}")
    contract = C.read_json(contract_path)
    if not isinstance(contract, dict):
        raise RuntimeError("contract must be a JSON object")
    frozen = validate_contract(contract)
    contract_sha256 = C.sha256_file(contract_path)

    cells: list[dict[str, Any]] = []
    source_audits: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        for method in EXPECTED_METHODS:
            cell, audit = load_cell(
                run_dir,
                contract_sha256,
                seed,
                method,
                float(frozen["common_target_loss"]),
                int(frozen["tokens_per_update"]),
            )
            cells.append(cell)
            source_audits.append(audit)
    pairing = validate_pairing(cells, run_dir)
    accepted_fingerprints = collect_accepted_cell_fingerprints(run_dir)
    accepted_fingerprint_sha256 = accepted_cells_fingerprint_sha256(
        accepted_fingerprints
    )
    by_seed, contrast_summary = paired_contrasts(
        cells, float(frozen["practical_margin"])
    )
    method_summary = method_summaries(cells)
    decision = build_decision(
        contrast_summary, float(frozen["practical_margin"])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    # A prior successful manifest is immutable evidence.  A controller should
    # select a new analysis directory rather than silently replacing it.
    committed_manifest = output_dir / "record28_analysis_manifest.json"
    if committed_manifest.exists():
        raise RuntimeError(f"analysis output is already committed: {committed_manifest}")

    cell_fields = (
        "seed",
        "method",
        "cproj_k_mode",
        "initial_val_loss",
        "final_val_loss",
        "best_val_loss",
        "tail5_mean",
        "normalized_auc",
        "steps_to_target",
        "tokens_to_target",
        "first_observed_step_at_or_below_target",
        "validation_rows",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
        "k_state_bytes",
        "optimizer_state_bytes",
        "timing_eligible",
        "contract_sha256",
        "source_snapshot_sha256",
        "derived_source_sha256",
        "data_fingerprint_sha256",
        "init_sha256",
        "accepted_pointer",
        "accepted_attempt",
        "scientific_manifest",
    )
    C.write_csv(output_dir / "record28_cells.csv", cells, cell_fields)
    C.write_csv(
        output_dir / "paired_deltas_by_seed.csv",
        by_seed,
        tuple(by_seed[0]),
    )
    C.write_csv(
        output_dir / "paired_contrasts.csv",
        contrast_summary,
        tuple(contrast_summary[0]),
    )
    C.write_csv(
        output_dir / "method_summary.csv",
        method_summary,
        tuple(method_summary[0]),
    )
    C.atomic_write_json(
        output_dir / "integrity_checks.json",
        {
            "passed": True,
            "contract": frozen,
            "pairing": pairing,
            "accepted_cells": 16,
            "quality_recomputed_from_metrics": True,
            "summary_cross_check_tolerance": FLOAT_TOLERANCE,
            "initial_cross_source_validation_tolerance": (
                INITIAL_VALIDATION_TOLERANCE
            ),
        },
    )
    C.atomic_write_json(
        output_dir / "source_audit.json",
        {
            "contract_path": str(contract_path),
            "contract_sha256": contract_sha256,
            "cells": source_audits,
        },
    )
    C.atomic_write_json(output_dir / "record28_decision.json", decision)
    C.atomic_write_text(
        output_dir / "RECORD28_ANALYSIS_REPORT.md",
        build_report(cells, contrast_summary, decision),
    )
    artifact_names = (
        "record28_cells.csv",
        "paired_deltas_by_seed.csv",
        "paired_contrasts.csv",
        "method_summary.csv",
        "integrity_checks.json",
        "source_audit.json",
        "record28_decision.json",
        "RECORD28_ANALYSIS_REPORT.md",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "status": "passed",
        "passed": True,
        "run_dir": str(run_dir),
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha256,
        "accepted_formal_cells": 16,
        "expected_formal_cells": 16,
        "seeds": list(EXPECTED_SEEDS),
        "methods": list(EXPECTED_METHODS),
        "primary_endpoint": "final_val_loss",
        "common_target_loss": frozen["common_target_loss"],
        "common_target_crossing_definition": (
            "first downward crossing, linear interpolation between adjacent "
            "validation checkpoints"
        ),
        "practical_margin": frozen["practical_margin"],
        "timing_eligible": False,
        "statistical_seed_append_gate_triggered": decision[
            "statistical_seed_append_gate_triggered"
        ],
        "decision": decision,
        "primary_classifications": decision["primary_classifications"],
        "accepted_cell_fingerprints": accepted_fingerprints,
        "accepted_cells_fingerprint_sha256": accepted_fingerprint_sha256,
        "artifacts": list(artifact_names),
        "artifact_hashes": {
            name: C.sha256_file(output_dir / name) for name in artifact_names
        },
    }
    # The manifest is the commit marker and is deliberately written last.
    C.atomic_write_json(committed_manifest, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or args.run_dir / "analysis"
    manifest = analyze(args.run_dir, args.contract, output_dir)
    print(f"Record #28 analysis manifest: {output_dir / 'record28_analysis_manifest.json'}")
    print(f"Record #28 accepted formal cells: {manifest['accepted_formal_cells']}/16")


if __name__ == "__main__":
    main()
