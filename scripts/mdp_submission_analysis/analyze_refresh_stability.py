"""Analyze refresh-boundary stability from paired K/gradient snapshots.

Input contract:
  snapshots.npz: k_before [N,d,d], k_after [N,d,d],
                 gradient_before [N,m,d], gradient_after [N,m,d],
                 optional matched_gradient [N,m,d],
                 optional inverse_before/inverse_after [N,d,d],
                 optional ridge_before/ridge_after [N],
                 optional loss_impulse_step48/loss_impulse_step80/
                          loss_impulse_auc [N]
  metadata.csv:  N rows with unit_id, origin, replica, stage and, for the
                 formal contract, module_id, layer_index, refresh_event_step

Origins and replicas are retained as nested replay units. They are never relabeled
as independent training seeds.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    ContractError,
    commit_manifest,
    ensure_new_output,
    mean,
    read_csv,
    read_json,
    sample_sd,
    sha256_file,
    write_csv,
)


AUDIT_SCHEMA = "mdp_refresh_stability_audit_v2"
SNAPSHOT_MANIFEST_SCHEMA = "mdp_refresh_snapshot_manifest_v2"
UNIT_FIELDS = [
    "unit_id",
    "origin",
    "replica",
    "stage",
    "module_id",
    "layer_index",
    "refresh_event_step",
    "source_method",
    "checkpoint_step",
    "gradient_semantics",
    "ridge_source",
    "ridge_before",
    "ridge_after",
    "condition_before",
    "condition_after",
    "minimum_eigenvalue_before",
    "minimum_eigenvalue_after",
    "k_asymmetry_before",
    "k_asymmetry_after",
    "relative_k_change",
    "relative_inverse_change",
    "inverse_residual_before",
    "inverse_residual_after",
    "runtime_inverse_residual_before",
    "runtime_inverse_residual_after",
    "runtime_inverse_relative_error_before",
    "runtime_inverse_relative_error_after",
    "relative_runtime_inverse_change",
    "resolvent_relative_residual",
    "resolvent_bound",
    "inverse_change_to_bound_ratio",
    "resolvent_bound_passed",
    "relative_gradient_change",
    "gradient_cosine",
    "relative_preconditioned_gradient_change",
    "preconditioned_gradient_cosine",
    "relative_polar_update_change",
    "polar_update_cosine",
    "relative_runtime_preconditioned_gradient_change",
    "runtime_preconditioned_gradient_cosine",
    "relative_runtime_polar_factor_change",
    "runtime_polar_factor_cosine",
    "relative_observed_preconditioned_gradient_change",
    "observed_preconditioned_gradient_cosine",
    "relative_observed_polar_update_change",
    "observed_polar_update_cosine",
    "finite_check_passed",
    "loss_impulse",
    "loss_impulse_step48",
    "loss_impulse_step80",
    "loss_impulse_auc",
]

SUMMARY_FIELDS = [
    "group_type",
    "group_value",
    "unit_count",
    "origin_count",
    "replica_count",
    "relative_k_change_mean",
    "relative_k_change_sd",
    "relative_inverse_change_mean",
    "relative_polar_update_change_mean",
    "loss_impulse_mean",
    "loss_impulse_step48_mean",
    "loss_impulse_step80_mean",
    "loss_impulse_auc_mean",
    "correlation_loss_impulse_vs_inverse_change",
    "correlation_loss_impulse_vs_update_change",
    "correlation_step48_vs_inverse_change",
    "correlation_step48_vs_update_change",
    "correlation_step80_vs_inverse_change",
    "correlation_step80_vs_update_change",
    "correlation_auc_vs_inverse_change",
    "correlation_auc_vs_update_change",
]


def _fro(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix, ord="fro"))


def _relative(observed: np.ndarray, reference: np.ndarray) -> float:
    return _fro(observed - reference) / max(_fro(reference), 1e-15)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-15:
        return math.nan
    return float(np.vdot(left, right).real / denominator)


def _polar(matrix: np.ndarray) -> np.ndarray:
    left, _, right_t = np.linalg.svd(matrix, full_matrices=False)
    return left @ right_t


def _correlation(left: list[float], right: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(left, right) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return math.nan
    x = np.asarray([pair[0] for pair in pairs], dtype=float)
    y = np.asarray([pair[1] for pair in pairs], dtype=float)
    if float(np.std(x)) <= 1e-15 or float(np.std(y)) <= 1e-15:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _summarize(rows: list[dict[str, Any]], group_type: str, group_value: str) -> dict[str, Any]:
    inverse_change = [float(row["relative_inverse_change"]) for row in rows]
    update_change = [float(row["relative_polar_update_change"]) for row in rows]
    impulses = [float(row["loss_impulse"]) for row in rows]
    impulses_step48 = [float(row["loss_impulse_step48"]) for row in rows]
    impulses_step80 = [float(row["loss_impulse_step80"]) for row in rows]
    impulses_auc = [float(row["loss_impulse_auc"]) for row in rows]
    k_change = [float(row["relative_k_change"]) for row in rows]
    return {
        "group_type": group_type,
        "group_value": group_value,
        "unit_count": len(rows),
        "origin_count": len({row["origin"] for row in rows}),
        "replica_count": len({(row["origin"], row["replica"]) for row in rows}),
        "relative_k_change_mean": mean(k_change),
        "relative_k_change_sd": sample_sd(k_change),
        "relative_inverse_change_mean": mean(inverse_change),
        "relative_polar_update_change_mean": mean(update_change),
        "loss_impulse_mean": mean(impulses),
        "loss_impulse_step48_mean": mean(impulses_step48),
        "loss_impulse_step80_mean": mean(impulses_step80),
        "loss_impulse_auc_mean": mean(impulses_auc),
        "correlation_loss_impulse_vs_inverse_change": _correlation(impulses, inverse_change),
        "correlation_loss_impulse_vs_update_change": _correlation(impulses, update_change),
        "correlation_step48_vs_inverse_change": _correlation(impulses_step48, inverse_change),
        "correlation_step48_vs_update_change": _correlation(impulses_step48, update_change),
        "correlation_step80_vs_inverse_change": _correlation(impulses_step80, inverse_change),
        "correlation_step80_vs_update_change": _correlation(impulses_step80, update_change),
        "correlation_auc_vs_inverse_change": _correlation(impulses_auc, inverse_change),
        "correlation_auc_vs_update_change": _correlation(impulses_auc, update_change),
    }


def analyze(
    snapshots_path: Path,
    metadata_path: Path,
    output_dir: Path,
    ridge_scale: float = 0.2,
    ridge_epsilon: float = 1e-8,
    residual_tolerance: float = 1e-8,
    synthetic: bool = False,
    formal_contract: bool = False,
    snapshot_manifest_path: Path | None = None,
    runtime_inverse_tolerance: float = 5e-3,
    symmetry_tolerance: float = 1e-6,
) -> dict[str, Any]:
    metadata = read_csv(metadata_path)
    if not metadata:
        raise ContractError("refresh metadata has no rows")
    required_meta = {"unit_id", "origin", "replica", "stage"}
    if not required_meta.issubset(metadata[0]):
        raise ContractError(f"refresh metadata requires columns: {sorted(required_meta)}")
    formal_meta = {
        "module_id",
        "layer_index",
        "refresh_event_step",
        "source_method",
        "checkpoint_step",
        "gradient_semantics",
    }
    if formal_contract and not formal_meta.issubset(metadata[0]):
        raise ContractError(
            f"formal refresh metadata requires columns: {sorted(required_meta | formal_meta)}"
        )
    with np.load(snapshots_path, allow_pickle=False) as archive:
        required_arrays = {"k_before", "k_after", "gradient_before", "gradient_after"}
        missing = sorted(required_arrays - set(archive.files))
        if missing:
            raise ContractError(f"refresh snapshots missing arrays: {missing}")
        k_before = np.asarray(archive["k_before"], dtype=float)
        k_after = np.asarray(archive["k_after"], dtype=float)
        gradient_before = np.asarray(archive["gradient_before"], dtype=float)
        gradient_after = np.asarray(archive["gradient_after"], dtype=float)
        legacy_loss_impulse = (
            np.asarray(archive["loss_impulse"], dtype=float)
            if "loss_impulse" in archive.files
            else np.full(len(metadata), np.nan)
        )
        loss_impulse_step48 = (
            np.asarray(archive["loss_impulse_step48"], dtype=float)
            if "loss_impulse_step48" in archive.files
            else np.full(len(metadata), np.nan)
        )
        loss_impulse_step80 = (
            np.asarray(archive["loss_impulse_step80"], dtype=float)
            if "loss_impulse_step80" in archive.files
            else np.full(len(metadata), np.nan)
        )
        loss_impulse_auc = (
            np.asarray(archive["loss_impulse_auc"], dtype=float)
            if "loss_impulse_auc" in archive.files
            else np.full(len(metadata), np.nan)
        )
        matched_gradient_supplied = "matched_gradient" in archive.files
        matched_gradient = (
            np.asarray(archive["matched_gradient"], dtype=float)
            if matched_gradient_supplied
            else gradient_before.copy()
        )
        stored_inverse_dtype_before = (
            str(archive["inverse_before"].dtype) if "inverse_before" in archive.files else ""
        )
        stored_inverse_dtype_after = (
            str(archive["inverse_after"].dtype) if "inverse_after" in archive.files else ""
        )
        stored_inverse_before = (
            np.asarray(archive["inverse_before"], dtype=float)
            if "inverse_before" in archive.files
            else None
        )
        stored_inverse_after = (
            np.asarray(archive["inverse_after"], dtype=float)
            if "inverse_after" in archive.files
            else None
        )
        ridge_before_array = (
            np.asarray(archive["ridge_before"], dtype=float)
            if "ridge_before" in archive.files
            else None
        )
        ridge_after_array = (
            np.asarray(archive["ridge_after"], dtype=float)
            if "ridge_after" in archive.files
            else None
        )
    snapshot_manifest_verified = False
    snapshot_manifest_sha256 = ""
    if snapshot_manifest_path is not None:
        snapshot_manifest = read_json(snapshot_manifest_path)
        snapshot_manifest_sha256 = sha256_file(snapshot_manifest_path)
        required_snapshot_fields = {
            "schema_version",
            "snapshots_sha256",
            "metadata_sha256",
            "runtime_contract_sha256",
            "source_hashes",
            "production_pipeline_replayed",
            "input_fingerprint_validation_passed",
        }
        if not required_snapshot_fields.issubset(snapshot_manifest):
            raise ContractError("refresh snapshot manifest is missing required provenance fields")
        if snapshot_manifest["schema_version"] != SNAPSHOT_MANIFEST_SCHEMA:
            raise ContractError(f"snapshot manifest schema must be {SNAPSHOT_MANIFEST_SCHEMA}")
        if str(snapshot_manifest["snapshots_sha256"]).lower() != sha256_file(snapshots_path):
            raise ContractError("snapshot manifest snapshots_sha256 mismatch")
        if str(snapshot_manifest["metadata_sha256"]).lower() != sha256_file(metadata_path):
            raise ContractError("snapshot manifest metadata_sha256 mismatch")
        source_hashes = snapshot_manifest["source_hashes"]
        runtime_hash = str(snapshot_manifest["runtime_contract_sha256"])
        snapshot_manifest_verified = (
            isinstance(source_hashes, dict)
            and bool(source_hashes)
            and all(len(str(value)) == 64 for value in source_hashes.values())
            and len(runtime_hash) == 64
            and bool(snapshot_manifest["production_pipeline_replayed"])
            and bool(snapshot_manifest["input_fingerprint_validation_passed"])
        )
        if not snapshot_manifest_verified:
            raise ContractError("snapshot manifest provenance or fingerprint validation failed")
    count = len(metadata)
    if any(array.shape[0] != count for array in (k_before, k_after, gradient_before, gradient_after)):
        raise ContractError("refresh snapshot batch dimensions do not match metadata row count")
    if k_before.ndim != 3 or k_before.shape != k_after.shape or k_before.shape[1] != k_before.shape[2]:
        raise ContractError("k_before/k_after must have identical [N,d,d] shapes")
    if gradient_before.ndim != 3 or gradient_before.shape != gradient_after.shape:
        raise ContractError("gradient_before/gradient_after must have identical [N,m,d] shapes")
    if matched_gradient.shape != gradient_before.shape:
        raise ContractError("matched_gradient must have the same [N,m,d] shape as gradients")
    if gradient_before.shape[2] != k_before.shape[1]:
        raise ContractError("gradient input dimension must match K dimension")
    for name, array in (
        ("loss_impulse", legacy_loss_impulse),
        ("loss_impulse_step48", loss_impulse_step48),
        ("loss_impulse_step80", loss_impulse_step80),
        ("loss_impulse_auc", loss_impulse_auc),
    ):
        if array.shape != (count,):
            raise ContractError(f"{name} must have shape [N]")
    if (stored_inverse_before is None) != (stored_inverse_after is None):
        raise ContractError("inverse_before and inverse_after must be supplied together")
    if stored_inverse_before is not None and (
        stored_inverse_before.shape != k_before.shape or stored_inverse_after.shape != k_after.shape
    ):
        raise ContractError("stored inverses must have the same [N,d,d] shape as K")
    if (ridge_before_array is None) != (ridge_after_array is None):
        raise ContractError("ridge_before and ridge_after must be supplied together")
    if ridge_before_array is not None and (
        ridge_before_array.shape != (count,) or ridge_after_array.shape != (count,)
    ):
        raise ContractError("ridge_before/ridge_after must have shape [N]")
    if len({row["unit_id"] for row in metadata}) != count:
        raise ContractError("metadata unit_id values must be unique")

    formal_missing: list[str] = []
    if not matched_gradient_supplied:
        formal_missing.append("matched_gradient")
    if stored_inverse_before is None:
        formal_missing.extend(["inverse_before", "inverse_after"])
    explicit_ridge_supplied = ridge_before_array is not None or all(
        row.get("ridge_before", "") not in ("", None)
        and row.get("ridge_after", "") not in ("", None)
        for row in metadata
    )
    if not explicit_ridge_supplied:
        formal_missing.extend(["ridge_before", "ridge_after"])
    if not snapshot_manifest_verified:
        formal_missing.append("verified_snapshot_manifest")
    for name, array in (
        ("loss_impulse_step48", loss_impulse_step48),
        ("loss_impulse_step80", loss_impulse_step80),
        ("loss_impulse_auc", loss_impulse_auc),
    ):
        if not np.all(np.isfinite(array)):
            formal_missing.append(name)
    origin_replicas: dict[str, set[str]] = defaultdict(set)
    for row in metadata:
        origin_replicas[str(row["origin"])].add(str(row["replica"]))
    expected_origins = {
        "early_muon",
        "early_newton_full",
        "late_muon",
        "late_newton_full",
    }
    replay_coverage_passed = set(origin_replicas) == expected_origins and all(
        replicas == {"0", "1", "2"} for replicas in origin_replicas.values()
    )
    if not replay_coverage_passed:
        formal_missing.append("mech09r_four_origins_by_three_replicas")
    expected_method = {
        "early_muon": "muon",
        "early_newton_full": "newton_full",
        "late_muon": "muon",
        "late_newton_full": "newton_full",
    }
    formal_metadata_semantics_passed = all(
        row.get("gradient_semantics") == "pre_polar_optimizer_input"
        and row.get("source_method") == expected_method.get(str(row["origin"]))
        and str(row.get("checkpoint_step"))
        == ("1000" if str(row["origin"]).startswith("early_") else "6200")
        for row in metadata
    )
    if not formal_metadata_semantics_passed:
        formal_missing.append("mech09r_metadata_semantics")
    formal_contract_satisfied = not formal_missing and formal_meta.issubset(metadata[0])
    if formal_contract and not formal_contract_satisfied:
        raise ContractError(
            "formal refresh contract is incomplete; missing or non-finite fields: "
            + ", ".join(sorted(set(formal_missing)))
        )

    unit_rows: list[dict[str, Any]] = []
    all_resolvent_checks_passed = True
    all_inverse_checks_passed = True
    all_runtime_inverse_checks_passed = True
    all_bound_checks_passed = True
    all_finite_checks_passed = True
    all_symmetry_checks_passed = True
    metadata_has_ridge = all(
        row.get("ridge_before", "") not in ("", None)
        and row.get("ridge_after", "") not in ("", None)
        for row in metadata
    )
    if ridge_before_array is not None:
        ridge_source = "snapshot_arrays"
    elif metadata_has_ridge:
        ridge_source = "metadata_columns"
    else:
        ridge_source = "trace_scaled_reconstruction"
    for index, meta in enumerate(metadata):
        asymmetry_before = _fro(k_before[index] - k_before[index].T) / max(
            _fro(k_before[index]), 1e-15
        )
        asymmetry_after = _fro(k_after[index] - k_after[index].T) / max(
            _fro(k_after[index]), 1e-15
        )
        symmetry_passed = max(asymmetry_before, asymmetry_after) <= symmetry_tolerance
        all_symmetry_checks_passed = all_symmetry_checks_passed and symmetry_passed
        before = 0.5 * (k_before[index] + k_before[index].T)
        after = 0.5 * (k_after[index] + k_after[index].T)
        dimension = before.shape[0]
        if ridge_before_array is not None:
            ridge_before = float(ridge_before_array[index])
            ridge_after = float(ridge_after_array[index])
        elif metadata_has_ridge:
            ridge_before = float(meta["ridge_before"])
            ridge_after = float(meta["ridge_after"])
        else:
            ridge_before = ridge_scale * float(np.trace(before) / dimension) + ridge_epsilon
            ridge_after = ridge_scale * float(np.trace(after) / dimension) + ridge_epsilon
        if not math.isfinite(ridge_before) or not math.isfinite(ridge_after):
            raise ContractError(f"{meta['unit_id']}: ridge is not finite")
        if ridge_before < 0 or ridge_after < 0:
            raise ContractError(f"{meta['unit_id']}: ridge must be non-negative")
        a_before = before + ridge_before * np.eye(dimension)
        a_after = after + ridge_after * np.eye(dimension)
        eigen_before = np.linalg.eigvalsh(a_before)
        eigen_after = np.linalg.eigvalsh(a_after)
        if float(eigen_before[0]) <= 0 or float(eigen_after[0]) <= 0:
            raise ContractError(f"{meta['unit_id']}: regularized K is not positive definite")
        exact_inverse_before = np.linalg.inv(a_before)
        exact_inverse_after = np.linalg.inv(a_after)
        runtime_inverse_before = (
            stored_inverse_before[index]
            if stored_inverse_before is not None
            else exact_inverse_before
        )
        runtime_inverse_after = (
            stored_inverse_after[index]
            if stored_inverse_after is not None
            else exact_inverse_after
        )
        identity = np.eye(dimension)
        inverse_residual_before = _fro(a_before @ exact_inverse_before - identity) / max(
            _fro(identity), 1e-15
        )
        inverse_residual_after = _fro(a_after @ exact_inverse_after - identity) / max(
            _fro(identity), 1e-15
        )
        inverse_passed = max(inverse_residual_before, inverse_residual_after) <= residual_tolerance
        all_inverse_checks_passed = all_inverse_checks_passed and inverse_passed
        runtime_inverse_residual_before = _fro(
            a_before @ runtime_inverse_before - identity
        ) / max(_fro(identity), 1e-15)
        runtime_inverse_residual_after = _fro(
            a_after @ runtime_inverse_after - identity
        ) / max(_fro(identity), 1e-15)
        runtime_inverse_relative_error_before = _relative(
            runtime_inverse_before, exact_inverse_before
        )
        runtime_inverse_relative_error_after = _relative(
            runtime_inverse_after, exact_inverse_after
        )
        runtime_inverse_passed = max(
            runtime_inverse_residual_before,
            runtime_inverse_residual_after,
            runtime_inverse_relative_error_before,
            runtime_inverse_relative_error_after,
        ) <= runtime_inverse_tolerance
        all_runtime_inverse_checks_passed = (
            all_runtime_inverse_checks_passed and runtime_inverse_passed
        )
        delta_a = a_after - a_before
        resolvent_prediction = -exact_inverse_after @ delta_a @ exact_inverse_before
        inverse_delta = exact_inverse_after - exact_inverse_before
        residual = _fro(inverse_delta - resolvent_prediction) / max(_fro(inverse_delta), 1e-15)
        all_resolvent_checks_passed = all_resolvent_checks_passed and residual <= residual_tolerance
        bound = float(np.linalg.norm(exact_inverse_before, 2) * np.linalg.norm(delta_a, 2) * np.linalg.norm(exact_inverse_after, 2))
        inverse_delta_norm = float(np.linalg.norm(inverse_delta, 2))
        bound_passed = inverse_delta_norm <= bound * (1.0 + residual_tolerance) + residual_tolerance
        all_bound_checks_passed = all_bound_checks_passed and bound_passed

        fixed_gradient = matched_gradient[index]
        pre_before = fixed_gradient @ exact_inverse_before
        pre_after = fixed_gradient @ exact_inverse_after
        polar_before = _polar(pre_before)
        polar_after = _polar(pre_after)
        runtime_pre_before = fixed_gradient @ runtime_inverse_before
        runtime_pre_after = fixed_gradient @ runtime_inverse_after
        runtime_polar_before = _polar(runtime_pre_before)
        runtime_polar_after = _polar(runtime_pre_after)
        observed_pre_before = gradient_before[index] @ exact_inverse_before
        observed_pre_after = gradient_after[index] @ exact_inverse_after
        observed_polar_before = _polar(observed_pre_before)
        observed_polar_after = _polar(observed_pre_after)
        numeric_values = [
            ridge_before,
            ridge_after,
            float(np.linalg.cond(a_before)),
            float(np.linalg.cond(a_after)),
            inverse_residual_before,
            inverse_residual_after,
            runtime_inverse_residual_before,
            runtime_inverse_residual_after,
            runtime_inverse_relative_error_before,
            runtime_inverse_relative_error_after,
            residual,
            bound,
            inverse_delta_norm,
            asymmetry_before,
            asymmetry_after,
            _relative(after, before),
            _relative(exact_inverse_after, exact_inverse_before),
            _relative(runtime_inverse_after, runtime_inverse_before),
            _relative(gradient_after[index], gradient_before[index]),
            _cosine(gradient_after[index], gradient_before[index]),
            _relative(pre_after, pre_before),
            _cosine(pre_after, pre_before),
            _relative(polar_after, polar_before),
            _cosine(polar_after, polar_before),
            _relative(runtime_pre_after, runtime_pre_before),
            _cosine(runtime_pre_after, runtime_pre_before),
            _relative(runtime_polar_after, runtime_polar_before),
            _cosine(runtime_polar_after, runtime_polar_before),
            _relative(observed_pre_after, observed_pre_before),
            _cosine(observed_pre_after, observed_pre_before),
            _relative(observed_polar_after, observed_polar_before),
            _cosine(observed_polar_after, observed_polar_before),
        ]
        finite_passed = all(math.isfinite(value) for value in numeric_values)
        all_finite_checks_passed = all_finite_checks_passed and finite_passed
        unit_rows.append(
            {
                "unit_id": meta["unit_id"],
                "origin": meta["origin"],
                "replica": meta["replica"],
                "stage": meta["stage"],
                "module_id": meta.get("module_id", ""),
                "layer_index": meta.get("layer_index", ""),
                "refresh_event_step": meta.get("refresh_event_step", ""),
                "source_method": meta.get("source_method", ""),
                "checkpoint_step": meta.get("checkpoint_step", ""),
                "gradient_semantics": meta.get("gradient_semantics", ""),
                "ridge_source": ridge_source,
                "ridge_before": ridge_before,
                "ridge_after": ridge_after,
                "condition_before": float(np.linalg.cond(a_before)),
                "condition_after": float(np.linalg.cond(a_after)),
                "minimum_eigenvalue_before": float(eigen_before[0]),
                "minimum_eigenvalue_after": float(eigen_after[0]),
                "k_asymmetry_before": asymmetry_before,
                "k_asymmetry_after": asymmetry_after,
                "relative_k_change": _relative(after, before),
                "relative_inverse_change": _relative(
                    exact_inverse_after, exact_inverse_before
                ),
                "inverse_residual_before": inverse_residual_before,
                "inverse_residual_after": inverse_residual_after,
                "runtime_inverse_residual_before": runtime_inverse_residual_before,
                "runtime_inverse_residual_after": runtime_inverse_residual_after,
                "runtime_inverse_relative_error_before": runtime_inverse_relative_error_before,
                "runtime_inverse_relative_error_after": runtime_inverse_relative_error_after,
                "relative_runtime_inverse_change": _relative(
                    runtime_inverse_after, runtime_inverse_before
                ),
                "resolvent_relative_residual": residual,
                "resolvent_bound": bound,
                "inverse_change_to_bound_ratio": inverse_delta_norm / max(bound, 1e-15),
                "resolvent_bound_passed": bound_passed,
                "relative_gradient_change": _relative(
                    gradient_after[index], gradient_before[index]
                ),
                "gradient_cosine": _cosine(gradient_after[index], gradient_before[index]),
                "relative_preconditioned_gradient_change": _relative(pre_after, pre_before),
                "preconditioned_gradient_cosine": _cosine(pre_after, pre_before),
                "relative_polar_update_change": _relative(polar_after, polar_before),
                "polar_update_cosine": _cosine(polar_after, polar_before),
                "relative_runtime_preconditioned_gradient_change": _relative(
                    runtime_pre_after, runtime_pre_before
                ),
                "runtime_preconditioned_gradient_cosine": _cosine(
                    runtime_pre_after, runtime_pre_before
                ),
                "relative_runtime_polar_factor_change": _relative(
                    runtime_polar_after, runtime_polar_before
                ),
                "runtime_polar_factor_cosine": _cosine(
                    runtime_polar_after, runtime_polar_before
                ),
                "relative_observed_preconditioned_gradient_change": _relative(
                    observed_pre_after, observed_pre_before
                ),
                "observed_preconditioned_gradient_cosine": _cosine(
                    observed_pre_after, observed_pre_before
                ),
                "relative_observed_polar_update_change": _relative(
                    observed_polar_after, observed_polar_before
                ),
                "observed_polar_update_cosine": _cosine(
                    observed_polar_after, observed_polar_before
                ),
                "finite_check_passed": finite_passed,
                "loss_impulse": float(legacy_loss_impulse[index]),
                "loss_impulse_step48": float(loss_impulse_step48[index]),
                "loss_impulse_step80": float(loss_impulse_step80[index]),
                "loss_impulse_auc": float(loss_impulse_auc[index]),
            }
        )

    summary_rows = [_summarize(unit_rows, "all", "all")]
    for group_type in (
        "origin",
        "stage",
        "source_method",
        "module_id",
        "refresh_event_step",
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in unit_rows:
            grouped[str(row[group_type])].append(row)
        for group_value in sorted(grouped):
            summary_rows.append(_summarize(grouped[group_value], group_type, group_value))

    manifest_name = "refresh_stability_manifest.json"
    ensure_new_output(output_dir, manifest_name)
    write_csv(output_dir / "refresh_unit_metrics.csv", unit_rows, UNIT_FIELDS)
    write_csv(output_dir / "refresh_stratified_summary.csv", summary_rows, SUMMARY_FIELDS)
    origin_count = len({row["origin"] for row in unit_rows})
    replica_count = len({(row["origin"], row["replica"]) for row in unit_rows})
    all_numeric_checks_passed = (
        all_resolvent_checks_passed
        and all_inverse_checks_passed
        and all_bound_checks_passed
        and all_finite_checks_passed
        and all_symmetry_checks_passed
    )
    all_formal_numeric_checks_passed = (
        all_numeric_checks_passed and all_runtime_inverse_checks_passed
    )
    formal_contract_satisfied = (
        formal_contract_satisfied and all_formal_numeric_checks_passed
    )
    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed" if (
            all_formal_numeric_checks_passed if formal_contract else all_numeric_checks_passed
        ) else "failed",
        "synthetic": synthetic,
        "claim_eligible": (
            not synthetic and all_numeric_checks_passed and formal_contract_satisfied
        ),
        "unit_count": len(unit_rows),
        "origin_count": origin_count,
        "nested_replica_count": replica_count,
        "replica_interpretation": "nested_replay_units_not_independent_training_seeds",
        "ridge_scale": ridge_scale,
        "ridge_epsilon": ridge_epsilon,
        "ridge_source": ridge_source,
        "matched_gradient_source": (
            "snapshot_array" if matched_gradient_supplied else "gradient_before"
        ),
        "stored_inverse_used": stored_inverse_before is not None,
        "stored_inverse_dtype_before": stored_inverse_dtype_before,
        "stored_inverse_dtype_after": stored_inverse_dtype_after,
        "runtime_inverse_tolerance": runtime_inverse_tolerance,
        "symmetry_tolerance": symmetry_tolerance,
        "snapshot_manifest_verified": snapshot_manifest_verified,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "replay_coverage_passed": replay_coverage_passed,
        "formal_metadata_semantics_passed": formal_metadata_semantics_passed,
        "formal_contract_requested": formal_contract,
        "formal_contract_satisfied": formal_contract_satisfied,
        "formal_contract_missing": sorted(set(formal_missing)),
        "analysis_class": (
            "formal_paper_evidence" if formal_contract_satisfied else "diagnostic_only"
        ),
        "loss_impulse_fields": [
            "loss_impulse_step48",
            "loss_impulse_step80",
            "loss_impulse_auc",
        ],
        "resolvent_residual_tolerance": residual_tolerance,
        "resolvent_inverse_semantics": "float64_exact_inverse_of_symmetrized_regularized_k",
        "runtime_inverse_semantics": "exported_production_inverse_audited_separately",
        "matched_gradient_metric_semantics": "fixed_g_times_exact_or_runtime_inverse",
        "polar_metric_semantics": "exact_svd_polar_factor_not_full_optimizer_update",
        "correlation_interpretation": "descriptive_nested_replay_units_no_seed_level_inference",
        "all_resolvent_checks_passed": all_resolvent_checks_passed,
        "all_inverse_checks_passed": all_inverse_checks_passed,
        "all_runtime_inverse_checks_passed": all_runtime_inverse_checks_passed,
        "all_resolvent_bound_checks_passed": all_bound_checks_passed,
        "all_symmetry_checks_passed": all_symmetry_checks_passed,
        "all_finite_checks_passed": all_finite_checks_passed,
        "all_numeric_checks_passed": all_numeric_checks_passed,
        "all_formal_numeric_checks_passed": all_formal_numeric_checks_passed,
        "snapshots_sha256": sha256_file(snapshots_path),
        "metadata_sha256": sha256_file(metadata_path),
        "numpy_version": np.__version__,
    }
    commit_manifest(
        output_dir,
        manifest_name,
        result,
        ["refresh_unit_metrics.csv", "refresh_stratified_summary.csv"],
    )
    if formal_contract and not all_formal_numeric_checks_passed:
        raise ContractError("one or more formal refresh stability checks failed")
    if not all_numeric_checks_passed:
        raise ContractError("one or more refresh stability numeric checks failed")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge-scale", type=float, default=0.2)
    parser.add_argument("--ridge-epsilon", type=float, default=1e-8)
    parser.add_argument("--residual-tolerance", type=float, default=1e-8)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--formal-contract", action="store_true")
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--runtime-inverse-tolerance", type=float, default=5e-3)
    parser.add_argument("--symmetry-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(
        args.snapshots.resolve(),
        args.metadata.resolve(),
        args.output_dir.resolve(),
        args.ridge_scale,
        args.ridge_epsilon,
        args.residual_tolerance,
        args.synthetic,
        args.formal_contract,
        args.snapshot_manifest.resolve() if args.snapshot_manifest else None,
        args.runtime_inverse_tolerance,
        args.symmetry_tolerance,
    )
    print(
        f"refresh-stability analysis passed: units={result['unit_count']} "
        f"origins={result['origin_count']} synthetic={result['synthetic']}"
    )


if __name__ == "__main__":
    main()
