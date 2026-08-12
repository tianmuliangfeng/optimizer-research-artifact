"""Audit analytic state-complexity formulas for full/block/diag/none routes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from common import (
    ContractError,
    commit_manifest,
    ensure_new_output,
    read_json,
    resolve_input,
    sha256_file,
    write_csv,
)


CONFIG_SCHEMA = "mdp_routing_complexity_config_v1"
AUDIT_SCHEMA = "mdp_routing_complexity_audit_v1"
ROUTES = ("full", "block", "diag", "none")
FIELDS = [
    "case_id",
    "route",
    "input_dim",
    "block_count",
    "layer_count",
    "stored_matrices",
    "dtype_bytes",
    "elements_per_module_matrix",
    "analytic_state_bytes",
    "analytic_state_mib",
    "relative_to_full",
    "measured_state_bytes",
    "relative_error",
    "measurement_required",
    "measurement_passed",
    "provenance_definition",
    "provenance_passed",
]


def route_elements(input_dim: int, block_count: int) -> dict[str, int]:
    if input_dim <= 0 or block_count <= 0:
        raise ContractError("input_dim and block_count must be positive")
    if input_dim % block_count:
        raise ContractError(
            f"input_dim={input_dim} must be divisible by block_count={block_count}"
        )
    width = input_dim // block_count
    return {
        "full": input_dim * input_dim,
        "block": block_count * width * width,
        "diag": input_dim,
        "none": 0,
    }


def audit(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ContractError(f"complexity config schema must be {CONFIG_SCHEMA}")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ContractError("complexity config requires a non-empty cases list")
    tolerance = float(config.get("measurement_relative_tolerance", 1e-9))
    require_measurement_coverage = bool(config.get("require_measurement_coverage", False))
    require_measurement_provenance = bool(config.get("require_measurement_provenance", False))
    rows: list[dict[str, Any]] = []
    all_measurements_passed = True
    all_required_measurements_present = True
    required_measurement_count = 0
    observed_required_measurement_count = 0
    all_provenance_passed = True
    source_sha256: dict[str, str] = {}
    seen_case_ids: set[str] = set()
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in seen_case_ids:
            raise ContractError(f"duplicate complexity case_id: {case_id}")
        seen_case_ids.add(case_id)
        input_dim = int(case["input_dim"])
        block_count = int(case["block_count"])
        layer_count = int(case["layer_count"])
        stored_matrices = int(case.get("stored_matrices", 2))
        dtype_bytes = int(case.get("dtype_bytes", 4))
        if min(layer_count, stored_matrices, dtype_bytes) <= 0:
            raise ContractError(f"{case_id}: layer_count/stored_matrices/dtype_bytes must be positive")
        elements = route_elements(input_dim, block_count)
        measured = case.get("measured_state_bytes", {})
        provenance = case.get("measurement_provenance", {})
        provenance_definition = str(provenance.get("definition", ""))
        provenance_passed = True
        if require_measurement_provenance or provenance:
            if provenance_definition != "route_total_k_state_bytes_minus_reference_total_k_state_bytes":
                raise ContractError(f"{case_id}: unsupported or missing measurement provenance definition")
            source_files = provenance.get("source_files")
            if not isinstance(source_files, list) or not source_files:
                raise ContractError(f"{case_id}: measurement provenance requires source_files")
            for source in source_files:
                source_path = resolve_input(config_path, str(source["path"]))
                if not source_path.is_file():
                    raise ContractError(f"{case_id}: provenance source does not exist: {source_path}")
                observed_hash = sha256_file(source_path)
                expected_hash = str(source["sha256"]).lower()
                if observed_hash != expected_hash:
                    raise ContractError(f"{case_id}: provenance source hash mismatch: {source_path}")
                source_sha256[str(source_path)] = observed_hash
            reference_total = float(provenance["reference_total_k_state_bytes"])
            route_totals = provenance.get("route_total_k_state_bytes", {})
            for route, observed_raw in measured.items():
                if route not in route_totals:
                    raise ContractError(f"{case_id}: provenance lacks total bytes for route={route}")
                derived = float(route_totals[route]) - reference_total
                if abs(derived - float(observed_raw)) > max(abs(float(observed_raw)), 1.0) * tolerance:
                    raise ContractError(f"{case_id}: measured bytes disagree with provenance for route={route}")
        all_provenance_passed = all_provenance_passed and provenance_passed
        required_routes = {str(route) for route in case.get("required_measured_routes", [])}
        if required_routes - set(ROUTES):
            raise ContractError(
                f"{case_id}: unsupported required_measured_routes={sorted(required_routes - set(ROUTES))}"
            )
        if require_measurement_coverage and not required_routes:
            raise ContractError(f"{case_id}: required_measured_routes must be declared")
        full_bytes = elements["full"] * layer_count * stored_matrices * dtype_bytes
        for route in ROUTES:
            analytic = elements[route] * layer_count * stored_matrices * dtype_bytes
            observed_raw = measured.get(route, "")
            observed: float | str = ""
            error: float | str = ""
            passed: bool | str = ""
            measurement_required = route in required_routes
            if measurement_required:
                required_measurement_count += 1
            if observed_raw != "" and observed_raw is not None:
                observed = float(observed_raw)
                denominator = max(abs(float(analytic)), 1.0)
                error = abs(observed - analytic) / denominator
                passed = error <= tolerance
                all_measurements_passed = all_measurements_passed and passed
                if measurement_required:
                    observed_required_measurement_count += 1
            elif measurement_required:
                passed = False
                all_required_measurements_present = False
                all_measurements_passed = False
            rows.append(
                {
                    "case_id": case_id,
                    "route": route,
                    "input_dim": input_dim,
                    "block_count": block_count,
                    "layer_count": layer_count,
                    "stored_matrices": stored_matrices,
                    "dtype_bytes": dtype_bytes,
                    "elements_per_module_matrix": elements[route],
                    "analytic_state_bytes": analytic,
                    "analytic_state_mib": analytic / (1024 * 1024),
                    "relative_to_full": analytic / full_bytes,
                    "measured_state_bytes": observed,
                    "relative_error": error,
                    "measurement_required": measurement_required,
                    "measurement_passed": passed,
                    "provenance_definition": provenance_definition,
                    "provenance_passed": provenance_passed,
                }
            )

    manifest_name = "routing_complexity_manifest.json"
    ensure_new_output(output_dir, manifest_name)
    write_csv(output_dir / "routing_complexity.csv", rows, FIELDS)
    synthetic = bool(config.get("synthetic", False))
    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed" if all_measurements_passed and all_required_measurements_present and all_provenance_passed else "failed",
        "synthetic": synthetic,
        "claim_eligible": not synthetic and all_measurements_passed and all_required_measurements_present and all_provenance_passed,
        "case_count": len(cases),
        "measurement_relative_tolerance": tolerance,
        "all_measurements_passed": all_measurements_passed,
        "require_measurement_coverage": require_measurement_coverage,
        "required_measurement_count": required_measurement_count,
        "observed_required_measurement_count": observed_required_measurement_count,
        "all_required_measurements_present": all_required_measurements_present,
        "require_measurement_provenance": require_measurement_provenance,
        "all_provenance_passed": all_provenance_passed,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_sha256,
    }
    commit_manifest(output_dir, manifest_name, result, ["routing_complexity.csv"])
    if not all_measurements_passed or not all_required_measurements_present or not all_provenance_passed:
        raise ContractError("one or more measured state sizes or provenance records failed")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.config.resolve(), args.output_dir.resolve())
    print(
        f"routing-complexity audit passed: cases={result['case_count']} synthetic={result['synthetic']}"
    )


if __name__ == "__main__":
    main()
