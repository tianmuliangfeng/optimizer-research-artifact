"""Numerically audit the algebra used by the method-deepening formulation.

This is an implementation/reference check, not an empirical experiment and not
a substitute for a formal proof. It verifies dimensions, positive-definite
ridge contracts, the resolvent identity/bound, and the alpha inverse derivative.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from common import ContractError, commit_manifest, ensure_new_output, write_csv


SCHEMA = "mdp_method_formulation_audit_v1"
FIELDS = ["check_id", "claim_class", "metric", "observed", "tolerance", "passed", "note"]


def _relative_error(observed: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(observed - expected) / max(np.linalg.norm(expected), 1e-15))


def _block_projection(matrix: np.ndarray, blocks: int) -> np.ndarray:
    dimension = matrix.shape[0]
    if dimension % blocks:
        raise ContractError("matrix dimension must be divisible by blocks")
    width = dimension // blocks
    output = np.zeros_like(matrix)
    for index in range(blocks):
        section = slice(index * width, (index + 1) * width)
        output[section, section] = matrix[section, section]
    return output


def audit(
    output_dir: Path,
    seed: int = 20260803,
    input_dim: int = 12,
    output_dim: int = 8,
    blocks: int = 3,
    tolerance: float = 1e-9,
    derivative_tolerance: float = 2e-6,
) -> dict[str, Any]:
    if input_dim <= 0 or output_dim <= 0 or input_dim % blocks:
        raise ContractError("invalid formulation audit dimensions")
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(4 * input_dim, input_dim))
    k_matrix = samples.T @ samples / samples.shape[0] + 0.3 * np.eye(input_dim)
    gradient = rng.normal(size=(output_dim, input_dim))
    ridge = 0.2 * float(np.trace(k_matrix) / input_dim)
    identity = np.eye(input_dim)

    full_p = np.linalg.inv(k_matrix + ridge * identity)
    diag_k = np.diag(np.diag(k_matrix))
    diag_p = np.diag(1.0 / (np.diag(k_matrix) + ridge))
    block_k = _block_projection(k_matrix, blocks)
    block_p = np.linalg.inv(block_k + ridge * identity)
    none_p = identity

    rows: list[dict[str, Any]] = []

    def add(check_id: str, claim_class: str, metric: str, observed: float, limit: float, note: str) -> None:
        rows.append(
            {
                "check_id": check_id,
                "claim_class": claim_class,
                "metric": metric,
                "observed": observed,
                "tolerance": limit,
                "passed": observed <= limit,
                "note": note,
            }
        )

    shape_error = 0.0 if all(
        matrix.shape == (input_dim, input_dim) for matrix in (full_p, diag_p, block_p, none_p)
    ) and (gradient @ full_p).shape == (output_dim, input_dim) else 1.0
    add("F01", "definition", "shape_error", shape_error, 0.0, "G is [m,n], P is [n,n], and GP is [m,n].")
    add("F02", "definition", "none_identity_error", _relative_error(none_p, identity), tolerance, "none sets only the eligible matrix right preconditioner to I.")
    min_eigenvalue = min(
        float(np.linalg.eigvalsh(matrix)[0]) for matrix in (full_p, diag_p, block_p)
    )
    add("F03", "ridge_contract", "nonpositive_preconditioner_eigenvalue", max(0.0, -min_eigenvalue), 0.0, "Positive ridge preserves SPD preconditioners.")

    perturbation = rng.normal(size=(input_dim, 3))
    delta_k = 0.015 * (perturbation @ perturbation.T) / perturbation.shape[1]
    k_after = k_matrix + delta_k
    ridge_after = 0.2 * float(np.trace(k_after) / input_dim)
    a_before = k_matrix + ridge * identity
    a_after = k_after + ridge_after * identity
    inverse_before = np.linalg.inv(a_before)
    inverse_after = np.linalg.inv(a_after)
    delta_a = a_after - a_before
    inverse_delta = inverse_after - inverse_before
    resolvent = -inverse_after @ delta_a @ inverse_before
    resolvent_residual = _relative_error(inverse_delta, resolvent)
    add("F04", "theorem_identity", "resolvent_relative_residual", resolvent_residual, tolerance, "A+^-1-A-^-1 = -A+^-1 DeltaA A-^-1.")
    bound = float(np.linalg.norm(inverse_after, 2) * np.linalg.norm(delta_a, 2) * np.linalg.norm(inverse_before, 2))
    bound_violation = max(0.0, float(np.linalg.norm(inverse_delta, 2)) - bound)
    add("F05", "theorem_bound", "inverse_bound_violation", bound_violation, tolerance, "Spectral-norm resolvent bound.")
    gradient_bound = float(np.linalg.norm(gradient, 2) * bound)
    gradient_violation = max(0.0, float(np.linalg.norm(gradient @ inverse_delta, 2)) - gradient_bound)
    add("F06", "theorem_bound", "matched_gradient_bound_violation", gradient_violation, tolerance, "The same frozen G is used on both sides.")

    alpha = 0.43
    alpha_step = 1e-5

    def alpha_matrix(value: float) -> np.ndarray:
        return diag_k + value * (k_matrix - diag_k) + ridge * identity

    traces = [float(np.trace(alpha_matrix(value))) for value in (0.0, 0.25, 0.5, 0.75, 1.0)]
    trace_spread = max(traces) - min(traces)
    add("F07", "theorem_identity", "alpha_trace_spread", trace_spread, tolerance, "Fixed trace-based ridge is constant along the alpha path.")
    inverse_alpha = np.linalg.inv(alpha_matrix(alpha))
    analytic_derivative = -inverse_alpha @ (k_matrix - diag_k) @ inverse_alpha
    finite_difference = (
        np.linalg.inv(alpha_matrix(alpha + alpha_step))
        - np.linalg.inv(alpha_matrix(alpha - alpha_step))
    ) / (2.0 * alpha_step)
    derivative_error = _relative_error(finite_difference, analytic_derivative)
    add("F08", "proof_target_check", "alpha_inverse_derivative_relative_error", derivative_error, derivative_tolerance, "Central finite difference checks dK_alpha^-1/dalpha.")

    all_passed = all(bool(row["passed"]) for row in rows)
    manifest_name = "method_formulation_manifest.json"
    ensure_new_output(output_dir, manifest_name)
    write_csv(output_dir / "method_formulation_checks.csv", rows, FIELDS)
    result = {
        "schema_version": SCHEMA,
        "status": "passed" if all_passed else "failed",
        "analysis_kind": "deterministic_numerical_reference_audit",
        "empirical_claim_eligible": False,
        "proof_status": "numerically_checked_not_formal_proof",
        "seed": seed,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "blocks": blocks,
        "check_count": len(rows),
        "all_checks_passed": all_passed,
        "numpy_version": np.__version__,
    }
    commit_manifest(output_dir, manifest_name, result, ["method_formulation_checks.csv"])
    if not all_passed:
        failures = [row["check_id"] for row in rows if not row["passed"]]
        raise ContractError(f"method formulation audit failed: {failures}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--input-dim", type=int, default=12)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--derivative-tolerance", type=float, default=2e-6)
    args = parser.parse_args()
    result = audit(
        args.output_dir.resolve(),
        args.seed,
        args.input_dim,
        args.output_dim,
        args.blocks,
        args.tolerance,
        args.derivative_tolerance,
    )
    print(f"method formulation audit passed: checks={result['check_count']}")


if __name__ == "__main__":
    main()
