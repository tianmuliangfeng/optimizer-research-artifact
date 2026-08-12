"""Numerically audit the coordinate equivariance retained by each routing policy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from common import ContractError, commit_manifest, ensure_new_output, write_csv


AUDIT_SCHEMA = "mdp_route_equivariance_audit_v1"
FIELDS = [
    "route",
    "transform",
    "relative_update_error",
    "tolerance",
    "expected_invariant",
    "check_passed",
]


def _polar(matrix: np.ndarray) -> np.ndarray:
    left, _, right_t = np.linalg.svd(matrix, full_matrices=False)
    return left @ right_t


def _preconditioner(k_matrix: np.ndarray, route: str, ridge: float, blocks: int) -> np.ndarray:
    dimension = k_matrix.shape[0]
    if route == "none":
        return np.eye(dimension)
    if route == "full":
        return np.linalg.inv(k_matrix + ridge * np.eye(dimension))
    if route == "diag":
        return np.diag(1.0 / (np.diag(k_matrix) + ridge))
    if route == "block":
        if dimension % blocks:
            raise ContractError("dimension must be divisible by blocks")
        width = dimension // blocks
        output = np.zeros_like(k_matrix)
        for index in range(blocks):
            section = slice(index * width, (index + 1) * width)
            output[section, section] = np.linalg.inv(
                k_matrix[section, section] + ridge * np.eye(width)
            )
        return output
    raise ContractError(f"unsupported route: {route}")


def _update(
    gradient: np.ndarray, k_matrix: np.ndarray, route: str, ridge: float, blocks: int
) -> np.ndarray:
    return _polar(gradient @ _preconditioner(k_matrix, route, ridge, blocks))


def _orthogonal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    matrix, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    return matrix


def _signed_permutation(rng: np.random.Generator, permutation: np.ndarray) -> np.ndarray:
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(permutation))
    return np.diag(signs) @ np.eye(len(permutation))[permutation]


def _relative_error(observed: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(observed - expected) / max(np.linalg.norm(expected), 1e-15))


def audit(
    output_dir: Path,
    seed: int = 20240731,
    input_dim: int = 12,
    output_dim: int = 8,
    blocks: int = 3,
    tolerance: float = 1e-9,
    cross_block_minimum_drift: float = 1e-5,
    synthetic: bool = False,
) -> dict[str, Any]:
    if input_dim % blocks:
        raise ContractError("input_dim must be divisible by blocks")
    rng = np.random.default_rng(seed)
    sample = rng.normal(size=(input_dim * 3, input_dim))
    k_matrix = sample.T @ sample / sample.shape[0] + 0.25 * np.eye(input_dim)
    gradient = rng.normal(size=(output_dim, input_dim))
    ridge = 0.15 * float(np.trace(k_matrix) / input_dim)
    arbitrary = _orthogonal(rng, input_dim)
    arbitrary_permutation = _signed_permutation(rng, rng.permutation(input_dim))
    width = input_dim // blocks
    within_indices = np.concatenate(
        [rng.permutation(np.arange(index * width, (index + 1) * width)) for index in range(blocks)]
    )
    block_preserving = _signed_permutation(rng, within_indices)
    block_diagonal_orthogonal = np.zeros((input_dim, input_dim))
    for index in range(blocks):
        section = slice(index * width, (index + 1) * width)
        block_diagonal_orthogonal[section, section] = _orthogonal(rng, width)
    block_order = rng.permutation(blocks)
    whole_block_indices = np.concatenate(
        [np.arange(index * width, (index + 1) * width) for index in block_order]
    )
    whole_block_permutation = np.eye(input_dim)[whole_block_indices]
    cross_indices = np.arange(input_dim).reshape(blocks, width).T.reshape(-1)
    cross_block = _signed_permutation(rng, cross_indices)

    checks = [
        ("full", "arbitrary_orthogonal", arbitrary, True),
        ("none", "arbitrary_orthogonal", arbitrary, True),
        ("diag", "signed_permutation", arbitrary_permutation, True),
        ("diag", "arbitrary_orthogonal", arbitrary, False),
        ("block", "block_preserving_signed_permutation", block_preserving, True),
        ("block", "block_diagonal_arbitrary_orthogonal", block_diagonal_orthogonal, True),
        ("block", "whole_block_permutation", whole_block_permutation, True),
        ("block", "cross_block_signed_permutation", cross_block, False),
    ]
    rows = []
    all_passed = True
    for route, transform_name, transform, expected_invariant in checks:
        baseline = _update(gradient, k_matrix, route, ridge, blocks)
        transformed_k = transform @ k_matrix @ transform.T
        transformed_gradient = gradient @ transform.T
        transformed_update = _update(transformed_gradient, transformed_k, route, ridge, blocks)
        error = _relative_error(transformed_update, baseline @ transform.T)
        passed = error <= tolerance if expected_invariant else error >= cross_block_minimum_drift
        all_passed = all_passed and passed
        rows.append(
            {
                "route": route,
                "transform": transform_name,
                "relative_update_error": error,
                "tolerance": tolerance if expected_invariant else cross_block_minimum_drift,
                "expected_invariant": expected_invariant,
                "check_passed": passed,
            }
        )

    manifest_name = "route_equivariance_manifest.json"
    ensure_new_output(output_dir, manifest_name)
    write_csv(output_dir / "route_equivariance.csv", rows, FIELDS)
    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed" if all_passed else "failed",
        "synthetic": synthetic,
        "claim_eligible": False,
        "numeric_reference_eligible": not synthetic and all_passed,
        "proof_status": "numerically_checked_not_formal_proof",
        "seed": seed,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "blocks": blocks,
        "invariance_tolerance": tolerance,
        "cross_block_minimum_drift": cross_block_minimum_drift,
        "all_checks_passed": all_passed,
        "numpy_version": np.__version__,
    }
    commit_manifest(output_dir, manifest_name, result, ["route_equivariance.csv"])
    if not all_passed:
        failures = [f"{row['route']}/{row['transform']}" for row in rows if not row["check_passed"]]
        raise ContractError(f"equivariance audit failed: {failures}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20240731)
    parser.add_argument("--input-dim", type=int, default=12)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--cross-block-minimum-drift", type=float, default=1e-5)
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(
        args.output_dir.resolve(),
        args.seed,
        args.input_dim,
        args.output_dim,
        args.blocks,
        args.tolerance,
        args.cross_block_minimum_drift,
        args.synthetic,
    )
    print(
        f"route-equivariance audit passed: checks=8 synthetic={result['synthetic']}"
    )


if __name__ == "__main__":
    main()
