#!/usr/bin/env python3
"""Independently validate a completed experiment-39 LR analysis bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_ROLES = {
    "muon",
    "original_newton_muon",
    "selective_none",
    "selective_diag",
}
EXPECTED_MULTIPLIERS = {0.8, 1.0, 1.2}
EXPECTED_ANALYSIS_VERSION = "2026-07-29.7"
EXPECTED_ARTIFACTS = {
    "lr_sensitivity_runs.csv",
    "lr_sensitivity_contrasts.csv",
    "lr_sensitivity_role_summary.csv",
    "source_manifest.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_output(output_dir: Path, contract_path: Path) -> dict[str, Any]:
    manifest_path = output_dir / "lr_sensitivity_manifest.json"
    manifest = read_json(manifest_path)
    contract = read_json(contract_path)
    if (
        manifest.get("passed") is not True
        or manifest.get("script_version") != EXPECTED_ANALYSIS_VERSION
        or manifest.get("evidence_class") != "supporting_only"
        or manifest.get("tuned_best_claim_allowed") is not False
        or manifest.get("diag_vs_none_primary") is not False
        or set(manifest.get("methods", ())) != EXPECTED_ROLES
        or set(manifest.get("multipliers", ())) != EXPECTED_MULTIPLIERS
        or manifest.get("seed") != contract["seed"]
        or manifest.get("budget_steps") != contract["budget_steps"]
        or manifest.get("warmdown_steps") != contract["warmdown_steps"]
        or manifest.get("run_cells") != 12
        or manifest.get("contract_sha256") != sha256_file(contract_path)
    ):
        raise RuntimeError("LR-sensitivity manifest acceptance failed")
    artifacts = set(manifest.get("artifacts", ()))
    hashes = manifest.get("output_sha256", {})
    if artifacts != EXPECTED_ARTIFACTS or set(hashes) != EXPECTED_ARTIFACTS:
        raise RuntimeError("LR-sensitivity artifact coverage failed")
    for name, expected in hashes.items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"LR-sensitivity output hash mismatch: {name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    manifest = validate_output(
        args.output_dir.expanduser().resolve(),
        args.contract.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "passed": True,
                "run_cells": manifest["run_cells"],
                "hashes_checked": len(manifest["output_sha256"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
