#!/usr/bin/env python3
"""Freeze the exact MECH-08 control files consumed by MECH-09."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-28.1"
CELLS = (
    "early_muon",
    "early_newton_full",
    "late_muon",
    "late_newton_full",
)
ALGORITHMS = (
    "muon",
    "original_newton_muon",
    "selective_diag",
    "selective_none",
)
REPLICAS = (0, 1, 2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def add_file(
    rows: list[dict[str, Any]], run: Path, relative: Path, role: str
) -> None:
    path = run / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    rows.append(
        {
            "relative_path": relative.as_posix(),
            "role": role,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    )


def build(run: Path) -> dict[str, Any]:
    analysis_manifest = run / "analysis" / "mech08_analysis_manifest.json"
    analysis = read_json(analysis_manifest)
    if analysis.get("passed") is not True:
        raise RuntimeError(f"MECH-08 analysis did not pass: {analysis_manifest}")

    rows: list[dict[str, Any]] = []
    add_file(
        rows,
        run,
        Path("analysis/mech08_analysis_manifest.json"),
        "analysis_manifest",
    )
    add_file(
        rows,
        run,
        Path("analysis/paired_contrasts.csv"),
        "analysis_paired_contrasts",
    )
    for cell in CELLS:
        add_file(
            rows,
            run,
            Path("checkpoint_hashes") / f"{cell}.json",
            "checkpoint_hash_certificate",
        )
    for cell in CELLS:
        for algorithm in ALGORITHMS:
            for replica in REPLICAS:
                base = (
                    Path("formal")
                    / cell
                    / algorithm
                    / f"replica_{replica}"
                )
                add_file(rows, run, base / "mech08_manifest.json", "worker_manifest")
                add_file(rows, run, base / "checks.json", "worker_checks")
                add_file(rows, run, base / "evaluation.csv", "evaluation")
                if algorithm == "original_newton_muon":
                    add_file(rows, run, base / "training.csv", "production_training")

    return {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": "MECH-09",
        "source_experiment": "MECH-08",
        "source_run_id": run.name,
        "source_run_path": (
            "${SNM_RESULTS_ROOT}/"
            f"36_mech08_short_horizon_rollout/{run.name}"
        ),
        "mech08_contract_sha256": analysis["contract_sha256"],
        "mech08_analysis_version": analysis["script_version"],
        "cells": list(CELLS),
        "algorithms": list(ALGORITHMS),
        "replicas": list(REPLICAS),
        "files": rows,
        "file_count": len(rows),
        "passed": len(rows) == 162,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mech08-run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = build(args.mech08_run_dir.resolve())
    if not payload["passed"]:
        raise SystemExit(f"unexpected reference file count: {payload['file_count']}")
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"MECH-09 MECH-08 control reference: {args.output.resolve()}")
