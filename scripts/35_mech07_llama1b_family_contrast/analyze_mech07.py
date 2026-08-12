#!/usr/bin/env python3
"""Aggregate the eight MECH-07 formal checkpoint diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-27.3"
CONTRASTS = (
    ("selective_diag_vs_muon", "selective_diag", "muon", "primary"),
    ("selective_none_vs_muon", "selective_none", "muon", "primary"),
    (
        "selective_diag_vs_original_newton_muon",
        "selective_diag",
        "original_newton_muon",
        "primary",
    ),
    (
        "selective_none_vs_original_newton_muon",
        "selective_none",
        "original_newton_muon",
        "primary",
    ),
    (
        "original_newton_muon_vs_muon",
        "original_newton_muon",
        "muon",
        "baseline",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty rows: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def summarize(values: list[float]) -> dict[str, Any]:
    margin = 1e-6
    negative = sum(value < -margin for value in values)
    positive = sum(value > margin for value in values)
    return {
        "cells": len(values),
        "mean_delta": statistics.mean(values),
        "median_delta": statistics.median(values),
        "sd_delta": statistics.stdev(values) if len(values) > 1 else 0.0,
        "negative_cells_left_better": negative,
        "positive_cells_left_worse": positive,
        "near_zero_cells": len(values) - negative - positive,
        "negative_fraction_left_better": negative / len(values),
        "positive_fraction_left_worse": positive / len(values),
    }


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    contract = read_json(args.contract.resolve())
    output = run / "analysis"
    output.mkdir(exist_ok=False)
    threshold = float(
        contract["analysis_thresholds"]["minimum_directional_cell_fraction"]
    )

    observations: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for spec in contract["checkpoints"]:
        formal = run / spec["cell"] / "formal"
        manifest = read_json(formal / "mech07_manifest.json")
        checks = read_json(formal / "checks.json")
        if manifest.get("passed") is not True or not checks or not all(checks.values()):
            raise RuntimeError(f"formal cell failed validation: {spec['cell']}")
        if (
            manifest.get("checkpoint_method") != spec["method"]
            or manifest.get("checkpoint_stage") != spec["stage"]
            or int(manifest.get("checkpoint_step")) != int(spec["step"])
        ):
            raise RuntimeError(f"formal identity mismatch: {spec['cell']}")
        rows = [
            row
            for row in read_csv(formal / "line_search_summary.csv")
            if row["scope"] == "all"
        ]
        expected_rows = int(manifest["repeats"]) * 2 * 4
        if len(rows) != expected_rows:
            raise RuntimeError(
                f"{spec['cell']}: summary rows {len(rows)} != {expected_rows}"
            )
        scores = {
            (
                int(row["repeat"]),
                row["direction"],
                row["algorithm"],
            ): float(row["best_relative_loss_delta"])
            for row in rows
        }
        for name, left, right, priority in CONTRASTS:
            values = []
            for repeat in range(int(manifest["repeats"])):
                for direction in ("A_to_B", "B_to_A"):
                    delta = (
                        scores[(repeat, direction, left)]
                        - scores[(repeat, direction, right)]
                    )
                    values.append(delta)
                    observations.append(
                        {
                            "checkpoint_cell": spec["cell"],
                            "checkpoint_stage": spec["stage"],
                            "checkpoint_method": spec["method"],
                            "checkpoint_step": spec["step"],
                            "priority": priority,
                            "contrast": name,
                            "left_algorithm": left,
                            "right_algorithm": right,
                            "repeat": repeat,
                            "direction": direction,
                            "relative_shadow_loss_delta_left_minus_right": delta,
                        }
                    )
            stats = summarize(values)
            checkpoint_summaries.append(
                {
                    "checkpoint_cell": spec["cell"],
                    "checkpoint_stage": spec["stage"],
                    "checkpoint_method": spec["method"],
                    "checkpoint_step": spec["step"],
                    "priority": priority,
                    "contrast": name,
                    **stats,
                    "stable_left_better": (
                        stats["median_delta"] < -1e-6
                        and stats["negative_fraction_left_better"] >= threshold
                    ),
                    "stable_left_worse": (
                        stats["median_delta"] > 1e-6
                        and stats["positive_fraction_left_worse"] >= threshold
                    ),
                }
            )
        source_rows.append(
            {
                "checkpoint_cell": spec["cell"],
                "manifest_sha256": sha256_file(formal / "mech07_manifest.json"),
                "summary_sha256": sha256_file(formal / "line_search_summary.csv"),
                "checks_passed": True,
            }
        )

    # Deliberate guard against reverting to the old priority.
    if any(
        {row["left_algorithm"], row["right_algorithm"]}
        == {"selective_diag", "selective_none"}
        for row in observations
    ):
        raise RuntimeError("diag-vs-none entered the MECH-07 primary analysis")

    stage_summaries: list[dict[str, Any]] = []
    for stage in ("early", "late"):
        for name, left, right, priority in CONTRASTS:
            values = [
                float(row["relative_shadow_loss_delta_left_minus_right"])
                for row in observations
                if row["checkpoint_stage"] == stage and row["contrast"] == name
            ]
            state_rows = [
                row
                for row in checkpoint_summaries
                if row["checkpoint_stage"] == stage and row["contrast"] == name
            ]
            stats = summarize(values)
            stage_summaries.append(
                {
                    "checkpoint_stage": stage,
                    "priority": priority,
                    "contrast": name,
                    "left_algorithm": left,
                    "right_algorithm": right,
                    **stats,
                    "checkpoint_states_left_better": sum(
                        bool(row["stable_left_better"]) for row in state_rows
                    ),
                    "checkpoint_states_left_worse": sum(
                        bool(row["stable_left_worse"]) for row in state_rows
                    ),
                    "checkpoint_states_total": len(state_rows),
                }
            )

    write_csv(output / "family_contrasts_by_cell.csv", observations)
    write_csv(output / "checkpoint_contrast_summary.csv", checkpoint_summaries)
    write_csv(output / "stage_contrast_summary.csv", stage_summaries)
    write_csv(output / "input_sources.csv", source_rows)

    report = [
        "# MECH-07 LLaMA-1B family-level local diagnostic",
        "",
        "This report compares each Selective proposal separately with Muon and "
        "the original Newton–Muon baseline. `diag vs none` is not a primary contrast.",
        "",
        "| stage | contrast | median relative shadow delta | left-better cells | checkpoint states left-better |",
        "|---|---|---:|---:|---:|",
    ]
    for row in stage_summaries:
        report.append(
            f"| {row['checkpoint_stage']} | `{row['contrast']}` | "
            f"{float(row['median_delta']):+.8e} | "
            f"{row['negative_cells_left_better']}/{row['cells']} | "
            f"{row['checkpoint_states_left_better']}/"
            f"{row['checkpoint_states_total']} |"
        )
    report.extend(
        [
            "",
            "Negative delta favors the left algorithm. These are matched local "
            "counterfactual updates with fresh build-split covariance and shared "
            "checkpoint momentum. They do not replace the existing three-seed "
            "long-run family ranking and do not establish cross-checkpoint causality.",
            "",
        ]
    )
    report_path = output / "MECH07_FAMILY_CONTRAST_REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    artifacts = [
        "MECH07_FAMILY_CONTRAST_REPORT.md",
        "checkpoint_contrast_summary.csv",
        "family_contrasts_by_cell.csv",
        "input_sources.csv",
        "stage_contrast_summary.csv",
    ]
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "passed": True,
        "contract_sha256": sha256_file(args.contract.resolve()),
        "formal_cells": len(contract["checkpoints"]),
        "primary_contrasts": 4,
        "baseline_contrasts": 1,
        "diag_vs_none_primary": False,
        "artifacts": artifacts,
        "output_sha256": {
            name: sha256_file(output / name) for name in artifacts
        },
    }
    write_json(output / "mech07_analysis_manifest.json", manifest)
    print(f"MECH-07 analysis manifest: {output / 'mech07_analysis_manifest.json'}")
    print(f"MECH-07 analysis artifacts: {output}")


if __name__ == "__main__":
    main()
