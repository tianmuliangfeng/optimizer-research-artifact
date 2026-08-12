#!/usr/bin/env python3
"""Analyze two MECH-06 formal checkpoints without reading 1B training rankings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "2026-07-27.1"
LABELS = ("early", "late")
LAYERS = (0, 6, 12, 17)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-formal-dir", required=True, type=Path)
    parser.add_argument("--late-formal-dir", required=True, type=Path)
    parser.add_argument("--confirmation-contract", required=True, type=Path)
    parser.add_argument("--mech05-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def values(rows: Iterable[dict[str, str]], key: str) -> list[float]:
    result = [float(row[key]) for row in rows]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"invalid {key}")
    return result


def median(data: Iterable[float]) -> float:
    return float(statistics.median(list(data)))


def sd(data: Iterable[float]) -> float:
    items = list(data)
    return float(statistics.stdev(items)) if len(items) > 1 else 0.0


def contrast(
    rows: list[dict[str, str]],
    candidate: str,
    comparator: str,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    scores = {
        (
            int(row["layer"]),
            int(row["repeat"]),
            row["direction"],
            row["candidate"],
        ): float(row["best_relative_loss_delta"])
        for row in rows
        if row["scope"] == "layer"
    }
    cells = []
    for layer in LAYERS:
        for repeat in range(4):
            for direction in ("A_to_B", "B_to_A"):
                candidate_score = scores[(layer, repeat, direction, candidate)]
                comparator_score = scores[(layer, repeat, direction, comparator)]
                cells.append(
                    {
                        "layer": layer,
                        "advantage": comparator_score - candidate_score,
                        "candidate_score": candidate_score,
                        "comparator_score": comparator_score,
                    }
                )
    advantages = [row["advantage"] for row in cells]
    material_layers = 0
    for layer in LAYERS:
        subset = [row for row in cells if row["layer"] == layer]
        envelope = max(
            sd(row["candidate_score"] for row in subset),
            sd(row["comparator_score"] for row in subset),
        )
        mean_advantage = statistics.mean(row["advantage"] for row in subset)
        material_layers += mean_advantage > max(
            envelope, thresholds["relative_shadow_loss_margin"]
        )
    positive = sum(value > 0 for value in advantages)
    positive_fraction = positive / len(advantages)
    median_advantage = median(advantages)
    stable = (
        median_advantage > thresholds["relative_shadow_loss_margin"]
        and positive_fraction >= thresholds["minimum_positive_cell_fraction"]
        and material_layers >= thresholds["minimum_positive_material_layers"]
    )
    return {
        "candidate": candidate,
        "comparator": comparator,
        "cells": len(cells),
        "median_advantage": median_advantage,
        "mean_advantage": statistics.mean(advantages),
        "advantage_sd": sd(advantages),
        "positive_cells": positive,
        "positive_cell_fraction": positive_fraction,
        "positive_material_layers": material_layers,
        "stable_positive_advantage": stable,
        "noise_overwhelms_median": abs(median_advantage) <= sd(advantages),
    }


def analyze_checkpoint(
    label: str,
    directory: Path,
    contract_sha: str,
    mech05_sha: str,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(directory / "mech06_manifest.json")
    checks = read_json(directory / "checks.json")
    geometry = read_csv(directory / "geometry.csv")
    stability = read_csv(directory / "stability.csv")
    line_search = read_csv(directory / "line_search_summary.csv")
    expected_geometry = 18 * 4
    expected_stability = 18 * math.comb(4, 2)
    validation = {
        "manifest_passed": manifest.get("passed") is True,
        "checks_passed": bool(checks) and all(checks.values()),
        "label_matches": manifest.get("checkpoint_label") == label,
        "contract_matches": manifest.get("confirmation_contract_sha256")
        == contract_sha,
        "mech05_matches": manifest.get("mech05_contract_sha256") == mech05_sha,
        "geometry_rows": len(geometry) == expected_geometry,
        "stability_rows": len(stability) == expected_stability,
        "hvp_not_run": manifest.get("hvp_run") is False,
        "training_rankings_not_read": manifest.get(
            "existing_training_rankings_read"
        )
        is False,
    }
    if not all(validation.values()):
        raise RuntimeError(f"{label} input validation failed: {validation}")
    diag_ratio = median(values(geometry, "diag_p95_over_p05"))
    offdiag = median(values(geometry, "offdiag_energy_fraction"))
    diagonal_cosine = median(values(stability, "diagonal_cosine"))
    covariance_drift = median(values(stability, "covariance_relative_drift"))
    top_overlap = median(values(stability, "top_eigenspace_overlap"))
    anisotropic = (
        diag_ratio >= thresholds["minimum_diagonal_anisotropy_p95_over_p05"]
    )
    stable_geometry = (
        diagonal_cosine >= thresholds["minimum_diagonal_cosine"]
        and covariance_drift <= thresholds["maximum_covariance_relative_drift"]
    )
    stable_non_diagonal = (
        top_overlap >= thresholds["minimum_top_eigenspace_overlap"]
        and covariance_drift <= thresholds["maximum_covariance_relative_drift"]
    )
    near_scalar = (
        diag_ratio <= thresholds["maximum_scalar_isotropy_p95_over_p05"]
        and offdiag <= thresholds["maximum_scalar_isotropy_offdiag_energy_fraction"]
    )
    diag = contrast(line_search, "diag", "none", thresholds)
    full = contrast(line_search, "dense_full", "diag", thresholds)
    if stable_non_diagonal and full["stable_positive_advantage"]:
        signal = "full_or_block"
        reason = "stable non-diagonal geometry plus stable full-over-diag held-out gain"
    elif anisotropic and stable_geometry and diag["stable_positive_advantage"]:
        signal = "diag"
        reason = "stable diagonal anisotropy plus stable diag-over-none held-out gain"
    elif near_scalar:
        signal = "none_or_muon_sufficient"
        reason = "near-scalar geometry satisfies the frozen none branch without rankings"
    else:
        signal = "uncertain"
        reason = (
            "complex-K functional gate failed or required long-run evidence is withheld; "
            "geometry/noise alone cannot select none"
        )
    result = {
        "checkpoint_label": label,
        "checkpoint_step": int(manifest["checkpoint_step"]),
        "diagnostic_signal": signal,
        "reason": reason,
        "median_diag_p95_over_p05": diag_ratio,
        "median_offdiag_energy_fraction": offdiag,
        "median_diagonal_cosine": diagonal_cosine,
        "median_covariance_relative_drift": covariance_drift,
        "median_top_eigenspace_overlap": top_overlap,
        "diagonal_anisotropy_present": anisotropic,
        "geometry_stable": stable_geometry,
        "stable_non_diagonal_subspace": stable_non_diagonal,
        "near_scalar_isotropy": near_scalar,
        "diag_vs_none": diag,
        "dense_full_vs_diag": full,
        "existing_training_rankings_used": False,
    }
    source = {
        "directory": str(directory.resolve()),
        "manifest_sha256": sha256_file(directory / "mech06_manifest.json"),
        "geometry_sha256": sha256_file(directory / "geometry.csv"),
        "stability_sha256": sha256_file(directory / "stability.csv"),
        "line_search_sha256": sha256_file(directory / "line_search_summary.csv"),
        "validation": validation,
    }
    return result, source


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    confirmation = read_json(args.confirmation_contract)
    mech05 = read_json(args.mech05_contract)
    contract_sha = sha256_file(args.confirmation_contract)
    mech05_sha = sha256_file(args.mech05_contract)
    if contract_sha == "" or mech05_sha != confirmation["mech05_contract"]["sha256"]:
        raise RuntimeError("contract hash mismatch")
    thresholds = mech05["thresholds"]
    results, sources = {}, {}
    for label, directory in (
        ("early", args.early_formal_dir),
        ("late", args.late_formal_dir),
    ):
        results[label], sources[label] = analyze_checkpoint(
            label, directory, contract_sha, mech05_sha, thresholds
        )
    early_signal = results["early"]["diagnostic_signal"]
    late_signal = results["late"]["diagnostic_signal"]
    trajectory = {
        "early_signal": early_signal,
        "late_signal": late_signal,
        "same_signal": early_signal == late_signal,
        "scale_confirmation": (
            early_signal if early_signal == late_signal else "stage_dependent_or_uncertain"
        ),
        "existing_training_rankings_used": False,
        "retrospective_ranking_evaluation_required_separately": True,
    }
    write_json(args.output_dir / "checkpoint_diagnostic_features.json", results)
    write_json(args.output_dir / "trajectory_prediction.json", trajectory)
    write_json(
        args.output_dir / "input_validation.json",
        {
            "script_version": SCRIPT_VERSION,
            "confirmation_contract_sha256": contract_sha,
            "mech05_contract_sha256": mech05_sha,
            "thresholds": thresholds,
            "sources": sources,
            "existing_training_rankings_read": False,
        },
    )
    rows = []
    for label in LABELS:
        result = results[label]
        rows.append(
            {
                "checkpoint_label": label,
                "checkpoint_step": result["checkpoint_step"],
                "diagnostic_signal": result["diagnostic_signal"],
                "median_diag_p95_over_p05": result["median_diag_p95_over_p05"],
                "median_diagonal_cosine": result["median_diagonal_cosine"],
                "median_covariance_relative_drift": result[
                    "median_covariance_relative_drift"
                ],
                "median_top_eigenspace_overlap": result[
                    "median_top_eigenspace_overlap"
                ],
                "diag_median_advantage": result["diag_vs_none"]["median_advantage"],
                "diag_positive_cells": result["diag_vs_none"]["positive_cells"],
                "diag_material_layers": result["diag_vs_none"][
                    "positive_material_layers"
                ],
                "full_median_advantage": result["dense_full_vs_diag"][
                    "median_advantage"
                ],
                "full_positive_cells": result["dense_full_vs_diag"][
                    "positive_cells"
                ],
                "full_material_layers": result["dense_full_vs_diag"][
                    "positive_material_layers"
                ],
            }
        )
    write_csv(args.output_dir / "checkpoint_summary.csv", rows)
    report = [
        "# MECH-06 LLaMA-1B diagnostic confirmation",
        "",
        f"Confirmation contract SHA-256: `{contract_sha}`  ",
        f"MECH-05 contract SHA-256: `{mech05_sha}`",
        "",
        "Existing LLaMA-1B training rankings were not read by this analyzer.",
        "Any comparison with those rankings is retrospective and must be reported separately.",
        "",
        "| checkpoint | step | diagnostic signal | diag positive | full positive |",
        "|---|---:|---|---:|---:|",
    ]
    for label in LABELS:
        result = results[label]
        report.append(
            f"| {label} | {result['checkpoint_step']} | "
            f"`{result['diagnostic_signal']}` | "
            f"{result['diag_vs_none']['positive_cells']}/32 | "
            f"{result['dense_full_vs_diag']['positive_cells']}/32 |"
        )
    report.extend(
        [
            "",
            f"Trajectory result: `{trajectory['scale_confirmation']}`.",
            "",
        ]
    )
    (args.output_dir / "MECH06_LLAMA1B_DIAGNOSTIC_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    output_files = sorted(path for path in args.output_dir.iterdir() if path.is_file())
    write_json(
        args.output_dir / "mech06_analysis_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "passed": True,
            "confirmation_contract_sha256": contract_sha,
            "mech05_contract_sha256": mech05_sha,
            "trajectory_prediction": trajectory,
            "existing_training_rankings_read": False,
            "output_sha256": {
                path.name: sha256_file(path) for path in output_files
            },
        },
    )
    print(f"MECH-06 analysis manifest: {args.output_dir / 'mech06_analysis_manifest.json'}")
    print(f"MECH-06 artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
