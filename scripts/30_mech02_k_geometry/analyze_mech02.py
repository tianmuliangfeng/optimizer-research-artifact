#!/usr/bin/env python3
"""Aggregate frozen MECH-02 formal geometry without importing PyTorch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-27.1"
EXPECTED_RUNNER_VERSION = "2026-07-27.1"
PRIMARY_METRICS = (
    "diag_cv",
    "diag_p95_over_p05",
    "offdiag_energy_fraction",
    "log_damped_condition_number",
    "damped_effective_rank_fraction",
    "damped_top1_mass",
    "damped_top10_mass",
    "mean_top_eigenspace_overlap",
    "mean_covariance_relative_drift",
)
MIN_MATERIAL_SAME_DIRECTION_LAYERS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-formal", type=Path, required=True)
    parser.add_argument("--gpt-bridge-formal", type=Path, required=True)
    parser.add_argument("--llama124-formal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty analysis table: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_formal(directory: Path, expected_family: str) -> dict[str, Any]:
    directory = directory.resolve()
    manifest = read_json(directory / "mech02_manifest.json")
    checks = read_json(directory / "checks.json")
    required_files = [
        "geometry.csv",
        "stability.csv",
        "geometry_contract.json",
        "batch_contract.json",
        "state_invariance.json",
    ]
    validation = {
        "directory": str(directory),
        "required_files_present": all((directory / name).is_file() for name in required_files),
        "manifest_passed": manifest.get("passed") is True,
        "analysis_tier_formal": manifest.get("analysis_tier") == "formal",
        "family_matches": manifest.get("family") == expected_family,
        "runner_version_matches": manifest.get("script_version")
        == EXPECTED_RUNNER_VERSION,
        "all_checks_passed": bool(checks) and all(checks.values()),
    }
    validation["passed"] = all(
        value for key, value in validation.items() if key not in {"directory", "passed"}
    )
    if not validation["passed"]:
        raise RuntimeError(f"formal artifact rejected: {validation}")
    return validation


def derived_geometry(row: dict[str, str]) -> dict[str, float]:
    width = float(row["activation_width"])
    return {
        "diag_cv": float(row["diag_cv"]),
        "diag_p95_over_p05": float(row["diag_p95_over_p05"]),
        "offdiag_energy_fraction": float(row["offdiag_energy_fraction"]),
        "log_damped_condition_number": math.log(
            max(float(row["damped_condition_number"]), 1.0)
        ),
        "damped_effective_rank_fraction": float(
            float(row["damped_effective_rank"]) / width
        ),
        "damped_top1_mass": float(row["damped_top1_mass"]),
        "damped_top10_mass": float(row["damped_top10_mass"]),
    }


def family_layer_summary(directory: Path, family: str) -> list[dict[str, Any]]:
    geometry = read_csv(directory / "geometry.csv")
    stability = read_csv(directory / "stability.csv")
    by_layer_metric: dict[tuple[int, str], list[float]] = {}
    for row in geometry:
        layer = int(row["layer"])
        for metric, value in derived_geometry(row).items():
            by_layer_metric.setdefault((layer, metric), []).append(value)
    stability_values: dict[tuple[int, str], list[float]] = {}
    for row in stability:
        layer = int(row["layer"])
        stability_values.setdefault(
            (layer, "mean_top_eigenspace_overlap"), []
        ).append(float(row["top_eigenspace_overlap"]))
        stability_values.setdefault(
            (layer, "mean_covariance_relative_drift"), []
        ).append(float(row["covariance_relative_drift"]))
    rows: list[dict[str, Any]] = []
    for (layer, metric), values in sorted(
        {**by_layer_metric, **stability_values}.items()
    ):
        rows.append(
            {
                "family": family,
                "layer": layer,
                "metric": metric,
                "mean": statistics.fmean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        )
    return rows


def index_summary(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {
        (str(row["family"]), int(row["layer"]), str(row["metric"])): row
        for row in rows
    }


def primary_comparisons(
    summary: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexed = index_summary(summary)
    rows = []
    metric_gate_rows = []
    for metric in PRIMARY_METRICS:
        signs = []
        material_same_direction_candidates = []
        for layer in range(12):
            gpt = indexed[("gpt_bridge", layer, metric)]
            llama = indexed[("llama124", layer, metric)]
            delta = float(llama["mean"] - gpt["mean"])
            noise = max(float(gpt["sd"]), float(llama["sd"]), 1e-12)
            material = abs(delta) > noise
            sign = 1 if delta > 0 else (-1 if delta < 0 else 0)
            if material:
                signs.append(sign)
            row = {
                "metric": metric,
                "layer": layer,
                "gpt_bridge_mean": gpt["mean"],
                "gpt_bridge_sd": gpt["sd"],
                "llama124_mean": llama["mean"],
                "llama124_sd": llama["sd"],
                "llama_minus_gpt": delta,
                "repeat_sd_envelope": noise,
                "absolute_delta_over_sd_envelope": abs(delta) / noise,
                "material_vs_repeat_sd": material,
                "direction": sign,
            }
            rows.append(row)
        positive = sum(sign > 0 for sign in signs)
        negative = sum(sign < 0 for sign in signs)
        same_direction_material = max(positive, negative)
        metric_gate_rows.append(
            {
                "metric": metric,
                "material_layers": len(signs),
                "positive_material_layers": positive,
                "negative_material_layers": negative,
                "same_direction_material_layers": same_direction_material,
                "passed": same_direction_material
                >= MIN_MATERIAL_SAME_DIRECTION_LAYERS,
            }
        )
    candidate_pass = any(row["passed"] for row in metric_gate_rows)
    gate = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "primary_contrast": "llama124_minus_gpt_bridge_same_host",
        "minimum_material_same_direction_layers": MIN_MATERIAL_SAME_DIRECTION_LAYERS,
        "metric_gates": metric_gate_rows,
        "geometry_gate_candidate_passed": candidate_pass,
        "mech03_authorized": False,
        "authorization_blocker": (
            "Freeze an explicit held-out diag-none prediction contract before "
            "MECH-03; geometry alone never auto-authorizes MECH-03."
        ),
    }
    return rows, gate


def gpt_runtime_robustness(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = index_summary(summary)
    rows = []
    for metric in PRIMARY_METRICS:
        for layer in range(12):
            native = indexed[("r1", layer, metric)]
            bridge = indexed[("gpt_bridge", layer, metric)]
            delta = float(bridge["mean"] - native["mean"])
            noise = max(float(native["sd"]), float(bridge["sd"]), 1e-12)
            rows.append(
                {
                    "metric": metric,
                    "layer": layer,
                    "gpt_bridge_minus_r1_native": delta,
                    "repeat_sd_envelope": noise,
                    "absolute_delta_over_sd_envelope": abs(delta) / noise,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    validations = {
        "r1": validate_formal(args.r1_formal, "r1"),
        "gpt_bridge": validate_formal(args.gpt_bridge_formal, "gpt_bridge"),
        "llama124": validate_formal(args.llama124_formal, "llama124"),
    }
    summary = []
    summary.extend(family_layer_summary(args.r1_formal.resolve(), "r1"))
    summary.extend(
        family_layer_summary(args.gpt_bridge_formal.resolve(), "gpt_bridge")
    )
    summary.extend(
        family_layer_summary(args.llama124_formal.resolve(), "llama124")
    )
    primary, gate = primary_comparisons(summary)
    robustness = gpt_runtime_robustness(summary)
    write_csv(output / "family_layer_summary.csv", summary)
    write_csv(output / "primary_cross_architecture.csv", primary)
    write_csv(output / "gpt_runtime_robustness.csv", robustness)
    atomic_json(output / "input_validation.json", validations)
    atomic_json(output / "geometry_gate.json", gate)
    atomic_json(
        output / "analysis_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "passed": all(row["passed"] for row in validations.values()),
            "geometry_gate_candidate_passed": gate[
                "geometry_gate_candidate_passed"
            ],
            "mech03_authorized": False,
            "artifacts": sorted(path.name for path in output.iterdir()),
        },
    )


if __name__ == "__main__":
    main()

