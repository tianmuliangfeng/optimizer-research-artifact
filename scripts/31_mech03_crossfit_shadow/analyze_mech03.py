#!/usr/bin/env python3
"""Aggregate frozen MECH-03 formal cross-fit shadow-loss results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-27.2"
EXPECTED_RUNNER_VERSION = "2026-07-27.2"
HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "prediction_contract.json"
FAMILIES = ("r1", "gpt_bridge", "llama124")
LAYERS = (0, 4, 8, 11)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
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


def validate_formal(
    directory: Path, expected_family: str, contract_sha256: str
) -> dict[str, Any]:
    directory = directory.resolve()
    manifest = read_json(directory / "mech03_manifest.json")
    checks = read_json(directory / "checks.json")
    batch = read_json(directory / "batch_contract.json")
    required = [
        "line_search_summary.csv",
        "shadow_losses.csv",
        "update_geometry.csv",
        "state_invariance.json",
        "prediction_contract.json",
    ]
    validation = {
        "directory": str(directory),
        "required_files_present": all((directory / name).is_file() for name in required),
        "manifest_passed": manifest.get("passed") is True,
        "analysis_tier_formal": manifest.get("analysis_tier") == "formal",
        "family_matches": manifest.get("family") == expected_family,
        "runner_version_matches": (
            manifest.get("script_version") == EXPECTED_RUNNER_VERSION
        ),
        "contract_sha256_matches": (
            manifest.get("prediction_contract_sha256") == contract_sha256
            and sha256_file(directory / "prediction_contract.json") == contract_sha256
        ),
        "all_checks_passed": bool(checks) and all(checks.values()),
        "batch_contract_complete": (
            batch.get("repeats") == 4
            and batch.get("batches_per_split") == 8
            and batch.get("all_windows_disjoint") is True
        ),
        "batch_contract_sha256": batch.get("contract_sha256", ""),
    }
    validation["passed"] = all(
        value for key, value in validation.items() if key not in {"directory", "passed"}
    )
    if not validation["passed"]:
        raise RuntimeError(f"formal artifact rejected: {validation}")
    return validation


def family_scores(directory: Path, family: str) -> list[dict[str, Any]]:
    rows = read_csv(directory / "line_search_summary.csv")
    indexed: dict[tuple[int, str, int, str], float] = {}
    for row in rows:
        if row["scope"] != "layer":
            continue
        key = (
            int(row["repeat"]),
            row["direction"],
            int(row["layer"]),
            row["candidate"],
        )
        indexed[key] = float(row["best_relative_loss_delta"])
    scores = []
    for repeat in range(4):
        for direction in ("A_to_B", "B_to_A"):
            for layer in LAYERS:
                diag = indexed[(repeat, direction, layer, "diag")]
                none = indexed[(repeat, direction, layer, "none")]
                scores.append(
                    {
                        "family": family,
                        "repeat": repeat,
                        "direction": direction,
                        "layer": layer,
                        "diag_best_relative_loss_delta": diag,
                        "none_best_relative_loss_delta": none,
                        "diag_minus_none": diag - none,
                    }
                )
    return scores


def index_scores(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str, int], dict[str, Any]]:
    return {
        (
            str(row["family"]),
            int(row["repeat"]),
            str(row["direction"]),
            int(row["layer"]),
        ): row
        for row in rows
    }


def primary_gate(
    scores: list[dict[str, Any]], contract: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    indexed = index_scores(scores)
    paired = []
    for repeat in range(4):
        for direction in ("A_to_B", "B_to_A"):
            for layer in LAYERS:
                gpt = indexed[("gpt_bridge", repeat, direction, layer)]
                llama = indexed[("llama124", repeat, direction, layer)]
                delta = float(llama["diag_minus_none"] - gpt["diag_minus_none"])
                paired.append(
                    {
                        "repeat": repeat,
                        "direction": direction,
                        "layer": layer,
                        "gpt_bridge_diag_minus_none": gpt["diag_minus_none"],
                        "llama124_diag_minus_none": llama["diag_minus_none"],
                        "llama_minus_gpt": delta,
                        "predicted_direction": delta > 0,
                    }
                )
    margin = float(
        contract["primary_gate"]["minimum_relative_loss_margin"]
    )
    layer_rows = []
    for layer in LAYERS:
        gpt_values = [
            float(row["diag_minus_none"])
            for row in scores
            if row["family"] == "gpt_bridge" and row["layer"] == layer
        ]
        llama_values = [
            float(row["diag_minus_none"])
            for row in scores
            if row["family"] == "llama124" and row["layer"] == layer
        ]
        gpt_mean = statistics.fmean(gpt_values)
        llama_mean = statistics.fmean(llama_values)
        gpt_sd = statistics.stdev(gpt_values)
        llama_sd = statistics.stdev(llama_values)
        cross = llama_mean - gpt_mean
        threshold = max(gpt_sd, llama_sd, margin)
        layer_rows.append(
            {
                "layer": layer,
                "gpt_bridge_mean_diag_minus_none": gpt_mean,
                "gpt_bridge_sd": gpt_sd,
                "llama124_mean_diag_minus_none": llama_mean,
                "llama124_sd": llama_sd,
                "llama_minus_gpt": cross,
                "material_threshold": threshold,
                "positive_material": cross > threshold,
            }
        )
    positive_cells = sum(bool(row["predicted_direction"]) for row in paired)
    positive_material_layers = sum(
        bool(row["positive_material"]) for row in layer_rows
    )
    median_cross = statistics.median(
        float(row["llama_minus_gpt"]) for row in paired
    )
    rules = {
        "paired_direction_consistency": (
            positive_cells
            >= int(contract["primary_gate"]["minimum_positive_paired_cells"])
        ),
        "material_layer_count": (
            positive_material_layers
            >= int(contract["primary_gate"]["minimum_positive_material_layers"])
        ),
        "aggregate_practical_margin": median_cross > margin,
    }
    gate = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "primary_contrast": "llama124_minus_gpt_bridge_same_host",
        "positive_paired_cells": positive_cells,
        "paired_cells_total": len(paired),
        "positive_material_layers": positive_material_layers,
        "layers_total": len(layer_rows),
        "median_paired_cross_architecture_score": median_cross,
        "minimum_relative_loss_margin": margin,
        "rules": rules,
        "prediction_gate_passed": all(rules.values()),
        "mech04_authorized": False,
        "authorization_note": (
            "MECH-03 never auto-authorizes MECH-04; a reviewed mechanism "
            "decision is required."
        ),
    }
    return paired, layer_rows, gate


def runtime_robustness(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = index_scores(scores)
    rows = []
    for repeat in range(4):
        for direction in ("A_to_B", "B_to_A"):
            for layer in LAYERS:
                native = indexed[("r1", repeat, direction, layer)]
                bridge = indexed[("gpt_bridge", repeat, direction, layer)]
                rows.append(
                    {
                        "repeat": repeat,
                        "direction": direction,
                        "layer": layer,
                        "r1_native_diag_minus_none": native["diag_minus_none"],
                        "gpt_bridge_diag_minus_none": bridge["diag_minus_none"],
                        "gpt_bridge_minus_r1_native": (
                            bridge["diag_minus_none"] - native["diag_minus_none"]
                        ),
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    contract = read_json(CONTRACT_PATH)
    contract_sha256 = sha256_file(CONTRACT_PATH)
    validations = {
        "r1": validate_formal(args.r1_formal, "r1", contract_sha256),
        "gpt_bridge": validate_formal(
            args.gpt_bridge_formal, "gpt_bridge", contract_sha256
        ),
        "llama124": validate_formal(
            args.llama124_formal, "llama124", contract_sha256
        ),
    }
    batch_contracts = {
        str(row["batch_contract_sha256"]) for row in validations.values()
    }
    if len(batch_contracts) != 1:
        raise RuntimeError(
            f"cross-host token batch contracts differ: {sorted(batch_contracts)}"
        )
    scores = []
    scores.extend(family_scores(args.r1_formal.resolve(), "r1"))
    scores.extend(
        family_scores(args.gpt_bridge_formal.resolve(), "gpt_bridge")
    )
    scores.extend(family_scores(args.llama124_formal.resolve(), "llama124"))
    paired, layers, gate = primary_gate(scores, contract)
    robustness = runtime_robustness(scores)
    write_csv(output / "family_cell_scores.csv", scores)
    write_csv(output / "primary_paired_cells.csv", paired)
    write_csv(output / "primary_layer_summary.csv", layers)
    write_csv(output / "gpt_runtime_robustness.csv", robustness)
    atomic_json(output / "input_validation.json", validations)
    atomic_json(output / "prediction_gate.json", gate)
    atomic_json(
        output / "analysis_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "passed": all(row["passed"] for row in validations.values()),
            "prediction_gate_passed": gate["prediction_gate_passed"],
            "mech04_authorized": False,
            "prediction_contract_sha256": contract_sha256,
            "artifacts": sorted(path.name for path in output.iterdir()),
        },
    )


if __name__ == "__main__":
    main()
