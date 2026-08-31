#!/usr/bin/env python3
"""Build and independently verify Experiment 57 Moonlight endpoint evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any

import protocol as P


PACKAGE_REL = Path("scripts/57_llama1b_10b_moonlight")
BUDGET_PHASES = {
    "tokens_3p2506b": ("backbone_4400", "cooldown_6200"),
    "tokens_6p9694b": ("backbone_4400", "backbone_11493", "cooldown_13293"),
    "tokens_approximately_10b": (
        "backbone_4400", "backbone_11493", "backbone_17273", "cooldown_19073",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--full-checkpoint-hash", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def validation_rows(paths: list[Path]) -> list[tuple[int, float]]:
    by_step: dict[int, float] = {}
    for path in paths:
        for row in P.read_metrics(path):
            if row.get("event") == "val":
                by_step[int(row["step"])] = float(row["loss"])
    return sorted(by_step.items())


def curve_stats(paths: list[Path]) -> tuple[float, float]:
    rows = validation_rows(paths)
    if not rows:
        raise RuntimeError("EX57 validation curve is empty")
    tail = statistics.fmean(loss for _, loss in rows[-5:])
    if len(rows) == 1 or rows[-1][0] == rows[0][0]:
        auc = rows[-1][1]
    else:
        area = sum(
            (right[0] - left[0]) * (left[1] + right[1]) / 2.0
            for left, right in zip(rows, rows[1:])
        )
        auc = area / (rows[-1][0] - rows[0][0])
    return float(tail), float(auc)


def expected_hparams(contract: dict[str, Any]) -> dict[str, Any]:
    spec = contract["moonlight"]
    return {
        "momentum": float(spec["momentum"]),
        "nesterov": bool(spec["nesterov"]),
        "ns_steps": int(spec["newton_schulz_steps"]),
        "weight_decay": float(spec["weight_decay"]),
    }


def moonlight_state(summary: dict[str, Any], contract: dict[str, Any]) -> dict[str, int]:
    schema = summary.get("moonlight_state_schema", {})
    expected_matrices = int(contract["profiles"]["1b"]["expected_matrix_tensors"])
    momentum_bytes = int(summary.get("momentum_buffer_bytes", 0))
    matrix_bytes = int(summary.get("moonlight_matrix_optimizer_state_bytes", 0))
    checks = {
        "hparams": summary.get("moonlight_hyperparameters") == expected_hparams(contract),
        "optimizer": isinstance(schema, dict) and schema.get("optimizer") == "R1MoonlightMuon",
        "state_keys": isinstance(schema, dict) and schema.get("tensor_state_keys") == ["momentum_buffer"],
        "matrices": isinstance(schema, dict)
        and int(schema.get("logical_matrix_parameters", -1)) == expected_matrices,
        "no_k": isinstance(schema, dict) and schema.get("contains_activation_k_state") is False,
        "no_factors": isinstance(schema, dict)
        and schema.get("contains_factor_or_eigendecomposition_state") is False,
        "bytes": momentum_bytes > 0 and matrix_bytes == momentum_bytes,
    }
    if not all(checks.values()):
        raise RuntimeError(f"EX57 Moonlight state audit failed: {checks}")
    return {
        "moonlight_momentum_state_bytes": momentum_bytes,
        "moonlight_matrix_optimizer_state_bytes": matrix_bytes,
    }


def collect_endpoints(run_dir: Path, contract: dict[str, Any]) -> list[dict[str, Any]]:
    phase_by_budget = {phase["budget_id"]: phase for phase in P.endpoint_phases(contract)}
    rows: list[dict[str, Any]] = []
    for seed in contract["formal"]["seeds"]:
        unit = run_dir / "formal/1b" / f"seed{seed}"
        unit_manifest = P.read_json(unit / "unit_manifest.json")
        for budget in contract["formal"]["accepted_1b_budget_ids"]:
            phase = phase_by_budget[budget]
            directory = unit / phase["id"]
            summary = P.read_json(directory / "summary.json")
            tail, auc = curve_stats([unit / phase_id / "metrics.csv" for phase_id in BUDGET_PHASES[budget]])
            checkpoint = unit_manifest["endpoints"][budget]
            rows.append({
                "scale": "1b",
                "budget_id": budget,
                "target_step": int(phase["target_step"]),
                "seed": int(seed),
                "method": "moonlight",
                "final_val_loss": float(summary["final_val_loss"]),
                "tail5_val_loss": tail,
                "normalized_val_auc": auc,
                "optimizer_state_bytes": int(summary["optimizer_state_bytes"]),
                "peak_allocated_bytes": int(summary["peak_allocated_bytes"]),
                **moonlight_state(summary, contract),
                "checkpoint_path": checkpoint["path"],
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_bytes": int(checkpoint["bytes"]),
            })
    return rows


def paired_rows(
    endpoints: list[dict[str, Any]], controls: list[dict[str, str]], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    control_index = {
        (row["scale"], row["budget_id"], row["method"], int(row["seed"])): row
        for row in controls
    }
    comparators = contract["formal"]["secondary_comparators"] + [contract["formal"]["primary_comparator"]]
    rows: list[dict[str, Any]] = []
    for endpoint in endpoints:
        for comparator in comparators:
            control = control_index[("1b", endpoint["budget_id"], comparator, int(endpoint["seed"]))]
            rows.append({
                "budget_id": endpoint["budget_id"],
                "seed": endpoint["seed"],
                "comparator": comparator,
                "moonlight_final_val_loss": endpoint["final_val_loss"],
                "comparator_final_val_loss": float(control["final_val_loss"]),
                "delta_moonlight_minus_comparator": endpoint["final_val_loss"] - float(control["final_val_loss"]),
            })
    return rows


def contrast_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in seed_rows:
        grouped.setdefault((row["budget_id"], row["comparator"]), []).append(
            float(row["delta_moonlight_minus_comparator"])
        )
    rows: list[dict[str, Any]] = []
    for (budget, comparator), values in sorted(grouped.items()):
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        half = 4.302652729911275 * sd / math.sqrt(len(values)) if len(values) == 3 else float("nan")
        rows.append({
            "budget_id": budget,
            "comparator": comparator,
            "n": len(values),
            "mean_delta_moonlight_minus_comparator": mean,
            "sample_sd": sd,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "moonlight_better_seed_count": sum(value < 0 for value in values),
            "moonlight_worse_seed_count": sum(value > 0 for value in values),
        })
    return rows


def build(run_dir: Path) -> None:
    snapshot = run_dir / "source_snapshot"
    contract_path = snapshot / PACKAGE_REL / "ex57_contract.json"
    contract = P.read_json(contract_path)
    P.assert_contract(contract)
    formal = P.read_json(run_dir / "formal/formal_manifest.json")
    if formal.get("passed") is not True or len(formal.get("units", [])) != 3:
        raise RuntimeError("EX57 analysis requires three passed formal units")
    controls_path = snapshot / PACKAGE_REL / contract["controls"]["path"]
    if P.sha256_file(controls_path) != contract["controls"]["sha256"]:
        raise RuntimeError("EX57 frozen controls changed")
    controls = read_csv(controls_path)
    endpoints = collect_endpoints(run_dir, contract)
    if len(endpoints) != 9:
        raise RuntimeError(f"EX57 expected 9 endpoint rows, observed {len(endpoints)}")
    analysis = run_dir / "analysis"
    endpoint_path = analysis / "endpoint_results.csv"
    write_csv(endpoint_path, endpoints, [
        "scale", "budget_id", "target_step", "seed", "method", "final_val_loss",
        "tail5_val_loss", "normalized_val_auc", "optimizer_state_bytes", "peak_allocated_bytes",
        "moonlight_momentum_state_bytes", "moonlight_matrix_optimizer_state_bytes",
        "checkpoint_path", "checkpoint_sha256", "checkpoint_bytes",
    ])
    seed_rows = paired_rows(endpoints, controls, contract)
    paired_path = analysis / "paired_seed_deltas.csv"
    write_csv(paired_path, seed_rows, [
        "budget_id", "seed", "comparator", "moonlight_final_val_loss",
        "comparator_final_val_loss", "delta_moonlight_minus_comparator",
    ])
    contrasts = contrast_rows(seed_rows)
    contrast_path = analysis / "paired_contrasts.csv"
    write_csv(contrast_path, contrasts, [
        "budget_id", "comparator", "n", "mean_delta_moonlight_minus_comparator",
        "sample_sd", "ci95_low", "ci95_high", "moonlight_better_seed_count",
        "moonlight_worse_seed_count",
    ])
    payload = {
        "schema_version": "ex57_moonlight_analysis_manifest_v1",
        "passed": True,
        "classification": "moonlight_positioned_against_accepted_llama1b_controls",
        "endpoint_rows": len(endpoints),
        "paired_seed_rows": len(seed_rows),
        "aggregate_contrasts": len(contrasts),
        "primary_muon_contrasts": [row for row in contrasts if row["comparator"] == "muon"],
        "practical_margin": contract["analysis"]["practical_loss_margin"],
        "timing_eligible": False,
        "selection_sha256": formal["selection_sha256"],
        "files": {
            "endpoint_results.csv": P.sha256_file(endpoint_path),
            "paired_seed_deltas.csv": P.sha256_file(paired_path),
            "paired_contrasts.csv": P.sha256_file(contrast_path),
            "frozen_controls.csv": P.sha256_file(controls_path),
        },
    }
    P.atomic_json(analysis / "analysis_manifest.json", payload)
    print("EX57 Moonlight analysis passed: 9 endpoint rows")


def verify(run_dir: Path, full_checkpoint_hash: bool) -> None:
    snapshot = run_dir / "source_snapshot"
    contract = P.read_json(snapshot / PACKAGE_REL / "ex57_contract.json")
    P.assert_contract(contract)
    analysis_path = run_dir / "analysis/analysis_manifest.json"
    analysis = P.read_json(analysis_path)
    formal = P.read_json(run_dir / "formal/formal_manifest.json")
    tuning = P.read_json(run_dir / "tuning/tuning_manifest.json")
    preflight = P.read_json(run_dir / "preflight/preflight_manifest.json")
    endpoints = collect_endpoints(run_dir, contract)
    checks: dict[str, bool] = {
        "analysis": analysis.get("passed") is True and analysis.get("endpoint_rows") == 9,
        "formal": formal.get("passed") is True and len(formal.get("units", [])) == 3,
        "tuning": tuning.get("passed") is True,
        "preflight": preflight.get("passed") is True,
        "selection_lineage": formal.get("selection_sha256") == tuning.get("selection_sha256"),
        "contract_lineage": formal.get("selected_contract_sha256") == tuning.get("selected_contract_sha256"),
        "independent": contract["formal"].get("independent_of_ex54") is True,
        "geometry": contract["fairness"].get("same_microbatch_geometry_as_ex48") is True,
        "snapshot": (snapshot / "source_snapshot_manifest.json").is_file(),
    }
    snapshot_manifest = P.read_json(snapshot / "source_snapshot_manifest.json")
    checks["snapshot_hashes"] = bool(snapshot_manifest.get("files")) and all(
        (snapshot / relative).is_file()
        and P.sha256_file(snapshot / relative) == record["sha256"]
        for relative, record in snapshot_manifest.get("files", {}).items()
    )
    for index, row in enumerate(endpoints):
        path = Path(row["checkpoint_path"])
        checks[f"checkpoint:{index}"] = path.is_file() and path.stat().st_size == int(row["checkpoint_bytes"])
        if full_checkpoint_hash and checks[f"checkpoint:{index}"]:
            checks[f"checkpoint_sha:{index}"] = P.sha256_file(path) == row["checkpoint_sha256"]
    # Every fork-only checkpoint must be retired after all children are certified.
    for seed in contract["formal"]["seeds"]:
        unit = run_dir / "formal/1b" / f"seed{seed}"
        manifest = P.read_json(unit / "unit_manifest.json")
        checks[f"unit:{seed}"] = manifest.get("passed") is True and len(manifest.get("endpoints", {})) == 3
        for phase in contract["phases"]:
            if phase["role"] != "fork_source":
                continue
            phase_manifest = P.read_json(unit / phase["id"] / "phase_manifest.json")
            checkpoint = phase_manifest["checkpoint"]
            retirement = P.read_json(unit / phase["id"] / "checkpoint_retirement.json")
            checks[f"retired:{seed}:{phase['id']}"] = (
                not Path(checkpoint["path"]).exists()
                and retirement.get("passed") is True
                and retirement.get("sha256") == checkpoint["sha256"]
                and retirement.get("children") == P.direct_children(contract, phase["id"])
            )
    payload = {
        "schema_version": "ex57_moonlight_verification_v1",
        "passed": all(checks.values()),
        "full_checkpoint_hash": bool(full_checkpoint_hash),
        "checks": checks,
        "analysis_manifest_sha256": P.sha256_file(analysis_path),
    }
    P.atomic_json(run_dir / "analysis/verification_manifest.json", payload)
    print(json.dumps({"passed": payload["passed"], "checks": checks}, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if args.mode == "build":
        build(run_dir)
    else:
        verify(run_dir, args.full_checkpoint_hash)


if __name__ == "__main__":
    main()
