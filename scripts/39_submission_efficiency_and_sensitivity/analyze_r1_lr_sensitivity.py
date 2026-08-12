#!/usr/bin/env python3
"""Validate and aggregate a completed two-lane R1 LR-sensitivity grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-29.7"
EXPECTED_LANE_COUNT = 2
EXPECTED_LANE_VERSION = "2026-07-29.4"
EXPECTED_PROTOCOL = "r1_shared_recipe_lr_multiplier_supporting_v1"
BASE_LR = {"muon": 0.0036, "block4": 0.004, "none": 0.004, "diag": 0.004}
MATRIX_LR = {
    "muon": 0.00036,
    "block4": 0.0004,
    "none": 0.0004,
    "diag": 0.0004,
}
ROLE = {
    "muon": "muon",
    "block4": "original_newton_muon",
    "none": "selective_none",
    "diag": "selective_diag",
}
CONTRASTS = [
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
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty output: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.contract)
    expected_hash = sha256_file(args.contract)
    runner_path = args.contract.parent / "run_r1_lr_sensitivity.py"
    worker_path = args.contract.parent / "r1_lr_sensitivity_worker.py"
    expected_runner_hash = sha256_file(runner_path)
    expected_worker_hash = sha256_file(worker_path)
    lane_paths = sorted(args.run_dir.glob("*/lane_manifest.json"))
    if len(lane_paths) != EXPECTED_LANE_COUNT:
        raise RuntimeError(f"expected two lane manifests, found {lane_paths}")
    lanes = [read_json(path) for path in lane_paths]
    if any(
        lane.get("status") != "completed"
        or lane.get("script_version") != EXPECTED_LANE_VERSION
        or lane.get("contract_sha256") != expected_hash
        or lane.get("runner_sha256") != expected_runner_hash
        or lane.get("worker_sha256") != expected_worker_hash
        for lane in lanes
    ):
        raise RuntimeError("lane acceptance failed")
    if {method for lane in lanes for method in lane["methods"]} != set(ROLE):
        raise RuntimeError("four-method coverage failed")
    if any(set(lane["multipliers"]) != {0.8, 1.0, 1.2} for lane in lanes):
        raise RuntimeError("multiplier coverage failed")
    if any(
        lane["seed"] != contract["seed"]
        or lane["budget_steps"] != contract["budget_steps"]
        or lane["warmdown_steps"] != contract["warmdown_steps"]
        for lane in lanes
    ):
        raise RuntimeError("seed or budget mismatch")

    run_rows: list[dict[str, Any]] = []
    init_hashes: dict[float, set[str]] = {}
    source_hashes: dict[float, dict[str, str]] = {}
    source_manifest_rows: list[dict[str, Any]] = []
    for lane_path, lane in zip(lane_paths, lanes):
        for entry in lane["entries"]:
            multiplier = float(entry["multiplier"])
            manifest_path = args.run_dir / entry["formal_manifest"]
            manifest = read_json(manifest_path)
            if (
                manifest.get("status") != "completed_valid"
                or manifest.get("wandb_complete") is not True
                or manifest.get("family") != contract["family"]
                or manifest.get("protocol") != EXPECTED_PROTOCOL
                or manifest.get("formal_evidence") is not False
                or manifest.get("evidence_profile")
                != "shared_recipe_lr_sensitivity_supporting"
                or manifest.get("seed") != contract["seed"]
                or manifest.get("methods") != lane["methods"]
            ):
                raise RuntimeError(f"formal manifest acceptance failed: {manifest_path}")
            summaries = manifest.get("summaries", ())
            if (
                len(summaries) != len(lane["methods"])
                or {summary.get("method") for summary in summaries}
                != set(lane["methods"])
            ):
                raise RuntimeError(
                    f"formal method coverage failed: {manifest_path}"
                )
            source_manifest_rows.append(
                {
                    "lane": lane["lane"],
                    "multiplier": multiplier,
                    "path": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
                    "methods": ",".join(manifest["methods"]),
                    "status": manifest["status"],
                }
            )
            for summary in manifest["summaries"]:
                method = summary["method"]
                if (
                    summary.get("controlled_seed") != contract["seed"]
                    or summary.get("final_val_step") != contract["budget_steps"]
                    or not math.isclose(
                        float(summary["base_learning_rate"]),
                        BASE_LR[method] * multiplier,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    or not math.isclose(
                        float(summary["matrix_learning_rate"]),
                        MATRIX_LR[method] * multiplier,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                ):
                    raise RuntimeError(
                        f"formal recipe mismatch: {manifest_path}: {method}"
                    )
                canonical = ROLE[method]
                init_hashes.setdefault(multiplier, set()).add(summary["init_sha256"])
                by_method = source_hashes.setdefault(multiplier, {})
                if method in by_method:
                    raise RuntimeError(
                        f"duplicate method at multiplier {multiplier}: {method}"
                    )
                by_method[method] = summary["derived_script_sha256"]
                run_rows.append(
                    {
                        "method": method,
                        "method_role": canonical,
                        "lr_multiplier": multiplier,
                        "seed": int(summary["controlled_seed"]),
                        "budget_steps": int(summary["final_val_step"]),
                        "base_learning_rate": float(summary["base_learning_rate"]),
                        "matrix_learning_rate": float(
                            summary["matrix_learning_rate"]
                        ),
                        "final_val_loss": float(summary["final_val_loss"]),
                        "best_val_loss": float(summary["best_val_loss"]),
                        "normalized_val_auc": float(summary["val_curve_mean"]),
                        "init_sha256": summary["init_sha256"],
                        "derived_script_sha256": summary["derived_script_sha256"],
                        "wandb_status": manifest["wandb_statuses"][method],
                        "evidence_class": "supporting_only",
                    }
                )
    if len(run_rows) != 12:
        raise RuntimeError(f"expected 12 sensitivity cells, found {len(run_rows)}")
    if any(len(values) != 1 for values in init_hashes.values()):
        raise RuntimeError(f"within-multiplier init mismatch: {init_hashes}")
    if len({next(iter(values)) for values in init_hashes.values()}) != 1:
        raise RuntimeError(f"cross-multiplier init mismatch: {init_hashes}")
    for multiplier, by_method in source_hashes.items():
        if set(by_method) != set(ROLE):
            raise RuntimeError(
                f"source coverage mismatch at multiplier {multiplier}: {by_method}"
            )
        if len({by_method[name] for name in ("block4", "none", "diag")}) != 1:
            raise RuntimeError(
                f"Newton source mismatch at multiplier {multiplier}: {by_method}"
            )
    for method in ROLE:
        hashes = {source_hashes[multiplier][method] for multiplier in source_hashes}
        if len(hashes) != 3:
            raise RuntimeError(
                f"{method} source did not change across all three multipliers: {hashes}"
            )

    lookup = {
        (float(row["lr_multiplier"]), row["method_role"]): row for row in run_rows
    }
    contrast_rows = []
    for multiplier in (0.8, 1.0, 1.2):
        for contrast, left, right, priority in CONTRASTS:
            left_row = lookup[(multiplier, left)]
            right_row = lookup[(multiplier, right)]
            contrast_rows.append(
                {
                    "lr_multiplier": multiplier,
                    "priority": priority,
                    "contrast": contrast,
                    "left_role": left,
                    "right_role": right,
                    "final_val_loss_delta_left_minus_right": (
                        left_row["final_val_loss"] - right_row["final_val_loss"]
                    ),
                    "normalized_val_auc_delta_left_minus_right": (
                        left_row["normalized_val_auc"]
                        - right_row["normalized_val_auc"]
                    ),
                    "negative_favors": "left",
                    "evidence_class": "supporting_only",
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "lr_sensitivity_runs.csv", run_rows)
    write_csv(args.output_dir / "lr_sensitivity_contrasts.csv", contrast_rows)
    write_csv(
        args.output_dir / "source_manifest.csv",
        source_manifest_rows,
    )
    role_rows = []
    for method_role in contract["primary_roles"]:
        selected = [row for row in run_rows if row["method_role"] == method_role]
        best = min(selected, key=lambda row: row["final_val_loss"])
        role_rows.append(
            {
                "method_role": method_role,
                "best_observed_multiplier": best["lr_multiplier"],
                "best_observed_final_val_loss": best["final_val_loss"],
                "loss_range_across_grid": max(
                    row["final_val_loss"] for row in selected
                )
                - min(row["final_val_loss"] for row in selected),
                "mean_final_val_loss": statistics.mean(
                    row["final_val_loss"] for row in selected
                ),
                "interpretation": "descriptive grid robustness; no tuned-best claim",
            }
        )
    write_csv(args.output_dir / "lr_sensitivity_role_summary.csv", role_rows)
    artifacts = [
        "lr_sensitivity_runs.csv",
        "lr_sensitivity_contrasts.csv",
        "lr_sensitivity_role_summary.csv",
        "source_manifest.csv",
    ]
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "evidence_class": "supporting_only",
        "submission_claim": "four-method final-recipe LR robustness",
        "tuned_best_claim_allowed": False,
        "diag_vs_none_primary": False,
        "methods": contract["primary_roles"],
        "multipliers": contract["recipe_lr_multipliers"],
        "seed": contract["seed"],
        "budget_steps": contract["budget_steps"],
        "warmdown_steps": contract["warmdown_steps"],
        "run_cells": len(run_rows),
        "contrast_rows": len(contrast_rows),
        "contract_sha256": expected_hash,
        "output_sha256": {
            name: sha256_file(args.output_dir / name) for name in artifacts
        },
        "artifacts": artifacts,
    }
    write_json(args.output_dir / "lr_sensitivity_manifest.json", manifest)
    print(f"R1 LR-sensitivity analysis: {args.output_dir}")
    print(
        f"R1 LR-sensitivity manifest: "
        f"{args.output_dir / 'lr_sensitivity_manifest.json'}"
    )


if __name__ == "__main__":
    main()
