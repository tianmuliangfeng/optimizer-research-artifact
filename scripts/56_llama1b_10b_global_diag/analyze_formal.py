#!/usr/bin/env python3
"""Build and verify the Experiment 56 long-token global-diagonal evidence."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import protocol as P


HERE = Path(__file__).resolve().parent
PACKAGE_REL = Path("scripts/56_llama1b_10b_global_diag")
ENDPOINT_FIELDS = [
    "budget_id", "target_step", "tokens_seen", "tokens_per_parameter",
    "method", "seed", "final_val_loss", "tail5_val_loss",
    "normalized_val_auc", "final_train_loss", "k_state_bytes",
    "checkpoint_path", "checkpoint_bytes", "checkpoint_sha256",
]
PHASE_SUMMARY_SCHEMA = "ex56_llama1b_global_diag_phase_summary_v1"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--full-checkpoint-hash", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def phase_chain(phase_id: str) -> list[str]:
    return {
        "cooldown_6200": ["backbone_4400", "cooldown_6200"],
        "cooldown_13293": ["backbone_4400", "backbone_11493", "cooldown_13293"],
        "cooldown_19073": [
            "backbone_4400", "backbone_11493", "backbone_17273", "cooldown_19073"
        ],
    }[phase_id]


def trajectory_validation(unit_dir: Path, endpoint: str) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for phase_id in phase_chain(endpoint):
        for row in P.read_metrics(unit_dir / phase_id / "metrics.csv"):
            if row["event"] == "val":
                merged[int(row["step"])] = {
                    "step": int(row["step"]), "loss": float(row["loss"])
                }
    return [merged[step] for step in sorted(merged)]


def normalized_auc(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2 or rows[0]["step"] != 0:
        raise RuntimeError("endpoint trajectory lacks step-zero validation")
    area = sum(
        (right["step"] - left["step"]) * (left["loss"] + right["loss"]) / 2.0
        for left, right in zip(rows, rows[1:])
    )
    return area / rows[-1]["step"]


def metric_summary_integrity_checks(
    metrics: list[dict[str, Any]],
    summary: dict[str, Any],
    manifest: dict[str, Any],
    phase: dict[str, Any],
    method: str,
    seed: int,
    contract: dict[str, Any],
    metrics_sha256: str,
) -> dict[str, bool]:
    """Cross-check the complete prespecified metric grid and phase summary.

    The primary endpoint is read from ``summary.json`` while tail/AUC outcomes
    are reconstructed from ``metrics.csv``.  Both files are hash-sealed, but
    their internal agreement must also be a hard gate: a duplicated or missing
    row must never silently change a scientific estimator.
    """

    start = int(phase["start_step"])
    target = int(phase["target_step"])
    every = int(contract["validation"]["regular_every_steps"])
    tokens_per_update = int(contract["training"]["tokens_per_update"])
    accumulation = int(contract["training"]["gradient_accumulation_steps"])
    prefetched = int(contract["data"]["prefetched_train_microbatches"])
    parameters = int(contract["profile"]["parameters"])

    expected_events: list[tuple[str, int]] = []
    for completed in range(start, target + 1):
        if P.should_validate(phase, completed, every):
            expected_events.append(("val", completed))
        if completed < target:
            expected_events.append(("train", completed + 1))

    actual_events = [(str(row.get("event")), int(row.get("step", -1))) for row in metrics]
    val = [row for row in metrics if row.get("event") == "val"]
    train = [row for row in metrics if row.get("event") == "train"]

    def expected_tpp(step: int) -> str:
        return f"{step * tokens_per_update / parameters:.12f}"

    row_identity = all(
        row.get("event") in ("train", "val")
        and row.get("phase_id") == phase["id"]
        and row.get("schedule") == phase["schedule"]
        for row in metrics
    )
    row_geometry = all(
        int(row.get("segment_step", -1)) == int(row["step"]) - start
        and int(row.get("tokens_seen", -1)) == int(row["step"]) * tokens_per_update
        and str(row.get("tokens_per_parameter")) == expected_tpp(int(row["step"]))
        for row in metrics
    )
    row_loader = all(
        int(row.get("loader_consumed_batches", -1))
        == prefetched + int(row["step"]) * accumulation
        and int(row.get("wrap_count", -1)) == 0
        for row in metrics
    )
    checkpoint = manifest.get("checkpoint", {})
    expected_tokens = target * tokens_per_update
    expected_summary_tpp = expected_tokens / parameters
    return {
        "metric_schema": bool(metrics)
        and all(set(row) == set(P.METRIC_FIELDS) for row in metrics),
        "metric_event_grid": actual_events == expected_events,
        "metric_row_identity": row_identity,
        "metric_row_geometry": row_geometry,
        "metric_loader_cursor": row_loader,
        "summary_schema": summary.get("schema_version") == PHASE_SUMMARY_SCHEMA,
        "summary_identity": summary.get("status") == "completed"
        and summary.get("engineering_pilot") is False
        and summary.get("method") == method
        and int(summary.get("seed", -1)) == seed
        and summary.get("phase") == phase,
        "summary_target": int(summary.get("completed_steps", -1)) == target
        and int(summary.get("tokens_seen", -1)) == expected_tokens
        and math.isclose(
            float(summary.get("tokens_per_parameter", float("nan"))),
            expected_summary_tpp,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "summary_metric_link": summary.get("metrics_sha256") == metrics_sha256,
        "summary_endpoint_values": bool(val)
        and bool(train)
        and float(summary.get("final_val_loss", float("nan"))) == float(val[-1]["loss"])
        and float(summary.get("final_train_loss", float("nan"))) == float(train[-1]["loss"]),
        "summary_checkpoint_link": summary.get("checkpoint_path") == checkpoint.get("path")
        and summary.get("checkpoint_sha256") == checkpoint.get("sha256")
        and int(summary.get("checkpoint_bytes", -1)) == int(checkpoint.get("bytes", -2)),
        "manifest_phase_contract": manifest.get("role") == phase["role"]
        and bool(checkpoint.get("retained")) is bool(phase["retain_checkpoint"]),
    }


def audit_phase(
    phase_dir: Path,
    phase: dict[str, Any],
    method: str,
    seed: int,
    contract: dict[str, Any],
    contract_sha: str,
    data_sha: str,
    full_checkpoint_hash: bool,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    manifest_path = phase_dir / "phase_manifest.json"
    summary_path = phase_dir / "summary.json"
    metrics_path = phase_dir / "metrics.csv"
    manifest = P.read_json(manifest_path)
    summary = P.read_json(summary_path)
    metrics = P.read_metrics(metrics_path)
    checkpoint = manifest["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    retirement_path = phase_dir / "checkpoint_retirement.json"
    expected_retained = bool(phase["retain_checkpoint"])
    present = checkpoint_path.is_file()
    size_ok = present and checkpoint_path.stat().st_size == int(checkpoint["bytes"])
    # Full re-hashing applies to the nine retained scientific endpoints.  Fork
    # checkpoints are deliberately deleted once every direct child passes and
    # are instead bound by their retirement certificates below.
    hash_ok = (not expected_retained) or not full_checkpoint_hash or (
        present and P.sha256_file(checkpoint_path) == checkpoint["sha256"]
    )
    retirement_ok = False
    if retirement_path.is_file():
        retirement = P.read_json(retirement_path)
        retirement_ok = (
            retirement.get("passed") is True
            and retirement.get("phase_id") == phase["id"]
            and retirement.get("checkpoint_sha256") == checkpoint["sha256"]
            and int(retirement.get("checkpoint_bytes", -1)) == int(checkpoint["bytes"])
            and retirement.get("direct_children") == P.direct_children(contract, phase["id"])
        )
    val = [row for row in metrics if row["event"] == "val"]
    train = [row for row in metrics if row["event"] == "train"]
    groups = summary.get("architecture", {}).get("preconditioner_groups", [])
    metrics_sha256 = P.sha256_file(metrics_path)
    internal_checks = metric_summary_integrity_checks(
        metrics,
        summary,
        manifest,
        phase,
        method,
        seed,
        contract,
        metrics_sha256,
    )
    checks = {
        "schema": manifest.get("schema_version") == P.PHASE_MANIFEST_SCHEMA,
        "passed": manifest.get("passed") is True,
        "identity": manifest.get("method") == method
        and int(manifest.get("seed", -1)) == seed
        and manifest.get("phase_id") == phase["id"],
        "contract": manifest.get("contract_sha256") == contract_sha
        and summary.get("contract_sha256") == contract_sha,
        "data": manifest.get("data_inventory_sha256") == data_sha
        and summary.get("data_inventory_sha256") == data_sha,
        "summary_hash": P.sha256_file(summary_path) == manifest.get("summary_sha256"),
        "metrics_hash": metrics_sha256 == manifest.get("metrics_sha256"),
        "target": int(summary.get("completed_steps", -1)) == int(phase["target_step"]),
        "accepted_ex48_initialization": summary.get("init_sha256")
        == contract["accepted_ex48_initialization_sha256"][str(seed)],
        "target_validation": bool(val) and int(val[-1]["step"]) == int(phase["target_step"]),
        "train_row_count": len(train) == int(phase["target_step"]) - int(phase["start_step"]),
        "finite": bool(metrics) and all(math.isfinite(float(row["loss"])) for row in metrics),
        "no_wrap": int(summary.get("loader_final", {}).get("wrap_count", -1)) == 0,
        "cursor": summary.get("loader_final") == summary.get("loader_expected"),
        "global_diag_route": summary.get("architecture", {}).get("global_diag_route") is True,
        "preconditioner_group_count": int(
            summary.get("architecture", {}).get("preconditioner_group_count", -1)
        ) == int(contract["profile"]["expected_preconditioner_groups"][method])
        and len(groups) == int(contract["profile"]["expected_preconditioner_groups"][method]),
        "all_groups_diagonal": bool(groups)
        and all(row.get("kind") == "diag" for row in groups),
        "k_state": int(summary.get("k_state_bytes", -1))
        == int(contract["profile"]["expected_global_diag_k_state_bytes"]),
        "dense_workspace_absent": int(summary.get("preconditioner_workspace_bytes", -1)) == 0,
        "retention": (
            expected_retained and present and size_ok and not retirement_path.exists()
        ) or (not expected_retained and not present and retirement_ok),
        "checkpoint_hash": hash_ok,
        **internal_checks,
    }
    return checks, manifest, summary


def audit_run(run_dir: Path, full_checkpoint_hash: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    snapshot = run_dir / "source_snapshot"
    contract_path = snapshot / PACKAGE_REL / "formal_contract.json"
    contract = P.read_json(contract_path)
    P.assert_contract(contract)
    contract_sha = P.sha256_file(contract_path)
    source_manifest_path = snapshot / "source_snapshot_manifest.json"
    source_manifest = P.read_json(source_manifest_path)
    source_files = source_manifest.get("files", {})
    source_checks = {
        relative: (snapshot / relative).is_file()
        and (snapshot / relative).stat().st_size == int(item["bytes"])
        and P.sha256_file(snapshot / relative) == item["sha256"]
        for relative, item in source_files.items()
    }
    data_path = run_dir / "data_audit.json"
    data = P.read_json(data_path)
    data_sha = data["inventory_sha256"]
    preflight = P.read_json(run_dir / "preflight_manifest.json")
    identity_path = run_dir / "run_identity.json"
    identity = P.read_json(identity_path)
    init_path = run_dir / "init_audit.json"
    initialization = P.read_json(init_path)
    pilot = P.read_json(run_dir / "pilot_manifest.json")
    suite = P.read_json(run_dir / "suite_status.json")
    checks: dict[str, bool] = {
        "source_snapshot": source_manifest.get("schema_version") == "ex56_source_snapshot_v1"
        and bool(source_checks) and all(source_checks.values()),
        "source_snapshot_identity": identity.get("source_snapshot_manifest_sha256")
        == P.sha256_file(source_manifest_path),
        "preflight": preflight.get("passed") is True,
        "preflight_artifacts": preflight.get("data_audit_sha256") == P.sha256_file(data_path)
        and preflight.get("init_audit_sha256") == P.sha256_file(init_path)
        and preflight.get("identity_sha256") == P.sha256_file(identity_path),
        "data": data.get("schema_version") == P.DATA_AUDIT_SCHEMA
        and data.get("passed") is True
        and data.get("full_hash") is True
        and data.get("content_projection_sha256")
        == contract["data"]["accepted_ex48_content_projection_sha256"]
        and P.content_inventory_sha256(data.get("inventory", {}))
        == contract["data"]["accepted_ex48_content_projection_sha256"],
        "initialization": initialization.get("passed") is True
        and initialization.get("accepted_ex48_initialization_sha256")
        == contract["accepted_ex48_initialization_sha256"],
        "pilot": pilot.get("schema_version") == "ex56_engineering_pilot_v1"
        and pilot.get("passed") is True
        and pilot.get("planned_interrupt_return_code") == 75
        and pilot.get("in_place_resume") is True
        and pilot.get("source_checkpoint_branch") is True
        and pilot.get("no_wrap") is True,
        "pilot_retirement": len(pilot.get("retired_pilot_checkpoints", [])) == 2
        and bool(pilot.get("retirement_prepared_at"))
        and bool(pilot.get("retirement_completed_at"))
        and all(not Path(row["path"]).exists() for row in pilot.get("retired_pilot_checkpoints", [])),
        "suite": suite.get("schema_version") == "ex56_suite_status_v1"
        and suite.get("passed") is True
        and int(suite.get("completed_units", -1)) == 3
        and int(suite.get("expected_units", -1)) == 3
        and not suite.get("failures"),
        "formal_unit_count": len(list((run_dir / "formal" / "global_diag").glob("seed*/unit_manifest.json"))) == 3,
        "control_hash": P.sha256_file(snapshot / PACKAGE_REL / "frozen_ex48_controls.csv")
        == contract["source_lineage"]["frozen_control_csv_sha256"],
    }
    endpoints: list[dict[str, Any]] = []
    endpoint_by_id = {row["id"]: row for row in P.endpoint_phases(contract)}
    for seed in contract["grid"]["seeds"]:
        unit_dir = run_dir / "formal" / "global_diag" / f"seed{seed}"
        unit = P.read_json(unit_dir / "unit_manifest.json")
        checks[f"unit:{seed}:identity"] = (
            unit.get("schema_version") == P.UNIT_MANIFEST_SCHEMA
            and unit.get("passed") is True
            and unit.get("method") == "global_diag"
            and int(unit.get("seed", -1)) == seed
            and unit.get("completed_phases") == [row["id"] for row in contract["phases"]]
            and set(unit.get("phases", {})) == {row["id"] for row in contract["phases"]}
        )
        phase_manifests: dict[str, dict[str, Any]] = {}
        for phase in contract["phases"]:
            phase_dir = unit_dir / phase["id"]
            phase_checks, manifest, summary = audit_phase(
                phase_dir, phase, "global_diag", seed,
                contract, contract_sha, data_sha, full_checkpoint_hash,
            )
            for key, value in phase_checks.items():
                checks[f"phase:{seed}:{phase['id']}:{key}"] = value
            frozen = unit.get("phases", {}).get(phase["id"], {})
            checks[f"unit:{seed}:{phase['id']}:sealed"] = (
                frozen.get("manifest_sha256") == P.sha256_file(phase_dir / "phase_manifest.json")
                and frozen.get("summary_sha256") == P.sha256_file(phase_dir / "summary.json")
                and frozen.get("metrics_sha256") == P.sha256_file(phase_dir / "metrics.csv")
            )
            parent = phase.get("parent")
            checks[f"phase:{seed}:{phase['id']}:source_lineage"] = (
                summary.get("source_checkpoint_sha256") is None
                if parent is None
                else summary.get("source_checkpoint_sha256")
                == phase_manifests[str(parent)]["checkpoint"]["sha256"]
            )
            phase_manifests[phase["id"]] = manifest
            if phase["id"] in endpoint_by_id:
                trajectory = trajectory_validation(unit_dir, phase["id"])
                checkpoint = manifest["checkpoint"]
                endpoints.append({
                    "budget_id": phase["budget_id"],
                    "target_step": phase["target_step"],
                    "tokens_seen": phase["actual_tokens"],
                    "tokens_per_parameter": f"{float(phase['tokens_per_parameter']):.12f}",
                    "method": "global_diag",
                    "seed": seed,
                    "final_val_loss": f"{float(summary['final_val_loss']):.9f}",
                    "tail5_val_loss": f"{statistics.mean(row['loss'] for row in trajectory[-5:]):.9f}",
                    "normalized_val_auc": f"{normalized_auc(trajectory):.9f}",
                    "final_train_loss": f"{float(summary['final_train_loss']):.9f}",
                    "k_state_bytes": int(summary["k_state_bytes"]),
                    "checkpoint_path": checkpoint["path"],
                    "checkpoint_bytes": checkpoint["bytes"],
                    "checkpoint_sha256": checkpoint["sha256"],
                })
        expected_retained = {
            phase["budget_id"]: phase_manifests[phase["id"]]["checkpoint"]
            for phase in P.endpoint_phases(contract)
        }
        checks[f"unit:{seed}:retained_endpoints"] = (
            unit.get("retained_endpoints") == expected_retained
        )
    checks["endpoint_count"] = len(endpoints) == 9
    return {
        "passed": all(checks.values()), "checks": checks,
        "contract": contract, "contract_sha256": contract_sha,
        "data_inventory_sha256": data_sha,
    }, endpoints


def paired_contrasts(
    new_rows: list[dict[str, Any]], controls: list[dict[str, str]], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    new = {(row["budget_id"], int(row["seed"])): float(row["final_val_loss"]) for row in new_rows}
    old = {
        (row["budget_id"], row["method"], int(row["seed"])): float(row["final_val_loss"])
        for row in controls
    }
    rows: list[dict[str, Any]] = []
    tcrit = 4.302652729911275
    for budget in contract["analysis"]["primary_budgets"]:
        for comparator in contract["analysis"]["comparators"]:
            values = [
                new[(budget, seed)] - old[(budget, comparator, seed)]
                for seed in contract["grid"]["seeds"]
            ]
            mean = statistics.mean(values)
            sd = statistics.stdev(values)
            half = tcrit * sd / math.sqrt(len(values))
            rows.append({
                "budget_id": budget,
                "contrast": f"global_diag-minus-{comparator}",
                "comparator": comparator,
                "seed2024": f"{values[0]:.9f}",
                "seed2025": f"{values[1]:.9f}",
                "seed2026": f"{values[2]:.9f}",
                "mean_difference": f"{mean:.9f}",
                "sample_sd": f"{sd:.9f}",
                "ci95_low": f"{mean - half:.9f}",
                "ci95_high": f"{mean + half:.9f}",
                "global_diag_better_seeds": sum(value < 0 for value in values),
                "global_diag_worse_seeds": sum(value > 0 for value in values),
                "practical_margin": contract["analysis"]["practical_loss_margin"],
            })
    return rows


def classify(rows: list[dict[str, Any]]) -> dict[str, Any]:
    final = [row for row in rows if row["budget_id"] == "tokens_approximately_10b"]
    signs = {
        row["comparator"]: {
            "mean_global_minus_comparator": float(row["mean_difference"]),
            "better_seed_count": int(row["global_diag_better_seeds"]),
            "worse_seed_count": int(row["global_diag_worse_seeds"]),
        }
        for row in final
    }
    best_mean = min(value["mean_global_minus_comparator"] for value in signs.values())
    worst_mean = max(value["mean_global_minus_comparator"] for value in signs.values())
    if worst_mean < -0.002:
        label = "global_diag_better_than_all_controls_on_paired_mean_at_10b"
    elif best_mean > 0.002:
        label = "global_diag_worse_than_all_controls_on_paired_mean_at_10b"
    else:
        label = "mixed_or_practically_equivalent_at_10b"
    return {"classification": label, "ten_billion_token_contrasts": signs}


def artifact_hashes(analysis_dir: Path) -> dict[str, str]:
    names = [
        "endpoint_results.csv", "unified_endpoint_results.csv", "paired_contrasts.csv",
        "classification.json", "EX56_FORMAL_ANALYSIS.md",
    ]
    return {name: P.sha256_file(analysis_dir / name) for name in names}


def build(run_dir: Path) -> dict[str, Any]:
    audit, endpoints = audit_run(run_dir, full_checkpoint_hash=False)
    if not audit["passed"]:
        raise RuntimeError(f"EX56 formal audit failed: {audit['checks']}")
    contract = audit["contract"]
    snapshot = run_dir / "source_snapshot"
    controls_path = snapshot / PACKAGE_REL / "frozen_ex48_controls.csv"
    controls = read_csv(controls_path)
    expected_keys = {
        (budget, method, seed)
        for budget in contract["analysis"]["primary_budgets"]
        for method in contract["analysis"]["comparators"]
        for seed in contract["grid"]["seeds"]
    }
    observed_keys = {(r["budget_id"], r["method"], int(r["seed"])) for r in controls}
    if observed_keys != expected_keys:
        raise RuntimeError("frozen EX48 control panel is incomplete or contains extra cells")
    analysis_dir = run_dir / "analysis"
    write_csv(analysis_dir / "endpoint_results.csv", ENDPOINT_FIELDS, endpoints)
    unified = [dict(row) | {"source": "ex48_frozen_control"} for row in controls]
    unified.extend(dict(row) | {"source": "ex56_formal"} for row in endpoints)
    unified_fields = [*ENDPOINT_FIELDS, "source"]
    for row in unified:
        for field in unified_fields:
            row.setdefault(field, "")
    write_csv(analysis_dir / "unified_endpoint_results.csv", unified_fields, unified)
    contrasts = paired_contrasts(endpoints, controls, contract)
    contrast_fields = list(contrasts[0])
    write_csv(analysis_dir / "paired_contrasts.csv", contrast_fields, contrasts)
    classification = classify(contrasts)
    P.atomic_json(analysis_dir / "classification.json", classification)
    lines = [
        "# Experiment 56 formal analysis", "",
        f"Scientific classification: `{classification['classification']}`.", "",
        "The new arm applies coordinate-diagonal curvature state to every eligible matrix and is",
        "paired by seed and budget against the frozen Experiment 48 controls. Timing is excluded.", "",
        "| budget | comparator | mean(global diag - comparator) | better seeds | worse seeds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in contrasts:
        lines.append(
            f"| {row['budget_id']} | {row['comparator']} | {row['mean_difference']} | "
            f"{row['global_diag_better_seeds']} | {row['global_diag_worse_seeds']} |"
        )
    (analysis_dir / "EX56_FORMAL_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = artifact_hashes(analysis_dir)
    manifest = {
        "schema_version": "ex56_analysis_manifest_v1",
        "passed": True,
        "status": "completed_valid",
        "experiment_id": "56_llama1b_10b_global_diag",
        "formal_units": 3,
        "primary_endpoints": 9,
        "control_endpoints": 36,
        "paired_contrasts": 12,
        "classification": classification["classification"],
        "timing_usable": False,
        "contract_sha256": audit["contract_sha256"],
        "data_inventory_sha256": audit["data_inventory_sha256"],
        "control_csv_sha256": P.sha256_file(controls_path),
        "integrity_checks": audit["checks"],
        "artifacts": hashes,
        "created_at": now_iso(),
    }
    P.atomic_json(analysis_dir / "analysis_manifest.json", manifest)
    checkpoint_rows = [
        {
            "budget_id": row["budget_id"], "seed": row["seed"],
            "path": row["checkpoint_path"], "bytes": row["checkpoint_bytes"],
            "sha256": row["checkpoint_sha256"],
        }
        for row in endpoints
    ]
    P.atomic_json(run_dir / "handoff_manifest.json", {
        "schema_version": "ex56_handoff_manifest_v1",
        "status": "completed", "passed": True,
        "experiment_id": "56_llama1b_10b_global_diag",
        "scientific_result": classification["classification"],
        "formal_units": 3, "primary_endpoints": 9,
        "timing_usable": False,
        "analysis_manifest_sha256": P.sha256_file(analysis_dir / "analysis_manifest.json"),
        "external_retained_checkpoints": checkpoint_rows,
    })
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def verify(run_dir: Path, full_checkpoint_hash: bool) -> dict[str, Any]:
    audit, endpoints = audit_run(run_dir, full_checkpoint_hash=full_checkpoint_hash)
    analysis = P.read_json(run_dir / "analysis" / "analysis_manifest.json")
    handoff = P.read_json(run_dir / "handoff_manifest.json")
    checks = {
        "run_audit": audit["passed"],
        "analysis_passed": analysis.get("passed") is True,
        "analysis_identity": analysis.get("experiment_id") == "56_llama1b_10b_global_diag",
        "analysis_counts": int(analysis.get("formal_units", -1)) == 3
        and int(analysis.get("primary_endpoints", -1)) == 9,
        "analysis_artifacts": analysis.get("artifacts") == artifact_hashes(run_dir / "analysis"),
        "handoff": handoff.get("passed") is True
        and handoff.get("experiment_id") == "56_llama1b_10b_global_diag",
        "handoff_hash": handoff.get("analysis_manifest_sha256")
        == P.sha256_file(run_dir / "analysis" / "analysis_manifest.json"),
        "retained_checkpoint_count": len(handoff.get("external_retained_checkpoints", [])) == 9,
        "endpoint_count": len(endpoints) == 9,
    }
    payload = {
        "schema_version": "ex56_native_verify_v1",
        "passed": all(checks.values()),
        "full_checkpoint_hash": full_checkpoint_hash,
        "checks": checks,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    try:
        payload = build(run_dir) if args.mode == "build" else verify(
            run_dir, args.full_checkpoint_hash
        )
        return 0 if payload.get("passed") is True else 2
    except Exception as exc:
        print(f"EX56 analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
