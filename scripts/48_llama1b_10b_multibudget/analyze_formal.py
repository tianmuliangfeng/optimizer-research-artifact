#!/usr/bin/env python3
"""Integrity validation and frozen endpoint analysis for experiment 48."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any

import protocol as P


HERE = Path(__file__).resolve().parent
PACKAGE_REL = Path("scripts/48_llama1b_10b_multibudget")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--full-checkpoint-hash", action="store_true")
    return parser.parse_args()


def phase_artifact_audit(
    phase_dir: Path,
    phase: dict[str, Any],
    method: str,
    seed: int,
    contract_sha256: str,
    data_inventory_sha256: str,
    full_checkpoint_hash: bool,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    manifest_path = phase_dir / "phase_manifest.json"
    summary_path = phase_dir / "summary.json"
    metrics_path = phase_dir / "metrics.csv"
    manifest = P.read_json(manifest_path)
    summary = P.read_json(summary_path)
    rows = P.read_metrics(metrics_path)
    checkpoint = manifest["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    retirement_path = phase_dir / "checkpoint_retirement.json"
    expected_retained = bool(phase["retain_checkpoint"])
    checkpoint_present = checkpoint_path.is_file()
    checkpoint_size = checkpoint_present and checkpoint_path.stat().st_size == int(checkpoint["bytes"])
    checkpoint_hash = True
    if full_checkpoint_hash and checkpoint_present:
        checkpoint_hash = P.sha256_file(checkpoint_path) == checkpoint["sha256"]
    retirement_ok = False
    if retirement_path.is_file():
        retirement = P.read_json(retirement_path)
        retirement_ok = (
            retirement.get("passed") is True
            and retirement.get("phase_id") == phase["id"]
            and retirement.get("checkpoint_sha256") == checkpoint["sha256"]
            and int(retirement.get("checkpoint_bytes", -1)) == int(checkpoint["bytes"])
            and retirement.get("direct_children") == P.direct_children(contract_global, phase["id"])
        )
    val_rows = [row for row in rows if row["event"] == "val"]
    train_rows = [row for row in rows if row["event"] == "train"]
    checks = {
        "manifest_schema": manifest.get("schema_version") == P.PHASE_MANIFEST_SCHEMA,
        "manifest_passed": manifest.get("passed") is True,
        "identity": manifest.get("method") == method
        and int(manifest.get("seed", -1)) == seed
        and manifest.get("phase_id") == phase["id"],
        "contract": manifest.get("contract_sha256") == contract_sha256
        and summary.get("contract_sha256") == contract_sha256,
        "data": manifest.get("data_inventory_sha256") == data_inventory_sha256
        and summary.get("data_inventory_sha256") == data_inventory_sha256,
        "summary_hash": P.sha256_file(summary_path) == manifest.get("summary_sha256"),
        "metrics_hash": P.sha256_file(metrics_path) == manifest.get("metrics_sha256"),
        "summary_status": summary.get("status") == "completed",
        "target_step": int(summary.get("completed_steps", -1)) == int(phase["target_step"]),
        "tokens": int(summary.get("tokens_seen", -1))
        == int(phase["target_step"]) * int(contract_global["training"]["tokens_per_update"]),
        "target_validation": bool(val_rows)
        and int(val_rows[-1]["step"]) == int(phase["target_step"]),
        "train_rows": len(train_rows) == int(phase["target_step"]) - int(phase["start_step"]),
        "finite": bool(rows) and all(math.isfinite(float(row["loss"])) for row in rows),
        "no_wrap": int(summary.get("loader_final", {}).get("wrap_count", -1)) == 0
        and all(int(row["wrap_count"]) == 0 for row in rows),
        "cursor": summary.get("loader_final") == summary.get("loader_expected"),
        "retention": (
            expected_retained and checkpoint_present and checkpoint_size and not retirement_path.exists()
        )
        or (
            not expected_retained and not checkpoint_present and retirement_ok
        ),
        "checkpoint_hash": checkpoint_hash,
    }
    return checks, manifest, summary


def endpoint_chain(phase_id: str) -> list[str]:
    return {
        "cooldown_6200": ["backbone_4400", "cooldown_6200"],
        "cooldown_13293": ["backbone_4400", "backbone_11493", "cooldown_13293"],
        "cooldown_19073": [
            "backbone_4400",
            "backbone_11493",
            "backbone_17273",
            "cooldown_19073",
        ],
    }[phase_id]


def trajectory_validation(unit_dir: Path, endpoint_phase: str) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for phase_id in endpoint_chain(endpoint_phase):
        for row in P.read_metrics(unit_dir / phase_id / "metrics.csv"):
            if row["event"] == "val":
                merged[int(row["step"])] = {
                    "step": int(row["step"]),
                    "loss": float(row["loss"]),
                    "tokens_seen": int(row["tokens_seen"]),
                    "phase_id": phase_id,
                }
    return [merged[step] for step in sorted(merged)]


def normalized_auc(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2 or rows[0]["step"] != 0:
        raise RuntimeError("endpoint trajectory lacks step-zero validation")
    area = 0.0
    for left, right in zip(rows, rows[1:]):
        area += (right["step"] - left["step"]) * (left["loss"] + right["loss"]) / 2.0
    return area / rows[-1]["step"]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def paired_contrasts(endpoint_rows: list[dict[str, Any]], contract: dict[str, Any]) -> list[dict[str, Any]]:
    by_key = {
        (row["budget_id"], row["method"], int(row["seed"])): float(row["final_val_loss"])
        for row in endpoint_rows
    }
    pairs = [
        ("down_none", "muon"),
        ("down_diag", "muon"),
        ("newton_full", "muon"),
        ("down_none", "newton_full"),
        ("down_diag", "newton_full"),
        ("down_none", "down_diag"),
    ]
    rows = []
    for budget in contract["analysis"]["primary_budgets"]:
        for left, right in pairs:
            values = [
                by_key[(budget, left, int(seed))] - by_key[(budget, right, int(seed))]
                for seed in contract["grid"]["seeds"]
            ]
            rows.append(
                {
                    "budget_id": budget,
                    "contrast": f"{left}-minus-{right}",
                    "left": left,
                    "right": right,
                    "seed2024": f"{values[0]:.9f}",
                    "seed2025": f"{values[1]:.9f}",
                    "seed2026": f"{values[2]:.9f}",
                    "mean_difference": f"{statistics.mean(values):.9f}",
                    "sample_sd": f"{statistics.stdev(values):.9f}",
                    "negative_seeds": sum(value < 0 for value in values),
                    "positive_seeds": sum(value > 0 for value in values),
                    "practical_margin": contract["analysis"]["practical_loss_margin"],
                }
            )
    return rows


def classify(contrast_rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    final_budget = "tokens_approximately_10b"
    selected = {
        row["contrast"]: row
        for row in contrast_rows
        if row["budget_id"] == final_budget
        and row["contrast"] in ("down_none-minus-muon", "down_diag-minus-muon")
    }
    margin = float(contract["analysis"]["practical_loss_margin"])
    recovery = [
        key
        for key, row in selected.items()
        if float(row["mean_difference"]) <= -margin and int(row["negative_seeds"]) >= 2
    ]
    persistent = all(
        float(row["mean_difference"]) >= margin and int(row["positive_seeds"]) >= 2
        for row in selected.values()
    )
    if recovery:
        label = "clear_selective_recovery"
    elif persistent:
        label = "persistent_muon_lead"
    else:
        label = "mixed_or_practically_equivalent"
    return {
        "classification": label,
        "recovery_methods": recovery,
        "practical_margin": margin,
        "frozen_rules": {
            "clear_selective_recovery": contract["analysis"]["clear_selective_recovery_rule"],
            "persistent_muon_lead": contract["analysis"]["persistent_muon_lead_rule"],
            "otherwise": contract["analysis"]["otherwise"],
        },
    }


def build_handoff(run_dir: Path, endpoint_rows: list[dict[str, Any]]) -> dict[str, Any]:
    included: dict[str, dict[str, Any]] = {}
    excluded_checkpoint_paths = set()
    for row in endpoint_rows:
        excluded_checkpoint_paths.add(str(Path(row["checkpoint_path"]).resolve()))
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp") or path.name == "handoff_manifest.json":
            continue
        if path.name in ("status.json", "worker.log", "wandb_upload.json"):
            continue
        if path.suffix == ".pt":
            continue
        relative = path.relative_to(run_dir).as_posix()
        included[relative] = {"bytes": path.stat().st_size, "sha256": P.sha256_file(path)}
    checkpoints = [
        {
            "method": row["method"],
            "seed": row["seed"],
            "budget_id": row["budget_id"],
            "path": row["checkpoint_path"],
            "bytes": row["checkpoint_bytes"],
            "sha256": row["checkpoint_sha256"],
            "included_in_small_handoff_archive": False,
        }
        for row in endpoint_rows
    ]
    return {
        "schema_version": "ex48_handoff_manifest_v1",
        "passed": True,
        "created_at": now_iso(),
        "small_artifacts": included,
        "external_retained_checkpoints": checkpoints,
        "note": "The normal handoff ZIP excludes 36 multi-GB endpoint checkpoints; their full hashes and sizes remain frozen here.",
    }


def audit_run(run_dir: Path, full_checkpoint_hash: bool) -> dict[str, Any]:
    global contract_global
    snapshot = run_dir / "source_snapshot"
    contract_path = snapshot / PACKAGE_REL / "formal_contract.json"
    contract = P.read_json(contract_path)
    contract_global = contract
    P.assert_contract(contract)
    contract_sha256 = P.sha256_file(contract_path)
    data_audit = P.read_json(run_dir / "data_audit.json")
    data_inventory_sha256 = data_audit["inventory_sha256"]
    source_manifest = P.read_json(snapshot / "source_snapshot_manifest.json")
    source_checks = {
        relative: (snapshot / relative).is_file()
        and (snapshot / relative).stat().st_size == int(item["bytes"])
        and P.sha256_file(snapshot / relative) == item["sha256"]
        for relative, item in source_manifest["files"].items()
    }
    top_checks = {
        "contract": all(P.validate_contract(contract).values()),
        "source_snapshot": bool(source_checks) and all(source_checks.values()),
        "preflight": P.read_json(run_dir / "preflight_manifest.json").get("passed") is True,
        "pilot": P.read_json(run_dir / "pilot_manifest.json").get("passed") is True,
        "suite": P.read_json(run_dir / "suite_status.json").get("passed") is True,
        "data": data_audit.get("passed") is True,
    }
    phase_failures: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    init_by_seed_method: dict[tuple[int, str], str] = {}
    lineage_checks: list[bool] = []
    retained_count = 0
    unit_count = 0
    for method in contract["grid"]["methods"]:
        for seed_value in contract["grid"]["seeds"]:
            seed = int(seed_value)
            unit_dir = run_dir / "formal" / method / f"seed{seed}"
            unit_manifest = P.read_json(unit_dir / "unit_manifest.json")
            unit_count += int(unit_manifest.get("passed") is True)
            phase_manifests: dict[str, dict[str, Any]] = {}
            phase_summaries: dict[str, dict[str, Any]] = {}
            for phase in contract["phases"]:
                checks, manifest, summary = phase_artifact_audit(
                    unit_dir / phase["id"],
                    phase,
                    method,
                    seed,
                    contract_sha256,
                    data_inventory_sha256,
                    full_checkpoint_hash,
                )
                phase_manifests[phase["id"]] = manifest
                phase_summaries[phase["id"]] = summary
                if not all(checks.values()):
                    phase_failures.append(
                        {"method": method, "seed": seed, "phase_id": phase["id"], "checks": checks}
                    )
                init_by_seed_method[(seed, method)] = summary["init_sha256"]
                parent = phase.get("parent")
                if parent is not None:
                    lineage_checks.append(
                        summary.get("source_checkpoint_sha256")
                        == phase_manifests[parent]["checkpoint"]["sha256"]
                    )
                if phase["role"] == "primary_endpoint":
                    trajectory = trajectory_validation(unit_dir, phase["id"])
                    checkpoint = manifest["checkpoint"]
                    retained_count += int(Path(checkpoint["path"]).is_file())
                    endpoint_rows.append(
                        {
                            "budget_id": phase["budget_id"],
                            "target_step": phase["target_step"],
                            "tokens_seen": summary["tokens_seen"],
                            "tokens_per_parameter": f"{summary['tokens_per_parameter']:.12f}",
                            "method": method,
                            "seed": seed,
                            "final_val_loss": f"{summary['final_val_loss']:.9f}",
                            "tail5_val_loss": f"{statistics.mean(row['loss'] for row in trajectory[-5:]):.9f}",
                            "normalized_val_auc": f"{normalized_auc(trajectory):.9f}",
                            "final_train_loss": f"{summary['final_train_loss']:.9f}",
                            "resume_count_total": sum(
                                int(phase_summaries[item]["resume_count"])
                                for item in endpoint_chain(phase["id"])
                            ),
                            "wrap_count": summary["loader_final"]["wrap_count"],
                            "checkpoint_path": checkpoint["path"],
                            "checkpoint_bytes": checkpoint["bytes"],
                            "checkpoint_sha256": checkpoint["sha256"],
                        }
                    )
    init_checks = []
    for seed_value in contract["grid"]["seeds"]:
        fingerprints = {
            init_by_seed_method[(int(seed_value), method)] for method in contract["grid"]["methods"]
        }
        init_checks.append(len(fingerprints) == 1)
    checks = top_checks | {
        "unit_count": unit_count == 12,
        "phase_count": len(phase_failures) == 0,
        "endpoint_count": len(endpoint_rows) == 36,
        "retained_checkpoint_count": retained_count == 36,
        "lineage": bool(lineage_checks) and all(lineage_checks),
        "same_seed_initialization": all(init_checks),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "phase_failures": phase_failures,
        "endpoint_rows": endpoint_rows,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "data_inventory_sha256": data_inventory_sha256,
        "full_checkpoint_hash": full_checkpoint_hash,
    }


def report_markdown(
    endpoint_rows: list[dict[str, Any]], contrast_rows: list[dict[str, Any]], classification: dict[str, Any]
) -> str:
    means: dict[tuple[str, str], float] = {}
    for budget in contract_global["analysis"]["primary_budgets"]:
        for method in contract_global["grid"]["methods"]:
            values = [
                float(row["final_val_loss"])
                for row in endpoint_rows
                if row["budget_id"] == budget and row["method"] == method
            ]
            means[(budget, method)] = statistics.mean(values)
    lines = [
        "# 实验 48：LLaMA-1B 三预算长 token 正式分析",
        "",
        f"完整性状态：**passed**。冻结分类：`{classification['classification']}`。",
        "",
        "三个预算点均来自相同 peak-LR 语义和 1800-step cooldown 的独立分叉终点；它们不是同一最终轨迹上的普通中途 checkpoint。",
        "",
        "| budget | down_none | down_diag | newton_full | muon |",
        "|---|---:|---:|---:|---:|",
    ]
    for budget in contract_global["analysis"]["primary_budgets"]:
        lines.append(
            "| "
            + budget
            + " | "
            + " | ".join(f"{means[(budget, method)]:.6f}" for method in contract_global["grid"]["methods"])
            + " |"
        )
    lines.extend(
        [
            "",
            "解释边界：本实验检验同一 LLaMA-1B 架构内部的训练阶段效应；不能单独识别纯架构因果，也不解释 refresh harm 的来源。",
            "",
            "配对差异、逐 seed 终点、tail-5 和 normalized AUC 分别见 `paired_contrasts.csv` 与 `endpoint_results.csv`。",
            "",
        ]
    )
    return "\n".join(lines)


def build(run_dir: Path) -> dict[str, Any]:
    audit = audit_run(run_dir, full_checkpoint_hash=False)
    if not audit["passed"]:
        raise RuntimeError(f"EX48 integrity audit failed: {audit['checks']} failures={audit['phase_failures'][:3]}")
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    endpoint_rows = audit["endpoint_rows"]
    contrasts = paired_contrasts(endpoint_rows, audit["contract"])
    classification = classify(contrasts, audit["contract"])
    write_csv(
        analysis_dir / "endpoint_results.csv",
        list(endpoint_rows[0].keys()),
        endpoint_rows,
    )
    write_csv(
        analysis_dir / "paired_contrasts.csv",
        list(contrasts[0].keys()),
        contrasts,
    )
    P.atomic_json(analysis_dir / "classification.json", classification)
    (analysis_dir / "EX48_FORMAL_ANALYSIS.md").write_text(
        report_markdown(endpoint_rows, contrasts, classification), encoding="utf-8"
    )
    artifacts = {}
    for path in sorted(analysis_dir.iterdir()):
        if path.is_file() and path.name != "analysis_manifest.json":
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": P.sha256_file(path)}
    manifest = {
        "schema_version": "ex48_analysis_manifest_v1",
        "passed": True,
        "claim_eligible": True,
        "integrity_checks": audit["checks"],
        "classification": classification["classification"],
        "formal_units": 12,
        "formal_phases": 72,
        "primary_endpoints": 36,
        "contract_sha256": audit["contract_sha256"],
        "data_inventory_sha256": audit["data_inventory_sha256"],
        "checkpoint_hash_policy": "full hashes recorded at phase completion; final verify mode re-hashes all 36 retained endpoints",
        "artifacts": artifacts,
        "created_at": now_iso(),
    }
    P.atomic_json(analysis_dir / "analysis_manifest.json", manifest)
    handoff = build_handoff(run_dir, endpoint_rows)
    P.atomic_json(run_dir / "handoff_manifest.json", handoff)
    return manifest


def verify(run_dir: Path, full_checkpoint_hash: bool) -> dict[str, Any]:
    audit = audit_run(run_dir, full_checkpoint_hash=full_checkpoint_hash)
    analysis_manifest = P.read_json(run_dir / "analysis" / "analysis_manifest.json")
    analysis_checks = {
        name: (run_dir / "analysis" / name).is_file()
        and (run_dir / "analysis" / name).stat().st_size == int(item["bytes"])
        and P.sha256_file(run_dir / "analysis" / name) == item["sha256"]
        for name, item in analysis_manifest["artifacts"].items()
    }
    handoff = P.read_json(run_dir / "handoff_manifest.json")
    handoff_checks = {
        relative: (run_dir / relative).is_file()
        and (run_dir / relative).stat().st_size == int(item["bytes"])
        and P.sha256_file(run_dir / relative) == item["sha256"]
        for relative, item in handoff["small_artifacts"].items()
    }
    checks = audit["checks"] | {
        "analysis_manifest": analysis_manifest.get("passed") is True
        and analysis_manifest.get("claim_eligible") is True,
        "analysis_artifacts": bool(analysis_checks) and all(analysis_checks.values()),
        "handoff": handoff.get("passed") is True and bool(handoff_checks) and all(handoff_checks.values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "full_checkpoint_hash": full_checkpoint_hash,
    }


contract_global: dict[str, Any] = {}


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if args.mode == "build":
        result = build(run_dir)
    else:
        result = verify(run_dir, args.full_checkpoint_hash)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
