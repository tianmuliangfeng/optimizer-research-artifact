#!/usr/bin/env python3
"""Independently audit and preserve the decision-relevant MECH-07 results.

This script is intentionally read-only with respect to the original experiment
artifacts.  It writes derived audit files under ``analysis/local_audit``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "2026-07-27.1"
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
SCOPES = ("family_core", "down", "all")
TIERS = ("smoke", "formal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--source-archive-sha256")
    parser.add_argument("--source-archive-bytes", type=int)
    parser.add_argument("--source-archive-entries", type=int)
    parser.add_argument("--source-archive-uncompressed-bytes", type=int)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        raise RuntimeError("cannot summarize an empty collection")
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


def all_true(value: dict[str, Any]) -> bool:
    return bool(value) and all(item is True for item in value.values())


def windows_disjoint(batch_contract: dict[str, Any]) -> bool:
    batches = batch_contract.get("batches")
    if not isinstance(batches, list) or not batches:
        return False
    intervals = sorted(
        (int(row["offset"]), int(row["exclusive_end"])) for row in batches
    )
    return all(left_end <= right_start for (_, left_end), (right_start, _) in zip(
        intervals, intervals[1:]
    ))


def count_csv(path: Path) -> int:
    return len(read_csv(path))


def format_delta(value: float) -> str:
    return f"{value:+.8e}"


def group_summaries(
    observations: Iterable[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    metadata: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in observations:
        key = tuple(row[name] for name in keys)
        grouped[key].append(float(row["relative_shadow_loss_delta_left_minus_right"]))
        metadata[key] = {
            "priority": row["priority"],
            "left_algorithm": row["left_algorithm"],
            "right_algorithm": row["right_algorithm"],
        }
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        identity = dict(zip(keys, key))
        rows.append({**identity, **metadata[key], **summarize(grouped[key])})
    return rows


def primary_stage_row(
    rows: list[dict[str, Any]], stage: str, contrast: str
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["checkpoint_stage"] == stage
        and row["scope"] == "all"
        and row["contrast"] == contrast
    ]
    if len(matches) != 1:
        raise RuntimeError(f"missing unique stage row: {stage} {contrast}")
    return matches[0]


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    contract_path = args.contract.resolve()
    contract = read_json(contract_path)
    analysis = run / "analysis"
    official_manifest = read_json(analysis / "mech07_analysis_manifest.json")
    output = analysis / "local_audit"
    output.mkdir(parents=False, exist_ok=True)

    checks: dict[str, bool] = {}
    checks["root_status_passed"] = read_json(run / "status.json").get("status") == "passed"
    preflight = read_json(run / "preflight.json")
    checks["preflight_passed"] = (
        preflight.get("passed") is True
        and all_true(preflight.get("checks", {}))
    )
    checks["checkpoint_inventory_passed"] = (
        read_json(run / "checkpoint_inventory.json").get("passed") is True
    )
    checks["official_analysis_passed"] = official_manifest.get("passed") is True
    checks["contract_hash_matches"] = (
        sha256_file(contract_path) == official_manifest.get("contract_sha256")
    )
    checks["diag_vs_none_not_primary"] = (
        official_manifest.get("diag_vs_none_primary") is False
    )

    official_hash_results: dict[str, bool] = {}
    for name, expected in official_manifest.get("output_sha256", {}).items():
        official_hash_results[name] = sha256_file(analysis / name) == expected
    checks["official_output_hashes_match"] = (
        bool(official_hash_results) and all(official_hash_results.values())
    )

    quality_rows: list[dict[str, Any]] = []
    formal_batch_hashes: set[str] = set()
    script_versions: set[str] = set()
    all_cell_tiers_passed = True
    all_row_counts_match = True
    all_windows_disjoint = True
    muon_identity_passed = True
    muon_state_entries = 0
    muon_momentum_tensors = 0

    observations: list[dict[str, Any]] = []
    for spec in contract["checkpoints"]:
        cell = str(spec["cell"])
        for tier in TIERS:
            directory = run / cell / tier
            manifest = read_json(directory / "mech07_manifest.json")
            tier_checks = read_json(directory / "checks.json")
            status = read_json(directory / "status.json")
            batch_contract = read_json(directory / "batch_contract.json")
            script_versions.add(str(manifest["script_version"]))

            expected_counts = {
                "primitive_update_rows": count_csv(
                    directory / "primitive_update_geometry.csv"
                ),
                "algorithm_update_rows": count_csv(
                    directory / "algorithm_update_geometry.csv"
                ),
                "line_search_summary_rows": count_csv(
                    directory / "line_search_summary.csv"
                ),
                "shadow_loss_rows": count_csv(directory / "shadow_losses.csv"),
            }
            row_counts_match = all(
                int(manifest[key]) == observed
                for key, observed in expected_counts.items()
            )
            identity_match = (
                manifest.get("cell") == cell
                and manifest.get("checkpoint_method") == spec["method"]
                and manifest.get("checkpoint_stage") == spec["stage"]
                and int(manifest.get("checkpoint_step")) == int(spec["step"])
            )
            disjoint = (
                batch_contract.get("all_windows_disjoint") is True
                and windows_disjoint(batch_contract)
            )
            passed = (
                manifest.get("passed") is True
                and all_true(tier_checks)
                and status.get("status") == "passed"
                and row_counts_match
                and identity_match
                and disjoint
            )
            all_cell_tiers_passed = all_cell_tiers_passed and passed
            all_row_counts_match = all_row_counts_match and row_counts_match
            all_windows_disjoint = all_windows_disjoint and disjoint
            quality_rows.append(
                {
                    "checkpoint_cell": cell,
                    "analysis_tier": tier,
                    "script_version": manifest["script_version"],
                    "manifest_passed": manifest.get("passed") is True,
                    "checks_all_true": all_true(tier_checks),
                    "status_passed": status.get("status") == "passed",
                    "identity_match": identity_match,
                    "row_counts_match": row_counts_match,
                    "batch_windows_disjoint": disjoint,
                    "passed": passed,
                }
            )
            if tier == "formal":
                formal_batch_hashes.add(sha256_file(directory / "batch_contract.json"))
                rows = read_csv(directory / "line_search_summary.csv")
                scores = {
                    (
                        row["scope"],
                        int(row["repeat"]),
                        row["direction"],
                        row["algorithm"],
                    ): float(row["best_relative_loss_delta"])
                    for row in rows
                }
                repeats = int(manifest["repeats"])
                for scope in SCOPES:
                    for name, left, right, priority in CONTRASTS:
                        for repeat in range(repeats):
                            for direction in ("A_to_B", "B_to_A"):
                                delta = (
                                    scores[(scope, repeat, direction, left)]
                                    - scores[(scope, repeat, direction, right)]
                                )
                                observations.append(
                                    {
                                        "checkpoint_cell": cell,
                                        "checkpoint_stage": spec["stage"],
                                        "checkpoint_method": spec["method"],
                                        "checkpoint_step": int(spec["step"]),
                                        "scope": scope,
                                        "priority": priority,
                                        "contrast": name,
                                        "left_algorithm": left,
                                        "right_algorithm": right,
                                        "repeat": repeat,
                                        "direction": direction,
                                        "relative_shadow_loss_delta_left_minus_right": delta,
                                    }
                                )
                if spec["method"] == "muon":
                    identity = read_json(directory / "method_identity_audit.json")
                    identity_passed = (
                        identity.get("passed") is True
                        and identity.get("muon_not_adamw_state_signature") is True
                        and identity.get("matrix_optimizer_state_keys") == ["momentum"]
                    )
                    muon_identity_passed = muon_identity_passed and identity_passed
                    muon_state_entries += int(
                        identity.get("matrix_optimizer_state_entries", 0)
                    )
                    muon_momentum_tensors += int(
                        identity.get("matrix_optimizer_momentum_tensors", 0)
                    )

    checks["all_16_cell_tiers_passed"] = all_cell_tiers_passed
    checks["all_row_counts_match_manifests"] = all_row_counts_match
    checks["all_batch_windows_disjoint"] = all_windows_disjoint
    checks["one_shared_formal_batch_contract"] = len(formal_batch_hashes) == 1
    checks["muon_identity_audits_passed"] = (
        muon_identity_passed
        and muon_state_entries == 252
        and muon_momentum_tensors == 252
    )
    checks["expected_independent_observations"] = len(observations) == 960
    checks["no_diag_vs_none_observations"] = not any(
        {row["left_algorithm"], row["right_algorithm"]}
        == {"selective_diag", "selective_none"}
        for row in observations
    )

    scope_rows = group_summaries(
        observations, ("checkpoint_stage", "scope", "contrast")
    )
    origin_rows = group_summaries(
        observations,
        (
            "checkpoint_cell",
            "checkpoint_stage",
            "checkpoint_method",
            "scope",
            "contrast",
        ),
    )

    official_stage_rows = read_csv(analysis / "stage_contrast_summary.csv")
    independent_all_rows = [
        row for row in scope_rows if row["scope"] == "all"
    ]
    official_lookup = {
        (row["checkpoint_stage"], row["contrast"]): row
        for row in official_stage_rows
    }
    independent_lookup = {
        (row["checkpoint_stage"], row["contrast"]): row
        for row in independent_all_rows
    }
    stage_matches = True
    for key, official in official_lookup.items():
        observed = independent_lookup[key]
        for field in ("mean_delta", "median_delta", "sd_delta"):
            stage_matches = stage_matches and abs(
                float(official[field]) - float(observed[field])
            ) < 1e-15
        for field in (
            "cells",
            "negative_cells_left_better",
            "positive_cells_left_worse",
            "near_zero_cells",
        ):
            stage_matches = stage_matches and int(official[field]) == int(
                observed[field]
            )
    checks["independent_stage_recompute_matches_official"] = stage_matches

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"MECH-07 local audit failed: {failed}")

    early_none_original = primary_stage_row(
        scope_rows, "early", "selective_none_vs_original_newton_muon"
    )
    early_none_muon = primary_stage_row(
        scope_rows, "early", "selective_none_vs_muon"
    )
    early_diag_original = primary_stage_row(
        scope_rows, "early", "selective_diag_vs_original_newton_muon"
    )
    late_primary = [
        row
        for row in scope_rows
        if row["checkpoint_stage"] == "late"
        and row["scope"] == "all"
        and row["priority"] == "primary"
    ]

    key_results = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "passed": True,
        "interpretation_rule": "negative delta favors the left algorithm",
        "archive": {
            "sha256": args.source_archive_sha256,
            "bytes": args.source_archive_bytes,
            "entries": args.source_archive_entries,
            "uncompressed_bytes": args.source_archive_uncompressed_bytes,
        },
        "data_quality": {
            "formal_cells": len(contract["checkpoints"]),
            "cell_tiers_audited": len(quality_rows),
            "independent_scope_contrast_observations": len(observations),
            "formal_batch_contract_sha256": next(iter(formal_batch_hashes)),
            "script_versions_present": sorted(script_versions),
            "muon_matrix_optimizer_state_entries_audited": muon_state_entries,
            "muon_momentum_tensors_audited": muon_momentum_tensors,
        },
        "early": {
            "selective_none_vs_original_newton_muon_median": early_none_original[
                "median_delta"
            ],
            "selective_none_vs_original_newton_muon_left_better_cells": (
                early_none_original["negative_cells_left_better"]
            ),
            "selective_none_vs_muon_median": early_none_muon["median_delta"],
            "selective_none_vs_muon_left_better_cells": early_none_muon[
                "negative_cells_left_better"
            ],
            "selective_diag_vs_original_newton_muon_median": early_diag_original[
                "median_delta"
            ],
            "selective_diag_vs_original_newton_muon_left_better_cells": (
                early_diag_original["negative_cells_left_better"]
            ),
        },
        "late": {
            "primary_all_scope_max_abs_median": max(
                abs(float(row["median_delta"])) for row in late_primary
            ),
            "primary_contrasts_with_stable_checkpoint_state_advantage": 1,
            "note": (
                "All primary all-scope medians are near zero and no contrast is "
                "stable across the four checkpoint origins."
            ),
        },
        "claim_boundary": {
            "supported": (
                "At the early LLaMA-1B checkpoint, removing down-projection "
                "preconditioning consistently improves the local update relative "
                "to original Newton-Muon, while retained dense family-core "
                "preconditioning leaves Selective-none worse than Muon."
            ),
            "not_supported": (
                "MECH-07 does not explain the late long-run Muon advantage and "
                "does not establish that one-step shadow loss causes the final "
                "training ranking."
            ),
        },
    }
    # The single late stable checkpoint-state count above is descriptive across
    # individual origins, not a stage-level stable result; compute it exactly.
    key_results["late"][
        "primary_contrasts_with_stable_checkpoint_state_advantage"
    ] = sum(
        1
        for row in read_csv(analysis / "checkpoint_contrast_summary.csv")
        if row["checkpoint_stage"] == "late"
        and row["priority"] == "primary"
        and row["stable_left_better"] == "True"
    )

    archive_audit = {
        "schema_version": 1,
        "source_archive_sha256": args.source_archive_sha256,
        "source_archive_bytes": args.source_archive_bytes,
        "source_archive_entries": args.source_archive_entries,
        "source_archive_uncompressed_bytes": args.source_archive_uncompressed_bytes,
        "archive_copied_into_workspace": False,
        "extracted_run_directory": run.name,
        "extracted_files_audited": sum(1 for path in run.rglob("*") if path.is_file()),
    }

    write_csv(output / "cell_quality_audit.csv", quality_rows)
    write_csv(output / "scope_contrast_summary.csv", scope_rows)
    write_csv(output / "origin_contrast_summary.csv", origin_rows)
    write_json(output / "key_results.json", key_results)
    write_json(output / "source_archive_audit.json", archive_audit)
    write_json(output / "checks.json", checks)

    report_lines = [
        "# MECH-07 independent local audit",
        "",
        "## Outcome",
        "",
        "The result package passes all integrity, provenance, identity, row-count, "
        "batch-disjointness, and independent-recomputation checks.",
        "",
        "## Decision-relevant result",
        "",
        "- Negative relative shadow-loss delta favors the left algorithm.",
        f"- Early Selective-none vs original Newton-Muon: median "
        f"{format_delta(float(early_none_original['median_delta']))}; "
        f"{early_none_original['negative_cells_left_better']}/"
        f"{early_none_original['cells']} local cells favor Selective-none.",
        f"- Early Selective-none vs Muon: median "
        f"{format_delta(float(early_none_muon['median_delta']))}; "
        f"{early_none_muon['negative_cells_left_better']}/"
        f"{early_none_muon['cells']} local cells favor Selective-none.",
        f"- Early Selective-diag vs original Newton-Muon: median "
        f"{format_delta(float(early_diag_original['median_delta']))}; "
        f"{early_diag_original['negative_cells_left_better']}/"
        f"{early_diag_original['cells']} local cells favor Selective-diag.",
        f"- Late primary all-scope contrasts: maximum absolute median "
        f"{max(abs(float(row['median_delta'])) for row in late_primary):.8e}; "
        "no contrast is stable across all four checkpoint origins.",
        "",
        "## Mechanism interpretation",
        "",
        "At the early checkpoint, the local Muon advantage decomposes into two "
        "penalties: dense preconditioning on the family-core targets and "
        "preconditioning on the down projection. Selective-none removes the "
        "second penalty and uniformly improves original Newton-Muon locally, "
        "but retains the first penalty and therefore still loses to Muon.",
        "",
        "At the late checkpoint, the local contrasts collapse to near-zero, "
        "origin-dependent values. This is a valid negative result: MECH-07 does "
        "not identify the mechanism behind the final long-run Muon advantage.",
        "",
        "## Evidence boundary",
        "",
        "These are matched, local counterfactual updates with fresh build-split "
        "covariance and shared checkpoint momentum. They support an early-phase "
        "mechanism decomposition. They do not establish long-horizon causality "
        "or replace the existing three-seed training comparison.",
        "",
        "## Audit inventory",
        "",
        f"- Formal checkpoint cells: {len(contract['checkpoints'])}",
        f"- Smoke/formal cell tiers audited: {len(quality_rows)}",
        f"- Independent scope-level observations: {len(observations)}",
        f"- Shared formal batch contract: `{next(iter(formal_batch_hashes))}`",
        f"- Script versions present: {', '.join(sorted(script_versions))}",
        f"- Source archive SHA-256: `{args.source_archive_sha256}`",
        "",
    ]
    (output / "MECH07_LOCAL_ARTIFACT_AUDIT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    artifacts = [
        "MECH07_LOCAL_ARTIFACT_AUDIT.md",
        "cell_quality_audit.csv",
        "checks.json",
        "key_results.json",
        "origin_contrast_summary.csv",
        "scope_contrast_summary.csv",
        "source_archive_audit.json",
    ]
    audit_manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "source_run": run.name,
        "source_contract_sha256": sha256_file(contract_path),
        "source_analysis_manifest_sha256": sha256_file(
            analysis / "mech07_analysis_manifest.json"
        ),
        "artifacts": artifacts,
        "output_sha256": {
            name: sha256_file(output / name) for name in artifacts
        },
    }
    write_json(output / "local_audit_manifest.json", audit_manifest)
    print(f"MECH-07 local audit PASS: {output}")
    print(f"MECH-07 local audit manifest: {output / 'local_audit_manifest.json'}")


if __name__ == "__main__":
    main()
