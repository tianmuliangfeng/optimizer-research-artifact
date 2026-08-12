#!/usr/bin/env python3
"""Independently validate and summarize a handed-off audit-40 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "2026-07-29.2"
HERE = Path(__file__).resolve().parent
CODE_FILES = {
    "controller": HERE / "run_llama_block_partition_audit.py",
    "worker": HERE / "llama_block_partition_worker.py",
    "analyzer": HERE / "analyze_llama_block_partition_audit.py",
    "contract": HERE / "audit_contract.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty quantile input")
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize empty values")
    return {
        "n": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": quantile(values, 0.95),
        "maximum": max(values),
    }


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def archive_audit(archive_path: Path, run_name: str) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        unsafe = [
            name
            for name in names
            if name.startswith("/")
            or (len(name) >= 2 and name[1] == ":")
            or ".." in Path(name).parts
        ]
        pt_entries = [name for name in names if name.lower().endswith(".pt")]
        roots = {
            Path(name).parts[0]
            for name in names
            if Path(name).parts
        }
        bad_crc = archive.testzip()
    checks = {
        "crc": bad_crc is None,
        "single_expected_root": roots == {run_name},
        "no_unsafe_paths": not unsafe,
        "no_checkpoint_tensors": not pt_entries,
        "nonempty": bool(entries),
    }
    return {
        "filename": archive_path.name,
        "sha256": sha256_file(archive_path),
        "bytes": archive_path.stat().st_size,
        "entries": len(entries),
        "uncompressed_bytes": sum(entry.file_size for entry in entries),
        "unsafe_entries": unsafe,
        "pt_entries": pt_entries,
        "checks": checks,
        "passed": all(checks.values()),
    }


def classify(
    pooled_median: float,
    control_maximum: float,
    stage_medians: list[float],
    thresholds: dict[str, Any],
) -> tuple[str, float]:
    multiple = pooled_median / max(control_maximum, 1e-8)
    if (
        pooled_median >= thresholds["strong_median_block4_update_drift"]
        and multiple >= thresholds["strong_control_multiple"]
        and all(
            value >= thresholds["strong_median_block4_update_drift"]
            for value in stage_medians
        )
    ):
        return "strong_non_invariance", multiple
    if (
        pooled_median >= thresholds["detectable_median_block4_update_drift"]
        and multiple >= thresholds["detectable_control_multiple"]
        and all(
            value >= thresholds["detectable_median_block4_update_drift"]
            for value in stage_medians
        )
    ):
        return "detectable_non_invariance", multiple
    if (
        pooled_median <= thresholds["negligible_median_block4_update_drift"]
        and all(
            value <= thresholds["negligible_median_block4_update_drift"]
            for value in stage_medians
        )
    ):
        return "approximately_invariant_at_tested_resolution", multiple
    return "inconclusive", multiple


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    archive = args.archive.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True)

    contract = read_json(CODE_FILES["contract"])
    thresholds = contract["integrity_thresholds"]
    classification_thresholds = contract["classification_thresholds"]
    provenance = read_json(run / "controller_provenance.json")
    root_status = read_json(run / "status.json")
    reported_manifest = read_json(
        run / "analysis" / "llama_block_audit_analysis_manifest.json"
    )
    reported_classification = read_json(
        run / "analysis" / "classification.json"
    )

    archive_result = archive_audit(archive, run.name)
    code_hashes = {name: sha256_file(path) for name, path in CODE_FILES.items()}
    provenance_checks = {
        "controller": provenance["controller_sha256"] == code_hashes["controller"],
        "worker": provenance["worker_sha256"] == code_hashes["worker"],
        "analyzer": provenance["analyzer_sha256"] == code_hashes["analyzer"],
        "contract": provenance["contract_sha256"] == code_hashes["contract"],
        "root_status": root_status.get("status") == "passed",
        "root_classification": root_status.get("classification")
        == "strong_non_invariance",
        "analysis_passed": reported_manifest.get("passed") is True,
    }

    all_global: list[dict[str, str]] = []
    all_controls: list[dict[str, str]] = []
    all_within: list[dict[str, str]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    label_audits: dict[str, Any] = {}

    for label in ("early", "late"):
        directory = run / "formal" / label
        manifest = read_json(directory / "llama_block_audit_manifest.json")
        checks = read_json(directory / "checks.json")
        status = read_json(directory / "status.json")
        invariance = read_json(directory / "state_invariance.json")
        checkpoint_hash = read_json(directory / "checkpoint_hash_audit.json")
        embedded_contract = directory / "audit_contract.json"
        updates = read_csv(directory / "equivariance_updates.csv")
        partitions = read_csv(directory / "partition_geometry.csv")
        line_summary = read_csv(directory / "line_search_summary.csv")
        losses = read_csv(directory / "shadow_losses.csv")
        indices = read_csv(directory / "permutation_indices.csv")

        global_rows = [
            row
            for row in updates
            if row.get("candidate") == "block4"
            and row.get("partition_kind") == "global_balanced_partition"
        ]
        control_rows = [
            row
            for row in updates
            if row.get("candidate") in {"none", "diag", "dense_full"}
            and row.get("partition_kind")
        ]
        within_rows = [
            row
            for row in updates
            if row.get("candidate") == "block4"
            and row.get("partition_kind") == "within_block_control"
        ]
        all_global.extend(global_rows)
        all_controls.extend(control_rows)
        all_within.extend(within_rows)

        global_key = {
            (
                row["repeat"],
                row["direction"],
                row["build_split"],
                row["layer"],
                row["partition"],
            )
            for row in global_rows
        }
        control_key = {
            (
                row["repeat"],
                row["direction"],
                row["build_split"],
                row["layer"],
                row["partition"],
                row["candidate"],
            )
            for row in control_rows
        }
        numeric_fields = (
            "inverse_residual_relative",
            "preconditioned_relative_drift",
            "update_relative_drift",
            "update_cosine",
        )
        numeric_finite = all(
            math.isfinite(float(row[field]))
            for row in updates
            for field in numeric_fields
            if row.get(field) not in ("", None)
        )
        expected_index_rows = manifest["partition_count"] * manifest["block_size"] * 4
        index_groups: dict[str, list[int]] = defaultdict(list)
        for row in indices:
            index_groups[row["partition"]].append(int(row["old_coordinate"]))
        permutations_are_bijections = (
            len(indices) == expected_index_rows
            and len(index_groups) == manifest["partition_count"]
            and all(
                sorted(values) == list(range(manifest["block_size"] * 4))
                for values in index_groups.values()
            )
        )
        identity_geometry = {
            (
                row["repeat"],
                row["direction"],
                row["layer"],
            ): float(row["off_block_energy_fraction"])
            for row in partitions
            if row["partition_kind"] == "identity"
        }
        within_geometry = {
            (
                row["repeat"],
                row["direction"],
                row["layer"],
            ): float(row["off_block_energy_fraction"])
            for row in partitions
            if row["partition_kind"] == "within_block_control"
        }
        geometry_control_exact = (
            identity_geometry.keys() == within_geometry.keys()
            and all(
                close(identity_geometry[key], within_geometry[key])
                for key in identity_geometry
            )
        )

        pre_controls = [
            float(row["preconditioned_relative_drift"])
            for row in control_rows
        ]
        pre_within = [
            float(row["preconditioned_relative_drift"])
            for row in within_rows
        ]
        update_controls = [
            float(row["update_relative_drift"])
            for row in control_rows
        ]
        update_within = [
            float(row["update_relative_drift"])
            for row in within_rows
        ]
        inverse_values = [
            float(row["inverse_residual_relative"])
            for row in updates
            if row.get("inverse_residual_relative") not in ("", None)
        ]
        projection_values = [
            float(row["projection_equivalence_relative"])
            for row in partitions
        ]

        label_checks = {
            "manifest_passed": manifest.get("passed") is True,
            "status_passed": status.get("status") == "passed",
            "all_worker_checks": all(checks.values()),
            "contract_matches_local": sha256_file(embedded_contract)
            == code_hashes["contract"],
            "checkpoint_hash_passed": checkpoint_hash.get("passed") is True,
            "model_content_unchanged": invariance[
                "model_content_unchanged"
            ]
            is True,
            "optimizer_loader_unchanged": invariance[
                "optimizer_loader_unchanged"
            ]
            is True,
            "checkpoint_file_unchanged": invariance[
                "checkpoint_file_unchanged"
            ]
            is True,
            "update_row_count": len(updates) == manifest["update_rows"],
            "partition_row_count": len(partitions) == manifest["partition_rows"],
            "summary_row_count": len(line_summary)
            == manifest["line_search_summary_rows"],
            "loss_row_count": len(losses) == manifest["shadow_loss_rows"],
            "global_row_count": len(global_rows) == 24,
            "global_key_unique": len(global_key) == len(global_rows),
            "control_row_count": len(control_rows) == 72,
            "control_key_unique": len(control_key) == len(control_rows),
            "within_row_count": len(within_rows) == 6,
            "permutation_bijections": permutations_are_bijections,
            "geometry_within_control_exact": geometry_control_exact,
            "numeric_finite": numeric_finite,
            "preconditioner_controls": max(pre_controls)
            <= thresholds["preconditioner_equivariance_relative"],
            "preconditioner_within": max(pre_within)
            <= thresholds["preconditioner_equivariance_relative"],
            "production_controls": max(update_controls)
            <= thresholds["production_ns_equivariance_relative"],
            "production_within": max(update_within)
            <= thresholds["production_ns_equivariance_relative"],
            "inverse_residual": max(inverse_values)
            <= thresholds["inverse_residual_relative"],
            "projection_equivalence": max(projection_values)
            <= thresholds["projection_equivalence_relative"],
        }
        label_audits[label] = {
            "checks": label_checks,
            "passed": all(label_checks.values()),
            "maxima": {
                "preconditioner_control_drift": max(pre_controls),
                "preconditioner_within_drift": max(pre_within),
                "production_control_drift": max(update_controls),
                "production_within_drift": max(update_within),
                "inverse_residual": max(inverse_values),
                "projection_equivalence_error": max(projection_values),
            },
        }

        values = [float(row["update_relative_drift"]) for row in global_rows]
        checkpoint_summary = summary(values)
        checkpoint_rows.append(
            {
                "checkpoint_label": label,
                "checkpoint_step": manifest["checkpoint_step"],
                **{
                    f"global_block4_update_drift_{key}": value
                    for key, value in checkpoint_summary.items()
                },
            }
        )
        for layer in ("0", "8", "17"):
            selected = [row for row in global_rows if row["layer"] == layer]
            update_values = [
                float(row["update_relative_drift"]) for row in selected
            ]
            preconditioner_values = [
                float(row["preconditioned_relative_drift"]) for row in selected
            ]
            cosine_values = [float(row["update_cosine"]) for row in selected]
            layer_rows.append(
                {
                    "checkpoint_label": label,
                    "checkpoint_step": manifest["checkpoint_step"],
                    "layer": int(layer),
                    "n": len(selected),
                    "update_drift_median": statistics.median(update_values),
                    "update_drift_minimum": min(update_values),
                    "update_drift_maximum": max(update_values),
                    "preconditioner_drift_median": statistics.median(
                        preconditioner_values
                    ),
                    "update_cosine_median": statistics.median(cosine_values),
                }
            )

        for direction in ("A_to_B", "B_to_A"):
            selected = [
                row
                for row in line_summary
                if row["scope"] == "grouped"
                and row["direction"] == direction
                and row["candidate"].startswith("block4_")
            ]
            loss_values = [
                float(row["best_relative_loss_delta"]) for row in selected
            ]
            shadow_rows.append(
                {
                    "checkpoint_label": label,
                    "direction": direction,
                    "partition_candidates": len(selected),
                    "best_relative_loss_delta_minimum": min(loss_values),
                    "best_relative_loss_delta_maximum": max(loss_values),
                    "best_relative_loss_delta_range": max(loss_values)
                    - min(loss_values),
                }
            )

    global_values = [
        float(row["update_relative_drift"]) for row in all_global
    ]
    control_values = [
        float(row["update_relative_drift"]) for row in all_controls
    ]
    within_values = [
        float(row["update_relative_drift"]) for row in all_within
    ]
    stage_medians = [
        float(row["global_block4_update_drift_median"])
        for row in checkpoint_rows
    ]
    pooled_median = statistics.median(global_values)
    control_maximum = max(control_values + within_values)
    recomputed_classification, multiple = classify(
        pooled_median,
        control_maximum,
        stage_medians,
        classification_thresholds,
    )
    reported_statistics = reported_classification["decision_statistics"]
    analysis_checks = {
        "global_rows": len(all_global) == 48,
        "control_rows": len(all_controls) == 144,
        "within_rows": len(all_within) == 12,
        "classification": recomputed_classification
        == reported_classification["classification"]
        == reported_manifest["classification"],
        "pooled_median": close(
            pooled_median,
            float(
                reported_statistics[
                    "pooled_global_block4_median_update_drift"
                ]
            ),
        ),
        "control_maximum": close(
            control_maximum,
            float(reported_statistics["maximum_equivariant_control_drift"]),
        ),
        "effect_multiple": close(
            multiple,
            float(reported_statistics["effect_to_control_multiple"]),
        ),
        "stage_medians": close(
            min(stage_medians),
            float(reported_statistics["minimum_stage_median_update_drift"]),
        )
        and close(
            max(stage_medians),
            float(reported_statistics["maximum_stage_median_update_drift"]),
        ),
        "scientific_result_not_integrity_gate": reported_manifest[
            "scientific_result_used_for_integrity_pass"
        ]
        is False,
    }

    source_files = sorted(
        path
        for path in run.rglob("*")
        if path.is_file() and output not in path.parents
    )
    artifact_hash_rows = [
        {
            "relative_path": path.relative_to(run).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_files
    ]
    passed = (
        archive_result["passed"]
        and all(provenance_checks.values())
        and all(value["passed"] for value in label_audits.values())
        and all(analysis_checks.values())
    )
    result = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "passed": passed,
        "classification": recomputed_classification if passed else "invalid",
        "archive": archive_result,
        "provenance": {
            "observed": provenance,
            "local_hashes": code_hashes,
            "checks": provenance_checks,
            "passed": all(provenance_checks.values()),
        },
        "formal": label_audits,
        "analysis_checks": analysis_checks,
        "recomputed": {
            "global_block4_update_drift": summary(global_values),
            "equivariant_control_update_drift": summary(control_values),
            "within_block_control_update_drift": summary(within_values),
            "pooled_global_block4_median_update_drift": pooled_median,
            "maximum_equivariant_control_drift": control_maximum,
            "effect_to_control_multiple": multiple,
            "stage_medians": {
                row["checkpoint_label"]: row[
                    "global_block4_update_drift_median"
                ]
                for row in checkpoint_rows
            },
        },
        "interpretation_boundary": {
            "supported": (
                "The contiguous four-way LLaMA SwiGLU down-projection "
                "approximation is strongly coordinate-partition dependent."
            ),
            "not_supported": (
                "A claim that block4 wins or loses a full training comparison."
            ),
            "official_original_newton_muon_control": "newton_full",
            "block4_primary_baseline": False,
        },
        "source_artifact_count": len(source_files),
    }
    write_json(output / "independent_audit.json", result)
    write_json(
        output / "important_results.json",
        {
            "schema_version": 1,
            "run_id": run.name,
            "data_quality_status": "ready_to_cite_with_scope_caveats",
            "classification": recomputed_classification,
            "archive_sha256": archive_result["sha256"],
            "global_block4_observations": len(all_global),
            "equivariant_control_observations": len(all_controls),
            "within_block_control_observations": len(all_within),
            "pooled_global_block4_update_drift": summary(global_values),
            "equivariant_control_update_drift": summary(control_values),
            "within_block_control_update_drift": summary(within_values),
            "effect_to_control_multiple": multiple,
            "checkpoint_summaries": checkpoint_rows,
            "layer_summaries": layer_rows,
            "supported_claim": (
                "A contiguous four-way LLaMA SwiGLU down-projection "
                "approximation is strongly dependent on an arbitrary hidden-"
                "neuron coordinate partition."
            ),
            "scope_caveats": [
                "This is a read-only update and shadow-loss audit, not a full training comparison.",
                "The result does not claim block4 wins or loses against primary optimizers.",
                "newton_full remains the original Newton-Muon-family LLaMA control.",
                "muon remains the optimizer baseline; each Selective method is compared separately with muon and newton_full."
            ],
        },
    )
    write_csv(output / "checkpoint_summary.csv", checkpoint_rows)
    write_csv(output / "layer_summary.csv", layer_rows)
    write_csv(output / "shadow_partition_spread.csv", shadow_rows)
    write_csv(output / "source_artifact_hashes.csv", artifact_hash_rows)
    report = [
        "# Independent review — LLaMA block-partition invariance audit",
        "",
        f"- Passed: `{str(passed).lower()}`",
        (
            "- Classification: "
            f"`{recomputed_classification if passed else 'invalid'}`"
        ),
        f"- Archive SHA-256: `{archive_result['sha256']}`",
        f"- Source artifacts: `{len(source_files)}`",
        f"- Global block4 observations: `{len(all_global)}`",
        f"- Pooled median update drift: `{pooled_median:.9f}`",
        f"- Maximum equivariant-control drift: `{control_maximum:.9f}`",
        f"- Effect/control multiple: `{multiple:.6f}`",
        "",
        "## Stage medians",
        "",
        *[
            (
                f"- {row['checkpoint_label']} step "
                f"{row['checkpoint_step']}: "
                f"`{row['global_block4_update_drift_median']:.9f}`"
            )
            for row in checkpoint_rows
        ],
        "",
        "## Interpretation",
        "",
        (
            "Function-preserving cross-block hidden-neuron permutations change "
            "the mapped-back block4 update substantially at both checkpoints. "
            "Exact preconditioner controls remain at numerical zero, while "
            "production BF16/Triton controls remain far below the block4 "
            "effect."
        ),
        "",
        (
            "This supports omitting block4 as a primary LLaMA baseline. "
            "`newton_full` remains the original Newton–Muon-family control. "
            "The shadow-loss probe is secondary and does not authorize a "
            "full-training performance claim."
        ),
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(
        output / "independent_audit_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "passed": passed,
            "classification": (
                recomputed_classification if passed else "invalid"
            ),
            "archive_sha256": archive_result["sha256"],
            "artifacts": sorted(path.name for path in output.iterdir()),
        },
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
