"""Normalize accepted experiment analyses into a provenance-preserving run ledger."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

from common import (
    MIB,
    ContractError,
    bool_cell,
    canonical_method,
    commit_manifest,
    ensure_new_output,
    optional_float,
    read_csv,
    read_json,
    required_float,
    resolve_input,
    sha256_file,
    validate_manifest_requirements,
    write_csv,
)


CATALOG_SCHEMA = "mdp_submission_input_catalog_v1"
LEDGER_SCHEMA = "mdp_evidence_ledger_v1"

LEDGER_FIELDS = [
    "source_id",
    "analysis_family",
    "scale_id",
    "architecture",
    "model_parameters",
    "train_tokens",
    "seed",
    "method",
    "method_raw",
    "run_name",
    "initial_val_loss",
    "final_val_loss",
    "best_val_loss",
    "tail5_mean",
    "normalized_auc",
    "k_state_bytes",
    "optimizer_state_bytes",
    "peak_memory_allocated_bytes",
    "timing_eligible",
    "evidence_status",
    "synthetic",
    "source_csv",
    "source_csv_sha256",
    "source_manifest",
    "source_manifest_sha256",
]

SOURCE_FIELDS = [
    "source_id",
    "schema",
    "analysis_family",
    "scale_id",
    "row_count",
    "method_count",
    "seed_count",
    "synthetic",
    "claim_eligible",
    "csv_path",
    "csv_sha256",
    "manifest_path",
    "manifest_sha256",
]


SCHEMA_COLUMNS: dict[str, dict[str, str | None]] = {
    "record_cells_v1": {
        "seed": "seed",
        "method": "method",
        "run_name": None,
        "initial": "initial_val_loss",
        "final": "final_val_loss",
        "best": "best_val_loss",
        "tail5": "tail5_mean",
        "auc": "normalized_auc",
        "k_bytes": "k_state_bytes",
        "optimizer_bytes": "optimizer_state_bytes",
        "peak_bytes": "peak_memory_allocated_bytes",
        "timing": "timing_eligible",
    },
    "r1_core_run_summary_v1": {
        "seed": "seed",
        "method": "method",
        "run_name": "run_name",
        "initial": "initial_val_loss",
        "final": "final_val_loss",
        "best": "best_val_loss",
        "tail5": "tail5_val_loss_mean",
        "auc": "normalized_val_auc",
        "k_mib": "k_state_mib",
        "optimizer_mib": "optimizer_state_mib",
        "peak_mib": "peak_memory_mib",
        "timing": None,
    },
    "r1_unified_run_summary_v1": {
        "seed": "seed",
        "method": "method",
        "run_name": "run_name",
        "initial": "initial_val_loss",
        "final": "final_val_loss",
        "best": "best_val_loss",
        "tail5": "tail5_val_loss_mean",
        "auc": "normalized_val_auc",
        "k_mib": None,
        "optimizer_mib": "optimizer_state_mib",
        "peak_mib": "peak_memory_mib",
        "timing": "timing_eligible",
    },
    "mousse_unified_run_summary_v1": {
        "seed": "seed",
        "method": "method",
        "run_name": "run_name",
        "initial": "initial_val_loss",
        "final": "final_val_loss",
        "best": "best_val_loss",
        "tail5": "tail5_val_loss_mean",
        "auc": "normalized_val_auc",
        "k_mib": None,
        "optimizer_mib": "optimizer_state_mib",
        "peak_mib": "peak_memory_mib",
        "timing": "timing_eligible",
    },
    "llama1b_run_summary_v1": {
        "seed": "seed",
        "method": "method",
        "run_name": "run_name",
        "initial": "initial_val_loss",
        "final": "final_val_loss",
        "best": "best_val_loss",
        "tail5": "tail5_val_loss_mean",
        "auc": "normalized_val_auc_0_6200",
        "k_mib": "expected_k_state_mib_from_preflight",
        "optimizer_mib": None,
        "peak_mib": None,
        "timing": "timing_usable_for_paper",
    },
    "normalized_run_summary_v1": {
        "seed": "seed",
        "method": "method",
        "run_name": "run_name",
        "initial": "initial_val_loss",
        "final": "final_val_loss",
        "best": "best_val_loss",
        "tail5": "tail5_mean",
        "auc": "normalized_auc",
        "k_bytes": "k_state_bytes",
        "optimizer_bytes": "optimizer_state_bytes",
        "peak_bytes": "peak_memory_allocated_bytes",
        "timing": "timing_eligible",
    },
}


def _field(row: Mapping[str, str], name: str | None) -> str:
    return "" if name is None else row.get(name, "")


def _bytes(row: Mapping[str, str], columns: Mapping[str, str | None], base: str) -> float:
    direct = columns.get(f"{base}_bytes")
    mib = columns.get(f"{base}_mib")
    if direct:
        return optional_float(row, direct)
    if mib:
        value = optional_float(row, mib)
        return value * MIB if math.isfinite(value) else math.nan
    return math.nan


def _normalize_row(
    row: Mapping[str, str], source: Mapping[str, Any], csv_path: Path, manifest_path: Path
) -> dict[str, Any]:
    schema = str(source["schema"])
    columns = dict(SCHEMA_COLUMNS[schema])
    columns.update(source.get("column_overrides", {}))
    context = f"{source['source_id']} row {row!r}"
    method_raw = _field(row, columns["method"])
    seed = _field(row, columns["seed"])
    if not seed:
        raise ContractError(f"{context}: empty seed")
    if not method_raw:
        raise ContractError(f"{context}: empty method")
    return {
        "source_id": source["source_id"],
        "analysis_family": source["analysis_family"],
        "scale_id": source["scale_id"],
        "architecture": source.get("architecture", ""),
        "model_parameters": source.get("model_parameters", ""),
        "train_tokens": source.get("train_tokens", ""),
        "seed": seed,
        "method": canonical_method(method_raw),
        "method_raw": method_raw,
        "run_name": _field(row, columns.get("run_name")),
        "initial_val_loss": required_float(row, str(columns["initial"]), context),
        "final_val_loss": required_float(row, str(columns["final"]), context),
        "best_val_loss": required_float(row, str(columns["best"]), context),
        "tail5_mean": required_float(row, str(columns["tail5"]), context),
        "normalized_auc": required_float(row, str(columns["auc"]), context),
        "k_state_bytes": _bytes(row, columns, "k"),
        "optimizer_state_bytes": _bytes(row, columns, "optimizer"),
        "peak_memory_allocated_bytes": _bytes(row, columns, "peak"),
        "timing_eligible": bool_cell(_field(row, columns.get("timing"))) if columns.get("timing") else False,
        "evidence_status": source.get("evidence_status", "accepted"),
        "synthetic": bool(source.get("synthetic", False)),
        "source_csv": str(csv_path),
        "source_csv_sha256": sha256_file(csv_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
    }


def build_ledger(catalog_path: Path, output_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    catalog = read_json(catalog_path)
    if catalog.get("schema_version") != CATALOG_SCHEMA:
        raise ContractError(
            f"catalog schema must be {CATALOG_SCHEMA!r}, got {catalog.get('schema_version')!r}"
        )
    sources = catalog.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ContractError("catalog must contain a non-empty sources list")

    catalog_synthetic = bool(catalog.get("synthetic", False))
    ledger_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    seen_keys: set[tuple[str, str, str, str]] = set()

    for source in sources:
        required = {"source_id", "schema", "analysis_family", "scale_id", "csv", "manifest"}
        missing = sorted(required - set(source))
        if missing:
            raise ContractError(f"source is missing fields: {missing}")
        source_id = str(source["source_id"])
        if source_id in source_ids:
            raise ContractError(f"duplicate source_id: {source_id}")
        source_ids.add(source_id)
        if source["schema"] not in SCHEMA_COLUMNS:
            raise ContractError(f"{source_id}: unsupported schema {source['schema']!r}")

        csv_path = resolve_input(catalog_path, str(source["csv"]))
        manifest_path = resolve_input(catalog_path, str(source["manifest"]))
        if not csv_path.is_file() or not manifest_path.is_file():
            raise ContractError(f"{source_id}: missing CSV or manifest: {csv_path}, {manifest_path}")
        source_manifest = read_json(manifest_path)
        validate_manifest_requirements(
            source_manifest, source.get("manifest_require", {}), source_id
        )
        source_copy = dict(source)
        source_copy["synthetic"] = catalog_synthetic or bool(source.get("synthetic", False))
        normalized = [
            _normalize_row(row, source_copy, csv_path, manifest_path) for row in read_csv(csv_path)
        ]
        if not normalized:
            raise ContractError(f"{source_id}: source CSV has no data rows")

        observed_methods = {row["method"] for row in normalized}
        observed_seeds = {str(row["seed"]) for row in normalized}
        if "expected_methods" in source:
            expected = {canonical_method(value) for value in source["expected_methods"]}
            if observed_methods != expected:
                raise ContractError(
                    f"{source_id}: method set mismatch; expected {sorted(expected)}, got {sorted(observed_methods)}"
                )
        if "expected_seeds" in source:
            expected_seeds = {str(value) for value in source["expected_seeds"]}
            if observed_seeds != expected_seeds:
                raise ContractError(
                    f"{source_id}: seed set mismatch; expected {sorted(expected_seeds)}, got {sorted(observed_seeds)}"
                )

        local_pairs: set[tuple[str, str]] = set()
        for row in normalized:
            pair = (str(row["seed"]), str(row["method"]))
            if pair in local_pairs:
                raise ContractError(f"{source_id}: duplicate seed/method cell: {pair}")
            local_pairs.add(pair)
            global_key = (
                str(row["analysis_family"]),
                str(row["scale_id"]),
                str(row["seed"]),
                str(row["method"]),
            )
            if global_key in seen_keys:
                raise ContractError(
                    "duplicate evidence cell across sources (family, scale, seed, method): "
                    f"{global_key}"
                )
            seen_keys.add(global_key)
        ledger_rows.extend(normalized)
        synthetic = bool(source_copy["synthetic"])
        source_rows.append(
            {
                "source_id": source_id,
                "schema": source["schema"],
                "analysis_family": source["analysis_family"],
                "scale_id": source["scale_id"],
                "row_count": len(normalized),
                "method_count": len(observed_methods),
                "seed_count": len(observed_seeds),
                "synthetic": synthetic,
                "claim_eligible": not synthetic,
                "csv_path": str(csv_path),
                "csv_sha256": sha256_file(csv_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )

    ledger_rows.sort(key=lambda row: (row["analysis_family"], row["scale_id"], str(row["seed"]), row["method"]))
    source_rows.sort(key=lambda row: row["source_id"])
    result = {
        "schema_version": LEDGER_SCHEMA,
        "status": "validated",
        "dry_run": dry_run,
        "synthetic": any(bool(row["synthetic"]) for row in ledger_rows),
        "claim_eligible": not any(bool(row["synthetic"]) for row in ledger_rows),
        "source_count": len(source_rows),
        "row_count": len(ledger_rows),
        "analysis_families": sorted({row["analysis_family"] for row in ledger_rows}),
        "scales": sorted({row["scale_id"] for row in ledger_rows}),
    }
    if dry_run:
        return result

    manifest_name = "evidence_ledger_manifest.json"
    ensure_new_output(output_dir, manifest_name)
    write_csv(output_dir / "evidence_ledger.csv", ledger_rows, LEDGER_FIELDS)
    write_csv(output_dir / "source_ledger.csv", source_rows, SOURCE_FIELDS)
    commit_manifest(
        output_dir,
        manifest_name,
        result,
        ["evidence_ledger.csv", "source_ledger.csv"],
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_ledger(args.catalog.resolve(), args.output_dir.resolve(), args.dry_run)
    print(
        f"evidence ledger {result['status']}: sources={result['source_count']} "
        f"rows={result['row_count']} synthetic={result['synthetic']} dry_run={result['dry_run']}"
    )


if __name__ == "__main__":
    main()
