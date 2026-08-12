"""Run the frozen local analysis pipeline from a single JSON configuration."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from analyze_cross_scale_pareto import analyze as analyze_cross_scale
from analyze_refresh_stability import analyze as analyze_refresh
from audit_route_equivariance import audit as audit_equivariance
from audit_routing_complexity import audit as audit_complexity
from audit_submission_bundle import audit as audit_bundle
from build_evidence_ledger import build_ledger
from build_submission_tables_figures import build as build_figures
from common import (
    ContractError,
    commit_manifest,
    ensure_new_output,
    read_json,
    resolve_input,
    sha256_file,
)


CONFIG_SCHEMA = "mdp_local_pipeline_config_v1"
PIPELINE_SCHEMA = "mdp_local_pipeline_run_v1"


def run(config_path: Path, output_root: Path, dry_run: bool = False) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ContractError(f"pipeline config schema must be {CONFIG_SCHEMA}")
    synthetic = bool(config.get("synthetic", False))
    for required in ("catalog", "complexity_config", "equivariance", "refresh"):
        if required not in config:
            raise ContractError(f"pipeline config is missing section: {required}")
    catalog_path = resolve_input(config_path, str(config["catalog"]))
    complexity_path = resolve_input(config_path, str(config["complexity_config"]))
    refresh_config = config["refresh"]
    if not isinstance(refresh_config, dict) or "enabled" not in refresh_config:
        raise ContractError("refresh section must explicitly set enabled=true/false")

    ledger_validation = build_ledger(catalog_path, output_root / "evidence", dry_run=True)
    if bool(ledger_validation["synthetic"]) != synthetic:
        raise ContractError("pipeline synthetic flag disagrees with evidence catalog")
    complexity_document = read_json(complexity_path)
    if bool(complexity_document.get("synthetic", False)) != synthetic:
        raise ContractError("pipeline synthetic flag disagrees with complexity config")
    if bool(refresh_config["enabled"]):
        for required in ("snapshots", "metadata"):
            if required not in refresh_config:
                raise ContractError(f"enabled refresh section is missing {required}")
        snapshots_path = resolve_input(config_path, str(refresh_config["snapshots"]))
        metadata_path = resolve_input(config_path, str(refresh_config["metadata"]))
        if not snapshots_path.is_file() or not metadata_path.is_file():
            raise ContractError("enabled refresh inputs do not exist")
        snapshot_manifest_path = None
        if bool(refresh_config.get("formal_contract", False)):
            if "snapshot_manifest" not in refresh_config:
                raise ContractError("formal refresh section requires snapshot_manifest")
            snapshot_manifest_path = resolve_input(
                config_path, str(refresh_config["snapshot_manifest"])
            )
            if not snapshot_manifest_path.is_file():
                raise ContractError("formal refresh snapshot_manifest does not exist")
    elif not str(refresh_config.get("reason", "")).strip():
        raise ContractError("disabled refresh section must record a reason")

    if dry_run:
        # Exercise the complete pipeline in an automatically removed directory.
        # This validates downstream schemas and numerical audits while honoring
        # the guarantee that --output-root remains untouched.
        with tempfile.TemporaryDirectory(prefix="mdp_pipeline_dryrun_") as temporary:
            audited = run(config_path, Path(temporary), dry_run=False)
        return {
            "schema_version": PIPELINE_SCHEMA,
            "status": "validated",
            "dry_run": True,
            "synthetic": synthetic,
            "claim_eligible": False,
            "refresh_enabled": bool(refresh_config["enabled"]),
            "source_count": ledger_validation["source_count"],
            "row_count": ledger_validation["row_count"],
            "temporary_bundle_manifest_count": audited["bundle_manifest_count"],
        }

    manifest_name = "local_pipeline_manifest.json"
    ensure_new_output(output_root, manifest_name)
    build_ledger(catalog_path, output_root / "evidence")
    analyze_cross_scale(
        output_root / "evidence",
        output_root / "cross_scale",
        float(config.get("practical_loss_margin", 0.002)),
        str(config.get("cross_scale_family", "gpt_scale")),
    )
    build_figures(output_root / "cross_scale", output_root / "figures")
    audit_complexity(complexity_path, output_root / "complexity")
    equivariance = config["equivariance"]
    audit_equivariance(
        output_root / "equivariance",
        int(equivariance.get("seed", 20240731)),
        int(equivariance.get("input_dim", 12)),
        int(equivariance.get("output_dim", 8)),
        int(equivariance.get("blocks", 3)),
        float(equivariance.get("tolerance", 1e-9)),
        float(equivariance.get("cross_block_minimum_drift", 1e-5)),
        synthetic,
    )
    stage_manifests = [
        output_root / "evidence" / "evidence_ledger_manifest.json",
        output_root / "cross_scale" / "cross_scale_analysis_manifest.json",
        output_root / "figures" / "submission_tables_figures_manifest.json",
        output_root / "complexity" / "routing_complexity_manifest.json",
        output_root / "equivariance" / "route_equivariance_manifest.json",
    ]
    if bool(refresh_config["enabled"]):
        analyze_refresh(
            snapshots_path,
            metadata_path,
            output_root / "refresh",
            float(refresh_config.get("ridge_scale", 0.2)),
            float(refresh_config.get("ridge_epsilon", 1e-8)),
            float(refresh_config.get("residual_tolerance", 1e-8)),
            synthetic,
            bool(refresh_config.get("formal_contract", False)),
            snapshot_manifest_path,
            float(refresh_config.get("runtime_inverse_tolerance", 5e-3)),
            float(refresh_config.get("symmetry_tolerance", 1e-6)),
        )
        stage_manifests.append(output_root / "refresh" / "refresh_stability_manifest.json")

    result = {
        "schema_version": PIPELINE_SCHEMA,
        "status": "passed",
        "dry_run": False,
        "synthetic": synthetic,
        "claim_eligible": not synthetic,
        "refresh_enabled": bool(refresh_config["enabled"]),
        "stage_manifest_sha256": {
            str(path.relative_to(output_root)): sha256_file(path) for path in stage_manifests
        },
    }
    commit_manifest(output_root, manifest_name, result, [])
    bundle_result = audit_bundle(output_root, output_root / "audit")
    result["bundle_manifest_count"] = bundle_result["manifest_count"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.config.resolve(), args.output_root.resolve(), args.dry_run)
    print(
        f"local pipeline {result['status']}: synthetic={result['synthetic']} "
        f"refresh={result['refresh_enabled']} dry_run={result['dry_run']}"
    )


if __name__ == "__main__":
    main()
