"""Finalize and hash-audit the partial method-deepening package."""

from __future__ import annotations

import argparse
import io
import unittest
from pathlib import Path
from typing import Any

from common import ContractError, commit_manifest, ensure_new_output, read_json, sha256_file


SCHEMA = "mdp_method_deepening_root_manifest_v2"
EXPECTED_HANDOFF_SHA256 = "76b407da7d779fb719f8c8069f0f68464b5554c065f154585eed251eecd3dd48"


def finalize(
    source_root: Path,
    package_root: Path,
    handoff_path: Path,
) -> dict[str, Any]:
    scripts_root = source_root / "scripts" / "mdp_submission_analysis"
    docs_root = source_root / "docs" / "method_deepening"
    component_paths = {
        "formulation": package_root / "bundle" / "formulation" / "method_formulation_manifest.json",
        "complexity": package_root / "bundle" / "complexity" / "routing_complexity_manifest.json",
        "equivariance": package_root / "bundle" / "equivariance" / "route_equivariance_manifest.json",
        "inventory": package_root / "bundle" / "inventory" / "method_deepening_inventory_manifest.json",
        "synthesis_v2": package_root / "bundle" / "synthesis_v2" / "method_deepening_package_manifest.json",
    }
    expected_status = {
        "formulation": "passed",
        "complexity": "passed",
        "equivariance": "passed",
        "inventory": "passed_inventory",
        "synthesis_v2": "partial",
    }
    documents = [
        docs_root / "SELECTIVE_ROUTING_FORMULATION.md",
        docs_root / "ROUTING_COMPLEXITY_AND_EQUIVARIANCE.md",
        docs_root / "REFRESH_STABILITY_ANALYSIS.md",
        docs_root / "METHOD_DEEPENING_AUDIT.md",
    ]
    package_files = [
        package_root / "PACKAGE_INDEX.md",
        package_root / "routing_complexity.json",
        package_root / "refresh_replay_contract.json",
    ]
    for path in [*component_paths.values(), *documents, *package_files]:
        if not path.is_file():
            raise ContractError(f"method-deepening finalization input is missing: {path}")
    components = {name: read_json(path) for name, path in component_paths.items()}
    for name, expected in expected_status.items():
        observed = str(components[name].get("status"))
        if observed != expected:
            raise ContractError(f"{name} status={observed!r}, expected={expected!r}")
    if components["inventory"].get("mdp04_status") != "blocked_data":
        raise ContractError("inventory no longer matches the partial MDP-04 package")
    if components["synthesis_v2"].get("component_status", {}).get("MDP-04") != "blocked_data":
        raise ContractError("synthesis no longer marks MDP-04 blocked_data")

    if not handoff_path.is_file():
        raise ContractError(
            "the frozen private HANDOFF snapshot must be supplied explicitly: "
            f"{handoff_path}"
        )
    handoff_hash = sha256_file(handoff_path)
    if handoff_hash != EXPECTED_HANDOFF_SHA256:
        raise ContractError("HANDOFF.md changed relative to the user-frozen snapshot")

    suite = unittest.defaultTestLoader.discover(str(scripts_root), pattern="test_*.py")
    stream = io.StringIO()
    test_result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    if not test_result.wasSuccessful():
        raise ContractError("method-deepening regression suite failed:\n" + stream.getvalue())

    manifest_name = "method_deepening_root_manifest_v2.json"
    ensure_new_output(package_root, manifest_name)
    source_paths = {
        **{f"component/{name}": path for name, path in component_paths.items()},
        **{f"document/{path.name}": path for path in documents},
    }
    script_names = [
        "analyze_refresh_stability.py",
        "audit_method_formulation.py",
        "audit_route_equivariance.py",
        "audit_routing_complexity.py",
        "audit_submission_bundle.py",
        "build_method_deepening_package.py",
        "finalize_method_deepening_package.py",
        "inventory_method_deepening_artifacts.py",
        "run_local_pipeline.py",
        "test_local_pipeline.py",
    ]
    for name in script_names:
        source_paths[f"script/{name}"] = scripts_root / name
    result = {
        "schema_version": SCHEMA,
        "status": "partial",
        "claim_eligible": False,
        "authoritative_synthesis": "bundle/synthesis_v2",
        "supersedes_root_manifest": "method_deepening_root_manifest.json",
        "component_status": {
            "MDP-01": "ready",
            "MDP-02": "ready",
            "MDP-03": "ready_with_rank_conditions",
            "MDP-04": "blocked_data",
        },
        "regression_tests_run": test_result.testsRun,
        "regression_tests_passed": test_result.wasSuccessful(),
        "regression_failures": len(test_result.failures),
        "regression_errors": len(test_result.errors),
        "handoff_sha256": handoff_hash,
        "handoff_unchanged": True,
        "accepted_experiment_mutation_performed": False,
        "large_training_launched": False,
        "source_sha256": {
            name: sha256_file(path) for name, path in sorted(source_paths.items())
        },
        "completion_blocker": "real paired MECH-09R refresh matrices and matched-gradient export are not local",
        "next_action": "deterministic_short_replay_or_streaming_metric_export_on_original_llama_host",
    }
    commit_manifest(
        package_root,
        manifest_name,
        result,
        ["PACKAGE_INDEX.md", "routing_complexity.json", "refresh_replay_contract.json"],
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="portable public source root; defaults to the repository containing this script",
    )
    parser.add_argument(
        "--handoff",
        type=Path,
        required=True,
        help="explicit path to the frozen private HANDOFF.md snapshot",
    )
    parser.add_argument("--package-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = finalize(
        args.source_root.resolve(),
        args.package_root.resolve(),
        args.handoff.resolve(),
    )
    print(
        f"method-deepening package finalized: status={result['status']} "
        f"tests={result['regression_tests_run']} handoff_unchanged={result['handoff_unchanged']}"
    )


if __name__ == "__main__":
    main()
