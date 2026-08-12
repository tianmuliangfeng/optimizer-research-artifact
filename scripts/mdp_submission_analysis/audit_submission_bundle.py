"""Verify hashes, statuses, and synthetic-claim gates for a local analysis bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    ContractError,
    atomic_write_text,
    commit_manifest,
    ensure_new_output,
    read_json,
    sha256_file,
    write_csv,
)


AUDIT_SCHEMA = "mdp_submission_bundle_audit_v1"
FIELDS = [
    "manifest",
    "schema_version",
    "status",
    "synthetic",
    "claim_eligible",
    "output_count",
    "hashes_passed",
]


def audit(bundle_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir_resolved = output_dir.resolve()
    manifests = []
    for path in sorted(bundle_dir.rglob("*_manifest.json")):
        try:
            if path.resolve().is_relative_to(output_dir_resolved):
                continue
        except ValueError:
            pass
        document = read_json(path)
        if document.get("mdp_manifest") is True:
            manifests.append((path, document))
    if not manifests:
        raise ContractError(f"no MDP manifests found under {bundle_dir}")

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    any_synthetic = False
    all_claim_eligible = True
    for path, document in manifests:
        status = str(document.get("status", ""))
        if status not in {"passed", "validated", "passed_inventory", "partial"}:
            errors.append(f"non-passing status in {path}: {status!r}")
        synthetic = bool(document.get("synthetic", False))
        claim_eligible = bool(document.get("claim_eligible", False))
        any_synthetic = any_synthetic or synthetic
        all_claim_eligible = all_claim_eligible and claim_eligible
        if synthetic and claim_eligible:
            errors.append(f"synthetic manifest incorrectly marked claim-eligible: {path}")
        outputs = document.get("outputs")
        if not isinstance(outputs, dict):
            errors.append(f"manifest outputs is not an object: {path}")
            outputs = {}
        hashes_passed = True
        for relative_name, expected_hash in outputs.items():
            artifact = path.parent / relative_name
            if not artifact.is_file():
                hashes_passed = False
                errors.append(f"missing committed artifact: {artifact}")
            elif sha256_file(artifact) != expected_hash:
                hashes_passed = False
                errors.append(f"hash mismatch for committed artifact: {artifact}")
        rows.append(
            {
                "manifest": str(path.relative_to(bundle_dir)),
                "schema_version": document.get("schema_version", ""),
                "status": status,
                "synthetic": synthetic,
                "claim_eligible": claim_eligible,
                "output_count": len(outputs),
                "hashes_passed": hashes_passed,
            }
        )

    manifest_name = "submission_bundle_audit_manifest.json"
    ensure_new_output(output_dir, manifest_name)
    write_csv(output_dir / "manifest_audit.csv", rows, FIELDS)
    report = ["# Submission bundle integrity audit", ""]
    if any_synthetic:
        report.extend(["> **SYNTHETIC TEST BUNDLE — INVALID FOR SCIENTIFIC CLAIMS.**", ""])
    report.extend(
        [
            f"- MDP manifests checked: {len(rows)}",
            f"- Integrity errors: {len(errors)}",
            f"- Claim eligible: {'yes' if all_claim_eligible and not any_synthetic and not errors else 'no'}",
            "",
        ]
    )
    if errors:
        report.append("## Errors")
        report.append("")
        report.extend(f"- {error}" for error in errors)
    atomic_write_text(output_dir / "BUNDLE_AUDIT.md", "\n".join(report) + "\n")
    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed" if not errors else "failed",
        "synthetic": any_synthetic,
        "claim_eligible": all_claim_eligible and not any_synthetic and not errors,
        "manifest_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
    }
    commit_manifest(
        output_dir,
        manifest_name,
        result,
        ["manifest_audit.csv", "BUNDLE_AUDIT.md"],
    )
    if errors:
        raise ContractError(f"submission bundle audit failed with {len(errors)} error(s)")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.bundle_dir.resolve(), args.output_dir.resolve())
    print(
        f"bundle audit passed: manifests={result['manifest_count']} "
        f"claim_eligible={result['claim_eligible']}"
    )


if __name__ == "__main__":
    main()
