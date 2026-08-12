#!/usr/bin/env python3
"""Read-only verification of an MDP-05 result directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(run_dir: Path) -> dict[str, Any]:
    identity = read_json(run_dir / "run_identity.json")
    status = read_json(run_dir / "status.json")
    analysis = read_json(run_dir / "analysis" / "analysis_manifest.json")
    handoff = read_json(run_dir / "handoff_manifest.json")
    artifact_checks = []
    for section in ("selected_artifacts", "analysis_artifacts"):
        for row in handoff[section]:
            path = run_dir / row["path"]
            artifact_checks.append(
                path.is_file()
                and path.stat().st_size == int(row["bytes"])
                and sha256_file(path) == row["sha256"]
            )
    selection_checks = []
    for path in sorted(run_dir.glob("formal/*/replica_*/unit_selection.json")):
        selection = read_json(path)
        attempt = path.parent / selection["selected_attempt"]
        manifest = attempt / "mdp05_unit_manifest.json"
        manifest_payload = read_json(manifest)
        seal = read_json(attempt / "worker_log_seal.json")
        selection_checks.append(
            selection.get("passed") is True
            and selection.get("manifest_sha256") == sha256_file(manifest)
            and manifest_payload.get("passed") is True
            and all(
                (attempt / name).is_file()
                and sha256_file(attempt / name) == expected
                for name, expected in manifest_payload.get(
                    "scientific_artifact_sha256", {}
                ).items()
            )
            and seal.get("sealed_after_worker_exit") is True
            and seal.get("sha256") == sha256_file(attempt / "worker.log")
        )
    checks = {
        "identity": identity.get("experiment") == "MDP-05"
        and identity.get("dry_run") is False,
        "status": status.get("status") == "completed",
        "analysis_integrity": analysis.get("integrity_passed") is True,
        "handoff": handoff.get("passed") is True,
        "artifact_hashes": bool(artifact_checks) and all(artifact_checks),
        "selected_units_12": len(selection_checks) == 12
        and all(selection_checks),
    }
    return {
        "run_dir": str(run_dir),
        "checks": checks,
        "integrity_passed": all(checks.values()),
        "scientific_result": analysis.get("scientific_result"),
        "claim_success": analysis.get("claim_success"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = inspect(args.run_dir.resolve())
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if result["integrity_passed"] else 2
    except Exception as exc:
        print(
            f"MDP-05 verification failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
