#!/usr/bin/env python3
"""Create/verify the exact EX43/44 50+1 data view underneath official-r0."""

from __future__ import annotations

import argparse
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_NAMES = [
    "fineweb_val_000000.bin",
    *[f"fineweb_train_{index:06d}.bin" for index in range(1, 51)],
]
EXPECTED_BYTES = 200_001_024


def validate_header(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        header = handle.read(1024)
    if len(header) != 1024:
        raise RuntimeError(f"truncated FineWeb header: {path}")
    magic, version, tokens = struct.unpack_from("<iii", header, 0)
    expected_bytes = 1024 + 2 * tokens
    checks = {
        "magic": magic == 20240520,
        "version": version == 1,
        "tokens": tokens == 100_000_000,
        "bytes": path.stat().st_size == expected_bytes == EXPECTED_BYTES,
    }
    if not all(checks.values()):
        raise RuntimeError(f"FineWeb shard differs from the EX43/44 format: {path}: {checks}")
    return {"magic": magic, "version": version, "tokens": tokens, "bytes": expected_bytes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    args = parser.parse_args()
    official = args.official_repo.expanduser().resolve()
    view_root = args.view_root.expanduser().absolute()
    if not official.is_dir() or not (official / "train_gpt_newton_muon_2.py").is_file():
        raise RuntimeError(f"official-r0 is incomplete: {official}")
    resolved_view_root = view_root.resolve() if view_root.exists() else view_root
    if resolved_view_root == official or not resolved_view_root.is_relative_to(official):
        raise RuntimeError("EX51 data view must be a dedicated directory inside official-r0")

    source = official / "data" / "fineweb10B"
    if not source.is_dir():
        raise RuntimeError(f"official-r0 FineWeb data entry is unavailable: {source}")
    shard_dir = view_root / "data" / "fineweb10B"
    shard_dir.mkdir(parents=True, exist_ok=True)
    observed = {path.name for path in shard_dir.glob("fineweb_*.bin")}
    unexpected = sorted(observed - set(EXPECTED_NAMES))
    if unexpected:
        raise RuntimeError(f"EX51 view contains unexpected data files: {unexpected}")

    rows = []
    for name in EXPECTED_NAMES:
        source_path = source / name
        if not source_path.is_file():
            raise RuntimeError(f"official-r0 is missing required EX51 shard: {source_path}")
        header = validate_header(source_path)
        target = shard_dir / name
        if os.path.lexists(target):
            if not target.is_symlink() or target.resolve() != source_path.resolve():
                raise RuntimeError(f"existing EX51 view entry is not the expected symlink: {target}")
        else:
            # Keep the link text rooted at official-r0.  Its source data entry
            # may itself be the R0-approved shared-data symlink.
            target.symlink_to(source_path.absolute())
        rows.append(
            {
                "name": name,
                "source_entry": str(source_path.absolute()),
                "resolved_source": str(source_path.resolve()),
                "view_entry": str(target.absolute()),
                **header,
            }
        )

    payload = {
        "schema_version": 1,
        "experiment_id": "51_moddedgpt_global_diag_scale",
        "passed": True,
        "created_or_verified_at": datetime.now(timezone.utc).isoformat(),
        "official_repo": str(official),
        "view_root": str(view_root),
        "train_shards": 50,
        "validation_shards": 1,
        "total_files": 51,
        "files": rows,
        "note": "Full-content SHA-256 and the accepted EX43/44 fingerprint are enforced by preflight.",
    }
    manifest = view_root / "ex51_data_view_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "view_root": str(view_root), "files": 51, "manifest": str(manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()
