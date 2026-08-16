#!/usr/bin/env python3
"""Create/verify the EX52 50-shard data view inside Newton-Muon-official-r0.

The command never deletes, truncates, copies, or rewrites a FineWeb shard.  It
only creates verified symbolic links for train shards 1--50 and the sole val
shard.  Re-running it verifies the existing view fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_SHARD_BYTES = 200_001_024
EXPECTED_TOTAL_BYTES = 10_200_052_224
EXPECTED_NAME_SIZE_SHA256 = "94d6cbfdd33d62346398e9cd0c1643933cd99ce67b9ac273ae9192a566dfe929"
EXPECTED_NAMES = [
    *[f"fineweb_train_{index:06d}.bin" for index in range(1, 51)],
    "fineweb_val_000000.bin",
]


def parse_index(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"unexpected FineWeb shard name: {path.name}") from exc


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
        "bytes": path.stat().st_size == expected_bytes == EXPECTED_SHARD_BYTES,
    }
    if not all(checks.values()):
        raise RuntimeError(f"FineWeb shard differs from the accepted format: {path}: {checks}")
    return {"magic": magic, "version": version, "tokens": tokens, "bytes": expected_bytes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--view-dir", type=Path, required=True)
    args = parser.parse_args()
    official = args.official_repo.expanduser().resolve()
    source_entry = args.source_dir.expanduser().absolute()
    source = source_entry.resolve()
    view = args.view_dir.expanduser().absolute()
    resolved_view = view.resolve() if view.exists() else view
    if not official.is_dir() or not (official / "triton_kernels.py").is_file():
        raise RuntimeError(f"Newton-Muon-official-r0 is incomplete: {official}")
    if source_entry != official / "data" / "fineweb10B":
        raise RuntimeError("EX52 source must be the FineWeb entry inside Newton-Muon-official-r0")
    if resolved_view == official or not resolved_view.is_relative_to(official):
        raise RuntimeError("EX52 data view must be a dedicated directory inside official-r0")
    if source == resolved_view:
        raise RuntimeError("source and view directories must differ")
    if not source.is_dir():
        raise RuntimeError(f"source data directory is missing: {source}")

    train_all = sorted(source.glob("fineweb_train_*.bin"), key=parse_index)
    val = sorted(source.glob("fineweb_val_*.bin"))
    indices = [parse_index(path) for path in train_all]
    missing = [index for index in range(1, 51) if index not in indices]
    if missing or len(val) != 1:
        raise RuntimeError(
            f"r0 data cannot supply EX52: missing train indices={missing}, val_count={len(val)}"
        )
    selected = [train_all[indices.index(index)] for index in range(1, 51)] + val
    if [path.name for path in selected] != EXPECTED_NAMES:
        raise RuntimeError("r0 shards differ from the frozen EX17/20 filename order")

    view.mkdir(parents=True, exist_ok=True)
    expected_names = {path.name for path in selected}
    observed_bin_names = {path.name for path in view.glob("*.bin")}
    unexpected = sorted(observed_bin_names - expected_names)
    if unexpected:
        raise RuntimeError(f"EX52 data view contains unexpected bin files: {unexpected}")

    rows = []
    for source_path in selected:
        header = validate_header(source_path)
        target = view / source_path.name
        if os.path.lexists(target):
            if not target.is_symlink() or target.resolve() != source_path.resolve():
                raise RuntimeError(f"existing view entry is not the expected symlink: {target}")
        else:
            target.symlink_to(source_path.resolve())
        rows.append(
            {
                "name": source_path.name,
                "source": str(source_path.resolve()),
                "view": str(target.absolute()),
                **header,
            }
        )

    names_and_sizes = "\n".join(f"{row['name']}\t{row['bytes']}" for row in rows).encode("utf-8")
    inventory_sha256 = hashlib.sha256(names_and_sizes).hexdigest()
    total_bytes = sum(int(row["bytes"]) for row in rows)
    if inventory_sha256 != EXPECTED_NAME_SIZE_SHA256 or total_bytes != EXPECTED_TOTAL_BYTES:
        raise RuntimeError("EX52 view does not match the accepted EX17/20 name/size inventory")
    payload = {
        "schema_version": 1,
        "experiment_id": "52_llama_global_diag_scale",
        "passed": True,
        "created_or_verified_at": datetime.now(timezone.utc).isoformat(),
        "source_repository": "Newton-Muon-official-r0",
        "official_repo": str(official),
        "source_dir": str(source),
        "view_dir": str(view),
        "train_count": 50,
        "val_count": 1,
        "first_train_index": 1,
        "last_train_index": 50,
        "total_bytes": total_bytes,
        "inventory_name_size_sha256": inventory_sha256,
        "note": "Full-content SHA-256 and the accepted EX17/20 fingerprint are enforced by suite preflight.",
        "files": rows,
    }
    manifest = view.parent / f"{view.name}_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "view_dir": str(view), "manifest": str(manifest), "train_count": 50, "val_count": 1}, sort_keys=True))


if __name__ == "__main__":
    main()
