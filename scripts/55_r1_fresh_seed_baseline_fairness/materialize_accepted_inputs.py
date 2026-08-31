#!/usr/bin/env python3
"""Restore byte-exact accepted EX55 inputs from portable base64 payloads."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path


PAYLOADS = {
    "historical_panel.csv": {
        "encoded": "historical_panel.csv.b64",
        "bytes": 7562,
        "sha256": "6da83e28c832c145676128e761de4309f49b04fe48f845a411c23043c2a8c42a",
    },
    "extended_selection.csv": {
        "encoded": "extended_selection.csv.b64",
        "bytes": 1786,
        "sha256": "2b38133a27f1625c26174cea98a78568026d9a4e9b7eca12accb53e79228c80d",
    },
    "mousse_selection.json": {
        "encoded": "mousse_selection.json.b64",
        "bytes": 921,
        "sha256": "46a09b8053d20c0e8259c5d946be8d9ed60bc1c950fba677a74b8e563e4a2e3c",
    },
    "malt_selection.json": {
        "encoded": "malt_selection.json.b64",
        "bytes": 7237,
        "sha256": "591348aad2d19f36ca405abe2162d329dc89246bc41fc1fec3cb0581a28fe4ee",
    },
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def materialize(encoded_root: Path, output_root: Path) -> dict[str, object]:
    encoded_root = encoded_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for output_name, spec in PAYLOADS.items():
        encoded_path = encoded_root / str(spec["encoded"])
        if not encoded_path.is_file():
            raise RuntimeError(f"missing encoded accepted input: {encoded_path}")
        try:
            payload = base64.b64decode(
                "".join(encoded_path.read_text(encoding="ascii").split()),
                validate=True,
            )
        except Exception as error:
            raise RuntimeError(f"invalid encoded accepted input: {encoded_path}") from error
        observed = sha256_bytes(payload)
        if len(payload) != int(spec["bytes"]) or observed != spec["sha256"]:
            raise RuntimeError(
                f"decoded accepted input failed byte/hash gate: {output_name}: "
                f"bytes={len(payload)} sha256={observed}"
            )
        target = output_root / output_name
        if target.is_file():
            existing = target.read_bytes()
            if len(existing) != int(spec["bytes"]) or sha256_bytes(existing) != spec["sha256"]:
                raise RuntimeError(f"existing materialized accepted input drift: {target}")
        else:
            temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            temporary.write_bytes(payload)
            os.replace(temporary, target)
        records.append(
            {
                "path": str(target),
                "bytes": int(spec["bytes"]),
                "sha256": str(spec["sha256"]),
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": "55_r1_fresh_seed_baseline_fairness",
        "passed": True,
        "files": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoded-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.encoded_root, args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
