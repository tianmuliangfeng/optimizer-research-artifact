#!/usr/bin/env python3
"""Freeze the path/mtime-independent EX48 FineWeb identity projection.

This maintainer utility is not run on the training host.  The committed JSON
it produces is independently SHA-bound by the EX54 contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = (
    "split", "ordinal", "name", "index", "tokens", "consumable_tokens",
    "bytes", "header_sha256", "sha256",
)


def project(rows: list[dict[str, object]], split: str) -> list[dict[str, object]]:
    return [
        {
            "split": split,
            "ordinal": ordinal,
            "name": str(row["name"]),
            "index": int(row["index"]),
            "tokens": int(row["tokens"]),
            "consumable_tokens": int(row["consumable_tokens"]),
            "bytes": int(row["bytes"]),
            "header_sha256": str(row["header_sha256"]),
            "sha256": str(row["sha256"]),
        }
        for ordinal, row in enumerate(rows)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--source-audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_audit.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "ex54_accepted_ex48_data_projection_v1",
        "source_experiment": 48,
        "source_run_id": "20260805T061608+0000",
        "source_data_audit_sha256": args.source_audit_sha256,
        "source_inventory_sha256": source["inventory_sha256"],
        "train_shard_count": int(source["train_shard_count"]),
        "validation_shard_count": int(source["validation_shard_count"]),
        "fields": list(FIELDS),
        "inventory": {
            "train": project(source["inventory"]["train"], "train"),
            "validation": project(source["inventory"]["validation"], "validation"),
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
