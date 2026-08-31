#!/usr/bin/env python3
"""Create the EX54/EX52 accepted 50-shard FineWeb view without copying data.

The EX54 124M track is intentionally bound to validation shard 000000 followed
by training shards 000001..000050 from the same accepted EX48/FineWeb stream.
This helper creates an atomic directory of absolute symbolic links. The EX54
controller still performs the authoritative full-byte SHA-256 audit before any
training starts; this helper does not relax or replace that gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--view-dir", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["name"]).encode("utf-8"))
        digest.update(str(row["bytes"]).encode("ascii"))
        digest.update(str(row["tokens"]).encode("ascii"))
        digest.update(str(row["sha256"]).encode("ascii"))
    return digest.hexdigest()


def expected_rows(
    projection: dict[str, Any],
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    inventory = projection.get("inventory")
    if not isinstance(inventory, dict):
        raise RuntimeError("accepted EX48 projection has no inventory")
    train = inventory.get("train")
    validation = inventory.get("validation")
    if not isinstance(train, list) or not isinstance(validation, list):
        raise RuntimeError("accepted EX48 projection inventory is malformed")
    if len(train) != 103 or len(validation) != 1:
        raise RuntimeError(
            "accepted EX48 projection count mismatch: "
            f"train={len(train)} validation={len(validation)}"
        )

    selected_train = train[:50]
    expected_train_names = [
        f"fineweb_train_{index:06d}.bin" for index in range(1, 51)
    ]
    observed_train_names = [str(row.get("name")) for row in selected_train]
    if observed_train_names != expected_train_names:
        raise RuntimeError(
            "accepted EX48 projection does not begin with "
            "train shards 000001..000050"
        )
    if str(validation[0].get("name")) != "fineweb_val_000000.bin":
        raise RuntimeError(
            "accepted EX48 projection validation shard is not 000000"
        )

    # EX52/EX54 intentionally fingerprints validation first, then train 1..50.
    rows = [validation[0], *selected_train]
    observed_fingerprint = content_fingerprint(rows)
    expected_fingerprint = str(
        contract["data"]["124m"][
            "accepted_full_content_fingerprint_sha256"
        ]
    )
    if observed_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "the accepted EX48 first-50 projection does not match the "
            "frozen EX52/EX54 fingerprint: "
            f"observed={observed_fingerprint} "
            f"expected={expected_fingerprint}"
        )
    return rows


def source_matches(
    source_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not source_dir.is_dir():
        raise RuntimeError(
            f"source FineWeb directory is absent: {source_dir}"
        )
    for row in rows:
        path = source_dir / str(row["name"])
        if not path.is_file():
            raise RuntimeError(f"required FineWeb shard is absent: {path}")
        expected_bytes = int(row["bytes"])
        observed_bytes = path.stat().st_size
        if observed_bytes != expected_bytes:
            raise RuntimeError(
                f"FineWeb shard byte size mismatch: {path} "
                f"observed={observed_bytes} expected={expected_bytes}"
            )


def view_is_exact(
    view_dir: Path,
    source_dir: Path,
    rows: list[dict[str, Any]],
) -> bool:
    if not view_dir.is_dir():
        return False
    expected_names = {str(row["name"]) for row in rows}
    observed_names = {path.name for path in view_dir.glob("*.bin")}
    if observed_names != expected_names:
        return False
    for name in sorted(expected_names):
        view_path = view_dir / name
        source_path = source_dir / name
        try:
            if (
                not view_path.is_file()
                or not os.path.samefile(view_path, source_path)
            ):
                return False
        except OSError:
            return False
    return True


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def write_view(
    source_dir: Path,
    view_dir: Path,
    rows: list[dict[str, Any]],
    projection_path: Path,
    contract_path: Path,
) -> None:
    view_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{view_dir.name}.tmp.",
            dir=str(view_dir.parent),
        )
    )
    backup: Path | None = None
    try:
        for row in rows:
            name = str(row["name"])
            os.symlink(
                str((source_dir / name).absolute()),
                str(temp_dir / name),
            )

        manifest = {
            "schema_version": "ex54_frozen50_symlink_view_v1",
            "source_dir": str(source_dir.absolute()),
            "view_dir": str(view_dir.absolute()),
            "projection": str(projection_path.absolute()),
            "projection_sha256": sha256_file(projection_path),
            "contract": str(contract_path.absolute()),
            "contract_sha256": sha256_file(contract_path),
            "train_shards": 50,
            "validation_shards": 1,
            "names": [str(row["name"]) for row in rows],
            "note": (
                "Path-only symlink view. EX54 preflight remains responsible "
                "for authoritative header and full-content SHA-256 validation."
            ),
        }
        (temp_dir / ".ex54_frozen50_view.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if view_dir.exists() or view_dir.is_symlink():
            backup = view_dir.with_name(
                f".{view_dir.name}.stale.{os.getpid()}"
            )
            remove_path(backup)
            os.replace(view_dir, backup)

        os.replace(temp_dir, view_dir)
        temp_dir = None

        if backup is not None:
            remove_path(backup)
            backup = None
    finally:
        if temp_dir is not None:
            remove_path(temp_dir)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().absolute()
    view_dir = args.view_dir.expanduser().absolute()
    projection_path = args.projection.expanduser().absolute()
    contract_path = args.contract.expanduser().absolute()

    if source_dir == view_dir:
        raise RuntimeError("source-dir and view-dir must be different")
    if not projection_path.is_file():
        raise RuntimeError(
            f"accepted projection is absent: {projection_path}"
        )
    if not contract_path.is_file():
        raise RuntimeError(f"EX54 contract is absent: {contract_path}")

    projection = read_json(projection_path)
    contract = read_json(contract_path)
    rows = expected_rows(projection, contract)
    source_matches(source_dir, rows)

    if view_is_exact(view_dir, source_dir, rows):
        print(f"EX54_FROZEN50_VIEW_READY={view_dir}")
        print("EX54_FROZEN50_VIEW_ACTION=reused")
        return

    write_view(
        source_dir,
        view_dir,
        rows,
        projection_path,
        contract_path,
    )
    if not view_is_exact(view_dir, source_dir, rows):
        raise RuntimeError(
            f"new frozen50 view failed its path-level audit: {view_dir}"
        )
    print(f"EX54_FROZEN50_VIEW_READY={view_dir}")
    print("EX54_FROZEN50_VIEW_ACTION=created")


if __name__ == "__main__":
    main()
