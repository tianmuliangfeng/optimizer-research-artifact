"""Build deterministic repository-level manifests after all content is frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


GENERATED = {"SHA256SUMS", "SUBMISSION_MANIFEST.json"}
FORBIDDEN_PARTS = {".git", ".wandb", "__pycache__", "full-archive"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_files(root: Path, *, include_manifest: bool) -> list[Path]:
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        if current_path == root:
            directories[:] = [name for name in directories if name != ".git"]
        for name in list(directories):
            path = current_path / name
            if path.is_symlink():
                raise RuntimeError(f"symlinked directory is forbidden: {path}")
        for name in names:
            path = current_path / name
            relative = path.relative_to(root)
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"non-regular file is forbidden: {path}")
            if any(part in FORBIDDEN_PARTS for part in relative.parts):
                raise RuntimeError(f"forbidden repository entry: {relative.as_posix()}")
            if relative.as_posix() == "SHA256SUMS":
                continue
            if not include_manifest and relative.as_posix() in GENERATED:
                continue
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def build(root: Path) -> None:
    root = root.resolve()
    payload_files = regular_files(root, include_manifest=False)
    records = [
        {
            "bytes": path.stat().st_size,
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
        }
        for path in payload_files
    ]
    manifest = {
        "anonymity_profile": "double_blind_v1",
        "artifact_name": "selective-newton-muon-anonymous-review",
        "excluded_components": [
            "full-archive",
            "raw training data",
            "model checkpoints",
            "raw W&B exports and caches",
            "private Git history",
        ],
        "included_components": ["source code", "core-results"],
        "license": "MIT",
        "payload_bytes": sum(record["bytes"] for record in records),
        "payload_file_count": len(records),
        "release_date": "2026-08-31",
        "schema_version": 1,
        "files": records,
    }
    manifest_path = root / "SUBMISSION_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    checksum_files = regular_files(root, include_manifest=True)
    lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in checksum_files
    ]
    (root / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    build(Path(args.root))
    print("submission manifest and SHA256SUMS written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
