"""Create a deterministic ZIP of the already validated anonymous repository."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


FIXED_TIME = (2026, 8, 31, 0, 0, 0)
ARCHIVE_ROOT = "selective-newton-muon-anonymous-review"


def build(root: Path, output: Path) -> None:
    root = root.resolve()
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite archive: {output}")
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and ".git" not in path.relative_to(root).parts
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            if any(part in {".git", ".wandb", "__pycache__", "full-archive"} for part in path.relative_to(root).parts):
                raise RuntimeError(f"forbidden ZIP entry: {relative}")
            info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{relative}", FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("output")
    args = parser.parse_args()
    build(Path(args.root), Path(args.output))
    print(f"submission ZIP written: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
