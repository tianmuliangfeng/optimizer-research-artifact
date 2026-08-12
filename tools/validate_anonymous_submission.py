"""Fail-closed validation for the double-blind repository snapshot."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


TEXT_SUFFIXES = {
    ".cfg", ".csv", ".ini", ".json", ".jsonl", ".md", ".py", ".sh",
    ".svg", ".tex", ".toml", ".tsv", ".txt", ".yaml", ".yml",
}
FORBIDDEN_NAMES = {".git", ".wandb", "__pycache__", "full-archive", "wandb"}
WINDOWS_PATH = re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
PRIVATE_POSIX = re.compile(
    r"/(?:data|home|Users|mnt|workspace|root)/"
    r"(?:[A-Z][0-9]{6,}|[A-Za-z][A-Za-z0-9_-]*[0-9]{8,})(?:/|\\)",
    re.I,
)
PUBLIC_WANDB = re.compile(r"https?://(?:api\.)?wandb\.ai/", re.I)
CONTAINER_HOST = re.compile(r"\bapp-[0-9a-f][a-z0-9-]{18,}\b", re.I)
IPV4 = re.compile(
    r"(?<![0-9.])(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})"
    r"(?:\.(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})){3}(?![0-9.])"
)
EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9.-])"
)


def validate(root: Path) -> None:
    root = root.resolve()
    errors: list[str] = []
    required = {
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "core-results/release_manifest.json",
        "core-results/tools/validate_core_results_package.py",
    }
    for relative in sorted(required):
        if not (root / relative).is_file():
            errors.append(f"missing required file: {relative}")

    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        parts = path.relative_to(root).parts
        # A GitHub Actions checkout necessarily has an untracked root .git
        # directory. It is transport metadata, not submission content. Ignore
        # that tree while continuing to reject forbidden names everywhere in
        # the tracked/work-tree payload.
        if parts and parts[0] == ".git":
            continue
        if any(part in FORBIDDEN_NAMES for part in parts):
            errors.append(f"forbidden repository entry: {relative}")
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig")
        # The negative-test module holds synthetic leak samples by construction.
        negative_fixture = relative == "tests/test_core_results_packaging.py"
        checks = {
            "Windows absolute path": WINDOWS_PATH,
            "private POSIX path": PRIVATE_POSIX,
            "public W&B account URL": PUBLIC_WANDB,
            "private container hostname": CONTAINER_HOST,
            "IPv4 address": IPV4,
            "email address": EMAIL,
        }
        for label, pattern in checks.items():
            if negative_fixture and label in {
                "public W&B account URL", "private container hostname"
            }:
                continue
            for match in pattern.finditer(text):
                if label == "email address" and match.group(0).lower().endswith(".invalid"):
                    continue
                errors.append(f"{relative}: {label}")

    if errors:
        raise RuntimeError("anonymous submission validation failed:\n" + "\n".join(errors[:50]))

    validator = root / "core-results" / "tools" / "validate_core_results_package.py"
    completed = subprocess.run(
        [sys.executable, "-B", str(validator), str(root / "core-results")],
        cwd=root,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"core-results validator failed with rc={completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    validate(Path(args.root))
    print("anonymous submission validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
