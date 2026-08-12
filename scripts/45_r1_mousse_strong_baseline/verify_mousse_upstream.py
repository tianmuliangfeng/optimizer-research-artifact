"""Verify a local clone against experiment 45's frozen Mousse authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path


COMMIT = "d00c1bf17790fbe56424ee5567cce80d8e75f4b2"
EXPECTED = {
    "LICENSE": "4d86e78ca5fca585db157b58577bd4123242337018d63f81747e4d1ae3d39989",
    "README.md": "4b3ee944724cafd28d2265ae76ccc4bccc15e4bfd2af98ee6d0c2763b24b0c83",
    "dion/dion/mousse.py": "29cddc3b76e8beeacb973511f71b43e4152f1f203d4eb5c66ba3012002e6d149",
    "dion/configs/mousse_160m.yaml": "93021531ca43b9a8eddb7cdda142ef6cb443407683067caaf56df48ade5f18d7",
    "dion/dion/newton_schulz_triton.py": "036bcd7ec21da3466245bfb062b426bf11fed07aaadd76bf7945c85d53be7383",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not repo.is_dir():
        raise RuntimeError(f"Mousse repository not found: {repo}")
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "rev-parse", "HEAD"],
        cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    observed_commit = result.stdout.strip() if result.returncode == 0 else ""
    failures = []
    if observed_commit != COMMIT:
        failures.append(f"commit {observed_commit!r} != {COMMIT}")
    hashes = {}
    for relative, expected in EXPECTED.items():
        path = repo / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        hashes[relative] = sha256_file(path)
        if hashes[relative] != expected:
            failures.append(f"SHA-256 mismatch for {relative}")
    license_path = repo / "LICENSE"
    license_text = license_path.read_text(encoding="utf-8", errors="replace") if license_path.is_file() else ""
    if "MIT License" not in license_text or "2026 ShikiNatsume" not in license_text:
        failures.append("MIT license identity mismatch")
    payload = {
        "status": "passed" if not failures else "failed",
        "repository": "https://github.com/Anti-Entrophic/Mousse",
        "local_repo": str(repo), "commit": observed_commit, "expected_commit": COMMIT,
        "license": "MIT", "observed_sha256": hashes, "expected_sha256": EXPECTED,
        "verified_at": datetime.now().astimezone().isoformat(), "failures": failures,
    }
    if args.output:
        args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
