#!/usr/bin/env python3
"""Controller for checkpoint hashing, smoke/formal gates, and MECH-06 analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-27.4"
CONTRACT_VERSION = "2026-07-27.1"
HERE = Path(__file__).resolve().parent
WORKER = HERE / "mech06_worker.py"
ANALYZER = HERE / "analyze_mech06.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-exe", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--early-checkpoint", required=True, type=Path)
    parser.add_argument("--early-source", required=True, type=Path)
    parser.add_argument("--early-profile", required=True, type=Path)
    parser.add_argument("--late-checkpoint", required=True, type=Path)
    parser.add_argument("--late-source", required=True, type=Path)
    parser.add_argument("--late-profile", required=True, type=Path)
    parser.add_argument("--triton-kernels", required=True, type=Path)
    parser.add_argument("--mech01-reference-smoke-dir", required=True, type=Path)
    parser.add_argument("--confirmation-contract", default=HERE / "confirmation_contract.json", type=Path)
    parser.add_argument("--mech05-contract", required=True, type=Path)
    parser.add_argument("--data-pattern", required=True)
    parser.add_argument("--host-id", default="llama-host-h100")
    parser.add_argument("--execution-domain", default="llama-host-llama1b")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_checkpoint(
    label: str, path: Path, expected: str, output: Path
) -> Path:
    resolved = path.resolve()
    before = resolved.stat()
    observed = sha256_file(resolved)
    after = resolved.stat()
    passed = (
        observed == expected
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )
    payload = {
        "schema_version": 1,
        "label": label,
        "path": str(resolved),
        "bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": observed,
        "expected_sha256": expected,
        "stable_during_hash": before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns,
        "hashed_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
    }
    certificate = output / f"{label}_checkpoint_hash.json"
    write_json(certificate, payload)
    if not passed:
        raise RuntimeError(f"checkpoint hash failed: {payload}")
    return certificate


def run(command: list[str], env: dict[str, str]) -> None:
    print("MECH-06 command:", json.dumps(command))
    subprocess.run(command, check=True, env=env)


def worker_command(
    args: argparse.Namespace,
    contract: dict[str, Any],
    label: str,
    tier: str,
    checkpoint: Path,
    source: Path,
    profile: Path,
    hash_certificate: Path,
    output: Path,
    smoke_manifest: Path | None,
) -> list[str]:
    config = contract[tier]
    geometry_offsets = [
        index * 4096
        for index in range(config["repeats"] * config["geometry_batches_per_repeat"])
    ]
    shadow_offsets = [
        index * 4096
        for index in range(
            config["repeats"] * 2 * config["shadow_batches_per_split"]
        )
    ]
    command = [
        str(args.python_exe),
        str(WORKER),
        "--output-dir",
        str(output),
        "--analysis-tier",
        tier,
        "--checkpoint-label",
        label,
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-hash-certificate",
        str(hash_certificate),
        "--source-script",
        str(source),
        "--profile-script",
        str(profile),
        "--triton-kernels",
        str(args.triton_kernels),
        "--mech01-reference-smoke-dir",
        str(args.mech01_reference_smoke_dir),
        "--confirmation-contract",
        str(args.confirmation_contract),
        "--mech05-contract",
        str(args.mech05_contract),
        "--data-pattern",
        args.data_pattern,
        "--geometry-layers",
        *map(str, config["geometry_layers"]),
        "--shadow-layers",
        *map(str, config["shadow_layers"]),
        "--geometry-offsets",
        *map(str, geometry_offsets),
        "--shadow-offsets",
        *map(str, shadow_offsets),
        "--repeats",
        str(config["repeats"]),
        "--geometry-batches-per-repeat",
        str(config["geometry_batches_per_repeat"]),
        "--shadow-batches-per-split",
        str(config["shadow_batches_per_split"]),
        "--device-batch-size",
        str(contract["formal"]["device_batch_size"]),
        "--sequence-length",
        str(contract["formal"]["sequence_length"]),
        "--max-geometry-rows",
        str(config["max_geometry_rows"]),
        "--max-shadow-rows",
        str(config["max_shadow_rows"]),
        "--ridge-mult",
        str(contract["formal"]["ridge_mult"]),
        "--ridge-eps",
        str(contract["formal"]["ridge_eps"]),
        "--top-k",
        str(contract["formal"]["top_k"]),
        "--momentum",
        str(contract["formal"]["momentum"]),
        "--ns-steps",
        str(contract["formal"]["ns_steps"]),
        "--step-multipliers",
        *map(str, contract["formal"]["step_multipliers"]),
        "--host-id",
        args.host_id,
        "--execution-domain",
        args.execution_domain,
    ]
    if smoke_manifest is not None:
        command.extend(["--smoke-manifest", str(smoke_manifest)])
    return command


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing existing output root: {args.output_root}")
    args.output_root.mkdir(parents=True)
    contract = read_json(args.confirmation_contract)
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError("contract/controller version mismatch")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    checkpoints = {
        "early": (args.early_checkpoint, args.early_source, args.early_profile),
        "late": (args.late_checkpoint, args.late_source, args.late_profile),
    }
    formal_dirs = {}
    for label, (checkpoint, source, profile) in checkpoints.items():
        expected = contract["checkpoints"][label]["sha256"]
        certificate = hash_checkpoint(
            label, checkpoint, expected, args.output_root
        )
        checkpoint_root = args.output_root / label
        smoke = checkpoint_root / "smoke"
        formal = checkpoint_root / "formal"
        smoke.mkdir(parents=True)
        run(
            worker_command(
                args,
                contract,
                label,
                "smoke",
                checkpoint,
                source,
                profile,
                certificate,
                smoke,
                None,
            ),
            env,
        )
        smoke_manifest = smoke / "mech06_manifest.json"
        if read_json(smoke_manifest).get("passed") is not True:
            raise RuntimeError(f"{label} smoke did not pass")
        formal.mkdir(parents=True)
        run(
            worker_command(
                args,
                contract,
                label,
                "formal",
                checkpoint,
                source,
                profile,
                certificate,
                formal,
                smoke_manifest,
            ),
            env,
        )
        if read_json(formal / "mech06_manifest.json").get("passed") is not True:
            raise RuntimeError(f"{label} formal did not pass")
        formal_dirs[label] = formal
    analysis = args.output_root / "analysis"
    run(
        [
            sys.executable,
            str(ANALYZER),
            "--early-formal-dir",
            str(formal_dirs["early"]),
            "--late-formal-dir",
            str(formal_dirs["late"]),
            "--confirmation-contract",
            str(args.confirmation_contract),
            "--mech05-contract",
            str(args.mech05_contract),
            "--output-dir",
            str(analysis),
        ],
        env,
    )
    print(f"MECH-06 artifacts: {args.output_root}")
    print(f"MECH-06 manifest: {analysis / 'mech06_analysis_manifest.json'}")


if __name__ == "__main__":
    main()
