#!/usr/bin/env python3
"""Controller for the eight-checkpoint MECH-07 family contrast."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-27.3"
COMPATIBLE_WORKER_VERSIONS = {"2026-07-27.2", SCRIPT_VERSION}
HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--contract", default=HERE / "family_contrast_contract.json", type=Path)
    parser.add_argument("--source-script", required=True, type=Path)
    parser.add_argument("--profile-script", required=True, type=Path)
    parser.add_argument("--triton-kernels", required=True, type=Path)
    parser.add_argument("--data-pattern", required=True)
    parser.add_argument("--child-python", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--host-id", default="llama-host-h100")
    parser.add_argument("--execution-domain", default="llama-host-llama1b")
    parser.add_argument("--stamp")
    parser.add_argument(
        "--resume-stamp",
        help="Resume an existing timestamp without re-hashing certified checkpoints.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))


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


def hash_checkpoint(spec: dict[str, Any], output: Path) -> dict[str, Any]:
    path = Path(spec["path"])
    before = path.stat()
    started = time.time()
    observed = sha256_file(path)
    after = path.stat()
    expected = spec["expected_sha256"]
    checks = {
        "stable_size": before.st_size == after.st_size,
        "stable_mtime": before.st_mtime_ns == after.st_mtime_ns,
        "known_hash_matches": expected is None or observed == expected,
    }
    payload = {
        "schema_version": 1,
        "cell": spec["cell"],
        "stage": spec["stage"],
        "method": spec["method"],
        "step": spec["step"],
        "path": str(path),
        "bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "sha256": observed,
        "expected_sha256": expected,
        "hash_mode": "known_expected" if expected else "frozen_path_first_observation",
        "elapsed_seconds": time.time() - started,
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_json(output, payload)
    if not payload["passed"]:
        raise RuntimeError(f"checkpoint hash audit failed: {payload}")
    return payload


def append_values(command: list[str], name: str, values: list[Any]) -> None:
    command.append(name)
    command.extend(str(value) for value in values)


def expanded_batch_offsets(
    contract: dict[str, Any], tier: str
) -> list[int]:
    """Expand one frozen base offset per repeat into disjoint A/B windows."""
    config = contract[tier]
    bases = [int(value) for value in config["repeat_offsets"]]
    repeats = int(config["repeats"])
    if len(bases) != repeats:
        raise RuntimeError(
            f"{tier} requires one base offset per repeat: "
            f"bases={len(bases)} repeats={repeats}"
        )
    batches_per_repeat = 2 * int(config["batches_per_split"])
    window_tokens = (
        int(contract["formal"]["device_batch_size"])
        * int(contract["formal"]["sequence_length"])
        + 1
    )
    offsets = [
        base + batch_index * window_tokens
        for base in bases
        for batch_index in range(batches_per_repeat)
    ]
    intervals = [(offset, offset + window_tokens) for offset in offsets]
    for left in range(len(intervals)):
        for right in range(left + 1, len(intervals)):
            if max(intervals[left][0], intervals[right][0]) < min(
                intervals[left][1], intervals[right][1]
            ):
                raise RuntimeError(
                    f"{tier} expanded batch windows overlap: "
                    f"{intervals[left]} and {intervals[right]}"
                )
    return offsets


def reusable_hash_certificate(
    spec: dict[str, Any], certificate: Path
) -> dict[str, Any] | None:
    if not certificate.is_file():
        return None
    payload = read_json(certificate)
    path = Path(spec["path"])
    stat = path.stat()
    expected = spec["expected_sha256"]
    checks = {
        "passed": payload.get("passed") is True,
        "cell": payload.get("cell") == spec["cell"],
        "path": payload.get("path") == spec["path"],
        "size": int(payload.get("bytes", -1)) == stat.st_size,
        "mtime": int(payload.get("mtime_ns", -1)) == stat.st_mtime_ns,
        "known_hash": expected is None or payload.get("sha256") == expected,
    }
    return payload if all(checks.values()) else None


def completed_manifest(
    path: Path,
    *,
    tier: str,
    cell: str,
    contract_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    manifest = read_json(path)
    return (
        manifest.get("passed") is True
        and manifest.get("analysis_tier") == tier
        and manifest.get("cell") == cell
        and manifest.get("contract_sha256") == contract_sha256
        and manifest.get("script_version") in COMPATIBLE_WORKER_VERSIONS
    )


def worker_command(
    args: argparse.Namespace,
    contract: dict[str, Any],
    spec: dict[str, Any],
    cert: Path,
    output: Path,
    tier: str,
    smoke_manifest: Path | None,
) -> list[str]:
    config = contract[tier]
    layers = (
        config["layers"] if tier == "smoke" else contract["targets"]["layers"]
    )
    kinds = (
        config["kinds"] if tier == "smoke" else contract["targets"]["kinds"]
    )
    command = [
        args.child_python,
        str(HERE / "mech07_worker.py"),
        "--output-dir",
        str(output),
        "--analysis-tier",
        tier,
        "--cell",
        spec["cell"],
        "--checkpoint",
        spec["path"],
        "--checkpoint-hash-certificate",
        str(cert),
        "--source-script",
        str(args.source_script),
        "--profile-script",
        str(args.profile_script),
        "--triton-kernels",
        str(args.triton_kernels),
        "--contract",
        str(args.contract),
        "--data-pattern",
        args.data_pattern,
        "--repeats",
        str(config["repeats"]),
        "--batches-per-split",
        str(config["batches_per_split"]),
        "--device-batch-size",
        str(contract["formal"]["device_batch_size"]),
        "--sequence-length",
        str(contract["formal"]["sequence_length"]),
        "--max-activation-rows",
        str(config["max_activation_rows"]),
        "--ridge-mult",
        str(contract["formal"]["ridge_mult"]),
        "--ridge-eps",
        str(contract["formal"]["ridge_eps"]),
        "--momentum",
        str(contract["formal"]["momentum"]),
        "--ns-steps",
        str(contract["formal"]["ns_steps"]),
        "--host-id",
        args.host_id,
        "--execution-domain",
        args.execution_domain,
    ]
    append_values(command, "--layers", layers)
    append_values(command, "--target-kinds", kinds)
    append_values(command, "--repeat-offsets", expanded_batch_offsets(contract, tier))
    append_values(
        command, "--step-multipliers", contract["formal"]["step_multipliers"]
    )
    if smoke_manifest is not None:
        command.extend(["--smoke-manifest", str(smoke_manifest)])
    return command


def main() -> None:
    args = parse_args()
    if args.stamp and args.resume_stamp:
        raise RuntimeError("--stamp and --resume-stamp are mutually exclusive")
    contract = read_json(args.contract.resolve())
    stamp = (
        args.resume_stamp
        or args.stamp
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S+0000")
    )
    run = args.output_root.resolve() / stamp
    if args.resume_stamp:
        if not run.is_dir():
            raise RuntimeError(f"resume directory does not exist: {run}")
    else:
        run.mkdir(parents=True, exist_ok=False)
    hashes = run / "checkpoint_hashes"
    hashes.mkdir(exist_ok=args.resume_stamp is not None)
    write_json(
        run / "status.json",
        {"status": "running", "script_version": SCRIPT_VERSION},
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    try:
        preflight_checks = {
            "source_exists": args.source_script.is_file(),
            "profile_exists": args.profile_script.is_file(),
            "triton_exists": args.triton_kernels.is_file(),
            "all_checkpoints_exist": all(
                Path(spec["path"]).is_file() for spec in contract["checkpoints"]
            ),
            "source_sha256": (
                args.source_script.is_file()
                and sha256_file(args.source_script)
                == contract["source_constraints"]["base_source_sha256"]
            ),
            "triton_sha256": (
                args.triton_kernels.is_file()
                and sha256_file(args.triton_kernels)
                == contract["source_constraints"]["triton_sha256"]
            ),
        }
        write_json(
            run / "preflight.json",
            {"checks": preflight_checks, "passed": all(preflight_checks.values())},
        )
        if not all(preflight_checks.values()):
            raise RuntimeError(f"MECH-07 preflight failed: {preflight_checks}")

        inventory = []
        certificates: dict[str, Path] = {}
        for spec in contract["checkpoints"]:
            cert = hashes / f"{spec['cell']}.json"
            reused = (
                reusable_hash_certificate(spec, cert)
                if args.resume_stamp
                else None
            )
            inventory.append(reused or hash_checkpoint(spec, cert))
            certificates[spec["cell"]] = cert
        write_json(
            run / "checkpoint_inventory.json",
            {
                "schema_version": 1,
                "contract_sha256": sha256_file(args.contract.resolve()),
                "cells": inventory,
                "passed": all(row["passed"] for row in inventory),
            },
        )

        # No CUDA diagnostic begins until all eight frozen paths have passed
        # existence, source, and full-hash preflight.
        contract_sha256 = sha256_file(args.contract.resolve())
        for spec in contract["checkpoints"]:
            cert = certificates[spec["cell"]]
            cell_dir = run / spec["cell"]
            smoke = cell_dir / "smoke"
            formal = cell_dir / "formal"
            if completed_manifest(
                formal / "mech07_manifest.json",
                tier="formal",
                cell=spec["cell"],
                contract_sha256=contract_sha256,
            ):
                print(f"MECH-07 resume: formal already passed for {spec['cell']}")
                continue
            smoke.mkdir(parents=True, exist_ok=True)
            if not completed_manifest(
                smoke / "mech07_manifest.json",
                tier="smoke",
                cell=spec["cell"],
                contract_sha256=contract_sha256,
            ):
                smoke_command = worker_command(
                    args, contract, spec, cert, smoke, "smoke", None
                )
                print("MECH-07 smoke command:", json.dumps(smoke_command))
                subprocess.run(smoke_command, check=True, env=env)
            else:
                print(f"MECH-07 resume: smoke already passed for {spec['cell']}")
            formal.mkdir(exist_ok=True)
            formal_command = worker_command(
                args,
                contract,
                spec,
                cert,
                formal,
                "formal",
                smoke / "mech07_manifest.json",
            )
            print("MECH-07 formal command:", json.dumps(formal_command))
            subprocess.run(formal_command, check=True, env=env)
        analysis_command = [
            sys.executable,
            str(HERE / "analyze_mech07.py"),
            "--run-dir",
            str(run),
            "--contract",
            str(args.contract.resolve()),
        ]
        subprocess.run(analysis_command, check=True)
        write_json(
            run / "status.json",
            {"status": "passed", "script_version": SCRIPT_VERSION},
        )
    except BaseException as exc:
        write_json(
            run / "status.json",
            {
                "status": "failed",
                "script_version": SCRIPT_VERSION,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(f"MECH-07 artifacts: {run}")
    print(f"MECH-07 manifest: {run / 'analysis' / 'mech07_analysis_manifest.json'}")


if __name__ == "__main__":
    main()
