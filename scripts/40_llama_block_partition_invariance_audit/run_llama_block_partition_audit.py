#!/usr/bin/env python3
"""Two-GPU controller for LLaMA block-partition smoke, formal, and analysis."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-29.2"
CONTRACT_VERSION = "2026-07-29.2"
WORKER_VERSION = "2026-07-29.2"
ANALYSIS_VERSION = "2026-07-29.2"
HERE = Path(__file__).resolve().parent
WORKER = HERE / "llama_block_partition_worker.py"
ANALYZER = HERE / "analyze_llama_block_partition_audit.py"


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
    parser.add_argument("--contract", default=HERE / "audit_contract.json", type=Path)
    parser.add_argument("--data-pattern", required=True)
    parser.add_argument("--gpus", nargs=2, default=("0", "1"))
    parser.add_argument("--host-id", default="llama-host-h100")
    parser.add_argument(
        "--execution-domain",
        default="llama-host-llama1b-block-partition-audit",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
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
    label: str, path: Path, expected: str, directory: Path
) -> Path:
    resolved = path.resolve()
    before = resolved.stat()
    observed = sha256_file(resolved)
    after = resolved.stat()
    stable = (
        before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )
    passed = stable and observed == expected
    payload = {
        "schema_version": 1,
        "label": label,
        "path": str(resolved),
        "bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": observed,
        "expected_sha256": expected,
        "stable_during_hash": stable,
        "hashed_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
    }
    certificate = directory / f"{label}_checkpoint_hash.json"
    write_json(certificate, payload)
    if not passed:
        raise RuntimeError(f"{label} checkpoint hash failed: {payload}")
    return certificate


def worker_command(
    args: argparse.Namespace,
    contract: dict[str, Any],
    tier: str,
    label: str,
    checkpoint: Path,
    source: Path,
    profile: Path,
    certificate: Path,
    output: Path,
    smoke_manifest: Path | None,
) -> list[str]:
    config = contract[tier]
    common = contract["common"]
    offsets = [
        index * 4096
        for index in range(
            config["repeats"] * 2 * config["batches_per_split"]
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
        str(certificate),
        "--source-script",
        str(source),
        "--profile-script",
        str(profile),
        "--triton-kernels",
        str(args.triton_kernels),
        "--contract",
        str(args.contract),
        "--data-pattern",
        args.data_pattern,
        "--layers",
        *map(str, config["layers"]),
        "--repeat-offsets",
        *map(str, offsets),
        "--repeats",
        str(config["repeats"]),
        "--batches-per-split",
        str(config["batches_per_split"]),
        "--device-batch-size",
        str(common["device_batch_size"]),
        "--sequence-length",
        str(common["sequence_length"]),
        "--max-activation-rows",
        str(config["max_activation_rows"]),
        "--global-permutation-seeds",
        *map(str, config["global_permutation_seeds"]),
        "--within-block-seed",
        str(common["within_block_seed"]),
        "--block-count",
        str(contract["architecture"]["block_count"]),
        "--ridge-mult",
        str(common["ridge_mult"]),
        "--ridge-eps",
        str(common["ridge_eps"]),
        "--momentum",
        str(common["momentum"]),
        "--ns-steps",
        str(common["ns_steps"]),
        "--step-multipliers",
        *map(str, config["step_multipliers"]),
        "--dense-control-permutation-count",
        str(common["dense_control_permutation_count"]),
        "--host-id",
        args.host_id,
        "--execution-domain",
        args.execution_domain,
    ]
    if smoke_manifest is not None:
        command.extend(["--smoke-manifest", str(smoke_manifest)])
    return command


def execute_job(
    label: str,
    gpu: str,
    command: list[str],
    output: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    print(
        f"LLaMA block audit start label={label} gpu={gpu} output={output}",
        flush=True,
    )
    completed = subprocess.run(command, env=env, check=False)
    return {
        "label": label,
        "gpu": gpu,
        "return_code": completed.returncode,
        "output": str(output),
    }


def run_stage(jobs: list[tuple[str, str, list[str], Path]]) -> list[dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [
            pool.submit(execute_job, label, gpu, command, output)
            for label, gpu, command, output in jobs
        ]
        results = [future.result() for future in futures]
    failures = [row for row in results if row["return_code"] != 0]
    if failures:
        raise RuntimeError(f"LLaMA block audit worker failures: {failures}")
    return results


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing existing output root: {args.output_root}")
    args.output_root.mkdir(parents=True)
    write_json(
        args.output_root / "status.json",
        {"status": "running", "script_version": SCRIPT_VERSION},
    )
    contract = read_json(args.contract.resolve())
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError("contract/controller version mismatch")
    if len(set(args.gpus)) != 2:
        raise RuntimeError("--gpus must name two distinct devices")
    write_json(
        args.output_root / "controller_provenance.json",
        {
            "script_version": SCRIPT_VERSION,
            "worker_version": WORKER_VERSION,
            "analysis_version": ANALYSIS_VERSION,
            "controller_sha256": sha256_file(Path(__file__).resolve()),
            "worker_sha256": sha256_file(WORKER),
            "analyzer_sha256": sha256_file(ANALYZER),
            "contract_sha256": sha256_file(args.contract.resolve()),
        },
    )

    checkpoint_specs = {
        "early": (
            args.early_checkpoint,
            args.early_source,
            args.early_profile,
            args.gpus[0],
        ),
        "late": (
            args.late_checkpoint,
            args.late_source,
            args.late_profile,
            args.gpus[1],
        ),
    }
    certificates = {}
    for label, (checkpoint, _source, _profile, _gpu) in checkpoint_specs.items():
        certificates[label] = hash_checkpoint(
            label,
            checkpoint,
            contract["checkpoints"][label]["sha256"],
            args.output_root,
        )

    smoke_jobs = []
    for label, (checkpoint, source, profile, gpu) in checkpoint_specs.items():
        output = args.output_root / "smoke" / label
        output.mkdir(parents=True)
        smoke_jobs.append(
            (
                f"smoke/{label}",
                gpu,
                worker_command(
                    args,
                    contract,
                    "smoke",
                    label,
                    checkpoint,
                    source,
                    profile,
                    certificates[label],
                    output,
                    None,
                ),
                output,
            )
        )
    write_json(
        args.output_root / "smoke_commands.json",
        {
            "jobs": [
                {
                    "label": label,
                    "gpu": gpu,
                    "command": command,
                    "output": str(output),
                }
                for label, gpu, command, output in smoke_jobs
            ]
        },
    )
    smoke_results = run_stage(smoke_jobs)
    for label in checkpoint_specs:
        manifest = read_json(
            args.output_root
            / "smoke"
            / label
            / "llama_block_audit_manifest.json"
        )
        if manifest.get("passed") is not True:
            raise RuntimeError(f"{label} smoke did not pass")

    formal_jobs = []
    for label, (checkpoint, source, profile, gpu) in checkpoint_specs.items():
        output = args.output_root / "formal" / label
        output.mkdir(parents=True)
        smoke_manifest = (
            args.output_root
            / "smoke"
            / label
            / "llama_block_audit_manifest.json"
        )
        formal_jobs.append(
            (
                f"formal/{label}",
                gpu,
                worker_command(
                    args,
                    contract,
                    "formal",
                    label,
                    checkpoint,
                    source,
                    profile,
                    certificates[label],
                    output,
                    smoke_manifest,
                ),
                output,
            )
        )
    write_json(
        args.output_root / "formal_commands.json",
        {
            "jobs": [
                {
                    "label": label,
                    "gpu": gpu,
                    "command": command,
                    "output": str(output),
                }
                for label, gpu, command, output in formal_jobs
            ]
        },
    )
    formal_results = run_stage(formal_jobs)
    for label in checkpoint_specs:
        manifest = read_json(
            args.output_root
            / "formal"
            / label
            / "llama_block_audit_manifest.json"
        )
        if manifest.get("passed") is not True:
            raise RuntimeError(f"{label} formal did not pass")

    analysis = args.output_root / "analysis"
    analysis.mkdir()
    analysis_command = [
        sys.executable,
        str(ANALYZER),
        "--run-dir",
        str(args.output_root),
        "--contract",
        str(args.contract),
        "--output-dir",
        str(analysis),
    ]
    print("LLaMA block audit analysis command:", json.dumps(analysis_command))
    subprocess.run(analysis_command, check=True)
    analysis_manifest = read_json(
        analysis / "llama_block_audit_analysis_manifest.json"
    )
    if analysis_manifest.get("passed") is not True:
        raise RuntimeError(f"analysis did not pass: {analysis_manifest}")
    write_json(
        args.output_root / "commands.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "worker_version": WORKER_VERSION,
            "analysis_version": ANALYSIS_VERSION,
            "smoke": smoke_results,
            "formal": formal_results,
            "analysis": analysis_command,
        },
    )
    write_json(
        args.output_root / "status.json",
        {
            "status": "passed",
            "script_version": SCRIPT_VERSION,
            "classification": analysis_manifest["classification"],
        },
    )
    print(f"LLaMA block audit artifacts: {args.output_root}")
    print(
        "LLaMA block audit manifest:",
        analysis / "llama_block_audit_analysis_manifest.json",
    )


if __name__ == "__main__":
    main()
