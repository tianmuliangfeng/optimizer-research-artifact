#!/usr/bin/env python3
"""Standard-library controller for MECH-02 checkpoint K-geometry."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_VERSION = "2026-07-27.1"
HERE = Path(__file__).resolve().parent
WORKER = HERE / "mech02_worker.py"
FAMILIES = ("r1", "gpt_bridge", "llama124")
DEFAULT_OFFSETS = (0, 4096, 8192, 12288, 16384, 20480, 24576, 28672)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-tier", choices=("smoke", "formal"), required=True)
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--python-exe", required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--hash-checkpoint", action="store_true")
    parser.add_argument("--source-script", type=Path, required=True)
    parser.add_argument("--triton-kernels", type=Path, required=True)
    parser.add_argument("--mech01-smoke-dir", type=Path, required=True)
    parser.add_argument("--data-pattern", required=True)
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--repeat-offsets", nargs="+", type=int, default=DEFAULT_OFFSETS)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--batches-per-repeat", type=int, default=2)
    parser.add_argument("--device-batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--max-activation-rows", type=int, default=2048)
    parser.add_argument("--ridge-mult", type=float, default=0.2)
    parser.add_argument("--ridge-eps", type=float, default=1e-8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--spectrum-dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--execution-domain", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-prefix", default="mech02")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if args.batches_per_repeat < 1:
        parser.error("--batches-per-repeat must be positive")
    if len(args.repeat_offsets) != args.repeats * args.batches_per_repeat:
        parser.error(
            "--repeat-offsets count must equal repeats * batches-per-repeat"
        )
    if len(set(args.repeat_offsets)) != len(args.repeat_offsets):
        parser.error("--repeat-offsets must be unique")
    if args.device_batch_size <= 0 or args.sequence_length <= 0:
        parser.error("batch size and sequence length must be positive")
    if args.max_activation_rows <= 0 or args.top_k <= 0:
        parser.error("row cap and top-k must be positive")
    if args.ridge_mult < 0 or args.ridge_eps <= 0:
        parser.error("ridge multiplier must be non-negative and epsilon positive")
    if len(args.checkpoint_sha256) != 64:
        parser.error("--checkpoint-sha256 must be a full SHA-256")
    if args.analysis_tier == "formal" and args.smoke_manifest is None:
        parser.error("--smoke-manifest is required for formal analysis")
    return args


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S+0000")


def output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir.resolve()
    artifact_root = HERE.parents[1]
    root = (
        Path(os.environ.get("SNM_RESULTS_ROOT", str(artifact_root / "runs"))).expanduser()
        / "30_mech02_k_geometry"
    )
    return (root / f"{timestamp()}_{args.run_prefix}_{args.family}").resolve()


def worker_command(args: argparse.Namespace, destination: Path) -> list[str]:
    command = [
        str(Path(args.python_exe).expanduser()),
        str(WORKER),
        "--output-dir",
        str(destination),
        "--analysis-tier",
        args.analysis_tier,
        "--family",
        args.family,
        "--method",
        args.method,
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--checkpoint-sha256",
        args.checkpoint_sha256.lower(),
        "--source-script",
        str(args.source_script.resolve()),
        "--triton-kernels",
        str(args.triton_kernels.resolve()),
        "--mech01-smoke-dir",
        str(args.mech01_smoke_dir.resolve()),
        "--data-pattern",
        args.data_pattern,
        "--repeat-offsets",
        *[str(value) for value in args.repeat_offsets],
        "--repeats",
        str(args.repeats),
        "--batches-per-repeat",
        str(args.batches_per_repeat),
        "--device-batch-size",
        str(args.device_batch_size),
        "--sequence-length",
        str(args.sequence_length),
        "--max-activation-rows",
        str(args.max_activation_rows),
        "--ridge-mult",
        str(args.ridge_mult),
        "--ridge-eps",
        str(args.ridge_eps),
        "--top-k",
        str(args.top_k),
        "--spectrum-dtype",
        args.spectrum_dtype,
        "--host-id",
        args.host_id,
        "--execution-domain",
        args.execution_domain,
    ]
    if args.hash_checkpoint:
        command.append("--hash-checkpoint")
    if args.smoke_manifest:
        command += ["--smoke-manifest", str(args.smoke_manifest.resolve())]
    if args.layers:
        command += ["--layers", *[str(value) for value in args.layers]]
    return command


def main() -> None:
    args = parse_args()
    destination = output_dir(args)
    command = worker_command(args, destination)
    print("MECH-02 output:", destination, flush=True)
    print("MECH-02 command:", json.dumps(command), flush=True)
    if args.dry_run:
        return
    destination.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run(command, check=False)
    manifest = destination / "mech02_manifest.json"
    if manifest.is_file():
        print("MECH-02 manifest:", manifest, flush=True)
    print("MECH-02 artifacts:", destination, flush=True)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
