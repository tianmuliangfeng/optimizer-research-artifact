#!/usr/bin/env python3
"""Standard-library controller for MECH-03 cross-fit shadow updates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_VERSION = "2026-07-27.2"
HERE = Path(__file__).resolve().parent
WORKER = HERE / "mech03_worker.py"
DEFAULT_CONTRACT = HERE / "prediction_contract.json"
FAMILIES = ("r1", "gpt_bridge", "llama124")
FORMAL_LAYERS = (0, 4, 8, 11)
FORMAL_REPEATS = 4
FORMAL_BATCHES_PER_SPLIT = 8
OFFSET_STRIDE = 4096
FORMAL_OFFSETS = tuple(
    range(
        0,
        FORMAL_REPEATS * 2 * FORMAL_BATCHES_PER_SPLIT * OFFSET_STRIDE,
        OFFSET_STRIDE,
    )
)
STEP_MULTIPLIERS = (0.0, 0.25, 0.5, 1.0)


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
    parser.add_argument("--mech02-formal-dir", type=Path, required=True)
    parser.add_argument("--prediction-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--data-pattern", required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=FORMAL_LAYERS)
    parser.add_argument("--repeat-offsets", nargs="+", type=int, default=FORMAL_OFFSETS)
    parser.add_argument("--repeats", type=int, default=FORMAL_REPEATS)
    parser.add_argument(
        "--batches-per-split", type=int, default=FORMAL_BATCHES_PER_SPLIT
    )
    parser.add_argument("--device-batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--max-activation-rows", type=int, default=2048)
    parser.add_argument("--ridge-mult", type=float, default=0.2)
    parser.add_argument("--ridge-eps", type=float, default=1e-8)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument(
        "--step-multipliers", nargs="+", type=float, default=STEP_MULTIPLIERS
    )
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--execution-domain", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-prefix", default="mech03")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    expected_offsets = args.repeats * 2 * args.batches_per_split
    if args.repeats < 1 or args.batches_per_split < 1:
        parser.error("repeats and batches-per-split must be positive")
    if len(args.repeat_offsets) != expected_offsets:
        parser.error(
            "--repeat-offsets count must equal repeats * 2 * batches-per-split"
        )
    if len(set(args.repeat_offsets)) != len(args.repeat_offsets):
        parser.error("--repeat-offsets must be unique")
    if len(set(args.layers)) != len(args.layers):
        parser.error("--layers must be unique")
    if args.device_batch_size != 1:
        parser.error("the frozen contract requires device-batch-size=1 per window")
    if args.sequence_length <= 0 or args.max_activation_rows <= 0:
        parser.error("sequence length and row cap must be positive")
    if args.ridge_mult < 0 or args.ridge_eps <= 0:
        parser.error("ridge multiplier must be non-negative and epsilon positive")
    if not 0 <= args.momentum < 1 or args.ns_steps <= 0:
        parser.error("invalid momentum or Newton-Schulz steps")
    if tuple(args.step_multipliers) != STEP_MULTIPLIERS:
        parser.error(f"step multipliers are frozen to {STEP_MULTIPLIERS}")
    if len(args.checkpoint_sha256) != 64:
        parser.error("--checkpoint-sha256 must be a full SHA-256")
    if args.analysis_tier == "formal":
        if args.smoke_manifest is None:
            parser.error("--smoke-manifest is required for formal analysis")
        if tuple(args.layers) != FORMAL_LAYERS:
            parser.error(f"formal layers are frozen to {FORMAL_LAYERS}")
        if args.repeats != FORMAL_REPEATS:
            parser.error(f"formal repeats are frozen to {FORMAL_REPEATS}")
        if args.batches_per_split != FORMAL_BATCHES_PER_SPLIT:
            parser.error(
                "formal batches-per-split are frozen to "
                f"{FORMAL_BATCHES_PER_SPLIT}"
            )
        if tuple(args.repeat_offsets) != FORMAL_OFFSETS:
            parser.error("formal repeat offsets differ from the frozen contract")
    return args


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S+0000")


def output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir.resolve()
    artifact_root = HERE.parents[1]
    root = (
        Path(os.environ.get("SNM_RESULTS_ROOT", str(artifact_root / "runs"))).expanduser()
        / "31_mech03_crossfit_shadow"
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
        "--mech02-formal-dir",
        str(args.mech02_formal_dir.resolve()),
        "--prediction-contract",
        str(args.prediction_contract.resolve()),
        "--data-pattern",
        args.data_pattern,
        "--layers",
        *[str(value) for value in args.layers],
        "--repeat-offsets",
        *[str(value) for value in args.repeat_offsets],
        "--repeats",
        str(args.repeats),
        "--batches-per-split",
        str(args.batches_per_split),
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
        "--momentum",
        str(args.momentum),
        "--ns-steps",
        str(args.ns_steps),
        "--step-multipliers",
        *[str(value) for value in args.step_multipliers],
        "--host-id",
        args.host_id,
        "--execution-domain",
        args.execution_domain,
    ]
    if args.hash_checkpoint:
        command.append("--hash-checkpoint")
    if args.smoke_manifest:
        command += ["--smoke-manifest", str(args.smoke_manifest.resolve())]
    return command


def main() -> None:
    args = parse_args()
    destination = output_dir(args)
    command = worker_command(args, destination)
    print("MECH-03 output:", destination, flush=True)
    print("MECH-03 command:", json.dumps(command), flush=True)
    if args.dry_run:
        return
    destination.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run(command, check=False)
    manifest = destination / "mech03_manifest.json"
    if manifest.is_file():
        print("MECH-03 manifest:", manifest, flush=True)
    print("MECH-03 artifacts:", destination, flush=True)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
