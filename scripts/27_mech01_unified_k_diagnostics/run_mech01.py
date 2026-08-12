#!/usr/bin/env python3
"""Controller for MECH-01 unified K-diagnostics validation.

The controller deliberately imports only the Python standard library.  The
checkpoint load and numerical work are delegated to ``mech01_worker.py`` with
the explicitly selected training interpreter.  This keeps controller/runtime
provenance separate and makes it possible to replay one tensor bundle in two
different pinned environments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-24.1"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESULTS_DIR = (
    Path(os.environ.get("SNM_RESULTS_ROOT", str(PROJECT_ROOT / "runs"))).expanduser()
    / "27_mech01_unified_k_diagnostics"
)
FAMILIES = ("r1", "gpt_bridge", "llama124", "llama1b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MECH-01 read-only checkpoint/schema preflight, numerical smoke, "
            "and fixed tensor-bundle runtime replay."
        )
    )
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--preflight", action="store_true")
    stage.add_argument("--numerical-smoke", action="store_true")
    stage.add_argument("--replay-bundle", action="store_true")
    stage.add_argument("--compare-replays", action="store_true")

    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--source-script",
        type=Path,
        help=(
            "Exact training source stored with the run.  For R1 this must be "
            "the generated train_r1_*.py, not an unrelated clean-repo file."
        ),
    )
    parser.add_argument(
        "--triton-kernels",
        type=Path,
        help="Exact triton_kernels.py paired with --source-script.",
    )
    parser.add_argument(
        "--profile-script",
        type=Path,
        help=(
            "LLaMA-1B shape wrapper saved with the run.  --source-script must "
            "still point to the copied audited base trainer."
        ),
    )
    parser.add_argument("--data-pattern")
    parser.add_argument(
        "--method",
        default="auto",
        help="Checkpoint method; auto is supported for preflight.",
    )
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--device-batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument(
        "--probe-offsets",
        nargs=4,
        type=int,
        default=(0, 4096, 8192, 12288),
        metavar=("BUILD_A", "BUILD_B", "HELDOUT_A", "HELDOUT_B"),
    )
    parser.add_argument("--max-activation-rows", type=int, default=2048)
    parser.add_argument("--ridge-mult", type=float, default=0.2)
    parser.add_argument("--ridge-eps", type=float, default=1e-8)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=("none", "diag", "block4", "dense_full"),
        choices=("none", "diag", "block4", "dense_full"),
    )
    parser.add_argument(
        "--spectrum-dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument("--export-bundle-layer", type=int)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--replay-a", type=Path)
    parser.add_argument("--replay-b", type=Path)
    parser.add_argument("--replay-label-a", default="runtime_a")
    parser.add_argument("--replay-label-b", default="runtime_b")
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument(
        "--checkpoint-sha256",
        default="",
        help="Trusted MECH-00 full hash. The large checkpoint is not re-hashed by default.",
    )
    parser.add_argument("--hash-checkpoint", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-prefix", default="mech01")
    parser.add_argument("--host-id", default="")
    parser.add_argument("--execution-domain", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_args(parser, args)
    return args


def selected_stage(args: argparse.Namespace) -> str:
    if args.preflight:
        return "preflight"
    if args.numerical_smoke:
        return "numerical_smoke"
    if args.replay_bundle:
        return "replay"
    return "compare"


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    stage = selected_stage(args)
    if stage in {"preflight", "numerical_smoke"}:
        for name in ("family", "checkpoint", "source_script", "triton_kernels"):
            if getattr(args, name) in (None, ""):
                parser.error(f"--{name.replace('_', '-')} is required for {stage}")
        if args.family == "llama1b" and args.profile_script is None:
            parser.error("--profile-script is required for the llama1b family")
    if stage == "numerical_smoke" and not args.data_pattern:
        parser.error("--data-pattern is required for --numerical-smoke")
    if stage == "replay":
        if args.bundle is None:
            parser.error("--bundle is required for --replay-bundle")
        if (
            args.family is None
            or args.source_script is None
            or args.triton_kernels is None
        ):
            parser.error(
                "--family, --source-script, and --triton-kernels are required "
                "for --replay-bundle"
            )
    if stage == "compare":
        if args.replay_a is None or args.replay_b is None:
            parser.error("--replay-a and --replay-b are required for --compare-replays")
    if args.device_batch_size <= 0 or args.sequence_length <= 0:
        parser.error("batch size and sequence length must be positive")
    if args.max_activation_rows <= 0:
        parser.error("--max-activation-rows must be positive")
    if args.ridge_mult < 0 or args.ridge_eps <= 0:
        parser.error("ridge multiplier must be non-negative and epsilon positive")
    if args.ns_steps <= 0:
        parser.error("--ns-steps must be positive")
    if len(set(args.probe_offsets)) != 4:
        parser.error("--probe-offsets must contain four distinct token offsets")
    if args.atol < 0 or args.rtol < 0:
        parser.error("comparison tolerances must be non-negative")


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S+0000")


def output_dir_for(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    family = args.family or "bundle"
    return (
        args.results_dir.resolve()
        / selected_stage(args)
        / f"{timestamp()}_{args.run_prefix}_{family}"
    )


def worker_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    worker = SCRIPT_DIR / "mech01_worker.py"
    stage = selected_stage(args)
    mode = {
        "preflight": "preflight",
        "numerical_smoke": "smoke",
        "replay": "replay",
    }[stage]
    command = [
        str(Path(args.python_exe).expanduser()),
        str(worker),
        "--mode",
        mode,
        "--output-dir",
        str(output_dir),
        "--ridge-mult",
        str(args.ridge_mult),
        "--ridge-eps",
        str(args.ridge_eps),
        "--momentum",
        str(args.momentum),
        "--ns-steps",
        str(args.ns_steps),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--host-id",
        args.host_id,
        "--execution-domain",
        args.execution_domain,
    ]
    if args.family:
        command += ["--family", args.family]
    if args.checkpoint:
        command += ["--checkpoint", str(args.checkpoint.resolve())]
    if args.source_script:
        command += ["--source-script", str(args.source_script.resolve())]
    if args.triton_kernels:
        command += ["--triton-kernels", str(args.triton_kernels.resolve())]
    if args.profile_script:
        command += ["--profile-script", str(args.profile_script.resolve())]
    if args.method:
        command += ["--method", args.method]
    if args.checkpoint_sha256:
        command += ["--checkpoint-sha256", args.checkpoint_sha256]
    if args.hash_checkpoint:
        command.append("--hash-checkpoint")
    if mode == "smoke":
        command += [
            "--data-pattern",
            args.data_pattern,
            "--device-batch-size",
            str(args.device_batch_size),
            "--sequence-length",
            str(args.sequence_length),
            "--max-activation-rows",
            str(args.max_activation_rows),
            "--spectrum-dtype",
            args.spectrum_dtype,
            "--probe-offsets",
            *[str(value) for value in args.probe_offsets],
            "--candidates",
            *args.candidates,
        ]
        if args.layers:
            command += ["--layers", *[str(value) for value in args.layers]]
        if args.export_bundle_layer is not None:
            command += ["--export-bundle-layer", str(args.export_bundle_layer)]
    if mode == "replay":
        command += ["--bundle", str(args.bundle.resolve())]
    return command


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def flatten_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_numbers(value[key], child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            flattened.update(flatten_numbers(item, child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flattened[prefix] = float(value)
    return flattened


def compare_replay_payloads(
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    bundle_a = payload_a.get("bundle_sha256")
    bundle_b = payload_b.get("bundle_sha256")
    source_a = payload_a.get("source_sha256")
    source_b = payload_b.get("source_sha256")
    triton_a = (payload_a.get("triton") or {}).get("sha256")
    triton_b = (payload_b.get("triton") or {}).get("sha256")
    worker_a = payload_a.get("script_version")
    worker_b = payload_b.get("script_version")
    rows_a = flatten_numbers(payload_a.get("results", {}))
    rows_b = flatten_numbers(payload_b.get("results", {}))
    keys_a = set(rows_a)
    keys_b = set(rows_b)
    comparisons = []
    for key in sorted(keys_a & keys_b):
        a = rows_a[key]
        b = rows_b[key]
        abs_diff = abs(a - b)
        limit = atol + rtol * abs(a)
        finite = math.isfinite(a) and math.isfinite(b)
        comparisons.append(
            {
                "metric": key,
                "a": a,
                "b": b,
                "abs_diff": abs_diff,
                "limit": limit,
                "pass": bool(finite and abs_diff <= limit),
            }
        )
    checks = {
        "same_bundle_sha256": bool(bundle_a and bundle_a == bundle_b),
        "same_diagnostic_source_sha256": bool(source_a and source_a == source_b),
        "same_triton_kernels_sha256": bool(triton_a and triton_a == triton_b),
        "same_worker_version": bool(worker_a and worker_a == worker_b),
        "metric_key_sets_equal": keys_a == keys_b,
        "all_metrics_within_tolerance": bool(comparisons)
        and all(row["pass"] for row in comparisons),
    }
    return {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "bundle_sha256_a": bundle_a,
        "bundle_sha256_b": bundle_b,
        "source_sha256_a": source_a,
        "source_sha256_b": source_b,
        "triton_kernels_sha256_a": triton_a,
        "triton_kernels_sha256_b": triton_b,
        "worker_version_a": worker_a,
        "worker_version_b": worker_b,
        "atol": atol,
        "rtol": rtol,
        "missing_from_a": sorted(keys_b - keys_a),
        "missing_from_b": sorted(keys_a - keys_b),
        "comparisons": comparisons,
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_compare(args: argparse.Namespace, output_dir: Path) -> int:
    payload_a = read_json(args.replay_a.resolve())
    payload_b = read_json(args.replay_b.resolve())
    result = compare_replay_payloads(
        payload_a, payload_b, atol=args.atol, rtol=args.rtol
    )
    result["replay_a"] = str(args.replay_a.resolve())
    result["replay_b"] = str(args.replay_b.resolve())
    result["label_a"] = args.replay_label_a
    result["label_b"] = args.replay_label_b
    write_json(output_dir / "runtime_equivalence.json", result)
    write_json(
        output_dir / "mech01_manifest.json",
        {
            "schema_version": 1,
            "stage": "compare",
            "passed": result["passed"],
            "runtime_equivalence": str(
                (output_dir / "runtime_equivalence.json").resolve()
            ),
            "script_version": SCRIPT_VERSION,
        },
    )
    return 0 if result["passed"] else 2


def main() -> None:
    args = parse_args()
    output_dir = output_dir_for(args)
    stage = selected_stage(args)
    if stage == "compare":
        command_description = [
            "internal-compare",
            str(args.replay_a),
            str(args.replay_b),
        ]
    else:
        command_description = worker_command(args, output_dir)
    print("MECH-01 stage:", stage, flush=True)
    print("MECH-01 output:", output_dir, flush=True)
    print("MECH-01 command:", json.dumps(command_description), flush=True)
    if args.dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=False)
    if stage == "compare":
        return_code = run_compare(args, output_dir)
    else:
        completed = subprocess.run(command_description, check=False)
        return_code = int(completed.returncode)
    print("MECH-01 artifacts:", output_dir, flush=True)
    manifest = output_dir / "mech01_manifest.json"
    if manifest.is_file():
        print("MECH-01 manifest: ", manifest, flush=True)
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
