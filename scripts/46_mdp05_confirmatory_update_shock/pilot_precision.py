#!/usr/bin/env python3
"""Outcome-blind synthetic precision calibration for the frozen MDP-05 mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import triton


SCRIPT_VERSION = "2026-08-04.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    certificate = output / "pilot_precision_certificate.json"
    try:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        runtime = contract["runtime_contract"]
        calibration = contract["precision_calibration"]
        checks = {
            "pilot_is_synthetic": contract["design"][
                "pilot_uses_checkpoint_or_outcome_data"
            ]
            is False,
            "fixed_mode": calibration["mode"]
            == "fixed_float64_slice_diagnostic",
            "full_float64_not_required": calibration[
                "full_5504_float64_required"
            ]
            is False,
            "python": sys.version.split()[0] == runtime["python"],
            "torch": torch.__version__ == runtime["torch"],
            "torch_cuda": torch.version.cuda == runtime["torch_cuda"],
            "triton": triton.__version__ == runtime["triton"],
            "numpy": np.__version__ == runtime["numpy"],
            "cuda": torch.cuda.is_available(),
            "one_visible_gpu": torch.cuda.device_count() == 1,
        }
        device = {}
        benchmark = {}
        if checks["cuda"] and checks["one_visible_gpu"]:
            properties = torch.cuda.get_device_properties(0)
            device = {
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory": properties.total_memory,
            }
            checks["device"] = (
                runtime["gpu_name_contains"] in properties.name
                and [properties.major, properties.minor]
                == runtime["compute_capability"]
                and properties.total_memory
                >= int(runtime["minimum_gpu_memory_bytes"])
            )
            torch.manual_seed(int(calibration["seed"]))
            torch.cuda.manual_seed_all(int(calibration["seed"]))
            torch.cuda.reset_peak_memory_stats()
            size = int(calibration["coordinate_count"])
            started = time.perf_counter()
            matrix = torch.randn((size, size), device="cuda", dtype=torch.float64)
            a = matrix.T @ matrix + 0.2 * torch.eye(
                size, device="cuda", dtype=torch.float64
            )
            inverse = torch.cholesky_inverse(torch.linalg.cholesky(a))
            residual = torch.linalg.norm(a @ inverse - torch.eye(size, device="cuda", dtype=torch.float64)) / math.sqrt(size)
            torch.cuda.synchronize()
            benchmark = {
                "mode": calibration["mode"],
                "matrix_size": size,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                "inverse_residual": float(residual.item()),
                "finite": bool(torch.isfinite(inverse).all().item()),
                "full_5504_square_was_allocated": False,
                "checkpoint_or_dataset_was_opened": False,
            }
            checks["benchmark_finite"] = benchmark["finite"] and math.isfinite(
                benchmark["inverse_residual"]
            )
        else:
            checks["device"] = False
            checks["benchmark_finite"] = False
        passed = all(checks.values())
        payload = {
            "schema_version": "mdp05_precision_pilot_certificate_v1",
            "script_version": SCRIPT_VERSION,
            "contract": str(args.contract.resolve()),
            "contract_sha256": sha256_file(args.contract.resolve()),
            "outcome_blind": True,
            "checkpoint_or_dataset_opened": False,
            "selected_mode": calibration["mode"],
            "device": device,
            "benchmark": benchmark,
            "checks": checks,
            "passed": passed,
        }
        atomic_json(certificate, payload)
        print(f"MDP05_PILOT_CERTIFICATE={certificate}", flush=True)
        print(f"MDP-05 precision pilot passed={passed}", flush=True)
        return 0 if passed else 2
    except Exception as exc:
        atomic_json(
            output / "pilot_status.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(
            f"MDP-05 pilot failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
