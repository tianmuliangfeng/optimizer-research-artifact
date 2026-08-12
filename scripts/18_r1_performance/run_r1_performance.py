"""Controller for isolated R1 performance experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from r1_perf_source_builder import CPROJ_MODE, METHODS, PerfSource, build_perf_source


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_ROOT = (
    Path(os.environ.get("SNM_RESULTS_ROOT", str(ARTIFACT_ROOT / "runs"))).expanduser()
    / "18_r1_performance"
    / "results"
)
OPERATOR_METHODS = ("none", "diag", "block4", "dense_full")
METADATA_RE = re.compile(
    r"R1_METADATA method=(?P<method>\S+) cproj_k_mode=(?P<mode>\S+) "
    r"seed=(?P<seed>\d+) init_sha256=(?P<sha>[0-9a-f]{64})"
)
VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+) val_loss:(?P<loss>[-+0-9.eE]+) "
    r"train_time:(?P<time>[-+0-9.eE]+)ms step_avg:(?P<avg>[-+0-9.eEnNaA]+)ms"
)
PEAK_RE = re.compile(r"peak memory consumption: (?P<mib>\d+) MiB")
KV_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>\d+)")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S+0000")


def lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--operator-benchmark", action="store_true")
    parser.add_argument("--numerical-smoke", action="store_true")
    parser.add_argument("--training-benchmark", action="store_true")
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--smoke-steps", type=int, default=34)
    parser.add_argument("--timed-steps", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--operator-warmup", type=int, default=3)
    parser.add_argument("--operator-repeats", type=int, default=10)
    args = parser.parse_args()
    modes = [
        args.dry_run,
        args.preflight,
        args.operator_benchmark,
        args.numerical_smoke,
        args.training_benchmark,
    ]
    if sum(bool(value) for value in modes) != 1:
        parser.error("select exactly one mode")
    if len(set(args.methods)) != len(args.methods):
        parser.error("--methods must not contain duplicates")
    if args.smoke_steps < 34:
        parser.error("--smoke-steps must reach the first K refresh at step 32")
    if args.timed_steps < 64:
        parser.error("--timed-steps must be at least 64")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.training_benchmark and args.smoke_manifest is None:
        parser.error("--training-benchmark requires --smoke-manifest")
    return args


def runtime_fingerprint(python_exe: Path, official_repo: Path) -> dict[str, Any]:
    code = r'''
import json, platform, sys
from pathlib import Path
import numpy, torch, triton
import triton_kernels
gpu = torch.cuda.get_device_properties(0)
print(json.dumps({
    "python_executable": str(Path(sys.executable).absolute()),
    "python": sys.version.replace("\n", " "),
    "platform": platform.platform(),
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "triton": triton.__version__,
    "triton_module": str(Path(triton.__file__).resolve()),
    "triton_kernels_module": str(Path(triton_kernels.__file__).resolve()),
    "gpu_name": gpu.name,
    "gpu_total_memory_bytes": gpu.total_memory,
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
}))
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(official_repo) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [str(python_exe), "-c", code],
        cwd=official_repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"training runtime validation failed:\n{result.stdout}\n{result.stderr}")
    lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"runtime fingerprint JSON missing:\n{result.stdout}")
    return json.loads(lines[-1])


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    official_repo = args.official_repo.resolve()
    python_exe = lexical_absolute(args.python_exe)
    data_dir = (args.data_dir or official_repo / "data" / "fineweb10B").resolve()
    for filename in ("train_gpt_muon_1.py", "train_gpt_newton_muon_1.py", "triton_kernels.py"):
        if not (official_repo / filename).is_file():
            raise FileNotFoundError(official_repo / filename)
    if not python_exe.is_file():
        raise FileNotFoundError(python_exe)
    train = sorted(data_dir.glob("fineweb_train_*.bin"))
    val = sorted(data_dir.glob("fineweb_val_*.bin"))
    if not train or not val:
        raise FileNotFoundError(f"FineWeb shards missing under {data_dir}")
    return official_repo, python_exe, data_dir


def build_all(official_repo: Path, methods: list[str]) -> dict[str, PerfSource]:
    return {method: build_perf_source(official_repo, method) for method in methods}


def plan_payload(
    args: argparse.Namespace,
    official_repo: Path,
    python_exe: Path,
    data_dir: Path,
    built: dict[str, PerfSource],
    runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "protocol": "r1_perf_v1",
        "scientific_boundary": {
            "quality_evidence": "existing R1; performance jobs are not pooled into quality statistics",
            "training_benchmark": "within-host relative timing only",
            "cross_host_rule": "never subtract an old-host method time from a new-host method time",
            "adamw_role": "end-to-end standard optimizer reference",
            "dense_full_role": "full c_proj covariance compute/memory upper bound",
        },
        "methods": list(built),
        "method_order": args.methods,
        "seed": args.seed,
        "official_repo": str(official_repo),
        "python_exe": str(python_exe),
        "data_dir": str(data_dir),
        "timed_steps": args.timed_steps,
        "warmup_steps_excluded_by_official_timer": 32,
        "training_total_updates": 32 + args.timed_steps,
        "repeats": args.repeats,
        "source": {
            method: {
                "base_script": source.base_script,
                "base_sha256": source.base_canonical_sha256,
                "derived_sha256": source.derived_sha256,
            }
            for method, source in built.items()
        },
        "runtime": runtime,
    }


def run_logged(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> tuple[int, float]:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        returncode = process.wait()
    return returncode, time.monotonic() - started


def environment_for(
    method: str,
    data_dir: Path,
    official_repo: Path,
    seed: int,
    steps: int,
    init_only: bool,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "R1_METHOD": method,
            "R1_CPROJ_K_MODE": CPROJ_MODE[method],
            "R1_SEED": str(seed),
            "R1_DATA_DIR": str(data_dir),
            "R1_INIT_ONLY": "1" if init_only else "0",
            "R1_SMOKE_TEST": "1",
            "R1_SMOKE_STEPS": str(steps),
            "R1_DISABLE_CHECKPOINT": "1",
            "PYTHONHASHSEED": str(seed),
            # Generated sources live under the artifact directory, so Python's
            # script directory is not the official repository.  Explicitly
            # expose the pinned local triton_kernels.py and preserve any
            # caller-supplied search path after it.
            "PYTHONPATH": str(official_repo)
            + os.pathsep
            + env.get("PYTHONPATH", ""),
        }
    )
    return env


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    metadata = list(METADATA_RE.finditer(text))
    vals = list(VAL_RE.finditer(text))
    peaks = list(PEAK_RE.finditer(text))
    k_line = next((line for line in text.splitlines() if line.startswith("R1_K_MEMORY ")), "")
    final_memory_line = next((line for line in text.splitlines() if line.startswith("R1_FINAL_MEMORY ")), "")
    if not metadata:
        raise RuntimeError(f"R1 metadata missing from {path}")
    result: dict[str, Any] = {
        "method": metadata[-1].group("method"),
        "cproj_k_mode": metadata[-1].group("mode"),
        "seed": int(metadata[-1].group("seed")),
        "init_sha256": metadata[-1].group("sha"),
        "k_memory": {m.group("key"): int(m.group("value")) for m in KV_RE.finditer(k_line)},
        "final_memory": {m.group("key"): int(m.group("value")) for m in KV_RE.finditer(final_memory_line)},
    }
    if vals:
        last = vals[-1]
        result.update(
            {
                "final_step": int(last.group("step")),
                "final_val_loss": float(last.group("loss")),
                "official_train_time_s": float(last.group("time")) / 1000.0,
                "official_step_avg_ms": float(last.group("avg")),
            }
        )
    if peaks:
        result["peak_memory_mib"] = int(peaks[-1].group("mib"))
    return result


def save_sources(batch: Path, built: dict[str, PerfSource]) -> None:
    source_dir = batch / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    for method, source in built.items():
        (source_dir / f"train_{method}.py").write_text(source.source, encoding="utf-8")
        (source_dir / f"train_{method}.diff").write_text(source.unified_diff, encoding="utf-8")


def rotated_order(methods: list[str], repeat: int) -> list[str]:
    offset = repeat % len(methods)
    return methods[offset:] + methods[:offset]


def run_training_batch(
    args: argparse.Namespace,
    official_repo: Path,
    python_exe: Path,
    data_dir: Path,
    built: dict[str, PerfSource],
    batch: Path,
    steps: int,
    repeats: int,
    init_only: bool = False,
) -> list[dict[str, Any]]:
    save_sources(batch, built)
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        order = rotated_order(args.methods, repeat)
        for position, method in enumerate(order):
            run_dir = batch / f"repeat{repeat + 1:02d}_{position + 1:02d}_{method}"
            run_dir.mkdir(parents=True, exist_ok=False)
            source_path = batch / "sources" / f"train_{method}.py"
            log_path = run_dir / "terminal.log"
            env = environment_for(
                method, data_dir, official_repo, args.seed, steps, init_only
            )
            returncode, wall = run_logged(
                [str(python_exe), str(source_path)], official_repo, env, log_path
            )
            if returncode != 0:
                raise RuntimeError(f"{method} failed with code {returncode}; see {log_path}")
            parsed = parse_log(log_path)
            parsed.update(
                {
                    "repeat": repeat + 1,
                    "position": position + 1,
                    "order": " ".join(order),
                    "wrapper_wall_elapsed_s": wall,
                    "source_sha256": built[method].derived_sha256,
                    "log_sha256": sha256_file(log_path),
                }
            )
            write_json(run_dir / "summary.json", parsed)
            rows.append(parsed)
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "method", "repeat", "position", "order", "seed", "init_sha256",
        "final_step", "final_val_loss", "official_train_time_s",
        "official_step_avg_ms", "peak_memory_mib", "wrapper_wall_elapsed_s",
        "source_sha256", "log_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]], timed_steps: int) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    block_times = [
        float(row["official_train_time_s"])
        for row in grouped.get("block4", [])
    ]
    block_median = statistics.median(block_times) if block_times else math.nan
    fields = [
        "method", "runs", "median_train_time_s", "mean_train_time_s",
        "stdev_train_time_s", "median_step_ms", "median_tokens_per_s",
        "delta_time_pct_vs_block4", "median_peak_memory_mib",
        "median_k_state_mib",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, method_rows in grouped.items():
            times = [float(row["official_train_time_s"]) for row in method_rows]
            step_ms = [float(row["official_step_avg_ms"]) for row in method_rows]
            peaks = [float(row.get("peak_memory_mib", math.nan)) for row in method_rows]
            k_mib = [
                float(row.get("k_memory", {}).get("k_state_bytes", 0)) / 1024**2
                for row in method_rows
            ]
            median_time = statistics.median(times)
            tokens_per_s = timed_steps * 512 * 1024 / median_time
            writer.writerow(
                {
                    "method": method,
                    "runs": len(method_rows),
                    "median_train_time_s": median_time,
                    "mean_train_time_s": statistics.fmean(times),
                    "stdev_train_time_s": statistics.stdev(times) if len(times) > 1 else math.nan,
                    "median_step_ms": statistics.median(step_ms),
                    "median_tokens_per_s": tokens_per_s,
                    "delta_time_pct_vs_block4": (
                        100.0 * (median_time - block_median) / block_median
                        if math.isfinite(block_median) and block_median > 0 else math.nan
                    ),
                    "median_peak_memory_mib": statistics.median(peaks),
                    "median_k_state_mib": statistics.median(k_mib),
                }
            )


def validate_smoke_manifest(path: Path, runtime: dict[str, Any], built: dict[str, PerfSource]) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("smoke manifest is not complete")
    if manifest.get("runtime") != runtime:
        raise RuntimeError("runtime fingerprint differs from smoke certificate")
    current = {method: source.derived_sha256 for method, source in built.items()}
    if manifest.get("source_sha256") != current:
        raise RuntimeError("derived source fingerprint differs from smoke certificate")
    if manifest.get("methods") != list(built):
        raise RuntimeError("method list differs from smoke certificate")
    return manifest


def main() -> None:
    args = parse_args()
    official_repo, python_exe, data_dir = validate_inputs(args)
    built = build_all(official_repo, args.methods)
    if args.dry_run:
        print(json.dumps(plan_payload(args, official_repo, python_exe, data_dir, built, None), indent=2))
        return

    runtime = runtime_fingerprint(python_exe, official_repo)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.preflight:
        batch = root / f"{utc_stamp()}_preflight_seed{args.seed}"
        batch.mkdir(parents=True, exist_ok=False)
        plan = plan_payload(args, official_repo, python_exe, data_dir, built, runtime)
        write_json(batch / "perf_plan.json", plan)
        rows = run_training_batch(
            args, official_repo, python_exe, data_dir, built, batch, 34, 1, init_only=True
        )
        hashes = {row["init_sha256"] for row in rows}
        if len(hashes) != 1:
            raise RuntimeError(f"initialization hashes differ: {hashes}")
        manifest = {
            "protocol": "r1_perf_preflight_v1",
            "status": "complete",
            "methods": list(built),
            "runtime": runtime,
            "source_sha256": {method: source.derived_sha256 for method, source in built.items()},
            "init_sha256": next(iter(hashes)),
            "artifact_dir": str(batch),
        }
        write_json(batch / "perf_manifest.json", manifest)
        print(f"R1-PERF preflight artifact: {batch}")
        return

    if args.operator_benchmark:
        batch = root / f"{utc_stamp()}_operators_seed{args.seed}"
        batch.mkdir(parents=True, exist_ok=False)
        write_json(batch / "perf_plan.json", plan_payload(args, official_repo, python_exe, data_dir, built, runtime))
        methods = [method for method in args.methods if method in OPERATOR_METHODS]
        if not methods:
            raise RuntimeError("operator benchmark needs one of none/diag/block4/dense_full")
        command = [
            str(python_exe),
            str(SCRIPT_DIR / "benchmark_r1_operators.py"),
            "--official-repo", str(official_repo),
            "--output-dir", str(batch / "operator_results"),
            "--methods", *methods,
            "--warmup", str(args.operator_warmup),
            "--repeats", str(args.operator_repeats),
            "--seed", str(args.seed),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(official_repo) + os.pathsep + env.get("PYTHONPATH", "")
        returncode, _ = run_logged(command, official_repo, env, batch / "terminal.log")
        if returncode != 0:
            raise RuntimeError(f"operator benchmark failed; see {batch / 'terminal.log'}")
        print(f"R1-PERF operator artifact: {batch}")
        return


    if args.numerical_smoke:
        batch = root / f"{utc_stamp()}_smoke_seed{args.seed}"
        batch.mkdir(parents=True, exist_ok=False)
        write_json(batch / "perf_plan.json", plan_payload(args, official_repo, python_exe, data_dir, built, runtime))
        rows = run_training_batch(
            args, official_repo, python_exe, data_dir, built, batch, args.smoke_steps, 1
        )
        hashes = {row["init_sha256"] for row in rows}
        if len(hashes) != 1:
            raise RuntimeError(f"initialization hashes differ: {hashes}")
        for row in rows:
            if row.get("final_step") != args.smoke_steps:
                raise RuntimeError(f"incomplete smoke for {row['method']}: {row}")
            if not math.isfinite(row.get("final_val_loss", math.nan)):
                raise RuntimeError(f"non-finite smoke loss for {row['method']}")
        write_summary_csv(batch / "smoke_summary.csv", rows)
        manifest = {
            "protocol": "r1_perf_numerical_smoke_v1",
            "status": "complete",
            "methods": list(built),
            "runtime": runtime,
            "source_sha256": {method: source.derived_sha256 for method, source in built.items()},
            "init_sha256": next(iter(hashes)),
            "smoke_steps": args.smoke_steps,
            "artifact_dir": str(batch),
        }
        write_json(batch / "perf_manifest.json", manifest)
        print(f"R1-PERF smoke artifact: {batch}")
        print(f"R1-PERF smoke manifest: {batch / 'perf_manifest.json'}")
        return

    validate_smoke_manifest(args.smoke_manifest.resolve(), runtime, built)
    batch = root / f"{utc_stamp()}_training_benchmark_seed{args.seed}"
    batch.mkdir(parents=True, exist_ok=False)
    write_json(batch / "perf_plan.json", plan_payload(args, official_repo, python_exe, data_dir, built, runtime))
    rows = run_training_batch(
        args,
        official_repo,
        python_exe,
        data_dir,
        built,
        batch,
        32 + args.timed_steps,
        args.repeats,
    )
    write_summary_csv(batch / "training_benchmark_runs.csv", rows)
    write_aggregate_csv(
        batch / "training_benchmark_summary.csv", rows, args.timed_steps
    )
    manifest = {
        "protocol": "r1_perf_training_benchmark_v1",
        "status": "complete",
        "methods": list(built),
        "runtime": runtime,
        "source_sha256": {method: source.derived_sha256 for method, source in built.items()},
        "timed_steps": args.timed_steps,
        "repeats": args.repeats,
        "artifact_dir": str(batch),
    }
    write_json(batch / "perf_manifest.json", manifest)
    print(f"R1-PERF training benchmark artifact: {batch}")


if __name__ == "__main__":
    main()
