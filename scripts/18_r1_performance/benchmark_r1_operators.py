"""CUDA microbenchmarks for the R1 c_proj preconditioner representations.

This is intentionally a component benchmark.  It does not claim full-training
throughput; the companion runner measures that with the exact R1 model.
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import Tensor


METHODS = ("none", "diag", "block4", "dense_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--tokens-per-microbatch", type=int, default=64 * 1024)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--refresh", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.layers <= 0 or args.width <= 0 or args.tokens_per_microbatch <= 0:
        parser.error("shape arguments must be positive")
    if args.accumulation_steps <= 0 or args.refresh <= 0:
        parser.error("accumulation and refresh must be positive")
    if args.warmup < 1 or args.repeats < 2:
        parser.error("use at least one warmup and two measured repeats")
    return args


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def measure_cuda(fn, warmup: int, repeats: int) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "median_ms": float(statistics.median(samples)),
        "mean_ms": float(statistics.fmean(samples)),
        "p05_ms": float(percentile(samples, 0.05)),
        "p95_ms": float(percentile(samples, 0.95)),
        "samples_ms": samples,
    }


def tensor_bytes(values) -> int:
    seen: set[tuple[str, int, int, int]] = set()
    total = 0
    for tensor in values:
        storage = tensor.untyped_storage()
        index = tensor.device.index if tensor.device.index is not None else -1
        key = (tensor.device.type, index, storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return int(total)


def self_test(XXT) -> dict[str, float]:
    torch.manual_seed(123)
    # Use tensor-core-friendly shapes; tiny matrices can exercise a different
    # Triton path than the formal R1 kernels.
    n, d = 1024, 128
    x = torch.randn(n, 4 * d, device="cuda", dtype=torch.bfloat16)
    dense_blocks = []
    for block in range(4):
        xb = x[:, block * d : (block + 1) * d].float()
        dense_blocks.append(xb.T @ xb / n)
    dense = torch.stack(dense_blocks)
    observed_diag = x.view(n, 4, d).square().float().mean(dim=0)
    diag_error = float((observed_diag - dense.diagonal(dim1=-2, dim2=-1)).abs().max())

    a = x.view(n, 4, d).permute(1, 2, 0).contiguous()
    out = torch.empty(4, d, d, device="cuda", dtype=torch.float32)
    XXT(a, out=out)
    out.mul_(1.0 / n)
    block_error = float((out - dense).abs().max())

    full_ref = x.float().T @ x.float() / n
    full_out = torch.empty(4 * d, 4 * d, device="cuda", dtype=torch.float32)
    XXT(x.T.contiguous(), out=full_out)
    full_out.mul_(1.0 / n)
    full_error = float((full_out - full_ref).abs().max())
    tolerance = 5e-2
    if max(diag_error, block_error, full_error) > tolerance:
        raise AssertionError(
            f"operator self-test failed: diag={diag_error} block4={block_error} full={full_error}"
        )
    return {
        "diag_max_abs_error": diag_error,
        "block4_max_abs_error": block_error,
        "dense_full_max_abs_error": full_error,
        "tolerance": tolerance,
    }


def benchmark_method(method: str, args: argparse.Namespace, XXT) -> dict:
    d = args.width
    four_d = 4 * d
    layers = args.layers
    n = args.tokens_per_microbatch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if method == "none":
        return {
            "method": method,
            "stats": {"median_ms": 0.0, "mean_ms": 0.0, "p05_ms": 0.0, "p95_ms": 0.0, "samples_ms": []},
            "inverse": {"median_ms": 0.0, "mean_ms": 0.0, "p05_ms": 0.0, "p95_ms": 0.0, "samples_ms": []},
            "apply": {"median_ms": 0.0, "mean_ms": 0.0, "p05_ms": 0.0, "p95_ms": 0.0, "samples_ms": []},
            "amortized_cproj_ms": 0.0,
            "k_state_bytes": 0,
            "peak_allocated_bytes": 0,
        }

    x = torch.randn(n, four_d, device="cuda", dtype=torch.bfloat16)
    gradients = torch.randn(layers, d, four_d, device="cuda", dtype=torch.float32)

    if method == "diag":
        accum = torch.zeros(layers, 4, d, device="cuda", dtype=torch.float32)
        covariance = torch.ones_like(accum)
        inverse = torch.ones_like(accum)

        @torch.compile
        def one_stat(x_2d: Tensor, target: Tensor) -> Tensor:
            target.add_(x_2d.view(x_2d.size(0), 4, d).square().mean(dim=0))
            return target

        def stats_fn():
            for _ in range(args.accumulation_steps):
                for layer in range(layers):
                    one_stat(x, accum[layer])

        def inverse_fn():
            ridge = covariance.mean(dim=-1) * 0.2 + 1e-8
            inverse.copy_((covariance + ridge.unsqueeze(-1)).reciprocal())

        def apply_fn():
            for layer in range(layers):
                gradients[layer].view(d, 4, d).mul_(inverse[layer].unsqueeze(0))

        state = [covariance, inverse]

    elif method == "block4":
        accum = torch.zeros(layers, 4, d, d, device="cuda", dtype=torch.float32)
        stat_tmp = torch.empty_like(accum)
        covariance = torch.eye(d, device="cuda", dtype=torch.float32).expand(layers, 4, d, d).clone()
        inverse = torch.empty_like(covariance)
        inverse.copy_(covariance)
        refresh_work = torch.empty(layers * 4, d, d, device="cuda", dtype=torch.float32)
        g_work = torch.empty(layers * 4, d, d, device="cuda", dtype=torch.float32)
        g_out = torch.empty_like(g_work)

        @torch.compile
        def one_stat(x_2d: Tensor, target: Tensor, tmp: Tensor) -> Tensor:
            a = x_2d.view(x_2d.size(0), 4, d).permute(1, 2, 0)
            XXT(a, out=tmp)
            tmp.mul_(1.0 / x_2d.size(0))
            target.add_(tmp)
            return target

        def stats_fn():
            for _ in range(args.accumulation_steps):
                for layer in range(layers):
                    one_stat(x, accum[layer], stat_tmp[layer])

        def inverse_fn():
            refresh_work.copy_(covariance.view(layers * 4, d, d))
            diagonal = refresh_work.diagonal(dim1=-2, dim2=-1)
            diagonal.add_((diagonal.mean(dim=-1) * 0.2 + 1e-8).unsqueeze(-1))
            factor, _ = torch.linalg.cholesky_ex(refresh_work, upper=False, check_errors=False)
            torch.cholesky_inverse(factor, upper=False, out=inverse.view(layers * 4, d, d))

        def apply_fn():
            g_work.view(layers, 4, d, d).copy_(
                gradients.view(layers, d, 4, d).permute(0, 2, 1, 3)
            )
            torch.bmm(g_work, inverse.view(layers * 4, d, d), out=g_out)
            gradients.view(layers, d, 4, d).copy_(g_out.view(layers, 4, d, d).permute(0, 2, 1, 3))

        state = [covariance, inverse]

    else:
        accum = torch.zeros(layers, four_d, four_d, device="cuda", dtype=torch.float32)
        stat_tmp = torch.empty_like(accum)
        covariance = torch.eye(four_d, device="cuda", dtype=torch.float32).expand(layers, four_d, four_d).clone()
        inverse = torch.empty_like(covariance)
        inverse.copy_(covariance)
        refresh_work = torch.empty_like(covariance)
        g_work = torch.empty_like(gradients)
        g_out = torch.empty_like(gradients)

        @torch.compile
        def one_stat(x_2d: Tensor, target: Tensor, tmp: Tensor) -> Tensor:
            XXT(x_2d.T, out=tmp)
            tmp.mul_(1.0 / x_2d.size(0))
            target.add_(tmp)
            return target

        def stats_fn():
            for _ in range(args.accumulation_steps):
                for layer in range(layers):
                    one_stat(x, accum[layer], stat_tmp[layer])

        def inverse_fn():
            refresh_work.copy_(covariance)
            diagonal = refresh_work.diagonal(dim1=-2, dim2=-1)
            diagonal.add_((diagonal.mean(dim=-1) * 0.2 + 1e-8).unsqueeze(-1))
            factor, _ = torch.linalg.cholesky_ex(refresh_work, upper=False, check_errors=False)
            torch.cholesky_inverse(factor, upper=False, out=inverse)

        def apply_fn():
            g_work.copy_(gradients)
            torch.bmm(g_work, inverse, out=g_out)
            gradients.copy_(g_out)

        state = [covariance, inverse]

    stats = measure_cuda(stats_fn, args.warmup, args.repeats)
    inverse_time = measure_cuda(inverse_fn, args.warmup, args.repeats)
    apply_time = measure_cuda(apply_fn, args.warmup, args.repeats)
    amortized = float(apply_time["median_ms"]) + (
        float(stats["median_ms"]) + float(inverse_time["median_ms"])
    ) / args.refresh
    result = {
        "method": method,
        "stats": stats,
        "inverse": inverse_time,
        "apply": apply_time,
        "amortized_cproj_ms": amortized,
        "k_state_bytes": tensor_bytes(state),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    del x, gradients, accum, covariance, inverse
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    official_repo = args.official_repo.resolve()
    if str(official_repo) not in sys.path:
        sys.path.insert(0, str(official_repo))
    from triton_kernels import XXT

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    audit = self_test(XXT)
    started = time.time()
    results = [benchmark_method(method, args, XXT) for method in args.methods]
    runtime = {
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
    }
    payload = {
        "protocol": "r1_perf_operator_components_v1",
        "arguments": vars(args) | {"official_repo": str(official_repo), "output_dir": str(args.output_dir.resolve())},
        "runtime": runtime,
        "self_test": audit,
        "results": results,
        "wall_elapsed_s": time.time() - started,
    }
    (args.output_dir / "operator_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (args.output_dir / "operator_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "method", "stats_median_ms", "inverse_median_ms", "apply_median_ms",
            "amortized_cproj_ms", "k_state_mib", "peak_allocated_mib",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "method": result["method"],
                "stats_median_ms": result["stats"]["median_ms"],
                "inverse_median_ms": result["inverse"]["median_ms"],
                "apply_median_ms": result["apply"]["median_ms"],
                "amortized_cproj_ms": result["amortized_cproj_ms"],
                "k_state_mib": result["k_state_bytes"] / 1024**2,
                "peak_allocated_mib": result["peak_allocated_bytes"] / 1024**2,
            })
    print(f"R1-PERF operator artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
