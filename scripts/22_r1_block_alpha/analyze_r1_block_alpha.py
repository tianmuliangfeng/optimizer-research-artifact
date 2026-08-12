"""Join the seed2026 block-alpha pilot with matched R1 diag/block4 endpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ALPHA = {"diag": 0.0, "alpha0": 0.0, "alpha0p25": 0.25, "alpha0p50": 0.5, "alpha0p75": 0.75, "block4": 1.0}
INTERIOR = ("alpha0", "alpha0p25", "alpha0p50", "alpha0p75")


def resolve_run(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "r1_summary.json").is_file():
        return path
    if path.name == "r1_summary.json":
        return path.parent
    raise RuntimeError(f"not an R1 run directory/summary: {path}")


def load_run(run_dir: Path, expected_method: str | None = None) -> dict[str, object]:
    summary = json.loads((run_dir / "r1_summary.json").read_text(encoding="utf-8"))
    method = str(summary["method"])
    if expected_method is not None and method != expected_method:
        raise RuntimeError(f"expected {expected_method} at {run_dir}, observed {method}")
    with (run_dir / "r1_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    vals = sorted(
        [(int(row["step"]), float(row["loss"])) for row in rows if row["event"] == "validation"],
        key=lambda item: item[0],
    )
    if not vals or vals[-1][0] != 6200:
        raise RuntimeError(f"{method} does not contain the formal step-6200 endpoint")
    if tuple(step for step, _ in vals) != tuple(range(0, 6201, 100)):
        raise RuntimeError(f"{method} validation grid is not exact 0:100:6200")
    area = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(vals, vals[1:]))
    return {
        "method": method,
        "alpha": ALPHA[method],
        "seed": int(summary["controlled_seed"]),
        "init_sha256": str(summary["init_sha256"]),
        "final_val_loss": vals[-1][1],
        "tail5_val_loss": sum(loss for _, loss in vals[-5:]) / 5,
        "normalized_val_auc": area / (vals[-1][0] - vals[0][0]),
        "run_dir": str(run_dir),
    }


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def correlation(left: list[float], right: list[float]) -> float:
    lmean, rmean = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((l - lmean) * (r - rmean) for l, r in zip(left, right))
    denominator = math.sqrt(sum((l - lmean) ** 2 for l in left) * sum((r - rmean) ** 2 for r in right))
    return numerator / denominator if denominator else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-batch", type=Path, required=True)
    parser.add_argument("--diag-run", type=Path, required=True)
    parser.add_argument("--block4-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    batch = args.alpha_batch.expanduser().resolve()
    manifest = json.loads((batch / "r1_manifest.json").read_text(encoding="utf-8"))
    by_method = {str(item["method"]): batch / str(item["run_name"]) for item in manifest.get("summaries", [])}
    missing = sorted(set(INTERIOR) - set(by_method))
    if missing:
        raise RuntimeError(f"alpha batch is missing completed pilot methods: {missing}")

    records = [load_run(resolve_run(args.diag_run), "diag")]
    records.extend(load_run(by_method[method], method) for method in INTERIOR)
    records.append(load_run(resolve_run(args.block4_run), "block4"))
    seeds = {item["seed"] for item in records}
    fingerprints = {item["init_sha256"] for item in records}
    if seeds != {2026}:
        raise RuntimeError(f"pilot endpoints must all be seed2026, observed {seeds}")
    if len(fingerprints) != 1:
        raise RuntimeError("diag/alpha/block4 endpoint initialization fingerprints differ")

    dose = [records[0], *records[2:]]  # efficient diag at alpha=0; omit duplicate dense alpha0.
    rho = correlation(
        average_ranks([float(item["alpha"]) for item in dose]),
        average_ranks([float(item["final_val_loss"]) for item in dose]),
    )
    diag, dense0, block4 = records[0], records[1], records[-1]
    final_equiv_delta = float(dense0["final_val_loss"]) - float(diag["final_val_loss"])
    tail_equiv_delta = float(dense0["tail5_val_loss"]) - float(diag["tail5_val_loss"])
    endpoint_delta = float(block4["final_val_loss"]) - float(diag["final_val_loss"])
    gates = {
        "dense_alpha0_final_equivalence": abs(final_equiv_delta) <= 0.001,
        "dense_alpha0_tail5_equivalence": abs(tail_equiv_delta) <= 0.001,
        "final_loss_directional_rho": rho >= 0.5,
        "block4_not_better_than_diag_at_primary_endpoint": endpoint_delta >= 0.0,
    }
    verdict = "expand_to_seeds_2024_2025" if all(gates.values()) else "stop_or_redesign_before_more_seeds"
    output = args.output_dir.expanduser().resolve() if args.output_dir else batch / "endpoint_analysis"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "block_alpha_endpoints.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = {
        "status": "valid",
        "primary_endpoint": "validation loss at step 6200",
        "records": records,
        "contrasts": {
            "dense_alpha0_minus_efficient_diag_final": final_equiv_delta,
            "dense_alpha0_minus_efficient_diag_tail5": tail_equiv_delta,
            "block4_minus_diag_final": endpoint_delta,
            "spearman_rho_alpha_vs_final_loss": rho,
        },
        "predeclared_gates": gates,
        "pilot_verdict": verdict,
    }
    (output / "block_alpha_pilot_verdict.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Analysis artifacts: {output}")


if __name__ == "__main__":
    main()
