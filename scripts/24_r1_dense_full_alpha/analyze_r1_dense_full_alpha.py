"""Analyze a completed seed2026 dense-full alpha batch under the frozen contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ALPHA = {
    "fullalpha0": 0.0,
    "fullalpha0p25": 0.25,
    "fullalpha0p50": 0.5,
    "fullalpha0p75": 0.75,
    "fullalpha1": 1.0,
}
BLOCK_METHOD = {0.0: "alpha0", 0.25: "alpha0p25", 0.5: "alpha0p50", 0.75: "alpha0p75", 1.0: "block4"}


def resolve_run(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if (resolved / "r1_summary.json").is_file():
        return resolved
    if resolved.name == "r1_summary.json" and resolved.is_file():
        return resolved.parent
    raise RuntimeError(f"not an R1 run directory/summary: {resolved}")


def load_run(run_dir: Path, expected_method: str) -> dict[str, object]:
    summary = json.loads((run_dir / "r1_summary.json").read_text(encoding="utf-8"))
    if summary.get("method") != expected_method:
        raise RuntimeError(f"expected {expected_method} at {run_dir}, observed {summary.get('method')}")
    with (run_dir / "r1_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    vals = sorted(
        [(int(row["step"]), float(row["loss"])) for row in rows if row["event"] == "validation"],
        key=lambda item: item[0],
    )
    if tuple(step for step, _ in vals) != tuple(range(0, 6201, 100)):
        raise RuntimeError(f"{expected_method} validation grid is not exact 0:100:6200")
    area = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(vals, vals[1:])) / 6200
    diagnostics_path = run_dir / "dense_full_alpha_diagnostics.csv"
    if not diagnostics_path.is_file():
        raise RuntimeError(f"missing dense-full alpha diagnostics: {diagnostics_path}")
    with diagnostics_path.open(encoding="utf-8", newline="") as handle:
        diagnostics = list(csv.DictReader(handle))
    failures = sum(int(float(row["cholesky_failures"])) for row in diagnostics if row["cholesky_failures"])
    return {
        "method": expected_method,
        "alpha": ALPHA[expected_method],
        "seed": int(summary["controlled_seed"]),
        "init_sha256": str(summary["init_sha256"]),
        "final_val_loss": vals[-1][1],
        "tail5_val_loss_mean": sum(loss for _, loss in vals[-5:]) / 5,
        "normalized_val_auc": area,
        "peak_memory_mib": float(summary["peak_memory_allocated_mib"]),
        "k_state_mib": float(summary["k_state_bytes"]) / (1024**2),
        "optimizer_state_mib": float(summary["optimizer_state_bytes"]) / (1024**2),
        "diagnostic_rows": len(diagnostics),
        "cholesky_failures": failures,
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-alpha-batch", type=Path, required=True)
    parser.add_argument("--diag-run", type=Path, required=True)
    parser.add_argument("--block-alpha-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    batch = args.full_alpha_batch.expanduser().resolve()
    manifest = json.loads((batch / "r1_manifest.json").read_text(encoding="utf-8"))
    by_method = {str(item["method"]): batch / str(item["run_name"]) for item in manifest.get("summaries", [])}
    missing = sorted(set(ALPHA) - set(by_method))
    if missing:
        raise RuntimeError(f"dense-full alpha batch is missing completed cells: {missing}")
    records = [load_run(by_method[method], method) for method in ALPHA]
    records.sort(key=lambda row: float(row["alpha"]))
    if {row["seed"] for row in records} != {2026}:
        raise RuntimeError("the frozen pilot requires seed2026")
    if len({row["init_sha256"] for row in records}) != 1:
        raise RuntimeError("dense-full alpha initialization fingerprints differ")

    diag_dir = resolve_run(args.diag_run)
    diag_summary = json.loads((diag_dir / "r1_summary.json").read_text(encoding="utf-8"))
    with (diag_dir / "r1_metrics.csv").open(encoding="utf-8", newline="") as handle:
        diag_rows = list(csv.DictReader(handle))
    diag_vals = sorted(
        [(int(row["step"]), float(row["loss"])) for row in diag_rows if row["event"] == "validation"],
        key=lambda item: item[0],
    )
    if tuple(step for step, _ in diag_vals) != tuple(range(0, 6201, 100)):
        raise RuntimeError("efficient diag validation grid is incomplete")
    diag_final = diag_vals[-1][1]
    diag_tail = sum(loss for _, loss in diag_vals[-5:]) / 5
    dense0 = records[0]
    final_equiv = float(dense0["final_val_loss"]) - diag_final
    tail_equiv = float(dense0["tail5_val_loss_mean"]) - diag_tail

    final_losses = [float(row["final_val_loss"]) for row in records]
    tail_losses = [float(row["tail5_val_loss_mean"]) for row in records]
    aucs = [float(row["normalized_val_auc"]) for row in records]
    adjacent = [right - left for left, right in zip(final_losses, final_losses[1:])]
    rho = correlation(average_ranks([float(row["alpha"]) for row in records]), average_ranks(final_losses))
    total_final = final_losses[-1] - final_losses[0]
    total_tail = tail_losses[-1] - tail_losses[0]
    total_auc = aucs[-1] - aucs[0]
    monotone_signal = (
        sum(delta >= 0 for delta in adjacent) >= 3
        and rho >= 0.8
        and total_final >= 0.001
        and total_tail >= 0
        and total_auc >= 0
    )

    topology: list[dict[str, object]] = []
    topology_material = False
    if args.block_alpha_summary is not None:
        with args.block_alpha_summary.expanduser().resolve().open(encoding="utf-8", newline="") as handle:
            block_rows = {row["method"]: row for row in csv.DictReader(handle)}
        for record in records:
            alpha = float(record["alpha"])
            block_method = BLOCK_METHOD[alpha]
            if block_method not in block_rows:
                raise RuntimeError(f"block-alpha summary is missing {block_method}")
            block = block_rows[block_method]
            item = {
                "alpha": alpha,
                "full_method": record["method"],
                "block_method": block_method,
                "final_delta_full_minus_block": float(record["final_val_loss"]) - float(block["final_val_loss"]),
                "tail5_delta_full_minus_block": float(record["tail5_val_loss_mean"]) - float(block["tail5_val_loss_mean"]),
                "auc_delta_full_minus_block": float(record["normalized_val_auc"]) - float(block["normalized_val_auc"]),
            }
            signs_agree = (
                math.copysign(1, float(item["final_delta_full_minus_block"]))
                == math.copysign(1, float(item["tail5_delta_full_minus_block"]))
                == math.copysign(1, float(item["auc_delta_full_minus_block"]))
            )
            item["endpoint_signs_agree"] = signs_agree
            item["material_abs_final_ge_0p002"] = abs(float(item["final_delta_full_minus_block"])) >= 0.002
            topology_material = topology_material or bool(item["material_abs_final_ge_0p002"] and signs_agree)
            topology.append(item)

    implementation_pass = (
        abs(final_equiv) <= 0.001
        and abs(tail_equiv) <= 0.001
        and all(int(row["cholesky_failures"]) == 0 and int(row["diagnostic_rows"]) >= 14 for row in records)
        and int(diag_summary["controlled_seed"]) == 2026
    )
    expansion_candidate = implementation_pass and (
        (monotone_signal and total_final >= 0.002) or topology_material
    )
    verdict = "candidate_for_confirmatory_seed_review" if expansion_candidate else "stop_or_redesign_before_more_seeds"
    output = args.output_dir.expanduser().resolve() if args.output_dir else batch / "dense_full_alpha_analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "dense_full_alpha_endpoints.csv", records)
    if topology:
        write_csv(output / "matched_topology_contrasts.csv", topology)
    result = {
        "status": "valid",
        "primary_endpoint": "validation loss at step 6200",
        "dense_alpha0_minus_efficient_diag_final": final_equiv,
        "dense_alpha0_minus_efficient_diag_tail5": tail_equiv,
        "spearman_rho_alpha_vs_final_loss": rho,
        "adjacent_final_deltas": adjacent,
        "alpha1_minus_alpha0": {"final": total_final, "tail5": total_tail, "auc": total_auc},
        "owt_like_monotone_signal": monotone_signal,
        "topology_material": topology_material if topology else None,
        "implementation_gates_pass": implementation_pass,
        "pilot_verdict": verdict,
        "timing_usable": False,
        "records": records,
    }
    (output / "dense_full_alpha_pilot_verdict.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Analysis artifacts: {output}")


if __name__ == "__main__":
    main()

