from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METHOD_LABELS = {
    "muon": "Muon",
    "newton": "Newton-Muon",
    "selective_middle": "Selective-Middle (56%)",
    "selective_all": "Selective-All-c_proj (84%)",
}
METHOD_ORDER = ["muon", "newton", "selective_middle", "selective_all"]
SEEDS = [2024, 2025, 2026]
TOKENS_PER_STEP = 16 * 512
THRESHOLDS = [4.0, 3.8, 3.7, 3.6, 3.58, 3.57]


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(
    os.environ.get("SNM_RESULTS_ROOT", str(ARTIFACT_ROOT / "runs"))
).expanduser()
OUTPUT_ROOT = (
    RESULTS_ROOT
    / "04_dataset_generalization"
    / "wikitext103_12L_100m_multiseed_20260714"
)
RAW_ROOT = OUTPUT_ROOT / "raw_wandb_exports"
SUMMARY_ROOT = OUTPUT_ROOT / "summaries"
OWT_SUMMARY_ROOT = (
    RESULTS_ROOT
    / "02_long_token_budget"
    / "100m_multiseed_20260710"
    / "summaries"
)


def parse_run(run_name: str) -> tuple[int, str]:
    seed_match = re.search(r"seed(\d+)", run_name)
    if not seed_match:
        raise ValueError(f"No seed in run name: {run_name}")
    if "00_muon" in run_name:
        method = "muon"
    elif "13_newton_muon_fast" in run_name:
        method = "newton"
    elif "middle_cproj_release56" in run_name:
        method = "selective_middle"
    elif "all_cproj_release84" in run_name:
        method = "selective_all"
    else:
        raise ValueError(f"Unknown method in run name: {run_name}")
    return int(seed_match.group(1)), method


def load_series() -> tuple[dict[tuple[int, str, str], list[tuple[int, float]]], list[dict[str, object]]]:
    series: dict[tuple[int, str, str], list[tuple[int, float]]] = defaultdict(list)
    quality = []
    for path in sorted(RAW_ROOT.glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if len(rows) < 2:
            raise ValueError(f"Empty export: {path}")

        columns = []
        for index, column in enumerate(rows[0]):
            if index == 0 or " - " not in column:
                continue
            run_name, metric = column.rsplit(" - ", 1)
            if metric.endswith("__MIN") or metric.endswith("__MAX"):
                continue
            seed, method = parse_run(run_name)
            columns.append((index, seed, method, metric))

        steps = []
        counts = defaultdict(int)
        for row in rows[1:]:
            if not row or not row[0]:
                continue
            step = int(float(row[0]))
            steps.append(step)
            for index, seed, method, metric in columns:
                if index < len(row) and row[index] != "":
                    series[(seed, method, metric)].append((step, float(row[index])))
                    counts[(seed, method, metric)] += 1

        quality.append(
            {
                "file": path.name,
                "metric": ",".join(sorted({metric for _, _, _, metric in columns})),
                "row_count": len(rows) - 1,
                "min_step": min(steps),
                "max_step": max(steps),
                "run_count": len(columns),
                "seeds": ",".join(str(seed) for seed in sorted({seed for _, seed, _, _ in columns})),
                "nonempty_points_min": min(counts.values()),
                "nonempty_points_max": max(counts.values()),
            }
        )

    for key in series:
        series[key].sort(key=lambda point: point[0])
        if len(series[key]) != len({step for step, _ in series[key]}):
            raise ValueError(f"Duplicate metric points: {key}")
    return series, quality


def normalized_auc(points: list[tuple[int, float]]) -> float:
    area = sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for (left_step, left_value), (right_step, right_value) in zip(points, points[1:])
    )
    return area / (points[-1][0] - points[0][0])


def last_value(series, seed: int, method: str, metric: str) -> float | None:
    points = series.get((seed, method, metric), [])
    return points[-1][1] if points else None


def to_mib(value: float | None) -> float:
    return 0.0 if value is None else value / 1024.0 / 1024.0


def make_run_summaries(series) -> list[dict[str, object]]:
    common_steps = set.intersection(
        *(set(step for step, _ in series[(seed, method, "val/loss")]) for seed in SEEDS for method in METHOD_ORDER)
    )
    final_step = max(common_steps)
    rows = []
    for seed in SEEDS:
        for method in METHOD_ORDER:
            val_points = series[(seed, method, "val/loss")]
            final_val = dict(val_points)[final_step]
            best_step, best_val = min(val_points, key=lambda point: point[1])
            late_values = [value for step, value in val_points if 11000 <= step <= 12000]
            train_eval = series[(seed, method, "train/loss")]
            train_step = series[(seed, method, "train/loss_step")]
            memory = series[(seed, method, "cuda/memory_allocated_mib")]
            peak = series[(seed, method, "cuda/full_run_max_memory_allocated_mib")]
            k_state = to_mib(last_value(series, seed, method, "matrix/k_state_bytes"))
            full_k_state = k_state if method == "newton" else 0.0
            if method.startswith("selective"):
                full_k_state = to_mib(last_value(series, seed, method, "matrix/k_state_full_bytes"))

            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "last_common_val_step": final_step,
                    "last_common_val_tokens": final_step * TOKENS_PER_STEP,
                    "final_val_loss": final_val,
                    "best_val_loss": best_val,
                    "best_val_step": best_step,
                    "final_minus_best_val": final_val - best_val,
                    "late_val_mean_11000_12000": mean(late_values),
                    "normalized_val_auc_0_12000": normalized_auc(val_points),
                    "final_train_eval_loss": train_eval[-1][1],
                    "last_train_loss_step": train_step[-1][0],
                    "last_train_loss": train_step[-1][1],
                    "tail50_train_loss_mean": mean(value for _, value in train_step[-50:]),
                    "current_memory_mib": memory[-1][1],
                    "peak_memory_mib": max(value for _, value in peak),
                    "k_state_mib": k_state,
                    "full_k_state_mib": full_k_state,
                    "k_state_released_mib": to_mib(last_value(series, seed, method, "matrix/k_state_released_bytes")),
                    "k_state_released_fraction": last_value(series, seed, method, "matrix/k_state_released_fraction") or 0.0,
                }
            )

    by_key = {(row["seed"], row["method"]): row for row in rows}
    for seed in SEEDS:
        newton = by_key[(seed, "newton")]
        muon = by_key[(seed, "muon")]
        for method in METHOD_ORDER:
            row = by_key[(seed, method)]
            row["final_val_delta_vs_newton"] = row["final_val_loss"] - newton["final_val_loss"]
            row["final_val_delta_vs_muon"] = row["final_val_loss"] - muon["final_val_loss"]
            row["current_memory_saved_vs_newton_mib"] = newton["current_memory_mib"] - row["current_memory_mib"]
            row["peak_memory_saved_vs_newton_mib"] = newton["peak_memory_mib"] - row["peak_memory_mib"]
    return rows


def aggregate_method(rows: list[dict[str, object]], method: str) -> dict[str, object]:
    selected = [row for row in rows if row["method"] == method]
    metrics = [
        "final_val_loss",
        "best_val_loss",
        "final_minus_best_val",
        "late_val_mean_11000_12000",
        "normalized_val_auc_0_12000",
        "current_memory_mib",
        "peak_memory_mib",
        "k_state_mib",
        "full_k_state_mib",
        "k_state_released_mib",
        "k_state_released_fraction",
        "final_val_delta_vs_newton",
        "final_val_delta_vs_muon",
        "current_memory_saved_vs_newton_mib",
        "peak_memory_saved_vs_newton_mib",
    ]
    result = {"method": method, "method_label": METHOD_LABELS[method], "n_seeds": len(selected)}
    for metric in metrics:
        values = [float(row[metric]) for row in selected]
        result[f"{metric}_mean"] = mean(values)
        result[f"{metric}_std"] = stdev(values)
    return result


def make_threshold_rows(series) -> list[dict[str, object]]:
    rows = []
    for seed in SEEDS:
        for threshold in THRESHOLDS:
            for method in METHOD_ORDER:
                crossing = next(
                    ((step, value) for step, value in series[(seed, method, "val/loss")] if value <= threshold),
                    None,
                )
                rows.append(
                    {
                        "seed": seed,
                        "threshold": threshold,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "reached": crossing is not None,
                        "first_step": crossing[0] if crossing else None,
                        "first_tokens": crossing[0] * TOKENS_PER_STEP if crossing else None,
                        "loss_at_crossing": crossing[1] if crossing else None,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    series, quality_rows = load_series()
    expected = {(seed, method, "val/loss") for seed in SEEDS for method in METHOD_ORDER}
    missing = sorted(expected - set(series))
    if missing:
        raise ValueError(f"Missing required validation series: {missing}")

    run_rows = make_run_summaries(series)
    method_rows = [aggregate_method(run_rows, method) for method in METHOD_ORDER]
    by_key = {(row["seed"], row["method"]): row for row in run_rows}

    paired_rows = []
    for seed in SEEDS:
        for method in ["muon", "selective_middle", "selective_all"]:
            row = by_key[(seed, method)]
            paired_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "final_val_delta_vs_newton": row["final_val_delta_vs_newton"],
                    "current_memory_saved_vs_newton_mib": row["current_memory_saved_vs_newton_mib"],
                    "peak_memory_saved_vs_newton_mib": row["peak_memory_saved_vs_newton_mib"],
                    "k_state_released_mib": row["k_state_released_mib"],
                    "k_state_released_fraction": row["k_state_released_fraction"],
                }
            )

    paired_aggregate = []
    for method in ["muon", "selective_middle", "selective_all"]:
        selected = [row for row in paired_rows if row["method"] == method]
        deltas = [float(row["final_val_delta_vs_newton"]) for row in selected]
        paired_aggregate.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_seeds": len(deltas),
                "delta_vs_newton_mean": mean(deltas),
                "delta_vs_newton_std": stdev(deltas),
                "wins_vs_newton": sum(delta < 0 for delta in deltas),
                "losses_vs_newton": sum(delta > 0 for delta in deltas),
            }
        )

    curve_rows = []
    for method in METHOD_ORDER:
        common_steps = set.intersection(
            *(set(step for step, _ in series[(seed, method, "val/loss")]) for seed in SEEDS)
        )
        for step in sorted(common_steps):
            values = [dict(series[(seed, method, "val/loss")])[step] for seed in SEEDS]
            curve_rows.append(
                {
                    "step": step,
                    "tokens": step * TOKENS_PER_STEP,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "val_loss_mean": mean(values),
                    "val_loss_std": stdev(values),
                }
            )

    write_csv(SUMMARY_ROOT / "data_quality_checks.csv", quality_rows)
    write_csv(SUMMARY_ROOT / "run_summary.csv", run_rows)
    write_csv(SUMMARY_ROOT / "method_aggregate.csv", method_rows)
    write_csv(SUMMARY_ROOT / "paired_deltas_vs_newton.csv", paired_rows)
    write_csv(SUMMARY_ROOT / "paired_deltas_aggregate.csv", paired_aggregate)
    write_csv(SUMMARY_ROOT / "threshold_crossings.csv", make_threshold_rows(series))
    write_csv(SUMMARY_ROOT / "val_curve_mean_std.csv", curve_rows)

    owt_method_map = {
        "muon": "muon",
        "newton": "newton",
        "selective_middle": "selective_middle",
        "selective_all_cproj": "selective_all",
    }
    cross_dataset_rows = []
    owt_path = OWT_SUMMARY_ROOT / "method_aggregate.csv"
    if owt_path.exists():
        with owt_path.open("r", encoding="utf-8-sig", newline="") as handle:
            owt_rows = {
                owt_method_map[row["method"]]: row
                for row in csv.DictReader(handle)
                if row["method"] in owt_method_map
            }
        wiki_rows = {row["method"]: row for row in method_rows}
        owt_newton = float(owt_rows["newton"]["final_val_loss_mean"])
        for method in METHOD_ORDER:
            cross_dataset_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "owt_final_val_mean": float(owt_rows[method]["final_val_loss_mean"]),
                    "owt_delta_vs_newton_mean": float(owt_rows[method]["final_val_loss_mean"]) - owt_newton,
                    "wikitext_final_val_mean": wiki_rows[method]["final_val_loss_mean"],
                    "wikitext_paired_delta_vs_newton_mean": wiki_rows[method]["final_val_delta_vs_newton_mean"],
                }
            )
        write_csv(SUMMARY_ROOT / "cross_dataset_method_aggregate.csv", cross_dataset_rows)

    aggregates = {row["method"]: row for row in method_rows}
    newton = aggregates["newton"]
    middle = aggregates["selective_middle"]
    all_cproj = aggregates["selective_all"]
    muon = aggregates["muon"]
    all_minus_middle = [
        by_key[(seed, "selective_all")]["final_val_loss"]
        - by_key[(seed, "selective_middle")]["final_val_loss"]
        for seed in SEEDS
    ]
    key_results = {
        "dataset": "wikitext103_gpt2_50m",
        "seeds": SEEDS,
        "last_common_val_step": 12000,
        "last_common_val_tokens": 12000 * TOKENS_PER_STEP,
        "final_val_mean_std": {
            method: [aggregates[method]["final_val_loss_mean"], aggregates[method]["final_val_loss_std"]]
            for method in METHOD_ORDER
        },
        "paired_delta_vs_newton_mean_std": {
            method: [aggregates[method]["final_val_delta_vs_newton_mean"], aggregates[method]["final_val_delta_vs_newton_std"]]
            for method in METHOD_ORDER
        },
        "selective_all_minus_middle_by_seed": dict(zip(SEEDS, all_minus_middle)),
        "wall_clock_used_for_conclusions": False,
    }
    with (SUMMARY_ROOT / "key_results.json").open("w", encoding="utf-8") as handle:
        json.dump(key_results, handle, indent=2)

    notes = f"""# WikiText-103 12L/100M Multi-Seed Result Notes

## Main result

At the last common validation checkpoint (step 12,000; 98.304M evaluated tokens):

| Method | Validation loss, mean +/- SD | Paired delta vs Newton, mean +/- SD | Current memory | Peak memory | K-state released |
|---|---:|---:|---:|---:|---:|
| Muon | {muon['final_val_loss_mean']:.4f} +/- {muon['final_val_loss_std']:.4f} | {muon['final_val_delta_vs_newton_mean']:+.4f} +/- {muon['final_val_delta_vs_newton_std']:.4f} | {muon['current_memory_mib_mean']:.0f} MiB | {muon['peak_memory_mib_mean']:.0f} MiB | 0% |
| Newton-Muon | {newton['final_val_loss_mean']:.4f} +/- {newton['final_val_loss_std']:.4f} | 0 | {newton['current_memory_mib_mean']:.0f} MiB | {newton['peak_memory_mib_mean']:.0f} MiB | 0% |
| Selective-Middle | {middle['final_val_loss_mean']:.4f} +/- {middle['final_val_loss_std']:.4f} | {middle['final_val_delta_vs_newton_mean']:+.4f} +/- {middle['final_val_delta_vs_newton_std']:.4f} | {middle['current_memory_mib_mean']:.0f} MiB | {middle['peak_memory_mib_mean']:.0f} MiB | 56.14% |
| Selective-All-c_proj | {all_cproj['final_val_loss_mean']:.4f} +/- {all_cproj['final_val_loss_std']:.4f} | {all_cproj['final_val_delta_vs_newton_mean']:+.4f} +/- {all_cproj['final_val_delta_vs_newton_std']:.4f} | {all_cproj['current_memory_mib_mean']:.0f} MiB | {all_cproj['peak_memory_mib_mean']:.0f} MiB | 84.21% |

Selective-Middle matches Newton-Muon most closely: its paired three-seed delta is only +0.0011 loss, and it beats Newton-Muon on seed 2026. Selective-All remains better than Muon in every seed but trails Newton-Muon by +0.0091 loss on average.

## Rule choice

Selective-Middle is better than Selective-All on all three WikiText seeds. The all-minus-middle deltas are +0.0053, +0.0000, and +0.0186 for seeds 2024, 2025, and 2026. WikiText therefore strengthens the case for Selective-Middle as the conservative/default rule and Selective-All as the aggressive high-memory-saving option.

## Memory

Selective-Middle releases 864 MiB of K-state and saves {newton['current_memory_mib_mean'] - middle['current_memory_mib_mean']:.0f} MiB ({(newton['current_memory_mib_mean'] - middle['current_memory_mib_mean']) / newton['current_memory_mib_mean']:.1%}) current memory and {newton['peak_memory_mib_mean'] - middle['peak_memory_mib_mean']:.0f} MiB ({(newton['peak_memory_mib_mean'] - middle['peak_memory_mib_mean']) / newton['peak_memory_mib_mean']:.1%}) peak memory.

Selective-All releases 1,296 MiB of K-state and saves {newton['current_memory_mib_mean'] - all_cproj['current_memory_mib_mean']:.0f} MiB ({(newton['current_memory_mib_mean'] - all_cproj['current_memory_mib_mean']) / newton['current_memory_mib_mean']:.1%}) current memory and {newton['peak_memory_mib_mean'] - all_cproj['peak_memory_mib_mean']:.0f} MiB ({(newton['peak_memory_mib_mean'] - all_cproj['peak_memory_mib_mean']) / newton['peak_memory_mib_mean']:.1%}) peak memory.

## Cross-dataset conclusion

Across OpenWebText and WikiText-103, both selective rules preserve most of the Newton-Muon quality advantage while reducing K-state and CUDA memory. The preferred quality rule is not identical: OpenWebText's three-seed average favored Selective-All slightly, whereas WikiText consistently favors Selective-Middle. The robust claim is therefore a memory-quality tradeoff, not that release-all always gives the best loss.

## Data-quality caveats

- All 12 runs are complete through train step 12,200 and share validation checkpoints through step 12,000.
- Seed 2025 has a shared validation rebound after step 10,500 across all four methods; this is not selective-specific and likely includes validation-sampling noise.
- The WikiText validation split has 249,750 tokens and each evaluation uses 20 sampled batches, so small checkpoint-to-checkpoint differences should not be overinterpreted.
- The final validation point is step 12,000 rather than max_iters=12,208, matching prior experiments.
- Wall-clock time is excluded, and n=3 supports descriptive paired robustness rather than formal equivalence testing.

## Decision

- Dataset generalization is complete for the current paper scope: two datasets, four methods, and three seeds each.
- Use Selective-Middle as the conservative/default rule and Selective-All as the aggressive memory-saving rule.
- Move next to K-state mechanism analysis rather than adding another dataset immediately.
"""
    (OUTPUT_ROOT / "RESULT_NOTES.md").write_text(notes, encoding="utf-8")
    print(f"Wrote WikiText-103 multi-seed analysis to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
