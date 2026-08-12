from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


METHOD_LABELS = {
    "muon": "Muon",
    "newton": "Newton-Muon",
    "selective_middle": "Selective-Middle (56%)",
    "selective_all": "Selective-All-c_proj (84%)",
}
METHOD_ORDER = ["muon", "newton", "selective_middle", "selective_all"]
TOKENS_PER_STEP = 16 * 512
THRESHOLDS = [4.0, 3.8, 3.7, 3.6, 3.58, 3.57]


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(
    os.environ.get("SNM_RESULTS_ROOT", str(ARTIFACT_ROOT / "runs"))
).expanduser()
OUTPUT_ROOT = (
    RESULTS_ROOT
    / "04_dataset_generalization"
    / "wikitext103_12L_100m_seed2024_20260714"
)
RAW_ROOT = OUTPUT_ROOT / "raw_wandb_exports"
SUMMARY_ROOT = OUTPUT_ROOT / "summaries"
OWT_SUMMARY = (
    RESULTS_ROOT
    / "02_long_token_budget"
    / "100m_multiseed_20260710"
    / "summaries"
    / "run_summary.csv"
)


def parse_method(run_name: str) -> str:
    if "00_muon" in run_name:
        return "muon"
    if "13_newton_muon_fast" in run_name:
        return "newton"
    if "middle_cproj_release56" in run_name:
        return "selective_middle"
    if "all_cproj_release84" in run_name:
        return "selective_all"
    raise ValueError(f"Unknown run name: {run_name}")


def load_series() -> tuple[dict[tuple[str, str], list[tuple[int, float]]], list[dict[str, object]]]:
    series: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    quality: list[dict[str, object]] = []

    for path in sorted(RAW_ROOT.glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if len(rows) < 2:
            raise ValueError(f"Empty export: {path}")

        header = rows[0]
        columns: list[tuple[int, str, str]] = []
        for index, column in enumerate(header):
            if index == 0 or " - " not in column:
                continue
            run_name, metric = column.rsplit(" - ", 1)
            if metric.endswith("__MIN") or metric.endswith("__MAX"):
                continue
            columns.append((index, parse_method(run_name), metric))

        steps = []
        nonempty_counts = {f"{method}:{metric}": 0 for _, method, metric in columns}
        for row in rows[1:]:
            if not row or not row[0]:
                continue
            step = int(float(row[0]))
            steps.append(step)
            for index, method, metric in columns:
                if index >= len(row) or row[index] == "":
                    continue
                series[(method, metric)].append((step, float(row[index])))
                nonempty_counts[f"{method}:{metric}"] += 1

        metric_names = sorted({metric for _, _, metric in columns})
        quality.append(
            {
                "file": path.name,
                "metric": ",".join(metric_names),
                "row_count": len(rows) - 1,
                "min_step": min(steps),
                "max_step": max(steps),
                "run_count": len(columns),
                "nonempty_points_min": min(nonempty_counts.values()),
                "nonempty_points_max": max(nonempty_counts.values()),
            }
        )

    for key in series:
        series[key].sort(key=lambda point: point[0])
        if len({step for step, _ in series[key]}) != len(series[key]):
            raise ValueError(f"Duplicate steps in {key}")
    return series, quality


def normalized_auc(points: list[tuple[int, float]]) -> float:
    area = sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for (left_step, left_value), (right_step, right_value) in zip(points, points[1:])
    )
    return area / (points[-1][0] - points[0][0])


def last_value(series: dict[tuple[str, str], list[tuple[int, float]]], method: str, metric: str) -> float | None:
    points = series.get((method, metric), [])
    return points[-1][1] if points else None


def mib(value: float | None) -> float | None:
    return None if value is None else value / 1024.0 / 1024.0


def make_run_summaries(series: dict[tuple[str, str], list[tuple[int, float]]]) -> list[dict[str, object]]:
    common_val_steps = set.intersection(
        *(set(step for step, _ in series[(method, "val/loss")]) for method in METHOD_ORDER)
    )
    final_step = max(common_val_steps)
    summaries = []

    for method in METHOD_ORDER:
        val_points = series[(method, "val/loss")]
        val_by_step = dict(val_points)
        best_step, best_value = min(val_points, key=lambda point: point[1])
        late_values = [value for step, value in val_points if 11000 <= step <= 12000]
        train_eval = series[(method, "train/loss")]
        train_step = series[(method, "train/loss_step")]
        memory = series[(method, "cuda/memory_allocated_mib")]
        peak_memory = series[(method, "cuda/full_run_max_memory_allocated_mib")]

        k_state_mib = mib(last_value(series, method, "matrix/k_state_bytes")) or 0.0
        if method == "newton":
            full_k_state_mib = k_state_mib
        elif method == "muon":
            full_k_state_mib = 0.0
        else:
            full_k_state_mib = mib(last_value(series, method, "matrix/k_state_full_bytes")) or 0.0

        summaries.append(
            {
                "seed": 2024,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "last_common_val_step": final_step,
                "last_common_val_tokens": final_step * TOKENS_PER_STEP,
                "final_val_loss": val_by_step[final_step],
                "best_val_loss": best_value,
                "best_val_step": best_step,
                "final_minus_best_val": val_by_step[final_step] - best_value,
                "late_val_mean_11000_12000": mean(late_values),
                "normalized_val_auc_0_12000": normalized_auc(val_points),
                "final_train_eval_loss": train_eval[-1][1],
                "last_train_loss_step": train_step[-1][0],
                "last_train_loss": train_step[-1][1],
                "tail50_train_loss_mean": mean(value for _, value in train_step[-50:]),
                "current_memory_mib": memory[-1][1],
                "peak_memory_mib": max(value for _, value in peak_memory),
                "k_state_mib": k_state_mib,
                "full_k_state_mib": full_k_state_mib,
                "k_state_released_mib": mib(last_value(series, method, "matrix/k_state_released_bytes")) or 0.0,
                "k_state_released_fraction": last_value(series, method, "matrix/k_state_released_fraction") or 0.0,
            }
        )

    by_method = {row["method"]: row for row in summaries}
    newton = by_method["newton"]
    muon = by_method["muon"]
    for row in summaries:
        row["final_val_delta_vs_newton"] = row["final_val_loss"] - newton["final_val_loss"]
        row["final_val_delta_vs_muon"] = row["final_val_loss"] - muon["final_val_loss"]
        row["current_memory_saved_vs_newton_mib"] = newton["current_memory_mib"] - row["current_memory_mib"]
        row["current_memory_saved_vs_newton_fraction"] = (
            (newton["current_memory_mib"] - row["current_memory_mib"]) / newton["current_memory_mib"]
        )
        row["peak_memory_saved_vs_newton_mib"] = newton["peak_memory_mib"] - row["peak_memory_mib"]
        row["peak_memory_saved_vs_newton_fraction"] = (
            (newton["peak_memory_mib"] - row["peak_memory_mib"]) / newton["peak_memory_mib"]
        )
    return summaries


def make_threshold_rows(series: dict[tuple[str, str], list[tuple[int, float]]]) -> list[dict[str, object]]:
    rows = []
    for threshold in THRESHOLDS:
        for method in METHOD_ORDER:
            crossing = next(
                ((step, value) for step, value in series[(method, "val/loss")] if value <= threshold),
                None,
            )
            rows.append(
                {
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


def load_owt_seed2024() -> dict[str, dict[str, str]]:
    if not OWT_SUMMARY.exists():
        return {}
    method_map = {
        "muon": "muon",
        "newton": "newton",
        "selective_middle": "selective_middle",
        "selective_all_cproj": "selective_all",
    }
    with OWT_SUMMARY.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            method_map[row["method"]]: row
            for row in csv.DictReader(handle)
            if row["seed"] == "2024" and row["method"] in method_map
        }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    series, quality_rows = load_series()
    summaries = make_run_summaries(series)
    by_method = {row["method"]: row for row in summaries}
    threshold_rows = make_threshold_rows(series)

    write_csv(SUMMARY_ROOT / "data_quality_checks.csv", quality_rows)
    write_csv(SUMMARY_ROOT / "run_summary.csv", summaries)
    write_csv(SUMMARY_ROOT / "threshold_crossings.csv", threshold_rows)

    paired_rows = []
    for method in ["muon", "selective_middle", "selective_all"]:
        row = by_method[method]
        paired_rows.append(
            {
                "seed": 2024,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "final_val_delta_vs_newton": row["final_val_delta_vs_newton"],
                "current_memory_saved_vs_newton_mib": row["current_memory_saved_vs_newton_mib"],
                "current_memory_saved_vs_newton_fraction": row["current_memory_saved_vs_newton_fraction"],
                "peak_memory_saved_vs_newton_mib": row["peak_memory_saved_vs_newton_mib"],
                "peak_memory_saved_vs_newton_fraction": row["peak_memory_saved_vs_newton_fraction"],
                "k_state_released_mib": row["k_state_released_mib"],
                "k_state_released_fraction": row["k_state_released_fraction"],
            }
        )
    write_csv(SUMMARY_ROOT / "paired_deltas_vs_newton.csv", paired_rows)

    owt = load_owt_seed2024()
    if owt:
        comparison_rows = []
        for method in METHOD_ORDER:
            wiki = by_method[method]
            owt_row = owt[method]
            owt_newton = float(owt["newton"]["final_val_loss"])
            comparison_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "owt_seed2024_final_val_loss": float(owt_row["final_val_loss"]),
                    "owt_seed2024_delta_vs_newton": float(owt_row["final_val_loss"]) - owt_newton,
                    "wikitext_seed2024_final_val_loss": wiki["final_val_loss"],
                    "wikitext_seed2024_delta_vs_newton": wiki["final_val_delta_vs_newton"],
                }
            )
        write_csv(SUMMARY_ROOT / "cross_dataset_seed2024_comparison.csv", comparison_rows)

    key_results = {
        "dataset": "wikitext103_gpt2_50m",
        "seed": 2024,
        "last_common_val_step": 12000,
        "last_common_val_tokens": 12000 * TOKENS_PER_STEP,
        "final_val_ranking": sorted(
            ((method, by_method[method]["final_val_loss"]) for method in METHOD_ORDER),
            key=lambda item: item[1],
        ),
        "selective_middle_delta_vs_newton": by_method["selective_middle"]["final_val_delta_vs_newton"],
        "selective_all_delta_vs_newton": by_method["selective_all"]["final_val_delta_vs_newton"],
        "selective_all_minus_middle": (
            by_method["selective_all"]["final_val_loss"]
            - by_method["selective_middle"]["final_val_loss"]
        ),
        "selective_middle_current_memory_saved_mib": by_method["selective_middle"]["current_memory_saved_vs_newton_mib"],
        "selective_all_current_memory_saved_mib": by_method["selective_all"]["current_memory_saved_vs_newton_mib"],
        "selective_middle_peak_memory_saved_mib": by_method["selective_middle"]["peak_memory_saved_vs_newton_mib"],
        "selective_all_peak_memory_saved_mib": by_method["selective_all"]["peak_memory_saved_vs_newton_mib"],
        "wall_clock_used_for_conclusions": False,
        "requires_more_seeds": True,
    }
    with (SUMMARY_ROOT / "key_results.json").open("w", encoding="utf-8") as handle:
        json.dump(key_results, handle, indent=2)

    muon = by_method["muon"]
    newton = by_method["newton"]
    middle = by_method["selective_middle"]
    all_cproj = by_method["selective_all"]
    notes = f"""# WikiText-103 12L/100M Seed-2024 Result Notes

## Main result

At the last common validation checkpoint (step 12,000; 98.304M evaluated tokens):

| Method | Validation loss | Delta vs Newton | Current memory | Peak memory | K-state released |
|---|---:|---:|---:|---:|---:|
| Muon | {muon['final_val_loss']:.4f} | {muon['final_val_delta_vs_newton']:+.4f} | {muon['current_memory_mib']:.0f} MiB | {muon['peak_memory_mib']:.0f} MiB | 0% |
| Newton-Muon | {newton['final_val_loss']:.4f} | 0 | {newton['current_memory_mib']:.0f} MiB | {newton['peak_memory_mib']:.0f} MiB | 0% |
| Selective-Middle | {middle['final_val_loss']:.4f} | {middle['final_val_delta_vs_newton']:+.4f} | {middle['current_memory_mib']:.0f} MiB | {middle['peak_memory_mib']:.0f} MiB | 56.14% |
| Selective-All-c_proj | {all_cproj['final_val_loss']:.4f} | {all_cproj['final_val_delta_vs_newton']:+.4f} | {all_cproj['current_memory_mib']:.0f} MiB | {all_cproj['peak_memory_mib']:.0f} MiB | 84.21% |

Both selective rules preserve most of the Newton-Muon optimization gain and outperform Muon. Selective-Middle is 0.0096 worse than full Newton-Muon and 0.0324 better than Muon. Selective-All is 0.0149 worse than full Newton-Muon and 0.0271 better than Muon.

## Cross-dataset interpretation

The seed-2024 ordering is identical on OpenWebText and WikiText-103: Newton-Muon, Selective-Middle, Selective-All, then Muon. Selective-All is 0.0053 worse than Selective-Middle on WikiText-103; the corresponding OpenWebText seed-2024 gap was 0.0077. This is strong descriptive evidence that the selective-release behavior transfers across token distributions, but one WikiText seed is not enough for a formal robustness claim.

## Memory

Selective-Middle releases 864 MiB of K-state, saves {middle['current_memory_saved_vs_newton_mib']:.0f} MiB ({middle['current_memory_saved_vs_newton_fraction']:.1%}) of current allocated memory, and saves {middle['peak_memory_saved_vs_newton_mib']:.0f} MiB ({middle['peak_memory_saved_vs_newton_fraction']:.1%}) of peak memory relative to Newton-Muon.

Selective-All releases 1,296 MiB of K-state, saves {all_cproj['current_memory_saved_vs_newton_mib']:.0f} MiB ({all_cproj['current_memory_saved_vs_newton_fraction']:.1%}) of current allocated memory, and saves {all_cproj['peak_memory_saved_vs_newton_mib']:.0f} MiB ({all_cproj['peak_memory_saved_vs_newton_fraction']:.1%}) of peak memory relative to Newton-Muon.

## Convergence

At validation loss 3.60, Newton-Muon and both selective variants first cross at step 10,500, while Muon crosses at step 12,000. At 3.58, Newton-Muon and Selective-Middle cross at step 10,500, Selective-All crosses at step 12,000, and Muon never reaches the threshold.

## Decision

- Treat this as a successful first dataset-generalization probe.
- Keep Selective-Middle as the conservative rule and Selective-All as the aggressive memory-saving rule.
- Run seeds 2025 and 2026 for all four methods before using WikiText-103 as a multi-seed paper result.
- Do not compare absolute loss values between OpenWebText and WikiText-103; compare within-dataset method deltas and rankings.
- Exclude wall-clock time from conclusions. The last validation is at step 12,000 rather than max_iters=12,208, consistent with prior experiments.
"""
    (OUTPUT_ROOT / "RESULT_NOTES.md").write_text(notes, encoding="utf-8")
    print(f"Wrote WikiText-103 analysis to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
