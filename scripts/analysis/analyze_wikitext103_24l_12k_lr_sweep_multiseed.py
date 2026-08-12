from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


SEEDS = [2024, 2025, 2026]
LEARNING_RATES = [0.01, 0.02]
METHODS = ["muon", "newton", "selective_middle", "selective_all"]
METHOD_LABELS = {
    "muon": "Muon",
    "newton": "Newton-Muon",
    "selective_middle": "Selective-Middle (release56)",
    "selective_all": "Selective-All-c_proj (release84)",
}
TOKENS_PER_STEP = 2 * 512


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(
    os.environ.get("SNM_RESULTS_ROOT", str(ARTIFACT_ROOT / "runs"))
).expanduser()
OUTPUT_ROOT = (
    RESULTS_ROOT
    / "04_dataset_generalization"
    / "wikitext103_24L_12k_lr_sweep_multiseed_20260715"
)
RAW_ROOT = OUTPUT_ROOT / "raw_wandb_exports"
SUMMARY_ROOT = OUTPUT_ROOT / "summaries"


def parse_run(run_name: str) -> tuple[float, int, str]:
    lr_match = re.search(r"mulr(\d+p\d+)", run_name)
    seed_match = re.search(r"seed(\d+)", run_name)
    if not lr_match or not seed_match:
        raise ValueError(f"Missing learning rate or seed in run name: {run_name}")
    learning_rate = float(lr_match.group(1).replace("p", "."))
    seed = int(seed_match.group(1))
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
    return learning_rate, seed, method


def load_series():
    series = defaultdict(list)
    quality_rows = []
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
            learning_rate, seed, method = parse_run(run_name)
            columns.append((index, learning_rate, seed, method, metric, run_name))

        counts = defaultdict(int)
        steps = []
        for row in rows[1:]:
            if not row or not row[0]:
                continue
            step = int(float(row[0]))
            steps.append(step)
            for index, learning_rate, seed, method, metric, _ in columns:
                if index < len(row) and row[index] != "":
                    key = (learning_rate, seed, method, metric)
                    series[key].append((step, float(row[index])))
                    counts[key] += 1

        metrics = sorted({metric for _, _, _, _, metric, _ in columns})
        run_names = {run_name for _, _, _, _, _, run_name in columns}
        quality_rows.append(
            {
                "file": path.name,
                "metrics": ",".join(metrics),
                "csv_rows": len(rows) - 1,
                "min_step": min(steps),
                "max_step": max(steps),
                "run_columns": len(run_names),
                "nonempty_points_min": min(counts.values()),
                "nonempty_points_max": max(counts.values()),
            }
        )

    for key, points in series.items():
        points.sort(key=lambda item: item[0])
        point_steps = [step for step, _ in points]
        if len(point_steps) != len(set(point_steps)):
            raise ValueError(f"Duplicate steps for {key}")
    return series, quality_rows


def last_value(series, key):
    points = series.get(key, [])
    return points[-1][1] if points else None


def to_mib(value):
    return 0.0 if value is None else value / 1024.0 / 1024.0


def normalized_auc(points):
    area = sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for (left_step, left_value), (right_step, right_value) in zip(points, points[1:])
    )
    return area / (points[-1][0] - points[0][0])


def make_run_rows(series):
    expected = [(lr, seed, method) for lr in LEARNING_RATES for seed in SEEDS for method in METHODS]
    val_keys = [(lr, seed, method, "val/loss") for lr, seed, method in expected]
    missing = [key for key in val_keys if key not in series]
    if missing:
        raise ValueError(f"Missing validation series: {missing}")
    common_steps = set.intersection(*(set(step for step, _ in series[key]) for key in val_keys))
    final_step = max(common_steps)

    rows = []
    for learning_rate, seed, method in expected:
        prefix = (learning_rate, seed, method)
        val_points = series[(*prefix, "val/loss")]
        final_val = dict(val_points)[final_step]
        best_step, best_val = min(val_points, key=lambda item: item[1])
        late_values = [value for step, value in val_points if 10000 <= step <= final_step]
        train_eval = series.get((*prefix, "train/loss"), [])
        train_step = series.get((*prefix, "train/loss_step"), [])
        current_memory = last_value(series, (*prefix, "cuda/memory_allocated_mib"))
        peak_points = series.get((*prefix, "cuda/full_run_max_memory_allocated_mib"), [])
        k_state = to_mib(last_value(series, (*prefix, "matrix/k_state_bytes")))
        full_k_state = 0.0
        if method == "newton":
            full_k_state = k_state
        elif method.startswith("selective"):
            full_k_state = to_mib(
                last_value(series, (*prefix, "matrix/k_state_full_bytes"))
            )

        rows.append(
            {
                "learning_rate": learning_rate,
                "seed": seed,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "last_common_val_step": final_step,
                "last_common_val_tokens": final_step * TOKENS_PER_STEP,
                "final_val_loss": final_val,
                "best_val_loss": best_val,
                "best_val_step": best_step,
                "final_minus_best_val": final_val - best_val,
                "late_val_mean_10000_11500": mean(late_values),
                "normalized_val_auc": normalized_auc(val_points),
                "final_train_eval_loss": train_eval[-1][1],
                "last_train_loss_step": train_step[-1][0],
                "tail50_train_loss_mean": mean(value for _, value in train_step[-50:]),
                "current_memory_mib": current_memory,
                "peak_memory_mib": max(value for _, value in peak_points),
                "k_state_mib": k_state,
                "full_k_state_mib": full_k_state,
                "k_state_released_mib": to_mib(
                    last_value(series, (*prefix, "matrix/k_state_released_bytes"))
                ),
                "k_state_released_fraction": last_value(
                    series, (*prefix, "matrix/k_state_released_fraction")
                )
                or 0.0,
            }
        )

    lookup = {(row["learning_rate"], row["seed"], row["method"]): row for row in rows}
    for row in rows:
        newton = lookup[(row["learning_rate"], row["seed"], "newton")]
        muon = lookup[(row["learning_rate"], row["seed"], "muon")]
        row["final_val_delta_vs_newton"] = row["final_val_loss"] - newton["final_val_loss"]
        row["final_val_delta_vs_muon"] = row["final_val_loss"] - muon["final_val_loss"]
        row["peak_memory_saved_vs_newton_mib"] = (
            newton["peak_memory_mib"] - row["peak_memory_mib"]
        )
    return rows, final_step


def aggregate_rows(run_rows):
    metrics = [
        "final_val_loss",
        "best_val_loss",
        "best_val_step",
        "final_minus_best_val",
        "late_val_mean_10000_11500",
        "normalized_val_auc",
        "tail50_train_loss_mean",
        "current_memory_mib",
        "peak_memory_mib",
        "k_state_mib",
        "full_k_state_mib",
        "k_state_released_mib",
        "k_state_released_fraction",
        "final_val_delta_vs_newton",
        "final_val_delta_vs_muon",
        "peak_memory_saved_vs_newton_mib",
    ]
    output = []
    for learning_rate in LEARNING_RATES:
        for method in METHODS:
            selected = [
                row
                for row in run_rows
                if row["learning_rate"] == learning_rate and row["method"] == method
            ]
            aggregate = {
                "learning_rate": learning_rate,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_seeds": len(selected),
            }
            for metric in metrics:
                values = [float(row[metric]) for row in selected]
                aggregate[f"{metric}_mean"] = mean(values)
                aggregate[f"{metric}_std"] = stdev(values)
            output.append(aggregate)
    return output


def make_lr_effect_rows(run_rows):
    lookup = {(row["learning_rate"], row["seed"], row["method"]): row for row in run_rows}
    rows = []
    for seed in SEEDS:
        for method in METHODS:
            low = lookup[(0.01, seed, method)]
            high = lookup[(0.02, seed, method)]
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "final_val_lr002_minus_lr001": high["final_val_loss"] - low["final_val_loss"],
                    "best_val_lr002_minus_lr001": high["best_val_loss"] - low["best_val_loss"],
                    "late_mean_lr002_minus_lr001": high["late_val_mean_10000_11500"]
                    - low["late_val_mean_10000_11500"],
                    "auc_lr002_minus_lr001": high["normalized_val_auc"]
                    - low["normalized_val_auc"],
                }
            )
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        for row in selected:
            row["method_mean_final_delta"] = mean(
                item["final_val_lr002_minus_lr001"] for item in selected
            )
            row["method_std_final_delta"] = stdev(
                item["final_val_lr002_minus_lr001"] for item in selected
            )
    return rows


def make_pairwise_rows(run_rows):
    lookup = {(row["learning_rate"], row["seed"], row["method"]): row for row in run_rows}
    rows = []
    for learning_rate in LEARNING_RATES:
        for seed in SEEDS:
            newton = lookup[(learning_rate, seed, "newton")]
            middle = lookup[(learning_rate, seed, "selective_middle")]
            all_cproj = lookup[(learning_rate, seed, "selective_all")]
            rows.append(
                {
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "middle_minus_newton_final_val": middle["final_val_loss"]
                    - newton["final_val_loss"],
                    "all_minus_newton_final_val": all_cproj["final_val_loss"]
                    - newton["final_val_loss"],
                    "all_minus_middle_final_val": all_cproj["final_val_loss"]
                    - middle["final_val_loss"],
                }
            )
    return rows


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    series, quality_rows = load_series()
    run_rows, final_step = make_run_rows(series)
    aggregate = aggregate_rows(run_rows)
    lr_effect = make_lr_effect_rows(run_rows)
    pairwise = make_pairwise_rows(run_rows)
    val_curve_rows = []
    for learning_rate in LEARNING_RATES:
        for seed in SEEDS:
            for method in METHODS:
                for step, value in series[(learning_rate, seed, method, "val/loss")]:
                    val_curve_rows.append(
                        {
                            "learning_rate": learning_rate,
                            "seed": seed,
                            "method": method,
                            "step": step,
                            "tokens": step * TOKENS_PER_STEP,
                            "val_loss": value,
                        }
                    )

    write_csv(SUMMARY_ROOT / "data_quality_checks.csv", quality_rows)
    write_csv(SUMMARY_ROOT / "run_summary.csv", run_rows)
    write_csv(SUMMARY_ROOT / "aggregate_by_lr_method.csv", aggregate)
    write_csv(SUMMARY_ROOT / "learning_rate_effect.csv", lr_effect)
    write_csv(SUMMARY_ROOT / "paired_method_deltas.csv", pairwise)
    write_csv(SUMMARY_ROOT / "val_curve_long.csv", val_curve_rows)

    key = {
        "config": {
            "dataset": "wikitext103_gpt2_50m",
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024,
            "batch_size": 2,
            "block_size": 512,
            "max_iters": 12000,
            "lr_decay_iters": 3000,
            "learning_rates": LEARNING_RATES,
            "seeds": SEEDS,
            "last_logged_train_step": 11980,
            "last_validation_step": final_step,
            "last_validation_tokens": final_step * TOKENS_PER_STEP,
        },
        "coverage": {
            "expected_runs": 24,
            "summarized_runs": len(run_rows),
            "raw_exports": len(list(RAW_ROOT.glob("*.csv"))),
        },
        "aggregate_by_lr_method": aggregate,
    }
    (SUMMARY_ROOT / "key_results.json").write_text(
        json.dumps(key, indent=2), encoding="utf-8"
    )
    print(f"summarized {len(run_rows)} runs; last validation step={final_step}")
    print(f"wrote summaries to {SUMMARY_ROOT}")


if __name__ == "__main__":
    main()
