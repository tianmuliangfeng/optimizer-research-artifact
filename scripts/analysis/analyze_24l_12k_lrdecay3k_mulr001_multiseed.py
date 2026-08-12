from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METHOD_LABELS = {
    "muon": "Muon",
    "newton": "Newton-Muon",
    "selective_all_cproj": "Selective-All-c_proj (84%)",
}
METHOD_ORDER = ["muon", "newton", "selective_all_cproj"]
TOKENS_PER_STEP = 2 * 512


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(
    os.environ.get("SNM_RESULTS_ROOT", str(ARTIFACT_ROOT / "runs"))
).expanduser()
OUTPUT_ROOT = (
    RESULTS_ROOT
    / "01_scale_up"
    / "24L_12k_lrdecay3k_mulr001_multiseed_20260713"
)
RAW_EXPORT_ROOT = OUTPUT_ROOT / "raw_wandb_exports"
SUMMARY_ROOT = OUTPUT_ROOT / "summaries"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
NOTES_ROOT = OUTPUT_ROOT / "notes"

SEED2024_ROOT = (
    RESULTS_ROOT
    / "01_scale_up"
    / "24L_12k_lrdecay3k_mulr_sweep_seed2024_20260713"
    / "summaries"
)


def parse_run_name(run_name: str) -> tuple[int, str, str]:
    seed_match = re.search(r"seed(\d+)", run_name)
    if not seed_match:
        raise ValueError(f"Could not parse seed from run name: {run_name}")
    seed = int(seed_match.group(1))

    if "00_muon" in run_name:
        method = "muon"
    elif "13_newton_muon_fast" in run_name:
        method = "newton"
    elif "all_cproj_release84" in run_name:
        method = "selective_all_cproj"
    else:
        raise ValueError(f"Could not parse method from run name: {run_name}")

    return seed, method, METHOD_LABELS[method]


def read_wandb_metric_exports(raw_root: Path) -> dict[tuple[int, str, str], list[tuple[int, float]]]:
    series_by_key: dict[tuple[int, str, str], list[tuple[int, float]]] = defaultdict(list)

    for path in sorted(raw_root.glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)

            columns: list[tuple[int, str, int, str]] = []
            for idx, column in enumerate(header):
                if idx == 0 or " - " not in column:
                    continue
                run_name, metric = column.rsplit(" - ", 1)
                if metric.endswith("__MIN") or metric.endswith("__MAX"):
                    continue
                seed, method, _ = parse_run_name(run_name)
                columns.append((idx, metric, seed, method))

            for row in reader:
                if not row or not row[0]:
                    continue
                step = int(float(row[0]))
                for idx, metric, seed, method in columns:
                    if idx >= len(row) or row[idx] == "":
                        continue
                    series_by_key[(seed, method, metric)].append((step, float(row[idx])))

    for key in list(series_by_key):
        series_by_key[key].sort(key=lambda item: item[0])

    return series_by_key


def value_at_step(series: list[tuple[int, float]], step: int) -> float | None:
    for series_step, value in series:
        if series_step == step:
            return value
    return None


def mib_from_bytes(value: float) -> float:
    return value / 1024.0 / 1024.0


def make_new_seed_summaries(
    series_by_key: dict[tuple[int, str, str], list[tuple[int, float]]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []

    for seed in [2025, 2026]:
        for method in METHOD_ORDER:
            val_series = series_by_key.get((seed, method, "val/loss"), [])
            if not val_series:
                continue

            final_step, final_val = val_series[-1]
            best_step, best_val = min(val_series, key=lambda item: item[1])
            late_values = [value for step, value in val_series if 9000 <= step <= 11500]

            row: dict[str, object] = {
                "seed": seed,
                "muon_learning_rate": 0.01,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "max_iters": 12000,
                "lr_decay_iters": 3000,
                "last_val_step": final_step,
                "last_val_tokens": final_step * TOKENS_PER_STEP,
                "last_val_tokens_millions": final_step * TOKENS_PER_STEP / 1_000_000,
                "final_val_loss": final_val,
                "best_val_loss": best_val,
                "best_val_step": best_step,
                "best_val_tokens": best_step * TOKENS_PER_STEP,
                "best_val_tokens_millions": best_step * TOKENS_PER_STEP / 1_000_000,
                "final_minus_best_val": final_val - best_val,
                "val_loss_at_2500": value_at_step(val_series, 2500),
                "val_loss_at_5000": value_at_step(val_series, 5000),
                "val_loss_at_10000": value_at_step(val_series, 10000),
                "val_loss_late_mean_9000_11500": mean(late_values) if late_values else None,
            }

            train_eval_series = series_by_key.get((seed, method, "train/loss"), [])
            if train_eval_series:
                train_final_step, train_final = train_eval_series[-1]
                train_best_step, train_best = min(train_eval_series, key=lambda item: item[1])
                row.update(
                    {
                        "final_train_eval_step": train_final_step,
                        "final_train_eval_loss": train_final,
                        "best_train_eval_loss": train_best,
                        "best_train_eval_step": train_best_step,
                    }
                )

            train_loss_series = series_by_key.get((seed, method, "train/loss_step"), [])
            if train_loss_series:
                loss_final_step, loss_final = train_loss_series[-1]
                loss_best_step, loss_best = min(train_loss_series, key=lambda item: item[1])
                row.update(
                    {
                        "last_train_loss_step": loss_final_step,
                        "last_train_loss": loss_final,
                        "best_train_loss_step": loss_best_step,
                        "best_train_loss": loss_best,
                    }
                )

            time_series = series_by_key.get((seed, method, "time_elapsed"), [])
            if time_series:
                last_time = time_series[-1][1]
                row["time_elapsed_sec_at_last_train_log"] = last_time
                last_train_step = row.get("last_train_loss_step")
                if isinstance(last_train_step, int) and last_time > 0:
                    row["nominal_tokens_per_sec_to_last_train_log"] = (
                        last_train_step * TOKENS_PER_STEP / last_time
                    )

            memory_series = series_by_key.get((seed, method, "cuda/memory_allocated_mib"), [])
            if memory_series:
                row["current_memory_step"] = memory_series[-1][0]
                row["current_memory_mib"] = memory_series[-1][1]

            peak_memory_series = series_by_key.get(
                (seed, method, "cuda/full_run_max_memory_allocated_mib"), []
            )
            if peak_memory_series:
                row["peak_memory_mib"] = peak_memory_series[-1][1]

            k_state_series = series_by_key.get((seed, method, "matrix/k_state_bytes"), [])
            if k_state_series:
                row["k_state_mib"] = mib_from_bytes(k_state_series[-1][1])

            full_k_state_series = series_by_key.get(
                (seed, method, "matrix/k_state_full_bytes"), []
            )
            if full_k_state_series:
                row["full_k_state_mib"] = mib_from_bytes(full_k_state_series[-1][1])
            elif method == "newton":
                row["full_k_state_mib"] = row.get("k_state_mib")
            elif method == "muon":
                row["full_k_state_mib"] = 0.0

            released_series = series_by_key.get(
                (seed, method, "matrix/k_state_released_bytes"), []
            )
            if released_series:
                row["k_state_released_mib"] = mib_from_bytes(released_series[-1][1])
            else:
                row["k_state_released_mib"] = 0.0

            released_fraction_series = series_by_key.get(
                (seed, method, "matrix/k_state_released_fraction"), []
            )
            if released_fraction_series:
                row["k_state_released_fraction"] = released_fraction_series[-1][1]
            else:
                row["k_state_released_fraction"] = 0.0

            summaries.append(row)

    return summaries


def load_seed2024_summary() -> list[dict[str, object]]:
    path = SEED2024_ROOT / "run_summary.csv"
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["muon_learning_rate"] != "0.01":
                continue
            if row["method"] not in METHOD_ORDER:
                continue
            converted: dict[str, object] = {}
            for key, value in row.items():
                if value == "":
                    converted[key] = None
                elif key in {"seed", "max_iters", "lr_decay_iters", "last_val_step", "best_val_step"}:
                    converted[key] = int(float(value))
                elif key in {"method", "method_label"}:
                    converted[key] = value
                else:
                    converted[key] = float(value)
            rows.append(converted)
    return rows


def make_new_val_curve(
    series_by_key: dict[tuple[int, str, str], list[tuple[int, float]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in [2025, 2026]:
        for method in METHOD_ORDER:
            for step, value in series_by_key.get((seed, method, "val/loss"), []):
                rows.append(
                    {
                        "seed": seed,
                        "muon_learning_rate": 0.01,
                        "step": step,
                        "tokens": step * TOKENS_PER_STEP,
                        "tokens_millions": step * TOKENS_PER_STEP / 1_000_000,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "val_loss": value,
                    }
                )
    return rows


def load_seed2024_val_curve() -> list[dict[str, object]]:
    path = SEED2024_ROOT / "val_curve_by_checkpoint.csv"
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["muon_learning_rate"] != "0.01":
                continue
            if row["method"] not in METHOD_ORDER:
                continue
            rows.append(
                {
                    "seed": 2024,
                    "muon_learning_rate": 0.01,
                    "step": int(float(row["step"])),
                    "tokens": int(float(row["tokens"])),
                    "tokens_millions": float(row["tokens_millions"]),
                    "method": row["method"],
                    "method_label": row["method_label"],
                    "val_loss": float(row["val_loss"]),
                }
            )
    return rows


def add_curve_ranks_and_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), int(row["step"]))].append(row)

    ranked_rows: list[dict[str, object]] = []
    for _, group in sorted(grouped.items()):
        values = {str(row["method"]): float(row["val_loss"]) for row in group}
        ranked_group = sorted(group, key=lambda row: float(row["val_loss"]))
        ranks = {str(row["method"]): rank + 1 for rank, row in enumerate(ranked_group)}
        for row in group:
            method = str(row["method"])
            enriched = dict(row)
            enriched["rank_at_seed_step"] = ranks[method]
            enriched["delta_vs_muon"] = values.get(method, math.nan) - values.get("muon", math.nan)
            enriched["delta_vs_newton"] = values.get(method, math.nan) - values.get("newton", math.nan)
            enriched["delta_vs_selective_all_cproj"] = values.get(method, math.nan) - values.get(
                "selective_all_cproj", math.nan
            )
            ranked_rows.append(enriched)

    return sorted(ranked_rows, key=lambda row: (int(row["seed"]), int(row["step"]), METHOD_ORDER.index(str(row["method"]))))


def numeric_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        values.append(float(value))
    return values


def summarize_values(rows: list[dict[str, object]], key: str) -> dict[str, object]:
    values = numeric_values(rows, key)
    if not values:
        return {f"{key}_mean": None, f"{key}_std": None}
    return {
        f"{key}_mean": mean(values),
        f"{key}_std": stdev(values) if len(values) > 1 else 0.0,
    }


def make_method_aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    metrics = [
        "final_val_loss",
        "best_val_loss",
        "val_loss_at_10000",
        "val_loss_late_mean_9000_11500",
        "final_minus_best_val",
        "current_memory_mib",
        "peak_memory_mib",
        "k_state_mib",
        "full_k_state_mib",
        "k_state_released_mib",
        "k_state_released_fraction",
    ]
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        aggregate: dict[str, object] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_seeds": len({row["seed"] for row in method_rows}),
        }
        for metric in metrics:
            aggregate.update(summarize_values(method_rows, metric))
        output.append(aggregate)
    return output


def make_val_curve_aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["step"]))].append(row)

    output: list[dict[str, object]] = []
    for (method, step), group in sorted(grouped.items(), key=lambda item: (item[0][1], METHOD_ORDER.index(item[0][0]))):
        values = numeric_values(group, "val_loss")
        output.append(
            {
                "step": step,
                "tokens": step * TOKENS_PER_STEP,
                "tokens_millions": step * TOKENS_PER_STEP / 1_000_000,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_seeds": len(values),
                "val_loss_mean": mean(values),
                "val_loss_std": stdev(values) if len(values) > 1 else 0.0,
                "val_loss_min": min(values),
                "val_loss_max": max(values),
            }
        )
    return output


def make_pairwise_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_seed: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_seed[int(row["seed"])][str(row["method"])] = row

    output: list[dict[str, object]] = []
    for seed, seed_rows in sorted(by_seed.items()):
        for baseline_method in ["muon", "newton"]:
            baseline = seed_rows.get(baseline_method)
            if baseline is None:
                continue
            for target_method in METHOD_ORDER:
                if target_method == baseline_method or target_method not in seed_rows:
                    continue
                target = seed_rows[target_method]
                row = {
                    "seed": seed,
                    "baseline_method": baseline_method,
                    "target_method": target_method,
                    "target_label": METHOD_LABELS[target_method],
                    "final_val_delta": float(target["final_val_loss"]) - float(baseline["final_val_loss"]),
                    "best_val_delta": float(target["best_val_loss"]) - float(baseline["best_val_loss"]),
                    "val_loss_at_10000_delta": float(target["val_loss_at_10000"]) - float(baseline["val_loss_at_10000"]),
                    "late_mean_delta": float(target["val_loss_late_mean_9000_11500"]) - float(
                        baseline["val_loss_late_mean_9000_11500"]
                    ),
                }
                for metric in ["current_memory_mib", "peak_memory_mib", "k_state_mib"]:
                    if target.get(metric) is not None and baseline.get(metric) is not None:
                        row[f"{metric}_delta"] = float(target[metric]) - float(baseline[metric])
                        row[f"{metric}_saved"] = float(baseline[metric]) - float(target[metric])
                output.append(row)
    return output


def make_pairwise_aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["baseline_method"]), str(row["target_method"]))].append(row)

    output: list[dict[str, object]] = []
    metrics = [
        "final_val_delta",
        "best_val_delta",
        "val_loss_at_10000_delta",
        "late_mean_delta",
        "current_memory_mib_saved",
        "peak_memory_mib_saved",
        "k_state_mib_saved",
    ]
    for (baseline, target), group in sorted(grouped.items()):
        row: dict[str, object] = {
            "baseline_method": baseline,
            "target_method": target,
            "target_label": METHOD_LABELS[target],
            "n_seeds": len(group),
            "target_final_better_seed_count": sum(1 for item in group if float(item["final_val_delta"]) < 0),
            "target_best_better_seed_count": sum(1 for item in group if float(item["best_val_delta"]) < 0),
        }
        for metric in metrics:
            row.update(summarize_values(group, metric))
        output.append(row)
    return output


def make_data_quality_checks(run_rows: list[dict[str, object]], raw_root: Path) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    raw_files = sorted(raw_root.glob("*.csv"))
    checks.append(
        {
            "check": "raw_export_file_count",
            "status": "pass" if len(raw_files) == 12 else "warn",
            "details": f"{len(raw_files)} CSV files found",
        }
    )

    expected_new = {(seed, method) for seed in [2025, 2026] for method in METHOD_ORDER}
    observed_new = {
        (int(row["seed"]), str(row["method"]))
        for row in run_rows
        if int(row["seed"]) in {2025, 2026}
    }
    missing_new = sorted(expected_new - observed_new)
    checks.append(
        {
            "check": "new_seed_method_coverage",
            "status": "pass" if not missing_new else "warn",
            "details": "all seed2025/2026 method combinations present"
            if not missing_new
            else f"missing {missing_new}",
        }
    )

    expected_all = {(seed, method) for seed in [2024, 2025, 2026] for method in METHOD_ORDER}
    observed_all = {(int(row["seed"]), str(row["method"])) for row in run_rows}
    missing_all = sorted(expected_all - observed_all)
    checks.append(
        {
            "check": "combined_seed_method_coverage",
            "status": "pass" if not missing_all else "warn",
            "details": "all three seeds and three methods present" if not missing_all else f"missing {missing_all}",
        }
    )

    last_steps = sorted({int(row["last_val_step"]) for row in run_rows})
    checks.append(
        {
            "check": "last_validation_step",
            "status": "pass" if last_steps == [11500] else "warn",
            "details": f"observed last validation steps: {last_steps}",
        }
    )

    muon_lrs = sorted({float(row["muon_learning_rate"]) for row in run_rows})
    lr_decay_iters = sorted({int(row["lr_decay_iters"]) for row in run_rows})
    checks.append(
        {
            "check": "schedule_consistency",
            "status": "pass" if muon_lrs == [0.01] and lr_decay_iters == [3000] else "warn",
            "details": f"muon_learning_rate={muon_lrs}, lr_decay_iters={lr_decay_iters}",
        }
    )

    return checks


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: object, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def create_line_svg(rows: list[dict[str, object]], path: Path) -> None:
    post_warmup = [row for row in rows if int(row["step"]) >= 500]
    width, height = 920, 520
    left, right, top, bottom = 72, 28, 28, 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_min, x_max = 500, 11500
    y_values = [float(row["val_loss_mean"]) for row in post_warmup]
    y_min = math.floor((min(y_values) - 0.05) * 10) / 10
    y_max = math.ceil((max(y_values) + 0.05) * 10) / 10
    colors = {"muon": "#4B8BBE", "newton": "#D95F02", "selective_all_cproj": "#2CA25F"}

    def x_pos(step: int) -> float:
        return left + (step - x_min) / (x_max - x_min) * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="72" y="24" font-family="Arial" font-size="18" font-weight="700">24L / 12k / lr_decay_iters=3000 / muon_lr=0.01: mean validation loss</text>',
    ]
    for i in range(6):
        y = y_min + (y_max - y_min) * i / 5
        py = y_pos(y)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{py:.2f}" y2="{py:.2f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="22" y="{py+4:.2f}" font-family="Arial" font-size="12" fill="#555">{y:.2f}</text>')
    for step in [500, 2500, 5000, 7500, 10000, 11500]:
        px = x_pos(step)
        parts.append(f'<line x1="{px:.2f}" x2="{px:.2f}" y1="{top}" y2="{height-bottom}" stroke="#f2f2f2"/>')
        parts.append(f'<text x="{px-18:.2f}" y="{height-32}" font-family="Arial" font-size="12" fill="#555">{step}</text>')
    parts.append(f'<line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" stroke="#777"/>')
    parts.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" stroke="#777"/>')

    for method in METHOD_ORDER:
        points = [
            (int(row["step"]), float(row["val_loss_mean"]))
            for row in post_warmup
            if row["method"] == method
        ]
        point_string = " ".join(f"{x_pos(step):.2f},{y_pos(value):.2f}" for step, value in points)
        parts.append(
            f'<polyline fill="none" stroke="{colors[method]}" stroke-width="3" points="{point_string}"/>'
        )

    for idx, method in enumerate(METHOD_ORDER):
        y = 56 + idx * 22
        parts.append(f'<line x1="690" x2="730" y1="{y}" y2="{y}" stroke="{colors[method]}" stroke-width="4"/>')
        parts.append(
            f'<text x="738" y="{y+4}" font-family="Arial" font-size="13" fill="#333">{METHOD_LABELS[method]}</text>'
        )
    parts.append('<text x="440" y="500" font-family="Arial" font-size="13" fill="#555">step</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def create_seed_bar_svg(rows: list[dict[str, object]], path: Path) -> None:
    width, height = 920, 500
    left, right, top, bottom = 78, 30, 34, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {"muon": "#4B8BBE", "newton": "#D95F02", "selective_all_cproj": "#2CA25F"}
    seeds = [2024, 2025, 2026]
    values = [float(row["final_val_loss"]) for row in rows]
    y_min = math.floor((min(values) - 0.03) * 10) / 10
    y_max = math.ceil((max(values) + 0.03) * 10) / 10

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="78" y="24" font-family="Arial" font-size="18" font-weight="700">Final validation loss by seed</text>',
    ]
    for i in range(6):
        y = y_min + (y_max - y_min) * i / 5
        py = y_pos(y)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{py:.2f}" y2="{py:.2f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="28" y="{py+4:.2f}" font-family="Arial" font-size="12" fill="#555">{y:.2f}</text>')

    group_w = plot_w / len(seeds)
    bar_w = 52
    by_seed_method = {(int(row["seed"]), str(row["method"])): row for row in rows}
    for seed_idx, seed in enumerate(seeds):
        group_left = left + seed_idx * group_w
        center = group_left + group_w / 2
        parts.append(f'<text x="{center-18:.2f}" y="{height-32}" font-family="Arial" font-size="13" fill="#555">{seed}</text>')
        for method_idx, method in enumerate(METHOD_ORDER):
            row = by_seed_method[(seed, method)]
            value = float(row["final_val_loss"])
            x = center + (method_idx - 1) * (bar_w + 12) - bar_w / 2
            y = y_pos(value)
            h = height - bottom - y
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w}" height="{h:.2f}" fill="{colors[method]}"/>'
            )
            parts.append(
                f'<text x="{x+2:.2f}" y="{y-5:.2f}" font-family="Arial" font-size="11" fill="#333">{value:.3f}</text>'
            )
    for idx, method in enumerate(METHOD_ORDER):
        x = 660
        y = 56 + idx * 22
        parts.append(f'<rect x="{x}" y="{y-10}" width="14" height="14" fill="{colors[method]}"/>')
        parts.append(
            f'<text x="{x+22}" y="{y+2}" font-family="Arial" font-size="13" fill="#333">{METHOD_LABELS[method]}</text>'
        )
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def write_notes(
    run_rows: list[dict[str, object]],
    method_aggregate: list[dict[str, object]],
    pairwise_aggregate: list[dict[str, object]],
    checks: list[dict[str, object]],
) -> None:
    agg_by_method = {str(row["method"]): row for row in method_aggregate}
    selective_vs_muon = next(
        row
        for row in pairwise_aggregate
        if row["baseline_method"] == "muon" and row["target_method"] == "selective_all_cproj"
    )
    selective_vs_newton = next(
        row
        for row in pairwise_aggregate
        if row["baseline_method"] == "newton" and row["target_method"] == "selective_all_cproj"
    )

    lines = [
        "# 24L 12k lr-decay3k muon-lr0.01 multiseed confirmation",
        "",
        "Config: 24 layers, n_embd=1024, n_head=16, batch_size=2, block_size=512, max_iters=12000, lr_decay_iters=3000, muon_learning_rate=0.01.",
        "",
        "Included runs: seed2024 from the prior matrix-LR sweep plus seed2025/2026 from the confirmation batch.",
        "",
        "## Main result",
        "",
        f"- Final val loss mean: Muon {fmt(agg_by_method['muon']['final_val_loss_mean'])}, Newton-Muon {fmt(agg_by_method['newton']['final_val_loss_mean'])}, Selective-All {fmt(agg_by_method['selective_all_cproj']['final_val_loss_mean'])}.",
        f"- Selective-All beats Muon on final val loss in {selective_vs_muon['target_final_better_seed_count']}/3 seeds with mean delta {fmt(selective_vs_muon['final_val_delta_mean'])}.",
        f"- Selective-All beats Newton-Muon on final val loss in {selective_vs_newton['target_final_better_seed_count']}/3 seeds with mean delta {fmt(selective_vs_newton['final_val_delta_mean'])}.",
        f"- Selective-All releases {fmt(agg_by_method['selective_all_cproj']['k_state_released_fraction_mean'] * 100, 2)}% of K-state and saves {fmt(selective_vs_newton['k_state_mib_saved_mean'], 1)} MiB of K-state versus full Newton-Muon.",
        "",
        "## Interpretation",
        "",
        "This confirms the 24L long-run setting after lowering matrix LR: the previous late-training degradation is largely fixed, and Selective-All is the strongest of the three tested methods at the endpoint and at the best checkpoint.",
        "",
        "Newton-Muon improves strongly over Muon at muon_learning_rate=0.01, but Selective-All keeps an additional loss advantage while using much less K-state memory.",
        "",
        "## Data quality",
        "",
    ]
    for check in checks:
        lines.append(f"- {check['check']}: {check['status']} ({check['details']})")

    lines.extend(
        [
            "",
            "## Saved artifacts",
            "",
            "- summaries/run_summary.csv",
            "- summaries/method_aggregate.csv",
            "- summaries/pairwise_deltas_by_seed.csv",
            "- summaries/pairwise_delta_aggregate.csv",
            "- summaries/val_curve_by_checkpoint.csv",
            "- summaries/val_curve_aggregate.csv",
            "- figures/val_loss_mean_curve_step500_11500.svg",
            "- figures/final_val_loss_by_seed.svg",
        ]
    )

    (NOTES_ROOT / "RESULT_NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    NOTES_ROOT.mkdir(parents=True, exist_ok=True)

    series_by_key = read_wandb_metric_exports(RAW_EXPORT_ROOT)
    run_rows = load_seed2024_summary() + make_new_seed_summaries(series_by_key)
    run_rows.sort(key=lambda row: (int(row["seed"]), METHOD_ORDER.index(str(row["method"]))))

    val_curve_rows = add_curve_ranks_and_deltas(load_seed2024_val_curve() + make_new_val_curve(series_by_key))
    val_curve_aggregate = make_val_curve_aggregate(val_curve_rows)
    method_aggregate = make_method_aggregate(run_rows)
    pairwise_deltas = make_pairwise_deltas(run_rows)
    pairwise_aggregate = make_pairwise_aggregate(pairwise_deltas)
    checks = make_data_quality_checks(run_rows, RAW_EXPORT_ROOT)

    write_csv(SUMMARY_ROOT / "run_summary.csv", run_rows)
    write_csv(SUMMARY_ROOT / "method_aggregate.csv", method_aggregate)
    write_csv(SUMMARY_ROOT / "pairwise_deltas_by_seed.csv", pairwise_deltas)
    write_csv(SUMMARY_ROOT / "pairwise_delta_aggregate.csv", pairwise_aggregate)
    write_csv(SUMMARY_ROOT / "val_curve_by_checkpoint.csv", val_curve_rows)
    write_csv(SUMMARY_ROOT / "val_curve_aggregate.csv", val_curve_aggregate)
    write_csv(SUMMARY_ROOT / "data_quality_checks.csv", checks, ["check", "status", "details"])
    create_line_svg(val_curve_aggregate, FIGURE_ROOT / "val_loss_mean_curve_step500_11500.svg")
    create_seed_bar_svg(run_rows, FIGURE_ROOT / "final_val_loss_by_seed.svg")
    write_notes(run_rows, method_aggregate, pairwise_aggregate, checks)

    print(f"Wrote summaries to {SUMMARY_ROOT}")
    print(f"Wrote figures to {FIGURE_ROOT}")
    print(f"Wrote notes to {NOTES_ROOT / 'RESULT_NOTES.md'}")


if __name__ == "__main__":
    main()
