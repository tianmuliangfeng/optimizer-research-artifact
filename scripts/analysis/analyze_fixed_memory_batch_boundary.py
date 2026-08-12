from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


METHOD_LABELS = {
    "muon": "Muon",
    "newton": "Newton-Muon",
    "selective_all_cproj": "Selective-All-c_proj (84%)",
}
METHOD_ORDER = ["muon", "newton", "selective_all_cproj"]
TOKENS_PER_SAMPLE = 512


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(
    os.environ.get("SNM_RESULTS_ROOT", str(ARTIFACT_ROOT / "runs"))
).expanduser()
DEFAULT_OUTPUT_ROOT = (
    RESULTS_ROOT
    / "03_fixed_memory"
    / "24L_batch_boundary_seed2024_20260714"
)
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
RAW_ROOT = OUTPUT_ROOT / "raw_wandb_exports"
SUMMARY_ROOT = OUTPUT_ROOT / "summaries"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
NOTES_ROOT = OUTPUT_ROOT / "notes"


def set_output_root(output_root: Path) -> None:
    global OUTPUT_ROOT, RAW_ROOT, SUMMARY_ROOT, FIGURE_ROOT, NOTES_ROOT
    OUTPUT_ROOT = output_root
    RAW_ROOT = OUTPUT_ROOT / "raw_wandb_exports"
    SUMMARY_ROOT = OUTPUT_ROOT / "summaries"
    FIGURE_ROOT = OUTPUT_ROOT / "figures"
    NOTES_ROOT = OUTPUT_ROOT / "notes"


def parse_run_name(run_name: str) -> tuple[int, int, str]:
    batch_match = re.search(r"_bs(\d+)_", run_name)
    seed_match = re.search(r"seed(\d+)", run_name)
    if not batch_match or not seed_match:
        raise ValueError(f"Could not parse run name: {run_name}")

    if "00_muon" in run_name:
        method = "muon"
    elif "13_newton_muon_fast" in run_name:
        method = "newton"
    elif "all_cproj_release84" in run_name:
        method = "selective_all_cproj"
    else:
        raise ValueError(f"Could not parse method from run name: {run_name}")

    return int(seed_match.group(1)), int(batch_match.group(1)), method


def read_wandb_exports() -> dict[tuple[int, int, str, str], list[tuple[int, float]]]:
    series_by_key: dict[tuple[int, int, str, str], list[tuple[int, float]]] = defaultdict(list)

    for path in sorted(RAW_ROOT.glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            columns: list[tuple[int, int, int, str, str]] = []
            for idx, column in enumerate(header):
                if idx == 0 or " - " not in column:
                    continue
                run_name, metric = column.rsplit(" - ", 1)
                if metric.endswith("__MIN") or metric.endswith("__MAX"):
                    continue
                seed, batch_size, method = parse_run_name(run_name)
                columns.append((idx, seed, batch_size, method, metric))

            for row in reader:
                if not row or not row[0]:
                    continue
                step = int(float(row[0]))
                for idx, seed, batch_size, method, metric in columns:
                    if idx >= len(row) or row[idx] == "":
                        continue
                    series_by_key[(seed, batch_size, method, metric)].append((step, float(row[idx])))

    for key in list(series_by_key):
        series_by_key[key].sort(key=lambda item: item[0])

    return series_by_key


def mib_from_bytes(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 1024.0 / 1024.0


def last(series_by_key, seed: int, batch_size: int, method: str, metric: str) -> tuple[int, float] | None:
    series = series_by_key.get((seed, batch_size, method, metric), [])
    if not series:
        return None
    return series[-1]


def best(series_by_key, seed: int, batch_size: int, method: str, metric: str) -> tuple[int, float] | None:
    series = series_by_key.get((seed, batch_size, method, metric), [])
    if not series:
        return None
    return min(series, key=lambda item: item[1])


def build_run_summary(series_by_key) -> list[dict[str, object]]:
    combos = sorted({(seed, batch_size, method) for seed, batch_size, method, _ in series_by_key})
    rows: list[dict[str, object]] = []

    for seed, batch_size, method in combos:
        val_last = last(series_by_key, seed, batch_size, method, "val/loss")
        val_best = best(series_by_key, seed, batch_size, method, "val/loss")
        train_last = last(series_by_key, seed, batch_size, method, "train/loss_step")
        time_last = last(series_by_key, seed, batch_size, method, "time_elapsed")
        peak_mem = last(series_by_key, seed, batch_size, method, "cuda/full_run_max_memory_allocated_mib")
        current_mem = last(series_by_key, seed, batch_size, method, "cuda/memory_allocated_mib")
        k_state = last(series_by_key, seed, batch_size, method, "matrix/k_state_bytes")
        k_full = last(series_by_key, seed, batch_size, method, "matrix/k_state_full_bytes")
        k_released = last(series_by_key, seed, batch_size, method, "matrix/k_state_released_bytes")
        k_release_frac = last(series_by_key, seed, batch_size, method, "matrix/k_state_released_fraction")

        completed = (
            val_last is not None
            and val_last[0] > 0
            and time_last is not None
            and time_last[0] > 0
            and peak_mem is not None
        )
        row: dict[str, object] = {
            "seed": seed,
            "batch_size": batch_size,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "completed": completed,
            "has_val_log": val_last is not None,
            "has_train_loss_step_log": train_last is not None,
            "has_time_elapsed_log": time_last is not None,
            "has_peak_memory_log": peak_mem is not None,
            "tokens_per_iter": batch_size * TOKENS_PER_SAMPLE,
        }
        if val_last:
            row["last_val_step"] = val_last[0]
            row["final_val_loss"] = val_last[1]
        if val_best:
            row["best_val_step"] = val_best[0]
            row["best_val_loss"] = val_best[1]
        if train_last:
            row["last_train_log_step"] = train_last[0]
            row["last_train_loss"] = train_last[1]
        if time_last:
            row["time_elapsed_step"] = time_last[0]
            row["time_elapsed_sec"] = time_last[1]
            progress_step = train_last[0] if train_last else time_last[0]
            if progress_step and time_last[1] > 0:
                row["nominal_tokens_per_sec"] = (
                    progress_step * batch_size * TOKENS_PER_SAMPLE / time_last[1]
                )
        if peak_mem:
            row["peak_memory_mib"] = peak_mem[1]
        if current_mem:
            row["current_memory_mib"] = current_mem[1]
        if k_state:
            row["k_state_mib"] = mib_from_bytes(k_state[1])
        if k_full:
            row["full_k_state_mib"] = mib_from_bytes(k_full[1])
        elif method == "newton":
            row["full_k_state_mib"] = row.get("k_state_mib")
        elif method == "muon":
            row["full_k_state_mib"] = 0.0
        if k_released:
            row["k_state_released_mib"] = mib_from_bytes(k_released[1])
        else:
            row["k_state_released_mib"] = 0.0
        if k_release_frac:
            row["k_state_released_fraction"] = k_release_frac[1]
        else:
            row["k_state_released_fraction"] = 0.0
        rows.append(row)

    return sorted(rows, key=lambda row: (int(row["batch_size"]), METHOD_ORDER.index(str(row["method"]))))


def linear_fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x_bar = mean(xs)
    y_bar = mean(ys)
    denom = sum((x - x_bar) ** 2 for x in xs)
    slope = sum((x - x_bar) * (y - y_bar) for x, y in points) / denom
    intercept = y_bar - slope * x_bar
    return intercept, slope


def build_memory_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    by_batch: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_batch[int(row["batch_size"])][str(row["method"])] = row

    for batch_size, methods in sorted(by_batch.items()):
        newton = methods.get("newton", {})
        selective = methods.get("selective_all_cproj", {})
        muon = methods.get("muon", {})
        row = {
            "batch_size": batch_size,
            "muon_completed": muon.get("completed"),
            "newton_completed": newton.get("completed"),
            "selective_completed": selective.get("completed"),
            "muon_peak_memory_mib": muon.get("peak_memory_mib"),
            "newton_peak_memory_mib": newton.get("peak_memory_mib"),
            "selective_peak_memory_mib": selective.get("peak_memory_mib"),
            "selective_peak_memory_saved_vs_newton_mib": None,
            "selective_peak_memory_ratio_vs_newton": None,
            "muon_current_memory_mib": muon.get("current_memory_mib"),
            "newton_current_memory_mib": newton.get("current_memory_mib"),
            "selective_current_memory_mib": selective.get("current_memory_mib"),
            "selective_current_memory_saved_vs_newton_mib": None,
            "selective_nominal_tokens_per_sec": selective.get("nominal_tokens_per_sec"),
            "newton_nominal_tokens_per_sec": newton.get("nominal_tokens_per_sec"),
            "muon_nominal_tokens_per_sec": muon.get("nominal_tokens_per_sec"),
            "selective_final_val_loss": selective.get("final_val_loss"),
            "newton_final_val_loss": newton.get("final_val_loss"),
            "muon_final_val_loss": muon.get("final_val_loss"),
        }
        if (
            newton.get("completed")
            and selective.get("completed")
            and newton.get("peak_memory_mib") is not None
            and selective.get("peak_memory_mib") is not None
        ):
            row["selective_peak_memory_saved_vs_newton_mib"] = (
                float(newton["peak_memory_mib"]) - float(selective["peak_memory_mib"])
            )
            row["selective_peak_memory_ratio_vs_newton"] = (
                float(selective["peak_memory_mib"]) / float(newton["peak_memory_mib"])
            )
        if (
            newton.get("completed")
            and selective.get("completed")
            and newton.get("current_memory_mib") is not None
            and selective.get("current_memory_mib") is not None
        ):
            row["selective_current_memory_saved_vs_newton_mib"] = (
                float(newton["current_memory_mib"]) - float(selective["current_memory_mib"])
            )
        output.append(row)
    return output


def build_linear_memory_fit(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    budgets_mib = [40 * 1024, 48 * 1024, 80 * 1024]
    for method in METHOD_ORDER:
        points = [
            (float(row["batch_size"]), float(row["peak_memory_mib"]))
            for row in rows
            if row["method"] == method and row.get("completed") and row.get("peak_memory_mib") is not None
        ]
        if len(points) >= 2:
            intercept, slope = linear_fit(points)
        else:
            intercept, slope = None, None
        row: dict[str, object] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "peak_memory_intercept_mib": intercept,
            "peak_memory_slope_mib_per_batch": slope,
            "observed_completed_batch_count": len(points),
            "observed_max_batch": max((batch for batch, _ in points), default=None),
            "observed_max_peak_memory_mib": max((memory for _, memory in points), default=None),
        }
        for budget in budgets_mib:
            if intercept is None or slope is None:
                row[f"estimated_max_batch_under_{budget // 1024}gib"] = None
            else:
                row[f"estimated_max_batch_under_{budget // 1024}gib"] = math.floor((budget - intercept) / slope)
        output.append(row)
    return output


def build_data_quality_checks(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    raw_files = sorted(RAW_ROOT.glob("*.csv"))
    checks.append(
        {
            "check": "raw_export_file_count",
            "status": "pass" if len(raw_files) > 0 else "warn",
            "details": f"{len(raw_files)} CSV files found",
        }
    )
    batch_sizes = sorted({int(row["batch_size"]) for row in rows})
    expected = {(batch, method) for batch in batch_sizes for method in METHOD_ORDER}
    observed = {(int(row["batch_size"]), str(row["method"])) for row in rows}
    missing = sorted(expected - observed)
    checks.append(
        {
            "check": "batch_method_coverage",
            "status": "pass" if not missing else "warn",
            "details": f"{len(batch_sizes)} batch sizes x 3 methods present" if not missing else f"missing {missing}",
        }
    )
    missing_val_count = sum(1 for row in rows if row.get("last_val_step") is None)
    last_val_steps = sorted({row.get("last_val_step") for row in rows if row.get("last_val_step") is not None})
    checks.append(
        {
            "check": "last_validation_step",
            "status": "pass" if last_val_steps == [60] and missing_val_count == 0 else "warn",
            "details": f"observed last validation steps: {last_val_steps}; missing_val_count={missing_val_count}",
        }
    )
    missing_train_count = sum(1 for row in rows if row.get("last_train_log_step") is None)
    last_train_steps = sorted(
        {row.get("last_train_log_step") for row in rows if row.get("last_train_log_step") is not None}
    )
    checks.append(
        {
            "check": "last_train_log_step",
            "status": "pass" if last_train_steps == [110] else "warn",
            "details": f"observed last train-log steps: {last_train_steps}; missing_train_count={missing_train_count}",
        }
    )
    missing_time_count = sum(1 for row in rows if row.get("time_elapsed_step") is None)
    time_steps = sorted({row.get("time_elapsed_step") for row in rows if row.get("time_elapsed_step") is not None})
    checks.append(
        {
            "check": "last_time_elapsed_step",
            "status": "pass" if time_steps and missing_time_count == 0 else "warn",
            "details": f"observed last time_elapsed steps: {time_steps}; missing_time_count={missing_time_count}",
        }
    )
    return checks


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 1) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def write_notes(memory_rows, fit_rows, checks) -> None:
    fit_by_method = {row["method"]: row for row in fit_rows}
    completed_batches = {
        "muon": [row["batch_size"] for row in memory_rows if row.get("muon_completed")],
        "newton": [row["batch_size"] for row in memory_rows if row.get("newton_completed")],
        "selective_all_cproj": [row["batch_size"] for row in memory_rows if row.get("selective_completed")],
    }
    separation_batches = [
        row["batch_size"]
        for row in memory_rows
        if row.get("selective_completed") and not row.get("newton_completed")
    ]
    both_failed_batches = [
        row["batch_size"]
        for row in memory_rows
        if not row.get("selective_completed") and not row.get("newton_completed")
    ]
    both_completed_rows = [
        row
        for row in memory_rows
        if row.get("selective_completed")
        and row.get("newton_completed")
        and row.get("selective_peak_memory_saved_vs_newton_mib") not in (None, "")
    ]

    lines = [
        "# 24L fixed-memory batch boundary sweep",
        "",
        "Config: seed2024, 24L/D1024/H16, block_size=512, max_iters=120, lr_decay_iters=3000, muon_learning_rate=0.01.",
        "",
        "## Main result",
        "",
        f"- Exported batch sizes: {memory_rows[0]['batch_size']} through {memory_rows[-1]['batch_size']}.",
        f"- Completed Muon batches: {completed_batches['muon']}.",
        f"- Completed Newton-Muon batches: {completed_batches['newton']}.",
        f"- Completed Selective-All batches: {completed_batches['selective_all_cproj']}.",
    ]
    if separation_batches:
        lines.append(
            f"- Feasibility separation: Selective-All completed while Newton-Muon did not at batch size(s) {separation_batches}."
        )
    if both_failed_batches:
        lines.append(f"- Both Newton-Muon and Selective-All failed to progress past step 0 at batch size(s) {both_failed_batches}.")
    if both_completed_rows:
        lines.append(
            f"- Across batches where both completed, Selective-All peak-memory saving versus Newton-Muon is about {fmt(mean(float(row['selective_peak_memory_saved_vs_newton_mib']) for row in both_completed_rows))} MiB."
        )

    lines.extend(
        [
            "",
            "## Boundary interpretation",
            "",
            "This run is a fixed-budget feasibility test. Step-0-only runs are treated as failed/OOM-before-training-progress and should not be used for loss or memory-efficiency comparisons.",
            "",
            "Linear peak-memory fits use completed runs only:",
        ]
    )
    for method in METHOD_ORDER:
        row = fit_by_method[method]
        if row.get("peak_memory_slope_mib_per_batch") is None:
            lines.append(
                f"- {METHOD_LABELS[method]}: not enough completed points for a fit; completed batch count {row['observed_completed_batch_count']}."
            )
        else:
            lines.append(
                f"- {METHOD_LABELS[method]}: intercept {fmt(row['peak_memory_intercept_mib'])} MiB, slope {fmt(row['peak_memory_slope_mib_per_batch'])} MiB/batch, estimated 40GiB max batch {row['estimated_max_batch_under_40gib']}."
            )

    lines.extend(["", "## Data quality"])
    for check in checks:
        lines.append(f"- {check['check']}: {check['status']} ({check['details']})")
    lines.extend(
        [
            "",
            "## Saved artifacts",
            "",
            "- summaries/run_summary.csv",
            "- summaries/batch_memory_summary.csv",
            "- summaries/linear_memory_fit.csv",
            "- summaries/data_quality_checks.csv",
        ]
    )
    NOTES_ROOT.mkdir(parents=True, exist_ok=True)
    (NOTES_ROOT / "RESULT_NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fixed-memory batch-boundary W&B exports.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    set_output_root(Path(args.output_root))

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    series_by_key = read_wandb_exports()
    run_rows = build_run_summary(series_by_key)
    memory_rows = build_memory_summary(run_rows)
    fit_rows = build_linear_memory_fit(run_rows)
    checks = build_data_quality_checks(run_rows)

    write_csv(SUMMARY_ROOT / "run_summary.csv", run_rows)
    write_csv(SUMMARY_ROOT / "batch_memory_summary.csv", memory_rows)
    write_csv(SUMMARY_ROOT / "linear_memory_fit.csv", fit_rows)
    write_csv(SUMMARY_ROOT / "data_quality_checks.csv", checks)
    write_notes(memory_rows, fit_rows, checks)

    print(f"Wrote summaries to {SUMMARY_ROOT}")
    print(f"Wrote notes to {NOTES_ROOT / 'RESULT_NOTES.md'}")


if __name__ == "__main__":
    main()
