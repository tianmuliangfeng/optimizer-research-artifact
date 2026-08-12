"""Audit experiment-45 pilot/formal W&B exports and build a provisional R1 panel.

W&B is treated as a non-authoritative external mirror.  Claim eligibility is
intentionally false until the sealed local experiment-45 artifacts are audited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "experiment45_mousse_wandb_review_v1"
EXPECTED_METRICS = (
    "val/loss",
    "train/loss_step",
    "tokens/seen",
    "memory/optimizer_state_mib",
    "memory/peak_allocated_mib",
    "lr/auxiliary",
    "lr/matrix",
)
METRIC_ORDER = {name: index for index, name in enumerate(EXPECTED_METRICS)}
MATRIX_LRS = {"080": 0.012, "100": 0.015, "120": 0.018}
TOTAL_STEPS = {"pilot": 1000, "formal": 6200}
TOKENS_PER_STEP = 524_288
SEEDS = (2024, 2025, 2026)
T_CRITICAL_DF2_95 = 4.302652729696142
RUN_PATTERN = re.compile(
    r"^(?P<run>mainconf_r1_mousse_(?P<phase>pilot|formal)_"
    r"mousse_lr(?P<lr_code>080|100|120)_seed(?P<seed>\d+)_"
    r"(?P<batch>\d{8}T\d{6}\+0000)) - (?P<metric>.+?)"
    r"(?P<companion>__(?:MIN|MAX))?$"
)
HISTORICAL_METHODS = {
    "diag": ("diag", "Newton-Muon diag"),
    "none": ("none", "Newton-Muon none"),
    "block4": ("block4", "Newton-Muon block4 (original)"),
    "muon": ("muon", "Muon"),
    "moonlight_r1scale": ("moonlight", "Moonlight Muon"),
    "normuon_r1scale": ("normuon", "NorMuon"),
    "adamw_low": ("adamw", "AdamW"),
}
DISPLAY_NAMES = {
    "diag": "Newton-Muon diag",
    "none": "Newton-Muon none",
    "mousse": "Mousse-R1",
    "block4": "Newton-Muon block4 (original)",
    "muon": "Muon",
    "moonlight": "Moonlight Muon",
    "normuon": "NorMuon",
    "adamw": "AdamW",
}
METHOD_ORDER = ("diag", "block4", "none", "mousse", "moonlight", "muon", "normuon", "adamw")
COMPARISONS = (
    ("selective_diag_minus_mousse", "diag", "mousse", "primary"),
    ("selective_none_minus_mousse", "none", "mousse", "primary"),
    ("mousse_minus_muon", "mousse", "muon", "anchor"),
    ("mousse_minus_original_newton_muon", "mousse", "block4", "anchor"),
    ("mousse_minus_moonlight", "mousse", "moonlight", "external_background"),
    ("mousse_minus_normuon", "mousse", "normuon", "external_background"),
    ("mousse_minus_adamw", "mousse", "adamw", "external_background"),
)


class AuditError(RuntimeError):
    """Raised when an export violates the frozen experiment-45 contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(raw: str, context: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{context}: non-numeric value {raw!r}") from exc
    if not math.isfinite(value):
        raise AuditError(f"{context}: non-finite value {raw!r}")
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def expected_steps(phase: str, metric: str) -> set[int]:
    terminal = TOTAL_STEPS[phase]
    if metric == "val/loss":
        return set(range(0, terminal + 1, 100))
    if metric == "train/loss_step":
        return set(range(20, terminal + 1, 20))
    if metric in ("tokens/seen", "lr/auxiliary", "lr/matrix"):
        return set(range(0, terminal + 1, 20))
    return {terminal}


def parse_export(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Step" not in reader.fieldnames:
            raise AuditError(f"{path}: missing Step column")
        parsed_fields: dict[str, re.Match[str]] = {}
        for field in reader.fieldnames:
            if field == "Step":
                continue
            match = RUN_PATTERN.match(field)
            if not match:
                raise AuditError(f"{path}: unexpected column {field!r}")
            parsed_fields[field] = match
        primary = {
            field: match for field, match in parsed_fields.items() if match.group("companion") is None
        }
        if not primary:
            raise AuditError(f"{path}: no primary metric columns")
        phases = {match.group("phase") for match in primary.values()}
        metrics = {match.group("metric") for match in primary.values()}
        if len(phases) != 1 or len(metrics) != 1:
            raise AuditError(f"{path}: mixed phases or metrics")
        phase = phases.pop()
        metric = metrics.pop()
        if metric not in EXPECTED_METRICS:
            raise AuditError(f"{path}: unexpected metric {metric!r}")
        values: dict[str, dict[int, float]] = defaultdict(dict)
        redundant_checks = 0
        row_count = 0
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            step_value = number(row["Step"], f"{path}:{row_number}:Step")
            if not step_value.is_integer():
                raise AuditError(f"{path}:{row_number}: non-integral step {step_value}")
            step = int(step_value)
            for field, match in primary.items():
                raw = (row.get(field) or "").strip()
                companions = [field + "__MIN", field + "__MAX"]
                if not raw:
                    if any((row.get(name) or "").strip() for name in companions):
                        raise AuditError(f"{path}:{row_number}: companion exists without primary")
                    continue
                value = number(raw, f"{path}:{row_number}:{field}")
                run = match.group("run")
                if step in values[run]:
                    raise AuditError(f"{path}: duplicate {run} step {step}")
                values[run][step] = value
                for companion in companions:
                    companion_raw = (row.get(companion) or "").strip()
                    if not companion_raw:
                        raise AuditError(f"{path}:{row_number}: missing {companion}")
                    if number(companion_raw, f"{path}:{row_number}:{companion}") != value:
                        raise AuditError(f"{path}:{row_number}: {companion} differs from primary")
                    redundant_checks += 1
        for run, series in values.items():
            observed = set(series)
            expected = expected_steps(phase, metric)
            if observed != expected:
                raise AuditError(
                    f"{path}: {run} {metric} step mismatch; "
                    f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
                )
        metadata = {}
        for match in primary.values():
            metadata[match.group("run")] = {
                "phase": match.group("phase"),
                "lr_code": match.group("lr_code"),
                "matrix_lr": MATRIX_LRS[match.group("lr_code")],
                "seed": int(match.group("seed")),
                "batch": match.group("batch"),
            }
        return {
            "path": path,
            "phase": phase,
            "metric": metric,
            "values": dict(values),
            "metadata": metadata,
            "row_count": row_count,
            "redundant_checks": redundant_checks,
        }


def validate_run_sets(exports: dict[tuple[str, str], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    run_metadata: dict[str, dict[str, Any]] = {}
    for phase in ("pilot", "formal"):
        run_sets = [set(exports[(phase, metric)]["values"]) for metric in EXPECTED_METRICS]
        if any(run_set != run_sets[0] for run_set in run_sets[1:]):
            raise AuditError(f"{phase}: the seven exports contain different run sets")
        for run in run_sets[0]:
            metadata = exports[(phase, "val/loss")]["metadata"][run]
            run_metadata[run] = metadata
        observed = {
            (item["lr_code"], item["seed"]) for run, item in run_metadata.items() if item["phase"] == phase
        }
        expected = (
            {("080", 2026), ("100", 2026), ("120", 2026)}
            if phase == "pilot"
            else {("100", seed) for seed in SEEDS}
        )
        if observed != expected:
            raise AuditError(f"{phase}: run contract mismatch: observed={sorted(observed)}")
    return run_metadata


def trapezoid_auc(series: dict[int, float]) -> float:
    points = sorted(series.items())
    span = points[-1][0] - points[0][0]
    return sum(
        (right_step - left_step) * (left_value + right_value) / 2
        for (left_step, left_value), (right_step, right_value) in zip(points, points[1:])
    ) / span


def summarize_run(
    run: str,
    metadata: dict[str, Any],
    exports: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    phase = metadata["phase"]
    series = {metric: exports[(phase, metric)]["values"][run] for metric in EXPECTED_METRICS}
    validation = sorted(series["val/loss"].items())
    tail3 = [value for _, value in validation[-3:]]
    tail5 = [value for _, value in validation[-5:]]
    terminal = TOTAL_STEPS[phase]
    tokens = series["tokens/seen"]
    for step, value in tokens.items():
        if value != step * TOKENS_PER_STEP:
            raise AuditError(f"{run}: token budget mismatch at step {step}: {value}")
    expected_matrix_lr = metadata["matrix_lr"]
    if max(series["lr/matrix"].values()) != expected_matrix_lr:
        raise AuditError(f"{run}: matrix LR maximum does not match the frozen cell")
    if max(series["lr/auxiliary"].values()) != 0.0036:
        raise AuditError(f"{run}: auxiliary LR maximum is not 0.0036")
    return {
        "phase": phase,
        "method": "mousse",
        "run_name": run,
        "seed": metadata["seed"],
        "seed_role": "tuning_seed" if metadata["seed"] == 2026 else "confirmatory_seed",
        "cell_id": f"mousse_lr{metadata['lr_code']}",
        "matrix_lr": metadata["matrix_lr"],
        "batch": metadata["batch"],
        "total_steps": terminal,
        "total_tokens": int(tokens[terminal]),
        "initial_val_loss": validation[0][1],
        "final_val_loss": validation[-1][1],
        "best_val_loss": min(value for _, value in validation),
        "best_val_step": min(validation, key=lambda item: item[1])[0],
        "tail3_val_loss_mean": statistics.mean(tail3),
        "tail5_val_loss_mean": statistics.mean(tail5),
        "tail5_val_loss_sd": statistics.stdev(tail5),
        "normalized_val_auc": trapezoid_auc(series["val/loss"]),
        "final_train_loss_step": series["train/loss_step"][terminal],
        "max_auxiliary_lr": max(series["lr/auxiliary"].values()),
        "max_matrix_lr": max(series["lr/matrix"].values()),
        "optimizer_state_mib": series["memory/optimizer_state_mib"][terminal],
        "peak_allocated_mib": series["memory/peak_allocated_mib"][terminal],
        "timing_eligible": False,
        "evidence_source": "experiment45_wandb_export_pending_local_artifact",
    }


def normalize_historical(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    normalized = []
    for row in rows:
        if row["method"] not in HISTORICAL_METHODS:
            continue
        method, display_name = HISTORICAL_METHODS[row["method"]]
        normalized.append(
            {
                "method": method,
                "display_name": display_name,
                "run_name": row["run_name"],
                "seed": int(row["seed"]),
                "seed_role": row.get("seed_role", "historical_frozen"),
                "initial_val_loss": float(row["initial_val_loss"]),
                "final_val_loss": float(row["final_val_loss"]),
                "best_val_loss": float(row["best_val_loss"]),
                "tail5_val_loss_mean": float(row["tail5_val_loss_mean"]),
                "normalized_val_auc": float(row["normalized_val_auc"]),
                "final_train_loss_step": float(row["final_train_loss_step"]),
                "optimizer_state_mib": float(row["optimizer_state_mib"]),
                "peak_allocated_mib": float(row["peak_memory_mib"]),
                "timing_eligible": False,
                "evidence_source": "frozen_experiment19_unified_summary",
            }
        )
    observed = {(row["method"], row["seed"]) for row in normalized}
    required = {(method, seed) for method, _ in HISTORICAL_METHODS.values() for seed in SEEDS}
    if observed != required:
        raise AuditError(
            "historical seven-method coverage mismatch; "
            f"missing={sorted(required - observed)}, unexpected={sorted(observed - required)}"
        )
    return normalized


def aggregate_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        finals = [float(row["final_val_loss"]) for row in selected]
        tail5 = [float(row["tail5_val_loss_mean"]) for row in selected]
        aucs = [float(row["normalized_val_auc"]) for row in selected]
        output.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "n_seeds": len(selected),
                "final_val_loss_mean": statistics.mean(finals),
                "final_val_loss_sample_sd": statistics.stdev(finals),
                "tail5_val_loss_mean": statistics.mean(tail5),
                "normalized_val_auc_mean": statistics.mean(aucs),
                "optimizer_state_mib_mean": statistics.mean(float(row["optimizer_state_mib"]) for row in selected),
                "peak_allocated_mib_mean": statistics.mean(float(row["peak_allocated_mib"]) for row in selected),
            }
        )
    for rank, row in enumerate(sorted(output, key=lambda item: item["final_val_loss_mean"]), start=1):
        row["final_loss_rank"] = rank
    return output


def paired_contrasts(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {(row["method"], int(row["seed"])): row for row in rows}
    seed_rows = []
    aggregate_rows = []
    for comparison, left, right, role in COMPARISONS:
        differences = []
        for seed in SEEDS:
            left_loss = float(indexed[(left, seed)]["final_val_loss"])
            right_loss = float(indexed[(right, seed)]["final_val_loss"])
            difference = left_loss - right_loss
            differences.append(difference)
            seed_rows.append(
                {
                    "comparison": comparison,
                    "role": role,
                    "seed": seed,
                    "left_method": left,
                    "right_method": right,
                    "left_final_val_loss": left_loss,
                    "right_final_val_loss": right_loss,
                    "difference_left_minus_right": difference,
                    "lower_is_better": True,
                }
            )
        mean = statistics.mean(differences)
        sample_sd = statistics.stdev(differences)
        half_width = T_CRITICAL_DF2_95 * sample_sd / math.sqrt(len(differences))
        lower, upper = mean - half_width, mean + half_width
        aggregate_rows.append(
            {
                "comparison": comparison,
                "role": role,
                "left_method": left,
                "right_method": right,
                "n_seeds": len(differences),
                "paired_mean_difference": mean,
                "paired_sample_sd": sample_sd,
                "paired_t_ci95_lower": lower,
                "paired_t_ci95_upper": upper,
                "left_better_seed_count": sum(value < 0 for value in differences),
                "tie_seed_count": sum(value == 0 for value in differences),
                "right_better_seed_count": sum(value > 0 for value in differences),
                "practical_equivalence_margin": 0.002,
                "point_within_equivalence_margin": abs(mean) <= 0.002,
                "ci_within_equivalence_margin": lower >= -0.002 and upper <= 0.002,
            }
        )
    return seed_rows, aggregate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, nargs="+", required=True)
    parser.add_argument("--historical-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = [path.resolve() for path in args.exports]
    if len(paths) != 14 or len(set(paths)) != 14:
        raise AuditError("exactly 14 unique W&B CSV exports are required")
    output = args.output_dir.resolve()
    if output.exists():
        raise AuditError(f"output directory already exists: {output}")
    output.mkdir(parents=True)

    parsed: dict[tuple[str, str], dict[str, Any]] = {}
    inventory = []
    for path in paths:
        if not path.is_file():
            raise AuditError(f"missing export: {path}")
        item = parse_export(path)
        key = (item["phase"], item["metric"])
        if key in parsed:
            raise AuditError(f"duplicate export for {key}")
        parsed[key] = item
        inventory.append(
            {
                "phase": item["phase"],
                "metric": item["metric"],
                "source_file": path.name,
                "source_sha256": sha256_file(path),
                "source_bytes": path.stat().st_size,
                "csv_rows": item["row_count"],
                "run_count": len(item["values"]),
                "primary_value_count": sum(len(series) for series in item["values"].values()),
                "redundant_min_max_checks": item["redundant_checks"],
            }
        )
    expected_keys = {(phase, metric) for phase in ("pilot", "formal") for metric in EXPECTED_METRICS}
    if set(parsed) != expected_keys:
        raise AuditError(f"export metric coverage mismatch: {sorted(set(parsed))}")
    metadata = validate_run_sets(parsed)

    raw_dir = output / "raw"
    raw_dir.mkdir()
    for path in paths:
        shutil.copy2(path, raw_dir / path.name)

    long_rows = {"pilot": [], "formal": []}
    for (phase, metric), item in parsed.items():
        for run, series in item["values"].items():
            info = metadata[run]
            for step, value in sorted(series.items()):
                long_rows[phase].append(
                    {
                        "phase": phase,
                        "run_name": run,
                        "seed": info["seed"],
                        "cell_id": f"mousse_lr{info['lr_code']}",
                        "matrix_lr": info["matrix_lr"],
                        "step": step,
                        "metric": metric,
                        "value": value,
                        "source_export": item["path"].name,
                    }
                )
    for rows in long_rows.values():
        rows.sort(key=lambda row: (row["run_name"], row["step"], METRIC_ORDER[row["metric"]]))

    summaries = [summarize_run(run, info, parsed) for run, info in metadata.items()]
    pilot_summaries = sorted(
        (row for row in summaries if row["phase"] == "pilot"), key=lambda row: row["matrix_lr"]
    )
    formal_summaries = sorted(
        (row for row in summaries if row["phase"] == "formal"), key=lambda row: row["seed"]
    )
    ranked_pilot = sorted(pilot_summaries, key=lambda row: row["final_val_loss"])
    center = next(row for row in pilot_summaries if row["cell_id"] == "mousse_lr100")
    selected = center if center["final_val_loss"] <= ranked_pilot[0]["final_val_loss"] + 0.002 else ranked_pilot[0]
    selection = {
        "status": "reconstructed_from_wandb_pending_local_manifest",
        "protocol": "mousse_r1_pilot_selection_v1",
        "selection_endpoint": "step-1000 validation loss",
        "center_tie_margin": 0.002,
        "selected_cell_id": selected["cell_id"],
        "selected_matrix_lr": selected["matrix_lr"],
        "best_observed_cell_id": ranked_pilot[0]["cell_id"],
        "best_observed_final_val_loss": ranked_pilot[0]["final_val_loss"],
        "center_final_val_loss": center["final_val_loss"],
        "center_gap_from_best": center["final_val_loss"] - ranked_pilot[0]["final_val_loss"],
        "ranked_cells": [
            {
                "cell_id": row["cell_id"],
                "matrix_lr": row["matrix_lr"],
                "final_val_loss": row["final_val_loss"],
            }
            for row in ranked_pilot
        ],
    }
    if selection["selected_cell_id"] != "mousse_lr100":
        raise AuditError("W&B-reconstructed pilot selection is not the formal lr100 cell")

    historical_path = args.historical_summary.resolve()
    historical = normalize_historical(historical_path)
    mousse_panel = []
    for row in formal_summaries:
        mousse_panel.append(
            {
                **row,
                "display_name": DISPLAY_NAMES["mousse"],
            }
        )
    unified = sorted(
        [*historical, *mousse_panel],
        key=lambda row: (int(row["seed"]), METHOD_ORDER.index(row["method"])),
    )
    if len(unified) != 24:
        raise AuditError(f"expected 24 rows in the provisional eight-method panel, observed {len(unified)}")
    aggregates = aggregate_methods(unified)
    seed_deltas, paired = paired_contrasts(unified)

    summary_fields = [
        "phase", "method", "run_name", "seed", "seed_role", "cell_id", "matrix_lr", "batch",
        "total_steps", "total_tokens", "initial_val_loss", "final_val_loss", "best_val_loss",
        "best_val_step", "tail3_val_loss_mean", "tail5_val_loss_mean", "tail5_val_loss_sd",
        "normalized_val_auc", "final_train_loss_step", "max_auxiliary_lr", "max_matrix_lr",
        "optimizer_state_mib", "peak_allocated_mib", "timing_eligible", "evidence_source",
    ]
    long_fields = ["phase", "run_name", "seed", "cell_id", "matrix_lr", "step", "metric", "value", "source_export"]
    write_csv(output / "wandb_export_inventory.csv", sorted(inventory, key=lambda row: (row["phase"], METRIC_ORDER[row["metric"]])), list(inventory[0]))
    write_csv(output / "mousse_pilot_history_long.csv", long_rows["pilot"], long_fields)
    write_csv(output / "mousse_formal_history_long.csv", long_rows["formal"], long_fields)
    write_csv(output / "mousse_pilot_run_summary.csv", pilot_summaries, summary_fields)
    write_csv(output / "mousse_formal_run_summary.csv", formal_summaries, summary_fields)
    write_csv(
        output / "r1_provisional_eight_method_run_summary.csv",
        unified,
        [
            "method", "display_name", "run_name", "seed", "seed_role", "initial_val_loss",
            "final_val_loss", "best_val_loss", "tail5_val_loss_mean", "normalized_val_auc",
            "final_train_loss_step", "optimizer_state_mib", "peak_allocated_mib", "timing_eligible",
            "evidence_source",
        ],
    )
    write_csv(output / "r1_provisional_eight_method_aggregate.csv", aggregates, list(aggregates[0]))
    write_csv(output / "mousse_provisional_paired_seed_deltas.csv", seed_deltas, list(seed_deltas[0]))
    write_csv(output / "mousse_provisional_paired_aggregate.csv", paired, list(paired[0]))
    (output / "pilot_selection_wandb_reconstruction.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    mousse_aggregate = next(row for row in aggregates if row["method"] == "mousse")
    ranking = sorted(aggregates, key=lambda row: row["final_val_loss_mean"])
    paired_lookup = {row["comparison"]: row for row in paired}
    report = [
        "# Experiment 45 W&B pilot/formal review",
        "",
        "## Audit status",
        "",
        "- W&B export audit: passed.",
        "- Phase/metric coverage: 2 phases x 7 metrics = 14/14 exports.",
        "- Runs: 3 pilot + 3 formal; formal seeds are exactly 2024/2025/2026.",
        f"- Redundant primary/MIN/MAX checks: {sum(row['redundant_min_max_checks'] for row in inventory)}; all exact.",
        "- Step grids, LR maxima, and token budgets match the frozen experiment-45 contract.",
        "- Evidence level: provisional W&B mirror. The sealed local artifacts remain required and authoritative.",
        "- Claim eligible: no, pending the local pilot/formal manifests and experiment-45 unified analyzer.",
        "- Formal timing is ineligible by design because two-GPU concurrency was used.",
        "",
        "## Pilot reconstruction",
        "",
    ]
    for row in ranked_pilot:
        report.append(f"- {row['cell_id']} (matrix LR {row['matrix_lr']:.3f}): step-1000 val/loss {row['final_val_loss']:.4f}")
    report.extend(
        [
            f"- Lowest observed cell: {selection['best_observed_cell_id']}.",
            f"- Center gap from the lowest cell: {selection['center_gap_from_best']:.4f} < 0.002.",
            "- Predeclared selection therefore chooses mousse_lr100 (matrix LR 0.015).",
            "",
            "## Formal Mousse result",
            "",
        ]
    )
    for row in formal_summaries:
        report.append(
            f"- seed {row['seed']}: final {row['final_val_loss']:.4f}; "
            f"tail-5 {row['tail5_val_loss_mean']:.5f}; normalized AUC {row['normalized_val_auc']:.6f}."
        )
    report.extend(
        [
            f"- Three-seed final mean +/- sample SD: {mousse_aggregate['final_val_loss_mean']:.6f} +/- {mousse_aggregate['final_val_loss_sample_sd']:.6f}.",
            f"- Three-seed tail-5 mean: {mousse_aggregate['tail5_val_loss_mean']:.6f}.",
            f"- Three-seed normalized validation AUC mean: {mousse_aggregate['normalized_val_auc_mean']:.6f}.",
            f"- Optimizer state: {mousse_aggregate['optimizer_state_mib_mean']:.3f} MiB; peak allocated: {mousse_aggregate['peak_allocated_mib_mean']:.1f} MiB.",
            "",
            "## Provisional controlled-R1 position",
            "",
        ]
    )
    for rank, row in enumerate(ranking, start=1):
        report.append(
            f"{rank}. {row['display_name']}: {row['final_val_loss_mean']:.6f} +/- {row['final_val_loss_sample_sd']:.6f}"
        )
    report.extend(["", "## Pre-registered paired endpoint contrasts", ""])
    for name, _, _, role in COMPARISONS:
        row = paired_lookup[name]
        report.append(
            f"- {name} ({role}): mean {row['paired_mean_difference']:+.6f}, "
            f"95% paired-t CI [{row['paired_t_ci95_lower']:+.6f}, {row['paired_t_ci95_upper']:+.6f}], "
            f"left better in {row['left_better_seed_count']}/3 seeds."
        )
    report.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Mousse is stronger than Muon, Moonlight, NorMuon, and AdamW at the final endpoint,",
            "but it does not erase the selective-routing result: diag is clearly better, while none",
            "is slightly better in all three seeds and its point estimate lies inside the +/-0.002",
            "practical margin. Mousse is worse than original block4 and uses substantially more",
            "optimizer state. These statements are provisional until exact local-artifact matching.",
        ]
    )
    (output / "WANDB_REVIEW.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    output_files = [path for path in output.rglob("*") if path.is_file()]
    manifest = {
        "schema_version": SCHEMA,
        "status": "passed_pending_local_artifact",
        "synthetic": False,
        "claim_eligible": False,
        "scientific_role": "provisional_external_wandb_mirror",
        "local_artifacts_authoritative": True,
        "formal_run_count": len(formal_summaries),
        "pilot_run_count": len(pilot_summaries),
        "formal_seed_coverage": sorted(row["seed"] for row in formal_summaries),
        "formal_budget_steps": 6200,
        "formal_budget_tokens": 3_250_585_600,
        "selected_cell_from_wandb": selection["selected_cell_id"],
        "selected_matrix_lr_from_wandb": selection["selected_matrix_lr"],
        "min_max_checks": sum(row["redundant_min_max_checks"] for row in inventory),
        "historical_summary": {
            "path": str(historical_path),
            "sha256": sha256_file(historical_path),
        },
        "source_exports": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
        "outputs": [
            {
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output_files)
        ],
        "pending_requirements": [
            "sealed experiment-45 artifact directory containing pilot_manifest.json",
            "three formal_summary.csv plus sibling formal_manifest.json files",
            "pilot_selection.json and local verification certificate",
            "experiment-45 controller terminal completion output or batch paths",
            "formal checkpoints/checkpoint metadata and W&B upload status contained in the sealed artifacts",
        ],
    }
    manifest_path = output / "wandb_review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Experiment 45 W&B review passed: pilot={len(pilot_summaries)} "
        f"formal={len(formal_summaries)} selected={selection['selected_cell_id']} "
        f"mousse_mean={mousse_aggregate['final_val_loss_mean']:.6f}"
    )


if __name__ == "__main__":
    main()
