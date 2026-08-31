#!/usr/bin/env python3
"""Build the EX55 fresh-seed and repaired 2024/2025/2027 panels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


CANONICAL_METHODS = (
    "block4", "diag", "none", "muon", "adamw", "normuon", "moonlight",
    "mousse", "malt", "malter_eq17",
)
ANALYZER_VERSION = "2026-08-19.1"
# Marker consumed by the command wrapper/controller when deciding whether an
# existing source snapshot already contains the formal-metrics tail-5 repair.
FORMAL_METRICS_TAIL5_LINEAGE = True
ACCEPTED_STATUS_PREFIXES = ("completed_valid",)
DISPLAY = {
    "block4": "Local block-4", "diag": "Local diag", "none": "Local none",
    "muon": "Muon", "adamw": "AdamW", "normuon": "NorMuon",
    "moonlight": "Moonlight", "mousse": "Mousse", "malt": "MALT",
    "malter_eq17": "MALTER-Eq17",
}
CELL_ALIASES = {
    "block4": {"block4"}, "diag": {"diag"}, "none": {"none"}, "muon": {"muon"},
    "adamw": {"adamw", "adamw_low"},
    "normuon": {"normuon", "normuon_r1scale"},
    "moonlight": {"moonlight", "moonlight_muon", "moonlight_r1scale"},
    "mousse": {"mousse", "mousse_lr100"},
    "malt": {"malt", "malt_lr0125"},
    "malter_eq17": {"malter_eq17", "malter_eq17_lr015"},
}
WORKER_METHOD_LABELS = {"moonlight": "moonlight_muon"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--historical-panel", type=Path, required=True)
    parser.add_argument("--formal-units", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-amendment", type=Path)
    parser.add_argument("--historical-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_evidence(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"{label} is not a file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite: {value!r}")
    return result


def summaries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload.get("summaries"), list):
        rows.extend(item for item in payload["summaries"] if isinstance(item, dict))
    if isinstance(payload.get("summary"), dict) and payload["summary"] not in rows:
        rows.append(payload["summary"])
    return rows


def pick_summary(payload: dict[str, Any], method: str, selected_cell: str) -> dict[str, Any]:
    candidates = summaries(payload)
    matched = []
    for row in candidates:
        observed_method = str(row.get("method", "")).strip().lower()
        observed_cell = str(row.get("cell_id", "")).strip().lower()
        exact = (
            observed_method == WORKER_METHOD_LABELS.get(method, method)
            and (
                observed_cell == selected_cell
                if observed_cell
                else selected_cell == method
            )
        )
        if exact:
            matched.append(row)
    if len(matched) != 1:
        raise RuntimeError(
            f"{method}: expected one matching summary for {selected_cell}, observed {len(matched)}"
        )
    row = matched[0]
    if row.get("evidence_valid") is not True:
        raise RuntimeError(f"{method}: formal summary is not locally evidence-valid")
    if int(row.get("controlled_seed", row.get("seed", -1))) != 2027:
        raise RuntimeError(f"{method}: formal summary is not seed 2027")
    if int(row.get("total_steps", row.get("final_val_step", -1))) != 6200:
        raise RuntimeError(f"{method}: formal summary is not the 6200-step protocol")
    return row


def exact_summary_identity(row: dict[str, Any], method: str, selected_cell: str) -> bool:
    observed_method = str(row.get("method", "")).strip().lower()
    observed_cell = str(row.get("cell_id", "")).strip().lower()
    return (
        observed_method == WORKER_METHOD_LABELS.get(method, method)
        and (observed_cell == selected_cell if observed_cell else selected_cell == method)
    )


def close_float(left: Any, right: Any, label: str) -> None:
    left_value = finite_float(left, f"{label} left")
    right_value = finite_float(right, f"{label} right")
    if not math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"{label} mismatch: {left_value} != {right_value}")


def resolve_formal_child_artifacts(
    manifest: Path,
    row: dict[str, Any],
    method: str,
    selected_cell: str,
    *,
    seed: int,
    formal_steps: int,
) -> dict[str, Path]:
    """Resolve the unique accepted formal child that owns the full metrics CSV.

    Aggregate manifests are heterogeneous across the inherited runner families.
    The scientific evidence needed for tail-5 lives in the accepted child run,
    so this resolver binds the aggregate row to exactly one child by immutable
    method/cell, seed, initialization, final-step, and final-loss metadata.
    """

    candidates: list[Path] = []
    run_name = str(row.get("run_name", "")).strip()
    if run_name:
        candidates.append(manifest.parent / run_name)
    checkpoint_raw = str(row.get("checkpoint_path", "")).strip()
    if checkpoint_raw:
        checkpoint_parent = Path(checkpoint_raw).expanduser().resolve().parent
        candidates.extend((checkpoint_parent, checkpoint_parent.parent))
    candidates.extend(path.parent for path in manifest.parent.glob("*/run_manifest.json"))
    unique = {path.expanduser().resolve(): path.expanduser().resolve() for path in candidates}

    matched: list[dict[str, Path]] = []
    for run_dir in unique.values():
        child_manifest = run_dir / "run_manifest.json"
        artifact_pairs = (
            (run_dir / "r1_summary.json", run_dir / "r1_metrics.csv"),
            (run_dir / "summary.json", run_dir / "metrics.csv"),
        )
        complete_pairs = [
            (summary_path, metrics_path)
            for summary_path, metrics_path in artifact_pairs
            if summary_path.is_file() and metrics_path.is_file()
        ]
        if not child_manifest.is_file() or len(complete_pairs) != 1:
            continue
        summary_path, metrics_path = complete_pairs[0]
        try:
            child_manifest_payload = read_json(child_manifest)
            child_summary = read_json(summary_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if not str(child_manifest_payload.get("status", "")).startswith(ACCEPTED_STATUS_PREFIXES):
            continue
        if not isinstance(child_summary, dict) or not exact_summary_identity(
            child_summary, method, selected_cell,
        ):
            continue
        if child_summary.get("evidence_valid") not in (None, True):
            continue
        child_seed = int(child_summary.get("controlled_seed", child_summary.get("seed", -1)))
        child_steps_raw = child_summary.get("total_steps", child_summary.get("final_val_step"))
        if child_seed != seed:
            continue
        if child_steps_raw is not None and int(child_steps_raw) != formal_steps:
            continue
        if child_summary.get("init_sha256") != row.get("init_sha256"):
            continue
        try:
            close_float(
                child_summary.get("final_val_loss"), row.get("final_val_loss"),
                f"{method} child/aggregate final validation",
            )
        except RuntimeError:
            continue
        aggregate_checkpoint = str(row.get("checkpoint_path", "")).strip()
        child_checkpoint = str(child_summary.get("checkpoint_path", "")).strip()
        if aggregate_checkpoint and child_checkpoint:
            if Path(aggregate_checkpoint).expanduser().resolve() != Path(child_checkpoint).expanduser().resolve():
                continue
        matched.append({
            "run_dir": run_dir,
            "manifest": child_manifest,
            "summary": summary_path,
            "metrics": metrics_path,
        })

    if len(matched) != 1:
        raise RuntimeError(
            f"{method}: expected one accepted formal child with complete metrics lineage, "
            f"found {len(matched)} under {manifest.parent}"
        )
    return matched[0]


def formal_metrics_tail5_evidence(
    manifest: Path,
    row: dict[str, Any],
    method: str,
    selected_cell: str,
    *,
    seed: int,
    formal_steps: int,
    validation_every: int,
) -> dict[str, Any]:
    """Reconstruct the formal tail-5 from a hash-bound accepted child metrics CSV."""

    if validation_every <= 0 or formal_steps < 4 * validation_every:
        raise RuntimeError("invalid formal validation cadence for tail-5 reconstruction")
    child = resolve_formal_child_artifacts(
        manifest, row, method, selected_cell, seed=seed, formal_steps=formal_steps,
    )
    metrics_path = child["metrics"]
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"event", "step", "loss"}
        observed_fields = set(reader.fieldnames or [])
        if not required_fields.issubset(observed_fields):
            raise RuntimeError(
                f"{method}: unsupported formal metrics schema {sorted(observed_fields)}; "
                "required event,step,loss"
            )
        records = list(reader)

    validation_rows: list[tuple[int, float]] = []
    for index, record in enumerate(records, start=2):
        if str(record.get("event", "")).strip().lower() != "validation":
            continue
        try:
            step = int(str(record.get("step", "")).strip())
        except ValueError as exc:
            raise RuntimeError(f"{method}: invalid validation step on metrics line {index}") from exc
        loss = finite_float(record.get("loss"), f"{method} metrics line {index} loss")
        validation_rows.append((step, loss))
    if not validation_rows:
        raise RuntimeError(f"{method}: formal metrics contain no validation rows")

    by_step: dict[int, float] = {}
    duplicate_rows_collapsed = 0
    for step, loss in validation_rows:
        if step in by_step:
            if not math.isclose(by_step[step], loss, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(
                    f"{method}: conflicting duplicate validation values at step {step}: "
                    f"{by_step[step]} vs {loss}"
                )
            duplicate_rows_collapsed += 1
            continue
        by_step[step] = loss

    required_tail_steps = [
        formal_steps - offset * validation_every for offset in range(4, -1, -1)
    ]
    missing = [step for step in required_tail_steps if step not in by_step]
    if missing:
        raise RuntimeError(f"{method}: formal metrics missing required tail-5 steps {missing}")
    if 0 not in by_step:
        raise RuntimeError(f"{method}: formal metrics missing step-0 validation")
    if max(by_step) != formal_steps:
        raise RuntimeError(
            f"{method}: formal metrics final validation step is {max(by_step)}, expected {formal_steps}"
        )

    tail_losses = [by_step[step] for step in required_tail_steps]
    tail_mean = statistics.mean(tail_losses)
    close_float(by_step[formal_steps], row.get("final_val_loss"), f"{method} metrics/aggregate final validation")

    aggregate_initial = None
    for name in ("initial_val_loss", "val_loss_step_0"):
        if row.get(name) not in (None, ""):
            aggregate_initial = row[name]
            break
    if aggregate_initial is not None:
        close_float(by_step[0], aggregate_initial, f"{method} metrics/aggregate initial validation")
    if row.get("tail5_val_loss_mean") not in (None, ""):
        close_float(tail_mean, row["tail5_val_loss_mean"], f"{method} metrics/aggregate tail-5")
    milestone_cross_checks = []
    for step in required_tail_steps:
        key = f"val_loss_step_{step}"
        if row.get(key) not in (None, ""):
            close_float(by_step[step], row[key], f"{method} metrics/aggregate {key}")
            milestone_cross_checks.append(step)

    return {
        "method": method,
        "selected_cell": selected_cell,
        "seed": seed,
        "source": "accepted_formal_child_metrics_csv",
        "aggregate_manifest": file_evidence(manifest, f"{method} aggregate manifest"),
        "child_manifest": file_evidence(child["manifest"], f"{method} child manifest"),
        "child_summary": file_evidence(child["summary"], f"{method} child summary"),
        "metrics": file_evidence(metrics_path, f"{method} formal metrics"),
        "validation_rows_raw": len(validation_rows),
        "validation_steps_unique": len(by_step),
        "duplicate_validation_rows_collapsed": duplicate_rows_collapsed,
        "initial_step": 0,
        "initial_val_loss": by_step[0],
        "final_step": formal_steps,
        "final_val_loss": by_step[formal_steps],
        "tail5_steps": required_tail_steps,
        "tail5_losses": tail_losses,
        "tail5_val_loss_mean": tail_mean,
        "aggregate_milestone_steps_cross_checked": milestone_cross_checks,
    }


def normalized_fresh_row(
    method: str,
    selected_cell: str,
    row: dict[str, Any],
    manifest: Path,
    metrics_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def first(*names: str) -> Any:
        for name in names:
            if row.get(name) not in (None, ""):
                return row[name]
        return ""

    init = str(first("init_sha256"))
    if len(init) != 64:
        raise RuntimeError(f"{method}: missing formal initialization SHA-256")
    optimizer_bytes = first("optimizer_state_bytes", "total_optimizer_state_bytes")
    optimizer_mib = first("optimizer_state_mib")
    if optimizer_mib == "":
        optimizer_mib = finite_float(optimizer_bytes, f"{method} optimizer bytes") / 1024**2
    peak = first("peak_memory_mib", "peak_memory_allocated_mib")
    tail5 = metrics_evidence["tail5_val_loss_mean"] if metrics_evidence is not None else first("tail5_val_loss_mean")
    if tail5 == "":
        milestone_curve = sorted(
            (
                (int(key.removeprefix("val_loss_step_")), finite_float(value, f"{method} {key}"))
                for key, value in row.items()
                if key.startswith("val_loss_step_")
            ),
            key=lambda item: item[0],
        )
        if len(milestone_curve) < 5:
            raise RuntimeError(f"{method}: insufficient validation milestones for tail-5")
        tail5 = statistics.mean(value for _, value in milestone_curve[-5:])
    initial = metrics_evidence["initial_val_loss"] if metrics_evidence is not None else first("initial_val_loss", "val_loss_step_0")
    normalized = {
        "method": method,
        "display_name": DISPLAY[method],
        "selected_cell": selected_cell,
        "seed": 2027,
        "seed_role": "fresh_confirmatory_seed",
        "init_sha256": init,
        "initial_val_loss": finite_float(initial, f"{method} initial loss"),
        "final_val_loss": finite_float(first("final_val_loss"), f"{method} final loss"),
        "best_val_loss": finite_float(first("best_val_loss"), f"{method} best loss"),
        "tail5_val_loss_mean": finite_float(tail5, f"{method} tail5"),
        "normalized_val_auc": finite_float(first("normalized_val_auc", "val_curve_mean"), f"{method} AUC"),
        "peak_memory_mib": finite_float(peak, f"{method} peak memory"),
        "optimizer_state_mib": finite_float(optimizer_mib, f"{method} optimizer state"),
        "timing_eligible": False,
        "formal_manifest": str(manifest),
        "formal_manifest_sha256": sha256_file(manifest),
    }
    if metrics_evidence is not None:
        metrics = metrics_evidence["metrics"]
        normalized.update({
            "tail5_source": metrics_evidence["source"],
            "tail5_validation_steps": ";".join(str(step) for step in metrics_evidence["tail5_steps"]),
            "formal_metrics_path": metrics["path"],
            "formal_metrics_bytes": metrics["bytes"],
            "formal_metrics_sha256": metrics["sha256"],
        })
    return normalized


def canonical_historical(row: dict[str, str]) -> str:
    raw = row.get("method", "").strip().lower()
    for method, aliases in CELL_ALIASES.items():
        if raw in aliases:
            return method
    raise RuntimeError(f"unknown historical method label: {raw!r}")


def normalize_historical(
    rows: list[dict[str, str]], selected_cells: dict[str, str],
) -> list[dict[str, Any]]:
    output = []
    seen: set[tuple[str, int]] = set()
    for source in rows:
        method = canonical_historical(source)
        seed = int(source["seed"])
        if seed not in {2024, 2025, 2026}:
            raise RuntimeError(f"unexpected historical seed: {seed}")
        key = (method, seed)
        if key in seen:
            raise RuntimeError(f"duplicate historical row: {key}")
        seen.add(key)
        output.append({
            "method": method,
            "display_name": DISPLAY[method],
            "selected_cell": selected_cells[method],
            "seed": seed,
            "seed_role": "selection_linked_seed" if seed == 2026 and method in {
                "adamw", "normuon", "moonlight", "mousse", "malt", "malter_eq17"
            } else "independent_confirmatory_seed",
            "init_sha256": source.get("init_sha256", ""),
            "initial_val_loss": finite_float(source["initial_val_loss"], f"{key} initial loss"),
            "final_val_loss": finite_float(source["final_val_loss"], f"{key} final loss"),
            "best_val_loss": finite_float(source["best_val_loss"], f"{key} best loss"),
            "tail5_val_loss_mean": finite_float(source["tail5_val_loss_mean"], f"{key} tail5"),
            "normalized_val_auc": finite_float(source["normalized_val_auc"], f"{key} AUC"),
            "peak_memory_mib": finite_float(source["peak_memory_mib"], f"{key} peak memory"),
            "optimizer_state_mib": finite_float(source["optimizer_state_mib"], f"{key} state"),
            "timing_eligible": False,
            "formal_manifest": "accepted_historical_panel",
            "formal_manifest_sha256": "",
        })
    expected = {(method, seed) for method in CANONICAL_METHODS for seed in (2024, 2025, 2026)}
    if seen != expected:
        raise RuntimeError(f"historical panel coverage mismatch: missing={sorted(expected-seen)} extra={sorted(seen-expected)}")
    return output


def method_summary(panel: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in CANONICAL_METHODS:
        subset = sorted((row for row in panel if row["method"] == method), key=lambda row: row["seed"])
        if [row["seed"] for row in subset] != [2024, 2025, 2027]:
            raise RuntimeError(f"{method}: repaired panel seed coverage mismatch")
        final = [float(row["final_val_loss"]) for row in subset]
        rows.append({
            "method": method,
            "display_name": DISPLAY[method],
            "n": 3,
            "seeds": "2024;2025;2027",
            "mean_final_val_loss": statistics.mean(final),
            "sample_sd_final_val_loss": statistics.stdev(final),
            "mean_tail5_val_loss": statistics.mean(float(row["tail5_val_loss_mean"]) for row in subset),
            "mean_normalized_val_auc": statistics.mean(float(row["normalized_val_auc"]) for row in subset),
            "mean_optimizer_state_mib": statistics.mean(float(row["optimizer_state_mib"]) for row in subset),
            "mean_peak_memory_mib": statistics.mean(float(row["peak_memory_mib"]) for row in subset),
            "rank_by_mean_final_loss": 0,
        })
    for rank, row in enumerate(sorted(rows, key=lambda item: item["mean_final_val_loss"]), 1):
        row["rank_by_mean_final_loss"] = rank
    return rows


def leave_out_summary(historical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in CANONICAL_METHODS:
        subset = sorted(
            (row for row in historical if row["method"] == method and row["seed"] in {2024, 2025}),
            key=lambda row: row["seed"],
        )
        values = [float(row["final_val_loss"]) for row in subset]
        rows.append({
            "method": method, "display_name": DISPLAY[method], "n": 2,
            "seeds": "2024;2025", "seed2024_final_val_loss": values[0],
            "seed2025_final_val_loss": values[1], "two_seed_descriptive_mean": statistics.mean(values),
            "two_seed_range": max(values) - min(values), "inferential_ci_or_p_value": "not_reported_n2_sensitivity_only",
            "rank_by_two_seed_mean": 0,
        })
    for rank, row in enumerate(sorted(rows, key=lambda item: item["two_seed_descriptive_mean"]), 1):
        row["rank_by_two_seed_mean"] = rank
    return rows


def main() -> None:
    args = parse_args()
    contract = read_json(args.contract)
    if contract.get("experiment_id") != "55_r1_fresh_seed_baseline_fairness":
        raise RuntimeError("wrong EX55 contract")
    if sha256_file(args.historical_panel) != contract["accepted_inputs"]["historical_panel_sha256"]:
        raise RuntimeError("historical accepted panel SHA-256 mismatch")
    selected_cells = {
        str(record["method"]): str(record["selected_cell"])
        for record in contract.get("methods", [])
    }
    if set(selected_cells) != set(CANONICAL_METHODS):
        raise RuntimeError("contract frozen-winner mapping is incomplete")
    historical_raw = read_csv(args.historical_panel)
    historical = normalize_historical(historical_raw, selected_cells)
    if args.historical_only:
        output = args.output_dir
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.historical_panel, output / "historical_2024_2025_2026_panel_source.csv")
        write_csv(output / "historical_2024_2025_2026_panel_annotated.csv", historical)
        sensitivity = leave_out_summary(historical)
        write_csv(output / "leave_selection_seed_out_2024_2025.csv", sensitivity)
        manifest_path = output / "analysis_preformal_manifest.json"
        artifacts = [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(output.iterdir()) if path.is_file() and path != manifest_path
        ]
        write_json(manifest_path, {
            "schema_version": 1,
            "experiment_id": "55_r1_fresh_seed_baseline_fairness",
            "analysis_role": "historical_only_zero_gpu_preformal_sensitivity",
            "passed": True,
            "historical_panel_sha256": sha256_file(args.historical_panel),
            "contract_sha256": sha256_file(args.contract),
            "historical_seeds": [2024, 2025, 2026],
            "leave_out_seeds": [2024, 2025],
            "leave_out_inference_policy": "descriptive_n2_no_ci_no_p_value",
            "method_count": len(CANONICAL_METHODS),
            "leave_out_ranking_by_two_seed_mean": [
                row["method"] for row in sorted(sensitivity, key=lambda row: row["rank_by_two_seed_mean"])
            ],
            "artifacts": artifacts,
            "created_at": datetime.now().astimezone().isoformat(),
        })
        print(f"EX55 preformal sensitivity passed: {manifest_path}")
        return
    if args.formal_units is None:
        raise RuntimeError("full EX55 analysis requires --formal-units")
    units = read_json(args.formal_units)
    records = units.get("units", [])
    if (
        units.get("passed") is not True
        or units.get("experiment_id") != "55_r1_fresh_seed_baseline_fairness"
        or units.get("formal_seed") != 2027
        or units.get("formal_units") != 10
        or units.get("all_frozen_winners_retained") is not True
        or units.get("retuning_performed") is not False
        or len(records) != 10
    ):
        raise RuntimeError("formal unit manifest is incomplete")
    pairing_path = Path(str(units.get("formal_smoke_pairing_manifest", "")))
    if (
        not pairing_path.is_file()
        or sha256_file(pairing_path) != units.get("formal_smoke_pairing_manifest_sha256")
    ):
        raise RuntimeError("formal-smoke pairing certificate lineage failed")
    pairing = read_json(pairing_path)
    if (
        pairing.get("passed") is not True
        or pairing.get("experiment_id") != "55_r1_fresh_seed_baseline_fairness"
        or pairing.get("seed") != 2027
        or pairing.get("formal_units") != 10
        or pairing.get("parameter_initialization_exact_across_all_methods") is not True
        or pairing.get("initial_validation", {}).get("within_stratum_exact") is not True
        or pairing.get("initial_validation", {}).get("formal_endpoint_validation_tokens") != 10_485_760
    ):
        raise RuntimeError("formal-smoke pairing certificate is incomplete")
    fresh: list[dict[str, Any]] = []
    metrics_lineage: list[dict[str, Any]] = []
    formal_steps = int(contract.get("protocol", {}).get("formal_steps", 6200))
    validation_every = int(contract.get("protocol", {}).get("validation_every", 100))
    if formal_steps != 6200 or validation_every != 100:
        raise RuntimeError(
            f"unexpected EX55 formal protocol: steps={formal_steps} validation_every={validation_every}"
        )
    for record in records:
        method = str(record["method"])
        selected_cell = str(record["selected_cell"])
        if method not in selected_cells or selected_cell != selected_cells[method]:
            raise RuntimeError(f"formal frozen-winner mapping drift: {method}/{selected_cell}")
        manifest = Path(str(record["manifest"]))
        if not manifest.is_file() or sha256_file(manifest) != record["manifest_sha256"]:
            raise RuntimeError(f"formal manifest lineage failed: {method}")
        summary = pick_summary(read_json(manifest), method, selected_cell)
        evidence = formal_metrics_tail5_evidence(
            manifest,
            summary,
            method,
            selected_cell,
            seed=2027,
            formal_steps=formal_steps,
            validation_every=validation_every,
        )
        metrics_lineage.append(evidence)
        fresh.append(normalized_fresh_row(method, selected_cell, summary, manifest, evidence))
    if {row["method"] for row in fresh} != set(CANONICAL_METHODS):
        raise RuntimeError("fresh panel method coverage mismatch")
    if {row["method"] for row in metrics_lineage} != set(CANONICAL_METHODS):
        raise RuntimeError("formal metrics lineage method coverage mismatch")
    metrics_paths = {row["metrics"]["path"] for row in metrics_lineage}
    if len(metrics_paths) != len(CANONICAL_METHODS):
        raise RuntimeError("formal methods unexpectedly share a metrics CSV")
    init_hashes = {row["init_sha256"] for row in fresh}
    initial_losses = {row["initial_val_loss"] for row in fresh}
    if len(init_hashes) != 1 or len(initial_losses) != 1:
        raise RuntimeError(f"fresh formal pairing failed: init_hashes={init_hashes}, initial_losses={initial_losses}")
    # EX15's engineering smoke intentionally evaluates one validation batch;
    # the other frozen formal-smoke runners use the full validation budget.
    # Pair smoke to formal by the exact parameter hash.  The full 6200-step
    # formal artifacts themselves must (and above do) agree on their full-
    # budget step-0 validation loss across all ten methods.
    if pairing.get("init_sha256") != next(iter(init_hashes)):
        raise RuntimeError("formal-smoke/formal initialization pairing drift")

    repaired = [row for row in historical if row["seed"] in {2024, 2025}] + fresh
    repaired.sort(key=lambda row: (CANONICAL_METHODS.index(row["method"]), row["seed"]))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    lineage_path = output / "formal_metrics_tail5_lineage.json"
    write_json(lineage_path, {
        "schema_version": 1,
        "experiment_id": "55_r1_fresh_seed_baseline_fairness",
        "passed": True,
        "role": "reconstruct_fresh_seed_tail5_from_hash_bound_formal_child_metrics",
        "formal_seed": 2027,
        "formal_steps": formal_steps,
        "validation_every": validation_every,
        "required_tail5_steps": [
            formal_steps - offset * validation_every for offset in range(4, -1, -1)
        ],
        "formal_units_path": str(args.formal_units.expanduser().resolve()),
        "formal_units_sha256": sha256_file(args.formal_units),
        "units": metrics_lineage,
        "created_at": datetime.now().astimezone().isoformat(),
    })
    shutil.copy2(args.historical_panel, output / "historical_2024_2025_2026_panel_source.csv")
    write_csv(output / "historical_2024_2025_2026_panel_annotated.csv", historical)
    write_csv(output / "fresh_seed2027_panel.csv", fresh)
    write_csv(output / "repaired_fresh_panel_2024_2025_2027.csv", repaired)
    summaries_out = method_summary(repaired)
    write_csv(output / "repaired_panel_method_summary.csv", summaries_out)
    sensitivity = leave_out_summary(historical)
    write_csv(output / "leave_selection_seed_out_2024_2025.csv", sensitivity)

    fresh_by_method = {row["method"]: row for row in fresh}
    paired = []
    for comparator in ("diag", "block4", "muon"):
        base = float(fresh_by_method[comparator]["final_val_loss"])
        for method in CANONICAL_METHODS:
            paired.append({
                "seed": 2027, "method": method, "comparator": comparator,
                "delta_method_minus_comparator": float(fresh_by_method[method]["final_val_loss"]) - base,
                "lower_is_better": True,
            })
    write_csv(output / "fresh_seed2027_paired_deltas.csv", paired)

    repaired_index = {(row["method"], int(row["seed"])): row for row in repaired}
    repaired_paired = []
    repaired_aggregate = []
    for comparator in ("diag", "block4", "muon"):
        for method in CANONICAL_METHODS:
            deltas = []
            for seed in (2024, 2025, 2027):
                delta = (
                    float(repaired_index[(method, seed)]["final_val_loss"])
                    - float(repaired_index[(comparator, seed)]["final_val_loss"])
                )
                deltas.append(delta)
                repaired_paired.append({
                    "seed": seed, "method": method, "comparator": comparator,
                    "delta_method_minus_comparator": delta, "lower_is_better": True,
                })
            repaired_aggregate.append({
                "method": method, "comparator": comparator, "n": 3,
                "mean_delta_method_minus_comparator": statistics.mean(deltas),
                "sample_sd_delta": statistics.stdev(deltas),
                "method_better_seed_count": sum(delta < 0 for delta in deltas),
                "method_worse_seed_count": sum(delta > 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
            })
    write_csv(output / "repaired_panel_paired_deltas.csv", repaired_paired)
    write_csv(output / "repaired_panel_paired_aggregate.csv", repaired_aggregate)

    artifacts = []
    manifest_path = output / "analysis_manifest.json"
    for path in sorted(output.iterdir()):
        if path.is_file() and path != manifest_path:
            artifacts.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    analyzer_record = file_evidence(Path(__file__), "EX55 analyzer")
    amendment_record = (
        file_evidence(args.analysis_amendment, "EX55 analysis amendment receipt")
        if args.analysis_amendment is not None
        else None
    )
    lineage_record = {
        "path": lineage_path.name,
        "bytes": lineage_path.stat().st_size,
        "sha256": sha256_file(lineage_path),
    }
    write_json(manifest_path, {
        "schema_version": 1,
        "experiment_id": "55_r1_fresh_seed_baseline_fairness",
        "passed": True,
        "classification": "fresh_seed_fairness_panel_completed",
        "analyzer_version": ANALYZER_VERSION,
        "analyzer": analyzer_record,
        "analysis_amendment": amendment_record,
        "formal_metrics_lineage": lineage_record,
        "checks": {
            "historical_panel_exact_hash": True, "historical_panel_30_rows": True,
            "fresh_seed2027_10_methods": True, "fresh_initialization_sha_identical": True,
            "fresh_initial_validation_identical": True, "repaired_panel_30_rows": True,
            "all_methods_retained": True, "timing_excluded": True,
            "leave_out_is_descriptive_n2_only": True,
            "fresh_tail5_reconstructed_from_hash_bound_formal_metrics": True,
            "formal_metrics_path_bytes_sha256_recorded_for_all_methods": True,
        },
        "historical_seeds": [2024, 2025, 2026],
        "repaired_seeds": [2024, 2025, 2027],
        "fresh_init_sha256": next(iter(init_hashes)),
        "fresh_initial_val_loss": next(iter(initial_losses)),
        "contract_sha256": sha256_file(args.contract),
        "historical_panel_sha256": sha256_file(args.historical_panel),
        "formal_units_sha256": sha256_file(args.formal_units),
        "method_ranking_by_repaired_mean_final_loss": [row["method"] for row in sorted(summaries_out, key=lambda row: row["rank_by_mean_final_loss"])],
        "leave_out_ranking_by_two_seed_mean": [row["method"] for row in sorted(sensitivity, key=lambda row: row["rank_by_two_seed_mean"])],
        "artifacts": artifacts,
        "created_at": datetime.now().astimezone().isoformat(),
    })
    print(f"EX55 analysis passed: {manifest_path}")


if __name__ == "__main__":
    main()
