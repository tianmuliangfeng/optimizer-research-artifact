#!/usr/bin/env python3
"""Verify Experiment 50 and build frozen paired comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-08-14.1"
EXPECTED_SEEDS = (2024, 2025, 2026)
EXPECTED_MEMORY = {
    "k_cov_bytes": 258_048,
    "k_inv_bytes": 258_048,
    "k_state_bytes": 516_096,
    "activation_stat_bytes": 258_240,
    "precond_workspace_bytes": 0,
}
T_95_DF2 = 4.302652729911275


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validation_tail(summary_path: Path) -> tuple[float, int]:
    metrics_path = summary_path.with_name("r1_metrics.csv")
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("event") == "validation"
        ]
    if not rows:
        raise RuntimeError(f"no validation rows in {metrics_path}")
    rows.sort(key=lambda row: int(row["step"]))
    tail = [float(row["loss"]) for row in rows[-5:]]
    return statistics.mean(tail), len(rows)


def collect_formal(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "formal").glob("seed*/**/r1_summary.json")):
        payload = read_json(path)
        if payload.get("method") != "global_diag":
            continue
        tail5, validation_points = validation_tail(path)
        rows.append(
            {
                "method": "global_diag",
                "seed": int(payload["controlled_seed"]),
                "final_val_loss": float(payload["final_val_loss"]),
                "best_val_loss": float(payload["best_val_loss"]),
                "tail5_val_loss_mean": tail5,
                "validation_points": validation_points,
                "final_val_step": int(payload["final_val_step"]),
                "init_sha256": str(payload["init_sha256"]),
                "derived_script_sha256": str(
                    payload.get("derived_script_sha256", "")
                ),
                "k_cov_bytes": int(payload["k_cov_bytes"]),
                "k_inv_bytes": int(payload["k_inv_bytes"]),
                "k_state_bytes": int(payload["k_state_bytes"]),
                "activation_stat_bytes": int(payload["activation_stat_bytes"]),
                "precond_workspace_bytes": int(
                    payload["precond_workspace_bytes"]
                ),
                "peak_memory_mib": float(payload["peak_memory_allocated_mib"]),
                "source_summary": str(path),
            }
        )
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    if set(by_seed) != set(EXPECTED_SEEDS) or any(
        len(value) != 1 for value in by_seed.values()
    ):
        counts = {seed: len(value) for seed, value in by_seed.items()}
        raise RuntimeError(
            f"expected one global-diag formal summary per seed {EXPECTED_SEEDS}; "
            f"observed={counts}"
        )
    return [by_seed[seed][0] for seed in EXPECTED_SEEDS]


def collect_controls(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != 12:
        raise RuntimeError(f"frozen controls must contain 12 rows, observed {len(rows)}")
    cells = {(row["method"], int(row["seed"])) for row in rows}
    expected = {
        (method, seed)
        for method in ("block4", "diag", "none", "muon")
        for seed in EXPECTED_SEEDS
    }
    if cells != expected:
        raise RuntimeError(f"frozen control grid mismatch: {cells ^ expected}")
    return rows


def mean_ci(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T_95_DF2 * sd / math.sqrt(len(values))
    return mean, sd, mean - half, mean + half


def classify_primary_delta(delta: float, margin: float) -> str:
    if delta > margin:
        return "global_diag_worse_than_selective_diag"
    if delta < -margin:
        return "global_diag_better_than_selective_diag"
    return "descriptively_close_not_formal_equivalence"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    controls_path = args.controls.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    contract = read_json(contract_path)
    formal = collect_formal(run_dir)
    controls = collect_controls(controls_path)

    checks: dict[str, bool] = {
        "contract_experiment_id": contract.get("experiment_id")
        == "50_r1_global_activation_diag",
        "formal_seed_grid": [row["seed"] for row in formal]
        == list(EXPECTED_SEEDS),
        "formal_endpoints": all(row["final_val_step"] == 6200 for row in formal),
        "formal_losses_finite": all(
            math.isfinite(row["final_val_loss"]) for row in formal
        ),
        "memory_route_exact": all(
            all(row[key] == value for key, value in EXPECTED_MEMORY.items())
            for row in formal
        ),
        "derived_source_one_hash": len(
            {row["derived_script_sha256"] for row in formal}
        )
        == 1,
        "derived_source_hash_nonempty": all(
            len(row["derived_script_sha256"]) == 64 for row in formal
        ),
        "control_grid": len(controls) == 12,
        "control_source_frozen": contract["frozen_controls"]["read_only"] is True,
        "timing_excluded": contract["execution_policy"]["timing_usable"] is False,
        "no_formal_equivalence_claim": contract["formal_equivalence_claim_allowed"]
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Experiment-50 verify failed: {checks}")

    control_lookup = {
        (row["method"], int(row["seed"])): float(row["final_val_loss"])
        for row in controls
    }
    per_seed: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for comparator in ("diag", "block4", "none", "muon"):
        values: list[float] = []
        for row in formal:
            seed = int(row["seed"])
            control = control_lookup[(comparator, seed)]
            delta = float(row["final_val_loss"]) - control
            values.append(delta)
            per_seed.append(
                {
                    "comparator": comparator,
                    "seed": seed,
                    "global_diag_final_val_loss": row["final_val_loss"],
                    "comparator_final_val_loss": control,
                    "delta_global_minus_comparator": delta,
                    "global_diag_better": delta < 0.0,
                }
            )
        mean, sd, low, high = mean_ci(values)
        aggregates.append(
            {
                "comparator": comparator,
                "n": len(values),
                "mean_delta_global_minus_comparator": mean,
                "sample_sd": sd,
                "ci95_low": low,
                "ci95_high": high,
                "global_diag_better_seed_count": sum(value < 0 for value in values),
                "global_diag_worse_seed_count": sum(value > 0 for value in values),
            }
        )

    primary = next(row for row in aggregates if row["comparator"] == "diag")
    delta = float(primary["mean_delta_global_minus_comparator"])
    margin = float(contract["practical_proximity_margin"])
    classification = classify_primary_delta(delta, margin)

    output_dir.mkdir(parents=True, exist_ok=True)
    formal_rows = [
        {
            **row,
            "k_state_mib": row["k_state_bytes"] / (1024**2),
        }
        for row in formal
    ]
    all_runs: list[dict[str, Any]] = []
    for row in controls:
        all_runs.append(
            {
                "method": row["method"],
                "seed": int(row["seed"]),
                "final_val_loss": float(row["final_val_loss"]),
                "best_val_loss": float(row["best_val_loss"]),
                "tail5_val_loss_mean": float(row["tail5_val_loss_mean"]),
                "k_state_mib": float(row["k_state_mib"]),
                "source": "frozen_experiment_15_control",
            }
        )
    for row in formal_rows:
        all_runs.append(
            {
                "method": "global_diag",
                "seed": row["seed"],
                "final_val_loss": row["final_val_loss"],
                "best_val_loss": row["best_val_loss"],
                "tail5_val_loss_mean": row["tail5_val_loss_mean"],
                "k_state_mib": row["k_state_mib"],
                "source": "experiment_50_formal",
            }
        )
    method_summary: list[dict[str, Any]] = []
    for method in ("global_diag", "diag", "block4", "none", "muon"):
        selected = [row for row in all_runs if row["method"] == method]
        losses = [float(row["final_val_loss"]) for row in selected]
        method_summary.append(
            {
                "method": method,
                "n": len(selected),
                "mean_final_val_loss": statistics.mean(losses),
                "sample_sd_final_val_loss": statistics.stdev(losses),
                "mean_tail5_val_loss": statistics.mean(
                    float(row["tail5_val_loss_mean"]) for row in selected
                ),
                "mean_k_state_mib": statistics.mean(
                    float(row["k_state_mib"]) for row in selected
                ),
            }
        )
    diag_state = next(
        float(row["mean_k_state_mib"])
        for row in method_summary
        if row["method"] == "diag"
    )
    global_state = EXPECTED_MEMORY["k_state_bytes"] / (1024**2)
    state_reduction_factor = diag_state / global_state
    write_csv(output_dir / "global_diag_formal_results.csv", formal_rows)
    write_csv(output_dir / "all_method_runs.csv", all_runs)
    write_csv(output_dir / "method_summary.csv", method_summary)
    write_csv(output_dir / "paired_contrasts.csv", per_seed)
    write_csv(output_dir / "aggregate_contrasts.csv", aggregates)

    report = f"""# Experiment 50: global activation-diagonal control

- classification: `{classification}`
- primary mean delta (global-diag minus selective-diag): `{delta:.6f}`
- 95% t interval: `[{float(primary['ci95_low']):.6f}, {float(primary['ci95_high']):.6f}]`
- direction: global-diag better in `{primary['global_diag_better_seed_count']}/3` seeds
- persistent K-state: `{EXPECTED_MEMORY['k_state_bytes']} bytes` (`0.4921875 MiB`)
- selective-diag/global-diag K-state ratio: `{state_reduction_factor:.3f}x`
- timing evidence: ineligible
- equivalence language: prohibited; the practical margin is descriptive only

Interpretation must follow the frozen three-branch policy in
`global_diag_contract.json`. This analyzer does not select a favorable branch
post hoc.
"""
    (output_dir / "EXPERIMENT_50_ANALYSIS.md").write_text(
        report, encoding="utf-8", newline="\n"
    )

    artifacts = [
        output_dir / "global_diag_formal_results.csv",
        output_dir / "all_method_runs.csv",
        output_dir / "method_summary.csv",
        output_dir / "paired_contrasts.csv",
        output_dir / "aggregate_contrasts.csv",
        output_dir / "EXPERIMENT_50_ANALYSIS.md",
    ]
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment_id": "50_r1_global_activation_diag",
        "status": "completed_valid",
        "passed": True,
        "claim_eligible": True,
        "classification": classification,
        "primary_contrast": primary,
        "checks": checks,
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "controls": str(controls_path),
        "controls_sha256": sha256_file(controls_path),
        "artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in artifacts
        ],
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    write_json(output_dir / "analysis_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
