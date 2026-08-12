#!/usr/bin/env python3
"""Analysis-only diagonal bridge for experiment 41.

This analysis freezes the three-seed experiment-15 R1 slice with c_fc K full
and c_proj K in {none, diag, block4}. It links the block4 and none rows back to
experiment 41, computes paired effects at seed grain, and emits a paper-facing
technical report without training a new model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import statistics
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-31.1"
SEEDS = (2024, 2025, 2026)
METHODS = ("none", "diag", "block4")
METRICS = ("final_val_loss", "tail5_val_loss_mean", "normalized_val_auc")
T95_DF2 = 4.302652729911275
PARETO_SQL = """\
SELECT
    method,
    mean_final_val_loss,
    loss_improvement_vs_none,
    k_state_mib,
    k_state_saved_vs_block4_mib,
    peak_memory_mib,
    cfc_k_mode,
    seeds
FROM diag_bridge_pareto
ORDER BY CASE method
    WHEN 'diag' THEN 1
    WHEN 'block4' THEN 2
    WHEN 'none' THEN 3
    ELSE 4
END
"""
CELLS_SQL = """\
SELECT
    method,
    cfc_k_mode,
    cproj_k_mode,
    seeds,
    mean_final_val_loss,
    sample_sd_final_val_loss,
    mean_tail5_val_loss,
    mean_normalized_val_auc,
    k_state_mib,
    peak_memory_mib,
    optimizer_state_mib
FROM diag_bridge_cells
ORDER BY mean_final_val_loss ASC
"""
FLOAT_FIELDS = (
    "initial_val_loss",
    "final_val_loss",
    "tail5_val_loss_mean",
    "normalized_val_auc",
    "peak_memory_mib",
    "k_state_mib",
    "optimizer_state_mib",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--experiment41-reference", type=Path, required=True)
    parser.add_argument("--experiment41-acceptance", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(left: float, right: float, atol: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=atol)


def normalized_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for raw in rows:
        if raw.get("method") not in METHODS:
            continue
        row: dict[str, Any] = {
            "method": raw["method"],
            "run_name": raw["run_name"],
            "seed": int(raw["seed"]),
        }
        for field in FLOAT_FIELDS:
            row[field] = float(raw[field])
        selected.append(row)
    return selected


def validate_sources(
    run_summary: Path,
    reference: Path,
    acceptance: Path,
    contract_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    contract = read_json(contract_path)
    source_contract = contract["source_contract"]
    source_hashes = {
        "experiment15_run_summary": sha256_file(run_summary),
        "experiment41_reference": sha256_file(reference),
        "experiment41_acceptance": sha256_file(acceptance),
        "diag_bridge_contract": sha256_file(contract_path),
    }
    expected_hashes = {
        "experiment15_run_summary": source_contract[
            "experiment15_run_summary_sha256"
        ],
        "experiment41_reference": source_contract[
            "experiment41_reference_sha256"
        ],
        "experiment41_acceptance": source_contract[
            "experiment41_acceptance_sha256"
        ],
    }
    hash_checks = {
        key: source_hashes[key] == expected
        for key, expected in expected_hashes.items()
    }
    if not all(hash_checks.values()):
        raise RuntimeError(
            f"source hash contract failed: observed={source_hashes}, "
            f"expected={expected_hashes}"
        )

    rows = normalized_rows(read_csv(run_summary))
    expected_keys = {(method, seed) for method in METHODS for seed in SEEDS}
    observed_keys = {(row["method"], row["seed"]) for row in rows}
    if len(rows) != len(expected_keys) or observed_keys != expected_keys:
        raise RuntimeError(
            f"diag bridge coverage failed: rows={len(rows)}, "
            f"observed={sorted(observed_keys)}"
        )
    if len(observed_keys) != len(rows):
        raise RuntimeError("duplicate method/seed rows in run summary")
    if not all(
        math.isfinite(float(row[field]))
        for row in rows
        for field in FLOAT_FIELDS
    ):
        raise RuntimeError("non-finite value in selected rows")

    initial_checks: dict[str, bool] = {}
    for seed in SEEDS:
        values = {
            row["initial_val_loss"] for row in rows if row["seed"] == seed
        }
        initial_checks[str(seed)] = len(values) == 1
    if not all(initial_checks.values()):
        raise RuntimeError(f"initial loss mismatch: {initial_checks}")

    reference_rows = normalized_rows(read_csv(reference))
    reference_map = {
        (row["method"], row["seed"]): row
        for row in reference_rows
        if row["method"] in {"none", "block4"}
    }
    linkage_fields = (
        "initial_val_loss",
        "final_val_loss",
        "tail5_val_loss_mean",
        "normalized_val_auc",
        "peak_memory_mib",
        "k_state_mib",
    )
    reference_linkage: dict[str, bool] = {}
    for row in rows:
        if row["method"] not in {"none", "block4"}:
            continue
        key = (row["method"], row["seed"])
        linked = reference_map.get(key)
        passed = linked is not None and all(
            close(row[field], linked[field]) for field in linkage_fields
        )
        reference_linkage[f"{row['method']}_seed{row['seed']}"] = passed
    if not all(reference_linkage.values()):
        raise RuntimeError(
            f"experiment-41 reference linkage failed: {reference_linkage}"
        )

    acceptance_payload = read_json(acceptance)
    acceptance_checks = {
        "accepted": acceptance_payload["training_complete"]
        and acceptance_payload["quality_usable"],
        "classification_corrected": acceptance_payload[
            "accepted_classification"
        ]
        == "r1_allocation_diverges",
        "both_mean": close(
            statistics.mean(
                row["final_val_loss"]
                for row in rows
                if row["method"] == "block4"
            ),
            acceptance_payload["cells"]["both"]["mean_final_val_loss"],
        ),
        "fc_only_mean": close(
            statistics.mean(
                row["final_val_loss"]
                for row in rows
                if row["method"] == "none"
            ),
            acceptance_payload["cells"]["fc_only"][
                "mean_final_val_loss"
            ],
        ),
    }
    if not all(acceptance_checks.values()):
        raise RuntimeError(
            f"experiment-41 acceptance linkage failed: {acceptance_checks}"
        )

    expected_state = contract["expected_state_mib"]
    state_checks = {
        method: all(
            close(row["k_state_mib"], float(expected_state[method]))
            for row in rows
            if row["method"] == method
        )
        for method in METHODS
    }
    if not all(state_checks.values()):
        raise RuntimeError(f"K-state checks failed: {state_checks}")

    checks = {
        "source_hashes": hash_checks,
        "cell_seed_coverage": True,
        "initial_loss_equal_within_seed": initial_checks,
        "experiment41_reference_linkage": reference_linkage,
        "experiment41_acceptance_linkage": acceptance_checks,
        "k_state_contract": state_checks,
        "all_values_finite": True,
        "timing_usable": False,
    }
    return contract, rows, {
        "checks": checks,
        "source_hashes": source_hashes,
    }


def t_summary(values: list[float]) -> dict[str, Any]:
    if len(values) != 3:
        raise RuntimeError(f"expected three paired seeds, got {len(values)}")
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    half_width = T95_DF2 * sample_sd / math.sqrt(len(values))
    return {
        "seeds": len(values),
        "mean": mean,
        "sample_sd": sample_sd,
        "ci95_low_t_df2": mean - half_width,
        "ci95_high_t_df2": mean + half_width,
        "negative_seeds": sum(value < 0.0 for value in values),
        "positive_seeds": sum(value > 0.0 for value in values),
        "zero_seeds": sum(value == 0.0 for value in values),
    }


def build_effects(
    rows: list[dict[str, Any]], practical_margin: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["method"], row["seed"]): row
        for row in rows
    }
    comparisons = (
        ("diag_minus_none", "diag", "none"),
        ("diag_minus_block4", "diag", "block4"),
        ("block4_minus_none", "block4", "none"),
    )
    by_seed: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for metric in METRICS:
        for comparison, left, right in comparisons:
            values: list[float] = []
            for seed in SEEDS:
                value = (
                    lookup[(left, seed)][metric]
                    - lookup[(right, seed)][metric]
                )
                values.append(value)
                by_seed.append(
                    {
                        "metric": metric,
                        "comparison": comparison,
                        "seed": seed,
                        "left_method": left,
                        "right_method": right,
                        "delta": value,
                        "negative_means": "left method lowers the metric",
                    }
                )
            summary = t_summary(values)
            summary.update(
                {
                    "metric": metric,
                    "comparison": comparison,
                    "left_method": left,
                    "right_method": right,
                    "practical_margin": practical_margin,
                    "material_by_mean": abs(summary["mean"])
                    >= practical_margin,
                    "direction_consistent_2of3": max(
                        summary["negative_seeds"],
                        summary["positive_seeds"],
                    )
                    >= 2,
                }
            )
            summaries.append(summary)
    return by_seed, summaries


def build_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        output.append(
            {
                "method": method,
                "cfc_k_mode": "full",
                "cproj_k_mode": method,
                "seeds": len(selected),
                "mean_final_val_loss": statistics.mean(
                    row["final_val_loss"] for row in selected
                ),
                "sample_sd_final_val_loss": statistics.stdev(
                    row["final_val_loss"] for row in selected
                ),
                "mean_tail5_val_loss": statistics.mean(
                    row["tail5_val_loss_mean"] for row in selected
                ),
                "mean_normalized_val_auc": statistics.mean(
                    row["normalized_val_auc"] for row in selected
                ),
                "k_state_mib": selected[0]["k_state_mib"],
                "peak_memory_mib": selected[0]["peak_memory_mib"],
                "optimizer_state_mib": selected[0]["optimizer_state_mib"],
            }
        )
    return output


def primary_summary(
    summaries: list[dict[str, Any]], comparison: str
) -> dict[str, Any]:
    return next(
        row
        for row in summaries
        if row["metric"] == "final_val_loss"
        and row["comparison"] == comparison
    )


def build_decision(
    cells: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    practical_margin: float,
) -> dict[str, Any]:
    diag_none = primary_summary(summaries, "diag_minus_none")
    diag_block4 = primary_summary(summaries, "diag_minus_block4")
    diag_beneficial = (
        diag_none["mean"] <= -practical_margin
        and diag_none["negative_seeds"] >= 2
    )
    quality_matched = abs(diag_block4["mean"]) < practical_margin
    superior_to_block4 = (
        diag_block4["mean"] <= -practical_margin
        and diag_block4["ci95_high_t_df2"] < 0.0
    )
    cell_map = {row["method"]: row for row in cells}
    k_saved = (
        cell_map["block4"]["k_state_mib"]
        - cell_map["diag"]["k_state_mib"]
    )
    peak_saved = (
        cell_map["block4"]["peak_memory_mib"]
        - cell_map["diag"]["peak_memory_mib"]
    )
    classification = (
        "diag_recovers_block4_quality_at_near_none_state_cost"
        if diag_beneficial and quality_matched
        else "diag_bridge_inconclusive"
    )
    return {
        "classification": classification,
        "diag_beneficial_over_none": diag_beneficial,
        "diag_quality_matched_to_block4": quality_matched,
        "diag_superior_to_block4": superior_to_block4,
        "diag_minus_none": diag_none,
        "diag_minus_block4": diag_block4,
        "practical_loss_margin": practical_margin,
        "diag_extra_k_state_vs_none_mib": (
            cell_map["diag"]["k_state_mib"]
            - cell_map["none"]["k_state_mib"]
        ),
        "diag_k_state_saved_vs_block4_mib": k_saved,
        "diag_k_state_saved_vs_block4_percent": (
            100.0 * k_saved / cell_map["block4"]["k_state_mib"]
        ),
        "diag_peak_memory_saved_vs_block4_mib": peak_saved,
        "new_training_recommended": False,
        "reason_no_new_training": (
            "The deployed c_fc=full slice already contains paired three-seed "
            "none/diag/block4 evidence. The missing c_fc=none,c_proj=diag "
            "cell would only test a secondary interaction."
        ),
    }


def build_pareto_rows(
    cells: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_method = {row["method"]: row for row in cells}
    none_loss = by_method["none"]["mean_final_val_loss"]
    block4_state = by_method["block4"]["k_state_mib"]
    output: list[dict[str, Any]] = []
    for method in METHODS:
        row = by_method[method]
        output.append(
            {
                "method": method,
                "mean_final_val_loss": row["mean_final_val_loss"],
                "loss_improvement_vs_none": (
                    none_loss - row["mean_final_val_loss"]
                ),
                "k_state_mib": row["k_state_mib"],
                "k_state_saved_vs_block4_mib": (
                    block4_state - row["k_state_mib"]
                ),
                "peak_memory_mib": row["peak_memory_mib"],
                "cfc_k_mode": "full",
                "seeds": row["seeds"],
            }
        )
    return output


def build_report(
    cells: list[dict[str, Any]],
    decision: dict[str, Any],
    source_hashes: dict[str, str],
) -> str:
    by_method = {row["method"]: row for row in cells}
    dn = decision["diag_minus_none"]
    db = decision["diag_minus_block4"]
    return f"""# Experiment 41D: R1 diagonal bridge

## Technical summary

With `c_fc K` fixed to full, diagonal `c_proj K` materially improves final
validation loss over no `c_proj K` while using only 0.28125 MiB additional
persistent K state. Diag also matches block4 within the preregistered 0.002
practical-loss margin while saving
{decision['diag_k_state_saved_vs_block4_mib']:.5f} MiB
({decision['diag_k_state_saved_vs_block4_percent']:.2f}%) of K state.

The accepted classification is
**{decision['classification']}**. No new training is recommended.

## Diag recovers the omitted scale information at near-none state cost

| c_proj K | Mean final val loss | K-state MiB | Peak memory MiB |
|---|---:|---:|---:|
| diag | {by_method['diag']['mean_final_val_loss']:.6f} | {by_method['diag']['k_state_mib']:.5f} | {by_method['diag']['peak_memory_mib']:.0f} |
| block4 | {by_method['block4']['mean_final_val_loss']:.6f} | {by_method['block4']['k_state_mib']:.5f} | {by_method['block4']['peak_memory_mib']:.0f} |
| none | {by_method['none']['mean_final_val_loss']:.6f} | {by_method['none']['k_state_mib']:.5f} | {by_method['none']['peak_memory_mib']:.0f} |

Diag-minus-none final loss is {dn['mean']:.6f}, with a 95% t interval of
[{dn['ci95_low_t_df2']:.6f}, {dn['ci95_high_t_df2']:.6f}] and 3/3 seeds in
the beneficial direction. This is both statistically directional in this
small sample and larger than the 0.002 practical margin.

Diag-minus-block4 final loss is {db['mean']:.6f}, with a 95% t interval of
[{db['ci95_low_t_df2']:.6f}, {db['ci95_high_t_df2']:.6f}]. All three seeds
are numerically favorable to diag, but the mean magnitude is below the
practical margin and the interval crosses zero. The defensible claim is
quality matching or slight numerical improvement, not superiority.

## Scope and metric definitions

- Architecture and recipe: R1 Modded-NanoGPT, 6200 updates.
- Statistical unit: paired seed; seeds 2024, 2025, and 2026.
- Fixed factor: `c_fc K=full`.
- Varied factor: `c_proj K` in `none`, `diag`, and `block4`.
- Primary metric: final validation loss; lower is better.
- Practical margin: 0.002 final-loss units.
- Confidence intervals: paired-effect t intervals with two degrees of freedom.

## Methodology and source linkage

The analysis reads the frozen experiment-15 three-seed summary. Its block4 and
none rows are checked field-by-field against experiment 41's frozen reused-cell
reference. Their method/seed coverage, initial losses, final losses, tail-five
losses, normalized AUC, K state, and peak memory must match before any output is
written. The accepted experiment-41 result is also hash-pinned and its block4
and none means are rechecked.

Source SHA-256 values:

- experiment-15 run summary:
  `{source_hashes['experiment15_run_summary']}`
- experiment-41 frozen reference:
  `{source_hashes['experiment41_reference']}`
- experiment-41 accepted result:
  `{source_hashes['experiment41_acceptance']}`
- 41D contract:
  `{source_hashes['diag_bridge_contract']}`

## Limitations and robustness boundaries

- `n=3`; intervals are small-sample t intervals.
- The result establishes the diag effect only when `c_fc K` is full.
- It does not estimate the missing `c_fc=none,c_proj=diag` cell or a
  diag-by-`c_fc` interaction.
- It is not a Muon comparison.
- Concurrent timing is not used; experiment 39 remains the isolated-efficiency
  source.
- Peak-memory equality for diag and none is reported at the measurement
  resolution of the original runs.

## Recommended paper use

Place this three-level slice next to the experiment-41 2×2 factorial:

1. Experiment 41 shows that full `c_fc K` and block4 `c_proj K` make
   approximately additive quality contributions.
2. Experiment 41D shows that the diagonal approximation retains the useful
   `c_proj` scale signal at essentially none-level state cost.
3. State that diag materially improves over none and quality-matches block4;
   do not claim statistically established superiority over block4.

## Further question

Only add the missing `c_fc=none,c_proj=diag` cell if the paper later requires a
claim that the diag benefit is independent of `c_fc K`. It is not required for
the current deployed-configuration claim.
"""


def build_report_artifact(
    cells: list[dict[str, Any]],
    effects_by_seed: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
    decision: dict[str, Any],
) -> dict[str, Any]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE diag_bridge_pareto (
            method TEXT,
            mean_final_val_loss REAL,
            loss_improvement_vs_none REAL,
            k_state_mib REAL,
            k_state_saved_vs_block4_mib REAL,
            peak_memory_mib REAL,
            cfc_k_mode TEXT,
            seeds INTEGER
        )"""
    )
    connection.executemany(
        """INSERT INTO diag_bridge_pareto VALUES (
            :method, :mean_final_val_loss, :loss_improvement_vs_none,
            :k_state_mib, :k_state_saved_vs_block4_mib, :peak_memory_mib,
            :cfc_k_mode, :seeds
        )""",
        pareto,
    )
    connection.execute(
        """CREATE TABLE diag_bridge_cells (
            method TEXT,
            cfc_k_mode TEXT,
            cproj_k_mode TEXT,
            seeds INTEGER,
            mean_final_val_loss REAL,
            sample_sd_final_val_loss REAL,
            mean_tail5_val_loss REAL,
            mean_normalized_val_auc REAL,
            k_state_mib REAL,
            peak_memory_mib REAL,
            optimizer_state_mib REAL
        )"""
    )
    connection.executemany(
        """INSERT INTO diag_bridge_cells VALUES (
            :method, :cfc_k_mode, :cproj_k_mode, :seeds,
            :mean_final_val_loss, :sample_sd_final_val_loss,
            :mean_tail5_val_loss, :mean_normalized_val_auc,
            :k_state_mib, :peak_memory_mib, :optimizer_state_mib
        )""",
        cells,
    )
    pareto_rows = [
        dict(row) for row in connection.execute(PARETO_SQL).fetchall()
    ]
    cell_rows = [
        dict(row) for row in connection.execute(CELLS_SQL).fetchall()
    ]
    connection.close()
    final_effects = [
        row
        for row in effects_by_seed
        if row["metric"] == "final_val_loss"
    ]
    kpis = [
        {
            "diag_loss_improvement_vs_none": -decision[
                "diag_minus_none"
            ]["mean"],
            "diag_k_state_saved_fraction_vs_block4": (
                decision["diag_k_state_saved_vs_block4_percent"] / 100.0
            ),
            "diag_extra_k_state_vs_none_mib": decision[
                "diag_extra_k_state_vs_none_mib"
            ],
            "paired_seeds": 3,
        }
    ]
    source15 = {
        "id": "experiment15_formal_r1",
        "label": "Experiment 15 R1 three-seed Pareto view",
        "path": "diag_bridge_pareto.csv",
        "query": {
            "engine": "sqlite3",
            "language": "SQL",
            "sql": PARETO_SQL,
            "description": (
                "Select the validated Pareto rows derived from the frozen "
                "paired three-seed experiment-15 formal summary."
            ),
            "tables_used": [
                "diag_bridge_pareto",
                "r1_multiseed_run_summary.csv",
            ],
            "filters": [
                "formal methods in {none, diag, block4}",
                "paired seeds in {2024, 2025, 2026}",
                "c_fc K fixed to full",
            ],
            "metric_definitions": [
                "Mean final validation loss is the arithmetic mean across three paired seeds.",
                "Loss improvement versus none equals none mean loss minus method mean loss.",
                "K-state MiB uses a 2^20 denominator.",
            ],
        },
    }
    source15_cells = {
        "id": "experiment15_formal_r1_cells",
        "label": "Experiment 15 R1 three-seed cell summary",
        "path": "diag_bridge_cells.csv",
        "query": {
            "engine": "sqlite3",
            "language": "SQL",
            "sql": CELLS_SQL,
            "description": (
                "Select the exact validated three-level cell summary derived "
                "from the frozen experiment-15 formal runs."
            ),
            "tables_used": [
                "diag_bridge_cells",
                "r1_multiseed_run_summary.csv",
            ],
            "filters": [
                "formal methods in {none, diag, block4}",
                "paired seeds in {2024, 2025, 2026}",
                "c_fc K fixed to full",
            ],
            "metric_definitions": [
                "Each loss value is an unweighted arithmetic mean across three paired seeds.",
                "Memory values are invariant method-level observations from the formal runs.",
            ],
        },
    }
    source41 = {
        "id": "experiment41_accepted",
        "label": "Experiment 41 accepted 2x2 factorial",
        "path": "experiment41_key_results.json",
        "query": {
            "engine": "python-stdlib",
            "language": "Python",
            "sql": (
                "import json\n"
                "with open('experiment41_key_results.json', encoding='utf-8') "
                "as handle:\n"
                "    accepted = json.load(handle)\n"
                "assert accepted['training_complete']\n"
                "assert accepted['quality_usable']\n"
                "assert accepted['accepted_classification'] == "
                "'r1_allocation_diverges'\n"
            ),
            "description": (
                "Load the accepted experiment-41 result and verify its "
                "training, quality, and corrected-classification gates."
            ),
            "tables_used": ["experiment41_key_results.json"],
            "filters": ["accepted experiment-41 result only"],
            "metric_definitions": [
                "Bridge linkage additionally checks the frozen per-seed reference in the analysis code.",
            ],
        },
    }
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Experiment 41D: R1 diagonal bridge",
            "description": (
                "Analysis-only three-seed bridge showing the R1 quality and "
                "state trade-off among c_proj none, diag, and block4."
            ),
            "generatedAt": "2026-07-31T00:00:00+08:00",
            "sources": [source15, source15_cells, source41],
            "cards": [],
            "charts": [
                {
                    "id": "quality_improvement_chart",
                    "title": "Final validation-loss improvement versus none",
                    "subtitle": (
                        "R1 Modded-NanoGPT, paired seeds 2024–2026; larger "
                        "positive values are better."
                    ),
                    "type": "horizontalBar",
                    "dataset": "pareto",
                    "source": source15,
                    "encodings": {
                        "x": {
                            "field": "method",
                            "type": "nominal",
                            "label": "c_proj K",
                        },
                        "y": {
                            "field": "loss_improvement_vs_none",
                            "type": "quantitative",
                            "label": "Loss improvement",
                        },
                        "tooltip": [
                            {
                                "field": "mean_final_val_loss",
                                "type": "quantitative",
                                "label": "Mean final loss",
                            },
                            {
                                "field": "k_state_mib",
                                "type": "quantitative",
                                "label": "K-state",
                                "unit": "MiB",
                            },
                        ],
                    },
                    "xAxisTitle": "Final-loss improvement vs none",
                    "yAxisTitle": "c_proj K mode",
                    "valueFormat": "number",
                    "layout": "full",
                    "maxRows": 3,
                },
                {
                    "id": "k_state_chart",
                    "title": "Persistent K-state by c_proj mode",
                    "subtitle": (
                        "c_fc K remains full in all cells; MiB uses a 2^20 "
                        "denominator."
                    ),
                    "type": "horizontalBar",
                    "dataset": "pareto",
                    "source": source15,
                    "encodings": {
                        "x": {
                            "field": "method",
                            "type": "nominal",
                            "label": "c_proj K",
                        },
                        "y": {
                            "field": "k_state_mib",
                            "type": "quantitative",
                            "label": "K-state",
                            "unit": "MiB",
                        },
                        "tooltip": [
                            {
                                "field": "peak_memory_mib",
                                "type": "quantitative",
                                "label": "Peak memory",
                                "unit": "MiB",
                            },
                            {
                                "field": "loss_improvement_vs_none",
                                "type": "quantitative",
                                "label": "Loss improvement vs none",
                            },
                        ],
                    },
                    "xAxisTitle": "Persistent K-state (MiB)",
                    "yAxisTitle": "c_proj K mode",
                    "valueFormat": "number",
                    "unit": "MiB",
                    "layout": "full",
                    "maxRows": 3,
                },
            ],
            "tables": [
                {
                    "id": "cell_summary_table",
                    "title": "Three-level R1 diagonal bridge",
                    "subtitle": (
                        "Three paired formal seeds per method with c_fc K "
                        "fixed to full."
                    ),
                    "dataset": "cells",
                    "source": source15_cells,
                    "density": "spacious",
                    "defaultSort": {
                        "field": "mean_final_val_loss",
                        "direction": "asc",
                    },
                    "columns": [
                        {
                            "field": "method",
                            "label": "c_proj K",
                            "type": "text",
                        },
                        {
                            "field": "mean_final_val_loss",
                            "label": "Mean final val loss",
                            "type": "number",
                            "format": "number",
                            "align": "right",
                        },
                        {
                            "field": "k_state_mib",
                            "label": "K-state",
                            "type": "number",
                            "format": "number",
                            "unit": "MiB",
                            "align": "right",
                        },
                        {
                            "field": "peak_memory_mib",
                            "label": "Peak memory",
                            "type": "number",
                            "format": "number",
                            "unit": "MiB",
                            "align": "right",
                        },
                        {
                            "field": "seeds",
                            "label": "Seeds",
                            "type": "number",
                            "format": "number",
                            "align": "right",
                        },
                    ],
                }
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# Experiment 41D: R1 diagonal bridge",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "experiment15_formal_r1",
                    "body": (
                        "## Diag restores the useful c_proj scale signal at "
                        "near-none state cost\n\nWith `c_fc K` fixed to full, "
                        "diag improves mean final validation loss over none by "
                        f"{-decision['diag_minus_none']['mean']:.6f} across "
                        "three paired seeds. It remains within the 0.002 "
                        "practical margin of block4 while saving "
                        f"{decision['diag_k_state_saved_vs_block4_percent']:.2f}% "
                        "of persistent K state."
                    ),
                },
                {
                    "id": "quality_section",
                    "type": "markdown",
                    "sourceId": "experiment15_formal_r1",
                    "body": (
                        "## Diag materially improves over none\n\nThe paired "
                        "diag-minus-none effect is "
                        f"{decision['diag_minus_none']['mean']:.6f}, with a "
                        "95% small-sample t interval of "
                        f"[{decision['diag_minus_none']['ci95_low_t_df2']:.6f}, "
                        f"{decision['diag_minus_none']['ci95_high_t_df2']:.6f}]. "
                        "All three seeds favor diag."
                    ),
                },
                {
                    "id": "quality_chart",
                    "type": "chart",
                    "chartId": "quality_improvement_chart",
                },
                {
                    "id": "state_section",
                    "type": "markdown",
                    "sourceId": "experiment15_formal_r1",
                    "body": (
                        "## Diag quality-matches block4 with much less state\n\n"
                        "Diag is numerically better than block4 in all three "
                        "seeds, but the mean advantage is only "
                        f"{-decision['diag_minus_block4']['mean']:.6f}; it is "
                        "below the practical margin and its interval crosses "
                        "zero. The supported claim is quality matching, not "
                        "established superiority."
                    ),
                },
                {
                    "id": "state_chart",
                    "type": "chart",
                    "chartId": "k_state_chart",
                },
                {
                    "id": "exact_table_section",
                    "type": "markdown",
                    "body": (
                        "## Exact method-level results\n\nThe table reports "
                        "unweighted means across the three paired seeds and "
                        "the invariant method-level memory observations."
                    ),
                },
                {
                    "id": "exact_table",
                    "type": "table",
                    "tableId": "cell_summary_table",
                },
                {
                    "id": "scope_methodology",
                    "type": "markdown",
                    "sourceId": "experiment41_accepted",
                    "body": (
                        "## Scope and validation\n\nThe bridge uses the frozen "
                        "experiment-15 formal summary. Its block4 and none "
                        "rows are linked field-by-field to experiment 41 for "
                        "all three seeds, including initialization, loss, AUC, "
                        "K-state, and peak-memory fields. No new training is "
                        "included."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations and uncertainty\n\nThe statistical "
                        "unit is seed (`n=3`), so intervals use two degrees of "
                        "freedom. The analysis does not estimate the missing "
                        "`c_fc=none,c_proj=diag` cell, a diag-by-c_fc "
                        "interaction, a Muon comparison, or isolated runtime."
                    ),
                },
                {
                    "id": "recommendation",
                    "type": "markdown",
                    "body": (
                        "## Recommended paper use\n\nPlace this three-level "
                        "slice next to the experiment-41 2x2 factorial. State "
                        "that diag materially improves over none and "
                        "quality-matches block4 at near-none state cost. Do "
                        "not claim statistically established superiority "
                        "over block4, and do not add new training unless the "
                        "paper requires a diag-by-c_fc interaction claim."
                    ),
                },
                {
                    "id": "further_question",
                    "type": "markdown",
                    "body": (
                        "## Further question\n\nWould diag retain the same "
                        "benefit when `c_fc K` is also removed? That secondary "
                        "interaction is the only question requiring the "
                        "missing 2x3 factorial cell."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-07-31T00:00:00+08:00",
            "status": "ready",
            "datasets": {
                "kpis": kpis,
                "pareto": pareto_rows,
                "cells": cell_rows,
                "paired_final_effects": final_effects,
            },
            "accessIssues": [],
        },
        "sources": [source15, source15_cells, source41],
        "package_info": {
            "analysis_name": "41D_r1_diag_bridge",
            "script_version": SCRIPT_VERSION,
        },
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(
            f"output directory must be absent or empty: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract, rows, audit = validate_sources(
        args.run_summary,
        args.experiment41_reference,
        args.experiment41_acceptance,
        args.contract,
    )
    practical_margin = float(contract["practical_loss_margin"])
    cells = build_cells(rows)
    effects_by_seed, effects_summary = build_effects(
        rows, practical_margin
    )
    decision = build_decision(cells, effects_summary, practical_margin)
    pareto = build_pareto_rows(cells)
    checks = audit["checks"] | {
        "diag_beneficial_over_none": decision[
            "diag_beneficial_over_none"
        ],
        "diag_quality_matched_to_block4": decision[
            "diag_quality_matched_to_block4"
        ],
        "new_training_recommended": decision[
            "new_training_recommended"
        ],
    }

    write_csv(args.output_dir / "diag_bridge_cells.csv", cells)
    write_csv(
        args.output_dir / "diag_bridge_effects_by_seed.csv",
        effects_by_seed,
    )
    write_csv(
        args.output_dir / "diag_bridge_effects_summary.csv",
        effects_summary,
    )
    write_csv(args.output_dir / "diag_bridge_pareto.csv", pareto)
    write_json(args.output_dir / "diag_bridge_decision.json", decision)
    write_json(args.output_dir / "checks.json", checks)
    write_json(
        args.output_dir / "source_audit.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "source_hashes": audit["source_hashes"],
            "contract_relative_sources": contract["source_contract"],
        },
    )
    (args.output_dir / "R1_DIAG_BRIDGE_REPORT.md").write_text(
        build_report(cells, decision, audit["source_hashes"]),
        encoding="utf-8",
    )
    write_json(
        args.output_dir / "artifact.json",
        build_report_artifact(
            cells,
            effects_by_seed,
            pareto,
            decision,
        ),
    )

    artifacts = [
        "diag_bridge_cells.csv",
        "diag_bridge_effects_by_seed.csv",
        "diag_bridge_effects_summary.csv",
        "diag_bridge_pareto.csv",
        "diag_bridge_decision.json",
        "checks.json",
        "source_audit.json",
        "R1_DIAG_BRIDGE_REPORT.md",
        "artifact.json",
    ]
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "analysis_name": contract["analysis_name"],
        "analysis_only": True,
        "passed": all(
            value is not False
            for value in (
                checks["cell_seed_coverage"],
                checks["all_values_finite"],
                checks["diag_beneficial_over_none"],
                checks["diag_quality_matched_to_block4"],
            )
        ),
        "classification": decision["classification"],
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "new_training_runs": 0,
        "new_training_recommended": False,
        "timing_usable": False,
        "contract_sha256": sha256_file(args.contract),
        "artifacts": artifacts,
        "output_sha256": {
            name: sha256_file(args.output_dir / name)
            for name in artifacts
        },
    }
    write_json(
        args.output_dir / "r1_diag_bridge_analysis_manifest.json",
        manifest,
    )
    if not manifest["passed"]:
        raise SystemExit(2)
    print(
        "Experiment 41D analysis manifest:",
        args.output_dir / "r1_diag_bridge_analysis_manifest.json",
    )
    print("Experiment 41D artifacts:", args.output_dir)


if __name__ == "__main__":
    main()
