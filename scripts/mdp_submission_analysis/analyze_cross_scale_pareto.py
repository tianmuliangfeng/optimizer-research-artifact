"""Run paired, scale-stratified loss and state-efficiency analyses.

The script deliberately forbids a pooled cross-scale seed estimate. Each scale is
an independent replication environment, not an additional batch of IID seeds.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    PRACTICAL_LOSS_MARGIN,
    ContractError,
    commit_manifest,
    ensure_new_output,
    mean,
    mean_ci95,
    optional_float,
    read_csv,
    read_json,
    sample_sd,
    sha256_file,
    write_csv,
    atomic_write_text,
)


ANALYSIS_SCHEMA = "mdp_cross_scale_analysis_v1"
CORE_METHODS = ("muon", "original", "diag", "none")
CONTRASTS = (
    ("diag", "muon", "primary_quality_gain_vs_muon"),
    ("diag", "original", "primary_compression_cost_vs_original"),
    ("none", "muon", "primary_quality_gain_vs_muon"),
    ("none", "original", "primary_compression_cost_vs_original"),
    ("diag", "none", "secondary_diag_vs_none"),
)

AGGREGATE_FIELDS = [
    "scale_id",
    "architecture",
    "model_parameters",
    "train_tokens",
    "method",
    "n_seeds",
    "final_loss_mean",
    "final_loss_sd",
    "tail5_mean",
    "normalized_auc_mean",
    "k_state_bytes_mean",
    "optimizer_state_bytes_mean",
    "peak_memory_allocated_bytes_mean",
]

DELTA_FIELDS = [
    "scale_id",
    "seed",
    "method_a",
    "method_b",
    "contrast_role",
    "delta_final_loss_a_minus_b",
    "delta_tail5_a_minus_b",
    "delta_auc_a_minus_b",
]

CONTRAST_FIELDS = [
    "scale_id",
    "architecture",
    "method_a",
    "method_b",
    "contrast_role",
    "n_pairs",
    "mean_delta_final_loss_a_minus_b",
    "sd_delta_final_loss",
    "ci95_low",
    "ci95_high",
    "practical_margin",
    "mean_effect_label",
    "interval_decision",
]

PARETO_FIELDS = [
    "scale_id",
    "state_metric",
    "method",
    "final_loss_mean",
    "state_bytes_mean",
    "pareto_nondominated",
    "dominated_by",
]


def _float(row: dict[str, str], field: str) -> float:
    value = optional_float(row, field)
    if not math.isfinite(value):
        raise ContractError(f"evidence ledger row has missing/non-finite {field}: {row}")
    return value


def _effect_label(delta: float, margin: float) -> str:
    if delta < -margin:
        return "materially_better"
    if delta > margin:
        return "materially_worse"
    return "within_practical_margin"


def _interval_decision(low: float, high: float, margin: float) -> str:
    if high < -margin:
        return "robust_material_improvement"
    if low > margin:
        return "robust_material_degradation"
    if low >= -margin and high <= margin:
        return "practical_equivalence_supported"
    return "inconclusive"


def _pareto_rows(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_scale: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for aggregate in aggregates:
        by_scale[str(aggregate["scale_id"])].append(aggregate)
    for scale_id, scale_rows in sorted(by_scale.items()):
        for state_metric in ("k_state_bytes_mean", "optimizer_state_bytes_mean"):
            candidates = [row for row in scale_rows if math.isfinite(float(row[state_metric]))]
            for row in candidates:
                dominators = []
                for other in candidates:
                    if other is row:
                        continue
                    loss_better = float(other["final_loss_mean"]) <= float(row["final_loss_mean"])
                    state_better = float(other[state_metric]) <= float(row[state_metric])
                    strict = (
                        float(other["final_loss_mean"]) < float(row["final_loss_mean"])
                        or float(other[state_metric]) < float(row[state_metric])
                    )
                    if loss_better and state_better and strict:
                        dominators.append(str(other["method"]))
                rows.append(
                    {
                        "scale_id": scale_id,
                        "state_metric": state_metric.removesuffix("_mean"),
                        "method": row["method"],
                        "final_loss_mean": row["final_loss_mean"],
                        "state_bytes_mean": row[state_metric],
                        "pareto_nondominated": not dominators,
                        "dominated_by": ";".join(sorted(dominators)),
                    }
                )
    return rows


def analyze(
    ledger_dir: Path,
    output_dir: Path,
    practical_margin: float = PRACTICAL_LOSS_MARGIN,
    family: str = "gpt_scale",
) -> dict[str, Any]:
    ledger_path = ledger_dir / "evidence_ledger.csv"
    ledger_manifest_path = ledger_dir / "evidence_ledger_manifest.json"
    if not ledger_path.is_file() or not ledger_manifest_path.is_file():
        raise ContractError(f"missing committed evidence ledger in {ledger_dir}")
    ledger_manifest = read_json(ledger_manifest_path)
    if ledger_manifest.get("status") != "validated":
        raise ContractError("evidence ledger manifest is not validated")
    if sha256_file(ledger_path) != ledger_manifest.get("outputs", {}).get("evidence_ledger.csv"):
        raise ContractError("evidence ledger hash does not match its committed manifest")
    rows = [row for row in read_csv(ledger_path) if row.get("analysis_family") == family]
    if not rows:
        raise ContractError(f"ledger contains no rows for analysis_family={family!r}")

    cells: dict[tuple[str, str, str], dict[str, str]] = {}
    scale_meta: dict[str, tuple[str, str, str]] = {}
    methods_by_scale: dict[str, set[str]] = defaultdict(set)
    seeds_by_scale: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        scale = row["scale_id"]
        method = row["method"]
        seed = row["seed"]
        if method not in CORE_METHODS:
            continue
        key = (scale, seed, method)
        if key in cells:
            raise ContractError(f"duplicate core cell in ledger: {key}")
        cells[key] = row
        methods_by_scale[scale].add(method)
        seeds_by_scale[scale].add(seed)
        meta = (row.get("architecture", ""), row.get("model_parameters", ""), row.get("train_tokens", ""))
        if scale in scale_meta and scale_meta[scale] != meta:
            raise ContractError(f"inconsistent scale metadata for {scale}")
        scale_meta[scale] = meta

    if len(methods_by_scale) < 2:
        raise ContractError("cross-scale analysis requires at least two independent scales")
    for scale in sorted(methods_by_scale):
        if methods_by_scale[scale] != set(CORE_METHODS):
            raise ContractError(
                f"{scale}: expected core methods {list(CORE_METHODS)}, got {sorted(methods_by_scale[scale])}"
            )
        for seed in seeds_by_scale[scale]:
            missing = [method for method in CORE_METHODS if (scale, seed, method) not in cells]
            if missing:
                raise ContractError(f"{scale}/{seed}: incomplete paired cells: {missing}")

    aggregate_rows: list[dict[str, Any]] = []
    for scale in sorted(methods_by_scale):
        architecture, parameters, tokens = scale_meta[scale]
        for method in CORE_METHODS:
            method_rows = [cells[(scale, seed, method)] for seed in sorted(seeds_by_scale[scale])]
            finals = [_float(row, "final_val_loss") for row in method_rows]
            aggregate_rows.append(
                {
                    "scale_id": scale,
                    "architecture": architecture,
                    "model_parameters": parameters,
                    "train_tokens": tokens,
                    "method": method,
                    "n_seeds": len(method_rows),
                    "final_loss_mean": mean(finals),
                    "final_loss_sd": sample_sd(finals),
                    "tail5_mean": mean(_float(row, "tail5_mean") for row in method_rows),
                    "normalized_auc_mean": mean(_float(row, "normalized_auc") for row in method_rows),
                    "k_state_bytes_mean": mean(optional_float(row, "k_state_bytes") for row in method_rows),
                    "optimizer_state_bytes_mean": mean(optional_float(row, "optimizer_state_bytes") for row in method_rows),
                    "peak_memory_allocated_bytes_mean": mean(optional_float(row, "peak_memory_allocated_bytes") for row in method_rows),
                }
            )

    delta_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for scale in sorted(methods_by_scale):
        architecture = scale_meta[scale][0]
        for method_a, method_b, role in CONTRASTS:
            deltas: list[float] = []
            for seed in sorted(seeds_by_scale[scale]):
                row_a = cells[(scale, seed, method_a)]
                row_b = cells[(scale, seed, method_b)]
                delta = _float(row_a, "final_val_loss") - _float(row_b, "final_val_loss")
                deltas.append(delta)
                delta_rows.append(
                    {
                        "scale_id": scale,
                        "seed": seed,
                        "method_a": method_a,
                        "method_b": method_b,
                        "contrast_role": role,
                        "delta_final_loss_a_minus_b": delta,
                        "delta_tail5_a_minus_b": _float(row_a, "tail5_mean") - _float(row_b, "tail5_mean"),
                        "delta_auc_a_minus_b": _float(row_a, "normalized_auc") - _float(row_b, "normalized_auc"),
                    }
                )
            center, sd, low, high = mean_ci95(deltas)
            contrast_rows.append(
                {
                    "scale_id": scale,
                    "architecture": architecture,
                    "method_a": method_a,
                    "method_b": method_b,
                    "contrast_role": role,
                    "n_pairs": len(deltas),
                    "mean_delta_final_loss_a_minus_b": center,
                    "sd_delta_final_loss": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "practical_margin": practical_margin,
                    "mean_effect_label": _effect_label(center, practical_margin),
                    "interval_decision": _interval_decision(low, high, practical_margin),
                }
            )

    pareto_rows = _pareto_rows(aggregate_rows)
    synthetic = bool(ledger_manifest.get("synthetic", False))
    watermark = "\n> **SYNTHETIC TEST DATA — INVALID FOR SCIENTIFIC CLAIMS.**\n" if synthetic else ""
    report_lines = [
        "# Cross-scale paired analysis",
        watermark,
        "Each scale is analyzed independently. No seeds are pooled across scales, and no cross-scale p-value is reported.",
        "",
        f"Practical loss margin: ±{practical_margin:.4f}.",
        "",
        "| Scale | Contrast (A − B) | n | Mean Δ final loss | 95% CI | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in contrast_rows:
        report_lines.append(
            f"| {row['scale_id']} | {row['method_a']} − {row['method_b']} | {row['n_pairs']} | "
            f"{row['mean_delta_final_loss_a_minus_b']:+.6f} | "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {row['interval_decision']} |"
        )
    report_lines.extend(
        [
            "",
            "Interpretation is scale-stratified: consistent signs across scales are replication evidence, not extra IID samples.",
        ]
    )

    manifest_name = "cross_scale_analysis_manifest.json"
    ensure_new_output(output_dir, manifest_name)
    write_csv(output_dir / "method_aggregate_by_scale.csv", aggregate_rows, AGGREGATE_FIELDS)
    write_csv(output_dir / "paired_deltas_by_scale_seed.csv", delta_rows, DELTA_FIELDS)
    write_csv(output_dir / "paired_contrasts_by_scale.csv", contrast_rows, CONTRAST_FIELDS)
    write_csv(output_dir / "pareto_frontier_by_scale.csv", pareto_rows, PARETO_FIELDS)
    atomic_write_text(output_dir / "CROSS_SCALE_ANALYSIS.md", "\n".join(report_lines) + "\n")
    result = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "passed",
        "synthetic": synthetic,
        "claim_eligible": bool(ledger_manifest.get("claim_eligible", False)) and not synthetic,
        "analysis_family": family,
        "scale_count": len(methods_by_scale),
        "scales": sorted(methods_by_scale),
        "seed_pooling_across_scales": False,
        "practical_loss_margin": practical_margin,
        "input_ledger_sha256": sha256_file(ledger_path),
        "input_ledger_manifest_sha256": sha256_file(ledger_manifest_path),
    }
    commit_manifest(
        output_dir,
        manifest_name,
        result,
        [
            "method_aggregate_by_scale.csv",
            "paired_deltas_by_scale_seed.csv",
            "paired_contrasts_by_scale.csv",
            "pareto_frontier_by_scale.csv",
            "CROSS_SCALE_ANALYSIS.md",
        ],
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--practical-margin", type=float, default=PRACTICAL_LOSS_MARGIN)
    parser.add_argument("--family", default="gpt_scale")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(
        args.ledger_dir.resolve(), args.output_dir.resolve(), args.practical_margin, args.family
    )
    print(
        f"cross-scale analysis passed: scales={result['scale_count']} "
        f"pooled={result['seed_pooling_across_scales']} synthetic={result['synthetic']}"
    )


if __name__ == "__main__":
    main()
