"""Build deterministic, dependency-free Markdown tables and SVG review figures."""

from __future__ import annotations

import argparse
import html
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from common import (
    MIB,
    ContractError,
    atomic_write_text,
    commit_manifest,
    ensure_new_output,
    optional_float,
    read_csv,
    read_json,
    sha256_file,
)


FIGURE_SCHEMA = "mdp_submission_tables_figures_v1"
COLORS = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"]


def _svg_document(width: int, height: int, body: Iterable[str], synthetic: bool) -> str:
    watermark = ""
    if synthetic:
        watermark = (
            f'<text x="{width / 2:.0f}" y="{height / 2:.0f}" text-anchor="middle" '
            'font-size="28" fill="#dc2626" opacity="0.18" '
            'transform="rotate(-18 {0:.0f} {1:.0f})">SYNTHETIC — NOT FOR CLAIMS</text>'.format(
                width / 2, height / 2
            )
        )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<g font-family="Arial, Helvetica, sans-serif" fill="#111827">',
            *body,
            watermark,
            "</g>",
            "</svg>",
        ]
    ) + "\n"


def _forest_svg(rows: list[dict[str, str]], synthetic: bool) -> str:
    selected = [
        row
        for row in rows
        if row["contrast_role"] != "secondary_diag_vs_none"
    ]
    values = []
    for row in selected:
        values.extend(
            [
                optional_float(row, "ci95_low"),
                optional_float(row, "ci95_high"),
                -abs(optional_float(row, "practical_margin")),
                abs(optional_float(row, "practical_margin")),
            ]
        )
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        raise ContractError("forest plot has no finite contrast intervals")
    maximum = max(abs(min(finite)), abs(max(finite)), 0.0025) * 1.15
    width = 980
    left, right, top, row_height = 355, 950, 70, 34
    height = top + len(selected) * row_height + 70

    def x(value: float) -> float:
        return left + (value + maximum) / (2 * maximum) * (right - left)

    body = [
        '<text x="20" y="30" font-size="20" font-weight="bold">Scale-stratified paired final-loss contrasts</text>',
        '<text x="20" y="52" font-size="12" fill="#4b5563">Negative favors method A; 95% paired t intervals; no cross-scale pooling.</text>',
        f'<line x1="{x(0):.1f}" y1="60" x2="{x(0):.1f}" y2="{height - 40}" stroke="#111827" stroke-width="1"/>',
    ]
    margin = abs(optional_float(selected[0], "practical_margin"))
    body.append(
        f'<rect x="{x(-margin):.1f}" y="60" width="{x(margin)-x(-margin):.1f}" height="{height-100}" fill="#e5e7eb" opacity="0.55"/>'
    )
    scale_order = {scale: index for index, scale in enumerate(sorted({row["scale_id"] for row in selected}))}
    for index, row in enumerate(selected):
        y = top + index * row_height
        low = optional_float(row, "ci95_low")
        high = optional_float(row, "ci95_high")
        center = optional_float(row, "mean_delta_final_loss_a_minus_b")
        label = f"{row['scale_id']}: {row['method_a']} − {row['method_b']}"
        color = COLORS[scale_order[row["scale_id"]] % len(COLORS)]
        body.extend(
            [
                f'<text x="20" y="{y+4}" font-size="12">{html.escape(label)}</text>',
                f'<line x1="{x(low):.1f}" y1="{y}" x2="{x(high):.1f}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                f'<line x1="{x(low):.1f}" y1="{y-5}" x2="{x(low):.1f}" y2="{y+5}" stroke="{color}"/>',
                f'<line x1="{x(high):.1f}" y1="{y-5}" x2="{x(high):.1f}" y2="{y+5}" stroke="{color}"/>',
                f'<circle cx="{x(center):.1f}" cy="{y}" r="4" fill="{color}"/>',
                f'<text x="{right+5}" y="{y+4}" font-size="10">{center:+.4f}</text>',
            ]
        )
    for tick in (-maximum, -maximum / 2, 0.0, maximum / 2, maximum):
        body.extend(
            [
                f'<line x1="{x(tick):.1f}" y1="{height-42}" x2="{x(tick):.1f}" y2="{height-37}" stroke="#111827"/>',
                f'<text x="{x(tick):.1f}" y="{height-22}" text-anchor="middle" font-size="10">{tick:+.4f}</text>',
            ]
        )
    return _svg_document(width, height, body, synthetic)


def _pareto_svg(rows: list[dict[str, str]], synthetic: bool) -> str:
    selected = [
        row for row in rows if row["state_metric"] == "optimizer_state_bytes"
    ]
    if not selected:
        selected = [row for row in rows if row["state_metric"] == "k_state_bytes"]
    points = []
    for row in selected:
        state = optional_float(row, "state_bytes_mean") / MIB
        loss = optional_float(row, "final_loss_mean")
        if math.isfinite(state) and math.isfinite(loss):
            points.append((row, state, loss))
    if not points:
        raise ContractError("Pareto plot has no finite state/loss points")
    width, height = 900, 560
    left, right, top, bottom = 90, 865, 65, 485
    states = [point[1] for point in points]
    losses = [point[2] for point in points]
    xmin, xmax = min(states), max(states)
    ymin, ymax = min(losses), max(losses)
    xpad = max((xmax - xmin) * 0.08, 0.1)
    ypad = max((ymax - ymin) * 0.12, 0.0005)
    xmin, xmax, ymin, ymax = xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad

    def x(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * (right - left)

    def y(value: float) -> float:
        return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

    body = [
        '<text x="20" y="30" font-size="20" font-weight="bold">Quality–state Pareto view</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#111827"/>',
        f'<text x="{(left+right)/2:.0f}" y="535" text-anchor="middle" font-size="12">Optimizer state (MiB; lower is better)</text>',
        f'<text x="20" y="{(top+bottom)/2:.0f}" transform="rotate(-90 20 {(top+bottom)/2:.0f})" text-anchor="middle" font-size="12">Final validation loss (lower is better)</text>',
    ]
    scales = sorted({row[0]["scale_id"] for row in points})
    colors = {scale: COLORS[index % len(COLORS)] for index, scale in enumerate(scales)}
    for row, state, loss in points:
        color = colors[row["scale_id"]]
        nondominated = row["pareto_nondominated"].lower() == "true"
        body.extend(
            [
                f'<circle cx="{x(state):.1f}" cy="{y(loss):.1f}" r="{6 if nondominated else 4}" fill="{color}" stroke="{color}" fill-opacity="{1 if nondominated else 0.25}"/>',
                f'<text x="{x(state)+7:.1f}" y="{y(loss)-5:.1f}" font-size="10">{html.escape(row["method"])} ({html.escape(row["scale_id"])})</text>',
            ]
        )
    for index, scale in enumerate(scales):
        body.extend(
            [
                f'<rect x="{650 + index*85}" y="22" width="10" height="10" fill="{colors[scale]}"/>',
                f'<text x="{664 + index*85}" y="31" font-size="10">{html.escape(scale)}</text>',
            ]
        )
    return _svg_document(width, height, body, synthetic)


def build(analysis_dir: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = analysis_dir / "cross_scale_analysis_manifest.json"
    if not manifest_path.is_file():
        raise ContractError(f"missing cross-scale manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    required_files = {
        "method_aggregate_by_scale.csv": analysis_dir / "method_aggregate_by_scale.csv",
        "paired_contrasts_by_scale.csv": analysis_dir / "paired_contrasts_by_scale.csv",
        "pareto_frontier_by_scale.csv": analysis_dir / "pareto_frontier_by_scale.csv",
    }
    for name, path in required_files.items():
        if not path.is_file() or sha256_file(path) != manifest.get("outputs", {}).get(name):
            raise ContractError(f"missing or hash-mismatched cross-scale artifact: {path}")
    aggregates = read_csv(required_files["method_aggregate_by_scale.csv"])
    contrasts = read_csv(required_files["paired_contrasts_by_scale.csv"])
    pareto = read_csv(required_files["pareto_frontier_by_scale.csv"])
    synthetic = bool(manifest.get("synthetic", False))

    lines = ["# Submission-ready analysis tables", ""]
    if synthetic:
        lines.extend(["> **SYNTHETIC TEST DATA — INVALID FOR SCIENTIFIC CLAIMS.**", ""])
    lines.extend(
        [
            "## Scale-stratified final loss",
            "",
            "| Scale | Method | Seeds | Final loss (mean ± SD) | Tail-5 mean | Normalized AUC | K state (MiB) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregates:
        state = optional_float(row, "k_state_bytes_mean") / MIB
        state_text = f"{state:.3f}" if math.isfinite(state) else "NA"
        lines.append(
            f"| {row['scale_id']} | {row['method']} | {row['n_seeds']} | "
            f"{float(row['final_loss_mean']):.6f} ± {float(row['final_loss_sd']):.6f} | "
            f"{float(row['tail5_mean']):.6f} | {float(row['normalized_auc_mean']):.6f} | {state_text} |"
        )
    lines.extend(
        [
            "",
            "## Paired contrasts",
            "",
            "| Scale | A − B | n | Mean Δ final loss | 95% CI | Interval decision |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in contrasts:
        lines.append(
            f"| {row['scale_id']} | {row['method_a']} − {row['method_b']} | {row['n_pairs']} | "
            f"{float(row['mean_delta_final_loss_a_minus_b']):+.6f} | "
            f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}] | {row['interval_decision']} |"
        )
    lines.extend(
        [
            "",
            "Scales are reported as separate replication environments; seed counts are never pooled across model sizes.",
        ]
    )

    manifest_name = "submission_tables_figures_manifest.json"
    ensure_new_output(output_dir, manifest_name)
    atomic_write_text(output_dir / "SUBMISSION_TABLES.md", "\n".join(lines) + "\n")
    atomic_write_text(output_dir / "cross_scale_forest.svg", _forest_svg(contrasts, synthetic))
    atomic_write_text(output_dir / "quality_state_pareto.svg", _pareto_svg(pareto, synthetic))
    result = {
        "schema_version": FIGURE_SCHEMA,
        "status": "passed",
        "synthetic": synthetic,
        "claim_eligible": bool(manifest.get("claim_eligible", False)) and not synthetic,
        "input_manifest_sha256": sha256_file(manifest_path),
        "scale_count": len({row["scale_id"] for row in aggregates}),
    }
    commit_manifest(
        output_dir,
        manifest_name,
        result,
        ["SUBMISSION_TABLES.md", "cross_scale_forest.svg", "quality_state_pareto.svg"],
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(args.analysis_dir.resolve(), args.output_dir.resolve())
    print(
        f"submission tables/figures passed: scales={result['scale_count']} synthetic={result['synthetic']}"
    )


if __name__ == "__main__":
    main()
