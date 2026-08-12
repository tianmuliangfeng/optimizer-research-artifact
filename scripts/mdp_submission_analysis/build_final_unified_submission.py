#!/usr/bin/env python3
"""Build the final read-only submission synthesis for experiments 38--45.

The script consumes accepted analysis artifacts only. It does not read training
checkpoints, alter source results, or pool seeds across model scales.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ANALYSIS_DATE = "2026-08-03"
EXPECTED_SEEDS = {2024, 2025, 2026}
METHOD_ORDER = ["diag", "block4", "none", "mousse", "moonlight", "muon", "normuon", "adamw"]


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    require(isinstance(data, dict), f"expected JSON object: {path}")
    return data


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    require(value not in (None, ""), f"missing numeric field {key}")
    return float(value)


def fmt(value: float, digits: int = 6, signed: bool = False) -> str:
    prefix = "+" if signed else ""
    return format(value, f"{prefix}.{digits}f")


def verify_manifest_outputs(base: Path, manifest: dict[str, Any]) -> int:
    outputs = manifest.get("output_sha256", {})
    require(isinstance(outputs, dict), f"missing output_sha256 in {base}")
    checked = 0
    for relative, expected in outputs.items():
        target = base / relative
        require(target.is_file(), f"manifest output missing: {target}")
        require(sha256(target) == expected, f"manifest output hash mismatch: {target}")
        checked += 1
    return checked


def source_entry(path: Path, role: str, status: str) -> dict[str, Any]:
    require(path.is_file(), f"source missing: {path}")
    return {
        "role": role,
        "status": status,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def external_panel(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    require(set(groups) == set(METHOD_ORDER), "experiment 45 method coverage mismatch")

    panel: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_rows = groups[method]
        seeds = {int(row["seed"]) for row in method_rows}
        require(seeds == EXPECTED_SEEDS, f"experiment 45 seed coverage mismatch for {method}")
        losses = [as_float(row, "final_val_loss") for row in method_rows]
        optimizer = [as_float(row, "optimizer_state_mib") for row in method_rows]
        peak = [as_float(row, "peak_memory_mib") for row in method_rows]
        require(all(row["timing_eligible"].lower() == "false" for row in method_rows), "experiment 45 timing leak")
        panel.append(
            {
                "method": method,
                "display_name": method_rows[0]["display_name"],
                "n_seeds": len(method_rows),
                "final_val_mean": statistics.fmean(losses),
                "final_val_sample_sd": statistics.stdev(losses),
                "optimizer_state_mib_mean": statistics.fmean(optimizer),
                "peak_memory_mib_mean": statistics.fmean(peak),
            }
        )

    panel.sort(key=lambda row: (row["final_val_mean"], row["optimizer_state_mib_mean"]))
    mousse_loss = next(row["final_val_mean"] for row in panel if row["method"] == "mousse")
    for rank, row in enumerate(panel, start=1):
        row["quality_rank"] = rank
        row["delta_final_loss_vs_mousse"] = row["final_val_mean"] - mousse_loss
        dominators = []
        for other in panel:
            if other is row:
                continue
            no_worse = (
                other["final_val_mean"] <= row["final_val_mean"]
                and other["optimizer_state_mib_mean"] <= row["optimizer_state_mib_mean"]
            )
            strictly_better = (
                other["final_val_mean"] < row["final_val_mean"]
                or other["optimizer_state_mib_mean"] < row["optimizer_state_mib_mean"]
            )
            if no_worse and strictly_better:
                dominators.append(other["method"])
        row["optimizer_state_pareto_nondominated"] = not dominators
        row["dominated_by"] = ";".join(sorted(dominators))
    return panel


def validate_external_aggregate(panel: list[dict[str, Any]], aggregate_rows: list[dict[str, str]]) -> None:
    supplied = {row["method"]: row for row in aggregate_rows}
    require(set(supplied) == set(METHOD_ORDER), "experiment 45 aggregate method coverage mismatch")
    for row in panel:
        other = supplied[row["method"]]
        require(int(other["n_seeds"]) == row["n_seeds"], f"aggregate seed mismatch: {row['method']}")
        for key in ("final_val_mean", "final_val_sample_sd"):
            require(math.isclose(float(other[key]), row[key], abs_tol=1e-12), f"aggregate mismatch: {row['method']} {key}")


def render_external_svg(panel: list[dict[str, Any]], path: Path) -> None:
    width, height = 920, 560
    left, right, top, bottom = 90, 35, 55, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    states = [row["optimizer_state_mib_mean"] for row in panel]
    losses = [row["final_val_mean"] for row in panel]
    x_min, x_max = min(states) * 0.94, max(states) * 1.04
    y_min, y_max = min(losses) - 0.006, max(losses) + 0.006

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def y(value: float) -> float:
        return top + (value - y_min) / (y_max - y_min) * plot_h

    colors = {"diag": "#0B6E4F", "none": "#2E86AB", "mousse": "#C44536"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="460" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">124M external-neighbor quality/state panel</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for i in range(6):
        value = x_min + i * (x_max - x_min) / 5
        px = x(value)
        lines.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_h}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{px:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{value:.0f}</text>')
    for i in range(6):
        value = y_min + i * (y_max - y_min) / 5
        py = y(value)
        lines.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left - 10}" y="{py + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.3f}</text>')
    for row in panel:
        px, py = x(row["optimizer_state_mib_mean"]), y(row["final_val_mean"])
        color = colors.get(row["method"], "#6B7280")
        radius = 7 if row["method"] in colors else 5
        label = html.escape(row["display_name"])
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="{color}" stroke="white" stroke-width="1.5"/>')
        lines.append(f'<text x="{px + 9:.1f}" y="{py - 7:.1f}" font-family="Arial" font-size="12" fill="#111">{label}</text>')
    lines.extend(
        [
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="14">Optimizer state (MiB, lower is better)</text>',
            f'<text x="20" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 20 {top + plot_h / 2:.1f})">Final validation loss (lower is better)</text>',
            '<text x="90" y="545" font-family="Arial" font-size="11" fill="#555">Timing is excluded; points aggregate paired seeds 2024/2025/2026.</text>',
            '</svg>',
        ]
    )
    write_text(path, "\n".join(lines))


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def build_report(
    cross_aggregate: list[dict[str, str]],
    contrasts: list[dict[str, str]],
    panel: list[dict[str, Any]],
    mousse_contrasts: list[dict[str, str]],
    mechanism: dict[str, Any],
    efficiency_r1: dict[str, Any],
    invariance: dict[str, Any],
    factorial: dict[str, Any],
    diag_bridge: dict[str, Any],
    efficiency_llama: dict[str, Any],
    complexity_rows: list[dict[str, str]],
    equivariance_rows: list[dict[str, str]],
) -> str:
    aggregate_index = {(row["scale_id"], row["method"]): row for row in cross_aggregate}
    contrast_index = {(row["scale_id"], row["method_a"], row["method_b"]): row for row in contrasts}
    scale_rows = []
    for scale in ("gpt124m", "gpt275m", "gpt455m"):
        methods = [aggregate_index[(scale, method)] for method in ("muon", "original", "diag", "none")]
        best = min(methods, key=lambda row: float(row["final_loss_mean"]))
        scale_rows.append(
            [
                scale,
                f"{int(best['model_parameters']) / 1e6:.0f}M",
                f"{float(best['train_tokens']) / float(best['model_parameters']):.2f}",
                best["method"],
                fmt(float(best["final_loss_mean"])),
                contrast_index[(scale, "none", "original")]["interval_decision"],
                contrast_index[(scale, "diag", "original")]["interval_decision"],
            ]
        )

    panel_rows = []
    for row in sorted(panel, key=lambda item: item["quality_rank"]):
        panel_rows.append(
            [
                str(row["quality_rank"]),
                row["display_name"],
                f"{fmt(row['final_val_mean'])} ± {fmt(row['final_val_sample_sd'])}",
                fmt(row["delta_final_loss_vs_mousse"], signed=True),
                f"{row['optimizer_state_mib_mean']:.1f}",
                f"{row['peak_memory_mib_mean']:.0f}",
                "yes" if row["optimizer_state_pareto_nondominated"] else "no",
            ]
        )

    mousse_index = {row["contrast"]: row for row in mousse_contrasts}
    r1_eff = {row["method"]: row for row in efficiency_r1["isolated_efficiency"]["summary"]}
    llama_eff = {row["method"]: row for row in efficiency_llama["method_summary"]}
    diag = next(row for row in panel if row["method"] == "diag")
    none = next(row for row in panel if row["method"] == "none")
    mousse = next(row for row in panel if row["method"] == "mousse")
    block4 = next(row for row in panel if row["method"] == "block4")
    block_route = next(row for row in complexity_rows if row["case_id"] == "r1_124m_mlp_contraction" and row["route"] == "block")
    diag_route = next(row for row in complexity_rows if row["case_id"] == "r1_124m_mlp_contraction" and row["route"] == "diag")
    cross_block = next(row for row in equivariance_rows if row["route"] == "block" and row["transform"] == "cross_block_signed_permutation")

    report = f"""# Final unified analysis: experiments 38–45

Date: {ANALYSIS_DATE}  
Status: **claim-eligible with explicit scope caveats**  
Practical final-loss margin: **±0.002**

## Executive decision

The completed evidence supports the paper's central allocation claim, but not a universal optimizer-route ranking. Selective K-state routing preserves most or all of the original Newton–Muon quality benefit while substantially reducing state. The preferred low-state route changes with environment: diagonal is the clear 124M choice, diag and none are practically interchangeable at 275M, and none has the best endpoint mean at 455M. Because model size, architecture details, and token/parameter ratios are not jointly controlled, this is evidence for **environment-dependent allocation**, not a monotonic scaling law.

Mousse is a useful 124M external neighbor. It beats Muon, Moonlight, NorMuon, and AdamW in final loss, but ranks fourth of eight: diag, block4, and none all have lower means. It is also dominated on the final-loss/optimizer-state plane. These results do not trigger a 275M Mousse extension.

The earlier provisional hypothesis that the original-over-Muon gain and the none-over-original gain are of similar size does **not** survive as a general cross-scale statement. At 455M both increments favor none/original in sequence but the second is smaller; at 275M none is essentially tied with original; at 124M none is materially worse than original.

## Cross-scale quality evidence

{markdown_table(['Environment', 'Parameters', 'Tokens/parameter', 'Best endpoint mean', 'Best mean loss', 'none vs original', 'diag vs original'], scale_rows)}

Key paired results (seed is the unit; scales are never pooled):

- 124M: diag−Muon = **−0.016033** [−0.018522, −0.013545]; none−Muon = **−0.010467** [−0.011717, −0.009216]. Diag improves materially over none by **−0.005567** [−0.006841, −0.004292].
- 275M: none−original = **−0.000074** [−0.001937, +0.001788], satisfying the preregistered practical-equivalence interval. Diag−none = **+0.000239** [−0.001060, +0.001538], also practically equivalent.
- 455M: none−Muon = **−0.002335** [−0.002764, −0.001905] and none−original = **−0.000821** [−0.001904, +0.000262]. The former is directional and slightly beyond the mean materiality threshold; its interval straddles −0.002, so it is not labelled a robust material improvement.

The endpoint preference changes from diag at 124M to an effectively interchangeable diag/none pair at 275M and none at 455M. Token/parameter ratios are 26.21, 2.42, and 6.88 respectively, so scale and training budget are confounded.

## 124M external-neighbor panel

{markdown_table(['Rank', 'Method', 'Final loss mean ± SD', 'Δ vs Mousse', 'Optimizer state MiB', 'Peak MiB', 'Pareto'], panel_rows)}

Mousse−Muon is **{fmt(float(mousse_index['mousse_minus_muon']['paired_mean']), signed=True)}** and Mousse−block4 is **{fmt(float(mousse_index['mousse_minus_original_newton_muon']['paired_mean']), signed=True)}**. Diag−Mousse is **{fmt(float(mousse_index['selective_diag_minus_mousse']['paired_mean']), signed=True)}**; none−Mousse is **{fmt(float(mousse_index['selective_none_minus_mousse']['paired_mean']), signed=True)}**. The none advantage is within the practical margin by mean but remains directional in all three seeds; it should not be described as established equivalence.

Relative to Mousse, diag saves **{mousse['optimizer_state_mib_mean'] - diag['optimizer_state_mib_mean']:.1f} MiB** of optimizer state ({100 * (mousse['optimizer_state_mib_mean'] - diag['optimizer_state_mib_mean']) / mousse['optimizer_state_mib_mean']:.1f}%) and **{mousse['peak_memory_mib_mean'] - diag['peak_memory_mib_mean']:.0f} MiB** peak memory while lowering loss. None has nearly identical state cost. Block4 also lowers loss and uses **{mousse['optimizer_state_mib_mean'] - block4['optimizer_state_mib_mean']:.1f} MiB** less optimizer state.

## Mechanistic synthesis

The evidence chain is now coherent at three levels:

1. **Allocation:** Experiment 41 finds both full `c_fc` K and block4 `c_proj` K beneficial on R1, with approximately additive main effects: {fmt(factorial['final_val_loss_effects']['cfc_main']['mean'])} and {fmt(factorial['final_val_loss_effects']['cproj_main']['mean'])}. This rules out the literal claim that removing `c_proj` K universally improves quality.
2. **Compression of useful information:** Experiment 41D shows that diagonal `c_proj` K improves over none by {fmt(diag_bridge['diag_minus_none']['mean'])} and matches block4 within the ±0.002 practical margin, while adding only {diag_bridge['diag_extra_k_state_vs_none_mib']:.5f} MiB over none. The defensible interpretation is that per-coordinate scale information carries high value in this R1 slice, whereas dense/block cross-coordinate state is not needed to recover block4-level quality here.
3. **Dynamics and boundary:** Experiment 38 supports stage-dependent short-horizon behavior and a causal role for scheduled down-projection refresh in its frozen 1B intervention tree. Experiment 40 shows that contiguous LLaMA block4 has pooled median update drift {invariance['pooled_global_block4_update_drift']['median']:.4f}, {invariance['effect_to_control_multiple']:.2f}× the equivariant control. Thus block4 is not an architecture-neutral definition of original Newton–Muon.

The analytic state calculation is consistent with this interpretation: at width 3072, block routing stores {float(block_route['relative_to_full']) * 100:.2f}% of full matrix state, while diag stores {float(diag_route['relative_to_full']) * 100:.5f}%. The local equivariance check passes its expected invariances and measures cross-block block-route drift {float(cross_block['relative_update_error']):.4f}.

These diagnostics strengthen the method narrative but do not replace end-to-end training comparisons. Refresh mediation remains a short-horizon causal result, not a full-training quality claim.

## Efficiency and memory

- R1 isolated H100 evidence (Experiment 39): median tokens/s are Muon {r1_eff['muon']['median_tokens_per_s']:.0f}, original block4 {r1_eff['block4']['median_tokens_per_s']:.0f}, none {r1_eff['none']['median_tokens_per_s']:.0f}, and diag {r1_eff['diag']['median_tokens_per_s']:.0f}. None/diag save 216 MiB K state and 864 MiB measured peak memory versus block4, with a small throughput advantage over block4 but remaining slower than Muon.
- LLaMA-1B isolated H100 evidence (Experiment 42): none and diag are about 4.0% faster than `newton_full`, cut K state by about 70.65%, and reduce timed peak allocated memory by about 17.85%; both remain about 1.5% slower than Muon. These are technical repeats on one host, not cross-seed quality evidence.
- Timing from Experiments 41, 43, 44, and 45 is excluded because the protocols do not provide isolated, paper-eligible timing. Raw throughput must not be compared numerically between Experiments 39 and 42.

## Submission-level conclusions

1. **Supported central claim:** selective allocation can retain Newton–Muon quality benefits at much lower persistent state; the strongest route is environment dependent.
2. **Supported 124M deployed choice:** diag is the best-supported quality/state choice and outperforms Mousse in every paired seed.
3. **Supported larger-GPT choice:** none is the most economical route at 275M and 455M; it is practically equivalent to original at both scales and has the best endpoint mean at 455M.
4. **Supported external-baseline statement:** Mousse is stronger than Muon and the other external optimizer baselines tested at 124M, but it does not match the selective/original Newton–Muon group and costs substantially more optimizer state.
5. **Supported mechanistic statement:** useful K information is module- and architecture-dependent; diagonal scale information can recover the deployed R1 `c_proj` benefit at near-none incremental state.
6. **Not supported:** a universal diag or none ranking, a monotonic scale law, a general equality of the two sequential loss gains, architecture-neutral block4, or 43/44/45 timing comparisons.

## Gate decision

The final unified evidence analysis is complete and claim-eligible. No immediate new large training run is justified by Experiments 43–45. The next planned step is the separately scoped method-deepening package (refresh/stability and related local diagnostics), followed by freezing paper tables, figures, limitations, and wording. A 275M Mousse run remains unnecessary unless the paper's positioning changes and requires cross-scale external-baseline coverage as an explicit reviewer-facing claim.
"""
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    bundle = args.bundle_root.resolve()
    output = args.output_dir.resolve()
    results = Path(
        os.environ.get("SNM_RESULTS_ROOT", str(workspace / "runs"))
    ).expanduser().resolve()

    paths = {
        "generator": Path(__file__).resolve(),
        "pipeline": bundle / "local_pipeline_manifest.json",
        "bundle_audit": bundle / "audit" / "submission_bundle_audit_manifest.json",
        "cross_aggregate": bundle / "cross_scale" / "method_aggregate_by_scale.csv",
        "contrasts": bundle / "cross_scale" / "paired_contrasts_by_scale.csv",
        "complexity": bundle / "complexity" / "routing_complexity.csv",
        "equivariance": bundle / "equivariance" / "route_equivariance.csv",
        "source_ledger": bundle / "evidence" / "source_ledger.csv",
        "mechanism_manifest": results / "38_unified_mechanism_synthesis" / "20260729T071451+0000" / "unified_mechanism_manifest.json",
        "mechanism_claims": results / "38_unified_mechanism_synthesis" / "20260729T071451+0000" / "claim_evidence_matrix.csv",
        "efficiency_r1": results / "39_submission_efficiency_and_sensitivity" / "analysis" / "handoff_20260730_independent_review" / "derived" / "independent_audit.json",
        "invariance_results": results / "40_llama_block_partition_invariance_audit" / "20260729T044926+0000" / "independent_review" / "important_results.json",
        "invariance_audit": results / "40_llama_block_partition_invariance_audit" / "20260729T044926+0000" / "independent_review" / "independent_audit.json",
        "factorial": results / "41_r1_kstate_module_factorial" / "analysis" / "accepted_20260731" / "experiment41_key_results.json",
        "diag_bridge": results / "41_r1_kstate_module_factorial" / "analysis" / "diag_bridge_20260731" / "diag_bridge_decision.json",
        "diag_manifest": results / "41_r1_kstate_module_factorial" / "analysis" / "diag_bridge_20260731" / "r1_diag_bridge_analysis_manifest.json",
        "efficiency_llama": results / "42_llama1b_isolated_efficiency" / "20260729T105505+0000" / "independent_review" / "important_results.json",
        "efficiency_llama_audit": results / "42_llama1b_isolated_efficiency" / "20260729T105505+0000" / "independent_review" / "independent_audit.json",
        "external_runs": results / "45_r1_mousse_strong_baseline" / "analysis" / "formal_20260803_authoritative" / "r1_unified_eight_method_run_summary.csv",
        "external_aggregate": results / "45_r1_mousse_strong_baseline" / "analysis" / "formal_20260803_authoritative" / "r1_unified_eight_method_aggregate.csv",
        "mousse_contrasts": results / "45_r1_mousse_strong_baseline" / "analysis" / "formal_20260803_authoritative" / "r1_mousse_paired_aggregate.csv",
        "external_manifest": results / "45_r1_mousse_strong_baseline" / "analysis" / "formal_20260803_authoritative" / "analysis_manifest.json",
        "external_local_audit": results / "45_r1_mousse_strong_baseline" / "analysis" / "formal_20260803_authoritative" / "local_artifact_audit_manifest.json",
    }
    for path in paths.values():
        require(path.is_file(), f"required input missing: {path}")

    pipeline = read_json(paths["pipeline"])
    bundle_audit = read_json(paths["bundle_audit"])
    mechanism = read_json(paths["mechanism_manifest"])
    efficiency_r1 = read_json(paths["efficiency_r1"])
    invariance = read_json(paths["invariance_results"])
    invariance_audit = read_json(paths["invariance_audit"])
    factorial = read_json(paths["factorial"])
    diag_bridge = read_json(paths["diag_bridge"])
    diag_manifest = read_json(paths["diag_manifest"])
    efficiency_llama = read_json(paths["efficiency_llama"])
    efficiency_llama_audit = read_json(paths["efficiency_llama_audit"])
    external_manifest = read_json(paths["external_manifest"])
    external_local_audit = read_json(paths["external_local_audit"])

    checks = {
        "pipeline_passed": pipeline.get("status") == "passed" and pipeline.get("claim_eligible") is True,
        "bundle_audit_passed": bundle_audit.get("status") == "passed" and bundle_audit.get("error_count") == 0,
        "mechanism_passed": mechanism.get("passed") is True,
        "r1_efficiency_numerical_valid": efficiency_r1.get("overall", {}).get("numerical_evidence_valid") is True,
        "r1_efficiency_remote_rerun_not_required": efficiency_r1.get("overall", {}).get("remote_rerun_required") is False,
        "invariance_audit_passed": invariance_audit.get("passed") is True,
        "factorial_accepted": factorial.get("status") == "accepted_with_analysis_errata" and factorial.get("quality_usable") is True,
        "diag_bridge_passed": diag_manifest.get("passed") is True and diag_bridge.get("new_training_recommended") is False,
        "llama_efficiency_audit_passed": efficiency_llama_audit.get("passed") is True,
        "external_quality_eligible": external_manifest.get("status") == "completed_valid" and external_local_audit.get("quality_claim_eligible") is True,
        "external_timing_excluded": external_local_audit.get("timing_claim_eligible") is False,
        "refresh_deferred": pipeline.get("refresh_enabled") is False,
    }
    require(all(checks.values()), "one or more final unified analysis gates failed")

    mechanism_hash_checks = verify_manifest_outputs(paths["mechanism_manifest"].parent, mechanism)
    diag_hash_checks = verify_manifest_outputs(paths["diag_manifest"].parent, diag_manifest)

    cross_aggregate = read_csv(paths["cross_aggregate"])
    contrasts = read_csv(paths["contrasts"])
    source_ledger = read_csv(paths["source_ledger"])
    require(len(cross_aggregate) == 12, "expected 12 cross-scale aggregate rows")
    require(len(contrasts) == 15, "expected 15 cross-scale contrasts")
    require(len(source_ledger) == 4, "expected four non-duplicated source ledgers")
    require({row["analysis_family"] for row in source_ledger} == {"gpt_scale", "external_baselines_124m"}, "analysis-family split failed")

    external_runs = read_csv(paths["external_runs"])
    panel = external_panel(external_runs)
    validate_external_aggregate(panel, read_csv(paths["external_aggregate"]))
    mousse_contrasts = read_csv(paths["mousse_contrasts"])
    require(len(mousse_contrasts) == 7, "expected seven Mousse contrasts")
    mousse = next(row for row in panel if row["method"] == "mousse")
    require(mousse["quality_rank"] == 4, "unexpected Mousse quality rank")
    require(mousse["optimizer_state_pareto_nondominated"] is False, "unexpected Mousse Pareto status")

    complexity_rows = read_csv(paths["complexity"])
    equivariance_rows = read_csv(paths["equivariance"])
    require(all(row["check_passed"].lower() == "true" for row in equivariance_rows), "equivariance diagnostic failed")

    evidence_rows = [
        {"workstream": "GPT quality across scale", "experiments": "15,43,44", "evidence_class": "formal paired training", "status": "passed", "claim_use": "primary", "boundary": "stratify by scale; never pool seeds"},
        {"workstream": "124M external neighbors", "experiments": "45", "evidence_class": "formal paired training", "status": "passed_with_caveats", "claim_use": "primary at 124M", "boundary": "quality only; Mousse tested at 124M"},
        {"workstream": "Unified mechanism synthesis", "experiments": "38", "evidence_class": "read-only synthesis plus short-horizon interventions", "status": "passed", "claim_use": "mechanistic/supporting", "boundary": "does not replace full training"},
        {"workstream": "R1 efficiency and sensitivity", "experiments": "39", "evidence_class": "isolated H100 repeats", "status": "numerical_valid_report_regenerated", "claim_use": "efficiency/supporting", "boundary": "source report semantics were stale; use audited JSON and this report"},
        {"workstream": "LLaMA block invariance", "experiments": "40", "evidence_class": "read-only update diagnostic", "status": "ready_to_cite_with_scope_caveats", "claim_use": "architecture boundary", "boundary": "not optimizer-quality evidence"},
        {"workstream": "R1 K-state factorial and diag bridge", "experiments": "41,41D", "evidence_class": "paired training plus accepted analysis-only bridge", "status": "accepted_with_errata", "claim_use": "allocation/mechanism", "boundary": "diag bridge fixes c_fc K to full"},
        {"workstream": "LLaMA-1B isolated efficiency", "experiments": "42", "evidence_class": "single-host technical repeats", "status": "ready_to_cite_with_scope_caveats", "claim_use": "efficiency", "boundary": "not optimizer-quality evidence; do not compare raw rate to ex39"},
        {"workstream": "Refresh/stability deepening", "experiments": "future", "evidence_class": "paired tensor snapshot analysis", "status": "deferred_by_design", "claim_use": "none in current bundle", "boundary": "requires separately preregistered snapshots"},
    ]

    claim_rows = [
        {"claim_id": "F01", "status": "supported", "scope": "GPT 124M", "claim": "Selective diag materially improves final loss over Muon and is the best-supported quality/state route.", "primary_evidence": "15/41D", "allowed_wording": "diag improves over Muon and matches/slightly improves block4 within the practical margin", "forbidden_wording": "diag is universally superior"},
        {"claim_id": "F02", "status": "supported", "scope": "GPT 275M", "claim": "Selective none preserves original Newton–Muon endpoint quality at lower state.", "primary_evidence": "43", "allowed_wording": "paired practical equivalence to original", "forbidden_wording": "all selective routes significantly beat Muon"},
        {"claim_id": "F03", "status": "supported", "scope": "GPT 455M", "claim": "Selective none preserves original quality and has the best endpoint mean.", "primary_evidence": "44", "allowed_wording": "equivalent to original; directional improvement over Muon", "forbidden_wording": "robust material improvement over Muon"},
        {"claim_id": "F04", "status": "supported", "scope": "cross-scale", "claim": "The preferred low-state route is environment dependent.", "primary_evidence": "15/43/44", "allowed_wording": "diag→interchangeable→none pattern across environments", "forbidden_wording": "monotonic parameter-scaling law"},
        {"claim_id": "F05", "status": "supported_with_caveat", "scope": "GPT 124M", "claim": "Mousse is a strong external baseline but does not match the selective/original group and is state-heavy.", "primary_evidence": "45", "allowed_wording": "Mousse beats Muon/external neighbors; ranks 4/8 and is optimizer-state dominated", "forbidden_wording": "Mousse is weak or universally inferior"},
        {"claim_id": "F06", "status": "supported", "scope": "R1 c_proj with c_fc full", "claim": "Diagonal K recovers block4-level quality at near-none incremental state.", "primary_evidence": "41D", "allowed_wording": "matches block4 within ±0.002 and materially beats none", "forbidden_wording": "diag benefit is independent of c_fc K"},
        {"claim_id": "F07", "status": "supported", "scope": "R1 allocation", "claim": "Both c_fc and c_proj K factors benefit quality and are approximately additive.", "primary_evidence": "41", "allowed_wording": "architecture-dependent quality/state allocation", "forbidden_wording": "removing c_proj K improves R1 quality"},
        {"claim_id": "F08", "status": "supported_with_scope_caveats", "scope": "R1 and LLaMA-1B isolated H100", "claim": "Selective routes reduce full-K memory/state and recover some full-K throughput overhead.", "primary_evidence": "39/42", "allowed_wording": "report each protocol separately", "forbidden_wording": "raw throughput is comparable across experiments or selective beats Muon speed/memory"},
        {"claim_id": "F09", "status": "supported", "scope": "LLaMA block4 diagnostic", "claim": "Contiguous block4 is coordinate-partition dependent and not architecture neutral.", "primary_evidence": "40", "allowed_wording": "newton_full remains the LLaMA full-K control", "forbidden_wording": "block4 full-training quality is disproven"},
        {"claim_id": "F10", "status": "not_supported", "scope": "cross-scale", "claim": "Original-over-Muon and none-over-original gains are generally equal.", "primary_evidence": "15/43/44", "allowed_wording": "report the scale-specific decomposition", "forbidden_wording": "present the provisional pattern as a replicated law"},
    ]

    limitation_rows = [
        {"limitation_id": "L01", "area": "statistics", "limitation": "Only 3–4 paired seeds per scale.", "mitigation": "paired t intervals, direction counts, practical margin, no pooled p-value"},
        {"limitation_id": "L02", "area": "scaling", "limitation": "Model size, recipe, and tokens/parameter change together.", "mitigation": "call environments replications, not a controlled scaling law"},
        {"limitation_id": "L03", "area": "external baseline", "limitation": "Mousse is formal only at 124M.", "mitigation": "scope the claim to 124M; 275M extension not triggered"},
        {"limitation_id": "L04", "area": "timing", "limitation": "43/44/45 and concurrent 41 timing are ineligible.", "mitigation": "use isolated Experiments 39 and 42 only"},
        {"limitation_id": "L05", "area": "mechanism", "limitation": "Refresh evidence is short-horizon and snapshot-limited.", "mitigation": "reserve full refresh/stability study for method deepening"},
        {"limitation_id": "L06", "area": "architecture", "limitation": "LLaMA block4 is coordinate dependent.", "mitigation": "use newton_full as LLaMA original-family control"},
        {"limitation_id": "L07", "area": "report provenance", "limitation": "Experiment 39's historical report text is stale despite valid numerics.", "mitigation": "cite the independent audit JSON and regenerated unified report"},
        {"limitation_id": "L08", "area": "checkpoint payload", "limitation": "Experiment 45 archive omits checkpoint tensors.", "mitigation": "quality is accepted via hashes, complete metrics, W&B exact match, and remote checkpoint metadata; no new checkpoint-dependent claim"},
    ]

    report = build_report(
        cross_aggregate,
        contrasts,
        panel,
        mousse_contrasts,
        mechanism,
        efficiency_r1,
        invariance,
        factorial,
        diag_bridge,
        efficiency_llama,
        complexity_rows,
        equivariance_rows,
    )

    if args.dry_run:
        print(f"final unified analysis validated: sources={len(paths)} claims={len(claim_rows)} dry_run=True")
        return 0

    require(not output.exists() or not any(output.iterdir()), f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_text(output / "FINAL_UNIFIED_ANALYSIS.md", report)
    write_csv(
        output / "external_neighbor_124m.csv",
        ["quality_rank", "method", "display_name", "n_seeds", "final_val_mean", "final_val_sample_sd", "delta_final_loss_vs_mousse", "optimizer_state_mib_mean", "peak_memory_mib_mean", "optimizer_state_pareto_nondominated", "dominated_by"],
        panel,
    )
    write_csv(output / "evidence_workstream_status.csv", list(evidence_rows[0]), evidence_rows)
    write_csv(output / "claim_matrix.csv", list(claim_rows[0]), claim_rows)
    write_csv(output / "limitations_matrix.csv", list(limitation_rows[0]), limitation_rows)
    render_external_svg(panel, output / "external_neighbor_quality_state.svg")

    source_status = {
        "pipeline": "passed",
        "bundle_audit": "passed",
        "mechanism": "passed",
        "r1_efficiency": "numerical_valid_report_regeneration_required",
        "invariance": "ready_to_cite_with_scope_caveats",
        "factorial": "accepted_with_analysis_errata",
        "diag_bridge": "passed",
        "llama_efficiency": "ready_to_cite_with_scope_caveats",
        "external": "quality_claim_eligible_timing_ineligible",
    }
    source_roles = {
        "generator": "final unified analysis generator",
        "pipeline": "formal analysis pipeline",
        "bundle_audit": "formal bundle audit",
        "mechanism_manifest": "mechanism synthesis",
        "mechanism_claims": "mechanism claim matrix",
        "efficiency_r1": "R1 isolated efficiency audit",
        "invariance_results": "LLaMA block invariance results",
        "invariance_audit": "LLaMA block invariance audit",
        "factorial": "R1 module factorial acceptance",
        "diag_bridge": "R1 diag bridge decision",
        "diag_manifest": "R1 diag bridge audit",
        "efficiency_llama": "LLaMA-1B efficiency results",
        "efficiency_llama_audit": "LLaMA-1B efficiency audit",
        "external_runs": "124M external-neighbor runs",
        "external_aggregate": "124M external-neighbor aggregate",
        "mousse_contrasts": "Mousse paired contrasts",
        "external_manifest": "Experiment 45 analysis manifest",
        "external_local_audit": "Experiment 45 local acceptance audit",
        "cross_aggregate": "cross-scale method aggregate",
        "contrasts": "cross-scale paired contrasts",
        "complexity": "routing state complexity",
        "equivariance": "routing equivariance diagnostic",
        "source_ledger": "deduplicated evidence-source ledger",
    }
    source_entries = []
    for key, role in source_roles.items():
        status_key = key
        if key.startswith("mechanism"):
            status_key = "mechanism"
        elif key == "efficiency_r1":
            status_key = "r1_efficiency"
        elif key.startswith("invariance"):
            status_key = "invariance"
        elif key.startswith("diag"):
            status_key = "diag_bridge"
        elif key.startswith("efficiency_llama"):
            status_key = "llama_efficiency"
        elif key.startswith("external") or key == "mousse_contrasts":
            status_key = "external"
        elif key in {"generator", "cross_aggregate", "contrasts", "complexity", "equivariance", "source_ledger"}:
            status_key = "pipeline"
        source_entries.append(source_entry(paths[key], role, source_status[status_key]))

    outputs = {}
    for name in ("FINAL_UNIFIED_ANALYSIS.md", "external_neighbor_124m.csv", "evidence_workstream_status.csv", "claim_matrix.csv", "limitations_matrix.csv", "external_neighbor_quality_state.svg"):
        target = output / name
        outputs[name] = {"sha256": sha256(target), "bytes": target.stat().st_size}
    manifest = {
        "schema_version": "final_unified_analysis_38_45_v1",
        "analysis_date": ANALYSIS_DATE,
        "status": "passed_with_caveats",
        "synthetic": False,
        "claim_eligible": True,
        "cross_scale_seed_pooling": False,
        "practical_loss_margin": 0.002,
        "mousse_scope": "124M external-neighbor panel only",
        "mousse_275m_extension_triggered": False,
        "refresh_analysis_included": False,
        "checks": checks,
        "hash_checks": {"mechanism_outputs": mechanism_hash_checks, "diag_bridge_outputs": diag_hash_checks},
        "source_files": source_entries,
        "outputs": outputs,
    }
    write_text(output / "final_unified_analysis_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"final unified analysis passed: output={output} claims={len(claim_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
