"""Build the source-pinned partial method-deepening package and figures."""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    ContractError,
    commit_manifest,
    ensure_new_output,
    mean,
    read_csv,
    read_json,
    sample_sd,
    sha256_file,
    write_csv,
)


AUDIT_SCHEMA = "mdp_method_deepening_package_v1"
STATUS_FIELDS = ["module_id", "status", "paper_eligible", "result", "remaining_gate"]
CLAIM_FIELDS = [
    "claim_id",
    "claim_class",
    "status",
    "scope",
    "allowed_wording",
    "forbidden_wording",
    "primary_evidence",
]
NEGATIVE_FIELDS = ["evidence_id", "finding", "implication", "source"]
MECHANISM_FIELDS = [
    "evidence_id",
    "evidence_class",
    "metric",
    "estimate",
    "uncertainty_or_support",
    "unit",
    "interpretation",
]
SOURCE_FIELDS = ["source_id", "relative_path", "sha256"]


def _source_paths(workspace_root: Path, audit_root: Path) -> dict[str, Path]:
    science = Path(
        os.environ.get("SNM_RESULTS_ROOT", str(workspace_root / "runs"))
    ).expanduser().resolve()
    return {
        "formulation_manifest": audit_root / "formulation" / "method_formulation_manifest.json",
        "complexity_csv": audit_root / "complexity" / "routing_complexity.csv",
        "complexity_manifest": audit_root / "complexity" / "routing_complexity_manifest.json",
        "equivariance_csv": audit_root / "equivariance" / "route_equivariance.csv",
        "equivariance_manifest": audit_root / "equivariance" / "route_equivariance_manifest.json",
        "inventory_manifest": audit_root / "inventory" / "method_deepening_inventory_manifest.json",
        "refresh_replay_contract": audit_root.parent / "refresh_replay_contract.json",
        "ex22_alpha": science / "22_r1_block_alpha" / "analysis" / "wandb_20260729_multiseed_confirmation" / "alpha_run_summary.csv",
        "ex22_results": science / "22_r1_block_alpha" / "analysis" / "wandb_20260729_multiseed_confirmation" / "important_results.json",
        "ex37_decision": science / "37_mech09_downproj_refresh_mediation" / "20260728T075907+0000" / "analysis" / "mediation_decision.json",
        "ex37_pairs": science / "37_mech09_downproj_refresh_mediation" / "20260728T075907+0000" / "analysis" / "paired_contrast_summary.csv",
        "ex37_auc": science / "37_mech09_downproj_refresh_mediation" / "20260728T075907+0000" / "analysis" / "auc_contrasts.csv",
        "ex37_manifest": science / "37_mech09_downproj_refresh_mediation" / "20260728T075907+0000" / "analysis" / "mech09r_analysis_manifest.json",
        "ex40_results": science / "40_llama_block_partition_invariance_audit" / "20260729T044926+0000" / "independent_review" / "important_results.json",
        "ex40_manifest": science / "40_llama_block_partition_invariance_audit" / "20260729T044926+0000" / "independent_review" / "independent_audit_manifest.json",
        "ex41_results": science / "41_r1_kstate_module_factorial" / "analysis" / "accepted_20260731" / "experiment41_key_results.json",
        "ex41d_decision": science / "41_r1_kstate_module_factorial" / "analysis" / "diag_bridge_20260731" / "diag_bridge_decision.json",
        "final_claims": science / "_shared" / "analysis" / "final_unified_38_45_20260803" / "bundle" / "unified_submission" / "claim_matrix.csv",
    }


def _validate_sources(paths: dict[str, Path], workspace_root: Path) -> list[dict[str, Any]]:
    rows = []
    for source_id, path in paths.items():
        if not path.is_file():
            raise ContractError(f"method-deepening source is missing: {source_id}={path}")
        try:
            relative = path.relative_to(workspace_root)
        except ValueError:
            relative = path
        rows.append(
            {
                "source_id": source_id,
                "relative_path": str(relative),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _plot_state_cost(complexity_rows: list[dict[str, str]], output: Path) -> None:
    cases = list(dict.fromkeys(row["case_id"] for row in complexity_rows))
    routes = ("full", "block", "diag", "none")
    colors = {"full": "#D55E00", "block": "#0072B2", "diag": "#009E73", "none": "#666666"}
    x = np.arange(len(cases), dtype=float)
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    for route_index, route in enumerate(routes):
        values = []
        measured = []
        for case in cases:
            row = next(
                item for item in complexity_rows
                if item["case_id"] == case and item["route"] == route
            )
            values.append(float(row["analytic_state_mib"]))
            measured.append(row["measured_state_bytes"] not in ("", None))
        positions = x + (route_index - 1.5) * width
        ax.bar(positions, values, width, label=route, color=colors[route], alpha=0.84)
        for position, value, observed in zip(positions, values, measured):
            if observed:
                ax.scatter(position, value, marker="o", s=30, facecolor="white", edgecolor="black", zorder=3)
    ax.set_yscale("symlog", linthresh=0.1, linscale=0.7)
    ax.set_ylabel("Incremental persistent K state (MiB, symlog)")
    ax.set_xticks(x, [case.replace("_increment", "").replace("_", "\n") for case in cases])
    ax.set_title("Routing state cost: analytic values with source-pinned measurements")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, frameon=False, loc="upper left")
    ax.text(
        0.995,
        0.02,
        "Open circles: measured increments; none is exactly zero",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    fig.savefig(
        output,
        format=output.suffix.lstrip("."),
        metadata={"Date": None, "Creator": "MDP local audit"},
    )
    plt.close(fig)


def _plot_equivariance(rows: list[dict[str, str]], output: Path) -> None:
    labels = [f"{row['route']}\n{row['transform'].replace('_', ' ')}" for row in rows]
    values = [float(row["relative_update_error"]) for row in rows]
    expected = [row["expected_invariant"].lower() == "true" for row in rows]
    colors = ["#0072B2" if invariant else "#D55E00" for invariant in expected]
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    positions = np.arange(len(rows))
    ax.bar(positions, values, color=colors, alpha=0.86)
    ax.axhline(1e-9, color="#0072B2", linestyle="--", linewidth=1, label="invariance tolerance")
    ax.axhline(1e-5, color="#D55E00", linestyle=":", linewidth=1.2, label="non-invariance floor")
    ax.set_yscale("log")
    ax.set_ylabel("Relative polar-factor equivariance error (log scale)")
    ax.set_xticks(positions, labels, rotation=22, ha="right")
    ax.set_title("Numerical reference checks for route transformation groups")
    ax.grid(axis="y", which="both", alpha=0.2)
    ax.legend(frameon=False)
    ax.text(
        0.995,
        0.02,
        "Numerical audit, not a formal proof",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
    )
    fig.savefig(
        output,
        format=output.suffix.lstrip("."),
        metadata={"Date": None, "Creator": "MDP local audit"},
    )
    plt.close(fig)


def _plot_alpha_refresh(
    alpha_rows: list[dict[str, str]], refresh_rows: list[dict[str, str]], output: Path
) -> None:
    dense = [
        row for row in alpha_rows
        if row["method"] in {"alpha0", "alpha0p25", "alpha0p50", "alpha0p75", "block4"}
    ]
    by_alpha: dict[float, list[float]] = defaultdict(list)
    for row in dense:
        by_alpha[float(row["alpha"])].append(float(row["final_val_loss"]))
    alphas = sorted(by_alpha)
    losses = [mean(by_alpha[value]) for value in alphas]
    loss_sd = [sample_sd(by_alpha[value]) for value in alphas]

    selected = [
        row for row in refresh_rows
        if int(row["optimizer_step"]) in {48, 80}
        and row["contrast"] in {
            "delayed_down_refresh_vs_production",
            "frozen_down_refresh_vs_production",
            "delayed_down_refresh_vs_frozen_down_refresh",
        }
    ]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.9), constrained_layout=True)
    left.errorbar(alphas, losses, yerr=loss_sd, marker="o", color="#0072B2", capsize=4)
    left.set_xlabel("Block-alpha interpolation coefficient")
    left.set_ylabel("Mean final validation loss ± seed SD")
    left.set_title("Experiment 22: interior alpha beats both endpoints")
    left.grid(alpha=0.25)

    palette = {
        "delayed_down_refresh_vs_production": "#0072B2",
        "frozen_down_refresh_vs_production": "#009E73",
        "delayed_down_refresh_vs_frozen_down_refresh": "#D55E00",
    }
    label_map = {
        "delayed_down_refresh_vs_production": "delayed − production",
        "frozen_down_refresh_vs_production": "frozen − production",
        "delayed_down_refresh_vs_frozen_down_refresh": "delayed − frozen",
    }
    for contrast in palette:
        rows = sorted(
            (row for row in selected if row["contrast"] == contrast),
            key=lambda row: int(row["optimizer_step"]),
        )
        right.errorbar(
            [int(row["optimizer_step"]) for row in rows],
            [float(row["mean_delta"]) for row in rows],
            yerr=[float(row["sd_across_paired_units"]) for row in rows],
            marker="o",
            capsize=4,
            color=palette[contrast],
            label=label_map[contrast],
        )
    right.axhline(0, color="black", linewidth=0.8)
    right.set_xlabel("Replay optimizer step")
    right.set_ylabel("Paired held-out loss difference ± nested-unit SD")
    right.set_title("Experiment 37: refresh intervention impulse")
    right.grid(alpha=0.25)
    right.legend(frameon=False, fontsize=9)
    fig.suptitle("Existing causal and interpolation evidence feeding the method-deepening package")
    fig.savefig(
        output,
        format=output.suffix.lstrip("."),
        metadata={"Date": None, "Creator": "MDP local audit"},
    )
    plt.close(fig)


def build(workspace_root: Path, audit_root: Path, output_dir: Path) -> dict[str, Any]:
    paths = _source_paths(workspace_root, audit_root)
    source_rows = _validate_sources(paths, workspace_root)
    formulation = read_json(paths["formulation_manifest"])
    complexity = read_json(paths["complexity_manifest"])
    equivariance = read_json(paths["equivariance_manifest"])
    inventory = read_json(paths["inventory_manifest"])
    ex22 = read_json(paths["ex22_results"])
    ex37 = read_json(paths["ex37_decision"])
    ex40 = read_json(paths["ex40_results"])
    ex41 = read_json(paths["ex41_results"])
    ex41d = read_json(paths["ex41d_decision"])
    if not (
        formulation.get("status") == "passed"
        and complexity.get("status") == "passed"
        and equivariance.get("status") == "passed"
        and inventory.get("mdp04_status") == "blocked_data"
        and ex37.get("classification") == "full_support"
    ):
        raise ContractError("method-deepening upstream gates do not match the frozen partial-package contract")

    status_rows = [
        {
            "module_id": "MDP-01",
            "status": "ready",
            "paper_eligible": True,
            "result": "budget-constrained static selective K-state routing is fully specified",
            "remaining_gate": "copy the frozen formulation into Methods without claiming a learned selector",
        },
        {
            "module_id": "MDP-02",
            "status": "ready",
            "paper_eligible": True,
            "result": "12 required measured increments reconcile exactly with analytic bytes across four cases",
            "remaining_gate": "keep persistent state separate from workspace and peak memory",
        },
        {
            "module_id": "MDP-03",
            "status": "ready_with_proof_text_required",
            "paper_eligible": True,
            "result": "8/8 numerical group checks pass; experiment 40 supplies network-level boundary evidence",
            "remaining_gate": "retain the formal proposition and rank assumptions; numerical checks are not proofs",
        },
        {
            "module_id": "MDP-04",
            "status": "blocked_data",
            "paper_eligible": False,
            "result": "loss-level causal mediation is accepted, but full paired K/inverse/matched-G tensors are absent locally",
            "remaining_gate": "run deterministic short replay or streaming metrics export on the original LLaMA host",
        },
    ]

    claim_rows = [
        {
            "claim_id": "MDP-C01",
            "claim_class": "definition",
            "status": "supported",
            "scope": "eligible matrix modules under a fixed static route and budget",
            "allowed_wording": "Selective K-state routing defines block, diag, and none preconditioners under a persistent-state budget.",
            "forbidden_wording": "The method learns or solves the optimal route automatically.",
            "primary_evidence": "MDP-01 formulation audit F01-F08",
        },
        {
            "claim_id": "MDP-C02",
            "claim_class": "analytic_plus_measured",
            "status": "supported",
            "scope": "R1/Record28/Record17 c_proj and LLaMA-1B down projection",
            "allowed_wording": "Wide contraction matrices are persistent K-state hotspots; diag and none sharply reduce that component.",
            "forbidden_wording": "All memory or peak-allocation savings equal the analytic K-state reduction.",
            "primary_evidence": "MDP-02 source-pinned 12/12 byte reconciliation; experiments 39 and 42",
        },
        {
            "claim_id": "MDP-C03",
            "claim_class": "proposition",
            "status": "supported_with_rank_conditions",
            "scope": "right-coordinate transformations and exact polar factors",
            "allowed_wording": "Full and none retain orthogonal equivariance; diag retains signed-permutation equivariance; block retains partition-preserving equivariance.",
            "forbidden_wording": "The numerical audit by itself proves the proposition, or block4 is architecture neutral.",
            "primary_evidence": "MDP-03 8/8 numerical references plus experiment 40",
        },
        {
            "claim_id": "MDP-C04",
            "claim_class": "theorem_identity",
            "status": "supported",
            "scope": "SPD regularized K before and after refresh",
            "allowed_wording": "The exact inverse jump obeys the resolvent identity and its spectral-norm bound.",
            "forbidden_wording": "The bound predicts long-horizon optimizer ranking or the full Muon update shock.",
            "primary_evidence": "MDP-01 F04-F06; exact algebra",
        },
        {
            "claim_id": "MDP-C05",
            "claim_class": "causal_empirical",
            "status": "supported_at_loss_level",
            "scope": "MECH-09R 4 origins x 3 nested replicas, short horizon",
            "allowed_wording": "Scheduled down-projection refresh causally mediates the observed short-horizon held-out-loss impulse.",
            "forbidden_wording": "The local package has already linked the impulse to full-matrix inverse or polar shocks.",
            "primary_evidence": "experiment 37 full_support; MDP-04 tensor inventory blocked_data",
        },
        {
            "claim_id": "MDP-C06",
            "claim_class": "cross_environment_empirical",
            "status": "not_supported_as_universal_rule",
            "scope": "GPT 124M/275M/455M and LLaMA evidence",
            "allowed_wording": "The preferred low-state route is environment dependent.",
            "forbidden_wording": "Diag or none is universally optimal, or model size alone determines the route.",
            "primary_evidence": "final unified claim F04 and negative route evidence",
        },
    ]

    negative_rows = [
        {
            "evidence_id": "NEG-01",
            "finding": "alpha=0.5 beats both dense alpha endpoints in all three R1 seeds; mean endpoint curvature is negative",
            "implication": "the evidence rejects a simple monotone less-curvature-is-always-better story",
            "source": "experiment 22",
        },
        {
            "evidence_id": "NEG-02",
            "finding": "diag is preferred at 124M, diag/none are interchangeable at 275M, and none has the best mean at 455M",
            "implication": "there is no universal static route ranking and no justified automatic selector objective yet",
            "source": "final unified experiments 43-45",
        },
        {
            "evidence_id": "NEG-03",
            "finding": "both c_fc and c_proj K factors improve R1 quality with an approximately additive interaction",
            "implication": "removing all c_proj curvature is not a universal mechanism claim",
            "source": "experiment 41",
        },
        {
            "evidence_id": "NEG-04",
            "finding": "contiguous LLaMA block4 has large drift under a function-preserving hidden-unit permutation",
            "implication": "fixed block boundaries are coordinate choices, not architecture-neutral objects",
            "source": "experiment 40",
        },
        {
            "evidence_id": "NEG-05",
            "finding": "no local paired refresh tensors or checkpoints are available",
            "implication": "matrix-level refresh mediation is blocked and cannot be filled with synthetic fixtures",
            "source": "MDP inventory",
        },
    ]

    pair_rows = read_csv(paths["ex37_pairs"])
    pair_lookup = {
        (row["contrast"], int(row["optimizer_step"])): row for row in pair_rows
    }
    auc_rows = read_csv(paths["ex37_auc"])
    auc_values: dict[str, list[float]] = defaultdict(list)
    for row in auc_rows:
        auc_values[row["contrast"]].append(float(row["auc_delta"]))
    curvature = ex22["curvature_summary"]
    factorial = ex41["final_val_loss_effects"]
    mechanism_rows = [
        {
            "evidence_id": "M01",
            "evidence_class": "empirical_interpolation",
            "metric": "mean_final_curvature_c",
            "estimate": curvature["mean_final_curvature_c"],
            "uncertainty_or_support": f"3/3 seeds negative; SD={curvature['sample_sd_final_curvature_c']:.6g}",
            "unit": "validation loss",
            "interpretation": "interior alpha is favored; monotone shrinkage is rejected",
        },
        {
            "evidence_id": "M02",
            "evidence_class": "causal_nested_replay",
            "metric": "delayed_minus_production_step48",
            "estimate": float(pair_lookup[("delayed_down_refresh_vs_production", 48)]["mean_delta"]),
            "uncertainty_or_support": f"SD={float(pair_lookup[('delayed_down_refresh_vs_production', 48)]['sd_across_paired_units']):.6g}; 12/12 negative",
            "unit": "held-out loss",
            "interpretation": "delaying the scheduled refresh protects immediately after production refresh",
        },
        {
            "evidence_id": "M03",
            "evidence_class": "causal_nested_replay",
            "metric": "delayed_minus_frozen_step80",
            "estimate": float(pair_lookup[("delayed_down_refresh_vs_frozen_down_refresh", 80)]["mean_delta"]),
            "uncertainty_or_support": f"SD={float(pair_lookup[('delayed_down_refresh_vs_frozen_down_refresh', 80)]['sd_across_paired_units']):.6g}; 12/12 positive",
            "unit": "held-out loss",
            "interpretation": "the delayed arm worsens after its own refresh relative to frozen",
        },
        {
            "evidence_id": "M04",
            "evidence_class": "causal_nested_replay",
            "metric": "frozen_minus_production_auc",
            "estimate": mean(auc_values["frozen_down_refresh_vs_production"]),
            "uncertainty_or_support": "12/12 nested units negative",
            "unit": "normalized short-horizon loss AUC",
            "interpretation": "freezing down-projection refresh protects across the replay horizon",
        },
        {
            "evidence_id": "M05",
            "evidence_class": "network_boundary_diagnostic",
            "metric": "pooled_block4_update_drift_median",
            "estimate": ex40["pooled_global_block4_update_drift"]["median"],
            "uncertainty_or_support": f"n=48; effect/control={ex40['effect_to_control_multiple']:.4g}x",
            "unit": "relative update drift",
            "interpretation": "fixed contiguous block4 is strongly coordinate-partition dependent",
        },
        {
            "evidence_id": "M06",
            "evidence_class": "factorial_empirical",
            "metric": "cfc_main_effect",
            "estimate": factorial["cfc_main"]["mean"],
            "uncertainty_or_support": "3/3 seeds beneficial",
            "unit": "final validation loss",
            "interpretation": "c_fc K contributes useful quality information",
        },
        {
            "evidence_id": "M07",
            "evidence_class": "factorial_empirical",
            "metric": "cproj_main_effect",
            "estimate": factorial["cproj_main"]["mean"],
            "uncertainty_or_support": "3/3 seeds beneficial",
            "unit": "final validation loss",
            "interpretation": "c_proj K also contributes useful quality information",
        },
        {
            "evidence_id": "M08",
            "evidence_class": "paired_empirical",
            "metric": "diag_minus_none",
            "estimate": ex41d["diag_minus_none"]["mean"],
            "uncertainty_or_support": "3/3 seeds negative; +0.28125 MiB K state",
            "unit": "final validation loss",
            "interpretation": "coordinatewise scale recovers the c_proj benefit at near-none state",
        },
    ]

    manifest_name = "method_deepening_package_manifest.json"
    ensure_new_output(output_dir, manifest_name)
    write_csv(output_dir / "method_deepening_status.csv", status_rows, STATUS_FIELDS)
    write_csv(output_dir / "claim_evidence_matrix.csv", claim_rows, CLAIM_FIELDS)
    write_csv(output_dir / "negative_evidence.csv", negative_rows, NEGATIVE_FIELDS)
    write_csv(output_dir / "mechanism_evidence_summary.csv", mechanism_rows, MECHANISM_FIELDS)
    write_csv(output_dir / "source_hashes.csv", source_rows, SOURCE_FIELDS)
    complexity_rows = read_csv(paths["complexity_csv"])
    equivariance_rows = read_csv(paths["equivariance_csv"])
    _plot_state_cost(complexity_rows, output_dir / "routing_state_cost.svg")
    _plot_equivariance(equivariance_rows, output_dir / "route_equivariance.svg")
    _plot_alpha_refresh(read_csv(paths["ex22_alpha"]), pair_rows, output_dir / "alpha_refresh_evidence.svg")
    outputs = [
        "method_deepening_status.csv",
        "claim_evidence_matrix.csv",
        "negative_evidence.csv",
        "mechanism_evidence_summary.csv",
        "source_hashes.csv",
        "routing_state_cost.svg",
        "route_equivariance.svg",
        "alpha_refresh_evidence.svg",
    ]
    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "partial",
        "claim_eligible": False,
        "component_status": {
            "MDP-01": "ready",
            "MDP-02": "ready",
            "MDP-03": "ready_with_proof_text_required",
            "MDP-04": "blocked_data",
        },
        "ready_component_count": 3,
        "blocked_component_count": 1,
        "scientific_reason_partial": "paired refresh matrices and matched-gradient snapshots are not available locally",
        "large_training_required": False,
        "next_action": "deterministic_short_replay_or_streaming_metric_export_on_original_llama_host",
        "source_sha256": {row["source_id"]: row["sha256"] for row in source_rows},
        "builder_script_sha256": sha256_file(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
    }
    commit_manifest(output_dir, manifest_name, result, outputs)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(
        args.workspace_root.resolve(),
        args.audit_root.resolve(),
        args.output_dir.resolve(),
    )
    print(
        f"method-deepening package built: status={result['status']} "
        f"ready={result['ready_component_count']} blocked={result['blocked_component_count']}"
    )


if __name__ == "__main__":
    main()
