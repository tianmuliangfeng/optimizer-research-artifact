#!/usr/bin/env python3
"""Independent, read-only validation of a completed 38 analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


PRIMARY = {
    "selective_diag_vs_muon",
    "selective_none_vs_muon",
    "selective_diag_vs_original_newton_muon",
    "selective_none_vs_original_newton_muon",
}
BASELINE = "original_newton_muon_vs_muon"
FOUNDATIONAL_MODES = {"diag", "none", "block4", "dense_full", "muon"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(
        (args.run_dir / "unified_mechanism_manifest.json").read_text(encoding="utf-8")
    )
    if not manifest["passed"] or manifest["diag_vs_none_primary"]:
        raise RuntimeError("manifest acceptance failed")
    for name, expected in manifest["output_sha256"].items():
        observed = sha256_file(args.run_dir / name)
        if observed != expected:
            raise RuntimeError(f"output hash mismatch: {name}")

    primary = read_csv(args.run_dir / "primary_training_contrasts.csv")
    by_family: dict[str, set[str]] = {}
    for row in primary:
        by_family.setdefault(row["family"], set()).add(row["contrast"])
    expected_contrasts = PRIMARY | {BASELINE}
    if set(by_family) != {"r1", "llama124", "llama1b"}:
        raise RuntimeError(f"unexpected families: {set(by_family)}")
    if any(contrasts != expected_contrasts for contrasts in by_family.values()):
        raise RuntimeError(f"primary contrast coverage failed: {by_family}")

    raw = read_csv(
        args.input_root
        / "36_mech08_short_horizon_rollout/20260727T102506+0000/analysis/paired_contrasts.csv"
    )
    output = read_csv(args.run_dir / "rollout_cluster_bootstrap.csv")
    for row in output:
        if row["checkpoint_stage"] != "all":
            continue
        selected = [
            source
            for source in raw
            if source["contrast"] == row["contrast"]
            and source["metric"] == row["metric"]
            and source["optimizer_step"] == row["optimizer_step"]
        ]
        expected_mean = statistics.mean(
            float(source["delta_left_minus_right"]) for source in selected
        )
        if not math.isclose(
            expected_mean, float(row["mean_delta"]), rel_tol=0.0, abs_tol=1e-15
        ):
            raise RuntimeError(f"rollout mean mismatch: {row}")

    alpha = read_csv(args.run_dir / "alpha_synthesis.csv")
    if {row["topology"] for row in alpha} != {"block", "dense_full"}:
        raise RuntimeError("alpha topology coverage failed")
    if not all(
        row["all_seed_curvatures_negative"] == "True"
        and row["alpha0p50_beats_both_endpoints_all_seeds"] == "True"
        for row in alpha
    ):
        raise RuntimeError("alpha confirmation mismatch")

    foundational = read_csv(args.run_dir / "foundational_module_structure.csv")
    if len(foundational) != 10:
        raise RuntimeError(f"foundational row count mismatch: {len(foundational)}")
    for dataset in ("OWT", "WikiText-103"):
        rows = [row for row in foundational if row["dataset"] == dataset]
        if {row["mode"] for row in rows} != FOUNDATIONAL_MODES:
            raise RuntimeError(f"{dataset}: foundational mode coverage failed")
        if any(int(row["seeds"]) != 3 for row in rows):
            raise RuntimeError(f"{dataset}: foundational seed coverage failed")
        ranked = {
            int(row["final_loss_rank"]): row["mode"] for row in rows
        }
        if ranked[1] != "diag" or ranked[2] != "none":
            raise RuntimeError(f"{dataset}: unexpected historical ranking {ranked}")

    owt_raw = read_csv(
        args.input_root
        / "06_kstate_spectrum/summaries/combined_all_seeds_comparison.csv"
    )
    for source_mode, output_mode in (
        ("diag", "diag"),
        ("none", "none"),
        ("block4", "block4"),
        ("full", "dense_full"),
        ("muon", "muon"),
    ):
        expected = statistics.mean(
            float(row["final_val_loss"])
            for row in owt_raw
            if row["mode"] == source_mode
        )
        observed = next(
            float(row["mean_final_val_loss"])
            for row in foundational
            if row["dataset"] == "OWT" and row["mode"] == output_mode
        )
        if not math.isclose(expected, observed, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"OWT foundational mean mismatch: {output_mode}")

    bridge = read_csv(args.run_dir / "complementary_bridge_summary.csv")
    if {row["dataset"] for row in bridge} != {"OWT", "WikiText-103"}:
        raise RuntimeError("complementary bridge dataset coverage failed")
    if any(
        row["classification"] != "cproj_only_worse_all_seeds"
        or int(row["cproj_only_worse_seeds"]) != 3
        or float(row["paired_delta_ci95_low_t_df2"]) <= 0
        for row in bridge
    ):
        raise RuntimeError("complementary bridge direction failed")

    architecture = read_csv(
        args.run_dir / "architecture_transfer_boundary.csv"
    )
    if len(architecture) != 1:
        raise RuntimeError(
            f"architecture transfer row count mismatch: {len(architecture)}"
        )
    boundary = architecture[0]
    if (
        boundary["classification"] != "strong_non_invariance"
        or boundary["block4_is_original_newton_muon"] != "False"
        or boundary["block4_is_primary_baseline"] != "False"
        or boundary["official_original_newton_muon_control"] != "newton_full"
    ):
        raise RuntimeError("architecture transfer interpretation failed")
    expected_boundary = json.loads(
        (
            args.input_root
            / "40_llama_block_partition_invariance_audit"
            / "20260729T044926+0000"
            / "analysis"
            / "classification.json"
        ).read_text(encoding="utf-8")
    )
    for field in (
        "pooled_global_block4_median_update_drift",
        "maximum_equivariant_control_drift",
        "effect_to_control_multiple",
    ):
        expected = float(expected_boundary["decision_statistics"][field])
        observed = float(boundary[field])
        if not math.isclose(expected, observed, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"architecture transfer value mismatch: {field}")

    claims = read_csv(args.run_dir / "claim_evidence_matrix.csv")
    if len(claims) != manifest["claims"] or any(not row["status"] for row in claims):
        raise RuntimeError("claim matrix validation failed")
    print(
        json.dumps(
            {
                "passed": True,
                "output_hashes": len(manifest["output_sha256"]),
                "formal_contrast_rows": len(primary),
                "rollout_rows_recomputed": sum(
                    row["checkpoint_stage"] == "all" for row in output
                ),
                "alpha_topologies": len(alpha),
                "foundational_rows": len(foundational),
                "bridge_datasets": len(bridge),
                "architecture_transfer_rows": len(architecture),
                "claims": len(claims),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
