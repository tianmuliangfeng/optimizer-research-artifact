"""Build the eight-method controlled-R1 Mousse panel and paired contrasts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


SEEDS = (2024, 2025, 2026)
METHOD_ORDER = ("diag", "none", "mousse", "muon", "block4", "moonlight", "normuon", "adamw")
CONTRASTS = (
    ("selective_diag_minus_mousse", "diag", "mousse", "primary"),
    ("selective_none_minus_mousse", "none", "mousse", "primary"),
    ("mousse_minus_muon", "mousse", "muon", "anchor"),
    ("mousse_minus_original_newton_muon", "mousse", "block4", "anchor"),
    ("mousse_minus_moonlight", "mousse", "moonlight", "external_background"),
    ("mousse_minus_normuon", "mousse", "normuon", "external_background"),
    ("mousse_minus_adamw", "mousse", "adamw", "external_background"),
)
DISPLAY = {
    "diag": "Newton-Muon diag", "none": "Newton-Muon none", "mousse": "Mousse-R1",
    "muon": "Muon", "block4": "Newton-Muon block4", "moonlight": "Moonlight Muon",
    "normuon": "NorMuon", "adamw": "AdamW",
}
METHOD_MAP = {"moonlight_r1scale": "moonlight", "normuon_r1scale": "normuon", "adamw_low": "adamw"}
T_CRIT_DF2 = 4.302652729911275
EQUIVALENCE_MARGIN = 0.002


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_summary(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        return path
    matches = list(path.glob("formal_summary.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one formal_summary.csv under {path}")
    return matches[0]


def locate_formal_manifest(summary: Path) -> Path:
    path = summary.with_name("formal_manifest.json")
    if not path.is_file():
        raise RuntimeError(f"Mousse summary has no sibling formal_manifest.json: {summary}")
    return path


def normalized_row(method: str, seed: int, row: dict[str, str], family: str) -> dict[str, object]:
    optimizer_state_mib = row.get("optimizer_state_mib")
    if not optimizer_state_mib and row.get("optimizer_state_bytes"):
        optimizer_state_mib = str(float(row["optimizer_state_bytes"]) / 1024**2)
    return {
        "method": method,
        "display_name": DISPLAY[method],
        "family": family,
        "run_name": row.get("run_name", ""),
        "seed": seed,
        "seed_role": "tuning_seed" if seed == 2026 else "confirmatory_seed",
        "initial_val_loss": float(row["initial_val_loss"]),
        "final_val_loss": float(row["final_val_loss"]),
        "best_val_loss": float(row["best_val_loss"]),
        "tail5_val_loss_mean": float(row["tail5_val_loss_mean"]),
        "normalized_val_auc": float(row["normalized_val_auc"]),
        "peak_memory_mib": float(row.get("peak_memory_mib") or row.get("peak_memory_allocated_mib") or "nan"),
        "optimizer_state_mib": float(optimizer_state_mib or "nan"),
        "timing_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mousse-summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--core-summary", type=Path, required=True)
    parser.add_argument("--extended-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [locate_summary(path) for path in args.mousse_summaries]
    mousse_manifest_paths = [locate_formal_manifest(path) for path in source_paths]
    mousse_manifests = [json.loads(path.read_text(encoding="utf-8")) for path in mousse_manifest_paths]
    for manifest in mousse_manifests:
        if manifest.get("status") not in ("completed_valid", "completed_valid_local_wandb_incomplete"):
            raise RuntimeError("Mousse formal manifest is not locally valid")
        if manifest.get("protocol") != "mousse_r1_selected_6200step_v1" or manifest.get("total_steps") != 6200:
            raise RuntimeError("Mousse formal manifest protocol/budget mismatch")
        if manifest.get("summary", {}).get("evidence_valid") is not True:
            raise RuntimeError("Mousse formal manifest has no valid local summary")
    source_fingerprints = {
        json.dumps(manifest.get("source_audit", {}), sort_keys=True) for manifest in mousse_manifests
    }
    runtime_fingerprints = {
        json.dumps(manifest.get("training_runtime_fingerprint", {}), sort_keys=True)
        for manifest in mousse_manifests
    }
    if len(source_fingerprints) != 1 or len(runtime_fingerprints) != 1:
        raise RuntimeError("Mousse formal seeds do not share one source/runtime fingerprint")
    mousse_raw = [row for path in source_paths for row in read_csv(path)]
    if len(mousse_raw) != 3:
        raise RuntimeError(f"expected three Mousse formal rows, observed {len(mousse_raw)}")
    mousse_rows: list[dict[str, object]] = []
    for row in mousse_raw:
        seed = int(row.get("controlled_seed") or row.get("seed") or -1)
        if row.get("method") != "mousse" or int(row["total_steps"]) != 6200 or int(row["total_tokens"]) != 3_250_585_600:
            raise RuntimeError(f"invalid Mousse formal row: {row}")
        mousse_rows.append(normalized_row("mousse", seed, row, "experiment45"))
    if {int(row["seed"]) for row in mousse_rows} != set(SEEDS):
        raise RuntimeError("Mousse formal seed coverage must be exactly 2024/2025/2026")

    core_path, extended_path = args.core_summary.resolve(), args.extended_summary.resolve()
    core_analysis_manifest = core_path.with_name("analysis_manifest.json")
    extended_analysis_manifest = extended_path.with_name("analysis_manifest.json")
    if not core_analysis_manifest.is_file() or not extended_analysis_manifest.is_file():
        raise RuntimeError("historical R1 analysis manifests are required")
    core_analysis = json.loads(core_analysis_manifest.read_text(encoding="utf-8"))
    extended_analysis = json.loads(extended_analysis_manifest.read_text(encoding="utf-8"))
    if core_analysis.get("status") != "PASS_WITH_CAVEATS" or extended_analysis.get("status") != "PASS_WITH_CAVEATS":
        raise RuntimeError("historical R1 analysis status is not the frozen accepted status")
    if int(core_analysis.get("quality_checks", {}).get("PASS", 0)) < 180:
        raise RuntimeError("core R1 quality-check coverage is incomplete")
    if int(extended_analysis.get("quality_checks", {}).get("PASS", 0)) < 195:
        raise RuntimeError("extended R1 quality-check coverage is incomplete")
    core_rows = [row for row in read_csv(core_path) if row["method"] in ("diag", "none", "muon", "block4")]
    extended_rows = [row for row in read_csv(extended_path) if row["method"] in METHOD_MAP]
    unified = list(mousse_rows)
    for row in core_rows:
        unified.append(normalized_row(row["method"], int(row["seed"]), row, "core_frozen"))
    for row in extended_rows:
        unified.append(normalized_row(METHOD_MAP[row["method"]], int(row["seed"]), row, "extended_frozen"))
    expected_pairs = {(method, seed) for method in METHOD_ORDER for seed in SEEDS}
    observed_pairs = {(str(row["method"]), int(row["seed"])) for row in unified}
    if observed_pairs != expected_pairs or len(unified) != len(expected_pairs):
        raise RuntimeError(f"eight-method coverage mismatch: missing={sorted(expected_pairs - observed_pairs)}")

    by_pair = {(str(row["method"]), int(row["seed"])): row for row in unified}
    identity_failures = []
    for seed in SEEDS:
        initials = {round(float(by_pair[(method, seed)]["initial_val_loss"]), 4) for method in METHOD_ORDER}
        if len(initials) != 1:
            identity_failures.append(f"seed {seed} initial validation losses differ: {sorted(initials)}")
    historical_manifest_caveat = (
        "Historical experiment-15/19 rows are frozen W&B exports plus accepted analysis "
        "manifests; their original per-run local source/runtime/checkpoint manifests are not "
        "present in this evidence bundle. Quality pairing is accepted at the frozen protocol "
        "level, but this limitation must be disclosed and timing comparisons remain prohibited."
    )
    identity = {
        "status": "passed_with_caveats" if not identity_failures else "failed",
        "protocol": "mousse_r1_historical_identity_reuse_v1",
        "paired_quality_eligible": not identity_failures,
        "strict_per_run_local_manifest_identity": False,
        "evidence_level": "Mousse local manifests plus historical W&B exports/frozen analysis manifests",
        "checks": {
            "seed_coverage": list(SEEDS), "model": "same pinned 12x12x768 R1 source derivation",
            "data_batch_schedule_validation": "same pinned R1 controller contract",
            "auxiliary_route": "same tied embedding/head fused AdamW",
            "packed_qkv": "same Q/K/V logical split",
            "initial_validation_equal_within_print_precision": not identity_failures,
            "mousse_source_runtime_manifest_consistency": True,
            "historical_quality_checks": {"core": 180, "extended": 195},
            "timing_eligible": False,
        },
        "source_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in [*source_paths, *mousse_manifest_paths, core_path, extended_path, core_analysis_manifest, extended_analysis_manifest]
        ],
        "failures": identity_failures,
        "caveats": [historical_manifest_caveat],
        "consequence_if_failed": "historical rows are background-only and paired claims are prohibited",
    }
    (output / "identity_reuse_certificate.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    if identity_failures:
        raise RuntimeError("historical identity audit failed: " + "; ".join(identity_failures))

    unified.sort(key=lambda row: (int(row["seed"]), METHOD_ORDER.index(str(row["method"]))))
    write_csv(output / "r1_unified_eight_method_run_summary.csv", unified)
    deltas: list[dict[str, object]] = []
    aggregate: list[dict[str, object]] = []
    for label, left, right, role in CONTRASTS:
        values = []
        for seed in SEEDS:
            delta = float(by_pair[(left, seed)]["final_val_loss"]) - float(by_pair[(right, seed)]["final_val_loss"])
            values.append(delta)
            deltas.append({"contrast": label, "role": role, "left": left, "right": right, "seed": seed, "delta_final_val_loss": delta})
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        half = T_CRIT_DF2 * sd / math.sqrt(3)
        aggregate.append(
            {
                "contrast": label, "role": role, "left": left, "right": right,
                "n_seeds": 3, "paired_mean": mean, "paired_sample_sd": sd,
                "paired_t_ci95_low": mean - half, "paired_t_ci95_high": mean + half,
                "left_better_count": sum(value < 0 for value in values),
                "right_better_count": sum(value > 0 for value in values),
                "practical_equivalence_margin": EQUIVALENCE_MARGIN,
                "mean_within_equivalence_margin": abs(mean) <= EQUIVALENCE_MARGIN,
            }
        )
    write_csv(output / "r1_mousse_paired_seed_deltas.csv", deltas)
    write_csv(output / "r1_mousse_paired_aggregate.csv", aggregate)
    method_aggregate = []
    for method in METHOD_ORDER:
        values = [float(by_pair[(method, seed)]["final_val_loss"]) for seed in SEEDS]
        method_aggregate.append({"method": method, "display_name": DISPLAY[method], "n_seeds": 3, "final_val_mean": statistics.mean(values), "final_val_sample_sd": statistics.stdev(values)})
    method_aggregate.sort(key=lambda row: float(row["final_val_mean"]))
    write_csv(output / "r1_unified_eight_method_aggregate.csv", method_aggregate)

    lines = [
        "# Experiment 45 controlled 124M R1 Mousse analysis", "",
        "Historical identity/reuse certificate: **passed with caveats**. Timing remains diagnostic only.",
        historical_manifest_caveat, "",
        "## Eight-method endpoint", "",
        "| rank | method | final val mean | seed SD |", "|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(method_aggregate, 1):
        lines.append(f"| {rank} | {row['display_name']} | {float(row['final_val_mean']):.6f} | {float(row['final_val_sample_sd']):.6f} |")
    lines.extend(["", "## Preregistered paired contrasts", "", "Negative means the left method has lower loss.", "", "| contrast | role | mean | 95% paired-t CI | direction |", "|---|---|---:|---:|---:|"])
    for row in aggregate:
        lines.append(f"| {row['contrast']} | {row['role']} | {float(row['paired_mean']):+.6f} | [{float(row['paired_t_ci95_low']):+.6f}, {float(row['paired_t_ci95_high']):+.6f}] | {row['left_better_count']}/3 left-better |")
    lines.extend(["", "With n=3, confidence intervals are descriptive; no large-sample significance language is warranted.", ""])
    (output / "EXPERIMENT_45_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "analysis_manifest.json").write_text(
        json.dumps({"status": "completed_valid", "protocol": "mousse_r1_unified_analysis_v1", "identity_certificate": "identity_reuse_certificate.json", "source_files": identity["source_files"], "outputs": ["r1_unified_eight_method_run_summary.csv", "r1_unified_eight_method_aggregate.csv", "r1_mousse_paired_seed_deltas.csv", "r1_mousse_paired_aggregate.csv", "EXPERIMENT_45_ANALYSIS.md"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
