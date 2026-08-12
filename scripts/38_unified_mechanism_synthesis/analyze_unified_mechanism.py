#!/usr/bin/env python3
"""Build the audited MECH-01--09R unified mechanism synthesis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "2026-07-29.8"
BOOTSTRAP_SEED = 20260729
BOOTSTRAP_SAMPLES = 5000
T_CRITICAL_95_DF2 = 4.302652729696142

PRIMARY_CONTRASTS = (
    "selective_diag_vs_muon",
    "selective_none_vs_muon",
    "selective_diag_vs_original_newton_muon",
    "selective_none_vs_original_newton_muon",
)
BASELINE_CONTRAST = "original_newton_muon_vs_muon"
ALLOWED_CONTRASTS = (*PRIMARY_CONTRASTS, BASELINE_CONTRAST)
FOUNDATIONAL_MODE_ORDER = (
    "diag",
    "none",
    "block4",
    "dense_full",
    "muon",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
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
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    for row in rows:
        if set(row) != set(fieldnames):
            raise RuntimeError(f"inconsistent CSV schema for {path}: {set(row)}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def nested_get(value: Any, dotted: str) -> Any:
    current = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted)
        current = current[part]
    return current


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def fmt(value: Any, digits: int = 6) -> str:
    if value in ("", None):
        return "—"
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def audit_sources(
    input_root: Path, registry: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Path]], dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    resolved: dict[str, dict[str, Path]] = {}
    problems: list[str] = []
    bundle_summary: dict[str, Any] = {}

    for bundle in registry["bundles"]:
        bundle_id = bundle["id"]
        if bundle_id in resolved:
            raise RuntimeError(f"duplicate bundle id: {bundle_id}")
        resolved[bundle_id] = {}
        bundle_passed = True
        for item in bundle["files"]:
            relative = item["path"]
            path = input_root / Path(relative)
            key = Path(relative).name
            if key in resolved[bundle_id]:
                key = relative
            resolved[bundle_id][key] = path
            exists = path.is_file()
            row_count: Any = ""
            columns_or_keys = ""
            acceptance_passed: Any = ""
            error = ""
            digest = ""
            size = ""
            try:
                if not exists:
                    raise FileNotFoundError(path)
                digest = sha256_file(path)
                size = path.stat().st_size
                if item["kind"] == "json":
                    value = read_json(path)
                    columns_or_keys = ",".join(sorted(value)) if isinstance(value, dict) else ""
                    acceptance = item.get("acceptance")
                    if acceptance:
                        observed = nested_get(value, acceptance["field"])
                        acceptance_passed = observed == acceptance["equals"]
                        if not acceptance_passed:
                            raise RuntimeError(
                                f"acceptance failed: {acceptance['field']}="
                                f"{observed!r}, expected {acceptance['equals']!r}"
                            )
                elif item["kind"] == "csv":
                    rows = read_csv(path)
                    if not rows:
                        raise RuntimeError("empty CSV")
                    row_count = len(rows)
                    columns = list(rows[0])
                    columns_or_keys = ",".join(columns)
                    missing = sorted(set(item.get("required_columns", ())) - set(columns))
                    if missing:
                        raise RuntimeError(f"missing columns: {missing}")
                else:
                    raise RuntimeError(f"unsupported kind: {item['kind']}")
            except Exception as exc:  # precise details are preserved in the audit
                bundle_passed = False
                error = repr(exc)
                problems.append(f"{bundle_id}:{relative}:{error}")
            audit_rows.append(
                {
                    "bundle_id": bundle_id,
                    "evidence_level": bundle["evidence_level"],
                    "relative_path": relative,
                    "kind": item["kind"],
                    "exists": exists,
                    "size_bytes": size,
                    "sha256": digest,
                    "row_count": row_count,
                    "columns_or_keys": columns_or_keys,
                    "acceptance_passed": acceptance_passed,
                    "error": error,
                }
            )
        bundle_summary[bundle_id] = {
            "passed": bundle_passed,
            "evidence_level": bundle["evidence_level"],
            "files": len(bundle["files"]),
        }
    if problems:
        raise RuntimeError("source audit failed:\n" + "\n".join(problems))
    return audit_rows, resolved, bundle_summary


def find_path(resolved: dict[str, dict[str, Path]], bundle: str, name: str) -> Path:
    matches = [
        path
        for key, path in resolved[bundle].items()
        if key == name or path.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{bundle}: expected one {name}, found {matches}")
    return matches[0]


def validate_contrast_set(
    rows: list[dict[str, str]], family_field: str | None = None
) -> None:
    if any(
        "diag" in row.get("contrast", "")
        and "none" in row.get("contrast", "")
        and row.get("contrast", "") not in ALLOWED_CONTRASTS
        for row in rows
    ):
        raise RuntimeError("diag-vs-none contaminated the primary contract")
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group = row[family_field] if family_field else "all"
        groups[group].add(row["contrast"])
    expected = set(ALLOWED_CONTRASTS)
    for group, observed in groups.items():
        if observed != expected:
            raise RuntimeError(
                f"contrast coverage mismatch for {group}: "
                f"missing={sorted(expected-observed)} extra={sorted(observed-expected)}"
            )


def load_primary_training(
    resolved: dict[str, dict[str, Path]]
) -> list[dict[str, Any]]:
    rows = read_csv(
        find_path(resolved, "formal_primary_training", "primary_contrasts_summary.csv")
    )
    validate_contrast_set(rows, "family")
    output: list[dict[str, Any]] = []
    for row in rows:
        expected_priority = "baseline" if row["contrast"] == BASELINE_CONTRAST else "primary"
        if row["priority"] != expected_priority:
            raise RuntimeError(f"priority mismatch: {row}")
        output.append(
            {
                "family": row["family"],
                "family_label": row["family_label"],
                "priority": row["priority"],
                "contrast": row["contrast"],
                "left_role": row["left_role"],
                "right_role": row["right_role"],
                "seeds": int(row["seeds"]),
                "final_delta_mean": float(row["final_delta_mean"]),
                "final_delta_ci95_low": float(row["final_delta_ci95_low"]),
                "final_delta_ci95_high": float(row["final_delta_ci95_high"]),
                "tail5_delta_mean": float(row["tail5_delta_mean"]),
                "auc_delta_mean": float(row["auc_delta_mean"]),
                "negative_seeds_left_better": int(row["negative_seeds_left_better"]),
                "positive_seeds_left_worse": int(row["positive_seeds_left_worse"]),
                "classification": row["classification"],
                "negative_delta_means": "left algorithm is better",
            }
        )
    return output


def cluster_bootstrap(
    rows: list[dict[str, str]], samples: int, seed: int
) -> dict[str, Any]:
    by_origin: dict[str, list[float]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["checkpoint_cell"], row["data_replica"])
        if key in seen:
            raise RuntimeError(f"duplicate MECH-08 paired unit: {key}")
        seen.add(key)
        by_origin[row["checkpoint_cell"]].append(float(row["delta_left_minus_right"]))
    origins = sorted(by_origin)
    if len(origins) < 2:
        raise RuntimeError(f"cluster bootstrap needs >=2 origins, got {origins}")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        values: list[float] = []
        for _ in origins:
            origin = rng.choice(origins)
            origin_values = by_origin[origin]
            values.extend(rng.choice(origin_values) for _ in origin_values)
        draws.append(statistics.mean(values))
    flat = [value for values in by_origin.values() for value in values]
    origin_means = [statistics.mean(by_origin[origin]) for origin in origins]
    ci_low = percentile(draws, 0.025)
    ci_high = percentile(draws, 0.975)
    if ci_high < 0:
        classification = "left_better"
    elif ci_low > 0:
        classification = "left_worse"
    else:
        classification = "uncertain"
    return {
        "clusters": len(origins),
        "paired_units": len(flat),
        "mean_delta": statistics.mean(flat),
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "origin_means_left_better": sum(value < 0 for value in origin_means),
        "origin_means_left_worse": sum(value > 0 for value in origin_means),
        "classification": classification,
    }


def build_rollout_bootstrap(
    resolved: dict[str, dict[str, Path]], samples: int
) -> list[dict[str, Any]]:
    rows = read_csv(find_path(resolved, "mech08_rollout", "paired_contrasts.csv"))
    validate_contrast_set(rows)
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["checkpoint_stage"],
            row["contrast"],
            row["metric"],
            row["optimizer_step"],
        )
        grouped[key].append(row)
        grouped[("all", row["contrast"], row["metric"], row["optimizer_step"])].append(
            row
        )
    output: list[dict[str, Any]] = []
    contrast_order = {name: index for index, name in enumerate(ALLOWED_CONTRASTS)}
    stage_order = {"early": 0, "late": 1, "all": 2}
    for key in sorted(
        grouped,
        key=lambda item: (
            stage_order[item[0]],
            contrast_order[item[1]],
            item[2],
            str(item[3]),
        ),
    ):
        stage, contrast, metric, optimizer_step = key
        result = cluster_bootstrap(
            grouped[key],
            samples=samples,
            seed=BOOTSTRAP_SEED
            + stage_order[stage] * 1000
            + contrast_order[contrast] * 100
            + sum(ord(char) for char in f"{metric}:{optimizer_step}"),
        )
        output.append(
            {
                "checkpoint_stage": stage,
                "priority": (
                    "baseline" if contrast == BASELINE_CONTRAST else "primary"
                ),
                "contrast": contrast,
                "metric": metric,
                "optimizer_step": optimizer_step,
                **result,
                "bootstrap_samples": samples,
                "bootstrap_seed_base": BOOTSTRAP_SEED,
                "negative_delta_means": "left algorithm is better",
            }
        )
    return output


def build_prediction_alignment(
    resolved: dict[str, dict[str, Path]]
) -> list[dict[str, Any]]:
    rows = read_csv(find_path(resolved, "mech08_rollout", "prediction_alignment.csv"))
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "contrast_scope": row["contrast_scope"],
                "metric": row["metric"],
                "optimizer_step": row["optimizer_step"],
                "origin_contrast_units": int(row["origin_contrast_units"]),
                "pearson": float(row["pearson"]),
                "spearman": float(row["spearman"]),
                "sign_concordance": float(row["sign_concordance"]),
                "evidence_level": "descriptive",
                "confirmatory": False,
            }
        )
    return output


def build_alpha_synthesis(
    resolved: dict[str, dict[str, Path]]
) -> list[dict[str, Any]]:
    specs = (
        ("block", "r1_block_alpha", "final_acceptance.json"),
        ("dense_full", "r1_dense_full_alpha", "final_audit_manifest.json"),
    )
    output: list[dict[str, Any]] = []
    for topology, bundle, acceptance_name in specs:
        rows = read_csv(find_path(resolved, bundle, "seed_curvature.csv"))
        seeds = [int(row["seed"]) for row in rows]
        if sorted(seeds) != [2024, 2025, 2026]:
            raise RuntimeError(f"{topology}: expected seeds 2024-2026, got {seeds}")
        curvatures = [float(row["final_curvature_c"]) for row in rows]
        midpoint_all = all(
            as_bool(row["alpha0p50_beats_both_endpoints_final"]) for row in rows
        )
        accepted = read_json(find_path(resolved, bundle, acceptance_name))
        classification = accepted.get(
            "scientific_classification", accepted.get("classification", "")
        )
        output.append(
            {
                "topology": topology,
                "seeds": len(seeds),
                "seed_list": ",".join(map(str, sorted(seeds))),
                "mean_final_curvature_c": statistics.mean(curvatures),
                "min_final_curvature_c": min(curvatures),
                "max_final_curvature_c": max(curvatures),
                "all_seed_curvatures_negative": all(value < 0 for value in curvatures),
                "alpha0p50_beats_both_endpoints_all_seeds": midpoint_all,
                "scientific_classification": classification,
                "claim": "nonmonotonic_dose_response_on_tested_grid",
                "caveat": "does_not_establish_a_universal_optimal_alpha",
            }
        )
    return output


def require_three_seed_rows(
    rows: list[dict[str, str]], label: str
) -> dict[int, dict[str, str]]:
    by_seed: dict[int, dict[str, str]] = {}
    for row in rows:
        seed = int(row["seed"])
        if seed in by_seed:
            raise RuntimeError(f"{label}: duplicate seed {seed}")
        by_seed[seed] = row
    if sorted(by_seed) != [2024, 2025, 2026]:
        raise RuntimeError(
            f"{label}: expected seeds 2024-2026, got {sorted(by_seed)}"
        )
    return by_seed


def build_foundational_module_structure(
    resolved: dict[str, dict[str, Path]]
) -> list[dict[str, Any]]:
    owt_rows = read_csv(
        find_path(
            resolved,
            "owt_foundational_module_allocation",
            "combined_all_seeds_comparison.csv",
        )
    )
    wiki_rows = read_csv(
        find_path(
            resolved,
            "wikitext_foundational_module_allocation",
            "wikitext_dual_alpha_run_summary.csv",
        )
    )
    wiki_core = read_csv(
        find_path(
            resolved,
            "wikitext_foundational_module_allocation",
            "run_summary.csv",
        )
    )

    raw_by_dataset: dict[str, dict[str, list[dict[str, str]]]] = {
        "OWT": defaultdict(list),
        "WikiText-103": defaultdict(list),
    }
    owt_mode_map = {
        "diag": "diag",
        "none": "none",
        "block4": "block4",
        "full": "dense_full",
        "muon": "muon",
    }
    for row in owt_rows:
        if row["mode"] in owt_mode_map:
            prepared = dict(row)
            prepared["_source_cohort"] = "owt_combined_24l"
            raw_by_dataset["OWT"][owt_mode_map[row["mode"]]].append(prepared)

    wiki_mode_map = {
        "diag": "diag",
        "none": "none",
        "block4": "block4",
        "full": "dense_full",
    }
    for row in wiki_rows:
        if row["mode"] in wiki_mode_map:
            if not as_bool(row["quality_eligible"]) or not as_bool(
                row["memory_eligible"]
            ):
                raise RuntimeError(f"WikiText structure row is ineligible: {row}")
            prepared = dict(row)
            prepared["_source_cohort"] = "wikitext_dual_alpha_24l"
            raw_by_dataset["WikiText-103"][wiki_mode_map[row["mode"]]].append(
                prepared
            )
    for row in wiki_core:
        if row["method"] != "muon" or not math.isclose(
            float(row["learning_rate"]), 0.01, rel_tol=0.0, abs_tol=1e-12
        ):
            continue
        prepared = dict(row)
        prepared["cproj_k_state_mib"] = "0"
        prepared["non_cproj_k_state_mib"] = "0"
        prepared["_source_cohort"] = "wikitext_matched_recipe_muon_reference"
        raw_by_dataset["WikiText-103"]["muon"].append(prepared)

    role_by_mode = {
        "diag": "selective_diagonal_cproj",
        "none": "selective_remove_cproj",
        "block4": "block4_structured_control",
        "dense_full": "dense_mechanism_control",
        "muon": "muon_baseline",
    }
    output: list[dict[str, Any]] = []
    mode_order = {mode: index for index, mode in enumerate(FOUNDATIONAL_MODE_ORDER)}
    for dataset in ("OWT", "WikiText-103"):
        by_mode = raw_by_dataset[dataset]
        if set(by_mode) != set(FOUNDATIONAL_MODE_ORDER):
            raise RuntimeError(
                f"{dataset}: foundational mode coverage mismatch: {sorted(by_mode)}"
            )
        indexed = {
            mode: require_three_seed_rows(rows, f"{dataset}:{mode}")
            for mode, rows in by_mode.items()
        }
        means = {
            mode: statistics.mean(
                float(row["final_val_loss"]) for row in seed_rows.values()
            )
            for mode, seed_rows in indexed.items()
        }
        ranks = {
            mode: rank + 1
            for rank, mode in enumerate(
                sorted(means, key=lambda name: (means[name], mode_order[name]))
            )
        }
        dense_k = statistics.mean(
            float(row["k_state_mib"])
            for row in indexed["dense_full"].values()
        )
        block_rows = indexed["block4"]
        block_cohort = {
            row["_source_cohort"] for row in block_rows.values()
        }
        for mode in FOUNDATIONAL_MODE_ORDER:
            seed_rows = indexed[mode]
            losses = [
                float(seed_rows[seed]["final_val_loss"])
                for seed in sorted(seed_rows)
            ]
            cohorts = {row["_source_cohort"] for row in seed_rows.values()}
            paired_deltas: list[float] = []
            if cohorts == block_cohort:
                paired_deltas = [
                    float(seed_rows[seed]["final_val_loss"])
                    - float(block_rows[seed]["final_val_loss"])
                    for seed in sorted(seed_rows)
                ]
            k_state = [
                float(seed_rows[seed]["k_state_mib"]) for seed in sorted(seed_rows)
            ]
            output.append(
                {
                    "dataset": dataset,
                    "architecture": "24L_GPT",
                    "mode": mode,
                    "comparison_role": role_by_mode[mode],
                    "source_cohort": ";".join(sorted(cohorts)),
                    "seeds": len(seed_rows),
                    "seed_list": ",".join(map(str, sorted(seed_rows))),
                    "mean_final_val_loss": statistics.mean(losses),
                    "sample_sd_final_val_loss": statistics.stdev(losses),
                    "final_loss_rank": ranks[mode],
                    "paired_delta_vs_block4_mean": (
                        statistics.mean(paired_deltas) if paired_deltas else ""
                    ),
                    "paired_seeds_better_than_block4": (
                        sum(value < 0 for value in paired_deltas)
                        if paired_deltas
                        else ""
                    ),
                    "paired_seeds_worse_than_block4": (
                        sum(value > 0 for value in paired_deltas)
                        if paired_deltas
                        else ""
                    ),
                    "mean_peak_memory_mib": statistics.mean(
                        float(row["peak_memory_mib"]) for row in seed_rows.values()
                    ),
                    "mean_k_state_mib": statistics.mean(k_state),
                    "mean_cproj_k_state_mib": statistics.mean(
                        float(row["cproj_k_state_mib"])
                        for row in seed_rows.values()
                    ),
                    "mean_non_cproj_k_state_mib": statistics.mean(
                        float(row["non_cproj_k_state_mib"])
                        for row in seed_rows.values()
                    ),
                    "k_state_released_vs_dense_fraction": 1.0
                    - statistics.mean(k_state) / dense_k,
                    "paper_timing_eligible": False,
                    "evidence_level": "supportive",
                    "caveat": (
                        "matched-recipe historical reference; not the dual-alpha cohort"
                        if dataset == "WikiText-103" and mode == "muon"
                        else "same-cohort three-seed structural comparison"
                    ),
                }
            )
    return output


def build_complementary_bridge(
    resolved: dict[str, dict[str, Path]]
) -> list[dict[str, Any]]:
    specs = (
        (
            "OWT",
            "owt_foundational_module_allocation",
            "muon_learning_rate",
            "selective_all_cproj",
        ),
        (
            "WikiText-103",
            "wikitext_foundational_module_allocation",
            "learning_rate",
            "selective_all",
        ),
    )
    output: list[dict[str, Any]] = []
    for dataset, bundle, lr_field, release_method in specs:
        bridge_rows = read_csv(find_path(resolved, bundle, "bridge_run_summary.csv"))
        core_rows = read_csv(find_path(resolved, bundle, "run_summary.csv"))
        bridge = require_three_seed_rows(
            [row for row in bridge_rows if row["method"] == "non_cproj_all"],
            f"{dataset}:cproj_only",
        )
        none_rows = require_three_seed_rows(
            [
                row
                for row in core_rows
                if row["method"] == release_method
                and math.isclose(
                    float(row[lr_field]), 0.01, rel_tol=0.0, abs_tol=1e-12
                )
            ],
            f"{dataset}:none",
        )
        deltas = [
            float(bridge[seed]["final_val_loss"])
            - float(none_rows[seed]["final_val_loss"])
            for seed in sorted(bridge)
        ]
        mean_delta = statistics.mean(deltas)
        half_width = T_CRITICAL_95_DF2 * statistics.stdev(deltas) / math.sqrt(3)
        output.append(
            {
                "dataset": dataset,
                "architecture": "24L_GPT",
                "contrast": "cproj_only_minus_none",
                "seeds": len(deltas),
                "seed_list": ",".join(map(str, sorted(bridge))),
                "cproj_only_mean_final_val_loss": statistics.mean(
                    float(row["final_val_loss"]) for row in bridge.values()
                ),
                "none_mean_final_val_loss": statistics.mean(
                    float(row["final_val_loss"]) for row in none_rows.values()
                ),
                "paired_delta_mean": mean_delta,
                "paired_delta_sample_sd": statistics.stdev(deltas),
                "paired_delta_ci95_low_t_df2": mean_delta - half_width,
                "paired_delta_ci95_high_t_df2": mean_delta + half_width,
                "cproj_only_worse_seeds": sum(value > 0 for value in deltas),
                "cproj_only_better_seeds": sum(value < 0 for value in deltas),
                "cproj_only_mean_k_state_mib": statistics.mean(
                    float(row["k_state_mib"]) for row in bridge.values()
                ),
                "none_mean_k_state_mib": statistics.mean(
                    float(row["k_state_mib"]) for row in none_rows.values()
                ),
                "cproj_only_mean_peak_memory_mib": statistics.mean(
                    float(row["peak_memory_mib"]) for row in bridge.values()
                ),
                "none_mean_peak_memory_mib": statistics.mean(
                    float(row["peak_memory_mib"]) for row in none_rows.values()
                ),
                "classification": (
                    "cproj_only_worse_all_seeds"
                    if all(value > 0 for value in deltas)
                    else "mixed"
                ),
                "evidence_level": "supportive",
                "interpretation": (
                    "useful historical K contribution resides outside c_proj; "
                    "c_proj-only curvature is insufficient"
                ),
            }
        )
    return output


def load_json_bundle(
    resolved: dict[str, dict[str, Path]], bundle: str, name: str
) -> Any:
    return read_json(find_path(resolved, bundle, name))


def build_architecture_transfer_boundary(
    resolved: dict[str, dict[str, Path]],
) -> list[dict[str, Any]]:
    bundle = "llama_block_partition_invariance"
    classification = load_json_bundle(resolved, bundle, "classification.json")
    checkpoints = read_csv(find_path(resolved, bundle, "checkpoint_summary.csv"))
    if classification["classification"] != "strong_non_invariance":
        raise RuntimeError("LLaMA block-partition classification changed")
    if {row["checkpoint_label"] for row in checkpoints} != {"early", "late"}:
        raise RuntimeError("LLaMA block-partition checkpoint coverage changed")
    embedded = {
        row["checkpoint_label"]: row
        for row in classification["checkpoint_summaries"]
    }
    for row in checkpoints:
        label = row["checkpoint_label"]
        for field in (
            "checkpoint_step",
            "global_block4_update_drift_median",
            "global_block4_update_drift_p95",
        ):
            observed = float(row[field])
            expected = float(embedded[label][field])
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
                raise RuntimeError(
                    f"LLaMA block-partition summary mismatch: {label}/{field}"
                )
    interpretation = classification["interpretation"]
    if (
        interpretation["block4_is_original_newton_muon"]
        or interpretation["block4_is_primary_baseline"]
        or interpretation["official_original_newton_muon_control"] != "newton_full"
    ):
        raise RuntimeError("LLaMA block-partition interpretation boundary changed")
    by_label = {row["checkpoint_label"]: row for row in checkpoints}
    decision = classification["decision_statistics"]
    return [
        {
            "architecture": "LLaMA-1B",
            "candidate": "contiguous_block4",
            "classification": classification["classification"],
            "pooled_global_block4_median_update_drift": float(
                decision["pooled_global_block4_median_update_drift"]
            ),
            "maximum_equivariant_control_drift": float(
                decision["maximum_equivariant_control_drift"]
            ),
            "effect_to_control_multiple": float(
                decision["effect_to_control_multiple"]
            ),
            "early_checkpoint_step": int(by_label["early"]["checkpoint_step"]),
            "early_update_drift_median": float(
                by_label["early"]["global_block4_update_drift_median"]
            ),
            "early_update_drift_p95": float(
                by_label["early"]["global_block4_update_drift_p95"]
            ),
            "late_checkpoint_step": int(by_label["late"]["checkpoint_step"]),
            "late_update_drift_median": float(
                by_label["late"]["global_block4_update_drift_median"]
            ),
            "late_update_drift_p95": float(
                by_label["late"]["global_block4_update_drift_p95"]
            ),
            "block4_is_original_newton_muon": False,
            "block4_is_primary_baseline": False,
            "official_original_newton_muon_control": "newton_full",
            "evidence_level": "limiting",
            "interpretation": interpretation["claim_if_supported"],
            "claim_not_authorized": interpretation["claim_not_authorized"],
        }
    ]


def build_mechanism_chain(
    resolved: dict[str, dict[str, Path]],
    rollout_rows: list[dict[str, Any]],
    alpha_rows: list[dict[str, Any]],
    foundational_rows: list[dict[str, Any]],
    bridge_rows: list[dict[str, Any]],
    architecture_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    geometry = load_json_bundle(resolved, "mech02_geometry", "geometry_gate.json")
    prediction = load_json_bundle(
        resolved, "mech03_shadow_prediction", "prediction_gate.json"
    )
    trajectory = load_json_bundle(
        resolved, "mech06_llama1b_diagnostic", "trajectory_prediction.json"
    )
    mediation = load_json_bundle(
        resolved, "mech09r_refresh_mediation", "mediation_decision.json"
    )
    main_rollouts = [
        row
        for row in rollout_rows
        if row["checkpoint_stage"] == "all"
        and row["contrast"] in PRIMARY_CONTRASTS
        and (
            (row["metric"] == "normalized_loss_auc")
            or (
                row["metric"] == "normalized_heldout_loss"
                and str(row["optimizer_step"]) == "128"
            )
        )
    ]
    rollout_counts = {
        label: sum(row["classification"] == label for row in main_rollouts)
        for label in ("left_better", "left_worse", "uncertain")
    }
    top_two_by_dataset = {
        dataset: [
            row["mode"]
            for row in sorted(
                (row for row in foundational_rows if row["dataset"] == dataset),
                key=lambda row: int(row["final_loss_rank"]),
            )[:2]
        ]
        for dataset in ("OWT", "WikiText-103")
    }
    architecture = architecture_rows[0]
    return [
        {
            "stage_order": 1,
            "stage": "foundational_module_allocation",
            "source_bundle": (
                "owt_foundational_module_allocation;"
                "wikitext_foundational_module_allocation"
            ),
            "evidence_level": "supportive",
            "status": "replicated_across_two_datasets",
            "result": (
                f"top-two modes: OWT={','.join(top_two_by_dataset['OWT'])}; "
                f"WikiText-103={','.join(top_two_by_dataset['WikiText-103'])}; "
                f"cproj-only worse in "
                f"{sum(row['cproj_only_worse_seeds'] for row in bridge_rows)}/"
                f"{sum(row['seeds'] for row in bridge_rows)} paired seeds"
            ),
            "claim_boundary": (
                "historical 24L GPT evidence; R1 module allocation remains unisolated"
            ),
        },
        {
            "stage_order": 2,
            "stage": "architecture_transfer_boundary",
            "source_bundle": "llama_block_partition_invariance",
            "evidence_level": "limiting",
            "status": architecture["classification"],
            "result": (
                "LLaMA contiguous block4 median update drift="
                f"{architecture['pooled_global_block4_median_update_drift']:.4f}; "
                "equivariant-control max="
                f"{architecture['maximum_equivariant_control_drift']:.4f}; "
                "effect/control="
                f"{architecture['effect_to_control_multiple']:.2f}x"
            ),
            "claim_boundary": (
                "coordinate-partition dependence, not a full-training performance ranking; "
                "official LLaMA original control remains newton_full"
            ),
        },
        {
            "stage_order": 3,
            "stage": "numerical_implementation",
            "source_bundle": "mech01_numerical",
            "evidence_level": "supportive",
            "status": "passed",
            "result": "fixed-tensor and cross-runtime numerical checks passed",
            "claim_boundary": "implementation validity, not optimizer superiority",
        },
        {
            "stage_order": 4,
            "stage": "k_geometry",
            "source_bundle": "mech02_geometry",
            "evidence_level": "descriptive",
            "status": (
                "candidate_signal"
                if geometry["geometry_gate_candidate_passed"]
                else "no_signal"
            ),
            "result": (
                f"{sum(item['passed'] for item in geometry['metric_gates'])}/"
                f"{len(geometry['metric_gates'])} geometry metric gates passed"
            ),
            "claim_boundary": "geometry did not authorize the next mechanism stage",
        },
        {
            "stage_order": 5,
            "stage": "one_step_crossfit_prediction",
            "source_bundle": "mech03_shadow_prediction",
            "evidence_level": "limiting",
            "status": "failed_prediction_gate",
            "result": (
                f"{prediction['positive_material_layers']}/"
                f"{prediction['layers_total']} material layers; "
                f"{prediction['positive_paired_cells']}/"
                f"{prediction['paired_cells_total']} positive paired cells"
            ),
            "claim_boundary": "one-step shadow loss is not a validated long-horizon proxy",
        },
        {
            "stage_order": 6,
            "stage": "llama1b_one_step_confirmation",
            "source_bundle": "mech06_llama1b_diagnostic",
            "evidence_level": "limiting",
            "status": "uncertain",
            "result": (
                f"early={trajectory['early_signal']}; late={trajectory['late_signal']}"
            ),
            "claim_boundary": "no retrospective ranking was used to rescue the proxy",
        },
        {
            "stage_order": 7,
            "stage": "frozen_checkpoint_family_shadow",
            "source_bundle": "mech07_family_shadow",
            "evidence_level": "supportive",
            "status": "stage_dependent",
            "result": "early and late checkpoint counterfactual rankings differ",
            "claim_boundary": "counterfactual shadow steps are not real training trajectories",
        },
        {
            "stage_order": 8,
            "stage": "real_128_step_rollout",
            "source_bundle": "mech08_rollout",
            "evidence_level": "supportive",
            "status": "mixed_and_stage_dependent",
            "result": (
                f"all-stage primary AUC/step128 rows: "
                f"{rollout_counts['left_better']} left-better, "
                f"{rollout_counts['left_worse']} left-worse, "
                f"{rollout_counts['uncertain']} uncertain"
            ),
            "claim_boundary": "128 steps do not replace full-budget training",
        },
        {
            "stage_order": 9,
            "stage": "down_projection_refresh_mediation",
            "source_bundle": "mech09r_refresh_mediation",
            "evidence_level": "confirmatory",
            "status": mediation["classification"],
            "result": (
                f"{mediation['directional_predictions_passed']}/3 frozen "
                "directional predictions passed"
            ),
            "claim_boundary": "causal within the frozen MECH-09R intervention tree",
        },
        {
            "stage_order": 10,
            "stage": "alpha_dose_response",
            "source_bundle": "r1_block_alpha;r1_dense_full_alpha",
            "evidence_level": "confirmatory",
            "status": "strong_confirmatory_support",
            "result": (
                f"{sum(row['all_seed_curvatures_negative'] for row in alpha_rows)}/"
                f"{len(alpha_rows)} topologies have negative curvature in all seeds"
            ),
            "claim_boundary": "tested-grid nonmonotonicity, not universal alpha optimality",
        },
        {
            "stage_order": 11,
            "stage": "formal_training_outcomes",
            "source_bundle": "formal_primary_training",
            "evidence_level": "confirmatory",
            "status": "authoritative_for_method_performance",
            "result": "three architectures × three seeds × five frozen contrasts",
            "claim_boundary": "mechanism evidence explains but does not replace this result",
        },
    ]


def build_claim_matrix(
    chain: list[dict[str, Any]],
    primary_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    alpha_rows: list[dict[str, Any]],
    foundational_rows: list[dict[str, Any]],
    bridge_rows: list[dict[str, Any]],
    architecture_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    primary_only = [row for row in primary_rows if row["priority"] == "primary"]
    material_better = sum(
        row["classification"] == "selective_or_left_materially_better"
        for row in primary_only
    )
    material_worse = sum(
        row["classification"] == "selective_or_left_materially_worse"
        for row in primary_only
    )
    within_margin = sum(
        row["classification"] == "within_practical_margin" for row in primary_only
    )
    auc_alignment = next(
        row
        for row in alignment_rows
        if row["contrast_scope"] == "primary"
        and row["metric"] == "normalized_loss_auc"
    )
    step128_alignment = next(
        row
        for row in alignment_rows
        if row["contrast_scope"] == "primary"
        and row["metric"] == "normalized_heldout_loss"
        and str(row["optimizer_step"]) == "128"
    )
    none_rows = [
        row for row in foundational_rows if row["mode"] == "none"
    ]
    diagonal_rank_first = sum(
        row["mode"] == "diag" and int(row["final_loss_rank"]) == 1
        for row in foundational_rows
    )
    none_rank_second = sum(
        int(row["final_loss_rank"]) == 2 for row in none_rows
    )
    bridge_worse = sum(row["cproj_only_worse_seeds"] for row in bridge_rows)
    bridge_pairs = sum(row["seeds"] for row in bridge_rows)
    geometry_result = next(
        row["result"] for row in chain if row["stage"] == "k_geometry"
    )
    architecture = architecture_rows[0]
    return [
        {
            "claim_id": "C01",
            "claim": "The numerical K/update diagnostic path is reproducible across the audited execution domains.",
            "claim_type": "validation",
            "evidence_level": "supportive",
            "source_bundles": "mech01_numerical",
            "status": "supported",
            "quantitative_evidence": "5/5 registered manifests passed",
            "caveat": "does not compare optimizer quality",
        },
        {
            "claim_id": "C02",
            "claim": "K geometry contains cross-architecture structure, but geometry alone is insufficient to predict the preferred optimizer.",
            "claim_type": "descriptive",
            "evidence_level": "descriptive",
            "source_bundles": "mech02_geometry;mech03_shadow_prediction",
            "status": "supported_with_limit",
            "quantitative_evidence": geometry_result,
            "caveat": "MECH-03 prediction gate failed",
        },
        {
            "claim_id": "C03",
            "claim": "One-step shadow loss is not a reliable stand-alone predictor of the 128-step trajectory.",
            "claim_type": "predictive_validation",
            "evidence_level": "limiting",
            "source_bundles": "mech03_shadow_prediction;mech06_llama1b_diagnostic;mech08_rollout",
            "status": "supported",
            "quantitative_evidence": (
                f"primary sign concordance: AUC={auc_alignment['sign_concordance']:.3f}, "
                f"step128={step128_alignment['sign_concordance']:.3f}"
            ),
            "caveat": "alignment is descriptive and uses only 16 origin-contrast units",
        },
        {
            "claim_id": "C04",
            "claim": "Short-horizon relative behavior is stage dependent rather than a fixed optimizer ranking.",
            "claim_type": "trajectory",
            "evidence_level": "supportive",
            "source_bundles": "mech07_family_shadow;mech08_rollout",
            "status": "supported",
            "quantitative_evidence": "early/late shadow and rollout contrasts change sign or certainty",
            "caveat": "128-step rollouts are not full training runs",
        },
        {
            "claim_id": "C05",
            "claim": "The scheduled down-projection refresh is a causal mediator of the post-refresh short-horizon degradation in MECH-09R.",
            "claim_type": "causal",
            "evidence_level": "confirmatory",
            "source_bundles": "mech09r_refresh_mediation",
            "status": "supported",
            "quantitative_evidence": "3/3 frozen directional predictions passed with exact shared prefixes",
            "caveat": "scope is the frozen 1B intervention tree and 128-step horizon",
        },
        {
            "claim_id": "C06",
            "claim": "Curvature mixing has a reproducible nonmonotonic dose response on the tested R1 alpha grid.",
            "claim_type": "confirmatory_response_curve",
            "evidence_level": "confirmatory",
            "source_bundles": "r1_block_alpha;r1_dense_full_alpha",
            "status": "supported",
            "quantitative_evidence": (
                f"{sum(row['alpha0p50_beats_both_endpoints_all_seeds'] for row in alpha_rows)}"
                f"/{len(alpha_rows)} topologies: alpha=0.5 beats both endpoints in all seeds"
            ),
            "caveat": "best alpha was descriptive; no universal alpha=0.5 claim",
        },
        {
            "claim_id": "C07",
            "claim": "Selective methods must be judged separately against Muon and original Newton–Muon.",
            "claim_type": "method_comparison",
            "evidence_level": "confirmatory",
            "source_bundles": "formal_primary_training",
            "status": "supported",
            "quantitative_evidence": (
                f"12 primary contrasts: {material_better} materially better, "
                f"{material_worse} materially worse, {within_margin} within margin"
            ),
            "caveat": "diag-versus-none is intentionally not a primary contrast",
        },
        {
            "claim_id": "C08",
            "claim": (
                "Historical 24-layer GPT evidence localizes the useful K-state "
                "allocation away from dense c_proj curvature."
            ),
            "claim_type": "module_allocation",
            "evidence_level": "supportive",
            "source_bundles": (
                "owt_foundational_module_allocation;"
                "wikitext_foundational_module_allocation"
            ),
            "status": "replicated_with_architecture_boundary",
            "quantitative_evidence": (
                f"diagonal ranked first in {diagonal_rank_first}/2 datasets; "
                f"none ranked second in {none_rank_second}/2; "
                f"cproj-only was worse in {bridge_worse}/{bridge_pairs} paired seeds; "
                f"none removed "
                f"{statistics.mean(row['k_state_released_vs_dense_fraction'] for row in none_rows):.2%} "
                "of dense K state"
            ),
            "caveat": (
                "supportive OWT/WikiText-103 24L evidence; R1 does not reproduce "
                "the exact none-versus-block4 ordering and needs a module factorial"
            ),
        },
        {
            "claim_id": "C09",
            "claim": (
                "A contiguous block4 approximation is coordinate-partition dependent "
                "on LLaMA and is not an architecture-neutral original Newton–Muon baseline."
            ),
            "claim_type": "architecture_transfer_boundary",
            "evidence_level": "limiting",
            "source_bundles": "llama_block_partition_invariance",
            "status": "supported",
            "quantitative_evidence": (
                "pooled block4 median update drift="
                f"{architecture['pooled_global_block4_median_update_drift']:.4f}; "
                "equivariant-control max="
                f"{architecture['maximum_equivariant_control_drift']:.4f}; "
                "effect/control="
                f"{architecture['effect_to_control_multiple']:.2f}x"
            ),
            "caveat": (
                "does not authorize a full-training performance ordering; the official "
                "LLaMA original Newton–Muon control remains newton_full"
            ),
        },
    ]


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def build_report(
    primary_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    alpha_rows: list[dict[str, Any]],
    foundational_rows: list[dict[str, Any]],
    bridge_rows: list[dict[str, Any]],
    architecture_rows: list[dict[str, Any]],
    chain: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    bootstrap_samples: int,
) -> str:
    primary_main = [row for row in primary_rows if row["priority"] == "primary"]
    material_better = sum(
        row["classification"] == "selective_or_left_materially_better"
        for row in primary_main
    )
    material_worse = sum(
        row["classification"] == "selective_or_left_materially_worse"
        for row in primary_main
    )
    within_margin = sum(
        row["classification"] == "within_practical_margin" for row in primary_main
    )
    main_rollout = [
        row
        for row in rollout_rows
        if row["checkpoint_stage"] == "all"
        and (
            row["metric"] == "normalized_loss_auc"
            or (
                row["metric"] == "normalized_heldout_loss"
                and str(row["optimizer_step"]) == "128"
            )
        )
    ]
    primary_alignment = [
        row
        for row in alignment_rows
        if row["contrast_scope"] == "primary"
        and (
            row["metric"] == "normalized_loss_auc"
            or str(row["optimizer_step"]) == "128"
        )
    ]
    chain_table = markdown_table(
        ["Stage", "Evidence", "Status", "Result", "Boundary"],
        (
            (
                row["stage"],
                row["evidence_level"],
                row["status"],
                row["result"],
                row["claim_boundary"],
            )
            for row in chain
        ),
    )
    training_table = markdown_table(
        ["Family", "Priority", "Contrast", "Final Δ", "95% CI", "Classification"],
        (
            (
                row["family_label"],
                row["priority"],
                row["contrast"],
                fmt(row["final_delta_mean"]),
                f"[{fmt(row['final_delta_ci95_low'])}, {fmt(row['final_delta_ci95_high'])}]",
                row["classification"],
            )
            for row in primary_rows
        ),
    )
    rollout_table = markdown_table(
        ["Stage", "Priority", "Contrast", "Metric", "Mean Δ", "Cluster 95% CI", "Result"],
        (
            (
                row["checkpoint_stage"],
                row["priority"],
                row["contrast"],
                (
                    "AUC"
                    if row["metric"] == "normalized_loss_auc"
                    else f"step {row['optimizer_step']}"
                ),
                fmt(row["mean_delta"]),
                f"[{fmt(row['bootstrap_ci95_low'])}, {fmt(row['bootstrap_ci95_high'])}]",
                row["classification"],
            )
            for row in main_rollout
        ),
    )
    alpha_table = markdown_table(
        ["Topology", "Seeds", "Mean curvature C", "All C<0", "α=.5 beats endpoints", "Class"],
        (
            (
                row["topology"],
                row["seeds"],
                fmt(row["mean_final_curvature_c"]),
                row["all_seed_curvatures_negative"],
                row["alpha0p50_beats_both_endpoints_all_seeds"],
                row["scientific_classification"],
            )
            for row in alpha_rows
        ),
    )
    foundational_table = markdown_table(
        [
            "Dataset",
            "Mode",
            "Final loss",
            "Rank",
            "Delta vs block4",
            "K state MiB",
            "c_proj K MiB",
            "Peak MiB",
        ],
        (
            (
                row["dataset"],
                row["mode"],
                fmt(row["mean_final_val_loss"]),
                row["final_loss_rank"],
                fmt(row["paired_delta_vs_block4_mean"]),
                fmt(row["mean_k_state_mib"]),
                fmt(row["mean_cproj_k_state_mib"]),
                fmt(row["mean_peak_memory_mib"]),
            )
            for row in foundational_rows
        ),
    )
    bridge_table = markdown_table(
        [
            "Dataset",
            "c_proj-only loss",
            "none loss",
            "Paired delta",
            "95% t CI",
            "Worse seeds",
        ],
        (
            (
                row["dataset"],
                fmt(row["cproj_only_mean_final_val_loss"]),
                fmt(row["none_mean_final_val_loss"]),
                fmt(row["paired_delta_mean"]),
                (
                    f"[{fmt(row['paired_delta_ci95_low_t_df2'])}, "
                    f"{fmt(row['paired_delta_ci95_high_t_df2'])}]"
                ),
                f"{row['cproj_only_worse_seeds']}/{row['seeds']}",
            )
            for row in bridge_rows
        ),
    )
    architecture_table = markdown_table(
        [
            "Architecture",
            "Candidate",
            "Classification",
            "Pooled median drift",
            "Control max",
            "Effect/control",
            "Original control",
        ],
        (
            (
                row["architecture"],
                row["candidate"],
                row["classification"],
                fmt(row["pooled_global_block4_median_update_drift"]),
                fmt(row["maximum_equivariant_control_drift"]),
                f"{row['effect_to_control_multiple']:.2f}x",
                row["official_original_newton_muon_control"],
            )
            for row in architecture_rows
        ),
    )
    alignment_table = markdown_table(
        ["Metric", "Step", "Pearson", "Spearman", "Sign concordance", "Units"],
        (
            (
                row["metric"],
                row["optimizer_step"],
                fmt(row["pearson"], 4),
                fmt(row["spearman"], 4),
                fmt(row["sign_concordance"], 4),
                row["origin_contrast_units"],
            )
            for row in primary_alignment
        ),
    )
    claim_table = markdown_table(
        ["ID", "Claim type", "Level", "Status", "Evidence", "Caveat"],
        (
            (
                row["claim_id"],
                row["claim_type"],
                row["evidence_level"],
                row["status"],
                row["quantitative_evidence"],
                row["caveat"],
            )
            for row in claims
        ),
    )
    return f"""# Selective Newton–Muon Unified Mechanism Synthesis

## Technical summary

The evidence supports a **Newton–Muon-family mechanism with stage-dependent
curvature refresh effects**, not a primary “diag versus none” story. In formal
three-seed training, the 12 frozen Selective contrasts contain
{material_better} materially better, {material_worse} materially worse, and
{within_margin} within-margin outcomes; therefore each Selective method must be
reported separately against both Muon and original Newton–Muon.

The strongest causal result is MECH-09R: all three frozen directional predictions
passed under exact shared prefixes, identifying the scheduled down-projection
refresh as a mediator of post-refresh short-horizon degradation. The R1 alpha
experiments independently show a nonmonotonic response on both block and dense-full
topologies, with alpha=0.5 beating both endpoints for every tested seed. This does
not establish a universal best alpha.

The negative result is equally important: MECH-03 failed its prediction gate,
MECH-06 remained uncertain, and MECH-08 prediction/trajectory sign concordance is
only descriptive. One-step shadow loss must not be presented as a validated
long-horizon selector.

The previously omitted 24-layer OWT and WikiText-103 studies now form the
foundational module-allocation layer. Diagonal c_proj retention ranks first and
none ranks second on both datasets. The complementary bridge is worse than
none in all six paired seeds, showing that c_proj-only K state is not the
useful part of the historical Newton contribution.

## The evidence chain narrows from numerical validity to a local causal mediator

{chain_table}

The chain deliberately separates implementation validation, descriptive geometry,
counterfactual shadow evidence, real rollout evidence, causal intervention, and
full-budget training. A later stage does not retroactively turn an earlier
descriptive diagnostic into a confirmatory predictor.

## Foundational OWT/WikiText module allocation is now part of the evidence chain

These historical three-seed studies vary the c_proj K structure while retaining
the same 24-layer GPT training family. The WikiText Muon row is a matched-recipe
historical reference rather than a row from the dual-alpha launch; consequently
its block4 paired delta is intentionally left blank.

{foundational_table}

`none` removes c_proj K while retaining non-c_proj K. `dense_full` is a dense
mechanism control, not the official block4 contraction. Timing from these sources
is not reused for paper claims; memory and K-state accounting remain explicit.

The complementary bridge reverses the allocation: it retains c_proj K but removes
non-c_proj K. Positive deltas below mean that the c_proj-only bridge is worse.

{bridge_table}

This isolates the historical result more sharply than a loss ranking alone:
removing the expensive c_proj state preserved the useful contribution, whereas
retaining only that state did not. The evidence is supportive and does not assert
that every architecture must share the same module allocation.

## LLaMA exposes a strict block4 transfer boundary

Experiment 40 is limiting evidence, not a training-performance comparison. Under
global hidden-coordinate permutations, the contiguous block4 LLaMA update changes
far more than the equivariant controls:

{architecture_table}

Therefore block4 is not treated as original Newton–Muon or as a primary LLaMA
baseline. The official LLaMA original Newton–Muon control remains `newton_full`.
This result does not authorize a loss ranking among full-training optimizers.

## Formal training requires both baselines for every Selective proposal

All deltas below are left minus right validation loss; negative values favor the
left algorithm. Each row aggregates seeds 2024, 2025, and 2026 using the already
accepted primary analysis.

{training_table}

This is the authoritative performance layer. The fact that a Selective method can
beat Muon while matching or losing to original Newton–Muon is precisely why the
two baselines cannot be collapsed.

## Real 128-step trajectories are mixed rather than a universal ranking

The table uses the all-stage MECH-08 comparisons at normalized AUC and step 128.
Intervals come from {bootstrap_samples} hierarchical bootstrap draws with
checkpoint origin as the outer cluster and replicas resampled within origin.

{rollout_table}

Short-horizon results diagnose when relative behavior changes; they do not replace
the 6200-step formal result. MECH-08 timing is excluded from all efficiency claims.

## One-step predictions do not reliably bridge to the rollout

{alignment_table}

The AUC and step-128 correlations/sign agreement are descriptive and based on 16
primary origin-contrast units. Together with the failed MECH-03 gate and uncertain
MECH-06 result, they rule out using one-step shadow loss as a stand-alone selection
criterion.

## Alpha confirms a nonmonotonic dose response, not a universal optimum

Curvature is defined as
`L(alpha=0.5) - 0.5 * [L(alpha=0) + L(alpha=1)]` at validation step 6200.
Negative values mean the midpoint beats the linear interpolation of endpoints.

{alpha_table}

The block and dense-full confirmations support the same response-curve claim.
Topology itself showed no material threshold effect in the accepted dense audit,
and timing from these concurrent runs remains ineligible for paper efficiency
claims.

## Scope, definitions, and statistical design

- Primary comparison set: four Selective-versus-baseline contrasts; original
  Newton–Muon versus Muon is a baseline contrast.
- Formal-training unit: seed within architecture.
- MECH-08 paired unit: checkpoint origin × data replica; checkpoint origin is the
  bootstrap cluster.
- MECH-09R causal unit: a branch from an exact shared prefix under the frozen
  intervention tree.
- Evidence levels: confirmatory, supportive, descriptive, and limiting as frozen
  in `UNIFIED_MECHANISM_CONTRACT.md`.

No checkpoint, raw W&B API, or unregistered training log was read by this
synthesis.

## Claim audit keeps negative and limiting evidence visible

{claim_table}

## Limitations and robustness boundaries

- MECH-09R establishes a local refresh mediator in the frozen LLaMA-1B design; it
  does not prove that every scale or architecture has the same mediator.
- MECH-08 contains only four checkpoint origins and a 128-step horizon, so its
  clustered intervals are intentionally conservative and should not be treated as
  a replacement for multi-seed formal training.
- The alpha result covers the tested five-point grid. Selecting the best point
  from that grid is descriptive.
- The OWT/WikiText module-allocation studies use a 24-layer GPT family. Their
  replicated direction motivates, but cannot substitute for, an R1 module
  factorial. The WikiText Muon row is a matched-recipe reference from a separate
  accepted launch.
- Experiment 40 establishes coordinate-partition dependence of contiguous block4
  on LLaMA. It is not evidence that block4 underperforms or outperforms another
  optimizer in full training.
- Existing mechanism and concurrent alpha timing is not paper-ready efficiency
  evidence.

## Recommended next step

Keep the 39 submission-efficiency and sensitivity audit as the currently running
R1 job. After it finishes, run the frozen 41 R1 module 2x2 factorial. The new
factorial adds only the missing cproj-only and all-K-off cells and reuses the
accepted block4 and none cells. Any further mechanism experiment is
conditional on the interaction estimate and seed consistency from that result.

## Further questions

- Does the down-projection refresh mediator reproduce outside the frozen LLaMA-1B
  branch design?
- Which existing throughput and peak-memory records meet exclusive-GPU,
  same-domain, same-shape, warmup, synchronization, and repetition requirements?
- Does a shared learning-rate multiplier grid preserve the formal ranking under an
  equal tuning budget?
- Does the R1 c_fc-by-c_proj factorial reproduce the historical allocation result,
  or reveal an architecture-specific interaction?
"""


def output_generated_at(output_dir: Path) -> str:
    stamp = output_dir.name
    if len(stamp) == 24 and stamp.endswith("+0000") and stamp[8] == "T":
        return (
            f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}T"
            f"{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}+00:00"
        )
    return "2026-07-29T00:00:00+00:00"


def build_artifact(
    output_dir: Path,
    registry: dict[str, Any],
    primary_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    alpha_rows: list[dict[str, Any]],
    foundational_rows: list[dict[str, Any]],
    bridge_rows: list[dict[str, Any]],
    architecture_rows: list[dict[str, Any]],
    chain: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    generated_at = output_generated_at(output_dir)
    main_rollout = [
        row
        for row in rollout_rows
        if row["checkpoint_stage"] == "all"
        and (
            row["metric"] == "normalized_loss_auc"
            or (
                row["metric"] == "normalized_heldout_loss"
                and str(row["optimizer_step"]) == "128"
            )
        )
    ]
    primary_alignment = [
        row
        for row in alignment_rows
        if row["contrast_scope"] == "primary"
        and (
            row["metric"] == "normalized_loss_auc"
            or str(row["optimizer_step"]) == "128"
        )
    ]
    primary_main = [row for row in primary_rows if row["priority"] == "primary"]
    class_counts = {
        label: sum(row["classification"] == label for row in primary_main)
        for label in (
            "selective_or_left_materially_better",
            "selective_or_left_materially_worse",
            "within_practical_margin",
        )
    }
    source_specs: list[dict[str, Any]] = []
    for bundle in registry["bundles"]:
        paths = [item["path"] for item in bundle["files"]]
        source_specs.append(
            {
                "id": bundle["id"],
                "label": bundle["id"].replace("_", " "),
                "path": paths[0],
                "query": {
                    "language": "python",
                    "description": (
                        "Read-only synthesis of the registered accepted files; "
                        f"bundle contains {len(paths)} file(s)."
                    ),
                    "tables_used": paths,
                    "filters": [
                        "Only paths frozen in source_registry.json",
                        "No checkpoints, online W&B API, or unregistered raw logs",
                    ],
                },
            }
        )
    sources_by_id = {source["id"]: source for source in source_specs}
    sources_by_id["formal_primary_training"]["query"].update(
        {
            "engine": "duckdb",
            "language": "sql",
            "sql": (
                "SELECT family, family_label, priority, contrast, left_role, "
                "right_role, seeds, final_delta_mean, final_delta_ci95_low, "
                "final_delta_ci95_high, tail5_delta_mean, auc_delta_mean, "
                "classification "
                "FROM read_csv_auto("
                "'34_selective_primary_comparison/20260727T083000+0000/"
                "primary_contrasts_summary.csv') "
                "ORDER BY family, priority DESC, contrast"
            ),
            "metric_definitions": [
                "final_delta_mean = mean final validation loss of left role minus right role across seeds 2024–2026",
                "negative final_delta_mean favors the left role",
            ],
        }
    )
    sources_by_id["alpha_joint"] = {
        "id": "alpha_joint",
        "label": "R1 block and dense-full alpha confirmations",
        "path": "22_r1_block_alpha/analysis/wandb_20260729_multiseed_confirmation/seed_curvature.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": "Joint read-only synthesis of the accepted block and dense-full alpha confirmations.",
            "sql": (
                "SELECT topology, seeds, seed_list, mean_final_curvature_c, "
                "min_final_curvature_c, max_final_curvature_c, "
                "all_seed_curvatures_negative, "
                "alpha0p50_beats_both_endpoints_all_seeds, "
                "scientific_classification "
                "FROM read_csv_auto('alpha_synthesis.csv') ORDER BY topology"
            ),
            "tables_used": [
                "alpha_synthesis.csv",
                "22_r1_block_alpha/analysis/wandb_20260729_multiseed_confirmation/seed_curvature.csv",
                "24_r1_dense_full_alpha/analysis/wandb_20260729_multiseed_confirmation/seed_curvature.csv",
            ],
            "metric_definitions": [
                "curvature C = L(alpha=0.5) - 0.5 * (L(alpha=0) + L(alpha=1)) at validation step 6200"
            ],
        },
    }
    sources_by_id["mechanism_joint"] = {
        "id": "mechanism_joint",
        "label": "Registered MECH-01–09R evidence chain",
        "path": "38_unified_mechanism_synthesis/source_registry.json",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": (
                f"Ordered evidence-level synthesis across all "
                f"{len(registry['bundles'])} registered source bundles."
            ),
            "sql": (
                "SELECT stage_order, stage, evidence_level, status, result, "
                "claim_boundary FROM read_csv_auto('mechanism_chain.csv') "
                "ORDER BY stage_order"
            ),
            "tables_used": ["mechanism_chain.csv"]
            + [
                item["path"]
                for bundle in registry["bundles"]
                for item in bundle["files"]
            ],
        },
    }
    sources_by_id["mech08_rollout"].update(
        {
            "path": "rollout_cluster_bootstrap.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "All-stage MECH-08 normalized AUC and step-128 clustered contrasts.",
                "sql": (
                    "SELECT * FROM read_csv_auto('rollout_cluster_bootstrap.csv') "
                    "WHERE checkpoint_stage = 'all' AND "
                    "(metric = 'normalized_loss_auc' OR optimizer_step = '128') "
                    "ORDER BY priority DESC, contrast, metric"
                ),
                "tables_used": [
                    "rollout_cluster_bootstrap.csv",
                    "36_mech08_short_horizon_rollout/20260727T102506+0000/analysis/paired_contrasts.csv",
                ],
                "metric_definitions": [
                    "mean_delta is left-minus-right normalized loss or normalized AUC",
                    "95% intervals use checkpoint origin as outer bootstrap cluster and replicas within origin",
                ],
            },
        }
    )
    sources_by_id["prediction_alignment_output"] = {
        "id": "prediction_alignment_output",
        "label": "MECH-08 prediction-to-rollout alignment",
        "path": "prediction_rollout_alignment.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": "Primary AUC and step-128 prediction alignment rows.",
            "sql": (
                "SELECT * FROM read_csv_auto('prediction_rollout_alignment.csv') "
                "WHERE contrast_scope = 'primary' AND "
                "(metric = 'normalized_loss_auc' OR optimizer_step = '128') "
                "ORDER BY metric, optimizer_step"
            ),
            "tables_used": [
                "prediction_rollout_alignment.csv",
                "36_mech08_short_horizon_rollout/20260727T102506+0000/analysis/prediction_alignment.csv",
            ],
        },
    }
    sources_by_id["claim_joint"] = {
        "id": "claim_joint",
        "label": "Unified mechanism claim/evidence matrix",
        "path": "claim_evidence_matrix.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": "Paper-facing claims with evidence level and required boundary.",
            "sql": (
                "SELECT claim_id, claim_type, evidence_level, status, "
                "quantitative_evidence, caveat "
                "FROM read_csv_auto('claim_evidence_matrix.csv') ORDER BY claim_id"
            ),
            "tables_used": [
                "claim_evidence_matrix.csv",
                "mechanism_chain.csv",
                "primary_training_contrasts.csv",
                "rollout_cluster_bootstrap.csv",
                "alpha_synthesis.csv",
            ],
        },
    }
    sources_by_id["foundational_module_joint"] = {
        "id": "foundational_module_joint",
        "label": "OWT and WikiText-103 foundational module allocation",
        "path": "foundational_module_structure.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": (
                "Three-seed 24-layer GPT c_proj structure and complementary "
                "module-allocation synthesis."
            ),
            "sql": (
                "SELECT dataset, mode, mean_final_val_loss, final_loss_rank, "
                "paired_delta_vs_block4_mean, mean_k_state_mib, "
                "mean_cproj_k_state_mib, mean_peak_memory_mib "
                "FROM read_csv_auto('foundational_module_structure.csv') "
                "ORDER BY dataset, final_loss_rank"
            ),
            "tables_used": [
                "foundational_module_structure.csv",
                "complementary_bridge_summary.csv",
            ],
            "metric_definitions": [
                "none removes c_proj K while retaining non-c_proj K",
                "cproj-only bridge retains c_proj K while removing non-c_proj K",
                "loss deltas are left minus right and negative favors the left role",
            ],
        },
    }
    sources_by_id["architecture_transfer_output"] = {
        "id": "architecture_transfer_output",
        "label": "LLaMA block4 architecture-transfer boundary",
        "path": "architecture_transfer_boundary.csv",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": (
                "Read-only synthesis of the accepted LLaMA coordinate-permutation audit."
            ),
            "sql": (
                "SELECT architecture, candidate, classification, "
                "pooled_global_block4_median_update_drift, "
                "maximum_equivariant_control_drift, effect_to_control_multiple, "
                "official_original_newton_muon_control "
                "FROM read_csv_auto('architecture_transfer_boundary.csv')"
            ),
            "tables_used": [
                "architecture_transfer_boundary.csv",
                "40_llama_block_partition_invariance_audit/20260729T044926+0000/"
                "analysis/classification.json",
                "40_llama_block_partition_invariance_audit/20260729T044926+0000/"
                "analysis/checkpoint_summary.csv",
            ],
        },
    }
    sources = list(sources_by_id.values())
    title = "Selective Newton–Muon Unified Mechanism Synthesis"
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}", "layout": "full"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Technical summary\n\n"
                "**The evidence supports a Newton–Muon-family mechanism with "
                "stage-dependent curvature refresh effects, not a primary diag-versus-none "
                "story.** Across the 12 frozen Selective formal-training contrasts, "
                f"{class_counts['selective_or_left_materially_better']} are materially "
                f"better, {class_counts['selective_or_left_materially_worse']} materially "
                f"worse, and {class_counts['within_practical_margin']} within the practical "
                "margin. MECH-09R supplies the strongest local causal evidence, while both "
                "R1 alpha topologies confirm a nonmonotonic tested-grid response. The "
                "previously omitted OWT/WikiText module studies now show the same top-two "
                "structural modes on both datasets and reject the complementary c_proj-only "
                "allocation in all six paired seeds.\n\n"
                "**The LLaMA transfer audit is a limiting boundary.** Contiguous "
                "block4 is strongly coordinate-partition dependent and therefore is "
                "not treated as original Newton–Muon on LLaMA; the official control "
                "remains newton_full.\n\n"
                "**One-step diagnostics are a limiting result.** MECH-03 failed its "
                "prediction gate, MECH-06 remained uncertain, and the descriptive "
                "MECH-08 prediction bridge does not justify a long-horizon selector."
            ),
        },
        {
            "id": "chain_intro",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## The chain narrows from implementation validity to a local causal mediator\n\n"
                "Each stage below has a distinct evidentiary job. Later results do not "
                "retroactively convert descriptive geometry or a failed proxy into "
                "confirmatory prediction."
            ),
        },
        {"id": "chain_table_block", "type": "table", "tableId": "chain_table"},
        {
            "id": "foundational_intro",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## OWT and WikiText-103 identify a historical module-allocation signal\n\n"
                "Diagonal c_proj retention ranks first and none ranks second on "
                "both 24-layer GPT datasets. The complementary c_proj-only bridge is "
                "worse than none in all six seed pairs. This is supportive evidence "
                "for allocating K state outside c_proj, with an explicit R1 boundary."
            ),
        },
        {
            "id": "foundational_table_block",
            "type": "table",
            "tableId": "foundational_table",
        },
        {"id": "bridge_table_block", "type": "table", "tableId": "bridge_table"},
        {
            "id": "architecture_intro",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## LLaMA exposes a strict block4 transfer boundary\n\n"
                "Experiment 40 finds strong non-invariance under hidden-coordinate "
                "permutations. This limits baseline transfer; it is not a full-training "
                "loss comparison."
            ),
        },
        {
            "id": "architecture_table_block",
            "type": "table",
            "tableId": "architecture_table",
        },
        {
            "id": "training_intro",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Formal training requires both baselines for every Selective proposal\n\n"
                "Loss deltas are left minus right, so negative values favor the left "
                "algorithm. Each row aggregates the accepted seeds 2024–2026. These "
                "full-budget results are authoritative for method performance."
            ),
        },
        {"id": "training_chart_block", "type": "chart", "chartId": "training_chart"},
        {"id": "training_table_block", "type": "table", "tableId": "training_table"},
        {
            "id": "rollout_intro",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Real 128-step trajectories are mixed rather than a universal ranking\n\n"
                f"The intervals use {bootstrap_samples} hierarchical bootstrap draws with "
                "checkpoint origin as the outer cluster and replicas resampled within "
                "origin. MECH-08 timing is excluded from efficiency claims."
            ),
        },
        {"id": "rollout_table_block", "type": "table", "tableId": "rollout_table"},
        {
            "id": "prediction_intro",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## One-step predictions do not reliably bridge to the rollout\n\n"
                "The alignment statistics are descriptive and cover only 16 primary "
                "origin-contrast units. They support retaining the failed/uncertain proxy "
                "results as a real limitation."
            ),
        },
        {"id": "alignment_table_block", "type": "table", "tableId": "alignment_table"},
        {
            "id": "alpha_intro",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Alpha confirms a nonmonotonic tested-grid response, not a universal optimum\n\n"
                "Negative curvature means alpha=0.5 beats the linear interpolation of "
                "the two endpoints. Both accepted topologies agree, but choosing the best "
                "grid point remains descriptive."
            ),
        },
        {"id": "alpha_table_block", "type": "table", "tableId": "alpha_table"},
        {
            "id": "scope",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Scope, definitions, and statistical design\n\n"
                "- Primary set: four Selective-versus-baseline contrasts; original "
                "Newton–Muon versus Muon is the family baseline.\n"
                "- Formal-training unit: seed within architecture.\n"
                "- MECH-08 unit: checkpoint origin × replica, clustered by origin.\n"
                "- MECH-09R claim: causal only within the frozen LLaMA-1B intervention tree.\n"
                "- Input boundary: registered accepted files only; no checkpoints or online API."
            ),
        },
        {
            "id": "claim_intro",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## The claim audit keeps negative evidence visible\n\n"
                "Every paper-facing mechanism statement is paired with its evidence "
                "level, quantitative support, and a boundary that prevents overclaiming."
            ),
        },
        {"id": "claim_table_block", "type": "table", "tableId": "claim_table"},
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## The strongest evidence is local, and the rollout is short\n\n"
                "MECH-09R identifies a refresh mediator only in its frozen 1B design. "
                "MECH-08 has four checkpoint origins and a 128-step horizon. The alpha "
                "claim covers a five-point grid. The historical module evidence uses a "
                "24-layer GPT family and therefore motivates, but cannot replace, an R1 "
                "factorial. Experiment 40 limits contiguous block4 transfer to LLaMA "
                "without ranking full-training methods. All concurrent timing remains "
                "ineligible for paper efficiency claims."
            ),
        },
        {
            "id": "next_step",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Recommended next step\n\n"
                "Let the currently running 39 submission-efficiency and sensitivity audit "
                "finish, then run the frozen 41 R1 c_fc-by-c_proj 2x2 factorial. The "
                "factorial reuses accepted block4 and none cells and trains only the "
                "two missing module allocations."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Further questions\n\n"
                "- Does the down-projection refresh mediator reproduce outside the frozen "
                "LLaMA-1B branch design?\n"
                "- Which existing throughput and peak-memory records meet exclusive-GPU "
                "and same-domain requirements?\n"
                "- Does a shared learning-rate grid preserve the formal ranking under an "
                "equal tuning budget?\n"
                "- Does the R1 c_fc-by-c_proj interaction reproduce the historical "
                "24-layer GPT allocation result?"
            ),
        },
    ]
    charts = [
        {
            "id": "training_chart",
            "title": "Formal validation-loss contrasts",
            "subtitle": (
                "Three architectures × three seeds; left-minus-right final loss, "
                "negative favors the left algorithm"
            ),
            "type": "bar",
            "dataset": "formal_training",
            "sourceId": "formal_primary_training",
            "encodings": {
                "x": {
                    "field": "contrast",
                    "type": "nominal",
                    "label": "Frozen contrast",
                },
                "y": {
                    "field": "final_delta_mean",
                    "type": "quantitative",
                    "label": "Final validation-loss delta",
                    "format": "number",
                },
                "color": {
                    "field": "family_label",
                    "type": "nominal",
                    "label": "Architecture",
                },
                "tooltip": [
                    {"field": "family_label", "type": "nominal", "label": "Architecture"},
                    {"field": "priority", "type": "nominal", "label": "Priority"},
                    {
                        "field": "final_delta_ci95_low",
                        "type": "quantitative",
                        "label": "95% CI low",
                    },
                    {
                        "field": "final_delta_ci95_high",
                        "type": "quantitative",
                        "label": "95% CI high",
                    },
                    {
                        "field": "classification",
                        "type": "nominal",
                        "label": "Classification",
                    },
                ],
            },
            "xAxisTitle": "Frozen contrast",
            "yAxisTitle": "Final validation-loss delta",
            "valueFormat": "number",
            "layout": "full",
            "maxRows": 15,
        }
    ]
    tables = [
        {
            "id": "chain_table",
            "title": "Mechanism evidence chain",
            "subtitle": "Ordered from numerical validation to formal training authority",
            "dataset": "mechanism_chain",
            "sourceId": "mechanism_joint",
            "defaultSort": {"field": "stage_order", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "stage_order", "label": "#", "type": "number"},
                {"field": "stage", "label": "Stage", "type": "text"},
                {"field": "evidence_level", "label": "Evidence", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "result", "label": "Result", "type": "text"},
                {"field": "claim_boundary", "label": "Boundary", "type": "text"},
            ],
        },
        {
            "id": "training_table",
            "title": "Formal three-seed contrasts",
            "subtitle": "Three architectures; negative loss delta favors the left algorithm",
            "dataset": "formal_training",
            "sourceId": "formal_primary_training",
            "defaultSort": {"field": "family_label", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "family_label", "label": "Family", "type": "text"},
                {"field": "priority", "label": "Priority", "type": "text"},
                {"field": "contrast", "label": "Contrast", "type": "text"},
                {
                    "field": "final_delta_mean",
                    "label": "Final Δ",
                    "type": "number",
                    "movement": True,
                },
                {"field": "final_delta_ci95_low", "label": "CI low", "type": "number"},
                {"field": "final_delta_ci95_high", "label": "CI high", "type": "number"},
                {"field": "classification", "label": "Classification", "type": "text"},
            ],
        },
        {
            "id": "rollout_table",
            "title": "MECH-08 clustered rollout contrasts",
            "subtitle": "All-stage normalized AUC and step-128 loss, four checkpoint origins",
            "dataset": "rollout_main",
            "sourceId": "prediction_alignment_output",
            "defaultSort": {"field": "contrast", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "priority", "label": "Priority", "type": "text"},
                {"field": "contrast", "label": "Contrast", "type": "text"},
                {"field": "metric", "label": "Metric", "type": "text"},
                {"field": "optimizer_step", "label": "Step", "type": "text"},
                {
                    "field": "mean_delta",
                    "label": "Mean Δ",
                    "type": "number",
                    "movement": True,
                },
                {"field": "bootstrap_ci95_low", "label": "CI low", "type": "number"},
                {"field": "bootstrap_ci95_high", "label": "CI high", "type": "number"},
                {"field": "classification", "label": "Result", "type": "text"},
            ],
        },
        {
            "id": "alignment_table",
            "title": "Prediction-to-rollout alignment",
            "subtitle": "Primary contrasts only; AUC and step 128",
            "dataset": "prediction_alignment",
            "sourceId": "mech08_rollout",
            "defaultSort": {"field": "metric", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "metric", "label": "Metric", "type": "text"},
                {"field": "optimizer_step", "label": "Step", "type": "text"},
                {"field": "pearson", "label": "Pearson", "type": "number"},
                {"field": "spearman", "label": "Spearman", "type": "number"},
                {"field": "sign_concordance", "label": "Sign agreement", "type": "number"},
                {"field": "origin_contrast_units", "label": "Units", "type": "number"},
            ],
        },
        {
            "id": "alpha_table",
            "title": "R1 alpha response-curve confirmation",
            "subtitle": "Seeds 2024–2026 across block and dense-full topology",
            "dataset": "alpha_synthesis",
            "sourceId": "alpha_joint",
            "defaultSort": {"field": "topology", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "topology", "label": "Topology", "type": "text"},
                {"field": "seeds", "label": "Seeds", "type": "number"},
                {
                    "field": "mean_final_curvature_c",
                    "label": "Mean curvature C",
                    "type": "number",
                    "movement": True,
                },
                {"field": "all_seed_curvatures_negative", "label": "All C<0", "type": "text"},
                {
                    "field": "alpha0p50_beats_both_endpoints_all_seeds",
                    "label": "α=.5 beats endpoints",
                    "type": "text",
                },
                {"field": "scientific_classification", "label": "Class", "type": "text"},
            ],
        },
        {
            "id": "claim_table",
            "title": "Paper-facing mechanism claim audit",
            "subtitle": "Claim status, evidence level, and required boundary",
            "dataset": "claim_matrix",
            "sourceId": "claim_joint",
            "defaultSort": {"field": "claim_id", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "claim_id", "label": "ID", "type": "text"},
                {"field": "claim_type", "label": "Claim type", "type": "text"},
                {"field": "evidence_level", "label": "Evidence", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "quantitative_evidence", "label": "Evidence note", "type": "text"},
                {"field": "caveat", "label": "Boundary", "type": "text"},
            ],
        },
        {
            "id": "foundational_table",
            "title": "Historical 24-layer GPT module structure",
            "subtitle": (
                "Three seeds per mode; final validation loss rank and K-state footprint"
            ),
            "dataset": "foundational_module_structure",
            "sourceId": "foundational_module_joint",
            "defaultSort": {"field": "final_loss_rank", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "dataset", "label": "Dataset", "type": "text"},
                {"field": "mode", "label": "Mode", "type": "text"},
                {
                    "field": "mean_final_val_loss",
                    "label": "Final loss",
                    "type": "number",
                    "movement": True,
                },
                {"field": "final_loss_rank", "label": "Rank", "type": "number"},
                {
                    "field": "paired_delta_vs_block4_mean",
                    "label": "Delta vs block4",
                    "type": "number",
                    "movement": True,
                },
                {"field": "mean_k_state_mib", "label": "K MiB", "type": "number"},
                {
                    "field": "mean_cproj_k_state_mib",
                    "label": "c_proj K MiB",
                    "type": "number",
                },
                {
                    "field": "mean_peak_memory_mib",
                    "label": "Peak MiB",
                    "type": "number",
                },
            ],
        },
        {
            "id": "bridge_table",
            "title": "Complementary module bridge",
            "subtitle": (
                "c_proj-only minus none final loss; positive means c_proj-only is worse"
            ),
            "dataset": "complementary_bridge",
            "sourceId": "foundational_module_joint",
            "defaultSort": {"field": "dataset", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "dataset", "label": "Dataset", "type": "text"},
                {
                    "field": "cproj_only_mean_final_val_loss",
                    "label": "c_proj-only loss",
                    "type": "number",
                },
                {
                    "field": "none_mean_final_val_loss",
                    "label": "none loss",
                    "type": "number",
                },
                {
                    "field": "paired_delta_mean",
                    "label": "Paired delta",
                    "type": "number",
                    "movement": True,
                },
                {
                    "field": "paired_delta_ci95_low_t_df2",
                    "label": "CI low",
                    "type": "number",
                },
                {
                    "field": "paired_delta_ci95_high_t_df2",
                    "label": "CI high",
                    "type": "number",
                },
                {
                    "field": "cproj_only_worse_seeds",
                    "label": "Worse seeds",
                    "type": "number",
                },
            ],
        },
        {
            "id": "architecture_table",
            "title": "LLaMA block4 architecture-transfer boundary",
            "subtitle": (
                "Coordinate-permutation update drift; not a full-training loss ranking"
            ),
            "dataset": "architecture_transfer_boundary",
            "sourceId": "architecture_transfer_output",
            "defaultSort": {"field": "architecture", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "architecture", "label": "Architecture", "type": "text"},
                {"field": "candidate", "label": "Candidate", "type": "text"},
                {"field": "classification", "label": "Classification", "type": "text"},
                {
                    "field": "pooled_global_block4_median_update_drift",
                    "label": "Median drift",
                    "type": "number",
                },
                {
                    "field": "maximum_equivariant_control_drift",
                    "label": "Control max",
                    "type": "number",
                },
                {
                    "field": "effect_to_control_multiple",
                    "label": "Effect/control",
                    "type": "number",
                },
                {
                    "field": "official_original_newton_muon_control",
                    "label": "Original control",
                    "type": "text",
                },
            ],
        },
    ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "Audited foundational OWT/WikiText module evidence, MECH-01–09R, "
            "LLaMA block4 transfer boundary, formal-training, and alpha synthesis."
        ),
        "generatedAt": generated_at,
        "blocks": blocks,
        "charts": charts,
        "tables": tables,
        "sources": sources,
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "mechanism_chain": chain,
            "formal_training": primary_rows,
            "rollout_main": main_rollout,
            "prediction_alignment": primary_alignment,
            "alpha_synthesis": alpha_rows,
            "foundational_module_structure": foundational_rows,
            "complementary_bridge": bridge_rows,
            "architecture_transfer_boundary": architecture_rows,
            "claim_matrix": claims,
        },
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise RuntimeError("--bootstrap-samples must be at least 100")
    registry = read_json(args.registry)
    if registry.get("schema_version") != 1:
        raise RuntimeError("unsupported source registry schema")

    audit_rows, resolved, bundle_summary = audit_sources(args.input_root, registry)
    primary_rows = load_primary_training(resolved)
    rollout_rows = build_rollout_bootstrap(resolved, args.bootstrap_samples)
    alignment_rows = build_prediction_alignment(resolved)
    alpha_rows = build_alpha_synthesis(resolved)
    foundational_rows = build_foundational_module_structure(resolved)
    bridge_rows = build_complementary_bridge(resolved)
    architecture_rows = build_architecture_transfer_boundary(resolved)
    chain = build_mechanism_chain(
        resolved,
        rollout_rows,
        alpha_rows,
        foundational_rows,
        bridge_rows,
        architecture_rows,
    )
    claims = build_claim_matrix(
        chain,
        primary_rows,
        alignment_rows,
        alpha_rows,
        foundational_rows,
        bridge_rows,
        architecture_rows,
    )

    if any(
        "diag" in row["claim"].lower()
        and "none" in row["claim"].lower()
        and "not a primary contrast" not in row["caveat"].lower()
        for row in claims
    ):
        raise RuntimeError("diag-vs-none entered the claim matrix")
    if [row["contrast"] for row in primary_rows[:5]] != list(ALLOWED_CONTRASTS):
        raise RuntimeError("primary comparison order changed")
    if not all(row["status"] != "" for row in claims):
        raise RuntimeError("claim matrix has empty status")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "source_audit.csv", audit_rows)
    write_json(
        args.output_dir / "source_audit.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "passed": all(item["passed"] for item in bundle_summary.values()),
            "bundles": bundle_summary,
            "registry_sha256": sha256_file(args.registry),
            "files": len(audit_rows),
        },
    )
    write_csv(args.output_dir / "primary_training_contrasts.csv", primary_rows)
    write_csv(args.output_dir / "rollout_cluster_bootstrap.csv", rollout_rows)
    write_csv(args.output_dir / "prediction_rollout_alignment.csv", alignment_rows)
    write_csv(args.output_dir / "alpha_synthesis.csv", alpha_rows)
    write_csv(
        args.output_dir / "foundational_module_structure.csv",
        foundational_rows,
    )
    write_csv(
        args.output_dir / "complementary_bridge_summary.csv",
        bridge_rows,
    )
    write_csv(
        args.output_dir / "architecture_transfer_boundary.csv",
        architecture_rows,
    )
    write_csv(args.output_dir / "mechanism_chain.csv", chain)
    write_csv(args.output_dir / "claim_evidence_matrix.csv", claims)
    report_name = "UNIFIED_MECHANISM_SYNTHESIS_REPORT.md"
    (args.output_dir / report_name).write_text(
        build_report(
            primary_rows,
            rollout_rows,
            alignment_rows,
            alpha_rows,
            foundational_rows,
            bridge_rows,
            architecture_rows,
            chain,
            claims,
            args.bootstrap_samples,
        ),
        encoding="utf-8",
    )
    artifact = build_artifact(
        args.output_dir,
        registry,
        primary_rows,
        rollout_rows,
        alignment_rows,
        alpha_rows,
        foundational_rows,
        bridge_rows,
        architecture_rows,
        chain,
        claims,
        args.bootstrap_samples,
    )
    write_json(args.output_dir / "artifact.json", artifact)
    write_json(
        args.output_dir / "report_source_notes.json",
        {
            "schema_version": 1,
            "audience": "technical",
            "delivery_mode": "portable_html",
            "required_structure_mapping": {
                "title": "title",
                "technical_summary": "technical_summary",
                "key_findings_with_evidence": [
                    "chain_intro",
                    "foundational_intro",
                    "architecture_intro",
                    "training_intro",
                    "rollout_intro",
                    "prediction_intro",
                    "alpha_intro",
                ],
                "scope_data_metrics": "scope",
                "methodology": ["scope", "claim_intro"],
                "limitations_robustness": "limitations",
                "recommended_next_steps": "next_step",
                "further_questions": "questions",
            },
            "chart_map": [
                {
                    "section": "training_intro",
                    "analytical_question": (
                        "How do the five frozen contrasts differ across architectures?"
                    ),
                    "family": "comparison",
                    "type": "grouped bar",
                    "dataset": "formal_training",
                    "fields": [
                        "contrast",
                        "final_delta_mean",
                        "family_label",
                        "final_delta_ci95_low",
                        "final_delta_ci95_high",
                    ],
                    "supported_claim": (
                        "Selective methods require both baselines and are architecture dependent."
                    ),
                    "palette_policy": "relaxed multi-category; architecture is the second categorical dimension",
                    "delivery": "artifact.json native chart training_chart",
                }
            ],
            "table_map": [
                {
                    "section": "foundational_intro",
                    "analytical_question": (
                        "Which c_proj K allocation preserves loss quality and memory "
                        "efficiency across OWT and WikiText-103?"
                    ),
                    "datasets": [
                        "foundational_module_structure",
                        "complementary_bridge",
                    ],
                    "supported_claim": (
                        "Historical useful K contribution is concentrated outside "
                        "dense c_proj curvature, with an explicit R1 boundary."
                    ),
                },
                {
                    "section": "architecture_intro",
                    "analytical_question": (
                        "Can contiguous block4 be transferred to LLaMA as an "
                        "architecture-neutral original Newton–Muon baseline?"
                    ),
                    "datasets": ["architecture_transfer_boundary"],
                    "supported_claim": (
                        "No: block4 is strongly coordinate-partition dependent on LLaMA; "
                        "newton_full remains the original control."
                    ),
                },
            ],
            "omitted_chart_reason": (
                "No cross-stage mechanism chart is created because stages use heterogeneous "
                "units and evidence levels. A normalized mechanism score would be misleading."
            ),
            "source_registry_sha256": sha256_file(args.registry),
        },
    )

    artifact_names = [
        "source_audit.csv",
        "source_audit.json",
        "primary_training_contrasts.csv",
        "rollout_cluster_bootstrap.csv",
        "prediction_rollout_alignment.csv",
        "alpha_synthesis.csv",
        "foundational_module_structure.csv",
        "complementary_bridge_summary.csv",
        "architecture_transfer_boundary.csv",
        "mechanism_chain.csv",
        "claim_evidence_matrix.csv",
        report_name,
        "artifact.json",
        "report_source_notes.json",
    ]
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "passed": True,
        "analysis_kind": "cpu_only_read_only_unified_mechanism_synthesis",
        "input_registry_sha256": sha256_file(args.registry),
        "contract_sha256": sha256_file(
            Path(__file__).with_name("UNIFIED_MECHANISM_CONTRACT.md")
        ),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "samples": args.bootstrap_samples,
            "outer_unit": "checkpoint_cell",
            "inner_unit": "data_replica",
        },
        "primary_contrasts": list(PRIMARY_CONTRASTS),
        "baseline_contrast": BASELINE_CONTRAST,
        "diag_vs_none_primary": False,
        "timing_usable_for_paper": False,
        "source_bundles": len(bundle_summary),
        "source_files": len(audit_rows),
        "formal_training_contrast_rows": len(primary_rows),
        "rollout_bootstrap_rows": len(rollout_rows),
        "foundational_structure_rows": len(foundational_rows),
        "complementary_bridge_rows": len(bridge_rows),
        "architecture_transfer_boundary_rows": len(architecture_rows),
        "claims": len(claims),
        "artifacts": artifact_names,
        "output_sha256": {
            name: sha256_file(args.output_dir / name) for name in artifact_names
        },
    }
    write_json(args.output_dir / "unified_mechanism_manifest.json", manifest)
    print(f"Unified mechanism manifest: {args.output_dir / 'unified_mechanism_manifest.json'}")
    print(f"Unified mechanism artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
