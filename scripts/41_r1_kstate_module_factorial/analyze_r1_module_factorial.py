#!/usr/bin/env python3
"""Analyze the completed R1 c_fc-by-c_proj 2x2 factorial."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-31.4"
T_CRITICAL_95_DF2 = 4.302652729696142
FORMAL_TOTAL_STEPS = 6200
K_STATE_CONTRACT_ERRATUM_VERSION = "2026-07-31.1"
K_STATE_COMPONENTS_MIB = {
    "shared_attention": 108.0,
    "c_fc_full": 54.0,
    "c_proj_block4": 216.0,
}
CORRECTED_TOTAL_K_STATE_MIB = {
    "both": 378.0,
    "fc_only": 162.0,
    "cproj_only": 324.0,
    "neither": 108.0,
}
SEEDS = (2024, 2025, 2026)
CELL_COORDS = {
    "both": ("full", "block4"),
    "fc_only": ("full", "none"),
    "cproj_only": ("none", "block4"),
    "neither": ("none", "none"),
}
METRICS = ("final_val_loss", "tail5_val_loss_mean", "normalized_val_auc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--existing-summary", type=Path, required=True)
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
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise RuntimeError(f"inconsistent schema: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_validation_curve_evidence(
    batch_dir: Path, summary: dict[str, Any]
) -> tuple[dict[str, float], dict[str, Any]]:
    """Recover derived validation metrics from the sealed per-run curve."""
    run_name = str(summary.get("run_name", ""))
    run_name_path = Path(run_name)
    if (
        not run_name
        or run_name_path.is_absolute()
        or len(run_name_path.parts) != 1
        or run_name_path.name != run_name
    ):
        raise RuntimeError(f"invalid run_name in R1 summary: {run_name!r}")
    metrics_path = batch_dir / run_name / "r1_metrics.csv"
    if not metrics_path.is_file():
        raise RuntimeError(
            f"{run_name}: missing sealed step-0 evidence: {metrics_path}"
        )
    rows = read_csv(metrics_path)
    validation_rows = [
        row
        for row in rows
        if row.get("event") == "validation"
    ]
    if len(validation_rows) < 5:
        raise RuntimeError(
            f"{run_name}: expected at least five validation rows in "
            f"{metrics_path}, observed {len(validation_rows)}"
        )
    validation_rows.sort(key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in validation_rows]
    if (
        len(set(steps)) != len(steps)
        or any(right <= left for left, right in zip(steps, steps[1:]))
        or steps[0] != 0
        or steps[-1] != FORMAL_TOTAL_STEPS
    ):
        raise RuntimeError(
            f"{run_name}: invalid formal validation-step coverage: "
            f"first={steps[0]}, last={steps[-1]}, rows={len(steps)}"
        )
    expected_method = str(summary.get("method", ""))
    expected_mode = str(summary.get("cproj_k_mode", ""))
    observed_methods = {row.get("method") for row in validation_rows}
    if observed_methods != {expected_method}:
        raise RuntimeError(
            f"{run_name}: validation-curve method mismatch "
            f"{sorted(str(value) for value in observed_methods)} "
            f"!= {expected_method!r}"
        )
    observed_modes = {row.get("cproj_k_mode") for row in validation_rows}
    if observed_modes != {expected_mode}:
        raise RuntimeError(
            f"{run_name}: validation-curve cproj mode mismatch "
            f"{sorted(str(value) for value in observed_modes)} "
            f"!= {expected_mode!r}"
        )
    losses = [float(row["loss"]) for row in validation_rows]
    if not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError(f"{run_name}: non-finite validation loss")
    area = sum(
        (right_step - left_step) * (left_loss + right_loss) / 2.0
        for left_step, right_step, left_loss, right_loss in zip(
            steps,
            steps[1:],
            losses,
            losses[1:],
        )
    )
    derived = {
        "initial_val_loss": losses[0],
        "final_val_loss": losses[-1],
        "tail5_val_loss_mean": statistics.fmean(losses[-5:]),
        "normalized_val_auc": area / FORMAL_TOTAL_STEPS,
    }
    required_summary_crosschecks = {
        "final_val_loss": derived["final_val_loss"],
        "val_curve_mean": derived["normalized_val_auc"],
    }
    optional_summary_crosschecks = {
        "initial_val_loss": derived["initial_val_loss"],
        "tail5_val_loss_mean": derived["tail5_val_loss_mean"],
        "normalized_val_auc": derived["normalized_val_auc"],
    }
    for key, expected in required_summary_crosschecks.items():
        if key not in summary or not math.isclose(
            float(summary[key]), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"{run_name}: {key} disagrees with sealed validation curve"
            )
    for key, expected in optional_summary_crosschecks.items():
        if key in summary and not math.isclose(
            float(summary[key]), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"{run_name}: embedded {key} disagrees with sealed validation curve"
            )
    if (
        int(summary.get("final_val_step", -1)) != FORMAL_TOTAL_STEPS
        or int(summary.get("validation_points", -1)) != len(validation_rows)
    ):
        raise RuntimeError(
            f"{run_name}: validation endpoints/count disagree with summary"
        )
    return derived, {
        "run_name": run_name,
        "path": str(metrics_path.resolve()),
        "sha256": sha256_file(metrics_path),
        "rows_read": len(rows),
        "validation_rows": len(validation_rows),
        "first_step": steps[0],
        "last_step": steps[-1],
        **derived,
        "method": expected_method,
        "cproj_k_mode": expected_mode,
    }


def t_summary(values: list[float]) -> dict[str, Any]:
    if len(values) != 3:
        raise RuntimeError(f"three-seed summary required, got {len(values)}")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T_CRITICAL_95_DF2 * sd / math.sqrt(3)
    return {
        "mean": mean,
        "sample_sd": sd,
        "ci95_low_t_df2": mean - half,
        "ci95_high_t_df2": mean + half,
        "negative_seeds": sum(value < 0 for value in values),
        "positive_seeds": sum(value > 0 for value in values),
        "zero_seeds": sum(value == 0 for value in values),
    }


def load_existing_cells(
    path: Path, contract: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_hash = contract["reused_summary"]["frozen_reference_sha256"]
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"reused R1 summary hash mismatch: {observed_hash} != {expected_hash}"
        )
    rows = read_csv(path)
    method_map = contract["reused_summary"]["methods"]
    output: list[dict[str, Any]] = []
    for method, cell in method_map.items():
        selected = [row for row in rows if row["method"] == method]
        by_seed = {int(row["seed"]): row for row in selected}
        if sorted(by_seed) != list(SEEDS) or len(selected) != 3:
            raise RuntimeError(f"existing {method}: three-seed coverage failed")
        cfc_mode, cproj_mode = CELL_COORDS[cell]
        for seed in SEEDS:
            row = by_seed[seed]
            output.append(
                {
                    "cell": cell,
                    "cell_source": "reuse_existing",
                    "seed": seed,
                    "cfc_k_mode": cfc_mode,
                    "cproj_k_mode": cproj_mode,
                    "method_label": (
                        "original_newton_muon_block4"
                        if cell == "both"
                        else "selective_none"
                    ),
                    "run_name": row["run_name"],
                    "initial_val_loss": float(row["initial_val_loss"]),
                    "final_val_loss": float(row["final_val_loss"]),
                    "tail5_val_loss_mean": float(row["tail5_val_loss_mean"]),
                    "normalized_val_auc": float(row["normalized_val_auc"]),
                    "peak_memory_mib": float(row["peak_memory_mib"]),
                    "k_state_mib": float(row["k_state_mib"]),
                    "wandb_uploaded": True,
                    "source_manifest": str(path),
                }
            )
    return output, {
        "path": str(path.resolve()),
        "sha256": observed_hash,
        "canonical_source_path": contract["reused_summary"][
            "canonical_relative_path"
        ],
        "canonical_source_sha256": contract["reused_summary"][
            "canonical_sha256"
        ],
        "rows_read": len(rows),
        "rows_reused": len(output),
    }


def load_new_cells(
    run_dir: Path, contract_sha256: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    manifests = sorted((run_dir / "formal").glob("seed*/**/r1_manifest.json"))
    if not manifests:
        raise RuntimeError(f"no formal R1 manifests found under {run_dir / 'formal'}")
    accepted_by_seed: dict[int, Path] = {}
    for path in manifests:
        payload = read_json(path)
        seed = int(payload.get("seed", -1))
        status = str(payload.get("status", ""))
        if (
            seed in SEEDS
            and status == "completed_valid"
            and payload.get("formal_evidence") is True
            and not payload.get("failures")
        ):
            if seed in accepted_by_seed:
                raise RuntimeError(
                    f"multiple accepted formal batches for seed {seed}: "
                    f"{accepted_by_seed[seed]}, {path}"
                )
            accepted_by_seed[seed] = path
    if sorted(accepted_by_seed) != list(SEEDS):
        raise RuntimeError(
            f"formal accepted seed coverage failed: {sorted(accepted_by_seed)}"
        )

    for seed in SEEDS:
        path = accepted_by_seed[seed]
        payload = read_json(path)
        if payload.get("wandb_complete") is not True:
            raise RuntimeError(f"seed {seed}: W&B upload incomplete")
        factorial = payload.get("module_factorial", {})
        if factorial.get("contract_sha256") != contract_sha256:
            raise RuntimeError(f"seed {seed}: factorial contract hash mismatch")
        summaries = payload.get("summaries", [])
        by_method = {row["method"]: row for row in summaries}
        if set(by_method) != {"block4", "none"}:
            raise RuntimeError(f"seed {seed}: new method coverage failed")
        init_fingerprints = {
            str(row.get("init_sha256", "")) for row in by_method.values()
        }
        if (
            len(init_fingerprints) != 1
            or len(next(iter(init_fingerprints))) != 64
            or any(
                character not in "0123456789abcdef"
                for character in next(iter(init_fingerprints)).lower()
            )
        ):
            raise RuntimeError(
                f"seed {seed}: new-cell initialization fingerprints mismatch "
                f"or are invalid: {sorted(init_fingerprints)}"
            )
        validation_curve_evidence: list[dict[str, Any]] = []
        for method, cell in (("block4", "cproj_only"), ("none", "neither")):
            row = by_method[method]
            cfc_mode, cproj_mode = CELL_COORDS[cell]
            if (
                row.get("cfc_k_mode") != cfc_mode
                or row.get("cproj_k_mode") != cproj_mode
                or row.get("factorial_cell") != cell
            ):
                raise RuntimeError(f"seed {seed} {cell}: runtime mode mismatch")
            validation_metrics, validation_evidence = load_validation_curve_evidence(
                path.parent, row
            )
            validation_curve_evidence.append(validation_evidence)
            output.append(
                {
                    "cell": cell,
                    "cell_source": "new_training",
                    "seed": seed,
                    "cfc_k_mode": cfc_mode,
                    "cproj_k_mode": cproj_mode,
                    "method_label": (
                        "module_bridge"
                        if cell == "cproj_only"
                        else "newton_recipe_all_none"
                    ),
                    "run_name": row["run_name"],
                    "initial_val_loss": validation_metrics["initial_val_loss"],
                    "final_val_loss": float(row["final_val_loss"]),
                    "tail5_val_loss_mean": validation_metrics[
                        "tail5_val_loss_mean"
                    ],
                    "normalized_val_auc": validation_metrics[
                        "normalized_val_auc"
                    ],
                    "peak_memory_mib": float(row["peak_memory_allocated_mib"]),
                    "k_state_mib": float(row["k_state_bytes"]) / (1024**2),
                    "wandb_uploaded": True,
                    "source_manifest": str(path.resolve()),
                }
            )
        audits.append(
            {
                "seed": seed,
                "manifest": str(path.resolve()),
                "manifest_sha256": sha256_file(path),
                "status": payload["status"],
                "wandb_complete": payload["wandb_complete"],
                "summaries": len(summaries),
                "initialization_sha256": next(iter(init_fingerprints)),
                "validation_curve_evidence": validation_curve_evidence,
            }
        )
    return output, audits


def validate_cells(
    rows: list[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, Any]:
    by_key = {(row["cell"], int(row["seed"])): row for row in rows}
    expected_keys = {(cell, seed) for cell in CELL_COORDS for seed in SEEDS}
    if set(by_key) != expected_keys or len(rows) != 12:
        raise RuntimeError("2x2 cell-by-seed coverage failed")
    declared_k = {
        key: float(value)
        for key, value in contract["expected_k_state_mib"].items()
    }
    if declared_k != {
        "both": 378.0,
        "fc_only": 162.0,
        "cproj_only": 216.0,
        "neither": 0.0,
    }:
        raise RuntimeError(
            "factorial contract K-state declaration changed; "
            "the frozen erratum is no longer applicable"
        )
    k_checks: dict[str, bool] = {}
    for cell in CELL_COORDS:
        observed = [by_key[(cell, seed)]["k_state_mib"] for seed in SEEDS]
        k_checks[cell] = all(
            math.isclose(
                value,
                CORRECTED_TOTAL_K_STATE_MIB[cell],
                rel_tol=0.0,
                abs_tol=0.01,
            )
            for value in observed
        )
    if not all(k_checks.values()):
        observed = {
            cell: [by_key[(cell, seed)]["k_state_mib"] for seed in SEEDS]
            for cell in CELL_COORDS
        }
        raise RuntimeError(
            f"corrected total K-state contract failed: "
            f"checks={k_checks}, observed_mib={observed}"
        )
    k_additivity: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        values = {
            cell: float(by_key[(cell, seed)]["k_state_mib"])
            for cell in CELL_COORDS
        }
        components = {
            "shared_attention": values["neither"],
            "c_fc_full_from_cproj_none": values["fc_only"] - values["neither"],
            "c_fc_full_from_cproj_block4": (
                values["both"] - values["cproj_only"]
            ),
            "c_proj_block4_from_cfc_none": (
                values["cproj_only"] - values["neither"]
            ),
            "c_proj_block4_from_cfc_full": (
                values["both"] - values["fc_only"]
            ),
        }
        component_checks = {
            "shared_attention": math.isclose(
                components["shared_attention"],
                K_STATE_COMPONENTS_MIB["shared_attention"],
                rel_tol=0.0,
                abs_tol=0.01,
            ),
            "c_fc_full": all(
                math.isclose(
                    value,
                    K_STATE_COMPONENTS_MIB["c_fc_full"],
                    rel_tol=0.0,
                    abs_tol=0.01,
                )
                for value in (
                    components["c_fc_full_from_cproj_none"],
                    components["c_fc_full_from_cproj_block4"],
                )
            ),
            "c_proj_block4": all(
                math.isclose(
                    value,
                    K_STATE_COMPONENTS_MIB["c_proj_block4"],
                    rel_tol=0.0,
                    abs_tol=0.01,
                )
                for value in (
                    components["c_proj_block4_from_cfc_none"],
                    components["c_proj_block4_from_cfc_full"],
                )
            ),
            "factorial_additivity": math.isclose(
                values["both"] - values["fc_only"]
                - values["cproj_only"] + values["neither"],
                0.0,
                rel_tol=0.0,
                abs_tol=0.01,
            ),
        }
        k_additivity[seed] = {
            "observed_total_mib": values,
            "derived_components_mib": components,
            "checks": component_checks,
            "passed": all(component_checks.values()),
        }
    if not all(row["passed"] for row in k_additivity.values()):
        raise RuntimeError(f"K-state factorial additivity failed: {k_additivity}")
    init_checks: dict[int, bool] = {}
    for seed in SEEDS:
        values = [by_key[(cell, seed)]["initial_val_loss"] for cell in CELL_COORDS]
        init_checks[seed] = max(values) - min(values) <= 1e-9
    if not all(init_checks.values()):
        raise RuntimeError(f"initial validation loss mismatch: {init_checks}")
    return {
        "cell_seed_coverage": True,
        "k_state_checks": k_checks,
        "k_state_additivity_by_seed": k_additivity,
        "k_state_contract_erratum": {
            "schema_version": 1,
            "erratum_version": K_STATE_CONTRACT_ERRATUM_VERSION,
            "required": True,
            "training_or_quality_results_affected": False,
            "reason": (
                "The frozen contract recorded the new cells' MLP-axis "
                "increments as total K-state and omitted the invariant "
                "108 MiB attention qkv/o K-state retained in every cell."
            ),
            "original_contract_unchanged": True,
            "declared_expected_k_state_mib": declared_k,
            "corrected_total_k_state_mib": CORRECTED_TOTAL_K_STATE_MIB,
            "component_k_state_mib": K_STATE_COMPONENTS_MIB,
            "derivation": {
                "architecture": "R1 Modded-NanoGPT, 12 layers, d=768",
                "dtype": "float32",
                "persistent_tensors_per_factor": [
                    "precond_cov",
                    "precond_inv_apply",
                ],
                "shared_attention": (
                    "12 layers * (qkv + attention o_proj) * "
                    "(cov + inverse) * 768^2 * 4 bytes = 108 MiB"
                ),
                "c_fc_full": (
                    "12 layers * c_fc * (cov + inverse) * "
                    "768^2 * 4 bytes = 54 MiB"
                ),
                "c_proj_block4": (
                    "12 layers * 4 blocks * c_proj * (cov + inverse) * "
                    "768^2 * 4 bytes = 216 MiB"
                ),
            },
            "observed_additivity_verified": True,
        },
        "initial_loss_equal_within_seed": init_checks,
        "timing_usable": False,
    }


def build_effects(
    rows: list[dict[str, Any]], practical_margin: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {(row["cell"], int(row["seed"])): row for row in rows}
    by_seed: list[dict[str, Any]] = []
    effect_names = (
        "cfc_main",
        "cproj_main",
        "interaction",
        "disable_cproj_when_cfc_full",
        "disable_cproj_when_cfc_none",
        "disable_cfc_when_cproj_block4",
        "disable_cfc_when_cproj_none",
    )
    for metric in METRICS:
        for seed in SEEDS:
            y11 = float(by_key[("both", seed)][metric])
            y10 = float(by_key[("fc_only", seed)][metric])
            y01 = float(by_key[("cproj_only", seed)][metric])
            y00 = float(by_key[("neither", seed)][metric])
            values = {
                "cfc_main": 0.5 * ((y11 - y01) + (y10 - y00)),
                "cproj_main": 0.5 * ((y11 - y10) + (y01 - y00)),
                "interaction": y11 - y10 - y01 + y00,
                "disable_cproj_when_cfc_full": y10 - y11,
                "disable_cproj_when_cfc_none": y00 - y01,
                "disable_cfc_when_cproj_block4": y01 - y11,
                "disable_cfc_when_cproj_none": y00 - y10,
            }
            for effect in effect_names:
                by_seed.append(
                    {
                        "metric": metric,
                        "seed": seed,
                        "effect": effect,
                        "value": values[effect],
                        "negative_means": (
                            "enabled factor lowers loss"
                            if effect in {"cfc_main", "cproj_main"}
                            else "left/minus operation lowers loss"
                        ),
                    }
                )
    summary: list[dict[str, Any]] = []
    for metric in METRICS:
        for effect in effect_names:
            values = [
                float(row["value"])
                for row in by_seed
                if row["metric"] == metric and row["effect"] == effect
            ]
            stats = t_summary(values)
            summary.append(
                {
                    "metric": metric,
                    "effect": effect,
                    "seeds": len(values),
                    **stats,
                    "practical_margin": practical_margin,
                    "material_by_mean": abs(stats["mean"]) >= practical_margin,
                    "direction_consistent_2of3": max(
                        stats["negative_seeds"], stats["positive_seeds"]
                    )
                    >= 2,
                }
            )
    return by_seed, summary


def classify(
    summaries: list[dict[str, Any]], practical_margin: float
) -> dict[str, Any]:
    primary = {
        row["effect"]: row
        for row in summaries
        if row["metric"] == "final_val_loss"
    }
    cfc = primary["cfc_main"]
    cproj = primary["cproj_main"]
    interaction = primary["interaction"]
    cfc_beneficial = (
        cfc["mean"] <= -practical_margin and cfc["negative_seeds"] >= 2
    )
    cproj_harmful = (
        cproj["mean"] >= practical_margin and cproj["positive_seeds"] >= 2
    )
    cproj_beneficial = (
        cproj["mean"] <= -practical_margin and cproj["negative_seeds"] >= 2
    )
    material_interaction = (
        abs(interaction["mean"]) >= practical_margin
        and max(interaction["negative_seeds"], interaction["positive_seeds"]) >= 2
    )
    if cfc_beneficial and cproj_harmful:
        classification = "historical_allocation_reproduced"
    elif cproj_beneficial:
        classification = "r1_allocation_diverges"
    elif cfc_beneficial:
        classification = "partial_non_cproj_support"
    else:
        classification = "inconclusive_within_margin"
    next_step = {
        "historical_allocation_reproduced": (
            "stop new mechanism training and integrate the architecture-qualified result"
        ),
        "partial_non_cproj_support": (
            "inspect simple effects; add no experiment unless the interaction changes the claim"
        ),
        "r1_allocation_diverges": (
            "treat OWT/WikiText allocation as architecture-specific and avoid post-hoc sweeps"
        ),
        "inconclusive_within_margin": (
            "combine with experiment 39, then decide whether two additional seeds are worth the claim"
        ),
    }[classification]
    return {
        "classification": classification,
        "material_interaction": material_interaction,
        "practical_margin": practical_margin,
        "cfc_main": cfc,
        "cproj_main": cproj,
        "interaction": interaction,
        "cfc_beneficial": cfc_beneficial,
        "cproj_beneficial": cproj_beneficial,
        "cproj_harmful": cproj_harmful,
        "next_step": next_step,
        "diag_in_factorial": False,
        "all_none_is_muon_baseline": False,
    }


def build_report(
    rows: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    means = {
        cell: statistics.mean(
            row["final_val_loss"] for row in rows if row["cell"] == cell
        )
        for cell in CELL_COORDS
    }
    primary = {
        row["effect"]: row
        for row in effects
        if row["metric"] == "final_val_loss"
        and row["effect"] in {"cfc_main", "cproj_main", "interaction"}
    }
    return f"""# R1 K-State Module 2x2 Factorial

## Result

Classification: **{decision['classification']}**.

Mean final validation loss:

- both: {means['both']:.6f}
- fc_only / Selective none: {means['fc_only']:.6f}
- cproj_only: {means['cproj_only']:.6f}
- neither / Newton-recipe all-none: {means['neither']:.6f}

## Factorial effects

Loss effects use enabled-minus-disabled coding; negative means enabling the K
factor lowers loss.

- c_fc main: {primary['cfc_main']['mean']:.6f}
  [{primary['cfc_main']['ci95_low_t_df2']:.6f},
  {primary['cfc_main']['ci95_high_t_df2']:.6f}]
- c_proj block4 main: {primary['cproj_main']['mean']:.6f}
  [{primary['cproj_main']['ci95_low_t_df2']:.6f},
  {primary['cproj_main']['ci95_high_t_df2']:.6f}]
- interaction: {primary['interaction']['mean']:.6f}
  [{primary['interaction']['ci95_low_t_df2']:.6f},
  {primary['interaction']['ci95_high_t_df2']:.6f}]

Material interaction: {decision['material_interaction']}.

## Boundaries

- The two c_fc=full cells are reused from the frozen experiment-15 summary.
- Only cproj_only and neither were newly trained.
- Neither uses the Newton recipe and must not be relabeled as the formal Muon
  baseline.
- Diag is not a factorial cell; it remains a separate Selective proposal.
- The frozen contract's new-cell K-state totals omitted the invariant 108 MiB
  attention K-state. The analysis applies a versioned erratum and verifies the
  corrected 108 + 54 + 216 MiB additive decomposition against every seed.
- Concurrent timing is ineligible for paper efficiency claims.

## Next step

{decision['next_step']}.
"""


def main() -> None:
    args = parse_args()
    contract = read_json(args.contract)
    if contract.get("schema_version") != 1:
        raise RuntimeError("unsupported factorial contract")
    contract_sha = sha256_file(args.contract)
    existing, existing_audit = load_existing_cells(args.existing_summary, contract)
    new, new_audits = load_new_cells(args.run_dir, contract_sha)
    rows = sorted(
        existing + new,
        key=lambda row: (list(CELL_COORDS).index(row["cell"]), row["seed"]),
    )
    checks = validate_cells(rows, contract)
    by_seed, effects = build_effects(
        rows, float(contract["practical_loss_margin"])
    )
    decision = classify(effects, float(contract["practical_loss_margin"]))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "factorial_cells.csv", rows)
    write_csv(args.output_dir / "factorial_effects_by_seed.csv", by_seed)
    write_csv(args.output_dir / "factorial_effects_summary.csv", effects)
    write_json(args.output_dir / "factorial_decision.json", decision)
    write_json(
        args.output_dir / "source_audit.json",
        {
            "existing": existing_audit,
            "new_formal_batches": new_audits,
            "contract": {
                "path": str(args.contract.resolve()),
                "sha256": contract_sha,
            },
        },
    )
    write_json(args.output_dir / "checks.json", checks)
    write_json(
        args.output_dir / "k_state_contract_erratum.json",
        checks["k_state_contract_erratum"],
    )
    report_name = "R1_MODULE_FACTORIAL_REPORT.md"
    (args.output_dir / report_name).write_text(
        build_report(rows, effects, decision), encoding="utf-8"
    )
    artifacts = [
        "factorial_cells.csv",
        "factorial_effects_by_seed.csv",
        "factorial_effects_summary.csv",
        "factorial_decision.json",
        "source_audit.json",
        "checks.json",
        "k_state_contract_erratum.json",
        report_name,
    ]
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "passed": True,
        "classification": decision["classification"],
        "material_interaction": decision["material_interaction"],
        "contract_sha256": contract_sha,
        "seeds": list(SEEDS),
        "cells": list(CELL_COORDS),
        "new_training_runs": 6,
        "reused_runs": 6,
        "timing_usable": False,
        "wandb_required_and_complete": True,
        "contract_erratum_applied": True,
        "contract_erratum_version": K_STATE_CONTRACT_ERRATUM_VERSION,
        "artifacts": artifacts,
        "output_sha256": {
            name: sha256_file(args.output_dir / name) for name in artifacts
        },
    }
    write_json(
        args.output_dir / "r1_module_factorial_analysis_manifest.json",
        manifest,
    )
    print(
        "R1 module factorial analysis manifest: "
        f"{args.output_dir / 'r1_module_factorial_analysis_manifest.json'}"
    )


if __name__ == "__main__":
    main()
