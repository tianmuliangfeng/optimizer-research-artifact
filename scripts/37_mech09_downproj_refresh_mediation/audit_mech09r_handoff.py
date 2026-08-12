#!/usr/bin/env python3
"""Independently audit and summarize a completed MECH-09R handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-07-29.1"
EXPECTED_CONTROLLER_VERSION = "2026-07-28.4"
EXPECTED_WORKER_VERSION = "2026-07-28.3"
EXPECTED_ANALYSIS_VERSION = "2026-07-28.3"
EXPECTED_CONTRACT_VERSION = "2026-07-28.2"
EXPECTED_EVALUATION_STEPS = [0, 16, 32, 48, 64, 80, 96, 112, 128]
ARMS = [
    "production_newton_muon",
    "delayed_down_refresh",
    "frozen_down_refresh",
]
CONTRASTS = [
    (
        "delayed_down_refresh_vs_production",
        "delayed_down_refresh",
        "production_newton_muon",
    ),
    (
        "frozen_down_refresh_vs_production",
        "frozen_down_refresh",
        "production_newton_muon",
    ),
    (
        "delayed_down_refresh_vs_frozen_down_refresh",
        "delayed_down_refresh",
        "frozen_down_refresh",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_frame(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy(dtype=float)).all())


def frames_match(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
    *,
    atol: float = 1e-15,
) -> bool:
    if set(observed.columns) != set(expected.columns):
        return False
    columns = sorted(observed.columns)
    left = observed[columns].sort_values(keys).reset_index(drop=True)
    right = expected[columns].sort_values(keys).reset_index(drop=True)
    if len(left) != len(right):
        return False
    for column in columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(
            right[column]
        ):
            if not np.allclose(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
                rtol=0.0,
                atol=atol,
                equal_nan=False,
            ):
                return False
        elif left[column].astype(str).tolist() != right[column].astype(str).tolist():
            return False
    return True


def normalized_archive_inventory(
    archive: Path, run_name: str
) -> dict[str, int]:
    output: dict[str, int] = {}
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            if info.is_dir():
                continue
            pure = Path(info.filename)
            if pure.is_absolute() or ".." in pure.parts:
                raise RuntimeError(f"unsafe archive entry: {info.filename}")
            parts = pure.parts
            if not parts or parts[0] != run_name:
                raise RuntimeError(
                    f"archive entry outside run root {run_name}: {info.filename}"
                )
            relative = Path(*parts[1:]).as_posix()
            if not relative:
                raise RuntimeError(f"invalid archive file entry: {info.filename}")
            output[relative] = int(info.file_size)
    return output


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    grouped = {
        str(origin): group[value_column].to_numpy(dtype=float)
        for origin, group in frame.groupby("checkpoint_cell", sort=True)
    }
    origins = np.array(sorted(grouped))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled_origins = rng.choice(origins, size=len(origins), replace=True)
        origin_values = []
        for origin in sampled_origins:
            values = grouped[str(origin)]
            origin_values.append(
                float(rng.choice(values, size=len(values), replace=True).mean())
            )
        draws[index] = float(np.mean(origin_values))
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def recompute_contrasts(
    evaluation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["checkpoint_cell", "data_replica", "optimizer_step"]
    normalized = evaluation.pivot(
        index=keys, columns="arm", values="normalized_loss"
    )
    heldout = evaluation.pivot(index=keys, columns="arm", values="heldout_loss")
    rows: list[dict[str, Any]] = []
    for contrast, left, right in CONTRASTS:
        for key in normalized.index:
            rows.append(
                {
                    "checkpoint_cell": str(key[0]),
                    "data_replica": int(key[1]),
                    "optimizer_step": int(key[2]),
                    "contrast": contrast,
                    "left": left,
                    "right": right,
                    "left_normalized_loss": float(normalized.loc[key, left]),
                    "right_normalized_loss": float(normalized.loc[key, right]),
                    "normalized_loss_delta": float(
                        normalized.loc[key, left] - normalized.loc[key, right]
                    ),
                    "heldout_loss_delta": float(
                        heldout.loc[key, left] - heldout.loc[key, right]
                    ),
                }
            )
    paired = pd.DataFrame(rows)

    summary_rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    for (contrast, step), group in paired.groupby(
        ["contrast", "optimizer_step"], sort=True
    ):
        origin_means = group.groupby("checkpoint_cell", sort=True)[
            ["normalized_loss_delta", "heldout_loss_delta"]
        ].mean()
        summary_rows.append(
            {
                "contrast": contrast,
                "optimizer_step": int(step),
                "paired_units": len(group),
                "origins": len(origin_means),
                "mean_normalized_loss_delta": float(
                    group["normalized_loss_delta"].mean()
                ),
                "sd_normalized_loss_delta": float(
                    group["normalized_loss_delta"].std(ddof=1)
                ),
                "mean_heldout_loss_delta": float(
                    group["heldout_loss_delta"].mean()
                ),
                "sd_heldout_loss_delta": float(
                    group["heldout_loss_delta"].std(ddof=1)
                ),
                "left_better_units": int(
                    (group["normalized_loss_delta"] < 0.0).sum()
                ),
                "left_worse_units": int(
                    (group["normalized_loss_delta"] > 0.0).sum()
                ),
                "left_better_origins": int(
                    (origin_means["normalized_loss_delta"] < 0.0).sum()
                ),
                "left_worse_origins": int(
                    (origin_means["normalized_loss_delta"] > 0.0).sum()
                ),
            }
        )
        for origin, values in origin_means.iterrows():
            origin_rows.append(
                {
                    "contrast": contrast,
                    "optimizer_step": int(step),
                    "checkpoint_cell": str(origin),
                    "mean_normalized_loss_delta": float(
                        values["normalized_loss_delta"]
                    ),
                    "mean_heldout_loss_delta": float(
                        values["heldout_loss_delta"]
                    ),
                }
            )

    auc_rows: list[dict[str, Any]] = []
    by_arm_auc: dict[str, dict[tuple[str, int], float]] = {}
    for arm in ARMS:
        by_arm_auc[arm] = {}
        arm_frame = evaluation[evaluation["arm"] == arm]
        for key, group in arm_frame.groupby(
            ["checkpoint_cell", "data_replica"], sort=True
        ):
            ordered = group.sort_values("optimizer_step")
            x = ordered["optimizer_step"].to_numpy(dtype=float)
            y = ordered["normalized_loss"].to_numpy(dtype=float)
            area = float(
                np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0)
                / (x[-1] - x[0])
            )
            by_arm_auc[arm][(str(key[0]), int(key[1]))] = area
    for contrast, left, right in CONTRASTS:
        for key in sorted(by_arm_auc[left]):
            left_auc = by_arm_auc[left][key]
            right_auc = by_arm_auc[right][key]
            auc_rows.append(
                {
                    "checkpoint_cell": key[0],
                    "data_replica": key[1],
                    "contrast": contrast,
                    "left": left,
                    "right": right,
                    "left_auc": left_auc,
                    "right_auc": right_auc,
                    "auc_delta": left_auc - right_auc,
                }
            )
    return (
        paired,
        pd.DataFrame(summary_rows),
        pd.DataFrame(origin_rows),
        pd.DataFrame(auc_rows),
    )


def endpoint(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    contrast: str,
    step: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    row = summary[
        (summary["contrast"] == contrast)
        & (summary["optimizer_step"] == step)
    ].iloc[0]
    units = paired[
        (paired["contrast"] == contrast)
        & (paired["optimizer_step"] == step)
    ]
    low, high = hierarchical_bootstrap(
        units,
        "normalized_loss_delta",
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "contrast": contrast,
        "optimizer_step": int(step),
        "mean_normalized_loss_delta": float(
            row["mean_normalized_loss_delta"]
        ),
        "descriptive_hierarchical_bootstrap_95_low": low,
        "descriptive_hierarchical_bootstrap_95_high": high,
        "mean_heldout_loss_delta": float(row["mean_heldout_loss_delta"]),
        "paired_units": int(row["paired_units"]),
        "left_better_units": int(row["left_better_units"]),
        "left_worse_units": int(row["left_worse_units"]),
        "origins": int(row["origins"]),
        "left_better_origins": int(row["left_better_origins"]),
        "left_worse_origins": int(row["left_worse_origins"]),
        "negative_delta_means": "left arm is better",
    }


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {output}")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")

    analysis = run / "analysis"
    exemplar = run / "formal" / "early_muon" / "replica_0"
    contract_path = exemplar / "refresh_mediation_repair_contract.json"
    contract = read_json(contract_path)
    contract_sha = sha256_file(contract_path)
    origins = [str(value) for value in contract["formal"]["origins"]]
    replicas = [int(value) for value in contract["formal"]["data_replicas"]]
    expected_workers = len(origins) * len(replicas)

    status = read_json(run / "status.json")
    identity = read_json(run / "run_identity.json")
    preflight = read_json(run / "preflight.json")
    inventory = read_json(run / "checkpoint_inventory.json")
    smoke_manifest = read_json(run / "smoke" / "smoke_manifest.json")
    formal_manifest = read_json(run / "formal" / "formal_manifest.json")
    analysis_manifest = read_json(
        analysis / "mech09r_analysis_manifest.json"
    )
    analysis_integrity = read_json(analysis / "integrity_checks.json")
    saved_decision = read_json(analysis / "mediation_decision.json")

    checks: dict[str, bool] = {
        "run_status_passed": status.get("status") == "passed",
        "controller_version": status.get("script_version")
        == EXPECTED_CONTROLLER_VERSION,
        "run_identity": identity.get("experiment") == "MECH-09R"
        and identity.get("legacy_invalid_run_reused") is False,
        "run_identity_contract": identity.get("contract_sha256")
        == contract_sha,
        "contract_version": contract.get("contract_version")
        == EXPECTED_CONTRACT_VERSION,
        "preflight": preflight.get("passed") is True
        and all(preflight.get("checks", {}).values()),
        "checkpoint_inventory": inventory.get("passed") is True
        and len(inventory.get("cells", [])) == 4,
        "smoke_manifest": smoke_manifest.get("passed") is True
        and smoke_manifest.get("completed_jobs") == 1
        and smoke_manifest.get("controller_version")
        == EXPECTED_CONTROLLER_VERSION
        and smoke_manifest.get("worker_version") == EXPECTED_WORKER_VERSION,
        "formal_manifest": formal_manifest.get("passed") is True
        and formal_manifest.get("completed_jobs") == expected_workers
        and formal_manifest.get("expected_jobs") == expected_workers,
        "analysis_manifest": analysis_manifest.get("passed") is True
        and analysis_manifest.get("script_version")
        == EXPECTED_ANALYSIS_VERSION
        and analysis_manifest.get("contract_sha256") == contract_sha,
        "analysis_integrity": analysis_integrity.get("passed") is True
        and all(analysis_integrity.get("checks", {}).values()),
        "legacy_invalid_run_not_reused": analysis_manifest.get(
            "legacy_invalid_run_reused"
        )
        is False,
        "timing_excluded": analysis_manifest.get(
            "timing_usable_for_paper"
        )
        is False,
    }

    archive_record: dict[str, Any] | None = None
    if args.archive is not None:
        archive = args.archive.resolve()
        zip_inventory = normalized_archive_inventory(archive, run.name)
        extracted_inventory = {
            relative: (run / relative).stat().st_size
            for relative in zip_inventory
            if (run / relative).is_file()
        }
        checks["archive_file_set"] = set(zip_inventory) == set(
            extracted_inventory
        )
        checks["archive_file_sizes"] = zip_inventory == extracted_inventory
        archive_record = {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
            "files": len(zip_inventory),
            "uncompressed_bytes": int(sum(zip_inventory.values())),
        }

    formal_jobs = {
        (str(row["cell"]), int(row["data_replica"])): row
        for row in formal_manifest["jobs"]
    }
    raw_evaluation: list[pd.DataFrame] = []
    raw_training: list[pd.DataFrame] = []
    worker_rows: list[dict[str, Any]] = []
    for cell in origins:
        spec = next(
            row for row in contract["checkpoints"] if row["cell"] == cell
        )
        for replica in replicas:
            directory = run / "formal" / cell / f"replica_{replica}"
            manifest_path = directory / "mech09r_manifest.json"
            manifest = read_json(manifest_path)
            worker_status = read_json(directory / "status.json")
            worker_checks = read_json(directory / "checks.json")
            branch = read_json(directory / "branch_audit.json")
            refresh = read_json(directory / "refresh_tree_audit.json")
            invariance = read_json(directory / "checkpoint_invariance.json")
            local_contract_sha = sha256_file(
                directory / "refresh_mediation_repair_contract.json"
            )
            formal_row = formal_jobs.get((cell, replica), {})
            label = f"{cell}/replica_{replica}"
            local_checks = {
                "manifest": manifest.get("passed") is True
                and manifest.get("analysis_tier") == "formal"
                and manifest.get("checkpoint_cell") == cell
                and int(manifest.get("data_replica", -1)) == replica,
                "worker_version": manifest.get("script_version")
                == EXPECTED_WORKER_VERSION,
                "status": worker_status.get("status") == "passed",
                "checks": bool(worker_checks)
                and all(worker_checks.values()),
                "branch": branch.get("passed") is True
                and branch.get("shared_evaluation_audit", {}).get("passed")
                is True,
                "refresh": refresh.get("passed") is True,
                "checkpoint_invariance": invariance.get(
                    "checkpoint_size_unchanged"
                )
                is True
                and invariance.get("checkpoint_mtime_unchanged") is True,
                "contract": local_contract_sha == contract_sha
                and manifest.get("contract_sha256") == contract_sha,
                "checkpoint_hash": manifest.get("checkpoint_sha256")
                == spec["expected_sha256"],
                "formal_manifest_hash": formal_row.get("manifest_sha256")
                == sha256_file(manifest_path),
                "causal_tree": manifest.get("causal_tree") is True,
                "legacy_not_reused": manifest.get(
                    "legacy_invalid_run_reused"
                )
                is False,
                "timing_excluded": manifest.get(
                    "timing_usable_for_paper"
                )
                is False,
            }

            evaluation = pd.read_csv(directory / "evaluation.csv")
            training = pd.read_csv(directory / "training.csv")
            local_checks.update(
                {
                    "evaluation_shape": len(evaluation)
                    == len(ARMS) * len(EXPECTED_EVALUATION_STEPS),
                    "evaluation_unique": not evaluation.duplicated(
                        ["arm", "optimizer_step"]
                    ).any(),
                    "evaluation_steps": sorted(
                        evaluation["optimizer_step"].astype(int).unique()
                    )
                    == EXPECTED_EVALUATION_STEPS,
                    "evaluation_arms": set(evaluation["arm"]) == set(ARMS),
                    "evaluation_finite": finite_frame(evaluation),
                    "training_shape": len(training)
                    == len(ARMS) * int(contract["formal"]["rollout_steps"]),
                    "training_unique": not training.duplicated(
                        ["arm", "optimizer_step"]
                    ).any(),
                    "training_arms": set(training["arm"]) == set(ARMS),
                    "training_finite": finite_frame(training),
                    "training_timing_excluded": (
                        training["timing_usable_for_paper"]
                        .astype(str)
                        .str.lower()
                        .eq("false")
                        .all()
                    ),
                }
            )
            for name, passed in local_checks.items():
                checks[f"{label}:{name}"] = bool(passed)
            worker_rows.append(
                {
                    "checkpoint_cell": cell,
                    "data_replica": replica,
                    "passed": all(local_checks.values()),
                    **local_checks,
                }
            )
            raw_evaluation.append(evaluation)
            raw_training.append(training)

    evaluation = pd.concat(raw_evaluation, ignore_index=True)
    training = pd.concat(raw_training, ignore_index=True)
    saved_evaluation = pd.read_csv(analysis / "evaluation_all.csv")
    saved_training = pd.read_csv(analysis / "training_all.csv")
    checks["raw_evaluation_matches_analysis"] = frames_match(
        evaluation,
        saved_evaluation,
        ["checkpoint_cell", "data_replica", "arm", "optimizer_step"],
    )
    checks["raw_training_matches_analysis"] = frames_match(
        training,
        saved_training,
        ["checkpoint_cell", "data_replica", "arm", "optimizer_step"],
    )

    paired, summary, by_origin, auc = recompute_contrasts(evaluation)
    saved_paired = pd.read_csv(analysis / "paired_contrasts.csv")
    saved_paired_for_compare = saved_paired.drop(
        columns=["negative_delta_means"]
    )
    recomputed_paired_for_compare = paired.drop(
        columns=["heldout_loss_delta"]
    )
    checks["paired_contrasts_recomputed"] = frames_match(
        recomputed_paired_for_compare,
        saved_paired_for_compare,
        ["checkpoint_cell", "data_replica", "contrast", "optimizer_step"],
    )
    saved_auc = pd.read_csv(analysis / "auc_contrasts.csv")
    checks["auc_recomputed"] = frames_match(
        auc,
        saved_auc,
        ["checkpoint_cell", "data_replica", "contrast"],
    )

    normalized = evaluation.pivot(
        index=["checkpoint_cell", "data_replica", "optimizer_step"],
        columns="arm",
        values="normalized_loss",
    )
    step16 = normalized.xs(16, level="optimizer_step")
    step48 = normalized.xs(48, level="optimizer_step")
    checks["pre_refresh_all_arms_exact"] = bool(
        (step16.max(axis=1) - step16.min(axis=1)).eq(0.0).all()
    )
    checks["pre_delayed_refresh_exact"] = bool(
        (
            step48["delayed_down_refresh"]
            - step48["frozen_down_refresh"]
        )
        .eq(0.0)
        .all()
    )

    delayed_step48 = summary[
        (summary["contrast"] == "delayed_down_refresh_vs_production")
        & (summary["optimizer_step"] == 48)
    ].iloc[0]
    frozen_step48 = summary[
        (summary["contrast"] == "frozen_down_refresh_vs_production")
        & (summary["optimizer_step"] == 48)
    ].iloc[0]
    delayed_step80 = summary[
        (
            summary["contrast"]
            == "delayed_down_refresh_vs_frozen_down_refresh"
        )
        & (summary["optimizer_step"] == 80)
    ].iloc[0]
    directional = {
        "delayed_protects_after_production_refresh": bool(
            delayed_step48["mean_normalized_loss_delta"] < 0.0
            and delayed_step48["left_better_units"] >= 9
            and delayed_step48["left_better_origins"] >= 3
        ),
        "frozen_protects_after_production_refresh": bool(
            frozen_step48["mean_normalized_loss_delta"] < 0.0
            and frozen_step48["left_better_units"] >= 9
            and frozen_step48["left_better_origins"] >= 3
        ),
        "delayed_worsens_after_its_refresh": bool(
            delayed_step80["mean_normalized_loss_delta"] > 0.0
            and delayed_step80["left_worse_units"] >= 9
            and delayed_step80["left_worse_origins"] >= 3
        ),
    }
    classification = (
        "full_support"
        if all(directional.values())
        else "partial_support"
        if sum(directional.values()) >= 2
        else "not_supported"
    )
    checks["classification_recomputed"] = (
        classification == saved_decision.get("classification")
        and directional == saved_decision.get("directional_predictions")
    )

    auc_summary_rows = []
    for contrast, group in auc.groupby("contrast", sort=True):
        origin_means = group.groupby("checkpoint_cell")["auc_delta"].mean()
        auc_summary_rows.append(
            {
                "contrast": contrast,
                "paired_units": len(group),
                "origins": len(origin_means),
                "mean_auc_delta": float(group["auc_delta"].mean()),
                "sd_auc_delta": float(group["auc_delta"].std(ddof=1)),
                "left_better_units": int((group["auc_delta"] < 0.0).sum()),
                "left_worse_units": int((group["auc_delta"] > 0.0).sum()),
                "left_better_origins": int((origin_means < 0.0).sum()),
                "left_worse_origins": int((origin_means > 0.0).sum()),
            }
        )
    auc_summary = pd.DataFrame(auc_summary_rows)

    key_endpoints = [
        endpoint(
            summary,
            paired,
            "delayed_down_refresh_vs_production",
            48,
            args.bootstrap_samples,
            2026072901,
        ),
        endpoint(
            summary,
            paired,
            "frozen_down_refresh_vs_production",
            48,
            args.bootstrap_samples,
            2026072902,
        ),
        endpoint(
            summary,
            paired,
            "delayed_down_refresh_vs_frozen_down_refresh",
            80,
            args.bootstrap_samples,
            2026072903,
        ),
        endpoint(
            summary,
            paired,
            "delayed_down_refresh_vs_production",
            128,
            args.bootstrap_samples,
            2026072904,
        ),
        endpoint(
            summary,
            paired,
            "frozen_down_refresh_vs_production",
            128,
            args.bootstrap_samples,
            2026072905,
        ),
    ]

    failed = sorted(name for name, passed in checks.items() if not passed)
    quality = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "run_id": run.name,
        "run_dir": str(run),
        "archive": archive_record,
        "contract_sha256": contract_sha,
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed,
        "passed": not failed,
        "assessment": "ready_to_cite_with_scope_caveats"
        if not failed
        else "needs_revision",
    }
    important = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "run_id": run.name,
        "experiment": "MECH-09R",
        "question": contract["scientific_question"],
        "classification": classification,
        "integrity_passed": not failed,
        "design": {
            "formal_workers": expected_workers,
            "checkpoint_origins": origins,
            "data_order_replicas_per_origin": len(replicas),
            "arms": ARMS,
            "rollout_steps": int(contract["formal"]["rollout_steps"]),
            "evaluation_steps": EXPECTED_EVALUATION_STEPS,
            "shared_prefix_all_arms_through_step": 31,
            "shared_prefix_delayed_and_frozen_through_step": 63,
        },
        "directional_predictions": directional,
        "key_endpoints": key_endpoints,
        "auc_summary": json.loads(
            auc_summary.to_json(orient="records")
        ),
        "interpretation": {
            "supported": (
                "Under the frozen restart protocol, a down-projection full-K "
                "refresh causally induces the acute post-refresh held-out-loss "
                "degradation. Delaying the refresh moves the degradation later; "
                "never refreshing the target down-projection K avoids it over "
                "the 128-step horizon."
            ),
            "not_supported_or_out_of_scope": [
                "The result does not establish permanent harm beyond the 128-step horizon.",
                "The result does not establish that every Newton-Muon versus Muon gap is caused by this refresh.",
                "The three data-order replicas and four checkpoint origins are not independent training seeds.",
                "Diagnostic timing is excluded from paper efficiency claims.",
                "Selective-diag versus Selective-none is not a primary comparison.",
            ],
        },
        "bootstrap_note": (
            "Intervals are descriptive hierarchical bootstrap intervals over "
            "four checkpoint origins and three data-order replicas per origin; "
            "they are not training-seed confidence intervals."
        ),
    }

    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(worker_rows).to_csv(
        output / "worker_audit.csv", index=False
    )
    paired.to_csv(output / "paired_contrasts_recomputed.csv", index=False)
    summary.to_csv(output / "contrast_trajectory_summary.csv", index=False)
    by_origin.to_csv(output / "contrast_by_origin.csv", index=False)
    auc.to_csv(output / "auc_contrasts_recomputed.csv", index=False)
    auc_summary.to_csv(output / "auc_summary.csv", index=False)
    write_json(output / "data_quality_audit.json", quality)
    write_json(output / "important_results.json", important)
    artifact_names = sorted(
        [
            "auc_contrasts_recomputed.csv",
            "auc_summary.csv",
            "contrast_by_origin.csv",
            "contrast_trajectory_summary.csv",
            "data_quality_audit.json",
            "important_results.json",
            "independent_audit_manifest.json",
            "paired_contrasts_recomputed.csv",
            "worker_audit.csv",
        ]
    )
    write_json(
        output / "independent_audit_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "run_id": run.name,
            "passed": not failed,
            "classification": classification,
            "artifacts": artifact_names,
        },
    )
    print(f"MECH-09R independent audit: {output}")
    print(f"MECH-09R audit passed: {not failed}")
    print(f"MECH-09R classification: {classification}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
