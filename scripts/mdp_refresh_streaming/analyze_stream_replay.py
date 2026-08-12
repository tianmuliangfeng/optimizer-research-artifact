#!/usr/bin/env python3
"""Independent local analysis for the completed MDP-04 stream replay.

This script never changes the remote handoff directory.  It verifies the
handoff manifest and emits compact, deterministic descriptive tables in a
separate output directory.  The 12 origin--replica units are the statistical
units; layer rows are nested measurements and are never treated as seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-08-03.2"
EXPECTED_ORIGINS = (
    "early_muon",
    "early_newton_full",
    "late_muon",
    "late_newton_full",
)
EXPECTED_EVENTS = ("production_refresh_32", "delayed_refresh_64")
EXPECTED_REPLICAS = (0, 1, 2)
EXPECTED_LAYERS = tuple(range(18))

PRIMARY_METRICS = (
    "relative_k_fro_change_layer_median",
    "relative_a_fro_change_layer_median",
    "relative_runtime_inverse_fro_change_layer_median",
    "matched_g_preconditioned_relative_change_layer_median",
    "runtime_ns5_update_relative_change_layer_median",
)

SECONDARY_METRICS = (
    "condition_proxy_before_layer_median",
    "condition_proxy_after_layer_median",
    "runtime_resolvent_relative_residual_layer_median",
    "matched_g_preconditioned_pooled_fro_ratio",
    "matched_g_preconditioned_pooled_cosine",
    "runtime_ns5_update_pooled_fro_ratio",
    "runtime_ns5_update_pooled_cosine",
)

LAYER_DIAGNOSTIC_METRICS = (
    "condition_proxy_before",
    "condition_proxy_after",
    "relative_a_fro_change",
    "relative_runtime_inverse_fro_change",
    "runtime_inverse_backward_residual_after",
    "probe_bank_condition_relative_disagreement",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    native_path = str(path)
    if os.name == "nt" and path.is_absolute() and not native_path.startswith("\\\\?\\"):
        native_path = "\\\\?\\" + native_path
    with open(native_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def finite_float(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite value: {value!r}")
    return result


def correlation(x: pd.Series, y: pd.Series, method: str) -> float:
    x_num = pd.to_numeric(x, errors="raise").astype(float)
    y_num = pd.to_numeric(y, errors="raise").astype(float)
    if x_num.nunique(dropna=False) < 2 or y_num.nunique(dropna=False) < 2:
        return float("nan")
    if method == "spearman":
        return finite_float(x_num.rank(method="average").corr(y_num.rank(method="average")))
    if method == "pearson":
        return finite_float(x_num.corr(y_num))
    raise ValueError(method)


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("quantile input is empty or non-finite")
    return {
        "min": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def local_selected_attempt(run_dir: Path, remote_path: str) -> Path:
    parts = Path(remote_path.replace("\\", "/")).parts
    try:
        index = parts.index(run_dir.name)
    except ValueError as exc:
        raise ValueError(f"selected attempt is outside run {run_dir.name}: {remote_path}") from exc
    return run_dir.joinpath(*parts[index + 1 :])


def verify_handoff(
    run_dir: Path,
    formal_manifest: dict[str, Any],
    layer: pd.DataFrame,
    unit_event: pd.DataFrame,
    outcomes: pd.DataFrame,
    joined: pd.DataFrame,
    slices: pd.DataFrame,
) -> dict[str, Any]:
    analysis_dir = run_dir / "analysis"
    expected_hashes = formal_manifest["artifacts"]
    artifact_hash_checks = {
        name: sha256(analysis_dir / name) == expected
        for name, expected in sorted(expected_hashes.items())
    }

    unit_checks: list[dict[str, Any]] = []
    for lineage in formal_manifest["unit_contract_lineage"]:
        attempt = local_selected_attempt(run_dir, lineage["selected_attempt"])
        status = read_json(attempt / "status.json")
        manifest_path = attempt / "stream_unit_manifest.json"
        manifest = read_json(manifest_path)
        selection = read_json(attempt.parent / "unit_selection.json")
        unit_artifact_hashes = {
            name: sha256(attempt / name) == expected
            for name, expected in sorted(manifest["artifact_sha256"].items())
        }
        artifact_mismatches = sorted(
            name for name, passed in unit_artifact_hashes.items() if not passed
        )
        scientific_artifact_mismatches = [
            name for name in artifact_mismatches if name != "worker.log"
        ]
        unit_checks.append(
            {
                "origin": lineage["origin"],
                "data_replica": int(lineage["data_replica"]),
                "lineage": lineage["lineage"],
                "attempt": attempt.relative_to(run_dir).as_posix(),
                "status_passed": status.get("status") == "passed",
                "manifest_passed": manifest.get("passed") is True,
                "manifest_layer_event_rows": manifest.get("layer_event_rows") == 36,
                "selection_passed": selection.get("passed") is True,
                "selection_attempt": selection.get("selected_attempt") == attempt.name,
                "selection_manifest_sha256": selection.get("manifest_sha256")
                == sha256(manifest_path),
                "scientific_artifact_hashes": not scientific_artifact_mismatches,
                "worker_log_hash_match_diagnostic": unit_artifact_hashes.get(
                    "worker.log", True
                ),
                "artifact_hash_mismatches": artifact_mismatches,
                "manifest_identity": (
                    manifest.get("origin") == lineage["origin"]
                    and int(manifest.get("data_replica")) == int(lineage["data_replica"])
                ),
            }
        )

    layer_keys = ["origin", "data_replica", "event_id", "layer_index"]
    unit_event_keys = ["origin", "data_replica", "event_id"]
    outcome_keys = ["origin", "data_replica"]
    expected_unit_event = {
        (origin, replica, event)
        for origin in EXPECTED_ORIGINS
        for replica in EXPECTED_REPLICAS
        for event in EXPECTED_EVENTS
    }
    expected_layer = {
        (origin, replica, event, layer_index)
        for origin in EXPECTED_ORIGINS
        for replica in EXPECTED_REPLICAS
        for event in EXPECTED_EVENTS
        for layer_index in EXPECTED_LAYERS
    }
    expected_outcomes = {
        (origin, replica) for origin in EXPECTED_ORIGINS for replica in EXPECTED_REPLICAS
    }

    observed_layer = set(map(tuple, layer[layer_keys].itertuples(index=False, name=None)))
    observed_unit_event = set(
        map(tuple, unit_event[unit_event_keys].itertuples(index=False, name=None))
    )
    observed_joined = set(map(tuple, joined[unit_event_keys].itertuples(index=False, name=None)))
    observed_outcomes = set(map(tuple, outcomes[outcome_keys].itertuples(index=False, name=None)))

    checks = {
        "formal_manifest_exists": True,
        "artifact_hashes": all(artifact_hash_checks.values()),
        "selected_units_12": len(unit_checks) == 12,
        "selected_units_passed": all(
            row["status_passed"]
            and row["manifest_passed"]
            and row["manifest_layer_event_rows"]
            and row["selection_passed"]
            and row["selection_attempt"]
            and row["selection_manifest_sha256"]
            and row["scientific_artifact_hashes"]
            and row["manifest_identity"]
            for row in unit_checks
        ),
        "layer_rows_432": len(layer) == 432,
        "layer_keys_unique": not layer.duplicated(layer_keys).any(),
        "layer_coverage_exact": observed_layer == expected_layer,
        "unit_event_rows_24": len(unit_event) == 24,
        "unit_event_keys_unique": not unit_event.duplicated(unit_event_keys).any(),
        "unit_event_coverage_exact": observed_unit_event == expected_unit_event,
        "joined_rows_24": len(joined) == 24,
        "joined_keys_unique": not joined.duplicated(unit_event_keys).any(),
        "joined_coverage_exact": observed_joined == expected_unit_event,
        "outcome_rows_12": len(outcomes) == 12,
        "outcome_keys_unique": not outcomes.duplicated(outcome_keys).any(),
        "outcome_coverage_exact": observed_outcomes == expected_outcomes,
        "validation_slice_rows_6": len(slices) == 6,
        "all_layer_values_finite": bool(layer["all_full_state_values_finite"].astype(str).str.lower().eq("true").all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"handoff verification failed: {checks}")
    return {
        "checks": checks,
        "artifact_hash_checks": artifact_hash_checks,
        "unit_checks": unit_checks,
    }


def add_design_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["stage"] = result["origin"].str.split("_", n=1).str[0]
    result["source_method"] = result["origin"].str.replace(
        r"^(early|late)_", "", regex=True
    )
    return result


def build_resolvent_strata(layer: pd.DataFrame, gate: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["stage", "source_method", "event_id", "layer_index"]
    for key, group in layer.groupby(group_columns, sort=True):
        residual = group["runtime_resolvent_relative_residual"].astype(float)
        rows.append(
            {
                **dict(zip(group_columns, key)),
                "nested_row_count": len(group),
                "residual_median": float(residual.median()),
                "residual_mean": float(residual.mean()),
                "residual_max": float(residual.max()),
                "hard_gate": gate,
                "above_gate_count": int((residual > gate).sum()),
                "condition_before_median": float(group["condition_proxy_before"].median()),
                "condition_after_median": float(group["condition_proxy_after"].median()),
                "relative_a_change_median": float(group["relative_a_fro_change"].median()),
                "relative_runtime_inverse_change_median": float(
                    group["relative_runtime_inverse_fro_change"].median()
                ),
                "matched_g_change_median": float(
                    group["matched_g_preconditioned_relative_change"].median()
                ),
                "runtime_ns5_change_median": float(
                    group["runtime_ns5_update_relative_change"].median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def build_matrix_loss_associations(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_id, group in joined.groupby("event_id", sort=True):
        for metric in PRIMARY_METRICS + SECONDARY_METRICS:
            for outcome in ("oriented_loss_harm", "oriented_auc_harm"):
                full_spearman = correlation(group[metric], group[outcome], "spearman")
                origin_means = group.groupby("origin", sort=True)[[metric, outcome]].mean()
                metric_centered = group[metric].astype(float) - group.groupby("origin")[
                    metric
                ].transform("mean")
                outcome_centered = group[outcome].astype(float) - group.groupby("origin")[
                    outcome
                ].transform("mean")
                leave_one_origin_out = []
                for origin in sorted(group["origin"].unique()):
                    subset = group.loc[group["origin"].ne(origin)]
                    leave_one_origin_out.append(
                        correlation(subset[metric], subset[outcome], "spearman")
                    )
                nonzero_signs = [
                    int(np.sign(value))
                    for value in leave_one_origin_out
                    if np.isfinite(value) and value != 0
                ]
                rows.append(
                    {
                        "event_id": event_id,
                        "metric": metric,
                        "outcome": outcome,
                        "nested_unit_count": len(group),
                        "aggregation_role": (
                            "primary_layer_median"
                            if metric in PRIMARY_METRICS
                            else "secondary_pre_registered"
                        ),
                        "spearman_rho": full_spearman,
                        "pearson_r": correlation(group[metric], group[outcome], "pearson"),
                        "origin_mean_count": len(origin_means),
                        "origin_mean_spearman_rho": correlation(
                            origin_means[metric], origin_means[outcome], "spearman"
                        ),
                        "within_origin_centered_pearson_r": correlation(
                            metric_centered, outcome_centered, "pearson"
                        ),
                        "leave_one_origin_out_spearman_min": float(
                            np.min(leave_one_origin_out)
                        ),
                        "leave_one_origin_out_spearman_max": float(
                            np.max(leave_one_origin_out)
                        ),
                        "leave_one_origin_out_sign_consistent": bool(
                            nonzero_signs
                            and all(sign == int(np.sign(full_spearman)) for sign in nonzero_signs)
                        ),
                        "inference_role": "descriptive_only_no_seed_level_p_value",
                        "formal_claim_eligible": False,
                    }
                )
    return pd.DataFrame(rows).sort_values(["event_id", "outcome", "metric"]).reset_index(drop=True)


def build_layer_diagnostic_associations(layer: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = {
        "all_nested_rows": np.ones(len(layer), dtype=bool),
        "early_nested_rows": layer["stage"].eq("early"),
        "late_nested_rows": layer["stage"].eq("late"),
        "layer3_nested_rows": layer["layer_index"].eq(3),
        "late_layer3_nested_rows": layer["stage"].eq("late") & layer["layer_index"].eq(3),
    }
    target = "runtime_resolvent_relative_residual"
    for scope, mask in scopes.items():
        group = layer.loc[mask]
        for metric in LAYER_DIAGNOSTIC_METRICS:
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "target": target,
                    "nested_row_count": len(group),
                    "spearman_rho": correlation(group[metric], group[target], "spearman"),
                    "pearson_r": correlation(group[metric], group[target], "pearson"),
                    "inference_role": "posthoc_numerical_diagnostic_nested_rows_not_independent",
                }
            )
    return pd.DataFrame(rows).sort_values(["scope", "metric"]).reset_index(drop=True)


def build_origin_event_summary(joined: pd.DataFrame) -> pd.DataFrame:
    metrics = list(PRIMARY_METRICS) + [
        "runtime_resolvent_relative_residual_layer_median",
        "oriented_loss_harm",
        "oriented_auc_harm",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in joined.groupby(["origin", "event_id"], sort=True):
        row: dict[str, Any] = {
            "origin": key[0],
            "event_id": key[1],
            "nested_replica_count": len(group),
        }
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["origin", "event_id"]).reset_index(drop=True)


def association_record(
    associations: pd.DataFrame, event_id: str, metric: str, outcome: str
) -> pd.Series:
    selected = associations.loc[
        associations["event_id"].eq(event_id)
        & associations["metric"].eq(metric)
        & associations["outcome"].eq(outcome)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one association row: event={event_id}, metric={metric}, outcome={outcome}"
        )
    return selected.iloc[0]


def fmt(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def build_report(
    formal_manifest: dict[str, Any],
    verification: dict[str, Any],
    failures: pd.DataFrame,
    layer: pd.DataFrame,
    slices: pd.DataFrame,
    matrix_loss: pd.DataFrame,
    gate: float,
) -> str:
    residual = layer["runtime_resolvent_relative_residual"].astype(float)
    non_layer3 = layer.loc[
        layer["layer_index"].ne(3), "runtime_resolvent_relative_residual"
    ].astype(float)
    slice_max = slices["float64_slice_resolvent_relative_residual"].astype(float).max()
    worker_log_mismatches = sum(
        not row["worker_log_hash_match_diagnostic"]
        for row in verification["unit_checks"]
    )

    def association_line(event: str, metric: str, outcome: str) -> str:
        row = association_record(matrix_loss, event, metric, outcome)
        event_label = "production@32" if event == "production_refresh_32" else "delayed@64"
        metric_label = {
            "matched_g_preconditioned_relative_change_layer_median": "matched-G median",
            "runtime_ns5_update_relative_change_layer_median": "NS5 median",
            "matched_g_preconditioned_pooled_fro_ratio": "matched-G pooled ratio",
            "runtime_ns5_update_pooled_fro_ratio": "NS5 pooled ratio",
            "relative_k_fro_change_layer_median": "Delta-K median",
            "relative_a_fro_change_layer_median": "Delta-A median",
            "relative_runtime_inverse_fro_change_layer_median": "runtime-inverse median",
        }[metric]
        return (
            f"| {event_label} | {metric_label} | {fmt(row['spearman_rho'])} | "
            f"{fmt(row['pearson_r'])} | {fmt(row['within_origin_centered_pearson_r'])} | "
            f"[{fmt(row['leave_one_origin_out_spearman_min'])}, "
            f"{fmt(row['leave_one_origin_out_spearman_max'])}] |"
        )

    loss_rows = []
    auc_rows = []
    upstream_rows = []
    for event in EXPECTED_EVENTS:
        for metric in (
            "matched_g_preconditioned_relative_change_layer_median",
            "runtime_ns5_update_relative_change_layer_median",
            "matched_g_preconditioned_pooled_fro_ratio",
            "runtime_ns5_update_pooled_fro_ratio",
        ):
            loss_rows.append(association_line(event, metric, "oriented_loss_harm"))
        for metric in (
            "matched_g_preconditioned_relative_change_layer_median",
            "runtime_ns5_update_relative_change_layer_median",
        ):
            auc_rows.append(association_line(event, metric, "oriented_auc_harm"))
        for metric in (
            "relative_k_fro_change_layer_median",
            "relative_a_fro_change_layer_median",
            "relative_runtime_inverse_fro_change_layer_median",
        ):
            upstream_rows.append(association_line(event, metric, "oriented_loss_harm"))

    production_residual = association_record(
        matrix_loss,
        "production_refresh_32",
        "runtime_resolvent_relative_residual_layer_median",
        "oriented_loss_harm",
    )
    delayed_residual = association_record(
        matrix_loss,
        "delayed_refresh_64",
        "runtime_resolvent_relative_residual_layer_median",
        "oriented_loss_harm",
    )

    lines = [
        "# MDP-04 final local audit",
        "",
        "Date: 2026-08-03  ",
        "Source run: `20260803T063912+0000`  ",
        "Computation: **complete (12/12 units)**  ",
        "Formal adjudication: **`numeric_gate_failed`**  ",
        "Matrix evidence status: **descriptive partial / non-claim-eligible**",
        "",
        "## 1. Delivery and integrity",
        "",
        "The handoff contains the exact frozen coverage: 12 selected origin--replica units, "
        "24 unit-event summaries, 432 layer-event rows, 12 accepted loss-outcome rows, "
        "and six validation slices. All selected unit manifests pass. The final lineage is "
        f"{formal_manifest['unit_contract_lineage_counts']['inherited_stricter_v3']} inherited "
        "stricter-v3 units plus "
        f"{formal_manifest['unit_contract_lineage_counts']['current_v4']} current-v4 units.",
        "",
        "All summary artifacts and all scientific files named by the 12 unit manifests "
        "match their recorded SHA-256 values. The only provenance diagnostic is that "
        f"`worker.log` differs from its unit-manifest hash in {worker_log_mismatches}/12 units, "
        "consistent with logs continuing to append after manifest creation. Logs are not "
        "scientific inputs and this does not alter any CSV/JSON/NPZ value.",
        "",
        "## 2. Why formal validation failed",
        "",
        "The frozen full-run threshold was not changed. "
        f"`runtime_resolvent_relative_residual <= {gate:.2f}` failed in "
        f"{len(failures)}/432 nested layer rows ({100 * len(failures) / len(layer):.2f}%). "
        f"The maximum is {residual.max():.8f}. Every failure is late-stage layer 3: five "
        "rows from `late_muon` and one from `late_newton_full`. The other 408 non-layer-3 "
        f"rows have maximum {non_layer3.max():.8f}, below the gate.",
        "",
        "The six failing rows have condition proxies around 26.3k--26.6k before refresh "
        "and 25.0k--25.4k after refresh. Across all nested layer rows the residual is strongly "
        "associated with the condition proxy (post-hoc numerical diagnostic, not seed-level "
        "inference). This localizes the problem to an ill-conditioned late layer rather than "
        "missing rows, non-finite values, source drift, or a failed shadow-to-actual update audit.",
        "",
        f"The registered 128-coordinate float64 slices have maximum resolvent residual "
        f"{slice_max:.3e}. They calibrate the implementation only, do not include layer 3, "
        "and were pre-labelled non-claim-eligible; they cannot rescue the full-run gate.",
        "",
        "## 3. Descriptive matrix-to-loss alignment",
        "",
        "All correlations below use the 12 nested origin--replica units separately for each "
        "event. They have no seed-level p-value. `within-origin` removes the four origin means; "
        "LOOO is the range after leaving out each origin in turn.",
        "",
        "### Oriented endpoint loss harm",
        "",
        "| Event | Metric | Spearman | Pearson | Within-origin Pearson | LOOO Spearman range |",
        "|---|---|---:|---:|---:|---:|",
        *loss_rows,
        "",
        "### Oriented AUC harm (primary layer medians)",
        "",
        "| Event | Metric | Spearman | Pearson | Within-origin Pearson | LOOO Spearman range |",
        "|---|---|---:|---:|---:|---:|",
        *auc_rows,
        "",
        "The downstream matched-gradient and actual source-pinned NS5 measures retain the "
        "same sign in every leave-one-origin-out check for both events and both loss outcomes. "
        "The pooled Frobenius ratios are secondary pre-registered aggregations; the 18-layer "
        "median is the primary aggregation.",
        "",
        "### Upstream magnitude checks",
        "",
        "| Event | Metric | Spearman | Pearson | Within-origin Pearson | LOOO Spearman range |",
        "|---|---|---:|---:|---:|---:|",
        *upstream_rows,
        "",
        "Raw K/A change and runtime-inverse change magnitude do not track loss harm "
        "consistently; their leave-one-origin-out signs change. By contrast, the shock after "
        "applying the same gradient and after the production NS5 pipeline is consistently "
        "aligned with harm. This favors a gradient- and update-conditioned mechanism over a "
        "simple `larger covariance change is worse` account.",
        "",
        "The failing resolvent proxy itself is not the observed loss mediator: its median "
        f"Spearman correlation with loss harm is {float(production_residual['spearman_rho']):.3f} "
        "at production and "
        f"{float(delayed_residual['spearman_rho']):.3f} at delayed refresh. The numerical-gate "
        "failure and the downstream update-alignment signal therefore need to be reported as "
        "two distinct facts.",
        "",
        "## 4. Scientific adjudication",
        "",
        "1. Experiment 37 remains accepted evidence that the scheduled down-projection refresh "
        "causes the short-horizon loss impulse under its frozen intervention tree.",
        "2. This replay adds a coherent descriptive signal: matched-G preconditioned-gradient "
        "shock and actual NS5 update shock track both endpoint and AUC harm across both registered "
        "events, including within-origin and leave-one-origin-out checks.",
        "3. MDP-04 cannot be promoted to formal claim-eligible evidence because one frozen hard "
        "gate failed. The threshold must not be relaxed and layer 3 must not be removed post hoc.",
        "4. No long-horizon optimizer ranking, universal route ranking, or automatic selector "
        "follows from these 12 nested replay units.",
        "",
        "## 5. Project decision",
        "",
        "Default action is to stop remote MDP-04 computation, preserve this result as a "
        "numerically limited but scientifically informative diagnostic, and proceed with local "
        "evidence freezing and paper writing. The accepted paper-level mechanism remains the "
        "loss-level causal refresh result from experiment 37; the matrix alignment may guide "
        "discussion, limitations, and a future independently pre-registered confirmation, but "
        "must not be presented as a passed formal MDP-04 claim.",
        "",
        "A future confirmation is optional, not queued. If explicitly authorized, it needs a "
        "new contract and independent data fixed before observation; reusing this run or changing "
        "the 0.01 gate cannot upgrade the present evidence.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--handoff-zip", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("output directory must be separate from the immutable handoff directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_dir = run_dir / "analysis"
    formal_manifest = read_json(analysis_dir / "formal_stream_manifest.json")
    contract_path = run_dir / "source_snapshot_v4" / "scripts" / "mdp_refresh_streaming" / "refresh_stream_contract.json"
    contract = read_json(contract_path)

    layer = add_design_columns(pd.read_csv(analysis_dir / "refresh_layer_event_metrics.csv"))
    unit_event = add_design_columns(pd.read_csv(analysis_dir / "refresh_unit_event_summary.csv"))
    outcomes = add_design_columns(pd.read_csv(analysis_dir / "refresh_unit_outcomes.csv"))
    joined = add_design_columns(pd.read_csv(analysis_dir / "refresh_unit_event_joined.csv"))
    slices = add_design_columns(pd.read_csv(analysis_dir / "refresh_validation_slices.csv"))

    verification = verify_handoff(
        run_dir, formal_manifest, layer, unit_event, outcomes, joined, slices
    )
    gate = finite_float(contract["hard_gates"]["runtime_resolvent_relative_residual_max"])

    failures = layer.loc[
        layer["runtime_resolvent_relative_residual"].astype(float) > gate
    ].copy()
    failure_columns = [
        "origin",
        "stage",
        "source_method",
        "data_replica",
        "event_id",
        "layer_index",
        "runtime_resolvent_relative_residual",
        "condition_proxy_before",
        "condition_proxy_after",
        "relative_a_fro_change",
        "relative_runtime_inverse_fro_change",
        "runtime_inverse_backward_residual_before",
        "runtime_inverse_backward_residual_after",
        "probe_bank_condition_relative_disagreement",
    ]
    failures = failures[failure_columns].sort_values(
        "runtime_resolvent_relative_residual", ascending=False
    )

    resolvent_strata = build_resolvent_strata(layer, gate)
    matrix_loss = build_matrix_loss_associations(joined)
    numerical_associations = build_layer_diagnostic_associations(layer)
    origin_event = build_origin_event_summary(joined)

    output_paths = {
        "resolvent_threshold_failures.csv": failures,
        "resolvent_by_stage_method_event_layer.csv": resolvent_strata,
        "matrix_loss_associations.csv": matrix_loss,
        "numerical_residual_associations.csv": numerical_associations,
        "origin_event_summary.csv": origin_event,
    }
    for name, frame in output_paths.items():
        write_csv(output_dir / name, frame)
    report_name = "MDP04_FINAL_LOCAL_AUDIT.md"
    (output_dir / report_name).write_text(
        build_report(
            formal_manifest,
            verification,
            failures,
            layer,
            slices,
            matrix_loss,
            gate,
        ),
        encoding="utf-8",
    )

    residual = layer["runtime_resolvent_relative_residual"].astype(float)
    layer3 = layer.loc[layer["layer_index"].eq(3), "runtime_resolvent_relative_residual"].astype(float)
    non_layer3 = layer.loc[layer["layer_index"].ne(3), "runtime_resolvent_relative_residual"].astype(float)
    late_layer3 = layer.loc[
        layer["stage"].eq("late") & layer["layer_index"].eq(3),
        "runtime_resolvent_relative_residual",
    ].astype(float)

    slice_residual = slices["float64_slice_resolvent_relative_residual"].astype(float)
    slice_eligible = slices["paper_empirical_claim_eligible"].astype(str).str.lower().eq("true")
    lineage_counts = formal_manifest["unit_contract_lineage_counts"]

    manifest: dict[str, Any] = {
        "schema_version": "mdp04_local_final_audit_v1",
        "script_version": SCRIPT_VERSION,
        "analyzer_source_sha256": sha256(Path(__file__).resolve()),
        "source_run_id": run_dir.name,
        "source_run_status": read_json(run_dir / "status.json")["status"],
        "formal_manifest_passed": formal_manifest["passed"],
        "computation_complete": True,
        "formal_adjudication": "numeric_gate_failed",
        "matrix_evidence_status": "descriptive_partial_non_claim_eligible",
        "handoff_verification": verification,
        "provenance_diagnostics": {
            "worker_log_hash_mismatch_units": int(
                sum(
                    not row["worker_log_hash_match_diagnostic"]
                    for row in verification["unit_checks"]
                )
            ),
            "scientific_artifact_hash_mismatch_units": int(
                sum(
                    not row["scientific_artifact_hashes"]
                    for row in verification["unit_checks"]
                )
            ),
            "worker_log_role": "diagnostic_only_not_a_scientific_input",
        },
        "coverage": {
            "selected_units": 12,
            "unit_event_rows": len(unit_event),
            "layer_event_rows": len(layer),
            "validation_slice_rows": len(slices),
            "lineage_counts": lineage_counts,
        },
        "formal_failure": {
            "failed_gate": "runtime_resolvent_relative_residual_max",
            "threshold": gate,
            "observed_max": float(residual.max()),
            "above_gate_rows": len(failures),
            "total_rows": len(layer),
            "above_gate_fraction": float(len(failures) / len(layer)),
            "all_failures_stage": sorted(failures["stage"].unique().tolist()),
            "all_failures_layer": sorted(int(value) for value in failures["layer_index"].unique()),
        },
        "resolvent_distributions": {
            "all_432_nested_rows": quantiles(residual),
            "layer3_24_nested_rows": quantiles(layer3),
            "non_layer3_408_nested_rows": quantiles(non_layer3),
            "late_layer3_12_nested_rows": quantiles(late_layer3),
        },
        "validation_slices": {
            "count": len(slices),
            "max_float64_resolvent_relative_residual": float(slice_residual.max()),
            "all_marked_non_claim_eligible": bool((~slice_eligible).all()),
        },
        "claim_boundaries": {
            "layers_are_nested_not_independent": True,
            "associations_are_descriptive_only": True,
            "formal_threshold_was_not_changed": True,
            "experiment37_loss_level_causal_result_is_unchanged": True,
            "long_horizon_optimizer_ranking_inference_forbidden": True,
        },
        "source_artifacts": {
            "formal_stream_manifest.json": sha256(analysis_dir / "formal_stream_manifest.json"),
            "refresh_stream_contract.json": sha256(contract_path),
            **{
                name: sha256(analysis_dir / name)
                for name in sorted(formal_manifest["artifacts"])
            },
        },
    }
    if args.handoff_zip is not None:
        zip_path = args.handoff_zip.resolve()
        manifest["handoff_zip"] = {
            "name": zip_path.name,
            "bytes": zip_path.stat().st_size,
            "sha256": sha256(zip_path),
        }

    manifest["derived_artifacts"] = {
        name: sha256(output_dir / name)
        for name in sorted([*output_paths, report_name])
    }
    write_json(output_dir / "mdp04_local_analysis_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
