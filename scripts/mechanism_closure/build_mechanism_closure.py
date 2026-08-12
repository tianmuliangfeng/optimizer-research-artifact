#!/usr/bin/env python3
"""Build and validate the immutable local mechanism-closure package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULTS_ROOT = Path(os.environ.get("SNM_RESULTS_ROOT", REPO_ROOT / "runs"))
CONTRACT_PATH = HERE / "closure_contract.json"
MANIFEST_NAME = "closure_manifest.json"
WORKBOOK_NAME = "mechanism_closure_workbook.xlsx"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    if not rows and not fields:
        raise ValueError(f"cannot infer fields for empty table: {path}")
    fieldnames = list(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for index in range(cursor, end):
            ranks[ordered[index][0]] = average
        cursor = end
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    require(len(x) == len(y) and len(x) >= 2, "pearson input length")
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(average_ranks(x), average_ranks(y))


def center_by_group(values: Sequence[float], groups: Sequence[str]) -> list[float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        buckets[group].append(value)
    means = {group: statistics.fmean(bucket) for group, bucket in buckets.items()}
    return [value - means[group] for value, group in zip(values, groups)]


def eta_squared(values: Sequence[float], groups: Sequence[str]) -> float:
    grand = statistics.fmean(values)
    buckets: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        buckets[group].append(value)
    between = sum(len(bucket) * (statistics.fmean(bucket) - grand) ** 2 for bucket in buckets.values())
    total = sum((value - grand) ** 2 for value in values)
    return between / total if total else 0.0


def median(values: Iterable[float]) -> float:
    return float(statistics.median(list(values)))


def zip_read(archive: zipfile.ZipFile, prefix: str, relative: str) -> bytes:
    return archive.read(prefix + relative)


def zip_json(archive: zipfile.ZipFile, prefix: str, relative: str) -> Any:
    return json.loads(zip_read(archive, prefix, relative).decode("utf-8"))


def zip_jsonl(archive: zipfile.ZipFile, prefix: str, relative: str) -> list[dict[str, Any]]:
    text = zip_read(archive, prefix, relative).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def resolve_inputs(contract: dict[str, Any], workspace_root: Path) -> dict[str, Path]:
    legacy_prefix = Path("experiment_csv") / "selective-newton-muon"
    resolved: dict[str, Path] = {}
    for item in contract["local_inputs"]:
        relative = Path(item["path"])
        try:
            relative = relative.relative_to(legacy_prefix)
        except ValueError:
            pass
        resolved[item["id"]] = workspace_root / relative
    return resolved


def audit_inputs(contract: dict[str, Any], workspace_root: Path, geo_zip: Path) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    rows: list[dict[str, Any]] = []
    resolved = resolve_inputs(contract, workspace_root)
    for item in contract["local_inputs"]:
        path = resolved[item["id"]]
        exists = path.is_file()
        observed = sha256_file(path) if exists else ""
        passed = exists and observed == item["sha256"]
        rows.append(
            {
                "input_id": item["id"],
                "kind": "local_file",
                "location": str(path),
                "bytes": path.stat().st_size if exists else "",
                "expected_sha256": item["sha256"],
                "observed_sha256": observed,
                "passed": passed,
                "note": "exact frozen input",
            }
        )
        require(passed, f"input audit failed: {item['id']} ({path})")

    geo = contract["geo01b"]
    exists = geo_zip.is_file()
    observed = sha256_file(geo_zip) if exists else ""
    size = geo_zip.stat().st_size if exists else 0
    passed = exists and size == geo["bytes"] and observed == geo["sha256"]
    rows.append(
        {
            "input_id": "geo01b_zip",
            "kind": "source_zip",
            "location": str(geo_zip),
            "bytes": size if exists else "",
            "expected_sha256": geo["sha256"],
            "observed_sha256": observed,
            "passed": passed,
            "note": "exact accepted GEO-01B handoff",
        }
    )
    require(passed, f"GEO-01B ZIP audit failed: {geo_zip}")

    geo01a = contract["geo01a_lineage"]
    rows.append(
        {
            "input_id": "geo01a_lineage",
            "kind": "accepted_lineage_only",
            "location": "source ZIP not present locally on closure date",
            "bytes": "",
            "expected_sha256": geo01a["source_zip_sha256"],
            "observed_sha256": "",
            "passed": True,
            "note": "engineering pilot only; excluded from scientific closure and not reverified",
        }
    )
    return rows, resolved


def verify_geo_handoff(
    archive: zipfile.ZipFile, prefix: str, expected_contract_sha: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int]:
    handoff = zip_json(archive, prefix, "handoff_manifest.json")
    verified = 0
    for item in handoff["files"]:
        data = zip_read(archive, prefix, item["path"])
        require(len(data) == int(item["bytes"]), f"GEO-01B byte mismatch: {item['path']}")
        require(sha256_bytes(data) == item["sha256"], f"GEO-01B hash mismatch: {item['path']}")
        verified += 1
    manifest = zip_json(archive, prefix, "analysis/analysis_manifest.json")
    status = zip_json(archive, prefix, "status.json")
    outcomes = zip_jsonl(archive, prefix, "combined/outcome_rows.jsonl")
    geometry = zip_jsonl(archive, prefix, "combined/geometry_rows.jsonl")
    require(manifest["integrity_passed"] is True, "GEO-01B integrity not passed")
    require(all(manifest["checks"].values()), "GEO-01B analysis checks not all true")
    require(manifest["contract_sha256"] == expected_contract_sha, "GEO-01B contract mismatch")
    require(manifest["phase"] == "discovery", "GEO-01B phase mismatch")
    require(manifest["claim_eligible"] is False, "GEO-01B must remain non-claim-eligible")
    require(manifest["confirmation_candidate"] is False, "GEO-01B candidate gate unexpectedly passed")
    require(manifest["confirmation_authorized"] is False, "GEO-01B confirmation unexpectedly authorized")
    require(status["scientific_result"] == "directional_geometry_not_supported", "GEO-01B status mismatch")
    return manifest, status, outcomes, geometry, verified


PREDICTORS = {
    "norm_only": "norm_only_predictor",
    "first_order": "first_order_predictor",
    "full_taylor": "full_taylor_predictor",
}


def derive_geo(
    outcomes: list[dict[str, Any]], geometry: list[dict[str, Any]], manifest: dict[str, Any], contract: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    geo = contract["geo01b"]
    require(len(outcomes) == geo["expected_outcome_rows"], "GEO-01B outcome row count")
    require(len(geometry) == geo["expected_geometry_rows"], "GEO-01B geometry row count")
    require(all(row.get("all_values_finite") is True for row in outcomes + geometry), "non-finite GEO-01B row")
    require(len({(row["origin"], row["data_replica"], row["event_id"]) for row in outcomes}) == len(outcomes), "duplicate outcome rows")
    require(len({(row["origin"], row["data_replica"]) for row in outcomes}) == geo["expected_units"], "GEO-01B unit count")

    unit_rows: list[dict[str, Any]] = []
    for row in sorted(outcomes, key=lambda x: (x["event_id"], x["origin"], x["data_replica"])):
        unit_rows.append(
            {
                "event_id": row["event_id"],
                "origin": row["origin"],
                "data_replica": row["data_replica"],
                "endpoint_step": row["endpoint_step"],
                "endpoint_normalized_loss_harm": row["endpoint_normalized_loss_harm"],
                "endpoint_raw_loss_harm": row["endpoint_raw_loss_harm"],
                "norm_only_predictor": row["norm_only_predictor"],
                "first_order_predictor": row["first_order_predictor"],
                "full_taylor_predictor": row["full_taylor_predictor"],
                "local_exact_delta_loss": row["local_exact_delta_loss"],
                "local_first_relative_error": row["local_first_relative_error"],
                "local_taylor_relative_error": row["local_taylor_relative_error"],
                "local_first_sign_match": row["local_first_sign_match"],
                "local_taylor_sign_match": row["local_taylor_sign_match"],
            }
        )

    event_rows: list[dict[str, Any]] = []
    predictor_rows: list[dict[str, Any]] = []
    event_payload: dict[str, Any] = {}
    for event_id in sorted({row["event_id"] for row in outcomes}):
        rows = [row for row in outcomes if row["event_id"] == event_id]
        harms = [float(row["endpoint_normalized_loss_harm"]) for row in rows]
        raw_harms = [float(row["endpoint_raw_loss_harm"]) for row in rows]
        origins = [str(row["origin"]) for row in rows]
        first_errors = [float(row["local_first_relative_error"]) for row in rows]
        taylor_errors = [float(row["local_taylor_relative_error"]) for row in rows]
        error_reduction = 1.0 - median(taylor_errors) / median(first_errors)
        result = manifest["event_results"][event_id]
        event_row = {
            "event_id": event_id,
            "unit_count": len(rows),
            "positive_units": sum(value > 0 for value in harms),
            "normalized_harm_mean": statistics.fmean(harms),
            "normalized_harm_median": median(harms),
            "normalized_harm_min": min(harms),
            "normalized_harm_max": max(harms),
            "raw_harm_mean": statistics.fmean(raw_harms),
            "raw_harm_median": median(raw_harms),
            "raw_harm_min": min(raw_harms),
            "raw_harm_max": max(raw_harms),
            "first_order_median_relative_error": median(first_errors),
            "full_taylor_median_relative_error": median(taylor_errors),
            "full_taylor_max_relative_error": max(taylor_errors),
            "curvature_local_error_reduction_fraction": error_reduction,
            "origin_eta_squared_harm": eta_squared(harms, origins),
            "directional_geometry_supported": result["directional_geometry_supported"],
            "curvature_increment_supported": result["curvature_increment_supported"],
        }
        require(close(event_row["normalized_harm_mean"], float(result["endpoint_harm_mean"])), f"GEO mean mismatch {event_id}")
        require(event_row["positive_units"] == result["endpoint_harm_positive_count"], f"GEO sign mismatch {event_id}")
        closure = result["local_closure"]
        require(close(event_row["first_order_median_relative_error"], closure["first_order_median_relative_error"]), f"GEO first-order mismatch {event_id}")
        require(close(event_row["full_taylor_median_relative_error"], closure["full_taylor_median_relative_error"]), f"GEO Taylor mismatch {event_id}")
        require(close(event_row["curvature_local_error_reduction_fraction"], closure["curvature_local_error_reduction_fraction"]), f"GEO reduction mismatch {event_id}")
        event_rows.append(event_row)

        event_payload[event_id] = {"harms": harms, "origins": origins}
        for predictor_id, field in PREDICTORS.items():
            values = [float(row[field]) for row in rows]
            pooled = spearman(values, harms)
            centered = spearman(center_by_group(values, origins), center_by_group(harms, origins))
            loo: dict[str, float] = {}
            for origin in sorted(set(origins)):
                keep = [index for index, value in enumerate(origins) if value != origin]
                loo[origin] = spearman([values[i] for i in keep], [harms[i] for i in keep])
            sign_accuracy = sum((a >= 0) == (b >= 0) for a, b in zip(values, harms)) / len(values)
            accepted = result["predictors"][predictor_id]
            require(close(pooled, accepted["pooled_spearman"]), f"GEO pooled correlation mismatch {event_id}/{predictor_id}")
            require(close(centered, accepted["origin_centered_spearman"]), f"GEO centered correlation mismatch {event_id}/{predictor_id}")
            predictor_rows.append(
                {
                    "source": "GEO-01B",
                    "event_id": event_id,
                    "predictor": predictor_id,
                    "correlation_family": "spearman",
                    "pooled_correlation": pooled,
                    "origin_centered_correlation": centered,
                    "minimum_leave_one_origin_out": min(loo.values()),
                    "sign_accuracy": sign_accuracy,
                    "candidate_gate_passed": False,
                    "claim_eligible": False,
                    "interpretation": "discovery predictor did not satisfy origin-independent confirmation gate",
                }
            )

    prod = event_payload["production_refresh_32"]
    delayed = event_payload["delayed_refresh_64"]
    keys_prod = {(row["origin"], row["data_replica"]): float(row["endpoint_normalized_loss_harm"]) for row in outcomes if row["event_id"] == "production_refresh_32"}
    keys_delayed = {(row["origin"], row["data_replica"]): float(row["endpoint_normalized_loss_harm"]) for row in outcomes if row["event_id"] == "delayed_refresh_64"}
    shared = sorted(keys_prod)
    cross_event = pearson([keys_prod[key] for key in shared], [keys_delayed[key] for key in shared])
    diagnostics = {
        "independent_recomputation_passed": True,
        "unit_count": geo["expected_units"],
        "outcome_rows": len(outcomes),
        "geometry_rows": len(geometry),
        "cross_event_pearson_harm": cross_event,
        "production_origin_eta_squared": next(row["origin_eta_squared_harm"] for row in event_rows if row["event_id"] == "production_refresh_32"),
        "delayed_origin_eta_squared": next(row["origin_eta_squared_harm"] for row in event_rows if row["event_id"] == "delayed_refresh_64"),
    }
    return unit_rows, event_rows, predictor_rows, diagnostics


def build_mdp05_tables(acceptance: dict[str, Any], primary_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(acceptance["integrity"]["passed"] is True, "MDP-05 acceptance integrity")
    require(acceptance["scientific_result"] == "partial_or_null", "MDP-05 result mismatch")
    require(acceptance["claim_success"] is False, "MDP-05 claim status mismatch")
    harm_rows = []
    for event_id, values in acceptance["direct_loss_effect"].items():
        require(values["positive_units"] == values["total_units"] == 12, f"MDP-05 harm sign mismatch {event_id}")
        harm_rows.append(
            {
                "source": "MDP-05",
                "event_id": event_id,
                "unit_count": values["total_units"],
                "positive_units": values["positive_units"],
                "normalized_harm_mean": values["mean_normalized_harm"],
                "normalized_harm_median": values["median_normalized_harm"],
                "normalized_harm_min": values["minimum"],
                "normalized_harm_max": values["maximum"],
                "claim_eligible": True,
                "interpretation": "replicated direct refresh harm; quantitative mediator claim not implied",
            }
        )
    tests = []
    for row in primary_rows:
        require(row["passed"].lower() == "false", "unexpected MDP-05 primary pass")
        tests.append(
            {
                "source": "MDP-05",
                "event_id": row["event_id"],
                "predictor": row["mediator"],
                "correlation_family": "pooled_spearman / centered_pearson",
                "pooled_correlation": float(row["spearman_rho"]),
                "origin_centered_correlation": float(row["within_origin_centered_pearson_r"]),
                "minimum_leave_one_origin_out": float(row["leave_one_origin_out_spearman_min"]),
                "one_sided_exact_p": float(row["one_sided_exact_p"]),
                "holm_adjusted_p": float(row["holm_adjusted_p"]),
                "candidate_gate_passed": False,
                "claim_eligible": True,
                "interpretation": "confirmatory quantitative mediation test did not pass multiplicity-controlled gate",
            }
        )
    require(len(tests) == 4, "MDP-05 primary test count")
    return harm_rows, tests


def evidence_ledger(unified_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in unified_rows:
        rows.append(
            {
                "order": int(row["stage_order"]),
                "study": row["stage"],
                "evidence_level": row["evidence_level"],
                "status": row["status"],
                "result": row["result"],
                "claim_boundary": row["claim_boundary"],
                "paper_role": "inherited from accepted unified synthesis",
            }
        )
    additions = [
        (12, "MDP-04 streamed replay", "diagnostic", "numerical_gate_failed", "12/12 units and descriptive signal recovered, but 6/432 resolvent rows exceeded the frozen 0.01 gate", "not formal paper evidence", "engineering audit / limitations"),
        (13, "MDP-05 direct refresh effect", "confirmatory", "supported", "positive normalized endpoint harm in 12/12 units at both production and delayed events", "causal only inside the frozen matched replay tree", "appendix evidence; concise main-text support"),
        (14, "MDP-05 scalar update-shock mediation", "confirmatory", "partial_or_null", "0/4 multiplicity-controlled primary tests passed", "no origin-independent quantitative mediator claim", "appendix and limitations"),
        (15, "GEO-01A curvature pilot", "engineering", "pilot_passed", "exact HVP/Taylor implementation and memory feasibility passed", "engineering only; local source unavailable at closure and not reverified", "provenance only"),
        (16, "GEO-01B immediate local geometry", "discovery", "descriptively_supported", "full Taylor median relative error 0.010%/0.028%; curvature reduces local approximation error by >99.6%", "immediate line loss only; non-claim-eligible discovery", "appendix descriptive evidence"),
        (17, "GEO-01B short-horizon predictor", "discovery", "not_supported", "origin-centered directional-geometry correlations failed the discovery gate and did not beat norm consistently", "no GEO-01C and no predictor claim", "limitations / negative result"),
        (18, "Mechanism closure decision", "adjudication", "closed", "refresh harm retained; simple origin-independent scalar/local predictor rejected; mechanism line frozen", "a new trajectory/state mechanism would be a new project, not a rescue", "paper writing boundary"),
    ]
    for order, study, level, status, result, boundary, role in additions:
        rows.append({"order": order, "study": study, "evidence_level": level, "status": status, "result": result, "claim_boundary": boundary, "paper_role": role})
    return rows


def claim_boundary_rows() -> list[dict[str, Any]]:
    return [
        {"claim_id": "MC-C01", "status": "allowed", "paper_location": "main text + appendix", "wording": "Under the frozen matched LLaMA-1B replay tree, scheduled down-projection refresh produces a reproducible short-horizon held-out loss impulse.", "evidence": "MECH-09R; MDP-05 12/12 positive at both events; GEO-01B 12/12 positive at both events", "prohibited_extension": "universal optimizer or architecture claim"},
        {"claim_id": "MC-C02", "status": "allowed_descriptive", "paper_location": "appendix", "wording": "For the tested refresh direction, a full local Taylor expansion accurately reconstructs the immediate counterfactual line-loss change.", "evidence": "GEO-01B full-Taylor median relative error 0.010% production and 0.028% delayed", "prohibited_extension": "prediction of multi-step training harm"},
        {"claim_id": "MC-C03", "status": "required_negative", "paper_location": "limitations + appendix", "wording": "Neither the preconditioned shock scalars nor the local directional-geometry predictors were confirmed as origin-independent quantitative mediators of short-horizon harm.", "evidence": "MDP-05 0/4 primary gates; GEO-01B candidate=false", "prohibited_extension": "claiming the mechanism is fully explained"},
        {"claim_id": "MC-C04", "status": "allowed_descriptive", "paper_location": "appendix", "wording": "Checkpoint origin accounts for most observed between-unit variation in refresh harm in these replay samples.", "evidence": "MDP-05 ~96%; GEO-01B 95.9% production and 98.0% delayed", "prohibited_extension": "causal attribution of origin components"},
        {"claim_id": "MC-C05", "status": "prohibited", "paper_location": "none", "wording": "Local curvature explains or predicts the longer-horizon loss penalty.", "evidence": "GEO-01B origin-centered and incremental gates failed", "prohibited_extension": "do not state or imply"},
        {"claim_id": "MC-C06", "status": "prohibited", "paper_location": "none", "wording": "The discovered geometry supports an automatic layer selector or rescue policy.", "evidence": "No rescue gate passed; GEO-01C not authorized", "prohibited_extension": "do not state or imply"},
    ]


def make_figures(output_dir: Path, harm_rows: list[dict[str, Any]], geo_predictor_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib and numpy are required for the closure figures") from exc

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10, "figure.dpi": 160})
    events = ["production_refresh_32", "delayed_refresh_64"]
    sources = ["MDP-05", "GEO-01B"]
    labels = ["Production refresh", "Delayed refresh"]
    x = np.arange(len(events))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = ["#315A8C", "#D97732"]
    for index, source in enumerate(sources):
        rows = [next(row for row in harm_rows if row["source"] == source and row["event_id"] == event) for event in events]
        means = np.array([row["normalized_harm_mean"] for row in rows])
        lower = means - np.array([row["normalized_harm_min"] for row in rows])
        upper = np.array([row["normalized_harm_max"] for row in rows]) - means
        positions = x + (index - 0.5) * width
        ax.bar(positions, means, width, label=source, color=colors[index], alpha=0.9)
        ax.errorbar(positions, means, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#222222", capsize=4, linewidth=1)
        for px, value in zip(positions, means):
            ax.text(px, value + 0.00028, "12/12 +", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean normalized endpoint loss harm")
    ax.set_title("Refresh harm replicates across independent mechanism packages")
    ax.legend(frameon=False)
    ax.set_ylim(0, 0.0075)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"fig_refresh_harm_replication.{suffix}", bbox_inches="tight")
    plt.close(fig)

    predictors = ["norm_only", "first_order", "full_taylor"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for axis, event, title in zip(axes, events, labels):
        rows = [next(row for row in geo_predictor_rows if row["event_id"] == event and row["predictor"] == predictor) for predictor in predictors]
        positions = np.arange(len(predictors))
        pooled = [row["pooled_correlation"] for row in rows]
        centered = [row["origin_centered_correlation"] for row in rows]
        axis.bar(positions - 0.18, pooled, 0.36, label="Pooled", color="#6B8E23")
        axis.bar(positions + 0.18, centered, 0.36, label="Origin-centered", color="#A24B4B")
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_xticks(positions, ["Norm", "1st order", "Full Taylor"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.set_ylim(-0.15, 1.0)
    axes[0].set_ylabel("Spearman correlation with endpoint harm")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("GEO-01B pooled associations do not transfer to within-origin prediction", y=1.01)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"fig_predictor_generalization.{suffix}", bbox_inches="tight")
    plt.close(fig)


def copy_snapshot(output_dir: Path, resolved: dict[str, Path], archive: zipfile.ZipFile, prefix: str) -> None:
    snapshot = output_dir / "source_snapshot"
    local_dir = snapshot / "local_inputs"
    geo_dir = snapshot / "geo01b"
    script_dir = snapshot / "mechanism_closure_scripts"
    local_dir.mkdir(parents=True)
    geo_dir.mkdir(parents=True)
    script_dir.mkdir(parents=True)
    for input_id, source in resolved.items():
        shutil.copy2(source, local_dir / f"{input_id}__{source.name}")
    selected = [
        "status.json",
        "run_identity.json",
        "runtime_preflight.json",
        "handoff_manifest.json",
        "analysis/analysis_manifest.json",
        "analysis/event_summary.csv",
        "combined/outcome_rows.jsonl",
        "combined/geometry_rows.jsonl",
    ]
    for relative in selected:
        target = geo_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zip_read(archive, prefix, relative))
    for source in sorted(HERE.iterdir()):
        if source.is_file() and source.suffix in {".py", ".json", ".md"}:
            shutil.copy2(source, script_dir / source.name)


def sync_script_snapshot(output_dir: Path) -> None:
    script_dir = output_dir / "source_snapshot" / "mechanism_closure_scripts"
    require(script_dir.is_dir(), "missing mechanism-closure script snapshot directory")
    for source in sorted(HERE.iterdir()):
        if source.is_file() and source.suffix in {".py", ".json", ".md", ".mjs"}:
            shutil.copy2(source, script_dir / source.name)


def make_report(
    output_dir: Path,
    mdp_harm: list[dict[str, Any]],
    geo_events: list[dict[str, Any]],
    geo_diag: dict[str, Any],
    handoff_files: int,
) -> None:
    by_key = {(row.get("source", "GEO-01B"), row["event_id"]): row for row in mdp_harm + [{**row, "source": "GEO-01B"} for row in geo_events]}
    mprod = by_key[("MDP-05", "production_refresh_32")]
    mdelay = by_key[("MDP-05", "delayed_refresh_64")]
    gprod = by_key[("GEO-01B", "production_refresh_32")]
    gdelay = by_key[("GEO-01B", "delayed_refresh_64")]
    report = f"""# 机制收尾报告（2026-08-05）

## 最终裁决

机制实验线在此收尾，不启动 GEO-01C，也不再以“补救”名义增加 GPU 机制实验。当前证据支持一个清晰但克制的结论：**在冻结的 LLaMA-1B 匹配回放树内，scheduled down-projection refresh 会稳定地产生短时 held-out loss impulse；但我们测试的单一 shock 标量和局部方向几何，均没有成为跨 checkpoint origin 稳健的定量解释器。**

这不是“机制实验没用”。正结果回答了 *where/when the harm is injected*；负结果回答了 *which tempting scalar explanations should not be claimed*。对论文而言，这比把探索性 pooled correlation 包装成机制更可靠。

## 证据完整性

- GEO-01B ZIP、{len(load_json(CONTRACT_PATH)['local_inputs'])} 个本地冻结输入均通过精确 SHA-256 检查。
- GEO-01B handoff 中 {handoff_files} 个文件全部通过字节数与 SHA-256 检查；12/12 discovery units、24 outcome rows、96 geometry rows 完整。
- 关键 GEO-01B 统计由本地脚本从 JSONL 独立重算，并逐项与远程 analysis manifest 对照通过。
- GEO-01A 只承担工程可行性角色；其 ZIP 当前不在本地，因此本包明确记录为“accepted lineage only / not reverified”，不用于科学收尾结论。
- MDP-04 的 12/12 单元已恢复，但原冻结数值 gate 失败（6/432 residual rows > 0.01），因此只作为诊断与复现教训，不作为正式证据。

## 稳健的正发现：refresh harm 重复出现

| 数据包 | 事件 | 正向单元 | normalized harm mean | median | range |
|---|---|---:|---:|---:|---:|
| MDP-05 | production refresh | 12/12 | {mprod['normalized_harm_mean']:.6f} | {mprod['normalized_harm_median']:.6f} | [{mprod['normalized_harm_min']:.6f}, {mprod['normalized_harm_max']:.6f}] |
| MDP-05 | delayed refresh | 12/12 | {mdelay['normalized_harm_mean']:.6f} | {mdelay['normalized_harm_median']:.6f} | [{mdelay['normalized_harm_min']:.6f}, {mdelay['normalized_harm_max']:.6f}] |
| GEO-01B | production refresh | 12/12 | {gprod['normalized_harm_mean']:.6f} | {gprod['normalized_harm_median']:.6f} | [{gprod['normalized_harm_min']:.6f}, {gprod['normalized_harm_max']:.6f}] |
| GEO-01B | delayed refresh | 12/12 | {gdelay['normalized_harm_mean']:.6f} | {gdelay['normalized_harm_median']:.6f} | [{gdelay['normalized_harm_min']:.6f}, {gdelay['normalized_harm_max']:.6f}] |

两套独立包在两个事件上都得到 12/12 同号，而且 mean 非常接近。这使“refresh 本身在冻结树内造成短时 loss harm”成为可以保留的因果证据；它不等价于对所有模型、训练阶段或优化器的普遍定律。

## 局部曲率究竟告诉了我们什么

GEO-01B 的 full Taylor 对即时 counterfactual line loss 的拟合非常准：production 的中位相对误差为 {gprod['full_taylor_median_relative_error']:.6%}，delayed 为 {gdelay['full_taylor_median_relative_error']:.6%}；相对一阶近似，加入曲率后本地误差分别下降 {gprod['curvature_local_error_reduction_fraction']:.3%} 与 {gdelay['curvature_local_error_reduction_fraction']:.3%}。

但这只说明 Hessian-vector product 与 Taylor 分解在**即时、固定 batch、给定方向**上数值闭合。它没有转化成跨 origin 的 16-step endpoint harm 预测：GEO-01B 的 confirmation candidate=false，curvature increment 也不满足门槛。因此“局部曲率能预测后续训练伤害”必须列为禁止表述。

## 为什么 pooled correlation 不能升级为机制

GEO-01B production full-Taylor 的 pooled Spearman 很高（0.853），但 origin-centered Spearman 只有 0.168；delayed 分别为 0.434 与 0.063。相反，简单 norm 在 production 的 centered Spearman 是 0.720，说明更复杂的 Taylor 标量没有在关键的 within-origin 检验中稳定胜出。

checkpoint origin 对 endpoint harm 的描述性解释比例为 production {geo_diag['production_origin_eta_squared']:.1%}、delayed {geo_diag['delayed_origin_eta_squared']:.1%}；两个事件的 unit-level harm 相关为 r={geo_diag['cross_event_pearson_harm']:.3f}。MDP-05 也得到约 96% 的 origin share。最合理的解释是：当前 pooled 排序主要携带 checkpoint/stage/method 状态，而不是一个可转移的单标量局部机制。

## 论文使用边界

正文可简洁表述 refresh loss impulse 的冻结树内因果证据；详细数据放 appendix。MDP-05 的 null mediation 与 GEO-01B 的 failed origin-independent gate 应主动进入 appendix/limitations，显示我们对机制主张做了强检验。

不得声称：普适 refresh 定律；局部曲率预测多步 harm；已经得到自动 layer selector；GEO-01B 是 confirmatory evidence；或机制已被完全解释。逐条措辞见 `claim_boundary.csv`。

## 后续动作

1. 冻结本目录，论文机制段落只从本包取数。
2. 不运行 GEO-01C；新的 trajectory/state-dependent 解释只能作为未来独立项目。
3. 下一项单独评估 LLaMA-1B 10B-token 实验的科学增益、计算成本与对投稿风险的净贡献；该决定不属于本机制包。
"""
    (output_dir / "MECHANISM_CLOSURE_REPORT.md").write_text(report, encoding="utf-8")


def artifact_inventory(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != MANIFEST_NAME:
            rows.append({"path": path.relative_to(output_dir).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def write_manifest(output_dir: Path, contract: dict[str, Any], *, finalized: bool) -> dict[str, Any]:
    workbook = output_dir / WORKBOOK_NAME
    if finalized:
        require(workbook.is_file(), f"missing workbook: {workbook}")
    artifacts = artifact_inventory(output_dir)
    manifest = {
        "schema_version": "mechanism_closure_manifest_v1",
        "package_id": contract["package_id"],
        "decision_date": contract["decision_date"],
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "status": "completed" if finalized else "awaiting_workbook",
        "passed": finalized,
        "mechanism_line_closed": True,
        "geo01c_authorized": False,
        "new_gpu_mechanism_work_authorized": False,
        "llama1b_10b_decision_deferred": True,
        "scientific_adjudication": {
            "refresh_harm": "supported_within_frozen_matched_replay_tree",
            "immediate_local_taylor_closure": "descriptively_supported",
            "origin_independent_scalar_mediation": "not_confirmed",
            "origin_independent_directional_geometry": "not_supported",
            "automatic_layer_selector": "not_supported",
        },
        "workbook_required": True,
        "workbook_present": workbook.is_file(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    dump_json(output_dir / MANIFEST_NAME, manifest)
    return manifest


def validate_package(output_dir: Path, require_final: bool = True) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_NAME
    require(manifest_path.is_file(), f"missing {MANIFEST_NAME}")
    manifest = load_json(manifest_path)
    require(manifest["mechanism_line_closed"] is True, "closure flag")
    require(manifest["geo01c_authorized"] is False, "GEO-01C flag")
    require(manifest["contract_sha256"] == sha256_file(CONTRACT_PATH), "contract hash")
    if require_final:
        require(manifest["passed"] is True and manifest["status"] == "completed", "package not finalized")
        require(manifest["workbook_present"] is True, "workbook flag")
    for item in manifest["artifacts"]:
        path = output_dir / item["path"]
        require(path.is_file(), f"missing artifact: {item['path']}")
        require(path.stat().st_size == item["bytes"], f"artifact bytes: {item['path']}")
        require(sha256_file(path) == item["sha256"], f"artifact hash: {item['path']}")
    required = [
        "MECHANISM_CLOSURE_REPORT.md",
        "mechanism_evidence_ledger.csv",
        "refresh_harm_replication.csv",
        "predictor_gate_summary.csv",
        "geo01b_event_summary.csv",
        "geo01b_unit_outcomes.csv",
        "claim_boundary.csv",
        "fig_refresh_harm_replication.png",
        "fig_predictor_generalization.png",
    ]
    if require_final:
        required.append(WORKBOOK_NAME)
    require(all((output_dir / name).is_file() for name in required), "required artifact missing")
    return {"passed": True, "artifact_count": len(manifest["artifacts"]), "status": manifest["status"]}


def build(output_dir: Path, geo_zip: Path, workspace_root: Path) -> None:
    require(not output_dir.exists(), f"immutable output already exists: {output_dir}")
    contract = load_json(CONTRACT_PATH)
    input_audit, resolved = audit_inputs(contract, workspace_root, geo_zip)
    output_dir.mkdir(parents=True)
    try:
        with zipfile.ZipFile(geo_zip) as archive:
            prefix = contract["geo01b"]["run_id"] + "/"
            manifest, _, outcomes, geometry, handoff_files = verify_geo_handoff(
                archive, prefix, contract["geo01b"]["contract_sha256"]
            )
            unit_rows, event_rows, geo_predictors, geo_diag = derive_geo(outcomes, geometry, manifest, contract)
            copy_snapshot(output_dir, resolved, archive, prefix)

        acceptance = load_json(resolved["mdp05_acceptance"])
        primary_rows = read_csv(resolved["mdp05_primary_tests"])
        mdp_harm, mdp_tests = build_mdp05_tables(acceptance, primary_rows)
        unified_chain = read_csv(resolved["unified_chain"])

        geo_harm = [
            {
                "source": "GEO-01B",
                "event_id": row["event_id"],
                "unit_count": row["unit_count"],
                "positive_units": row["positive_units"],
                "normalized_harm_mean": row["normalized_harm_mean"],
                "normalized_harm_median": row["normalized_harm_median"],
                "normalized_harm_min": row["normalized_harm_min"],
                "normalized_harm_max": row["normalized_harm_max"],
                "claim_eligible": False,
                "interpretation": "independent discovery replication; supports closure context but is not confirmatory",
            }
            for row in event_rows
        ]
        all_harm = sorted(mdp_harm + geo_harm, key=lambda row: (row["event_id"], row["source"]))
        all_predictors = mdp_tests + geo_predictors

        write_csv(output_dir / "input_audit.csv", input_audit)
        dump_json(output_dir / "input_audit.json", {"passed": all(row["passed"] for row in input_audit), "rows": input_audit, "geo01b_handoff_files_verified": handoff_files})
        write_csv(output_dir / "mechanism_evidence_ledger.csv", evidence_ledger(unified_chain))
        write_csv(output_dir / "refresh_harm_replication.csv", all_harm)
        write_csv(output_dir / "predictor_gate_summary.csv", all_predictors)
        write_csv(output_dir / "geo01b_unit_outcomes.csv", unit_rows)
        write_csv(output_dir / "geo01b_event_summary.csv", event_rows)
        write_csv(output_dir / "claim_boundary.csv", claim_boundary_rows())
        dump_json(output_dir / "geo01b_independent_recomputation.json", geo_diag)
        dump_json(output_dir / "closure_decision.json", contract["decision"] | {"scientific_result": "mechanism_line_closed_with_bounded_positive_and_negative_findings"})
        make_figures(output_dir, all_harm, geo_predictors)
        make_report(output_dir, mdp_harm, event_rows, geo_diag, handoff_files)
        write_manifest(output_dir, contract, finalized=False)
        validate_package(output_dir, require_final=False)
    except Exception:
        marker = output_dir / "BUILD_FAILED.txt"
        marker.write_text("Build failed; this directory must not be treated as an accepted package.\n", encoding="utf-8")
        raise


def check(geo_zip: Path, workspace_root: Path) -> dict[str, Any]:
    contract = load_json(CONTRACT_PATH)
    input_audit, _ = audit_inputs(contract, workspace_root, geo_zip)
    with zipfile.ZipFile(geo_zip) as archive:
        prefix = contract["geo01b"]["run_id"] + "/"
        manifest, _, outcomes, geometry, handoff_files = verify_geo_handoff(archive, prefix, contract["geo01b"]["contract_sha256"])
        _, event_rows, _, diag = derive_geo(outcomes, geometry, manifest, contract)
    return {"passed": True, "input_count": len(input_audit), "handoff_files_verified": handoff_files, "events": len(event_rows), **diag}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "build"):
        sub = subparsers.add_parser(name)
        sub.add_argument(
            "--workspace-root",
            type=Path,
            default=RESULTS_ROOT,
            help="Root containing imported experiment result directories",
        )
        sub.add_argument(
            "--geo01b-zip",
            type=Path,
            default=REPO_ROOT / load_json(CONTRACT_PATH)["geo01b"]["default_zip"],
        )
        if name == "build":
            sub.add_argument("--output-dir", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--output-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--allow-awaiting-workbook", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "check":
        payload = check(args.geo01b_zip.resolve(), args.workspace_root.resolve())
    elif args.command == "build":
        build(args.output_dir.resolve(), args.geo01b_zip.resolve(), args.workspace_root.resolve())
        payload = {"passed": True, "status": "awaiting_workbook", "output_dir": str(args.output_dir.resolve())}
    elif args.command == "finalize":
        contract = load_json(CONTRACT_PATH)
        sync_script_snapshot(args.output_dir.resolve())
        write_manifest(args.output_dir.resolve(), contract, finalized=True)
        payload = validate_package(args.output_dir.resolve(), require_final=True)
    else:
        payload = validate_package(args.output_dir.resolve(), require_final=not args.allow_awaiting_workbook)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
