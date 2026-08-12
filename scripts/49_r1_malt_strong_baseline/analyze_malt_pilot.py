"""Independently audit and select both frozen Experiment-49 pilot methods."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FAMILY = "49_r1_malt_strong_baseline"
PILOT_PROTOCOL = "malt_r1_focused_grid_pilot_v4"
SELECTION_PROTOCOL = "malt_r1_focused_grid_selection_v4"
ANALYSIS_PROTOCOL = "malt_r1_pilot_analysis_v4"
ACCEPTED_MANIFEST_STATUSES = {
    "completed_valid",
    "completed_valid_local_wandb_incomplete",
}
PILOT_SEED = 2026
PILOT_STEPS = 1000
TOKENS_PER_STEP = 512 * 1024
PILOT_TOKENS = PILOT_STEPS * TOKENS_PER_STEP
TIE_MARGIN = 0.002

MALT_LOWER_BOUNDARY_LR = 0.0064
MALT_UPPER_BOUNDARY_LR = 0.0160
MALTER_CENTER_LR = 0.012
MALTER_LOWER_BOUNDARY_LR = 0.007
MALTER_UPPER_BOUNDARY_LR = 0.025

CELL_SPECS: dict[str, dict[str, object]] = {
    "malt_lr0160": {
        "method": "malt",
        "matrix_lr": MALT_UPPER_BOUNDARY_LR,
        "formal_eligible": True,
    },
    "malt_lr0125": {"method": "malt", "matrix_lr": 0.0125, "formal_eligible": True},
    "malt_lr0100": {"method": "malt", "matrix_lr": 0.0100, "formal_eligible": True},
    "malt_lr0090": {"method": "malt", "matrix_lr": 0.0090, "formal_eligible": True},
    "malt_lr0080": {"method": "malt", "matrix_lr": 0.0080, "formal_eligible": True},
    "malt_lr0064": {"method": "malt", "matrix_lr": 0.0064, "formal_eligible": True},
    "malter_eq17_lr007": {
        "method": "malter_eq17",
        "matrix_lr": MALTER_LOWER_BOUNDARY_LR,
        "formal_eligible": True,
    },
    "malter_eq17_lr009": {"method": "malter_eq17", "matrix_lr": 0.009, "formal_eligible": True},
    "malter_eq17_lr012": {"method": "malter_eq17", "matrix_lr": MALTER_CENTER_LR, "formal_eligible": True},
    "malter_eq17_lr015": {"method": "malter_eq17", "matrix_lr": 0.015, "formal_eligible": True},
    "malter_eq17_lr018": {"method": "malter_eq17", "matrix_lr": 0.018, "formal_eligible": True},
    "malter_eq17_lr025": {
        "method": "malter_eq17",
        "matrix_lr": MALTER_UPPER_BOUNDARY_LR,
        "formal_eligible": True,
    },
}

METHOD_SELECTION_RULES: dict[str, dict[str, object]] = {
    "malt": {
        "center_lr": None,
        "prefer_center": False,
        "lower_boundary_lr": MALT_LOWER_BOUNDARY_LR,
        "upper_boundary_lr": MALT_UPPER_BOUNDARY_LR,
    },
    "malter_eq17": {
        "center_lr": MALTER_CENTER_LR,
        "prefer_center": True,
        "lower_boundary_lr": MALTER_LOWER_BOUNDARY_LR,
        "upper_boundary_lr": MALTER_UPPER_BOUNDARY_LR,
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def locate_inputs(path: Path) -> tuple[Path, Path, Path | None]:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        manifest_path = resolved / "pilot_manifest.json"
    elif resolved.is_file():
        manifest_path = resolved
    else:
        raise RuntimeError(f"pilot input does not exist: {resolved}")
    if manifest_path.name != "pilot_manifest.json" or not manifest_path.is_file():
        raise RuntimeError(
            "input must be pilot_manifest.json or its batch directory: "
            f"{resolved}"
        )
    summary_path = manifest_path.with_name("pilot_summary.csv")
    if not summary_path.is_file():
        raise RuntimeError(f"pilot manifest has no sibling pilot_summary.csv: {manifest_path}")
    runner_selection = manifest_path.with_name("pilot_selection.json")
    return manifest_path, summary_path, runner_selection if runner_selection.is_file() else None


def as_bool(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{label} must be an explicit boolean, observed {value!r}")


def finite_float(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is non-finite: {value!r}")
    return result


def exact_int(value: object, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not an integer: {value!r}") from exc
    if isinstance(value, float) and value != result:
        raise RuntimeError(f"{label} is not integral: {value!r}")
    return result


def normalize_summary(raw: dict[str, object], *, source: str) -> dict[str, object]:
    cell_id = str(raw.get("cell_id", ""))
    if cell_id not in CELL_SPECS:
        raise RuntimeError(f"{source}: unexpected pilot cell_id {cell_id!r}")
    expected = CELL_SPECS[cell_id]
    method = str(raw.get("method", ""))
    if method != expected["method"]:
        raise RuntimeError(
            f"{source}: method mismatch for {cell_id}: "
            f"expected {expected['method']!r}, observed {method!r}"
        )
    lr = finite_float(raw.get("matrix_lr"), label=f"{source}:{cell_id}:matrix_lr")
    if not math.isclose(lr, float(expected["matrix_lr"]), rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(
            f"{source}: LR mismatch for {cell_id}: "
            f"expected {expected['matrix_lr']}, observed {lr}"
        )
    seed_value = raw.get("controlled_seed", raw.get("seed"))
    seed = exact_int(seed_value, label=f"{source}:{cell_id}:seed")
    if seed != PILOT_SEED:
        raise RuntimeError(f"{source}: {cell_id} must use seed {PILOT_SEED}, observed {seed}")
    steps = exact_int(raw.get("total_steps"), label=f"{source}:{cell_id}:total_steps")
    if steps != PILOT_STEPS:
        raise RuntimeError(
            f"{source}: {cell_id} must run {PILOT_STEPS} steps, observed {steps}"
        )
    if as_bool(raw.get("evidence_valid"), label=f"{source}:{cell_id}:evidence_valid") is not True:
        raise RuntimeError(f"{source}: {cell_id} was not accepted as valid local evidence")
    if "formal_eligible" in raw:
        formal_eligible = as_bool(
            raw["formal_eligible"], label=f"{source}:{cell_id}:formal_eligible"
        )
        if formal_eligible is not bool(expected["formal_eligible"]):
            raise RuntimeError(
                f"{source}: formal_eligible mismatch for {cell_id}: {formal_eligible}"
            )
    final_loss = finite_float(
        raw.get("final_val_loss"), label=f"{source}:{cell_id}:final_val_loss"
    )
    endpoint_loss = finite_float(
        raw.get("val_loss_step_1000"), label=f"{source}:{cell_id}:val_loss_step_1000"
    )
    if not math.isclose(final_loss, endpoint_loss, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"{source}: final/step-1000 loss mismatch for {cell_id}: "
            f"{final_loss} vs {endpoint_loss}"
        )
    if "total_tokens" in raw and raw.get("total_tokens") not in (None, ""):
        tokens = exact_int(raw["total_tokens"], label=f"{source}:{cell_id}:total_tokens")
        if tokens != PILOT_TOKENS:
            raise RuntimeError(
                f"{source}: token budget mismatch for {cell_id}: {tokens}"
            )
    return {
        "cell_id": cell_id,
        "method": method,
        "matrix_lr": lr,
        "controlled_seed": seed,
        "total_steps": steps,
        "total_tokens": PILOT_TOKENS,
        "evidence_valid": True,
        "formal_eligible": bool(expected["formal_eligible"]),
        "final_val_loss": final_loss,
        "val_loss_step_1000": endpoint_loss,
    }


def validate_exact_grid(rows: list[dict[str, object]], *, source: str) -> list[dict[str, object]]:
    if len(rows) != len(CELL_SPECS):
        raise RuntimeError(
            f"{source}: expected {len(CELL_SPECS)} pilot summaries, observed {len(rows)}"
        )
    normalized = [normalize_summary(row, source=source) for row in rows]
    ids = [str(row["cell_id"]) for row in normalized]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{source}: duplicate pilot cell IDs: {ids}")
    observed = set(ids)
    expected = set(CELL_SPECS)
    if observed != expected:
        raise RuntimeError(
            f"{source}: frozen grid mismatch; missing={sorted(expected - observed)} "
            f"unexpected={sorted(observed - expected)}"
        )
    return normalized


def compare_manifest_and_csv(
    manifest_rows: list[dict[str, object]], csv_rows: list[dict[str, object]]
) -> None:
    manifest_by_id = {str(row["cell_id"]): row for row in manifest_rows}
    csv_by_id = {str(row["cell_id"]): row for row in csv_rows}
    for cell_id in sorted(CELL_SPECS):
        left, right = manifest_by_id[cell_id], csv_by_id[cell_id]
        for key in (
            "method",
            "matrix_lr",
            "controlled_seed",
            "total_steps",
            "total_tokens",
            "evidence_valid",
            "formal_eligible",
            "final_val_loss",
            "val_loss_step_1000",
        ):
            if left[key] != right[key]:
                raise RuntimeError(
                    f"pilot manifest/CSV mismatch for {cell_id}:{key}: "
                    f"{left[key]!r} vs {right[key]!r}"
                )


def rank_method(rows: list[dict[str, object]], method: str) -> list[dict[str, object]]:
    method_rows = [row for row in rows if row["method"] == method]
    ranked = sorted(
        method_rows,
        key=lambda row: (
            float(row["final_val_loss"]),
            float(row["matrix_lr"]),
            str(row["cell_id"]),
        ),
    )
    return [
        {
            "method": method,
            "method_role": "formal_candidate",
            "method_rank": rank,
            **row,
        }
        for rank, row in enumerate(ranked, 1)
    ]


def select_method(
    ranked: list[dict[str, object]],
    *,
    method: str,
    center_lr: float | None,
    prefer_center: bool,
    lower_boundary_lr: float,
    upper_boundary_lr: float,
) -> dict[str, object]:
    if not ranked or any(row["method"] != method for row in ranked):
        raise RuntimeError(f"invalid ranked rows for {method}")
    if prefer_center is not (center_lr is not None):
        raise RuntimeError(
            f"{method}: center preference and center LR must either both be set or both be absent"
        )
    raw_best = ranked[0]
    center = (
        next(
            row
            for row in ranked
            if math.isclose(
                float(row["matrix_lr"]), center_lr, rel_tol=0.0, abs_tol=1e-15
            )
        )
        if center_lr is not None
        else None
    )
    raw_best_lr = float(raw_best["matrix_lr"])
    raw_best_is_lower_boundary = math.isclose(
        raw_best_lr, lower_boundary_lr, rel_tol=0.0, abs_tol=1e-15
    )
    raw_best_is_upper_boundary = math.isclose(
        raw_best_lr, upper_boundary_lr, rel_tol=0.0, abs_tol=1e-15
    )
    minimum_loss = float(raw_best["final_val_loss"])
    minimum_rows = [
        row for row in ranked if float(row["final_val_loss"]) == minimum_loss
    ]
    minimum_tied_cell_ids = [str(row["cell_id"]) for row in minimum_rows]
    lower_boundary = any(
        math.isclose(
            float(row["matrix_lr"]), lower_boundary_lr, rel_tol=0.0, abs_tol=1e-15
        )
        for row in minimum_rows
    )
    upper_boundary = any(
        math.isclose(
            float(row["matrix_lr"]), upper_boundary_lr, rel_tol=0.0, abs_tol=1e-15
        )
        for row in minimum_rows
    )
    boundary_side = (
        "both"
        if lower_boundary and upper_boundary
        else "lower"
        if lower_boundary
        else "upper"
        if upper_boundary
        else None
    )
    common: dict[str, object] = {
        "method": method,
        "role": "formal_candidate",
        "formal_eligible": True,
        "selection_performed": True,
        "selection_endpoint": "step-1000 validation loss",
        "selection_policy": (
            "paper_center_within_best_plus_0.002" if prefer_center else "raw_endpoint_best"
        ),
        "center_cell_id": center["cell_id"] if center is not None else None,
        "center_tie_margin": TIE_MARGIN if prefer_center else None,
        "paper_center_lr": center_lr,
        "center_preferred_if_within_margin_of_best": prefer_center,
        "lower_boundary_lr": lower_boundary_lr,
        "upper_boundary_lr": upper_boundary_lr,
        "raw_best_cell_id": raw_best["cell_id"],
        "raw_best_matrix_lr": raw_best["matrix_lr"],
        "raw_best_final_val_loss": raw_best["final_val_loss"],
        "minimum_tied_cell_ids": minimum_tied_cell_ids,
        "minimum_includes_lower_boundary": lower_boundary,
        "minimum_includes_upper_boundary": upper_boundary,
        "boundary_tie_policy": "any_boundary_at_reported_minimum_blocks_formal",
        "ranked_cells": [
            {
                "cell_id": row["cell_id"],
                "matrix_lr": row["matrix_lr"],
                "final_val_loss": row["final_val_loss"],
            }
            for row in ranked
        ],
    }
    if boundary_side is not None:
        return {
            **common,
            "status": "boundary_inconclusive",
            "scientific_result": f"{method}_boundary_inconclusive",
            "formal_allowed": False,
            "formal_eligible": False,
            "selected_cell_id": None,
            "selected_matrix_lr": None,
            "selected_final_val_loss": None,
            "boundary_rule_triggered": True,
            "boundary_side": boundary_side,
            "raw_best_is_lower_boundary": raw_best_is_lower_boundary,
            "raw_best_is_upper_boundary": raw_best_is_upper_boundary,
            "selection_reason": "boundary_inconclusive",
            "boundary_explanation": (
                f"The reported minimum for {method} includes the frozen "
                f"{boundary_side} boundary; formal is blocked pending a separately "
                f"frozen {'bidirectional' if boundary_side == 'both' else 'downward' if lower_boundary else 'upward'}-grid amendment."
            ),
        }
    center_within_margin = bool(
        prefer_center
        and center is not None
        and float(center["final_val_loss"])
        <= float(raw_best["final_val_loss"]) + TIE_MARGIN
    )
    selected = center if center_within_margin else raw_best
    assert selected is not None
    return {
        **common,
        "status": "selected",
        "scientific_result": f"{method}_selected",
        "formal_allowed": True,
        "selected_cell_id": selected["cell_id"],
        "selected_matrix_lr": selected["matrix_lr"],
        "selected_final_val_loss": selected["final_val_loss"],
        "boundary_rule_triggered": False,
        "boundary_side": None,
        "raw_best_is_lower_boundary": False,
        "raw_best_is_upper_boundary": False,
        "selection_reason": (
            "paper_center_within_best_plus_0.002"
            if center_within_margin
            else "raw_endpoint_best"
        ),
    }


def recompute_selection(rows: list[dict[str, object]], manifest_path: Path) -> dict[str, object]:
    ranked_by_method = {
        method: rank_method(rows, method) for method in METHOD_SELECTION_RULES
    }
    if len(ranked_by_method["malt"]) != 6 or len(ranked_by_method["malter_eq17"]) != 6:
        raise RuntimeError("method coverage must be exactly six MALT and six MALTER-Eq17 cells")
    required_formal_methods = ["malt", "malter_eq17"]
    method_selections = {
        method: select_method(ranked_by_method[method], method=method, **rules)
        for method, rules in METHOD_SELECTION_RULES.items()
    }
    blocking_methods = [
        method
        for method, selection in method_selections.items()
        if selection["formal_allowed"] is not True
    ]
    formal_allowed = not blocking_methods
    malt_selection = method_selections["malt"]
    selected_methods = {
        method: (
            {
                "cell_id": selection["selected_cell_id"],
                "matrix_lr": selection["selected_matrix_lr"],
                "final_val_loss": selection["selected_final_val_loss"],
            }
            if selection["status"] == "selected"
            else None
        )
        for method, selection in method_selections.items()
    }
    return {
        "protocol": SELECTION_PROTOCOL,
        "certificate_role": "independent_pilot_analysis_selection",
        "status": "selected" if formal_allowed else "boundary_inconclusive",
        "scientific_result": (
            "dual_methods_selected"
            if formal_allowed
            else "boundary_inconclusive"
        ),
        "formal_allowed": formal_allowed,
        "formal_eligible": formal_allowed,
        "formal_policy": "all_methods_fail_closed",
        "required_formal_methods": required_formal_methods,
        "blocking_methods": blocking_methods,
        "seed": PILOT_SEED,
        "pilot_steps": PILOT_STEPS,
        "selection_endpoint": "step-1000 validation loss",
        "grid_design": "fresh_v4_focused_malt_upper_grid_dual_method",
        "malt_execution_order": [
            float(spec["matrix_lr"])
            for spec in CELL_SPECS.values()
            if spec["method"] == "malt"
        ],
        "malt_selection_policy": "raw_endpoint_best",
        "malter_center_tie_margin": TIE_MARGIN,
        "malter_center_preferred_if_within_margin_of_best": True,
        "pilot_manifest": str(manifest_path),
        "pilot_manifest_sha256": sha256_file(manifest_path),
        "selected_methods": selected_methods,
        "boundary_rule_triggered": bool(blocking_methods),
        "boundary_methods": blocking_methods,
        "boundary_sides": {
            method: method_selections[method]["boundary_side"]
            for method in blocking_methods
        },
        "selections": method_selections,
        # Compatibility aliases are diagnostic only. Formal consumers must
        # validate the complete nested dual-method envelope and global gate.
        "selected_cell_id": malt_selection["selected_cell_id"],
        "selected_matrix_lr": malt_selection["selected_matrix_lr"],
        "selected_final_val_loss": malt_selection["selected_final_val_loss"],
        "raw_best_cell_id": malt_selection["raw_best_cell_id"],
        "raw_best_matrix_lr": malt_selection["raw_best_matrix_lr"],
        "raw_best_final_val_loss": malt_selection["raw_best_final_val_loss"],
        "malt_ranked_cells": malt_selection["ranked_cells"],
        "malter_eq17_role": "formal_candidate",
        "malter_eq17_selected_cell_id": method_selections["malter_eq17"][
            "selected_cell_id"
        ],
        "malter_eq17_selected_matrix_lr": method_selections["malter_eq17"][
            "selected_matrix_lr"
        ],
        "malter_eq17_ranked_cells": method_selections["malter_eq17"][
            "ranked_cells"
        ],
    }


def compare_runner_selection(path: Path | None, recomputed: dict[str, object]) -> dict[str, object]:
    if path is None:
        return {
            "status": "not_present_analyzer_selection_is_authoritative",
            "path": None,
            "sha256": None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed_selections = payload.get("selections")
    if not isinstance(observed_selections, dict):
        observed_selections = {}
    expected_selections = recomputed["selections"]
    assert isinstance(expected_selections, dict)
    method_fields = (
        "method",
        "selection_policy",
        "center_cell_id",
        "center_tie_margin",
        "status",
        "formal_allowed",
        "formal_eligible",
        "selected_cell_id",
        "selected_matrix_lr",
        "selection_reason",
        "raw_best_cell_id",
        "raw_best_matrix_lr",
        "minimum_tied_cell_ids",
        "minimum_includes_lower_boundary",
        "minimum_includes_upper_boundary",
        "boundary_tie_policy",
        "boundary_rule_triggered",
        "boundary_side",
        "ranked_cells",
    )
    checks = {
        "certificate_role": payload.get("certificate_role")
        == "runner_preselection_crosscheck",
        "protocol": payload.get("protocol") == SELECTION_PROTOCOL,
        "status": payload.get("status") == recomputed["status"],
        "scientific_result": payload.get("scientific_result")
        == recomputed["scientific_result"],
        "seed": payload.get("seed") == PILOT_SEED,
        "pilot_steps": payload.get("pilot_steps") == PILOT_STEPS,
        "selection_endpoint": payload.get("selection_endpoint")
        == recomputed["selection_endpoint"],
        "formal_allowed": payload.get("formal_allowed")
        is recomputed["formal_allowed"],
        "formal_eligible": payload.get("formal_eligible")
        is recomputed["formal_eligible"],
        "required_formal_methods": payload.get("required_formal_methods")
        == recomputed["required_formal_methods"],
        "blocking_methods": payload.get("blocking_methods")
        == recomputed["blocking_methods"],
        "grid_design": payload.get("grid_design") == recomputed["grid_design"],
        "malt_execution_order": payload.get("malt_execution_order")
        == recomputed["malt_execution_order"],
        "malt_selection_policy": payload.get("malt_selection_policy")
        == recomputed["malt_selection_policy"],
        "malter_center_tie_margin": payload.get("malter_center_tie_margin")
        == recomputed["malter_center_tie_margin"],
        "malter_center_preferred_if_within_margin_of_best": payload.get(
            "malter_center_preferred_if_within_margin_of_best"
        )
        is recomputed["malter_center_preferred_if_within_margin_of_best"],
        "pilot_manifest": payload.get("pilot_manifest")
        == recomputed["pilot_manifest"],
        "pilot_manifest_sha256": payload.get("pilot_manifest_sha256")
        == recomputed["pilot_manifest_sha256"],
        "boundary_rule_triggered": payload.get("boundary_rule_triggered")
        is recomputed["boundary_rule_triggered"],
    }
    for method in recomputed["required_formal_methods"]:
        observed = observed_selections.get(method)
        expected = expected_selections[method]
        checks[f"{method}_selection_present"] = isinstance(observed, dict)
        for field in method_fields:
            checks[f"{method}:{field}"] = (
                isinstance(observed, dict) and observed.get(field) == expected[field]
            )
    if not all(checks.values()):
        raise RuntimeError(f"runner pilot selection disagrees with independent analysis: {checks}")
    return {
        "status": "matched",
        "path": str(path),
        "sha256": sha256_file(path),
        "checks": checks,
    }


def analyze(input_path: Path, output_dir: Path) -> dict[str, Path]:
    manifest_path, summary_path, runner_selection_path = locate_inputs(input_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in ACCEPTED_MANIFEST_STATUSES:
        raise RuntimeError(
            f"pilot manifest is not accepted local evidence: {manifest.get('status')!r}"
        )
    if manifest.get("family") not in (None, FAMILY):
        raise RuntimeError(f"pilot family mismatch: {manifest.get('family')!r}")
    if manifest.get("protocol") != PILOT_PROTOCOL:
        raise RuntimeError(f"pilot protocol mismatch: {manifest.get('protocol')!r}")
    if exact_int(manifest.get("seed"), label="pilot manifest seed") != PILOT_SEED:
        raise RuntimeError("pilot aggregate manifest must use seed 2026")
    if exact_int(manifest.get("total_steps"), label="pilot manifest total_steps") != PILOT_STEPS:
        raise RuntimeError("pilot aggregate manifest must use exactly 1000 steps")
    failures = manifest.get("failures", [])
    if failures not in (None, []):
        raise RuntimeError(f"pilot aggregate manifest records failures: {failures!r}")
    summaries = manifest.get("summaries")
    if not isinstance(summaries, list):
        raise RuntimeError("pilot manifest summaries must be a list")
    manifest_rows = validate_exact_grid(summaries, source="pilot_manifest.json")
    csv_rows = validate_exact_grid(read_csv(summary_path), source="pilot_summary.csv")
    compare_manifest_and_csv(manifest_rows, csv_rows)
    selection = recompute_selection(manifest_rows, manifest_path)
    runner_crosscheck = compare_runner_selection(runner_selection_path, selection)

    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    ranking_path = output / "pilot_ranking.csv"
    selection_path = output / "pilot_selection_verified.json"
    manifest_out_path = output / "pilot_analysis_manifest.json"
    manifest_hash_path = output / "pilot_analysis_manifest.sha256"

    ranking_rows = rank_method(manifest_rows, "malt") + rank_method(
        manifest_rows, "malter_eq17"
    )
    write_csv(ranking_path, ranking_rows)
    write_json(selection_path, selection)
    analysis_manifest: dict[str, Any] = {
        "status": "completed_valid",
        "protocol": ANALYSIS_PROTOCOL,
        "certificate_role": "independent_pilot_analysis_manifest",
        "family": FAMILY,
        "scientific_result": selection["scientific_result"],
        "formal_allowed": selection["formal_allowed"],
        "implementation_label": "paper-derived independent implementation",
        "official_author_code_public_at_freeze": False,
        "input_files": [
            {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        ],
        "runner_selection_crosscheck": runner_crosscheck,
        "checks": {
            "accepted_aggregate_manifest": True,
            "exact_twelve_cell_grid": True,
            "exact_method_lr_mapping": True,
            "all_seed_2026": True,
            "all_1000_steps": True,
            "all_evidence_valid": True,
            "manifest_csv_exact_match": True,
            "malt_formal_selection_rule_applied": True,
            "malter_eq17_formal_selection_rule_applied": True,
            "dual_method_fail_closed_rule_applied": True,
        },
        "outputs": [
            {"path": ranking_path.name, "sha256": sha256_file(ranking_path)},
            {"path": selection_path.name, "sha256": sha256_file(selection_path)},
        ],
    }
    write_json(manifest_out_path, analysis_manifest)
    manifest_hash_path.write_text(
        f"{sha256_file(manifest_out_path)}  {manifest_out_path.name}\n", encoding="ascii"
    )
    return {
        "ranking": ranking_path,
        "selection": selection_path,
        "manifest": manifest_out_path,
        "manifest_sha256": manifest_hash_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pilot",
        type=Path,
        help="pilot_manifest.json or the batch directory containing it",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = analyze(args.pilot, args.output_dir)
    print(outputs["manifest"])


if __name__ == "__main__":
    main()
