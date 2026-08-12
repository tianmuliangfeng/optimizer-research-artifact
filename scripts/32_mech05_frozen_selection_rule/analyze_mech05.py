#!/usr/bin/env python3
"""MECH-05: freeze and apply a conservative K-representation selection rule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "2026-07-27.1"
EXPECTED_FAMILIES = ("r1", "llama124")
EXPECTED_LAYERS = tuple(range(12))
EXPECTED_MECH03_LAYERS = (0, 4, 8, 11)
EXPECTED_REPEATS = tuple(range(4))
EXPECTED_DIRECTIONS = ("A_to_B", "B_to_A")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=here / "selection_rule_contract.json")
    parser.add_argument("--r1-mech02-formal-dir", required=True, type=Path)
    parser.add_argument("--llama124-mech02-formal-dir", required=True, type=Path)
    parser.add_argument("--r1-mech03-formal-dir", required=True, type=Path)
    parser.add_argument("--llama124-mech03-formal-dir", required=True, type=Path)
    parser.add_argument("--r1-run-summary", required=True, type=Path)
    parser.add_argument("--llama124-run-summary", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {row}")
    return value


def median(values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        raise ValueError("median of empty sequence")
    return float(statistics.median(data))


def sample_sd(values: Iterable[float]) -> float:
    data = list(values)
    return float(statistics.stdev(data)) if len(data) > 1 else 0.0


def validate_output_dir(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)


def validate_manifest(formal_dir: Path, expected_family: str, stage: str) -> dict[str, Any]:
    candidates = [formal_dir / f"{stage}_manifest.json", formal_dir / "mech03_manifest.json"]
    if stage == "mech02":
        candidates = [formal_dir / "mech02_manifest.json"]
    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None:
        raise FileNotFoundError(f"missing {stage} manifest under {formal_dir}")
    manifest = read_json(manifest_path)
    if manifest.get("passed") is not True:
        raise ValueError(f"{stage} manifest did not pass: {manifest_path}")
    family = manifest.get("family")
    if family is not None and family != expected_family:
        raise ValueError(
            f"{stage} family mismatch for {expected_family}: {family} in {manifest_path}"
        )
    return {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "passed": True}


def validate_geometry(
    formal_dir: Path, family: str
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    geometry_path = formal_dir / "geometry.csv"
    stability_path = formal_dir / "stability.csv"
    geometry = read_csv(geometry_path)
    stability = read_csv(stability_path)
    expected_geometry = {
        (layer, repeat) for layer in EXPECTED_LAYERS for repeat in EXPECTED_REPEATS
    }
    observed_geometry = {
        (int(row["layer"]), int(row["repeat"]))
        for row in geometry
        if row["family"] == family
    }
    expected_stability = {
        (layer, repeat_a, repeat_b)
        for layer in EXPECTED_LAYERS
        for repeat_a in EXPECTED_REPEATS
        for repeat_b in EXPECTED_REPEATS
        if repeat_a < repeat_b
    }
    observed_stability = {
        (int(row["layer"]), int(row["repeat_a"]), int(row["repeat_b"]))
        for row in stability
        if row["family"] == family
    }
    if observed_geometry != expected_geometry:
        raise ValueError(f"{family} MECH-02 geometry coverage mismatch")
    if observed_stability != expected_stability:
        raise ValueError(f"{family} MECH-02 stability coverage mismatch")
    return geometry, stability, {
        "geometry_path": str(geometry_path),
        "geometry_sha256": sha256_file(geometry_path),
        "geometry_rows": len(geometry),
        "stability_path": str(stability_path),
        "stability_sha256": sha256_file(stability_path),
        "stability_rows": len(stability),
    }


def validate_line_search(
    formal_dir: Path, family: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    path = formal_dir / "line_search_summary.csv"
    rows = [
        row
        for row in read_csv(path)
        if row["family"] == family and row["scope"] == "layer"
    ]
    expected_candidates = (
        {"none", "diag", "block4", "dense_full"}
        if family == "r1"
        else {"none", "diag", "dense_full"}
    )
    coverage: dict[str, set[tuple[int, int, str]]] = defaultdict(set)
    for row in rows:
        coverage[row["candidate"]].add(
            (int(row["layer"]), int(row["repeat"]), row["direction"])
        )
    expected_cells = {
        (layer, repeat, direction)
        for layer in EXPECTED_MECH03_LAYERS
        for repeat in EXPECTED_REPEATS
        for direction in EXPECTED_DIRECTIONS
    }
    if set(coverage) != expected_candidates:
        raise ValueError(f"{family} candidate mismatch: {sorted(coverage)}")
    for candidate in expected_candidates:
        if coverage[candidate] != expected_cells:
            raise ValueError(f"{family}/{candidate} MECH-03 coverage mismatch")
    return rows, {
        "line_search_path": str(path),
        "line_search_sha256": sha256_file(path),
        "layer_rows": len(rows),
        "candidate_rows": {key: len(value) for key, value in sorted(coverage.items())},
    }


def validate_longrun(
    path: Path, family: str, contract: dict[str, Any]
) -> tuple[dict[str, float], dict[str, Any]]:
    rows = read_csv(path)
    seed_rows = [row for row in rows if int(row["seed"]) == 2026]
    by_method = {row["method"]: as_float(row, "final_val_loss") for row in seed_rows}
    method_map = contract["longrun_method_map"][family]
    losses: dict[str, float] = {}
    missing: list[str] = []
    for canonical, source_name in method_map.items():
        if source_name is None:
            continue
        if source_name not in by_method:
            missing.append(source_name)
        else:
            losses[canonical] = by_method[source_name]
    if missing:
        raise ValueError(f"{family} seed2026 run summary missing methods: {missing}")
    return losses, {
        "path": str(path),
        "sha256": sha256_file(path),
        "seed": 2026,
        "source_methods": sorted(by_method),
    }


def geometry_features(
    geometry: list[dict[str, str]],
    stability: list[dict[str, str]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    diag_ratio = median(as_float(row, "diag_p95_over_p05") for row in geometry)
    offdiag = median(as_float(row, "offdiag_energy_fraction") for row in geometry)
    diagonal_cosine = median(as_float(row, "diagonal_cosine") for row in stability)
    covariance_drift = median(
        as_float(row, "covariance_relative_drift") for row in stability
    )
    top_overlap = median(as_float(row, "top_eigenspace_overlap") for row in stability)
    anisotropic = (
        diag_ratio >= thresholds["minimum_diagonal_anisotropy_p95_over_p05"]
    )
    stable = (
        diagonal_cosine >= thresholds["minimum_diagonal_cosine"]
        and covariance_drift <= thresholds["maximum_covariance_relative_drift"]
    )
    near_scalar = (
        diag_ratio <= thresholds["maximum_scalar_isotropy_p95_over_p05"]
        and offdiag <= thresholds["maximum_scalar_isotropy_offdiag_energy_fraction"]
    )
    stable_non_diagonal = (
        top_overlap >= thresholds["minimum_top_eigenspace_overlap"]
        and covariance_drift <= thresholds["maximum_covariance_relative_drift"]
    )
    return {
        "median_diag_p95_over_p05": diag_ratio,
        "median_offdiag_energy_fraction": offdiag,
        "median_diagonal_cosine": diagonal_cosine,
        "median_covariance_relative_drift": covariance_drift,
        "median_top_eigenspace_overlap": top_overlap,
        "diagonal_anisotropy_present": anisotropic,
        "geometry_stable": stable,
        "stable_diagonal_anisotropy": anisotropic and stable,
        "near_scalar_isotropy": near_scalar,
        "stable_non_diagonal_subspace": stable_non_diagonal,
    }


def heldout_contrast(
    rows: list[dict[str, str]],
    candidate: str,
    comparator: str,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    scores: dict[tuple[int, int, str, str], float] = {}
    for row in rows:
        key = (
            int(row["layer"]),
            int(row["repeat"]),
            row["direction"],
            row["candidate"],
        )
        scores[key] = as_float(row, "best_relative_loss_delta")
    cells: list[dict[str, Any]] = []
    for layer in EXPECTED_MECH03_LAYERS:
        for repeat in EXPECTED_REPEATS:
            for direction in EXPECTED_DIRECTIONS:
                candidate_score = scores[(layer, repeat, direction, candidate)]
                comparator_score = scores[(layer, repeat, direction, comparator)]
                cells.append(
                    {
                        "layer": layer,
                        "repeat": repeat,
                        "direction": direction,
                        "advantage": comparator_score - candidate_score,
                        "candidate_score": candidate_score,
                        "comparator_score": comparator_score,
                    }
                )
    advantages = [row["advantage"] for row in cells]
    positive = sum(value > 0.0 for value in advantages)
    layer_rows: list[dict[str, Any]] = []
    for layer in EXPECTED_MECH03_LAYERS:
        subset = [row for row in cells if row["layer"] == layer]
        mean_advantage = statistics.mean(row["advantage"] for row in subset)
        envelope = max(
            sample_sd(row["candidate_score"] for row in subset),
            sample_sd(row["comparator_score"] for row in subset),
        )
        material = mean_advantage > max(
            envelope, thresholds["relative_shadow_loss_margin"]
        )
        layer_rows.append(
            {
                "layer": layer,
                "mean_advantage": mean_advantage,
                "repeat_sd_envelope": envelope,
                "positive_material": material,
            }
        )
    material_layers = sum(row["positive_material"] for row in layer_rows)
    positive_fraction = positive / len(cells)
    median_advantage = median(advantages)
    stable_positive = (
        median_advantage > thresholds["relative_shadow_loss_margin"]
        and positive_fraction >= thresholds["minimum_positive_cell_fraction"]
        and material_layers >= thresholds["minimum_positive_material_layers"]
    )
    return {
        "candidate": candidate,
        "comparator": comparator,
        "advantage_definition": "comparator_best_relative_loss_delta_minus_candidate_best_relative_loss_delta",
        "cells": len(cells),
        "median_advantage": median_advantage,
        "mean_advantage": statistics.mean(advantages),
        "advantage_sd": sample_sd(advantages),
        "positive_cells": positive,
        "positive_cell_fraction": positive_fraction,
        "positive_material_layers": material_layers,
        "stable_positive_advantage": stable_positive,
        "layer_summary": layer_rows,
    }


def longrun_features(losses: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    none_loss = losses["none"]
    advantages = {
        candidate: none_loss - loss
        for candidate, loss in losses.items()
        if candidate not in {"none", "muon"}
    }
    best_candidate = max(advantages, key=advantages.get)
    best_advantage = advantages[best_candidate]
    margin = thresholds["longrun_practical_loss_margin"]
    return {
        "losses": losses,
        "candidate_advantage_over_none": advantages,
        "best_k_candidate": best_candidate,
        "best_k_advantage_over_none": best_advantage,
        "material_k_gain_over_none": best_advantage > margin,
        "none_muon_loss_delta": none_loss - losses["muon"],
        "none_and_muon_practically_tied": abs(none_loss - losses["muon"]) <= margin,
    }


def choose_decision(
    family: str,
    geometry: dict[str, Any],
    heldout: dict[str, dict[str, Any]],
    longrun: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    diag_shadow = heldout["diag_vs_none"]["stable_positive_advantage"]
    diag_signal = geometry["stable_diagonal_anisotropy"] and diag_shadow
    extra_names = [name for name in heldout if name != "diag_vs_none"]
    stable_extra = [
        name for name in extra_names if heldout[name]["stable_positive_advantage"]
    ]
    offdiag_signal = geometry["stable_non_diagonal_subspace"] and bool(stable_extra)
    margin = thresholds["longrun_practical_loss_margin"]
    gains = longrun["candidate_advantage_over_none"]
    diag_longrun_contradiction = gains.get("diag", 0.0) < -margin
    extra_longrun_contradiction = all(
        gains.get(heldout[name]["candidate"], 0.0)
        < gains.get("diag", 0.0) - margin
        for name in stable_extra
    ) if stable_extra else False
    no_stable_complexity_gain = not diag_shadow and not stable_extra
    none_evidence = geometry["near_scalar_isotropy"] or (
        no_stable_complexity_gain and not longrun["material_k_gain_over_none"]
    )

    reasons: list[str] = []
    if offdiag_signal and not extra_longrun_contradiction:
        decision = "full_or_block"
        reasons.append("stable non-diagonal geometry and held-out incremental gain")
    elif diag_signal and not diag_longrun_contradiction:
        decision = "diag"
        reasons.append("stable diagonal anisotropy and held-out diag gain")
    elif none_evidence:
        decision = "none_or_muon_sufficient"
        reasons.append("no stable held-out complexity gain and no material long-run K gain")
    else:
        decision = "uncertain"
        if longrun["material_k_gain_over_none"] and no_stable_complexity_gain:
            reasons.append("long-run K gain conflicts with unstable endpoint shadow evidence")
        if geometry["diagonal_anisotropy_present"] and not geometry["geometry_stable"]:
            reasons.append("anisotropy is present but covariance/diagonal stability gate fails")
        if not reasons:
            reasons.append("no decision branch satisfied all frozen requirements")
    return {
        "family": family,
        "decision": decision,
        "reasons": reasons,
        "diag_signal": diag_signal,
        "offdiag_signal": offdiag_signal,
        "no_stable_complexity_gain": no_stable_complexity_gain,
        "none_evidence": none_evidence,
        "longrun_material_k_gain_over_none": longrun["material_k_gain_over_none"],
        "geometry_only_selection_used": False,
    }


def build_report(
    contract_sha: str,
    features: dict[str, Any],
    decisions: dict[str, Any],
    validation: dict[str, Any],
) -> str:
    lines = [
        "# MECH-05 冻结 K 表示选择规则",
        "",
        f"脚本版本：`{SCRIPT_VERSION}`  ",
        f"冻结 contract SHA-256：`{contract_sha}`  ",
        "状态：规则已冻结；无新训练；MECH-04 未启用。",
        "",
        "## 1. 防泄漏边界",
        "",
        "- Discovery 只使用 R1 与 LLaMA-124M 的 seed2026、step6200 数据；",
        "- GPT bridge 只承担 runtime robustness，不进入选择决策；",
        "- LLaMA-1B 三 seed formal 排名在冻结前已经被读取，因此只能作为",
        "  retrospective context，未用于阈值或规则生成；",
        "- 尚未生成/读取的 MECH-06-L1B 诊断产物是下一项 confirmation；",
        "- 几何幅度不能单独触发 `diag/full/block`。",
        "",
        "## 2. 冻结阈值",
        "",
        "| 阈值 | 数值 |",
        "|---|---:|",
    ]
    for name, value in validation["thresholds"].items():
        lines.append(f"| `{name}` | {value} |")
    lines.extend(
        [
            "",
            "## 3. Discovery 结果",
            "",
            "| family | diag p95/p05 | diagonal cosine | covariance drift | "
            "diag held-out positive | best long-run K gain vs none | decision |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for family in EXPECTED_FAMILIES:
        f = features[family]
        g = f["geometry"]
        h = f["heldout"]["diag_vs_none"]
        lr = f["longrun"]
        lines.append(
            f"| {family} | {g['median_diag_p95_over_p05']:.4g} | "
            f"{g['median_diagonal_cosine']:.4g} | "
            f"{g['median_covariance_relative_drift']:.4g} | "
            f"{h['positive_cells']}/{h['cells']} | "
            f"{lr['best_k_advantage_over_none']:+.6f} | "
            f"`{decisions[family]['decision']}` |"
        )
    lines.extend(["", "## 4. 解释", ""])
    for family in EXPECTED_FAMILIES:
        decision = decisions[family]
        lines.append(
            f"- **{family} → `{decision['decision']}`**："
            + "；".join(decision["reasons"])
            + "。"
        )
    lines.extend(
        [
            "",
            "这是一条保守规则：若 endpoint 机制指标与已有长程质量结果冲突，",
            "输出 `uncertain`，不把冲突强行压成某个 K-mode。`none_or_muon_sufficient`",
            "只表示在冻结 practical margin 下没有证据支持更复杂的目标 K-state；",
            "它不把 `none` 与 Muon 当成同一种优化器。",
            "",
            "## 5. 下一步",
            "",
            "运行 MECH-06-L1B 只读诊断，将冻结 contract 原样应用到尚未读取的",
            "1B 诊断特征。已有 1B formal 排名不得被重新包装为 prospective confirmation。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    validate_output_dir(args.output_dir)
    contract = read_json(args.contract)
    if contract.get("contract_version") != "2026-07-27.1":
        raise ValueError("unexpected contract version")
    if contract["discovery_scope"]["families"] != list(EXPECTED_FAMILIES):
        raise ValueError("contract discovery families changed")
    thresholds = contract["thresholds"]

    dirs = {
        "r1": {
            "mech02": args.r1_mech02_formal_dir,
            "mech03": args.r1_mech03_formal_dir,
            "summary": args.r1_run_summary,
        },
        "llama124": {
            "mech02": args.llama124_mech02_formal_dir,
            "mech03": args.llama124_mech03_formal_dir,
            "summary": args.llama124_run_summary,
        },
    }
    validation: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "analysis_script_path": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "contract_path": str(args.contract),
        "contract_sha256": sha256_file(args.contract),
        "thresholds": thresholds,
        "families": {},
        "checks": {},
    }
    features: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    feature_rows: list[dict[str, Any]] = []

    for family in EXPECTED_FAMILIES:
        spec = dirs[family]
        mech02_manifest = validate_manifest(spec["mech02"], family, "mech02")
        mech03_manifest = validate_manifest(spec["mech03"], family, "mech03")
        geometry_rows, stability_rows, geometry_validation = validate_geometry(
            spec["mech02"], family
        )
        line_rows, line_validation = validate_line_search(spec["mech03"], family)
        losses, longrun_validation = validate_longrun(
            spec["summary"], family, contract
        )
        geometry = geometry_features(geometry_rows, stability_rows, thresholds)
        heldout = {
            "diag_vs_none": heldout_contrast(
                line_rows, "diag", "none", thresholds
            ),
            "dense_full_vs_diag": heldout_contrast(
                line_rows, "dense_full", "diag", thresholds
            ),
        }
        if family == "r1":
            heldout["block4_vs_diag"] = heldout_contrast(
                line_rows, "block4", "diag", thresholds
            )
        longrun = longrun_features(losses, thresholds)
        features[family] = {
            "geometry": geometry,
            "heldout": heldout,
            "longrun": longrun,
        }
        decisions[family] = choose_decision(
            family, geometry, heldout, longrun, thresholds
        )
        validation["families"][family] = {
            "mech02_manifest": mech02_manifest,
            "mech03_manifest": mech03_manifest,
            "geometry": geometry_validation,
            "line_search": line_validation,
            "longrun": longrun_validation,
        }
        feature_rows.append(
            {
                "family": family,
                **{key: value for key, value in geometry.items() if not isinstance(value, bool)},
                "stable_diagonal_anisotropy": geometry["stable_diagonal_anisotropy"],
                "stable_non_diagonal_subspace": geometry["stable_non_diagonal_subspace"],
                "diag_median_advantage": heldout["diag_vs_none"]["median_advantage"],
                "diag_positive_cell_fraction": heldout["diag_vs_none"][
                    "positive_cell_fraction"
                ],
                "diag_positive_material_layers": heldout["diag_vs_none"][
                    "positive_material_layers"
                ],
                "diag_stable_positive_advantage": heldout["diag_vs_none"][
                    "stable_positive_advantage"
                ],
                "best_longrun_k_candidate": longrun["best_k_candidate"],
                "best_longrun_k_advantage_over_none": longrun[
                    "best_k_advantage_over_none"
                ],
                "decision": decisions[family]["decision"],
            }
        )

    validation["checks"] = {
        "discovery_families_exact": set(features) == set(EXPECTED_FAMILIES),
        "llama1b_not_an_input": all(
            "llama1b" not in str(value).lower()
            for family_value in dirs.values()
            for value in family_value.values()
        ),
        "geometry_only_selection_disabled": contract["anti_leakage"][
            "allow_geometry_only_selection"
        ]
        is False,
        "mech04_rescue_disabled": contract["anti_leakage"][
            "allow_mech04_rescue"
        ]
        is False,
        "all_decisions_valid": all(
            row["decision"]
            in {"diag", "full_or_block", "none_or_muon_sufficient", "uncertain"}
            for row in decisions.values()
        ),
    }
    if not all(validation["checks"].values()):
        raise RuntimeError(f"MECH-05 validation failed: {validation['checks']}")

    contract_snapshot = args.output_dir / "frozen_selection_rule.json"
    shutil.copyfile(args.contract, contract_snapshot)
    write_json(args.output_dir / "input_validation.json", validation)
    write_json(args.output_dir / "discovery_features.json", features)
    write_json(args.output_dir / "selection_decisions.json", decisions)
    write_csv(
        args.output_dir / "discovery_summary.csv",
        feature_rows,
        list(feature_rows[0]),
    )
    report = build_report(validation["contract_sha256"], features, decisions, validation)
    (args.output_dir / "MECH05_FROZEN_SELECTION_RULE_REPORT.md").write_text(
        report, encoding="utf-8"
    )

    output_files = sorted(
        path for path in args.output_dir.iterdir() if path.is_file()
    )
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "passed": True,
        "analysis_script_sha256": validation["analysis_script_sha256"],
        "contract_sha256": validation["contract_sha256"],
        "discovery_scope": contract["discovery_scope"],
        "known_before_freeze_but_excluded_from_rule_generation": contract[
            "known_before_freeze_but_excluded_from_rule_generation"
        ],
        "unread_confirmation_sets": contract["unread_confirmation_sets"],
        "decisions": {
            family: value["decision"] for family, value in decisions.items()
        },
        "output_sha256": {
            path.name: sha256_file(path) for path in output_files
        },
    }
    write_json(args.output_dir / "mech05_manifest.json", manifest)
    print(f"MECH-05 manifest: {args.output_dir / 'mech05_manifest.json'}")
    print(f"MECH-05 artifacts: {args.output_dir}")
    for family in EXPECTED_FAMILIES:
        print(f"MECH-05 decision {family}: {decisions[family]['decision']}")


if __name__ == "__main__":
    main()
