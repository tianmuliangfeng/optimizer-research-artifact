#!/usr/bin/env python3
"""Build and independently verify the local-primary EX54 analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any

import protocol as P


HERE = Path(__file__).resolve().parent
PACKAGE_REL = Path("scripts/54_llama_moonlight_multiscale_multibudget")
BUDGET_PHASES = {
    "tokens_3p2506b": ["backbone_4400", "cooldown_6200"],
    "tokens_6p9694b": ["backbone_4400", "backbone_11493", "cooldown_13293"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("full", "non10b"), default="full")
    parser.add_argument("--full-checkpoint-hash", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validation_rows(paths: list[Path]) -> list[dict[str, float]]:
    by_step: dict[int, float] = {}
    for path in paths:
        for row in P.read_metrics(path):
            if row.get("event") == "val":
                by_step[int(row["step"])] = float(row["loss"])
    return [{"step": step, "loss": by_step[step]} for step in sorted(by_step)]


def curve_stats(rows: list[dict[str, float]]) -> tuple[float, float]:
    if not rows:
        raise RuntimeError("validation curve is empty")
    tail = statistics.fmean(row["loss"] for row in rows[-5:])
    if len(rows) == 1 or rows[-1]["step"] == rows[0]["step"]:
        auc = rows[-1]["loss"]
    else:
        area = sum(
            (right["step"] - left["step"]) * (left["loss"] + right["loss"]) / 2.0
            for left, right in zip(rows, rows[1:])
        )
        auc = area / (rows[-1]["step"] - rows[0]["step"])
    return float(tail), float(auc)


def expected_moonlight_hyperparameters(contract: dict[str, Any]) -> dict[str, Any]:
    moonlight = contract["moonlight"]
    return {
        "momentum": float(moonlight["momentum"]),
        "nesterov": bool(moonlight["nesterov"]),
        "ns_steps": int(moonlight["newton_schulz_steps"]),
        "weight_decay": float(moonlight["weight_decay"]),
    }


def state_fields(
    summary: dict[str, Any], contract: dict[str, Any], scale: str
) -> dict[str, int]:
    matrix_state = int(summary.get("moonlight_matrix_optimizer_state_bytes", 0))
    momentum_state = int(summary.get("momentum_buffer_bytes", 0))
    schema = summary.get("moonlight_state_schema", {})
    expected_matrices = int(contract["profiles"][scale]["expected_matrix_tensors"])
    if (
        matrix_state <= 0
        or momentum_state <= 0
        or matrix_state != momentum_state
        or summary.get("moonlight_hyperparameters")
        != expected_moonlight_hyperparameters(contract)
        or not isinstance(schema, dict)
        or schema.get("optimizer") != "R1MoonlightMuon"
        or schema.get("tensor_state_keys") != ["momentum_buffer"]
        or int(schema.get("logical_matrix_parameters", -1)) != expected_matrices
        or schema.get("contains_activation_k_state") is not False
        or schema.get("contains_factor_or_eigendecomposition_state") is not False
    ):
        raise RuntimeError(
            "EX54 Moonlight momentum-state audit is missing or internally inconsistent"
        )
    return {
        "moonlight_momentum_state_bytes": momentum_state,
        "moonlight_matrix_optimizer_state_bytes": matrix_state,
    }


def collect_endpoints(
    run_dir: Path, contract: dict[str, Any], *, scope: str = "full"
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in contract["formal"]["seeds"]:
        unit124 = run_dir / "formal/124m" / f"seed{seed}"
        summary = P.read_json(unit124 / "summary.json")
        tail, auc = curve_stats(validation_rows([unit124 / "metrics.csv"]))
        rows.append({
            "scale": "124m", "budget_id": "tokens_3p2506b", "target_step": 6200,
            "seed": seed, "method": "moonlight", "final_val_loss": float(summary["final_val_loss"]),
            "tail5_val_loss": tail, "normalized_val_auc": auc,
            "optimizer_state_bytes": int(summary["optimizer_state_bytes"]),
            "peak_allocated_bytes": int(summary["peak_allocated_bytes"]),
            "checkpoint_path": summary["checkpoint_path"],
            "checkpoint_sha256": P.read_json(unit124 / "unit_manifest.json")["checkpoint_sha256"],
            "checkpoint_bytes": P.read_json(unit124 / "unit_manifest.json")["checkpoint_bytes"],
            **state_fields(summary, contract, "124m"),
        })
        unit1b = run_dir / "formal/1b" / f"seed{seed}"
        unit_manifest = P.read_json(
            unit1b / ("non10b_manifest.json" if scope == "non10b" else "unit_manifest.json")
        )
        phase_by_budget = {phase["budget_id"]: phase for phase in P.endpoint_phases(contract)}
        budgets = tuple(BUDGET_PHASES)
        for budget in budgets:
            phase_ids = BUDGET_PHASES[budget]
            phase = phase_by_budget[budget]
            directory = unit1b / phase["id"]
            summary = P.read_json(directory / "summary.json")
            tail, auc = curve_stats(validation_rows([unit1b / phase_id / "metrics.csv" for phase_id in phase_ids]))
            checkpoint = unit_manifest["endpoints"][budget]
            rows.append({
                "scale": "1b", "budget_id": budget, "target_step": phase["target_step"],
                "seed": seed, "method": "moonlight", "final_val_loss": float(summary["final_val_loss"]),
                "tail5_val_loss": tail, "normalized_val_auc": auc,
                "optimizer_state_bytes": int(summary["optimizer_state_bytes"]),
                "peak_allocated_bytes": int(summary["peak_allocated_bytes"]),
                "checkpoint_path": checkpoint["path"], "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_bytes": checkpoint["bytes"],
                **state_fields(summary, contract, "1b"),
            })
    return rows


def aggregate(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 4.302652729911275 * sd / math.sqrt(len(values)) if len(values) == 3 else float("nan")
    return {"mean": mean, "sample_sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def paired_rows(
    endpoints: list[dict[str, Any]],
    controls: list[dict[str, str]],
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    control_index = {
        (row["scale"], row["budget_id"], row["method"], int(row["seed"])): row
        for row in controls
    }
    rows: list[dict[str, Any]] = []
    comparators = contract["formal"]["secondary_comparators"] + [
        contract["formal"]["primary_comparator"]
    ]
    for endpoint in endpoints:
        for comparator in comparators:
            control = control_index[
                (
                    endpoint["scale"], endpoint["budget_id"], comparator,
                    int(endpoint["seed"]),
                )
            ]
            rows.append({
                "scale": endpoint["scale"], "budget_id": endpoint["budget_id"],
                "seed": endpoint["seed"], "comparator": comparator,
                "moonlight_final_val_loss": endpoint["final_val_loss"],
                "comparator_final_val_loss": control["final_val_loss"],
                "delta_moonlight_minus_comparator": float(endpoint["final_val_loss"])
                - float(control["final_val_loss"]),
            })
    return rows


def contrast_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in seed_rows:
        grouped.setdefault(
            (row["scale"], row["budget_id"], row["comparator"]), []
        ).append(float(row["delta_moonlight_minus_comparator"]))
    rows: list[dict[str, Any]] = []
    for (scale, budget, comparator), values in sorted(grouped.items()):
        stats = aggregate(values)
        rows.append({
            "scale": scale, "budget_id": budget, "comparator": comparator,
            "n": len(values),
            "mean_delta_moonlight_minus_comparator": stats["mean"],
            "sample_sd": stats["sample_sd"], "ci95_low": stats["ci95_low"],
            "ci95_high": stats["ci95_high"],
            "moonlight_better_seed_count": sum(value < 0 for value in values),
            "moonlight_worse_seed_count": sum(value > 0 for value in values),
        })
    return rows


def build(run_dir: Path, scope: str = "full") -> None:
    snapshot = run_dir / "source_snapshot"
    contract_path = snapshot / PACKAGE_REL / "ex54_contract.json"
    contract = P.read_json(contract_path)
    P.assert_contract(contract)
    formal = P.read_json(
        run_dir / "formal" /
        ("formal_non10b_manifest.json" if scope == "non10b" else "formal_manifest.json")
    )
    if formal.get("passed") is not True or len(formal.get("units", [])) != 6:
        raise RuntimeError("EX54 analysis requires six passed formal units")
    controls_path = snapshot / PACKAGE_REL / contract["controls"]["path"]
    if P.sha256_file(controls_path) != contract["controls"]["sha256"]:
        raise RuntimeError("EX54 frozen controls changed")
    controls = read_csv(controls_path)
    if len(controls) != int(contract["controls"]["rows"]):
        raise RuntimeError("EX54 frozen control row count changed")
    endpoints = collect_endpoints(run_dir, contract, scope=scope)
    expected_endpoint_rows = 9
    if len(endpoints) != expected_endpoint_rows:
        raise RuntimeError(
            f"EX54 expected {expected_endpoint_rows} endpoint rows, observed {len(endpoints)}"
        )
    endpoint_fields = [
        "scale", "budget_id", "target_step", "seed", "method", "final_val_loss",
        "tail5_val_loss", "normalized_val_auc", "optimizer_state_bytes", "peak_allocated_bytes",
        "moonlight_momentum_state_bytes", "moonlight_matrix_optimizer_state_bytes",
        "checkpoint_path", "checkpoint_sha256", "checkpoint_bytes",
    ]
    analysis = run_dir / "analysis"
    endpoint_path = analysis / "endpoint_results.csv"
    write_csv(endpoint_path, endpoints, endpoint_fields)

    seed_rows = paired_rows(endpoints, controls, contract)
    seed_path = analysis / "paired_seed_deltas.csv"
    write_csv(seed_path, seed_rows, ["scale", "budget_id", "seed", "comparator", "moonlight_final_val_loss", "comparator_final_val_loss", "delta_moonlight_minus_comparator"])
    contrasts = contrast_rows(seed_rows)
    contrast_path = analysis / "paired_contrasts.csv"
    write_csv(contrast_path, contrasts, ["scale", "budget_id", "comparator", "n", "mean_delta_moonlight_minus_comparator", "sample_sd", "ci95_low", "ci95_high", "moonlight_better_seed_count", "moonlight_worse_seed_count"])
    primary = [row for row in contrasts if row["comparator"] == "muon"]
    manifest = {
        "schema_version": "ex54_moonlight_analysis_manifest_v1", "passed": True, "scope": scope,
        "classification": "moonlight_positioned_against_accepted_llama_controls",
        "endpoint_rows": len(endpoints), "paired_seed_rows": len(seed_rows), "aggregate_contrasts": len(contrasts),
        "primary_muon_contrasts": primary, "practical_margin": contract["analysis"]["practical_loss_margin"],
        "timing_eligible": False, "selection_sha256": formal["selection_sha256"],
        "files": {
            "endpoint_results.csv": P.sha256_file(endpoint_path),
            "paired_seed_deltas.csv": P.sha256_file(seed_path),
            "paired_contrasts.csv": P.sha256_file(contrast_path),
            "frozen_controls.csv": P.sha256_file(controls_path),
        },
    }
    P.atomic_json(analysis / "analysis_manifest.json", manifest)
    print(f"EX54 analysis passed=True endpoints={len(endpoints)}")


def rows_equivalent(
    observed: list[dict[str, str]], expected: list[dict[str, Any]]
) -> bool:
    if len(observed) != len(expected):
        return False
    try:
        for actual, wanted in zip(observed, expected, strict=True):
            if set(actual) != set(wanted):
                return False
            for key, value in wanted.items():
                if isinstance(value, bool):
                    if actual[key] != str(value):
                        return False
                elif isinstance(value, int):
                    if int(actual[key]) != value:
                        return False
                elif isinstance(value, float):
                    if not math.isclose(
                        float(actual[key]), value, rel_tol=1e-12, abs_tol=1e-12
                    ):
                        return False
                elif actual[key] != str(value):
                    return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def verify(run_dir: Path, full_hash: bool, scope: str = "full") -> None:
    snapshot = run_dir / "source_snapshot"
    contract = P.read_json(snapshot / PACKAGE_REL / "ex54_contract.json")
    P.assert_contract(contract)
    manifest_path = run_dir / "analysis/analysis_manifest.json"
    manifest = P.read_json(manifest_path)
    checks: dict[str, bool] = {
        "analysis_passed": manifest.get("passed") is True,
        "analysis_scope": manifest.get("scope", "full") == scope,
    }
    snapshot_manifest = P.read_json(snapshot / "source_snapshot_manifest.json")
    checks["source_snapshot"] = bool(snapshot_manifest.get("files")) and all(
        (snapshot / relative).is_file()
        and P.sha256_file(snapshot / relative) == row["sha256"]
        for relative, row in snapshot_manifest.get("files", {}).items()
    )
    projection_path = (
        snapshot / PACKAGE_REL / contract["data"]["1b"]["accepted_projection_path"]
    )
    checks["accepted_ex48_data_projection_file"] = (
        projection_path.is_file()
        and P.sha256_file(projection_path)
        == contract["data"]["1b"]["accepted_projection_sha256"]
    )
    accepted_projection = (
        P.read_json(projection_path)
        if checks["accepted_ex48_data_projection_file"]
        else {}
    )
    projection_checks = P.validate_accepted_data_projection(
        accepted_projection, contract
    )
    checks["accepted_ex48_data_projection_contract"] = bool(projection_checks) and all(
        projection_checks.values()
    )
    for name, expected in manifest.get("files", {}).items():
        path = snapshot / PACKAGE_REL / name if name == "frozen_controls.csv" else run_dir / "analysis" / name
        checks[f"file:{name}"] = path.is_file() and P.sha256_file(path) == expected
    controls_path = snapshot / PACKAGE_REL / contract["controls"]["path"]
    controls = read_csv(controls_path)
    checks["frozen_controls"] = (
        P.sha256_file(controls_path) == contract["controls"]["sha256"]
        and len(controls) == int(contract["controls"]["rows"])
    )
    endpoints = read_csv(run_dir / "analysis/endpoint_results.csv")
    expected_endpoints = collect_endpoints(run_dir, contract, scope=scope)
    checks["endpoint_rows"] = rows_equivalent(endpoints, expected_endpoints)
    seed_rows = read_csv(run_dir / "analysis/paired_seed_deltas.csv")
    expected_seed_rows = paired_rows(expected_endpoints, controls, contract)
    checks["paired_seed_rows"] = rows_equivalent(seed_rows, expected_seed_rows)
    contrasts = read_csv(run_dir / "analysis/paired_contrasts.csv")
    checks["paired_contrasts"] = rows_equivalent(
        contrasts, contrast_rows(expected_seed_rows)
    )
    for index, row in enumerate(endpoints):
        checkpoint = Path(row["checkpoint_path"])
        checks[f"checkpoint_exists:{index}"] = checkpoint.is_file() and checkpoint.stat().st_size == int(row["checkpoint_bytes"])
        if full_hash and checks[f"checkpoint_exists:{index}"]:
            checks[f"checkpoint_sha:{index}"] = P.sha256_file(checkpoint) == row["checkpoint_sha256"]
    formal = P.read_json(
        run_dir / "formal" /
        ("formal_non10b_manifest.json" if scope == "non10b" else "formal_manifest.json")
    )
    tuning = P.read_json(run_dir / "tuning/tuning_manifest.json")
    selection_path = Path(tuning["selection_path"])
    selected_contract_path = Path(tuning["selected_contract_path"])
    checks["selection"] = (
        P.sha256_file(selection_path) == tuning["selection_sha256"]
        and P.sha256_file(selected_contract_path)
        == tuning["selected_contract_sha256"]
    )
    checks["formal"] = (
        formal.get("passed") is True
        and len(formal.get("units", [])) == 6
        and formal.get("selection_sha256") == tuning["selection_sha256"]
        and formal.get("selected_contract_sha256")
        == tuning["selected_contract_sha256"]
    )
    if scope == "non10b":
        checks["formal_non10b_boundary"] = (
            formal.get("experiment_scope_complete") is True
            and formal.get("independent_of_ex57") is True
            and formal.get("completed_1b_budget_ids")
            == ["tokens_3p2506b", "tokens_6p9694b"]
            and not (run_dir / "formal/formal_manifest.json").exists()
        )
    checks["tuning"] = tuning.get("passed") is True
    checks["tuning_formal_seed_isolation"] = (
        {int(contract["tuning"][scale]["seed"]) for scale in ("124m", "1b")}
        .isdisjoint({int(seed) for seed in contract["formal"]["seeds"]})
        and tuning.get("formal_outcomes_observed") is False
    )
    selected_contract = P.read_json(selected_contract_path)
    selection_payload = P.read_json(selection_path)
    checks["selected_contract_freeze"] = (
        selected_contract.get("selection_manifest_sha256")
        == tuning["selection_sha256"]
        and selected_contract.get("selected_configs")
        == {
            scale: selection_payload["scales"][scale]["selected_cell"]
            for scale in ("124m", "1b")
        }
        and selection_payload.get("scales") == tuning.get("selected")
    )
    pilot_path = run_dir / "tuning/engineering_pilot_manifest.json"
    checks["engineering_pilot"] = (
        pilot_path.is_file()
        and P.sha256_file(pilot_path) == tuning.get("engineering_pilot_sha256")
        and P.read_json(pilot_path).get("passed") is True
        and P.read_json(pilot_path).get("quality_eligible") is False
        and P.read_json(pilot_path).get("timing_eligible") is False
    )
    preflight = P.read_json(run_dir / "preflight/preflight_manifest.json")
    checks["preflight"] = preflight.get("passed") is True
    init_units = preflight.get("init_audit", {}).get("units", {})
    for scale in ("124m", "1b"):
        for seed in contract["formal"]["seeds"]:
            init_row = init_units.get(f"{scale}/seed{seed}", {})
            checks[f"preflight_init:{scale}:{seed}"] = (
                init_row.get("payload", {}).get("init_sha256")
                == contract["accepted_init_sha256"][scale][str(seed)]
                and bool(init_row.get("checks"))
                and all(init_row.get("checks", {}).values())
            )
    for scale in ("124m", "1b"):
        data_path = run_dir / "preflight" / f"data_{scale}.json"
        data_audit = P.read_json(data_path)
        metadata = P.verify_data_metadata(data_audit)
        checks[f"data_metadata:{scale}"] = (
            data_audit.get("passed") is True
            and bool(metadata)
            and all(metadata.values())
            and P.sha256_file(data_path)
            == preflight.get(f"data_{scale}_audit_sha256")
        )
        if scale == "1b":
            projected = P.project_data_inventory(data_audit["inventory"])
            checks["data_1b_matches_accepted_ex48_projection"] = (
                projected == accepted_projection.get("inventory")
                and P.canonical_sha256(projected)
                == contract["data"]["1b"]["accepted_projection_inventory_sha256"]
                and data_audit.get("accepted_projection_inventory_sha256")
                == contract["data"]["1b"]["accepted_projection_inventory_sha256"]
            )
        if full_hash:
            root = Path(data_audit["data_dir"])
            checks[f"data_full_hash:{scale}"] = all(
                (root / row["name"]).is_file()
                and P.sha256_file(root / row["name"]) == row["sha256"]
                for split in ("train", "validation")
                for row in data_audit["inventory"][split]
            )
    selection_sha = tuning["selection_sha256"]
    contract_sha = tuning["selected_contract_sha256"]
    data124_sha = P.read_json(run_dir / "preflight/data_124m.json")[
        "inventory_sha256"
    ]
    data1b_sha = P.read_json(run_dir / "preflight/data_1b.json")[
        "inventory_sha256"
    ]
    formal_rows = {
        (str(row["scale"]), int(row["seed"])): row
        for row in formal.get("units", [])
    }
    for seed in contract["formal"]["seeds"]:
        unit124 = run_dir / "formal/124m" / f"seed{seed}"
        unit124_manifest = P.read_json(unit124 / "unit_manifest.json")
        summary124 = P.read_json(unit124 / "summary.json")
        checks[f"lineage:124m:{seed}"] = (
            summary124.get("moonlight_selection_sha256") == selection_sha
            and summary124.get("moonlight_contract_sha256") == contract_sha
            and summary124.get("moonlight_data_inventory_sha256") == data124_sha
            and summary124.get("init_sha256")
            == contract["accepted_init_sha256"]["124m"][str(seed)]
        )
        checkpoint124 = unit124 / "checkpoint_latest.pt"
        checks[f"unit_integrity:124m:{seed}"] = (
            formal_rows.get(("124m", int(seed))) == unit124_manifest
            and P.sha256_file(unit124 / "summary.json")
            == unit124_manifest.get("summary_sha256")
            and P.sha256_file(unit124 / "metrics.csv")
            == unit124_manifest.get("metrics_sha256")
            and checkpoint124.is_file()
            and checkpoint124.stat().st_size
            == int(unit124_manifest.get("checkpoint_bytes", -1))
            and (not full_hash or P.sha256_file(checkpoint124)
                 == unit124_manifest.get("checkpoint_sha256"))
        )
        unit1b = run_dir / "formal/1b" / f"seed{seed}"
        unit1b_manifest = P.read_json(
            unit1b / ("non10b_manifest.json" if scope == "non10b" else "unit_manifest.json")
        )
        checks[f"unit_manifest:1b:{seed}"] = (
            formal_rows.get(("1b", int(seed))) == unit1b_manifest
            and unit1b_manifest.get("selection_sha256") == selection_sha
            and unit1b_manifest.get("selected_contract_sha256") == contract_sha
            and unit1b_manifest.get("data_inventory_sha256") == data1b_sha
        )
        phase_rows = (
            [row for row in contract["phases"] if row["id"] in {
                "backbone_4400", "cooldown_6200", "backbone_11493", "cooldown_13293"
            }]
            if scope == "non10b" else contract["phases"]
        )
        for phase in phase_rows:
            directory = unit1b / phase["id"]
            phase_manifest = P.read_json(directory / "phase_manifest.json")
            phase_summary = P.read_json(directory / "summary.json")
            checks[f"lineage:1b:{seed}:{phase['id']}"] = (
                phase_manifest.get("contract_sha256") == contract_sha
                and phase_manifest.get("data_inventory_sha256") == data1b_sha
                and phase_summary.get("moonlight_selection_sha256") == selection_sha
                and phase_summary.get("contract_sha256") == contract_sha
                and phase_summary.get("data_inventory_sha256") == data1b_sha
                and phase_summary.get("init_sha256")
                == contract["accepted_init_sha256"]["1b"][str(seed)]
            )
            checkpoint = phase_manifest["checkpoint"]
            checkpoint_path = Path(checkpoint["path"])
            phase_hashes = (
                P.sha256_file(directory / "summary.json")
                == phase_manifest.get("summary_sha256")
                and P.sha256_file(directory / "metrics.csv")
                == phase_manifest.get("metrics_sha256")
                and P.sha256_file(directory / "phase_manifest.json")
                == unit1b_manifest.get("phases", {}).get(phase["id"], {}).get(
                    "manifest_sha256"
                )
                and P.sha256_file(directory / "summary.json")
                == unit1b_manifest.get("phases", {}).get(phase["id"], {}).get(
                    "summary_sha256"
                )
            )
            if phase["role"] == "primary_endpoint":
                checkpoint_gate = (
                    checkpoint_path.is_file()
                    and checkpoint_path.stat().st_size == int(checkpoint["bytes"])
                    and (not full_hash or P.sha256_file(checkpoint_path)
                         == checkpoint["sha256"])
                    and unit1b_manifest.get("endpoints", {}).get(
                        phase["budget_id"]
                    ) == checkpoint
                )
            else:
                retirement = P.read_json(directory / "checkpoint_retirement.json")
                checkpoint_gate = (
                    not checkpoint_path.exists()
                    and retirement.get("passed") is True
                    and retirement.get("sha256") == checkpoint["sha256"]
                    and int(retirement.get("bytes", -1)) == int(checkpoint["bytes"])
                    and retirement.get("children")
                    == P.direct_children(contract, phase["id"])
                )
            checks[f"phase_integrity:1b:{seed}:{phase['id']}"] = (
                phase_hashes and checkpoint_gate
            )
    payload = {
        "schema_version": "ex54_moonlight_independent_verify_v1", "scope": scope, "passed": all(checks.values()),
        "full_checkpoint_hash": full_hash, "checks": checks,
        "analysis_manifest_sha256": P.sha256_file(manifest_path),
    }
    P.atomic_json(run_dir / "analysis/verification_manifest.json", payload)
    print(json.dumps({"passed": payload["passed"], "checks": checks}, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if args.mode == "build":
        build(run_dir, args.scope)
    else:
        verify(run_dir, args.full_checkpoint_hash, args.scope)


if __name__ == "__main__":
    main()
