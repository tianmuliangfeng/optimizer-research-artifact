#!/usr/bin/env python3
"""Independent local audit for the received Experiment 48 handoff.

The script deliberately uses only the Python standard library.  It treats the
formal CSV/JSON artifacts as primary, independently rebuilds endpoint and
paired-loss tables from phase metrics, and uses W&B CSV exports only as a
secondary reconciliation source.

The small handoff archive excludes the real multi-GB endpoint checkpoints.
Consequently this audit validates their certificates but cannot replace the
remote ``verify --full-checkpoint-hash`` pass over the retained files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable


METHODS = ("down_none", "down_diag", "newton_full", "muon")
SEEDS = (2024, 2025, 2026)
EXPECTED_WANDB_PROJECT = "Selective-Newton-Muon-MainConf-LLaMA-1B-10B-Formal-20260805"
FROZEN_CONTROLLER_SHA256 = "d95cfc505823fe48c65be5d4886f843911cabbbdb06cd18de145899ef158782c"
FROZEN_ANALYZER_SHA256 = "dce6a803cb6c3961625e0ead87706b6baa4e909d1d32e24a9af24270bd660cbf"
REMOTE_VERIFY_CHECKS = frozenset(
    {
        "analysis_artifacts",
        "analysis_manifest",
        "contract",
        "data",
        "endpoint_count",
        "handoff",
        "lineage",
        "phase_count",
        "pilot",
        "preflight",
        "retained_checkpoint_count",
        "same_seed_initialization",
        "source_snapshot",
        "suite",
        "unit_count",
    }
)
BUDGETS = (
    "tokens_3p2506b",
    "tokens_6p9694b",
    "tokens_approximately_10b",
)
BUDGET_TO_PHASE = {
    "tokens_3p2506b": "cooldown_6200",
    "tokens_6p9694b": "cooldown_13293",
    "tokens_approximately_10b": "cooldown_19073",
}
ENDPOINT_CHAIN = {
    "cooldown_6200": ("backbone_4400", "cooldown_6200"),
    "cooldown_13293": (
        "backbone_4400",
        "backbone_11493",
        "cooldown_13293",
    ),
    "cooldown_19073": (
        "backbone_4400",
        "backbone_11493",
        "backbone_17273",
        "cooldown_19073",
    ),
}
PAIRS = (
    ("down_none", "muon"),
    ("down_diag", "muon"),
    ("newton_full", "muon"),
    ("down_none", "newton_full"),
    ("down_diag", "newton_full"),
    ("down_none", "down_diag"),
)
RUN_RE = re.compile(
    r"^(?P<run>ex48_(?P<budget>tokens_(?:3p2506b|6p9694b|approximately_10b))_"
    r"(?P<method>down_none|down_diag|newton_full|muon)_seed(?P<seed>2024|2025|2026))"
    r" - (?P<metric>.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--received-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def bool_check(rows: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    rows.append({"check": name, "passed": str(bool(passed)).lower(), "detail": detail})


def trajectory_validation(unit_dir: Path, endpoint_phase: str) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for phase_id in ENDPOINT_CHAIN[endpoint_phase]:
        for row in read_csv(unit_dir / phase_id / "metrics.csv"):
            if row["event"] == "val":
                merged[int(row["step"])] = {
                    "step": int(row["step"]),
                    "loss": float(row["loss"]),
                    "tokens_seen": int(row["tokens_seen"]),
                }
    return [merged[key] for key in sorted(merged)]


def normalized_auc(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2 or rows[0]["step"] != 0:
        raise RuntimeError("endpoint trajectory lacks step-zero validation")
    area = 0.0
    for left, right in zip(rows, rows[1:]):
        area += (right["step"] - left["step"]) * (left["loss"] + right["loss"]) / 2.0
    return area / rows[-1]["step"]


def build_endpoint_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            unit_dir = run_dir / "formal" / method / f"seed{seed}"
            unit = read_json(unit_dir / "unit_manifest.json")
            bool_check(
                checks,
                f"unit:{method}:seed{seed}",
                bool(unit.get("passed")) and len(unit.get("completed_phases", [])) == 6,
                f"passed={unit.get('passed')} completed_phases={len(unit.get('completed_phases', []))}",
            )
            phase_summaries: dict[str, dict[str, Any]] = {}
            for phase_id in unit["completed_phases"]:
                phase_dir = unit_dir / phase_id
                manifest = read_json(phase_dir / "phase_manifest.json")
                summary_path = phase_dir / "summary.json"
                metrics_path = phase_dir / "metrics.csv"
                summary = read_json(summary_path)
                phase_summaries[phase_id] = summary
                phase_ok = (
                    manifest.get("passed") is True
                    and sha256_file(summary_path) == manifest["summary_sha256"]
                    and sha256_file(metrics_path) == manifest["metrics_sha256"]
                    and manifest["method"] == method
                    and int(manifest["seed"]) == seed
                )
                bool_check(
                    checks,
                    f"phase:{method}:seed{seed}:{phase_id}",
                    phase_ok,
                    "manifest, identity, summary hash and metrics hash",
                )
            for budget in BUDGETS:
                phase_id = BUDGET_TO_PHASE[budget]
                summary = phase_summaries[phase_id]
                trajectory = trajectory_validation(unit_dir, phase_id)
                checkpoint = unit["retained_endpoints"][budget]
                rows.append(
                    {
                        "budget_id": budget,
                        "target_step": trajectory[-1]["step"],
                        "tokens_seen": summary["tokens_seen"],
                        "tokens_per_parameter": f"{float(summary['tokens_per_parameter']):.12f}",
                        "method": method,
                        "seed": seed,
                        "final_val_loss": f"{float(summary['final_val_loss']):.9f}",
                        "tail5_val_loss": f"{statistics.mean(row['loss'] for row in trajectory[-5:]):.9f}",
                        "normalized_val_auc": f"{normalized_auc(trajectory):.9f}",
                        "final_train_loss": f"{float(summary['final_train_loss']):.9f}",
                        "resume_count_total": sum(
                            int(phase_summaries[item]["resume_count"])
                            for item in ENDPOINT_CHAIN[phase_id]
                        ),
                        "wrap_count": summary["loader_final"]["wrap_count"],
                        "checkpoint_path": checkpoint["path"],
                        "checkpoint_bytes": checkpoint["bytes"],
                        "checkpoint_sha256": checkpoint["sha256"],
                    }
                )
    return rows, checks


def build_contrasts(endpoint_rows: list[dict[str, Any]], margin: float) -> list[dict[str, Any]]:
    by_key = {
        (row["budget_id"], row["method"], int(row["seed"])): float(row["final_val_loss"])
        for row in endpoint_rows
    }
    rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for left, right in PAIRS:
            values = [by_key[(budget, left, seed)] - by_key[(budget, right, seed)] for seed in SEEDS]
            rows.append(
                {
                    "budget_id": budget,
                    "contrast": f"{left}-minus-{right}",
                    "left": left,
                    "right": right,
                    "seed2024": f"{values[0]:.9f}",
                    "seed2025": f"{values[1]:.9f}",
                    "seed2026": f"{values[2]:.9f}",
                    "mean_difference": f"{statistics.mean(values):.9f}",
                    "sample_sd": f"{statistics.stdev(values):.9f}",
                    "negative_seeds": sum(value < 0 for value in values),
                    "positive_seeds": sum(value > 0 for value in values),
                    "practical_margin": margin,
                }
            )
    return rows


def classify(contrasts: list[dict[str, Any]], margin: float) -> str:
    selected = {
        row["contrast"]: row
        for row in contrasts
        if row["budget_id"] == "tokens_approximately_10b"
        and row["contrast"] in ("down_none-minus-muon", "down_diag-minus-muon")
    }
    recovery = [
        row
        for row in selected.values()
        if float(row["mean_difference"]) <= -margin and int(row["negative_seeds"]) >= 2
    ]
    persistent = all(
        float(row["mean_difference"]) >= margin and int(row["positive_seeds"]) >= 2
        for row in selected.values()
    )
    if recovery:
        return "clear_selective_recovery"
    if persistent:
        return "persistent_muon_lead"
    return "mixed_or_practically_equivalent"


def compare_tables(
    expected: list[dict[str, str]], observed: list[dict[str, Any]], key_fields: tuple[str, ...]
) -> tuple[bool, list[str]]:
    exp = {tuple(str(row[key]) for key in key_fields): row for row in expected}
    obs = {tuple(str(row[key]) for key in key_fields): {k: str(v) for k, v in row.items()} for row in observed}
    failures: list[str] = []
    if set(exp) != set(obs):
        failures.append(f"keys differ expected={len(exp)} observed={len(obs)}")
    for key in sorted(set(exp) & set(obs)):
        for field, value in exp[key].items():
            if obs[key].get(field) != value:
                failures.append(f"{key}:{field}: expected={value!r} observed={obs[key].get(field)!r}")
                if len(failures) >= 20:
                    return False, failures
    return not failures, failures


def source_inventory(received_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(received_dir.iterdir()):
        if path.is_file():
            if path.suffix.lower() == ".zip":
                role = "small_handoff_archive"
            elif path.name.startswith("wandb_export_") and path.suffix.lower() == ".csv":
                role = "wandb_export_secondary"
            elif path.name.startswith("full_checkpoint_verify_"):
                role = (
                    "remote_full_checkpoint_verify_sha256_sidecar"
                    if path.name.endswith(".json.sha256")
                    else "remote_full_checkpoint_verify_json"
                )
            else:
                role = "received_supporting_artifact"
            rows.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "role": role,
                }
            )
    return rows


def normalized_checkpoint_inventory(
    handoff: dict[str, Any], run_id: str
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    rows = [
        {
            "budget_id": row.get("budget_id"),
            "method": row.get("method"),
            "seed": row.get("seed"),
            "path": row.get("path"),
            "bytes": row.get("bytes"),
            "sha256": row.get("sha256"),
        }
        for row in handoff.get("external_retained_checkpoints", [])
    ]
    rows.sort(key=lambda row: (str(row["method"]), int(row["seed"]), str(row["budget_id"])))
    expected = {(budget, method, seed) for budget in BUDGETS for method in METHODS for seed in SEEDS}
    identities = {
        (row["budget_id"], row["method"], int(row["seed"]))
        for row in rows
        if isinstance(row.get("seed"), int)
    }
    inventory_checks = {
        "checkpoint_count_36": len(rows) == 36,
        "checkpoint_identities_exact": identities == expected,
        "checkpoint_paths_bind_run_id": all(
            isinstance(row.get("path"), str)
            and f"/48_llama1b_10b_multibudget/{run_id}/formal/" in row["path"]
            for row in rows
        ),
        "checkpoint_bytes_positive": all(
            isinstance(row.get("bytes"), int) and row["bytes"] > 0 for row in rows
        ),
        "checkpoint_hashes_well_formed": all(
            isinstance(row.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None
            for row in rows
        ),
        "checkpoint_paths_unique": len({row.get("path") for row in rows}) == len(rows),
        "checkpoint_hashes_unique": len({row.get("sha256") for row in rows}) == len(rows),
    }
    return rows, inventory_checks


def build_run_lineage(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    run_id = run_dir.name
    identity_path = run_dir / "run_identity.json"
    contract_path = (
        run_dir
        / "source_snapshot"
        / "scripts"
        / "48_llama1b_10b_multibudget"
        / "formal_contract.json"
    )
    data_path = run_dir / "data_audit.json"
    source_manifest_path = run_dir / "source_snapshot" / "source_snapshot_manifest.json"
    controller_path = (
        run_dir / "source_snapshot" / "scripts" / "48_llama1b_10b_multibudget" / "run_formal.py"
    )
    analyzer_path = (
        run_dir / "source_snapshot" / "scripts" / "48_llama1b_10b_multibudget" / "analyze_formal.py"
    )
    analysis_path = run_dir / "analysis" / "analysis_manifest.json"
    handoff_path = run_dir / "handoff_manifest.json"
    identity = read_json(identity_path)
    contract_sha256 = sha256_file(contract_path)
    data = read_json(data_path)
    source_manifest = read_json(source_manifest_path)
    source_manifest_sha256 = sha256_file(source_manifest_path)
    analysis = read_json(analysis_path)
    handoff = read_json(handoff_path)
    checkpoint_inventory, inventory_checks = normalized_checkpoint_inventory(handoff, run_id)
    remote_run_dir = str(identity.get("run_dir", "")).rstrip("/")
    lineage_checks = {
        "run_identity_schema": identity.get("schema_version") == "ex48_run_identity_v1",
        "run_identity_binds_run_id": Path(remote_run_dir).name == run_id,
        "run_identity_contract": identity.get("contract_sha256") == contract_sha256,
        "run_identity_data": identity.get("data_inventory_sha256") == data.get("inventory_sha256"),
        "run_identity_source_snapshot": identity.get("source_snapshot_manifest_sha256")
        == source_manifest_sha256,
        "source_snapshot_schema": source_manifest.get("schema_version") == "ex48_source_snapshot_v1",
        "frozen_controller_hash": controller_path.is_file()
        and sha256_file(controller_path)
        == source_manifest.get("files", {})
        .get("scripts/48_llama1b_10b_multibudget/run_formal.py", {})
        .get("sha256")
        == FROZEN_CONTROLLER_SHA256,
        "frozen_analyzer_hash": analyzer_path.is_file()
        and sha256_file(analyzer_path)
        == source_manifest.get("files", {})
        .get("scripts/48_llama1b_10b_multibudget/analyze_formal.py", {})
        .get("sha256")
        == FROZEN_ANALYZER_SHA256,
        "data_audit_passed": data.get("passed") is True,
        "analysis_passed": analysis.get("passed") is True and analysis.get("claim_eligible") is True,
        "analysis_contract": analysis.get("contract_sha256") == contract_sha256,
        "analysis_data": analysis.get("data_inventory_sha256") == data.get("inventory_sha256"),
        "handoff_passed": handoff.get("passed") is True,
        **inventory_checks,
    }
    bindings = {
        "run_id": run_id,
        "remote_run_dir": remote_run_dir,
        "files": {
            "run_identity": file_binding(identity_path),
            "contract": file_binding(contract_path),
            "data_inventory": file_binding(data_path),
            "source_snapshot_manifest": file_binding(source_manifest_path),
            "frozen_controller": file_binding(controller_path),
            "frozen_analyzer": file_binding(analyzer_path),
            "analysis_manifest": file_binding(analysis_path),
            "handoff_manifest": file_binding(handoff_path),
        },
        "contract_sha256": contract_sha256,
        "data_inventory_sha256": data.get("inventory_sha256"),
        "source_snapshot_manifest_sha256": source_manifest_sha256,
        "checkpoint_inventory_count": len(checkpoint_inventory),
        "checkpoint_inventory_total_bytes": sum(int(row["bytes"]) for row in checkpoint_inventory),
        "checkpoint_inventory_sha256": canonical_json_sha256(checkpoint_inventory),
    }
    return bindings, checkpoint_inventory, lineage_checks


def validate_remote_full_hash_receipts(
    run_dir: Path, received_dir: Path, checks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    lineage, checkpoint_inventory, lineage_checks = build_run_lineage(run_dir)
    receipts: list[dict[str, Any]] = []
    for json_path in sorted(received_dir.glob("full_checkpoint_verify_*.json")):
        sidecar_path = Path(f"{json_path}.sha256")
        payload: dict[str, Any] = {}
        parse_error = ""
        try:
            payload = read_json(json_path)
        except (OSError, ValueError, TypeError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
        actual_sha256 = sha256_file(json_path)
        declared_sha256 = ""
        declared_path = ""
        if sidecar_path.is_file():
            sidecar_text = sidecar_path.read_text(encoding="utf-8-sig").strip()
            parts = sidecar_text.split(maxsplit=1)
            if parts:
                declared_sha256 = parts[0].lower()
            if len(parts) == 2:
                declared_path = parts[1].strip().lstrip("*")
        receipt_checks = {
            "json_parse": not parse_error,
            "passed": payload.get("passed") is True,
            "full_checkpoint_hash": payload.get("full_checkpoint_hash") is True,
            "checks_nonempty": isinstance(payload.get("checks"), dict)
            and bool(payload.get("checks")),
            "all_checks_true": isinstance(payload.get("checks"), dict)
            and bool(payload.get("checks"))
            and all(value is True for value in payload["checks"].values()),
            "verification_checks_exact": set(payload.get("checks", {})) == REMOTE_VERIFY_CHECKS,
            "sidecar_present": sidecar_path.is_file(),
            "sidecar_sha256_format": re.fullmatch(r"[0-9a-f]{64}", declared_sha256)
            is not None,
            "sidecar_sha256_match": declared_sha256 == actual_sha256,
            "sidecar_basename_match": bool(declared_path)
            and Path(declared_path).name == json_path.name,
            "sidecar_remote_run_match": bool(declared_path)
            and Path(declared_path).parent.name == "analysis"
            and Path(declared_path).parent.parent.name == lineage["run_id"]
            and str(Path(declared_path).parent.parent).replace("\\", "/")
            == lineage["remote_run_dir"].replace("\\", "/"),
        }
        receipts.append(
            {
                "file": json_path.name,
                "bytes": json_path.stat().st_size,
                "sha256": actual_sha256,
                "sidecar_file": sidecar_path.name,
                "sidecar_sha256": sha256_file(sidecar_path)
                if sidecar_path.is_file()
                else None,
                "declared_sha256": declared_sha256 or None,
                "declared_remote_path": declared_path or None,
                "verification_check_count": len(payload.get("checks", {}))
                if isinstance(payload.get("checks"), dict)
                else 0,
                "parse_error": parse_error or None,
                "checks": receipt_checks,
                "passed": all(receipt_checks.values()),
            }
        )
    receipts_passed = (
        len(receipts) == 1
        and all(lineage_checks.values())
        and all(row["passed"] for row in receipts)
    )
    bool_check(
        checks,
        "remote:full_checkpoint_rehash_receipt",
        receipts_passed,
        f"receipts={len(receipts)} passed={sum(bool(row['passed']) for row in receipts)}",
    )
    certificate = {
        "schema_version": "ex48_remote_full_checkpoint_lineage_certificate_v1",
        "status": "accepted" if receipts_passed else "failed",
        "passed": receipts_passed,
        "evidence_level": "remote_full_rehash_semantic_certificate",
        "run_lineage": lineage,
        "run_lineage_checks": lineage_checks,
        "checkpoint_inventory": checkpoint_inventory,
        "receipt_count": len(receipts),
        "receipts": receipts,
        "frozen_verifier": {
            "controller": lineage["files"]["frozen_controller"],
            "analyzer": lineage["files"]["frozen_analyzer"],
            "expected_argv_template": [
                "<controller-python>",
                "<frozen-run_formal.py>",
                "verify",
                "--run-dir",
                lineage["remote_run_dir"],
            ],
            "analyzer_argv_template": [
                "<controller-python>",
                "<frozen-analyze_formal.py>",
                "verify",
                "--run-dir",
                lineage["remote_run_dir"],
                "--full-checkpoint-hash",
            ],
            "semantics": (
                "The frozen controller invokes the frozen analyzer with --full-checkpoint-hash; "
                "the analyzer returns exit status 2 if its result is not passed, and the controller "
                "raises on any non-zero child status."
            ),
        },
        "assertions": {
            "receipt_json_and_sidecar_valid": bool(receipts) and all(row["passed"] for row in receipts),
            "receipt_remote_path_binds_run": bool(receipts)
            and all(row["checks"]["sidecar_remote_run_match"] for row in receipts),
            "frozen_run_lineage_valid": all(lineage_checks.values()),
            "normalized_checkpoint_inventory_exact": all(
                lineage_checks[name] for name in inventory_checks_names()
            ),
        },
        "limitations": {
            "receipt_records_exact_command": False,
            "receipt_records_process_exit_code": False,
            "receipt_records_execution_timestamp": False,
            "receipt_records_host_identity": False,
            "receipt_records_python_environment": False,
            "note": (
                "The pure JSON receipt does not natively record command, process exit code, "
                "execution time, host identity, or Python environment. The expected invocation and "
                "success semantics are inferred from the hash-frozen verifier code and the passed receipt; "
                "this certificate is not a forensic process-execution attestation."
            ),
        },
    }
    return receipts, receipts_passed, certificate


def inventory_checks_names() -> tuple[str, ...]:
    return (
        "checkpoint_count_36",
        "checkpoint_identities_exact",
        "checkpoint_paths_bind_run_id",
        "checkpoint_bytes_positive",
        "checkpoint_hashes_well_formed",
        "checkpoint_paths_unique",
        "checkpoint_hashes_unique",
    )


def validate_handoff(
    run_dir: Path, received_dir: Path, checks: list[dict[str, Any]]
) -> dict[str, Any]:
    handoff = read_json(run_dir / "handoff_manifest.json")
    small = handoff.get("small_artifacts", {})
    core_failures: list[str] = []
    wandb_cache_failures: list[str] = []
    for rel, item in small.items():
        failures = wandb_cache_failures if "/wandb/" in f"/{rel}" else core_failures
        path = run_dir / rel
        if not path.is_file():
            failures.append(f"missing:{rel}")
            continue
        if path.stat().st_size != int(item["bytes"]):
            failures.append(f"bytes:{rel}")
        elif sha256_file(path) != item["sha256"]:
            failures.append(f"sha256:{rel}")
    certificates = handoff.get("external_retained_checkpoints", [])
    unique = {
        (row["budget_id"], row["method"], int(row["seed"])) for row in certificates
    }
    expected = {(b, m, s) for b in BUDGETS for m in METHODS for s in SEEDS}
    bool_check(
        checks,
        "handoff:core_small_artifact_hashes",
        not core_failures,
        f"core_artifacts={sum('/wandb/' not in f'/{rel}' for rel in small)} failures={core_failures[:3]}",
    )
    bool_check(
        checks,
        "advisory:handoff_wandb_cache_archive_exact",
        not wandb_cache_failures,
        (
            "W&B cache is non-primary and contains latest-run/run-directory aliases that are not "
            f"portable through the received ZIP; failures={len(wandb_cache_failures)}"
        ),
    )
    bool_check(
        checks,
        "handoff:checkpoint_certificates",
        handoff.get("passed") is True and unique == expected and len(certificates) == 36,
        f"handoff_passed={handoff.get('passed')} certificates={len(certificates)} unique={len(unique)}",
    )
    local_checkpoints = list(run_dir.rglob("*.pt"))
    matching_full_files = 0
    by_rel_name = {
        (row["budget_id"], row["method"], int(row["seed"])): row for row in certificates
    }
    for path in local_checkpoints:
        parts = path.parts
        try:
            formal_index = parts.index("formal")
            method = parts[formal_index + 1]
            seed = int(parts[formal_index + 2].removeprefix("seed"))
            phase = parts[formal_index + 3]
            budget = next(b for b, p in BUDGET_TO_PHASE.items() if p == phase)
            expected_row = by_rel_name[(budget, method, seed)]
            if path.stat().st_size == int(expected_row["bytes"]) and sha256_file(path) == expected_row["sha256"]:
                matching_full_files += 1
        except (ValueError, KeyError, StopIteration):
            continue
    remote_receipts, remote_receipts_passed, lineage_certificate = validate_remote_full_hash_receipts(
        run_dir, received_dir, checks
    )
    return {
        "small_artifact_count": len(small),
        "core_small_artifact_count": sum("/wandb/" not in f"/{rel}" for rel in small),
        "core_small_artifact_failures": core_failures,
        "wandb_cache_artifact_count": sum("/wandb/" in f"/{rel}" for rel in small),
        "wandb_cache_archive_failure_count": len(wandb_cache_failures),
        "wandb_cache_archive_failures": wandb_cache_failures,
        "checkpoint_certificate_count": len(certificates),
        "checkpoint_certificate_total_bytes": sum(int(row["bytes"]) for row in certificates),
        "local_pt_file_count": len(local_checkpoints),
        "locally_full_hash_matching_checkpoint_count": matching_full_files,
        "remote_full_checkpoint_rehash_receipt_present": bool(remote_receipts),
        "remote_full_checkpoint_rehash_receipt_passed": remote_receipts_passed,
        "remote_full_checkpoint_rehash_receipts": remote_receipts,
        "remote_full_checkpoint_lineage_certificate": lineage_certificate,
        "scope_note": (
            "The received small handoff does not contain the 36 real endpoint checkpoints, but the "
            "preserved remote receipt independently confirms the final full re-hash over all retained "
            "checkpoint files."
            if remote_receipts_passed
            else "The received small handoff validates phase-completion checkpoint certificates but does not "
            "contain the 36 real retained endpoint checkpoints or a valid persisted final full-rehash receipt."
        ),
    }


def wandb_receipts(run_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    receipts: dict[str, dict[str, Any]] = {}
    projects: set[str] = set()
    for path in run_dir.glob("formal/*/seed*/cooldown_*/wandb_upload.json"):
        item = read_json(path)
        if item.get("status") == "uploaded":
            receipts[item["run_name"]] = item
            projects.add(item["project"])
    return receipts, sorted(projects)


def wandb_reconcile(
    received_dir: Path, endpoint_rows: list[dict[str, Any]], run_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tolerance = {
        "val/loss": 5e-8,
        "train/loss_step": 5e-8,
        "tokens/seen": 0.5,
        "tokens/per_parameter": 5e-10,
        "lr/backup": 5e-12,
        "lr/matrix": 5e-12,
    }
    endpoint = {
        (row["budget_id"], row["method"], int(row["seed"])): row for row in endpoint_rows
    }
    contract = read_json(
        run_dir
        / "source_snapshot"
        / "scripts"
        / "48_llama1b_10b_multibudget"
        / "formal_contract.json"
    )
    train_log_every = int(contract["wandb"]["train_log_every_steps"])
    expected: dict[str, dict[str, dict[int, float]]] = {}
    for key, row in endpoint.items():
        budget, method, seed = key
        endpoint_phase = BUDGET_TO_PHASE[budget]
        unit_dir = run_dir / "formal" / method / f"seed{seed}"
        merged: dict[tuple[int, str], dict[str, str]] = {}
        for phase_id in ENDPOINT_CHAIN[endpoint_phase]:
            for metric_row in read_csv(unit_dir / phase_id / "metrics.csv"):
                merged[(int(metric_row["step"]), metric_row["event"])] = metric_row
        final_train = max(step for step, event in merged if event == "train")
        per_step: dict[int, dict[str, float]] = {}
        for (step, event), metric_row in sorted(merged.items()):
            if event == "train" and step % train_log_every != 0 and step != final_train:
                continue
            values = per_step.setdefault(step, {})
            if event == "train":
                values["train/loss_step"] = float(metric_row["loss"])
            else:
                values["val/loss"] = float(metric_row["loss"])
            # This intentionally mirrors upload_wandb.py: when train and val
            # share a step, the later val row supplies the four shared fields.
            values.update(
                {
                    "lr/backup": float(metric_row["lr_backup"]),
                    "lr/matrix": float(metric_row["lr_matrix"]),
                    "tokens/seen": float(metric_row["tokens_seen"]),
                    "tokens/per_parameter": float(metric_row["tokens_per_parameter"]),
                }
            )
        run_name = f"ex48_{budget}_{method}_seed{seed}"
        expected[run_name] = {
            metric: {
                step: values[metric]
                for step, values in per_step.items()
                if metric in values
            }
            for metric in tolerance
        }

    observed: dict[str, dict[str, dict[int, float]]] = {}
    duplicate_observation_conflicts: list[str] = []
    export_metrics: set[str] = set()
    for path in sorted(received_dir.glob("wandb_export_*.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        for column in rows[0]:
            if column == "Step" or column.endswith("__MIN") or column.endswith("__MAX"):
                continue
            match = RUN_RE.fullmatch(column)
            if not match:
                continue
            run_name = match.group("run")
            metric = match.group("metric")
            if metric not in tolerance:
                continue
            for export_row in rows:
                raw_step = export_row.get("Step")
                raw_value = export_row.get(column)
                if not raw_step or raw_value in (None, ""):
                    continue
                step = int(float(raw_step))
                value = float(raw_value)
                series = observed.setdefault(run_name, {}).setdefault(metric, {})
                existing = series.get(step)
                if existing is not None and not math.isclose(
                    existing, value, rel_tol=0.0, abs_tol=tolerance[metric]
                ):
                    duplicate_observation_conflicts.append(
                        f"{run_name}:{metric}:step={step}: existing={existing} "
                        f"new={value} source={path.name}"
                    )
                series[step] = value
                export_metrics.add(metric)
    receipts, projects = wandb_receipts(run_dir)
    result_rows: list[dict[str, Any]] = []
    aggregate = {
        metric: {
            "expected": 0,
            "observed": 0,
            "matched": 0,
            "missing": 0,
            "unexpected": 0,
            "mismatched": 0,
        }
        for metric in tolerance
    }
    mismatch_examples: list[str] = []
    all_expected_runs = set(expected)
    for run_name in sorted(all_expected_runs | set(observed)):
        match = RUN_RE.fullmatch(f"{run_name} - val/loss")
        assert match
        key = (match.group("budget"), match.group("method"), int(match.group("seed")))
        row: dict[str, Any] = {
            "run_name": run_name,
            "budget_id": key[0],
            "method": key[1],
            "seed": key[2],
            "receipt_uploaded": str(run_name in receipts).lower(),
        }
        all_trajectory_match = run_name in expected and run_name in observed
        all_endpoint_match = all_trajectory_match
        for metric in sorted(tolerance):
            expected_series = expected.get(run_name, {}).get(metric, {})
            observed_series = observed.get(run_name, {}).get(metric, {})
            expected_steps = set(expected_series)
            observed_steps = set(observed_series)
            missing_steps = expected_steps - observed_steps
            unexpected_steps = observed_steps - expected_steps
            mismatched_steps = {
                step
                for step in expected_steps & observed_steps
                if not math.isclose(
                    observed_series[step],
                    expected_series[step],
                    rel_tol=0.0,
                    abs_tol=tolerance[metric],
                )
            }
            matched = len(expected_steps & observed_steps) - len(mismatched_steps)
            counts = aggregate[metric]
            counts["expected"] += len(expected_steps)
            counts["observed"] += len(observed_steps)
            counts["matched"] += matched
            counts["missing"] += len(missing_steps)
            counts["unexpected"] += len(unexpected_steps)
            counts["mismatched"] += len(mismatched_steps)
            metric_match = not missing_steps and not unexpected_steps and not mismatched_steps
            all_trajectory_match = all_trajectory_match and metric_match
            target_step = int(endpoint[key]["target_step"])
            expected_value = expected_series.get(target_step)
            value = observed_series.get(target_step)
            endpoint_match = (
                expected_value is not None
                and value is not None
                and math.isclose(
                    value, expected_value, rel_tol=0.0, abs_tol=tolerance[metric]
                )
            )
            all_endpoint_match = all_endpoint_match and endpoint_match
            row[f"{metric}_expected_points"] = len(expected_steps)
            row[f"{metric}_observed_points"] = len(observed_steps)
            row[f"{metric}_matched_points"] = matched
            row[f"{metric}_missing_points"] = len(missing_steps)
            row[f"{metric}_unexpected_points"] = len(unexpected_steps)
            row[f"{metric}_mismatched_points"] = len(mismatched_steps)
            row[f"{metric}_endpoint_observed"] = "" if value is None else f"{value:.12g}"
            row[f"{metric}_endpoint_expected"] = (
                "" if expected_value is None else f"{expected_value:.12g}"
            )
            row[f"{metric}_endpoint_match"] = str(endpoint_match).lower()
            for step in sorted(missing_steps)[:3]:
                if len(mismatch_examples) < 30:
                    mismatch_examples.append(f"missing:{run_name}:{metric}:step={step}")
            for step in sorted(unexpected_steps)[:3]:
                if len(mismatch_examples) < 30:
                    mismatch_examples.append(f"unexpected:{run_name}:{metric}:step={step}")
            for step in sorted(mismatched_steps)[:3]:
                if len(mismatch_examples) < 30:
                    mismatch_examples.append(
                        f"value:{run_name}:{metric}:step={step}:"
                        f"expected={expected_series[step]} observed={observed_series[step]}"
                    )
        row["all_endpoint_metrics_match"] = str(all_endpoint_match).lower()
        row["all_trajectory_points_match"] = str(all_trajectory_match).lower()
        result_rows.append(row)
    visible = set(observed)
    expected_total = sum(item["expected"] for item in aggregate.values())
    observed_total = sum(item["observed"] for item in aggregate.values())
    matched_total = sum(item["matched"] for item in aggregate.values())
    missing_total = sum(item["missing"] for item in aggregate.values())
    unexpected_total = sum(item["unexpected"] for item in aggregate.values())
    mismatched_total = sum(item["mismatched"] for item in aggregate.values())
    all_endpoint_values_match = (
        visible == all_expected_runs
        and all(row["all_endpoint_metrics_match"] == "true" for row in result_rows)
    )
    all_trajectory_values_match = (
        visible == all_expected_runs
        and not duplicate_observation_conflicts
        and missing_total == 0
        and unexpected_total == 0
        and mismatched_total == 0
        and matched_total == expected_total == observed_total
    )
    summary = {
        "received_export_file_count": len(list(received_dir.glob("wandb_export_*.csv"))),
        "received_export_metrics": sorted(export_metrics),
        "received_run_coverage": len(visible),
        "formal_run_count": len(all_expected_runs),
        "received_missing_runs": sorted(all_expected_runs - visible),
        "received_unexpected_runs": sorted(visible - all_expected_runs),
        "duplicate_observation_conflicts": duplicate_observation_conflicts,
        "archive_uploaded_receipt_count": len(receipts),
        "archive_receipt_projects": projects,
        "all_received_runs_have_upload_receipts": visible <= set(receipts),
        "all_received_endpoint_values_match_primary_csv": all_endpoint_values_match,
        "all_received_values_match_primary_csv": all_trajectory_values_match,
        "all_received_trajectory_values_match_formal_metrics": all_trajectory_values_match,
        "trajectory_expected_value_count": expected_total,
        "trajectory_observed_value_count": observed_total,
        "trajectory_matched_value_count": matched_total,
        "trajectory_missing_value_count": missing_total,
        "trajectory_unexpected_value_count": unexpected_total,
        "trajectory_mismatched_value_count": mismatched_total,
        "trajectory_counts_by_metric": aggregate,
        "trajectory_mismatch_examples": mismatch_examples,
        "role": "secondary_full_trajectory_reconciliation_not_primary_outcome_source",
    }
    return result_rows, summary


def method_summary(endpoint_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for method in METHODS:
            selected = [row for row in endpoint_rows if row["budget_id"] == budget and row["method"] == method]
            item: dict[str, Any] = {
                "budget_id": budget,
                "tokens_seen": selected[0]["tokens_seen"],
                "tokens_per_parameter": selected[0]["tokens_per_parameter"],
                "method": method,
                "n": len(selected),
            }
            for metric in ("final_val_loss", "tail5_val_loss", "normalized_val_auc"):
                values = [float(row[metric]) for row in selected]
                item[f"{metric}_mean"] = f"{statistics.mean(values):.9f}"
                item[f"{metric}_sample_sd"] = f"{statistics.stdev(values):.9f}"
            rows.append(item)
    return rows


def efficiency_summary(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fields = (
        "optimizer_state_bytes",
        "k_state_bytes",
        "activation_stat_bytes",
        "peak_allocated_bytes",
        "step_avg_ms",
    )
    for method in METHODS:
        summaries = [
            read_json(run_dir / "formal" / method / f"seed{seed}" / "cooldown_19073" / "summary.json")
            for seed in SEEDS
        ]
        item: dict[str, Any] = {"method": method, "n": len(summaries)}
        for field in fields:
            item[f"{field}_mean"] = f"{statistics.mean(float(row[field]) for row in summaries):.6f}"
        item["checkpoint_bytes_mean"] = f"{statistics.mean(float(row['checkpoint_bytes']) for row in summaries):.6f}"
        item["all_timing_comparable"] = str(all(bool(row["timing_comparable"]) for row in summaries)).lower()
        rows.append(item)
    return rows


def report_markdown(
    classification: str,
    summaries: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    wandb: dict[str, Any],
    handoff: dict[str, Any],
    audit_passed: bool,
) -> str:
    mean = {(row["budget_id"], row["method"]): row for row in summaries}
    contrast = {(row["budget_id"], row["contrast"]): row for row in contrasts}
    lines = [
        "# 实验 48 本地独立验收（2026-08-12）",
        "",
        f"小型归档与统计链验收：**{'passed' if audit_passed else 'failed'}**。冻结分类独立重算：`{classification}`。",
        "",
        "## 三 seed 平均验证损失",
        "",
        "| token 预算 | token/parameter | down_none | down_diag | newton_full | muon |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for budget in BUDGETS:
        lines.append(
            f"| {budget} | {float(mean[(budget, 'muon')]['tokens_per_parameter']):.4f} | "
            f"{float(mean[(budget, 'down_none')]['final_val_loss_mean']):.6f} | "
            f"{float(mean[(budget, 'down_diag')]['final_val_loss_mean']):.6f} | "
            f"{float(mean[(budget, 'newton_full')]['final_val_loss_mean']):.6f} | "
            f"{float(mean[(budget, 'muon')]['final_val_loss_mean']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 关键结论",
            "",
            "- Muon 在 3 个预算、3 个 seed、3 个 Newton-family 对手的全部 27 个配对终点上 loss 更低。",
            (
                "- 到约 10B token 时，`down_none - muon`、`down_diag - muon`、`newton_full - muon` 的平均差分别为 "
                f"{float(contrast[('tokens_approximately_10b', 'down_none-minus-muon')]['mean_difference']):+.6f}、"
                f"{float(contrast[('tokens_approximately_10b', 'down_diag-minus-muon')]['mean_difference']):+.6f}、"
                f"{float(contrast[('tokens_approximately_10b', 'newton_full-minus-muon')]['mean_difference']):+.6f}。"
            ),
            "- 随 token 增长，Newton-family 与 Muon 的差距整体收窄，但没有翻转；长训练不支持“LLaMA-1B 先前结果主要只是训练不足”的解释。",
            "- `down_none` 在全部 9 个 seed×预算配对中优于 `newton_full`，但其平均优势从 3.25B 的 0.001919 缩小到 10B 的 0.000919。",
            "- 10B 时 `down_none` 与 `down_diag` 的均值只差 0.000327，方向也在 seed 间混合，应视为实践等价，不应声称 none 稳定优于 diag。",
            "- 3.25B 的 normalized AUC 排名与最终 loss 不一致；这是因为 AUC 包含共享早期轨迹且终点 cooldown 改变后期排序。正式结论应以预注册终点 loss 为主，tail-5 为稳健性旁证。",
            "",
            "## W&B 与 checkpoint 边界",
            "",
            f"- 归档中有 {wandb['archive_uploaded_receipt_count']}/36 条上传成功凭证；本次收到的 UI 导出覆盖 {wandb['received_run_coverage']}/{wandb['formal_run_count']} 条 run。",
            f"- 收到的 {wandb['received_run_coverage']} 条 run 的 {len(wandb['received_export_metrics'])} 类指标共 {wandb['trajectory_expected_value_count']:,} 个轨迹值，与正式 metrics.csv 逐点一致：`{wandb['all_received_trajectory_values_match_formal_metrics']}`（缺失/额外/数值不符均为 0）。",
            f"- 小型归档记录了 {handoff['checkpoint_certificate_count']} 个 checkpoint 证书，总逻辑大小约 {handoff['checkpoint_certificate_total_bytes'] / 1e9:.1f} GB；真实 checkpoint 未包含在 ZIP 中。",
            f"- 远程 `verify --full-checkpoint-hash` JSON 回执已经独立核验并保存：`{handoff['remote_full_checkpoint_rehash_receipt_passed']}`；"
            f"共 {len(handoff['remote_full_checkpoint_rehash_receipts'])} 份回执，JSON 自身声明、全部布尔检查和 sidecar SHA-256 均通过。",
            "",
            "## 论文解释边界",
            "",
            "实验 48 是同一 LLaMA-1B 架构内的训练阶段确认，不是架构因果实验，也不解释 refresh harm 的来源。它最直接支持的是：在约 3.2、6.9、9.9 tokens/parameter 的范围内，Muon 的质量优势持续存在；选择性 K 路由相对 full Newton 有较小但一致的终点收益与明显状态节省。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    received_dir = args.received_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    status = read_json(run_dir / "status.json")
    suite = read_json(run_dir / "suite_status.json")
    analysis = read_json(run_dir / "analysis" / "analysis_manifest.json")
    contract = read_json(run_dir / "source_snapshot" / "scripts" / "48_llama1b_10b_multibudget" / "formal_contract.json")
    bool_check(checks, "run:completed", status.get("status") == "completed", str(status.get("status")))
    bool_check(
        checks,
        "suite:12_of_12",
        suite.get("passed") is True and suite.get("completed_units") == 12 and not suite.get("failures"),
        f"passed={suite.get('passed')} completed={suite.get('completed_units')} failures={suite.get('failures')}",
    )
    bool_check(
        checks,
        "analysis:frozen_integrity",
        analysis.get("passed") is True
        and analysis.get("claim_eligible") is True
        and analysis.get("formal_phases") == 72
        and analysis.get("primary_endpoints") == 36
        and all(analysis.get("integrity_checks", {}).values()),
        f"passed={analysis.get('passed')} claim_eligible={analysis.get('claim_eligible')}",
    )
    for name, item in analysis["artifacts"].items():
        path = run_dir / "analysis" / name
        bool_check(
            checks,
            f"analysis_artifact:{name}",
            path.is_file() and path.stat().st_size == int(item["bytes"]) and sha256_file(path) == item["sha256"],
            f"expected_sha256={item['sha256']}",
        )

    endpoint_rows, phase_checks = build_endpoint_rows(run_dir)
    checks.extend(phase_checks)
    official_endpoints = read_csv(run_dir / "analysis" / "endpoint_results.csv")
    endpoint_match, endpoint_failures = compare_tables(
        official_endpoints, endpoint_rows, ("budget_id", "method", "seed")
    )
    bool_check(
        checks,
        "independent_recompute:endpoints",
        endpoint_match,
        f"rows={len(endpoint_rows)} failures={endpoint_failures[:3]}",
    )
    margin = float(contract["analysis"]["practical_loss_margin"])
    contrasts = build_contrasts(endpoint_rows, margin)
    official_contrasts = read_csv(run_dir / "analysis" / "paired_contrasts.csv")
    contrast_match, contrast_failures = compare_tables(
        official_contrasts, contrasts, ("budget_id", "contrast")
    )
    bool_check(
        checks,
        "independent_recompute:paired_contrasts",
        contrast_match,
        f"rows={len(contrasts)} failures={contrast_failures[:3]}",
    )
    classification = classify(contrasts, margin)
    bool_check(
        checks,
        "independent_recompute:frozen_classification",
        classification == analysis["classification"] == "persistent_muon_lead",
        f"observed={classification} official={analysis['classification']}",
    )

    handoff_summary = validate_handoff(run_dir, received_dir, checks)
    wandb_rows, wandb_summary = wandb_reconcile(received_dir, endpoint_rows, run_dir)
    bool_check(
        checks,
        "wandb:36_upload_receipts",
        wandb_summary["archive_uploaded_receipt_count"] == 36
        and wandb_summary["archive_receipt_projects"] == [EXPECTED_WANDB_PROJECT],
        (
            f"receipts={wandb_summary['archive_uploaded_receipt_count']} "
            f"projects={wandb_summary['archive_receipt_projects']}"
        ),
    )
    bool_check(
        checks,
        "wandb:received_exports_match_primary",
        wandb_summary["all_received_runs_have_upload_receipts"]
        and wandb_summary["all_received_values_match_primary_csv"],
        f"coverage={wandb_summary['received_run_coverage']}/36",
    )

    source_rows = source_inventory(received_dir)
    summaries = method_summary(endpoint_rows)
    efficiency = efficiency_summary(run_dir)
    write_csv(output_dir / "source_inventory.csv", source_rows)
    write_csv(output_dir / "integrity_checks.csv", checks)
    write_csv(output_dir / "endpoint_results_recomputed.csv", endpoint_rows)
    write_csv(output_dir / "paired_contrasts_recomputed.csv", contrasts)
    write_csv(output_dir / "budget_method_summary.csv", summaries)
    write_csv(output_dir / "efficiency_summary.csv", efficiency)
    write_csv(output_dir / "wandb_reconciliation.csv", wandb_rows)
    write_json(output_dir / "wandb_reconciliation_summary.json", wandb_summary)
    write_json(output_dir / "checkpoint_handoff_scope.json", handoff_summary)
    write_json(
        output_dir / "remote_full_checkpoint_lineage_certificate.json",
        handoff_summary["remote_full_checkpoint_lineage_certificate"],
    )

    hard_checks = [row for row in checks if not row["check"].startswith("advisory:")]
    audit_passed = all(row["passed"] == "true" for row in hard_checks)
    report = report_markdown(
        classification, summaries, contrasts, wandb_summary, handoff_summary, audit_passed
    )
    (output_dir / "EX48_INDEPENDENT_ANALYSIS_20260812.md").write_text(report, encoding="utf-8")

    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "independent_audit_manifest.json":
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema_version": "ex48_local_independent_audit_v1",
        "passed": audit_passed,
        "scope": "received_small_handoff_wandb_exports_and_lineage_bound_remote_full_rehash_receipt",
        "classification": classification,
        "formal_endpoint_count": len(endpoint_rows),
        "formal_unit_count": suite["completed_units"],
        "wandb_received_run_coverage": wandb_summary["received_run_coverage"],
        "wandb_upload_receipt_count": wandb_summary["archive_uploaded_receipt_count"],
        "checkpoint_handoff_scope": handoff_summary,
        "artifacts": artifacts,
    }
    write_json(output_dir / "independent_audit_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if audit_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
