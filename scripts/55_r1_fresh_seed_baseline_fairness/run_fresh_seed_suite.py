#!/usr/bin/env python3
"""Staged, resumable controller for Experiment 55."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-08-19.1"
EXPERIMENT = "55_r1_fresh_seed_baseline_fairness"
# Marker consumed by the wrapper so old source snapshots use the live amended
# controller while newly created snapshots remain fully snapshot-driven.
FORMAL_METRICS_TAIL5_LINEAGE = True
ENGINEERING_SEED = 5501
FORMAL_SEED = 2027
SMOKE_STEPS = 34
FORMAL_VALIDATION_TOKENS = 10_485_760
# EX15's numerical-smoke source deliberately reduces validation to one device
# batch.  The other frozen runners' formal-smoke modes retain the full formal
# validation budget.  These smoke losses are therefore comparable only within
# a validation-token stratum; all 6200-step formal runs use the full budget.
SMOKE_VALIDATION_TOKENS = {
    "core": 64 * 1024,
    "extended": FORMAL_VALIDATION_TOKENS,
    "mousse": FORMAL_VALIDATION_TOKENS,
    "malt": FORMAL_VALIDATION_TOKENS,
    "malter_eq17": FORMAL_VALIDATION_TOKENS,
}
STAGES = ("preflight", "pilot", "formal", "verify", "all")
ACCEPTED_STATUS_PREFIXES = ("completed_valid",)
GROUPS = {
    "core": ("block4", "diag", "none", "muon"),
    "extended": ("adamw", "normuon", "moonlight"),
    "mousse": ("mousse",),
    "malt": ("malt",),
    "malter_eq17": ("malter_eq17",),
}
SELECTED_CELLS = {
    "block4": "block4", "diag": "diag", "none": "none", "muon": "muon",
    "adamw": "adamw_low", "normuon": "normuon_r1scale",
    "moonlight": "moonlight_r1scale", "mousse": "mousse_lr100",
    "malt": "malt_lr0125", "malter_eq17": "malter_eq17_lr015",
}
# EX55 uses paper-facing canonical labels.  EX19's frozen worker predates that
# normalization and exposes Moonlight as `moonlight_muon` on its CLI/manifests.
# This interface-only map must never alter the selected cell or analysis label.
WORKER_METHOD_LABELS = {
    "block4": "block4", "diag": "diag", "none": "none", "muon": "muon",
    "adamw": "adamw", "normuon": "normuon", "moonlight": "moonlight_muon",
    "mousse": "mousse", "malt": "malt", "malter_eq17": "malter_eq17",
}
AGGREGATE_STEMS = {
    "pilot": "pilot_smoke",
    "formal_smoke": "formal_smoke",
    "formal": "formal",
}
_DATA_INVENTORY_CACHE: dict[Path, Path] = {}
_DATA_INVENTORY_STAT_SIGNATURES: dict[Path, tuple[tuple[str, int, int], ...]] = {}
EXPECTED_MALT_PILOT_SOURCE_MANIFESTS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--training-python", required=True)
    parser.add_argument("--gpus", nargs="+", default=["1"])
    parser.add_argument("--historical-panel", type=Path, required=True)
    parser.add_argument("--extended-selection", type=Path, required=True)
    parser.add_argument("--mousse-selection", type=Path, required=True)
    parser.add_argument("--malt-selection", type=Path, required=True)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-project", default="Selective-Curvature-State-EX55-FreshSeed-20260817")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.gpus != ["1"]:
        parser.error("EX55 is frozen to --gpus 1")
    return args


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"expected file is absent: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def append_command(run_dir: Path, payload: dict[str, Any]) -> None:
    path = run_dir / "commands.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def copy_tree(source: Path, target: Path) -> None:
    if not target.exists():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))


def verify_source_bindings(repo: Path, contract: dict[str, Any]) -> None:
    failures = []
    for relative, expected in contract["source_bindings"].items():
        path = repo / relative
        observed = sha256_file(path) if path.is_file() else None
        if observed != expected:
            failures.append({"path": relative, "expected": expected, "observed": observed})
    if failures:
        raise RuntimeError(f"EX55 frozen source binding failed: {failures}")


def ensure_snapshot(args: argparse.Namespace) -> Path:
    snapshot = args.run_dir / "source_snapshot"
    manifest = snapshot / "source_snapshot_manifest.json"
    if manifest.is_file():
        payload = read_json(manifest)
        for record in payload.get("files", []):
            path = snapshot / record["path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"source snapshot drift: {path}")
        if payload.get("passed") is not True:
            raise RuntimeError("existing source snapshot is not accepted")
        return snapshot
    contract = read_json(args.repo / "scripts/55_r1_fresh_seed_baseline_fairness/ex55_contract.json")
    verify_source_bindings(args.repo, contract)
    for name in (
        "14_official_newton_muon_r0", "15_official_newton_muon_r1",
        "19_r1_extended_baselines", "45_r1_mousse_strong_baseline",
        "49_r1_malt_strong_baseline", "55_r1_fresh_seed_baseline_fairness", "_shared",
    ):
        copy_tree(args.repo / "scripts" / name, snapshot / "scripts" / name)
    wrapper = args.repo / "commands/55_r1_fresh_seed_baseline_fairness/20260817_ex55_r1_fresh_seed_baseline_fairness.sh"
    (snapshot / "commands").mkdir(parents=True, exist_ok=True)
    shutil.copy2(wrapper, snapshot / "commands" / wrapper.name)
    files = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and path != manifest:
            files.append({
                "path": path.relative_to(snapshot).as_posix(), "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    write_json(manifest, {
        "schema_version": 1, "experiment_id": EXPERIMENT, "passed": True,
        "created_at": datetime.now().astimezone().isoformat(), "file_count": len(files), "files": files,
    })
    return snapshot


def ensure_controller_amendment(args: argparse.Namespace) -> dict[str, Any] | None:
    """Bind a schema-compatibility controller fix to an existing frozen run.

    The accepted smoke artifacts, workers, frozen winners, seeds, and training
    contract are unchanged.  Only the extraction of step-0 validation evidence
    from heterogeneous upstream runner schemas and its validation-token-stratum
    interpretation are amended.
    """
    snapshot_controller = (
        args.run_dir / "source_snapshot/scripts/55_r1_fresh_seed_baseline_fairness/"
        "run_fresh_seed_suite.py"
    )
    live_controller = Path(__file__).resolve()
    if not snapshot_controller.is_file() or live_controller == snapshot_controller.resolve():
        return None
    amended_controller_sha256 = sha256_file(live_controller)
    # Name amendments by content instead of overwriting the first repair
    # receipt.  This run may already contain the earlier aggregate-schema
    # compatibility amendment; the filename-schema follow-up must extend that
    # audit trail without invalidating it.
    path = args.run_dir / (
        f"controller_amendment_pairing_schema_{amended_controller_sha256[:16]}.json"
    )
    predecessors = [
        {
            "path": str(candidate),
            "sha256": sha256_file(candidate),
        }
        for candidate in sorted(args.run_dir.glob("controller_amendment_pairing_schema*.json"))
        if candidate.resolve() != path.resolve()
    ]
    payload = {
        "schema_version": "ex55_controller_amendment_v2",
        "experiment_id": EXPERIMENT,
        "passed": True,
        "receipt_path": str(path),
        "created_at": datetime.now().astimezone().isoformat(),
        "reason": (
            "read_step0_validation_from_hash_bound_child_metrics_across_accepted_"
            "aggregate_and_artifact_filename_schemas_and_compare_smoke_losses_"
            "only_within_validation_token_strata"
        ),
        "scientific_training_contract_changed": False,
        "outcome_based_change": False,
        "method_or_hyperparameter_changed": False,
        "seed_changed": False,
        "accepted_smoke_artifacts_reused": True,
        "original_source_snapshot_manifest_sha256": sha256_file(
            args.run_dir / "source_snapshot/source_snapshot_manifest.json"
        ),
        "original_controller_sha256": sha256_file(snapshot_controller),
        "amended_controller_sha256": amended_controller_sha256,
        "predecessor_amendments": predecessors,
    }
    if path.is_file():
        existing = read_json(path)
        for key, value in payload.items():
            if key != "created_at" and existing.get(key) != value:
                raise RuntimeError("EX55 controller amendment changed across resume")
        return existing
    write_json(path, payload)
    return payload


def frozen_paths(args: argparse.Namespace) -> dict[str, Path]:
    snapshot = ensure_snapshot(args)
    scripts = snapshot / "scripts"
    return {
        "contract": scripts / "55_r1_fresh_seed_baseline_fairness/ex55_contract.json",
        "analyzer": scripts / "55_r1_fresh_seed_baseline_fairness/analyze_fresh_seed_panel.py",
        "core": scripts / "15_official_newton_muon_r1/run_official_newton_muon_r1.py",
        "extended": scripts / "19_r1_extended_baselines/run_r1_extended_baselines.py",
        "mousse": scripts / "45_r1_mousse_strong_baseline/run_r1_mousse.py",
        "malt": scripts / "49_r1_malt_strong_baseline/run_r1_malt.py",
    }


def resolve_analysis_analyzer(
    args: argparse.Namespace, snapshot_analyzer: Path,
) -> tuple[Path, Path | None]:
    """Select the analyzer without mutating the immutable source snapshot.

    Existing EX55 runs may contain the pre-repair analyzer in their snapshot.
    In that case verification uses the live amended analyzer and writes a
    hash-bound amendment receipt.  Fresh runs whose snapshot already contains
    the repair continue to use the snapshot copy.
    """

    live_analyzer = (
        args.repo / "scripts/55_r1_fresh_seed_baseline_fairness/analyze_fresh_seed_panel.py"
    ).expanduser().resolve()
    snapshot_analyzer = snapshot_analyzer.expanduser().resolve()
    if not snapshot_analyzer.is_file():
        raise RuntimeError(f"snapshot analyzer is absent: {snapshot_analyzer}")
    if not live_analyzer.is_file():
        raise RuntimeError(f"live analyzer is absent: {live_analyzer}")
    if sha256_file(live_analyzer) == sha256_file(snapshot_analyzer):
        return snapshot_analyzer, None
    marker = "FORMAL_METRICS_TAIL5_LINEAGE = True"
    if marker not in live_analyzer.read_text(encoding="utf-8"):
        raise RuntimeError(
            "live analyzer differs from the snapshot but does not expose the "
            "formal-metrics tail-5 compatibility marker"
        )

    amended_sha256 = sha256_file(live_analyzer)
    receipt_path = args.run_dir / f"analysis_amendment_tail5_metrics_{amended_sha256[:16]}.json"
    payload = {
        "schema_version": "ex55_analysis_amendment_v1",
        "experiment_id": EXPERIMENT,
        "passed": True,
        "created_at": datetime.now().astimezone().isoformat(),
        "reason": "reconstruct_tail5_from_hash_bound_accepted_formal_child_metrics",
        "scientific_training_contract_changed": False,
        "outcome_based_change": False,
        "method_or_hyperparameter_changed": False,
        "seed_changed": False,
        "formal_training_rerun_required": False,
        "accepted_formal_artifacts_reused": True,
        "aggregate_manifests_modified": False,
        "metrics_files_modified": False,
        "source_snapshot_manifest": file_record(
            args.run_dir / "source_snapshot/source_snapshot_manifest.json"
        ),
        "original_analyzer": file_record(snapshot_analyzer),
        "amended_analyzer": file_record(live_analyzer),
        "receipt_path": str(receipt_path.resolve()),
    }
    if receipt_path.is_file():
        existing = read_json(receipt_path)
        for key, value in payload.items():
            if key != "created_at" and existing.get(key) != value:
                raise RuntimeError("EX55 analysis amendment changed across verify/resume")
        return live_analyzer, receipt_path.resolve()
    write_json(receipt_path, payload)
    return live_analyzer, receipt_path.resolve()


def prepare_accepted_inputs(args: argparse.Namespace) -> dict[str, Path]:
    contract = read_json(frozen_paths(args)["contract"])
    specs = {
        "historical_panel": (args.historical_panel, contract["accepted_inputs"]["historical_panel_sha256"], "historical_panel.csv"),
        "extended_selection": (args.extended_selection, contract["accepted_inputs"]["extended_selection_sha256"], "extended_selection.csv"),
        "mousse_selection": (args.mousse_selection, contract["accepted_inputs"]["mousse_selection_sha256"], "mousse_selection.json"),
        "malt_selection": (args.malt_selection, contract["accepted_inputs"]["malt_selection_sha256"], "malt_selection.json"),
    }
    output: dict[str, Path] = {}
    target_root = args.run_dir / "accepted_inputs"
    target_root.mkdir(parents=True, exist_ok=True)
    for label, (source, expected, target_name) in specs.items():
        target = target_root / target_name
        if target.is_file():
            if sha256_file(target) != expected:
                raise RuntimeError(f"run-local accepted input drift: {target}")
            output[label] = target
            continue
        source = source.expanduser().resolve()
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"{label} is missing or has the wrong frozen SHA-256: {source}")
        if not target.exists():
            shutil.copy2(source, target)
        if sha256_file(target) != expected:
            raise RuntimeError(f"copied accepted input drift: {target}")
        output[label] = target
    manifest_path = target_root / "accepted_inputs_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        records = manifest.get("files", {})
        for label, target in output.items():
            record = records.get(label, {}) if isinstance(records, dict) else {}
            if (
                Path(str(record.get("run_copy", ""))).resolve() != target.resolve()
                or record.get("sha256") != sha256_file(target)
            ):
                raise RuntimeError(f"run-local accepted input manifest drift: {label}")
    else:
        write_json(manifest_path, {
            "schema_version": 1,
            "experiment_id": EXPERIMENT,
            "passed": True,
            "policy": "copy_once_then_resume_from_run_local_hash_bound_inputs",
            "files": {
                label: {
                    "source_at_preflight": str(specs[label][0].expanduser()),
                    "run_copy": str(target.resolve()),
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
                for label, target in output.items()
            },
        })
    return output


def audit_extended_selection(path: Path) -> dict[str, Any]:
    """Bind the three extended-baseline winners to their accepted pilot table."""
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    observed = {
        str(row.get("method", "")).strip().lower(): str(row.get("cell", "")).strip().lower()
        for row in rows
    }
    expected = {method: SELECTED_CELLS[method] for method in ("adamw", "normuon", "moonlight")}
    checks = {
        "exact_three_rows": len(rows) == 3,
        "exact_methods_and_cells": observed == expected,
        "all_advance": all(str(row.get("formal_seed2026_decision", "")).strip().lower() == "advance" for row in rows),
        "seed2026_pilot": all(int(row.get("seed", -1)) == 2026 for row in rows),
        "quality_usable": all(str(row.get("quality_usable", "")).strip().lower() == "true" for row in rows),
    }
    if not all(checks.values()):
        raise RuntimeError(f"accepted extended-baseline selection failed: {checks}")
    return {"passed": True, "checks": checks, "selected_cells": observed, "sha256": sha256_file(path)}


def audit_mousse_selection(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    checks = {
        "selected": payload.get("status") == "selected",
        "protocol": payload.get("protocol") == "mousse_r1_pilot_selection_v1",
        "seed_and_steps": payload.get("seed") == 2026 and payload.get("pilot_steps") == 1000,
        "selected_cell": payload.get("selected_cell_id") == SELECTED_CELLS["mousse"],
        "selected_lr": float(payload.get("selected_matrix_lr", -1)) == 0.015,
    }
    if not all(checks.values()):
        raise RuntimeError(f"accepted Mousse selection failed: {checks}")
    return {"passed": True, "checks": checks, "sha256": sha256_file(path)}


def audit_malt_winner_selection(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    selected = payload.get("selected_methods")
    selected = selected if isinstance(selected, dict) else {}
    checks = {
        "selected": payload.get("status") == "selected",
        "protocol": payload.get("protocol") == "malt_r1_focused_grid_selection_v4",
        "seed_and_steps": payload.get("seed") == 2026 and payload.get("pilot_steps") == 1000,
        "formal_allowed": payload.get("formal_allowed") is True,
        "required_methods": payload.get("required_formal_methods") == ["malt", "malter_eq17"],
        "malt_cell": isinstance(selected.get("malt"), dict)
        and selected["malt"].get("cell_id") == SELECTED_CELLS["malt"],
        "malter_cell": isinstance(selected.get("malter_eq17"), dict)
        and selected["malter_eq17"].get("cell_id") == SELECTED_CELLS["malter_eq17"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"accepted MALT/MALTER selection failed: {checks}")
    return {"passed": True, "checks": checks, "sha256": sha256_file(path)}


def build_data_inventory_payload(data_dir: Path) -> dict[str, Any]:
    names = [f"fineweb_train_{index:06d}.bin" for index in range(1, 51)]
    records = []
    print(
        "EX55 hashing the frozen R1 FineWeb inventory "
        "(train 000001--000050 plus validation 000000).",
        flush=True,
    )
    for index, name in enumerate(names, start=1):
        shard = data_dir / name
        if not shard.is_file() or shard.stat().st_size <= 0:
            raise RuntimeError(f"missing frozen R1 data shard: {shard}")
        records.append({
            "name": name,
            "bytes": shard.stat().st_size,
            "sha256": sha256_file(shard),
        })
        if index % 10 == 0:
            print(f"EX55 data hash progress: {index}/50", flush=True)
    validation_path = data_dir / "fineweb_val_000000.bin"
    if not validation_path.is_file() or validation_path.stat().st_size <= 0:
        raise RuntimeError(f"missing validation shard: {validation_path}")
    validation = {
        "name": validation_path.name,
        "bytes": validation_path.stat().st_size,
        "sha256": sha256_file(validation_path),
    }
    return {
        "schema_version": 2,
        "status": "passed",
        "selection_policy": "exact_train_000001_through_000050_and_val_000000",
        "data_dir": str(data_dir),
        "ordered_train_shards": records,
        "validation_shard": validation,
        "selected_total_bytes": sum(int(row["bytes"]) for row in records)
        + int(validation["bytes"]),
        "extra_train_shards_are_ignored": True,
    }


def validate_data_inventory_structure(payload: dict[str, Any], data_dir: Path) -> None:
    expected_names = [f"fineweb_train_{index:06d}.bin" for index in range(1, 51)]
    records = payload.get("ordered_train_shards")
    validation = payload.get("validation_shard")
    checks = {
        "schema_version": payload.get("schema_version") == 2,
        "status": payload.get("status") == "passed",
        "selection_policy": payload.get("selection_policy")
        == "exact_train_000001_through_000050_and_val_000000",
        "data_dir": Path(str(payload.get("data_dir", ""))).resolve() == data_dir.resolve(),
        "train_records": isinstance(records, list)
        and [row.get("name") for row in records if isinstance(row, dict)] == expected_names,
        "validation_record": isinstance(validation, dict)
        and validation.get("name") == "fineweb_val_000000.bin",
        "extra_shards_ignored": payload.get("extra_train_shards_are_ignored") is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen data inventory structure failed: {checks}")
    assert isinstance(records, list) and isinstance(validation, dict)
    total = 0
    for record in [*records, validation]:
        if not isinstance(record, dict):
            raise RuntimeError("frozen data inventory contains a non-object record")
        digest = str(record.get("sha256", ""))
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"frozen data inventory lacks a full SHA-256: {record}")
        shard = data_dir / str(record["name"])
        if not shard.is_file() or shard.stat().st_size != int(record.get("bytes", -1)):
            raise RuntimeError(f"frozen data inventory file/size drift: {shard}")
        total += int(record["bytes"])
    if total != int(payload.get("selected_total_bytes", -1)):
        raise RuntimeError("frozen data inventory byte total drift")


def data_inventory_stat_signature(data_dir: Path) -> tuple[tuple[str, int, int], ...]:
    names = [f"fineweb_train_{index:06d}.bin" for index in range(1, 51)]
    names.append("fineweb_val_000000.bin")
    signature = []
    for name in names:
        stat = (data_dir / name).stat()
        signature.append((name, stat.st_size, stat.st_mtime_ns))
    return tuple(signature)


def frozen_data_inventory(args: argparse.Namespace, *, full_verify: bool = False) -> Path:
    path = args.run_dir / "frozen_data_inventory.json"
    data_dir = (args.official_repo / "data/fineweb10B").resolve()
    # A caller asking for `full_verify` must re-read content.  Size/mtime is an
    # optimization hint, not an integrity certificate (mtime can be restored).
    if path in _DATA_INVENTORY_CACHE and not full_verify:
        return _DATA_INVENTORY_CACHE[path]
    if path.is_file():
        accepted = read_json(path)
        validate_data_inventory_structure(accepted, data_dir)
        if full_verify:
            observed = build_data_inventory_payload(data_dir)
            if observed != accepted:
                raise RuntimeError("frozen EX55 FineWeb content changed after its full-hash freeze")
            _DATA_INVENTORY_STAT_SIGNATURES[path] = data_inventory_stat_signature(data_dir)
    else:
        accepted = build_data_inventory_payload(data_dir)
        write_json(path, accepted)
        _DATA_INVENTORY_STAT_SIGNATURES[path] = data_inventory_stat_signature(data_dir)
    _DATA_INVENTORY_CACHE[path] = path
    return path


def audit_malt_selection_lineage(
    selection_path: Path, inventory_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the exact deep lineage consumed by the frozen MALT worker."""
    selection = read_json(selection_path)
    pilot_manifest = Path(str(selection.get("pilot_manifest", ""))).expanduser().resolve()
    expected = str(selection.get("pilot_manifest_sha256", ""))
    if not pilot_manifest.is_file() or sha256_file(pilot_manifest) != expected:
        raise RuntimeError(f"MALT accepted pilot-manifest lineage is unavailable: {pilot_manifest}")
    payload = read_json(pilot_manifest)
    source_manifests = payload.get("source_manifests", [])
    if (
        not isinstance(source_manifests, list)
        or len(source_manifests) != EXPECTED_MALT_PILOT_SOURCE_MANIFESTS
    ):
        raise RuntimeError(
            "MALT accepted pilot aggregate must expose all twelve source manifests"
        )
    checked = 0
    source_audits = []
    if isinstance(payload.get("source_audit"), dict):
        source_audits.append(payload["source_audit"])
    for record in source_manifests:
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"MALT accepted pilot source-manifest lineage failed: {path}")
        source_payload = read_json(path)
        source_audit = source_payload.get("source_audit")
        if not isinstance(source_audit, dict):
            raise RuntimeError(f"MALT accepted pilot source audit is missing: {path}")
        source_audits.append(source_audit)
        checked += 1
    if not source_audits:
        raise RuntimeError("MALT accepted pilot manifest exposes no source audit")
    canonical_audits = {
        json.dumps(audit, sort_keys=True, separators=(",", ":")) for audit in source_audits
    }
    if len(canonical_audits) != 1:
        raise RuntimeError("MALT accepted pilot source audits disagree")
    inventory_hashes = {
        str(audit.get("data_inventory_certificate_sha256", "")) for audit in source_audits
    }
    if len(inventory_hashes) != 1 or len(next(iter(inventory_hashes))) != 64:
        raise RuntimeError("MALT accepted pilot inventory lineage is incomplete")
    expected_inventory_sha256 = next(iter(inventory_hashes))
    if inventory_path is not None and sha256_file(inventory_path) != expected_inventory_sha256:
        raise RuntimeError(
            "EX55 full-hash data inventory is not byte-identical to the accepted MALT pilot inventory"
        )
    return {
        "passed": True, "pilot_manifest": str(pilot_manifest),
        "pilot_manifest_sha256": expected, "source_manifests_checked": checked,
        "source_audit": source_audits[0],
        "data_inventory_certificate_sha256": expected_inventory_sha256,
    }


def prepare_self_contained_malt_selection(
    args: argparse.Namespace, inventory_path: Path,
) -> Path:
    """Freeze accepted MALT pilot lineage under the EX55 run for later resume."""
    inputs = prepare_accepted_inputs(args)
    source_selection = inputs["malt_selection"]
    target_root = args.run_dir / "accepted_inputs/malt_self_contained"
    target_selection = target_root / "pilot_selection_verified_ex55.json"
    target_pilot = target_root / "pilot_manifest.json"
    receipt_path = target_root / "lineage_receipt.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        files = receipt.get("files", [])
        if (
            receipt.get("passed") is not True
            or receipt.get("source_selection_sha256") != sha256_file(source_selection)
            or receipt.get("data_inventory_certificate_sha256") != sha256_file(inventory_path)
            or not isinstance(files, list)
        ):
            raise RuntimeError("run-local MALT lineage receipt drift")
        for record in files:
            path = Path(str(record.get("path", "")))
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise RuntimeError(f"run-local MALT lineage file drift: {path}")
        audit_malt_selection_lineage(target_selection, inventory_path)
        return target_selection

    lineage = audit_malt_selection_lineage(source_selection, inventory_path)
    source_pilot = Path(str(lineage["pilot_manifest"]))
    pilot_payload = read_json(source_pilot)
    target_root.mkdir(parents=True, exist_ok=True)
    copied_records = []
    files = []
    for index, record in enumerate(pilot_payload.get("source_manifests", [])):
        source = Path(str(record["path"])).expanduser().resolve()
        safe_cell = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(record.get("cell_id", f"unit_{index:02d}"))
        )
        target = target_root / "source_manifests" / f"{index:02d}_{safe_cell}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = sha256_file(target)
        if digest != record["sha256"]:
            raise RuntimeError(f"copied MALT pilot source manifest drift: {target}")
        copied = dict(record)
        copied["path"] = str(target.resolve())
        copied_records.append(copied)
        files.append({"path": str(target.resolve()), "sha256": digest})
    pilot_payload["source_manifests"] = copied_records
    if isinstance(pilot_payload.get("data_inventory"), dict):
        pilot_payload["data_inventory"] = {
            **pilot_payload["data_inventory"],
            "path": str(inventory_path.resolve()),
            "sha256": sha256_file(inventory_path),
        }
    write_json(target_pilot, pilot_payload)
    files.append({"path": str(target_pilot.resolve()), "sha256": sha256_file(target_pilot)})

    selection_payload = read_json(source_selection)
    selection_payload["pilot_manifest"] = str(target_pilot.resolve())
    selection_payload["pilot_manifest_sha256"] = sha256_file(target_pilot)
    selection_payload["ex55_lineage_rebind"] = {
        "role": "path_rebind_only_no_reselection",
        "source_selection_sha256": sha256_file(source_selection),
        "source_pilot_manifest_sha256": lineage["pilot_manifest_sha256"],
        "data_inventory_certificate_sha256": sha256_file(inventory_path),
    }
    write_json(target_selection, selection_payload)
    files.append({"path": str(target_selection.resolve()), "sha256": sha256_file(target_selection)})
    write_json(receipt_path, {
        "schema_version": 1,
        "passed": True,
        "source_selection": str(source_selection.resolve()),
        "source_selection_sha256": sha256_file(source_selection),
        "source_pilot_manifest_sha256": lineage["pilot_manifest_sha256"],
        "data_inventory": str(inventory_path.resolve()),
        "data_inventory_certificate_sha256": sha256_file(inventory_path),
        "selection_values_changed": False,
        "files": files,
    })
    audit_malt_selection_lineage(target_selection, inventory_path)
    return target_selection


def extract_summaries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    if isinstance(payload.get("summaries"), list):
        rows.extend(row for row in payload["summaries"] if isinstance(row, dict))
    if isinstance(payload.get("summary"), dict) and payload["summary"] not in rows:
        rows.append(payload["summary"])
    return rows


def summary_is_exact_frozen_winner(row: dict[str, Any], method: str) -> bool:
    """Reject a right optimizer family paired with the wrong selected cell."""
    expected_cell = SELECTED_CELLS[method]
    observed_method = str(row.get("method", "")).strip().lower()
    observed_cell = str(row.get("cell_id", "")).strip().lower()
    expected_worker_method = WORKER_METHOD_LABELS[method]
    if observed_cell:
        return observed_method == expected_worker_method and observed_cell == expected_cell
    # The four core routes expose only `method`; their cell and method names coincide.
    return expected_cell == method and observed_method == expected_worker_method


def manifest_matches(path: Path, *, expected_methods: tuple[str, ...], seed: int, formal: bool) -> bool:
    try:
        payload = read_json(path)
    except Exception:
        return False
    status = str(payload.get("status", ""))
    if not status.startswith(ACCEPTED_STATUS_PREFIXES) or payload.get("failures"):
        return False
    if int(payload.get("seed", -1)) != seed:
        return False
    raw_rows = payload.get("summaries")
    if not isinstance(raw_rows, list):
        return False
    rows = [row for row in raw_rows if isinstance(row, dict)]
    if len(rows) != len(expected_methods):
        return False
    observed = set()
    for row in rows:
        if row.get("evidence_valid") is not True:
            return False
        matches = [method for method in expected_methods if summary_is_exact_frozen_winner(row, method)]
        if len(matches) != 1:
            return False
        observed.add(matches[0])
        row_seed = row.get("controlled_seed", row.get("seed"))
        if row_seed is None or int(row_seed) != seed:
            return False
        if formal:
            if int(row.get("total_steps", row.get("final_val_step", -1))) != 6200:
                return False
            checkpoint = Path(str(row.get("checkpoint_path", "")))
            expected_bytes = int(row.get("checkpoint_bytes", -1))
            if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                return False
            if expected_bytes > 0 and checkpoint.stat().st_size != expected_bytes:
                return False
    return observed == set(expected_methods)


def aggregate_manifest_name(group: str, stage: str) -> str:
    if group == "core":
        return "r1_manifest.json"
    if group == "extended" and stage == "pilot":
        # The upstream numerical-smoke mode intentionally forces its three
        # center cells.  EX55 must smoke the accepted formal winners instead,
        # so its outcome-free seed-5501 engineering run uses formal-smoke mode.
        return "formal_smoke_manifest.json"
    if stage not in AGGREGATE_STEMS:
        raise ValueError(f"no aggregate manifest for stage={stage!r}")
    return f"{AGGREGATE_STEMS[stage]}_manifest.json"


def aggregate_plan_name(group: str, stage: str) -> str:
    if group == "core":
        return "r1_plan.json"
    if group == "extended" and stage == "pilot":
        return "formal_smoke_plan.json"
    if stage not in AGGREGATE_STEMS:
        raise ValueError(f"no aggregate plan for stage={stage!r}")
    return f"{AGGREGATE_STEMS[stage]}_plan.json"


def accepted_batch(
    stage_dir: Path, *, group: str, stage: str,
    expected_methods: tuple[str, ...], seed: int, formal: bool,
) -> Path | None:
    """Accept only the runner's batch-level aggregate, never a child run manifest."""
    manifest_name = aggregate_manifest_name(group, stage)
    accepted = []
    for path in stage_dir.glob(f"*/{manifest_name}"):
        if manifest_matches(path, expected_methods=expected_methods, seed=seed, formal=formal):
            accepted.append(path)
    if len(accepted) > 1:
        raise RuntimeError(f"multiple accepted batches in {stage_dir}: {accepted}")
    return accepted[0] if accepted else None


def resumable_batch(stage_dir: Path, *, group: str, stage: str) -> Path | None:
    plans = sorted(
        stage_dir.glob(f"*/{aggregate_plan_name(group, stage)}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return plans[0].parent if plans else None


def common_args(args: argparse.Namespace, worker: Path, seed: int, result_dir: Path) -> list[str]:
    return [
        sys.executable, str(worker), "--official-repo", str(args.official_repo),
        "--python-exe", args.training_python, "--seed", str(seed), "--results-dir", str(result_dir),
    ]


def add_wandb(args: argparse.Namespace, command: list[str], *, disabled: bool) -> None:
    command.extend(["--wandb-mode", "disabled" if disabled else args.wandb_mode])
    if not disabled:
        command.extend(["--wandb-project", args.wandb_project])
        if args.wandb_entity:
            command.extend(["--wandb-entity", args.wandb_entity])


def group_command(
    args: argparse.Namespace, group: str, stage: str, result_dir: Path,
    *, seed: int, smoke_manifest: Path | None = None,
) -> list[str]:
    paths = frozen_paths(args)
    inputs = prepare_accepted_inputs(args)
    methods = GROUPS[group]
    worker_key = "malt" if group in {"malt", "malter_eq17"} else group
    command = common_args(args, paths[worker_key], seed, result_dir)
    disabled = stage != "formal"
    if worker_key == "malt":
        inventory = frozen_data_inventory(args)
        command.extend(["--data-inventory-certificate", str(inventory)])

    if group == "core":
        command.extend(["--methods", *methods, "--run-prefix", "ex55_r1_fresh_seed"])
        if stage in {"pilot", "formal_smoke"}:
            command.extend(["--numerical-smoke", "--smoke-steps", str(SMOKE_STEPS)])
        elif stage == "formal":
            if smoke_manifest is None:
                raise RuntimeError("core formal requires its seed-2027 smoke")
            command.extend(["--smoke-manifest", str(smoke_manifest)])
        else:
            command.append("--preflight")
        command.append("--continue-on-error")
    elif group == "extended":
        command.extend([
            "--methods",
            *(WORKER_METHOD_LABELS[method] for method in methods),
        ])
        if stage == "pilot":
            command.extend(["--formal-smoke", "--smoke-steps", str(SMOKE_STEPS)])
        elif stage == "formal_smoke":
            command.extend(["--formal-smoke", "--smoke-steps", str(SMOKE_STEPS)])
        elif stage == "formal":
            if smoke_manifest is None:
                raise RuntimeError("extended formal requires its seed-2027 smoke")
            command.extend(["--formal", "--smoke-manifest", str(smoke_manifest)])
        else:
            command.append("--preflight")
        command.append("--continue-on-error")
    elif group == "mousse":
        if stage == "pilot":
            command.extend(["--numerical-smoke", "--smoke-steps", str(SMOKE_STEPS), "--cells", "mousse_lr100"])
        elif stage == "formal_smoke":
            command.extend(["--formal-smoke", "--smoke-steps", str(SMOKE_STEPS), "--selection-certificate", str(inputs["mousse_selection"])])
        elif stage == "formal":
            if smoke_manifest is None:
                raise RuntimeError("Mousse formal requires its seed-2027 smoke")
            command.extend(["--formal", "--selection-certificate", str(inputs["mousse_selection"]), "--smoke-manifest", str(smoke_manifest)])
        else:
            command.append("--preflight")
    else:
        method = group
        selected_cell = SELECTED_CELLS[method]
        if stage == "pilot":
            command.extend(["--numerical-smoke", "--smoke-steps", str(SMOKE_STEPS), "--cells", selected_cell])
        elif stage == "formal_smoke":
            selection = prepare_self_contained_malt_selection(args, inventory)
            command.extend(["--formal-smoke", "--smoke-steps", str(SMOKE_STEPS), "--selection-certificate", str(selection), "--selected-method", method])
        elif stage == "formal":
            if smoke_manifest is None:
                raise RuntimeError(f"{method} formal requires its seed-2027 smoke")
            selection = prepare_self_contained_malt_selection(args, inventory)
            command.extend(["--formal", "--selection-certificate", str(selection), "--selected-method", method, "--smoke-manifest", str(smoke_manifest)])
        else:
            command.append("--preflight")
    if stage != "preflight":
        add_wandb(args, command, disabled=disabled)
    previous = None if stage == "preflight" else resumable_batch(
        result_dir, group=group, stage=stage,
    )
    if previous is not None:
        command.extend(["--resume-batch", str(previous)])
    return command


def run_jobs(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> None:
    pending = list(jobs)
    active: dict[str, dict[str, Any]] = {}
    failures = []
    logs = args.run_dir / "controller_logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            label = str(job["label"])
            log = logs / f"{label.replace('/', '_')}.log"
            handle = log.open("a", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["NANOGPT_WORKSPACE_ROOT"] = str(args.repo.parent)
            env["SELECTIVE_NEWTON_MUON_MAIN_CONFERENCE_REPO"] = str(args.repo)
            command = job["command"]
            append_command(args.run_dir, {
                "label": label, "gpu": gpu, "command": command, "command_text": shlex.join(command),
                "log": str(log), "started_at": datetime.now().astimezone().isoformat(),
            })
            print(f"START gpu={gpu} label={label}", flush=True)
            process = subprocess.Popen(command, cwd=args.repo, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            active[gpu] = {**job, "process": process, "handle": handle, "log": log}
        finished = []
        for gpu, job in active.items():
            code = job["process"].poll()
            if code is None:
                continue
            job["handle"].close()
            accepted = accepted_batch(
                job["result_dir"], group=job["group"], stage=job["stage"],
                expected_methods=job["methods"], seed=job["seed"], formal=job["formal"],
            )
            print(f"END gpu={gpu} label={job['label']} return_code={code} accepted={accepted is not None}", flush=True)
            if code != 0 and accepted is None:
                failures.append({"label": job["label"], "gpu": gpu, "return_code": code, "log": str(job["log"])})
            finished.append(gpu)
        for gpu in finished:
            del active[gpu]
        if failures:
            for job in active.values():
                job["process"].terminate()
            for job in active.values():
                try:
                    job["process"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    job["process"].kill()
                job["handle"].close()
            write_json(args.run_dir / "worker_failures.json", failures)
            raise RuntimeError(f"EX55 worker failure: {failures}")
        if pending or active:
            time.sleep(2)


def require_pass(path: Path, label: str) -> None:
    if not path.is_file() or read_json(path).get("passed") is not True:
        raise RuntimeError(f"{label} has not passed: {path}")


def recorded_file_matches(record: Any, *, expected_path: Path | None = None) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        if expected_path is not None and path != expected_path.expanduser().resolve():
            return False
        return (
            path.is_file()
            and path.stat().st_size == int(record.get("bytes", -1))
            and sha256_file(path) == record.get("sha256")
        )
    except (OSError, ValueError, TypeError):
        return False


def formal_metrics_lineage_matches(
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    contract: Path,
    formal_units: Path,
) -> bool:
    try:
        reference = payload.get("formal_metrics_lineage")
        if not isinstance(reference, dict):
            return False
        raw_path = Path(str(reference.get("path", "")))
        lineage_path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
        lineage_path = lineage_path.expanduser().resolve()
        if (
            not lineage_path.is_file()
            or lineage_path.stat().st_size != int(reference.get("bytes", -1))
            or sha256_file(lineage_path) != reference.get("sha256")
        ):
            return False
        lineage = read_json(lineage_path)
        units_payload = read_json(formal_units)
        contract_payload = read_json(contract)
        formal_steps = int(contract_payload["protocol"]["formal_steps"])
        validation_every = int(contract_payload["protocol"]["validation_every"])
        expected_tail = [
            formal_steps - offset * validation_every for offset in range(4, -1, -1)
        ]
        if (
            lineage.get("passed") is not True
            or lineage.get("experiment_id") != EXPERIMENT
            or lineage.get("formal_seed") != FORMAL_SEED
            or int(lineage.get("formal_steps", -1)) != formal_steps
            or int(lineage.get("validation_every", -1)) != validation_every
            or lineage.get("required_tail5_steps") != expected_tail
            or lineage.get("formal_units_sha256") != sha256_file(formal_units)
        ):
            return False
        formal_records = units_payload.get("units")
        lineage_records = lineage.get("units")
        if not isinstance(formal_records, list) or not isinstance(lineage_records, list):
            return False
        formal_index = {str(record.get("method", "")): record for record in formal_records}
        if set(formal_index) != set(SELECTED_CELLS) or len(lineage_records) != len(SELECTED_CELLS):
            return False
        seen_methods: set[str] = set()
        seen_metrics: set[str] = set()
        for record in lineage_records:
            if not isinstance(record, dict):
                return False
            method = str(record.get("method", ""))
            if method not in SELECTED_CELLS or method in seen_methods:
                return False
            if (
                record.get("selected_cell") != SELECTED_CELLS[method]
                or int(record.get("seed", -1)) != FORMAL_SEED
                or record.get("source") != "accepted_formal_child_metrics_csv"
                or record.get("tail5_steps") != expected_tail
                or int(record.get("final_step", -1)) != formal_steps
                or int(record.get("initial_step", -1)) != 0
            ):
                return False
            losses = record.get("tail5_losses")
            if not isinstance(losses, list) or len(losses) != 5:
                return False
            finite_losses = [float(value) for value in losses]
            if not all(math.isfinite(value) for value in finite_losses):
                return False
            mean = float(record.get("tail5_val_loss_mean"))
            if not math.isfinite(mean) or not math.isclose(
                mean, math.fsum(finite_losses) / 5.0, rel_tol=1e-12, abs_tol=1e-12,
            ):
                return False
            for key in ("aggregate_manifest", "child_manifest", "child_summary", "metrics"):
                if not recorded_file_matches(record.get(key)):
                    return False
            formal_record = formal_index[method]
            aggregate = record["aggregate_manifest"]
            if (
                Path(str(aggregate["path"])).expanduser().resolve()
                != Path(str(formal_record.get("manifest", ""))).expanduser().resolve()
                or aggregate["sha256"] != formal_record.get("manifest_sha256")
            ):
                return False
            metrics_path = str(Path(str(record["metrics"]["path"])).expanduser().resolve())
            if metrics_path in seen_metrics:
                return False
            seen_metrics.add(metrics_path)
            seen_methods.add(method)
        return seen_methods == set(SELECTED_CELLS)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def analysis_manifest_matches(
    manifest_path: Path, *, contract: Path, historical_panel: Path,
    formal_units: Path | None = None,
    analyzer: Path | None = None,
    analysis_amendment: Path | None = None,
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        payload = read_json(manifest_path)
        if (
            payload.get("passed") is not True
            or payload.get("experiment_id") != EXPERIMENT
            or payload.get("contract_sha256") != sha256_file(contract)
            or payload.get("historical_panel_sha256") != sha256_file(historical_panel)
        ):
            return False
        if formal_units is not None and payload.get("formal_units_sha256") != sha256_file(formal_units):
            return False
        if analyzer is not None and not recorded_file_matches(
            payload.get("analyzer"), expected_path=analyzer,
        ):
            return False
        if analysis_amendment is not None:
            if not recorded_file_matches(
                payload.get("analysis_amendment"), expected_path=analysis_amendment,
            ):
                return False
        elif payload.get("analysis_amendment") not in (None, {}):
            return False
        if formal_units is not None and not formal_metrics_lineage_matches(
            manifest_path, payload, contract=contract, formal_units=formal_units,
        ):
            return False
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            return False
        for record in artifacts:
            path = manifest_path.parent / str(record.get("path", ""))
            if (
                not path.is_file()
                or path.stat().st_size != int(record.get("bytes", -1))
                or sha256_file(path) != record.get("sha256")
            ):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def run_preformal_sensitivity(
    args: argparse.Namespace, paths: dict[str, Path], inputs: dict[str, Path],
) -> Path:
    output = args.run_dir / "analysis_preformal"
    manifest = output / "analysis_preformal_manifest.json"
    if analysis_manifest_matches(
        manifest, contract=paths["contract"], historical_panel=inputs["historical_panel"],
    ):
        return manifest
    command = [
        sys.executable, str(paths["analyzer"]),
        "--run-dir", str(args.run_dir),
        "--contract", str(paths["contract"]),
        "--historical-panel", str(inputs["historical_panel"]),
        "--output-dir", str(output),
        "--historical-only",
    ]
    append_command(args.run_dir, {
        "label": "preflight/historical_only_sensitivity",
        "command": command,
        "command_text": shlex.join(command),
    })
    subprocess.run(command, cwd=args.repo, check=True)
    if not analysis_manifest_matches(
        manifest, contract=paths["contract"], historical_panel=inputs["historical_panel"],
    ):
        raise RuntimeError("EX55 preformal historical-only sensitivity failed integrity validation")
    return manifest


def run_preflight(args: argparse.Namespace) -> None:
    status = args.run_dir / "preflight_manifest.json"
    if status.is_file() and read_json(status).get("passed") is True:
        print("skip passed preflight")
        return
    paths = frozen_paths(args)
    inputs = prepare_accepted_inputs(args)
    extended_selection_audit = audit_extended_selection(inputs["extended_selection"])
    mousse_selection_audit = audit_mousse_selection(inputs["mousse_selection"])
    malt_winner_audit = audit_malt_winner_selection(inputs["malt_selection"])
    preformal_manifest = run_preformal_sensitivity(args, paths, inputs)
    inventory = frozen_data_inventory(args, full_verify=True)
    self_contained_malt_selection = prepare_self_contained_malt_selection(args, inventory)
    contract = read_json(paths["contract"])
    checks = {
        "contract": contract["experiment_id"] == EXPERIMENT,
        "worker_method_labels": contract.get("worker_method_labels")
        == {"moonlight": "moonlight_muon"},
        "historical_panel_hash": sha256_file(inputs["historical_panel"]) == read_json(paths["contract"])["accepted_inputs"]["historical_panel_sha256"],
        "extended_selection_hash": sha256_file(inputs["extended_selection"]) == read_json(paths["contract"])["accepted_inputs"]["extended_selection_sha256"],
        "extended_selection_frozen_winners": extended_selection_audit["passed"] is True,
        "mousse_selection_hash": sha256_file(inputs["mousse_selection"]) == read_json(paths["contract"])["accepted_inputs"]["mousse_selection_sha256"],
        "malt_selection_hash": sha256_file(inputs["malt_selection"]) == read_json(paths["contract"])["accepted_inputs"]["malt_selection_sha256"],
        "mousse_selection_frozen_winner": mousse_selection_audit["passed"] is True,
        "malt_selection_frozen_winners": malt_winner_audit["passed"] is True,
        "data_inventory": read_json(inventory).get("status") == "passed",
    }
    malt_lineage = audit_malt_selection_lineage(self_contained_malt_selection, inventory)
    checks["malt_selection_source_lineage"] = malt_lineage["passed"] is True
    checks["malt_selection_self_contained"] = (
        self_contained_malt_selection.is_relative_to(args.run_dir)
    )
    logs = args.run_dir / "controller_logs"
    logs.mkdir(parents=True, exist_ok=True)
    commands = [
        ("core", group_command(args, "core", "preflight", args.run_dir / "preflight/core", seed=ENGINEERING_SEED)),
        ("extended", group_command(args, "extended", "preflight", args.run_dir / "preflight/extended", seed=ENGINEERING_SEED)),
        ("mousse", group_command(args, "mousse", "preflight", args.run_dir / "preflight/mousse", seed=ENGINEERING_SEED)),
        ("malt", group_command(args, "malt", "preflight", args.run_dir / "preflight/malt", seed=ENGINEERING_SEED)),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus[0]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    worker_failures: dict[str, Any] = {}
    for label, command in commands:
        log = logs / f"preflight_{label}.log"
        append_command(args.run_dir, {"label": f"preflight/{label}", "gpu": args.gpus[0], "command": command, "log": str(log)})
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=args.repo, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        checks[f"worker_{label}"] = completed.returncode == 0
        if completed.returncode != 0:
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            worker_failures[label] = {
                "return_code": completed.returncode,
                "log": str(log),
                "tail": lines[-80:],
            }
    payload = {
        "schema_version": 1, "experiment_id": EXPERIMENT, "passed": all(checks.values()),
        "checks": checks, "engineering_seed": ENGINEERING_SEED,
        "worker_failures": worker_failures,
        "contract_sha256": sha256_file(paths["contract"]),
        "source_snapshot_manifest_sha256": sha256_file(args.run_dir / "source_snapshot/source_snapshot_manifest.json"),
        "accepted_inputs": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in inputs.items()},
        "accepted_inputs_manifest": {
            "path": str(args.run_dir / "accepted_inputs/accepted_inputs_manifest.json"),
            "sha256": sha256_file(args.run_dir / "accepted_inputs/accepted_inputs_manifest.json"),
        },
        "malt_selection_source_lineage": malt_lineage,
        "extended_selection_audit": extended_selection_audit,
        "mousse_selection_audit": mousse_selection_audit,
        "malt_winner_selection_audit": malt_winner_audit,
        "preformal_sensitivity": {
            "path": str(preformal_manifest),
            "sha256": sha256_file(preformal_manifest),
        },
        "malt_self_contained_selection": {
            "path": str(self_contained_malt_selection),
            "sha256": sha256_file(self_contained_malt_selection),
        },
        "frozen_data_inventory": str(inventory), "frozen_data_inventory_sha256": sha256_file(inventory),
    }
    write_json(status, payload)
    if not payload["passed"]:
        raise RuntimeError(
            f"EX55 preflight failed: checks={checks} worker_failures={worker_failures}"
        )


def jobs_for_stage(args: argparse.Namespace, stage: str, seed: int) -> list[dict[str, Any]]:
    jobs = []
    for group, methods in GROUPS.items():
        result_dir = args.run_dir / stage / group / f"seed{seed}"
        formal = stage == "formal"
        if accepted_batch(
            result_dir, group=group, stage=stage,
            expected_methods=methods, seed=seed, formal=formal,
        ) is not None:
            print(f"skip accepted {stage}/{group}")
            continue
        smoke_manifest = None
        if formal:
            smoke_dir = args.run_dir / "formal_smoke" / group / f"seed{seed}"
            smoke_manifest = accepted_batch(
                smoke_dir, group=group, stage="formal_smoke",
                expected_methods=methods, seed=seed, formal=False,
            )
            if smoke_manifest is None:
                raise RuntimeError(f"formal smoke missing for {group}")
        jobs.append({
            "label": f"{stage}/{group}/seed{seed}", "group": group, "stage": stage,
            "methods": methods, "seed": seed,
            "formal": formal, "result_dir": result_dir,
            "command": group_command(args, group, stage, result_dir, seed=seed, smoke_manifest=smoke_manifest),
        })
    return jobs


def assert_complete(args: argparse.Namespace, stage: str, seed: int, *, formal: bool) -> dict[str, Path]:
    accepted = {}
    for group, methods in GROUPS.items():
        path = accepted_batch(
            args.run_dir / stage / group / f"seed{seed}", group=group, stage=stage,
            expected_methods=methods, seed=seed, formal=formal,
        )
        if path is None:
            raise RuntimeError(f"EX55 {stage} incomplete for {group}")
        accepted[group] = path
    return accepted


def summary_for_method(manifest: Path, method: str) -> dict[str, Any]:
    rows = read_json(manifest).get("summaries", [])
    matched = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        if summary_is_exact_frozen_winner(row, method):
            matched.append(row)
    if len(matched) != 1:
        raise RuntimeError(f"{method}: expected one formal summary in {manifest}, found {len(matched)}")
    return matched[0]


def child_run_for_summary(
    manifest: Path, row: dict[str, Any], method: str, seed: int,
) -> dict[str, Path]:
    """Resolve one accepted child run without assuming one runner schema.

    The core, extended, Mousse, and MALT runners do not expose identical
    aggregate-summary fields.  Some include ``run_name`` and some expose it
    only in the child manifest.  Match the immutable method/cell, seed, and
    initialization metadata and require a unique accepted child.
    """
    candidates: list[Path] = []
    run_name = str(row.get("run_name", "")).strip()
    if run_name:
        candidates.append(manifest.parent / run_name)
    candidates.extend(path.parent for path in manifest.parent.glob("*/run_manifest.json"))
    unique: dict[Path, Path] = {path.resolve(): path for path in candidates}
    matched: list[dict[str, Path]] = []
    for run_dir in unique.values():
        child_manifest = run_dir / "run_manifest.json"
        # Accepted runner families use two filename schemas.  The core runner
        # writes r1_summary.json/r1_metrics.csv; newer family runners generally
        # use summary.json/metrics.csv.  Require one complete pair.
        artifact_pairs = [
            (run_dir / "r1_summary.json", run_dir / "r1_metrics.csv"),
            (run_dir / "summary.json", run_dir / "metrics.csv"),
        ]
        complete_pairs = [
            (summary_path, metrics_path)
            for summary_path, metrics_path in artifact_pairs
            if summary_path.is_file() and metrics_path.is_file()
        ]
        if not child_manifest.is_file() or len(complete_pairs) != 1:
            continue
        summary_path, metrics_path = complete_pairs[0]
        try:
            child = read_json(child_manifest)
            child_summary = read_json(summary_path)
        except Exception:
            continue
        if not str(child.get("status", "")).startswith(ACCEPTED_STATUS_PREFIXES):
            continue
        if not summary_is_exact_frozen_winner(child_summary, method):
            continue
        child_seed = int(child_summary.get("controlled_seed", child_summary.get("seed", -1)))
        if child_seed != seed or child_summary.get("init_sha256") != row.get("init_sha256"):
            continue
        matched.append({
            "run_dir": run_dir,
            "manifest": child_manifest,
            "summary": summary_path,
            "metrics": metrics_path,
        })
    if len(matched) != 1:
        raise RuntimeError(
            f"{method}: expected one accepted formal-smoke child for metrics lineage, "
            f"found {len(matched)} under {manifest.parent}"
        )
    return matched[0]


def initial_validation_evidence(
    manifest: Path, row: dict[str, Any], method: str, seed: int, group: str,
) -> dict[str, Any]:
    expected_validation_tokens = SMOKE_VALIDATION_TOKENS[group]
    aggregate = read_json(manifest)
    declared_validation_tokens = row.get("val_tokens", aggregate.get("val_tokens"))
    if (
        declared_validation_tokens is not None
        and int(declared_validation_tokens) != expected_validation_tokens
    ):
        raise RuntimeError(
            f"{method}: formal-smoke validation-token contract drift: "
            f"{declared_validation_tokens} != {expected_validation_tokens}"
        )
    direct = row.get("initial_val_loss", row.get("val_loss_step_0"))
    if direct is not None:
        value = float(direct)
        if not math.isfinite(value):
            raise RuntimeError(f"{method}: non-finite initial validation loss")
        return {
            "initial_val_loss": value,
            "validation_tokens": expected_validation_tokens,
            "source": "aggregate_summary",
            "metrics_path": None,
            "metrics_sha256": None,
            "child_manifest_path": None,
            "child_manifest_sha256": None,
        }

    child_artifacts = child_run_for_summary(manifest, row, method, seed)
    metrics_path = child_artifacts["metrics"]
    child_manifest = child_artifacts["manifest"]
    child_summary = child_artifacts["summary"]
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    validation = [
        record for record in records
        if str(record.get("event", "")).strip().lower() == "validation"
    ]
    initial = [record for record in validation if int(record.get("step", -1)) == 0]
    if len(initial) != 1 or not validation:
        raise RuntimeError(f"{method}: expected exactly one step-0 validation metric")
    value = float(initial[0]["loss"])
    if not math.isfinite(value):
        raise RuntimeError(f"{method}: non-finite step-0 validation loss")
    final = max(validation, key=lambda record: int(record["step"]))
    if float(final["loss"]) != float(row["final_val_loss"]):
        raise RuntimeError(f"{method}: metrics/aggregate final validation mismatch")
    if group == "core" and row.get("evidence_profile") != "exact_shape_numerical_smoke":
        raise RuntimeError(f"{method}: core smoke evidence profile drift")
    return {
        "initial_val_loss": value,
        "validation_tokens": expected_validation_tokens,
        "source": "accepted_child_metrics_step_0",
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "child_manifest_path": str(child_manifest),
        "child_manifest_sha256": sha256_file(child_manifest),
        "child_summary_path": str(child_summary),
        "child_summary_sha256": sha256_file(child_summary),
    }


def make_checkpoint_certificate(manifest: Path, method: str) -> dict[str, Any]:
    summary = summary_for_method(manifest, method)
    checkpoint = Path(str(summary.get("checkpoint_path", ""))).expanduser().resolve()
    expected_bytes = int(summary.get("checkpoint_bytes", -1))
    if not checkpoint.is_file() or expected_bytes <= 0 or checkpoint.stat().st_size != expected_bytes:
        raise RuntimeError(f"{method}: formal checkpoint path/size audit failed: {checkpoint}")
    digest = sha256_file(checkpoint)
    summary_digest = str(summary.get("checkpoint_sha256", ""))
    if summary_digest and summary_digest != digest:
        raise RuntimeError(f"{method}: runner checkpoint SHA-256 disagrees with EX55")
    return {
        "path": str(checkpoint),
        "bytes": expected_bytes,
        "sha256": digest,
        "runner_reported_sha256": summary_digest or None,
    }


def certify_paired_initialization(manifests: dict[str, Path], seed: int) -> dict[str, Any]:
    records = []
    for group, methods in GROUPS.items():
        manifest = manifests[group]
        for method in methods:
            row = summary_for_method(manifest, method)
            row_seed = int(row.get("controlled_seed", row.get("seed", -1)))
            init_sha256 = str(row.get("init_sha256", ""))
            if row_seed != seed or len(init_sha256) != 64:
                raise RuntimeError(f"{method}: formal-smoke pairing metadata is incomplete")
            initial_evidence = initial_validation_evidence(
                manifest, row, method, seed, group,
            )
            records.append({
                "method": method,
                "selected_cell": SELECTED_CELLS[method],
                "seed": row_seed,
                "init_sha256": init_sha256,
                **initial_evidence,
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
            })
    init_hashes = {record["init_sha256"] for record in records}
    losses_by_validation_tokens: dict[int, set[float]] = {}
    for record in records:
        losses_by_validation_tokens.setdefault(
            int(record["validation_tokens"]), set()
        ).add(float(record["initial_val_loss"]))
    within_protocol_exact = all(
        len(losses) == 1 for losses in losses_by_validation_tokens.values()
    )
    if len(records) != 10 or len(init_hashes) != 1 or not within_protocol_exact:
        raise RuntimeError(
            f"EX55 paired formal-smoke initialization failed: "
            f"records={len(records)} init_hashes={init_hashes} "
            f"losses_by_validation_tokens={losses_by_validation_tokens}"
        )
    return {
        "schema_version": 2,
        "experiment_id": EXPERIMENT,
        "passed": True,
        "seed": seed,
        "formal_units": len(records),
        "init_sha256": next(iter(init_hashes)),
        "parameter_initialization_exact_across_all_methods": True,
        "initial_validation": {
            "role": "diagnostic_within_validation_token_stratum_not_cross_stratum_pairing_gate",
            "within_stratum_exact": within_protocol_exact,
            "cross_method_exact_comparison_eligible": len(losses_by_validation_tokens) == 1,
            "loss_by_validation_tokens": {
                str(tokens): next(iter(losses))
                for tokens, losses in sorted(losses_by_validation_tokens.items())
            },
            "formal_endpoint_validation_tokens": FORMAL_VALIDATION_TOKENS,
        },
        "records": records,
    }


def verify_checkpoint_certificates(formal_units_path: Path) -> list[dict[str, Any]]:
    payload = read_json(formal_units_path)
    records = payload.get("units", [])
    if payload.get("passed") is not True or not isinstance(records, list) or len(records) != 10:
        raise RuntimeError("formal unit certificate is incomplete")
    verified = []
    seen_methods = set()
    seen_paths = set()
    for record in records:
        method = str(record.get("method", ""))
        certificate = record.get("checkpoint")
        manifest = Path(str(record.get("manifest", "")))
        if method not in SELECTED_CELLS or method in seen_methods or not isinstance(certificate, dict):
            raise RuntimeError(f"invalid formal unit checkpoint record: {record}")
        if not manifest.is_file() or sha256_file(manifest) != record.get("manifest_sha256"):
            raise RuntimeError(f"{method}: formal aggregate manifest lineage failed")
        checkpoint = Path(str(certificate.get("path", ""))).expanduser().resolve()
        if str(checkpoint) in seen_paths:
            raise RuntimeError(f"formal methods unexpectedly share a checkpoint: {checkpoint}")
        expected_bytes = int(certificate.get("bytes", -1))
        expected_sha256 = str(certificate.get("sha256", ""))
        if (
            not checkpoint.is_file()
            or checkpoint.stat().st_size != expected_bytes
            or sha256_file(checkpoint) != expected_sha256
        ):
            raise RuntimeError(f"{method}: final checkpoint bytes/hash verification failed: {checkpoint}")
        summary = summary_for_method(manifest, method)
        if (
            Path(str(summary.get("checkpoint_path", ""))).expanduser().resolve() != checkpoint
            or int(summary.get("checkpoint_bytes", -1)) != expected_bytes
        ):
            raise RuntimeError(f"{method}: checkpoint certificate no longer matches its summary")
        runner_digest = str(summary.get("checkpoint_sha256", ""))
        if runner_digest and runner_digest != expected_sha256:
            raise RuntimeError(f"{method}: runner checkpoint hash no longer matches EX55")
        seen_methods.add(method)
        seen_paths.add(str(checkpoint))
        verified.append({
            "method": method,
            "path": str(checkpoint),
            "bytes": expected_bytes,
            "sha256": expected_sha256,
            "passed": True,
        })
    if seen_methods != set(SELECTED_CELLS):
        raise RuntimeError("checkpoint certificate method coverage mismatch")
    return verified


def run_pilot(args: argparse.Namespace) -> None:
    require_pass(args.run_dir / "preflight_manifest.json", "preflight")
    frozen_data_inventory(args, full_verify=True)
    run_jobs(args, jobs_for_stage(args, "pilot", ENGINEERING_SEED))
    accepted = assert_complete(args, "pilot", ENGINEERING_SEED, formal=False)
    write_json(args.run_dir / "pilot_manifest.json", {
        "schema_version": 1, "experiment_id": EXPERIMENT, "passed": True,
        "seed": ENGINEERING_SEED, "seed_role": "outcome_free_engineering_only",
        "steps": SMOKE_STEPS, "configuration_selection_performed": False,
        "formal_methods_dropped": [], "accepted_batches": {key: str(value) for key, value in accepted.items()},
    })


def run_formal(args: argparse.Namespace) -> None:
    amendment = ensure_controller_amendment(args)
    require_pass(args.run_dir / "pilot_manifest.json", "pilot")
    frozen_data_inventory(args, full_verify=True)
    run_jobs(args, jobs_for_stage(args, "formal_smoke", FORMAL_SEED))
    formal_smoke = assert_complete(args, "formal_smoke", FORMAL_SEED, formal=False)
    pairing_path = args.run_dir / "formal_smoke_pairing_manifest.json"
    pairing = certify_paired_initialization(formal_smoke, FORMAL_SEED)
    if amendment is not None:
        amendment_path = Path(str(amendment["receipt_path"])).expanduser().resolve()
        pairing["controller_amendment_path"] = str(amendment_path)
        pairing["controller_amendment_sha256"] = sha256_file(amendment_path)
    else:
        pairing["controller_amendment_path"] = None
        pairing["controller_amendment_sha256"] = None
    write_json(pairing_path, pairing)
    run_jobs(args, jobs_for_stage(args, "formal", FORMAL_SEED))
    accepted = assert_complete(args, "formal", FORMAL_SEED, formal=True)
    contract = read_json(frozen_paths(args)["contract"])
    units = []
    for group, methods in GROUPS.items():
        manifest = accepted[group]
        for method in methods:
            units.append({
                "method": method, "selected_cell": SELECTED_CELLS[method], "family": group,
                "seed": FORMAL_SEED, "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
                "checkpoint": make_checkpoint_certificate(manifest, method),
            })
    if len(units) != contract["execution_policy"]["formal_units"]:
        raise RuntimeError("EX55 formal unit count drift")
    write_json(args.run_dir / "formal_units_manifest.json", {
        "schema_version": 1, "experiment_id": EXPERIMENT, "passed": True,
        "formal_seed": FORMAL_SEED, "formal_units": 10, "units": units,
        "formal_smoke_pairing_manifest": str(pairing_path),
        "formal_smoke_pairing_manifest_sha256": sha256_file(pairing_path),
        "all_frozen_winners_retained": True, "retuning_performed": False,
        "wandb_required_for_scientific_validity": False, "timing_usable": False,
    })


def run_verify(args: argparse.Namespace) -> None:
    formal_units_path = args.run_dir / "formal_units_manifest.json"
    require_pass(formal_units_path, "formal")
    paths = frozen_paths(args)
    inputs = prepare_accepted_inputs(args)
    extended_selection_audit = audit_extended_selection(inputs["extended_selection"])
    mousse_selection_audit = audit_mousse_selection(inputs["mousse_selection"])
    malt_winner_audit = audit_malt_winner_selection(inputs["malt_selection"])
    preformal_manifest = run_preformal_sensitivity(args, paths, inputs)
    inventory = frozen_data_inventory(args, full_verify=True)
    checkpoints = verify_checkpoint_certificates(formal_units_path)
    checkpoint_manifest = args.run_dir / "checkpoint_verification_manifest.json"
    write_json(checkpoint_manifest, {
        "schema_version": 1,
        "experiment_id": EXPERIMENT,
        "passed": True,
        "formal_units": len(checkpoints),
        "checkpoints": checkpoints,
        "data_inventory": str(inventory),
        "data_inventory_sha256": sha256_file(inventory),
        "full_data_content_reverified": True,
        "verified_at": datetime.now().astimezone().isoformat(),
    })
    analyzer, analysis_amendment = resolve_analysis_analyzer(args, paths["analyzer"])
    analysis_dir = args.run_dir / "analysis"
    manifest = analysis_dir / "analysis_manifest.json"
    if not analysis_manifest_matches(
        manifest, contract=paths["contract"], historical_panel=inputs["historical_panel"],
        formal_units=formal_units_path, analyzer=analyzer,
        analysis_amendment=analysis_amendment,
    ):
        command = [
            sys.executable, str(analyzer), "--run-dir", str(args.run_dir),
            "--contract", str(paths["contract"]), "--historical-panel", str(inputs["historical_panel"]),
            "--formal-units", str(args.run_dir / "formal_units_manifest.json"),
            "--output-dir", str(analysis_dir),
        ]
        if analysis_amendment is not None:
            command.extend(["--analysis-amendment", str(analysis_amendment)])
        append_command(args.run_dir, {"label": "verify", "command": command, "command_text": shlex.join(command)})
        subprocess.run(command, cwd=args.repo, check=True)
    if not analysis_manifest_matches(
        manifest, contract=paths["contract"], historical_panel=inputs["historical_panel"],
        formal_units=formal_units_path, analyzer=analyzer,
        analysis_amendment=analysis_amendment,
    ):
        raise RuntimeError("EX55 final analysis failed integrity validation")
    write_json(args.run_dir / "handoff_manifest.json", {
        "schema_version": 1, "experiment_id": EXPERIMENT, "status": "completed", "passed": True,
        "scientific_result": read_json(manifest)["classification"],
        "formal_seed": FORMAL_SEED, "formal_units": 10,
        "historical_panel_preserved": True, "repaired_panel_seeds": [2024, 2025, 2027],
        "analysis_manifest": str(manifest), "analysis_manifest_sha256": sha256_file(manifest),
        "analysis_analyzer": file_record(analyzer),
        "analysis_amendment": file_record(analysis_amendment) if analysis_amendment is not None else None,
        "checkpoint_verification_manifest": str(checkpoint_manifest),
        "checkpoint_verification_manifest_sha256": sha256_file(checkpoint_manifest),
        "preformal_sensitivity_manifest": str(preformal_manifest),
        "preformal_sensitivity_manifest_sha256": sha256_file(preformal_manifest),
        "extended_selection_audit": extended_selection_audit,
        "mousse_selection_audit": mousse_selection_audit,
        "malt_winner_selection_audit": malt_winner_audit,
        "timing_usable": False, "completed_at": datetime.now().astimezone().isoformat(),
    })
    print("Experiment 55 completed.")
    print(f"Artifacts: {args.run_dir}")
    print(f"Analysis: {manifest}")


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.expanduser().absolute()
    args.repo = args.repo.expanduser().resolve()
    args.official_repo = args.official_repo.expanduser().resolve()
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.resume:
        raise RuntimeError(f"run directory is nonempty; pass --resume: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    paths = frozen_paths(args)
    contract = read_json(paths["contract"])
    prepare_accepted_inputs(args)
    write_json(args.run_dir / "suite_plan.json", {
        "schema_version": 1, "script_version": SCRIPT_VERSION, "experiment_id": EXPERIMENT,
        "stage_requested": args.stage, "run_dir": str(args.run_dir), "repo": str(args.repo),
        "official_repo": str(args.official_repo), "controller_python": sys.executable,
        "training_python": args.training_python, "gpus": args.gpus,
        "engineering_seed": ENGINEERING_SEED, "formal_seed": FORMAL_SEED,
        "methods": list(SELECTED_CELLS), "selected_cells": SELECTED_CELLS,
        "contract_sha256": sha256_file(paths["contract"]), "wandb_mode": args.wandb_mode,
        "retuning_performed": False, "timing_usable": False,
    })
    stages = ("preflight", "pilot", "formal", "verify") if args.stage == "all" else (args.stage,)
    dispatch = {"preflight": run_preflight, "pilot": run_pilot, "formal": run_formal, "verify": run_verify}
    for stage in stages:
        dispatch[stage](args)


if __name__ == "__main__":
    main()
