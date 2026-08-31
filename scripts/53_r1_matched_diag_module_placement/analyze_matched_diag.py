#!/usr/bin/env python3
"""Fail-closed verification and paired analysis for Experiment 53."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-08-17.2"
ARMS = (
    "all_none",
    "c_fc_diag",
    "c_proj_diag",
    "c_fc_c_proj_diag",
    "o_proj_diag",
)
FORMAL_SEEDS = (2024, 2025, 2026)
SINGLE_MODULE_ARMS = ("c_fc_diag", "c_proj_diag", "o_proj_diag")
T_95_DF2 = 4.302652729911275
OFFICIAL_COMMIT = "df78af0db523d8bceb25af4919a3e3e7082b80f3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_run_path(run_dir: Path, relative: str, label: str) -> Path:
    candidate = (run_dir / relative).resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes Experiment-53 run directory: {relative}") from exc
    return candidate


def validate_source_snapshot(run_dir: Path) -> str:
    root = run_dir / "source_snapshot"
    manifest_path = root / "source_snapshot_manifest.json"
    payload = read_json(manifest_path)
    files = payload.get("files")
    if (
        payload.get("experiment_id") != "53_r1_matched_diag_module_placement"
        or payload.get("passed") is not True
        or not isinstance(files, list)
        or payload.get("file_count") != len(files)
        or not files
    ):
        raise RuntimeError("invalid Experiment-53 source snapshot manifest")
    seen: set[str] = set()
    for item in files:
        relative = str(item.get("path", ""))
        if relative in seen:
            raise RuntimeError(f"duplicate source snapshot entry: {relative}")
        seen.add(relative)
        path = safe_run_path(root, relative, "source snapshot entry")
        if (
            not path.is_file()
            or path.stat().st_size != int(item.get("bytes", -1))
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"source snapshot integrity failure: {path}")
    required = {
        "scripts/14_official_newton_muon_r0/run_official_newton_muon_r0.py",
        "scripts/15_official_newton_muon_r1/run_official_newton_muon_r1.py",
        "scripts/50_r1_global_activation_diag/global_diag_source_builder.py",
        "scripts/53_r1_matched_diag_module_placement/matched_diag_contract.json",
        "scripts/53_r1_matched_diag_module_placement/matched_diag_source_builder.py",
        "scripts/53_r1_matched_diag_module_placement/run_matched_diag.py",
        "scripts/53_r1_matched_diag_module_placement/run_matched_diag_suite.py",
        "scripts/53_r1_matched_diag_module_placement/analyze_matched_diag.py",
    }
    if not required.issubset(seen):
        raise RuntimeError(
            f"source snapshot misses required lineage files: {sorted(required - seen)}"
        )
    return sha256_file(manifest_path)


def _accepted_relative_map(
    run_dir: Path, payload: dict[str, Any], expected: set[str], label: str
) -> dict[str, Path]:
    mapping = payload.get("accepted_batches")
    if not isinstance(mapping, dict) or set(mapping) != expected:
        raise RuntimeError(f"{label} accepted-batch grid mismatch")
    output: dict[str, Path] = {}
    for key, relative in mapping.items():
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise RuntimeError(f"{label} accepted path is not relocatable: {relative!r}")
        path = safe_run_path(run_dir, relative, f"{label} accepted batch")
        if path.name != "r1_manifest.json" or not path.is_file():
            raise RuntimeError(f"{label} accepted manifest is missing: {path}")
        output[str(key)] = path
    return output


def validate_batch_lineage(
    batch_path: Path,
    *,
    arm: str,
    seed: int,
    pilot: bool,
    expected_source_sha256: str,
    expected_data_dir: str,
    pilot_manifest_sha256: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    batch = read_json(batch_path)
    expected_status = (
        {"completed_valid_smoke"}
        if pilot
        else {"completed_valid", "completed_valid_local_wandb_incomplete"}
    )
    expected_protocol = (
        "r1_matched_diag_module_placement_engineering_pilot"
        if pilot
        else "r1_matched_diag_module_placement_formal"
    )
    checks = {
        "family": batch.get("family") == "53_r1_matched_diag_module_placement",
        "protocol": batch.get("protocol") == expected_protocol,
        "batch_kind": batch.get("batch_kind") == ("smoke" if pilot else "formal"),
        "status": batch.get("status") in expected_status,
        "official_commit": batch.get("official_commit") == OFFICIAL_COMMIT,
        "methods": batch.get("methods") == [arm],
        "seed": batch.get("seed") == seed,
        "failures": batch.get("failures") == [],
        "formal_evidence": batch.get("formal_evidence") is (not pilot),
        "profile": batch.get("evidence_profile")
        == ("exact_shape_numerical_smoke" if pilot else "formal"),
        "steps": batch.get("smoke_steps") == (34 if pilot else None),
        "source": batch.get("derived_source_sha256") == {arm: expected_source_sha256},
    }
    isolation = batch.get("resource_isolation")
    checks["gpu_isolation"] = (
        isinstance(isolation, dict)
        and isolation.get("one_process_one_gpu") is True
        and isolation.get("visible_device_count") == 1
    )
    data = batch.get("data")
    checks["data"] = (
        isinstance(data, dict)
        and str(Path(str(data.get("data_dir", ""))).resolve())
        == str(Path(expected_data_dir).resolve())
        and int(data.get("train_shards", 0)) >= 50
        and int(data.get("validation_shards", 0)) >= 1
        and data.get("first_train_shard") == "fineweb_train_000001.bin"
        and data.get("first_validation_shard") == "fineweb_val_000000.bin"
        and data.get("data_magic") == 20240520
    )
    audit = batch.get("initialization_audit")
    checks["init"] = (
        isinstance(audit, dict)
        and audit.get("seed") == seed
        and audit.get("all_methods_identical") is True
        and isinstance(audit.get("init_sha256"), str)
        and len(str(audit.get("init_sha256"))) == 64
    )
    runtime = batch.get("training_runtime_fingerprint")
    checks["runtime"] = (
        isinstance(runtime, dict)
        and all(
            runtime.get(key) not in (None, "")
            for key in (
                "python_executable",
                "python_version",
                "numpy",
                "torch",
                "torch_cuda",
                "triton",
                "gpu_name",
                "gpu_total_memory_bytes",
            )
        )
        and "H100" in str(runtime.get("gpu_name", "")).upper()
        and int(runtime.get("gpu_total_memory_bytes", 0)) >= 80_000_000_000
    )
    if pilot:
        checks["no_smoke_certificate"] = batch.get("smoke_certificate") is None
    else:
        certificate = batch.get("smoke_certificate")
        checks["pilot_certificate"] = (
            isinstance(certificate, dict)
            and certificate.get("validated") is True
            and certificate.get("engineering_seed") == 2053
            and certificate.get("formal_seed_independent") is True
            and certificate.get("outcome_eligible") is False
            and certificate.get("manifest_sha256") == pilot_manifest_sha256
        )
    summaries = batch.get("summaries")
    checks["one_summary"] = isinstance(summaries, list) and len(summaries) == 1
    if not all(checks.values()):
        raise RuntimeError(f"Experiment-53 batch lineage failed for {batch_path}: {checks}")
    batch_summary = summaries[0]
    run_name = str(batch_summary.get("run_name", ""))
    if not run_name or Path(run_name).name != run_name:
        raise RuntimeError(f"invalid Experiment-53 run name: {run_name!r}")
    summary_path = batch_path.parent / run_name / "r1_summary.json"
    run_manifest_path = summary_path.with_name("run_manifest.json")
    if not summary_path.is_file() or not run_manifest_path.is_file():
        raise RuntimeError(f"missing Experiment-53 accepted run artifacts: {summary_path}")
    summary = read_json(summary_path)
    run_manifest = read_json(run_manifest_path)
    for key, value in batch_summary.items():
        if summary.get(key) != value:
            raise RuntimeError(f"batch/file summary mismatch for {arm}/seed{seed}: {key}")
    source = run_manifest.get("source")
    run_checks = {
        "status": run_manifest.get("status") == "completed_valid_smoke"
        if pilot
        else run_manifest.get("status") == "completed_valid",
        "family": run_manifest.get("experiment_family")
        == "53_r1_matched_diag_module_placement",
        "protocol": run_manifest.get("protocol") == expected_protocol,
        "method": run_manifest.get("method") == arm,
        "seed": run_manifest.get("controlled_seed") == seed,
        "formal": run_manifest.get("formal_evidence") is (not pilot),
        "source": isinstance(source, dict)
        and source.get("derived_script_sha256") == expected_source_sha256,
    }
    if not all(run_checks.values()):
        raise RuntimeError(
            f"Experiment-53 run-manifest lineage failed for {run_manifest_path}: {run_checks}"
        )
    source_manifest_path = summary_path.with_name("source_manifest.json")
    if not source_manifest_path.is_file():
        raise RuntimeError(f"missing Experiment-53 source manifest: {source_manifest_path}")
    source_manifest = read_json(source_manifest_path)
    if not isinstance(source, dict) or any(
        source_manifest.get(key) != value for key, value in source.items()
    ):
        raise RuntimeError(f"source-manifest embedding mismatch: {source_manifest_path}")
    trainers = list((summary_path.parent / "workspace").glob("train_r1_*.py"))
    if len(trainers) != 1 or sha256_file(trainers[0]) != expected_source_sha256:
        raise RuntimeError(f"derived trainer integrity failed for {summary_path.parent}")
    return summary_path, batch, run_manifest


def validate_evidence_graph(
    run_dir: Path, contract_path: Path
) -> tuple[dict[tuple[int, str], Path], dict[str, Any]]:
    contract_sha = sha256_file(contract_path)
    snapshot_sha = validate_source_snapshot(run_dir)
    frozen_contract = (
        run_dir
        / "source_snapshot/scripts/53_r1_matched_diag_module_placement/matched_diag_contract.json"
    )
    if sha256_file(frozen_contract) != contract_sha:
        raise RuntimeError("Experiment-53 analyzer contract differs from frozen snapshot")
    preflight_path = run_dir / "preflight_manifest.json"
    pilot_suite_path = run_dir / "pilot_manifest.json"
    formal_suite_path = run_dir / "formal_manifest.json"
    inventory_path = run_dir / "data_inventory.json"
    receipt_path = run_dir / "data_verify_receipt.json"
    preflight = read_json(preflight_path)
    pilot_suite = read_json(pilot_suite_path)
    formal_suite = read_json(formal_suite_path)
    inventory = read_json(inventory_path)
    receipt = read_json(receipt_path)
    inventory_sha = sha256_file(inventory_path)
    entries = inventory.get("entries")
    if not isinstance(entries, list) or len(entries) != 51:
        raise RuntimeError("Experiment-53 frozen data inventory is incomplete")
    expected_names = [
        *(f"fineweb_train_{index:06d}.bin" for index in range(1, 51)),
        "fineweb_val_000000.bin",
    ]
    if [item.get("name") for item in entries] != expected_names:
        raise RuntimeError("Experiment-53 frozen data shard set/order is invalid")
    for index, item in enumerate(entries):
        expected_split = "validation" if index == 50 else "train"
        expected_index = 0 if index == 50 else index + 1
        if (
            item.get("split") != expected_split
            or item.get("index") != expected_index
            or item.get("magic") != 20240520
            or int(item.get("bytes", 0)) <= 4
            or not isinstance(item.get("sha256"), str)
            or len(str(item.get("sha256"))) != 64
        ):
            raise RuntimeError(f"invalid Experiment-53 data inventory entry: {item}")
    stable_entries = [
        {
            "name": item.get("name"),
            "split": item.get("split"),
            "index": item.get("index"),
            "bytes": item.get("bytes"),
            "magic": item.get("magic"),
            "sha256": item.get("sha256"),
        }
        for item in entries
    ]
    projection = hashlib.sha256(
        json.dumps(stable_entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    data_dir = str(inventory.get("data_dir", ""))
    for item in entries:
        shard = Path(data_dir) / str(item["name"])
        if not shard.is_file() or shard.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"Experiment-53 data changed after full-hash receipt: {shard}")
    top_checks = {
        "preflight_passed": preflight.get("passed") is True,
        "preflight_contract": preflight.get("contract_sha256") == contract_sha,
        "preflight_snapshot": preflight.get("source_snapshot_manifest_sha256")
        == snapshot_sha,
        "preflight_data_inventory": preflight.get("data_inventory_sha256")
        == inventory_sha,
        "preflight_data_projection": preflight.get("data_content_projection_sha256")
        == projection,
        "inventory_passed": inventory.get("passed") is True,
        "inventory_projection": inventory.get("content_projection_sha256") == projection,
        "receipt_passed": receipt.get("passed") is True,
        "receipt_full_hash": receipt.get("full_content_rehash") is True,
        "receipt_inventory": receipt.get("data_inventory_sha256") == inventory_sha,
        "receipt_projection": receipt.get("data_content_projection_sha256") == projection,
        "pilot_suite": pilot_suite.get("passed") is True,
        "pilot_seed": pilot_suite.get("seed") == 2053,
        "pilot_steps": pilot_suite.get("steps") == 34,
        "pilot_outcome_ineligible": pilot_suite.get("outcome_eligible") is False,
        "pilot_no_selection": pilot_suite.get("configuration_selection_allowed") is False,
        "pilot_arms": pilot_suite.get("arms") == list(ARMS),
        "pilot_inventory": pilot_suite.get("data_inventory_sha256") == inventory_sha,
        "pilot_snapshot": pilot_suite.get("source_snapshot_manifest_sha256") == snapshot_sha,
        "formal_suite": formal_suite.get("passed") is True,
        "formal_seeds": formal_suite.get("formal_seeds") == list(FORMAL_SEEDS),
        "formal_arms": formal_suite.get("arms") == list(ARMS),
        "formal_units": formal_suite.get("formal_units") == 15,
        "formal_inventory": formal_suite.get("data_inventory_sha256") == inventory_sha,
        "formal_snapshot": formal_suite.get("source_snapshot_manifest_sha256") == snapshot_sha,
        "wandb_secondary": formal_suite.get("wandb_required_for_scientific_validity")
        is False,
        "timing_ineligible": formal_suite.get("timing_usable") is False,
    }
    expected_source = preflight.get("derived_script_sha256")
    top_checks["derived_source"] = isinstance(expected_source, str) and len(expected_source) == 64
    if not all(top_checks.values()):
        raise RuntimeError(f"Experiment-53 top-level evidence graph failed: {top_checks}")
    pilot_paths = _accepted_relative_map(
        run_dir, pilot_suite, set(ARMS), "pilot"
    )
    pilot_hashes: dict[str, str] = {}
    pilot_init_hashes: set[str] = set()
    runtimes: list[str] = []
    for arm in ARMS:
        summary_path, batch, _run = validate_batch_lineage(
            pilot_paths[arm],
            arm=arm,
            seed=2053,
            pilot=True,
            expected_source_sha256=str(expected_source),
            expected_data_dir=data_dir,
        )
        pilot_hashes[arm] = sha256_file(pilot_paths[arm])
        pilot_init_hashes.add(str(read_json(summary_path).get("init_sha256")))
        runtimes.append(
            json.dumps(batch.get("training_runtime_fingerprint"), sort_keys=True)
        )
    if len(pilot_init_hashes) != 1:
        raise RuntimeError("Experiment-53 pilot arms do not share one initialization")
    formal_keys = {f"seed{seed}/{arm}" for seed in FORMAL_SEEDS for arm in ARMS}
    formal_paths = _accepted_relative_map(
        run_dir, formal_suite, formal_keys, "formal"
    )
    summaries: dict[tuple[int, str], Path] = {}
    lineage_artifacts: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        for arm in ARMS:
            key = f"seed{seed}/{arm}"
            summary_path, batch, run_manifest = validate_batch_lineage(
                formal_paths[key],
                arm=arm,
                seed=seed,
                pilot=False,
                expected_source_sha256=str(expected_source),
                expected_data_dir=data_dir,
                pilot_manifest_sha256=pilot_hashes[arm],
            )
            summaries[(seed, arm)] = summary_path
            runtimes.append(
                json.dumps(batch.get("training_runtime_fingerprint"), sort_keys=True)
            )
            artifact_paths = {
                "batch_manifest": formal_paths[key],
                "run_manifest": summary_path.with_name("run_manifest.json"),
                "summary": summary_path,
                "metrics": summary_path.with_name("r1_metrics.csv"),
                "stdout": summary_path.with_name("training_stdout.log"),
                "training_log": summary_path.with_name("training_log_with_source.txt"),
                "source_manifest": summary_path.with_name("source_manifest.json"),
            }
            if any(not path.is_file() for path in artifact_paths.values()):
                raise RuntimeError(f"formal unit artifact set is incomplete for {key}")
            lineage_artifacts.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "files": {
                        label: {
                            "path": path.relative_to(run_dir).as_posix(),
                            "bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                        for label, path in artifact_paths.items()
                    },
                    "wandb_status": (
                        run_manifest.get("wandb", {}).get("status")
                        if isinstance(run_manifest.get("wandb"), dict)
                        else "unknown"
                    ),
                }
            )
    if len(set(runtimes)) != 1:
        raise RuntimeError("Experiment-53 pilot/formal runtime fingerprints are not identical")
    return summaries, {
        "contract_sha256": contract_sha,
        "source_snapshot_manifest_sha256": snapshot_sha,
        "data_inventory_sha256": inventory_sha,
        "data_verify_receipt_sha256": sha256_file(receipt_path),
        "preflight_manifest_sha256": sha256_file(preflight_path),
        "pilot_manifest_sha256": sha256_file(pilot_suite_path),
        "formal_manifest_sha256": sha256_file(formal_suite_path),
        "data_content_projection_sha256": projection,
        "derived_script_sha256": expected_source,
        "pilot_batch_sha256": pilot_hashes,
        "formal_unit_artifacts": lineage_artifacts,
    }


def validation_curve(
    summary_path: Path, *, arm: str, seed: int
) -> list[dict[str, Any]]:
    metrics_path = summary_path.with_name("r1_metrics.csv")
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        all_rows = [dict(row) for row in csv.DictReader(handle)]
    if not all_rows:
        raise RuntimeError(f"empty Experiment-53 metrics: {metrics_path}")
    for row in all_rows:
        step = int(row["step"])
        if (
            row.get("method") != arm
            or int(row.get("total_steps", -1)) != 6200
            or int(row.get("tokens_seen", -1)) != step * 524288
            or not math.isfinite(float(row["loss"]))
        ):
            raise RuntimeError(f"invalid Experiment-53 metric row: {metrics_path}: {row}")
    rows = sorted(
        [row for row in all_rows if row.get("event") == "validation"],
        key=lambda row: int(row["step"]),
    )
    train = sorted(
        [row for row in all_rows if row.get("event") == "train"],
        key=lambda row: int(row["step"]),
    )
    if [int(row["step"]) for row in rows] != list(range(0, 6201, 100)):
        raise RuntimeError(f"invalid validation grid: {metrics_path}")
    if [int(row["step"]) for row in train] != list(range(1, 6201)):
        raise RuntimeError(f"invalid train grid: {metrics_path}")
    return rows


def normalized_auc(rows: list[dict[str, Any]]) -> float:
    area = 0.0
    for left, right in zip(rows, rows[1:]):
        width = int(right["step"]) - int(left["step"])
        area += width * (float(left["loss"]) + float(right["loss"])) / 2.0
    span = int(rows[-1]["step"]) - int(rows[0]["step"])
    if span <= 0:
        raise RuntimeError("validation curve has no positive span")
    return area / span


def collect_formal(
    run_dir: Path,
    contract: dict[str, Any],
    accepted_summaries: dict[tuple[int, str], Path],
) -> list[dict[str, Any]]:
    expected_memory = contract["expected_memory_bytes"]
    collected: dict[tuple[int, str], dict[str, Any]] = {}
    for expected_key, summary_path in sorted(accepted_summaries.items()):
        payload = read_json(summary_path)
        arm = str(payload.get("method"))
        seed = int(payload.get("controlled_seed", -1))
        if (seed, arm) != expected_key:
            raise RuntimeError(
                f"accepted summary identity mismatch: {(seed, arm)} != {expected_key}"
            )
        key = (seed, arm)
        if key in collected:
            raise RuntimeError(f"multiple formal summaries for {key}")
        manifest_path = summary_path.with_name("run_manifest.json")
        manifest = read_json(manifest_path)
        if manifest.get("status") not in {
            "completed_valid",
            "completed_valid_local_wandb_incomplete",
        }:
            raise RuntimeError(f"formal unit is not accepted: {manifest_path}")
        if payload.get("formal_evidence") is not True or payload.get("evidence_valid") is not True:
            raise RuntimeError(f"formal evidence flags failed: {summary_path}")
        curve = validation_curve(summary_path, arm=arm, seed=seed)
        memory = {
            name: int(payload[name])
            for name in (
                "k_cov_bytes",
                "k_inv_bytes",
                "k_state_bytes",
                "activation_stat_bytes",
                "precond_workspace_bytes",
            )
        }
        if memory != expected_memory[arm]:
            raise RuntimeError(
                f"memory contract mismatch for {key}: {memory} != {expected_memory[arm]}"
            )
        config = next(item for item in contract["arms"] if item["arm"] == arm)
        observed_modes = {
            "c_fc": str(payload["cfc_k_mode"]),
            "c_proj": str(payload["cproj_k_mode"]),
            "o_proj": str(payload["oproj_k_mode"]),
            "qkv": str(payload["qkv_k_mode"]),
        }
        expected_modes = {name: str(config[name]) for name in observed_modes}
        if observed_modes != expected_modes:
            raise RuntimeError(
                f"module-mode contract mismatch for {key}: {observed_modes} != {expected_modes}"
            )
        stdout = summary_path.with_name("training_stdout.log").read_text(
            encoding="utf-8", errors="replace"
        )
        metadata_line = (
            "R1_MATCHED_DIAG_METADATA "
            f"arm={arm} cfc={expected_modes['c_fc']} cproj={expected_modes['c_proj']} "
            f"oproj={expected_modes['o_proj']} qkv=none dense_workspace=0"
        )
        input_count = (12 if expected_modes["c_fc"] == "diag" else 0) + (
            12 if expected_modes["o_proj"] == "diag" else 0
        )
        proj_count = 12 if expected_modes["c_proj"] == "diag" else 0
        route_line = (
            "R1_MATCHED_DIAG_ROUTE "
            f"arm={arm} input_diag_params={input_count} "
            f"proj_diag_params={proj_count} dense_refresh_blocks=0"
        )
        if stdout.splitlines().count(metadata_line) != 1 or stdout.splitlines().count(route_line) != 1:
            raise RuntimeError(f"runtime layer-coverage/route certificate failed for {key}")
        if (
            int(payload.get("final_train_step", -1)) != 6200
            or int(payload.get("train_points", -1)) != 6200
            or int(payload.get("validation_points", -1)) != 63
            or float(payload.get("base_learning_rate", math.nan)) != 0.004
            or float(payload.get("matrix_learning_rate", math.nan)) != 0.0004
        ):
            raise RuntimeError(f"training-budget/learning-rate mismatch for {key}")
        checkpoint_relative = payload.get("checkpoint_relative_path")
        checkpoint_sha = payload.get("checkpoint_sha256")
        if (
            not isinstance(checkpoint_relative, str)
            or Path(checkpoint_relative).is_absolute()
            or not isinstance(checkpoint_sha, str)
            or len(checkpoint_sha) != 64
        ):
            raise RuntimeError(f"checkpoint certificate missing for {key}")
        checkpoint = safe_run_path(
            summary_path.parent, checkpoint_relative, "formal checkpoint"
        )
        if (
            not checkpoint.is_file()
            or checkpoint.stat().st_size != int(payload.get("checkpoint_bytes", -1))
            or sha256_file(checkpoint) != checkpoint_sha
        ):
            raise RuntimeError(f"checkpoint integrity failure for {key}: {checkpoint}")
        row = {
            "seed": seed,
            "arm": arm,
            "c_fc": str(payload["cfc_k_mode"]),
            "c_proj": str(payload["cproj_k_mode"]),
            "o_proj": str(payload["oproj_k_mode"]),
            "qkv": str(payload["qkv_k_mode"]),
            "final_val_loss": float(payload["final_val_loss"]),
            "tail5_val_loss_mean": statistics.mean(
                float(item["loss"]) for item in curve[-5:]
            ),
            "normalized_val_auc": normalized_auc(curve),
            "best_val_loss": float(payload["best_val_loss"]),
            "final_val_step": int(payload["final_val_step"]),
            "validation_points": len(curve),
            "init_sha256": str(payload["init_sha256"]),
            "derived_script_sha256": str(payload["derived_script_sha256"]),
            **memory,
            "k_state_mib": memory["k_state_bytes"] / (1024**2),
            "optimizer_state_bytes": int(payload["optimizer_state_bytes"]),
            "peak_memory_allocated_mib": int(payload["peak_memory_allocated_mib"]),
            "timing_usable": False,
            "source_summary": summary_path.relative_to(run_dir).as_posix(),
            "checkpoint_relative_path": checkpoint.relative_to(run_dir).as_posix(),
            "checkpoint_sha256": checkpoint_sha,
        }
        if not all(
            math.isfinite(float(row[key]))
            for key in (
                "final_val_loss",
                "tail5_val_loss_mean",
                "normalized_val_auc",
                "best_val_loss",
            )
        ):
            raise RuntimeError(f"nonfinite formal metric in {key}: {row}")
        if row["qkv"] != "none" or row["precond_workspace_bytes"] != 0:
            raise RuntimeError(f"forbidden route reached in {key}: {row}")
        collected[key] = row
    expected = {(seed, arm) for seed in FORMAL_SEEDS for arm in ARMS}
    if set(collected) != expected:
        raise RuntimeError(
            f"formal grid mismatch: missing={sorted(expected - set(collected))} "
            f"extra={sorted(set(collected) - expected)}"
        )
    rows = [collected[(seed, arm)] for seed in FORMAL_SEEDS for arm in ARMS]
    for seed in FORMAL_SEEDS:
        hashes = {row["init_sha256"] for row in rows if row["seed"] == seed}
        if len(hashes) != 1:
            raise RuntimeError(f"seed {seed} has unpaired initialization hashes: {hashes}")
    if len({row["init_sha256"] for row in rows}) != len(FORMAL_SEEDS):
        raise RuntimeError("formal seed initialization hashes are not distinct")
    if len({row["derived_script_sha256"] for row in rows}) != 1:
        raise RuntimeError("formal arms do not share one parameterized derived source")
    return rows


def mean_ci(values: list[float]) -> dict[str, float | int]:
    if len(values) != 3:
        raise RuntimeError(f"Experiment-53 paired aggregate expects n=3, got {len(values)}")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T_95_DF2 * sd / math.sqrt(3)
    return {
        "n": 3,
        "mean": mean,
        "sample_sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "negative_seed_count": sum(value < 0 for value in values),
        "positive_seed_count": sum(value > 0 for value in values),
        "zero_seed_count": sum(value == 0 for value in values),
    }


def paired_rows(formal: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(int(row["seed"]), str(row["arm"])): row for row in formal}
    comparisons = (
        ("c_fc_diag_minus_all_none", "c_fc_diag", "all_none"),
        ("c_proj_diag_minus_all_none", "c_proj_diag", "all_none"),
        ("o_proj_diag_minus_all_none", "o_proj_diag", "all_none"),
        ("c_fc_c_proj_diag_minus_all_none", "c_fc_c_proj_diag", "all_none"),
        ("c_fc_diag_minus_c_proj_diag", "c_fc_diag", "c_proj_diag"),
        ("c_fc_diag_minus_o_proj_diag", "c_fc_diag", "o_proj_diag"),
        ("c_proj_diag_minus_o_proj_diag", "c_proj_diag", "o_proj_diag"),
    )
    per_seed: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    for contrast, left, right in comparisons:
        values: list[float] = []
        for seed in FORMAL_SEEDS:
            left_loss = float(lookup[(seed, left)]["final_val_loss"])
            right_loss = float(lookup[(seed, right)]["final_val_loss"])
            delta = left_loss - right_loss
            values.append(delta)
            per_seed.append(
                {
                    "contrast": contrast,
                    "seed": seed,
                    "method_a": left,
                    "method_b": right,
                    "method_a_final_val_loss": left_loss,
                    "method_b_final_val_loss": right_loss,
                    "delta_a_minus_b": delta,
                    "lower_is_better": True,
                }
            )
        aggregate.append(
            {
                "contrast": contrast,
                "method_a": left,
                "method_b": right,
                "delta_definition": "method_a_minus_method_b; negative favors method_a",
                **mean_ci(values),
            }
        )
    return per_seed, aggregate


def factorial_rows(formal: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(int(row["seed"]), str(row["arm"])): float(row["final_val_loss"]) for row in formal}
    per_seed: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        none = lookup[(seed, "all_none")]
        fc = lookup[(seed, "c_fc_diag")]
        proj = lookup[(seed, "c_proj_diag")]
        both = lookup[(seed, "c_fc_c_proj_diag")]
        per_seed.append(
            {
                "seed": seed,
                "c_fc_main_effect": ((fc - none) + (both - proj)) / 2.0,
                "c_proj_main_effect": ((proj - none) + (both - fc)) / 2.0,
                "factorial_interaction": both - fc - proj + none,
                "effect_definition": "retained_diag_minus_removed_diag; negative lowers loss",
            }
        )
    aggregate: list[dict[str, Any]] = []
    for effect in ("c_fc_main_effect", "c_proj_main_effect", "factorial_interaction"):
        aggregate.append({"effect": effect, **mean_ci([float(row[effect]) for row in per_seed])})
    return per_seed, aggregate


def method_summary(formal: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    anchor = {int(row["seed"]): float(row["final_val_loss"]) for row in formal if row["arm"] == "all_none"}
    for arm in ARMS:
        selected = [row for row in formal if row["arm"] == arm]
        losses = [float(row["final_val_loss"]) for row in selected]
        state_mib = statistics.mean(float(row["k_state_mib"]) for row in selected)
        mean_loss = statistics.mean(losses)
        benefits = [anchor[int(row["seed"])] - float(row["final_val_loss"]) for row in selected]
        output.append(
            {
                "arm": arm,
                "n": 3,
                "mean_final_val_loss": mean_loss,
                "sample_sd_final_val_loss": statistics.stdev(losses),
                "mean_tail5_val_loss": statistics.mean(float(row["tail5_val_loss_mean"]) for row in selected),
                "mean_normalized_val_auc": statistics.mean(float(row["normalized_val_auc"]) for row in selected),
                "mean_k_state_mib": state_mib,
                "mean_optimizer_state_bytes": statistics.mean(float(row["optimizer_state_bytes"]) for row in selected),
                "mean_peak_memory_allocated_mib": statistics.mean(float(row["peak_memory_allocated_mib"]) for row in selected),
                "mean_loss_benefit_vs_all_none": statistics.mean(benefits),
                "mean_loss_benefit_per_k_state_mib": statistics.mean(benefits) / state_mib if state_mib > 0 else "",
                "timing_usable": False,
            }
        )
    output.sort(key=lambda row: float(row["mean_final_val_loss"]))
    for rank, row in enumerate(output, start=1):
        row["descriptive_mean_loss_rank"] = rank
    return output


def analyze(run_dir: Path, contract_path: Path, output_dir: Path) -> dict[str, Any]:
    contract = read_json(contract_path)
    if contract.get("experiment_id") != "53_r1_matched_diag_module_placement":
        raise RuntimeError("wrong Experiment-53 contract")
    accepted_summaries, evidence_graph = validate_evidence_graph(
        run_dir, contract_path
    )
    formal = collect_formal(run_dir, contract, accepted_summaries)
    per_seed, aggregates = paired_rows(formal)
    effects_by_seed, effects = factorial_rows(formal)
    summaries = method_summary(formal)
    best = str(summaries[0]["arm"])
    checks = {
        "formal_unit_count": len(formal) == 15,
        "formal_grid_exact": {(row["seed"], row["arm"]) for row in formal}
        == {(seed, arm) for seed in FORMAL_SEEDS for arm in ARMS},
        "endpoints_exact": all(row["final_val_step"] == 6200 for row in formal),
        "finite_losses": all(math.isfinite(float(row["final_val_loss"])) for row in formal),
        "paired_initialization": all(
            len({row["init_sha256"] for row in formal if row["seed"] == seed}) == 1
            for seed in FORMAL_SEEDS
        ),
        "one_parameterized_source": len({row["derived_script_sha256"] for row in formal}) == 1,
        "qkv_none": all(row["qkv"] == "none" for row in formal),
        "dense_workspace_forbidden": all(row["precond_workspace_bytes"] == 0 for row in formal),
        "timing_excluded": contract["execution_policy"]["timing_usable"] is False,
        "pilot_outcome_ineligible": contract["pilot"]["outcome_eligible"] is False,
        "source_snapshot_bound": bool(
            evidence_graph["source_snapshot_manifest_sha256"]
        ),
        "data_content_rehashed": bool(
            evidence_graph["data_content_projection_sha256"]
        ),
        "checkpoint_hashes_verified": all(
            isinstance(row.get("checkpoint_sha256"), str) for row in formal
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Experiment-53 analysis checks failed: {checks}")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "formal_results": output_dir / "formal_results.csv",
        "method_summary": output_dir / "method_summary.csv",
        "paired_contrasts": output_dir / "paired_contrasts_by_seed.csv",
        "aggregate_contrasts": output_dir / "aggregate_contrasts.csv",
        "factorial_by_seed": output_dir / "factorial_effects_by_seed.csv",
        "factorial_aggregate": output_dir / "factorial_effects_aggregate.csv",
    }
    write_csv(outputs["formal_results"], formal)
    write_csv(outputs["method_summary"], summaries)
    write_csv(outputs["paired_contrasts"], per_seed)
    write_csv(outputs["aggregate_contrasts"], aggregates)
    write_csv(outputs["factorial_by_seed"], effects_by_seed)
    write_csv(outputs["factorial_aggregate"], effects)
    report_path = output_dir / "EXPERIMENT_53_ANALYSIS.md"
    report_path.write_text(
        "# Experiment 53: matched diagonal module placement\n\n"
        f"- integrity: `passed` (15/15 formal units)\n"
        f"- descriptive lowest endpoint mean: `{best}`\n"
        "- primary evidence: final-step validation loss with within-seed paired contrasts\n"
        "- factorial: `c_fc x c_proj` diagonal/none, including per-seed effects\n"
        "- attention extension: the same diagonal representation at `o_proj`\n"
        "- QKV route: `none` in every arm\n"
        "- dense preconditioner workspace: forbidden and observed zero\n"
        "- timing: ineligible\n\n"
        "The rank is descriptive, not a universal module ranking. Interpret paired effects, "
        "uncertainty, and state cost together; no arm was selected or removed from pilot loss.\n",
        encoding="utf-8",
        newline="\n",
    )
    artifacts = [*outputs.values(), report_path]
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment_id": "53_r1_matched_diag_module_placement",
        "status": "completed_valid",
        "passed": True,
        "claim_eligible": True,
        "classification": "matched_placement_completed",
        "descriptive_lowest_mean_arm": best,
        "formal_units": 15,
        "checks": checks,
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "evidence_graph": evidence_graph,
        "artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in artifacts
        ],
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    write_json(output_dir / "analysis_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = analyze(
        args.run_dir.expanduser().resolve(),
        args.contract.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
