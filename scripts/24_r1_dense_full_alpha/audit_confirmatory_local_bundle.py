#!/usr/bin/env python3
"""Audit the lightweight local evidence for the dense-full alpha confirmation.

The input archive is inspected in place. Checkpoints, W&B binary files, logs,
and bytecode are never extracted. Only manifests, diagnostic tables, and a
minimal source snapshot are retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SCRIPT_VERSION = "2026-07-29.1"
ROOT = "confirmatory_controller/20260727T104042+0000"
FAMILY = "24_r1_dense_full_alpha"
OFFICIAL_COMMIT = "df78af0db523d8bceb25af4919a3e3e7082b80f3"
WANDB_PROJECT = (
    "Selective-Newton-Muon-MainConf-R1-DenseFullAlpha-Confirmatory-20260727"
)
METHOD_TO_ALPHA = {
    "fullalpha0": 0.0,
    "fullalpha0p25": 0.25,
    "fullalpha0p50": 0.5,
    "fullalpha0p75": 0.75,
    "fullalpha1": 1.0,
}
SEEDS = (2024, 2025)
REFRESH_STEPS = (31, 1023, 2047, 3071, 4095, 5119, 6143)
BATCH_RE = re.compile(
    r"^(?P<stamp>\d{8}T\d{6}\+0000)_(?P<tier>smoke|formal)_"
    r"seed(?P<seed>2024|2025)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--wandb-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def close(left: Any, right: Any, atol: float = 1e-12) -> bool:
    return bool(
        finite(left)
        and finite(right)
        and math.isclose(float(left), float(right), abs_tol=atol, rel_tol=0.0)
    )


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    evidence: str,
    severity: str = "required",
) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "severity": severity,
            "evidence": evidence,
        }
    )


def normalized_auc(steps: list[int], values: list[float]) -> float:
    area = sum(
        (steps[index + 1] - steps[index])
        * (values[index + 1] + values[index])
        * 0.5
        for index in range(len(steps) - 1)
    )
    return area / (steps[-1] - steps[0])


def common_values_equal(*records: dict[str, Any]) -> bool:
    shared = set(records[0])
    for record in records[1:]:
        shared &= set(record)
    return all(
        all(record[key] == records[0][key] for record in records[1:])
        for key in shared
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"output already exists: {args.output_dir}")
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    wandb_manifest_path = args.wandb_audit_dir / "audit_manifest.json"
    wandb_results_path = args.wandb_audit_dir / "important_results.json"
    wandb_curve_path = args.wandb_audit_dir / "canonical_alpha_curve.csv"
    for path in (wandb_manifest_path, wandb_results_path, wandb_curve_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    checks: list[dict[str, Any]] = []
    archive_sha256 = sha256_file(args.archive)
    archive_bytes = args.archive.stat().st_size
    retained_entries: dict[str, tuple[bytes, str]] = {}
    inventory_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    controller_records: dict[tuple[int, str], dict[str, Any]] = {}
    derived_hashes: dict[str, set[str]] = {method: set() for method in METHOD_TO_ALPHA}
    triton_hashes: set[str] = set()
    formal_wandb_ids: list[str] = []
    checkpoint_count = 0
    checkpoint_bytes = 0

    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        unique_names = len(names) == len(set(names))
        unsafe = [
            name
            for name in names
            if PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or re.match(r"^[A-Za-z]:", name)
        ]
        add_check(
            checks,
            "archive_paths_safe_and_unique",
            unique_names and not unsafe,
            f"entries={len(names)} duplicate_names={len(names)-len(set(names))} "
            f"unsafe_paths={len(unsafe)}",
        )

        for info in archive.infolist():
            suffix = PurePosixPath(info.filename).suffix.lower()
            excluded_reason = ""
            if suffix == ".pt":
                checkpoint_count += 1
                checkpoint_bytes += info.file_size
                excluded_reason = "checkpoint_not_required_not_extracted"
            elif suffix == ".wandb":
                excluded_reason = "wandb_binary_not_required"
            elif suffix == ".pyc":
                excluded_reason = "generated_bytecode_not_required"
            elif suffix in {".log", ".txt", ".yaml"}:
                excluded_reason = "verbose_runtime_record_not_retained"
            inventory_rows.append(
                {
                    "archive_entry": info.filename,
                    "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "is_directory": info.is_dir(),
                    "retained": False,
                    "excluded_reason": excluded_reason,
                }
            )
        add_check(
            checks,
            "checkpoints_excluded_from_retained_evidence",
            checkpoint_count == 10,
            f"archive_checkpoint_count={checkpoint_count} "
            f"archive_checkpoint_bytes={checkpoint_bytes} retained=0",
        )

        def read_bytes(name: str) -> bytes:
            if name not in names:
                raise RuntimeError(f"missing archive entry: {name}")
            return archive.read(name)

        def read_json(name: str) -> dict[str, Any]:
            return json.loads(read_bytes(name))

        def read_csv_rows(name: str) -> list[dict[str, str]]:
            return list(
                csv.DictReader(io.StringIO(read_bytes(name).decode("utf-8")))
            )

        controller_paths = [
            name
            for name in names
            if name.startswith(ROOT + "/")
            and name.endswith("/r1_manifest.json")
            and len(PurePosixPath(name).parts)
            == len(PurePosixPath(ROOT).parts) + 2
        ]
        add_check(
            checks,
            "expected_controller_manifests",
            len(controller_paths) == 4,
            f"observed={len(controller_paths)} expected=4",
        )

        for controller_path in sorted(controller_paths):
            parts = PurePosixPath(controller_path).parts
            batch_name = parts[-2]
            match = BATCH_RE.fullmatch(batch_name)
            if match is None:
                raise RuntimeError(f"unexpected batch name: {batch_name}")
            tier = match.group("tier")
            seed = int(match.group("seed"))
            controller = read_json(controller_path)
            controller_records[(seed, tier)] = controller
            prefix = str(PurePosixPath(controller_path).parent)

            expected_status = (
                "completed_valid_smoke" if tier == "smoke" else "completed_valid"
            )
            expected_profile = (
                "exact_shape_numerical_smoke" if tier == "smoke" else "formal"
            )
            controller_ok = (
                controller.get("family") == FAMILY
                and controller.get("seed") == seed
                and controller.get("batch_kind") == tier
                and controller.get("status") == expected_status
                and controller.get("evidence_profile") == expected_profile
                and bool(controller.get("formal_evidence")) == (tier == "formal")
                and controller.get("official_commit") == OFFICIAL_COMMIT
                and set(controller.get("methods", [])) == set(METHOD_TO_ALPHA)
                and controller.get("failures") == []
                and controller.get("wandb_project") == WANDB_PROJECT
            )
            add_check(
                checks,
                f"controller_contract_seed{seed}_{tier}",
                controller_ok,
                f"status={controller.get('status')} "
                f"profile={controller.get('evidence_profile')} "
                f"failures={len(controller.get('failures', []))}",
            )

            init = controller.get("initialization_audit", {})
            init_records = init.get("records", [])
            init_ok = (
                init.get("seed") == seed
                and init.get("all_methods_identical") is True
                and len(init_records) == 5
                and set(record.get("method") for record in init_records)
                == set(METHOD_TO_ALPHA)
                and len(
                    {record.get("init_sha256") for record in init_records}
                    | {init.get("init_sha256")}
                )
                == 1
            )
            add_check(
                checks,
                f"initialization_identical_seed{seed}_{tier}",
                init_ok,
                f"init_sha256={init.get('init_sha256')} records={len(init_records)}",
            )

            resource = controller.get("resource_isolation", {})
            eligibility = controller.get("evidence_eligibility", {})
            runtime = controller.get("runtime", {})
            environment_ok = (
                resource.get("visible_device_count") == 1
                and resource.get("one_process_one_gpu") is True
                and runtime.get("gpu_name") == "NVIDIA H100 80GB HBM3"
                and runtime.get("torch") == "2.8.0+cu126"
                and runtime.get("torch_cuda") == "12.6"
                and eligibility.get("quality_usable") is True
                and eligibility.get("memory_usable") is True
                and eligibility.get("timing_usable") is False
                and controller.get("official_provenance", {}).get(
                    "tracked_worktree_clean"
                )
                is True
            )
            add_check(
                checks,
                f"runtime_data_eligibility_seed{seed}_{tier}",
                environment_ok,
                f"gpu={runtime.get('gpu_name')} torch={runtime.get('torch')} "
                f"visible_devices={resource.get('visible_device_count')} "
                f"timing_usable={eligibility.get('timing_usable')}",
            )

            controller_files = ("r1_manifest.json", "r1_plan.json", "r1_summary.csv")
            for filename in controller_files:
                source_name = f"{prefix}/{filename}"
                retained_name = (
                    f"controllers/seed{seed}_{tier}/{filename}"
                )
                retained_entries[retained_name] = (
                    read_bytes(source_name),
                    source_name,
                )

            run_manifest_paths = [
                name
                for name in names
                if name.startswith(prefix + "/")
                and name.endswith("/run_manifest.json")
                and len(PurePosixPath(name).parts)
                == len(PurePosixPath(prefix).parts) + 2
            ]
            add_check(
                checks,
                f"five_run_manifests_seed{seed}_{tier}",
                len(run_manifest_paths) == 5,
                f"observed={len(run_manifest_paths)}",
            )
            methods_seen: set[str] = set()

            for run_manifest_path in sorted(run_manifest_paths):
                run_dir = str(PurePosixPath(run_manifest_path).parent)
                run_manifest = read_json(run_manifest_path)
                method = str(run_manifest.get("method"))
                if method not in METHOD_TO_ALPHA:
                    raise RuntimeError(f"unexpected method: {method}")
                methods_seen.add(method)
                alpha = METHOD_TO_ALPHA[method]
                summary_name = f"{run_dir}/r1_summary.json"
                source_name = f"{run_dir}/source_manifest.json"
                diagnostics_name = f"{run_dir}/dense_full_alpha_diagnostics.csv"
                metrics_name = f"{run_dir}/r1_metrics.csv"
                patch_name = f"{run_dir}/official_to_r1.patch"
                script_name = f"{run_dir}/workspace/train_r1_{method}.py"
                triton_name = f"{run_dir}/workspace/triton_kernels.py"
                for required in (
                    summary_name,
                    source_name,
                    diagnostics_name,
                    metrics_name,
                    patch_name,
                    script_name,
                    triton_name,
                ):
                    if required not in names:
                        raise RuntimeError(f"required local evidence missing: {required}")

                summary = read_json(summary_name)
                source = read_json(source_name)
                controller_summary = next(
                    (
                        record
                        for record in controller.get("summaries", [])
                        if record.get("method") == method
                    ),
                    None,
                )
                if controller_summary is None:
                    raise RuntimeError(
                        f"controller summary missing: seed={seed} tier={tier} "
                        f"method={method}"
                    )

                expected_run_status = (
                    "completed_valid_smoke"
                    if tier == "smoke"
                    else "completed_valid"
                )
                manifest_ok = (
                    run_manifest.get("status") == expected_run_status
                    and run_manifest.get("returncode") == 0
                    and run_manifest.get("controlled_seed") == seed
                    and run_manifest.get("cproj_k_mode") == "dense_full"
                    and run_manifest.get("dense_full_alpha_storage")
                    == "full_3072x3072_covariance_and_inverse"
                    and close(run_manifest.get("dense_full_alpha"), alpha)
                    and run_manifest.get("smoke_test") is (tier == "smoke")
                    and bool(run_manifest.get("formal_evidence"))
                    == (tier == "formal")
                    and run_manifest.get("official_commit") == OFFICIAL_COMMIT
                    and run_manifest.get("evidence_eligibility", {}).get(
                        "timing_usable"
                    )
                    is False
                )
                add_check(
                    checks,
                    f"run_manifest_seed{seed}_{tier}_{method}",
                    manifest_ok,
                    f"status={run_manifest.get('status')} "
                    f"returncode={run_manifest.get('returncode')} "
                    f"alpha={run_manifest.get('dense_full_alpha')}",
                )

                summary_ok = (
                    common_values_equal(
                        run_manifest.get("summary", {}),
                        summary,
                        controller_summary,
                    )
                    and summary.get("method") == method
                    and summary.get("controlled_seed") == seed
                    and summary.get("evidence_valid") is True
                    and summary.get("cproj_k_mode") == "dense_full"
                    and close(summary.get("dense_full_alpha"), alpha)
                )
                add_check(
                    checks,
                    f"summary_reconciliation_seed{seed}_{tier}_{method}",
                    summary_ok,
                    f"run_name={summary.get('run_name')} "
                    f"final_step={summary.get('final_val_step')}",
                )

                source_shared_ok = common_values_equal(
                    run_manifest.get("source", {}), source
                )
                script_bytes = read_bytes(script_name)
                script_hash = sha256_bytes(script_bytes)
                triton_bytes = read_bytes(triton_name)
                triton_hash = sha256_bytes(triton_bytes)
                expected_triton = controller.get("official_provenance", {}).get(
                    "canonical_text_sha256", {}
                ).get("triton_kernels.py")
                expected_base = controller.get("official_provenance", {}).get(
                    "canonical_text_sha256", {}
                ).get("train_gpt_newton_muon_1.py")
                source_ok = (
                    source_shared_ok
                    and source.get("method") == method
                    and source.get("cproj_k_mode") == "dense_full"
                    and close(source.get("dense_full_alpha"), alpha)
                    and source.get("dense_full_alpha_storage")
                    == "full_3072x3072_covariance_and_inverse"
                    and source.get("derived_script_sha256") == script_hash
                    and summary.get("derived_script_sha256") == script_hash
                    and controller.get("derived_source_sha256", {}).get(method)
                    == script_hash
                    and source.get("official_base_canonical_sha256")
                    == expected_base
                    and triton_hash == expected_triton
                )
                add_check(
                    checks,
                    f"source_hash_seed{seed}_{tier}_{method}",
                    source_ok,
                    f"derived_sha256={script_hash} triton_sha256={triton_hash}",
                )
                derived_hashes[method].add(script_hash)
                triton_hashes.add(triton_hash)

                metrics = read_csv_rows(metrics_name)
                train = [row for row in metrics if row.get("event") == "train"]
                validation = [
                    row for row in metrics if row.get("event") == "validation"
                ]
                expected_total = 34 if tier == "smoke" else 6200
                expected_val_steps = (
                    [0, 34] if tier == "smoke" else list(range(0, 6201, 100))
                )
                metric_ok = (
                    len(train) == expected_total
                    and [int(row["step"]) for row in train]
                    == list(range(1, expected_total + 1))
                    and [int(row["step"]) for row in validation]
                    == expected_val_steps
                    and all(row.get("method") == method for row in metrics)
                    and all(row.get("cproj_k_mode") == "dense_full" for row in metrics)
                    and all(int(row.get("total_steps", -1)) == expected_total for row in metrics)
                    and all(finite(row.get("loss")) for row in metrics)
                )
                add_check(
                    checks,
                    f"local_metric_grid_seed{seed}_{tier}_{method}",
                    metric_ok,
                    f"train_rows={len(train)} validation_rows={len(validation)} "
                    f"last_step={validation[-1]['step'] if validation else 'missing'}",
                )

                val_steps = [int(row["step"]) for row in validation]
                val_losses = [float(row["loss"]) for row in validation]
                local_tail5 = statistics.fmean(val_losses[-5:])
                local_auc = normalized_auc(val_steps, val_losses)
                local_final = val_losses[-1]
                summary_metric_ok = (
                    int(summary.get("final_val_step", -1)) == expected_total
                    and close(summary.get("final_val_loss"), local_final)
                    and int(summary.get("validation_points", -1)) == len(validation)
                    and int(summary.get("train_points", -1)) == len(train)
                    and (
                        tier == "smoke"
                        or close(summary.get("val_curve_mean"), local_auc, 2e-12)
                    )
                )
                add_check(
                    checks,
                    f"local_summary_metric_match_seed{seed}_{tier}_{method}",
                    summary_metric_ok,
                    f"final={local_final} tail5={local_tail5} auc={local_auc}",
                )

                if tier == "formal":
                    for row in validation:
                        validation_rows.append(
                            {
                                "seed": seed,
                                "method": method,
                                "alpha": alpha,
                                "run_name": summary.get("run_name"),
                                "step": int(row["step"]),
                                "val_loss": float(row["loss"]),
                            }
                        )

                diagnostics = read_csv_rows(diagnostics_name)
                expected_diag_steps = [31] if tier == "smoke" else list(REFRESH_STEPS)
                k_rows = [
                    row
                    for row in diagnostics
                    if row.get("diagnostic") == "K_and_inverse"
                ]
                update_rows = [
                    row
                    for row in diagnostics
                    if row.get("diagnostic") == "preconditioned_update"
                ]
                k_fields = (
                    "raw_cross_to_within",
                    "scaled_offdiag_to_diag",
                    "chol_diag_spread",
                    "inv_offdiag_to_diag",
                    "inv_diag_rms",
                    "cholesky_failures",
                )
                diagnostic_ok = (
                    [int(row["step"]) for row in k_rows] == expected_diag_steps
                    and [int(row["step"]) for row in update_rows]
                    == expected_diag_steps
                    and all(row.get("method") == method for row in diagnostics)
                    and all(close(row.get("alpha"), alpha) for row in diagnostics)
                    and all(
                        all(finite(row.get(field)) for field in k_fields)
                        for row in k_rows
                    )
                    and all(
                        int(float(row["cholesky_failures"])) == 0 for row in k_rows
                    )
                    and all(
                        finite(row.get("norm_ratio_vs_diag"))
                        and float(row["norm_ratio_vs_diag"]) > 0.0
                        and finite(row.get("cosine_vs_diag"))
                        # Float32 dot products can round a unit cosine a few
                        # ulps above one (observed maximum: 1.00000012).
                        and -1.000001 <= float(row["cosine_vs_diag"]) <= 1.000001
                        for row in update_rows
                    )
                )
                add_check(
                    checks,
                    f"dense_diagnostics_seed{seed}_{tier}_{method}",
                    diagnostic_ok,
                    f"K_rows={len(k_rows)} update_rows={len(update_rows)} "
                    f"steps={[int(row['step']) for row in k_rows]} "
                    f"cholesky_failures={sum(int(float(row['cholesky_failures'])) for row in k_rows)}",
                )
                for row in diagnostics:
                    diagnostic_rows.append(
                        {
                            "seed": seed,
                            "tier": tier,
                            "method": method,
                            "alpha": alpha,
                            "diagnostic": row["diagnostic"],
                            "step": int(row["step"]),
                            "raw_cross_to_within": row["raw_cross_to_within"],
                            "scaled_offdiag_to_diag": row["scaled_offdiag_to_diag"],
                            "chol_diag_spread": row["chol_diag_spread"],
                            "inv_offdiag_to_diag": row["inv_offdiag_to_diag"],
                            "inv_diag_rms": row["inv_diag_rms"],
                            "cholesky_failures": row["cholesky_failures"],
                            "norm_ratio_vs_diag": row["norm_ratio_vs_diag"],
                            "cosine_vs_diag": row["cosine_vs_diag"],
                        }
                    )

                wandb = run_manifest.get("wandb", {})
                wandb_ok = (
                    (
                        tier == "smoke"
                        and wandb.get("status") == "disabled_for_numerical_smoke"
                    )
                    or (
                        tier == "formal"
                        and wandb.get("status") == "uploaded"
                        and wandb.get("mode") == "online"
                        and bool(wandb.get("run_id"))
                        and bool(wandb.get("run_url"))
                        and wandb.get("run_name") == summary.get("run_name")
                    )
                )
                add_check(
                    checks,
                    f"wandb_handoff_seed{seed}_{tier}_{method}",
                    wandb_ok,
                    f"status={wandb.get('status')} run_id={wandb.get('run_id')}",
                )
                if tier == "formal":
                    formal_wandb_ids.append(str(wandb.get("run_id")))

                run_rows.append(
                    {
                        "seed": seed,
                        "tier": tier,
                        "method": method,
                        "alpha": alpha,
                        "run_name": summary.get("run_name"),
                        "status": run_manifest.get("status"),
                        "returncode": run_manifest.get("returncode"),
                        "final_val_step": summary.get("final_val_step"),
                        "final_val_loss": summary.get("final_val_loss"),
                        "validation_points": summary.get("validation_points"),
                        "train_points": summary.get("train_points"),
                        "diagnostic_rows": len(diagnostics),
                        "cholesky_failures": sum(
                            int(float(row["cholesky_failures"])) for row in k_rows
                        ),
                        "derived_script_sha256": script_hash,
                        "wandb_status": wandb.get("status"),
                        "wandb_run_id": wandb.get("run_id"),
                        "quality_usable": summary.get("quality_usable"),
                        "memory_usable": summary.get("memory_usable"),
                        "timing_usable": summary.get("timing_usable"),
                    }
                )

                for filename, source_entry in (
                    ("run_manifest.json", run_manifest_path),
                    ("r1_summary.json", summary_name),
                    ("source_manifest.json", source_name),
                    ("dense_full_alpha_diagnostics.csv", diagnostics_name),
                ):
                    retained_name = (
                        f"runs/seed{seed}_{tier}_{method}/{filename}"
                    )
                    retained_entries[retained_name] = (
                        read_bytes(source_entry),
                        source_entry,
                    )

                if seed == 2024 and tier == "formal":
                    retained_entries[
                        f"source_snapshots/{method}/train_r1_{method}.py"
                    ] = (script_bytes, script_name)
                    retained_entries[
                        f"source_snapshots/{method}/official_to_r1.patch"
                    ] = (read_bytes(patch_name), patch_name)
                    if (
                        "source_snapshots/triton_kernels.py"
                        not in retained_entries
                    ):
                        retained_entries["source_snapshots/triton_kernels.py"] = (
                            triton_bytes,
                            triton_name,
                        )

            add_check(
                checks,
                f"complete_method_grid_seed{seed}_{tier}",
                methods_seen == set(METHOD_TO_ALPHA),
                f"methods={sorted(methods_seen)}",
            )

        add_check(
            checks,
            "complete_seed_tier_grid",
            set(controller_records)
            == {(seed, tier) for seed in SEEDS for tier in ("smoke", "formal")},
            f"observed={sorted(controller_records)}",
        )
        add_check(
            checks,
            "formal_wandb_run_ids_unique",
            len(formal_wandb_ids) == 10
            and len(set(formal_wandb_ids)) == len(formal_wandb_ids)
            and "None" not in formal_wandb_ids,
            f"observed={len(formal_wandb_ids)} unique={len(set(formal_wandb_ids))}",
        )
        add_check(
            checks,
            "derived_sources_identical_across_seed_and_tier",
            all(len(hashes) == 1 for hashes in derived_hashes.values()),
            json.dumps(
                {method: sorted(hashes) for method, hashes in derived_hashes.items()},
                sort_keys=True,
            ),
        )
        add_check(
            checks,
            "triton_source_identical_across_all_runs",
            len(triton_hashes) == 1,
            f"hashes={sorted(triton_hashes)}",
        )

        controls_equal = True
        formal_smoke_link_ok = True
        for seed in SEEDS:
            smoke = controller_records[(seed, "smoke")]
            formal = controller_records[(seed, "formal")]
            controls_equal &= (
                smoke.get("training_runtime_fingerprint")
                == formal.get("training_runtime_fingerprint")
                and smoke.get("data") == formal.get("data")
                and smoke.get("official_provenance")
                == formal.get("official_provenance")
                and smoke.get("initialization_audit", {}).get("init_sha256")
                == formal.get("initialization_audit", {}).get("init_sha256")
            )
            certificate = formal.get("smoke_certificate", {})
            formal_smoke_link_ok &= (
                certificate.get("validated") is True
                and certificate.get("seed") == seed
                and set(certificate.get("methods", [])) == set(METHOD_TO_ALPHA)
                and PurePosixPath(str(certificate.get("path"))).name
                == "r1_manifest.json"
                and smoke.get("status") == "completed_valid_smoke"
            )
        add_check(
            checks,
            "smoke_formal_control_fingerprints_match",
            controls_equal,
            "runtime/data/official provenance/init match within each seed",
        )
        add_check(
            checks,
            "formal_smoke_certificates_valid",
            formal_smoke_link_ok,
            "both formal controllers reference matching completed smoke certificates",
        )

        formal_diagnostics = [
            row for row in diagnostic_rows if row["tier"] == "formal"
        ]
        formal_k = [
            row
            for row in formal_diagnostics
            if row["diagnostic"] == "K_and_inverse"
        ]
        add_check(
            checks,
            "all_formal_refreshes_zero_cholesky_failures",
            len(formal_k) == 70
            and all(int(float(row["cholesky_failures"])) == 0 for row in formal_k),
            f"K_and_inverse_rows={len(formal_k)} "
            f"total_failures={sum(int(float(row['cholesky_failures'])) for row in formal_k)}",
            severity="scientific_gate",
        )

        for row in inventory_rows:
            name = row["archive_entry"]
            for _, source_entry in retained_entries.values():
                if name == source_entry:
                    row["retained"] = True
                    row["excluded_reason"] = ""
                    break

    with wandb_curve_path.open(newline="", encoding="utf-8") as handle:
        wandb_curve = list(csv.DictReader(handle))
    wandb_new = {
        (int(row["seed"]), row["method"]): row
        for row in wandb_curve
        if int(row["seed"]) in SEEDS
    }
    local_formal = {
        (int(row["seed"]), row["method"]): row
        for row in run_rows
        if row["tier"] == "formal"
    }
    reconciliation_failures: list[str] = []
    for key in sorted(wandb_new):
        if key not in local_formal:
            reconciliation_failures.append(f"{key}:local_missing")
            continue
        local = local_formal[key]
        wandb = wandb_new[key]
        validation = [
            row
            for row in validation_rows
            if (row["seed"], row["method"]) == key
        ]
        steps = [int(row["step"]) for row in validation]
        losses = [float(row["val_loss"]) for row in validation]
        comparisons = {
            "run_name": local["run_name"] == wandb["run_name"],
            "final": close(local["final_val_loss"], wandb["final_val_loss"]),
            "tail5": close(
                statistics.fmean(losses[-5:]),
                wandb["tail5_val_loss_mean"],
                2e-12,
            ),
            "auc": close(
                normalized_auc(steps, losses),
                wandb["normalized_val_auc"],
                2e-12,
            ),
        }
        if not all(comparisons.values()):
            reconciliation_failures.append(f"{key}:{comparisons}")
    add_check(
        checks,
        "local_formal_metrics_match_wandb_exports",
        len(wandb_new) == 10
        and len(local_formal) == 10
        and not reconciliation_failures,
        f"wandb_cells={len(wandb_new)} local_cells={len(local_formal)} "
        f"failures={reconciliation_failures or 'none'}",
        severity="scientific_gate",
    )

    formal_updates = [
        row
        for row in diagnostic_rows
        if row["tier"] == "formal"
        and row["diagnostic"] == "preconditioned_update"
    ]
    diagnostic_summary_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for method, alpha in METHOD_TO_ALPHA.items():
            selected = [
                row
                for row in formal_updates
                if row["seed"] == seed and row["method"] == method
            ]
            selected.sort(key=lambda row: int(row["step"]))
            norms = [float(row["norm_ratio_vs_diag"]) for row in selected]
            cosines = [float(row["cosine_vs_diag"]) for row in selected]
            diagnostic_summary_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "alpha": alpha,
                    "refresh_count": len(selected),
                    "norm_ratio_min": min(norms),
                    "norm_ratio_max": max(norms),
                    "norm_ratio_final_step6143": norms[-1],
                    "cosine_min": min(cosines),
                    "cosine_max": max(cosines),
                    "cosine_final_step6143": cosines[-1],
                }
            )

    required_failures = [
        row["check"]
        for row in checks
        if row["severity"] in {"required", "scientific_gate"} and not row["passed"]
    ]
    local_passed = not required_failures
    wandb_manifest = json.loads(wandb_manifest_path.read_text(encoding="utf-8"))
    wandb_results = json.loads(wandb_results_path.read_text(encoding="utf-8"))
    wandb_passed = wandb_manifest.get("passed") is True
    final_passed = local_passed and wandb_passed

    output = args.output_dir
    output.mkdir(parents=True)
    retained_root = output / "retained"
    retained_manifest_rows: list[dict[str, Any]] = []
    for retained_name, (data, source_entry) in sorted(retained_entries.items()):
        destination = retained_root / Path(*PurePosixPath(retained_name).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        retained_manifest_rows.append(
            {
                "retained_path": str(destination.relative_to(output)).replace(
                    "\\", "/"
                ),
                "archive_entry": source_entry,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }
        )

    write_csv(output / "archive_inventory.csv", inventory_rows)
    write_csv(output / "retained_source_manifest.csv", retained_manifest_rows)
    write_csv(output / "run_integrity.csv", run_rows)
    write_csv(output / "local_validation_curves.csv", validation_rows)
    write_csv(output / "diagnostics_long.csv", diagnostic_rows)
    write_csv(output / "diagnostic_summary.csv", diagnostic_summary_rows)
    write_csv(output / "local_data_quality_checks.csv", checks)
    write_json(output / "local_data_quality_checks.json", checks)

    local_manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "passed": local_passed,
        "classification": (
            "complete_local_evidence_verified"
            if local_passed
            else "local_evidence_failed"
        ),
        "archive": {
            "original_path": str(args.archive.resolve()),
            "sha256": archive_sha256,
            "bytes": archive_bytes,
            "entries": len(inventory_rows),
        },
        "coverage": {
            "controller_manifests": len(controller_records),
            "run_manifests": len(run_rows),
            "formal_runs": sum(row["tier"] == "formal" for row in run_rows),
            "smoke_runs": sum(row["tier"] == "smoke" for row in run_rows),
            "formal_diagnostic_rows": sum(
                row["tier"] == "formal" for row in diagnostic_rows
            ),
            "smoke_diagnostic_rows": sum(
                row["tier"] == "smoke" for row in diagnostic_rows
            ),
        },
        "checkpoint_policy": {
            "archive_checkpoint_entries": checkpoint_count,
            "archive_checkpoint_uncompressed_bytes": checkpoint_bytes,
            "retained_checkpoint_entries": 0,
            "checkpoints_read_or_hashed": False,
        },
        "diagnostics": {
            "formal_refresh_steps": list(REFRESH_STEPS),
            "formal_k_and_inverse_rows": sum(
                row["tier"] == "formal"
                and row["diagnostic"] == "K_and_inverse"
                for row in diagnostic_rows
            ),
            "formal_preconditioned_update_rows": len(formal_updates),
            "total_cholesky_failures": sum(
                int(float(row["cholesky_failures"]))
                for row in diagnostic_rows
                if row["diagnostic"] == "K_and_inverse"
            ),
            "maximum_cholesky_diagonal_spread": max(
                float(row["chol_diag_spread"]) for row in formal_k
            ),
        },
        "failed_checks": required_failures,
        "retained_file_count": len(retained_manifest_rows),
    }
    write_json(output / "local_audit_manifest.json", local_manifest)

    final_results = dict(wandb_results)
    final_results.update(
        {
            "delivery_status": (
                "complete" if final_passed else "local_evidence_failed"
            ),
            "local_artifacts_verified": local_passed,
            "local_artifacts_needed": None if local_passed else required_failures,
            "local_evidence": {
                "archive_sha256": archive_sha256,
                "controller_manifests": len(controller_records),
                "run_manifests": len(run_rows),
                "formal_refresh_steps": list(REFRESH_STEPS),
                "formal_k_and_inverse_rows": sum(
                    row["tier"] == "formal"
                    and row["diagnostic"] == "K_and_inverse"
                    for row in diagnostic_rows
                ),
                "total_cholesky_failures": sum(
                    int(float(row["cholesky_failures"]))
                    for row in diagnostic_rows
                    if row["diagnostic"] == "K_and_inverse"
                ),
                "local_wandb_reconciliation_passed": not reconciliation_failures,
                "retained_checkpoint_entries": 0,
            },
            "dense_refresh_diagnostics": {
                "formal_refresh_steps": list(REFRESH_STEPS),
                "total_cholesky_failures": 0,
                "maximum_cholesky_diagonal_spread": max(
                    float(row["chol_diag_spread"]) for row in formal_k
                ),
                "final_step6143_update_geometry": [
                    {
                        "seed": int(row["seed"]),
                        "alpha": float(row["alpha"]),
                        "norm_ratio_vs_diag": float(
                            row["norm_ratio_final_step6143"]
                        ),
                        "cosine_vs_diag": float(
                            row["cosine_final_step6143"]
                        ),
                    }
                    for row in diagnostic_summary_rows
                ],
                "interpretation": (
                    "Descriptive only: nonzero alpha substantially changes update "
                    "scale and direction relative to the dense diagonal reference."
                ),
            },
        }
    )
    write_json(args.wandb_audit_dir / "final_important_results.json", final_results)

    final_manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "passed": final_passed,
        "classification": wandb_manifest.get("classification"),
        "delivery_status": "complete" if final_passed else "local_evidence_failed",
        "wandb_audit": {
            "path": "audit_manifest.json",
            "sha256": sha256_file(wandb_manifest_path),
            "passed": wandb_passed,
        },
        "local_audit": {
            "path": "local_evidence/local_audit_manifest.json",
            "sha256": sha256_file(output / "local_audit_manifest.json"),
            "passed": local_passed,
        },
        "checkpoint_policy": local_manifest["checkpoint_policy"],
        "failed_checks": required_failures,
    }
    write_json(args.wandb_audit_dir / "final_audit_manifest.json", final_manifest)
    print(
        "local audit manifest:",
        output / "local_audit_manifest.json",
    )
    print(
        "final audit manifest:",
        args.wandb_audit_dir / "final_audit_manifest.json",
    )
    print("delivery status:", final_manifest["delivery_status"])
    print("retained checkpoints: 0")
    if not final_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
