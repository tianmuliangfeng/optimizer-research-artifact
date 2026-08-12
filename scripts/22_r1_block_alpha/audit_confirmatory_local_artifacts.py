#!/usr/bin/env python3
"""Audit the R1 block-alpha confirmatory controller handoff ZIP."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from io import BytesIO, StringIO
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any
import zipfile


SCRIPT_VERSION = "2026-07-29.3"
FAMILY = "22_r1_block_alpha"
CONTROLLER_PROTOCOL = "r1_block_alpha_seeds2024_2025_confirmatory_v1"
FORMAL_PROTOCOL = (
    "official_newton_muon_1_r1_dense_block_alpha_"
    "seeds2024_2025_confirmatory_v1"
)
SMOKE_PROTOCOL = (
    "official_newton_muon_1_r1_dense_block_alpha_exact_shape_smoke_v1"
)
OFFICIAL_COMMIT = "df78af0db523d8bceb25af4919a3e3e7082b80f3"
OFFICIAL_BASE_SHA256 = (
    "48383e333334e4f29bbae3365ac4142226c27750ede5739ab53c0dafbbcb7730"
)
TRITON_SHA256 = (
    "b51ac50c699b05306619d92cb9ec6edadd266d8118c53f5b9726db76480ea16d"
)
EXPECTED_SEEDS = (2024, 2025)
EXPECTED_METHODS = ("alpha0", "alpha0p25", "alpha0p50", "alpha0p75")
METHOD_ALPHA = {
    "alpha0": 0.0,
    "alpha0p25": 0.25,
    "alpha0p50": 0.5,
    "alpha0p75": 0.75,
}
BATCH_RE = re.compile(
    r"^(?P<stamp>\d{8}T\d{6}\+0000)_(?P<kind>smoke|formal)_"
    r"seed(?P<seed>2024|2025)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--wandb-analysis-dir", required=True, type=Path)
    parser.add_argument("--preserved-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-transfer-policy",
        choices=("required", "excluded_by_user_due_size"),
        default="required",
    )
    parser.add_argument(
        "--partial-checkpoints-removed",
        action="store_true",
        help=(
            "Record that unusable partial checkpoint entries were removed "
            "from the preserved extracted tree after their hashes were saved."
        ),
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def normalized_auc(points: list[tuple[int, float]]) -> float:
    ordered = sorted(points)
    return sum(
        (right[0] - left[0]) * (left[1] + right[1]) / 2.0
        for left, right in zip(ordered, ordered[1:])
    ) / (ordered[-1][0] - ordered[0][0])


def close(left: float, right: float, *, atol: float = 1e-12) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= atol


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and not re.match(r"^[A-Za-z]:", name)
        and not name.startswith("\\")
    )


def load_csv_path(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    archive = args.archive.resolve()
    wandb_dir = args.wandb_analysis_dir.resolve()
    preserved_root = args.preserved_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")

    checks: list[dict[str, Any]] = []

    def check(
        name: str,
        passed: bool,
        detail: str,
        *,
        severity: str = "required",
    ) -> None:
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "severity": severity,
                "detail": detail,
            }
        )

    curve_rows = load_csv_path(wandb_dir / "canonical_alpha_curve.csv")
    canonical = {
        (int(row["seed"]), float(row["alpha"])): row
        for row in curve_rows
    }
    history_rows = load_csv_path(
        wandb_dir / "confirmatory_history_long.csv"
    )
    wandb_validation: dict[tuple[int, str], list[tuple[int, float]]] = {}
    for row in history_rows:
        if row["metric"] != "val/loss":
            continue
        key = (int(row["seed"]), row["method"])
        wandb_validation.setdefault(key, []).append(
            (int(row["step"]), float(row["value"]))
        )
    for series in wandb_validation.values():
        series.sort()

    archive_sha256 = sha256_file(archive)
    source_inventory: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    formal_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []

    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        files = [entry for entry in entries if not entry.is_dir()]
        names = [entry.filename for entry in entries]
        file_names = [entry.filename for entry in files]
        top_levels = sorted(
            {PurePosixPath(name).parts[0] for name in names if name}
        )
        expected_top = archive.stem

        check("archive_crc", bundle.testzip() is None, "ZipFile.testzip")
        check(
            "archive_member_paths_safe",
            all(safe_member(name) for name in names),
            f"entries={len(entries)}",
        )
        check(
            "archive_member_names_unique",
            len(names) == len(set(names)),
            f"entries={len(entries)}",
        )
        check(
            "archive_single_expected_root",
            top_levels == [expected_top],
            f"top_levels={top_levels}",
        )
        check(
            "archive_file_count",
            len(files) == 345,
            f"observed={len(files)} expected=345",
        )
        check(
            "archive_uncompressed_bytes",
            sum(entry.file_size for entry in files) == 130532323,
            (
                f"observed={sum(entry.file_size for entry in files)} "
                "expected=130532323"
            ),
        )

        member_cache: dict[str, bytes] = {}

        def read_member(name: str) -> bytes:
            if name not in member_cache:
                member_cache[name] = bundle.read(name)
            return member_cache[name]

        def read_json(name: str) -> dict[str, Any]:
            return json.loads(read_member(name).decode("utf-8"))

        def read_csv(name: str) -> list[dict[str, str]]:
            text = read_member(name).decode("utf-8")
            return list(csv.DictReader(StringIO(text)))

        controller_name = f"{expected_top}/confirmatory_controller.json"
        check(
            "controller_present",
            controller_name in file_names,
            controller_name,
        )
        controller = read_json(controller_name)
        check(
            "controller_protocol",
            controller.get("protocol") == CONTROLLER_PROTOCOL,
            str(controller.get("protocol")),
        )
        check(
            "controller_batch_id",
            controller.get("batch_id") == expected_top,
            str(controller.get("batch_id")),
        )
        check(
            "controller_completed",
            controller.get("status") == "completed",
            str(controller.get("status")),
        )
        check(
            "controller_no_failures",
            controller.get("failures") == [],
            repr(controller.get("failures")),
        )
        check(
            "controller_seed_grid",
            controller.get("seeds") == list(EXPECTED_SEEDS),
            repr(controller.get("seeds")),
        )
        seed_status = controller.get("seed_status", {})
        for seed in EXPECTED_SEEDS:
            value = seed_status.get(str(seed), {})
            check(
                f"controller_seed{seed}_completed",
                value.get("status") == "completed",
                str(value.get("status")),
            )
            check(
                f"controller_seed{seed}_wandb_complete",
                value.get("wandb_complete") is True,
                repr(value.get("wandb_complete")),
            )

        batch_manifest_names = [
            name
            for name in file_names
            if name.endswith("/r1_manifest.json")
            and len(PurePosixPath(name).parts) == 3
        ]
        discovered_batches: dict[tuple[str, int], tuple[str, str]] = {}
        for name in batch_manifest_names:
            batch_dir = PurePosixPath(name).parts[1]
            match = BATCH_RE.fullmatch(batch_dir)
            if match:
                key = (match.group("kind"), int(match.group("seed")))
                discovered_batches[key] = (batch_dir, name)
        check(
            "batch_grid",
            set(discovered_batches)
            == {
                (kind, seed)
                for kind in ("smoke", "formal")
                for seed in EXPECTED_SEEDS
            },
            repr(sorted(discovered_batches)),
        )

        expected_init: dict[int, str] = {}
        derived_hashes: dict[str, str] = {}
        formal_wandb_names: set[str] = set()
        checkpoint_names: set[str] = set()

        for kind in ("smoke", "formal"):
            for seed in EXPECTED_SEEDS:
                batch_dir, manifest_name = discovered_batches[(kind, seed)]
                manifest = read_json(manifest_name)
                prefix = f"{expected_top}/{batch_dir}/"
                summaries = manifest.get("summaries", [])
                methods = tuple(manifest.get("methods", []))
                expected_status = (
                    "completed_valid_smoke"
                    if kind == "smoke"
                    else "completed_valid"
                )
                expected_protocol = (
                    SMOKE_PROTOCOL if kind == "smoke" else FORMAL_PROTOCOL
                )
                check(
                    f"{kind}_seed{seed}:family",
                    manifest.get("family") == FAMILY,
                    str(manifest.get("family")),
                )
                check(
                    f"{kind}_seed{seed}:kind_seed",
                    manifest.get("batch_kind") == kind
                    and manifest.get("seed") == seed,
                    (
                        f"kind={manifest.get('batch_kind')} "
                        f"seed={manifest.get('seed')}"
                    ),
                )
                check(
                    f"{kind}_seed{seed}:protocol",
                    manifest.get("protocol") == expected_protocol,
                    str(manifest.get("protocol")),
                )
                check(
                    f"{kind}_seed{seed}:status",
                    manifest.get("status") == expected_status,
                    str(manifest.get("status")),
                )
                check(
                    f"{kind}_seed{seed}:methods",
                    methods == EXPECTED_METHODS,
                    repr(methods),
                )
                check(
                    f"{kind}_seed{seed}:no_failures",
                    manifest.get("failures") == [],
                    repr(manifest.get("failures")),
                )
                check(
                    f"{kind}_seed{seed}:official_commit",
                    manifest.get("official_commit") == OFFICIAL_COMMIT,
                    str(manifest.get("official_commit")),
                )
                init_audit = manifest.get("initialization_audit", {})
                init_sha = str(init_audit.get("init_sha256", ""))
                check(
                    f"{kind}_seed{seed}:initialization_identical",
                    init_audit.get("all_methods_identical") is True
                    and manifest.get(
                        "formal_initialization_fingerprints_identical"
                    )
                    is True,
                    f"init_sha256={init_sha}",
                )
                if seed in expected_init:
                    check(
                        f"{kind}_seed{seed}:init_matches_other_tier",
                        expected_init[seed] == init_sha,
                        (
                            f"observed={init_sha} "
                            f"expected={expected_init[seed]}"
                        ),
                    )
                else:
                    expected_init[seed] = init_sha
                check(
                    f"{kind}_seed{seed}:summary_grid",
                    len(summaries) == 4
                    and {row.get("method") for row in summaries}
                    == set(EXPECTED_METHODS),
                    f"rows={len(summaries)}",
                )
                eligibility = manifest.get("evidence_eligibility", {})
                check(
                    f"{kind}_seed{seed}:eligibility",
                    eligibility.get("quality_usable") is True
                    and eligibility.get("memory_usable") is True
                    and eligibility.get("timing_usable") is False,
                    repr(eligibility),
                )
                isolation = manifest.get("resource_isolation", {})
                check(
                    f"{kind}_seed{seed}:resource_isolation",
                    isolation.get("one_process_one_gpu") is True
                    and isolation.get("concurrent_node_training") is True,
                    repr(isolation),
                )
                wandb_statuses = manifest.get("wandb_statuses", {})
                expected_wandb = (
                    "disabled_for_numerical_smoke"
                    if kind == "smoke"
                    else "uploaded"
                )
                check(
                    f"{kind}_seed{seed}:wandb_status_grid",
                    manifest.get("wandb_complete") is True
                    and set(wandb_statuses) == set(EXPECTED_METHODS)
                    and set(wandb_statuses.values()) == {expected_wandb},
                    repr(wandb_statuses),
                )
                if kind == "formal":
                    certificate = manifest.get("smoke_certificate", {})
                    check(
                        f"formal_seed{seed}:smoke_certificate",
                        certificate.get("validated") is True
                        and certificate.get("seed") == seed
                        and tuple(certificate.get("methods", []))
                        == EXPECTED_METHODS,
                        repr(certificate),
                    )

                run_manifest_names = [
                    name
                    for name in file_names
                    if name.startswith(prefix)
                    and name.endswith("/run_manifest.json")
                    and len(PurePosixPath(name).parts) == 4
                ]
                check(
                    f"{kind}_seed{seed}:run_manifest_count",
                    len(run_manifest_names) == 4,
                    f"observed={len(run_manifest_names)}",
                )
                run_methods: set[str] = set()

                for run_manifest_name in run_manifest_names:
                    run_dir = PurePosixPath(run_manifest_name).parts[2]
                    run_prefix = (
                        f"{expected_top}/{batch_dir}/{run_dir}/"
                    )
                    run_manifest = read_json(run_manifest_name)
                    method = str(run_manifest.get("method"))
                    run_methods.add(method)
                    run_label = f"{kind}_seed{seed}_{method}"
                    expected_formal = kind == "formal"
                    check(
                        f"{run_label}:identity",
                        method in EXPECTED_METHODS
                        and run_manifest.get("controlled_seed") == seed
                        and run_manifest.get("experiment_family") == FAMILY,
                        (
                            f"method={method} "
                            f"seed={run_manifest.get('controlled_seed')}"
                        ),
                    )
                    check(
                        f"{run_label}:status",
                        run_manifest.get("status")
                        == (
                            "completed_valid"
                            if expected_formal
                            else "completed_valid_smoke"
                        )
                        and run_manifest.get("returncode") == 0,
                        (
                            f"status={run_manifest.get('status')} "
                            f"returncode={run_manifest.get('returncode')}"
                        ),
                    )
                    check(
                        f"{run_label}:tier",
                        run_manifest.get("formal_evidence")
                        is expected_formal
                        and run_manifest.get("evidence_profile")
                        == ("formal" if expected_formal else "exact_shape_numerical_smoke"),
                        (
                            f"formal={run_manifest.get('formal_evidence')} "
                            f"profile={run_manifest.get('evidence_profile')}"
                        ),
                    )
                    run_eligibility = run_manifest.get(
                        "evidence_eligibility", {}
                    )
                    check(
                        f"{run_label}:eligibility",
                        run_eligibility.get("quality_usable") is True
                        and run_eligibility.get("memory_usable") is True
                        and run_eligibility.get("timing_usable") is False,
                        repr(run_eligibility),
                    )
                    summary_name = run_prefix + "r1_summary.json"
                    source_name = run_prefix + "source_manifest.json"
                    metrics_name = run_prefix + "r1_metrics.csv"
                    log_name = run_prefix + "training_log_with_source.txt"
                    derived_name = (
                        run_prefix
                        + f"workspace/train_r1_{method}.py"
                    )
                    triton_name = run_prefix + "workspace/triton_kernels.py"
                    required = {
                        summary_name,
                        source_name,
                        metrics_name,
                        log_name,
                        derived_name,
                        triton_name,
                    }
                    check(
                        f"{run_label}:required_files",
                        required.issubset(file_names),
                        repr(sorted(required - set(file_names))),
                    )
                    summary = read_json(summary_name)
                    source = read_json(source_name)
                    metrics = read_csv(metrics_name)
                    derived_sha = sha256_bytes(read_member(derived_name))
                    triton_sha = sha256_bytes(read_member(triton_name))
                    check(
                        f"{run_label}:derived_source_hash",
                        derived_sha
                        == source.get("derived_script_sha256")
                        == summary.get("derived_script_sha256")
                        == run_manifest.get("source", {}).get(
                            "derived_script_sha256"
                        ),
                        (
                            f"actual={derived_sha} "
                            f"declared={source.get('derived_script_sha256')}"
                        ),
                    )
                    check(
                        f"{run_label}:official_base_hash",
                        source.get("official_base_canonical_sha256")
                        == OFFICIAL_BASE_SHA256,
                        str(
                            source.get("official_base_canonical_sha256")
                        ),
                    )
                    check(
                        f"{run_label}:triton_hash",
                        triton_sha == TRITON_SHA256,
                        triton_sha,
                    )
                    expected_alpha = METHOD_ALPHA[method]
                    check(
                        f"{run_label}:alpha_storage",
                        float(summary.get("block_alpha"))
                        == expected_alpha
                        and summary.get("block_alpha_storage")
                        == "dense_official_block4"
                        and float(source.get("block_alpha"))
                        == expected_alpha,
                        (
                            f"summary_alpha={summary.get('block_alpha')} "
                            f"source_alpha={source.get('block_alpha')}"
                        ),
                    )
                    batch_summary = next(
                        row
                        for row in summaries
                        if row.get("method") == method
                    )
                    common_summary_keys = (
                        "method",
                        "controlled_seed",
                        "init_sha256",
                        "final_val_step",
                        "final_val_loss",
                        "final_train_step",
                        "final_train_loss",
                        "derived_script_sha256",
                    )
                    check(
                        f"{run_label}:summary_consistency",
                        all(
                            summary.get(key) == batch_summary.get(key)
                            for key in common_summary_keys
                        )
                        and all(
                            summary.get(key)
                            == run_manifest.get("summary", {}).get(key)
                            for key in common_summary_keys
                        ),
                        repr(common_summary_keys),
                    )
                    expected_total = 6200 if expected_formal else 34
                    validation = [
                        row
                        for row in metrics
                        if row["event"] == "validation"
                    ]
                    train = [
                        row for row in metrics if row["event"] == "train"
                    ]
                    expected_val_steps = (
                        list(range(0, 6201, 100))
                        if expected_formal
                        else [0, 34]
                    )
                    check(
                        f"{run_label}:metrics_shape",
                        len(train) == expected_total
                        and [int(row["step"]) for row in train]
                        == list(range(1, expected_total + 1))
                        and [
                            int(row["step"]) for row in validation
                        ]
                        == expected_val_steps,
                        (
                            f"rows={len(metrics)} train={len(train)} "
                            f"validation={len(validation)}"
                        ),
                    )
                    check(
                        f"{run_label}:metrics_identity",
                        all(
                            row["method"] == method
                            and row["cproj_k_mode"] == "block4"
                            and int(row["total_steps"]) == expected_total
                            and math.isfinite(float(row["loss"]))
                            for row in metrics
                        ),
                        f"rows={len(metrics)}",
                    )
                    final_val = float(validation[-1]["loss"])
                    check(
                        f"{run_label}:metrics_summary_endpoint",
                        int(validation[-1]["step"])
                        == int(summary["final_val_step"])
                        and close(
                            final_val,
                            float(summary["final_val_loss"]),
                        ),
                        (
                            f"metrics={final_val} "
                            f"summary={summary['final_val_loss']}"
                        ),
                    )
                    if expected_formal:
                        points = [
                            (int(row["step"]), float(row["loss"]))
                            for row in validation
                        ]
                        tail5 = sum(value for _, value in points[-5:]) / 5
                        auc = normalized_auc(points)
                        canonical_row = canonical[(seed, expected_alpha)]
                        local_series = points
                        exported_series = wandb_validation[(seed, method)]
                        check(
                            f"{run_label}:wandb_validation_curve",
                            local_series == exported_series,
                            (
                                f"local_points={len(local_series)} "
                                f"wandb_points={len(exported_series)}"
                            ),
                        )
                        check(
                            f"{run_label}:wandb_summary_metrics",
                            close(
                                final_val,
                                float(
                                    canonical_row["final_val_loss"]
                                ),
                            )
                            and close(
                                tail5,
                                float(
                                    canonical_row[
                                        "tail5_val_loss_mean"
                                    ]
                                ),
                            )
                            and close(
                                auc,
                                float(
                                    canonical_row[
                                        "normalized_val_auc"
                                    ]
                                ),
                            ),
                            (
                                f"final={final_val};tail5={tail5};"
                                f"auc={auc}"
                            ),
                        )
                        wandb = run_manifest.get("wandb", {})
                        run_name = str(run_manifest.get("run_name"))
                        formal_wandb_names.add(run_name)
                        check(
                            f"{run_label}:wandb_identity",
                            wandb.get("status") == "uploaded"
                            and wandb.get("mode") == "online"
                            and wandb.get("run_name") == run_name
                            and bool(wandb.get("run_id"))
                            and str(wandb.get("run_url", "")).endswith(
                                str(wandb.get("run_id"))
                            ),
                            repr(wandb),
                        )
                        checkpoint_candidates = [
                            name
                            for name in file_names
                            if name.startswith(
                                run_prefix + "workspace/logs/"
                            )
                            and name.endswith(
                                "/state_step006200.pt"
                            )
                        ]
                        check(
                            f"{run_label}:checkpoint_entry_count",
                            len(checkpoint_candidates) == 1,
                            repr(checkpoint_candidates),
                            severity="handoff",
                        )
                        if checkpoint_candidates:
                            checkpoint_name = checkpoint_candidates[0]
                            checkpoint_names.add(checkpoint_name)
                            checkpoint_bytes = read_member(
                                checkpoint_name
                            )
                            try:
                                with zipfile.ZipFile(
                                    BytesIO(checkpoint_bytes)
                                ) as checkpoint_bundle:
                                    checkpoint_container_valid = (
                                        checkpoint_bundle.testzip() is None
                                    )
                            except zipfile.BadZipFile:
                                checkpoint_container_valid = False
                            checkpoint_rows.append(
                                {
                                    "seed": seed,
                                    "method": method,
                                    "member": checkpoint_name,
                                    "transferred_bytes": len(
                                        checkpoint_bytes
                                    ),
                                    "declared_checkpoint_bytes": int(
                                        summary["checkpoint_bytes"]
                                    ),
                                    "transfer_fraction": len(
                                        checkpoint_bytes
                                    )
                                    / int(summary["checkpoint_bytes"]),
                                    "sha256": sha256_bytes(
                                        checkpoint_bytes
                                    ),
                                    "pytorch_zip_container_valid": (
                                        checkpoint_container_valid
                                    ),
                                    "transfer_policy": (
                                        args.checkpoint_transfer_policy
                                    ),
                                    "usable_locally": False,
                                }
                            )
                            if (
                                args.checkpoint_transfer_policy
                                == "required"
                            ):
                                check(
                                    f"{run_label}:checkpoint_valid",
                                    checkpoint_container_valid,
                                    (
                                        f"bytes={len(checkpoint_bytes)} "
                                        f"declared={summary['checkpoint_bytes']}"
                                    ),
                                    severity="handoff",
                                )
                            else:
                                check(
                                    f"{run_label}:checkpoint_exclusion_recorded",
                                    not checkpoint_container_valid
                                    and len(checkpoint_bytes)
                                    < int(summary["checkpoint_bytes"]),
                                    (
                                        "user-confirmed transfer exclusion; "
                                        f"bytes={len(checkpoint_bytes)} "
                                        f"declared={summary['checkpoint_bytes']}"
                                    ),
                                    severity="documented_exclusion",
                                )
                        formal_rows.append(
                            {
                                "seed": seed,
                                "method": method,
                                "alpha": expected_alpha,
                                "run_name": run_name,
                                "run_id": wandb.get("run_id"),
                                "final_val_loss": final_val,
                                "tail5_val_loss_mean": tail5,
                                "normalized_val_auc": auc,
                                "initial_val_loss": float(
                                    validation[0]["loss"]
                                ),
                                "init_sha256": summary["init_sha256"],
                                "derived_script_sha256": derived_sha,
                                "metrics_rows": len(metrics),
                                "checkpoint_transferred": False,
                                "quality_usable": True,
                                "memory_usable": True,
                                "timing_usable": False,
                            }
                        )

                check(
                    f"{kind}_seed{seed}:run_method_grid",
                    run_methods == set(EXPECTED_METHODS),
                    repr(sorted(run_methods)),
                )
                batch_rows.append(
                    {
                        "kind": kind,
                        "seed": seed,
                        "batch_dir": batch_dir,
                        "status": manifest.get("status"),
                        "methods": ",".join(methods),
                        "init_sha256": init_sha,
                        "official_commit": manifest.get(
                            "official_commit"
                        ),
                        "wandb_complete": manifest.get(
                            "wandb_complete"
                        ),
                        "quality_usable": eligibility.get(
                            "quality_usable"
                        ),
                        "memory_usable": eligibility.get(
                            "memory_usable"
                        ),
                        "timing_usable": eligibility.get(
                            "timing_usable"
                        ),
                    }
                )

        exported_run_names = {
            row["run_name"]
            for row in curve_rows
            if int(row["seed"]) in EXPECTED_SEEDS
            and float(row["alpha"]) < 1.0
        }
        check(
            "formal_wandb_run_identity_grid",
            formal_wandb_names == exported_run_names,
            (
                f"local={sorted(formal_wandb_names)};"
                f"exports={sorted(exported_run_names)}"
            ),
        )
        check(
            "checkpoint_entry_grid",
            len(checkpoint_names) == 8,
            f"observed={len(checkpoint_names)}",
            severity="handoff",
        )

        for entry in files:
            value = read_member(entry.filename)
            source_inventory.append(
                {
                    "member": entry.filename,
                    "bytes": entry.file_size,
                    "compressed_bytes": entry.compress_size,
                    "crc32": f"{entry.CRC:08x}",
                    "sha256": sha256_bytes(value),
                    "checkpoint_transfer_excluded": (
                        entry.filename in checkpoint_names
                        and args.checkpoint_transfer_policy
                        == "excluded_by_user_due_size"
                    ),
                }
            )

    required_failures = [
        row
        for row in checks
        if not row["passed"]
        and row["severity"] in {"required", "handoff"}
    ]
    documented_exclusion_failures = [
        row
        for row in checks
        if not row["passed"]
        and row["severity"] == "documented_exclusion"
    ]
    checkpoint_excluded = (
        args.checkpoint_transfer_policy == "excluded_by_user_due_size"
    )
    accepted = not required_failures and not documented_exclusion_failures
    delivery_status = (
        "accepted_checkpoint_transfer_excluded"
        if accepted and checkpoint_excluded
        else "accepted"
        if accepted
        else "needs_revision"
    )

    output.mkdir(parents=True, exist_ok=False)
    with (output / "local_artifact_checks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=checks[0].keys())
        writer.writeheader()
        writer.writerows(checks)
    for name, rows in (
        ("local_batch_summary.csv", batch_rows),
        ("local_formal_run_summary.csv", formal_rows),
        ("local_checkpoint_inventory.csv", checkpoint_rows),
        ("local_source_inventory.csv", source_inventory),
    ):
        with (output / name).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    check_counts = Counter(
        "passed" if row["passed"] else "failed" for row in checks
    )
    write_json(
        output / "local_artifact_audit.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "archive": {
                "path": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": archive_sha256,
                "entries": len(source_inventory),
            },
            "preserved_root": str(preserved_root),
            "controller_batch_id": archive.stem,
            "formal_runs": len(formal_rows),
            "smoke_runs": 8,
            "checks": len(checks),
            "passed_checks": check_counts["passed"],
            "failed_checks": [
                row["check"] for row in checks if not row["passed"]
            ],
            "checkpoint_transfer": {
                "policy": args.checkpoint_transfer_policy,
                "expected_remote_checkpoints": 8,
                "usable_local_checkpoints": 0,
                "partial_entries_removed_from_preserved_tree": (
                    args.partial_checkpoints_removed
                ),
                "scientific_result_blocked": False,
            },
            "scientific_evidence_accepted": accepted,
            "delivery_status": delivery_status,
        },
    )
    write_json(
        output / "local_artifact_audit_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "passed": accepted,
            "delivery_status": delivery_status,
            "checkpoint_transfer_policy": (
                args.checkpoint_transfer_policy
            ),
            "artifacts": [
                "local_artifact_audit.json",
                "local_artifact_checks.csv",
                "local_batch_summary.csv",
                "local_checkpoint_inventory.csv",
                "local_formal_run_summary.csv",
                "local_source_inventory.csv",
            ],
        },
    )
    print(f"Local artifact audit: {output}")
    print(f"Checks: {check_counts['passed']}/{len(checks)} passed")
    print(f"Delivery status: {delivery_status}")
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
