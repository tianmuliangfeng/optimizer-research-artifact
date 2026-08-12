#!/usr/bin/env python3
"""Read-only audit of submission efficiency and sensitivity evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-29.6"
CANONICAL_METHOD = {
    "muon": "muon",
    "block4": "original_newton_muon",
    "newton_full": "original_newton_muon",
    "none": "selective_none",
    "down_none": "selective_none",
    "diag": "selective_diag",
    "down_diag": "selective_diag",
}
METHOD_ORDER = {
    "muon": 1,
    "original_newton_muon": 2,
    "selective_none": 3,
    "selective_diag": 4,
}
REQUIRED_ROLES = set(METHOD_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--portable-root", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only and args.output_dir is None:
        parser.error("--output-dir is required unless --preflight-only is used")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    for row in rows:
        if set(row) != set(fields):
            raise RuntimeError(f"inconsistent schema for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def role(method: str) -> str:
    if method not in CANONICAL_METHOD:
        raise RuntimeError(f"unregistered method: {method}")
    return CANONICAL_METHOD[method]


def resolve_registered_source(
    input_root: Path,
    spec: dict[str, Any],
    portable_root: Path | None = None,
) -> tuple[Path, str]:
    """Resolve one registered source without binding the audit to one archive layout.

    The registry's canonical ``path`` remains authoritative.  ``candidate_paths``
    and ``fallback_globs`` only support relocations of the same immutable artifact
    inside its experiment-number directory.  A fallback must resolve uniquely;
    ambiguity is an audit failure rather than an invitation to pick a convenient
    file.
    """

    canonical = input_root / spec["path"]
    if canonical.is_file():
        return canonical, spec["path"]

    candidates: list[Path] = []
    for relative in spec.get("candidate_paths", ()):
        path = input_root / relative
        if path.is_file():
            candidates.append(path)
    for pattern in spec.get("fallback_globs", ()):
        candidates.extend(path for path in input_root.glob(pattern) if path.is_file())

    unique: dict[str, Path] = {}
    root_resolved = input_root.resolve()
    for path in candidates:
        resolved = path.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise RuntimeError(
                f"registered source escaped input root: {resolved}"
            ) from exc
        unique[str(resolved)] = resolved

    if not unique and portable_root is not None and spec.get("portable_path"):
        portable = portable_root / spec["portable_path"]
        if portable.is_file():
            return portable.resolve(), f"portable_snapshot/{spec['portable_path']}"

    if not unique:
        attempted = [spec["path"], *spec.get("candidate_paths", ())]
        globs = list(spec.get("fallback_globs", ()))
        raise FileNotFoundError(
            f"{canonical}; no relocation candidate found "
            f"(paths={attempted}, globs={globs})"
        )
    if len(unique) != 1:
        matches = sorted(unique)
        raise RuntimeError(
            f"ambiguous relocated source {spec['id']}: {matches}"
        )
    path = next(iter(unique.values()))
    return path, path.relative_to(root_resolved).as_posix()


def audit_required_sources(
    input_root: Path,
    registry: dict[str, Any],
    portable_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    rows: list[dict[str, Any]] = []
    resolved: dict[str, Path] = {}
    failures: list[str] = []
    for spec in registry["required_files"]:
        path = input_root / spec["path"]
        selected_relative = spec["path"]
        exists = False
        error = ""
        row_count: Any = ""
        columns_or_keys = ""
        digest = ""
        try:
            if registry.get("portable_snapshot_required") is True:
                if portable_root is None:
                    raise RuntimeError("portable snapshot root is required")
                portable_name = spec.get("portable_path")
                portable_digest = spec.get("portable_sha256")
                if not portable_name or not portable_digest:
                    raise RuntimeError(
                        "portable snapshot path/hash is missing from registry"
                    )
                portable = portable_root / portable_name
                if not portable.is_file():
                    raise FileNotFoundError(
                        f"portable snapshot file is missing: {portable}"
                    )
                observed_portable_digest = sha256_file(portable)
                if observed_portable_digest != portable_digest:
                    raise RuntimeError(
                        "portable snapshot hash mismatch: "
                        f"{portable_name}: {observed_portable_digest} "
                        f"!= {portable_digest}"
                    )
            path, selected_relative = resolve_registered_source(
                input_root, spec, portable_root
            )
            resolved[spec["id"]] = path
            exists = True
            digest = sha256_file(path)
            if spec["kind"] == "csv":
                values = read_csv(path)
                if not values:
                    raise RuntimeError("empty CSV")
                row_count = len(values)
                columns = set(values[0])
                missing = set(spec.get("required_columns", ())) - columns
                if missing:
                    raise RuntimeError(f"missing columns: {sorted(missing)}")
                columns_or_keys = ",".join(sorted(columns))
            elif spec["kind"] == "json":
                value = read_json(path)
                if not isinstance(value, dict):
                    raise RuntimeError("JSON root must be an object")
                columns_or_keys = ",".join(sorted(value))
            else:
                raise RuntimeError(f"unsupported kind: {spec['kind']}")
        except Exception as exc:
            error = repr(exc)
            failures.append(f"{spec['id']}: {error}")
        rows.append(
            {
                "source_id": spec["id"],
                "relative_path": selected_relative,
                "required": True,
                "exists": exists,
                "kind": spec["kind"],
                "size_bytes": path.stat().st_size if exists else "",
                "sha256": digest,
                "row_count": row_count,
                "columns_or_keys": columns_or_keys,
                "error": error,
            }
        )
    if failures:
        raise RuntimeError("source audit failed:\n" + "\n".join(failures))
    return rows, resolved


def performance_bundle_errors(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    summary: list[dict[str, str]],
    runs: list[dict[str, str]],
    preflight: dict[str, Any],
    postflight: dict[str, Any],
    provenance: dict[str, Any],
    registry: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    methods = list(registry["performance_required_roles"])
    method_set = set(methods)
    repeats = int(manifest.get("repeats", 0))
    timed_steps = int(manifest.get("timed_steps", 0))
    expected_final_step = 32 + timed_steps
    if manifest.get("status") != "complete":
        errors.append("manifest status")
    if manifest.get("runtime", {}).get("gpu_name") != "NVIDIA H100 80GB HBM3":
        errors.append("runtime GPU")
    if set(manifest.get("methods", ())) != method_set:
        errors.append("manifest method coverage")
    if timed_steps != int(registry["performance_min_timed_steps"]):
        errors.append("timed-step budget")
    if repeats < int(registry["performance_min_repeats"]):
        errors.append("repeat count")
    for label, certificate in (
        ("preflight", preflight),
        ("postflight", postflight),
    ):
        if (
            certificate.get("passed") is not True
            or certificate.get("active_compute_processes")
            or len(certificate.get("gpus", ()))
            < int(registry["performance_required_gpu_count"])
            or not {0, 1}.issubset(
                {int(gpu["index"]) for gpu in certificate.get("gpus", ())}
            )
        ):
            errors.append(f"{label} exclusivity")
    try:
        pre_time = datetime.fromisoformat(preflight["created_at"])
        post_time = datetime.fromisoformat(postflight["created_at"])
        manifest_time = datetime.fromtimestamp(
            manifest_path.stat().st_mtime, timezone.utc
        )
        if not pre_time < manifest_time < post_time:
            errors.append("certificate chronology")
    except (KeyError, TypeError, ValueError, OSError):
        errors.append("certificate timestamps")
    repo = provenance.get("provenance", {})
    if (
        provenance.get("passed") is not True
        or repo.get("official_commit") != registry["official_commit"]
        or repo.get("tracked_worktree_clean") is not True
        or not repo.get("canonical_text_sha256")
    ):
        errors.append("official-repository provenance")
    if len(runs) != len(methods) * repeats:
        errors.append("raw run count")
    init_hashes = {row.get("init_sha256", "") for row in runs}
    if len(init_hashes) != 1 or "" in init_hashes:
        errors.append("initialization hash")
    source_hashes = manifest.get("source_sha256", {})
    batch = manifest_path.parent
    for method in methods:
        method_rows = [row for row in runs if row.get("method") == method]
        if len(method_rows) != repeats:
            errors.append(f"{method} repeat coverage")
            continue
        positions = {int(row["position"]) for row in method_rows}
        if positions != set(range(1, len(methods) + 1)):
            errors.append(f"{method} rotation coverage")
        for row in method_rows:
            try:
                values = (
                    float(row["official_train_time_s"]),
                    float(row["official_step_avg_ms"]),
                    float(row["wrapper_wall_elapsed_s"]),
                )
                if (
                    int(row["seed"]) != 2026
                    or int(row["final_step"]) != expected_final_step
                    or not all(math.isfinite(value) and value > 0 for value in values)
                    or row["source_sha256"] != source_hashes[method]
                ):
                    errors.append(f"{method} raw run contract")
                log = batch / (
                    f"repeat{int(row['repeat']):02d}_"
                    f"{int(row['position']):02d}_{method}/terminal.log"
                )
                if (
                    not log.is_file()
                    or sha256_file(log) != row["log_sha256"]
                ):
                    errors.append(f"{method} log hash")
            except (KeyError, TypeError, ValueError, OSError):
                errors.append(f"{method} malformed raw run")
    if {row.get("method") for row in summary} != method_set:
        errors.append("summary method coverage")
    else:
        for row in summary:
            method = row["method"]
            method_rows = [item for item in runs if item.get("method") == method]
            try:
                times = [
                    float(item["official_train_time_s"]) for item in method_rows
                ]
                step_ms = [
                    float(item["official_step_avg_ms"]) for item in method_rows
                ]
                expected_tokens = timed_steps * 512 * 1024 / statistics.median(times)
                if (
                    int(row["runs"]) != repeats
                    or not math.isclose(
                        float(row["median_train_time_s"]),
                        statistics.median(times),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        float(row["median_step_ms"]),
                        statistics.median(step_ms),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        float(row["median_tokens_per_s"]),
                        expected_tokens,
                        rel_tol=1e-12,
                        abs_tol=1e-9,
                    )
                ):
                    errors.append(f"{method} summary recomputation")
            except (KeyError, TypeError, ValueError, statistics.StatisticsError):
                errors.append(f"{method} malformed summary")
    return sorted(set(errors))


def discover_performance(
    input_root: Path, registry: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    records: list[dict[str, Any]] = []
    roots = [
        input_root / relative
        for relative in registry["optional_performance_roots"]
    ]
    existing_roots = [root for root in roots if root.is_dir()]
    if not existing_roots:
        records.append(
            {
                "source_id": "isolated_r1_performance",
                "relative_path": ",".join(registry["optional_performance_roots"]),
                "required": False,
                "exists": False,
                "kind": "directory",
                "size_bytes": "",
                "sha256": "",
                "row_count": "",
                "columns_or_keys": "",
                "error": "paper-grade performance output root is absent",
            }
        )
        return records, None

    candidates: list[dict[str, Any]] = []
    for root in existing_roots:
        for manifest_path in sorted(root.rglob("perf_manifest.json")):
            try:
                manifest = read_json(manifest_path)
                if manifest.get("protocol") != "r1_perf_training_benchmark_v1":
                    continue
                batch = manifest_path.parent
                summary_path = batch / "training_benchmark_summary.csv"
                runs_path = batch / "training_benchmark_runs.csv"
                if not summary_path.is_file() or not runs_path.is_file():
                    continue
                summary = read_csv(summary_path)
                runs = read_csv(runs_path)
                certificate_path = batch.parent / "exclusive_node_preflight.json"
                postflight_path = batch.parent / "exclusive_node_postflight.json"
                provenance_path = batch.parent / "official_repo_provenance.json"
                certificate = (
                    read_json(certificate_path) if certificate_path.is_file() else {}
                )
                postflight = (
                    read_json(postflight_path) if postflight_path.is_file() else {}
                )
                provenance = (
                    read_json(provenance_path) if provenance_path.is_file() else {}
                )
                errors = performance_bundle_errors(
                    manifest_path=manifest_path,
                    manifest=manifest,
                    summary=summary,
                    runs=runs,
                    preflight=certificate,
                    postflight=postflight,
                    provenance=provenance,
                    registry=registry,
                )
                candidates.append(
                    {
                        "manifest_path": manifest_path,
                        "manifest": manifest,
                        "summary_path": summary_path,
                        "runs_path": runs_path,
                        "certificate_path": certificate_path,
                        "postflight_path": postflight_path,
                        "provenance_path": provenance_path,
                        "summary": summary,
                        "runs": runs,
                        "eligible": not errors,
                        "errors": errors,
                    }
                )
            except Exception:
                continue
    selected = next((item for item in reversed(candidates) if item["eligible"]), None)
    for root in existing_roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name in {
                "perf_manifest.json",
                "training_benchmark_summary.csv",
                "training_benchmark_runs.csv",
                "exclusive_node_preflight.json",
                "exclusive_node_postflight.json",
                "official_repo_provenance.json",
            }:
                records.append(
                    {
                        "source_id": "isolated_r1_performance",
                        "relative_path": path.relative_to(input_root).as_posix(),
                        "required": False,
                        "exists": True,
                        "kind": "json" if path.suffix == ".json" else "csv",
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "row_count": len(read_csv(path)) if path.suffix == ".csv" else "",
                        "columns_or_keys": "",
                        "error": "",
                    }
                )
    if not records:
        records.append(
            {
                "source_id": "isolated_r1_performance",
                "relative_path": ",".join(registry["optional_performance_roots"]),
                "required": False,
                "exists": True,
                "kind": "directory",
                "size_bytes": "",
                "sha256": "",
                "row_count": "",
                "columns_or_keys": "",
                "error": "no complete training benchmark triplet found",
            }
        )
    return records, selected


def discover_sensitivity(
    input_root: Path, registry: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    relative = registry["optional_sensitivity_root"]
    root = input_root / relative
    if not root.is_dir():
        return [
            {
                "source_id": "r1_four_method_lr_sensitivity",
                "relative_path": relative,
                "required": False,
                "exists": False,
                "kind": "directory",
                "size_bytes": "",
                "sha256": "",
                "row_count": "",
                "columns_or_keys": "",
                "error": "accepted four-method sensitivity output is absent",
            }
        ], None
    accepted = []
    records = []
    for path in sorted(root.rglob("lr_sensitivity_manifest.json")):
        error = ""
        value: dict[str, Any] | None = None
        try:
            value = read_json(path)
            if (
                value.get("passed") is not True
                or value.get("evidence_class") != "supporting_only"
                or value.get("diag_vs_none_primary") is not False
                or set(value.get("methods", ())) != REQUIRED_ROLES
                or set(value.get("multipliers", ())) != {0.8, 1.0, 1.2}
                or int(value.get("run_cells", 0)) != 12
            ):
                raise RuntimeError("sensitivity manifest acceptance failed")
            for name, digest in value["output_sha256"].items():
                if sha256_file(path.parent / name) != digest:
                    raise RuntimeError(f"output hash mismatch: {name}")
            accepted.append({"manifest_path": path, "manifest": value})
        except Exception as exc:
            error = repr(exc)
        records.append(
            {
                "source_id": "r1_four_method_lr_sensitivity",
                "relative_path": path.relative_to(input_root).as_posix(),
                "required": False,
                "exists": True,
                "kind": "json",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "row_count": "",
                "columns_or_keys": ",".join(sorted(value)) if value else "",
                "error": error,
            }
        )
    if not records:
        records.append(
            {
                "source_id": "r1_four_method_lr_sensitivity",
                "relative_path": relative,
                "required": False,
                "exists": True,
                "kind": "directory",
                "size_bytes": "",
                "sha256": "",
                "row_count": "",
                "columns_or_keys": "",
                "error": "no LR sensitivity analysis manifest found",
            }
        )
    return records, accepted[-1] if accepted else None


def validate_foundations(paths: dict[str, Path]) -> dict[str, Any]:
    unified = read_json(paths["unified_mechanism_manifest"])
    if not unified.get("passed") or unified.get("diag_vs_none_primary"):
        raise RuntimeError("unified mechanism acceptance failed")
    exact = read_json(paths["capacity_exact_manifest"])
    fine = read_json(paths["capacity_fine_manifest"])
    fine_source = read_json(paths["capacity_fine_source_manifest"])
    runtime_plan = read_json(paths["capacity_runtime_plan"])
    cross = read_json(paths["r1_lr_cross_manifest"])
    llama = read_json(paths["llama1b_formal_manifest"])
    checks = {
        "unified_mechanism_passed": bool(unified.get("passed")),
        "diag_vs_none_not_primary": not bool(unified.get("diag_vs_none_primary")),
        "exact_capacity_complete": exact.get("status") == "complete",
        "exact_capacity_quality_pass": exact.get("quality_checks", {}).get("failed") == 0,
        "fine_capacity_complete": fine.get("status") == "complete",
        "fine_capacity_source_complete": fine_source.get("status") == "completed",
        "capacity_pre_cell_free_fraction": (
            float(fine_source["plan"]["minimum_pre_cell_free_gpu_fraction"]) >= 0.98
        ),
        "capacity_runtime_h100": runtime_plan["runtime"]["gpu_name"]
        == "NVIDIA H100 80GB HBM3",
        "capacity_sequence_length_1024": int(fine_source["plan"]["sequence_length"])
        == 1024,
        "lr_cross_timing_ineligible": cross.get("timing_usable") is False,
        "llama1b_timing_ineligible": llama.get("timing_usable") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"foundation checks failed: {checks}")
    return checks


def build_capacity(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        canonical = role(row["method"])
        width = int(row["resolved_width"])
        if width != 1:
            raise RuntimeError(f"capacity boundary unresolved: {row}")
        output.append(
            {
                "method_role": canonical,
                "source_method": row["method"],
                "method_order": METHOD_ORDER[canonical],
                "max_success_device_batch": int(row["max_success_device_batch"]),
                "first_oom_device_batch": int(row["first_oom_device_batch"]),
                "max_success_global_batch": int(row["max_success_global_batch"]),
                "first_oom_global_batch": int(row["first_oom_global_batch"]),
                "resolved_width": width,
                "gain_vs_original_device_batch": int(
                    row["gain_vs_newton_full_device_batch"]
                ),
                "gain_vs_original_pct": float(row["gain_vs_newton_full_pct"]),
                "evidence_class": "paper_ready",
            }
        )
    if {row["method_role"] for row in output} != REQUIRED_ROLES:
        raise RuntimeError("capacity method coverage failed")
    return sorted(output, key=lambda row: row["method_order"])


def build_memory(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        canonical = role(row["method"])
        output.append(
            {
                "method_role": canonical,
                "source_method": row["method"],
                "method_order": METHOD_ORDER[canonical],
                "device_batch_size": 32,
                "model_parameter_mib": float(row["model_parameter_mib"]),
                "optimizer_state_mib": float(row["optimizer_state_mib"]),
                "k_state_mib": float(row["k_state_mib"]),
                "preconditioner_workspace_mib": float(
                    row["preconditioner_workspace_mib"]
                ),
                "peak_allocated_mib": float(
                    row["peak_allocated_mib_at_batch32"]
                ),
                "peak_reserved_mib": float(
                    row["peak_reserved_mib_at_batch32"]
                ),
                "reserved_headroom_mib": float(
                    row["reserved_headroom_mib_at_batch32"]
                ),
                "evidence_class": "paper_ready",
            }
        )
    if {row["method_role"] for row in output} != REQUIRED_ROLES:
        raise RuntimeError("memory method coverage failed")
    return sorted(output, key=lambda row: row["method_order"])


def build_sensitivity(
    paths: dict[str, Path],
    completed_sensitivity: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    b24 = read_csv(paths["historical_b24_lr_grid"])
    cross = read_csv(paths["r1_lr_cross_summary"])
    alpha = read_csv(paths["unified_alpha"])
    b24_roles = {role(row["method"]) for row in b24 if row["method"] != "full"}
    b24_lrs = sorted({float(row["matrix_lr"]) for row in b24})
    cross_roles = {role(row["method"]) for row in cross}
    cross_levels = sorted({row["lr_level"] for row in cross})
    return [
        {
            "sensitivity_type": "learning_rate",
            "source": "B24 reference LR grid",
            "recipe_match": "no",
            "architecture": "OpenWebText GPT-2 24L/1024",
            "method_roles": ",".join(sorted(b24_roles)),
            "grid": ",".join(f"{value:g}" for value in b24_lrs),
            "seeds": ",".join(sorted({row["seed"] for row in b24})),
            "budget": "3000 train steps; final validation at 2500",
            "classification": "historical_only",
            "limitation": "single seed and non-final architecture/LR scale",
        },
        {
            "sensitivity_type": "learning_rate",
            "source": "R1 Muon-diag LR cross",
            "recipe_match": "partial",
            "architecture": "final R1",
            "method_roles": ",".join(sorted(cross_roles)),
            "grid": ",".join(cross_levels),
            "seeds": ",".join(sorted({row["seed"] for row in cross})),
            "budget": "6200 steps",
            "classification": "supporting_only",
            "limitation": "covers Muon and Selective-diag only; no four-method shared grid",
        },
        {
            "sensitivity_type": "alpha_response",
            "source": "accepted block and dense-full alpha confirmations",
            "recipe_match": "yes",
            "architecture": "final R1",
            "method_roles": "original_newton_muon,selective_diag",
            "grid": "0,0.25,0.5,0.75,1",
            "seeds": ",".join(
                sorted(
                    {
                        seed
                        for row in alpha
                        for seed in str(row.get("seed_list", "2024,2025,2026")).split(",")
                    }
                )
            ),
            "budget": "6200 steps",
            "classification": "paper_ready",
            "limitation": "mechanism alpha response; not a substitute for LR tuning",
        },
        {
            "sensitivity_type": "learning_rate",
            "source": (
                str(completed_sensitivity["manifest_path"])
                if completed_sensitivity
                else "required final-recipe shared multiplier grid"
            ),
            "recipe_match": "yes" if completed_sensitivity else "required",
            "architecture": "final R1",
            "method_roles": ",".join(sorted(REQUIRED_ROLES)),
            "grid": "0.8x,1.0x,1.2x",
            "seeds": "2026",
            "budget": "3000 equal-budget steps",
            "classification": "supporting_only" if completed_sensitivity else "missing",
            "limitation": (
                "single-seed supporting robustness; no tuned-best claim"
                if completed_sensitivity
                else "needed for four-method tuning robustness; no tuned-best claim"
            ),
        },
    ]


def build_metric_eligibility(
    performance: dict[str, Any] | None,
    completed_sensitivity: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    throughput_class = "paper_ready" if performance else "missing"
    perf_source = (
        str(performance["manifest_path"])
        if performance
        else "18_r1_performance/results (absent)"
    )
    perf_reason = (
        "complete same-runtime benchmark with >=512 timed steps and 4 balanced rotated repeats"
        if performance
        else "no accepted isolated benchmark outputs; quality-run timing is not comparable"
    )
    return [
        {
            "metric": "tokens_per_s",
            "classification": throughput_class,
            "source": perf_source,
            "paper_use": "main efficiency table" if performance else "blocked",
            "reason": perf_reason,
            "followup_id": "" if performance else "EFF-ISO-R1",
        },
        {
            "metric": "steps_per_s",
            "classification": throughput_class,
            "source": perf_source,
            "paper_use": "main efficiency table" if performance else "blocked",
            "reason": perf_reason,
            "followup_id": "" if performance else "EFF-ISO-R1",
        },
        {
            "metric": "peak_allocated_mib",
            "classification": "paper_ready",
            "source": "LLaMA-1B capacity fine, device batch 32",
            "paper_use": "main efficiency table",
            "reason": "same H100/runtime/config with explicit CUDA peak and >=98% pre-cell free GPU",
            "followup_id": "",
        },
        {
            "metric": "peak_reserved_mib",
            "classification": "paper_ready",
            "source": "LLaMA-1B capacity fine, device batch 32",
            "paper_use": "main efficiency table",
            "reason": "same H100/runtime/config with explicit CUDA reserved peak",
            "followup_id": "",
        },
        {
            "metric": "optimizer_and_k_state_mib",
            "classification": "paper_ready",
            "source": "LLaMA-1B capacity fine, device batch 32",
            "paper_use": "main efficiency table",
            "reason": "exact state byte accounting for all four method roles",
            "followup_id": "",
        },
        {
            "metric": "maximum_device_batch",
            "classification": "paper_ready",
            "source": "LLaMA-1B exact capacity confirmation",
            "paper_use": "main capacity table",
            "reason": "success/OOM boundary resolved to adjacent integer batches for all roles",
            "followup_id": "",
        },
        {
            "metric": "quality_run_wall_clock",
            "classification": "historical_only",
            "source": "formal R1/LLaMA training logs",
            "paper_use": "do not use for efficiency claims",
            "reason": "descriptive timing lacks isolated repeated same-order protocol",
            "followup_id": "",
        },
        {
            "metric": "four_method_lr_sensitivity",
            "classification": "supporting_only" if completed_sensitivity else "missing",
            "source": (
                str(completed_sensitivity["manifest_path"])
                if completed_sensitivity
                else "historical B24 grid plus partial R1 LR cross"
            ),
            "paper_use": (
                "robustness appendix"
                if completed_sensitivity
                else "robustness appendix blocked"
            ),
            "reason": (
                "final-recipe 0.8x/1.0x/1.2x grid covers all four roles at equal budget"
                if completed_sensitivity
                else "no final-recipe shared multiplier grid across all four roles"
            ),
            "followup_id": "" if completed_sensitivity else "SENS-R1-4WAY",
        },
        {
            "metric": "alpha_response_sensitivity",
            "classification": "paper_ready",
            "source": "block and dense-full alpha confirmation",
            "paper_use": "mechanism appendix",
            "reason": "three seeds and fixed five-point alpha grid; not LR tuning evidence",
            "followup_id": "",
        },
    ]


def performance_rows(performance: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not performance:
        return [
            {
                "method_role": canonical,
                "median_step_ms": "",
                "steps_per_s": "",
                "median_tokens_per_s": "",
                "runs": 0,
                "classification": "missing",
            }
            for canonical in sorted(REQUIRED_ROLES, key=METHOD_ORDER.get)
        ]
    output = []
    for row in performance["summary"]:
        if row["method"] not in CANONICAL_METHOD:
            continue
        canonical = role(row["method"])
        if canonical not in REQUIRED_ROLES:
            continue
        step_ms = float(row["median_step_ms"])
        output.append(
            {
                "method_role": canonical,
                "median_step_ms": step_ms,
                "steps_per_s": 1000.0 / step_ms,
                "median_tokens_per_s": float(row["median_tokens_per_s"]),
                "runs": int(row["runs"]),
                "classification": "paper_ready",
            }
        )
    if {row["method_role"] for row in output} != REQUIRED_ROLES:
        raise RuntimeError("performance role coverage failed after eligibility")
    return sorted(output, key=lambda row: METHOD_ORDER[row["method_role"]])


def followup_contract(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = {row["followup_id"] for row in metric_rows if row["followup_id"]}
    experiments = []
    if "EFF-ISO-R1" in gaps:
        experiments.append(
            {
                "id": "EFF-ISO-R1",
                "priority": 1,
                "purpose": "paper-grade tokens/s and steps/s for four primary roles",
                "reuse_code": "scripts/18_r1_performance/run_r1_performance.py",
                "methods": ["diag", "block4", "none", "muon"],
                "requirements": {
                    "exclusive_node": True,
                    "warmup_steps": 32,
                    "timed_steps": 512,
                    "rotated_repeats": 4,
                    "wandb_in_timed_process": False,
                },
                "estimated_new_training_runs": 12,
            }
        )
    if "SENS-R1-4WAY" in gaps:
        experiments.append(
            {
                "id": "SENS-R1-4WAY",
                "priority": 2,
                "purpose": "final-recipe four-method LR robustness, not tuned-best selection",
                "methods": [
                    "diag",
                    "none",
                    "block4",
                    "muon",
                ],
                "recipe_lr_multipliers": [0.8, 1.0, 1.2],
                "seed": 2026,
                "requirements": {
                    "equal_tokens": True,
                    "equal_validation_schedule": True,
                    "same_data_and_initialization": True,
                    "selection_rule": "report curves and endpoint/AUC robustness; no tuned-best claim",
                },
                "classification_if_completed": "supporting_only",
                "estimated_cells": 12,
            }
        )
    return {
        "schema_version": 1,
        "contract_version": "2026-07-29.2",
        "submission_ready_before_followup": not experiments,
        "experiments": experiments,
        "explicit_non_requirements": [
            "do not rerun LLaMA-1B memory or capacity",
            "do not rerun alpha",
            "do not promote diag-versus-none to a primary comparison",
            "do not use MECH-08 or quality-run timing for efficiency",
        ],
    }


def build_gap_matrix(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = {
        "EFF-ISO-R1": (
            "isolated throughput",
            "main efficiency table lacks paper-grade tokens/s and steps/s",
            "run existing R1-PERF smoke plus 512-step x 4 balanced benchmark",
        ),
        "SENS-R1-4WAY": (
            "final-recipe LR sensitivity",
            "no equal-budget four-role shared multiplier grid",
            "run 0.8x/1.0x/1.2x on one frozen seed; report as supporting only",
        ),
    }
    observed = {row["followup_id"] for row in metric_rows if row["followup_id"]}
    return [
        {
            "gap_id": gap_id,
            "priority": index,
            "gap": mapping[gap_id][0],
            "why_blocking": mapping[gap_id][1],
            "minimal_resolution": mapping[gap_id][2],
            "status": "missing",
        }
        for index, gap_id in enumerate(
            [item for item in ("EFF-ISO-R1", "SENS-R1-4WAY") if item in observed],
            start=1,
        )
    ] or [
        {
            "gap_id": "NONE",
            "priority": 0,
            "gap": "none",
            "why_blocking": "all frozen metrics are eligible",
            "minimal_resolution": "no new remote experiment",
            "status": "complete",
        }
    ]


def build_report(
    metric_rows: list[dict[str, Any]],
    capacity: list[dict[str, Any]],
    memory: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> str:
    by_role = {row["method_role"]: row for row in memory}
    original = by_role["original_newton_muon"]
    none = by_role["selective_none"]
    diag = by_role["selective_diag"]
    missing = [row for row in metric_rows if row["classification"] == "missing"]
    cap = {row["method_role"]: row for row in capacity}
    lines = [
        "# Submission efficiency and sensitivity evidence audit",
        "",
        "## Technical summary",
        "",
        f"The audit itself passes, but submission evidence is not yet complete: "
        f"{len(missing)} metric rows remain missing. Existing LLaMA-1B evidence is "
        "already sufficient for fixed-batch CUDA memory, exact optimizer/K-state "
        "accounting, and adjacent-integer capacity boundaries. Existing training "
        "wall-clock fields remain historical only.",
        "",
        "## Reusable paper-ready evidence",
        "",
        f"At device batch 32, Selective-none saves "
        f"{original['peak_allocated_mib'] - none['peak_allocated_mib']:.1f} MiB and "
        f"Selective-diag saves "
        f"{original['peak_allocated_mib'] - diag['peak_allocated_mib']:.1f} MiB of "
        "peak allocated CUDA memory versus original Newton-Muon. Both Selective "
        f"methods reach device batch {cap['selective_none']['max_success_device_batch']}, "
        f"versus {cap['original_newton_muon']['max_success_device_batch']} for original "
        f"Newton-Muon and {cap['muon']['max_success_device_batch']} for Muon.",
        "",
        "## Evidence boundaries",
        "",
        "- `tokens/s` and `steps/s` are missing because no accepted R1-PERF output exists.",
        "- Formal training timers are descriptive and cannot replace an isolated benchmark.",
        "- The B24 LR grid is historical; the R1 LR cross covers only Muon and Selective-diag.",
        "- Alpha is valid mechanism sensitivity, not learning-rate tuning fairness.",
        "",
        "## Minimal follow-up",
        "",
    ]
    for gap in gaps:
        if gap["gap_id"] == "NONE":
            lines.append("- No remote follow-up is required.")
        else:
            lines.append(
                f"- **{gap['gap_id']}**: {gap['minimal_resolution']}."
            )
    lines.extend(
        [
            "",
            "## Methodology",
            "",
            "The audit uses only paths frozen in `evidence_registry.json`, hashes every "
            "required file, independently normalizes the four method roles, and keeps "
            "audit validity separate from submission readiness. The comparison hierarchy "
            "is Selective versus Muon and original Newton-Muon; diag versus none is not "
            "a primary contrast.",
            "",
            "## Limitations",
            "",
            "Capacity and memory are measured on a single H100 configuration and are "
            "capacity evidence, not throughput evidence. The proposed LR grid is a "
            "single-seed robustness audit and must not be described as tuned-best "
            "multi-seed confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def make_artifact(
    generated_at: str,
    metric_rows: list[dict[str, Any]],
    capacity: list[dict[str, Any]],
    memory: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    title = "Submission efficiency and sensitivity evidence audit"
    sources = [
        {
            "id": "capacity_source",
            "label": "Accepted LLaMA-1B capacity and memory evidence",
            "path": "20_llama_swiglu_1b/analysis/capacity_fine_20260722_seed2026/derived/state_and_batch32_memory.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Four-role fixed-batch memory and exact capacity tables.",
                "sql": (
                    "SELECT method_role, peak_allocated_mib, peak_reserved_mib, "
                    "optimizer_state_mib, k_state_mib "
                    "FROM read_csv_auto('fixed_batch_memory.csv') "
                    "ORDER BY method_order"
                ),
                "tables_used": [
                    "fixed_batch_memory.csv",
                    "capacity_boundary.csv"
                ],
                "metric_definitions": [
                    "peak_allocated_mib is torch CUDA maximum allocated memory at device batch 32",
                    "maximum device batch is the completed batch immediately below the first OOM"
                ]
            }
        },
        {
            "id": "eligibility_source",
            "label": "Frozen evidence eligibility audit",
            "path": "metric_eligibility.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Paper-ready, supporting, historical, and missing classifications.",
                "sql": (
                    "SELECT metric, classification, paper_use, reason, followup_id "
                    "FROM read_csv_auto('metric_eligibility.csv')"
                ),
                "tables_used": [
                    "metric_eligibility.csv",
                    "sensitivity_coverage.csv",
                    "gap_matrix.csv"
                ]
            }
        }
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}", "layout": "full"},
        {
            "id": "summary",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Technical summary\n\nThe accepted 1B runs already support the "
                "memory/state/capacity claims. Isolated throughput and a final-recipe "
                "four-method LR grid remain missing. Audit completion is not submission readiness."
            ),
        },
        {
            "id": "eligibility_intro",
            "type": "markdown",
            "layout": "full",
            "body": "## Metric eligibility\n\nOnly evidence that passes the frozen comparability contract enters a paper-facing table.",
        },
        {"id": "eligibility_table_block", "type": "table", "tableId": "eligibility_table"},
        {
            "id": "memory_intro",
            "type": "markdown",
            "layout": "full",
            "body": "## Fixed-batch H100 memory\n\nPeak allocated memory at device batch 32; lower is better.",
        },
        {"id": "memory_chart_block", "type": "chart", "chartId": "memory_chart"},
        {"id": "memory_table_block", "type": "table", "tableId": "memory_table"},
        {
            "id": "capacity_intro",
            "type": "markdown",
            "layout": "full",
            "body": "## Exact capacity boundary\n\nEvery maximum-success batch is adjacent to the first OOM batch.",
        },
        {"id": "capacity_table_block", "type": "table", "tableId": "capacity_table"},
        {
            "id": "sensitivity_intro",
            "type": "markdown",
            "layout": "full",
            "body": "## Sensitivity coverage\n\nAlpha is mechanism evidence; it does not replace four-method LR robustness.",
        },
        {"id": "sensitivity_table_block", "type": "table", "tableId": "sensitivity_table"},
        {
            "id": "gaps_intro",
            "type": "markdown",
            "layout": "full",
            "body": "## Minimal follow-up\n\nOnly unresolved paper-facing gaps are retained.",
        },
        {"id": "gap_table_block", "type": "table", "tableId": "gap_table"},
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Limitations and claim boundary\n\nCapacity runs are not timing runs. "
                "The planned LR grid is supporting robustness, not a tuned-best or universal ranking claim. "
                "Selective-diag versus Selective-none remains secondary."
            ),
        },
    ]
    charts = [
        {
            "id": "memory_chart",
            "title": "Peak allocated CUDA memory at device batch 32",
            "subtitle": "LLaMA-1B, NVIDIA H100 80GB HBM3; same runtime and sequence length",
            "type": "bar",
            "dataset": "memory",
            "sourceId": "capacity_source",
            "encodings": {
                "x": {"field": "method_role", "type": "nominal", "label": "Method role"},
                "y": {
                    "field": "peak_allocated_mib",
                    "type": "quantitative",
                    "label": "Peak allocated (MiB)",
                    "format": "number"
                },
                "color": {"field": "method_role", "type": "nominal", "label": "Method role"},
                "tooltip": [
                    {"field": "optimizer_state_mib", "type": "quantitative", "label": "Optimizer state (MiB)"},
                    {"field": "k_state_mib", "type": "quantitative", "label": "K state (MiB)"},
                    {"field": "peak_reserved_mib", "type": "quantitative", "label": "Peak reserved (MiB)"}
                ]
            },
            "xAxisTitle": "Method role",
            "yAxisTitle": "Peak allocated (MiB)",
            "valueFormat": "number",
            "layout": "full",
            "maxRows": 4
        }
    ]
    tables = [
        {
            "id": "eligibility_table",
            "title": "Paper-facing metric eligibility",
            "dataset": "eligibility",
            "sourceId": "eligibility_source",
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "metric", "label": "Metric", "type": "text"},
                {"field": "classification", "label": "Class", "type": "text"},
                {"field": "paper_use", "label": "Paper use", "type": "text"},
                {"field": "reason", "label": "Reason", "type": "text"}
            ]
        },
        {
            "id": "memory_table",
            "title": "Fixed-batch memory and state",
            "dataset": "memory",
            "sourceId": "capacity_source",
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "method_role", "label": "Method", "type": "text"},
                {"field": "peak_allocated_mib", "label": "Peak allocated MiB", "type": "number"},
                {"field": "peak_reserved_mib", "label": "Peak reserved MiB", "type": "number"},
                {"field": "optimizer_state_mib", "label": "Optimizer state MiB", "type": "number"},
                {"field": "k_state_mib", "label": "K state MiB", "type": "number"}
            ]
        },
        {
            "id": "capacity_table",
            "title": "Exact device-batch capacity",
            "dataset": "capacity",
            "sourceId": "capacity_source",
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "method_role", "label": "Method", "type": "text"},
                {"field": "max_success_device_batch", "label": "Max success", "type": "number"},
                {"field": "first_oom_device_batch", "label": "First OOM", "type": "number"},
                {"field": "gain_vs_original_pct", "label": "Gain vs original (%)", "type": "number"}
            ]
        },
        {
            "id": "sensitivity_table",
            "title": "Sensitivity evidence coverage",
            "dataset": "sensitivity",
            "sourceId": "eligibility_source",
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "sensitivity_type", "label": "Type", "type": "text"},
                {"field": "source", "label": "Source", "type": "text"},
                {"field": "method_roles", "label": "Method roles", "type": "text"},
                {"field": "grid", "label": "Grid", "type": "text"},
                {"field": "classification", "label": "Class", "type": "text"},
                {"field": "limitation", "label": "Limitation", "type": "text"}
            ]
        },
        {
            "id": "gap_table",
            "title": "Minimal unresolved gaps",
            "dataset": "gaps",
            "sourceId": "eligibility_source",
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "gap_id", "label": "ID", "type": "text"},
                {"field": "gap", "label": "Gap", "type": "text"},
                {"field": "why_blocking", "label": "Why it matters", "type": "text"},
                {"field": "minimal_resolution", "label": "Minimal resolution", "type": "text"}
            ]
        }
    ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "Read-only audit of accepted efficiency and sensitivity evidence.",
        "generatedAt": generated_at,
        "blocks": blocks,
        "charts": charts,
        "tables": tables,
        "sources": sources
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "eligibility": metric_rows,
                "capacity": capacity,
                "memory": memory,
                "sensitivity": sensitivity,
                "gaps": gaps
            }
        },
        "sources": sources
    }


def main() -> None:
    args = parse_args()
    registry = read_json(args.registry)
    if registry.get("schema_version") != 1:
        raise RuntimeError("unsupported registry schema")
    portable_root = (
        args.portable_root
        if args.portable_root is not None
        else args.registry.parent / "source_snapshot"
    )
    source_rows, paths = audit_required_sources(
        args.input_root, registry, portable_root
    )
    foundation_checks = validate_foundations(paths)
    capacity = build_capacity(read_csv(paths["capacity_exact_boundary"]))
    memory = build_memory(read_csv(paths["fixed_batch_memory"]))
    # This validates every required historical sensitivity source even when the
    # expensive optional follow-ups have not yet been run.
    required_sensitivity = build_sensitivity(paths, None)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "passed": True,
                    "mode": "preflight_only",
                    "required_sources": len(source_rows),
                    "direct_sources": sum(
                        not str(row["relative_path"]).startswith(
                            "portable_snapshot/"
                        )
                        for row in source_rows
                    ),
                    "portable_sources": sum(
                        str(row["relative_path"]).startswith(
                            "portable_snapshot/"
                        )
                        for row in source_rows
                    ),
                    "foundation_checks": foundation_checks,
                    "capacity_roles": sorted(
                        row["method_role"] for row in capacity
                    ),
                    "memory_roles": sorted(
                        row["method_role"] for row in memory
                    ),
                    "required_sensitivity_rows": len(required_sensitivity),
                },
                sort_keys=True,
            )
        )
        return

    performance_source_rows, performance = discover_performance(
        args.input_root, registry
    )
    sensitivity_source_rows, completed_sensitivity = discover_sensitivity(
        args.input_root, registry
    )
    source_rows.extend(performance_source_rows)
    source_rows.extend(sensitivity_source_rows)
    sensitivity = build_sensitivity(paths, completed_sensitivity)
    metric_rows = build_metric_eligibility(performance, completed_sensitivity)
    throughput = performance_rows(performance)
    gaps = build_gap_matrix(metric_rows)
    followup = followup_contract(metric_rows)
    generated_at = datetime.now(timezone.utc).isoformat()
    assert args.output_dir is not None
    args.output_dir.mkdir(parents=True, exist_ok=False)

    output_files = {
        "source_inventory.csv": source_rows,
        "metric_eligibility.csv": metric_rows,
        "capacity_boundary.csv": capacity,
        "fixed_batch_memory.csv": memory,
        "sensitivity_coverage.csv": sensitivity,
        "throughput_summary.csv": throughput,
        "gap_matrix.csv": gaps,
    }
    for name, rows in output_files.items():
        write_csv(args.output_dir / name, rows)
    write_json(args.output_dir / "minimal_followup_contract.json", followup)
    report = build_report(metric_rows, capacity, memory, sensitivity, gaps)
    (args.output_dir / "SUBMISSION_EVIDENCE_AUDIT_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    artifact = make_artifact(
        generated_at, metric_rows, capacity, memory, sensitivity, gaps
    )
    write_json(args.output_dir / "artifact.json", artifact)
    write_json(
        args.output_dir / "report_source_notes.json",
        {
            "schema_version": 1,
            "audience": "technical",
            "delivery_mode": "portable_html",
            "chart_map": [
                {
                    "section": "memory",
                    "type": "bar",
                    "dataset": "memory",
                    "analytical_question": "How much fixed-batch peak memory does each frozen role require?",
                    "supported_claim": "Selective roles reduce H100 peak memory versus original Newton-Muon."
                }
            ],
            "omitted_chart_reason": "Throughput has no accepted values; a missing-data chart would be misleading.",
            "comparison_priority": "Selective methods versus Muon and original Newton-Muon; diag versus none secondary only."
        }
    )

    artifact_names = sorted(
        list(output_files)
        + [
            "minimal_followup_contract.json",
            "SUBMISSION_EVIDENCE_AUDIT_REPORT.md",
            "artifact.json",
            "report_source_notes.json",
        ]
    )
    output_hashes = {
        name: sha256_file(args.output_dir / name) for name in artifact_names
    }
    missing_metrics = [
        row["metric"] for row in metric_rows if row["classification"] == "missing"
    ]
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "contract_version": registry["contract_version"],
        "created_at": generated_at,
        "audit_passed": True,
        "submission_ready": not missing_metrics,
        "comparison_priority": "selective_vs_muon_and_original_newton_muon",
        "diag_vs_none_primary": False,
        "required_sources": len(registry["required_files"]),
        "isolated_performance_found": performance is not None,
        "paper_ready_metrics": sum(
            row["classification"] == "paper_ready" for row in metric_rows
        ),
        "missing_metrics": missing_metrics,
        "blocking_followups": [
            item["id"] for item in followup["experiments"]
        ],
        "foundation_checks": foundation_checks,
        "output_sha256": output_hashes,
        "artifacts": artifact_names,
    }
    write_json(args.output_dir / "submission_evidence_manifest.json", manifest)
    print(f"Submission evidence artifacts: {args.output_dir}")
    print(
        f"Submission evidence manifest:  "
        f"{args.output_dir / 'submission_evidence_manifest.json'}"
    )


if __name__ == "__main__":
    main()
