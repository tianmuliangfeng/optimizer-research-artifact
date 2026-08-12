#!/usr/bin/env python3
"""Independently audit and summarize experiment 42 formal efficiency cells.

The analysis intentionally reads only the frozen contract and the completed
formal cell evidence: cell manifests, worker manifests, trainer summaries, and
the before/after exclusive-node certificates.  It never reads benchmark loss
or historical quality rankings because experiment 42 is an implementation
efficiency audit, not an optimizer-quality experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_VERSION = "2026-07-29.1"
CONTROLLER_VERSION = "2026-07-29.1"
WORKER_VERSION = "2026-07-29.1"
GPU_MONITOR_VERSION = "2026-07-29.1"
CERTIFIER_VERSION = "2026-07-29.1"
T_CRITICAL_DF3_95 = 3.182446
MIB = 1024**2
FROZEN_METHOD_ORDER = ["muon", "newton_full", "down_none", "down_diag"]
FROZEN_PRIMARY_CONTRASTS = [
    {"candidate": "down_none", "reference": "muon"},
    {"candidate": "down_none", "reference": "newton_full"},
    {"candidate": "down_diag", "reference": "muon"},
    {"candidate": "down_diag", "reference": "newton_full"},
    {"candidate": "newton_full", "reference": "muon"},
]
STABLE_RUNTIME_FIELDS = (
    "python_executable",
    "python_version",
    "numpy",
    "torch",
    "torch_cuda",
    "triton",
    "triton_kernels_sha256",
    "gpu_name",
    "gpu_total_memory_bytes",
    "gpu_capability",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_runtime_fingerprint(runtime: Any) -> str:
    if not isinstance(runtime, dict):
        raise RuntimeError("trainer runtime metadata is not an object")
    stable = {field: runtime.get(field) for field in STABLE_RUNTIME_FIELDS}
    return canonical_json_sha256(stable)


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def finite_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric measurement")
    observed = float(value)
    if not math.isfinite(observed):
        raise ValueError(f"non-finite value: {value!r}")
    return observed


def exact_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not an integer measurement")
    observed = int(value)
    if float(value) != float(observed):
        raise ValueError(f"non-integral value: {value!r}")
    return observed


def sample_stdev(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def mean_ci95(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 4:
        raise RuntimeError(f"the frozen design requires four paired values, got {len(values)}")
    mean = statistics.mean(values)
    half_width = T_CRITICAL_DF3_95 * sample_stdev(values) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def describe(values: Sequence[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    standard_deviation = sample_stdev(values)
    return {
        "mean": mean,
        "median": statistics.median(values),
        "stdev": standard_deviation,
        "cv_pct": 100.0 * standard_deviation / mean if mean else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


@dataclass
class Audit:
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def fail(self, check: str, message: str) -> None:
        self.checks[check] = False
        self.errors.append(f"{check}: {message}")

    def pass_if(self, check: str, condition: bool, message: str) -> bool:
        if condition:
            self.checks.setdefault(check, True)
            return True
        self.fail(check, message)
        return False

    def guard(self, check: str, operation: Any, context: str) -> Any:
        try:
            result = operation()
            self.checks.setdefault(check, True)
            return result
        except Exception as error:  # keep a complete integrity report
            self.fail(check, f"{context}: {error!r}")
            return None


def resolve_evidence_path(run_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty path string")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    candidate = candidate.resolve(strict=True)
    try:
        candidate.relative_to(run_dir)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes run-dir: {candidate}") from error
    if not candidate.is_file():
        raise RuntimeError(f"{label} is not a file: {candidate}")
    return candidate


def require_hash(
    audit: Audit,
    check: str,
    path: Path,
    expected: Any,
    label: str,
) -> bool:
    if not path.is_file():
        audit.fail(check, f"{label} is missing: {path}")
        return False
    if not is_sha256(expected):
        audit.fail(check, f"{label} has malformed expected SHA256: {expected!r}")
        return False
    try:
        observed = sha256_file(path)
    except OSError as error:
        audit.fail(check, f"{label} cannot be hashed: {error!r}")
        return False
    if observed != expected:
        audit.fail(
            check,
            f"{label} SHA256 mismatch: expected={expected}, observed={observed}",
        )
        return False
    audit.checks.setdefault(check, True)
    return True


def fingerprint_key(value: Any) -> str:
    if value is None or value == "" or value == {} or value == []:
        raise RuntimeError("empty fingerprint")
    return canonical_json(value)


def fingerprint_display(value: Any) -> str:
    return value if isinstance(value, str) else canonical_json(value)


def numeric_equal(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            finite_float(left),
            finite_float(right),
            rel_tol=1e-10,
            abs_tol=1e-8,
        )
    except (TypeError, ValueError):
        return False


def validate_certificate(
    payload: Mapping[str, Any],
    required_gpu_indices: Sequence[str],
    required_gpu_name: str,
) -> tuple[bool, float, list[str]]:
    errors: list[str] = []
    required = sorted(int(index) for index in required_gpu_indices)
    if payload.get("passed") is not True:
        errors.append("certificate passed is not true")
    if payload.get("script_version") != CERTIFIER_VERSION:
        errors.append(
            "certificate script version mismatch: "
            f"expected={CERTIFIER_VERSION}, "
            f"observed={payload.get('script_version')!r}"
        )
    certificate_checks = payload.get("checks")
    if (
        not isinstance(certificate_checks, dict)
        or not certificate_checks
        or not all(value is True for value in certificate_checks.values())
    ):
        errors.append(
            f"certificate checks are absent or not all true: {certificate_checks!r}"
        )
    try:
        observed_required = sorted(int(index) for index in payload.get("required_gpus", []))
    except (TypeError, ValueError):
        observed_required = []
    if observed_required != required:
        errors.append(
            f"required_gpus mismatch: expected={required}, observed={observed_required}"
        )
    processes = payload.get("active_compute_processes")
    if processes != []:
        errors.append(f"active_compute_processes is not empty: {processes!r}")
    gpu_rows = payload.get("gpus")
    if not isinstance(gpu_rows, list):
        gpu_rows = []
        errors.append("gpus is not a list")
    by_index: dict[int, Mapping[str, Any]] = {}
    for row in gpu_rows:
        if isinstance(row, dict):
            try:
                by_index[int(row["index"])] = row
            except (KeyError, TypeError, ValueError):
                errors.append(f"malformed GPU row: {row!r}")
    if sorted(by_index) != required:
        errors.append(
            f"exact GPU inventory mismatch: expected={required}, "
            f"observed={sorted(by_index)}"
        )
    uuids = [
        row.get("uuid")
        for row in by_index.values()
        if isinstance(row.get("uuid"), str) and row.get("uuid")
    ]
    if len(uuids) != len(required) or len(set(uuids)) != len(required):
        errors.append(f"GPU UUID inventory is missing or non-unique: {uuids!r}")
    free_fractions: list[float] = []
    for index in required:
        row = by_index.get(index)
        if row is None:
            errors.append(f"required GPU {index} missing")
            continue
        if row.get("name") != required_gpu_name:
            errors.append(
                f"GPU {index} name mismatch: expected={required_gpu_name!r}, "
                f"observed={row.get('name')!r}"
            )
        try:
            free_fraction = finite_float(row.get("free_fraction"))
        except (TypeError, ValueError) as error:
            errors.append(f"GPU {index} free_fraction invalid: {error}")
            continue
        free_fractions.append(free_fraction)
        if free_fraction < 0.98:
            errors.append(f"GPU {index} free_fraction below 0.98: {free_fraction}")
    minimum = min(free_fractions) if free_fractions else float("nan")
    return not errors, minimum, errors


def validate_gpu_isolation_monitor(
    payload: Mapping[str, Any],
    *,
    timing_gpu: int,
    idle_gpus: Sequence[int],
    interval_seconds: float,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    checks = payload.get("checks")
    if payload.get("passed") is not True:
        errors.append("monitor passed is not true")
    if payload.get("script_version") != GPU_MONITOR_VERSION:
        errors.append(
            "monitor script version mismatch: "
            f"expected={GPU_MONITOR_VERSION}, "
            f"observed={payload.get('script_version')!r}"
        )
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        errors.append(f"monitor checks are absent or not all true: {checks!r}")
    if payload.get("timing_gpu") != timing_gpu:
        errors.append(
            f"timing_gpu mismatch: expected={timing_gpu}, "
            f"observed={payload.get('timing_gpu')!r}"
        )
    if payload.get("idle_gpus") != list(idle_gpus):
        errors.append(
            f"idle_gpus mismatch: expected={list(idle_gpus)}, "
            f"observed={payload.get('idle_gpus')!r}"
        )
    try:
        if exact_int(payload.get("sample_count")) < 2:
            errors.append(f"sample_count below two: {payload.get('sample_count')!r}")
    except (TypeError, ValueError):
        errors.append(f"invalid sample_count: {payload.get('sample_count')!r}")
    try:
        if not math.isclose(
            finite_float(payload.get("sample_interval_seconds")),
            interval_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            errors.append(
                "sample interval mismatch: "
                f"expected={interval_seconds}, "
                f"observed={payload.get('sample_interval_seconds')!r}"
            )
    except (TypeError, ValueError):
        errors.append(
            f"invalid sample interval: {payload.get('sample_interval_seconds')!r}"
        )
    if not isinstance(checks, dict) or checks.get(
        "single_timing_process_identity"
    ) is not True:
        errors.append("single_timing_process_identity is not true")
    if not isinstance(checks, dict) or checks.get(
        "idle_gpu_processes_absent"
    ) is not True:
        errors.append("idle_gpu_processes_absent is not true")
    timing_process_pids = payload.get("timing_process_pids")
    if not isinstance(timing_process_pids, list) or len(timing_process_pids) != 1:
        errors.append(
            f"expected exactly one timing process identity: {timing_process_pids!r}"
        )
    if payload.get("idle_process_events") != []:
        errors.append("idle_process_events is not empty")
    if payload.get("query_errors") != []:
        errors.append("query_errors is not empty")
    return not errors, errors


def certificate_gpu_uuid_map(payload: Mapping[str, Any]) -> dict[str, str]:
    gpu_rows = payload.get("gpus")
    if not isinstance(gpu_rows, list):
        return {}
    observed: dict[str, str] = {}
    for row in gpu_rows:
        if not isinstance(row, dict):
            return {}
        try:
            index = str(exact_int(row.get("index")))
        except (TypeError, ValueError):
            return {}
        uuid = row.get("uuid")
        if not isinstance(uuid, str) or not uuid:
            return {}
        observed[index] = uuid
    return observed


def metric_row(
    *,
    repeat_index: int,
    position_index: int,
    method: str,
    role: str,
    cell: Mapping[str, Any],
    worker: Mapping[str, Any],
    summary: Mapping[str, Any],
    before_minimum_free: float,
    after_minimum_free: float,
    worker_path: Path,
    summary_path: Path,
    run_dir: Path,
    monitor: Mapping[str, Any],
    tokens_per_update: int,
) -> dict[str, Any]:
    step_avg_ms = finite_float(summary["step_avg_ms"])
    if step_avg_ms <= 0:
        raise ValueError(f"step_avg_ms must be positive, got {step_avg_ms}")
    return {
        "repeat_index": repeat_index,
        "position_index": position_index,
        "method": method,
        "method_role": role,
        "attempt": cell.get("attempt"),
        "steady_train_s": finite_float(summary["steady_train_s"]),
        "steady_steps": exact_int(summary["steady_steps"]),
        "step_avg_ms": step_avg_ms,
        "steps_per_s": 1000.0 / step_avg_ms,
        "tokens_per_s": tokens_per_update * 1000.0 / step_avg_ms,
        "tokens_seen": exact_int(summary["tokens_seen"]),
        "timed_training_peak_allocated_bytes": exact_int(
            summary["timed_training_peak_allocated_bytes"]
        ),
        "timed_training_peak_allocated_mib": exact_int(
            summary["timed_training_peak_allocated_bytes"]
        )
        / MIB,
        "timed_training_peak_reserved_bytes": exact_int(
            summary["timed_training_peak_reserved_bytes"]
        ),
        "timed_training_peak_reserved_mib": exact_int(
            summary["timed_training_peak_reserved_bytes"]
        )
        / MIB,
        "k_state_bytes": exact_int(summary["k_state_bytes"]),
        "k_state_mib": exact_int(summary["k_state_bytes"]) / MIB,
        "optimizer_state_bytes": exact_int(summary["optimizer_state_bytes"]),
        "optimizer_state_mib": exact_int(summary["optimizer_state_bytes"]) / MIB,
        "model_parameter_bytes": exact_int(summary["model_parameter_bytes"]),
        "init_sha256": summary["init_sha256"],
        "runtime_fingerprint": fingerprint_display(cell["runtime_fingerprint"]),
        "data_fingerprint": fingerprint_display(cell["data_fingerprint"]),
        "derived_base_sha256": worker["derived_base_sha256"],
        "profile_wrapper_sha256": worker["profile_wrapper_sha256"],
        "exclusive_before_minimum_free_fraction": before_minimum_free,
        "exclusive_after_minimum_free_fraction": after_minimum_free,
        "gpu_monitor_sample_count": exact_int(monitor["sample_count"]),
        "gpu_monitor_timing_process_pid": exact_int(
            monitor["timing_process_pids"][0]
        ),
        "worker_manifest": worker_path.relative_to(run_dir).as_posix(),
        "trainer_summary": summary_path.relative_to(run_dir).as_posix(),
    }


RAW_FIELDS = [
    "repeat_index",
    "position_index",
    "method",
    "method_role",
    "attempt",
    "steady_train_s",
    "steady_steps",
    "step_avg_ms",
    "steps_per_s",
    "tokens_per_s",
    "tokens_seen",
    "timed_training_peak_allocated_bytes",
    "timed_training_peak_allocated_mib",
    "timed_training_peak_reserved_bytes",
    "timed_training_peak_reserved_mib",
    "k_state_bytes",
    "k_state_mib",
    "optimizer_state_bytes",
    "optimizer_state_mib",
    "model_parameter_bytes",
    "init_sha256",
    "runtime_fingerprint",
    "data_fingerprint",
    "derived_base_sha256",
    "profile_wrapper_sha256",
    "exclusive_before_minimum_free_fraction",
    "exclusive_after_minimum_free_fraction",
    "gpu_monitor_sample_count",
    "gpu_monitor_timing_process_pid",
    "worker_manifest",
    "trainer_summary",
]


SUMMARY_METRICS = [
    "step_avg_ms",
    "steps_per_s",
    "tokens_per_s",
    "timed_training_peak_allocated_bytes",
    "timed_training_peak_reserved_bytes",
    "k_state_bytes",
    "optimizer_state_bytes",
]


def build_method_summary(
    raw_rows: Sequence[Mapping[str, Any]],
    method_order: Sequence[str],
    roles: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    fields = ["method", "method_role", "repeats"]
    for metric in SUMMARY_METRICS:
        fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_median",
                f"{metric}_stdev",
                f"{metric}_cv_pct",
                f"{metric}_minimum",
                f"{metric}_maximum",
            ]
        )
    fields.extend(
        [
            "timed_training_peak_allocated_mib_median",
            "timed_training_peak_reserved_mib_median",
            "k_state_mib_exact",
            "optimizer_state_mib_median",
        ]
    )
    for method in method_order:
        selected = [row for row in raw_rows if row["method"] == method]
        row: dict[str, Any] = {
            "method": method,
            "method_role": roles[method],
            "repeats": len(selected),
        }
        for metric in SUMMARY_METRICS:
            description = describe([finite_float(item[metric]) for item in selected])
            for statistic, value in description.items():
                row[f"{metric}_{statistic}"] = value
        row["timed_training_peak_allocated_mib_median"] = (
            row["timed_training_peak_allocated_bytes_median"] / MIB
        )
        row["timed_training_peak_reserved_mib_median"] = (
            row["timed_training_peak_reserved_bytes_median"] / MIB
        )
        row["k_state_mib_exact"] = row["k_state_bytes_median"] / MIB
        row["optimizer_state_mib_median"] = row["optimizer_state_bytes_median"] / MIB
        rows.append(row)
    return rows, fields


def throughput_band_classification(metric: str, relative_mean_pct: float | None, band: float) -> str:
    if metric not in {"step_avg_ms", "steps_per_s", "tokens_per_s"}:
        return "not_classified_by_throughput_band"
    if relative_mean_pct is None:
        return "undefined_relative_change"
    if abs(relative_mean_pct) <= band:
        return f"within_descriptive_plus_or_minus_{band:g}_pct_band"
    direction = "higher" if relative_mean_pct > 0 else "lower"
    return f"candidate_{direction}_{metric}_outside_descriptive_band"


def build_paired_contrasts(
    raw_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, str]],
    practical_band_pct: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    lookup = {
        (exact_int(row["repeat_index"]), str(row["method"])): row for row in raw_rows
    }
    fields = [
        "contrast_scope",
        "candidate",
        "reference",
        "metric",
        "n_pairs",
        "delta_definition",
        "repeat_0_delta",
        "repeat_1_delta",
        "repeat_2_delta",
        "repeat_3_delta",
        "mean_delta",
        "median_delta",
        "stdev_delta",
        "ci95_lower",
        "ci95_upper",
        "t_critical_df3",
        "repeat_0_relative_pct",
        "repeat_1_relative_pct",
        "repeat_2_relative_pct",
        "repeat_3_relative_pct",
        "mean_relative_pct",
        "relative_ci95_lower_pct",
        "relative_ci95_upper_pct",
        "throughput_band_pct",
        "descriptive_classification",
    ]
    rows: list[dict[str, Any]] = []
    for contrast in contrasts:
        candidate = contrast["candidate"]
        reference = contrast["reference"]
        for metric in SUMMARY_METRICS:
            deltas: list[float] = []
            relative: list[float] = []
            relative_defined = True
            for repeat_index in range(4):
                candidate_value = finite_float(lookup[(repeat_index, candidate)][metric])
                reference_value = finite_float(lookup[(repeat_index, reference)][metric])
                deltas.append(candidate_value - reference_value)
                if reference_value == 0:
                    relative_defined = False
                else:
                    relative.append(100.0 * (candidate_value / reference_value - 1.0))
            mean_delta, lower, upper = mean_ci95(deltas)
            relative_mean: float | None = None
            relative_lower: float | None = None
            relative_upper: float | None = None
            if relative_defined:
                relative_mean, relative_lower, relative_upper = mean_ci95(relative)
            row: dict[str, Any] = {
                "contrast_scope": "primary",
                "candidate": candidate,
                "reference": reference,
                "metric": metric,
                "n_pairs": 4,
                "delta_definition": "candidate_minus_reference_within_repeat",
                **{f"repeat_{index}_delta": value for index, value in enumerate(deltas)},
                "mean_delta": mean_delta,
                "median_delta": statistics.median(deltas),
                "stdev_delta": sample_stdev(deltas),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "t_critical_df3": T_CRITICAL_DF3_95,
                **{
                    f"repeat_{index}_relative_pct": (
                        relative[index] if relative_defined else ""
                    )
                    for index in range(4)
                },
                "mean_relative_pct": relative_mean if relative_mean is not None else "",
                "relative_ci95_lower_pct": (
                    relative_lower if relative_lower is not None else ""
                ),
                "relative_ci95_upper_pct": (
                    relative_upper if relative_upper is not None else ""
                ),
                "throughput_band_pct": (
                    practical_band_pct
                    if metric in {"step_avg_ms", "steps_per_s", "tokens_per_s"}
                    else ""
                ),
                "descriptive_classification": throughput_band_classification(
                    metric, relative_mean, practical_band_pct
                ),
            }
            rows.append(row)
    return rows, fields


def build_position_effects(
    raw_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    fields = [
        "position_index",
        "metric",
        "n",
        "methods",
        "mean",
        "median",
        "stdev",
        "cv_pct",
        "overall_mean",
        "position_minus_overall_pct",
    ]
    rows: list[dict[str, Any]] = []
    for metric in ["step_avg_ms", "steps_per_s", "tokens_per_s"]:
        overall = statistics.mean([finite_float(row[metric]) for row in raw_rows])
        for position_index in range(4):
            selected = [
                row for row in raw_rows if exact_int(row["position_index"]) == position_index
            ]
            values = [finite_float(row[metric]) for row in selected]
            description = describe(values)
            rows.append(
                {
                    "position_index": position_index,
                    "metric": metric,
                    "n": len(values),
                    "methods": ",".join(str(row["method"]) for row in selected),
                    "mean": description["mean"],
                    "median": description["median"],
                    "stdev": description["stdev"],
                    "cv_pct": description["cv_pct"],
                    "overall_mean": overall,
                    "position_minus_overall_pct": (
                        100.0 * (description["mean"] / overall - 1.0)
                    ),
                }
            )
    return rows, fields


def format_number(value: Any, digits: int = 3) -> str:
    try:
        return f"{finite_float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


SNAPSHOT_REQUIRED_FILES = (
    "efficiency_common.py",
    "source_builder.py",
    "llama1b_efficiency_worker.py",
    "gpu_isolation_monitor.py",
    "run_llama1b_efficiency.py",
    "analyze_llama1b_efficiency.py",
    "certify_exclusive_node.py",
    "efficiency_contract.json",
    "train_llama_swiglu_efficiency_base.py",
    "train_llama_swiglu_1b.py",
    "train_llama_swiglu_efficiency_base.diff",
)


def audit_preflight(
    run_dir: Path,
    contract_path: Path,
    contract: Mapping[str, Any],
    audit: Audit,
) -> tuple[dict[str, Any], str]:
    run_contract_path = run_dir / "efficiency_contract.json"
    preflight_path = run_dir / "preflight.json"
    if not run_contract_path.is_file() or not preflight_path.is_file():
        audit.fail(
            "root_evidence_present",
            "run contract or preflight.json is missing",
        )
        return {}, ""
    audit.checks["root_evidence_present"] = True
    run_contract_sha = sha256_file(run_contract_path)
    supplied_contract_sha = sha256_file(contract_path)
    audit.pass_if(
        "run_contract_matches_analysis_contract",
        run_contract_sha == supplied_contract_sha,
        (
            f"run contract SHA={run_contract_sha} differs from supplied "
            f"contract SHA={supplied_contract_sha}"
        ),
    )
    preflight = audit.guard(
        "preflight_readable", lambda: read_json(preflight_path), str(preflight_path)
    )
    if preflight is None:
        return {}, ""
    preflight_sha = sha256_file(preflight_path)
    expected_runtime = preflight.get("stable_runtime")
    runtime = preflight.get("runtime")
    runtime_stable = (
        {field: runtime.get(field) for field in STABLE_RUNTIME_FIELDS}
        if isinstance(runtime, dict)
        else None
    )
    runtime_fingerprint = (
        canonical_json_sha256(expected_runtime)
        if isinstance(expected_runtime, dict)
        else None
    )
    source_contract = contract.get("source_contract", {})
    official = preflight.get("official_repo_audit")
    data = preflight.get("data_audit")
    init = preflight.get("init_audit")
    inventory = preflight.get("gpu_inventory")
    preflight_core_ok = (
        preflight.get("passed") is True
        and preflight.get("script_version") == CONTROLLER_VERSION
        and preflight.get("contract_sha256") == supplied_contract_sha
        and preflight.get("physical_timing_gpu")
        == contract.get("execution_policy", {}).get("physical_timing_gpu")
        and preflight.get("required_gpus")
        == contract.get("execution_policy", {}).get("required_idle_physical_gpus")
        and isinstance(expected_runtime, dict)
        and runtime_stable == expected_runtime
        and preflight.get("runtime_fingerprint") == runtime_fingerprint
        and isinstance(runtime, dict)
        and runtime.get("gpu_name")
        == contract.get("execution_policy", {}).get("required_gpu_name")
        and runtime.get("triton_kernels_sha256")
        == source_contract.get("triton_kernels_sha256")
    )
    audit.pass_if(
        "preflight_contract_runtime",
        preflight_core_ok,
        "preflight pass/contract/GPU/stable-runtime evidence differs",
    )
    official_ok = (
        isinstance(official, dict)
        and official.get("passed") is True
        and official.get("commit") == source_contract.get("official_repo_commit")
        and official.get("triton_kernels_sha256")
        == source_contract.get("triton_kernels_sha256")
    )
    audit.pass_if(
        "preflight_official_source",
        official_ok,
        "official repository commit or Triton-kernel evidence differs",
    )
    data_without_fingerprint: dict[str, Any] = {}
    if isinstance(data, dict):
        data_without_fingerprint = dict(data)
        data_without_fingerprint.pop("fingerprint", None)
    data_fingerprint = (
        canonical_json_sha256(data_without_fingerprint)
        if data_without_fingerprint
        else None
    )
    try:
        data_rows_valid = isinstance(data, dict) and isinstance(
            data.get("files"), list
        ) and all(
            isinstance(row, dict)
            and is_sha256(row.get("sha256"))
            and exact_int(row.get("tokens")) > 0
            and exact_int(row.get("bytes")) > 0
            for row in data["files"]
        )
    except (TypeError, ValueError):
        data_rows_valid = False
    data_ok = (
        isinstance(data, dict)
        and data.get("train_shard_count") == 50
        and data.get("validation_shard_count") == 1
        and isinstance(data.get("files"), list)
        and len(data["files"]) == 51
        and data.get("fingerprint") == data_fingerprint
        and data_rows_valid
    )
    audit.pass_if(
        "preflight_data_fingerprint",
        data_ok,
        "preflight data inventory or canonical fingerprint differs",
    )
    init_methods = init.get("methods") if isinstance(init, dict) else None
    init_fingerprint = (
        canonical_json_sha256(init_methods) if isinstance(init_methods, dict) else None
    )
    frozen_init = contract.get("frozen_configuration", {}).get(
        "initialization_sha256"
    )
    init_ok = (
        isinstance(init, dict)
        and init.get("common_init_sha256") == frozen_init
        and init.get("fingerprint") == init_fingerprint
        and isinstance(init_methods, dict)
        and set(init_methods) == set(contract.get("method_order", []))
        and all(
            isinstance(init_methods.get(method), dict)
            and (row := init_methods[method]).get("method") == method
            and row.get("init_sha256") == frozen_init
            for method in contract.get("method_order", [])
        )
    )
    audit.pass_if(
        "preflight_initialization_fingerprint",
        init_ok,
        "preflight initialization inventory or fingerprint differs",
    )
    required_indices = {
        int(value)
        for value in contract.get("execution_policy", {}).get(
            "required_idle_physical_gpus", []
        )
    }
    gpu_rows = inventory.get("gpus") if isinstance(inventory, dict) else None
    try:
        observed_gpu_indices = (
            {exact_int(row.get("index")) for row in gpu_rows}
            if isinstance(gpu_rows, list)
            else set()
        )
    except (TypeError, ValueError):
        observed_gpu_indices = set()
    inventory_ok = (
        isinstance(inventory, dict)
        and isinstance(gpu_rows, list)
        and observed_gpu_indices == required_indices
        and all(
            row.get("name")
            == contract.get("execution_policy", {}).get("required_gpu_name")
            for row in gpu_rows
        )
        and inventory.get("fingerprint") == canonical_json_sha256(gpu_rows)
    )
    audit.pass_if(
        "preflight_gpu_inventory",
        inventory_ok,
        "physical GPU inventory differs from the frozen two-H100 node",
    )

    source_audit = preflight.get("source_audit")
    source_files = source_audit.get("files") if isinstance(source_audit, dict) else None
    snapshot = run_dir / "source_snapshot"
    source_errors: list[str] = []
    if not isinstance(source_files, dict):
        source_errors.append("source_audit.files is not an object")
        source_files = {}
    for name in SNAPSHOT_REQUIRED_FILES:
        expected = source_files.get(name)
        path = snapshot / name
        if not path.is_file():
            source_errors.append(f"snapshot file missing: {name}")
        elif not is_sha256(expected):
            source_errors.append(f"snapshot SHA malformed: {name}={expected!r}")
        elif sha256_file(path) != expected:
            source_errors.append(f"snapshot SHA mismatch: {name}")
    source_internal_ok = (
        isinstance(source_audit, dict)
        and source_audit.get("base_trainer_sha256")
        == source_contract.get("base_trainer_sha256")
        and source_audit.get("profile_wrapper_source_sha256")
        == source_contract.get("profile_wrapper_sha256")
        and source_audit.get("derived_base_sha256")
        == source_files.get("train_llama_swiglu_efficiency_base.py")
        == source_contract.get("derived_efficiency_base_sha256")
        and source_audit.get("profile_wrapper_sha256")
        == source_files.get("train_llama_swiglu_1b.py")
        == source_contract.get("profile_wrapper_sha256")
        and source_audit.get("source_diff_sha256")
        == source_files.get("train_llama_swiglu_efficiency_base.diff")
        and source_files.get("efficiency_contract.json") == supplied_contract_sha
    )
    if not source_internal_ok:
        source_errors.append("source-audit internal pins differ from contract/files")
    audit.pass_if(
        "preflight_source_snapshot_hashes",
        not source_errors,
        "; ".join(source_errors),
    )
    preflight_certificate = run_dir / "preflight_exclusive_node.json"
    if preflight_certificate.is_file():
        certificate_hash_ok = (
            sha256_file(preflight_certificate)
            == preflight.get("exclusive_node_certificate_sha256")
        )
        certificate_payload = audit.guard(
            "preflight_exclusive_certificate_readable",
            lambda: read_json(preflight_certificate),
            str(preflight_certificate),
        )
        certificate_valid = False
        if certificate_payload is not None:
            certificate_valid = validate_certificate(
                certificate_payload,
                contract.get("execution_policy", {}).get(
                    "required_idle_physical_gpus", []
                ),
                contract.get("execution_policy", {}).get("required_gpu_name", ""),
            )[0]
        audit.pass_if(
            "preflight_exclusive_certificate",
            certificate_hash_ok and certificate_valid,
            "preflight exclusive-node certificate hash or payload differs",
        )
    else:
        audit.fail(
            "preflight_exclusive_certificate",
            "preflight_exclusive_node.json is missing",
        )
    return preflight, preflight_sha


def audit_execution_manifest(
    run_dir: Path,
    contract_sha256: str,
    preflight_sha256: str,
    preflight_data_fingerprint: Any,
    expected_paths: Mapping[Path, tuple[int, int, str]],
    audit: Audit,
) -> tuple[dict[str, Any], str]:
    path = run_dir / "execution_manifest.json"
    if not path.is_file():
        audit.fail("execution_manifest_present", "execution_manifest.json is missing")
        return {}, ""
    audit.checks["execution_manifest_present"] = True
    payload = audit.guard(
        "execution_manifest_readable", lambda: read_json(path), str(path)
    )
    if payload is None:
        return {}, ""
    execution_sha256 = sha256_file(path)
    expected_relative = [
        manifest.relative_to(run_dir).as_posix() for manifest in expected_paths
    ]
    observed_relative = payload.get("formal_cell_manifests")
    observed_hashes = payload.get("formal_cell_manifest_sha256")
    hashes_ok = isinstance(observed_hashes, dict) and set(observed_hashes) == set(
        expected_relative
    )
    if hashes_ok:
        hashes_ok = all(
            (run_dir / relative).is_file()
            and is_sha256(observed_hashes[relative])
            and sha256_file(run_dir / relative) == observed_hashes[relative]
            for relative in expected_relative
        )
    smoke_path_value = payload.get("smoke_manifest")
    smoke_hash_ok = False
    if isinstance(smoke_path_value, str):
        try:
            smoke_path = resolve_evidence_path(
                run_dir, smoke_path_value, "smoke_manifest"
            )
            smoke_hash_ok = require_hash(
                audit,
                "execution_smoke_manifest_hash",
                smoke_path,
                payload.get("smoke_manifest_sha256"),
                "execution smoke manifest",
            )
        except Exception as error:
            audit.fail("execution_smoke_manifest_hash", repr(error))
    postflight_ok = False
    postflight_value = payload.get("postflight_data_audit")
    if isinstance(postflight_value, str):
        try:
            postflight_path = resolve_evidence_path(
                run_dir, postflight_value, "postflight_data_audit"
            )
            postflight_hash_ok = require_hash(
                audit,
                "execution_postflight_data_hash",
                postflight_path,
                payload.get("postflight_data_audit_sha256"),
                "postflight data audit",
            )
            postflight = read_json(postflight_path)
            postflight_without_fingerprint = dict(postflight)
            postflight_without_fingerprint.pop("fingerprint", None)
            postflight_ok = (
                postflight_hash_ok
                and postflight.get("fingerprint") == preflight_data_fingerprint
                and postflight.get("fingerprint")
                == canonical_json_sha256(postflight_without_fingerprint)
                and postflight.get("train_shard_count") == 50
                and postflight.get("validation_shard_count") == 1
                and isinstance(postflight.get("files"), list)
                and len(postflight["files"]) == 51
            )
        except Exception as error:
            audit.fail("execution_postflight_data_hash", repr(error))
    audit.pass_if(
        "execution_postflight_data_fingerprint",
        postflight_ok,
        "sealed postflight data inventory differs from preflight",
    )
    manifest_ok = (
        payload.get("status") == "completed"
        and payload.get("passed") is True
        and payload.get("script_version") == CONTROLLER_VERSION
        and payload.get("contract_sha256") == contract_sha256
        and payload.get("preflight_sha256") == preflight_sha256
        and payload.get("formal_cell_count") == 16
        and observed_relative == expected_relative
        and hashes_ok
        and smoke_hash_ok
        and postflight_ok
    )
    audit.pass_if(
        "execution_manifest_and_cell_hash_inventory",
        manifest_ok,
        "execution manifest status, order, count, or sealed cell hashes differ",
    )
    return payload, execution_sha256


def build_report(
    *,
    passed: bool,
    contract: Mapping[str, Any],
    method_rows: Sequence[Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
    position_rows: Sequence[Mapping[str, Any]],
    audit: Audit,
    common: Mapping[str, Any],
) -> str:
    lines = [
        "# LLaMA-1B isolated efficiency audit",
        "",
        f"Integrity status: **{'PASS' if passed else 'INVALID'}**.",
        "",
        (
            "This status certifies evidence integrity only. It is not an optimizer-"
            "quality or algorithm-superiority decision; experiment 20 remains the "
            "frozen quality source."
        ),
        "",
        "## Frozen design",
        "",
        (
            f"Four methods were run for {contract['execution_policy']['repeats']} "
            "rotated repeats on physical GPU 0 while both node GPUs were certified "
            "idle before and after every timed cell; a sealed process monitor also "
            "audited GPU 1 throughout each cell. The first "
            f"{contract['frozen_configuration']['warmup_updates_excluded']} updates "
            "were excluded and "
            f"{contract['frozen_configuration']['timed_updates']} updates were timed."
        ),
        "",
        "## Method-level measurements",
        "",
        "| Method | median ms/update | mean tokens/s | throughput CV | peak allocated MiB | peak reserved MiB | K-state MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            "| {method} | {step} | {tokens} | {cv}% | {allocated} | {reserved} | {kstate} |".format(
                method=row["method"],
                step=format_number(row["step_avg_ms_median"]),
                tokens=format_number(row["tokens_per_s_mean"], 0),
                cv=format_number(row["tokens_per_s_cv_pct"], 2),
                allocated=format_number(
                    row["timed_training_peak_allocated_mib_median"], 1
                ),
                reserved=format_number(
                    row["timed_training_peak_reserved_mib_median"], 1
                ),
                kstate=format_number(row["k_state_mib_exact"], 1),
            )
        )
    lines.extend(
        [
            "",
            "## Frozen primary contrasts",
            "",
            (
                "The table reports paired candidate-minus-reference changes across "
                "the four rotations. The ±1% band is descriptive only."
            ),
            "",
            "| Candidate | Reference | mean paired tokens/s change | 95% t CI | descriptive classification |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in contrast_rows:
        if row["metric"] != "tokens_per_s":
            continue
        lines.append(
            "| {candidate} | {reference} | {mean}% | [{lower}%, {upper}%] | {classification} |".format(
                candidate=row["candidate"],
                reference=row["reference"],
                mean=format_number(row["mean_relative_pct"], 2),
                lower=format_number(row["relative_ci95_lower_pct"], 2),
                upper=format_number(row["relative_ci95_upper_pct"], 2),
                classification=row["descriptive_classification"],
            )
        )
    lines.extend(
        [
            "",
            (
                "Selective-diag versus Selective-none is deliberately not a primary "
                "contrast and is not added to this table."
            ),
            "",
            "## Rotation-position diagnostic",
            "",
            "| Position | mean tokens/s | position versus overall |",
            "|---:|---:|---:|",
        ]
    )
    for row in position_rows:
        if row["metric"] != "tokens_per_s":
            continue
        lines.append(
            f"| {row['position_index']} | {format_number(row['mean'], 0)} | "
            f"{format_number(row['position_minus_overall_pct'], 2)}% |"
        )
    lines.extend(
        [
            "",
            "## Evidence fingerprints",
            "",
            f"- Contract SHA256: `{common.get('contract_sha256', 'NA')}`",
            f"- Preflight SHA256: `{common.get('preflight_sha256', 'NA')}`",
            (
                "- Execution manifest SHA256: "
                f"`{common.get('execution_manifest_sha256_actual', 'NA')}`"
            ),
            f"- Initialization SHA256: `{common.get('common_init_sha256', 'NA')}`",
            f"- Derived trainer SHA256: `{common.get('derived_base_sha256', 'NA')}`",
            f"- Profile wrapper SHA256: `{common.get('profile_wrapper_sha256', 'NA')}`",
            f"- Runtime fingerprint: `{common.get('runtime_fingerprint', 'NA')}`",
            f"- Data fingerprint: `{common.get('data_fingerprint', 'NA')}`",
            "",
            "## Integrity checks",
            "",
        ]
    )
    for name, value in audit.checks.items():
        lines.append(f"- {name}: `{'PASS' if value else 'FAIL'}`")
    if audit.errors:
        lines.extend(["", "### Integrity errors", ""])
        lines.extend(f"- {error}" for error in audit.errors)
    return "\n".join(lines) + "\n"


def analyze(run_dir: Path, contract_path: Path, output_dir: Path) -> bool:
    run_dir = run_dir.resolve(strict=True)
    contract_path = contract_path.resolve(strict=True)
    output_dir = output_dir.resolve() if output_dir.exists() else output_dir.absolute()
    contract = read_json(contract_path)
    contract_sha256 = sha256_file(contract_path)
    audit = Audit()

    audit.pass_if(
        "contract_identity",
        contract.get("experiment") == "42_llama1b_isolated_efficiency"
        and contract.get("schema_version") == 1
        and contract.get("contract_version") == "2026-07-29.3",
        "unexpected contract identity or schema",
    )
    methods = contract.get("method_order")
    roles = contract.get("method_roles")
    orders = contract.get("execution_policy", {}).get("orders")
    contrasts = contract.get("primary_contrasts")
    expected_k = contract.get("expected_k_state_bytes")
    frozen = contract.get("frozen_configuration", {})
    execution = contract.get("execution_policy", {})
    if not (
        isinstance(methods, list)
        and len(methods) == 4
        and len(set(methods)) == 4
        and isinstance(roles, dict)
        and isinstance(orders, list)
        and len(orders) == 4
        and isinstance(contrasts, list)
        and isinstance(expected_k, dict)
    ):
        audit.fail("contract_structure", "frozen method/order/contrast/K structure is malformed")
        methods = methods if isinstance(methods, list) else []
        roles = roles if isinstance(roles, dict) else {}
        orders = orders if isinstance(orders, list) else []
        contrasts = contrasts if isinstance(contrasts, list) else []
        expected_k = expected_k if isinstance(expected_k, dict) else {}
    else:
        audit.checks["contract_structure"] = True
    audit.pass_if(
        "frozen_method_and_primary_contrast_priority",
        methods == FROZEN_METHOD_ORDER
        and contrasts == FROZEN_PRIMARY_CONTRASTS
        and not any(
            {contrast.get("candidate"), contrast.get("reference")}
            == {"down_none", "down_diag"}
            for contrast in contrasts
            if isinstance(contrast, dict)
        ),
        (
            "method order or primary contrasts drifted; Selective-diag versus "
            "Selective-none must remain secondary"
        ),
    )

    expected_paths: dict[Path, tuple[int, int, str]] = {}
    if len(orders) == 4:
        for repeat_index, order in enumerate(orders):
            if order != [orders[repeat_index][index] for index in range(len(order))]:
                audit.fail("rotation_exact", f"repeat {repeat_index} order is malformed")
            if sorted(order) != sorted(methods):
                audit.fail(
                    "rotation_exact",
                    f"repeat {repeat_index} is not a permutation of method_order: {order}",
                )
            for position_index, method in enumerate(order):
                expected_paths[
                    run_dir / "formal" / f"repeat_{repeat_index}" / method / "cell_manifest.json"
                ] = (repeat_index, position_index, method)
        position_assignments = [
            [orders[repeat][position] for repeat in range(4)] for position in range(4)
        ]
        if not all(sorted(assignment) == sorted(methods) for assignment in position_assignments):
            audit.fail(
                "rotation_exact",
                f"each method does not occupy each position once: {position_assignments}",
            )
        else:
            audit.checks.setdefault("rotation_exact", True)

    observed_manifest_paths = (
        set((run_dir / "formal").rglob("cell_manifest.json"))
        if (run_dir / "formal").is_dir()
        else set()
    )
    audit.pass_if(
        "formal_cell_inventory",
        len(expected_paths) == int(contract.get("acceptance", {}).get("formal_rows", 16))
        and observed_manifest_paths == set(expected_paths),
        (
            f"expected={sorted(str(path.relative_to(run_dir)) for path in expected_paths)}, "
            f"observed={sorted(str(path.relative_to(run_dir)) for path in observed_manifest_paths)}"
        ),
    )
    preflight, preflight_sha256 = audit_preflight(
        run_dir, contract_path, contract, audit
    )
    execution_manifest, execution_manifest_sha256 = audit_execution_manifest(
        run_dir,
        contract_sha256,
        preflight_sha256,
        (
            preflight.get("data_audit", {}).get("fingerprint")
            if isinstance(preflight.get("data_audit"), dict)
            else None
        ),
        expected_paths,
        audit,
    )
    preflight_source = (
        preflight.get("source_audit")
        if isinstance(preflight.get("source_audit"), dict)
        else {}
    )
    preflight_data = (
        preflight.get("data_audit")
        if isinstance(preflight.get("data_audit"), dict)
        else {}
    )
    preflight_init = (
        preflight.get("init_audit")
        if isinstance(preflight.get("init_audit"), dict)
        else {}
    )

    raw_rows: list[dict[str, Any]] = []
    common_values: dict[str, list[Any]] = {
        "preflight_sha256": [],
        "runtime_fingerprint": [],
        "data_fingerprint": [],
        "common_init_sha256": [],
        "derived_base_sha256": [],
        "profile_wrapper_sha256": [],
    }
    for manifest_path, (repeat_index, position_index, method) in expected_paths.items():
        if not manifest_path.is_file():
            continue
        cell = audit.guard(
            "cell_manifests_readable",
            lambda path=manifest_path: read_json(path),
            str(manifest_path),
        )
        if cell is None:
            continue
        coordinate_ok = (
            cell.get("passed") is True
            and cell.get("script_version") == CONTROLLER_VERSION
            and cell.get("tier") == "formal"
            and cell.get("repeat_index") == repeat_index
            and cell.get("position_index") == position_index
            and cell.get("method") == method
            and cell.get("contract_sha256") == contract_sha256
            and cell.get("preflight_sha256") == preflight_sha256
            and cell.get("runtime_fingerprint")
            == preflight.get("runtime_fingerprint")
            and cell.get("data_fingerprint")
            == preflight_data.get("fingerprint")
            and cell.get("common_init_sha256")
            == preflight_init.get("common_init_sha256")
        )
        audit.pass_if(
            "cell_coordinates_and_contract",
            coordinate_ok,
            f"{manifest_path}: coordinate/pass/contract mismatch",
        )
        try:
            worker_path = resolve_evidence_path(
                run_dir, cell.get("worker_manifest"), "worker_manifest"
            )
            summary_path = resolve_evidence_path(
                run_dir, cell.get("trainer_summary"), "trainer_summary"
            )
            before_path = resolve_evidence_path(
                run_dir, cell.get("exclusive_before"), "exclusive_before"
            )
            after_path = resolve_evidence_path(
                run_dir, cell.get("exclusive_after"), "exclusive_after"
            )
            monitor_path = resolve_evidence_path(
                run_dir,
                cell.get("gpu_isolation_monitor"),
                "gpu_isolation_monitor",
            )
        except Exception as error:
            audit.fail("cell_evidence_paths", f"{manifest_path}: {error!r}")
            continue
        attempt_dir = worker_path.parent
        locality_ok = (
            summary_path.parent == attempt_dir / "trainer"
            and before_path.parent == attempt_dir
            and after_path.parent == attempt_dir
            and monitor_path.parent == attempt_dir
            and attempt_dir.parent == manifest_path.parent
            and attempt_dir.name.startswith("attempt_")
        )
        if cell.get("attempt_dir"):
            declared_attempt = Path(str(cell["attempt_dir"]))
            if not declared_attempt.is_absolute():
                declared_attempt = run_dir / declared_attempt
            locality_ok = locality_ok and declared_attempt.resolve() == attempt_dir
        audit.pass_if(
            "cell_evidence_locality",
            locality_ok,
            f"{manifest_path}: evidence is not sealed inside one cell attempt",
        )
        hashes_ok = all(
            [
                require_hash(
                    audit,
                    "cell_evidence_hashes",
                    worker_path,
                    cell.get("worker_manifest_sha256"),
                    f"{manifest_path}: worker manifest",
                ),
                require_hash(
                    audit,
                    "cell_evidence_hashes",
                    summary_path,
                    cell.get("trainer_summary_sha256"),
                    f"{manifest_path}: trainer summary",
                ),
                require_hash(
                    audit,
                    "cell_evidence_hashes",
                    before_path,
                    cell.get("exclusive_before_sha256"),
                    f"{manifest_path}: before certificate",
                ),
                require_hash(
                    audit,
                    "cell_evidence_hashes",
                    after_path,
                    cell.get("exclusive_after_sha256"),
                    f"{manifest_path}: after certificate",
                ),
                require_hash(
                    audit,
                    "cell_evidence_hashes",
                    monitor_path,
                    cell.get("gpu_isolation_monitor_sha256"),
                    f"{manifest_path}: GPU isolation monitor",
                ),
            ]
        )
        if not hashes_ok:
            continue
        worker = audit.guard(
            "worker_manifests_readable", lambda path=worker_path: read_json(path), str(worker_path)
        )
        summary = audit.guard(
            "trainer_summaries_readable", lambda path=summary_path: read_json(path), str(summary_path)
        )
        before = audit.guard(
            "exclusive_certificates_readable",
            lambda path=before_path: read_json(path),
            str(before_path),
        )
        after = audit.guard(
            "exclusive_certificates_readable",
            lambda path=after_path: read_json(path),
            str(after_path),
        )
        monitor = audit.guard(
            "gpu_isolation_monitors_readable",
            lambda path=monitor_path: read_json(path),
            str(monitor_path),
        )
        if None in (worker, summary, before, after, monitor):
            continue
        metrics_path = attempt_dir / "trainer" / "metrics.csv"
        terminal_log_path = attempt_dir / "terminal.log"
        trainer_base_path = (
            attempt_dir / "trainer" / "train_llama_swiglu_base.py"
        )
        worker_artifacts_ok = all(
            [
                require_hash(
                    audit,
                    "worker_sealed_artifact_hashes",
                    metrics_path,
                    worker.get("metrics_sha256"),
                    f"{worker_path}: metrics.csv",
                ),
                require_hash(
                    audit,
                    "worker_sealed_artifact_hashes",
                    terminal_log_path,
                    worker.get("terminal_log_sha256"),
                    f"{worker_path}: terminal.log",
                ),
                require_hash(
                    audit,
                    "worker_sealed_artifact_hashes",
                    trainer_base_path,
                    worker.get("trainer_local_base_sha256"),
                    f"{worker_path}: trainer-local base source",
                ),
            ]
        )

        worker_ok = (
            worker.get("passed") is True
            and worker.get("script_version") == WORKER_VERSION
            and worker.get("tier") == "formal"
            and worker.get("method") == method
            and worker.get("repeat_index") == repeat_index
            and worker.get("position_index") == position_index
            and worker.get("contract_sha256") == contract_sha256
            and worker.get("preflight_sha256") == cell.get("preflight_sha256")
            and worker.get("summary_sha256") == sha256_file(summary_path)
            and worker_artifacts_ok
            and worker.get("derived_base_sha256")
            == preflight_source.get("derived_base_sha256")
            and worker.get("profile_wrapper_sha256")
            == preflight_source.get("profile_wrapper_sha256")
            and worker.get("trainer_local_base_sha256")
            == preflight_source.get("derived_base_sha256")
            and cell.get("observed") == worker.get("observed")
        )
        audit.pass_if(
            "worker_manifest_integrity",
            worker_ok,
            f"{worker_path}: worker pass/coordinate/hash/source mismatch",
        )
        observed = worker.get("observed")
        if not isinstance(observed, dict):
            audit.fail("worker_observed_matches_summary", f"{worker_path}: observed is not an object")
            continue

        expected_summary_values: dict[str, Any] = {
            "status": "completed",
            "method": method,
            "seed": frozen.get("seed"),
            "completed_steps": frozen.get("total_updates"),
            "steady_steps": frozen.get("timed_updates"),
            "tokens_seen": frozen.get("total_updates")
            * frozen.get("tokens_per_update"),
            "resume_count": 0,
            "timing_comparable": True,
            "checkpoint_path": "",
            "peak_memory_stats_reset": True,
            "peak_reset_after_completed_step": frozen.get(
                "warmup_updates_excluded"
            ),
            "timed_step_first": frozen.get("warmup_updates_excluded") + 1,
            "timed_step_last": frozen.get("total_updates"),
        }
        summary_ok = all(summary.get(key) == value for key, value in expected_summary_values.items())
        audit.pass_if(
            "frozen_measurement_counts_and_no_resume",
            summary_ok,
            (
                f"{summary_path}: expected frozen summary values "
                f"{expected_summary_values}, observed subset="
                f"{ {key: summary.get(key) for key in expected_summary_values} }"
            ),
        )
        summary_config = summary.get("config")
        expected_config = {
            "num_iterations": frozen.get("total_updates"),
            "global_batch_size": frozen.get("global_batch_size"),
            "device_batch_size": frozen.get("device_batch_size"),
            "sequence_length": frozen.get("sequence_length"),
            "val_every": frozen.get("validation_every"),
            "val_tokens": frozen.get("validation_tokens"),
            "warmdown_iters": frozen.get("warmdown_updates"),
            "backup_lr": frozen.get("backup_lr"),
            "matrix_lr": frozen.get("matrix_lr"),
            "adamw_matrix_lr": frozen.get("adamw_matrix_lr"),
            "checkpoint_every": frozen.get("checkpoint_every"),
            "resume": frozen.get("resume_policy"),
            "no_save_final": True,
        }
        audit.pass_if(
            "frozen_trainer_configuration",
            isinstance(summary_config, dict)
            and all(summary_config.get(key) == value for key, value in expected_config.items()),
            (
                f"{summary_path}: frozen trainer configuration mismatch; "
                f"expected={expected_config}"
            ),
        )
        architecture = summary.get("architecture")
        audit.pass_if(
            "frozen_architecture",
            isinstance(architecture, dict)
            and architecture.get("parameter_count") == frozen.get("parameter_count")
            and architecture.get("base_trainer_sha256")
            == worker.get("derived_base_sha256"),
            f"{summary_path}: parameter count or derived trainer fingerprint mismatch",
        )
        runtime_hash = audit.guard(
            "summary_runtime_fingerprint",
            lambda payload=summary.get("runtime"): stable_runtime_fingerprint(payload),
            str(summary_path),
        )
        audit.pass_if(
            "summary_runtime_fingerprint",
            runtime_hash is not None
            and runtime_hash == cell.get("runtime_fingerprint")
            and runtime_hash == observed.get("runtime_fingerprint"),
            f"{summary_path}: stable runtime fingerprint mismatch",
        )
        metric_keys = [
            "steady_train_s",
            "steady_steps",
            "step_avg_ms",
            "timed_training_peak_allocated_bytes",
            "timed_training_peak_reserved_bytes",
            "k_state_bytes",
            "optimizer_state_bytes",
            "model_parameter_bytes",
            "resume_count",
            "completed_steps",
            "tokens_seen",
        ]
        observed_matches = all(
            key in observed
            and key in summary
            and (
                observed[key] == summary[key]
                if isinstance(summary[key], (str, bool))
                else numeric_equal(observed[key], summary[key])
            )
            for key in metric_keys
        )
        observed_matches = (
            observed_matches
            and observed.get("init_sha256") == summary.get("init_sha256")
            and observed.get("runtime_fingerprint") == cell.get("runtime_fingerprint")
            and observed.get("data_fingerprint") == cell.get("data_fingerprint")
            and observed.get("timing_comparable") is True
        )
        audit.pass_if(
            "worker_observed_matches_summary",
            observed_matches,
            f"{worker_path}: observed measurements/fingerprints differ from evidence",
        )
        metric_audit = worker.get("metric_audit")
        expected_metric_audit = {
            "train_rows": frozen.get("total_updates"),
            "validation_rows": 2,
            "timed_rows": frozen.get("timed_updates"),
            "timed_step_first": frozen.get("warmup_updates_excluded") + 1,
            "timed_step_last": frozen.get("total_updates"),
        }
        audit.pass_if(
            "worker_metric_sequence_audit",
            isinstance(metric_audit, dict)
            and all(
                metric_audit.get(key) == value
                for key, value in expected_metric_audit.items()
            )
            and numeric_equal(
                metric_audit.get("timed_interval_sum_s"),
                summary.get("steady_train_s"),
            ),
            (
                f"{worker_path}: sealed metric-sequence audit differs from "
                f"{expected_metric_audit}"
            ),
        )
        try:
            steady_train_s = finite_float(summary["steady_train_s"])
            steady_steps = exact_int(summary["steady_steps"])
            step_avg_ms = finite_float(summary["step_avg_ms"])
            peak_allocated = exact_int(
                summary["timed_training_peak_allocated_bytes"]
            )
            peak_reserved = exact_int(
                summary["timed_training_peak_reserved_bytes"]
            )
            k_state = exact_int(summary["k_state_bytes"])
            optimizer_state = exact_int(summary["optimizer_state_bytes"])
            model_bytes = exact_int(summary["model_parameter_bytes"])
            finite_ok = (
                steady_train_s > 0
                and steady_steps > 0
                and step_avg_ms > 0
                and peak_allocated > 0
                and peak_reserved >= peak_allocated
                and optimizer_state >= 0
                and model_bytes > 0
                and math.isclose(
                    step_avg_ms,
                    1000.0 * steady_train_s / steady_steps,
                    rel_tol=1e-9,
                    abs_tol=1e-7,
                )
            )
            boundary_values = [
                exact_int(summary[key])
                for key in [
                    "allocated_bytes_at_timing_reset",
                    "reserved_bytes_at_timing_reset",
                    "allocated_bytes_at_timed_end",
                    "reserved_bytes_at_timed_end",
                ]
            ]
            finite_ok = (
                finite_ok
                and all(value >= 0 for value in boundary_values)
                and boundary_values[0] <= peak_allocated
                and boundary_values[1] <= peak_reserved
                and boundary_values[2] <= peak_allocated
                and boundary_values[3] <= peak_reserved
            )
        except (KeyError, TypeError, ValueError) as error:
            finite_ok = False
            k_state = -1
            audit.fail("finite_timing_and_memory", f"{summary_path}: {error!r}")
        audit.pass_if(
            "finite_timing_and_memory",
            finite_ok,
            f"{summary_path}: timing/memory values are invalid or internally inconsistent",
        )
        audit.pass_if(
            "exact_k_state_bytes",
            k_state == expected_k.get(method),
            f"{summary_path}: expected K={expected_k.get(method)}, observed={k_state}",
        )
        expected_state = contract.get("expected_state_bytes", {})
        audit.pass_if(
            "exact_model_and_optimizer_state_bytes",
            model_bytes == expected_state.get("model_parameter_bytes")
            and optimizer_state
            == expected_state.get("optimizer_state_bytes", {}).get(method),
            (
                f"{summary_path}: expected model/optimizer bytes="
                f"{expected_state.get('model_parameter_bytes')}/"
                f"{expected_state.get('optimizer_state_bytes', {}).get(method)}, "
                f"observed={model_bytes}/{optimizer_state}"
            ),
        )
        init_sha256 = summary.get("init_sha256")
        audit.pass_if(
            "initialization_hash_format",
            is_sha256(init_sha256)
            and init_sha256 == cell.get("common_init_sha256")
            and init_sha256 == observed.get("init_sha256")
            and init_sha256 == frozen.get("initialization_sha256"),
            f"{summary_path}: initialization fingerprint mismatch",
        )
        wrapper_expected = contract.get("source_contract", {}).get(
            "profile_wrapper_sha256"
        )
        audit.pass_if(
            "source_hash_pins",
            worker.get("profile_wrapper_sha256") == wrapper_expected
            and worker.get("derived_base_sha256")
            == contract.get("source_contract", {}).get(
                "derived_efficiency_base_sha256"
            ),
            f"{worker_path}: source hashes do not satisfy the frozen contract",
        )

        certificate_results: list[tuple[bool, float, list[str]]] = []
        for certificate in (before, after):
            certificate_results.append(
                validate_certificate(
                    certificate,
                    execution.get("required_idle_physical_gpus", []),
                    execution.get("required_gpu_name", ""),
                )
            )
        certificates_ok = all(result[0] for result in certificate_results)
        audit.pass_if(
            "exclusive_certificates",
            certificates_ok,
            (
                f"{manifest_path}: "
                f"{certificate_results[0][2] + certificate_results[1][2]}"
            ),
        )
        timing_gpu = int(execution.get("physical_timing_gpu"))
        idle_gpus = sorted(
            int(index)
            for index in execution.get("required_idle_physical_gpus", [])
            if int(index) != timing_gpu
        )
        monitor_ok, monitor_errors = validate_gpu_isolation_monitor(
            monitor,
            timing_gpu=timing_gpu,
            idle_gpus=idle_gpus,
            interval_seconds=finite_float(
                execution.get("continuous_gpu_process_monitor_interval_seconds")
            ),
        )
        audit.pass_if(
            "continuous_gpu_isolation_monitors",
            monitor_ok,
            f"{monitor_path}: {monitor_errors}",
        )
        before_uuid_map = certificate_gpu_uuid_map(before)
        after_uuid_map = certificate_gpu_uuid_map(after)
        monitor_uuid_map = monitor.get("gpu_index_to_uuid")
        required_uuid_keys = {
            str(int(index))
            for index in execution.get("required_idle_physical_gpus", [])
        }
        audit.pass_if(
            "gpu_identity_stable_before_during_after",
            before_uuid_map == after_uuid_map == monitor_uuid_map
            and set(before_uuid_map) == required_uuid_keys,
            (
                f"{manifest_path}: before={before_uuid_map}, "
                f"monitor={monitor_uuid_map}, after={after_uuid_map}"
            ),
        )
        try:
            row = metric_row(
                repeat_index=repeat_index,
                position_index=position_index,
                method=method,
                role=roles[method],
                cell=cell,
                worker=worker,
                summary=summary,
                before_minimum_free=certificate_results[0][1],
                after_minimum_free=certificate_results[1][1],
                worker_path=worker_path,
                summary_path=summary_path,
                run_dir=run_dir,
                monitor=monitor,
                tokens_per_update=exact_int(frozen["tokens_per_update"]),
            )
        except Exception as error:
            audit.fail("raw_measurement_extraction", f"{summary_path}: {error!r}")
            continue
        audit.pass_if(
            "worker_derived_throughput_matches",
            numeric_equal(observed.get("steps_per_s"), row["steps_per_s"])
            and numeric_equal(observed.get("tokens_per_s"), row["tokens_per_s"]),
            f"{worker_path}: derived steps/s or tokens/s mismatch",
        )
        raw_rows.append(row)
        for key in common_values:
            if key == "derived_base_sha256":
                value = worker.get(key)
            elif key == "profile_wrapper_sha256":
                value = worker.get(key)
            elif key == "common_init_sha256":
                value = cell.get(key)
            else:
                value = cell.get(key)
            common_values[key].append(value)

    audit.pass_if(
        "formal_rows_complete",
        len(raw_rows) == int(contract.get("acceptance", {}).get("formal_rows", 16)),
        f"expected 16 validated rows, observed {len(raw_rows)}",
    )
    if raw_rows:
        coordinate_set = {
            (row["repeat_index"], row["position_index"], row["method"]) for row in raw_rows
        }
        expected_coordinate_set = {
            (repeat_index, position_index, method)
            for repeat_index, order in enumerate(orders)
            for position_index, method in enumerate(order)
        }
        audit.pass_if(
            "rotation_observed_exact",
            coordinate_set == expected_coordinate_set,
            f"observed coordinates differ from contract: {coordinate_set}",
        )
    else:
        audit.fail("rotation_observed_exact", "no validated rows")

    common: dict[str, Any] = {
        "contract_sha256": contract_sha256,
        "preflight_sha256_actual": preflight_sha256,
        "execution_manifest_sha256_actual": execution_manifest_sha256,
        "source_snapshot_files": preflight_source.get("files"),
    }
    for key, values in common_values.items():
        normalized: list[str] = []
        for value in values:
            try:
                normalized.append(fingerprint_key(value))
            except RuntimeError:
                normalized.append("")
        check_name = {
            "preflight_sha256": "common_preflight_fingerprint",
            "runtime_fingerprint": "common_runtime_fingerprint",
            "data_fingerprint": "common_data_fingerprint",
            "common_init_sha256": "common_initialization_fingerprint",
            "derived_base_sha256": "common_derived_source_fingerprint",
            "profile_wrapper_sha256": "common_wrapper_source_fingerprint",
        }[key]
        equal_nonempty = (
            len(values) == 16
            and bool(normalized)
            and all(value == normalized[0] and value for value in normalized)
        )
        audit.pass_if(
            check_name,
            equal_nonempty,
            f"{key} is empty or differs across cells",
        )
        common[key] = values[0] if equal_nonempty else None
    audit.pass_if(
        "preflight_hash_format",
        is_sha256(common.get("preflight_sha256")),
        f"malformed preflight SHA256: {common.get('preflight_sha256')!r}",
    )
    audit.pass_if(
        "initialization_hash_common_and_valid",
        is_sha256(common.get("common_init_sha256"))
        and common.get("common_init_sha256")
        == frozen.get("initialization_sha256"),
        f"malformed initialization SHA256: {common.get('common_init_sha256')!r}",
    )

    checks_before_overall = dict(audit.checks)
    passed = bool(checks_before_overall) and all(checks_before_overall.values()) and not audit.errors
    audit.checks["all_integrity_checks_passed"] = passed
    audit.details = {
        "expected_formal_rows": contract.get("acceptance", {}).get("formal_rows"),
        "validated_formal_rows": len(raw_rows),
        "expected_orders": orders,
        "observed_coordinates": [
            [row["repeat_index"], row["position_index"], row["method"]]
            for row in raw_rows
        ],
        "common_fingerprints": common,
        "preflight_status": preflight.get("passed"),
        "execution_manifest_status": execution_manifest.get("status"),
        "execution_manifest_sha256": execution_manifest_sha256,
        "t_critical_df3_95": T_CRITICAL_DF3_95,
        "throughput_practical_band_pct": contract.get("measurement_policy", {}).get(
            "throughput_practical_band_pct"
        ),
        "pass_meaning": (
            "integrity only; not optimizer quality or algorithm superiority"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    method_rows: list[dict[str, Any]] = []
    method_fields: list[str] = []
    contrast_rows: list[dict[str, Any]] = []
    contrast_fields: list[str] = []
    position_rows: list[dict[str, Any]] = []
    position_fields: list[str] = []
    if len(raw_rows) == 16:
        raw_rows.sort(key=lambda row: (row["repeat_index"], row["position_index"]))
        method_rows, method_fields = build_method_summary(raw_rows, methods, roles)
        contrast_rows, contrast_fields = build_paired_contrasts(
            raw_rows,
            contrasts,
            finite_float(
                contract.get("measurement_policy", {}).get(
                    "throughput_practical_band_pct"
                )
            ),
        )
        position_rows, position_fields = build_position_effects(raw_rows)
        write_csv(output_dir / "raw_runs.csv", raw_rows, RAW_FIELDS)
        write_csv(output_dir / "method_summary.csv", method_rows, method_fields)
        write_csv(
            output_dir / "paired_contrasts.csv", contrast_rows, contrast_fields
        )
        write_csv(
            output_dir / "position_effects.csv", position_rows, position_fields
        )

    integrity_payload = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "passed": passed,
        "checks": audit.checks,
        "errors": audit.errors,
        "details": audit.details,
    }
    atomic_write_json(output_dir / "integrity_checks.json", integrity_payload)
    atomic_write_text(
        output_dir / "report.md",
        build_report(
            passed=passed,
            contract=contract,
            method_rows=method_rows,
            contrast_rows=contrast_rows,
            position_rows=position_rows,
            audit=audit,
            common=common,
        ),
    )
    artifact_names = [
        name
        for name in [
            "raw_runs.csv",
            "method_summary.csv",
            "paired_contrasts.csv",
            "position_effects.csv",
            "integrity_checks.json",
            "report.md",
        ]
        if (output_dir / name).is_file()
    ]
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "invalid",
        "passed": passed,
        "analysis_tier": "formal",
        "experiment": contract.get("experiment"),
        "evidence_class": contract.get("evidence_class"),
        "run_dir": str(run_dir),
        "contract": str(contract_path),
        "contract_sha256": contract_sha256,
        "formal_rows": len(raw_rows),
        "method_rows": len(method_rows),
        "paired_contrast_rows": len(contrast_rows),
        "position_effect_rows": len(position_rows),
        "method_order": methods,
        "primary_contrasts": contrasts,
        "common_fingerprints": common,
        "analysis_pass_meaning": (
            "integrity only; not optimizer quality or algorithm superiority"
        ),
        "artifacts": [
            {
                "path": name,
                "sha256": sha256_file(output_dir / name),
            }
            for name in artifact_names
        ],
    }
    atomic_write_json(
        output_dir / "llama1b_efficiency_analysis_manifest.json", manifest
    )
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "analysis"
    passed = analyze(args.run_dir, args.contract, output_dir)
    print(f"LLaMA-1B efficiency analysis artifacts: {output_dir}")
    print(
        "LLaMA-1B efficiency analysis manifest: "
        f"{output_dir / 'llama1b_efficiency_analysis_manifest.json'}"
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
