"""Shared, dependency-free utilities for the local submission-analysis pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MIB = 1024 * 1024
PRACTICAL_LOSS_MARGIN = 0.002

METHOD_ALIASES = {
    "newton_full": "original",
    "original_newton_muon": "original",
    "block4": "original",
    "original": "original",
    "selective_diag": "diag",
    "down_diag": "diag",
    "diag": "diag",
    "selective_none": "none",
    "down_none": "none",
    "none": "none",
    "muon": "muon",
    "mousse": "mousse",
    "moonlight": "moonlight",
    "normuon": "normuon",
    "adamw": "adamw",
}


class ContractError(RuntimeError):
    """Raised when an input violates the frozen analysis contract."""


def canonical_method(value: str) -> str:
    key = value.strip().lower()
    if key not in METHOD_ALIASES:
        raise ContractError(f"unknown method label: {value!r}")
    return METHOD_ALIASES[key]


def ensure_new_output(output_dir: Path, manifest_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    committed = output_dir / manifest_name
    if committed.exists():
        raise ContractError(
            f"refusing to overwrite committed analysis: {committed}; use a new output directory"
        )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_cell(row.get(key, "")) for key in fieldnames})
    os.replace(temporary, path)


def format_cell(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input(anchor: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = anchor.parent / path
    return path.resolve()


def required_float(row: Mapping[str, str], field: str, context: str) -> float:
    raw = row.get(field, "")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{context}: missing/non-numeric {field}: {raw!r}") from exc
    if not math.isfinite(value):
        raise ContractError(f"{context}: non-finite {field}: {raw!r}")
    return value


def optional_float(row: Mapping[str, str], field: str) -> float:
    raw = row.get(field, "")
    if raw is None or str(raw).strip() == "":
        return math.nan
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def bool_cell(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "eligible", "passed"}


def mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else math.nan


def sample_sd(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.stdev(finite) if len(finite) >= 2 else math.nan


T_975 = {
    1: 12.706204736,
    2: 4.302652730,
    3: 3.182446305,
    4: 2.776445105,
    5: 2.570581836,
    6: 2.446911851,
    7: 2.364624252,
    8: 2.306004135,
    9: 2.262157163,
    10: 2.228138852,
    11: 2.200985160,
    12: 2.178812830,
    13: 2.160368656,
    14: 2.144786688,
    15: 2.131449546,
    16: 2.119905299,
    17: 2.109815578,
    18: 2.100922040,
    19: 2.093024054,
    20: 2.085963447,
    24: 2.063898562,
    29: 2.045229642,
    39: 2.022690920,
    59: 2.000995378,
    119: 1.980099876,
}


def t_critical_975(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        return math.nan
    if degrees_of_freedom in T_975:
        return T_975[degrees_of_freedom]
    larger = sorted(key for key in T_975 if key >= degrees_of_freedom)
    return T_975[larger[0]] if larger else 1.959963985


def mean_ci95(values: Sequence[float]) -> tuple[float, float, float, float]:
    center = mean(values)
    sd = sample_sd(values)
    if len(values) < 2:
        return center, sd, math.nan, math.nan
    half_width = t_critical_975(len(values) - 1) * sd / math.sqrt(len(values))
    return center, sd, center - half_width, center + half_width


def manifest_value(document: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = document
    for key in dotted_key.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ContractError(f"manifest is missing required key: {dotted_key}")
        value = value[key]
    return value


def validate_manifest_requirements(
    document: Mapping[str, Any], requirements: Mapping[str, Any], context: str
) -> None:
    for key, expected in requirements.items():
        observed = manifest_value(document, key)
        if observed != expected:
            raise ContractError(
                f"{context}: manifest requirement {key!r} expected {expected!r}, got {observed!r}"
            )


def commit_manifest(
    output_dir: Path,
    manifest_name: str,
    payload: dict[str, Any],
    output_names: Sequence[str],
) -> Path:
    payload = dict(payload)
    payload["mdp_manifest"] = True
    payload["outputs"] = {
        name: sha256_file(output_dir / name) for name in sorted(output_names)
    }
    manifest_path = output_dir / manifest_name
    write_json(manifest_path, payload)
    return manifest_path

