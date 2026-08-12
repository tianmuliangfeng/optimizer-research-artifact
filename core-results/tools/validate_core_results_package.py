#!/usr/bin/env python3
"""Validate a relocatable ``core-results`` evidence package.

The validator deliberately uses only the Python 3.10 standard library.  It is
also copied into every built package as
``tools/validate_core_results_package.py`` so that, after moving or unpacking
the directory, the following remains sufficient::

    python tools/validate_core_results_package.py .

The validation contract is fail-closed: every regular file must be registered,
all recorded hashes must match, JSON/CSV files must parse, package references
must be package-root-relative, and private absolute paths are forbidden.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import posixpath
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
REQUIRED_STATUS_FIELDS = (
    "integrity_status",
    "scientific_status",
    "claim_eligibility",
    "paper_role",
)
GENERATED_FILES = {
    "README.md",
    "artifact_manifest.json",
    "evidence_index.csv",
    "provenance/release_selection.json",
    "provenance/omission_ledger.json",
    "release_manifest.json",
    "SHA256SUMS",
    "tools/validate_core_results_package.py",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".svg",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_FILE_SUFFIXES = {
    ".ckpt",
    ".gz",
    ".log",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tgz",
    ".zip",
}
FORBIDDEN_COMPONENTS = {
    ".cache",
    ".wandb",
    "__pycache__",
    "cache",
    "checkpoints",
    "logs",
    "wandb",
}

# Written so this source file does not itself contain a private absolute path.
WINDOWS_ABSOLUTE_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
UNC_RE = re.compile(r"(?<![\\])\\\\[^\\\s]+\\[^\\\s]+")
PRIVATE_POSIX_RE = re.compile(
    "/" + r"(?:data|home|Users|mnt|workspace|root)(?:/|\\)", re.IGNORECASE
)
PATH_KEY_RE = re.compile(
    r"(?:^|_)(?:artifact|checksums?|csv|destination|dir|directory|file|"
    r"manifest|package|path|report|source|validator)(?:$|_)",
    re.IGNORECASE,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LEGACY_PROJECT_TOKEN = "selective-newton-" + "muon-main-conference"
SUBMISSION_ANONYMITY_PROFILE = "double_blind_v1"
ANONYMIZED_WANDB_PREFIX = "ANONYMIZED_WANDB_"
ANONYMIZED_HOST_PREFIX = "ANONYMIZED_CONTAINER_HOST_"
PUBLIC_WANDB_RE = re.compile(r"https?://(?:api\.)?wandb\.ai(?:/|\b)", re.IGNORECASE)
WANDB_ENTITY_RE = re.compile(
    r"(?i)(?:\bwandb[_-]?entity\b|[\"']entity[\"']\s*[:=])"
    r"(?!\s*[\"']?ANONYMIZED_WANDB_ENTITY_)"
)
PRIVATE_CONTAINER_HOST_RE = re.compile(r"\bapp-[a-z0-9][a-z0-9-]{18,}\b", re.IGNORECASE)
PRIVATE_HOST_ASSIGNMENT_RE = re.compile(
    r"(?i)[\"'](?:hostname|host_name|container_hostname)[\"']\s*[:=]"
    r"(?!\s*[\"']?ANONYMIZED_CONTAINER_HOST_)"
)
EX48_EXPERIMENT_ID = "48_llama1b_10b_multibudget"
EX48_GATE_ROLES = (
    "analysis_manifest",
    "generic_verify",
    "native_verify",
    "source_data_resume_lineage",
)


class ValidationError(RuntimeError):
    """Raised when a package violates the release contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    def no_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"duplicate JSON key in {path}: {key!r}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=no_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON {path}: {exc}") from exc


def _is_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES or path.name == "SHA256SUMS":
        return True
    try:
        sample = path.read_bytes()[:8192]
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} must be a non-empty string")
    if WINDOWS_ABSOLUTE_RE.search(value) or UNC_RE.search(value):
        raise ValidationError(f"{label} is an absolute Windows path: {value!r}")
    normalized = value.replace("\\", "/")
    if normalized.startswith("/"):
        raise ValidationError(f"{label} is an absolute path: {value!r}")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValidationError(f"{label} is not a canonical package-relative path: {value!r}")
    if normalized != pure.as_posix():
        raise ValidationError(f"{label} must use canonical '/' separators: {value!r}")
    return normalized


def _iter_regular_files(root: Path) -> tuple[set[str], list[str]]:
    files: set[str] = set()
    empty_dirs: list[str] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            candidate = current_path / name
            if candidate.is_symlink():
                raise ValidationError(f"symlinked directory is forbidden: {candidate}")
        if current_path != root and not directories and not filenames:
            empty_dirs.append(current_path.relative_to(root).as_posix())
        for name in filenames:
            candidate = current_path / name
            if candidate.is_symlink():
                raise ValidationError(f"symlinked file is forbidden: {candidate}")
            if not candidate.is_file():
                raise ValidationError(f"non-regular package entry: {candidate}")
            files.add(candidate.relative_to(root).as_posix())
    return files, empty_dirs


def _check_forbidden_path(relative: str) -> None:
    pure = PurePosixPath(relative)
    lowered_parts = [part.lower() for part in pure.parts]
    if any(part in FORBIDDEN_COMPONENTS for part in lowered_parts):
        raise ValidationError(f"raw/cache directory is forbidden: {relative}")
    lowered_name = pure.name.lower()
    if pure.suffix.lower() in FORBIDDEN_FILE_SUFFIXES:
        raise ValidationError(f"archive/checkpoint/log file is forbidden: {relative}")
    if "wandb_export" in lowered_name:
        raise ValidationError(f"raw W&B export is forbidden: {relative}")


def _scan_private_paths(path: Path, relative: str) -> None:
    if not _is_text(path):
        return
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise ValidationError(f"declared text file is not UTF-8: {relative}: {exc}") from exc
    findings: list[str] = []
    if WINDOWS_ABSOLUTE_RE.search(text):
        findings.append("Windows drive path")
    if UNC_RE.search(text):
        findings.append("UNC path")
    if PRIVATE_POSIX_RE.search(text):
        findings.append("private POSIX path")
    if LEGACY_PROJECT_TOKEN.lower() in text.lower():
        findings.append("legacy private project token")
    if PUBLIC_WANDB_RE.search(text):
        findings.append("public W&B URL")
    # The packaged validator necessarily carries the detector expression itself.
    # Scan all evidence text, while excluding only this exact generated tool copy.
    if relative != "tools/validate_core_results_package.py" and WANDB_ENTITY_RE.search(text):
        findings.append("non-anonymous W&B entity")
    if PRIVATE_CONTAINER_HOST_RE.search(text) or PRIVATE_HOST_ASSIGNMENT_RE.search(text):
        findings.append("private container hostname")
    if findings:
        raise ValidationError(f"private absolute path in {relative}: {', '.join(findings)}")


def _looks_like_reference(value: str) -> bool:
    if not value or value == "PRIVATE_PATH_REDACTED":
        return False
    if "://" in value or value.startswith("doi:"):
        return False
    if any(character.isspace() for character in value):
        return False
    return "/" in value or "\\" in value or bool(Path(value).suffix)


def _audit_reference(value: str, label: str, root: Path, require_exists: bool) -> None:
    if not _looks_like_reference(value):
        return
    relative = _relative_path(value, label)
    if require_exists and not (root / Path(*PurePosixPath(relative).parts)).is_file():
        raise ValidationError(f"{label} points outside/unavailable in package: {relative}")


def _walk_json_references(value: Any, root: Path, label: str = "json") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_label = f"{label}.{key}"
            if isinstance(child, str) and PATH_KEY_RE.search(str(key)):
                require_exists = str(key).lower() in {
                    "artifact_manifest",
                    "checksums",
                    "evidence_index",
                    "package_path",
                    "validator",
                }
                _audit_reference(child, child_label, root, require_exists)
            else:
                _walk_json_references(child, root, child_label)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_json_references(child, root, f"{label}[{index}]")


def _audit_csv(path: Path, relative: str, root: Path) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValidationError(f"CSV has no header: {relative}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValidationError(f"CSV has duplicate columns: {relative}")
            for row_index, row in enumerate(reader, start=2):
                if None in row:
                    raise ValidationError(f"CSV row has extra cells: {relative}:{row_index}")
                for key, value in row.items():
                    if value and PATH_KEY_RE.search(key):
                        require_exists = key.lower() in {
                            "artifact_manifest",
                            "checksums",
                            "evidence_index",
                            "package_path",
                            "validator",
                        }
                        _audit_reference(
                            value,
                            f"{relative}:{row_index}:{key}",
                            root,
                            require_exists,
                        )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValidationError(f"invalid CSV {relative}: {exc}") from exc


def _package_link_target(
    value: str,
    owner_relative: str,
    artifacts_by_path: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any]] | None:
    stripped = value.strip().replace("\\", "/")
    if not stripped or "://" in stripped or "PRIVATE_PATH_REDACTED" in stripped:
        return None
    candidates = [
        stripped.removeprefix("./"),
        posixpath.normpath(
            posixpath.join(posixpath.dirname(owner_relative), stripped)
        ).removeprefix("./"),
    ]
    for candidate in candidates:
        record = artifacts_by_path.get(candidate)
        if record is not None:
            return candidate, record
    return None


def _internal_path_hash_bases(path_key: str) -> tuple[str, ...]:
    normalized = path_key.lower()
    stem = re.sub(
        r"(?i)(?:_?(?:path|file|artifact|manifest|csv|report|source))$",
        "",
        normalized,
    ).rstrip("_")
    return tuple(dict.fromkeys(value for value in (normalized, stem) if value))


def _internal_byte_field(key: str) -> bool:
    normalized = key.lower()
    return normalized in {"bytes", "size_bytes", "source_bytes", "package_bytes"} or (
        normalized.endswith("_bytes")
        and not normalized.endswith("_sha256_bytes")
    )


def _internal_companion_hash_keys(
    path_key: str, hash_keys: Iterable[str], allow_generic: bool
) -> tuple[str, ...]:
    candidates: set[str] = set()
    for base in _internal_path_hash_bases(path_key):
        candidates.update(
            {
                f"{base}_sha256",
                f"{base}_hash",
                f"{base}_digest",
                f"{base}_expected_sha256",
                f"{base}_source_sha256",
                f"{base}_package_sha256",
            }
        )
    if allow_generic:
        candidates.update(
            {
                "sha256",
                "hash",
                "digest",
                "expected_sha256",
                "source_sha256",
                "package_sha256",
            }
        )
    return tuple(key for key in hash_keys if key.lower() in candidates)


def _internal_dual_hash_fields(
    path_key: str, hash_keys: Iterable[str], allow_generic: bool
) -> tuple[str, str]:
    if allow_generic:
        return "source_sha256", "package_sha256"
    named_hashes = [
        key
        for key in hash_keys
        if key.lower() not in {"sha256", "source_sha256", "package_sha256"}
    ]
    if named_hashes:
        base = re.sub(
            r"(?i)(?:_?(?:source|package)_sha256|_?sha256|_?hash|_?digest)$",
            "",
            named_hashes[0],
        )
    else:
        bases = _internal_path_hash_bases(path_key)
        base = bases[-1] if bases else "artifact"
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_") or "artifact"
    return f"{base}_source_sha256", f"{base}_package_sha256"


def _internal_companion_byte_keys(
    path_key: str, byte_keys: Iterable[str], allow_generic: bool
) -> tuple[str, ...]:
    candidates: set[str] = set()
    for base in _internal_path_hash_bases(path_key):
        candidates.update(
            {
                f"{base}_bytes",
                f"{base}_size_bytes",
                f"{base}_source_bytes",
                f"{base}_package_bytes",
            }
        )
    if allow_generic:
        candidates.update({"bytes", "size_bytes", "source_bytes", "package_bytes"})
    return tuple(key for key in byte_keys if key.lower() in candidates)


def _internal_dual_byte_fields(
    path_key: str, byte_keys: Iterable[str], allow_generic: bool
) -> tuple[str, str]:
    if allow_generic:
        return "source_bytes", "package_bytes"
    named_bytes = [
        key
        for key in byte_keys
        if key.lower() not in {"bytes", "size_bytes", "source_bytes", "package_bytes"}
    ]
    if named_bytes:
        base = re.sub(
            r"(?i)(?:_?(?:source|package)_bytes|_?size_bytes|_?bytes)$",
            "",
            named_bytes[0],
        )
    else:
        bases = _internal_path_hash_bases(path_key)
        base = bases[-1] if bases else "artifact"
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_") or "artifact"
    return f"{base}_source_bytes", f"{base}_package_bytes"


def _normalized_bytes(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        return int(value)
    return None


def _validate_hash_link_mapping(
    value: Any,
    owner_relative: str,
    artifacts_by_path: Mapping[str, Mapping[str, Any]],
    label: str,
) -> None:
    if isinstance(value, Mapping):
        hash_keys = [
            str(key)
            for key in value
            if isinstance(key, str)
            and ("sha256" in key.lower() or key.lower() in {"hash", "digest"})
        ]
        byte_keys = [
            str(key)
            for key in value
            if isinstance(key, str) and _internal_byte_field(key)
        ]
        internal_paths: list[tuple[str, str, Mapping[str, Any]]] = []
        for key, child in value.items():
            if not isinstance(key, str) or not isinstance(child, str) or not PATH_KEY_RE.search(key):
                continue
            target = _package_link_target(child, owner_relative, artifacts_by_path)
            if target is not None:
                target_path, target_record = target
                internal_paths.append((key, target_path, target_record))
        allow_generic = len(internal_paths) == 1
        for key, target_path, target_record in internal_paths:
            companion_keys = _internal_companion_hash_keys(
                key, hash_keys, allow_generic
            )
            if not companion_keys:
                continue
            source_sha256 = str(target_record["source_sha256"])
            package_sha256 = str(target_record["package_sha256"])
            source_field, package_field = _internal_dual_hash_fields(
                key, companion_keys, allow_generic
            )
            companion_byte_keys = _internal_companion_byte_keys(
                key, byte_keys, allow_generic
            )
            source_bytes_field, package_bytes_field = _internal_dual_byte_fields(
                key, companion_byte_keys, allow_generic
            )
            failures: list[str] = []
            if value[key].replace("\\", "/") != target_path:
                failures.append(f"path={value[key]!r}")
            if source_sha256 != package_sha256:
                if value.get(source_field) != source_sha256:
                    failures.append(source_field)
                if value.get(package_field) != package_sha256:
                    failures.append(package_field)
            for hash_key in companion_keys:
                digest = value.get(hash_key)
                if hash_key.lower() == source_field.lower() or hash_key.lower().endswith(
                    "_source_sha256"
                ):
                    expected = source_sha256
                else:
                    expected = package_sha256
                if digest != expected:
                    failures.append(f"{hash_key}={digest!r}")
            if companion_byte_keys:
                source_bytes = int(target_record["source_bytes"])
                package_bytes = int(target_record["package_bytes"])
                if _normalized_bytes(value.get(source_bytes_field)) != source_bytes:
                    failures.append(source_bytes_field)
                if _normalized_bytes(value.get(package_bytes_field)) != package_bytes:
                    failures.append(package_bytes_field)
                for byte_key in companion_byte_keys:
                    observed_bytes = _normalized_bytes(value.get(byte_key))
                    if (
                        byte_key.lower() == source_bytes_field.lower()
                        or byte_key.lower().endswith("_source_bytes")
                    ):
                        expected_bytes = source_bytes
                    else:
                        expected_bytes = package_bytes
                    if observed_bytes != expected_bytes:
                        failures.append(f"{byte_key}={value.get(byte_key)!r}")
            if failures:
                raise ValidationError(
                    f"internal package hash link mismatch at {label}.{key}: "
                    f"target={target_path} failures={failures}"
                )
        for key, child in value.items():
            _validate_hash_link_mapping(
                child,
                owner_relative,
                artifacts_by_path,
                f"{label}.{key}",
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_hash_link_mapping(
                child,
                owner_relative,
                artifacts_by_path,
                f"{label}[{index}]",
            )


SHA256_SIDECAR_RE = re.compile(
    r"(?m)^(?P<digest>[0-9a-fA-F]{64})[ \t]+\*?(?P<path>\S+)[^\r\n]*\r?$"
)


def _validate_internal_package_hash_links(
    root: Path,
    artifacts_by_path: Mapping[str, Mapping[str, Any]],
) -> None:
    for relative, _record in sorted(artifacts_by_path.items()):
        path = root / Path(*PurePosixPath(relative).parts)
        suffix = path.suffix.lower()
        if suffix == ".json":
            _validate_hash_link_mapping(
                _load_json(path), relative, artifacts_by_path, relative
            )
        elif suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle, delimiter=delimiter)
                    if reader.fieldnames is None:
                        raise ValidationError(f"CSV has no header: {relative}")
                    for row_index, row in enumerate(reader, start=2):
                        _validate_hash_link_mapping(
                            row,
                            relative,
                            artifacts_by_path,
                            f"{relative}:{row_index}",
                        )
            except (OSError, UnicodeError, csv.Error) as exc:
                raise ValidationError(f"invalid CSV {relative}: {exc}") from exc
        elif suffix == ".sha256":
            try:
                text = path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeError) as exc:
                raise ValidationError(f"invalid SHA-256 sidecar {relative}: {exc}") from exc
            for match in SHA256_SIDECAR_RE.finditer(text):
                target = _package_link_target(
                    match.group("path"), relative, artifacts_by_path
                )
                if target is None:
                    continue
                target_path, target_record = target
                if match.group("digest").lower() != target_record["package_sha256"]:
                    raise ValidationError(
                        f"SHA-256 sidecar uses source/stale hash in {relative}: "
                        f"target={target_path}"
                    )


def _read_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read SHA256SUMS: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValidationError(f"blank SHA256SUMS line: {line_number}")
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValidationError(f"malformed SHA256SUMS line {line_number}: {line!r}")
        digest, relative_value = match.groups()
        relative = _relative_path(relative_value, f"SHA256SUMS:{line_number}")
        if relative in result:
            raise ValidationError(f"duplicate SHA256SUMS path: {relative}")
        result[relative] = digest
    return result


def _integrity_status_is_accepted(value: Any) -> bool:
    return isinstance(value, str) and (
        value == "accepted" or value.startswith("accepted_")
    )


def _has_experiment_48(artifacts: list[Mapping[str, Any]], release: Mapping[str, Any]) -> bool:
    included = release.get("included_experiments", [])
    if not isinstance(included, list) or EX48_EXPERIMENT_ID not in included:
        return False
    return any(
        _integrity_status_is_accepted(record.get("integrity_status"))
        and isinstance(record.get("experiment_ids"), list)
        and EX48_EXPERIMENT_ID in record["experiment_ids"]
        for record in artifacts
    )


def _ex48_gate_paths(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValidationError("final release must contain ex48_final_gate")
    if set(value) != {"experiment_id", "artifacts"}:
        raise ValidationError(
            "ex48_final_gate must contain only experiment_id and artifact bindings"
        )
    if value.get("experiment_id") != EX48_EXPERIMENT_ID:
        raise ValidationError(
            f"ex48_final_gate.experiment_id must equal {EX48_EXPERIMENT_ID}"
        )
    bindings = value.get("artifacts")
    if not isinstance(bindings, Mapping) or set(bindings) != set(EX48_GATE_ROLES):
        raise ValidationError(
            f"ex48_final_gate.artifacts must bind exactly {list(EX48_GATE_ROLES)}"
        )
    paths = {
        role: _relative_path(bindings[role], f"ex48_final_gate.artifacts.{role}")
        for role in EX48_GATE_ROLES
    }
    if len(set(paths.values())) != len(paths):
        raise ValidationError("ex48_final_gate artifact bindings must be distinct")
    return paths


def _require_true_checks(
    payload: Mapping[str, Any], label: str, required: Iterable[str]
) -> Mapping[str, Any]:
    checks = payload.get("checks")
    if not isinstance(checks, Mapping):
        checks = payload.get("integrity_checks")
    if not isinstance(checks, Mapping):
        raise ValidationError(f"{label} lacks a checks mapping")
    missing_or_false = [key for key in required if checks.get(key) is not True]
    if missing_or_false:
        raise ValidationError(f"{label} failed required checks: {missing_or_false}")
    return checks


def _validate_ex48_certificate_payloads(
    payloads: Mapping[str, Any],
) -> None:
    if set(payloads) != set(EX48_GATE_ROLES):
        raise ValidationError("EX48 certificate payload set is incomplete")
    if not all(isinstance(payloads[role], Mapping) for role in EX48_GATE_ROLES):
        raise ValidationError("every EX48 gate artifact must contain a JSON object")

    analysis = payloads["analysis_manifest"]
    assert isinstance(analysis, Mapping)
    if (
        analysis.get("schema_version") != "ex48_analysis_manifest_v1"
        or analysis.get("passed") is not True
        or analysis.get("claim_eligible") is not True
        or analysis.get("formal_units") != 12
        or analysis.get("primary_endpoints") != 36
    ):
        raise ValidationError("EX48 analysis manifest does not certify 12 units/36 endpoints")
    for key in ("contract_sha256", "data_inventory_sha256"):
        if not isinstance(analysis.get(key), str) or not SHA256_RE.fullmatch(analysis[key]):
            raise ValidationError(f"EX48 analysis manifest has invalid {key}")
    if not isinstance(analysis.get("artifacts"), Mapping) or not analysis["artifacts"]:
        raise ValidationError("EX48 analysis manifest has no content-bound analysis artifacts")
    _require_true_checks(
        analysis,
        "EX48 analysis manifest",
        (
            "contract",
            "source_snapshot",
            "preflight",
            "pilot",
            "suite",
            "data",
            "unit_count",
            "phase_count",
            "endpoint_count",
            "retained_checkpoint_count",
            "lineage",
            "same_seed_initialization",
        ),
    )

    generic = payloads["generic_verify"]
    assert isinstance(generic, Mapping)
    lineage = generic.get("lineage")
    if (
        generic.get("schema_version") != "selective_newton_muon_archive_verify_v1"
        or generic.get("passed") is not True
        or not isinstance(lineage, Mapping)
        or lineage.get("experiment_id") != EX48_EXPERIMENT_ID
        or not isinstance(generic.get("json_file_count"), int)
        or generic["json_file_count"] <= 0
        or not isinstance(generic.get("source_snapshot_manifest_count"), int)
        or generic["source_snapshot_manifest_count"] <= 0
    ):
        raise ValidationError("EX48 generic verification certificate is not claim-bound")
    _require_true_checks(
        generic,
        "EX48 generic verification certificate",
        (
            "run_dir_within_results_root",
            "json_present",
            "all_json_parse",
            "source_snapshot_manifest_present",
            "source_snapshot_manifests",
            "sealed_snapshot_requirement_satisfied",
            "experiment_lineage_bound",
        ),
    )

    native = payloads["native_verify"]
    assert isinstance(native, Mapping)
    if native.get("passed") is not True or native.get("full_checkpoint_hash") is not True:
        raise ValidationError("EX48 native verification did not full-hash all endpoints")
    native_checks = native.get("checks")
    if (
        not isinstance(native_checks, Mapping)
        or not native_checks
        or any(value is not True for value in native_checks.values())
    ):
        raise ValidationError("EX48 native verification contains failed checks")
    _require_true_checks(
        native,
        "EX48 native verification",
        ("analysis_manifest", "analysis_artifacts", "handoff", "unit_count", "endpoint_count"),
    )

    resume = payloads["source_data_resume_lineage"]
    assert isinstance(resume, Mapping)
    retired = resume.get("retired_pilot_checkpoints")
    if (
        resume.get("schema_version") != "ex48_engineering_pilot_v1"
        or resume.get("passed") is not True
        or resume.get("planned_interrupt_return_code") != 75
        or resume.get("in_place_resume") is not True
        or resume.get("source_checkpoint_branch") is not True
        or resume.get("no_wrap") is not True
        or not isinstance(retired, list)
        or len(retired) != 2
    ):
        raise ValidationError("EX48 resume-lineage certificate is incomplete")
    for index, checkpoint in enumerate(retired):
        if (
            not isinstance(checkpoint, Mapping)
            or not isinstance(checkpoint.get("sha256"), str)
            or not SHA256_RE.fullmatch(checkpoint["sha256"])
            or not isinstance(checkpoint.get("bytes"), int)
            or checkpoint["bytes"] <= 0
        ):
            raise ValidationError(f"invalid EX48 retired checkpoint certificate {index}")


def _validate_ex48_gate(
    root: Path,
    value: Any,
    artifacts_by_path: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    paths = _ex48_gate_paths(value)
    payloads: dict[str, Any] = {}
    for role, relative in paths.items():
        record = artifacts_by_path.get(relative)
        if (
            record is None
            or not _integrity_status_is_accepted(record.get("integrity_status"))
            or not isinstance(record.get("experiment_ids"), list)
            or EX48_EXPERIMENT_ID not in record["experiment_ids"]
        ):
            raise ValidationError(
                f"EX48 gate role {role} is not bound to an accepted EX48 artifact: {relative}"
            )
        payloads[role] = _load_json(root / Path(*PurePosixPath(relative).parts))
    _validate_ex48_certificate_payloads(payloads)
    return set(paths.values())


def _validate_public_snapshots(
    root: Path,
    release: Mapping[str, Any],
    artifacts: list[Mapping[str, Any]],
    *,
    require_ex48_anchor: bool,
    required_ex48_artifacts: set[str],
) -> None:
    catalog_value = release.get("public_catalog")
    anchors_value = release.get("accepted_result_anchors")
    if catalog_value is None and anchors_value is None:
        if require_ex48_anchor:
            raise ValidationError("final release lacks public catalog/accepted-anchor snapshots")
        return
    catalog_relative = _relative_path(catalog_value, "release.public_catalog")
    anchors_relative = _relative_path(
        anchors_value, "release.accepted_result_anchors"
    )
    catalog_path = root / Path(*PurePosixPath(catalog_relative).parts)
    anchors_path = root / Path(*PurePosixPath(anchors_relative).parts)
    if not catalog_path.is_file() or not anchors_path.is_file():
        raise ValidationError("public snapshot references do not resolve inside the package")

    catalog = _load_json(catalog_path)
    catalog_rows = catalog.get("experiments") if isinstance(catalog, Mapping) else None
    if not isinstance(catalog_rows, list):
        raise ValidationError("public catalog snapshot lacks experiments list")
    catalog_ids: set[str] = set()
    for index, row in enumerate(catalog_rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("experiment_id"), str):
            raise ValidationError(f"invalid public catalog experiment row {index}")
        experiment_id = str(row["experiment_id"])
        if experiment_id in catalog_ids:
            raise ValidationError(f"duplicate public catalog experiment_id: {experiment_id}")
        catalog_ids.add(experiment_id)

    included_value = release.get("included_experiments", [])
    if not isinstance(included_value, list) or not all(
        isinstance(value, str) and value for value in included_value
    ):
        raise ValidationError("release.included_experiments must be a list of catalog IDs")
    included = set(included_value)
    unknown = included - catalog_ids
    if unknown:
        raise ValidationError(f"included experiments absent from public catalog: {sorted(unknown)}")
    artifact_ids: set[str] = set()
    for record in artifacts:
        values = record.get("experiment_ids", [])
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValidationError(
                f"artifact {record.get('evidence_id')} has invalid experiment_ids"
            )
        artifact_ids.update(value for value in values if value)
    if not artifact_ids.issubset(included):
        raise ValidationError(
            f"artifact experiment IDs missing from release index: {sorted(artifact_ids - included)}"
        )

    anchors = _load_json(anchors_path)
    anchor_rows = anchors.get("records") if isinstance(anchors, Mapping) else None
    if not isinstance(anchor_rows, list):
        raise ValidationError("accepted-result anchor snapshot lacks records list")
    accepted_ex48 = False
    ex48_artifacts = [
        record
        for record in artifacts
        if _integrity_status_is_accepted(record.get("integrity_status"))
        and isinstance(record.get("experiment_ids"), list)
        and EX48_EXPERIMENT_ID in record["experiment_ids"]
    ]
    for row_index, row in enumerate(anchor_rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("experiment_id"), str):
            raise ValidationError(f"invalid accepted-result anchor row {row_index}")
        anchor_values = row.get("anchors")
        if not isinstance(anchor_values, list):
            raise ValidationError(f"anchor row {row_index} lacks anchors list")
        for anchor_index, anchor in enumerate(anchor_values):
            if not isinstance(anchor, Mapping):
                raise ValidationError(f"invalid anchor {row_index}:{anchor_index}")
            _relative_path(anchor.get("path"), f"anchor[{row_index}][{anchor_index}].path")
            digest = anchor.get("sha256")
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise ValidationError(f"invalid anchor SHA-256 {row_index}:{anchor_index}")
        if (
            row.get("accepted") is True
            and row["experiment_id"] == EX48_EXPERIMENT_ID
        ):
            if not anchor_values:
                raise ValidationError(
                    "accepted-result EX48 row must contain at least one content-bound anchor"
                )
            anchored_paths: set[str] = set()
            for anchor_index, anchor in enumerate(anchor_values):
                anchor_path = _relative_path(
                    anchor.get("path"), f"anchor[{row_index}][{anchor_index}].path"
                )
                matches: list[tuple[Mapping[str, Any], str, str]] = []
                for record in ex48_artifacts:
                    package_path = str(record.get("package_path", ""))
                    source_relpath = str(record.get("source_relpath", ""))
                    if anchor_path == package_path:
                        matches.append((record, str(record.get("package_sha256", "")), "package"))
                    if anchor_path == source_relpath:
                        candidate = (record, str(record.get("source_sha256", "")), "source")
                        if not matches or candidate[:2] != matches[-1][:2]:
                            matches.append(candidate)
                if not matches:
                    raise ValidationError(
                        f"EX48 anchor does not resolve to an accepted artifact: {anchor_path}"
                    )
                if len(matches) != 1:
                    raise ValidationError(f"EX48 anchor path is ambiguous: {anchor_path}")
                record, expected_digest, path_semantics = matches[0]
                if anchor.get("sha256") != expected_digest:
                    raise ValidationError(
                        f"EX48 anchor hash does not match {path_semantics} artifact: {anchor_path}"
                    )
                anchored_paths.add(str(record["package_path"]))
            missing_gate_anchors = required_ex48_artifacts - anchored_paths
            if missing_gate_anchors:
                raise ValidationError(
                    f"EX48 gate artifacts lack accepted anchors: {sorted(missing_gate_anchors)}"
                )
            accepted_ex48 = True
    if require_ex48_anchor and not accepted_ex48:
        raise ValidationError(
            f"accepted-result anchors lack accepted=true {EX48_EXPERIMENT_ID}"
        )


def _validate_public_source_parity(
    root: Path,
    public_source_root: Path | str,
    release: Mapping[str, Any],
    artifacts_by_path: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind packaged public snapshots to the exact source tree used at release time."""

    source_root = Path(public_source_root).resolve()
    if not source_root.is_dir():
        raise ValidationError(f"public source root is not a directory: {source_root}")
    for role in ("public_catalog", "accepted_result_anchors"):
        package_relative = _relative_path(
            release.get(role), f"release_manifest.{role}"
        )
        record = artifacts_by_path.get(package_relative)
        if record is None:
            raise ValidationError(f"public snapshot is not an artifact: {package_relative}")
        source_relative = _relative_path(
            record.get("source_relpath"), f"{role}.source_relpath"
        )
        source_path = source_root / Path(*PurePosixPath(source_relative).parts)
        if not source_path.is_file() or source_path.is_symlink():
            raise ValidationError(f"public source snapshot is missing: {source_relative}")
        source_sha256 = _sha256(source_path)
        source_bytes = source_path.stat().st_size
        if (
            source_sha256 != record.get("source_sha256")
            or source_bytes != record.get("source_bytes")
        ):
            raise ValidationError(f"public source snapshot drifted during build: {source_relative}")
        package_path = root / Path(*PurePosixPath(package_relative).parts)
        if source_path.read_bytes() != package_path.read_bytes():
            raise ValidationError(
                f"public source snapshot is not byte-identical in package: {source_relative}"
            )


def _selection_pending_markers(value: Any, label: str = "selection") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_label = f"{label}.{key}"
            if "pending" in str(key).lower():
                findings.append(child_label)
            findings.extend(_selection_pending_markers(child, child_label))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_selection_pending_markers(child, f"{label}[{index}]"))
    elif isinstance(value, str) and re.search(
        r"(?i)pending.{0,80}(?:ex)?48|(?:ex)?48.{0,80}pending", value
    ):
        findings.append(label)
    return findings


def _normalized_selection_omissions(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = selection.get("omissions", [])
    if not isinstance(value, list):
        raise ValidationError("selection.omissions must be a list")
    expected_keys = {
        "omission_id",
        "experiment_ids",
        "artifact_class",
        "logical_name",
        "source_root",
        "source",
        "source_sha256",
        "source_bytes",
        "reason",
        "anchor_package_path",
        "anchor_row",
    }
    omission_ids: set[str] = set()
    source_keys: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        label = f"selection.omissions[{index}]"
        if not isinstance(item, Mapping) or set(item) != expected_keys:
            raise ValidationError(f"{label} has the wrong schema")
        omission_id = item.get("omission_id")
        if not isinstance(omission_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", omission_id):
            raise ValidationError(f"{label}.omission_id is invalid")
        if omission_id in omission_ids:
            raise ValidationError(f"duplicate omission_id: {omission_id}")
        omission_ids.add(omission_id)
        experiment_ids = item.get("experiment_ids")
        if not isinstance(experiment_ids, list) or not experiment_ids or not all(
            isinstance(experiment_id, str) and experiment_id for experiment_id in experiment_ids
        ):
            raise ValidationError(f"{label}.experiment_ids is invalid")
        artifact_class = item.get("artifact_class")
        reason = item.get("reason")
        logical_name = item.get("logical_name")
        source_alias = item.get("source_root")
        if not isinstance(artifact_class, str) or not artifact_class.strip():
            raise ValidationError(f"{label}.artifact_class is invalid")
        if not isinstance(reason, str) or not reason.strip():
            raise ValidationError(f"{label}.reason is invalid")
        if (
            not isinstance(logical_name, str)
            or not logical_name
            or PurePosixPath(logical_name).name != logical_name
        ):
            raise ValidationError(f"{label}.logical_name is invalid")
        if not isinstance(source_alias, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_-]*", source_alias
        ):
            raise ValidationError(f"{label}.source_root is invalid")
        source_relpath = _relative_path(item.get("source"), f"{label}.source")
        source_key = (source_alias, source_relpath)
        if source_key in source_keys:
            raise ValidationError(f"duplicate omitted source: {source_key}")
        source_keys.add(source_key)
        source_sha256 = item.get("source_sha256")
        if not isinstance(source_sha256, str) or not SHA256_RE.fullmatch(source_sha256):
            raise ValidationError(f"{label}.source_sha256 is invalid")
        source_bytes = item.get("source_bytes")
        if (
            not isinstance(source_bytes, int)
            or isinstance(source_bytes, bool)
            or source_bytes < 0
        ):
            raise ValidationError(f"{label}.source_bytes is invalid")
        anchor_package_path = _relative_path(
            item.get("anchor_package_path"), f"{label}.anchor_package_path"
        )
        anchor_row = item.get("anchor_row")
        if not isinstance(anchor_row, Mapping) or set(anchor_row) != {"kind", "file_name"}:
            raise ValidationError(f"{label}.anchor_row has the wrong schema")
        anchor_kind = anchor_row.get("kind")
        anchor_file_name = anchor_row.get("file_name")
        if not isinstance(anchor_kind, str) or not anchor_kind:
            raise ValidationError(f"{label}.anchor_row.kind is invalid")
        if not isinstance(anchor_file_name, str) or not anchor_file_name:
            raise ValidationError(f"{label}.anchor_row.file_name is invalid")
        normalized.append(
            {
                "anchor_package_path": anchor_package_path,
                "anchor_row": {"file_name": anchor_file_name, "kind": anchor_kind},
                "artifact_class": artifact_class,
                "bundled": False,
                "experiment_ids": list(experiment_ids),
                "logical_name": logical_name,
                "omission_id": omission_id,
                "reason": reason,
                "source_alias": source_alias,
                "source_bytes": source_bytes,
                "source_relpath": source_relpath,
                "source_sha256": source_sha256,
                "verified_at_build": True,
            }
        )
    return sorted(normalized, key=lambda item: item["omission_id"])


def _validate_omission_ledger(
    root: Path,
    release: Mapping[str, Any],
    selection: Mapping[str, Any],
    artifacts_by_path: Mapping[str, Mapping[str, Any]],
) -> None:
    ledger_relative = _relative_path(
        release.get("omission_ledger"), "release_manifest.omission_ledger"
    )
    if ledger_relative != "provenance/omission_ledger.json":
        raise ValidationError("release omission_ledger path is not canonical")
    ledger = _load_json(root / Path(*PurePosixPath(ledger_relative).parts))
    if not isinstance(ledger, Mapping) or set(ledger) != {
        "omissions",
        "package_name",
        "schema_version",
    }:
        raise ValidationError("omission ledger has the wrong top-level schema")
    if ledger.get("package_name") != "core-results" or ledger.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError("omission ledger identity/schema mismatch")
    expected = _normalized_selection_omissions(selection)
    observed = ledger.get("omissions")
    if observed != expected:
        raise ValidationError("omission ledger differs from the reviewed selection")
    if release.get("omission_count") != len(expected):
        raise ValidationError("release omission_count differs from omission ledger")

    included = release.get("included_experiments")
    if not isinstance(included, list) or not all(isinstance(value, str) for value in included):
        raise ValidationError("release included_experiments is invalid for omission audit")
    included_set = set(included)
    selected_sources = {
        (str(record.get("source_alias", "")), str(record.get("source_relpath", "")))
        for record in artifacts_by_path.values()
    }
    for item in expected:
        source_key = (item["source_alias"], item["source_relpath"])
        if source_key in selected_sources:
            raise ValidationError(
                f"source is both bundled and omitted: {item['omission_id']}"
            )
        unknown = set(item["experiment_ids"]) - included_set
        if unknown:
            raise ValidationError(
                f"omission references experiments outside release: {sorted(unknown)}"
            )
        anchor_package_path = item["anchor_package_path"]
        if anchor_package_path not in artifacts_by_path:
            raise ValidationError(
                f"omission anchor artifact is absent: {anchor_package_path}"
            )
        anchor_path = root / Path(*PurePosixPath(anchor_package_path).parts)
        try:
            with anchor_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ValidationError(
                        f"omission anchor CSV has no header: {anchor_package_path}"
                    )
                required_columns = {"kind", "file_name", "sha256", "size_bytes"}
                if not required_columns.issubset(reader.fieldnames):
                    raise ValidationError(
                        f"omission anchor CSV lacks required columns: {anchor_package_path}"
                    )
                matches = [
                    row
                    for row in reader
                    if row.get("kind") == item["anchor_row"]["kind"]
                    and row.get("file_name") == item["anchor_row"]["file_name"]
                ]
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ValidationError(f"cannot read omission anchor CSV: {exc}") from exc
        if len(matches) != 1:
            raise ValidationError(
                f"omission anchor row is not unique: {item['omission_id']}"
            )
        match = matches[0]
        try:
            anchor_bytes = int(str(match.get("size_bytes", "")))
        except ValueError as exc:
            raise ValidationError(
                f"omission anchor size is invalid: {item['omission_id']}"
            ) from exc
        if (
            str(match.get("sha256", "")).lower() != item["source_sha256"]
            or anchor_bytes != item["source_bytes"]
        ):
            raise ValidationError(
                f"omission anchor hash/size mismatch: {item['omission_id']}"
            )


def validate_package(
    package_root: Path | str,
    *,
    public_source_root: Path | str | None = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Validate *package_root* and return a deterministic audit report."""

    root = Path(package_root).resolve()
    errors: list[str] = []
    try:
        if not root.is_dir():
            raise ValidationError(f"package root is not a directory: {root}")
        files, empty_dirs = _iter_regular_files(root)
        if empty_dirs:
            raise ValidationError(f"empty directories are forbidden: {sorted(empty_dirs)}")
        missing_generated = GENERATED_FILES - files
        if missing_generated:
            raise ValidationError(f"missing generated package files: {sorted(missing_generated)}")
        for relative in sorted(files):
            _relative_path(relative, "package file")
            _check_forbidden_path(relative)

        artifacts_payload = _load_json(root / "artifact_manifest.json")
        release = _load_json(root / "release_manifest.json")
        if not isinstance(artifacts_payload, dict) or not isinstance(release, dict):
            raise ValidationError("top-level manifests must be JSON objects")
        if artifacts_payload.get("schema_version") != SCHEMA_VERSION:
            raise ValidationError("unsupported artifact manifest schema_version")
        if release.get("schema_version") != SCHEMA_VERSION:
            raise ValidationError("unsupported release manifest schema_version")
        if release.get("package_name") != "core-results":
            raise ValidationError("release package_name must be exactly 'core-results'")
        if release.get("anonymity_profile") != SUBMISSION_ANONYMITY_PROFILE:
            raise ValidationError(
                f"release anonymity_profile must equal {SUBMISSION_ANONYMITY_PROFILE}"
            )
        if release.get("selection") != "provenance/release_selection.json":
            raise ValidationError("release selection reference is missing or non-portable")
        selection_snapshot = _load_json(root / "provenance/release_selection.json")
        if not isinstance(selection_snapshot, dict):
            raise ValidationError("release selection snapshot must be a JSON object")
        if selection_snapshot.get("package_name") != "core-results":
            raise ValidationError("release selection snapshot has the wrong package_name")
        if release.get("selection_sha256") != _sha256(
            root / "provenance/release_selection.json"
        ):
            raise ValidationError("release selection snapshot hash mismatch")

        artifacts = artifacts_payload.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValidationError("artifact_manifest.artifacts must be a non-empty list")
        payload_paths: set[str] = set()
        artifacts_by_path: dict[str, Mapping[str, Any]] = {}
        evidence_ids: set[str] = set()
        for index, record_value in enumerate(artifacts):
            if not isinstance(record_value, dict):
                raise ValidationError(f"artifact record {index} is not an object")
            record: Mapping[str, Any] = record_value
            evidence_id = record.get("evidence_id")
            if not isinstance(evidence_id, str) or not evidence_id:
                raise ValidationError(f"artifact record {index} has no evidence_id")
            if evidence_id in evidence_ids:
                raise ValidationError(f"duplicate evidence_id: {evidence_id}")
            evidence_ids.add(evidence_id)
            experiment_ids = record.get("experiment_ids")
            if not isinstance(experiment_ids, list) or not all(
                isinstance(value, str) and value for value in experiment_ids
            ):
                raise ValidationError(f"{evidence_id} has invalid experiment_ids")
            for field in REQUIRED_STATUS_FIELDS:
                if not isinstance(record.get(field), str) or not record[field]:
                    raise ValidationError(f"{evidence_id} lacks separate {field}")
            package_path = _relative_path(record.get("package_path"), f"{evidence_id}.package_path")
            source_relpath = _relative_path(
                record.get("source_relpath"), f"{evidence_id}.source_relpath"
            )
            del source_relpath
            if package_path in payload_paths or package_path in GENERATED_FILES:
                raise ValidationError(f"duplicate/reserved package_path: {package_path}")
            payload_paths.add(package_path)
            artifacts_by_path[package_path] = record
            for hash_field in ("source_sha256", "package_sha256"):
                digest = record.get(hash_field)
                if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                    raise ValidationError(f"invalid {hash_field} for {evidence_id}")
            for byte_field in ("source_bytes", "package_bytes"):
                byte_count = record.get(byte_field)
                if (
                    not isinstance(byte_count, int)
                    or isinstance(byte_count, bool)
                    or byte_count < 0
                ):
                    raise ValidationError(f"invalid {byte_field} for {evidence_id}")
            artifact_path = root / Path(*PurePosixPath(package_path).parts)
            if not artifact_path.is_file():
                raise ValidationError(f"missing payload artifact: {package_path}")
            observed = _sha256(artifact_path)
            if observed != record["package_sha256"]:
                raise ValidationError(
                    f"package hash mismatch for {package_path}: {observed} != "
                    f"{record['package_sha256']}"
                )
            if artifact_path.stat().st_size != record["package_bytes"]:
                raise ValidationError(f"package byte-size mismatch for {package_path}")

        _validate_omission_ledger(
            root, release, selection_snapshot, artifacts_by_path
        )
        _validate_internal_package_hash_links(root, artifacts_by_path)

        generated = release.get("generated_files")
        if not isinstance(generated, list):
            raise ValidationError("release_manifest.generated_files must be a list")
        generated_set = {
            _relative_path(value, "release_manifest.generated_files") for value in generated
        }
        if generated_set != GENERATED_FILES:
            raise ValidationError(
                f"generated file registration mismatch: {sorted(generated_set)}"
            )
        expected_files = payload_paths | GENERATED_FILES
        if files != expected_files:
            raise ValidationError(
                "unregistered or missing files: "
                f"extra={sorted(files - expected_files)} missing={sorted(expected_files - files)}"
            )

        checksums = _read_checksums(root / "SHA256SUMS")
        checksum_targets = files - {"SHA256SUMS"}
        if set(checksums) != checksum_targets:
            raise ValidationError(
                "SHA256SUMS coverage mismatch: "
                f"extra={sorted(set(checksums) - checksum_targets)} "
                f"missing={sorted(checksum_targets - set(checksums))}"
            )
        for relative, expected in sorted(checksums.items()):
            observed = _sha256(root / Path(*PurePosixPath(relative).parts))
            if observed != expected:
                raise ValidationError(f"SHA256SUMS mismatch for {relative}")

        mode = release.get("release_mode")
        pending_keys = [key for key in release if key.lower().startswith("pending")]
        ex48_gate_artifacts: set[str] = set()
        if mode == "draft":
            pending = release.get("pending_experiments")
            if not isinstance(pending, list) or not pending:
                raise ValidationError("draft release must declare non-empty pending_experiments")
            selection_pending = selection_snapshot.get(
                "pending_experiments", selection_snapshot.get("pending_inputs", [])
            )
            if not isinstance(selection_pending, list):
                raise ValidationError("draft selection pending inputs must be a list")
            normalized_selection_pending = [
                item.get("id") if isinstance(item, Mapping) else item
                for item in selection_pending
            ]
            if pending != normalized_selection_pending:
                raise ValidationError(
                    "draft release pending_experiments differs from selection snapshot"
                )
        elif mode == "final":
            if pending_keys:
                raise ValidationError(f"final release must omit pending fields: {pending_keys}")
            if not _has_experiment_48(artifacts, release):
                raise ValidationError(
                    "final release must include accepted EX48 artifacts for exact experiment "
                    f"{EX48_EXPERIMENT_ID}"
                )
            ex48_gate_artifacts = _validate_ex48_gate(
                root, release.get("ex48_final_gate"), artifacts_by_path
            )
            selection_markers = _selection_pending_markers(selection_snapshot)
            if selection_markers:
                raise ValidationError(
                    f"final selection snapshot contains pending markers: {selection_markers}"
                )
            readme = (root / "README.md").read_text(encoding="utf-8")
            if re.search(r"(?i)pending.{0,80}(?:ex)?48|(?:ex)?48.{0,80}pending", readme):
                raise ValidationError("final README must not contain an EX48 pending marker")
        else:
            raise ValidationError("release_mode must be 'draft' or 'final'")

        _validate_public_snapshots(
            root,
            release,
            artifacts,
            require_ex48_anchor=(mode == "final"),
            required_ex48_artifacts=ex48_gate_artifacts,
        )
        if public_source_root is not None:
            _validate_public_source_parity(
                root, public_source_root, release, artifacts_by_path
            )

        for relative in sorted(files):
            path = root / Path(*PurePosixPath(relative).parts)
            _scan_private_paths(path, relative)
            if path.suffix.lower() == ".json":
                _walk_json_references(_load_json(path), root, relative)
            elif path.suffix.lower() in {".csv", ".tsv"}:
                _audit_csv(path, relative, root)
    except ValidationError as exc:
        errors.append(str(exc))

    report = {
        "schema_version": SCHEMA_VERSION,
        "package_name": "core-results",
        "passed": not errors,
        "errors": errors,
    }
    if errors and raise_on_error:
        raise ValidationError(errors[0])
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "package_root",
        nargs="?",
        default=".",
        help="core-results directory (default: current directory)",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON audit report")
    parser.add_argument(
        "--public-source-root",
        help="optional public source tree for byte-parity checks of catalog/anchor snapshots",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = validate_package(
        args.package_root,
        public_source_root=args.public_source_root,
        raise_on_error=False,
    )
    if args.json or not report["passed"]:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("core-results validation passed")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
