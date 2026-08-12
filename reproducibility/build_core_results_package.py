#!/usr/bin/env python3
"""Build the relocatable, compact ``core-results`` evidence package.

The builder consumes a reviewed JSON selection plus one or more source roots
provided *only* on the command line.  It verifies every selected source hash,
copies only explicitly selected files (or files reached through an explicit CSV
dependency graph), removes private absolute paths from textual payloads, and
emits a self-validating package.

Example (run from the public source repository root)::

    python reproducibility/build_core_results_package.py \
      --selection reproducibility/core_results_selection.json \
      --source-root results=/path/to/accepted/results \
      --source-root source=/path/to/public/source \
      --mode draft --replace

The default output is the exact sibling directory ``core-results``.  Absolute
source roots are runtime inputs and are never persisted in the package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import posixpath
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from validate_core_results_package import (
    EX48_EXPERIMENT_ID,
    GENERATED_FILES,
    REQUIRED_STATUS_FIELDS,
    TEXT_SUFFIXES,
    ValidationError,
    _ex48_gate_paths,
    _validate_ex48_certificate_payloads,
    validate_package,
)


SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WINDOWS_ABSOLUTE_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
UNC_RE = re.compile(r"(?<![\\])\\\\[^\\\s]+\\[^\\\s]+")
PRIVATE_POSIX_RE = re.compile(
    "/" + r"(?:data|home|Users|mnt|workspace|root)(?:/|\\)", re.IGNORECASE
)
# Conservative full-path matchers used after known selected paths are remapped.
WINDOWS_PATH_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])[A-Za-z]:(?:\\\\|\\|/)[^\s\"'<>|,;)}\]]+"
)
UNC_PATH_RE = re.compile(r"(?<![\\])\\\\[^\s\"'<>|,;)}\]]+")
PRIVATE_POSIX_PATH_RE = re.compile(
    "/" + r"(?:data|home|Users|mnt|workspace|root)/[^\s\"'<>|,;)}\]]*",
    re.IGNORECASE,
)
PATH_KEY_RE = re.compile(
    r"(?:^|_)(?:artifact|checksums?|csv|destination|dir|directory|file|"
    r"manifest|package|path|report|source|validator)(?:$|_)",
    re.IGNORECASE,
)
LEGACY_PROJECT_TOKEN = "selective-newton-" + "muon-main-conference"
SUBMISSION_ANONYMITY_PROFILE = "double_blind_v1"
ANONYMIZED_WANDB_PREFIX = "ANONYMIZED_WANDB_"
ANONYMIZED_HOST_PREFIX = "ANONYMIZED_CONTAINER_HOST_"
WANDB_RUN_URL_RE = re.compile(
    r"https?://(?:api\.)?wandb\.ai/"
    r"(?P<entity>[^/\s\"'<>]+)/(?P<project>[^/\s\"'<>]+)/runs/"
    r"(?P<run_id>[^\s\"'<>),;\]}]+)",
    re.IGNORECASE,
)
WANDB_URL_RE = re.compile(
    r"https?://(?:api\.)?wandb\.ai(?:/[^\s\"'<>),;\]}]*)?",
    re.IGNORECASE,
)
CONTAINER_HOST_RE = re.compile(r"\bapp-[a-z0-9][a-z0-9-]{18,}\b", re.IGNORECASE)
QUOTED_IDENTITY_ASSIGNMENT_RE = re.compile(
    r"(?P<prefix>(?P<key_quote>[\"'])(?P<key>"
    r"(?:wandb[_-]?)?(?:entity|project|run[_-]?id|run[_-]?name)"
    r")(?P=key_quote)\s*[:=]\s*(?P<value_quote>[\"']))"
    r"(?P<value>.*?)(?P=value_quote)",
    re.IGNORECASE,
)
UNQUOTED_WANDB_ASSIGNMENT_RE = re.compile(
    r"(?P<prefix>\b(?P<key>wandb[_-]?(?:entity|project|run[_-]?id|run[_-]?name))"
    r"\b\s*[:=]\s*)(?P<value>[^\s,;}\]]+)",
    re.IGNORECASE,
)


class BuildError(RuntimeError):
    """Raised when the reviewed selection or source evidence is invalid."""


def _split_lines_with_endings(text: str) -> Iterable[tuple[str, str]]:
    """Yield logical lines while retaining the exact original newline style."""

    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        yield body, line[len(body) :]
    if text and not text.endswith(("\r", "\n")):
        return
    if not text:
        return


def _wandb_identity_kind(key: str, wandb_context: bool) -> str | None:
    """Return the submission-token namespace for a W&B identity field."""

    normalized = key.lower().replace("-", "_")
    if normalized.startswith("wandb_"):
        normalized = normalized[len("wandb_") :]
        wandb_context = True
    if not wandb_context:
        return None
    return {
        "entity": "ENTITY",
        "project": "PROJECT",
        "run_id": "RUN_ID",
        "run_name": "RUN_NAME",
    }.get(normalized)


@dataclass
class SubmissionIdentityAnonymizer:
    """Assign deterministic, non-identifying tokens during one package build.

    Plans are processed in sorted package-path order, and JSON/CSV traversal is
    deterministic.  Sequential tokens therefore preserve equality and
    distinctness without publishing a reversible hash of the private value.
    The map is deliberately memory-only and is never emitted in the package.
    """

    identities: dict[str, dict[str, str]] = field(default_factory=dict)

    def token(self, kind: str, value: str) -> str:
        normalized_kind = kind.upper()
        if value.startswith(ANONYMIZED_WANDB_PREFIX) or value.startswith(
            ANONYMIZED_HOST_PREFIX
        ):
            return value
        values = self.identities.setdefault(normalized_kind, {})
        if value not in values:
            prefix = (
                ANONYMIZED_HOST_PREFIX
                if normalized_kind == "HOST"
                else f"{ANONYMIZED_WANDB_PREFIX}{normalized_kind}_"
            )
            values[value] = f"{prefix}{len(values) + 1:04d}"
        return values[value]

    def _wandb_url(self, match: re.Match[str]) -> str:
        return (
            "https://wandb.invalid/"
            f"{self.token('ENTITY', match.group('entity'))}/"
            f"{self.token('PROJECT', match.group('project'))}/runs/"
            f"{self.token('RUN_ID', match.group('run_id'))}"
        )

    def anonymize_text(self, text: str, *, wandb_context: bool = False) -> str:
        """Anonymize identity-bearing strings while retaining scientific text."""

        result = WANDB_RUN_URL_RE.sub(self._wandb_url, text)
        result = WANDB_URL_RE.sub(
            lambda match: self.token("URL", match.group(0)), result
        )
        result = CONTAINER_HOST_RE.sub(
            lambda match: self.token("HOST", match.group(0)), result
        )

        output: list[str] = []
        for original_line, ending in _split_lines_with_endings(result):
            line_context = wandb_context or "wandb" in original_line.lower()

            def replace_quoted(match: re.Match[str]) -> str:
                kind = _wandb_identity_kind(match.group("key"), line_context)
                if kind is None:
                    return match.group(0)
                return (
                    match.group("prefix")
                    + self.token(kind, match.group("value"))
                    + match.group("value_quote")
                )

            line = QUOTED_IDENTITY_ASSIGNMENT_RE.sub(replace_quoted, original_line)

            def replace_unquoted(match: re.Match[str]) -> str:
                kind = _wandb_identity_kind(match.group("key"), True)
                assert kind is not None
                return match.group("prefix") + self.token(kind, match.group("value"))

            line = UNQUOTED_WANDB_ASSIGNMENT_RE.sub(replace_unquoted, line)
            output.append(line + ending)
        return "".join(output)

    def anonymize_json(
        self, value: Any, key_hint: str = "", *, wandb_context: bool = False
    ) -> tuple[Any, bool]:
        """Anonymize structured W&B identities without touching formal run IDs."""

        if isinstance(value, str):
            kind = _wandb_identity_kind(key_hint, wandb_context)
            if kind is not None:
                result = self.token(kind, value)
            elif key_hint.lower() in {"hostname", "host_name", "container_hostname"}:
                result = self.token("HOST", value)
            else:
                result = self.anonymize_text(value)
            return result, result != value
        if isinstance(value, list):
            output: list[Any] = []
            changed = False
            for child in value:
                transformed, child_changed = self.anonymize_json(
                    child, key_hint, wandb_context=wandb_context
                )
                output.append(transformed)
                changed = changed or child_changed
            return output, changed
        if isinstance(value, dict):
            output_object: dict[str, Any] = {}
            changed = False
            has_wandb_context = wandb_context or any(
                "wandb" in str(key).lower() for key in value
            )
            for key, child in value.items():
                if isinstance(child, str):
                    kind = _wandb_identity_kind(str(key), has_wandb_context)
                    if kind is not None:
                        transformed = self.token(kind, child)
                        child_changed = transformed != child
                    elif str(key).lower() in {
                        "hostname",
                        "host_name",
                        "container_hostname",
                    }:
                        transformed = self.token("HOST", child)
                        child_changed = transformed != child
                    else:
                        transformed, child_changed = self.anonymize_json(
                            child, str(key), wandb_context=has_wandb_context
                        )
                else:
                    transformed, child_changed = self.anonymize_json(
                        child, str(key), wandb_context=has_wandb_context
                    )
                output_object[key] = transformed
                changed = changed or child_changed
            return output_object, changed
        return value, False


@dataclass
class ArtifactPlan:
    evidence_id: str
    source_alias: str
    source_relpath: str
    package_path: str
    expected_sha256: str
    integrity_status: str
    scientific_status: str
    claim_eligibility: str
    paper_role: str
    experiment_id: str = ""
    experiment_ids: tuple[str, ...] = field(default_factory=tuple)
    workstream: str = ""
    selection_method: str = "direct"
    source_hash_basis: str = "selection_anchor"
    graph_id: str = ""
    tier: str = ""
    csv_path_columns: tuple[str, ...] = field(default_factory=tuple)
    csv_hash_columns: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    csv_shared_hash_column: str = ""
    csv_relative_to: str = "source_root"
    legacy_root_markers: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OmissionPlan:
    """A byte-verified source intentionally excluded from the compact package."""

    omission_id: str
    experiment_ids: tuple[str, ...]
    artifact_class: str
    logical_name: str
    source_alias: str
    source_relpath: str
    source_sha256: str
    source_bytes: int
    reason: str
    anchor_package_path: str
    anchor_kind: str
    anchor_file_name: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _load_json(path: Path) -> Any:
    def no_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BuildError(f"duplicate JSON key in selection: {key!r}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=no_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot read selection {path}: {exc}") from exc


def _canonical_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BuildError(f"{label} must be a non-empty relative path")
    if WINDOWS_ABSOLUTE_RE.search(value) or UNC_RE.search(value):
        raise BuildError(f"{label} must not be an absolute Windows path: {value!r}")
    normalized = value.replace("\\", "/")
    if normalized.startswith("/"):
        raise BuildError(f"{label} must not be absolute: {value!r}")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise BuildError(f"{label} is not canonical: {value!r}")
    if normalized != pure.as_posix():
        raise BuildError(f"{label} must use canonical '/' separators: {value!r}")
    return normalized


def _parse_source_roots(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise BuildError(f"--source-root must be ALIAS=PATH, got {value!r}")
        alias, path_value = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", alias):
            raise BuildError(f"invalid source-root alias: {alias!r}")
        if alias in roots:
            raise BuildError(f"duplicate source-root alias: {alias}")
        root = Path(path_value).expanduser().resolve()
        if not root.is_dir():
            raise BuildError(f"source root is not a directory: {alias}")
        roots[alias] = root
    if not roots:
        raise BuildError("at least one --source-root ALIAS=PATH is required")
    return roots


def _source_file(roots: Mapping[str, Path], alias: str, relative: str) -> Path:
    if alias not in roots:
        raise BuildError(f"selection uses undeclared source-root alias: {alias!r}")
    root = roots[alias]
    path = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise BuildError(f"source path escapes root {alias}: {relative}") from exc
    if not path.is_file() or path.is_symlink():
        raise BuildError(f"selected source is missing, non-regular, or symlinked: {alias}:{relative}")
    return path


def _status(item: Mapping[str, Any], field_name: str, inherited: Mapping[str, str] | None = None) -> str:
    candidates = [item.get(field_name)]
    status_object = item.get("status")
    if isinstance(status_object, Mapping):
        candidates.append(status_object.get(field_name))
    if inherited:
        candidates.append(inherited.get(field_name))
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise BuildError(f"selection item lacks separate non-empty {field_name}")


def _expected_hash(item: Mapping[str, Any], fallback: str | None, label: str) -> str:
    value = item.get("source_sha256", item.get("expected_sha256", item.get("sha256", fallback)))
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value.lower()):
        raise BuildError(f"{label} lacks a valid anchored source SHA-256")
    return value.lower()


def _source_fields(item: Mapping[str, Any], label: str) -> tuple[str, str]:
    alias = item.get("source_root", item.get("source_alias", item.get("root")))
    relative_value = item.get("source", item.get("source_path", item.get("path")))
    if not isinstance(alias, str) or not alias:
        raise BuildError(f"{label} lacks source_root alias")
    return alias, _canonical_relative(relative_value, f"{label}.source")


def _destination(item: Mapping[str, Any], label: str) -> str:
    value = item.get("package_path", item.get("destination", item.get("dest")))
    return _canonical_relative(value, f"{label}.package_path")


def _anchor_map(selection: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    payload = selection.get("source_anchors", selection.get("anchors", {}))
    result: dict[tuple[str, str], str] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(value, Mapping):
                alias = str(key)
                for relative, digest in value.items():
                    canonical = _canonical_relative(relative, f"source_anchors.{alias}")
                    normalized = str(digest).lower()
                    if not SHA256_RE.fullmatch(normalized):
                        raise BuildError(f"invalid source anchor for {alias}:{canonical}")
                    result[(alias, canonical)] = normalized
            elif isinstance(key, str) and ":" in key:
                alias, relative = key.split(":", 1)
                canonical = _canonical_relative(relative, f"source_anchors.{key}")
                normalized = str(value).lower()
                if not SHA256_RE.fullmatch(normalized):
                    raise BuildError(f"invalid source anchor for {key}")
                result[(alias, canonical)] = normalized
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            if not isinstance(item, Mapping):
                raise BuildError(f"source_anchors[{index}] must be an object")
            alias, relative = _source_fields(item, f"source_anchors[{index}]")
            result[(alias, relative)] = _expected_hash(item, None, f"source_anchors[{index}]")
    elif payload:
        raise BuildError("source_anchors must be an object or list")
    return result


def _direct_entries(selection: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("direct_files", "artifacts", "files"):
        value = selection.get(key)
        if value is not None:
            if not isinstance(value, list):
                raise BuildError(f"selection.{key} must be a list")
            result: list[Mapping[str, Any]] = []
            for index, item in enumerate(value):
                if not isinstance(item, Mapping):
                    raise BuildError(f"selection.{key}[{index}] must be an object")
                result.append(item)
            return result
    return []


def _infer_experiment_id(record_id: str) -> str:
    match = re.search(r"(?i)(?:^|_)(?:ex)?([0-9]{1,3})(?:[a-z]?)(?:_|$)", record_id)
    return match.group(1) if match else ""


def _record_entries(selection: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = selection.get("records", [])
    if not isinstance(value, list):
        raise BuildError("selection.records must be a list")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise BuildError(f"selection.records[{index}] must be an object")
        result.append(item)
    return result


def _record_plans(
    selection: Mapping[str, Any], roots: Mapping[str, Path]
) -> list[ArtifactPlan]:
    """Expand compact record-level whitelists after verifying each authority anchor."""

    plans: list[ArtifactPlan] = []
    for index, record in enumerate(_record_entries(selection)):
        label = f"records[{index}]"
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise BuildError(f"{label} lacks id")
        alias = record.get("source_root", record.get("source_alias", "results"))
        if not isinstance(alias, str) or alias not in roots:
            raise BuildError(f"{label} uses undeclared source_root alias {alias!r}")
        source_prefix = _canonical_relative(record.get("source_prefix"), f"{label}.source_prefix")
        destination_prefix = _canonical_relative(
            record.get("destination_prefix"), f"{label}.destination_prefix"
        )
        anchor = record.get("source_anchor")
        if not isinstance(anchor, Mapping):
            raise BuildError(f"{label}.source_anchor must be an object")
        anchor_file = _canonical_relative(anchor.get("file"), f"{label}.source_anchor.file")
        anchor_relative = f"{source_prefix}/{anchor_file}"
        anchor_expected = _expected_hash(anchor, None, f"{label}.source_anchor")
        anchor_source = _source_file(roots, alias, anchor_relative)
        anchor_observed = _sha256_file(anchor_source)
        if anchor_observed != anchor_expected:
            raise BuildError(
                f"record source anchor mismatch for {record_id}: expected "
                f"{anchor_expected}, observed {anchor_observed}"
            )
        files_value = record.get("files")
        if not isinstance(files_value, list) or not files_value:
            raise BuildError(f"{label}.files must be a non-empty explicit whitelist")
        seen_files: set[str] = set()
        inherited = {field_name: _status(record, field_name) for field_name in REQUIRED_STATUS_FIELDS}
        experiment_ids_value = record.get("experiment_ids")
        if not isinstance(experiment_ids_value, list) or not experiment_ids_value or not all(
            isinstance(value, str) and value for value in experiment_ids_value
        ):
            raise BuildError(f"{label}.experiment_ids must be a non-empty list of catalog IDs")
        record_experiment_ids = tuple(experiment_ids_value)
        for file_index, file_value in enumerate(files_value):
            file_relative = _canonical_relative(file_value, f"{label}.files[{file_index}]")
            if any(character in file_relative for character in "*?[]"):
                raise BuildError(f"glob syntax is forbidden in {label}.files: {file_relative}")
            if file_relative in seen_files:
                raise BuildError(f"duplicate file in {label}: {file_relative}")
            seen_files.add(file_relative)
            source_relative = f"{source_prefix}/{file_relative}"
            source = _source_file(roots, alias, source_relative)
            observed = _sha256_file(source)
            expected = anchor_expected if file_relative == anchor_file else observed
            slug = re.sub(r"[^A-Za-z0-9]+", "_", file_relative).strip("_")[:80]
            suffix = hashlib.sha256(file_relative.encode("utf-8")).hexdigest()[:10]
            plans.append(
                ArtifactPlan(
                    evidence_id=f"{record_id}__{slug}__{suffix}",
                    source_alias=alias,
                    source_relpath=source_relative,
                    package_path=f"{destination_prefix}/{file_relative}",
                    expected_sha256=expected,
                    integrity_status=inherited["integrity_status"],
                    scientific_status=inherited["scientific_status"],
                    claim_eligibility=inherited["claim_eligibility"],
                    paper_role=inherited["paper_role"],
                    experiment_id=str(record.get("experiment_id", record_experiment_ids[0])),
                    experiment_ids=record_experiment_ids,
                    workstream=str(record.get("workstream", "")),
                    selection_method="record_file_whitelist",
                    source_hash_basis=(
                        "selection_anchor"
                        if file_relative == anchor_file
                        else "observed_after_record_anchor_verification"
                    ),
                    tier=str(record.get("tier", "")),
                )
            )
    return plans


def _graph_entries(selection: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("csv_graph_dependencies", "csv_graphs", "dependency_graphs"):
        value = selection.get(key)
        if value is not None:
            if isinstance(value, Mapping):
                return [value]
            if not isinstance(value, list):
                raise BuildError(f"selection.{key} must be an object or list")
            result: list[Mapping[str, Any]] = []
            for index, item in enumerate(value):
                if not isinstance(item, Mapping):
                    raise BuildError(f"selection.{key}[{index}] must be an object")
                result.append(item)
            return result
    return []


def _make_direct_plan(
    item: Mapping[str, Any], index: int, anchors: Mapping[tuple[str, str], str]
) -> ArtifactPlan:
    label = f"direct_files[{index}]"
    alias, relative = _source_fields(item, label)
    evidence_id = item.get("evidence_id", item.get("id"))
    if not isinstance(evidence_id, str) or not evidence_id:
        raise BuildError(f"{label} lacks evidence_id")
    experiment_ids_value = item.get("experiment_ids", [])
    if not isinstance(experiment_ids_value, list) or not all(
        isinstance(value, str) and value for value in experiment_ids_value
    ):
        raise BuildError(f"{label}.experiment_ids must be a list of strings when present")
    primary_experiment_id = str(item.get("experiment_id", ""))
    experiment_ids = tuple(experiment_ids_value) or (
        (primary_experiment_id,) if primary_experiment_id else ()
    )
    return ArtifactPlan(
        evidence_id=evidence_id,
        source_alias=alias,
        source_relpath=relative,
        package_path=_destination(item, label),
        expected_sha256=_expected_hash(item, anchors.get((alias, relative)), label),
        integrity_status=_status(item, "integrity_status"),
        scientific_status=_status(item, "scientific_status"),
        claim_eligibility=_status(item, "claim_eligibility"),
        paper_role=_status(item, "paper_role"),
        experiment_id=primary_experiment_id or (experiment_ids[0] if experiment_ids else ""),
        experiment_ids=experiment_ids,
        workstream=str(item.get("workstream", "")),
        tier=str(item.get("tier", "")),
    )


def _normalize_graph_roots(graph: Mapping[str, Any], label: str) -> list[Mapping[str, Any]]:
    if "root_manifest" in graph:
        value = {
            "source": graph.get("root_manifest"),
            "source_sha256": graph.get("root_manifest_sha256"),
        }
    else:
        value = graph.get("entry_csvs", graph.get("roots", graph.get("root_csv")))
    if isinstance(value, str):
        return [{"source": value}]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        normalized: list[Mapping[str, Any]] = []
        for index, entry in enumerate(value):
            if isinstance(entry, str):
                normalized.append({"source": entry})
            elif isinstance(entry, Mapping):
                normalized.append(entry)
            else:
                raise BuildError(f"{label}.roots[{index}] must be string/object")
        return normalized
    raise BuildError(f"{label} must declare root_csv/roots/entry_csvs")


def _resolve_dependency(
    value: str,
    alias: str,
    current_relative: str,
    roots: Mapping[str, Path],
    relative_to: str,
    legacy_root_markers: Sequence[str] = (),
) -> str:
    stripped = value.strip()
    if not stripped or "://" in stripped:
        raise BuildError(f"CSV dependency is empty or remote: {value!r}")
    root = roots[alias]
    candidate_path = Path(stripped)
    is_legacy_absolute = (
        candidate_path.is_absolute()
        or bool(WINDOWS_ABSOLUTE_RE.search(stripped))
        or stripped.startswith("/")
    )
    if is_legacy_absolute:
        candidate = candidate_path.resolve()
        try:
            relative = candidate.relative_to(root)
            return relative.as_posix()
        except ValueError:
            normalized_legacy = re.sub(r"/+", "/", stripped.replace("\\", "/"))
            recovered: str | None = None
            for marker in legacy_root_markers:
                normalized_marker = marker.replace("\\", "/").strip("/")
                marker_token = "/" + normalized_marker + "/"
                haystack = "/" + normalized_legacy.strip("/") + "/"
                index = haystack.lower().find(marker_token.lower())
                if index >= 0:
                    recovered = haystack[index + len(marker_token) :].rstrip("/")
                    break
            if recovered is None:
                raise BuildError(
                    f"legacy absolute CSV dependency has no configured project-root marker: {value!r}"
                )
            candidate = (root / Path(*PurePosixPath(recovered).parts)).resolve()
    elif relative_to == "csv_parent":
        current_parent = Path(*PurePosixPath(current_relative).parent.parts)
        candidate = (root / current_parent / candidate_path).resolve()
    elif relative_to == "source_root":
        candidate = (root / candidate_path).resolve()
    else:
        raise BuildError(f"unsupported csv relative_to: {relative_to!r}")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise BuildError(f"CSV dependency escapes source root {alias}: {value!r}") from exc
    return relative.as_posix()


def _split_dependency_cell(value: str, delimiter: str | None) -> list[str]:
    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise BuildError(f"invalid JSON dependency list: {stripped!r}") from exc
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise BuildError("CSV JSON dependency cell must be a list of strings")
        return [item for item in parsed if item]
    if delimiter:
        return [item.strip() for item in stripped.split(delimiter) if item.strip()]
    return [stripped]


def _graph_plans(
    graph: Mapping[str, Any],
    graph_index: int,
    roots: Mapping[str, Path],
    anchors: MutableMapping[tuple[str, str], str],
) -> list[ArtifactPlan]:
    label = f"csv_graphs[{graph_index}]"
    graph_id = graph.get("graph_id", graph.get("id"))
    if not isinstance(graph_id, str) or not graph_id:
        raise BuildError(f"{label} lacks graph_id")
    alias = graph.get("source_root", graph.get("source_alias", graph.get("root")))
    if not isinstance(alias, str) or alias not in roots:
        raise BuildError(f"{label} uses an undeclared source_root")
    path_columns_value = graph.get("path_columns")
    if path_columns_value is None and isinstance(graph.get("path_column"), str):
        path_columns_value = [graph["path_column"]]
    if not isinstance(path_columns_value, list) or not path_columns_value or not all(
        isinstance(value, str) and value for value in path_columns_value
    ):
        raise BuildError(f"{label}.path_columns must be a non-empty list of names")
    path_columns = tuple(path_columns_value)
    hash_columns_value = graph.get("hash_columns", {})
    if hash_columns_value and not isinstance(hash_columns_value, Mapping):
        raise BuildError(f"{label}.hash_columns must be an object")
    hash_columns = dict(hash_columns_value) if isinstance(hash_columns_value, Mapping) else {}
    shared_hash_column = graph.get("sha256_column", graph.get("hash_column"))
    destination_root = _canonical_relative(
        graph.get("destination_root", graph.get("package_prefix", f"evidence/{graph_id}")),
        f"{label}.destination_root",
    )
    relative_to = str(graph.get("relative_to", "source_root"))
    recursive = bool(graph.get("recursive", True))
    flatten_destinations = bool(graph.get("flatten_destinations", False))
    delimiter_value = graph.get("path_delimiter")
    delimiter = str(delimiter_value) if delimiter_value is not None else None
    markers_value = graph.get("legacy_root_markers", graph.get("result_root_markers", []))
    if not isinstance(markers_value, list) or not all(
        isinstance(value, str) and value.strip() for value in markers_value
    ):
        raise BuildError(f"{label}.legacy_root_markers must be a list of relative markers")
    legacy_root_markers: tuple[str, ...] = tuple(
        _canonical_relative(value.strip().strip("/\\"), f"{label}.legacy_root_markers")
        for value in markers_value
    )
    inherited = {field_name: _status(graph, field_name) for field_name in REQUIRED_STATUS_FIELDS}
    experiment_id = str(graph.get("experiment_id", ""))
    graph_experiment_ids_value = graph.get("experiment_ids", [])
    if not isinstance(graph_experiment_ids_value, list) or not all(
        isinstance(value, str) and value for value in graph_experiment_ids_value
    ):
        raise BuildError(f"{label}.experiment_ids must be a list of strings when present")
    graph_experiment_ids = tuple(graph_experiment_ids_value) or (
        (experiment_id,) if experiment_id else ()
    )
    workstream = str(graph.get("workstream", ""))

    plans: dict[str, ArtifactPlan] = {}
    queue: list[str] = []
    root_destinations: dict[str, str] = {}
    roots_payload = _normalize_graph_roots(graph, label)

    def graph_destination(relative: str) -> str:
        if not flatten_destinations:
            return f"{destination_root}/{relative}"
        pure = PurePosixPath(relative)
        first = pure.parts[0]
        name = pure.name
        suffix = PurePosixPath(name).suffix
        stem = name[: -len(suffix)] if suffix else name
        short_hash = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
        return f"{destination_root}/{first}/{stem}__{short_hash}{suffix}"

    for root_index, root_item in enumerate(roots_payload):
        root_alias = root_item.get("source_root", root_item.get("source_alias", alias))
        if root_alias != alias:
            raise BuildError(f"{label}.roots[{root_index}] must use graph source_root {alias}")
        relative_value = root_item.get("source", root_item.get("source_path", root_item.get("path")))
        relative = _canonical_relative(relative_value, f"{label}.roots[{root_index}]")
        anchors[(alias, relative)] = _expected_hash(
            root_item,
            anchors.get((alias, relative)),
            f"{label}.roots[{root_index}]",
        )
        if "root_manifest" in graph and len(roots_payload) == 1:
            root_destinations[relative] = graph_destination(relative)
        queue.append(relative)

    traversal_roots = set(queue)

    processed_csv: set[str] = set()
    while queue:
        relative = queue.pop(0)
        expected = anchors.get((alias, relative))
        if not expected:
            raise BuildError(f"CSV graph dependency lacks source anchor: {alias}:{relative}")
        if relative not in plans:
            slug = re.sub(r"[^A-Za-z0-9]+", "_", relative).strip("_")[:80]
            suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
            plans[relative] = ArtifactPlan(
                evidence_id=f"{graph_id}__{slug}__{suffix}",
                source_alias=alias,
                source_relpath=relative,
                package_path=root_destinations.get(relative, graph_destination(relative)),
                expected_sha256=expected,
                integrity_status=inherited["integrity_status"],
                scientific_status=inherited["scientific_status"],
                claim_eligibility=inherited["claim_eligibility"],
                paper_role=inherited["paper_role"],
                experiment_id=experiment_id,
                experiment_ids=graph_experiment_ids,
                workstream=workstream,
                selection_method="csv_graph",
                graph_id=graph_id,
                tier=str(graph.get("tier", "")),
                csv_path_columns=(
                    path_columns
                    if relative.lower().endswith(".csv")
                    and (recursive or relative in traversal_roots)
                    else ()
                ),
                csv_hash_columns=tuple(
                    sorted((str(path_column), str(hash_column)) for path_column, hash_column in hash_columns.items())
                ),
                csv_shared_hash_column=str(shared_hash_column or ""),
                csv_relative_to=relative_to,
                legacy_root_markers=legacy_root_markers,
            )
        if not relative.lower().endswith(".csv") or relative in processed_csv:
            continue
        if not recursive and relative not in traversal_roots:
            continue
        processed_csv.add(relative)
        source_csv = _source_file(roots, alias, relative)
        try:
            with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise BuildError(f"CSV graph node has no header: {alias}:{relative}")
                for row_number, row in enumerate(reader, start=2):
                    if None in row:
                        raise BuildError(f"CSV graph row has extra cells: {relative}:{row_number}")
                    for column in path_columns:
                        if column not in row:
                            continue
                        for dependency_value in _split_dependency_cell(row.get(column, ""), delimiter):
                            dependency = _resolve_dependency(
                                dependency_value,
                                alias,
                                relative,
                                roots,
                                relative_to,
                                legacy_root_markers,
                            )
                            hash_column = hash_columns.get(column, shared_hash_column)
                            row_hash = row.get(str(hash_column), "") if hash_column else ""
                            if row_hash:
                                normalized_hash = row_hash.strip().lower()
                                if not SHA256_RE.fullmatch(normalized_hash):
                                    raise BuildError(
                                        f"bad dependency hash {relative}:{row_number}:{hash_column}"
                                    )
                                old = anchors.get((alias, dependency))
                                if old and old != normalized_hash:
                                    raise BuildError(f"conflicting anchors for {alias}:{dependency}")
                                anchors[(alias, dependency)] = normalized_hash
                            if (alias, dependency) not in anchors:
                                raise BuildError(
                                    f"unanchored CSV dependency {relative}:{row_number}:{column}: "
                                    f"{dependency}"
                                )
                            if dependency not in plans:
                                queue.append(dependency)
                            elif recursive and dependency.lower().endswith(".csv"):
                                queue.append(dependency)
        except (OSError, UnicodeError, csv.Error) as exc:
            raise BuildError(f"cannot traverse CSV graph {alias}:{relative}: {exc}") from exc
    return [plans[key] for key in sorted(plans)]


def _validate_plan(plans: Sequence[ArtifactPlan], roots: Mapping[str, Path]) -> None:
    if not plans:
        raise BuildError("selection resolved to no artifacts")
    evidence_ids: set[str] = set()
    destinations: set[str] = set()
    for plan in plans:
        if plan.evidence_id in evidence_ids:
            raise BuildError(f"duplicate evidence_id: {plan.evidence_id}")
        if plan.package_path in destinations or plan.package_path in GENERATED_FILES:
            raise BuildError(f"duplicate/reserved package path: {plan.package_path}")
        evidence_ids.add(plan.evidence_id)
        destinations.add(plan.package_path)
        _source_file(roots, plan.source_alias, plan.source_relpath)
        lowered_parts = [part.lower() for part in PurePosixPath(plan.package_path).parts]
        lowered_name = PurePosixPath(plan.package_path).name.lower()
        if any(part in {".cache", ".wandb", "__pycache__", "cache", "logs", "wandb"} for part in lowered_parts):
            raise BuildError(f"forbidden raw/cache destination: {plan.package_path}")
        if Path(lowered_name).suffix in {".ckpt", ".gz", ".log", ".pt", ".pth", ".tar", ".tgz", ".zip"}:
            raise BuildError(f"forbidden archive/checkpoint/log destination: {plan.package_path}")
        if "wandb_export" in lowered_name:
            raise BuildError(f"forbidden raw W&B export destination: {plan.package_path}")


def _parse_and_verify_omissions(
    selection: Mapping[str, Any],
    plans: Sequence[ArtifactPlan],
    roots: Mapping[str, Path],
) -> list[OmissionPlan]:
    """Verify every intentionally omitted byte source and its compact hash anchor."""

    value = selection.get("omissions", [])
    if not isinstance(value, list):
        raise BuildError("selection.omissions must be a list")
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
    plans_by_package = {plan.package_path: plan for plan in plans}
    selected_sources = {(plan.source_alias, plan.source_relpath) for plan in plans}
    included_experiments = set(_included_experiments(selection, plans))
    omission_ids: set[str] = set()
    omission_sources: set[tuple[str, str]] = set()
    result: list[OmissionPlan] = []
    for index, item in enumerate(value):
        label = f"omissions[{index}]"
        if not isinstance(item, Mapping) or set(item) != expected_keys:
            actual = sorted(item) if isinstance(item, Mapping) else type(item).__name__
            raise BuildError(
                f"{label} must contain exactly {sorted(expected_keys)}; got {actual}"
            )
        omission_id = item.get("omission_id")
        if not isinstance(omission_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", omission_id):
            raise BuildError(f"{label}.omission_id is invalid")
        if omission_id in omission_ids:
            raise BuildError(f"duplicate omission_id: {omission_id}")
        omission_ids.add(omission_id)

        experiment_ids_value = item.get("experiment_ids")
        if not isinstance(experiment_ids_value, list) or not experiment_ids_value or not all(
            isinstance(experiment_id, str) and experiment_id
            for experiment_id in experiment_ids_value
        ):
            raise BuildError(f"{label}.experiment_ids must be a non-empty string list")
        experiment_ids = tuple(experiment_ids_value)
        unknown_experiments = set(experiment_ids) - included_experiments
        if unknown_experiments:
            raise BuildError(
                f"{label}.experiment_ids are not included: {sorted(unknown_experiments)}"
            )

        artifact_class = item.get("artifact_class")
        reason = item.get("reason")
        logical_name = item.get("logical_name")
        if not isinstance(artifact_class, str) or not artifact_class.strip():
            raise BuildError(f"{label}.artifact_class must be non-empty")
        if not isinstance(reason, str) or not reason.strip():
            raise BuildError(f"{label}.reason must be non-empty")
        if (
            not isinstance(logical_name, str)
            or not logical_name
            or PurePosixPath(logical_name).name != logical_name
        ):
            raise BuildError(f"{label}.logical_name must be a basename")

        source_alias, source_relpath = _source_fields(item, label)
        source_key = (source_alias, source_relpath)
        if source_key in selected_sources:
            raise BuildError(
                f"source cannot be both bundled and intentionally omitted: "
                f"{source_alias}:{source_relpath}"
            )
        if source_key in omission_sources:
            raise BuildError(f"duplicate omitted source: {source_alias}:{source_relpath}")
        omission_sources.add(source_key)
        source_sha256 = _expected_hash(item, None, label)
        source_bytes = item.get("source_bytes")
        if (
            not isinstance(source_bytes, int)
            or isinstance(source_bytes, bool)
            or source_bytes < 0
        ):
            raise BuildError(f"{label}.source_bytes must be a non-negative integer")
        source = _source_file(roots, source_alias, source_relpath)
        observed_sha256 = _sha256_file(source)
        observed_bytes = source.stat().st_size
        if observed_sha256 != source_sha256 or observed_bytes != source_bytes:
            raise BuildError(
                f"omitted source anchor mismatch for {omission_id}: "
                f"sha256={observed_sha256} bytes={observed_bytes}"
            )

        anchor_package_path = _canonical_relative(
            item.get("anchor_package_path"), f"{label}.anchor_package_path"
        )
        anchor_plan = plans_by_package.get(anchor_package_path)
        if anchor_plan is None:
            raise BuildError(
                f"{label}.anchor_package_path is not a selected artifact: "
                f"{anchor_package_path}"
            )
        if PurePosixPath(anchor_package_path).suffix.lower() != ".csv":
            raise BuildError(f"{label}.anchor_package_path must identify a CSV artifact")
        anchor_row = item.get("anchor_row")
        if not isinstance(anchor_row, Mapping) or set(anchor_row) != {"kind", "file_name"}:
            raise BuildError(f"{label}.anchor_row must contain exactly kind and file_name")
        anchor_kind = anchor_row.get("kind")
        anchor_file_name = anchor_row.get("file_name")
        if not isinstance(anchor_kind, str) or not anchor_kind:
            raise BuildError(f"{label}.anchor_row.kind must be non-empty")
        if not isinstance(anchor_file_name, str) or not anchor_file_name:
            raise BuildError(f"{label}.anchor_row.file_name must be non-empty")
        anchor_source = _source_file(
            roots, anchor_plan.source_alias, anchor_plan.source_relpath
        )
        try:
            with anchor_source.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise BuildError(f"omission anchor CSV has no header: {anchor_package_path}")
                required_columns = {"kind", "file_name", "sha256", "size_bytes"}
                if not required_columns.issubset(reader.fieldnames):
                    raise BuildError(
                        f"omission anchor CSV lacks columns {sorted(required_columns)}: "
                        f"{anchor_package_path}"
                    )
                matches = [
                    row
                    for row in reader
                    if row.get("kind") == anchor_kind
                    and row.get("file_name") == anchor_file_name
                ]
        except (OSError, UnicodeError, csv.Error) as exc:
            raise BuildError(f"cannot read omission anchor CSV {anchor_package_path}: {exc}") from exc
        if len(matches) != 1:
            raise BuildError(
                f"omission anchor row must match exactly once for {omission_id}; "
                f"observed={len(matches)}"
            )
        anchored = matches[0]
        try:
            anchored_bytes = int(str(anchored.get("size_bytes", "")))
        except ValueError as exc:
            raise BuildError(f"omission anchor size is invalid for {omission_id}") from exc
        if (
            str(anchored.get("sha256", "")).lower() != source_sha256
            or anchored_bytes != source_bytes
        ):
            raise BuildError(f"omission anchor digest/size mismatch for {omission_id}")

        result.append(
            OmissionPlan(
                omission_id=omission_id,
                experiment_ids=experiment_ids,
                artifact_class=artifact_class,
                logical_name=logical_name,
                source_alias=source_alias,
                source_relpath=source_relpath,
                source_sha256=source_sha256,
                source_bytes=source_bytes,
                reason=reason,
                anchor_package_path=anchor_package_path,
                anchor_kind=anchor_kind,
                anchor_file_name=anchor_file_name,
            )
        )
    return sorted(result, key=lambda item: item.omission_id)


def _omission_ledger_bytes(omissions: Sequence[OmissionPlan]) -> bytes:
    return _json_bytes(
        {
            "omissions": [
                {
                    "anchor_package_path": item.anchor_package_path,
                    "anchor_row": {
                        "file_name": item.anchor_file_name,
                        "kind": item.anchor_kind,
                    },
                    "artifact_class": item.artifact_class,
                    "bundled": False,
                    "experiment_ids": list(item.experiment_ids),
                    "logical_name": item.logical_name,
                    "omission_id": item.omission_id,
                    "reason": item.reason,
                    "source_alias": item.source_alias,
                    "source_bytes": item.source_bytes,
                    "source_relpath": item.source_relpath,
                    "source_sha256": item.source_sha256,
                    "verified_at_build": True,
                }
                for item in omissions
            ],
            "package_name": "core-results",
            "schema_version": SCHEMA_VERSION,
        }
    )


def _known_path_replacements(
    plans: Sequence[ArtifactPlan], roots: Mapping[str, Path]
) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for plan in plans:
        absolute = str(_source_file(roots, plan.source_alias, plan.source_relpath))
        variants = {
            absolute,
            absolute.replace("\\", "/"),
            absolute.replace("\\", "\\\\"),
        }
        try:
            variants.add(Path(absolute).as_posix())
        except OSError:
            pass
        for variant in variants:
            replacements[variant] = plan.package_path
    return replacements


def _sanitize_text(
    text: str,
    replacements: Mapping[str, str],
    anonymizer: SubmissionIdentityAnonymizer,
    *,
    wandb_context: bool = False,
) -> str:
    result = text.replace(LEGACY_PROJECT_TOKEN, "project-results")
    for absolute in sorted(replacements, key=len, reverse=True):
        result = result.replace(absolute, replacements[absolute])
    result = WINDOWS_PATH_RE.sub("PRIVATE_PATH_REDACTED", result)
    result = UNC_PATH_RE.sub("PRIVATE_PATH_REDACTED", result)
    result = PRIVATE_POSIX_PATH_RE.sub("PRIVATE_PATH_REDACTED", result)
    return anonymizer.anonymize_text(result, wandb_context=wandb_context)


def _portable_json_bytes(
    raw: bytes,
    replacements: Mapping[str, str],
    anonymizer: SubmissionIdentityAnonymizer,
) -> tuple[bytes, bool]:
    """Sanitize JSON values and keys without collapsing distinct path-valued keys."""

    def no_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BuildError(f"selected JSON contains duplicate source key: {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"selected .json file is invalid UTF-8 JSON: {exc}") from exc

    package_destinations = set(replacements.values())

    def transform(value: Any, key_hint: str = "") -> tuple[Any, bool]:
        if isinstance(value, str):
            sanitized = _sanitize_text(value, replacements, anonymizer)
            if PATH_KEY_RE.search(key_hint):
                normalized = sanitized.replace("\\", "/")
                unsafe_reference = (
                    "PRIVATE_PATH_REDACTED" in sanitized
                    or WINDOWS_ABSOLUTE_RE.search(sanitized) is not None
                    or UNC_RE.search(sanitized) is not None
                    or sanitized.startswith("/")
                    or re.search(r"(?i)^file://", sanitized) is not None
                    or any(part == ".." for part in PurePosixPath(normalized).parts)
                )
                if unsafe_reference and sanitized not in package_destinations:
                    sanitized = (
                        "PRIVATE_PATH_REDACTED__"
                        + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
                    )
            return sanitized, sanitized != value
        if isinstance(value, list):
            output: list[Any] = []
            changed = False
            for child in value:
                transformed, child_changed = transform(child, key_hint)
                output.append(transformed)
                changed = changed or child_changed
            return output, changed
        if isinstance(value, dict):
            output_object: dict[str, Any] = {}
            changed = False
            for original_key, child in value.items():
                sanitized_key = _sanitize_text(original_key, replacements, anonymizer)
                if "PRIVATE_PATH_REDACTED" in sanitized_key:
                    sanitized_key = (
                        sanitized_key
                        + "__"
                        + hashlib.sha256(original_key.encode("utf-8")).hexdigest()[:12]
                    )
                if sanitized_key in output_object:
                    sanitized_key = (
                        sanitized_key
                        + "__"
                        + hashlib.sha256(original_key.encode("utf-8")).hexdigest()[:12]
                    )
                transformed, child_changed = transform(child, original_key)
                output_object[sanitized_key] = transformed
                changed = changed or child_changed or sanitized_key != original_key
            return output_object, changed
        return value, False

    transformed, changed = transform(parsed)
    transformed, identity_changed = anonymizer.anonymize_json(transformed)
    changed = changed or identity_changed
    if not changed:
        return raw, False
    return _json_bytes(transformed), True


def _is_text_source(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    sample = path.read_bytes()[:8192]
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _csv_graph_bytes(
    plan: ArtifactPlan,
    source: Path,
    roots: Mapping[str, Path],
    source_to_destination: Mapping[tuple[str, str], str],
    replacements: Mapping[str, str],
    anonymizer: SubmissionIdentityAnonymizer,
) -> bytes:
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise BuildError(f"CSV graph node has no header: {plan.source_relpath}")
            fieldnames = list(reader.fieldnames)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise BuildError(f"cannot rewrite CSV graph node {plan.source_relpath}: {exc}") from exc
    for row in rows:
        for column in plan.csv_path_columns:
            value = row.get(column, "")
            if not value or value.strip().startswith("["):
                # JSON-list cells are sanitized textually; use one path per cell for
                # graph CSVs that need executable package references.
                row[column] = _sanitize_text(
                    value or "", replacements, anonymizer, wandb_context="wandb" in column.lower()
                )
                continue
            try:
                dependency = _resolve_dependency(
                    value,
                    plan.source_alias,
                    plan.source_relpath,
                    roots,
                    plan.csv_relative_to,
                    plan.legacy_root_markers,
                )
            except BuildError:
                row[column] = _sanitize_text(
                    value, replacements, anonymizer, wandb_context="wandb" in column.lower()
                )
                continue
            destination = source_to_destination.get((plan.source_alias, dependency))
            row[column] = destination if destination else "PRIVATE_PATH_REDACTED"
        for key in fieldnames:
            row[key] = _sanitize_text(
                row.get(key, "") or "",
                replacements,
                anonymizer,
                wandb_context="wandb" in key.lower(),
            )
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _artifact_bytes(
    plan: ArtifactPlan,
    source: Path,
    roots: Mapping[str, Path],
    source_to_destination: Mapping[tuple[str, str], str],
    replacements: Mapping[str, str],
    anonymizer: SubmissionIdentityAnonymizer,
) -> tuple[bytes, str]:
    raw = source.read_bytes()
    observed = _sha256_bytes(raw)
    if observed != plan.expected_sha256:
        raise BuildError(
            f"source anchor mismatch for {plan.source_alias}:{plan.source_relpath}: "
            f"expected {plan.expected_sha256}, observed {observed}"
        )
    if not _is_text_source(source):
        return raw, "binary_identity"
    if plan.csv_path_columns and source.suffix.lower() == ".csv":
        transformed = _csv_graph_bytes(
            plan, source, roots, source_to_destination, replacements, anonymizer
        )
        return transformed, "csv_graph_package_relative_rewrite"
    if source.suffix.lower() == ".json":
        transformed_json, changed = _portable_json_bytes(raw, replacements, anonymizer)
        return (
            transformed_json,
            "json_structured_portable_rewrite" if changed else "text_identity",
        )
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BuildError(f"selected text file is not UTF-8: {plan.source_relpath}") from exc
    sanitized = _sanitize_text(text, replacements, anonymizer).encode("utf-8")
    return sanitized, "text_identity" if sanitized == raw else "submission_sanitized"


def _hash_field(key: str) -> bool:
    normalized = key.lower()
    return "sha256" in normalized or normalized in {"hash", "digest"}


def _byte_field(key: str) -> bool:
    normalized = key.lower()
    return normalized in {"bytes", "size_bytes", "source_bytes", "package_bytes"} or (
        normalized.endswith("_bytes")
        and not normalized.endswith("_sha256_bytes")
    )


def _path_hash_bases(path_key: str) -> tuple[str, ...]:
    normalized = path_key.lower()
    stem = re.sub(
        r"(?i)(?:_?(?:path|file|artifact|manifest|csv|report|source))$",
        "",
        normalized,
    ).rstrip("_")
    return tuple(dict.fromkeys(value for value in (normalized, stem) if value))


def _companion_hash_keys(
    path_key: str, hash_keys: Sequence[str], allow_generic: bool
) -> tuple[str, ...]:
    """Bind a path to hash fields by schema-like names, never by digest value."""

    candidates: set[str] = set()
    for base in _path_hash_bases(path_key):
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


def _companion_byte_keys(
    path_key: str, byte_keys: Sequence[str], allow_generic: bool
) -> tuple[str, ...]:
    candidates: set[str] = set()
    for base in _path_hash_bases(path_key):
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


def _internal_reference(
    value: str,
    owner: ArtifactPlan,
    plans_by_package: Mapping[str, ArtifactPlan],
    plans_by_source: Mapping[tuple[str, str], ArtifactPlan],
) -> ArtifactPlan | None:
    """Resolve a selected source/package reference without consulting private roots."""

    stripped = value.strip().replace("\\", "/")
    if not stripped or "://" in stripped or "PRIVATE_PATH_REDACTED" in stripped:
        return None
    package_candidates = [stripped]
    package_candidates.append(
        posixpath.normpath(posixpath.join(posixpath.dirname(owner.package_path), stripped))
    )
    for candidate in package_candidates:
        normalized = candidate.removeprefix("./")
        target = plans_by_package.get(normalized)
        if target is not None:
            return target

    source_candidates = [stripped]
    source_candidates.append(
        posixpath.normpath(posixpath.join(posixpath.dirname(owner.source_relpath), stripped))
    )
    for candidate in source_candidates:
        normalized = candidate.removeprefix("./")
        target = plans_by_source.get((owner.source_alias, normalized))
        if target is not None:
            return target
    return None


def _json_links(
    value: Any,
    owner: ArtifactPlan,
    plans_by_package: Mapping[str, ArtifactPlan],
    plans_by_source: Mapping[tuple[str, str], ArtifactPlan],
) -> list[tuple[MutableMapping[str, Any], str, ArtifactPlan, tuple[str, ...]]]:
    links: list[tuple[MutableMapping[str, Any], str, ArtifactPlan, tuple[str, ...]]] = []
    if isinstance(value, MutableMapping):
        hash_keys = [str(key) for key in value if isinstance(key, str) and _hash_field(key)]
        internal_paths: list[tuple[str, ArtifactPlan]] = []
        for key, child in list(value.items()):
            if not isinstance(key, str) or not isinstance(child, str) or not PATH_KEY_RE.search(key):
                continue
            target = _internal_reference(
                child, owner, plans_by_package, plans_by_source
            )
            if target is not None:
                internal_paths.append((key, target))
        allow_generic = len(internal_paths) == 1
        for key, target in internal_paths:
            matching = _companion_hash_keys(key, hash_keys, allow_generic)
            if matching:
                links.append((value, key, target, matching))
        for child in list(value.values()):
            links.extend(
                _json_links(child, owner, plans_by_package, plans_by_source)
            )
    elif isinstance(value, list):
        for child in value:
            links.extend(
                _json_links(child, owner, plans_by_package, plans_by_source)
            )
    return links


def _csv_rows(payload: bytes, owner: ArtifactPlan) -> tuple[list[str], list[dict[str, str]]]:
    import io

    try:
        delimiter = "\t" if PurePosixPath(owner.package_path).suffix.lower() == ".tsv" else ","
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter=delimiter)
        if reader.fieldnames is None:
            raise BuildError(f"selected CSV has no header: {owner.source_relpath}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise BuildError(f"cannot parse packaged CSV {owner.source_relpath}: {exc}") from exc
    if any(None in row for row in rows):
        raise BuildError(f"selected CSV has extra cells: {owner.source_relpath}")
    return fieldnames, rows


def _csv_links(
    rows: Sequence[MutableMapping[str, str]],
    owner: ArtifactPlan,
    plans_by_package: Mapping[str, ArtifactPlan],
    plans_by_source: Mapping[tuple[str, str], ArtifactPlan],
) -> list[tuple[MutableMapping[str, str], str, ArtifactPlan, tuple[str, ...]]]:
    links: list[tuple[MutableMapping[str, str], str, ArtifactPlan, tuple[str, ...]]] = []
    configured_hashes = dict(owner.csv_hash_columns)
    for row in rows:
        hash_keys = [str(key) for key in row if isinstance(key, str) and _hash_field(key)]
        internal_paths: list[tuple[str, ArtifactPlan]] = []
        for key, value in list(row.items()):
            if not isinstance(key, str) or not isinstance(value, str) or not PATH_KEY_RE.search(key):
                continue
            target = _internal_reference(
                value, owner, plans_by_package, plans_by_source
            )
            if target is not None:
                internal_paths.append((key, target))
        allow_generic = len(internal_paths) == 1
        for key, target in internal_paths:
            preferred = configured_hashes.get(key, owner.csv_shared_hash_column)
            matching = list(_companion_hash_keys(key, hash_keys, allow_generic))
            if preferred and preferred in row:
                matching = [preferred] + [name for name in matching if name != preferred]
            if matching:
                links.append((row, key, target, tuple(matching)))
    return links


SHA256_SIDECAR_RE = re.compile(
    r"(?m)^(?P<digest>[0-9a-fA-F]{64})(?P<separator>[ \t]+\*?)(?P<path>\S+)"
    r"(?P<tail>[^\r\n]*)(?P<newline>\r?)$"
)


def _sidecar_links(
    payload: bytes,
    owner: ArtifactPlan,
    plans_by_package: Mapping[str, ArtifactPlan],
    plans_by_source: Mapping[tuple[str, str], ArtifactPlan],
) -> list[tuple[str, ArtifactPlan]]:
    if not owner.package_path.lower().endswith(".sha256"):
        return []
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BuildError(f"SHA-256 sidecar is not UTF-8: {owner.source_relpath}") from exc
    links: list[tuple[str, ArtifactPlan]] = []
    for match in SHA256_SIDECAR_RE.finditer(text):
        target = _internal_reference(
            match.group("path"), owner, plans_by_package, plans_by_source
        )
        if target is not None:
            links.append((match.group(0), target))
    return links


def _internal_dependencies(
    payload: bytes,
    owner: ArtifactPlan,
    plans_by_package: Mapping[str, ArtifactPlan],
    plans_by_source: Mapping[tuple[str, str], ArtifactPlan],
) -> set[str]:
    suffix = PurePosixPath(owner.package_path).suffix.lower()
    if suffix == ".json":
        try:
            value = json.loads(payload.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BuildError(f"packaged JSON is invalid: {owner.source_relpath}: {exc}") from exc
        return {
            target.package_path
            for _container, _path_key, target, _hash_keys in _json_links(
                value, owner, plans_by_package, plans_by_source
            )
        }
    if suffix in {".csv", ".tsv"}:
        _fieldnames, rows = _csv_rows(payload, owner)
        return {
            target.package_path
            for _row, _path_key, target, _hash_keys in _csv_links(
                rows, owner, plans_by_package, plans_by_source
            )
        }
    return {
        target.package_path
        for _line, target in _sidecar_links(
            payload, owner, plans_by_package, plans_by_source
        )
    }


def _set_dual_hash_fields(
    container: MutableMapping[str, Any],
    source_sha256: str,
    package_sha256: str,
    source_field: str = "source_sha256",
    package_field: str = "package_sha256",
) -> None:
    container[source_field] = source_sha256
    container[package_field] = package_sha256


def _per_reference_hash_fields(path_key: str, hash_keys: Sequence[str]) -> tuple[str, str]:
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
        base = re.sub(
            r"(?i)(?:_?(?:path|file|artifact|manifest|csv|report|source))$",
            "",
            path_key,
        )
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_") or "artifact"
    return f"{base}_source_sha256", f"{base}_package_sha256"


def _per_reference_byte_fields(path_key: str, byte_keys: Sequence[str]) -> tuple[str, str]:
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
        base = re.sub(
            r"(?i)(?:_?(?:path|file|artifact|manifest|csv|report|source))$",
            "",
            path_key,
        )
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_") or "artifact"
    return f"{base}_source_bytes", f"{base}_package_bytes"


def _rewrite_internal_links(
    payload: bytes,
    owner: ArtifactPlan,
    plans_by_package: Mapping[str, ArtifactPlan],
    plans_by_source: Mapping[tuple[str, str], ArtifactPlan],
    package_hashes: Mapping[str, str],
    source_sizes: Mapping[str, int],
    package_sizes: Mapping[str, int],
) -> tuple[bytes, bool]:
    suffix = PurePosixPath(owner.package_path).suffix.lower()
    if suffix == ".json":
        try:
            value = json.loads(payload.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BuildError(f"packaged JSON is invalid: {owner.source_relpath}: {exc}") from exc
        links = _json_links(value, owner, plans_by_package, plans_by_source)
        changed = False
        links_per_container: dict[int, int] = {}
        for container, _path_key, _target, _hash_keys in links:
            links_per_container[id(container)] = links_per_container.get(id(container), 0) + 1
        for container, path_key, target, hash_keys in links:
            package_sha256 = package_hashes[target.package_path]
            source_bytes = source_sizes[target.package_path]
            package_bytes = package_sizes[target.package_path]
            byte_keys = [str(key) for key in container if isinstance(key, str) and _byte_field(key)]
            companion_byte_keys = _companion_byte_keys(
                path_key, byte_keys, links_per_container[id(container)] == 1
            )
            if links_per_container[id(container)] == 1:
                source_field, package_field = "source_sha256", "package_sha256"
                source_bytes_field, package_bytes_field = "source_bytes", "package_bytes"
            else:
                source_field, package_field = _per_reference_hash_fields(
                    path_key, hash_keys
                )
                source_bytes_field, package_bytes_field = _per_reference_byte_fields(
                    path_key, companion_byte_keys
                )
            desired_hashes = {
                hash_key: (
                    target.expected_sha256
                    if hash_key.lower() == source_field.lower()
                    or hash_key.lower().endswith("_source_sha256")
                    else package_sha256
                )
                for hash_key in hash_keys
            }
            desired_sizes = {
                byte_key: (
                    source_bytes
                    if byte_key.lower() == source_bytes_field.lower()
                    or byte_key.lower().endswith("_source_bytes")
                    else package_bytes
                )
                for byte_key in companion_byte_keys
            }
            needs_rewrite = (
                container[path_key] != target.package_path
                or container.get(source_field) != target.expected_sha256
                or container.get(package_field) != package_sha256
                or any(container.get(key) != digest for key, digest in desired_hashes.items())
                or (
                    bool(companion_byte_keys)
                    and (
                        container.get(source_bytes_field) != source_bytes
                        or container.get(package_bytes_field) != package_bytes
                        or any(container.get(key) != size for key, size in desired_sizes.items())
                    )
                )
            )
            if not needs_rewrite:
                continue
            container[path_key] = target.package_path
            container.update(desired_hashes)
            _set_dual_hash_fields(
                container,
                target.expected_sha256,
                package_sha256,
                source_field,
                package_field,
            )
            if companion_byte_keys:
                container.update(desired_sizes)
                container[source_bytes_field] = source_bytes
                container[package_bytes_field] = package_bytes
            changed = True
        return (_json_bytes(value), True) if changed else (payload, False)

    if suffix in {".csv", ".tsv"}:
        import io

        fieldnames, rows = _csv_rows(payload, owner)
        links = _csv_links(rows, owner, plans_by_package, plans_by_source)
        changed = False
        links_per_row: dict[int, int] = {}
        for row, _path_key, _target, _hash_keys in links:
            links_per_row[id(row)] = links_per_row.get(id(row), 0) + 1
        for row, path_key, target, hash_keys in links:
            package_sha256 = package_hashes[target.package_path]
            source_bytes = source_sizes[target.package_path]
            package_bytes = package_sizes[target.package_path]
            byte_keys = [str(key) for key in row if isinstance(key, str) and _byte_field(key)]
            companion_byte_keys = _companion_byte_keys(
                path_key, byte_keys, links_per_row[id(row)] == 1
            )
            if links_per_row[id(row)] == 1:
                source_field, package_field = "source_sha256", "package_sha256"
                source_bytes_field, package_bytes_field = "source_bytes", "package_bytes"
            else:
                source_field, package_field = _per_reference_hash_fields(
                    path_key, hash_keys
                )
                source_bytes_field, package_bytes_field = _per_reference_byte_fields(
                    path_key, companion_byte_keys
                )
            desired_hashes = {
                hash_key: (
                    target.expected_sha256
                    if hash_key.lower() == source_field.lower()
                    or hash_key.lower().endswith("_source_sha256")
                    else package_sha256
                )
                for hash_key in hash_keys
            }
            desired_sizes = {
                byte_key: (
                    str(source_bytes)
                    if byte_key.lower() == source_bytes_field.lower()
                    or byte_key.lower().endswith("_source_bytes")
                    else str(package_bytes)
                )
                for byte_key in companion_byte_keys
            }
            needs_rewrite = (
                row[path_key] != target.package_path
                or row.get(source_field) != target.expected_sha256
                or row.get(package_field) != package_sha256
                or any(row.get(key) != digest for key, digest in desired_hashes.items())
                or (
                    bool(companion_byte_keys)
                    and (
                        row.get(source_bytes_field) != str(source_bytes)
                        or row.get(package_bytes_field) != str(package_bytes)
                        or any(row.get(key) != size for key, size in desired_sizes.items())
                    )
                )
            )
            if not needs_rewrite:
                continue
            row[path_key] = target.package_path
            row.update(desired_hashes)
            _set_dual_hash_fields(
                row,
                target.expected_sha256,
                package_sha256,
                source_field,
                package_field,
            )
            if companion_byte_keys:
                row.update(desired_sizes)
                row[source_bytes_field] = str(source_bytes)
                row[package_bytes_field] = str(package_bytes)
            for column in (
                source_field,
                package_field,
                *(
                    (source_bytes_field, package_bytes_field)
                    if companion_byte_keys
                    else ()
                ),
            ):
                if column not in fieldnames:
                    fieldnames.append(column)
            changed = True
        if not changed:
            return payload, False
        output = io.StringIO(newline="")
        delimiter = "\t" if suffix == ".tsv" else ","
        writer = csv.DictWriter(
            output, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue().encode("utf-8"), True

    sidecar_links = _sidecar_links(
        payload, owner, plans_by_package, plans_by_source
    )
    if not sidecar_links:
        return payload, False
    text = payload.decode("utf-8-sig")
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        target = _internal_reference(
            match.group("path"), owner, plans_by_package, plans_by_source
        )
        if target is None:
            return match.group(0)
        package_sha256 = package_hashes[target.package_path]
        path_changed = match.group("path") != target.package_path
        hash_changed = match.group("digest").lower() != package_sha256
        if not path_changed and not hash_changed:
            return match.group(0)
        changed = True
        return (
            package_sha256
            + match.group("separator")
            + target.package_path
            + match.group("tail")
            + match.group("newline")
        )

    transformed = SHA256_SIDECAR_RE.sub(replace, text)
    return (transformed.encode("utf-8"), True) if changed else (payload, False)


def _finalize_internal_hash_links(
    plans: Sequence[ArtifactPlan],
    payloads: Mapping[str, bytes],
    transformations: Mapping[str, str],
    source_sizes: Mapping[str, int],
) -> tuple[dict[str, bytes], dict[str, str]]:
    """Rewrite source hashes beside package paths after child payloads are final.

    A parent manifest's package hash depends on the final package hash of every
    sanitized child it names.  Resolve this graph leaf-first instead of trying
    to patch hashes in a single copy pass.  Cycles cannot have self-consistent
    cryptographic references and therefore fail closed.
    """

    plans_by_package = {plan.package_path: plan for plan in plans}
    plans_by_source = {
        (plan.source_alias, plan.source_relpath): plan for plan in plans
    }
    dependencies = {
        plan.package_path: _internal_dependencies(
            payloads[plan.package_path], plan, plans_by_package, plans_by_source
        )
        for plan in plans
    }
    finalized: dict[str, bytes] = {}
    final_transformations = dict(transformations)
    package_hashes: dict[str, str] = {}
    package_sizes: dict[str, int] = {}
    active: list[str] = []

    def visit(package_path: str) -> None:
        if package_path in finalized:
            return
        if package_path in active:
            cycle = " -> ".join(active[active.index(package_path) :] + [package_path])
            raise BuildError(f"cyclic internal package hash references: {cycle}")
        active.append(package_path)
        for dependency in sorted(dependencies[package_path]):
            visit(dependency)
        plan = plans_by_package[package_path]
        rewritten, changed = _rewrite_internal_links(
            payloads[package_path],
            plan,
            plans_by_package,
            plans_by_source,
            package_hashes,
            source_sizes,
            package_sizes,
        )
        finalized[package_path] = rewritten
        package_hashes[package_path] = _sha256_bytes(rewritten)
        package_sizes[package_path] = len(rewritten)
        if changed:
            original = final_transformations[package_path]
            final_transformations[package_path] = (
                "package_hash_links_rewritten"
                if original == "text_identity"
                else original + "+package_hash_links_rewritten"
            )
        active.pop()

    for plan in plans:
        visit(plan.package_path)
    return finalized, final_transformations


def _write_file(root: Path, relative: str, payload: bytes) -> None:
    destination = root / Path(*PurePosixPath(relative).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def _pending(selection: Mapping[str, Any]) -> tuple[list[Any], list[str]]:
    keys = [key for key in selection if key.lower().startswith("pending")]
    values = selection.get("pending_experiments", selection.get("pending_inputs", []))
    normalized: list[Any] = []
    if isinstance(values, list):
        for item in values:
            if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                normalized.append(item["id"])
            else:
                normalized.append(item)
    return normalized, keys


def _walk_pending_markers(value: Any, label: str = "selection") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_label = f"{label}.{key}"
            if "pending" in str(key).lower():
                findings.append(child_label)
            findings.extend(_walk_pending_markers(child, child_label))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_walk_pending_markers(child, f"{label}[{index}]"))
    elif isinstance(value, str) and re.search(
        r"(?i)pending.{0,80}(?:ex)?48|(?:ex)?48.{0,80}pending", value
    ):
        findings.append(label)
    return findings


def _assert_selection_portable(selection_raw: bytes) -> None:
    try:
        text = selection_raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BuildError("selection JSON must be UTF-8") from exc
    findings: list[str] = []
    if WINDOWS_ABSOLUTE_RE.search(text):
        findings.append("Windows absolute path")
    if UNC_RE.search(text):
        findings.append("UNC path")
    if PRIVATE_POSIX_RE.search(text):
        findings.append("private POSIX path")
    if re.search(r"(?i)file://", text):
        findings.append("file URI")
    if findings:
        raise BuildError(
            "selection must contain only source aliases and portable relative paths: "
            + ", ".join(findings)
        )


def _included_experiments(selection: Mapping[str, Any], plans: Sequence[ArtifactPlan]) -> list[str]:
    explicit = selection.get("included_experiments")
    values: set[str] = set()
    if isinstance(explicit, list):
        values.update(str(value) for value in explicit if str(value))
    values.update(plan.experiment_id for plan in plans if plan.experiment_id)
    for plan in plans:
        values.update(plan.experiment_ids)
    return sorted(values)


def _public_snapshot_paths(
    selection: Mapping[str, Any], plans: Sequence[ArtifactPlan]
) -> dict[str, str]:
    by_id = {plan.evidence_id: plan.package_path for plan in plans}
    result: dict[str, str] = {}
    inferred = {
        "public_catalog": by_id.get("public_experiment_catalog", ""),
        "accepted_result_anchors": by_id.get("accepted_result_anchors", ""),
    }
    configured = selection.get("public_snapshots", {})
    if configured and not isinstance(configured, Mapping):
        raise BuildError("selection.public_snapshots must be an object")
    aliases = {
        "public_catalog": ("public_catalog", "catalog"),
        "accepted_result_anchors": ("accepted_result_anchors", "anchors"),
    }
    for output_key, input_keys in aliases.items():
        value = ""
        if isinstance(configured, Mapping):
            for input_key in input_keys:
                configured_value = configured.get(input_key)
                if isinstance(configured_value, str) and configured_value:
                    value = _canonical_relative(
                        configured_value, f"public_snapshots.{input_key}"
                    )
                    break
        value = value or inferred[output_key]
        if value:
            if value not in {plan.package_path for plan in plans}:
                raise BuildError(f"public snapshot is not selected as an artifact: {value}")
            result[output_key] = value
    return result


def _accepted_ex48_in_snapshot(
    plans: Sequence[ArtifactPlan],
    roots: Mapping[str, Path],
    required_paths: set[str],
) -> bool:
    plan = next(
        (candidate for candidate in plans if candidate.evidence_id == "accepted_result_anchors"),
        None,
    )
    if plan is None:
        return False
    try:
        payload = json.loads(
            _source_file(roots, plan.source_alias, plan.source_relpath).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    records = payload.get("records", []) if isinstance(payload, Mapping) else []
    accepted_plans = [
        candidate
        for candidate in plans
        if EX48_EXPERIMENT_ID in candidate.experiment_ids
        and (
            candidate.integrity_status == "accepted"
            or candidate.integrity_status.startswith("accepted_")
        )
    ]
    if not isinstance(records, list):
        return False
    for record in records:
        if (
            not isinstance(record, Mapping)
            or record.get("accepted") is not True
            or record.get("experiment_id") != EX48_EXPERIMENT_ID
            or not isinstance(record.get("anchors"), list)
            or not record["anchors"]
        ):
            continue
        anchored_paths: set[str] = set()
        for anchor in record["anchors"]:
            if not isinstance(anchor, Mapping):
                return False
            path = anchor.get("path")
            digest = anchor.get("sha256")
            if (
                not isinstance(path, str)
                or not isinstance(digest, str)
                or not SHA256_RE.fullmatch(digest)
            ):
                return False
            matches: list[tuple[ArtifactPlan, str]] = []
            for candidate in accepted_plans:
                if path == candidate.source_relpath:
                    matches.append((candidate, "source"))
                elif path == candidate.package_path:
                    matches.append((candidate, "package"))
            if len(matches) != 1:
                return False
            target, semantics = matches[0]
            if semantics == "source" and digest != target.expected_sha256:
                return False
            anchored_paths.add(target.package_path)
        return required_paths.issubset(anchored_paths)
    return False


def _enforce_release_mode(selection: Mapping[str, Any], mode: str, plans: Sequence[ArtifactPlan]) -> list[Any]:
    pending, pending_keys = _pending(selection)
    included = _included_experiments(selection, plans)
    if mode == "draft":
        if not pending:
            raise BuildError("draft selection must declare non-empty pending_experiments")
    elif mode == "final":
        all_pending_markers = _walk_pending_markers(selection)
        if pending_keys or all_pending_markers:
            raise BuildError(
                "final selection must omit pending fields/EX48 pending markers entirely: "
                f"{sorted(set(pending_keys + all_pending_markers))}"
            )
        accepted_ex48_artifact = any(
            EX48_EXPERIMENT_ID in plan.experiment_ids
            and (
                plan.integrity_status == "accepted"
                or plan.integrity_status.startswith("accepted_")
            )
            for plan in plans
        )
        if EX48_EXPERIMENT_ID not in included or not accepted_ex48_artifact:
            raise BuildError(
                "final selection must include accepted EX48 artifacts for exact experiment "
                f"{EX48_EXPERIMENT_ID}"
            )
    else:
        raise BuildError("mode must be 'draft' or 'final'")
    return pending


def _ex48_final_gate(
    selection: Mapping[str, Any],
    plans: Sequence[ArtifactPlan],
    roots: Mapping[str, Path],
) -> dict[str, Any]:
    value = selection.get("ex48_final_gate")
    try:
        paths = _ex48_gate_paths(value)
    except ValidationError as exc:
        raise BuildError(str(exc)) from exc
    plans_by_package = {plan.package_path: plan for plan in plans}
    payloads: dict[str, Any] = {}
    for role, package_path in paths.items():
        plan = plans_by_package.get(package_path)
        if (
            plan is None
            or EX48_EXPERIMENT_ID not in plan.experiment_ids
            or not (
                plan.integrity_status == "accepted"
                or plan.integrity_status.startswith("accepted_")
            )
        ):
            raise BuildError(
                f"EX48 gate role {role} is not bound to an accepted EX48 artifact: "
                f"{package_path}"
            )
        try:
            payloads[role] = json.loads(
                _source_file(roots, plan.source_alias, plan.source_relpath).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BuildError(f"cannot read EX48 gate artifact {package_path}: {exc}") from exc
    try:
        _validate_ex48_certificate_payloads(payloads)
    except ValidationError as exc:
        raise BuildError(str(exc)) from exc
    return {"experiment_id": EX48_EXPERIMENT_ID, "artifacts": paths}


def _readme_bytes(
    mode: str,
    artifact_count: int,
    omission_count: int,
    pending: Sequence[Any],
) -> bytes:
    lines = [
        "# core-results",
        "",
        "This directory is the compact, portable evidence release for the paper.",
        f"It contains {artifact_count} explicitly selected artifacts plus integrity metadata.",
        "Every package reference is relative to this directory; source-machine roots are not retained.",
        "",
        "## Validation",
        "",
        "Run from this directory after copying or moving it:",
        "",
        "```text",
        "python tools/validate_core_results_package.py .",
        "```",
        "",
        "The validator checks the complete file inventory, SHA-256 hashes, JSON/CSV parsing,",
        "privacy and path portability, evidence statuses, and the release-mode gates.",
        "",
        "## Intentional compact-package omissions",
        "",
        f"{omission_count} byte-verified full-archive inputs are intentionally not bundled.",
        "Their source hashes, sizes, reasons, and compact anchor rows are registered in",
        "`provenance/omission_ledger.json`; omission never means missing evidence.",
        "",
        "## Release state",
        "",
    ]
    if mode == "draft":
        lines.extend(
            [
                "This is a draft package and is not the final complete release.",
                "Pending experiments: " + ", ".join(str(value) for value in pending) + ".",
                "No empty experiment directory is created for unavailable evidence.",
            ]
        )
    else:
        lines.extend(
            [
                "This is the final validated package.",
                "EX48 formal, endpoint-checkpoint, analysis, verification, and resume-lineage gates passed.",
            ]
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _safe_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.name != "core-results":
        raise BuildError("output directory name must be exactly 'core-results'")
    if resolved == Path(resolved.anchor):
        raise BuildError("refusing to use a filesystem root as output")
    return resolved


def _atomic_install(stage: Path, output: Path, replace: bool) -> None:
    if output.exists() and not replace:
        raise BuildError(f"output exists; pass --replace after review: {output}")
    backup = output.parent / f".core-results.backup-{os.getpid()}"
    if backup.exists():
        shutil.rmtree(backup)
    if output.exists():
        output.rename(backup)
    try:
        stage.rename(output)
    except Exception:
        if backup.exists() and not output.exists():
            backup.rename(output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def build_package(
    selection_path: Path | str,
    source_roots: Mapping[str, Path | str],
    output_path: Path | str,
    *,
    mode: str | None = None,
    replace: bool = False,
) -> Path:
    """Build, validate, and atomically install a ``core-results`` package."""

    selection_file = Path(selection_path).resolve()
    selection_raw = selection_file.read_bytes()
    _assert_selection_portable(selection_raw)
    selection_value = _load_json(selection_file)
    if not isinstance(selection_value, Mapping):
        raise BuildError("selection must be a JSON object")
    selection: Mapping[str, Any] = selection_value
    if selection.get("package_name", "core-results") != "core-results":
        raise BuildError("selection package_name must be exactly 'core-results'")
    roots = {alias: Path(value).resolve() for alias, value in source_roots.items()}
    for alias, root in roots.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", alias) or not root.is_dir():
            raise BuildError(f"invalid runtime source root: {alias}")
    if not roots:
        raise BuildError("at least one runtime source root is required")

    anchors = _anchor_map(selection)
    plans = _record_plans(selection, roots)
    plans.extend(
        _make_direct_plan(item, index, anchors)
        for index, item in enumerate(_direct_entries(selection))
    )
    for index, graph in enumerate(_graph_entries(selection)):
        plans.extend(_graph_plans(graph, index, roots, anchors))
    plans.sort(key=lambda plan: (plan.package_path, plan.evidence_id))
    _validate_plan(plans, roots)
    omissions = _parse_and_verify_omissions(selection, plans, roots)
    selected_mode = mode or str(selection.get("release_mode", "draft"))
    pending = _enforce_release_mode(selection, selected_mode, plans)
    final_gate = (
        _ex48_final_gate(selection, plans, roots)
        if selected_mode == "final"
        else None
    )
    public_snapshots = _public_snapshot_paths(selection, plans)
    if selected_mode == "final":
        if set(public_snapshots) != {"public_catalog", "accepted_result_anchors"}:
            raise BuildError("final release requires public catalog and accepted-anchor snapshots")
        required_paths = set(final_gate["artifacts"].values())
        if not _accepted_ex48_in_snapshot(plans, roots, required_paths):
            raise BuildError(
                "final release requires package-bound accepted EX48 anchors for every gate artifact"
            )

    output = _safe_output(Path(output_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".core-results.building-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=False)
    try:
        _write_file(stage, "provenance/release_selection.json", selection_raw)
        _write_file(
            stage,
            "provenance/omission_ledger.json",
            _omission_ledger_bytes(omissions),
        )
        replacements = _known_path_replacements(plans, roots)
        source_to_destination = {
            (plan.source_alias, plan.source_relpath): plan.package_path for plan in plans
        }
        anonymizer = SubmissionIdentityAnonymizer()
        prepared_payloads: dict[str, bytes] = {}
        prepared_transformations: dict[str, str] = {}
        for plan in plans:
            source = _source_file(roots, plan.source_alias, plan.source_relpath)
            payload, transformation = _artifact_bytes(
                plan, source, roots, source_to_destination, replacements, anonymizer
            )
            prepared_payloads[plan.package_path] = payload
            prepared_transformations[plan.package_path] = transformation

        source_sizes = {
            plan.package_path: _source_file(
                roots, plan.source_alias, plan.source_relpath
            ).stat().st_size
            for plan in plans
        }
        finalized_payloads, finalized_transformations = _finalize_internal_hash_links(
            plans, prepared_payloads, prepared_transformations, source_sizes
        )
        records: list[dict[str, Any]] = []
        for plan in plans:
            source = _source_file(roots, plan.source_alias, plan.source_relpath)
            payload = finalized_payloads[plan.package_path]
            transformation = finalized_transformations[plan.package_path]
            _write_file(stage, plan.package_path, payload)
            records.append(
                {
                    "claim_eligibility": plan.claim_eligibility,
                    "evidence_id": plan.evidence_id,
                    "experiment_id": plan.experiment_id,
                    "experiment_ids": list(plan.experiment_ids),
                    "integrity_status": plan.integrity_status,
                    "package_bytes": len(payload),
                    "package_path": plan.package_path,
                    "package_sha256": _sha256_bytes(payload),
                    "paper_role": plan.paper_role,
                    "scientific_status": plan.scientific_status,
                    "selection_method": plan.selection_method,
                    "source_hash_basis": plan.source_hash_basis,
                    "source_alias": plan.source_alias,
                    "source_bytes": source.stat().st_size,
                    "source_relpath": plan.source_relpath,
                    "source_sha256": plan.expected_sha256,
                    "transformation": transformation,
                    "tier": plan.tier,
                    "workstream": plan.workstream,
                }
            )

        artifact_manifest = {
            "artifacts": records,
            "package_name": "core-results",
            "schema_version": SCHEMA_VERSION,
        }
        _write_file(stage, "artifact_manifest.json", _json_bytes(artifact_manifest))

        import io

        evidence_columns = [
            "evidence_id",
            "experiment_id",
            "experiment_ids",
            "workstream",
            "integrity_status",
            "scientific_status",
            "claim_eligibility",
            "paper_role",
            "package_path",
            "package_sha256",
            "source_alias",
            "source_relpath",
            "source_sha256",
            "source_hash_basis",
            "transformation",
            "tier",
        ]
        evidence_output = io.StringIO(newline="")
        writer = csv.DictWriter(evidence_output, fieldnames=evidence_columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {column: record.get(column, "") for column in evidence_columns}
            row["experiment_ids"] = ";".join(record.get("experiment_ids", []))
            writer.writerow(row)
        _write_file(stage, "evidence_index.csv", evidence_output.getvalue().encode("utf-8"))

        _write_file(
            stage,
            "README.md",
            _readme_bytes(selected_mode, len(records), len(omissions), pending),
        )

        validator_source = Path(__file__).with_name("validate_core_results_package.py")
        _write_file(
            stage,
            "tools/validate_core_results_package.py",
            validator_source.read_bytes(),
        )
        release_manifest: dict[str, Any] = {
            "artifact_count": len(records),
            "anonymity_profile": SUBMISSION_ANONYMITY_PROFILE,
            "artifact_manifest": "artifact_manifest.json",
            "checksums": "SHA256SUMS",
            "evidence_index": "evidence_index.csv",
            "generated_files": sorted(GENERATED_FILES),
            "included_experiments": _included_experiments(selection, plans),
            "omission_count": len(omissions),
            "omission_ledger": "provenance/omission_ledger.json",
            "package_name": "core-results",
            "release_mode": selected_mode,
            "schema_version": SCHEMA_VERSION,
            "selection": "provenance/release_selection.json",
            "selection_sha256": _sha256_bytes(selection_raw),
            "validator": "tools/validate_core_results_package.py",
        }
        release_manifest.update(public_snapshots)
        if selected_mode == "draft":
            release_manifest["pending_experiments"] = pending
        else:
            release_manifest["ex48_final_gate"] = final_gate
        _write_file(stage, "release_manifest.json", _json_bytes(release_manifest))

        checksum_lines: list[str] = []
        for path in sorted(candidate for candidate in stage.rglob("*") if candidate.is_file()):
            relative = path.relative_to(stage).as_posix()
            checksum_lines.append(f"{_sha256_file(path)}  {relative}")
        _write_file(stage, "SHA256SUMS", ("\n".join(checksum_lines) + "\n").encode("utf-8"))

        try:
            validate_package(stage, public_source_root=roots.get("source"))
        except ValidationError as exc:
            raise BuildError(f"built package failed self-validation: {exc}") from exc
        _atomic_install(stage, output, replace)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return output


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, help="reviewed selection JSON")
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        metavar="ALIAS=PATH",
        help="runtime source root; repeat for each alias (never persisted)",
    )
    parser.add_argument("--mode", choices=("draft", "final"))
    parser.add_argument(
        "--output",
        default=str(Path.cwd().parent / "core-results"),
        help="output directory; basename must be exactly core-results",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="atomically replace an existing core-results directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        roots = _parse_source_roots(args.source_root)
        output = build_package(
            args.selection,
            roots,
            args.output,
            mode=args.mode,
            replace=args.replace,
        )
    except (BuildError, OSError) as exc:
        print(f"core-results build failed: {exc}", file=sys.stderr)
        return 2
    print(f"core-results built and validated: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
