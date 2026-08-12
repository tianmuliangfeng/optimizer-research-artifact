#!/usr/bin/env python3
"""Safe, metadata-driven experiment reproduction dispatcher.

The dispatcher intentionally does not know any experiment-specific command.
It discovers ``experiments/<id>/metadata.json`` below the release root and
uses the entrypoints declared there.  Every action is plan-first.  Executing a
plan requires the exact SHA-256 printed by the preceding plan as a receipt.

Only the Python 3.10 standard library is used so that archive inspection does
not depend on a training environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "selective_newton_muon_reproduction_plan_v1"
JSON_INDENT = 2
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RUNNABLE_SUFFIXES = {".bat", ".cmd", ".ps1", ".py", ".sh"}

# These variables are needed to locate the selected executable and its dynamic
# libraries on common Linux/Windows hosts.  They are copied into the printed
# plan and therefore covered by ``plan_sha256``.  No other ambient variable is
# inherited at execution time: experiment, GPU, data, W&B, and SNM settings
# must be supplied explicitly with ``--env KEY=VALUE`` (or by metadata).
RUNTIME_ENV_ALLOWLIST = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "HOME",
    "USERPROFILE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "PYTHONIOENCODING",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CONDA_PREFIX",
    "VIRTUAL_ENV",
    "NVIDIA_VISIBLE_DEVICES",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
)
DANGEROUS_EXPLICIT_ENV = {
    "BASH_ENV",
    "ENV",
    "LD_PRELOAD",
    "LD_AUDIT",
    "PYTHONPATH",
    "PYTHONHOME",
    "PROMPT_COMMAND",
    "SHELLOPTS",
    "BASHOPTS",
}
USER_ARGUMENT_FLAGS_WITHOUT_VALUES = {
    "--dry-run",
    "--preflight",
    "--numerical-smoke",
    "--formal-smoke",
    "--pilot",
    "--formal",
}


class ReproductionError(RuntimeError):
    """A user-facing contract or integrity failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def parse_env_assignments(values: Sequence[str] | None) -> dict[str, str]:
    """Parse explicit, receipt-bound ``KEY=VALUE`` assignments."""

    result: dict[str, str] = {}
    for raw in values or ():
        if "=" not in raw:
            raise ReproductionError(f"--env must use KEY=VALUE syntax: {raw!r}")
        key, value = raw.split("=", 1)
        if not ENV_NAME_RE.fullmatch(key):
            raise ReproductionError(f"invalid --env variable name: {key!r}")
        if key in DANGEROUS_EXPLICIT_ENV or key.startswith("DYLD_"):
            raise ReproductionError(
                f"--env variable can inject code outside the frozen plan: {key}"
            )
        if "\x00" in value:
            raise ReproductionError(f"--env value contains NUL: {key}")
        if key in result:
            raise ReproductionError(f"duplicate --env assignment: {key}")
        result[key] = value
    return result


def bound_runtime_environment() -> dict[str, str]:
    """Return the minimal ambient environment, fully recorded in the plan."""

    result = {
        key: os.environ[key]
        for key in RUNTIME_ENV_ALLOWLIST
        if key in os.environ
    }
    result["PYTHONNOUSERSITE"] = "1"
    return result


def missing_user_argument_groups(
    requirements: Sequence[Mapping[str, Sequence[str]]],
    user_args: Sequence[str],
) -> list[dict[str, list[str]]]:
    def present(option: str) -> bool:
        for index, token in enumerate(user_args):
            if token.startswith(option + "=") and token != option + "=":
                return True
            if token != option:
                continue
            if option in USER_ARGUMENT_FLAGS_WITHOUT_VALUES:
                return True
            if index + 1 < len(user_args) and not user_args[index + 1].startswith("--"):
                return True
        return False

    missing: list[dict[str, list[str]]] = []
    for group in requirements:
        kind, options = next(iter(group.items()))
        satisfied = (
            all(present(option) for option in options)
            if kind == "all_of"
            else any(present(option) for option in options)
        )
        if not satisfied:
            missing.append({kind: list(options)})
    return missing


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolved_child(path: Path, root: Path, *, label: str) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    if resolved == resolved_root or not is_relative_to(resolved, resolved_root):
        raise ReproductionError(
            f"{label} must be a strict child of {resolved_root}: {resolved}"
        )
    return resolved


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReproductionError(f"invalid JSON {path}: {exc}") from exc


def iter_files(root: Path) -> Iterable[Path]:
    """Yield regular files without following directory symlinks."""

    if not root.is_dir():
        return
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name
            for name in directories
            if not (current_path / name).is_symlink()
        )
        for name in sorted(files):
            path = current_path / name
            if path.is_file() and not path.is_symlink():
                yield path


def tree_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for path in iter_files(root):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    rows.sort(key=lambda row: row["path"])
    return rows, hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def experiments_root(release_root: Path) -> Path:
    return release_root.expanduser().resolve() / "experiments"


def discover_experiment_paths(release_root: Path) -> list[Path]:
    root = experiments_root(release_root)
    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and not path.is_symlink()
        ),
        key=lambda path: path.name,
    )


def load_metadata(experiment_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    metadata_path = experiment_dir / "metadata.json"
    errors: list[str] = []
    if not metadata_path.is_file():
        return None, ["missing metadata.json"]
    try:
        value = read_json(metadata_path)
    except ReproductionError as exc:
        return None, [str(exc)]
    if not isinstance(value, dict):
        return None, ["metadata.json must contain a JSON object"]
    experiment_id = value.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        errors.append("metadata experiment_id must be a non-empty string")
    elif experiment_id != experiment_dir.name:
        errors.append(
            f"metadata experiment_id {experiment_id!r} does not match directory "
            f"{experiment_dir.name!r}"
        )
    status = value.get("status")
    if not isinstance(status, str) or not status.strip():
        errors.append("metadata status must be a non-empty string")
    entrypoints = value.get("entrypoints", {})
    if not isinstance(entrypoints, (dict, list)):
        errors.append("metadata entrypoints must be an object or array")
    native_modes = value.get("native_modes", {})
    if native_modes is not None and not isinstance(native_modes, dict):
        errors.append("metadata native_modes must be an object")
    default_entrypoint = value.get("default_entrypoint")
    if default_entrypoint is not None and (
        not isinstance(default_entrypoint, str) or not default_entrypoint
    ):
        errors.append("metadata default_entrypoint must be a non-empty string")
    legacy_roots = value.get("legacy_result_roots", [])
    if not isinstance(legacy_roots, list) or not all(
        isinstance(item, str) and item.strip() for item in legacy_roots
    ):
        errors.append("metadata legacy_result_roots must be an array of paths")
    else:
        for item in legacy_roots:
            normalized = Path(item.replace("\\", "/"))
            if (
                normalized.is_absolute()
                or ".." in normalized.parts
                or normalized == Path(".")
            ):
                errors.append(
                    "metadata legacy_result_roots must contain safe relative paths"
                )
                break
    return value, errors


def auto_entrypoints(
    release_root: Path, experiment_dir: Path
) -> dict[str, dict[str, Any]]:
    command_root = release_root / "commands" / experiment_dir.name
    result: dict[str, dict[str, Any]] = {}
    if not command_root.is_dir():
        return result
    for path in iter_files(command_root):
        if path.suffix.lower() not in RUNNABLE_SUFFIXES:
            continue
        relative = path.relative_to(release_root).as_posix()
        base = path.relative_to(command_root).with_suffix("").as_posix()
        name = base.replace("/", "__")
        if name in result:
            raise ReproductionError(f"auto-discovered duplicate entrypoint {name!r}")
        result[name] = {"path": relative, "args": [], "native_modes": {}}
    return result


def normalize_mode_map(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ReproductionError(f"{label} must be an object")
    result: dict[str, Any] = {}
    for name, spec in value.items():
        if not isinstance(name, str) or not name:
            raise ReproductionError(f"{label} contains an invalid mode name")
        if isinstance(spec, str):
            result[name] = {"args": [spec]}
        elif isinstance(spec, list):
            result[name] = {"args": spec}
        elif isinstance(spec, dict):
            result[name] = dict(spec)
        elif spec is True:
            result[name] = {"args": []}
        else:
            raise ReproductionError(
                f"{label}.{name} must be a string, array, object, or true"
            )
    return result


def normalize_required_user_arguments(value: Any, *, label: str) -> list[dict[str, list[str]]]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise ReproductionError(f"{label} must be an array")
    result: list[dict[str, list[str]]] = []
    for index, raw_group in enumerate(value):
        if not isinstance(raw_group, dict) or len(raw_group) != 1:
            raise ReproductionError(
                f"{label}[{index}] must contain exactly one all_of/one_of group"
            )
        kind, options = next(iter(raw_group.items()))
        if kind not in {"all_of", "one_of"}:
            raise ReproductionError(f"{label}[{index}] has invalid group {kind!r}")
        if not isinstance(options, list) or not options or not all(
            isinstance(option, str) and option.startswith("--") for option in options
        ):
            raise ReproductionError(f"{label}[{index}] options must be CLI flags")
        result.append({kind: list(options)})
    return result


def normalize_entrypoints(
    release_root: Path, experiment_dir: Path, metadata: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    raw = metadata.get("entrypoints")
    if raw in (None, {}, []):
        return auto_entrypoints(release_root, experiment_dir)

    pairs: list[tuple[str, Any]] = []
    if isinstance(raw, dict):
        pairs = list(raw.items())
    elif isinstance(raw, list):
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ReproductionError(f"entrypoints[{index}] must be an object")
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise ReproductionError(f"entrypoints[{index}] has no valid name")
            pairs.append((name, item))
    else:
        raise ReproductionError("metadata entrypoints must be an object or array")

    result: dict[str, dict[str, Any]] = {}
    for name, raw_spec in pairs:
        if not isinstance(name, str) or not name:
            raise ReproductionError("entrypoint names must be non-empty strings")
        if name in result:
            raise ReproductionError(f"duplicate entrypoint name {name!r}")
        if isinstance(raw_spec, str):
            spec: dict[str, Any] = {"path": raw_spec}
        elif isinstance(raw_spec, dict):
            spec = dict(raw_spec)
        else:
            raise ReproductionError(f"entrypoint {name!r} must be a path or object")
        relative = spec.get("path")
        if not isinstance(relative, str) or not relative:
            raise ReproductionError(f"entrypoint {name!r} has no valid path")
        args = spec.get("args", [])
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise ReproductionError(f"entrypoint {name!r} args must be strings")
        env = spec.get("env", {})
        if not isinstance(env, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in env.items()
        ):
            raise ReproductionError(f"entrypoint {name!r} env must map strings to strings")
        result[name] = {
            "path": relative,
            "args": list(args),
            "env": dict(env),
            "native_modes": normalize_mode_map(
                spec.get("native_modes", {}),
                label=f"entrypoints.{name}.native_modes",
            ),
            "required_user_arguments": normalize_required_user_arguments(
                spec.get("required_user_arguments", []),
                label=f"entrypoints.{name}.required_user_arguments",
            ),
        }
    return result


def metadata_code_dir(
    release_root: Path, experiment_dir: Path, metadata: Mapping[str, Any] | None
) -> Path:
    relative = metadata.get("code_directory") if metadata is not None else None
    if isinstance(relative, str) and relative:
        candidate = (release_root / relative).resolve()
        allowed = (release_root / "scripts").resolve()
        if is_relative_to(candidate, allowed):
            return candidate
    return (release_root / "scripts" / experiment_dir.name).resolve()


def experiment_record(release_root: Path, experiment_dir: Path) -> dict[str, Any]:
    metadata, metadata_errors = load_metadata(experiment_dir)
    code_dir = metadata_code_dir(release_root, experiment_dir, metadata)
    commands_dir = release_root / "commands" / experiment_dir.name
    entrypoints: dict[str, dict[str, Any]] = {}
    entrypoint_error: str | None = None
    if metadata is not None:
        try:
            entrypoints = normalize_entrypoints(release_root, experiment_dir, metadata)
        except ReproductionError as exc:
            entrypoint_error = str(exc)
            metadata_errors.append(entrypoint_error)

    has_code = code_dir.is_dir() and any(iter_files(code_dir))
    has_commands = commands_dir.is_dir() and any(iter_files(commands_dir))
    if metadata is not None and isinstance(metadata.get("status"), str):
        status = str(metadata["status"])
    elif not has_code and not has_commands:
        status = "planned_placeholder"
    else:
        status = "invalid_metadata"
    native_mode_names = {
        mode
        for entrypoint in entrypoints.values()
        for mode in entrypoint.get("native_modes", {})
    }
    if metadata is not None:
        try:
            native_mode_names.update(
                normalize_mode_map(
                    metadata.get("native_modes", {}), label="native_modes"
                )
            )
        except ReproductionError as exc:
            if str(exc) not in metadata_errors:
                metadata_errors.append(str(exc))
    native_modes = sorted(native_mode_names)
    return {
        "experiment_id": experiment_dir.name,
        "path": str(experiment_dir),
        "status": status,
        "metadata_present": metadata is not None,
        "metadata_valid": metadata is not None and not metadata_errors,
        "metadata_errors": metadata_errors,
        "code_present": has_code,
        "commands_present": has_commands,
        "entrypoints": sorted(entrypoints),
        "native_modes": native_modes,
    }


def find_experiment(release_root: Path, experiment_id: str) -> Path:
    candidates = discover_experiment_paths(release_root)
    by_name = {path.name: path for path in candidates}
    if experiment_id in by_name:
        return by_name[experiment_id]
    metadata_matches: list[Path] = []
    for path in candidates:
        metadata, _errors = load_metadata(path)
        if metadata is not None and metadata.get("experiment_id") == experiment_id:
            metadata_matches.append(path)
    if len(metadata_matches) == 1:
        return metadata_matches[0]
    if len(metadata_matches) > 1:
        raise ReproductionError(f"ambiguous experiment_id {experiment_id!r}")
    raise ReproductionError(f"unknown experiment_id {experiment_id!r}")


def validated_experiment(
    release_root: Path, experiment_id: str
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    experiment_dir = find_experiment(release_root, experiment_id)
    metadata, errors = load_metadata(experiment_dir)
    if metadata is None or errors:
        raise ReproductionError(
            f"experiment {experiment_dir.name} metadata is invalid: {errors}"
        )
    if metadata.get("status") in {"planned_placeholder", "planned_not_implemented"}:
        raise ReproductionError(f"experiment {experiment_dir.name} is a planned placeholder")
    entrypoints = normalize_entrypoints(release_root, experiment_dir, metadata)
    if not entrypoints:
        raise ReproductionError(f"experiment {experiment_dir.name} has no entrypoints")
    return experiment_dir, metadata, entrypoints


def validate_entrypoint_path(release_root: Path, relative: str) -> Path:
    candidate = (release_root / relative).resolve()
    allowed_roots = (
        (release_root / "commands").resolve(),
        (release_root / "scripts").resolve(),
    )
    if not any(is_relative_to(candidate, root) for root in allowed_roots):
        raise ReproductionError(
            "entrypoint must stay under release commands/ or scripts/: "
            f"{candidate}"
        )
    if not candidate.is_file():
        raise ReproductionError(f"entrypoint is missing: {candidate}")
    return candidate


def native_mode_spec(
    metadata: Mapping[str, Any],
    entrypoints: Mapping[str, Mapping[str, Any]],
    action: str,
    requested_entrypoint: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    aliases = (action, action.replace("-", "_"))
    top_modes = normalize_mode_map(metadata.get("native_modes", {}), label="native_modes")
    for alias in aliases:
        if alias in top_modes:
            spec = dict(top_modes[alias])
            selected = spec.get("entrypoint")
            if selected is not None and not isinstance(selected, str):
                raise ReproductionError(f"native_modes.{alias}.entrypoint must be a string")
            if requested_entrypoint and selected and requested_entrypoint != selected:
                raise ReproductionError(
                    f"requested entrypoint {requested_entrypoint!r} conflicts with "
                    f"native mode entrypoint {selected!r}"
                )
            return requested_entrypoint or selected, spec

    if requested_entrypoint is not None:
        if requested_entrypoint not in entrypoints:
            raise ReproductionError(f"unknown entrypoint {requested_entrypoint!r}")
        modes = entrypoints[requested_entrypoint].get("native_modes", {})
        for alias in aliases:
            if alias in modes:
                return requested_entrypoint, dict(modes[alias])
    else:
        providers: list[tuple[str, dict[str, Any]]] = []
        for entrypoint_name, entrypoint in entrypoints.items():
            modes = entrypoint.get("native_modes", {})
            for alias in aliases:
                if alias in modes:
                    providers.append((entrypoint_name, dict(modes[alias])))
                    break
        if len(providers) == 1:
            return providers[0]
    return requested_entrypoint, None


def experiment_source_inventory(
    release_root: Path,
    experiment_dir: Path,
    metadata: Mapping[str, Any],
    entrypoints: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Hash the metadata, declared code directory, and entrypoint files."""

    paths: set[Path] = {(experiment_dir / "metadata.json").resolve()}
    code_dir = metadata_code_dir(release_root, experiment_dir, metadata)
    if code_dir.is_dir():
        paths.update(path.resolve() for path in iter_files(code_dir))
    for entrypoint in entrypoints.values():
        relative = entrypoint.get("path")
        if isinstance(relative, str) and relative:
            candidate = (release_root / relative).resolve()
            if candidate.is_file() and not candidate.is_symlink():
                paths.add(candidate)
    rows = [
        {
            "path": path.relative_to(release_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths, key=lambda item: item.as_posix())
    ]
    return rows, hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def select_entrypoint(
    entrypoints: Mapping[str, Mapping[str, Any]], requested: str | None
) -> tuple[str, Mapping[str, Any]]:
    if requested is not None:
        if requested not in entrypoints:
            raise ReproductionError(
                f"unknown entrypoint {requested!r}; available={sorted(entrypoints)}"
            )
        return requested, entrypoints[requested]
    if len(entrypoints) != 1:
        raise ReproductionError(
            "multiple entrypoints require --entrypoint; available="
            + ",".join(sorted(entrypoints))
        )
    name = next(iter(entrypoints))
    return name, entrypoints[name]


def render(value: str, variables: Mapping[str, str]) -> str:
    result = value
    for key, replacement in variables.items():
        result = result.replace("{" + key + "}", replacement)
    unresolved = re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", result)
    if unresolved:
        raise ReproductionError(f"unresolved template values in {value!r}: {unresolved}")
    return result


def invocation_prefix(entrypoint: Path) -> list[str]:
    suffix = entrypoint.suffix.lower()
    if suffix == ".py":
        return [sys.executable, str(entrypoint)]
    if suffix == ".sh":
        return ["bash", str(entrypoint)]
    if suffix == ".ps1":
        executable = "powershell.exe" if os.name == "nt" else "pwsh"
        return [executable, "-NoProfile", "-File", str(entrypoint)]
    if suffix in {".bat", ".cmd"}:
        return ["cmd.exe", "/c", str(entrypoint)]
    return [str(entrypoint)]


def build_action_plan(
    release_root: Path,
    experiment_id: str,
    action: str,
    *,
    entrypoint_name: str | None = None,
    run_dir: Path | None = None,
    results_root: Path | None = None,
    env_overrides: Mapping[str, str] | None = None,
    user_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    experiment_dir, metadata, entrypoints = validated_experiment(
        release_root, experiment_id
    )
    mode_spec: dict[str, Any] | None = None
    selected_hint = entrypoint_name
    if action == "reproduce":
        reproducibility = metadata.get("reproducibility", {})
        if not isinstance(reproducibility, dict) or reproducibility.get(
            "fresh_rerun"
        ) is False:
            raise ReproductionError(
                f"experiment {experiment_dir.name} does not declare a fresh rerun"
            )
        if selected_hint is None:
            default_entrypoint = metadata.get("default_entrypoint")
            if isinstance(default_entrypoint, str) and default_entrypoint:
                selected_hint = default_entrypoint
    if action in {"resume", "native-verify"}:
        selected_hint, mode_spec = native_mode_spec(
            metadata, entrypoints, action, entrypoint_name
        )
        if mode_spec is None:
            raise ReproductionError(
                f"experiment {experiment_dir.name} does not declare native mode {action!r}"
            )
    elif action == "reproduce":
        selected_hint, mode_spec = native_mode_spec(
            metadata, entrypoints, "reproduce", selected_hint
        )

    selected_name, entrypoint = select_entrypoint(entrypoints, selected_hint)
    executable = validate_entrypoint_path(release_root, str(entrypoint["path"]))
    receipt_user_args = list(user_args or ())
    if not all(
        isinstance(item, str) and item and "\x00" not in item
        for item in receipt_user_args
    ):
        raise ReproductionError("--arg values must be non-empty strings without NUL")

    checked_run_dir: Path | None = None
    if action in {"resume", "native-verify"}:
        if run_dir is None or results_root is None:
            raise ReproductionError(f"{action} requires --run-dir and --results-root")
        checked_run_dir = resolved_child(run_dir, results_root, label="run-dir")
        if not checked_run_dir.is_dir():
            raise ReproductionError(f"run-dir does not exist: {checked_run_dir}")
        if metadata.get("legacy_result_roots") and matched_legacy_result_root(
            checked_run_dir, results_root.expanduser().resolve(), metadata
        ) is None:
            raise ReproductionError(
                f"{action} run-dir does not match an experiment result root"
            )

    variables = {
        "experiment_dir": str(experiment_dir),
        "release_root": str(release_root.expanduser().resolve()),
        "run_dir": str(checked_run_dir) if checked_run_dir is not None else "",
        "results_root": str(results_root.expanduser().resolve())
        if results_root is not None
        else "",
    }
    args = [render(item, variables) for item in entrypoint.get("args", [])]
    declared_env = {
        key: render(value, variables)
        for key, value in entrypoint.get("env", {}).items()
    }
    if mode_spec is not None:
        mode_args = mode_spec.get("args", [])
        if not isinstance(mode_args, list) or not all(
            isinstance(item, str) for item in mode_args
        ):
            raise ReproductionError(f"native mode {action!r} args must be strings")
        args.extend(render(item, variables) for item in mode_args)
        mode_env = mode_spec.get("env", {})
        if not isinstance(mode_env, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in mode_env.items()
        ):
            raise ReproductionError(f"native mode {action!r} env must map strings to strings")
        declared_env.update(
            {key: render(value, variables) for key, value in mode_env.items()}
        )
    args.extend(receipt_user_args)
    required_user_arguments = entrypoint.get("required_user_arguments", [])
    missing_user_arguments = missing_user_argument_groups(
        required_user_arguments, receipt_user_args
    )

    explicit_env = dict(env_overrides or {})
    for key, value in explicit_env.items():
        if not isinstance(key, str) or not ENV_NAME_RE.fullmatch(key):
            raise ReproductionError(f"invalid explicit environment name: {key!r}")
        if key in DANGEROUS_EXPLICIT_ENV or key.startswith("DYLD_"):
            raise ReproductionError(
                f"explicit environment can inject code outside the frozen plan: {key}"
            )
        if not isinstance(value, str) or "\x00" in value:
            raise ReproductionError(f"invalid explicit environment value: {key}")
        if key in declared_env and declared_env[key] != value:
            raise ReproductionError(
                f"--env cannot override metadata-bound environment variable {key}"
            )

    required_env: list[str] = []
    required_env_files: list[str] = []
    if mode_spec is not None:
        raw_required = mode_spec.get("required_env", [])
        if not isinstance(raw_required, list) or not all(
            isinstance(item, str) and ENV_NAME_RE.fullmatch(item)
            for item in raw_required
        ):
            raise ReproductionError(f"native mode {action!r} required_env is invalid")
        required_env = list(raw_required)
        raw_required_files = mode_spec.get("required_env_files", [])
        if not isinstance(raw_required_files, list) or not all(
            isinstance(item, str) and ENV_NAME_RE.fullmatch(item)
            for item in raw_required_files
        ):
            raise ReproductionError(
                f"native mode {action!r} required_env_files is invalid"
            )
        required_env_files = list(raw_required_files)

    external_input_files: list[dict[str, Any]] = []
    for key in required_env_files:
        raw_path = explicit_env.get(key, declared_env.get(key, ""))
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser().resolve()
        if not candidate.is_file() or candidate.is_symlink():
            raise ReproductionError(
                f"native mode {action!r} requires a regular file for --env {key}"
            )
        explicit_env[key] = str(candidate)
        external_input_files.append(
            {
                "environment_key": key,
                "path": str(candidate),
                "bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )

    runtime_env = bound_runtime_environment()
    environment = dict(runtime_env)
    environment.update(declared_env)
    environment.update(explicit_env)
    missing_required = [key for key in required_env if not environment.get(key)]
    if missing_required:
        raise ReproductionError(
            f"native mode {action!r} requires explicit --env assignments: "
            + ",".join(missing_required)
        )

    inventory, tree_sha256 = experiment_source_inventory(
        release_root, experiment_dir, metadata, entrypoints
    )
    base_plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "action": action,
        "experiment_id": experiment_dir.name,
        "experiment_status": metadata["status"],
        "entrypoint": selected_name,
        "entrypoint_relative_path": executable.relative_to(release_root).as_posix(),
        "entrypoint_sha256": sha256_file(executable),
        "command": invocation_prefix(executable) + args,
        "user_arguments": receipt_user_args,
        "required_user_arguments": required_user_arguments,
        "missing_required_user_arguments": missing_user_arguments,
        "environment": dict(sorted(environment.items())),
        "environment_policy": "receipt_bound_allowlist_v1",
        "runtime_environment_keys": sorted(runtime_env),
        "explicit_environment_keys": sorted(explicit_env),
        "external_input_files": sorted(
            external_input_files, key=lambda row: row["environment_key"]
        ),
        # Artifact launchers and Python runners are authored relative to the
        # release root, while their absolute entrypoint is still recorded in
        # the command.  Keeping one cwd avoids entrypoint-dependent behavior.
        "working_directory": str(release_root.expanduser().resolve()),
        "run_dir": str(checked_run_dir) if checked_run_dir is not None else None,
        "source_file_count": len(inventory),
        "source_tree_sha256": tree_sha256,
        "metadata_sha256": sha256_file(experiment_dir / "metadata.json"),
    }
    base_plan["plan_sha256"] = hashlib.sha256(
        canonical_json_bytes(base_plan)
    ).hexdigest()
    return base_plan


def manifest_lineage_matches(
    payload: Mapping[str, Any],
    *,
    experiment_id: str | None,
    metadata: Mapping[str, Any] | None,
) -> bool:
    """Bind a sealed snapshot to an experiment, not merely to valid hashes."""

    if experiment_id is None or metadata is None:
        return True
    explicit_ids = [payload.get("experiment_id"), payload.get("experiment")]
    lineage = payload.get("lineage")
    if isinstance(lineage, dict):
        explicit_ids.extend(
            [lineage.get("experiment_id"), lineage.get("experiment")]
        )
    if experiment_id in explicit_ids:
        return True

    code_directory = metadata.get("code_directory")
    raw_mapping = payload.get("file_sha256")
    if raw_mapping is None:
        raw_mapping = payload.get("files")
    if isinstance(code_directory, str) and code_directory and isinstance(
        raw_mapping, dict
    ):
        marker = code_directory.replace("\\", "/").strip("/")
        for raw in raw_mapping:
            if not isinstance(raw, str):
                continue
            normalized = raw.replace("\\", "/").strip("/")
            if normalized == marker or normalized.startswith(marker + "/"):
                return True
    return False


def matched_legacy_result_root(
    run_dir: Path,
    results_root: Path,
    metadata: Mapping[str, Any] | None,
) -> str | None:
    if metadata is None:
        return None
    relative_parts = tuple(
        os.path.normcase(part) for part in run_dir.relative_to(results_root).parts
    )
    raw_roots = metadata.get("legacy_result_roots", [])
    if not isinstance(raw_roots, list):
        return None
    for raw in raw_roots:
        if not isinstance(raw, str) or not raw:
            continue
        candidate = Path(raw.replace("\\", "/"))
        if candidate.is_absolute() or ".." in candidate.parts:
            continue
        candidate_parts = tuple(
            os.path.normcase(part) for part in candidate.parts if part != "."
        )
        if (
            candidate_parts
            and len(relative_parts) > len(candidate_parts)
            and relative_parts[: len(candidate_parts)] == candidate_parts
        ):
            return raw
    return None


def validate_snapshot_manifest(
    path: Path,
    run_dir: Path,
    *,
    experiment_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {
            "path": path.relative_to(run_dir).as_posix(),
            "passed": False,
            "errors": ["manifest is not an object"],
            "checked_files": 0,
        }
    raw_mapping: Any = payload.get("file_sha256")
    mapping_kind = "file_sha256"
    if raw_mapping is None:
        raw_mapping = payload.get("files")
        mapping_kind = "files"
    errors: list[str] = []
    checked = 0
    expected_inventory: set[str] = set()
    if payload.get("passed") is False:
        errors.append("manifest explicitly reports passed=false")
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        errors.append("missing non-empty file_sha256/files mapping")
    else:
        snapshot_root = path.parent.resolve()
        for relative, raw_expected in sorted(raw_mapping.items()):
            if not isinstance(relative, str) or not relative:
                errors.append("invalid relative file key")
                continue
            expected_inventory.add(Path(relative).as_posix())
            expected: Any = raw_expected
            expected_bytes: int | None = None
            if isinstance(raw_expected, dict):
                expected = raw_expected.get("sha256")
                if isinstance(raw_expected.get("bytes"), int):
                    expected_bytes = int(raw_expected["bytes"])
            if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
                errors.append(f"invalid SHA-256 for {relative}")
                continue
            candidate = (path.parent / relative).resolve()
            if not is_relative_to(candidate, snapshot_root):
                errors.append(f"snapshot path escapes manifest directory: {relative}")
                continue
            if not candidate.is_file() or candidate.is_symlink():
                errors.append(f"missing snapshot file: {relative}")
                continue
            checked += 1
            if expected_bytes is not None and candidate.stat().st_size != expected_bytes:
                errors.append(f"byte-size mismatch: {relative}")
            if sha256_file(candidate) != expected:
                errors.append(f"SHA-256 mismatch: {relative}")
        observed_inventory = {
            item.relative_to(snapshot_root).as_posix()
            for item in iter_files(snapshot_root)
            if item.resolve() != path.resolve()
        }
        if observed_inventory != expected_inventory:
            errors.append(
                "snapshot inventory mismatch: "
                f"missing={sorted(expected_inventory - observed_inventory)},"
                f"extra={sorted(observed_inventory - expected_inventory)}"
            )
        symlinks: list[str] = []
        for current, directories, files in os.walk(snapshot_root, followlinks=False):
            current_path = Path(current)
            for name in [*directories, *files]:
                candidate = current_path / name
                if candidate.is_symlink():
                    symlinks.append(candidate.relative_to(snapshot_root).as_posix())
        if symlinks:
            errors.append(f"snapshot contains symlinks: {sorted(symlinks)}")
    lineage_match = manifest_lineage_matches(
        payload, experiment_id=experiment_id, metadata=metadata
    )
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "mapping": mapping_kind,
        "checked_files": checked,
        "passed": not errors,
        "experiment_lineage_match": lineage_match,
        "errors": errors,
    }


def verify_run(
    run_dir: Path,
    results_root: Path,
    *,
    experiment_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checked_run_dir = resolved_child(run_dir, results_root, label="run-dir")
    checked_results_root = results_root.expanduser().resolve()
    if not checked_run_dir.is_dir():
        raise ReproductionError(f"run-dir does not exist: {checked_run_dir}")

    json_errors: list[dict[str, str]] = []
    json_count = 0
    snapshots: list[dict[str, Any]] = []
    for path in iter_files(checked_run_dir):
        if path.suffix.lower() != ".json":
            continue
        json_count += 1
        try:
            read_json(path)
        except ReproductionError as exc:
            json_errors.append(
                {
                    "path": path.relative_to(checked_run_dir).as_posix(),
                    "error": str(exc),
                }
            )
            continue
        if path.name == "source_snapshot_manifest.json":
            snapshots.append(
                validate_snapshot_manifest(
                    path,
                    checked_run_dir,
                    experiment_id=experiment_id,
                    metadata=metadata,
                )
            )
    matched_root = matched_legacy_result_root(
        checked_run_dir, checked_results_root, metadata
    )
    snapshots_valid = bool(snapshots) and all(row["passed"] for row in snapshots)
    snapshot_lineage = any(
        row["passed"] and row["experiment_lineage_match"] for row in snapshots
    )
    lineage_bound = matched_root is not None or snapshot_lineage
    reproducibility = metadata.get("reproducibility", {}) if metadata else {}
    sealed_snapshot_declared = (
        isinstance(reproducibility, dict)
        and reproducibility.get("source_freeze") == "sealed_source_snapshot"
    )
    sealed_snapshot_required = sealed_snapshot_declared and matched_root is None
    # A declared legacy-result-root match is an explicit alternative lineage
    # for compact accepted archives that intentionally omit the bulky frozen
    # snapshot.  If a snapshot is present, however, every manifest is a hard
    # integrity gate.
    snapshot_integrity = not snapshots or snapshots_valid
    checks = {
        "run_dir_within_results_root": True,
        "json_present": json_count > 0,
        "all_json_parse": not json_errors,
        # False when absent: do not report the vacuous truth of all([]).
        "source_snapshot_manifest_present": bool(snapshots),
        "source_snapshot_manifests": snapshots_valid,
        "sealed_snapshot_declared": sealed_snapshot_declared,
        "sealed_snapshot_required": sealed_snapshot_required,
        "sealed_snapshot_requirement_satisfied": (
            not sealed_snapshot_required or snapshots_valid or matched_root is not None
        ),
        "legacy_result_root_match": matched_root is not None,
        "experiment_lineage_bound": lineage_bound,
    }
    passed = (
        checks["run_dir_within_results_root"]
        and checks["json_present"]
        and checks["all_json_parse"]
        and snapshot_integrity
        and checks["experiment_lineage_bound"]
    )
    return {
        "schema_version": "selective_newton_muon_archive_verify_v1",
        "run_dir": str(checked_run_dir),
        "results_root": str(checked_results_root),
        "passed": passed,
        "checks": checks,
        "json_file_count": json_count,
        "json_errors": json_errors,
        "source_snapshot_manifest_count": len(snapshots),
        "source_snapshot_manifests": snapshots,
        "lineage": {
            "experiment_id": experiment_id,
            "matched_legacy_result_root": matched_root,
            "snapshot_manifest_match": snapshot_lineage,
        },
        "verification_scope": (
            "recursive JSON syntax and sealed source-snapshot hashes; "
            "experiment-native scientific gates require native-verify"
        ),
    }


def inspect_experiment(release_root: Path, experiment_id: str) -> dict[str, Any]:
    experiment_dir = find_experiment(release_root, experiment_id)
    record = experiment_record(release_root, experiment_dir)
    metadata, errors = load_metadata(experiment_dir)
    entrypoints: dict[str, dict[str, Any]] = {}
    if metadata is not None and not errors:
        entrypoints = normalize_entrypoints(release_root, experiment_dir, metadata)
        inventory, tree_sha256 = experiment_source_inventory(
            release_root, experiment_dir, metadata, entrypoints
        )
    else:
        inventory, tree_sha256 = tree_inventory(experiment_dir)
    record.update(
        {
            "source_file_count": len(inventory),
            "source_tree_sha256": tree_sha256,
            "files": inventory,
        }
    )
    return record


def execute_plan(plan: Mapping[str, Any], receipt: str | None) -> int:
    expected = str(plan["plan_sha256"])
    canonical_plan = dict(plan)
    canonical_plan.pop("plan_sha256", None)
    observed = hashlib.sha256(canonical_json_bytes(canonical_plan)).hexdigest()
    if observed != expected:
        raise ReproductionError(
            "plan contents no longer match plan_sha256; rebuild and inspect the plan"
        )
    if receipt is None:
        raise ReproductionError(
            "--execute requires --receipt equal to the printed plan_sha256"
        )
    if receipt.lower() != expected:
        raise ReproductionError(
            f"receipt mismatch: expected plan_sha256 {expected}, observed {receipt}"
        )
    missing_user_arguments = plan.get("missing_required_user_arguments", [])
    if missing_user_arguments:
        raise ReproductionError(
            "execution requires receipt-bound --arg values for: "
            + json.dumps(missing_user_arguments, sort_keys=True)
        )
    # Do not merge the ambient process environment here.  Every variable made
    # visible to the child was printed in the plan and is covered by the
    # receipt; unrecorded SNM/experiment variables therefore cannot silently
    # redirect data, GPUs, W&B, interpreters, or result paths.
    environment = {
        str(key): str(value) for key, value in plan["environment"].items()
    }
    completed = subprocess.run(
        [str(value) for value in plan["command"]],
        cwd=str(plan["working_directory"]),
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    default_release = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Inspect and dispatch Selective Newton-Muon artifact reproductions."
    )
    parser.add_argument("--release-root", type=Path, default=default_release)
    subparsers = parser.add_subparsers(dest="action", required=True)

    subparsers.add_parser("list")
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("experiment_id")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("experiment_id")
    verify_parser.add_argument("--run-dir", type=Path, required=True)
    verify_parser.add_argument("--results-root", type=Path, required=True)

    for action in ("reproduce", "resume", "native-verify"):
        action_parser = subparsers.add_parser(action)
        action_parser.add_argument("experiment_id")
        action_parser.add_argument("--entrypoint")
        action_parser.add_argument(
            "--env",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help=(
                "receipt-bound child environment assignment; repeat for each "
                "SNM, experiment, GPU, data, or logging setting"
            ),
        )
        action_parser.add_argument(
            "--arg",
            action="append",
            default=[],
            metavar="VALUE",
            help=(
                "receipt-bound argument forwarded to the selected entrypoint; "
                "use --arg=--flag when VALUE begins with '--'"
            ),
        )
        if action != "reproduce":
            action_parser.add_argument("--run-dir", type=Path, required=True)
            action_parser.add_argument("--results-root", type=Path, required=True)
        action_parser.add_argument("--execute", action="store_true")
        action_parser.add_argument("--receipt")
    return parser


def print_json(value: Any, *, stream: Any | None = None) -> None:
    # Resolve stdout at call time so embedding applications and unit tests can
    # safely redirect it.  A default argument would retain the import-time
    # stream object instead.
    if stream is None:
        stream = sys.stdout
    print(
        json.dumps(value, ensure_ascii=False, indent=JSON_INDENT, sort_keys=True),
        file=stream,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        release_root = args.release_root.expanduser().resolve()
        if args.action == "list":
            payload = {
                "release_root": str(release_root),
                "experiments": [
                    experiment_record(release_root, path)
                    for path in discover_experiment_paths(release_root)
                ],
            }
            print_json(payload)
            return 0
        if args.action == "inspect":
            print_json(inspect_experiment(release_root, args.experiment_id))
            return 0
        if args.action == "verify":
            experiment_dir = find_experiment(release_root, args.experiment_id)
            metadata, metadata_errors = load_metadata(experiment_dir)
            if metadata is None or metadata_errors:
                raise ReproductionError(
                    f"experiment {experiment_dir.name} metadata is invalid: "
                    f"{metadata_errors}"
                )
            if metadata.get("status") in {
                "planned_not_implemented",
                "planned_placeholder",
            }:
                raise ReproductionError(
                    f"experiment {experiment_dir.name} is planned and has no "
                    "verifiable accepted result"
                )
            payload = verify_run(
                args.run_dir,
                args.results_root,
                experiment_id=experiment_dir.name,
                metadata=metadata,
            )
            payload["experiment_id"] = args.experiment_id
            print_json(payload)
            return 0 if payload["passed"] else 2

        run_dir = getattr(args, "run_dir", None)
        results_root = getattr(args, "results_root", None)
        plan = build_action_plan(
            release_root,
            args.experiment_id,
            args.action,
            entrypoint_name=args.entrypoint,
            run_dir=run_dir,
            results_root=results_root,
            env_overrides=parse_env_assignments(getattr(args, "env", [])),
            user_args=getattr(args, "arg", []),
        )
        print_json(plan)
        if not args.execute:
            if args.receipt is not None:
                raise ReproductionError("--receipt is only valid together with --execute")
            return 0
        return execute_plan(plan, args.receipt)
    except ReproductionError as exc:
        print_json(
            {"passed": False, "error": str(exc), "action": args.action},
            stream=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
