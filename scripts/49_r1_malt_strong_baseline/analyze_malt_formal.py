"""Build the sealed ten-method R1 MALT-family panel for Experiment 49.

The analyzer accepts only explicit inputs: three MALT and three MALTER-Eq17
formal summaries, their six formal manifests, and the accepted Experiment-45
eight-method summary and analysis manifest, all bound to one independently
issued dual-method selection certificate.  It never discovers or mutates
historical evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


SEEDS = (2024, 2025, 2026)
EXPERIMENT45_METHODS = (
    "diag",
    "none",
    "mousse",
    "muon",
    "block4",
    "moonlight",
    "normuon",
    "adamw",
)
FORMAL_METHODS = ("malt", "malter_eq17")
METHOD_ORDER = (*EXPERIMENT45_METHODS, *FORMAL_METHODS)
DISPLAY = {
    "diag": "Newton-Muon diag",
    "none": "Newton-Muon none",
    "mousse": "Mousse-R1",
    "muon": "Muon",
    "block4": "Newton-Muon block4",
    "moonlight": "Moonlight Muon",
    "normuon": "NorMuon",
    "adamw": "AdamW",
    "malt": "MALT-R1 adaptation",
    "malter_eq17": "MALTER-Eq17-R1 adaptation",
}
MALT_METHOD_LABEL = "MALT-R1 adaptation"
MALTER_METHOD_LABEL = "MALTER-Eq17-R1 adaptation"
FORMAL_METHOD_LABELS = {
    "malt": MALT_METHOD_LABEL,
    "malter_eq17": MALTER_METHOD_LABEL,
}
MALT_FAMILY = "49_r1_malt_strong_baseline"
MALT_FORMAL_PROTOCOL = "malt_r1_selected_6200step_v4"
SELECTION_PROTOCOL = "malt_r1_focused_grid_selection_v4"
MALT_ACCEPTED_STATUSES = {
    "completed_valid",
    "completed_valid_local_wandb_incomplete",
}
EXPERIMENT45_PROTOCOL = "mousse_r1_unified_analysis_v1"
FORMAL_STEPS = 6200
TOKENS_PER_STEP = 512 * 1024
FORMAL_TOKENS = FORMAL_STEPS * TOKENS_PER_STEP
T_CRIT_DF2 = 4.302652729911275
PRACTICAL_MARGIN = 0.002
EXPECTED_RUNTIME = {
    "python": "3.10.12",
    "torch": "2.8.0+cu126",
    "torch_cuda": "12.6",
    "triton": "3.4.0",
    "numpy": "2.2.6",
}
BASELINE_CONTRASTS = (
    ("muon", "muon", "anchor"),
    ("original_newton_muon", "block4", "anchor"),
    ("selective_none", "none", "primary"),
    ("selective_diag", "diag", "primary"),
    ("mousse", "mousse", "external_curvature_baseline"),
)
CONTRASTS = tuple(
    (f"{method}_minus_{suffix}", method, baseline, role)
    for method in FORMAL_METHODS
    for suffix, baseline, role in BASELINE_CONTRASTS
) + (("malt_minus_malter_eq17", "malt", "malter_eq17", "family_internal"),)
NUMERIC_FIELDS = (
    "initial_val_loss",
    "final_val_loss",
    "best_val_loss",
    "tail5_val_loss_mean",
    "normalized_val_auc",
)
SUMMARY_INTEGER_FIELDS = (
    "peak_memory_allocated_mib",
    "hidden_optimizer_state_bytes",
    "total_optimizer_state_bytes",
    "auxiliary_optimizer_state_bytes",
    "optimizer_state_bytes",
    "model_parameter_bytes",
    "malt_momentum_bytes",
    "malt_row_ema_bytes",
    "malt_col_ema_bytes",
    "malt_nu_bytes",
)
EXPECTED_HIDDEN_STATE_BYTES = {
    "malt": 340_402_176,
    "malter_eq17": 340_402_464,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--malt-summaries",
        type=Path,
        nargs=3,
        required=True,
        metavar=("SEED_SUMMARY_1", "SEED_SUMMARY_2", "SEED_SUMMARY_3"),
    )
    parser.add_argument(
        "--malt-manifests",
        type=Path,
        nargs=3,
        required=True,
        metavar=("SEED_MANIFEST_1", "SEED_MANIFEST_2", "SEED_MANIFEST_3"),
    )
    parser.add_argument(
        "--malter-summaries",
        type=Path,
        nargs=3,
        required=True,
        metavar=("SEED_SUMMARY_1", "SEED_SUMMARY_2", "SEED_SUMMARY_3"),
    )
    parser.add_argument(
        "--malter-manifests",
        type=Path,
        nargs=3,
        required=True,
        metavar=("SEED_MANIFEST_1", "SEED_MANIFEST_2", "SEED_MANIFEST_3"),
    )
    parser.add_argument("--selection-certificate", type=Path, required=True)
    parser.add_argument("--experiment45-summary", type=Path, required=True)
    parser.add_argument(
        "--experiment45-analysis-manifest", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_input(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"{label} is not a file: {resolved}")
    return resolved


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty CSV: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise RuntimeError(f"CSV rows have inconsistent schemas: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def required_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be an integer, observed boolean")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be an integer: {value!r}") from exc
    return parsed


def required_float(value: object, label: str) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise RuntimeError(f"{label} must be finite: {parsed!r}")
    return parsed


def parse_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise RuntimeError(f"{label} must be boolean: {value!r}")


def required_sha256(value: object, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise RuntimeError(f"{label} must be a lowercase-or-uppercase SHA-256 hex digest")
    return normalized


def optimizer_state_mib(row: dict[str, Any], label: str) -> float:
    value = row.get("optimizer_state_mib")
    if value not in (None, ""):
        return required_float(value, f"{label}.optimizer_state_mib")
    byte_value = row.get("optimizer_state_bytes")
    if byte_value in (None, ""):
        raise RuntimeError(f"{label} has no optimizer-state measurement")
    return required_float(byte_value, f"{label}.optimizer_state_bytes") / 1024**2


def peak_memory_mib(row: dict[str, Any], label: str) -> float:
    for key in ("peak_memory_mib", "peak_memory_allocated_mib"):
        if row.get(key) not in (None, ""):
            return required_float(row[key], f"{label}.{key}")
    raise RuntimeError(f"{label} has no peak-memory measurement")


def normalized_row(
    method: str,
    seed: int,
    row: dict[str, Any],
    family: str,
    adaptation_label: str = "",
) -> dict[str, object]:
    label = f"{method}/seed{seed}"
    values = {
        field: required_float(row.get(field), f"{label}.{field}")
        for field in NUMERIC_FIELDS
    }
    checkpoint_bytes = row.get("checkpoint_bytes")
    return {
        "method": method,
        "display_name": DISPLAY[method],
        "adaptation_label": adaptation_label,
        "family": family,
        "run_name": str(row.get("run_name", "")),
        "seed": seed,
        "seed_role": "tuning_seed" if seed == 2026 else "confirmatory_seed",
        "init_sha256": str(row.get("init_sha256", "")),
        "checkpoint_bytes": (
            required_int(checkpoint_bytes, f"{label}.checkpoint_bytes")
            if checkpoint_bytes not in (None, "")
            else 0
        ),
        "checkpoint_sha256": str(row.get("checkpoint_sha256", "")),
        **values,
        "peak_memory_mib": peak_memory_mib(row, label),
        "optimizer_state_mib": optimizer_state_mib(row, label),
        "timing_eligible": False,
    }


def validate_experiment45(
    summary_path: Path, manifest_path: Path
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    manifest = read_json_object(manifest_path, "Experiment-45 analysis manifest")
    if manifest.get("status") != "completed_valid":
        raise RuntimeError("Experiment-45 analysis manifest is not accepted")
    if manifest.get("protocol") != EXPERIMENT45_PROTOCOL:
        raise RuntimeError("Experiment-45 analysis protocol mismatch")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or summary_path.name not in outputs:
        raise RuntimeError(
            "Experiment-45 manifest does not declare the supplied eight-method summary"
        )
    if not isinstance(manifest.get("source_files"), list) or not manifest["source_files"]:
        raise RuntimeError("Experiment-45 accepted source ledger is missing")

    raw_rows = read_csv(summary_path)
    if len(raw_rows) != len(EXPERIMENT45_METHODS) * len(SEEDS):
        raise RuntimeError(
            "Experiment-45 summary must contain exactly 24 method/seed rows"
        )
    observed: set[tuple[str, int]] = set()
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(raw_rows):
        method = str(raw.get("method", ""))
        if method not in EXPERIMENT45_METHODS:
            raise RuntimeError(f"unexpected Experiment-45 method: {method!r}")
        seed = required_int(raw.get("seed"), f"Experiment-45 row {index}.seed")
        if seed not in SEEDS:
            raise RuntimeError(f"unexpected Experiment-45 seed: {seed}")
        pair = (method, seed)
        if pair in observed:
            raise RuntimeError(f"duplicate Experiment-45 method/seed row: {pair}")
        observed.add(pair)
        if parse_bool(
            raw.get("timing_eligible", False), f"Experiment-45 {pair}.timing_eligible"
        ):
            raise RuntimeError("Experiment-45 quality timing must remain ineligible")
        rows.append(
            normalized_row(
                method,
                seed,
                raw,
                str(raw.get("family") or "experiment45_frozen"),
            )
        )
    expected = {(method, seed) for method in EXPERIMENT45_METHODS for seed in SEEDS}
    if observed != expected:
        raise RuntimeError(
            f"Experiment-45 coverage mismatch: missing={sorted(expected - observed)}"
        )
    return rows, manifest


def validate_selection_certificate(
    path: Path,
) -> tuple[dict[str, Any], str]:
    payload = read_json_object(path, "dual-method selection certificate")
    failures: list[str] = []
    if payload.get("status") != "selected":
        failures.append("status is not selected")
    if payload.get("protocol") != SELECTION_PROTOCOL:
        failures.append("protocol mismatch")
    if payload.get("certificate_role") != "independent_pilot_analysis_selection":
        failures.append("certificate was not issued by the independent pilot analyzer")
    if payload.get("scientific_result") != "dual_methods_selected":
        failures.append("scientific result is not dual_methods_selected")
    if payload.get("formal_allowed") is not True:
        failures.append("dual-method formal gate is closed")
    if payload.get("required_formal_methods") != list(FORMAL_METHODS):
        failures.append("required formal methods are not exactly MALT and MALTER-Eq17")
    if required_int(payload.get("seed"), "selection.seed") != 2026:
        failures.append("selection seed is not 2026")
    if required_int(payload.get("pilot_steps"), "selection.pilot_steps") != 1000:
        failures.append("selection pilot length is not 1000 steps")
    try:
        required_sha256(
            payload.get("pilot_manifest_sha256"),
            "selection.pilot_manifest_sha256",
        )
    except RuntimeError as exc:
        failures.append(str(exc))

    selections = payload.get("selections")
    if not isinstance(selections, dict) or set(selections) != set(FORMAL_METHODS):
        failures.append("selections do not contain exactly MALT and MALTER-Eq17")
        selections = {}
    for method in FORMAL_METHODS:
        entry = selections.get(method)
        if not isinstance(entry, dict):
            failures.append(f"selection entry is missing for {method}")
            continue
        if (
            entry.get("method") != method
            or entry.get("status") != "selected"
            or entry.get("formal_allowed") is not True
            or entry.get("formal_eligible") is not True
            or entry.get("boundary_rule_triggered") is not False
        ):
            failures.append(f"selection entry is not formal-eligible for {method}")
        if not str(entry.get("selected_cell_id", "")):
            failures.append(f"selected cell is missing for {method}")
        try:
            required_float(
                entry.get("selected_matrix_lr"),
                f"selection.selections.{method}.selected_matrix_lr",
            )
        except RuntimeError as exc:
            failures.append(str(exc))
    if failures:
        raise RuntimeError("selection certificate rejected: " + "; ".join(failures))
    return payload, sha256_file(path)


def _manifest_method_label(
    method: str,
    manifest: dict[str, Any],
    embedded: dict[str, Any],
    raw: dict[str, str],
) -> str:
    expected = FORMAL_METHOD_LABELS[method]
    required_labels = {
        "CSV": raw.get("adaptation_label"),
        "manifest summary": embedded.get("adaptation_label"),
    }
    for source_name, value in required_labels.items():
        if value in (None, ""):
            raise RuntimeError(
                f"{DISPLAY[method]} {source_name} does not persist the required "
                f"method label {expected!r}"
            )
        if str(value) != expected:
            raise RuntimeError(
                f"{DISPLAY[method]} method-label mismatch in {source_name}: {value!r}"
            )
    manifest_label = manifest.get("adaptation_label")
    if manifest_label not in (None, "") and str(manifest_label) != expected:
        raise RuntimeError(
            f"{DISPLAY[method]} method-label mismatch in formal manifest: "
            f"{manifest_label!r}"
        )
    return expected


def validate_formal_inputs(
    method: str,
    summary_paths: list[Path],
    manifest_paths: list[Path],
    selection_payload: dict[str, Any],
    selection_sha256: str,
) -> tuple[
    list[dict[str, object]],
    list[tuple[int, Path, Path, dict[str, Any]]],
    list[str],
]:
    if method not in FORMAL_METHODS:
        raise RuntimeError(f"unsupported Experiment-49 formal method: {method!r}")
    method_name = DISPLAY[method]
    if len(summary_paths) != len(SEEDS) or len(manifest_paths) != len(SEEDS):
        raise RuntimeError(f"{method_name} requires exactly three summaries and manifests")
    summary_by_seed: dict[int, tuple[Path, dict[str, str]]] = {}
    for path in summary_paths:
        rows = read_csv(path)
        if len(rows) != 1:
            raise RuntimeError(f"{method_name} formal summary must contain one row: {path}")
        row = rows[0]
        seed = required_int(
            row.get("controlled_seed") or row.get("seed"), f"{path}.seed"
        )
        if seed not in SEEDS:
            raise RuntimeError(f"unexpected {method_name} formal seed: {seed}")
        if seed in summary_by_seed:
            raise RuntimeError(f"duplicate {method_name} formal summary for seed {seed}")
        summary_by_seed[seed] = (path, row)

    manifest_by_seed: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in manifest_paths:
        manifest = read_json_object(path, f"{method_name} formal manifest")
        seed = required_int(manifest.get("seed"), f"{path}.seed")
        if seed not in SEEDS:
            raise RuntimeError(
                f"unexpected {method_name} formal manifest seed: {seed}"
            )
        if seed in manifest_by_seed:
            raise RuntimeError(f"duplicate {method_name} formal manifest for seed {seed}")
        manifest_by_seed[seed] = (path, manifest)

    if set(summary_by_seed) != set(SEEDS) or set(manifest_by_seed) != set(SEEDS):
        raise RuntimeError(
            f"{method_name} formal seed coverage must be exactly 2024/2025/2026"
        )

    normalized: list[dict[str, object]] = []
    evidence: list[tuple[int, Path, Path, dict[str, Any]]] = []
    wandb_caveats: list[str] = []
    source_fingerprints: set[str] = set()
    runtime_fingerprints: set[str] = set()
    data_fingerprints: set[str] = set()
    for seed in SEEDS:
        summary_path, raw = summary_by_seed[seed]
        manifest_path, manifest = manifest_by_seed[seed]
        status = manifest.get("status")
        if status not in MALT_ACCEPTED_STATUSES:
            raise RuntimeError(
                f"{method_name} seed {seed} formal manifest is not locally accepted: "
                f"{status!r}"
            )
        if status == "completed_valid_local_wandb_incomplete":
            wandb_caveats.append(
                f"{method} seed {seed}: W&B upload incomplete; local evidence accepted"
            )
        if manifest.get("family") != MALT_FAMILY:
            raise RuntimeError(f"{method_name} seed {seed} experiment-family mismatch")
        if manifest.get("protocol") != MALT_FORMAL_PROTOCOL:
            raise RuntimeError(f"{method_name} seed {seed} formal protocol mismatch")
        if manifest.get("batch_kind") != "formal":
            raise RuntimeError(f"{method_name} seed {seed} is not a formal batch")
        if manifest.get("formal_evidence") is not True:
            raise RuntimeError(f"{method_name} seed {seed} is not marked as formal evidence")
        if required_int(manifest.get("total_steps"), "manifest.total_steps") != FORMAL_STEPS:
            raise RuntimeError(f"{method_name} seed {seed} formal step budget mismatch")
        if required_int(manifest.get("total_tokens"), "manifest.total_tokens") != FORMAL_TOKENS:
            raise RuntimeError(f"{method_name} seed {seed} formal token budget mismatch")
        if manifest.get("timing_eligible") is not False:
            raise RuntimeError(f"{method_name} seed {seed} quality timing must be ineligible")
        embedded = manifest.get("summary")
        if not isinstance(embedded, dict):
            raise RuntimeError(f"{method_name} seed {seed} has no embedded accepted summary")
        if embedded.get("evidence_valid") is not True:
            raise RuntimeError(
                f"{method_name} seed {seed} embedded summary is not evidence-valid"
            )
        method_label = _manifest_method_label(method, manifest, embedded, raw)

        cell = manifest.get("cell")
        if not isinstance(cell, dict) or cell.get("method") != method:
            raise RuntimeError(f"{method_name} seed {seed} formal cell method mismatch")
        selected = selection_payload["selections"][method]
        selected_cell_id = str(selected["selected_cell_id"])
        selected_matrix_lr = required_float(
            selected["selected_matrix_lr"],
            f"selection.selections.{method}.selected_matrix_lr",
        )
        if (
            str(cell.get("cell_id", "")) != selected_cell_id
            or required_float(cell.get("matrix_lr"), f"{method_name}.cell.matrix_lr")
            != selected_matrix_lr
        ):
            raise RuntimeError(
                f"{method_name} seed {seed} formal cell/LR does not match selection"
            )
        embedded_selection = manifest.get("selection_certificate")
        if (
            not isinstance(embedded_selection, dict)
            or str(embedded_selection.get("sha256", "")).lower() != selection_sha256
            or embedded_selection.get("validated_selected_method") != method
            or embedded_selection.get("protocol") != selection_payload["protocol"]
            or embedded_selection.get("scientific_result")
            != selection_payload["scientific_result"]
            or embedded_selection.get("required_formal_methods")
            != selection_payload["required_formal_methods"]
            or embedded_selection.get("selections") != selection_payload["selections"]
            or any(
                embedded_selection.get(key) != value
                for key, value in selection_payload.items()
            )
        ):
            raise RuntimeError(
                f"{method_name} seed {seed} embedded selection certificate/seal mismatch"
            )

        for source_name, source in (("CSV", raw), ("manifest summary", embedded)):
            if source.get("method") != method:
                raise RuntimeError(
                    f"{method_name} seed {seed} {source_name} method must be {method!r}"
                )
            source_seed = required_int(
                source.get("controlled_seed") or source.get("seed"),
                f"{method_name} seed {seed} {source_name}.seed",
            )
            if source_seed != seed:
                raise RuntimeError(f"{method_name} seed {seed} {source_name} seed mismatch")
            if required_int(source.get("total_steps"), "summary.total_steps") != FORMAL_STEPS:
                raise RuntimeError(
                    f"{method_name} seed {seed} {source_name} step budget mismatch"
                )
            if required_int(source.get("total_tokens"), "summary.total_tokens") != FORMAL_TOKENS:
                raise RuntimeError(
                    f"{method_name} seed {seed} {source_name} token budget mismatch"
                )
            if source.get("evidence_profile") != MALT_FORMAL_PROTOCOL:
                raise RuntimeError(
                    f"{method_name} seed {seed} {source_name} evidence-profile mismatch"
                )
            if not parse_bool(
                source.get("formal_evidence"),
                f"{method_name} seed {seed} {source_name}.formal_evidence",
            ):
                raise RuntimeError(
                    f"{method_name} seed {seed} {source_name} is not formal evidence"
                )
            if not parse_bool(
                source.get("evidence_valid"),
                f"{method_name} seed {seed} {source_name}.evidence_valid",
            ):
                raise RuntimeError(
                    f"{method_name} seed {seed} {source_name} is not evidence-valid"
                )
            if parse_bool(
                source.get("timing_eligible"),
                f"{method_name} seed {seed} {source_name}.timing_eligible",
            ):
                raise RuntimeError(
                    f"{method_name} seed {seed} {source_name} timing must be ineligible"
                )

        for field in NUMERIC_FIELDS:
            csv_value = required_float(raw.get(field), f"{summary_path}.{field}")
            embedded_value = required_float(
                embedded.get(field), f"{manifest_path}.summary.{field}"
            )
            if csv_value != embedded_value:
                raise RuntimeError(
                    f"{method_name} seed {seed} CSV/manifest summary mismatch for {field}: "
                    f"{csv_value} != {embedded_value}"
                )

        for field in SUMMARY_INTEGER_FIELDS:
            csv_value = required_int(raw.get(field), f"{summary_path}.{field}")
            embedded_value = required_int(
                embedded.get(field), f"{manifest_path}.summary.{field}"
            )
            if csv_value != embedded_value:
                raise RuntimeError(
                    f"{method_name} seed {seed} CSV/manifest summary mismatch for "
                    f"{field}: {csv_value} != {embedded_value}"
                )
        if required_int(raw.get("hidden_optimizer_state_bytes"), "hidden state bytes") != EXPECTED_HIDDEN_STATE_BYTES[method]:
            raise RuntimeError(
                f"{method_name} seed {seed} hidden optimizer-state byte count mismatch"
            )
        if required_int(raw.get("optimizer_state_bytes"), "optimizer state bytes") != required_int(
            raw.get("total_optimizer_state_bytes"), "total optimizer state bytes"
        ):
            raise RuntimeError(
                f"{method_name} seed {seed} total optimizer-state aliases disagree"
            )

        state_schema = embedded.get("state_schema")
        expected_roles = {
            "malt_momentum": 48,
            "malt_row_ema": 72,
            "malt_col_ema": 72,
            "malt_last_alpha_min": 48,
            "malt_last_alpha_max": 48,
        }
        if method == "malter_eq17":
            expected_roles["malt_nu"] = 72
        if (
            not isinstance(state_schema, dict)
            or state_schema.get("roles") != expected_roles
            or state_schema.get("contains_activation_k_state") is not False
            or state_schema.get("optimizer_group_steps") != [FORMAL_STEPS]
            or state_schema.get("numerical_checks_passed") is not True
        ):
            raise RuntimeError(
                f"{method_name} seed {seed} optimizer state-schema audit failed"
            )

        for field in ("run_name", "cell_id"):
            csv_value = str(raw.get(field, ""))
            embedded_value = str(embedded.get(field, ""))
            if not csv_value or csv_value != embedded_value:
                raise RuntimeError(
                    f"{method_name} seed {seed} CSV/manifest summary mismatch for {field}"
                )
        if str(cell.get("cell_id", "")) != str(raw["cell_id"]):
            raise RuntimeError(f"{method_name} seed {seed} formal cell-id mismatch")
        matrix_lrs = [
            required_float(source.get("matrix_lr"), f"{method_name}.{name}.matrix_lr")
            for name, source in (("CSV", raw), ("manifest summary", embedded), ("cell", cell))
        ]
        if len(set(matrix_lrs)) != 1:
            raise RuntimeError(f"{method_name} seed {seed} matrix-LR mismatch")

        init_hashes = {
            required_sha256(raw.get("init_sha256"), f"{summary_path}.init_sha256"),
            required_sha256(
                embedded.get("init_sha256"),
                f"{manifest_path}.summary.init_sha256",
            ),
        }
        if len(init_hashes) != 1:
            raise RuntimeError(f"{method_name} seed {seed} initialization hash mismatch")
        checkpoint_hashes = {
            required_sha256(
                raw.get("checkpoint_sha256"), f"{summary_path}.checkpoint_sha256"
            ),
            required_sha256(
                embedded.get("checkpoint_sha256"),
                f"{manifest_path}.summary.checkpoint_sha256",
            ),
        }
        checkpoint_sizes = {
            required_int(raw.get("checkpoint_bytes"), f"{summary_path}.checkpoint_bytes"),
            required_int(
                embedded.get("checkpoint_bytes"),
                f"{manifest_path}.summary.checkpoint_bytes",
            ),
        }
        if len(checkpoint_hashes) != 1 or len(checkpoint_sizes) != 1:
            raise RuntimeError(f"{method_name} seed {seed} checkpoint seal mismatch")
        checkpoint_size = next(iter(checkpoint_sizes))
        checkpoint_sha256 = next(iter(checkpoint_hashes))
        if checkpoint_size <= 0:
            raise RuntimeError(f"{method_name} seed {seed} formal checkpoint is empty")
        checkpoint_paths = {
            Path(str(raw.get("checkpoint_path", ""))).expanduser().resolve(),
            Path(str(embedded.get("checkpoint_path", ""))).expanduser().resolve(),
        }
        if len(checkpoint_paths) != 1:
            raise RuntimeError(f"{method_name} seed {seed} checkpoint path mismatch")
        checkpoint_path = next(iter(checkpoint_paths))
        if (
            not checkpoint_path.is_file()
            or checkpoint_path.stat().st_size != checkpoint_size
            or sha256_file(checkpoint_path) != checkpoint_sha256
        ):
            raise RuntimeError(
                f"{method_name} seed {seed} actual checkpoint certificate failed"
            )

        source_audit = manifest.get("source_audit")
        runtime = manifest.get("training_runtime_fingerprint")
        exact_runtime = manifest.get("exact_runtime_contract")
        data_inventory = manifest.get("data_inventory")
        if not isinstance(source_audit, dict) or not source_audit:
            raise RuntimeError(f"{method_name} seed {seed} source audit is missing")
        if not isinstance(runtime, dict) or not runtime:
            raise RuntimeError(f"{method_name} seed {seed} runtime fingerprint is missing")
        if (
            not isinstance(exact_runtime, dict)
            or exact_runtime.get("status") != "passed"
            or exact_runtime.get("expected") != EXPECTED_RUNTIME
            or exact_runtime.get("observed") != EXPECTED_RUNTIME
        ):
            raise RuntimeError(f"{method_name} seed {seed} exact runtime contract failed")
        if (
            not isinstance(data_inventory, dict)
            or data_inventory.get("status") != "passed"
            or required_int(
                data_inventory.get("train_shard_count"),
                f"{method_name} seed {seed} data train_shard_count",
            )
            != 50
            or required_int(
                data_inventory.get("validation_shard_count"),
                f"{method_name} seed {seed} data validation_shard_count",
            )
            != 1
        ):
            raise RuntimeError(f"{method_name} seed {seed} frozen data inventory failed")
        required_sha256(
            data_inventory.get("sha256"),
            f"{method_name} seed {seed} data_inventory.sha256",
        )
        source_fingerprints.add(sha256_json(source_audit))
        runtime_fingerprints.add(sha256_json(runtime))
        data_fingerprints.add(sha256_json(data_inventory))
        normalized.append(
            normalized_row(method, seed, raw, "experiment49", method_label)
        )
        evidence.append((seed, summary_path, manifest_path, manifest))

    if len(source_fingerprints) != 1:
        raise RuntimeError(
            f"{method_name} formal seeds do not share one source fingerprint"
        )
    if len(runtime_fingerprints) != 1:
        raise RuntimeError(
            f"{method_name} formal seeds do not share one runtime fingerprint"
        )
    if len(data_fingerprints) != 1:
        raise RuntimeError(
            f"{method_name} formal seeds do not share one frozen data inventory"
        )
    return normalized, evidence, wandb_caveats


def build_contrasts(
    by_pair: dict[tuple[str, int], dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    deltas: list[dict[str, object]] = []
    aggregate: list[dict[str, object]] = []
    for label, left, right, role in CONTRASTS:
        values: list[float] = []
        for seed in SEEDS:
            delta = float(by_pair[(left, seed)]["final_val_loss"]) - float(
                by_pair[(right, seed)]["final_val_loss"]
            )
            values.append(delta)
            deltas.append(
                {
                    "contrast": label,
                    "role": role,
                    "left": left,
                    "right": right,
                    "seed": seed,
                    "delta_final_val_loss": delta,
                }
            )
        mean = statistics.mean(values)
        sample_sd = statistics.stdev(values)
        half_width = T_CRIT_DF2 * sample_sd / math.sqrt(len(SEEDS))
        aggregate.append(
            {
                "contrast": label,
                "role": role,
                "left": left,
                "right": right,
                "n_seeds": len(SEEDS),
                "degrees_of_freedom": 2,
                "paired_mean": mean,
                "paired_sample_sd": sample_sd,
                "paired_t_ci95_low": mean - half_width,
                "paired_t_ci95_high": mean + half_width,
                "ci_interpretation": "descriptive_only_n3",
                "left_better_count": sum(value < 0 for value in values),
                "right_better_count": sum(value > 0 for value in values),
                "practical_margin": PRACTICAL_MARGIN,
                "mean_within_practical_margin": abs(mean) <= PRACTICAL_MARGIN,
                "practical_margin_interpretation": "descriptive_not_equivalence_test",
            }
        )
    return deltas, aggregate


def build_method_aggregate(
    by_pair: dict[tuple[str, int], dict[str, object]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        values = [
            float(by_pair[(method, seed)]["final_val_loss"]) for seed in SEEDS
        ]
        rows.append(
            {
                "method": method,
                "display_name": DISPLAY[method],
                "n_seeds": len(SEEDS),
                "final_val_mean": statistics.mean(values),
                "final_val_sample_sd": statistics.stdev(values),
            }
        )
    return sorted(rows, key=lambda row: float(row["final_val_mean"]))


def build_report(
    method_aggregate: list[dict[str, object]],
    contrasts: list[dict[str, object]],
    caveats: list[str],
) -> str:
    lines = [
        "# Experiment 49 controlled 124M R1 MALT-family analysis",
        "",
        "The accepted Experiment-45 eight-method panel is reused read-only and "
        "MALT-R1 adaptation and MALTER-Eq17-R1 adaptation are added as the ninth "
        "and tenth methods.",
        "",
        "All quality-run timing remains ineligible. Input and output files are "
        "sealed by SHA-256 in `analysis_manifest.json`.",
        "",
        "## Ten-method endpoint",
        "",
        "| rank | method | final val mean | seed SD |",
        "|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(method_aggregate, 1):
        lines.append(
            f"| {rank} | {row['display_name']} | "
            f"{float(row['final_val_mean']):.6f} | "
            f"{float(row['final_val_sample_sd']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen paired contrasts",
            "",
            "Negative means the named left method has lower final validation loss.",
            "",
            "| contrast | role | mean | descriptive 95% paired-t CI | direction | within 0.002 |",
            "|---|---|---:|---:|---:|:---:|",
        ]
    )
    for row in contrasts:
        lines.append(
            f"| {row['contrast']} | {row['role']} | "
            f"{float(row['paired_mean']):+.6f} | "
            f"[{float(row['paired_t_ci95_low']):+.6f}, "
            f"{float(row['paired_t_ci95_high']):+.6f}] | "
            f"{row['left_better_count']}/3 {row['left']}-better | "
            f"{'yes' if row['mean_within_practical_margin'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "With only three paired seeds, the t intervals are descriptive. "
            "A mean inside the 0.002 practical margin is not an equivalence test "
            "and must not be reported as established statistical equivalence.",
        ]
    )
    if caveats:
        lines.extend(["", "## Evidence-transfer caveats", ""])
        lines.extend(f"- {item}" for item in caveats)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    malt_summary_paths = [
        resolve_input(path, "MALT formal summary") for path in args.malt_summaries
    ]
    malt_manifest_paths = [
        resolve_input(path, "MALT formal manifest") for path in args.malt_manifests
    ]
    malter_summary_paths = [
        resolve_input(path, "MALTER-Eq17 formal summary")
        for path in args.malter_summaries
    ]
    malter_manifest_paths = [
        resolve_input(path, "MALTER-Eq17 formal manifest")
        for path in args.malter_manifests
    ]
    selection_certificate = resolve_input(
        args.selection_certificate, "dual-method selection certificate"
    )
    experiment45_summary = resolve_input(
        args.experiment45_summary, "Experiment-45 eight-method summary"
    )
    experiment45_manifest = resolve_input(
        args.experiment45_analysis_manifest, "Experiment-45 analysis manifest"
    )
    all_inputs = [
        *malt_summary_paths,
        *malt_manifest_paths,
        *malter_summary_paths,
        *malter_manifest_paths,
        selection_certificate,
        experiment45_summary,
        experiment45_manifest,
    ]
    if len(set(all_inputs)) != len(all_inputs):
        raise RuntimeError("all fifteen analysis inputs must be distinct files")

    selection_payload, selection_sha256 = validate_selection_certificate(
        selection_certificate
    )
    historical_rows, historical_manifest_payload = validate_experiment45(
        experiment45_summary, experiment45_manifest
    )
    malt_rows, malt_evidence, malt_wandb_caveats = validate_formal_inputs(
        "malt",
        malt_summary_paths,
        malt_manifest_paths,
        selection_payload,
        selection_sha256,
    )
    malter_rows, malter_evidence, malter_wandb_caveats = validate_formal_inputs(
        "malter_eq17",
        malter_summary_paths,
        malter_manifest_paths,
        selection_payload,
        selection_sha256,
    )
    formal_evidence = {
        "malt": malt_evidence,
        "malter_eq17": malter_evidence,
    }
    all_formal_manifests = [
        item[3] for evidence in formal_evidence.values() for item in evidence
    ]
    for field, description in (
        ("source_audit", "source fingerprint"),
        ("training_runtime_fingerprint", "runtime fingerprint"),
        ("data_inventory", "frozen data inventory"),
    ):
        fingerprints = {sha256_json(manifest[field]) for manifest in all_formal_manifests}
        if len(fingerprints) != 1:
            raise RuntimeError(
                f"MALT and MALTER-Eq17 formal evidence do not share one {description}"
            )

    unified = [*historical_rows, *malt_rows, *malter_rows]
    expected_pairs = {(method, seed) for method in METHOD_ORDER for seed in SEEDS}
    observed_pairs = {(str(row["method"]), int(row["seed"])) for row in unified}
    if observed_pairs != expected_pairs or len(unified) != len(expected_pairs):
        raise RuntimeError(
            f"ten-method coverage mismatch: missing={sorted(expected_pairs - observed_pairs)}"
        )
    by_pair = {
        (str(row["method"]), int(row["seed"])): row for row in unified
    }

    identity_failures: list[str] = []
    for seed in SEEDS:
        initials = {
            round(float(by_pair[(method, seed)]["initial_val_loss"]), 4)
            for method in METHOD_ORDER
        }
        if len(initials) != 1:
            identity_failures.append(
                f"seed {seed} initial validation losses differ: {sorted(initials)}"
            )
    if identity_failures:
        raise RuntimeError("paired identity audit failed: " + "; ".join(identity_failures))

    for seed in SEEDS:
        formal_init_hashes = {
            required_sha256(
                by_pair[(method, seed)]["init_sha256"],
                f"{method}/seed{seed}.init_sha256",
            )
            for method in FORMAL_METHODS
        }
        if len(formal_init_hashes) != 1:
            raise RuntimeError(
                f"MALT/MALTER-Eq17 seed {seed} initialization hashes differ"
            )

    input_records: list[dict[str, str]] = [
        {
            "role": "dual_method_selection_certificate",
            "path": str(selection_certificate),
            "sha256": selection_sha256,
        }
    ]
    for method in FORMAL_METHODS:
        for seed, summary_path, _, _ in formal_evidence[method]:
            input_records.append(
                {
                    "role": f"{method}_formal_summary_seed{seed}",
                    "path": str(summary_path),
                    "sha256": sha256_file(summary_path),
                }
            )
        for seed, _, manifest_path, _ in formal_evidence[method]:
            input_records.append(
                {
                    "role": f"{method}_formal_manifest_seed{seed}",
                    "path": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
                }
            )
    input_records.extend(
        [
            {
                "role": "experiment45_eight_method_summary",
                "path": str(experiment45_summary),
                "sha256": sha256_file(experiment45_summary),
            },
            {
                "role": "experiment45_analysis_manifest",
                "path": str(experiment45_manifest),
                "sha256": sha256_file(experiment45_manifest),
            },
        ]
    )
    input_bundle_sha256 = sha256_json(input_records)
    evidence_caveat = (
        "Experiment 45 is reused through its accepted eight-method analysis "
        "summary and manifest; Experiment 49 does not rewrite its historical rows."
    )
    malter_caveat = (
        "MALTER-Eq17-R1 adaptation denotes the frozen Equation-17 single-eta "
        "paper-derived interpretation and is not an official MALTER reproduction."
    )
    wandb_caveats = [*malt_wandb_caveats, *malter_wandb_caveats]
    identity = {
        "schema_version": 2,
        "status": "passed_with_caveats" if wandb_caveats else "passed",
        "protocol": "malt_r1_experiment45_identity_reuse_v4",
        "paired_quality_eligible": True,
        "method_labels": dict(FORMAL_METHOD_LABELS),
        "seed_coverage": list(SEEDS),
        "checks": {
            "experiment45_status": historical_manifest_payload["status"],
            "experiment45_protocol": historical_manifest_payload["protocol"],
            "experiment45_exact_24_rows": True,
            "dual_method_selection_certificate_sha256_verified": True,
            "formal_cells_and_lrs_match_selection": True,
            "malt_three_locally_accepted_formal_manifests": True,
            "malter_eq17_three_locally_accepted_formal_manifests": True,
            "formal_protocol": MALT_FORMAL_PROTOCOL,
            "formal_steps": FORMAL_STEPS,
            "formal_tokens": FORMAL_TOKENS,
            "malt_method_label_exact": True,
            "malter_eq17_method_label_exact": True,
            "formal_checkpoint_files_hash_and_size_verified": True,
            "formal_memory_and_state_schema_crosschecked": True,
            "formal_source_runtime_data_consistent_across_methods_and_seeds": True,
            "malt_malter_initialization_hash_equal_by_seed": True,
            "initial_validation_equal_within_four_decimals": True,
            "timing_eligible": False,
        },
        "input_files": input_records,
        "input_bundle_sha256": input_bundle_sha256,
        "caveats": [evidence_caveat, malter_caveat, *wandb_caveats],
    }

    unified.sort(key=lambda row: (int(row["seed"]), METHOD_ORDER.index(str(row["method"]))))
    deltas, contrasts = build_contrasts(by_pair)
    method_aggregate = build_method_aggregate(by_pair)
    report = build_report(method_aggregate, contrasts, identity["caveats"])

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "identity_reuse_certificate.json": lambda path: write_json(path, identity),
        "r1_unified_ten_method_run_summary.csv": lambda path: write_csv(path, unified),
        "r1_unified_ten_method_aggregate.csv": lambda path: write_csv(path, method_aggregate),
        "r1_malt_family_paired_seed_deltas.csv": lambda path: write_csv(path, deltas),
        "r1_malt_family_paired_aggregate.csv": lambda path: write_csv(path, contrasts),
        "EXPERIMENT_49_ANALYSIS.md": lambda path: path.write_text(
            report, encoding="utf-8", newline="\n"
        ),
    }
    for name, writer in artifacts.items():
        writer(output / name)
    output_sha256 = {name: sha256_file(output / name) for name in artifacts}
    output_bundle_sha256 = sha256_json(output_sha256)
    analysis_manifest = {
        "schema_version": 2,
        "status": "completed_valid",
        "protocol": "malt_r1_ten_method_analysis_v4",
        "experiment": 49,
        "formal_protocol": MALT_FORMAL_PROTOCOL,
        "selection_protocol": SELECTION_PROTOCOL,
        "selection_certificate": {
            "path": str(selection_certificate),
            "sha256": selection_sha256,
        },
        "selection_certificate_sha256": selection_sha256,
        "method_labels": dict(FORMAL_METHOD_LABELS),
        "seed_coverage": list(SEEDS),
        "method_order": list(METHOD_ORDER),
        "n_methods": len(METHOD_ORDER),
        "n_run_rows": len(unified),
        "n_formal_methods": len(FORMAL_METHODS),
        "n_formal_runs": len(FORMAL_METHODS) * len(SEEDS),
        "n_paired_contrasts": len(CONTRASTS),
        "n_paired_seed_deltas": len(deltas),
        "paired_ci_policy": "descriptive_paired_t_df2_n3",
        "practical_margin": PRACTICAL_MARGIN,
        "practical_margin_policy": "descriptive_not_equivalence_test",
        "input_files": input_records,
        "input_bundle_sha256": input_bundle_sha256,
        "outputs": list(artifacts),
        "output_sha256": output_sha256,
        "output_bundle_sha256": output_bundle_sha256,
        "identity_certificate": "identity_reuse_certificate.json",
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "formal_manifest_statuses": {
            method: {
                str(seed): manifest["status"]
                for seed, _, _, manifest in formal_evidence[method]
            }
            for method in FORMAL_METHODS
        },
    }
    manifest_path = output / "analysis_manifest.json"
    write_json(manifest_path, analysis_manifest)
    manifest_sha256 = sha256_file(manifest_path)
    (output / "analysis_manifest.sha256").write_text(
        f"{manifest_sha256}  analysis_manifest.json\n",
        encoding="ascii",
        newline="\n",
    )
    print(output)
    print(f"ANALYSIS_MANIFEST={manifest_path}")
    print(f"ANALYSIS_MANIFEST_SHA256={manifest_sha256}")


if __name__ == "__main__":
    main()
