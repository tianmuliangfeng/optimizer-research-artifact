"""Audit mixed W&B history exports against accepted experiment-43/44 metrics."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from pathlib import Path
from typing import Any

from common import (
    ContractError,
    atomic_write_text,
    commit_manifest,
    ensure_new_output,
    read_csv,
    read_json,
    sha256_file,
    write_csv,
)


SCHEMA = "mdp_wandb_history_confirmation_v1"
RUN_PATTERN = re.compile(
    r"^(?P<run>record(?P<record>17|28)_"
    r"(?P<method>muon|original_newton_muon|selective_diag|selective_none)_"
    r"seed(?P<seed>\d+)) - (?P<metric>.+)$"
)
METRICS = {
    "validation/loss": "validation_loss",
    "train/tokens": "train_tokens",
    "diagnostic/train_time_ms": "diagnostic_train_time_ms",
}
LONG_FIELDS = [
    "experiment",
    "record",
    "run_name",
    "seed",
    "method",
    "step",
    "validation_loss",
    "train_tokens",
    "diagnostic_train_time_ms",
]
AUDIT_FIELDS = [
    "experiment",
    "run_name",
    "seed",
    "method",
    "history_points",
    "expected_history_points",
    "final_step",
    "wandb_final_validation_loss",
    "local_final_validation_loss",
    "max_abs_validation_loss_difference",
    "max_abs_token_difference",
    "max_abs_train_time_ms_difference",
    "exact_match",
]


def _number(raw: str, context: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ContractError(f"{context}: non-numeric value {raw!r}") from exc
    if not math.isfinite(value):
        raise ContractError(f"{context}: non-finite value {raw!r}")
    return value


def _load_wide_export(
    path: Path, expected_metric: str
) -> tuple[dict[tuple[str, int], float], set[str], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Step" not in reader.fieldnames:
            raise ContractError(f"{path}: missing Step column")
        columns: dict[str, str] = {}
        for field in reader.fieldnames:
            match = RUN_PATTERN.match(field)
            if match and match.group("metric") == expected_metric:
                columns[match.group("run")] = field
        if not columns:
            raise ContractError(f"{path}: no columns for metric {expected_metric!r}")
        values: dict[tuple[str, int], float] = {}
        redundant_checks = 0
        for row_number, row in enumerate(reader, start=2):
            step_value = _number(row["Step"], f"{path}:{row_number}:Step")
            if not step_value.is_integer():
                raise ContractError(f"{path}:{row_number}: non-integral Step {step_value}")
            step = int(step_value)
            for run_name, field in columns.items():
                raw = row.get(field, "").strip()
                if not raw:
                    continue
                value = _number(raw, f"{path}:{row_number}:{field}")
                key = (run_name, step)
                if key in values:
                    raise ContractError(f"{path}: duplicate run/step value {key}")
                values[key] = value
                for suffix in ("__MIN", "__MAX"):
                    companion = field + suffix
                    if companion not in row or not row[companion].strip():
                        raise ContractError(f"{path}:{row_number}: missing {companion}")
                    companion_value = _number(
                        row[companion], f"{path}:{row_number}:{companion}"
                    )
                    if companion_value != value:
                        raise ContractError(
                            f"{path}:{row_number}: {companion} differs from primary value"
                        )
                    redundant_checks += 1
    return values, set(columns), redundant_checks


def _accepted_metrics(run_dir: Path, seed: str, method: str) -> list[dict[str, str]]:
    cell_dir = run_dir / "formal" / f"seed{seed}" / method
    accepted = read_json(cell_dir / "accepted.json")
    attempt = cell_dir / str(accepted["attempt_dir"])
    metrics_path = attempt / "metrics.csv"
    if not metrics_path.is_file():
        raise ContractError(f"accepted metrics are missing: {metrics_path}")
    return read_csv(metrics_path)


def _local_contract(run_dir: Path, record: str) -> dict[str, dict[str, Any]]:
    cells_path = run_dir / "analysis" / f"record{record}_cells.csv"
    manifest_path = run_dir / "analysis" / f"record{record}_analysis_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "passed" or manifest.get("passed") is not True:
        raise ContractError(f"record{record} analysis manifest is not accepted")
    expected: dict[str, dict[str, Any]] = {}
    for cell in read_csv(cells_path):
        seed = str(cell["seed"])
        method = str(cell["method"])
        run_name = f"record{record}_{method}_seed{seed}"
        local_rows = _accepted_metrics(run_dir, seed, method)
        expected[run_name] = {
            "experiment": "43" if record == "28" else "44",
            "record": record,
            "seed": seed,
            "method": method,
            "cell": cell,
            "metrics": {int(row["step"]): row for row in local_rows},
        }
    return expected


def audit(
    loss_csv: Path,
    tokens_csv: Path,
    time_csv: Path,
    record28_run_dir: Path,
    record17_run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest_name = "wandb_history_confirmation_manifest.json"
    ensure_new_output(output_dir, manifest_name)
    export_paths = {
        "validation/loss": loss_csv,
        "train/tokens": tokens_csv,
        "diagnostic/train_time_ms": time_csv,
    }
    export_values: dict[str, dict[tuple[str, int], float]] = {}
    run_sets: list[set[str]] = []
    redundant_check_count = 0
    for metric, path in export_paths.items():
        values, runs, checks = _load_wide_export(path, metric)
        export_values[metric] = values
        run_sets.append(runs)
        redundant_check_count += checks
    if not all(runs == run_sets[0] for runs in run_sets[1:]):
        raise ContractError("the three W&B exports contain different run sets")

    expected = {
        **_local_contract(record28_run_dir, "28"),
        **_local_contract(record17_run_dir, "17"),
    }
    observed_runs = run_sets[0]
    if observed_runs != set(expected):
        raise ContractError(
            "W&B/local run-set mismatch; "
            f"missing={sorted(set(expected) - observed_runs)}, "
            f"unexpected={sorted(observed_runs - set(expected))}"
        )

    long_rows_by_experiment: dict[str, list[dict[str, Any]]] = {"43": [], "44": []}
    audit_rows: list[dict[str, Any]] = []
    global_maxima = {metric: 0.0 for metric in METRICS}
    for run_name in sorted(expected):
        contract = expected[run_name]
        local = contract["metrics"]
        observed_steps = {
            step for name, step in export_values["validation/loss"] if name == run_name
        }
        if observed_steps != set(local):
            raise ContractError(
                f"{run_name}: W&B/local step mismatch; "
                f"missing={sorted(set(local) - observed_steps)}, "
                f"unexpected={sorted(observed_steps - set(local))}"
            )
        maxima: dict[str, float] = {}
        for metric, local_field in (
            ("validation/loss", "val_loss"),
            ("train/tokens", "tokens"),
            ("diagnostic/train_time_ms", "train_time_ms"),
        ):
            metric_steps = {
                step for name, step in export_values[metric] if name == run_name
            }
            if metric_steps != observed_steps:
                raise ContractError(f"{run_name}: {metric} has incomplete steps")
            differences = [
                abs(export_values[metric][(run_name, step)] - float(local[step][local_field]))
                for step in observed_steps
            ]
            maxima[metric] = max(differences, default=0.0)
            global_maxima[metric] = max(global_maxima[metric], maxima[metric])
        final_step = max(observed_steps)
        final_loss = export_values["validation/loss"][(run_name, final_step)]
        local_final = float(contract["cell"]["final_val_loss"])
        if final_loss != local_final:
            raise ContractError(f"{run_name}: W&B endpoint differs from record cells")
        for step in sorted(observed_steps):
            long_rows_by_experiment[contract["experiment"]].append(
                {
                    "experiment": contract["experiment"],
                    "record": contract["record"],
                    "run_name": run_name,
                    "seed": contract["seed"],
                    "method": contract["method"],
                    "step": step,
                    "validation_loss": export_values["validation/loss"][(run_name, step)],
                    "train_tokens": export_values["train/tokens"][(run_name, step)],
                    "diagnostic_train_time_ms": export_values["diagnostic/train_time_ms"][(run_name, step)],
                }
            )
        audit_rows.append(
            {
                "experiment": contract["experiment"],
                "run_name": run_name,
                "seed": contract["seed"],
                "method": contract["method"],
                "history_points": len(observed_steps),
                "expected_history_points": len(local),
                "final_step": final_step,
                "wandb_final_validation_loss": final_loss,
                "local_final_validation_loss": local_final,
                "max_abs_validation_loss_difference": maxima["validation/loss"],
                "max_abs_token_difference": maxima["train/tokens"],
                "max_abs_train_time_ms_difference": maxima["diagnostic/train_time_ms"],
                "exact_match": all(value == 0.0 for value in maxima.values()),
            }
        )

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_names = {
        "validation/loss": "raw/wandb_validation_loss.csv",
        "train/tokens": "raw/wandb_train_tokens.csv",
        "diagnostic/train_time_ms": "raw/wandb_diagnostic_train_time_ms.csv",
    }
    for metric, path in export_paths.items():
        shutil.copy2(path, output_dir / raw_names[metric])
    write_csv(
        output_dir / "experiment43_wandb_history.csv",
        long_rows_by_experiment["43"],
        LONG_FIELDS,
    )
    write_csv(
        output_dir / "experiment44_wandb_history.csv",
        long_rows_by_experiment["44"],
        LONG_FIELDS,
    )
    write_csv(output_dir / "wandb_run_audit.csv", audit_rows, AUDIT_FIELDS)

    all_exact = all(row["exact_match"] for row in audit_rows)
    report = [
        "# Experiments 43/44 W&B history confirmation",
        "",
        f"- Runs: {len(audit_rows)} (experiment 43: 16; experiment 44: 12)",
        f"- Experiment 43 history rows: {len(long_rows_by_experiment['43'])}",
        f"- Experiment 44 history rows: {len(long_rows_by_experiment['44'])}",
        f"- W&B MIN/MAX redundancy checks: {redundant_check_count}",
        f"- Exact checkpoint agreement with accepted local metrics: {all_exact}",
        "- Local sealed artifacts remain authoritative; W&B is an external mirror.",
        "- `diagnostic/train_time_ms` is retained for provenance but is timing-ineligible.",
        "",
        "The mixed W&B exports were separated by deterministic run name. No smoothing,",
        "aggregation across seeds, or interpolation was applied.",
    ]
    atomic_write_text(output_dir / "WANDB_CONFIRMATION.md", "\n".join(report) + "\n")
    output_names = [
        "experiment43_wandb_history.csv",
        "experiment44_wandb_history.csv",
        "wandb_run_audit.csv",
        "WANDB_CONFIRMATION.md",
        *raw_names.values(),
    ]
    result = {
        "schema_version": SCHEMA,
        "status": "passed" if all_exact else "failed",
        "synthetic": False,
        "claim_eligible": False,
        "scientific_role": "external_exact_mirror_confirmation",
        "local_artifacts_authoritative": True,
        "run_count": len(audit_rows),
        "experiment43_run_count": sum(row["experiment"] == "43" for row in audit_rows),
        "experiment44_run_count": sum(row["experiment"] == "44" for row in audit_rows),
        "experiment43_history_rows": len(long_rows_by_experiment["43"]),
        "experiment44_history_rows": len(long_rows_by_experiment["44"]),
        "redundant_min_max_checks": redundant_check_count,
        "all_histories_exact": all_exact,
        "global_max_abs_differences": global_maxima,
        "source_exports": {
            metric: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for metric, path in export_paths.items()
        },
        "record28_run_dir": str(record28_run_dir),
        "record17_run_dir": str(record17_run_dir),
    }
    commit_manifest(output_dir, manifest_name, result, output_names)
    if not all_exact:
        raise ContractError("one or more W&B histories differ from accepted local metrics")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-csv", type=Path, required=True)
    parser.add_argument("--tokens-csv", type=Path, required=True)
    parser.add_argument("--time-csv", type=Path, required=True)
    parser.add_argument("--record28-run-dir", type=Path, required=True)
    parser.add_argument("--record17-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(
        args.loss_csv.resolve(),
        args.tokens_csv.resolve(),
        args.time_csv.resolve(),
        args.record28_run_dir.resolve(),
        args.record17_run_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(
        f"W&B history confirmation passed: runs={result['run_count']} "
        f"exact={result['all_histories_exact']}"
    )


if __name__ == "__main__":
    main()
