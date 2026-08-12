"""Audit the R1 extended-baseline formal exports and merge them with core R1.

The script intentionally treats W&B timing as descriptive-only.  It audits the
scalar exports, preserves raw copies, reconstructs one row per formal run, and
computes seed-paired comparisons against the already frozen core R1 summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


RUN_RE = re.compile(
    r"^mainconf_r1_extended_formal_"
    r"(?P<cell>moonlight_r1scale|normuon_r1scale|adamw_low)_"
    r"seed(?P<seed>2024|2025|2026)_"
    r"(?P<batch>\d{8}T\d{6}[+-]\d{4})$"
)

EXPECTED_CELLS = ("adamw_low", "normuon_r1scale", "moonlight_r1scale")
EXPECTED_SEEDS = (2024, 2025, 2026)
EXPECTED_STEPS = {
    "val/loss": tuple(range(0, 6201, 100)),
    "train/loss_step": tuple(range(20, 6201, 20)),
    "time/train_s": tuple(range(0, 6201, 20)),
    "performance/step_avg_ms": tuple(range(40, 6201, 20)),
    "memory/optimizer_state_mib": (6200,),
    "memory/peak_allocated_mib": (6200,),
    "lr/auxiliary": tuple(range(0, 6201, 20)),
    "lr/matrix": tuple(range(0, 6201, 20)),
}
EXPECTED_LR = {
    "adamw_low": (0.0027, 0.000432),
    "normuon_r1scale": (0.0003, 0.0100),
    "moonlight_r1scale": (0.0018, 0.0018),
}
DISPLAY_NAME = {
    "diag": "Newton–Muon diag",
    "block4": "Newton–Muon block4",
    "none": "Newton–Muon none",
    "muon": "Muon",
    "moonlight_r1scale": "Moonlight Muon",
    "normuon_r1scale": "NorMuon",
    "adamw_low": "AdamW",
}
CORE_METHODS = ("diag", "block4", "none", "muon")
EXTENDED_METHODS = ("moonlight_r1scale", "normuon_r1scale", "adamw_low")
METRIC_COLUMNS = {
    "final_val_loss": "final_val_loss",
    "tail3_val_loss_mean": "tail3_val_loss_mean",
    "tail5_val_loss_mean": "tail5_val_loss_mean",
    "normalized_val_auc": "normalized_val_auc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exports", type=Path, nargs=8)
    parser.add_argument("--core-summary", type=Path, required=True)
    parser.add_argument("--core-history-2024-2025", type=Path, required=True)
    parser.add_argument("--core-history-2026", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column != "Step"
        and not column.endswith("__MIN")
        and not column.endswith("__MAX")
    ]


def split_column(column: str) -> tuple[str, str, str, int, str]:
    run_name, metric = column.rsplit(" - ", 1)
    match = RUN_RE.fullmatch(run_name)
    if match is None:
        raise ValueError(f"unrecognized formal run name: {run_name}")
    return (
        run_name,
        metric,
        match.group("cell"),
        int(match.group("seed")),
        match.group("batch"),
    )


def add_check(
    checks: list[dict[str, object]],
    check: str,
    passed: bool,
    evidence: str,
    severity: str = "critical",
) -> None:
    checks.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "severity_if_failed": severity,
            "evidence": evidence,
        }
    )


def sample_sd(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else math.nan


def summarize_run(group: pd.DataFrame) -> dict[str, object]:
    cell = str(group.method.iloc[0])
    seed = int(group.seed.iloc[0])
    val = group[group.metric == "val/loss"].sort_values("step")
    train = group[group.metric == "train/loss_step"].sort_values("step")
    train_time = group[group.metric == "time/train_s"].sort_values("step")
    step_time = group[group.metric == "performance/step_avg_ms"].sort_values("step")
    aux_lr = group[group.metric == "lr/auxiliary"].value
    matrix_lr = group[group.metric == "lr/matrix"].value
    optimizer_state = group[group.metric == "memory/optimizer_state_mib"].value
    peak = group[group.metric == "memory/peak_allocated_mib"].value
    final = float(val.value.iloc[-1])
    return {
        "method": cell,
        "display_name": DISPLAY_NAME[cell],
        "family": "extended_pilot_tuned",
        "run_name": str(group.run_name.iloc[0]),
        "seed": seed,
        "batch": str(group.batch.iloc[0]),
        "seed_role": "tuning_seed" if seed == 2026 else "confirmatory_seed",
        "initial_val_loss": float(val.value.iloc[0]),
        "final_val_loss": final,
        "best_val_loss": float(val.value.min()),
        "tail3_val_loss_mean": float(val.tail(3).value.mean()),
        "tail5_val_loss_mean": float(val.tail(5).value.mean()),
        "tail5_val_loss_sd": sample_sd(val.tail(5).value),
        "normalized_val_auc": float(
            np.trapezoid(val.value.to_numpy(), val.step.to_numpy())
            / float(val.step.iloc[-1] - val.step.iloc[0])
        ),
        "final_train_loss_step": float(train.value.iloc[-1]),
        "train_time_s_descriptive_only": float(train_time.value.iloc[-1]),
        "final_step_avg_ms_descriptive_only": float(step_time.value.iloc[-1]),
        "max_auxiliary_lr": float(aux_lr.max()),
        "max_matrix_lr": float(matrix_lr.max()),
        "peak_memory_mib": float(peak.iloc[0]),
        "optimizer_state_mib": float(optimizer_state.iloc[0]),
        "final_perplexity": math.exp(final),
        "timing_eligible": False,
    }


def aggregate_methods(unified: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in unified.groupby("method", sort=False):
        seeds = sorted(int(seed) for seed in group.seed)
        rows.append(
            {
                "method": method,
                "display_name": DISPLAY_NAME[method],
                "family": str(group.family.iloc[0]),
                "seeds": ",".join(str(seed) for seed in seeds),
                "n_seeds": len(group),
                "final_val_mean": float(group.final_val_loss.mean()),
                "final_val_sd": sample_sd(group.final_val_loss),
                "best_val_mean": float(group.best_val_loss.mean()),
                "tail3_mean": float(group.tail3_val_loss_mean.mean()),
                "tail3_between_seed_sd": sample_sd(group.tail3_val_loss_mean),
                "tail5_mean": float(group.tail5_val_loss_mean.mean()),
                "tail5_between_seed_sd": sample_sd(group.tail5_val_loss_mean),
                "normalized_auc_mean": float(group.normalized_val_auc.mean()),
                "normalized_auc_sd": sample_sd(group.normalized_val_auc),
                "peak_memory_mib_mean": float(group.peak_memory_mib.mean()),
                "optimizer_state_mib_mean": float(group.optimizer_state_mib.mean()),
            }
        )
    aggregate = pd.DataFrame(rows)
    aggregate["final_rank"] = aggregate.final_val_mean.rank(method="min").astype(int)
    aggregate["tail3_rank"] = aggregate.tail3_mean.rank(method="min").astype(int)
    aggregate["tail5_rank"] = aggregate.tail5_mean.rank(method="min").astype(int)
    aggregate["auc_rank"] = aggregate.normalized_auc_mean.rank(method="min").astype(int)
    return aggregate.sort_values(["final_rank", "method"]).reset_index(drop=True)


def paired_tables(unified: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    index = unified.set_index(["method", "seed"])
    for metric_name, column in METRIC_COLUMNS.items():
        for candidate in EXTENDED_METHODS:
            for reference in CORE_METHODS:
                deltas: list[float] = []
                for seed in EXPECTED_SEEDS:
                    candidate_value = float(index.loc[(candidate, seed), column])
                    reference_value = float(index.loc[(reference, seed), column])
                    delta = candidate_value - reference_value
                    deltas.append(delta)
                    seed_rows.append(
                        {
                            "metric": metric_name,
                            "candidate": candidate,
                            "candidate_display": DISPLAY_NAME[candidate],
                            "reference": reference,
                            "reference_display": DISPLAY_NAME[reference],
                            "seed": seed,
                            "candidate_value": candidate_value,
                            "reference_value": reference_value,
                            "delta_candidate_minus_reference": delta,
                            "candidate_better": delta < 0,
                        }
                    )
                series = pd.Series(deltas, dtype=float)
                confirmatory = pd.Series(deltas[:2], dtype=float)
                aggregate_rows.append(
                    {
                        "metric": metric_name,
                        "candidate": candidate,
                        "candidate_display": DISPLAY_NAME[candidate],
                        "reference": reference,
                        "reference_display": DISPLAY_NAME[reference],
                        "paired_mean_delta_all3": float(series.mean()),
                        "paired_sd_delta_all3": sample_sd(series),
                        "wins_all3": int((series < 0).sum()),
                        "losses_all3": int((series > 0).sum()),
                        "paired_mean_delta_confirmatory_2024_2025": float(
                            confirmatory.mean()
                        ),
                        "paired_sd_delta_confirmatory_2024_2025": sample_sd(
                            confirmatory
                        ),
                        "wins_confirmatory": int((confirmatory < 0).sum()),
                        "losses_confirmatory": int((confirmatory > 0).sum()),
                    }
                )
    return pd.DataFrame(seed_rows), pd.DataFrame(aggregate_rows)


def curve_comparisons(history: pd.DataFrame) -> pd.DataFrame:
    val = history[(history.metric == "val/loss") & (history.step >= 100)].copy()
    index = val.set_index(["method", "seed", "step"]).value
    rows: list[dict[str, object]] = []
    steps = tuple(range(100, 6201, 100))
    for candidate in EXTENDED_METHODS:
        for reference in CORE_METHODS:
            for seed in EXPECTED_SEEDS:
                deltas = np.array(
                    [
                        float(index.loc[(candidate, seed, step)])
                        - float(index.loc[(reference, seed, step)])
                        for step in steps
                    ],
                    dtype=float,
                )
                rows.append(
                    {
                        "candidate": candidate,
                        "candidate_display": DISPLAY_NAME[candidate],
                        "reference": reference,
                        "reference_display": DISPLAY_NAME[reference],
                        "seed": seed,
                        "candidate_better_checkpoints": int((deltas < 0).sum()),
                        "ties": int((deltas == 0).sum()),
                        "candidate_worse_checkpoints": int((deltas > 0).sum()),
                        "total_post_initial_checkpoints": len(deltas),
                        "mean_curve_delta_candidate_minus_reference": float(
                            deltas.mean()
                        ),
                        "final_delta_candidate_minus_reference": float(deltas[-1]),
                        "max_advantage_candidate": float(deltas.min()),
                        "max_disadvantage_candidate": float(deltas.max()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    raw_dir = output / "raw_wandb_exports"
    reference_dir = output / "reference"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    metric_files: dict[str, str] = {}

    for source_arg in args.exports:
        source = source_arg.resolve()
        copied = raw_dir / source.name
        shutil.copy2(source, copied)
        frame = pd.read_csv(source)
        bases = base_columns(frame)
        metrics = {split_column(column)[1] for column in bases}
        add_check(
            checks,
            f"{source.name}: exactly one metric",
            len(metrics) == 1,
            repr(sorted(metrics)),
        )
        if len(metrics) != 1:
            continue
        metric = next(iter(metrics))
        add_check(
            checks,
            f"{metric}: unique export file",
            metric not in metric_files,
            source.name,
        )
        metric_files[metric] = source.name
        add_check(
            checks,
            f"{metric}: exact nine-run coverage",
            len(bases) == 9,
            f"observed={len(bases)}",
        )
        observed_steps = tuple(int(step) for step in frame.Step.tolist())
        expected_steps = EXPECTED_STEPS.get(metric)
        add_check(
            checks,
            f"{metric}: exact step coverage",
            observed_steps == expected_steps,
            f"observed_first_last={observed_steps[:2]}...{observed_steps[-2:]}; "
            f"expected_points={len(expected_steps) if expected_steps else 0}",
        )

        for column in bases:
            run_name, parsed_metric, cell, seed, batch = split_column(column)
            values = frame[column]
            min_values = frame[f"{column}__MIN"]
            max_values = frame[f"{column}__MAX"]
            add_check(
                checks,
                f"{cell} seed{seed} {metric}: MIN/MAX mirrors",
                values.equals(min_values) and values.equals(max_values),
                f"points={len(values)}",
                "high",
            )
            finite = bool(np.isfinite(values.to_numpy(dtype=float)).all())
            add_check(
                checks,
                f"{cell} seed{seed} {metric}: finite",
                finite,
                f"finite={int(np.isfinite(values.to_numpy(dtype=float)).sum())}/{len(values)}",
            )
            for step, value in zip(frame.Step, values, strict=True):
                rows.append(
                    {
                        "method": cell,
                        "run_name": run_name,
                        "seed": seed,
                        "batch": batch,
                        "metric": parsed_metric,
                        "step": int(step),
                        "value": float(value),
                        "source_file": source.name,
                    }
                )
        sources.append(
            {
                "source_type": "wandb_export",
                "source_file": source.name,
                "sha256": sha256(source),
                "bytes": source.stat().st_size,
                "metric": metric,
                "rows": len(frame),
                "run_columns": len(bases),
                "preserved_copy": str(copied),
            }
        )

    history = pd.DataFrame(rows).sort_values(["method", "seed", "metric", "step"])
    add_check(
        checks,
        "exact expected metric set",
        set(metric_files) == set(EXPECTED_STEPS),
        repr(sorted(metric_files)),
    )
    observed_cells = tuple(sorted(history.method.unique()))
    observed_seeds = tuple(sorted(int(seed) for seed in history.seed.unique()))
    add_check(
        checks,
        "exact expected three methods",
        observed_cells == tuple(sorted(EXPECTED_CELLS)),
        repr(observed_cells),
    )
    add_check(
        checks,
        "exact expected three seeds",
        observed_seeds == EXPECTED_SEEDS,
        repr(observed_seeds),
    )
    duplicate_count = int(
        history.duplicated(["method", "seed", "metric", "step"]).sum()
    )
    add_check(
        checks,
        "no duplicate method/seed/metric/step grain",
        duplicate_count == 0,
        f"duplicates={duplicate_count}",
    )

    for seed in EXPECTED_SEEDS:
        initial = history[
            (history.seed == seed)
            & (history.metric == "val/loss")
            & (history.step == 0)
        ].set_index("method").value
        add_check(
            checks,
            f"seed{seed}: identical extended initialization loss",
            initial.nunique() == 1,
            repr(initial.to_dict()),
        )
    for cell, (expected_aux, expected_matrix) in EXPECTED_LR.items():
        for seed in EXPECTED_SEEDS:
            aux = history[
                (history.method == cell)
                & (history.seed == seed)
                & (history.metric == "lr/auxiliary")
            ].value.max()
            matrix = history[
                (history.method == cell)
                & (history.seed == seed)
                & (history.metric == "lr/matrix")
            ].value.max()
            add_check(
                checks,
                f"{cell} seed{seed}: expected frozen LR",
                math.isclose(float(aux), expected_aux)
                and math.isclose(float(matrix), expected_matrix),
                f"aux={aux}; matrix={matrix}",
            )

    extended_summary = pd.DataFrame(
        [
            summarize_run(group)
            for _, group in history.groupby(["method", "seed"], sort=False)
        ]
    ).sort_values(["seed", "final_val_loss", "method"])

    core_source = args.core_summary.resolve()
    core_copy = reference_dir / core_source.name
    shutil.copy2(core_source, core_copy)
    core = pd.read_csv(core_source)
    core["display_name"] = core.method.map(DISPLAY_NAME)
    core["family"] = "core_frozen"
    core["batch"] = ""
    core["seed_role"] = "core_prespecified"
    core["max_auxiliary_lr"] = core.max_adamw_lr
    core["timing_eligible"] = False
    required_unified_columns = list(extended_summary.columns)
    for column in required_unified_columns:
        if column not in core.columns:
            core[column] = np.nan
    core_for_merge = core[required_unified_columns].copy()
    unified = pd.concat([core_for_merge, extended_summary], ignore_index=True)
    unified["final_rank_within_seed"] = (
        unified.groupby("seed").final_val_loss.rank(method="min").astype(int)
    )
    unified = unified.sort_values(["seed", "final_rank_within_seed", "method"])

    for seed in EXPECTED_SEEDS:
        initial = unified[unified.seed == seed].initial_val_loss
        add_check(
            checks,
            f"seed{seed}: identical initialization across unified seven methods",
            initial.nunique() == 1,
            repr(sorted(float(value) for value in initial.unique())),
        )

    method_aggregate = aggregate_methods(unified)
    paired_seed, paired_aggregate = paired_tables(unified)

    core_histories: list[pd.DataFrame] = []
    for source_arg, copied_name in (
        (args.core_history_2024_2025, "core_history_2024_2025.csv"),
        (args.core_history_2026, "core_history_2026.csv"),
    ):
        source = source_arg.resolve()
        shutil.copy2(source, reference_dir / copied_name)
        core_history = pd.read_csv(source)
        core_histories.append(core_history)
        sources.append(
            {
                "source_type": "frozen_core_reference",
                "source_file": source.name,
                "sha256": sha256(source),
                "bytes": source.stat().st_size,
                "metric": "multiple",
                "rows": len(core_history),
                "run_columns": 4,
                "preserved_copy": str(reference_dir / copied_name),
            }
        )
    unified_history = pd.concat(core_histories + [history], ignore_index=True)
    curve_table = curve_comparisons(unified_history)

    confirmatory = unified[unified.seed.isin([2024, 2025])].copy()
    confirmatory_aggregate = aggregate_methods(confirmatory)

    checks_frame = pd.DataFrame(checks)
    sources.append(
        {
            "source_type": "frozen_core_reference",
            "source_file": core_source.name,
            "sha256": sha256(core_source),
            "bytes": core_source.stat().st_size,
            "metric": "run_summary",
            "rows": len(core),
            "run_columns": 4,
            "preserved_copy": str(core_copy),
        }
    )
    source_frame = pd.DataFrame(sources)

    history.to_csv(output / "extended_formal_history_long.csv", index=False)
    extended_summary.to_csv(output / "extended_formal_run_summary.csv", index=False)
    unified.to_csv(output / "r1_unified_seven_method_run_summary.csv", index=False)
    method_aggregate.to_csv(output / "r1_unified_seven_method_aggregate.csv", index=False)
    confirmatory_aggregate.to_csv(
        output / "r1_confirmatory_2024_2025_method_aggregate.csv", index=False
    )
    paired_seed.to_csv(output / "r1_extended_vs_core_paired_seed_deltas.csv", index=False)
    paired_aggregate.to_csv(
        output / "r1_extended_vs_core_paired_aggregate.csv", index=False
    )
    curve_table.to_csv(output / "r1_extended_vs_core_curve_comparison.csv", index=False)
    checks_frame.to_csv(output / "data_quality_checks.csv", index=False)
    source_frame.to_csv(output / "source_manifest.csv", index=False)

    counts = checks_frame.status.value_counts().to_dict()
    status = "PASS_WITH_CAVEATS" if counts.get("FAIL", 0) == 0 else "FAIL"
    manifest = {
        "created_at": "2026-07-23",
        "status": status,
        "extended_formal_run_count": int(len(extended_summary)),
        "unified_run_count": int(len(unified)),
        "methods": list(method_aggregate.method),
        "seeds": list(EXPECTED_SEEDS),
        "quality_checks": {key: int(value) for key, value in counts.items()},
        "timing_eligible": False,
        "caveats": [
            "W&B CSV cannot prove local source/runtime/init/resume/checkpoint manifests.",
            "Seed 2026 selected the extended-baseline LR; seeds 2024/2025 are confirmatory.",
            "Timing is descriptive-only because this was not the isolated R1-PERF protocol.",
        ],
        "outputs": sorted(
            [
                path.name
                for path in output.glob("*")
                if path.is_file() and path.name != "analysis_manifest.json"
            ]
        ),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
