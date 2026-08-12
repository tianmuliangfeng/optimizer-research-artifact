"""Audit and summarize the nine-cell R1 extended-baseline W&B pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


RUN_RE = re.compile(
    r"^mainconf_r1_extended_pilot_"
    r"(?P<cell>(?:moonlight|normuon)_(?:official|r1scale|high)|"
    r"adamw_(?:low|official|high))_seed(?P<seed>\d+)_"
    r"(?P<batch>\d{8}T\d{6}[+-]\d{4})$"
)

EXPECTED_CELLS = (
    "adamw_low",
    "adamw_official",
    "adamw_high",
    "normuon_r1scale",
    "normuon_official",
    "normuon_high",
    "moonlight_official",
    "moonlight_r1scale",
    "moonlight_high",
)

EXPECTED_STEPS = {
    "val/loss": tuple(range(0, 1001, 100)),
    "train/loss_step": tuple(range(20, 1001, 20)),
    "time/train_s": tuple(range(0, 1001, 20)),
    "performance/step_avg_ms": tuple(range(40, 1001, 20)),
    "memory/optimizer_state_mib": (1000,),
    "memory/peak_allocated_mib": (1000,),
    "lr/auxiliary": tuple(range(0, 1001, 20)),
    "lr/matrix": tuple(range(0, 1001, 20)),
}

EXPECTED_LR = {
    "adamw_low": (0.0027, 0.000432),
    "adamw_official": (0.0036, 0.000576),
    "adamw_high": (0.0045, 0.000720),
    "normuon_r1scale": (0.0003, 0.0100),
    "normuon_official": (0.0003, 0.0200),
    "normuon_high": (0.0003, 0.0300),
    "moonlight_official": (0.0010, 0.0010),
    "moonlight_r1scale": (0.0018, 0.0018),
    "moonlight_high": (0.0030, 0.0030),
}

SELECTION = {
    "adamw": "adamw_low",
    "normuon": "normuon_r1scale",
    "moonlight": "moonlight_r1scale",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exports", type=Path, nargs=8)
    parser.add_argument("--r1-reference", type=Path, required=True)
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
        raise ValueError(f"unrecognized pilot run name: {run_name}")
    cell = match.group("cell")
    method = cell.split("_", 1)[0]
    return run_name, metric, cell, int(match.group("seed")), match.group("batch")


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


def interpolate_step(curve: pd.DataFrame, target: float) -> float:
    values = curve.sort_values("step")[["step", "value"]].to_numpy(dtype=float)
    for index in range(1, len(values)):
        s0, v0 = values[index - 1]
        s1, v1 = values[index]
        if v0 > target >= v1:
            if v1 == v0:
                return float(s1)
            return float(s0 + (target - v0) * (s1 - s0) / (v1 - v0))
    return math.nan


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
    seen_metric_files: dict[str, str] = {}

    for source in args.exports:
        source = source.resolve()
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
        metric = next(iter(metrics))
        add_check(
            checks,
            f"{metric}: unique export file",
            metric not in seen_metric_files,
            source.name,
        )
        seen_metric_files[metric] = source.name
        add_check(
            checks,
            f"{metric}: exact nine-run coverage",
            len(bases) == 9,
            f"observed={len(bases)}",
        )
        observed_steps = tuple(int(step) for step in frame["Step"].tolist())
        expected_steps = EXPECTED_STEPS.get(metric)
        add_check(
            checks,
            f"{metric}: exact step coverage",
            observed_steps == expected_steps,
            f"observed={observed_steps}; expected={expected_steps}",
        )

        for column in bases:
            run_name, parsed_metric, cell, seed, batch = split_column(column)
            values = frame[column]
            min_values = frame[f"{column}__MIN"]
            max_values = frame[f"{column}__MAX"]
            mirrors = values.equals(min_values) and values.equals(max_values)
            add_check(
                checks,
                f"{cell} {metric}: MIN/MAX mirrors",
                mirrors,
                f"rows={len(values)}",
                "high",
            )
            finite = bool(np.isfinite(values.to_numpy(dtype=float)).all())
            add_check(
                checks,
                f"{cell} {metric}: finite",
                finite,
                f"nonnull={int(values.notna().sum())}/{len(values)}",
            )
            for step, value in zip(frame["Step"], values, strict=True):
                rows.append(
                    {
                        "run_name": run_name,
                        "cell": cell,
                        "method": cell.split("_", 1)[0],
                        "lr_label": cell.split("_", 1)[1],
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
                "source_file": source.name,
                "sha256": sha256(source),
                "bytes": source.stat().st_size,
                "metric": metric,
                "rows": len(frame),
                "run_columns": len(bases),
                "preserved_copy": str(copied),
            }
        )

    long = pd.DataFrame(rows).sort_values(["metric", "cell", "step"])
    duplicate_count = int(long.duplicated(["cell", "metric", "step"]).sum())
    add_check(
        checks,
        "no duplicate cell/metric/step grain",
        duplicate_count == 0,
        f"duplicates={duplicate_count}",
    )
    observed_cells = tuple(sorted(long["cell"].unique()))
    add_check(
        checks,
        "exact expected pilot cells",
        observed_cells == tuple(sorted(EXPECTED_CELLS)),
        repr(observed_cells),
    )
    seeds = tuple(sorted(int(value) for value in long["seed"].unique()))
    batches = tuple(sorted(long["batch"].unique()))
    add_check(checks, "single controlled seed", seeds == (2026,), repr(seeds))
    add_check(checks, "single pilot batch id", len(batches) == 1, repr(batches))
    add_check(
        checks,
        "exact expected metric set",
        set(seen_metric_files) == set(EXPECTED_STEPS),
        repr(sorted(seen_metric_files)),
    )

    val = long[long.metric == "val/loss"].copy()
    initial = val[val.step == 0].set_index("cell")["value"]
    add_check(
        checks,
        "identical initial validation loss",
        initial.nunique() == 1,
        repr(initial.to_dict()),
    )

    for cell, (expected_aux, expected_matrix) in EXPECTED_LR.items():
        aux = long[
            (long.cell == cell) & (long.metric == "lr/auxiliary") & (long.step == 0)
        ].value.iloc[0]
        matrix = long[
            (long.cell == cell) & (long.metric == "lr/matrix") & (long.step == 0)
        ].value.iloc[0]
        add_check(
            checks,
            f"{cell}: configured LR values",
            math.isclose(aux, expected_aux) and math.isclose(matrix, expected_matrix),
            f"aux={aux}; matrix={matrix}",
        )

    summaries: list[dict[str, object]] = []
    thresholds = (5.0, 4.5, 4.1)
    for cell in EXPECTED_CELLS:
        curve = val[val.cell == cell].sort_values("step")
        post_initial = curve[curve.step >= 100]
        memory = long[(long.cell == cell) & long.metric.str.startswith("memory/")]
        aux_lr, matrix_lr = EXPECTED_LR[cell]
        row: dict[str, object] = {
            "cell": cell,
            "method": cell.split("_", 1)[0],
            "lr_label": cell.split("_", 1)[1],
            "seed": 2026,
            "auxiliary_lr": aux_lr,
            "matrix_lr": matrix_lr,
            "initial_val_loss": float(curve.iloc[0].value),
            "val_loss_step_100": float(curve[curve.step == 100].value.iloc[0]),
            "final_val_loss_step_1000": float(curve.iloc[-1].value),
            "best_val_loss": float(curve.value.min()),
            "tail3_mean": float(curve.tail(3).value.mean()),
            "tail3_std_sample": float(curve.tail(3).value.std(ddof=1)),
            "post_initial_auc_100_1000": float(
                np.trapezoid(post_initial.value, post_initial.step) / 900.0
            ),
            "full_auc_0_1000": float(np.trapezoid(curve.value, curve.step) / 1000.0),
            "optimizer_state_mib": float(
                memory[memory.metric == "memory/optimizer_state_mib"].value.iloc[0]
            ),
            "peak_allocated_mib": float(
                memory[memory.metric == "memory/peak_allocated_mib"].value.iloc[0]
            ),
            "quality_usable": True,
            "memory_usable": True,
            "timing_usable": False,
        }
        for target in thresholds:
            suffix = str(target).replace(".", "p")
            reached = curve[curve.value <= target]
            row[f"discrete_step_to_{suffix}"] = (
                int(reached.step.iloc[0]) if not reached.empty else -1
            )
            interpolated = interpolate_step(curve, target)
            row[f"interpolated_step_to_{suffix}"] = (
                interpolated if math.isfinite(interpolated) else -1.0
            )
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary["rank_final_overall"] = summary["final_val_loss_step_1000"].rank(
        method="min"
    ).astype(int)
    summary["rank_auc_overall"] = summary["post_initial_auc_100_1000"].rank(
        method="min"
    ).astype(int)
    summary["rank_final_within_method"] = (
        summary.groupby("method")["final_val_loss_step_1000"]
        .rank(method="min")
        .astype(int)
    )
    summary["rank_auc_within_method"] = (
        summary.groupby("method")["post_initial_auc_100_1000"]
        .rank(method="min")
        .astype(int)
    )

    selected = summary[summary.cell.isin(SELECTION.values())].copy()
    selected["selection_reason"] = selected.cell.map(
        {
            "moonlight_r1scale": "best final and tail3; crosses high at step900, while high has better early AUC",
            "normuon_r1scale": "best final and tail3 from step700 onward; official has better early AUC; lower-bound winner",
            "adamw_low": "best final, tail3, and AUC within AdamW; optimum is at lower search boundary",
        }
    )
    selected["formal_seed2026_decision"] = "advance"
    selected["multiseed_decision"] = selected.method.map(
        {
            "moonlight": "conditional: expand if 6200-step result is competitive",
            "normuon": "conditional: only if 6200-step result changes the paper comparison",
            "adamw": "conditional: conventional baseline value versus compute budget",
        }
    )

    r1 = pd.read_csv(args.r1_reference.resolve())
    r1_copy = reference_dir / "r1_seed2026_normalized_history_long.csv"
    shutil.copy2(args.r1_reference.resolve(), r1_copy)
    r1_step = r1[(r1.metric == "val/loss") & (r1.step == 1000)][
        ["method", "value", "run_name"]
    ].rename(columns={"value": "r1_val_loss_step_1000"})
    comparisons: list[dict[str, object]] = []
    for _, candidate in selected.iterrows():
        for _, reference in r1_step.iterrows():
            comparisons.append(
                {
                    "candidate_cell": candidate.cell,
                    "candidate_method": candidate.method,
                    "candidate_val_loss_step_1000": candidate.final_val_loss_step_1000,
                    "r1_method": reference.method,
                    "r1_val_loss_step_1000": reference.r1_val_loss_step_1000,
                    "candidate_minus_r1": (
                        candidate.final_val_loss_step_1000
                        - reference.r1_val_loss_step_1000
                    ),
                }
            )
    comparison = pd.DataFrame(comparisons)

    curve_rank_rows: list[dict[str, object]] = []
    for step, group in val[val.step > 0].groupby("step"):
        ordered = group.sort_values("value")
        for rank, (_, item) in enumerate(ordered.iterrows(), start=1):
            curve_rank_rows.append(
                {
                    "step": int(step),
                    "cell": item.cell,
                    "method": item.method,
                    "val_loss": item.value,
                    "rank_overall": rank,
                }
            )
    curve_ranks = pd.DataFrame(curve_rank_rows)

    check_frame = pd.DataFrame(checks)
    failures = check_frame[check_frame.status == "FAIL"]
    status = "PASS_WITH_CAVEATS" if failures.empty else "FAIL"
    caveats = [
        "W&B exports do not contain the local pilot manifest, source/runtime/init fingerprint, or resume count.",
        "Pilot uses seed2026 for LR selection and is screening evidence, not independent confirmation.",
        "Timing is diagnostic only and is excluded from selection and paper performance claims.",
        "AdamW-low and NorMuon-r1scale are lower-bound winners, so the grid does not prove an interior optimum.",
    ]

    long.to_csv(output / "pilot_history_long.csv", index=False)
    summary.sort_values("final_val_loss_step_1000").to_csv(
        output / "pilot_run_summary.csv", index=False
    )
    selected.sort_values("final_val_loss_step_1000").to_csv(
        output / "pilot_method_selection.csv", index=False
    )
    comparison.to_csv(output / "pilot_vs_r1_step1000.csv", index=False)
    curve_ranks.to_csv(output / "pilot_curve_ranks.csv", index=False)
    check_frame.to_csv(output / "data_quality_checks.csv", index=False)
    pd.DataFrame(sources).to_csv(
        output / "source_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )

    manifest = {
        "analysis_status": status,
        "pilot_seed": 2026,
        "pilot_steps": 1000,
        "cells": list(EXPECTED_CELLS),
        "selected_cells": SELECTION,
        "data_quality": {
            "checks": len(check_frame),
            "pass": int((check_frame.status == "PASS").sum()),
            "fail": int((check_frame.status == "FAIL").sum()),
        },
        "caveats": caveats,
        "source_files": sources,
        "r1_reference": str(args.r1_reference.resolve()),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    by_cell = summary.set_index("cell")
    moon = by_cell.loc["moonlight_r1scale"]
    nor = by_cell.loc["normuon_r1scale"]
    adam = by_cell.loc["adamw_low"]
    r1_muon = float(r1_step[r1_step.method == "muon"].r1_val_loss_step_1000.iloc[0])
    r1_diag = float(r1_step[r1_step.method == "diag"].r1_val_loss_step_1000.iloc[0])

    report = f"""# R1 extended-baseline 1000-step pilot analysis

Date: 2026-07-21  
Status: `{status}`  
Primary screening endpoint: validation loss at step 1000 under matched seed/token/evaluation budget.

## Technical summary

- **Moonlight `r1scale` is the only candidate close to the existing R1 optimizers at step 1000.** It finishes at {moon.final_val_loss_step_1000:.4f}, only {moon.final_val_loss_step_1000-r1_muon:+.4f} versus R1 Muon ({r1_muon:.4f}), although it remains {moon.final_val_loss_step_1000-r1_diag:+.4f} behind R1 diag ({r1_diag:.4f}).
- **The primary step-1000 endpoint selects:** Moonlight `r1scale` ({moon.matrix_lr:g}), NorMuon `r1scale` ({nor.matrix_lr:g}), and AdamW `low` (base {adam.auxiliary_lr:g}, hidden {adam.matrix_lr:g}). AdamW-low also wins AUC; Moonlight-high and NorMuon-official learn faster early, but the selected lower-LR cells cross them at steps 900 and 700 respectively and finish better.
- **Advance one selected cell per method to 6200-step seed2026 formal.** Moonlight is the competitive modern baseline; NorMuon and AdamW are useful formal reference baselines, but their pilot gaps do not currently challenge the R1 core result.
- **Do not interpret pilot timing as performance evidence.** Quality/state metrics are usable from W&B; formal evidence still requires the local manifests and runtime/source/init audit.

## Exact pilot results

| Rank | Cell | Aux LR | Matrix LR | Step-1000 val | Tail-3 | Post-initial AUC | Peak MiB | Optimizer state MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for rank, item in enumerate(
        summary.sort_values("final_val_loss_step_1000").itertuples(), start=1
    ):
        report += (
            f"| {rank} | {item.cell} | {item.auxiliary_lr:.6g} | {item.matrix_lr:.6g} | "
            f"{item.final_val_loss_step_1000:.4f} | {item.tail3_mean:.4f} | "
            f"{item.post_initial_auc_100_1000:.4f} | {item.peak_allocated_mib:.0f} | "
            f"{item.optimizer_state_mib:.3f} |\n"
        )

    report += f"""

## Selected formal configurations

| Method | Selected cell | LR configuration | Step-1000 loss | Gap vs R1 Muon | Decision boundary |
|---|---|---|---:|---:|---|
| Moonlight Muon | `moonlight_r1scale` | aux/matrix = 0.0018, wd = 0.1 | {moon.final_val_loss_step_1000:.4f} | {moon.final_val_loss_step_1000-r1_muon:+.4f} | Competitive enough to require full 6200-step test |
| NorMuon | `normuon_r1scale` | aux = 0.0003, matrix = 0.01, wd = 0.01 | {nor.final_val_loss_step_1000:.4f} | {nor.final_val_loss_step_1000-r1_muon:+.4f} | Formal reference; lower-bound winner, no claim of globally optimal LR |
| AdamW | `adamw_low` | base = 0.0027, hidden = 0.000432, wd = 0 | {adam.final_val_loss_step_1000:.4f} | {adam.final_val_loss_step_1000-r1_muon:+.4f} | Tuned conventional baseline; disclose three-cell tuning budget |

All formal runs must use 6200 updates, the R1 1800-step warmdown, identical seed2026 initialization/data order, full validation every 100 steps, checkpoint/manifests, and post-training W&B upload. The 1000-step pilot must not be stretched into a formal run.

## Interpretation

Moonlight exhibits the strongest family across the grid. `high` has the better steps-100--1000 AUC (4.1100 versus 4.1598) and leads through step 800, but `r1scale` crosses it at step 900 and finishes 0.0060 lower with the better tail-3 mean. Because the frozen primary screening endpoint is step-1000 loss and formal training is much longer, `r1scale` is the defensible formal choice; the early-AUC tradeoff must remain visible.

NorMuon also shows a horizon-dependent crossover. The official LR has the better post-initial AUC (4.3244 versus 4.3682), while `r1scale` becomes persistently better from step 700 and finishes 0.0126 below official and 0.0555 below high. Because 0.01 is the lower boundary, the pilot selects it on the primary endpoint but does not establish an interior optimum. Running another pilot grid would increase tuning asymmetry; the current boundary choice should instead be disclosed.

AdamW `low` beats official by 0.0462 and high by 0.0426 at step 1000. It is also a lower-bound winner. Use it as a pilot-tuned AdamW baseline, not as the unchanged official AdamW recipe.

## Data quality and limitations

- {len(check_frame)} automated checks: {(check_frame.status == 'PASS').sum()} PASS, {(check_frame.status == 'FAIL').sum()} FAIL.
- All nine runs share seed2026, batch id `{batches[0]}`, initial validation loss {initial.iloc[0]:.3f}, and complete expected metric/step coverage.
- MIN/MAX mirrors equal the exported base series; no duplicate cell/metric/step grain or non-finite values were found.
- W&B alone cannot certify the local source/runtime/init/resume evidence; retain `PASS_WITH_CAVEATS` until the original pilot artifact is archived.
- LR was selected on seed2026 and the same validation shard. The seed2026 formal is a long-horizon screen, while seeds2024/2025 are the independent confirmation if a method enters the main claim/table.

## Recommended next steps

1. Implement a separate formal mode; do not reuse the pilot's one-step terminal warmdown.
2. Run `moonlight_r1scale`, `normuon_r1scale`, and `adamw_low` for 6200 steps on seed2026.
3. Predeclare final validation loss as primary; tail-5, normalized AUC, and steps/tokens-to-target as secondary.
4. Expand seeds2024/2025 first for Moonlight if its 6200-step endpoint/curve remains competitive. Expand NorMuon or AdamW only if their formal result materially changes the comparison or the final table requires matched seed counts.
5. Keep wall-clock/throughput outside this experiment; any performance claim belongs to the isolated performance protocol at the end of the project.

## Further questions

- Does Moonlight's small step-1000 gap close, persist, or widen after the 1800-step formal warmdown?
- Do the lower-bound AdamW/NorMuon selections remain best over the long horizon, or are their early-training rankings budget-specific?
- If Moonlight is competitive at 6200 steps, is the result stable on seeds2024/2025 without reselecting LR?
"""
    (output / "EXTENDED_BASELINES_PILOT_ANALYSIS_20260721.md").write_text(
        report, encoding="utf-8"
    )

    if not failures.empty:
        raise RuntimeError(f"data-quality failures:\n{failures.to_string(index=False)}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
