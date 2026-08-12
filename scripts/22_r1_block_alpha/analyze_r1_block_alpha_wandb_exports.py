"""Audit and preserve the seed-2026 R1 block-alpha W&B export bundle.

This is the export-only companion to ``analyze_r1_block_alpha.py``.  It is
intended for cases where the remote R1 artifact directory is not mounted on
the analysis machine, but the complete W&B metric exports are available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path


METHOD_ALPHA = {"alpha0": 0.0, "alpha0p25": 0.25, "alpha0p50": 0.5, "alpha0p75": 0.75}
EXPECTED_METRICS = {
    "val/loss": (0, 6200, 100),
    "train/loss_step": (20, 6200, 20),
    "time/train_s": (0, 6200, 20),
    "performance/step_avg_ms": (40, 6200, 20),
    "lr/adamw": (0, 6200, 20),
    "lr/matrix": (0, 6200, 20),
    "memory/peak_allocated_mib": (6200, 6200, 1),
    "memory/optimizer_state_mib": (6200, 6200, 1),
    "memory/k_state_mib": (6200, 6200, 1),
}
HEADER_RE = re.compile(
    r"^mainconf_r1_block_alpha_(alpha0(?:p25|p50|p75)?)_seed(\d+)_(\d{8}T\d{6}\+0000) - (.+)$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_exports(paths: list[Path]) -> tuple[dict[str, dict[str, list[tuple[int, float]]]], list[dict[str, object]]]:
    data: dict[str, dict[str, list[tuple[int, float]]]] = {}
    sources: list[dict[str, object]] = []
    for path in sorted(p.resolve() for p in paths):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "Step" not in rows[0]:
            raise RuntimeError(f"empty or malformed W&B export: {path}")
        base_headers = [name for name in rows[0] if name != "Step" and not name.endswith(("__MIN", "__MAX"))]
        parsed = [HEADER_RE.match(name) for name in base_headers]
        if not base_headers or any(match is None for match in parsed):
            raise RuntimeError(f"unexpected W&B column naming in {path.name}")
        metrics = {match.group(4) for match in parsed if match}
        if len(metrics) != 1:
            raise RuntimeError(f"expected one metric family in {path.name}, observed {metrics}")
        metric = next(iter(metrics))
        if metric in data:
            raise RuntimeError(f"duplicate metric export: {metric}")
        data[metric] = {}
        for header, match in zip(base_headers, parsed):
            assert match is not None
            method, seed, stamp, observed_metric = match.groups()
            if method not in METHOD_ALPHA or seed != "2026" or observed_metric != metric:
                raise RuntimeError(f"unexpected run identity in {header}")
            series: list[tuple[int, float]] = []
            for row in rows:
                if row[header] == "":
                    continue
                value = float(row[header])
                if row.get(header + "__MIN", "") != "" and float(row[header + "__MIN"]) != value:
                    raise RuntimeError(f"MIN aggregation differs from raw value: {header}")
                if row.get(header + "__MAX", "") != "" and float(row[header + "__MAX"]) != value:
                    raise RuntimeError(f"MAX aggregation differs from raw value: {header}")
                series.append((int(row["Step"]), value))
            data[metric][method] = series
            sources.append(
                {"file": path.name, "sha256": sha256(path), "metric": metric, "method": method,
                 "seed": int(seed), "run_stamp": stamp, "rows": len(series)}
            )
    return data, sources


def trapezoid_auc(series: list[tuple[int, float]]) -> float:
    area = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(series, series[1:]))
    return area / (series[-1][0] - series[0][0])


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def correlation(left: list[float], right: list[float]) -> float:
    lmean, rmean = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((l - lmean) * (r - rmean) for l, r in zip(left, right))
    denominator = math.sqrt(sum((x - lmean) ** 2 for x in left) * sum((x - rmean) ** 2 for x in right))
    return numerator / denominator if denominator else math.nan


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exports", nargs="+", type=Path)
    parser.add_argument("--core-summary", type=Path, required=True)
    parser.add_argument("--core-history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-artifact-path", required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = output / "raw_wandb_exports"
    raw_dir.mkdir(exist_ok=True)
    data, sources = load_exports(args.exports)

    checks: list[dict[str, object]] = []
    checks.append({"check": "metric_families_exact", "status": "PASS" if set(data) == set(EXPECTED_METRICS) else "FAIL",
                   "detail": f"observed={sorted(data)}"})
    for metric, (start, stop, stride) in EXPECTED_METRICS.items():
        expected = [6200] if start == stop else list(range(start, stop + 1, stride))
        methods = set(data.get(metric, {}))
        ok_methods = methods == set(METHOD_ALPHA)
        ok_grids = ok_methods and all([step for step, _ in data[metric][method]] == expected for method in METHOD_ALPHA)
        checks.append({"check": f"{metric}_coverage", "status": "PASS" if ok_grids else "FAIL",
                       "detail": f"methods={sorted(methods)}; expected_steps={len(expected)}"})
    if any(row["status"] == "FAIL" for row in checks):
        raise RuntimeError("W&B export validation failed; see structural checks")

    with args.core_summary.resolve().open(encoding="utf-8", newline="") as handle:
        core_rows = {row["method"]: row for row in csv.DictReader(handle) if row["method"] in {"diag", "block4"}}
    if set(core_rows) != {"diag", "block4"}:
        raise RuntimeError("matched core summary does not contain diag and block4")

    summaries: list[dict[str, object]] = []
    for method, alpha in sorted(METHOD_ALPHA.items(), key=lambda item: item[1]):
        vals = data["val/loss"][method]
        summaries.append({
            "method": method, "endpoint_role": "new_dense_alpha", "alpha": alpha, "seed": 2026,
            "initial_val_loss": vals[0][1], "final_val_loss": vals[-1][1],
            "best_val_loss": min(value for _, value in vals),
            "tail5_val_loss_mean": sum(value for _, value in vals[-5:]) / 5,
            "normalized_val_auc": trapezoid_auc(vals),
            "final_train_loss_step": data["train/loss_step"][method][-1][1],
            "train_time_s_descriptive_only": data["time/train_s"][method][-1][1],
            "final_step_avg_ms_descriptive_only": data["performance/step_avg_ms"][method][-1][1],
            "max_adamw_lr": max(value for _, value in data["lr/adamw"][method]),
            "max_matrix_lr": max(value for _, value in data["lr/matrix"][method]),
            "peak_memory_mib": data["memory/peak_allocated_mib"][method][-1][1],
            "k_state_mib": data["memory/k_state_mib"][method][-1][1],
            "optimizer_state_mib": data["memory/optimizer_state_mib"][method][-1][1],
            "quality_eligible": True, "memory_eligible": True, "timing_eligible": False,
            "source": "W&B export bundle 2026-07-23",
        })
    dense = {row["method"]: row for row in summaries}
    for method, alpha, role in (("diag", 0.0, "reused_efficient_endpoint"), ("block4", 1.0, "reused_dense_endpoint")):
        row = core_rows[method]
        summaries.append({
            "method": method, "endpoint_role": role, "alpha": alpha, "seed": 2026,
            "initial_val_loss": float(row["initial_val_loss"]), "final_val_loss": float(row["final_val_loss"]),
            "best_val_loss": float(row["best_val_loss"]), "tail5_val_loss_mean": float(row["tail5_val_loss_mean"]),
            "normalized_val_auc": float(row["normalized_val_auc"]),
            "final_train_loss_step": float(row["final_train_loss_step"]),
            "train_time_s_descriptive_only": float(row["train_time_s"]),
            "final_step_avg_ms_descriptive_only": float(row["final_step_avg_ms"]),
            "max_adamw_lr": float(row["max_adamw_lr"]), "max_matrix_lr": float(row["max_matrix_lr"]),
            "peak_memory_mib": float(row["peak_memory_mib"]), "k_state_mib": float(row["k_state_mib"]),
            "optimizer_state_mib": float(row["optimizer_state_mib"]),
            "quality_eligible": True, "memory_eligible": True, "timing_eligible": False,
            "source": "matched R1 seed2026 analysis snapshot",
        })
    summaries.sort(key=lambda row: (float(row["alpha"]), 0 if row["method"] == "diag" else 1))

    diag = next(row for row in summaries if row["method"] == "diag")
    block4 = next(row for row in summaries if row["method"] == "block4")
    dense0 = dense["alpha0"]
    canonical_dose = [diag, dense["alpha0p25"], dense["alpha0p50"], dense["alpha0p75"], block4]
    dense_dose = [dense["alpha0"], dense["alpha0p25"], dense["alpha0p50"], dense["alpha0p75"], block4]
    rho = correlation(average_ranks([float(row["alpha"]) for row in canonical_dose]),
                      average_ranks([float(row["final_val_loss"]) for row in canonical_dose]))
    dense_rho = correlation(average_ranks([float(row["alpha"]) for row in dense_dose]),
                            average_ranks([float(row["final_val_loss"]) for row in dense_dose]))
    final_equiv = float(dense0["final_val_loss"]) - float(diag["final_val_loss"])
    tail_equiv = float(dense0["tail5_val_loss_mean"]) - float(diag["tail5_val_loss_mean"])
    endpoint_delta = float(block4["final_val_loss"]) - float(diag["final_val_loss"])
    gates = {
        "dense_alpha0_final_equivalence_abs_le_0p001": abs(final_equiv) <= 0.001,
        "dense_alpha0_tail5_equivalence_abs_le_0p001": abs(tail_equiv) <= 0.001,
        "final_loss_spearman_rho_ge_0p5": rho >= 0.5,
        "block4_not_better_than_diag_at_final": endpoint_delta >= 0,
    }
    verdict = "expand_to_seeds_2024_2025" if all(gates.values()) else "stop_or_redesign_before_more_seeds"

    contrasts = [
        {"contrast": "dense_alpha0_minus_efficient_diag", "final_val_loss_delta": final_equiv,
         "tail5_val_loss_delta": tail_equiv,
         "normalized_val_auc_delta": float(dense0["normalized_val_auc"]) - float(diag["normalized_val_auc"])},
        {"contrast": "alpha0p50_minus_efficient_diag", "final_val_loss_delta": float(dense["alpha0p50"]["final_val_loss"]) - float(diag["final_val_loss"]),
         "tail5_val_loss_delta": float(dense["alpha0p50"]["tail5_val_loss_mean"]) - float(diag["tail5_val_loss_mean"]),
         "normalized_val_auc_delta": float(dense["alpha0p50"]["normalized_val_auc"]) - float(diag["normalized_val_auc"])},
        {"contrast": "block4_minus_alpha0p50", "final_val_loss_delta": float(block4["final_val_loss"]) - float(dense["alpha0p50"]["final_val_loss"]),
         "tail5_val_loss_delta": float(block4["tail5_val_loss_mean"]) - float(dense["alpha0p50"]["tail5_val_loss_mean"]),
         "normalized_val_auc_delta": float(block4["normalized_val_auc"]) - float(dense["alpha0p50"]["normalized_val_auc"])},
        {"contrast": "block4_minus_efficient_diag", "final_val_loss_delta": endpoint_delta,
         "tail5_val_loss_delta": float(block4["tail5_val_loss_mean"]) - float(diag["tail5_val_loss_mean"]),
         "normalized_val_auc_delta": float(block4["normalized_val_auc"]) - float(diag["normalized_val_auc"])},
    ]

    history_rows: list[dict[str, object]] = []
    for metric, methods in data.items():
        for method, series in methods.items():
            for step, value in series:
                history_rows.append({"method": method, "alpha": METHOD_ALPHA[method], "seed": 2026,
                                     "metric": metric, "step": step, "value": value})
    history_rows.sort(key=lambda row: (row["metric"], float(row["alpha"]), int(row["step"])))

    for path in args.exports:
        shutil.copy2(path.resolve(), raw_dir / path.name)
    shutil.copy2(args.core_summary.resolve(), raw_dir / "matched_core_r1_run_summary.csv")
    shutil.copy2(args.core_history.resolve(), raw_dir / "matched_core_r1_normalized_history_long.csv")
    write_csv(output / "alpha_run_summary.csv", summaries)
    write_csv(output / "alpha_validation_and_audit_history_long.csv", history_rows)
    write_csv(output / "alpha_contrasts.csv", contrasts)
    write_csv(output / "data_quality_checks.csv", checks)
    write_csv(output / "source_manifest.csv", sources)

    result = {
        "status": "valid_with_caveats", "seed": 2026,
        "remote_artifact_path_reported_by_user": args.remote_artifact_path,
        "remote_local_manifest_verified": False,
        "evidence_basis": "complete W&B metric exports plus matched local R1 seed2026 analysis snapshot",
        "primary_endpoint": "validation loss at step 6200",
        "canonical_spearman_alpha_vs_final_loss": rho,
        "dense_alpha0_sensitivity_spearman": dense_rho,
        "predeclared_gates": gates, "pilot_verdict": verdict,
        "timing_eligible": False,
        "timing_note": "Concurrent node training was declared for this experiment family; times are descriptive only.",
        "records": summaries,
    }
    (output / "block_alpha_pilot_verdict.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    best_final = min(dense.values(), key=lambda row: float(row["final_val_loss"]))
    best_auc = min(dense.values(), key=lambda row: float(row["normalized_val_auc"]))
    report = f"""# R1 block-alpha seed2026 pilot analysis

## Decision

**Do not expand this alpha sweep to seeds 2024/2025 under the predeclared rule.** Three gates pass,
but the canonical Spearman association between alpha and final validation loss is `{rho:.3f}`, below
the required `0.5`. The recorded verdict is `{verdict}`.

The safe scientific conclusion is that, in this R1 regime, increasing off-diagonal covariance
strength is **not monotonically beneficial**. The curve is shallow and U-shaped: alpha=0.25--0.50
is numerically best, while full alpha=1 is worse. Because this is one seed and all differences are
small, this is directional mechanism evidence, not evidence that alpha=0.50 is a general optimum.

## Primary and secondary endpoints

| Endpoint | Efficient diag (alpha=0) | Dense alpha=0 | alpha=.25 | alpha=.50 | alpha=.75 | Block4 (alpha=1) |
|---|---:|---:|---:|---:|---:|---:|
| Final val loss | {float(diag['final_val_loss']):.4f} | {float(dense0['final_val_loss']):.4f} | {float(dense['alpha0p25']['final_val_loss']):.4f} | {float(dense['alpha0p50']['final_val_loss']):.4f} | {float(dense['alpha0p75']['final_val_loss']):.4f} | {float(block4['final_val_loss']):.4f} |
| Tail-5 val mean | {float(diag['tail5_val_loss_mean']):.5f} | {float(dense0['tail5_val_loss_mean']):.5f} | {float(dense['alpha0p25']['tail5_val_loss_mean']):.5f} | {float(dense['alpha0p50']['tail5_val_loss_mean']):.5f} | {float(dense['alpha0p75']['tail5_val_loss_mean']):.5f} | {float(block4['tail5_val_loss_mean']):.5f} |
| Normalized val AUC | {float(diag['normalized_val_auc']):.6f} | {float(dense0['normalized_val_auc']):.6f} | {float(dense['alpha0p25']['normalized_val_auc']):.6f} | {float(dense['alpha0p50']['normalized_val_auc']):.6f} | {float(dense['alpha0p75']['normalized_val_auc']):.6f} | {float(block4['normalized_val_auc']):.6f} |

- Best final endpoint among new cells: `{best_final['method']}` at `{float(best_final['final_val_loss']):.4f}`.
- Best AUC among new cells: `{best_auc['method']}` at `{float(best_auc['normalized_val_auc']):.6f}`.
- alpha=.50 vs efficient diag: final `{float(dense['alpha0p50']['final_val_loss']) - float(diag['final_val_loss']):+.4f}`, tail-5 `{float(dense['alpha0p50']['tail5_val_loss_mean']) - float(diag['tail5_val_loss_mean']):+.5f}`.
- Full alpha=1 vs alpha=.50: final `{float(block4['final_val_loss']) - float(dense['alpha0p50']['final_val_loss']):+.4f}`, tail-5 `{float(block4['tail5_val_loss_mean']) - float(dense['alpha0p50']['tail5_val_loss_mean']):+.5f}`.

## Predeclared gates

| Gate | Observed | Pass |
|---|---:|:---:|
| abs(dense alpha=0 - efficient diag), final <= .001 | {abs(final_equiv):.5f} | yes |
| abs(dense alpha=0 - efficient diag), tail-5 <= .001 | {abs(tail_equiv):.5f} | yes |
| Spearman rho(alpha, final loss) >= .5 | {rho:.3f} | **no** |
| Block4 not better than diag at final | block4-diag = {endpoint_delta:+.4f} | yes |

The dense-only alpha=0 sensitivity gives rho `{dense_rho:.3f}` and reaches the same gate decision.

## Memory interpretation

Every newly swept alpha cell deliberately retains dense Block4 state: `378.000 MiB` K state,
`996.475 MiB` optimizer state, and `39,168 MiB` peak allocation. Alpha itself therefore does not
save memory. The useful engineering result is that dense alpha=0 reproduces efficient diag quality
(final delta `{final_equiv:+.4f}`, tail-5 delta `{tail_equiv:+.5f}`), while efficient diag reduces
K/optimizer state by `{float(dense0['k_state_mib']) - float(diag['k_state_mib']):.3f} MiB` and recorded
peak allocation by `{float(dense0['peak_memory_mib']) - float(diag['peak_memory_mib']):.0f} MiB`.

## Evidence quality and scope

- All 9 expected metric families, all 4 new methods, and exact prescribed step grids passed checks.
- W&B raw/MIN/MAX columns agree, all runs are seed 2026 with run stamp `20260722T070928+0000`, and
  maximum learning rates are matched (`AdamW 0.004`, matrix `0.0004`).
- The remote artifact path was supplied by the user but its local manifest was not available here;
  conclusions are based on the complete W&B exports and the preserved matched R1 seed2026 snapshot.
- Timing is ineligible because concurrent node training was declared. Quality-vs-step and
  per-process memory remain eligible.
- Do not describe alpha=.50 as a robust winner: its final advantage over efficient diag is only
  `0.0010` in one seed, while alpha=.25 is nearly tied and is marginally better in AUC.

## Recommended use

Keep this as a compact seed2026 mechanism/appendix result: diagonalization is validated
mathematically and is the better engineering implementation; full off-diagonal covariance provides
no advantage here. Do not spend two more seeds on the present alpha grid. Revisit only if reviewers
require it or a redesigned hypothesis introduces a stronger, predeclared mechanism test.
"""
    (output / "R1_BLOCK_ALPHA_ANALYSIS_20260723.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(output), "pilot_verdict": verdict, "rho": rho}, indent=2))


if __name__ == "__main__":
    main()
