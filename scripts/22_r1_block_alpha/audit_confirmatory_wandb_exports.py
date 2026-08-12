#!/usr/bin/env python3
"""Audit and preserve the R1 block-alpha multi-seed W&B confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-07-29.2"
METHOD_ALPHA = {
    "alpha0": 0.0,
    "alpha0p25": 0.25,
    "alpha0p50": 0.5,
    "alpha0p75": 0.75,
}
EXPECTED_SEEDS = [2024, 2025]
ALL_SEEDS = [2024, 2025, 2026]
EXPECTED_METRICS = {
    "val/loss": list(range(0, 6201, 100)),
    "train/loss_step": list(range(20, 6201, 20)),
    "time/train_s": list(range(0, 6201, 20)),
    "performance/step_avg_ms": list(range(40, 6201, 20)),
    "lr/adamw": list(range(0, 6201, 20)),
    "lr/matrix": list(range(0, 6201, 20)),
    "memory/peak_allocated_mib": [6200],
    "memory/optimizer_state_mib": [6200],
    "memory/k_state_mib": [6200],
}
HEADER_RE = re.compile(
    r"^(?P<run>mainconf_r1_block_alpha_confirmatory_"
    r"(?P<method>alpha0(?:p25|p50|p75)?)_"
    r"seed(?P<seed>2024|2025)_"
    r"(?P<stamp>\d{8}T\d{6}\+0000)) - "
    r"(?P<metric>.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports", nargs="+", type=Path)
    parser.add_argument("--official-summary", required=True, type=Path)
    parser.add_argument("--official-history", required=True, type=Path)
    parser.add_argument("--official-checks", required=True, type=Path)
    parser.add_argument("--seed2026-summary", required=True, type=Path)
    parser.add_argument("--seed2026-history", required=True, type=Path)
    parser.add_argument("--seed2026-checks", required=True, type=Path)
    parser.add_argument("--seed2026-verdict", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--local-artifact-root", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def normalized_auc(series: list[tuple[int, float]]) -> float:
    ordered = sorted(series)
    area = sum(
        (right[0] - left[0]) * (left[1] + right[1]) / 2.0
        for left, right in zip(ordered, ordered[1:])
    )
    return area / (ordered[-1][0] - ordered[0][0])


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            output[index] = rank
        start = end
    return output


def pearson(left: list[float], right: list[float]) -> float:
    lmean = sum(left) / len(left)
    rmean = sum(right) / len(right)
    numerator = sum(
        (lvalue - lmean) * (rvalue - rmean)
        for lvalue, rvalue in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - lmean) ** 2 for value in left)
        * sum((value - rmean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else math.nan


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(rankdata(left), rankdata(right))


def load_exports(
    paths: list[Path],
) -> tuple[
    dict[str, dict[tuple[int, str], list[tuple[int, float]]]],
    dict[tuple[int, str], str],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    data: dict[
        str, dict[tuple[int, str], list[tuple[int, float]]]
    ] = {}
    run_names: dict[tuple[int, str], str] = {}
    checks: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for source in sorted(path.resolve() for path in paths):
        with source.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        base_headers = [
            name
            for name in fieldnames
            if name != "Step"
            and not name.endswith(("__MIN", "__MAX"))
        ]
        parsed = [HEADER_RE.match(name) for name in base_headers]
        metrics = {
            match.group("metric") for match in parsed if match is not None
        }
        checks.append(
            {
                "check": f"{source.name}:nonempty",
                "passed": bool(rows) and "Step" in fieldnames,
                "detail": f"rows={len(rows)}",
            }
        )
        checks.append(
            {
                "check": f"{source.name}:headers",
                "passed": len(base_headers) == 8
                and all(match is not None for match in parsed),
                "detail": f"base_headers={len(base_headers)}",
            }
        )
        checks.append(
            {
                "check": f"{source.name}:one_metric",
                "passed": len(metrics) == 1,
                "detail": f"metrics={sorted(metrics)}",
            }
        )
        if not rows or len(base_headers) != 8 or any(
            match is None for match in parsed
        ) or len(metrics) != 1:
            continue
        metric = next(iter(metrics))
        checks.append(
            {
                "check": f"{source.name}:known_metric",
                "passed": metric in EXPECTED_METRICS,
                "detail": metric,
            }
        )
        if metric in data:
            raise RuntimeError(f"duplicate metric export: {metric}")
        data[metric] = {}
        identities = set()
        for header, match in zip(base_headers, parsed):
            assert match is not None
            seed = int(match.group("seed"))
            method = match.group("method")
            run_name = match.group("run")
            observed_metric = match.group("metric")
            identity = (seed, method)
            identities.add(identity)
            existing = run_names.setdefault(identity, run_name)
            checks.append(
                {
                    "check": (
                        f"{source.name}:{seed}:{method}:run_name_consistent"
                    ),
                    "passed": existing == run_name,
                    "detail": run_name,
                }
            )
            series: list[tuple[int, float]] = []
            mirror_passed = True
            finite_passed = True
            for row in rows:
                raw = row.get(header, "")
                if raw in ("", None):
                    continue
                value = float(raw)
                step = int(row["Step"])
                finite_passed &= math.isfinite(value)
                for suffix in ("__MIN", "__MAX"):
                    mirror = row.get(header + suffix, "")
                    mirror_passed &= mirror not in ("", None)
                    if mirror not in ("", None):
                        mirror_passed &= float(mirror) == value
                series.append((step, value))
            data[metric][identity] = series
            checks.extend(
                [
                    {
                        "check": (
                            f"{metric}:{seed}:{method}:finite"
                        ),
                        "passed": finite_passed,
                        "detail": f"points={len(series)}",
                    },
                    {
                        "check": (
                            f"{metric}:{seed}:{method}:min_max_mirror"
                        ),
                        "passed": mirror_passed,
                        "detail": f"points={len(series)}",
                    },
                    {
                        "check": (
                            f"{metric}:{seed}:{method}:step_grid"
                        ),
                        "passed": [step for step, _ in series]
                        == EXPECTED_METRICS.get(metric, []),
                        "detail": (
                            f"observed={len(series)};"
                            f"expected={len(EXPECTED_METRICS.get(metric, []))}"
                        ),
                    },
                ]
            )
        expected_identities = {
            (seed, method)
            for seed in EXPECTED_SEEDS
            for method in METHOD_ALPHA
        }
        checks.append(
            {
                "check": f"{metric}:identity_grid",
                "passed": identities == expected_identities,
                "detail": f"observed={sorted(identities)}",
            }
        )
        sources.append(
            {
                "file": source.name,
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "metric": metric,
                "rows": len(rows),
                "base_run_columns": len(base_headers),
            }
        )
    checks.append(
        {
            "check": "metric_families_exact",
            "passed": set(data) == set(EXPECTED_METRICS),
            "detail": f"observed={sorted(data)}",
        }
    )
    checks.append(
        {
            "check": "run_identity_grid_exact",
            "passed": set(run_names)
            == {
                (seed, method)
                for seed in EXPECTED_SEEDS
                for method in METHOD_ALPHA
            },
            "detail": f"observed={sorted(run_names)}",
        }
    )
    return data, run_names, checks, sources


def source_checks_pass(path: Path) -> bool:
    frame = pd.read_csv(path)
    status_column = "status"
    if status_column not in frame:
        return False
    return bool(frame[status_column].astype(str).str.upper().eq("PASS").all())


def new_run_summaries(
    data: dict[str, dict[tuple[int, str], list[tuple[int, float]]]],
    run_names: dict[tuple[int, str], str],
) -> list[dict[str, Any]]:
    output = []
    for seed in EXPECTED_SEEDS:
        for method, alpha in sorted(
            METHOD_ALPHA.items(), key=lambda item: item[1]
        ):
            key = (seed, method)
            validation = data["val/loss"][key]
            output.append(
                {
                    "method": method,
                    "endpoint_role": "new_dense_alpha",
                    "alpha": alpha,
                    "seed": seed,
                    "run_name": run_names[key],
                    "initial_val_loss": validation[0][1],
                    "final_val_loss": validation[-1][1],
                    "best_val_loss": min(
                        value for _, value in validation
                    ),
                    "tail5_val_loss_mean": float(
                        np.mean([value for _, value in validation[-5:]])
                    ),
                    "normalized_val_auc": normalized_auc(validation),
                    "final_train_loss_step": data[
                        "train/loss_step"
                    ][key][-1][1],
                    "train_time_s_descriptive_only": data[
                        "time/train_s"
                    ][key][-1][1],
                    "final_step_avg_ms_descriptive_only": data[
                        "performance/step_avg_ms"
                    ][key][-1][1],
                    "max_adamw_lr": max(
                        value for _, value in data["lr/adamw"][key]
                    ),
                    "max_matrix_lr": max(
                        value for _, value in data["lr/matrix"][key]
                    ),
                    "peak_memory_mib": data[
                        "memory/peak_allocated_mib"
                    ][key][-1][1],
                    "k_state_mib": data["memory/k_state_mib"][key][-1][1],
                    "optimizer_state_mib": data[
                        "memory/optimizer_state_mib"
                    ][key][-1][1],
                    "quality_eligible": True,
                    "memory_eligible": True,
                    "timing_eligible": False,
                    "local_manifest_verified": False,
                    "source": "W&B confirmation export 2026-07-29",
                }
            )
    return output


def matched_endpoint_rows(
    official_summary: Path,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(official_summary)
    frame = frame[
        frame["seed"].isin(EXPECTED_SEEDS)
        & frame["method"].isin(["diag", "block4"])
    ]
    output = []
    for row in frame.to_dict(orient="records"):
        method = str(row["method"])
        output.append(
            {
                "method": method,
                "endpoint_role": (
                    "reused_efficient_control"
                    if method == "diag"
                    else "reused_dense_endpoint"
                ),
                "alpha": 0.0 if method == "diag" else 1.0,
                "seed": int(row["seed"]),
                "run_name": str(row["run_name"]),
                "initial_val_loss": float(row["initial_val_loss"]),
                "final_val_loss": float(row["final_val_loss"]),
                "best_val_loss": float(row["best_val_loss"]),
                "tail5_val_loss_mean": float(
                    row["tail5_val_loss_mean"]
                ),
                "normalized_val_auc": float(row["normalized_val_auc"]),
                "final_train_loss_step": float(
                    row["final_train_loss_step"]
                ),
                "train_time_s_descriptive_only": float(
                    row["train_time_s_descriptive_only"]
                ),
                "final_step_avg_ms_descriptive_only": float(
                    row["final_step_avg_ms_descriptive_only"]
                ),
                "max_adamw_lr": float(row["max_adamw_lr"]),
                "max_matrix_lr": float(row["max_matrix_lr"]),
                "peak_memory_mib": float(row["peak_memory_mib"]),
                "k_state_mib": float(row["k_state_mib"]),
                "optimizer_state_mib": float(
                    row["optimizer_state_mib"]
                ),
                "quality_eligible": True,
                "memory_eligible": True,
                "timing_eligible": False,
                "local_manifest_verified": True,
                "source": "matched official R1 multiseed analysis",
            }
        )
    return output


def seed2026_rows(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    methods = {
        "diag",
        "alpha0",
        "alpha0p25",
        "alpha0p50",
        "alpha0p75",
        "block4",
    }
    frame = frame[
        (frame["seed"].astype(int) == 2026)
        & frame["method"].isin(methods)
    ].copy()
    frame["run_name"] = frame["method"].map(
        lambda method: f"seed2026_existing_{method}"
    )
    frame["local_manifest_verified"] = False
    return frame.to_dict(orient="records")


def curve_and_curvature(
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical_methods = {
        0.0: "alpha0",
        0.25: "alpha0p25",
        0.5: "alpha0p50",
        0.75: "alpha0p75",
        1.0: "block4",
    }
    curve_rows = []
    control_rows = []
    curvature_rows = []
    for seed in ALL_SEEDS:
        seed_frame = summary[summary["seed"].astype(int) == seed]
        points = []
        for alpha, method in canonical_methods.items():
            row = seed_frame[seed_frame["method"] == method]
            if len(row) != 1:
                raise RuntimeError(
                    f"missing canonical point seed={seed} method={method}"
                )
            record = row.iloc[0].to_dict()
            curve_rows.append(record)
            points.append(record)
        diag = seed_frame[seed_frame["method"] == "diag"]
        dense0 = seed_frame[seed_frame["method"] == "alpha0"]
        if len(diag) != 1 or len(dense0) != 1:
            raise RuntimeError(f"missing alpha0/diag control seed={seed}")
        diag_record = diag.iloc[0]
        dense_record = dense0.iloc[0]
        control_rows.append(
            {
                "seed": seed,
                "contrast": "dense_alpha0_minus_efficient_diag",
                "final_val_loss_delta": float(
                    dense_record["final_val_loss"]
                    - diag_record["final_val_loss"]
                ),
                "tail5_val_loss_delta": float(
                    dense_record["tail5_val_loss_mean"]
                    - diag_record["tail5_val_loss_mean"]
                ),
                "normalized_val_auc_delta": float(
                    dense_record["normalized_val_auc"]
                    - diag_record["normalized_val_auc"]
                ),
            }
        )
        by_alpha = {float(row["alpha"]): row for row in points}
        final_c = float(
            by_alpha[0.5]["final_val_loss"]
            - 0.5
            * (
                by_alpha[0.0]["final_val_loss"]
                + by_alpha[1.0]["final_val_loss"]
            )
        )
        tail_c = float(
            by_alpha[0.5]["tail5_val_loss_mean"]
            - 0.5
            * (
                by_alpha[0.0]["tail5_val_loss_mean"]
                + by_alpha[1.0]["tail5_val_loss_mean"]
            )
        )
        auc_c = float(
            by_alpha[0.5]["normalized_val_auc"]
            - 0.5
            * (
                by_alpha[0.0]["normalized_val_auc"]
                + by_alpha[1.0]["normalized_val_auc"]
            )
        )
        alphas = sorted(by_alpha)
        final_losses = [
            float(by_alpha[alpha]["final_val_loss"]) for alpha in alphas
        ]
        best_index = int(np.argmin(final_losses))
        curvature_rows.append(
            {
                "seed": seed,
                "final_curvature_c": final_c,
                "tail5_curvature_c": tail_c,
                "auc_curvature_c": auc_c,
                "alpha0p50_beats_alpha0_final": (
                    by_alpha[0.5]["final_val_loss"]
                    < by_alpha[0.0]["final_val_loss"]
                ),
                "alpha0p50_beats_alpha1_final": (
                    by_alpha[0.5]["final_val_loss"]
                    < by_alpha[1.0]["final_val_loss"]
                ),
                "alpha0p50_beats_both_endpoints_final": (
                    by_alpha[0.5]["final_val_loss"]
                    < by_alpha[0.0]["final_val_loss"]
                    and by_alpha[0.5]["final_val_loss"]
                    < by_alpha[1.0]["final_val_loss"]
                ),
                "best_alpha_descriptive": alphas[best_index],
                "spearman_alpha_vs_final_loss": spearman(
                    alphas, final_losses
                ),
            }
        )
    return (
        pd.DataFrame(curve_rows),
        pd.DataFrame(control_rows),
        pd.DataFrame(curvature_rows),
    )


def local_artifact_audit(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {
            "provided": False,
            "passed": False,
            "status": "pending",
            "detail": (
                "Remote confirmatory batch manifests, numerical-smoke "
                "certificates, and local result files were not supplied."
            ),
        }
    resolved = root.resolve()
    manifests = sorted(resolved.rglob("r1_manifest.json"))
    statuses = sorted(resolved.rglob("status.json"))
    return {
        "provided": True,
        "path": str(resolved),
        "manifest_files": len(manifests),
        "status_files": len(statuses),
        "passed": len(manifests) >= 8 and len(statuses) >= 8,
        "status": (
            "passed"
            if len(manifests) >= 8 and len(statuses) >= 8
            else "incomplete"
        ),
        "detail": "File-count gate only; a future revision must inspect fields.",
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")

    data, run_names, checks, source_rows = load_exports(args.exports)
    official_checks_ok = source_checks_pass(args.official_checks.resolve())
    seed2026_checks_ok = source_checks_pass(args.seed2026_checks.resolve())
    checks.extend(
        [
            {
                "check": "matched_official_source_checks",
                "passed": official_checks_ok,
                "detail": str(args.official_checks.resolve()),
            },
            {
                "check": "seed2026_source_checks",
                "passed": seed2026_checks_ok,
                "detail": str(args.seed2026_checks.resolve()),
            },
        ]
    )

    new_rows = new_run_summaries(data, run_names)
    official_rows = matched_endpoint_rows(args.official_summary.resolve())
    prior_rows = seed2026_rows(args.seed2026_summary.resolve())
    summary = pd.DataFrame(new_rows + official_rows + prior_rows)
    summary["seed"] = summary["seed"].astype(int)
    summary["alpha"] = summary["alpha"].astype(float)
    summary = summary.sort_values(
        ["seed", "alpha", "method"]
    ).reset_index(drop=True)

    expected_summary_grid = {
        (seed, method)
        for seed in ALL_SEEDS
        for method in [
            "diag",
            "alpha0",
            "alpha0p25",
            "alpha0p50",
            "alpha0p75",
            "block4",
        ]
    }
    observed_summary_grid = set(
        zip(summary["seed"], summary["method"])
    )
    checks.append(
        {
            "check": "three_seed_summary_grid",
            "passed": observed_summary_grid == expected_summary_grid,
            "detail": f"rows={len(summary)}",
        }
    )
    for seed in ALL_SEEDS:
        values = summary[summary["seed"] == seed]["initial_val_loss"]
        checks.append(
            {
                "check": f"seed{seed}:shared_initial_validation",
                "passed": len(values) == 6 and values.nunique() == 1,
                "detail": (
                    f"values={sorted(values.astype(float).unique())}"
                ),
            }
        )

    official_history = pd.read_csv(args.official_history.resolve())
    for seed in EXPECTED_SEEDS:
        for metric in ("lr/adamw", "lr/matrix", "val/loss"):
            anchor = official_history[
                (official_history["seed"].astype(int) == seed)
                & (official_history["method"] == "block4")
                & (official_history["metric"] == metric)
            ].sort_values("step")
            new_key = (seed, "alpha0")
            new_series = data[metric][new_key]
            checks.append(
                {
                    "check": f"seed{seed}:{metric}:matches_block4_grid",
                    "passed": anchor["step"].astype(int).tolist()
                    == [step for step, _ in new_series],
                    "detail": f"points={len(anchor)}",
                }
            )
            if metric.startswith("lr/"):
                anchor_values = anchor["value"].to_numpy(dtype=float)
                new_values = np.array(
                    [value for _, value in new_series],
                    dtype=float,
                )
                maximum_abs_delta = float(
                    np.max(np.abs(anchor_values - new_values))
                )
                checks.append(
                    {
                        "check": (
                            f"seed{seed}:{metric}:matches_block4_values"
                        ),
                        "passed": np.allclose(
                            anchor_values,
                            new_values,
                            rtol=0.0,
                            atol=1e-15,
                        ),
                        "detail": (
                            f"points={len(anchor)};"
                            f"maximum_abs_delta={maximum_abs_delta:.17g};"
                            "atol=1e-15;rtol=0"
                        ),
                    }
                )

    curve, controls, curvature = curve_and_curvature(summary)
    new_confirmation = curvature[curvature["seed"].isin(EXPECTED_SEEDS)]
    strict_new_seed_confirmation = bool(
        new_confirmation[
            "alpha0p50_beats_both_endpoints_final"
        ].all()
    )
    final_values = curvature["final_curvature_c"].to_numpy(dtype=float)
    curvature_summary = {
        "mean_final_curvature_c": float(final_values.mean()),
        "sample_sd_final_curvature_c": float(final_values.std(ddof=1)),
        "negative_seed_count": int((final_values < 0.0).sum()),
        "positive_seed_count": int((final_values > 0.0).sum()),
        "strict_alpha0p50_beats_both_endpoints_in_both_new_seeds": (
            strict_new_seed_confirmation
        ),
    }
    wandb_failed = sorted(
        row["check"] for row in checks if not row["passed"]
    )
    local_audit = local_artifact_audit(args.local_artifact_root)
    scientific_classification = (
        "strong_confirmatory_support"
        if not wandb_failed
        and strict_new_seed_confirmation
        and int((final_values < 0.0).sum()) == 3
        else "mixed_or_invalid"
    )
    delivery_status = (
        "accepted"
        if not wandb_failed and local_audit["passed"]
        else "wandb_complete_local_artifacts_pending"
        if not wandb_failed
        else "needs_revision"
    )

    output.mkdir(parents=True, exist_ok=False)
    raw_dir = output / "raw_wandb_exports"
    raw_dir.mkdir()
    matched_dir = output / "matched_sources"
    matched_dir.mkdir()
    for path in args.exports:
        shutil.copy2(path.resolve(), raw_dir / path.name)
    matched_sources = [
        args.official_summary,
        args.official_history,
        args.official_checks,
        args.seed2026_summary,
        args.seed2026_history,
        args.seed2026_checks,
        args.seed2026_verdict,
    ]
    for path in matched_sources:
        shutil.copy2(path.resolve(), matched_dir / path.name)

    history_rows = []
    for metric, identities in data.items():
        for (seed, method), series in identities.items():
            for step, value in series:
                history_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "alpha": METHOD_ALPHA[method],
                        "metric": metric,
                        "step": step,
                        "value": value,
                        "run_name": run_names[(seed, method)],
                    }
                )
    history = pd.DataFrame(history_rows).sort_values(
        ["metric", "seed", "alpha", "step"]
    )

    summary.to_csv(output / "alpha_run_summary.csv", index=False)
    curve.to_csv(output / "canonical_alpha_curve.csv", index=False)
    controls.to_csv(
        output / "dense_alpha0_vs_efficient_diag.csv", index=False
    )
    curvature.to_csv(output / "seed_curvature.csv", index=False)
    history.to_csv(
        output / "confirmatory_history_long.csv", index=False
    )
    pd.DataFrame(checks).to_csv(
        output / "data_quality_checks.csv", index=False
    )
    pd.DataFrame(source_rows).to_csv(
        output / "source_manifest.csv", index=False
    )
    write_json(
        output / "data_quality_audit.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "wandb_exports": {
                "files": len(args.exports),
                "new_runs": len(run_names),
                "checks": len(checks),
                "failed_checks": wandb_failed,
                "passed": not wandb_failed,
            },
            "matched_official_source_passed": official_checks_ok,
            "matched_seed2026_source_passed": seed2026_checks_ok,
            "local_artifacts": local_audit,
            "delivery_status": delivery_status,
        },
    )
    write_json(
        output / "important_results.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "experiment": "R1 block-alpha multi-seed confirmation",
            "primary_endpoint": "validation loss at step 6200",
            "scientific_classification": scientific_classification,
            "delivery_status": delivery_status,
            "curvature_summary": curvature_summary,
            "seed_curvature": json.loads(
                curvature.to_json(orient="records")
            ),
            "dense_alpha0_vs_efficient_diag": json.loads(
                controls.to_json(orient="records")
            ),
            "timing_eligible": False,
            "memory_claim": (
                "No alpha-dependent memory claim: all dense alpha cells "
                "store the same block-local dense state."
            ),
            "required_caveats": [
                "Seed2026 is exploratory and motivated the confirmation; it is not an independent confirmatory seed.",
                "The best alpha selected from the same five points is descriptive, not universally optimal.",
                "Concurrent-node timing is ineligible for paper efficiency claims.",
                "Formal delivery remains open until remote local manifests and smoke/hash certificates are audited.",
            ],
        },
    )
    artifacts = sorted(
        [
            "alpha_run_summary.csv",
            "canonical_alpha_curve.csv",
            "confirmatory_history_long.csv",
            "data_quality_audit.json",
            "data_quality_checks.csv",
            "dense_alpha0_vs_efficient_diag.csv",
            "important_results.json",
            "matched_sources/",
            "raw_wandb_exports/",
            "seed_curvature.csv",
            "source_manifest.csv",
        ]
    )
    write_json(
        output / "audit_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "wandb_passed": not wandb_failed,
            "scientific_classification": scientific_classification,
            "delivery_status": delivery_status,
            "artifacts": artifacts,
        },
    )
    print(f"R1 block-alpha audit: {output}")
    print(f"W&B checks passed: {not wandb_failed}")
    print(f"Scientific classification: {scientific_classification}")
    print(f"Delivery status: {delivery_status}")
    if wandb_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
