"""Analyze partial formal reference-reset validation-loss W&B exports."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


RUN_RE = re.compile(
    r"^mainconf_reference_lr001_(?P<suite>.+)_"
    r"(?P<label>diag_cproj_k|no_cproj_k|paper_block4|muon_blog)_seed(?P<seed>[0-9]+)$"
)
METHODS = {
    "diag_cproj_k": "diag",
    "no_cproj_k": "none",
    "paper_block4": "block4",
    "muon_blog": "muon",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_column(column: str) -> dict[str, object]:
    if not column.endswith(" - val/loss"):
        raise ValueError(f"Unexpected base column: {column}")
    run_name = column[: -len(" - val/loss")]
    match = RUN_RE.match(run_name)
    if match is None:
        raise ValueError(f"Unexpected run name: {run_name}")
    return {
        "run_name": run_name,
        "suite": match.group("suite"),
        "method": METHODS[match.group("label")],
        "seed": int(match.group("seed")),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw_wandb_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)

    long_parts: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    duplicate_equal: list[bool] = []

    for source_arg in args.inputs:
        source = source_arg.resolve()
        frame = pd.read_csv(source)
        base_columns = [
            column
            for column in frame.columns
            if column != "Step" and not column.endswith("__MIN") and not column.endswith("__MAX")
        ]
        seeds = set()
        for column in base_columns:
            metadata = parse_column(column)
            seeds.add(metadata["seed"])
            part = pd.DataFrame(
                {
                    "step": pd.to_numeric(frame["Step"], errors="coerce"),
                    "val_loss": pd.to_numeric(frame[column], errors="coerce"),
                }
            )
            for key, value in metadata.items():
                part[key] = value
            long_parts.append(part)
            for suffix in ("__MIN", "__MAX"):
                other = column + suffix
                if other in frame.columns:
                    duplicate_equal.append(
                        bool(
                            np.allclose(
                                pd.to_numeric(frame[column], errors="coerce"),
                                pd.to_numeric(frame[other], errors="coerce"),
                                rtol=0,
                                atol=0,
                                equal_nan=True,
                            )
                        )
                    )

        seed_label = "_".join(str(seed) for seed in sorted(seeds))
        destination = raw_dir / f"val_loss_seed{seed_label}.csv"
        shutil.copy2(source, destination)
        inventory.append(
            {
                "source_path": str(source),
                "copied_path": str(destination),
                "sha256": sha256(source),
                "csv_rows": len(frame),
                "base_run_columns": len(base_columns),
                "step_min": pd.to_numeric(frame["Step"], errors="coerce").min(),
                "step_max": pd.to_numeric(frame["Step"], errors="coerce").max(),
                "seeds": seed_label,
            }
        )

    long = pd.concat(long_parts, ignore_index=True).sort_values(["seed", "method", "step"])
    run_identity = long[["run_name", "suite", "method", "seed"]].drop_duplicates()

    checks: list[dict[str, str]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "details": detail})

    observed_seeds = sorted(run_identity["seed"].unique())
    observed_methods = sorted(run_identity["method"].unique())
    suites = sorted(run_identity["suite"].unique())
    check("input_files", len(args.inputs) == len(observed_seeds), f"files={len(args.inputs)}, seeds={observed_seeds}")
    check("one_suite", len(suites) == 1, f"suites={suites}")
    check("method_coverage", set(observed_methods) == set(METHODS.values()), f"methods={observed_methods}")
    check(
        "complete_seed_method_grid",
        len(run_identity) == len(observed_seeds) * len(METHODS),
        f"runs={len(run_identity)}, expected={len(observed_seeds) * len(METHODS)}",
    )
    counts = long.dropna(subset=["val_loss"]).groupby("run_name").size()
    max_steps = long.dropna(subset=["val_loss"]).groupby("run_name")["step"].max()
    min_steps = long.dropna(subset=["val_loss"]).groupby("run_name")["step"].min()
    check(
        "uniform_step_coverage",
        counts.nunique() == 1 and min_steps.nunique() == 1 and max_steps.nunique() == 1,
        f"points={sorted(counts.unique())}, min={sorted(min_steps.unique())}, max={sorted(max_steps.unique())}",
    )
    check("finite_loss", bool(np.isfinite(long["val_loss"].dropna()).all()), "all exported values finite")
    initial = long.loc[long["step"] == long["step"].min()]
    per_seed_initial_unique = initial.groupby("seed")["val_loss"].nunique()
    check(
        "identical_initial_loss_within_seed",
        bool((per_seed_initial_unique == 1).all()),
        f"unique_counts={per_seed_initial_unique.to_dict()}",
    )
    check(
        "wandb_min_max_duplicates",
        bool(duplicate_equal and all(duplicate_equal)),
        f"checked={len(duplicate_equal)}, all_identical={all(duplicate_equal)}",
    )

    summaries: list[dict[str, object]] = []
    for run_name, frame in long.groupby("run_name"):
        frame = frame.dropna(subset=["val_loss"]).sort_values("step")
        identity = frame.iloc[0]
        final_step = float(frame["step"].max())
        late = frame.loc[frame["step"] >= final_step - 2000]
        best_row = frame.loc[frame["val_loss"].idxmin()]
        span = float(frame["step"].max() - frame["step"].min())
        summaries.append(
            {
                "suite": identity["suite"],
                "seed": int(identity["seed"]),
                "method": identity["method"],
                "run_name": run_name,
                "last_val_step": final_step,
                "val_loss_last": float(frame.iloc[-1]["val_loss"]),
                "best_val_loss": float(best_row["val_loss"]),
                "best_val_step": float(best_row["step"]),
                "late_val_mean_steps_last2000": float(late["val_loss"].mean()),
                "normalized_val_auc": float(np.trapezoid(frame["val_loss"], frame["step"]) / span),
                "val_rebound_last_minus_best": float(frame.iloc[-1]["val_loss"] - best_row["val_loss"]),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(["seed", "val_loss_last"])

    paired_rows: list[dict[str, object]] = []
    for seed, seed_rows in summary.groupby("seed"):
        muon = seed_rows.loc[seed_rows["method"] == "muon"].iloc[0]
        seed_long = long.loc[long["seed"] == seed]
        muon_curve = seed_long.loc[seed_long["method"] == "muon", ["step", "val_loss"]].rename(
            columns={"val_loss": "muon_val_loss"}
        )
        for row in seed_rows.itertuples(index=False):
            method_curve = seed_long.loc[
                seed_long["method"] == row.method, ["step", "val_loss"]
            ].merge(muon_curve, on="step", validate="one_to_one")
            nonzero = method_curve.loc[method_curve["step"] > 0]
            muon_wins = nonzero["muon_val_loss"] < nonzero["val_loss"]
            paired_rows.append(
                {
                    "seed": seed,
                    "method": row.method,
                    "val_loss_last": row.val_loss_last,
                    "last_delta_vs_muon": row.val_loss_last - muon["val_loss_last"],
                    "late_mean_delta_vs_muon": row.late_val_mean_steps_last2000
                    - muon["late_val_mean_steps_last2000"],
                    "auc_delta_vs_muon": row.normalized_val_auc - muon["normalized_val_auc"],
                    "muon_lower_checkpoint_count_excluding_step0": int(muon_wins.sum()),
                    "compared_checkpoint_count_excluding_step0": int(len(nonzero)),
                    "first_checkpoint_muon_lower": (
                        float(nonzero.loc[muon_wins, "step"].min()) if bool(muon_wins.any()) else np.nan
                    ),
                }
            )
    paired = pd.DataFrame(paired_rows).sort_values(["seed", "last_delta_vs_muon"])

    aggregate = (
        summary.groupby("method", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            val_loss_last_mean=("val_loss_last", "mean"),
            val_loss_last_std=("val_loss_last", "std"),
            best_val_loss_mean=("best_val_loss", "mean"),
            late_val_mean=("late_val_mean_steps_last2000", "mean"),
            normalized_val_auc_mean=("normalized_val_auc", "mean"),
        )
        .sort_values("val_loss_last_mean")
    )
    muon_mean = float(aggregate.loc[aggregate["method"] == "muon", "val_loss_last_mean"].iloc[0])
    aggregate["last_mean_delta_vs_muon"] = aggregate["val_loss_last_mean"] - muon_mean

    mean_curve = (
        long.groupby(["method", "step"], as_index=False)
        .agg(val_loss_mean=("val_loss", "mean"), val_loss_std=("val_loss", "std"), seeds=("seed", "nunique"))
        .sort_values(["step", "val_loss_mean"])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory).to_csv(output_dir / "source_inventory.csv", index=False)
    pd.DataFrame(checks).to_csv(output_dir / "data_quality_checks.csv", index=False)
    long.to_csv(output_dir / "val_loss_long.csv", index=False)
    summary.to_csv(output_dir / "run_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_vs_muon.csv", index=False)
    aggregate.to_csv(output_dir / "method_aggregate.csv", index=False)
    mean_curve.to_csv(output_dir / "mean_val_loss_curve.csv", index=False)

    failed = sum(row["status"] != "PASS" for row in checks)
    print(f"runs={len(summary)}, seeds={observed_seeds}, failed_checks={failed}")
    print(aggregate.to_string(index=False))
    print("\nPaired vs Muon:")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
