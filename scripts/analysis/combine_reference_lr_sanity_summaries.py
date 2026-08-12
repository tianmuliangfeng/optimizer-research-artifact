"""Combine B12 and B24 reference-LR summaries into one comparable table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


UNIFIED_COLUMNS = [
    "suite",
    "method",
    "run_name",
    "seed",
    "n_layer",
    "n_embd",
    "max_iters",
    "matrix_lr",
    "val_last_checkpoint_step",
    "val_loss_at_last_checkpoint",
    "best_val_loss",
    "best_val_step",
    "late_val_mean",
    "normalized_val_auc",
    "total_k_state_mib",
    "cproj_k_state_mib",
    "non_cproj_k_state_mib",
    "k_state_released_fraction",
    "cuda_allocated_mib",
    "cuda_full_run_peak_mib",
    "time_last_logged_step",
    "time_elapsed_last_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b12-summary", required=True, type=Path)
    parser.add_argument("--b24-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def normalize_summary(frame: pd.DataFrame, default_suite: str) -> pd.DataFrame:
    """Normalize both the current grid schema and the legacy B24 gate schema."""
    normalized = frame.copy()
    aliases = {
        "final_val_step": "val_last_checkpoint_step",
        "final_val_loss": "val_loss_at_last_checkpoint",
        "late_val_mean_last20pct": "late_val_mean",
        "cuda_allocated_last_mib": "cuda_allocated_mib",
        "last_time_logged_step": "time_last_logged_step",
        "final_common_val_step": "val_last_checkpoint_step",
        "final_common_val_loss": "val_loss_at_last_checkpoint",
        "late_val_mean_steps1500_2500": "late_val_mean",
        "normalized_val_auc_steps0_2500": "normalized_val_auc",
        "time_elapsed_step2980_s": "time_elapsed_last_s",
    }
    for source, destination in aliases.items():
        if source in normalized.columns and destination not in normalized.columns:
            normalized = normalized.rename(columns={source: destination})
    if "suite" not in normalized.columns:
        normalized["suite"] = default_suite
    return normalized.reindex(columns=UNIFIED_COLUMNS)


def main() -> None:
    args = parse_args()
    b12 = pd.read_csv(args.b12_summary)
    b24 = pd.read_csv(args.b24_summary)

    b12_unified = normalize_summary(b12, "owt12l_3k")
    b24_unified = normalize_summary(b24, "owt24l_3k")

    combined = pd.concat(
        [b12_unified, b24_unified],
        ignore_index=True,
    ).sort_values(["n_layer", "matrix_lr", "val_loss_at_last_checkpoint", "method"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {len(combined)} runs to {args.output}")


if __name__ == "__main__":
    main()
