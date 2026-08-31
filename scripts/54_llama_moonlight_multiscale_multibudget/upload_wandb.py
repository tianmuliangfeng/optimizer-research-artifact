#!/usr/bin/env python3
"""Retryable post-training W&B upload for EX54 Moonlight evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import protocol as P


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    analysis = P.read_json(run_dir / "analysis/analysis_manifest.json")
    completion = P.read_json(run_dir / "completion_manifest.json")
    verification_path = run_dir / "analysis/verification_manifest.json"
    verification = P.read_json(verification_path)
    if not (
        analysis.get("passed") is True
        and completion.get("passed") is True
        and completion.get("independent_of_ex57") is True
        and completion.get("full_checkpoint_hash_verified") is True
        and completion.get("verification_manifest_sha256")
        == P.sha256_file(verification_path)
        and verification.get("passed") is True
        and verification.get("full_checkpoint_hash") is True
    ):
        raise RuntimeError("EX54 Moonlight upload requires the accepted full-hash local receipt")

    endpoints = read_csv(run_dir / "analysis/endpoint_results.csv")
    receipt_path = run_dir / "wandb_upload_manifest.json"
    if args.mode == "disabled":
        P.atomic_json(receipt_path, {
            "schema_version": "ex54_moonlight_wandb_upload_v1",
            "status": "disabled",
            "passed": True,
            "local_primary_unchanged": True,
            "uploaded_runs": 0,
        })
        print("EX54 Moonlight W&B upload disabled; local primary evidence remains accepted")
        return

    import wandb

    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in endpoints:
        grouped.setdefault((row["scale"], int(row["seed"])), []).append(row)
    receipts: list[dict[str, Any]] = []
    for (scale, seed), rows in sorted(grouped.items()):
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            mode=args.mode,
            name=f"ex54_moonlight_{scale}_seed{seed}",
            group=f"ex54_moonlight_{scale}",
            job_type="post_training_upload",
            config={
                "experiment": 54,
                "method": "moonlight",
                "scale": scale,
                "seed": seed,
                "timing_eligible": False,
                "local_primary": True,
            },
            reinit=True,
        )
        assert run is not None
        for row in sorted(rows, key=lambda item: int(item["target_step"])):
            run.log({
                "val/loss": float(row["final_val_loss"]),
                "val/tail5": float(row["tail5_val_loss"]),
                "val/normalized_auc": float(row["normalized_val_auc"]),
                "state/optimizer_bytes": int(row["optimizer_state_bytes"]),
                "state/moonlight_momentum_bytes": int(row["moonlight_momentum_state_bytes"]),
                "state/moonlight_matrix_optimizer_bytes": int(
                    row["moonlight_matrix_optimizer_state_bytes"]
                ),
                "memory/peak_allocated_bytes": int(row["peak_allocated_bytes"]),
                "tokens/target_step": int(row["target_step"]),
            }, step=int(row["target_step"]))
        receipts.append({"scale": scale, "seed": seed, "wandb_id": run.id, "url": run.url})
        run.finish()

    P.atomic_json(receipt_path, {
        "schema_version": "ex54_moonlight_wandb_upload_v1",
        "status": "uploaded",
        "passed": True,
        "local_primary_unchanged": True,
        "uploaded_runs": len(receipts),
        "receipts": receipts,
    })
    print(f"EX54 Moonlight uploaded {len(receipts)} W&B runs")


if __name__ == "__main__":
    main()
