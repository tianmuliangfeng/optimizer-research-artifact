"""Independently verify the frozen experiment-45 pilot selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CELL_LRS = {"mousse_lr080": 0.012, "mousse_lr100": 0.015, "mousse_lr120": 0.018}
CENTER = "mousse_lr100"
MARGIN = 0.002


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_path = args.pilot_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in ("completed_valid", "completed_valid_local_wandb_incomplete"):
        raise RuntimeError("pilot manifest is not locally valid")
    if manifest.get("protocol") != "mousse_r1_three_point_pilot_v1":
        raise RuntimeError("pilot protocol mismatch")
    if manifest.get("seed") != 2026 or manifest.get("total_steps") != 1000:
        raise RuntimeError("selection requires the seed-2026 1000-step pilot")
    summaries = manifest.get("summaries", [])
    if {row.get("cell_id") for row in summaries} != set(CELL_LRS):
        raise RuntimeError("pilot does not contain the exact frozen three-cell grid")
    for row in summaries:
        if row.get("evidence_valid") is not True:
            raise RuntimeError(f"invalid pilot cell: {row.get('cell_id')}")
        if float(row.get("matrix_lr", -1)) != CELL_LRS[str(row["cell_id"])]:
            raise RuntimeError(f"pilot LR mismatch: {row}")
    ranked = sorted(summaries, key=lambda row: float(row["final_val_loss"]))
    center = next(row for row in summaries if row["cell_id"] == CENTER)
    selected = center if float(center["final_val_loss"]) <= float(ranked[0]["final_val_loss"]) + MARGIN else ranked[0]
    payload = {
        "status": "selected",
        "protocol": "mousse_r1_pilot_selection_v1",
        "seed": 2026,
        "pilot_steps": 1000,
        "selection_endpoint": "step-1000 validation loss",
        "center_tie_margin": MARGIN,
        "center_preferred_if_within_margin_of_best": True,
        "selected_cell_id": selected["cell_id"],
        "selected_matrix_lr": selected["matrix_lr"],
        "pilot_manifest": str(manifest_path),
        "pilot_manifest_sha256": sha256_file(manifest_path),
        "ranked_cells": [
            {"cell_id": row["cell_id"], "matrix_lr": row["matrix_lr"], "final_val_loss": row["final_val_loss"]}
            for row in ranked
        ],
    }
    output = (args.output or manifest_path.with_name("pilot_selection_verified.json")).resolve()
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
