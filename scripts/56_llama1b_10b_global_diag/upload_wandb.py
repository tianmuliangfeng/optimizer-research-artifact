#!/usr/bin/env python3
"""Post-endpoint W&B uploader; local CSV/JSON remains the primary evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import protocol as P


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--unit-dir", type=Path, required=True)
    parser.add_argument("--endpoint-phase", required=True)
    parser.add_argument("--mode", choices=("online", "offline"), required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity")
    parser.add_argument("--init-timeout", type=int, default=120)
    return parser.parse_args()


def phase_chain(endpoint: str) -> list[str]:
    return {
        "cooldown_6200": ["backbone_4400", "cooldown_6200"],
        "cooldown_13293": ["backbone_4400", "backbone_11493", "cooldown_13293"],
        "cooldown_19073": [
            "backbone_4400",
            "backbone_11493",
            "backbone_17273",
            "cooldown_19073",
        ],
    }[endpoint]


def main() -> int:
    args = parse_args()
    contract_path = args.contract.resolve()
    contract = P.read_json(contract_path)
    P.assert_contract(contract)
    phase = P.phase_map(contract)[args.endpoint_phase]
    if phase["role"] != "primary_endpoint":
        raise RuntimeError("W&B upload requires a primary endpoint phase")
    unit_dir = args.unit_dir.resolve()
    method = unit_dir.parent.name
    seed = int(unit_dir.name.removeprefix("seed"))
    if method not in contract["grid"]["methods"] or seed not in contract["grid"]["seeds"]:
        raise RuntimeError("unit identity is outside the frozen formal grid")
    endpoint_dir = unit_dir / args.endpoint_phase
    summary = P.read_json(endpoint_dir / "summary.json")
    inputs = {
        phase_id: P.sha256_file(unit_dir / phase_id / "metrics.csv")
        for phase_id in phase_chain(args.endpoint_phase)
    }
    inputs["summary"] = P.sha256_file(endpoint_dir / "summary.json")
    output = endpoint_dir / "wandb_upload.json"
    if output.is_file():
        prior = P.read_json(output)
        if prior.get("status") == "uploaded" and prior.get("inputs") == inputs:
            print(json.dumps(prior, indent=2, sort_keys=True))
            return 0

    merged: dict[tuple[int, str], dict[str, str]] = {}
    for phase_id in phase_chain(args.endpoint_phase):
        for row in P.read_metrics(unit_dir / phase_id / "metrics.csv"):
            merged[(int(row["step"]), row["event"])] = row
    identity = {
        "experiment": 56,
        "family": contract["family"],
        "contract_sha256": P.sha256_file(contract_path),
        "budget_id": phase["budget_id"],
        "endpoint_step": phase["target_step"],
        "method": method,
        "seed": seed,
        "architecture": contract["profile"]["name"],
        "parameters": contract["profile"]["parameters"],
        "tokens_per_parameter": phase["tokens_per_parameter"],
        "lr_policy": contract["lr_policy"]["name"],
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    run_name = f"ex56_{phase['budget_id']}_{method}_seed{seed}"
    try:
        import wandb

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            id=run_id,
            resume="allow",
            name=run_name,
            group=f"ex56_{phase['budget_id']}",
            mode=args.mode,
            dir=str(endpoint_dir),
            config=identity
            | {
                "wandb_upload_timing": contract["wandb"]["upload_timing"],
                "local_metrics_primary": True,
                "init_sha256": summary["init_sha256"],
                "resume_count_endpoint_segment": summary["resume_count"],
            },
            tags=["ex56", "llama1b", "fineweb10B", "global-diag", "long-token", "formal"],
            settings=wandb.Settings(init_timeout=args.init_timeout),
        )
        try:
            per_step: dict[int, dict[str, float]] = {}
            final_train = max(step for step, event in merged if event == "train")
            for (step, event), row in sorted(merged.items()):
                if event == "train" and step % int(contract["wandb"]["train_log_every_steps"]) != 0 and step != final_train:
                    continue
                values = per_step.setdefault(step, {})
                if event == "train":
                    values["train/loss_step"] = float(row["loss"])
                else:
                    values["val/loss"] = float(row["loss"])
                values.update(
                    {
                        "lr/backup": float(row["lr_backup"]),
                        "lr/matrix": float(row["lr_matrix"]),
                        "tokens/seen": float(row["tokens_seen"]),
                        "tokens/per_parameter": float(row["tokens_per_parameter"]),
                    }
                )
            for step in sorted(per_step):
                wandb.log(per_step[step], step=step)
            run.summary.update(
                {
                    "final_val_loss": summary["final_val_loss"],
                    "final_train_loss": summary["final_train_loss"],
                    "endpoint_tokens": summary["tokens_seen"],
                    "tokens_per_parameter": summary["tokens_per_parameter"],
                    "peak_allocated_bytes_endpoint_segment": summary["peak_allocated_bytes"],
                    "k_state_bytes": summary["k_state_bytes"],
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                    "local_metrics_sha256": summary["metrics_sha256"],
                    "timing_claim_eligible": False,
                }
            )
        finally:
            run.finish()
        payload: dict[str, Any] = {
            "schema_version": "ex56_wandb_upload_v1",
            "status": "uploaded",
            "mode": args.mode,
            "project": args.project,
            "entity": args.entity,
            "run_id": run_id,
            "run_name": run_name,
            "inputs": inputs,
        }
        P.atomic_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        payload = {
            "schema_version": "ex56_wandb_upload_v1",
            "status": "pending_after_failure",
            "error": f"{type(exc).__name__}: {exc}",
            "project": args.project,
            "run_id": run_id,
            "run_name": run_name,
            "inputs": inputs,
        }
        P.atomic_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
