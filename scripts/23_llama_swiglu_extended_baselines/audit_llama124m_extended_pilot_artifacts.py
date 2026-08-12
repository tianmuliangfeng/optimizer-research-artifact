#!/usr/bin/env python3
"""Audit local artifacts for the frozen six-cell LLaMA-124M pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_CELLS = {
    "normuon_low": ("normuon", 0.0003, 0.005, 0.01),
    "normuon_r1scale": ("normuon", 0.0003, 0.01, 0.01),
    "normuon_official": ("normuon", 0.0003, 0.02, 0.01),
    "moonlight_official": ("moonlight_muon", 0.001, 0.001, 0.1),
    "moonlight_r1scale": ("moonlight_muon", 0.0018, 0.0018, 0.1),
    "moonlight_high": ("moonlight_muon", 0.003, 0.003, 0.1),
}
METRIC_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--wandb-long", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(left: float, right: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return math.isclose(left, right, rel_tol=0.0, abs_tol=METRIC_TOLERANCE)


def add_check(
    checks: list[dict[str, Any]],
    check: str,
    passed: bool,
    detail: str,
    severity_if_failed: str = "high",
) -> None:
    checks.append(
        {
            "check": check,
            "passed": bool(passed),
            "severity_if_failed": severity_if_failed,
            "detail": detail,
        }
    )


def expected_wandb(
    path: Path,
) -> dict[tuple[str, str, int], float]:
    output: dict[tuple[str, str, int], float] = {}
    for row in load_csv(path):
        output[(row["run_name"], row["metric"], int(row["step"]))] = float(row["value"])
    return output


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "llama_extended_manifest.json"
    manifest = load_json(manifest_path)
    checks: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    wandb = expected_wandb(args.wandb_long.resolve())

    add_check(checks, "manifest_status", manifest.get("status") == "completed", repr(manifest.get("status")))
    add_check(checks, "manifest_kind", manifest.get("kind") == "pilot", repr(manifest.get("kind")))
    add_check(
        checks,
        "manifest_protocol",
        manifest.get("protocol") == "llama_swiglu_124m_extended_progressive_v1",
        repr(manifest.get("protocol")),
    )
    add_check(checks, "manifest_seed", manifest.get("seeds") == [2026], repr(manifest.get("seeds")))
    add_check(
        checks,
        "manifest_config",
        manifest.get("config")
        == {
            "device_batch_size": 64,
            "global_batch_size": 512,
            "sequence_length": 1024,
            "steps": 1000,
            "val_every": 100,
            "val_tokens": 10_485_760,
            "warmdown_steps": 1,
        },
        json.dumps(manifest.get("config"), sort_keys=True),
    )
    add_check(
        checks,
        "manifest_no_failures",
        manifest.get("failed_tasks") == {},
        repr(manifest.get("failed_tasks")),
    )
    cells = {
        row["cell_id"]: (
            row["method"],
            row["auxiliary_lr"],
            row["matrix_lr"],
            row["weight_decay"],
        )
        for row in manifest.get("cells", [])
    }
    add_check(checks, "frozen_cell_grid", cells == EXPECTED_CELLS, repr(cells))
    add_check(
        checks,
        "completed_task_count",
        len(manifest.get("completed_tasks", [])) == 6,
        f"count={len(manifest.get('completed_tasks', []))}",
    )
    add_check(
        checks,
        "common_init",
        manifest.get("init_audit", {}).get("common_init_sha256_by_seed", {}).get("2026")
        == "a10054c8ab0a610dfe5f2cac3e21c1d44ab838df1ec4ad131e4646f6b919818d",
        repr(manifest.get("init_audit", {}).get("common_init_sha256_by_seed")),
    )
    routing = manifest.get("init_audit", {}).get("routing", {})
    add_check(
        checks,
        "parameter_routing",
        routing.get("matrix_tensor_count") == 84
        and routing.get("backup_tensor_count") == 26
        and routing.get("qkv_layout") == "separate; packed-QKV split is not applied",
        repr(routing),
    )

    task_by_cell = {row["cell_id"]: row for row in manifest.get("completed_tasks", [])}
    for cell_id, recipe in EXPECTED_CELLS.items():
        matches = list(artifact_dir.glob(f"*_{cell_id}_seed2026"))
        add_check(
            checks,
            f"{cell_id}:unique_run_dir",
            len(matches) == 1,
            f"matches={[str(path) for path in matches]}",
        )
        if len(matches) != 1:
            continue
        run_dir = matches[0]
        required = {
            "metrics.csv",
            "status.json",
            "summary.json",
            "terminal.log",
            "wandb_upload.json",
        }
        observed = {path.name for path in run_dir.iterdir() if path.is_file()}
        add_check(
            checks,
            f"{cell_id}:required_files",
            required <= observed,
            f"observed={sorted(observed)}",
        )
        summary_path = run_dir / "summary.json"
        summary = load_json(summary_path)
        task = task_by_cell.get(cell_id, {})
        add_check(
            checks,
            f"{cell_id}:summary_hash",
            sha256(summary_path) == task.get("summary_sha256"),
            f"local={sha256(summary_path)}, manifest={task.get('summary_sha256')}",
        )
        method, auxiliary_lr, matrix_lr, weight_decay = recipe
        extended = summary.get("extended_optimizer", {})
        add_check(
            checks,
            f"{cell_id}:summary_contract",
            summary.get("status") == "completed"
            and summary.get("method") == method
            and summary.get("seed") == 2026
            and summary.get("completed_steps") == 1000
            and summary.get("tokens_seen") == 524_288_000
            and summary.get("resume_count") == 0
            and summary.get("checkpoint_path") == ""
            and summary.get("init_sha256")
            == manifest["init_audit"]["common_init_sha256_by_seed"]["2026"]
            and extended.get("auxiliary_lr") == auxiliary_lr
            and extended.get("matrix_lr") == matrix_lr
            and extended.get("weight_decay") == weight_decay
            and summary.get("k_state_bytes") == 0,
            json.dumps(
                {
                    "status": summary.get("status"),
                    "method": summary.get("method"),
                    "steps": summary.get("completed_steps"),
                    "tokens": summary.get("tokens_seen"),
                    "resume": summary.get("resume_count"),
                    "checkpoint": summary.get("checkpoint_path"),
                    "extended_optimizer": extended,
                    "k_state_bytes": summary.get("k_state_bytes"),
                },
                sort_keys=True,
            ),
        )
        finite_fields = (
            "final_val_loss",
            "best_val_loss",
            "final_train_loss",
            "optimizer_state_bytes",
            "model_parameter_bytes",
            "peak_allocated_bytes",
            "train_s",
            "step_avg_ms",
        )
        add_check(
            checks,
            f"{cell_id}:finite_summary_fields",
            all(
                isinstance(summary.get(field), (int, float))
                and math.isfinite(float(summary[field]))
                and float(summary[field]) >= 0
                for field in finite_fields
            ),
            f"fields={finite_fields}",
        )

        status = load_json(run_dir / "status.json")
        add_check(
            checks,
            f"{cell_id}:status_file",
            status.get("status") == "completed"
            and status.get("completed_steps") == 1000
            and status.get("resume_count") == 0,
            repr(status),
        )
        upload = load_json(run_dir / "wandb_upload.json")
        add_check(
            checks,
            f"{cell_id}:wandb_upload",
            upload == task.get("wandb")
            and upload.get("status") == "uploaded"
            and upload.get("run_name") == f"llama124m_ext_pilot_{cell_id}_seed2026",
            repr(upload),
        )

        metrics = load_csv(run_dir / "metrics.csv")
        train_rows = [row for row in metrics if row["event"] == "train"]
        val_rows = [row for row in metrics if row["event"] == "val"]
        add_check(
            checks,
            f"{cell_id}:metric_grid",
            len(train_rows) == 1000
            and [int(row["step"]) for row in train_rows] == list(range(1, 1001))
            and [int(row["step"]) for row in val_rows] == list(range(0, 1001, 100)),
            f"train={len(train_rows)}, val_steps={[row['step'] for row in val_rows]}",
        )
        add_check(
            checks,
            f"{cell_id}:metric_validity",
            all(math.isfinite(float(row["loss"])) for row in metrics)
            and all(
                int(row["tokens_seen"]) == int(row["step"]) * 524_288
                for row in metrics
            )
            and all(float(row["lr_backup"]) == auxiliary_lr for row in metrics)
            and all(float(row["lr_matrix"]) == matrix_lr for row in metrics),
            "loss finite; token and LR traces follow the frozen contract",
        )
        run_name = f"llama124m_ext_pilot_{cell_id}_seed2026"
        mismatches = 0
        compared = 0
        local_by_event_step = {
            (row["event"], int(row["step"])): row for row in metrics
        }
        for (candidate_run, metric, step), remote_value in wandb.items():
            if candidate_run != run_name:
                continue
            event = "val" if metric == "val/loss" or step == 0 else "train"
            local = local_by_event_step.get((event, step))
            if local is None:
                mismatches += 1
                continue
            field = {
                "val/loss": "loss",
                "train/loss_step": "loss",
                "tokens/seen": "tokens_seen",
                "time/train_s": "train_s",
                "performance/step_avg_ms": "step_avg_ms",
                "lr/auxiliary": "lr_backup",
                "lr/matrix": "lr_matrix",
            }[metric]
            compared += 1
            if not close(float(local[field]), remote_value):
                mismatches += 1
        add_check(
            checks,
            f"{cell_id}:wandb_local_reconciliation",
            compared == 316 and mismatches == 0,
            f"compared={compared}, mismatches={mismatches}",
        )
        add_check(
            checks,
            f"{cell_id}:summary_metric_reconciliation",
            close(float(summary["final_val_loss"]), float(val_rows[-1]["loss"]))
            and close(float(summary["final_train_loss"]), float(train_rows[-1]["loss"]))
            and int(summary["tokens_seen"]) == int(train_rows[-1]["tokens_seen"]),
            "summary final endpoints equal local metrics.csv",
        )
        summaries.append(
            {
                "cell_id": cell_id,
                "method": method,
                "matrix_lr": matrix_lr,
                "auxiliary_lr": auxiliary_lr,
                "final_val_loss": summary["final_val_loss"],
                "final_train_loss": summary["final_train_loss"],
                "optimizer_state_bytes": summary["optimizer_state_bytes"],
                "model_parameter_bytes": summary["model_parameter_bytes"],
                "k_state_bytes": summary["k_state_bytes"],
                "peak_allocated_bytes": summary["peak_allocated_bytes"],
                "peak_allocated_mib": summary["peak_allocated_mib"],
                "train_s": summary["train_s"],
                "step_avg_ms": summary["step_avg_ms"],
                "resume_count": summary["resume_count"],
                "wandb_run_id": upload["run_id"],
            }
        )

    failures = [row for row in checks if not row["passed"]]
    write_csv(output_dir / "local_artifact_audit_checks.csv", checks)
    write_csv(output_dir / "local_artifact_run_summary.csv", summaries)
    result = {
        "status": "passed" if not failures else "failed",
        "checks_passed": len(checks) - len(failures),
        "checks_total": len(checks),
        "artifact_dir": str(artifact_dir),
        "manifest_sha256": sha256(manifest_path),
        "failures": failures,
    }
    (output_dir / "local_artifact_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
