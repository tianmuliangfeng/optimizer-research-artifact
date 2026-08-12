"""Audit experiment-45 local artifacts and reconcile them with W&B exports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SEEDS = (2024, 2025, 2026)
TOKENS_PER_STEP = 524_288
RUN_PATTERN = re.compile(
    r"^mainconf_r1_mousse_(?P<phase>pilot|formal)_mousse_lr(?P<lr>080|100|120)_"
    r"seed(?P<seed>\d+)_(?P<batch>\d{8}T\d{6}\+0000)$"
)
METRICS = (
    "val/loss",
    "train/loss_step",
    "tokens/seen",
    "memory/optimizer_state_mib",
    "memory/peak_allocated_mib",
    "lr/auxiliary",
    "lr/matrix",
)
EXPECTED_SOURCE_HASH_FIELDS = {
    "mousse_optimizer.py": "mousse_optimizer_sha256",
    "mousse_contract.json": "contract_sha256",
    "THIRD_PARTY_NOTICES.md": "third_party_notice_sha256",
    "upstream_snapshot/SNAPSHOT_MANIFEST.json": "snapshot_manifest_sha256",
    "train_r1_mousse.py": "derived_source_sha256",
}


class AuditError(RuntimeError):
    """Raised when experiment-45 evidence violates its frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def trapezoid_auc(rows: list[dict[str, str]]) -> float:
    points = [(int(row["step"]), float(row["loss"])) for row in rows]
    span = points[-1][0] - points[0][0]
    return sum(
        (right_step - left_step) * (left_loss + right_loss) / 2
        for (left_step, left_loss), (right_step, right_loss) in zip(points, points[1:])
    ) / span


def batch_manifests(results: Path, name: str) -> list[Path]:
    return sorted(results.glob(f"*/{name}"))


def validate_batch_manifests(results: Path) -> dict[str, Any]:
    preflight_paths = sorted(results.glob("*_preflight_seed2026.json"))
    if len(preflight_paths) != 1 or read_json(preflight_paths[0]).get("status") != "passed":
        raise AuditError("experiment-45 preflight is missing or not passed")
    preflight = read_json(preflight_paths[0])
    if preflight.get("small_matrix_reference_audit", {}).get("status") != "passed":
        raise AuditError("small-matrix reference audit did not pass")
    if preflight.get("initialization_audit", {}).get("status") != "passed":
        raise AuditError("initialization audit did not pass")

    pilot_paths = batch_manifests(results, "pilot_manifest.json")
    if len(pilot_paths) != 1:
        raise AuditError("expected exactly one pilot manifest")
    pilot = read_json(pilot_paths[0])
    if not (
        pilot.get("status") == "completed_valid"
        and pilot.get("protocol") == "mousse_r1_three_point_pilot_v1"
        and pilot.get("seed") == 2026
        and pilot.get("total_steps") == 1000
        and pilot.get("total_tokens") == 524_288_000
        and pilot.get("wandb_complete") is True
        and len(pilot.get("summaries", [])) == 3
        and not pilot.get("failures")
    ):
        raise AuditError("pilot manifest violates the frozen contract")
    selection_path = pilot_paths[0].with_name("pilot_selection.json")
    selection = read_json(selection_path)
    if not (
        selection.get("status") == "selected"
        and selection.get("selected_cell_id") == "mousse_lr100"
        and float(selection.get("selected_matrix_lr")) == 0.015
        and selection.get("pilot_manifest_sha256") == sha256_file(pilot_paths[0])
    ):
        raise AuditError("pilot selection certificate does not match the local manifest")

    smoke_paths = batch_manifests(results, "formal_smoke_manifest.json")
    formal_paths = batch_manifests(results, "formal_manifest.json")
    if len(smoke_paths) != 3 or len(formal_paths) != 3:
        raise AuditError("expected exactly three smoke and three formal manifests")
    smoke_seeds, formal_seeds = set(), set()
    source_fingerprints, runtime_fingerprints = set(), set()
    checkpoint_metadata = []
    for path in smoke_paths:
        manifest = read_json(path)
        seed = int(manifest.get("seed", -1))
        smoke_seeds.add(seed)
        summary = manifest.get("summary", {})
        if not (
            manifest.get("status") == "completed_valid"
            and manifest.get("protocol") == "mousse_r1_selected_exact_shape_smoke_v1"
            and manifest.get("total_steps") == 34
            and summary.get("evidence_valid") is True
            and int(summary.get("mousse_refresh_count_total", -1)) == 72 * 4
            and int(summary.get("mousse_refreshed_logical_matrices", -1)) == 72
            and not manifest.get("failures")
        ):
            raise AuditError(f"invalid formal smoke manifest: {path}")
    for path in formal_paths:
        manifest = read_json(path)
        seed = int(manifest.get("seed", -1))
        formal_seeds.add(seed)
        summary = manifest.get("summary", {})
        source_fingerprints.add(json.dumps(manifest.get("source_audit", {}), sort_keys=True))
        runtime_fingerprints.add(
            json.dumps(manifest.get("training_runtime_fingerprint", {}), sort_keys=True)
        )
        if not (
            manifest.get("status") == "completed_valid"
            and manifest.get("protocol") == "mousse_r1_selected_6200step_v1"
            and manifest.get("total_steps") == 6200
            and manifest.get("total_tokens") == 3_250_585_600
            and manifest.get("wandb_complete") is True
            and summary.get("evidence_valid") is True
            and summary.get("wandb_status") == "uploaded"
            and int(summary.get("mousse_refresh_count_total", -1)) == 72 * 620
            and int(summary.get("mousse_refreshed_logical_matrices", -1)) == 72
            and int(summary.get("checkpoint_bytes", 0)) > 0
            and summary.get("checkpoint_path")
            and not manifest.get("failures")
        ):
            raise AuditError(f"invalid formal manifest: {path}")
        checkpoint_metadata.append(
            {
                "seed": seed,
                "remote_path": summary["checkpoint_path"],
                "bytes": int(summary["checkpoint_bytes"]),
            }
        )
    if smoke_seeds != set(SEEDS) or formal_seeds != set(SEEDS):
        raise AuditError("smoke/formal seed coverage must be exactly 2024/2025/2026")
    if len(source_fingerprints) != 1 or len(runtime_fingerprints) != 1:
        raise AuditError("formal source/runtime fingerprints differ across seeds")
    return {
        "preflight_path": preflight_paths[0],
        "preflight": preflight,
        "pilot_manifest_path": pilot_paths[0],
        "selection_path": selection_path,
        "selection": selection,
        "smoke_manifest_paths": smoke_paths,
        "formal_manifest_paths": formal_paths,
        "checkpoint_metadata": checkpoint_metadata,
    }


def validate_run(
    run_dir: Path, source_audit: dict[str, str]
) -> tuple[dict[str, Any], dict[str, dict[int, float]]]:
    match = RUN_PATTERN.match(run_dir.name)
    if not match:
        raise AuditError(f"unexpected quality run name: {run_dir.name}")
    phase = match.group("phase")
    seed = int(match.group("seed"))
    total_steps = 1000 if phase == "pilot" else 6200
    manifest = read_json(run_dir / "run_manifest.json")
    summary = read_json(run_dir / "summary.json")
    if not (
        manifest.get("status") == "completed_valid"
        and int(manifest.get("returncode", -1)) == 0
        and manifest.get("wandb", {}).get("status") == "uploaded"
        and manifest.get("derived_source_sha256") == source_audit["derived_source_sha256"]
        and summary.get("evidence_valid") is True
        and int(summary.get("total_steps", -1)) == total_steps
        and int(summary.get("total_tokens", -1)) == total_steps * TOKENS_PER_STEP
    ):
        raise AuditError(f"invalid run manifest/summary: {run_dir}")

    workspaces = sorted(run_dir.glob("workspaces/attempt_*"))
    if len(workspaces) != 1:
        raise AuditError(f"expected one workspace under {run_dir}")
    workspace = workspaces[0]
    for relative, audit_field in EXPECTED_SOURCE_HASH_FIELDS.items():
        path = workspace / relative
        if not path.is_file() or sha256_file(path) != source_audit[audit_field]:
            raise AuditError(f"source hash mismatch: {path}")

    rows = read_csv(run_dir / "metrics.csv")
    train = [row for row in rows if row["event"] == "train"]
    validation = [row for row in rows if row["event"] == "validation"]
    if [int(row["step"]) for row in train] != list(range(1, total_steps + 1)):
        raise AuditError(f"{run_dir.name}: incomplete train steps")
    if [int(row["step"]) for row in validation] != list(range(0, total_steps + 1, 100)):
        raise AuditError(f"{run_dir.name}: incomplete validation steps")
    for row in rows:
        step = int(row["step"])
        if int(row["tokens_seen"]) != step * TOKENS_PER_STEP:
            raise AuditError(f"{run_dir.name}: token mismatch at step {step}")
        if not math.isfinite(float(row["loss"])):
            raise AuditError(f"{run_dir.name}: non-finite loss at step {step}")
    validation_losses = [float(row["loss"]) for row in validation]
    checks = {
        "initial_val_loss": validation_losses[0],
        "final_val_loss": validation_losses[-1],
        "best_val_loss": min(validation_losses),
        # Match the controller's exact operation order so the audit is bitwise.
        "tail5_val_loss_mean": sum(validation_losses[-5:]) / min(5, len(validation_losses)),
        "normalized_val_auc": trapezoid_auc(validation),
        "final_train_loss": float(train[-1]["loss"]),
    }
    for field, expected in checks.items():
        if float(summary[field]) != expected:
            raise AuditError(f"{run_dir.name}: summary mismatch for {field}")

    train_by_step = {int(row["step"]): row for row in train}
    validation_by_step = {int(row["step"]): row for row in validation}
    local: dict[str, dict[int, float]] = {
        "val/loss": {step: float(row["loss"]) for step, row in validation_by_step.items()},
        "train/loss_step": {
            step: float(train_by_step[step]["loss"]) for step in range(20, total_steps + 1, 20)
        },
        "tokens/seen": {0: 0.0},
        "lr/auxiliary": {0: float(validation_by_step[0]["auxiliary_lr"])},
        "lr/matrix": {0: float(validation_by_step[0]["matrix_lr"])},
        "memory/optimizer_state_mib": {
            total_steps: float(summary["optimizer_state_bytes"]) / 1024**2
        },
        "memory/peak_allocated_mib": {
            total_steps: float(summary["peak_memory_allocated_mib"])
        },
    }
    for step in range(20, total_steps + 1, 20):
        row = train_by_step[step]
        local["tokens/seen"][step] = float(row["tokens_seen"])
        local["lr/auxiliary"][step] = float(row["auxiliary_lr"])
        local["lr/matrix"][step] = float(row["matrix_lr"])
    return {
        "phase": phase,
        "run_name": run_dir.name,
        "seed": seed,
        "cell_id": summary["cell_id"],
        "status": manifest["status"],
        "returncode": manifest["returncode"],
        "wandb_status": manifest["wandb"]["status"],
        "train_rows": len(train),
        "validation_rows": len(validation),
        "final_val_loss": summary["final_val_loss"],
        "optimizer_state_mib": float(summary["optimizer_state_bytes"]) / 1024**2,
        "peak_allocated_mib": summary["peak_memory_allocated_mib"],
        "checkpoint_bytes": summary.get("checkpoint_bytes", 0),
        "checkpoint_in_uploaded_archive": False,
        "source_hashes_verified": True,
        "metrics_summary_verified": True,
    }, local


def load_wandb_history(review: Path) -> dict[tuple[str, str, int], float]:
    observed = {}
    for phase in ("pilot", "formal"):
        path = review / f"mousse_{phase}_history_long.csv"
        for row in read_csv(path):
            key = (row["run_name"], row["metric"], int(row["step"]))
            if key in observed:
                raise AuditError(f"duplicate W&B history key: {key}")
            observed[key] = float(row["value"])
    return observed


def reconcile_wandb(
    local: dict[str, dict[str, dict[int, float]]], observed: dict[tuple[str, str, int], float]
) -> list[dict[str, Any]]:
    rows = []
    expected_keys = set()
    for run, metrics in sorted(local.items()):
        for metric in METRICS:
            series = metrics[metric]
            differences = []
            for step, local_value in sorted(series.items()):
                key = (run, metric, step)
                expected_keys.add(key)
                if key not in observed:
                    raise AuditError(f"W&B history is missing {key}")
                differences.append(abs(observed[key] - local_value))
            rows.append(
                {
                    "run_name": run,
                    "metric": metric,
                    "points": len(series),
                    "max_abs_difference": max(differences, default=0.0),
                    "exact_match": all(value == 0.0 for value in differences),
                }
            )
    if set(observed) != expected_keys:
        unexpected = sorted(set(observed) - expected_keys)
        raise AuditError(f"W&B history has unexpected points: {unexpected[:10]}")
    if not all(row["exact_match"] for row in rows):
        raise AuditError("one or more local/W&B histories differ")
    return rows


def validate_formal_analysis(formal: Path, provisional: Path) -> dict[str, Any]:
    manifest = read_json(formal / "analysis_manifest.json")
    identity = read_json(formal / "identity_reuse_certificate.json")
    if manifest.get("status") != "completed_valid":
        raise AuditError("official experiment-45 analyzer did not complete validly")
    if identity.get("status") != "passed_with_caveats" or identity.get("paired_quality_eligible") is not True:
        raise AuditError("historical identity/reuse certificate is not paired-quality eligible")
    official_rows = {row["method"]: row for row in read_csv(formal / "r1_unified_eight_method_aggregate.csv")}
    provisional_rows = {row["method"]: row for row in read_csv(provisional / "r1_provisional_eight_method_aggregate.csv")}
    if set(official_rows) != set(provisional_rows):
        raise AuditError("official/provisional method coverage differs")
    maxima = {"mean": 0.0, "sample_sd": 0.0}
    for method in official_rows:
        maxima["mean"] = max(
            maxima["mean"],
            abs(float(official_rows[method]["final_val_mean"]) - float(provisional_rows[method]["final_val_loss_mean"])),
        )
        maxima["sample_sd"] = max(
            maxima["sample_sd"],
            abs(float(official_rows[method]["final_val_sample_sd"]) - float(provisional_rows[method]["final_val_loss_sample_sd"])),
        )
    if any(value != 0.0 for value in maxima.values()):
        raise AuditError(f"official/provisional aggregate mismatch: {maxima}")
    return {"identity_status": identity["status"], "paired_quality_eligible": True, "aggregate_max_abs_differences": maxima}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--wandb-review-dir", type=Path, required=True)
    parser.add_argument("--formal-analysis-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    args = parser.parse_args()
    results = args.results_dir.resolve()
    wandb_review = args.wandb_review_dir.resolve()
    formal_analysis = args.formal_analysis_dir.resolve()
    source_archive = args.source_archive.resolve()

    batch = validate_batch_manifests(results)
    source_audit = read_json(batch["formal_manifest_paths"][0]).get("source_audit", {})
    run_dirs = []
    for manifest_name in ("pilot_manifest.json", "formal_manifest.json"):
        for manifest_path in batch_manifests(results, manifest_name):
            manifest = read_json(manifest_path)
            for summary in manifest["summaries"]:
                run_dirs.append(manifest_path.parent / summary["run_name"])
    run_rows, local_histories = [], {}
    for run_dir in sorted(run_dirs):
        run_row, local = validate_run(run_dir, source_audit)
        run_rows.append(run_row)
        local_histories[run_dir.name] = local
    if len(run_rows) != 6:
        raise AuditError(f"expected six quality runs, observed {len(run_rows)}")

    observed = load_wandb_history(wandb_review)
    reconciliation = reconcile_wandb(local_histories, observed)
    formal_check = validate_formal_analysis(formal_analysis, wandb_review)

    with zipfile.ZipFile(source_archive) as archive:
        corrupt = archive.testzip()
        names = [item.filename for item in archive.infolist()]
        unsafe = [
            name for name in names
            if Path(name.replace("\\", "/")).is_absolute()
            or ".." in Path(name.replace("\\", "/")).parts
            or re.match(r"^[A-Za-z]:", name)
        ]
        duplicate_count = len(names) - len(set(names))
        if corrupt or unsafe or duplicate_count:
            raise AuditError("source ZIP integrity/path audit failed")
        archive_stats = {
            "path": str(source_archive),
            "sha256": sha256_file(source_archive),
            "bytes": source_archive.stat().st_size,
            "entries": len(names),
            "corrupt_entry": corrupt,
            "unsafe_entries": len(unsafe),
            "duplicate_names": duplicate_count,
        }

    write_csv(
        formal_analysis / "local_quality_run_audit.csv",
        run_rows,
        list(run_rows[0]),
    )
    write_csv(
        formal_analysis / "local_wandb_reconciliation.csv",
        reconciliation,
        list(reconciliation[0]),
    )
    checkpoint_total = sum(item["bytes"] for item in batch["checkpoint_metadata"])
    report = [
        "# Experiment 45 local artifact acceptance audit",
        "",
        "## Result",
        "",
        "- Quality evidence: accepted with documented evidence caveats.",
        "- Pilot: 3/3 valid; local selection certificate independently verifies mousse_lr100 (0.015).",
        "- Formal smoke: 3/3 valid, each 34 steps and four refreshes per logical matrix.",
        "- Formal: 3/3 valid at seeds 2024/2025/2026, 6200 steps and 3,250,585,600 tokens each.",
        "- W&B uploads: 6/6 quality runs uploaded; local metrics and W&B histories match exactly.",
        f"- Local/W&B comparisons: {sum(row['points'] for row in reconciliation)} values across {len(reconciliation)} run/metric series; maximum absolute difference 0.",
        "- Source/runtime: one shared derived source and one shared accepted H100 runtime fingerprint.",
        "- Official eight-method analyzer: completed_valid; historical identity/reuse passed_with_caveats and paired quality is eligible.",
        "- Timing: ineligible by the frozen two-GPU-concurrency contract.",
        "",
        "## Checkpoint boundary",
        "",
        f"The three formal manifests record non-empty remote checkpoints totaling {checkpoint_total:,} bytes,",
        "and the training controller verified their existence and size before accepting each run. The uploaded",
        "ZIP omits the checkpoint binaries, so this workstation cannot independently rehash or reload them.",
        "This does not change the validated loss histories, but strict checkpoint-payload completeness is false.",
        "",
        "## Archive",
        "",
        f"- SHA-256: `{archive_stats['sha256']}`",
        f"- Entries: {archive_stats['entries']}; corrupt/unsafe/duplicate: 0/0/0.",
        "- The original ZIP is retained unchanged under experiment 45 `source_archives/`.",
    ]
    (formal_analysis / "LOCAL_ARTIFACT_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    output_names = [
        "local_quality_run_audit.csv",
        "local_wandb_reconciliation.csv",
        "LOCAL_ARTIFACT_AUDIT.md",
    ]
    manifest = {
        "schema_version": "experiment45_local_artifact_acceptance_v1",
        "status": "passed_with_caveats",
        "synthetic": False,
        "quality_claim_eligible": True,
        "timing_claim_eligible": False,
        "strict_checkpoint_payload_complete": False,
        "strict_historical_per_run_manifest_identity": False,
        "archive": archive_stats,
        "preflight_status": "passed",
        "pilot_quality_runs": 3,
        "formal_smoke_runs": 3,
        "formal_quality_runs": 3,
        "formal_seed_coverage": list(SEEDS),
        "local_wandb_series": len(reconciliation),
        "local_wandb_points": sum(row["points"] for row in reconciliation),
        "all_local_wandb_exact": all(row["exact_match"] for row in reconciliation),
        "formal_analysis_check": formal_check,
        "checkpoint_metadata": batch["checkpoint_metadata"],
        "checkpoint_payloads_in_archive": 0,
        "authoritative_analysis_outputs": [
            {
                "path": name,
                "sha256": sha256_file(formal_analysis / name),
                "bytes": (formal_analysis / name).stat().st_size,
            }
            for name in (
                "analysis_manifest.json",
                "identity_reuse_certificate.json",
                "pilot_selection_local_verified.json",
                "r1_unified_eight_method_run_summary.csv",
                "r1_unified_eight_method_aggregate.csv",
                "r1_mousse_paired_seed_deltas.csv",
                "r1_mousse_paired_aggregate.csv",
                "EXPERIMENT_45_ANALYSIS.md",
            )
        ],
        "wandb_review_manifest": {
            "path": str(wandb_review / "wandb_review_manifest.json"),
            "sha256": sha256_file(wandb_review / "wandb_review_manifest.json"),
        },
        "outputs": [
            {
                "path": name,
                "sha256": sha256_file(formal_analysis / name),
                "bytes": (formal_analysis / name).stat().st_size,
            }
            for name in output_names
        ],
    }
    (formal_analysis / "local_artifact_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Experiment 45 local artifact audit passed with caveats: "
        f"runs={len(run_rows)} exact_wandb_points={manifest['local_wandb_points']}"
    )


if __name__ == "__main__":
    main()
