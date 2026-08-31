#!/usr/bin/env python3
"""Staged, resumable orchestration for Experiment 53."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from matched_diag_source_builder import build_matched_diag_sources


SCRIPT_VERSION = "2026-08-17.3"
ARMS = (
    "all_none",
    "c_fc_diag",
    "c_proj_diag",
    "c_fc_c_proj_diag",
    "o_proj_diag",
)
PILOT_SEED = 2053
FORMAL_SEEDS = (2024, 2025, 2026)
STAGES = ("preflight", "pilot", "formal", "verify", "all", "resume")
DATA_MAGIC = 20240520
OFFICIAL_COMMIT = "df78af0db523d8bceb25af4919a3e3e7082b80f3"
REQUIRED_TRAIN_SHARDS = tuple(
    f"fineweb_train_{index:06d}.bin" for index in range(1, 51)
)
REQUIRED_VAL_SHARD = "fineweb_val_000000.bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.gpus or len(args.gpus) != len(set(args.gpus)):
        parser.error("--gpus must be a nonempty unique list")
    if args.stage == "resume" and not args.resume:
        parser.error("--stage resume requires --resume")
    return args


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_projection(entries: list[dict[str, Any]]) -> str:
    stable = [
        {
            "name": item["name"],
            "split": item["split"],
            "index": item["index"],
            "bytes": item["bytes"],
            "magic": item["magic"],
            "sha256": item["sha256"],
        }
        for item in entries
    ]
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_data_inventory(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = (args.official_repo / "data/fineweb10B").resolve()
    required = [*REQUIRED_TRAIN_SHARDS, REQUIRED_VAL_SHARD]
    entries: list[dict[str, Any]] = []
    for name in required:
        path = data_dir / name
        if not path.is_file():
            raise RuntimeError(f"Experiment-53 required data shard is missing: {path}")
        with path.open("rb") as handle:
            raw = handle.read(4)
        if len(raw) != 4:
            raise RuntimeError(f"Experiment-53 data shard is too short: {path}")
        magic = int(struct.unpack("<i", raw)[0])
        if magic != DATA_MAGIC:
            raise RuntimeError(f"Experiment-53 data magic mismatch: {path}: {magic}")
        stat = path.stat()
        is_validation = name == REQUIRED_VAL_SHARD
        entries.append(
            {
                "name": name,
                "split": "validation" if is_validation else "train",
                "index": 0 if is_validation else int(name[-10:-4]),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "magic": magic,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema_version": 1,
        "experiment_id": "53_r1_matched_diag_module_placement",
        "passed": True,
        "data_dir": str(data_dir),
        "required_train_shards": 50,
        "required_validation_shards": 1,
        "entries": entries,
        "content_projection_sha256": _data_projection(entries),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    write_json(args.run_dir / "data_inventory.json", payload)
    return payload


def verify_data_inventory(args: argparse.Namespace, *, full_hash: bool) -> dict[str, Any]:
    path = args.run_dir / "data_inventory.json"
    if not path.is_file():
        raise RuntimeError(f"Experiment-53 data inventory is missing: {path}")
    payload = read_json(path)
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 51:
        raise RuntimeError("Experiment-53 data inventory does not contain 50 train + 1 val shards")
    if payload.get("content_projection_sha256") != _data_projection(entries):
        raise RuntimeError("Experiment-53 data inventory projection is internally inconsistent")
    expected_names = [*REQUIRED_TRAIN_SHARDS, REQUIRED_VAL_SHARD]
    if [item.get("name") for item in entries] != expected_names:
        raise RuntimeError("Experiment-53 data inventory shard order/name set drifted")
    data_dir = (args.official_repo / "data/fineweb10B").resolve()
    if Path(str(payload.get("data_dir", ""))).resolve() != data_dir:
        raise RuntimeError("Experiment-53 official data directory differs from preflight")
    for item in entries:
        shard = data_dir / str(item["name"])
        if not shard.is_file():
            raise RuntimeError(f"Experiment-53 inventoried shard vanished: {shard}")
        stat = shard.stat()
        if stat.st_size != int(item["bytes"]):
            raise RuntimeError(f"Experiment-53 shard size drift: {shard}")
        with shard.open("rb") as handle:
            raw = handle.read(4)
        if len(raw) != 4 or int(struct.unpack("<i", raw)[0]) != int(item["magic"]):
            raise RuntimeError(f"Experiment-53 shard header drift: {shard}")
        must_hash = full_hash or stat.st_mtime_ns != int(item.get("mtime_ns", -1))
        if must_hash and sha256_file(shard) != item["sha256"]:
            raise RuntimeError(f"Experiment-53 shard content drift: {shard}")
    return payload


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def append_command(run_dir: Path, payload: dict[str, Any]) -> None:
    path = run_dir / "commands.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def copy_source_tree(source: Path, target: Path) -> None:
    if target.exists():
        return
    if not source.is_dir():
        raise RuntimeError(f"missing source dependency: {source}")
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def ensure_source_snapshot(args: argparse.Namespace) -> Path:
    snapshot = args.run_dir / "source_snapshot"
    manifest_path = snapshot / "source_snapshot_manifest.json"
    if manifest_path.is_file():
        payload = read_json(manifest_path)
        if payload.get("passed") is not True:
            raise RuntimeError(f"invalid source snapshot manifest: {manifest_path}")
        for item in payload.get("files", []):
            path = snapshot / item["path"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"source snapshot drift: {path}")
        return snapshot
    scripts = args.repo / "scripts"
    for name in (
        "14_official_newton_muon_r0",
        "15_official_newton_muon_r1",
        "50_r1_global_activation_diag",
        "53_r1_matched_diag_module_placement",
        "_shared",
    ):
        copy_source_tree(scripts / name, snapshot / "scripts" / name)
    wrapper = args.repo / "commands/53_r1_matched_diag_module_placement/20260817_ex53_r1_matched_diag_module_placement.sh"
    if not wrapper.is_file():
        raise RuntimeError(f"missing Experiment-53 command wrapper: {wrapper}")
    (snapshot / "commands").mkdir(parents=True, exist_ok=True)
    shutil.copy2(wrapper, snapshot / "commands" / wrapper.name)
    files: list[dict[str, Any]] = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and path != manifest_path:
            files.append(
                {
                    "path": path.relative_to(snapshot).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "experiment_id": "53_r1_matched_diag_module_placement",
            "passed": True,
            "file_count": len(files),
            "files": files,
            "created_at": datetime.now().astimezone().isoformat(),
        },
    )
    return snapshot


def frozen_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = ensure_source_snapshot(args) / "scripts/53_r1_matched_diag_module_placement"
    return {
        "worker": root / "run_matched_diag.py",
        "analyzer": root / "analyze_matched_diag.py",
        "contract": root / "matched_diag_contract.json",
    }


def validate_batch_manifest(
    path: Path, *, pilot: bool, arm: str, seed: int
) -> dict[str, Any]:
    payload = read_json(path)
    valid_statuses = (
        {"completed_valid_smoke"}
        if pilot
        else {"completed_valid", "completed_valid_local_wandb_incomplete"}
    )
    expected_protocol = (
        "r1_matched_diag_module_placement_engineering_pilot"
        if pilot
        else "r1_matched_diag_module_placement_formal"
    )
    expected_profile = "exact_shape_numerical_smoke" if pilot else "formal"
    checks = {
        "family": payload.get("family") == "53_r1_matched_diag_module_placement",
        "protocol": payload.get("protocol") == expected_protocol,
        "batch_kind": payload.get("batch_kind") == ("smoke" if pilot else "formal"),
        "status": payload.get("status") in valid_statuses,
        "seed": payload.get("seed") == seed,
        "methods": payload.get("methods") == [arm],
        "official_commit": payload.get("official_commit") == OFFICIAL_COMMIT,
        "failures": payload.get("failures") == [],
        "formal_evidence": payload.get("formal_evidence") is (not pilot),
        "evidence_profile": payload.get("evidence_profile") == expected_profile,
        "smoke_steps": payload.get("smoke_steps") == (34 if pilot else None),
    }
    isolation = payload.get("resource_isolation")
    checks["one_process_one_gpu"] = (
        isinstance(isolation, dict)
        and isolation.get("one_process_one_gpu") is True
        and isolation.get("visible_device_count") == 1
    )
    audit = payload.get("initialization_audit")
    checks["initialization_audit"] = (
        isinstance(audit, dict)
        and audit.get("seed") == seed
        and audit.get("all_methods_identical") is True
        and isinstance(audit.get("init_sha256"), str)
        and len(str(audit.get("init_sha256"))) == 64
    )
    sources = payload.get("derived_source_sha256")
    checks["derived_source"] = (
        isinstance(sources, dict)
        and set(sources) == {arm}
        and isinstance(sources.get(arm), str)
        and len(str(sources.get(arm))) == 64
    )
    summaries = payload.get("summaries")
    checks["one_summary"] = isinstance(summaries, list) and len(summaries) == 1
    if isinstance(summaries, list) and len(summaries) == 1:
        item = summaries[0]
        checks["summary_identity"] = (
            isinstance(item, dict)
            and item.get("method") == arm
            and item.get("controlled_seed") == seed
            and item.get("evidence_valid") is True
            and item.get("formal_evidence") is (not pilot)
            and item.get("init_sha256") == audit.get("init_sha256")
            and item.get("derived_script_sha256") == sources.get(arm)
        )
        if pilot and isinstance(item, dict):
            checks["pilot_outcome_ineligible"] = (
                item.get("quality_usable") is False
                and item.get("outcome_eligible") is False
                and item.get("configuration_selection_allowed") is False
            )
        if isinstance(item, dict):
            run_name = str(item.get("run_name", ""))
            checks["run_name"] = bool(run_name) and Path(run_name).name == run_name
            run_root = path.parent / run_name
            summary_path = run_root / "r1_summary.json"
            run_manifest_path = run_root / "run_manifest.json"
            metrics_path = run_root / "r1_metrics.csv"
            checks["unit_artifacts"] = (
                summary_path.is_file()
                and run_manifest_path.is_file()
                and metrics_path.is_file()
            )
            if checks["unit_artifacts"]:
                file_summary = read_json(summary_path)
                checks["summary_file_identity"] = all(
                    file_summary.get(key) == value for key, value in item.items()
                )
                run_manifest = read_json(run_manifest_path)
                checks["run_manifest_status"] = run_manifest.get("status") == (
                    "completed_valid_smoke" if pilot else "completed_valid"
                )
                if not pilot:
                    relative = file_summary.get("checkpoint_relative_path")
                    checks["checkpoint_certificate"] = (
                        isinstance(relative, str)
                        and not Path(relative).is_absolute()
                        and isinstance(file_summary.get("checkpoint_sha256"), str)
                        and len(str(file_summary.get("checkpoint_sha256"))) == 64
                    )
                    if checks["checkpoint_certificate"]:
                        checkpoint = (run_root / str(relative)).resolve()
                        try:
                            checkpoint.relative_to(run_root.resolve())
                            inside = True
                        except ValueError:
                            inside = False
                        checks["checkpoint_fast_integrity"] = (
                            inside
                            and checkpoint.is_file()
                            and checkpoint.stat().st_size
                            == int(file_summary.get("checkpoint_bytes", -1))
                        )
    if not all(checks.values()):
        raise RuntimeError(f"invalid Experiment-53 batch manifest {path}: {checks}")
    return payload


def accepted_batch(stage_dir: Path, *, pilot: bool, arm: str, seed: int) -> Path | None:
    valid_statuses = (
        {"completed_valid_smoke"}
        if pilot
        else {"completed_valid", "completed_valid_local_wandb_incomplete"}
    )
    accepted: list[Path] = []
    for path in sorted(stage_dir.glob("**/r1_manifest.json")):
        payload = read_json(path)
        if payload.get("status") not in valid_statuses:
            continue
        if payload.get("seed") != seed or payload.get("methods") != [arm]:
            continue
        validate_batch_manifest(path, pilot=pilot, arm=arm, seed=seed)
        accepted.append(path)
    if len(accepted) > 1:
        raise RuntimeError(f"multiple accepted batches under {stage_dir}: {accepted}")
    return accepted[0] if accepted else None


def resumable_batch(stage_dir: Path) -> Path | None:
    plans = sorted(
        stage_dir.glob("**/r1_plan.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return plans[0].parent if plans else None


def worker_command(
    args: argparse.Namespace,
    *,
    arm: str,
    seed: int,
    stage: str,
    pilot_manifest: Path | None = None,
) -> list[str]:
    paths = frozen_paths(args)
    stage_dir = args.run_dir / stage / f"seed{seed}" / arm
    command = [
        sys.executable,
        str(paths["worker"]),
        "--official-repo",
        str(args.official_repo),
        "--python-exe",
        args.python_exe,
        "--seed",
        str(seed),
        "--methods",
        arm,
        "--results-dir",
        str(stage_dir),
        "--run-prefix",
        "mainconf_r1_matched_diag_placement",
        "--continue-on-error",
    ]
    previous = resumable_batch(stage_dir)
    if previous is not None:
        command.extend(["--resume-batch", str(previous)])
    if stage == "pilot":
        command.extend(
            [
                "--numerical-smoke",
                "--smoke-steps",
                "34",
                "--wandb-mode",
                "disabled",
            ]
        )
    else:
        if pilot_manifest is None:
            raise RuntimeError(f"formal {arm}/seed{seed} lacks a pilot certificate")
        command.extend(
            [
                "--smoke-manifest",
                str(pilot_manifest),
                "--wandb-mode",
                args.wandb_mode,
            ]
        )
        if args.wandb_project:
            command.extend(["--wandb-project", args.wandb_project])
        if args.wandb_entity:
            command.extend(["--wandb-entity", args.wandb_entity])
    return command


def run_jobs(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> None:
    pending = list(jobs)
    active: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    logs = args.run_dir / "controller_logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            label = str(job["label"])
            log_path = logs / f"{label.replace('/', '_')}.log"
            handle = log_path.open("a", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["NANOGPT_WORKSPACE_ROOT"] = str(args.repo.parent)
            env["SELECTIVE_NEWTON_MUON_MAIN_CONFERENCE_REPO"] = str(args.repo)
            command = list(job["command"])
            append_command(
                args.run_dir,
                {
                    "label": label,
                    "gpu": gpu,
                    "command": command,
                    "command_text": command_text(command),
                    "log": str(log_path),
                    "started_at": datetime.now().astimezone().isoformat(),
                },
            )
            print(f"START gpu={gpu} label={label}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=args.repo,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[gpu] = {
                **job,
                "process": process,
                "handle": handle,
                "log": log_path,
            }
        finished: list[str] = []
        for gpu, job in active.items():
            code = job["process"].poll()
            if code is None:
                continue
            job["handle"].close()
            print(f"END gpu={gpu} label={job['label']} return_code={code}", flush=True)
            if code != 0:
                failures.append(
                    {
                        "label": job["label"],
                        "gpu": gpu,
                        "return_code": code,
                        "log": str(job["log"]),
                    }
                )
            finished.append(gpu)
        for gpu in finished:
            del active[gpu]
        if failures:
            for job in active.values():
                job["process"].terminate()
            for job in active.values():
                try:
                    job["process"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    job["process"].kill()
                job["handle"].close()
            write_json(args.run_dir / "worker_failures.json", failures)
            raise RuntimeError(f"Experiment-53 worker failure: {failures}")
        if pending or active:
            time.sleep(2)


def require_pass(path: Path, label: str) -> None:
    if not path.is_file() or read_json(path).get("passed") is not True:
        raise RuntimeError(f"{label} has not passed: {path}")


def run_preflight(args: argparse.Namespace) -> None:
    status = args.run_dir / "preflight_manifest.json"
    if status.is_file() and read_json(status).get("passed") is True:
        verify_data_inventory(args, full_hash=True)
        print("skip passed preflight")
        return
    paths = frozen_paths(args)
    log = args.run_dir / "controller_logs/preflight.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(paths["worker"]),
        "--official-repo",
        str(args.official_repo),
        "--python-exe",
        args.python_exe,
        "--seed",
        str(PILOT_SEED),
        "--methods",
        *ARMS,
        "--preflight",
        "--wandb-mode",
        "disabled",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus[0]
    env["NANOGPT_WORKSPACE_ROOT"] = str(args.repo.parent)
    env["SELECTIVE_NEWTON_MUON_MAIN_CONFERENCE_REPO"] = str(args.repo)
    append_command(
        args.run_dir,
        {
            "label": "preflight",
            "gpu": args.gpus[0],
            "command": command,
            "command_text": command_text(command),
            "log": str(log),
        },
    )
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=args.repo,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        payload = {
            "schema_version": 1,
            "experiment_id": "53_r1_matched_diag_module_placement",
            "passed": False,
            "return_code": completed.returncode,
            "log": str(log),
        }
        write_json(status, payload)
        raise RuntimeError(f"Experiment-53 preflight failed; inspect {log}")
    data_inventory = create_data_inventory(args)
    built = build_matched_diag_sources(args.official_repo)
    derived_hashes = {item.derived_sha256 for item in built.values()}
    base_hashes = {item.base_canonical_sha256 for item in built.values()}
    if len(derived_hashes) != 1 or len(base_hashes) != 1:
        raise RuntimeError("Experiment-53 preflight did not produce one shared source lineage")
    payload = {
        "schema_version": 1,
        "experiment_id": "53_r1_matched_diag_module_placement",
        "passed": True,
        "return_code": completed.returncode,
        "pilot_seed": PILOT_SEED,
        "arms": list(ARMS),
        "gpu": args.gpus[0],
        "contract_sha256": sha256_file(paths["contract"]),
        "source_snapshot_manifest_sha256": sha256_file(
            args.run_dir / "source_snapshot/source_snapshot_manifest.json"
        ),
        "official_base_script": next(iter(built.values())).base_script,
        "official_base_canonical_sha256": next(iter(base_hashes)),
        "derived_script_sha256": next(iter(derived_hashes)),
        "data_inventory": "data_inventory.json",
        "data_inventory_sha256": sha256_file(args.run_dir / "data_inventory.json"),
        "data_content_projection_sha256": data_inventory["content_projection_sha256"],
        "log": str(log),
    }
    write_json(status, payload)


def run_pilot(args: argparse.Namespace) -> None:
    require_pass(args.run_dir / "preflight_manifest.json", "preflight")
    verify_data_inventory(args, full_hash=False)
    jobs: list[dict[str, Any]] = []
    for arm in ARMS:
        stage_dir = args.run_dir / "pilot" / f"seed{PILOT_SEED}" / arm
        if accepted_batch(stage_dir, pilot=True, arm=arm, seed=PILOT_SEED) is None:
            jobs.append(
                {
                    "label": f"pilot/{arm}",
                    "command": worker_command(
                        args, arm=arm, seed=PILOT_SEED, stage="pilot"
                    ),
                }
            )
    run_jobs(args, jobs)
    accepted = {
        arm: accepted_batch(
            args.run_dir / "pilot" / f"seed{PILOT_SEED}" / arm,
            pilot=True,
            arm=arm,
            seed=PILOT_SEED,
        )
        for arm in ARMS
    }
    if any(path is None for path in accepted.values()):
        raise RuntimeError(f"Experiment-53 pilot incomplete: {accepted}")
    write_json(
        args.run_dir / "pilot_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": "53_r1_matched_diag_module_placement",
            "passed": True,
            "seed": PILOT_SEED,
            "steps": 34,
            "outcome_eligible": False,
            "configuration_selection_allowed": False,
            "formal_seed_independent": True,
            "arms": list(ARMS),
            "accepted_batches": {
                arm: path.relative_to(args.run_dir).as_posix()
                for arm, path in accepted.items()
            },
            "data_inventory_sha256": sha256_file(args.run_dir / "data_inventory.json"),
            "source_snapshot_manifest_sha256": sha256_file(
                args.run_dir / "source_snapshot/source_snapshot_manifest.json"
            ),
        },
    )


def run_formal(args: argparse.Namespace) -> None:
    require_pass(args.run_dir / "pilot_manifest.json", "pilot")
    verify_data_inventory(args, full_hash=False)
    pilot_certificates = {
        arm: accepted_batch(
            args.run_dir / "pilot" / f"seed{PILOT_SEED}" / arm,
            pilot=True,
            arm=arm,
            seed=PILOT_SEED,
        )
        for arm in ARMS
    }
    if any(path is None for path in pilot_certificates.values()):
        raise RuntimeError(f"accepted pilot certificate vanished: {pilot_certificates}")
    jobs: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        for arm in ARMS:
            stage_dir = args.run_dir / "formal" / f"seed{seed}" / arm
            if accepted_batch(stage_dir, pilot=False, arm=arm, seed=seed) is not None:
                print(f"skip accepted formal {arm}/seed{seed}")
                continue
            jobs.append(
                {
                    "label": f"formal/seed{seed}/{arm}",
                    "command": worker_command(
                        args,
                        arm=arm,
                        seed=seed,
                        stage="formal",
                        pilot_manifest=pilot_certificates[arm],
                    ),
                }
            )
    run_jobs(args, jobs)
    accepted = {
        f"seed{seed}/{arm}": accepted_batch(
            args.run_dir / "formal" / f"seed{seed}" / arm,
            pilot=False,
            arm=arm,
            seed=seed,
        )
        for seed in FORMAL_SEEDS
        for arm in ARMS
    }
    if any(path is None for path in accepted.values()):
        raise RuntimeError(f"Experiment-53 formal incomplete: {accepted}")
    write_json(
        args.run_dir / "formal_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": "53_r1_matched_diag_module_placement",
            "passed": True,
            "pilot_seed": PILOT_SEED,
            "formal_seeds": list(FORMAL_SEEDS),
            "arms": list(ARMS),
            "formal_units": 15,
            "accepted_batches": {
                key: path.relative_to(args.run_dir).as_posix()
                for key, path in accepted.items()
            },
            "data_inventory_sha256": sha256_file(args.run_dir / "data_inventory.json"),
            "source_snapshot_manifest_sha256": sha256_file(
                args.run_dir / "source_snapshot/source_snapshot_manifest.json"
            ),
            "wandb_required_for_scientific_validity": False,
            "timing_usable": False,
        },
    )


def run_verify(args: argparse.Namespace) -> None:
    require_pass(args.run_dir / "formal_manifest.json", "formal")
    data_inventory = verify_data_inventory(args, full_hash=True)
    write_json(
        args.run_dir / "data_verify_receipt.json",
        {
            "schema_version": 1,
            "experiment_id": "53_r1_matched_diag_module_placement",
            "passed": True,
            "full_content_rehash": True,
            "data_inventory_sha256": sha256_file(args.run_dir / "data_inventory.json"),
            "data_content_projection_sha256": data_inventory[
                "content_projection_sha256"
            ],
            "verified_at": datetime.now().astimezone().isoformat(),
        },
    )
    paths = frozen_paths(args)
    output = args.run_dir / "analysis"
    manifest = output / "analysis_manifest.json"
    command = [
        sys.executable,
        str(paths["analyzer"]),
        "--run-dir",
        str(args.run_dir),
        "--contract",
        str(paths["contract"]),
        "--output-dir",
        str(output),
    ]
    append_command(
        args.run_dir,
        {"label": "verify", "command": command, "command_text": command_text(command)},
    )
    subprocess.run(command, cwd=args.repo, check=True)
    require_pass(manifest, "analysis")
    analysis = read_json(manifest)
    write_json(
        args.run_dir / "handoff_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": "53_r1_matched_diag_module_placement",
            "status": "completed",
            "passed": True,
            "scientific_result": analysis["classification"],
            "descriptive_lowest_mean_arm": analysis["descriptive_lowest_mean_arm"],
            "analysis_manifest": str(manifest),
            "analysis_manifest_sha256": sha256_file(manifest),
            "pilot_units": 5,
            "formal_units": 15,
            "timing_usable": False,
            "completed_at": datetime.now().astimezone().isoformat(),
        },
    )
    print("Experiment 53 completed.")
    print(f"Artifacts: {args.run_dir}")
    print(f"Analysis: {manifest}")
    print(f"Descriptive lowest mean arm: {analysis['descriptive_lowest_mean_arm']}")


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    args.official_repo = args.official_repo.expanduser().resolve()
    nonempty = args.run_dir.exists() and any(args.run_dir.iterdir())
    if nonempty and not args.resume:
        raise RuntimeError(
            f"run directory is nonempty; pass --resume to continue: {args.run_dir}"
        )
    if args.stage == "resume" and not nonempty:
        raise RuntimeError(f"resume requires an existing nonempty run directory: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    paths = frozen_paths(args)
    contract = read_json(paths["contract"])
    expected_gpus = contract["execution_policy"]["physical_gpus"]
    if args.gpus != expected_gpus:
        raise RuntimeError(
            f"Experiment 53 is frozen to GPUs {expected_gpus}; observed {args.gpus}"
        )
    write_json(
        args.run_dir / "suite_plan.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "experiment_id": "53_r1_matched_diag_module_placement",
            "stage_requested": args.stage,
            "run_dir": str(args.run_dir),
            "repo": str(args.repo),
            "official_repo": str(args.official_repo),
            "controller_python": sys.executable,
            "training_python": args.python_exe,
            "gpus": args.gpus,
            "pilot_seed": PILOT_SEED,
            "formal_seeds": list(FORMAL_SEEDS),
            "arms": list(ARMS),
            "pilot_units": 5,
            "formal_units": 15,
            "contract_sha256": sha256_file(paths["contract"]),
            "wandb_mode": args.wandb_mode,
            "wandb_secondary": True,
            "timing_usable": False,
        },
    )
    stages = (
        ("preflight", "pilot", "formal", "verify")
        if args.stage in {"all", "resume"}
        else (args.stage,)
    )
    dispatch = {
        "preflight": run_preflight,
        "pilot": run_pilot,
        "formal": run_formal,
        "verify": run_verify,
    }
    for stage in stages:
        dispatch[stage](args)


if __name__ == "__main__":
    main()
