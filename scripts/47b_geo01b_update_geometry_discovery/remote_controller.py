#!/usr/bin/env python3
"""Seal, run, resume, analyze, and verify GEO-01B discovery."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import protocol as P


SCRIPT_VERSION = "2026-08-04.1"
NEW_ROOT = "scripts/47b_geo01b_update_geometry_discovery"
OLD_ROOT = "scripts/47_update_geometry_curvature"
COMMAND = "commands/47b_geo01b_update_geometry_discovery/20260804_ex47b_geo01b_discovery.sh"
SOURCE_FILES = (
    COMMAND,
    f"{NEW_ROOT}/README.md",
    f"{NEW_ROOT}/geo01b_contract.json",
    f"{NEW_ROOT}/protocol.py",
    f"{NEW_ROOT}/geo01b_worker.py",
    f"{NEW_ROOT}/analyze_geo01b.py",
    f"{NEW_ROOT}/remote_controller.py",
    f"{NEW_ROOT}/run_geo01b.py",
    f"{NEW_ROOT}/test_geo01b.py",
    f"{OLD_ROOT}/protocol.py",
    f"{OLD_ROOT}/geometry_core.py",
    f"{OLD_ROOT}/geo01_worker.py",
    f"{OLD_ROOT}/remote_controller.py",
    "scripts/mdp_refresh_streaming/stream_metrics.py",
    "scripts/mdp_refresh_streaming/pinned_ex37_runtime/triton_kernels.py",
    "scripts/46_mdp05_confirmatory_update_shock/smoke_worker.py",
    "scripts/37_mech09_downproj_refresh_mediation/mech09r_worker.py",
    "scripts/37_mech09_downproj_refresh_mediation/mech09_worker.py",
    "scripts/37_mech09_downproj_refresh_mediation/refresh_mediation_repair_contract.json",
    "scripts/37_mech09_downproj_refresh_mediation/mech08_control_reference.json",
    "scripts/36_mech08_short_horizon_rollout/mech08_worker.py",
    "scripts/27_mech01_unified_k_diagnostics/mech01_worker.py",
)


def _load_base() -> Any:
    path = HERE.parent / "47_update_geometry_curvature" / "remote_controller.py"
    spec = importlib.util.spec_from_file_location("geo01a_remote_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import accepted GEO-01A controller: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
sha256_file = P.sha256_file
absolute_without_resolving = BASE.absolute_without_resolving
source_templates = BASE.source_templates
base_arguments = BASE.base_arguments
runtime_preflight = BASE.runtime_preflight
next_attempt = BASE.next_attempt
selected_attempt = BASE.selected_attempt
run_job = BASE.run_job


def live_source_hashes() -> dict[str, str]:
    result = {}
    for relative in SOURCE_FILES:
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(f"required GEO-01B source is missing: {path}")
        result[relative] = sha256_file(path)
    return result


def snapshot_sources(run_dir: Path) -> tuple[Path, Path]:
    snapshot = run_dir / "source_snapshot"
    manifest_path = snapshot / "source_snapshot_manifest.json"
    live = live_source_hashes()
    if snapshot.exists():
        manifest = P.read_json(manifest_path)
        checks = {
            relative: (snapshot / relative).is_file()
            and sha256_file(snapshot / relative) == digest
            for relative, digest in manifest.get("files", {}).items()
        }
        if manifest.get("files") != live or not all(checks.values()):
            raise RuntimeError("sealed GEO-01B sources differ from live sources")
        return snapshot, manifest_path
    snapshot.mkdir(parents=True, exist_ok=False)
    for relative in SOURCE_FILES:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, target)
    P.atomic_json(
        manifest_path,
        {
            "schema_version": "geo01b_source_snapshot_v1",
            "controller_version": SCRIPT_VERSION,
            "files": {
                relative: sha256_file(snapshot / relative)
                for relative in SOURCE_FILES
            },
            "passed": True,
        },
    )
    return snapshot, manifest_path


def source_preflight(
    source_run: Path,
    templates: dict[str, list[str]],
    contract_path: Path,
    base_contract_path: Path,
) -> dict[str, Any]:
    contract = P.read_json(contract_path)
    source_manifest = P.read_json(source_run / "formal" / "formal_manifest.json")
    source_paths = []
    for arguments in templates.values():
        for option in (
            "--checkpoint",
            "--checkpoint-hash-certificate",
            "--source-script",
            "--profile-script",
        ):
            source_paths.append(Path(BASE.option_value(arguments, option)))
    checks = {
        "contract": all(P.validate_contract(contract).values()),
        "source_contract_hash": sha256_file(base_contract_path)
        == contract["source_lineage"].get(
            "public_execution_contract_sha256",
            contract["source_lineage"]["accepted_execution_contract_sha256"],
        ),
        "source_manifest": source_manifest.get("passed") is True
        and int(source_manifest.get("completed_jobs", -1)) == 12,
        "source_commands": (source_run / "commands.jsonl").is_file(),
        "source_paths": all(path.is_file() for path in source_paths),
        "outcome_blind_formula": contract["source_lineage"][
            "mdp05_outcomes_used_to_select_formula"
        ]
        is False,
        "accepted_pilot": contract["geo01a_pilot_lineage"][
            "engineering_integrity_passed"
        ]
        is True,
    }
    return {"checks": checks, "passed": all(checks.values())}


def select(unit_dir: Path, attempt: Path, result: dict[str, Any]) -> None:
    if result["passed"]:
        P.atomic_json(
            unit_dir / "unit_selection.json",
            {
                "schema_version": "geo01b_unit_selection_v1",
                "selected_attempt": attempt.name,
                "result": result,
                "passed": True,
            },
        )


def write_combined(
    run_dir: Path, attempts: list[Path], filename: str, output_name: str
) -> Path:
    output = run_dir / "combined" / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for attempt in attempts:
            source = attempt / filename
            if not source.is_file():
                raise FileNotFoundError(f"combined input is missing: {source}")
            text = source.read_text(encoding="utf-8")
            handle.write(text)
            if text and not text.endswith("\n"):
                handle.write("\n")
    os.replace(temporary, output)
    return output


def create_handoff(
    run_dir: Path, smoke_attempt: Path, attempts: list[Path]
) -> dict[str, Any]:
    paths = [
        run_dir / "run_identity.json",
        run_dir / "preflight.json",
        run_dir / "runtime_preflight.json",
        run_dir / "source_snapshot" / "source_snapshot_manifest.json",
        run_dir / "sealed" / "derived_execution_contract.json",
        run_dir / "sealed" / "offset_collision_certificate.json",
        run_dir / "sealed" / "discovery_plan.json",
        smoke_attempt / "mech09r_manifest.json",
        smoke_attempt / "worker_log_seal.json",
        run_dir / "combined" / "geometry_rows.jsonl",
        run_dir / "combined" / "outcome_rows.jsonl",
        run_dir / "analysis" / "analysis_manifest.json",
        run_dir / "analysis" / "event_summary.csv",
    ]
    for attempt in attempts:
        paths.extend(
            [
                attempt / "geo01b_unit_manifest.json",
                attempt / "geo01b_status.json",
                attempt / "worker_log_seal.json",
                attempt / "geo01b_geometry_rows.jsonl",
                attempt / "geo01b_outcome_rows.jsonl",
            ]
        )
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"handoff artifact is missing: {path}")
        rows.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema_version": "geo01b_handoff_manifest_v1",
        "files": rows,
        "unit_count": len(attempts),
        "no_large_files": all(row["bytes"] < 250_000_000 for row in rows),
        "discovery_claim_eligible": False,
        "confirmation_authorized": False,
    }
    payload["passed"] = (
        payload["unit_count"] == 12 and payload["no_large_files"]
    )
    P.atomic_json(run_dir / "handoff_manifest.json", payload)
    return payload


def audit_handoff(run_dir: Path, handoff: dict[str, Any]) -> dict[str, bool]:
    rows = handoff.get("files", [])
    paths = [str(row.get("path", "")) for row in rows]
    files = []
    for row in rows:
        path = run_dir / str(row.get("path", ""))
        files.append(
            path.is_file()
            and path.stat().st_size == int(row.get("bytes", -1))
            and sha256_file(path) == row.get("sha256")
        )
    return {
        "schema": handoff.get("schema_version") == "geo01b_handoff_manifest_v1",
        "passed": handoff.get("passed") is True,
        "unit_count": int(handoff.get("unit_count", -1)) == 12,
        "claim_boundary": handoff.get("discovery_claim_eligible") is False
        and handoff.get("confirmation_authorized") is False,
        "unique_paths": bool(paths) and len(paths) == len(set(paths)),
        "files": all(files),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("dry-run", "discovery", "resume", "verify")
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--child-python", type=Path)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    return parser.parse_args()


def controller(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    live_contract = HERE / "geo01b_contract.json"
    contract_sha = sha256_file(live_contract)
    if args.mode == "verify":
        identity = P.read_json(run_dir / "run_identity.json")
        status = P.read_json(run_dir / "status.json")
        analysis = P.read_json(run_dir / "analysis" / "analysis_manifest.json")
        handoff = P.read_json(run_dir / "handoff_manifest.json")
        handoff_checks = audit_handoff(run_dir, handoff)
        checks = {
            "identity": identity.get("experiment") == "GEO-01B",
            "contract": identity.get("contract_sha256") == contract_sha,
            "completed": status.get("status") == "completed",
            "analysis_integrity": analysis.get("integrity_passed") is True,
            "handoff": all(handoff_checks.values()),
            "not_claim_eligible": status.get("claim_eligible") is False,
            "confirmation_not_authorized": status.get(
                "confirmation_authorized"
            )
            is False,
        }
        print(json.dumps({"checks": checks, "passed": all(checks.values())}, indent=2))
        return 0 if all(checks.values()) else 2
    if args.source_run is None or args.child_python is None:
        raise ValueError(f"{args.mode} requires --source-run and --child-python")
    source_run = args.source_run.resolve()
    child_python = absolute_without_resolving(args.child_python)
    resume = args.mode == "resume"
    if resume:
        identity = P.read_json(run_dir / "run_identity.json")
        checks = {
            "experiment": identity.get("experiment") == "GEO-01B",
            "contract": identity.get("contract_sha256") == contract_sha,
            "not_dry_run": identity.get("dry_run") is False,
        }
        if not all(checks.values()):
            raise RuntimeError(f"resume identity mismatch: {checks}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        P.atomic_json(
            run_dir / "run_identity.json",
            {
                "schema_version": "geo01b_run_identity_v1",
                "experiment": "GEO-01B",
                "experiment_number": 47,
                "phase": "discovery",
                "contract_sha256": contract_sha,
                "controller_version": SCRIPT_VERSION,
                "dry_run": args.mode == "dry-run",
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    P.atomic_json(run_dir / "status.json", {"status": "preflight"})
    snapshot, snapshot_manifest = snapshot_sources(run_dir)
    contract_path = snapshot / NEW_ROOT / "geo01b_contract.json"
    base_contract = snapshot / "scripts/37_mech09_downproj_refresh_mediation/refresh_mediation_repair_contract.json"
    contract = P.read_json(contract_path)
    templates = source_templates(source_run)
    preflight = source_preflight(
        source_run, templates, contract_path, base_contract
    )
    preflight["checks"].update(
        {
            "child_python": child_python.is_file(),
            "two_unique_gpus": len(args.gpus) == 2
            and len(set(args.gpus)) == 2,
        }
    )
    preflight["passed"] = all(preflight["checks"].values())
    P.atomic_json(run_dir / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError(f"GEO-01B source preflight failed: {preflight['checks']}")

    sealed = run_dir / "sealed"
    sealed.mkdir(exist_ok=resume)
    execution_path = sealed / "derived_execution_contract.json"
    derived = P.derive_execution_contract(
        P.read_json(base_contract), contract, contract_sha
    )
    if execution_path.is_file() and P.read_json(execution_path) != derived:
        raise RuntimeError("sealed GEO-01B execution contract changed")
    P.atomic_json(execution_path, derived)
    execution_sha = sha256_file(execution_path)
    offsets = P.build_offset_certificate(contract)
    P.atomic_json(sealed / "offset_collision_certificate.json", offsets)
    if not offsets["passed"]:
        raise RuntimeError(f"GEO-01B offset collision: {offsets['checks']}")
    jobs = P.job_matrix(contract)
    plan = {
        "schema_version": "geo01b_discovery_plan_v1",
        "contract_sha256": contract_sha,
        "execution_contract_sha256": execution_sha,
        "source_snapshot_manifest_sha256": sha256_file(snapshot_manifest),
        "smoke": contract["remote_smoke"],
        "jobs": jobs,
        "unit_count": len(jobs),
        "events_per_unit": 2,
        "all_down_layers": 18,
        "maximum_parallel_jobs": 2,
        "outcome_blind_until_contract_sealed": True,
        "confirmation_authorized": False,
        "claim_eligible": False,
        "passed": len(jobs) == 12,
    }
    plan_path = sealed / "discovery_plan.json"
    if plan_path.is_file() and P.read_json(plan_path) != plan:
        raise RuntimeError("sealed GEO-01B discovery plan changed")
    P.atomic_json(plan_path, plan)
    if args.mode == "dry-run":
        P.atomic_json(
            run_dir / "status.json",
            {
                "status": "dry_run_passed",
                "phase": "discovery",
                "claim_eligible": False,
                "confirmation_authorized": False,
            },
        )
        print(f"GEO-01B sealed dry-run passed: {run_dir}")
        return 0

    runtime = runtime_preflight(
        child_python, args.gpus, contract["runtime_contract"]
    )
    P.atomic_json(run_dir / "runtime_preflight.json", runtime)
    if not runtime.get("passed"):
        raise RuntimeError(f"GEO-01B runtime preflight failed: {runtime}")

    smoke = contract["remote_smoke"]
    smoke_dir = run_dir / "smoke" / smoke["origin"] / f"replica_{smoke['data_replica']}"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_attempt = selected_attempt(
        smoke_dir,
        "mech09r_manifest.json",
        {
            "checkpoint_cell": smoke["origin"],
            "data_replica": int(smoke["data_replica"]),
            "contract_sha256": execution_sha,
        },
    )
    if smoke_attempt is None:
        smoke_attempt = next_attempt(smoke_dir)
        worker_args = base_arguments(
            templates[smoke["origin"]],
            output=smoke_attempt,
            tier="smoke",
            origin=smoke["origin"],
            replica=int(smoke["data_replica"]),
            execution_contract=execution_path,
            snapshot=snapshot,
            smoke_manifest=None,
        )
        result = run_job(
            label=f"smoke/{smoke['origin']}/replica_{smoke['data_replica']}",
            command=[
                os.path.abspath(os.fspath(child_python)),
                str(snapshot / "scripts/46_mdp05_confirmatory_update_shock/smoke_worker.py"),
                *worker_args,
            ],
            attempt=smoke_attempt,
            gpu=args.gpus[0],
            manifest_name="mech09r_manifest.json",
            status_name="status.json",
            expected={
                "origin": ("checkpoint_cell", smoke["origin"]),
                "replica": ("data_replica", int(smoke["data_replica"])),
                "contract": ("contract_sha256", execution_sha),
            },
        )
        select(smoke_dir, smoke_attempt, result)
        if not result["passed"]:
            raise RuntimeError(f"GEO-01B smoke failed; see {smoke_attempt / 'worker.log'}")
    smoke_manifest = smoke_attempt / "mech09r_manifest.json"

    def run_unit(job: dict[str, Any], gpu: str) -> tuple[dict[str, Any], Path]:
        origin = str(job["origin"])
        replica = int(job["data_replica"])
        unit_dir = run_dir / "discovery" / origin / f"replica_{replica}"
        unit_dir.mkdir(parents=True, exist_ok=True)
        expected = {
            "origin": origin,
            "data_replica": replica,
            "contract_sha256": contract_sha,
            "execution_contract_sha256": execution_sha,
            "phase": "discovery",
            "claim_eligible": False,
            "full_direction_persisted": False,
            "full_hessian_constructed": False,
        }
        selected = selected_attempt(
            unit_dir, "geo01b_unit_manifest.json", expected
        )
        if selected is not None:
            print(f"skip passed unit: {job['label']}", flush=True)
            return {"label": job["label"], "passed": True, "skipped": True}, selected
        attempt = next_attempt(unit_dir)
        accepted_args = base_arguments(
            templates[origin],
            output=attempt,
            tier="formal",
            origin=origin,
            replica=replica,
            execution_contract=execution_path,
            snapshot=snapshot,
            smoke_manifest=smoke_manifest,
        )
        command = [
            os.path.abspath(os.fspath(child_python)),
            str(snapshot / NEW_ROOT / "geo01b_worker.py"),
            "--geo01b-output-dir",
            str(attempt),
            "--geo01b-contract",
            str(contract_path),
            "--source-snapshot-manifest",
            str(snapshot_manifest),
            "--",
            *accepted_args,
        ]
        result = run_job(
            label=str(job["label"]),
            command=command,
            attempt=attempt,
            gpu=gpu,
            manifest_name="geo01b_unit_manifest.json",
            status_name="geo01b_status.json",
            expected={key: (key, value) for key, value in expected.items()},
        )
        select(unit_dir, attempt, result)
        return result, attempt

    lanes = [jobs[0::2], jobs[1::2]]

    def run_lane(lane: list[dict[str, Any]], gpu: str) -> list[tuple[dict[str, Any], Path]]:
        results = []
        for job in lane:
            result = run_unit(job, gpu)
            results.append(result)
            if not result[0]["passed"]:
                break
        return results

    P.atomic_json(
        run_dir / "status.json",
        {
            "status": "discovery_running",
            "unit_count": 12,
            "claim_eligible": False,
            "confirmation_authorized": False,
        },
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run_lane, lanes[index], args.gpus[index])
            for index in range(2)
        ]
        lane_results = [future.result() for future in futures]
    completed = [item for lane in lane_results for item in lane]
    failures = [result for result, _ in completed if not result["passed"]]
    if failures or len(completed) != len(jobs):
        raise RuntimeError(
            f"GEO-01B discovery incomplete: completed={len(completed)}/12 failures={failures}"
        )
    by_label = {result["label"]: attempt for result, attempt in completed}
    attempts = [by_label[job["label"]] for job in jobs]
    geometry = write_combined(
        run_dir, attempts, "geo01b_geometry_rows.jsonl", "geometry_rows.jsonl"
    )
    outcomes = write_combined(
        run_dir, attempts, "geo01b_outcome_rows.jsonl", "outcome_rows.jsonl"
    )
    analysis_dir = run_dir / "analysis"
    completed_analysis = subprocess.run(
        [
            os.path.abspath(sys.executable),
            str(snapshot / NEW_ROOT / "analyze_geo01b.py"),
            "--geometry-jsonl",
            str(geometry),
            "--outcomes-jsonl",
            str(outcomes),
            "--output-dir",
            str(analysis_dir),
            "--contract",
            str(contract_path),
        ],
        check=False,
        cwd=str(run_dir),
    )
    if completed_analysis.returncode != 0:
        raise RuntimeError(f"GEO-01B analyzer failed rc={completed_analysis.returncode}")
    analysis = P.read_json(analysis_dir / "analysis_manifest.json")
    if analysis.get("integrity_passed") is not True:
        raise RuntimeError("GEO-01B analysis integrity failed")
    handoff = create_handoff(run_dir, smoke_attempt, attempts)
    if not handoff["passed"]:
        raise RuntimeError("GEO-01B handoff failed")
    P.atomic_json(
        run_dir / "status.json",
        {
            "status": "completed",
            "phase": "discovery",
            "scientific_result": analysis["scientific_result"],
            "curvature_increment_result": analysis["curvature_increment_result"],
            "confirmation_candidate": analysis["confirmation_candidate"],
            "confirmation_authorized": False,
            "claim_eligible": False,
            "llama_10b_triggered": False,
        },
    )
    print("GEO-01B discovery completed.")
    print(f"Artifacts: {run_dir}")
    print(f"Analysis: {analysis_dir / 'analysis_manifest.json'}")
    print(f"Scientific result: {analysis['scientific_result']}")
    print("Confirmation remains unauthorized pending review and a new contract.")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return controller(args)
    except BaseException as exc:
        run_dir = args.run_dir.resolve()
        if run_dir.is_dir():
            P.atomic_json(
                run_dir / "status.json",
                {
                    "status": "failed_or_incomplete",
                    "controller_version": SCRIPT_VERSION,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "resume_allowed_same_contract": args.mode != "dry-run",
                    "claim_eligible": False,
                    "confirmation_authorized": False,
                },
            )
        print(
            f"GEO-01B stopped cleanly: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
