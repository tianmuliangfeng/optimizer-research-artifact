#!/usr/bin/env python3
"""Experiment 57: self-contained LLaMA-1B Moonlight three-budget baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import protocol as P
import runtime as E


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE_REL = Path("scripts/57_llama1b_10b_moonlight")
CONTRACT_NAME = "ex57_contract.json"
FORMAL_SEEDS = (2024, 2025, 2026)
DEVICE_BATCH_SIZE_1B = 8
GRADIENT_ACCUMULATION_STEPS_1B = 64
FAIRNESS_PROTOCOL = "ex57_moonlight_exact_ex48_geometry_v1"
PREFLIGHT_SCHEMA = "ex57_moonlight_preflight_v1"
TUNING_SCHEMA = "ex57_moonlight_tuning_manifest_v1"
FORMAL_SCHEMA = "ex57_moonlight_formal_manifest_v1"
EXPECTED_COMPILE_CACHE_POLICY = "per_physical_gpu"
# Exact frozen source-contract hash produced by EX57 fair-parallel v2.  That
# contract already froze three independent single-GPU workers and timing as
# ineligible, but predates the execution-only per-GPU TorchInductor cache
# field added in v3.  Only this known lineage may use the compatibility path.
LEGACY_V2_SOURCE_CONTRACT_SHA256 = (
    "2e561fb977f9624fac66590074d942ed1a8484d678b0ae6bdb19febebd0f0e3b"
)
COMPILE_CACHE_COMPAT_SCHEMA = "ex57_compile_cache_contract_compatibility_v1"
COMPILE_CACHE_COMPAT_RECEIPT = "compile_cache_policy_compatibility_amendment_v1.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def absolute_without_resolving_symlinks(path: Path) -> Path:
    """Make an absolute path while preserving a virtualenv interpreter symlink.

    Resolving ``venv/bin/python`` to ``/usr/bin/python3.10`` discards the
    adjacent ``pyvenv.cfg`` context, so the child starts in the system Python
    environment.  ``abspath`` removes relative components but intentionally
    leaves the final symlink untouched.
    """
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def inherited_pythonpath_env(*prepend: Path) -> dict[str, str]:
    """Prepend project paths without discarding the frozen runtime search path."""
    env = os.environ.copy()
    paths = [str(path.resolve()) for path in prepend]
    inherited = env.get("PYTHONPATH")
    if inherited:
        paths.append(inherited)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _source_snapshot_contract_path(args: argparse.Namespace) -> Path:
    return args.run_dir / "source_snapshot" / PACKAGE_REL / CONTRACT_NAME


def _compile_cache_receipt_path(args: argparse.Namespace) -> Path:
    return args.run_dir / COMPILE_CACHE_COMPAT_RECEIPT


def _compile_cache_runtime_bindings(args: argparse.Namespace) -> dict[str, Any]:
    """Probe the live runtime's per-GPU cache binder without launching training."""
    binder = getattr(E, "bind_gpu_compile_cache", None)
    if not callable(binder):
        raise RuntimeError("EX57 runtime lacks bind_gpu_compile_cache")
    observed: dict[str, Any] = {}
    for gpu in args.gpus:
        env: dict[str, str] = {}
        binder(env, args.run_dir, int(gpu))
        expected = str(
            (args.run_dir / "_compile_cache" / f"gpu{int(gpu)}").absolute()
        )
        row = {
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
            "expected_physical_gpu": env.get("EX57_EXPECTED_PHYSICAL_GPU"),
            "torchinductor_cache_dir": env.get("TORCHINDUCTOR_CACHE_DIR"),
            "expected_cache_dir": expected,
        }
        if (
            row["expected_physical_gpu"] != str(int(gpu))
            or row["torchinductor_cache_dir"] != expected
        ):
            raise RuntimeError(
                f"EX57 per-GPU compile-cache runtime probe failed gpu{gpu}: {row}"
            )
        observed[str(int(gpu))] = row
    return observed


def _legacy_compile_cache_checks(
    args: argparse.Namespace, contract: dict[str, Any]
) -> tuple[dict[str, bool], Path, str, dict[str, Any]]:
    source_contract_path = _source_snapshot_contract_path(args)
    source_sha = (
        P.sha256_file(source_contract_path) if source_contract_path.is_file() else ""
    )
    execution = contract.get("execution", {})
    runtime_bindings: dict[str, Any] = {}
    runtime_ok = False
    try:
        runtime_bindings = _compile_cache_runtime_bindings(args)
        runtime_ok = True
    except Exception:
        runtime_ok = False
    checks = {
        "known_v2_source_contract": source_sha == LEGACY_V2_SOURCE_CONTRACT_SHA256,
        "policy_field_absent": execution.get("compile_cache_policy") is None,
        "physical_gpus": execution.get("physical_gpus") == [0, 1, 2],
        "tuning_parallel_workers": int(execution.get("tuning_parallel_workers", -1)) == 3,
        "formal_parallel_workers": int(execution.get("formal_parallel_workers", -1)) == 3,
        "single_gpu_jobs_not_ddp": execution.get("ddp") is False,
        "tuning_seed_isolated": int(contract["tuning"]["1b"]["seed"])
        not in {int(seed) for seed in contract["formal"]["seeds"]},
        "timing_ineligible": contract.get("fairness", {}).get("timing_eligible") is False
        and contract.get("formal", {}).get("timing_eligible") is False,
        "runtime_enforces_per_gpu_cache": runtime_ok,
    }
    return checks, source_contract_path, source_sha, runtime_bindings


def resolve_compile_cache_policy(
    args: argparse.Namespace, contract: dict[str, Any]
) -> dict[str, Any]:
    """Resolve v3 policy while preserving a frozen v2 source snapshot.

    The v2 EX57 scientific contract already froze GPU0/1/2 parallel jobs and
    marked timing ineligible.  v3 added only an execution-level cache-isolation
    field.  Existing v2 runs therefore receive an immutable amendment receipt;
    the source snapshot and selected formal contract are never rewritten.
    """
    declared = contract.get("execution", {}).get("compile_cache_policy")
    if declared is not None:
        if declared != EXPECTED_COMPILE_CACHE_POLICY:
            raise RuntimeError(
                f"EX57 unsupported compile-cache policy: {declared!r}"
            )
        _compile_cache_runtime_bindings(args)
        return {
            "policy": EXPECTED_COMPILE_CACHE_POLICY,
            "source": "frozen_contract",
            "receipt_path": None,
            "receipt_sha256": None,
        }

    checks, source_path, source_sha, bindings = _legacy_compile_cache_checks(
        args, contract
    )
    if not all(checks.values()):
        raise RuntimeError(
            "EX57 legacy compile-cache compatibility failed: "
            f"checks={checks} source_contract_sha256={source_sha}"
        )

    receipt_path = _compile_cache_receipt_path(args)
    payload = {
        "schema_version": COMPILE_CACHE_COMPAT_SCHEMA,
        "passed": True,
        "created_at": now_iso(),
        "source_snapshot_contract_path": str(source_path),
        "source_snapshot_contract_sha256": source_sha,
        "legacy_contract_generation": "ex57_fair_parallel_v2",
        "effective_compile_cache_policy": EXPECTED_COMPILE_CACHE_POLICY,
        "policy_scope": "execution_only",
        "runtime_bindings": bindings,
        "scientific_protocol_unchanged": True,
        "timing_eligible": False,
        "checks": checks,
        "unchanged_scientific_fields": [
            "method",
            "model",
            "data",
            "lr_grid",
            "tuning_seed_5701",
            "formal_seeds_2024_2025_2026",
            "device_batch_8",
            "accumulation_64",
            "global_batch_512",
            "token_budgets",
        ],
    }
    if receipt_path.is_file():
        existing = P.read_json(receipt_path)
        stable_keys = (
            "schema_version",
            "passed",
            "source_snapshot_contract_path",
            "source_snapshot_contract_sha256",
            "legacy_contract_generation",
            "effective_compile_cache_policy",
            "policy_scope",
            "runtime_bindings",
            "scientific_protocol_unchanged",
            "timing_eligible",
            "checks",
            "unchanged_scientific_fields",
        )
        if any(existing.get(key) != payload.get(key) for key in stable_keys):
            raise RuntimeError(
                "EX57 compile-cache compatibility receipt changed across resume"
            )
    else:
        P.atomic_json(receipt_path, payload)

    return {
        "policy": EXPECTED_COMPILE_CACHE_POLICY,
        "source": "legacy_v2_execution_amendment",
        "receipt_path": str(receipt_path),
        "receipt_sha256": P.sha256_file(receipt_path),
    }


def compile_cache_policy_replay_valid(
    args: argparse.Namespace, contract: dict[str, Any]
) -> bool:
    try:
        declared = contract.get("execution", {}).get("compile_cache_policy")
        if declared is not None:
            return (
                declared == EXPECTED_COMPILE_CACHE_POLICY
                and bool(_compile_cache_runtime_bindings(args))
            )
        checks, source_path, source_sha, bindings = _legacy_compile_cache_checks(
            args, contract
        )
        if not all(checks.values()):
            return False
        receipt_path = _compile_cache_receipt_path(args)
        if not receipt_path.is_file():
            return False
        receipt = P.read_json(receipt_path)
        return all(
            (
                receipt.get("schema_version") == COMPILE_CACHE_COMPAT_SCHEMA,
                receipt.get("passed") is True,
                receipt.get("source_snapshot_contract_path") == str(source_path),
                receipt.get("source_snapshot_contract_sha256") == source_sha,
                receipt.get("effective_compile_cache_policy")
                == EXPECTED_COMPILE_CACHE_POLICY,
                receipt.get("policy_scope") == "execution_only",
                receipt.get("runtime_bindings") == bindings,
                receipt.get("scientific_protocol_unchanged") is True,
                receipt.get("timing_eligible") is False,
                receipt.get("checks") == checks,
            )
        )
    except Exception:
        return False


def compile_cache_manifest_matches(
    args: argparse.Namespace,
    contract: dict[str, Any],
    manifest: dict[str, Any],
) -> bool:
    """Replay the policy receipt and bind manifest pointers to its exact hash."""
    if not compile_cache_policy_replay_valid(args, contract):
        return False
    declared = contract.get("execution", {}).get("compile_cache_policy")
    if declared is not None:
        return all(
            (
                manifest.get("compile_cache_policy")
                == EXPECTED_COMPILE_CACHE_POLICY,
                manifest.get("compile_cache_policy_source") == "frozen_contract",
                manifest.get("compile_cache_compatibility_receipt") is None,
                manifest.get("compile_cache_compatibility_receipt_sha256") is None,
            )
        )
    receipt_path = _compile_cache_receipt_path(args)
    return all(
        (
            manifest.get("compile_cache_policy")
            == EXPECTED_COMPILE_CACHE_POLICY,
            manifest.get("compile_cache_policy_source")
            == "legacy_v2_execution_amendment",
            manifest.get("compile_cache_compatibility_receipt")
            == str(receipt_path),
            receipt_path.is_file(),
            manifest.get("compile_cache_compatibility_receipt_sha256")
            == (P.sha256_file(receipt_path) if receipt_path.is_file() else None),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("check", "preflight", "tuning", "formal", "resume", "verify", "all"),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--official-repo", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--training-python", type=Path)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--wandb-project", default="Selective-Newton-Muon-MainConf-EX57-LLaMA1B-Moonlight-10B-20260819")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()
    if args.gpus != [0, 1, 2]:
        parser.error("EX57 is frozen to physical GPUs 0, 1, and 2")
    if args.stage != "check":
        for name in ("run_dir", "official_repo", "data_dir", "training_python"):
            if getattr(args, name) is None:
                parser.error(f"{args.stage} requires --{name.replace('_', '-')}")
    return args


def runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=args.run_dir,
        repo=args.repo,
        official_repo=args.official_repo,
        data124_dir=args.data_dir,
        data1b_dir=args.data_dir,
        training_python=args.training_python,
        gpus=args.gpus,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        device_batch_size_1b=DEVICE_BATCH_SIZE_1B,
    )


def live_contract(args: argparse.Namespace) -> dict[str, Any]:
    contract = P.read_json(args.repo / PACKAGE_REL / CONTRACT_NAME)
    P.assert_contract(contract)
    return contract


def check(args: argparse.Namespace) -> None:
    contract = live_contract(args)
    source = E.check_sources(args.repo.resolve())
    checks = {
        "contract": all(P.validate_contract(contract).values()),
        "sources": source.get("passed") is True,
        "independent_of_ex54": contract["formal"].get("independent_of_ex54") is True,
        "gpu_assignment": contract["execution"].get("physical_gpus") == [0, 1, 2],
        "ex48_geometry": contract["fairness"].get("same_microbatch_geometry_as_ex48") is True,
        "full_three_budget_graph": [row.get("budget_id") for row in P.endpoint_phases(contract)]
        == ["tokens_3p2506b", "tokens_6p9694b", "tokens_approximately_10b"],
        "tuning_formal_seed_isolation": int(contract["tuning"]["1b"]["seed"])
        not in {int(seed) for seed in contract["formal"]["seeds"]},
        "tuning_parallel_workers": int(contract["execution"].get("tuning_parallel_workers", -1)) == 3,
        "formal_parallel_workers": int(contract["execution"].get("formal_parallel_workers", -1)) == 3,
        "per_gpu_compile_cache": contract["execution"].get("compile_cache_policy") == "per_physical_gpu",
    }
    payload = {"passed": all(checks.values()), "checks": checks, "source_check": source}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise RuntimeError(f"EX57 check failed: {checks}")


def snapshot_sources(args: argparse.Namespace) -> Path:
    return E.snapshot_sources(args.run_dir, args.repo.resolve())


def init_audit_1b(
    args: argparse.Namespace, snapshot: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    rargs = runtime_args(args)
    center = next(row for row in contract["tuning"]["1b"]["cells"] if row["id"] == contract["tuning"]["1b"]["center_cell"])
    units: dict[str, Any] = {}
    for seed in contract["formal"]["seeds"]:
        command, env = E.init_command(rargs, snapshot, "1b", int(seed), center)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
        completed = subprocess.run(command, env=env, text=True, capture_output=True)
        lines = [line for line in completed.stdout.splitlines() if line.startswith("LLAMA_INIT_AUDIT ")]
        if completed.returncode or len(lines) != 1:
            raise RuntimeError(
                f"EX57 init audit failed seed{seed}: {completed.stdout}\n{completed.stderr}"
            )
        payload = json.loads(lines[0].split(" ", 1)[1])
        profile = contract["profiles"]["1b"]
        checks = {
            "init": payload.get("init_sha256") == contract["accepted_init_sha256"]["1b"][str(seed)],
            "parameters": int(payload["architecture"]["parameter_count"]) == int(profile["parameters"]),
            "matrix_tensors": int(payload["architecture"]["matrix_tensor_count"]) == int(profile["expected_matrix_tensors"]),
            "backup_tensors": int(payload["architecture"]["backup_tensor_count"]) == int(profile["expected_backup_tensors"]),
            "no_activation_k": int(payload["architecture"]["preconditioner_group_count"]) == 0,
        }
        if not all(checks.values()):
            raise RuntimeError(f"EX57 init mismatch seed{seed}: {checks}")
        units[f"1b/seed{seed}"] = {"checks": checks, "payload": payload}
    return {"passed": True, "units": units}


def preflight_valid(args: argparse.Namespace) -> bool:
    try:
        path = args.run_dir / "preflight/preflight_manifest.json"
        manifest = P.read_json(path)
        snapshot = args.run_dir / "source_snapshot"
        contract_path = snapshot / PACKAGE_REL / CONTRACT_NAME
        data_path = args.run_dir / "preflight/data_1b.json"
        if not (contract_path.is_file() and data_path.is_file()):
            return False
        contract = P.read_json(contract_path)
        P.assert_contract(contract)
        data = P.read_json(data_path)
        metadata = P.verify_data_metadata(data)
        checks = (
            manifest.get("schema_version") == PREFLIGHT_SCHEMA,
            manifest.get("passed") is True,
            manifest.get("gpus") == [0, 1, 2] == args.gpus,
            manifest.get("fairness_protocol") == FAIRNESS_PROTOCOL,
            manifest.get("paths") == {
                "official_repo": str(args.official_repo.resolve()),
                "data1b": str(args.data_dir.resolve()),
                "training_python": str(args.training_python.absolute()),
            },
            manifest.get("contract_sha256") == P.sha256_file(contract_path),
            manifest.get("data_1b_audit_sha256") == P.sha256_file(data_path),
            data.get("passed") is True,
            bool(metadata) and all(metadata.values()),
            data.get("accepted_projection_inventory_sha256")
            == contract["data"]["1b"]["accepted_projection_inventory_sha256"],
        )
        return all(checks)
    except Exception:
        return False


def preflight(args: argparse.Namespace) -> None:
    if preflight_valid(args):
        print(f"skip passed EX57 preflight: {args.run_dir}")
        return
    snapshot = snapshot_sources(args)
    contract_path = snapshot / PACKAGE_REL / CONTRACT_NAME
    contract = P.read_json(contract_path)
    P.assert_contract(contract)
    rargs = runtime_args(args)

    runtime = E.runtime_probe(rargs, snapshot, contract)
    if runtime.get("passed") is not True:
        raise RuntimeError(f"EX57 runtime preflight failed: {runtime}")

    projection_path = snapshot / PACKAGE_REL / contract["data"]["1b"]["accepted_projection_path"]
    projection_file_ok = (
        projection_path.is_file()
        and P.sha256_file(projection_path) == contract["data"]["1b"]["accepted_projection_sha256"]
    )
    projection = P.read_json(projection_path) if projection_file_ok else {}
    projection_checks = P.validate_accepted_data_projection(projection, contract)
    if not projection_file_ok or not all(projection_checks.values()):
        raise RuntimeError(
            f"EX57 accepted EX48 projection failed: file={projection_file_ok} checks={projection_checks}"
        )

    data = P.audit_data_dir(
        args.data_dir.resolve(), contract, "1b", full_hash=True, accepted_projection=projection
    )
    if data.get("passed") is not True:
        raise RuntimeError(f"EX57 data audit failed: {data.get('checks')}")
    args.run_dir.joinpath("preflight").mkdir(parents=True, exist_ok=True)
    data_path = args.run_dir / "preflight/data_1b.json"
    P.atomic_json(data_path, data)

    free = shutil.disk_usage(args.run_dir).free
    minimum = int(contract["checkpoint_retention"]["minimum_free_disk_bytes"])
    if free < minimum:
        raise RuntimeError(f"EX57 requires at least {minimum} free bytes, observed {free}")

    init = init_audit_1b(args, snapshot, contract)
    audit_code = (
        "import json; from moonlight_optimizer import run_small_matrix_reference_audit; "
        "print(json.dumps(run_small_matrix_reference_audit('cuda'),sort_keys=True))"
    )
    center = next(row for row in contract["tuning"]["1b"]["cells"] if row["id"] == contract["tuning"]["1b"]["center_cell"])
    env = E.worker_env(
        snapshot, args.official_repo, args.gpus[0], center, P.sha256_file(contract_path), "preflight_small_audit"
    )
    completed = subprocess.run(
        [str(args.training_python.absolute()), "-c", audit_code],
        env=env,
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"EX57 Moonlight small-matrix audit failed: {completed.stdout}\n{completed.stderr}"
        )
    small = json.loads(completed.stdout.strip().splitlines()[-1])
    checks = {
        "snapshot": True,
        "runtime": runtime.get("passed") is True,
        "projection": projection_file_ok and all(projection_checks.values()),
        "data": data.get("passed") is True,
        "disk": free >= minimum,
        "init": init.get("passed") is True,
        "small_matrix": small.get("passed") is True
        and small.get("state_schema", {}).get("contains_activation_k_state") is False
        and small.get("state_schema", {}).get("contains_factor_or_eigendecomposition_state") is False,
        "ex48_geometry": contract["fairness"].get("same_microbatch_geometry_as_ex48") is True,
        "independent_of_ex54": contract["formal"].get("independent_of_ex54") is True,
    }
    payload = {
        "schema_version": PREFLIGHT_SCHEMA,
        "passed": all(checks.values()),
        "created_at": now_iso(),
        "checks": checks,
        "runtime": runtime,
        "free_bytes": free,
        "init_audit": init,
        "small_matrix_audit": small,
        "contract_sha256": P.sha256_file(contract_path),
        "fairness_protocol": FAIRNESS_PROTOCOL,
        "accepted_ex48_data_projection_sha256": P.sha256_file(projection_path),
        "accepted_ex48_data_projection_inventory_sha256": contract["data"]["1b"]["accepted_projection_inventory_sha256"],
        "data_1b_audit_sha256": P.sha256_file(data_path),
        "source_snapshot_manifest_sha256": P.sha256_file(snapshot / "source_snapshot_manifest.json"),
        "paths": {
            "official_repo": str(args.official_repo.resolve()),
            "data1b": str(args.data_dir.resolve()),
            "training_python": str(args.training_python.absolute()),
        },
        "gpus": args.gpus,
    }
    P.atomic_json(args.run_dir / "preflight/preflight_manifest.json", payload)
    if not payload["passed"]:
        raise RuntimeError(f"EX57 preflight failed: {checks}")
    P.atomic_json(args.run_dir / "status.json", {"status": "preflight_passed", "updated_at": now_iso()})
    print(f"EX57 Moonlight preflight passed. Artifacts: {args.run_dir}")


def write_selected_contract(
    run_dir: Path,
    contract: dict[str, Any],
    selection_sha: str,
    selected: dict[str, Any],
) -> Path:
    payload = json.loads(json.dumps(contract))
    payload["selection_manifest_sha256"] = selection_sha
    payload["selected_configs"] = {"1b": selected["selected_cell"]}
    payload["training"]["matrix_lr"] = float(selected["selected_cell"]["matrix_lr"])
    payload["training"]["backup_lr"] = float(selected["selected_cell"]["backup_lr"])
    path = run_dir / "tuning/selected_formal_contract.json"
    if path.is_file():
        existing = P.read_json(path)
        P.assert_contract(existing)
        if (
            existing.get("selection_manifest_sha256") != selection_sha
            or existing.get("selected_configs") != payload["selected_configs"]
            or float(existing["training"]["matrix_lr"]) != float(payload["training"]["matrix_lr"])
            or float(existing["training"]["backup_lr"]) != float(payload["training"]["backup_lr"])
        ):
            raise RuntimeError("EX57 selected formal contract changed across resume")
        return path
    P.atomic_json(path, payload)
    P.assert_contract(P.read_json(path))
    return path


def tuning_valid(args: argparse.Namespace) -> bool:
    try:
        path = args.run_dir / "tuning/tuning_manifest.json"
        manifest = P.read_json(path)
        if manifest.get("schema_version") != TUNING_SCHEMA or manifest.get("passed") is not True:
            return False
        selection = Path(manifest["selection_path"])
        selected_contract = Path(manifest["selected_contract_path"])
        if not (
            selection.is_file()
            and selected_contract.is_file()
            and P.sha256_file(selection) == manifest["selection_sha256"]
            and P.sha256_file(selected_contract) == manifest["selected_contract_sha256"]
        ):
            return False
        frozen = P.read_json(selected_contract)
        P.assert_contract(frozen)
        return (
            manifest.get("formal_outcomes_observed") is False
            and manifest.get("formal_seed_overlap") is False
            and int(manifest.get("tuning_seed", -1)) == int(frozen["tuning"]["1b"]["seed"])
            and int(manifest.get("tuning_seed", -1)) not in {int(seed) for seed in frozen["formal"]["seeds"]}
            and manifest.get("tuning_gpus_used") == [0, 1, 2]
            and sorted(int(value) for value in manifest.get("gpu_assignment", {}).values()) == [0, 1, 2]
            and manifest.get("parallel_single_gpu_jobs") is True
            and compile_cache_manifest_matches(args, frozen, manifest)
            and frozen.get("selection_manifest_sha256") == manifest["selection_sha256"]
            and frozen.get("selected_configs", {}).get("1b") == manifest.get("selected_1b", {}).get("selected_cell")
        )
    except Exception:
        return False


def tuning(args: argparse.Namespace) -> None:
    if not preflight_valid(args):
        raise RuntimeError("EX57 tuning requires passed preflight")
    if tuning_valid(args):
        print(f"skip passed EX57 tuning: {args.run_dir}")
        return
    snapshot = args.run_dir / "source_snapshot"
    contract = P.read_json(snapshot / PACKAGE_REL / CONTRACT_NAME)
    P.assert_contract(contract)
    rargs = runtime_args(args)
    cells = contract["tuning"]["1b"]["cells"]
    tuning_seed = int(contract["tuning"]["1b"]["seed"])
    formal_seeds = {int(seed) for seed in contract["formal"]["seeds"]}
    if tuning_seed in formal_seeds:
        raise RuntimeError(f"EX57 fairness violation: tuning seed {tuning_seed} overlaps formal seeds")
    rows = E.schedule(
        cells,
        args.gpus,
        lambda cell, gpu: E.run_tuning_cell(rargs, snapshot, contract, "1b", cell, gpu),
    )
    tuning_gpus = sorted({int(row["physical_gpu"]) for row in rows})
    if tuning_gpus != args.gpus:
        raise RuntimeError(
            f"EX57 tuning must exercise all physical GPUs {args.gpus}; observed {tuning_gpus}"
        )
    selected = E.select_cell(contract, "1b", rows)
    selection_path = args.run_dir / "tuning/selection.json"
    selection_payload = {
        "schema_version": "ex57_moonlight_selection_v1",
        "created_at": now_iso(),
        "tuning_only": True,
        "formal_seed_overlap": False,
        "rule": contract["tuning"]["winner_rule"],
        "scale": "1b",
        "selected": selected,
    }
    if selection_path.is_file():
        existing = P.read_json(selection_path)
        stable = {key: value for key, value in selection_payload.items() if key != "created_at"}
        if any(existing.get(key) != value for key, value in stable.items()):
            raise RuntimeError("EX57 frozen selection changed across resume")
    else:
        P.atomic_json(selection_path, selection_payload)
    selection_sha = P.sha256_file(selection_path)
    selected_contract = write_selected_contract(args.run_dir, contract, selection_sha, selected)
    # Materialize/audit the local long-worker adapter now, before formal outcomes exist.
    adapter = E.runtime_long_worker_adapter(rargs, snapshot)
    compile_cache = resolve_compile_cache_policy(args, contract)
    payload = {
        "schema_version": TUNING_SCHEMA,
        "passed": True,
        "created_at": now_iso(),
        "selection_path": str(selection_path),
        "selection_sha256": selection_sha,
        "selected_contract_path": str(selected_contract),
        "selected_contract_sha256": P.sha256_file(selected_contract),
        "selected_1b": selected,
        "tuning_seed": tuning_seed,
        "formal_seeds": sorted(formal_seeds),
        "formal_seed_overlap": False,
        "tuning_gpus_used": tuning_gpus,
        "gpu_assignment": {row["cell"]["id"]: int(row["physical_gpu"]) for row in rows},
        "parallel_single_gpu_jobs": True,
        "compile_cache_policy": compile_cache["policy"],
        "compile_cache_policy_source": compile_cache["source"],
        "compile_cache_compatibility_receipt": compile_cache["receipt_path"],
        "compile_cache_compatibility_receipt_sha256": compile_cache["receipt_sha256"],
        "formal_outcomes_observed": False,
        "timing_eligible": False,
        "long_worker_adapter": str(adapter),
        "long_worker_adapter_sha256": P.sha256_file(adapter),
        "fairness_protocol": FAIRNESS_PROTOCOL,
    }
    P.atomic_json(args.run_dir / "tuning/tuning_manifest.json", payload)
    if not tuning_valid(args):
        raise RuntimeError("EX57 tuning manifest failed replay")
    P.atomic_json(args.run_dir / "status.json", {"status": "tuning_passed", "updated_at": now_iso()})
    print(f"EX57 Moonlight tuning passed. Selected: {selected['selected_cell']['id']}")


def formal_valid(args: argparse.Namespace) -> bool:
    try:
        formal = P.read_json(args.run_dir / "formal/formal_manifest.json")
        tune = P.read_json(args.run_dir / "tuning/tuning_manifest.json")
        if not (
            formal.get("schema_version") == FORMAL_SCHEMA
            and formal.get("passed") is True
            and len(formal.get("units", [])) == 3
            and formal.get("selection_sha256") == tune.get("selection_sha256")
            and formal.get("selected_contract_sha256") == tune.get("selected_contract_sha256")
            and sorted(int(row.get("seed", -1)) for row in formal.get("units", [])) == list(FORMAL_SEEDS)
            and formal.get("formal_gpus_used") == [0, 1, 2]
            and sorted(int(v) for v in formal.get("one_seed_per_gpu", {}).values()) == [0, 1, 2]
            and formal.get("parallel_single_gpu_jobs") is True
        ):
            return False
        contract = P.read_json(Path(tune["selected_contract_path"]))
        if not compile_cache_manifest_matches(args, contract, formal):
            return False
        data_sha = P.read_json(args.run_dir / "preflight/data_1b.json")["inventory_sha256"]
        return all(
            E.long_unit_valid(
                args.run_dir / "formal/1b" / f"seed{seed}",
                contract,
                tune["selection_sha256"],
                tune["selected_contract_sha256"],
                data_sha,
                seed,
            )
            for seed in FORMAL_SEEDS
        )
    except Exception:
        return False


def formal(args: argparse.Namespace) -> None:
    if not preflight_valid(args) or not tuning_valid(args):
        raise RuntimeError("EX57 formal requires passed preflight and tuning")
    if formal_valid(args):
        print(f"skip passed EX57 formal: {args.run_dir}")
        return
    snapshot = args.run_dir / "source_snapshot"
    tune = P.read_json(args.run_dir / "tuning/tuning_manifest.json")
    selected_contract = Path(tune["selected_contract_path"])
    contract = P.read_json(selected_contract)
    P.assert_contract(contract)
    selected = {"1b": tune["selected_1b"]}
    rargs = runtime_args(args)
    rows = E.schedule(
        list(FORMAL_SEEDS),
        args.gpus,
        lambda seed, gpu: E.run_formal_1b(
            rargs,
            snapshot,
            contract,
            selected,
            tune["selection_sha256"],
            selected_contract,
            int(seed),
            gpu,
        ),
    )
    formal_gpus = sorted({int(row["physical_gpu"]) for row in rows})
    if formal_gpus != args.gpus:
        raise RuntimeError(
            f"EX57 formal must exercise all physical GPUs {args.gpus}; observed {formal_gpus}"
        )
    compile_cache = resolve_compile_cache_policy(args, contract)
    payload = {
        "schema_version": FORMAL_SCHEMA,
        "passed": len(rows) == 3 and all(row.get("passed") is True for row in rows),
        "created_at": now_iso(),
        "units": sorted(rows, key=lambda row: int(row["seed"])),
        "selection_sha256": tune["selection_sha256"],
        "selected_contract_sha256": tune["selected_contract_sha256"],
        "budget_ids": contract["formal"]["accepted_1b_budget_ids"],
        "one_seed_per_gpu": {str(row["seed"]): int(row["physical_gpu"]) for row in rows},
        "formal_gpus_used": formal_gpus,
        "parallel_single_gpu_jobs": True,
        "compile_cache_policy": compile_cache["policy"],
        "compile_cache_policy_source": compile_cache["source"],
        "compile_cache_compatibility_receipt": compile_cache["receipt_path"],
        "compile_cache_compatibility_receipt_sha256": compile_cache["receipt_sha256"],
        "ddp": False,
        "timing_eligible": False,
        "fairness_protocol": FAIRNESS_PROTOCOL,
        "independent_of_ex54": True,
    }
    P.atomic_json(args.run_dir / "formal/formal_manifest.json", payload)
    if not payload["passed"] or not formal_valid(args):
        raise RuntimeError("EX57 Moonlight formal stage failed replay")
    P.atomic_json(args.run_dir / "status.json", {"status": "formal_passed", "updated_at": now_iso()})
    print(f"EX57 Moonlight formal passed. Artifacts: {args.run_dir}")


def verify(args: argparse.Namespace) -> None:
    if not formal_valid(args):
        raise RuntimeError("EX57 verify requires a completed formal stage")
    snapshot = args.run_dir / "source_snapshot"
    analyzer = snapshot / PACKAGE_REL / "analyze.py"
    build = subprocess.run(
        [str(args.training_python.absolute()), str(analyzer), "build", "--run-dir", str(args.run_dir)],
        text=True,
        capture_output=True,
        env=inherited_pythonpath_env(snapshot / PACKAGE_REL),
    )
    print(build.stdout, end="")
    if build.returncode:
        raise RuntimeError(f"EX57 analysis failed: {build.stdout}\n{build.stderr}")
    audit = subprocess.run(
        [
            str(args.training_python.absolute()), str(analyzer), "verify",
            "--run-dir", str(args.run_dir), "--full-checkpoint-hash",
        ],
        text=True,
        capture_output=True,
        env=inherited_pythonpath_env(snapshot / PACKAGE_REL),
    )
    print(audit.stdout, end="")
    if audit.returncode:
        raise RuntimeError(f"EX57 verification failed: {audit.stdout}\n{audit.stderr}")
    analysis_path = args.run_dir / "analysis/analysis_manifest.json"
    verification_path = args.run_dir / "analysis/verification_manifest.json"
    analysis = P.read_json(analysis_path)
    verification = P.read_json(verification_path)
    payload = {
        "schema_version": "ex57_moonlight_completion_v1",
        "status": "completed",
        "passed": analysis.get("passed") is True and verification.get("passed") is True,
        "formal_units": 3,
        "budget_ids": ["tokens_3p2506b", "tokens_6p9694b", "tokens_approximately_10b"],
        "analysis_manifest": str(analysis_path),
        "analysis_manifest_sha256": P.sha256_file(analysis_path),
        "verification_manifest": str(verification_path),
        "verification_manifest_sha256": P.sha256_file(verification_path),
        "formal_manifest_sha256": P.sha256_file(args.run_dir / "formal/formal_manifest.json"),
        "full_checkpoint_hash_verified": verification.get("full_checkpoint_hash") is True,
        "independent_of_ex54": True,
        "timing_usable": False,
        "wandb_required": False,
    }
    P.atomic_json(args.run_dir / "completion_manifest.json", payload)
    if not payload["passed"]:
        raise RuntimeError("EX57 completion receipt failed")
    P.atomic_json(args.run_dir / "status.json", {"status": "completed", "updated_at": now_iso()})
    print(f"EX57 Moonlight suite completed. Artifacts: {args.run_dir}")


def main() -> None:
    args = parse_args()
    if args.stage == "check":
        check(args)
        return
    args.run_dir = args.run_dir.resolve()
    args.repo = args.repo.resolve()
    args.official_repo = args.official_repo.resolve()
    args.data_dir = args.data_dir.resolve()
    # Preserve the virtualenv launcher path.  Path.resolve() follows the
    # venv/bin/python symlink to /usr/bin/python3.10 and silently switches the
    # nested probe and workers to the system torch/CUDA stack.
    args.training_python = absolute_without_resolving_symlinks(args.training_python)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "preflight":
        preflight(args)
    elif args.stage == "tuning":
        tuning(args)
    elif args.stage == "formal":
        formal(args)
    elif args.stage == "verify":
        verify(args)
    elif args.stage in ("all", "resume"):
        preflight(args)
        tuning(args)
        formal(args)
        verify(args)


if __name__ == "__main__":
    main()
