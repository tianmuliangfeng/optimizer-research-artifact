#!/usr/bin/env python3
"""Seal, tune, run, resume, analyze, and verify Experiment 57."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Barrier
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable

import protocol as P
import source_builder


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parents[1]
PACKAGE_REL = Path("scripts/57_llama1b_10b_moonlight")
CONTRACT_NAME = "ex57_contract.json"
PARENT_FILES = tuple(source_builder.PARENT_SHA256)
LONG_BUDGET_NVIDIA_DRIVER = "580.95.05"
ACCEPTED_EX48_RUNTIME_PREFLIGHT_SHA256 = (
    "dba54302be07d4e336c3b2f58b59e0cf4eb62e7505414fac7b38a66de8f3b092"
)
NON10B_PHASE_IDS = (
    "backbone_4400", "cooldown_6200", "backbone_11493", "cooldown_13293",
)
TEN_B_PHASE_IDS = ("backbone_17273", "cooldown_19073")
DEVICE_BATCH_SIZE_1B = 8
GRADIENT_ACCUMULATION_STEPS_1B = 64
MEMORY_SAFE_PROTOCOL = "ex57_moonlight_ex48_geometry_v1"
MEMORY_CANARY_SEED = 5403
MEMORY_CANARY_STEPS = 0
PREFLIGHT_SCHEMA = "ex57_moonlight_preflight_v1"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "check", "preflight", "tuning", "formal_non10b",
            "formal", "verify", "upload", "all", "resume",
        ),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--repo", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--official-repo", type=Path)
    parser.add_argument("--data124-dir", type=Path)
    parser.add_argument("--data1b-dir", type=Path)
    parser.add_argument("--training-python", type=Path)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--wandb-project", default="Selective-Newton-Muon-MainConf-EX57-LLaMA-Moonlight-20260819")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()
    if args.stage != "check":
        for name in ("run_dir", "official_repo", "data124_dir", "data1b_dir", "training_python"):
            if getattr(args, name) is None:
                parser.error(f"{args.stage} requires --{name.replace('_', '-')}")
    if args.gpus != [0, 1, 2]:
        parser.error("Experiment 57 requires physical GPUs 0, 1, and 2")
    args.device_batch_size_1b = DEVICE_BATCH_SIZE_1B
    return args


def package_contract(root: Path) -> Path:
    return root / PACKAGE_REL / CONTRACT_NAME


def package_files(root: Path) -> list[Path]:
    package = root / PACKAGE_REL
    files = [
        path for path in package.iterdir()
        if path.is_file() and path.suffix in {".py", ".json", ".csv", ".md"}
    ]
    command = root / "commands/57_llama1b_10b_moonlight/20260819_ex57_llama1b_10b_moonlight.sh"
    if command.is_file():
        files.append(command)
    return sorted(files)


def check_sources(root: Path) -> dict[str, Any]:
    contract = P.read_json(package_contract(root))
    P.assert_contract(contract)
    checks: dict[str, bool] = {}
    rows: dict[str, Any] = {}
    for relative, expected in source_builder.PARENT_SHA256.items():
        path = root / relative
        observed = P.sha256_file(path) if path.is_file() else None
        checks[f"parent:{relative}"] = observed == expected
        rows[relative] = observed
    controls = root / PACKAGE_REL / contract["controls"]["path"]
    checks["controls"] = controls.is_file() and P.sha256_file(controls) == contract["controls"]["sha256"]
    projection = root / PACKAGE_REL / contract["data"]["1b"]["accepted_projection_path"]
    checks["accepted_ex48_data_projection_file"] = (
        projection.is_file()
        and P.sha256_file(projection) == contract["data"]["1b"]["accepted_projection_sha256"]
    )
    if checks["accepted_ex48_data_projection_file"]:
        projection_checks = P.validate_accepted_data_projection(P.read_json(projection), contract)
        checks["accepted_ex48_data_projection_contract"] = bool(projection_checks) and all(projection_checks.values())
    else:
        checks["accepted_ex48_data_projection_contract"] = False
    optimizer = root / PACKAGE_REL / "moonlight_optimizer.py"
    try:
        optimizer_source = optimizer.read_text(encoding="utf-8")
        compile(optimizer_source, str(optimizer), "exec")
        transfer = source_builder.audit_moonlight_transfer(root, optimizer_source)
        derived = source_builder.build(root)
        compile(derived.trainer, "<derived_moonlight_trainer>", "exec")
        compile(derived.long_worker_parent, "<derived_moonlight_long_worker>", "exec")
        checks["moonlight_optimizer"] = True
        checks["moonlight_ex19_algorithm_subtrees"] = bool(transfer["passed"])
        checks["deterministic_builder"] = True
        rows["moonlight_transfer"] = transfer
        rows["derived_trainer_sha256"] = source_builder.sha256_text(derived.trainer)
        rows["derived_long_worker_sha256"] = source_builder.sha256_text(derived.long_worker_parent)
        rows["moonlight_optimizer_sha256"] = source_builder.sha256_text(derived.moonlight_optimizer)
    except Exception as exc:
        checks["moonlight_optimizer"] = False
        checks["moonlight_ex19_algorithm_subtrees"] = False
        checks["deterministic_builder"] = False
        rows["builder_error"] = repr(exc)
    return {"passed": all(checks.values()), "checks": checks, "sources": rows}

def snapshot_sources(run_dir: Path, live_root: Path) -> Path:
    snapshot = run_dir / "source_snapshot"
    manifest_path = snapshot / "source_snapshot_manifest.json"
    if snapshot.exists():
        manifest = P.read_json(manifest_path)
        required = {
            (PACKAGE_REL / name).as_posix()
            for name in (
                "ex57_contract.json", "accepted_ex48_data_projection.json",
                "protocol.py", "run_suite.py", "analyze.py", "long_worker.py",
                "source_builder.py", "moonlight_optimizer.py", "runtime.py", "frozen_controls.csv",
            )
        } | {
            "derived/train_llama_moonlight.py",
            "derived/train_llama_moonlight_1b.py",
            "derived/ex48_train_segment_parent.py",
            "derived/moonlight_optimizer.py",
        } | set(PARENT_FILES)
        checks = {
            relative: (snapshot / relative).is_file()
            and P.sha256_file(snapshot / relative) == row["sha256"]
            for relative, row in manifest.get("files", {}).items()
        }
        if (
            not checks
            or not required.issubset(set(manifest.get("files", {})))
            or not all(checks.values())
        ):
            raise RuntimeError("EX57 source snapshot is incomplete or changed")
        snapshot_contract = P.read_json(snapshot / PACKAGE_REL / CONTRACT_NAME)
        P.assert_contract(snapshot_contract)
        projection_path = (
            snapshot / PACKAGE_REL
            / snapshot_contract["data"]["1b"]["accepted_projection_path"]
        )
        if (
            P.sha256_file(projection_path)
            != snapshot_contract["data"]["1b"]["accepted_projection_sha256"]
            or not all(
                P.validate_accepted_data_projection(
                    P.read_json(projection_path), snapshot_contract
                ).values()
            )
        ):
            raise RuntimeError("EX57 snapshot has no accepted EX48 data projection")
        return snapshot
    live = check_sources(live_root)
    if not live["passed"]:
        raise RuntimeError(f"EX57 live source check failed: {live['checks']}")
    snapshot.mkdir(parents=True, exist_ok=False)
    sources = package_files(live_root) + [live_root / relative for relative in PARENT_FILES]
    for source in sources:
        relative = source.relative_to(live_root)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    derived_manifest = source_builder.write_derived(snapshot, snapshot / "derived")
    all_files = [path for path in snapshot.rglob("*") if path.is_file() and path != manifest_path]
    payload = {
        "schema_version": "ex57_moonlight_source_snapshot_v1",
        "created_at": now_iso(),
        "derived": derived_manifest,
        "files": {
            path.relative_to(snapshot).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": P.sha256_file(path),
            }
            for path in sorted(all_files)
        },
    }
    P.atomic_json(manifest_path, payload)
    return snapshot


def worker_env(
    snapshot: Path,
    official_repo: Path,
    gpu: int | None,
    config: dict[str, Any],
    contract_sha: str,
    selection_sha: str,
) -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(official_repo.resolve()), str((snapshot / "derived").resolve())]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["MOONLIGHT_WEIGHT_DECAY"] = str(config.get("weight_decay", 0.1))
    env["MOONLIGHT_CONTRACT_SHA256"] = contract_sha
    env["MOONLIGHT_SELECTION_SHA256"] = selection_sha
    if gpu is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def bind_gpu_compile_cache(
    env: dict[str, str],
    run_dir: Path,
    gpu: int,
) -> dict[str, str]:
    """Give each physical GPU its own TorchInductor cache.

    Tuning/formal jobs are intentionally independent single-GPU processes.
    A shared default /tmp/torchinductor_<user> cache can make simultaneously
    launched torch.compile workers wait on the same cold-start cache/compile
    artifacts. Per-GPU cache roots remove that cross-worker serialization
    without changing model math, seeds, data order, batch geometry, or the
    selected optimizer configuration. Timing remains ineligible.
    """
    cache_dir = (run_dir / "_compile_cache" / f"gpu{int(gpu)}").absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    env["EX57_EXPECTED_PHYSICAL_GPU"] = str(int(gpu))
    return env


def run_logged(command: list[str], env: dict[str, str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND " + json.dumps(command) + "\n")
        binding = {
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
            "expected_physical_gpu": env.get("EX57_EXPECTED_PHYSICAL_GPU"),
            "torchinductor_cache_dir": env.get("TORCHINDUCTOR_CACHE_DIR"),
        }
        handle.write("EX57_GPU_BINDING " + json.dumps(binding, sort_keys=True) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        handle.write(f"EX57_WORKER_PID {process.pid}\n")
        handle.flush()
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return process.wait()


def runtime_probe(args: argparse.Namespace, snapshot: Path, contract: dict[str, Any]) -> dict[str, Any]:
    code = r'''
import hashlib,json,pathlib,sys,numpy,torch,triton
expected=json.loads(sys.argv[1]); kernel=pathlib.Path(sys.argv[2])
def sha(path):
 d=hashlib.sha256()
 with path.open("rb") as h:
  for b in iter(lambda:h.read(4*1024*1024),b""): d.update(b)
 return d.hexdigest()
devices=[]
for i in range(torch.cuda.device_count()):
 p=torch.cuda.get_device_properties(i)
 devices.append({"index":i,"name":p.name,"compute_capability":[p.major,p.minor],"total_memory":p.total_memory})
observed={"executable":str(pathlib.Path(sys.executable).absolute()),"prefix":sys.prefix,"base_prefix":sys.base_prefix,"pythonpath":__import__("os").environ.get("PYTHONPATH"),"torch_file":str(pathlib.Path(torch.__file__).absolute()),"numpy_file":str(pathlib.Path(numpy.__file__).absolute()),"python":sys.version.split()[0],"torch":torch.__version__,"torch_cuda":torch.version.cuda,"triton":triton.__version__,"numpy":numpy.__version__,"device_count":torch.cuda.device_count(),"devices":devices,"triton_kernels_sha256":sha(kernel) if kernel.is_file() else None}
checks={"requested_executable":observed["executable"]==expected["requested_executable"],"python":observed["python"]==expected["python"],"torch":observed["torch"]==expected["torch"],"torch_cuda":observed["torch_cuda"]==expected["torch_cuda"],"triton":observed["triton"]==expected["triton"],"numpy":observed["numpy"]==expected["numpy"],"cuda":torch.cuda.is_available(),"count":observed["device_count"]==expected["selected_count"],"kernel":observed["triton_kernels_sha256"]==expected["accepted_triton_kernels_sha256"],"devices":len(devices)==expected["selected_count"] and all(expected["gpu_name_contains"] in d["name"] and d["compute_capability"]==expected["compute_capability"] and d["total_memory"]>=expected["minimum_gpu_memory_bytes"] for d in devices)}
print(json.dumps({"passed":all(checks.values()),"checks":checks,"observed":observed},sort_keys=True))
raise SystemExit(0 if all(checks.values()) else 2)
'''
    expected = dict(contract["runtime"])
    expected["selected_count"] = len(args.gpus)
    expected["requested_executable"] = str(args.training_python.absolute())
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in args.gpus)
    # Do not replace PYTHONPATH here.  On the frozen EX48/EX57 host the
    # torch-2.8 runtime is partly selected by the inherited Python search
    # path.  Replacing it with OFFICIAL_REPO made this nested probe fall back
    # to the system torch-2.3 stack even though training_python itself was
    # correct.  The probe only hashes triton_kernels.py by path, so it does not
    # need OFFICIAL_REPO on PYTHONPATH at all.
    completed = subprocess.run(
        [str(args.training_python.absolute()), "-c", code, json.dumps(expected), str(args.official_repo / "triton_kernels.py")],
        env=env,
        text=True,
        capture_output=True,
    )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception:
        payload = {"passed": False, "stdout": completed.stdout, "stderr": completed.stderr}
    payload["return_code"] = completed.returncode
    return payload


def init_command(
    args: argparse.Namespace, snapshot: Path, scale: str, seed: int, config: dict[str, Any]
) -> tuple[list[str], dict[str, str]]:
    trainer = snapshot / "derived/train_llama_moonlight.py"
    output = args.run_dir / "preflight/init_only" / scale / f"seed{seed}"
    device_batch_size_1b = int(
        getattr(args, "device_batch_size_1b", DEVICE_BATCH_SIZE_1B)
    )
    command = [
        str(args.training_python.absolute()),
        str(trainer if scale == "124m" else snapshot / "derived/train_llama_moonlight_1b.py"),
        "--method", "moonlight", "--data-dir", str(args.data124_dir if scale == "124m" else args.data1b_dir),
        "--output-dir", str(output), "--seed", str(seed), "--num-iterations", "1",
        "--global-batch-size", "512", "--device-batch-size", "64" if scale == "124m" else str(device_batch_size_1b),
        "--sequence-length", "1024", "--val-every", "1", "--val-tokens", "65536" if scale == "124m" else "8192",
        "--warmdown-iters", "0", "--backup-lr", str(config.get("backup_lr", config["matrix_lr"])), "--matrix-lr", str(config["matrix_lr"]),
        "--adamw-matrix-lr", "0.000576", "--checkpoint-every", "0", "--resume", "never", "--init-only",
    ]
    env = worker_env(snapshot, args.official_repo, None, config, P.sha256_file(snapshot / PACKAGE_REL / CONTRACT_NAME), "preflight_init_only")
    if scale == "1b":
        env["LLAMA_1B_BASE_TRAINER"] = str(trainer)
        env["LLAMA_1B_BASE_TRAINER_SHA256"] = P.sha256_file(trainer)
    return command, env


def init_audit(args: argparse.Namespace, snapshot: Path, contract: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for scale in ("124m", "1b"):
        config = contract["tuning"][scale]["cells"][1]
        for seed in contract["formal"]["seeds"]:
            command, env = init_command(args, snapshot, scale, int(seed), config)
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
            completed = subprocess.run(command, env=env, text=True, capture_output=True)
            lines = [line for line in completed.stdout.splitlines() if line.startswith("LLAMA_INIT_AUDIT ")]
            if completed.returncode or len(lines) != 1:
                raise RuntimeError(f"EX57 init audit failed {scale}/{seed}: {completed.stdout}\n{completed.stderr}")
            payload = json.loads(lines[0].split(" ", 1)[1])
            expected = contract["accepted_init_sha256"][scale][str(seed)]
            profile = contract["profiles"][scale]
            checks = {
                "init": payload["init_sha256"] == expected,
                "parameters": int(payload["architecture"]["parameter_count"]) == int(profile["parameters"]),
                "matrix_tensors": int(payload["architecture"]["matrix_tensor_count"]) == int(profile["expected_matrix_tensors"]),
                "backup_tensors": int(payload["architecture"]["backup_tensor_count"]) == int(profile["expected_backup_tensors"]),
                "no_activation_k": int(payload["architecture"]["preconditioner_group_count"]) == 0,
            }
            if not all(checks.values()):
                raise RuntimeError(f"EX57 init mismatch {scale}/{seed}: {checks}")
            rows[f"{scale}/seed{seed}"] = {"checks": checks, "payload": payload}
    return {"passed": True, "units": rows}


def require_preflight_context(args: argparse.Namespace) -> dict[str, Any]:
    manifest = P.read_json(args.run_dir / "preflight/preflight_manifest.json")
    paths = manifest.get("paths", {})
    expected_paths = {
        "official_repo": str(args.official_repo.resolve()),
        "data124": str(args.data124_dir.resolve()),
        "data1b": str(args.data1b_dir.resolve()),
        "training_python": str(args.training_python.absolute()),
    }
    checks = {
        "schema": manifest.get("schema_version") == PREFLIGHT_SCHEMA,
        "passed": manifest.get("passed") is True,
        "paths": paths == expected_paths,
        "gpus": manifest.get("gpus") == [0, 1] == args.gpus,
        "ex48_geometry": manifest.get("fairness_protocol") == MEMORY_SAFE_PROTOCOL,
    }
    snapshot_contract = args.run_dir / "source_snapshot" / PACKAGE_REL / CONTRACT_NAME
    checks["contract"] = (
        snapshot_contract.is_file()
        and P.sha256_file(snapshot_contract) == manifest.get("contract_sha256")
        and P.read_json(snapshot_contract).get("schema_version") == P.SCHEMA
    )
    for scale in ("124m", "1b"):
        audit_path = args.run_dir / "preflight" / f"data_{scale}.json"
        audit = P.read_json(audit_path)
        metadata = P.verify_data_metadata(audit)
        checks[f"data_metadata_{scale}"] = bool(metadata) and all(metadata.values())
        checks[f"data_manifest_hash_{scale}"] = P.sha256_file(audit_path) == manifest.get(f"data_{scale}_audit_sha256")
        if scale == "1b":
            checks["data_1b_accepted_projection"] = (
                audit.get("accepted_projection_inventory_sha256")
                == manifest.get("accepted_ex48_data_projection_inventory_sha256")
            )
    if not all(checks.values()):
        raise RuntimeError(f"EX57 invocation no longer matches passed preflight: {checks}")
    return manifest

def preflight(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = run_dir / "preflight/preflight_manifest.json"
    if existing.is_file() and P.read_json(existing).get("passed") is True:
        require_preflight_context(args)
        print(f"skip passed EX57 preflight: {run_dir}")
        return
    snapshot = snapshot_sources(run_dir, args.repo.resolve())
    contract_path = snapshot / PACKAGE_REL / CONTRACT_NAME
    contract = P.read_json(contract_path)
    P.assert_contract(contract)
    runtime = runtime_probe(args, snapshot, contract)
    if not runtime.get("passed"):
        raise RuntimeError(f"EX57 runtime preflight failed: {runtime}")
    projection_path = snapshot / PACKAGE_REL / contract["data"]["1b"]["accepted_projection_path"]
    projection_file_pass = projection_path.is_file() and P.sha256_file(projection_path) == contract["data"]["1b"]["accepted_projection_sha256"]
    accepted_projection = P.read_json(projection_path) if projection_file_pass else {}
    projection_checks = P.validate_accepted_data_projection(accepted_projection, contract)
    projection_contract_pass = bool(projection_checks) and all(projection_checks.values())
    if not projection_file_pass or not projection_contract_pass:
        raise RuntimeError(f"EX57 accepted EX48 data projection failed: file={projection_file_pass} checks={projection_checks}")
    data124 = P.audit_data_dir(args.data124_dir.resolve(), contract, "124m", full_hash=True)
    data1b = P.audit_data_dir(args.data1b_dir.resolve(), contract, "1b", full_hash=True, accepted_projection=accepted_projection)
    if not data124["passed"] or not data1b["passed"]:
        raise RuntimeError(f"EX57 data preflight failed: 124m={data124['checks']} 1b={data1b['checks']}")
    free = shutil.disk_usage(run_dir).free
    disk_pass = free >= int(contract["checkpoint_retention"]["minimum_free_disk_bytes"])
    if not disk_pass:
        raise RuntimeError(f"EX57 requires 600 GB free, observed {free}")
    P.atomic_json(run_dir / "preflight/data_124m.json", data124)
    P.atomic_json(run_dir / "preflight/data_1b.json", data1b)
    init = init_audit(args, snapshot, contract)
    audit_code = "import json; from moonlight_optimizer import run_small_matrix_reference_audit; print(json.dumps(run_small_matrix_reference_audit('cpu'),sort_keys=True))"
    env = worker_env(snapshot, args.official_repo, None, contract["tuning"]["124m"]["cells"][1], P.sha256_file(contract_path), "preflight_small_audit")
    completed = subprocess.run([str(args.training_python.absolute()), "-c", audit_code], env=env, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"EX57 Moonlight reference audit failed: {completed.stdout}\n{completed.stderr}")
    small = json.loads(completed.stdout.strip().splitlines()[-1])
    checks = {
        "snapshot": True,
        "runtime": runtime["passed"] is True,
        "accepted_ex48_data_projection_file": projection_file_pass,
        "accepted_ex48_data_projection_contract": projection_contract_pass,
        "data124": data124["passed"] is True,
        "data1b": data1b["passed"] is True,
        "ex48_microbatch_geometry": contract["fairness"]["same_microbatch_geometry_as_ex48"] is True,
        "disk": disk_pass,
        "init": init["passed"] is True,
        "small_matrix": small.get("passed") is True
        and small.get("state_schema", {}).get("contains_activation_k_state") is False
        and small.get("state_schema", {}).get("contains_factor_or_eigendecomposition_state") is False,
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
        "fairness_protocol": MEMORY_SAFE_PROTOCOL,
        "accepted_ex48_data_projection_sha256": P.sha256_file(projection_path),
        "accepted_ex48_data_projection_inventory_sha256": contract["data"]["1b"]["accepted_projection_inventory_sha256"],
        "data_124m_audit_sha256": P.sha256_file(run_dir / "preflight/data_124m.json"),
        "data_1b_audit_sha256": P.sha256_file(run_dir / "preflight/data_1b.json"),
        "source_snapshot_manifest_sha256": P.sha256_file(snapshot / "source_snapshot_manifest.json"),
        "paths": {"official_repo": str(args.official_repo.resolve()), "data124": str(args.data124_dir.resolve()), "data1b": str(args.data1b_dir.resolve()), "training_python": str(args.training_python.absolute())},
        "gpus": args.gpus,
    }
    P.atomic_json(existing, payload)
    if not payload["passed"]:
        raise RuntimeError(f"EX57 preflight failed: {checks}")
    P.atomic_json(run_dir / "status.json", {"status": "preflight_passed", "updated_at": now_iso()})
    print(f"EX57 preflight passed. Artifacts: {run_dir}")

def expected_moonlight_hyperparameters(contract: dict[str, Any]) -> dict[str, Any]:
    moonlight = contract["moonlight"]
    return {
        "momentum": float(moonlight["momentum"]),
        "nesterov": bool(moonlight["nesterov"]),
        "ns_steps": int(moonlight["newton_schulz_steps"]),
        "weight_decay": float(moonlight["weight_decay"]),
    }

def moonlight_runtime_checks(
    summary: dict[str, Any], contract: dict[str, Any], scale: str
) -> dict[str, bool]:
    schema = summary.get("moonlight_state_schema")
    expected_matrices = int(contract["profiles"][scale]["expected_matrix_tensors"])
    observed_hyperparameters = summary.get("moonlight_hyperparameters")
    return {
        "hyperparameters": observed_hyperparameters == expected_moonlight_hyperparameters(contract),
        "all_matrix_states": isinstance(schema, dict)
        and schema.get("optimizer") == "R1MoonlightMuon"
        and schema.get("tensor_state_keys") == ["momentum_buffer"]
        and int(schema.get("logical_matrix_parameters", -1)) == expected_matrices
        and schema.get("contains_activation_k_state") is False
        and schema.get("contains_factor_or_eigendecomposition_state") is False,
    }

def validate_local_summary(
    path: Path,
    *,
    scale: str,
    seed: int,
    steps: int,
    contract: dict[str, Any],
    contract_sha: str,
    selection_sha: str,
    data_inventory_sha: str,
) -> dict[str, Any]:
    summary = P.read_json(path)
    profile = contract["profiles"][scale]
    runtime_checks = moonlight_runtime_checks(summary, contract, scale)
    matrix_state = int(summary.get("moonlight_matrix_optimizer_state_bytes", 0))
    checks = {
        "status": summary.get("status") == "completed",
        "method": summary.get("method") == "moonlight",
        "seed": int(summary.get("seed", -1)) == int(seed),
        "steps": int(summary.get("completed_steps", -1)) == int(steps),
        "tokens": int(summary.get("tokens_seen", -1)) == int(steps) * int(contract["training"]["tokens_per_update"]),
        "parameters": int(summary.get("architecture", {}).get("parameter_count", -1)) == int(profile["parameters"]),
        "init": summary.get("init_sha256") == contract["accepted_init_sha256"].get(scale, {}).get(str(seed), summary.get("init_sha256")),
        "finite": all(P.finite_number(summary.get(key)) for key in ("final_val_loss", "final_train_loss")),
        "no_k": int(summary.get("k_state_bytes", -1)) == 0,
        "contract": summary.get("moonlight_contract_sha256") == contract_sha,
        "selection": summary.get("moonlight_selection_sha256") == selection_sha,
        "data_inventory": summary.get("moonlight_data_inventory_sha256") == data_inventory_sha,
        "timing": summary.get("timing_comparable") is False,
        "moonlight_state": matrix_state > 0
        and int(summary.get("optimizer_state_bytes", 0)) >= matrix_state
        and int(summary.get("peak_allocated_bytes", 0)) > 0,
        "moonlight_hyperparameters": runtime_checks["hyperparameters"],
        "all_matrix_states": runtime_checks["all_matrix_states"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid EX57 local summary {path}: {checks}")
    return summary

def direct_command(
    args: argparse.Namespace,
    snapshot: Path,
    scale: str,
    seed: int,
    steps: int,
    output: Path,
    config: dict[str, Any],
    *,
    warmdown: int,
    checkpoint_every: int,
    no_save_final: bool,
) -> list[str]:
    trainer = runtime_direct_trainer(args, snapshot)
    script = trainer if scale == "124m" else snapshot / "derived/train_llama_moonlight_1b.py"
    device_batch_size_1b = int(getattr(args, "device_batch_size_1b", DEVICE_BATCH_SIZE_1B))
    base_lr = float(config["matrix_lr"])
    backup_lr = float(config.get("backup_lr", base_lr))
    if backup_lr != base_lr:
        raise RuntimeError("Moonlight contract requires one shared base LR for matrix and backup routes")
    command = [
        str(args.training_python.absolute()), str(script), "--method", "moonlight",
        "--data-dir", str(args.data124_dir if scale == "124m" else args.data1b_dir),
        "--output-dir", str(output), "--seed", str(seed), "--num-iterations", str(steps),
        "--global-batch-size", "512", "--device-batch-size", "64" if scale == "124m" else str(device_batch_size_1b),
        "--sequence-length", "1024", "--val-every", "100", "--val-tokens", "10485760",
        "--warmdown-iters", str(warmdown), "--backup-lr", str(backup_lr), "--matrix-lr", str(base_lr),
        "--adamw-matrix-lr", "0.000576", "--checkpoint-every", str(checkpoint_every), "--resume", "auto",
    ]
    if no_save_final:
        command.append("--no-save-final")
    return command

def direct_env(args: argparse.Namespace, snapshot: Path, scale: str, gpu: int, config: dict[str, Any], contract_sha: str, selection_sha: str) -> dict[str, str]:
    env = worker_env(snapshot, args.official_repo, gpu, config, contract_sha, selection_sha)
    bind_gpu_compile_cache(env, args.run_dir, gpu)
    data_audit = P.read_json(args.run_dir / "preflight" / f"data_{scale}.json")
    env["MOONLIGHT_DATA_INVENTORY_SHA256"] = str(data_audit["inventory_sha256"])
    if scale == "1b":
        trainer = runtime_direct_trainer(args, snapshot)
        env["LLAMA_1B_BASE_TRAINER"] = str(trainer)
        env["LLAMA_1B_BASE_TRAINER_SHA256"] = P.sha256_file(trainer)
    return env

def run_memory_canary(
    args: argparse.Namespace,
    snapshot: Path,
    contract: dict[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    raise RuntimeError("EX57 Moonlight does not use an optimizer-specific memory-canary stage; this function must not be called")

def tuning_cell_valid(
    attempt: Path,
    *,
    scale: str,
    cell: dict[str, Any],
    seed: int,
    steps: int,
    contract: dict[str, Any],
    contract_sha: str,
    selection_marker: str,
    data_inventory_sha: str,
) -> bool:
    try:
        manifest = P.read_json(attempt / "cell_manifest.json")
        summary = validate_local_summary(
            attempt / "summary.json",
            scale=scale,
            seed=seed,
            steps=steps,
            contract=contract,
            contract_sha=contract_sha,
            selection_sha=selection_marker,
            data_inventory_sha=data_inventory_sha,
        )
        return all(
            (
                manifest.get("passed") is True,
                manifest.get("scale") == scale,
                manifest.get("cell") == cell,
                int(manifest.get("seed", -1)) == seed,
                int(manifest.get("steps", -1)) == steps,
                int(manifest.get("physical_gpu", -1)) in {int(value) for value in contract["execution"]["physical_gpus"]},
                manifest.get("contract_sha256") == contract_sha,
                manifest.get("selection_marker") == selection_marker,
                manifest.get("data_inventory_sha256") == data_inventory_sha,
                P.sha256_file(attempt / "summary.json")
                == manifest.get("summary_sha256"),
                P.sha256_file(attempt / "metrics.csv")
                == manifest.get("metrics_sha256"),
                float(manifest.get("final_val_loss"))
                == float(summary["final_val_loss"]),
            )
        )
    except Exception:
        return False


def run_tuning_cell(args: argparse.Namespace, snapshot: Path, contract: dict[str, Any], scale: str, cell: dict[str, Any], gpu: int) -> dict[str, Any]:
    root = args.run_dir / "tuning" / scale / cell["id"]
    seed = int(contract["tuning"][scale]["seed"])
    steps = int(contract["tuning"][scale]["updates"])
    contract_sha = P.sha256_file(snapshot / PACKAGE_REL / CONTRACT_NAME)
    marker = "tuning_unselected_" + contract_sha
    data_inventory_sha = P.read_json(
        args.run_dir / "preflight" / f"data_{scale}.json"
    )["inventory_sha256"]
    for attempt in sorted(root.glob("attempt_*")):
        if tuning_cell_valid(
            attempt,
            scale=scale,
            cell=cell,
            seed=seed,
            steps=steps,
            contract=contract,
            contract_sha=contract_sha,
            selection_marker=marker,
            data_inventory_sha=data_inventory_sha,
        ):
            return P.read_json(attempt / "cell_manifest.json")
    attempt = root / f"attempt_{len(list(root.glob('attempt_*'))) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    command = direct_command(args, snapshot, scale, seed, steps, attempt, cell, warmdown=0, checkpoint_every=0, no_save_final=True)
    rc = run_logged(command, direct_env(args, snapshot, scale, gpu, cell, contract_sha, marker), attempt / "worker.log")
    if rc:
        raise RuntimeError(f"EX57 tuning cell failed {scale}/{cell['id']} rc={rc}")
    summary = validate_local_summary(
        attempt / "summary.json",
        scale=scale,
        seed=seed,
        steps=steps,
        contract=contract,
        contract_sha=contract_sha,
        selection_sha=marker,
        data_inventory_sha=data_inventory_sha,
    )
    payload = {
        "schema_version": "ex57_tuning_cell_v1", "passed": True, "scale": scale,
        "cell": cell, "seed": seed, "steps": steps, "physical_gpu": int(gpu),
        "final_val_loss": summary["final_val_loss"],
        "contract_sha256": contract_sha, "selection_marker": marker,
        "data_inventory_sha256": data_inventory_sha,
        "summary_sha256": P.sha256_file(attempt / "summary.json"),
        "metrics_sha256": P.sha256_file(attempt / "metrics.csv"),
        "quality_eligible": False, "timing_eligible": False,
    }
    P.atomic_json(attempt / "cell_manifest.json", payload)
    return payload


def schedule(items: list[Any], gpus: list[int], fn: Callable[[Any, int], Any]) -> list[Any]:
    """Run fixed per-GPU queues concurrently and record the physical GPU.

    EX57 has exactly three tuning cells and three formal seeds on GPUs 0/1/2.
    Each job remains a *single-GPU* scientific run; concurrency only reduces
    wall-clock time and therefore does not change batch geometry or optimizer
    semantics. A barrier makes the first job on every populated GPU begin at
    the same scheduling wave, which also makes multi-GPU utilization explicit.
    """
    if not gpus:
        raise ValueError("EX57 scheduler requires at least one GPU")
    indexed = list(enumerate(items))
    buckets: dict[int, list[tuple[int, Any]]] = {gpu: [] for gpu in gpus}
    for index, item in indexed:
        buckets[gpus[index % len(gpus)]].append((index, item))
    active = [(gpu, rows) for gpu, rows in buckets.items() if rows]
    start_barrier = Barrier(len(active)) if len(active) > 1 else None

    def run_bucket(gpu: int, rows: list[tuple[int, Any]]) -> list[tuple[int, Any]]:
        if start_barrier is not None:
            start_barrier.wait()
        out: list[tuple[int, Any]] = []
        for index, item in rows:
            print(f"EX57_SCHED_START gpu={gpu} index={index} item={item}", flush=True)
            result = fn(item, gpu)
            if isinstance(result, dict):
                result = dict(result)
                result.setdefault("physical_gpu", int(gpu))
            print(f"EX57_SCHED_DONE gpu={gpu} index={index} item={item}", flush=True)
            out.append((index, result))
        return out

    completed: list[tuple[int, Any]] = []
    with ThreadPoolExecutor(max_workers=len(active)) as executor:
        pending = [executor.submit(run_bucket, gpu, rows) for gpu, rows in active]
        for future in as_completed(pending):
            completed.extend(future.result())
    return [result for _, result in sorted(completed, key=lambda row: row[0])]


def select_cell(contract: dict[str, Any], scale: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    spec = contract["tuning"][scale]
    by_id = {row["cell"]["id"]: row for row in rows}
    if set(by_id) != {cell["id"] for cell in spec["cells"]}:
        raise RuntimeError(f"incomplete EX57 tuning grid for {scale}")
    best = min(rows, key=lambda row: (float(row["final_val_loss"]), row["cell"]["id"]))
    center = by_id[spec["center_cell"]]
    selected = center if float(center["final_val_loss"]) <= float(best["final_val_loss"]) + float(contract["tuning"]["center_tie_margin"]) else best
    return {"scale": scale, "selected_cell": selected["cell"], "selected_loss": selected["final_val_loss"], "best_observed_cell": best["cell"]["id"], "center_retained": selected is center, "cells": sorted(rows, key=lambda row: row["cell"]["id"])}


def write_selected_contract(run_dir: Path, snapshot: Path, contract: dict[str, Any], selection_sha: str, selected: dict[str, Any]) -> Path:
    payload = json.loads(json.dumps(contract))
    payload["selection_manifest_sha256"] = selection_sha
    payload["selected_configs"] = {scale: selected[scale]["selected_cell"] for scale in ("124m", "1b")}
    payload["training"]["matrix_lr"] = float(selected["1b"]["selected_cell"]["matrix_lr"])
    payload["training"]["backup_lr"] = float(selected["1b"]["selected_cell"].get("backup_lr", payload["training"]["matrix_lr"]))
    path = run_dir / "tuning/selected_formal_contract.json"
    if path.is_file():
        existing = P.read_json(path)
        P.assert_contract(existing)
        stable_checks = {
            "selection": existing.get("selection_manifest_sha256") == selection_sha,
            "configs": existing.get("selected_configs") == payload["selected_configs"],
            "matrix_lr": float(existing["training"]["matrix_lr"])
            == float(payload["training"]["matrix_lr"]),
            "backup_lr": float(existing["training"]["backup_lr"])
            == float(payload["training"]["backup_lr"]),
        }
        if not all(stable_checks.values()):
            raise RuntimeError(
                "EX57 selected contract changed across resume: "
                f"{stable_checks}"
            )
        return path
    P.atomic_json(path, payload)
    P.assert_contract(P.read_json(path))
    return path


def long_worker_profile(contract: dict[str, Any]) -> dict[str, Any]:
    """Adapt EX57's scale-indexed audit profile to the EX48 worker contract."""

    profile = dict(contract["profiles"]["1b"])
    profile.update(
        {
            "n_layer": 18,
            "n_head": 16,
            "n_embd": 2048,
            "intermediate_size": 5504,
            "expected_preconditioner_groups": {"moonlight": 0},
        }
    )
    return profile


LONG_WORKER_PROFILE_COMPAT_MARKER = "EX57_LONG_WORKER_CONTRACT_COMPAT_V2"
LONG_WORKER_PROFILE_COMPAT_SOURCE_MARKER = (
    "EX57_LONG_WORKER_CONTRACT_COMPAT_V3_SOURCE_VIEW"
)


def patch_long_worker_profile_adapter(source: str) -> str:
    """Inject EX48's worker view without changing frozen contract bytes."""

    # Fresh EX57 Moonlight snapshots already carry the complete V3 source-view
    # adapter in long_worker.py.  Treat that implementation as authoritative
    # and do not try to apply the legacy V2 string patch a second time.
    if LONG_WORKER_PROFILE_COMPAT_SOURCE_MARKER in source:
        required_v3_fragments = (
            "base_trainer_sha256 = protocol.sha256_file(base_trainer_path)",
            "payload[\"profile\"] = profile",
            "payload[\"grid\"] = grid",
            "payload[\"accepted_sources\"] = accepted_sources",
        )
        missing = [fragment for fragment in required_v3_fragments if fragment not in source]
        if missing:
            raise RuntimeError(
                "EX57 built-in V3 long-worker compatibility adapter is incomplete: "
                f"{missing}"
            )
        return source
    if LONG_WORKER_PROFILE_COMPAT_MARKER in source:
        return source
    old = (
        "    module.P = protocol\n"
        "    return int(module.main())"
    )
    new = (
        "    module.P = protocol\n"
        f"    # {LONG_WORKER_PROFILE_COMPAT_MARKER}\n"
        "    original_read_json = protocol.read_json\n"
        "    contract_index = sys.argv.index('--contract')\n"
        "    contract_path = Path(sys.argv[contract_index + 1]).resolve()\n"
        "\n"
        "    def read_json_with_worker_profile(path: Path):\n"
        "        payload = original_read_json(path)\n"
        "        if Path(path).resolve() == contract_path:\n"
        "            payload = dict(payload)\n"
        "            profile = dict(payload['profiles']['1b'])\n"
        "            profile.update(payload.get('profile', {}))\n"
        "            profile.update({\n"
        "                'n_layer': 18, 'n_head': 16, 'n_embd': 2048,\n"
        "                'intermediate_size': 5504,\n"
        "                'expected_preconditioner_groups': {'moonlight': 0},\n"
        "            })\n"
        "            payload['profile'] = profile\n"
        "            grid = dict(payload.get('grid', {}))\n"
        "            grid.update({\n"
        "                'methods': ['moonlight'],\n"
        "                'seeds': list(payload['formal']['seeds']),\n"
        "                'formal_units': len(payload['formal']['seeds']),\n"
        "                'host_count': 1,\n"
        "                'gpus': len(payload['execution']['physical_gpus']),\n"
        "            })\n"
        "            payload['grid'] = grid\n"
        "        return payload\n"
        "\n"
        "    protocol.read_json = read_json_with_worker_profile\n"
        "    return int(module.main())"
    )
    if source.count(old) != 1:
        raise RuntimeError("EX57 long-worker profile adapter anchor changed")
    return source.replace(old, new, 1)


def runtime_long_worker_adapter(args: argparse.Namespace, snapshot: Path) -> Path:
    original = snapshot / PACKAGE_REL / "long_worker.py"
    source = original.read_text(encoding="utf-8")
    patched = patch_long_worker_profile_adapter(source)
    target = original
    if patched != source:
        target = snapshot / PACKAGE_REL / "long_worker_contract_compat_v2.py"
        if target.is_file():
            if target.read_text(encoding="utf-8") != patched:
                raise RuntimeError("EX57 long-worker adapter changed across resume")
        else:
            P.atomic_text(target, patched)
    receipt_path = args.run_dir / "long_worker_contract_compatibility_amendment_v2.json"
    payload = {
        "schema_version": "ex57_long_worker_contract_compatibility_amendment_v2",
        "passed": True,
        "created_at": now_iso(),
        "reason": "supply_complete_1b_profile_and_formal_grid_to_accepted_ex48_worker_in_memory",
        "outcome_based_change": False,
        "scientific_training_contract_changed": False,
        "selected_contract_bytes_changed": False,
        "method_hyperparameter_seed_data_or_schedule_changed": False,
        "compatibility_was_built_into_frozen_adapter": target == original,
        "injected_worker_view_keys": ["profile", "grid"],
        "original_source_snapshot_manifest_sha256": P.sha256_file(
            snapshot / "source_snapshot_manifest.json"
        ),
        "original_adapter_sha256": P.sha256_file(original),
        "amended_adapter_sha256": P.sha256_file(target),
        "amended_adapter_path": str(target),
    }
    if receipt_path.is_file():
        existing = P.read_json(receipt_path)
        stable = set(payload) - {"created_at"}
        if any(existing.get(key) != payload[key] for key in stable):
            raise RuntimeError("EX57 long-worker contract compatibility receipt changed")
    else:
        P.atomic_json(receipt_path, payload)
    return target


def runtime_direct_trainer(args: argparse.Namespace, snapshot: Path) -> Path:
    """Return a hash-receipted RNG-compatible trainer for direct resumes.

    Older EX57 snapshots were frozen before the CPU-ByteTensor normalization
    was added.  They remain immutable.  A deterministic sibling source and an
    amendment receipt preserve the original source hash while allowing the
    already-written checkpoint to resume in the same run directory.
    """

    original = snapshot / "derived/train_llama_moonlight.py"
    source = original.read_text(encoding="utf-8")
    patched = source_builder.patch_rng_restore(source)
    target = original
    if patched != source:
        target = snapshot / "derived/train_llama_moonlight_rng_cpu_compat_v1.py"
        if target.is_file():
            if target.read_text(encoding="utf-8") != patched:
                raise RuntimeError("EX57 RNG compatibility source changed across resume")
        else:
            P.atomic_text(target, patched)
    receipt_path = args.run_dir / "rng_resume_compatibility_amendment_v1.json"
    payload = {
        "schema_version": "ex57_rng_resume_compatibility_amendment_v1",
        "passed": True,
        "created_at": now_iso(),
        "reason": "normalize_cuda_mapped_rng_states_to_cpu_bytetensors",
        "compatibility_was_built_into_frozen_trainer": target == original,
        "outcome_based_change": False,
        "scientific_training_contract_changed": False,
        "method_or_hyperparameter_changed": False,
        "seed_data_step_or_schedule_changed": False,
        "model_and_optimizer_checkpoint_tensors_remain_cuda_resident": True,
        "original_source_snapshot_manifest_sha256": P.sha256_file(
            snapshot / "source_snapshot_manifest.json"
        ),
        "original_trainer_sha256": P.sha256_file(original),
        "amended_trainer_sha256": P.sha256_file(target),
        "amended_trainer_path": str(target),
    }
    if receipt_path.is_file():
        existing = P.read_json(receipt_path)
        stable = set(payload) - {"created_at"}
        if any(existing.get(key) != payload[key] for key in stable):
            raise RuntimeError("EX57 RNG compatibility amendment no longer matches this run")
    else:
        P.atomic_json(receipt_path, payload)
    return target


def long_worker_command(
    args: argparse.Namespace, snapshot: Path, selected_contract: Path, phase: str, output: Path,
    *, seed: int, source: dict[str, Any] | None = None, max_new_steps: int | None = None,
) -> list[str]:
    command = [
        str(args.training_python.absolute()), str(runtime_long_worker_adapter(args, snapshot)),
        "--parent-worker", str(snapshot / "derived/ex48_train_segment_parent.py"),
        "--contract", str(selected_contract), "--base-trainer", str(snapshot / "derived/train_llama_moonlight.py"),
        "--data-audit", str(args.run_dir / "preflight/data_1b.json"), "--method", "moonlight",
        "--seed", str(seed), "--phase-id", phase, "--output-dir", str(output), "--resume", "auto",
    ]
    if source:
        command += ["--source-checkpoint", str(source["path"]), "--source-checkpoint-sha256", str(source["sha256"])]
    if max_new_steps is not None:
        command += ["--max-new-steps", str(max_new_steps)]
    return command


def engineering_pilot_manifest_valid(
    args: argparse.Namespace,
    contract: dict[str, Any],
    selection_sha: str,
    selected_contract_sha: str,
) -> bool:
    try:
        path = args.run_dir / "tuning/engineering_pilot_manifest.json"
        payload = P.read_json(path)
        cert124 = payload["124m"]
        retirement124_path = (
            args.run_dir
            / "tuning/engineering_pilot/124m_resume/checkpoint_retirement.json"
        )
        if not (
            payload.get("passed") is True
            and payload.get("quality_eligible") is False
            and payload.get("timing_eligible") is False
            and payload.get("selection_sha256") == selection_sha
            and payload.get("selected_contract_sha256") == selected_contract_sha
            and int(cert124.get("bytes", 0)) > 0
            and len(str(cert124.get("sha256", ""))) == 64
            and int(cert124.get("resume_count", 0)) >= 1
            and not Path(cert124["path"]).exists()
            and retirement124_path.is_file()
            and P.sha256_file(retirement124_path)
            == cert124.get("retirement_sha256")
            and P.read_json(retirement124_path).get("children")
            == ["engineering_pilot_complete"]
        ):
            return False
        validate_local_summary(
            args.run_dir / "tuning/engineering_pilot/124m_resume/summary.json",
            scale="124m", seed=5401, steps=4, contract=contract,
            contract_sha=selected_contract_sha, selection_sha=selection_sha,
            data_inventory_sha=P.read_json(
                args.run_dir / "preflight/data_124m.json"
            )["inventory_sha256"],
        )
        data1b_sha = P.read_json(args.run_dir / "preflight/data_1b.json")[
            "inventory_sha256"
        ]
        for name in ("1b_base", "1b_branch"):
            if not phase_valid(
                args.run_dir / "tuning/engineering_pilot" / name,
                selected_contract_sha, 5701, data1b_sha,
                selection_sha=selection_sha, contract=contract,
            ):
                return False
            phase_manifest = P.read_json(
                args.run_dir
                / "tuning/engineering_pilot"
                / name
                / "phase_manifest.json"
            )
            retirement = P.read_json(
                args.run_dir
                / "tuning/engineering_pilot"
                / name
                / "checkpoint_retirement.json"
            )
            if not (
                not Path(phase_manifest["checkpoint"]["path"]).exists()
                and retirement.get("children") == ["engineering_pilot_complete"]
                and retirement.get("sha256")
                == phase_manifest["checkpoint"]["sha256"]
                and int(retirement.get("bytes", -1))
                == int(phase_manifest["checkpoint"]["bytes"])
            ):
                return False
        base_manifest = P.read_json(
            args.run_dir / "tuning/engineering_pilot/1b_base/phase_manifest.json"
        )
        branch_summary = P.read_json(
            args.run_dir / "tuning/engineering_pilot/1b_branch/summary.json"
        )
        return (
            int(P.read_json(
                args.run_dir / "tuning/engineering_pilot/1b_base/summary.json"
            ).get("resume_count", 0)) >= 1
            and branch_summary.get("source_checkpoint_sha256")
            == base_manifest["checkpoint"]["sha256"]
            and payload.get("1b_retired")
            == [
                base_manifest["checkpoint"],
                P.read_json(
                    args.run_dir
                    / "tuning/engineering_pilot/1b_branch/phase_manifest.json"
                )["checkpoint"],
            ]
        )
    except Exception:
        return False


def run_engineering_pilots(args: argparse.Namespace, snapshot: Path, contract: dict[str, Any], selected: dict[str, Any], selection_sha: str, selected_contract: Path) -> dict[str, Any]:
    manifest_path = args.run_dir / "tuning/engineering_pilot_manifest.json"
    selected_contract_sha = P.sha256_file(selected_contract)
    if manifest_path.is_file() and engineering_pilot_manifest_valid(
        args, contract, selection_sha, selected_contract_sha
    ):
        return P.read_json(manifest_path)
    contract_sha = selected_contract_sha
    # 124M: complete two updates, then resume the same exact checkpoint to four.
    cfg124 = selected["124m"]["selected_cell"]
    out124 = args.run_dir / "tuning/engineering_pilot/124m_resume"
    out124.mkdir(parents=True, exist_ok=True)
    for target in (2, 4):
        existing_summary = (
            P.read_json(out124 / "summary.json")
            if (out124 / "summary.json").is_file()
            else {}
        )
        if int(existing_summary.get("completed_steps", -1)) >= target:
            continue
        command = direct_command(args, snapshot, "124m", 5401, target, out124, cfg124, warmdown=0, checkpoint_every=1, no_save_final=False)
        rc = run_logged(command, direct_env(args, snapshot, "124m", args.gpus[0], cfg124, contract_sha, selection_sha), out124 / "worker.log")
        if rc:
            raise RuntimeError(f"EX57 124M exact-resume pilot failed target={target} rc={rc}")
    summary124 = validate_local_summary(
        out124 / "summary.json",
        scale="124m",
        seed=5401,
        steps=4,
        contract=contract,
        contract_sha=contract_sha,
        selection_sha=selection_sha,
        data_inventory_sha=P.read_json(args.run_dir / "preflight/data_124m.json")[
            "inventory_sha256"
        ],
    )
    checkpoint124 = out124 / "checkpoint_latest.pt"
    retirement124 = out124 / "checkpoint_retirement.json"
    if checkpoint124.is_file():
        checkpoint124_row = {
            "path": str(checkpoint124), "bytes": checkpoint124.stat().st_size,
            "sha256": P.sha256_file(checkpoint124),
        }
    elif retirement124.is_file():
        retired_row = P.read_json(retirement124)
        checkpoint124_row = {
            "path": retired_row["path"], "bytes": retired_row["bytes"],
            "sha256": retired_row["sha256"],
        }
    else:
        raise RuntimeError("EX57 124M pilot checkpoint disappeared without a certificate")
    complete_checkpoint_retirement(
        out124, checkpoint124_row, ["engineering_pilot_complete"]
    )
    cert124 = checkpoint124_row | {
        "resume_count": summary124["resume_count"],
        "retirement_sha256": P.sha256_file(retirement124),
    }

    # 1B: accepted EX48 worker planned interruption, in-place resume, then branch.
    cfg1b = selected["1b"]["selected_cell"]
    base = args.run_dir / "tuning/engineering_pilot/1b_base"
    env = worker_env(snapshot, args.official_repo, args.gpus[-1], cfg1b, contract_sha, selection_sha)
    bind_gpu_compile_cache(env, args.run_dir, args.gpus[-1])
    data1b_sha = P.read_json(args.run_dir / "preflight/data_1b.json")[
        "inventory_sha256"
    ]
    if not phase_valid(
        base, contract_sha, 5701, data1b_sha,
        selection_sha=selection_sha, contract=contract,
        full_checkpoint_hash=True,
    ):
        if (base / "phase_manifest.json").is_file():
            raise RuntimeError("EX57 1B pilot base manifest exists but failed audit")
        if not (base / "checkpoint_latest.pt").is_file():
            first = long_worker_command(
                args, snapshot, selected_contract, "pilot_base_2", base,
                seed=5701, max_new_steps=1,
            )
            rc = run_logged(first, env, base / "worker.log")
            if rc != 75:
                raise RuntimeError(
                    f"EX57 1B pilot expected planned rc75, observed {rc}"
                )
        rc = run_logged(
            long_worker_command(
                args, snapshot, selected_contract, "pilot_base_2", base,
                seed=5701,
            ),
            env,
            base / "worker.log",
        )
        if rc:
            raise RuntimeError(f"EX57 1B in-place resume pilot failed rc={rc}")
    base_manifest = P.read_json(base / "phase_manifest.json")
    if not phase_valid(
        base, contract_sha, 5701, data1b_sha,
        selection_sha=selection_sha, contract=contract,
        full_checkpoint_hash=True,
    ):
        raise RuntimeError("EX57 1B resumed pilot base failed state/lineage/hash audit")
    if int(P.read_json(base / "summary.json").get("resume_count", 0)) < 1:
        raise RuntimeError("EX57 1B pilot did not record the planned in-place resume")
    branch = args.run_dir / "tuning/engineering_pilot/1b_branch"
    if not phase_valid(
        branch, contract_sha, 5701, data1b_sha,
        selection_sha=selection_sha, contract=contract,
        full_checkpoint_hash=True,
    ):
        if (branch / "phase_manifest.json").is_file():
            raise RuntimeError("EX57 1B pilot branch manifest exists but failed audit")
        source_path = Path(base_manifest["checkpoint"]["path"])
        if not source_path.is_file():
            raise RuntimeError("EX57 1B pilot base was retired before its branch passed")
        rc = run_logged(
            long_worker_command(
                args, snapshot, selected_contract, "pilot_branch_4", branch,
                seed=5701, source=base_manifest["checkpoint"],
            ),
            env,
            branch / "worker.log",
        )
        if rc:
            raise RuntimeError(f"EX57 1B branch pilot failed rc={rc}")
    if not phase_valid(
        branch, contract_sha, 5701, data1b_sha,
        selection_sha=selection_sha, contract=contract,
        full_checkpoint_hash=True,
    ):
        raise RuntimeError("EX57 1B pilot branch failed state/lineage/hash audit")
    if P.read_json(branch / "summary.json").get("source_checkpoint_sha256") != base_manifest["checkpoint"]["sha256"]:
        raise RuntimeError("EX57 1B pilot branch did not originate at the certified fork")
    retired = []
    for directory in (base, branch):
        phase_manifest = P.read_json(directory / "phase_manifest.json")
        complete_checkpoint_retirement(
            directory, phase_manifest["checkpoint"], ["engineering_pilot_complete"]
        )
        retired.append(phase_manifest["checkpoint"])
    payload = {
        "schema_version": "ex57_engineering_pilot_v1", "passed": True,
        "quality_eligible": False, "timing_eligible": False,
        "selection_sha256": selection_sha, "selected_contract_sha256": contract_sha,
        "124m": cert124, "1b_retired": retired,
    }
    P.atomic_json(manifest_path, payload)
    if not engineering_pilot_manifest_valid(
        args, contract, selection_sha, selected_contract_sha
    ):
        raise RuntimeError("EX57 engineering pilot manifest failed independent replay")
    return payload


def tuning_manifest_valid(
    args: argparse.Namespace, snapshot: Path, contract: dict[str, Any]
) -> bool:
    try:
        manifest = P.read_json(args.run_dir / "tuning/tuning_manifest.json")
        if manifest.get("passed") is not True:
            return False
        selection_path = Path(manifest["selection_path"])
        selected_contract_path = Path(manifest["selected_contract_path"])
        if (
            P.sha256_file(selection_path) != manifest["selection_sha256"]
            or P.sha256_file(selected_contract_path)
            != manifest["selected_contract_sha256"]
        ):
            return False
        selection = P.read_json(selection_path)
        selected_contract = P.read_json(selected_contract_path)
        P.assert_contract(selected_contract)
        if (
            selection.get("scales") != manifest.get("selected")
            or selected_contract.get("selection_manifest_sha256")
            != manifest["selection_sha256"]
            or selected_contract.get("selected_configs")
            != {
                scale: manifest["selected"][scale]["selected_cell"]
                for scale in ("124m", "1b")
            }
        ):
            return False
        pilot_path = args.run_dir / "tuning/engineering_pilot_manifest.json"
        pilot = P.read_json(pilot_path)
        rng_receipt = args.run_dir / "rng_resume_compatibility_amendment_v1.json"
        profile_receipt = (
            args.run_dir / "long_worker_contract_compatibility_amendment_v2.json"
        )
        if not (
            pilot.get("passed") is True
            and pilot.get("selection_sha256") == manifest["selection_sha256"]
            and pilot.get("selected_contract_sha256")
            == manifest["selected_contract_sha256"]
            and P.sha256_file(pilot_path) == manifest["engineering_pilot_sha256"]
            and engineering_pilot_manifest_valid(
                args, contract, manifest["selection_sha256"],
                manifest["selected_contract_sha256"],
            )
            and rng_receipt.is_file()
            and P.sha256_file(rng_receipt)
            == manifest.get("rng_resume_compatibility_amendment_sha256")
            and P.read_json(rng_receipt).get("passed") is True
            and profile_receipt.is_file()
            and P.sha256_file(profile_receipt)
            == manifest.get(
                "long_worker_profile_compatibility_amendment_sha256"
            )
            and P.read_json(profile_receipt).get("passed") is True
        ):
            return False
        original_contract_sha = P.sha256_file(snapshot / PACKAGE_REL / CONTRACT_NAME)
        for scale in ("124m", "1b"):
            seed = int(contract["tuning"][scale]["seed"])
            steps = int(contract["tuning"][scale]["updates"])
            marker = "tuning_unselected_" + original_contract_sha
            data_sha = P.read_json(
                args.run_dir / "preflight" / f"data_{scale}.json"
            )["inventory_sha256"]
            for cell in contract["tuning"][scale]["cells"]:
                attempts = sorted(
                    (args.run_dir / "tuning" / scale / cell["id"]).glob("attempt_*")
                )
                if not any(
                    tuning_cell_valid(
                        attempt, scale=scale, cell=cell, seed=seed, steps=steps,
                        contract=contract, contract_sha=original_contract_sha,
                        selection_marker=marker, data_inventory_sha=data_sha,
                    )
                    for attempt in attempts
                ):
                    return False
        return True
    except Exception:
        return False


def tuning(args: argparse.Namespace) -> None:
    pre = require_preflight_context(args)
    if pre.get("passed") is not True:
        raise RuntimeError("EX57 tuning requires passed preflight")
    snapshot = args.run_dir / "source_snapshot"
    runtime_direct_trainer(args, snapshot)
    contract = P.read_json(snapshot / PACKAGE_REL / CONTRACT_NAME)
    final_manifest = args.run_dir / "tuning/tuning_manifest.json"
    if final_manifest.is_file() and tuning_manifest_valid(args, snapshot, contract):
        print(f"skip passed EX57 tuning: {args.run_dir}")
        return
    selected: dict[str, Any] = {}
    for scale in ("124m", "1b"):
        cells = contract["tuning"][scale]["cells"]
        rows = schedule(cells, args.gpus, lambda cell, gpu, s=scale: run_tuning_cell(args, snapshot, contract, s, cell, gpu))
        selected[scale] = select_cell(contract, scale, rows)
    selection_path = args.run_dir / "tuning/selection.json"
    selection_payload = {
        "schema_version": "ex57_scale_specific_selection_v1", "created_at": now_iso(),
        "tuning_only": True, "formal_seed_overlap": False,
        "rule": contract["tuning"]["winner_rule"], "scales": selected,
    }
    if selection_path.is_file():
        existing_selection = P.read_json(selection_path)
        stable_checks = {
            "schema": existing_selection.get("schema_version")
            == selection_payload["schema_version"],
            "tuning_only": existing_selection.get("tuning_only") is True,
            "formal_seed_overlap": existing_selection.get("formal_seed_overlap") is False,
            "rule": existing_selection.get("rule") == selection_payload["rule"],
            "scales": existing_selection.get("scales") == selected,
        }
        if not all(stable_checks.values()):
            raise RuntimeError(
                "EX57 frozen tuning selection changed across resume: "
                f"{stable_checks}"
            )
    else:
        P.atomic_json(selection_path, selection_payload)
    selection_sha = P.sha256_file(selection_path)
    selected_contract = write_selected_contract(args.run_dir, snapshot, contract, selection_sha, selected)
    pilot = run_engineering_pilots(args, snapshot, contract, selected, selection_sha, selected_contract)
    runtime_long_worker_adapter(args, snapshot)
    payload = {
        "schema_version": "ex57_tuning_manifest_v1", "passed": pilot.get("passed") is True,
        "selection_path": str(selection_path), "selection_sha256": selection_sha,
        "selected_contract_path": str(selected_contract), "selected_contract_sha256": P.sha256_file(selected_contract),
        "selected": selected, "engineering_pilot_sha256": P.sha256_file(args.run_dir / "tuning/engineering_pilot_manifest.json"),
        "rng_resume_compatibility_amendment_sha256": P.sha256_file(
            args.run_dir / "rng_resume_compatibility_amendment_v1.json"
        ),
        "long_worker_profile_compatibility_amendment_sha256": P.sha256_file(
            args.run_dir / "long_worker_contract_compatibility_amendment_v2.json"
        ),
        "formal_outcomes_observed": False, "timing_eligible": False,
    }
    P.atomic_json(final_manifest, payload)
    P.atomic_json(args.run_dir / "status.json", {"status": "tuning_passed", "updated_at": now_iso()})
    print(f"EX57 tuning passed. Artifacts: {args.run_dir}")


def local_unit_valid(
    path: Path,
    contract: dict[str, Any],
    selection_sha: str,
    selected_contract_sha: str,
    data_inventory_sha: str,
    seed: int,
) -> bool:
    try:
        manifest = P.read_json(path / "unit_manifest.json")
        summary = validate_local_summary(
            path / "summary.json",
            scale="124m",
            seed=seed,
            steps=6200,
            contract=contract,
            contract_sha=selected_contract_sha,
            selection_sha=selection_sha,
            data_inventory_sha=data_inventory_sha,
        )
        return (
            manifest.get("passed") is True and int(manifest.get("seed", -1)) == seed
            and manifest.get("selection_sha256") == selection_sha
            and manifest.get("selected_contract_sha256") == selected_contract_sha
            and manifest.get("data_inventory_sha256") == data_inventory_sha
            and P.sha256_file(path / "summary.json") == manifest.get("summary_sha256")
            and P.sha256_file(path / "metrics.csv") == manifest.get("metrics_sha256")
            and (path / "checkpoint_latest.pt").stat().st_size == int(manifest.get("checkpoint_bytes", -1))
            and P.sha256_file(path / "checkpoint_latest.pt") == manifest.get("checkpoint_sha256")
            and float(manifest.get("final_val_loss")) == float(summary["final_val_loss"])
        )
    except Exception:
        return False


def run_formal_124(args: argparse.Namespace, snapshot: Path, contract: dict[str, Any], selection: dict[str, Any], selection_sha: str, selected_contract: Path, seed: int, gpu: int) -> dict[str, Any]:
    unit = args.run_dir / "formal/124m" / f"seed{seed}"
    config = selection["124m"]["selected_cell"]
    contract_sha = P.sha256_file(selected_contract)
    data_inventory_sha = P.read_json(args.run_dir / "preflight/data_124m.json")[
        "inventory_sha256"
    ]
    if local_unit_valid(
        unit,
        contract,
        selection_sha,
        contract_sha,
        data_inventory_sha,
        seed,
    ):
        return P.read_json(unit / "unit_manifest.json")
    unit.mkdir(parents=True, exist_ok=True)
    command = direct_command(args, snapshot, "124m", seed, 6200, unit, config, warmdown=1800, checkpoint_every=128, no_save_final=False)
    rc = run_logged(command, direct_env(args, snapshot, "124m", gpu, config, contract_sha, selection_sha), unit / "worker.log")
    if rc:
        raise RuntimeError(f"EX57 formal 124M seed{seed} failed rc={rc}")
    summary = validate_local_summary(
        unit / "summary.json",
        scale="124m",
        seed=seed,
        steps=6200,
        contract=contract,
        contract_sha=contract_sha,
        selection_sha=selection_sha,
        data_inventory_sha=data_inventory_sha,
    )
    checkpoint = unit / "checkpoint_latest.pt"
    payload = {
        "schema_version": P.UNIT_MANIFEST_SCHEMA, "passed": True, "scale": "124m", "seed": seed,
        "selection_sha256": selection_sha, "selected_contract_sha256": contract_sha,
        "data_inventory_sha256": data_inventory_sha,
        "summary_sha256": P.sha256_file(unit / "summary.json"), "metrics_sha256": P.sha256_file(unit / "metrics.csv"),
        "checkpoint_sha256": P.sha256_file(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "final_val_loss": summary["final_val_loss"], "optimizer_state_bytes": summary["optimizer_state_bytes"],
        "peak_allocated_bytes": summary["peak_allocated_bytes"], "timing_eligible": False,
    }
    P.atomic_json(unit / "unit_manifest.json", payload)
    return payload


def retirement_valid(directory: Path, checkpoint: dict[str, Any]) -> bool:
    path = directory / "checkpoint_retirement.json"
    if not path.is_file():
        return False
    row = P.read_json(path)
    return (
        row.get("passed") is True
        and row.get("path") == checkpoint.get("path")
        and row.get("sha256") == checkpoint.get("sha256")
        and int(row.get("bytes", -1)) == int(checkpoint.get("bytes", -2))
        and isinstance(row.get("children"), list)
    )


def complete_checkpoint_retirement(
    directory: Path, checkpoint: dict[str, Any], children: list[str]
) -> dict[str, Any]:
    """Certificate first, then idempotently remove one large checkpoint."""
    path = Path(checkpoint["path"])
    retirement = directory / "checkpoint_retirement.json"
    if retirement.is_file():
        payload = P.read_json(retirement)
        checks = {
            "passed": payload.get("passed") is True,
            "path": payload.get("path") == str(path),
            "bytes": int(payload.get("bytes", -1)) == int(checkpoint["bytes"]),
            "sha256": payload.get("sha256") == checkpoint["sha256"],
            "children": payload.get("children") == children,
        }
        if not all(checks.values()):
            raise RuntimeError(f"changed EX57 retirement certificate: {checks}")
    else:
        if not path.is_file():
            raise RuntimeError(
                f"EX57 checkpoint disappeared before its retirement certificate: {path}"
            )
        if (
            path.stat().st_size != int(checkpoint["bytes"])
            or P.sha256_file(path) != checkpoint["sha256"]
        ):
            raise RuntimeError(f"cannot retire changed EX57 checkpoint: {path}")
        payload = {
            "schema_version": "ex57_checkpoint_retirement_v1",
            "passed": True,
            "path": str(path),
            "bytes": int(checkpoint["bytes"]),
            "sha256": checkpoint["sha256"],
            "children": children,
            "prepared_at": now_iso(),
        }
        P.atomic_json(retirement, payload)
    if path.is_file():
        if (
            path.stat().st_size != int(checkpoint["bytes"])
            or P.sha256_file(path) != checkpoint["sha256"]
        ):
            raise RuntimeError(f"changed EX57 checkpoint after certification: {path}")
        path.unlink()
    return payload


def phase_valid(
    directory: Path,
    contract_sha: str,
    seed: int,
    data_inventory_sha: str | None = None,
    *,
    selection_sha: str | None = None,
    expected_init_sha: str | None = None,
    contract: dict[str, Any] | None = None,
    full_checkpoint_hash: bool = False,
) -> bool:
    try:
        manifest = P.read_json(directory / "phase_manifest.json")
        summary_path = directory / "summary.json"
        summary_payload = P.read_json(summary_path)
        metrics = directory / "metrics.csv"
        checkpoint = manifest["checkpoint"]
        moonlight_schema = summary_payload.get("moonlight_state_schema")
        matrix_state = int(summary_payload.get("moonlight_matrix_optimizer_state_bytes", 0))
        checks = [
            manifest.get("passed") is True,
            manifest.get("method") == "moonlight",
            int(manifest.get("seed", -1)) == seed,
            manifest.get("contract_sha256") == contract_sha,
            P.sha256_file(summary_path) == manifest.get("summary_sha256"),
            P.sha256_file(metrics) == manifest.get("metrics_sha256"),
            summary_payload.get("status") == "completed",
            summary_payload.get("method") == "moonlight",
            int(summary_payload.get("seed", -1)) == seed,
            summary_payload.get("timing_comparable") is False,
            int(summary_payload.get("k_state_bytes", -1)) == 0,
            isinstance(moonlight_schema, dict)
            and moonlight_schema.get("contains_activation_k_state") is False
            and moonlight_schema.get("contains_factor_or_eigendecomposition_state") is False
            and moonlight_schema.get("tensor_state_keys") == ["momentum_buffer"],
            matrix_state > 0,
            int(summary_payload.get("optimizer_state_bytes", 0)) >= matrix_state,
            int(summary_payload.get("peak_allocated_bytes", 0)) > 0,
            all(P.finite_number(summary_payload.get(key)) for key in ("final_val_loss", "final_train_loss")),
        ]
        if data_inventory_sha is not None:
            checks.append(
                manifest.get("data_inventory_sha256") == data_inventory_sha
                and summary_payload.get("data_inventory_sha256") == data_inventory_sha
            )
        checks.append(summary_payload.get("contract_sha256") == contract_sha)
        if selection_sha is not None:
            checks.append(summary_payload.get("moonlight_selection_sha256") == selection_sha)
        if expected_init_sha is not None:
            checks.append(summary_payload.get("init_sha256") == expected_init_sha)
        if contract is not None:
            runtime_checks = moonlight_runtime_checks(summary_payload, contract, "1b")
            checks.extend(runtime_checks.values())
        checkpoint_path = Path(checkpoint["path"])
        if checkpoint_path.is_file():
            checks.append(checkpoint_path.stat().st_size == int(checkpoint["bytes"]))
            if full_checkpoint_hash:
                checks.append(P.sha256_file(checkpoint_path) == checkpoint["sha256"])
        else:
            checks.append(
                manifest.get("role") in ("fork_source", "engineering_pilot")
                and retirement_valid(directory, checkpoint)
            )
        return all(checks)
    except Exception:
        return False

def retire_phase(
    directory: Path,
    children: list[str],
    unit: Path,
    data_inventory_sha: str,
    selection_sha: str,
    expected_init_sha: str,
    contract: dict[str, Any],
) -> None:
    manifest = P.read_json(directory / "phase_manifest.json")
    checkpoint = manifest["checkpoint"]
    path = Path(checkpoint["path"])
    if retirement_valid(directory, checkpoint):
        complete_checkpoint_retirement(directory, checkpoint, children)
        return
    if not all(
        phase_valid(
            unit / child,
            manifest["contract_sha256"],
            int(manifest["seed"]),
            data_inventory_sha,
            selection_sha=selection_sha,
            expected_init_sha=expected_init_sha,
            contract=contract,
        )
        for child in children
    ):
        return
    complete_checkpoint_retirement(directory, checkpoint, children)


def long_unit_valid(
    unit: Path,
    contract: dict[str, Any],
    selection_sha: str,
    selected_contract_sha: str,
    data_inventory_sha: str,
    seed: int,
) -> bool:
    try:
        manifest = P.read_json(unit / "unit_manifest.json")
        checks = [
            manifest.get("passed") is True,
            manifest.get("scale") == "1b",
            int(manifest.get("seed", -1)) == seed,
            manifest.get("selection_sha256") == selection_sha,
            manifest.get("selected_contract_sha256") == selected_contract_sha,
            manifest.get("data_inventory_sha256") == data_inventory_sha,
        ]
        for phase in contract["phases"]:
            directory = unit / phase["id"]
            checks.append(
                phase_valid(
                    directory,
                    selected_contract_sha,
                    seed,
                    data_inventory_sha,
                    selection_sha=selection_sha,
                    expected_init_sha=contract["accepted_init_sha256"]["1b"][str(seed)],
                    contract=contract,
                    full_checkpoint_hash=phase["role"] == "primary_endpoint",
                )
            )
            if phase["role"] == "fork_source":
                checkpoint_row = P.read_json(
                    directory / "phase_manifest.json"
                )["checkpoint"]
                retirement_row = P.read_json(
                    directory / "checkpoint_retirement.json"
                )
                checks.append(
                    not Path(checkpoint_row["path"]).exists()
                    and retirement_row.get("passed") is True
                    and retirement_row.get("path") == checkpoint_row["path"]
                    and retirement_row.get("sha256") == checkpoint_row["sha256"]
                    and int(retirement_row.get("bytes", -1))
                    == int(checkpoint_row["bytes"])
                    and retirement_row.get("children")
                    == P.direct_children(contract, phase["id"])
                )
            phase_row = manifest.get("phases", {}).get(phase["id"], {})
            checks.extend(
                (
                    P.sha256_file(directory / "phase_manifest.json")
                    == phase_row.get("manifest_sha256"),
                    P.sha256_file(directory / "summary.json")
                    == phase_row.get("summary_sha256"),
                )
            )
        for phase in P.endpoint_phases(contract):
            expected = P.read_json(unit / phase["id"] / "phase_manifest.json")[
                "checkpoint"
            ]
            checks.append(
                manifest.get("endpoints", {}).get(phase["budget_id"]) == expected
            )
        return all(checks)
    except Exception:
        return False


def non10b_unit_valid(
    unit: Path,
    contract: dict[str, Any],
    selection_sha: str,
    contract_sha: str,
    data_inventory_sha: str,
    seed: int,
) -> bool:
    try:
        manifest = P.read_json(unit / "non10b_manifest.json")
        expected_init_sha = contract["accepted_init_sha256"]["1b"][str(seed)]
        checks = [
            manifest.get("passed") is True,
            manifest.get("stage") == "formal_non10b",
            manifest.get("scale") == "1b",
            int(manifest.get("seed", -1)) == seed,
            manifest.get("selection_sha256") == selection_sha,
            manifest.get("selected_contract_sha256") == contract_sha,
            manifest.get("data_inventory_sha256") == data_inventory_sha,
            tuple(manifest.get("completed_phase_ids", ())) == NON10B_PHASE_IDS,
        ]
        phases_by_id = {row["id"]: row for row in contract["phases"]}
        for phase_id in NON10B_PHASE_IDS:
            phase = phases_by_id[phase_id]
            directory = unit / phase_id
            checks.append(
                phase_valid(
                    directory, contract_sha, seed, data_inventory_sha,
                    selection_sha=selection_sha,
                    expected_init_sha=expected_init_sha,
                    contract=contract,
                    full_checkpoint_hash=phase["role"] == "primary_endpoint",
                )
            )
            phase_row = manifest.get("phases", {}).get(phase_id, {})
            checks.extend((
                P.sha256_file(directory / "phase_manifest.json")
                == phase_row.get("manifest_sha256"),
                P.sha256_file(directory / "summary.json")
                == phase_row.get("summary_sha256"),
            ))
        # EX57 has no continuation experiment. Both fork-only checkpoints are
        # retired once their in-scope children are certified.
        for fork_id in ("backbone_4400", "backbone_11493"):
            checkpoint = P.read_json(unit / fork_id / "phase_manifest.json")["checkpoint"]
            checks.append(
                not Path(checkpoint["path"]).exists()
                and retirement_valid(unit / fork_id, checkpoint)
            )
        expected_budgets = {"tokens_3p2506b", "tokens_6p9694b"}
        checks.append(set(manifest.get("endpoints", {})) == expected_budgets)
        for phase_id in ("cooldown_6200", "cooldown_13293"):
            phase = phases_by_id[phase_id]
            expected = P.read_json(unit / phase_id / "phase_manifest.json")["checkpoint"]
            checks.append(manifest["endpoints"].get(phase["budget_id"]) == expected)
        return all(checks)
    except Exception:
        return False


def run_formal_1b(
    args: argparse.Namespace,
    snapshot: Path,
    contract: dict[str, Any],
    selection: dict[str, Any],
    selection_sha: str,
    selected_contract: Path,
    seed: int,
    gpu: int,
    *,
    phase_ids: tuple[str, ...] | None = None,
    finalize: bool = True,
) -> dict[str, Any]:
    unit = args.run_dir / "formal/1b" / f"seed{seed}"
    unit_manifest = unit / "unit_manifest.json"
    contract_sha = P.sha256_file(selected_contract)
    data_inventory_sha = P.read_json(args.run_dir / "preflight/data_1b.json")[
        "inventory_sha256"
    ]
    if finalize and long_unit_valid(
        unit,
        contract,
        selection_sha,
        contract_sha,
        data_inventory_sha,
        seed,
    ):
        return P.read_json(unit_manifest)
    unit.mkdir(parents=True, exist_ok=True)
    config = selection["1b"]["selected_cell"]
    env = worker_env(snapshot, args.official_repo, gpu, config, contract_sha, selection_sha)
    bind_gpu_compile_cache(env, args.run_dir, gpu)
    env["MOONLIGHT_DATA_INVENTORY_SHA256"] = data_inventory_sha
    all_phases = contract["phases"]
    selected_phase_ids = tuple(row["id"] for row in all_phases) if phase_ids is None else phase_ids
    phases_by_id = {row["id"]: row for row in all_phases}
    phases = [phases_by_id[phase_id] for phase_id in selected_phase_ids]
    if not finalize and non10b_unit_valid(
        unit, contract, selection_sha, contract_sha, data_inventory_sha, seed
    ):
        return P.read_json(unit / "non10b_manifest.json")
    phase_rows: dict[str, Any] = {}
    for phase in phases:
        phase_dir = unit / phase["id"]
        expected_init_sha = contract["accepted_init_sha256"]["1b"][str(seed)]
        if not phase_valid(
            phase_dir, contract_sha, seed, data_inventory_sha,
            selection_sha=selection_sha, expected_init_sha=expected_init_sha,
            contract=contract,
        ):
            source = None
            if phase.get("parent") and not (phase_dir / "checkpoint_latest.pt").is_file():
                parent = P.read_json(unit / phase["parent"] / "phase_manifest.json")
                source = parent["checkpoint"]
            command = long_worker_command(args, snapshot, selected_contract, phase["id"], phase_dir, seed=seed, source=source)
            rc = run_logged(command, env, phase_dir / "worker.log")
            if rc or not phase_valid(
                phase_dir, contract_sha, seed, data_inventory_sha,
                selection_sha=selection_sha, expected_init_sha=expected_init_sha,
                contract=contract,
            ):
                raise RuntimeError(f"EX57 formal 1B seed{seed}/{phase['id']} failed rc={rc}")
        phase_rows[phase["id"]] = {"manifest_sha256": P.sha256_file(phase_dir / "phase_manifest.json"), "summary_sha256": P.sha256_file(phase_dir / "summary.json")}
        for candidate in all_phases:
            if candidate["role"] != "fork_source":
                continue
            children = P.direct_children(contract, candidate["id"])
            if children and all(
                phase_valid(
                    unit / child, contract_sha, seed, data_inventory_sha,
                    selection_sha=selection_sha, expected_init_sha=expected_init_sha,
                    contract=contract,
                )
                for child in children
            ):
                retire_phase(
                    unit / candidate["id"], children, unit, data_inventory_sha,
                    selection_sha, expected_init_sha, contract,
                )
    if finalize:
        # The final unit certificate seals the complete phase graph, including
        # the phases produced by formal_non10b in an earlier invocation.
        phase_rows = {
            phase["id"]: {
                "manifest_sha256": P.sha256_file(
                    unit / phase["id"] / "phase_manifest.json"
                ),
                "summary_sha256": P.sha256_file(unit / phase["id"] / "summary.json"),
            }
            for phase in all_phases
        }
    endpoints = {}
    endpoint_phases = [
        phase for phase in P.endpoint_phases(contract)
        if finalize or phase["id"] in selected_phase_ids
    ]
    for phase in endpoint_phases:
        manifest = P.read_json(unit / phase["id"] / "phase_manifest.json")
        checkpoint = Path(manifest["checkpoint"]["path"])
        if not checkpoint.is_file() or P.sha256_file(checkpoint) != manifest["checkpoint"]["sha256"]:
            raise RuntimeError(f"EX57 retained endpoint hash failed: {checkpoint}")
        endpoints[phase["budget_id"]] = manifest["checkpoint"]
    payload = {
        "schema_version": P.UNIT_MANIFEST_SCHEMA if finalize else "ex57_non10b_unit_v1",
        "passed": True,
        "stage": "formal" if finalize else "formal_non10b",
        "scale": "1b", "seed": seed, "physical_gpu": int(gpu),
        "selection_sha256": selection_sha, "selected_contract_sha256": contract_sha,
        "data_inventory_sha256": data_inventory_sha,
        "completed_phase_ids": list(selected_phase_ids),
        "phases": phase_rows, "endpoints": endpoints, "timing_eligible": False,
    }
    P.atomic_json(unit_manifest if finalize else unit / "non10b_manifest.json", payload)
    return payload


def ensure_controller_amendment(args: argparse.Namespace) -> dict[str, Any] | None:
    # Fresh Moonlight EX57 runs are self-contained. There is no cross-experiment
    # handoff/controller amendment path.
    return None

def require_10b_runtime_admission(args: argparse.Namespace) -> dict[str, Any]:
    raise RuntimeError("EX57 Moonlight has no 10B continuation; Experiment 57 is independent")

def formal_context(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], dict[str, Any], str, Path]:
    require_preflight_context(args)
    tune = P.read_json(args.run_dir / "tuning/tuning_manifest.json")
    snapshot = args.run_dir / "source_snapshot"
    contract = P.read_json(snapshot / PACKAGE_REL / CONTRACT_NAME)
    if tune.get("passed") is not True or not tuning_manifest_valid(
        args, snapshot, contract
    ):
        raise RuntimeError("EX57 formal requires passed tuning")
    selection_path = Path(tune["selection_path"])
    if P.sha256_file(selection_path) != tune["selection_sha256"]:
        raise RuntimeError("EX57 selection manifest changed")
    selected_contract = Path(tune["selected_contract_path"])
    if P.sha256_file(selected_contract) != tune["selected_contract_sha256"]:
        raise RuntimeError("EX57 selected contract changed")
    selection = P.read_json(selection_path)["scales"]
    selection_sha = tune["selection_sha256"]
    return snapshot, contract, selection, selection_sha, selected_contract


def formal_non10b(args: argparse.Namespace) -> None:
    snapshot, contract, selection, selection_sha, selected_contract = formal_context(args)
    tasks = [("124m", int(seed)) for seed in contract["formal"]["seeds"]] + [
        ("1b", int(seed)) for seed in contract["formal"]["seeds"]
    ]
    def run(item: tuple[str, int], gpu: int) -> dict[str, Any]:
        scale, seed = item
        if scale == "124m":
            return run_formal_124(args, snapshot, contract, selection, selection_sha, selected_contract, seed, gpu)
        return run_formal_1b(
            args, snapshot, contract, selection, selection_sha, selected_contract,
            seed, gpu, phase_ids=NON10B_PHASE_IDS, finalize=False,
        )
    rows = schedule(tasks, args.gpus, run)
    payload = {
        "schema_version": "ex57_moonlight_formal_manifest_v1",
        "passed": len(rows) == 6 and all(row.get("passed") is True for row in rows),
        "units": sorted(rows, key=lambda row: (row["scale"], row["seed"])),
        "selection_sha256": selection_sha,
        "selected_contract_sha256": P.sha256_file(selected_contract),
        "completed_1b_budget_ids": ["tokens_3p2506b", "tokens_6p9694b"],
        "experiment_scope_complete": True,
        "independent_of_ex57": True,
        "timing_eligible": False,
    }
    # Keep the historical filename so the mature non-10B analyzer remains
    # reusable, but this is EX57's final scientific boundary, not a handoff.
    P.atomic_json(args.run_dir / "formal/formal_non10b_manifest.json", payload)
    if not payload["passed"]:
        raise RuntimeError("EX57 Moonlight formal stage incomplete")
    P.atomic_json(args.run_dir / "status.json", {"status": "formal_passed", "updated_at": now_iso()})
    print(f"EX57 Moonlight formal stage passed. Artifacts: {args.run_dir}")

def formal_10b(args: argparse.Namespace) -> None:
    raise RuntimeError("EX57 Moonlight ends at 6.97B; EX57 is a separate full experiment")

def formal(args: argparse.Namespace) -> None:
    formal_non10b(args)


def verify(args: argparse.Namespace) -> None:
    require_preflight_context(args)
    formal_path = args.run_dir / "formal/formal_non10b_manifest.json"
    if not formal_path.is_file() or P.read_json(formal_path).get("passed") is not True:
        raise RuntimeError("EX57 verify requires a passed formal stage")
    analyzer = Path(__file__).resolve().parent / "analyze.py"
    completed = subprocess.run(
        [sys.executable, str(analyzer), "build", "--run-dir", str(args.run_dir), "--scope", "non10b"],
        text=True, capture_output=True,
    )
    print(completed.stdout, end="")
    if completed.returncode:
        raise RuntimeError(f"EX57 analysis failed: {completed.stdout}\n{completed.stderr}")
    audit = subprocess.run(
        [sys.executable, str(analyzer), "verify", "--run-dir", str(args.run_dir),
         "--scope", "non10b", "--full-checkpoint-hash"],
        text=True, capture_output=True,
    )
    print(audit.stdout, end="")
    if audit.returncode:
        raise RuntimeError(f"EX57 verification failed: {audit.stdout}\n{audit.stderr}")
    analysis = P.read_json(args.run_dir / "analysis/analysis_manifest.json")
    verification_path = args.run_dir / "analysis/verification_manifest.json"
    verification = P.read_json(verification_path)
    if not (verification.get("passed") is True and verification.get("full_checkpoint_hash") is True):
        raise RuntimeError("EX57 full native verification receipt is missing")
    payload = {
        "schema_version": "ex57_moonlight_completion_v1",
        "status": "completed",
        "passed": analysis.get("passed") is True,
        "formal_units": 6,
        "analysis_manifest": str(args.run_dir / "analysis/analysis_manifest.json"),
        "analysis_manifest_sha256": P.sha256_file(args.run_dir / "analysis/analysis_manifest.json"),
        "verification_manifest": str(verification_path),
        "verification_manifest_sha256": P.sha256_file(verification_path),
        "formal_manifest_sha256": P.sha256_file(formal_path),
        "scope": "124m_and_1b_through_6p97b",
        "independent_of_ex57": True,
        "full_checkpoint_hash_verified": True,
        "timing_usable": False,
        "wandb_required": False,
    }
    P.atomic_json(args.run_dir / "completion_manifest.json", payload)
    P.atomic_json(args.run_dir / "status.json", {"status": "completed", "updated_at": now_iso()})
    print(f"EX57 Moonlight suite completed. Artifacts: {args.run_dir}")

def upload(args: argparse.Namespace) -> None:
    snapshot = args.run_dir / "source_snapshot"
    uploader = snapshot / PACKAGE_REL / "upload_wandb.py"
    command = [sys.executable, str(uploader), "--run-dir", str(args.run_dir), "--mode", args.wandb_mode, "--project", args.wandb_project]
    if args.wandb_entity:
        command += ["--entity", args.wandb_entity]
    completed = subprocess.run(command)
    if completed.returncode:
        raise RuntimeError(f"EX57 W&B upload failed rc={completed.returncode}")


def check(args: argparse.Namespace) -> None:
    payload = check_sources(args.repo.resolve())
    payload["plan"] = {
        "tuning_units": 6,
        "formal_units": 6,
        "formal_124m_runs": 3,
        "formal_1b_long_runs": 3,
        "formal_seeds": [2024, 2025, 2026],
        "formal_stages": ["formal"],
        "formal_1b_budget_ids": ["tokens_3p2506b", "tokens_6p9694b"],
        "independent_of_ex57": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    if args.stage == "check":
        check(args)
        return
    args.run_dir = args.run_dir.resolve()
    args.official_repo = args.official_repo.resolve()
    args.data124_dir = args.data124_dir.resolve()
    args.data1b_dir = args.data1b_dir.resolve()
    if args.stage == "preflight":
        preflight(args)
    elif args.stage == "tuning":
        tuning(args)
    elif args.stage == "formal_non10b":
        formal_non10b(args)
    elif args.stage == "formal":
        formal(args)
    elif args.stage == "verify":
        verify(args)
    elif args.stage == "upload":
        upload(args)
    elif args.stage in ("all", "resume"):
        preflight(args)
        tuning(args)
        formal_non10b(args)
        verify(args)
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"EX57 stopped cleanly: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
