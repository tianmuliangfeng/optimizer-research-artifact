#!/usr/bin/env python3
"""Controller for the four-method LLaMA-1B isolated efficiency audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import efficiency_common as common
import source_builder


SCRIPT_VERSION = "2026-07-29.1"
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONTRACT = HERE / "efficiency_contract.json"
CERTIFIER = HERE / "certify_exclusive_node.py"
SNAPSHOT_FILES = (
    "efficiency_common.py",
    "source_builder.py",
    "llama1b_efficiency_worker.py",
    "gpu_isolation_monitor.py",
    "run_llama1b_efficiency.py",
    "analyze_llama1b_efficiency.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--physical-gpu", default="0")
    parser.add_argument("--required-gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--host-id", default="llama-host-h100")
    parser.add_argument(
        "--execution-domain", default="llama-host-llama1b-isolated-efficiency"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_contract(contract: dict[str, Any]) -> None:
    methods = contract["method_order"]
    execution = contract["execution_policy"]
    frozen = contract["frozen_configuration"]
    failures: list[str] = []
    if methods != ["muon", "newton_full", "down_none", "down_diag"]:
        failures.append("method order differs from the preregistration")
    orders = execution["orders"]
    expected_orders = [
        ["muon", "newton_full", "down_none", "down_diag"],
        ["newton_full", "down_none", "down_diag", "muon"],
        ["down_none", "down_diag", "muon", "newton_full"],
        ["down_diag", "muon", "newton_full", "down_none"],
    ]
    repeats = int(execution["repeats"])
    if repeats != 4 or len(orders) != repeats:
        failures.append("expected four order rotations")
    if orders != expected_orders:
        failures.append("execution orders differ from the preregistered cyclic rotations")
    if any(sorted(order) != sorted(methods) for order in orders):
        failures.append("an order is not a permutation of the four methods")
    for method in methods:
        positions = [
            position
            for order in orders
            for position, observed in enumerate(order)
            if observed == method
        ]
        if sorted(positions) != [0, 1, 2, 3]:
            failures.append(f"{method} does not occupy each position once")
    if int(frozen["total_updates"]) != int(
        frozen["warmup_updates_excluded"]
    ) + int(frozen["timed_updates"]):
        failures.append("warmup + timed update arithmetic differs")
    if int(frozen["tokens_per_update"]) != int(
        frozen["global_batch_size"]
    ) * int(frozen["sequence_length"]):
        failures.append("tokens per update arithmetic differs")
    if int(frozen["validation_tokens"]) != int(
        frozen["device_batch_size"]
    ) * int(frozen["sequence_length"]):
        failures.append("validation is not the preregistered one-microbatch gate")
    if int(frozen["checkpoint_every"]) != 0:
        failures.append("checkpoint I/O is enabled")
    if frozen["resume_policy"] != "never" or frozen["wandb_policy"] != (
        "disabled_for_timing"
    ):
        failures.append("resume or W&B timing policy differs")
    source_contract = contract["source_contract"]
    if source_contract["base_trainer_sha256"] != source_builder.PINNED_BASE_SHA256:
        failures.append("base trainer source pin differs")
    if (
        source_contract["profile_wrapper_sha256"]
        != source_builder.PINNED_WRAPPER_SHA256
    ):
        failures.append("profile wrapper source pin differs")
    if (
        source_contract["derived_efficiency_base_sha256"]
        != source_builder.PINNED_DERIVED_SHA256
    ):
        failures.append("derived efficiency trainer source pin differs")
    if execution["physical_timing_gpu"] not in execution[
        "required_idle_physical_gpus"
    ]:
        failures.append("timing GPU is absent from required node GPUs")
    primary = {
        (row["candidate"], row["reference"])
        for row in contract["primary_contrasts"]
    }
    expected_primary = {
        ("down_none", "muon"),
        ("down_none", "newton_full"),
        ("down_diag", "muon"),
        ("down_diag", "newton_full"),
        ("newton_full", "muon"),
    }
    if primary != expected_primary:
        failures.append("primary contrasts differ from the frozen family comparisons")
    if failures:
        raise RuntimeError("efficiency contract validation failed:\n- " + "\n- ".join(failures))


def append_command(run_dir: Path, label: str, command: list[str]) -> None:
    with (run_dir / "commands.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "created_at": common.now_iso(),
                    "label": label,
                    "command": command,
                },
                sort_keys=True,
            )
            + "\n"
        )


def run_command(
    run_dir: Path,
    label: str,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    append_command(run_dir, label, command)
    return subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
        check=check,
    )


def csv_rows(text: str) -> list[list[str]]:
    if not text.strip():
        return []
    return [
        [value.strip() for value in row]
        for row in csv.reader(StringIO(text))
        if any(value.strip() for value in row)
    ]


def gpu_inventory() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"nvidia-smi inventory failed: {result.stdout}")
    gpus = [
        {
            "index": int(row[0]),
            "uuid": row[1],
            "name": row[2],
            "memory_total_mib": int(row[3]),
            "driver_version": row[4],
        }
        for row in csv_rows(result.stdout)
    ]
    return {"gpus": gpus, "fingerprint": common.canonical_json_sha256(gpus)}


def validate_gpu_inventory(
    inventory: dict[str, Any], required_gpus: list[str], contract: dict[str, Any]
) -> None:
    required = {int(index) for index in required_gpus}
    observed = {int(gpu["index"]) for gpu in inventory["gpus"]}
    required_name = contract["execution_policy"]["required_gpu_name"]
    if observed != required:
        raise RuntimeError(
            f"isolated node must expose exactly GPUs {sorted(required)}; observed {sorted(observed)}"
        )
    if any(gpu["name"] != required_name for gpu in inventory["gpus"]):
        raise RuntimeError(f"GPU model differs from contract: {inventory['gpus']}")


def runtime_probe(
    python_exe: Path, official_repo: Path, physical_gpu: str
) -> dict[str, Any]:
    probe = r"""
import json, pathlib, sys
import numpy, torch, triton, triton_kernels
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")
gpu = torch.cuda.get_device_properties(0)
payload = {
    "python_executable": str(pathlib.Path(sys.executable).absolute()),
    "python_version": list(sys.version_info[:3]),
    "python_full": sys.version.replace("\n", " "),
    "numpy": str(numpy.__version__),
    "torch": str(torch.__version__),
    "torch_cuda": torch.version.cuda,
    "triton": str(triton.__version__),
    "triton_module": str(pathlib.Path(triton.__file__).resolve()),
    "triton_kernels_module": str(pathlib.Path(triton_kernels.__file__).resolve()),
    "gpu_name": gpu.name,
    "gpu_total_memory_bytes": int(gpu.total_memory),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
}
print("RUNTIME_JSON " + json.dumps(payload, sort_keys=True))
"""
    env = common.subprocess_environment(
        official_repo.resolve(), physical_gpu=physical_gpu
    )
    result = subprocess.run(
        [str(common.lexical_absolute(python_exe)), "-c", probe],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"runtime probe failed:\n{result.stdout}")
    lines = [
        line for line in result.stdout.splitlines() if line.startswith("RUNTIME_JSON ")
    ]
    if len(lines) != 1:
        raise RuntimeError(f"runtime probe produced no unique payload:\n{result.stdout}")
    payload = json.loads(lines[0].split(" ", 1)[1])
    payload["triton_kernels_sha256"] = common.sha256_file(
        official_repo.resolve() / "triton_kernels.py"
    )
    return payload


def k_state_bytes(architecture: dict[str, Any]) -> int:
    total = 0
    for group in architecture["preconditioner_groups"]:
        width = int(group["input_width"])
        elements = width * width if group["kind"] == "dense" else width
        total += elements * 4 * 2
    return total


def run_init_audit(
    args: argparse.Namespace,
    run_dir: Path,
    contract: dict[str, Any],
    snapshot: Path,
    derived_sha: str,
) -> dict[str, Any]:
    observed: dict[str, dict[str, Any]] = {}
    for method in contract["method_order"]:
        output = run_dir / "_preflight_init" / method
        command = [
            str(common.lexical_absolute(args.python_exe)),
            str(snapshot / "train_llama_swiglu_1b.py"),
            "--method",
            method,
            "--data-dir",
            str(args.data_dir.resolve()),
            "--output-dir",
            str(output),
            "--seed",
            str(contract["frozen_configuration"]["seed"]),
            "--init-only",
        ]
        env = common.subprocess_environment(
            args.official_repo.resolve(),
            derived_base=snapshot / "train_llama_swiglu_efficiency_base.py",
            derived_base_sha256=derived_sha,
            physical_gpu=args.physical_gpu,
        )
        result = run_command(
            run_dir,
            f"preflight/init/{method}",
            command,
            env=env,
            check=False,
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError(
                f"initialization audit failed for {method}:\n{result.stdout}"
            )
        lines = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("LLAMA_INIT_AUDIT ")
        ]
        if len(lines) != 1:
            raise RuntimeError(f"no unique init audit for {method}:\n{result.stdout}")
        observed[method] = json.loads(lines[0].split(" ", 1)[1])
    fingerprints = {row["init_sha256"] for row in observed.values()}
    expected_init = contract["frozen_configuration"]["initialization_sha256"]
    if fingerprints != {expected_init}:
        raise RuntimeError(
            f"initialization fingerprint differs: {fingerprints} != {expected_init}"
        )
    expected_groups = {
        "muon": 0,
        "newton_full": 72,
        "down_none": 54,
        "down_diag": 72,
    }
    for method, row in observed.items():
        architecture = row["architecture"]
        checks = {
            "method": row["method"] == method,
            "seed": row["seed"] == int(contract["frozen_configuration"]["seed"]),
            "parameter_count": architecture["parameter_count"]
            == int(contract["frozen_configuration"]["parameter_count"]),
            "profile": architecture.get("profile", {}).get("name")
            == "llama_swiglu_1b_v1",
            "base_trainer": architecture.get("base_trainer_sha256") == derived_sha,
            "matrix_tensors": architecture["matrix_tensor_count"] == 126,
            "backup_tensors": architecture["backup_tensor_count"] == 38,
            "preconditioner_groups": architecture["preconditioner_group_count"]
            == expected_groups[method],
            "embedding_head_tied": architecture["embedding_head_tied"] is True,
            "bias_absent": architecture["bias_parameter_count"] == 0,
            "k_state_bytes": k_state_bytes(architecture)
            == int(contract["expected_k_state_bytes"][method]),
        }
        if not all(checks.values()):
            raise RuntimeError(f"initialization architecture failed for {method}: {checks}")
    return {
        "common_init_sha256": expected_init,
        "methods": observed,
        "fingerprint": common.canonical_json_sha256(observed),
    }


def snapshot_sources(run_dir: Path, contract_path: Path) -> tuple[Path, dict[str, Any]]:
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    bundle = source_builder.build_source_bundle()
    materialized = source_builder.materialize(bundle, snapshot)
    files: dict[str, str] = {}
    for name in SNAPSHOT_FILES:
        source = HERE / name
        if not source.is_file():
            raise FileNotFoundError(f"missing experiment-42 source: {source}")
        target = snapshot / name
        shutil.copy2(source, target)
        files[name] = common.sha256_file(target)
    shutil.copy2(CERTIFIER, snapshot / "certify_exclusive_node.py")
    files["certify_exclusive_node.py"] = common.sha256_file(
        snapshot / "certify_exclusive_node.py"
    )
    shutil.copy2(contract_path, snapshot / "efficiency_contract.json")
    files["efficiency_contract.json"] = common.sha256_file(
        snapshot / "efficiency_contract.json"
    )
    files.update(
        {
            "train_llama_swiglu_efficiency_base.py": materialized["derived_base"],
            "train_llama_swiglu_1b.py": materialized["profile_wrapper"],
            "train_llama_swiglu_efficiency_base.diff": materialized["source_diff"],
        }
    )
    audit = {
        "base_trainer_sha256": bundle.base_sha256,
        "profile_wrapper_source_sha256": bundle.wrapper_sha256,
        "derived_base_sha256": bundle.derived_sha256,
        "profile_wrapper_sha256": materialized["profile_wrapper"],
        "source_diff_sha256": materialized["source_diff"],
        "files": files,
    }
    return snapshot, audit


def verify_snapshot(snapshot: Path, source_audit: dict[str, Any]) -> None:
    failures: list[str] = []
    for name, expected in source_audit["files"].items():
        path = snapshot / name
        if not path.is_file() or common.sha256_file(path) != expected:
            failures.append(name)
    if failures:
        raise RuntimeError(f"source snapshot is incomplete or changed: {failures}")


def certify_node(
    run_dir: Path,
    snapshot: Path,
    output: Path,
    required_gpus: list[str],
    label: str,
) -> subprocess.CompletedProcess[str]:
    command = [
        str(common.lexical_absolute(Path(sys.executable))),
        str(snapshot / "certify_exclusive_node.py"),
        "--output",
        str(output),
        "--required-gpus",
        *required_gpus,
    ]
    return run_command(run_dir, label, command, check=False)


def build_preflight(
    args: argparse.Namespace,
    run_dir: Path,
    contract_path: Path,
    contract: dict[str, Any],
    snapshot: Path,
    source_audit: dict[str, Any],
) -> dict[str, Any]:
    certificate = run_dir / "preflight_exclusive_node.json"
    result = certify_node(
        run_dir,
        snapshot,
        certificate,
        args.required_gpus,
        "preflight/exclusive_node",
    )
    if result.returncode:
        raise RuntimeError(f"preflight exclusive-node certificate failed: {certificate}")
    official = common.audit_official_repo(args.official_repo, contract)
    data = common.audit_data(args.data_dir)
    inventory = gpu_inventory()
    validate_gpu_inventory(inventory, args.required_gpus, contract)
    runtime = runtime_probe(args.python_exe, args.official_repo, args.physical_gpu)
    stable = common.stable_runtime(runtime)
    runtime_fingerprint = common.canonical_json_sha256(stable)
    if runtime["gpu_name"] != contract["execution_policy"]["required_gpu_name"]:
        raise RuntimeError(f"runtime GPU differs: {runtime['gpu_name']}")
    if (
        runtime["triton_kernels_sha256"]
        != contract["source_contract"]["triton_kernels_sha256"]
    ):
        raise RuntimeError("runtime imported a different triton_kernels.py")
    init = run_init_audit(
        args,
        run_dir,
        contract,
        snapshot,
        source_audit["derived_base_sha256"],
    )
    preflight = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at": common.now_iso(),
        "passed": True,
        "contract_sha256": common.sha256_file(contract_path),
        "source_audit": source_audit,
        "official_repo_audit": official,
        "data_audit": data,
        "gpu_inventory": inventory,
        "runtime": runtime,
        "stable_runtime": stable,
        "runtime_fingerprint": runtime_fingerprint,
        "init_audit": init,
        "exclusive_node_certificate_sha256": common.sha256_file(certificate),
        "host_id": args.host_id,
        "execution_domain": args.execution_domain,
        "physical_timing_gpu": args.physical_gpu,
        "required_gpus": args.required_gpus,
    }
    common.atomic_write_json(run_dir / "preflight.json", preflight)
    return preflight


def revalidate_preflight(
    args: argparse.Namespace,
    run_dir: Path,
    contract: dict[str, Any],
    snapshot: Path,
    preflight: dict[str, Any],
) -> None:
    verify_snapshot(snapshot, preflight["source_audit"])
    current_sources = {
        name: common.sha256_file(HERE / name) for name in SNAPSHOT_FILES
    }
    snapshot_sources = {
        name: preflight["source_audit"]["files"][name] for name in SNAPSHOT_FILES
    }
    if current_sources != snapshot_sources:
        raise RuntimeError(
            "current experiment-42 controller sources differ from the immutable snapshot"
        )
    if common.sha256_file(CERTIFIER) != preflight["source_audit"]["files"][
        "certify_exclusive_node.py"
    ]:
        raise RuntimeError("exclusive-node certifier changed before resume")
    if common.sha256_file(run_dir / "efficiency_contract.json") != preflight[
        "contract_sha256"
    ]:
        raise RuntimeError("run contract changed before resume")
    official = common.audit_official_repo(args.official_repo, contract)
    if official["commit"] != preflight["official_repo_audit"]["commit"] or official[
        "triton_kernels_sha256"
    ] != preflight["official_repo_audit"]["triton_kernels_sha256"]:
        raise RuntimeError("official repository differs from immutable preflight")
    data = common.audit_data(args.data_dir)
    if data["fingerprint"] != preflight["data_audit"]["fingerprint"]:
        raise RuntimeError("FineWeb data differs from immutable preflight")
    inventory = gpu_inventory()
    validate_gpu_inventory(inventory, args.required_gpus, contract)
    if inventory["fingerprint"] != preflight["gpu_inventory"]["fingerprint"]:
        raise RuntimeError("physical GPU inventory differs from immutable preflight")
    runtime = runtime_probe(args.python_exe, args.official_repo, args.physical_gpu)
    if common.canonical_json_sha256(common.stable_runtime(runtime)) != preflight[
        "runtime_fingerprint"
    ]:
        raise RuntimeError("training runtime differs from immutable preflight")
    recheck = run_dir / f"resume_exclusive_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    result = certify_node(
        run_dir,
        snapshot,
        recheck,
        args.required_gpus,
        "resume/exclusive_node",
    )
    if result.returncode:
        raise RuntimeError(f"node is not exclusive for resume: {recheck}")


def validate_worker_manifest(
    path: Path,
    *,
    tier: str,
    method: str,
    repeat_index: int,
    position_index: int,
    contract_sha: str,
    preflight_sha: str,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    payload = common.read_json(path)
    checks = {
        "passed": payload.get("passed") is True,
        "tier": payload.get("tier") == tier,
        "method": payload.get("method") == method,
        "repeat_index": payload.get("repeat_index") == repeat_index,
        "position_index": payload.get("position_index") == position_index,
        "contract_sha": payload.get("contract_sha256") == contract_sha,
        "preflight_sha": payload.get("preflight_sha256") == preflight_sha,
        "runtime": payload.get("observed", {}).get("runtime_fingerprint")
        == preflight["runtime_fingerprint"],
        "data": payload.get("observed", {}).get("data_fingerprint")
        == preflight["data_audit"]["fingerprint"],
        "init": payload.get("observed", {}).get("init_sha256")
        == preflight["init_audit"]["common_init_sha256"],
        "resume": payload.get("observed", {}).get("resume_count") == 0,
        "timing_comparable": payload.get("observed", {}).get("timing_comparable")
        is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"worker manifest validation failed: {path}: {checks}")
    trainer = path.parent / "trainer"
    artifacts = {
        "summary.json": payload["summary_sha256"],
        "metrics.csv": payload["metrics_sha256"],
        "train_llama_swiglu_base.py": payload["trainer_local_base_sha256"],
        "../terminal.log": payload["terminal_log_sha256"],
    }
    for relative, expected in artifacts.items():
        artifact = trainer / relative
        if not artifact.is_file() or common.sha256_file(artifact) != expected:
            raise RuntimeError(f"worker artifact changed: {artifact}")
    return payload


def next_attempt(cell_dir: Path) -> tuple[str, Path]:
    observed: list[int] = []
    for path in cell_dir.glob("attempt_*"):
        try:
            observed.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    number = max(observed, default=0) + 1
    name = f"attempt_{number:03d}"
    attempt = cell_dir / name
    attempt.mkdir(parents=True, exist_ok=False)
    return name, attempt


def worker_command(
    args: argparse.Namespace,
    run_dir: Path,
    snapshot: Path,
    contract_path: Path,
    *,
    output: Path,
    tier: str,
    method: str,
    repeat_index: int,
    position_index: int,
) -> list[str]:
    return [
        str(common.lexical_absolute(Path(sys.executable))),
        str(snapshot / "llama1b_efficiency_worker.py"),
        "--output-dir",
        str(output),
        "--analysis-tier",
        tier,
        "--method",
        method,
        "--repeat-index",
        str(repeat_index),
        "--position-index",
        str(position_index),
        "--official-repo",
        str(args.official_repo.resolve()),
        "--python-exe",
        str(common.lexical_absolute(args.python_exe)),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--derived-base",
        str(snapshot / "train_llama_swiglu_efficiency_base.py"),
        "--profile-wrapper",
        str(snapshot / "train_llama_swiglu_1b.py"),
        "--contract",
        str(contract_path),
        "--preflight",
        str(run_dir / "preflight.json"),
        "--physical-gpu",
        args.physical_gpu,
        "--host-id",
        args.host_id,
        "--execution-domain",
        args.execution_domain,
    ]


def run_smoke(
    args: argparse.Namespace,
    run_dir: Path,
    snapshot: Path,
    contract_path: Path,
    contract: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    smoke_manifest_path = run_dir / "smoke" / "smoke_manifest.json"
    contract_sha = common.sha256_file(contract_path)
    preflight_sha = common.sha256_file(run_dir / "preflight.json")
    manifests: list[str] = []
    hashes: dict[str, str] = {}
    for position, method in enumerate(contract["method_order"]):
        cell_dir = run_dir / "smoke" / method
        completed = cell_dir / "cell_manifest.json"
        if completed.exists():
            cell = common.read_json(completed)
            worker_path = common.resolve_run_path(cell["worker_manifest"], run_dir)
            validate_worker_manifest(
                worker_path,
                tier="smoke",
                method=method,
                repeat_index=0,
                position_index=position,
                contract_sha=contract_sha,
                preflight_sha=preflight_sha,
                preflight=preflight,
            )
        else:
            attempt_name, attempt = next_attempt(cell_dir)
            command = worker_command(
                args,
                run_dir,
                snapshot,
                contract_path,
                output=attempt,
                tier="smoke",
                method=method,
                repeat_index=0,
                position_index=position,
            )
            print(f"MECH-42 smoke start method={method}", flush=True)
            result = run_command(
                run_dir, f"smoke/{method}/{attempt_name}", command, check=False
            )
            if result.returncode:
                raise RuntimeError(
                    f"smoke failed for {method}; inspect {attempt / 'status.json'}"
                )
            worker_path = attempt / "worker_manifest.json"
            validate_worker_manifest(
                worker_path,
                tier="smoke",
                method=method,
                repeat_index=0,
                position_index=position,
                contract_sha=contract_sha,
                preflight_sha=preflight_sha,
                preflight=preflight,
            )
            cell = {
                "schema_version": 1,
                "script_version": SCRIPT_VERSION,
                "passed": True,
                "tier": "smoke",
                "method": method,
                "repeat_index": 0,
                "position_index": position,
                "attempt": attempt_name,
                "worker_manifest": common.relative_to_run(worker_path, run_dir),
                "worker_manifest_sha256": common.sha256_file(worker_path),
                "completed_at": common.now_iso(),
            }
            common.atomic_write_json(completed, cell)
        manifests.append(common.relative_to_run(completed, run_dir))
        hashes[manifests[-1]] = common.sha256_file(completed)
    payload = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "passed": True,
        "tier": "smoke",
        "contract_sha256": contract_sha,
        "preflight_sha256": preflight_sha,
        "cell_manifests": manifests,
        "cell_manifest_sha256": hashes,
        "completed_at": common.now_iso(),
    }
    if smoke_manifest_path.exists():
        existing = common.read_json(smoke_manifest_path)
        if existing != payload:
            # Timestamp is immutable once the gate has been issued.
            comparable = dict(existing)
            comparable["completed_at"] = payload["completed_at"]
            if comparable != payload:
                raise RuntimeError("existing smoke manifest differs from validated cells")
            return existing
    else:
        common.atomic_write_json(smoke_manifest_path, payload)
    return payload


def read_passed_certificate(path: Path, required_gpus: list[str]) -> dict[str, Any]:
    payload = common.read_json(path)
    checks = {
        "passed": payload.get("passed") is True,
        "required_gpus": payload.get("required_gpus")
        == [int(index) for index in required_gpus],
        "processes": payload.get("active_compute_processes") == [],
        "gpu_count": len(payload.get("gpus", [])) == len(required_gpus),
        "names": all(
            gpu.get("name") == "NVIDIA H100 80GB HBM3"
            for gpu in payload.get("gpus", [])
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"exclusive certificate failed validation: {path}: {checks}")
    return payload


def validate_completed_cell(
    cell_path: Path,
    run_dir: Path,
    contract: dict[str, Any],
    preflight: dict[str, Any],
    *,
    repeat_index: int,
    position_index: int,
    method: str,
) -> dict[str, Any]:
    cell = common.read_json(cell_path)
    contract_sha = common.sha256_file(run_dir / "efficiency_contract.json")
    preflight_sha = common.sha256_file(run_dir / "preflight.json")
    exact = {
        "passed": True,
        "tier": "formal",
        "repeat_index": repeat_index,
        "position_index": position_index,
        "method": method,
        "contract_sha256": contract_sha,
        "preflight_sha256": preflight_sha,
    }
    if any(cell.get(key) != value for key, value in exact.items()):
        raise RuntimeError(f"completed cell metadata differs: {cell_path}")
    paths = {
        key: common.resolve_run_path(cell[key], run_dir)
        for key in (
            "worker_manifest",
            "trainer_summary",
            "exclusive_before",
            "exclusive_after",
            "gpu_isolation_monitor",
        )
    }
    for key, path in paths.items():
        expected = cell[f"{key}_sha256"]
        if not path.is_file() or common.sha256_file(path) != expected:
            raise RuntimeError(f"completed cell artifact changed: {key} {path}")
    validate_worker_manifest(
        paths["worker_manifest"],
        tier="formal",
        method=method,
        repeat_index=repeat_index,
        position_index=position_index,
        contract_sha=contract_sha,
        preflight_sha=preflight_sha,
        preflight=preflight,
    )
    before = read_passed_certificate(paths["exclusive_before"], ["0", "1"])
    after = read_passed_certificate(paths["exclusive_after"], ["0", "1"])
    monitor = common.read_json(paths["gpu_isolation_monitor"])
    if monitor.get("passed") is not True:
        raise RuntimeError(f"GPU isolation monitor failed: {paths['gpu_isolation_monitor']}")
    expected_monitor_identity = {
        str(row["index"]): row["uuid"] for row in before["gpus"]
    }
    if monitor.get("gpu_index_to_uuid") != expected_monitor_identity:
        raise RuntimeError("monitor GPU UUIDs differ from the exclusive certificates")
    if before["gpus"] != after["gpus"]:
        # Memory-free values and timestamps can differ; identity must not.
        before_identity = [(row["index"], row["uuid"], row["name"]) for row in before["gpus"]]
        after_identity = [(row["index"], row["uuid"], row["name"]) for row in after["gpus"]]
        if before_identity != after_identity:
            raise RuntimeError("physical GPU identity changed across timed cell")
    return cell


def run_formal_cell(
    args: argparse.Namespace,
    run_dir: Path,
    snapshot: Path,
    contract_path: Path,
    contract: dict[str, Any],
    preflight: dict[str, Any],
    *,
    repeat_index: int,
    position_index: int,
    method: str,
) -> dict[str, Any]:
    cell_dir = run_dir / "formal" / f"repeat_{repeat_index}" / method
    cell_path = cell_dir / "cell_manifest.json"
    if cell_path.exists():
        validated = validate_completed_cell(
            cell_path,
            run_dir,
            contract,
            preflight,
            repeat_index=repeat_index,
            position_index=position_index,
            method=method,
        )
        print(
            f"MECH-42 reuse repeat={repeat_index} position={position_index} method={method}",
            flush=True,
        )
        return validated
    attempt_name, attempt = next_attempt(cell_dir)
    before_path = attempt / "exclusive_before.json"
    before = certify_node(
        run_dir,
        snapshot,
        before_path,
        args.required_gpus,
        f"formal/repeat_{repeat_index}/{method}/{attempt_name}/exclusive_before",
    )
    if before.returncode:
        raise RuntimeError(f"node is not idle before timed cell: {before_path}")
    read_passed_certificate(before_path, args.required_gpus)
    stop_file = attempt / "gpu_monitor.stop"
    monitor_command = [
        str(common.lexical_absolute(Path(sys.executable))),
        str(snapshot / "gpu_isolation_monitor.py"),
        "--output-dir",
        str(attempt),
        "--stop-file",
        str(stop_file),
        "--timing-gpu",
        args.physical_gpu,
        "--idle-gpus",
        *[
            index
            for index in args.required_gpus
            if index != args.physical_gpu
        ],
        "--interval-seconds",
        str(
            contract["execution_policy"][
                "continuous_gpu_process_monitor_interval_seconds"
            ]
        ),
    ]
    append_command(
        run_dir,
        f"formal/repeat_{repeat_index}/{method}/{attempt_name}/gpu_monitor",
        monitor_command,
    )
    monitor_log = (attempt / "gpu_monitor_terminal.log").open("w", encoding="utf-8")
    monitor = subprocess.Popen(
        monitor_command,
        stdout=monitor_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    monitor_samples = attempt / "gpu_isolation_samples.jsonl"
    monitor_ready = False
    for _ in range(150):
        if monitor_samples.is_file() and monitor_samples.stat().st_size > 0:
            monitor_ready = True
            break
        if monitor.poll() is not None:
            break
        time.sleep(0.1)
    if not monitor_ready:
        stop_file.write_text(common.now_iso() + "\n", encoding="utf-8")
        try:
            monitor.wait(timeout=15)
        except subprocess.TimeoutExpired:
            monitor.terminate()
            monitor.wait(timeout=10)
        monitor_log.close()
        raise RuntimeError(
            f"GPU isolation monitor did not become ready: {attempt}"
        )
    command = worker_command(
        args,
        run_dir,
        snapshot,
        contract_path,
        output=attempt,
        tier="formal",
        method=method,
        repeat_index=repeat_index,
        position_index=position_index,
    )
    print(
        f"MECH-42 formal start repeat={repeat_index + 1}/4 "
        f"position={position_index + 1}/4 method={method}",
        flush=True,
    )
    worker_result: subprocess.CompletedProcess[str] | None = None
    try:
        worker_result = run_command(
            run_dir,
            f"formal/repeat_{repeat_index}/{method}/{attempt_name}/worker",
            command,
            check=False,
        )
    finally:
        stop_file.write_text(common.now_iso() + "\n", encoding="utf-8")
        try:
            monitor_return = monitor.wait(timeout=30)
        except subprocess.TimeoutExpired:
            monitor.terminate()
            monitor_return = monitor.wait(timeout=10)
        monitor_log.close()
    after_path = attempt / "exclusive_after.json"
    after = certify_node(
        run_dir,
        snapshot,
        after_path,
        args.required_gpus,
        f"formal/repeat_{repeat_index}/{method}/{attempt_name}/exclusive_after",
    )
    if worker_result is None or worker_result.returncode:
        raise RuntimeError(f"timed worker failed; inspect {attempt / 'status.json'}")
    if monitor_return:
        raise RuntimeError(
            f"continuous GPU isolation failed; inspect {attempt / 'gpu_isolation_monitor.json'}"
        )
    if after.returncode:
        raise RuntimeError(f"node is not idle after timed cell: {after_path}")
    read_passed_certificate(after_path, args.required_gpus)
    monitor_path = attempt / "gpu_isolation_monitor.json"
    monitor_payload = common.read_json(monitor_path)
    if monitor_payload.get("passed") is not True:
        raise RuntimeError(f"continuous GPU isolation failed: {monitor_path}")
    before_payload = read_passed_certificate(before_path, args.required_gpus)
    after_payload = read_passed_certificate(after_path, args.required_gpus)
    before_identity = {
        str(row["index"]): row["uuid"] for row in before_payload["gpus"]
    }
    after_identity = {
        str(row["index"]): row["uuid"] for row in after_payload["gpus"]
    }
    if (
        before_identity != after_identity
        or monitor_payload.get("gpu_index_to_uuid") != before_identity
    ):
        raise RuntimeError("GPU UUID identity changed across certificate/monitor evidence")
    contract_sha = common.sha256_file(contract_path)
    preflight_sha = common.sha256_file(run_dir / "preflight.json")
    worker_path = attempt / "worker_manifest.json"
    worker = validate_worker_manifest(
        worker_path,
        tier="formal",
        method=method,
        repeat_index=repeat_index,
        position_index=position_index,
        contract_sha=contract_sha,
        preflight_sha=preflight_sha,
        preflight=preflight,
    )
    trainer_summary = attempt / "trainer" / "summary.json"
    cell = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "passed": True,
        "tier": "formal",
        "repeat_index": repeat_index,
        "position_index": position_index,
        "method": method,
        "attempt": attempt_name,
        "attempt_dir": common.relative_to_run(attempt, run_dir),
        "contract_sha256": contract_sha,
        "preflight_sha256": preflight_sha,
        "runtime_fingerprint": preflight["runtime_fingerprint"],
        "data_fingerprint": preflight["data_audit"]["fingerprint"],
        "common_init_sha256": preflight["init_audit"]["common_init_sha256"],
        "worker_manifest": common.relative_to_run(worker_path, run_dir),
        "worker_manifest_sha256": common.sha256_file(worker_path),
        "trainer_summary": common.relative_to_run(trainer_summary, run_dir),
        "trainer_summary_sha256": common.sha256_file(trainer_summary),
        "exclusive_before": common.relative_to_run(before_path, run_dir),
        "exclusive_before_sha256": common.sha256_file(before_path),
        "exclusive_after": common.relative_to_run(after_path, run_dir),
        "exclusive_after_sha256": common.sha256_file(after_path),
        "gpu_isolation_monitor": common.relative_to_run(monitor_path, run_dir),
        "gpu_isolation_monitor_sha256": common.sha256_file(monitor_path),
        "observed": worker["observed"],
        "completed_at": common.now_iso(),
    }
    common.atomic_write_json(cell_path, cell)
    print(
        f"MECH-42 formal passed repeat={repeat_index + 1}/4 method={method}",
        flush=True,
    )
    return cell


def main() -> None:
    args = parse_args()
    contract_source = args.contract.resolve()
    contract = common.read_json(contract_source)
    validate_contract(contract)
    source_builder.build_source_bundle()
    if args.dry_run:
        payload = {
            "passed": True,
            "script_version": SCRIPT_VERSION,
            "methods": contract["method_order"],
            "orders": contract["execution_policy"]["orders"],
            "formal_cells": 16,
            "timed_updates_per_cell": contract["frozen_configuration"][
                "timed_updates"
            ],
            "timed_tokens_per_cell": int(
                contract["frozen_configuration"]["timed_updates"]
            )
            * int(contract["frozen_configuration"]["tokens_per_update"]),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "controller_status.json"
    try:
        existing_entries = list(run_dir.iterdir())
        if existing_entries and not args.resume:
            raise RuntimeError(
                f"run directory is non-empty; use --resume for {run_dir}"
            )
        contract_path = run_dir / "efficiency_contract.json"
        preflight_path = run_dir / "preflight.json"
        if preflight_path.exists():
            if not args.resume:
                raise RuntimeError("immutable preflight already exists without --resume")
            if not contract_path.is_file():
                raise RuntimeError("run contract is missing")
            if common.sha256_file(contract_source) != common.sha256_file(contract_path):
                raise RuntimeError("current contract differs from the run contract")
            snapshot = run_dir / "source_snapshot"
            preflight = common.read_json(preflight_path)
            revalidate_preflight(args, run_dir, contract, snapshot, preflight)
        else:
            if contract_path.exists():
                if common.sha256_file(contract_source) != common.sha256_file(contract_path):
                    raise RuntimeError("partial run contract differs")
            else:
                shutil.copy2(contract_source, contract_path)
            print("MECH-42 preflight: freezing source snapshot", flush=True)
            snapshot, source_audit = snapshot_sources(run_dir, contract_path)
            print(
                "MECH-42 preflight: certifying node, runtime, initialization, and "
                "hashing the frozen FineWeb cache",
                flush=True,
            )
            preflight = build_preflight(
                args,
                run_dir,
                contract_path,
                contract,
                snapshot,
                source_audit,
            )
        plan_path = run_dir / "run_plan.json"
        plan = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "contract_sha256": common.sha256_file(contract_path),
            "preflight_sha256": common.sha256_file(preflight_path),
            "official_repo": str(args.official_repo.resolve()),
            "python_exe": str(common.lexical_absolute(args.python_exe)),
            "data_dir": str(args.data_dir.resolve()),
            "physical_gpu": args.physical_gpu,
            "required_gpus": args.required_gpus,
            "host_id": args.host_id,
            "execution_domain": args.execution_domain,
            "orders": contract["execution_policy"]["orders"],
        }
        if plan_path.exists():
            if common.read_json(plan_path) != plan:
                raise RuntimeError("resume arguments differ from immutable run plan")
        else:
            common.atomic_write_json(plan_path, plan)
        common.atomic_write_json(
            status_path,
            {
                "status": "smoke",
                "script_version": SCRIPT_VERSION,
                "updated_at": common.now_iso(),
            },
        )
        smoke = run_smoke(
            args, run_dir, snapshot, contract_path, contract, preflight
        )
        cell_paths: list[str] = []
        for repeat_index, order in enumerate(contract["execution_policy"]["orders"]):
            for position_index, method in enumerate(order):
                common.atomic_write_json(
                    status_path,
                    {
                        "status": "formal",
                        "script_version": SCRIPT_VERSION,
                        "repeat_index": repeat_index,
                        "position_index": position_index,
                        "method": method,
                        "updated_at": common.now_iso(),
                    },
                )
                run_formal_cell(
                    args,
                    run_dir,
                    snapshot,
                    contract_path,
                    contract,
                    preflight,
                    repeat_index=repeat_index,
                    position_index=position_index,
                    method=method,
                )
                path = (
                    run_dir
                    / "formal"
                    / f"repeat_{repeat_index}"
                    / method
                    / "cell_manifest.json"
                )
                cell_paths.append(common.relative_to_run(path, run_dir))
        print("MECH-42 postflight: re-hashing the frozen FineWeb cache", flush=True)
        postflight_data = common.audit_data(args.data_dir)
        if (
            postflight_data["fingerprint"]
            != preflight["data_audit"]["fingerprint"]
        ):
            raise RuntimeError("FineWeb data changed during the isolated benchmark")
        postflight_data_path = run_dir / "postflight_data_audit.json"
        common.atomic_write_json(postflight_data_path, postflight_data)
        execution = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "status": "completed",
            "passed": True,
            "contract_sha256": common.sha256_file(contract_path),
            "preflight_sha256": common.sha256_file(preflight_path),
            "smoke_manifest": common.relative_to_run(
                run_dir / "smoke" / "smoke_manifest.json", run_dir
            ),
            "smoke_manifest_sha256": common.sha256_file(
                run_dir / "smoke" / "smoke_manifest.json"
            ),
            "formal_cell_manifests": cell_paths,
            "formal_cell_manifest_sha256": {
                relative: common.sha256_file(run_dir / relative)
                for relative in cell_paths
            },
            "formal_cell_count": len(cell_paths),
            "postflight_data_audit": common.relative_to_run(
                postflight_data_path, run_dir
            ),
            "postflight_data_audit_sha256": common.sha256_file(
                postflight_data_path
            ),
            "completed_at": common.now_iso(),
        }
        common.atomic_write_json(run_dir / "execution_manifest.json", execution)
        common.atomic_write_json(
            status_path,
            {
                "status": "completed",
                "script_version": SCRIPT_VERSION,
                "passed": True,
                "execution_manifest": str(run_dir / "execution_manifest.json"),
                "completed_at": common.now_iso(),
            },
        )
        print(f"MECH-42 execution manifest: {run_dir / 'execution_manifest.json'}")
    except KeyboardInterrupt:
        common.atomic_write_json(
            status_path,
            {
                "status": "interrupted",
                "script_version": SCRIPT_VERSION,
                "updated_at": common.now_iso(),
            },
        )
        raise
    except BaseException as exc:
        common.atomic_write_json(
            status_path,
            {
                "status": "failed",
                "script_version": SCRIPT_VERSION,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "updated_at": common.now_iso(),
            },
        )
        traceback.print_exc()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
