#!/usr/bin/env python3
"""Single-GPU, exact-resume segment worker for experiment 56.

The worker reuses the accepted LLaMA/SwiGLU model and optimizer implementation,
but owns the long-horizon data loader, phase schedule, lineage, metrics, and
checkpoint format.  A formal endpoint is assembled from immutable plateau and
cooldown segments by the controller.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any

import protocol as P


HERE = Path(__file__).resolve().parent
STOP_REQUESTED = False
np: Any = None
torch: Any = None


def stop_handler(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"EX56_STOP_REQUESTED signal={signum}", flush=True)


def load_base_trainer(path: Path, expected_sha256: str) -> Any:
    path = path.resolve()
    if not path.is_file() or P.sha256_file(path) != expected_sha256:
        raise RuntimeError("accepted LLaMA base trainer source mismatch")
    spec = importlib.util.spec_from_file_location("ex56_global_diag_llama_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import accepted trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bind_profile(base: Any, contract: dict[str, Any]) -> None:
    profile = contract["profile"]
    original_config = base.ModelConfig
    original_audit = base.architecture_audit

    def model_config_1b(*, sequence_length: int = 1024, **_: Any) -> Any:
        return original_config(
            n_layer=int(profile["n_layer"]),
            n_head=int(profile["n_head"]),
            n_embd=int(profile["n_embd"]),
            intermediate_size=int(profile["intermediate_size"]),
            sequence_length=int(sequence_length),
        )

    def architecture_audit_1b(model: Any) -> dict[str, Any]:
        payload = original_audit(model)
        payload["architecture"] = profile["name"]
        payload["profile"] = dict(profile)
        payload["base_trainer_sha256"] = contract["accepted_sources"][
            "scripts/17_llama_swiglu_validation/train_llama_swiglu.py"
        ]
        if int(payload["parameter_count"]) != int(profile["parameters"]):
            raise RuntimeError("LLaMA-1B parameter count drift")
        return payload

    base.ModelConfig = model_config_1b
    base.architecture_audit = architecture_audit_1b


class NoWrapSequentialShardLoader:
    def __init__(
        self,
        base: Any,
        files: list[Path],
        batch_size: int,
        sequence_length: int,
    ) -> None:
        if not files:
            raise FileNotFoundError("empty shard inventory")
        self.base = base
        self.files = [path.resolve() for path in files]
        self.batch_size = int(batch_size)
        self.sequence_length = int(sequence_length)
        self.current_shard = 0
        self.current_position = 0
        self.wrap_count = 0
        self.consumed_batches = 0
        self.tokens = base.load_data_shard(self.files[0])

    def reset(self) -> None:
        self.current_shard = 0
        self.current_position = 0
        self.wrap_count = 0
        self.consumed_batches = 0
        self.tokens = self.base.load_data_shard(self.files[0])

    def advance(self) -> None:
        next_shard = self.current_shard + 1
        if next_shard >= len(self.files):
            self.wrap_count += 1
            raise RuntimeError("EX56 no-wrap loader exhausted its unique shard stream")
        self.current_shard = next_shard
        self.current_position = 0
        self.tokens = self.base.load_data_shard(self.files[self.current_shard])

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        count = self.batch_size * self.sequence_length
        buffer = self.tokens[self.current_position : self.current_position + count + 1]
        if len(buffer) != count + 1:
            raise RuntimeError("EX56 loader position invariant failed")
        tensor = torch.tensor(buffer.astype(np.int32), dtype=torch.long)
        x = tensor[:-1].view(self.batch_size, self.sequence_length)
        y = tensor[1:].view(self.batch_size, self.sequence_length)
        self.current_position += count
        self.consumed_batches += 1
        if self.current_position + count + 1 > len(self.tokens):
            self.advance()
        return x.cuda(non_blocking=True), y.cuda(non_blocking=True)

    def state_dict(self) -> dict[str, int]:
        return {
            "current_shard": int(self.current_shard),
            "current_position": int(self.current_position),
            "wrap_count": int(self.wrap_count),
            "consumed_batches": int(self.consumed_batches),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        shard = int(payload["current_shard"])
        position = int(payload["current_position"])
        if not 0 <= shard < len(self.files):
            raise ValueError("checkpoint loader shard is out of range")
        if int(payload.get("wrap_count", -1)) != 0:
            raise RuntimeError("checkpoint has a nonzero train-data wrap count")
        self.current_shard = shard
        self.current_position = position
        self.wrap_count = int(payload["wrap_count"])
        self.consumed_batches = int(payload["consumed_batches"])
        self.tokens = self.base.load_data_shard(self.files[shard])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--base-trainer", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--source-checkpoint-sha256")
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument("--max-new-steps", type=int)
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()
    if args.max_new_steps is not None and args.max_new_steps <= 0:
        parser.error("--max-new-steps must be positive")
    if bool(args.source_checkpoint) != bool(args.source_checkpoint_sha256):
        parser.error("source checkpoint and sha256 must be supplied together")
    return args


def resolve_phase(contract: dict[str, Any], phase_id: str) -> tuple[dict[str, Any], bool]:
    phases = P.phase_map(contract)
    if phase_id in phases:
        return dict(phases[phase_id]), False
    pilot = contract["pilot"]
    if phase_id == "pilot_base_2":
        return {
            "id": phase_id,
            "parent": None,
            "start_step": 0,
            "target_step": int(pilot["first_target_step"]),
            "schedule": "plateau",
            "role": "engineering_pilot",
            "retain_checkpoint": False,
        }, True
    if phase_id == "pilot_branch_4":
        return {
            "id": phase_id,
            "parent": "pilot_base_2",
            "start_step": int(pilot["first_target_step"]),
            "target_step": int(pilot["branch_target_step"]),
            "schedule": "linear_cooldown",
            "role": "engineering_pilot",
            "retain_checkpoint": False,
        }, True
    raise KeyError(f"unknown phase: {phase_id}")


def audit_architecture(audit: dict[str, Any], method: str, contract: dict[str, Any]) -> None:
    profile = contract["profile"]
    expected = {
        "parameter_count": int(profile["parameters"]),
        "matrix_tensor_count": int(profile["expected_matrix_tensors"]),
        "backup_tensor_count": int(profile["expected_backup_tensors"]),
        "preconditioner_group_count": int(profile["expected_preconditioner_groups"][method]),
    }
    checks = {key: int(audit.get(key, -1)) == value for key, value in expected.items()}
    checks["tied"] = audit.get("embedding_head_tied") is True
    checks["bias_free"] = int(audit.get("bias_parameter_count", -1)) == 0
    checks["global_diag_route"] = audit.get("global_diag_route") is True
    checks["dense_k_workspace_forbidden"] = audit.get("dense_k_workspace_allowed") is False
    checks["all_groups_diagonal"] = all(
        row.get("kind") == "diag" for row in audit.get("preconditioner_groups", [])
    )
    if not all(checks.values()):
        raise RuntimeError(f"EX56 architecture audit failed: {checks}")


def runtime_metadata(base: Any, contract: dict[str, Any]) -> dict[str, Any]:
    payload = base.runtime_metadata()
    expected = contract["runtime"]
    checks = {
        "python": ".".join(str(value) for value in payload["python_version"]) == expected["python"],
        "torch": payload["torch"] == expected["torch"],
        "torch_cuda": payload["torch_cuda"] == expected["torch_cuda"],
        "triton": payload["triton"] == expected["triton"],
        "numpy": payload["numpy"] == expected["numpy"],
        "gpu": expected["gpu_name_contains"] in payload["gpu_name"]
        and payload["gpu_capability"] == expected["compute_capability"]
        and int(payload["gpu_total_memory_bytes"]) >= int(expected["minimum_gpu_memory_bytes"]),
        "triton_kernels": payload["triton_kernels_sha256"]
        == expected["accepted_triton_kernels_sha256"],
    }
    payload["contract_checks"] = checks
    if not all(checks.values()):
        raise RuntimeError(f"EX56 worker runtime mismatch: {checks}")
    return payload


def checkpoint_payload(
    *,
    base: Any,
    raw_model: Any,
    optimizers: list[Any],
    loader: NoWrapSequentialShardLoader,
    x: torch.Tensor,
    y: torch.Tensor,
    completed_steps: int,
    train_s: float,
    steady_train_s: float,
    steady_steps: int,
    peak_allocated_bytes: int,
    resume_count: int,
    init_sha256: str,
    method: str,
    seed: int,
    phase: dict[str, Any],
    contract_sha256: str,
    data_inventory_sha256: str,
    source_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    return {
        "format_version": P.CHECKPOINT_SCHEMA,
        "contract_sha256": contract_sha256,
        "data_inventory_sha256": data_inventory_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "method": method,
        "seed": int(seed),
        "phase_id": phase["id"],
        "phase_start_step": int(phase["start_step"]),
        "phase_target_step": int(phase["target_step"]),
        "completed_steps": int(completed_steps),
        "model": raw_model.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "train_loader": loader.state_dict(),
        "next_x": x.detach().cpu(),
        "next_y": y.detach().cpu(),
        "rng": base.capture_rng_state(),
        "train_s": float(train_s),
        "steady_train_s": float(steady_train_s),
        "steady_steps": int(steady_steps),
        "peak_allocated_bytes": int(peak_allocated_bytes),
        "resume_count": int(resume_count),
        "init_sha256": init_sha256,
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    torch.cuda.synchronize()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_checkpoint_rng_state(base: Any, payload: dict[str, Any]) -> None:
    """Restore RNG after a CUDA-mapped load without CUDA RNG-state tensors.

    Formal checkpoints are loaded directly onto CUDA so that multi-GB model and
    optimizer states do not make a second CPU-resident copy.  That map also
    moves the tiny RNG ByteTensors saved from CPU to CUDA, while PyTorch's RNG
    restore APIs require CPU ByteTensors.  Normalize only those state tensors
    back to CPU and leave the large checkpoint payload CUDA-resident.
    """

    normalized = dict(payload)
    normalized["torch_cpu"] = payload["torch_cpu"].cpu()
    normalized["torch_cuda"] = [state.cpu() for state in payload["torch_cuda"]]
    base.restore_rng_state(normalized)


def validate_checkpoint_identity(
    checkpoint: dict[str, Any],
    *,
    args: argparse.Namespace,
    phase: dict[str, Any],
    contract_sha256: str,
    data_inventory_sha256: str,
    init_sha256: str,
    in_place: bool,
) -> None:
    expected_phase = phase["id"] if in_place else phase.get("parent")
    expected_step = int(checkpoint["completed_steps"]) if in_place else int(phase["start_step"])
    checks = {
        "schema": checkpoint.get("format_version") == P.CHECKPOINT_SCHEMA,
        "contract": checkpoint.get("contract_sha256") == contract_sha256,
        "data": checkpoint.get("data_inventory_sha256") == data_inventory_sha256,
        "method": checkpoint.get("method") == args.method,
        "seed": int(checkpoint.get("seed", -1)) == int(args.seed),
        "phase": checkpoint.get("phase_id") == expected_phase,
        "step": int(checkpoint.get("completed_steps", -1)) == expected_step,
        "init": checkpoint.get("init_sha256") == init_sha256,
        "no_wrap": int(checkpoint.get("train_loader", {}).get("wrap_count", -1)) == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"EX56 checkpoint identity mismatch: {checks}")


def main() -> int:
    global STOP_REQUESTED, np, torch
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    args = parse_args()
    import numpy as numpy_module
    import torch as torch_module

    np = numpy_module
    torch = torch_module
    contract_path = args.contract.resolve()
    contract = P.read_json(contract_path)
    P.assert_contract(contract)
    contract_sha256 = P.sha256_file(contract_path)
    phase, engineering_pilot = resolve_phase(contract, args.phase_id)
    if engineering_pilot:
        if args.method != contract["pilot"]["method"] or int(args.seed) != int(contract["pilot"]["seed"]):
            raise RuntimeError("engineering pilot method/seed mismatch")
    elif args.method not in contract["grid"]["methods"] or int(args.seed) not in contract["grid"]["seeds"]:
        raise RuntimeError("formal method/seed is outside the frozen grid")

    base_sha = contract["accepted_sources"]["scripts/17_llama_swiglu_validation/train_llama_swiglu.py"]
    base = load_base_trainer(args.base_trainer, base_sha)
    bind_profile(base, contract)
    training = contract["training"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = base.ModelConfig(sequence_length=int(training["sequence_length"]))
    model_cpu = base.LlamaForCausalLM(config, args.method)
    init_sha256 = base.hash_named_parameters(model_cpu)
    architecture = base.architecture_audit(model_cpu)
    audit_architecture(architecture, args.method, contract)
    print(
        "EX56_INIT_AUDIT "
        + json.dumps(
            {
                "method": args.method,
                "seed": args.seed,
                "phase_id": phase["id"],
                "init_sha256": init_sha256,
                "architecture": architecture,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.init_only:
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if base.TRITON_KERNEL_IMPORT_ERROR is not None:
        raise RuntimeError(f"accepted triton kernels unavailable: {base.TRITON_KERNEL_IMPORT_ERROR!r}")
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_device(0)
    torch.cuda.manual_seed_all(args.seed)
    runtime = runtime_metadata(base, contract)

    data_audit = P.read_json(args.data_audit.resolve())
    if data_audit.get("schema_version") != P.DATA_AUDIT_SCHEMA or data_audit.get("passed") is not True:
        raise RuntimeError("formal data audit is missing or failed")
    metadata_checks = P.verify_data_metadata(data_audit)
    if not metadata_checks or not all(metadata_checks.values()):
        bad = [key for key, value in metadata_checks.items() if not value]
        raise RuntimeError(f"FineWeb metadata changed after preflight: {bad[:10]}")
    data_inventory_sha256 = str(data_audit["inventory_sha256"])
    data_root = Path(data_audit["data_dir"])
    train_files = [data_root / row["name"] for row in data_audit["inventory"]["train"]]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "phase_manifest.json"
    status_path = output_dir / "status.json"
    if manifest_path.exists():
        raise RuntimeError("passed phase is immutable; controller should have skipped it")
    if metrics_path.exists() and not checkpoint_path.exists():
        raise RuntimeError("metrics exist without an exact-resume checkpoint")

    train_loader = NoWrapSequentialShardLoader(
        base, train_files, int(training["device_batch_size"]), int(training["sequence_length"])
    )
    # Validation restarts from the frozen validation shard for every event, as
    # in the accepted 6200-step trainer.  The no-wrap requirement applies only
    # to the continuously consumed training stream.
    val_loader = base.SequentialShardLoader(
        str(data_root / contract["data"]["validation_pattern"]),
        int(training["device_batch_size"]),
        int(training["sequence_length"]),
    )
    raw_model = model_cpu.cuda()
    optimizer_args = argparse.Namespace(
        method=args.method,
        backup_lr=float(training["backup_lr"]),
        matrix_lr=float(training["matrix_lr"]),
        adamw_matrix_lr=float(training["adamw_matrix_lr"]),
    )
    optimizers, matrix_optimizer = base.make_optimizers(raw_model, optimizer_args)
    compiled_model = torch.compile(raw_model)
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    completed_steps = int(phase["start_step"])
    train_s = 0.0
    steady_train_s = 0.0
    steady_steps = 0
    peak_allocated_bytes = 0
    resume_count = 0
    source_checkpoint_sha256 = args.source_checkpoint_sha256
    checkpoint: dict[str, Any] | None = None
    if args.resume == "auto" and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        validate_checkpoint_identity(
            checkpoint,
            args=args,
            phase=phase,
            contract_sha256=contract_sha256,
            data_inventory_sha256=data_inventory_sha256,
            init_sha256=init_sha256,
            in_place=True,
        )
        raw_model.load_state_dict(checkpoint["model"])
        for optimizer, state in zip(optimizers, checkpoint["optimizers"]):
            optimizer.load_state_dict(state)
        train_loader.load_state_dict(checkpoint["train_loader"])
        x = checkpoint["next_x"].cuda(non_blocking=True)
        y = checkpoint["next_y"].cuda(non_blocking=True)
        completed_steps = int(checkpoint["completed_steps"])
        train_s = float(checkpoint["train_s"])
        steady_train_s = float(checkpoint["steady_train_s"])
        steady_steps = int(checkpoint["steady_steps"])
        peak_allocated_bytes = int(checkpoint["peak_allocated_bytes"])
        resume_count = int(checkpoint.get("resume_count", 0)) + 1
        source_checkpoint_sha256 = checkpoint.get("source_checkpoint_sha256")
        restore_checkpoint_rng_state(base, checkpoint["rng"])
        P.trim_metrics(metrics_path, completed_steps)
        print(f"EX56_RESUME phase={phase['id']} completed_steps={completed_steps}", flush=True)
    elif args.source_checkpoint is not None:
        source_path = args.source_checkpoint.resolve()
        observed_source_sha = P.sha256_file(source_path)
        if observed_source_sha != args.source_checkpoint_sha256:
            raise RuntimeError("source checkpoint sha256 mismatch")
        checkpoint = torch.load(source_path, map_location="cuda", weights_only=False)
        validate_checkpoint_identity(
            checkpoint,
            args=args,
            phase=phase,
            contract_sha256=contract_sha256,
            data_inventory_sha256=data_inventory_sha256,
            init_sha256=init_sha256,
            in_place=False,
        )
        raw_model.load_state_dict(checkpoint["model"])
        for optimizer, state in zip(optimizers, checkpoint["optimizers"]):
            optimizer.load_state_dict(state)
        train_loader.load_state_dict(checkpoint["train_loader"])
        x = checkpoint["next_x"].cuda(non_blocking=True)
        y = checkpoint["next_y"].cuda(non_blocking=True)
        completed_steps = int(checkpoint["completed_steps"])
        restore_checkpoint_rng_state(base, checkpoint["rng"])
        print(f"EX56_BRANCH phase={phase['id']} source={source_path}", flush=True)
    else:
        if int(phase["start_step"]) != 0:
            raise RuntimeError("non-root phase requires a source checkpoint")
        x, y = train_loader.next_batch()

    # The loaded checkpoint can contain many GB of CUDA tensors.  Model and
    # optimizer state have already been restored, so retaining the temporary
    # payload would create an avoidable resume-only memory spike during the
    # first compiled update.
    if checkpoint is not None:
        del checkpoint
        torch.cuda.empty_cache()

    expected = P.expected_cursor(completed_steps, data_audit, contract)
    if not P.cursor_matches(train_loader.state_dict(), expected):
        raise RuntimeError(
            f"loader cursor mismatch at phase entry: observed={train_loader.state_dict()} expected={expected}"
        )
    target_step = int(phase["target_step"])
    if not int(phase["start_step"]) <= completed_steps <= target_step:
        raise RuntimeError("checkpoint step is outside phase bounds")

    start_completed = completed_steps
    accumulation_steps = int(training["gradient_accumulation_steps"])
    val_steps = (
        1
        if engineering_pilot
        else int(contract["validation"]["tokens_per_evaluation"])
        // int(training["microbatch_tokens"])
    )
    torch.cuda.reset_peak_memory_stats()
    P.atomic_json(
        status_path,
        {
            "status": "training",
            "method": args.method,
            "seed": args.seed,
            "phase_id": phase["id"],
            "completed_steps": completed_steps,
            "resume_count": resume_count,
            "runtime": runtime,
        },
    )

    def save_current() -> None:
        expected_cursor = P.expected_cursor(completed_steps, data_audit, contract)
        observed_cursor = train_loader.state_dict()
        if not P.cursor_matches(observed_cursor, expected_cursor):
            raise RuntimeError(
                f"loader cursor mismatch before checkpoint: observed={observed_cursor} expected={expected_cursor}"
            )
        save_checkpoint(
            checkpoint_path,
            checkpoint_payload(
                base=base,
                raw_model=raw_model,
                optimizers=optimizers,
                loader=train_loader,
                x=x,
                y=y,
                completed_steps=completed_steps,
                train_s=train_s,
                steady_train_s=steady_train_s,
                steady_steps=steady_steps,
                peak_allocated_bytes=peak_allocated_bytes,
                resume_count=resume_count,
                init_sha256=init_sha256,
                method=args.method,
                seed=args.seed,
                phase=phase,
                contract_sha256=contract_sha256,
                data_inventory_sha256=data_inventory_sha256,
                source_checkpoint_sha256=source_checkpoint_sha256,
            ),
        )

    while True:
        if P.should_validate(phase, completed_steps, int(contract["validation"]["regular_every_steps"])):
            raw_model.eval()
            val_loader.reset()
            val_loss = torch.zeros((), device="cuda", dtype=torch.float32)
            for _ in range(val_steps):
                x_val, y_val = val_loader.next_batch()
                with torch.no_grad(), autocast:
                    _, loss = compiled_model(x_val, y_val, return_logits=False, precond_flag=False)
                assert loss is not None
                val_loss += loss.detach()
            val_value = float((val_loss / val_steps).item())
            if not math.isfinite(val_value):
                raise FloatingPointError(f"non-finite validation loss at {completed_steps}")
            avg_ms = 1000.0 * steady_train_s / steady_steps if steady_steps else float("nan")
            P.append_metric(
                metrics_path,
                {
                    "event": "val",
                    "phase_id": phase["id"],
                    "schedule": phase["schedule"],
                    "step": completed_steps,
                    "segment_step": completed_steps - int(phase["start_step"]),
                    "loss": f"{val_value:.9f}",
                    "train_s": f"{train_s:.9f}",
                    "steady_train_s": f"{steady_train_s:.9f}",
                    "step_avg_ms": f"{avg_ms:.6f}",
                    "lr_backup": f"{optimizers[0].param_groups[0]['lr']:.12g}",
                    "lr_matrix": f"{optimizers[1].param_groups[0]['lr']:.12g}",
                    "tokens_seen": completed_steps * int(training["tokens_per_update"]),
                    "tokens_per_parameter": f"{completed_steps * int(training['tokens_per_update']) / int(contract['profile']['parameters']):.12f}",
                    "loader_consumed_batches": train_loader.consumed_batches,
                    "wrap_count": train_loader.wrap_count,
                },
            )
            print(
                f"EX56 phase={phase['id']} step={completed_steps}/{target_step} val_loss={val_value:.4f}",
                flush=True,
            )

        if STOP_REQUESTED:
            save_current()
            P.atomic_json(
                status_path,
                {
                    "status": "interrupted",
                    "method": args.method,
                    "seed": args.seed,
                    "phase_id": phase["id"],
                    "completed_steps": completed_steps,
                    "resume_count": resume_count,
                    "checkpoint": str(checkpoint_path),
                    "planned_stop": False,
                },
            )
            return 75

        if completed_steps == target_step:
            break

        multiplier = P.lr_multiplier(phase, completed_steps)
        optimizers[0].param_groups[0]["lr"] = float(training["backup_lr"]) * multiplier
        optimizers[1].param_groups[0]["lr"] = float(training["matrix_lr"]) * multiplier
        if isinstance(matrix_optimizer, base.SharedInputNewtonMuon):
            matrix_optimizer.global_step = completed_steps
            precond_flag = matrix_optimizer.precond_flag_for_step(completed_steps)
        else:
            precond_flag = False

        raw_model.train()
        torch.cuda.synchronize()
        update_start = time.perf_counter()
        for _ in range(accumulation_steps):
            with autocast:
                _, loss = compiled_model(x, y, return_logits=False, precond_flag=precond_flag)
                assert loss is not None
                train_loss = loss.detach()
                scaled_loss = loss / accumulation_steps
            x, y = train_loader.next_batch()
            scaled_loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        raw_model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        update_s = time.perf_counter() - update_start
        train_s += update_s
        completed_steps += 1
        if completed_steps - int(phase["start_step"]) > 32:
            steady_train_s += update_s
            steady_steps += 1
        loss_value = float(train_loss.item())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"non-finite train loss at {completed_steps}")
        avg_ms = 1000.0 * steady_train_s / steady_steps if steady_steps else float("nan")
        P.append_metric(
            metrics_path,
            {
                "event": "train",
                "phase_id": phase["id"],
                "schedule": phase["schedule"],
                "step": completed_steps,
                "segment_step": completed_steps - int(phase["start_step"]),
                "loss": f"{loss_value:.9f}",
                "train_s": f"{train_s:.9f}",
                "steady_train_s": f"{steady_train_s:.9f}",
                "step_avg_ms": f"{avg_ms:.6f}",
                "lr_backup": f"{optimizers[0].param_groups[0]['lr']:.12g}",
                "lr_matrix": f"{optimizers[1].param_groups[0]['lr']:.12g}",
                "tokens_seen": completed_steps * int(training["tokens_per_update"]),
                "tokens_per_parameter": f"{completed_steps * int(training['tokens_per_update']) / int(contract['profile']['parameters']):.12f}",
                "loader_consumed_batches": train_loader.consumed_batches,
                "wrap_count": train_loader.wrap_count,
            },
        )
        print(
            f"EX56 phase={phase['id']} step={completed_steps}/{target_step} train_loss={loss_value:.4f} step_avg={avg_ms:.2f}ms",
            flush=True,
        )
        peak_allocated_bytes = max(peak_allocated_bytes, int(torch.cuda.max_memory_allocated()))
        checkpoint_due = completed_steps % int(training["checkpoint_every_steps"]) == 0
        planned_stop = args.max_new_steps is not None and completed_steps - start_completed >= args.max_new_steps
        if checkpoint_due or STOP_REQUESTED or planned_stop:
            save_current()
            P.atomic_json(
                status_path,
                {
                    "status": "interrupted" if STOP_REQUESTED or planned_stop else "training",
                    "method": args.method,
                    "seed": args.seed,
                    "phase_id": phase["id"],
                    "completed_steps": completed_steps,
                    "resume_count": resume_count,
                    "checkpoint": str(checkpoint_path),
                    "planned_stop": bool(planned_stop),
                },
            )
        if STOP_REQUESTED or planned_stop:
            return 75

    peak_allocated_bytes = max(peak_allocated_bytes, int(torch.cuda.max_memory_allocated()))
    save_current()
    rows = P.read_metrics(metrics_path)
    val_rows = [row for row in rows if row["event"] == "val"]
    train_rows = [row for row in rows if row["event"] == "train"]
    if not val_rows or int(val_rows[-1]["step"]) != target_step:
        raise RuntimeError("phase target validation is missing")
    expected_final_cursor = P.expected_cursor(target_step, data_audit, contract)
    observed_final_cursor = train_loader.state_dict()
    checks = {
        "completed": completed_steps == target_step,
        "target_validation": int(val_rows[-1]["step"]) == target_step,
        "finite": all(math.isfinite(float(row["loss"])) for row in rows),
        "no_wrap": train_loader.wrap_count == 0,
        "cursor": P.cursor_matches(observed_final_cursor, expected_final_cursor),
        "checkpoint": checkpoint_path.is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"EX56 phase completion checks failed: {checks}")
    checkpoint_sha256 = P.sha256_file(checkpoint_path)
    memory = base.activation_state_memory(raw_model)
    if isinstance(matrix_optimizer, base.SharedInputNewtonMuon):
        memory.update(matrix_optimizer.memory_audit())
    else:
        memory.update(
            {
                "k_cov_bytes": 0,
                "k_inv_bytes": 0,
                "k_state_bytes": 0,
                "preconditioner_workspace_bytes": 0,
            }
        )
    memory_checks = {
        "k_state_bytes": int(memory.get("k_state_bytes", -1))
        == int(contract["profile"]["expected_global_diag_k_state_bytes"]),
        "dense_workspace_absent": int(memory.get("preconditioner_workspace_bytes", -1)) == 0,
    }
    if not all(memory_checks.values()):
        raise RuntimeError(f"EX56 global-diagonal memory audit failed: {memory_checks}")
    summary = {
        "schema_version": "ex56_llama1b_global_diag_phase_summary_v1",
        "status": "completed",
        "engineering_pilot": engineering_pilot,
        "method": args.method,
        "seed": args.seed,
        "phase": phase,
        "completed_steps": completed_steps,
        "tokens_seen": completed_steps * int(training["tokens_per_update"]),
        "tokens_per_parameter": completed_steps
        * int(training["tokens_per_update"])
        / int(contract["profile"]["parameters"]),
        "final_val_loss": float(val_rows[-1]["loss"]),
        "best_segment_val_loss": min(float(row["loss"]) for row in val_rows),
        "final_train_loss": float(train_rows[-1]["loss"]),
        "init_sha256": init_sha256,
        "architecture": architecture,
        "runtime": runtime,
        "contract_sha256": contract_sha256,
        "data_inventory_sha256": data_inventory_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "metrics_path": str(metrics_path),
        "metrics_sha256": P.sha256_file(metrics_path),
        "loader_final": observed_final_cursor,
        "loader_expected": expected_final_cursor,
        "resume_count": resume_count,
        "timing_comparable": resume_count == 0,
        "train_s": train_s,
        "steady_train_s": steady_train_s,
        "steady_steps": steady_steps,
        "step_avg_ms": 1000.0 * steady_train_s / max(1, steady_steps),
        "peak_allocated_bytes": peak_allocated_bytes,
        "optimizer_state_bytes": base.optimizer_state_bytes(optimizers),
        "global_diag_memory_checks": memory_checks,
        **memory,
    }
    P.atomic_json(summary_path, summary)
    manifest = {
        "schema_version": P.PHASE_MANIFEST_SCHEMA,
        "passed": True,
        "method": args.method,
        "seed": args.seed,
        "phase_id": phase["id"],
        "role": phase["role"],
        "contract_sha256": contract_sha256,
        "data_inventory_sha256": data_inventory_sha256,
        "summary_sha256": P.sha256_file(summary_path),
        "metrics_sha256": summary["metrics_sha256"],
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "bytes": checkpoint_path.stat().st_size,
            "retained": bool(phase["retain_checkpoint"]),
        },
        "checks": checks,
    }
    P.atomic_json(manifest_path, manifest)
    P.atomic_json(
        status_path,
        {
            "status": "completed",
            "method": args.method,
            "seed": args.seed,
            "phase_id": phase["id"],
            "summary": str(summary_path),
            "manifest": str(manifest_path),
            "resume_count": resume_count,
        },
    )
    print("EX56_PHASE_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
