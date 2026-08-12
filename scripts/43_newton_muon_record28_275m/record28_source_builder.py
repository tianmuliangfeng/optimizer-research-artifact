#!/usr/bin/env python3
"""Build the four frozen experiment-43 training sources from pinned Git blobs.

The upstream worktree is never used as source.  Every transformation is an
exact, single-occurrence replacement, so upstream drift fails closed.
"""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path

from record28_common import (
    EXPECTED_CANONICAL_SHA256,
    METHODS,
    canonical_bytes,
    git_blob,
)


SCRIPT_VERSION = "2026-07-31.1"


@dataclass(frozen=True)
class DerivedSource:
    method: str
    cproj_k_mode: str
    base_script: str
    base_canonical_sha256: str
    derived_sha256: str
    source: str
    unified_diff: str


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Record-28 source expected exactly one {label!r} anchor; observed {count}"
        )
    return source.replace(old, new, 1)


def replace_exact_count(
    source: str, old: str, new: str, label: str, expected_count: int
) -> str:
    count = source.count(old)
    if count != expected_count:
        raise RuntimeError(
            f"Record-28 source expected {expected_count} {label!r} anchors; "
            f"observed {count}"
        )
    return source.replace(old, new)


COMMON_IMPORTS_OLD = """import copy
import glob
from dataclasses import dataclass
"""

COMMON_IMPORTS_NEW = """import copy
import glob
import random
import hashlib
import json
import numpy as np
from dataclasses import dataclass
"""


COMMON_CONTROL = """args = Hyperparameters()

# Experiment-43 controlled overlay.  Formal mathematics and budget are pinned
# to the upstream Newton-Muon-2 near-Record-#28 recipe; only the declared
# mlp.c_proj K representation differs among Newton-family methods.
RECORD28_METHOD = os.environ["RECORD28_METHOD"]
RECORD28_CPROJ_K_MODE = os.environ["RECORD28_CPROJ_K_MODE"]
RECORD28_SEED = int(os.environ["RECORD28_SEED"])
RECORD28_STAGE = os.environ["RECORD28_STAGE"]
RECORD28_OUTPUT_DIR = os.path.abspath(os.environ["RECORD28_OUTPUT_DIR"])
RECORD28_CELL_ID = os.environ["RECORD28_CELL_ID"]
RECORD28_NUM_ITERATIONS = int(os.environ["RECORD28_NUM_ITERATIONS"])
RECORD28_SCHEDULE_ITERATIONS = 1695
RECORD28_VAL_LOSS_EVERY = int(os.environ["RECORD28_VAL_LOSS_EVERY"])
RECORD28_VAL_TOKENS = int(os.environ["RECORD28_VAL_TOKENS"])
RECORD28_SAVE_CHECKPOINT = os.environ.get("RECORD28_SAVE_CHECKPOINT", "0") == "1"
RECORD28_INIT_ONLY = os.environ.get("RECORD28_INIT_ONLY", "0") == "1"
if RECORD28_STAGE not in ("smoke", "formal"):
    raise ValueError(f"invalid RECORD28_STAGE={RECORD28_STAGE!r}")
if RECORD28_STAGE == "formal":
    if (RECORD28_NUM_ITERATIONS, RECORD28_VAL_LOSS_EVERY, RECORD28_VAL_TOKENS) != (1695, 50, 10485760):
        raise ValueError("formal Record-28 budget/protocol drift")
elif RECORD28_NUM_ITERATIONS < 18:
    raise ValueError("smoke must cross the first Newton refresh at step 15")
args.num_iterations = RECORD28_NUM_ITERATIONS
args.val_loss_every = RECORD28_VAL_LOSS_EVERY
args.val_tokens = RECORD28_VAL_TOKENS
args.save_checkpoint = RECORD28_SAVE_CHECKPOINT
args.run_id = RECORD28_CELL_ID
os.makedirs(RECORD28_OUTPUT_DIR, exist_ok=True)
"""


MODE_VALIDATION_MUON = """if RECORD28_METHOD != "muon" or RECORD28_CPROJ_K_MODE != "not_applicable":
    raise ValueError("Muon source requires method=muon and cproj mode=not_applicable")
"""


MODE_VALIDATION_NEWTON = """if RECORD28_METHOD not in ("original_newton_muon", "selective_none", "selective_diag"):
    raise ValueError(f"invalid Newton-family method={RECORD28_METHOD!r}")
_record28_expected_modes = {
    "original_newton_muon": "block4",
    "selective_none": "none",
    "selective_diag": "diag",
}
if RECORD28_CPROJ_K_MODE != _record28_expected_modes[RECORD28_METHOD]:
    raise ValueError("Newton-family method/c_proj K-mode mismatch")
"""


WORLD_SIZE_OLD = """world_size = int(os.environ["WORLD_SIZE"])
assert 8 % world_size == 0, "world_size must be a divisor of 8"
"""

WORLD_SIZE_NEW = """world_size = int(os.environ["WORLD_SIZE"])
if world_size != 1:
    raise RuntimeError(f"experiment 43 is frozen to single-GPU world_size=1, observed {world_size}")
assert 8 % world_size == 0, "world_size must be a divisor of 8"
"""


SEED_AND_AUDIT = """master_process = (rank == 0) # this process will do logging, checkpointing etc.

# The upstream scripts do not set a seed.  Pair all four methods by seeding
# before model construction. DATA loader order itself is deterministic.
random.seed(RECORD28_SEED)
np.random.seed(RECORD28_SEED)
torch.manual_seed(RECORD28_SEED)
torch.cuda.manual_seed_all(RECORD28_SEED)

def _record28_tensor_bytes(tensor: Tensor) -> bytes:
    cpu = tensor.detach().contiguous().cpu()
    return cpu.view(torch.uint8).numpy().tobytes()

def _record28_model_fingerprint(module: nn.Module) -> tuple[str, str, int]:
    value_digest = hashlib.sha256()
    structure_digest = hashlib.sha256()
    parameter_count = 0
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            descriptor = f"{name}|{tuple(parameter.shape)}|{parameter.dtype}"
            encoded = descriptor.encode("utf-8")
            value_digest.update(encoded)
            value_digest.update(_record28_tensor_bytes(parameter))
            structure_digest.update(encoded)
            parameter_count += parameter.numel()
    return value_digest.hexdigest(), structure_digest.hexdigest(), parameter_count

def _record28_digest_value(digest, value) -> None:
    if isinstance(value, Tensor):
        digest.update(
            f"tensor|{tuple(value.shape)}|{value.dtype}|{value.device.type}".encode()
        )
        digest.update(_record28_tensor_bytes(value))
    elif isinstance(value, dict):
        digest.update(b"dict{")
        for key in sorted(value, key=lambda item: repr(item)):
            _record28_digest_value(digest, key)
            _record28_digest_value(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + b"[")
        for child in value:
            _record28_digest_value(digest, child)
        digest.update(b"]")
    else:
        digest.update(f"{type(value).__name__}:{value!r}".encode())

def _record28_optimizer_fingerprint(optimizers) -> str:
    digest = hashlib.sha256()
    for optimizer in optimizers:
        _record28_digest_value(digest, optimizer.state_dict())
    return digest.hexdigest()

def _record28_iter_tensors(value):
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _record28_iter_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _record28_iter_tensors(child)

def _record28_storage_key(tensor: Tensor):
    storage = tensor.untyped_storage()
    index = tensor.device.index if tensor.device.index is not None else -1
    return tensor.device.type, index, storage.data_ptr(), storage.nbytes()

def _record28_unique_storage_bytes(tensors) -> int:
    storages = {}
    for tensor in tensors:
        key = _record28_storage_key(tensor)
        storages[key] = key[-1]
    return int(sum(storages.values()))
"""


LOGGING_OLD = """if master_process:
    run_id = args.run_id
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
"""

LOGGING_NEW = """if master_process:
    run_id = args.run_id
    logfile = os.path.join(RECORD28_OUTPUT_DIR, "training.log")
    print(logfile)
"""


MODEL_FINGERPRINT_ANCHOR = """for param in model.parameters():
    dist.broadcast(param.detach(), 0)
"""

MODEL_FINGERPRINT_BLOCK = MODEL_FINGERPRINT_ANCHOR + """
record28_init_sha256, record28_structure_sha256, record28_parameter_count = _record28_model_fingerprint(model)
if record28_parameter_count != 275743572:
    raise RuntimeError(f"parameter-count drift: {record28_parameter_count}")
record28_metadata = {
    "schema_version": 1,
    "method": RECORD28_METHOD,
    "cproj_k_mode": RECORD28_CPROJ_K_MODE,
    "seed": RECORD28_SEED,
    "stage": RECORD28_STAGE,
    "init_sha256": record28_init_sha256,
    "parameter_structure_sha256": record28_structure_sha256,
    "parameter_count": record28_parameter_count,
    "train_iterations": args.num_iterations,
    "schedule_iterations": RECORD28_SCHEDULE_ITERATIONS,
    "train_seq_len": args.train_seq_len,
    "val_seq_len": args.val_seq_len,
    "grad_accum_steps": grad_accum_steps,
    "tokens_per_update": world_size * grad_accum_steps * args.train_seq_len,
    "world_size": world_size,
}
print0("RECORD28_METADATA " + json.dumps(record28_metadata, sort_keys=True), console=True)
if RECORD28_INIT_ONLY:
    dist.destroy_process_group()
    raise SystemExit(0)
"""


RESET_PEAK_OLD = """del train_loader, initial_state

########################################
"""

RESET_PEAK_NEW = """random.setstate(record28_rng_state["python"])
np.random.set_state(record28_rng_state["numpy"])
torch.set_rng_state(record28_rng_state["torch"])
torch.cuda.set_rng_state_all(record28_rng_state["cuda"])
record28_raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
record28_post_warmup_sha256, _, _ = _record28_model_fingerprint(record28_raw_model)
record28_optimizer_sha256_after = _record28_optimizer_fingerprint(optimizers)
record28_preconditioner_step = int(getattr(optimizer2, "_precond_step", 0))
record28_accumulators_zero = all(
    torch.count_nonzero(buffer).item() == 0
    for name, buffer in record28_raw_model.named_buffers()
    if "xtx_accum" in name or "xtx_count" in name
)
record28_gradients_clear = all(
    parameter.grad is None for parameter in record28_raw_model.parameters()
)
record28_warmup_reset = {
    "model_sha256": record28_post_warmup_sha256,
    "model_matches_initial": record28_post_warmup_sha256 == record28_init_sha256,
    "optimizer_sha256_before": record28_optimizer_sha256_before,
    "optimizer_sha256_after": record28_optimizer_sha256_after,
    "optimizer_matches_initial": (
        record28_optimizer_sha256_after == record28_optimizer_sha256_before
    ),
    "preconditioner_step": record28_preconditioner_step,
    "preconditioner_step_zero": record28_preconditioner_step == 0,
    "activation_accumulators_zero": record28_accumulators_zero,
    "gradients_clear": record28_gradients_clear,
}
if not all((
    record28_warmup_reset["model_matches_initial"],
    record28_warmup_reset["optimizer_matches_initial"],
    record28_warmup_reset["preconditioner_step_zero"],
    record28_warmup_reset["activation_accumulators_zero"],
    record28_warmup_reset["gradients_clear"],
)):
    raise RuntimeError(f"post-warmup reset audit failed: {record28_warmup_reset}")
print0("RECORD28_WARMUP_RESET " + json.dumps(record28_warmup_reset, sort_keys=True), console=True)
del train_loader, initial_state, initial_optimizer_states, record28_rng_state
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

########################################
"""


WARMUP_STATE_OLD = """warmup_steps = 17
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
"""

WARMUP_STATE_NEW = """warmup_steps = 17
record28_rng_state = dict(
    python=random.getstate(),
    numpy=np.random.get_state(),
    torch=torch.get_rng_state(),
    cuda=torch.cuda.get_rng_state_all(),
)
initial_state = dict(model=copy.deepcopy(model.state_dict()))
# Preserve Parameter identities and arbitrary custom state values exactly.
# Optimizer.load_state_dict recursively transforms iterable metadata such as
# the Newton-state "kind" labels and is therefore not a valid warmup restore
# mechanism for this custom optimizer state.
initial_optimizer_states = [
    [
        (parameter, copy.deepcopy(state))
        for parameter, state in optimizer.state.items()
    ]
    for optimizer in optimizers
]
record28_optimizer_sha256_before = _record28_optimizer_fingerprint(optimizers)
"""

WARMUP_RESTORE_OLD = """for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
"""

WARMUP_RESTORE_NEW = """for optimizer, optimizer_states in zip(optimizers, initial_optimizer_states):
    optimizer.state.clear()
    for parameter, state in optimizer_states:
        optimizer.state[parameter] = state
"""


CHECKPOINT_OLD = """        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
"""

CHECKPOINT_NEW = """        if master_process and args.save_checkpoint:
            log = dict(
                step=step,
                code=code,
                model=record28_raw_model.state_dict(),
                record28_metadata=record28_metadata,
                checkpoint_scope="model_only",
                optimizer_resume_supported=False,
                preconditioner_step=int(getattr(optimizer2, "_precond_step", 0)),
            )
            checkpoint_path = os.path.join(RECORD28_OUTPUT_DIR, f"state_step{step:06d}.pt")
            temporary_checkpoint = checkpoint_path + ".tmp"
            torch.save(log, temporary_checkpoint)
            os.replace(temporary_checkpoint, checkpoint_path)
            print0(f"RECORD28_CHECKPOINT {checkpoint_path}", console=True)
"""


LAST_STEP_OLD = """    if last_step:
        if master_process and args.save_checkpoint:
"""

LAST_STEP_NEW = """    if last_step:
        record28_training_peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
        record28_training_peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
        if master_process and args.save_checkpoint:
"""


VALIDATION_LOG_OLD = """        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
"""

VALIDATION_LOG_NEW = VALIDATION_LOG_OLD + """        record28_validation = {
            "step": int(step),
            "total_steps": int(train_steps),
            "val_loss": float(val_loss.item()),
            "train_time_ms": int(round(training_time_ms)),
            "step_avg_ms": float(training_time_ms / max(step, 1)),
            "tokens": int(step * world_size * grad_accum_steps * args.train_seq_len),
        }
        print0("RECORD28_VAL " + json.dumps(record28_validation, sort_keys=True), console=True)
"""


FINAL_MEMORY_OLD = """print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()"""

FINAL_MEMORY_NEW = """if master_process:
    optimizer_tensors = []
    for optimizer in optimizers:
        optimizer_tensors.extend(_record28_iter_tensors(optimizer.state))
    k_cov_tensors = []
    k_inv_tensors = []
    k_workspace_tensors = []
    for state in optimizer2.state.values():
        if isinstance(state, dict):
            if isinstance(state.get("cov"), Tensor):
                k_cov_tensors.append(state["cov"])
            if isinstance(state.get("inv"), Tensor):
                k_inv_tensors.append(state["inv"])
            for key in ("Kbuf", "precond_buf"):
                if isinstance(state.get(key), Tensor):
                    k_workspace_tensors.append(state[key])
    optimizer_runtime_cache_tensors = []
    for cache_name in ("_eye_cache", "_eye4_cache"):
        cache = getattr(optimizer2, cache_name, {})
        if isinstance(cache, dict):
            optimizer_runtime_cache_tensors.extend(
                tensor for tensor in cache.values() if isinstance(tensor, Tensor)
            )
    cproj_cov_tensors = []
    cproj_inv_tensors = []
    cproj_workspace_tensors = []
    cproj_kinds = []
    for state in optimizer2.state.values():
        if not isinstance(state, dict) or state.get("kind") not in ("c_proj", "c_proj_diag"):
            continue
        cproj_kinds.append(state["kind"])
        if isinstance(state.get("cov"), Tensor):
            cproj_cov_tensors.append(state["cov"])
        if isinstance(state.get("inv"), Tensor):
            cproj_inv_tensors.append(state["inv"])
        for key in ("Kbuf", "precond_buf"):
            if isinstance(state.get(key), Tensor):
                cproj_workspace_tensors.append(state[key])
    activation_stat_tensors = []
    activation_workspace_tensors = []
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    for name, tensor in raw_model.named_buffers():
        if "xtx_accum" in name or "xtx_count" in name:
            activation_stat_tensors.append(tensor)
        elif "xtx_tmp" in name:
            activation_workspace_tensors.append(tensor)
    cproj_activation_stat_tensors = [
        tensor for name, tensor in raw_model.named_buffers()
        if "precond_proj_xtx_accum" in name or "precond_proj_xtx_count" in name
    ]
    cproj_activation_workspace_tensors = [
        tensor for name, tensor in raw_model.named_buffers()
        if "precond_proj_xtx_tmp" in name
    ]
    preconditioner_step = int(getattr(optimizer2, "_precond_step", 0))
    final_audit = {
        "optimizer_state_bytes": _record28_unique_storage_bytes(optimizer_tensors),
        "optimizer_runtime_cache_bytes": _record28_unique_storage_bytes(optimizer_runtime_cache_tensors),
        "optimizer_runtime_total_bytes": _record28_unique_storage_bytes(optimizer_tensors + optimizer_runtime_cache_tensors),
        "k_cov_bytes": _record28_unique_storage_bytes(k_cov_tensors),
        "k_inv_bytes": _record28_unique_storage_bytes(k_inv_tensors),
        "k_state_bytes": _record28_unique_storage_bytes(k_cov_tensors + k_inv_tensors),
        "k_workspace_bytes": _record28_unique_storage_bytes(k_workspace_tensors),
        "activation_stat_bytes": _record28_unique_storage_bytes(activation_stat_tensors),
        "activation_workspace_bytes": _record28_unique_storage_bytes(activation_workspace_tensors),
        "cproj_k_kind": sorted(set(cproj_kinds)),
        "cproj_k_parameter_count": len(cproj_kinds),
        "cproj_cov_bytes": _record28_unique_storage_bytes(cproj_cov_tensors),
        "cproj_inv_bytes": _record28_unique_storage_bytes(cproj_inv_tensors),
        "cproj_workspace_bytes": _record28_unique_storage_bytes(cproj_workspace_tensors),
        "cproj_activation_stat_bytes": _record28_unique_storage_bytes(cproj_activation_stat_tensors),
        "cproj_activation_workspace_bytes": _record28_unique_storage_bytes(cproj_activation_workspace_tensors),
        "cproj_cov_shapes": [list(tensor.shape) for tensor in cproj_cov_tensors],
        "cproj_inv_shapes": [list(tensor.shape) for tensor in cproj_inv_tensors],
        "peak_memory_allocated_bytes": record28_training_peak_allocated_bytes,
        "peak_memory_reserved_bytes": record28_training_peak_reserved_bytes,
        "preconditioner_step": preconditioner_step,
        "refresh_count": int(getattr(optimizer2, "_record28_refresh_count", 0)),
        "first_refresh_zero_based_step": 15 if hasattr(optimizer2, "_precond_step") else None,
        "all_finite": all(torch.isfinite(parameter).all().item() for parameter in raw_model.parameters()),
        "k_tensors_all_finite": all(
            torch.isfinite(tensor).all().item()
            for tensor in (k_cov_tensors + k_inv_tensors)
        ),
    }
    final_audit["total_preconditioner_bytes"] = _record28_unique_storage_bytes(
        k_cov_tensors
        + k_inv_tensors
        + k_workspace_tensors
        + optimizer_runtime_cache_tensors
        + activation_stat_tensors
        + activation_workspace_tensors
    )
    print0("RECORD28_FINAL_AUDIT " + json.dumps(final_audit, sort_keys=True), console=True)
    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
           f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()"""


DIAG_CUSTOM_OP = """
@torch.compile(dynamic=False, fullgraph=True)
def _accum_xtx_diag4_impl_v3(x_2d: Tensor, accum: Tensor, count: Tensor) -> None:
    with torch.no_grad():
        token_count = x_2d.size(0)
        d = x_2d.size(1) // 4
        blocks = x_2d.view(token_count, 4, d)
        diagonal_sum = blocks.square().sum(dim=0, dtype=torch.float32)
        accum.add_(diagonal_sum, alpha=1.0 / token_count)
        count.add_(1)

@torch.library.custom_op("nanogpt::accum_xtx_diag4_v3", mutates_args=("accum", "count"))
def accum_xtx_diag4_op_v3(x_2d: Tensor, accum: Tensor, count: Tensor) -> Tensor:
    _accum_xtx_diag4_impl_v3(x_2d, accum, count)
    return _dummy_scalar_like(accum)

@accum_xtx_diag4_op_v3.register_fake
def _(x_2d: Tensor, accum: Tensor, count: Tensor):
    return accum.new_empty(())
"""


DIAG_INSERT_OLD = """@accum_xtx_blocks4_op_v3.register_fake
def _(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor):
    return accum.new_empty(())
"""

DIAG_INSERT_NEW = DIAG_INSERT_OLD + DIAG_CUSTOM_OP


CPROJ_ATTACH_OLD = """            # ---- MLP c_proj ----
            p = block.mlp.c_proj
            d = block.mlp.model_dim
            st = self.state[p]
            st["kind"] = "c_proj"
            st["d"] = d

            I = self._get_eye(d, p.device)
            if ("cov" not in st) or (st["cov"].shape != (4, d, d)) or (st["cov"].device != p.device):
                st["cov"] = (PRECOND_INIT_DIAG * I).unsqueeze(0).repeat(4, 1, 1).contiguous()
            if ("inv" not in st) or (st["inv"].shape != (4, d, d)) or (st["inv"].device != p.device):
                st["inv"] = I.unsqueeze(0).repeat(4, 1, 1).contiguous()
            if ("Kbuf" not in st) or (st["Kbuf"].shape != (4, d, d)) or (st["Kbuf"].device != p.device):
                st["Kbuf"] = torch.empty((4, d, d), device=p.device, dtype=torch.float32)
            if ("precond_buf" not in st) or (st["precond_buf"].shape != p.shape) or (st["precond_buf"].device != p.device) or (st["precond_buf"].dtype != p.dtype):
                st["precond_buf"] = torch.empty_like(p)

            self._precond_map[p] = dict(
                kind="c_proj",
                d=d,
                xtx=block.mlp.precond_proj_xtx_accum,
                cnt=block.mlp.precond_proj_xtx_count,
            )
"""

CPROJ_ATTACH_NEW = """            # ---- MLP c_proj: the only experiment-43 Newton-family intervention ----
            p = block.mlp.c_proj
            d = block.mlp.model_dim
            if RECORD28_CPROJ_K_MODE == "block4":
                st = self.state[p]
                st["kind"] = "c_proj"
                st["d"] = d
                I = self._get_eye(d, p.device)
                if ("cov" not in st) or (st["cov"].shape != (4, d, d)) or (st["cov"].device != p.device):
                    st["cov"] = (PRECOND_INIT_DIAG * I).unsqueeze(0).repeat(4, 1, 1).contiguous()
                if ("inv" not in st) or (st["inv"].shape != (4, d, d)) or (st["inv"].device != p.device):
                    st["inv"] = I.unsqueeze(0).repeat(4, 1, 1).contiguous()
                if ("Kbuf" not in st) or (st["Kbuf"].shape != (4, d, d)) or (st["Kbuf"].device != p.device):
                    st["Kbuf"] = torch.empty((4, d, d), device=p.device, dtype=torch.float32)
                if ("precond_buf" not in st) or (st["precond_buf"].shape != p.shape) or (st["precond_buf"].device != p.device) or (st["precond_buf"].dtype != p.dtype):
                    st["precond_buf"] = torch.empty_like(p)
                self._precond_map[p] = dict(
                    kind="c_proj", d=d,
                    xtx=block.mlp.precond_proj_xtx_accum,
                    cnt=block.mlp.precond_proj_xtx_count,
                )
            elif RECORD28_CPROJ_K_MODE == "diag":
                st = self.state[p]
                st["kind"] = "c_proj_diag"
                st["d"] = d
                if ("cov" not in st) or (st["cov"].shape != (4, d)) or (st["cov"].device != p.device):
                    st["cov"] = torch.full((4, d), PRECOND_INIT_DIAG, device=p.device, dtype=torch.float32)
                if ("inv" not in st) or (st["inv"].shape != (4, d)) or (st["inv"].device != p.device):
                    st["inv"] = torch.ones((4, d), device=p.device, dtype=torch.float32)
                if ("precond_buf" not in st) or (st["precond_buf"].shape != p.shape) or (st["precond_buf"].device != p.device) or (st["precond_buf"].dtype != p.dtype):
                    st["precond_buf"] = torch.empty_like(p)
                self._precond_map[p] = dict(
                    kind="c_proj_diag", d=d,
                    xtx=block.mlp.precond_proj_xtx_accum,
                    cnt=block.mlp.precond_proj_xtx_count,
                )
            elif RECORD28_CPROJ_K_MODE == "none":
                pass
            else:
                raise ValueError(f"unsupported c_proj K mode={RECORD28_CPROJ_K_MODE!r}")
"""


CPROJ_REFRESH_OLD = """        if kind == "c_proj":
            xtx: Tensor = ref["xtx"]
            cnt: Tensor = ref["cnt"]
            denom = cnt.to(torch.float32).clamp_min(1.0)
            xtx.mul_(denom.reciprocal())
            cnt.zero_()

            cov: Tensor = st["cov"]
            inv: Tensor = st["inv"]
            Kbuf: Tensor = st["Kbuf"]

            cov.mul_(PRECOND_EWMA).add_(xtx, alpha=(1.0 - PRECOND_EWMA))

            tr = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
            ridge = tr / float(d) * 0.2

            Kbuf.copy_(cov)
            Kbuf.diagonal(dim1=-2, dim2=-1).add_(ridge[:, None])

            eye4 = self._get_eye4(d, p.device)
            self._chol_inv_inplace(Kbuf, eye4, inv)

            xtx.zero_()
            return
"""

CPROJ_REFRESH_NEW = CPROJ_REFRESH_OLD + """
        if kind == "c_proj_diag":
            xtx: Tensor = ref["xtx"]
            cnt: Tensor = ref["cnt"]
            denom = cnt.to(torch.float32).clamp_min(1.0)
            xtx.mul_(denom.reciprocal())
            cnt.zero_()
            cov: Tensor = st["cov"]
            inv: Tensor = st["inv"]
            cov.mul_(PRECOND_EWMA).add_(xtx, alpha=(1.0 - PRECOND_EWMA))
            ridge = cov.mean(dim=-1) * 0.2
            inv.copy_((cov + ridge.unsqueeze(-1)).reciprocal())
            xtx.zero_()
            return
"""


CPROJ_APPLY_OLD = """        if kind == "c_proj":
            d = ref["d"]
            for i in range(4):
                gblk = grad[:, i*d:(i+1)*d]
                oblk = buf[:, i*d:(i+1)*d]
                torch.mm(gblk, inv[i], out=oblk)
            return buf
"""

CPROJ_APPLY_NEW = CPROJ_APPLY_OLD + """        if kind == "c_proj_diag":
            d = ref["d"]
            buf.copy_(grad)
            buf.view(d, 4, d).mul_(inv.unsqueeze(0))
            return buf
"""


REFRESH_COUNTER_INIT_OLD = """        self._precond_step = 0
        self._precond_attached = False
"""

REFRESH_COUNTER_INIT_NEW = """        self._precond_step = 0
        self._record28_refresh_count = 0
        self._precond_attached = False
"""


REFRESH_COUNTER_RESET_OLD = """    def reset_precond_step(self):
        self._precond_step = 0
"""

REFRESH_COUNTER_RESET_NEW = """    def reset_precond_step(self):
        self._precond_step = 0
        self._record28_refresh_count = 0
"""


REFRESH_COUNTER_STEP_OLD = """        do_refresh = self._precond_attached and (self._precond_step % PRECOND_EVERY == PRECOND_EVERY - 1)

        reduce_scatter_futures: list[torch.Future] = []
"""

REFRESH_COUNTER_STEP_NEW = """        do_refresh = self._precond_attached and (self._precond_step % PRECOND_EVERY == PRECOND_EVERY - 1)
        if do_refresh:
            self._record28_refresh_count += 1

        reduce_scatter_futures: list[torch.Future] = []
"""


MLP_STATE_OLD = """        self.precond_proj_xtx_accum = nn.Buffer(torch.zeros((4, dim, dim), dtype=torch.float32), persistent=False)
        self.precond_proj_xtx_tmp   = nn.Buffer(torch.empty((4, dim, dim), dtype=torch.float32), persistent=False)
        self.precond_proj_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.int32), persistent=False)
"""

MLP_STATE_NEW = """        if RECORD28_CPROJ_K_MODE == "block4":
            self.precond_proj_xtx_accum = nn.Buffer(torch.zeros((4, dim, dim), dtype=torch.float32), persistent=False)
            self.precond_proj_xtx_tmp = nn.Buffer(torch.empty((4, dim, dim), dtype=torch.float32), persistent=False)
            self.precond_proj_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.int32), persistent=False)
        elif RECORD28_CPROJ_K_MODE == "diag":
            self.precond_proj_xtx_accum = nn.Buffer(torch.zeros((4, dim), dtype=torch.float32), persistent=False)
            self.precond_proj_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.int32), persistent=False)
        elif RECORD28_CPROJ_K_MODE != "none":
            raise ValueError(f"unsupported c_proj K mode={RECORD28_CPROJ_K_MODE!r}")
"""


MLP_FORWARD_OLD = """        if precond_flag:
            x2d = x.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx_blocks4_v3(x2d, self.precond_proj_xtx_accum, self.precond_proj_xtx_count, self.precond_proj_xtx_tmp)
"""

MLP_FORWARD_NEW = """        if precond_flag and RECORD28_CPROJ_K_MODE != "none":
            x2d = x.flatten(0, -2)
            if RECORD28_CPROJ_K_MODE == "block4":
                torch.ops.nanogpt.accum_xtx_blocks4_v3(x2d, self.precond_proj_xtx_accum, self.precond_proj_xtx_count, self.precond_proj_xtx_tmp)
            else:
                torch.ops.nanogpt.accum_xtx_diag4_v3(x2d, self.precond_proj_xtx_accum, self.precond_proj_xtx_count)
"""


def apply_common(source: str, *, newton: bool) -> str:
    source = replace_once(source, COMMON_IMPORTS_OLD, COMMON_IMPORTS_NEW, "imports")
    source = replace_once(
        source,
        "args = Hyperparameters()\n",
        COMMON_CONTROL
        + (MODE_VALIDATION_NEWTON if newton else MODE_VALIDATION_MUON),
        "controlled hyperparameters",
    )
    source = replace_once(
        source, WORLD_SIZE_OLD, WORLD_SIZE_NEW, "single-GPU world size"
    )
    source = replace_exact_count(
        source,
        "x = step / args.num_iterations # progress in training",
        "x = step / RECORD28_SCHEDULE_ITERATIONS # frozen formal schedule, including smoke prefix",
        "formal schedule denominator",
        2,
    )
    source = replace_once(
        source,
        "master_process = (rank == 0) # this process will do logging, checkpointing etc.\n",
        SEED_AND_AUDIT,
        "seed and audit insertion",
    )
    source = replace_once(source, LOGGING_OLD, LOGGING_NEW, "isolated logfile")
    source = replace_once(
        source,
        MODEL_FINGERPRINT_ANCHOR,
        MODEL_FINGERPRINT_BLOCK,
        "paired model initialization audit",
    )
    source = replace_once(
        source, WARMUP_STATE_OLD, WARMUP_STATE_NEW, "warmup RNG snapshot"
    )
    source = replace_once(
        source,
        WARMUP_RESTORE_OLD,
        WARMUP_RESTORE_NEW,
        "identity-preserving optimizer warmup restore",
    )
    source = replace_once(
        source, VALIDATION_LOG_OLD, VALIDATION_LOG_NEW, "full-precision validation log"
    )
    source = replace_once(
        source, LAST_STEP_OLD, LAST_STEP_NEW, "training-only peak capture"
    )
    source = replace_once(
        source, RESET_PEAK_OLD, RESET_PEAK_NEW, "post-warmup peak reset"
    )
    source = replace_once(
        source, CHECKPOINT_OLD, CHECKPOINT_NEW, "isolated final checkpoint"
    )
    source = replace_once(
        source, FINAL_MEMORY_OLD, FINAL_MEMORY_NEW, "final memory audit"
    )
    return source


def apply_newton_modes(source: str) -> str:
    source = replace_once(
        source, DIAG_INSERT_OLD, DIAG_INSERT_NEW, "diagonal custom op"
    )
    source = replace_once(
        source, CPROJ_ATTACH_OLD, CPROJ_ATTACH_NEW, "c_proj state routing"
    )
    source = replace_once(
        source, CPROJ_REFRESH_OLD, CPROJ_REFRESH_NEW, "diagonal refresh"
    )
    source = replace_once(
        source, CPROJ_APPLY_OLD, CPROJ_APPLY_NEW, "diagonal application"
    )
    source = replace_once(
        source, MLP_STATE_OLD, MLP_STATE_NEW, "c_proj activation state"
    )
    source = replace_once(
        source, MLP_FORWARD_OLD, MLP_FORWARD_NEW, "c_proj accumulation routing"
    )
    source = replace_once(
        source,
        REFRESH_COUNTER_INIT_OLD,
        REFRESH_COUNTER_INIT_NEW,
        "refresh counter initialization",
    )
    source = replace_once(
        source,
        REFRESH_COUNTER_RESET_OLD,
        REFRESH_COUNTER_RESET_NEW,
        "refresh counter reset",
    )
    source = replace_once(
        source,
        REFRESH_COUNTER_STEP_OLD,
        REFRESH_COUNTER_STEP_NEW,
        "refresh counter increment",
    )
    return source


def assert_source_contract(source: str, method: str) -> None:
    required = (
        "RECORD28_METADATA ",
        "RECORD28_FINAL_AUDIT ",
        "record28_parameter_count != 275743572",
        "torch.cuda.reset_peak_memory_stats()",
        "state_step{step:06d}.pt",
        "RECORD28_NUM_ITERATIONS",
        "record28_optimizer_sha256_before",
        "record28_optimizer_sha256_after",
        "optimizer_matches_initial",
        "optimizer.state[parameter] = state",
    )
    missing = [anchor for anchor in required if anchor not in source]
    if missing:
        raise RuntimeError(f"{method}: derived source is missing anchors {missing}")
    if "torch._dynamo.config.compiled_autograd = True" in source:
        raise RuntimeError(
            f"{method}: PyTorch 2.8 compiled-autograd must remain disabled"
        )
    if method == "muon":
        forbidden = (
            "RECORD28_CPROJ_K_MODE == \"diag\"",
            "accum_xtx_diag4_v3",
        )
        present = [anchor for anchor in forbidden if anchor in source]
        if present:
            raise RuntimeError(f"Muon source contains Newton overlays: {present}")
    else:
        required_newton = (
            'RECORD28_CPROJ_K_MODE == "block4"',
            'RECORD28_CPROJ_K_MODE == "diag"',
            'RECORD28_CPROJ_K_MODE == "none"',
            "accum_xtx_diag4_v3",
            'kind == "c_proj_diag"',
        )
        missing = [anchor for anchor in required_newton if anchor not in source]
        if missing:
            raise RuntimeError(f"{method}: missing Newton-mode anchors {missing}")


def build_source(official_repo: Path, method: str) -> DerivedSource:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    newton = method != "muon"
    base_script = (
        "train_gpt_newton_muon_2.py" if newton else "train_gpt_muon_2.py"
    )
    base_raw = git_blob(official_repo, base_script)
    base_sha256 = hashlib.sha256(base_raw).hexdigest()
    expected = EXPECTED_CANONICAL_SHA256[base_script]
    if base_sha256 != expected:
        raise RuntimeError(
            f"{base_script} canonical hash mismatch: expected {expected}, observed {base_sha256}"
        )
    source = base_raw.decode("utf-8")
    source = apply_common(source, newton=newton)
    if newton:
        source = apply_newton_modes(source)
    compile(source, f"<experiment43-{method}>", "exec")
    assert_source_contract(source, method)
    cproj_mode = {
        "muon": "not_applicable",
        "original_newton_muon": "block4",
        "selective_none": "none",
        "selective_diag": "diag",
    }[method]
    diff = "".join(
        difflib.unified_diff(
            base_raw.decode("utf-8").splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile=f"upstream/{base_script}",
            tofile=f"experiment43/train_{method}.py",
        )
    )
    return DerivedSource(
        method=method,
        cproj_k_mode=cproj_mode,
        base_script=base_script,
        base_canonical_sha256=base_sha256,
        derived_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source=source,
        unified_diff=diff,
    )


def build_all_sources(official_repo: Path) -> dict[str, DerivedSource]:
    sources = {method: build_source(official_repo, method) for method in METHODS}
    newton_hashes = {
        sources[method].derived_sha256
        for method in METHODS
        if method != "muon"
    }
    if len(newton_hashes) != 1:
        raise RuntimeError(
            "Newton-family methods must share one environment-parameterized source"
        )
    return sources


def self_test_diag_math() -> None:
    """Check the declared diagonal operation against explicit diagonal matrices."""

    import random

    generator = random.Random(43028)
    samples, d = 19, 7
    x = [
        [generator.gauss(0, 1) for _ in range(4 * d)]
        for _ in range(samples)
    ]
    diag = [
        [
            sum(row[block * d + column] ** 2 for row in x) / samples
            for column in range(d)
        ]
        for block in range(4)
    ]
    dense_diagonal = [
        [
            sum(
                row[block * d + column] * row[block * d + column]
                for row in x
            )
            / samples
            for column in range(d)
        ]
        for block in range(4)
    ]
    if diag != dense_diagonal:
        raise AssertionError("diagonal accumulation differs from dense diagonal")
    gradient = [
        [generator.gauss(0, 1) for _ in range(4 * d)] for _ in range(d)
    ]
    inverse = [
        [generator.random() + 0.1 for _ in range(d)] for _ in range(4)
    ]
    observed = [
        gradient[row][block * d + column] * inverse[block][column]
        for row in range(d)
        for block in range(4)
        for column in range(d)
    ]
    expected = []
    for row in range(d):
        for block in range(4):
            diagonal_matrix = [
                [
                    inverse[block][inner] if inner == column else 0.0
                    for column in range(d)
                ]
                for inner in range(d)
            ]
            for column in range(d):
                expected.append(
                    sum(
                        gradient[row][block * d + inner]
                        * diagonal_matrix[inner][column]
                        for inner in range(d)
                    )
                )
    if any(abs(left - right) > 1e-15 for left, right in zip(observed, expected)):
        raise AssertionError("diagonal right-application contract failed")
