#!/usr/bin/env python3
"""Build the frozen experiment-44 trainer from the vendored Record #17 source.

The Medium-Track Record #17 trainer is the sole algorithmic base.  The four
experiment methods share one generated program and are selected at runtime by
the sealed environment.  Every edit uses a single-occurrence/count-checked
anchor; source drift therefore fails closed before a GPU process is launched.
"""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path

from record17_common import (
    METHODS,
    RECORD17_UPSTREAM_CANONICAL_SHA256,
    audit_vendored_record17,
    canonical_bytes,
)


SCRIPT_VERSION = "2026-07-30.5"
BASE_SCRIPT = "record17_train_gpt_medium.py"
BASE_PATH = Path(__file__).resolve().parent / "upstream" / BASE_SCRIPT


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
            f"Record-17 source expected exactly one {label!r} anchor; observed {count}"
        )
    return source.replace(old, new, 1)


def replace_exact_count(
    source: str, old: str, new: str, label: str, expected_count: int
) -> str:
    count = source.count(old)
    if count != expected_count:
        raise RuntimeError(
            f"Record-17 source expected {expected_count} {label!r} anchors; "
            f"observed {count}"
        )
    return source.replace(old, new)


def replace_region_once(
    source: str, start: str, end: str, replacement: str, label: str
) -> str:
    start_count = source.count(start)
    end_count = source.count(end)
    if start_count != 1 or end_count != 1:
        raise RuntimeError(
            f"Record-17 source expected one {label!r} region; "
            f"start={start_count}, end={end_count}"
        )
    begin = source.index(start)
    finish = source.index(end, begin)
    if finish <= begin:
        raise RuntimeError(f"Record-17 malformed {label!r} region")
    return source[:begin] + replacement + source[finish:]


COMMON_IMPORTS_OLD = """import time
import copy
from dataclasses import dataclass
"""

COMMON_IMPORTS_NEW = """import time
import copy
import random
import hashlib
import json
import glob
import numpy as np
from dataclasses import dataclass
"""


COMPILED_AUTOGRAD_OLD = """torch._dynamo.config.compiled_autograd = True
"""


COMPILED_AUTOGRAD_NEW = """# PyTorch 2.8 cannot nest compiled-autograd's FX trace around this trainer's
# FlexAttention graph.  Keep FlexAttention and torch.compile(model), but use
# the standard AOTAutograd backward path uniformly for all four methods.
torch._dynamo.config.compiled_autograd = False
"""


CONTROL_BLOCK = """args = Hyperparameters()

# Experiment-44 controlled overlay.  All four methods execute this exact
# source.  Within the Newton family the sole algorithmic intervention is the
# representation of the MLP output projection's activation covariance.
RECORD17_METHOD = os.environ["RECORD17_METHOD"]
RECORD17_CPROJ_K_MODE = os.environ["RECORD17_CPROJ_K_MODE"]
RECORD17_SEED = int(os.environ["RECORD17_SEED"])
RECORD17_STAGE = os.environ["RECORD17_STAGE"]
RECORD17_OUTPUT_DIR = os.path.abspath(os.environ["RECORD17_OUTPUT_DIR"])
RECORD17_DATA_PATH = os.path.abspath(os.environ["DATA_PATH"])
RECORD17_CELL_ID = os.environ["RECORD17_CELL_ID"]
RECORD17_NUM_ITERATIONS = int(os.environ["RECORD17_NUM_ITERATIONS"])
RECORD17_SCHEDULE_ITERATIONS = 5960
RECORD17_VAL_LOSS_EVERY = int(os.environ["RECORD17_VAL_LOSS_EVERY"])
RECORD17_VAL_TOKENS = int(os.environ["RECORD17_VAL_TOKENS"])
RECORD17_SAVE_CHECKPOINT = os.environ.get("RECORD17_SAVE_CHECKPOINT", "0") == "1"
RECORD17_INIT_ONLY = os.environ.get("RECORD17_INIT_ONLY", "0") == "1"
RECORD17_GRAD_ACCUM_STEPS = 8
RECORD17_WARMUP_UPDATES = 26
RECORD17_PRECOND_EVERY = 24
RECORD17_PRECOND_EWMA = 0.90
RECORD17_PRECOND_INIT_DIAG = 1e-3
RECORD17_RIDGE_MULT = 0.20
_record17_expected_modes = {
    "muon": "not_applicable",
    "original_newton_muon": "block4",
    "selective_none": "none",
    "selective_diag": "diag",
}
if RECORD17_METHOD not in _record17_expected_modes:
    raise ValueError(f"invalid RECORD17_METHOD={RECORD17_METHOD!r}")
if RECORD17_CPROJ_K_MODE != _record17_expected_modes[RECORD17_METHOD]:
    raise ValueError("method/c_proj K-mode mismatch")
RECORD17_NEWTON_ACTIVE = RECORD17_METHOD != "muon"
if RECORD17_STAGE not in ("smoke", "formal"):
    raise ValueError(f"invalid RECORD17_STAGE={RECORD17_STAGE!r}")
if RECORD17_STAGE == "formal":
    if (
        RECORD17_NUM_ITERATIONS,
        RECORD17_VAL_LOSS_EVERY,
        RECORD17_VAL_TOKENS,
    ) != (5960, 125, 10485760):
        raise ValueError("formal Record-17 budget/protocol drift")
elif RECORD17_NUM_ITERATIONS < 27:
    raise ValueError(
        "smoke must execute at least 27 counted updates, crossing refresh 24"
    )
args.num_iterations = RECORD17_NUM_ITERATIONS
args.val_loss_every = RECORD17_VAL_LOSS_EVERY
args.val_tokens = RECORD17_VAL_TOKENS
args.save_checkpoint = RECORD17_SAVE_CHECKPOINT
if not os.path.isdir(RECORD17_DATA_PATH):
    raise RuntimeError(f"DATA_PATH is not a directory: {RECORD17_DATA_PATH}")
args.train_files = os.path.join(
    RECORD17_DATA_PATH, "data", "fineweb10B", "fineweb_train_*.bin"
)
args.val_files = os.path.join(
    RECORD17_DATA_PATH, "data", "fineweb10B", "fineweb_val_*.bin"
)
os.makedirs(RECORD17_OUTPUT_DIR, exist_ok=True)
"""


WORLD_SIZE_OLD = """world_size = int(os.environ["WORLD_SIZE"])
assert world_size == 8 # this code is designed for 8xH100
"""

WORLD_SIZE_NEW = """world_size = int(os.environ["WORLD_SIZE"])
if world_size != 1:
    raise RuntimeError(
        f"experiment 44 is frozen to single-GPU world_size=1, observed {world_size}"
    )
"""


DATA_GLOB_OLD = """def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = sorted(Path.cwd().glob(filename_pattern))
"""

DATA_GLOB_NEW = """def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    if not os.path.isabs(filename_pattern):
        raise RuntimeError(
            f"data pattern must be absolute: {filename_pattern}"
        )
    files = sorted(Path(path) for path in glob.glob(filename_pattern))
    if not files:
        raise RuntimeError(
            f"no data shards matched absolute pattern: {filename_pattern}"
        )
"""

RELATIVE_DATA_DEFAULTS_OLD = """    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
"""

RELATIVE_DATA_DEFAULTS_NEW = """    # Set fail-closed from the absolute DATA_PATH environment below.
    train_files = None
    val_files = None
"""


SEED_AND_AUDIT = """master_process = (rank == 0) # this process will do logging, checkpointing etc.

# Seed before model construction so every method in a paired seed starts from
# identical parameter values.  Data iteration is deterministic and is rebuilt
# from its first shard after instrumentation warmup.
random.seed(RECORD17_SEED)
np.random.seed(RECORD17_SEED)
torch.manual_seed(RECORD17_SEED)
torch.cuda.manual_seed_all(RECORD17_SEED)

def _record17_tensor_bytes(tensor: Tensor) -> bytes:
    cpu = tensor.detach().contiguous().cpu()
    return cpu.view(torch.uint8).numpy().tobytes()

def _record17_model_fingerprint(module: nn.Module) -> tuple[str, str, int]:
    value_digest = hashlib.sha256()
    structure_digest = hashlib.sha256()
    parameter_count = 0
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            descriptor = f"{name}|{tuple(parameter.shape)}|{parameter.dtype}"
            encoded = descriptor.encode("utf-8")
            value_digest.update(encoded)
            value_digest.update(_record17_tensor_bytes(parameter))
            structure_digest.update(encoded)
            parameter_count += parameter.numel()
    return value_digest.hexdigest(), structure_digest.hexdigest(), parameter_count

def _record17_digest_value(digest, value) -> None:
    if isinstance(value, Tensor):
        digest.update(
            f"tensor|{tuple(value.shape)}|{value.dtype}|{value.device.type}".encode()
        )
        digest.update(_record17_tensor_bytes(value))
    elif isinstance(value, dict):
        digest.update(b"dict{")
        for key in sorted(value, key=lambda item: repr(item)):
            _record17_digest_value(digest, key)
            _record17_digest_value(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + b"[")
        for child in value:
            _record17_digest_value(digest, child)
        digest.update(b"]")
    else:
        digest.update(f"{type(value).__name__}:{value!r}".encode())

def _record17_optimizer_fingerprint(optimizers) -> str:
    digest = hashlib.sha256()
    for optimizer in optimizers:
        _record17_digest_value(digest, optimizer.state_dict())
    return digest.hexdigest()

def _record17_rng_fingerprint() -> str:
    digest = hashlib.sha256()
    _record17_digest_value(digest, random.getstate())
    numpy_state = np.random.get_state()
    _record17_digest_value(digest, numpy_state[0])
    digest.update(numpy_state[1].tobytes())
    _record17_digest_value(digest, numpy_state[2:])
    _record17_digest_value(digest, torch.get_rng_state())
    _record17_digest_value(digest, torch.cuda.get_rng_state_all())
    return digest.hexdigest()

def _record17_iter_tensors(value):
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _record17_iter_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _record17_iter_tensors(child)

def _record17_storage_key(tensor: Tensor):
    storage = tensor.untyped_storage()
    index = tensor.device.index if tensor.device.index is not None else -1
    return tensor.device.type, index, storage.data_ptr(), storage.nbytes()

def _record17_unique_storage_bytes(tensors) -> int:
    storages = {}
    for tensor in tensors:
        key = _record17_storage_key(tensor)
        storages[key] = key[-1]
    return int(sum(storages.values()))
"""


LOGGING_OLD = """if master_process:
    run_id_full = f"{run_id:03d}_{uuid.uuid4()}"
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id_full}.txt"
    print(logfile)
"""

LOGGING_NEW = """if master_process:
    run_id_full = RECORD17_CELL_ID
    logfile = os.path.join(RECORD17_OUTPUT_DIR, "training.log")
    print(logfile)
"""


MODEL_FINGERPRINT_ANCHOR = """for param in model.parameters():
    dist.broadcast(param.detach(), 0)
"""

MODEL_FINGERPRINT_BLOCK = MODEL_FINGERPRINT_ANCHOR + """
record17_init_sha256, record17_structure_sha256, record17_parameter_count = (
    _record17_model_fingerprint(model)
)
if record17_parameter_count != 454496336:
    raise RuntimeError(f"parameter-count drift: {record17_parameter_count}")
record17_metadata = {
    "schema_version": 1,
    "method": RECORD17_METHOD,
    "cproj_k_mode": RECORD17_CPROJ_K_MODE,
    "compiled_autograd_enabled": bool(
        torch._dynamo.config.compiled_autograd
    ),
    "seed": RECORD17_SEED,
    "stage": RECORD17_STAGE,
    "init_sha256": record17_init_sha256,
    "parameter_structure_sha256": record17_structure_sha256,
    "parameter_count": record17_parameter_count,
    "train_iterations": args.num_iterations,
    "schedule_iterations": RECORD17_SCHEDULE_ITERATIONS,
    "train_seq_len": args.train_seq_len,
    "val_seq_len": args.val_seq_len,
    "grad_accum_steps": RECORD17_GRAD_ACCUM_STEPS,
    "tokens_per_update": (
        world_size * RECORD17_GRAD_ACCUM_STEPS * args.train_seq_len
    ),
    "loader_global_batch_tokens": (
        world_size * RECORD17_GRAD_ACCUM_STEPS * args.train_seq_len
    ),
    "loader_split_microbatches": RECORD17_GRAD_ACCUM_STEPS,
    "global_batch_loaded_before_split": True,
    "newton_k_statistics_scope": (
        "all_8_sequential_microbatches_per_counted_update_on_single_h100"
    ),
    "newton_k_statistics_tokens_per_counted_update": (
        world_size * RECORD17_GRAD_ACCUM_STEPS * args.train_seq_len
    ),
    "strict_hypothetical_8rank_owner_local_k_equivalence_claimed": False,
    "data_path": RECORD17_DATA_PATH,
    "train_file_pattern": args.train_files,
    "validation_file_pattern": args.val_files,
    "world_size": world_size,
    # These are frozen method-family recipe constants.  Muon reports them for
    # paired protocol audit while allocating/applying no preconditioner state.
    "precondition_refresh_every": RECORD17_PRECOND_EVERY,
    "precondition_ewma": RECORD17_PRECOND_EWMA,
    "precondition_ridge_multiplier": RECORD17_RIDGE_MULT,
}
print0("RECORD17_METADATA " + json.dumps(record17_metadata, sort_keys=True), console=True)
if RECORD17_INIT_ONLY:
    dist.destroy_process_group()
    raise SystemExit(0)
"""


CUSTOM_OPS_AND_OPTIMIZER_START = """# -----------------------------------------------------------------------------
# Experiment-44 FP32 activation-statistics custom operators.

@torch.library.custom_op(
    "record17::accum_xtx_dense_v1",
    mutates_args=("accum", "count", "workspace"),
)
def record17_accum_xtx_dense_v1(
    x_2d: Tensor, accum: Tensor, count: Tensor, workspace: Tensor
) -> Tensor:
    if x_2d.ndim != 2 or accum.dtype != torch.float32:
        raise RuntimeError("invalid dense XTX custom-op contract")
    with torch.no_grad():
        gram = torch.mm(x_2d.mT, x_2d, out_dtype=torch.float32)
        workspace.copy_(gram).mul_(1.0 / x_2d.size(0))
        accum.add_(workspace)
        count.add_(1)
    return accum.new_empty(())

@record17_accum_xtx_dense_v1.register_fake
def _(x_2d: Tensor, accum: Tensor, count: Tensor, workspace: Tensor):
    return accum.new_empty(())

@torch.library.custom_op(
    "record17::accum_xtx_blocks4_v1",
    mutates_args=("accum", "count", "workspace"),
)
def record17_accum_xtx_blocks4_v1(
    x_2d: Tensor, accum: Tensor, count: Tensor, workspace: Tensor
) -> Tensor:
    if x_2d.ndim != 2 or x_2d.size(1) % 4 != 0:
        raise RuntimeError("invalid block4 XTX custom-op contract")
    with torch.no_grad():
        token_count = x_2d.size(0)
        d = x_2d.size(1) // 4
        blocks = x_2d.view(token_count, 4, d).transpose(0, 1)
        gram = torch.bmm(blocks.mT, blocks, out_dtype=torch.float32)
        workspace.copy_(gram).mul_(1.0 / token_count)
        accum.add_(workspace)
        count.add_(1)
    return accum.new_empty(())

@record17_accum_xtx_blocks4_v1.register_fake
def _(x_2d: Tensor, accum: Tensor, count: Tensor, workspace: Tensor):
    return accum.new_empty(())

@torch.library.custom_op(
    "record17::accum_xtx_diag4_v1",
    mutates_args=("accum", "count"),
)
def record17_accum_xtx_diag4_v1(
    x_2d: Tensor, accum: Tensor, count: Tensor
) -> Tensor:
    if x_2d.ndim != 2 or x_2d.size(1) % 4 != 0:
        raise RuntimeError("invalid diag4 XTX custom-op contract")
    with torch.no_grad():
        token_count = x_2d.size(0)
        d = x_2d.size(1) // 4
        blocks_fp32 = x_2d.view(token_count, 4, d).float()
        accum.add_(
            blocks_fp32.square().sum(dim=0), alpha=1.0 / token_count
        )
        count.add_(1)
    return accum.new_empty(())

@record17_accum_xtx_diag4_v1.register_fake
def _(x_2d: Tensor, accum: Tensor, count: Tensor):
    return accum.new_empty(())

# -----------------------------------------------------------------------------
# Muon optimizer with an optional Newton activation-covariance preconditioner.

"""


OPTIMIZER_CLASS = r'''class Muon(torch.optim.Optimizer):
    """Record-17 Muon, optionally preceded by the frozen Newton K transform."""

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        rank=0,
        world_size=1,
    ):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        assert all(
            p.dtype == torch.bfloat16
            for group in self.param_groups
            for p in group["params"]
        )
        self._precond_step = 0
        self._record17_refresh_count = 0
        self._precond_attached = False
        self._precond_map: dict[Tensor, dict] = {}
        self._eye_cache: dict[tuple[int, torch.device], Tensor] = {}
        self._eye4_cache: dict[tuple[int, torch.device], Tensor] = {}
        self._record17_fp32_apply_count = 0
        self._record17_raw_grad_dtypes: set[str] = set()
        self._record17_preconditioned_grad_dtypes: set[str] = set()

    def reset_precond_step(self):
        self._precond_step = 0
        self._record17_refresh_count = 0
        self._record17_fp32_apply_count = 0
        self._record17_raw_grad_dtypes.clear()
        self._record17_preconditioned_grad_dtypes.clear()

    def _get_eye(self, d: int, device) -> Tensor:
        key = (d, device)
        eye = self._eye_cache.get(key)
        if eye is None:
            eye = torch.eye(d, device=device, dtype=torch.float32)
            self._eye_cache[key] = eye
        return eye

    def _get_eye4(self, d: int, device) -> Tensor:
        key = (d, device)
        eye4 = self._eye4_cache.get(key)
        if eye4 is None:
            eye4 = (
                torch.eye(d, device=device, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(4, 1, 1)
                .contiguous()
            )
            self._eye4_cache[key] = eye4
        return eye4

    def _ensure_dense_state(
        self, p: Tensor, state: dict, shape: tuple[int, ...], kind: str
    ) -> None:
        state["kind"] = kind
        if (
            "cov" not in state
            or tuple(state["cov"].shape) != shape
            or state["cov"].device != p.device
            or state["cov"].dtype != torch.float32
        ):
            if len(shape) == 3:
                _, d, _ = shape
                state["cov"] = (
                    RECORD17_PRECOND_INIT_DIAG
                    * self._get_eye(d, p.device).unsqueeze(0)
                ).repeat(shape[0], 1, 1).contiguous()
            else:
                d = shape[0]
                state["cov"] = (
                    RECORD17_PRECOND_INIT_DIAG
                    * self._get_eye(d, p.device)
                ).contiguous()
        if (
            "inv" not in state
            or tuple(state["inv"].shape) != shape
            or state["inv"].device != p.device
            or state["inv"].dtype != torch.float32
        ):
            if len(shape) == 3:
                _, d, _ = shape
                state["inv"] = (
                    self._get_eye(d, p.device).unsqueeze(0)
                    .repeat(shape[0], 1, 1).contiguous()
                )
            else:
                state["inv"] = self._get_eye(shape[0], p.device).clone()
        if (
            "Kbuf" not in state
            or tuple(state["Kbuf"].shape) != shape
            or state["Kbuf"].device != p.device
            or state["Kbuf"].dtype != torch.float32
        ):
            state["Kbuf"] = torch.empty(
                shape, device=p.device, dtype=torch.float32
            )
        if (
            "precond_buf" not in state
            or state["precond_buf"].shape != p.shape
            or state["precond_buf"].device != p.device
            or state["precond_buf"].dtype != torch.float32
        ):
            # Hidden parameters/gradients are BF16, but G K^-1 and this
            # workspace are deliberately FP32.
            state["precond_buf"] = torch.empty(
                p.shape, device=p.device, dtype=torch.float32
            )

    @torch.no_grad()
    def attach_preconditioner(self, model: nn.Module):
        if not RECORD17_NEWTON_ACTIVE:
            raise RuntimeError("Muon must not attach or allocate Newton K state")
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        self._precond_map.clear()
        for block in model.blocks:
            if block.attn is not None:
                p = block.attn.qkvo_w
                d = p.size(-1)
                state = self.state[p]
                self._ensure_dense_state(p, state, (4, d, d), "qkvo")
                state["d"] = d
                self._precond_map[p] = {
                    "kind": "qkvo",
                    "d": d,
                    "xtx_qkv": block.attn.precond_qkv_xtx_accum,
                    "cnt_qkv": block.attn.precond_qkv_xtx_count,
                    "xtx_o": block.attn.precond_o_xtx_accum,
                    "cnt_o": block.attn.precond_o_xtx_count,
                }

            p = block.mlp.fc_w
            d = p.size(-1)
            state = self.state[p]
            self._ensure_dense_state(p, state, (d, d), "fc_w")
            state["d"] = d
            self._precond_map[p] = {
                "kind": "fc_w",
                "d": d,
                "xtx": block.mlp.precond_fc_xtx_accum,
                "cnt": block.mlp.precond_fc_xtx_count,
            }

            p = block.mlp.proj_w
            d = p.size(0)
            if RECORD17_CPROJ_K_MODE == "block4":
                state = self.state[p]
                self._ensure_dense_state(
                    p, state, (4, d, d), "proj_w_block4"
                )
                state["d"] = d
                self._precond_map[p] = {
                    "kind": "proj_w_block4",
                    "d": d,
                    "xtx": block.mlp.precond_proj_xtx_accum,
                    "cnt": block.mlp.precond_proj_xtx_count,
                }
            elif RECORD17_CPROJ_K_MODE == "diag":
                state = self.state[p]
                state["kind"] = "proj_w_diag"
                state["d"] = d
                if (
                    "cov" not in state
                    or state["cov"].shape != (4, d)
                    or state["cov"].device != p.device
                    or state["cov"].dtype != torch.float32
                ):
                    state["cov"] = torch.full(
                        (4, d),
                        RECORD17_PRECOND_INIT_DIAG,
                        device=p.device,
                        dtype=torch.float32,
                    )
                if (
                    "inv" not in state
                    or state["inv"].shape != (4, d)
                    or state["inv"].device != p.device
                    or state["inv"].dtype != torch.float32
                ):
                    state["inv"] = torch.ones(
                        (4, d), device=p.device, dtype=torch.float32
                    )
                if (
                    "precond_buf" not in state
                    or state["precond_buf"].shape != p.shape
                    or state["precond_buf"].device != p.device
                    or state["precond_buf"].dtype != torch.float32
                ):
                    state["precond_buf"] = torch.empty(
                        p.shape, device=p.device, dtype=torch.float32
                    )
                self._precond_map[p] = {
                    "kind": "proj_w_diag",
                    "d": d,
                    "xtx": block.mlp.precond_proj_xtx_accum,
                    "cnt": block.mlp.precond_proj_xtx_count,
                }
            elif RECORD17_CPROJ_K_MODE == "none":
                pass
            else:
                raise ValueError(
                    f"unsupported c_proj K mode={RECORD17_CPROJ_K_MODE!r}"
                )
        self._precond_attached = True

    @torch.no_grad()
    def _chol_inverse(self, Kbuf: Tensor, out_inv: Tensor) -> None:
        if Kbuf.dtype != torch.float32 or out_inv.dtype != torch.float32:
            raise RuntimeError("inverse path must remain FP32")
        factor, info = torch.linalg.cholesky_ex(Kbuf, check_errors=False)
        if torch.count_nonzero(info).item() != 0:
            raise RuntimeError(
                f"FP32 Cholesky failed with info={info.detach().cpu().tolist()}"
            )
        torch.cholesky_inverse(factor, upper=False, out=out_inv)

    @torch.no_grad()
    def _refresh_one(self, p: Tensor, state: dict) -> None:
        ref = self._precond_map[p]
        kind = ref["kind"]
        d = ref["d"]
        cov: Tensor = state["cov"]
        inv: Tensor = state["inv"]

        if kind == "qkvo":
            qkv = ref["xtx_qkv"]
            qkv_count = ref["cnt_qkv"]
            out = ref["xtx_o"]
            out_count = ref["cnt_o"]
            if qkv_count.item() == 0 or out_count.item() == 0:
                raise RuntimeError("qkvo refresh has no activation samples")
            qkv.mul_(qkv_count.to(torch.float32).reciprocal())
            out.mul_(out_count.to(torch.float32).reciprocal())
            qkv_count.zero_()
            out_count.zero_()
            cov[:3].mul_(RECORD17_PRECOND_EWMA).add_(
                qkv.unsqueeze(0), alpha=1.0 - RECORD17_PRECOND_EWMA
            )
            cov[3].mul_(RECORD17_PRECOND_EWMA).add_(
                out, alpha=1.0 - RECORD17_PRECOND_EWMA
            )
            ridge = (
                cov.diagonal(dim1=-2, dim2=-1).mean(-1)
                * RECORD17_RIDGE_MULT
            )
            Kbuf = state["Kbuf"]
            Kbuf[0].copy_(cov[0])
            Kbuf[0].diagonal().add_(ridge[0])
            Kbuf[3].copy_(cov[3])
            Kbuf[3].diagonal().add_(ridge[3])
            self._chol_inverse(Kbuf[0], inv[0])
            inv[1].copy_(inv[0])
            inv[2].copy_(inv[0])
            self._chol_inverse(Kbuf[3], inv[3])
            qkv.zero_()
            out.zero_()
            return

        xtx = ref["xtx"]
        count = ref["cnt"]
        if count.item() == 0:
            raise RuntimeError(f"{kind} refresh has no activation samples")
        xtx.mul_(count.to(torch.float32).reciprocal())
        count.zero_()
        cov.mul_(RECORD17_PRECOND_EWMA).add_(
            xtx, alpha=1.0 - RECORD17_PRECOND_EWMA
        )
        if kind == "proj_w_diag":
            ridge = cov.mean(-1) * RECORD17_RIDGE_MULT
            inv.copy_((cov + ridge.unsqueeze(-1)).reciprocal())
        else:
            ridge = (
                cov.diagonal(dim1=-2, dim2=-1).mean(-1)
                * RECORD17_RIDGE_MULT
            )
            Kbuf = state["Kbuf"]
            Kbuf.copy_(cov)
            if Kbuf.ndim == 3:
                Kbuf.diagonal(dim1=-2, dim2=-1).add_(ridge.unsqueeze(-1))
            else:
                Kbuf.diagonal().add_(ridge)
            self._chol_inverse(Kbuf, inv)
        xtx.zero_()

    @torch.no_grad()
    def _apply_inv(self, p: Tensor, raw_grad: Tensor, state: dict) -> Tensor:
        ref = self._precond_map[p]
        kind = ref["kind"]
        inv: Tensor = state["inv"]
        buf: Tensor = state["precond_buf"]
        self._record17_raw_grad_dtypes.add(str(raw_grad.dtype))
        if raw_grad.dtype != torch.bfloat16:
            raise RuntimeError(
                f"hidden raw gradient must be BF16, observed {raw_grad.dtype}"
            )
        if inv.dtype != torch.float32 or buf.dtype != torch.float32:
            raise RuntimeError("inverse and precondition workspace must be FP32")
        grad_fp32 = raw_grad.float()
        if kind == "qkvo":
            torch.bmm(grad_fp32, inv, out=buf)
        elif kind == "fc_w":
            # fc_w is (4d,d): activation covariance acts on the right.
            torch.mm(grad_fp32, inv, out=buf)
        elif kind == "proj_w_block4":
            # proj_w is (d,4d): apply four independent dxd right blocks.
            d = ref["d"]
            for block in range(4):
                torch.mm(
                    grad_fp32[:, block * d : (block + 1) * d],
                    inv[block],
                    out=buf[:, block * d : (block + 1) * d],
                )
        elif kind == "proj_w_diag":
            d = ref["d"]
            buf.copy_(grad_fp32)
            buf.view(d, 4, d).mul_(inv.unsqueeze(0))
        else:
            raise RuntimeError(f"unknown precondition kind={kind!r}")
        if buf.dtype != torch.float32:
            raise RuntimeError("G K^-1 output must be FP32")
        self._record17_fp32_apply_count += 1
        self._record17_preconditioned_grad_dtypes.add(str(buf.dtype))
        return buf

    @torch.no_grad()
    def step(self):
        futures: list[torch.Future] = []
        do_refresh = (
            self._precond_attached
            and self._precond_step % RECORD17_PRECOND_EVERY
            == RECORD17_PRECOND_EVERY - 1
        )
        if do_refresh:
            self._record17_refresh_count += 1
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * self.world_size
            momentum = torch._as_tensor_fullprec(group["momentum"])
            for base_i in range(len(params))[:: self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    state = self.state[p]
                    if "mantissa" not in state:
                        state["mantissa"] = torch.zeros_like(
                            p, dtype=torch.uint16
                        )
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(
                            p, dtype=torch.float32
                        )
                    if do_refresh and p in self._precond_map:
                        self._refresh_one(p, state)
                    grad = p.grad
                    if self._precond_attached and p in self._precond_map:
                        grad = self._apply_inv(p, grad, state)
                    update(
                        p.view(torch.uint16),
                        state["mantissa"],
                        state["momentum_buffer"],
                        grad,
                        momentum,
                        eff_lr=torch._as_tensor_fullprec(
                            group["lr"]
                            * max(1, p.size(-2) / p.size(-1)) ** 0.5
                        ),
                        eff_weight_decay=torch._as_tensor_fullprec(
                            group["lr"]
                            * group["weight_decay"]
                            * getattr(p, "wd_mul", 1.0)
                        ),
                    )
                futures.append(
                    dist.all_gather(
                        params_pad[base_i : base_i + self.world_size],
                        params_pad[base_i + self.rank],
                        async_op=True,
                    ).get_future()
                )
        torch.futures.collect_all(futures).wait()
        if self._precond_attached:
            self._precond_step += 1

'''


ATTENTION_INIT_OLD = """        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask, lambdas: Tensor):
"""

ATTENTION_INIT_NEW = """        self.attn_scale = 0.12
        if RECORD17_NEWTON_ACTIVE:
            self.precond_qkv_xtx_accum = nn.Buffer(
                torch.zeros((dim, dim), dtype=torch.float32), persistent=False
            )
            self.precond_qkv_xtx_tmp = nn.Buffer(
                torch.empty((dim, dim), dtype=torch.float32), persistent=False
            )
            self.precond_qkv_xtx_count = nn.Buffer(
                torch.zeros((), dtype=torch.int32), persistent=False
            )
            self.precond_o_xtx_accum = nn.Buffer(
                torch.zeros((dim, dim), dtype=torch.float32), persistent=False
            )
            self.precond_o_xtx_tmp = nn.Buffer(
                torch.empty((dim, dim), dtype=torch.float32), persistent=False
            )
            self.precond_o_xtx_count = nn.Buffer(
                torch.zeros((), dtype=torch.int32), persistent=False
            )

    def forward(
        self,
        x: Tensor,
        ve: Tensor | None,
        block_mask: BlockMask,
        lambdas: Tensor,
        precond_flag: bool,
    ):
"""


ATTENTION_QKV_OLD = """        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkvo_w[:3].flatten(end_dim=1)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
"""

ATTENTION_QKV_NEW = """        assert B == 1, "Must use batch size = 1 for FlexAttention"
        if precond_flag:
            x2d = x.flatten(0, -2)
            torch.ops.record17.accum_xtx_dense_v1(
                x2d,
                self.precond_qkv_xtx_accum,
                self.precond_qkv_xtx_count,
                self.precond_qkv_xtx_tmp,
            )
        q, k, v = F.linear(x, self.qkvo_w[:3].flatten(end_dim=1)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
"""


ATTENTION_O_OLD = """        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, self.qkvo_w[3])
"""

ATTENTION_O_NEW = """        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        if precond_flag:
            y2d = y.flatten(0, -2)
            torch.ops.record17.accum_xtx_dense_v1(
                y2d,
                self.precond_o_xtx_accum,
                self.precond_o_xtx_count,
                self.precond_o_xtx_tmp,
            )
        y = F.linear(y, self.qkvo_w[3])
"""


MLP_INIT_OLD = """        self.fc_w.wd_mul = 2.0
        self.proj_w.wd_mul = 2.0

    def forward(self, x: Tensor):
        x = F.linear(x, self.fc_w)
"""

MLP_INIT_NEW = """        self.fc_w.wd_mul = 2.0
        self.proj_w.wd_mul = 2.0
        if RECORD17_NEWTON_ACTIVE:
            self.precond_fc_xtx_accum = nn.Buffer(
                torch.zeros((dim, dim), dtype=torch.float32), persistent=False
            )
            self.precond_fc_xtx_tmp = nn.Buffer(
                torch.empty((dim, dim), dtype=torch.float32), persistent=False
            )
            self.precond_fc_xtx_count = nn.Buffer(
                torch.zeros((), dtype=torch.int32), persistent=False
            )
            if RECORD17_CPROJ_K_MODE == "block4":
                self.precond_proj_xtx_accum = nn.Buffer(
                    torch.zeros((4, dim, dim), dtype=torch.float32),
                    persistent=False,
                )
                self.precond_proj_xtx_tmp = nn.Buffer(
                    torch.empty((4, dim, dim), dtype=torch.float32),
                    persistent=False,
                )
                self.precond_proj_xtx_count = nn.Buffer(
                    torch.zeros((), dtype=torch.int32), persistent=False
                )
            elif RECORD17_CPROJ_K_MODE == "diag":
                self.precond_proj_xtx_accum = nn.Buffer(
                    torch.zeros((4, dim), dtype=torch.float32), persistent=False
                )
                self.precond_proj_xtx_count = nn.Buffer(
                    torch.zeros((), dtype=torch.int32), persistent=False
                )
            elif RECORD17_CPROJ_K_MODE != "none":
                raise ValueError(
                    f"unsupported c_proj K mode={RECORD17_CPROJ_K_MODE!r}"
                )

    def forward(self, x: Tensor, precond_flag: bool):
        if precond_flag:
            x2d = x.flatten(0, -2)
            torch.ops.record17.accum_xtx_dense_v1(
                x2d,
                self.precond_fc_xtx_accum,
                self.precond_fc_xtx_count,
                self.precond_fc_xtx_tmp,
            )
        x = F.linear(x, self.fc_w)
"""


MLP_PROJ_OLD = """        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.linear(x, self.proj_w)
"""

MLP_PROJ_NEW = """        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        if precond_flag and RECORD17_CPROJ_K_MODE != "none":
            x2d = x.flatten(0, -2)
            if RECORD17_CPROJ_K_MODE == "block4":
                torch.ops.record17.accum_xtx_blocks4_v1(
                    x2d,
                    self.precond_proj_xtx_accum,
                    self.precond_proj_xtx_count,
                    self.precond_proj_xtx_tmp,
                )
            else:
                torch.ops.record17.accum_xtx_diag4_v1(
                    x2d,
                    self.precond_proj_xtx_accum,
                    self.precond_proj_xtx_count,
                )
        x = F.linear(x, self.proj_w)
"""


BLOCK_FORWARD_OLD = """    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask, lambdas: Tensor, sa_lambdas: Tensor):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(x, ve, block_mask, sa_lambdas)
        x = x + self.mlp(norm(x))
"""

BLOCK_FORWARD_NEW = """    def forward(
        self,
        x: Tensor,
        ve: Tensor | None,
        x0: Tensor,
        block_mask: BlockMask,
        lambdas: Tensor,
        sa_lambdas: Tensor,
        precond_flag: bool,
    ):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(
                x, ve, block_mask, sa_lambdas, precond_flag
            )
        x = x + self.mlp(norm(x), precond_flag)
"""


GPT_FORWARD_OLD = """    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
"""

GPT_FORWARD_NEW = """    def forward(
        self,
        input_seq: Tensor,
        target_seq: Tensor,
        sliding_window_num_blocks: Tensor,
        precond_flag: bool,
    ):
"""


GPT_BLOCK_CALL_OLD = """            x = self.blocks[i](x, ve[i], x0, block_masks[i], lambdas[i], sa_lambdas[i])
"""

GPT_BLOCK_CALL_NEW = """            x = self.blocks[i](
                x,
                ve[i],
                x0,
                block_masks[i],
                lambdas[i],
                sa_lambdas[i],
                precond_flag,
            )
"""


ATTACH_ANCHOR = """model: nn.Module = torch.compile(model, dynamic=False)
"""

ATTACH_BLOCK = """if RECORD17_NEWTON_ACTIVE:
    optimizer2.attach_preconditioner(model)
model: nn.Module = torch.compile(model, dynamic=False)
"""


ZERO_BUFFERS_HELPER = """@torch.no_grad()
def record17_zero_activation_buffers(module: nn.Module) -> None:
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod
    for name, buffer in module.named_buffers():
        if "precond_" in name and (
            "xtx_accum" in name or "xtx_count" in name
        ):
            buffer.zero_()

"""


WARMUP_START = """########################################
#            Warmup kernels            #
########################################
"""

TRAINING_START = """########################################
#        Training and validation       #
########################################
"""

WARMUP_BLOCK = WARMUP_START + """
# Instrumentation warmup deliberately crosses refresh update 24 and executes
# two post-refresh updates.  It is never scientific training: model, Adam,
# Muon/K, activation accumulators, RNG, loader position, and gradients are
# restored before counted update zero.
warmup_steps = RECORD17_WARMUP_UPDATES
record17_rng_state = dict(
    python=random.getstate(),
    numpy=np.random.get_state(),
    torch=torch.get_rng_state(),
    cuda=torch.cuda.get_rng_state_all(),
)
record17_rng_sha256_before = _record17_rng_fingerprint()
initial_state = copy.deepcopy(
    dict(model=model.state_dict())
)
# Keep parameter identities while copying only state values.  Optimizer
# load_state_dict may cast arbitrary floating state to the BF16 parameter
# dtype; direct restoration is required to preserve FP32 K/inverse/workspace
# tensors exactly.
initial_optimizer_states = [
    [
        (parameter, copy.deepcopy(state))
        for parameter, state in optimizer.state.items()
    ]
    for optimizer in optimizers
]
record17_optimizer_sha256_before = _record17_optimizer_fingerprint(optimizers)
warmup_loader = distributed_data_generator(
    args.train_files,
    world_size * RECORD17_GRAD_ACCUM_STEPS * args.train_seq_len,
    rank,
    world_size,
)
for warmup_step in range(warmup_steps):
    precond_flag = (
        RECORD17_NEWTON_ACTIVE
        and warmup_step % RECORD17_PRECOND_EVERY
        == RECORD17_PRECOND_EVERY - 1
    )
    global_inputs, global_targets = next(warmup_loader)
    input_microbatches = global_inputs.chunk(RECORD17_GRAD_ACCUM_STEPS)
    target_microbatches = global_targets.chunk(RECORD17_GRAD_ACCUM_STEPS)
    for inputs, targets in zip(input_microbatches, target_microbatches):
        loss = model(
            inputs,
            targets,
            get_window_size_blocks(warmup_step),
            precond_flag,
        )
        (loss / RECORD17_GRAD_ACCUM_STEPS).backward()
    for parameter in model.parameters():
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
    for optimizer in optimizers:
        optimizer.step()
    model.zero_grad(set_to_none=True)
record17_warmup_refresh_count = int(optimizer2._record17_refresh_count)
record17_warmup_fp32_apply_count = int(
    optimizer2._record17_fp32_apply_count
)
model.load_state_dict(initial_state["model"])
for optimizer, optimizer_states in zip(
    optimizers, initial_optimizer_states
):
    optimizer.state.clear()
    for parameter, state in optimizer_states:
        optimizer.state[parameter] = state
optimizer2.reset_precond_step()
record17_zero_activation_buffers(model)
model.zero_grad(set_to_none=True)
random.setstate(record17_rng_state["python"])
np.random.set_state(record17_rng_state["numpy"])
torch.set_rng_state(record17_rng_state["torch"])
torch.cuda.set_rng_state_all(record17_rng_state["cuda"])
record17_raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
record17_post_warmup_sha256, _, _ = _record17_model_fingerprint(
    record17_raw_model
)
record17_optimizer_sha256_after = _record17_optimizer_fingerprint(optimizers)
record17_rng_sha256_after = _record17_rng_fingerprint()
record17_accumulators_zero = all(
    torch.count_nonzero(buffer).item() == 0
    for name, buffer in record17_raw_model.named_buffers()
    if "precond_" in name
    and ("xtx_accum" in name or "xtx_count" in name)
)
record17_gradients_clear = all(
    parameter.grad is None for parameter in record17_raw_model.parameters()
)
record17_warmup_reset = {
    "warmup_updates": warmup_steps,
    "crossed_refresh_24": warmup_steps >= 24,
    "post_refresh_updates": warmup_steps - 24,
    "warmup_refresh_count_before_restore": record17_warmup_refresh_count,
    "warmup_fp32_apply_count_before_restore": (
        record17_warmup_fp32_apply_count
    ),
    "warmup_newton_path_exercised": (
        record17_warmup_refresh_count == 1
        and record17_warmup_fp32_apply_count > 0
        if RECORD17_NEWTON_ACTIVE
        else (
            record17_warmup_refresh_count == 0
            and record17_warmup_fp32_apply_count == 0
        )
    ),
    "model_sha256": record17_post_warmup_sha256,
    "model_matches_initial": (
        record17_post_warmup_sha256 == record17_init_sha256
    ),
    "optimizer_sha256_before": record17_optimizer_sha256_before,
    "optimizer_sha256_after": record17_optimizer_sha256_after,
    "optimizer_matches_initial": (
        record17_optimizer_sha256_after
        == record17_optimizer_sha256_before
    ),
    "preconditioner_step": int(optimizer2._precond_step),
    "preconditioner_step_zero": int(optimizer2._precond_step) == 0,
    "activation_accumulators_zero": record17_accumulators_zero,
    "gradients_clear": record17_gradients_clear,
    "rng_sha256_before": record17_rng_sha256_before,
    "rng_sha256_after": record17_rng_sha256_after,
    "rng_matches_initial": (
        record17_rng_sha256_after == record17_rng_sha256_before
    ),
    "loader_recreated_from_start": True,
}
if not all(
    (
        record17_warmup_reset["crossed_refresh_24"],
        record17_warmup_reset["post_refresh_updates"] >= 2,
        record17_warmup_reset["warmup_newton_path_exercised"],
        record17_warmup_reset["model_matches_initial"],
        record17_warmup_reset["optimizer_matches_initial"],
        record17_warmup_reset["preconditioner_step_zero"],
        record17_warmup_reset["activation_accumulators_zero"],
        record17_warmup_reset["gradients_clear"],
        record17_warmup_reset["rng_matches_initial"],
        record17_warmup_reset["loader_recreated_from_start"],
    )
):
    raise RuntimeError(
        f"post-warmup reset audit failed: {record17_warmup_reset}"
    )
print0(
    "RECORD17_WARMUP_RESET "
    + json.dumps(record17_warmup_reset, sort_keys=True),
    console=True,
)
del (
    warmup_loader,
    initial_state,
    initial_optimizer_states,
    record17_rng_state,
    global_inputs,
    global_targets,
    input_microbatches,
    target_microbatches,
    inputs,
    targets,
    loss,
)
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

"""


TRAIN_LOADER_OLD = """torch.cuda.reset_peak_memory_stats()
train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
"""

TRAIN_LOADER_NEW = """train_loader = distributed_data_generator(
    args.train_files,
    world_size * RECORD17_GRAD_ACCUM_STEPS * args.train_seq_len,
    rank,
    world_size,
)
"""


VALIDATION_CALL_OLD = """                val_loss += model(inputs, targets, get_window_size_blocks(step))
"""

VALIDATION_CALL_NEW = """                val_loss += model(
                    inputs, targets, get_window_size_blocks(step), False
                )
"""


VALIDATION_LOG_OLD = """        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
"""

VALIDATION_LOG_NEW = VALIDATION_LOG_OLD + """        record17_validation = {
            "step": int(step),
            "total_steps": int(train_steps),
            "val_loss": float(val_loss.item()),
            "train_time_ms": int(round(training_time_ms)),
            "step_avg_ms": float(training_time_ms / max(step, 1)),
            "tokens": int(
                step
                * world_size
                * RECORD17_GRAD_ACCUM_STEPS
                * args.train_seq_len
            ),
        }
        print0(
            "RECORD17_VAL "
            + json.dumps(record17_validation, sort_keys=True),
            console=True,
        )
"""


LAST_STEP_OLD = """    if last_step:
        if master_process and args.save_checkpoint:
"""

LAST_STEP_NEW = """    if last_step:
        record17_counted_run_peak_allocated_bytes = int(
            torch.cuda.max_memory_allocated()
        )
        record17_counted_run_peak_reserved_bytes = int(
            torch.cuda.max_memory_reserved()
        )
        if master_process and args.save_checkpoint:
"""


CHECKPOINT_OLD = """        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id_full}", exist_ok=True)
            torch.save(log, f"logs/{run_id_full}/state_step{step:06d}.pt")
"""

CHECKPOINT_NEW = """        if master_process and args.save_checkpoint:
            log = dict(
                step=step,
                code=code,
                model=record17_raw_model.state_dict(),
                record17_metadata=record17_metadata,
                checkpoint_scope="model_only",
                optimizer_resume_supported=False,
                preconditioner_step=int(optimizer2._precond_step),
            )
            checkpoint_path = os.path.join(
                RECORD17_OUTPUT_DIR, f"state_step{step:06d}.pt"
            )
            temporary_checkpoint = checkpoint_path + ".tmp"
            torch.save(log, temporary_checkpoint)
            os.replace(temporary_checkpoint, checkpoint_path)
            print0(
                f"RECORD17_CHECKPOINT {checkpoint_path}", console=True
            )
"""


TRAINING_STEP_OLD = """    inputs, targets = next(train_loader)
    model(inputs, targets, get_window_size_blocks(step)).backward()
    opt2futures = {
"""

TRAINING_STEP_NEW = """    precond_flag = (
        RECORD17_NEWTON_ACTIVE
        and step % RECORD17_PRECOND_EVERY
        == RECORD17_PRECOND_EVERY - 1
    )
    global_inputs, global_targets = next(train_loader)
    input_microbatches = global_inputs.chunk(RECORD17_GRAD_ACCUM_STEPS)
    target_microbatches = global_targets.chunk(RECORD17_GRAD_ACCUM_STEPS)
    for inputs, targets in zip(input_microbatches, target_microbatches):
        loss = model(
            inputs, targets, get_window_size_blocks(step), precond_flag
        )
        (loss / RECORD17_GRAD_ACCUM_STEPS).backward()
    opt2futures = {
"""


FINAL_MEMORY_OLD = """print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()
"""

FINAL_MEMORY_NEW = """if master_process:
    optimizer_tensors = []
    for optimizer in optimizers:
        optimizer_tensors.extend(_record17_iter_tensors(optimizer.state))
    k_cov_tensors = []
    k_inv_tensors = []
    k_workspace_tensors = []
    cproj_cov_tensors = []
    cproj_inv_tensors = []
    cproj_workspace_tensors = []
    cproj_kinds = []
    for state in optimizer2.state.values():
        if not isinstance(state, dict):
            continue
        cov = state.get("cov")
        inv = state.get("inv")
        if isinstance(cov, Tensor):
            k_cov_tensors.append(cov)
        if isinstance(inv, Tensor):
            k_inv_tensors.append(inv)
        for key in ("Kbuf", "precond_buf"):
            tensor = state.get(key)
            if isinstance(tensor, Tensor):
                k_workspace_tensors.append(tensor)
        if state.get("kind") in ("proj_w_block4", "proj_w_diag"):
            cproj_kinds.append(state["kind"])
            if isinstance(cov, Tensor):
                cproj_cov_tensors.append(cov)
            if isinstance(inv, Tensor):
                cproj_inv_tensors.append(inv)
            for key in ("Kbuf", "precond_buf"):
                tensor = state.get(key)
                if isinstance(tensor, Tensor):
                    cproj_workspace_tensors.append(tensor)
    optimizer_runtime_cache_tensors = []
    for cache_name in ("_eye_cache", "_eye4_cache"):
        cache = getattr(optimizer2, cache_name, {})
        if isinstance(cache, dict):
            optimizer_runtime_cache_tensors.extend(
                tensor
                for tensor in cache.values()
                if isinstance(tensor, Tensor)
            )
    activation_stat_tensors = []
    activation_workspace_tensors = []
    cproj_activation_stat_tensors = []
    cproj_activation_workspace_tensors = []
    for name, tensor in record17_raw_model.named_buffers():
        if "precond_" not in name:
            continue
        if "xtx_accum" in name or "xtx_count" in name:
            activation_stat_tensors.append(tensor)
        elif "xtx_tmp" in name:
            activation_workspace_tensors.append(tensor)
        if "precond_proj_" in name:
            if "xtx_accum" in name or "xtx_count" in name:
                cproj_activation_stat_tensors.append(tensor)
            elif "xtx_tmp" in name:
                cproj_activation_workspace_tensors.append(tensor)
    inverse_dtypes = sorted({str(tensor.dtype) for tensor in k_inv_tensors})
    precondition_buffer_dtypes = sorted(
        {
            str(state["precond_buf"].dtype)
            for state in optimizer2.state.values()
            if isinstance(state, dict)
            and isinstance(state.get("precond_buf"), Tensor)
        }
    )
    hidden_parameter_dtypes = sorted(
        {str(parameter.dtype) for parameter in hidden_matrix_params}
    )
    fp32_precondition_contract_passed = (
        (
            inverse_dtypes == ["torch.float32"]
            and precondition_buffer_dtypes == ["torch.float32"]
            and optimizer2._record17_raw_grad_dtypes == {"torch.bfloat16"}
            and optimizer2._record17_preconditioned_grad_dtypes
            == {"torch.float32"}
            and optimizer2._record17_fp32_apply_count > 0
        )
        if RECORD17_NEWTON_ACTIVE
        else (
            inverse_dtypes == []
            and precondition_buffer_dtypes == []
            and optimizer2._record17_fp32_apply_count == 0
        )
    )
    final_audit = {
        "optimizer_state_bytes": _record17_unique_storage_bytes(
            optimizer_tensors
        ),
        "optimizer_runtime_cache_bytes": _record17_unique_storage_bytes(
            optimizer_runtime_cache_tensors
        ),
        "optimizer_runtime_total_bytes": _record17_unique_storage_bytes(
            optimizer_tensors + optimizer_runtime_cache_tensors
        ),
        "k_cov_bytes": _record17_unique_storage_bytes(k_cov_tensors),
        "k_inv_bytes": _record17_unique_storage_bytes(k_inv_tensors),
        "k_state_bytes": _record17_unique_storage_bytes(
            k_cov_tensors + k_inv_tensors
        ),
        "k_workspace_bytes": _record17_unique_storage_bytes(
            k_workspace_tensors
        ),
        "activation_stat_bytes": _record17_unique_storage_bytes(
            activation_stat_tensors
        ),
        "activation_workspace_bytes": _record17_unique_storage_bytes(
            activation_workspace_tensors
        ),
        "cproj_k_kind": sorted(set(cproj_kinds)),
        "cproj_k_parameter_count": len(cproj_kinds),
        "cproj_cov_bytes": _record17_unique_storage_bytes(
            cproj_cov_tensors
        ),
        "cproj_inv_bytes": _record17_unique_storage_bytes(
            cproj_inv_tensors
        ),
        "cproj_workspace_bytes": _record17_unique_storage_bytes(
            cproj_workspace_tensors
        ),
        "cproj_activation_stat_bytes": _record17_unique_storage_bytes(
            cproj_activation_stat_tensors
        ),
        "cproj_activation_workspace_bytes": _record17_unique_storage_bytes(
            cproj_activation_workspace_tensors
        ),
        "cproj_cov_shapes": [
            list(tensor.shape) for tensor in cproj_cov_tensors
        ],
        "cproj_inv_shapes": [
            list(tensor.shape) for tensor in cproj_inv_tensors
        ],
        "hidden_parameter_dtypes": hidden_parameter_dtypes,
        "inverse_dtypes": inverse_dtypes,
        "precondition_buffer_dtypes": precondition_buffer_dtypes,
        "raw_gradient_dtypes_seen_by_preconditioner": sorted(
            optimizer2._record17_raw_grad_dtypes
        ),
        "preconditioned_gradient_dtypes": sorted(
            optimizer2._record17_preconditioned_grad_dtypes
        ),
        "fp32_precondition_application_count": int(
            optimizer2._record17_fp32_apply_count
        ),
        "raw_gradients_cast_to_fp32": (
            optimizer2._record17_preconditioned_grad_dtypes
            == {"torch.float32"}
            if RECORD17_NEWTON_ACTIVE
            # The unmodified upstream Muon update begins with
            # ``grad = grad.float()`` before momentum/NS.
            else True
        ),
        "fp32_precondition_contract_passed": (
            fp32_precondition_contract_passed
        ),
        "peak_memory_allocated_bytes": (
            record17_counted_run_peak_allocated_bytes
        ),
        "peak_memory_reserved_bytes": (
            record17_counted_run_peak_reserved_bytes
        ),
        "peak_memory_scope": (
            "counted_run_after_warmup_reset_including_validation"
        ),
        "preconditioner_step": int(optimizer2._precond_step),
        "refresh_count": int(optimizer2._record17_refresh_count),
        "first_refresh_zero_based_step": (
            RECORD17_PRECOND_EVERY - 1
            if RECORD17_NEWTON_ACTIVE
            else None
        ),
        "all_finite": all(
            torch.isfinite(parameter).all().item()
            for parameter in record17_raw_model.parameters()
        ),
        # Workspaces are allocated with torch.empty and are not evidence
        # tensors.  Only covariance and inverse values must be finite.
        "k_tensors_all_finite": all(
            torch.isfinite(tensor).all().item()
            for tensor in (k_cov_tensors + k_inv_tensors)
        ),
    }
    final_audit["total_preconditioner_bytes"] = (
        _record17_unique_storage_bytes(
            k_cov_tensors
            + k_inv_tensors
            + k_workspace_tensors
            + optimizer_runtime_cache_tensors
            + activation_stat_tensors
            + activation_workspace_tensors
        )
    )
    if not fp32_precondition_contract_passed:
        raise RuntimeError(
            f"FP32 precondition contract failed: {final_audit}"
        )
    print0(
        "RECORD17_FINAL_AUDIT "
        + json.dumps(final_audit, sort_keys=True),
        console=True,
    )
    print0(
        f"peak memory allocated: "
        f"{torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: "
        f"{torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
        console=True,
    )
dist.destroy_process_group()
"""


def apply_overlay(base_source: str) -> str:
    source = replace_once(
        base_source,
        COMPILED_AUTOGRAD_OLD,
        COMPILED_AUTOGRAD_NEW,
        "PyTorch 2.8 compiled-autograd policy",
    )
    source = replace_once(
        source, COMMON_IMPORTS_OLD, COMMON_IMPORTS_NEW, "imports"
    )
    source = replace_once(
        source,
        "args = Hyperparameters()\n",
        CONTROL_BLOCK,
        "controlled hyperparameters",
    )
    source = replace_once(
        source, WORLD_SIZE_OLD, WORLD_SIZE_NEW, "single-GPU world size"
    )
    source = replace_once(
        source, DATA_GLOB_OLD, DATA_GLOB_NEW, "absolute data glob"
    )
    source = replace_once(
        source,
        RELATIVE_DATA_DEFAULTS_OLD,
        RELATIVE_DATA_DEFAULTS_NEW,
        "relative data defaults",
    )
    source = replace_exact_count(
        source,
        "x = step / args.num_iterations # progress in training",
        "x = step / RECORD17_SCHEDULE_ITERATIONS "
        "# frozen formal schedule, including smoke prefix",
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
    source = replace_region_once(
        source,
        "class Muon(torch.optim.Optimizer):\n",
        "# -----------------------------------------------------------------------------\n"
        "# PyTorch nn.Module definitions for the model\n",
        OPTIMIZER_CLASS,
        "Muon optimizer",
    )
    source = replace_once(
        source,
        "# -----------------------------------------------------------------------------\n# Muon optimizer\n\n",
        CUSTOM_OPS_AND_OPTIMIZER_START,
        "custom-op insertion",
    )
    source = replace_once(
        source,
        ATTENTION_INIT_OLD,
        ATTENTION_INIT_NEW,
        "attention covariance state",
    )
    source = replace_once(
        source,
        ATTENTION_QKV_OLD,
        ATTENTION_QKV_NEW,
        "attention input covariance",
    )
    source = replace_once(
        source,
        ATTENTION_O_OLD,
        ATTENTION_O_NEW,
        "attention output covariance",
    )
    source = replace_once(
        source, MLP_INIT_OLD, MLP_INIT_NEW, "MLP covariance state"
    )
    source = replace_once(
        source, MLP_PROJ_OLD, MLP_PROJ_NEW, "MLP projection covariance"
    )
    source = replace_once(
        source, BLOCK_FORWARD_OLD, BLOCK_FORWARD_NEW, "Block precondition flag"
    )
    source = replace_once(
        source, GPT_FORWARD_OLD, GPT_FORWARD_NEW, "GPT precondition flag"
    )
    source = replace_once(
        source,
        GPT_BLOCK_CALL_OLD,
        GPT_BLOCK_CALL_NEW,
        "GPT block precondition flag",
    )
    source = replace_once(
        source, ATTACH_ANCHOR, ATTACH_BLOCK, "preconditioner attachment"
    )
    source = replace_once(
        source,
        WARMUP_START,
        ZERO_BUFFERS_HELPER + WARMUP_START,
        "activation reset helper",
    )
    source = replace_region_once(
        source,
        WARMUP_START,
        TRAINING_START,
        WARMUP_BLOCK,
        "instrumentation warmup",
    )
    source = replace_once(
        source,
        TRAIN_LOADER_OLD,
        TRAIN_LOADER_NEW,
        "counted loader reset",
    )
    source = replace_once(
        source,
        VALIDATION_CALL_OLD,
        VALIDATION_CALL_NEW,
        "validation precondition flag",
    )
    source = replace_once(
        source,
        VALIDATION_LOG_OLD,
        VALIDATION_LOG_NEW,
        "full-precision validation log",
    )
    source = replace_once(
        source, LAST_STEP_OLD, LAST_STEP_NEW, "training peak capture"
    )
    source = replace_once(
        source, CHECKPOINT_OLD, CHECKPOINT_NEW, "model-only checkpoint"
    )
    source = replace_once(
        source,
        TRAINING_STEP_OLD,
        TRAINING_STEP_NEW,
        "single-GPU gradient accumulation",
    )
    source = replace_once(
        source, FINAL_MEMORY_OLD, FINAL_MEMORY_NEW, "final numerical audit"
    )
    return source


def assert_source_contract(source: str) -> None:
    required = (
        "RECORD17_METADATA ",
        "RECORD17_VAL ",
        "RECORD17_WARMUP_RESET ",
        "RECORD17_FINAL_AUDIT ",
        "record17_parameter_count != 454496336",
        "RECORD17_SCHEDULE_ITERATIONS = 5960",
        "RECORD17_GRAD_ACCUM_STEPS = 8",
        'RECORD17_DATA_PATH = os.path.abspath(os.environ["DATA_PATH"])',
        "glob.glob(filename_pattern)",
        "RECORD17_WARMUP_UPDATES = 26",
        "RECORD17_PRECOND_EVERY = 24",
        "RECORD17_PRECOND_EWMA = 0.90",
        "RECORD17_RIDGE_MULT = 0.20",
        "torch._dynamo.config.compiled_autograd = False",
        '"compiled_autograd_enabled"',
        'RECORD17_CPROJ_K_MODE == "block4"',
        'RECORD17_CPROJ_K_MODE == "diag"',
        'RECORD17_CPROJ_K_MODE == "none"',
        "raw_grad.float()",
        "dtype=torch.float32",
        '"fp32_precondition_contract_passed"',
        "torch.cuda.empty_cache()",
        "torch.cuda.reset_peak_memory_stats()",
        "state_step{step:06d}.pt",
    )
    missing = [anchor for anchor in required if anchor not in source]
    if missing:
        raise RuntimeError(f"derived source is missing anchors {missing}")
    forbidden = (
        "assert world_size == 8",
        'train_files = "data/fineweb10B/',
        'val_files = "data/fineweb10B/',
        "Path.cwd().glob(filename_pattern)",
        "x = step / args.num_iterations # progress in training",
        "torch.empty_like(p)\n                self._precond_map",
        "torch._dynamo.config.compiled_autograd = True",
    )
    present = [anchor for anchor in forbidden if anchor in source]
    if present:
        raise RuntimeError(f"derived source retained forbidden anchors {present}")


def _load_base_source() -> tuple[str, str]:
    audit_vendored_record17(BASE_PATH)
    raw = canonical_bytes(BASE_PATH.read_bytes())
    observed = hashlib.sha256(raw).hexdigest()
    if observed != RECORD17_UPSTREAM_CANONICAL_SHA256:
        raise RuntimeError(
            "Record #17 canonical hash mismatch: "
            f"expected {RECORD17_UPSTREAM_CANONICAL_SHA256}, observed {observed}"
        )
    return raw.decode("utf-8"), observed


def build_source(_official_repo: Path, method: str) -> DerivedSource:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    base_source, base_sha256 = _load_base_source()
    source = apply_overlay(base_source)
    compile(source, f"<experiment44-{method}>", "exec")
    assert_source_contract(source)
    cproj_mode = {
        "muon": "not_applicable",
        "original_newton_muon": "block4",
        "selective_none": "none",
        "selective_diag": "diag",
    }[method]
    diff = "".join(
        difflib.unified_diff(
            base_source.splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile=f"upstream/{BASE_SCRIPT}",
            tofile="experiment44/train_record17_unified.py",
        )
    )
    return DerivedSource(
        method=method,
        cproj_k_mode=cproj_mode,
        base_script=BASE_SCRIPT,
        base_canonical_sha256=base_sha256,
        derived_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source=source,
        unified_diff=diff,
    )


def build_all_sources(official_repo: Path) -> dict[str, DerivedSource]:
    sources = {
        method: build_source(official_repo, method) for method in METHODS
    }
    hashes = {item.derived_sha256 for item in sources.values()}
    if len(hashes) != 1:
        raise RuntimeError(
            "all four methods must share one environment-dispatched source"
        )
    return sources


def self_test_diag_math() -> None:
    """Pure-CPU reference tests for dense/block/diagonal right application."""

    import random

    generator = random.Random(44017)
    samples, d = 19, 7
    x = [
        [generator.gauss(0, 1) for _ in range(4 * d)]
        for _ in range(samples)
    ]
    diagonal = [
        [
            sum(row[block * d + column] ** 2 for row in x) / samples
            for column in range(d)
        ]
        for block in range(4)
    ]
    explicit = [
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
    if diagonal != explicit:
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
            for column in range(d):
                expected.append(
                    sum(
                        gradient[row][block * d + inner]
                        * (
                            inverse[block][inner]
                            if inner == column
                            else 0.0
                        )
                        for inner in range(d)
                    )
                )
    if any(abs(a - b) > 1e-15 for a, b in zip(observed, expected)):
        raise AssertionError("diagonal right-application contract failed")


def self_test_right_precondition_math() -> None:
    """Pure-CPU shape/orientation check for qkvo, fc_w, and proj_w."""

    import random

    generator = random.Random(44170)
    d = 5

    def matmul(left, right):
        return [
            [
                sum(left[i][k] * right[k][j] for k in range(len(right)))
                for j in range(len(right[0]))
            ]
            for i in range(len(left))
        ]

    inverse = [
        [generator.gauss(0, 1) for _ in range(d)] for _ in range(d)
    ]
    fc = [
        [generator.gauss(0, 1) for _ in range(d)] for _ in range(4 * d)
    ]
    if len(matmul(fc, inverse)) != 4 * d:
        raise AssertionError("fc_w must be right-preconditioned")
    qkvo = [
        [
            [generator.gauss(0, 1) for _ in range(d)]
            for _ in range(d)
        ]
        for _ in range(4)
    ]
    if any(len(matmul(matrix, inverse)) != d for matrix in qkvo):
        raise AssertionError("qkvo must be right-preconditioned")
    proj = [
        [generator.gauss(0, 1) for _ in range(4 * d)] for _ in range(d)
    ]
    blocks = [
        [row[block * d : (block + 1) * d] for row in proj]
        for block in range(4)
    ]
    rebuilt = [
        sum((matmul(block, inverse)[row] for block in blocks), [])
        for row in range(d)
    ]
    if len(rebuilt) != d or any(len(row) != 4 * d for row in rebuilt):
        raise AssertionError("proj_w blockwise right-precondition contract failed")
