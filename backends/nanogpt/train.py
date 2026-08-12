"""
Single-process nanoGPT training entry for Selective Newton-Muon experiments.

Example:
python train.py config/train_shakespeare_char.py config/selective/10_selective_v2_top75_fast_warmup100.py
"""

import math
import os
import pickle
import shutil
import time
import csv
from contextlib import nullcontext

import numpy as np
import torch

from model import GPT, GPTConfig
from optimizer_factory import INPUT_TRACKED_OPTIMIZERS, MUON_FAMILY_OPTIMIZERS, build_optimizer
from quadratic_probe import (
    expanded_probe_modes,
    parse_float_csv,
    parse_int_csv,
    parse_mode_csv,
    run_cproj_quadratic_probe_repeated,
    write_quadratic_probe_artifacts,
)
from temporal_quadratic_probe import (
    run_cproj_temporal_quadratic_probe,
    temporal_candidate_names,
    validate_temporal_modes,
    write_temporal_quadratic_probe_artifacts,
)


def safe_export_name(value):
    value = str(value or "run").strip() or "run"
    return "".join("_" if ch in '<>:"/\\|?*' else ch for ch in value)


# I/O
out_dir = "out"
csv_export_dir = None
eval_interval = 200
log_interval = 10
eval_iters = 20
eval_only = False
always_save_checkpoint = False
save_checkpoint = True
init_from = "scratch"  # "scratch" or "resume"

# W&B
wandb_log = False
wandb_project = "Selective-Newton-Muon"
wandb_run_name = "debug-run"
wandb_mode = "offline"
wandb_group = "manual"
wandb_tags = "block-sigma,newton-muon,nanogpt"
wandb_log_profile = "paper"  # use "full" only for an explicit diagnostics run
wandb_log_tables = False  # large probe tables are opt-in

# Data
dataset = "shakespeare_char"
require_real_tiny_shakespeare = True
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128

# Model
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1
bias = False

# Optimizer
optimizer_type = "selective_newton_muon"
learning_rate = 1e-3
max_iters = 1000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

muon_learning_rate = 0.02
muon_momentum = 0.95
muon_ns_steps = 5
muon_nesterov = False
muon_momentum_ema = False
muon_split_qkv = False
muon_adjust_lr_for_shape = False
muon_ns_compute_dtype = "float32"  # "float32" or "bfloat16"
matrix_weight_decay = 0.0
matrix_eps = 1e-8

input_beta = 0.95
input_ridge = 0.2
input_refresh = 32
input_max_samples = 2048
input_first_refresh_step = 0
input_init_scale = 1.0
input_init_inverse_scale = 0.0  # <= 0 means reciprocal(input_init_scale)

# Mechanism control: vary only the mlp.c_proj activation K structure.
cproj_k_mode = "block4"  # "none", "full", "block4", "diag", "scalar", "alpha", or "block_alpha"
cproj_k_layers = ""  # empty means apply cproj_k_mode to every mlp.c_proj depth
cproj_k_blocks = 4
cproj_k_reference_mode = "full"  # storage/accounting reference: "full" or "block4"
cproj_k_offdiag_alpha = 0.5
cproj_shadow_k_modes = ""  # diagnostic-only EMA K/momentum states, e.g. "diag,full"
cproj_shadow_k_layers = ""  # empty means all c_proj layers

selective_fraction = 0.75
selective_min_active = 1
selective_warmup_steps = 100
selective_score_interval = 25
selective_score_beta = 0.9
selective_score_threshold = 0.0
selective_score_mode = "gain_over_cost"
selective_cond_power = 1.0
selective_cost_power = 0.25
selective_freeze_after_warmup = True
selective_log_diagnostics = False
selective_release_inactive_k_state = False
selective_selection_mode = "fraction"
selective_release_k_fraction = 0.0
selective_static_mask_path = ""
selective_static_mask_seed = -1  # -1 means use the current training seed
selective_static_mask_run_name = ""
selective_static_mask_target_release_k_fraction = -1.0
selective_shape_prior_policy = "mlp_c_proj_first"
selective_shape_prior_tiebreak = "late_layers_first"

sigma_block_size = 64
sigma_beta = 0.95
sigma_lambda_max = 0.25
sigma_lambda_start = 200
sigma_lambda_warmup = 1000
sigma_eps = 1e-4
sigma_refresh = 16
sigma_stat_source = "base"  # "base", "sigma", or "mixed"
sigma_stat_mixed_rho = 0.0
match_base_fro_norm = True
diagnostic_interval = 50

# Cheap Muon probe
cheap_muon_probe_enabled = False

# Update similarity probe
update_similarity_probe_enabled = False
update_similarity_probe_interval = 25
update_similarity_probe_start_step = 0
update_similarity_probe_stop_step = -1

# Local quadratic-score probe. It is off by default and writes detailed rows
# to local CSV artifacts rather than creating W&B time-series panels.
cproj_quadratic_probe_enabled = False
cproj_quadratic_probe_variant = "fresh"  # "fresh" (P1) or "temporal" (P2)
cproj_quadratic_probe_steps = "10000"
cproj_quadratic_probe_layers = "0,11,23"
cproj_quadratic_probe_modes = "none,scalar,diag,block4,full"
cproj_quadratic_probe_batch_size = 1
cproj_quadratic_probe_block_size = 128
cproj_quadratic_probe_build_repeats = 1
cproj_quadratic_probe_heldout_batches = 1
cproj_quadratic_probe_include_none_repeat = False
cproj_quadratic_probe_normmatch_modes = ""
cproj_quadratic_probe_line_search_multipliers = "0,0.25,0.5,1,2"
cproj_quadratic_probe_exact_hvp = True
cproj_quadratic_probe_exact_svd = False
cproj_quadratic_probe_exact_svd_repeats = 0
cproj_quadratic_probe_svd_compute_dtype = "float32"  # "float32" or "float64"
cproj_quadratic_probe_precision = "training"  # "training" or strict "float32"
cproj_quadratic_probe_line_search = True
cproj_quadratic_probe_heldout_line_search = True
cproj_quadratic_probe_output_dir = ""

# LR schedule
decay_lr = True
warmup_iters = 20
lr_decay_iters = 1000
min_lr = 1e-4

# System
device = "cuda" if torch.cuda.is_available() else "cpu"
cuda_memory_fraction = 0.0  # optional per-process CUDA allocator cap; 0 disables it
cuda_memory_budget_gib = 0.0  # optional absolute cap converted to fraction; 0 disables it
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else ("float16" if torch.cuda.is_available() else "float32")
)
compile = False
seed = 1337

# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py", encoding="utf-8").read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

quadratic_probe_steps = parse_int_csv(cproj_quadratic_probe_steps)
quadratic_probe_layers = parse_int_csv(cproj_quadratic_probe_layers)
quadratic_probe_modes = parse_mode_csv(cproj_quadratic_probe_modes)
cproj_k_layer_list = parse_int_csv(cproj_k_layers)
cproj_shadow_k_mode_list = [
    item.strip()
    for item in str(cproj_shadow_k_modes).split(",")
    if item.strip()
]
cproj_shadow_k_layer_list = parse_int_csv(cproj_shadow_k_layers)
invalid_cproj_k_layers = [
    layer for layer in cproj_k_layer_list if not 0 <= layer < n_layer
]
if invalid_cproj_k_layers:
    raise ValueError(
        f"cproj_k_layers must be in [0, {n_layer}); got {invalid_cproj_k_layers}"
    )
quadratic_probe_normmatch_modes = [
    item.strip()
    for item in str(cproj_quadratic_probe_normmatch_modes).split(",")
    if item.strip()
]
if cproj_quadratic_probe_variant == "fresh":
    quadratic_probe_effective_modes = expanded_probe_modes(
        quadratic_probe_modes,
        include_none_repeat=bool(cproj_quadratic_probe_include_none_repeat),
        normmatch_modes=quadratic_probe_normmatch_modes,
    )
elif cproj_quadratic_probe_variant == "temporal":
    validate_temporal_modes(quadratic_probe_modes)
    quadratic_probe_effective_modes = temporal_candidate_names(
        quadratic_probe_modes,
        include_exact_svd=bool(cproj_quadratic_probe_exact_svd),
    )
else:
    raise ValueError(
        "cproj_quadratic_probe_variant must be 'fresh' or 'temporal'; "
        f"got {cproj_quadratic_probe_variant!r}"
    )
quadratic_probe_line_search_multipliers = parse_float_csv(
    cproj_quadratic_probe_line_search_multipliers
)
if cproj_quadratic_probe_enabled:
    if not quadratic_probe_steps:
        raise ValueError("cproj_quadratic_probe_steps cannot be empty")
    if min(quadratic_probe_steps) < 0 or max(quadratic_probe_steps) >= max_iters:
        raise ValueError(
            "quadratic-probe steps must satisfy 0 <= step < max_iters; "
            f"got steps={quadratic_probe_steps}, max_iters={max_iters}"
        )
    if not quadratic_probe_layers:
        raise ValueError("cproj_quadratic_probe_layers cannot be empty")
    invalid_layers = [layer for layer in quadratic_probe_layers if not 0 <= layer < n_layer]
    if invalid_layers:
        raise ValueError(
            f"quadratic-probe layers must be in [0, {n_layer}); got {invalid_layers}"
        )
    if cproj_quadratic_probe_batch_size <= 0:
        raise ValueError("cproj_quadratic_probe_batch_size must be positive")
    if not 0 < cproj_quadratic_probe_block_size <= block_size:
        raise ValueError(
            "cproj_quadratic_probe_block_size must be in (0, block_size]; "
            f"got {cproj_quadratic_probe_block_size}, block_size={block_size}"
        )
    if cproj_quadratic_probe_build_repeats <= 0:
        raise ValueError("cproj_quadratic_probe_build_repeats must be positive")
    if cproj_quadratic_probe_heldout_batches < 0:
        raise ValueError("cproj_quadratic_probe_heldout_batches cannot be negative")
    if (
        cproj_quadratic_probe_heldout_line_search
        and cproj_quadratic_probe_heldout_batches <= 0
    ):
        raise ValueError(
            "heldout line search requires cproj_quadratic_probe_heldout_batches > 0"
        )
    if (
        not cproj_quadratic_probe_heldout_line_search
        and cproj_quadratic_probe_heldout_batches != 0
    ):
        raise ValueError(
            "set cproj_quadratic_probe_heldout_batches=0 when heldout line search is disabled"
        )
    if cproj_quadratic_probe_exact_svd:
        if not (
            0
            < cproj_quadratic_probe_exact_svd_repeats
            <= cproj_quadratic_probe_build_repeats
        ):
            raise ValueError(
                "exact SVD requires exact_svd_repeats in "
                "[1, cproj_quadratic_probe_build_repeats]"
            )
    elif cproj_quadratic_probe_exact_svd_repeats != 0:
        raise ValueError(
            "cproj_quadratic_probe_exact_svd_repeats must be 0 when exact SVD is disabled"
        )
    if cproj_quadratic_probe_svd_compute_dtype not in ("float32", "float64"):
        raise ValueError(
            "cproj_quadratic_probe_svd_compute_dtype must be "
            "'float32' or 'float64'"
        )
    if cproj_quadratic_probe_precision not in ("training", "float32"):
        raise ValueError(
            "cproj_quadratic_probe_precision must be 'training' or 'float32'"
        )
    if any(multiplier < 0 for multiplier in quadratic_probe_line_search_multipliers):
        raise ValueError("quadratic-probe line-search multipliers must be non-negative")
    if cproj_quadratic_probe_line_search and 0.0 not in quadratic_probe_line_search_multipliers:
        raise ValueError("quadratic-probe line search must include multiplier 0")
    if cproj_quadratic_probe_variant == "temporal":
        if optimizer_type != "cproj_k_mode_newton_muon":
            raise ValueError(
                "temporal P2 probe requires optimizer_type='cproj_k_mode_newton_muon'"
            )
        if cproj_k_mode != "none":
            raise ValueError(
                "temporal P2 probe requires a c_proj none trajectory"
            )
        if cproj_quadratic_probe_include_none_repeat:
            raise ValueError("temporal P2 probe does not use none_repeat")
        if quadratic_probe_normmatch_modes:
            raise ValueError("temporal P2 probe does not use P1 normmatch modes")
        required_shadows = set(quadratic_probe_modes) - {"none"}
        missing_shadows = required_shadows - set(cproj_shadow_k_mode_list)
        if missing_shadows:
            raise ValueError(
                "temporal P2 probe is missing shadow optimizer states for "
                f"{sorted(missing_shadows)}; cproj_shadow_k_modes="
                f"{cproj_shadow_k_mode_list}"
            )
        if cproj_shadow_k_layer_list:
            missing_shadow_layers = set(quadratic_probe_layers) - set(
                cproj_shadow_k_layer_list
            )
            if missing_shadow_layers:
                raise ValueError(
                    "temporal P2 probe layers are missing shadow histories: "
                    f"{sorted(missing_shadow_layers)}"
                )

torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
if device_type == "cpu" and dtype != "float32":
    print("Forcing dtype=float32 on CPU.")
    dtype = "float32"
if device_type == "cuda":
    cuda_device = torch.device(device)
    cuda_device_index = cuda_device.index
    if cuda_device_index is None:
        cuda_device_index = torch.cuda.current_device()
    memory_fraction = float(cuda_memory_fraction)
    if cuda_memory_budget_gib and cuda_memory_budget_gib > 0:
        total_memory_bytes = torch.cuda.get_device_properties(cuda_device_index).total_memory
        budget_bytes = float(cuda_memory_budget_gib) * (1024**3)
        memory_fraction = min(budget_bytes / total_memory_bytes, 1.0)
    if memory_fraction and memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(memory_fraction, cuda_device_index)
        print(f"Set CUDA per-process memory fraction to {memory_fraction:.4f}.")
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if compile and device_type == "cpu" and os.name == "nt" and shutil.which("cl") is None:
    print("Disabling torch.compile on Windows CPU because MSVC cl.exe was not found.")
    compile = False
if compile and optimizer_type in INPUT_TRACKED_OPTIMIZERS:
    print(f"Disabling torch.compile for {optimizer_type}; it uses Linear input-cache hooks.")
    compile = False
config["compile"] = compile

os.makedirs(out_dir, exist_ok=True)

data_dir = os.path.join("data", dataset)
train_path = os.path.join(data_dir, "train.bin")
val_path = os.path.join(data_dir, "val.bin")
if not os.path.exists(train_path) or not os.path.exists(val_path):
    raise FileNotFoundError(
        f"Missing dataset files under {data_dir}. Run: python data/{dataset}/prepare.py"
    )
train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
val_data = np.memmap(val_path, dtype=np.uint16, mode="r")


def get_batch_sized(split, sample_batch_size, sample_block_size):
    data = train_data if split == "train" else val_data
    if len(data) <= sample_block_size + 1:
        raise ValueError(
            f"{split} split is too small for sample_block_size={sample_block_size}"
        )
    ix = torch.randint(len(data) - sample_block_size - 1, (sample_batch_size,))
    x = torch.stack(
        [
            torch.from_numpy((data[i : i + sample_block_size]).astype(np.int64))
            for i in ix
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                (data[i + 1 : i + 1 + sample_block_size]).astype(np.int64)
            )
            for i in ix
        ]
    )
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def get_batch(split):
    return get_batch_sized(split, batch_size, block_size)


meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
meta = {}
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} from {meta_path}")

if require_real_tiny_shakespeare and dataset == "shakespeare_char":
    total_chars = len(train_data) + len(val_data)
    source_kind = meta.get("source_kind")
    if source_kind != "tinyshakespeare" or total_chars < 1_000_000 or meta_vocab_size is None or meta_vocab_size < 60:
        raise RuntimeError(
            "This formal config requires the real Tiny Shakespeare dataset. "
            f"Found source_kind={source_kind!r}, total_chars={total_chars:,}, "
            f"vocab_size={meta_vocab_size}. Run: python data/shakespeare_char/prepare.py --force-download"
        )

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=meta_vocab_size or 50304,
    dropout=dropout,
)

iter_num = 0
best_val_loss = 1e9
if init_from == "scratch":
    print("Initializing a new model from scratch")
    model = GPT(GPTConfig(**model_args))
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args.update(checkpoint["model_args"])
    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
else:
    raise ValueError("init_from must be 'scratch' or 'resume'")

model.to(device)

optimizer_args = {
    "optimizer_type": optimizer_type,
    "weight_decay": weight_decay,
    "learning_rate": learning_rate,
    "beta1": beta1,
    "beta2": beta2,
    "device_type": device_type,
    "muon_learning_rate": muon_learning_rate,
    "muon_momentum": muon_momentum,
    "muon_ns_steps": muon_ns_steps,
    "muon_nesterov": muon_nesterov,
    "muon_momentum_ema": muon_momentum_ema,
    "muon_split_qkv": muon_split_qkv,
    "muon_adjust_lr_for_shape": muon_adjust_lr_for_shape,
    "muon_ns_compute_dtype": muon_ns_compute_dtype,
    "matrix_weight_decay": matrix_weight_decay,
    "matrix_eps": matrix_eps,
    "input_beta": input_beta,
    "input_ridge": input_ridge,
    "input_refresh": input_refresh,
    "input_max_samples": None if input_max_samples <= 0 else input_max_samples,
    "input_first_refresh_step": input_first_refresh_step,
    "input_init_scale": input_init_scale,
    "input_init_inverse_scale": (
        None if input_init_inverse_scale <= 0 else input_init_inverse_scale
    ),
    "cproj_k_mode": cproj_k_mode,
    "cproj_k_layers": cproj_k_layer_list,
    "cproj_k_blocks": cproj_k_blocks,
    "cproj_k_reference_mode": cproj_k_reference_mode,
    "cproj_k_offdiag_alpha": cproj_k_offdiag_alpha,
    "cproj_shadow_k_modes": cproj_shadow_k_mode_list,
    "cproj_shadow_k_layers": cproj_shadow_k_layer_list,
    "selective_fraction": selective_fraction,
    "selective_min_active": selective_min_active,
    "selective_warmup_steps": selective_warmup_steps,
    "selective_score_interval": selective_score_interval,
    "selective_score_beta": selective_score_beta,
    "selective_score_threshold": selective_score_threshold,
    "selective_score_mode": selective_score_mode,
    "selective_cond_power": selective_cond_power,
    "selective_cost_power": selective_cost_power,
    "selective_freeze_after_warmup": selective_freeze_after_warmup,
    "selective_log_diagnostics": selective_log_diagnostics,
    "selective_release_inactive_k_state": selective_release_inactive_k_state,
    "selective_selection_mode": selective_selection_mode,
    "selective_release_k_fraction": selective_release_k_fraction,
    "selective_static_mask_path": selective_static_mask_path,
    "selective_static_mask_seed": selective_static_mask_seed,
    "selective_static_mask_run_name": selective_static_mask_run_name,
    "selective_static_mask_target_release_k_fraction": selective_static_mask_target_release_k_fraction,
    "selective_shape_prior_policy": selective_shape_prior_policy,
    "selective_shape_prior_tiebreak": selective_shape_prior_tiebreak,
    "cheap_muon_probe_enabled": cheap_muon_probe_enabled,
    "update_similarity_probe_enabled": update_similarity_probe_enabled,
    "update_similarity_probe_interval": update_similarity_probe_interval,
    "update_similarity_probe_start_step": update_similarity_probe_start_step,
    "update_similarity_probe_stop_step": update_similarity_probe_stop_step,
    "seed": seed,
    "sigma_block_size": sigma_block_size,
    "sigma_beta": sigma_beta,
    "sigma_lambda_max": sigma_lambda_max,
    "sigma_lambda_start": sigma_lambda_start,
    "sigma_lambda_warmup": sigma_lambda_warmup,
    "sigma_eps": sigma_eps,
    "sigma_refresh": sigma_refresh,
    "sigma_stat_source": sigma_stat_source,
    "sigma_stat_mixed_rho": sigma_stat_mixed_rho,
    "match_base_fro_norm": match_base_fro_norm,
    "diagnostic_interval": diagnostic_interval,
}
optimizer, hook_handles = build_optimizer(model, optimizer_args)
stats_owner_for_probe_check = getattr(optimizer, "optimizer_muon", optimizer)
if update_similarity_probe_enabled:
    if optimizer_type != "selective_newton_muon":
        raise RuntimeError("update_similarity_probe requires optimizer_type=selective_newton_muon")
    probe_group_flags = [
        bool(group.get("update_similarity_probe_enabled", False))
        for group in getattr(stats_owner_for_probe_check, "param_groups", [])
    ]
    print(f"update similarity probe optimizer group flags: {probe_group_flags}")
    if not probe_group_flags or not all(probe_group_flags):
        raise RuntimeError(
            "update_similarity_probe_enabled=True in train.py config, but it was not "
            "passed into the matrix optimizer param groups. Check optimizer_args in train.py."
        )

if init_from == "resume":
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except Exception as exc:
        print(f"Warning: could not resume optimizer state, starting fresh: {exc}")
    checkpoint = None

if compile:
    print("compiling the model...")
    model = torch.compile(model)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16" and device_type == "cuda"))


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def cuda_memory_log():
    if device_type != "cuda":
        return {}
    global full_run_max_memory_allocated_mib
    global full_run_max_memory_reserved_mib
    global post_release_peak_tracking
    torch.cuda.synchronize()
    cuda_device = torch.device(device)
    mib = 1024 * 1024
    allocated = torch.cuda.memory_allocated(cuda_device) / mib
    reserved = torch.cuda.memory_reserved(cuda_device) / mib
    max_allocated = torch.cuda.max_memory_allocated(cuda_device) / mib
    max_reserved = torch.cuda.max_memory_reserved(cuda_device) / mib
    full_run_max_memory_allocated_mib = max(full_run_max_memory_allocated_mib, max_allocated)
    full_run_max_memory_reserved_mib = max(full_run_max_memory_reserved_mib, max_reserved)
    stats = {
        "cuda/memory_allocated_mib": allocated,
        "cuda/memory_reserved_mib": reserved,
        "cuda/max_memory_allocated_mib": max_allocated,
        "cuda/max_memory_reserved_mib": max_reserved,
        "cuda/window_max_memory_allocated_mib": max_allocated,
        "cuda/window_max_memory_reserved_mib": max_reserved,
        "cuda/full_run_max_memory_allocated_mib": full_run_max_memory_allocated_mib,
        "cuda/full_run_max_memory_reserved_mib": full_run_max_memory_reserved_mib,
        "system/cuda_memory_allocated_mib": allocated,
        "system/cuda_memory_reserved_mib": reserved,
        "system/max_cuda_memory_allocated_mib": max_allocated,
        "system/max_cuda_memory_reserved_mib": max_reserved,
    }
    if post_release_peak_tracking:
        stats.update(
            {
                "cuda/post_release_max_memory_allocated_mib": max_allocated,
                "cuda/post_release_max_memory_reserved_mib": max_reserved,
                "cuda/post_release_memory_allocated_mib": allocated,
                "cuda/post_release_memory_reserved_mib": reserved,
            }
        )
    return stats


PAPER_WANDB_CORE_KEYS = {
    # W&B already uses the supplied step as the x-axis, so a separate `iter`
    # series is redundant. Keep only the curves and resource counters that are
    # routinely used in experiment decisions.
    "train/loss_step",
    "val/loss",
    "lr/adamw",
    "lr/matrix",
    "time_elapsed",
    "cuda/memory_allocated_mib",
    "cuda/full_run_max_memory_allocated_mib",
    "matrix/k_state_bytes",
    "matrix/k_state_released_fraction",
    "matrix/cproj_k_state_bytes",
    "matrix/non_cproj_k_state_bytes",
    "matrix/cproj_target_layer_count",
    "matrix/cproj_target_layers_all",
    "matrix/cproj_mode_applied_params",
    "matrix/cproj_none_params",
    "matrix/cproj_full_params",
    "matrix/cproj_diag_params",
}

PAPER_WANDB_UPDATE_SIMILARITY_KEYS = {
    "matrix/update_similarity_probe_params",
    "matrix/update_similarity_probe_samples",
    "matrix/update_similarity_probe_update_cos_mean",
    "matrix/update_similarity_probe_relative_gap_mean",
    "matrix/update_similarity_probe_h4_h8_update_cos_mean",
    "matrix/update_similarity_probe_h4_h8_relative_gap_mean",
    "matrix/update_similarity_probe_h2_h9_update_cos_mean",
    "matrix/update_similarity_probe_h2_h9_relative_gap_mean",
    "matrix/update_similarity_probe_h3_h8_update_cos_mean",
    "matrix/update_similarity_probe_h3_h8_relative_gap_mean",
}


def paper_wandb_keys():
    keys = set(PAPER_WANDB_CORE_KEYS)
    if update_similarity_probe_enabled:
        keys.update(PAPER_WANDB_UPDATE_SIMILARITY_KEYS)
    return keys


def filter_wandb_log_dict(log_dict):
    if wandb_log_profile == "full":
        return log_dict
    if wandb_log_profile == "paper":
        allowed_keys = paper_wandb_keys()
        return {key: value for key, value in log_dict.items() if key in allowed_keys}
    raise ValueError(f"Unknown wandb_log_profile: {wandb_log_profile}")


def reset_cuda_post_release_peak():
    if device_type != "cuda":
        return
    global post_release_peak_tracking
    cuda_memory_log()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    post_release_peak_tracking = True


def matrix_stats_for_wandb(stats):
    filtered = {}
    for key, value in stats.items():
        if key == "step":
            continue
        if key == "selective_log_diagnostics":
            continue
        if key.startswith("update_similarity_probe_") and stats.get(
            "update_similarity_probe_samples", 0
        ) <= 0:
            continue
        if optimizer_type not in ("diag_sigma_newton_muon", "block_sigma_newton_muon"):
            if key.startswith("sigma_") or key in ("num_blocks", "sigma_offdiag_mean"):
                continue
        if optimizer_type not in ("selective_newton_muon", "cproj_k_mode_newton_muon"):
            if (
                key.startswith("selective_")
                or key.startswith("target_")
                or key in (
                    "full_k_state_bytes",
                    "selected_k_state_bytes",
                    "inactive_k_state_bytes",
                    "inactive_k_state_fraction",
                    "release_budget_error_bytes",
                )
            ):
                continue
        if key.startswith("cos_") and not stats.get("selective_log_diagnostics", 0):
            continue
        filtered[f"matrix/{key}"] = value
    return filtered


def format_matrix_stats(stats):
    parts = [f"active={stats.get('active_params', 0)}"]
    if "k_state_bytes" in stats:
        parts.append(f"K_state={stats.get('k_state_bytes', 0) / (1024 * 1024):.2f}MiB")
    if stats.get("input_k_cond_count", 0) or stats.get("input_k_cond_mean", 0.0):
        parts.append(
            f"K_cond={stats.get('input_k_cond_mean', 0.0):.2f}/"
            f"{stats.get('input_k_cond_max', 0.0):.2f}"
        )
    if optimizer_type == "selective_newton_muon":
        parts.append(f"newton_active={stats.get('selective_newton_params', 0):.0f}")
        parts.append(f"released={100.0 * stats.get('k_state_released_fraction', 0.0):.1f}%")
        parts.append(f"score={stats.get('selective_score_mean', 0.0):.4f}")
    if optimizer_type in ("diag_sigma_newton_muon", "block_sigma_newton_muon"):
        parts.append(f"sigma_lambda={stats.get('sigma_lambda', 0.0):.3f}")
        parts.append(
            f"sigma_aniso={stats.get('sigma_anisotropy_mean', 0.0):.2f}/"
            f"{stats.get('sigma_anisotropy_max', 0.0):.2f}"
        )
        parts.append(f"offdiag={stats.get('sigma_offdiag_mean', 0.0):.3f}")
    if stats.get("cheap_muon_probe_params", 0):
        parts.append(f"probe_layers={stats.get('cheap_muon_probe_params', 0):.0f}")
        parts.append(f"g_muon_cos={stats.get('cheap_muon_probe_grad_muon_cos_mean', 0.0):.4f}")
    if stats.get("update_similarity_probe_params", 0):
        parts.append(f"upd_cos={stats.get('update_similarity_probe_update_cos_mean', 0.0):.4f}")
        parts.append(f"h2h9_cos={stats.get('update_similarity_probe_h2_h9_update_cos_mean', 0.0):.4f}")
    if stats.get("selective_log_diagnostics", 0):
        parts.append(f"cos(update,base)={stats.get('cos_update_vs_base', 0.0):.3f}")
    return ", ".join(parts)


def get_lr(it):
    if not decay_lr:
        return learning_rate
    if it < warmup_iters:
        return learning_rate * it / max(1, warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log:
    import wandb

    wandb_tags_list = [tag.strip() for tag in wandb_tags.split(",") if tag.strip()]
    wandb_run = wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=config,
        mode=wandb_mode,
        group=wandb_group,
        job_type=optimizer_type,
        tags=wandb_tags_list,
        dir=os.getcwd(),
    )
    if getattr(wandb_run, "url", None):
        print(f"W&B run: {wandb_run.url}")

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"optimizer_type={optimizer_type}, device={device}, dtype={dtype}")

x, y = get_batch("train")
full_run_max_memory_allocated_mib = 0.0
full_run_max_memory_reserved_mib = 0.0
post_release_peak_tracking = False
if device_type == "cuda":
    torch.cuda.reset_peak_memory_stats(torch.device(device))
start_time = time.time()
local_time = time.time()
running_mfu = -1.0
update_similarity_probe_guard_checked = False
quadratic_probe_direction_rows = []
quadratic_probe_line_rows = []
quadratic_probe_metadata = []
quadratic_probe_completed_steps = set()
if cproj_quadratic_probe_enabled:
    if cproj_quadratic_probe_output_dir:
        quadratic_probe_export_dir = os.path.abspath(cproj_quadratic_probe_output_dir)
    else:
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        artifact_root = os.path.dirname(os.path.dirname(backend_dir))
        results_root = os.environ.get(
            "SNM_RESULTS_ROOT", os.path.join(artifact_root, "runs")
        )
        quadratic_probe_export_dir = os.path.join(
            results_root,
            "06_kstate_spectrum",
            "quadratic_probe",
            safe_export_name(wandb_run_name or os.path.basename(os.path.normpath(out_dir))),
        )
    quadratic_probe_config = {
        "seed": int(seed),
        "dataset": dataset,
        "wandb_run_name": wandb_run_name,
        "optimizer_type": optimizer_type,
        "probe_variant": cproj_quadratic_probe_variant,
        "trajectory_cproj_k_mode": cproj_k_mode,
        "shadow_cproj_k_modes": cproj_shadow_k_mode_list,
        "shadow_cproj_k_layers": cproj_shadow_k_layer_list,
        "steps": quadratic_probe_steps,
        "layers": quadratic_probe_layers,
        "base_modes": quadratic_probe_modes,
        "modes": quadratic_probe_effective_modes,
        "probe_batch_size": int(cproj_quadratic_probe_batch_size),
        "probe_block_size": int(cproj_quadratic_probe_block_size),
        "build_repeats": int(cproj_quadratic_probe_build_repeats),
        "heldout_batches": int(cproj_quadratic_probe_heldout_batches),
        "include_none_repeat": bool(cproj_quadratic_probe_include_none_repeat),
        "normmatch_modes": quadratic_probe_normmatch_modes,
        "line_search_multipliers": quadratic_probe_line_search_multipliers,
        "exact_hvp": bool(cproj_quadratic_probe_exact_hvp),
        "exact_svd": bool(cproj_quadratic_probe_exact_svd),
        "exact_svd_repeats": int(cproj_quadratic_probe_exact_svd_repeats),
        "svd_compute_dtype": cproj_quadratic_probe_svd_compute_dtype,
        "probe_precision": cproj_quadratic_probe_precision,
        "tf32_disabled_for_float32_probe": (
            cproj_quadratic_probe_precision == "float32"
        ),
        "line_search": bool(cproj_quadratic_probe_line_search),
        "heldout_line_search": bool(cproj_quadratic_probe_heldout_line_search),
        "probe_model_mode": "eval",
        "input_ridge": float(input_ridge),
        "cproj_k_blocks": int(cproj_k_blocks),
        "muon_ns_steps": int(muon_ns_steps),
        "matrix_eps": float(matrix_eps),
        "muon_learning_rate": float(muon_learning_rate),
        "model": {
            "n_layer": int(n_layer),
            "n_head": int(n_head),
            "n_embd": int(n_embd),
            "block_size": int(block_size),
        },
    }
    print(
        "quadratic probe enabled: "
        f"variant={cproj_quadratic_probe_variant}, "
        f"steps={quadratic_probe_steps}, layers={quadratic_probe_layers}, "
        f"modes={quadratic_probe_effective_modes}, "
        f"build_repeats={cproj_quadratic_probe_build_repeats}, "
        f"heldout_batches={cproj_quadratic_probe_heldout_batches}, "
        f"precision={cproj_quadratic_probe_precision}, "
        f"output={quadratic_probe_export_dir}"
    )

    def write_current_quadratic_probe_artifacts(expected_steps):
        if cproj_quadratic_probe_variant == "temporal":
            return write_temporal_quadratic_probe_artifacts(
                quadratic_probe_export_dir,
                quadratic_probe_direction_rows,
                quadratic_probe_line_rows,
                config=quadratic_probe_config,
                metadata=quadratic_probe_metadata,
                expected_steps=expected_steps,
                expected_layers=quadratic_probe_layers,
                modes=quadratic_probe_modes,
                expected_build_repeats=cproj_quadratic_probe_build_repeats,
                expected_heldout_batches=cproj_quadratic_probe_heldout_batches,
                line_search_multipliers=quadratic_probe_line_search_multipliers,
                exact_hvp=cproj_quadratic_probe_exact_hvp,
                exact_svd_repeats=cproj_quadratic_probe_exact_svd_repeats,
                probe_precision=cproj_quadratic_probe_precision,
                line_search=cproj_quadratic_probe_line_search,
                svd_compute_dtype=cproj_quadratic_probe_svd_compute_dtype,
            )
        return write_quadratic_probe_artifacts(
            quadratic_probe_export_dir,
            quadratic_probe_direction_rows,
            quadratic_probe_line_rows,
            config=quadratic_probe_config,
            metadata=quadratic_probe_metadata,
            expected_steps=expected_steps,
            expected_layers=quadratic_probe_layers,
            expected_modes=quadratic_probe_effective_modes,
            expected_build_repeats=cproj_quadratic_probe_build_repeats,
            expected_heldout_batches=cproj_quadratic_probe_heldout_batches,
            line_search_multipliers=quadratic_probe_line_search_multipliers,
            exact_hvp=cproj_quadratic_probe_exact_hvp,
            exact_svd_repeats=cproj_quadratic_probe_exact_svd_repeats,
            include_none_repeat=cproj_quadratic_probe_include_none_repeat,
            normmatch_modes=quadratic_probe_normmatch_modes,
            probe_precision=cproj_quadratic_probe_precision,
            line_search=cproj_quadratic_probe_line_search,
        )

while True:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        if "ns_steps" in param_group:
            param_group["lr"] = muon_learning_rate
        else:
            param_group["lr"] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            eval_log = {
                "iter": iter_num,
                "train/loss": losses["train"],
                "val/loss": losses["val"],
                "lr/adamw": lr,
                "lr/matrix": muon_learning_rate if optimizer_type in MUON_FAMILY_OPTIMIZERS else 0.0,
                "mfu": running_mfu * 100.0,
            }
            eval_log.update(cuda_memory_log())
            wandb.log(filter_wandb_log_dict(eval_log), step=iter_num)
        is_best = losses["val"] < best_val_loss
        if is_best:
            best_val_loss = losses["val"]
            if wandb_log:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_iter"] = iter_num
        if (
            save_checkpoint
            and (is_best or always_save_checkpoint)
            and iter_num > 0
        ):
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if hasattr(optimizer, "state_dict") else None,
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    if (
        cproj_quadratic_probe_enabled
        and iter_num in quadratic_probe_steps
        and iter_num not in quadratic_probe_completed_steps
    ):
        print(f"running c_proj quadratic probe at step {iter_num}")
        if device_type == "cuda":
            # Preserve the ordinary training peak before the diagnostic allocates
            # its second-order graph, then start an isolated probe peak window.
            cuda_memory_log()
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if device_type == "cuda" else None
        )
        try:
            probe_batches = [
                get_batch_sized(
                    "train",
                    cproj_quadratic_probe_batch_size,
                    cproj_quadratic_probe_block_size,
                )
                for _ in range(cproj_quadratic_probe_build_repeats)
            ]
            heldout_probe_batches = []
            if cproj_quadratic_probe_heldout_line_search:
                heldout_probe_batches = [
                    get_batch_sized(
                        "train",
                        cproj_quadratic_probe_batch_size,
                        cproj_quadratic_probe_block_size,
                    )
                    for _ in range(cproj_quadratic_probe_heldout_batches)
                ]
            probe_dtype = (
                torch.float32
                if cproj_quadratic_probe_precision == "float32"
                else ptdtype
            )
            svd_compute_dtype = (
                torch.float64
                if cproj_quadratic_probe_svd_compute_dtype == "float64"
                else torch.float32
            )
            if cproj_quadratic_probe_variant == "temporal":
                new_rows, new_line_rows, probe_meta = (
                    run_cproj_temporal_quadratic_probe(
                        model,
                        stats_owner_for_probe_check,
                        probe_batches,
                        heldout_probe_batches,
                        step=iter_num,
                        layers=quadratic_probe_layers,
                        modes=quadratic_probe_modes,
                        ridge=input_ridge,
                        blocks=cproj_k_blocks,
                        ns_steps=muon_ns_steps,
                        matrix_eps=matrix_eps,
                        matrix_learning_rate=muon_learning_rate,
                        line_search_multipliers=(
                            quadratic_probe_line_search_multipliers
                        ),
                        exact_hvp=cproj_quadratic_probe_exact_hvp,
                        exact_svd_repeats=(
                            cproj_quadratic_probe_exact_svd_repeats
                        ),
                        line_search=cproj_quadratic_probe_line_search,
                        device_type=device_type,
                        autocast_dtype=probe_dtype,
                        svd_compute_dtype=svd_compute_dtype,
                    )
                )
            else:
                new_rows, new_line_rows, probe_meta = (
                    run_cproj_quadratic_probe_repeated(
                        model,
                        probe_batches,
                        heldout_probe_batches,
                        step=iter_num,
                        layers=quadratic_probe_layers,
                        modes=quadratic_probe_modes,
                        include_none_repeat=(
                            cproj_quadratic_probe_include_none_repeat
                        ),
                        normmatch_modes=quadratic_probe_normmatch_modes,
                        ridge=input_ridge,
                        blocks=cproj_k_blocks,
                        ns_steps=muon_ns_steps,
                        matrix_eps=matrix_eps,
                        matrix_learning_rate=muon_learning_rate,
                        line_search_multipliers=(
                            quadratic_probe_line_search_multipliers
                        ),
                        exact_hvp=cproj_quadratic_probe_exact_hvp,
                        exact_svd=cproj_quadratic_probe_exact_svd,
                        exact_svd_repeats=(
                            cproj_quadratic_probe_exact_svd_repeats
                        ),
                        line_search=cproj_quadratic_probe_line_search,
                        device_type=device_type,
                        autocast_dtype=probe_dtype,
                    )
                )
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

        quadratic_probe_direction_rows.extend(new_rows)
        quadratic_probe_line_rows.extend(new_line_rows)
        quadratic_probe_metadata.append(probe_meta)
        quadratic_probe_completed_steps.add(iter_num)
        probe_paths = write_current_quadratic_probe_artifacts(
            sorted(quadratic_probe_completed_steps)
        )
        print(
            "quadratic probe completed: "
            f"rows={len(new_rows)}, line_search_rows={len(new_line_rows)}, "
            f"seconds={probe_meta['probe_seconds']:.1f}, "
            f"peak={probe_meta['probe_peak_memory_mib']:.1f} MiB"
        )
        probe_long_key = (
            "temporal_quadratic_probe_long"
            if cproj_quadratic_probe_variant == "temporal"
            else "quadratic_probe_long"
        )
        print(f"quadratic probe artifacts: {probe_paths[probe_long_key]}")
        if device_type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Exclude the diagnostic-only HVP allocations from subsequent
            # ordinary-training peak memory logs.
            torch.cuda.reset_peak_memory_stats(torch.device(device))
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            _, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
        x, y = get_batch("train")
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    stats_owner = getattr(optimizer, "optimizer_muon", optimizer)
    consume_release_event = getattr(stats_owner, "consume_state_release_event", None)
    if callable(consume_release_event) and consume_release_event():
        reset_cuda_post_release_peak()
        print("reset CUDA post-release peak memory stats after optimizer state release")

    t1 = time.time()
    dt = t1 - local_time
    local_time = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        if iter_num >= 5:
            running_mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

        log_dict = {
            "iter": iter_num,
            "train/loss_step": lossf,
            "time_elapsed": time.time() - start_time,
        }
        log_dict.update(cuda_memory_log())
        stats_owner = getattr(optimizer, "optimizer_muon", optimizer)
        if hasattr(stats_owner, "last_stats") and stats_owner.last_stats:
            stats = stats_owner.last_stats
            if stats.get("active_params", 0) > 0:
                if (
                    update_similarity_probe_enabled
                    and not update_similarity_probe_guard_checked
                    and iter_num
                    >= int(update_similarity_probe_start_step) + int(update_similarity_probe_interval)
                    and stats.get("update_similarity_probe_samples", 0) <= 0
                ):
                    raise RuntimeError(
                        "update_similarity_probe_enabled=True but no update similarity "
                        "samples were collected after the first probe interval. "
                        "The most likely cause is a stale optimizer_factory.py or "
                        "optimizers.py on the machine running this job."
                    )
                if update_similarity_probe_enabled and stats.get("update_similarity_probe_samples", 0) > 0:
                    update_similarity_probe_guard_checked = True
                print(" -> [matrix optimizer] " + format_matrix_stats(stats))
                log_dict.update(matrix_stats_for_wandb(stats))
        if wandb_log:
            wandb.log(filter_wandb_log_dict(log_dict), step=iter_num)

    iter_num += 1
    if iter_num >= max_iters:
        break

if cproj_quadratic_probe_enabled:
    missing_probe_steps = sorted(set(quadratic_probe_steps) - quadratic_probe_completed_steps)
    if missing_probe_steps:
        raise RuntimeError(f"quadratic probe did not execute at steps {missing_probe_steps}")
    probe_paths = write_current_quadratic_probe_artifacts(quadratic_probe_steps)
    probe_summary_key = (
        "temporal_quadratic_probe_summary"
        if cproj_quadratic_probe_variant == "temporal"
        else "quadratic_probe_summary"
    )
    print(f"final quadratic probe summary: {probe_paths[probe_summary_key]}")

for handle in hook_handles:
    handle.remove()

stats_owner = getattr(optimizer, "optimizer_muon", optimizer)
if hasattr(stats_owner, "get_selective_layer_report"):
    report_rows = stats_owner.get_selective_layer_report()
    if report_rows:
        final_stats = getattr(stats_owner, "last_stats", {}) or {}
        actual_release_fraction = float(
            final_stats.get(
                "k_state_released_fraction",
                final_stats.get("inactive_k_state_fraction", 0.0),
            )
        )
        report_context = {
            "seed": int(seed),
            "dataset": dataset,
            "wandb_project": wandb_project,
            "wandb_group": wandb_group,
            "wandb_run_name": wandb_run_name,
            "optimizer_type": optimizer_type,
            "target_release_k_fraction": float(selective_release_k_fraction),
            "static_mask_path": selective_static_mask_path,
            "static_mask_seed": int(selective_static_mask_seed),
            "static_mask_run_name": selective_static_mask_run_name,
            "static_mask_target_release_k_fraction": float(selective_static_mask_target_release_k_fraction),
            "shape_prior_policy": selective_shape_prior_policy,
            "shape_prior_tiebreak": selective_shape_prior_tiebreak,
            "actual_release_k_fraction": actual_release_fraction,
            "actual_release_k_state_bytes": int(final_stats.get("k_state_released_bytes", 0)),
            "current_k_state_bytes": int(final_stats.get("k_state_bytes", 0)),
            "full_k_state_bytes": int(final_stats.get("k_state_full_bytes", 0)),
        }
        report_rows = [{**report_context, **row, "actual_release_k_fraction": actual_release_fraction} for row in report_rows]
        report_export_dir = csv_export_dir
        if not report_export_dir:
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            artifact_root = os.path.dirname(os.path.dirname(backend_dir))
            results_root = os.environ.get(
                "SNM_RESULTS_ROOT", os.path.join(artifact_root, "runs")
            )
            report_export_dir = os.path.join(
                results_root,
                "run_artifacts",
                safe_export_name(wandb_run_name or os.path.basename(os.path.normpath(out_dir))),
            )
        os.makedirs(report_export_dir, exist_ok=True)
        report_path = os.path.join(report_export_dir, "selective_layer_report.csv")
        columns = list(report_rows[0].keys())
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"wrote selective layer report to {report_path}")
        if wandb_log and wandb_log_tables:
            table = wandb.Table(
                columns=columns,
                data=[[row.get(column) for column in columns] for row in report_rows],
            )
            wandb.log({"matrix/selective_layer_report": table}, step=iter_num)
            wandb.save(report_path)
        elif wandb_log:
            wandb.save(report_path)

if hasattr(stats_owner, "get_cheap_muon_probe_report"):
    probe_rows = stats_owner.get_cheap_muon_probe_report()
    if probe_rows:
        probe_context = {
            "seed": int(seed),
            "dataset": dataset,
            "wandb_project": wandb_project,
            "wandb_group": wandb_group,
            "wandb_run_name": wandb_run_name,
            "optimizer_type": optimizer_type,
            "probe_steps": int(max_iters),
            "cheap_muon_probe_enabled": bool(cheap_muon_probe_enabled),
        }
        probe_rows = [{**probe_context, **row} for row in probe_rows]
        probe_export_dir = csv_export_dir
        if not probe_export_dir:
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            artifact_root = os.path.dirname(os.path.dirname(backend_dir))
            results_root = os.environ.get(
                "SNM_RESULTS_ROOT", os.path.join(artifact_root, "runs")
            )
            probe_export_dir = os.path.join(
                results_root,
                "run_artifacts",
                safe_export_name(wandb_run_name or os.path.basename(os.path.normpath(out_dir))),
            )
        os.makedirs(probe_export_dir, exist_ok=True)
        probe_path = os.path.join(probe_export_dir, "cheap_muon_probe_report.csv")
        probe_columns = list(probe_rows[0].keys())
        with open(probe_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=probe_columns)
            writer.writeheader()
            writer.writerows(probe_rows)
        print(f"wrote cheap Muon probe report to {probe_path}")
        if wandb_log and wandb_log_tables:
            table = wandb.Table(
                columns=probe_columns,
                data=[[row.get(column) for column in probe_columns] for row in probe_rows],
            )
            wandb.log({"matrix/cheap_muon_probe_report": table}, step=iter_num)
            wandb.save(probe_path)
        elif wandb_log:
            wandb.save(probe_path)

if hasattr(stats_owner, "get_update_similarity_probe_report"):
    similarity_rows = stats_owner.get_update_similarity_probe_report()
    if similarity_rows:
        similarity_context = {
            "seed": int(seed),
            "dataset": dataset,
            "wandb_project": wandb_project,
            "wandb_group": wandb_group,
            "wandb_run_name": wandb_run_name,
            "optimizer_type": optimizer_type,
            "probe_steps": int(max_iters),
            "update_similarity_probe_enabled": bool(update_similarity_probe_enabled),
            "update_similarity_probe_interval": int(update_similarity_probe_interval),
            "update_similarity_probe_start_step": int(update_similarity_probe_start_step),
            "update_similarity_probe_stop_step": int(update_similarity_probe_stop_step),
        }
        similarity_rows = [{**similarity_context, **row} for row in similarity_rows]
        similarity_export_dir = csv_export_dir
        if not similarity_export_dir:
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            artifact_root = os.path.dirname(os.path.dirname(backend_dir))
            results_root = os.environ.get(
                "SNM_RESULTS_ROOT", os.path.join(artifact_root, "runs")
            )
            similarity_export_dir = os.path.join(
                results_root,
                "run_artifacts",
                safe_export_name(wandb_run_name or os.path.basename(os.path.normpath(out_dir))),
            )
        os.makedirs(similarity_export_dir, exist_ok=True)
        similarity_path = os.path.join(similarity_export_dir, "update_similarity_probe_report.csv")
        similarity_columns = list(similarity_rows[0].keys())
        with open(similarity_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=similarity_columns)
            writer.writeheader()
            writer.writerows(similarity_rows)
        print(f"wrote update similarity probe report to {similarity_path}")
        if wandb_log and wandb_log_tables:
            table = wandb.Table(
                columns=similarity_columns,
                data=[[row.get(column) for column in similarity_columns] for row in similarity_rows],
            )
            wandb.log({"matrix/update_similarity_probe_report": table}, step=iter_num)
            wandb.save(similarity_path)
        elif wandb_log:
            wandb.save(similarity_path)
    elif update_similarity_probe_enabled:
        print(
            "warning: update_similarity_probe_enabled=True but no update similarity "
            "samples were collected; check optimizer_factory.py pass-through and "
            "SelectiveNewtonMuon probe settings"
        )

if wandb_log:
    final_memory = filter_wandb_log_dict(cuda_memory_log())
    for key, value in final_memory.items():
        wandb.run.summary[key] = value

if wandb_log:
    wandb.finish()
