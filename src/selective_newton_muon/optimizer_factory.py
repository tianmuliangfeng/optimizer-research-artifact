import csv
import os
import re

import torch

from .optimizers import (
    BlockSigmaNewtonMuon,
    CProjKModeNewtonMuon,
    DiagSigmaNewtonMuon,
    HybridOptimizer,
    NewtonMuon,
    PureMuon,
    SelectiveNewtonMuon,
)


MUON_FAMILY_OPTIMIZERS = (
    "muon",
    "newton_muon",
    "selective_newton_muon",
    "diag_sigma_newton_muon",
    "block_sigma_newton_muon",
    "cproj_k_mode_newton_muon",
)

INPUT_TRACKED_OPTIMIZERS = (
    "newton_muon",
    "selective_newton_muon",
    "diag_sigma_newton_muon",
    "block_sigma_newton_muon",
    "cproj_k_mode_newton_muon",
)


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def load_selective_static_mask(args):
    path = args.get("selective_static_mask_path", "")
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"selective_static_mask_path not found: {path}")

    run_name = str(args.get("selective_static_mask_run_name", "") or "")
    seed_filter = int(args.get("selective_static_mask_seed", -1))
    if seed_filter < 0:
        seed_filter = int(args.get("seed", -1))
    target_release = float(args.get("selective_static_mask_target_release_k_fraction", -1.0))

    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if run_name and row.get("wandb_run_name") != run_name:
                continue
            if seed_filter >= 0 and row.get("seed", "") != str(seed_filter):
                continue
            if target_release >= 0.0:
                try:
                    row_target = float(row.get("target_release_k_fraction", "nan"))
                except ValueError:
                    row_target = float("nan")
                if abs(row_target - target_release) > 1e-8:
                    continue
            rows.append(row)

    if not rows:
        raise ValueError(
            "No rows matched selective static mask filters: "
            f"path={path}, run_name={run_name!r}, seed={seed_filter}, target_release={target_release}"
        )

    report_names = {row["name"] for row in rows if row.get("name")}
    newton_names = {row["name"] for row in rows if row.get("name") and _truthy(row.get("selected", ""))}
    rank_by_name = {}
    for row in rows:
        name = row.get("name")
        if not name:
            continue
        try:
            rank_by_name[name] = int(float(row.get("rank", 0)))
        except ValueError:
            pass

    label = (
        f"{os.path.basename(path)}|seed={seed_filter}|"
        f"target={target_release if target_release >= 0.0 else 'any'}"
    )
    if run_name:
        label += f"|run={run_name}"
    return {
        "path": path,
        "newton_names": newton_names,
        "report_names": report_names,
        "rank_by_name": rank_by_name,
        "label": label,
        "rows": rows,
    }


def _input_cov_full_state_bytes(param, include_eye=True) -> int:
    n = int(param.shape[1])
    tensor_count = 3 if include_eye else 2
    return n * n * torch.empty((), dtype=torch.float32, device=param.device).element_size() * tensor_count


def _layer_index(name: str) -> int:
    match = re.search(r"transformer\.h\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def cproj_param_needs_input_hook(
    name: str,
    *,
    cproj_k_mode: str,
    cproj_k_layers=(),
    cproj_shadow_k_modes=(),
    cproj_shadow_k_layers=(),
) -> bool:
    """Return whether a matrix needs its module input cached for c_proj K modes.

    An empty ``cproj_k_layers`` applies the requested mode to every
    ``mlp.c_proj``.  Non-targeted ``mlp.c_proj`` matrices stay on the full
    Newton path and therefore still need hooks.  A targeted ``none`` matrix
    needs no hook unless a diagnostic shadow state is requested for it.
    """
    if ".mlp.c_proj.weight" not in name:
        return True
    layer = _layer_index(name)
    target_layers = set(int(item) for item in cproj_k_layers)
    is_target = not target_layers or layer in target_layers
    if not is_target or cproj_k_mode != "none":
        return True
    shadow_modes = tuple(cproj_shadow_k_modes)
    if not shadow_modes:
        return False
    shadow_layers = set(int(item) for item in cproj_shadow_k_layers)
    return not shadow_layers or layer in shadow_layers


def _shape_prior_release_priority(name: str, policy: str) -> int:
    valid_policies = (
        "cheap",
        "mlp_c_proj_first",
        "mlp_c_proj_early_band",
        "mlp_c_proj_middle_band",
        "mlp_c_proj_late_band",
        "mlp_c_proj_edge_band",
    )
    if policy not in valid_policies:
        raise ValueError(f"Unknown selective_shape_prior_policy: {policy}")
    if ".mlp.c_proj.weight" in name:
        return 400
    if ".mlp.c_fc.weight" in name:
        return 200
    if ".attn.c_proj.weight" in name:
        return 50
    if ".attn.c_attn.weight" in name:
        return 25
    return 0


def _shape_prior_tie_key(layer_idx: int, layer_count: int, policy: str, tiebreak: str):
    if policy == "mlp_c_proj_early_band":
        return (layer_idx,)
    if policy == "mlp_c_proj_late_band":
        return (-layer_idx,)
    if policy == "mlp_c_proj_middle_band":
        center = (layer_count - 1) / 2.0
        return (abs(layer_idx - center), layer_idx)
    if policy == "mlp_c_proj_edge_band":
        edge_distance = min(layer_idx, max(0, layer_count - 1 - layer_idx))
        return (edge_distance, layer_idx)

    if tiebreak == "late_layers_first":
        return (-layer_idx,)
    if tiebreak == "early_layers_first":
        return (layer_idx,)
    raise ValueError(f"Unknown selective_shape_prior_tiebreak: {tiebreak}")


def build_shape_prior_static_mask(matrix_params, param_to_name, args):
    policy = args.get("selective_shape_prior_policy", "mlp_c_proj_first")
    tiebreak = args.get("selective_shape_prior_tiebreak", "late_layers_first")
    release_fraction = float(args.get("selective_release_k_fraction", 0.0))
    if not 0.0 <= release_fraction < 1.0:
        raise ValueError("selective_release_k_fraction must be in [0, 1) for shape_prior")

    report_names = {param_to_name[p] for p in matrix_params}
    total_bytes = sum(_input_cov_full_state_bytes(p, include_eye=True) for p in matrix_params)
    target_bytes = int(round(total_bytes * release_fraction))
    layer_count = max((_layer_index(param_to_name[p]) for p in matrix_params), default=-1) + 1

    items = []
    for p in matrix_params:
        name = param_to_name[p]
        layer_idx = _layer_index(name)
        priority = _shape_prior_release_priority(name, policy)
        bytes_p = _input_cov_full_state_bytes(p, include_eye=True)
        tie_key = _shape_prior_tie_key(layer_idx, layer_count, policy, tiebreak)
        items.append(
            {
                "name": name,
                "bytes": bytes_p,
                "priority": priority,
                "layer_idx": layer_idx,
                "tie_key": tie_key,
            }
        )

    ranked = sorted(
        items,
        key=lambda item: (
            -item["priority"],
            item["tie_key"],
            -item["bytes"],
            item["name"],
        ),
    )

    released_names = set()
    released_bytes = 0
    for item in ranked:
        if target_bytes <= 0 or item["priority"] <= 0:
            break
        next_bytes = released_bytes + item["bytes"]
        if abs(next_bytes - target_bytes) <= abs(released_bytes - target_bytes):
            released_names.add(item["name"])
            released_bytes = next_bytes
        else:
            break

    newton_names = report_names - released_names
    rank_by_name = {item["name"]: idx for idx, item in enumerate(ranked, start=1)}
    label = (
        f"shape_prior|policy={policy}|tiebreak={tiebreak}|"
        f"target={release_fraction}|released_mib={released_bytes / (1024 * 1024):.2f}"
    )
    return {
        "path": "",
        "newton_names": newton_names,
        "report_names": report_names,
        "rank_by_name": rank_by_name,
        "label": label,
        "released_names": released_names,
        "target_release_bytes": target_bytes,
        "actual_release_bytes": released_bytes,
    }


def use_muon_family_param(name, param):
    return (
        param.ndim == 2
        and "weight" in name
        and "ln" not in name
        and "wte" not in name
        and "wpe" not in name
        and "lm_head" not in name
    )


def build_optimizer(model, args):
    optimizer_type = args["optimizer_type"]
    hook_handles = []

    if optimizer_type == "adamw":
        print("--- using AdamW optimizer ---")
        optimizer = model.configure_optimizers(
            args["weight_decay"],
            args["learning_rate"],
            (args["beta1"], args["beta2"]),
            args["device_type"],
        )
        return optimizer, hook_handles

    if optimizer_type not in MUON_FAMILY_OPTIMIZERS:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

    print(f"--- using {optimizer_type} + AdamW hybrid optimizer ---")
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    matrix_params = []
    param_to_name = {}
    adamw_decay_params = []
    adamw_nodecay_params = []
    for name, param in param_dict.items():
        if use_muon_family_param(name, param):
            matrix_params.append(param)
            param_to_name[param] = name
        elif param.ndim >= 2:
            adamw_decay_params.append(param)
        else:
            adamw_nodecay_params.append(param)

    static_mask = None
    if optimizer_type == "selective_newton_muon":
        if args.get("selective_selection_mode") == "shape_prior":
            static_mask = build_shape_prior_static_mask(matrix_params, param_to_name, args)
        else:
            static_mask = load_selective_static_mask(args)
        if static_mask is not None:
            matrix_names = set(param_to_name.values())
            missing_from_report = sorted(matrix_names - static_mask["report_names"])
            extra_in_report = sorted(static_mask["report_names"] - matrix_names)
            if missing_from_report:
                preview = ", ".join(missing_from_report[:5])
                print(
                    "warning: static selective mask does not cover "
                    f"{len(missing_from_report)} matrix params; defaulting them to Newton. "
                    f"Examples: {preview}"
                )
            if extra_in_report:
                preview = ", ".join(extra_in_report[:5])
                print(
                    "warning: static selective mask has "
                    f"{len(extra_in_report)} names not found in this model. Examples: {preview}"
                )
            print(
                "loaded static selective mask: "
                f"{len(static_mask['newton_names'])}/{len(static_mask['report_names'])} "
                f"Newton layers from {static_mask['label']}"
            )
            if "released_names" in static_mask:
                released_preview = ", ".join(sorted(static_mask["released_names"])[:8])
                print(
                    "shape-prior Muon-only layers: "
                    f"{len(static_mask['released_names'])}; {released_preview}"
                )

    adamw_groups = [
        {"params": adamw_decay_params, "weight_decay": args["weight_decay"]},
        {"params": adamw_nodecay_params, "weight_decay": 0.0},
    ]
    optimizer_adamw = torch.optim.AdamW(
        adamw_groups,
        lr=args["learning_rate"],
        betas=(args["beta1"], args["beta2"]),
    )

    param_to_module = {}
    if optimizer_type in INPUT_TRACKED_OPTIMIZERS:
        hook_params = matrix_params
        if static_mask is not None:
            hook_params = [
                p
                for p in matrix_params
                if param_to_name[p] in static_mask["newton_names"]
                or param_to_name[p] not in static_mask["report_names"]
            ]
        elif (
            optimizer_type == "cproj_k_mode_newton_muon"
            and args.get("cproj_k_mode", "block4") == "none"
        ):
            hook_params = [
                p
                for p in matrix_params
                if cproj_param_needs_input_hook(
                    param_to_name[p],
                    cproj_k_mode=args.get("cproj_k_mode", "block4"),
                    cproj_k_layers=args.get("cproj_k_layers", ()),
                    cproj_shadow_k_modes=args.get("cproj_shadow_k_modes", ()),
                    cproj_shadow_k_layers=args.get("cproj_shadow_k_layers", ()),
                )
            ]
        param_to_module, hook_handles = register_input_cache_hooks(model, hook_params)
        print(f"registered Linear input-cache hooks: {len(hook_handles)}")

    common_kwargs = dict(
        lr=args["muon_learning_rate"],
        momentum=args["muon_momentum"],
        weight_decay=args["matrix_weight_decay"],
        ns_steps=args["muon_ns_steps"],
        eps=args["matrix_eps"],
    )

    if optimizer_type == "muon":
        optimizer_muon = PureMuon(
            matrix_params,
            param_to_name=param_to_name,
            cheap_muon_probe_enabled=args.get("cheap_muon_probe_enabled", False),
            nesterov=args.get("muon_nesterov", False),
            momentum_ema=args.get("muon_momentum_ema", False),
            split_qkv=args.get("muon_split_qkv", False),
            adjust_lr_for_shape=args.get("muon_adjust_lr_for_shape", False),
            ns_compute_dtype=args.get("muon_ns_compute_dtype", "float32"),
            **common_kwargs,
        )
    elif optimizer_type == "newton_muon":
        optimizer_muon = NewtonMuon(
            matrix_params,
            param_to_module=param_to_module,
            input_beta=args["input_beta"],
            input_ridge=args["input_ridge"],
            input_refresh=args["input_refresh"],
            input_max_samples=args["input_max_samples"],
            input_first_refresh_step=args.get("input_first_refresh_step", 0),
            input_init_scale=args.get("input_init_scale", 1.0),
            input_init_inverse_scale=args.get("input_init_inverse_scale"),
            diagnostic_interval=args["diagnostic_interval"],
            **common_kwargs,
        )
    elif optimizer_type == "selective_newton_muon":
        if args.get("update_similarity_probe_enabled", False):
            print(
                "update similarity probe enabled: "
                f"interval={args.get('update_similarity_probe_interval', 25)}, "
                f"start={args.get('update_similarity_probe_start_step', 0)}, "
                f"stop={args.get('update_similarity_probe_stop_step', -1)}"
            )
        optimizer_muon = SelectiveNewtonMuon(
            matrix_params,
            param_to_module=param_to_module,
            param_to_name=param_to_name,
            input_beta=args["input_beta"],
            input_ridge=args["input_ridge"],
            input_refresh=args["input_refresh"],
            input_max_samples=args["input_max_samples"],
            diagnostic_interval=args["diagnostic_interval"],
            selective_fraction=args["selective_fraction"],
            selective_min_active=args["selective_min_active"],
            selective_warmup_steps=args["selective_warmup_steps"],
            selective_score_interval=args["selective_score_interval"],
            selective_score_beta=args["selective_score_beta"],
            selective_score_threshold=args["selective_score_threshold"],
            selective_score_mode=args["selective_score_mode"],
            selective_cond_power=args["selective_cond_power"],
            selective_cost_power=args["selective_cost_power"],
            selective_freeze_after_warmup=args["selective_freeze_after_warmup"],
            selective_log_diagnostics=args["selective_log_diagnostics"],
            selective_release_inactive_k_state=args["selective_release_inactive_k_state"],
            selective_selection_mode=args["selective_selection_mode"],
            selective_release_k_fraction=args["selective_release_k_fraction"],
            selective_static_newton_names=(static_mask["newton_names"] if static_mask else None),
            selective_static_report_names=(static_mask["report_names"] if static_mask else None),
            selective_static_rank_by_name=(static_mask["rank_by_name"] if static_mask else None),
            selective_static_mask_label=(static_mask["label"] if static_mask else ""),
            update_similarity_probe_enabled=args.get("update_similarity_probe_enabled", False),
            update_similarity_probe_interval=args.get("update_similarity_probe_interval", 25),
            update_similarity_probe_start_step=args.get("update_similarity_probe_start_step", 0),
            update_similarity_probe_stop_step=args.get("update_similarity_probe_stop_step", -1),
            **common_kwargs,
        )
    elif optimizer_type == "cproj_k_mode_newton_muon":
        optimizer_muon = CProjKModeNewtonMuon(
            matrix_params,
            param_to_module=param_to_module,
            param_to_name=param_to_name,
            input_beta=args["input_beta"],
            input_ridge=args["input_ridge"],
            input_refresh=args["input_refresh"],
            input_max_samples=args["input_max_samples"],
            input_first_refresh_step=args.get("input_first_refresh_step", 0),
            input_init_scale=args.get("input_init_scale", 1.0),
            input_init_inverse_scale=args.get("input_init_inverse_scale"),
            diagnostic_interval=args["diagnostic_interval"],
            cproj_k_mode=args.get("cproj_k_mode", "block4"),
            cproj_k_layers=args.get("cproj_k_layers", ()),
            cproj_k_blocks=args.get("cproj_k_blocks", 4),
            cproj_k_reference_mode=args.get("cproj_k_reference_mode", "full"),
            cproj_k_offdiag_alpha=args.get("cproj_k_offdiag_alpha", 0.5),
            cproj_shadow_k_modes=args.get("cproj_shadow_k_modes", ()),
            cproj_shadow_k_layers=args.get("cproj_shadow_k_layers", ()),
            nesterov=args.get("muon_nesterov", False),
            momentum_ema=args.get("muon_momentum_ema", False),
            split_qkv=args.get("muon_split_qkv", False),
            adjust_lr_for_shape=args.get("muon_adjust_lr_for_shape", False),
            ns_compute_dtype=args.get("muon_ns_compute_dtype", "float32"),
            **common_kwargs,
        )
    elif optimizer_type == "diag_sigma_newton_muon":
        optimizer_muon = DiagSigmaNewtonMuon(
            matrix_params,
            param_to_module=param_to_module,
            input_beta=args["input_beta"],
            input_ridge=args["input_ridge"],
            input_refresh=args["input_refresh"],
            input_max_samples=args["input_max_samples"],
            sigma_beta=args["sigma_beta"],
            sigma_lambda_max=args["sigma_lambda_max"],
            sigma_lambda_start=args["sigma_lambda_start"],
            sigma_lambda_warmup=args["sigma_lambda_warmup"],
            sigma_eps=args["sigma_eps"],
            stat_source=args["sigma_stat_source"],
            stat_mixed_rho=args["sigma_stat_mixed_rho"],
            match_base_fro_norm=args["match_base_fro_norm"],
            diagnostic_interval=args["diagnostic_interval"],
            **common_kwargs,
        )
    else:
        optimizer_muon = BlockSigmaNewtonMuon(
            matrix_params,
            param_to_module=param_to_module,
            input_beta=args["input_beta"],
            input_ridge=args["input_ridge"],
            input_refresh=args["input_refresh"],
            input_max_samples=args["input_max_samples"],
            sigma_block_size=args["sigma_block_size"],
            sigma_beta=args["sigma_beta"],
            sigma_lambda_max=args["sigma_lambda_max"],
            sigma_lambda_start=args["sigma_lambda_start"],
            sigma_lambda_warmup=args["sigma_lambda_warmup"],
            sigma_eps=args["sigma_eps"],
            sigma_refresh=args["sigma_refresh"],
            stat_source=args["sigma_stat_source"],
            stat_mixed_rho=args["sigma_stat_mixed_rho"],
            match_base_fro_norm=args["match_base_fro_norm"],
            diagnostic_interval=args["diagnostic_interval"],
            **common_kwargs,
        )

    print(
        f"matrix params: {len(matrix_params)}, "
        f"AdamW decay params: {len(adamw_decay_params)}, "
        f"AdamW no-decay params: {len(adamw_nodecay_params)}"
    )
    return HybridOptimizer(optimizer_adamw, optimizer_muon), hook_handles


def register_input_cache_hooks(model, matrix_params):
    hook_handles = []
    param_ids = {id(p) for p in matrix_params}
    param_to_module = {}

    def cache_input(module, inputs, output):
        module._last_input = inputs[0].detach()

    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and id(module.weight) in param_ids:
            param_to_module[module.weight] = module
            hook_handles.append(module.register_forward_hook(cache_input))

    missing = len(matrix_params) - len(param_to_module)
    if missing:
        print(f"warning: {missing} matrix params did not map to nn.Linear modules")
    return param_to_module, hook_handles
