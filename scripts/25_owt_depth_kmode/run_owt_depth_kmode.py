"""Run the paired OpenWebText depth x c_proj-K-mode experiment.

The experiment applies ``none`` or ``diag`` only to the selected
``mlp.c_proj`` depths.  Every unselected c_proj and every other eligible
matrix remains on the full Newton-Muon path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from ast import literal_eval
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from project_paths import SOURCE_REPO
from runner_utils import (
    append_common_train_overrides,
    append_model_overrides,
    base_train_cmd,
    baseline_train_cmd,
    ensure_data,
    run_cmd,
    write_command_record,
)


BASE_CONFIG = "config/train_openwebtext_gpt2_50m_tier3.py"
MECHANISM_CONFIG = "config/mechanism/42_cproj_k_structure.py"
FAMILY = "25_owt_depth_kmode"
MIB = 1024 * 1024
RULE_LAYERS = {
    "early": (0, 1, 2, 3, 4, 5, 6, 7),
    "center": (2, 3, 4, 5, 6, 7, 8, 9),
    "late": (4, 5, 6, 7, 8, 9, 10, 11),
    "edge": (0, 1, 2, 3, 8, 9, 10, 11),
    "all": tuple(range(12)),
}
VALID_MODES = ("none", "diag")
VALID_ANCHORS = ("full", "muon")


def build_parser(
    *,
    dataset_default: str = "openwebtext_gpt2_50m",
    wandb_project_default: str = (
        "Selective-Newton-Muon-MainConf-OWT-Depth-KMode-20260724"
    ),
    run_prefix_default: str = "mainconf_owt_12L_depth_kmode",
    description: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description
        or (
            "Paired depth x K-mode experiment: selected mlp.c_proj depths use "
            "none or diag while all other matrices retain full Newton-Muon."
        )
    )
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--dry-run", action="store_true")
    phase.add_argument("--numerical-smoke", action="store_true")
    phase.add_argument("--formal", action="store_true")
    parser.add_argument("--python-exe", default=None)
    parser.add_argument("--dataset", default=dataset_default)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument(
        "--rules",
        nargs="+",
        choices=tuple(RULE_LAYERS),
        default=list(RULE_LAYERS),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=VALID_MODES,
        default=list(VALID_MODES),
    )
    parser.add_argument(
        "--anchors",
        nargs="*",
        choices=VALID_ANCHORS,
        default=list(VALID_ANCHORS),
    )
    parser.add_argument("--base-config", default=BASE_CONFIG)
    parser.add_argument("--mechanism-config", default=MECHANISM_CONFIG)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--lr-decay-iters", type=int, default=5000)
    parser.add_argument("--muon-learning-rate", type=float, default=0.02)
    parser.add_argument("--input-beta", type=float, default=0.95)
    parser.add_argument("--input-ridge", type=float, default=0.2)
    parser.add_argument("--input-refresh", type=int, default=32)
    parser.add_argument("--input-max-samples", type=int, default=2048)
    parser.add_argument("--smoke-steps", type=int, default=34)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--wandb-project",
        default=wandb_project_default,
    )
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=("online", "offline", "disabled"),
    )
    parser.add_argument("--wandb-log-profile", default="paper", choices=("paper", "full"))
    parser.add_argument("--wandb-log-tables", action="store_true")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--run-prefix", default=run_prefix_default)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-write-commands", action="store_false", dest="write_commands")
    parser.set_defaults(write_commands=True)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def unique(items):
    return list(dict.fromkeys(items))


def validate_args(args: argparse.Namespace) -> None:
    if args.n_layer != 12:
        raise ValueError(
            "The preregistered depth rules are defined for n_layer=12; "
            f"got n_layer={args.n_layer}"
        )
    if args.smoke_steps < 34:
        raise ValueError("--smoke-steps must be at least 34 to cross the step-32 K refresh")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("training seeds must be non-negative")
    if args.max_iters <= 0 or args.lr_decay_iters <= 0:
        raise ValueError("formal iteration counts must be positive")


def validate_source_support() -> None:
    required = {
        SOURCE_REPO / "train.py": "cproj_k_layers",
        SOURCE_REPO / "optimizer_factory.py": "cproj_param_needs_input_hook",
        SOURCE_REPO / "optimizers.py": "cproj_k_layers=()",
    }
    missing = []
    for path, marker in required.items():
        if not path.exists() or marker not in path.read_text(encoding="utf-8"):
            missing.append(f"{path}: missing {marker!r}")
    if missing:
        raise RuntimeError(
            "The source repo does not contain the depth-selective K-mode patch:\n- "
            + "\n- ".join(missing)
        )


def quoted_csv(values: tuple[int, ...]) -> str:
    return repr(",".join(str(value) for value in values))


def append_kmode_overrides(
    cmd: list[str],
    args: argparse.Namespace,
    *,
    mode: str,
    layers: tuple[int, ...],
) -> None:
    cmd.extend(
        [
            f"--cproj_k_mode={mode}",
            f"--cproj_k_layers={quoted_csv(layers)}",
            "--cproj_k_reference_mode=full",
            f"--input_beta={args.input_beta}",
            f"--input_ridge={args.input_ridge}",
            f"--input_refresh={args.input_refresh}",
            f"--input_max_samples={args.input_max_samples}",
            "--diagnostic_interval=0",
            "--update_similarity_probe_enabled=False",
            "--save_checkpoint=False",
        ]
    )


def configure_phase(args: argparse.Namespace) -> tuple[list[int], list[str], list[str], list[str]]:
    if args.numerical_smoke:
        args.max_iters = args.smoke_steps
        args.lr_decay_iters = args.smoke_steps
        args.eval_interval = args.smoke_steps
        args.eval_iters = min(args.eval_iters, 2)
        args.log_interval = min(args.log_interval, 10)
        args.wandb_mode = "disabled"
        return [2026], ["center", "all"], list(VALID_MODES), list(VALID_ANCHORS)
    return (
        unique(args.seeds),
        unique(args.rules),
        unique(args.modes),
        unique(args.anchors),
    )


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    seeds, rules, modes, anchors = configure_phase(args)
    commands = []
    phase = "smoke" if args.numerical_smoke else "formal"
    for seed in seeds:
        args.seed = seed
        group = args.wandb_group or f"{args.run_prefix}_{phase}_seed{seed}"
        for rule in rules:
            layers = RULE_LAYERS[rule]
            for mode in modes:
                variant = f"{rule}_{mode}"
                run_name = f"{args.run_prefix}_{phase}_{variant}_seed{seed}"
                tags = (
                    f"publication,{args.dataset},depth_kmode,{phase},"
                    f"rule_{rule},cproj_{mode},non_target_full,non_cproj_full"
                )
                cmd = base_train_cmd(
                    args,
                    config=args.base_config,
                    run_name=run_name,
                    group=group,
                    method=f"depth_{rule}_{mode}",
                    tags=tags,
                )
                cmd.insert(3, args.mechanism_config)
                append_model_overrides(cmd, args)
                append_common_train_overrides(cmd, args)
                append_kmode_overrides(cmd, args, mode=mode, layers=layers)
                commands.append(cmd)

        if "full" in anchors:
            run_name = f"{args.run_prefix}_{phase}_anchor_full_seed{seed}"
            cmd = base_train_cmd(
                args,
                config=args.base_config,
                run_name=run_name,
                group=group,
                method="anchor_full",
                tags=f"publication,{args.dataset},depth_kmode,{phase},anchor_full",
            )
            cmd.insert(3, args.mechanism_config)
            append_model_overrides(cmd, args)
            append_common_train_overrides(cmd, args)
            append_kmode_overrides(
                cmd,
                args,
                mode="full",
                layers=tuple(range(args.n_layer)),
            )
            commands.append(cmd)

        if "muon" in anchors:
            run_name = f"{args.run_prefix}_{phase}_anchor_muon_seed{seed}"
            cmd = baseline_train_cmd(
                args,
                config=args.base_config,
                run_name=run_name,
                group=group,
                method="muon",
                tags=f"publication,{args.dataset},depth_kmode,{phase},anchor_muon",
            )
            append_model_overrides(cmd, args)
            append_common_train_overrides(cmd, args)
            cmd.append("--save_checkpoint=False")
            commands.append(cmd)
    return commands


def option_value(cmd: list[str], name: str) -> str:
    prefix = f"--{name}="
    values = [part[len(prefix) :] for part in cmd if part.startswith(prefix)]
    if len(values) != 1:
        raise RuntimeError(f"Expected exactly one {prefix} option, found {values}")
    return values[0]


def validate_commands(args: argparse.Namespace, commands: list[list[str]]) -> None:
    seeds, rules, modes, anchors = configure_phase(args)
    expected = len(seeds) * (len(rules) * len(modes) + len(anchors))
    if len(commands) != expected:
        raise RuntimeError(f"Expected {expected} commands, generated {len(commands)}")
    names = [option_value(cmd, "wandb_run_name") for cmd in commands]
    if len(names) != len(set(names)):
        raise RuntimeError("Generated duplicate W&B run names")

    for cmd in commands:
        run_name = option_value(cmd, "wandb_run_name")
        if "anchor_muon" in run_name:
            continue
        mode = option_value(cmd, "cproj_k_mode")
        layers = literal_eval(option_value(cmd, "cproj_k_layers"))
        if not isinstance(layers, str):
            raise RuntimeError(f"{run_name}: cproj_k_layers must remain a string override")
        parsed_layers = tuple(int(item) for item in layers.split(",") if item)
        if not parsed_layers:
            raise RuntimeError(f"{run_name}: target layer list is empty")
        if any(not 0 <= layer < args.n_layer for layer in parsed_layers):
            raise RuntimeError(f"{run_name}: invalid target layers {parsed_layers}")
        if mode not in ("none", "diag", "full"):
            raise RuntimeError(f"{run_name}: unexpected c_proj mode {mode}")


def expected_k_state_mib(args: argparse.Namespace, mode: str, layer_count: int) -> float:
    d = args.n_embd
    element_size = 4
    tensors_per_full = 3
    full_per_cproj = (4 * d) ** 2 * element_size * tensors_per_full
    diag_per_cproj = 2 * (4 * d) * element_size
    non_cproj = args.n_layer * 3 * d * d * element_size * tensors_per_full
    unselected_full = (args.n_layer - layer_count) * full_per_cproj
    if mode == "none":
        selected = 0
    elif mode == "diag":
        selected = layer_count * diag_per_cproj
    elif mode == "full":
        selected = layer_count * full_per_cproj
    else:
        raise ValueError(mode)
    return (non_cproj + unselected_full + selected) / MIB


def print_plan(args: argparse.Namespace) -> None:
    seeds, rules, modes, anchors = configure_phase(args)
    print(
        f"Depth x K-mode plan: seeds={seeds}, rules={rules}, modes={modes}, "
        f"anchors={anchors}"
    )
    full_mib = expected_k_state_mib(args, "full", args.n_layer)
    for rule in rules:
        layers = RULE_LAYERS[rule]
        for mode in modes:
            mib = expected_k_state_mib(args, mode, len(layers))
            print(
                f"  {rule:>6}/{mode:<4}: layers={layers}, K={mib:.6f} MiB, "
                f"released_vs_full={100 * (1 - mib / full_mib):.4f}%"
            )


def run_experiment(
    args: argparse.Namespace,
    *,
    family: str = FAMILY,
    dataset_validator=None,
) -> None:
    validate_args(args)
    validate_source_support()
    if not ensure_data(args.dataset, args.dry_run):
        raise SystemExit(1)
    if dataset_validator is not None:
        dataset_validator(args)
    print_plan(args)
    commands = build_commands(args)
    validate_commands(args, commands)
    write_command_record(
        family=family,
        run_prefix=args.run_prefix,
        commands=commands,
        dry_run=args.dry_run,
        enabled=args.write_commands,
    )

    failures = []
    for cmd in commands:
        try:
            run_cmd(cmd, args.dry_run)
        except subprocess.CalledProcessError as exc:
            name = option_value(cmd, "wandb_run_name")
            print(
                f"RUN_FAILED name={name} returncode={exc.returncode}",
                file=sys.stderr,
                flush=True,
            )
            failures.append((name, exc.returncode))
            if not args.continue_on_error:
                raise
    if failures:
        detail = ", ".join(f"{name}(exit={code})" for name, code in failures)
        raise RuntimeError(f"{len(failures)} training run(s) failed: {detail}")


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
