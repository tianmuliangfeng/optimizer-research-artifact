from __future__ import annotations

import argparse
import subprocess
import sys
from ast import literal_eval
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from runner_utils import (
    append_common_train_overrides,
    append_model_overrides,
    base_train_cmd,
    ensure_data,
    run_cmd,
    write_command_record,
)


BASE_CONFIG = "config/train_openwebtext_gpt2_50m_tier4.py"
MECHANISM_CONFIG = "config/mechanism/42_cproj_k_structure.py"
FAMILY = "06_kstate_spectrum"
VALID_MODES = (
    "none",
    "full",
    "block4",
    "diag",
    "scalar",
    "alpha",
    "block_alpha",
)
ALPHA_MODES = ("alpha", "block_alpha")
DEFAULT_OFFDIAG_ALPHAS = (0.25, 0.50, 0.75)
MIB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare mlp.c_proj activation-K structures while every other matrix "
            "uses the unchanged full Newton-Muon path."
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-exe", default=None)
    parser.add_argument("--dataset", default="openwebtext_gpt2_50m")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--modes", nargs="+", choices=VALID_MODES, default=["block4"])
    parser.add_argument(
        "--offdiag-alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_OFFDIAG_ALPHAS),
        help=(
            "Non-diagonal multipliers for --modes alpha. The diagonal of K is unchanged; "
            "the same values apply to --modes block_alpha. alpha=0 is allowed as "
            "a storage/code-path control, while alpha=1 is excluded because the "
            "existing full or block4 endpoint is reused."
        ),
    )
    parser.add_argument("--base-config", default=BASE_CONFIG)
    parser.add_argument("--mechanism-config", default=MECHANISM_CONFIG)
    parser.add_argument("--n-layer", type=int, default=24)
    parser.add_argument("--n-head", type=int, default=16)
    parser.add_argument("--n-embd", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=12000)
    parser.add_argument("--lr-decay-iters", type=int, default=3000)
    parser.add_argument("--muon-learning-rate", type=float, default=0.01)
    parser.add_argument("--input-beta", type=float, default=0.95)
    parser.add_argument("--input-ridge", type=float, default=0.2)
    parser.add_argument("--input-refresh", type=int, default=32)
    parser.add_argument("--input-max-samples", type=int, default=2048)
    parser.add_argument("--cproj-k-blocks", type=int, default=4)
    parser.add_argument("--diagnostic-interval", type=int, default=0)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--save-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save best-model checkpoints. Disabled by default because the mechanism "
            "batch is evaluated from W&B curves and otherwise creates one large "
            "checkpoint per run."
        ),
    )
    parser.add_argument("--always-save-checkpoint", action="store_true", default=False)
    parser.add_argument(
        "--wandb-project",
        default="Selective-Newton-Muon-MainConf-CProjK-Mechanism",
    )
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-profile", default="paper", choices=["full", "paper"])
    parser.add_argument("--wandb-log-tables", action="store_true")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--run-prefix", default="mainconf_mechanism_24L_cproj_k")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run the remaining variants after a child training command fails.",
    )
    parser.add_argument("--no-write-commands", action="store_false", dest="write_commands")
    parser.set_defaults(write_commands=True)
    return parser.parse_args()


def alpha_label(alpha: float) -> str:
    return format(alpha, ".6g").replace(".", "p")


def validate_args(args: argparse.Namespace) -> None:
    if any(mode in args.modes for mode in ALPHA_MODES):
        invalid = [alpha for alpha in args.offdiag_alphas if not 0.0 <= alpha < 1.0]
        if invalid:
            raise ValueError(
                "--offdiag-alphas must be in [0, 1); alpha=0 is the dense-storage "
                "or block-storage control and the discrete endpoint is reused for "
                f"alpha=1, got {invalid}"
            )


def append_mechanism_overrides(
    cmd: list[str],
    args: argparse.Namespace,
    mode: str,
    offdiag_alpha: float | None,
) -> None:
    cmd.extend(
        [
            f"--cproj_k_mode={mode}",
            f"--cproj_k_blocks={args.cproj_k_blocks}",
            f"--input_beta={args.input_beta}",
            f"--input_ridge={args.input_ridge}",
            f"--input_refresh={args.input_refresh}",
            f"--input_max_samples={args.input_max_samples}",
            f"--diagnostic_interval={args.diagnostic_interval}",
            "--update_similarity_probe_enabled=False",
            f"--save_checkpoint={'True' if args.save_checkpoint else 'False'}",
        ]
    )
    if mode in ALPHA_MODES:
        if offdiag_alpha is None:
            raise ValueError(f"{mode} mode requires an off-diagonal multiplier")
        # Keep a decimal point for zero so nanoGPT's configurator parses the
        # override as float rather than int (the declared default is 0.5).
        cmd.append(f"--cproj_k_offdiag_alpha={float(offdiag_alpha)!r}")


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    commands = []
    seen = set()
    modes = []
    for mode in args.modes:
        if mode not in seen:
            seen.add(mode)
            modes.append(mode)
    alphas = list(dict.fromkeys(args.offdiag_alphas))
    for seed in dict.fromkeys(args.seeds):
        args.seed = seed
        group = args.wandb_group or f"{args.run_prefix}_seed{seed}"
        for mode in modes:
            mode_alphas = alphas if mode in ALPHA_MODES else [None]
            for offdiag_alpha in mode_alphas:
                if offdiag_alpha is None:
                    variant = mode
                    method = f"cproj_k_{mode}"
                    alpha_tag = ""
                else:
                    label = alpha_label(offdiag_alpha)
                    topology = "dense" if mode == "alpha" else "block"
                    variant = f"{topology}_alpha{label}"
                    method = f"cproj_k_{topology}_alpha_{label}"
                    if offdiag_alpha == 0.0:
                        alpha_tag = (
                            f",offdiag_alpha_0,{topology}_storage_alpha_0,"
                            f"{topology}_alpha_path"
                        )
                    else:
                        alpha_tag = (
                            f",offdiag_alpha_{label},{topology}_alpha_path"
                        )
                run_name = (
                    f"{args.run_prefix}_{variant}_L{args.n_layer}_D{args.n_embd}_"
                    f"lr{args.muon_learning_rate:g}_seed{seed}"
                )
                tags = (
                    f"publication,{args.dataset},mechanism,k_structure,L{args.n_layer},"
                    f"D{args.n_embd},cproj_{mode}{alpha_tag},non_cproj_full_newton"
                )
                cmd = base_train_cmd(
                    args,
                    config=args.base_config,
                    run_name=run_name,
                    group=group,
                    method=method,
                    tags=tags,
                )
                cmd.insert(3, args.mechanism_config)
                append_model_overrides(cmd, args)
                append_common_train_overrides(cmd, args)
                append_mechanism_overrides(cmd, args, mode, offdiag_alpha)
                commands.append(cmd)
    return commands


def expected_k_state_bytes(args: argparse.Namespace, mode: str) -> tuple[int, int, int]:
    """Return non-cproj, cproj, and total bytes for the float32 K/K_inv/eye states."""
    d = args.n_embd
    layers = args.n_layer
    full_factor_bytes = d * d * 4 * 3
    non_cproj = layers * 3 * full_factor_bytes
    if mode == "none":
        cproj = 0
    elif mode == "full":
        cproj = layers * (4 * d) * (4 * d) * 4 * 3
    elif mode == "block4":
        block_width = (4 * d) // args.cproj_k_blocks
        if block_width * args.cproj_k_blocks != 4 * d:
            raise ValueError("4 * n_embd must be divisible by cproj_k_blocks")
        cproj = layers * args.cproj_k_blocks * block_width * block_width * 4 * 3
    elif mode == "diag":
        cproj = layers * 2 * (4 * d) * 4
    elif mode == "scalar":
        # One float32 running scalar and its inverse per c_proj matrix.
        cproj = layers * 2 * 4
    elif mode == "alpha":
        # Including alpha=0, this deliberately uses the dense (4d) x (4d)
        # implementation so storage/code-path effects are held fixed.
        cproj = layers * (4 * d) * (4 * d) * 4 * 3
    elif mode == "block_alpha":
        block_width = (4 * d) // args.cproj_k_blocks
        if block_width * args.cproj_k_blocks != 4 * d:
            raise ValueError("4 * n_embd must be divisible by cproj_k_blocks")
        cproj = layers * args.cproj_k_blocks * block_width * block_width * 4 * 3
    else:
        raise ValueError(f"unknown mode: {mode}")
    return non_cproj, cproj, non_cproj + cproj


def print_expected_memory(args: argparse.Namespace) -> None:
    full_total = expected_k_state_bytes(args, "full")[2]
    print("Expected float32 K-state allocation:")
    for mode in dict.fromkeys(args.modes):
        labels = (
            [f"{mode}={alpha:g}" for alpha in dict.fromkeys(args.offdiag_alphas)]
            if mode in ALPHA_MODES
            else [mode]
        )
        non_cproj, cproj, total = expected_k_state_bytes(args, mode)
        released = 1.0 - total / full_total
        for label in labels:
            print(
                f"  {label:>10}: total={total / MIB:.2f} MiB, "
                f"cproj={cproj / MIB:.6f} MiB, non_cproj={non_cproj / MIB:.2f} MiB, "
                f"released_vs_full={100.0 * released:.2f}%"
            )


def option_value(cmd: list[str], name: str) -> str:
    prefix = f"--{name}="
    values = [part[len(prefix) :] for part in cmd if part.startswith(prefix)]
    if len(values) != 1:
        raise RuntimeError(f"Expected exactly one {prefix} option, found {values}")
    return values[0]


def validate_commands(args: argparse.Namespace, commands: list[list[str]]) -> None:
    modes = list(dict.fromkeys(args.modes))
    alpha_count = len(list(dict.fromkeys(args.offdiag_alphas)))
    expected_per_seed = sum(alpha_count if mode in ALPHA_MODES else 1 for mode in modes)
    expected_count = len(list(dict.fromkeys(args.seeds))) * expected_per_seed
    if len(commands) != expected_count:
        raise RuntimeError(f"Expected {expected_count} commands, generated {len(commands)}")

    run_names = [option_value(cmd, "wandb_run_name") for cmd in commands]
    if len(run_names) != len(set(run_names)):
        raise RuntimeError(f"Duplicate W&B run names: {run_names}")

    common_required = {
        "input_beta": format(args.input_beta, "g"),
        "input_ridge": format(args.input_ridge, "g"),
        "input_refresh": str(args.input_refresh),
        "input_max_samples": str(args.input_max_samples),
        "diagnostic_interval": str(args.diagnostic_interval),
        "update_similarity_probe_enabled": "False",
        "wandb_log_profile": args.wandb_log_profile,
        "wandb_log_tables": "True" if args.wandb_log_tables else "False",
        "save_checkpoint": "True" if args.save_checkpoint else "False",
    }
    for cmd in commands:
        if args.mechanism_config not in cmd:
            raise RuntimeError("A command is missing the c_proj K-structure config")
        for name, expected in common_required.items():
            actual = option_value(cmd, name)
            if actual != expected:
                raise RuntimeError(
                    f"{option_value(cmd, 'wandb_run_name')}: --{name}={actual}, expected {expected}"
                )
        mode = option_value(cmd, "cproj_k_mode")
        if mode in ALPHA_MODES:
            alpha_literal = option_value(cmd, "cproj_k_offdiag_alpha")
            parsed_alpha = literal_eval(alpha_literal)
            if type(parsed_alpha) is not float:
                raise RuntimeError(
                    "cproj_k_offdiag_alpha must be emitted as a float literal for "
                    f"nanoGPT configurator compatibility, got {alpha_literal!r}"
                )
            alpha = float(parsed_alpha)
            if not 0.0 <= alpha < 1.0:
                raise RuntimeError(f"Generated invalid {mode} control: {alpha}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not ensure_data(args.dataset, args.dry_run):
        raise SystemExit(1)
    print_expected_memory(args)
    commands = build_commands(args)
    validate_commands(args, commands)
    write_command_record(
        family=FAMILY,
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
            run_name = option_value(cmd, "wandb_run_name")
            print(
                f"RUN_FAILED name={run_name} returncode={exc.returncode}",
                file=sys.stderr,
                flush=True,
            )
            failures.append((run_name, exc.returncode))
            if not args.continue_on_error:
                raise
    if failures:
        detail = ", ".join(f"{name}(exit={code})" for name, code in failures)
        raise RuntimeError(f"{len(failures)} training run(s) failed: {detail}")


if __name__ == "__main__":
    main()
