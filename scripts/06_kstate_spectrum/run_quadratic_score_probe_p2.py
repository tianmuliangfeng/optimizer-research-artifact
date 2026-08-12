from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from project_paths import EXPERIMENT_RESULTS_ROOT
from runner_utils import (
    append_common_train_overrides,
    append_model_overrides,
    base_train_cmd,
    bool_literal,
    ensure_data,
    run_cmd,
    write_command_record,
)


BASE_CONFIG = "config/train_openwebtext_gpt2_50m_tier4.py"
MECHANISM_CONFIG = "config/mechanism/42_cproj_k_structure.py"
FAMILY = "06_kstate_spectrum"
VALID_MODES = ("none", "diag", "full")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the seed2026 P2 optimizer-state-aware c_proj mechanism probe. "
            "The main update remains none, while diag/full EMA K and momentum "
            "are accumulated as non-intervening shadow states."
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-exe", default=None)
    parser.add_argument("--dataset", default="openwebtext_gpt2_50m")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--base-config", default=BASE_CONFIG)
    parser.add_argument("--mechanism-config", default=MECHANISM_CONFIG)
    parser.add_argument("--n-layer", type=int, default=24)
    parser.add_argument("--n-head", type=int, default=16)
    parser.add_argument("--n-embd", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=10001)
    parser.add_argument("--lr-decay-iters", type=int, default=3000)
    parser.add_argument("--muon-learning-rate", type=float, default=0.01)
    parser.add_argument("--input-beta", type=float, default=0.95)
    parser.add_argument("--input-ridge", type=float, default=0.2)
    parser.add_argument("--input-refresh", type=int, default=32)
    parser.add_argument("--input-max-samples", type=int, default=2048)
    parser.add_argument(
        "--shadow-k-modes",
        nargs="+",
        choices=("diag", "full"),
        default=["diag", "full"],
    )
    parser.add_argument("--probe-steps", type=int, nargs="+", default=[10000])
    parser.add_argument("--probe-layers", type=int, nargs="+", default=[0, 11, 23])
    parser.add_argument(
        "--probe-modes",
        nargs="+",
        choices=VALID_MODES,
        default=list(VALID_MODES),
    )
    parser.add_argument("--probe-batch-size", type=int, default=1)
    parser.add_argument("--probe-block-size", type=int, default=128)
    parser.add_argument("--build-repeats", type=int, default=4)
    parser.add_argument("--heldout-batches", type=int, default=8)
    parser.add_argument(
        "--line-search-multipliers",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0, 2.0],
    )
    parser.add_argument("--no-exact-hvp", action="store_false", dest="exact_hvp")
    parser.add_argument("--no-exact-svd", action="store_false", dest="exact_svd")
    parser.add_argument("--exact-svd-repeats", type=int, default=4)
    parser.add_argument(
        "--svd-compute-dtype",
        default="float32",
        choices=["float32", "float64"],
        help=(
            "Linear-algebra dtype used only inside exact SVD. The returned "
            "direction is cast back to the probe dtype."
        ),
    )
    parser.add_argument(
        "--probe-precision",
        default="float32",
        choices=["training", "float32"],
    )
    parser.add_argument("--no-line-search", action="store_false", dest="line_search")
    parser.set_defaults(exact_hvp=True, exact_svd=True, line_search=True)
    parser.add_argument("--diagnostic-interval", type=int, default=0)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--always-save-checkpoint", action="store_true", default=False)
    parser.add_argument(
        "--wandb-project",
        default="Selective-Newton-Muon-MainConf-QuadraticProbe",
    )
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--wandb-log-profile",
        default="paper",
        choices=["full", "paper"],
    )
    parser.add_argument("--wandb-log-tables", action="store_true")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--run-prefix", default="mainconf_quadprobe_p2")
    parser.add_argument(
        "--no-write-commands",
        action="store_false",
        dest="write_commands",
    )
    parser.set_defaults(write_commands=True)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def csv_literal(values) -> str:
    return ",".join(str(value) for value in values)


def string_override(value: str) -> str:
    return repr(str(value))


def validate_args(args: argparse.Namespace) -> None:
    if not args.probe_steps:
        raise ValueError("--probe-steps cannot be empty")
    if min(args.probe_steps) < 0 or max(args.probe_steps) >= args.max_iters:
        raise ValueError(
            "--probe-steps must satisfy 0 <= step < max_iters; "
            f"got {args.probe_steps}, max_iters={args.max_iters}"
        )
    if not args.probe_layers:
        raise ValueError("--probe-layers cannot be empty")
    invalid_layers = [
        layer for layer in args.probe_layers if not 0 <= layer < args.n_layer
    ]
    if invalid_layers:
        raise ValueError(f"probe layers outside [0, {args.n_layer}): {invalid_layers}")
    if "none" not in args.probe_modes:
        raise ValueError("--probe-modes must include none")
    if len(args.probe_modes) != len(set(args.probe_modes)):
        raise ValueError(f"duplicate --probe-modes: {args.probe_modes}")
    if len(args.shadow_k_modes) != len(set(args.shadow_k_modes)):
        raise ValueError(f"duplicate --shadow-k-modes: {args.shadow_k_modes}")
    missing_shadows = set(args.probe_modes) - {"none"} - set(args.shadow_k_modes)
    if missing_shadows:
        raise ValueError(
            "every non-none probe mode requires a shadow state; missing "
            f"{sorted(missing_shadows)}"
        )
    if args.probe_batch_size <= 0:
        raise ValueError("--probe-batch-size must be positive")
    if not 0 < args.probe_block_size <= args.block_size:
        raise ValueError("--probe-block-size must be in (0, block-size]")
    if args.build_repeats <= 0:
        raise ValueError("--build-repeats must be positive")
    if args.heldout_batches <= 0:
        raise ValueError("--heldout-batches must be positive")
    if args.exact_svd:
        if not 0 < args.exact_svd_repeats <= args.build_repeats:
            raise ValueError(
                "--exact-svd-repeats must be in [1, build-repeats]"
            )
    elif args.exact_svd_repeats != 0:
        raise ValueError(
            "use --exact-svd-repeats 0 together with --no-exact-svd"
        )
    if any(multiplier < 0 for multiplier in args.line_search_multipliers):
        raise ValueError("--line-search-multipliers must be non-negative")
    if args.line_search and 0.0 not in args.line_search_multipliers:
        raise ValueError("line-search multipliers must include 0")
    if args.input_ridge <= 0:
        raise ValueError("diag/full K probes require positive --input-ridge")


def option_value(cmd: list[str], name: str) -> str:
    prefix = f"--{name}="
    values = [part[len(prefix) :] for part in cmd if part.startswith(prefix)]
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {prefix} option, found {values}")
    return values[0]


def build_command(args: argparse.Namespace) -> list[str]:
    protocol = str(getattr(args, "probe_protocol", "p2"))
    group = args.wandb_group or f"{args.run_prefix}_seed{args.seed}"
    run_name = (
        f"{args.run_prefix}_none_shadowdiagfull_L{args.n_layer}_D{args.n_embd}_"
        f"step{max(args.probe_steps)}_seed{args.seed}"
    )
    tags = (
        f"publication,{args.dataset},mechanism,quadratic_score_{protocol},"
        "optimizer_state,shadow_ema_k,shadow_momentum,exact_svd_intervention,"
        f"float32_probe,cproj_none_trajectory,L{args.n_layer},D{args.n_embd}"
    )
    cmd = base_train_cmd(
        args,
        config=args.base_config,
        run_name=run_name,
        group=group,
        method=f"cproj_quadratic_probe_{protocol}",
        tags=tags,
    )
    cmd.insert(3, args.mechanism_config)
    append_model_overrides(cmd, args)
    append_common_train_overrides(cmd, args)
    output_dir = (
        EXPERIMENT_RESULTS_ROOT
        / FAMILY
        / "quadratic_probe_p2"
        / run_name
    )
    cmd.extend(
        [
            "--cproj_k_mode=none",
            "--cproj_k_blocks=4",
            "--save_checkpoint=False",
            (
                "--cproj_shadow_k_modes="
                f"{string_override(csv_literal(args.shadow_k_modes))}"
            ),
            (
                "--cproj_shadow_k_layers="
                f"{string_override(csv_literal(args.probe_layers))}"
            ),
            f"--input_beta={args.input_beta}",
            f"--input_ridge={args.input_ridge}",
            f"--input_refresh={args.input_refresh}",
            f"--input_max_samples={args.input_max_samples}",
            f"--diagnostic_interval={args.diagnostic_interval}",
            "--update_similarity_probe_enabled=False",
            "--cproj_quadratic_probe_enabled=True",
            f"--cproj_quadratic_probe_variant={string_override('temporal')}",
            (
                "--cproj_quadratic_probe_steps="
                f"{string_override(csv_literal(args.probe_steps))}"
            ),
            (
                "--cproj_quadratic_probe_layers="
                f"{string_override(csv_literal(args.probe_layers))}"
            ),
            (
                "--cproj_quadratic_probe_modes="
                f"{string_override(csv_literal(args.probe_modes))}"
            ),
            f"--cproj_quadratic_probe_batch_size={args.probe_batch_size}",
            f"--cproj_quadratic_probe_block_size={args.probe_block_size}",
            f"--cproj_quadratic_probe_build_repeats={args.build_repeats}",
            f"--cproj_quadratic_probe_heldout_batches={args.heldout_batches}",
            "--cproj_quadratic_probe_include_none_repeat=False",
            f"--cproj_quadratic_probe_normmatch_modes={string_override('')}",
            (
                "--cproj_quadratic_probe_line_search_multipliers="
                f"{string_override(csv_literal(args.line_search_multipliers))}"
            ),
            f"--cproj_quadratic_probe_exact_hvp={bool_literal(args.exact_hvp)}",
            f"--cproj_quadratic_probe_exact_svd={bool_literal(args.exact_svd)}",
            (
                "--cproj_quadratic_probe_exact_svd_repeats="
                f"{args.exact_svd_repeats}"
            ),
            (
                "--cproj_quadratic_probe_svd_compute_dtype="
                f"{string_override(args.svd_compute_dtype)}"
            ),
            (
                "--cproj_quadratic_probe_precision="
                f"{string_override(args.probe_precision)}"
            ),
            f"--cproj_quadratic_probe_line_search={bool_literal(args.line_search)}",
            "--cproj_quadratic_probe_heldout_line_search=True",
            (
                "--cproj_quadratic_probe_output_dir="
                f"{string_override(str(output_dir))}"
            ),
        ]
    )
    return cmd


def validate_command(args: argparse.Namespace, cmd: list[str]) -> None:
    required = {
        "cproj_k_mode": "none",
        "save_checkpoint": "False",
        "cproj_shadow_k_modes": string_override(
            csv_literal(args.shadow_k_modes)
        ),
        "cproj_shadow_k_layers": string_override(
            csv_literal(args.probe_layers)
        ),
        "cproj_quadratic_probe_enabled": "True",
        "cproj_quadratic_probe_variant": string_override("temporal"),
        "cproj_quadratic_probe_steps": string_override(
            csv_literal(args.probe_steps)
        ),
        "cproj_quadratic_probe_layers": string_override(
            csv_literal(args.probe_layers)
        ),
        "cproj_quadratic_probe_modes": string_override(
            csv_literal(args.probe_modes)
        ),
        "cproj_quadratic_probe_build_repeats": str(args.build_repeats),
        "cproj_quadratic_probe_heldout_batches": str(args.heldout_batches),
        "cproj_quadratic_probe_include_none_repeat": "False",
        "cproj_quadratic_probe_normmatch_modes": string_override(""),
        "cproj_quadratic_probe_exact_hvp": bool_literal(args.exact_hvp),
        "cproj_quadratic_probe_exact_svd": bool_literal(args.exact_svd),
        "cproj_quadratic_probe_exact_svd_repeats": str(args.exact_svd_repeats),
        "cproj_quadratic_probe_svd_compute_dtype": string_override(
            args.svd_compute_dtype
        ),
        "cproj_quadratic_probe_precision": string_override(
            args.probe_precision
        ),
        "cproj_quadratic_probe_line_search": bool_literal(args.line_search),
        "cproj_quadratic_probe_heldout_line_search": "True",
        "update_similarity_probe_enabled": "False",
        "wandb_log_profile": args.wandb_log_profile,
        "wandb_log_tables": "True" if args.wandb_log_tables else "False",
    }
    for name, expected in required.items():
        actual = option_value(cmd, name)
        if actual != expected:
            raise RuntimeError(f"--{name}={actual}, expected {expected}")
    if args.mechanism_config not in cmd:
        raise RuntimeError("quadratic P2 command is missing mechanism config")


def print_expected_outputs(args: argparse.Namespace) -> None:
    ns5_candidates = 3 * len(args.probe_modes)
    svd_candidates = len(args.probe_modes)
    directions_per_step = len(args.probe_layers) * (
        args.build_repeats * ns5_candidates
        + args.exact_svd_repeats * svd_candidates
    )
    direction_rows = len(args.probe_steps) * directions_per_step
    splits = 1 + args.heldout_batches
    line_rows = (
        direction_rows * len(args.line_search_multipliers) * splits
        if args.line_search
        else 0
    )
    print(
        "Expected temporal quadratic-probe P2 output: "
        f"{direction_rows} direction rows and {line_rows} line-search rows."
    )
    print(
        "NS5 conditions: fresh-gradient, EMA-K gradient, and EMA-K shadow "
        "momentum for none/diag/full. Exact SVD is applied to fresh-gradient "
        f"directions for {args.exact_svd_repeats}/{args.build_repeats} builds."
    )
    print(
        "The model trajectory remains c_proj none. Shadow states are diagnostic "
        "only and are allocated only for the probed layers."
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not ensure_data(args.dataset, args.dry_run):
        raise SystemExit(1)
    cmd = build_command(args)
    validate_command(args, cmd)
    print_expected_outputs(args)
    write_command_record(
        family=FAMILY,
        run_prefix=args.run_prefix,
        commands=[cmd],
        dry_run=args.dry_run,
        enabled=args.write_commands,
    )
    run_cmd(cmd, args.dry_run)


if __name__ == "__main__":
    main()
