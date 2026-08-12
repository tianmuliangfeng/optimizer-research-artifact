from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
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


MECHANISM_CONFIG = "config/mechanism/42_cproj_k_structure.py"
FAMILY = "06_kstate_spectrum"
DEFAULT_ANCHORS = ("owt_12l_100m", "wikitext_24l_12k", "wikitext_12l_100m")
MIB = 1024 * 1024


@dataclass(frozen=True)
class Anchor:
    key: str
    label: str
    dataset: str
    base_config: str
    n_layer: int
    n_head: int
    n_embd: int
    batch_size: int
    block_size: int
    gradient_accumulation_steps: int
    max_iters: int
    lr_decay_iters: int
    muon_learning_rate: float

    @property
    def consumed_tokens(self) -> int:
        return (
            self.max_iters
            * self.batch_size
            * self.block_size
            * self.gradient_accumulation_steps
        )


ANCHORS = {
    "owt_12l_100m": Anchor(
        key="owt_12l_100m",
        label="OpenWebText 12L/100M",
        dataset="openwebtext_gpt2_50m",
        base_config="config/train_openwebtext_gpt2_50m_tier3.py",
        n_layer=12,
        n_head=12,
        n_embd=768,
        batch_size=16,
        block_size=512,
        gradient_accumulation_steps=1,
        max_iters=12208,
        lr_decay_iters=12208,
        muon_learning_rate=0.02,
    ),
    "wikitext_24l_12k": Anchor(
        key="wikitext_24l_12k",
        label="WikiText-103 24L/12k",
        dataset="wikitext103_gpt2_50m",
        base_config="config/train_openwebtext_gpt2_50m_tier4.py",
        n_layer=24,
        n_head=16,
        n_embd=1024,
        batch_size=2,
        block_size=512,
        gradient_accumulation_steps=1,
        max_iters=12000,
        lr_decay_iters=3000,
        muon_learning_rate=0.01,
    ),
    "wikitext_12l_100m": Anchor(
        key="wikitext_12l_100m",
        label="WikiText-103 12L/100M probe",
        dataset="wikitext103_gpt2_50m",
        base_config="config/train_openwebtext_gpt2_50m_tier3.py",
        n_layer=12,
        n_head=12,
        n_embd=768,
        batch_size=16,
        block_size=512,
        gradient_accumulation_steps=1,
        max_iters=12208,
        lr_decay_iters=12208,
        muon_learning_rate=0.02,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the seven preregistered targeted diagonal-c_proj generalization jobs. "
            "All non-c_proj matrices keep the full Newton-Muon path."
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-exe", default=None)
    parser.add_argument(
        "--anchors",
        nargs="+",
        choices=tuple(ANCHORS),
        default=list(DEFAULT_ANCHORS),
        help="Run all three preregistered anchors by default; select a subset to resume failures.",
    )
    parser.add_argument("--owt-12l-seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument(
        "--wikitext-24l-seeds", type=int, nargs="+", default=[2024, 2025, 2026]
    )
    parser.add_argument("--wikitext-12l-seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--input-beta", type=float, default=0.95)
    parser.add_argument("--input-ridge", type=float, default=0.2)
    parser.add_argument("--input-refresh", type=int, default=32)
    parser.add_argument("--input-max-samples", type=int, default=2048)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--always-save-checkpoint", action="store_true", default=False)
    parser.add_argument(
        "--wandb-project",
        default="Selective-Newton-Muon-MainConf-CProjK-Diag-Generalization",
    )
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--run-prefix", default="mainconf_targeted_diag")
    parser.add_argument("--no-write-commands", action="store_false", dest="write_commands")
    parser.set_defaults(
        write_commands=True,
        wandb_log_profile="paper",
        wandb_log_tables=False,
        diagnostic_interval=0,
        cproj_k_mode="diag",
        cproj_k_blocks=4,
    )
    return parser.parse_args()


def seeds_for_anchor(args: argparse.Namespace, key: str) -> list[int]:
    if key == "owt_12l_100m":
        seeds = args.owt_12l_seeds
    elif key == "wikitext_24l_12k":
        seeds = args.wikitext_24l_seeds
    elif key == "wikitext_12l_100m":
        seeds = args.wikitext_12l_seeds
    else:
        raise ValueError(f"Unknown anchor: {key}")
    return list(dict.fromkeys(seeds))


def validate_args(args: argparse.Namespace) -> None:
    args.anchors = list(dict.fromkeys(args.anchors))
    if not args.anchors:
        raise ValueError("At least one anchor is required")
    for key in args.anchors:
        seeds = seeds_for_anchor(args, key)
        if not seeds:
            raise ValueError(f"Anchor {key} requires at least one seed")
        if any(seed < 0 for seed in seeds):
            raise ValueError(f"Anchor {key} received a negative seed: {seeds}")
    if not 0.0 <= args.input_beta < 1.0:
        raise ValueError("--input-beta must be in [0, 1)")
    if args.input_ridge < 0.0:
        raise ValueError("--input-ridge must be non-negative")
    if args.input_refresh <= 0 or args.input_max_samples <= 0:
        raise ValueError("input refresh and max samples must be positive")


def expected_diag_k_state_mib(anchor: Anchor) -> tuple[float, float, float, float]:
    d = anchor.n_embd
    layers = anchor.n_layer
    full_d_factor_bytes = d * d * 4 * 3
    non_cproj_bytes = layers * 3 * full_d_factor_bytes
    cproj_diag_bytes = layers * 2 * (4 * d) * 4
    cproj_full_bytes = layers * (4 * d) * (4 * d) * 4 * 3
    retained_bytes = non_cproj_bytes + cproj_diag_bytes
    full_bytes = non_cproj_bytes + cproj_full_bytes
    released_fraction = 1.0 - retained_bytes / full_bytes
    return (
        retained_bytes / MIB,
        cproj_diag_bytes / MIB,
        non_cproj_bytes / MIB,
        released_fraction,
    )


def run_namespace(args: argparse.Namespace, anchor: Anchor, seed: int) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "dataset": anchor.dataset,
            "seed": seed,
            "n_layer": anchor.n_layer,
            "n_head": anchor.n_head,
            "n_embd": anchor.n_embd,
            "batch_size": anchor.batch_size,
            "block_size": anchor.block_size,
            "gradient_accumulation_steps": anchor.gradient_accumulation_steps,
            "max_iters": anchor.max_iters,
            "lr_decay_iters": anchor.lr_decay_iters,
            "muon_learning_rate": anchor.muon_learning_rate,
        }
    )
    return argparse.Namespace(**values)


def append_diag_overrides(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--cproj_k_mode=diag",
            f"--cproj_k_blocks={args.cproj_k_blocks}",
            f"--input_beta={args.input_beta}",
            f"--input_ridge={args.input_ridge}",
            f"--input_refresh={args.input_refresh}",
            f"--input_max_samples={args.input_max_samples}",
            "--diagnostic_interval=0",
            "--update_similarity_probe_enabled=False",
        ]
    )


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    commands: list[list[str]] = []
    for key in args.anchors:
        anchor = ANCHORS[key]
        lr_label = format(anchor.muon_learning_rate, ".6g").replace(".", "p")
        tags = (
            f"publication,mechanism_generalization,targeted_diag,{anchor.dataset},"
            f"{anchor.key},L{anchor.n_layer},D{anchor.n_embd},steps_{anchor.max_iters},"
            f"tokens_{anchor.consumed_tokens},mulr{lr_label}"
        )
        group = f"{args.run_prefix}_{anchor.key}"
        for seed in seeds_for_anchor(args, key):
            current = run_namespace(args, anchor, seed)
            run_name = f"{args.run_prefix}_{anchor.key}_cproj_diag_seed{seed}"
            cmd = base_train_cmd(
                current,
                config=anchor.base_config,
                run_name=run_name,
                group=group,
                method="cproj_k_diag",
                tags=tags,
            )
            cmd.insert(3, MECHANISM_CONFIG)
            append_model_overrides(cmd, current)
            append_common_train_overrides(cmd, current)
            append_diag_overrides(cmd, current)
            commands.append(cmd)
    return commands


def option_value(cmd: list[str], name: str) -> str:
    prefix = f"--{name}="
    values = [part[len(prefix) :] for part in cmd if part.startswith(prefix)]
    if len(values) != 1:
        raise RuntimeError(f"Expected exactly one {prefix} option, found {values}")
    return values[0]


def validate_commands(args: argparse.Namespace, commands: list[list[str]]) -> None:
    expected_count = sum(len(seeds_for_anchor(args, key)) for key in args.anchors)
    if len(commands) != expected_count:
        raise RuntimeError(f"Expected {expected_count} commands, generated {len(commands)}")
    run_names = [option_value(cmd, "wandb_run_name") for cmd in commands]
    if len(run_names) != len(set(run_names)):
        raise RuntimeError(f"Duplicate W&B run names: {run_names}")

    for cmd in commands:
        if MECHANISM_CONFIG not in cmd:
            raise RuntimeError("A command is missing the c_proj K-structure mechanism config")
        required = {
            "cproj_k_mode": "diag",
            "input_beta": format(args.input_beta, "g"),
            "input_ridge": format(args.input_ridge, "g"),
            "input_refresh": str(args.input_refresh),
            "input_max_samples": str(args.input_max_samples),
            "diagnostic_interval": "0",
            "update_similarity_probe_enabled": "False",
            "wandb_log_profile": "paper",
            "wandb_log_tables": "False",
        }
        for name, expected in required.items():
            actual = option_value(cmd, name)
            if actual != expected:
                raise RuntimeError(
                    f"{option_value(cmd, 'wandb_run_name')}: --{name}={actual}, expected {expected}"
                )


def print_plan(args: argparse.Namespace, commands: list[list[str]]) -> None:
    print("Targeted diagonal generalization plan:")
    for key in args.anchors:
        anchor = ANCHORS[key]
        seeds = seeds_for_anchor(args, key)
        total_mib, cproj_mib, non_cproj_mib, released = expected_diag_k_state_mib(anchor)
        print(
            f"  {anchor.label}: seeds={seeds}, runs={len(seeds)}, "
            f"steps={anchor.max_iters}, tokens={anchor.consumed_tokens}, "
            f"matrix_lr={anchor.muon_learning_rate:g}, diag_K={total_mib:.2f} MiB "
            f"(c_proj={cproj_mib:.2f}, non_c_proj={non_cproj_mib:.2f}, "
            f"released={100.0 * released:.2f}%)"
        )
    print(f"  total runs: {len(commands)}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    for dataset in dict.fromkeys(ANCHORS[key].dataset for key in args.anchors):
        if not ensure_data(dataset, args.dry_run):
            raise SystemExit(1)
    commands = build_commands(args)
    validate_commands(args, commands)
    print_plan(args, commands)
    write_command_record(
        family=FAMILY,
        run_prefix=args.run_prefix,
        commands=commands,
        dry_run=args.dry_run,
        enabled=args.write_commands,
    )
    for cmd in commands:
        run_cmd(cmd, args.dry_run)


if __name__ == "__main__":
    main()
