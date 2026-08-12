from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from project_paths import EXPERIMENT_RESULTS_ROOT
from runner_utils import (
    DEFAULT_MIDDLE_RELEASE,
    RELEASE_ALL_CPROJ,
    append_common_train_overrides,
    append_model_overrides,
    baseline_train_cmd,
    build_model_mask,
    ensure_data,
    iters_for_tokens,
    method_choices,
    read_mask_target,
    release_label,
    run_cmd,
    selective_train_cmd,
    write_command_record,
)


BASE_CONFIG = "config/train_openwebtext_gpt2_50m_tier3.py"
FAMILY = "04_dataset_generalization"


def default_mask_dir() -> str:
    return str(EXPERIMENT_RESULTS_ROOT / "_shared" / "masks" / FAMILY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 12L/100M-token WikiText-103 dataset-generalization comparison."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-exe", default=None)
    parser.add_argument("--dataset", default="wikitext103_gpt2_50m")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024])
    parser.add_argument("--methods", nargs="+", choices=method_choices(), default=method_choices())
    parser.add_argument("--base-config", default=BASE_CONFIG)
    parser.add_argument("--target-tokens", type=int, default=100_000_000)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--lr-decay-iters", type=int, default=None)
    parser.add_argument("--muon-learning-rate", type=float, default=0.02)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--always-save-checkpoint", action="store_true", default=False)
    parser.add_argument("--static-mask-seed", type=int, default=2024)
    parser.add_argument("--middle-release-frac", type=float, default=DEFAULT_MIDDLE_RELEASE)
    parser.add_argument("--release-all-cproj-frac", type=float, default=RELEASE_ALL_CPROJ)
    parser.add_argument("--mask-dir", default=default_mask_dir())
    parser.add_argument(
        "--wandb-project",
        default="Selective-Newton-Muon-MainConf-DatasetGeneralization-WikiText103",
    )
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-profile", default="paper", choices=["full", "paper"])
    parser.add_argument("--wandb-log-tables", action="store_true")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--run-prefix", default="mainconf_wikitext103_12L_100m")
    parser.add_argument("--no-write-commands", action="store_false", dest="write_commands")
    parser.set_defaults(write_commands=True)
    return parser.parse_args()


def finalize_iters(args: argparse.Namespace) -> None:
    if args.max_iters is None:
        args.max_iters = iters_for_tokens(
            args.target_tokens,
            args.batch_size,
            args.block_size,
            args.gradient_accumulation_steps,
        )


def build_masks(args: argparse.Namespace) -> tuple[str | None, str | None]:
    mask_root = Path(args.mask_dir) / f"L{args.n_layer}_D{args.n_embd}"
    group = args.wandb_group or f"{args.run_prefix}_masks"
    middle_mask = None
    release_all_mask = None
    if "selective" in args.methods:
        middle_mask = build_model_mask(
            python_exe=args.python_exe,
            output_dir=mask_root / "middle",
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            target_release_frac=args.middle_release_frac,
            dataset=args.dataset,
            mask_seed=args.static_mask_seed,
            run_prefix=args.run_prefix,
            wandb_project=args.wandb_project,
            wandb_group=group,
            dry_run=args.dry_run,
        )
    if "release_all_cproj" in args.methods:
        release_all_mask = build_model_mask(
            python_exe=args.python_exe,
            output_dir=mask_root / "release_all_cproj",
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            target_release_frac=args.release_all_cproj_frac,
            dataset=args.dataset,
            mask_seed=args.static_mask_seed,
            run_prefix=args.run_prefix,
            wandb_project=args.wandb_project,
            wandb_group=group,
            dry_run=args.dry_run,
        )
    return middle_mask, release_all_mask


def build_commands(
    args: argparse.Namespace,
    middle_mask: str | None,
    release_all_mask: str | None,
) -> list[list[str]]:
    commands: list[list[str]] = []
    tags = (
        f"publication,dataset_generalization,{args.dataset},"
        f"target_tokens_{args.target_tokens},L{args.n_layer},D{args.n_embd}"
    )
    for seed in args.seeds:
        args.seed = seed
        group = args.wandb_group or f"{args.run_prefix}_seed{seed}"
        for method in args.methods:
            if method in ("muon", "newton"):
                method_label = "00_muon" if method == "muon" else "13_newton_muon_fast"
                run_name = f"{args.run_prefix}_{method_label}_seed{seed}"
                cmd = baseline_train_cmd(
                    args,
                    config=args.base_config,
                    run_name=run_name,
                    group=group,
                    method=method,
                    tags=tags,
                )
            elif method == "selective":
                if middle_mask is None:
                    raise ValueError("selective method needs a middle-band mask")
                release_frac = args.middle_release_frac if args.dry_run else read_mask_target(middle_mask)
                label = f"middle_cproj_{release_label(release_frac)}"
                run_name = f"{args.run_prefix}_{label}_seed{seed}"
                cmd = selective_train_cmd(
                    args,
                    config=args.base_config,
                    run_name=run_name,
                    group=group,
                    tags=tags,
                    mask_path=middle_mask,
                    release_frac=release_frac,
                    method_label="selective_middle_cproj",
                )
            elif method == "release_all_cproj":
                if release_all_mask is None:
                    raise ValueError("release_all_cproj method needs a release-all mask")
                release_frac = (
                    args.release_all_cproj_frac if args.dry_run else read_mask_target(release_all_mask)
                )
                label = f"all_cproj_{release_label(release_frac)}"
                run_name = f"{args.run_prefix}_{label}_seed{seed}"
                cmd = selective_train_cmd(
                    args,
                    config=args.base_config,
                    run_name=run_name,
                    group=group,
                    tags=tags,
                    mask_path=release_all_mask,
                    release_frac=release_frac,
                    method_label="release_all_cproj",
                )
            else:
                raise ValueError(f"unknown method: {method}")
            append_model_overrides(cmd, args)
            append_common_train_overrides(cmd, args)
            commands.append(cmd)
    return commands


def main() -> None:
    args = parse_args()
    finalize_iters(args)
    if not ensure_data(args.dataset, args.dry_run):
        raise SystemExit(1)
    middle_mask, release_all_mask = build_masks(args)
    commands = build_commands(args, middle_mask, release_all_mask)
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
