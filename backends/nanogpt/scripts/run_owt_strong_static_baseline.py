import argparse
import os
import subprocess
import sys


BASE_CONFIGS = {
    "tier1": "config/train_openwebtext_gpt2_tier1.py",
    "tier2": "config/train_openwebtext_gpt2_tier2.py",
    "tier3": "config/train_openwebtext_gpt2_tier3.py",
}

RUN_CONFIGS = {
    "muon": ("00_muon", "config/baselines/00_muon.py"),
    "cheap_muon_probe": (
        "38_cheap_muon_probe",
        "config/probe/38_cheap_muon_probe.py",
    ),
    "cheap_muon_importance_replay": (
        "39_cheap_muon_importance_replay_release40",
        "config/probe/39_cheap_muon_importance_replay_release40.py",
    ),
    "newton": ("13_newton_muon_fast", "config/baselines/13_newton_muon_fast.py"),
    "dynamic_release40": (
        "22_selective_byte_release40_warmup100",
        "config/storage/22_selective_byte_release40_warmup100.py",
    ),
    "cheap_release40": (
        "37_cheap_release40",
        "config/storage/37_cheap_release40.py",
    ),
    "band_middle_release40": (
        "34_band_prior_middle_release40",
        "config/storage/34_band_prior_middle_release40.py",
    ),
}

DEFAULT_RUNS = ("newton", "band_middle_release40")


def ensure_data(args):
    data_dir = os.path.join("data", "openwebtext_gpt2")
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return True
    print("Missing OpenWebText GPT-2 data.")
    print("Run: python data/openwebtext_gpt2/prepare.py")
    return args.dry_run


def selected_runs(args):
    if args.only:
        return args.only
    runs = list(DEFAULT_RUNS)
    if args.include_muon:
        runs.insert(0, "muon")
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tiers", nargs="+", default=["tier2"], choices=sorted(BASE_CONFIGS))
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 3407, 9001],
        help="default extends the existing 1337/2024 tier2 band-prior evidence",
    )
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--include-muon",
        action="store_true",
        help="also run Muon so Newton-over-Muon gain preservation can be recomputed",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(RUN_CONFIGS),
        default=None,
        help="override the default run set",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="Selective-Newton-Muon-OWT-StrongStatic",
    )
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-profile", type=str, default="paper", choices=["full", "paper"])
    parser.add_argument("--wandb-log-tables", action="store_true")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--run-prefix", type=str, default="owt_strong_static")
    args = parser.parse_args()

    if not ensure_data(args):
        raise SystemExit(1)

    wandb_enabled = args.wandb_mode != "disabled"
    run_keys = selected_runs(args)

    for tier in args.tiers:
        base_cfg = BASE_CONFIGS[tier]
        for seed in args.seeds:
            group = args.wandb_group or f"{args.run_prefix}_{tier}_seed{seed}"
            for run_key in run_keys:
                label, cfg = RUN_CONFIGS[run_key]
                run_name = f"{args.run_prefix}_{tier}_{label}_seed{seed}"
                out_dir = f"out_{args.run_prefix}_{tier}_{label}_seed{seed}"
                tags = (
                    f"formal,openwebtext_gpt2,{tier},selective_newton_muon,"
                    f"{run_key},strong_static_baseline,storage_pareto"
                )
                cmd = [
                    sys.executable,
                    "train.py",
                    base_cfg,
                    cfg,
                    f"--seed={seed}",
                    f"--out_dir={out_dir}",
                    f"--wandb_log={wandb_enabled}",
                    f"--wandb_project={args.wandb_project}",
                    f"--wandb_mode={args.wandb_mode}",
                    f"--wandb_log_profile={args.wandb_log_profile}",
                    f"--wandb_log_tables={args.wandb_log_tables}",
                    f"--wandb_group={group}",
                    f"--wandb_run_name={run_name}",
                    f"--wandb_tags={tags}",
                ]
                if args.max_iters is not None:
                    cmd.append(f"--max_iters={args.max_iters}")
                    cmd.append(f"--lr_decay_iters={args.max_iters}")
                if args.batch_size is not None:
                    cmd.append(f"--batch_size={args.batch_size}")
                if args.eval_iters is not None:
                    cmd.append(f"--eval_iters={args.eval_iters}")
                if args.device is not None:
                    cmd.append(f"--device={args.device}")
                    if args.device == "cpu":
                        cmd.append("--dtype=float32")
                print(" ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
