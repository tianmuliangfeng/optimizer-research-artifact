import argparse
import os
import subprocess
import sys


BASE_CONFIGS = {
    "tier1": "config/train_openwebtext_gpt2_tier1.py",
    "tier2": "config/train_openwebtext_gpt2_tier2.py",
}

RUN_CONFIGS = [
    ("31_shape_prior_release40", "config/storage/31_shape_prior_release40.py"),
    ("32_shape_prior_release15", "config/storage/32_shape_prior_release15.py"),
]


def ensure_data(args):
    data_dir = os.path.join("data", "openwebtext_gpt2")
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return True
    print("Missing OpenWebText GPT-2 data.")
    print("Run: python data/openwebtext_gpt2/prepare.py")
    return args.dry_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tiers", nargs="+", default=["tier2"], choices=sorted(BASE_CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 2024])
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="Selective-Newton-Muon-OWT-ShapePrior")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--run-prefix", type=str, default="owt_shape_prior")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[label for label, _ in RUN_CONFIGS],
        default=None,
        help="optionally run only selected shape-prior configs",
    )
    args = parser.parse_args()

    if not ensure_data(args):
        raise SystemExit(1)

    wandb_enabled = args.wandb_mode != "disabled"
    run_configs = RUN_CONFIGS
    if args.only:
        wanted = set(args.only)
        run_configs = [(label, cfg) for label, cfg in RUN_CONFIGS if label in wanted]

    for tier in args.tiers:
        base_cfg = BASE_CONFIGS[tier]
        for seed in args.seeds:
            group = args.wandb_group or f"{args.run_prefix}_{tier}_seed{seed}"
            for label, cfg in run_configs:
                run_name = f"{args.run_prefix}_{tier}_{label}_seed{seed}"
                out_dir = f"out_{args.run_prefix}_{tier}_{label}_seed{seed}"
                tags = f"formal,openwebtext_gpt2,{tier},selective_newton_muon,shape_prior,storage_pareto"
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
                    f"--wandb_group={group}",
                    f"--wandb_run_name={run_name}",
                    f"--wandb_tags={tags}",
                ]
                if args.max_iters is not None:
                    cmd.append(f"--max_iters={args.max_iters}")
                    cmd.append(f"--lr_decay_iters={args.max_iters}")
                if args.device is not None:
                    cmd.append(f"--device={args.device}")
                    if args.device == "cpu":
                        cmd.append("--dtype=float32")
                print(" ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
