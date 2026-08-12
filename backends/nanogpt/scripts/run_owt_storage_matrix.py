import argparse
import os
import subprocess
import sys


BASE_CONFIGS = {
    "tier1": "config/train_openwebtext_gpt2_tier1.py",
    "tier2": "config/train_openwebtext_gpt2_tier2.py",
}

RUN_CONFIGS = [
    ("00_muon", "config/baselines/00_muon.py"),
    ("13_newton_muon_fast", "config/baselines/13_newton_muon_fast.py"),
    ("23_release15_w100", "config/storage/23_selective_byte_release15_warmup100.py"),
    ("22_release40_w100", "config/storage/22_selective_byte_release40_warmup100.py"),
]


def maybe_prepare_data(args):
    data_dir = os.path.join("data", "openwebtext_gpt2")
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path) and not args.prepare_data:
        return True
    if not args.prepare_data:
        print("Missing OpenWebText GPT-2 data.")
        print("Run: python data/openwebtext_gpt2/prepare.py")
        return args.dry_run

    cmd = [
        sys.executable,
        "data/openwebtext_gpt2/prepare.py",
        f"--max-train-tokens={args.max_train_tokens}",
        f"--max-val-tokens={args.max_val_tokens}",
        f"--val-fraction={args.val_fraction}",
    ]
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print commands without running")
    parser.add_argument("--prepare-data", action="store_true", help="prepare OpenWebText GPT-2 data before running")
    parser.add_argument("--max-train-tokens", type=int, default=10_000_000)
    parser.add_argument("--max-val-tokens", type=int, default=200_000)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--tiers", nargs="+", default=["tier1"], choices=sorted(BASE_CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337])
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="Selective-Newton-Muon-OWT")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--run-prefix", type=str, default="owt_storage")
    args = parser.parse_args()

    if not maybe_prepare_data(args):
        raise SystemExit(1)

    wandb_enabled = args.wandb_mode != "disabled"
    run_configs = ([] if args.skip_baselines else RUN_CONFIGS[:2]) + RUN_CONFIGS[2:]

    for tier in args.tiers:
        base_cfg = BASE_CONFIGS[tier]
        for seed in args.seeds:
            group = args.wandb_group or f"{args.run_prefix}_{tier}_seed{seed}"
            for label, cfg in run_configs:
                run_name = f"{args.run_prefix}_{tier}_{label}_seed{seed}"
                out_dir = f"out_{args.run_prefix}_{tier}_{label}_seed{seed}"
                tags = f"formal,openwebtext_gpt2,{tier},selective_newton_muon,storage_pareto"
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
