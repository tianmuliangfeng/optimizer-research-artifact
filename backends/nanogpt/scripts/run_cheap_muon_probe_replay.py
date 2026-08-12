import argparse
import os
import re
import subprocess
import sys


BASE_CONFIGS = {
    "tier1": "config/train_openwebtext_gpt2_tier1.py",
    "tier2": "config/train_openwebtext_gpt2_tier2.py",
    "tier3": "config/train_openwebtext_gpt2_tier3.py",
}

REPLAY_CONFIG = "config/probe/40_cheap_muon_probe_mask_replay_release40.py"


def ensure_data(args):
    data_dir = os.path.join("data", "openwebtext_gpt2")
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return True
    print("Missing OpenWebText GPT-2 data.")
    print("Run: python data/openwebtext_gpt2/prepare.py")
    return args.dry_run


def infer_rule_name(mask_path):
    name = os.path.basename(mask_path)
    name = re.sub(r"\.csv$", "", name)
    name = re.sub(r"^cheap_muon_probe_mask_", "", name)
    name = re.sub(r"_release\d+$", "", name)
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return name or "mask"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tier", default="tier3", choices=sorted(BASE_CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024])
    parser.add_argument("--mask-paths", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="Selective-Newton-Muon-OWT-Tier3Smoke-B16")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--run-prefix", type=str, default="owt_tier3_smoke_b16")
    args = parser.parse_args()

    if not ensure_data(args):
        raise SystemExit(1)

    for mask_path in args.mask_paths:
        if not os.path.exists(mask_path) and not args.dry_run:
            raise FileNotFoundError(f"mask path not found: {mask_path}")

    wandb_enabled = args.wandb_mode != "disabled"
    base_cfg = BASE_CONFIGS[args.tier]
    for seed in args.seeds:
        group = args.wandb_group or f"{args.run_prefix}_{args.tier}_seed{seed}"
        for mask_path in args.mask_paths:
            rule = infer_rule_name(mask_path)
            label = f"40_cheap_muon_{rule}_replay_release40"
            run_name = f"{args.run_prefix}_{args.tier}_{label}_seed{seed}"
            out_dir = f"out_{run_name}"
            tags = (
                f"formal,openwebtext_gpt2,{args.tier},selective_newton_muon,"
                f"cheap_muon_probe,{rule},mask_replay,storage_pareto"
            )
            cmd = [
                sys.executable,
                "train.py",
                base_cfg,
                REPLAY_CONFIG,
                f"--seed={seed}",
                f"--out_dir={out_dir}",
                f"--wandb_log={wandb_enabled}",
                f"--wandb_project={args.wandb_project}",
                f"--wandb_mode={args.wandb_mode}",
                f"--wandb_group={group}",
                f"--wandb_run_name={run_name}",
                f"--wandb_tags={tags}",
                f"--selective_static_mask_path={mask_path}",
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
