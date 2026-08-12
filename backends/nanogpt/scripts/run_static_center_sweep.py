import argparse
import csv
import glob
import os
import re
import subprocess
import sys


BASE_CONFIGS = {
    "tier1": "config/train_openwebtext_gpt2_tier1.py",
    "tier2": "config/train_openwebtext_gpt2_tier2.py",
    "tier3": "config/train_openwebtext_gpt2_tier3.py",
}

REPLAY_CONFIG = "config/storage/38_static_center_mask_replay.py"


def ensure_data(args):
    data_dir = os.path.join("data", "openwebtext_gpt2")
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return True
    print("Missing OpenWebText GPT-2 data.")
    print("Run: python data/openwebtext_gpt2/prepare.py")
    return args.dry_run


def release_label(release_frac):
    return f"release{int(round(release_frac * 100)):02d}"


def infer_rule_name(mask_path):
    name = os.path.basename(mask_path)
    name = re.sub(r"\.csv$", "", name)
    name = re.sub(r"_release\d+$", "", name)
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return name or "static_center"


def read_mask_target(mask_path):
    with open(mask_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ("target_release_k_fraction", "actual_release_k_fraction"):
                value = row.get(key, "")
                if value != "":
                    return float(value)
            break
    raise ValueError(f"could not read target release fraction from {mask_path}")


def expand_mask_paths(args):
    mask_paths = list(args.mask_paths or [])
    if args.mask_dir:
        pattern = os.path.join(args.mask_dir, "static_center_h*_release*.csv")
        mask_paths.extend(sorted(glob.glob(pattern)))
    if not mask_paths:
        raise ValueError("provide --mask-paths or --mask-dir")
    return sorted(mask_paths, key=lambda path: (read_mask_target(path), infer_rule_name(path)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tier", default="tier3", choices=sorted(BASE_CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024])
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--mask-paths", nargs="+", default=None)
    parser.add_argument("--static-mask-seed", type=int, default=2024)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="Selective-Newton-Muon-OWT-Tier3StaticSweep-B16")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-profile", type=str, default="paper", choices=["full", "paper"])
    parser.add_argument("--wandb-log-tables", action="store_true")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--run-prefix", type=str, default="owt_tier3_static_sweep_b16")
    args = parser.parse_args()

    if not ensure_data(args):
        raise SystemExit(1)

    mask_paths = expand_mask_paths(args)
    for mask_path in mask_paths:
        if not os.path.exists(mask_path) and not args.dry_run:
            raise FileNotFoundError(f"mask path not found: {mask_path}")

    wandb_enabled = args.wandb_mode != "disabled"
    base_cfg = BASE_CONFIGS[args.tier]
    for seed in args.seeds:
        group = args.wandb_group or f"{args.run_prefix}_{args.tier}_seed{seed}"
        for mask_path in mask_paths:
            target_release = read_mask_target(mask_path)
            rule = infer_rule_name(mask_path)
            label = f"38_{rule}_{release_label(target_release)}"
            run_name = f"{args.run_prefix}_{args.tier}_{label}_seed{seed}"
            out_dir = f"out_{run_name}"
            tags = (
                f"formal,openwebtext_gpt2,{args.tier},selective_newton_muon,"
                f"static_center_sweep,{rule},mask_replay,storage_pareto"
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
                f"--wandb_log_profile={args.wandb_log_profile}",
                f"--wandb_log_tables={args.wandb_log_tables}",
                f"--wandb_group={group}",
                f"--wandb_run_name={run_name}",
                f"--wandb_tags={tags}",
                f"--selective_static_mask_path={mask_path}",
                f"--selective_static_mask_seed={args.static_mask_seed}",
                f"--selective_static_mask_target_release_k_fraction={target_release}",
                f"--selective_release_k_fraction={target_release}",
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
