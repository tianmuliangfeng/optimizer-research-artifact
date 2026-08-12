import argparse
import subprocess
import sys


BASELINE_CONFIGS = [
    ("00_muon", "config/baselines/00_muon.py"),
    ("13_newton_muon_fast", "config/baselines/13_newton_muon_fast.py"),
]

STORAGE_CONFIGS = [
    ("19_byte_release00_w100", "config/storage/19_selective_byte_release00_warmup100.py"),
    ("20_byte_release10_w100", "config/storage/20_selective_byte_release10_warmup100.py"),
    ("23_byte_release15_w100", "config/storage/23_selective_byte_release15_warmup100.py"),
    ("24_byte_release20_w100", "config/storage/24_selective_byte_release20_warmup100.py"),
    ("21_byte_release25_w100", "config/storage/21_selective_byte_release25_warmup100.py"),
    ("25_byte_release30_w100", "config/storage/25_selective_byte_release30_warmup100.py"),
    ("26_byte_release35_w100", "config/storage/26_selective_byte_release35_warmup100.py"),
    ("22_byte_release40_w100", "config/storage/22_selective_byte_release40_warmup100.py"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print commands without running")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337])
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="Selective-Newton-Muon-Storage-Pareto-Exact")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--run-prefix", type=str, default="tiny_shakespeare_storage_pareto_exact")
    args = parser.parse_args()

    wandb_enabled = args.wandb_mode != "disabled"
    configs = ([] if args.skip_baselines else BASELINE_CONFIGS) + STORAGE_CONFIGS

    for seed in args.seeds:
        group = args.wandb_group or f"{args.run_prefix}_seed{seed}"
        for label, cfg in configs:
            run_name = f"{args.run_prefix}_{label}_seed{seed}"
            cmd = [
                sys.executable,
                "train.py",
                "config/train_shakespeare_char.py",
                cfg,
                f"--seed={seed}",
                f"--wandb_log={wandb_enabled}",
                f"--wandb_project={args.wandb_project}",
                f"--wandb_mode={'online' if args.wandb_mode == 'disabled' else args.wandb_mode}",
                f"--wandb_group={group}",
                f"--wandb_run_name={run_name}",
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
