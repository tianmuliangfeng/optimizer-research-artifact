import argparse
import subprocess
import sys


BASELINE_CONFIGS = [
    ("00_muon", "config/baselines/00_muon.py"),
    ("13_newton_muon_fast", "config/baselines/13_newton_muon_fast.py"),
]

SELECTIVE_CONFIGS = [
    ("10_top75_w100_fast", "config/selective/10_selective_v2_top75_fast_warmup100.py"),
    ("07_top50_w100_fast", "config/ablations/07_selective_v2_top50_fast_warmup100.py"),
    ("08_top50_w50_fast", "config/ablations/08_selective_v2_top50_fast_warmup50.py"),
    ("09_top50_w25_fast", "config/ablations/09_selective_v2_top50_fast_warmup25.py"),
    ("11_top75_w50_fast", "config/ablations/11_selective_v2_top75_fast_warmup50.py"),
    ("12_top75_w25_fast", "config/ablations/12_selective_v2_top75_fast_warmup25.py"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print commands without running")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="Selective-Newton-Muon")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--run-prefix", type=str, default="tiny_shakespeare_selective_fast")
    args = parser.parse_args()

    group = args.wandb_group or f"{args.run_prefix}_warmup_sweep_seed{args.seed}"
    wandb_enabled = args.wandb_mode != "disabled"
    configs = ([] if args.skip_baselines else BASELINE_CONFIGS) + SELECTIVE_CONFIGS

    for label, cfg in configs:
        run_name = f"{args.run_prefix}_{label}_seed{args.seed}"
        cmd = [
            sys.executable,
            "train.py",
            "config/train_shakespeare_char.py",
            cfg,
            f"--seed={args.seed}",
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
