import argparse
import os
import subprocess
import sys


BASE_CONFIG = "config/train_openwebtext_gpt2_tier3.py"
PROBE_CONFIG = "config/probe/41_update_similarity_probe.py"


def ensure_data(dataset: str, dry_run: bool) -> bool:
    data_dir = os.path.join("data", dataset)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return True
    print(f"Missing dataset files under {data_dir}.")
    print(f"Run: python data/{dataset}/prepare.py")
    return dry_run


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset", default="openwebtext_gpt2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--probe-interval", type=int, default=25)
    parser.add_argument("--probe-start-step", type=int, default=0)
    parser.add_argument("--probe-stop-step", type=int, default=-1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="Selective-Newton-Muon-OWT-Mechanism")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-profile", type=str, default="paper", choices=["full", "paper"])
    parser.add_argument("--wandb-log-tables", action="store_true")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--run-prefix", type=str, default="owt_tier3_update_similarity_probe")
    args = parser.parse_args()

    if not ensure_data(args.dataset, args.dry_run):
        raise SystemExit(1)

    wandb_enabled = args.wandb_mode != "disabled"
    for seed in args.seeds:
        group = args.wandb_group or f"{args.run_prefix}_seed{seed}"
        run_name = f"{args.run_prefix}_seed{seed}"
        cmd = [
            sys.executable,
            "train.py",
            BASE_CONFIG,
            PROBE_CONFIG,
            f"--dataset={args.dataset}",
            f"--seed={seed}",
            f"--batch_size={args.batch_size}",
            f"--max_iters={args.max_iters}",
            f"--lr_decay_iters={args.max_iters}",
            f"--eval_iters={args.eval_iters}",
            f"--eval_interval={args.eval_interval}",
            f"--out_dir=out_{run_name}",
            f"--wandb_log={wandb_enabled}",
            f"--wandb_project={args.wandb_project}",
            f"--wandb_mode={args.wandb_mode}",
            f"--wandb_log_profile={args.wandb_log_profile}",
            f"--wandb_log_tables={args.wandb_log_tables}",
            f"--wandb_group={group}",
            f"--wandb_run_name={run_name}",
            f"--wandb_tags=formal,{args.dataset},tier3,mechanism,update_similarity_probe",
            "--optimizer_type=selective_newton_muon",
            "--selective_selection_mode=k_release_budget",
            "--selective_release_k_fraction=0.0",
            "--selective_warmup_steps=0",
            "--selective_release_inactive_k_state=False",
            "--update_similarity_probe_enabled=True",
            f"--update_similarity_probe_interval={args.probe_interval}",
            f"--update_similarity_probe_start_step={args.probe_start_step}",
            f"--update_similarity_probe_stop_step={args.probe_stop_step}",
        ]
        if args.device is not None:
            cmd.append(f"--device={args.device}")
            if args.device == "cpu":
                cmd.append("--dtype=float32")
        run_cmd(cmd, args.dry_run)


if __name__ == "__main__":
    main()
