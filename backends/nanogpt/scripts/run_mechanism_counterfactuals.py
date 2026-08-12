import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from pathlib import Path


BASE_CONFIGS = {
    "tier1": "config/train_openwebtext_gpt2_tier1.py",
    "tier2": "config/train_openwebtext_gpt2_tier2.py",
    "tier3": "config/train_openwebtext_gpt2_tier3.py",
}
REPLAY_CONFIG = "config/storage/38_static_center_mask_replay.py"


def default_mask_dir():
    artifact_root = Path(__file__).resolve().parents[3]
    results_root = Path(
        os.environ.get("SNM_RESULTS_ROOT", str(artifact_root / "runs"))
    ).expanduser()
    return (
        results_root
        / "analysis_exports"
        / "owt_tier3_mechanism_counterfactual_masks_20260708/masks"
    )


def ensure_data(dataset: str, dry_run: bool) -> bool:
    data_dir = os.path.join("data", dataset)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return True
    print(f"Missing dataset files under {data_dir}.")
    print(f"Run: python data/{dataset}/prepare.py")
    return dry_run


def release_label(release_frac):
    return f"release{int(round(release_frac * 100)):02d}"


def infer_rule_name(mask_path):
    name = os.path.basename(mask_path)
    name = re.sub(r"\.csv$", "", name)
    name = re.sub(r"_release\d+$", "", name)
    name = re.sub(r"^mechanism_", "", name)
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return name or "mechanism"


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
        mask_paths.extend(sorted(glob.glob(os.path.join(args.mask_dir, "mechanism_*_release*.csv"))))
    if not mask_paths:
        raise ValueError(
            "No mechanism counterfactual masks found. First run "
            "scripts/build_mechanism_counterfactual_masks.py, or pass --mask-dir pointing to "
            "the directory containing mechanism_*_release*.csv files. "
            f"Current --mask-dir={args.mask_dir!r}"
        )
    return sorted(mask_paths, key=lambda path: (read_mask_target(path), infer_rule_name(path)))


def run_cmd(cmd, dry_run):
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tier", default="tier3", choices=sorted(BASE_CONFIGS))
    parser.add_argument("--dataset", default="openwebtext_gpt2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--mask-dir", default=str(default_mask_dir()))
    parser.add_argument("--mask-paths", nargs="+", default=None)
    parser.add_argument("--static-mask-seed", type=int, default=2024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="Selective-Newton-Muon-OWT-Mechanism")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-profile", type=str, default="paper", choices=["full", "paper"])
    parser.add_argument("--wandb-log-tables", action="store_true")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--run-prefix", type=str, default="owt_tier3_mechanism_cf")
    args = parser.parse_args()

    if not ensure_data(args.dataset, args.dry_run):
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
            label = f"38_mechanism_{rule}_{release_label(target_release)}"
            run_name = f"{args.run_prefix}_{args.tier}_{label}_seed{seed}"
            out_dir = f"out_{run_name}"
            tags = (
                f"formal,{args.dataset},{args.tier},selective_newton_muon,"
                f"mechanism_counterfactual,{rule},mask_replay"
            )
            cmd = [
                sys.executable,
                "train.py",
                base_cfg,
                REPLAY_CONFIG,
                f"--dataset={args.dataset}",
                f"--seed={seed}",
                f"--batch_size={args.batch_size}",
                f"--max_iters={args.max_iters}",
                f"--lr_decay_iters={args.max_iters}",
                f"--eval_iters={args.eval_iters}",
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
                "--update_similarity_probe_enabled=False",
            ]
            if args.device is not None:
                cmd.append(f"--device={args.device}")
                if args.device == "cpu":
                    cmd.append("--dtype=float32")
            run_cmd(cmd, args.dry_run)


if __name__ == "__main__":
    main()
