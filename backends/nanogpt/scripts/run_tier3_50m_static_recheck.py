import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


BASE_CONFIG = "config/train_openwebtext_gpt2_50m_tier3.py"
REPLAY_CONFIG = "config/storage/38_static_center_mask_replay.py"
BASELINE_CONFIGS = {
    "muon": ("00_muon", "config/baselines/00_muon.py"),
    "newton": ("13_newton_muon_fast", "config/baselines/13_newton_muon_fast.py"),
}
_ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
_RESULTS_ROOT = Path(
    os.environ.get("SNM_RESULTS_ROOT", str(_ARTIFACT_ROOT / "runs"))
).expanduser()
_MASK_ROOT = _RESULTS_ROOT / "analysis_exports/owt_tier3_static_center_sweep_20260707/masks"
DEFAULT_CANDIDATE_MASKS = (
    str(_MASK_ROOT / "static_center_h4_h8_release35.csv"),
    str(_MASK_ROOT / "static_center_h2_h9_release56.csv"),
)
ANCHOR_MASK = str(_MASK_ROOT / "static_center_h3_h8_release42.csv")


def ensure_data(args):
    data_dir = os.path.join("data", args.dataset)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return True
    print(f"Missing dataset files under {data_dir}.")
    print(
        "Run: python data/openwebtext_gpt2/prepare.py "
        f"--output-dir data/{args.dataset} --max-train-tokens 50000000 "
        "--max-val-tokens 1000000 --val-fraction 0.01 --force"
    )
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


def resolve_path(path):
    return path if os.path.isabs(path) else os.path.abspath(path)


def selected_masks(args):
    masks = list(args.mask_paths or DEFAULT_CANDIDATE_MASKS)
    if args.include_anchor_h3_h8:
        masks.append(ANCHOR_MASK)
    return [resolve_path(path) for path in masks]


def run_cmd(cmd, dry_run):
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset", default="openwebtext_gpt2_50m")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024])
    parser.add_argument("--mask-paths", nargs="+", default=None)
    parser.add_argument("--include-anchor-h3-h8", action="store_true")
    parser.add_argument("--static-mask-seed", type=int, default=2024)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="Selective-Newton-Muon-OWT-50M-Tier3")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-profile", type=str, default="paper", choices=["full", "paper"])
    parser.add_argument("--wandb-log-tables", action="store_true")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--run-prefix", type=str, default="owt_50m_tier3_recheck")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-candidates", action="store_true")
    args = parser.parse_args()

    if not ensure_data(args):
        raise SystemExit(1)

    wandb_enabled = args.wandb_mode != "disabled"
    masks = selected_masks(args)
    for mask_path in masks:
        if not os.path.exists(mask_path) and not args.dry_run:
            raise FileNotFoundError(f"mask path not found: {mask_path}")

    for seed in args.seeds:
        group = args.wandb_group or f"{args.run_prefix}_seed{seed}"
        if not args.skip_baselines:
            for method, (label, cfg) in BASELINE_CONFIGS.items():
                run_name = f"{args.run_prefix}_{label}_seed{seed}"
                cmd = [
                    sys.executable,
                    "train.py",
                    BASE_CONFIG,
                    cfg,
                    f"--dataset={args.dataset}",
                    f"--seed={seed}",
                    f"--out_dir=out_{run_name}",
                    f"--wandb_log={wandb_enabled}",
                    f"--wandb_project={args.wandb_project}",
                    f"--wandb_mode={args.wandb_mode}",
                    f"--wandb_log_profile={args.wandb_log_profile}",
                    f"--wandb_log_tables={args.wandb_log_tables}",
                    f"--wandb_group={group}",
                    f"--wandb_run_name={run_name}",
                    f"--wandb_tags=formal,{args.dataset},tier3,{method},large_data_recheck,storage_pareto",
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
                run_cmd(cmd, args.dry_run)

        if not args.skip_candidates:
            for mask_path in masks:
                target_release = read_mask_target(mask_path)
                rule = infer_rule_name(mask_path)
                label = f"38_{rule}_{release_label(target_release)}"
                run_name = f"{args.run_prefix}_{label}_seed{seed}"
                cmd = [
                    sys.executable,
                    "train.py",
                    BASE_CONFIG,
                    REPLAY_CONFIG,
                    f"--dataset={args.dataset}",
                    f"--seed={seed}",
                    f"--out_dir=out_{run_name}",
                    f"--wandb_log={wandb_enabled}",
                    f"--wandb_project={args.wandb_project}",
                    f"--wandb_mode={args.wandb_mode}",
                    f"--wandb_log_profile={args.wandb_log_profile}",
                    f"--wandb_log_tables={args.wandb_log_tables}",
                    f"--wandb_group={group}",
                    f"--wandb_run_name={run_name}",
                    f"--wandb_tags=formal,{args.dataset},tier3,selective_newton_muon,{rule},large_data_recheck,storage_pareto",
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
                run_cmd(cmd, args.dry_run)


if __name__ == "__main__":
    main()
