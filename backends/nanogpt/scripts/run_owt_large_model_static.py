import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


BASE_CONFIGS = {
    "tier4": {
        "config": "config/train_openwebtext_gpt2_50m_tier4.py",
        "n_layer": 18,
        "n_embd": 768,
    },
}
BASELINE_CONFIGS = {
    "muon": ("00_muon", "config/baselines/00_muon.py"),
    "newton": ("13_newton_muon_fast", "config/baselines/13_newton_muon_fast.py"),
}
REPLAY_CONFIG = "config/storage/38_static_center_mask_replay.py"
DEFAULT_TARGET_RELEASE = 0.5614035087719298


def default_mask_dir():
    artifact_root = Path(__file__).resolve().parents[3]
    results_root = Path(
        os.environ.get("SNM_RESULTS_ROOT", str(artifact_root / "runs"))
    ).expanduser()
    return (
        results_root
        / "analysis_exports"
        / "owt_50m_large_model_static_20260708/masks"
    )


def ensure_data(dataset, dry_run):
    data_dir = os.path.join("data", dataset)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return True
    print(f"Missing dataset files under {data_dir}.")
    print(
        f"Run: python data/openwebtext_gpt2/prepare.py --output-dir data/{dataset} "
        "--max-train-tokens 50000000 --max-val-tokens 1000000 --val-fraction 0.01 --force"
    )
    return dry_run


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


def release_label(release_frac):
    return f"release{int(round(release_frac * 100)):02d}"


def run_cmd(cmd, dry_run):
    print(" ".join(str(part) for part in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def build_mask(args, tier_spec, mask_dir):
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    summary_path = mask_dir / "large_model_center_cproj_mask_summary.csv"
    cmd = [
        sys.executable,
        "scripts/build_model_center_cproj_mask.py",
        f"--output-dir={mask_dir}",
        f"--n-layer={tier_spec['n_layer']}",
        f"--n-embd={tier_spec['n_embd']}",
        f"--target-release-frac={args.target_release_frac}",
        f"--dataset={args.dataset}",
        f"--mask-seed={args.static_mask_seed}",
        f"--run-prefix={args.run_prefix}_{args.tier}",
        f"--wandb-project={args.wandb_project}",
        f"--wandb-group={args.wandb_group or args.run_prefix}",
    ]
    run_cmd(cmd, args.dry_run)
    if args.dry_run and not summary_path.exists():
        return str(
            mask_dir
            / (
                f"large_center_cproj_L{tier_spec['n_layer']}_D{tier_spec['n_embd']}_"
                f"AUTO_{release_label(args.target_release_frac)}.csv"
            )
        )
    if not summary_path.exists():
        raise FileNotFoundError(f"mask summary not found in {mask_dir}")
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty mask summary: {summary_path}")
    return rows[0]["mask_path"]


def selected_methods(args):
    if args.methods:
        return args.methods
    return ["muon", "newton", "selective"]


def append_common_overrides(cmd, args):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tier", default="tier4", choices=sorted(BASE_CONFIGS))
    parser.add_argument("--dataset", default="openwebtext_gpt2_50m")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024])
    parser.add_argument("--methods", nargs="+", choices=["muon", "newton", "selective"], default=None)
    parser.add_argument("--mask-path", default="")
    parser.add_argument("--mask-dir", default=str(default_mask_dir()))
    parser.add_argument("--skip-build-mask", action="store_true")
    parser.add_argument("--static-mask-seed", type=int, default=2024)
    parser.add_argument("--target-release-frac", type=float, default=DEFAULT_TARGET_RELEASE)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="Selective-Newton-Muon-OWT-LargeModel")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-profile", type=str, default="paper", choices=["full", "paper"])
    parser.add_argument("--wandb-log-tables", action="store_true")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--run-prefix", type=str, default="owt_50m_large_model_static")
    args = parser.parse_args()

    if not ensure_data(args.dataset, args.dry_run):
        raise SystemExit(1)

    tier_spec = BASE_CONFIGS[args.tier]
    base_cfg = tier_spec["config"]
    methods = selected_methods(args)
    wandb_enabled = args.wandb_mode != "disabled"

    mask_path = args.mask_path
    if "selective" in methods:
        if not mask_path and not args.skip_build_mask:
            mask_path = build_mask(args, tier_spec, args.mask_dir)
        if not mask_path and not args.dry_run:
            raise ValueError("selective method needs --mask-path or mask build enabled")
        if mask_path and not os.path.exists(mask_path) and not args.dry_run:
            raise FileNotFoundError(f"mask path not found: {mask_path}")

    for seed in args.seeds:
        group = args.wandb_group or f"{args.run_prefix}_{args.tier}_seed{seed}"
        for method in methods:
            if method in BASELINE_CONFIGS:
                label, cfg = BASELINE_CONFIGS[method]
                run_name = f"{args.run_prefix}_{args.tier}_{label}_seed{seed}"
                cmd = [
                    sys.executable,
                    "train.py",
                    base_cfg,
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
                    f"--wandb_tags=formal,{args.dataset},{args.tier},{method},large_model_scale",
                ]
                append_common_overrides(cmd, args)
                run_cmd(cmd, args.dry_run)
                continue

            target_release = read_mask_target(mask_path) if mask_path and not args.dry_run else args.target_release_frac
            label = f"38_large_center_cproj_{release_label(target_release)}"
            run_name = f"{args.run_prefix}_{args.tier}_{label}_seed{seed}"
            cmd = [
                sys.executable,
                "train.py",
                base_cfg,
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
                f"--wandb_tags=formal,{args.dataset},{args.tier},selective_newton_muon,large_center_cproj,large_model_scale",
                f"--selective_static_mask_path={mask_path}",
                f"--selective_static_mask_seed={args.static_mask_seed}",
                f"--selective_static_mask_target_release_k_fraction={target_release}",
                f"--selective_release_k_fraction={target_release}",
                "--update_similarity_probe_enabled=False",
            ]
            append_common_overrides(cmd, args)
            run_cmd(cmd, args.dry_run)


if __name__ == "__main__":
    main()
