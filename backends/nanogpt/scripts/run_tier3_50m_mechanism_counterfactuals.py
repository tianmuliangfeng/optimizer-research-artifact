import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


BASE_CONFIG = "config/train_openwebtext_gpt2_50m_tier3.py"
REPLAY_CONFIG = "config/storage/38_static_center_mask_replay.py"

RULE_TO_MASK = {
    "center": "mechanism_cproj_center_h2_h9_release56.csv",
    "early": "mechanism_cproj_early_h0_h7_release56.csv",
    "late": "mechanism_cproj_late_h4_h11_release56.csv",
    "edge": "mechanism_cproj_edge_h0_h3_h8_h11_release56.csv",
    "random_s0": "mechanism_cproj_random8_s0_release56.csv",
    "cproj_h5_h6": "mechanism_cproj_middle_h5_h6_release14.csv",
    "non_cproj_all": "mechanism_non_cproj_all_release16.csv",
    "attn_c_attn_all": "mechanism_attn_c_attn_all_release05.csv",
    "attn_c_proj_all": "mechanism_attn_c_proj_all_release05.csv",
    "mlp_c_fc_all": "mechanism_mlp_c_fc_all_release05.csv",
    "attn_c_attn_h5_h6": "mechanism_attn_c_attn_h5_h6_release01.csv",
    "attn_c_proj_h5_h6": "mechanism_attn_c_proj_h5_h6_release01.csv",
    "mlp_c_fc_h5_h6": "mechanism_mlp_c_fc_h5_h6_release01.csv",
}
DEFAULT_RULES = ("early", "late", "edge", "random_s0")
RULE_HELP = (
    "Rule names to replay. Built-ins: center, early, late, edge, random_s0, "
    "cproj_h5_h6, non_cproj_all, attn_c_attn_all, attn_c_proj_all, mlp_c_fc_all, "
    "attn_c_attn_h5_h6, attn_c_proj_h5_h6, mlp_c_fc_h5_h6. "
    "Additional random masks can be passed as random_sN after generating "
    "mechanism_cproj_random8_sN_release56.csv."
)


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


def ensure_data(dataset, dry_run):
    data_dir = os.path.join("data", dataset)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return True
    print(f"Missing dataset files under {data_dir}.")
    print(
        "Run: python data/openwebtext_gpt2/prepare.py "
        f"--output-dir data/{dataset} --max-train-tokens 50000000 "
        "--max-val-tokens 1000000 --val-fraction 0.01 --force"
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


def selected_rules(args):
    rules = list(args.rules)
    if args.include_center and "center" not in rules:
        rules.insert(0, "center")
    return rules


def mask_filename_for_rule(rule):
    if rule in RULE_TO_MASK:
        return RULE_TO_MASK[rule]
    if rule.startswith("random_s"):
        seed_text = rule.rsplit("_s", 1)[-1]
        if seed_text.isdigit():
            return f"mechanism_cproj_random8_s{seed_text}_release56.csv"
    raise ValueError(
        f"unknown rule={rule!r}. Use one of {sorted(RULE_TO_MASK)} "
        "or random_sN for a generated random mask."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset", default="openwebtext_gpt2_50m")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--rules", nargs="+", default=list(DEFAULT_RULES), help=RULE_HELP)
    parser.add_argument(
        "--include-center",
        action="store_true",
        help="also rerun center h2-h9 in this project; otherwise reuse the existing 50M center runs",
    )
    parser.add_argument("--mask-dir", default=str(default_mask_dir()))
    parser.add_argument("--static-mask-seed", type=int, default=2024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="Selective-Newton-Muon-OWT-50M-Mechanism")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-profile", type=str, default="paper", choices=["full", "paper"])
    parser.add_argument("--wandb-log-tables", action="store_true")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--run-prefix", type=str, default="owt_50m_tier3_mechanism_cf")
    args = parser.parse_args()

    if not ensure_data(args.dataset, args.dry_run):
        raise SystemExit(1)

    mask_dir = Path(args.mask_dir)
    rules = selected_rules(args)
    mask_paths = []
    for rule in rules:
        mask_path = mask_dir / mask_filename_for_rule(rule)
        if not mask_path.exists() and not args.dry_run:
            raise FileNotFoundError(f"mask path not found for rule={rule}: {mask_path}")
        mask_paths.append((rule, mask_path))

    wandb_enabled = args.wandb_mode != "disabled"
    for seed in args.seeds:
        group = args.wandb_group or f"{args.run_prefix}_seed{seed}"
        for rule, mask_path in mask_paths:
            target_release = read_mask_target(mask_path) if mask_path.exists() else 0.5614035087719298
            label = f"38_50m_mechanism_{rule}_{release_label(target_release)}"
            run_name = f"{args.run_prefix}_{label}_seed{seed}"
            tags = (
                f"formal,{args.dataset},tier3,selective_newton_muon,"
                f"50m_mechanism_counterfactual,{rule},mask_replay"
            )
            cmd = [
                sys.executable,
                "train.py",
                BASE_CONFIG,
                REPLAY_CONFIG,
                f"--dataset={args.dataset}",
                f"--seed={seed}",
                f"--batch_size={args.batch_size}",
                f"--max_iters={args.max_iters}",
                f"--lr_decay_iters={args.max_iters}",
                f"--eval_iters={args.eval_iters}",
                f"--out_dir=out_{run_name}",
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
