from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from project_paths import EXPERIMENT_RESULTS_ROOT
from runner_utils import (
    append_common_train_overrides,
    append_model_overrides,
    ensure_data,
    release_label,
    run_cmd,
    selective_train_cmd,
    write_command_record,
)


BASE_CONFIG = "config/train_openwebtext_gpt2_50m_tier4.py"
FAMILY = "01_scale_up"
MIB = 1024 * 1024


def default_mask_dir() -> str:
    return str(EXPERIMENT_RESULTS_ROOT / "_shared" / "masks" / FAMILY / "module_bridge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 24L all-non-mlp.c_proj K-state release bridge control. "
            "Existing full-Newton and all-mlp.c_proj runs are reused."
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-exe", default=None)
    parser.add_argument("--dataset", default="openwebtext_gpt2_50m")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024])
    parser.add_argument("--base-config", default=BASE_CONFIG)
    parser.add_argument("--n-layer", type=int, default=24)
    parser.add_argument("--n-head", type=int, default=16)
    parser.add_argument("--n-embd", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=12000)
    parser.add_argument("--lr-decay-iters", type=int, default=3000)
    parser.add_argument("--muon-learning-rate", type=float, default=0.01)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--always-save-checkpoint", action="store_true", default=False)
    parser.add_argument("--static-mask-seed", type=int, default=2024)
    parser.add_argument("--mask-dir", default=default_mask_dir())
    parser.add_argument(
        "--wandb-project",
        default="Selective-Newton-Muon-MainConf-Mechanism-ScaleBridge",
    )
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-profile", default="paper", choices=["full", "paper"])
    parser.add_argument("--wandb-log-tables", action="store_true")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--run-prefix", default="mainconf_mechanism_24L_non_cproj_bridge")
    parser.add_argument("--no-write-commands", action="store_false", dest="write_commands")
    parser.set_defaults(write_commands=True)
    return parser.parse_args()


def k_state_bytes(cols: int) -> int:
    return cols * cols * 4 * 3


def matrix_rows(n_layer: int, n_embd: int) -> list[dict[str, object]]:
    specs = (
        ("attn.c_attn", "attn.c_attn.weight", 3 * n_embd, n_embd),
        ("attn.c_proj", "attn.c_proj.weight", n_embd, n_embd),
        ("mlp.c_fc", "mlp.c_fc.weight", 4 * n_embd, n_embd),
        ("mlp.c_proj", "mlp.c_proj.weight", n_embd, 4 * n_embd),
    )
    rows = []
    for layer in range(n_layer):
        for module_type, suffix, out_dim, in_dim in specs:
            rows.append(
                {
                    "name": f"transformer.h.{layer}.{suffix}",
                    "layer": layer,
                    "module_type": module_type,
                    "rows": out_dim,
                    "cols": in_dim,
                    "shape": f"{out_dim}x{in_dim}",
                    "k_state_full_bytes": k_state_bytes(in_dim),
                }
            )
    return rows


def build_non_cproj_mask(args: argparse.Namespace) -> tuple[str, float]:
    rows = matrix_rows(args.n_layer, args.n_embd)
    total_bytes = sum(int(row["k_state_full_bytes"]) for row in rows)
    released_names = {
        str(row["name"]) for row in rows if row["module_type"] != "mlp.c_proj"
    }
    released_bytes = sum(
        int(row["k_state_full_bytes"])
        for row in rows
        if str(row["name"]) in released_names
    )
    release_frac = released_bytes / total_bytes
    label = release_label(release_frac)
    output_dir = Path(args.mask_dir) / f"L{args.n_layer}_D{args.n_embd}" / "non_cproj_all"
    mask_path = output_dir / (
        f"non_cproj_all_L{args.n_layer}_D{args.n_embd}_{label}.csv"
    )
    if args.dry_run:
        return str(mask_path), release_frac

    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        rows,
        key=lambda row: (
            0 if str(row["name"]) in released_names else 1,
            str(row["module_type"]),
            int(row["layer"]),
        ),
    )
    rank_by_name = {str(row["name"]): index for index, row in enumerate(ranked, start=1)}
    fieldnames = [
        "seed",
        "dataset",
        "wandb_project",
        "wandb_group",
        "wandb_run_name",
        "optimizer_type",
        "mask_rule",
        "target_release_k_fraction",
        "target_release_k_state_bytes",
        "actual_release_k_state_bytes",
        "actual_release_k_fraction",
        "rank",
        "name",
        "shape",
        "rows",
        "cols",
        "score",
        "gain",
        "cost_proxy",
        "rule_details",
        "source_report",
        "k_state_full_bytes",
        "k_state_bytes_before_release",
        "k_state_bytes_after_release",
        "selected",
        "released",
        "selection_mode",
        "static_mask_label",
    ]
    with mask_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            name = str(row["name"])
            released = name in released_names
            full_bytes = int(row["k_state_full_bytes"])
            writer.writerow(
                {
                    "seed": args.static_mask_seed,
                    "dataset": args.dataset,
                    "wandb_project": args.wandb_project,
                    "wandb_group": args.wandb_group or f"{args.run_prefix}_masks",
                    "wandb_run_name": f"{args.run_prefix}__mask_non_cproj_all",
                    "optimizer_type": "selective_newton_muon",
                    "mask_rule": "non_cproj_all",
                    "target_release_k_fraction": release_frac,
                    "target_release_k_state_bytes": released_bytes,
                    "actual_release_k_state_bytes": released_bytes,
                    "actual_release_k_fraction": release_frac,
                    "rank": rank_by_name[name],
                    "name": name,
                    "shape": row["shape"],
                    "rows": row["rows"],
                    "cols": row["cols"],
                    "score": 0,
                    "gain": 0,
                    "cost_proxy": 0,
                    "rule_details": (
                        "24L scale bridge: release every eligible matrix K-state "
                        "except mlp.c_proj"
                    ),
                    "source_report": "model_spec",
                    "k_state_full_bytes": full_bytes,
                    "k_state_bytes_before_release": full_bytes,
                    "k_state_bytes_after_release": 0 if released else full_bytes,
                    "selected": 0 if released else 1,
                    "released": 1 if released else 0,
                    "selection_mode": "oracle_static",
                    "static_mask_label": (
                        f"non_cproj_all|actual={release_frac:.12g}|"
                        f"released_mib={released_bytes / MIB:.2f}"
                    ),
                }
            )
    print(
        f"wrote {mask_path}: released {released_bytes / MIB:.2f} MiB "
        f"({100.0 * release_frac:.2f}%)"
    )
    return str(mask_path), release_frac


def build_commands(
    args: argparse.Namespace, mask_path: str, release_frac: float
) -> list[list[str]]:
    commands = []
    label = release_label(release_frac)
    tags = (
        f"publication,{args.dataset},mechanism,scale_bridge,L{args.n_layer},"
        f"D{args.n_embd},non_cproj_all,{label},reuse_newton_and_release84"
    )
    for seed in args.seeds:
        args.seed = seed
        group = args.wandb_group or f"{args.run_prefix}_seed{seed}"
        run_name = f"{args.run_prefix}_L{args.n_layer}_D{args.n_embd}_{label}_seed{seed}"
        cmd = selective_train_cmd(
            args,
            config=args.base_config,
            run_name=run_name,
            group=group,
            tags=tags,
            mask_path=mask_path,
            release_frac=release_frac,
            method_label="release_all_non_cproj",
        )
        append_model_overrides(cmd, args)
        append_common_train_overrides(cmd, args)
        commands.append(cmd)
    return commands


def main() -> None:
    args = parse_args()
    if not ensure_data(args.dataset, args.dry_run):
        raise SystemExit(1)
    mask_path, release_frac = build_non_cproj_mask(args)
    commands = build_commands(args, mask_path, release_frac)
    write_command_record(
        family=FAMILY,
        run_prefix=args.run_prefix,
        commands=commands,
        dry_run=args.dry_run,
        enabled=args.write_commands,
    )
    for cmd in commands:
        run_cmd(cmd, args.dry_run)


if __name__ == "__main__":
    main()
