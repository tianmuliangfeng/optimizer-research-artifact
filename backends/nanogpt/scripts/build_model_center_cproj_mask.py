import argparse
import csv
import os
from pathlib import Path


MIB = 1024 * 1024
DEFAULT_TARGET_RELEASE = 0.5614035087719298


def release_label(release_frac):
    return f"release{int(round(release_frac * 100)):02d}"


def default_output_dir():
    artifact_root = Path(__file__).resolve().parents[3]
    results_root = Path(
        os.environ.get("SNM_RESULTS_ROOT", str(artifact_root / "runs"))
    ).expanduser()
    return (
        results_root
        / "analysis_exports"
        / "owt_50m_large_model_static_20260708/masks"
    )


def k_state_bytes(cols):
    return int(cols) * int(cols) * 4 * 3


def matrix_rows(n_layer, n_embd):
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


def centered_window(n_layer, count):
    count = max(1, min(n_layer, count))
    start = (n_layer - count) // 2
    end = start + count - 1
    return start, end


def choose_cproj_window(rows, n_layer, target_release_frac):
    total_bytes = sum(row["k_state_full_bytes"] for row in rows)
    cproj_bytes = next(row["k_state_full_bytes"] for row in rows if row["module_type"] == "mlp.c_proj")
    target_bytes = target_release_frac * total_bytes
    best = None
    for count in range(1, n_layer + 1):
        start, end = centered_window(n_layer, count)
        released_bytes = count * cproj_bytes
        actual = released_bytes / total_bytes
        err = abs(released_bytes - target_bytes)
        candidate = (err, -count, start, end, released_bytes, actual)
        if best is None or candidate < best:
            best = candidate
    _, _, start, end, released_bytes, actual = best
    return start, end, released_bytes, actual


def rank_rows(rows, released_names):
    return sorted(
        rows,
        key=lambda row: (
            0 if row["name"] in released_names else 1,
            row["module_type"],
            row["layer"],
            row["name"],
        ),
    )


def write_mask(path, rows, released_names, args, start, end, released_bytes, actual_frac):
    total_bytes = sum(row["k_state_full_bytes"] for row in rows)
    ranked = rank_rows(rows, released_names)
    rank_by_name = {row["name"]: idx for idx, row in enumerate(ranked, start=1)}
    rule = f"large_center_cproj_h{start}_h{end}"
    details = (
        f"model-spec generated middle mlp.c_proj window h{start}-h{end}; "
        f"n_layer={args.n_layer}; n_embd={args.n_embd}; target_release={args.target_release_frac}"
    )
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
        "muon_momentum_bytes_before_release",
        "muon_momentum_bytes_after_release",
        "newton_momentum_bytes_before_release",
        "newton_momentum_bytes_after_release",
        "selected",
        "released",
        "selection_mode",
        "static_mask_label",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            released = row["name"] in released_names
            full_bytes = row["k_state_full_bytes"]
            writer.writerow(
                {
                    "seed": args.mask_seed,
                    "dataset": args.dataset,
                    "wandb_project": args.wandb_project,
                    "wandb_group": args.wandb_group,
                    "wandb_run_name": f"{args.run_prefix}__mask_{rule}",
                    "optimizer_type": "selective_newton_muon",
                    "mask_rule": rule,
                    "target_release_k_fraction": actual_frac,
                    "target_release_k_state_bytes": released_bytes,
                    "actual_release_k_state_bytes": released_bytes,
                    "actual_release_k_fraction": actual_frac,
                    "rank": rank_by_name[row["name"]],
                    "name": row["name"],
                    "shape": row["shape"],
                    "rows": row["rows"],
                    "cols": row["cols"],
                    "score": 0,
                    "gain": 0,
                    "cost_proxy": 0,
                    "rule_details": details,
                    "source_report": "model_spec",
                    "k_state_full_bytes": full_bytes,
                    "k_state_bytes_before_release": full_bytes,
                    "k_state_bytes_after_release": 0 if released else full_bytes,
                    "muon_momentum_bytes_before_release": 0,
                    "muon_momentum_bytes_after_release": 0,
                    "newton_momentum_bytes_before_release": 0,
                    "newton_momentum_bytes_after_release": 0,
                    "selected": 0 if released else 1,
                    "released": 1 if released else 0,
                    "selection_mode": "oracle_static",
                    "static_mask_label": (
                        f"{rule}|actual={actual_frac:.12g}|"
                        f"released_mib={released_bytes / MIB:.2f}"
                    ),
                }
            )
    return {
        "rule": rule,
        "mask_path": str(path),
        "target_release_k_fraction": actual_frac,
        "target_release_k_state_mib": released_bytes / MIB,
        "actual_release_k_fraction": actual_frac,
        "actual_release_k_state_mib": released_bytes / MIB,
        "full_k_state_mib": total_bytes / MIB,
        "released_layers": len(released_names),
        "released_names": ";".join(name for name in sorted(released_names)),
        "rule_details": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--n-layer", type=int, required=True)
    parser.add_argument("--n-embd", type=int, required=True)
    parser.add_argument("--target-release-frac", type=float, default=DEFAULT_TARGET_RELEASE)
    parser.add_argument("--dataset", default="openwebtext_gpt2_50m")
    parser.add_argument("--mask-seed", type=int, default=2024)
    parser.add_argument("--run-prefix", default="owt_50m_large_model_static")
    parser.add_argument("--wandb-project", default="Selective-Newton-Muon-OWT-LargeModel")
    parser.add_argument("--wandb-group", default="large_model_static_masks")
    args = parser.parse_args()

    rows = matrix_rows(args.n_layer, args.n_embd)
    start, end, released_bytes, actual_frac = choose_cproj_window(
        rows, args.n_layer, args.target_release_frac
    )
    released_names = {
        row["name"]
        for row in rows
        if row["module_type"] == "mlp.c_proj" and start <= row["layer"] <= end
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dummy_path = output_dir / f"large_center_cproj_L{args.n_layer}_D{args.n_embd}_h{start}_h{end}.csv"
    summary = write_mask(dummy_path, rows, released_names, args, start, end, released_bytes, actual_frac)
    final_path = output_dir / (
        f"large_center_cproj_L{args.n_layer}_D{args.n_embd}_h{start}_h{end}_"
        f"{release_label(actual_frac)}.csv"
    )
    if final_path != dummy_path:
        os.replace(dummy_path, final_path)
        summary["mask_path"] = str(final_path)

    summary_path = output_dir / "large_model_center_cproj_mask_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(
        f"{summary['rule']}: released {summary['actual_release_k_state_mib']:.2f} MiB "
        f"({100.0 * summary['actual_release_k_fraction']:.2f}%), "
        f"layers={summary['released_layers']}"
    )
    print(f"wrote mask to {summary['mask_path']}")
    print(f"wrote mask summary to {summary_path}")


if __name__ == "__main__":
    main()
