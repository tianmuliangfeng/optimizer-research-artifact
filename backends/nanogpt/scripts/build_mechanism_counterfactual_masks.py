import argparse
import csv
import os
import random
import re
from pathlib import Path


MIB = 1024 * 1024


DEFAULT_RULES = (
    "cproj_center_h2_h9",
    "cproj_early_h0_h7",
    "cproj_late_h4_h11",
    "cproj_edge_h0_h3_h8_h11",
    "cproj_random8_s0",
    "non_cproj_all",
    "cproj_middle_h5_h6",
)

MODULE_ALL_RULES = {
    "attn_c_attn_all": ("attn.c_attn", "release all attention QKV projection K-state"),
    "attn_c_proj_all": ("attn.c_proj", "release all attention output projection K-state"),
    "mlp_c_fc_all": ("mlp.c_fc", "release all MLP input projection K-state"),
}

SAME_LAYER_MODULE_RULES = {
    "attn_c_attn_h5_h6": (
        "attn.c_attn",
        range(5, 7),
        "same-layer control: release attention QKV projection K-state in h5-h6",
    ),
    "attn_c_proj_h5_h6": (
        "attn.c_proj",
        range(5, 7),
        "same-layer control: release attention output projection K-state in h5-h6",
    ),
    "mlp_c_fc_h5_h6": (
        "mlp.c_fc",
        range(5, 7),
        "same-layer control: release MLP input projection K-state in h5-h6",
    ),
}


def parse_int(value, default=0):
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def layer_index(name):
    match = re.search(r"transformer\.h\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def module_type(name):
    if ".attn.c_attn.weight" in name:
        return "attn.c_attn"
    if ".attn.c_proj.weight" in name:
        return "attn.c_proj"
    if ".mlp.c_fc.weight" in name:
        return "mlp.c_fc"
    if ".mlp.c_proj.weight" in name:
        return "mlp.c_proj"
    return "other"


def k_state_full_bytes(row):
    for key in ("k_state_full_bytes", "k_state_bytes_before_release"):
        value = parse_int(row.get(key, ""), default=-1)
        if value > 0:
            return value
    cols = parse_int(row.get("cols", ""), default=-1)
    if cols > 0:
        return cols * cols * 4 * 3
    shape = row.get("shape", "")
    match = re.fullmatch(r"(\d+)x(\d+)", shape)
    if match:
        cols = int(match.group(2))
        return cols * cols * 4 * 3
    raise ValueError(f"could not infer k_state_full_bytes for {row.get('name', '<unnamed>')}")


def source_value(row, key, default=""):
    value = row.get(key, "")
    return default if value == "" else value


def release_label(release_frac):
    return f"release{int(round(release_frac * 100)):02d}"


def default_source_path():
    artifact_root = Path(__file__).resolve().parents[3]
    results_root = Path(
        os.environ.get("SNM_RESULTS_ROOT", str(artifact_root / "runs"))
    ).expanduser()
    return (
        results_root
        / "analysis_exports"
        / "owt_tier3_static_center_sweep_20260707/masks/static_center_h2_h9_release56.csv"
    )


def default_output_dir():
    artifact_root = Path(__file__).resolve().parents[3]
    results_root = Path(
        os.environ.get("SNM_RESULTS_ROOT", str(artifact_root / "runs"))
    ).expanduser()
    return (
        results_root
        / "analysis_exports"
        / "owt_tier3_mechanism_counterfactual_masks_20260708/masks"
    )


def read_source_rows(path, source_seed):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.DictReader(f))
    if not raw_rows:
        raise ValueError(f"empty source report: {path}")

    rows = []
    seen = set()
    for row in raw_rows:
        if source_seed >= 0 and row.get("seed", "") != str(source_seed):
            continue
        name = row.get("name", "")
        if not name or name in seen:
            continue
        enriched = dict(row)
        enriched["_layer_idx"] = layer_index(name)
        enriched["_module_type"] = module_type(name)
        enriched["_k_state_full_bytes"] = k_state_full_bytes(row)
        rows.append(enriched)
        seen.add(name)

    if not rows:
        raise ValueError(f"no rows matched source_seed={source_seed} in {path}")
    return rows


def cproj_name_set(rows, layers):
    layer_set = set(layers)
    return {
        row["name"]
        for row in rows
        if row["_module_type"] == "mlp.c_proj" and row["_layer_idx"] in layer_set
    }


def module_name_set(rows, module_name, layers):
    layer_set = set(layers)
    return {
        row["name"]
        for row in rows
        if row["_module_type"] == module_name and row["_layer_idx"] in layer_set
    }


def released_names_for_rule(rows, rule):
    cproj_layers = sorted({row["_layer_idx"] for row in rows if row["_module_type"] == "mlp.c_proj"})
    if rule == "cproj_center_h2_h9":
        return cproj_name_set(rows, range(2, 10)), "main center mlp.c_proj h2-h9"
    if rule == "cproj_early_h0_h7":
        return cproj_name_set(rows, range(0, 8)), "same-budget early mlp.c_proj h0-h7"
    if rule == "cproj_late_h4_h11":
        return cproj_name_set(rows, range(4, 12)), "same-budget late mlp.c_proj h4-h11"
    if rule == "cproj_edge_h0_h3_h8_h11":
        return cproj_name_set(rows, list(range(0, 4)) + list(range(8, 12))), "same-budget edge mlp.c_proj h0-h3 plus h8-h11"
    if rule.startswith("cproj_random8_s"):
        seed_text = rule.rsplit("_s", 1)[-1]
        seed = int(seed_text)
        rng = random.Random(seed)
        chosen = sorted(rng.sample(cproj_layers, 8))
        return cproj_name_set(rows, chosen), f"same-budget random 8 mlp.c_proj layers seed={seed}; layers={chosen}"
    if rule == "non_cproj_all":
        return {row["name"] for row in rows if row["_module_type"] != "mlp.c_proj"}, (
            "module-type stress test: release all non-mlp.c_proj K-state; "
            "not same-budget because non-cproj K-state is much smaller"
        )
    if rule in MODULE_ALL_RULES:
        module_name, details = MODULE_ALL_RULES[rule]
        return {row["name"] for row in rows if row["_module_type"] == module_name}, details
    if rule in SAME_LAYER_MODULE_RULES:
        module_name, layers, details = SAME_LAYER_MODULE_RULES[rule]
        return module_name_set(rows, module_name, layers), details
    if rule == "cproj_middle_h5_h6":
        return cproj_name_set(rows, range(5, 7)), (
            "rough small-budget mlp.c_proj comparison for non_cproj_all"
        )
    raise ValueError(f"unknown counterfactual rule: {rule}")


def rank_rows(rows, released_names):
    def key(row):
        released_rank = 0 if row["name"] in released_names else 1
        return (released_rank, row["_module_type"], row["_layer_idx"], row["name"])

    return sorted(rows, key=key)


def write_mask(path, rows, released_names, rule, details, mask_seed, source_path):
    total_bytes = sum(row["_k_state_full_bytes"] for row in rows)
    released_bytes = sum(row["_k_state_full_bytes"] for row in rows if row["name"] in released_names)
    if released_bytes <= 0:
        raise ValueError(f"rule {rule} released no K-state")
    release_frac = released_bytes / total_bytes if total_bytes > 0 else 0.0
    ranked = rank_rows(rows, released_names)
    rank_by_name = {row["name"]: rank for rank, row in enumerate(ranked, start=1)}

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
            name = row["name"]
            released = name in released_names
            full_bytes = row["_k_state_full_bytes"]
            writer.writerow(
                {
                    "seed": mask_seed,
                    "dataset": source_value(row, "dataset", "openwebtext_gpt2"),
                    "wandb_project": source_value(row, "wandb_project"),
                    "wandb_group": source_value(row, "wandb_group"),
                    "wandb_run_name": f"{source_value(row, 'wandb_run_name')}__mask_{rule}",
                    "optimizer_type": "selective_newton_muon",
                    "mask_rule": rule,
                    "target_release_k_fraction": release_frac,
                    "target_release_k_state_bytes": released_bytes,
                    "actual_release_k_state_bytes": released_bytes,
                    "actual_release_k_fraction": release_frac,
                    "rank": rank_by_name[name],
                    "name": name,
                    "shape": source_value(row, "shape"),
                    "rows": source_value(row, "rows"),
                    "cols": source_value(row, "cols"),
                    "score": source_value(row, "score", 0),
                    "gain": source_value(row, "gain", 0),
                    "cost_proxy": source_value(row, "cost_proxy", 0),
                    "rule_details": details,
                    "source_report": source_path,
                    "k_state_full_bytes": full_bytes,
                    "k_state_bytes_before_release": full_bytes,
                    "k_state_bytes_after_release": 0 if released else full_bytes,
                    "muon_momentum_bytes_before_release": source_value(row, "muon_momentum_bytes_before_release", 0),
                    "muon_momentum_bytes_after_release": source_value(row, "muon_momentum_bytes_after_release", 0),
                    "newton_momentum_bytes_before_release": source_value(row, "newton_momentum_bytes_before_release", 0),
                    "newton_momentum_bytes_after_release": source_value(row, "newton_momentum_bytes_after_release", 0),
                    "selected": 0 if released else 1,
                    "released": 1 if released else 0,
                    "selection_mode": "oracle_static",
                    "static_mask_label": f"{rule}|actual={release_frac:.12g}|released_mib={released_bytes / MIB:.2f}",
                }
            )

    released_ranked = [row for row in ranked if row["name"] in released_names]
    released_module_counts = {}
    for row in released_ranked:
        released_module_counts[row["_module_type"]] = released_module_counts.get(row["_module_type"], 0) + 1
    return {
        "rule": rule,
        "mask_path": str(path),
        "actual_release_k_fraction": release_frac,
        "actual_release_k_state_mib": released_bytes / MIB,
        "full_k_state_mib": total_bytes / MIB,
        "released_layers": len(released_names),
        "released_module_counts": ";".join(f"{k}:{v}" for k, v in sorted(released_module_counts.items())),
        "released_names": ";".join(row["name"] for row in released_ranked),
        "rule_details": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", default=str(default_source_path()))
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--rules", nargs="+", default=list(DEFAULT_RULES))
    parser.add_argument("--source-seed", type=int, default=2024)
    parser.add_argument("--mask-seed", type=int, default=2024)
    args = parser.parse_args()

    source_report = os.path.abspath(args.source_report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_source_rows(source_report, args.source_seed)

    summary_rows = []
    for rule in args.rules:
        released_names, details = released_names_for_rule(rows, rule)
        dummy_path = output_dir / f"mechanism_{rule}.csv"
        summary = write_mask(dummy_path, rows, released_names, rule, details, args.mask_seed, source_report)
        final_path = output_dir / f"mechanism_{rule}_{release_label(summary['actual_release_k_fraction'])}.csv"
        if final_path != dummy_path:
            os.replace(dummy_path, final_path)
            summary["mask_path"] = str(final_path)
        summary_rows.append(summary)
        print(
            f"{rule}: released {summary['actual_release_k_state_mib']:.2f} MiB "
            f"({100.0 * summary['actual_release_k_fraction']:.2f}%), "
            f"layers={summary['released_layers']}, modules={summary['released_module_counts']}"
        )

    summary_path = output_dir / "mechanism_counterfactual_mask_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote mask summary to {summary_path}")


if __name__ == "__main__":
    main()
