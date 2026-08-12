import argparse
import csv
import os
import re
from pathlib import Path


DEFAULT_WINDOWS = ("h4-h7", "h3-h7", "h4-h8", "h3-h8", "h2-h8", "h3-h9", "h2-h9")
MIB = 1024 * 1024


def parse_int(value, default=0):
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_window(value):
    match = re.fullmatch(r"h?(\d+)[-_]h?(\d+)", value.strip())
    if not match:
        raise ValueError(f"invalid window {value!r}; expected h3-h8")
    start = int(match.group(1))
    end = int(match.group(2))
    if start > end:
        raise ValueError(f"window start must be <= end: {value}")
    return start, end


def layer_index(name):
    match = re.search(r"transformer\.h\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def is_mlp_c_proj(row):
    return ".mlp.c_proj.weight" in row.get("name", "")


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


def release_label(release_frac):
    return f"release{int(round(release_frac * 100)):02d}"


def read_source_report(path, source_seed, source_run_name):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.DictReader(f))
    if not raw_rows:
        raise ValueError(f"empty source report: {path}")

    rows = []
    seen = set()
    for row in raw_rows:
        if source_seed >= 0 and row.get("seed", "") != str(source_seed):
            continue
        if source_run_name and row.get("wandb_run_name", "") != source_run_name:
            continue
        name = row.get("name", "")
        if not name or name in seen:
            continue
        enriched = dict(row)
        enriched["_layer_idx"] = layer_index(name)
        enriched["_k_state_full_bytes"] = k_state_full_bytes(row)
        rows.append(enriched)
        seen.add(name)

    if not rows:
        raise ValueError(
            "No source rows matched filters: "
            f"path={path}, source_seed={source_seed}, source_run_name={source_run_name!r}"
        )
    return rows


def rank_rows(rows, released_names):
    def layer_key(row):
        return (row.get("_layer_idx", -1), row.get("name", ""))

    released = sorted([row for row in rows if row.get("name", "") in released_names], key=layer_key)
    cproj_remaining = sorted(
        [row for row in rows if is_mlp_c_proj(row) and row.get("name", "") not in released_names],
        key=layer_key,
    )
    other = sorted([row for row in rows if not is_mlp_c_proj(row)], key=lambda row: row.get("name", ""))
    return released + cproj_remaining + other


def source_value(row, key, default=""):
    value = row.get(key, "")
    return default if value == "" else value


def write_mask(path, rows, window, mask_seed, source_path):
    start, end = window
    total_bytes = sum(row["_k_state_full_bytes"] for row in rows)
    released_names = {
        row.get("name", "")
        for row in rows
        if is_mlp_c_proj(row) and start <= row.get("_layer_idx", -1) <= end
    }
    if not released_names:
        raise ValueError(f"window h{start}-h{end} released no mlp.c_proj rows")

    released_bytes = sum(row["_k_state_full_bytes"] for row in rows if row.get("name", "") in released_names)
    release_frac = released_bytes / total_bytes if total_bytes > 0 else 0.0
    ranked = rank_rows(rows, released_names)
    rank_by_name = {row.get("name", ""): rank for rank, row in enumerate(ranked, start=1)}
    rule = f"static_center_h{start}_h{end}"
    details = f"fixed middle mlp.c_proj window h{start}-h{end}"

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
            name = row.get("name", "")
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
                    "rank": rank_by_name.get(name, len(rank_by_name) + 1),
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

    released_ranked = [row for row in ranked if row.get("name", "") in released_names]
    return {
        "rule": rule,
        "window": f"h{start}-h{end}",
        "mask_path": path,
        "target_release_k_fraction": release_frac,
        "target_release_k_state_mib": released_bytes / MIB,
        "actual_release_k_fraction": release_frac,
        "actual_release_k_state_mib": released_bytes / MIB,
        "released_layers": len(released_names),
        "released_names": ";".join(row.get("name", "") for row in released_ranked),
        "full_k_state_mib": total_bytes / MIB,
    }


def default_output_dir(source_report):
    artifact_root = Path(__file__).resolve().parents[3]
    results_root = Path(
        os.environ.get("SNM_RESULTS_ROOT", str(artifact_root / "runs"))
    ).expanduser()
    return str(
        results_root
        / "analysis_exports"
        / "owt_tier3_static_center_sweep_20260707"
        / "masks"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--windows", nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--source-seed", type=int, default=-1)
    parser.add_argument("--source-run-name", default="")
    parser.add_argument("--mask-seed", type=int, default=2024)
    args = parser.parse_args()

    output_dir = args.output_dir or default_output_dir(args.source_report)
    os.makedirs(output_dir, exist_ok=True)

    source_report = os.path.abspath(args.source_report)
    rows = read_source_report(source_report, args.source_seed, args.source_run_name)
    summary_rows = []
    for window_text in args.windows:
        start, end = parse_window(window_text)
        dummy_frac = (end - start + 1) / max(1, len([row for row in rows if is_mlp_c_proj(row)]))
        mask_name = f"static_center_h{start}_h{end}_{release_label(dummy_frac)}.csv"
        mask_path = os.path.join(output_dir, mask_name)
        summary = write_mask(mask_path, rows, (start, end), args.mask_seed, source_report)
        actual_label = release_label(float(summary["actual_release_k_fraction"]))
        final_path = os.path.join(output_dir, f"static_center_h{start}_h{end}_{actual_label}.csv")
        if final_path != mask_path:
            os.replace(mask_path, final_path)
            summary["mask_path"] = final_path
        summary_rows.append(summary)
        print(
            f"{summary['rule']}: released {summary['actual_release_k_state_mib']:.2f} MiB "
            f"({100.0 * summary['actual_release_k_fraction']:.2f}%); "
            + summary["released_names"]
        )

    summary_path = os.path.join(output_dir, "static_center_sweep_mask_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote mask summary to {summary_path}")


if __name__ == "__main__":
    main()
