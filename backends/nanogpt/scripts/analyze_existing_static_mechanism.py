import argparse
import csv
import os
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def f(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def fmt(x: float, digits: int = 6) -> str:
    return f"{x:.{digits}f}"


def window_note(method: str) -> str:
    notes = {
        "static_center_h4_h8": "conservative center window; best low-loss 10M sweep point",
        "static_center_h2_h9": "aggressive center window; current strongest memory-saving point",
        "static_center_h3_h8": "historical band anchor; weaker than h4-h8/h2-h9 in controlled replay",
        "static_center_h2_h8": "same-budget evidence that shifting the window too early can hurt",
        "static_center_h3_h9": "same-budget evidence that layer position matters",
        "static_center_h3_h7": "same-budget counterpart to h4-h8 at 35.09% release",
        "static_center_h4_h7": "narrow center window; lower memory saving",
    }
    return notes.get(method, "")


def build_window_sweep(static_compare: list[dict], layer_rows: list[dict]) -> list[dict]:
    by_method = {row["method"]: row for row in layer_rows}
    rows = []
    for row in static_compare:
        method = row["method"]
        layer = by_method.get(method, {})
        rows.append(
            {
                "evidence_type": "static_center_window_sweep",
                "dataset": "OpenWebText-GPT2 10M train / 200k val",
                "model": "tier3 12L-12H-768d",
                "batch_size": "16",
                "seeds": "2024",
                "method": method,
                "release_window": row.get("release_window", ""),
                "released_layers": layer.get("released_layers", ""),
                "released_cproj_layers": layer.get("released_cproj_layers", ""),
                "release_frac_pct": fmt(f(row, "release_frac_pct")),
                "k_state_released_mib": fmt(f(row, "k_state_released_mib"), 3),
                "k_state_remaining_mib": fmt(f(row, "k_state_remaining_mib"), 3),
                "best_val_loss": fmt(f(row, "best_val_loss")),
                "delta_val_vs_newton": fmt(f(row, "delta_val_vs_newton")),
                "newton_gain_preserved": fmt(f(row, "newton_gain_preserved")),
                "time_vs_newton_x": fmt(f(row, "time_vs_newton_x")),
                "peak_saved_vs_newton_pct": fmt(f(row, "peak_saved_vs_newton_pct")),
                "current_saved_vs_newton_pct": fmt(f(row, "current_saved_vs_newton_pct")),
                "released_names": layer.get("released_names", ""),
                "mechanism_note": window_note(method),
            }
        )
    return rows


def build_same_budget(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        release = f(row, "release_frac_pct")
        gap = f(row, "abs_val_gap")
        out.append(
            {
                "evidence_type": "same_budget_position_sensitivity",
                "dataset": "OpenWebText-GPT2 10M train / 200k val",
                "model": "tier3 12L-12H-768d",
                "batch_size": "16",
                "seeds": "2024",
                "release_frac_pct": fmt(release),
                "method_a": row.get("method_a", ""),
                "window_a": row.get("window_a", ""),
                "val_a": fmt(f(row, "val_a")),
                "method_b": row.get("method_b", ""),
                "window_b": row.get("window_b", ""),
                "val_b": fmt(f(row, "val_b")),
                "abs_val_gap": fmt(gap),
                "better_window": row.get("better_window", ""),
                "mechanism_note": (
                    "At matched K-state release, changing only layer position changes loss; "
                    "release fraction alone does not explain the result."
                ),
            }
        )
    return out


def rows_by_method(rows: list[dict]) -> dict[str, dict]:
    return {row["method"]: row for row in rows}


def build_scale_consistency(main10: list[dict], main50: list[dict]) -> list[dict]:
    ten = rows_by_method(main10)
    fifty = rows_by_method(main50)
    newton10 = f(ten["newton"], "val_loss_mean")
    newton50 = f(fifty["newton"], "val_loss_mean")
    methods = ["newton", "static_center_h4_h8", "static_center_h2_h9", "static_center_h3_h8"]
    rows = []
    for method in methods:
        if method not in ten or method not in fifty:
            continue
        r10 = ten[method]
        r50 = fifty[method]
        delta10 = f(r10, "val_loss_mean") - newton10
        delta50 = f(r50, "val_loss_mean") - newton50
        note = ""
        if method == "static_center_h4_h8":
            note = "Conservative window remains near full Newton-Muon from 10M to 50M."
        elif method == "static_center_h2_h9":
            note = "Aggressive window improves from mild 10M degradation to 50M parity/slight win."
        elif method == "static_center_h3_h8":
            note = "Historical anchor remains weaker than the two selected center candidates."
        elif method == "newton":
            note = "Full Newton-Muon reference."
        rows.append(
            {
                "evidence_type": "scale_consistency",
                "method": method,
                "release_window": r10.get("release_window", r50.get("release_window", "")),
                "release_frac_pct": fmt(f(r10, "release_frac_pct")),
                "val_loss_10m_mean": fmt(f(r10, "val_loss_mean")),
                "delta_10m_vs_newton": fmt(delta10),
                "gain_preserved_10m": fmt(f(r10, "newton_gain_preserved_mean")),
                "peak_saving_10m_pct": fmt(f(r10, "peak_saving_vs_newton_pct_mean")),
                "current_saving_10m_pct": fmt(f(r10, "current_saving_vs_newton_pct_mean")),
                "val_loss_50m_mean": fmt(f(r50, "val_loss_mean")),
                "delta_50m_vs_newton": fmt(delta50),
                "gain_preserved_50m": fmt(f(r50, "newton_gain_preserved_mean")),
                "peak_saving_50m_pct": fmt(f(r50, "peak_saving_vs_newton_pct_mean")),
                "current_saving_50m_pct": fmt(f(r50, "current_saving_vs_newton_pct_mean")),
                "scale_note": note,
            }
        )
    return rows


def first(path: Path) -> dict:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows[0]


def build_auxiliary_controls(root: Path) -> list[dict]:
    b16 = first(
        root
        / "analysis_exports/owt_tier3_b16_20260707/owt_tier3_b16_compact.csv"
    )
    cheap = first(
        root
        / "analysis_exports/owt_tier3_cheap_20260707/owt_tier3_cheap_vs_band_compare.csv"
    )
    imp = first(
        root
        / "analysis_exports/owt_tier3_cheap_muon_importance_replay_20260707/owt_tier3_cheap_muon_importance_replay_compare.csv"
    )
    contiguous = first(
        root
        / "analysis_exports/owt_tier3_cheap_muon_contiguous_window_l025_20260707/owt_tier3_cheap_muon_contiguous_window_l025_compare.csv"
    )
    fixed = first(
        root
        / "analysis_exports/owt_tier3_cheap_muon_fixed_window_h2_h7_20260707/owt_tier3_cheap_muon_fixed_window_h2_h7_compare.csv"
    )

    def row(method, window, release_frac, val, delta, gain, note):
        return {
            "evidence_type": "auxiliary_negative_or_anchor_control",
            "dataset": "OpenWebText-GPT2 10M train / 200k val",
            "model": "tier3 12L-12H-768d",
            "batch_size": "16",
            "seeds": "2024",
            "method": method,
            "release_window": window,
            "release_frac_pct": fmt(release_frac * 100.0),
            "best_val_loss": fmt(val),
            "delta_val_vs_newton": fmt(delta),
            "newton_gain_preserved": fmt(gain),
            "mechanism_note": note,
        }

    return [
        row(
            "band_middle_release40",
            "h3-h8",
            f(b16, "k_release_frac"),
            f(b16, "band_best_val_loss"),
            f(b16, "best_val_loss_delta_vs_newton"),
            f(b16, "newton_gain_preserved"),
            "Historical band anchor: useful, but weaker than the best static center sweep points.",
        ),
        row(
            "ordinary_cheap",
            "cheap-selected",
            f(cheap, "k_release_frac"),
            f(cheap, "cheap_best_val_loss"),
            f(cheap, "cheap_val_delta_vs_newton"),
            f(cheap, "cheap_newton_gain_preserved"),
            "Cheap/static heuristic underperforms center windows; module/position structure matters.",
        ),
        row(
            "cheap_muon_importance_replay",
            "probe-importance",
            f(imp, "k_release_frac"),
            f(imp, "importance_replay_best_val_loss"),
            f(imp, "importance_delta_vs_newton"),
            f(imp, "importance_newton_gain_preserved"),
            "Dynamic cheap-probe replay is a negative control for simple online importance rules.",
        ),
        row(
            "cheap_muon_contiguous_window_l025",
            "h4-h9",
            f(contiguous, "k_release_frac"),
            f(contiguous, "contiguous_window_l025_best_val_loss"),
            f(contiguous, "window_delta_vs_newton"),
            f(contiguous, "window_newton_gain_preserved"),
            "Probe-derived contiguous window remains weaker than static center candidates.",
        ),
        row(
            "cheap_muon_fixed_window_h2_h7",
            "h2-h7",
            f(fixed, "k_release_frac"),
            f(fixed, "fixed_window_h2_h7_best_val_loss"),
            f(fixed, "h2h7_delta_vs_newton"),
            f(fixed, "h2h7_newton_gain_preserved"),
            "Conservative fixed window improves over cheap rules but is not the best center release.",
        ),
    ]


def write_notes(path: Path) -> None:
    text = """Tier3 static mechanism analysis from existing data
Date: 2026-07-08

Purpose:
This folder collects the first mechanism-analysis group without running new training.
It converts existing static sweep, multiseed, 50M recheck, and auxiliary negative-control results into paper-facing mechanism evidence.

Mechanism question:
Why can we release K-state in the middle mlp.c_proj layers without materially hurting validation loss?

Evidence assembled here:
1. static_center_window_sweep.csv
   Shows the controlled tier3 center-window sweep at seed 2024.
   All rows release mlp.c_proj windows, so the main variable is layer position and window width.

2. same_budget_position_sensitivity.csv
   Shows matched-release comparisons.
   At 35.09% release, h4-h8 beats h3-h7 by 0.00193 val loss.
   At 49.12% release, h3-h9 beats h2-h8 by 0.00585 val loss.
   This supports the claim that release fraction alone does not determine the result.

3. scale_consistency_10m_50m.csv
   Compares multiseed 10M and 50M aggregates.
   h4-h8 stays near full Newton-Muon.
   h2-h9 moves from mild 10M degradation to 50M parity/slight win while keeping larger memory savings.

4. auxiliary_negative_controls.csv
   Summarizes band, cheap, cheap-probe importance replay, probe-derived contiguous window, and fixed h2-h7 controls.
   These results show that not every 40%-ish release rule works equally well.

Current paper-facing mechanism statement:
The useful release pattern is structured rather than arbitrary. For tier3, middle mlp.c_proj layers form a tolerant K-state release region: releasing them gives substantial K-state and CUDA-memory savings, while preserving full Newton-Muon quality across the larger 50M-token replication. Same-budget comparisons and cheap/probe controls indicate that both module type and layer position matter, not only the total released K-state fraction.

Limit:
This first group is behavioral evidence from existing outcomes. It does not yet directly measure Muon-vs-Newton update similarity. That should be the second mechanism group if we want a stronger mechanistic explanation.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to $SNM_RESULTS_ROOT/mechanism_analysis/tier3_static_existing_data_20260708",
    )
    args = parser.parse_args()

    artifact_root = Path(__file__).resolve().parents[3]
    results_root = Path(
        os.environ.get("SNM_RESULTS_ROOT", str(artifact_root / "runs"))
    ).expanduser()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else results_root
        / "mechanism_analysis/tier3_static_existing_data_20260708"
    )

    static_dir = results_root / "paper_key_results/tier3_static_center_sweep_20260707/summaries"
    main_dir = results_root / "paper_key_results/tier3_main_results_seed2024_2026_20260707/summaries"
    fifty_dir = results_root / "paper_key_results/tier3_50m_static_recheck_20260708/summaries"

    static_compare = read_csv(static_dir / "tier3_static_center_sweep_paper_compare_table.csv")
    layer_rows = read_csv(static_dir / "tier3_static_center_sweep_layer_release_summary.csv")
    same_budget = read_csv(static_dir / "tier3_static_center_sweep_same_budget_position_sensitivity.csv")
    main10 = read_csv(main_dir / "tier3_paper_main_compact_table_seed2024_2026.csv")
    main50 = read_csv(fifty_dir / "tier3_50m_paper_compact_table_seed2024_2025.csv")

    window_rows = build_window_sweep(static_compare, layer_rows)
    same_budget_rows = build_same_budget(same_budget)
    scale_rows = build_scale_consistency(main10, main50)
    aux_rows = build_auxiliary_controls(results_root)

    write_csv(
        output_dir / "static_center_window_sweep.csv",
        window_rows,
        [
            "evidence_type",
            "dataset",
            "model",
            "batch_size",
            "seeds",
            "method",
            "release_window",
            "released_layers",
            "released_cproj_layers",
            "release_frac_pct",
            "k_state_released_mib",
            "k_state_remaining_mib",
            "best_val_loss",
            "delta_val_vs_newton",
            "newton_gain_preserved",
            "time_vs_newton_x",
            "peak_saved_vs_newton_pct",
            "current_saved_vs_newton_pct",
            "released_names",
            "mechanism_note",
        ],
    )
    write_csv(
        output_dir / "same_budget_position_sensitivity.csv",
        same_budget_rows,
        [
            "evidence_type",
            "dataset",
            "model",
            "batch_size",
            "seeds",
            "release_frac_pct",
            "method_a",
            "window_a",
            "val_a",
            "method_b",
            "window_b",
            "val_b",
            "abs_val_gap",
            "better_window",
            "mechanism_note",
        ],
    )
    write_csv(
        output_dir / "scale_consistency_10m_50m.csv",
        scale_rows,
        [
            "evidence_type",
            "method",
            "release_window",
            "release_frac_pct",
            "val_loss_10m_mean",
            "delta_10m_vs_newton",
            "gain_preserved_10m",
            "peak_saving_10m_pct",
            "current_saving_10m_pct",
            "val_loss_50m_mean",
            "delta_50m_vs_newton",
            "gain_preserved_50m",
            "peak_saving_50m_pct",
            "current_saving_50m_pct",
            "scale_note",
        ],
    )
    write_csv(
        output_dir / "auxiliary_negative_controls.csv",
        aux_rows,
        [
            "evidence_type",
            "dataset",
            "model",
            "batch_size",
            "seeds",
            "method",
            "release_window",
            "release_frac_pct",
            "best_val_loss",
            "delta_val_vs_newton",
            "newton_gain_preserved",
            "mechanism_note",
        ],
    )
    write_notes(output_dir / "MECHANISM_NOTES.txt")

    print(f"wrote mechanism analysis to {output_dir}")
    for name in [
        "static_center_window_sweep.csv",
        "same_budget_position_sensitivity.csv",
        "scale_consistency_10m_50m.csv",
        "auxiliary_negative_controls.csv",
        "MECHANISM_NOTES.txt",
    ]:
        print(output_dir / name)


if __name__ == "__main__":
    main()
