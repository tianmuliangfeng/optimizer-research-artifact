"""Run the all-layer temporal P3 probe with FP64 exact-SVD controls."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_quadratic_score_probe_p2 as p2
from project_paths import EXPERIMENT_RESULTS_ROOT
from runner_utils import ensure_data, run_cmd, write_command_record


FAMILY = "06_kstate_spectrum"
OUTPUT_FAMILY = "quadratic_probe_p3_layer_map"
DEFAULT_PREFIX = "mainconf_quadprobe_p3_layermap_fp64svd"


def parse_args():
    parser = p2.build_parser()
    parser.description = (
        "Run the seed2026 P3 all-layer temporal c_proj mechanism probe. "
        "The parameter trajectory remains none; diag/full EMA K and momentum "
        "are shadow-only. All 24 layers are line-searched, while exact SVD uses "
        "FP64 internally and is checked against the predeclared row-orthogonality "
        "gate."
    )
    parser.set_defaults(
        probe_layers=list(range(24)),
        exact_hvp=False,
        exact_svd=True,
        exact_svd_repeats=4,
        svd_compute_dtype="float64",
        run_prefix=DEFAULT_PREFIX,
    )
    args = parser.parse_args()
    args.probe_protocol = "p3"
    return args


def run_name(args) -> str:
    return (
        f"{args.run_prefix}_none_shadowdiagfull_L{args.n_layer}_D{args.n_embd}_"
        f"step{max(args.probe_steps)}_seed{args.seed}"
    )


def replace_option(cmd: list[str], name: str, value: str) -> None:
    prefix = f"--{name}="
    matches = [index for index, part in enumerate(cmd) if part.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {prefix} option, found {matches}")
    cmd[matches[0]] = f"{prefix}{value}"


def build_command(args) -> list[str]:
    cmd = p2.build_command(args)
    output_dir = (
        EXPERIMENT_RESULTS_ROOT
        / FAMILY
        / OUTPUT_FAMILY
        / run_name(args)
    )
    replace_option(
        cmd,
        "cproj_quadratic_probe_output_dir",
        p2.string_override(str(output_dir)),
    )
    return cmd


def validate_p3(args, cmd: list[str]) -> None:
    p2.validate_args(args)
    p2.validate_command(args, cmd)
    if args.svd_compute_dtype != "float64":
        raise ValueError(
            "P3's repaired exact-SVD arm requires "
            "--svd-compute-dtype float64"
        )
    if args.exact_hvp:
        raise ValueError(
            "P3 layer mapping intentionally disables exact HVP; it was not "
            "decision-relevant in P2 and would multiply all-layer probe cost"
        )
    expected_output = (
        EXPERIMENT_RESULTS_ROOT
        / FAMILY
        / OUTPUT_FAMILY
        / run_name(args)
    )
    actual_output = p2.option_value(cmd, "cproj_quadratic_probe_output_dir")
    if actual_output != p2.string_override(str(expected_output)):
        raise RuntimeError(
            f"P3 output={actual_output}, expected "
            f"{p2.string_override(str(expected_output))}"
        )


def print_expected_outputs(args) -> None:
    ns5_candidates = 3 * len(args.probe_modes)
    svd_candidates = len(args.probe_modes)
    directions_per_step = len(args.probe_layers) * (
        args.build_repeats * ns5_candidates
        + args.exact_svd_repeats * svd_candidates
    )
    direction_rows = len(args.probe_steps) * directions_per_step
    line_rows = (
        direction_rows
        * (1 + args.heldout_batches)
        * len(args.line_search_multipliers)
        if args.line_search
        else 0
    )
    shadow_state_gib = len(args.probe_layers) * 224.03125 / 1024
    print(
        "Expected P3 output: "
        f"{direction_rows} direction rows and {line_rows} line-search rows."
    )
    print(
        f"Layers={args.probe_layers}; FP64 SVD builds="
        f"{args.exact_svd_repeats}/{args.build_repeats}; exact HVP disabled."
    )
    print(
        "Estimated persistent diagnostic shadow state from P2 scaling: "
        f"about {shadow_state_gib:.2f} GiB. It is probe-only and does not "
        "change the none parameter trajectory."
    )
    print(
        "Primary decision: layer-wise heldout ema_momentum diag/full minus "
        "matched none at eta=0.01 and 0.02. SVD results are accepted only if "
        "every FP64 direction passes row-orthogonality residual <=1e-4."
    )


def main() -> None:
    args = parse_args()
    if not ensure_data(args.dataset, args.dry_run):
        raise SystemExit(1)
    cmd = build_command(args)
    validate_p3(args, cmd)
    print_expected_outputs(args)
    write_command_record(
        family=FAMILY,
        run_prefix=args.run_prefix,
        commands=[cmd],
        dry_run=args.dry_run,
        enabled=args.write_commands,
    )
    run_cmd(cmd, args.dry_run)


if __name__ == "__main__":
    main()
