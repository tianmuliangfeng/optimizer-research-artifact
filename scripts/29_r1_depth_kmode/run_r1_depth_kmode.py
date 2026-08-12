#!/usr/bin/env python3
"""Run one audited seed/method shard of the official-R1 depth experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
R1_DIR = SCRIPT_DIR.parent / "15_official_newton_muon_r1"
sys.path.insert(0, str(R1_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import run_official_newton_muon_r1 as r1
import r1_depth_source_builder as builder


FAMILY = "29_r1_depth_kmode"
SMOKE_PROTOCOL = "official_newton_muon_1_r1_depth_kmode_exact_shape_smoke_v1"
FORMAL_PROTOCOL = "official_newton_muon_1_r1_depth_kmode_three_seed_v1"
DEFAULT_PROJECT = "Selective-Newton-Muon-MainConf-R1-Depth-KMode-20260725"
DEFAULT_RUN_PREFIX = "mainconf_r1_depth_kmode"


def _spec(method: str) -> r1.MethodSpec:
    if method == "muon":
        return r1.MethodSpec(
            name="muon",
            base_script="train_gpt_muon_1.py",
            cproj_k_mode="muon",
            base_learning_rate=0.0036,
            matrix_learning_rate=0.00036,
            role="matched_official_muon_anchor",
        )
    if method == "block4":
        return r1.MethodSpec(
            name="block4",
            base_script="train_gpt_newton_muon_1.py",
            cproj_k_mode="block4",
            base_learning_rate=0.0040,
            matrix_learning_rate=0.00040,
            role="matched_official_block4_anchor",
        )
    rule, mode, layers = builder.method_contract(method)
    return r1.MethodSpec(
        name=method,
        base_script="train_gpt_newton_muon_1.py",
        cproj_k_mode=mode,
        base_learning_rate=0.0040,
        matrix_learning_rate=0.00040,
        role=(
            f"depth_rule={rule};selected_layers={','.join(map(str, layers))};"
            f"selected_mode={mode};unselected_mode=block4"
        ),
    )


METHODS = {method: _spec(method) for method in builder.ALLOWED_METHODS}


def experiment_family(_args: argparse.Namespace) -> str:
    return FAMILY


def experiment_protocol(
    args: argparse.Namespace, *, smoke: bool | None = None
) -> str:
    is_smoke = args.numerical_smoke if smoke is None else smoke
    return SMOKE_PROTOCOL if is_smoke else FORMAL_PROTOCOL


def experiment_specs(_args: argparse.Namespace) -> dict[str, r1.MethodSpec]:
    return METHODS


def evidence_eligibility(_args: argparse.Namespace) -> dict[str, object]:
    return {
        "quality_usable": True,
        "memory_usable": True,
        "timing_usable": False,
        "reason": (
            "R1 depth is a quality/mechanism experiment; multiple physical GPUs "
            "may train concurrently on the same node"
        ),
    }


def build_all_sources(
    repo: Path, *, lr_cross: bool = False
) -> dict[str, r1.DerivedSource]:
    if lr_cross:
        raise RuntimeError("R1 depth does not support LR-cross mode")
    builder.self_test_contract()
    r1.self_test_diag_math()
    built = {
        method: builder.build_source(repo, method)
        for method in builder.ALLOWED_METHODS
    }
    treatment_hashes = {
        built[method].derived_sha256 for method in builder.DEPTH_METHODS
    }
    if len(treatment_hashes) != len(builder.DEPTH_METHODS):
        raise RuntimeError("R1 depth treatment sources are not all distinct")
    return built


def controlled_env(
    args: argparse.Namespace,
    spec: r1.MethodSpec,
    data_dir: Path,
    *,
    init_only: bool = False,
    smoke_test: bool = False,
    smoke_steps: int = 10,
) -> dict[str, str]:
    env = _ORIGINAL_CONTROLLED_ENV(
        args,
        spec,
        data_dir,
        init_only=init_only,
        smoke_test=smoke_test,
        smoke_steps=smoke_steps,
    )
    if spec.name in builder.DEPTH_METHODS:
        rule, mode, layers = builder.method_contract(spec.name)
        env["R1_DEPTH_RULE"] = rule
        env["R1_CPROJ_K_MODE"] = mode
        env["R1_CPROJ_K_LAYERS"] = ",".join(map(str, layers))
    else:
        env["R1_DEPTH_RULE"] = "anchor"
        env["R1_CPROJ_K_LAYERS"] = ",".join(map(str, range(12)))
    # Checkpoints are ~10 GiB each and are not an estimand for this mechanism
    # family. Metrics, source, manifests, stdout, and W&B history remain saved.
    env["R1_DISABLE_CHECKPOINT"] = "1"
    return env


def parse_args() -> argparse.Namespace:
    args = _ORIGINAL_PARSE_ARGS()
    if args.host_bridge or args.lr_cross:
        raise SystemExit("R1 depth does not support --host-bridge or --lr-cross")
    if args.numerical_smoke and args.smoke_steps < 34:
        raise SystemExit(
            "R1 depth numerical smoke requires --smoke-steps >= 34 "
            "to cross the first K refresh at step 32"
        )
    return args


def visible_device_record(args: argparse.Namespace) -> dict[str, object]:
    record = _ORIGINAL_VISIBLE_DEVICE_RECORD(args)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if not args.dry_run and len(devices) != 1:
        raise RuntimeError(
            "R1 depth requires exactly one visible physical GPU per controller; "
            f"observed CUDA_VISIBLE_DEVICES={visible!r}"
        )
    return record


def write_json_with_contract(path: Path, payload: dict[str, object]) -> None:
    enriched = dict(payload)
    enriched["r1_depth_contract"] = {
        "rules": {name: list(layers) for name, layers in builder.RULE_LAYERS.items()},
        "selected_modes": list(builder.SELECTED_MODES),
        "unselected_cproj_mode": "block4",
        "anchors": list(builder.ANCHORS),
        "seeds": [2024, 2025, 2026],
        "primary_endpoint": "validation loss at step 6200",
        "primary_contrast": "within-seed, within-rule diag minus none",
        "secondary_endpoints": [
            "tail-5 validation mean",
            "normalized validation AUC",
            "K-state bytes",
            "optimizer-state bytes",
            "peak allocated memory",
        ],
        "checkpoint_policy": "disabled; approximately 10 GiB per run avoided",
        "timing_usable": False,
    }
    method = enriched.get("method")
    if isinstance(method, str) and method in builder.DEPTH_METHODS:
        rule, mode, layers = builder.method_contract(method)
        enriched["depth_rule"] = rule
        enriched["selected_cproj_k_mode"] = mode
        enriched["selected_cproj_layers"] = list(layers)
        enriched["unselected_cproj_k_mode"] = "block4"
    _ORIGINAL_WRITE_JSON(path, enriched)


def _inject_defaults() -> None:
    argv = sys.argv[1:]
    if "--methods" not in argv:
        sys.argv.extend(["--methods", *builder.ALLOWED_METHODS])
    if "--results-dir" not in argv:
        sys.argv.extend(
            [
                "--results-dir",
                str(r1.EXPERIMENT_RESULTS_ROOT / FAMILY / "results"),
            ]
        )
    if "--run-prefix" not in argv:
        sys.argv.extend(["--run-prefix", DEFAULT_RUN_PREFIX])
    if "--wandb-project" not in argv:
        sys.argv.extend(["--wandb-project", DEFAULT_PROJECT])


def install_overlay() -> None:
    global _ORIGINAL_CONTROLLED_ENV
    global _ORIGINAL_PARSE_ARGS
    global _ORIGINAL_VISIBLE_DEVICE_RECORD
    global _ORIGINAL_WRITE_JSON
    _ORIGINAL_CONTROLLED_ENV = r1.controlled_env
    _ORIGINAL_PARSE_ARGS = r1.parse_args
    _ORIGINAL_VISIBLE_DEVICE_RECORD = r1.visible_device_record
    _ORIGINAL_WRITE_JSON = r1.write_json
    r1.METHODS = METHODS
    r1.experiment_family = experiment_family
    r1.experiment_protocol = experiment_protocol
    r1.experiment_specs = experiment_specs
    r1.evidence_eligibility = evidence_eligibility
    r1.build_all_sources = build_all_sources
    r1.controlled_env = controlled_env
    r1.parse_args = parse_args
    r1.visible_device_record = visible_device_record
    r1.write_json = write_json_with_contract
    r1.FORMAL_PROFILE = r1.RunProfile(
        name="formal_no_checkpoint",
        total_steps=r1.FULL_NUM_ITERATIONS,
        validation_steps=tuple(range(0, r1.FULL_NUM_ITERATIONS + 1, 100)),
        formal_evidence=True,
        require_checkpoint=False,
    )


def main() -> None:
    install_overlay()
    _inject_defaults()
    r1.main()


if __name__ == "__main__":
    main()
