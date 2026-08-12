"""Run the R1-native dense block-alpha mechanism experiment.

This entry point composes the audited R1 controller rather than changing it.
The experiment gets its own family/protocol, source hashes, smoke certificate,
W&B group, and resumable batch manifests.
"""

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
from r1_block_alpha_source_builder import (
    ALLOWED_METHODS,
    ALPHA_BY_METHOD,
    build_source,
    self_test_alpha_math,
)


FAMILY = "22_r1_block_alpha"
SMOKE_PROTOCOL = "official_newton_muon_1_r1_dense_block_alpha_exact_shape_smoke_v1"
FORMAL_PROTOCOL = "official_newton_muon_1_r1_dense_block_alpha_seed2026_pilot_v1"
CONFIRMATORY_PROTOCOL = (
    "official_newton_muon_1_r1_dense_block_alpha_seeds2024_2025_confirmatory_v1"
)
DEFAULT_PROJECT = "Selective-Newton-Muon-MainConf-R1-BlockAlpha-20260722"
DEFAULT_RUN_PREFIX = "mainconf_r1_block_alpha"
CONFIRMATORY_PROJECT = (
    "Selective-Newton-Muon-MainConf-R1-BlockAlpha-Confirmatory-20260724"
)
CONFIRMATORY_RUN_PREFIX = "mainconf_r1_block_alpha_confirmatory"

METHODS = {
    method: r1.MethodSpec(
        name=method,
        base_script="train_gpt_newton_muon_1.py",
        cproj_k_mode="block4",
        base_learning_rate=0.0040,
        matrix_learning_rate=0.00040,
        role=f"dense_block4_offdiag_alpha_{alpha:g}",
    )
    for method, alpha in ALPHA_BY_METHOD.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "R1-native block-local alpha sweep: dense alpha=0, 0.25, 0.50, 0.75. "
            "Existing matched diag/block4 runs supply analysis endpoints."
        )
    )
    parser.add_argument("--official-repo", type=Path, default=r1.r0.default_official_repo())
    parser.add_argument("--python-exe", default="python")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--methods", nargs="+", choices=ALLOWED_METHODS, default=list(ALLOWED_METHODS))
    parser.add_argument(
        "--confirmatory",
        action="store_true",
        help=(
            "Run the separately preregistered seed-2024/2025 confirmation. "
            "It requires all four dense alpha cells."
        ),
    )
    parser.add_argument("--concurrent-node-training", action="store_true")
    parser.add_argument("--concurrent-workload", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--numerical-smoke", action="store_true")
    parser.add_argument("--smoke-test", dest="numerical_smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-steps", type=int, default=34)
    parser.add_argument("--smoke-manifest", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--resume-batch", type=Path, default=None)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-train-log-every", type=int, default=20)
    parser.add_argument("--wandb-init-timeout", type=int, default=120)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    # Compatibility fields consumed by the shared R1 controller.
    args.lr_cross = False
    args.host_bridge = False
    if args.results_dir is None:
        args.results_dir = r1.EXPERIMENT_RESULTS_ROOT / FAMILY / "results"
    if args.confirmatory:
        if args.seed not in (2024, 2025):
            parser.error("--confirmatory requires --seed 2024 or --seed 2025")
        if args.methods != list(ALLOWED_METHODS):
            parser.error(
                "--confirmatory requires the complete ordered method set: "
                + " ".join(ALLOWED_METHODS)
            )
        if args.wandb_project == DEFAULT_PROJECT:
            args.wandb_project = CONFIRMATORY_PROJECT
        if args.run_prefix == DEFAULT_RUN_PREFIX:
            args.run_prefix = CONFIRMATORY_RUN_PREFIX
    elif args.seed != 2026:
        parser.error(
            "non-2026 block-alpha runs require --confirmatory so they cannot be "
            "mislabelled as an expansion of the stopped seed-2026 pilot"
        )
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if len(args.methods) != len(set(args.methods)):
        parser.error("--methods contains duplicates")
    if args.concurrent_workload and not args.concurrent_node_training:
        parser.error("--concurrent-workload requires --concurrent-node-training")
    if args.preflight and args.numerical_smoke:
        parser.error("choose either --preflight or --numerical-smoke")
    if args.numerical_smoke and args.smoke_steps < 34:
        parser.error("block-alpha smoke needs at least 34 steps to cross the step-32 K refresh")
    if args.wandb_train_log_every <= 0 or args.wandb_init_timeout <= 0:
        parser.error("W&B logging interval and init timeout must be positive")
    if not (args.dry_run or args.preflight or args.numerical_smoke or args.resume_batch is not None) and args.smoke_manifest is None:
        parser.error("formal block-alpha pilot requires --smoke-manifest")
    if args.resume_batch is not None and (args.dry_run or args.preflight or args.numerical_smoke):
        parser.error("--resume-batch cannot be combined with dry-run, preflight, or smoke")
    return args


def experiment_family(_args: argparse.Namespace) -> str:
    return FAMILY


def experiment_protocol(args: argparse.Namespace, *, smoke: bool | None = None) -> str:
    is_smoke = args.numerical_smoke if smoke is None else smoke
    if is_smoke:
        return SMOKE_PROTOCOL
    return CONFIRMATORY_PROTOCOL if args.confirmatory else FORMAL_PROTOCOL


def experiment_specs(_args: argparse.Namespace) -> dict[str, r1.MethodSpec]:
    return METHODS


def evidence_eligibility(_args: argparse.Namespace) -> dict[str, object]:
    return {
        "quality_usable": True,
        "memory_usable": True,
        "timing_usable": False,
        "reason": "R1 block-alpha is a quality/mechanism experiment; timing belongs to R1-PERF",
    }


def visible_device_record(args: argparse.Namespace) -> dict[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if not args.dry_run and len(devices) != 1:
        raise RuntimeError(
            "R1 block-alpha requires exactly one visible physical GPU. "
            "Set CUDA_VISIBLE_DEVICES=0 (or the intended single GPU) before launch."
        )
    return {
        "cuda_visible_devices": visible or None,
        "visible_device_count": len(devices) if visible else None,
        "one_process_one_gpu": len(devices) == 1,
        "concurrent_node_training": bool(args.concurrent_node_training),
        "concurrent_workload": args.concurrent_workload,
    }


def build_all_sources(repo: Path, *, lr_cross: bool = False) -> dict[str, r1.DerivedSource]:
    if lr_cross:
        raise RuntimeError("block-alpha does not support the R1 LR-cross mode")
    self_test_alpha_math()
    built = {method: build_source(repo, method) for method in ALLOWED_METHODS}
    for method, derived in built.items():
        expected = r1.r0.EXPECTED_CANONICAL_SHA256[derived.base_script]
        if derived.base_canonical_sha256 != expected:
            raise RuntimeError(f"{method} official base hash mismatch: {derived.base_canonical_sha256} != {expected}")
    if len({item.derived_sha256 for item in built.values()}) != len(built):
        raise RuntimeError("each embedded alpha must produce a distinct audited source")
    return built


_base_write_json = r1.write_json


def write_json_with_alpha_contract(path: Path, payload: dict[str, object]) -> None:
    enriched = dict(payload)
    method = enriched.get("method")
    if isinstance(method, str) and method in ALPHA_BY_METHOD:
        enriched["block_alpha"] = ALPHA_BY_METHOD[method]
        enriched["block_alpha_storage"] = "dense_official_block4"
    if path.name in {"r1_plan.json", "r1_manifest.json"}:
        confirmatory = enriched.get("protocol") == CONFIRMATORY_PROTOCOL
        enriched["block_alpha_design"] = {
            "definition": "for each official dxd c_proj block: diag(K) + alpha * (K - diag(K))",
            "method_to_alpha": ALPHA_BY_METHOD,
            "raw_ema_state": "unchanged dense official block4 covariance",
            "ridge": "unchanged official rule computed after interpolation; diagonal is invariant",
            "alpha0_role": "dense-storage engineering equivalence control against efficient diag",
            "alpha1_role": "reuse matched official R1 block4 endpoint; do not rerun",
            "primary_endpoint": "validation loss at step 6200",
            "secondary_endpoints": ["mean of final five validation points", "normalized validation AUC"],
            **(
                {
                    "confirmatory_estimand": (
                        "paired five-point block-local alpha response, using the "
                        "same-seed official block4 run as the exact alpha=1 endpoint"
                    ),
                    "confirmatory_primary_contrast": (
                        "L(alpha=0.5) - 0.5 * (L(alpha=0) + L(alpha=1))"
                    ),
                    "confirmatory_seeds": [2024, 2025],
                    "seed2026_role": "previously completed exploratory pilot",
                }
                if confirmatory
                else {
                    "pilot_expansion_gate": {
                        "dense_alpha0_abs_final_delta_vs_diag_max": 0.001,
                        "dense_alpha0_abs_tail5_delta_vs_diag_max": 0.001,
                        "final_loss_spearman_rho_min": 0.5,
                        "block4_minus_diag_final_delta_min": 0.0,
                    }
                }
            ),
        }
        enriched["interpretation_boundary"] = {
            "primary_estimand": "effect of block-local off-diagonal covariance strength in official R1 c_proj",
            "controlled": "same seed/data/init/LR/EMA/ridge/refresh schedule; only alpha differs",
            "storage": "all newly run alpha cells retain dense block4 state; no memory-efficiency claim",
            "endpoints": "matched efficient diag and official block4 R1 runs are reused only in analysis",
            "timing": "ineligible; this family is not a performance experiment",
            "evidence_stage": (
                "seeds2024/2025 are a separately preregistered confirmation; "
                "seed2026 remains the exploratory pilot"
                if confirmatory
                else "seed2026 is directional evidence, not a multi-seed confirmatory claim"
            ),
        }
    _base_write_json(path, enriched)


def install_experiment_overlay() -> None:
    r1.parse_args = parse_args
    r1.experiment_family = experiment_family
    r1.experiment_protocol = experiment_protocol
    r1.experiment_specs = experiment_specs
    r1.evidence_eligibility = evidence_eligibility
    r1.visible_device_record = visible_device_record
    r1.build_all_sources = build_all_sources
    r1.write_json = write_json_with_alpha_contract


def main() -> None:
    install_experiment_overlay()
    r1.main()


if __name__ == "__main__":
    main()
