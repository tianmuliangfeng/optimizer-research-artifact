"""Run the R1 dense-full c_proj alpha pilot or frozen confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
R1_DIR = SCRIPT_DIR.parent / "15_official_newton_muon_r1"
for directory in (R1_DIR, SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_official_newton_muon_r1 as r1  # noqa: E402
from r1_dense_full_alpha_source_builder import (  # noqa: E402
    ALLOWED_METHODS,
    ALPHA_BY_METHOD,
    build_source,
    self_test_alpha_math,
)


FAMILY = "24_r1_dense_full_alpha"
SMOKE_PROTOCOL = "official_newton_muon_1_r1_dense_full_alpha_exact_shape_smoke_v1"
FORMAL_PROTOCOL = "official_newton_muon_1_r1_dense_full_alpha_seed2026_pilot_v1"
CONFIRMATORY_PROTOCOL = (
    "official_newton_muon_1_r1_dense_full_alpha_seeds2024_2025_confirmatory_v1"
)
DEFAULT_PROJECT = "Selective-Newton-Muon-MainConf-R1-DenseFullAlpha-20260723"
DEFAULT_RUN_PREFIX = "mainconf_r1_dense_full_alpha"

METHODS = {
    method: r1.MethodSpec(
        name=method,
        base_script="train_gpt_newton_muon_1.py",
        cproj_k_mode="dense_full",
        base_learning_rate=0.0040,
        matrix_learning_rate=0.00040,
        role=f"dense_full_offdiag_alpha_{alpha:g}",
    )
    for method, alpha in ALPHA_BY_METHOD.items()
}

K_DIAG_RE = re.compile(
    r"^R1_FULL_ALPHA_K step=(?P<step>\d+) alpha=(?P<alpha>\S+) "
    r"raw_cross_to_within=(?P<raw_cross_to_within>\S+) "
    r"scaled_offdiag_to_diag=(?P<scaled_offdiag_to_diag>\S+) "
    r"chol_diag_spread=(?P<chol_diag_spread>\S+) "
    r"inv_offdiag_to_diag=(?P<inv_offdiag_to_diag>\S+) "
    r"inv_diag_rms=(?P<inv_diag_rms>\S+) "
    r"cholesky_failures=(?P<cholesky_failures>\d+)$"
)
UPDATE_DIAG_RE = re.compile(
    r"^R1_FULL_ALPHA_UPDATE step=(?P<step>\d+) alpha=(?P<alpha>\S+) "
    r"norm_ratio_vs_diag=(?P<norm_ratio_vs_diag>\S+) "
    r"cosine_vs_diag=(?P<cosine_vs_diag>\S+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "R1 dense-full c_proj alpha sweep at alpha=0,0.25,0.50,0.75,1. "
            "All cells retain the complete 3072x3072 covariance path."
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
            "Run the separately frozen seed-2024/2025 confirmation. This does "
            "not imply that the seed-2026 automatic expansion gate passed."
        ),
    )
    parser.add_argument(
        "--allow-seed-expansion",
        action="store_true",
        help="Legacy flag retained for CLI compatibility; confirmation still requires --confirmatory.",
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
    args.lr_cross = False
    args.host_bridge = False
    if args.results_dir is None:
        args.results_dir = r1.EXPERIMENT_RESULTS_ROOT / FAMILY / "results"
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.confirmatory and args.seed not in (2024, 2025):
        parser.error("--confirmatory supports only seeds 2024 and 2025")
    if args.seed != 2026 and not args.confirmatory:
        parser.error(
            "non-pilot seeds require the separately frozen --confirmatory protocol"
        )
    if args.confirmatory and args.methods != list(ALLOWED_METHODS):
        parser.error("--confirmatory requires the complete five-alpha method grid")
    if len(args.methods) != len(set(args.methods)):
        parser.error("--methods contains duplicates")
    if args.concurrent_workload and not args.concurrent_node_training:
        parser.error("--concurrent-workload requires --concurrent-node-training")
    if args.preflight and args.numerical_smoke:
        parser.error("choose either --preflight or --numerical-smoke")
    if args.numerical_smoke and args.smoke_steps < 34:
        parser.error("dense-full alpha smoke needs at least 34 steps to cross the first inverse refresh")
    formal_start = not (args.dry_run or args.preflight or args.numerical_smoke or args.resume_batch is not None)
    if formal_start and args.smoke_manifest is None:
        parser.error("formal dense-full alpha pilot requires --smoke-manifest")
    if args.resume_batch is not None and (args.dry_run or args.preflight or args.numerical_smoke):
        parser.error("--resume-batch cannot be combined with dry-run, preflight, or smoke")
    if args.wandb_train_log_every <= 0 or args.wandb_init_timeout <= 0:
        parser.error("W&B logging interval and init timeout must be positive")
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
        "reason": "dense-full alpha is a mechanism experiment; diagnostics and node concurrency invalidate timing inference",
    }


def visible_device_record(args: argparse.Namespace) -> dict[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if not args.dry_run and len(devices) != 1:
        raise RuntimeError(
            "R1 dense-full alpha requires exactly one visible physical GPU. "
            "Set CUDA_VISIBLE_DEVICES to the intended single GPU before launch."
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
        raise RuntimeError("dense-full alpha does not support the R1 LR-cross mode")
    self_test_alpha_math()
    built = {method: build_source(repo, method) for method in ALLOWED_METHODS}
    for method, derived in built.items():
        expected = r1.r0.EXPECTED_CANONICAL_SHA256[derived.base_script]
        if derived.base_canonical_sha256 != expected:
            raise RuntimeError(f"{method} official base hash mismatch: {derived.base_canonical_sha256} != {expected}")
    if len({item.derived_sha256 for item in built.values()}) != len(built):
        raise RuntimeError("every dense-full alpha cell must produce a distinct audited source")
    return built


_base_write_json = r1.write_json
_base_parse_metrics = r1.parse_metrics


def write_json_with_alpha_contract(path: Path, payload: dict[str, object]) -> None:
    enriched = dict(payload)
    method = enriched.get("method")
    if isinstance(method, str) and method in ALPHA_BY_METHOD:
        enriched["dense_full_alpha"] = ALPHA_BY_METHOD[method]
        enriched["dense_full_alpha_storage"] = "full_3072x3072_covariance_and_inverse"
    if path.name in {"r1_plan.json", "r1_manifest.json"}:
        confirmatory = (
            enriched.get("protocol") == CONFIRMATORY_PROTOCOL
            or (
                enriched.get("protocol") == SMOKE_PROTOCOL
                and enriched.get("seed") in (2024, 2025)
            )
        )
        enriched["dense_full_alpha_design"] = {
            "definition": "K_alpha = diag(K_full) + alpha * (K_full - diag(K_full))",
            "method_to_alpha": ALPHA_BY_METHOD,
            "topology": "within-block and cross-block covariance are restored together",
            "matched_block_curve": "22_r1_block_alpha restores within-block covariance only",
            "primary_endpoint": "validation loss at step 6200",
            "secondary_endpoints": ["tail-five validation mean", "normalized validation AUC"],
            "diagnostics": [
                "raw cross-block/within-block covariance norm ratio",
                "scaled offdiagonal/diagonal covariance norm ratio",
                "Cholesky diagonal spread",
                "inverse offdiagonal/diagonal norm ratio",
                "inverse diagonal RMS",
                "preconditioned update norm ratio and cosine versus dense diagonal reference",
            ],
            "seed_policy": (
                {
                    "study": "separately frozen contradiction-resolution confirmation",
                    "confirmatory_seeds": [2024, 2025],
                    "seed2026_role": "previously completed exploratory pilot",
                    "old_automatic_expansion_gate_passed": False,
                    "contract": "CONFIRMATORY_CONTRACT_20260727.md",
                }
                if confirmatory
                else "seed2026 pilot; no automatic seed expansion"
            ),
        }
        enriched["interpretation_boundary"] = {
            "primary_estimand": "effect of complete c_proj off-diagonal covariance strength at R1 shape",
            "topology_estimand": "full-alpha curve versus the completed block4-alpha curve at matched alpha",
            "controlled": "same seed/data/init/LR/EMA/ridge/refresh and dense-full storage; only alpha differs",
            "timing": "ineligible because dense diagnostics add synchronization and the node may be concurrent",
            "study_role": (
                "seed-2024/2025 confirmation; seed2026 remains exploratory"
                if confirmatory
                else "single-seed mechanism pilot"
            ),
            "alpha_optimum_claim": "no universal optimum claim",
        }
    _base_write_json(path, enriched)


def _parse_diagnostics(stdout_path: Path, method: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = K_DIAG_RE.match(line)
        if match:
            item: dict[str, object] = {"method": method, "diagnostic": "K_and_inverse"}
            for key, value in match.groupdict().items():
                item[key] = int(value) if key in {"step", "cholesky_failures"} else float(value)
            rows.append(item)
            continue
        match = UPDATE_DIAG_RE.match(line)
        if match:
            item = {"method": method, "diagnostic": "preconditioned_update"}
            for key, value in match.groupdict().items():
                item[key] = int(value) if key == "step" else float(value)
            rows.append(item)
    return rows


def parse_metrics_with_diagnostics(*args, **kwargs):
    rows, summary = _base_parse_metrics(*args, **kwargs)
    stdout_path = Path(args[1] if len(args) > 1 else kwargs["stdout_path"])
    spec = args[2] if len(args) > 2 else kwargs["spec"]
    diagnostics = _parse_diagnostics(stdout_path, spec.name)
    output_path = stdout_path.parent / "dense_full_alpha_diagnostics.csv"
    fields = [
        "method", "diagnostic", "step", "alpha", "raw_cross_to_within",
        "scaled_offdiag_to_diag", "chol_diag_spread", "inv_offdiag_to_diag",
        "inv_diag_rms", "cholesky_failures", "norm_ratio_vs_diag", "cosine_vs_diag",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in diagnostics:
            writer.writerow({field: item.get(field, "") for field in fields})
    expected_steps = {31} if not getattr(args[4] if len(args) > 4 else kwargs["profile"], "formal_evidence") else {31, 1023, 2047, 3071, 4095, 5119, 6143}
    k_steps = {int(item["step"]) for item in diagnostics if item["diagnostic"] == "K_and_inverse"}
    update_steps = {int(item["step"]) for item in diagnostics if item["diagnostic"] == "preconditioned_update"}
    if not expected_steps.issubset(k_steps) or not expected_steps.issubset(update_steps):
        raise RuntimeError(
            f"dense-full alpha diagnostics incomplete for {spec.name}: K={sorted(k_steps)}, update={sorted(update_steps)}"
        )
    if any(int(item.get("cholesky_failures", 0)) != 0 for item in diagnostics):
        raise RuntimeError(f"dense-full alpha Cholesky failure observed for {spec.name}")
    summary["dense_full_alpha"] = ALPHA_BY_METHOD[spec.name]
    summary["mechanism_diagnostic_rows"] = len(diagnostics)
    summary["mechanism_diagnostics_path"] = str(output_path.resolve())
    final_update = max(
        (item for item in diagnostics if item["diagnostic"] == "preconditioned_update"),
        key=lambda item: int(item["step"]),
    )
    summary["final_update_norm_ratio_vs_diag"] = float(final_update["norm_ratio_vs_diag"])
    summary["final_update_cosine_vs_diag"] = float(final_update["cosine_vs_diag"])
    return rows, summary


def install_experiment_overlay() -> None:
    r1.parse_args = parse_args
    r1.experiment_family = experiment_family
    r1.experiment_protocol = experiment_protocol
    r1.experiment_specs = experiment_specs
    r1.evidence_eligibility = evidence_eligibility
    r1.visible_device_record = visible_device_record
    r1.build_all_sources = build_all_sources
    r1.write_json = write_json_with_alpha_contract
    r1.parse_metrics = parse_metrics_with_diagnostics


def main() -> None:
    install_experiment_overlay()
    r1.main()


if __name__ == "__main__":
    main()
