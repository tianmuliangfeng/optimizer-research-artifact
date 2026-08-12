#!/usr/bin/env python3
"""Run one R1 shared-recipe-LR-multiplier cell via the audited R1 controller."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2026-07-29.3"
FAMILY = "39_r1_shared_lr_sensitivity"
SMOKE_PROTOCOL = "r1_shared_recipe_lr_multiplier_exact_shape_smoke_v1"
FORMAL_PROTOCOL = "r1_shared_recipe_lr_multiplier_supporting_v1"


def load_r1_controller(repo: Path) -> Any:
    path = repo / "scripts/15_official_newton_muon_r1/run_official_newton_muon_r1.py"
    spec = importlib.util.spec_from_file_location("r1_formal_controller", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import R1 controller: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--lr-multiplier", type=float, required=True)
    parser.add_argument("--budget-steps", type=int, default=3000)
    parser.add_argument("--warmdown-steps", type=int, default=871)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=34)
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--resume-batch", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    args = parser.parse_args()
    allowed = {"diag", "none", "block4", "muon"}
    if not args.methods or set(args.methods) - allowed:
        parser.error(f"--methods must be a non-empty subset of {sorted(allowed)}")
    if len(args.methods) != len(set(args.methods)):
        parser.error("--methods contains duplicates")
    if args.lr_multiplier not in {0.8, 1.0, 1.2}:
        parser.error("--lr-multiplier must be one of 0.8, 1.0, 1.2")
    if args.budget_steps < 100 or args.budget_steps % 100:
        parser.error("--budget-steps must be a positive multiple of 100")
    if args.warmdown_steps < 1 or args.warmdown_steps >= args.budget_steps:
        parser.error("--warmdown-steps must be within the training budget")
    if not args.smoke and args.smoke_manifest is None:
        parser.error("formal sensitivity requires --smoke-manifest")
    return args


def scaled_specs(module: Any, multiplier: float) -> dict[str, Any]:
    return {
        name: replace(
            value,
            base_learning_rate=value.base_learning_rate * multiplier,
            matrix_learning_rate=value.matrix_learning_rate * multiplier,
            role=f"{value.role}_recipe_lr_multiplier_{multiplier:g}",
        )
        for name, value in module.METHODS.items()
    }


def derive_sensitivity_source(
    module: Any,
    repo: Path,
    derived: Any,
    *,
    multiplier: float,
    budget_steps: int,
    warmdown_steps: int,
) -> Any:
    source = derived.source
    replacements = [
        (
            "    num_iterations : int = 6200",
            f"    num_iterations : int = {budget_steps}",
            "training budget",
        ),
        (
            "    warmdown_iters : int = 1800",
            f"    warmdown_iters : int = {warmdown_steps}",
            "warmdown budget",
        ),
    ]
    base_lr = 0.0036 if derived.method == "muon" else 0.0040
    replacements.append(
        (
            f"    learning_rate : float = {base_lr:.4f}",
            f"    learning_rate : float = {base_lr * multiplier:.8f}",
            "recipe LR multiplier",
        )
    )
    for old, new, label in replacements:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"{derived.method}: expected one {label} anchor {old!r}, found {count}"
            )
        source = source.replace(old, new, 1)
    compile(source, f"<R1-shared-LR-{derived.method}-{multiplier:g}>", "exec")
    base_source = (
        (repo / derived.base_script)
        .read_bytes()
        .replace(b"\r\n", b"\n")
        .decode("utf-8")
    )
    diff = "".join(
        difflib.unified_diff(
            base_source.splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile=f"official/{derived.base_script}",
            tofile=(
                f"r1_shared_lr_m{str(multiplier).replace('.', 'p')}/"
                f"train_r1_{derived.method}.py"
            ),
        )
    )
    return module.DerivedSource(
        method=derived.method,
        base_script=derived.base_script,
        base_canonical_sha256=derived.base_canonical_sha256,
        derived_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source=source,
        unified_diff=diff,
    )


def configure_controller(module: Any, args: argparse.Namespace) -> None:
    specs = scaled_specs(module, args.lr_multiplier)
    original_build_all_sources = module.build_all_sources

    def build_sensitivity_sources(repo: Path, *, lr_cross: bool = False):
        if lr_cross:
            raise RuntimeError("LR-cross mode is forbidden inside the shared grid")
        built = original_build_all_sources(repo, lr_cross=False)
        return {
            name: derive_sensitivity_source(
                module,
                repo,
                value,
                multiplier=args.lr_multiplier,
                budget_steps=args.budget_steps,
                warmdown_steps=args.warmdown_steps,
            )
            for name, value in built.items()
        }

    module.METHODS = specs
    module.build_all_sources = build_sensitivity_sources
    module.FAMILY = FAMILY
    module.DEFAULT_PROJECT = args.wandb_project
    module.DEFAULT_RUN_PREFIX = (
        f"mainconf_r1_lr_sensitivity_m{str(args.lr_multiplier).replace('.', 'p')}"
    )
    module.R1_SMOKE_PROTOCOL = SMOKE_PROTOCOL
    module.R1_FORMAL_PROTOCOL = FORMAL_PROTOCOL
    module.FORMAL_PROFILE = module.RunProfile(
        name="shared_recipe_lr_sensitivity_supporting",
        total_steps=args.budget_steps,
        validation_steps=tuple(range(0, args.budget_steps + 1, 100)),
        formal_evidence=False,
        require_checkpoint=True,
    )
    module.lr_multiplier = lambda step, total_steps: (
        1.0
        if step < total_steps - args.warmdown_steps
        else max(0.0, (total_steps - step) / args.warmdown_steps)
    )
    module.experiment_specs = lambda _args: specs
    module.experiment_family = lambda _args: FAMILY
    module.experiment_protocol = lambda _args, smoke=None: (
        SMOKE_PROTOCOL
        if (_args.numerical_smoke if smoke is None else smoke)
        else FORMAL_PROTOCOL
    )
    module.evidence_eligibility = lambda _args: {
        "quality_usable": True,
        "memory_usable": False,
        "timing_usable": False,
        "evidence_class": "supporting_only",
        "reason": (
            "single-seed final-recipe shared multiplier grid; concurrent two-GPU "
            "execution excludes timing and tuned-best claims"
        ),
    }
    controller_args = argparse.Namespace(
        official_repo=args.official_repo,
        python_exe=args.python_exe,
        seed=args.seed,
        lr_cross=False,
        host_bridge=False,
        concurrent_node_training=True,
        concurrent_workload="paired R1 LR-sensitivity lane on sibling H100",
        methods=list(args.methods),
        dry_run=False,
        preflight=args.preflight,
        numerical_smoke=args.smoke,
        smoke_steps=args.smoke_steps,
        smoke_manifest=args.smoke_manifest,
        results_dir=args.results_dir,
        resume_batch=args.resume_batch,
        run_prefix=module.DEFAULT_RUN_PREFIX,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_train_log_every=20,
        wandb_init_timeout=120,
        continue_on_error=False,
    )
    module.parse_args = lambda: controller_args


def main() -> None:
    args = parse_args()
    module = load_r1_controller(args.repo.resolve())
    configure_controller(module, args)
    module.main()


if __name__ == "__main__":
    main()
