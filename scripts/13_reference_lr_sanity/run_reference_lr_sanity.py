from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from project_paths import EXPERIMENT_RESULTS_ROOT, SOURCE_REPO
from runner_utils import (
    append_common_train_overrides,
    append_model_overrides,
    base_train_cmd,
    command_text,
    ensure_data,
    write_command_record,
)


FAMILY = "13_reference_lr_sanity"
PAPER_BLOCK4_CONFIG = "config/baselines/14_newton_muon_paper_block4.py"
MUON_BLOG_CONFIG = "config/baselines/15_muon_blog_reference.py"
VALID_METHODS = ("muon", "block4", "none", "diag", "full")
REFERENCE_PROJECTS = {
    12: "Selective-Newton-Muon-MainConf-ReferenceSanity-12L-20260717",
    24: "Selective-Newton-Muon-MainConf-ReferenceSanity-24L-20260717",
}


@dataclass(frozen=True)
class Suite:
    name: str
    dataset: str
    base_config: str
    n_layer: int
    n_head: int
    n_embd: int
    batch_size: int
    block_size: int
    gradient_accumulation_steps: int
    max_iters: int
    lr_decay_iters: int
    matrix_lrs: tuple[float, ...]
    reference_lr: float

    @property
    def nominal_tokens(self) -> int:
        return (
            self.max_iters
            * self.batch_size
            * self.block_size
            * self.gradient_accumulation_steps
        )


SUITES = {
    "owt12l_3k": Suite(
        name="owt12l_3k",
        dataset="openwebtext_gpt2_50m",
        base_config="config/train_openwebtext_gpt2_50m_tier3.py",
        n_layer=12,
        n_head=12,
        n_embd=768,
        batch_size=16,
        block_size=512,
        gradient_accumulation_steps=1,
        max_iters=5000,
        # Match the first 3000 steps of the formal 12L/100M run instead of
        # compressing its AdamW decay schedule into this short sanity run.
        lr_decay_iters=12208,
        matrix_lrs=(0.005, 0.01, 0.02),
        reference_lr=0.01,
    ),
    "owt24l_3k": Suite(
        name="owt24l_3k",
        dataset="openwebtext_gpt2_50m",
        base_config="config/train_openwebtext_gpt2_50m_tier4.py",
        n_layer=24,
        n_head=16,
        n_embd=1024,
        batch_size=2,
        block_size=512,
        gradient_accumulation_steps=1,
        max_iters=3000,
        lr_decay_iters=3000,
        matrix_lrs=(0.005, 0.01, 0.02),
        reference_lr=0.01,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reference-aligned short LR sanity matrix and an optional "
            "fast ordering gate for Muon, block4, none, diag, and dense full."
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Run all selected methods for 33 steps on the first selected suite "
            "with W&B disabled, including the first K refresh."
        ),
    )
    parser.add_argument(
        "--quick-ordering-gate",
        action="store_true",
        help=(
            "Use only each suite's formal/reference matrix LR. This gives five "
            "runs per suite instead of the full LR grid."
        ),
    )
    parser.add_argument(
        "--lr-grid-remainder",
        action="store_true",
        help=(
            "Run only non-reference learning rates. Use this after the quick "
            "ordering gate to complete B without repeating any runs."
        ),
    )
    parser.add_argument("--python-exe", default=None)
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=tuple(SUITES),
        default=list(SUITES),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=VALID_METHODS,
        default=list(VALID_METHODS),
    )
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--wandb-project",
        default=None,
        help=(
            "Optional override sending every selected suite to one project. "
            "By default 12L and 24L use separate new projects."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=("online", "offline", "disabled"),
    )
    parser.add_argument(
        "--run-prefix",
        default="mainconf_reference_lr_sanity",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue the queue after an individual run fails.",
    )
    parser.add_argument(
        "--no-write-commands",
        action="store_false",
        dest="write_commands",
    )
    parser.set_defaults(write_commands=True)
    args = parser.parse_args()
    selected_modes = sum(
        int(value)
        for value in (
            args.preflight,
            args.quick_ordering_gate,
            args.lr_grid_remainder,
        )
    )
    if selected_modes > 1:
        parser.error(
            "--preflight, --quick-ordering-gate, and --lr-grid-remainder "
            "are mutually exclusive"
        )
    return args


def project_for_suite(args: argparse.Namespace, suite: Suite) -> str:
    if args.wandb_project:
        return args.wandb_project
    return REFERENCE_PROJECTS[suite.n_layer]


def lr_label(matrix_lr: float) -> str:
    return f"{matrix_lr:g}".replace(".", "p")


def method_label(method: str) -> str:
    return {
        "muon": "muon_blog",
        "block4": "paper_block4",
        "none": "no_cproj_k",
        "diag": "diag_cproj_k",
        "full": "dense_full_cproj_k_control",
    }[method]


def effective_plan(
    args: argparse.Namespace,
) -> list[tuple[Suite, int, float, str]]:
    suites = [SUITES[name] for name in args.suites]
    if args.preflight:
        suite = suites[0]
        seed = args.seeds[0]
        return [
            (suite, seed, suite.reference_lr, method)
            for method in args.methods
        ]

    plan: list[tuple[Suite, int, float, str]] = []
    for suite in suites:
        if args.quick_ordering_gate:
            matrix_lrs = (suite.reference_lr,)
        elif args.lr_grid_remainder:
            matrix_lrs = tuple(
                lr for lr in suite.matrix_lrs if lr != suite.reference_lr
            )
        else:
            matrix_lrs = suite.matrix_lrs
        for seed in args.seeds:
            for matrix_lr in matrix_lrs:
                for method in args.methods:
                    plan.append((suite, seed, matrix_lr, method))
    return plan


def build_command(
    args: argparse.Namespace,
    suite: Suite,
    seed: int,
    matrix_lr: float,
    method: str,
) -> list[str]:
    is_preflight = bool(args.preflight)
    max_iters = 33 if is_preflight else suite.max_iters
    lr_decay_iters = 33 if is_preflight else suite.lr_decay_iters
    eval_iters = 1 if is_preflight else args.eval_iters
    eval_interval = 32 if is_preflight else args.eval_interval
    log_interval = 10 if is_preflight else args.log_interval
    wandb_mode = "disabled" if is_preflight else args.wandb_mode
    plan_tag = "preflight" if is_preflight else (
        "quick_ordering_gate"
        if args.quick_ordering_gate
        else "lr_grid_remainder"
        if args.lr_grid_remainder
        else "lr_grid"
    )
    prefix = f"{args.run_prefix}_preflight" if is_preflight else args.run_prefix
    label = method_label(method)
    run_name = (
        f"{prefix}_{suite.name}_{label}_lr{lr_label(matrix_lr)}_seed{seed}"
    )
    group = (
        f"{args.run_prefix}_{suite.name}_lr{lr_label(matrix_lr)}_seed{seed}"
    )
    tags = ",".join(
        [
            "publication",
            "reference_alignment",
            "reference_lr_sanity",
            plan_tag,
            "newton_muon_paper_structure",
            "cproj_reference_block4",
            suite.dataset,
            f"L{suite.n_layer}",
            f"D{suite.n_embd}",
            f"steps_{suite.max_iters}",
            f"lrdecay_{suite.lr_decay_iters}",
            f"nominal_tokens_{suite.nominal_tokens}",
            f"mulr_{matrix_lr:g}",
        ]
    )

    command_args = argparse.Namespace(
        python_exe=args.python_exe,
        dataset=suite.dataset,
        seed=seed,
        wandb_project=project_for_suite(args, suite),
        wandb_mode=wandb_mode,
        wandb_log_profile="paper",
        # Experiment B intentionally never uploads W&B tables. The compact
        # paper profile keeps only scalar curves/counters needed for the
        # loss, time-to-loss, and memory comparisons.
        wandb_log_tables=False,
        n_layer=suite.n_layer,
        n_head=suite.n_head,
        n_embd=suite.n_embd,
        max_iters=max_iters,
        lr_decay_iters=lr_decay_iters,
        eval_iters=eval_iters,
        eval_interval=eval_interval,
        log_interval=log_interval,
        batch_size=suite.batch_size,
        block_size=suite.block_size,
        gradient_accumulation_steps=suite.gradient_accumulation_steps,
        muon_learning_rate=matrix_lr,
        device=args.device,
        always_save_checkpoint=False,
    )
    cmd = base_train_cmd(
        command_args,
        config=suite.base_config,
        run_name=run_name,
        group=group,
        method=label,
        tags=tags,
    )
    cmd.insert(3, MUON_BLOG_CONFIG if method == "muon" else PAPER_BLOCK4_CONFIG)
    cmd.extend(
        [
            f"--optimizer_type={'muon' if method == 'muon' else 'cproj_k_mode_newton_muon'}",
            "--muon_nesterov=True",
            "--muon_momentum_ema=True",
            "--muon_split_qkv=True",
            "--muon_adjust_lr_for_shape=True",
            "--muon_ns_compute_dtype=bfloat16",
            "--muon_momentum=0.95",
            "--muon_ns_steps=5",
            "--matrix_weight_decay=0.0",
        ]
    )
    if method != "muon":
        cmd.extend(
            [
                f"--cproj_k_mode={method}",
                "--cproj_k_blocks=4",
                "--cproj_k_reference_mode=block4",
                "--input_beta=0.95",
                "--input_ridge=0.2",
                "--input_refresh=32",
                "--input_max_samples=0",
                "--input_first_refresh_step=31",
                "--input_init_scale=0.001",
                "--input_init_inverse_scale=1.0",
                "--diagnostic_interval=0",
            ]
        )
    append_model_overrides(cmd, command_args)
    append_common_train_overrides(cmd, command_args)
    return cmd


def validate_plan(
    args: argparse.Namespace,
    plan: list[tuple[Suite, int, float, str]],
    commands: list[list[str]],
) -> None:
    if not plan:
        raise ValueError("empty reference LR sanity plan")
    if len(plan) != len(commands):
        raise AssertionError("plan/command count mismatch")

    run_names: list[str] = []
    for (suite, seed, matrix_lr, method), cmd in zip(plan, commands):
        expected = {
            "--muon_nesterov=True",
            "--muon_momentum_ema=True",
            "--muon_split_qkv=True",
            "--muon_adjust_lr_for_shape=True",
            "--muon_ns_compute_dtype=bfloat16",
            "--muon_momentum=0.95",
            "--muon_ns_steps=5",
            "--matrix_weight_decay=0.0",
            f"--muon_learning_rate={matrix_lr}",
            "--wandb_log_profile=paper",
            "--wandb_log_tables=False",
        }
        if method == "muon":
            expected.add("--optimizer_type=muon")
        else:
            expected.update(
                {
                    "--optimizer_type=cproj_k_mode_newton_muon",
                    f"--cproj_k_mode={method}",
                    "--cproj_k_blocks=4",
                    "--cproj_k_reference_mode=block4",
                    "--input_beta=0.95",
                    "--input_ridge=0.2",
                    "--input_refresh=32",
                    "--input_max_samples=0",
                    "--input_first_refresh_step=31",
                    "--input_init_scale=0.001",
                    "--input_init_inverse_scale=1.0",
                }
            )
        missing = sorted(expected - set(cmd))
        if missing:
            raise AssertionError(
                f"{suite.name}/lr{matrix_lr:g}/seed{seed}/{method} "
                f"missing overrides: {missing}"
            )
        if any(part == "--wandb_log_tables=True" for part in cmd):
            raise AssertionError("experiment B must not upload W&B tables")
        run_name_option = next(
            part for part in cmd if part.startswith("--wandb_run_name=")
        )
        run_names.append(run_name_option.split("=", 1)[1])
        project_option = next(
            part for part in cmd if part.startswith("--wandb_project=")
        )
        expected_project = project_for_suite(args, suite)
        if project_option != f"--wandb_project={expected_project}":
            raise AssertionError(
                f"unexpected W&B project for {suite.name}: {project_option}"
            )

    if len(run_names) != len(set(run_names)):
        raise AssertionError("duplicate W&B run names in reference LR sanity plan")


def status_path(args: argparse.Namespace) -> Path:
    output_dir = EXPERIMENT_RESULTS_ROOT / FAMILY / "run_status"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{args.run_prefix}_status_{timestamp}.csv"


def write_status(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "suite",
        "seed",
        "matrix_lr",
        "method",
        "return_code",
        "status",
        "command",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_plan_summary(
    args: argparse.Namespace,
    plan: list[tuple[Suite, int, float, str]],
) -> None:
    mode = "preflight" if args.preflight else (
        "quick ordering gate"
        if args.quick_ordering_gate
        else "LR-grid remainder"
        if args.lr_grid_remainder
        else "full LR grid"
    )
    projects = {project_for_suite(args, suite) for suite, _, _, _ in plan}
    print(
        f"reference LR sanity {mode}: {len(plan)} run(s) across "
        f"{len(projects)} project(s)"
    )
    for suite_name in dict.fromkeys(item[0].name for item in plan):
        suite_rows = [item for item in plan if item[0].name == suite_name]
        matrix_lrs = sorted({item[2] for item in suite_rows})
        methods = sorted({item[3] for item in suite_rows})
        print(
            f"  {suite_name}: {len(suite_rows)} run(s), "
            f"LRs={matrix_lrs}, methods={methods}, "
            f"project={project_for_suite(args, SUITES[suite_name])}"
        )


def main() -> None:
    args = parse_args()
    plan = effective_plan(args)
    for dataset in sorted({suite.dataset for suite, _, _, _ in plan}):
        if not ensure_data(dataset, args.dry_run):
            raise SystemExit(1)

    commands = [
        build_command(args, suite, seed, matrix_lr, method)
        for suite, seed, matrix_lr, method in plan
    ]
    validate_plan(args, plan, commands)
    print_plan_summary(args, plan)
    write_command_record(
        family=FAMILY,
        run_prefix=args.run_prefix,
        commands=commands,
        dry_run=args.dry_run,
        enabled=args.write_commands,
    )
    if args.dry_run:
        for cmd in commands:
            print(command_text(cmd))
        return

    path = status_path(args)
    status_rows: list[dict[str, object]] = []
    failures = 0
    for (suite, seed, matrix_lr, method), cmd in zip(plan, commands):
        print(command_text(cmd))
        completed = subprocess.run(cmd, cwd=SOURCE_REPO, check=False)
        return_code = int(completed.returncode)
        failures += int(return_code != 0)
        status_rows.append(
            {
                "suite": suite.name,
                "seed": seed,
                "matrix_lr": matrix_lr,
                "method": method,
                "return_code": return_code,
                "status": "completed" if return_code == 0 else "failed",
                "command": command_text(cmd),
            }
        )
        write_status(path, status_rows)
        if return_code != 0 and not args.continue_on_error:
            raise subprocess.CalledProcessError(return_code, cmd)

    print(f"wrote run status to {path}")
    if failures:
        raise SystemExit(f"{failures} sanity run(s) failed; see {path}")


if __name__ == "__main__":
    main()
