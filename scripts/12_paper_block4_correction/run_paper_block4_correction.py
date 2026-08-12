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


FAMILY = "12_paper_block4_correction"
PAPER_BLOCK4_CONFIG = "config/baselines/14_newton_muon_paper_block4.py"
MUON_BLOG_CONFIG = "config/baselines/15_muon_blog_reference.py"
FORMAL_MATRIX_LR = 0.01
REFERENCE_PROJECTS = {
    12: "Selective-Newton-Muon-MainConf-ReferenceReset-LR001-12L-20260717",
    18: "Selective-Newton-Muon-MainConf-ReferenceReset-LR001-18L-20260717",
    24: "Selective-Newton-Muon-MainConf-ReferenceReset-LR001-24L-20260717",
}
VALID_METHODS = ("muon", "block4", "none", "diag")


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
    muon_learning_rate: float
    default_methods: tuple[str, ...] = VALID_METHODS
    purpose: str = ""

    @property
    def nominal_tokens(self) -> int:
        return (
            self.max_iters
            * self.batch_size
            * self.block_size
            * self.gradient_accumulation_steps
        )


SUITES = {
    "owt12l_5k": Suite(
        name="owt12l_5k",
        dataset="openwebtext_gpt2_50m",
        base_config="config/train_openwebtext_gpt2_50m_tier3.py",
        n_layer=12,
        n_head=12,
        n_embd=768,
        batch_size=16,
        block_size=512,
        gradient_accumulation_steps=1,
        max_iters=5000,
        lr_decay_iters=5000,
        muon_learning_rate=FORMAL_MATRIX_LR,
        purpose="repair the original 12L/50M-token main-result baseline",
    ),
    "owt12l_100m": Suite(
        name="owt12l_100m",
        dataset="openwebtext_gpt2_50m",
        base_config="config/train_openwebtext_gpt2_50m_tier3.py",
        n_layer=12,
        n_head=12,
        n_embd=768,
        batch_size=16,
        block_size=512,
        gradient_accumulation_steps=1,
        max_iters=12208,
        lr_decay_iters=12208,
        muon_learning_rate=FORMAL_MATRIX_LR,
        purpose="repair the OpenWebText 12L long-budget baseline",
    ),
    "wikitext12l_100m": Suite(
        name="wikitext12l_100m",
        dataset="wikitext103_gpt2_50m",
        base_config="config/train_openwebtext_gpt2_50m_tier3.py",
        n_layer=12,
        n_head=12,
        n_embd=768,
        batch_size=16,
        block_size=512,
        gradient_accumulation_steps=1,
        max_iters=12208,
        lr_decay_iters=12208,
        muon_learning_rate=FORMAL_MATRIX_LR,
        purpose="repair the WikiText-103 12L dataset-generalization baseline",
    ),
    "owt18l_3k": Suite(
        name="owt18l_3k",
        dataset="openwebtext_gpt2_50m",
        base_config="config/train_openwebtext_gpt2_50m_tier4.py",
        n_layer=18,
        n_head=12,
        n_embd=768,
        batch_size=8,
        block_size=512,
        gradient_accumulation_steps=1,
        max_iters=3000,
        lr_decay_iters=3000,
        muon_learning_rate=FORMAL_MATRIX_LR,
        purpose=(
            "replace the legacy 18L depth-scaling comparison with four "
            "implementation-aligned methods"
        ),
    ),
    "owt24l_12k": Suite(
        name="owt24l_12k",
        dataset="openwebtext_gpt2_50m",
        base_config="config/train_openwebtext_gpt2_50m_tier4.py",
        n_layer=24,
        n_head=16,
        n_embd=1024,
        batch_size=2,
        block_size=512,
        gradient_accumulation_steps=1,
        max_iters=12000,
        lr_decay_iters=3000,
        muon_learning_rate=FORMAL_MATRIX_LR,
        purpose="repair the primary OpenWebText 24L scale-up baseline",
    ),
    "wikitext24l_12k_lr001": Suite(
        name="wikitext24l_12k_lr001",
        dataset="wikitext103_gpt2_50m",
        base_config="config/train_openwebtext_gpt2_50m_tier4.py",
        n_layer=24,
        n_head=16,
        n_embd=1024,
        batch_size=2,
        block_size=512,
        gradient_accumulation_steps=1,
        max_iters=12000,
        lr_decay_iters=3000,
        muon_learning_rate=FORMAL_MATRIX_LR,
        purpose="repair the primary WikiText-103 24L cross-dataset baseline",
    ),
}

DEFAULT_SUITE_ORDER = (
    "owt12l_100m",
    "wikitext12l_100m",
    "owt18l_3k",
    "owt24l_12k",
    "wikitext24l_12k_lr001",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair legacy cross-dataset/depth comparisons with the Newton-Muon "
            "paper's four-block mlp.c_proj K structure."
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Run W&B-disabled 12L Muon and block4 jobs for 33 steps, "
            "including the first K refresh."
        ),
    )
    parser.add_argument("--python-exe", default=None)
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=tuple(SUITES),
        default=list(DEFAULT_SUITE_ORDER),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=VALID_METHODS,
        default=None,
        help=(
            "Override each suite's method list. By default every suite reruns "
            "Muon, paper-block4 Newton-Muon, none, and diag."
        ),
    )
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--run-prefix",
        default="mainconf_reference_lr001",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Finish the remaining overnight queue if an individual run fails.",
    )
    parser.add_argument("--no-write-commands", action="store_false", dest="write_commands")
    parser.set_defaults(write_commands=True)
    return parser.parse_args()


def effective_plan(args: argparse.Namespace) -> list[tuple[Suite, int, str]]:
    if args.preflight:
        return [
            (SUITES["owt12l_5k"], 2026, "muon"),
            (SUITES["owt12l_5k"], 2026, "block4"),
        ]
    plan = []
    for suite_name in args.suites:
        suite = SUITES[suite_name]
        methods = tuple(args.methods) if args.methods is not None else suite.default_methods
        for seed in args.seeds:
            for method in methods:
                plan.append((suite, seed, method))
    return plan


def method_label(method: str) -> str:
    return {
        "muon": "muon_blog",
        "block4": "paper_block4",
        "none": "no_cproj_k",
        "diag": "diag_cproj_k",
    }[method]


def project_for_suite(args: argparse.Namespace, suite: Suite) -> str:
    try:
        return REFERENCE_PROJECTS[suite.n_layer]
    except KeyError as exc:
        raise ValueError(
            f"no default reference-reset project for n_layer={suite.n_layer}"
        ) from exc


def build_command(
    args: argparse.Namespace,
    suite: Suite,
    seed: int,
    method: str,
) -> list[str]:
    is_preflight = bool(args.preflight)
    max_iters = 33 if is_preflight else suite.max_iters
    lr_decay_iters = 33 if is_preflight else suite.lr_decay_iters
    eval_iters = 1 if is_preflight else args.eval_iters
    eval_interval = 32 if is_preflight else args.eval_interval
    log_interval = 10 if is_preflight else args.log_interval
    wandb_mode = "disabled" if is_preflight else args.wandb_mode
    run_prefix = (
        f"{args.run_prefix}_preflight"
        if is_preflight
        else args.run_prefix
    )
    label = method_label(method)
    run_name = f"{run_prefix}_{suite.name}_{label}_seed{seed}"
    group = f"{args.run_prefix}_{suite.name}_seed{seed}"
    tags = ",".join(
        [
            "publication",
            "paper_implementation_correction",
            "newton_muon_paper_structure",
            "cproj_reference_block4",
            "cproj_4xdxd",
            suite.dataset,
            f"L{suite.n_layer}",
            f"D{suite.n_embd}",
            f"steps_{suite.max_iters}",
            f"nominal_tokens_{suite.nominal_tokens}",
            f"mulr_{suite.muon_learning_rate:g}",
        ]
    )

    command_args = argparse.Namespace(
        python_exe=args.python_exe,
        dataset=suite.dataset,
        seed=seed,
        wandb_project=project_for_suite(args, suite),
        wandb_mode=wandb_mode,
        # Formal reset runs keep only the scalar paper profile. Tables are
        # deliberately unavailable to avoid recreating the noisy W&B layout.
        wandb_log_profile="paper",
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
        muon_learning_rate=suite.muon_learning_rate,
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


def status_path(args: argparse.Namespace) -> Path:
    output_dir = EXPERIMENT_RESULTS_ROOT / FAMILY / "run_status"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{args.run_prefix}_status_{timestamp}.csv"


def write_status(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "suite",
        "seed",
        "method",
        "return_code",
        "status",
        "command",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_plan(plan: list[tuple[Suite, int, str]], commands: list[list[str]]) -> None:
    if not plan:
        raise ValueError("empty correction plan")
    if len(plan) != len(commands):
        raise AssertionError("plan/command count mismatch")
    run_names = []
    observed_projects: dict[int, set[str]] = {}
    for (suite, seed, method), cmd in zip(plan, commands):
        expected = {
            "--muon_nesterov=True",
            "--muon_momentum_ema=True",
            "--muon_split_qkv=True",
            "--muon_adjust_lr_for_shape=True",
            "--muon_ns_compute_dtype=bfloat16",
            "--muon_momentum=0.95",
            "--muon_ns_steps=5",
            "--matrix_weight_decay=0.0",
            f"--muon_learning_rate={FORMAL_MATRIX_LR}",
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
            raise AssertionError(f"{suite.name}/seed{seed} missing overrides: {missing}")
        matrix_lr_options = [
            part for part in cmd if part.startswith("--muon_learning_rate=")
        ]
        if matrix_lr_options != [f"--muon_learning_rate={FORMAL_MATRIX_LR}"]:
            raise AssertionError(
                f"{suite.name}/seed{seed} must use the single formal matrix LR "
                f"{FORMAL_MATRIX_LR}; got {matrix_lr_options}"
            )
        if any(part == "--wandb_log_tables=True" for part in cmd):
            raise AssertionError("formal reference-reset runs must not upload W&B tables")
        name_option = next(part for part in cmd if part.startswith("--wandb_run_name="))
        run_names.append(name_option.split("=", 1)[1])
        project_option = next(part for part in cmd if part.startswith("--wandb_project="))
        project = project_option.split("=", 1)[1]
        expected_project = REFERENCE_PROJECTS[suite.n_layer]
        if project != expected_project:
            raise AssertionError(
                f"{suite.name}/seed{seed} must use project {expected_project}; "
                f"got {project}"
            )
        observed_projects.setdefault(suite.n_layer, set()).add(project)
    if len(run_names) != len(set(run_names)):
        raise AssertionError("duplicate W&B run names in correction plan")
    for n_layer, projects in observed_projects.items():
        if len(projects) != 1:
            raise AssertionError(
                f"{n_layer}L commands span multiple W&B projects: {sorted(projects)}"
            )


def main() -> None:
    args = parse_args()
    plan = effective_plan(args)
    for dataset in sorted({suite.dataset for suite, _, _ in plan}):
        if not ensure_data(dataset, args.dry_run):
            raise SystemExit(1)

    commands = [
        build_command(args, suite, seed, method)
        for suite, seed, method in plan
    ]
    validate_plan(plan, commands)
    print(
        f"paper-block4 correction plan: {len(commands)} runs across "
        f"{len({project_for_suite(args, suite) for suite, _, _ in plan})} project(s)"
    )
    for suite_name in dict.fromkeys(suite.name for suite, _, _ in plan):
        suite = SUITES[suite_name]
        suite_runs = sum(1 for item in plan if item[0].name == suite_name)
        print(
            f"  {suite_name}: {suite_runs} run(s), "
            f"{suite.n_layer}L/{suite.n_embd}D, {suite.nominal_tokens} nominal tokens, "
            f"project={project_for_suite(args, suite)}"
        )

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
    for (suite, seed, method), cmd in zip(plan, commands):
        print(command_text(cmd))
        completed = subprocess.run(cmd, cwd=SOURCE_REPO, check=False)
        return_code = int(completed.returncode)
        failures += int(return_code != 0)
        status_rows.append(
            {
                "suite": suite.name,
                "seed": seed,
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
        raise SystemExit(f"{failures} correction run(s) failed; see {path}")


if __name__ == "__main__":
    main()
