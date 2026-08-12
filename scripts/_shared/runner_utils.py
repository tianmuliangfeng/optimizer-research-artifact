from __future__ import annotations

import csv
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from project_paths import EXPERIMENT_RESULTS_ROOT, SOURCE_REPO


BASELINE_CONFIGS = {
    "muon": ("00_muon", "config/baselines/00_muon.py"),
    "newton": ("13_newton_muon_fast", "config/baselines/13_newton_muon_fast.py"),
}

REPLAY_CONFIG = "config/storage/38_static_center_mask_replay.py"
DEFAULT_MIDDLE_RELEASE = 0.5614035087719298
RELEASE_ALL_CPROJ = 0.8421052631578947


def release_label(release_frac: float) -> str:
    return f"release{int(round(release_frac * 100)):02d}"


def bool_literal(value: bool) -> str:
    return "True" if value else "False"


def command_text(cmd: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in cmd])


def ensure_data(dataset: str, dry_run: bool) -> bool:
    data_dir = SOURCE_REPO / "data" / dataset
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    if train_path.exists() and val_path.exists():
        return True
    print(f"Missing dataset files under {data_dir}.")
    if dataset.startswith("wikitext103_gpt2"):
        print(
            "Prepare WikiText-103 from the artifact root: "
            "python scripts/04_dataset_generalization/prepare_wikitext103_gpt2.py "
            f"--output-dir backends/nanogpt/data/{dataset} "
            "--max-train-tokens 50000000 --max-val-tokens 1000000 --force"
        )
    else:
        print(
            "Prepare data first, for example from the stable repo: "
            f"python data/openwebtext_gpt2/prepare.py --output-dir data/{dataset} "
            "--max-train-tokens 50000000 --max-val-tokens 1000000 --val-fraction 0.01 --force"
        )
    return dry_run


def read_mask_target(mask_path: str | Path) -> float:
    with open(mask_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ("target_release_k_fraction", "actual_release_k_fraction"):
                value = row.get(key, "")
                if value != "":
                    return float(value)
            break
    raise ValueError(f"could not read release fraction from {mask_path}")


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print(command_text(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=SOURCE_REPO, check=True)


def build_model_mask(
    *,
    python_exe: str | None = None,
    output_dir: Path,
    n_layer: int,
    n_embd: int,
    target_release_frac: float,
    dataset: str,
    mask_seed: int,
    run_prefix: str,
    wandb_project: str,
    wandb_group: str,
    dry_run: bool,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "large_model_center_cproj_mask_summary.csv"
    cmd = [
        python_exe or sys.executable,
        "scripts/build_model_center_cproj_mask.py",
        f"--output-dir={output_dir}",
        f"--n-layer={n_layer}",
        f"--n-embd={n_embd}",
        f"--target-release-frac={target_release_frac}",
        f"--dataset={dataset}",
        f"--mask-seed={mask_seed}",
        f"--run-prefix={run_prefix}",
        f"--wandb-project={wandb_project}",
        f"--wandb-group={wandb_group}",
    ]
    run_cmd(cmd, dry_run)
    if dry_run and not summary_path.exists():
        return str(
            output_dir
            / f"large_center_cproj_L{n_layer}_D{n_embd}_AUTO_{release_label(target_release_frac)}.csv"
        )
    if not summary_path.exists():
        raise FileNotFoundError(f"mask summary not found in {output_dir}")
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty mask summary: {summary_path}")
    return rows[0]["mask_path"]


def append_if_present(cmd: list[str], name: str, value) -> None:
    if value is not None:
        cmd.append(f"--{name}={value}")


def append_common_train_overrides(cmd: list[str], args) -> None:
    append_if_present(cmd, "max_iters", args.max_iters)
    lr_decay_iters = getattr(args, "lr_decay_iters", None)
    if lr_decay_iters is None:
        lr_decay_iters = args.max_iters
    append_if_present(cmd, "lr_decay_iters", lr_decay_iters)
    append_if_present(cmd, "eval_iters", args.eval_iters)
    append_if_present(cmd, "eval_interval", args.eval_interval)
    append_if_present(cmd, "log_interval", args.log_interval)
    append_if_present(cmd, "batch_size", args.batch_size)
    append_if_present(cmd, "block_size", args.block_size)
    append_if_present(cmd, "gradient_accumulation_steps", args.gradient_accumulation_steps)
    append_if_present(cmd, "muon_learning_rate", getattr(args, "muon_learning_rate", None))
    append_if_present(cmd, "cuda_memory_fraction", getattr(args, "cuda_memory_fraction", None))
    append_if_present(cmd, "cuda_memory_budget_gib", getattr(args, "cuda_memory_budget_gib", None))
    append_if_present(cmd, "device", args.device)
    if args.device == "cpu":
        cmd.append("--dtype=float32")
    if getattr(args, "always_save_checkpoint", None) is not None:
        cmd.append(f"--always_save_checkpoint={bool_literal(args.always_save_checkpoint)}")


def append_model_overrides(cmd: list[str], args) -> None:
    append_if_present(cmd, "n_layer", args.n_layer)
    append_if_present(cmd, "n_head", args.n_head)
    append_if_present(cmd, "n_embd", args.n_embd)


def wandb_enabled(args) -> bool:
    return args.wandb_mode != "disabled"


def base_train_cmd(args, *, config: str, run_name: str, group: str, method: str, tags: str) -> list[str]:
    return [
        getattr(args, "python_exe", None) or sys.executable,
        "train.py",
        config,
        f"--dataset={args.dataset}",
        f"--seed={args.seed}",
        f"--out_dir=out_{run_name}",
        f"--wandb_log={bool_literal(wandb_enabled(args))}",
        f"--wandb_project={args.wandb_project}",
        f"--wandb_mode={args.wandb_mode if wandb_enabled(args) else 'offline'}",
        f"--wandb_log_profile={args.wandb_log_profile}",
        f"--wandb_log_tables={bool_literal(args.wandb_log_tables)}",
        f"--wandb_group={group}",
        f"--wandb_run_name={run_name}",
        f"--wandb_tags={tags},{method}",
    ]


def baseline_train_cmd(args, *, config: str, run_name: str, group: str, method: str, tags: str) -> list[str]:
    _, method_config = BASELINE_CONFIGS[method]
    cmd = base_train_cmd(args, config=config, run_name=run_name, group=group, method=method, tags=tags)
    cmd.insert(3, method_config)
    return cmd


def selective_train_cmd(
    args,
    *,
    config: str,
    run_name: str,
    group: str,
    tags: str,
    mask_path: str,
    release_frac: float,
    method_label: str,
) -> list[str]:
    cmd = base_train_cmd(
        args,
        config=config,
        run_name=run_name,
        group=group,
        method=method_label,
        tags=tags,
    )
    cmd.insert(3, REPLAY_CONFIG)
    cmd.extend(
        [
            f"--selective_static_mask_path={mask_path}",
            f"--selective_static_mask_seed={args.static_mask_seed}",
            f"--selective_static_mask_target_release_k_fraction={release_frac}",
            f"--selective_release_k_fraction={release_frac}",
            "--update_similarity_probe_enabled=False",
        ]
    )
    return cmd


def write_command_record(
    *,
    family: str,
    run_prefix: str,
    commands: list[list[str]],
    dry_run: bool,
    enabled: bool,
) -> Path | None:
    if not enabled:
        return None
    command_dir = EXPERIMENT_RESULTS_ROOT / family / "commands"
    command_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = command_dir / f"{run_prefix}_commands_{timestamp}.md"
    lines = [
        f"# {run_prefix} commands",
        "",
        f"- generated_at: {timestamp}",
        f"- dry_run: {dry_run}",
        f"- working_directory: {SOURCE_REPO}",
        "",
    ]
    for idx, cmd in enumerate(commands, start=1):
        lines.extend([f"## Command {idx}", "", "```powershell", f"cd {SOURCE_REPO}", command_text(cmd), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote command record to {path}")
    return path


def iters_for_tokens(target_tokens: int, batch_size: int, block_size: int, grad_accum: int) -> int:
    tokens_per_iter = batch_size * block_size * grad_accum
    return int(math.ceil(target_tokens / tokens_per_iter))


def method_choices() -> list[str]:
    return ["muon", "newton", "selective", "release_all_cproj"]
