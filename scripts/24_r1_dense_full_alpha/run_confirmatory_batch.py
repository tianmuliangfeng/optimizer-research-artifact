#!/usr/bin/env python3
"""Run the frozen seed-2024/2025 R1 dense-full-alpha confirmation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_r1_dense_full_alpha.py"
METHODS = (
    "fullalpha0",
    "fullalpha0p25",
    "fullalpha0p50",
    "fullalpha0p75",
    "fullalpha1",
)
DEFAULT_RESULTS_ROOT = (
    Path(os.environ.get("SNM_RESULTS_ROOT", str(SCRIPT_DIR.parents[1] / "runs"))).expanduser()
    / "24_r1_dense_full_alpha"
    / "results"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--smoke-steps", type=int, default=34)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--wandb-project",
        default=(
            "Selective-Newton-Muon-MainConf-R1-"
            "DenseFullAlpha-Confirmatory-20260727"
        ),
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb-train-log-every", type=int, default=20)
    parser.add_argument("--wandb-init-timeout", type=int, default=120)
    parser.add_argument(
        "--run-prefix", default="mainconf_r1_dense_full_alpha_confirmatory"
    )
    parser.add_argument("--concurrent-node-training", action="store_true")
    parser.add_argument("--concurrent-workload", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seeds != list(dict.fromkeys(args.seeds)):
        parser.error("--seeds contains duplicates")
    if any(seed not in (2024, 2025) for seed in args.seeds):
        parser.error("--seeds may contain only 2024 and 2025")
    if args.smoke_steps < 34:
        parser.error("--smoke-steps must be at least 34")
    if args.concurrent_workload and not args.concurrent_node_training:
        parser.error(
            "--concurrent-workload requires --concurrent-node-training"
        )
    if args.wandb_mode != "online":
        parser.error("formal confirmatory evidence requires --wandb-mode online")
    return args


def common_command(args: argparse.Namespace, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--official-repo",
        str(args.official_repo),
        "--python-exe",
        args.python_exe,
        "--seed",
        str(seed),
        "--confirmatory",
        "--methods",
        *METHODS,
        "--run-prefix",
        args.run_prefix,
        "--wandb-project",
        args.wandb_project,
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-train-log-every",
        str(args.wandb_train_log_every),
        "--wandb-init-timeout",
        str(args.wandb_init_timeout),
    ]
    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    if args.concurrent_node_training:
        command.append("--concurrent-node-training")
    if args.concurrent_workload:
        command.extend(["--concurrent-workload", args.concurrent_workload])
    return command


def run(command: list[str], *, dry_run: bool = False) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=SCRIPT_DIR.parents[1], check=True)


def write_state(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.dry_run:
        for seed in args.seeds:
            run(
                common_command(args, seed) + ["--dry-run"],
                dry_run=True,
            )
        return

    batch_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    batch_root = (
        args.results_root.expanduser().resolve()
        / "confirmatory_controller"
        / batch_id
    )
    batch_root.mkdir(parents=True, exist_ok=False)
    state_path = batch_root / "confirmatory_controller.json"
    state: dict[str, object] = {
        "protocol": (
            "r1_dense_full_alpha_seeds2024_2025_confirmatory_v1"
        ),
        "contract": str(
            (SCRIPT_DIR / "CONFIRMATORY_CONTRACT_20260727.md").resolve()
        ),
        "batch_id": batch_id,
        "seeds": args.seeds,
        "status": "running",
        "seed_status": {},
        "wandb_required": True,
    }
    write_state(state_path, state)

    failures = []
    for seed in args.seeds:
        seed_status: dict[str, object] = {"status": "smoke_running"}
        state["seed_status"][str(seed)] = seed_status
        write_state(state_path, state)
        base = common_command(args, seed)
        smoke_command = base + [
            "--numerical-smoke",
            "--smoke-steps",
            str(args.smoke_steps),
            "--wandb-mode",
            "disabled",
            "--results-dir",
            str(batch_root),
        ]
        try:
            run(smoke_command)
            smoke_manifests = sorted(
                batch_root.glob(f"*_smoke_seed{seed}/r1_manifest.json")
            )
            if len(smoke_manifests) != 1:
                raise RuntimeError(
                    f"expected one seed-{seed} smoke manifest, "
                    f"found {smoke_manifests}"
                )
            smoke_manifest = smoke_manifests[0].resolve()
            seed_status.update(
                {
                    "status": "formal_running",
                    "smoke_manifest": str(smoke_manifest),
                }
            )
            write_state(state_path, state)
            formal_command = base + [
                "--smoke-manifest",
                str(smoke_manifest),
                "--results-dir",
                str(batch_root),
            ]
            if args.continue_on_error:
                formal_command.append("--continue-on-error")
            run(formal_command)
            formal_manifests = sorted(
                batch_root.glob(f"*_formal_seed{seed}/r1_manifest.json")
            )
            if len(formal_manifests) != 1:
                raise RuntimeError(
                    f"expected one seed-{seed} formal manifest, "
                    f"found {formal_manifests}"
                )
            formal_manifest = formal_manifests[0].resolve()
            formal_payload = json.loads(
                formal_manifest.read_text(encoding="utf-8")
            )
            seed_status.update(
                {
                    "formal_manifest": str(formal_manifest),
                    "formal_batch": str(formal_manifest.parent),
                    "wandb_complete": formal_payload.get("wandb_complete"),
                }
            )
            if formal_payload.get("status") != "completed_valid":
                raise RuntimeError(
                    f"seed-{seed} formal batch is not fully valid: "
                    f"{formal_payload.get('status')!r}"
                )
            if formal_payload.get("wandb_complete") is not True:
                raise RuntimeError(
                    f"seed-{seed} local evidence completed but W&B is "
                    f"incomplete; resume exact batch {formal_manifest.parent}"
                )
            seed_status["status"] = "completed"
        except BaseException as exc:
            seed_status.update({"status": "failed", "error": repr(exc)})
            failures.append((seed, repr(exc)))
            if not args.continue_on_error:
                state["status"] = "failed"
                write_state(state_path, state)
                raise
        finally:
            write_state(state_path, state)

    state["status"] = "completed" if not failures else "completed_with_failures"
    state["failures"] = failures
    write_state(state_path, state)
    print(f"R1 dense-full confirmatory controller: {state_path}", flush=True)
    if failures:
        raise RuntimeError(f"R1 dense-full confirmatory failures: {failures}")


if __name__ == "__main__":
    main()
