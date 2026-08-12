"""Run the preregistered R1 block-alpha confirmation unattended.

For each requested seed this controller runs the exact-shape four-method smoke,
discovers its manifest, and then starts the matching four formal cells.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_r1_block_alpha.py"
DEFAULT_RESULTS_ROOT = (
    Path(os.environ.get("SNM_RESULTS_ROOT", str(SCRIPT_DIR.parents[1] / "runs"))).expanduser()
    / "22_r1_block_alpha"
    / "results"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unattended seed-2024/2025 R1 block-alpha confirmation: "
            "matching smoke followed by four formal alpha cells per seed."
        )
    )
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--smoke-steps", type=int, default=34)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--wandb-project",
        default="Selective-Newton-Muon-MainConf-R1-BlockAlpha-Confirmatory-20260724",
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb-train-log-every", type=int, default=20)
    parser.add_argument("--wandb-init-timeout", type=int, default=120)
    parser.add_argument("--run-prefix", default="mainconf_r1_block_alpha_confirmatory")
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
        parser.error("--concurrent-workload requires --concurrent-node-training")
    return args


def common_command(args: argparse.Namespace, seed: int) -> list[str]:
    cmd = [
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
        "alpha0",
        "alpha0p25",
        "alpha0p50",
        "alpha0p75",
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
        cmd.extend(["--wandb-entity", args.wandb_entity])
    if args.concurrent_node_training:
        cmd.append("--concurrent-node-training")
    if args.concurrent_workload:
        cmd.extend(["--concurrent-workload", args.concurrent_workload])
    return cmd


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print(subprocess.list2cmdline(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=SCRIPT_DIR.parents[1], check=True)


def write_state(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.dry_run:
        for seed in args.seeds:
            run(common_command(args, seed) + ["--dry-run"], dry_run=True)
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
        "protocol": "r1_block_alpha_seeds2024_2025_confirmatory_v1",
        "batch_id": batch_id,
        "seeds": args.seeds,
        "status": "running",
        "seed_status": {},
    }
    write_state(state_path, state)

    failures = []
    for seed in args.seeds:
        seed_status: dict[str, object] = {"status": "smoke_running"}
        state["seed_status"][str(seed)] = seed_status
        write_state(state_path, state)
        base = common_command(args, seed)
        smoke_cmd = base + [
            "--numerical-smoke",
            "--smoke-steps",
            str(args.smoke_steps),
            "--wandb-mode",
            "disabled",
            "--results-dir",
            str(batch_root),
        ]
        try:
            run(smoke_cmd)
            manifests = sorted(
                batch_root.glob(f"*_smoke_seed{seed}/r1_manifest.json")
            )
            if len(manifests) != 1:
                raise RuntimeError(
                    f"expected one seed-{seed} smoke manifest, found {manifests}"
                )
            smoke_manifest = manifests[0].resolve()
            seed_status.update(
                {
                    "status": "formal_running",
                    "smoke_manifest": str(smoke_manifest),
                }
            )
            write_state(state_path, state)
            formal_cmd = base + [
                "--smoke-manifest",
                str(smoke_manifest),
                "--results-dir",
                str(batch_root),
            ]
            if args.continue_on_error:
                formal_cmd.append("--continue-on-error")
            run(formal_cmd)
            formal_manifests = sorted(
                batch_root.glob(f"*_formal_seed{seed}/r1_manifest.json")
            )
            if len(formal_manifests) != 1:
                raise RuntimeError(
                    f"expected one seed-{seed} formal manifest, found {formal_manifests}"
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
                    f"seed-{seed} local evidence completed but W&B is incomplete; "
                    f"resume exact batch {formal_manifest.parent}"
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
    print(f"R1 confirmatory controller: {state_path}", flush=True)
    if failures:
        raise RuntimeError(f"R1 confirmatory failures: {failures}")


if __name__ == "__main__":
    main()
