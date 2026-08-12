"""Run the three-seed WikiText-103 depth x c_proj K-mode experiment.

This is the cross-dataset replication of family 25.  It reuses the audited
depth-routing implementation while pinning the GPT-2-tokenized WikiText-103
50M-token subset and recording a content fingerprint before any training
process starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR / "25_owt_depth_kmode"))
sys.path.insert(0, str(SCRIPTS_DIR / "_shared"))

import run_owt_depth_kmode as base
from project_paths import EXPERIMENT_RESULTS_ROOT, SOURCE_REPO


FAMILY = "28_wikitext_depth_kmode"
DATASET = "wikitext103_gpt2_50m"
WANDB_PROJECT = (
    "Selective-Newton-Muon-MainConf-WikiText103-Depth-KMode-20260724"
)
RUN_PREFIX = "mainconf_wikitext103_12L_depth_kmode"

# These two binaries are path-independent and were used by the existing
# WikiText-103 project experiments.  Metadata files embed absolute paths and
# are therefore audited semantically rather than pinned by hash.
EXPECTED_TRAIN_SHA256 = (
    "58c04ef835efade28c303561b99873eed64ac6a4060c5d715b4fb6538ae3cd34"
)
EXPECTED_VAL_SHA256 = (
    "397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979"
)
EXPECTED_DATASET_FIELDS = {
    "dataset": "wikitext",
    "dataset_config": "wikitext-103-raw-v1",
    "tokenizer": "gpt2",
    "train_split": "train",
    "val_split": "validation",
    "train_tokens": 50_000_000,
    "val_tokens": 249_750,
    "vocab_size": 50_257,
}


def parse_args() -> argparse.Namespace:
    build_parser = getattr(base, "build_parser", None)
    if callable(build_parser):
        parser = build_parser(
            dataset_default=DATASET,
            wandb_project_default=WANDB_PROJECT,
            run_prefix_default=RUN_PREFIX,
            description=(
                "Three-seed WikiText-103 replication of the paired depth x "
                "K-mode experiment. Selected mlp.c_proj depths use none or "
                "diag; every unselected c_proj and non-c_proj matrix retains "
                "full Newton-Muon."
            ),
        )
    else:
        # Backward compatibility for hosts that have the original family-25
        # runner.  Family 28 must not require rewriting a completed experiment
        # runner merely to obtain its parser factory.
        parser = argparse.ArgumentParser(
            description=(
                "Three-seed WikiText-103 replication of the paired depth x "
                "K-mode experiment."
            )
        )
        phase = parser.add_mutually_exclusive_group(required=True)
        phase.add_argument("--dry-run", action="store_true")
        phase.add_argument("--numerical-smoke", action="store_true")
        phase.add_argument("--formal", action="store_true")
        parser.add_argument("--python-exe", default=None)
        parser.add_argument("--dataset", default=DATASET)
        parser.add_argument(
            "--seeds", type=int, nargs="+", default=[2024, 2025, 2026]
        )
        parser.add_argument(
            "--rules",
            nargs="+",
            choices=tuple(base.RULE_LAYERS),
            default=list(base.RULE_LAYERS),
        )
        parser.add_argument(
            "--modes",
            nargs="+",
            choices=base.VALID_MODES,
            default=list(base.VALID_MODES),
        )
        parser.add_argument(
            "--anchors",
            nargs="*",
            choices=base.VALID_ANCHORS,
            default=list(base.VALID_ANCHORS),
        )
        parser.add_argument("--base-config", default=base.BASE_CONFIG)
        parser.add_argument("--mechanism-config", default=base.MECHANISM_CONFIG)
        parser.add_argument("--n-layer", type=int, default=12)
        parser.add_argument("--n-head", type=int, default=12)
        parser.add_argument("--n-embd", type=int, default=768)
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--block-size", type=int, default=512)
        parser.add_argument(
            "--gradient-accumulation-steps", type=int, default=1
        )
        parser.add_argument("--max-iters", type=int, default=5000)
        parser.add_argument("--lr-decay-iters", type=int, default=5000)
        parser.add_argument("--muon-learning-rate", type=float, default=0.02)
        parser.add_argument("--input-beta", type=float, default=0.95)
        parser.add_argument("--input-ridge", type=float, default=0.2)
        parser.add_argument("--input-refresh", type=int, default=32)
        parser.add_argument("--input-max-samples", type=int, default=2048)
        parser.add_argument("--smoke-steps", type=int, default=34)
        parser.add_argument("--eval-iters", type=int, default=20)
        parser.add_argument("--eval-interval", type=int, default=500)
        parser.add_argument("--log-interval", type=int, default=20)
        parser.add_argument("--device", default=None)
        parser.add_argument("--wandb-project", default=WANDB_PROJECT)
        parser.add_argument(
            "--wandb-mode",
            default="online",
            choices=("online", "offline", "disabled"),
        )
        parser.add_argument(
            "--wandb-log-profile",
            default="paper",
            choices=("paper", "full"),
        )
        parser.add_argument("--wandb-log-tables", action="store_true")
        parser.add_argument("--wandb-group", default=None)
        parser.add_argument("--run-prefix", default=RUN_PREFIX)
        parser.add_argument("--continue-on-error", action="store_true")
        parser.add_argument(
            "--no-write-commands",
            action="store_false",
            dest="write_commands",
        )
        parser.set_defaults(write_commands=True)
    parser.add_argument(
        "--expected-train-sha256",
        default=EXPECTED_TRAIN_SHA256,
        help="Required SHA-256 of the pinned WikiText train.bin.",
    )
    parser.add_argument(
        "--expected-val-sha256",
        default=EXPECTED_VAL_SHA256,
        help="Required SHA-256 of the pinned WikiText val.bin.",
    )
    parser.add_argument(
        "--data-audit-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for the data fingerprint JSON. Defaults to "
            "the experiment artifact root."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_wikitext_dataset(args: argparse.Namespace) -> None:
    if args.dataset != DATASET:
        raise ValueError(
            f"Family {FAMILY} is pinned to --dataset={DATASET}; got {args.dataset!r}"
        )

    data_dir = SOURCE_REPO / "data" / args.dataset
    paths = {
        "train_bin": data_dir / "train.bin",
        "val_bin": data_dir / "val.bin",
        "meta_pkl": data_dir / "meta.pkl",
        "prepare_summary": data_dir / "prepare_summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Pinned WikiText dataset is incomplete:\n- " + "\n- ".join(missing)
        )

    summary = json.loads(paths["prepare_summary"].read_text(encoding="utf-8"))
    failures = []
    for field, expected in EXPECTED_DATASET_FIELDS.items():
        observed = summary.get(field)
        if observed != expected:
            failures.append(f"{field}: observed={observed!r}, expected={expected!r}")

    expected_sizes = {
        "train_bin": EXPECTED_DATASET_FIELDS["train_tokens"] * 2,
        "val_bin": EXPECTED_DATASET_FIELDS["val_tokens"] * 2,
    }
    for key, expected in expected_sizes.items():
        observed = paths[key].stat().st_size
        if observed != expected:
            failures.append(
                f"{key} bytes: observed={observed}, expected={expected}"
            )

    hashes = {
        key: sha256_file(path)
        for key, path in paths.items()
    }
    expected_hashes = {
        "train_bin": args.expected_train_sha256.lower(),
        "val_bin": args.expected_val_sha256.lower(),
    }
    for key, expected in expected_hashes.items():
        observed = hashes[key].lower()
        if observed != expected:
            failures.append(
                f"{key} SHA-256: observed={observed}, expected={expected}"
            )

    audit = {
        "schema_version": 1,
        "family": FAMILY,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if failures else "passed",
        "data_dir": str(data_dir.resolve()),
        "expected_fields": EXPECTED_DATASET_FIELDS,
        "observed_fields": {
            field: summary.get(field) for field in EXPECTED_DATASET_FIELDS
        },
        "files": {
            key: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": hashes[key],
            }
            for key, path in paths.items()
        },
        "failures": failures,
    }
    audit_dir = args.data_audit_dir or (
        EXPERIMENT_RESULTS_ROOT / FAMILY / "data_audit"
    )
    audit_path = audit_dir / "wikitext103_gpt2_50m_fingerprint.json"
    atomic_json(audit_path, audit)
    print(f"WikiText data audit: {audit_path}")
    print(
        "WikiText fingerprints: "
        f"train={hashes['train_bin']} val={hashes['val_bin']}"
    )
    if failures:
        raise RuntimeError(
            "Pinned WikiText data audit failed:\n- " + "\n- ".join(failures)
        )


def main() -> None:
    args = parse_args()
    run_experiment = getattr(base, "run_experiment", None)
    if callable(run_experiment):
        run_experiment(
            args,
            family=FAMILY,
            dataset_validator=validate_wikitext_dataset,
        )
        return

    # Backward-compatible execution path for the original family-25 runner.
    base.validate_args(args)
    base.validate_source_support()
    if not base.ensure_data(args.dataset, args.dry_run):
        raise SystemExit(1)
    validate_wikitext_dataset(args)
    base.print_plan(args)
    commands = base.build_commands(args)
    base.validate_commands(args, commands)
    base.write_command_record(
        family=FAMILY,
        run_prefix=args.run_prefix,
        commands=commands,
        dry_run=args.dry_run,
        enabled=args.write_commands,
    )

    failures = []
    for cmd in commands:
        try:
            base.run_cmd(cmd, args.dry_run)
        except subprocess.CalledProcessError as exc:
            name = base.option_value(cmd, "wandb_run_name")
            print(
                f"RUN_FAILED name={name} returncode={exc.returncode}",
                file=sys.stderr,
                flush=True,
            )
            failures.append((name, exc.returncode))
            if not args.continue_on_error:
                raise
    if failures:
        detail = ", ".join(f"{name}(exit={code})" for name, code in failures)
        raise RuntimeError(f"{len(failures)} training run(s) failed: {detail}")


if __name__ == "__main__":
    main()
