"""Stage-gated controller for the pinned single-GPU 1.014B LLaMA profile."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / "17_llama_swiglu_validation"
BASE_RUNNER_PATH = BASE_DIR / "run_llama_swiglu_validation.py"
BASE_TRAINER_PATH = BASE_DIR / "train_llama_swiglu.py"
TRAINER_PATH = HERE / "train_llama_swiglu_1b.py"
PINNED_BASE_TRAINER_SHA256 = "b72eb0d2a1dfa91b61cd49b4784b3e0739ecebc2fd3228b8f719cec125706f2a"

PROFILE = {
    "name": "llama_swiglu_1b_v1",
    "n_layer": 18,
    "n_head": 16,
    "n_embd": 2048,
    "intermediate_size": 5504,
    "expected_parameter_count": 1_013_690_368,
    "expected_matrix_tensors": 126,
    "expected_backup_tensors": 38,
    "expected_preconditioner_groups": {
        "newton_full": 72,
        "down_diag": 72,
        "down_none": 54,
        "muon": 0,
        "adamw": 0,
    },
}
METHOD_ORDER = ("down_diag", "down_none", "newton_full", "muon", "adamw")
METHOD_SET = frozenset(METHOD_ORDER)


def load_base_runner() -> Any:
    spec = importlib.util.spec_from_file_location("llama_swiglu_1b_runner_base", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base runner from {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_runner()
original_subprocess_env = base.subprocess_env
BASE_TRAINER_SHA256 = base.sha256_file(BASE_TRAINER_PATH)
if BASE_TRAINER_SHA256 != PINNED_BASE_TRAINER_SHA256:
    raise RuntimeError(
        "scripts/17 LLaMA trainer differs from the version audited for the 1B profile: "
        f"{BASE_TRAINER_SHA256} != {PINNED_BASE_TRAINER_SHA256}"
    )


def default_output_root(stage: str) -> Path:
    artifact_root = HERE.parents[1]
    results_root = Path(
        os.environ.get("SNM_RESULTS_ROOT", str(artifact_root / "runs"))
    ).expanduser()
    return results_root / "20_llama_swiglu_1b" / stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Staged 1.014B LLaMA/SwiGLU quality and scale validation"
    )
    parser.add_argument("--stage", choices=("dry-run", "preflight", "probe", "smoke", "medium", "formal"), required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--python-exe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_SET), default=list(METHOD_ORDER))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--medium-steps", type=int, choices=(1000, 2000), default=1000)
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--medium-manifest", type=Path)
    parser.add_argument("--resume-batch", type=Path)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-train-log-every", type=int, default=20)
    parser.add_argument("--wandb-init-timeout", type=int, default=120)
    parser.add_argument("--checkpoint-every", type=int, default=128)
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--backup-lr", type=float, default=0.0036)
    parser.add_argument("--matrix-lr", type=float, default=0.01)
    parser.add_argument("--adamw-matrix-lr", type=float, default=0.000576)
    args = parser.parse_args()

    if len(args.methods) != len(set(args.methods)):
        parser.error("--methods contains duplicates")
    if 512 % args.device_batch_size:
        parser.error("--device-batch-size must divide the global batch size 512")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")

    args.execution_stage = args.stage
    args.dry_run = args.stage == "dry-run"
    args.preflight = args.stage == "preflight"
    args.numerical_smoke = args.stage in ("probe", "smoke")
    args.smoke_steps = 1 if args.stage == "probe" else 34
    args.expected_steps = {
        "dry-run": 6200,
        "preflight": 6200,
        "probe": 1,
        "smoke": 34,
        "medium": args.medium_steps,
        "formal": 6200,
    }[args.stage]
    args.output_root = args.output_root or default_output_root(args.stage)

    if args.wandb_mode is None:
        args.wandb_mode = "online" if args.stage in ("medium", "formal") else "disabled"
    if args.wandb_project is None:
        suffix = "Formal" if args.stage == "formal" else "Medium" if args.stage == "medium" else "Planning"
        args.wandb_project = f"Selective-Newton-Muon-MainConf-LLaMA-SwiGLU-1B-{suffix}-20260721"

    if args.stage in ("probe", "smoke") and args.wandb_mode != "disabled":
        parser.error("probe and smoke must use --wandb-mode disabled")
    if args.resume_batch and args.stage not in ("medium", "formal"):
        parser.error("only medium/formal batches can resume")
    if args.resume_batch and (args.smoke_manifest or args.medium_manifest):
        parser.error("resume uses the certificates already recorded in the saved plan")
    if not args.resume_batch and args.stage == "medium" and not args.smoke_manifest:
        parser.error("medium stage requires --smoke-manifest")
    if not args.resume_batch and args.stage == "formal":
        if not args.smoke_manifest or not args.medium_manifest:
            parser.error("formal stage requires both --smoke-manifest and --medium-manifest")
    if args.medium_manifest and args.stage != "formal":
        parser.error("--medium-manifest is only valid for formal stage")
    return args


def common_config(args: argparse.Namespace, smoke: bool) -> dict[str, Any]:
    stage = args.execution_stage
    steps = args.expected_steps
    short = stage in ("probe", "smoke")
    return {
        "num_iterations": steps,
        "global_batch_size": 512,
        "device_batch_size": args.device_batch_size,
        "sequence_length": 1024,
        "val_every": steps if short else 100,
        "val_tokens": args.device_batch_size * 1024 if short else 10_485_760,
        # Medium is a plateau-LR stability/cost screen, not a truncated formal
        # schedule. Formal alone carries the 1800-step warmdown.
        "warmdown_iters": 1 if short else 0 if stage == "medium" else 1800,
        "backup_lr": args.backup_lr,
        "matrix_lr": args.matrix_lr,
        "adamw_matrix_lr": args.adamw_matrix_lr,
        "checkpoint_every": 0 if short else args.checkpoint_every,
    }


def subprocess_env(official_repo: Path) -> dict[str, str]:
    # `base.subprocess_env` is monkey-patched to this wrapper below so that
    # base.main() uses the 1B-specific environment.  Keep and call the
    # original function captured before patching; calling through `base`
    # here would recurse indefinitely during preflight/runtime validation.
    env = original_subprocess_env(official_repo)
    env["LLAMA_1B_BASE_TRAINER"] = str(BASE_TRAINER_PATH.resolve())
    env["LLAMA_1B_BASE_TRAINER_SHA256"] = BASE_TRAINER_SHA256
    return env


def run_init_audit(
    args: argparse.Namespace, official_repo: Path, data_dir: Path, script: Path
) -> dict[str, Any]:
    observed: dict[str, dict[str, Any]] = {}
    for method in args.methods:
        result = base.subprocess.run(
            base.init_command(args, method, data_dir, script),
            env=subprocess_env(official_repo),
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"initialization audit failed for {method}:\n{result.stdout}\n{result.stderr}")
        lines = [line for line in result.stdout.splitlines() if line.startswith("LLAMA_INIT_AUDIT ")]
        if len(lines) != 1:
            raise RuntimeError(f"no unique init payload for {method}:\n{result.stdout}")
        observed[method] = json.loads(lines[0].split(" ", 1)[1])

    fingerprints = {payload["init_sha256"] for payload in observed.values()}
    if len(fingerprints) != 1:
        raise RuntimeError(f"method initialization fingerprints differ: {fingerprints}")
    for method, payload in observed.items():
        architecture = payload["architecture"]
        failures: list[str] = []
        expected = {
            "parameter_count": PROFILE["expected_parameter_count"],
            "matrix_tensor_count": PROFILE["expected_matrix_tensors"],
            "backup_tensor_count": PROFILE["expected_backup_tensors"],
            "preconditioner_group_count": PROFILE["expected_preconditioner_groups"][method],
        }
        for key, value in expected.items():
            if architecture.get(key) != value:
                failures.append(f"{key}={architecture.get(key)} expected={value}")
        observed_profile = architecture.get("profile", {})
        for key in ("name", "n_layer", "n_head", "n_embd", "intermediate_size", "expected_parameter_count"):
            if observed_profile.get(key) != PROFILE.get(key):
                failures.append(f"1B profile metadata mismatch for {key}")
        if architecture.get("base_trainer_sha256") != BASE_TRAINER_SHA256:
            failures.append("base trainer hash mismatch")
        if not architecture.get("embedding_head_tied"):
            failures.append("embedding/head are not tied")
        if architecture.get("bias_parameter_count") != 0:
            failures.append("bias parameters are present")
        if failures:
            raise RuntimeError(f"architecture audit failed for {method}: " + "; ".join(failures))

    k_bytes: dict[str, int] = {}
    for method, payload in observed.items():
        total = 0
        for group in payload["architecture"]["preconditioner_groups"]:
            width = int(group["input_width"])
            elements = width * width if group["kind"] == "dense" else width
            total += elements * 4 * 2
        k_bytes[method] = total
    return {
        "common_init_sha256": next(iter(fingerprints)),
        "methods": observed,
        "expected_k_state_bytes": k_bytes,
        "profile": PROFILE,
        "base_trainer_sha256": BASE_TRAINER_SHA256,
    }


def validate_summary(
    summary_path: Path,
    method: str,
    args: argparse.Namespace,
    runtime: dict[str, Any],
    init_audit: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    if not summary_path.is_file():
        raise RuntimeError(f"missing summary for {method}: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    expected_steps = args.expected_steps
    if payload.get("status") != "completed":
        failures.append("status is not completed")
    if payload.get("method") != method or payload.get("seed") != args.seed:
        failures.append("method/seed mismatch")
    if payload.get("completed_steps") != expected_steps:
        failures.append(f"completed_steps != {expected_steps}")
    if payload.get("init_sha256") != init_audit["common_init_sha256"]:
        failures.append("initialization fingerprint mismatch")
    architecture = payload.get("architecture", {})
    if architecture.get("parameter_count") != PROFILE["expected_parameter_count"]:
        failures.append("parameter count mismatch")
    if architecture.get("base_trainer_sha256") != BASE_TRAINER_SHA256:
        failures.append("base trainer source mismatch")
    for key in ("final_val_loss", "best_val_loss", "final_train_loss"):
        value = payload.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"{key} is non-finite")
    if payload.get("k_state_bytes") != init_audit["expected_k_state_bytes"][method]:
        failures.append("K-state byte count mismatch")
    if base.stable_runtime(payload.get("runtime", {})) != base.stable_runtime(runtime):
        failures.append("training runtime differs from preflight")
    config = common_config(args, smoke)
    try:
        base.validate_metric_evidence(
            summary_path.with_name("metrics.csv"),
            total_steps=expected_steps,
            val_every=config["val_every"],
            global_batch_size=config["global_batch_size"],
            sequence_length=config["sequence_length"],
        )
    except Exception as exc:
        failures.append(f"metric evidence invalid: {exc}")
    checkpoint = str(payload.get("checkpoint_path", ""))
    short = args.execution_stage in ("probe", "smoke")
    if short and checkpoint:
        failures.append("probe/smoke unexpectedly saved a checkpoint")
    if not short and (not checkpoint or not Path(checkpoint).is_file()):
        failures.append("medium/formal run has no final resumable checkpoint")
    if failures:
        raise RuntimeError(f"summary validation failed for {method}:\n- " + "\n- ".join(failures))
    return payload


def validate_stage_manifest(
    path: Path,
    expected_stage: str,
    args: argparse.Namespace,
    runtime: dict[str, Any],
    data_audit: dict[str, Any],
    init_audit: dict[str, Any],
    script_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    failures: list[str] = []
    if payload.get("status") != "completed" or payload.get("execution_stage") != expected_stage:
        failures.append(f"not a completed {expected_stage} manifest")
    if payload.get("seed") != args.seed:
        failures.append("seed differs")
    if payload.get("profile") != PROFILE:
        failures.append("1B profile differs")
    if payload.get("script_sha256") != script_sha256:
        failures.append("1B wrapper source differs")
    if payload.get("base_trainer_sha256") != BASE_TRAINER_SHA256:
        failures.append("base trainer source differs")
    if payload.get("data_audit", {}).get("fingerprint") != data_audit["fingerprint"]:
        failures.append("FineWeb data differs")
    if base.stable_runtime(payload.get("runtime", {})) != base.stable_runtime(runtime):
        failures.append("stable runtime differs")
    if payload.get("init_audit", {}).get("common_init_sha256") != init_audit["common_init_sha256"]:
        failures.append("initialization fingerprint differs")
    missing = set(args.methods) - set(payload.get("completed_methods", []))
    if missing:
        failures.append(f"certificate lacks requested methods: {sorted(missing)}")
    certificate_config = payload.get("config", {})
    current_config = common_config(args, args.execution_stage in ("probe", "smoke"))
    for key in (
        "global_batch_size",
        "device_batch_size",
        "sequence_length",
        "backup_lr",
        "matrix_lr",
        "adamw_matrix_lr",
    ):
        if certificate_config.get(key) != current_config.get(key):
            failures.append(
                f"configuration differs for {key}: "
                f"{certificate_config.get(key)} != {current_config.get(key)}"
            )
    if expected_stage == "medium" and int(payload.get("config", {}).get("num_iterations", 0)) < 1000:
        failures.append("medium certificate is shorter than 1000 steps")
    if failures:
        raise RuntimeError(f"{expected_stage} certificate validation failed:\n- " + "\n- ".join(failures))
    return payload


def validate_smoke_certificate(
    path: Path,
    args: argparse.Namespace,
    runtime: dict[str, Any],
    data_audit: dict[str, Any],
    script_sha256: str,
    init_audit: dict[str, Any],
) -> dict[str, Any]:
    smoke = validate_stage_manifest(path, "smoke", args, runtime, data_audit, init_audit, script_sha256)
    if args.execution_stage == "formal":
        validate_stage_manifest(
            args.medium_manifest, "medium", args, runtime, data_audit, init_audit, script_sha256
        )
    return smoke


def plan_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = original_plan_payload(*args, **kwargs)
    namespace = args[0]
    payload.update(
        {
            "family": "llama_swiglu_1b_scale_validation",
            "execution_stage": namespace.execution_stage,
            "evidence_class": "formal_quality" if namespace.execution_stage == "formal" else "screening_only",
            "profile": PROFILE,
            "base_trainer_sha256": BASE_TRAINER_SHA256,
            "timing_eligible": False,
        }
    )
    if namespace.medium_manifest:
        payload["medium_certificate_path"] = str(namespace.medium_manifest.resolve())
        payload["medium_certificate_sha256"] = base.sha256_file(namespace.medium_manifest.resolve())
    return payload


def validate_resume_plan(
    plan: dict[str, Any],
    args: argparse.Namespace,
    runtime: dict[str, Any],
    data_audit: dict[str, Any],
    init_audit: dict[str, Any],
    script_sha256: str,
) -> None:
    original_validate_resume_plan(plan, args, runtime, data_audit, init_audit, script_sha256)
    failures: list[str] = []
    if plan.get("execution_stage") != args.execution_stage:
        failures.append("execution stage differs")
    if plan.get("profile") != PROFILE:
        failures.append("1B profile differs")
    if plan.get("base_trainer_sha256") != BASE_TRAINER_SHA256:
        failures.append("base trainer source differs")
    if failures:
        raise RuntimeError("1B resume validation failed:\n- " + "\n- ".join(failures))


def print_plan(args: argparse.Namespace, data_dir: Path) -> None:
    print("LLaMA/SwiGLU 1.014B staged plan")
    print(f"stage:                {args.execution_stage}")
    print(f"profile:              {json.dumps(PROFILE, sort_keys=True)}")
    print(f"training interpreter: {args.python_exe}")
    print(f"official support repo:{args.official_repo.resolve()}")
    print(f"FineWeb data:         {data_dir}")
    print(f"seed:                 {args.seed}")
    print(f"methods:              {' -> '.join(args.methods)}")
    print(f"config:               {json.dumps(common_config(args, args.numerical_smoke), sort_keys=True)}")
    print(f"W&B:                  {args.wandb_mode} / {args.wandb_project}")
    print("timing_eligible:      false")


original_plan_payload = base.plan_payload
original_validate_resume_plan = base.validate_resume_plan
base.parse_args = parse_args
base.common_config = common_config
base.subprocess_env = subprocess_env
base.run_init_audit = run_init_audit
base.validate_summary = validate_summary
base.validate_smoke_certificate = validate_smoke_certificate
base.plan_payload = plan_payload
base.validate_resume_plan = validate_resume_plan
base.print_plan = print_plan
base.training_script_path = lambda: TRAINER_PATH.resolve()
base.EXPECTED_PARAMETER_COUNT = PROFILE["expected_parameter_count"]
base.EXPECTED_MATRIX_TENSORS = PROFILE["expected_matrix_tensors"]
base.EXPECTED_BACKUP_TENSORS = PROFILE["expected_backup_tensors"]
base.METHOD_ORDER = METHOD_ORDER
base.METHOD_SET = METHOD_SET


if __name__ == "__main__":
    base.main()
