"""Run the controlled official-architecture R1 comparison.

R1 keeps the pinned Newton-Muon-1 architecture/data/training recipe and fixes
the model initialization across four methods. Muon uses the official Muon LR;
block4/none/diag use the official Newton-Muon LR and differ only in the
mlp.c_proj K representation.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
R0_DIR = SCRIPT_DIR.parent / "14_official_newton_muon_r0"
sys.path.insert(0, str(R0_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import run_official_newton_muon_r0 as r0
from r1_source_builder import ALLOWED_METHODS, DerivedSource, build_source, self_test_diag_math


sys.path.insert(0, str(SCRIPT_DIR.parent / "_shared"))
from project_paths import EXPERIMENT_RESULTS_ROOT


FAMILY = "15_official_newton_muon_r1"
DEFAULT_PROJECT = "Selective-Newton-Muon-MainConf-OfficialR1-Controlled-20260717"
DEFAULT_RUN_PREFIX = "mainconf_official_r1"
LR_CROSS_FAMILY = "16_official_newton_muon_r1_lr_cross"
LR_CROSS_PROJECT = "Selective-Newton-Muon-MainConf-OfficialR1-LRCross-20260720"
LR_CROSS_RUN_PREFIX = "mainconf_official_r1_lr_cross"
HOST_BRIDGE_FAMILY = "21_gpt_r1_host_bridge"
HOST_BRIDGE_PROJECT = "Selective-Newton-Muon-MainConf-GPT-R1-HostBridge-20260721"
HOST_BRIDGE_RUN_PREFIX = "mainconf_gpt_r1_host_bridge"
R1_SMOKE_PROTOCOL = "official_newton_muon_1_r1_exact_shape_numerical_smoke"
R1_FORMAL_PROTOCOL = "official_newton_muon_1_r1_controlled_cproj_k"
LR_CROSS_SMOKE_PROTOCOL = (
    "official_newton_muon_1_r1_lr_cross_exact_shape_numerical_smoke"
)
LR_CROSS_FORMAL_PROTOCOL = "official_newton_muon_1_r1_muon_diag_2x2_lr_cross"
HOST_BRIDGE_SMOKE_PROTOCOL = (
    "official_newton_muon_1_r1_diag_none_host_bridge_exact_shape_smoke"
)
HOST_BRIDGE_FORMAL_PROTOCOL = (
    "official_newton_muon_1_r1_diag_none_host_bridge_formal"
)
TOKENS_PER_STEP = 512 * 1024
FULL_NUM_ITERATIONS = 6200
FULL_WARMDOWN_ITERS = 1800
TIMING_RESET_STEP = 32
LOSS_THRESHOLDS = (3.6, 3.5, 3.4, 3.3)
MILESTONE_STEPS = (1000, 3000, 4400, 6200)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    base_script: str
    cproj_k_mode: str
    base_learning_rate: float
    matrix_learning_rate: float
    role: str


@dataclass(frozen=True)
class RunProfile:
    name: str
    total_steps: int
    validation_steps: tuple[int, ...]
    formal_evidence: bool
    require_checkpoint: bool


FORMAL_PROFILE = RunProfile(
    name="formal",
    total_steps=FULL_NUM_ITERATIONS,
    validation_steps=tuple(range(0, FULL_NUM_ITERATIONS + 1, 100)),
    formal_evidence=True,
    require_checkpoint=True,
)


def numerical_smoke_profile(steps: int) -> RunProfile:
    return RunProfile(
        name="exact_shape_numerical_smoke",
        total_steps=steps,
        validation_steps=(0, steps),
        formal_evidence=False,
        require_checkpoint=False,
    )


METHODS = {
    "muon": MethodSpec(
        name="muon",
        base_script="train_gpt_muon_1.py",
        cproj_k_mode="muon",
        base_learning_rate=0.0036,
        matrix_learning_rate=0.00036,
        role="official_recipe_baseline",
    ),
    "block4": MethodSpec(
        name="block4",
        base_script="train_gpt_newton_muon_1.py",
        cproj_k_mode="block4",
        base_learning_rate=0.0040,
        matrix_learning_rate=0.00040,
        role="official_newton_muon_control",
    ),
    "none": MethodSpec(
        name="none",
        base_script="train_gpt_newton_muon_1.py",
        cproj_k_mode="none",
        base_learning_rate=0.0040,
        matrix_learning_rate=0.00040,
        role="cproj_k_ablation",
    ),
    "diag": MethodSpec(
        name="diag",
        base_script="train_gpt_newton_muon_1.py",
        cproj_k_mode="diag",
        base_learning_rate=0.0040,
        matrix_learning_rate=0.00040,
        role="cproj_diagonal_k",
    ),
}


# These are the two missing cells in the Muon-vs-diag 2x2 design. The other
# two cells are the completed/current R1 runs: Muon at 0.0036 and diag at
# 0.0040. Keeping the internal method names unchanged preserves the audited
# optimizer and c_proj implementations; the independent family/protocol and
# method roles prevent these runs from being confused with official recipes.
LR_CROSS_METHODS = {
    "muon": MethodSpec(
        name="muon",
        base_script="train_gpt_muon_1.py",
        cproj_k_mode="muon",
        base_learning_rate=0.0040,
        matrix_learning_rate=0.00040,
        role="lr_cross_muon_at_newton_lr",
    ),
    "diag": MethodSpec(
        name="diag",
        base_script="train_gpt_newton_muon_1.py",
        cproj_k_mode="diag",
        base_learning_rate=0.0036,
        matrix_learning_rate=0.00036,
        role="lr_cross_diag_at_muon_lr",
    ),
}


def experiment_family(args: argparse.Namespace) -> str:
    if getattr(args, "host_bridge", False):
        return HOST_BRIDGE_FAMILY
    return LR_CROSS_FAMILY if args.lr_cross else FAMILY


def experiment_protocol(args: argparse.Namespace, *, smoke: bool | None = None) -> str:
    is_smoke = args.numerical_smoke if smoke is None else smoke
    if getattr(args, "host_bridge", False):
        return HOST_BRIDGE_SMOKE_PROTOCOL if is_smoke else HOST_BRIDGE_FORMAL_PROTOCOL
    if args.lr_cross:
        return LR_CROSS_SMOKE_PROTOCOL if is_smoke else LR_CROSS_FORMAL_PROTOCOL
    return R1_SMOKE_PROTOCOL if is_smoke else R1_FORMAL_PROTOCOL


def experiment_specs(args: argparse.Namespace) -> dict[str, MethodSpec]:
    return LR_CROSS_METHODS if args.lr_cross else METHODS


def evidence_eligibility(args: argparse.Namespace) -> dict[str, object]:
    """Declare which evidence classes are valid for this run family."""
    if getattr(args, "host_bridge", False):
        return {
            "quality_usable": True,
            "memory_usable": True,
            "timing_usable": False,
            "reason": (
                "host bridge estimates quality/state only; node-level concurrency and "
                "cross-host execution invalidate wall-clock, throughput, and energy claims"
            ),
        }
    return {
        "quality_usable": True,
        "memory_usable": True,
        "timing_usable": False,
        "reason": (
            "LR-cross quality run may share the node"
            if args.lr_cross
            else "R1 is a quality/state experiment; formal timing belongs to R1-PERF"
        ),
    }


def visible_device_record(args: argparse.Namespace) -> dict[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if getattr(args, "host_bridge", False) and not args.dry_run and len(devices) != 1:
        raise RuntimeError(
            "--host-bridge requires CUDA_VISIBLE_DEVICES to name exactly one physical GPU; "
            f"observed {visible!r}. On the current host use: export CUDA_VISIBLE_DEVICES=1"
        )
    return {
        "cuda_visible_devices": visible or None,
        "visible_device_count": len(devices) if visible else None,
        "one_process_one_gpu": len(devices) == 1 if visible else False,
        "concurrent_node_training": bool(getattr(args, "concurrent_node_training", False)),
        "concurrent_workload": getattr(args, "concurrent_workload", None),
    }


META_RE = re.compile(
    r"^R1_METADATA method=(?P<method>\w+) cproj_k_mode=(?P<mode>\w+) "
    r"seed=(?P<seed>\d+) init_sha256=(?P<sha>[0-9a-f]{64})$"
)
K_MEMORY_RE = re.compile(
    r"^R1_K_MEMORY k_cov_bytes=(?P<k_cov>\d+) k_inv_bytes=(?P<k_inv>\d+) "
    r"k_state_bytes=(?P<k_state>\d+) activation_stat_bytes=(?P<activation>\d+) "
    r"precond_workspace_bytes=(?P<workspace>\d+) total_precond_bytes=(?P<total>\d+)$"
)
FINAL_MEMORY_RE = re.compile(
    r"^R1_FINAL_MEMORY optimizer_state_bytes=(?P<optimizer>\d+) "
    r"model_parameter_bytes=(?P<model>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled Muon/block4/none/diag on the official Newton-Muon-1 H100 setup."
    )
    parser.add_argument("--official-repo", type=Path, default=r0.default_official_repo())
    parser.add_argument("--python-exe", default="python")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--lr-cross",
        action="store_true",
        help=(
            "Run only the two missing Muon/diag 2x2 LR-cross cells: "
            "Muon at the Newton LR and diag at the Muon LR."
        ),
    )
    parser.add_argument(
        "--host-bridge",
        action="store_true",
        help=(
            "Run the GPT R1 diag/none bridge on the LLaMA host. This is a separate "
            "quality/state family and never produces timing evidence."
        ),
    )
    parser.add_argument(
        "--concurrent-node-training",
        action="store_true",
        help="Record that another physical GPU on the same node is training concurrently.",
    )
    parser.add_argument(
        "--concurrent-workload",
        default=None,
        help="Short audit label for the other node workload, e.g. llama_swiglu_seed2024_gpu0.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS),
        default=None,
        help=(
            "Default R1: diag none block4 muon. Default --lr-cross: muon diag. "
            "Use a subset only for a documented retry."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate source/data/runtime and audit identical initialization fingerprints; do not train.",
    )
    parser.add_argument(
        "--numerical-smoke",
        action="store_true",
        help=(
            "Run an exact-formal-shape finite numerical smoke for every selected method, "
            "without checkpoints or W&B; not formal evidence."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        dest="numerical_smoke",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--smoke-steps", type=int, default=10)
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        default=None,
        help="Required for formal R1; must be a matching exact-shape R1 smoke manifest.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--resume-batch",
        type=Path,
        default=None,
        help=(
            "Resume an existing R1 batch directory. Valid completed methods are revalidated "
            "and skipped; only an interrupted/failed method is restarted from step zero."
        ),
    )
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb-train-log-every", type=int, default=20)
    parser.add_argument("--wandb-init-timeout", type=int, default=120)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.methods is None:
        args.methods = (
            ["diag", "none"]
            if args.host_bridge
            else ["muon", "diag"]
            if args.lr_cross
            else ["diag", "none", "block4", "muon"]
        )
    if args.host_bridge and args.lr_cross:
        parser.error("--host-bridge and --lr-cross are mutually exclusive")
    if args.host_bridge and (set(args.methods) - {"diag", "none"}):
        parser.error("--host-bridge supports only --methods diag none")
    if args.lr_cross and set(args.methods) - set(LR_CROSS_METHODS):
        parser.error("--lr-cross supports only --methods muon diag")
    if args.results_dir is None:
        args.results_dir = EXPERIMENT_RESULTS_ROOT / experiment_family(args) / "results"
    if args.run_prefix is None:
        args.run_prefix = (
            HOST_BRIDGE_RUN_PREFIX
            if args.host_bridge
            else LR_CROSS_RUN_PREFIX
            if args.lr_cross
            else DEFAULT_RUN_PREFIX
        )
    if args.wandb_project is None:
        args.wandb_project = (
            HOST_BRIDGE_PROJECT
            if args.host_bridge
            else LR_CROSS_PROJECT
            if args.lr_cross
            else DEFAULT_PROJECT
        )
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.wandb_train_log_every <= 0:
        parser.error("--wandb-train-log-every must be positive")
    if args.wandb_init_timeout <= 0:
        parser.error("--wandb-init-timeout must be positive")
    if args.smoke_steps < 2:
        parser.error("--smoke-steps must be at least 2 so the second optimizer update is exercised")
    if args.host_bridge and args.numerical_smoke and args.smoke_steps < 34:
        parser.error("host-bridge smoke must use at least 34 steps to cross the K refresh at step 32")
    if len(args.methods) != len(set(args.methods)):
        parser.error("--methods contains duplicates")
    if args.preflight and args.numerical_smoke:
        parser.error("choose either --preflight or --numerical-smoke")
    if args.concurrent_node_training and not args.host_bridge:
        parser.error("--concurrent-node-training is currently only valid with --host-bridge")
    if args.concurrent_workload and not args.concurrent_node_training:
        parser.error("--concurrent-workload requires --concurrent-node-training")
    if not (
        args.dry_run
        or args.preflight
        or args.numerical_smoke
        or args.resume_batch is not None
    ) and args.smoke_manifest is None:
        parser.error("formal R1 requires --smoke-manifest from a matching exact-shape smoke")
    if args.resume_batch is not None and (args.dry_run or args.preflight):
        parser.error("--resume-batch cannot be combined with --dry-run or --preflight")
    return args


def command_text(command: Iterable[str]) -> str:
    return shlex.join([str(part) for part in command])


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_wandb_online_access(enabled: bool) -> dict[str, object]:
    if not enabled:
        return {"required": False, "status": "not_checked"}
    try:
        import wandb

        api = wandb.Api(timeout=30)
        viewer = api.viewer
        if callable(viewer):
            viewer = viewer()
        if not viewer:
            raise RuntimeError("W&B API returned no authenticated viewer")
        return {
            "required": True,
            "status": "authenticated_online",
            "base_url": os.environ.get("WANDB_BASE_URL", "default"),
        }
    except Exception as exc:
        raise RuntimeError(
            "W&B online readiness check failed in the controller runtime. "
            "Fix login/network before starting formal R1 so the validated local runs can be uploaded: "
            f"{exc!r}"
        ) from exc


def override_derived_learning_rate(
    repo: Path,
    derived: DerivedSource,
    *,
    original_literal: str,
    crossed_literal: str,
) -> DerivedSource:
    """Create an auditable LR-cross source with one exact LR replacement."""
    old = f"    learning_rate : float = {original_literal}\n"
    new = f"    learning_rate : float = {crossed_literal}\n"
    count = derived.source.count(old)
    if count != 1:
        raise RuntimeError(
            f"R1 LR-cross expected one {old.strip()!r} anchor for {derived.method}, "
            f"observed {count}"
        )
    crossed_source = derived.source.replace(old, new, 1)
    compile(crossed_source, f"<R1-LR-cross-{derived.method}>", "exec")
    base_source = (
        (repo / derived.base_script)
        .read_bytes()
        .replace(b"\r\n", b"\n")
        .decode("utf-8")
    )
    diff = "".join(
        difflib.unified_diff(
            base_source.splitlines(keepends=True),
            crossed_source.splitlines(keepends=True),
            fromfile=f"official/{derived.base_script}",
            tofile=f"r1_lr_cross/train_r1_{derived.method}.py",
        )
    )
    return DerivedSource(
        method=derived.method,
        base_script=derived.base_script,
        base_canonical_sha256=derived.base_canonical_sha256,
        derived_sha256=hashlib.sha256(crossed_source.encode("utf-8")).hexdigest(),
        source=crossed_source,
        unified_diff=diff,
    )


def build_all_sources(repo: Path, *, lr_cross: bool = False) -> dict[str, DerivedSource]:
    self_test_diag_math()
    method_names = ("muon", "diag") if lr_cross else ALLOWED_METHODS
    built = {method: build_source(repo, method) for method in method_names}
    if lr_cross:
        built["muon"] = override_derived_learning_rate(
            repo,
            built["muon"],
            original_literal="0.0036",
            crossed_literal="0.0040",
        )
        built["diag"] = override_derived_learning_rate(
            repo,
            built["diag"],
            original_literal="0.0040",
            crossed_literal="0.0036",
        )
    expected_hashes = r0.EXPECTED_CANONICAL_SHA256
    for method, derived in built.items():
        expected = expected_hashes[derived.base_script]
        if derived.base_canonical_sha256 != expected:
            raise RuntimeError(
                f"R1 {method} base hash mismatch: {derived.base_canonical_sha256} != {expected}"
            )
    if not lr_cross and len(
        {built[name].derived_sha256 for name in ("block4", "none", "diag")}
    ) != 1:
        raise RuntimeError("Newton R1 variants must share one parameterized derived source")
    return built


def source_fingerprints(built: dict[str, DerivedSource]) -> dict[str, str]:
    return {method: derived.derived_sha256 for method, derived in built.items()}


def validate_smoke_manifest(
    path: Path,
    runtime: dict[str, object],
    methods: list[str],
    seed: int,
    built: dict[str, DerivedSource],
    expected_protocol: str = R1_SMOKE_PROTOCOL,
    expected_cuda_visible_devices: str | None = None,
) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"R1 smoke manifest not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    failures: list[str] = []
    if payload.get("protocol") != expected_protocol:
        failures.append("manifest protocol is not the exact-shape R1 numerical smoke")
    if payload.get("official_commit") not in (None, r0.OFFICIAL_COMMIT):
        failures.append("official commit does not match the pinned R1 base commit")
    if payload.get("seed") != seed:
        failures.append(f"controlled seed differs: smoke={payload.get('seed')!r}, formal={seed}")
    if payload.get("failures"):
        failures.append("smoke manifest contains failures")
    if not payload.get("initialization_audit", {}).get("all_methods_identical"):
        failures.append("smoke initialization audit did not prove identical initial parameters")

    summaries = payload.get("summaries", [])
    completed = {
        str(item.get("method"))
        for item in summaries
        if isinstance(item, dict) and item.get("evidence_valid") is True
    }
    missing_methods = sorted(set(methods) - completed)
    if missing_methods:
        failures.append(f"smoke did not validate requested methods: {missing_methods}")

    observed_runtime = r0.normalize_runtime_fingerprint(
        payload.get("training_runtime_fingerprint")
    )
    expected_runtime = r0.normalize_runtime_fingerprint(r0.runtime_fingerprint(runtime))
    if observed_runtime != expected_runtime:
        failures.append(
            "training runtime fingerprint differs from the R1 smoke certificate; "
            f"observed={observed_runtime!r}, current={expected_runtime!r}"
        )

    if expected_cuda_visible_devices is not None:
        resource_isolation = payload.get("resource_isolation")
        observed_visible = (
            resource_isolation.get("cuda_visible_devices")
            if isinstance(resource_isolation, dict)
            else None
        )
        if observed_visible != expected_cuda_visible_devices:
            failures.append(
                "CUDA_VISIBLE_DEVICES differs from the host-bridge smoke certificate; "
                f"observed={observed_visible!r}, current={expected_cuda_visible_devices!r}"
            )

    observed_sources = payload.get("derived_source_sha256")
    expected_sources = source_fingerprints(built)
    if not isinstance(observed_sources, dict):
        failures.append("smoke manifest has no derived-source fingerprints")
    else:
        mismatched_sources = [
            method
            for method in methods
            if observed_sources.get(method) != expected_sources[method]
        ]
        if mismatched_sources:
            failures.append(f"derived R1 source differs for methods: {mismatched_sources}")

    if failures:
        raise RuntimeError("R1 smoke certificate check failed:\n- " + "\n- ".join(failures))
    return {
        "path": str(resolved),
        "validated": True,
        "methods": sorted(completed),
        "seed": seed,
        "cuda_visible_devices": expected_cuda_visible_devices,
    }


def validate_resume_plan(
    payload: dict[str, object],
    args: argparse.Namespace,
    runtime: dict[str, object],
    built: dict[str, DerivedSource],
    init_audit: dict[str, object],
) -> None:
    failures: list[str] = []
    if payload.get("family") != experiment_family(args):
        failures.append("batch family is not R1")
    if payload.get("seed") != args.seed:
        failures.append(f"seed differs: batch={payload.get('seed')!r}, requested={args.seed}")
    if payload.get("methods") != args.methods:
        failures.append(
            f"method order/set differs: batch={payload.get('methods')!r}, requested={args.methods!r}"
        )
    expected_protocol = experiment_protocol(args)
    if payload.get("protocol") != expected_protocol:
        failures.append(
            f"batch protocol differs: {payload.get('protocol')!r} != {expected_protocol!r}"
        )
    if payload.get("official_commit") != r0.OFFICIAL_COMMIT:
        failures.append("official commit differs")
    if payload.get("run_prefix", args.run_prefix) != args.run_prefix:
        failures.append("run prefix differs")
    if getattr(args, "host_bridge", False):
        prior_isolation = payload.get("resource_isolation")
        current_isolation = visible_device_record(args)
        if not isinstance(prior_isolation, dict):
            failures.append("host-bridge batch has no resource-isolation record")
        elif prior_isolation.get("cuda_visible_devices") != current_isolation.get(
            "cuda_visible_devices"
        ):
            failures.append("host-bridge CUDA_VISIBLE_DEVICES differs from interrupted batch")
    observed_runtime = r0.normalize_runtime_fingerprint(
        payload.get("training_runtime_fingerprint")
    )
    expected_runtime = r0.normalize_runtime_fingerprint(r0.runtime_fingerprint(runtime))
    if observed_runtime != expected_runtime:
        failures.append("training runtime differs from the interrupted batch")
    if payload.get("derived_source_sha256") != source_fingerprints(built):
        failures.append("derived source fingerprints differ from the interrupted batch")
    prior_audit = payload.get("initialization_audit")
    if not isinstance(prior_audit, dict) or prior_audit.get("init_sha256") != init_audit.get("init_sha256"):
        failures.append("initialization fingerprint differs from the interrupted batch")
    if failures:
        raise RuntimeError("R1 resume validation failed:\n- " + "\n- ".join(failures))


def print_plan(args: argparse.Namespace, repo: Path, built: dict[str, DerivedSource]) -> None:
    specs = experiment_specs(args)
    print(f"Official repository: {repo}")
    print(f"Pinned commit:       {r0.OFFICIAL_COMMIT}")
    print(f"Experiment family:   {experiment_family(args)}")
    print(f"Protocol:            {experiment_protocol(args)}")
    print(f"Controlled seed:     {args.seed}")
    print(f"W&B project:         {args.wandb_project}")
    if getattr(args, "host_bridge", False):
        isolation = visible_device_record(args)
        print(f"Visible physical GPU: {isolation['cuda_visible_devices']}")
        print("Evidence policy:      quality/state usable; timing always ineligible")
    print(
        "Run mode:            "
        + (
            f"exact-shape numerical smoke ({args.smoke_steps} steps)"
            if args.numerical_smoke
            else "formal R1"
        )
    )
    if args.resume_batch is not None:
        print(f"Resume batch:        {args.resume_batch}")
    print("Planned jobs:")
    for index, method in enumerate(args.methods, start=1):
        spec = specs[method]
        print(
            f"  {index}. {method:6s} | c_proj={spec.cproj_k_mode:6s} "
            f"base_lr={spec.base_learning_rate:g} matrix_lr={spec.matrix_learning_rate:g} "
            f"derived_sha256={built[method].derived_sha256[:12]}"
        )


def controlled_env(
    args: argparse.Namespace,
    spec: MethodSpec,
    data_dir: Path,
    *,
    init_only: bool = False,
    smoke_test: bool = False,
    smoke_steps: int = 10,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": str(args.seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "R1_METHOD": spec.name,
            "R1_CPROJ_K_MODE": spec.cproj_k_mode,
            "R1_SEED": str(args.seed),
            "R1_DATA_DIR": str(data_dir.resolve()),
            "R1_INIT_ONLY": "1" if init_only else "0",
            "R1_SMOKE_TEST": "1" if smoke_test else "0",
            "R1_SMOKE_STEPS": str(smoke_steps),
            "R1_DISABLE_CHECKPOINT": "1" if smoke_test or init_only else "0",
        }
    )
    return env


def materialize_source(
    directory: Path,
    repo: Path,
    derived: DerivedSource,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    script_path = directory / f"train_r1_{derived.method}.py"
    script_path.write_text(derived.source, encoding="utf-8", newline="\n")
    shutil.copy2(repo / "triton_kernels.py", directory / "triton_kernels.py")
    return script_path


def parse_metadata(stdout: str) -> dict[str, object]:
    matches = [META_RE.match(line) for line in stdout.splitlines()]
    matches = [match for match in matches if match is not None]
    if len(matches) != 1:
        raise RuntimeError(f"expected one R1_METADATA line, observed {len(matches)}")
    match = matches[0]
    return {
        "method": match.group("method"),
        "cproj_k_mode": match.group("mode"),
        "seed": int(match.group("seed")),
        "init_sha256": match.group("sha"),
    }


def initialization_audit(
    args: argparse.Namespace,
    repo: Path,
    data_dir: Path,
    built: dict[str, DerivedSource],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    specs = experiment_specs(args)
    with tempfile.TemporaryDirectory(prefix="official_r1_init_audit_") as temp:
        root = Path(temp)
        for method in built:
            spec = specs[method]
            workspace = root / method
            script = materialize_source(workspace, repo, built[method])
            result = subprocess.run(
                [args.python_exe, script.name],
                cwd=workspace,
                env=controlled_env(args, spec, data_dir, init_only=True),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=900,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"R1 initialization audit failed for {method}:\n{result.stdout[-8000:]}"
                )
            metadata = parse_metadata(result.stdout)
            if metadata["method"] != method or metadata["seed"] != args.seed:
                raise RuntimeError(f"bad R1 initialization metadata for {method}: {metadata}")
            records.append(metadata)
            print(f"Initialization audit {method}: {metadata['init_sha256']}")
    fingerprints = {str(record["init_sha256"]) for record in records}
    if len(fingerprints) != 1:
        raise RuntimeError(f"R1 initial model fingerprints differ across methods: {records}")
    return {
        "seed": args.seed,
        "all_methods_identical": True,
        "init_sha256": next(iter(fingerprints)),
        "records": records,
    }


def lr_multiplier(step: int, total_steps: int) -> float:
    warmdown = 1 if total_steps != FULL_NUM_ITERATIONS else FULL_WARMDOWN_ITERS
    if step < total_steps - warmdown:
        return 1.0
    return max(0.0, (total_steps - step) / warmdown)


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def parse_key_value_line(stdout: str, regex: re.Pattern[str], label: str) -> dict[str, int]:
    matches = [regex.match(line) for line in stdout.splitlines()]
    matches = [match for match in matches if match is not None]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label} line, observed {len(matches)}")
    return {key: int(value) for key, value in matches[0].groupdict().items()}


def curve_mean(rows: list[dict[str, object]]) -> float:
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    if len(ordered) < 2:
        return math.nan
    area = 0.0
    for left, right in zip(ordered, ordered[1:]):
        width = int(right["step"]) - int(left["step"])
        area += width * (float(left["loss"]) + float(right["loss"])) / 2.0
    span = int(ordered[-1]["step"]) - int(ordered[0]["step"])
    return area / span if span > 0 else math.nan


def threshold_key(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def validate_metric_evidence(
    rows: list[dict[str, object]],
    spec: MethodSpec,
    profile: RunProfile,
    metadata: dict[str, object],
    k_memory: dict[str, int],
    final_memory: dict[str, int],
    peak_mib: int,
    checkpoint_path: Path | None,
    expected_seed: int,
    expected_init_sha256: str,
) -> None:
    failures: list[str] = []
    val_rows = sorted(
        [row for row in rows if row["event"] == "validation"],
        key=lambda row: int(row["step"]),
    )
    train_rows = sorted(
        [row for row in rows if row["event"] == "train"],
        key=lambda row: int(row["step"]),
    )
    observed_val_steps = tuple(int(row["step"]) for row in val_rows)
    observed_train_steps = tuple(int(row["step"]) for row in train_rows)
    expected_train_steps = tuple(range(1, profile.total_steps + 1))
    if observed_val_steps != profile.validation_steps:
        failures.append(
            f"validation steps mismatch: observed count={len(observed_val_steps)}, "
            f"first={observed_val_steps[:3]}, last={observed_val_steps[-3:] if observed_val_steps else ()}; "
            f"expected={profile.validation_steps}"
        )
    if observed_train_steps != expected_train_steps:
        failures.append(
            f"train steps mismatch: observed count={len(observed_train_steps)}, "
            f"first={observed_train_steps[:3]}, last={observed_train_steps[-3:] if observed_train_steps else ()}; "
            f"expected 1..{profile.total_steps}"
        )
    wrong_totals = sorted(
        {int(row["total_steps"]) for row in rows if int(row["total_steps"]) != profile.total_steps}
    )
    if wrong_totals:
        failures.append(
            f"unexpected total_steps fields: {wrong_totals}; expected {profile.total_steps}"
        )

    nonfinite = [
        {"event": row["event"], "step": row["step"], "loss": row["loss"]}
        for row in rows
        if not math.isfinite(float(row["loss"]))
    ]
    if nonfinite:
        first = nonfinite[0]
        raise r0.EvidenceValidationError(
            "invalid_nonfinite",
            f"non-finite {first['event']} loss at step {first['step']}: {first['loss']}",
            {"first_nonfinite": first, "nonfinite_count": len(nonfinite)},
        )

    if metadata.get("method") != spec.name:
        failures.append(f"metadata method mismatch: {metadata.get('method')!r} != {spec.name!r}")
    if metadata.get("cproj_k_mode") != spec.cproj_k_mode:
        failures.append(
            f"metadata c_proj mode mismatch: {metadata.get('cproj_k_mode')!r} != {spec.cproj_k_mode!r}"
        )
    if metadata.get("seed") != expected_seed:
        failures.append(f"metadata seed mismatch: {metadata.get('seed')!r} != {expected_seed}")
    if metadata.get("init_sha256") != expected_init_sha256:
        failures.append("formal model initialization SHA differs from the audited initialization")

    if any(int(row["official_train_time_ms"]) < 0 for row in rows):
        failures.append("negative official training time")
    post_reset_train_times = [
        int(row["official_train_time_ms"])
        for row in train_rows
        if int(row["step"]) >= 33
    ]
    if any(right < left for left, right in zip(post_reset_train_times, post_reset_train_times[1:])):
        failures.append("train_time decreases after the official step-32 timing reset")
    val_times = [
        int(row["official_train_time_ms"])
        for row in val_rows
        if int(row["step"]) >= min(100, profile.total_steps)
    ]
    if any(right < left for left, right in zip(val_times, val_times[1:])):
        failures.append("validation train_time is not monotone")

    if peak_mib <= 0:
        failures.append(f"missing or invalid peak memory: {peak_mib}")
    if final_memory["optimizer"] <= 0 or final_memory["model"] <= 0:
        failures.append(f"invalid final optimizer/model memory report: {final_memory}")
    if any(value < 0 for value in k_memory.values()):
        failures.append(f"negative K/preconditioner memory report: {k_memory}")
    if k_memory["total"] < k_memory["k_state"]:
        failures.append("total preconditioner bytes are smaller than persistent K-state bytes")
    if spec.name == "muon" and any(k_memory.values()):
        failures.append(f"Muon unexpectedly reports Newton K/preconditioner state: {k_memory}")
    if spec.name != "muon" and k_memory["total"] <= 0:
        failures.append(f"Newton variant has no reported preconditioner state: {k_memory}")
    if profile.require_checkpoint and (
        checkpoint_path is None or not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0
    ):
        failures.append("formal R1 is missing its final checkpoint")
    if not profile.require_checkpoint and checkpoint_path is not None:
        failures.append("numerical smoke unexpectedly produced a checkpoint")

    if failures:
        raise r0.EvidenceValidationError(
            "invalid_incomplete",
            "R1 evidence validation failed:\n- " + "\n- ".join(failures),
            {
                "observed_validation_points": len(val_rows),
                "observed_train_points": len(train_rows),
                "expected_validation_points": len(profile.validation_steps),
                "expected_train_points": profile.total_steps,
            },
        )


def parse_metrics(
    log_path: Path,
    stdout_path: Path,
    spec: MethodSpec,
    checkpoint_path: Path | None,
    profile: RunProfile,
    expected_seed: int,
    expected_init_sha256: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = r0.VAL_RE.match(line) or r0.TRAIN_RE.match(line)
        if match is None:
            continue
        step = int(match.group("step"))
        total = int(match.group("total"))
        event = "validation" if "val_loss:" in line else "train"
        multiplier = lr_multiplier(step, total)
        rows.append(
            {
                "method": spec.name,
                "cproj_k_mode": spec.cproj_k_mode,
                "event": event,
                "step": step,
                "total_steps": total,
                "tokens_seen": step * TOKENS_PER_STEP,
                "loss": float(match.group("loss")),
                "official_train_time_ms": int(match.group("time")),
                "step_avg_ms": parse_float(match.group("avg")),
                "lr_multiplier": multiplier,
                "adamw_lr": spec.base_learning_rate * multiplier,
                "matrix_lr": spec.matrix_learning_rate * multiplier,
            }
        )

    val_rows = sorted(
        [row for row in rows if row["event"] == "validation"],
        key=lambda row: int(row["step"]),
    )
    train_rows = sorted(
        [row for row in rows if row["event"] == "train"],
        key=lambda row: int(row["step"]),
    )
    if not val_rows or not train_rows:
        raise RuntimeError(f"could not recover R1 loss rows from {log_path}")

    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    metadata = parse_metadata(stdout)
    k_memory = parse_key_value_line(stdout, K_MEMORY_RE, "R1_K_MEMORY")
    final_memory = parse_key_value_line(stdout, FINAL_MEMORY_RE, "R1_FINAL_MEMORY")
    peak_matches = [r0.PEAK_RE.match(line) for line in stdout.splitlines()]
    peak_matches = [match for match in peak_matches if match is not None]
    if len(peak_matches) != 1:
        raise RuntimeError(f"expected one peak-memory line, observed {len(peak_matches)}")
    peak_mib = int(peak_matches[0].group("mib"))

    validate_metric_evidence(
        rows,
        spec,
        profile,
        metadata,
        k_memory,
        final_memory,
        peak_mib,
        checkpoint_path,
        expected_seed,
        expected_init_sha256,
    )

    final_val = val_rows[-1]
    best_val = min(val_rows, key=lambda row: float(row["loss"]))
    final_train = train_rows[-1]
    timed_updates = max(0, int(final_val["step"]) - TIMING_RESET_STEP)
    official_train_time_s = float(final_val["official_train_time_ms"]) / 1000.0
    summary: dict[str, object] = {
        "method": spec.name,
        "method_role": spec.role,
        "cproj_k_mode": spec.cproj_k_mode,
        "controlled_seed": metadata["seed"],
        "init_sha256": metadata["init_sha256"],
        "base_learning_rate": spec.base_learning_rate,
        "matrix_learning_rate": spec.matrix_learning_rate,
        "final_val_step": final_val["step"],
        "final_val_loss": final_val["loss"],
        "best_val_step": best_val["step"],
        "best_val_loss": best_val["loss"],
        "final_train_step": final_train["step"],
        "final_train_loss": final_train["loss"],
        "val_curve_mean": curve_mean(val_rows),
        "official_train_time_s": official_train_time_s,
        "timed_updates": timed_updates,
        "timed_tokens": timed_updates * TOKENS_PER_STEP,
        "timed_tokens_per_s": (
            timed_updates * TOKENS_PER_STEP / official_train_time_s
            if official_train_time_s > 0
            else math.nan
        ),
        "peak_memory_allocated_mib": peak_mib,
        "k_cov_bytes": k_memory["k_cov"],
        "k_inv_bytes": k_memory["k_inv"],
        "k_state_bytes": k_memory["k_state"],
        "activation_stat_bytes": k_memory["activation"],
        "precond_workspace_bytes": k_memory["workspace"],
        "total_precond_bytes": k_memory["total"],
        "optimizer_state_bytes": final_memory["optimizer"],
        "model_parameter_bytes": final_memory["model"],
        "checkpoint_bytes": checkpoint_path.stat().st_size if checkpoint_path else 0,
        "validation_points": len(val_rows),
        "train_points": len(train_rows),
        "evidence_profile": profile.name,
        "formal_evidence": profile.formal_evidence,
        "evidence_valid": True,
    }
    for milestone in MILESTONE_STEPS:
        matching = [row for row in val_rows if int(row["step"]) == milestone]
        if matching:
            summary[f"val_loss_step_{milestone}"] = matching[0]["loss"]
    for threshold in LOSS_THRESHOLDS:
        reached = next((row for row in val_rows if float(row["loss"]) <= threshold), None)
        suffix = threshold_key(threshold)
        summary[f"step_to_val_loss_{suffix}"] = int(reached["step"]) if reached else -1
        summary[f"tokens_to_val_loss_{suffix}"] = int(reached["tokens_seen"]) if reached else -1
        summary[f"train_time_s_to_val_loss_{suffix}"] = (
            float(reached["official_train_time_ms"]) / 1000.0 if reached else -1.0
        )
    return rows, summary


def upload_to_wandb(
    args: argparse.Namespace,
    run_dir: Path,
    run_name: str,
    group: str,
    spec: MethodSpec,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    provenance: dict[str, object],
    runtime: dict[str, object],
    derived: DerivedSource,
) -> dict[str, object]:
    if args.wandb_mode == "disabled" or args.numerical_smoke:
        return {"status": "disabled"}
    try:
        import wandb

        config = {
            "experiment_family": experiment_family(args),
            "protocol": experiment_protocol(args, smoke=False),
            "method": spec.name,
            "method_role": spec.role,
            "cproj_k_mode": spec.cproj_k_mode,
            "seed": args.seed,
            "seed_policy": "python_numpy_torch_cuda_before_model_init",
            "resource_isolation": visible_device_record(args),
            "evidence_eligibility": evidence_eligibility(args),
            "official_repository_url": r0.OFFICIAL_REPOSITORY_URL,
            "official_commit": r0.OFFICIAL_COMMIT,
            "official_base_script": spec.base_script,
            "official_base_canonical_sha256": derived.base_canonical_sha256,
            "derived_script_sha256": derived.derived_sha256,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "batch_size_sequences": 512,
            "device_batch_size_sequences": 64,
            "sequence_length": 1024,
            "tokens_per_step": TOKENS_PER_STEP,
            "num_iterations": FULL_NUM_ITERATIONS,
            "warmup_iters": 0,
            "warmdown_iters": FULL_WARMDOWN_ITERS,
            "val_loss_every": 100,
            "val_tokens": 10485760,
            "base_learning_rate": spec.base_learning_rate,
            "matrix_learning_rate": spec.matrix_learning_rate,
            "learning_rate_design": (
                "muon_diag_2x2_absolute_lr_cross_missing_cell"
                if args.lr_cross
                else "official_method_specific_recipe"
            ),
            "learning_rate_cell": spec.role,
            "muon_lr_policy": (
                "newton_absolute_lr_cross_cell"
                if args.lr_cross and spec.name == "muon"
                else "official_method_specific_recipe"
            ),
            "newton_variant_control": (
                "diag_at_muon_absolute_lr_cross_cell"
                if args.lr_cross and spec.name == "diag"
                else "shared_lr_and_only_cproj_k_mode_differs"
            ),
            "timing_inference_policy": evidence_eligibility(args)["reason"],
            "wandb_upload_timing": "after_training_completed",
            "wandb_tables_enabled": False,
            "gpu_name": runtime.get("gpu_name"),
        }
        tags = [
            "publication",
            "official_architecture",
            "newton_muon_r1",
            "controlled_seed",
            f"seed{args.seed}",
            "h100",
            spec.name,
            f"cproj_{spec.cproj_k_mode}",
        ]
        if args.lr_cross:
            tags.extend(["lr_cross", "muon_diag_2x2", spec.role])
        if args.host_bridge:
            tags.extend(["host_bridge", "gpt_on_llama_host", "timing_ineligible"])
        wandb_run_id = hashlib.sha256(run_name.encode("utf-8")).hexdigest()[:12]
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=wandb_run_id,
            resume="allow",
            name=run_name,
            group=group,
            mode=args.wandb_mode,
            dir=str(run_dir),
            tags=tags,
            config=config,
            reinit=True,
            settings=wandb.Settings(init_timeout=args.wandb_init_timeout),
        )
        per_step: dict[int, dict[str, float]] = {}
        final_train_step = max(
            int(row["step"]) for row in rows if str(row["event"]) == "train"
        )
        for row in rows:
            step = int(row["step"])
            event = str(row["event"])
            if event == "train" and step % args.wandb_train_log_every != 0 and step != final_train_step:
                continue
            values = per_step.setdefault(step, {})
            values["time/train_s"] = float(row["official_train_time_ms"]) / 1000.0
            step_avg_ms = float(row["step_avg_ms"])
            if math.isfinite(step_avg_ms):
                values["performance/step_avg_ms"] = step_avg_ms
            values["lr/adamw"] = float(row["adamw_lr"])
            values["lr/matrix"] = float(row["matrix_lr"])
            if event == "validation":
                values["val/loss"] = float(row["loss"])
            else:
                values["train/loss_step"] = float(row["loss"])
        final_step = int(summary["final_val_step"])
        memory_values = per_step.setdefault(final_step, {})
        memory_values["memory/peak_allocated_mib"] = float(summary["peak_memory_allocated_mib"])
        memory_values["memory/k_state_mib"] = float(summary["k_state_bytes"]) / (1024**2)
        memory_values["memory/optimizer_state_mib"] = float(summary["optimizer_state_bytes"]) / (1024**2)
        for step in sorted(per_step):
            wandb.log(per_step[step], step=step)
        for key, value in summary.items():
            if isinstance(value, (int, float, str, bool)):
                run.summary[key] = value
        run.summary["official_commit"] = provenance["official_commit"]
        run.finish()
        return {
            "status": "uploaded",
            "mode": args.wandb_mode,
            "run_name": run_name,
            "run_id": getattr(run, "id", None),
            "run_url": getattr(run, "url", None),
        }
    except Exception as exc:
        return {"status": "failed", "error": repr(exc), "run_name": run_name}


def find_single_log(workspace: Path) -> Path:
    logs = list((workspace / "logs").glob("*.txt"))
    if len(logs) != 1:
        raise RuntimeError(f"expected one R1 log in {workspace}, observed {len(logs)}")
    return logs[0]


def find_checkpoint(workspace: Path) -> Path | None:
    checkpoints = list((workspace / "logs").glob("*/state_step*.pt"))
    if not checkpoints:
        return None
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected at most one final checkpoint, observed {len(checkpoints)}")
    return checkpoints[0]


def base_run_name(
    args: argparse.Namespace,
    spec: MethodSpec,
    batch_id: str,
    profile: RunProfile,
) -> str:
    suffix = (
        f"exact_shape_smoke{profile.total_steps}"
        if args.numerical_smoke
        else f"seed{args.seed}"
    )
    return f"{args.run_prefix}_{spec.name}_{suffix}_{batch_id}"


def next_attempt_run_name(batch_dir: Path, base_name: str) -> str:
    if not (batch_dir / base_name).exists():
        return base_name
    attempt = 2
    while (batch_dir / f"{base_name}_retry{attempt:02d}").exists():
        attempt += 1
    return f"{base_name}_retry{attempt:02d}"


def recover_completed_result(
    args: argparse.Namespace,
    batch_dir: Path,
    batch_id: str,
    spec: MethodSpec,
    derived: DerivedSource,
    provenance: dict[str, object],
    runtime: dict[str, object],
    profile: RunProfile,
    expected_init_sha256: str,
) -> tuple[dict[str, object], list[dict[str, object]]] | None:
    """Revalidate and reuse a completed attempt; retry only its W&B upload if needed."""
    base_name = base_run_name(args, spec, batch_id, profile)
    candidates = sorted(
        [path for path in batch_dir.glob(f"{base_name}*") if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    reusable_statuses = {
        "completed_valid_local",
        "completed_valid",
        "completed_valid_smoke",
    }
    for run_dir in candidates:
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") not in reusable_statuses:
                continue
            source = manifest.get("source")
            if not isinstance(source, dict) or source.get("derived_script_sha256") != derived.derived_sha256:
                raise RuntimeError("derived source fingerprint differs from the current R1 source")
            workspace = run_dir / "workspace"
            copied_log = run_dir / "training_log_with_source.txt"
            stdout_path = run_dir / "training_stdout.log"
            checkpoint = find_checkpoint(workspace)
            rows, summary = parse_metrics(
                copied_log,
                stdout_path,
                spec,
                checkpoint,
                profile,
                args.seed,
                expected_init_sha256,
            )
            summary.update(evidence_eligibility(args))
            summary.update(
                {
                    "run_name": run_dir.name,
                    "wrapper_wall_elapsed_s": manifest.get("wrapper_wall_elapsed_s", math.nan),
                    "official_log_path": str(copied_log.resolve()),
                    "checkpoint_path": str(checkpoint.resolve()) if checkpoint else "",
                    "derived_script_sha256": derived.derived_sha256,
                }
            )
            old_summary = manifest.get("summary")
            if isinstance(old_summary, dict):
                for key in ("final_val_step", "final_val_loss", "final_train_step", "final_train_loss"):
                    if str(old_summary.get(key)) != str(summary.get(key)):
                        raise RuntimeError(f"saved summary disagrees with reparsed evidence for {key}")

            upload = manifest.get("wandb")
            uploaded = isinstance(upload, dict) and upload.get("status") == "uploaded"
            if profile.formal_evidence and not uploaded:
                manifest.update(
                    {
                        "status": "completed_valid_local",
                        "summary": summary,
                        "resume_revalidated_at": datetime.now().astimezone().isoformat(),
                        "wandb": {"status": "retrying"},
                    }
                )
                write_json(manifest_path, manifest)
                group = f"{args.run_prefix}_seed{args.seed}_{batch_id}"
                try:
                    upload = upload_to_wandb(
                        args,
                        run_dir,
                        run_dir.name,
                        group,
                        spec,
                        rows,
                        summary,
                        provenance,
                        runtime,
                        derived,
                    )
                except BaseException as exc:
                    manifest["wandb"] = {"status": "interrupted", "error": repr(exc)}
                    write_json(manifest_path, manifest)
                    raise
                manifest["wandb"] = upload
                manifest["status"] = "completed_valid"
                write_json(manifest_path, manifest)
            print(f"Resume: revalidated and skipped completed R1 {spec.name}: {run_dir}")
            return summary, rows
        except (OSError, ValueError, RuntimeError, r0.EvidenceValidationError) as exc:
            print(
                f"Resume: cannot reuse {run_dir.name}; preserving it and starting a new attempt: {exc}",
                file=sys.stderr,
            )
    return None


def execute_one(
    args: argparse.Namespace,
    repo: Path,
    data_dir: Path,
    batch_dir: Path,
    batch_id: str,
    spec: MethodSpec,
    derived: DerivedSource,
    provenance: dict[str, object],
    runtime: dict[str, object],
    profile: RunProfile,
    expected_init_sha256: str,
    run_name_override: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    smoke = args.numerical_smoke
    suffix = f"exact_shape_smoke{profile.total_steps}" if smoke else f"seed{args.seed}"
    run_name = run_name_override or f"{args.run_prefix}_{spec.name}_{suffix}_{batch_id}"
    run_dir = batch_dir / run_name
    workspace = run_dir / "workspace"
    script_path = materialize_source(workspace, repo, derived)
    (run_dir / "official_to_r1.patch").write_text(derived.unified_diff, encoding="utf-8")
    source_manifest = {
        "experiment_family": experiment_family(args),
        "protocol": experiment_protocol(args),
        "method": spec.name,
        "method_role": spec.role,
        "official_base_script": derived.base_script,
        "official_base_canonical_sha256": derived.base_canonical_sha256,
        "derived_script": str(script_path.resolve()),
        "derived_script_sha256": derived.derived_sha256,
        "cproj_k_mode": spec.cproj_k_mode,
        "base_learning_rate": spec.base_learning_rate,
        "matrix_learning_rate": spec.matrix_learning_rate,
        "evidence_profile": profile.name,
        "formal_evidence": profile.formal_evidence,
        "resource_isolation": visible_device_record(args),
        "evidence_eligibility": evidence_eligibility(args),
        "smoke_steps": profile.total_steps if not profile.formal_evidence else None,
        "training_shape": {
            "batch_size_sequences": 512,
            "device_batch_size_sequences": 64,
            "sequence_length": 1024,
            "train_accumulation_steps": 8,
        },
    }
    write_json(run_dir / "source_manifest.json", source_manifest)

    command = [args.python_exe, script_path.name]
    stdout_path = run_dir / "training_stdout.log"
    started_at = datetime.now().astimezone()
    wall_start = time.monotonic()
    manifest: dict[str, object] = {
        "experiment_family": experiment_family(args),
        "protocol": experiment_protocol(args),
        "run_name": run_name,
        "method": spec.name,
        "cproj_k_mode": spec.cproj_k_mode,
        "controlled_seed": args.seed,
        "command": command,
        "command_text": command_text(command),
        "cwd": str(workspace.resolve()),
        "environment_controls": {
            "PYTHONHASHSEED": str(args.seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "R1_SEED": str(args.seed),
            "R1_SMOKE_TEST": "1" if smoke else "0",
            "R1_SMOKE_STEPS": str(profile.total_steps),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "resource_isolation": visible_device_record(args),
        "evidence_eligibility": evidence_eligibility(args),
        "status": "running",
        "started_at": started_at.isoformat(),
        "smoke_test": smoke,
        "evidence_profile": profile.name,
        "formal_evidence": profile.formal_evidence,
        "official_commit": r0.OFFICIAL_COMMIT,
        "source": source_manifest,
        "wandb": {"status": "not_started"},
    }
    write_json(run_dir / "run_manifest.json", manifest)
    with stdout_path.open("w", encoding="utf-8", buffering=1) as output:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=controlled_env(
                args,
                spec,
                data_dir,
                smoke_test=smoke,
                smoke_steps=profile.total_steps,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        first_nonfinite = r0.stream_process_with_finite_gate(process, output)
        returncode = process.wait()
    wall_elapsed_s = time.monotonic() - wall_start
    finished_at = datetime.now().astimezone()
    manifest.update(
        {
            "returncode": returncode,
            "finished_at": finished_at.isoformat(),
            "wrapper_wall_elapsed_s": wall_elapsed_s,
        }
    )
    if first_nonfinite is not None:
        manifest.update(
            {
                "status": "invalid_nonfinite",
                "first_nonfinite": first_nonfinite,
            }
        )
        write_json(run_dir / "run_manifest.json", manifest)
        raise r0.EvidenceValidationError(
            "invalid_nonfinite",
            f"R1 numerical gate stopped {run_name} at {first_nonfinite}",
            {"first_nonfinite": first_nonfinite},
        )
    if returncode != 0:
        manifest["status"] = "training_failed"
        write_json(run_dir / "run_manifest.json", manifest)
        raise subprocess.CalledProcessError(returncode, command)

    official_log = find_single_log(workspace)
    copied_log = run_dir / "training_log_with_source.txt"
    shutil.copy2(official_log, copied_log)
    checkpoint = find_checkpoint(workspace)
    try:
        rows, summary = parse_metrics(
            copied_log,
            stdout_path,
            spec,
            checkpoint,
            profile,
            args.seed,
            expected_init_sha256,
        )
        summary.update(evidence_eligibility(args))
    except r0.EvidenceValidationError as exc:
        manifest.update(
            {
                "status": exc.status,
                "validation_error": str(exc),
                "validation_details": exc.details,
            }
        )
        write_json(run_dir / "run_manifest.json", manifest)
        raise
    summary.update(
        {
            "run_name": run_name,
            "wrapper_wall_elapsed_s": wall_elapsed_s,
            "official_log_path": str(official_log.resolve()),
            "checkpoint_path": str(checkpoint.resolve()) if checkpoint else "",
            "derived_script_sha256": derived.derived_sha256,
        }
    )
    write_csv(run_dir / "r1_metrics.csv", rows)
    write_json(run_dir / "r1_summary.json", summary)

    # Persist valid local evidence before any network operation. Even a hard
    # interruption during W&B initialization cannot erase the validity verdict.
    manifest.update(
        {
            "status": "completed_valid_local",
            "summary": summary,
            "metrics_csv": str((run_dir / "r1_metrics.csv").resolve()),
            "copied_training_log": str(copied_log.resolve()),
            "checkpoint_path": str(checkpoint.resolve()) if checkpoint else "",
            "wandb": {"status": "pending" if not smoke else "disabled_for_numerical_smoke"},
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)

    group = f"{args.run_prefix}_seed{args.seed}_{batch_id}"
    try:
        upload = (
            {"status": "disabled_for_numerical_smoke"}
            if smoke
            else upload_to_wandb(
                args,
                run_dir,
                run_name,
                group,
                spec,
                rows,
                summary,
                provenance,
                runtime,
                derived,
            )
        )
    except BaseException as exc:
        manifest["wandb"] = {
            "status": "interrupted",
            "error": repr(exc),
            "run_name": run_name,
        }
        write_json(run_dir / "run_manifest.json", manifest)
        raise
    if upload.get("status") == "failed":
        print(f"Warning: R1 training completed but W&B upload failed: {upload.get('error')}", file=sys.stderr)
    manifest.update(
        {
            "status": "completed_valid_smoke" if smoke else "completed_valid",
            "wandb": upload,
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)
    return summary, rows


def common_target_rows(results: list[tuple[dict[str, object], list[dict[str, object]]]]) -> list[dict[str, object]]:
    if not results:
        return []
    target = max(float(summary["final_val_loss"]) for summary, _ in results)
    output: list[dict[str, object]] = []
    for summary, rows in results:
        val_rows = sorted(
            [row for row in rows if row["event"] == "validation"],
            key=lambda row: int(row["step"]),
        )
        reached = next((row for row in val_rows if float(row["loss"]) <= target), None)
        output.append(
            {
                "method": summary["method"],
                "common_target_val_loss": target,
                "first_observed_step": int(reached["step"]) if reached else -1,
                "tokens_seen": int(reached["tokens_seen"]) if reached else -1,
                "official_train_time_s": (
                    float(reached["official_train_time_ms"]) / 1000.0 if reached else -1.0
                ),
                "observed_val_loss": float(reached["loss"]) if reached else math.nan,
            }
        )
    return output


def collect_wandb_statuses(
    batch_dir: Path,
    summaries: list[dict[str, object]],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for summary in summaries:
        method = str(summary["method"])
        manifest_path = batch_dir / str(summary["run_name"]) / "run_manifest.json"
        status = "missing_manifest"
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            wandb_payload = payload.get("wandb")
            if isinstance(wandb_payload, dict):
                status = str(wandb_payload.get("status", "unknown"))
        statuses[method] = status
    return statuses


def main() -> None:
    args = parse_args()
    resource_isolation = visible_device_record(args)
    repo = args.official_repo.expanduser().resolve()
    provenance = r0.validate_official_repo(repo)
    specs = experiment_specs(args)
    built = build_all_sources(repo, lr_cross=args.lr_cross)
    if args.host_bridge:
        built = {method: built[method] for method in args.methods}
    resume_plan: dict[str, object] | None = None
    resume_batch_dir: Path | None = None
    if args.resume_batch is not None:
        resume_batch_dir = args.resume_batch.expanduser().resolve()
        plan_path = resume_batch_dir / "r1_plan.json"
        if not plan_path.is_file():
            raise RuntimeError(f"R1 resume batch has no r1_plan.json: {resume_batch_dir}")
        resume_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        saved_certificate = resume_plan.get("smoke_certificate")
        if args.smoke_manifest is None and isinstance(saved_certificate, dict):
            saved_path = saved_certificate.get("path")
            if saved_path:
                args.smoke_manifest = Path(str(saved_path))
    print_plan(args, repo, built)
    if args.dry_run:
        print(f"Dry-run complete: {len(args.methods)} controlled job(s); no data/GPU/W&B used.")
        return

    data = r0.validate_data(repo)
    data_dir = Path(str(data["data_dir"]))
    runtime = r0.validate_runtime(repo, args.python_exe)
    rejection_reason = str(runtime.get("runtime_rejection_reason", ""))
    if rejection_reason and not args.numerical_smoke:
        raise RuntimeError(
            "Selected training runtime is blocked for formal R1:\n"
            f"- {runtime.get('python_executable')}\n"
            f"- torch={runtime.get('torch')} CUDA={runtime.get('torch_cuda')} "
            f"Triton={runtime.get('triton')}\n"
            f"- {rejection_reason}\n"
            "Use an isolated compatible training interpreter, then pass the matching "
            "exact-shape numerical smoke and provide its --smoke-manifest."
        )
    if rejection_reason:
        print(f"Warning: testing a previously rejected runtime: {rejection_reason}", file=sys.stderr)
    controller_runtime = r0.validate_controller_runtime(
        require_wandb=args.wandb_mode != "disabled" and not args.numerical_smoke
    )
    wandb_readiness = validate_wandb_online_access(
        enabled=args.wandb_mode == "online" and not args.numerical_smoke
    )
    print(
        f"Runtime: {runtime['gpu_name']}, {runtime['gpu_total_memory_gib']:.2f} GiB, "
        f"{data['train_shards']} train shards."
    )
    init_audit = initialization_audit(args, repo, data_dir, built)
    if args.preflight:
        print(
            f"Preflight complete: code/data/runtime valid and all {len(built)} methods share init "
            f"{init_audit['init_sha256']}; no training started."
        )
        return

    if resume_plan is not None:
        validate_resume_plan(resume_plan, args, runtime, built, init_audit)

    smoke_certificate = None
    if not args.numerical_smoke:
        assert args.smoke_manifest is not None
        smoke_certificate = validate_smoke_manifest(
            args.smoke_manifest,
            runtime,
            args.methods,
            args.seed,
            built,
            experiment_protocol(args, smoke=True),
            (
                str(resource_isolation["cuda_visible_devices"])
                if args.host_bridge
                else None
            ),
        )

    profile = (
        numerical_smoke_profile(args.smoke_steps)
        if args.numerical_smoke
        else FORMAL_PROFILE
    )

    if resume_plan is not None:
        assert resume_batch_dir is not None
        batch_id = str(resume_plan["batch_id"])
        batch_kind = str(resume_plan["batch_kind"])
        batch_dir = resume_batch_dir
    else:
        batch_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        batch_kind = "smoke" if args.numerical_smoke else "formal"
        batch_dir = args.results_dir.expanduser().resolve() / f"{batch_id}_{batch_kind}_seed{args.seed}"
        batch_dir.mkdir(parents=True, exist_ok=False)
    plan: dict[str, object] = {
        "family": experiment_family(args),
        "batch_id": batch_id,
        "batch_kind": batch_kind,
        "protocol": experiment_protocol(args),
        "lr_cross": args.lr_cross,
        "host_bridge": args.host_bridge,
        "official_commit": r0.OFFICIAL_COMMIT,
        "methods": args.methods,
        "run_prefix": args.run_prefix,
        "seed": args.seed,
        "initialization_audit": init_audit,
        "official_provenance": provenance,
        "data": data,
        "runtime": runtime,
        "training_runtime_fingerprint": r0.runtime_fingerprint(runtime),
        "controller_runtime": controller_runtime,
        "wandb_readiness": wandb_readiness,
        "derived_source_sha256": source_fingerprints(built),
        "evidence_profile": profile.name,
        "smoke_steps": profile.total_steps if args.numerical_smoke else None,
        "formal_evidence": profile.formal_evidence,
        "resource_isolation": resource_isolation,
        "evidence_eligibility": evidence_eligibility(args),
        "smoke_certificate": smoke_certificate,
        "wandb_project": args.wandb_project,
        "wandb_mode": "disabled" if args.numerical_smoke else args.wandb_mode,
        "method_specs": {name: specs[name].__dict__ for name in args.methods},
        "interpretation_boundary": (
            {
                "primary_estimand": (
                    "GPT R1 diag-minus-none quality delta on the same H100 host/runtime "
                    "used by LLaMA/SwiGLU"
                ),
                "architecture_inference": (
                    "compare this within-host GPT delta with the within-host LLaMA delta; "
                    "do not compare cross-host timing"
                ),
                "quality": "usable if step/token/data/init/completion gates pass",
                "memory": "per-GPU allocated/state evidence remains usable on an exclusive GPU",
                "timing": "ineligible even when recorded; concurrent node workload is expected",
            }
            if args.host_bridge
            else
            {
                "primary_estimand": (
                    "Muon-vs-diag method effect at both shared absolute LR levels, "
                    "paired by controlled seed"
                ),
                "existing_cells": "Muon@0.0036 and diag@0.0040 come from formal R1",
                "new_cells": "Muon@0.0040 and diag@0.0036 are produced here",
                "timing": (
                    "quality is primary; do not pool timing with R1 while both physical "
                    "GPUs share one host"
                ),
            }
            if args.lr_cross
            else {
                "block4_none_diag": "same LR and only mlp.c_proj K representation differs",
                "muon_vs_newton_variants": (
                    "official method-specific recipe comparison, not shared-LR causality"
                ),
                "single_seed": "pilot evidence; expand seeds only after direction is established",
            }
        ),
    }
    if resume_plan is None:
        write_json(batch_dir / "r1_plan.json", plan)
    else:
        plan = {
            **resume_plan,
            "last_resumed_at": datetime.now().astimezone().isoformat(),
            "last_resume_controller_runtime": controller_runtime,
        }

    results: list[tuple[dict[str, object], list[dict[str, object]]]] = []
    failures: list[dict[str, object]] = []
    write_json(
        batch_dir / "r1_manifest.json",
        {**plan, "status": "running", "summaries": [], "failures": []},
    )
    for method in args.methods:
        spec = specs[method]
        print(f"\n=== R1 {method}: seed={args.seed}, c_proj={spec.cproj_k_mode} ===")
        try:
            recovered = None
            if resume_plan is not None:
                recovered = recover_completed_result(
                    args,
                    batch_dir,
                    batch_id,
                    spec,
                    built[method],
                    provenance,
                    runtime,
                    profile,
                    str(init_audit["init_sha256"]),
                )
            if recovered is not None:
                results.append(recovered)
            else:
                base_name = base_run_name(args, spec, batch_id, profile)
                attempt_name = next_attempt_run_name(batch_dir, base_name)
                if attempt_name != base_name:
                    print(f"Resume: starting a fresh attempt for {method}: {attempt_name}")
                results.append(
                    execute_one(
                        args,
                        repo,
                        data_dir,
                        batch_dir,
                        batch_id,
                        spec,
                        built[method],
                        provenance,
                        runtime,
                        profile,
                        str(init_audit["init_sha256"]),
                        run_name_override=attempt_name,
                    )
                )
        except Exception as exc:
            failures.append({"method": method, "error": repr(exc)})
            print(f"R1 {method} failed: {exc}", file=sys.stderr)
            write_json(
                batch_dir / "r1_manifest.json",
                {
                    **plan,
                    "status": "failed",
                    "summaries": [summary for summary, _ in results],
                    "failures": failures,
                },
            )
            if not args.continue_on_error:
                break
        else:
            write_json(
                batch_dir / "r1_manifest.json",
                {
                    **plan,
                    "status": "running",
                    "summaries": [summary for summary, _ in results],
                    "failures": failures,
                },
            )

    summaries = [summary for summary, _ in results]
    write_csv(batch_dir / "r1_summary.csv", summaries)
    write_csv(batch_dir / "r1_common_target_comparison.csv", common_target_rows(results))
    observed_inits = {str(summary["init_sha256"]) for summary in summaries}
    expected_init = str(init_audit["init_sha256"])
    all_requested_completed = len(summaries) == len(args.methods)
    fingerprints_identical = observed_inits == {expected_init} if summaries else False
    wandb_statuses = collect_wandb_statuses(batch_dir, summaries)
    wandb_complete = args.numerical_smoke or (
        all_requested_completed
        and all(status == "uploaded" for status in wandb_statuses.values())
    )
    final_manifest = {
        **plan,
        "status": (
            "completed_valid_smoke"
            if args.numerical_smoke and all_requested_completed and not failures
            else "completed_valid"
            if all_requested_completed and not failures and wandb_complete
            else "completed_valid_local_wandb_incomplete"
            if all_requested_completed and not failures
            else "failed"
        ),
        "formal_initialization_fingerprints_identical": fingerprints_identical,
        "wandb_statuses": wandb_statuses,
        "wandb_complete": wandb_complete,
        "summaries": summaries,
        "failures": failures,
    }
    write_json(batch_dir / "r1_manifest.json", final_manifest)
    print(f"R1 artifacts: {batch_dir}")
    if summaries and not fingerprints_identical:
        raise RuntimeError(
            f"formal R1 initialization fingerprints differ from audit: "
            f"observed={observed_inits}, expected={expected_init}"
        )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
