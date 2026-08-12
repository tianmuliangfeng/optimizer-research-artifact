"""Run the two-job official Newton-Muon-1 H100 reproduction gate.

The official training scripts are executed unchanged in a pinned checkout.
This wrapper only validates provenance/data/hardware, records the commands,
captures stdout and official logs, derives scalar CSV summaries, and uploads
scalar curves to W&B *after* training so that W&B cannot perturb benchmark
timing.  It deliberately does not set a seed because the official scripts do
not set one.
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
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from project_paths import EXPERIMENT_RESULTS_ROOT


FAMILY = "14_official_newton_muon_r0"
OFFICIAL_REPOSITORY_URL = "https://github.com/zhehangdu/Newton-Muon.git"
OFFICIAL_COMMIT = "df78af0db523d8bceb25af4919a3e3e7082b80f3"
DEFAULT_PROJECT = "Selective-Newton-Muon-MainConf-OfficialR0-H100-20260717"
DEFAULT_RUN_PREFIX = "mainconf_official_r0_newton_muon1"
EXPECTED_TRAIN_SHARDS = 50
DATA_MAGIC = 20240520
FORMAL_STEPS = 6200
FORMAL_VAL_EVERY = 100
DEFAULT_SMOKE_STEPS = 10
REJECTED_RUNTIME_PREFIXES = (
    (
        "2.12.1+cu130",
        "This exact PyTorch/CUDA family produced non-finite loss from optimizer step 2 "
        "in both official R0 methods on 2026-07-18.",
    ),
)

EXPECTED_CANONICAL_SHA256 = {
    "train_gpt_muon_1.py": "8e3e990a9a010a9f8ddee0e6d111ac7b83acedc7f41d2d3370de5f404c9aab59",
    "train_gpt_newton_muon_1.py": "48383e333334e4f29bbae3365ac4142226c27750ede5739ab53c0dafbbcb7730",
    "triton_kernels.py": "b51ac50c699b05306619d92cb9ec6edadd266d8118c53f5b9726db76480ea16d",
    "data/cached_fineweb10B.py": "adcc9f7d81ed1ac115a66d08d94d8d3e5c7425cabaf856da1f1fb106af87d09b",
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    script: str
    official_learning_rate: float
    official_matrix_learning_rate: float
    label: str


@dataclass(frozen=True)
class RunProfile:
    name: str
    total_steps: int
    validation_steps: tuple[int, ...]
    require_peak_memory: bool = True


class EvidenceValidationError(RuntimeError):
    """The process ran, but its output is not valid experimental evidence."""

    def __init__(self, status: str, message: str, details: dict[str, object] | None = None):
        super().__init__(message)
        self.status = status
        self.details = details or {}


METHODS = {
    "muon": MethodSpec(
        name="muon",
        script="train_gpt_muon_1.py",
        official_learning_rate=0.0036,
        official_matrix_learning_rate=0.00036,
        label="official_muon_1",
    ),
    "block4": MethodSpec(
        name="block4",
        script="train_gpt_newton_muon_1.py",
        official_learning_rate=0.0040,
        official_matrix_learning_rate=0.00040,
        label="official_newton_muon_1_block4",
    ),
}

FLOAT_TOKEN = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf(?:inity)?)"
VAL_RE = re.compile(
    rf"^step:(?P<step>\d+)/(?P<total>\d+) val_loss:(?P<loss>{FLOAT_TOKEN}) "
    rf"train_time:(?P<time>\d+)ms step_avg:(?P<avg>{FLOAT_TOKEN})ms$",
    re.IGNORECASE,
)
TRAIN_RE = re.compile(
    rf"^step:(?P<step>\d+)/(?P<total>\d+) train_loss:(?P<loss>{FLOAT_TOKEN}) "
    rf"train_time:(?P<time>\d+)ms step_avg:(?P<avg>{FLOAT_TOKEN})ms$",
    re.IGNORECASE,
)
PEAK_RE = re.compile(r"^peak memory consumption: (?P<mib>\d+) MiB$")

FORMAL_PROFILE = RunProfile(
    name="formal",
    total_steps=FORMAL_STEPS,
    validation_steps=tuple(range(0, FORMAL_STEPS + 1, FORMAL_VAL_EVERY)),
)


def default_official_repo() -> Path:
    value = os.environ.get("SNM_OFFICIAL_REPO") or os.environ.get(
        "NEWTON_MUON_OFFICIAL_REPO"
    )
    if value:
        return Path(value).expanduser()
    artifact_root = Path(__file__).resolve().parents[2]
    return artifact_root / "third_party" / "Newton-Muon-official-r0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official Newton-Muon-1 H100 reproduction gate: one "
            "unchanged official Muon job and one unchanged official block4 job."
        )
    )
    parser.add_argument("--official-repo", type=Path, default=default_official_repo())
    parser.add_argument("--python-exe", default="python")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS),
        default=list(METHODS),
        help="Default is exactly the two R0 jobs: muon block4.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate pinned code, 50 FineWeb shards, Python deps, and an >=80GB H100; do not train.",
    )
    parser.add_argument(
        "--numerical-smoke",
        action="store_true",
        help=(
            "Run an auditable derived-source numerical smoke with the official "
            "training batch/sequence shape. This is a compatibility gate, not R0 evidence."
        ),
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=DEFAULT_SMOKE_STEPS,
        help="Optimizer steps for --numerical-smoke (default: 10; minimum: 2).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=EXPERIMENT_RESULTS_ROOT / FAMILY / "results",
    )
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Scalar upload occurs only after each official training job completes.",
    )
    parser.add_argument(
        "--wandb-train-log-every",
        type=int,
        default=20,
        help="Downsample the per-step official train loss when uploading scalar curves.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        default=None,
        help=(
            "Successful r0_manifest.json from --numerical-smoke. Required for formal R0; "
            "the runtime fingerprint and requested methods must match."
        ),
    )
    args = parser.parse_args()
    if args.wandb_train_log_every <= 0:
        parser.error("--wandb-train-log-every must be positive")
    if len(args.methods) != len(set(args.methods)):
        parser.error("--methods contains duplicates")
    selected_modes = sum(bool(value) for value in (args.dry_run, args.preflight, args.numerical_smoke))
    if selected_modes > 1:
        parser.error("choose at most one of --dry-run, --preflight, and --numerical-smoke")
    if args.smoke_steps < 2:
        parser.error("--smoke-steps must be at least 2 so the second update is exercised")
    if args.numerical_smoke and args.wandb_mode != "disabled":
        parser.error("--numerical-smoke requires --wandb-mode disabled")
    if not (args.dry_run or args.preflight or args.numerical_smoke) and args.smoke_manifest is None:
        parser.error("formal R0 requires --smoke-manifest from a successful exact-shape numerical smoke")
    return args


def canonical_text_sha256(path: Path) -> str:
    """Hash tracked text while treating Git's LF and Windows CRLF equally."""
    canonical_bytes = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical_bytes).hexdigest()


def command_text(command: Iterable[str]) -> str:
    return shlex.join([str(part) for part in command])


def run_git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo.resolve().as_posix()}",
            "-C",
            str(repo),
            *arguments,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def validate_official_repo(repo: Path) -> dict[str, object]:
    repo = repo.resolve()
    if not repo.is_dir():
        raise RuntimeError(
            f"Official repository not found: {repo}\n"
            f"Clone {OFFICIAL_REPOSITORY_URL} and check out {OFFICIAL_COMMIT}."
        )

    observed_hashes: dict[str, str] = {}
    failures: list[str] = []
    for relative, expected in EXPECTED_CANONICAL_SHA256.items():
        path = repo / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        observed = canonical_text_sha256(path)
        observed_hashes[relative] = observed
        if observed != expected:
            failures.append(
                f"canonical SHA256 mismatch for {relative}: {observed} != {expected}"
            )

    commit_result = run_git(repo, "rev-parse", "HEAD")
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else ""
    if commit != OFFICIAL_COMMIT:
        failures.append(
            f"official commit mismatch: {commit or commit_result.stderr.strip()} != {OFFICIAL_COMMIT}"
        )

    dirty_result = run_git(repo, "status", "--porcelain", "--untracked-files=no")
    dirty = dirty_result.stdout.strip() if dirty_result.returncode == 0 else "git status failed"
    if dirty:
        failures.append(f"tracked official files are modified:\n{dirty}")

    if failures:
        raise RuntimeError("Official-code provenance check failed:\n- " + "\n- ".join(failures))

    return {
        "official_repository_url": OFFICIAL_REPOSITORY_URL,
        "official_repo": str(repo),
        "official_commit": commit,
        "tracked_worktree_clean": True,
        "canonical_text_sha256": observed_hashes,
    }


def read_magic(path: Path) -> int:
    with path.open("rb") as handle:
        raw = handle.read(4)
    if len(raw) != 4:
        raise RuntimeError(f"Data shard is too short: {path}")
    return int(struct.unpack("<i", raw)[0])


def validate_data(repo: Path) -> dict[str, object]:
    data_dir = repo / "data" / "fineweb10B"
    if not data_dir.exists():
        link_note = ""
        if data_dir.is_symlink():
            link_note = f" (broken symlink -> {os.readlink(data_dir)})"
        raise RuntimeError(f"FineWeb10B data directory is unavailable: {data_dir}{link_note}")
    train_shards = sorted(data_dir.glob("fineweb_train_*.bin"))
    val_shards = sorted(data_dir.glob("fineweb_val_*.bin"))
    required_train_shards = [
        data_dir / f"fineweb_train_{index:06d}.bin"
        for index in range(1, EXPECTED_TRAIN_SHARDS + 1)
    ]
    required_val_shard = data_dir / "fineweb_val_000000.bin"
    failures: list[str] = []
    missing_train = [path.name for path in required_train_shards if not path.is_file()]
    if missing_train:
        failures.append(
            f"missing {len(missing_train)} of the required shards 000001--000050: "
            + ", ".join(missing_train[:5])
        )
    if not required_val_shard.is_file():
        failures.append("missing fineweb_val_000000.bin")
    if not failures:
        for path in [*required_train_shards, required_val_shard]:
            if read_magic(path) != DATA_MAGIC:
                failures.append(f"bad data magic in {path}")
    if failures:
        raise RuntimeError(
            "FineWeb10B data check failed:\n- "
            + "\n- ".join(failures)
            + f"\nRun: {sys.executable} {repo / 'data' / 'cached_fineweb10B.py'} 50"
        )
    return {
        "data_dir": str(data_dir.resolve()),
        "data_entry": str(data_dir),
        "data_entry_is_symlink": data_dir.is_symlink(),
        "data_entry_symlink_target": os.readlink(data_dir) if data_dir.is_symlink() else "",
        "train_shards": len(train_shards),
        "validation_shards": len(val_shards),
        "first_train_shard": train_shards[0].name,
        "first_validation_shard": val_shards[0].name,
        "data_magic": DATA_MAGIC,
    }


def validate_runtime(repo: Path, python_exe: str) -> dict[str, object]:
    code = """
import json
import numpy
import sys
import torch
import triton
import triton_kernels
payload = {
    "python_executable": sys.executable,
    "python": __import__("sys").version,
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "triton": getattr(triton, "__version__", "unknown"),
    "triton_module": getattr(triton, "__file__", "unknown"),
    "triton_kernels_module": getattr(triton_kernels, "__file__", "unknown"),
    "cuda_available": torch.cuda.is_available(),
}
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    payload.update({"gpu_name": p.name, "gpu_total_memory_bytes": p.total_memory})
print(json.dumps(payload))
"""
    result = subprocess.run(
        [python_exe, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=repo,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "No module named 'triton.tools.tensor_descriptor'" in stderr:
            raise RuntimeError(
                f"Training runtime validation failed for {python_exe}: the installed Triton is "
                "too old for the pinned official triton_kernels.py because it does not provide "
                "triton.tools.tensor_descriptor.TensorDescriptor. Do not patch or bypass the "
                "official kernel import. Create an isolated compatible training environment "
                "(the project-tested candidate is PyTorch 2.8.0+cu126 with Triton 3.4.0), then "
                "rerun preflight and the exact-shape numerical smoke.\nOriginal traceback:\n"
                + stderr
            )
        raise RuntimeError(
            f"Training runtime validation failed for {python_exe}:\n{stderr}"
        )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if not payload.get("cuda_available"):
        raise RuntimeError("CUDA is unavailable in the selected training Python")
    gpu_name = str(payload.get("gpu_name", ""))
    memory_gib = float(payload.get("gpu_total_memory_bytes", 0)) / (1024**3)
    if "H100" not in gpu_name.upper():
        raise RuntimeError(f"Official R0 is the H100 profile, but GPU 0 is {gpu_name!r}")
    # Nominal 80GB H100s report slightly less than 80 GiB to software.
    if memory_gib < 75.0:
        raise RuntimeError(f"Official Newton-Muon-1 requires an >=80GB GPU; observed {memory_gib:.2f} GiB")
    payload["gpu_total_memory_gib"] = memory_gib
    payload["runtime_rejection_reason"] = runtime_rejection_reason(payload)
    return payload


def runtime_rejection_reason(runtime: dict[str, object]) -> str:
    torch_version = str(runtime.get("torch", ""))
    for prefix, reason in REJECTED_RUNTIME_PREFIXES:
        if torch_version.startswith(prefix):
            return reason
    return ""


def validate_controller_runtime(require_wandb: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "python_executable": sys.executable,
        "python": sys.version,
        "wandb_required": require_wandb,
    }
    if require_wandb:
        try:
            import wandb
        except Exception as exc:
            raise RuntimeError(
                "W&B upload is performed by the controller Python, but wandb cannot be imported "
                f"from {sys.executable}: {exc!r}"
            ) from exc
        payload["wandb"] = getattr(wandb, "__version__", "unknown")
        payload["wandb_module"] = getattr(wandb, "__file__", "unknown")
    return payload


def runtime_fingerprint(runtime: dict[str, object]) -> dict[str, object]:
    keys = (
        "python_executable",
        "numpy",
        "torch",
        "torch_cuda",
        "triton",
        "triton_module",
        "triton_kernels_module",
        "gpu_name",
        "gpu_total_memory_bytes",
    )
    fingerprint = {key: runtime.get(key) for key in keys}
    fingerprint["python_version"] = _python_semantic_version(runtime)
    return fingerprint


def _python_semantic_version(runtime: dict[str, object]) -> str:
    explicit = str(runtime.get("python_version", "")).strip()
    if explicit:
        return explicit
    # Legacy smoke manifests stored the complete sys.version string, including a
    # distribution-specific build date. That timestamp is not a numerical-runtime
    # compatibility boundary; major.minor.micro is.
    full = str(runtime.get("python", "")).strip()
    return full.split()[0] if full else ""


def normalize_runtime_fingerprint(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    material_keys = (
        "python_executable",
        "numpy",
        "torch",
        "torch_cuda",
        "triton",
        "triton_module",
        "triton_kernels_module",
        "gpu_name",
        "gpu_total_memory_bytes",
    )
    normalized = {key: payload.get(key) for key in material_keys}
    normalized["python_version"] = _python_semantic_version(payload)
    return normalized


def validate_smoke_manifest(
    path: Path,
    runtime: dict[str, object],
    methods: list[str],
) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"Smoke manifest not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    failures: list[str] = []
    if payload.get("protocol") != "official_newton_muon_1_h100_exact_shape_numerical_smoke":
        failures.append("manifest protocol is not the exact-shape R0 numerical smoke")
    if payload.get("official_commit") not in (None, OFFICIAL_COMMIT):
        failures.append("official commit does not match the pinned R0 commit")
    if payload.get("failures"):
        failures.append("smoke manifest contains failures")
    completed = {str(item.get("method")) for item in payload.get("summaries", [])}
    missing_methods = sorted(set(methods) - completed)
    if missing_methods:
        failures.append(f"smoke did not validate requested methods: {missing_methods}")
    observed_fingerprint = normalize_runtime_fingerprint(
        payload.get("training_runtime_fingerprint")
    )
    expected_fingerprint = normalize_runtime_fingerprint(runtime_fingerprint(runtime))
    if observed_fingerprint != expected_fingerprint:
        failures.append(
            "training runtime fingerprint differs from the smoke certificate; "
            f"observed={observed_fingerprint!r}, current={expected_fingerprint!r}"
        )
    if failures:
        raise RuntimeError("Smoke certificate check failed:\n- " + "\n- ".join(failures))
    return {"path": str(resolved), "validated": True, "methods": sorted(completed)}


def lr_multiplier(
    step: int,
    num_iterations: int = FORMAL_STEPS,
    warmdown_iters: int = 1800,
) -> float:
    if step < num_iterations - warmdown_iters:
        return 1.0
    return max(0.0, (num_iterations - step) / warmdown_iters)


def parse_number(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def match_metric_line(line: str) -> tuple[str, re.Match[str]] | None:
    stripped = line.strip()
    match = VAL_RE.match(stripped)
    if match is not None:
        return "validation", match
    match = TRAIN_RE.match(stripped)
    if match is not None:
        return "train", match
    return None


def recover_metrics(
    log_path: Path,
    stdout_path: Path,
    spec: MethodSpec,
    profile: RunProfile = FORMAL_PROFILE,
) -> tuple[list[dict[str, object]], float]:
    rows: list[dict[str, object]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = match_metric_line(line)
        if parsed is None:
            continue
        event, match = parsed
        step = int(match.group("step"))
        multiplier = lr_multiplier(
            step,
            num_iterations=profile.total_steps,
            warmdown_iters=1 if profile.name == "exact_shape_numerical_smoke" else 1800,
        )
        rows.append(
            {
                "method": spec.name,
                "event": event,
                "step": step,
                "total_steps": int(match.group("total")),
                "loss": float(match.group("loss")),
                "official_train_time_ms": int(match.group("time")),
                "step_avg_ms": parse_number(match.group("avg")),
                "lr_multiplier": multiplier,
                "adamw_lr": spec.official_learning_rate * multiplier,
                "matrix_lr": spec.official_matrix_learning_rate * multiplier,
            }
        )

    peak_mib = math.nan
    for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PEAK_RE.match(line)
        if match:
            peak_mib = float(match.group("mib"))
    return rows, peak_mib


def _expected_train_steps(profile: RunProfile) -> tuple[int, ...]:
    return tuple(range(1, profile.total_steps + 1))


def validate_and_summarize_metrics(
    rows: list[dict[str, object]],
    peak_mib: float,
    spec: MethodSpec,
    profile: RunProfile,
) -> dict[str, object]:
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
    if observed_val_steps != profile.validation_steps:
        failures.append(
            f"validation steps mismatch: observed {observed_val_steps[:8]}..."
            f"{observed_val_steps[-3:] if observed_val_steps else ()}, expected "
            f"{profile.validation_steps[:8]}...{profile.validation_steps[-3:]}"
        )
    expected_train_steps = _expected_train_steps(profile)
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
        failures.append(f"unexpected total_steps fields: {wrong_totals}; expected {profile.total_steps}")

    nonfinite = [
        {"event": row["event"], "step": row["step"], "loss": row["loss"]}
        for row in rows
        if not math.isfinite(float(row["loss"]))
    ]
    if nonfinite:
        first = nonfinite[0]
        raise EvidenceValidationError(
            "invalid_nonfinite",
            f"non-finite {first['event']} loss at step {first['step']}: {first['loss']}",
            {"first_nonfinite": first, "nonfinite_count": len(nonfinite)},
        )

    for event_rows, label in ((val_rows, "validation"), (train_rows, "train")):
        if any(int(row["official_train_time_ms"]) < 0 for row in event_rows):
            failures.append(f"negative official training time in {label} rows")
    # The upstream timer intentionally resets at step 32. After that reset it must be monotone.
    post_reset_train_times = [
        int(row["official_train_time_ms"]) for row in train_rows if int(row["step"]) >= 33
    ]
    if any(right < left for left, right in zip(post_reset_train_times, post_reset_train_times[1:])):
        failures.append("train_time decreases after the official step-32 timing reset")
    val_times = [int(row["official_train_time_ms"]) for row in val_rows if int(row["step"]) >= 100]
    if any(right < left for left, right in zip(val_times, val_times[1:])):
        failures.append("validation train_time is not monotone")
    if profile.require_peak_memory and (not math.isfinite(peak_mib) or peak_mib <= 0):
        failures.append(f"missing or invalid peak memory: {peak_mib}")

    if failures:
        raise EvidenceValidationError(
            "invalid_incomplete",
            "R0 evidence validation failed:\n- " + "\n- ".join(failures),
            {
                "observed_validation_points": len(val_rows),
                "observed_train_points": len(train_rows),
                "expected_validation_points": len(profile.validation_steps),
                "expected_train_points": profile.total_steps,
            },
        )

    final_val = val_rows[-1]
    best_val = min(val_rows, key=lambda row: float(row["loss"]))
    final_train = train_rows[-1]
    summary = {
        "method": spec.name,
        "script": spec.script,
        "official_seed_controlled": False,
        "official_base_learning_rate": spec.official_learning_rate,
        "official_matrix_learning_rate": spec.official_matrix_learning_rate,
        "final_val_step": final_val["step"],
        "final_val_loss": final_val["loss"],
        "best_val_step": best_val["step"],
        "best_val_loss": best_val["loss"],
        "final_train_step": final_train["step"],
        "final_train_loss": final_train["loss"],
        "official_train_time_s": float(final_val["official_train_time_ms"]) / 1000.0,
        "peak_memory_allocated_mib": peak_mib,
        "validation_points": len(val_rows),
        "train_points": len(train_rows),
        "evidence_profile": profile.name,
        "evidence_valid": True,
    }
    return summary


def parse_metrics(
    log_path: Path,
    stdout_path: Path,
    spec: MethodSpec,
    profile: RunProfile = FORMAL_PROFILE,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows, peak_mib = recover_metrics(log_path, stdout_path, spec, profile)
    summary = validate_and_summarize_metrics(rows, peak_mib, spec, profile)
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_numerical_smoke_source(
    repo: Path,
    run_dir: Path,
    spec: MethodSpec,
    smoke_steps: int,
) -> tuple[Path, dict[str, object]]:
    """Create an auditable short source while preserving formal train shapes and math."""
    official_path = repo / spec.script
    official_source = official_path.read_text(encoding="utf-8")
    anchor = "args = Hyperparameters()\n"
    if official_source.count(anchor) != 1:
        raise RuntimeError(f"Expected one Hyperparameters anchor in {official_path}")
    overlay = (
        anchor
        + "\n# R0 exact-shape numerical-smoke overlay. Not formal R0 evidence.\n"
        + f"args.num_iterations = {smoke_steps}\n"
        # A one-step warmdown changes only the post-final scheduler value; all smoke updates use LR=1x.
        + "args.warmdown_iters = 1\n"
        + f"args.val_loss_every = {smoke_steps}\n"
        + "args.val_tokens = args.device_batch_size * args.sequence_length\n"
        + "args.save_every = 0\n"
    )
    derived_source = official_source.replace(anchor, overlay, 1)
    checkpoint_anchor = (
        "if master_process and (last_step or "
        "(args.save_every > 0 and step % args.save_every == 0)):\n"
    )
    if derived_source.count(checkpoint_anchor) != 1:
        raise RuntimeError(f"Expected one checkpoint anchor in {official_path}")
    derived_source = derived_source.replace(
        checkpoint_anchor,
        "if False and master_process and (last_step or "
        "(args.save_every > 0 and step % args.save_every == 0)):\n",
        1,
    )

    source_dir = run_dir / "derived_smoke_source"
    source_dir.mkdir(parents=True, exist_ok=False)
    derived_path = source_dir / spec.script
    derived_path.write_text(derived_source, encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(
            official_source.splitlines(keepends=True),
            derived_source.splitlines(keepends=True),
            fromfile=f"official/{spec.script}",
            tofile=f"numerical_smoke/{spec.script}",
        )
    )
    diff_path = source_dir / "official_to_numerical_smoke.patch"
    diff_path.write_text(diff, encoding="utf-8")
    manifest = {
        "formal_evidence": False,
        "purpose": "exact_shape_numerical_compatibility_gate",
        "official_script": spec.script,
        "official_canonical_sha256": canonical_text_sha256(official_path),
        "derived_script": str(derived_path.resolve()),
        "derived_sha256": hashlib.sha256(derived_path.read_bytes()).hexdigest(),
        "patch": str(diff_path.resolve()),
        "smoke_steps": smoke_steps,
        "training_shape_preserved": {
            "batch_size_sequences": 512,
            "device_batch_size_sequences": 64,
            "sequence_length": 1024,
            "train_accumulation_steps": 8,
        },
        "validation_tokens_reduced": True,
        "checkpoint_disabled": True,
    }
    write_json(source_dir / "source_manifest.json", manifest)
    return derived_path, manifest


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def stream_process_with_finite_gate(
    process: subprocess.Popen[str],
    output,
) -> dict[str, object] | None:
    """Mirror stdout and terminate on the first parsed non-finite train/val loss."""
    assert process.stdout is not None
    first_nonfinite: dict[str, object] | None = None
    for line in process.stdout:
        print(line, end="", flush=True)
        output.write(line)
        parsed = match_metric_line(line)
        if parsed is None or first_nonfinite is not None:
            continue
        event, match = parsed
        loss = parse_number(match.group("loss"))
        if not math.isfinite(loss):
            first_nonfinite = {
                "event": event,
                "step": int(match.group("step")),
                "total_steps": int(match.group("total")),
                "loss_token": match.group("loss"),
            }
            print(
                "R0 numerical gate: terminating child after non-finite "
                f"{event} loss at step {first_nonfinite['step']}: {first_nonfinite['loss_token']}",
                file=sys.stderr,
                flush=True,
            )
            terminate_process(process)
    return first_nonfinite


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
) -> dict[str, object]:
    if args.wandb_mode == "disabled":
        return {"status": "disabled"}
    try:
        import wandb

        config = {
            "experiment_family": FAMILY,
            "protocol": "official_newton_muon_1_h100_unchanged",
            "method": spec.name,
            "official_script": spec.script,
            "official_repository_url": OFFICIAL_REPOSITORY_URL,
            "official_commit": OFFICIAL_COMMIT,
            "official_seed_controlled": False,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "batch_size_sequences": 512,
            "device_batch_size_sequences": 64,
            "sequence_length": 1024,
            "num_iterations": 6200,
            "warmdown_iters": 1800,
            "val_loss_every": 100,
            "val_tokens": 10485760,
            "base_learning_rate": spec.official_learning_rate,
            "matrix_learning_rate": spec.official_matrix_learning_rate,
            "wandb_upload_timing": "after_official_training_completed",
            "wandb_tables_enabled": False,
            "gpu_name": runtime.get("gpu_name"),
        }
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=group,
            mode=args.wandb_mode,
            dir=str(run_dir),
            tags=[
                "publication",
                "official_reproduction",
                "newton_muon_r0",
                "official_newton_muon_1",
                "h100",
                "seed_uncontrolled_official",
                spec.label,
            ],
            config=config,
            reinit=True,
        )
        per_step: dict[int, dict[str, float]] = {}
        for row in rows:
            step = int(row["step"])
            event = str(row["event"])
            if event == "train" and step % args.wandb_train_log_every != 0 and step != 6200:
                continue
            values = per_step.setdefault(step, {})
            values["official/train_time_s"] = float(row["official_train_time_ms"]) / 1000.0
            values["official/step_avg_ms"] = float(row["step_avg_ms"])
            values["lr/adamw"] = float(row["adamw_lr"])
            values["lr/matrix"] = float(row["matrix_lr"])
            if event == "validation":
                values["val/loss"] = float(row["loss"])
            else:
                values["train/loss_step"] = float(row["loss"])
        for step in sorted(per_step):
            wandb.log(per_step[step], step=step)
        for key, value in summary.items():
            if isinstance(value, (int, float, str, bool)):
                run.summary[key] = value
        run.summary["official_commit"] = provenance["official_commit"]
        run.finish()
        return {"status": "uploaded", "mode": args.wandb_mode, "run_name": run_name}
    except Exception as exc:  # training evidence remains valid even if network upload fails
        return {"status": "failed", "error": repr(exc), "run_name": run_name}


def find_new_official_log(repo: Path, before: set[Path]) -> Path:
    candidates = set((repo / "logs").glob("*.txt")) - before
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one new official log, observed {len(candidates)}: "
            + ", ".join(str(path) for path in sorted(candidates))
        )
    return next(iter(candidates))


def execute_one(
    args: argparse.Namespace,
    repo: Path,
    batch_dir: Path,
    batch_id: str,
    spec: MethodSpec,
    provenance: dict[str, object],
    runtime: dict[str, object],
) -> dict[str, object]:
    smoke = bool(args.numerical_smoke)
    suffix = f"exact_shape_smoke{args.smoke_steps}" if smoke else "formal"
    run_name = f"{args.run_prefix}_{spec.label}_{suffix}_{batch_id}"
    run_dir = batch_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    source_manifest: dict[str, object] | None = None
    if smoke:
        script_path, source_manifest = build_numerical_smoke_source(
            repo, run_dir, spec, args.smoke_steps
        )
        command = [args.python_exe, str(script_path.resolve())]
        profile = RunProfile(
            name="exact_shape_numerical_smoke",
            total_steps=args.smoke_steps,
            validation_steps=(0, args.smoke_steps),
        )
    else:
        command = [args.python_exe, spec.script]
        profile = FORMAL_PROFILE
    stdout_path = run_dir / "official_stdout.log"
    before_logs = set((repo / "logs").glob("*.txt"))
    started_at = datetime.now().astimezone()
    wall_start = time.monotonic()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")

    with stdout_path.open("w", encoding="utf-8", buffering=1) as output:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        first_nonfinite = stream_process_with_finite_gate(process, output)
        returncode = process.wait()

    wall_elapsed_s = time.monotonic() - wall_start
    finished_at = datetime.now().astimezone()
    base_manifest: dict[str, object] = {
        "run_name": run_name,
        "method": spec.name,
        "command": command,
        "command_text": command_text(command),
        "cwd": str(repo),
        "returncode": returncode,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "wrapper_wall_elapsed_s": wall_elapsed_s,
        "official_seed_controlled": False,
        "official_commit": OFFICIAL_COMMIT,
        "evidence_profile": profile.name,
        "formal_evidence": not smoke,
        "source": source_manifest,
    }
    if first_nonfinite is not None:
        base_manifest.update(
            {
                "status": "invalid_nonfinite",
                "first_nonfinite": first_nonfinite,
            }
        )
        write_json(run_dir / "run_manifest.json", base_manifest)
        raise EvidenceValidationError(
            "invalid_nonfinite",
            f"non-finite {first_nonfinite['event']} loss at step {first_nonfinite['step']}",
            {"first_nonfinite": first_nonfinite},
        )
    if returncode != 0:
        base_manifest["status"] = "training_failed"
        write_json(run_dir / "run_manifest.json", base_manifest)
        raise subprocess.CalledProcessError(returncode, command)

    official_log = find_new_official_log(repo, before_logs)
    copied_log = run_dir / "official_training_log.txt"
    shutil.copy2(official_log, copied_log)
    state_dir = repo / "logs" / official_log.stem
    rows, peak_mib = recover_metrics(copied_log, stdout_path, spec, profile)
    write_csv(run_dir / "official_metrics.csv", rows)
    try:
        summary = validate_and_summarize_metrics(rows, peak_mib, spec, profile)
    except EvidenceValidationError as exc:
        base_manifest.update(
            {
                "status": exc.status,
                "validation_error": str(exc),
                "validation_details": exc.details,
                "official_log_path": str(official_log.resolve()),
                "copied_official_log": str(copied_log.resolve()),
                "metrics_csv": str((run_dir / "official_metrics.csv").resolve()),
            }
        )
        write_json(run_dir / "run_manifest.json", base_manifest)
        raise
    summary.update(
        {
            "run_name": run_name,
            "wrapper_wall_elapsed_s": wall_elapsed_s,
            "official_log_path": str(official_log.resolve()),
            "official_state_dir": str(state_dir.resolve()),
            "formal_evidence": not smoke,
        }
    )
    write_json(run_dir / "official_summary.json", summary)

    group = f"{args.run_prefix}_{batch_id}"
    upload = (
        {"status": "disabled_for_numerical_smoke"}
        if smoke
        else upload_to_wandb(
            args, run_dir, run_name, group, spec, rows, summary, provenance, runtime
        )
    )
    if upload.get("status") == "failed":
        print(
            f"Warning: official training completed but W&B upload failed for {run_name}: "
            f"{upload.get('error')}",
            file=sys.stderr,
        )
    base_manifest.update(
        {
            "status": "completed_valid_smoke" if smoke else "completed_valid",
            "official_log_path": str(official_log.resolve()),
            "copied_official_log": str(copied_log.resolve()),
            "official_state_dir": str(state_dir.resolve()),
            "summary": summary,
            "wandb": upload,
        }
    )
    write_json(run_dir / "run_manifest.json", base_manifest)
    return summary


def print_plan(args: argparse.Namespace, repo: Path) -> None:
    print(f"Official repository: {repo}")
    print(f"Pinned commit:       {OFFICIAL_COMMIT}")
    print("Official seed:       not explicitly controlled by upstream scripts")
    print(f"W&B project:         {args.wandb_project}")
    print(
        "Run mode:            "
        + (f"exact-shape numerical smoke ({args.smoke_steps} steps)" if args.numerical_smoke else "formal R0")
    )
    print("Planned jobs:")
    for index, method in enumerate(args.methods, start=1):
        spec = METHODS[method]
        print(
            f"  {index}. {method:6s} | base_lr={spec.official_learning_rate:g} "
            f"matrix_lr={spec.official_matrix_learning_rate:g} | "
            f"{command_text([args.python_exe, spec.script])}"
        )


def main() -> None:
    args = parse_args()
    repo = args.official_repo.expanduser().resolve()
    provenance = validate_official_repo(repo)
    print_plan(args, repo)

    if args.dry_run:
        print(f"Dry-run complete: {len(args.methods)} job(s), no training or W&B upload.")
        return

    data = validate_data(repo)
    runtime = validate_runtime(repo, args.python_exe)
    rejection_reason = str(runtime.get("runtime_rejection_reason", ""))
    if rejection_reason and not args.numerical_smoke:
        raise RuntimeError(
            "Selected training runtime is blocked for formal R0:\n"
            f"- {runtime.get('python_executable')}\n"
            f"- torch={runtime.get('torch')} CUDA={runtime.get('torch_cuda')} "
            f"Triton={runtime.get('triton')}\n"
            f"- {rejection_reason}\n"
            "Use the same H100 with a different training interpreter, then pass the exact-shape "
            "numerical smoke and provide its --smoke-manifest."
        )
    if rejection_reason:
        print(f"Warning: testing a previously rejected runtime: {rejection_reason}", file=sys.stderr)
    controller_runtime = validate_controller_runtime(
        require_wandb=args.wandb_mode != "disabled" and not args.numerical_smoke
    )
    smoke_certificate = None
    if not (args.preflight or args.numerical_smoke):
        assert args.smoke_manifest is not None
        smoke_certificate = validate_smoke_manifest(args.smoke_manifest, runtime, args.methods)
    print(
        f"Preflight: {runtime['gpu_name']}, {runtime['gpu_total_memory_gib']:.2f} GiB, "
        f"{data['train_shards']} train shards, {data['validation_shards']} val shard(s)."
    )
    if args.preflight:
        print("Preflight complete: official code/data/runtime are ready; no training started.")
        return

    batch_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    batch_dir = args.results_dir.expanduser().resolve() / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    protocol = (
        "official_newton_muon_1_h100_exact_shape_numerical_smoke"
        if args.numerical_smoke
        else "official_newton_muon_1_h100_unchanged"
    )
    plan = {
        "family": FAMILY,
        "batch_id": batch_id,
        "protocol": protocol,
        "methods": args.methods,
        "official_commit": OFFICIAL_COMMIT,
        "official_seed_controlled": False,
        "provenance": provenance,
        "data": data,
        "runtime": runtime,
        "training_runtime_fingerprint": runtime_fingerprint(runtime),
        "controller_runtime": controller_runtime,
        "smoke_steps": args.smoke_steps if args.numerical_smoke else None,
        "formal_evidence": not args.numerical_smoke,
        "smoke_certificate": smoke_certificate,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
        "commands": {
            method: [args.python_exe, METHODS[method].script] for method in args.methods
        },
    }
    write_json(batch_dir / "r0_plan.json", plan)
    (batch_dir / "commands.txt").write_text(
        "\n".join(
            command_text([args.python_exe, METHODS[method].script]) for method in args.methods
        )
        + "\n",
        encoding="utf-8",
    )

    summaries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for method in args.methods:
        spec = METHODS[method]
        source_kind = "derived exact-shape smoke" if args.numerical_smoke else "unchanged official"
        print(f"\n=== R0 {method}: starting {source_kind} script {spec.script} ===")
        try:
            summaries.append(
                execute_one(args, repo, batch_dir, batch_id, spec, provenance, runtime)
            )
        except Exception as exc:
            failures.append({"method": method, "error": repr(exc)})
            print(f"R0 {method} failed: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    write_csv(batch_dir / "r0_summary.csv", summaries)
    final_manifest = {**plan, "summaries": summaries, "failures": failures}
    write_json(batch_dir / "r0_manifest.json", final_manifest)
    print(f"R0 artifacts: {batch_dir}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
