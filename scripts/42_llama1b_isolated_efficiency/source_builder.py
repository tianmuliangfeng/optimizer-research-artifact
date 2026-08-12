#!/usr/bin/env python3
"""Build the minimally derived LLaMA trainer used by experiment 42."""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BASE_TRAINER = (
    REPO_ROOT
    / "scripts"
    / "17_llama_swiglu_validation"
    / "train_llama_swiglu.py"
)
PROFILE_WRAPPER = (
    REPO_ROOT
    / "scripts"
    / "20_llama_swiglu_1b"
    / "train_llama_swiglu_1b.py"
)
PINNED_BASE_SHA256 = "b72eb0d2a1dfa91b61cd49b4784b3e0739ecebc2fd3228b8f719cec125706f2a"
PINNED_WRAPPER_SHA256 = "043c758f3d5eb5d1abc9e1f9029a8d085a238cf169ef69ba86580014699dc401"
PINNED_DERIVED_SHA256 = "a21f364b73ea859e1ca62bc75600de348b00a0ce5fcbd315ed7e848eb2b4666a"


@dataclass(frozen=True)
class SourceBundle:
    base_source: str
    base_sha256: str
    derived_source: str
    derived_sha256: str
    derived_diff: str
    wrapper_source: str
    wrapper_sha256: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(source: str, needle: str, replacement: str, label: str) -> str:
    count = source.count(needle)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source anchor, observed {count}")
    return source.replace(needle, replacement, 1)


def build_source_bundle() -> SourceBundle:
    if sha256_file(BASE_TRAINER) != PINNED_BASE_SHA256:
        raise RuntimeError("experiment-17 base trainer hash drift")
    if sha256_file(PROFILE_WRAPPER) != PINNED_WRAPPER_SHA256:
        raise RuntimeError("experiment-20 LLaMA-1B wrapper hash drift")

    base_source = BASE_TRAINER.read_text(encoding="utf-8")
    wrapper_source = PROFILE_WRAPPER.read_text(encoding="utf-8")
    derived = base_source
    peak_state_anchor = """    torch.cuda.reset_peak_memory_stats()
    atomic_write_json(
"""
    peak_state_replacement = """    timing_peak_reset_performed = False
    allocated_bytes_at_timing_reset = 0
    reserved_bytes_at_timing_reset = 0
    timed_peak_allocated_bytes = 0
    timed_peak_reserved_bytes = 0
    allocated_bytes_at_timed_end = 0
    reserved_bytes_at_timed_end = 0
    atomic_write_json(
"""
    derived = replace_once(
        derived,
        peak_state_anchor,
        peak_state_replacement,
        "timed-window peak state",
    )
    timed_update_anchor = """        completed_steps += 1
        if completed_steps > 32:
            steady_train_s += update_s
            steady_steps += 1
"""
    timed_update_replacement = """        completed_steps += 1
        if completed_steps == 32:
            # Compile, initial validation, and warmup allocation behavior are
            # outside the paper timing/memory window.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            allocated_bytes_at_timing_reset = int(torch.cuda.memory_allocated())
            reserved_bytes_at_timing_reset = int(torch.cuda.memory_reserved())
            timing_peak_reset_performed = True
        if completed_steps > 32:
            steady_train_s += update_s
            steady_steps += 1
            timed_peak_allocated_bytes = max(
                timed_peak_allocated_bytes, int(torch.cuda.max_memory_allocated())
            )
            timed_peak_reserved_bytes = max(
                timed_peak_reserved_bytes, int(torch.cuda.max_memory_reserved())
            )
            allocated_bytes_at_timed_end = int(torch.cuda.memory_allocated())
            reserved_bytes_at_timed_end = int(torch.cuda.memory_reserved())
"""
    derived = replace_once(
        derived,
        timed_update_anchor,
        timed_update_replacement,
        "post-warmup timed memory window",
    )
    summary_anchor = """        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_allocated_mib": peak_allocated_bytes / (1024**2),
"""
    summary_replacement = """        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_allocated_mib": peak_allocated_bytes / (1024**2),
        "timed_training_peak_allocated_bytes": timed_peak_allocated_bytes,
        "timed_training_peak_allocated_mib": timed_peak_allocated_bytes / (1024**2),
        "timed_training_peak_reserved_bytes": timed_peak_reserved_bytes,
        "timed_training_peak_reserved_mib": timed_peak_reserved_bytes / (1024**2),
        "peak_memory_stats_reset": timing_peak_reset_performed,
        "peak_reset_after_completed_step": 32,
        "timed_step_first": 33,
        "timed_step_last": completed_steps,
        "allocated_bytes_at_timing_reset": allocated_bytes_at_timing_reset,
        "reserved_bytes_at_timing_reset": reserved_bytes_at_timing_reset,
        "allocated_bytes_at_timed_end": allocated_bytes_at_timed_end,
        "reserved_bytes_at_timed_end": reserved_bytes_at_timed_end,
        "peak_measurement_scope": (
            "CUDA peak statistics reset after completed update 32; "
            "training updates 33 through the final update; final validation excluded"
        ),
        "timing_measurement_scope": (
            "CUDA-synchronized optimizer updates only; first 32 updates excluded"
        ),
"""
    derived = replace_once(
        derived,
        summary_anchor,
        summary_replacement,
        "summary peak-reserved fields",
    )
    compile(derived, str(BASE_TRAINER), "exec")
    diff = "".join(
        difflib.unified_diff(
            base_source.splitlines(keepends=True),
            derived.splitlines(keepends=True),
            fromfile="scripts/17_llama_swiglu_validation/train_llama_swiglu.py",
            tofile="experiment42/train_llama_swiglu_efficiency_base.py",
        )
    )
    if not diff:
        raise RuntimeError("derived efficiency source unexpectedly equals the base")
    derived_sha256 = sha256_bytes(derived.encode("utf-8"))
    if derived_sha256 != PINNED_DERIVED_SHA256:
        raise RuntimeError(
            "derived efficiency trainer differs from the frozen contract: "
            f"{derived_sha256} != {PINNED_DERIVED_SHA256}"
        )
    return SourceBundle(
        base_source=base_source,
        base_sha256=sha256_bytes(base_source.encode("utf-8")),
        derived_source=derived,
        derived_sha256=derived_sha256,
        derived_diff=diff,
        wrapper_source=wrapper_source,
        wrapper_sha256=sha256_bytes(wrapper_source.encode("utf-8")),
    )


def materialize(bundle: SourceBundle, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / "train_llama_swiglu_efficiency_base.py"
    wrapper_path = output_dir / "train_llama_swiglu_1b.py"
    diff_path = output_dir / "train_llama_swiglu_efficiency_base.diff"
    # Byte writes preserve the frozen LF representation on Windows and Linux.
    base_path.write_bytes(bundle.derived_source.encode("utf-8"))
    wrapper_path.write_bytes(bundle.wrapper_source.encode("utf-8"))
    diff_path.write_bytes(bundle.derived_diff.encode("utf-8"))
    observed = {
        "derived_base": sha256_file(base_path),
        "profile_wrapper": sha256_file(wrapper_path),
        "source_diff": sha256_file(diff_path),
    }
    if observed["derived_base"] != bundle.derived_sha256:
        raise RuntimeError("materialized derived trainer hash mismatch")
    if observed["profile_wrapper"] != bundle.wrapper_sha256:
        raise RuntimeError("materialized profile wrapper hash mismatch")
    return observed
