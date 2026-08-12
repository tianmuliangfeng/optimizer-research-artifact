"""Capacity-only 1.014B trainer wrapper.

This keeps the audited base trainer immutable and adds only peak CUDA reserved
memory to the completed summary.  It is not a quality-training entry point.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Any


PROFILE = {
    "name": "llama_swiglu_1b_v1",
    "n_layer": 18,
    "n_head": 16,
    "n_embd": 2048,
    "intermediate_size": 5504,
    "expected_parameter_count": 1_013_690_368,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def argument_value(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"missing required trainer argument {name}") from exc


def load_audited_trainer() -> Any:
    source_text = os.environ.get("LLAMA_1B_BASE_TRAINER")
    expected_sha = os.environ.get("LLAMA_1B_BASE_TRAINER_SHA256")
    if not source_text or not expected_sha:
        raise RuntimeError("controller did not bind the audited base trainer source")
    source = Path(source_text).resolve()
    if not source.is_file() or sha256_file(source) != expected_sha:
        raise RuntimeError("audited base trainer is missing or changed")
    output_dir = Path(argument_value("--output-dir")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_source = output_dir / "train_llama_swiglu_base.py"
    if not artifact_source.exists():
        shutil.copy2(source, artifact_source)
    if sha256_file(artifact_source) != expected_sha:
        raise RuntimeError("artifact copy of the base trainer has a different hash")
    spec = importlib.util.spec_from_file_location("llama_swiglu_1b_capacity_base", artifact_source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import audited trainer from {artifact_source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    trainer = load_audited_trainer()
    original_config = trainer.ModelConfig
    original_architecture_audit = trainer.architecture_audit
    original_atomic_write_json = trainer.atomic_write_json

    def model_config_1b(*, sequence_length: int = 1024, **_: Any) -> Any:
        return original_config(
            n_layer=PROFILE["n_layer"],
            n_head=PROFILE["n_head"],
            n_embd=PROFILE["n_embd"],
            intermediate_size=PROFILE["intermediate_size"],
            sequence_length=sequence_length,
        )

    def architecture_audit_1b(model: Any) -> dict[str, Any]:
        payload = original_architecture_audit(model)
        payload["architecture"] = PROFILE["name"]
        payload["profile"] = dict(PROFILE)
        payload["base_trainer_sha256"] = os.environ["LLAMA_1B_BASE_TRAINER_SHA256"]
        if payload["parameter_count"] != PROFILE["expected_parameter_count"]:
            raise RuntimeError("1B parameter count drift")
        return payload

    def capacity_atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        if path.name == "summary.json" and payload.get("status") == "completed":
            payload = dict(payload)
            reserved = int(trainer.torch.cuda.max_memory_reserved())
            payload["peak_reserved_bytes"] = reserved
            payload["peak_reserved_mib"] = reserved / (1024**2)
            payload["evidence_class"] = "capacity_only"
            payload["timing_comparable"] = False
        original_atomic_write_json(path, payload)

    trainer.ModelConfig = model_config_1b
    trainer.architecture_audit = architecture_audit_1b
    trainer.atomic_write_json = capacity_atomic_write_json
    trainer.main()


if __name__ == "__main__":
    main()
