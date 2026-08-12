"""Capacity-only instrumentation for the audited extended LLaMA-124M trainer.

The optimizer and model implementation stay in ``train_llama_swiglu_extended``.
This wrapper only enriches completed summaries with peak CUDA reserved memory
and marks the run as ineligible for quality or timing comparisons.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ADAPTER = HERE / "train_llama_swiglu_extended.py"


def load_adapter() -> Any:
    spec = importlib.util.spec_from_file_location(
        "llama_swiglu_extended_capacity_adapter", ADAPTER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extended adapter from {ADAPTER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    import torch

    adapter = load_adapter()
    original_load_module = adapter.load_module

    def capacity_load_module(name: str, path: Path) -> Any:
        module = original_load_module(name, path)
        if path.resolve() != adapter.BASE_TRAINER_PATH.resolve():
            return module
        original_atomic_write_json = module.atomic_write_json

        def capacity_atomic_write_json(
            output_path: Path, payload: dict[str, Any]
        ) -> None:
            if output_path.name == "summary.json" and payload.get("status") == "completed":
                payload = dict(payload)
                reserved = int(torch.cuda.max_memory_reserved())
                payload["peak_reserved_bytes"] = reserved
                payload["peak_reserved_mib"] = reserved / (1024**2)
                payload["evidence_class"] = "capacity_only"
                payload["quality_comparable"] = False
                payload["timing_comparable"] = False
            original_atomic_write_json(output_path, payload)

        module.atomic_write_json = capacity_atomic_write_json
        return module

    adapter.load_module = capacity_load_module
    adapter.main()


if __name__ == "__main__":
    main()
