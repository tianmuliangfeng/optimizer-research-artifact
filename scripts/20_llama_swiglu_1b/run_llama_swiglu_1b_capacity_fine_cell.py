"""Internal fixed-accumulation worker for the 1B fine capacity sweep.

The quality runner remains byte-for-byte unchanged.  This worker relaxes its
global-batch-512 parser guard only inside a capacity-only subprocess and then
sets global batch to ``device_batch_size * accumulation_steps``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import run_llama_swiglu_1b as quality


CAPACITY_TRAINER = Path(__file__).with_name("train_llama_swiglu_1b_capacity.py").resolve()
INTERNAL_FLAG = "--capacity-accumulation-steps"
original_parse_args = quality.parse_args
original_common_config = quality.common_config


def split_internal_args(argv: list[str]) -> tuple[list[str], int, int]:
    cleaned = list(argv)
    try:
        flag_index = cleaned.index(INTERNAL_FLAG)
        accumulation_steps = int(cleaned[flag_index + 1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"missing or invalid internal flag {INTERNAL_FLAG}") from exc
    del cleaned[flag_index : flag_index + 2]
    if accumulation_steps != 8:
        raise RuntimeError("fine-capacity protocol requires exactly 8 accumulation steps")
    try:
        batch_index = cleaned.index("--device-batch-size")
        requested_batch = int(cleaned[batch_index + 1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("fine-capacity worker requires --device-batch-size") from exc
    if requested_batch <= 0:
        raise RuntimeError("device batch size must be positive")
    # Let the unchanged quality parser validate every other argument using a
    # legal placeholder divisor of 512.  The requested value is restored on
    # the returned namespace before any plan, audit, or training command runs.
    cleaned[batch_index + 1] = "8"
    return cleaned, requested_batch, accumulation_steps


def fine_parse_args() -> Any:
    original_argv = list(sys.argv)
    cleaned, requested_batch, accumulation_steps = split_internal_args(original_argv)
    try:
        sys.argv = cleaned
        args = original_parse_args()
    finally:
        sys.argv = original_argv
    if args.execution_stage != "smoke" or args.wandb_mode != "disabled":
        raise RuntimeError("fine-capacity worker is restricted to W&B-disabled smoke cells")
    args.device_batch_size = requested_batch
    args.capacity_accumulation_steps = accumulation_steps
    return args


def fine_common_config(args: Any, smoke: bool) -> dict[str, Any]:
    config = original_common_config(args, smoke)
    accumulation_steps = int(getattr(args, "capacity_accumulation_steps", 8))
    config["global_batch_size"] = int(args.device_batch_size) * accumulation_steps
    return config


quality.TRAINER_PATH = CAPACITY_TRAINER
quality.common_config = fine_common_config
quality.base.parse_args = fine_parse_args
quality.base.common_config = fine_common_config
quality.base.training_script_path = lambda: CAPACITY_TRAINER


if __name__ == "__main__":
    quality.base.main()
