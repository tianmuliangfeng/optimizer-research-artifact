"""Internal worker binding the staged 1B controller to the capacity trainer."""

from pathlib import Path

import run_llama_swiglu_1b as quality


CAPACITY_TRAINER = Path(__file__).with_name("train_llama_swiglu_1b_capacity.py").resolve()
quality.TRAINER_PATH = CAPACITY_TRAINER
quality.base.training_script_path = lambda: CAPACITY_TRAINER


if __name__ == "__main__":
    quality.base.main()
