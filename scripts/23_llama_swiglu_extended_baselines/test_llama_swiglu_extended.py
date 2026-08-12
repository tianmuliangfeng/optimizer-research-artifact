from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def load(name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("llama_extended_runner_test", "run_llama_swiglu_extended.py")
adapter = load("llama_extended_adapter_test", "train_llama_swiglu_extended.py")


class ExtendedLlamaTests(unittest.TestCase):
    def args(self):
        return type(
            "Args",
            (),
            {
                "python_exe": Path("/venv/bin/python"),
                "official_repo": Path("/official"),
                "device_batch_size": 64,
                "checkpoint_every": 128,
            },
        )()

    def test_equal_pilot_budget_per_new_method(self) -> None:
        counts = {}
        for cell in runner.CELLS:
            counts[cell.method] = counts.get(cell.method, 0) + 1
        self.assertEqual(counts, {"normuon": 3, "moonlight_muon": 3})

    def test_grid_is_frozen_around_r1_selected_cells(self) -> None:
        self.assertEqual(
            [cell.matrix_lr for cell in runner.CELLS if cell.method == "normuon"],
            [0.005, 0.01, 0.02],
        )
        self.assertEqual(
            [cell.matrix_lr for cell in runner.CELLS if cell.method == "moonlight_muon"],
            [0.001, 0.0018, 0.003],
        )

    def test_pilot_is_prefix_without_checkpoint(self) -> None:
        cell = runner.CELL_BY_ID["normuon_r1scale"]
        command = runner.train_command(
            self.args(), cell, 2026, Path("/results/cell"), "pilot"
        )
        joined = " ".join(command)
        self.assertIn("--num-iterations 1000", joined)
        self.assertIn("--warmdown-iters 1", joined)
        self.assertIn("--checkpoint-every 0", joined)
        self.assertIn("--no-save-final", command)

    def test_formal_has_full_budget_and_resume(self) -> None:
        cell = runner.CELL_BY_ID["moonlight_high"]
        command = runner.train_command(
            self.args(), cell, 2024, Path("/results/cell"), "formal"
        )
        joined = " ".join(command)
        self.assertIn("--num-iterations 6200", joined)
        self.assertIn("--warmdown-iters 1800", joined)
        self.assertIn("--resume auto", joined)
        self.assertNotIn("--no-save-final", command)

    def test_formal_accepts_a_frozen_moonlight_only_subset(self) -> None:
        self.assertEqual(
            runner.validate_formal_cell_ids(["moonlight_high"]),
            ["moonlight_high"],
        )
        self.assertEqual(
            runner.validate_formal_cell_ids(
                ["moonlight_high", "normuon_r1scale"]
            ),
            ["moonlight_high", "normuon_r1scale"],
        )

    def test_formal_rejects_unfrozen_or_duplicate_cells(self) -> None:
        with self.assertRaisesRegex(ValueError, "pilot-frozen"):
            runner.validate_formal_cell_ids(["moonlight_r1scale"])
        with self.assertRaisesRegex(ValueError, "unique"):
            runner.validate_formal_cell_ids(["moonlight_high", "moonlight_high"])

    def test_adapter_consumes_only_extended_argument(self) -> None:
        argv, decay = adapter.extract_extended_args(
            ["train.py", "--method", "normuon", "--extended-weight-decay", "0.01"]
        )
        self.assertEqual(decay, 0.01)
        self.assertEqual(argv, ["train.py", "--method", "normuon"])

    def test_source_bundle_covers_all_implementation_files(self) -> None:
        bundle = runner.source_bundle()
        self.assertEqual(
            set(bundle["sha256"]),
            {"runner", "adapter", "base_trainer", "extended_optimizers"},
        )
        self.assertEqual(len(bundle["bundle_sha256"]), 64)

    def test_pilot_selection_certificate_binds_frozen_recipe(self) -> None:
        cell = runner.CELL_BY_ID["moonlight_high"]
        bundle = runner.source_bundle()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_path = root / "summary.json"
            summary = {
                "status": "completed",
                "completed_steps": 1000,
                "seed": 2026,
                "resume_count": 0,
                "checkpoint_path": "",
                "extended_optimizer": {
                    "method": cell.method,
                    "auxiliary_lr": cell.auxiliary_lr,
                    "matrix_lr": cell.matrix_lr,
                    "weight_decay": cell.weight_decay,
                },
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            manifest = {
                "status": "completed",
                "kind": "pilot",
                "protocol": "llama_swiglu_124m_extended_progressive_v1",
                "seeds": [2026],
                "failed_tasks": {},
                "data": {"fingerprint": "data"},
                "runtime": {},
                "source_bundle": {
                    "sha256": {
                        key: bundle["sha256"][key]
                        for key in (
                            "adapter",
                            "base_trainer",
                            "extended_optimizers",
                        )
                    }
                },
                "cells": [runner.asdict(cell)],
                "completed_tasks": [
                    {
                        "cell_id": cell.cell_id,
                        "summary_path": str(summary_path),
                        "summary_sha256": runner.sha256_file(summary_path),
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            args = type("Args", (), {"cells": ["moonlight_high"]})()
            observed = runner.validate_pilot_selection(
                manifest_path,
                args,
                runtime={},
                data={"fingerprint": "data"},
                bundle=bundle,
            )
            self.assertEqual(observed["status"], "completed")


if __name__ == "__main__":
    unittest.main()
