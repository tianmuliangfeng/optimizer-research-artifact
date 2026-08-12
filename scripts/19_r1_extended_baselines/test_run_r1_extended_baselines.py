from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_r1_extended_baselines.py")
SPEC = importlib.util.spec_from_file_location("run_r1_extended_baselines", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
r1x = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r1x
SPEC.loader.exec_module(r1x)


def args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "formal": False,
        "formal_smoke": False,
        "numerical_smoke": False,
        "pilot": False,
        "methods": list(r1x.ALLOWED_METHODS),
        "cells": None,
        "smoke_steps": 34,
        "pilot_steps": 1000,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class FormalProfileTests(unittest.TestCase):
    def test_formal_profile_is_frozen(self) -> None:
        value = args(formal=True)
        self.assertEqual(r1x.total_steps(value), 6200)
        self.assertEqual(r1x.warmdown_steps(value), 1800)
        self.assertEqual(r1x.protocol(value), r1x.FORMAL_PROTOCOL)
        self.assertEqual(
            [cell.cell_id for cell in r1x.selected_cells(value)],
            list(r1x.FORMAL_CELL_IDS),
        )

    def test_formal_smoke_uses_same_selected_cells_without_warmdown(self) -> None:
        value = args(formal_smoke=True)
        self.assertEqual(r1x.total_steps(value), 34)
        self.assertEqual(r1x.warmdown_steps(value), 1)
        self.assertEqual(r1x.protocol(value), r1x.FORMAL_SMOKE_PROTOCOL)
        self.assertEqual(
            [cell.cell_id for cell in r1x.selected_cells(value)],
            list(r1x.FORMAL_CELL_IDS),
        )

    def test_checkpoint_discovery_requires_at_most_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            folder = workspace / "logs" / "run"
            folder.mkdir(parents=True)
            self.assertIsNone(r1x.find_checkpoint(workspace))
            first = folder / "state_step006200.pt"
            first.write_bytes(b"checkpoint")
            self.assertEqual(r1x.find_checkpoint(workspace), first)
            (folder / "state_step006201.pt").write_bytes(b"duplicate")
            with self.assertRaises(RuntimeError):
                r1x.find_checkpoint(workspace)

    def test_curve_mean_is_trapezoidal(self) -> None:
        rows = [
            {"step": 0, "loss": 5.0},
            {"step": 100, "loss": 3.0},
            {"step": 300, "loss": 2.0},
        ]
        self.assertAlmostEqual(r1x.curve_mean(rows), (400.0 + 500.0) / 300.0)


if __name__ == "__main__":
    unittest.main()
