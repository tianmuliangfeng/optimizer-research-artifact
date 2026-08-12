from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import run_r1_mousse as r1m


def args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "formal": False, "formal_smoke": False, "pilot": False,
        "numerical_smoke": False, "preflight": False,
        "pilot_steps": 1000, "smoke_steps": 34,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class MousseRunnerTests(unittest.TestCase):
    def test_formal_budget_is_frozen(self) -> None:
        value = args(formal=True)
        self.assertEqual(r1m.total_steps(value), 6200)
        self.assertEqual(r1m.warmdown_steps(value), 1800)
        self.assertEqual(r1m.protocol(value), r1m.FORMAL_PROTOCOL)

    def test_formal_smoke_crosses_four_refreshes(self) -> None:
        value = args(formal_smoke=True)
        self.assertEqual(r1m.total_steps(value), 34)
        self.assertEqual(1 + (r1m.total_steps(value) - 1) // 10, 4)

    def test_center_tie_rule(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "pilot_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            summaries = [
                {"cell_id": "mousse_lr080", "matrix_lr": 0.012, "final_val_loss": 3.7000},
                {"cell_id": "mousse_lr100", "matrix_lr": 0.015, "final_val_loss": 3.7019},
                {"cell_id": "mousse_lr120", "matrix_lr": 0.018, "final_val_loss": 3.7100},
            ]
            selection = r1m.make_selection(root, summaries, manifest)
            payload = json.loads(selection.read_text(encoding="utf-8"))
            self.assertEqual(payload["selected_cell_id"], "mousse_lr100")

    def test_non_tied_winner_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "pilot_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            summaries = [
                {"cell_id": "mousse_lr080", "matrix_lr": 0.012, "final_val_loss": 3.7000},
                {"cell_id": "mousse_lr100", "matrix_lr": 0.015, "final_val_loss": 3.7021},
                {"cell_id": "mousse_lr120", "matrix_lr": 0.018, "final_val_loss": 3.7100},
            ]
            payload = json.loads(r1m.make_selection(root, summaries, manifest).read_text(encoding="utf-8"))
            self.assertEqual(payload["selected_cell_id"], "mousse_lr080")

    def test_validation_grid_includes_terminal_step(self) -> None:
        self.assertEqual(r1m.validation_steps(34, 100), [0, 34])
        self.assertEqual(r1m.validation_steps(1000, 100)[-1], 1000)


if __name__ == "__main__":
    unittest.main()
