#!/usr/bin/env python3
"""CPU-only orchestration tests for Experiment 50."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from run_global_diag_suite import accepted_batch, ensure_source_snapshot, read_json


class GlobalDiagSuiteTests(unittest.TestCase):
    def test_snapshot_is_hash_sealed_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = types.SimpleNamespace(
                run_dir=Path(temp),
                repo=REPO,
            )
            first = ensure_source_snapshot(args)
            second = ensure_source_snapshot(args)
            self.assertEqual(first, second)
            manifest = read_json(first / "source_snapshot_manifest.json")
            self.assertTrue(manifest["passed"])
            self.assertGreater(manifest["file_count"], 10)
            self.assertTrue(
                (
                    first
                    / "scripts/50_r1_global_activation_diag/run_global_diag.py"
                ).is_file()
            )
            self.assertFalse(any(first.rglob("*.pyc")))
            self.assertFalse(any(first.rglob("__pycache__")))

    def test_local_wandb_incomplete_is_valid_formal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "batch"
            root.mkdir()
            path = root / "r1_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "completed_valid_local_wandb_incomplete",
                        "failures": [],
                        "summaries": [{"method": "global_diag"}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(accepted_batch(Path(temp), pilot=False), path)
            self.assertIsNone(accepted_batch(Path(temp), pilot=True))

    def test_failed_or_duplicate_units_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "batch"
            root.mkdir()
            (root / "r1_manifest.json").write_text(
                json.dumps(
                    {"status": "failed", "failures": ["x"], "summaries": []}
                ),
                encoding="utf-8",
            )
            self.assertIsNone(accepted_batch(Path(temp), pilot=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
