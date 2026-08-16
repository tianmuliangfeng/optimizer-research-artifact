#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import analyze_global_diag_scale as analyze


HERE = Path(__file__).resolve().parent


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AnalyzeGlobalDiagScaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = json.loads(
            (HERE / "global_diag_scale_contract.json").read_text(encoding="utf-8")
        )

    def make_unit(self, root: Path, scale: str, seed: int) -> Path:
        attempt = root / "formal" / scale / f"seed{seed}" / "attempt_001"
        recipe = self.contract["frozen_recipes"][scale]
        summary = {
            "method": "global_diag",
            "seed": seed,
            "stage": "formal",
            "final_step": recipe["updates"],
            "train_tokens": recipe["train_tokens"],
            "final_val_loss": 3.0 + seed / 1_000_000,
            "best_val_loss": 2.9 + seed / 1_000_000,
            "tail5_mean": 3.01 + seed / 1_000_000,
            "normalized_auc": 3.2,
        }
        summary_path = attempt / "summary.json"
        write_json(summary_path, summary)
        scientific = {
            "passed": True,
            "method": "global_diag",
            "seed": seed,
            "stage": "formal",
            "data_fingerprint_sha256": self.contract["data"]["accepted_fingerprint_sha256"],
            "derived_source_sha256": "a" * 64,
            "source_snapshot_sha256": "b" * 64,
            "artifact_hashes": {"summary.json": sha256(summary_path)},
        }
        scientific_path = attempt / "scientific_manifest.json"
        write_json(scientific_path, scientific)
        cert = {
            "passed": True,
            "scale": scale,
            "method": "global_diag",
            "stage": "formal",
            "seed": seed,
            "data_fingerprint_sha256": self.contract["data"]["accepted_fingerprint_sha256"],
            "training_source_sha256": "a" * 64,
            "source_snapshot_manifest_sha256": "b" * 64,
            "summary_sha256": sha256(summary_path),
            "legacy_validator_manifest_sha256": sha256(scientific_path),
            "expected_memory": {"k_state_bytes": 1},
        }
        cert_path = attempt / "ex51_unit_manifest.json"
        write_json(cert_path, cert)
        return cert_path

    def test_collect_formal_requires_complete_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            certificates = [
                self.make_unit(root, scale, seed)
                for scale in analyze.SCALES
                for seed in analyze.FORMAL_SEEDS[scale]
            ]
            rows = analyze.collect_formal(root, self.contract)
            self.assertEqual(len(rows), 7)
            method_rows, paired = analyze.summarize(
                rows,
                analyze.controls(HERE / "frozen_scale_controls.csv"),
                float(self.contract["analysis"]["descriptive_margin"]),
            )
            global_rows = {
                row["scale"]: row
                for row in method_rows
                if row["method"] == "global_diag"
            }
            self.assertEqual(global_rows["275m"]["n"], 4)
            self.assertEqual(global_rows["455m"]["n"], 3)
            self.assertIn("final_val_loss_mean", global_rows["275m"])
            self.assertNotIn("tail5_mean", global_rows["275m"])
            self.assertEqual(
                {row["n"] for row in paired if row["scale"] == "275m"}, {4}
            )
            self.assertEqual(
                {row["n"] for row in paired if row["scale"] == "455m"}, {3}
            )
            tampered = json.loads(certificates[0].read_text(encoding="utf-8"))
            tampered["data_fingerprint_sha256"] = "0" * 64
            write_json(certificates[0], tampered)
            with self.assertRaisesRegex(RuntimeError, "lineage failure"):
                analyze.collect_formal(root, self.contract)


if __name__ == "__main__":
    unittest.main()
