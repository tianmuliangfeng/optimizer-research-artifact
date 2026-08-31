#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ex55_analyzer", HERE / "analyze_fresh_seed_panel.py")
assert SPEC and SPEC.loader
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


class AnalyzeFreshSeedPanelTests(unittest.TestCase):
    def test_canonical_historical_maps_frozen_winner_cells(self) -> None:
        self.assertEqual(ANALYZER.canonical_historical({"method": "adamw_low"}), "adamw")
        self.assertEqual(ANALYZER.canonical_historical({"method": "moonlight_r1scale"}), "moonlight")
        self.assertEqual(ANALYZER.canonical_historical({"method": "malter_eq17"}), "malter_eq17")
        payload = {
            "summaries": [{
                "method": "moonlight_muon", "cell_id": "moonlight_r1scale",
                "controlled_seed": 2027, "total_steps": 6200,
                "evidence_valid": True,
            }]
        }
        self.assertEqual(
            ANALYZER.pick_summary(payload, "moonlight", "moonlight_r1scale")["method"],
            "moonlight_muon",
        )

    def test_leave_out_is_explicitly_n2_descriptive(self) -> None:
        rows = []
        for offset, method in enumerate(ANALYZER.CANONICAL_METHODS):
            for seed in (2024, 2025, 2026):
                rows.append({"method": method, "seed": seed, "final_val_loss": 3.0 + offset / 100 + (seed - 2024) / 1000})
        result = ANALYZER.leave_out_summary(rows)
        self.assertEqual(len(result), 10)
        self.assertTrue(all(row["n"] == 2 for row in result))
        self.assertTrue(all(row["inferential_ci_or_p_value"] == "not_reported_n2_sensitivity_only" for row in result))

    def test_historical_annotation_uses_contract_frozen_selected_cells(self) -> None:
        rows = []
        selected = {
            method: {
                "adamw": "adamw_low", "normuon": "normuon_r1scale",
                "moonlight": "moonlight_r1scale", "mousse": "mousse_lr100",
                "malt": "malt_lr0125", "malter_eq17": "malter_eq17_lr015",
            }.get(method, method)
            for method in ANALYZER.CANONICAL_METHODS
        }
        for method in ANALYZER.CANONICAL_METHODS:
            for seed in (2024, 2025, 2026):
                rows.append({
                    "method": method, "seed": str(seed), "init_sha256": "a" * 64,
                    "initial_val_loss": "10", "final_val_loss": "3", "best_val_loss": "3",
                    "tail5_val_loss_mean": "3", "normalized_val_auc": "3.5",
                    "peak_memory_mib": "100", "optimizer_state_mib": "1",
                })
        normalized = ANALYZER.normalize_historical(rows, selected)
        by_method = {row["method"]: row["selected_cell"] for row in normalized}
        self.assertEqual(by_method, selected)

    def test_pick_summary_rejects_nonformal_seed(self) -> None:
        payload = {"summaries": [{
            "method": "diag", "controlled_seed": 2026, "total_steps": 6200,
            "evidence_valid": True,
        }]}
        with self.assertRaises(RuntimeError):
            ANALYZER.pick_summary(payload, "diag", "diag")

    def test_normalized_fresh_row_requires_init_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manifest = Path(raw) / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            row = {
                "initial_val_loss": 10.0, "final_val_loss": 3.0, "best_val_loss": 3.0,
                "tail5_val_loss_mean": 3.1, "normalized_val_auc": 3.5,
                "peak_memory_allocated_mib": 100, "optimizer_state_bytes": 1048576,
            }
            with self.assertRaises(RuntimeError):
                ANALYZER.normalized_fresh_row("diag", "diag", row, manifest)

    def test_core_summary_uses_hash_bound_manifest_milestones_for_initial_and_tail(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "r1_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            row = {
                "run_name": "core_run", "init_sha256": "a" * 64,
                "final_val_loss": 3.0, "best_val_loss": 3.0, "val_curve_mean": 3.5,
                "peak_memory_allocated_mib": 100, "optimizer_state_bytes": 1048576,
                "val_loss_step_0": 10.0, "val_loss_step_5800": 3.4,
                "val_loss_step_5900": 3.3, "val_loss_step_6000": 3.2,
                "val_loss_step_6100": 3.1, "val_loss_step_6200": 3.0,
            }
            normalized = ANALYZER.normalized_fresh_row("diag", "diag", row, manifest)
            self.assertEqual(normalized["initial_val_loss"], 10.0)
            self.assertAlmostEqual(normalized["tail5_val_loss_mean"], 3.2)

    def test_formal_tail5_is_reconstructed_from_hash_bound_child_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run_dir = root / "run_block4"
            run_dir.mkdir()
            checkpoint = run_dir / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            summary = {
                "method": "block4", "controlled_seed": 2027, "total_steps": 6200,
                "evidence_valid": True, "init_sha256": "a" * 64,
                "initial_val_loss": 10.0, "final_val_loss": 3.0,
                "best_val_loss": 2.9, "val_curve_mean": 3.5,
                "peak_memory_allocated_mib": 100, "optimizer_state_bytes": 1048576,
                "checkpoint_path": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
                # Deliberately expose fewer than five aggregate milestones: the
                # repaired analyzer must not mistake this for a short trajectory.
                "val_loss_step_6100": 3.1, "val_loss_step_6200": 3.0,
                "run_name": run_dir.name,
            }
            manifest = root / "r1_manifest.json"
            manifest.write_text(json.dumps({"summaries": [summary]}) + "\n", encoding="utf-8")
            (run_dir / "r1_summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"status": "completed_valid", "summary": summary}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "r1_metrics.csv").write_text(
                "event,step,loss\n"
                "validation,0,10.0\n"
                "train,6000,2.8\n"
                "validation,5800,3.4\n"
                "validation,5900,3.3\n"
                "validation,6000,3.2\n"
                "validation,6100,3.1\n"
                "validation,6200,3.0\n",
                encoding="utf-8",
            )
            evidence = ANALYZER.formal_metrics_tail5_evidence(
                manifest, summary, "block4", "block4",
                seed=2027, formal_steps=6200, validation_every=100,
            )
            self.assertEqual(evidence["tail5_steps"], [5800, 5900, 6000, 6100, 6200])
            self.assertAlmostEqual(evidence["tail5_val_loss_mean"], 3.2)
            self.assertEqual(evidence["metrics"]["bytes"], (run_dir / "r1_metrics.csv").stat().st_size)
            self.assertEqual(evidence["metrics"]["sha256"], ANALYZER.sha256_file(run_dir / "r1_metrics.csv"))
            normalized = ANALYZER.normalized_fresh_row(
                "block4", "block4", summary, manifest, evidence,
            )
            self.assertEqual(normalized["tail5_source"], "accepted_formal_child_metrics_csv")
            self.assertAlmostEqual(normalized["tail5_val_loss_mean"], 3.2)

    def test_formal_tail5_rejects_missing_required_milestone(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run_dir = root / "run_diag"
            run_dir.mkdir()
            summary = {
                "method": "diag", "controlled_seed": 2027, "total_steps": 6200,
                "evidence_valid": True, "init_sha256": "a" * 64,
                "final_val_loss": 3.0, "run_name": run_dir.name,
            }
            manifest = root / "r1_manifest.json"
            manifest.write_text(json.dumps({"summaries": [summary]}) + "\n", encoding="utf-8")
            (run_dir / "r1_summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"status": "completed_valid"}) + "\n", encoding="utf-8",
            )
            (run_dir / "r1_metrics.csv").write_text(
                "event,step,loss\nvalidation,0,10\nvalidation,5800,3.4\n"
                "validation,5900,3.3\nvalidation,6000,3.2\nvalidation,6200,3.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "missing required tail-5 steps"):
                ANALYZER.formal_metrics_tail5_evidence(
                    manifest, summary, "diag", "diag",
                    seed=2027, formal_steps=6200, validation_every=100,
                )

    def test_historical_only_mode_materializes_preformal_sensitivity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            historical = root / "historical.csv"
            fields = [
                "method", "seed", "init_sha256", "initial_val_loss", "final_val_loss",
                "best_val_loss", "tail5_val_loss_mean", "normalized_val_auc",
                "peak_memory_mib", "optimizer_state_mib",
            ]
            with historical.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for offset, method in enumerate(ANALYZER.CANONICAL_METHODS):
                    for seed in (2024, 2025, 2026):
                        writer.writerow({
                            "method": method, "seed": seed, "init_sha256": "",
                            "initial_val_loss": 10, "final_val_loss": 3 + offset / 100,
                            "best_val_loss": 3, "tail5_val_loss_mean": 3.1,
                            "normalized_val_auc": 3.5, "peak_memory_mib": 100,
                            "optimizer_state_mib": 1,
                        })
            digest = hashlib.sha256(historical.read_bytes()).hexdigest()
            selected = {
                method: {
                    "adamw": "adamw_low", "normuon": "normuon_r1scale",
                    "moonlight": "moonlight_r1scale", "mousse": "mousse_lr100",
                    "malt": "malt_lr0125", "malter_eq17": "malter_eq17_lr015",
                }.get(method, method)
                for method in ANALYZER.CANONICAL_METHODS
            }
            contract = root / "contract.json"
            contract.write_text(json.dumps({
                "experiment_id": "55_r1_fresh_seed_baseline_fairness",
                "accepted_inputs": {"historical_panel_sha256": digest},
                "methods": [{"method": method, "selected_cell": selected[method]} for method in ANALYZER.CANONICAL_METHODS],
            }) + "\n", encoding="utf-8")
            output = root / "analysis_preformal"
            argv = [
                "analyze", "--run-dir", str(root), "--contract", str(contract),
                "--historical-panel", str(historical), "--output-dir", str(output),
                "--historical-only",
            ]
            with mock.patch.object(sys, "argv", argv):
                ANALYZER.main()
            manifest = json.loads((output / "analysis_preformal_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["passed"])
            self.assertEqual(manifest["leave_out_seeds"], [2024, 2025])
            self.assertTrue((output / "leave_selection_seed_out_2024_2025.csv").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
