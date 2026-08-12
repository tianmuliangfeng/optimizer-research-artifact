#!/usr/bin/env python3
"""Unit tests for the experiment-44 independent analyzer."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location(
    "analyze_record17", HERE / "analyze_record17.py"
)
assert SPEC and SPEC.loader
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
import run_record17_suite as S


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class Record17AnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.contract = self.root / "record17_contract.json"
        write_json(
            self.contract,
            {
                "schema_version": 1,
                "seeds": list(A.EXPECTED_SEEDS),
                "methods": list(A.EXPECTED_METHODS),
                "training": {
                    "total_steps": A.EXPECTED_TOTAL_STEPS,
                    "tokens_per_update": A.EXPECTED_TOKENS_PER_UPDATE,
                    "train_tokens": A.EXPECTED_TRAIN_TOKENS,
                },
                "analysis": {
                    "common_target_loss": A.COMMON_TARGET_LOSS,
                    "practical_margin": A.PRACTICAL_MARGIN,
                },
            },
        )
        self.contract_sha = sha256(self.contract)
        self._make_complete_run()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _loss(self, seed: int, method: str, step: int) -> float:
        method_shift = {
            "muon": 0.006,
            "original_newton_muon": 0.001,
            "selective_none": -0.001,
            "selective_diag": -0.002,
        }[method]
        seed_shift = (seed - 2024) * 0.0002
        progress = step / A.EXPECTED_TOTAL_STEPS
        return 3.55 - 0.28 * progress + method_shift * progress + seed_shift

    def _make_complete_run(self) -> None:
        source_snapshot = "a" * 64
        data_fingerprint = "b" * 64
        for seed in A.EXPECTED_SEEDS:
            init = hashlib.sha256(f"init:{seed}".encode()).hexdigest()
            for method in A.EXPECTED_METHODS:
                cell = self.run / "formal" / f"seed{seed}" / method
                attempt = cell / "attempt_001"
                attempt.mkdir(parents=True)
                metrics_path = attempt / "metrics.csv"
                with metrics_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=(
                            "step",
                            "total_steps",
                            "val_loss",
                            "train_time_ms",
                            "step_avg_ms",
                            "tokens",
                        ),
                    )
                    writer.writeheader()
                    for step in A.EXPECTED_VALIDATION_STEPS:
                        writer.writerow(
                            {
                                "step": step,
                                "total_steps": A.EXPECTED_TOTAL_STEPS,
                                "val_loss": repr(self._loss(seed, method, step)),
                                "train_time_ms": 0,
                                "step_avg_ms": 0,
                                "tokens": step * A.EXPECTED_TOKENS_PER_UPDATE,
                            }
                        )
                rows = A.validation_rows(metrics_path)
                computed = A.recompute_metrics(
                    rows, A.COMMON_TARGET_LOSS, A.EXPECTED_TOKENS_PER_UPDATE
                )
                summary = {
                    "final_val_loss": computed["final_val_loss"],
                    "best_val_loss": computed["best_val_loss"],
                    "final_step": A.EXPECTED_TOTAL_STEPS,
                    "train_tokens": A.EXPECTED_TRAIN_TOKENS,
                    "tokens_per_update": A.EXPECTED_TOKENS_PER_UPDATE,
                    "tail5_mean": computed["tail5_mean"],
                    "normalized_auc": computed["normalized_auc"],
                    "steps_to_target": computed["steps_to_target"],
                    "tokens_to_target": computed["tokens_to_target"],
                    "peak_memory_allocated_bytes": 1_000_000,
                    "peak_memory_reserved_bytes": 2_000_000,
                    "peak_memory_scope": (
                        "counted_run_after_warmup_reset_including_validation"
                    ),
                    "k_state_bytes": {
                        "muon": 0,
                        "original_newton_muon": 3000,
                        "selective_none": 1000,
                        "selective_diag": 2000,
                    }[method],
                    "optimizer_state_bytes": 3_000_000,
                }
                summary_path = attempt / "summary.json"
                write_json(summary_path, summary)
                cproj_mode = {
                    "muon": "not_applicable",
                    "original_newton_muon": "block4",
                    "selective_none": "none",
                    "selective_diag": "diag",
                }[method]
                command_path = attempt / "command.json"
                write_json(
                    command_path,
                    {
                        "command": ["python", "derived_source.py"],
                        "environment": {
                            "RECORD17_METHOD": method,
                            "RECORD17_CPROJ_K_MODE": cproj_mode,
                        },
                    },
                )
                manifest = {
                    "passed": True,
                    "status": "scientifically_complete",
                    "stage": "formal",
                    "seed": seed,
                    "method": method,
                    "cproj_k_mode": cproj_mode,
                    "cell_key": A.C.cell_key("formal", seed, method),
                    "contract_sha256": self.contract_sha,
                    "source_snapshot_sha256": source_snapshot,
                    "derived_source_sha256": hashlib.sha256(
                        b"source:record17_environment_dispatched"
                    ).hexdigest(),
                    "data_fingerprint_sha256": data_fingerprint,
                    "init_sha256": init,
                    "total_steps": A.EXPECTED_TOTAL_STEPS,
                    "train_tokens": A.EXPECTED_TRAIN_TOKENS,
                    "timing_eligible": False,
                    "artifact_hashes": {
                        "command.json": sha256(command_path),
                        "metrics.csv": sha256(metrics_path),
                        "summary.json": sha256(summary_path),
                    },
                }
                manifest_path = attempt / "scientific_manifest.json"
                write_json(manifest_path, manifest)
                write_json(
                    cell / "accepted.json",
                    {
                        "cell_key": A.C.cell_key("formal", seed, method),
                        "attempt_dir": "attempt_001",
                        "scientific_manifest_sha256": sha256(manifest_path),
                    },
                )

    def test_complete_analysis_and_primary_delta(self) -> None:
        output = self.run / "analysis"
        manifest = A.analyze(self.run, self.contract, output)
        self.assertTrue(manifest["passed"])
        self.assertEqual(manifest["accepted_formal_cells"], 12)
        self.assertEqual(len(manifest["accepted_cell_fingerprints"]), 12)
        self.assertEqual(
            manifest["accepted_cells_fingerprint_sha256"],
            A.accepted_cells_fingerprint_sha256(
                manifest["accepted_cell_fingerprints"]
            ),
        )
        self.assertEqual(
            manifest["primary_classifications"],
            manifest["decision"]["primary_classifications"],
        )
        reusable, checks = S.validate_reusable_analysis(
            output / "record17_analysis_manifest.json", self.run
        )
        self.assertIsNotNone(reusable, checks)
        self.assertTrue(checks["passed_all"])
        final_fields = S.final_analysis_fields(manifest)
        self.assertEqual(
            final_fields["analysis_primary_classifications"],
            manifest["decision"]["primary_classifications"],
        )
        self.assertEqual(
            final_fields["analysis_accepted_cells_fingerprint_sha256"],
            manifest["accepted_cells_fingerprint_sha256"],
        )
        with (output / "paired_contrasts.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            contrasts = list(csv.DictReader(handle))
        selected = [
            row
            for row in contrasts
            if row["contrast"] == "selective_none_vs_muon"
            and row["metric"] == "final_val_loss"
        ]
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(float(selected[0]["mean"]), -0.007, places=12)
        self.assertEqual(
            selected[0]["classification"], "candidate_better_beyond_margin"
        )

    def test_target_crossing_is_linearly_interpolated(self) -> None:
        rows = [
            {"step": 0, "val_loss": 3.4},
            {"step": 50, "val_loss": 3.2},
        ]
        steps, tokens, observed = A.target_crossing(
            rows, 3.3, A.EXPECTED_TOKENS_PER_UPDATE
        )
        self.assertAlmostEqual(steps or -1, 25.0)
        self.assertAlmostEqual(tokens or -1, 25.0 * A.EXPECTED_TOKENS_PER_UPDATE)
        self.assertEqual(observed, 50)

    def test_init_mismatch_is_rejected(self) -> None:
        manifest_path = (
            self.run
            / "formal"
            / "seed2025"
            / "selective_diag"
            / "attempt_001"
            / "scientific_manifest.json"
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["init_sha256"] = "f" * 64
        write_json(manifest_path, payload)
        accepted_path = manifest_path.parent.parent / "accepted.json"
        accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
        accepted["scientific_manifest_sha256"] = sha256(manifest_path)
        write_json(accepted_path, accepted)
        with self.assertRaisesRegex(RuntimeError, "pairing/source/data audit"):
            A.analyze(self.run, self.contract, self.run / "analysis")

    def test_command_environment_mismatch_is_rejected(self) -> None:
        command_path = (
            self.run
            / "formal"
            / "seed2024"
            / "selective_diag"
            / "attempt_001"
            / "command.json"
        )
        payload = json.loads(command_path.read_text(encoding="utf-8"))
        payload["environment"]["RECORD17_CPROJ_K_MODE"] = "block4"
        write_json(command_path, payload)
        # Re-seal the deliberately incorrect command so the failure is routing,
        # not merely an artifact-hash mismatch.
        manifest_path = command_path.parent / "scientific_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifact_hashes"]["command.json"] = sha256(command_path)
        write_json(manifest_path, manifest)
        accepted_path = manifest_path.parent.parent / "accepted.json"
        accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
        accepted["scientific_manifest_sha256"] = sha256(manifest_path)
        write_json(accepted_path, accepted)
        with self.assertRaisesRegex(RuntimeError, "method environment mismatch"):
            A.analyze(self.run, self.contract, self.run / "analysis")

    def test_t_summary_uses_df2_student_t(self) -> None:
        summary = A.t_summary([-0.004, -0.003, -0.002])
        self.assertEqual(summary["n_seeds"], 3)
        expected_sd = 0.001
        self.assertAlmostEqual(summary["sample_sd"], expected_sd)
        self.assertAlmostEqual(
            summary["ci95_high_t_df2"] - summary["mean"],
            A.T_CRITICAL_95_DF2 * expected_sd / math.sqrt(3),
        )

    def test_any_practically_unresolved_primary_ci_triggers_seed_gate(self) -> None:
        contrast_names = (
            "selective_none_vs_muon",
            "selective_none_vs_original",
            "selective_diag_vs_muon",
            "selective_diag_vs_original",
        )
        rows = [
            {
                "family": "primary",
                "metric": "final_val_loss",
                "contrast": contrast,
                "classification": "candidate_better_beyond_margin",
            }
            for contrast in contrast_names
        ]
        # A CI can be directionally negative while crossing only the -margin
        # boundary.  It is still practically unresolved and must enter the
        # pre-registered seed-extension gate.
        rows[0]["classification"] = (
            "direction_candidate_better_but_practically_unresolved"
        )
        decision = A.build_decision(rows, A.PRACTICAL_MARGIN)
        self.assertTrue(decision["statistical_seed_append_gate_triggered"])
        self.assertEqual(
            decision["ambiguous_primary_contrasts"],
            ["selective_none_vs_muon"],
        )

    def test_checked_in_contract_matches_frozen_analysis(self) -> None:
        checked_in = json.loads(
            (HERE / "record17_contract.json").read_text(encoding="utf-8")
        )
        frozen = A.validate_contract(checked_in)
        self.assertTrue(all(frozen["checks"].values()))

    def test_corrupt_committed_analysis_selects_fresh_retry(self) -> None:
        output = self.run / "analysis"
        A.analyze(self.run, self.contract, output)
        committed = output / "record17_analysis_manifest.json"
        committed_before = committed.read_bytes()
        report = output / "RECORD17_ANALYSIS_REPORT.md"
        report.write_text(
            report.read_text(encoding="utf-8") + "\ncorruption\n",
            encoding="utf-8",
        )
        reusable, checks = S.validate_reusable_analysis(committed, self.run)
        self.assertIsNone(reusable)
        self.assertFalse(checks["artifact_hashes"])
        selected_manifest, selected_output, _audits = S.select_analysis_output(
            self.run
        )
        self.assertIsNone(selected_manifest)
        self.assertEqual(selected_output, self.run / "analysis_retry_001")
        self.assertEqual(committed.read_bytes(), committed_before)

    def test_changed_accepted_pointer_invalidates_analysis(self) -> None:
        output = self.run / "analysis"
        A.analyze(self.run, self.contract, output)
        pointer = (
            self.run
            / "formal"
            / "seed2024"
            / "muon"
            / "accepted.json"
        )
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        payload["accepted_at"] = "changed-after-analysis"
        write_json(pointer, payload)
        reusable, checks = S.validate_reusable_analysis(
            output / "record17_analysis_manifest.json", self.run
        )
        self.assertIsNone(reusable)
        self.assertFalse(checks["accepted_cell_fingerprints"])


if __name__ == "__main__":
    unittest.main()
