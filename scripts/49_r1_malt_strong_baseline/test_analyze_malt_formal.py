from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import analyze_malt_formal as analyzer
import analyze_malt_pilot as pilot_analyzer


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYZER = SCRIPT_DIR / "analyze_malt_formal.py"
SEEDS = (2024, 2025, 2026)
HISTORICAL_METHODS = (
    "diag",
    "none",
    "mousse",
    "muon",
    "block4",
    "moonlight",
    "normuon",
    "adamw",
)
FORMAL_METHODS = ("malt", "malter_eq17")
LABELS = {
    "malt": "MALT-R1 adaptation",
    "malter_eq17": "MALTER-Eq17-R1 adaptation",
}
CELL_IDS = {
    "malt": "malt_lr0100",
    "malter_eq17": "malter_eq17_lr015",
}
MATRIX_LRS = {"malt": 0.0100, "malter_eq17": 0.015}
INITIAL = {2024: 10.9462, 2025: 10.9869, 2026: 10.9790}
FINAL = {
    "diag": 3.261,
    "none": 3.267,
    "mousse": 3.268,
    "muon": 3.277,
    "block4": 3.263,
    "moonlight": 3.275,
    "normuon": 3.335,
    "adamw": 3.395,
    "malt": 3.260,
    "malter_eq17": 3.258,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def metric_row(method: str, seed: int, final: float) -> dict[str, object]:
    return {
        "method": method,
        "run_name": f"{method}_seed{seed}",
        "seed": seed,
        "initial_val_loss": INITIAL[seed],
        "final_val_loss": final + (seed - 2025) * 0.0001,
        "best_val_loss": final + (seed - 2025) * 0.0001,
        "tail5_val_loss_mean": final + 0.008,
        "normalized_val_auc": final + 0.35,
        "peak_memory_mib": 38000,
        "optimizer_state_mib": 700,
        "timing_eligible": False,
    }


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.ex45_summary = (
            root / "experiment45/r1_unified_eight_method_run_summary.csv"
        )
        self.ex45_manifest = root / "experiment45/analysis_manifest.json"
        historical = [
            metric_row(method, seed, FINAL[method])
            for seed in SEEDS
            for method in HISTORICAL_METHODS
        ]
        write_csv(self.ex45_summary, historical)
        write_json(
            self.ex45_manifest,
            {
                "status": "completed_valid",
                "protocol": "mousse_r1_unified_analysis_v1",
                "identity_certificate": "identity_reuse_certificate.json",
                "source_files": [{"path": "frozen", "sha256": "a" * 64}],
                "outputs": [self.ex45_summary.name],
            },
        )

        self.selection = root / "pilot/pilot_selection_verified.json"
        selections = {
            method: {
                "method": method,
                "selection_policy": (
                    "raw_endpoint_best"
                    if method == "malt"
                    else "paper_center_within_best_plus_0.002"
                ),
                "center_cell_id": (
                    None if method == "malt" else CELL_IDS[method]
                ),
                "center_tie_margin": None if method == "malt" else 0.002,
                "status": "selected",
                "formal_allowed": True,
                "formal_eligible": True,
                "boundary_rule_triggered": False,
                "boundary_side": None,
                "selected_cell_id": CELL_IDS[method],
                "selected_matrix_lr": MATRIX_LRS[method],
                "selection_reason": (
                    "raw_endpoint_best"
                    if method == "malt"
                    else "paper_center_within_best_plus_0.002"
                ),
                "ranked_cells": [
                    {
                        "rank": 1,
                        "cell_id": CELL_IDS[method],
                        "matrix_lr": MATRIX_LRS[method],
                        "final_val_loss": FINAL[method] + 0.5,
                    }
                ],
            }
            for method in FORMAL_METHODS
        }
        self.selection_payload: dict[str, Any] = {
            "status": "selected",
            "protocol": "malt_r1_focused_grid_selection_v4",
            "certificate_role": "independent_pilot_analysis_selection",
            "scientific_result": "dual_methods_selected",
            "formal_allowed": True,
            "required_formal_methods": list(FORMAL_METHODS),
            "seed": 2026,
            "pilot_steps": 1000,
            "selection_endpoint": "step-1000 validation loss",
            "grid_design": "fresh_v4_focused_malt_upper_grid_dual_method",
            "malt_selection_policy": "raw_endpoint_best",
            "malter_center_tie_margin": 0.002,
            "malter_center_preferred_if_within_margin_of_best": True,
            "pilot_manifest": "pilot_manifest.json",
            "pilot_manifest_sha256": "b" * 64,
            "selections": selections,
        }
        write_json(self.selection, self.selection_payload)
        selection_sha256 = sha256_file(self.selection)

        self.formal_summaries: dict[str, list[Path]] = {
            method: [] for method in FORMAL_METHODS
        }
        self.formal_manifests: dict[str, list[Path]] = {
            method: [] for method in FORMAL_METHODS
        }
        for method in FORMAL_METHODS:
            for seed in (2026, 2024, 2025):
                directory = root / f"{method}_seed{seed}"
                summary_path = directory / "formal_summary.csv"
                manifest_path = directory / "formal_manifest.json"
                checkpoint_path = directory / "checkpoint.pt"
                checkpoint_payload = f"checkpoint-{method}-{seed}".encode("utf-8")
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_bytes(checkpoint_payload)
                checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
                hidden_state_bytes = (
                    340_402_176 if method == "malt" else 340_402_464
                )
                auxiliary_state_bytes = 1_048_576
                total_state_bytes = hidden_state_bytes + auxiliary_state_bytes
                roles = {
                    "malt_momentum": 48,
                    "malt_row_ema": 72,
                    "malt_col_ema": 72,
                    "malt_last_alpha_min": 48,
                    "malt_last_alpha_max": 48,
                }
                if method == "malter_eq17":
                    roles["malt_nu"] = 72
                state_schema = {
                    "roles": roles,
                    "contains_activation_k_state": False,
                    "optimizer_group_steps": [6200],
                    "numerical_checks_passed": True,
                }
                row = {
                    **metric_row(method, seed, FINAL[method]),
                    "controlled_seed": seed,
                    "cell_id": CELL_IDS[method],
                    "matrix_lr": MATRIX_LRS[method],
                    "adaptation_label": LABELS[method],
                    "init_sha256": sha256_text(f"init-{seed}"),
                    "total_steps": 6200,
                    "total_tokens": 3_250_585_600,
                    "evidence_profile": "malt_r1_selected_6200step_v4",
                    "formal_evidence": True,
                    "evidence_valid": True,
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_bytes": len(checkpoint_payload),
                    "checkpoint_sha256": checkpoint_sha256,
                    "peak_memory_allocated_mib": 38100,
                    "hidden_optimizer_state_bytes": hidden_state_bytes,
                    "total_optimizer_state_bytes": total_state_bytes,
                    "auxiliary_optimizer_state_bytes": auxiliary_state_bytes,
                    "optimizer_state_bytes": total_state_bytes,
                    "model_parameter_bytes": 496_000_000,
                    "malt_momentum_bytes": 339_738_624,
                    "malt_row_ema_bytes": 331_776,
                    "malt_col_ema_bytes": 331_776,
                    "malt_nu_bytes": 0 if method == "malt" else 288,
                    "state_schema": state_schema,
                }
                row.pop("peak_memory_mib")
                row.pop("optimizer_state_mib")
                write_csv(summary_path, [row])
                embedded_selection = {
                    "path": str(self.selection),
                    "sha256": selection_sha256,
                    "validated_selected_method": method,
                    **json.loads(json.dumps(self.selection_payload)),
                }
                write_json(
                    manifest_path,
                    {
                        "status": "completed_valid",
                        "family": "49_r1_malt_strong_baseline",
                        "protocol": "malt_r1_selected_6200step_v4",
                        "batch_kind": "formal",
                        "seed": seed,
                        "adaptation_label": LABELS[method],
                        "total_steps": 6200,
                        "total_tokens": 3_250_585_600,
                        "formal_evidence": True,
                        "timing_eligible": False,
                        "cell": {
                            "cell_id": CELL_IDS[method],
                            "matrix_lr": MATRIX_LRS[method],
                            "method": method,
                            "formal_eligible": True,
                        },
                        "selection_certificate": embedded_selection,
                        "source_audit": {
                            "derived_source_sha256": "c" * 64,
                            "contract_sha256": "e" * 64,
                        },
                        "training_runtime_fingerprint": {
                            "gpu_name": "NVIDIA H100 80GB HBM3",
                            "torch": "2.8.0+cu126",
                        },
                        "exact_runtime_contract": {
                            "status": "passed",
                            "expected": dict(analyzer.EXPECTED_RUNTIME),
                            "observed": dict(analyzer.EXPECTED_RUNTIME),
                        },
                        "data_inventory": {
                            "status": "passed",
                            "sha256": "d" * 64,
                            "train_shard_count": 50,
                            "validation_shard_count": 1,
                            "selected_total_bytes": 1234,
                        },
                        "summary": dict(row),
                    },
                )
                self.formal_summaries[method].append(summary_path)
                self.formal_manifests[method].append(manifest_path)

    def command(self, output: Path) -> list[str]:
        return [
            sys.executable,
            str(ANALYZER),
            "--malt-summaries",
            *(str(path) for path in self.formal_summaries["malt"]),
            "--malt-manifests",
            *(str(path) for path in self.formal_manifests["malt"]),
            "--malter-summaries",
            *(str(path) for path in self.formal_summaries["malter_eq17"]),
            "--malter-manifests",
            *(str(path) for path in self.formal_manifests["malter_eq17"]),
            "--selection-certificate",
            str(self.selection),
            "--experiment45-summary",
            str(self.ex45_summary),
            "--experiment45-analysis-manifest",
            str(self.ex45_manifest),
            "--output-dir",
            str(output),
        ]

    def run(self, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.command(output),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    def manifest_path(self, method: str, seed: int) -> Path:
        return next(
            path
            for path in self.formal_manifests[method]
            if f"seed{seed}" in str(path)
        )

    def summary_path(self, method: str, seed: int) -> Path:
        return next(
            path
            for path in self.formal_summaries[method]
            if f"seed{seed}" in str(path)
        )

    def read_manifest(self, method: str, seed: int) -> dict[str, Any]:
        return json.loads(self.manifest_path(method, seed).read_text(encoding="utf-8"))

    def write_manifest(
        self, method: str, seed: int, payload: dict[str, Any]
    ) -> None:
        write_json(self.manifest_path(method, seed), payload)


class FormalAnalyzerTests(unittest.TestCase):
    def test_accepts_exact_fresh_v4_pilot_analyzer_certificate_shape(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            pilot_manifest = root / "pilot_manifest.json"
            write_json(pilot_manifest, {"status": "completed_valid"})
            rows: list[dict[str, object]] = []
            for cell_id, spec in pilot_analyzer.CELL_SPECS.items():
                method = str(spec["method"])
                matrix_lr = float(spec["matrix_lr"])
                if method == "malt":
                    final_loss = 3.0 + abs(matrix_lr - 0.0100)
                elif matrix_lr == 0.015:
                    final_loss = 3.0
                elif matrix_lr == 0.012:
                    final_loss = 3.001
                else:
                    final_loss = 3.01 + abs(matrix_lr - 0.015)
                rows.append(
                    {
                        "cell_id": cell_id,
                        "method": method,
                        "matrix_lr": matrix_lr,
                        "final_val_loss": final_loss,
                    }
                )
            payload = pilot_analyzer.recompute_selection(rows, pilot_manifest)
            certificate = root / "pilot_selection_verified.json"
            write_json(certificate, payload)

            accepted, observed_sha256 = analyzer.validate_selection_certificate(
                certificate
            )
            self.assertEqual(observed_sha256, sha256_file(certificate))
            self.assertEqual(
                accepted["protocol"], "malt_r1_focused_grid_selection_v4"
            )
            self.assertEqual(
                accepted["selections"]["malt"]["selection_policy"],
                "raw_endpoint_best",
            )
            self.assertEqual(
                accepted["selections"]["malter_eq17"]["selection_policy"],
                "paper_center_within_best_plus_0.002",
            )

    def test_builds_sealed_ten_method_panel_and_eleven_contrasts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = Fixture(root)
            output = root / "analysis"
            result = fixture.run(output)
            self.assertEqual(result.returncode, 0, result.stdout)

            run_summary = output / "r1_unified_ten_method_run_summary.csv"
            run_rows = read_csv(run_summary)
            self.assertEqual(len(run_rows), 30)
            self.assertEqual(
                {(row["method"], int(row["seed"])) for row in run_rows},
                {
                    (method, seed)
                    for method in (*HISTORICAL_METHODS, *FORMAL_METHODS)
                    for seed in SEEDS
                },
            )
            for method in FORMAL_METHODS:
                rows = [row for row in run_rows if row["method"] == method]
                self.assertEqual(
                    {row["adaptation_label"] for row in rows}, {LABELS[method]}
                )
                self.assertTrue(all(len(row["checkpoint_sha256"]) == 64 for row in rows))

            aggregate_rows = read_csv(
                output / "r1_unified_ten_method_aggregate.csv"
            )
            self.assertEqual(len(aggregate_rows), 10)

            contrasts = read_csv(output / "r1_malt_family_paired_aggregate.csv")
            deltas = read_csv(output / "r1_malt_family_paired_seed_deltas.csv")
            self.assertEqual(len(contrasts), 11)
            self.assertEqual(len(deltas), 33)
            core = {"muon", "block4", "none", "diag", "mousse"}
            for method in FORMAL_METHODS:
                method_rows = [row for row in contrasts if row["left"] == method]
                if method == "malt":
                    baseline_rows = [row for row in method_rows if row["right"] in core]
                else:
                    baseline_rows = method_rows
                self.assertEqual({row["right"] for row in baseline_rows}, core)
            family_rows = [
                row for row in contrasts if row["contrast"] == "malt_minus_malter_eq17"
            ]
            self.assertEqual(len(family_rows), 1)
            self.assertEqual(family_rows[0]["role"], "family_internal")
            self.assertTrue(all(row["n_seeds"] == "3" for row in contrasts))
            self.assertTrue(
                all(row["ci_interpretation"] == "descriptive_only_n3" for row in contrasts)
            )
            self.assertTrue(all(row["practical_margin"] == "0.002" for row in contrasts))

            manifest_path = output / "analysis_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed_valid")
            self.assertEqual(manifest["protocol"], "malt_r1_ten_method_analysis_v4")
            self.assertEqual(
                manifest["formal_protocol"], "malt_r1_selected_6200step_v4"
            )
            self.assertEqual(manifest["n_methods"], 10)
            self.assertEqual(manifest["n_run_rows"], 30)
            self.assertEqual(manifest["n_formal_methods"], 2)
            self.assertEqual(manifest["n_formal_runs"], 6)
            self.assertEqual(manifest["n_paired_contrasts"], 11)
            self.assertEqual(manifest["n_paired_seed_deltas"], 33)
            self.assertEqual(len(manifest["input_files"]), 15)
            self.assertEqual(
                manifest["selection_certificate_sha256"], sha256_file(fixture.selection)
            )
            self.assertEqual(
                set(manifest["formal_manifest_statuses"]), set(FORMAL_METHODS)
            )
            for record in manifest["input_files"]:
                self.assertEqual(record["sha256"], sha256_file(Path(record["path"])))
            for name, expected in manifest["output_sha256"].items():
                self.assertEqual(expected, sha256_file(output / name))
            seal = (output / "analysis_manifest.sha256").read_text(encoding="ascii")
            self.assertEqual(
                seal,
                f"{sha256_file(manifest_path)}  analysis_manifest.json\n",
            )
            identity = json.loads(
                (output / "identity_reuse_certificate.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(identity["method_labels"], LABELS)
            self.assertEqual(
                identity["protocol"], "malt_r1_experiment45_identity_reuse_v4"
            )
            self.assertTrue(
                identity["checks"][
                    "dual_method_selection_certificate_sha256_verified"
                ]
            )
            report = (output / "EXPERIMENT_49_ANALYSIS.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("MALTER-Eq17-R1 adaptation", report)
            self.assertIn("Ten-method endpoint", report)

    def test_rejects_nonformal_budget(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = Fixture(root)
            payload = fixture.read_manifest("malter_eq17", 2025)
            payload["total_steps"] = 6199
            fixture.write_manifest("malter_eq17", 2025, payload)
            output = root / "analysis"
            result = fixture.run(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("formal step budget mismatch", result.stdout)
            self.assertFalse(output.exists())

    def test_rejects_wrong_adaptation_label_for_either_method(self) -> None:
        for method in FORMAL_METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                fixture = Fixture(root)
                payload = fixture.read_manifest(method, 2024)
                payload["adaptation_label"] = "official reproduction"
                fixture.write_manifest(method, 2024, payload)
                output = root / "analysis"
                result = fixture.run(output)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("method-label mismatch", result.stdout)
                self.assertFalse(output.exists())

    def test_rejects_selection_cell_or_seal_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = Fixture(root)
            payload = fixture.read_manifest("malter_eq17", 2025)
            payload["cell"]["matrix_lr"] = 0.018
            fixture.write_manifest("malter_eq17", 2025, payload)
            output = root / "analysis"
            result = fixture.run(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("formal cell/LR does not match selection", result.stdout)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = Fixture(root)
            payload = json.loads(fixture.selection.read_text(encoding="utf-8"))
            payload["grid_design"] = "tampered-after-formal"
            write_json(fixture.selection, payload)
            output = root / "analysis"
            result = fixture.run(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("selection certificate/seal mismatch", result.stdout)
            self.assertFalse(output.exists())

    def test_rejects_nonindependent_selection_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = Fixture(root)
            payload = json.loads(fixture.selection.read_text(encoding="utf-8"))
            payload["certificate_role"] = "runner_preselection_crosscheck"
            write_json(fixture.selection, payload)
            output = root / "analysis"
            result = fixture.run(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "certificate was not issued by the independent pilot analyzer",
                result.stdout,
            )
            self.assertFalse(output.exists())

    def test_rejects_stale_v3_selection_and_formal_protocols(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = Fixture(root)
            payload = json.loads(fixture.selection.read_text(encoding="utf-8"))
            payload["protocol"] = "malt_r1_extended_grid_selection_v3"
            write_json(fixture.selection, payload)
            output = root / "analysis"
            result = fixture.run(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("protocol mismatch", result.stdout)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = Fixture(root)
            payload = fixture.read_manifest("malt", 2024)
            payload["protocol"] = "malt_r1_selected_6200step_v3"
            fixture.write_manifest("malt", 2024, payload)
            output = root / "analysis"
            result = fixture.run(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("formal protocol mismatch", result.stdout)
            self.assertFalse(output.exists())

    def test_rejects_malter_method_or_checkpoint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = Fixture(root)
            path = fixture.summary_path("malter_eq17", 2026)
            rows = read_csv(path)
            rows[0]["method"] = "malt"
            write_csv(path, rows)
            output = root / "analysis"
            result = fixture.run(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("method must be 'malter_eq17'", result.stdout)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = Fixture(root)
            path = fixture.summary_path("malter_eq17", 2026)
            rows = read_csv(path)
            rows[0]["checkpoint_sha256"] = "f" * 64
            write_csv(path, rows)
            output = root / "analysis"
            result = fixture.run(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("checkpoint seal mismatch", result.stdout)
            self.assertFalse(output.exists())

    def test_rejects_cross_method_source_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = Fixture(root)
            for seed in SEEDS:
                payload = fixture.read_manifest("malter_eq17", seed)
                payload["source_audit"] = {
                    "derived_source_sha256": "9" * 64,
                    "contract_sha256": "e" * 64,
                }
                fixture.write_manifest("malter_eq17", seed, payload)
            output = root / "analysis"
            result = fixture.run(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("do not share one source fingerprint", result.stdout)
            self.assertFalse(output.exists())

    def test_rejects_memory_drift_and_actual_checkpoint_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = Fixture(root)
            path = fixture.summary_path("malt", 2024)
            rows = read_csv(path)
            rows[0]["peak_memory_allocated_mib"] = "38101"
            write_csv(path, rows)
            output = root / "analysis"
            result = fixture.run(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "CSV/manifest summary mismatch for peak_memory_allocated_mib",
                result.stdout,
            )
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = Fixture(root)
            manifest = fixture.read_manifest("malter_eq17", 2025)
            checkpoint = Path(manifest["summary"]["checkpoint_path"])
            checkpoint.write_bytes(b"tampered-checkpoint")
            output = root / "analysis"
            result = fixture.run(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("actual checkpoint certificate failed", result.stdout)
            self.assertFalse(output.exists())

    def test_rejects_unaccepted_experiment45_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = Fixture(root)
            payload = json.loads(fixture.ex45_manifest.read_text(encoding="utf-8"))
            payload["status"] = "failed"
            write_json(fixture.ex45_manifest, payload)
            output = root / "analysis"
            result = fixture.run(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Experiment-45 analysis manifest is not accepted", result.stdout)
            self.assertFalse(output.exists())

    def test_rejects_incomplete_malter_seed_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = Fixture(root)
            payload = fixture.read_manifest("malter_eq17", 2026)
            payload["seed"] = 2027
            fixture.write_manifest("malter_eq17", 2026, payload)
            output = root / "analysis"
            result = fixture.run(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "unexpected MALTER-Eq17-R1 adaptation formal manifest seed: 2027",
                result.stdout,
            )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
