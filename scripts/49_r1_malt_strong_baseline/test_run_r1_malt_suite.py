from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_r1_malt as runner
import run_r1_malt_suite as suite
from test_analyze_malt_formal import Fixture as FormalFixture
from test_analyze_malt_formal import write_csv as write_fixture_csv


class MALTSuiteTests(unittest.TestCase):
    def test_frozen_data_inventory_selects_first_fifty_and_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            official = root / "official"
            data = official / "data" / "fineweb10B"
            data.mkdir(parents=True)
            for index in range(1, 53):
                (data / f"fineweb_train_{index:06d}.bin").write_bytes(
                    f"train-{index}".encode()
                )
            (data / "fineweb_val_000000.bin").write_bytes(b"validation")
            run_dir = root / "results" / "49" / "run"
            run_dir.mkdir(parents=True)
            args = argparse.Namespace(official_repo=official, run_dir=run_dir)
            certificate = suite.freeze_or_validate_data_inventory(args)
            payload = json.loads(certificate.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["ordered_train_shards"]), 50)
            self.assertEqual(
                payload["ordered_train_shards"][-1]["name"],
                "fineweb_train_000050.bin",
            )
            self.assertTrue(payload["extra_train_shards_are_ignored"])
            self.assertEqual(
                suite.freeze_or_validate_data_inventory(args), certificate
            )
            (data / "fineweb_train_000050.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "FineWeb content changed"):
                suite.freeze_or_validate_data_inventory(args)

    def test_split_pilot_units_are_sealed_into_one_verified_grid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw)
            for index, cell in enumerate(runner.PILOT_CELLS):
                batch = run_dir / "pilot" / cell.cell_id / f"batch_{index:02d}"
                batch.mkdir(parents=True)
                loss = 3.2 + index * 0.001
                if cell.cell_id in {"malt_lr0100", "malter_eq17_lr015"}:
                    loss = 2.9
                if cell.cell_id == runner.MALTER_CENTER_CELL_ID:
                    loss = 2.901
                summary = {
                    **runner.asdict(cell),
                    "method": cell.method,
                    "controlled_seed": 2026,
                    "total_steps": 1000,
                    "total_tokens": 1000 * runner.TOKENS_PER_STEP,
                    "evidence_valid": True,
                    "init_sha256": "same-init",
                    "wandb_status": "disabled",
                    "final_val_loss": loss,
                    "val_loss_step_1000": loss,
                }
                (batch / "pilot_plan.json").write_text("{}", encoding="utf-8")
                (batch / "pilot_manifest.json").write_text(
                    json.dumps(
                        {
                            "family": runner.FAMILY,
                            "protocol": runner.PILOT_PROTOCOL,
                            "seed": 2026,
                            "total_steps": 1000,
                            "status": "completed_valid",
                            "summaries": [summary],
                            "failures": [],
                            "wandb_complete": True,
                            "wandb_mode": "disabled",
                            "source_audit": {"source": "frozen"},
                            "training_runtime_fingerprint": {"runtime": "frozen"},
                            "exact_runtime_contract": {
                                "status": "passed",
                                "expected": dict(runner.EXPECTED_TRAINING_RUNTIME),
                                "observed": dict(runner.EXPECTED_TRAINING_RUNTIME),
                            },
                            "data_inventory": {
                                "status": "passed",
                                "sha256": "d" * 64,
                                "train_shard_count": 50,
                                "validation_shard_count": 1,
                            },
                            "initialization_audit": {"init_sha256": "same-init"},
                        }
                    ),
                    encoding="utf-8",
                )
            args = argparse.Namespace(
                run_dir=run_dir,
                repo=Path(__file__).resolve().parents[2],
                wandb_mode="disabled",
            )
            selection = suite.aggregate_pilot(args)
            payload = json.loads(selection.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "selected")
            self.assertTrue(payload["formal_allowed"])
            self.assertEqual(
                payload["selections"]["malt"]["selected_cell_id"],
                "malt_lr0100",
            )
            self.assertEqual(
                payload["selections"]["malter_eq17"]["selected_cell_id"],
                runner.MALTER_CENTER_CELL_ID,
            )
            aggregate = json.loads(
                (run_dir / "pilot/aggregate/pilot_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(aggregate["summaries"]), 12)
            self.assertEqual(
                aggregate["malt_execution_order"],
                [0.0160, 0.0125, 0.0100, 0.0090, 0.0080, 0.0064],
            )
            self.assertEqual(
                aggregate["malter_execution_order"],
                [0.007, 0.009, 0.012, 0.015, 0.018, 0.025],
            )

    def test_dual_method_formal_uses_six_unique_smoke_and_formal_units(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw)
            selection = run_dir / "pilot_selection_verified.json"
            selection.write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                run_dir=run_dir,
                repo=Path(__file__).resolve().parents[2],
                official_repo=run_dir / "official",
                training_python="/frozen/training/python",
                wandb_mode="disabled",
                wandb_entity=None,
            )
            observed_labels: list[str] = []

            def materialize(_args: argparse.Namespace, jobs: list[dict[str, object]]) -> None:
                for job in jobs:
                    label = str(job["label"])
                    observed_labels.append(label)
                    command = [str(value) for value in job["command"]]
                    stage_dir = Path(command[command.index("--results-dir") + 1])
                    method = command[command.index("--selected-method") + 1]
                    seed = int(command[command.index("--seed") + 1])
                    mode = "formal_smoke" if label.startswith("formal_smoke/") else "formal"
                    batch = stage_dir / "batch"
                    batch.mkdir(parents=True)
                    summary = {
                        "evidence_valid": True,
                        "wandb_status": "disabled",
                        "controlled_seed": seed,
                        "method": method,
                    }
                    (batch / f"{mode}_manifest.json").write_text(
                        json.dumps(
                            {
                                "status": "completed_valid",
                                "failures": [],
                                "summaries": [summary],
                                "wandb_mode": "disabled",
                                "wandb_complete": True,
                            }
                        ),
                        encoding="utf-8",
                    )
                    if mode == "formal":
                        (batch / "formal_summary.csv").write_text(
                            "method,controlled_seed\n"
                            f"{method},{seed}\n",
                            encoding="utf-8",
                        )

            with patch.object(suite, "validated_selection", return_value=selection), patch.object(
                suite, "execute_jobs", side_effect=materialize
            ):
                accepted = suite.run_formal(args)

            self.assertEqual(len(accepted), 6)
            self.assertEqual(
                {label for label in observed_labels},
                {
                    f"{stage}/{method}/seed{seed}"
                    for stage in ("formal_smoke", "formal")
                    for method in suite.FORMAL_METHODS
                    for seed in suite.FORMAL_SEEDS
                },
            )
            self.assertEqual(
                {(method, seed) for method, seed, _, _ in accepted},
                {
                    (method, seed)
                    for method in suite.FORMAL_METHODS
                    for seed in suite.FORMAL_SEEDS
                },
            )

    def test_verify_builds_and_reuses_sealed_dual_method_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run_dir = root / "run"
            run_dir.mkdir()
            fixture = FormalFixture(root / "fixture")

            for method in suite.FORMAL_METHODS:
                for seed in suite.FORMAL_SEEDS:
                    batch = run_dir / "formal" / method / f"seed{seed}" / "batch"
                    batch.mkdir(parents=True)
                    manifest = fixture.read_manifest(method, seed)
                    summary = dict(manifest["summary"])
                    summary["wandb_status"] = "disabled"
                    write_fixture_csv(batch / "formal_summary.csv", [summary])
                    manifest.update(
                        {
                            "summary": summary,
                            "summaries": [summary],
                            "failures": [],
                            "wandb_mode": "disabled",
                            "wandb_complete": True,
                        }
                    )
                    suite.write_json(batch / "formal_manifest.json", manifest)

            suite.write_json(run_dir / "suite_plan.json", {"synthetic": True})
            suite.write_json(
                run_dir / "frozen_data_inventory.json", {"synthetic": True}
            )
            args = argparse.Namespace(
                run_dir=run_dir,
                repo=Path(__file__).resolve().parents[2],
                wandb_mode="disabled",
                experiment45_summary=fixture.ex45_summary,
                experiment45_analysis_manifest=fixture.ex45_manifest,
            )

            with patch.object(
                suite, "validated_selection", return_value=fixture.selection
            ):
                handoff_path = suite.verify_and_analyze(args)

            handoff = suite.read_json(handoff_path)
            self.assertEqual(handoff["formal_methods"], list(suite.FORMAL_METHODS))
            self.assertEqual(handoff["n_formal_units"], 6)
            self.assertEqual(
                {(unit["method"], unit["seed"]) for unit in handoff["formal_units"]},
                {
                    (method, seed)
                    for method in suite.FORMAL_METHODS
                    for seed in suite.FORMAL_SEEDS
                },
            )

            analysis_manifest_path = Path(handoff["analysis_manifest"])
            self.assertTrue(suite.validate_analysis_bundle(analysis_manifest_path))
            self.assertEqual(
                handoff["analysis_manifest_sha256"],
                suite.sha256_file(analysis_manifest_path),
            )
            self.assertEqual(
                analysis_manifest_path.with_suffix(".sha256")
                .read_text(encoding="ascii")
                .split()[0],
                suite.sha256_file(analysis_manifest_path),
            )
            analysis_manifest = suite.read_json(analysis_manifest_path)
            self.assertEqual(analysis_manifest["n_methods"], 10)
            self.assertEqual(analysis_manifest["n_run_rows"], 30)
            unified_path = (
                analysis_manifest_path.parent
                / "r1_unified_ten_method_run_summary.csv"
            )
            with unified_path.open(encoding="utf-8", newline="") as handle:
                unified = list(csv.DictReader(handle))
            self.assertEqual(len(unified), 30)
            self.assertEqual(len({row["method"] for row in unified}), 10)

            first_analysis_manifest = str(analysis_manifest_path)
            first_analysis_sha256 = handoff["analysis_manifest_sha256"]
            with patch.object(
                suite, "validated_selection", return_value=fixture.selection
            ), patch.object(suite.subprocess, "run") as analyzer_run:
                second_handoff_path = suite.verify_and_analyze(args)
            analyzer_run.assert_not_called()
            second_handoff = suite.read_json(second_handoff_path)
            self.assertEqual(
                second_handoff["analysis_manifest"], first_analysis_manifest
            )
            self.assertEqual(
                second_handoff["analysis_manifest_sha256"], first_analysis_sha256
            )
            self.assertTrue(
                suite.validate_analysis_bundle(
                    Path(second_handoff["analysis_manifest"])
                )
            )


if __name__ == "__main__":
    unittest.main()
