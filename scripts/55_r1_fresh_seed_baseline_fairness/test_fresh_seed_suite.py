#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ex55_suite", HERE / "run_fresh_seed_suite.py")
assert SPEC and SPEC.loader
SUITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUITE)
EX49_SPEC = importlib.util.spec_from_file_location(
    "ex49_suite_for_ex55_test",
    HERE.parent / "49_r1_malt_strong_baseline/run_r1_malt_suite.py",
)
assert EX49_SPEC and EX49_SPEC.loader
EX49_SUITE = importlib.util.module_from_spec(EX49_SPEC)
EX49_SPEC.loader.exec_module(EX49_SUITE)
MATERIALIZER_SPEC = importlib.util.spec_from_file_location(
    "ex55_accepted_input_materializer",
    HERE / "materialize_accepted_inputs.py",
)
assert MATERIALIZER_SPEC and MATERIALIZER_SPEC.loader
MATERIALIZER = importlib.util.module_from_spec(MATERIALIZER_SPEC)
MATERIALIZER_SPEC.loader.exec_module(MATERIALIZER)


class FreshSeedSuiteTests(unittest.TestCase):
    def test_contract_covers_exactly_ten_frozen_winners(self) -> None:
        contract = json.loads((HERE / "ex55_contract.json").read_text(encoding="utf-8"))
        methods = contract["methods"]
        self.assertEqual(len(methods), 10)
        self.assertEqual({row["method"] for row in methods}, set(SUITE.SELECTED_CELLS))
        self.assertEqual(contract["protocol"]["engineering_seed"], 5501)
        self.assertEqual(contract["protocol"]["fresh_formal_seed"], 2027)
        self.assertEqual(contract["protocol"]["repaired_panel_seeds"], [2024, 2025, 2027])
        self.assertEqual(contract["worker_method_labels"], {"moonlight": "moonlight_muon"})

    def test_packaged_accepted_inputs_restore_byte_exact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "materialized"
            manifest = MATERIALIZER.materialize(HERE / "accepted_inputs_encoded", output)
            self.assertTrue(manifest["passed"])
            self.assertEqual(len(manifest["files"]), 4)
            for name, spec in MATERIALIZER.PAYLOADS.items():
                payload = (output / name).read_bytes()
                self.assertEqual(len(payload), spec["bytes"])
                self.assertEqual(SUITE.sha256_file(output / name), spec["sha256"])

    def test_manifest_match_accepts_local_valid_and_rejects_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            manifest = root / "formal_manifest.json"
            payload = {
                "status": "completed_valid_local_wandb_incomplete", "seed": 2027, "failures": [],
                "summaries": [{
                    "method": "mousse", "cell_id": "mousse_lr100", "controlled_seed": 2027,
                    "total_steps": 6200, "evidence_valid": True,
                    "checkpoint_path": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
                }],
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(SUITE.manifest_matches(manifest, expected_methods=("mousse",), seed=2027, formal=True))
            checkpoint.unlink()
            self.assertFalse(SUITE.manifest_matches(manifest, expected_methods=("mousse",), seed=2027, formal=True))

    def test_accepted_batch_ignores_child_run_manifest_in_real_two_level_layout(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            batch = root / "20260817_formal_seed2027"
            child = batch / "ex55_mousse_lr100"
            child.mkdir(parents=True)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            summary = {
                "method": "mousse", "cell_id": "mousse_lr100", "controlled_seed": 2027,
                "total_steps": 6200, "evidence_valid": True,
                "checkpoint_path": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
            }
            child_manifest = child / "run_manifest.json"
            child_manifest.write_text(json.dumps({
                "status": "completed_valid", "seed": 2027, "summaries": [summary], "failures": [],
            }), encoding="utf-8")
            self.assertIsNone(SUITE.accepted_batch(
                root, group="mousse", stage="formal", expected_methods=("mousse",),
                seed=2027, formal=True,
            ))
            aggregate = batch / "formal_manifest.json"
            aggregate.write_text(json.dumps({
                "status": "completed_valid", "seed": 2027, "summaries": [summary], "failures": [],
            }), encoding="utf-8")
            self.assertEqual(SUITE.accepted_batch(
                root, group="mousse", stage="formal", expected_methods=("mousse",),
                seed=2027, formal=True,
            ), aggregate)

    def test_full_hash_inventory_and_self_contained_malt_lineage_jointly(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            official = root / "official"
            data = official / "data/fineweb10B"
            data.mkdir(parents=True)
            for index in range(1, 51):
                (data / f"fineweb_train_{index:06d}.bin").write_bytes(f"train-{index}".encode())
            (data / "fineweb_val_000000.bin").write_bytes(b"validation")
            args = SimpleNamespace(run_dir=root / "run", official_repo=official)
            args.run_dir.mkdir()
            inventory = SUITE.frozen_data_inventory(args, full_verify=True)
            inventory_payload = json.loads(inventory.read_text(encoding="utf-8"))
            self.assertTrue(all(len(row["sha256"]) == 64 for row in inventory_payload["ordered_train_shards"]))
            ex49_run = root / "ex49_run"
            ex49_run.mkdir()
            ex49_inventory = EX49_SUITE.freeze_or_validate_data_inventory(
                SimpleNamespace(official_repo=official, run_dir=ex49_run)
            )
            self.assertEqual(SUITE.sha256_file(inventory), SUITE.sha256_file(ex49_inventory))
            source_audit = {"code": "frozen", "data_inventory_certificate_sha256": SUITE.sha256_file(inventory)}
            source_records = []
            source_root = root / "accepted_pilot/source_manifests"
            source_root.mkdir(parents=True)
            for index in range(12):
                source = source_root / f"source_{index:02d}.json"
                source.write_text(json.dumps({"source_audit": source_audit}) + "\n", encoding="utf-8")
                source_records.append({
                    "cell_id": f"cell_{index:02d}", "path": str(source),
                    "sha256": SUITE.sha256_file(source),
                })
            pilot = root / "accepted_pilot/pilot_manifest.json"
            pilot.write_text(json.dumps({
                "source_audit": source_audit, "source_manifests": source_records,
                "data_inventory": {"path": "/old/inventory.json", "sha256": SUITE.sha256_file(inventory)},
            }) + "\n", encoding="utf-8")
            selection = root / "run/accepted_inputs/malt_selection.json"
            selection.parent.mkdir(parents=True)
            selection.write_text(json.dumps({
                "status": "selected", "pilot_manifest": str(pilot),
                "pilot_manifest_sha256": SUITE.sha256_file(pilot),
            }) + "\n", encoding="utf-8")
            with mock.patch.object(SUITE, "prepare_accepted_inputs", return_value={"malt_selection": selection}):
                derived = SUITE.prepare_self_contained_malt_selection(args, inventory)
                self.assertTrue(SUITE.audit_malt_selection_lineage(derived, inventory)["passed"])
                for path in source_root.iterdir():
                    path.unlink()
                pilot.unlink()
                self.assertEqual(SUITE.prepare_self_contained_malt_selection(args, inventory), derived)
            target = data / "fineweb_train_000050.bin"
            original_stat = target.stat()
            target.write_bytes(b"xxxxxxxx")
            os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            with self.assertRaisesRegex(RuntimeError, "content changed"):
                SUITE.frozen_data_inventory(args, full_verify=True)

    def test_checkpoint_certificates_rehash_all_ten_units(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            units = []
            for index, method in enumerate(SUITE.SELECTED_CELLS):
                checkpoint = root / f"{method}.pt"
                checkpoint.write_bytes(f"checkpoint-{index}".encode())
                manifest = root / f"{method}_formal_manifest.json"
                manifest.write_text(json.dumps({"summaries": [{
                    "method": SUITE.WORKER_METHOD_LABELS[method],
                    "cell_id": SUITE.SELECTED_CELLS[method],
                    "checkpoint_path": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
                }]}) + "\n", encoding="utf-8")
                units.append({
                    "method": method, "manifest": str(manifest),
                    "manifest_sha256": SUITE.sha256_file(manifest),
                    "checkpoint": SUITE.make_checkpoint_certificate(manifest, method),
                })
            certificate = root / "formal_units_manifest.json"
            certificate.write_text(json.dumps({"passed": True, "units": units}) + "\n", encoding="utf-8")
            self.assertEqual(len(SUITE.verify_checkpoint_certificates(certificate)), 10)
            first = Path(units[0]["checkpoint"]["path"])
            first.write_bytes(b"x" * first.stat().st_size)
            with self.assertRaisesRegex(RuntimeError, "bytes/hash"):
                SUITE.verify_checkpoint_certificates(certificate)

    def test_formal_smoke_pairing_covers_all_ten_frozen_winners(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifests = {}
            for group, methods in SUITE.GROUPS.items():
                manifest = root / f"{group}.json"
                initial_loss = 10.9984 if group == "core" else 10.9937
                summaries = [{
                    "method": SUITE.WORKER_METHOD_LABELS[method],
                    "cell_id": SUITE.SELECTED_CELLS[method] if SUITE.SELECTED_CELLS[method] != method else "",
                    "controlled_seed": 2027,
                    "init_sha256": "a" * 64,
                    "val_loss_step_0": initial_loss,
                    "val_tokens": SUITE.SMOKE_VALIDATION_TOKENS[group],
                    "final_val_loss": 3.0,
                } for method in methods]
                if group == "core":
                    # The accepted core runner omits step-0 loss from its
                    # aggregate summary; EX55 must recover it from the
                    # accepted child metrics without rerunning the smoke.
                    for method, summary in zip(methods, summaries):
                        summary.pop("val_loss_step_0")
                        summary["evidence_profile"] = "exact_shape_numerical_smoke"
                        summary["run_name"] = f"run_{method}"
                        run_dir = root / f"run_{method}"
                        run_dir.mkdir()
                        # Match the accepted core runner's actual artifact
                        # filenames, not the newer family-runner convention.
                        (run_dir / "r1_summary.json").write_text(
                            json.dumps(summary) + "\n", encoding="utf-8"
                        )
                        (run_dir / "run_manifest.json").write_text(
                            json.dumps({"status": "completed_valid", "summary": summary}) + "\n",
                            encoding="utf-8",
                        )
                        (run_dir / "r1_metrics.csv").write_text(
                            "event,step,loss\nvalidation,0,10.9984\nvalidation,34,3.0\n",
                            encoding="utf-8",
                        )
                manifest.write_text(json.dumps({"summaries": summaries}) + "\n", encoding="utf-8")
                manifests[group] = manifest
            receipt = SUITE.certify_paired_initialization(manifests, 2027)
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["formal_units"], 10)
            self.assertTrue(receipt["parameter_initialization_exact_across_all_methods"])
            self.assertTrue(receipt["initial_validation"]["within_stratum_exact"])
            self.assertFalse(
                receipt["initial_validation"]["cross_method_exact_comparison_eligible"]
            )
            self.assertEqual(
                receipt["initial_validation"]["loss_by_validation_tokens"],
                {"65536": 10.9984, "10485760": 10.9937},
            )
            core_records = [
                row for row in receipt["records"] if row["method"] in SUITE.GROUPS["core"]
            ]
            self.assertEqual(
                {row["source"] for row in core_records},
                {"accepted_child_metrics_step_0"},
            )
            payload = json.loads(manifests["mousse"].read_text(encoding="utf-8"))
            payload["summaries"][0]["init_sha256"] = "b" * 64
            manifests["mousse"].write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "paired formal-smoke initialization"):
                SUITE.certify_paired_initialization(manifests, 2027)

    def test_formal_groups_total_ten_units(self) -> None:
        flattened = [method for methods in SUITE.GROUPS.values() for method in methods]
        self.assertEqual(len(flattened), 10)
        self.assertEqual(len(set(flattened)), 10)

    def test_controller_amendment_extends_existing_receipt_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw)
            snapshot_dir = (
                run_dir / "source_snapshot/scripts/55_r1_fresh_seed_baseline_fairness"
            )
            snapshot_dir.mkdir(parents=True)
            (snapshot_dir / "run_fresh_seed_suite.py").write_text(
                "# frozen old controller\n", encoding="utf-8"
            )
            snapshot_manifest = run_dir / "source_snapshot/source_snapshot_manifest.json"
            snapshot_manifest.write_text("{}\n", encoding="utf-8")
            predecessor = run_dir / "controller_amendment_pairing_schema_v1.json"
            predecessor.write_text(
                json.dumps({"schema_version": "ex55_controller_amendment_v1"}) + "\n",
                encoding="utf-8",
            )
            predecessor_before = predecessor.read_bytes()
            receipt = SUITE.ensure_controller_amendment(SimpleNamespace(run_dir=run_dir))
            self.assertIsNotNone(receipt)
            self.assertEqual(receipt["schema_version"], "ex55_controller_amendment_v2")
            self.assertEqual(predecessor.read_bytes(), predecessor_before)
            self.assertTrue(Path(receipt["receipt_path"]).is_file())
            self.assertEqual(
                SUITE.sha256_file(Path(receipt["receipt_path"])),
                SUITE.sha256_file(
                    next(
                        path for path in run_dir.glob("controller_amendment_pairing_schema_*.json")
                        if path.name != predecessor.name
                    )
                ),
            )
            self.assertEqual(len(receipt["predecessor_amendments"]), 1)
            self.assertEqual(
                receipt["predecessor_amendments"][0]["sha256"],
                SUITE.sha256_file(predecessor),
            )
            # Repeating the same resume reuses the content-addressed amendment.
            self.assertEqual(
                SUITE.ensure_controller_amendment(SimpleNamespace(run_dir=run_dir)), receipt
            )

    def test_resume_prefers_hash_valid_run_local_accepted_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            accepted = root / "run/accepted_inputs"
            accepted.mkdir(parents=True)
            files = {
                "historical_panel_sha256": accepted / "historical_panel.csv",
                "extended_selection_sha256": accepted / "extended_selection.csv",
                "mousse_selection_sha256": accepted / "mousse_selection.json",
                "malt_selection_sha256": accepted / "malt_selection.json",
            }
            for index, path in enumerate(files.values()):
                path.write_text(f"accepted-{index}\n", encoding="utf-8")
            contract = root / "contract.json"
            contract.write_text(json.dumps({
                "accepted_inputs": {key: SUITE.sha256_file(path) for key, path in files.items()}
            }) + "\n", encoding="utf-8")
            args = SimpleNamespace(
                run_dir=root / "run",
                historical_panel=root / "deleted/historical.csv",
                extended_selection=root / "deleted/extended.csv",
                mousse_selection=root / "deleted/mousse.json",
                malt_selection=root / "deleted/malt.json",
            )
            with mock.patch.object(SUITE, "frozen_paths", return_value={"contract": contract}):
                result = SUITE.prepare_accepted_inputs(args)
            self.assertEqual(set(result.values()), set(files.values()))

    def test_command_plan_dry_run_freezes_winners_and_seeds(self) -> None:
        args = SimpleNamespace(
            official_repo=Path("/official-r0"), training_python="/venv/train-python",
            wandb_mode="online", wandb_project="ex55-test", wandb_entity=None,
            run_dir=Path("/results/ex55"), repo=Path("/repo"),
        )
        paths = {
            "core": Path("/snapshot/core.py"), "extended": Path("/snapshot/extended.py"),
            "mousse": Path("/snapshot/mousse.py"), "malt": Path("/snapshot/malt.py"),
        }
        inputs = {
            "mousse_selection": Path("/inputs/mousse.json"),
            "malt_selection": Path("/inputs/malt.json"),
        }
        with (
            mock.patch.object(SUITE, "frozen_paths", return_value=paths),
            mock.patch.object(SUITE, "prepare_accepted_inputs", return_value=inputs),
            mock.patch.object(SUITE, "frozen_data_inventory", return_value=Path("/inputs/data.json")),
            mock.patch.object(SUITE, "prepare_self_contained_malt_selection", return_value=Path("/inputs/malt_self_contained.json")),
            mock.patch.object(SUITE, "resumable_batch", return_value=None),
        ):
            extended = SUITE.group_command(
                args, "extended", "formal", Path("/results/extended"), seed=2027,
                smoke_manifest=Path("/smoke/extended.json"),
            )
            malt = SUITE.group_command(
                args, "malt", "formal", Path("/results/malt"), seed=2027,
                smoke_manifest=Path("/smoke/malt.json"),
            )
        self.assertIn("--formal", extended)
        self.assertEqual(extended[extended.index("--seed") + 1], "2027")
        self.assertEqual(
            extended[extended.index("--methods") + 1:extended.index("--methods") + 4],
            ["adamw", "normuon", "moonlight_muon"],
        )
        self.assertIn("--selected-method", malt)
        self.assertEqual(malt[malt.index("--selected-method") + 1], "malt")
        self.assertTrue(any(value.replace("\\", "/").endswith("/inputs/malt_self_contained.json") for value in malt))

    def test_extended_engineering_smoke_uses_frozen_formal_cells(self) -> None:
        args = SimpleNamespace(
            official_repo=Path("/official-r0"), training_python="/venv/train-python",
            wandb_mode="online", wandb_project="ex55-test", wandb_entity=None,
            run_dir=Path("/results/ex55"), repo=Path("/repo"),
        )
        with (
            mock.patch.object(SUITE, "frozen_paths", return_value={"extended": Path("/snapshot/extended.py")}),
            mock.patch.object(SUITE, "prepare_accepted_inputs", return_value={}),
            mock.patch.object(SUITE, "resumable_batch", return_value=None),
        ):
            command = SUITE.group_command(
                args, "extended", "pilot", Path("/results/extended"), seed=5501,
            )
        self.assertIn("--formal-smoke", command)
        self.assertNotIn("--numerical-smoke", command)
        self.assertEqual(SUITE.aggregate_manifest_name("extended", "pilot"), "formal_smoke_manifest.json")
        self.assertIn("moonlight_muon", command)
        self.assertNotIn("moonlight", command)

    def test_wrong_cell_with_right_family_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            manifest = root / "formal_manifest.json"
            manifest.write_text(json.dumps({
                "status": "completed_valid", "seed": 2027, "failures": [],
                "summaries": [{
                    "method": "malt", "cell_id": "malt_lr0160", "controlled_seed": 2027,
                    "total_steps": 6200, "evidence_valid": True,
                    "checkpoint_path": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
                }],
            }), encoding="utf-8")
            self.assertFalse(SUITE.manifest_matches(
                manifest, expected_methods=("malt",), seed=2027, formal=True,
            ))

    def test_extended_selection_is_exactly_the_three_frozen_winners(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "selection.csv"
            path.write_text(
                "method,cell,seed,formal_seed2026_decision,quality_usable\n"
                "adamw,adamw_low,2026,advance,True\n"
                "normuon,normuon_r1scale,2026,advance,True\n"
                "moonlight,moonlight_r1scale,2026,advance,True\n",
                encoding="utf-8",
            )
            self.assertTrue(SUITE.audit_extended_selection(path)["passed"])
            path.write_text(path.read_text(encoding="utf-8").replace("adamw_low", "adamw_high"), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "extended-baseline selection"):
                SUITE.audit_extended_selection(path)

    def test_mousse_and_malt_selection_content_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            mousse = root / "mousse.json"
            mousse.write_text(json.dumps({
                "status": "selected", "protocol": "mousse_r1_pilot_selection_v1",
                "seed": 2026, "pilot_steps": 1000, "selected_cell_id": "mousse_lr100",
                "selected_matrix_lr": 0.015,
            }), encoding="utf-8")
            self.assertTrue(SUITE.audit_mousse_selection(mousse)["passed"])
            malt = root / "malt.json"
            malt.write_text(json.dumps({
                "status": "selected", "protocol": "malt_r1_focused_grid_selection_v4",
                "seed": 2026, "pilot_steps": 1000, "formal_allowed": True,
                "required_formal_methods": ["malt", "malter_eq17"],
                "selected_methods": {
                    "malt": {"cell_id": "malt_lr0125"},
                    "malter_eq17": {"cell_id": "malter_eq17_lr015"},
                },
            }), encoding="utf-8")
            self.assertTrue(SUITE.audit_malt_winner_selection(malt)["passed"])
            payload = json.loads(malt.read_text(encoding="utf-8"))
            payload["selected_methods"]["malt"]["cell_id"] = "malt_lr0160"
            malt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "MALT/MALTER selection"):
                SUITE.audit_malt_winner_selection(malt)

    def test_analysis_manifest_binding_detects_artifact_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = root / "contract.json"
            historical = root / "historical.csv"
            artifact = root / "analysis" / "leave.csv"
            artifact.parent.mkdir()
            contract.write_text("{}\n", encoding="utf-8")
            historical.write_text("x\n", encoding="utf-8")
            artifact.write_text("ok\n", encoding="utf-8")
            manifest = artifact.parent / "analysis_preformal_manifest.json"
            manifest.write_text(json.dumps({
                "passed": True, "experiment_id": SUITE.EXPERIMENT,
                "contract_sha256": SUITE.sha256_file(contract),
                "historical_panel_sha256": SUITE.sha256_file(historical),
                "artifacts": [{"path": artifact.name, "bytes": artifact.stat().st_size, "sha256": SUITE.sha256_file(artifact)}],
            }), encoding="utf-8")
            self.assertTrue(SUITE.analysis_manifest_matches(
                manifest, contract=contract, historical_panel=historical,
            ))
            artifact.write_bytes(b"xx\n")
            self.assertFalse(SUITE.analysis_manifest_matches(
                manifest, contract=contract, historical_panel=historical,
            ))

    def test_formal_analysis_binding_rehashes_external_metrics_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            analysis = root / "analysis"
            analysis.mkdir()
            contract = root / "contract.json"
            contract.write_text(json.dumps({
                "protocol": {"formal_steps": 6200, "validation_every": 100},
            }) + "\n", encoding="utf-8")
            historical = root / "historical.csv"
            historical.write_text("x\n", encoding="utf-8")
            formal_records = []
            lineage_records = []
            for index, method in enumerate(SUITE.SELECTED_CELLS):
                aggregate = root / f"{method}_formal_manifest.json"
                aggregate.write_text(json.dumps({"method": method}) + "\n", encoding="utf-8")
                child_manifest = root / f"{method}_run_manifest.json"
                child_manifest.write_text("{}\n", encoding="utf-8")
                child_summary = root / f"{method}_summary.json"
                child_summary.write_text("{}\n", encoding="utf-8")
                metrics = root / f"{method}_metrics.csv"
                metrics.write_text(
                    "event,step,loss\nvalidation,5800,3.4\nvalidation,5900,3.3\n"
                    "validation,6000,3.2\nvalidation,6100,3.1\nvalidation,6200,3.0\n",
                    encoding="utf-8",
                )
                formal_records.append({
                    "method": method, "manifest": str(aggregate),
                    "manifest_sha256": SUITE.sha256_file(aggregate),
                })
                lineage_records.append({
                    "method": method, "selected_cell": SUITE.SELECTED_CELLS[method],
                    "seed": 2027, "source": "accepted_formal_child_metrics_csv",
                    "initial_step": 0, "final_step": 6200,
                    "tail5_steps": [5800, 5900, 6000, 6100, 6200],
                    "tail5_losses": [3.4, 3.3, 3.2, 3.1, 3.0],
                    "tail5_val_loss_mean": 3.2,
                    "aggregate_manifest": SUITE.file_record(aggregate),
                    "child_manifest": SUITE.file_record(child_manifest),
                    "child_summary": SUITE.file_record(child_summary),
                    "metrics": SUITE.file_record(metrics),
                })
            formal_units = root / "formal_units_manifest.json"
            formal_units.write_text(json.dumps({
                "passed": True, "units": formal_records,
            }) + "\n", encoding="utf-8")
            lineage = analysis / "formal_metrics_tail5_lineage.json"
            lineage.write_text(json.dumps({
                "passed": True, "experiment_id": SUITE.EXPERIMENT,
                "formal_seed": 2027, "formal_steps": 6200, "validation_every": 100,
                "required_tail5_steps": [5800, 5900, 6000, 6100, 6200],
                "formal_units_sha256": SUITE.sha256_file(formal_units),
                "units": lineage_records,
            }) + "\n", encoding="utf-8")
            artifact = analysis / "panel.csv"
            artifact.write_text("ok\n", encoding="utf-8")
            manifest = analysis / "analysis_manifest.json"
            manifest.write_text(json.dumps({
                "passed": True, "experiment_id": SUITE.EXPERIMENT,
                "contract_sha256": SUITE.sha256_file(contract),
                "historical_panel_sha256": SUITE.sha256_file(historical),
                "formal_units_sha256": SUITE.sha256_file(formal_units),
                "formal_metrics_lineage": {
                    "path": lineage.name, "bytes": lineage.stat().st_size,
                    "sha256": SUITE.sha256_file(lineage),
                },
                "artifacts": [
                    {"path": artifact.name, "bytes": artifact.stat().st_size, "sha256": SUITE.sha256_file(artifact)},
                    {"path": lineage.name, "bytes": lineage.stat().st_size, "sha256": SUITE.sha256_file(lineage)},
                ],
            }) + "\n", encoding="utf-8")
            self.assertTrue(SUITE.analysis_manifest_matches(
                manifest, contract=contract, historical_panel=historical,
                formal_units=formal_units,
            ))
            first_metrics = Path(lineage_records[0]["metrics"]["path"])
            first_metrics.write_text("tampered\n", encoding="utf-8")
            self.assertFalse(SUITE.analysis_manifest_matches(
                manifest, contract=contract, historical_panel=historical,
                formal_units=formal_units,
            ))

    def test_wrapper_exposes_explicit_resume(self) -> None:
        wrapper = (HERE.parent.parent / "commands/55_r1_fresh_seed_baseline_fairness/20260817_ex55_r1_fresh_seed_baseline_fairness.sh").read_text(encoding="utf-8")
        self.assertIn("preflight|pilot|formal|verify|resume|all", wrapper)
        self.assertIn('CONTROLLER_STAGE="all"', wrapper)
        self.assertIn("EX55_GPUS:-1", wrapper)
        self.assertIn("55_r1_fresh_seed_baseline_fairness/accepted_inputs_encoded", wrapper)
        self.assertIn("materialize_accepted_inputs.py", wrapper)
        self.assertIn("initial_validation_evidence", wrapper)
        self.assertIn("FORMAL_METRICS_TAIL5_LINEAGE = True", wrapper)

    def test_execution_contract_reserves_only_gpu_one(self) -> None:
        contract = json.loads((HERE / "ex55_contract.json").read_text(encoding="utf-8"))
        policy = contract["execution_policy"]
        self.assertEqual(policy["physical_gpus"], ["1"])
        self.assertEqual(policy["maximum_concurrent_training_processes"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
