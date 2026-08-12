from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from audit_submission_bundle import audit as audit_bundle
from audit_method_formulation import audit as audit_formulation
from audit_routing_complexity import audit as audit_complexity
from analyze_refresh_stability import analyze as analyze_refresh
from build_evidence_ledger import build_ledger
from build_final_unified_submission import external_panel
from common import ContractError, read_csv, read_json, sha256_file, write_csv, write_json
from run_local_pipeline import run


SOURCE_FIELDS = [
    "seed",
    "method",
    "run_name",
    "initial_val_loss",
    "final_val_loss",
    "best_val_loss",
    "tail5_mean",
    "normalized_auc",
    "k_state_bytes",
    "optimizer_state_bytes",
    "peak_memory_allocated_bytes",
    "timing_eligible",
]


def make_fixture(root: Path) -> Path:
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    methods = ("muon", "original", "diag", "none")
    method_delta = {"muon": 0.0, "original": -0.0025, "diag": -0.0042, "none": -0.0050}
    k_state = {"muon": 0, "original": 100 * 1024 * 1024, "diag": 1024 * 1024, "none": 0}
    optimizer_state = {
        "muon": 200 * 1024 * 1024,
        "original": 300 * 1024 * 1024,
        "diag": 201 * 1024 * 1024,
        "none": 200 * 1024 * 1024,
    }
    sources = []
    scales = (
        ("gpt124m", 124_000_000, 3_250_585_600, 3.55),
        ("gpt275m", 275_743_572, 666_501_120, 3.28),
        ("gpt455m", 454_496_336, 3_124_756_480, 2.92),
    )
    for scale_index, (scale, parameters, tokens, baseline) in enumerate(scales):
        rows = []
        for seed_index, seed in enumerate((2024, 2025, 2026)):
            for method in methods:
                final = baseline + 0.0002 * seed_index + method_delta[method]
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "run_name": f"synthetic_{scale}_{method}_{seed}",
                        "initial_val_loss": baseline + 1.0,
                        "final_val_loss": final,
                        "best_val_loss": final - 0.001,
                        "tail5_mean": final + 0.0005,
                        "normalized_auc": final + 0.05,
                        "k_state_bytes": k_state[method] * (scale_index + 1),
                        "optimizer_state_bytes": optimizer_state[method] * (scale_index + 1),
                        "peak_memory_allocated_bytes": (400 + 10 * scale_index) * 1024 * 1024,
                        "timing_eligible": False,
                    }
                )
        csv_path = inputs / f"{scale}.csv"
        manifest_path = inputs / f"{scale}_manifest.json"
        write_csv(csv_path, rows, SOURCE_FIELDS)
        write_json(manifest_path, {"status": "passed", "synthetic": True})
        sources.append(
            {
                "source_id": f"source_{scale}",
                "schema": "normalized_run_summary_v1",
                "analysis_family": "gpt_scale",
                "scale_id": scale,
                "architecture": "synthetic-gpt",
                "model_parameters": parameters,
                "train_tokens": tokens,
                "csv": csv_path.name,
                "manifest": manifest_path.name,
                "manifest_require": {"status": "passed"},
                "expected_methods": list(methods),
                "expected_seeds": [2024, 2025, 2026],
            }
        )
    catalog_path = inputs / "catalog.json"
    write_json(
        catalog_path,
        {
            "schema_version": "mdp_submission_input_catalog_v1",
            "synthetic": True,
            "sources": sources,
        },
    )

    dimension, blocks, layers, stored, dtype = 12, 3, 2, 2, 4
    elements = {
        "full": dimension * dimension,
        "block": blocks * (dimension // blocks) ** 2,
        "diag": dimension,
        "none": 0,
    }
    measured = {route: value * layers * stored * dtype for route, value in elements.items()}
    write_json(
        inputs / "complexity.json",
        {
            "schema_version": "mdp_routing_complexity_config_v1",
            "synthetic": True,
            "cases": [
                {
                    "case_id": "synthetic_case",
                    "input_dim": dimension,
                    "block_count": blocks,
                    "layer_count": layers,
                    "stored_matrices": stored,
                    "dtype_bytes": dtype,
                    "measured_state_bytes": measured,
                }
            ],
        },
    )

    rng = np.random.default_rng(1234)
    unit_count, k_dim, output_dim = 12, 6, 4
    k_before = np.empty((unit_count, k_dim, k_dim))
    k_after = np.empty_like(k_before)
    gradient_before = rng.normal(size=(unit_count, output_dim, k_dim))
    gradient_after = gradient_before + 0.01 * rng.normal(size=gradient_before.shape)
    loss_impulse = np.empty(unit_count)
    metadata_rows = []
    for index in range(unit_count):
        sample = rng.normal(size=(18, k_dim))
        base = sample.T @ sample / sample.shape[0] + 0.4 * np.eye(k_dim)
        direction = rng.normal(size=(k_dim, 1))
        perturbation = (0.004 + index * 0.0004) * (direction @ direction.T) / k_dim
        k_before[index] = base
        k_after[index] = base + perturbation
        loss_impulse[index] = float(np.linalg.norm(perturbation)) + index * 0.0001
        origin = f"origin{index // 3}"
        replica = index % 3
        stage = "early" if index < 6 else "late"
        metadata_rows.append(
            {
                "unit_id": f"{origin}_replica{replica}_{stage}",
                "origin": origin,
                "replica": replica,
                "stage": stage,
            }
        )
    np.savez(
        inputs / "snapshots.npz",
        k_before=k_before,
        k_after=k_after,
        gradient_before=gradient_before,
        gradient_after=gradient_after,
        loss_impulse=loss_impulse,
    )
    write_csv(inputs / "metadata.csv", metadata_rows, ["unit_id", "origin", "replica", "stage"])

    config_path = root / "pipeline.json"
    write_json(
        config_path,
        {
            "schema_version": "mdp_local_pipeline_config_v1",
            "synthetic": True,
            "catalog": "inputs/catalog.json",
            "complexity_config": "inputs/complexity.json",
            "practical_loss_margin": 0.002,
            "cross_scale_family": "gpt_scale",
            "equivariance": {
                "seed": 20240731,
                "input_dim": 12,
                "output_dim": 8,
                "blocks": 3,
                "tolerance": 1e-9,
                "cross_block_minimum_drift": 1e-5,
            },
            "refresh": {
                "enabled": True,
                "snapshots": "inputs/snapshots.npz",
                "metadata": "inputs/metadata.csv",
                "ridge_scale": 0.15,
                "residual_tolerance": 1e-8,
            },
        },
    )
    return config_path


def make_formal_refresh_fixture(root: Path, corrupt_runtime_inverse: bool = False):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260803)
    origins = (
        "early_muon",
        "early_newton_full",
        "late_muon",
        "late_newton_full",
    )
    count, dimension, output_dim = 12, 4, 3
    k_before = np.empty((count, dimension, dimension))
    k_after = np.empty_like(k_before)
    matched_gradient = rng.normal(size=(count, output_dim, dimension))
    gradient_before = matched_gradient + 0.01 * rng.normal(size=matched_gradient.shape)
    gradient_after = matched_gradient + 0.01 * rng.normal(size=matched_gradient.shape)
    ridge_before = np.empty(count)
    ridge_after = np.empty(count)
    inverse_before = np.empty_like(k_before)
    inverse_after = np.empty_like(k_after)
    metadata = []
    for index in range(count):
        sample = rng.normal(size=(16, dimension))
        before = sample.T @ sample / sample.shape[0] + 0.5 * np.eye(dimension)
        vector = rng.normal(size=(dimension, 1))
        after = before + 0.005 * vector @ vector.T / dimension
        k_before[index] = before
        k_after[index] = after
        ridge_before[index] = 0.2 * np.trace(before) / dimension + 1e-8
        ridge_after[index] = 0.2 * np.trace(after) / dimension + 1e-8
        inverse_before[index] = np.linalg.inv(
            before + ridge_before[index] * np.eye(dimension)
        )
        inverse_after[index] = np.linalg.inv(
            after + ridge_after[index] * np.eye(dimension)
        )
        origin = origins[index // 3]
        replica = index % 3
        early = origin.startswith("early_")
        source_method = "newton_full" if origin.endswith("newton_full") else "muon"
        metadata.append(
            {
                "unit_id": f"{origin}_replica{replica}_layer9_event32",
                "origin": origin,
                "replica": replica,
                "stage": "early" if early else "late",
                "module_id": "layers.9.down_input",
                "layer_index": 9,
                "refresh_event_step": 32,
                "source_method": source_method,
                "checkpoint_step": 1000 if early else 6200,
                "gradient_semantics": "pre_polar_optimizer_input",
            }
        )
    if corrupt_runtime_inverse:
        inverse_after[0, 0, 0] += 0.2
    snapshots = root / "snapshots.npz"
    np.savez(
        snapshots,
        k_before=k_before,
        k_after=k_after,
        gradient_before=gradient_before,
        gradient_after=gradient_after,
        matched_gradient=matched_gradient,
        inverse_before=inverse_before,
        inverse_after=inverse_after,
        ridge_before=ridge_before,
        ridge_after=ridge_after,
        loss_impulse_step48=np.linspace(-0.006, -0.004, count),
        loss_impulse_step80=np.linspace(-0.004, -0.002, count),
        loss_impulse_auc=np.linspace(-0.5, -0.3, count),
    )
    metadata_path = root / "metadata.csv"
    write_csv(metadata_path, metadata, list(metadata[0]))
    manifest = root / "snapshot_manifest.json"
    write_json(
        manifest,
        {
            "schema_version": "mdp_refresh_snapshot_manifest_v2",
            "snapshots_sha256": sha256_file(snapshots),
            "metadata_sha256": sha256_file(metadata_path),
            "runtime_contract_sha256": "a" * 64,
            "source_hashes": {"checkpoint": "b" * 64},
            "production_pipeline_replayed": True,
            "input_fingerprint_validation_passed": True,
        },
    )
    return snapshots, metadata_path, manifest


class LocalPipelineTests(unittest.TestCase):
    def test_formal_refresh_contract_and_runtime_inverse_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots, metadata, manifest = make_formal_refresh_fixture(root / "valid")
            result = analyze_refresh(
                snapshots,
                metadata,
                root / "valid_output",
                formal_contract=True,
                snapshot_manifest_path=manifest,
            )
            self.assertTrue(result["claim_eligible"])
            self.assertTrue(result["snapshot_manifest_verified"])
            self.assertTrue(result["all_runtime_inverse_checks_passed"])
            with self.assertRaises(ContractError):
                analyze_refresh(
                    snapshots,
                    metadata,
                    root / "missing_manifest_output",
                    formal_contract=True,
                )
            bad_snapshots, bad_metadata, bad_manifest = make_formal_refresh_fixture(
                root / "corrupt", corrupt_runtime_inverse=True
            )
            with self.assertRaises(ContractError):
                analyze_refresh(
                    bad_snapshots,
                    bad_metadata,
                    root / "corrupt_output",
                    formal_contract=True,
                    snapshot_manifest_path=bad_manifest,
                )
            failed = read_json(root / "corrupt_output" / "refresh_stability_manifest.json")
            self.assertFalse(failed["all_runtime_inverse_checks_passed"])

    def test_method_formulation_reference_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = audit_formulation(Path(temporary) / "formulation")
            self.assertTrue(result["all_checks_passed"])
            self.assertEqual(result["proof_status"], "numerically_checked_not_formal_proof")
            self.assertFalse(result["empirical_claim_eligible"])

    def test_required_complexity_measurement_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "complexity.json"
            write_json(
                config,
                {
                    "schema_version": "mdp_routing_complexity_config_v1",
                    "synthetic": False,
                    "require_measurement_coverage": True,
                    "cases": [
                        {
                            "case_id": "missing_diag",
                            "input_dim": 12,
                            "block_count": 3,
                            "layer_count": 2,
                            "stored_matrices": 2,
                            "dtype_bytes": 4,
                            "required_measured_routes": ["block", "diag", "none"],
                            "measured_state_bytes": {"block": 768, "none": 0},
                        }
                    ],
                },
            )
            with self.assertRaisesRegex(ContractError, "measured state sizes"):
                audit_complexity(config, root / "unused")

    def test_external_neighbor_panel_rank_and_pareto(self) -> None:
        losses = {
            "diag": 3.2611,
            "block4": 3.2622,
            "none": 3.2667,
            "mousse": 3.2680,
            "moonlight": 3.2750,
            "muon": 3.2771,
            "normuon": 3.3345,
            "adamw": 3.4004,
        }
        optimizer_state = {
            "diag": 780.8,
            "block4": 996.5,
            "none": 780.5,
            "mousse": 2887.1,
            "moonlight": 618.5,
            "muon": 618.5,
            "normuon": 618.8,
            "adamw": 942.5,
        }
        rows = []
        for method, loss in losses.items():
            for offset, seed in enumerate((2024, 2025, 2026)):
                rows.append(
                    {
                        "method": method,
                        "display_name": method,
                        "seed": str(seed),
                        "final_val_loss": str(loss + (offset - 1) * 0.0001),
                        "optimizer_state_mib": str(optimizer_state[method]),
                        "peak_memory_mib": "38000",
                        "timing_eligible": "False",
                    }
                )
        panel = external_panel(rows)
        indexed = {row["method"]: row for row in panel}
        self.assertEqual(indexed["mousse"]["quality_rank"], 4)
        self.assertFalse(indexed["mousse"]["optimizer_state_pareto_nondominated"])
        self.assertTrue(indexed["diag"]["optimizer_state_pareto_nondominated"])

    def test_experiment_43_44_45_schema_adapters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record_fields = [
                "seed", "method", "initial_val_loss", "final_val_loss", "best_val_loss",
                "tail5_mean", "normalized_auc", "k_state_bytes", "optimizer_state_bytes",
                "peak_memory_allocated_bytes", "timing_eligible",
            ]
            r1_fields = [
                "seed", "method", "run_name", "initial_val_loss", "final_val_loss",
                "best_val_loss", "tail5_val_loss_mean", "normalized_val_auc", "k_state_mib",
                "optimizer_state_mib", "peak_memory_mib",
            ]
            unified_fields = [
                "seed", "method", "run_name", "initial_val_loss", "final_val_loss",
                "best_val_loss", "tail5_val_loss_mean", "normalized_val_auc",
                "optimizer_state_mib", "peak_memory_mib", "timing_eligible",
            ]
            write_csv(
                root / "record.csv",
                [{
                    "seed": 2024, "method": "original_newton_muon", "initial_val_loss": 5.0,
                    "final_val_loss": 3.2, "best_val_loss": 3.19, "tail5_mean": 3.21,
                    "normalized_auc": 3.5, "k_state_bytes": 1234,
                    "optimizer_state_bytes": 5678, "peak_memory_allocated_bytes": 9999,
                    "timing_eligible": False,
                }],
                record_fields,
            )
            write_csv(
                root / "r1.csv",
                [{
                    "seed": 2024, "method": "block4", "run_name": "r1", "initial_val_loss": 5.0,
                    "final_val_loss": 3.5, "best_val_loss": 3.49, "tail5_val_loss_mean": 3.51,
                    "normalized_val_auc": 3.8, "k_state_mib": 2.0,
                    "optimizer_state_mib": 3.0, "peak_memory_mib": 4.0,
                }],
                r1_fields,
            )
            write_csv(
                root / "mousse.csv",
                [{
                    "seed": 2024, "method": "mousse", "run_name": "mousse", "initial_val_loss": 5.0,
                    "final_val_loss": 3.55, "best_val_loss": 3.54, "tail5_val_loss_mean": 3.56,
                    "normalized_val_auc": 3.9, "optimizer_state_mib": 5.0,
                    "peak_memory_mib": 6.0, "timing_eligible": False,
                }],
                unified_fields,
            )
            for name in ("record", "r1", "mousse"):
                write_json(root / f"{name}_manifest.json", {"status": "passed"})
            write_json(
                root / "catalog.json",
                {
                    "schema_version": "mdp_submission_input_catalog_v1",
                    "synthetic": True,
                    "sources": [
                        {
                            "source_id": "record", "schema": "record_cells_v1",
                            "analysis_family": "record", "scale_id": "275m", "csv": "record.csv",
                            "manifest": "record_manifest.json", "manifest_require": {"status": "passed"},
                            "expected_methods": ["original"], "expected_seeds": [2024],
                        },
                        {
                            "source_id": "r1", "schema": "r1_core_run_summary_v1",
                            "analysis_family": "r1", "scale_id": "124m", "csv": "r1.csv",
                            "manifest": "r1_manifest.json", "manifest_require": {"status": "passed"},
                            "expected_methods": ["original"], "expected_seeds": [2024],
                        },
                        {
                            "source_id": "mousse", "schema": "mousse_unified_run_summary_v1",
                            "analysis_family": "external", "scale_id": "124m", "csv": "mousse.csv",
                            "manifest": "mousse_manifest.json", "manifest_require": {"status": "passed"},
                            "expected_methods": ["mousse"], "expected_seeds": [2024],
                        },
                    ],
                },
            )
            output = root / "ledger"
            build_ledger(root / "catalog.json", output)
            rows = {row["source_id"]: row for row in read_csv(output / "evidence_ledger.csv")}
            self.assertEqual(rows["record"]["method"], "original")
            self.assertEqual(rows["r1"]["method"], "original")
            self.assertEqual(float(rows["r1"]["k_state_bytes"]), 2 * 1024 * 1024)
            self.assertEqual(rows["mousse"]["method"], "mousse")
            self.assertEqual(float(rows["mousse"]["optimizer_state_bytes"]), 5 * 1024 * 1024)

    def test_end_to_end_synthetic_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_fixture(root)
            output = root / "bundle"
            result = run(config, output)
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["synthetic"])
            self.assertFalse(read_json(output / "audit" / "submission_bundle_audit_manifest.json")["claim_eligible"])

            ledger = read_csv(output / "evidence" / "evidence_ledger.csv")
            self.assertEqual(len(ledger), 36)
            contrasts = read_csv(output / "cross_scale" / "paired_contrasts_by_scale.csv")
            self.assertEqual(len(contrasts), 15)
            self.assertEqual({row["n_pairs"] for row in contrasts}, {"3"})
            cross_manifest = read_json(output / "cross_scale" / "cross_scale_analysis_manifest.json")
            self.assertFalse(cross_manifest["seed_pooling_across_scales"])
            self.assertEqual(cross_manifest["scale_count"], 3)

            report = (output / "figures" / "SUBMISSION_TABLES.md").read_text(encoding="utf-8")
            self.assertIn("SYNTHETIC TEST DATA", report)
            ET.parse(output / "figures" / "cross_scale_forest.svg")
            ET.parse(output / "figures" / "quality_state_pareto.svg")
            refresh = read_json(output / "refresh" / "refresh_stability_manifest.json")
            self.assertEqual(refresh["origin_count"], 4)
            self.assertEqual(refresh["nested_replica_count"], 12)
            self.assertEqual(
                refresh["replica_interpretation"],
                "nested_replay_units_not_independent_training_seeds",
            )
            self.assertTrue(refresh["all_numeric_checks_passed"])
            self.assertTrue(refresh["all_inverse_checks_passed"])
            self.assertEqual(refresh["matched_gradient_source"], "gradient_before")
            equivariance = read_json(output / "equivariance" / "route_equivariance_manifest.json")
            self.assertTrue(equivariance["all_checks_passed"])
            self.assertEqual(len(read_csv(output / "equivariance" / "route_equivariance.csv")), 8)

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_fixture(root)
            output = root / "dry_run_output"
            result = run(config, output, dry_run=True)
            self.assertEqual(result["status"], "validated")
            self.assertFalse(output.exists())

    def test_duplicate_evidence_cell_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_fixture(root)
            pipeline = read_json(config)
            catalog_path = root / pipeline["catalog"]
            catalog = read_json(catalog_path)
            duplicate = dict(catalog["sources"][0])
            duplicate["source_id"] = "duplicate_source"
            catalog["sources"].append(duplicate)
            write_json(catalog_path, catalog)
            with self.assertRaisesRegex(ContractError, "duplicate evidence cell"):
                build_ledger(catalog_path, root / "unused", dry_run=True)

    def test_bundle_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_fixture(root)
            output = root / "bundle"
            run(config, output)
            target = output / "figures" / "cross_scale_forest.svg"
            target.write_text(target.read_text(encoding="utf-8") + "<!-- tampered -->\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "bundle audit failed"):
                audit_bundle(output, output / "reaudit")


if __name__ == "__main__":
    unittest.main()
