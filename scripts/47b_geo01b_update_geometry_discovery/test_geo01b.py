#!/usr/bin/env python3
"""Local CPU tests for experiment 47 / GEO-01B discovery."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import analyze_geo01b as A
import geo01b_worker as W
import protocol as P
import remote_controller as R


def synthetic_rows() -> tuple[list[dict], list[dict]]:
    contract_path = HERE / "geo01b_contract.json"
    contract_sha = P.sha256_file(contract_path)
    geometry = []
    outcomes = []
    for origin_index, origin in enumerate(P.ORIGINS):
        for replica_index, replica in enumerate(P.REPLICAS):
            target = 0.10 + 0.02 * origin_index + 0.01 * replica_index
            full = target
            first = 0.10 + 0.02 * origin_index + (0.03, 0.01, 0.02)[replica_index]
            norm = 0.10 + 0.02 * origin_index + (0.03, 0.02, 0.01)[replica_index]
            for event_index, event in enumerate(P.EVENTS):
                multiplier = 1.0 + 0.1 * event_index
                values = {
                    "relative_direction_fro_norm": norm * multiplier,
                    "first_order_alignment": first * multiplier,
                    "taylor_actual_delta_loss": full * multiplier,
                }
                for scope in P.SCOPES:
                    geometry.append(
                        {
                            "origin": origin,
                            "data_replica": replica,
                            "event_id": event,
                            "scope_id": scope,
                            "contract_sha256": contract_sha,
                            **values,
                            "all_values_finite": True,
                            "parameters_unchanged": True,
                        }
                    )
                outcomes.append(
                    {
                        "origin": origin,
                        "data_replica": replica,
                        "event_id": event,
                        "norm_only_predictor": values[
                            "relative_direction_fro_norm"
                        ],
                        "first_order_predictor": values[
                            "first_order_alignment"
                        ],
                        "full_taylor_predictor": values[
                            "taylor_actual_delta_loss"
                        ],
                        "local_exact_delta_loss": full * multiplier,
                        "local_first_relative_error": 0.25,
                        "local_taylor_relative_error": 0.01,
                        "local_first_sign_match": True,
                        "local_taylor_sign_match": True,
                        "endpoint_normalized_loss_harm": target * multiplier,
                        "endpoint_raw_loss_harm": target * multiplier,
                        "trapezoid_normalized_auc_harm": target * multiplier * 8.0,
                        "all_values_finite": True,
                    }
                )
    return geometry, outcomes


class Geo01BTests(unittest.TestCase):
    def test_contract_and_offsets_are_frozen(self) -> None:
        contract = P.read_json(HERE / "geo01b_contract.json")
        checks = P.validate_contract(contract)
        self.assertTrue(all(checks.values()), checks)
        certificate = P.build_offset_certificate(contract)
        self.assertTrue(certificate["passed"], certificate)
        self.assertFalse(contract["execution"]["confirmation_enabled"])
        self.assertFalse(contract["claim_boundary"]["llama_10b_triggered"])

    def test_job_grid_is_exact_and_new(self) -> None:
        contract = P.read_json(HERE / "geo01b_contract.json")
        jobs = P.job_matrix(contract)
        self.assertEqual(len(jobs), 12)
        self.assertEqual(
            {(row["origin"], row["data_replica"]) for row in jobs},
            {(origin, replica) for origin in P.ORIGINS for replica in P.REPLICAS},
        )
        self.assertTrue(all(row["data_replica"] >= 9 for row in jobs))

    def test_execution_contract_has_exact_outer_grid(self) -> None:
        contract = P.read_json(HERE / "geo01b_contract.json")
        source = P.read_json(
            HERE.parent
            / "37_mech09_downproj_refresh_mediation"
            / "refresh_mediation_repair_contract.json"
        )
        derived = P.derive_execution_contract(source, contract, "a" * 64)
        checks = P.validate_derived_execution_contract(derived, contract)
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual(derived["formal"]["data_replicas"], [9, 10, 11])
        self.assertEqual(
            derived["stopping_rule"]["maximum_new_formal_jobs"], 12
        )

    def test_rank_helpers_cover_ties_and_centering(self) -> None:
        self.assertEqual(P.ranks([2, 1, 2]), [2.5, 1.0, 2.5])
        self.assertAlmostEqual(P.spearman([1, 2, 3], [3, 2, 1]), -1.0)
        centered = P.centered([1, 3, 10, 14], ["a", "a", "b", "b"])
        self.assertEqual(centered, [-1.0, 1.0, -2.0, 2.0])

    def test_synthetic_supported_result_respects_claim_boundary(self) -> None:
        geometry, outcomes = synthetic_rows()
        contract_path = HERE / "geo01b_contract.json"
        result = A.analyze(
            geometry,
            outcomes,
            P.read_json(contract_path),
            P.sha256_file(contract_path),
        )
        self.assertTrue(result["integrity_passed"], result)
        self.assertEqual(result["scientific_result"], "directional_geometry_supported")
        self.assertEqual(result["curvature_increment_result"], "curvature_increment_supported")
        self.assertTrue(result["confirmation_candidate"])
        self.assertFalse(result["confirmation_authorized"])
        self.assertFalse(result["claim_eligible"])

    def test_analyzer_refuses_incomplete_outcome_grid(self) -> None:
        geometry, outcomes = synthetic_rows()
        contract_path = HERE / "geo01b_contract.json"
        with self.assertRaises(RuntimeError):
            A.analyze(
                geometry,
                outcomes[:-1],
                P.read_json(contract_path),
                P.sha256_file(contract_path),
            )

    def test_analyzer_detects_contract_lineage_mismatch(self) -> None:
        geometry, outcomes = synthetic_rows()
        geometry[0]["contract_sha256"] = "0" * 64
        contract_path = HERE / "geo01b_contract.json"
        result = A.analyze(
            geometry,
            outcomes,
            P.read_json(contract_path),
            P.sha256_file(contract_path),
        )
        self.assertFalse(result["integrity_passed"])
        self.assertEqual(result["scientific_result"], "integrity_failed")

    def test_worker_outcome_lineage_uses_frozen_arm_and_nodes(self) -> None:
        contract = P.read_json(HERE / "geo01b_contract.json")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            fields = (
                "arm",
                "trajectory_node",
                "optimizer_step",
                "heldout_loss",
                "normalized_loss",
            )
            rows = []
            for event in contract["discovery"]["events"]:
                for role, arm_key, trajectory_key, bias in (
                    ("treatment", "treatment_arm", "treatment_trajectory", 0.02),
                    ("reference", "reference_arm", "reference_trajectory", 0.00),
                ):
                    del role
                    for step in (event["completed_step"], event["endpoint_step"]):
                        rows.append(
                            {
                                "arm": event[arm_key],
                                "trajectory_node": event[trajectory_key],
                                "optimizer_step": step,
                                "heldout_loss": 3.0 + bias + step / 10000.0,
                                "normalized_loss": 1.0 + bias + step / 10000.0,
                            }
                        )
            with (output / "evaluation.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            recorder = object.__new__(W.DiscoveryRecorder)
            recorder.output = output
            recorder.discovery_contract = contract
            recorder.worker_args = argparse.Namespace(
                cell="early_muon", data_replica=9
            )
            recorder.rows = [
                {
                    "event_id": event,
                    "scope_id": "all_down",
                    "relative_direction_fro_norm": 0.1,
                    "first_order_alignment": 0.015,
                    "taylor_actual_delta_loss": 0.019,
                    "exact_actual_delta_loss": 0.02,
                }
                for event in P.EVENTS
            ]
            outcomes = recorder.build_outcomes()
            self.assertEqual(len(outcomes), 2)
            self.assertTrue(
                all(row["endpoint_normalized_loss_harm"] > 0 for row in outcomes)
            )
            self.assertTrue(all(row["all_values_finite"] for row in outcomes))

    def test_remote_controller_full_dry_run_is_source_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "accepted_source"
            (source / "formal").mkdir(parents=True)
            P.atomic_json(
                source / "formal" / "formal_manifest.json",
                {"passed": True, "completed_jobs": 12},
            )
            pinned = root / "pinned.py"
            pinned.write_text("# fixture\n", encoding="utf-8")
            rows = []
            for origin in P.ORIGINS:
                command = [
                    sys.executable,
                    str(pinned),
                    "--output-dir",
                    str(root / "old"),
                    "--analysis-tier",
                    "formal",
                    "--cell",
                    origin,
                    "--data-replica",
                    "0",
                    "--contract",
                    str(pinned),
                    "--triton-kernels",
                    str(pinned),
                    "--mech08-control-reference",
                    str(pinned),
                    "--checkpoint",
                    str(pinned),
                    "--checkpoint-hash-certificate",
                    str(pinned),
                    "--source-script",
                    str(pinned),
                    "--profile-script",
                    str(pinned),
                    "--smoke-manifest",
                    str(pinned),
                ]
                rows.append(
                    {"label": f"formal/{origin}/replica_0", "command": command}
                )
            (source / "commands.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            run_dir = root / "dryrun"
            result = R.controller(
                argparse.Namespace(
                    mode="dry-run",
                    run_dir=run_dir,
                    source_run=source,
                    child_python=Path(sys.executable),
                    gpus=["0", "1"],
                )
            )
            self.assertEqual(result, 0)
            status = P.read_json(run_dir / "status.json")
            self.assertEqual(status["status"], "dry_run_passed")
            plan = P.read_json(run_dir / "sealed" / "discovery_plan.json")
            self.assertEqual(plan["unit_count"], 12)
            self.assertEqual(plan["maximum_parallel_jobs"], 2)
            self.assertFalse(plan["confirmation_authorized"])

    def test_training_python_is_not_symlink_resolved(self) -> None:
        relative = Path("frozen_venv") / "bin" / "python"
        self.assertEqual(
            R.absolute_without_resolving(relative), Path.absolute(relative)
        )
        source = Path(R.__file__).read_text(encoding="utf-8")
        self.assertIn("child_python = absolute_without_resolving", source)
        self.assertNotIn("args.child_python.resolve()", source)

    def test_launcher_pins_both_python_roles_and_blocks_confirmation(self) -> None:
        launcher = (
            REPO / "commands/47b_geo01b_update_geometry_discovery/20260804_ex47b_geo01b_discovery.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("GEO01B_CONTROLLER_PYTHON", launcher)
        self.assertIn("GEO01B_TRAINING_PYTHON", launcher)
        self.assertIn("${SNM_TRAINING_PYTHON}", launcher)
        self.assertIn("confirmation|llama-10b", launcher)

    def test_gpu_lanes_are_fixed_and_sequential(self) -> None:
        source = Path(R.__file__).read_text(encoding="utf-8")
        self.assertIn("lanes = [jobs[0::2], jobs[1::2]]", source)
        self.assertIn("for job in lane:", source)
        self.assertIn("ThreadPoolExecutor(max_workers=2)", source)
        self.assertNotIn("pool.submit(run_unit", source)


if __name__ == "__main__":
    unittest.main()
