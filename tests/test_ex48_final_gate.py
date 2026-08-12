from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "reproducibility"
    / "validate_core_results_package.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("core_results_validator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)

import build_core_results_package as B  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def certificate_payloads() -> dict[str, dict[str, object]]:
    analysis_checks = {
        key: True
        for key in (
            "contract",
            "source_snapshot",
            "preflight",
            "pilot",
            "suite",
            "data",
            "unit_count",
            "phase_count",
            "endpoint_count",
            "retained_checkpoint_count",
            "lineage",
            "same_seed_initialization",
        )
    }
    generic_checks = {
        key: True
        for key in (
            "run_dir_within_results_root",
            "json_present",
            "all_json_parse",
            "source_snapshot_manifest_present",
            "source_snapshot_manifests",
            "sealed_snapshot_requirement_satisfied",
            "experiment_lineage_bound",
        )
    }
    native_checks = dict(analysis_checks)
    native_checks.update(
        {"analysis_manifest": True, "analysis_artifacts": True, "handoff": True}
    )
    return {
        "analysis_manifest": {
            "schema_version": "ex48_analysis_manifest_v1",
            "passed": True,
            "claim_eligible": True,
            "formal_units": 12,
            "primary_endpoints": 36,
            "contract_sha256": "2" * 64,
            "data_inventory_sha256": "3" * 64,
            "integrity_checks": analysis_checks,
            "artifacts": {"endpoint_summary.csv": {"sha256": "4" * 64, "bytes": 10}},
        },
        "generic_verify": {
            "schema_version": "selective_newton_muon_archive_verify_v1",
            "passed": True,
            "checks": generic_checks,
            "json_file_count": 100,
            "source_snapshot_manifest_count": 1,
            "lineage": {"experiment_id": V.EX48_EXPERIMENT_ID},
        },
        "native_verify": {
            "passed": True,
            "full_checkpoint_hash": True,
            "checks": native_checks,
        },
        "source_data_resume_lineage": {
            "schema_version": "ex48_engineering_pilot_v1",
            "passed": True,
            "planned_interrupt_return_code": 75,
            "in_place_resume": True,
            "source_checkpoint_branch": True,
            "no_wrap": True,
            "retired_pilot_checkpoints": [
                {"sha256": "5" * 64, "bytes": 100},
                {"sha256": "6" * 64, "bytes": 100},
            ],
        },
    }


def gate_paths() -> dict[str, str]:
    return {
        role: f"evidence/ex48/{role}.json" for role in V.EX48_GATE_ROLES
    }


def gate_value() -> dict[str, object]:
    return {"experiment_id": V.EX48_EXPERIMENT_ID, "artifacts": gate_paths()}


class Ex48FinalGateTests(unittest.TestCase):
    @staticmethod
    def artifact_plan(
        *,
        experiment_ids: tuple[str, ...],
        integrity_status: str = "accepted",
        evidence_id: str = "artifact",
        source_relpath: str = "artifact.json",
        package_path: str | None = None,
    ) -> B.ArtifactPlan:
        return B.ArtifactPlan(
            evidence_id=evidence_id,
            source_alias="results",
            source_relpath=source_relpath,
            package_path=package_path or f"evidence/{source_relpath}",
            expected_sha256="0" * 64,
            integrity_status=integrity_status,
            scientific_status="supported_with_scope",
            claim_eligibility="claim_eligible",
            paper_role="main_table",
            experiment_id=experiment_ids[0],
            experiment_ids=experiment_ids,
        )

    def test_artifact_detection_requires_exact_accepted_experiment(self) -> None:
        decoy = {
            "evidence_id": "audit_48_deadbeef",
            "experiment_ids": ["15_r1_small"],
            "integrity_status": "accepted",
            "source_relpath": "reports/hash_48.json",
            "package_path": "evidence/hash_48.json",
        }
        self.assertFalse(
            V._has_experiment_48(
                [decoy], {"included_experiments": ["audit_48_deadbeef"]}
            )
        )
        exact = {
            "experiment_ids": [V.EX48_EXPERIMENT_ID],
            "integrity_status": "accepted_with_independent_review",
        }
        self.assertTrue(
            V._has_experiment_48(
                [exact], {"included_experiments": [V.EX48_EXPERIMENT_ID]}
            )
        )

    def test_builder_rejects_decoy_48_tokens_and_requires_accepted_artifact(self) -> None:
        decoy = self.artifact_plan(
            experiment_ids=("15_r1_small",),
            evidence_id="audit_48_deadbeef",
            source_relpath="hash_48.json",
        )
        with self.assertRaisesRegex(B.BuildError, "exact experiment"):
            B._enforce_release_mode(
                {"included_experiments": ["audit_48_deadbeef"]}, "final", [decoy]
            )

    def test_certificate_content_is_derived_and_fail_closed(self) -> None:
        payloads = certificate_payloads()
        V._validate_ex48_certificate_payloads(payloads)
        mutations = (
            ("analysis_manifest", "formal_units", 11),
            ("analysis_manifest", "primary_endpoints", 35),
            ("generic_verify", "passed", False),
            ("native_verify", "full_checkpoint_hash", False),
            ("source_data_resume_lineage", "in_place_resume", False),
        )
        for role, key, value in mutations:
            invalid = copy.deepcopy(payloads)
            invalid[role][key] = value
            with self.subTest(role=role, key=key), self.assertRaises(V.ValidationError):
                V._validate_ex48_certificate_payloads(invalid)

    def test_builder_gate_reads_actual_selected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plans = []
            for role, payload in certificate_payloads().items():
                source = f"certificates/{role}.json"
                write_json(root / source, payload)
                plans.append(
                    self.artifact_plan(
                        experiment_ids=(V.EX48_EXPERIMENT_ID,),
                        evidence_id=f"ex48_{role}",
                        source_relpath=source,
                        package_path=gate_paths()[role],
                    )
                )
            normalized = B._ex48_final_gate(
                {"ex48_final_gate": gate_value()}, plans, {"results": root}
            )
            self.assertEqual(normalized, gate_value())
            plans[0].integrity_status = "pending"
            with self.assertRaisesRegex(B.BuildError, "accepted EX48 artifact"):
                B._ex48_final_gate(
                    {"ex48_final_gate": gate_value()}, plans, {"results": root}
                )

    def test_builder_anchor_snapshot_requires_every_gate_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            certificate_plans = [
                self.artifact_plan(
                    experiment_ids=(V.EX48_EXPERIMENT_ID,),
                    evidence_id=f"ex48_{role}",
                    source_relpath=f"certificates/{role}.json",
                    package_path=path,
                )
                for role, path in gate_paths().items()
            ]
            anchor_plan = self.artifact_plan(
                experiment_ids=(V.EX48_EXPERIMENT_ID,),
                evidence_id="accepted_result_anchors",
                source_relpath="anchors.json",
                package_path="provenance/anchors.json",
            )
            plans = certificate_plans + [anchor_plan]
            complete = [
                {"path": path, "sha256": str(index + 1) * 64}
                for index, path in enumerate(gate_paths().values())
            ]
            for anchors, expected in ((complete[:-1], False), (complete, True)):
                write_json(
                    root / "anchors.json",
                    {
                        "records": [
                            {
                                "experiment_id": V.EX48_EXPERIMENT_ID,
                                "accepted": True,
                                "anchors": anchors,
                            }
                        ]
                    },
                )
                self.assertEqual(
                    B._accepted_ex48_in_snapshot(
                        plans, {"results": root}, set(gate_paths().values())
                    ),
                    expected,
                )

    def test_standalone_gate_reads_package_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records: dict[str, dict[str, object]] = {}
            for role, payload in certificate_payloads().items():
                relative = gate_paths()[role]
                path = root / Path(*Path(relative).parts)
                write_json(path, payload)
                records[relative] = {
                    "package_path": relative,
                    "package_sha256": sha256(path),
                    "integrity_status": "accepted",
                    "experiment_ids": [V.EX48_EXPERIMENT_ID],
                }
            self.assertEqual(
                V._validate_ex48_gate(root, gate_value(), records),
                set(gate_paths().values()),
            )
            analysis_path = root / Path(*Path(gate_paths()["analysis_manifest"]).parts)
            invalid = certificate_payloads()["analysis_manifest"]
            invalid["formal_units"] = 1
            write_json(analysis_path, invalid)
            with self.assertRaisesRegex(V.ValidationError, "12 units/36 endpoints"):
                V._validate_ex48_gate(root, gate_value(), records)

    def test_accepted_anchor_path_and_hash_bind_package_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "catalog.json",
                {"experiments": [{"experiment_id": V.EX48_EXPERIMENT_ID}]},
            )
            release = {
                "included_experiments": [V.EX48_EXPERIMENT_ID],
                "public_catalog": "catalog.json",
                "accepted_result_anchors": "anchors.json",
            }
            artifacts = [
                {
                    "experiment_ids": [V.EX48_EXPERIMENT_ID],
                    "integrity_status": "accepted",
                    "package_path": path,
                    "package_sha256": str(index + 1) * 64,
                    "source_relpath": f"source/ex48/{index}.json",
                    "source_sha256": chr(ord("a") + index) * 64,
                }
                for index, path in enumerate(gate_paths().values())
            ]
            anchors = [
                {"path": row["package_path"], "sha256": row["package_sha256"]}
                for row in artifacts
            ]
            write_json(
                root / "anchors.json",
                {
                    "records": [
                        {
                            "experiment_id": V.EX48_EXPERIMENT_ID,
                            "accepted": True,
                            "anchors": anchors,
                        }
                    ]
                },
            )
            V._validate_public_snapshots(
                root,
                release,
                artifacts,
                require_ex48_anchor=True,
                required_ex48_artifacts=set(gate_paths().values()),
            )
            anchors[0] = {
                "path": artifacts[0]["source_relpath"],
                "sha256": artifacts[0]["source_sha256"],
            }
            write_json(
                root / "anchors.json",
                {
                    "records": [
                        {
                            "experiment_id": V.EX48_EXPERIMENT_ID,
                            "accepted": True,
                            "anchors": anchors,
                        }
                    ]
                },
            )
            V._validate_public_snapshots(
                root,
                release,
                artifacts,
                require_ex48_anchor=True,
                required_ex48_artifacts=set(gate_paths().values()),
            )
            anchors[0]["sha256"] = "f" * 64
            write_json(
                root / "anchors.json",
                {
                    "records": [
                        {
                            "experiment_id": V.EX48_EXPERIMENT_ID,
                            "accepted": True,
                            "anchors": anchors,
                        }
                    ]
                },
            )
            with self.assertRaisesRegex(V.ValidationError, "hash does not match"):
                V._validate_public_snapshots(
                    root,
                    release,
                    artifacts,
                    require_ex48_anchor=True,
                    required_ex48_artifacts=set(gate_paths().values()),
                )


if __name__ == "__main__":
    unittest.main()
