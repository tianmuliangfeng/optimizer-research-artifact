from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = REPO_ROOT / "reproducibility"
sys.path.insert(0, str(REPRO_ROOT))

from build_core_results_package import (  # noqa: E402
    BuildError,
    SubmissionIdentityAnonymizer,
    build_package,
)
from validate_core_results_package import (  # noqa: E402
    ValidationError,
    _scan_private_paths,
    validate_package,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status() -> dict[str, str]:
    return {
        "integrity_status": "accepted",
        "scientific_status": "supported_with_scope",
        "claim_eligibility": "claim_eligible",
        "paper_role": "main_table",
    }


class SubmissionAnonymityTests(unittest.TestCase):
    def test_anonymizer_rewrites_wandb_identity_url_and_container_host(self) -> None:
        service_url = "https://" + "wandb" + ".ai/"
        private_url = (
            service_url + "private-author/private-project/runs/private-run-id"
        )
        private_host = "app-" + "f" * 32 + "-worker-01"
        source = (
            "formal_seed:wandb_identity,"
            "{'run_name': 'private-run-name', 'run_id': 'private-run-id', "
            "'run_url': '"
            + private_url
            + "', 'hostname': '"
            + private_host
            + "'}\n"
        )
        anonymizer = SubmissionIdentityAnonymizer()
        released = anonymizer.anonymize_text(source)

        self.assertNotIn("private-author", released)
        self.assertNotIn("private-project", released)
        self.assertNotIn("private-run-name", released)
        self.assertNotIn(private_host, released)
        self.assertIn("https://wandb.invalid/ANONYMIZED_WANDB_ENTITY_0001", released)
        self.assertIn("ANONYMIZED_WANDB_RUN_ID_0001", released)
        self.assertIn("ANONYMIZED_WANDB_RUN_NAME_0001", released)
        self.assertIn("ANONYMIZED_CONTAINER_HOST_0001", released)
        self.assertEqual(released, anonymizer.anonymize_text(released))

    def test_validator_identity_gate_accepts_tokens_and_rejects_leaks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.json"
            path.write_text(
                json.dumps(
                    {
                        "entity": "ANONYMIZED_WANDB_ENTITY_0001",
                        "hostname": "ANONYMIZED_CONTAINER_HOST_0001",
                        "url": (
                            "https://wandb.invalid/ANONYMIZED_WANDB_ENTITY_0001/"
                            "ANONYMIZED_WANDB_PROJECT_0001/runs/"
                            "ANONYMIZED_WANDB_RUN_ID_0001"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            _scan_private_paths(path, "evidence.json")

            for leaked in (
                "https://" + "wandb" + ".ai/private-author/private-project/runs/private-id",
                "{'entity': 'private-author'}",
                "app-" + "f" * 32 + "-worker-01",
            ):
                path.write_text(leaked, encoding="utf-8")
                with self.assertRaises(ValidationError):
                    _scan_private_paths(path, "evidence.json")


def ex48_certificate_payloads() -> dict[str, dict[str, object]]:
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
            "artifacts": {"summary.csv": {"sha256": "4" * 64, "bytes": 10}},
        },
        "generic_verify": {
            "schema_version": "selective_newton_muon_archive_verify_v1",
            "passed": True,
            "checks": {
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
            },
            "json_file_count": 100,
            "source_snapshot_manifest_count": 1,
            "lineage": {"experiment_id": "48_llama1b_10b_multibudget"},
            "run_dir": "/data/private/ex48/run",
            "results_root": "/data/private/ex48",
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
                {"path": "/data/private/ex48/pilot-a.pt", "sha256": "5" * 64, "bytes": 100},
                {"path": "/data/private/ex48/pilot-b.pt", "sha256": "6" * 64, "bytes": 100},
            ],
        },
    }


class CoreResultsPackagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "accepted-results"
        (self.source / "direct").mkdir(parents=True)
        (self.source / "graph").mkdir(parents=True)
        self.selection = self.root / "selection.json"
        self.output = self.root / "core-results"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_selection(self, payload: dict[str, object]) -> None:
        self.selection.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def base_direct(self, *, experiment_id: str = "15") -> dict[str, object]:
        note = self.source / "direct" / "note.md"
        note.write_text(
            "selected="
            + str(note)
            + "\nprivate=/"
            + "data/secret/run/output.csv\nlegacy=selective-newton-"
            + "muon-main-conference\n",
            encoding="utf-8",
        )
        return {
            "evidence_id": f"ex{experiment_id}_note",
            "experiment_id": experiment_id,
            "source_root": "results",
            "source": "direct/note.md",
            "package_path": f"evidence/ex{experiment_id}/note.md",
            "source_sha256": digest(note),
            "workstream": "quality",
            **status(),
        }

    def public_snapshot_entries(
        self,
    ) -> tuple[Path, Path, Path, list[dict[str, object]]]:
        public_source = self.root / "public-source"
        catalog = public_source / "experiments" / "catalog.json"
        anchors = public_source / "provenance" / "accepted_result_anchors.json"
        catalog.parent.mkdir(parents=True)
        anchors.parent.mkdir(parents=True)
        catalog.write_text(
            json.dumps(
                {"experiments": [{"experiment_id": "15"}]},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        anchors.write_text(
            json.dumps({"records": []}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        entries = [
            {
                "evidence_id": "public_experiment_catalog",
                "source_root": "source",
                "source": "experiments/catalog.json",
                "package_path": "provenance/public_source/experiments/catalog.json",
                "source_sha256": digest(catalog),
                "workstream": "provenance",
                **status(),
            },
            {
                "evidence_id": "accepted_result_anchors",
                "source_root": "source",
                "source": "provenance/accepted_result_anchors.json",
                "package_path": (
                    "provenance/public_source/provenance/accepted_result_anchors.json"
                ),
                "source_sha256": digest(anchors),
                "workstream": "provenance",
                **status(),
            },
        ]
        return public_source, catalog, anchors, entries

    def test_draft_build_is_relocatable_graph_aware_and_reproducible(self) -> None:
        direct = self.base_direct()
        summary = self.source / "direct" / "checkpoint_summary.csv"
        summary.write_text("step,value\n6200,3.1\n", encoding="utf-8")
        summary_entry = {
            "evidence_id": "checkpoint_summary_is_compact_evidence",
            "experiment_id": "20",
            "source_root": "results",
            "source": "direct/checkpoint_summary.csv",
            "package_path": "evidence/systems/checkpoint_summary.csv",
            "source_sha256": digest(summary),
            "workstream": "systems",
            **status(),
        }

        child = self.source / "graph" / "value.json"
        child.write_text(
            json.dumps(
                {
                    str(self.source / "private-a"): {"ok": True},
                    str(self.source / "private-b"): {"ok": False},
                    "run_directory": "../raw-formal-tree",
                    "metric": 1.25,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        graph = self.source / "graph" / "index.csv"
        with graph.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["artifact_path", "sha256"], lineterminator="\n"
            )
            writer.writeheader()
            writer.writerow({"artifact_path": "graph/value.json", "sha256": digest(child)})

        self.write_selection(
            {
                "schema_version": 1,
                "package_name": "core-results",
                "release_mode": "draft",
                "pending_experiments": ["EX48 acceptance"],
                "included_experiments": ["15", "20"],
                "direct_files": [direct, summary_entry],
                "csv_graph_dependencies": [
                    {
                        "graph_id": "submission_graph",
                        "source_root": "results",
                        "root_csv": {
                            "source": "graph/index.csv",
                            "source_sha256": digest(graph),
                        },
                        "path_columns": ["artifact_path"],
                        "hash_columns": {"artifact_path": "sha256"},
                        "destination_root": "evidence/dependency_graph",
                        "relative_to": "source_root",
                        "experiment_id": "38",
                        "workstream": "mechanism",
                        **status(),
                    }
                ],
            }
        )

        built = build_package(
            self.selection, {"results": self.source}, self.output, mode="draft"
        )
        self.assertEqual(built, self.output.resolve())
        self.assertTrue(validate_package(built)["passed"])
        self.assertTrue((built / "evidence/systems/checkpoint_summary.csv").is_file())

        index_text = (built / "evidence/dependency_graph/graph/index.csv").read_text(
            encoding="utf-8"
        )
        self.assertIn("evidence/dependency_graph/graph/value.json", index_text)
        released_child = json.loads(
            (built / "evidence/dependency_graph/graph/value.json").read_text(encoding="utf-8")
        )
        redacted_keys = [key for key in released_child if key.startswith("PRIVATE_PATH_REDACTED__")]
        self.assertEqual(len(redacted_keys), 2)
        self.assertEqual(len(set(redacted_keys)), 2)
        self.assertTrue(released_child[redacted_keys[0]]["ok"] in {True, False})
        self.assertTrue(released_child["run_directory"].startswith("PRIVATE_PATH_REDACTED__"))
        self.assertEqual(released_child["metric"], 1.25)
        all_text = "\n".join(
            path.read_text(encoding="utf-8-sig")
            for path in built.rglob("*")
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".py"}
        )
        self.assertNotRegex(all_text, r"[A-Za-z]:[\\/]")
        self.assertNotIn("/" + "data/", all_text)
        self.assertNotIn("selective-newton-" + "muon-main-conference", all_text)
        self.assertIn("project-results", all_text)

        first = {
            path.relative_to(built).as_posix(): path.read_bytes()
            for path in built.rglob("*")
            if path.is_file()
        }
        build_package(
            self.selection,
            {"results": self.source},
            self.output,
            mode="draft",
            replace=True,
        )
        second = {
            path.relative_to(built).as_posix(): path.read_bytes()
            for path in built.rglob("*")
            if path.is_file()
        }
        self.assertEqual(first, second)

        moved_parent = self.root / "moved"
        moved_parent.mkdir()
        moved = moved_parent / "core-results"
        shutil.move(str(built), str(moved))
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "tools/validate_core_results_package.py", "."],
            cwd=moved,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_source_anchor_mismatch_fails_closed(self) -> None:
        entry = self.base_direct()
        entry["source_sha256"] = "0" * 64
        self.write_selection(
            {
                "package_name": "core-results",
                "release_mode": "draft",
                "pending_experiments": ["EX48"],
                "direct_files": [entry],
            }
        )
        with self.assertRaisesRegex(BuildError, "source anchor mismatch"):
            build_package(self.selection, {"results": self.source}, self.output)

    def test_public_source_parity_is_optional_for_movable_standalone_audit(self) -> None:
        direct = self.base_direct()
        public_source, catalog, anchors, snapshot_entries = self.public_snapshot_entries()
        self.write_selection(
            {
                "package_name": "core-results",
                "release_mode": "draft",
                "pending_experiments": ["EX48"],
                "included_experiments": ["15"],
                "direct_files": [direct, *snapshot_entries],
            }
        )

        built = build_package(
            self.selection,
            {"results": self.source, "source": public_source},
            self.output,
        )
        self.assertTrue(
            validate_package(built, public_source_root=public_source)["passed"]
        )

        artifact_manifest = json.loads(
            (built / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        records = {
            record["evidence_id"]: record
            for record in artifact_manifest["artifacts"]
        }
        for evidence_id, source_path, source_relpath in (
            ("public_experiment_catalog", catalog, "experiments/catalog.json"),
            (
                "accepted_result_anchors",
                anchors,
                "provenance/accepted_result_anchors.json",
            ),
        ):
            record = records[evidence_id]
            package_path = built / Path(*record["package_path"].split("/"))
            self.assertEqual(record["source_alias"], "source")
            self.assertEqual(record["source_relpath"], source_relpath)
            self.assertEqual(record["source_sha256"], digest(source_path))
            self.assertEqual(record["source_bytes"], source_path.stat().st_size)
            self.assertEqual(record["package_sha256"], digest(package_path))
            self.assertEqual(record["package_bytes"], package_path.stat().st_size)
            self.assertEqual(source_path.read_bytes(), package_path.read_bytes())

        moved = self.root / "relocated" / "core-results"
        moved.parent.mkdir()
        shutil.move(str(built), str(moved))
        self.assertTrue(validate_package(moved)["passed"])

        catalog.write_text(
            json.dumps(
                {"experiments": [{"experiment_id": "15", "drifted": True}]},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        report = validate_package(
            moved, public_source_root=public_source, raise_on_error=False
        )
        self.assertFalse(report["passed"])
        self.assertIn("public source snapshot drifted", report["errors"][0])
        self.assertTrue(validate_package(moved)["passed"])

    def test_build_end_parity_closes_catalog_drift_race(self) -> None:
        direct = self.base_direct()
        public_source, catalog, _, snapshot_entries = self.public_snapshot_entries()
        self.write_selection(
            {
                "package_name": "core-results",
                "release_mode": "draft",
                "pending_experiments": ["EX48"],
                "included_experiments": ["15"],
                "direct_files": [direct, *snapshot_entries],
            }
        )

        def drift_before_final_validation(
            package_root: Path | str, **kwargs: object
        ) -> dict[str, object]:
            catalog.write_text(
                json.dumps(
                    {"experiments": [{"experiment_id": "15", "drifted": True}]},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return validate_package(package_root, **kwargs)

        with mock.patch(
            "build_core_results_package.validate_package",
            side_effect=drift_before_final_validation,
        ):
            with self.assertRaisesRegex(
                BuildError, "public source snapshot drifted during build"
            ):
                build_package(
                    self.selection,
                    {"results": self.source, "source": public_source},
                    self.output,
                )
        self.assertFalse(self.output.exists())

    def test_final_requires_no_pending_key_and_accepted_ex48(self) -> None:
        ex48_id = "48_llama1b_10b_multibudget"
        entry = self.base_direct(experiment_id=ex48_id)
        gate_paths = {
            role: f"evidence/ex48/{role}.json"
            for role in ex48_certificate_payloads()
        }
        certificate_entries = []
        for role, certificate in ex48_certificate_payloads().items():
            source = self.source / "direct" / f"ex48_{role}.json"
            source.write_text(
                json.dumps(certificate, sort_keys=True) + "\n", encoding="utf-8"
            )
            certificate_entries.append(
                {
                    "evidence_id": f"ex48_{role}",
                    "experiment_id": ex48_id,
                    "source_root": "results",
                    "source": f"direct/ex48_{role}.json",
                    "package_path": gate_paths[role],
                    "source_sha256": digest(source),
                    "workstream": "long_token_confirmation",
                    **status(),
                }
            )
        catalog = self.source / "direct" / "catalog.json"
        catalog.write_text(
            json.dumps({"experiments": [{"experiment_id": ex48_id}]}) + "\n",
            encoding="utf-8",
        )
        anchors = self.source / "direct" / "accepted_result_anchors.json"
        anchors.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "experiment_id": ex48_id,
                            "accepted": True,
                            "anchors": [
                                {
                                    "path": certificate_entry["source"],
                                    "sha256": certificate_entry["source_sha256"],
                                }
                                for role, certificate_entry in zip(
                                    gate_paths, certificate_entries
                                )
                            ],
                        }
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        catalog_entry = {
            "id": "public_experiment_catalog",
            "source_root": "results",
            "source": "direct/catalog.json",
            "package_path": "provenance/public_source/experiments/catalog.json",
            "source_sha256": digest(catalog),
            **status(),
        }
        anchors_entry = {
            "id": "accepted_result_anchors",
            "source_root": "results",
            "source": "direct/accepted_result_anchors.json",
            "package_path": "provenance/public_source/provenance/accepted_result_anchors.json",
            "source_sha256": digest(anchors),
            **status(),
        }
        payload = {
            "package_name": "core-results",
            "release_mode": "final",
            "included_experiments": [ex48_id],
            "ex48_final_gate": {
                "experiment_id": ex48_id,
                "artifacts": gate_paths,
            },
            "direct_files": [
                entry,
                *certificate_entries,
                catalog_entry,
                anchors_entry,
            ],
        }
        self.write_selection(payload)
        build_package(self.selection, {"results": self.source}, self.output, mode="final")
        release = json.loads((self.output / "release_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(release["release_mode"], "final")
        self.assertFalse(any(key.startswith("pending") for key in release))

        shutil.rmtree(self.output)
        payload["pending_experiments"] = []
        self.write_selection(payload)
        with self.assertRaisesRegex(BuildError, "omit pending fields"):
            build_package(self.selection, {"results": self.source}, self.output, mode="final")

        payload.pop("pending_experiments")
        payload["included_experiments"] = ["15"]
        payload["direct_files"][0]["experiment_id"] = "15"  # type: ignore[index]
        payload["direct_files"][0]["evidence_id"] = "ex15_note"  # type: ignore[index]
        payload["direct_files"][0]["package_path"] = "evidence/ex15/note.md"  # type: ignore[index]
        for direct_file in payload["direct_files"]:  # type: ignore[assignment]
            if str(direct_file.get("evidence_id", "")).startswith("ex48_"):
                direct_file["experiment_id"] = "15"
        self.write_selection(payload)
        with self.assertRaisesRegex(BuildError, "EX48"):
            build_package(self.selection, {"results": self.source}, self.output, mode="final")

    def test_validator_rejects_unregistered_and_private_files(self) -> None:
        entry = self.base_direct()
        self.write_selection(
            {
                "package_name": "core-results",
                "release_mode": "draft",
                "pending_experiments": ["EX48"],
                "direct_files": [entry],
            }
        )
        build_package(self.selection, {"results": self.source}, self.output)
        (self.output / "rogue.txt").write_text(
            "private=/" + "data/secret/file.csv\n", encoding="utf-8"
        )
        report = validate_package(self.output, raise_on_error=False)
        self.assertFalse(report["passed"])
        self.assertIn("unregistered", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
