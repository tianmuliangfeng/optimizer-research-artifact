from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = REPO_ROOT / "reproducibility"
sys.path.insert(0, str(REPRO_ROOT))

from build_core_results_package import BuildError, build_package  # noqa: E402
from validate_core_results_package import validate_package  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_package_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    relatives = [line.split("  ", 1)[1] for line in checksum_path.read_text().splitlines()]
    checksum_path.write_text(
        "".join(f"{digest(root / relative)}  {relative}\n" for relative in relatives),
        encoding="utf-8",
    )


def resign_artifact(root: Path, package_path: str) -> None:
    """Re-sign package-level envelopes without repairing an internal link."""

    artifact = root / package_path
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["artifacts"]:
        if record["package_path"] == package_path:
            record["package_sha256"] = digest(artifact)
            record["package_bytes"] = artifact.stat().st_size
            break
    else:  # pragma: no cover - test setup guard
        raise AssertionError(f"artifact record not found: {package_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    refresh_package_checksums(root)


def statuses() -> dict[str, str]:
    return {
        "integrity_status": "accepted",
        "scientific_status": "supported_with_scope",
        "claim_eligibility": "claim_eligible",
        "paper_role": "main_table",
    }


class CoreResultsDualHashTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "accepted-results"
        self.source.mkdir()
        self.selection = self.root / "selection.json"
        self.output = self.root / "core-results"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_selection(self, payload: dict[str, object]) -> None:
        self.selection.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def direct_entry(self, name: str, package_path: str) -> dict[str, object]:
        source = self.source / name
        return {
            "evidence_id": "dual_hash_" + name.replace(".", "_"),
            "experiment_id": "45",
            "source_root": "results",
            "source": name,
            "package_path": package_path,
            "source_sha256": digest(source),
            **statuses(),
        }

    def test_json_chain_and_sha256_sidecar_use_final_package_hashes(self) -> None:
        leaf = self.source / "leaf.json"
        leaf.write_text(
            json.dumps({"private_run_path": "/data/private/formal/run"}) + "\n",
            encoding="utf-8",
        )
        parent = self.source / "parent.json"
        parent.write_text(
            json.dumps({"artifact": {"path": str(leaf), "sha256": "3" * 64}})
            + "\n",
            encoding="utf-8",
        )
        grand = self.source / "grand.json"
        grand.write_text(
            json.dumps({"manifest": {"path": "parent.json", "sha256": digest(parent)}})
            + "\n",
            encoding="utf-8",
        )
        sidecar = self.source / "parent.sha256"
        sidecar.write_text(f"{'3' * 64}  parent.json\n", encoding="utf-8")

        self.write_selection(
            {
                "package_name": "core-results",
                "release_mode": "draft",
                "pending_experiments": ["48_llama1b_10b_multibudget"],
                "direct_files": [
                    self.direct_entry("leaf.json", "evidence/leaf.json"),
                    self.direct_entry("parent.json", "evidence/parent.json"),
                    self.direct_entry("grand.json", "evidence/grand.json"),
                    self.direct_entry("parent.sha256", "evidence/parent.sha256"),
                ],
            }
        )

        built = build_package(
            self.selection, {"results": self.source}, self.output, mode="draft"
        )
        self.assertTrue(validate_package(built)["passed"])

        leaf_package_sha256 = digest(built / "evidence/leaf.json")
        self.assertNotEqual(leaf_package_sha256, digest(leaf))
        released_parent = json.loads(
            (built / "evidence/parent.json").read_text(encoding="utf-8")
        )["artifact"]
        self.assertEqual(released_parent["path"], "evidence/leaf.json")
        self.assertEqual(released_parent["source_sha256"], digest(leaf))
        self.assertEqual(released_parent["package_sha256"], leaf_package_sha256)
        self.assertEqual(released_parent["sha256"], leaf_package_sha256)

        parent_package_sha256 = digest(built / "evidence/parent.json")
        released_grand = json.loads(
            (built / "evidence/grand.json").read_text(encoding="utf-8")
        )["manifest"]
        self.assertEqual(released_grand["path"], "evidence/parent.json")
        self.assertEqual(released_grand["source_sha256"], digest(parent))
        self.assertEqual(released_grand["package_sha256"], parent_package_sha256)
        self.assertEqual(released_grand["sha256"], parent_package_sha256)

        sidecar_text = (built / "evidence/parent.sha256").read_text(encoding="utf-8")
        self.assertEqual(
            sidecar_text, f"{parent_package_sha256}  evidence/parent.json\n"
        )

        records = {
            row["package_path"]: row
            for row in json.loads(
                (built / "artifact_manifest.json").read_text(encoding="utf-8")
            )["artifacts"]
        }
        self.assertEqual(records["evidence/leaf.json"]["source_sha256"], digest(leaf))
        self.assertEqual(
            records["evidence/leaf.json"]["package_sha256"], leaf_package_sha256
        )
        self.assertIn(
            "package_hash_links_rewritten",
            records["evidence/parent.json"]["transformation"],
        )

        stale_grand = json.loads(
            (built / "evidence/grand.json").read_text(encoding="utf-8")
        )
        stale_grand["manifest"]["sha256"] = "3" * 64
        (built / "evidence/grand.json").write_text(
            json.dumps(stale_grand, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifact_manifest = json.loads(
            (built / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        for record in artifact_manifest["artifacts"]:
            if record["package_path"] == "evidence/grand.json":
                record["package_sha256"] = digest(built / "evidence/grand.json")
                record["package_bytes"] = (built / "evidence/grand.json").stat().st_size
        (built / "artifact_manifest.json").write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        refresh_package_checksums(built)
        report = validate_package(built, raise_on_error=False)
        self.assertFalse(report["passed"])
        self.assertIn("internal package hash link mismatch", report["errors"][0])

    def test_identity_target_path_moves_and_multi_link_fields_do_not_collide(self) -> None:
        identity = self.source / "identity.json"
        identity.write_text(json.dumps({"metric": 1.0}) + "\n", encoding="utf-8")
        first = self.source / "first.json"
        first.write_text(json.dumps({"path": "/data/private/first"}) + "\n", encoding="utf-8")
        second = self.source / "second.json"
        second.write_text(json.dumps({"path": "/data/private/second"}) + "\n", encoding="utf-8")
        manifest = self.source / "multi.json"
        manifest.write_text(
            json.dumps(
                {
                    "identity_path": "identity.json",
                    "identity_sha256": digest(identity),
                    "first_path": "first.json",
                    "first_sha256": digest(first),
                    "second_path": "second.json",
                    "second_sha256": digest(second),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.write_selection(
            {
                "package_name": "core-results",
                "release_mode": "draft",
                "pending_experiments": ["48_llama1b_10b_multibudget"],
                "direct_files": [
                    self.direct_entry("identity.json", "evidence/moved/identity.json"),
                    self.direct_entry("first.json", "evidence/moved/first.json"),
                    self.direct_entry("second.json", "evidence/moved/second.json"),
                    self.direct_entry("multi.json", "evidence/multi.json"),
                ],
            }
        )
        built = build_package(
            self.selection, {"results": self.source}, self.output, mode="draft"
        )
        released = json.loads(
            (built / "evidence/multi.json").read_text(encoding="utf-8")
        )
        self.assertEqual(released["identity_path"], "evidence/moved/identity.json")
        self.assertEqual(released["identity_sha256"], digest(identity))
        self.assertEqual(released["identity_source_sha256"], digest(identity))
        self.assertEqual(released["identity_package_sha256"], digest(identity))
        self.assertNotIn("source_sha256", released)
        self.assertNotIn("package_sha256", released)
        for prefix, source in (("first", first), ("second", second)):
            package_path = f"evidence/moved/{prefix}.json"
            package_sha256 = digest(built / package_path)
            self.assertEqual(released[f"{prefix}_path"], package_path)
            self.assertEqual(released[f"{prefix}_sha256"], package_sha256)
            self.assertEqual(released[f"{prefix}_source_sha256"], digest(source))
            self.assertEqual(released[f"{prefix}_package_sha256"], package_sha256)
        self.assertTrue(validate_package(built)["passed"])

    def test_csv_graph_keeps_source_hash_and_links_with_package_hash(self) -> None:
        graph = self.source / "graph"
        graph.mkdir()
        child = graph / "value.json"
        child.write_text(
            json.dumps({"private_run_path": "/data/private/formal/run"}) + "\n",
            encoding="utf-8",
        )
        index = graph / "index.csv"
        with index.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["artifact_path", "sha256"], lineterminator="\n"
            )
            writer.writeheader()
            writer.writerow({"artifact_path": "graph/value.json", "sha256": digest(child)})

        direct_child = self.source / "direct_child.json"
        direct_child.write_text(
            json.dumps({"private_run_path": "/data/private/direct"}) + "\n",
            encoding="utf-8",
        )
        direct_links = self.source / "direct_links.csv"
        with direct_links.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["artifact_path", "artifact_sha256"],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow(
                {"artifact_path": "direct_child.json", "artifact_sha256": "3" * 64}
            )

        self.write_selection(
            {
                "package_name": "core-results",
                "release_mode": "draft",
                "pending_experiments": ["48_llama1b_10b_multibudget"],
                "direct_files": [
                    self.direct_entry("direct_child.json", "evidence/direct/child.json"),
                    self.direct_entry("direct_links.csv", "evidence/direct/links.csv"),
                ],
                "csv_graph_dependencies": [
                    {
                        "graph_id": "dual_hash_graph",
                        "source_root": "results",
                        "root_csv": {
                            "source": "graph/index.csv",
                            "source_sha256": digest(index),
                        },
                        "path_columns": ["artifact_path"],
                        "hash_columns": {"artifact_path": "sha256"},
                        "destination_root": "evidence/dependency_graph",
                        "relative_to": "source_root",
                        "experiment_id": "38",
                        **statuses(),
                    }
                ],
            }
        )

        built = build_package(
            self.selection, {"results": self.source}, self.output, mode="draft"
        )
        released_child = built / "evidence/dependency_graph/graph/value.json"
        released_index = built / "evidence/dependency_graph/graph/index.csv"
        with released_index.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        child_package_sha256 = digest(released_child)
        self.assertNotEqual(child_package_sha256, digest(child))
        self.assertEqual(
            row["artifact_path"], "evidence/dependency_graph/graph/value.json"
        )
        self.assertEqual(row["source_sha256"], digest(child))
        self.assertEqual(row["package_sha256"], child_package_sha256)
        self.assertEqual(row["sha256"], child_package_sha256)

        direct_child_package_sha256 = digest(built / "evidence/direct/child.json")
        with (built / "evidence/direct/links.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            direct_rows = list(csv.DictReader(handle))
        self.assertEqual(len(direct_rows), 1)
        direct_row = direct_rows[0]
        self.assertEqual(direct_row["artifact_path"], "evidence/direct/child.json")
        self.assertEqual(direct_row["artifact_sha256"], direct_child_package_sha256)
        self.assertEqual(direct_row["source_sha256"], digest(direct_child))
        self.assertEqual(direct_row["package_sha256"], direct_child_package_sha256)
        self.assertTrue(validate_package(built)["passed"])

    def test_json_link_bytes_follow_packaged_child_and_tampering_fails_closed(self) -> None:
        child = self.source / "sized_child.json"
        child.write_text(
            json.dumps({"private_run_path": "/data/private/formal/run"}) + "\n",
            encoding="utf-8",
        )
        parent = self.source / "sized_parent.json"
        parent.write_text(
            json.dumps(
                {
                    "artifact": {
                        "path": "sized_child.json",
                        "sha256": digest(child),
                        "bytes": child.stat().st_size,
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.write_selection(
            {
                "package_name": "core-results",
                "release_mode": "draft",
                "pending_experiments": ["48_llama1b_10b_multibudget"],
                "direct_files": [
                    self.direct_entry("sized_child.json", "evidence/sized_child.json"),
                    self.direct_entry("sized_parent.json", "evidence/sized_parent.json"),
                ],
            }
        )

        built = build_package(
            self.selection, {"results": self.source}, self.output, mode="draft"
        )
        packaged_child = built / "evidence/sized_child.json"
        source_bytes = child.stat().st_size
        package_bytes = packaged_child.stat().st_size
        self.assertNotEqual(source_bytes, package_bytes)
        released = json.loads(
            (built / "evidence/sized_parent.json").read_text(encoding="utf-8")
        )["artifact"]
        self.assertEqual(released["path"], "evidence/sized_child.json")
        self.assertEqual(released["bytes"], package_bytes)
        self.assertEqual(released["source_bytes"], source_bytes)
        self.assertEqual(released["package_bytes"], package_bytes)
        self.assertTrue(validate_package(built)["passed"])

        released["bytes"] = package_bytes + 1
        (built / "evidence/sized_parent.json").write_text(
            json.dumps({"artifact": released}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        resign_artifact(built, "evidence/sized_parent.json")
        report = validate_package(built, raise_on_error=False)
        self.assertFalse(report["passed"])
        self.assertIn("internal package hash link mismatch", report["errors"][0])

    def test_csv_and_tsv_sizes_follow_packaged_child_and_tampering_fails_closed(
        self,
    ) -> None:
        child = self.source / "table_child.json"
        child.write_text(
            json.dumps({"private_run_path": "/data/private/formal/run"}) + "\n",
            encoding="utf-8",
        )
        direct_files = [
            self.direct_entry("table_child.json", "evidence/table_child.json")
        ]
        for suffix, delimiter in (("csv", ","), ("tsv", "\t")):
            table = self.source / f"size_index.{suffix}"
            with table.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["relative_path", "sha256", "size_bytes"],
                    delimiter=delimiter,
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "relative_path": "table_child.json",
                        "sha256": digest(child),
                        "size_bytes": child.stat().st_size,
                    }
                )
            direct_files.append(
                self.direct_entry(table.name, f"evidence/{table.name}")
            )
        self.write_selection(
            {
                "package_name": "core-results",
                "release_mode": "draft",
                "pending_experiments": ["48_llama1b_10b_multibudget"],
                "direct_files": direct_files,
            }
        )

        built = build_package(
            self.selection, {"results": self.source}, self.output, mode="draft"
        )
        packaged_child = built / "evidence/table_child.json"
        source_bytes = child.stat().st_size
        package_bytes = packaged_child.stat().st_size
        self.assertNotEqual(source_bytes, package_bytes)
        for suffix, delimiter in (("csv", ","), ("tsv", "\t")):
            with (built / f"evidence/size_index.{suffix}").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle, delimiter=delimiter))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["relative_path"], "evidence/table_child.json")
            self.assertEqual(int(row["size_bytes"]), package_bytes)
            self.assertEqual(int(row["source_bytes"]), source_bytes)
            self.assertEqual(int(row["package_bytes"]), package_bytes)
        self.assertTrue(validate_package(built)["passed"])

        tsv_path = built / "evidence/size_index.tsv"
        with tsv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        rows[0]["size_bytes"] = str(package_bytes + 1)
        with tsv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        resign_artifact(built, "evidence/size_index.tsv")
        report = validate_package(built, raise_on_error=False)
        self.assertFalse(report["passed"])
        self.assertIn("internal package hash link mismatch", report["errors"][0])

    def test_cyclic_internal_hash_references_fail_closed(self) -> None:
        first = self.source / "cycle_first.json"
        second = self.source / "cycle_second.json"
        first.write_text(
            json.dumps({"artifact": {"path": "cycle_second.json", "sha256": "1" * 64}})
            + "\n",
            encoding="utf-8",
        )
        second.write_text(
            json.dumps({"artifact": {"path": "cycle_first.json", "sha256": digest(first)}})
            + "\n",
            encoding="utf-8",
        )
        self.write_selection(
            {
                "package_name": "core-results",
                "release_mode": "draft",
                "pending_experiments": ["48_llama1b_10b_multibudget"],
                "direct_files": [
                    self.direct_entry("cycle_first.json", "evidence/cycle_first.json"),
                    self.direct_entry("cycle_second.json", "evidence/cycle_second.json"),
                ],
            }
        )

        with self.assertRaisesRegex(
            BuildError, "cyclic internal package hash references"
        ):
            build_package(
                self.selection, {"results": self.source}, self.output, mode="draft"
            )


if __name__ == "__main__":
    unittest.main()
