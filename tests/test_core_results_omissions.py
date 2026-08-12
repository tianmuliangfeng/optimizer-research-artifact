from __future__ import annotations

import csv
import hashlib
import json
import shutil
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


def statuses() -> dict[str, str]:
    return {
        "integrity_status": "accepted",
        "scientific_status": "supported_with_scope",
        "claim_eligibility": "claim_eligible",
        "paper_role": "robustness",
    }


class CoreResultsOmissionTests(unittest.TestCase):
    experiment_id = "29_r1_depth_kmode"
    anchor_package_path = (
        "results/03_robustness_and_controls/ex29_r1_depth_kmode/"
        "analysis_20260809_formal/input_manifest.csv"
    )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "accepted-results"
        self.archive = (
            self.source
            / "29_r1_depth_kmode/source_bundle/29_r1_depth_kmode_20260809.zip"
        )
        self.raw = (
            self.source
            / "29_r1_depth_kmode/analysis_20260809_formal/raw_wandb_exports/"
            "wandb_export_2026-08-09T13_17_37.904+08_00.csv"
        )
        self.anchor = (
            self.source
            / "29_r1_depth_kmode/analysis_20260809_formal/input_manifest.csv"
        )
        self.archive.parent.mkdir(parents=True)
        self.raw.parent.mkdir(parents=True)
        self.archive.write_bytes(b"PK\x03\x04accepted EX29 full result archive\n")
        self.raw.write_text("Step,val/loss\n100,3.75\n", encoding="utf-8")
        self._write_anchor()
        self.selection = self.root / "selection.json"
        self.output = self.root / "core-results"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_anchor(
        self,
        *,
        include_archive: bool = True,
        raw_kind: str = "wandb_export",
        raw_file_name: str | None = None,
    ) -> None:
        rows: list[dict[str, object]] = []
        if include_archive:
            rows.append(
                {
                    "kind": "remote_bundle_zip",
                    # Deliberately differs from the retained source basename.  The
                    # anchor contract is the logical file name, not a path guess.
                    "file_name": "29_r1_depth_kmode.zip",
                    "sha256": digest(self.archive),
                    "size_bytes": self.archive.stat().st_size,
                }
            )
        rows.append(
            {
                "kind": raw_kind,
                "file_name": raw_file_name or self.raw.name,
                "sha256": digest(self.raw),
                "size_bytes": self.raw.stat().st_size,
            }
        )
        self.anchor.parent.mkdir(parents=True, exist_ok=True)
        with self.anchor.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["kind", "file_name", "sha256", "size_bytes"],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    def _omission(
        self,
        *,
        omission_id: str,
        artifact_class: str,
        logical_name: str,
        source: str,
        source_path: Path,
        anchor_kind: str,
    ) -> dict[str, object]:
        return {
            "omission_id": omission_id,
            "experiment_ids": [self.experiment_id],
            "artifact_class": artifact_class,
            "logical_name": logical_name,
            "source_root": "results",
            "source": source,
            "source_sha256": digest(source_path),
            "source_bytes": source_path.stat().st_size,
            "reason": "hash-anchored raw input intentionally excluded from the compact package",
            "anchor_package_path": self.anchor_package_path,
            "anchor_row": {"kind": anchor_kind, "file_name": logical_name},
        }

    def _payload(self) -> dict[str, object]:
        anchor_source = "29_r1_depth_kmode/analysis_20260809_formal/input_manifest.csv"
        archive_source = (
            "29_r1_depth_kmode/source_bundle/29_r1_depth_kmode_20260809.zip"
        )
        raw_source = (
            "29_r1_depth_kmode/analysis_20260809_formal/raw_wandb_exports/"
            + self.raw.name
        )
        return {
            "schema_version": 1,
            "package_name": "core-results",
            "release_mode": "draft",
            "pending_experiments": ["48_llama1b_10b_multibudget"],
            "included_experiments": [self.experiment_id],
            "direct_files": [
                {
                    "evidence_id": "ex29_input_manifest",
                    "experiment_id": self.experiment_id,
                    "source_root": "results",
                    "source": anchor_source,
                    "package_path": self.anchor_package_path,
                    "source_sha256": digest(self.anchor),
                    "workstream": "robustness",
                    **statuses(),
                }
            ],
            "omissions": [
                self._omission(
                    omission_id="ex29_full_result_archive",
                    artifact_class="full_result_archive",
                    logical_name="29_r1_depth_kmode.zip",
                    source=archive_source,
                    source_path=self.archive,
                    anchor_kind="remote_bundle_zip",
                ),
                self._omission(
                    omission_id="ex29_raw_wandb_val_loss",
                    artifact_class="raw_wandb_export",
                    logical_name=self.raw.name,
                    source=raw_source,
                    source_path=self.raw,
                    anchor_kind="wandb_export",
                ),
            ],
        }

    def _write_selection(self, payload: dict[str, object]) -> None:
        self.selection.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _build(self, payload: dict[str, object] | None = None) -> Path:
        self._write_selection(payload or self._payload())
        return build_package(
            self.selection, {"results": self.source}, self.output, mode="draft"
        )

    def test_hash_anchored_omissions_are_relocatable_and_not_bundled(self) -> None:
        built = self._build()
        ledger_path = built / "provenance/omission_ledger.json"
        self.assertTrue(ledger_path.is_file())
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        omissions = ledger["omissions"]
        self.assertEqual(len(omissions), 2)
        by_id = {row["omission_id"]: row for row in omissions}
        self.assertEqual(
            by_id["ex29_full_result_archive"]["logical_name"],
            "29_r1_depth_kmode.zip",
        )
        self.assertTrue(
            by_id["ex29_full_result_archive"]["source_relpath"].endswith(
                "29_r1_depth_kmode_20260809.zip"
            )
        )
        self.assertNotEqual(
            Path(by_id["ex29_full_result_archive"]["source_relpath"]).name,
            by_id["ex29_full_result_archive"]["logical_name"],
        )

        release = json.loads(
            (built / "release_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(release["omission_count"], 2)
        self.assertEqual(
            release["omission_ledger"], "provenance/omission_ledger.json"
        )
        artifact_manifest = json.loads(
            (built / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        selected_sources = {
            (row["source_alias"], row["source_relpath"])
            for row in artifact_manifest["artifacts"]
        }
        for omission in self._payload()["omissions"]:  # type: ignore[index]
            self.assertNotIn(
                (omission["source_root"], omission["source"]), selected_sources
            )
        self.assertFalse(any(path.name == self.raw.name for path in built.rglob("*")))
        self.assertFalse(any(path.name == self.archive.name for path in built.rglob("*")))
        self.assertTrue(validate_package(built)["passed"])

        moved = self.root / "relocated-core-results"
        shutil.move(str(built), str(moved))
        self.assertTrue(validate_package(moved)["passed"])

    def test_wrong_omitted_source_hash_or_size_fails_closed(self) -> None:
        for field, bad_value in (("source_sha256", "0" * 64), ("source_bytes", 1)):
            with self.subTest(field=field):
                if self.output.exists():
                    shutil.rmtree(self.output)
                payload = self._payload()
                payload["omissions"][0][field] = bad_value  # type: ignore[index]
                self._write_selection(payload)
                try:
                    with self.assertRaises(BuildError):
                        build_package(
                            self.selection,
                            {"results": self.source},
                            self.output,
                            mode="draft",
                        )
                finally:
                    if self.output.exists():
                        shutil.rmtree(self.output)

    def test_missing_or_inconsistent_anchor_row_fails_closed(self) -> None:
        cases = (
            {"include_archive": False},
            {"raw_kind": "wrong_kind"},
            {"raw_file_name": "wrong-logical-name.csv"},
        )
        for mutation in cases:
            with self.subTest(mutation=mutation):
                if self.output.exists():
                    shutil.rmtree(self.output)
                self._write_anchor(**mutation)
                payload = self._payload()
                # Re-anchor the selected CSV after deliberately changing it.  A
                # failure must come from omission/anchor semantics, not source drift.
                payload["direct_files"][0]["source_sha256"] = digest(self.anchor)  # type: ignore[index]
                self._write_selection(payload)
                try:
                    with self.assertRaises(BuildError):
                        build_package(
                            self.selection,
                            {"results": self.source},
                            self.output,
                            mode="draft",
                        )
                finally:
                    if self.output.exists():
                        shutil.rmtree(self.output)
        self._write_anchor()

    def test_same_source_cannot_be_selected_and_omitted(self) -> None:
        payload = self._payload()
        raw_omission = payload["omissions"][1]  # type: ignore[index]
        payload["direct_files"].append(  # type: ignore[union-attr]
            {
                "evidence_id": "forbidden_selected_raw",
                "experiment_id": self.experiment_id,
                "source_root": "results",
                "source": raw_omission["source"],
                "package_path": "results/forbidden/raw.csv",
                "source_sha256": raw_omission["source_sha256"],
                **statuses(),
            }
        )
        self._write_selection(payload)
        with self.assertRaises(BuildError):
            build_package(
                self.selection, {"results": self.source}, self.output, mode="draft"
            )

    def test_tampered_ledger_fails_even_after_resigning_sha256sums(self) -> None:
        built = self._build()
        ledger_path = built / "provenance/omission_ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["omissions"][0]["reason"] = "tampered-but-re-signed"
        ledger_path.write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        refresh_package_checksums(built)
        report = validate_package(built, raise_on_error=False)
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("omission" in message.lower() for message in report["errors"]),
            report["errors"],
        )


if __name__ == "__main__":
    unittest.main()
