from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
WINDOWS_ABSOLUTE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:(?:[\\/]|\\\\)")
REMOTE_DATA_ROOT = re.compile(
    r"(?<![A-Za-z0-9_}$>])/data/[A-Za-z][A-Za-z0-9._-]*[0-9]{6,}"
)
USER_HOME = re.compile(r"(?i)(?:[A-Za-z]:[\\/]+Users[\\/]+|/home/)[^/\\\s]+")
IPV4 = re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
WANDB_PUBLIC_URL = re.compile(r"https?://(?:www\.)?wandb\.ai/", re.IGNORECASE)
CONTAINER_HOST = re.compile(r"\bapp-[a-z0-9-]{12,}\b", re.IGNORECASE)
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReleaseIntegrityTests(unittest.TestCase):
    def test_all_python_sources_parse(self) -> None:
        for path in ROOT.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_catalog_paths_exist(self) -> None:
        catalog = json.loads(
            (ROOT / "experiments" / "catalog.json").read_text(encoding="utf-8")
        )
        expected_code_directories = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "scripts").iterdir()
            if path.is_dir()
            and (
                re.fullmatch(r"\d+[a-z]?_.+", path.name)
                or path.name == "mdp_refresh_streaming"
            )
        }
        self.assertEqual(
            {row["code_directory"] for row in catalog["experiments"]},
            expected_code_directories,
        )
        for row in catalog["experiments"]:
            with self.subTest(experiment=row["experiment_id"]):
                self.assertTrue((ROOT / row["code_directory"]).is_dir())
                for entrypoint in row["entrypoints"].values():
                    self.assertTrue((ROOT / entrypoint["path"]).is_file())

    def test_experiment_catalog_is_generated_current(self) -> None:
        generator_path = ROOT / "reproducibility" / "build_catalog.py"
        spec = importlib.util.spec_from_file_location(
            "release_build_catalog", generator_path
        )
        assert spec and spec.loader
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)

        generated: list[dict] = []
        expected_ids: list[str] = []
        for experiment_id, script_dir in generator.script_experiments():
            metadata = generator.build_metadata(experiment_id, script_dir)
            generated.append(metadata)
            expected_ids.append(experiment_id)

            metadata_path = ROOT / "experiments" / experiment_id / "metadata.json"
            with self.subTest(experiment=experiment_id):
                self.assertTrue(metadata_path.is_file(), metadata_path)
                self.assertEqual(
                    json.loads(metadata_path.read_text(encoding="utf-8")),
                    metadata,
                )
                # ``Path.write_text`` uses the host newline convention in the
                # generator. Compare the exact generated text after Python's
                # universal-newline normalization so this guard is portable
                # between Windows release assembly and POSIX CI.
                expected_text = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
                self.assertEqual(
                    metadata_path.read_text(encoding="utf-8"), expected_text
                )

        checked_in_ids = sorted(
            path.parent.name
            for path in (ROOT / "experiments").glob("*/metadata.json")
        )
        self.assertEqual(checked_in_ids, sorted(expected_ids))

        expected_catalog = {"schema_version": 1, "experiments": generated}
        catalog_path = ROOT / "experiments" / "catalog.json"
        self.assertEqual(
            json.loads(catalog_path.read_text(encoding="utf-8")),
            expected_catalog,
        )
        expected_catalog_text = (
            json.dumps(expected_catalog, indent=2, sort_keys=True) + "\n"
        )
        self.assertEqual(
            catalog_path.read_text(encoding="utf-8"), expected_catalog_text
        )

    def test_core_results_selection_anchors_current_public_snapshots(self) -> None:
        selection = json.loads(
            (ROOT / "reproducibility" / "core_results_selection.json").read_text(
                encoding="utf-8"
            )
        )
        direct_by_id = {row["id"]: row for row in selection["direct_files"]}
        expected = {
            "public_experiment_catalog": ROOT / "experiments" / "catalog.json",
            "accepted_result_anchors": (
                ROOT / "provenance" / "accepted_result_anchors.json"
            ),
        }
        self.assertTrue(set(expected).issubset(direct_by_id))
        for record_id, target in expected.items():
            with self.subTest(record_id=record_id):
                row = direct_by_id[record_id]
                self.assertEqual((ROOT / row["source"]).resolve(), target.resolve())
                self.assertEqual(row["source_sha256"], sha256(target))

    def test_sha256sum_files_are_current(self) -> None:
        for sums in ROOT.rglob("SHA256SUMS"):
            for line in sums.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                match = re.fullmatch(r"([0-9a-fA-F]{64})\s+(.+)", line)
                self.assertIsNotNone(match, f"bad SHA256SUMS line: {sums}: {line}")
                assert match is not None
                target = (sums.parent / match.group(2).strip()).resolve()
                self.assertTrue(target.is_file(), target)
                self.assertEqual(sha256(target), match.group(1).lower(), target)

    def test_public_source_inventory_is_current(self) -> None:
        inventory = json.loads(
            (ROOT / "provenance" / "source_copy_inventory.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(inventory["record_count"], len(inventory["records"]))
        role_roots = {
            "legacy_experiment_repo": ROOT / "scripts",
            "legacy_backend_repo": ROOT / "backends" / "nanogpt",
        }
        for row in inventory["records"]:
            with self.subTest(path=row["relative_path"], role=row["source_role"]):
                target = role_roots[row["source_role"]] / row["relative_path"]
                self.assertTrue(target.is_file(), target)
                self.assertEqual(sha256(target), row["public_sha256"], target)
                self.assertEqual(
                    row["packaging_changed"],
                    row["public_sha256"] != row["original_sha256"],
                )

    def test_public_command_inventory_is_current(self) -> None:
        inventory = json.loads(
            (ROOT / "provenance" / "legacy_command_inventory.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(inventory["record_count"], len(inventory["records"]))
        for row in inventory["records"]:
            with self.subTest(path=row["public_path"]):
                target = ROOT / row["public_path"]
                self.assertTrue(target.is_file(), target)
                self.assertEqual(sha256(target), row["public_sha256"], target)

    def test_public_contract_lineage_is_current(self) -> None:
        lineage = json.loads(
            (ROOT / "provenance" / "public_contract_lineage.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(lineage["record_count"], len(lineage["records"]))
        for row in lineage["records"]:
            with self.subTest(path=row["relative_path"]):
                target = ROOT / "scripts" / row["relative_path"]
                self.assertTrue(target.is_file(), target)
                self.assertEqual(sha256(target), row["public_sha256"], target)
                self.assertEqual(
                    row["packaging_changed"],
                    row["public_sha256"] != row["original_sha256"],
                )

    def test_submission_portable_snapshot_hashes_are_current(self) -> None:
        experiment = ROOT / "scripts" / "39_submission_efficiency_and_sensitivity"
        registry = json.loads(
            (experiment / "evidence_registry.json").read_text(encoding="utf-8")
        )
        for row in registry["required_files"]:
            with self.subTest(path=row["portable_path"]):
                target = experiment / "source_snapshot" / row["portable_path"]
                self.assertTrue(target.is_file(), target)
                self.assertEqual(sha256(target), row["portable_sha256"], target)

    def test_no_generated_python_cache(self) -> None:
        caches = [path for path in ROOT.rglob("__pycache__") if path.is_dir()]
        self.assertEqual(caches, [])
        self.assertEqual(list(ROOT.rglob("*.pyc")), [])

    def test_no_private_machine_identifiers(self) -> None:
        findings: list[str] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            relative = path.relative_to(ROOT).as_posix()
            if path == Path(__file__).resolve():
                continue
            checks = {
                "remote data root": REMOTE_DATA_ROOT,
                "user home": USER_HOME,
                "IPv4 address": IPV4,
                "public W&B URL": WANDB_PUBLIC_URL,
                "private container hostname": CONTAINER_HOST,
            }
            for label, pattern in checks.items():
                if pattern.search(text):
                    findings.append(f"{relative}: {label}")
            for match in EMAIL.finditer(text):
                if not match.group(0).lower().endswith(".invalid"):
                    findings.append(f"{relative}: email address")
            if WINDOWS_ABSOLUTE.search(text):
                findings.append(f"{relative}: Windows absolute path")
        self.assertEqual(findings, [])

    def test_launchers_do_not_construct_the_private_workspace_layout(self) -> None:
        findings: list[str] = []
        for root in (ROOT / "commands", ROOT / "scripts"):
            for path in root.rglob("*.sh"):
                text = path.read_text(encoding="utf-8")
                if re.search(r"experiment_csv[\\/]+[^$/{\\\s]+", text):
                    findings.append(
                        f"{path.relative_to(ROOT).as_posix()}: private result layout"
                    )
        self.assertEqual(findings, [])

    def test_public_path_module_resolves_release_layout(self) -> None:
        path = ROOT / "scripts" / "_shared" / "project_paths.py"
        old_repo = os.environ.get("SNM_REPO")
        old_results = os.environ.get("SNM_RESULTS_ROOT")
        try:
            os.environ["SNM_REPO"] = str(ROOT)
            os.environ["SNM_RESULTS_ROOT"] = str(ROOT / "runs")
            spec = importlib.util.spec_from_file_location("public_project_paths", path)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.ARTIFACT_ROOT, ROOT)
            self.assertEqual(module.SOURCE_REPO, ROOT / "backends" / "nanogpt")
            self.assertEqual(module.EXPERIMENT_DATA_ROOT, ROOT / "runs")
            self.assertEqual(module.EXPERIMENT_RESULTS_ROOT, ROOT / "runs")
        finally:
            if old_repo is None:
                os.environ.pop("SNM_REPO", None)
            else:
                os.environ["SNM_REPO"] = old_repo
            if old_results is None:
                os.environ.pop("SNM_RESULTS_ROOT", None)
            else:
                os.environ["SNM_RESULTS_ROOT"] = old_results

    def test_public_path_module_resolves_after_directory_rename(self) -> None:
        source = ROOT / "scripts" / "_shared" / "project_paths.py"
        names = (
            "SNM_REPO",
            "SNM_ARTIFACT_ROOT",
            "SNM_WORKSPACE_ROOT",
            "SNM_RESULTS_ROOT",
            "SELECTIVE_NEWTON_MUON_SOURCE_REPO",
            "SELECTIVE_NEWTON_MUON_REPO",
        )
        previous = {name: os.environ.pop(name, None) for name in names}
        try:
            with tempfile.TemporaryDirectory() as temporary:
                relocated = Path(temporary) / "arbitrary-release-directory-name"
                module_path = relocated / "scripts" / "_shared" / "project_paths.py"
                module_path.parent.mkdir(parents=True)
                (relocated / "backends" / "nanogpt").mkdir(parents=True)
                (relocated / "pyproject.toml").write_text(
                    "[project]\nname = 'portable-layout-test'\n",
                    encoding="utf-8",
                )
                module_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
                spec = importlib.util.spec_from_file_location(
                    "relocated_public_project_paths", module_path
                )
                assert spec and spec.loader
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertEqual(module.ARTIFACT_ROOT, relocated.resolve())
                self.assertEqual(
                    module.SOURCE_REPO,
                    relocated.resolve() / "backends" / "nanogpt",
                )
                self.assertEqual(
                    module.EXPERIMENT_RESULTS_ROOT,
                    relocated.resolve() / "runs",
                )
        finally:
            for name, value in previous.items():
                if value is not None:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
