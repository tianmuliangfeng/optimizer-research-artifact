from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "reproducibility" / "reproduce.py"
)
SPEC = importlib.util.spec_from_file_location("artifact_reproduce", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)

CATALOG_MODULE_PATH = MODULE_PATH.with_name("build_catalog.py")
CATALOG_SPEC = importlib.util.spec_from_file_location(
    "artifact_build_catalog", CATALOG_MODULE_PATH
)
assert CATALOG_SPEC is not None and CATALOG_SPEC.loader is not None
B = importlib.util.module_from_spec(CATALOG_SPEC)
CATALOG_SPEC.loader.exec_module(B)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReproductionFrameworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.release = self.root / "release"
        self.experiments = self.release / "experiments"
        self.results = self.root / "results"
        self.experiments.mkdir(parents=True)
        self.results.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_experiment(
        self,
        experiment_id: str = "43_record28",
        *,
        metadata: dict | None = None,
        commands: tuple[str, ...] = ("run.py",),
    ) -> Path:
        experiment = self.experiments / experiment_id
        experiment.mkdir(parents=True)
        code_root = self.release / "scripts" / experiment_id
        code_root.mkdir(parents=True)
        (code_root / "frozen.txt").write_text("code\n", encoding="utf-8")
        command_root = self.release / "commands" / experiment_id
        command_root.mkdir(parents=True)
        for name in commands:
            (command_root / name).write_text(
                "from pathlib import Path\n"
                "import os\n"
                "marker = os.environ.get('TEST_MARKER')\n"
                "if marker:\n"
                "    Path(marker).write_text('executed', encoding='utf-8')\n",
                encoding="utf-8",
            )
        if metadata is None:
            metadata = {
                "experiment_id": experiment_id,
                "status": "completed",
                "code_directory": f"scripts/{experiment_id}",
                "entrypoints": {
                    "main": {"path": f"commands/{experiment_id}/run.py"}
                },
            }
        write_json(experiment / "metadata.json", metadata)
        return experiment

    def call_main(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = R.main(["--release-root", str(self.release), *arguments])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_list_discovers_valid_and_placeholder_experiments(self) -> None:
        self.make_experiment()
        (self.experiments / "05_future").mkdir()
        code, stdout, stderr = self.call_main("list")
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        rows = {row["experiment_id"]: row for row in payload["experiments"]}
        self.assertTrue(rows["43_record28"]["metadata_valid"])
        self.assertEqual(rows["43_record28"]["entrypoints"], ["main"])
        self.assertEqual(rows["05_future"]["status"], "planned_placeholder")
        self.assertFalse(rows["05_future"]["metadata_present"])

    def test_list_and_inspect_merge_native_modes_from_all_entrypoints(self) -> None:
        experiment_id = "48_llama1b_10b_multibudget"
        metadata = {
            "experiment_id": experiment_id,
            "status": "implemented",
            "code_directory": f"scripts/{experiment_id}",
            "entrypoints": {
                "control": {
                    "path": f"commands/{experiment_id}/control.py",
                    "native_modes": {
                        "resume": {"args": ["resume"]},
                        "native-verify": {"args": ["verify"]},
                    },
                },
                "reproduce": {
                    "path": f"commands/{experiment_id}/reproduce.py",
                    "native_modes": {"reproduce": []},
                },
            },
        }
        self.make_experiment(
            experiment_id,
            metadata=metadata,
            commands=("control.py", "reproduce.py"),
        )

        code, stdout, stderr = self.call_main("list")
        self.assertEqual(code, 0, stderr)
        row = json.loads(stdout)["experiments"][0]
        self.assertEqual(
            row["native_modes"], ["native-verify", "reproduce", "resume"]
        )

        inspected = R.inspect_experiment(self.release, experiment_id)
        self.assertEqual(
            inspected["native_modes"], ["native-verify", "reproduce", "resume"]
        )

    def test_inspect_has_deterministic_tree_hash(self) -> None:
        self.make_experiment()
        first = R.inspect_experiment(self.release, "43_record28")
        second = R.inspect_experiment(self.release, "43_record28")
        self.assertEqual(first["source_tree_sha256"], second["source_tree_sha256"])
        self.assertRegex(first["source_tree_sha256"], r"^[0-9a-f]{64}$")
        self.assertGreaterEqual(first["source_file_count"], 3)

    def test_verify_is_read_only_and_checks_file_sha256(self) -> None:
        self.make_experiment()
        run = self.results / "run_ok"
        snapshot = run / "source_snapshot"
        snapshot.mkdir(parents=True)
        frozen = snapshot / "worker.py"
        frozen.write_text("print('ok')\n", encoding="utf-8")
        write_json(
            snapshot / "source_snapshot_manifest.json",
            {"passed": True, "file_sha256": {"worker.py": digest(frozen)}},
        )
        write_json(run / "nested" / "result.json", {"passed": True})
        before = {
            path.relative_to(run).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns, digest(path))
            for path in R.iter_files(run)
        }
        payload = R.verify_run(run, self.results)
        after = {
            path.relative_to(run).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns, digest(path))
            for path in R.iter_files(run)
        }
        self.assertTrue(payload["passed"], payload)
        self.assertEqual(payload["json_file_count"], 2)
        self.assertEqual(payload["source_snapshot_manifest_count"], 1)
        self.assertEqual(before, after)

    def test_verify_detects_tamper_and_invalid_json(self) -> None:
        self.make_experiment()
        run = self.results / "run_bad"
        snapshot = run / "source_snapshot"
        snapshot.mkdir(parents=True)
        frozen = snapshot / "worker.py"
        frozen.write_text("changed\n", encoding="utf-8")
        write_json(
            snapshot / "source_snapshot_manifest.json",
            {"file_sha256": {"worker.py": "0" * 64}},
        )
        (run / "broken.json").write_text("{", encoding="utf-8")
        payload = R.verify_run(run, self.results)
        self.assertFalse(payload["passed"])
        self.assertFalse(payload["checks"]["all_json_parse"])
        self.assertFalse(payload["checks"]["source_snapshot_manifests"])

    def test_verify_rejects_an_empty_directory(self) -> None:
        self.make_experiment()
        run = self.results / "empty_run"
        run.mkdir()
        payload = R.verify_run(run, self.results)
        self.assertFalse(payload["passed"])
        self.assertFalse(payload["checks"]["json_present"])

    def test_verify_supports_richer_files_mapping(self) -> None:
        self.make_experiment()
        run = self.results / "run_files"
        snapshot = run / "source_snapshot"
        snapshot.mkdir(parents=True)
        frozen = snapshot / "worker.py"
        frozen.write_text("x = 1\n", encoding="utf-8")
        write_json(
            snapshot / "source_snapshot_manifest.json",
            {
                "files": {
                    "worker.py": {
                        "bytes": frozen.stat().st_size,
                        "sha256": digest(frozen),
                    }
                }
            },
        )
        self.assertTrue(R.verify_run(run, self.results)["passed"])

    def test_run_dir_must_be_strictly_inside_results_root(self) -> None:
        self.make_experiment()
        outside = self.root / "outside"
        outside.mkdir()
        with self.assertRaisesRegex(R.ReproductionError, "strict child"):
            R.verify_run(outside, self.results)
        with self.assertRaisesRegex(R.ReproductionError, "strict child"):
            R.verify_run(self.results, self.results)

    def test_reproduce_plan_is_deterministic_and_receipt_gates_execution(self) -> None:
        marker = self.root / "marker.txt"
        metadata = {
            "experiment_id": "48_long",
            "status": "ready",
            "entrypoints": {
                "main": {
                    "path": "commands/48_long/run.py",
                    "env": {"TEST_MARKER": str(marker)},
                }
            },
        }
        self.make_experiment("48_long", metadata=metadata)
        plan = R.build_action_plan(self.release, "48_long", "reproduce")
        repeated = R.build_action_plan(self.release, "48_long", "reproduce")
        self.assertEqual(plan["plan_sha256"], repeated["plan_sha256"])
        self.assertRegex(plan["entrypoint_sha256"], r"^[0-9a-f]{64}$")
        self.assertFalse(marker.exists())

        with self.assertRaisesRegex(R.ReproductionError, "requires --receipt"):
            R.execute_plan(plan, None)
        with self.assertRaisesRegex(R.ReproductionError, "receipt mismatch"):
            R.execute_plan(plan, "0" * 64)
        self.assertFalse(marker.exists())
        self.assertEqual(R.execute_plan(plan, plan["plan_sha256"]), 0)
        self.assertEqual(marker.read_text(encoding="utf-8"), "executed")

    def test_cli_execute_without_receipt_is_rejected(self) -> None:
        self.make_experiment()
        code, stdout, stderr = self.call_main("reproduce", "43_record28", "--execute")
        self.assertEqual(code, 2)
        self.assertRegex(json.loads(stdout)["plan_sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("--receipt", json.loads(stderr)["error"])

    def test_multiple_entrypoints_require_explicit_selection(self) -> None:
        metadata = {
            "experiment_id": "20_multi",
            "status": "ready",
            "entrypoints": {
                "capacity": "commands/20_multi/capacity.py",
                "formal": "commands/20_multi/formal.py",
            },
        }
        self.make_experiment(
            "20_multi",
            metadata=metadata,
            commands=("capacity.py", "formal.py"),
        )
        with self.assertRaisesRegex(R.ReproductionError, "require --entrypoint"):
            R.build_action_plan(self.release, "20_multi", "reproduce")
        plan = R.build_action_plan(
            self.release, "20_multi", "reproduce", entrypoint_name="formal"
        )
        self.assertEqual(plan["entrypoint"], "formal")

    def test_metadata_default_entrypoint_resolves_multiple_providers(self) -> None:
        metadata = {
            "experiment_id": "20_default",
            "status": "ready",
            "default_entrypoint": "formal",
            "entrypoints": {
                "capacity": {
                    "path": "commands/20_default/capacity.py",
                    "native_modes": {"reproduce": []},
                },
                "formal": {
                    "path": "commands/20_default/formal.py",
                    "native_modes": {"reproduce": []},
                },
            },
        }
        self.make_experiment(
            "20_default",
            metadata=metadata,
            commands=("capacity.py", "formal.py"),
        )
        plan = R.build_action_plan(self.release, "20_default", "reproduce")
        self.assertEqual(plan["entrypoint"], "formal")

    def test_required_user_arguments_are_plannable_but_gate_execution(self) -> None:
        metadata = {
            "experiment_id": "17_required",
            "status": "ready",
            "default_entrypoint": "main",
            "entrypoints": {
                "main": {
                    "path": "commands/17_required/run.py",
                    "native_modes": {"reproduce": []},
                    "required_user_arguments": [
                        {"all_of": ["--official-repo", "--python-exe"]}
                    ],
                }
            },
        }
        self.make_experiment("17_required", metadata=metadata)
        incomplete = R.build_action_plan(
            self.release, "17_required", "reproduce"
        )
        self.assertTrue(incomplete["missing_required_user_arguments"])
        with self.assertRaisesRegex(R.ReproductionError, "receipt-bound --arg"):
            R.execute_plan(incomplete, incomplete["plan_sha256"])

        valueless = R.build_action_plan(
            self.release,
            "17_required",
            "reproduce",
            user_args=["--official-repo", "--python-exe"],
        )
        self.assertTrue(valueless["missing_required_user_arguments"])

        complete = R.build_action_plan(
            self.release,
            "17_required",
            "reproduce",
            user_args=[
                "--official-repo",
                "/official",
                "--python-exe",
                "/training/python",
            ],
        )
        self.assertEqual(complete["missing_required_user_arguments"], [])
        self.assertNotEqual(incomplete["plan_sha256"], complete["plan_sha256"])

    def test_cli_verify_binds_experiment_path_and_rejects_planned(self) -> None:
        self.make_experiment(
            "43_record28",
            metadata={
                "experiment_id": "43_record28",
                "status": "completed",
                "code_directory": "scripts/43_record28",
                "entrypoints": {"main": "commands/43_record28/run.py"},
                "legacy_result_roots": ["43_record28"],
                "reproducibility": {"source_freeze": "partial"},
            },
        )
        self.make_experiment(
            "48_long",
            metadata={
                "experiment_id": "48_long",
                "status": "completed",
                "code_directory": "scripts/48_long",
                "entrypoints": {"main": "commands/48_long/run.py"},
                "legacy_result_roots": ["48_long"],
                "reproducibility": {"source_freeze": "partial"},
            },
        )
        wrong_run = self.results / "48_long" / "run_1"
        write_json(wrong_run / "result.json", {"passed": True})
        code, stdout, stderr = self.call_main(
            "verify",
            "43_record28",
            "--results-root",
            str(self.results),
            "--run-dir",
            str(wrong_run),
        )
        self.assertEqual(code, 2, stderr)
        self.assertFalse(json.loads(stdout)["checks"]["experiment_lineage_bound"])

        correct_run = self.results / "43_record28" / "run_1"
        write_json(correct_run / "result.json", {"passed": True})
        code, stdout, stderr = self.call_main(
            "verify",
            "43_record28",
            "--results-root",
            str(self.results),
            "--run-dir",
            str(correct_run),
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(
            json.loads(stdout)["lineage"]["matched_legacy_result_root"],
            "43_record28",
        )

        self.make_experiment(
            "05_future",
            metadata={
                "experiment_id": "05_future",
                "status": "planned_not_implemented",
                "entrypoints": {},
                "legacy_result_roots": ["05_future"],
            },
        )
        code, stdout, stderr = self.call_main(
            "verify",
            "05_future",
            "--results-root",
            str(self.results),
            "--run-dir",
            str(correct_run),
        )
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("planned", json.loads(stderr)["error"])

    def test_verify_uses_path_alternative_and_nonvacuous_exact_snapshot(self) -> None:
        metadata = {
            "experiment_id": "48_sealed",
            "status": "completed",
            "code_directory": "scripts/48_sealed",
            "entrypoints": {"main": "commands/48_sealed/run.py"},
            "legacy_result_roots": ["48_sealed"],
            "reproducibility": {"source_freeze": "sealed_source_snapshot"},
        }
        self.make_experiment("48_sealed", metadata=metadata)
        run = self.results / "48_sealed" / "run_1"
        write_json(run / "result.json", {"passed": True})
        payload = R.verify_run(
            run,
            self.results,
            experiment_id="48_sealed",
            metadata=metadata,
        )
        self.assertTrue(payload["passed"], payload)
        self.assertFalse(payload["checks"]["source_snapshot_manifest_present"])
        self.assertFalse(payload["checks"]["source_snapshot_manifests"])

        snapshot = run / "source_snapshot"
        frozen = snapshot / "scripts" / "48_sealed" / "worker.py"
        frozen.parent.mkdir(parents=True)
        frozen.write_text("x = 1\n", encoding="utf-8")
        write_json(
            snapshot / "source_snapshot_manifest.json",
            {
                "passed": True,
                "files": {
                    "scripts/48_sealed/worker.py": {
                        "bytes": frozen.stat().st_size,
                        "sha256": digest(frozen),
                    }
                },
            },
        )
        payload = R.verify_run(
            run,
            self.results,
            experiment_id="48_sealed",
            metadata=metadata,
        )
        self.assertTrue(payload["passed"], payload)

        (snapshot / "unlisted.py").write_text("tamper\n", encoding="utf-8")
        payload = R.verify_run(
            run,
            self.results,
            experiment_id="48_sealed",
            metadata=metadata,
        )
        self.assertFalse(payload["passed"])
        self.assertIn(
            "snapshot inventory mismatch",
            " ".join(payload["source_snapshot_manifests"][0]["errors"]),
        )

    def test_environment_is_explicit_bound_and_receipt_detects_mutation(self) -> None:
        marker = self.root / "ambient-marker.txt"
        self.make_experiment("48_env")
        with mock.patch.dict(
            os.environ,
            {"TEST_MARKER": str(marker), "SNM_RESULTS_ROOT": "ambient"},
            clear=False,
        ):
            plan = R.build_action_plan(self.release, "48_env", "reproduce")
            self.assertNotIn("TEST_MARKER", plan["environment"])
            self.assertNotIn("SNM_RESULTS_ROOT", plan["environment"])
            self.assertEqual(R.execute_plan(plan, plan["plan_sha256"]), 0)
        self.assertFalse(marker.exists())

        first = R.build_action_plan(
            self.release,
            "48_env",
            "reproduce",
            env_overrides={"SNM_RESULTS_ROOT": "one"},
        )
        second = R.build_action_plan(
            self.release,
            "48_env",
            "reproduce",
            env_overrides={"SNM_RESULTS_ROOT": "two"},
        )
        self.assertNotEqual(first["plan_sha256"], second["plan_sha256"])
        with self.assertRaisesRegex(R.ReproductionError, "inject code"):
            R.build_action_plan(
                self.release,
                "48_env",
                "reproduce",
                env_overrides={"BASH_ENV": "payload.sh"},
            )

        mutated = dict(first)
        mutated["command"] = [*first["command"], "--unexpected"]
        with self.assertRaisesRegex(R.ReproductionError, "no longer match"):
            R.execute_plan(mutated, first["plan_sha256"])

    def test_required_resume_certificate_is_content_bound(self) -> None:
        metadata = {
            "experiment_id": "46_mdp05",
            "status": "completed",
            "entrypoints": {"main": "commands/46_mdp05/run.py"},
            "native_modes": {
                "resume": {
                    "entrypoint": "main",
                    "args": ["resume"],
                    "env": {"RUN_DIR": "{run_dir}"},
                    "required_env": ["MDP05_PILOT_CERTIFICATE"],
                    "required_env_files": ["MDP05_PILOT_CERTIFICATE"],
                }
            },
        }
        self.make_experiment("46_mdp05", metadata=metadata)
        run = self.results / "run"
        run.mkdir()
        with self.assertRaisesRegex(R.ReproductionError, "requires explicit --env"):
            R.build_action_plan(
                self.release,
                "46_mdp05",
                "resume",
                run_dir=run,
                results_root=self.results,
            )

        certificate = self.root / "pilot_precision_certificate.json"
        write_json(certificate, {"passed": True, "precision": "fp64"})
        first = R.build_action_plan(
            self.release,
            "46_mdp05",
            "resume",
            run_dir=run,
            results_root=self.results,
            env_overrides={"MDP05_PILOT_CERTIFICATE": str(certificate)},
        )
        self.assertEqual(len(first["external_input_files"]), 1)
        self.assertEqual(
            first["external_input_files"][0]["sha256"], digest(certificate)
        )
        write_json(certificate, {"passed": True, "precision": "changed"})
        second = R.build_action_plan(
            self.release,
            "46_mdp05",
            "resume",
            run_dir=run,
            results_root=self.results,
            env_overrides={"MDP05_PILOT_CERTIFICATE": str(certificate)},
        )
        self.assertNotEqual(first["plan_sha256"], second["plan_sha256"])

    def test_native_action_rejects_another_experiment_result_root(self) -> None:
        metadata = {
            "experiment_id": "47_geo",
            "status": "completed",
            "entrypoints": {"main": "commands/47_geo/run.py"},
            "legacy_result_roots": ["47_geo"],
            "native_modes": {
                "native-verify": {
                    "entrypoint": "main",
                    "args": ["verify"],
                    "env": {"RUN_DIR": "{run_dir}"},
                }
            },
        }
        self.make_experiment("47_geo", metadata=metadata)
        wrong_run = self.results / "48_long" / "run_1"
        wrong_run.mkdir(parents=True)
        with self.assertRaisesRegex(R.ReproductionError, "experiment result root"):
            R.build_action_plan(
                self.release,
                "47_geo",
                "native-verify",
                run_dir=wrong_run,
                results_root=self.results,
            )

    def test_catalog_archival_and_multi_entrypoint_contracts(self) -> None:
        mdp04 = B.build_metadata(
            "mdp04_refresh_streaming", B.SCRIPTS / "mdp_refresh_streaming"
        )
        self.assertFalse(mdp04["reproducibility"]["fresh_rerun"])
        self.assertFalse(mdp04["reproducibility"]["native_resume"])
        self.assertTrue(mdp04["reproducibility"]["native_verify"])
        self.assertNotIn("default_entrypoint", mdp04)
        mdp04_readme = B.experiment_readme(mdp04)
        self.assertNotIn("reproduce mdp04_refresh_streaming\n", mdp04_readme)
        self.assertIn("native-verify", mdp04_readme)

        mdp05 = B.build_metadata(
            "46_mdp05_confirmatory_update_shock",
            B.SCRIPTS / "46_mdp05_confirmatory_update_shock",
        )
        resume_spec = mdp05["entrypoints"][
            "command:20260804_ex46_mdp05_confirmatory_update_shock"
        ]["native_modes"]["resume"]
        self.assertEqual(
            resume_spec["required_env_files"], ["MDP05_PILOT_CERTIFICATE"]
        )

        multi = B.build_metadata(
            "20_llama_swiglu_1b", B.SCRIPTS / "20_llama_swiglu_1b"
        )
        self.assertNotIn("default_entrypoint", multi)
        multi_readme = B.experiment_readme(multi)
        self.assertIn("--entrypoint script:run_llama_swiglu_1b", multi_readme)

        llama17 = B.build_metadata(
            "17_llama_swiglu_validation",
            B.SCRIPTS / "17_llama_swiglu_validation",
        )
        self.assertTrue(llama17["reproducibility"]["fresh_rerun"])
        self.assertFalse(llama17["reproducibility"]["one_click_rerun"])
        llama17_readme = B.experiment_readme(llama17)
        self.assertIn("--arg=--official-repo", llama17_readme)

    def test_every_advertised_fresh_entrypoint_builds_a_complete_plan(self) -> None:
        for experiment_id, script_dir in B.script_experiments():
            metadata = B.build_metadata(experiment_id, script_dir)
            if not metadata["reproducibility"]["fresh_rerun"]:
                continue
            providers = [
                name
                for name, row in metadata["entrypoints"].items()
                if "reproduce" in row.get("native_modes", {})
            ]
            if metadata.get("default_entrypoint"):
                providers = [metadata["default_entrypoint"]]
            for provider in providers:
                forwarded = [
                    item.removeprefix("--arg=")
                    for item in B.example_user_arguments(
                        metadata["entrypoints"][provider]
                    )
                ]
                plan = R.build_action_plan(
                    B.ROOT,
                    experiment_id,
                    "reproduce",
                    entrypoint_name=(
                        None if metadata.get("default_entrypoint") else provider
                    ),
                    user_args=forwarded,
                )
                self.assertEqual(
                    plan["missing_required_user_arguments"],
                    [],
                    f"{experiment_id}:{provider}",
                )

    def test_resume_and_native_verify_require_declared_native_modes(self) -> None:
        self.make_experiment()
        run = self.results / "run"
        run.mkdir()
        for action in ("resume", "native-verify"):
            with self.assertRaisesRegex(R.ReproductionError, "does not declare"):
                R.build_action_plan(
                    self.release,
                    "43_record28",
                    action,
                    run_dir=run,
                    results_root=self.results,
                )

        metadata = {
            "experiment_id": "46_mdp05",
            "status": "completed",
            "entrypoints": {"main": "commands/46_mdp05/run.py"},
            "native_modes": {
                "resume": {
                    "entrypoint": "main",
                    "args": ["resume"],
                    "env": {"RUN_DIR": "{run_dir}"},
                },
                "native-verify": {
                    "entrypoint": "main",
                    "args": ["verify", "--run-dir", "{run_dir}"],
                },
            },
        }
        self.make_experiment("46_mdp05", metadata=metadata)
        resume = R.build_action_plan(
            self.release,
            "46_mdp05",
            "resume",
            run_dir=run,
            results_root=self.results,
        )
        native = R.build_action_plan(
            self.release,
            "46_mdp05",
            "native-verify",
            run_dir=run,
            results_root=self.results,
        )
        self.assertEqual(resume["environment"]["RUN_DIR"], str(run.resolve()))
        self.assertEqual(resume["command"][-1], "resume")
        self.assertEqual(native["command"][-2:], ["--run-dir", str(run.resolve())])

    def test_entrypoint_cannot_escape_commands_directory(self) -> None:
        metadata = {
            "experiment_id": "bad_path",
            "status": "ready",
            "entrypoints": {"main": "code/frozen.txt"},
        }
        self.make_experiment("bad_path", metadata=metadata)
        with self.assertRaisesRegex(R.ReproductionError, "must stay under"):
            R.build_action_plan(self.release, "bad_path", "reproduce")


if __name__ == "__main__":
    unittest.main()
