from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import r1_depth_source_builder as builder
import run_r1_depth_kmode as runner


def load_batch_module():
    path = SCRIPT_DIR / "run_three_seed_batch.py"
    spec = importlib.util.spec_from_file_location("r1_depth_three_seed_batch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


batch = load_batch_module()


class R1DepthContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        explicit = os.environ.get("SNM_OFFICIAL_REPO")
        candidates = (
            ARTIFACT_ROOT / "third_party" / "Newton-Muon-official-r0",
            ARTIFACT_ROOT / "third_party" / "Newton-Muon-official",
        )
        cls.official_repo = (
            Path(explicit).expanduser().resolve()
            if explicit
            else next((path for path in candidates if path.is_dir()), candidates[0])
        )

    def test_frozen_method_matrix(self) -> None:
        builder.self_test_contract()
        self.assertEqual(len(builder.DEPTH_METHODS), 10)
        self.assertEqual(len(builder.ALLOWED_METHODS), 12)
        self.assertEqual(builder.ANCHORS, ("block4", "muon"))
        for rule in ("early", "center", "late", "edge"):
            self.assertEqual(len(builder.RULE_LAYERS[rule]), 8)
        self.assertEqual(builder.RULE_LAYERS["all"], tuple(range(12)))

    def test_generated_sources_compile_and_use_block4_fallback(self) -> None:
        if not self.official_repo.is_dir():
            self.skipTest(f"official repository unavailable: {self.official_repo}")
        hashes: set[str] = set()
        for method in builder.DEPTH_METHODS:
            derived = builder.build_source(self.official_repo, method)
            compile(derived.source, f"<test-{method}>", "exec")
            hashes.add(derived.derived_sha256)
            self.assertIn("def __init__(self, config, layer_idx: int):", derived.source)
            self.assertIn('else "block4"', derived.source)
            self.assertIn("R1_DEPTH_ROUTING", derived.source)
            self.assertIn(
                "Block(config, layer_idx) for layer_idx in range(config.n_layer)",
                derived.source,
            )
            self.assertNotIn(
                "self.r1_cproj_k_mode = (\n"
                "            self.r1_cproj_k_mode",
                derived.source,
            )
        self.assertEqual(len(hashes), len(builder.DEPTH_METHODS))

    def test_anchor_sources_are_exact_base_runner_sources(self) -> None:
        if not self.official_repo.is_dir():
            self.skipTest(f"official repository unavailable: {self.official_repo}")
        for method in builder.ANCHORS:
            observed = builder.build_source(self.official_repo, method)
            expected = builder.base.build_source(self.official_repo, method)
            self.assertEqual(observed.derived_sha256, expected.derived_sha256)
            self.assertEqual(observed.source, expected.source)

    def test_specs_freeze_lr_and_intervention_boundary(self) -> None:
        self.assertEqual(runner.METHODS["early_diag"].base_learning_rate, 0.004)
        self.assertEqual(runner.METHODS["early_diag"].matrix_learning_rate, 0.0004)
        self.assertIn("unselected_mode=block4", runner.METHODS["early_diag"].role)
        self.assertEqual(runner.METHODS["block4"].base_learning_rate, 0.004)
        self.assertEqual(runner.METHODS["muon"].base_learning_rate, 0.0036)

    def test_controlled_environment_routes_selected_and_unselected_layers(self) -> None:
        old = getattr(runner, "_ORIGINAL_CONTROLLED_ENV", None)
        runner._ORIGINAL_CONTROLLED_ENV = lambda *args, **kwargs: {
            "R1_DISABLE_CHECKPOINT": "0"
        }
        try:
            args = Namespace(seed=2026)
            env = runner.controlled_env(
                args,
                runner.METHODS["edge_diag"],
                Path("/data"),
                smoke_test=False,
            )
        finally:
            if old is None:
                del runner._ORIGINAL_CONTROLLED_ENV
            else:
                runner._ORIGINAL_CONTROLLED_ENV = old
        self.assertEqual(env["R1_DEPTH_RULE"], "edge")
        self.assertEqual(env["R1_CPROJ_K_MODE"], "diag")
        self.assertEqual(env["R1_CPROJ_K_LAYERS"], "0,1,2,3,8,9,10,11")
        self.assertEqual(env["R1_DISABLE_CHECKPOINT"], "1")

    def test_two_gpu_shards_cover_every_method_once(self) -> None:
        flattened = [method for shard in batch.METHOD_SHARDS for method in shard]
        self.assertEqual(len(flattened), 12)
        self.assertEqual(set(flattened), set(builder.ALLOWED_METHODS))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(tuple(map(len, batch.METHOD_SHARDS)), (6, 6))
        anchor_shard = batch.METHOD_SHARDS[1]
        self.assertTrue(
            {"all_none", "all_diag", "block4", "muon"}.issubset(anchor_shard)
        )

    def test_seed_crossover_keeps_balanced_gpu_queues(self) -> None:
        jobs = batch.scheduled_jobs([2024, 2025, 2026], ["0", "1"])
        assignment = {
            (int(job["seed"]), int(job["shard"])): str(job["device"])
            for job in jobs
        }
        self.assertEqual(assignment[(2024, 0)], "0")
        self.assertEqual(assignment[(2024, 1)], "1")
        self.assertEqual(assignment[(2025, 0)], "1")
        self.assertEqual(assignment[(2025, 1)], "0")
        self.assertEqual(assignment[(2026, 0)], "0")
        self.assertEqual(assignment[(2026, 1)], "1")
        self.assertEqual(
            sum(job["device"] == "0" for job in jobs),
            sum(job["device"] == "1" for job in jobs),
        )

    def test_batch_commands_keep_job_result_namespaces_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = Namespace(
                official_repo=Path("/official"),
                python_exe="/runtime/python",
                smoke_steps=34,
                run_prefix="depth",
                wandb_project="project",
                wandb_mode="online",
                wandb_entity=None,
            )
            left = batch.command_for(
                args,
                2024,
                batch.METHOD_SHARDS[0],
                root / "seed2024_shard0",
                smoke=True,
            )
            right = batch.command_for(
                args,
                2024,
                batch.METHOD_SHARDS[1],
                root / "seed2024_shard1",
                smoke=True,
            )
            self.assertIn(str(root / "seed2024_shard0"), left)
            self.assertIn(str(root / "seed2024_shard1"), right)
            self.assertNotEqual(left, right)
            self.assertIn("--smoke-steps", left)

    def test_batch_controller_preflight_requires_split_and_wandb(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            training_python = Path(temp) / "training-python"
            training_python.touch()
            fake_wandb = SimpleNamespace(__version__="test-version")
            with mock.patch.object(
                batch.importlib, "import_module", return_value=fake_wandb
            ) as importer:
                payload = batch.controller_runtime_preflight(
                    str(training_python), "online"
                )
            importer.assert_called_once_with("wandb")
            self.assertTrue(payload["interpreters_separate"])
            self.assertEqual(payload["wandb_version"], "test-version")

        with self.assertRaisesRegex(RuntimeError, "must be separate"):
            batch.controller_runtime_preflight(sys.executable, "online")

    def test_batch_controller_preflight_does_not_resolve_venv_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            controller_entry = root / "controller-venv" / "bin" / "python"
            training_entry = root / "training-venv" / "bin" / "python"
            controller_entry.parent.mkdir(parents=True)
            training_entry.parent.mkdir(parents=True)
            controller_entry.touch()
            training_entry.touch()
            fake_wandb = SimpleNamespace(__version__="test-version")
            with (
                mock.patch.object(batch.sys, "executable", str(controller_entry)),
                mock.patch.object(
                    batch.importlib, "import_module", return_value=fake_wandb
                ),
                mock.patch.object(
                    batch.Path,
                    "resolve",
                    side_effect=AssertionError("venv paths must not be resolved"),
                ),
            ):
                payload = batch.controller_runtime_preflight(
                    str(training_entry), "online"
                )
            self.assertEqual(payload["controller_python"], str(controller_entry))
            self.assertEqual(payload["training_python"], str(training_entry))

    def test_batch_controller_preflight_reports_missing_wandb(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            training_python = Path(temp) / "training-python"
            training_python.touch()
            with mock.patch.object(
                batch.importlib,
                "import_module",
                side_effect=ModuleNotFoundError("wandb"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "controller Python cannot import wandb"
                ):
                    batch.controller_runtime_preflight(
                        str(training_python), "online"
                    )


if __name__ == "__main__":
    unittest.main()
