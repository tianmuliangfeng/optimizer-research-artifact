"""GPU-free structural tests for the staged 1B experiment."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module("llama_swiglu_1b_test_runner", HERE / "run_llama_swiglu_1b.py")
trainer = load_module("llama_swiglu_1b_test_trainer", HERE / "train_llama_swiglu_1b.py")


class Llama1BProtocolTests(unittest.TestCase):
    def parse(self, *extra: str):
        argv = [
            "run_llama_swiglu_1b.py",
            "--official-repo",
            "/official",
            "--python-exe",
            "/runtime/python",
            *extra,
        ]
        with mock.patch.object(sys, "argv", argv):
            return runner.parse_args()

    def test_profile_is_exactly_the_planned_1014b_shape(self) -> None:
        profile = runner.PROFILE
        d = profile["n_embd"]
        ff = profile["intermediate_size"]
        layers = profile["n_layer"]
        vocab = 50_257
        expected = vocab * d + layers * (4 * d * d + 3 * d * ff + 2 * d) + d
        self.assertEqual(expected, 1_013_690_368)
        self.assertEqual(profile["expected_parameter_count"], expected)
        self.assertEqual(trainer.PROFILE["expected_parameter_count"], expected)

    def test_exact_k_state_predictions_match_plan(self) -> None:
        d, ff, layers = 2048, 5504, 18
        none = 3 * d * d * 8 * layers
        diag = none + ff * 8 * layers
        full = (3 * d * d + ff * ff) * 8 * layers
        self.assertEqual(none / 2**20, 1728.0)
        self.assertEqual(diag / 2**20, 1728.755859375)
        self.assertEqual(full / 2**20, 5888.25)

    def test_probe_and_smoke_are_short_noncheckpoint_stages(self) -> None:
        probe = self.parse("--stage", "probe")
        smoke = self.parse("--stage", "smoke")
        probe_config = runner.common_config(probe, True)
        smoke_config = runner.common_config(smoke, True)
        self.assertEqual(probe_config["num_iterations"], 1)
        self.assertEqual(smoke_config["num_iterations"], 34)
        self.assertEqual(probe_config["checkpoint_every"], 0)
        self.assertEqual(smoke_config["checkpoint_every"], 0)

    def test_medium_is_plateau_screen_and_formal_has_1800_warmdown(self) -> None:
        medium = self.parse("--stage", "medium", "--smoke-manifest", "/smoke.json")
        formal = self.parse(
            "--stage",
            "formal",
            "--smoke-manifest",
            "/smoke.json",
            "--medium-manifest",
            "/medium.json",
        )
        medium_config = runner.common_config(medium, False)
        formal_config = runner.common_config(formal, False)
        self.assertEqual(medium_config["num_iterations"], 1000)
        self.assertEqual(medium_config["warmdown_iters"], 0)
        self.assertEqual(formal_config["num_iterations"], 6200)
        self.assertEqual(formal_config["warmdown_iters"], 1800)

    def test_formal_refuses_missing_stage_certificates(self) -> None:
        with self.assertRaises(SystemExit):
            self.parse("--stage", "formal")
        with self.assertRaises(SystemExit):
            self.parse("--stage", "formal", "--smoke-manifest", "/smoke.json")

    def test_resume_is_limited_to_medium_and_formal(self) -> None:
        with self.assertRaises(SystemExit):
            self.parse("--stage", "smoke", "--resume-batch", "/batch")
        resumed = self.parse("--stage", "formal", "--resume-batch", "/batch")
        self.assertEqual(resumed.execution_stage, "formal")

    def test_device_batch_must_preserve_global_batch(self) -> None:
        with self.assertRaises(SystemExit):
            self.parse("--stage", "probe", "--device-batch-size", "7")
        args = self.parse("--stage", "probe", "--device-batch-size", "8")
        self.assertEqual(args.device_batch_size, 8)

    def test_wrapper_binds_and_archives_base_trainer(self) -> None:
        source = (HERE / "train_llama_swiglu_1b.py").read_text(encoding="utf-8")
        self.assertIn("LLAMA_1B_BASE_TRAINER_SHA256", source)
        self.assertIn("train_llama_swiglu_base.py", source)
        self.assertIn("parameter count drift", source)
        self.assertEqual(runner.BASE_TRAINER_SHA256, runner.PINNED_BASE_TRAINER_SHA256)

    def test_subprocess_env_calls_original_base_function_without_recursion(self) -> None:
        official = Path("/official-support-repo")
        env = runner.subprocess_env(official)
        self.assertEqual(env["PYTHONPATH"].split(runner.os.pathsep)[0], str(official))
        self.assertEqual(env["LLAMA_1B_BASE_TRAINER"], str(runner.BASE_TRAINER_PATH.resolve()))
        self.assertEqual(env["LLAMA_1B_BASE_TRAINER_SHA256"], runner.BASE_TRAINER_SHA256)

    def test_certificate_rejects_device_batch_drift(self) -> None:
        smoke_args = self.parse("--stage", "smoke", "--methods", "down_diag")
        medium_args = self.parse(
            "--stage",
            "medium",
            "--methods",
            "down_diag",
            "--device-batch-size",
            "4",
            "--smoke-manifest",
            "/smoke.json",
        )
        payload = {
            "status": "completed",
            "execution_stage": "smoke",
            "seed": 2026,
            "profile": runner.PROFILE,
            "script_sha256": "wrapper",
            "base_trainer_sha256": runner.BASE_TRAINER_SHA256,
            "data_audit": {"fingerprint": "data"},
            "runtime": {},
            "init_audit": {"common_init_sha256": "init"},
            "completed_methods": ["down_diag"],
            "config": runner.common_config(smoke_args, True),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "device_batch_size"):
                runner.validate_stage_manifest(
                    path,
                    "smoke",
                    medium_args,
                    {},
                    {"fingerprint": "data"},
                    {"common_init_sha256": "init"},
                    "wrapper",
                )


if __name__ == "__main__":
    unittest.main()
