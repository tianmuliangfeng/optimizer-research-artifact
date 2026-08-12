from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_official_newton_muon_r1.py")
SPEC = importlib.util.spec_from_file_location("run_official_newton_muon_r1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
r1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r1
SPEC.loader.exec_module(r1)


class R1SafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = r1.METHODS["muon"]
        self.profile = r1.RunProfile(
            name="test",
            total_steps=3,
            validation_steps=(0, 3),
            formal_evidence=False,
            require_checkpoint=False,
        )

    def row(self, event: str, step: int, loss: float) -> dict[str, object]:
        return {
            "method": "muon",
            "cproj_k_mode": "muon",
            "event": event,
            "step": step,
            "total_steps": 3,
            "tokens_seen": step * r1.TOKENS_PER_STEP,
            "loss": loss,
            "official_train_time_ms": step * 10,
            "step_avg_ms": math.nan if step == 0 else 10.0,
            "lr_multiplier": 1.0,
            "adamw_lr": 0.0036,
            "matrix_lr": 0.00036,
        }

    def valid_rows(self) -> list[dict[str, object]]:
        return [
            self.row("validation", 0, 10.0),
            self.row("train", 1, 9.0),
            self.row("train", 2, 8.0),
            self.row("train", 3, 7.0),
            self.row("validation", 3, 6.0),
        ]

    def validate(self, rows: list[dict[str, object]]) -> None:
        r1.validate_metric_evidence(
            rows=rows,
            spec=self.spec,
            profile=self.profile,
            metadata={
                "method": "muon",
                "cproj_k_mode": "muon",
                "seed": 2026,
                "init_sha256": "a" * 64,
            },
            k_memory={
                "k_cov": 0,
                "k_inv": 0,
                "k_state": 0,
                "activation": 0,
                "workspace": 0,
                "total": 0,
            },
            final_memory={"optimizer": 10, "model": 10},
            peak_mib=100,
            checkpoint_path=None,
            expected_seed=2026,
            expected_init_sha256="a" * 64,
        )

    def test_complete_finite_rows_pass(self) -> None:
        self.validate(self.valid_rows())

    def test_nonfinite_rows_fail(self) -> None:
        rows = self.valid_rows()
        rows[2]["loss"] = math.nan
        with self.assertRaises(r1.r0.EvidenceValidationError) as caught:
            self.validate(rows)
        self.assertEqual(caught.exception.status, "invalid_nonfinite")

    def test_missing_final_step_fails(self) -> None:
        rows = [row for row in self.valid_rows() if row["step"] != 3]
        with self.assertRaises(r1.r0.EvidenceValidationError) as caught:
            self.validate(rows)
        self.assertEqual(caught.exception.status, "invalid_incomplete")

    def test_smoke_overlay_preserves_formal_train_shape(self) -> None:
        overlay = r1.build_source.__globals__["COMMON_CONTROL"]
        self.assertIn("args.num_iterations = R1_SMOKE_STEPS", overlay)
        self.assertIn("args.val_tokens = args.device_batch_size * args.sequence_length", overlay)
        self.assertNotIn("args.batch_size = 1", overlay)
        self.assertNotIn("args.device_batch_size = 1", overlay)
        self.assertNotIn("args.sequence_length = 128", overlay)

    def test_smoke_certificate_requires_runtime_seed_methods_and_sources(self) -> None:
        runtime = {
            "python_executable": "/runtime/python",
            "python": "3.10.12 (main, date) [GCC]",
            "numpy": "2.2.6",
            "torch": "2.8.0+cu126",
            "torch_cuda": "12.6",
            "triton": "3.4.0",
            "triton_module": "/runtime/triton/__init__.py",
            "triton_kernels_module": "/repo/triton_kernels.py",
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "gpu_total_memory_bytes": 80_000_000_000,
        }
        built = {
            method: r1.DerivedSource(method, "base.py", "b", method, "", "")
            for method in r1.ALLOWED_METHODS
        }
        payload = {
            "protocol": "official_newton_muon_1_r1_exact_shape_numerical_smoke",
            "official_commit": r1.r0.OFFICIAL_COMMIT,
            "seed": 2026,
            "failures": [],
            "initialization_audit": {"all_methods_identical": True},
            "summaries": [
                {"method": method, "evidence_valid": True}
                for method in r1.ALLOWED_METHODS
            ],
            "training_runtime_fingerprint": r1.r0.runtime_fingerprint(runtime),
            "derived_source_sha256": r1.source_fingerprints(built),
        }
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "r1_manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            certificate = r1.validate_smoke_manifest(
                manifest, runtime, ["diag", "none", "block4", "muon"], 2026, built
            )
            self.assertTrue(certificate["validated"])
            changed = dict(runtime)
            changed["torch"] = "different"
            with self.assertRaises(RuntimeError):
                r1.validate_smoke_manifest(
                    manifest, changed, ["diag", "none", "block4", "muon"], 2026, built
                )

    def test_retry_name_preserves_interrupted_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = "r1_diag"
            (root / base).mkdir()
            self.assertEqual(r1.next_attempt_run_name(root, base), "r1_diag_retry02")
            (root / "r1_diag_retry02").mkdir()
            self.assertEqual(r1.next_attempt_run_name(root, base), "r1_diag_retry03")

    def test_lr_cross_protocol_and_specs_are_isolated(self) -> None:
        args = Namespace(lr_cross=True, numerical_smoke=False)
        self.assertEqual(r1.experiment_family(args), r1.LR_CROSS_FAMILY)
        self.assertEqual(r1.experiment_protocol(args), r1.LR_CROSS_FORMAL_PROTOCOL)
        self.assertEqual(
            r1.experiment_protocol(args, smoke=True), r1.LR_CROSS_SMOKE_PROTOCOL
        )
        specs = r1.experiment_specs(args)
        self.assertEqual(set(specs), {"muon", "diag"})
        self.assertEqual(specs["muon"].base_learning_rate, 0.0040)
        self.assertEqual(specs["diag"].base_learning_rate, 0.0036)

    def test_lr_cross_source_changes_exactly_one_learning_rate_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            base_source = "class Hyperparameters:\n    learning_rate : float = 0.0036\n"
            (repo / "train.py").write_text(base_source, encoding="utf-8")
            derived = r1.DerivedSource(
                method="muon",
                base_script="train.py",
                base_canonical_sha256="base",
                derived_sha256="old",
                source=base_source,
                unified_diff="",
            )
            crossed = r1.override_derived_learning_rate(
                repo,
                derived,
                original_literal="0.0036",
                crossed_literal="0.0040",
            )
            self.assertIn("learning_rate : float = 0.0040", crossed.source)
            self.assertNotIn("learning_rate : float = 0.0036", crossed.source)
            self.assertNotEqual(crossed.derived_sha256, derived.derived_sha256)
            self.assertIn("+    learning_rate : float = 0.0040", crossed.unified_diff)


if __name__ == "__main__":
    unittest.main()
