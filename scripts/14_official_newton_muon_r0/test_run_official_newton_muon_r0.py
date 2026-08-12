from __future__ import annotations

import importlib.util
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_official_newton_muon_r0.py")
SPEC = importlib.util.spec_from_file_location("run_official_newton_muon_r0", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
r0 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r0
SPEC.loader.exec_module(r0)


class R0ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = r0.METHODS["muon"]
        self.profile = r0.RunProfile(
            name="test",
            total_steps=3,
            validation_steps=(0, 3),
        )

    def row(self, event: str, step: int, loss: float) -> dict[str, object]:
        return {
            "method": "muon",
            "event": event,
            "step": step,
            "total_steps": 3,
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

    def test_regex_accepts_and_exposes_special_float_tokens(self) -> None:
        for token in ("nan", "+nan", "-NaN", "inf", "+INF", "-infinity"):
            parsed = r0.match_metric_line(
                f"step:2/6200 train_loss:{token} train_time:10ms step_avg:5.0ms"
            )
            self.assertIsNotNone(parsed, token)
            assert parsed is not None
            self.assertFalse(math.isfinite(r0.parse_number(parsed[1].group("loss"))))

    def test_complete_finite_profile_is_valid(self) -> None:
        summary = r0.validate_and_summarize_metrics(
            self.valid_rows(), 1234.0, self.spec, self.profile
        )
        self.assertTrue(summary["evidence_valid"])
        self.assertEqual(summary["final_train_step"], 3)
        self.assertEqual(summary["final_val_step"], 3)

    def test_nonfinite_loss_is_invalid_even_when_all_steps_exist(self) -> None:
        rows = self.valid_rows()
        rows[2]["loss"] = math.nan
        with self.assertRaises(r0.EvidenceValidationError) as caught:
            r0.validate_and_summarize_metrics(rows, 1234.0, self.spec, self.profile)
        self.assertEqual(caught.exception.status, "invalid_nonfinite")

    def test_return_zero_shape_with_missing_final_step_is_invalid(self) -> None:
        rows = [row for row in self.valid_rows() if row["step"] != 3]
        with self.assertRaises(r0.EvidenceValidationError) as caught:
            r0.validate_and_summarize_metrics(rows, 1234.0, self.spec, self.profile)
        self.assertEqual(caught.exception.status, "invalid_incomplete")

    def test_missing_peak_memory_is_invalid(self) -> None:
        with self.assertRaises(r0.EvidenceValidationError) as caught:
            r0.validate_and_summarize_metrics(
                self.valid_rows(), math.nan, self.spec, self.profile
            )
        self.assertEqual(caught.exception.status, "invalid_incomplete")

    def test_stream_gate_terminates_on_second_step_nan(self) -> None:
        code = (
            "import time\n"
            "print('step:1/10 train_loss:10.0 train_time:1ms step_avg:nanms', flush=True)\n"
            "print('step:2/10 train_loss:nan train_time:2ms step_avg:1.0ms', flush=True)\n"
            "time.sleep(30)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output = io.StringIO()
        observed = r0.stream_process_with_finite_gate(process, output)
        process.wait(timeout=5)
        assert process.stdout is not None
        process.stdout.close()
        self.assertIsNotNone(observed)
        assert observed is not None
        self.assertEqual(observed["step"], 2)
        self.assertEqual(observed["event"], "train")

    def test_known_failed_torch_runtime_is_rejected(self) -> None:
        reason = r0.runtime_rejection_reason({"torch": "2.12.1+cu130"})
        self.assertIn("step 2", reason)

    def test_smoke_certificate_requires_matching_runtime_and_methods(self) -> None:
        runtime = {
            "python_executable": "/usr/bin/python",
            "python": "3.10.12 (main, Nov 20 2023) [GCC 11.4.0]",
            "numpy": "1.24.4",
            "torch": "2.3.0a0.nv24.04",
            "torch_cuda": "12.4",
            "triton": "2.3",
            "triton_module": "/runtime/triton/__init__.py",
            "triton_kernels_module": "/repo/triton_kernels.py",
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "gpu_total_memory_bytes": 80_000_000_000,
        }
        legacy_fingerprint = r0.runtime_fingerprint(runtime)
        legacy_fingerprint.pop("python_version")
        legacy_fingerprint["python"] = "3.10.12 (main, Jun 22 2026) [GCC 11.4.0]"
        payload = {
            "protocol": "official_newton_muon_1_h100_exact_shape_numerical_smoke",
            "official_commit": r0.OFFICIAL_COMMIT,
            "failures": [],
            "summaries": [{"method": "muon"}, {"method": "block4"}],
            "training_runtime_fingerprint": legacy_fingerprint,
        }
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "r0_manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            result = r0.validate_smoke_manifest(
                manifest, runtime, ["muon", "block4"]
            )
            self.assertTrue(result["validated"])
            changed = dict(runtime)
            changed["torch"] = "different"
            with self.assertRaises(RuntimeError):
                r0.validate_smoke_manifest(manifest, changed, ["muon", "block4"])

    def test_smoke_source_preserves_train_shape_and_disables_checkpoint(self) -> None:
        official = (
            "args = Hyperparameters()\n"
            "if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):\n"
            "    save()\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            run_dir = root / "run"
            repo.mkdir()
            run_dir.mkdir()
            (repo / self.spec.script).write_text(official, encoding="utf-8")
            derived, manifest = r0.build_numerical_smoke_source(
                repo, run_dir, self.spec, 10
            )
            text = derived.read_text(encoding="utf-8")
            self.assertIn("args.num_iterations = 10", text)
            self.assertIn("args.warmdown_iters = 1", text)
            self.assertIn("if False and master_process", text)
            self.assertEqual(
                manifest["training_shape_preserved"]["sequence_length"], 1024
            )
            self.assertFalse(manifest["formal_evidence"])


if __name__ == "__main__":
    unittest.main()
