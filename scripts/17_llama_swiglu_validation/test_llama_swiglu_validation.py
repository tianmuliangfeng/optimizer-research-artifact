from __future__ import annotations

import importlib.util
import csv
import json
import tempfile
import unittest
from unittest import mock
from argparse import Namespace
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_llama_swiglu_validation.py")
SPEC = importlib.util.spec_from_file_location("run_llama_swiglu_validation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class LlamaSwiGLUSafetyTests(unittest.TestCase):
    def args(self) -> Namespace:
        return Namespace(
            smoke_steps=34,
            checkpoint_every=128,
            device_batch_size=64,
            backup_lr=0.0036,
            matrix_lr=0.01,
            adamw_matrix_lr=0.000576,
            methods=list(runner.METHOD_ORDER),
            seed=2026,
            python_exe=Path("/runtime/python"),
            official_repo=Path("/repo"),
            wandb_project="test",
            wandb_mode="disabled",
        )

    def runtime(self) -> dict[str, object]:
        return {
            "python_executable": "/runtime/python",
            "python_version": [3, 10, 12],
            "python_full": "build date deliberately ignored",
            "numpy": "2.2.6",
            "torch": "2.8.0+cu126",
            "torch_cuda": "12.6",
            "triton": "3.4.0",
            "triton_kernels_sha256": "a" * 64,
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "gpu_total_memory_bytes": 85017624576,
            "gpu_capability": [9, 0],
        }

    def init_audit(self) -> dict[str, object]:
        return {
            "common_init_sha256": "b" * 64,
            "expected_k_state_bytes": {
                "down_diag": 169_869_312 + 196_608,
                "down_none": 169_869_312,
                "newton_full": 169_869_312 + 402_653_184,
                "muon": 0,
                "adamw": 0,
            },
        }

    def test_formal_shape_and_budget_are_pinned(self) -> None:
        config = runner.common_config(self.args(), smoke=False)
        self.assertEqual(config["num_iterations"], 6200)
        self.assertEqual(config["global_batch_size"], 512)
        self.assertEqual(config["sequence_length"], 1024)
        self.assertEqual(config["val_tokens"], 10485760)
        self.assertEqual(config["matrix_lr"], 0.01)

    def test_smoke_reaches_first_preconditioner_refresh(self) -> None:
        config = runner.common_config(self.args(), smoke=True)
        self.assertGreaterEqual(config["num_iterations"], 32)
        self.assertEqual(config["global_batch_size"], 512)
        self.assertEqual(config["sequence_length"], 1024)
        self.assertEqual(config["checkpoint_every"], 0)

    def test_expected_shared_k_state_accounting(self) -> None:
        audit = self.init_audit()["expected_k_state_bytes"]
        self.assertEqual(audit["down_none"] / 2**20, 162.0)
        self.assertEqual(audit["down_diag"] / 2**20, 162.1875)
        self.assertEqual(audit["newton_full"] / 2**20, 546.0)

    def test_runtime_fingerprint_ignores_python_build_date_only(self) -> None:
        left = self.runtime()
        right = dict(left)
        right["python_full"] = "different non-semantic build date"
        self.assertEqual(runner.stable_runtime(left), runner.stable_runtime(right))
        right["torch"] = "different"
        self.assertNotEqual(runner.stable_runtime(left), runner.stable_runtime(right))

    def test_virtualenv_interpreter_path_is_not_resolved(self) -> None:
        with mock.patch.object(Path, "resolve", side_effect=AssertionError("must not resolve")):
            observed = runner.lexical_absolute(Path("venv/bin/python"))
        self.assertTrue(observed.is_absolute())
        self.assertTrue(str(observed).replace("\\", "/").endswith("venv/bin/python"))

    def test_training_custom_ops_use_runtime_tensor_annotations(self) -> None:
        source = Path(__file__).with_name("train_llama_swiglu.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("from __future__ import annotations", source)
        self.assertIn("def accum_xtx_op(x_2d: Tensor", source)
        self.assertIn("def accum_diag_op(x_2d: Tensor", source)

    def test_resume_rejects_method_or_source_drift(self) -> None:
        args = self.args()
        data = {"fingerprint": "data"}
        plan = {
            "batch_kind": "formal",
            "seed": args.seed,
            "methods": args.methods,
            "config": runner.common_config(args, False),
            "script_sha256": "source",
            "data_audit": data,
            "runtime": self.runtime(),
            "init_audit": {"common_init_sha256": "b" * 64},
        }
        runner.validate_resume_plan(
            plan, args, self.runtime(), data, self.init_audit(), "source"
        )
        changed = dict(plan)
        changed["methods"] = list(reversed(args.methods))
        with self.assertRaises(RuntimeError):
            runner.validate_resume_plan(
                changed, args, self.runtime(), data, self.init_audit(), "source"
            )

    def test_smoke_certificate_requires_every_formal_method(self) -> None:
        args = self.args()
        data = {"fingerprint": "data"}
        payload = {
            "batch_kind": "smoke",
            "status": "completed",
            "script_sha256": "source",
            "data_audit": data,
            "runtime": self.runtime(),
            "init_audit": {"common_init_sha256": "b" * 64},
            "completed_methods": args.methods,
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "llama_manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            runner.validate_smoke_certificate(
                path,
                args,
                self.runtime(),
                data,
                "source",
                self.init_audit(),
            )
            payload["completed_methods"] = args.methods[:-1]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                runner.validate_smoke_certificate(
                    path,
                    args,
                    self.runtime(),
                    data,
                    "source",
                    self.init_audit(),
                )

    def test_metric_evidence_rejects_missing_train_step(self) -> None:
        fields = (
            "event",
            "step",
            "loss",
            "train_s",
            "steady_train_s",
            "step_avg_ms",
            "lr_backup",
            "lr_matrix",
            "tokens_seen",
        )
        rows = [
            {"event": "val", "step": 0, "loss": 10, "train_s": 0, "tokens_seen": 0},
            {"event": "train", "step": 1, "loss": 9, "train_s": 1, "tokens_seen": 8},
            {"event": "train", "step": 2, "loss": 8, "train_s": 2, "tokens_seen": 16},
            {"event": "val", "step": 2, "loss": 7, "train_s": 2, "tokens_seen": 16},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            runner.validate_metric_evidence(
                path,
                total_steps=2,
                val_every=2,
                global_batch_size=2,
                sequence_length=4,
            )
            rows.pop(1)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                runner.validate_metric_evidence(
                    path,
                    total_steps=2,
                    val_every=2,
                    global_batch_size=2,
                    sequence_length=4,
                )


if __name__ == "__main__":
    unittest.main()
