#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_mech01.py")
SPEC = importlib.util.spec_from_file_location("run_mech01", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReplayComparisonTests(unittest.TestCase):
    def test_identical_replay_passes(self) -> None:
        payload = {
            "bundle_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "triton": {"sha256": "c" * 64},
            "script_version": "test",
            "results": {
                "covariance": {"diag_cv": 0.25},
                "candidates": {
                    "none": {"update_norm": 3.0, "update_cosine_to_none": 1.0}
                },
            },
        }
        result = MODULE.compare_replay_payloads(
            payload, payload, atol=1e-7, rtol=1e-6
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["same_bundle_sha256"])

    def test_bundle_or_metric_mismatch_fails(self) -> None:
        left = {
            "bundle_sha256": "a" * 64,
            "source_sha256": "c" * 64,
            "triton": {"sha256": "d" * 64},
            "script_version": "test",
            "results": {"candidate": {"update_norm": 1.0}},
        }
        right = {
            "bundle_sha256": "b" * 64,
            "source_sha256": "c" * 64,
            "triton": {"sha256": "d" * 64},
            "script_version": "test",
            "results": {"candidate": {"update_norm": 2.0}},
        }
        result = MODULE.compare_replay_payloads(
            left, right, atol=1e-7, rtol=1e-6
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["same_bundle_sha256"])
        self.assertFalse(result["checks"]["all_metrics_within_tolerance"])

    def test_atomic_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            MODULE.write_json(path, {"passed": True, "value": 4})
            self.assertEqual(
                MODULE.read_json(path), {"passed": True, "value": 4}
            )

    def test_worker_has_no_wandb_or_optimizer_step(self) -> None:
        worker = SCRIPT.with_name("mech01_worker.py").read_text(encoding="utf-8")
        self.assertNotIn("import wandb", worker)
        self.assertNotIn(".step()", worker)
        self.assertIn("precond_flag=False", worker)
        self.assertIn("dont_inherit=True", worker)
        self.assertIn('module["R1_CPROJ_K_MODE"] = method', worker)
        self.assertIn('module["R1_METHOD"] = method', worker)
        self.assertIn('"source_runtime_globals"', worker)

    def test_archived_source_annotations_are_not_postponed(self) -> None:
        source = (
            "class Tensor:\n"
            "    pass\n"
            "def custom_op(x: Tensor) -> Tensor:\n"
            "    return x\n"
        )
        namespace: dict[str, object] = {}
        # This test module itself enables postponed annotations.  The archived
        # R1 source does not, so the MECH-01 loader must not inherit our flag.
        exec(
            compile(
                source,
                "<archived-r1-source>",
                "exec",
                dont_inherit=True,
            ),
            namespace,
        )
        tensor_type = namespace["Tensor"]
        annotations = namespace["custom_op"].__annotations__  # type: ignore[union-attr]
        self.assertIs(annotations["x"], tensor_type)
        self.assertIs(annotations["return"], tensor_type)


if __name__ == "__main__":
    unittest.main()
