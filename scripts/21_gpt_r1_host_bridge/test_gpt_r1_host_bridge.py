from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("run_gpt_r1_host_bridge.py")
SPEC = importlib.util.spec_from_file_location("run_gpt_r1_host_bridge", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
bridge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bridge
SPEC.loader.exec_module(bridge)


class HostBridgeTests(unittest.TestCase):
    def args(self, **overrides: object) -> Namespace:
        values: dict[str, object] = {
            "host_bridge": True,
            "lr_cross": False,
            "numerical_smoke": False,
            "dry_run": False,
            "concurrent_node_training": True,
            "concurrent_workload": "llama_swiglu_seed2024_gpu0",
        }
        values.update(overrides)
        return Namespace(**values)

    def test_wrapper_injects_host_bridge_once(self) -> None:
        argv = bridge.bridge_argv(["runner.py", "--dry-run"])
        self.assertEqual(argv, ["runner.py", "--host-bridge", "--dry-run"])
        self.assertEqual(bridge.bridge_argv(argv), argv)

    def test_base_runner_exposes_required_bridge_api(self) -> None:
        bridge.require_bridge_capable_r1()

    def test_separate_family_protocol_and_timing_policy(self) -> None:
        args = self.args()
        self.assertEqual(bridge.r1.experiment_family(args), bridge.r1.HOST_BRIDGE_FAMILY)
        self.assertEqual(
            bridge.r1.experiment_protocol(args), bridge.r1.HOST_BRIDGE_FORMAL_PROTOCOL
        )
        self.assertEqual(
            bridge.r1.experiment_protocol(args, smoke=True),
            bridge.r1.HOST_BRIDGE_SMOKE_PROTOCOL,
        )
        policy = bridge.r1.evidence_eligibility(args)
        self.assertTrue(policy["quality_usable"])
        self.assertTrue(policy["memory_usable"])
        self.assertFalse(policy["timing_usable"])

    def test_bridge_requires_exactly_one_visible_gpu(self) -> None:
        args = self.args()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False):
            record = bridge.r1.visible_device_record(args)
        self.assertEqual(record["cuda_visible_devices"], "1")
        self.assertTrue(record["one_process_one_gpu"])

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
            with self.assertRaises(RuntimeError):
                bridge.r1.visible_device_record(args)


if __name__ == "__main__":
    unittest.main()
