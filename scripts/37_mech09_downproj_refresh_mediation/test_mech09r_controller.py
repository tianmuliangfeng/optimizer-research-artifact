#!/usr/bin/env python3
"""CPU-only resume and scheduling tests for MECH-09R."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def load_controller():
    path = HERE / "run_mech09r.py"
    spec = importlib.util.spec_from_file_location(
        "run_mech09r_tested", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load_controller()
CONTRACT = json.loads(
    (HERE / "refresh_mediation_repair_contract.json").read_text(
        encoding="utf-8"
    )
)


class Mech09RControllerTests(unittest.TestCase):
    def test_smoke_aggregate_matches_worker_gate_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            manifest_path = (
                run
                / "smoke"
                / "early_muon"
                / "replica_0"
                / "mech09r_manifest.json"
            )
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "script_version": R.WORKER_VERSION,
                        "contract_sha256": "contract",
                        "causal_tree": True,
                    }
                ),
                encoding="utf-8",
            )
            contract = {
                "smoke": {
                    "origins": ["early_muon"],
                    "data_replicas": [0],
                }
            }
            aggregate = R.aggregate_tier(
                run, contract, "smoke", "contract"
            )
            self.assertTrue(aggregate["passed"])
            self.assertEqual(
                aggregate["script_version"], R.WORKER_VERSION
            )
            self.assertEqual(aggregate["analysis_tier"], "smoke")
            self.assertTrue(aggregate["causal_tree"])

    def test_completed_manifest_requires_repair_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mech09r_manifest.json"
            payload = {
                "passed": True,
                "script_version": R.WORKER_VERSION,
                "analysis_tier": "formal",
                "checkpoint_cell": "early_muon",
                "data_replica": 0,
                "contract_sha256": "contract",
                "causal_tree": True,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(
                R.completed_manifest(
                    path,
                    tier="formal",
                    cell="early_muon",
                    replica=0,
                    contract_sha256="contract",
                )
            )
            payload["causal_tree"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(
                R.completed_manifest(
                    path,
                    tier="formal",
                    cell="early_muon",
                    replica=0,
                    contract_sha256="contract",
                )
            )

    def test_build_jobs_has_one_worker_per_unit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            certificates = {
                row["cell"]: root / f"{row['cell']}.json"
                for row in CONTRACT["checkpoints"]
            }
            args = SimpleNamespace(
                child_python="/python",
                source_script=HERE / "source.py",
                profile_script=HERE / "profile.py",
                triton_kernels=HERE / "triton.py",
                contract=HERE / "refresh_mediation_repair_contract.json",
                mech08_control_reference=HERE
                / "mech08_control_reference.json",
                train_data_pattern="/train_*.bin",
                val_data_pattern="/val_*.bin",
                host_id="host",
                execution_domain="domain",
            )
            jobs = R.build_jobs(
                args,
                CONTRACT,
                root,
                certificates,
                "formal",
                "contract",
                root / "smoke_manifest.json",
            )
            self.assertEqual(len(jobs), 12)
            labels = {row["label"] for row in jobs}
            self.assertEqual(len(labels), 12)
            self.assertTrue(
                all("delayed_down_refresh" not in label for label in labels)
            )
            self.assertTrue(
                all("frozen_down_refresh" not in label for label in labels)
            )
            self.assertTrue(
                all(
                    row["command"][1].endswith("mech09r_worker.py")
                    for row in jobs
                )
            )


if __name__ == "__main__":
    unittest.main()
