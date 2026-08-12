#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = load("protocol", HERE / "protocol.py")
R = load("ex48_runner_test", HERE / "run_formal.py")
A = load("ex48_analyzer_test", HERE / "analyze_formal.py")
W = load("ex48_worker_test", HERE / "train_segment.py")


def fake_phase(
    root: Path,
    method: str,
    seed: int,
    phase_id: str,
    retained: bool,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    summary = root / "summary.json"
    metrics = root / "metrics.csv"
    checkpoint = root / "checkpoint_latest.pt"
    P.atomic_json(summary, {"status": "completed"})
    metrics.write_text(
        ",".join(P.METRIC_FIELDS) + "\n" + ",".join(["val", phase_id, "plateau", "1", "1", "3.0"] + ["0"] * 9) + "\n",
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"small-checkpoint")
    payload = {
        "schema_version": P.PHASE_MANIFEST_SCHEMA,
        "passed": True,
        "method": method,
        "seed": seed,
        "phase_id": phase_id,
        "summary_sha256": P.sha256_file(summary),
        "metrics_sha256": P.sha256_file(metrics),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": P.sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "retained": retained,
        },
    }
    P.atomic_json(root / "phase_manifest.json", payload)
    return payload


class ControllerTests(unittest.TestCase):
    def test_public_path_only_source_hash_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            source = repo / "source.py"
            source.write_text("# public path-only rewrite\n", encoding="utf-8")
            public_hash = P.sha256_file(source)
            contract = {
                "accepted_sources": {"source.py": "0" * 64},
                "public_source_hashes": {"source.py": public_hash},
            }
            previous = R.SNAPSHOT_FILES
            try:
                R.SNAPSHOT_FILES = ("source.py",)
                self.assertTrue(R.check_live_sources(repo, contract)["passed"])
                source.write_text("# unauthorized change\n", encoding="utf-8")
                self.assertFalse(R.check_live_sources(repo, contract)["passed"])
            finally:
                R.SNAPSHOT_FILES = previous

    def test_four_gpu_lpt_queue_is_complete_and_balanced(self) -> None:
        units = [
            {"method": method, "seed": seed}
            for method in ("newton_full", "down_diag", "down_none", "muon")
            for seed in (2024, 2025, 2026)
        ]
        units.sort(key=lambda row: (-R.STEP_SECONDS[row["method"]], row["seed"], row["method"]))
        queues, loads = R.build_gpu_queues(units, [0, 1, 2, 3])
        assigned = [
            (row["method"], row["seed"])
            for gpu in sorted(queues)
            for row in queues[gpu]
        ]
        expected = [(row["method"], row["seed"]) for row in units]
        self.assertCountEqual(assigned, expected)
        self.assertTrue(all(len(queue) == 3 for queue in queues.values()), queues)
        self.assertLessEqual(max(loads.values()) - min(loads.values()), max(R.STEP_SECONDS.values()))

    def test_cuda_mapped_rng_states_are_normalized_to_cpu(self) -> None:
        class FakeTensor:
            def __init__(self, device: str) -> None:
                self.device = device

            def cpu(self):
                return FakeTensor("cpu")

        class FakeBase:
            restored = None

            def restore_rng_state(self, payload):
                self.restored = payload

        base = FakeBase()
        original = {
            "python": "python-state",
            "numpy": "numpy-state",
            "torch_cpu": FakeTensor("cuda"),
            "torch_cuda": [FakeTensor("cuda"), FakeTensor("cuda")],
        }
        W.restore_checkpoint_rng_state(base, original)
        self.assertEqual(base.restored["torch_cpu"].device, "cpu")
        self.assertTrue(all(state.device == "cpu" for state in base.restored["torch_cuda"]))
        self.assertEqual(original["torch_cpu"].device, "cuda")

    def test_retained_and_retired_phase_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "phase"
            manifest = fake_phase(root, "muon", 2024, "p", retained=True)
            expected = {"method": "muon", "seed": 2024, "phase_id": "p"}
            self.assertTrue(R.phase_manifest_valid(root, expected, full_checkpoint_hash=True))
            checkpoint = Path(manifest["checkpoint"]["path"])
            checkpoint.unlink()
            manifest["checkpoint"]["retained"] = False
            P.atomic_json(root / "phase_manifest.json", manifest)
            P.atomic_json(
                root / "checkpoint_retirement.json",
                {
                    "passed": True,
                    "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                    "checkpoint_bytes": manifest["checkpoint"]["bytes"],
                },
            )
            self.assertTrue(R.phase_manifest_valid(root, expected))

    def test_fork_retirement_waits_for_all_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "formal" / "muon" / "seed2024"
            parent = fake_phase(unit / "parent", "muon", 2024, "parent", retained=False)
            fake_phase(unit / "left", "muon", 2024, "left", retained=True)
            contract = {
                "phases": [
                    {"id": "parent", "parent": None},
                    {"id": "left", "parent": "parent"},
                    {"id": "right", "parent": "parent"},
                ]
            }
            R.retire_fork(unit, "parent", contract)
            self.assertTrue(Path(parent["checkpoint"]["path"]).is_file())
            fake_phase(unit / "right", "muon", 2024, "right", retained=True)
            R.retire_fork(unit, "parent", contract)
            self.assertFalse(Path(parent["checkpoint"]["path"]).exists())
            self.assertTrue((unit / "parent" / "checkpoint_retirement.json").is_file())

    def test_auc(self) -> None:
        rows = [
            {"step": 0, "loss": 4.0},
            {"step": 5, "loss": 3.0},
            {"step": 10, "loss": 2.0},
        ]
        self.assertAlmostEqual(A.normalized_auc(rows), 3.0)

    def test_frozen_classification(self) -> None:
        contract = {
            "analysis": {
                "practical_loss_margin": 0.002,
                "clear_selective_recovery_rule": "r",
                "persistent_muon_lead_rule": "p",
                "otherwise": "m",
            }
        }
        rows = [
            {
                "budget_id": "tokens_approximately_10b",
                "contrast": "down_none-minus-muon",
                "mean_difference": "-0.003",
                "negative_seeds": 2,
                "positive_seeds": 1,
            },
            {
                "budget_id": "tokens_approximately_10b",
                "contrast": "down_diag-minus-muon",
                "mean_difference": "0.001",
                "negative_seeds": 1,
                "positive_seeds": 2,
            },
        ]
        self.assertEqual(A.classify(rows, contract)["classification"], "clear_selective_recovery")


if __name__ == "__main__":
    unittest.main()
