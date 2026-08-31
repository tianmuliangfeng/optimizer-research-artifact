from pathlib import Path
import json
import unittest
import tempfile
import time
import threading

import run_suite as R

HERE = Path(__file__).resolve().parent


class ControllerTest(unittest.TestCase):
    def test_ex54_is_non10b_and_independent(self) -> None:
        contract = json.loads((HERE / "ex54_contract.json").read_text())
        self.assertEqual(R.DEVICE_BATCH_SIZE_1B, 8)
        self.assertEqual(R.GRADIENT_ACCUMULATION_STEPS_1B, 64)
        self.assertEqual(R.TEN_B_PHASE_IDS, ())
        self.assertTrue(contract["formal"]["independent_of_ex57"])
        self.assertEqual([p["id"] for p in contract["phases"]], [
            "backbone_4400", "cooldown_6200", "backbone_11493", "cooldown_13293"
        ])


    def test_scheduler_uses_both_gpus(self) -> None:
        rows = R.schedule(
            list(range(6)),
            [0, 1],
            lambda item, gpu: {"item": item, "seen_gpu": gpu},
        )
        self.assertEqual({row["physical_gpu"] for row in rows}, {0, 1})
        self.assertEqual({row["seen_gpu"] for row in rows}, {0, 1})

    def test_contract_declares_parallel_single_gpu_jobs(self) -> None:
        contract = json.loads((HERE / "ex54_contract.json").read_text())
        self.assertEqual(contract["execution"]["tuning_parallel_workers"], 2)
        self.assertEqual(contract["execution"]["formal_parallel_workers"], 2)
        self.assertFalse(contract["execution"]["ddp"])


    def test_per_gpu_compile_cache_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            env0 = R.bind_gpu_compile_cache({}, run_dir, 0)
            env1 = R.bind_gpu_compile_cache({}, run_dir, 1)
            self.assertNotEqual(env0["TORCHINDUCTOR_CACHE_DIR"], env1["TORCHINDUCTOR_CACHE_DIR"])
            self.assertTrue(env0["TORCHINDUCTOR_CACHE_DIR"].endswith("gpu0"))
            self.assertTrue(env1["TORCHINDUCTOR_CACHE_DIR"].endswith("gpu1"))
            self.assertEqual(env0["EX54_EXPECTED_PHYSICAL_GPU"], "0")
            self.assertEqual(env1["EX54_EXPECTED_PHYSICAL_GPU"], "1")

    def test_scheduler_starts_two_jobs_concurrently(self) -> None:
        starts = {}
        lock = threading.Lock()

        def worker(item, gpu):
            with lock:
                starts[gpu] = time.monotonic()
            time.sleep(0.20)
            return {"item": item, "seen_gpu": gpu}

        begin = time.monotonic()
        rows = R.schedule([0, 1], [0, 1], worker)
        elapsed = time.monotonic() - begin
        self.assertEqual(sorted(starts), [0, 1])
        self.assertLess(max(starts.values()) - min(starts.values()), 0.15)
        self.assertLess(elapsed, 0.45)
        self.assertEqual({row["seen_gpu"] for row in rows}, {0, 1})

    def test_scheduler_preserves_persisted_physical_gpu_on_resume(self) -> None:
        persisted = {
            0: {"item": 0, "physical_gpu": 1},
            1: {"item": 1, "physical_gpu": 0},
        }
        rows = R.schedule(
            [0, 1], [0, 1], lambda item, gpu: dict(persisted[item])
        )
        self.assertEqual([row["physical_gpu"] for row in rows], [1, 0])

    def test_selection_resume_view_ignores_only_physical_gpu(self) -> None:
        existing = {
            "1b": {
                "selected_cell": {"id": "lr0018", "matrix_lr": 0.0018},
                "selected_loss": 2.5,
                "cells": [{
                    "cell": {"id": "lr0018", "matrix_lr": 0.0018},
                    "final_val_loss": 2.5,
                    "summary_sha256": "abc",
                    "physical_gpu": 0,
                }],
            }
        }
        replay = json.loads(json.dumps(existing))
        replay["1b"]["cells"][0]["physical_gpu"] = 1
        self.assertEqual(
            R.tuning_selection_scientific_view(existing),
            R.tuning_selection_scientific_view(replay),
        )

        changed = json.loads(json.dumps(replay))
        changed["1b"]["cells"][0]["final_val_loss"] = 2.6
        self.assertNotEqual(
            R.tuning_selection_scientific_view(existing),
            R.tuning_selection_scientific_view(changed),
        )


if __name__ == "__main__":
    unittest.main()
