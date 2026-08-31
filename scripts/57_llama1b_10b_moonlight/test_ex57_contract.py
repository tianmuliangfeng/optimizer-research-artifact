from pathlib import Path
import json
import unittest
from unittest import mock
import argparse
import os
import subprocess
import tempfile
import time
import threading

import runtime as E

import protocol as P
import run_suite as R
import source_builder as S

HERE = Path(__file__).resolve().parent


class EX57MoonlightContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = json.loads((HERE / "ex57_contract.json").read_text())

    def test_contract_is_valid(self) -> None:
        checks = P.validate_contract(self.contract)
        self.assertTrue(all(checks.values()), checks)

    def test_independent_from_ex54(self) -> None:
        source = (HERE / "run_suite.py").read_text()
        runtime = (HERE / "runtime.py").read_text()
        long_worker = (HERE / "long_worker.py").read_text()
        self.assertNotIn('import run_suite as E', source)
        self.assertNotIn('scripts/54_', source)
        self.assertNotIn('scripts/54_', runtime)
        self.assertIn("base_trainer_sha256", long_worker)
        self.assertIn(
            '"scripts/17_llama_swiglu_validation/train_llama_swiglu.py"',
            long_worker,
        )
        self.assertTrue(self.contract["formal"]["independent_of_ex54"])
        self.assertEqual(R.DEVICE_BATCH_SIZE_1B, 8)
        self.assertEqual(R.GRADIENT_ACCUMULATION_STEPS_1B, 64)

    def test_builtin_v3_long_worker_is_not_repatched_as_legacy_v2(self) -> None:
        source = (HERE / "long_worker.py").read_text()
        self.assertIn(E.LONG_WORKER_PROFILE_COMPAT_SOURCE_MARKER, source)
        self.assertEqual(E.patch_long_worker_profile_adapter(source), source)

        incomplete = source.replace(
            'payload["accepted_sources"] = accepted_sources',
            '# deliberately removed accepted-source binding',
            1,
        )
        with self.assertRaisesRegex(RuntimeError, "V3.*incomplete"):
            E.patch_long_worker_profile_adapter(incomplete)

    def test_three_formal_endpoints(self) -> None:
        self.assertEqual(
            [phase["budget_id"] for phase in P.endpoint_phases(self.contract)],
            ["tokens_3p2506b", "tokens_6p9694b", "tokens_approximately_10b"],
        )
        self.assertEqual(self.contract["execution"]["physical_gpus"], [0, 1, 2])

    def test_optimizer_has_no_mousse_factor_path(self) -> None:
        source = (HERE / "moonlight_optimizer.py").read_text()
        self.assertIn("class R1MoonlightMuon", source)
        self.assertIn("logical_matrix_slices", source)
        self.assertNotIn("torch.linalg.eigh", source)
        self.assertNotIn("factor_epsilon", source)


    def test_runtime_probe_preserves_inherited_pythonpath(self) -> None:
        captured = {}

        def fake_run(command, *, env, text, capture_output):
            captured["command"] = command
            captured["env"] = env
            return subprocess.CompletedProcess(
                command, 0,
                stdout='{"passed": true, "checks": {}, "observed": {}}\n',
                stderr="",
            )

        args = argparse.Namespace(
            training_python=Path("/frozen/runtime/bin/python"),
            official_repo=Path("/official/repo"),
            gpus=[0, 1, 2],
        )
        sentinel = "/frozen/runtime/site-packages:/existing/path"
        with mock.patch.dict(os.environ, {"PYTHONPATH": sentinel}, clear=False):
            with mock.patch.object(E.subprocess, "run", side_effect=fake_run):
                result = E.runtime_probe(args, Path("/snapshot"), self.contract)
        self.assertTrue(result["passed"])
        self.assertEqual(captured["env"]["PYTHONPATH"], sentinel)
        self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "0,1,2")

    def test_analysis_env_prepends_without_clobbering(self) -> None:
        sentinel = "/frozen/runtime/site-packages:/existing/path"
        with mock.patch.dict(os.environ, {"PYTHONPATH": sentinel}, clear=False):
            package = Path("/snapshot/package")
            env = R.inherited_pythonpath_env(package)
        self.assertEqual(
            env["PYTHONPATH"],
            os.pathsep.join([str(package.absolute()), sentinel]),
        )


    @unittest.skipIf(os.name == "nt", "Windows developer mode is not guaranteed")
    def test_training_python_path_preserves_virtualenv_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            system_python = root / "usr/bin/python3.10"
            system_python.parent.mkdir(parents=True)
            system_python.touch()
            venv_python = root / "venv/bin/python"
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(system_python)

            observed = R.absolute_without_resolving_symlinks(venv_python)

            self.assertEqual(observed, venv_python.absolute())
            self.assertNotEqual(observed, system_python.resolve())
            self.assertTrue(observed.is_symlink())

    def test_training_python_path_is_lexical_and_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "venv/bin/python"
            observed = R.absolute_without_resolving_symlinks(path)
            self.assertEqual(observed, path.absolute())

    def test_moonlight_algorithm_is_exact_ex19_transfer(self) -> None:
        optimizer = (HERE / "moonlight_optimizer.py").read_text()
        receipt = S.audit_moonlight_transfer(HERE.parents[1], optimizer)
        self.assertTrue(receipt["passed"])
        self.assertEqual(
            receipt["reference_sha256"],
            "bf39d7e1b435ef737833046c564ce8770d858d1aa474c9d7f11a914057253655",
        )
        with self.assertRaises(RuntimeError):
            S.audit_moonlight_transfer(
                HERE.parents[1],
                optimizer.replace(
                    "orthogonal.mul_(0.2 * math.sqrt",
                    "orthogonal.mul_(0.21 * math.sqrt",
                    1,
                ),
            )

    def test_controller_never_resolves_training_python_symlink(self) -> None:
        source = (HERE / "run_suite.py").read_text()
        self.assertNotIn(
            "args.training_python = args.training_python.resolve()", source
        )


    def test_tuning_seed_is_disjoint_from_formal_seeds(self) -> None:
        tuning_seed = int(self.contract["tuning"]["1b"]["seed"])
        formal = {int(seed) for seed in self.contract["formal"]["seeds"]}
        self.assertEqual(tuning_seed, 5701)
        self.assertNotIn(tuning_seed, formal)
        self.assertTrue(self.contract["fairness"]["tuning_formal_seed_disjoint"])

    def test_scheduler_uses_all_three_gpus_for_three_jobs(self) -> None:
        rows = E.schedule(
            ["lr0010", "lr0018", "lr0030"],
            [0, 1, 2],
            lambda item, gpu: {"cell": {"id": item}, "seen_gpu": gpu},
        )
        self.assertEqual(sorted(row["physical_gpu"] for row in rows), [0, 1, 2])
        self.assertEqual([row["seen_gpu"] for row in rows], [0, 1, 2])


    def test_per_gpu_compile_cache_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            env0 = E.bind_gpu_compile_cache({}, run_dir, 0)
            env1 = E.bind_gpu_compile_cache({}, run_dir, 1)
            env2 = E.bind_gpu_compile_cache({}, run_dir, 2)
            self.assertNotEqual(env0["TORCHINDUCTOR_CACHE_DIR"], env1["TORCHINDUCTOR_CACHE_DIR"])
            self.assertNotEqual(env1["TORCHINDUCTOR_CACHE_DIR"], env2["TORCHINDUCTOR_CACHE_DIR"])
            self.assertTrue(env0["TORCHINDUCTOR_CACHE_DIR"].endswith("gpu0"))
            self.assertTrue(env1["TORCHINDUCTOR_CACHE_DIR"].endswith("gpu1"))
            self.assertTrue(env2["TORCHINDUCTOR_CACHE_DIR"].endswith("gpu2"))
            self.assertEqual(env0["EX57_EXPECTED_PHYSICAL_GPU"], "0")
            self.assertEqual(env1["EX57_EXPECTED_PHYSICAL_GPU"], "1")
            self.assertEqual(env2["EX57_EXPECTED_PHYSICAL_GPU"], "2")

    def test_legacy_v2_contract_compile_cache_compatibility_receipt(self) -> None:
        legacy_fixture = HERE / "legacy_ex57_contract_fair_parallel_v2.json"
        self.assertEqual(
            P.sha256_file(legacy_fixture),
            R.LEGACY_V2_SOURCE_CONTRACT_SHA256,
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            source_contract = (
                run_dir
                / "source_snapshot"
                / R.PACKAGE_REL
                / R.CONTRACT_NAME
            )
            source_contract.parent.mkdir(parents=True, exist_ok=True)
            source_contract.write_bytes(legacy_fixture.read_bytes())
            legacy = P.read_json(source_contract)
            before = P.sha256_file(source_contract)
            args = argparse.Namespace(run_dir=run_dir, gpus=[0, 1, 2])

            resolved = R.resolve_compile_cache_policy(args, legacy)
            self.assertEqual(resolved["policy"], "per_physical_gpu")
            self.assertEqual(resolved["source"], "legacy_v2_execution_amendment")
            self.assertEqual(P.sha256_file(source_contract), before)

            receipt_path = Path(resolved["receipt_path"])
            receipt = P.read_json(receipt_path)
            self.assertTrue(receipt["passed"])
            self.assertTrue(receipt["scientific_protocol_unchanged"])
            self.assertFalse(receipt["timing_eligible"])
            self.assertEqual(
                receipt["source_snapshot_contract_sha256"], before
            )
            self.assertTrue(all(receipt["checks"].values()))
            self.assertTrue(R.compile_cache_policy_replay_valid(args, legacy))

            replay = R.resolve_compile_cache_policy(args, legacy)
            self.assertEqual(replay, resolved)

    def test_explicit_compile_cache_policy_needs_no_amendment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            args = argparse.Namespace(run_dir=run_dir, gpus=[0, 1, 2])
            resolved = R.resolve_compile_cache_policy(args, self.contract)
            self.assertEqual(resolved["policy"], "per_physical_gpu")
            self.assertEqual(resolved["source"], "frozen_contract")
            self.assertIsNone(resolved["receipt_path"])
            self.assertFalse((run_dir / R.COMPILE_CACHE_COMPAT_RECEIPT).exists())

    def test_legacy_compatibility_rejects_unknown_source_hash(self) -> None:
        legacy_fixture = HERE / "legacy_ex57_contract_fair_parallel_v2.json"
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            source_contract = (
                run_dir
                / "source_snapshot"
                / R.PACKAGE_REL
                / R.CONTRACT_NAME
            )
            source_contract.parent.mkdir(parents=True, exist_ok=True)
            legacy = json.loads(legacy_fixture.read_text())
            legacy["execution"]["note"] += " changed"
            P.atomic_json(source_contract, legacy)
            args = argparse.Namespace(run_dir=run_dir, gpus=[0, 1, 2])
            with self.assertRaises(RuntimeError):
                R.resolve_compile_cache_policy(args, legacy)

    def test_controller_does_not_direct_index_optional_compile_cache_metadata(self) -> None:
        source = (HERE / "run_suite.py").read_text()
        self.assertNotIn(
            'contract["execution"]["compile_cache_policy"]', source
        )

    def test_scheduler_starts_three_jobs_concurrently(self) -> None:
        starts = {}
        lock = threading.Lock()

        def worker(item, gpu):
            with lock:
                starts[gpu] = time.monotonic()
            time.sleep(0.20)
            return {"cell": {"id": item}, "seen_gpu": gpu}

        begin = time.monotonic()
        rows = E.schedule(["lr0010", "lr0018", "lr0030"], [0, 1, 2], worker)
        elapsed = time.monotonic() - begin
        self.assertEqual(sorted(starts), [0, 1, 2])
        self.assertLess(max(starts.values()) - min(starts.values()), 0.15)
        self.assertLess(elapsed, 0.50)
        self.assertEqual([row["seen_gpu"] for row in rows], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
