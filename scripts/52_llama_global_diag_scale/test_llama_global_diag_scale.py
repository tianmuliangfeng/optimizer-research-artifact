#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import llama_global_diag_source_builder as builder
import run_llama_global_diag_suite as suite


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OFFICIAL_FIXTURE = (
    Path(os.environ.get("SNM_RESULTS_ROOT", REPO / "runs"))
    / "43_newton_muon_record28_275m"
    / "20260731T014352+0000/source_snapshot/upstream"
)
DATA_CERTIFICATE_FIXTURE = (
    Path(os.environ.get("SNM_RESULTS_ROOT", REPO / "runs"))
    / "43_newton_muon_record28_275m"
    / "20260731T014352+0000/preflight/data_certificate.json"
)


class LlamaGlobalDiagScaleTests(unittest.TestCase):
    def test_contract_and_controls(self) -> None:
        contract = json.loads((HERE / "llama_global_diag_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["experiment_id"], "52_llama_global_diag_scale")
        self.assertEqual(contract["formal"]["formal_units"], 6)
        self.assertEqual(contract["1b"]["mandatory_medium_steps"], 1000)
        self.assertTrue(contract["data"]["extended_10b_run_excluded"])
        self.assertEqual(contract["contract_version"], "2026-08-14.4")
        self.assertEqual(
            contract["data"]["accepted_full_content_fingerprint_sha256"],
            "1202c308d21ea690c17b958b98cbe40c65969a21928230950401f777adda8c68",
        )
        self.assertEqual(
            contract["data"]["historical_parent_controller_fingerprint"],
            "57d23fdc7fb4c59c3685b0c603deaeaa7d0be6b28c62a9d6a2aa493118ceb8a2",
        )
        self.assertEqual(
            hashlib.sha256((HERE / "frozen_llama_controls.csv").read_bytes()).hexdigest(),
            contract["frozen_controls"]["sha256"],
        )
        with (HERE / "frozen_llama_controls.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({(r["scale"], r["method"], r["seed"]) for r in rows}), 24)

    def test_source_derivation_is_deterministic_and_minimal(self) -> None:
        first = builder.build(REPO)
        second = builder.build(REPO)
        self.assertEqual(first, second)
        self.assertIn('NEWTON_METHODS = ("newton_full", "down_none", "down_diag", "global_diag")', first.trainer)
        self.assertEqual(first.trainer.count('"kind": "diag" if self.method == "global_diag" else "dense",'), 3)
        self.assertIn('torch.empty(1, 1) if global_diag', first.trainer)
        self.assertIn('METHOD_ORDER = ("global_diag",)', first.runner124)
        self.assertIn('os.environ.get("EX52_DATA_DIR"', first.runner124)
        self.assertIn('METHOD_ORDER = ("global_diag",)', first.runner1b)
        for name, source in (("trainer", first.trainer), ("runner124", first.runner124), ("runner1b", first.runner1b), ("wrapper1b", first.wrapper1b)):
            compile(source, f"<ex52-{name}>", "exec")

    def test_exact_k_state_bytes(self) -> None:
        self.assertEqual(builder.expected_k_state_bytes("124m"), 417_792)
        self.assertEqual(builder.expected_k_state_bytes("1b"), 1_677_312)
        self.assertLess(builder.expected_k_state_bytes("1b"), 2 * 1024 * 1024)

    def test_stage_specific_parent_configs_are_frozen(self) -> None:
        contract = json.loads((HERE / "llama_global_diag_contract.json").read_text(encoding="utf-8"))
        pilot_1b = suite.expected_controller_config(contract, "1b", "pilot")
        self.assertEqual(pilot_1b["num_iterations"], 34)
        self.assertEqual(pilot_1b["val_every"], 34)
        self.assertEqual(pilot_1b["val_tokens"], 8 * 1024)
        self.assertEqual(pilot_1b["checkpoint_every"], 0)
        pilot_124m = suite.expected_controller_config(contract, "124m", "pilot")
        self.assertEqual(pilot_124m["val_tokens"], 64 * 1024)
        self.assertEqual(pilot_124m["checkpoint_every"], 0)
        screen = suite.expected_controller_config(contract, "1b", "screen")
        self.assertEqual(screen["num_iterations"], 1000)
        self.assertEqual(screen["val_every"], 100)
        self.assertEqual(screen["val_tokens"], 10_485_760)
        self.assertEqual(screen["warmdown_iters"], 0)
        self.assertEqual(screen["checkpoint_every"], 128)
        formal = suite.expected_controller_config(contract, "1b", "formal")
        self.assertEqual(formal["num_iterations"], 6200)
        self.assertEqual(formal["warmdown_iters"], 1800)
        self.assertEqual(formal["checkpoint_every"], 128)
        legacy = json.loads(json.dumps(contract))
        legacy["1b"].pop("adamw_matrix_lr")
        self.assertEqual(
            suite.expected_controller_config(legacy, "1b", "pilot")["adamw_matrix_lr"],
            0.000576,
        )

    def test_parent_source_hashes_are_frozen(self) -> None:
        contract = json.loads((HERE / "llama_global_diag_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(builder.PARENT_SOURCE_SHA256, contract["parent_source_sha256"])
        self.assertEqual(
            builder.PUBLIC_PARENT_SOURCE_SHA256,
            contract["public_source_hashes"],
        )
        for relative, expected in builder.PARENT_SOURCE_SHA256.items():
            observed = hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
            self.assertIn(
                observed,
                {expected, builder.PUBLIC_PARENT_SOURCE_SHA256.get(relative, expected)},
            )

    @unittest.skipUnless(DATA_CERTIFICATE_FIXTURE.is_file(), "accepted full data certificate unavailable")
    def test_accepted_data_certificate_closes_fingerprint(self) -> None:
        contract = json.loads((HERE / "llama_global_diag_contract.json").read_text(encoding="utf-8"))
        certificate = json.loads(DATA_CERTIFICATE_FIXTURE.read_text(encoding="utf-8"))
        self.assertTrue(certificate["passed"])
        self.assertEqual(
            suite.data_fingerprint(certificate["files"]),
            contract["data"]["accepted_full_content_fingerprint_sha256"],
        )
        historical = suite.controller_inventory_payload(
            contract["data"]["historical_parent_data_dir"], certificate
        )
        self.assertEqual(
            historical["fingerprint"],
            contract["data"]["historical_parent_controller_fingerprint"],
        )
        active = suite.controller_inventory_payload(
            "/portable-official-r0/data/fineweb10B_ex52_frozen50",
            certificate,
        )
        self.assertEqual(
            active["fingerprint"],
            "062fe9fa6793c56d24e78a0318b02669848ee9039c8fe4e4bd7fba56e1e1e435",
        )
        self.assertNotEqual(active["fingerprint"], historical["fingerprint"])

    @unittest.skipUnless((OFFICIAL_FIXTURE / "triton_kernels.py").is_file(), "accepted r0 source fixture unavailable")
    def test_source_snapshot_is_reusable_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = Namespace(
                run_dir=root / "run",
                repo=REPO,
                official_repo=OFFICIAL_FIXTURE,
            )
            first = suite.source_snapshot(args)
            second = suite.source_snapshot(args)
            self.assertEqual(first, second)
            manifest = json.loads((first / "source_snapshot_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["passed"])
            self.assertEqual(manifest["contract_version"], "2026-08-14.4")
            target = first / manifest["files"][0]["path"]
            target.write_bytes(target.read_bytes() + b"tamper")
            with self.assertRaisesRegex(RuntimeError, "source snapshot drift"):
                suite.source_snapshot(args)

    @unittest.skipUnless((OFFICIAL_FIXTURE / "triton_kernels.py").is_file(), "accepted r0 source fixture unavailable")
    def test_v3_snapshot_gets_frozen_certificate_only_amendment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = Namespace(run_dir=root / "run", repo=REPO, official_repo=OFFICIAL_FIXTURE)
            snap = suite.source_snapshot(args)
            contract_path = snap / "scripts/52_llama_global_diag_scale/llama_global_diag_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["contract_version"] = "2026-08-14.3"
            suite.write_json(contract_path, contract)
            manifest_path = snap / "source_snapshot_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["contract_version"] = "2026-08-14.3"
            manifest["contract_sha256"] = suite.sha256_file(contract_path)
            relative = "scripts/52_llama_global_diag_scale/llama_global_diag_contract.json"
            row = next(item for item in manifest["files"] if item["path"] == relative)
            row["bytes"] = contract_path.stat().st_size
            row["sha256"] = suite.sha256_file(contract_path)
            suite.write_json(manifest_path, manifest)
            self.assertEqual(
                suite.source_snapshot(args), snap, "the frozen .3 training snapshot remains valid"
            )
            implementation = suite.controller_implementation(args, snap)
            self.assertEqual(implementation["kind"], "certificate_only_amendment")
            self.assertFalse(implementation["scientific_contract_changed"])
            amendment = json.loads(Path(implementation["amendment_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(amendment["scope"], "certificate_validation_only")
            self.assertFalse(amendment["training_artifacts_changed"])

    @unittest.skipUnless((OFFICIAL_FIXTURE / "triton_kernels.py").is_file(), "accepted r0 source fixture unavailable")
    def test_unit_certificate_binds_parent_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = Namespace(
                run_dir=root / "run",
                repo=REPO,
                official_repo=OFFICIAL_FIXTURE,
                data_dir=OFFICIAL_FIXTURE / "data/fineweb10B_ex52_frozen50",
            )
            snap = suite.source_snapshot(args)
            contract_path = snap / "scripts/52_llama_global_diag_scale/llama_global_diag_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            data_cert = args.run_dir / "preflight/data_certificate.json"
            data_cert.parent.mkdir(parents=True, exist_ok=True)
            files = [
                {
                    "path": str(args.data_dir / name),
                    "bytes": 200001024,
                    "num_tokens": 100000000,
                }
                for name in suite.EXPECTED_DATA_NAMES
            ]
            data_payload = {
                "passed": True,
                "fingerprint_sha256": contract["data"]["accepted_full_content_fingerprint_sha256"],
                "files": files,
            }
            data_cert.write_text(json.dumps(data_payload), encoding="utf-8")
            controller_inventory = suite.controller_inventory_payload(str(args.data_dir), data_payload)
            preflight = {
                "passed": True,
                "experiment_id": "52_llama_global_diag_scale",
                "official_repo": str(args.official_repo),
                "data_dir": str(args.data_dir),
                "contract_sha256": suite.sha256_file(contract_path),
                "source_snapshot_manifest_sha256": suite.sha256_file(snap / "source_snapshot_manifest.json"),
                "data_certificate_sha256": suite.sha256_file(data_cert),
                "data_fingerprint_sha256": contract["data"]["accepted_full_content_fingerprint_sha256"],
                "controller_inventory": controller_inventory,
                "controller_inventory_fingerprint": controller_inventory["fingerprint"],
            }
            suite.write_json(args.run_dir / "preflight/preflight_manifest.json", preflight)

            batch = args.run_dir / "pilot/124m/seed2024/batch"
            method = batch / "01_global_diag"
            method.mkdir(parents=True)
            snapshot = json.loads((snap / "source_snapshot_manifest.json").read_text(encoding="utf-8"))
            script_sha = snapshot["derived_source_sha256"]["scripts/17_llama_swiglu_validation/train_llama_swiglu.py"]
            init_sha = contract["accepted_init_sha256"]["124m"]["2024"]
            config = suite.expected_controller_config(contract, "124m", "pilot")
            common = {
                "status": "completed",
                "batch_kind": "smoke",
                "completed_methods": ["global_diag"],
                "failed_methods": [],
                "seed": 2024,
                "script_sha256": script_sha,
                "data_audit": controller_inventory,
                "init_audit": {"common_init_sha256": init_sha},
                "config": config,
            }
            suite.write_json(batch / "llama_manifest.json", common)
            suite.write_json(batch / "llama_plan.json", common)
            summary = {
                "status": "completed",
                "method": "global_diag",
                "seed": 2024,
                "completed_steps": 34,
                "tokens_seen": 34 * 512 * 1024,
                "init_sha256": init_sha,
                "final_val_loss": 4.0,
                "best_val_loss": 4.0,
                "final_train_loss": 4.1,
                "activation_scratch_bytes": 8,
                "k_state_bytes": 417792,
                "architecture": {
                    "parameter_count": 123551232,
                    "global_diag_route": True,
                    "preconditioner_group_count": 48,
                    "preconditioner_groups": [{"kind": "diag"} for _ in range(48)],
                },
            }
            suite.write_json(method / "summary.json", summary)
            with (method / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("event", "step", "loss", "tokens_seen"))
                writer.writeheader()
                writer.writerow({"event": "val", "step": 0, "loss": 5.0, "tokens_seen": 0})
                for step in range(1, 35):
                    writer.writerow({"event": "train", "step": step, "loss": 4.1, "tokens_seen": step * 512 * 1024})
                writer.writerow({"event": "val", "step": 34, "loss": 4.0, "tokens_seen": 34 * 512 * 1024})
            cert = suite.certify_unit(args, "pilot", "124m", 2024, "smoke", args.run_dir / "pilot/124m/seed2024")
            self.assertTrue(suite.validate_unit_certificate(args, cert, "pilot", "124m", 2024, "smoke"))
            summary["final_val_loss"] = 3.9
            suite.write_json(method / "summary.json", summary)
            self.assertFalse(suite.validate_unit_certificate(args, cert, "pilot", "124m", 2024, "smoke"))


if __name__ == "__main__": unittest.main()
