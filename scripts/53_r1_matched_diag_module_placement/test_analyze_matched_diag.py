#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_matched_diag import ARMS, FORMAL_SEEDS, analyze, read_json
from matched_diag_source_builder import ARM_CONFIGS, EXPECTED_MEMORY


EFFECTS = {
    "all_none": 0.0,
    "c_fc_diag": -0.004,
    "c_proj_diag": -0.003,
    "c_fc_c_proj_diag": -0.006,
    "o_proj_diag": -0.002,
}
OFFICIAL_COMMIT = "df78af0db523d8bceb25af4919a3e3e7082b80f3"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def make_fixture(root: Path) -> None:
    contract_path = SCRIPT_DIR / "matched_diag_contract.json"
    contract_sha = sha256(contract_path)
    trainer_bytes = b"# frozen derived trainer\n"
    trainer_sha = hashlib.sha256(trainer_bytes).hexdigest()
    snapshot = root / "source_snapshot"
    snapshot_names = (
        "scripts/14_official_newton_muon_r0/run_official_newton_muon_r0.py",
        "scripts/15_official_newton_muon_r1/run_official_newton_muon_r1.py",
        "scripts/50_r1_global_activation_diag/global_diag_source_builder.py",
        "scripts/53_r1_matched_diag_module_placement/matched_diag_contract.json",
        "scripts/53_r1_matched_diag_module_placement/matched_diag_source_builder.py",
        "scripts/53_r1_matched_diag_module_placement/run_matched_diag.py",
        "scripts/53_r1_matched_diag_module_placement/run_matched_diag_suite.py",
        "scripts/53_r1_matched_diag_module_placement/analyze_matched_diag.py",
    )
    frozen_files = []
    for name in snapshot_names:
        frozen = snapshot / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(
            contract_path.read_bytes()
            if name.endswith("matched_diag_contract.json")
            else (name + "\n").encode()
        )
        frozen_files.append(frozen)
    snapshot_manifest = {
        "schema_version": 1,
        "experiment_id": "53_r1_matched_diag_module_placement",
        "passed": True,
        "file_count": len(frozen_files),
        "files": [
            {
                "path": frozen.relative_to(snapshot).as_posix(),
                "bytes": frozen.stat().st_size,
                "sha256": sha256(frozen),
            }
            for frozen in frozen_files
        ],
    }
    write_json(snapshot / "source_snapshot_manifest.json", snapshot_manifest)
    snapshot_sha = sha256(snapshot / "source_snapshot_manifest.json")

    data_dir = root / "fixture_data"
    data_dir.mkdir(parents=True)
    entries = []
    names = [f"fineweb_train_{index:06d}.bin" for index in range(1, 51)] + [
        "fineweb_val_000000.bin"
    ]
    for name in names:
        shard = data_dir / name
        shard.write_bytes((20240520).to_bytes(4, "little", signed=True) + name.encode())
        is_val = name.startswith("fineweb_val")
        entries.append(
            {
                "name": name,
                "split": "validation" if is_val else "train",
                "index": 0 if is_val else int(name[-10:-4]),
                "bytes": shard.stat().st_size,
                "mtime_ns": shard.stat().st_mtime_ns,
                "magic": 20240520,
                "sha256": sha256(shard),
            }
        )
    stable = [
        {key: item[key] for key in ("name", "split", "index", "bytes", "magic", "sha256")}
        for item in entries
    ]
    projection = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    inventory = {
        "schema_version": 1,
        "experiment_id": "53_r1_matched_diag_module_placement",
        "passed": True,
        "data_dir": str(data_dir.resolve()),
        "required_train_shards": 50,
        "required_validation_shards": 1,
        "entries": entries,
        "content_projection_sha256": projection,
    }
    write_json(root / "data_inventory.json", inventory)
    inventory_sha = sha256(root / "data_inventory.json")
    write_json(
        root / "data_verify_receipt.json",
        {
            "passed": True,
            "full_content_rehash": True,
            "data_inventory_sha256": inventory_sha,
            "data_content_projection_sha256": projection,
        },
    )
    write_json(
        root / "preflight_manifest.json",
        {
            "passed": True,
            "contract_sha256": contract_sha,
            "source_snapshot_manifest_sha256": snapshot_sha,
            "data_inventory_sha256": inventory_sha,
            "data_content_projection_sha256": projection,
            "derived_script_sha256": trainer_sha,
        },
    )
    data_record = {
        "data_dir": str(data_dir.resolve()),
        "train_shards": 50,
        "validation_shards": 1,
        "first_train_shard": "fineweb_train_000001.bin",
        "first_validation_shard": "fineweb_val_000000.bin",
        "data_magic": 20240520,
    }
    runtime = {
        "python_executable": "/fixture/python",
        "python_version": "3.10.12",
        "numpy": "2.2.6",
        "torch": "2.8.0+cu126",
        "torch_cuda": "12.6",
        "triton": "3.4.0",
        "triton_module": "/fixture/triton",
        "triton_kernels_module": "/fixture/triton_kernels.py",
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_total_memory_bytes": 85169143808,
    }
    pilot_paths = {}
    pilot_hashes = {}
    for arm in ARMS:
        batch_dir = root / "pilot" / "seed2053" / arm / "batch"
        run = batch_dir / "run"
        workspace = run / "workspace"
        workspace.mkdir(parents=True)
        (workspace / f"train_r1_{arm}.py").write_bytes(trainer_bytes)
        pilot_summary = {
            "method": arm,
            "controlled_seed": 2053,
            "formal_evidence": False,
            "evidence_valid": True,
            "init_sha256": "f" * 64,
            "derived_script_sha256": trainer_sha,
            "run_name": "run",
            "quality_usable": False,
            "memory_usable": True,
            "timing_usable": False,
            "outcome_eligible": False,
            "configuration_selection_allowed": False,
        }
        write_json(run / "r1_summary.json", pilot_summary)
        write_json(
            run / "run_manifest.json",
            {
                "status": "completed_valid_smoke",
                "experiment_family": "53_r1_matched_diag_module_placement",
                "protocol": "r1_matched_diag_module_placement_engineering_pilot",
                "method": arm,
                "controlled_seed": 2053,
                "formal_evidence": False,
                "source": {"derived_script_sha256": trainer_sha},
            },
        )
        write_json(
            run / "source_manifest.json",
            {"derived_script_sha256": trainer_sha},
        )
        batch = {
            "family": "53_r1_matched_diag_module_placement",
            "protocol": "r1_matched_diag_module_placement_engineering_pilot",
            "batch_kind": "smoke",
            "status": "completed_valid_smoke",
            "official_commit": OFFICIAL_COMMIT,
            "methods": [arm],
            "seed": 2053,
            "failures": [],
            "formal_evidence": False,
            "evidence_profile": "exact_shape_numerical_smoke",
            "smoke_steps": 34,
            "derived_source_sha256": {arm: trainer_sha},
            "resource_isolation": {"one_process_one_gpu": True, "visible_device_count": 1},
            "initialization_audit": {
                "seed": 2053,
                "all_methods_identical": True,
                "init_sha256": "f" * 64,
            },
            "training_runtime_fingerprint": runtime,
            "data": data_record,
            "smoke_certificate": None,
            "summaries": [pilot_summary],
        }
        write_json(batch_dir / "r1_manifest.json", batch)
        relative = (batch_dir / "r1_manifest.json").relative_to(root).as_posix()
        pilot_paths[arm] = relative
        pilot_hashes[arm] = sha256(batch_dir / "r1_manifest.json")
    write_json(
        root / "pilot_manifest.json",
        {
            "passed": True,
            "seed": 2053,
            "steps": 34,
            "outcome_eligible": False,
            "configuration_selection_allowed": False,
            "arms": list(ARMS),
            "accepted_batches": pilot_paths,
            "data_inventory_sha256": inventory_sha,
            "source_snapshot_manifest_sha256": snapshot_sha,
        },
    )

    formal_paths = {}
    for seed_index, seed in enumerate(FORMAL_SEEDS):
        init_sha = f"{seed:064x}"[-64:]
        for arm in ARMS:
            batch_dir = root / "formal" / f"seed{seed}" / arm / "batch"
            unit = batch_dir / "run"
            unit.mkdir(parents=True, exist_ok=True)
            workspace = unit / "workspace"
            workspace.mkdir()
            (workspace / f"train_r1_{arm}.py").write_bytes(trainer_bytes)
            checkpoint = workspace / "logs/state_step06200.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(f"checkpoint-{seed}-{arm}".encode())
            final = 3.3 + seed_index * 0.001 + EFFECTS[arm]
            config = ARM_CONFIGS[arm]
            memory = EXPECTED_MEMORY[arm]
            summary = {
                "method": arm,
                "controlled_seed": seed,
                "formal_evidence": True,
                "evidence_valid": True,
                "cfc_k_mode": config["c_fc"],
                "cproj_k_mode": config["c_proj"],
                "oproj_k_mode": config["o_proj"],
                "qkv_k_mode": config["qkv"],
                "final_val_loss": final,
                "best_val_loss": final - 0.001,
                "final_val_step": 6200,
                "init_sha256": init_sha,
                "derived_script_sha256": trainer_sha,
                "optimizer_state_bytes": 1_000_000 + memory["k_state_bytes"],
                "peak_memory_allocated_mib": 3000,
                "base_learning_rate": 0.004,
                "matrix_learning_rate": 0.0004,
                "final_train_step": 6200,
                "train_points": 6200,
                "validation_points": 63,
                "run_name": "run",
                "checkpoint_relative_path": checkpoint.relative_to(unit).as_posix(),
                "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                **memory,
            }
            write_json(unit / "r1_summary.json", summary)
            write_json(
                unit / "run_manifest.json",
                {
                    "status": "completed_valid",
                    "experiment_family": "53_r1_matched_diag_module_placement",
                    "protocol": "r1_matched_diag_module_placement_formal",
                    "method": arm,
                    "controlled_seed": seed,
                    "formal_evidence": True,
                    "source": {"derived_script_sha256": trainer_sha},
                    "wandb": {"status": "failed"},
                },
            )
            write_json(
                unit / "source_manifest.json",
                {"derived_script_sha256": trainer_sha},
            )
            metrics = []
            for step in range(1, 6201):
                metrics.append(
                    {
                        "method": arm,
                        "event": "train",
                        "step": step,
                        "total_steps": 6200,
                        "tokens_seen": step * 524288,
                        "loss": final + 0.1,
                    }
                )
            for step in range(0, 6201, 100):
                metrics.append(
                    {
                        "method": arm,
                        "event": "validation",
                        "step": step,
                        "total_steps": 6200,
                        "tokens_seen": step * 524288,
                        "loss": final + (6200 - step) * 0.00001,
                    }
                )
            write_csv(unit / "r1_metrics.csv", metrics)
            input_count = (12 if config["c_fc"] == "diag" else 0) + (
                12 if config["o_proj"] == "diag" else 0
            )
            proj_count = 12 if config["c_proj"] == "diag" else 0
            (unit / "training_stdout.log").write_text(
                "R1_MATCHED_DIAG_METADATA "
                f"arm={arm} cfc={config['c_fc']} cproj={config['c_proj']} "
                f"oproj={config['o_proj']} qkv=none dense_workspace=0\n"
                "R1_MATCHED_DIAG_ROUTE "
                f"arm={arm} input_diag_params={input_count} "
                f"proj_diag_params={proj_count} dense_refresh_blocks=0\n",
                encoding="utf-8",
            )
            (unit / "training_log_with_source.txt").write_text("fixture\n", encoding="utf-8")
            batch = {
                "family": "53_r1_matched_diag_module_placement",
                "protocol": "r1_matched_diag_module_placement_formal",
                "batch_kind": "formal",
                "status": "completed_valid_local_wandb_incomplete",
                "official_commit": OFFICIAL_COMMIT,
                "methods": [arm],
                "seed": seed,
                "failures": [],
                "formal_evidence": True,
                "evidence_profile": "formal",
                "smoke_steps": None,
                "derived_source_sha256": {arm: trainer_sha},
                "resource_isolation": {"one_process_one_gpu": True, "visible_device_count": 1},
                "initialization_audit": {
                    "seed": seed,
                    "all_methods_identical": True,
                    "init_sha256": init_sha,
                },
                "training_runtime_fingerprint": runtime,
                "data": data_record,
                "smoke_certificate": {
                    "validated": True,
                    "engineering_seed": 2053,
                    "formal_seed_independent": True,
                    "outcome_eligible": False,
                    "manifest_sha256": pilot_hashes[arm],
                },
                "summaries": [summary],
            }
            write_json(batch_dir / "r1_manifest.json", batch)
            formal_paths[f"seed{seed}/{arm}"] = (
                batch_dir / "r1_manifest.json"
            ).relative_to(root).as_posix()
    write_json(
        root / "formal_manifest.json",
        {
            "passed": True,
            "formal_seeds": list(FORMAL_SEEDS),
            "arms": list(ARMS),
            "formal_units": 15,
            "accepted_batches": formal_paths,
            "data_inventory_sha256": inventory_sha,
            "source_snapshot_manifest_sha256": snapshot_sha,
            "wandb_required_for_scientific_validity": False,
            "timing_usable": False,
        },
    )


class MatchedDiagAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "run"
        self.output = self.run_dir / "analysis"
        make_fixture(self.run_dir)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_analysis(self) -> dict[str, object]:
        return analyze(self.run_dir, SCRIPT_DIR / "matched_diag_contract.json", self.output)

    def test_complete_fixture_passes(self) -> None:
        manifest = self.run_analysis()
        self.assertTrue(manifest["passed"])
        self.assertEqual(manifest["formal_units"], 15)
        self.assertEqual(manifest["descriptive_lowest_mean_arm"], "c_fc_c_proj_diag")

    def test_all_required_outputs_exist(self) -> None:
        self.run_analysis()
        expected = {
            "formal_results.csv",
            "method_summary.csv",
            "paired_contrasts_by_seed.csv",
            "aggregate_contrasts.csv",
            "factorial_effects_by_seed.csv",
            "factorial_effects_aggregate.csv",
            "EXPERIMENT_53_ANALYSIS.md",
            "analysis_manifest.json",
        }
        self.assertTrue(expected.issubset({path.name for path in self.output.iterdir()}))

    def test_factorial_effects_are_exact(self) -> None:
        self.run_analysis()
        with (self.output / "factorial_effects_by_seed.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertTrue(math.isclose(float(row["c_fc_main_effect"]), -0.0035, abs_tol=1e-12))
            self.assertTrue(math.isclose(float(row["c_proj_main_effect"]), -0.0025, abs_tol=1e-12))
            self.assertTrue(math.isclose(float(row["factorial_interaction"]), 0.001, abs_tol=1e-12))

    def test_paired_contrast_sign_is_a_minus_b(self) -> None:
        self.run_analysis()
        with (self.output / "aggregate_contrasts.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        fc = next(row for row in rows if row["contrast"] == "c_fc_diag_minus_all_none")
        self.assertTrue(math.isclose(float(fc["mean"]), -0.004, abs_tol=1e-12))
        self.assertEqual(int(fc["negative_seed_count"]), 3)

    def test_missing_unit_fails_closed(self) -> None:
        target = next((self.run_dir / "formal/seed2024/all_none").glob("**/r1_summary.json"))
        target.unlink()
        with self.assertRaisesRegex(RuntimeError, "missing Experiment-53 accepted run artifacts"):
            self.run_analysis()

    def test_workspace_drift_fails_closed(self) -> None:
        target = next((self.run_dir / "formal/seed2024/c_fc_diag").glob("**/r1_summary.json"))
        payload = read_json(target)
        payload["precond_workspace_bytes"] = 4
        target.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "batch/file summary mismatch"):
            self.run_analysis()

    def test_unpaired_initialization_fails_closed(self) -> None:
        target = next((self.run_dir / "formal/seed2024/o_proj_diag").glob("**/r1_summary.json"))
        payload = read_json(target)
        payload["init_sha256"] = "b" * 64
        target.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "batch/file summary mismatch"):
            self.run_analysis()

    def test_nonfinite_loss_fails_closed(self) -> None:
        target = next((self.run_dir / "formal/seed2024/o_proj_diag").glob("**/r1_summary.json"))
        payload = read_json(target)
        payload["final_val_loss"] = float("nan")
        target.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.run_analysis()

    def test_checkpoint_tamper_fails_closed(self) -> None:
        summary_path = next(
            (self.run_dir / "formal/seed2024/c_proj_diag").glob("**/r1_summary.json")
        )
        summary = read_json(summary_path)
        checkpoint = summary_path.parent / summary["checkpoint_relative_path"]
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
        with self.assertRaisesRegex(RuntimeError, "checkpoint integrity failure"):
            self.run_analysis()

    def test_runtime_route_certificate_tamper_fails_closed(self) -> None:
        stdout = next(
            (self.run_dir / "formal/seed2025/o_proj_diag").glob(
                "**/training_stdout.log"
            )
        )
        stdout.write_text("R1_MATCHED_DIAG_ROUTE arm=o_proj_diag input_diag_params=0\n")
        with self.assertRaisesRegex(RuntimeError, "route certificate failed"):
            self.run_analysis()


if __name__ == "__main__":
    unittest.main(verbosity=2)
