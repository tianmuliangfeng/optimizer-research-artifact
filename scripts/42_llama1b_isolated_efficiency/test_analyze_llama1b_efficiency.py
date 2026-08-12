#!/usr/bin/env python3
"""CPU-only synthetic integrity tests for the experiment-42 analyzer."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import analyze_llama1b_efficiency as analysis
import source_builder


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONTRACT_SOURCE = HERE / "efficiency_contract.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path, run_dir: Path) -> str:
    return path.relative_to(run_dir).as_posix()


def certificate() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "script_version": analysis.CERTIFIER_VERSION,
        "passed": True,
        "required_gpus": [0, 1],
        "active_compute_processes": [],
        "gpus": [
            {
                "index": index,
                "uuid": f"GPU-SYNTHETIC-{index}",
                "name": "NVIDIA H100 80GB HBM3",
                "free_fraction": 0.99,
            }
            for index in (0, 1)
        ],
        "checks": {
            "exact_gpu_inventory": True,
            "gpu_names": True,
            "minimum_free_fraction": True,
            "active_compute_processes_absent": True,
        },
    }


def data_audit() -> dict[str, Any]:
    rows = [
        {
            "name": f"fineweb_train_{index:06d}.bin",
            "bytes": 200_000_012,
            "tokens": 100_000_000,
            "sha256": f"{index + 1:064x}",
        }
        for index in range(50)
    ]
    rows.append(
        {
            "name": "fineweb_val_000000.bin",
            "bytes": 20_000_012,
            "tokens": 10_000_000,
            "sha256": "f" * 64,
        }
    )
    payload = {
        "data_dir": "/synthetic/fineweb10B",
        "train_shard_count": 50,
        "validation_shard_count": 1,
        "files": rows,
        "total_tokens_in_train_headers": 5_000_000_000,
        "total_bytes": sum(row["bytes"] for row in rows),
    }
    payload["fingerprint"] = analysis.canonical_json_sha256(payload)
    return payload


def snapshot_sources(run_dir: Path, contract: dict[str, Any]) -> dict[str, Any]:
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir(parents=True)
    bundle = source_builder.build_source_bundle()
    materialized = source_builder.materialize(bundle, snapshot)
    files: dict[str, str] = {}
    for name in (
        "efficiency_common.py",
        "source_builder.py",
        "llama1b_efficiency_worker.py",
        "gpu_isolation_monitor.py",
        "run_llama1b_efficiency.py",
        "analyze_llama1b_efficiency.py",
    ):
        shutil.copy2(HERE / name, snapshot / name)
        files[name] = sha256_file(snapshot / name)
    certifier = HERE / "certify_exclusive_node.py"
    shutil.copy2(certifier, snapshot / "certify_exclusive_node.py")
    files["certify_exclusive_node.py"] = sha256_file(
        snapshot / "certify_exclusive_node.py"
    )
    shutil.copy2(run_dir / "efficiency_contract.json", snapshot / "efficiency_contract.json")
    files["efficiency_contract.json"] = sha256_file(
        snapshot / "efficiency_contract.json"
    )
    files.update(
        {
            "train_llama_swiglu_efficiency_base.py": materialized["derived_base"],
            "train_llama_swiglu_1b.py": materialized["profile_wrapper"],
            "train_llama_swiglu_efficiency_base.diff": materialized["source_diff"],
        }
    )
    source_contract = contract["source_contract"]
    return {
        "base_trainer_sha256": source_contract["base_trainer_sha256"],
        "profile_wrapper_source_sha256": source_contract[
            "profile_wrapper_sha256"
        ],
        "derived_base_sha256": source_contract[
            "derived_efficiency_base_sha256"
        ],
        "profile_wrapper_sha256": source_contract["profile_wrapper_sha256"],
        "source_diff_sha256": materialized["source_diff"],
        "files": files,
    }


def build_synthetic_run(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True)
    shutil.copy2(CONTRACT_SOURCE, run_dir / "efficiency_contract.json")
    contract_path = run_dir / "efficiency_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_sha = sha256_file(contract_path)
    source_audit = snapshot_sources(run_dir, contract)
    runtime = {
        "python_executable": "/synthetic/venv/bin/python",
        "python_version": [3, 10, 12],
        "numpy": "2.1.0",
        "torch": "2.8.0",
        "torch_cuda": "12.6",
        "triton": "3.4.0",
        "triton_kernels_sha256": contract["source_contract"][
            "triton_kernels_sha256"
        ],
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_total_memory_bytes": 85_056_864_256,
        "gpu_capability": [9, 0],
    }
    runtime_fingerprint = analysis.canonical_json_sha256(runtime)
    data = data_audit()
    init_sha = contract["frozen_configuration"]["initialization_sha256"]
    init_methods = {
        method: {
            "method": method,
            "seed": contract["frozen_configuration"]["seed"],
            "init_sha256": init_sha,
        }
        for method in contract["method_order"]
    }
    init_audit = {
        "common_init_sha256": init_sha,
        "methods": init_methods,
        "fingerprint": analysis.canonical_json_sha256(init_methods),
    }
    gpu_rows = [
        {
            "index": index,
            "uuid": f"GPU-SYNTHETIC-{index}",
            "name": "NVIDIA H100 80GB HBM3",
            "memory_total_mib": 81_559,
            "driver_version": "synthetic",
        }
        for index in (0, 1)
    ]
    preflight_certificate = run_dir / "preflight_exclusive_node.json"
    write_json(preflight_certificate, certificate())
    preflight = {
        "schema_version": 1,
        "script_version": analysis.CONTROLLER_VERSION,
        "passed": True,
        "contract_sha256": contract_sha,
        "source_audit": source_audit,
        "official_repo_audit": {
            "passed": True,
            "commit": contract["source_contract"]["official_repo_commit"],
            "triton_kernels_sha256": contract["source_contract"][
                "triton_kernels_sha256"
            ],
        },
        "data_audit": data,
        "gpu_inventory": {
            "gpus": gpu_rows,
            "fingerprint": analysis.canonical_json_sha256(gpu_rows),
        },
        "runtime": runtime,
        "stable_runtime": runtime,
        "runtime_fingerprint": runtime_fingerprint,
        "init_audit": init_audit,
        "exclusive_node_certificate_sha256": sha256_file(preflight_certificate),
        "physical_timing_gpu": "0",
        "required_gpus": ["0", "1"],
    }
    preflight_path = run_dir / "preflight.json"
    write_json(preflight_path, preflight)
    preflight_sha = sha256_file(preflight_path)

    cell_paths: list[str] = []
    cell_hashes: dict[str, str] = {}
    derived_path = (
        run_dir / "source_snapshot" / "train_llama_swiglu_efficiency_base.py"
    )
    for repeat_index, order in enumerate(contract["execution_policy"]["orders"]):
        for position_index, method in enumerate(order):
            cell_dir = (
                run_dir / "formal" / f"repeat_{repeat_index}" / method
            )
            attempt = cell_dir / "attempt_001"
            trainer = attempt / "trainer"
            trainer.mkdir(parents=True)
            shutil.copy2(derived_path, trainer / "train_llama_swiglu_base.py")
            (trainer / "metrics.csv").write_text(
                "event,step\nsynthetic,1\n", encoding="utf-8"
            )
            (attempt / "terminal.log").write_text(
                "synthetic trainer log\n", encoding="utf-8"
            )
            step_ms = 100.0 + repeat_index + position_index / 10.0
            steady_train_s = step_ms * 512 / 1000.0
            summary = {
                "status": "completed",
                "method": method,
                "seed": 2026,
                "completed_steps": 544,
                "steady_steps": 512,
                "tokens_seen": 544 * 524_288,
                "resume_count": 0,
                "timing_comparable": True,
                "checkpoint_path": "",
                "peak_memory_stats_reset": True,
                "peak_reset_after_completed_step": 32,
                "timed_step_first": 33,
                "timed_step_last": 544,
                "steady_train_s": steady_train_s,
                "step_avg_ms": step_ms,
                "timed_training_peak_allocated_bytes": 20_000_000_000
                + position_index,
                "timed_training_peak_reserved_bytes": 21_000_000_000
                + position_index,
                "allocated_bytes_at_timing_reset": 10_000_000_000,
                "reserved_bytes_at_timing_reset": 11_000_000_000,
                "allocated_bytes_at_timed_end": 19_000_000_000,
                "reserved_bytes_at_timed_end": 20_000_000_000,
                "k_state_bytes": contract["expected_k_state_bytes"][method],
                "optimizer_state_bytes": contract["expected_state_bytes"][
                    "optimizer_state_bytes"
                ][method],
                "model_parameter_bytes": contract["expected_state_bytes"][
                    "model_parameter_bytes"
                ],
                "init_sha256": init_sha,
                "runtime": runtime,
                "architecture": {
                    "parameter_count": contract["frozen_configuration"][
                        "parameter_count"
                    ],
                    "base_trainer_sha256": source_audit["derived_base_sha256"],
                },
                "config": {
                    "num_iterations": 544,
                    "global_batch_size": 512,
                    "device_batch_size": 8,
                    "sequence_length": 1024,
                    "val_every": 544,
                    "val_tokens": 8192,
                    "warmdown_iters": 0,
                    "backup_lr": 0.0036,
                    "matrix_lr": 0.01,
                    "adamw_matrix_lr": 0.000576,
                    "checkpoint_every": 0,
                    "resume": "never",
                    "no_save_final": True,
                },
            }
            summary_path = trainer / "summary.json"
            write_json(summary_path, summary)
            before_path = attempt / "exclusive_before.json"
            after_path = attempt / "exclusive_after.json"
            write_json(before_path, certificate())
            write_json(after_path, certificate())
            monitor_checks = {
                key: True
                for key in (
                    "samples_present",
                    "query_errors_absent",
                    "inventory_stable",
                    "timing_gpu_present",
                    "idle_gpus_present",
                    "gpu_names",
                    "idle_gpu_processes_absent",
                    "at_most_one_timing_process_per_sample",
                    "single_timing_process_identity",
                )
            }
            monitor = {
                "script_version": analysis.GPU_MONITOR_VERSION,
                "passed": True,
                "checks": monitor_checks,
                "timing_gpu": 0,
                "idle_gpus": [1],
                "sample_count": 3,
                "sample_interval_seconds": 10.0,
                "timing_process_pids": [12_345],
                "gpu_index_to_uuid": {
                    "0": "GPU-SYNTHETIC-0",
                    "1": "GPU-SYNTHETIC-1",
                },
                "idle_process_events": [],
                "query_errors": [],
            }
            monitor_path = attempt / "gpu_isolation_monitor.json"
            write_json(monitor_path, monitor)
            observed = {
                key: summary[key]
                for key in (
                    "steady_train_s",
                    "steady_steps",
                    "step_avg_ms",
                    "timed_training_peak_allocated_bytes",
                    "timed_training_peak_reserved_bytes",
                    "k_state_bytes",
                    "optimizer_state_bytes",
                    "model_parameter_bytes",
                    "init_sha256",
                    "resume_count",
                    "timing_comparable",
                    "completed_steps",
                    "tokens_seen",
                )
            }
            observed.update(
                {
                    "steps_per_s": 1000.0 / step_ms,
                    "tokens_per_s": 524_288 * 1000.0 / step_ms,
                    "runtime_fingerprint": runtime_fingerprint,
                    "data_fingerprint": data["fingerprint"],
                }
            )
            worker = {
                "schema_version": 1,
                "script_version": analysis.WORKER_VERSION,
                "passed": True,
                "tier": "formal",
                "method": method,
                "repeat_index": repeat_index,
                "position_index": position_index,
                "contract_sha256": contract_sha,
                "preflight_sha256": preflight_sha,
                "summary_sha256": sha256_file(summary_path),
                "metrics_sha256": sha256_file(trainer / "metrics.csv"),
                "terminal_log_sha256": sha256_file(attempt / "terminal.log"),
                "derived_base_sha256": source_audit["derived_base_sha256"],
                "profile_wrapper_sha256": source_audit["profile_wrapper_sha256"],
                "trainer_local_base_sha256": sha256_file(
                    trainer / "train_llama_swiglu_base.py"
                ),
                "metric_audit": {
                    "train_rows": 544,
                    "validation_rows": 2,
                    "timed_rows": 512,
                    "timed_step_first": 33,
                    "timed_step_last": 544,
                    "timed_interval_sum_s": steady_train_s,
                },
                "observed": observed,
            }
            worker_path = attempt / "worker_manifest.json"
            write_json(worker_path, worker)
            cell = {
                "schema_version": 1,
                "script_version": analysis.CONTROLLER_VERSION,
                "passed": True,
                "tier": "formal",
                "repeat_index": repeat_index,
                "position_index": position_index,
                "method": method,
                "attempt": "attempt_001",
                "attempt_dir": relative(attempt, run_dir),
                "contract_sha256": contract_sha,
                "preflight_sha256": preflight_sha,
                "runtime_fingerprint": runtime_fingerprint,
                "data_fingerprint": data["fingerprint"],
                "common_init_sha256": init_sha,
                "worker_manifest": relative(worker_path, run_dir),
                "worker_manifest_sha256": sha256_file(worker_path),
                "trainer_summary": relative(summary_path, run_dir),
                "trainer_summary_sha256": sha256_file(summary_path),
                "exclusive_before": relative(before_path, run_dir),
                "exclusive_before_sha256": sha256_file(before_path),
                "exclusive_after": relative(after_path, run_dir),
                "exclusive_after_sha256": sha256_file(after_path),
                "gpu_isolation_monitor": relative(monitor_path, run_dir),
                "gpu_isolation_monitor_sha256": sha256_file(monitor_path),
                "observed": observed,
            }
            cell_path = cell_dir / "cell_manifest.json"
            write_json(cell_path, cell)
            cell_relative = relative(cell_path, run_dir)
            cell_paths.append(cell_relative)
            cell_hashes[cell_relative] = sha256_file(cell_path)

    smoke_path = run_dir / "smoke" / "smoke_manifest.json"
    write_json(smoke_path, {"passed": True, "synthetic": True})
    postflight_path = run_dir / "postflight_data_audit.json"
    write_json(postflight_path, data)
    execution = {
        "schema_version": 1,
        "script_version": analysis.CONTROLLER_VERSION,
        "status": "completed",
        "passed": True,
        "contract_sha256": contract_sha,
        "preflight_sha256": preflight_sha,
        "smoke_manifest": relative(smoke_path, run_dir),
        "smoke_manifest_sha256": sha256_file(smoke_path),
        "formal_cell_manifests": cell_paths,
        "formal_cell_manifest_sha256": cell_hashes,
        "formal_cell_count": 16,
        "postflight_data_audit": relative(postflight_path, run_dir),
        "postflight_data_audit_sha256": sha256_file(postflight_path),
    }
    write_json(run_dir / "execution_manifest.json", execution)
    return contract_path


class AnalyzeLlamaEfficiencyTests(unittest.TestCase):
    def test_valid_matrix_and_three_tamper_classes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mech42_analysis_test_") as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            build_synthetic_run(baseline)
            monitor_tamper = root / "monitor_tamper"
            metrics_tamper = root / "metrics_tamper"
            preflight_tamper = root / "preflight_tamper"
            for destination in (monitor_tamper, metrics_tamper, preflight_tamper):
                shutil.copytree(baseline, destination)

            self.assertTrue(
                analysis.analyze(
                    baseline,
                    baseline / "efficiency_contract.json",
                    baseline / "analysis",
                )
            )
            manifest = json.loads(
                (
                    baseline
                    / "analysis"
                    / "llama1b_efficiency_analysis_manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["formal_rows"], 16)
            self.assertEqual(manifest["paired_contrast_rows"], 35)
            self.assertEqual(manifest["position_effect_rows"], 12)
            with (
                baseline / "analysis" / "paired_contrasts.csv"
            ).open(encoding="utf-8", newline="") as handle:
                pairs = list(csv.DictReader(handle))
            self.assertFalse(
                any(
                    {row["candidate"], row["reference"]}
                    == {"down_none", "down_diag"}
                    for row in pairs
                )
            )

            monitor_path = (
                monitor_tamper
                / "formal"
                / "repeat_0"
                / "muon"
                / "attempt_001"
                / "gpu_isolation_monitor.json"
            )
            monitor = json.loads(monitor_path.read_text(encoding="utf-8"))
            monitor["checks"]["idle_gpu_processes_absent"] = False
            write_json(monitor_path, monitor)
            self.assertFalse(
                analysis.analyze(
                    monitor_tamper,
                    monitor_tamper / "efficiency_contract.json",
                    monitor_tamper / "analysis_tampered",
                )
            )

            metrics_path = (
                metrics_tamper
                / "formal"
                / "repeat_0"
                / "muon"
                / "attempt_001"
                / "trainer"
                / "metrics.csv"
            )
            metrics_path.write_text("tampered\n", encoding="utf-8")
            self.assertFalse(
                analysis.analyze(
                    metrics_tamper,
                    metrics_tamper / "efficiency_contract.json",
                    metrics_tamper / "analysis_tampered",
                )
            )

            preflight_path = preflight_tamper / "preflight.json"
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            preflight["tampered"] = True
            write_json(preflight_path, preflight)
            self.assertFalse(
                analysis.analyze(
                    preflight_tamper,
                    preflight_tamper / "efficiency_contract.json",
                    preflight_tamper / "analysis_tampered",
                )
            )


if __name__ == "__main__":
    unittest.main()
