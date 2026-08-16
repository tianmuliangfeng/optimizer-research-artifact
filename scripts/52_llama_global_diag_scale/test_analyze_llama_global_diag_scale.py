#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import analyze_llama_global_diag_scale as analyze


HERE = Path(__file__).resolve().parent


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


class AnalyzeLlamaGlobalDiagScaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract_path = HERE / "llama_global_diag_contract.json"
        self.contract = json.loads(self.contract_path.read_text(encoding="utf-8"))

    def make_cell(self, run: Path, scale: str, seed: int) -> Path:
        root = run / "formal" / scale / f"seed{seed}" / "batch"
        method = root / "01_global_diag"
        method.mkdir(parents=True)
        raw_manifest = root / "llama_manifest.json"
        raw_plan = root / "llama_plan.json"
        summary_path = method / "summary.json"
        metrics_path = method / "metrics.csv"
        dependency = run / "dependencies" / f"{scale}_{seed}.json"
        write_json(raw_manifest, {"status": "completed"})
        write_json(raw_plan, {"seed": seed})
        write_json(dependency, {"passed": True})
        parameter_count = int(self.contract[scale]["parameter_count"])
        k_state = 417792 if scale == "124m" else 1677312
        summary = {
            "method": "global_diag",
            "seed": seed,
            "completed_steps": 6200,
            "tokens_seen": 3_250_585_600,
            "k_state_bytes": k_state,
            "activation_scratch_bytes": 8,
            "final_val_loss": 3.2,
            "best_val_loss": 3.1,
            "architecture": {
                "parameter_count": parameter_count,
                "global_diag_route": True,
                "preconditioner_groups": [{"kind": "diag"}],
            },
        }
        write_json(summary_path, summary)
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("event", "step", "loss"))
            writer.writeheader()
            for step in range(0, 6201, 100):
                writer.writerow({"event": "val", "step": step, "loss": 4.0 - step / 10000})
        snapshot_sha = analyze.sha256_file(run / "source_snapshot/source_snapshot_manifest.json")
        certificate = {
            "passed": True,
            "experiment_id": "52_llama_global_diag_scale",
            "stage": "formal",
            "controller_stage": "formal",
            "scale": scale,
            "seed": seed,
            "method": "global_diag",
            "checks": {"all": True},
            "contract_sha256": analyze.sha256_file(self.contract_path),
            "source_snapshot_manifest_sha256": snapshot_sha,
            "data_fingerprint_sha256": self.contract["data"]["accepted_full_content_fingerprint_sha256"],
            "init_sha256": self.contract["accepted_init_sha256"][scale][str(seed)],
            "dependency_certificate": str(dependency),
            "dependency_certificate_sha256": analyze.sha256_file(dependency),
            "artifacts": {
                "llama_manifest.json": analyze.sha256_file(raw_manifest),
                "llama_plan.json": analyze.sha256_file(raw_plan),
                "summary.json": analyze.sha256_file(summary_path),
                "metrics.csv": analyze.sha256_file(metrics_path),
            },
            "paths": {
                "llama_manifest": str(raw_manifest),
                "llama_plan": str(raw_plan),
                "summary": str(summary_path),
                "metrics": str(metrics_path),
            },
        }
        cert = root / "ex52_unit_manifest.json"
        write_json(cert, certificate)
        return cert

    def test_formal_rows_require_complete_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            write_json(run / "source_snapshot/source_snapshot_manifest.json", {"passed": True})
            certificates = [
                self.make_cell(run, scale, seed)
                for scale in analyze.SCALES
                for seed in analyze.SEEDS
            ]
            rows = analyze.formal_rows(
                run,
                self.contract,
                analyze.sha256_file(self.contract_path),
            )
            self.assertEqual(len(rows), 6)
            tampered = json.loads(certificates[0].read_text(encoding="utf-8"))
            tampered["data_fingerprint_sha256"] = "0" * 64
            write_json(certificates[0], tampered)
            with self.assertRaisesRegex(RuntimeError, "certificate failed"):
                analyze.formal_rows(
                    run,
                    self.contract,
                    analyze.sha256_file(self.contract_path),
                )


if __name__ == "__main__":
    unittest.main()
