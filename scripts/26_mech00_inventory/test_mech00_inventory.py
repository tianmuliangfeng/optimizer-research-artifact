#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_mech00_inventory.py"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        llama = root / "llama" / "20260724_formal_seed2026"
        llama_run = llama / "01_down_none"
        llama_ckpt = llama_run / "checkpoint_latest.pt"
        llama_ckpt.parent.mkdir(parents=True)
        llama_ckpt.write_bytes(b"synthetic-llama-checkpoint")
        write_json(
            llama / "llama_manifest.json",
            {
                "family": "llama_swiglu_parameter_matched_r1",
                "batch_kind": "formal",
                "execution_stage": "medium",
                "seed": 2026,
                "runtime": {
                    "gpu_name": "Synthetic H100",
                    "torch": "2.8.0",
                    "torch_cuda": "12.6",
                },
                "config": {"num_iterations": 6200},
            },
        )
        write_json(
            llama_run / "summary.json",
            {
                "status": "completed",
                "method": "down_none",
                "seed": 2026,
                "completed_steps": 6200,
                "tokens_seen": 123,
                "checkpoint_path": str(llama_ckpt),
                "architecture": {"parameter_count": 124_000_000, "n_layer": 12},
                "runtime": {"python_executable": "/synthetic/python"},
                "init_sha256": "init-llama",
            },
        )

        r1 = root / "r1" / "20260724_formal_seed2026" / "01_none"
        r1_ckpt = r1 / "workspace" / "logs" / "run" / "state_step6200.pt"
        r1_ckpt.parent.mkdir(parents=True)
        r1_ckpt.write_bytes(b"synthetic-r1-checkpoint")
        write_json(
            r1 / "r1_summary.json",
            {
                "status": "completed",
                "method": "none",
                "controlled_seed": 2026,
                "final_train_step": 6200,
                "checkpoint_path": str(r1_ckpt),
                "derived_script_sha256": "source-r1",
            },
        )
        write_json(
            r1 / "run_manifest.json",
            {
                "status": "completed_valid_local",
                "training_runtime_fingerprint": {
                    "gpu_name": "Synthetic H100",
                    "torch": "2.8.0",
                },
            },
        )
        write_json(
            r1.parent / "r1_manifest.json",
            {
                "formal_evidence": True,
                "evidence_profile": "formal_evidence",
                "controlled_seed": 2026,
            },
        )

        output = root / "output"
        command = [
            sys.executable,
            str(RUNNER),
            "--host-id",
            "synthetic-host",
            "--execution-domain",
            "unit-test",
            "--input",
            f"llama={llama.parent}",
            "--input",
            f"r1={r1.parent.parent}",
            "--family-hint",
            "llama=llama_124m",
            "--family-hint",
            "r1=r1_native",
            "--methods",
            "down_none",
            "none",
            "--hash-mode",
            "full",
            "--output-dir",
            str(output),
            "--strict",
        ]
        subprocess.run(command, check=True)

        inventory = read_csv(output / "checkpoint_inventory.csv")
        assert len(inventory) == 2, inventory
        by_method = {row["method"]: row for row in inventory}
        assert by_method["down_none"]["exact_resume_expected"] == "True"
        assert by_method["down_none"]["checkpoint_schema_verified"] == "False"
        assert by_method["down_none"]["evidence_kind"] == "medium"
        assert by_method["none"]["fresh_geometry_ready"] == "True"
        assert by_method["none"]["exact_resume_expected"] == "False"

        hashes = read_csv(output / "checkpoint_hashes.csv")
        assert len(hashes) == 2
        assert all(row["hash_status"] == "verified_stable" for row in hashes)
        assert all(len(row["sha256"]) == 64 for row in hashes)

        step_map = read_csv(output / "available_step_map.csv")
        exact_6200 = [
            row
            for row in step_map
            if row["target_step"] == "6200" and row["exact_match"] == "True"
        ]
        assert len(exact_6200) == 2
        assert llama_ckpt.read_bytes() == b"synthetic-llama-checkpoint"
        assert r1_ckpt.read_bytes() == b"synthetic-r1-checkpoint"

        required = {
            "checkpoint_inventory.csv",
            "checkpoint_hashes.csv",
            "available_step_map.csv",
            "input_discovery.csv",
            "source_inventory.csv",
            "runtime_inventory.json",
            "diagnostic_data_contract.json",
            "audit_checks.csv",
            "mech00_manifest.json",
        }
        assert required.issubset({path.name for path in output.iterdir()})

    print("MECH-00 inventory test passed")


if __name__ == "__main__":
    main()
