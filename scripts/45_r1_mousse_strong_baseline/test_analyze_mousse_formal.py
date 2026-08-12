from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_ROOT = Path(
    os.environ.get("SNM_RESULTS_ROOT", str(ARTIFACT_ROOT / "runs"))
).expanduser()
CORE = RESULTS_ROOT / "15_official_newton_muon_r1/analysis/wandb_20260721_multiseed_factorial/r1_multiseed_run_summary.csv"
EXTENDED = RESULTS_ROOT / "19_r1_extended_baselines/analysis/wandb_20260723_formal_multiseed_unified/extended_formal_run_summary.csv"
EVIDENCE_AVAILABLE = CORE.is_file() and EXTENDED.is_file()


class FormalAnalyzerTests(unittest.TestCase):
    @unittest.skipUnless(
        EVIDENCE_AVAILABLE,
        "accepted R1 comparison summaries are not bundled with the source release",
    )
    def test_synthetic_mousse_rows_form_eight_method_panel(self) -> None:
        initial = {2024: 10.9462, 2025: 10.9869, 2026: 10.9790}
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs = []
            for seed in (2026, 2024, 2025):
                folder = root / f"seed{seed}"
                folder.mkdir()
                path = folder / "formal_summary.csv"
                row = {
                    "method": "mousse", "controlled_seed": seed, "total_steps": 6200,
                    "total_tokens": 3_250_585_600, "initial_val_loss": initial[seed],
                    "final_val_loss": 3.27, "best_val_loss": 3.27,
                    "tail5_val_loss_mean": 3.28, "normalized_val_auc": 3.62,
                    "peak_memory_allocated_mib": 41000, "optimizer_state_bytes": 3_000_000_000,
                    "run_name": f"mousse_seed{seed}",
                }
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                (folder / "formal_manifest.json").write_text(
                    json.dumps(
                        {
                            "status": "completed_valid",
                            "protocol": "mousse_r1_selected_6200step_v1",
                            "total_steps": 6200,
                            "source_audit": {"derived_source_sha256": "same-source"},
                            "training_runtime_fingerprint": {"gpu_name": "NVIDIA H100 80GB HBM3"},
                            "summary": {"evidence_valid": True},
                        }
                    ) + "\n",
                    encoding="utf-8",
                )
                inputs.append(folder)
            output = root / "analysis"
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT_DIR / "analyze_mousse_formal.py"),
                    "--mousse-summaries", *(str(path) for path in inputs),
                    "--core-summary", str(CORE), "--extended-summary", str(EXTENDED),
                    "--output-dir", str(output),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            identity = json.loads((output / "identity_reuse_certificate.json").read_text(encoding="utf-8"))
            self.assertEqual(identity["status"], "passed_with_caveats")
            self.assertTrue(identity["paired_quality_eligible"])
            self.assertFalse(identity["strict_per_run_local_manifest_identity"])
            with (output / "r1_unified_eight_method_run_summary.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 24)
            with (output / "r1_mousse_paired_aggregate.csv").open(encoding="utf-8", newline="") as handle:
                contrasts = list(csv.DictReader(handle))
            self.assertEqual(len(contrasts), 7)


if __name__ == "__main__":
    unittest.main()
