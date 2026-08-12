#!/usr/bin/env python3
"""Local contract tests for experiment 41."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import factorial_source_builder as B
import analyze_r1_module_factorial as A
import run_r1_module_factorial_suite as S


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
_official_env = os.environ.get("SNM_OFFICIAL_REPO")
_official_candidates = (
    REPO_ROOT / "third_party" / "Newton-Muon-official-r0",
    REPO_ROOT / "third_party" / "Newton-Muon-official",
)
OFFICIAL_REPO = (
    Path(_official_env).expanduser().resolve()
    if _official_env
    else next((path for path in _official_candidates if path.is_dir()), _official_candidates[0])
)


def write_metrics(
    batch: Path,
    *,
    run_name: str,
    method: str,
    initial_val_loss: float,
    final_val_loss: float,
) -> None:
    run_dir = batch / run_name
    run_dir.mkdir()
    with (run_dir / "r1_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "cproj_k_mode", "event", "step", "loss"),
        )
        writer.writeheader()
        validation_steps = (0, 1000, 2000, 3000, 4000, 6200)
        for step in validation_steps:
            progress = step / A.FORMAL_TOTAL_STEPS
            writer.writerow(
                {
                    "method": method,
                    "cproj_k_mode": method,
                    "event": "validation",
                    "step": step,
                    "loss": initial_val_loss
                    + progress * (final_val_loss - initial_val_loss),
                }
            )
        writer.writerow(
            {
                "method": method,
                "cproj_k_mode": method,
                "event": "train",
                "step": 1,
                "loss": initial_val_loss - 0.1,
            }
        )


class R1ModuleFactorialTests(unittest.TestCase):
    def test_contract_has_exact_factorial(self) -> None:
        contract = json.loads(
            (SCRIPT_DIR / "factorial_contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(contract["axes"]["c_fc_k"], ["none", "full"])
        self.assertEqual(contract["axes"]["c_proj_k"], ["none", "block4"])
        self.assertEqual(
            set(contract["new_training_cells"]), {"cproj_only", "neither"}
        )
        self.assertEqual(set(contract["reused_cells"]), {"both", "fc_only"})
        self.assertFalse(contract["evidence_policy"]["all_none_is_muon_baseline"])
        self.assertEqual(
            contract["execution_policy"]["physical_gpus"], ["0", "1"]
        )
        self.assertEqual(
            contract["execution_policy"]["maximum_concurrent_training_processes"],
            2,
        )
        self.assertTrue(
            contract["execution_policy"]["one_visible_gpu_per_process"]
        )
        reference = SCRIPT_DIR / "existing_cells_reference.csv"
        self.assertEqual(
            hashlib.sha256(reference.read_bytes()).hexdigest(),
            contract["reused_summary"]["frozen_reference_sha256"],
        )

    def test_builder_rejects_unregistered_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            B.build_factorial_source(OFFICIAL_REPO, "diag")

    @unittest.skipUnless(
        OFFICIAL_REPO.is_dir(), f"pinned upstream repo unavailable: {OFFICIAL_REPO}"
    )
    def test_parameterized_sources_are_identical(self) -> None:
        built = B.build_factorial_sources(OFFICIAL_REPO)
        self.assertEqual(
            built["block4"].derived_sha256, built["none"].derived_sha256
        )
        B.assert_factorial_source_contract(built["block4"].source)

    @unittest.skipUnless(
        OFFICIAL_REPO.is_dir(), f"pinned upstream repo unavailable: {OFFICIAL_REPO}"
    )
    def test_runtime_contract_requires_cfc_none(self) -> None:
        source = B.build_factorial_source(OFFICIAL_REPO, "block4").source
        self.assertIn('if R1_CFC_K_MODE != "none":', source)
        self.assertIn("self.c_fc.weight._stats_ref = None", source)
        self.assertIn(
            'if precond_flag and R1_CFC_K_MODE == "full":', source
        )

    def test_resume_reuses_completed_retry_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "analysis").mkdir()
            retry = root / "analysis_retry_1"
            retry.mkdir()
            (retry / "r1_module_factorial_analysis_manifest.json").write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            self.assertEqual(S.analysis_dir(root), retry)

    def test_suite_separates_controller_and_training_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(
                run_dir=root / "run",
                repo=REPO_ROOT,
                official_repo=OFFICIAL_REPO,
                python_exe="/isolated/training/python",
                wandb_project=None,
                wandb_entity=None,
            )
            command = S.runner_command(args, seed=2026, stage="smoke")
            self.assertEqual(command[0], S.controller_python())
            training_index = command.index("--python-exe") + 1
            self.assertEqual(command[training_index], args.python_exe)
            self.assertNotEqual(command[0], command[training_index])

    def test_controller_python_preserves_venv_symlink_entrypoint(self) -> None:
        venv_entrypoint = "${SNM_CONTROLLER_PYTHON}"
        with mock.patch.object(S.sys, "executable", venv_entrypoint):
            self.assertEqual(S.controller_python(), venv_entrypoint)

    @unittest.skipUnless(
        OFFICIAL_REPO.is_dir(), f"pinned upstream repo unavailable: {OFFICIAL_REPO}"
    )
    def test_source_compiles_after_materialization(self) -> None:
        source = B.build_factorial_source(OFFICIAL_REPO, "none").source
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train_r1_none.py"
            path.write_text(source, encoding="utf-8")
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_factorial_effects_recover_known_main_effects(self) -> None:
        rows = []
        values = {
            "both": 1.0,
            "fc_only": 1.2,
            "cproj_only": 1.3,
            "neither": 1.4,
        }
        for cell, (cfc, cproj) in A.CELL_COORDS.items():
            for seed in A.SEEDS:
                rows.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "cfc_k_mode": cfc,
                        "cproj_k_mode": cproj,
                        "final_val_loss": values[cell],
                        "tail5_val_loss_mean": values[cell],
                        "normalized_val_auc": values[cell],
                    }
                )
        _, summary = A.build_effects(rows, 0.002)
        final = {
            row["effect"]: row
            for row in summary
            if row["metric"] == "final_val_loss"
        }
        self.assertAlmostEqual(final["cfc_main"]["mean"], -0.25)
        self.assertAlmostEqual(final["cproj_main"]["mean"], -0.15)
        self.assertAlmostEqual(final["interaction"]["mean"], -0.1)

    def test_decision_prioritizes_beneficial_cproj_when_both_factors_help(
        self,
    ) -> None:
        summaries = [
            {
                "metric": "final_val_loss",
                "effect": "cfc_main",
                "mean": -0.0035,
                "negative_seeds": 3,
                "positive_seeds": 0,
            },
            {
                "metric": "final_val_loss",
                "effect": "cproj_main",
                "mean": -0.0046,
                "negative_seeds": 3,
                "positive_seeds": 0,
            },
            {
                "metric": "final_val_loss",
                "effect": "interaction",
                "mean": 0.0002,
                "negative_seeds": 1,
                "positive_seeds": 1,
            },
        ]
        decision = A.classify(summaries, 0.002)
        self.assertEqual(decision["classification"], "r1_allocation_diverges")
        self.assertTrue(decision["cfc_beneficial"])
        self.assertTrue(decision["cproj_beneficial"])
        self.assertFalse(decision["cproj_harmful"])

    def test_new_cell_manifest_loader_requires_both_cells(self) -> None:
        contract_sha = A.sha256_file(SCRIPT_DIR / "factorial_contract.json")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed in A.SEEDS:
                batch = root / "formal" / f"seed{seed}" / "batch"
                batch.mkdir(parents=True)
                summaries = []
                for method, cell, k_mib in (
                    ("block4", "cproj_only", 324.0),
                    ("none", "neither", 108.0),
                ):
                    run_name = f"{cell}_{seed}"
                    write_metrics(
                        batch,
                        run_name=run_name,
                        method=method,
                        initial_val_loss=10.0 + seed / 10000,
                        final_val_loss=3.0,
                    )
                    val_curve_mean = (
                        10.0 + seed / 10000 + 3.0
                    ) / 2.0
                    summaries.append(
                        {
                            "method": method,
                            "cproj_k_mode": method,
                            "cfc_k_mode": "none",
                            "factorial_cell": cell,
                            "run_name": run_name,
                            "init_sha256": f"{seed:064x}",
                            "final_val_step": A.FORMAL_TOTAL_STEPS,
                            "final_val_loss": 3.0,
                            "val_curve_mean": val_curve_mean,
                            "validation_points": 6,
                            "peak_memory_allocated_mib": 38000,
                            "k_state_bytes": int(k_mib * 1024**2),
                        }
                    )
                (batch / "r1_manifest.json").write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "status": "completed_valid",
                            "formal_evidence": True,
                            "failures": [],
                            "wandb_complete": True,
                            "module_factorial": {
                                "contract_sha256": contract_sha
                            },
                            "summaries": summaries,
                        }
                    ),
                    encoding="utf-8",
                )
            rows, audits = A.load_new_cells(root, contract_sha)
            self.assertEqual(len(rows), 6)
            self.assertEqual(len(audits), 3)
            self.assertTrue(
                all(
                    len(audit["validation_curve_evidence"]) == 2
                    and all(
                        evidence["validation_rows"] == 6
                        and len(evidence["sha256"]) == 64
                        for evidence in audit["validation_curve_evidence"]
                    )
                    for audit in audits
                )
            )
            for row in rows:
                expected_auc = (
                    10.0 + row["seed"] / 10000 + row["final_val_loss"]
                ) / 2.0
                self.assertAlmostEqual(row["normalized_val_auc"], expected_auc)
            self.assertEqual(
                {row["factorial_cell"] for row in rows}
                if rows and "factorial_cell" in rows[0]
                else {row["cell"] for row in rows},
                {"cproj_only", "neither"},
            )

    def test_existing_reference_and_new_cells_form_complete_factorial(self) -> None:
        contract_path = SCRIPT_DIR / "factorial_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract_sha = A.sha256_file(contract_path)
        initial = {2024: 10.9462, 2025: 10.9869, 2026: 10.979}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed in A.SEEDS:
                batch = root / "formal" / f"seed{seed}" / "batch"
                batch.mkdir(parents=True)
                summaries = []
                for method, cell, k_mib, final in (
                    ("block4", "cproj_only", 324.0, 3.26),
                    ("none", "neither", 108.0, 3.27),
                ):
                    run_name = f"{cell}_{seed}"
                    write_metrics(
                        batch,
                        run_name=run_name,
                        method=method,
                        initial_val_loss=initial[seed],
                        final_val_loss=final,
                    )
                    summaries.append(
                        {
                            "method": method,
                            "cproj_k_mode": method,
                            "cfc_k_mode": "none",
                            "factorial_cell": cell,
                            "run_name": run_name,
                            "init_sha256": f"{seed:064x}",
                            "final_val_step": A.FORMAL_TOTAL_STEPS,
                            "final_val_loss": final,
                            "val_curve_mean": (initial[seed] + final) / 2.0,
                            "validation_points": 6,
                            "peak_memory_allocated_mib": 38000,
                            "k_state_bytes": int(k_mib * 1024**2),
                        }
                    )
                (batch / "r1_manifest.json").write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "status": "completed_valid",
                            "formal_evidence": True,
                            "failures": [],
                            "wandb_complete": True,
                            "module_factorial": {
                                "contract_sha256": contract_sha
                            },
                            "summaries": summaries,
                        }
                    ),
                    encoding="utf-8",
                )
            existing, _ = A.load_existing_cells(
                SCRIPT_DIR / "existing_cells_reference.csv", contract
            )
            new, _ = A.load_new_cells(root, contract_sha)
            checks = A.validate_cells(existing + new, contract)
            self.assertTrue(checks["cell_seed_coverage"])
            self.assertTrue(checks["k_state_contract_erratum"]["required"])
            self.assertEqual(
                checks["k_state_contract_erratum"][
                    "corrected_total_k_state_mib"
                ],
                {
                    "both": 378.0,
                    "fc_only": 162.0,
                    "cproj_only": 324.0,
                    "neither": 108.0,
                },
            )
            self.assertTrue(
                all(
                    row["passed"]
                    for row in checks["k_state_additivity_by_seed"].values()
                )
            )
            _, summary = A.build_effects(
                existing + new, contract["practical_loss_margin"]
            )
            self.assertEqual(
                {row["effect"] for row in summary if row["metric"] == "final_val_loss"},
                {
                    "cfc_main",
                    "cproj_main",
                    "interaction",
                    "disable_cproj_when_cfc_full",
                    "disable_cproj_when_cfc_none",
                    "disable_cfc_when_cproj_block4",
                    "disable_cfc_when_cproj_none",
                },
            )

    def test_new_cell_loader_rejects_mismatched_initialization_hashes(self) -> None:
        contract_sha = A.sha256_file(SCRIPT_DIR / "factorial_contract.json")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed in A.SEEDS:
                batch = root / "formal" / f"seed{seed}" / "batch"
                batch.mkdir(parents=True)
                summaries = []
                for index, (method, cell, k_mib) in enumerate(
                    (
                        ("block4", "cproj_only", 324.0),
                        ("none", "neither", 108.0),
                    )
                ):
                    run_name = f"{cell}_{seed}"
                    write_metrics(
                        batch,
                        run_name=run_name,
                        method=method,
                        initial_val_loss=10.0,
                        final_val_loss=3.0,
                    )
                    summaries.append(
                        {
                            "method": method,
                            "cproj_k_mode": method,
                            "cfc_k_mode": "none",
                            "factorial_cell": cell,
                            "run_name": run_name,
                            "init_sha256": f"{seed + index:064x}",
                            "final_val_step": A.FORMAL_TOTAL_STEPS,
                            "final_val_loss": 3.0,
                            "val_curve_mean": 6.5,
                            "validation_points": 6,
                            "peak_memory_allocated_mib": 38000,
                            "k_state_bytes": int(k_mib * 1024**2),
                        }
                    )
                (batch / "r1_manifest.json").write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "status": "completed_valid",
                            "formal_evidence": True,
                            "failures": [],
                            "wandb_complete": True,
                            "module_factorial": {
                                "contract_sha256": contract_sha
                            },
                            "summaries": summaries,
                        }
                    ),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(
                RuntimeError, "initialization fingerprints mismatch"
            ):
                A.load_new_cells(root, contract_sha)


if __name__ == "__main__":
    unittest.main()
