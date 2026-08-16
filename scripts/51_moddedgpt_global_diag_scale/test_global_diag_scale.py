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

import global_diag_scale_source_builder as builder
import run_global_diag_scale_suite as suite


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OFFICIAL = Path(
    os.environ.get("EX51_OFFICIAL_REPO", REPO.parent / "Newton-Muon-official-r0")
)


class GlobalDiagScaleTests(unittest.TestCase):
    def test_contract_and_controls_are_frozen(self) -> None:
        contract = json.loads((HERE / "global_diag_scale_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["experiment_id"], "51_moddedgpt_global_diag_scale")
        self.assertEqual(contract["formal"]["formal_units"], 7)
        self.assertEqual(
            contract["formal"]["seeds_by_scale"],
            {"275m": [2024, 2025, 2026, 2027], "455m": [2024, 2025, 2026]},
        )
        self.assertEqual(
            contract["data"]["accepted_fingerprint_sha256"],
            "1202c308d21ea690c17b958b98cbe40c65969a21928230950401f777adda8c68",
        )
        self.assertEqual(
            contract["frozen_controls"]["sha256"],
            "efb4805e3ac97cc1d69d396533d8d2b296de337be0cb65006544920561755d79",
        )
        self.assertEqual(
            hashlib.sha256((HERE / "frozen_scale_controls.csv").read_bytes()).hexdigest(),
            contract["frozen_controls"]["sha256"],
        )
        with (HERE / "frozen_scale_controls.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 28)
        self.assertEqual(len({(r["scale"], r["method"], r["seed"]) for r in rows}), 28)
        seed2027 = {
            row["method"]: float(row["final_val_loss"])
            for row in rows
            if row["scale"] == "275m" and row["seed"] == "2027"
        }
        self.assertEqual(
            seed2027,
            {
                "muon": 3.2763938903808594,
                "original_newton_muon": 3.2754123210906982,
                "selective_none": 3.27660870552063,
                "selective_diag": 3.277913808822632,
            },
        )
        self.assertEqual(
            suite.FORMAL_SEEDS,
            {"275m": (2024, 2025, 2026, 2027), "455m": (2024, 2025, 2026)},
        )
        self.assertEqual(contract["runtime"]["allowed_gpu_counts"], [1, 2])
        self.assertFalse(contract["runtime"]["dynamic_gpu_expansion_allowed"])

    @unittest.skipUnless(OFFICIAL.is_dir(), "local upstream checkout unavailable")
    def test_both_sources_are_deterministic_and_dense_activation_scratch_free(self) -> None:
        for scale, expected in (("275m", 503_808), ("455m", 901_120)):
            first = builder.build_source(OFFICIAL, scale)
            second = builder.build_source(OFFICIAL, scale)
            self.assertEqual(first.derived_sha256, second.derived_sha256)
            self.assertIn(f"EX51_GLOBAL_DIAG_ROUTE scale={scale}", first.source)
            self.assertIn("dense_activation_workspace=0", first.source)
            self.assertNotIn("dense_k_workspace=0", first.source)
            self.assertEqual(builder.expected_memory(scale)["k_state_bytes"], expected)
            compile(first.source, f"<ex51-{scale}>", "exec")

    @unittest.skipUnless(OFFICIAL.is_dir(), "local upstream checkout unavailable")
    def test_source_snapshot_is_reusable_and_bound_to_official_repo(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = Namespace(
                run_dir=Path(temp),
                repo=REPO,
                official_repo=OFFICIAL.resolve(),
            )
            first = suite.snapshot(args)
            second = suite.snapshot(args)
            self.assertEqual(first, second)
            manifest = json.loads(
                (first / "source_snapshot_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["passed"])
            self.assertEqual(manifest["official_repo"], str(OFFICIAL.resolve()))
            self.assertEqual(set(manifest["derived_training_sources"]), {"275m", "455m"})

    def test_launcher_has_no_second_official_repo_default(self) -> None:
        wrapper = (
            REPO
            / "commands/51_moddedgpt_global_diag_scale/20260814_ex51_moddedgpt_global_diag_scale.sh"
        ).read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'EX51_OFFICIAL_REPO="${EX51_OFFICIAL_REPO:-${SNM_OFFICIAL_REPO}}"',
            wrapper,
        )
        self.assertIn(
            'EX51_DATA_REPO_ROOT="${EX51_DATA_REPO_ROOT:-${EX51_OFFICIAL_REPO}/ex51_frozen50_data_repo}"',
            wrapper,
        )
        self.assertNotIn(
            'EX51_DATA_REPO_ROOT="${EX51_DATA_REPO_ROOT:-${EX51_WORKSPACE}/Newton-'
            + 'Muon-official}"',
            wrapper,
        )
        self.assertIn("preflight|pilot|formal|upload|verify|all", wrapper)
        self.assertIn("requires one or two GPU ids", wrapper)

    def test_preflight_gpu_allocation_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            manifest = run_dir / "preflight/preflight_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps({"passed": True, "requested_gpus": ["0"]}),
                encoding="utf-8",
            )
            suite.require_frozen_gpu_allocation(
                Namespace(run_dir=run_dir, gpus=["0"])
            )
            with self.assertRaisesRegex(RuntimeError, "does not support changing"):
                suite.require_frozen_gpu_allocation(
                    Namespace(run_dir=run_dir, gpus=["0", "1"])
                )

    def test_upload_is_explicit_and_never_changes_the_training_command(self) -> None:
        args = Namespace(
            run_dir=Path("/sealed/run"),
            training_python=Path("/training/python"),
            data_repo_root=Path("/official-r0/ex51_frozen50_data_repo"),
            wandb_mode="online",
            wandb_project="ex51-test",
            wandb_entity=None,
        )
        normal = suite.command(args, "275m", 2024, "formal", Path("/attempt"), "0")
        upload = suite.command(
            args,
            "275m",
            2024,
            "formal",
            Path("/attempt"),
            "0",
            upload_only=True,
        )
        self.assertNotIn("--upload-only", normal)
        self.assertEqual(upload[:-1], normal)
        self.assertEqual(upload[-1], "--upload-only")
        self.assertEqual(suite.STAGES, ("preflight", "pilot", "formal", "upload", "verify", "all"))

    def test_closed_form_memory_certificates(self) -> None:
        self.assertEqual(builder.expected_memory("275m"), {"k_cov_bytes": 251_904, "k_inv_bytes": 251_904, "k_state_bytes": 503_808, "activation_stat_bytes": 252_088})
        self.assertEqual(builder.expected_memory("455m"), {"k_cov_bytes": 450_560, "k_inv_bytes": 450_560, "k_state_bytes": 901_120, "activation_stat_bytes": 450_808})

    def test_record17_compiled_diagonal_op_has_no_unresolved_dummy_helper(self) -> None:
        self.assertNotIn("return _record17_dummy_scalar(", builder.GENERIC_DIAG_RECORD17)
        self.assertIn("return accum.new_empty(())", builder.GENERIC_DIAG_RECORD17)

    def test_record28_overlay_admits_global_diag_method(self) -> None:
        source = (
            'if RECORD28_METHOD not in ("original_newton_muon", '
            '"selective_none", "selective_diag"):\n'
            '    "selective_diag": "diag",\n}'
        )
        overlaid = builder.replace_once(
            source,
            'if RECORD28_METHOD not in ("original_newton_muon", '
            '"selective_none", "selective_diag"):',
            'if RECORD28_METHOD not in ("original_newton_muon", '
            '"selective_none", "selective_diag", "global_diag"):',
            "test allowlist",
        )
        self.assertIn('"selective_diag", "global_diag")', overlaid)


if __name__ == "__main__": unittest.main()
