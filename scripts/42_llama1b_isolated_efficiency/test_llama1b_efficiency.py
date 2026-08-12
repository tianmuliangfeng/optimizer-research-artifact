#!/usr/bin/env python3
"""CPU-only contract tests for experiment 42."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import efficiency_common as common
import llama1b_efficiency_worker as worker
import run_llama1b_efficiency as controller
import source_builder


class Experiment42Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = common.read_json(HERE / "efficiency_contract.json")

    def test_contract_and_rotation_are_exact(self) -> None:
        controller.validate_contract(self.contract)
        methods = self.contract["method_order"]
        orders = self.contract["execution_policy"]["orders"]
        self.assertEqual(
            orders,
            [
                ["muon", "newton_full", "down_none", "down_diag"],
                ["newton_full", "down_none", "down_diag", "muon"],
                ["down_none", "down_diag", "muon", "newton_full"],
                ["down_diag", "muon", "newton_full", "down_none"],
            ],
        )
        for method in methods:
            self.assertEqual(
                sorted(order.index(method) for order in orders), [0, 1, 2, 3]
            )
        frozen = self.contract["frozen_configuration"]
        self.assertEqual(frozen["total_updates"], 544)
        self.assertEqual(frozen["timed_updates"], 512)
        self.assertEqual(frozen["timed_updates"] * frozen["tokens_per_update"], 268_435_456)
        self.assertEqual(
            sum(step % 32 == 0 for step in range(33, 545)),
            16,
        )

    def test_primary_contrasts_do_not_promote_selective_internal_comparison(
        self,
    ) -> None:
        contrasts = {
            (row["candidate"], row["reference"])
            for row in self.contract["primary_contrasts"]
        }
        self.assertIn(("down_none", "muon"), contrasts)
        self.assertIn(("down_diag", "muon"), contrasts)
        self.assertIn(("down_none", "newton_full"), contrasts)
        self.assertIn(("down_diag", "newton_full"), contrasts)
        self.assertNotIn(("down_diag", "down_none"), contrasts)
        self.assertNotIn(("down_none", "down_diag"), contrasts)

    def test_derived_source_has_post_warmup_memory_window(self) -> None:
        bundle = source_builder.build_source_bundle()
        source = bundle.derived_source
        self.assertIn("if completed_steps == 32:", source)
        self.assertIn("torch.cuda.reset_peak_memory_stats()", source)
        self.assertEqual(source.count("torch.cuda.reset_peak_memory_stats()"), 1)
        self.assertIn('"timed_training_peak_allocated_bytes"', source)
        self.assertIn('"timed_training_peak_reserved_bytes"', source)
        self.assertIn('"peak_reset_after_completed_step": 32', source)
        capture = source.index(
            "timed_peak_reserved_bytes = max(\n", source.index("if completed_steps > 32:")
        )
        summary_use = source.index(
            '"timed_training_peak_reserved_bytes": timed_peak_reserved_bytes'
        )
        self.assertLess(capture, summary_use)
        self.assertEqual(source.count("timed_peak_reserved_bytes = max("), 1)
        compile(source, "derived_efficiency_trainer.py", "exec")
        with tempfile.TemporaryDirectory() as temporary:
            observed = source_builder.materialize(bundle, Path(temporary))
            self.assertEqual(observed["derived_base"], bundle.derived_sha256)
            self.assertEqual(observed["profile_wrapper"], bundle.wrapper_sha256)

    def test_tier_configuration_and_command_disable_side_effects(self) -> None:
        formal = worker.tier_configuration(self.contract, "formal")
        smoke = worker.tier_configuration(self.contract, "smoke")
        self.assertEqual(formal["num_iterations"], 544)
        self.assertEqual(formal["steady_steps"], 512)
        self.assertEqual(formal["val_tokens"], 8192)
        self.assertEqual(formal["warmdown_iters"], 0)
        self.assertEqual(smoke["num_iterations"], 34)
        self.assertEqual(smoke["steady_steps"], 2)
        args = type(
            "Args",
            (),
            {
                "python_exe": Path("/venv/bin/python"),
                "profile_wrapper": Path("/snapshot/wrapper.py"),
                "method": "down_none",
                "data_dir": Path("/data"),
            },
        )()
        command = worker.build_training_command(args, formal, Path("/output"))
        self.assertEqual(command[command.index("--resume") + 1], "never")
        self.assertEqual(command[command.index("--checkpoint-every") + 1], "0")
        self.assertIn("--no-save-final", command)
        self.assertNotIn("wandb", " ".join(command).lower())

    def write_metrics(
        self, path: Path, *, omit_step: int | None = None, duplicate_step: int | None = None
    ) -> None:
        fields = [
            "event",
            "step",
            "loss",
            "train_s",
            "steady_train_s",
            "step_avg_ms",
            "lr_backup",
            "lr_matrix",
            "tokens_seen",
        ]
        rows = [
            {
                "event": "val",
                "step": 0,
                "loss": 10.0,
                "train_s": 0.0,
                "steady_train_s": 0.0,
                "step_avg_ms": "nan",
                "lr_backup": 0.0036,
                "lr_matrix": 0.01,
                "tokens_seen": 0,
            }
        ]
        for step in range(1, 35):
            if step == omit_step:
                continue
            steady = max(0, step - 32) * 0.1
            row = {
                "event": "train",
                "step": step,
                "loss": 9.0,
                "train_s": step * 0.2,
                "steady_train_s": steady,
                "step_avg_ms": "nan" if step <= 32 else 100.0,
                "lr_backup": 0.0036,
                "lr_matrix": 0.01,
                "tokens_seen": step * 524_288,
            }
            rows.append(row)
            if step == duplicate_step:
                rows.append(dict(row))
        rows.append(
            {
                "event": "val",
                "step": 34,
                "loss": 8.0,
                "train_s": 6.8,
                "steady_train_s": 0.2,
                "step_avg_ms": 100.0,
                "lr_backup": 0.0036,
                "lr_matrix": 0.01,
                "tokens_seen": 34 * 524_288,
            }
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_metric_validator_accepts_exact_two_step_smoke_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.csv"
            self.write_metrics(path)
            audit = worker.validate_metrics(
                path,
                total_steps=34,
                timed_steps=2,
                tokens_per_update=524_288,
                steady_train_s=0.2,
            )
            self.assertEqual(audit["timed_rows"], 2)
            self.assertEqual(audit["timed_step_first"], 33)
            self.assertEqual(audit["timed_step_last"], 34)

    def test_metric_validator_rejects_missing_or_duplicate_steps(self) -> None:
        for keyword in ({"omit_step": 17}, {"duplicate_step": 17}):
            with self.subTest(keyword=keyword), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "metrics.csv"
                self.write_metrics(path, **keyword)
                with self.assertRaisesRegex(RuntimeError, "missing, duplicated"):
                    worker.validate_metrics(
                        path,
                        total_steps=34,
                        timed_steps=2,
                        tokens_per_update=524_288,
                        steady_train_s=0.2,
                    )

    def test_expected_state_bytes_are_pinned(self) -> None:
        self.assertEqual(
            self.contract["expected_state_bytes"]["model_parameter_bytes"],
            4_054_761_472,
        )
        expected = self.contract["expected_state_bytes"]["optimizer_state_bytes"]
        self.assertEqual(expected["muon"], 4_466_770_072)
        self.assertEqual(expected["newton_full"], 10_641_047_704)
        self.assertEqual(expected["down_none"], 6_278_709_400)
        self.assertEqual(expected["down_diag"], 6_279_501_976)

    def test_next_attempt_never_overwrites_failed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary)
            name1, path1 = controller.next_attempt(cell)
            (path1 / "failure.txt").write_text("preserve", encoding="utf-8")
            name2, path2 = controller.next_attempt(cell)
            self.assertEqual(name1, "attempt_001")
            self.assertEqual(name2, "attempt_002")
            self.assertEqual((path1 / "failure.txt").read_text(), "preserve")
            self.assertTrue(path2.is_dir())

    def test_official_repo_audit_rejects_staged_semantic_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.email", "audit@example.invalid"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.name", "Experiment 42"],
                check=True,
            )
            kernel = repo / "triton_kernels.py"
            other = repo / "tracked.txt"
            kernel.write_text("PINNED = True\n", encoding="utf-8", newline="\n")
            other.write_text("original\n", encoding="utf-8", newline="\n")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-m", "baseline"],
                check=True,
                capture_output=True,
            )
            commit = (
                subprocess.run(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                .stdout.strip()
            )
            contract = {
                "source_contract": {
                    "official_repo_commit": commit,
                    "triton_kernels_sha256": common.sha256_file(kernel),
                }
            }
            other.write_text("semantic change\n", encoding="utf-8", newline="\n")
            subprocess.run(
                ["git", "-C", str(repo), "add", "tracked.txt"], check=True
            )
            with self.assertRaisesRegex(RuntimeError, "official repository audit"):
                common.audit_official_repo(repo, contract)


if __name__ == "__main__":
    unittest.main()
