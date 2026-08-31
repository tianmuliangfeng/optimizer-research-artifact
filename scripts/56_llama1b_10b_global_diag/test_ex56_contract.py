from __future__ import annotations

import csv
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = load("ex56_test_protocol", HERE / "protocol.py")
sys.modules["protocol"] = P
BUILDER = load("ex56_test_builder", HERE / "llama_global_diag_source_builder.py")
RUNNER = load("ex56_test_runner", HERE / "run_formal.py")
ANALYZER = load("ex56_test_analyzer", HERE / "analyze_formal.py")


class Ex56ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads((HERE / "formal_contract.json").read_text(encoding="utf-8"))

    def test_contract_is_fully_frozen(self) -> None:
        checks = P.validate_contract(self.contract)
        self.assertTrue(all(checks.values()), checks)

    def test_grid_and_from_scratch_boundary(self) -> None:
        self.assertEqual(self.contract["grid"]["methods"], ["global_diag"])
        self.assertEqual(self.contract["grid"]["seeds"], [2024, 2025, 2026])
        self.assertEqual(self.contract["grid"]["formal_units"], 3)
        self.assertEqual(self.contract["grid"]["physical_gpus"], [3])
        self.assertEqual(self.contract["runtime"]["gpu_count"], 1)
        self.assertEqual(RUNNER.REQUIRED_PHYSICAL_GPUS, [3])
        runner_source = (HERE / "run_formal.py").read_text(encoding="utf-8")
        self.assertIn('"physical_gpu_selection"', runner_source)
        self.assertIn('"CUDA_VISIBLE_DEVICES": ",".join', runner_source)
        self.assertTrue(self.contract["lr_policy"]["from_scratch_required"])
        self.assertTrue(self.contract["lr_policy"]["ex52_checkpoint_resume_forbidden"])
        self.assertEqual(
            set(self.contract["accepted_ex48_initialization_sha256"]),
            {"2024", "2025", "2026"},
        )

    def test_phase_graph_matches_ex48_budget_semantics(self) -> None:
        self.assertEqual(
            [(row["id"], row.get("parent"), row["target_step"]) for row in self.contract["phases"]],
            [
                ("backbone_4400", None, 4400),
                ("cooldown_6200", "backbone_4400", 6200),
                ("backbone_11493", "backbone_4400", 11493),
                ("cooldown_13293", "backbone_11493", 13293),
                ("backbone_17273", "backbone_11493", 17273),
                ("cooldown_19073", "backbone_17273", 19073),
            ],
        )

    def test_data_is_content_bound_to_accepted_ex48(self) -> None:
        data = self.contract["data"]
        self.assertEqual(
            data["accepted_ex48_inventory_sha256"],
            "76848d39697aa6e0a7083e4445185fd7215b4f7ddad411d60a7cd6943c384b21",
        )
        self.assertEqual(
            data["accepted_ex48_content_projection_sha256"],
            "2820057bcdfa76afd9523c612c7e6846b3c5f545c98066c129c8b9d4c06a9b10",
        )
        source = (HERE / "run_formal.py").read_text(encoding="utf-8")
        self.assertIn('accepted_ex48_content_projection', source)

        row = {
            "name": "fineweb_train_000001.bin", "index": 1, "tokens": 9,
            "consumable_tokens": 8, "bytes": 1042,
            "header_sha256": "a" * 64, "sha256": "b" * 64,
            "mtime_ns": 1,
        }
        identity = {"train": [row], "validation": [dict(row, name="fineweb_val_000000.bin", index=0)]}
        observed = P.content_inventory_sha256(identity)
        identity["train"][0]["mtime_ns"] = 999
        self.assertEqual(P.content_inventory_sha256(identity), observed)
        identity["train"][0]["sha256"] = "c" * 64
        self.assertNotEqual(P.content_inventory_sha256(identity), observed)

    def test_init_audit_covers_every_formal_seed(self) -> None:
        source = (HERE / "run_formal.py").read_text(encoding="utf-8")
        self.assertIn('for seed in contract["grid"]["seeds"]', source)
        self.assertIn('"accepted_ex48_seed_hashes"', source)
        analyzer = (HERE / "analyze_formal.py").read_text(encoding="utf-8")
        self.assertIn('"accepted_ex48_initialization"', analyzer)

    def test_global_diag_source_is_deterministic_and_pinned(self) -> None:
        built = BUILDER.build(REPO)
        digest = BUILDER.sha256_text(built.trainer)
        self.assertEqual(digest, self.contract["source_lineage"]["global_diag_trainer_sha256"])
        self.assertEqual(digest, self.contract["accepted_sources"]["scripts/17_llama_swiglu_validation/train_llama_swiglu.py"])
        self.assertIn('"global_diag"', built.trainer)
        self.assertIn('"kind": "diag" if self.method == "global_diag" else "dense"', built.trainer)

    def test_live_source_gate_includes_generated_sources(self) -> None:
        payload = RUNNER.check_live_sources(REPO, self.contract)
        self.assertTrue(payload["passed"], payload["checks"])
        trainer = payload["files"]["scripts/17_llama_swiglu_validation/train_llama_swiglu.py"]
        self.assertEqual(trainer["sha256"], self.contract["source_lineage"]["global_diag_trainer_sha256"])

    def test_single_gpu_queue_is_deterministic(self) -> None:
        units = [
            {"method": "global_diag", "seed": seed} for seed in (2024, 2025, 2026)
        ]
        queues, loads = RUNNER.build_gpu_queues(units, [3])
        self.assertEqual(sum(len(value) for value in queues.values()), 3)
        self.assertEqual([row["seed"] for row in queues[3]], [2024, 2025, 2026])
        self.assertEqual(loads[3], 3 * RUNNER.STEP_SECONDS["global_diag"])

    def test_control_projection_is_complete_and_hash_bound(self) -> None:
        controls = HERE / "frozen_ex48_controls.csv"
        self.assertEqual(P.sha256_file(controls), self.contract["source_lineage"]["frozen_control_csv_sha256"])
        with controls.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 36)
        self.assertEqual({row["method"] for row in rows}, {"down_none", "down_diag", "newton_full", "muon"})
        self.assertEqual({int(row["seed"]) for row in rows}, {2024, 2025, 2026})
        self.assertEqual(len({(r["budget_id"], r["method"], r["seed"]) for r in rows}), 36)

    def test_paired_contrast_sign_convention(self) -> None:
        controls = [
            {"budget_id": b, "method": m, "seed": str(seed), "final_val_loss": "2.0"}
            for b in self.contract["analysis"]["primary_budgets"]
            for m in self.contract["analysis"]["comparators"]
            for seed in self.contract["grid"]["seeds"]
        ]
        new = [
            {"budget_id": b, "seed": seed, "final_val_loss": "1.9"}
            for b in self.contract["analysis"]["primary_budgets"]
            for seed in self.contract["grid"]["seeds"]
        ]
        rows = ANALYZER.paired_contrasts(new, controls, self.contract)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(float(row["mean_difference"]) < 0 for row in rows))
        self.assertTrue(all(int(row["global_diag_better_seeds"]) == 3 for row in rows))

    def test_metric_and_summary_integrity_gates_tampering(self) -> None:
        phase = {
            "id": "test_phase",
            "parent": None,
            "start_step": 0,
            "target_step": 2,
            "schedule": "plateau",
            "role": "primary_endpoint",
            "retain_checkpoint": True,
        }
        contract = {
            "validation": {"regular_every_steps": 1},
            "training": {"tokens_per_update": 10, "gradient_accumulation_steps": 2},
            "data": {"prefetched_train_microbatches": 1},
            "profile": {"parameters": 5},
        }
        events = [("val", 0), ("train", 1), ("val", 1), ("train", 2), ("val", 2)]
        metrics = []
        for event, step in events:
            row = {field: "" for field in P.METRIC_FIELDS}
            row.update({
                "event": event,
                "phase_id": phase["id"],
                "schedule": phase["schedule"],
                "step": str(step),
                "segment_step": str(step),
                "loss": f"{3.0 - step / 10:.9f}",
                "tokens_seen": str(step * 10),
                "tokens_per_parameter": f"{step * 10 / 5:.12f}",
                "loader_consumed_batches": str(1 + step * 2),
                "wrap_count": "0",
            })
            metrics.append(row)
        checkpoint = {"path": "/tmp/checkpoint.pt", "sha256": "c" * 64, "bytes": 10, "retained": True}
        manifest = {"role": phase["role"], "checkpoint": checkpoint}
        summary = {
            "schema_version": ANALYZER.PHASE_SUMMARY_SCHEMA,
            "status": "completed",
            "engineering_pilot": False,
            "method": "global_diag",
            "seed": 2024,
            "phase": phase,
            "completed_steps": 2,
            "tokens_seen": 20,
            "tokens_per_parameter": 4.0,
            "metrics_sha256": "m" * 64,
            "final_val_loss": float(metrics[-1]["loss"]),
            "final_train_loss": float(metrics[-2]["loss"]),
            "checkpoint_path": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_bytes": checkpoint["bytes"],
        }

        def audit(rows, payload):
            return ANALYZER.metric_summary_integrity_checks(
                rows, payload, manifest, phase, "global_diag", 2024,
                contract, "m" * 64,
            )

        self.assertTrue(all(audit(metrics, summary).values()))

        bad_summary = copy.deepcopy(summary)
        bad_summary["final_val_loss"] += 0.1
        self.assertFalse(audit(metrics, bad_summary)["summary_endpoint_values"])

        missing = copy.deepcopy(metrics)
        missing.pop(2)
        self.assertFalse(audit(missing, summary)["metric_event_grid"])

        duplicated = copy.deepcopy(metrics)
        duplicated.insert(2, copy.deepcopy(duplicated[1]))
        self.assertFalse(audit(duplicated, summary)["metric_event_grid"])

        wrong_identity = copy.deepcopy(metrics)
        wrong_identity[0]["phase_id"] = "other_phase"
        self.assertFalse(audit(wrong_identity, summary)["metric_row_identity"])

        wrong_geometry = copy.deepcopy(metrics)
        wrong_geometry[1]["tokens_seen"] = "999"
        self.assertFalse(audit(wrong_geometry, summary)["metric_row_geometry"])

        wrapped = copy.deepcopy(metrics)
        wrapped[-1]["wrap_count"] = "1"
        self.assertFalse(audit(wrapped, summary)["metric_loader_cursor"])

    def test_native_verify_receipt_is_persisted(self) -> None:
        source = (HERE / "run_formal.py").read_text(encoding="utf-8")
        self.assertIn('native_verify_full.json', source)
        self.assertIn('payload.get("full_checkpoint_hash") is not True', source)
        self.assertIn('"native_full_checkpoint_verify"', source)

    def test_retired_forks_are_not_rehashed(self) -> None:
        source = (HERE / "analyze_formal.py").read_text(encoding="utf-8")
        self.assertIn('(not expected_retained) or not full_checkpoint_hash', source)
        self.assertIn('"preconditioner_group_count"', source)

    def test_fork_retirement_is_certificate_first_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            unit = Path(temporary) / "formal" / "global_diag" / "seed2024"
            phase = unit / "fork"
            phase.mkdir(parents=True)
            checkpoint = phase / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            digest = hashlib.sha256(b"checkpoint").hexdigest()
            P.atomic_json(phase / "phase_manifest.json", {
                "checkpoint": {"path": str(checkpoint), "sha256": digest, "bytes": 10}
            })
            contract = {"phases": [
                {"id": "fork", "parent": None},
                {"id": "child", "parent": "fork"},
            ]}
            with mock.patch.object(RUNNER, "phase_manifest_valid", return_value=True):
                RUNNER.retire_fork(unit, "fork", contract)
                self.assertTrue((phase / "checkpoint_retirement.json").is_file())
                self.assertFalse(checkpoint.exists())
                # Emulate an interruption after the certificate write but
                # before cleanup; resume must safely finish the deletion.
                checkpoint.write_bytes(b"checkpoint")
                RUNNER.retire_fork(unit, "fork", contract)
                self.assertFalse(checkpoint.exists())

    def test_pilot_retirement_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index in range(2):
                path = root / f"pilot{index}.pt"
                payload = f"pilot-{index}".encode()
                path.write_bytes(payload)
                rows.append({
                    "path": str(path), "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                })
            manifest = root / "pilot_manifest.json"
            P.atomic_json(manifest, {"passed": True, "retired_pilot_checkpoints": rows})
            receipt = RUNNER.complete_pilot_retirement(manifest)
            self.assertTrue(receipt["retirement_completed_at"])
            self.assertTrue(all(not Path(row["path"]).exists() for row in rows))
            RUNNER.complete_pilot_retirement(manifest)

    def test_command_wrapper_has_restart_safe_all_mode(self) -> None:
        wrapper = (REPO / "commands/56_llama1b_10b_global_diag/20260817_ex56_llama1b_10b_global_diag.sh").read_text(encoding="utf-8")
        self.assertIn("EX56_OFFICIAL_REPO", wrapper)
        self.assertIn("Newton-Muon-official-r0", wrapper)
        self.assertIn("suite_plan.json", wrapper)
        self.assertIn("run_controller resume", wrapper)
        self.assertNotIn("fineweb10B_ex52_frozen50", wrapper)


if __name__ == "__main__":
    unittest.main()
