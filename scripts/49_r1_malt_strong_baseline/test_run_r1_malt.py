from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import run_r1_malt as ex49


def args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "formal": False,
        "formal_smoke": False,
        "pilot": False,
        "numerical_smoke": False,
        "preflight": False,
        "pilot_steps": 1000,
        "smoke_steps": 34,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def pilot_summaries(
    *,
    malt_best_cell_id: str = "malt_lr0100",
    malt_best_loss: float = 3.0,
    malter_center_loss: float = 3.0019,
    malter_internal_loss: float = 3.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, cell in enumerate(ex49.PILOT_CELLS):
        loss = 3.2 + index * 0.001
        if cell.cell_id == malt_best_cell_id:
            loss = malt_best_loss
        if cell.cell_id == "malter_eq17_lr015":
            loss = malter_internal_loss
        if cell.cell_id == ex49.MALTER_CENTER_CELL_ID:
            loss = malter_center_loss
        rows.append(
            {
                **ex49.asdict(cell),
                "method": cell.method,
                "final_val_loss": loss,
                "evidence_valid": True,
            }
        )
    return rows


class MALTControllerTests(unittest.TestCase):
    def test_frozen_grid_and_evidence_roles(self) -> None:
        self.assertEqual(len(ex49.PILOT_CELLS), 12)
        self.assertEqual(sum(cell.method == "malt" for cell in ex49.PILOT_CELLS), 6)
        self.assertEqual(sum(cell.method == "malter_eq17" for cell in ex49.PILOT_CELLS), 6)
        self.assertTrue(all(cell.formal_eligible for cell in ex49.PILOT_CELLS))
        self.assertEqual(
            [cell.matrix_lr for cell in ex49.PILOT_CELLS[:6]],
            [0.0160, 0.0125, 0.0100, 0.0090, 0.0080, 0.0064],
        )
        self.assertEqual(ex49.MALT_LOWER_BOUNDARY_CELL_ID, "malt_lr0064")
        self.assertEqual(ex49.MALT_UPPER_BOUNDARY_CELL_ID, "malt_lr0160")
        self.assertEqual(ex49.MATRIX_WEIGHT_DECAY, 0.1)

    def test_formal_budget_is_frozen(self) -> None:
        value = args(formal=True)
        self.assertEqual(ex49.total_steps(value), 6200)
        self.assertEqual(ex49.warmdown_steps(value), 1800)
        self.assertEqual(ex49.protocol(value), ex49.FORMAL_PROTOCOL)

    def test_malt_raw_endpoint_and_malter_center_rule_select_both_methods(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "pilot_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            payload = json.loads(
                ex49.make_selection(
                    root,
                    pilot_summaries(),
                    manifest,
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "selected")
            self.assertEqual(payload["selections"]["malt"]["selected_cell_id"], "malt_lr0100")
            self.assertEqual(payload["selections"]["malt"]["selection_policy"], "raw_endpoint_best")
            self.assertIsNone(payload["selections"]["malt"]["center_cell_id"])
            self.assertIsNone(payload["selections"]["malt"]["center_tie_margin"])
            self.assertEqual(
                payload["selections"]["malter_eq17"]["selected_cell_id"],
                ex49.MALTER_CENTER_CELL_ID,
            )
            self.assertEqual(
                payload["selections"]["malter_eq17"]["selection_policy"],
                "paper_center_within_best_plus_0.002",
            )
            self.assertTrue(payload["formal_eligible"])
            self.assertEqual(payload["malter_eq17_role"], "formal_candidate")

    def test_upper_boundary_tied_at_minimum_blocks_formal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "pilot_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            rows = pilot_summaries()
            for row in rows:
                if row["cell_id"] == ex49.MALT_UPPER_BOUNDARY_CELL_ID:
                    row["final_val_loss"] = 3.0
            payload = json.loads(ex49.make_selection(root, rows, manifest).read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "boundary_inconclusive")
            self.assertFalse(payload["formal_eligible"])
            self.assertEqual(payload["raw_best_cell_id"], "malt_lr0100")
            self.assertEqual(
                payload["selections"]["malt"]["minimum_tied_cell_ids"],
                ["malt_lr0100", ex49.MALT_UPPER_BOUNDARY_CELL_ID],
            )
            self.assertEqual(payload["selections"]["malt"]["boundary_side"], "upper")
            with self.assertRaises(RuntimeError):
                ex49.validate_selection(root / "pilot_selection.json", "malt")

    def test_raw_lower_boundary_also_blocks_formal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "pilot_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            rows = pilot_summaries()
            for row in rows:
                if row["cell_id"] == ex49.MALT_LOWER_BOUNDARY_CELL_ID:
                    row["final_val_loss"] = 3.0
            payload = json.loads(
                ex49.make_selection(root, rows, manifest).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "boundary_inconclusive")
            self.assertEqual(payload["boundary_side"], "lower")
            self.assertFalse(payload["formal_eligible"])

    def test_malter_upper_boundary_blocks_complete_panel(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "pilot_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            rows = pilot_summaries()
            for row in rows:
                if row["cell_id"] == ex49.MALTER_UPPER_BOUNDARY_CELL_ID:
                    row["final_val_loss"] = 3.0
            payload = json.loads(
                ex49.make_selection(root, rows, manifest).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "boundary_inconclusive")
            self.assertEqual(payload["blocking_methods"], ["malter_eq17"])
            self.assertFalse(payload["formal_allowed"])

    def test_runner_generated_nonboundary_selection_cannot_start_formal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "pilot_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            selection = ex49.make_selection(
                root,
                pilot_summaries(),
                manifest,
            )
            payload = json.loads(selection.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "selected")
            self.assertEqual(payload["certificate_role"], "runner_preselection_crosscheck")
            with self.assertRaisesRegex(RuntimeError, "independent pilot analyzer"):
                ex49.validate_selection(selection, "malt")

    def test_v4_analyzer_certificate_accepts_both_methods_and_rejects_v3(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            selection = root / "pilot_selection_verified.json"
            manifest = root / "pilot_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            payload = json.loads(
                ex49.make_selection(root, pilot_summaries(), manifest).read_text(encoding="utf-8")
            )
            payload["certificate_role"] = "independent_pilot_analysis_selection"
            selection.write_text(json.dumps(payload), encoding="utf-8")
            malt_cell, _ = ex49.validate_selection(selection, "malt")
            malter_cell, _ = ex49.validate_selection(selection, "malter_eq17")
            self.assertEqual(malt_cell.cell_id, "malt_lr0100")
            self.assertEqual(malter_cell.cell_id, ex49.MALTER_CENTER_CELL_ID)

            payload["protocol"] = "malt_r1_extended_grid_selection_v3"
            selection.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "status/protocol mismatch"):
                ex49.validate_selection(selection, "malt")

    def test_validation_grid_includes_terminal_step(self) -> None:
        self.assertEqual(ex49.validation_steps(34, 100), [0, 34])
        self.assertEqual(ex49.validation_steps(1000, 100)[-1], 1000)

    def test_smoke_validation_is_method_and_state_aware(self) -> None:
        runtime = {
            "python": "3.10.12",
            "python_executable": "/frozen/train/python",
            "numpy": "2.2.6",
            "torch": "2.8.0+cu126",
            "torch_cuda": "12.6",
            "triton": "3.4.0",
            "triton_module": "/frozen/triton.py",
            "triton_kernels_module": "/frozen/triton_kernels.py",
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "gpu_total_memory_bytes": 85_169_143_808,
        }
        derived = SimpleNamespace(derived_sha256="d" * 64)
        init = {"init_sha256": "i" * 64}
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for method, cell_id, label, state_bytes, nu_bytes in (
                (
                    "malt",
                    "malt_lr0100",
                    "MALT-R1 adaptation",
                    ex49.MALT_EXPECTED_HIDDEN_STATE_BYTES,
                    0,
                ),
                (
                    "malter_eq17",
                    "malter_eq17_lr015",
                    "MALTER-Eq17-R1 adaptation",
                    ex49.MALTER_EXPECTED_HIDDEN_STATE_BYTES,
                    288,
                ),
            ):
                cell = next(item for item in ex49.PILOT_CELLS if item.cell_id == cell_id)
                roles = {
                    "malt_momentum": 48,
                    "malt_row_ema": 72,
                    "malt_col_ema": 72,
                }
                if method == "malter_eq17":
                    roles["malt_nu"] = 72
                payload = {
                    "status": "completed_valid",
                    "protocol": ex49.SMOKE_PROTOCOL,
                    "seed": 2024,
                    "total_steps": ex49.SMOKE_STEPS,
                    "cell": ex49.asdict(cell),
                    "initialization_audit": init,
                    "source_audit": {"derived_source_sha256": derived.derived_sha256},
                    "training_runtime_fingerprint": ex49.r0.runtime_fingerprint(runtime),
                    "summary": {
                        "evidence_valid": True,
                        "method": method,
                        "adaptation_label": label,
                        "hidden_optimizer_state_bytes": state_bytes,
                        "malt_nu_bytes": nu_bytes,
                        "state_schema": {"roles": roles},
                    },
                }
                path = root / f"{method}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertEqual(
                    ex49.validate_smoke(
                        path,
                        args(seed=2024),
                        cell,
                        runtime,
                        init,
                        derived,
                    )["status"],
                    "accepted",
                )

    def test_exact_runtime_contract_rejects_version_drift(self) -> None:
        runtime = {
            "python": "3.10.12 (main, test build)",
            "torch": "2.8.0+cu126",
            "torch_cuda": "12.6",
            "triton": "3.4.0",
            "numpy": "2.2.6",
            "python_executable": "/frozen/venv/bin/python",
        }
        self.assertEqual(
            ex49.validate_exact_training_runtime(runtime)["status"], "passed"
        )
        runtime["torch"] = "2.8.1+cu126"
        with self.assertRaisesRegex(RuntimeError, "exact training runtime mismatch"):
            ex49.validate_exact_training_runtime(runtime)


if __name__ == "__main__":
    unittest.main()
