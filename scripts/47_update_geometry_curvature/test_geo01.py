#!/usr/bin/env python3
"""Local CPU tests for experiment 47 / GEO-01."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_geo01 as A
import geometry_core as G
import protocol as P
import remote_controller as R


class QuadraticToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]],
                dtype=torch.float64,
            )
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, **_: object):
        prediction = x @ self.weight.t()
        return prediction, (prediction - y).square().mean()


class Geo01Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(47)

    def test_contract_is_frozen_pilot_only(self) -> None:
        contract = P.read_json(HERE / "geo01_contract.json")
        checks = P.validate_contract(contract)
        self.assertTrue(all(checks.values()), checks)
        offsets = P.build_offset_certificate(contract)
        self.assertTrue(offsets["passed"], offsets)
        self.assertFalse(contract["execution"]["discovery_enabled"])
        self.assertFalse(contract["execution"]["confirmation_enabled"])
        self.assertFalse(contract["claim_boundary"]["llama_10b_triggered"])

    def test_execution_contract_derivation_is_pilot_bounded(self) -> None:
        contract = P.read_json(HERE / "geo01_contract.json")
        base = P.read_json(
            HERE.parent
            / "37_mech09_downproj_refresh_mediation"
            / "refresh_mediation_repair_contract.json"
        )
        derived = P.derive_execution_contract(base, contract, "a" * 64)
        self.assertEqual(derived["formal"]["origins"], ["early_muon"])
        self.assertEqual(derived["formal"]["data_replicas"], [7])
        self.assertEqual(derived["smoke"]["data_replicas"], [8])
        # This is the accepted source-worker safety cap, not the GEO-01
        # scheduler count. The outer pilot plan still contains one unit.
        self.assertEqual(derived["stopping_rule"]["maximum_new_formal_jobs"], 12)
        compatibility = P.validate_derived_execution_contract(derived, contract)
        self.assertTrue(all(compatibility.values()), compatibility)

    def test_counterfactual_direction_matches_manual_identity_ns(self) -> None:
        gradient = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
        )
        momentum = torch.tensor(
            [[0.5, -0.5], [1.0, -1.0]], dtype=torch.float32
        )
        before = torch.eye(2, dtype=torch.float32)
        after = torch.diag(torch.tensor([2.0, 0.5]))
        beta = 0.8
        lr = 0.02

        def identity_ns(value: torch.Tensor, _: int) -> torch.Tensor:
            return value

        direction, audit = G.counterfactual_update_direction(
            raw_gradient=gradient,
            historical_momentum=momentum,
            inverse_reference=before,
            inverse_treatment=after,
            momentum_beta=beta,
            learning_rate=lr,
            ns_steps=5,
            ns_update=identity_ns,
        )

        def update(preconditioned: torch.Tensor) -> torch.Tensor:
            next_momentum = momentum.lerp(preconditioned, 1.0 - beta)
            return torch.lerp(preconditioned, next_momentum, beta)

        expected = -lr * (
            update(gradient @ after) - update(gradient @ before)
        )
        torch.testing.assert_close(direction, expected)
        self.assertTrue(audit["finite"])
        self.assertTrue(audit["nonzero"])

    def test_directional_hvp_and_line_loss_close_on_quadratic(self) -> None:
        model = QuadraticToy()
        batches = [
            (
                torch.randn(7, 3, dtype=torch.float64),
                torch.randn(7, 2, dtype=torch.float64),
            ),
            (
                torch.randn(5, 3, dtype=torch.float64),
                torch.randn(5, 2, dtype=torch.float64),
            ),
        ]
        direction = {
            "weight": torch.tensor(
                [[1.0e-3, -2.0e-3, 3.0e-3], [2.0e-3, 1.0e-3, -1.0e-3]],
                dtype=torch.float64,
            )
        }
        before = G.tensor_sha256(model.weight)
        row = G.measure_directional_geometry(
            model=model,
            batches=batches,
            named_direction=direction,
        )
        self.assertTrue(row["all_values_finite"])
        self.assertTrue(row["parameters_unchanged"])
        self.assertEqual(row["attention_backend"], "math_only_for_second_order")
        self.assertEqual(before, G.tensor_sha256(model.weight))
        self.assertLessEqual(row["fd_first_relative_error"], 1.0e-8)
        self.assertLessEqual(row["fd_curvature_relative_error"], 1.0e-7)
        self.assertLessEqual(abs(row["taylor_residual"]), 1.0e-11)

    def test_analyzer_refuses_disabled_confirmation(self) -> None:
        contract = P.read_json(HERE / "geo01_contract.json")
        with self.assertRaises(RuntimeError):
            A.summarize([], "confirmation", contract)

    def test_pilot_analyzer_has_no_claim_upgrade(self) -> None:
        contract = P.read_json(HERE / "geo01_contract.json")
        rows = []
        for spec in contract["pilot"]["scopes"]:
            rows.append(
                {
                    "scope_id": spec["scope_id"],
                    **{field: 0.1 for field in A.REQUIRED_NUMERIC},
                    "parameters_unchanged": True,
                }
            )
        summary = A.summarize(rows, "pilot", contract)
        self.assertTrue(summary["integrity_passed"])
        self.assertFalse(summary["claim_eligible"])
        self.assertEqual(
            summary["scientific_result"],
            "engineering_pilot_only_no_scientific_claim",
        )

    def test_remote_controller_full_dry_run_is_source_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "accepted_source"
            (source / "formal").mkdir(parents=True)
            P.atomic_json(
                source / "formal" / "formal_manifest.json",
                {"passed": True, "completed_jobs": 12},
            )
            pinned = root / "pinned_source_file.py"
            pinned.write_text("# source-pinned test fixture\n", encoding="utf-8")
            rows = []
            for origin in (
                "early_muon",
                "early_newton_full",
                "late_muon",
                "late_newton_full",
            ):
                arguments = [
                    sys.executable,
                    str(pinned),
                    "--output-dir",
                    str(root / "old_output"),
                    "--analysis-tier",
                    "formal",
                    "--cell",
                    origin,
                    "--data-replica",
                    "0",
                    "--contract",
                    str(pinned),
                    "--triton-kernels",
                    str(pinned),
                    "--mech08-control-reference",
                    str(pinned),
                    "--checkpoint",
                    str(pinned),
                    "--checkpoint-hash-certificate",
                    str(pinned),
                    "--source-script",
                    str(pinned),
                    "--profile-script",
                    str(pinned),
                    "--smoke-manifest",
                    str(pinned),
                ]
                rows.append(
                    {
                        "label": f"formal/{origin}/replica_0",
                        "command": arguments,
                    }
                )
            (source / "commands.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            run_dir = root / "dry_run"
            result = R.controller(
                argparse.Namespace(
                    mode="dry-run",
                    run_dir=run_dir,
                    source_run=source,
                    child_python=Path(sys.executable),
                    gpus=["0", "1"],
                )
            )
            self.assertEqual(result, 0)
            status = P.read_json(run_dir / "status.json")
            self.assertEqual(status["status"], "dry_run_passed")
            plan = P.read_json(run_dir / "sealed" / "pilot_plan.json")
            self.assertEqual(plan["formal_units"], 1)
            derived = P.read_json(
                run_dir / "sealed" / "derived_execution_contract.json"
            )
            self.assertEqual(
                derived["stopping_rule"]["maximum_new_formal_jobs"], 12
            )

    def test_runtime_preflight_normalizes_version_objects(self) -> None:
        source = Path(R.__file__).read_text(encoding="utf-8")
        self.assertIn('"torch": str(torch.__version__)', source)
        self.assertIn('"numpy": str(numpy.__version__)', source)
        self.assertIn('"executable": str(sys.executable)', source)
        self.assertIn('"observed": observed', source)
        self.assertIn('"expected": expected', source)
        self.assertIn('["requested_executable"] = requested_executable', source)

    def test_launcher_pins_dedicated_training_python(self) -> None:
        launcher = (
            HERE.parent.parent
            / "commands"
            / "47_update_geometry_curvature"
            / "20260804_ex47_update_geometry_curvature.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("GEO01_TRAINING_PYTHON", launcher)
        self.assertIn(
            "${SNM_TRAINING_PYTHON}",
            launcher,
        )
        self.assertIn('--child-python "$TRAINING_PYTHON"', launcher)
        self.assertNotIn('CHILD_PYTHON="${CHILD_PYTHON', launcher)

    def test_training_python_path_is_not_symlink_resolved(self) -> None:
        relative = Path("frozen_venv") / "bin" / "python"
        observed = R.absolute_without_resolving(relative)
        self.assertEqual(observed, Path.absolute(relative))
        source = Path(R.__file__).read_text(encoding="utf-8")
        self.assertIn("child_python = absolute_without_resolving", source)
        self.assertNotIn("child_python = args.child_python.resolve()", source)
        self.assertIn('"virtualenv_active"', source)


if __name__ == "__main__":
    unittest.main()
