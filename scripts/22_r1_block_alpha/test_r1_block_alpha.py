from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import r1_block_alpha_source_builder as builder
import run_r1_block_alpha as runner


class BlockAlphaContractTests(unittest.TestCase):
    def test_reference_endpoints(self) -> None:
        builder.self_test_alpha_math()

    def test_method_mapping_is_predeclared(self) -> None:
        self.assertEqual(
            builder.ALPHA_BY_METHOD,
            {"alpha0": 0.0, "alpha0p25": 0.25, "alpha0p50": 0.5, "alpha0p75": 0.75},
        )

    def test_smoke_must_cross_first_inverse_refresh(self) -> None:
        with mock.patch("sys.argv", ["runner", "--numerical-smoke", "--smoke-steps", "33"]):
            with self.assertRaises(SystemExit):
                runner.parse_args()

    def test_manifest_enrichment_records_alpha_contract(self) -> None:
        captured: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as temp:
            old = runner._base_write_json
            try:
                runner._base_write_json = lambda _path, payload: captured.update(payload)
                runner.write_json_with_alpha_contract(Path(temp) / "r1_plan.json", {"status": "running"})
            finally:
                runner._base_write_json = old
        self.assertEqual(captured["block_alpha_design"]["primary_endpoint"], "validation loss at step 6200")
        self.assertFalse(captured["interpretation_boundary"]["timing"].startswith("usable"))

    def test_nonpilot_seed_requires_confirmatory_label(self) -> None:
        with mock.patch("sys.argv", ["runner", "--seed", "2024", "--dry-run"]):
            with self.assertRaises(SystemExit):
                runner.parse_args()

    def test_confirmatory_protocol_and_complete_grid(self) -> None:
        with mock.patch(
            "sys.argv",
            ["runner", "--seed", "2024", "--confirmatory", "--dry-run"],
        ):
            args = runner.parse_args()
        self.assertEqual(args.methods, list(builder.ALLOWED_METHODS))
        self.assertEqual(
            runner.experiment_protocol(args),
            runner.CONFIRMATORY_PROTOCOL,
        )
        self.assertEqual(
            runner.experiment_protocol(args, smoke=True),
            runner.SMOKE_PROTOCOL,
        )


if __name__ == "__main__":
    unittest.main()
