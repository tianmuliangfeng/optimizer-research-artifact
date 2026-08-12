from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
ARTIFACT_ROOT = HERE.parents[1]
_official_env = os.environ.get("SNM_OFFICIAL_REPO")
_official_candidates = (
    ARTIFACT_ROOT / "third_party" / "Newton-Muon-official-r0",
    ARTIFACT_ROOT / "third_party" / "Newton-Muon-official",
)
OFFICIAL = (
    Path(_official_env).expanduser().resolve()
    if _official_env
    else next((path for path in _official_candidates if path.is_dir()), _official_candidates[0])
)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import r1_dense_full_alpha_source_builder as builder  # noqa: E402
import run_r1_dense_full_alpha as runner  # noqa: E402


@unittest.skipUnless(OFFICIAL.is_dir(), f"pinned upstream repo unavailable: {OFFICIAL}")
class DenseFullAlphaSourceTests(unittest.TestCase):
    def test_reference_math_and_method_mapping(self) -> None:
        builder.self_test_alpha_math()
        self.assertEqual(
            builder.ALPHA_BY_METHOD,
            {
                "fullalpha0": 0.0,
                "fullalpha0p25": 0.25,
                "fullalpha0p50": 0.5,
                "fullalpha0p75": 0.75,
                "fullalpha1": 1.0,
            },
        )

    def test_all_sources_are_distinct_and_compile(self) -> None:
        hashes = set()
        for method in builder.ALLOWED_METHODS:
            built = builder.build_source(OFFICIAL, method)
            compile(built.source, f"<{method}>", "exec")
            hashes.add(built.derived_sha256)
            self.assertIn('"kind": "c_proj_full"', built.source)
            self.assertIn("R1_DENSE_FULL_ALPHA", built.source)
            self.assertIn("raw_cross_to_within", built.source)
            self.assertIn("cosine_vs_diag", built.source)
        self.assertEqual(len(hashes), len(builder.ALLOWED_METHODS))

    def test_endpoints_embed_exact_alpha(self) -> None:
        zero = builder.build_source(OFFICIAL, "fullalpha0").source
        one = builder.build_source(OFFICIAL, "fullalpha1").source
        self.assertIn("R1_DENSE_FULL_ALPHA = 0.0", zero)
        self.assertIn("R1_DENSE_FULL_ALPHA = 1.0", one)
        self.assertIn("work.mul_(R1_DENSE_FULL_ALPHA)", zero)


class DenseFullAlphaRunnerTests(unittest.TestCase):
    def test_smoke_must_cross_first_inverse_refresh(self) -> None:
        with mock.patch("sys.argv", ["runner", "--numerical-smoke", "--smoke-steps", "33"]):
            with self.assertRaises(SystemExit):
                runner.parse_args()

    def test_formal_nonpilot_seed_requires_explicit_release(self) -> None:
        with mock.patch("sys.argv", ["runner", "--seed", "2025", "--smoke-manifest", "x.json"]):
            with self.assertRaises(SystemExit):
                runner.parse_args()

    def test_legacy_expansion_flag_does_not_relabel_failed_gate(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "runner",
                "--seed",
                "2025",
                "--allow-seed-expansion",
                "--smoke-manifest",
                "x.json",
            ],
        ):
            with self.assertRaises(SystemExit):
                runner.parse_args()

    def test_confirmatory_protocol_requires_complete_grid(self) -> None:
        with mock.patch(
            "sys.argv",
            ["runner", "--seed", "2025", "--confirmatory", "--dry-run"],
        ):
            args = runner.parse_args()
        self.assertEqual(args.methods, list(builder.ALLOWED_METHODS))
        self.assertEqual(
            runner.experiment_protocol(args),
            runner.CONFIRMATORY_PROTOCOL,
        )
        with mock.patch(
            "sys.argv",
            [
                "runner",
                "--seed",
                "2025",
                "--confirmatory",
                "--methods",
                "fullalpha0",
                "--dry-run",
            ],
        ):
            with self.assertRaises(SystemExit):
                runner.parse_args()

    def test_manifest_contract_records_topology(self) -> None:
        captured: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as temporary:
            old = runner._base_write_json
            try:
                runner._base_write_json = lambda _path, payload: captured.update(payload)
                runner.write_json_with_alpha_contract(
                    Path(temporary) / "r1_plan.json", {"status": "running"}
                )
            finally:
                runner._base_write_json = old
        design = captured["dense_full_alpha_design"]
        self.assertIn("cross-block", design["topology"])
        self.assertEqual(design["primary_endpoint"], "validation loss at step 6200")

    def test_diagnostic_parser(self) -> None:
        text = """R1_FULL_ALPHA_K step=31 alpha=0.5 raw_cross_to_within=1.2 scaled_offdiag_to_diag=0.3 chol_diag_spread=2.0 inv_offdiag_to_diag=0.4 inv_diag_rms=5.0 cholesky_failures=0
R1_FULL_ALPHA_UPDATE step=31 alpha=0.5 norm_ratio_vs_diag=0.9 cosine_vs_diag=0.8
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training_stdout.log"
            path.write_text(text, encoding="utf-8")
            rows = runner._parse_diagnostics(path, "fullalpha0p50")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["cholesky_failures"], 0)
        self.assertAlmostEqual(rows[1]["cosine_vs_diag"], 0.8)


if __name__ == "__main__":
    unittest.main()
