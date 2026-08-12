"""Local, GPU-free structural tests for the R1 extended-baseline builder."""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path

from extended_source_builder import (
    ALLOWED_METHODS,
    OFFICIAL_ADAM_CANONICAL_SHA256,
    OFFICIAL_ADAM_SCRIPT,
    build_source,
    canonical_sha256,
)
from run_r1_extended_baselines import CENTER_CELL_IDS, PILOT_CELLS, warmdown_steps


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parents[1]
_official_env = os.environ.get("SNM_OFFICIAL_REPO")
_official_candidates = (
    ARTIFACT_ROOT / "third_party" / "Newton-Muon-official-r0",
    ARTIFACT_ROOT / "third_party" / "Newton-Muon-official",
)
OFFICIAL_REPO = (
    Path(_official_env).expanduser().resolve()
    if _official_env
    else next((path for path in _official_candidates if path.is_dir()), _official_candidates[0])
)


class ExtendedSourceBuilderTests(unittest.TestCase):
    @unittest.skipUnless(
        OFFICIAL_REPO.is_dir(), f"pinned upstream repo unavailable: {OFFICIAL_REPO}"
    )
    def test_pinned_official_adam_hash(self) -> None:
        observed = canonical_sha256((OFFICIAL_REPO / OFFICIAL_ADAM_SCRIPT).read_bytes())
        self.assertEqual(observed, OFFICIAL_ADAM_CANONICAL_SHA256)

    @unittest.skipUnless(
        OFFICIAL_REPO.is_dir(), f"pinned upstream repo unavailable: {OFFICIAL_REPO}"
    )
    def test_all_derived_sources_compile_and_embed_required_audits(self) -> None:
        hashes = set()
        for method in ALLOWED_METHODS:
            derived = build_source(OFFICIAL_REPO, method)
            ast.parse(derived.source)
            hashes.add(derived.derived_sha256)
            self.assertIn("R1X_METADATA ", derived.source)
            self.assertIn("R1X_ROUTING ", derived.source)
            self.assertIn("R1X_HYPERPARAMS ", derived.source)
            self.assertIn("R1X_FINAL_MEMORY ", derived.source)
            self.assertIn("R1X_DISABLE_CHECKPOINT", derived.source)
            self.assertIn(f'R1X_METHOD != "{method}"', derived.source)
        self.assertEqual(len(hashes), len(ALLOWED_METHODS))

    @unittest.skipUnless(
        OFFICIAL_REPO.is_dir(), f"pinned upstream repo unavailable: {OFFICIAL_REPO}"
    )
    def test_optimizer_specific_semantics_are_not_mixed(self) -> None:
        adamw = build_source(OFFICIAL_REPO, "adamw").source
        normuon = build_source(OFFICIAL_REPO, "normuon").source
        moonlight = build_source(OFFICIAL_REPO, "moonlight_muon").source
        self.assertIn("optimizer2 = torch.optim.AdamW", adamw)
        self.assertNotIn("optimizer2 = R1NorMuon", adamw)
        self.assertIn("optimizer2 = R1NorMuon", normuon)
        self.assertIn("beta2=0.95", normuon)
        self.assertIn("eps=1e-10", normuon)
        self.assertIn("optimizer2 = R1MoonlightMuon", moonlight)
        self.assertIn("weight_decay=R1X_WEIGHT_DECAY", moonlight)
        self.assertIn("eps=1e-8", moonlight)

    def test_pilot_grid_contains_authority_and_r1_scale_cells(self) -> None:
        ids = {cell.cell_id for cell in PILOT_CELLS}
        self.assertEqual(len(ids), 9)
        self.assertTrue(set(CENTER_CELL_IDS) <= ids)
        per_method = {
            method: [cell for cell in PILOT_CELLS if cell.method == method]
            for method in ALLOWED_METHODS
        }
        self.assertTrue(all(len(cells) == 3 for cells in per_method.values()))
        for cell in per_method["adamw"]:
            self.assertAlmostEqual(cell.matrix_lr, 0.16 * cell.auxiliary_lr)
            self.assertEqual(cell.weight_decay, 0.0)
        self.assertIn("normuon_official", ids)
        self.assertIn("moonlight_official", ids)
        self.assertIn("normuon_r1scale", ids)
        self.assertIn("moonlight_r1scale", ids)

    def test_short_run_is_a_formal_prefix_schedule(self) -> None:
        self.assertEqual(warmdown_steps(10), 1)
        self.assertEqual(warmdown_steps(1000), 1)

    def test_optimizer_module_declares_qkv_and_reference_audits(self) -> None:
        source = (SCRIPT_DIR / "extended_optimizers.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("rows == 3 * cols", source)
        self.assertIn("second_momentum_buffer", source)
        self.assertIn("0.2 * math.sqrt", source)
        self.assertIn("run_single_step_reference_audit", source)
        self.assertIn("adamw_decoupled_single_step", source)
        self.assertIn("packed_vs_split_distinct", source)


if __name__ == "__main__":
    unittest.main()
