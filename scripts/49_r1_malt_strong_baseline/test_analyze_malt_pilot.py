from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import analyze_malt_pilot as pilot
import run_r1_malt as runner


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYZER = SCRIPT_DIR / "analyze_malt_pilot.py"
MALT = (
    ("malt_lr0160", 0.0160),
    ("malt_lr0125", 0.0125),
    ("malt_lr0100", 0.0100),
    ("malt_lr0090", 0.0090),
    ("malt_lr0080", 0.0080),
    ("malt_lr0064", 0.0064),
)
MALTER = (
    ("malter_eq17_lr007", 0.007),
    ("malter_eq17_lr009", 0.009),
    ("malter_eq17_lr012", 0.012),
    ("malter_eq17_lr015", 0.015),
    ("malter_eq17_lr018", 0.018),
    ("malter_eq17_lr025", 0.025),
)


def write_batch(
    root: Path,
    losses: dict[str, float],
    *,
    status: str = "completed_valid",
    protocol: str = "malt_r1_focused_grid_pilot_v4",
) -> Path:
    rows: list[dict[str, object]] = []
    for method, cells in (("malt", MALT), ("malter_eq17", MALTER)):
        for cell_id, lr in cells:
            loss = losses[cell_id]
            rows.append(
                {
                    "cell_id": cell_id,
                    "method": method,
                    "matrix_lr": lr,
                    "formal_eligible": True,
                    "controlled_seed": 2026,
                    "total_steps": 1000,
                    "total_tokens": 524_288_000,
                    "evidence_valid": True,
                    "final_val_loss": loss,
                    "val_loss_step_1000": loss,
                }
            )
    manifest = {
        "status": status,
        "family": "49_r1_malt_strong_baseline",
        "protocol": protocol,
        "seed": 2026,
        "total_steps": 1000,
        "failures": [],
        "summaries": rows,
    }
    (root / "pilot_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    with (root / "pilot_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def default_losses() -> dict[str, float]:
    values = {
        "malt_lr0160": 3.58,
        "malt_lr0125": 3.52,
        "malt_lr0100": 3.50,
        "malt_lr0090": 3.51,
        "malt_lr0080": 3.53,
        "malt_lr0064": 3.57,
        "malter_eq17_lr007": 3.48,
        "malter_eq17_lr009": 3.43,
        "malter_eq17_lr012": 3.401,
        "malter_eq17_lr015": 3.400,
        "malter_eq17_lr018": 3.42,
        "malter_eq17_lr025": 3.47,
    }
    return values


def write_matching_runner_selection(batch: Path) -> None:
    manifest_path = batch / "pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runner.make_selection(batch, manifest["summaries"], manifest_path)


def run_analyzer(batch: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ANALYZER), str(batch), "--output-dir", str(output)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def read_selection(output: Path) -> dict[str, object]:
    return json.loads(
        (output / "pilot_selection_verified.json").read_text(encoding="utf-8")
    )


class MaltPilotAnalyzerTests(unittest.TestCase):
    def test_selected_dual_method_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            batch = root / "batch"
            batch.mkdir()
            write_batch(
                batch,
                default_losses(),
                status="completed_valid_local_wandb_incomplete",
            )
            output = root / "analysis"
            result = run_analyzer(batch, output)
            self.assertEqual(result.returncode, 0, result.stdout)
            selection = read_selection(output)
            self.assertEqual(selection["protocol"], "malt_r1_focused_grid_selection_v4")
            self.assertEqual(selection["status"], "selected")
            self.assertTrue(selection["formal_allowed"])
            self.assertEqual(
                selection["certificate_role"],
                "independent_pilot_analysis_selection",
            )
            self.assertEqual(
                selection["required_formal_methods"], ["malt", "malter_eq17"]
            )
            malt = selection["selections"]["malt"]
            self.assertEqual(malt["selected_cell_id"], "malt_lr0100")
            self.assertEqual(malt["selection_policy"], "raw_endpoint_best")
            self.assertEqual(malt["selection_reason"], "raw_endpoint_best")
            self.assertIsNone(malt["center_cell_id"])
            self.assertIsNone(malt["center_tie_margin"])
            self.assertIsNone(malt["paper_center_lr"])
            self.assertFalse(malt["center_preferred_if_within_margin_of_best"])
            malter = selection["selections"]["malter_eq17"]
            self.assertEqual(malter["selected_cell_id"], "malter_eq17_lr012")
            self.assertEqual(
                malter["selection_reason"], "paper_center_within_best_plus_0.002"
            )
            manifest = json.loads(
                (output / "pilot_analysis_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["protocol"], "malt_r1_pilot_analysis_v4")
            self.assertTrue(manifest["checks"]["exact_twelve_cell_grid"])
            self.assertEqual(
                manifest["runner_selection_crosscheck"]["status"],
                "not_present_analyzer_selection_is_authoritative",
            )
            sidecar = (output / "pilot_analysis_manifest.sha256").read_text(
                encoding="ascii"
            )
            observed_hash, observed_name = sidecar.strip().split(maxsplit=1)
            self.assertEqual(observed_name, "pilot_analysis_manifest.json")
            self.assertEqual(
                observed_hash,
                hashlib.sha256((output / observed_name).read_bytes()).hexdigest(),
            )
            with (output / "pilot_ranking.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertEqual([row["method"] for row in rows[:6]], ["malt"] * 6)
            self.assertEqual(
                [row["method"] for row in rows[6:]], ["malter_eq17"] * 6
            )

    def test_malt_has_no_center_preference(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            batch = root / "batch"
            batch.mkdir()
            losses = default_losses()
            losses["malt_lr0090"] = 3.5000
            losses["malt_lr0080"] = 3.5010
            losses["malt_lr0100"] = 3.5100
            write_batch(batch, losses)
            result = run_analyzer(batch, root / "analysis")
            self.assertEqual(result.returncode, 0, result.stdout)
            malt = read_selection(root / "analysis")["selections"]["malt"]
            self.assertEqual(malt["selected_cell_id"], "malt_lr0090")
            self.assertEqual(malt["selection_reason"], "raw_endpoint_best")

    def test_malter_raw_best_outside_margin_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            batch = root / "batch"
            batch.mkdir()
            losses = default_losses()
            losses["malter_eq17_lr015"] = 3.4000
            losses["malter_eq17_lr012"] = 3.4021
            write_batch(batch, losses)
            result = run_analyzer(batch, root / "analysis")
            self.assertEqual(result.returncode, 0, result.stdout)
            malter = read_selection(root / "analysis")["selections"]["malter_eq17"]
            self.assertEqual(malter["selected_cell_id"], "malter_eq17_lr015")
            self.assertEqual(malter["selection_reason"], "raw_endpoint_best")

    def test_upper_boundary_raw_best_blocks_formal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            batch = root / "batch"
            batch.mkdir()
            losses = default_losses()
            losses["malt_lr0160"] = 3.3000
            write_batch(batch, losses)
            result = run_analyzer(batch, root / "analysis")
            self.assertEqual(result.returncode, 0, result.stdout)
            selection = read_selection(root / "analysis")
            self.assertEqual(selection["status"], "boundary_inconclusive")
            self.assertFalse(selection["formal_allowed"])
            self.assertEqual(selection["blocking_methods"], ["malt"])
            malt = selection["selections"]["malt"]
            self.assertEqual(malt["boundary_side"], "upper")
            self.assertTrue(malt["raw_best_is_upper_boundary"])
            self.assertIsNone(malt["selected_cell_id"])
            self.assertEqual(malt["selection_reason"], "boundary_inconclusive")

    def test_lower_boundary_raw_best_blocks_and_matches_runner(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            batch = root / "batch"
            batch.mkdir()
            losses = default_losses()
            losses["malt_lr0064"] = 3.3000
            write_batch(batch, losses)
            write_matching_runner_selection(batch)
            result = run_analyzer(batch, root / "analysis")
            self.assertEqual(result.returncode, 0, result.stdout)
            selection = read_selection(root / "analysis")
            malt = selection["selections"]["malt"]
            self.assertEqual(malt["boundary_side"], "lower")
            self.assertEqual(malt["lower_boundary_lr"], 0.0064)
            self.assertEqual(malt["upper_boundary_lr"], 0.016)
            manifest = json.loads(
                (root / "analysis/pilot_analysis_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["runner_selection_crosscheck"]["status"], "matched"
            )

    def test_boundary_tied_at_reported_precision_blocks_formal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            batch = root / "batch"
            batch.mkdir()
            losses = default_losses()
            losses["malt_lr0100"] = 3.3000
            losses["malt_lr0160"] = 3.3000
            write_batch(batch, losses)
            write_matching_runner_selection(batch)
            result = run_analyzer(batch, root / "analysis")
            self.assertEqual(result.returncode, 0, result.stdout)
            malt = read_selection(root / "analysis")["selections"]["malt"]
            self.assertEqual(malt["raw_best_cell_id"], "malt_lr0100")
            self.assertFalse(malt["raw_best_is_upper_boundary"])
            self.assertTrue(malt["minimum_includes_upper_boundary"])
            self.assertEqual(malt["boundary_side"], "upper")
            self.assertFalse(malt["formal_allowed"])

    def test_each_malter_boundary_blocks_all_formal_fail_closed(self) -> None:
        for cell_id, lr, side in (
            ("malter_eq17_lr007", 0.007, "lower"),
            ("malter_eq17_lr025", 0.025, "upper"),
        ):
            with self.subTest(side=side), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                batch = root / "batch"
                batch.mkdir()
                losses = default_losses()
                losses[cell_id] = 3.2000
                write_batch(batch, losses)
                write_matching_runner_selection(batch)
                result = run_analyzer(batch, root / "analysis")
                self.assertEqual(result.returncode, 0, result.stdout)
                selection = read_selection(root / "analysis")
                self.assertFalse(selection["formal_allowed"])
                self.assertEqual(selection["blocking_methods"], ["malter_eq17"])
                self.assertTrue(selection["selections"]["malt"]["formal_allowed"])
                malter = selection["selections"]["malter_eq17"]
                self.assertEqual(malter["raw_best_matrix_lr"], lr)
                self.assertEqual(malter["boundary_side"], side)

    def test_rejects_invalid_cell_even_when_aggregate_is_locally_valid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            batch = root / "batch"
            batch.mkdir()
            write_batch(batch, default_losses())
            manifest_path = batch / "pilot_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["summaries"][0]["evidence_valid"] = False
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            result = run_analyzer(batch, root / "analysis")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not accepted as valid local evidence", result.stdout)

    def test_rejects_incomplete_twelve_cell_grid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            batch = root / "batch"
            batch.mkdir()
            write_batch(batch, default_losses())
            manifest_path = batch / "pilot_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["summaries"] = [
                row
                for row in manifest["summaries"]
                if row["cell_id"] != "malt_lr0125"
            ]
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            result = run_analyzer(batch, root / "analysis")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected 12 pilot summaries", result.stdout)

    def test_rejects_v3_protocol_and_artifact_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            batch = root / "batch"
            batch.mkdir()
            write_batch(
                batch,
                default_losses(),
                protocol="malt_r1_extended_grid_pilot_v3",
            )
            result = run_analyzer(batch, root / "analysis")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pilot protocol mismatch", result.stdout)

    def test_strict_method_lr_seed_steps_and_acceptance_gates(self) -> None:
        base: dict[str, object] = {
            "cell_id": "malt_lr0064",
            "method": "malt",
            "matrix_lr": 0.0064,
            "formal_eligible": True,
            "controlled_seed": 2026,
            "total_steps": 1000,
            "total_tokens": 524_288_000,
            "evidence_valid": True,
            "final_val_loss": 3.5,
            "val_loss_step_1000": 3.5,
        }
        for field, value, message in (
            ("method", "malter_eq17", "method mismatch"),
            ("matrix_lr", 0.0065, "LR mismatch"),
            ("controlled_seed", 2025, "must use seed 2026"),
            ("total_steps", 999, "must run 1000 steps"),
            ("evidence_valid", False, "not accepted as valid local evidence"),
        ):
            with self.subTest(field=field):
                row = dict(base)
                row[field] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    pilot.normalize_summary(row, source="unit")

    def test_matching_runner_selection_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            batch = root / "batch"
            batch.mkdir()
            write_batch(batch, default_losses())
            write_matching_runner_selection(batch)
            result = run_analyzer(batch, root / "analysis")
            self.assertEqual(result.returncode, 0, result.stdout)
            manifest = json.loads(
                (root / "analysis/pilot_analysis_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["runner_selection_crosscheck"]["status"], "matched"
            )

    def test_rejects_runner_selection_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            batch = root / "batch"
            batch.mkdir()
            write_batch(batch, default_losses())
            write_matching_runner_selection(batch)
            runner_path = batch / "pilot_selection.json"
            wrong = json.loads(runner_path.read_text(encoding="utf-8"))
            wrong["selections"]["malt"]["selected_cell_id"] = "malt_lr0090"
            wrong["selections"]["malt"]["selected_matrix_lr"] = 0.009
            runner_path.write_text(json.dumps(wrong) + "\n", encoding="utf-8")
            result = run_analyzer(batch, root / "analysis")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("disagrees with independent analysis", result.stdout)

    def test_rejects_runner_that_claims_independent_certificate_role(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            batch = root / "batch"
            batch.mkdir()
            write_batch(batch, default_losses())
            write_matching_runner_selection(batch)
            runner_path = batch / "pilot_selection.json"
            payload = json.loads(runner_path.read_text(encoding="utf-8"))
            payload["certificate_role"] = "independent_pilot_analysis_selection"
            runner_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            result = run_analyzer(batch, root / "analysis")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("disagrees with independent analysis", result.stdout)


if __name__ == "__main__":
    unittest.main()
