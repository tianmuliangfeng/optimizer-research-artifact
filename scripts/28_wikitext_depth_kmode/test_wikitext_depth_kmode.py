from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_wikitext_depth_kmode as wiki


class WikiTextDepthKModeContractTest(unittest.TestCase):
    def parse(self, *extra: str):
        argv = ["run_wikitext_depth_kmode.py", *extra]
        with mock.patch.object(sys, "argv", argv):
            return wiki.parse_args()

    def test_formal_defaults_generate_exact_36_run_contract(self) -> None:
        args = self.parse("--formal", "--no-write-commands")
        self.assertEqual(args.dataset, wiki.DATASET)
        self.assertEqual(args.seeds, [2024, 2025, 2026])
        self.assertEqual(args.run_prefix, wiki.RUN_PREFIX)
        self.assertEqual(args.wandb_project, wiki.WANDB_PROJECT)

        commands = wiki.base.build_commands(args)
        wiki.base.validate_commands(args, commands)
        self.assertEqual(len(commands), 36)

        names = [wiki.base.option_value(cmd, "wandb_run_name") for cmd in commands]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(wiki.RUN_PREFIX in name for name in names))
        self.assertTrue(
            all(
                wiki.base.option_value(cmd, "dataset") == wiki.DATASET
                for cmd in commands
            )
        )
        for seed in (2024, 2025, 2026):
            self.assertEqual(sum(name.endswith(f"seed{seed}") for name in names), 12)

    def test_smoke_is_six_paths_and_crosses_first_refresh(self) -> None:
        args = self.parse("--numerical-smoke", "--no-write-commands")
        commands = wiki.base.build_commands(args)
        wiki.base.validate_commands(args, commands)
        self.assertEqual(len(commands), 6)
        self.assertEqual(args.max_iters, 34)
        names = [wiki.base.option_value(cmd, "wandb_run_name") for cmd in commands]
        self.assertTrue(all(name.endswith("seed2026") for name in names))
        self.assertTrue(any("_center_none_" in name for name in names))
        self.assertTrue(any("_all_diag_" in name for name in names))
        self.assertTrue(any("_anchor_full_" in name for name in names))
        self.assertTrue(any("_anchor_muon_" in name for name in names))

    def test_depth_rules_and_reference_path_are_frozen(self) -> None:
        args = self.parse("--formal", "--seeds", "2026", "--no-write-commands")
        commands = wiki.base.build_commands(args)
        early_diag = next(
            cmd
            for cmd in commands
            if "_formal_early_diag_seed2026"
            in wiki.base.option_value(cmd, "wandb_run_name")
        )
        self.assertEqual(
            wiki.base.option_value(early_diag, "cproj_k_layers"),
            repr("0,1,2,3,4,5,6,7"),
        )
        self.assertEqual(
            wiki.base.option_value(early_diag, "cproj_k_reference_mode"),
            "full",
        )
        self.assertEqual(
            wiki.base.option_value(early_diag, "cproj_k_mode"),
            "diag",
        )

    def test_owt_base_defaults_remain_unchanged(self) -> None:
        parser = wiki.base.build_parser()
        args = parser.parse_args(["--dry-run", "--no-write-commands"])
        self.assertEqual(args.dataset, "openwebtext_gpt2_50m")
        self.assertEqual(args.run_prefix, "mainconf_owt_12L_depth_kmode")

    def test_pinned_binary_hashes_are_normalized(self) -> None:
        args = self.parse("--dry-run", "--no-write-commands")
        self.assertEqual(
            args.expected_train_sha256,
            wiki.EXPECTED_TRAIN_SHA256,
        )
        self.assertEqual(
            args.expected_val_sha256,
            wiki.EXPECTED_VAL_SHA256,
        )
        self.assertEqual(len(args.expected_train_sha256), 64)
        self.assertEqual(len(args.expected_val_sha256), 64)

    def test_parser_is_compatible_with_original_family25_runner(self) -> None:
        with mock.patch.object(wiki.base, "build_parser", None):
            args = self.parse("--formal", "--no-write-commands")
        self.assertEqual(args.dataset, wiki.DATASET)
        self.assertEqual(args.seeds, [2024, 2025, 2026])
        commands = wiki.base.build_commands(args)
        wiki.base.validate_commands(args, commands)
        self.assertEqual(len(commands), 36)

    def test_execution_is_compatible_with_original_family25_runner(self) -> None:
        argv = [
            "run_wikitext_depth_kmode.py",
            "--dry-run",
            "--no-write-commands",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(wiki.base, "build_parser", None),
            mock.patch.object(wiki.base, "run_experiment", None),
            mock.patch.object(wiki.base, "validate_args") as validate_args,
            mock.patch.object(
                wiki.base, "validate_source_support"
            ) as validate_source_support,
            mock.patch.object(wiki.base, "ensure_data", return_value=True),
            mock.patch.object(wiki, "validate_wikitext_dataset") as data_check,
            mock.patch.object(wiki.base, "print_plan"),
            mock.patch.object(wiki.base, "build_commands", return_value=[]),
            mock.patch.object(wiki.base, "validate_commands"),
            mock.patch.object(wiki.base, "write_command_record"),
        ):
            wiki.main()
        validate_args.assert_called_once()
        validate_source_support.assert_called_once()
        data_check.assert_called_once()


if __name__ == "__main__":
    unittest.main()
