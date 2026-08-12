from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
COMMANDS = ROOT / "commands"
EXPERIMENTS = ROOT / "experiments"

NUMBERED = re.compile(r"^\d{2}[a-z]?_.+")
SPECIAL_SCRIPT_DIRS = {"mdp_refresh_streaming": "mdp04_refresh_streaming"}
PLANNED = {
    "05_memory_efficient_baselines",
    "07_random_irregular_masks",
    "09_uniformity_vs_irregularity",
    "10_budget_aware_rules",
    "11_dynamic_release",
}
ANALYSIS_ONLY = {
    "32_mech05_frozen_selection_rule",
    "34_selective_primary_comparison",
    "38_unified_mechanism_synthesis",
}
SEALED_SOURCE = {
    "43_newton_muon_record28_275m",
    "44_newton_muon_record17_455m",
    "46_mdp05_confirmatory_update_shock",
    "47_update_geometry_curvature",
    "47b_geo01b_update_geometry_discovery",
    "48_llama1b_10b_multibudget",
    "mdp04_refresh_streaming",
}
PARTIAL_SOURCE = {
    "22_r1_block_alpha",
    "24_r1_dense_full_alpha",
    "35_mech07_llama1b_family_contrast",
    "36_mech08_short_horizon_rollout",
    "37_mech09_downproj_refresh_mediation",
    "39_submission_efficiency_and_sensitivity",
    "40_llama_block_partition_invariance_audit",
    "41_r1_kstate_module_factorial",
    "42_llama1b_isolated_efficiency",
    "45_r1_mousse_strong_baseline",
    "29_r1_depth_kmode",
    "49_r1_malt_strong_baseline",
}

# Only explicit positional-mode launchers are exposed as native resume/verify.
# Older launchers use experiment-specific environment variables; their recovery
# instructions remain visible in the copied command, but are not guessed here.
NATIVE_MODES = {
    "29_r1_depth_kmode": {},
    "mdp04_refresh_streaming": {
        "native-verify": {
            "args": ["archive-verify"],
            "env": {"MDP04_RUN_DIR": "{run_dir}"},
        },
    },
    "46_mdp05_confirmatory_update_shock": {
        "resume": {
            "args": ["resume"],
            "env": {"MDP05_RUN_DIR": "{run_dir}"},
            "required_env": ["MDP05_PILOT_CERTIFICATE"],
            "required_env_files": ["MDP05_PILOT_CERTIFICATE"],
        },
        "native-verify": {
            "args": ["verify"],
            "env": {"MDP05_RUN_DIR": "{run_dir}"},
        },
    },
    "47_update_geometry_curvature": {
        "resume": {"args": ["resume"], "env": {"RUN_DIR": "{run_dir}"}},
        "native-verify": {"args": ["verify"], "env": {"RUN_DIR": "{run_dir}"}},
    },
    "47b_geo01b_update_geometry_discovery": {
        "resume": {"args": ["resume"], "env": {"RUN_DIR": "{run_dir}"}},
        "native-verify": {"args": ["verify"], "env": {"RUN_DIR": "{run_dir}"}},
    },
    "48_llama1b_10b_multibudget": {
        "resume": {"args": ["resume"], "env": {"EX48_RUN_DIR": "{run_dir}"}},
        "native-verify": {
            "args": ["verify"],
            "env": {"EX48_RUN_DIR": "{run_dir}"},
        },
    },
    "49_r1_malt_strong_baseline": {
        "resume": {"args": ["all"], "env": {"EX49_RUN_DIR": "{run_dir}"}},
        "native-verify": {
            "args": ["verify"],
            "env": {"EX49_RUN_DIR": "{run_dir}"},
        },
    },
}

EXTRA_PYTHON_ENTRYPOINTS = {
    "29_r1_depth_kmode": ["analyze_r1_depth_kmode_formal.py"],
    "48_llama1b_10b_multibudget": ["audit_received_results.py"],
}

RESULT_OVERRIDES = {
    "mdp04_refresh_streaming": [
        "_shared/analysis/method_deepening_mdp04_refresh_replay"
    ],
    "46_mdp05_confirmatory_update_shock": [
        "_shared/analysis/method_deepening_mdp05_confirmatory_update_shock"
    ],
    "47_update_geometry_curvature": [
        "_shared/analysis/method_deepening_geo01_update_curvature"
    ],
    "47b_geo01b_update_geometry_discovery": [
        "47b_geo01b_update_geometry_discovery",
        "_shared/analysis/method_deepening_geo01b_update_geometry_discovery",
    ],
    "48_llama1b_10b_multibudget": ["48_llama1b_10b_multibudget"],
}

EXPERIMENT_README_SECTIONS = {
    "48_llama1b_10b_multibudget": [
        "## Accepted execution geometry",
        "",
        "The replacement formal protocol uses one physical host with exactly four H100",
        '80GB GPUs (`EX48_GPUS="0 1 2 3"` by default). The interrupted two-GPU attempt',
        "was deleted, is not resumable, and is not accepted as evidence. This amendment",
        "changes scheduling and wall-clock only; methods, seeds, token budgets, data",
        "order, and analysis rules remain frozen.",
        "",
        "## Independent acceptance audit",
        "",
        "After a formal run has been copied locally, independently rebuild the endpoint",
        "and paired tables and bind the persisted remote full-checkpoint re-hash receipt:",
        "",
        "```bash",
        "python scripts/48_llama1b_10b_multibudget/audit_received_results.py \\",
        "  --run-dir /path/to/results/48_llama1b_10b_multibudget/RUN_ID \\",
        "  --received-dir /path/to/received-files \\",
        "  --output-dir /path/to/final-acceptance",
        "```",
        "",
        "The receipt is a semantic certificate for the frozen verifier's full re-hash",
        "of all 36 retained endpoints. It is not a forensic execution log: the compact",
        "JSON does not itself record the command, process exit code, host, or Python",
        "environment, so the audit binds its semantics to the hash-frozen verifier.",
        "",
    ],
}

# These are user-supplied protocol inputs, not values the release packager can
# honestly invent.  ``reproduce.py`` permits plan inspection without them but
# refuses execution until every all_of/one_of group is satisfied through
# receipt-bound ``--arg`` values.
REQUIRED_USER_ARGUMENTS: dict[tuple[str, str], list[dict[str, list[str]]]] = {
    ("17_llama_swiglu_validation", "run_llama_swiglu_validation"): [
        {"all_of": ["--official-repo", "--python-exe"]},
    ],
    ("18_r1_performance", "run_r1_performance"): [
        {"all_of": ["--official-repo", "--python-exe"]},
    ],
    ("19_r1_extended_baselines", "run_r1_extended_baselines"): [
        {
            "one_of": [
                "--preflight",
                "--numerical-smoke",
                "--formal-smoke",
                "--pilot",
                "--formal",
            ]
        },
    ],
    ("20_llama_swiglu_1b", "run_llama_swiglu_1b"): [
        {"all_of": ["--stage", "--official-repo", "--python-exe"]},
    ],
    ("20_llama_swiglu_1b", "run_llama_swiglu_1b_capacity"): [
        {"all_of": ["--official-repo", "--python-exe"]},
    ],
    ("20_llama_swiglu_1b", "run_llama_swiglu_1b_capacity_cell"): [
        {"all_of": ["--stage", "--official-repo", "--python-exe"]},
    ],
    ("20_llama_swiglu_1b", "run_llama_swiglu_1b_capacity_exact"): [
        {"all_of": ["--fine-manifest", "--official-repo", "--python-exe"]},
    ],
    ("20_llama_swiglu_1b", "run_llama_swiglu_1b_capacity_fine"): [
        {"all_of": ["--official-repo", "--python-exe"]},
    ],
    ("20_llama_swiglu_1b", "run_llama_swiglu_1b_capacity_fine_cell"): [
        {
            "all_of": [
                "--stage",
                "--official-repo",
                "--python-exe",
                "--capacity-accumulation-steps",
                "--device-batch-size",
            ]
        },
    ],
    ("23_llama_swiglu_extended_baselines", "run_llama_swiglu_extended"): [
        {"all_of": ["--official-repo", "--python-exe"]},
        {
            "one_of": [
                "--dry-run",
                "--preflight",
                "--numerical-smoke",
                "--pilot",
                "--formal-smoke",
                "--formal",
            ]
        },
    ],
    (
        "23_llama_swiglu_extended_baselines",
        "run_llama_swiglu_extended_capacity",
    ): [
        {
            "all_of": [
                "--official-repo",
                "--python-exe",
                "--pilot-manifest",
            ]
        },
    ],
    ("25_owt_depth_kmode", "run_owt_depth_kmode"): [
        {"one_of": ["--dry-run", "--numerical-smoke", "--formal"]},
    ],
    ("28_wikitext_depth_kmode", "run_wikitext_depth_kmode"): [
        {"one_of": ["--dry-run", "--numerical-smoke", "--formal"]},
    ],
    ("29_r1_depth_kmode", "run_three_seed_batch"): [
        {"all_of": ["--official-repo", "--python-exe", "--results-dir"]},
    ],
    ("29_r1_depth_kmode", "run_r1_depth_kmode"): [
        {"all_of": ["--smoke-manifest"]},
    ],
    ("29_r1_depth_kmode", "analyze_r1_depth_kmode_formal"): [
        {
            "all_of": [
                "--bundle-root",
                "--batch-id",
                "--source-zip",
                "--wandb-inputs",
                "--reference-results-root",
                "--output-dir",
            ]
        },
    ],
}


def title_for(experiment_id: str) -> str:
    if experiment_id == "49_r1_malt_strong_baseline":
        return "49 R1 MALT Strong Baseline"
    return experiment_id.replace("_", " ").replace("mdp04", "MDP-04").title()


def script_experiments() -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for path in sorted(SCRIPTS.iterdir()):
        if not path.is_dir():
            continue
        if NUMBERED.fullmatch(path.name):
            rows.append((path.name, path))
        elif path.name in SPECIAL_SCRIPT_DIRS:
            rows.append((SPECIAL_SCRIPT_DIRS[path.name], path))
    return rows


def python_entrypoints(script_dir: Path) -> list[Path]:
    runners = sorted(
        path
        for path in script_dir.glob("run_*.py")
        if not path.name.startswith("test_")
    )
    runners.extend(sorted(script_dir.glob("run_*.sh")))
    if runners:
        return runners
    analysis = []
    for prefix in ("analyze_", "validate_", "audit_", "build_"):
        analysis.extend(sorted(script_dir.glob(f"{prefix}*.py")))
    return sorted(set(analysis))


def command_entrypoints(experiment_id: str) -> list[Path]:
    directory = COMMANDS / experiment_id
    return sorted(directory.glob("*.sh")) if directory.is_dir() else []


def make_entrypoint(path: Path, *, experiment_id: str, kind: str) -> dict:
    relative = path.relative_to(ROOT).as_posix()
    modes = {"reproduce": []}
    if kind == "shell" and path.stem == "reproduce_full":
        modes = {"reproduce": []}
    elif kind == "shell" and path.stem == "reproduce_archived":
        modes = {}
    elif kind == "shell" and experiment_id in NATIVE_MODES:
        modes = NATIVE_MODES[experiment_id]
    result = {
        "path": relative,
        "kind": kind,
        "args": [],
        "native_modes": modes,
    }
    requirements = REQUIRED_USER_ARGUMENTS.get((experiment_id, path.stem), [])
    if requirements:
        result["required_user_arguments"] = requirements
    return result


def build_metadata(experiment_id: str, script_dir: Path) -> dict:
    commands = command_entrypoints(experiment_id)
    python = python_entrypoints(script_dir)
    python.extend(
        script_dir / name
        for name in EXTRA_PYTHON_ENTRYPOINTS.get(experiment_id, [])
    )
    python = sorted(set(python))
    entrypoints: dict[str, dict] = {}
    for path in commands:
        entrypoints[f"command:{path.stem}"] = make_entrypoint(
            path, experiment_id=experiment_id, kind="shell"
        )
    for path in python:
        key = f"script:{path.stem}"
        row = make_entrypoint(
            path,
            experiment_id=experiment_id,
            kind="shell" if path.suffix == ".sh" else "python",
        )
        # When an authored shell launcher exists it is the reproduction
        # authority. Python entrypoints remain inspectable and explicitly
        # selectable, but do not compete for the default reproduce action.
        if commands:
            row["native_modes"] = {}
        entrypoints[key] = row

    if experiment_id in PLANNED:
        status = "planned_not_implemented"
    elif experiment_id in ANALYSIS_ONLY:
        status = "analysis_only"
    elif entrypoints:
        status = "implemented"
    else:
        status = "missing_entrypoint"

    if experiment_id in SEALED_SOURCE:
        source_freeze = "sealed_source_snapshot"
    elif experiment_id in PARTIAL_SOURCE:
        source_freeze = "partial_or_run_specific_snapshot"
    else:
        source_freeze = "legacy_command_or_live_source"

    result_roots = RESULT_OVERRIDES.get(experiment_id, [experiment_id])
    reproduce_providers = sorted(
        name
        for name, row in entrypoints.items()
        if "reproduce" in row.get("native_modes", {})
    )
    fresh_rerun = (
        bool(reproduce_providers)
        and status != "planned_not_implemented"
        and experiment_id != "mdp04_refresh_streaming"
    )
    one_click_rerun = (
        fresh_rerun
        and len(reproduce_providers) == 1
        and not entrypoints[reproduce_providers[0]].get("required_user_arguments")
    )
    metadata = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "title": title_for(experiment_id),
        "status": status,
        "code_directory": script_dir.relative_to(ROOT).as_posix(),
        "entrypoints": entrypoints,
        "legacy_result_roots": result_roots,
        "reproducibility": {
            "fresh_rerun": fresh_rerun,
            "one_click_rerun": one_click_rerun,
            "generic_artifact_verify": status != "planned_not_implemented",
            "source_freeze": source_freeze,
            "native_resume": any(
                "resume" in row.get("native_modes", {})
                for row in entrypoints.values()
            ),
            "native_verify": any(
                "native-verify" in row.get("native_modes", {})
                for row in entrypoints.values()
            ),
        },
    }
    if len(reproduce_providers) == 1:
        metadata["default_entrypoint"] = reproduce_providers[0]
    return metadata


BOOLEAN_USER_ARGUMENTS = {
    "--dry-run",
    "--preflight",
    "--numerical-smoke",
    "--formal-smoke",
    "--pilot",
    "--formal",
}


def example_user_arguments(entrypoint: dict) -> list[str]:
    selected: list[str] = []
    for group in entrypoint.get("required_user_arguments", []):
        kind, options = next(iter(group.items()))
        chosen = list(options)
        if kind == "one_of":
            chosen = ["--formal" if "--formal" in options else options[0]]
        for option in chosen:
            selected.append(f"--arg={option}")
            if option not in BOOLEAN_USER_ARGUMENTS:
                if option == "--stage":
                    value = "formal"
                elif option == "--capacity-accumulation-steps":
                    value = "8"
                elif option == "--device-batch-size":
                    value = "16"
                else:
                    value = "/path/to/value"
                selected.append(f"--arg={value}")
    return selected


def reproduce_command(metadata: dict, provider: str | None) -> str:
    experiment_id = metadata["experiment_id"]
    command = f"python reproducibility/reproduce.py reproduce {experiment_id}"
    if provider is not None:
        command += f" --entrypoint {provider}"
        entrypoint = metadata["entrypoints"][provider]
    else:
        entrypoint = metadata["entrypoints"][metadata["default_entrypoint"]]
    forwarded = example_user_arguments(entrypoint)
    if forwarded:
        command += " " + " ".join(forwarded)
    return command


def experiment_readme(metadata: dict) -> str:
    experiment_id = metadata["experiment_id"]
    reproduction = metadata["reproducibility"]
    entrypoints = metadata["entrypoints"]
    reproduce_providers = sorted(
        name
        for name, row in entrypoints.items()
        if "reproduce" in row.get("native_modes", {})
    )
    legacy_root = metadata.get("legacy_result_roots", [experiment_id])[0]
    example_run = f"/path/to/results/{legacy_root}/RUN_ID"
    lines = [
        f"# {metadata['title']}",
        "",
        f"- Status: `{metadata['status']}`",
        f"- Code: `../../{metadata['code_directory']}`",
        f"- Source-freeze tier: `{reproduction['source_freeze']}`",
        f"- Generic artifact verification: `{str(reproduction['generic_artifact_verify']).lower()}`",
        f"- Fresh rerun: `{str(reproduction['fresh_rerun']).lower()}`",
        f"- One-click rerun: `{str(reproduction['one_click_rerun']).lower()}`",
        f"- Native resume: `{str(reproduction['native_resume']).lower()}`",
        f"- Native verification: `{str(reproduction['native_verify']).lower()}`",
        "",
    ]
    lines.extend(EXPERIMENT_README_SECTIONS.get(experiment_id, []))
    if metadata["status"] == "planned_not_implemented":
        lines.extend(
            [
                "This identifier was reserved during planning, but no experiment",
                "implementation or accepted result exists. It is intentionally not",
                "presented as reproducible evidence.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Inspect the frozen public entrypoints:",
                "",
                "```bash",
                f"python reproducibility/reproduce.py inspect {experiment_id}",
                "```",
                "",
            ]
        )
        if reproduction["fresh_rerun"]:
            lines.extend(
                [
                    "Build a fresh reproduction plan (read-only by default).",
                    "Machine-specific settings must use repeated `--env KEY=VALUE`",
                    "arguments, while runner-specific options use repeated `--arg`",
                    "values, so both are covered by the plan receipt. A plan may be",
                    "inspected with missing required arguments, but execution is refused",
                    "until every declared argument group is satisfied:",
                    "",
                    "```bash",
                ]
            )
            default = metadata.get("default_entrypoint")
            if isinstance(default, str):
                lines.append(reproduce_command(metadata, None))
            else:
                for provider in reproduce_providers:
                    lines.append(reproduce_command(metadata, provider))
            lines.extend(["```", ""])
        else:
            lines.extend(
                [
                    "A fresh rerun is not declared for this archived experiment.",
                    "Use the native archival validator below when available.",
                    "",
                ]
            )

        lines.extend(
            [
                "Verify an existing result without training. The run path must",
                "match a declared legacy result root or carry a matching sealed",
                "source-snapshot lineage:",
                "",
                "```bash",
                "python reproducibility/reproduce.py verify \\",
                f"  {experiment_id} --results-root /path/to/results \\",
                f"  --run-dir {example_run}",
                "```",
                "",
            ]
        )
        if reproduction["native_resume"]:
            required = sorted(
                {
                    key
                    for row in entrypoints.values()
                    for spec_name, spec in row.get("native_modes", {}).items()
                    if spec_name == "resume" and isinstance(spec, dict)
                    for key in spec.get("required_env", [])
                }
            )
            lines.extend(
                [
                    "Resume an interrupted native run:",
                    "",
                    "```bash",
                    "python reproducibility/reproduce.py resume \\",
                    f"  {experiment_id} --results-root /path/to/results \\",
                    f"  --run-dir {example_run}" + (" \\" if required else ""),
                ]
            )
            for index, key in enumerate(required):
                suffix = " \\" if index + 1 < len(required) else ""
                lines.append(f"  --env {key}=/path/to/value{suffix}")
            lines.extend(["```", ""])
        if reproduction["native_verify"]:
            lines.extend(
                [
                    "Run the experiment-specific native validator:",
                    "",
                    "```bash",
                    "python reproducibility/reproduce.py native-verify \\",
                    f"  {experiment_id} --results-root /path/to/results \\",
                    f"  --run-dir {example_run}",
                    "```",
                    "",
                ]
            )
    lines.extend(["## Entrypoints", ""])
    if not entrypoints:
        lines.append("None.")
    else:
        for name, row in sorted(entrypoints.items()):
            modes = ", ".join(sorted(row.get("native_modes", {}))) or "explicit selection only"
            if row.get("required_user_arguments"):
                modes += "; requires receipt-bound --arg values"
            lines.append(f"- `{name}` → `{row['path']}` ({modes})")
    lines.extend(
        [
            "",
            "Historical results are not stored in this source repository. See",
            "`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.",
            "",
        ]
    )
    return "\n".join(lines)


def command_readme(metadata: dict) -> str:
    experiment_id = metadata["experiment_id"]
    return "\n".join(
        [
            f"# Commands for {metadata['title']}",
            "",
            "Launchers in this directory are parameterized public copies. Historical",
            "originals remain private; their hashes are recorded in",
            "`../../provenance/legacy_command_inventory.json`.",
            "Use the guarded dispatcher from the repository root:",
            "",
            "```bash",
            f"python reproducibility/reproduce.py reproduce {experiment_id}",
            "```",
            "",
            "The first call only prints a SHA-256-bound plan. Execution requires a",
            "second call with `--execute --receipt <plan_sha256>`.",
            "",
        ]
    )


def experiment_index(catalog: list[dict]) -> str:
    lines = [
        "# Experiment index",
        "",
        "This table is generated from the per-experiment metadata. `verify` means",
        "the common read-only artifact verifier; native scientific validation is",
        "reported separately.",
        "",
        "| Experiment | Status | Source freeze | Fresh rerun | One-click | Resume | Native verify |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in catalog:
        repro = row["reproducibility"]
        lines.append(
            "| [{id}](../experiments/{id}/) | `{status}` | `{freeze}` | {fresh} | {one_click} | {resume} | {verify} |".format(
                id=row["experiment_id"],
                status=row["status"],
                freeze=repro["source_freeze"],
                fresh="yes" if repro["fresh_rerun"] else "no",
                one_click="yes" if repro["one_click_rerun"] else "no",
                resume="yes" if repro["native_resume"] else "no",
                verify="yes" if repro["native_verify"] else "no",
            )
        )
    lines.extend(["", "Generated by `reproducibility/build_catalog.py`.", ""])
    return "\n".join(lines)


def main() -> int:
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    catalog = []
    for experiment_id, script_dir in script_experiments():
        metadata = build_metadata(experiment_id, script_dir)
        output_dir = EXPERIMENTS / experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "metadata.json"
        output.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "README.md").write_text(
            experiment_readme(metadata), encoding="utf-8", newline="\n"
        )
        catalog.append(metadata)
    (EXPERIMENTS / "catalog.json").write_text(
        json.dumps(
            {"schema_version": 1, "experiments": catalog},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "EXPERIMENT_INDEX.md").write_text(
        experiment_index(catalog), encoding="utf-8", newline="\n"
    )
    print(f"wrote {len(catalog)} experiment metadata records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
