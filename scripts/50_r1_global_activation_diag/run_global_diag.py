#!/usr/bin/env python3
"""Run one Experiment-50 global-diag seed using the audited R1 controller."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_VERSION = "2026-08-14.1"
SCRIPT_DIR = Path(__file__).resolve().parent
R1_DIR = SCRIPT_DIR.parent / "15_official_newton_muon_r1"
sys.path.insert(0, str(R1_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import run_official_newton_muon_r1 as R1
from global_diag_source_builder import (
    build_global_diag_sources,
    expected_memory_contract,
)


FAMILY = "50_r1_global_activation_diag"
PROJECT = "anonymous-optimizer-artifact-ex50"
RUN_PREFIX = "mainconf_r1_global_diag"
SMOKE_PROTOCOL = "r1_global_activation_diag_exact_shape_pilot"
FORMAL_PROTOCOL = "r1_global_activation_diag_formal"
CONTRACT_PATH = SCRIPT_DIR / "global_diag_contract.json"
CONTRACT_SHA256 = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

METHODS = {
    "global_diag": R1.MethodSpec(
        name="global_diag",
        base_script="train_gpt_newton_muon_1.py",
        cproj_k_mode="diag",
        base_learning_rate=0.0040,
        matrix_learning_rate=0.00040,
        role="confirmatory_all_eligible_activation_diagonal",
    )
}

GLOBAL_METADATA_RE = re.compile(
    r"^R1_GLOBAL_DIAG_METADATA route=(?P<route>\w+) "
    r"dense_workspace=(?P<workspace>\d+)$"
)
GLOBAL_ROUTE_RE = re.compile(
    r"^R1_GLOBAL_DIAG_ROUTE input_diag_params=(?P<input>\d+) "
    r"proj_diag_params=(?P<proj>\d+) dense_refresh_blocks=(?P<dense>\d+)$"
)

ORIGINAL_PARSE_ARGS = R1.parse_args
ORIGINAL_CONTROLLED_ENV = R1.controlled_env
ORIGINAL_PARSE_METRICS = R1.parse_metrics
ORIGINAL_WRITE_JSON = R1.write_json
ORIGINAL_INITIALIZATION_AUDIT = R1.initialization_audit


def experiment_family(_args) -> str:
    return FAMILY


def experiment_protocol(args, *, smoke: bool | None = None) -> str:
    is_smoke = args.numerical_smoke if smoke is None else smoke
    return SMOKE_PROTOCOL if is_smoke else FORMAL_PROTOCOL


def evidence_eligibility(_args) -> dict[str, object]:
    return {
        "quality_usable": True,
        "memory_usable": True,
        "timing_usable": False,
        "reason": (
            "three-seed quality/state control with up to two physical GPUs "
            "training concurrently; no isolated timing claim is allowed"
        ),
    }


def build_all_sources(repo: Path, *, lr_cross: bool = False):
    if lr_cross:
        raise RuntimeError("Experiment 50 forbids LR-cross mode")
    return build_global_diag_sources(repo)


def controlled_env(args, spec, data_dir: Path, **kwargs) -> dict[str, str]:
    env = ORIGINAL_CONTROLLED_ENV(args, spec, data_dir, **kwargs)
    env["R1_GLOBAL_DIAG"] = "1"
    env["R1_GLOBAL_DIAG_CONTRACT_SHA256"] = CONTRACT_SHA256
    return env


def initialization_audit(
    args, repo: Path, data_dir: Path, built
) -> dict[str, object]:
    """Bind every global-diag seed to the unmodified selective-diag init."""
    global_audit = ORIGINAL_INITIALIZATION_AUDIT(args, repo, data_dir, built)
    reference_spec = R1.MethodSpec(
        name="diag",
        base_script="train_gpt_newton_muon_1.py",
        cproj_k_mode="diag",
        base_learning_rate=0.0040,
        matrix_learning_rate=0.00040,
        role="read_only_initialization_reference",
    )
    reference = R1.build_source(repo, "diag")
    with tempfile.TemporaryDirectory(prefix="ex50_reference_init_") as temp:
        workspace = Path(temp) / "diag"
        script = R1.materialize_source(workspace, repo, reference)
        completed = subprocess.run(
            [args.python_exe, script.name],
            cwd=workspace,
            env=ORIGINAL_CONTROLLED_ENV(
                args, reference_spec, data_dir, init_only=True
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=900,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "Experiment-50 selective-diag initialization reference failed:\n"
            + completed.stdout[-8000:]
        )
    reference_metadata = R1.parse_metadata(completed.stdout)
    if reference_metadata["init_sha256"] != global_audit["init_sha256"]:
        raise RuntimeError(
            "global-diag changed model initialization: "
            f"global={global_audit['init_sha256']} "
            f"selective_diag={reference_metadata['init_sha256']}"
        )
    return {
        **global_audit,
        "selective_diag_reference_init_sha256": reference_metadata["init_sha256"],
        "selective_diag_reference_derived_sha256": reference.derived_sha256,
        "global_diag_matches_selective_diag_initialization": True,
    }


def parse_args():
    args = ORIGINAL_PARSE_ARGS()
    if args.lr_cross or args.host_bridge:
        raise RuntimeError("Experiment 50 forbids LR-cross and host-bridge modes")
    if args.methods is None or args.methods != ["global_diag"]:
        raise RuntimeError(
            "Experiment 50 must request exactly --methods global_diag; "
            "all comparison methods are frozen read-only controls"
        )
    return args


def parse_metrics(*args, **kwargs):
    rows, summary = ORIGINAL_PARSE_METRICS(*args, **kwargs)
    stdout_path: Path = args[1]
    matches = [
        match
        for line in stdout_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if (match := GLOBAL_METADATA_RE.match(line)) is not None
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one R1_GLOBAL_DIAG_METADATA line, observed {len(matches)}"
        )
    observed = matches[0].groupdict()
    if observed != {
        "route": "all_eligible_activation_diagonal",
        "workspace": "0",
    }:
        raise RuntimeError(f"global-diag runtime metadata mismatch: {observed}")
    route_matches = [
        match
        for line in stdout_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if (match := GLOBAL_ROUTE_RE.match(line)) is not None
    ]
    if len(route_matches) != 1:
        raise RuntimeError(
            f"expected one R1_GLOBAL_DIAG_ROUTE line, observed {len(route_matches)}"
        )
    route = route_matches[0].groupdict()
    if route != {"input": "36", "proj": "12", "dense": "0"}:
        raise RuntimeError(f"global-diag apply route mismatch: {route}")

    expected = expected_memory_contract()
    observed_memory = {
        key: int(summary[key])
        for key in (
            "k_cov_bytes",
            "k_inv_bytes",
            "k_state_bytes",
            "activation_stat_bytes",
            "precond_workspace_bytes",
        )
    }
    if observed_memory != expected:
        raise RuntimeError(
            f"global-diag memory route mismatch: expected={expected}, "
            f"observed={observed_memory}"
        )
    for row in rows:
        row["global_diag_route"] = "all_eligible_activation_diagonal"
    summary.update(
        {
            "global_diag_route": "all_eligible_activation_diagonal",
            "global_diag_contract_sha256": CONTRACT_SHA256,
            "eligible_matrix_parameters": 48,
            "input_diagonal_factors": 84,
            "dense_cholesky_reachable": False,
            "script_version": SCRIPT_VERSION,
        }
    )
    return rows, summary


def enriched_json(path: Path, payload: dict[str, object]) -> None:
    enriched = copy.deepcopy(payload)
    enriched["global_diag"] = {
        "script_version": SCRIPT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "route": "all_eligible_activation_diagonal",
        "expected_memory": expected_memory_contract(),
        "timing_usable": False,
        "wandb_required_for_scientific_validity": False,
    }
    controls = enriched.get("environment_controls")
    if isinstance(controls, dict):
        controls["R1_GLOBAL_DIAG"] = "1"
        controls["R1_GLOBAL_DIAG_CONTRACT_SHA256"] = CONTRACT_SHA256
    ORIGINAL_WRITE_JSON(path, enriched)


def install_overrides() -> None:
    R1.FAMILY = FAMILY
    R1.DEFAULT_PROJECT = PROJECT
    R1.DEFAULT_RUN_PREFIX = RUN_PREFIX
    R1.R1_SMOKE_PROTOCOL = SMOKE_PROTOCOL
    R1.R1_FORMAL_PROTOCOL = FORMAL_PROTOCOL
    R1.METHODS = METHODS
    R1.parse_args = parse_args
    R1.experiment_family = experiment_family
    R1.experiment_protocol = experiment_protocol
    R1.evidence_eligibility = evidence_eligibility
    R1.build_all_sources = build_all_sources
    R1.controlled_env = controlled_env
    R1.initialization_audit = initialization_audit
    R1.parse_metrics = parse_metrics
    R1.write_json = enriched_json


def main() -> None:
    install_overrides()
    os.environ["R1_GLOBAL_DIAG"] = "1"
    R1.main()


if __name__ == "__main__":
    main()
