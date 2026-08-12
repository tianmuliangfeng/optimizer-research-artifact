#!/usr/bin/env python3
"""Run one seed of the two missing R1 module-factorial cells.

This controller reuses the audited experiment-15 execution/metric/W&B machinery
but replaces the source builder, family metadata, and c_fc environment contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path


SCRIPT_VERSION = "2026-07-29.4"
SCRIPT_DIR = Path(__file__).resolve().parent
R1_DIR = SCRIPT_DIR.parent / "15_official_newton_muon_r1"
sys.path.insert(0, str(R1_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import run_official_newton_muon_r1 as R1
from factorial_source_builder import build_factorial_sources


FAMILY = "41_r1_kstate_module_factorial"
PROJECT = "Selective-Newton-Muon-MainConf-R1-KState-Module-Factorial-20260729"
RUN_PREFIX = "mainconf_r1_kstate_factorial_cfc_none"
SMOKE_PROTOCOL = "r1_kstate_module_factorial_exact_shape_smoke"
FORMAL_PROTOCOL = "r1_kstate_module_factorial_formal"
CONTRACT_PATH = SCRIPT_DIR / "factorial_contract.json"
CONTRACT_SHA256 = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

METHODS = {
    "block4": R1.MethodSpec(
        name="block4",
        base_script="train_gpt_newton_muon_1.py",
        cproj_k_mode="block4",
        base_learning_rate=0.0040,
        matrix_learning_rate=0.00040,
        role="factorial_cfc_none_cproj_block4",
    ),
    "none": R1.MethodSpec(
        name="none",
        base_script="train_gpt_newton_muon_1.py",
        cproj_k_mode="none",
        base_learning_rate=0.0040,
        matrix_learning_rate=0.00040,
        role="factorial_cfc_none_cproj_none",
    ),
}
CELL_BY_METHOD = {"block4": "cproj_only", "none": "neither"}
FACTORIAL_METADATA_RE = re.compile(
    r"^R1_FACTORIAL_METADATA cfc_k_mode=(?P<cfc>\w+) "
    r"cproj_k_mode=(?P<cproj>\w+)$"
)


ORIGINAL_PARSE_ARGS = R1.parse_args
ORIGINAL_CONTROLLED_ENV = R1.controlled_env
ORIGINAL_PARSE_METRICS = R1.parse_metrics
ORIGINAL_WRITE_JSON = R1.write_json


def experiment_family(_args) -> str:
    return FAMILY


def experiment_protocol(_args, *, smoke: bool | None = None) -> str:
    is_smoke = _args.numerical_smoke if smoke is None else smoke
    return SMOKE_PROTOCOL if is_smoke else FORMAL_PROTOCOL


def evidence_eligibility(_args) -> dict[str, object]:
    return {
        "quality_usable": True,
        "memory_usable": True,
        "timing_usable": False,
        "reason": (
            "module-factorial quality/state run with two physical GPUs training "
            "concurrently; timing belongs to experiment 39"
        ),
    }


def build_all_sources(repo: Path, *, lr_cross: bool = False):
    if lr_cross:
        raise RuntimeError("module factorial cannot use --lr-cross")
    return build_factorial_sources(repo)


def controlled_env(args, spec, data_dir: Path, **kwargs) -> dict[str, str]:
    env = ORIGINAL_CONTROLLED_ENV(args, spec, data_dir, **kwargs)
    env["R1_CFC_K_MODE"] = "none"
    env["R1_FACTORIAL_CONTRACT_SHA256"] = CONTRACT_SHA256
    return env


def parse_args():
    args = ORIGINAL_PARSE_ARGS()
    if args.lr_cross or args.host_bridge:
        raise RuntimeError("experiment 41 forbids LR-cross and host-bridge modes")
    if args.methods is None or set(args.methods) != {"block4", "none"}:
        raise RuntimeError(
            "experiment 41 must request exactly --methods block4 none; "
            "existing cells are read-only reused"
        )
    if not args.numerical_smoke and not args.preflight and not args.dry_run:
        if args.wandb_mode != "online":
            raise RuntimeError("formal experiment 41 requires --wandb-mode online")
    return args


def parse_metrics(*args, **kwargs):
    rows, summary = ORIGINAL_PARSE_METRICS(*args, **kwargs)
    stdout_path: Path = args[1]
    spec = args[2]
    matches = [
        match
        for line in stdout_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if (match := FACTORIAL_METADATA_RE.match(line)) is not None
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one R1_FACTORIAL_METADATA line, observed {len(matches)}"
        )
    observed = matches[0].groupdict()
    if observed != {"cfc": "none", "cproj": spec.cproj_k_mode}:
        raise RuntimeError(f"factorial runtime metadata mismatch: {observed}")
    cell = CELL_BY_METHOD[spec.name]
    for row in rows:
        row["cfc_k_mode"] = "none"
        row["factorial_cell"] = cell
    summary.update(
        {
            "cfc_k_mode": "none",
            "factorial_cell": cell,
            "factorial_contract_sha256": CONTRACT_SHA256,
            "script_version": SCRIPT_VERSION,
        }
    )
    return rows, summary


def enriched_json(path: Path, payload: dict[str, object]) -> None:
    enriched = copy.deepcopy(payload)
    enriched["module_factorial"] = {
        "script_version": SCRIPT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "cfc_k_mode": "none",
        "new_training_cells": CONTRACT["new_training_cells"],
        "reused_cells": CONTRACT["reused_cells"],
        "timing_usable": False,
    }
    method = enriched.get("method")
    if isinstance(method, str) and method in CELL_BY_METHOD:
        enriched["cfc_k_mode"] = "none"
        enriched["factorial_cell"] = CELL_BY_METHOD[method]
    source = enriched.get("source")
    if isinstance(source, dict):
        source["cfc_k_mode"] = "none"
        if isinstance(method, str) and method in CELL_BY_METHOD:
            source["factorial_cell"] = CELL_BY_METHOD[method]
    controls = enriched.get("environment_controls")
    if isinstance(controls, dict):
        controls["R1_CFC_K_MODE"] = "none"
        controls["R1_FACTORIAL_CONTRACT_SHA256"] = CONTRACT_SHA256
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
    R1.parse_metrics = parse_metrics
    R1.write_json = enriched_json


def main() -> None:
    install_overrides()
    os.environ["R1_CFC_K_MODE"] = "none"
    R1.main()


if __name__ == "__main__":
    main()
