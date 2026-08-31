#!/usr/bin/env python3
"""Run one Experiment-53 matched-diagonal unit through the audited R1 runner."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path


SCRIPT_VERSION = "2026-08-17.2"
SCRIPT_DIR = Path(__file__).resolve().parent
R1_DIR = SCRIPT_DIR.parent / "15_official_newton_muon_r1"
sys.path.insert(0, str(R1_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import run_official_newton_muon_r1 as R1
from matched_diag_source_builder import (
    ARM_CONFIGS,
    build_matched_diag_sources,
    expected_memory_contract,
)


FAMILY = "53_r1_matched_diag_module_placement"
PROJECT = "Selective-Newton-Muon-MainConf-R1-Matched-Diag-Placement-20260817"
RUN_PREFIX = "mainconf_r1_matched_diag_placement"
PILOT_PROTOCOL = "r1_matched_diag_module_placement_engineering_pilot"
FORMAL_PROTOCOL = "r1_matched_diag_module_placement_formal"
PILOT_SEED = 2053
CONTRACT_PATH = SCRIPT_DIR / "matched_diag_contract.json"
CONTRACT_SHA256 = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


ROLES = {
    "all_none": "state_free_anchor",
    "c_fc_diag": "mlp_expansion_placement",
    "c_proj_diag": "mlp_contraction_placement",
    "c_fc_c_proj_diag": "factorial_joint_placement",
    "o_proj_diag": "attention_output_placement",
}

METHODS = {
    arm: R1.MethodSpec(
        name=arm,
        base_script="train_gpt_newton_muon_1.py",
        cproj_k_mode=config["c_proj"],
        base_learning_rate=0.0040,
        matrix_learning_rate=0.00040,
        role=ROLES[arm],
    )
    for arm, config in ARM_CONFIGS.items()
}

MATCHED_METADATA_RE = re.compile(
    r"^R1_MATCHED_DIAG_METADATA arm=(?P<arm>\w+) cfc=(?P<cfc>\w+) "
    r"cproj=(?P<cproj>\w+) oproj=(?P<oproj>\w+) qkv=(?P<qkv>\w+) "
    r"dense_workspace=(?P<workspace>\d+)$"
)
MATCHED_ROUTE_RE = re.compile(
    r"^R1_MATCHED_DIAG_ROUTE arm=(?P<arm>\w+) "
    r"input_diag_params=(?P<input>\d+) proj_diag_params=(?P<proj>\d+) "
    r"dense_refresh_blocks=(?P<dense>\d+)$"
)


ORIGINAL_PARSE_ARGS = R1.parse_args
ORIGINAL_CONTROLLED_ENV = R1.controlled_env
ORIGINAL_PARSE_METRICS = R1.parse_metrics
ORIGINAL_WRITE_JSON = R1.write_json
ORIGINAL_VALIDATE_METRIC_EVIDENCE = R1.validate_metric_evidence
ORIGINAL_VALIDATE_WANDB_ONLINE_ACCESS = R1.validate_wandb_online_access
_REQUESTED_METHODS: tuple[str, ...] = tuple(ARM_CONFIGS)


def experiment_family(_args) -> str:
    return FAMILY


def experiment_protocol(args, *, smoke: bool | None = None) -> str:
    is_smoke = args.numerical_smoke if smoke is None else smoke
    return PILOT_PROTOCOL if is_smoke else FORMAL_PROTOCOL


def evidence_eligibility(args) -> dict[str, object]:
    engineering_only = bool(getattr(args, "numerical_smoke", False))
    return {
        "quality_usable": not engineering_only,
        "memory_usable": True,
        "timing_usable": False,
        "outcome_eligible": not engineering_only,
        "configuration_selection_allowed": False,
        "reason": (
            "engineering-only exact-shape route/state check; losses cannot select arms or settings"
            if engineering_only
            else "paired quality/state experiment with concurrent physical-GPU units; "
            "wall-clock and throughput are not scientific evidence"
        ),
    }


def validate_secondary_wandb_access(enabled: bool) -> dict[str, object]:
    """Never let the secondary mirror invalidate primary local evidence."""
    if not enabled:
        return {"required": False, "status": "not_checked_secondary"}
    try:
        payload = ORIGINAL_VALIDATE_WANDB_ONLINE_ACCESS(True)
        payload["required"] = False
        payload["scientific_validity_dependency"] = False
        return payload
    except Exception as exc:
        return {
            "required": False,
            "status": "unavailable_secondary",
            "scientific_validity_dependency": False,
            "error": repr(exc),
        }


def parse_args():
    global _REQUESTED_METHODS
    args = ORIGINAL_PARSE_ARGS()
    if args.lr_cross or args.host_bridge:
        raise RuntimeError("Experiment 53 forbids LR-cross and host-bridge modes")
    if args.methods is None or not args.methods:
        raise RuntimeError("Experiment 53 requires at least one explicit arm")
    unknown = set(args.methods) - set(ARM_CONFIGS)
    if unknown:
        raise RuntimeError(f"Experiment 53 received unknown arms: {sorted(unknown)}")
    if args.preflight:
        if set(args.methods) != set(ARM_CONFIGS):
            raise RuntimeError("Experiment-53 preflight must audit all five arms")
    elif len(args.methods) != 1:
        raise RuntimeError("pilot/formal Experiment-53 workers must run exactly one arm")
    _REQUESTED_METHODS = tuple(args.methods)
    return args


def build_all_sources(repo: Path, *, lr_cross: bool = False):
    if lr_cross:
        raise RuntimeError("Experiment 53 forbids LR-cross mode")
    built = build_matched_diag_sources(repo)
    return {arm: built[arm] for arm in _REQUESTED_METHODS}


def controlled_env(args, spec, data_dir: Path, **kwargs) -> dict[str, str]:
    env = ORIGINAL_CONTROLLED_ENV(args, spec, data_dir, **kwargs)
    config = ARM_CONFIGS[spec.name]
    env.update(
        {
            "R1_CFC_K_MODE": config["c_fc"],
            "R1_CPROJ_K_MODE": config["c_proj"],
            "R1_OPROJ_K_MODE": config["o_proj"],
            "R1_QKV_K_MODE": config["qkv"],
            "R1_MATCHED_DIAG_CONTRACT_SHA256": CONTRACT_SHA256,
        }
    )
    return env


def validate_engineering_pilot_manifest(
    path: Path,
    runtime: dict[str, object],
    methods: list[str],
    _formal_seed: int,
    built,
    expected_protocol: str = PILOT_PROTOCOL,
    expected_cuda_visible_devices: str | None = None,
) -> dict[str, object]:
    """Validate an independent seed-2053 certificate without seed aliasing."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"Experiment-53 pilot manifest not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    failures: list[str] = []
    if payload.get("status") != "completed_valid_smoke":
        failures.append("pilot status is not completed_valid_smoke")
    if payload.get("protocol") != expected_protocol:
        failures.append("pilot protocol mismatch")
    if payload.get("batch_kind") != "smoke":
        failures.append("pilot batch kind is not smoke")
    if payload.get("evidence_profile") != "exact_shape_numerical_smoke":
        failures.append("pilot evidence profile is not exact-shape numerical smoke")
    if payload.get("smoke_steps") != 34:
        failures.append("pilot smoke-step count is not 34")
    if payload.get("formal_evidence") is not False:
        failures.append("pilot is incorrectly marked formal")
    if payload.get("methods") != methods:
        failures.append("pilot method list differs from the requested formal arm")
    if payload.get("seed") != PILOT_SEED:
        failures.append(f"pilot seed is not {PILOT_SEED}")
    if payload.get("failures"):
        failures.append("pilot manifest contains failures")
    audit = payload.get("initialization_audit")
    if not isinstance(audit, dict) or audit.get("all_methods_identical") is not True:
        failures.append("pilot initialization audit did not pass")
    summaries = payload.get("summaries")
    if not isinstance(summaries, list):
        failures.append("pilot summaries are missing")
        summaries = []
    elif len(summaries) != len(methods):
        failures.append("pilot summary count differs from the requested formal arm")
    completed = {
        str(item.get("method"))
        for item in summaries
        if isinstance(item, dict)
        and item.get("evidence_valid") is True
        and item.get("formal_evidence") is False
    }
    missing = sorted(set(methods) - completed)
    if missing:
        failures.append(f"pilot did not certify methods: {missing}")
    for item in summaries:
        if not isinstance(item, dict) or str(item.get("method")) not in methods:
            continue
        eligibility = {
            key: item.get(key)
            for key in (
                "quality_usable",
                "memory_usable",
                "timing_usable",
                "outcome_eligible",
                "configuration_selection_allowed",
            )
        }
        if eligibility != {
            "quality_usable": False,
            "memory_usable": True,
            "timing_usable": False,
            "outcome_eligible": False,
            "configuration_selection_allowed": False,
        }:
            failures.append(
                f"pilot eligibility mismatch for {item.get('method')}: {eligibility}"
            )
    observed_runtime = R1.r0.normalize_runtime_fingerprint(
        payload.get("training_runtime_fingerprint")
    )
    expected_runtime = R1.r0.normalize_runtime_fingerprint(
        R1.r0.runtime_fingerprint(runtime)
    )
    if observed_runtime != expected_runtime:
        failures.append("pilot runtime fingerprint differs from formal runtime")
    expected_sources = {arm: item.derived_sha256 for arm, item in built.items()}
    observed_sources = payload.get("derived_source_sha256")
    if not isinstance(observed_sources, dict) or any(
        observed_sources.get(arm) != expected_sources[arm] for arm in methods
    ):
        failures.append("pilot derived-source fingerprint mismatch")
    if expected_cuda_visible_devices is not None:
        isolation = payload.get("resource_isolation")
        visible = isolation.get("cuda_visible_devices") if isinstance(isolation, dict) else None
        if visible != expected_cuda_visible_devices:
            failures.append("pilot visible-GPU certificate mismatch")
    if failures:
        raise RuntimeError("Experiment-53 pilot certificate failed:\n- " + "\n- ".join(failures))
    return {
        "path": str(resolved),
        "manifest_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "validated": True,
        "methods": sorted(completed),
        "engineering_seed": PILOT_SEED,
        "formal_seed_independent": True,
        "outcome_eligible": False,
    }


def validate_metric_evidence(
    rows,
    spec,
    profile,
    metadata,
    k_memory,
    final_memory,
    peak_mib,
    checkpoint_path,
    expected_seed,
    expected_init_sha256,
) -> None:
    """Permit the prespecified zero-state Newton arm without weakening other gates."""
    if spec.name != "all_none":
        ORIGINAL_VALIDATE_METRIC_EVIDENCE(
            rows,
            spec,
            profile,
            metadata,
            k_memory,
            final_memory,
            peak_mib,
            checkpoint_path,
            expected_seed,
            expected_init_sha256,
        )
        return
    if metadata.get("method") != "all_none" or metadata.get("cproj_k_mode") != "none":
        raise RuntimeError(f"all_none metadata mismatch: {metadata}")
    if any(int(k_memory[key]) != 0 for key in ("k_cov", "k_inv", "k_state", "activation", "workspace", "total")):
        raise RuntimeError(f"all_none is not state-free: {k_memory}")
    synthetic_spec = R1.MethodSpec(
        name="muon",
        base_script=spec.base_script,
        cproj_k_mode="muon",
        base_learning_rate=spec.base_learning_rate,
        matrix_learning_rate=spec.matrix_learning_rate,
        role="validation_only_state_free_sentinel",
    )
    synthetic_metadata = dict(metadata)
    synthetic_metadata.update({"method": "muon", "cproj_k_mode": "muon"})
    ORIGINAL_VALIDATE_METRIC_EVIDENCE(
        rows,
        synthetic_spec,
        profile,
        synthetic_metadata,
        k_memory,
        final_memory,
        peak_mib,
        checkpoint_path,
        expected_seed,
        expected_init_sha256,
    )


def _single_match(regex: re.Pattern[str], stdout: str, label: str) -> re.Match[str]:
    matches = [match for line in stdout.splitlines() if (match := regex.match(line))]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label} line, observed {len(matches)}")
    return matches[0]


def parse_metrics(*args, **kwargs):
    rows, summary = ORIGINAL_PARSE_METRICS(*args, **kwargs)
    stdout_path: Path = args[1]
    spec = args[2]
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    metadata = _single_match(MATCHED_METADATA_RE, stdout, "R1_MATCHED_DIAG_METADATA").groupdict()
    route = _single_match(MATCHED_ROUTE_RE, stdout, "R1_MATCHED_DIAG_ROUTE").groupdict()
    config = ARM_CONFIGS[spec.name]
    expected_metadata = {
        "arm": spec.name,
        "cfc": config["c_fc"],
        "cproj": config["c_proj"],
        "oproj": config["o_proj"],
        "qkv": "none",
        "workspace": "0",
    }
    if metadata != expected_metadata:
        raise RuntimeError(
            f"Experiment-53 runtime metadata mismatch: {metadata} != {expected_metadata}"
        )
    expected_route = {
        "arm": spec.name,
        "input": str((12 if config["c_fc"] == "diag" else 0) + (12 if config["o_proj"] == "diag" else 0)),
        "proj": str(12 if config["c_proj"] == "diag" else 0),
        "dense": "0",
    }
    if route != expected_route:
        raise RuntimeError(f"Experiment-53 route mismatch: {route} != {expected_route}")
    expected_memory = expected_memory_contract(spec.name)
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
    if observed_memory != expected_memory:
        raise RuntimeError(
            f"Experiment-53 memory mismatch for {spec.name}: "
            f"observed={observed_memory} expected={expected_memory}"
        )
    for row in rows:
        row.update(
            {
                "placement_arm": spec.name,
                "cfc_k_mode": config["c_fc"],
                "cproj_k_mode": config["c_proj"],
                "oproj_k_mode": config["o_proj"],
                "qkv_k_mode": "none",
            }
        )
    summary.update(
        {
            "placement_arm": spec.name,
            "cfc_k_mode": config["c_fc"],
            "cproj_k_mode": config["c_proj"],
            "oproj_k_mode": config["o_proj"],
            "qkv_k_mode": "none",
            "matched_diag_contract_sha256": CONTRACT_SHA256,
            "dense_preconditioner_reachable": False,
            "script_version": SCRIPT_VERSION,
        }
    )
    return rows, summary


def enriched_json(path: Path, payload: dict[str, object]) -> None:
    enriched = copy.deepcopy(payload)
    method = enriched.get("method")
    config = ARM_CONFIGS.get(str(method))
    enriched["matched_diag_module_placement"] = {
        "script_version": SCRIPT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "pilot_seed": PILOT_SEED,
        "formal_seed_independent": True,
        "qkv_mode": "none",
        "dense_workspace_forbidden": True,
        "timing_usable": False,
    }
    if config is not None:
        enriched["placement_arm"] = method
        enriched["module_modes"] = config
    controls = enriched.get("environment_controls")
    if isinstance(controls, dict) and config is not None:
        controls.update(
            {
                "R1_CFC_K_MODE": config["c_fc"],
                "R1_CPROJ_K_MODE": config["c_proj"],
                "R1_OPROJ_K_MODE": config["o_proj"],
                "R1_QKV_K_MODE": "none",
                "R1_MATCHED_DIAG_CONTRACT_SHA256": CONTRACT_SHA256,
            }
        )
    if path.name == "r1_summary.json":
        checkpoint_value = enriched.get("checkpoint_path")
        if checkpoint_value:
            checkpoint = Path(str(checkpoint_value)).expanduser().resolve()
            if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                raise RuntimeError(
                    f"Experiment-53 cannot certify checkpoint for {path}: {checkpoint}"
                )
            run_root = path.parent.expanduser().resolve()
            try:
                relative = checkpoint.relative_to(run_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Experiment-53 checkpoint escapes its run directory: {checkpoint}"
                ) from exc
            digest = hashlib.sha256()
            with checkpoint.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            enriched.update(
                {
                    "checkpoint_relative_path": relative.as_posix(),
                    "checkpoint_sha256": digest.hexdigest(),
                    "checkpoint_bytes": checkpoint.stat().st_size,
                }
            )
    ORIGINAL_WRITE_JSON(path, enriched)


def install_overrides() -> None:
    R1.FAMILY = FAMILY
    R1.DEFAULT_PROJECT = PROJECT
    R1.DEFAULT_RUN_PREFIX = RUN_PREFIX
    R1.R1_SMOKE_PROTOCOL = PILOT_PROTOCOL
    R1.R1_FORMAL_PROTOCOL = FORMAL_PROTOCOL
    R1.METHODS = METHODS
    R1.parse_args = parse_args
    R1.experiment_family = experiment_family
    R1.experiment_protocol = experiment_protocol
    R1.evidence_eligibility = evidence_eligibility
    R1.validate_wandb_online_access = validate_secondary_wandb_access
    R1.build_all_sources = build_all_sources
    R1.controlled_env = controlled_env
    R1.validate_smoke_manifest = validate_engineering_pilot_manifest
    R1.validate_metric_evidence = validate_metric_evidence
    R1.parse_metrics = parse_metrics
    R1.write_json = enriched_json


def main() -> None:
    install_overrides()
    R1.main()


if __name__ == "__main__":
    main()
