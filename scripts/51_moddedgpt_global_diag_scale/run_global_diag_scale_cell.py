#!/usr/bin/env python3
"""Adapt the accepted Experiment-43/44 cell validators for one EX51 unit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent


def load_module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def argument(name: str) -> Path:
    try:
        return Path(sys.argv[sys.argv.index(name) + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"missing {name}") from exc


def argument_text(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"missing {name}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    scale = os.environ.get("EX51_SCALE")
    if scale not in ("275m", "455m"):
        raise RuntimeError("EX51_SCALE must be 275m or 455m")
    from global_diag_scale_source_builder import expected_memory

    legacy_dir = SCRIPTS / (
        "43_newton_muon_record28_275m"
        if scale == "275m"
        else "44_newton_muon_record17_455m"
    )
    common_name = "record28_common" if scale == "275m" else "record17_common"
    cell_name = "run_record28_cell" if scale == "275m" else "run_record17_cell"
    common = load_module(common_name, legacy_dir / f"{common_name}.py")
    cell = load_module(f"ex51_{cell_name}", legacy_dir / f"{cell_name}.py")
    common.METHODS = ("global_diag",)
    cell.CPROJ_MODES["global_diag"] = "diag"
    cell.CPROJ_SCHEMA_EXPECTED["global_diag"] = dict(
        cell.CPROJ_SCHEMA_EXPECTED["selective_diag"]
    )
    if scale == "455m":
        cell.PRECONDITIONED_PARAMETER_COUNTS["global_diag"] = 47

    cell.main()

    attempt = argument("--attempt-dir")
    stage = argument_text("--stage")
    seed = int(argument_text("--seed"))
    training_source = argument("--training-source")
    contract_path = argument("--contract")
    source_snapshot_manifest = argument("--source-snapshot-manifest")
    data_certificate_path = argument("--data-certificate")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    scientific_path = attempt / "scientific_manifest.json"
    summary_path = attempt / "summary.json"
    scientific = json.loads(scientific_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data_certificate = json.loads(data_certificate_path.read_text(encoding="utf-8"))
    parsed = common.parse_training_log(attempt / "training.log")
    audit = parsed["final_audit"]
    expected = expected_memory(scale)
    route = (
        f"EX51_GLOBAL_DIAG_ROUTE scale={scale} "
        f"parameters={35 if scale == '275m' else 47} dense_activation_workspace=0"
    )
    log_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (attempt / "stdout.log", attempt / "training.log")
        if path.is_file()
    )
    checks = {
        "route_marker": route in log_text,
        "k_cov_bytes": int(audit.get("k_cov_bytes", -1)) == expected["k_cov_bytes"],
        "k_inv_bytes": int(audit.get("k_inv_bytes", -1)) == expected["k_inv_bytes"],
        "k_state_bytes": int(audit.get("k_state_bytes", -1)) == expected["k_state_bytes"],
        "activation_stat_bytes": int(audit.get("activation_stat_bytes", -1))
        == expected["activation_stat_bytes"],
        # A parameter-shaped gradient buffer is required by the accepted
        # optimizer implementation.  It is not a dense K factor.  The
        # source/route certificate and zero activation workspace are the
        # relevant no-dense-factor gates.
        "gradient_precondition_buffer_present": int(audit.get("k_workspace_bytes", 0)) > 0,
        "dense_activation_workspace_absent": int(
            audit.get("activation_workspace_bytes", -1)
        )
        == 0,
        "finite": audit.get("all_finite") is True
        and audit.get("k_tensors_all_finite") is True,
        "legacy_manifest_passed": scientific.get("passed") is True,
        "method_exact": scientific.get("method") == "global_diag"
        and summary.get("method") == "global_diag",
        "scale_exact": scale in contract.get("formal", {}).get("scales", []),
        "stage_exact": scientific.get("stage") == stage
        and summary.get("stage") == stage,
        "seed_exact": int(scientific.get("seed", -1)) == seed
        and int(summary.get("seed", -1)) == seed,
        "parameter_count_exact": int(scientific.get("parameter_count", -1))
        == int(contract["frozen_recipes"][scale]["parameter_count"]),
        "data_certificate_passed": data_certificate.get("passed") is True,
        "data_fingerprint_exact": scientific.get("data_fingerprint_sha256")
        == data_certificate.get("fingerprint_sha256")
        == contract["data"]["accepted_fingerprint_sha256"],
        "training_source_exact": Path(str(scientific.get("training_source", ""))).resolve()
        == training_source,
        "training_source_sha256_exact": scientific.get("derived_source_sha256")
        == sha256_file(training_source),
        "source_snapshot_sha256_exact": scientific.get("source_snapshot_sha256")
        == sha256_file(source_snapshot_manifest),
        "summary_hash_exact": scientific.get("artifact_hashes", {}).get("summary.json")
        == sha256_file(summary_path),
        "steps_self_consistent": int(scientific.get("total_steps", -1))
        == int(summary.get("final_step", -2)),
        "tokens_self_consistent": int(scientific.get("train_tokens", -1))
        == int(summary.get("train_tokens", -2)),
        "formal_schedule_exact": stage != "formal"
        or (
            int(scientific.get("total_steps", -1))
            == int(contract["frozen_recipes"][scale]["updates"])
            and int(scientific.get("train_tokens", -1))
            == int(contract["frozen_recipes"][scale]["train_tokens"])
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"EX51 global-diag certificate failed: {checks}")
    payload = {
        "schema_version": 1,
        "experiment_id": "51_moddedgpt_global_diag_scale",
        "scale": scale,
        "method": "global_diag",
        "stage": stage,
        "seed": seed,
        "passed": True,
        "checks": checks,
        "expected_memory": expected,
        "data_fingerprint_sha256": data_certificate["fingerprint_sha256"],
        "training_source_sha256": sha256_file(training_source),
        "source_snapshot_manifest_sha256": sha256_file(source_snapshot_manifest),
        "summary_sha256": sha256_file(summary_path),
        "legacy_validator_manifest": str(scientific_path),
        "legacy_validator_manifest_sha256": sha256_file(scientific_path),
    }
    common.atomic_write_json(attempt / "ex51_unit_manifest.json", payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
