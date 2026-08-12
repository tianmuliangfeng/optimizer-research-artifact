#!/usr/bin/env python3
"""Read-only CUDA worker for MECH-07 LLaMA-1B family contrasts."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import math
import shutil
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn


SCRIPT_VERSION = "2026-07-27.3"
CONTRACT_VERSION = "2026-07-27.1"
HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M1 = load_module(
    "mech07_m1",
    HERE.parent / "27_mech01_unified_k_diagnostics" / "mech01_worker.py",
)
M2 = load_module(
    "mech07_m2", HERE.parent / "30_mech02_k_geometry" / "mech02_worker.py"
)
M3 = load_module(
    "mech07_m3", HERE.parent / "31_mech03_crossfit_shadow" / "mech03_worker.py"
)
M6 = load_module(
    "mech07_m6",
    HERE.parent / "33_mech06_llama1b_confirmation" / "mech06_worker.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--analysis-tier", required=True, choices=("smoke", "formal"))
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-hash-certificate", required=True, type=Path)
    parser.add_argument("--source-script", required=True, type=Path)
    parser.add_argument("--profile-script", required=True, type=Path)
    parser.add_argument("--triton-kernels", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--data-pattern", required=True)
    parser.add_argument("--layers", nargs="+", required=True, type=int)
    parser.add_argument("--target-kinds", nargs="+", required=True)
    parser.add_argument("--repeat-offsets", nargs="+", required=True, type=int)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--batches-per-split", required=True, type=int)
    parser.add_argument("--device-batch-size", required=True, type=int)
    parser.add_argument("--sequence-length", required=True, type=int)
    parser.add_argument("--max-activation-rows", required=True, type=int)
    parser.add_argument("--ridge-mult", required=True, type=float)
    parser.add_argument("--ridge-eps", required=True, type=float)
    parser.add_argument("--momentum", required=True, type=float)
    parser.add_argument("--ns-steps", required=True, type=int)
    parser.add_argument("--step-multipliers", nargs="+", required=True, type=float)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--execution-domain", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_spec(contract: dict[str, Any], cell: str) -> dict[str, Any]:
    matches = [row for row in contract["checkpoints"] if row["cell"] == cell]
    if len(matches) != 1:
        raise RuntimeError(f"contract checkpoint cell is not unique: {cell}")
    return matches[0]


def validate_contract(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_json(args.contract.resolve())
    spec = checkpoint_spec(contract, args.cell)
    cert = read_json(args.checkpoint_hash_certificate.resolve())
    algorithms = contract["algorithms"]
    comparisons = contract["comparison_contract"]
    checks = {
        "contract_version": contract.get("contract_version") == CONTRACT_VERSION,
        "family": contract.get("family") == "llama1b",
        "cell_path": str(args.checkpoint.resolve()) == spec["path"],
        "certificate_cell": cert.get("cell") == args.cell,
        "certificate_path": cert.get("path") == spec["path"],
        "certificate_passed": cert.get("passed") is True,
        "known_hash_matches": (
            spec["expected_sha256"] is None
            or cert.get("sha256") == spec["expected_sha256"]
        ),
        "source_sha256": (
            M1.sha256_file(args.source_script.resolve())
            == contract["source_constraints"]["base_source_sha256"]
        ),
        "triton_sha256": (
            M1.sha256_file(args.triton_kernels.resolve())
            == contract["source_constraints"]["triton_sha256"]
        ),
        "algorithms_exact": set(algorithms)
        == {
            "muon",
            "original_newton_muon",
            "selective_diag",
            "selective_none",
        },
        "diag_none_excluded": (
            "selective_diag_vs_selective_none"
            in comparisons["excluded_from_primary"]
        ),
        "new_training_disabled": contract["interpretation"]["new_training"] is False,
        "hvp_disabled": contract["interpretation"]["hvp_authorized"] is False,
    }
    audit = {
        "contract_sha256": M1.sha256_file(args.contract.resolve()),
        "checkpoint_spec": spec,
        "certificate": cert,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return contract, audit


def validate_smoke_gate(
    args: argparse.Namespace, contract_sha: str
) -> dict[str, Any]:
    if args.analysis_tier == "smoke":
        return {"required": False, "passed": True}
    if args.smoke_manifest is None:
        return {"required": True, "passed": False, "reason": "missing manifest"}
    manifest = read_json(args.smoke_manifest.resolve())
    checks = {
        "passed": manifest.get("passed") is True,
        "tier": manifest.get("analysis_tier") == "smoke",
        "cell": manifest.get("cell") == args.cell,
        "contract_sha": manifest.get("contract_sha256") == contract_sha,
        "worker_version": manifest.get("script_version") == SCRIPT_VERSION,
    }
    return {
        "required": True,
        "manifest": str(args.smoke_manifest.resolve()),
        "manifest_sha256": M1.sha256_file(args.smoke_manifest.resolve()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def method_identity_audit(
    checkpoint: dict[str, Any],
    expected_method: str,
    inferred_method: str,
) -> dict[str, Any]:
    """Resolve the model-state ambiguity between Muon and AdamW."""
    optimizers = checkpoint.get("optimizers")
    matrix_optimizer = (
        optimizers[1]
        if isinstance(optimizers, list)
        and len(optimizers) >= 2
        and isinstance(optimizers[1], dict)
        else {}
    )
    state = matrix_optimizer.get("state", {})
    entries = list(state.values()) if isinstance(state, dict) else []
    state_keys = sorted(
        {
            str(key)
            for entry in entries
            if isinstance(entry, dict)
            for key in entry
        }
    )
    momentum_tensors = sum(
        isinstance(entry, dict)
        and isinstance(entry.get("momentum"), Tensor)
        for entry in entries
    )
    exact_match = inferred_method == expected_method
    ambiguous_muon_signature = (
        expected_method == "muon"
        and inferred_method == "muon_or_adamw"
        and bool(entries)
        and momentum_tensors == len(entries)
        and "exp_avg" not in state_keys
        and "exp_avg_sq" not in state_keys
    )
    return {
        "expected_method": expected_method,
        "model_state_inferred_method": inferred_method,
        "exact_match": exact_match,
        "matrix_optimizer_state_entries": len(entries),
        "matrix_optimizer_state_keys": state_keys,
        "matrix_optimizer_momentum_tensors": momentum_tensors,
        "muon_not_adamw_state_signature": ambiguous_muon_signature,
        "passed": exact_match or ambiguous_muon_signature,
    }


def target_maps(
    model: nn.Module, layers: list[int], kinds: list[str]
) -> tuple[
    dict[int, nn.Module],
    dict[int, nn.Parameter],
    dict[int, str],
    dict[int, dict[str, Any]],
]:
    modules: dict[int, nn.Module] = {}
    weights: dict[int, nn.Parameter] = {}
    names: dict[int, str] = {}
    metadata: dict[int, dict[str, Any]] = {}
    target_id = 0
    for layer in layers:
        block = model.layers[layer]
        available = {
            "q_proj": (block.attn.q_proj, f"layers.{layer}.attn.q_proj.weight"),
            "o_proj": (block.attn.o_proj, f"layers.{layer}.attn.o_proj.weight"),
            "gate_proj": (
                block.mlp.gate_proj,
                f"layers.{layer}.mlp.gate_proj.weight",
            ),
            "down_proj": (
                block.mlp.down_proj,
                f"layers.{layer}.mlp.down_proj.weight",
            ),
        }
        for kind in kinds:
            if kind not in available:
                raise ValueError(f"unsupported target kind: {kind}")
            module, name = available[kind]
            modules[target_id] = module
            weights[target_id] = module.weight
            names[target_id] = name
            metadata[target_id] = {
                "target_id": target_id,
                "layer": layer,
                "target_kind": kind,
                "algorithm_group": "down" if kind == "down_proj" else "family_core",
                "parameter_name": name,
                "shape": list(module.weight.shape),
            }
            target_id += 1
    return modules, weights, names, metadata


def batch_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_pattern=args.data_pattern,
        repeat_offsets=args.repeat_offsets,
        repeats=args.repeats,
        batches_per_split=args.batches_per_split,
        device_batch_size=args.device_batch_size,
        sequence_length=args.sequence_length,
    )


def assemble_algorithm_updates(
    primitive: dict[int, dict[str, Tensor]],
    metadata: dict[int, dict[str, Any]],
    algorithms: dict[str, dict[str, str]],
) -> tuple[dict[int, dict[str, Tensor]], list[dict[str, Any]]]:
    updates: dict[int, dict[str, Tensor]] = {}
    rows: list[dict[str, Any]] = []
    for target_id in sorted(primitive):
        group = metadata[target_id]["algorithm_group"]
        updates[target_id] = {}
        for algorithm, mapping in algorithms.items():
            representation = mapping[group]
            update = primitive[target_id][representation]
            updates[target_id][algorithm] = update
            rows.append(
                {
                    **metadata[target_id],
                    "algorithm": algorithm,
                    "representation": representation,
                    "update_norm": float(torch.linalg.vector_norm(update.float())),
                    "update_sha256": M1.tensor_sha256(update),
                    "cosine_to_muon": 0.0,
                    "cosine_to_original_newton_muon": 0.0,
                }
            )
        muon = updates[target_id]["muon"].float()
        original = updates[target_id]["original_newton_muon"].float()
        for row in rows[-len(algorithms) :]:
            update = updates[target_id][row["algorithm"]].float()
            row["cosine_to_muon"] = M1.matrix_cosine(update, muon)
            row["cosine_to_original_newton_muon"] = M1.matrix_cosine(
                update, original
            )
    return updates, rows


def evaluate_family_line_search(
    model: nn.Module,
    weights: dict[int, nn.Parameter],
    metadata: dict[int, dict[str, Any]],
    updates: dict[int, dict[str, Tensor]],
    heldout_batch: tuple[Tensor, Tensor],
    algorithms: list[str],
    step_multipliers: list[float],
    optimizer_hyperparameters: dict[str, Any],
    repeat: int,
    direction: str,
    build_split: str,
    eval_split: str,
    evaluation_batches: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    originals = {
        target_id: weight.detach().clone()
        for target_id, weight in weights.items()
    }
    step_hyperparameters = {
        int(row["layer"]): row for row in optimizer_hyperparameters["targets"]
    }
    scopes = [
        (
            "family_core",
            [
                target_id
                for target_id in weights
                if metadata[target_id]["algorithm_group"] == "family_core"
            ],
        ),
        (
            "down",
            [
                target_id
                for target_id in weights
                if metadata[target_id]["algorithm_group"] == "down"
            ],
        ),
        ("all", sorted(weights)),
    ]
    snapshot = M3.rng_snapshot()
    baseline = M3.evaluate_loss(model, heldout_batch, snapshot)
    loss_rows: list[dict[str, Any]] = []
    try:
        for scope, target_ids in scopes:
            for algorithm in algorithms:
                for multiplier in step_multipliers:
                    if multiplier == 0:
                        loss = baseline
                    else:
                        M3.apply_shadow(
                            weights,
                            originals,
                            updates,
                            target_ids,
                            algorithm,
                            multiplier,
                            step_hyperparameters,
                        )
                        loss = M3.evaluate_loss(model, heldout_batch, snapshot)
                        M3.restore_weights(weights, originals)
                    delta = loss - baseline
                    effective_lrs = [
                        step_hyperparameters[target_id][
                            "effective_update_learning_rate"
                        ]
                        for target_id in target_ids
                    ]
                    loss_rows.append(
                        {
                            "repeat": repeat,
                            "direction": direction,
                            "build_split": build_split,
                            "eval_split": eval_split,
                            "scope": scope,
                            "target_count": len(target_ids),
                            "algorithm": algorithm,
                            "step_multiplier": multiplier,
                            "evaluation_batches": evaluation_batches,
                            "baseline_loss": baseline,
                            "shadow_loss": loss,
                            "loss_delta": delta,
                            "relative_loss_delta": delta
                            / max(abs(baseline), 1e-30),
                            "effective_step_size_min": multiplier
                            * min(effective_lrs),
                            "effective_step_size_max": multiplier
                            * max(effective_lrs),
                        }
                    )
    finally:
        M3.restore_weights(weights, originals)
    summary: list[dict[str, Any]] = []
    for scope in ("family_core", "down", "all"):
        for algorithm in algorithms:
            subset = [
                row
                for row in loss_rows
                if row["scope"] == scope and row["algorithm"] == algorithm
            ]
            best = min(
                subset,
                key=lambda row: (
                    float(row["shadow_loss"]),
                    float(row["step_multiplier"]),
                ),
            )
            summary.append(
                {
                    "repeat": repeat,
                    "direction": direction,
                    "build_split": build_split,
                    "eval_split": eval_split,
                    "scope": scope,
                    "algorithm": algorithm,
                    "evaluation_batches": evaluation_batches,
                    "baseline_loss": baseline,
                    "best_loss": best["shadow_loss"],
                    "best_loss_delta": best["loss_delta"],
                    "best_relative_loss_delta": best["relative_loss_delta"],
                    "best_step_multiplier": best["step_multiplier"],
                    "best_effective_step_size_min": best[
                        "effective_step_size_min"
                    ],
                    "best_effective_step_size_max": best[
                        "effective_step_size_max"
                    ],
                }
            )
    return loss_rows, summary


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MECH-07 requires CUDA")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    output = args.output_dir.resolve()
    if not output.is_dir():
        raise RuntimeError(f"controller must create output directory: {output}")
    M1.atomic_json(
        output / "status.json",
        {"status": "running", "script_version": SCRIPT_VERSION},
    )

    contract, contract_audit = validate_contract(args)
    if not contract_audit["passed"]:
        raise RuntimeError(f"contract audit failed: {contract_audit}")
    smoke_gate = validate_smoke_gate(args, contract_audit["contract_sha256"])
    if not smoke_gate["passed"]:
        raise RuntimeError(f"smoke gate failed: {smoke_gate}")
    shutil.copyfile(args.contract, output / "family_contrast_contract.json")

    spec = contract_audit["checkpoint_spec"]
    checkpoint_path = args.checkpoint.resolve()
    before_stat = checkpoint_path.stat()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actual_sha = contract_audit["certificate"]["sha256"]
    model_state, schema = M1.checkpoint_schema(
        checkpoint_path,
        checkpoint,
        "llama1b",
        args.source_script.resolve(),
        actual_sha,
        False,
    )
    M1.attach_profile_provenance(schema, args.profile_script)
    architecture = schema["architecture"]
    method_audit = method_identity_audit(
        checkpoint, spec["method"], schema["method_inferred"]
    )
    schema["method_identity_audit"] = method_audit
    architecture_checks = {
        "step": int(schema["step"]) == int(spec["step"]),
        "method": method_audit["passed"],
        "n_layer": architecture["n_layer"] == contract["architecture"]["n_layer"],
        "n_embd": architecture["n_embd"] == contract["architecture"]["n_embd"],
    }
    M1.atomic_json(output / "checkpoint_schema.json", schema)
    M1.atomic_json(output / "method_identity_audit.json", method_audit)
    if not schema["passed"] or not all(architecture_checks.values()):
        raise RuntimeError(
            f"checkpoint schema/architecture failed: {architecture_checks}"
        )

    source_runtime, production_ns, triton_audit = M1.load_source_runtime(
        "llama1b", args.source_script.resolve(), args.triton_kernels.resolve()
    )
    source_config = M1.configure_source_runtime_globals(
        "llama1b", source_runtime, spec["method"]
    )
    model = M1.build_model(
        "llama1b", source_runtime, architecture, spec["method"]
    )
    incompatible = model.load_state_dict(model_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"model load mismatch: {incompatible}")

    modules, weights, names, metadata = target_maps(
        model, args.layers, args.target_kinds
    )
    momenta, momentum_audit = M1.extract_target_momenta(
        checkpoint, model, "llama1b", names
    )
    optimizer_hyperparameters = M3.matrix_optimizer_hyperparameters(
        checkpoint, weights
    )
    model_before = M1.model_state_signature(model, names.values())
    aux_before = M1.checkpoint_aux_signature(checkpoint)
    model.cuda()

    batches, batch_contract = M3.read_crossfit_batches(batch_args(args))
    split_builds: dict[int, dict[str, dict[str, Any]]] = {}
    for repeat in range(args.repeats):
        split_builds[repeat] = {}
        for split in M3.SPLITS:
            split_builds[repeat][split] = M3.collect_build_split(
                model,
                modules,
                weights,
                {},
                batches[repeat][split],
                args.max_activation_rows,
            )

    primitive_rows: list[dict[str, Any]] = []
    algorithm_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    algorithm_names = list(contract["algorithms"])
    for repeat in range(args.repeats):
        for direction, build_split, eval_split in M3.DIRECTIONS:
            build = split_builds[repeat][build_split]
            primitive, primitive_geometry = M6.candidate_updates(
                build["activations"],
                build["gradients"],
                momenta,
                args,
                production_ns,
                repeat,
                direction,
                build_split,
            )
            for row in primitive_geometry:
                row.update(metadata[int(row["layer"])])
                row["checkpoint_cell"] = args.cell
                row["checkpoint_method"] = spec["method"]
                row["checkpoint_stage"] = spec["stage"]
            assembled, assembled_geometry = assemble_algorithm_updates(
                primitive, metadata, contract["algorithms"]
            )
            for row in assembled_geometry:
                row.update(
                    {
                        "repeat": repeat,
                        "direction": direction,
                        "build_split": build_split,
                        "checkpoint_cell": args.cell,
                        "checkpoint_method": spec["method"],
                        "checkpoint_stage": spec["stage"],
                    }
                )
            losses, summaries = evaluate_family_line_search(
                model,
                weights,
                metadata,
                assembled,
                batches[repeat][eval_split],
                algorithm_names,
                args.step_multipliers,
                optimizer_hyperparameters,
                repeat,
                direction,
                build_split,
                eval_split,
                args.batches_per_split,
            )
            for row in [*losses, *summaries]:
                row.update(
                    {
                        "checkpoint_cell": args.cell,
                        "checkpoint_method": spec["method"],
                        "checkpoint_stage": spec["stage"],
                    }
                )
            primitive_rows.extend(primitive_geometry)
            algorithm_rows.extend(assembled_geometry)
            loss_rows.extend(losses)
            summary_rows.extend(summaries)
            del primitive, assembled
            torch.cuda.empty_cache()

    model_after = M1.model_state_signature(model, names.values())
    aux_after = M1.checkpoint_aux_signature(checkpoint)
    after_stat = checkpoint_path.stat()
    invariance = {
        "model_content_unchanged": M3.model_content_unchanged(
            model_before, model_after
        ),
        "optimizer_loader_unchanged": aux_before == aux_after,
        "checkpoint_file_unchanged": (
            before_stat.st_size == after_stat.st_size
            and before_stat.st_mtime_ns == after_stat.st_mtime_ns
        ),
        "model_signature_before": model_before,
        "model_signature_after": model_after,
        "optimizer_loader_signature_before": aux_before,
        "optimizer_loader_signature_after": aux_after,
        "checkpoint_stat_before": {
            "size": before_stat.st_size,
            "mtime_ns": before_stat.st_mtime_ns,
        },
        "checkpoint_stat_after": {
            "size": after_stat.st_size,
            "mtime_ns": after_stat.st_mtime_ns,
        },
    }
    directions = args.repeats * len(M3.DIRECTIONS)
    targets = len(metadata)
    expected_primitive = directions * targets * 3
    expected_algorithm = directions * targets * len(algorithm_names)
    expected_summary = directions * 3 * len(algorithm_names)
    expected_losses = expected_summary * len(args.step_multipliers)
    checks = {
        "contract_audit": contract_audit["passed"],
        "smoke_gate": smoke_gate["passed"],
        "checkpoint_schema": schema["passed"] and all(architecture_checks.values()),
        "source_runtime": source_config["passed"],
        "triton_provenance": triton_audit["passed"],
        "batch_contract": batch_contract["all_windows_disjoint"],
        "historical_momentum": momentum_audit["all_present"],
        "optimizer_hyperparameters": optimizer_hyperparameters["passed"],
        "primitive_rows": len(primitive_rows) == expected_primitive,
        "algorithm_rows": len(algorithm_rows) == expected_algorithm,
        "summary_rows": len(summary_rows) == expected_summary,
        "loss_rows": len(loss_rows) == expected_losses,
        "primitive_finite": M1.finite_numbers(primitive_rows),
        "algorithm_finite": M1.finite_numbers(algorithm_rows),
        "shadow_finite": M1.finite_numbers(loss_rows)
        and M1.finite_numbers(summary_rows),
        "woodbury_health": all(
            row["candidate"] != "dense_full"
            or float(row["inverse_residual_relative"]) <= 1e-3
            for row in primitive_rows
        ),
        "model_content_unchanged": invariance["model_content_unchanged"],
        "optimizer_loader_unchanged": invariance["optimizer_loader_unchanged"],
        "checkpoint_file_unchanged": invariance["checkpoint_file_unchanged"],
        "no_optimizer_step": True,
        "hvp_not_run": True,
    }
    passed = all(checks.values())
    artifacts = {
        "contract_audit.json": contract_audit,
        "smoke_gate.json": smoke_gate,
        "checkpoint_schema.json": schema,
        "method_identity_audit.json": method_audit,
        "architecture_checks.json": architecture_checks,
        "source_runtime_config.json": source_config,
        "triton_audit.json": triton_audit,
        "batch_contract.json": batch_contract,
        "target_metadata.json": {"targets": list(metadata.values())},
        "momentum_audit.json": momentum_audit,
        "matrix_optimizer_hyperparameters.json": optimizer_hyperparameters,
        "state_invariance.json": invariance,
        "checks.json": checks,
    }
    for name, value in artifacts.items():
        M1.atomic_json(output / name, value)
    write_csv(output / "primitive_update_geometry.csv", primitive_rows)
    write_csv(output / "algorithm_update_geometry.csv", algorithm_rows)
    write_csv(output / "shadow_losses.csv", loss_rows)
    write_csv(output / "line_search_summary.csv", summary_rows)
    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "analysis_tier": args.analysis_tier,
        "passed": passed,
        "cell": args.cell,
        "checkpoint_method": spec["method"],
        "checkpoint_stage": spec["stage"],
        "checkpoint_step": spec["step"],
        "checkpoint_sha256": actual_sha,
        "contract_sha256": contract_audit["contract_sha256"],
        "layers": args.layers,
        "target_kinds": args.target_kinds,
        "targets": targets,
        "repeats": args.repeats,
        "algorithms": algorithm_names,
        "primitive_update_rows": len(primitive_rows),
        "algorithm_update_rows": len(algorithm_rows),
        "shadow_loss_rows": len(loss_rows),
        "line_search_summary_rows": len(summary_rows),
        "new_training": False,
        "optimizer_step_called": False,
        "hvp_run": False,
        "artifacts": sorted(path.name for path in output.iterdir()),
    }
    M1.atomic_json(output / "mech07_manifest.json", manifest)
    M1.atomic_json(
        output / "status.json",
        {
            "status": "passed" if passed else "failed",
            "script_version": SCRIPT_VERSION,
        },
    )
    if not passed:
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    try:
        run_worker(args)
    except Exception as exc:
        output = args.output_dir.resolve()
        if output.is_dir():
            M1.atomic_json(
                output / "status.json",
                {
                    "status": "failed",
                    "script_version": SCRIPT_VERSION,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    main()
