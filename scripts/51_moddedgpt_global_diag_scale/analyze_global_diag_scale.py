#!/usr/bin/env python3
"""Fail-closed analysis for Experiment 51."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

SCALES = ("275m", "455m")
FORMAL_SEEDS = {"275m": (2024, 2025, 2026, 2027), "455m": (2024, 2025, 2026)}
CONTROL_METHODS = ("muon", "original_newton_muon", "selective_none", "selective_diag")
T95_DF2 = 4.302652729911275
T95_DF3 = 3.182446305284263


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_formal(run_dir: Path, contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_fingerprint = contract["data"]["accepted_fingerprint_sha256"]
    for scale in SCALES:
        for seed in FORMAL_SEEDS[scale]:
            root = run_dir / "formal" / scale / f"seed{seed}"
            accepted = []
            for cert in sorted(root.glob("attempt_*/ex51_unit_manifest.json")):
                payload = read_json(cert)
                scientific = cert.with_name("scientific_manifest.json")
                summary = cert.with_name("summary.json")
                if payload.get("passed") is True and scientific.is_file() and summary.is_file():
                    sm = read_json(scientific)
                    summary_payload = read_json(summary)
                    lineage = {
                        "unit_scale": payload.get("scale") == scale,
                        "unit_method": payload.get("method") == "global_diag",
                        "unit_stage": payload.get("stage") == "formal",
                        "unit_seed": int(payload.get("seed", -1)) == seed,
                        "unit_data": payload.get("data_fingerprint_sha256")
                        == expected_fingerprint,
                        "legacy_hash": payload.get("legacy_validator_manifest_sha256")
                        == sha256_file(scientific),
                        "summary_hash": payload.get("summary_sha256")
                        == sha256_file(summary),
                        "legacy_passed": sm.get("passed") is True,
                        "legacy_stage": sm.get("stage") == "formal",
                        "legacy_method": sm.get("method") == "global_diag",
                        "legacy_seed": int(sm.get("seed", -1)) == seed,
                        "legacy_data": sm.get("data_fingerprint_sha256")
                        == expected_fingerprint,
                        "legacy_source": sm.get("derived_source_sha256")
                        == payload.get("training_source_sha256"),
                        "legacy_snapshot": sm.get("source_snapshot_sha256")
                        == payload.get("source_snapshot_manifest_sha256"),
                        "summary_method": summary_payload.get("method") == "global_diag",
                        "summary_seed": int(summary_payload.get("seed", -1)) == seed,
                        "summary_stage": summary_payload.get("stage") == "formal",
                        "formal_steps": int(summary_payload.get("final_step", -1))
                        == int(contract["frozen_recipes"][scale]["updates"]),
                        "formal_tokens": int(summary_payload.get("train_tokens", -1))
                        == int(contract["frozen_recipes"][scale]["train_tokens"]),
                    }
                    if not all(lineage.values()):
                        raise RuntimeError(
                            f"EX51 lineage failure for {scale}/seed{seed}: {lineage}"
                        )
                    accepted.append((cert, summary_payload, payload))
            if len(accepted) != 1:
                raise RuntimeError(f"expected one accepted unit for {scale}/seed{seed}; observed {len(accepted)}")
            cert, summary, payload = accepted[0]
            rows.append({
                "scale": scale,
                "method": "global_diag",
                "seed": seed,
                "final_val_loss": float(summary["final_val_loss"]),
                "best_val_loss": float(summary["best_val_loss"]),
                "tail5_mean": float(summary["tail5_mean"]),
                "normalized_auc": float(summary["normalized_auc"]),
                "k_state_bytes": int(payload["expected_memory"]["k_state_bytes"]),
                "source_certificate": str(cert),
            })
    return rows


def controls(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    cells = {(r["scale"], r["method"], int(r["seed"])) for r in rows}
    expected = {
        (scale, method, seed)
        for scale in SCALES
        for method in CONTROL_METHODS
        for seed in FORMAL_SEEDS[scale]
    }
    if cells != expected or len(rows) != 28:
        raise RuntimeError(f"frozen control grid mismatch: {cells ^ expected}")
    return rows


def summarize(formal: list[dict[str, Any]], frozen: list[dict[str, Any]], margin: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = [
        *formal,
        *[
            {
                **row,
                "seed": int(row["seed"]),
                "final_val_loss": float(row["final_val_loss"]),
            }
            for row in frozen
        ],
    ]
    method_rows: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for scale in SCALES:
        seeds = FORMAL_SEEDS[scale]
        n = len(seeds)
        t95 = T95_DF3 if n == 4 else T95_DF2
        methods = ("global_diag", *CONTROL_METHODS)
        for method in methods:
            vals = [float(r["final_val_loss"]) for r in all_rows if r["scale"] == scale and r["method"] == method]
            if len(vals) != n:
                raise RuntimeError(f"missing {scale}/{method} values")
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals)
            half = t95 * sd / math.sqrt(n)
            method_rows.append({"scale": scale, "method": method, "n": n, "final_val_loss_mean": mean, "final_val_loss_sd": sd, "ci95_low": mean-half, "ci95_high": mean+half})
        for control in CONTROL_METHODS:
            deltas = []
            for seed in seeds:
                new = next(float(r["final_val_loss"]) for r in all_rows if r["scale"] == scale and r["method"] == "global_diag" and int(r["seed"]) == seed)
                old = next(float(r["final_val_loss"]) for r in all_rows if r["scale"] == scale and r["method"] == control and int(r["seed"]) == seed)
                deltas.append(new-old)
            mean = statistics.mean(deltas)
            label = "descriptively_close"
            if mean > margin:
                label = "global_diag_worse"
            elif mean < -margin:
                label = "global_diag_better"
            paired.append({"scale": scale, "contrast": f"global_diag-minus-{control}", "n": n, "mean_delta": mean, "wins": sum(v < 0 for v in deltas), "classification": label})
    return method_rows, paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.contract)
    if contract.get("experiment_id") != "51_moddedgpt_global_diag_scale":
        raise RuntimeError("wrong Experiment-51 contract")
    if sha256_file(args.controls) != contract["frozen_controls"]["sha256"]:
        raise RuntimeError("Experiment-51 frozen control table hash mismatch")
    formal = collect_formal(args.run_dir, contract)
    frozen = controls(args.controls)
    method_rows, paired = summarize(formal, frozen, float(contract["analysis"]["descriptive_margin"]))
    output = args.run_dir / "analysis"
    write_csv(output / "formal_endpoints.csv", formal)
    write_csv(output / "method_summary.csv", method_rows)
    write_csv(output / "paired_contrasts.csv", paired)
    primary = [r for r in paired if r["contrast"] == "global_diag-minus-selective_diag"]
    checks = {"formal_units": len(formal) == 7, "primary_contrasts": len(primary) == 2, "all_finite": all(math.isfinite(float(r["final_val_loss"])) for r in formal), "data_fingerprint_bound": all(read_json(Path(r["source_certificate"]))["data_fingerprint_sha256"] == contract["data"]["accepted_fingerprint_sha256"] for r in formal), "controls_hash_bound": sha256_file(args.controls) == contract["frozen_controls"]["sha256"], "timing_excluded": contract["analysis"]["timing_eligible"] is False}
    manifest = {"schema_version": 1, "experiment_id": contract["experiment_id"], "passed": all(checks.values()), "checks": checks, "contract_sha256": sha256_file(args.contract), "controls_sha256": sha256_file(args.controls), "data_fingerprint_sha256": contract["data"]["accepted_fingerprint_sha256"], "scientific_result": {r["scale"]: r["classification"] for r in primary}, "artifacts": {name: sha256_file(output / name) for name in ("formal_endpoints.csv", "method_summary.csv", "paired_contrasts.csv")}}
    write_json(output / "analysis_manifest.json", manifest)
    if not manifest["passed"]:
        raise SystemExit(2)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
