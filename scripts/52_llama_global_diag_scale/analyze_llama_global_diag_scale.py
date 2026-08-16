#!/usr/bin/env python3
"""Formal endpoint analysis for Experiment 52."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

SCALES = ("124m", "1b")
SEEDS = (2024, 2025, 2026)
CONTROLS = ("down_diag", "down_none", "newton_full", "muon")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def accepted_certificate(root: Path) -> Path:
    paths = [
        path
        for path in sorted(root.glob("**/ex52_unit_manifest.json"))
        if read_json(path).get("passed") is True
    ]
    if len(paths) != 1:
        raise RuntimeError(f"expected one accepted EX52 certificate below {root}; observed {len(paths)}")
    return paths[0]


def curve(metrics: Path) -> tuple[float, float, int]:
    with metrics.open("r", encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["event"] == "val"]
    rows.sort(key=lambda r: int(r["step"]))
    if len(rows) != 63 or int(rows[-1]["step"]) != 6200:
        raise RuntimeError(f"invalid formal validation grid: {metrics}")
    losses = [float(r["loss"]) for r in rows]
    area = sum((int(b["step"])-int(a["step"])) * (float(a["loss"])+float(b["loss"]))/2 for a,b in zip(rows, rows[1:])) / 6200
    return statistics.mean(losses[-5:]), area, len(rows)


def formal_rows(run_dir: Path, contract: dict[str, Any], contract_sha256: str) -> list[dict[str, Any]]:
    rows = []
    expected_k = {"124m": 417792, "1b": 1677312}
    snapshot_manifest = run_dir / "source_snapshot/source_snapshot_manifest.json"
    snapshot_sha256 = sha256_file(snapshot_manifest)
    expected_data = contract["data"]["accepted_full_content_fingerprint_sha256"]
    for scale in SCALES:
        for seed in SEEDS:
            certificate = accepted_certificate(run_dir / "formal" / scale / f"seed{seed}")
            cert = read_json(certificate)
            summary_path = Path(cert["paths"]["summary"])
            metrics_path = Path(cert["paths"]["metrics"])
            summary = read_json(summary_path)
            tail5, auc, points = curve(metrics_path)
            architecture = summary.get("architecture", {})
            checks = {
                "certificate_identity": cert.get("experiment_id") == "52_llama_global_diag_scale",
                "certificate_stage": cert.get("stage") == "formal" and cert.get("controller_stage") == "formal",
                "certificate_scale": cert.get("scale") == scale,
                "certificate_seed": int(cert.get("seed", -1)) == seed,
                "certificate_method": cert.get("method") == "global_diag",
                "certificate_checks": bool(cert.get("checks")) and all(value is True for value in cert["checks"].values()),
                "contract": cert.get("contract_sha256") == contract_sha256,
                "snapshot": cert.get("source_snapshot_manifest_sha256") == snapshot_sha256,
                "data": cert.get("data_fingerprint_sha256") == expected_data,
                "summary_hash": cert.get("artifacts", {}).get("summary.json") == sha256_file(summary_path),
                "metrics_hash": cert.get("artifacts", {}).get("metrics.csv") == sha256_file(metrics_path),
                "parent_manifest_hash": cert.get("artifacts", {}).get("llama_manifest.json") == sha256_file(Path(cert["paths"]["llama_manifest"])),
                "parent_plan_hash": cert.get("artifacts", {}).get("llama_plan.json") == sha256_file(Path(cert["paths"]["llama_plan"])),
                "dependency_hash": isinstance(cert.get("dependency_certificate"), str) and Path(cert["dependency_certificate"]).is_file() and cert.get("dependency_certificate_sha256") == sha256_file(Path(cert["dependency_certificate"])),
                "accepted_init": cert.get("init_sha256") == contract["accepted_init_sha256"][scale][str(seed)],
                "method": summary.get("method") == "global_diag",
                "seed": int(summary.get("seed", -1)) == seed,
                "steps": int(summary.get("completed_steps", -1)) == 6200,
                "tokens": int(summary.get("tokens_seen", -1)) == 3_250_585_600,
                "k_state": int(summary.get("k_state_bytes", -1)) == expected_k[scale],
                "global_route": architecture.get("global_diag_route") is True,
                # The optimizer retains parameter-shaped gradient workspaces;
                # only dense K/activation scratch is forbidden.  The two
                # scalar 1x1 scratch tensors total exactly eight FP32 bytes.
                "no_dense_activation_scratch": int(summary.get("activation_scratch_bytes", -1)) == 8,
                "all_groups_diag": {g["kind"] for g in architecture.get("preconditioner_groups", [])} == {"diag"},
            }
            if not all(checks.values()):
                raise RuntimeError(f"EX52 certificate failed {scale}/seed{seed}: {checks}")
            rows.append({"scale": scale, "method": "global_diag", "seed": seed, "final_val_loss": float(summary["final_val_loss"]), "best_val_loss": float(summary["best_val_loss"]), "tail5_mean": tail5, "normalized_auc": auc, "validation_points": points, "k_state_bytes": expected_k[scale], "summary_sha256": sha256_file(summary_path), "unit_certificate_sha256": sha256_file(certificate)})
    return rows


def control_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(r) for r in csv.DictReader(handle)]
    cells = {(r["scale"], r["method"], int(r["seed"])) for r in rows}
    expected = {(s,m,seed) for s in SCALES for m in CONTROLS for seed in SEEDS}
    if len(rows) != 24 or cells != expected:
        raise RuntimeError(f"frozen LLaMA controls mismatch: {cells ^ expected}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.contract)
    if contract.get("experiment_id") != "52_llama_global_diag_scale":
        raise RuntimeError("wrong EX52 contract")
    contract_sha256 = sha256_file(args.contract)
    if contract.get("contract_version") not in ("2026-08-14.3", "2026-08-14.4"):
        raise RuntimeError("EX52 analyzer requires a supported .3/.4 contract")
    if sha256_file(args.controls) != contract["frozen_controls"]["sha256"]:
        raise RuntimeError("frozen LLaMA controls hash differs from the contract")
    formal = formal_rows(args.run_dir, contract, contract_sha256)
    frozen = control_rows(args.controls)
    all_rows = [*formal, *[{**r, "seed": int(r["seed"]), "final_val_loss": float(r["final_val_loss"]), "tail5_mean": float(r["tail5_mean"])} for r in frozen]]
    margin = float(contract["analysis"]["descriptive_margin"])
    contrasts = []
    summaries = []
    for scale in SCALES:
        for method in ("global_diag", *CONTROLS):
            values = [float(r["final_val_loss"]) for r in all_rows if r["scale"] == scale and r["method"] == method]
            summaries.append({"scale": scale, "method": method, "n": len(values), "final_val_loss_mean": statistics.mean(values), "final_val_loss_sd": statistics.stdev(values)})
        for control in CONTROLS:
            deltas = []
            for seed in SEEDS:
                new = next(float(r["final_val_loss"]) for r in all_rows if r["scale"] == scale and r["method"] == "global_diag" and int(r["seed"]) == seed)
                old = next(float(r["final_val_loss"]) for r in all_rows if r["scale"] == scale and r["method"] == control and int(r["seed"]) == seed)
                deltas.append(new-old)
            mean = statistics.mean(deltas)
            label = "descriptively_close" if abs(mean) <= margin else "global_diag_better" if mean < 0 else "global_diag_worse"
            contrasts.append({"scale": scale, "contrast": f"global_diag-minus-{control}", "mean_delta": mean, "wins": sum(v < 0 for v in deltas), "classification": label})
    out = args.run_dir / "analysis"
    write_csv(out / "formal_endpoints.csv", formal)
    write_csv(out / "method_summary.csv", summaries)
    write_csv(out / "paired_contrasts.csv", contrasts)
    primary = [r for r in contrasts if r["contrast"] == "global_diag-minus-down_diag"]
    snapshot_sha256 = sha256_file(args.run_dir / "source_snapshot/source_snapshot_manifest.json")
    preflight = read_json(args.run_dir / "preflight/preflight_manifest.json")
    checks = {"formal_units": len(formal) == 6, "primary_contrasts": len(primary) == 2, "finite": all(math.isfinite(r["final_val_loss"]) for r in formal), "controls_hash_bound": sha256_file(args.controls) == contract["frozen_controls"]["sha256"], "data_fingerprint_bound": preflight.get("data_fingerprint_sha256") == contract["data"]["accepted_full_content_fingerprint_sha256"], "timing_excluded": contract["analysis"]["timing_eligible"] is False}
    manifest = {"schema_version": 2, "experiment_id": contract["experiment_id"], "passed": all(checks.values()), "checks": checks, "contract_sha256": contract_sha256, "controls_sha256": sha256_file(args.controls), "data_fingerprint_sha256": preflight.get("data_fingerprint_sha256"), "source_snapshot_manifest_sha256": snapshot_sha256, "scientific_result": {r["scale"]: r["classification"] for r in primary}, "artifacts": {name: sha256_file(out/name) for name in ("formal_endpoints.csv", "method_summary.csv", "paired_contrasts.csv")}}
    write_json(out / "analysis_manifest.json", manifest)
    if not manifest["passed"]: raise SystemExit(2)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__": main()
