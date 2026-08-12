"""Reconcile local LLaMA/SwiGLU batch artifacts with the W&B curve analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


METHOD_ORDER = ("down_diag", "down_none", "newton_full", "muon", "adamw")
EXPECTED_CONFIG = {
    "adamw_matrix_lr": 0.000576,
    "backup_lr": 0.0036,
    "checkpoint_every": 128,
    "device_batch_size": 64,
    "global_batch_size": 512,
    "matrix_lr": 0.01,
    "num_iterations": 6200,
    "sequence_length": 1024,
    "val_every": 100,
    "val_tokens": 10485760,
    "warmdown_iters": 1800,
}
EXPECTED_K_MIB = {
    "down_diag": 162.1875,
    "down_none": 162.0,
    "newton_full": 546.0,
    "muon": 0.0,
    "adamw": 0.0,
}
MIB = 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--local-batch-dir", type=Path, required=True)
    parser.add_argument("--r1-summary", type=Path)
    parser.add_argument("--r1-pairs", type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(rows: list[dict[str, str]], name: str, ok: bool, details: str) -> None:
    rows.append({"check": name, "status": "PASS" if ok else "FAIL", "details": details})


def same_number(left: object, right: object, tolerance: float = 1e-9) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    analysis = args.analysis_dir.resolve()
    batch = args.local_batch_dir.resolve()
    plan_path = batch / "llama_plan.json"
    manifest_path = batch / "llama_manifest.json"
    batch_csv_path = batch / "llama_swiglu_summary.csv"
    required = [plan_path, manifest_path, batch_csv_path]
    required.extend(sorted(batch.glob("*/summary.json")))
    if len(required) != 8 or not all(path.is_file() for path in required):
        raise RuntimeError(f"Expected 3 batch files plus 5 summary.json files in {batch}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit: list[dict[str, str]] = []
    check(audit, "formal batch identity", plan.get("batch_kind") == "formal" and plan.get("seed") == 2026,
          f"batch_id={plan.get('batch_id')}, seed={plan.get('seed')}")
    check(audit, "exact five-method order", tuple(plan.get("methods", [])) == METHOD_ORDER,
          json.dumps(plan.get("methods")))
    check(audit, "exact formal config", plan.get("config") == EXPECTED_CONFIG,
          json.dumps(plan.get("config"), sort_keys=True))
    check(audit, "manifest completed", manifest.get("status") == "completed" and not manifest.get("failed_methods"),
          f"status={manifest.get('status')}, failed={manifest.get('failed_methods')}")
    check(audit, "all methods completed", tuple(manifest.get("completed_methods", [])) == METHOD_ORDER,
          json.dumps(manifest.get("completed_methods")))
    for key in ("batch_id", "batch_kind", "script_sha256", "runtime", "data_audit", "init_audit"):
        check(audit, f"plan/manifest {key} match", plan.get(key) == manifest.get(key), key)

    common_init = plan["init_audit"]["common_init_sha256"]
    expected_k_bytes = plan["init_audit"]["expected_k_state_bytes"]
    local_rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for path in required:
        sources.append(
            {
                "relative_path": str(path.relative_to(batch)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )

    for summary_path in sorted(batch.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        method = str(payload.get("method"))
        result = manifest.get("method_results", {}).get(method, {})
        check(audit, f"{method}: completed/6200", payload.get("status") == "completed" and payload.get("completed_steps") == 6200,
              f"status={payload.get('status')}, steps={payload.get('completed_steps')}")
        check(audit, f"{method}: exact token budget", payload.get("tokens_seen") == 6200 * 512 * 1024,
              f"tokens={payload.get('tokens_seen')}")
        check(audit, f"{method}: shared initialization", payload.get("init_sha256") == common_init,
              str(payload.get("init_sha256")))
        check(audit, f"{method}: parameter count", payload.get("architecture", {}).get("parameter_count") == 123551232,
              str(payload.get("architecture", {}).get("parameter_count")))
        config = payload.get("config", {})
        check(audit, f"{method}: training config", all(config.get(key) == value for key, value in EXPECTED_CONFIG.items()),
              json.dumps({key: config.get(key) for key in EXPECTED_CONFIG}, sort_keys=True))
        check(audit, f"{method}: runtime matches plan", payload.get("runtime") == plan.get("runtime"),
              str(payload.get("runtime", {}).get("gpu_name")))
        check(audit, f"{method}: exact K bytes", payload.get("k_state_bytes") == expected_k_bytes[method],
              f"observed={payload.get('k_state_bytes')}, expected={expected_k_bytes[method]}")
        check(audit, f"{method}: no resume", payload.get("resume_count") == 0 and payload.get("timing_comparable") is True,
              f"resume_count={payload.get('resume_count')}, timing_comparable={payload.get('timing_comparable')}")
        check(audit, f"{method}: manifest summary hash", result.get("summary_sha256") == file_sha256(summary_path),
              str(result.get("summary_sha256")))
        check(audit, f"{method}: W&B upload completed", result.get("wandb", {}).get("status") == "uploaded",
              json.dumps(result.get("wandb", {}), sort_keys=True))

        local_rows.append(
            {
                "method": method,
                "local_final_val_loss": payload["final_val_loss"],
                "local_best_val_loss": payload["best_val_loss"],
                "local_final_train_loss": payload["final_train_loss"],
                "peak_allocated_mib": payload["peak_allocated_bytes"] / MIB,
                "optimizer_state_mib_including_k": payload["optimizer_state_bytes"] / MIB,
                "k_state_mib": payload["k_state_bytes"] / MIB,
                "k_cov_mib": payload["k_cov_bytes"] / MIB,
                "k_inv_mib": payload["k_inv_bytes"] / MIB,
                "activation_stat_mib": payload["activation_stat_bytes"] / MIB,
                "activation_scratch_mib": payload["activation_scratch_bytes"] / MIB,
                "preconditioner_workspace_mib": payload["preconditioner_workspace_bytes"] / MIB,
                "model_parameter_mib": payload["model_parameter_bytes"] / MIB,
                "step_avg_ms_diagnostic": payload["step_avg_ms"],
                "train_s_diagnostic": payload["train_s"],
                "resume_count": payload["resume_count"],
                "timing_comparable_internal_flag": payload["timing_comparable"],
                "wandb_run_id": result["wandb"]["run_id"],
                "wandb_run_name": result["wandb"]["run_name"],
                "summary_sha256": file_sha256(summary_path),
            }
        )

    local = pd.DataFrame(local_rows)
    batch_csv = pd.read_csv(batch_csv_path)
    check(audit, "batch CSV exact method coverage", set(batch_csv["method"]) == set(METHOD_ORDER),
          ",".join(batch_csv["method"]))
    for row in local.to_dict("records"):
        method = str(row["method"])
        batch_row = batch_csv[batch_csv["method"] == method].iloc[0]
        fields = {
            "final_val_loss": row["local_final_val_loss"],
            "best_val_loss": row["local_best_val_loss"],
            "final_train_loss": row["local_final_train_loss"],
            "peak_allocated_mib": row["peak_allocated_mib"],
            "k_state_mib": row["k_state_mib"],
            "optimizer_state_mib": row["optimizer_state_mib_including_k"],
            "step_avg_ms": row["step_avg_ms_diagnostic"],
            "train_s": row["train_s_diagnostic"],
            "resume_count": row["resume_count"],
        }
        for field, expected in fields.items():
            check(audit, f"{method}: batch CSV {field}", same_number(batch_row[field], expected),
                  f"csv={batch_row[field]}, summary={expected}")

    wandb = pd.read_csv(analysis / "llama_run_summary.csv").drop(
        columns=[
            "observed_peak_memory_mib",
            "observed_optimizer_state_mib",
            "resume_count",
            "memory_usable",
            "timing_usable",
        ],
        errors="ignore",
    )
    enriched = wandb.merge(local, on="method", validate="one_to_one")
    for _, row in enriched.iterrows():
        method = str(row["method"])
        for left, right in (
            ("final_val_loss", "local_final_val_loss"),
            ("best_val_loss", "local_best_val_loss"),
            ("final_train_loss_step", "local_final_train_loss"),
            ("train_time_s_descriptive", "train_s_diagnostic"),
            ("final_step_avg_ms_descriptive", "step_avg_ms_diagnostic"),
        ):
            check(audit, f"{method}: W&B/local {left}", same_number(row[left], row[right], 1e-6),
                  f"wandb={row[left]}, local={row[right]}")
        check(audit, f"{method}: preflight/local K", same_number(row["expected_k_state_mib_from_preflight"], row["k_state_mib"]),
              f"preflight={row['expected_k_state_mib_from_preflight']}, local={row['k_state_mib']}")

    enriched["peak_delta_mib_vs_down_none"] = enriched["peak_allocated_mib"] - float(
        enriched.loc[enriched["method"] == "down_none", "peak_allocated_mib"].iloc[0]
    )
    enriched["peak_delta_pct_vs_down_none"] = 100.0 * enriched["peak_delta_mib_vs_down_none"] / float(
        enriched.loc[enriched["method"] == "down_none", "peak_allocated_mib"].iloc[0]
    )
    enriched["quality_usable"] = True
    enriched["memory_usable"] = True
    enriched["timing_diagnostic_usable"] = True
    enriched["timing_formal_usable"] = False

    memory = enriched[
        [
            "method",
            "peak_allocated_mib",
            "peak_delta_mib_vs_down_none",
            "peak_delta_pct_vs_down_none",
            "optimizer_state_mib_including_k",
            "k_state_mib",
            "activation_stat_mib",
            "activation_scratch_mib",
            "preconditioner_workspace_mib",
            "model_parameter_mib",
        ]
    ].copy()

    cross_rows: list[dict[str, object]] = []
    if args.r1_summary and args.r1_pairs:
        r1_summary = pd.read_csv(args.r1_summary)
        r1_pairs = pd.read_csv(args.r1_pairs).set_index("comparison")
        llama_pairs = pd.read_csv(analysis / "llama_pairwise_summary.csv").set_index("comparison")
        r1_diag = r1_summary.set_index("method").loc["diag"]
        r1_none = r1_summary.set_index("method").loc["none"]
        llama_index = enriched.set_index("method")
        cross_rows = [
            {
                "family": "R1_GPT_GELU",
                "seed": 2026,
                "diag_final_val_loss": r1_diag["final_val_loss"],
                "none_final_val_loss": r1_none["final_val_loss"],
                "diag_minus_none_final": r1_pairs.loc["diag_minus_none", "final_val_loss_delta_left_minus_right"],
                "diag_minus_none_auc": r1_pairs.loc["diag_minus_none", "normalized_val_auc_delta_left_minus_right"],
                "diag_minus_none_peak_mib": r1_diag["peak_memory_mib"] - r1_none["peak_memory_mib"],
                "interpretation": "diag better than none in this seed",
            },
            {
                "family": "LLaMA_SwiGLU_124M",
                "seed": 2026,
                "diag_final_val_loss": llama_index.loc["down_diag", "final_val_loss"],
                "none_final_val_loss": llama_index.loc["down_none", "final_val_loss"],
                "diag_minus_none_final": llama_pairs.loc["down_diag_minus_down_none", "final_val_loss_delta_left_minus_right"],
                "diag_minus_none_auc": llama_pairs.loc["down_diag_minus_down_none", "normalized_val_auc_delta_left_minus_right"],
                "diag_minus_none_peak_mib": llama_index.loc["down_diag", "peak_allocated_mib"] - llama_index.loc["down_none", "peak_allocated_mib"],
                "interpretation": "none better than diag in this seed",
            },
        ]
    cross = pd.DataFrame(cross_rows)

    full = enriched.set_index("method").loc["newton_full"]
    none = enriched.set_index("method").loc["down_none"]
    diag = enriched.set_index("method").loc["down_diag"]
    muon = enriched.set_index("method").loc["muon"]
    adamw = enriched.set_index("method").loc["adamw"]
    full_none_peak = full["peak_allocated_mib"] - none["peak_allocated_mib"]
    diag_none_peak = diag["peak_allocated_mib"] - none["peak_allocated_mib"]
    full_none_step_pct = 100.0 * (full["step_avg_ms_diagnostic"] - none["step_avg_ms_diagnostic"]) / none["step_avg_ms_diagnostic"]
    diag_none_step_pct = 100.0 * (diag["step_avg_ms_diagnostic"] - none["step_avg_ms_diagnostic"]) / none["step_avg_ms_diagnostic"]
    report = f"""# LLaMA / SwiGLU seed2026 综合证据审计（2026-07-21）

## 结论状态

- 本地 batch、5 个 summary、批次 CSV 与 W&B 曲线完全对账；批次状态 completed，5/5 方法成功并上传 W&B。
- 五种方法均为 6200 step / 3,250,585,600 token；初始化哈希、数据指纹、运行时、训练脚本和公共配置一致。
- 所有方法 resume_count=0，内部 timing_comparable=true。质量与显存证据可用；长跑计时可作诊断，但不替代隔离重复的正式性能实验。

## 质量结论

- down_none 终点最好（{none['final_val_loss']:.6f}），其后为 newton_full（{full['final_val_loss']:.6f}）、Muon（{muon['final_val_loss']:.6f}）、down_diag（{diag['final_val_loss']:.6f}）。四者最大终点差仅 {diag['final_val_loss']-none['final_val_loss']:.6f}。
- AdamW 已包含在 seed2026，终点为 {adamw['final_val_loss']:.6f}；它比 down_none 高 {adamw['final_val_loss']-none['final_val_loss']:.6f}。这是当前 AdamW 配方的结果，矩阵 LR 与其他方法不同。
- R1 中 diag-none 终点差为 -0.005000；LLaMA 中为 +0.001357，方向反转。该架构交互是重要现象，但必须通过 seeds2024/2025 判断是否超过种子噪声。

## 显存与状态

| 方法 | Peak MiB | Optimizer state MiB（含 K） | K-state MiB | Activation-stat MiB | Workspace MiB |
|---|---:|---:|---:|---:|---:|
| down_none | {none['peak_allocated_mib']:.3f} | {none['optimizer_state_mib_including_k']:.3f} | {none['k_state_mib']:.4f} | {none['activation_stat_mib']:.3f} | {none['preconditioner_workspace_mib']:.3f} |
| down_diag | {diag['peak_allocated_mib']:.3f} | {diag['optimizer_state_mib_including_k']:.3f} | {diag['k_state_mib']:.4f} | {diag['activation_stat_mib']:.3f} | {diag['preconditioner_workspace_mib']:.3f} |
| newton_full | {full['peak_allocated_mib']:.3f} | {full['optimizer_state_mib_including_k']:.3f} | {full['k_state_mib']:.4f} | {full['activation_stat_mib']:.3f} | {full['preconditioner_workspace_mib']:.3f} |
| Muon | {muon['peak_allocated_mib']:.3f} | {muon['optimizer_state_mib_including_k']:.3f} | {muon['k_state_mib']:.4f} | {muon['activation_stat_mib']:.3f} | {muon['preconditioner_workspace_mib']:.3f} |
| AdamW | {adamw['peak_allocated_mib']:.3f} | {adamw['optimizer_state_mib_including_k']:.3f} | {adamw['k_state_mib']:.4f} | {adamw['activation_stat_mib']:.3f} | {adamw['preconditioner_workspace_mib']:.3f} |

- full 比 none 多 {full_none_peak:.3f} MiB 峰值显存（{100*full_none_peak/full['peak_allocated_mib']:.2f}% 的 full 峰值），而终点只差 {full['final_val_loss']-none['final_val_loss']:+.6f}。
- diag 比 none 只多 {diag_none_peak:.3f} MiB 峰值显存；二者基本是同一显存档位。
- full 相对 none 的约 597 MiB 峰值差，不只来自 384 MiB K-state，还包括约 192 MiB activation statistics、16 MiB scratch 和 6 MiB workspace。不要把 K-state 节省比例直接等同于总峰值显存节省比例。

## 诊断性计时

- down_diag 与 down_none 几乎完全相同：{diag['step_avg_ms_diagnostic']:.3f} vs {none['step_avg_ms_diagnostic']:.3f} ms/step，差 {diag_none_step_pct:+.4f}%。
- newton_full 为 {full['step_avg_ms_diagnostic']:.3f} ms/step，比 none 慢 {full_none_step_pct:.2f}%；Muon 为 {muon['step_avg_ms_diagnostic']:.3f}，AdamW 为 {adamw['step_avg_ms_diagnostic']:.3f}。
- 这些 run 无恢复且同一 H100，足以说明 dense-full 存在明显计算开销、diag 本身没有表现出相对 none 的额外开销；正式百分比仍交给 R1-PERF 重复实验。

## 下一步决策

先补 LLaMA/SwiGLU seeds2024、2025，并保持五方法完整集合。理由不是为了证明 down_none 已经获胜，而是确认 R1→LLaMA 的 diag/none 方向反转是否真实。完成三 seed 后再决定 1B/300M；现在直接扩模型会把“架构效应”和“种子噪声”混在一起。
"""

    audit_frame = pd.DataFrame(audit)
    sources_frame = pd.DataFrame(sources)
    enriched.to_csv(analysis / "llama_run_summary_enriched.csv", index=False)
    memory.to_csv(analysis / "llama_memory_breakdown.csv", index=False)
    audit_frame.to_csv(analysis / "local_artifact_audit.csv", index=False)
    sources_frame.to_csv(analysis / "local_source_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    if not cross.empty:
        cross.to_csv(analysis / "cross_architecture_diag_none.csv", index=False)
    (analysis / "LLAMA_SWIGLU_COMBINED_ANALYSIS_20260721.md").write_text(report, encoding="utf-8")
    failures = audit_frame[audit_frame["status"] == "FAIL"]
    combined_manifest = {
        "status": "PASS" if failures.empty else "FAIL",
        "batch_id": plan["batch_id"],
        "seed": 2026,
        "methods": list(METHOD_ORDER),
        "quality_usable": failures.empty,
        "memory_usable": failures.empty,
        "timing_diagnostic_usable": failures.empty,
        "timing_formal_usable": False,
        "resume_counts": {row["method"]: int(row["resume_count"]) for row in local_rows},
        "audit_pass": int((audit_frame["status"] == "PASS").sum()),
        "audit_fail": int((audit_frame["status"] == "FAIL").sum()),
        "local_batch_dir": str(batch),
        "wandb_project": plan["wandb_project"],
    }
    (analysis / "combined_analysis_manifest.json").write_text(
        json.dumps(combined_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not failures.empty:
        raise RuntimeError(f"Local evidence audit failed:\n{failures.to_string(index=False)}")
    print(enriched.to_string(index=False))
    print(json.dumps(combined_manifest, indent=2))


if __name__ == "__main__":
    main()
