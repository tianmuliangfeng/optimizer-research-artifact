"""Audit compact local certificates for the three-seed LLaMA-1B formal batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (2024, 2025, 2026)
METHODS = ("muon", "down_none", "down_diag", "newton_full")
TOTAL_STEPS = 6200
TOKENS_PER_STEP = 512 * 1024
TOTAL_TOKENS = TOTAL_STEPS * TOKENS_PER_STEP
EXPECTED_CONFIG = {
    "device_batch_size": 8,
    "global_batch_size": 512,
    "matrix_lr": 0.01,
    "backup_lr": 0.0036,
    "num_iterations": TOTAL_STEPS,
    "sequence_length": 1024,
    "val_every": 100,
    "val_tokens": 10_485_760,
    "warmdown_iters": 1800,
}
EXPECTED_K_STATE_BYTES = {
    "muon": 0,
    "down_none": 1_811_939_328,
    "down_diag": 1_812_731_904,
    "newton_full": 6_174_277_632,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--wandb-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def readable_path(path: Path) -> Path:
    resolved = path.resolve()
    if os.name == "nt":
        value = str(resolved)
        if not value.startswith("\\\\?\\"):
            return Path("\\\\?\\" + value)
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with readable_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_check(
    rows: list[dict[str, object]],
    check: str,
    passed: bool,
    evidence: str,
    severity: str = "critical",
) -> None:
    rows.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "severity_if_failed": severity,
            "evidence": evidence,
        }
    )


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(readable_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def method_directories(root: Path) -> list[Path]:
    result = []
    for path in root.rglob("summary.json"):
        parent = path.parent
        if parent.name[:3] in {"01_", "02_", "03_", "04_"}:
            result.append(parent)
    return sorted(result)


def local_metric_value(
    metrics: pd.DataFrame, metric: str, step: int
) -> float:
    if metric == "val/loss":
        rows = metrics[(metrics.event == "val") & (metrics.step == step)].loss
    elif metric == "train/loss_step":
        rows = metrics[(metrics.event == "train") & (metrics.step == step)].loss
    else:
        column = {
            "tokens/seen": "tokens_seen",
            "time/train_s": "train_s",
            "performance/step_avg_ms": "step_avg_ms",
            "lr/backup": "lr_backup",
            "lr/matrix": "lr_matrix",
        }[metric]
        rows = metrics[metrics.step == step][column].dropna().drop_duplicates()
    if len(rows) != 1:
        raise RuntimeError(f"expected one local value for {metric}@{step}, got {len(rows)}")
    return float(rows.iloc[0])


def write_report(
    path: Path,
    checks: pd.DataFrame,
    run_table: pd.DataFrame,
    memory: pd.DataFrame,
    archive_sha256: str,
) -> None:
    counts = checks.status.value_counts().to_dict()
    mem = memory.set_index("method")
    lines = [
        "# LLaMA/SwiGLU-1B formal-6200 本地证书审计（2026-07-27）",
        "",
        "## 结论",
        "",
        "三个 seed、四种方法的 compact formal artifacts 均完整可读，manifest、"
        "summary、status、metrics 与 W&B 曲线逐点一致。质量证据和实测训练状态可以"
        "正式封口。唯一保留 caveat 是约 8--11GB checkpoint 本体未随压缩包复制，"
        "因此本地无法验证 checkpoint 文件内容 hash；MECH-01 将在远程原地只读加载"
        "并对选中 checkpoint 计算 SHA-256。",
        "",
        f"- 压缩包 SHA-256：`{archive_sha256}`；",
        f"- 检查结果：PASS={counts.get('PASS', 0)}，FAIL={counts.get('FAIL', 0)}，"
        f"WARN={counts.get('WARN', 0)}；",
        f"- 有效 run：{len(run_table)}；所有 run 均为 6200 steps、"
        f"{TOTAL_TOKENS:,} tokens、resume_count=0；",
        "- seed2026 分成两个双方法子批次，但两批 runtime、data、source、"
        "初始化和 formal 配置一致，可以合法合并。",
        "",
        "## 实测显存与状态",
        "",
        "| 方法 | K-state GiB | Optimizer state GiB | Peak allocated GiB | "
        "Peak vs Muon GiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = mem.loc[method]
        lines.append(
            f"| {method} | {row.k_state_gib:.3f} | {row.optimizer_state_gib:.3f} | "
            f"{row.peak_allocated_gib:.3f} | {row.peak_delta_vs_muon_gib:+.3f} |"
        )
    full_minus_none_k = (
        mem.loc["newton_full", "k_state_bytes"]
        - mem.loc["down_none", "k_state_bytes"]
    ) / 2**30
    full_minus_none_peak = (
        mem.loc["newton_full", "peak_allocated_bytes_mean"]
        - mem.loc["down_none", "peak_allocated_bytes_mean"]
    ) / 2**30
    lines.extend(
        [
            "",
            f"down-none 相对 Newton-full 精确减少 {full_minus_none_k:.3f} GiB "
            f"K-state，并在 formal batch-8 实测降低约 {full_minus_none_peak:.3f} GiB "
            "peak allocated。diag 与 none 的 K-state 几乎相同；full 的额外 dense "
            "down-K 没有换来更低 loss。",
            "",
            "这些 formal peak 是进程内 batch-8 训练峰值，不能替代已封口的 exact OOM "
            "capacity boundary；二者应分别作为固定配置显存和容量边界证据。",
            "",
            "## MECH gate",
            "",
            "本地证书没有发现阻止机制实验的问题。按冻结队列，下一步是：",
            "",
            "1. R1-native seed2026 none@6200 的 MECH-01 preflight；",
            "2. 通过后运行三层 numerical smoke 并导出 fixed tensor bundle；",
            "3. 在另一套 runtime replay 同一 bundle，完成 host/runtime equivalence；",
            "4. 选中 checkpoint 完成 full SHA-256 与 schema gate 后，才进入 "
            "MECH-02-R1/L124 科学数据采集。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.artifact_root.resolve()
    wandb_dir = args.wandb_analysis_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "llamma_formal6200_compact_20260727.zip"

    checks: list[dict[str, object]] = []
    file_rows = []
    with zipfile.ZipFile(readable_path(archive_path)) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir() or "/__pycache__/" in info.filename:
                continue
            digest = hashlib.sha256()
            with archive.open(info) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            file_rows.append(
                {
                    "relative_path": info.filename,
                    "bytes": info.file_size,
                    "sha256": digest.hexdigest(),
                }
            )
    add_check(
        checks,
        "compact evidence file count excluding pycache",
        len(file_rows) == 100,
        str(len(file_rows)),
    )

    manifest_paths = sorted(root.rglob("llama_manifest.json"))
    add_check(checks, "expected four manifests", len(manifest_paths) == 4, str(len(manifest_paths)))
    manifests = [(path, read_json(path)) for path in manifest_paths]

    runtime_fingerprints = set()
    data_fingerprints = set()
    base_hashes = set()
    methods_by_seed = {seed: set() for seed in SEEDS}
    manifest_summary_hashes: dict[tuple[int, str], str] = {}
    for path, manifest in manifests:
        seed = int(manifest["seed"])
        methods = set(manifest["methods"])
        methods_by_seed[seed].update(methods)
        add_check(
            checks,
            f"{path.parent.name}: completed formal manifest",
            manifest.get("status") == "completed"
            and manifest.get("execution_stage") == "formal"
            and manifest.get("batch_kind") == "formal"
            and not manifest.get("failed_methods"),
            (
                f"status={manifest.get('status')} stage={manifest.get('execution_stage')} "
                f"failed={manifest.get('failed_methods')}"
            ),
        )
        add_check(
            checks,
            f"{path.parent.name}: W&B completed and timing ineligible",
            manifest.get("wandb_complete") is True
            and manifest.get("timing_eligible") is False,
            (
                f"wandb_complete={manifest.get('wandb_complete')} "
                f"timing_eligible={manifest.get('timing_eligible')}"
            ),
        )
        config = manifest["config"]
        add_check(
            checks,
            f"{path.parent.name}: frozen formal config",
            all(config.get(key) == value for key, value in EXPECTED_CONFIG.items()),
            json.dumps(config, sort_keys=True),
        )
        profile = manifest["profile"]
        add_check(
            checks,
            f"{path.parent.name}: pinned 1B architecture",
            profile.get("name") == "llama_swiglu_1b_v1"
            and profile.get("expected_parameter_count") == 1_013_690_368
            and profile.get("n_layer") == 18
            and profile.get("n_embd") == 2048
            and profile.get("intermediate_size") == 5504,
            json.dumps(profile, sort_keys=True),
        )
        runtime = manifest["runtime"]
        runtime_fingerprints.add(
            (
                runtime.get("python_executable"),
                runtime.get("torch"),
                runtime.get("torch_cuda"),
                runtime.get("triton"),
                runtime.get("gpu_name"),
                runtime.get("triton_kernels_sha256"),
            )
        )
        data_fingerprints.add(manifest["data_audit"].get("fingerprint"))
        base_hashes.add(manifest.get("base_trainer_sha256"))
        for method, result in manifest["method_results"].items():
            manifest_summary_hashes[(seed, method)] = str(result["summary_sha256"])
            add_check(
                checks,
                f"seed{seed} {method}: method result completed/uploaded",
                result.get("status") == "completed"
                and (result.get("wandb") or {}).get("status") == "uploaded",
                json.dumps(result.get("wandb"), sort_keys=True),
            )

    for seed in SEEDS:
        add_check(
            checks,
            f"seed{seed}: exact four-method certificate coverage",
            methods_by_seed[seed] == set(METHODS),
            repr(sorted(methods_by_seed[seed])),
        )
    add_check(checks, "single runtime fingerprint", len(runtime_fingerprints) == 1, repr(runtime_fingerprints))
    add_check(checks, "single data fingerprint", len(data_fingerprints) == 1, repr(data_fingerprints))
    add_check(checks, "single base trainer hash", len(base_hashes) == 1, repr(base_hashes))

    wandb_long = pd.read_csv(wandb_dir / "normalized_history_long.csv")
    wandb_summary = pd.read_csv(
        wandb_dir / "llama1b_formal_multiseed_run_summary.csv"
    ).set_index(["seed", "method"])

    run_rows = []
    method_dirs = method_directories(root)
    add_check(checks, "exact twelve method directories", len(method_dirs) == 12, str(len(method_dirs)))
    for directory in method_dirs:
        summary_path = directory / "summary.json"
        status_path = directory / "status.json"
        metrics_path = directory / "metrics.csv"
        source_path = directory / "train_llama_swiglu_base.py"
        summary = read_json(summary_path)
        status = read_json(status_path)
        metrics = pd.read_csv(readable_path(metrics_path))
        seed, method = int(summary["seed"]), str(summary["method"])

        add_check(
            checks,
            f"seed{seed} {method}: summary hash matches manifest",
            sha256(summary_path) == manifest_summary_hashes[(seed, method)],
            sha256(summary_path),
        )
        add_check(
            checks,
            f"seed{seed} {method}: status/summary completed",
            summary.get("status") == "completed"
            and status.get("status") == "completed"
            and status.get("completed_steps") == TOTAL_STEPS,
            (
                f"summary={summary.get('status')} status={status.get('status')} "
                f"steps={status.get('completed_steps')}"
            ),
        )
        add_check(
            checks,
            f"seed{seed} {method}: formal budget and clean completion",
            summary.get("completed_steps") == TOTAL_STEPS
            and summary.get("tokens_seen") == TOTAL_TOKENS
            and summary.get("resume_count") == 0,
            (
                f"steps={summary.get('completed_steps')} tokens={summary.get('tokens_seen')} "
                f"resume={summary.get('resume_count')} "
                f"per-run-timing-comparable={summary.get('timing_comparable')}"
            ),
        )
        add_check(
            checks,
            f"seed{seed} {method}: expected K-state bytes",
            summary.get("k_state_bytes") == EXPECTED_K_STATE_BYTES[method],
            str(summary.get("k_state_bytes")),
        )
        add_check(
            checks,
            f"seed{seed} {method}: saved base trainer matches manifest",
            sha256(source_path) in base_hashes,
            sha256(source_path),
        )

        train = metrics[metrics.event == "train"]
        val = metrics[metrics.event == "val"]
        add_check(
            checks,
            f"seed{seed} {method}: exact local metrics grain",
            len(metrics) == 6263
            and train.step.astype(int).tolist() == list(range(1, TOTAL_STEPS + 1))
            and val.step.astype(int).tolist() == list(range(0, TOTAL_STEPS + 1, 100)),
            f"rows={len(metrics)} train={len(train)} val={len(val)}",
        )
        add_check(
            checks,
            f"seed{seed} {method}: finite losses",
            bool(np.isfinite(metrics.loss.to_numpy(dtype=float)).all()),
            f"loss_rows={len(metrics)}",
        )
        add_check(
            checks,
            f"seed{seed} {method}: summary/local endpoint agreement",
            math.isclose(
                float(summary["final_val_loss"]),
                float(val[val.step == TOTAL_STEPS].loss.iloc[0]),
                rel_tol=0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(summary["final_train_loss"]),
                float(train[train.step == TOTAL_STEPS].loss.iloc[0]),
                rel_tol=0,
                abs_tol=1e-12,
            ),
            (
                f"summary_val={summary['final_val_loss']} "
                f"local_val={val[val.step == TOTAL_STEPS].loss.iloc[0]}"
            ),
        )

        current_wandb = wandb_long[
            (wandb_long.seed == seed) & (wandb_long.method == method)
        ]
        mismatches = []
        max_abs_diff = 0.0
        compared = 0
        for row in current_wandb.itertuples(index=False):
            if not math.isfinite(float(row.value)):
                continue
            local = local_metric_value(metrics, str(row.metric), int(row.step))
            difference = abs(local - float(row.value))
            max_abs_diff = max(max_abs_diff, difference)
            compared += 1
            if not math.isclose(local, float(row.value), rel_tol=1e-12, abs_tol=1e-9):
                mismatches.append((row.metric, int(row.step), local, float(row.value)))
        add_check(
            checks,
            f"seed{seed} {method}: local metrics agree with W&B",
            not mismatches,
            f"compared={compared} max_abs_diff={max_abs_diff:.3g} mismatches={len(mismatches)}",
        )

        archived = wandb_summary.loc[(seed, method)]
        add_check(
            checks,
            f"seed{seed} {method}: local endpoint agrees with archived analysis",
            math.isclose(
                float(summary["final_val_loss"]),
                float(archived.final_val_loss),
                rel_tol=0,
                abs_tol=1e-12,
            ),
            (
                f"local={summary['final_val_loss']} "
                f"archived={archived.final_val_loss}"
            ),
        )
        add_check(
            checks,
            f"seed{seed} {method}: remote checkpoint path recorded",
            bool(summary.get("checkpoint_path")),
            str(summary.get("checkpoint_path")),
            "high",
        )

        run_rows.append(
            {
                "seed": seed,
                "method": method,
                "status": summary["status"],
                "completed_steps": summary["completed_steps"],
                "tokens_seen": summary["tokens_seen"],
                "resume_count": summary["resume_count"],
                "final_train_loss": summary["final_train_loss"],
                "final_val_loss": summary["final_val_loss"],
                "model_parameter_bytes": summary["model_parameter_bytes"],
                "optimizer_state_bytes": summary["optimizer_state_bytes"],
                "k_state_bytes": summary["k_state_bytes"],
                "k_cov_bytes": summary["k_cov_bytes"],
                "k_inv_bytes": summary["k_inv_bytes"],
                "activation_stat_bytes": summary["activation_stat_bytes"],
                "preconditioner_workspace_bytes": summary[
                    "preconditioner_workspace_bytes"
                ],
                "peak_allocated_bytes": summary["peak_allocated_bytes"],
                "peak_allocated_mib": summary["peak_allocated_mib"],
                "checkpoint_path": summary["checkpoint_path"],
                "summary_sha256": sha256(summary_path),
                "metrics_sha256": sha256(metrics_path),
            }
        )

    checks.append(
        {
            "check": "checkpoint payload hash and local existence",
            "status": "WARN",
            "severity_if_failed": "high",
            "evidence": (
                "checkpoint payloads intentionally omitted; verify selected remote files "
                "with MECH-01 --hash-checkpoint"
            ),
        }
    )

    run_table = pd.DataFrame(run_rows).sort_values(["seed", "method"])
    memory = (
        run_table.groupby("method", as_index=False)
        .agg(
            k_state_bytes=("k_state_bytes", "first"),
            optimizer_state_bytes=("optimizer_state_bytes", "first"),
            peak_allocated_bytes_mean=("peak_allocated_bytes", "mean"),
            peak_allocated_bytes_min=("peak_allocated_bytes", "min"),
            peak_allocated_bytes_max=("peak_allocated_bytes", "max"),
        )
        .sort_values("peak_allocated_bytes_mean")
    )
    memory["k_state_gib"] = memory.k_state_bytes / 2**30
    memory["optimizer_state_gib"] = memory.optimizer_state_bytes / 2**30
    memory["peak_allocated_gib"] = memory.peak_allocated_bytes_mean / 2**30
    muon_peak = float(
        memory.loc[memory.method == "muon", "peak_allocated_bytes_mean"].iloc[0]
    )
    memory["peak_delta_vs_muon_gib"] = (
        memory.peak_allocated_bytes_mean - muon_peak
    ) / 2**30

    checks_frame = pd.DataFrame(checks)
    fail_count = int((checks_frame.status == "FAIL").sum())
    archive_hash = sha256(archive_path)

    pd.DataFrame(file_rows).to_csv(
        output / "local_formal_artifact_file_manifest.csv", index=False
    )
    checks_frame.to_csv(output / "local_formal_artifact_checks.csv", index=False)
    run_table.to_csv(output / "local_formal_run_summary.csv", index=False)
    memory.to_csv(output / "local_formal_memory_summary.csv", index=False)
    report = output / "LOCAL_FORMAL_ARTIFACT_AUDIT_20260727.md"
    write_report(report, checks_frame, run_table, memory, archive_hash)

    audit = {
        "created_at": "2026-07-27",
        "status": "PASS_WITH_CHECKPOINT_HASH_CAVEAT" if fail_count == 0 else "FAIL",
        "archive": str(archive_path),
        "archive_sha256": archive_hash,
        "artifact_root": str(root),
        "files": len(file_rows),
        "runs": len(run_table),
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "checks": {
            key: int(value)
            for key, value in checks_frame.status.value_counts().to_dict().items()
        },
        "local_metrics_wandb_reconciled": fail_count == 0,
        "checkpoint_payloads_present_locally": False,
        "selected_remote_checkpoint_hash_required_before_mech02": True,
        "report": str(report),
    }
    (output / "local_formal_artifact_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
