#!/usr/bin/env python3
"""Build a frozen, CPU-only value decision for the LLaMA-1B 10B candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
CONTRACT = HERE / "value_assessment_contract.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def dump_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_inputs(contract: dict[str, Any], workspace: Path) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    paths: dict[str, Path] = {}
    audit: list[dict[str, Any]] = []
    for item in contract["inputs"]:
        raw_path = str(item["path"])
        results_prefix = "${SNM_RESULTS_ROOT}/"
        if raw_path.startswith(results_prefix):
            results_root = Path(
                os.environ.get("SNM_RESULTS_ROOT", str(workspace / "runs"))
            ).expanduser()
            path = results_root / raw_path[len(results_prefix) :]
        else:
            expanded = Path(os.path.expandvars(raw_path)).expanduser()
            path = expanded if expanded.is_absolute() else workspace / expanded
        observed = sha256_file(path) if path.is_file() else ""
        passed = path.is_file() and observed == item["sha256"]
        audit.append(
            {
                "input_id": item["id"],
                "path": str(path),
                "expected_sha256": item["sha256"],
                "observed_sha256": observed,
                "passed": passed,
            }
        )
        if not passed:
            raise RuntimeError(f"input audit failed: {item['id']} ({path})")
        paths[item["id"]] = path
    return paths, audit


def build_decision(contract: dict[str, Any], workspace: Path) -> dict[str, Any]:
    paths, audit = audit_inputs(contract, workspace)
    feasibility_manifest = load_json(paths["feasibility_manifest"])
    feasibility = load_json(paths["feasibility_report"])
    llama = load_json(paths["llama_multiseed_manifest"])
    unified = load_json(paths["final_unified_manifest"])
    mechanism = load_json(paths["mechanism_closure_manifest"])

    checks = {
        "assessment_non_launchable": contract["launch_authorized"] is False,
        "feasibility_plan_passed": feasibility_manifest["passed"] is True,
        "technical_prerequisites_not_passed": feasibility_manifest["technical_prerequisites_passed"] is False,
        "remote_data_not_audited": feasibility_manifest["data_audit_passed"] is None,
        "llama_fixed_recipe_replication_complete": llama["fixed_recipe_multiseed_replication_complete"] is True,
        "llama_muon_wins_3_of_3": llama["primary_result"]["muon_final_wins"] == 3,
        "unified_claim_eligible": unified["claim_eligible"] is True,
        "unified_pipeline_checks": all(unified["checks"].values()),
        "mechanism_line_closed": mechanism["passed"] is True and mechanism["mechanism_line_closed"] is True,
        "geo01c_not_authorized": mechanism["geo01c_authorized"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"scientific lineage check failed: {checks}")

    gates = contract["gate_assessment"]
    passed_gates = sum(row["status"] == "pass" for row in gates)
    all_required_passed = all(row["status"] == "pass" for row in gates)
    if all_required_passed:
        raise RuntimeError("assessment contract unexpectedly authorizes every gate")

    budget = feasibility["budget"]
    one_seed_raw_hours = float(budget["raw_training_gpu_hours"])
    decision = {
        "schema_version": contract["schema_version"],
        "assessment_date": contract["assessment_date"],
        "status": contract["status"],
        "decision": contract["decision"],
        "launch_authorized": False,
        "input_audit": audit,
        "lineage_checks": checks,
        "gate_summary": {
            "all_required_gates": True,
            "passed": passed_gates,
            "total": len(gates),
            "all_required_passed": all_required_passed,
            "rows": gates,
        },
        "cost": {
            "four_method_aggregate_training_tokens": budget["aggregate_training_tokens"],
            "one_seed_raw_h100_hours": one_seed_raw_hours,
            "two_h100_wall_days_with_overhead": budget["gpu_scenarios"]["2"]["wall_seconds"] / 86400.0,
            "four_h100_wall_hours_with_overhead": budget["gpu_scenarios"]["4"]["wall_seconds"] / 3600.0,
            "validation_tokens_per_method": budget["validation_tokens_per_method"],
            "retained_checkpoint_gb": budget["retained_checkpoint_bytes"] / 1e9,
            "recommended_free_disk_gb": budget["recommended_minimum_free_disk_bytes"] / 1e9,
            "three_seed_raw_h100_hours_if_confirmation_triggered": one_seed_raw_hours * 3,
            "three_seed_two_h100_wall_days_estimate": budget["gpu_scenarios"]["2"]["wall_seconds"] / 86400.0 * 3,
        },
        "scientific_value": {
            "unique_question": "Whether the Muon-over-Newton-family ordering on the same LLaMA-1B architecture persists from 3.25B to 6.97B and approximately 10B tokens.",
            "does_not_answer": [
                "the causal origin of the refresh harm",
                "a pure architecture effect",
                "a population-level long-horizon ranking from one seed",
                "a schedule-matched continuation of the accepted 6200-step run",
            ],
            "scenario_impacts": [
                {
                    "result": "Muon remains ahead at 6.97B and 10B",
                    "paper_impact": "Strengthens an existing LLaMA regime limitation; central claim and abstract remain unchanged.",
                    "follow_up": "No extra seed is automatically justified unless the long-horizon ranking becomes a central claim."
                },
                {
                    "result": "Selective re-overtakes by 6.97B or 10B",
                    "paper_impact": "Materially changes the LLaMA boundary into a training-stage result.",
                    "follow_up": "Requires preregistered seeds 2024 and 2025 before a population-level claim."
                },
                {
                    "result": "Differences are small or unstable",
                    "paper_impact": "Adds an inconclusive single-seed appendix result.",
                    "follow_up": "Do not expand unless a frozen practical-margin seed gate passes."
                }
            ],
            "current_submission_priority": "interesting_but_not_a_current_acceptance_critical_gap",
        },
        "recommendation": {
            "now": "Do not launch or implement a launch controller now.",
            "reopen_rule": contract["reopen_rule"],
            "if_reopened_first_steps": [
                "Run the read-only 101+ shard header/inventory audit.",
                "Freeze one from-scratch 19073-step LR policy and state that step 6200 is not schedule-matched to the accepted run.",
                "Implement a new no-wrap loader, wrap_count certificate, forced milestones, and exact-resume controller under a new contract.",
                "Retain all four methods and the conditional two-seed confirmation gate."
            ],
        },
    }
    return decision


def markdown(decision: dict[str, Any]) -> str:
    cost = decision["cost"]
    gates = decision["gate_summary"]
    lines = [
        "# LLaMA-1B 约 10B-token 价值评估（2026-08-05）",
        "",
        "## 决策",
        "",
        "**当前不启动，也不编写可启动的远程 controller；保留为 reviewer-triggered contingency。**",
        "",
        f"冻结 gate 要求全部通过；当前仅 `{gates['passed']}/{gates['total']}` 通过。GPU 空闲不构成重新打开 gate 的理由。",
        "",
        "## 它真正能回答什么",
        "",
        "它能在同一 LLaMA-1B 架构上检验：3.25B tokens 后 Muon 的领先是否持续到约 6.97B 和 10B。这个问题有科学价值，尤其能直接回应‘当前 1B 是否训练不足’。",
        "",
        "但它不能解释 refresh harm 的因果来源，不能单独识别纯架构效应，也不能用一个 seed 建立总体长程排名。由于 accepted 6200-step recipe 已在末端降到零 LR，候选 19073-step run 必须使用新的 from-scratch 长日程，因此其 step 6200 不是原 formal 的 schedule-matched replication。",
        "",
        "## 为什么当前不值得启动",
        "",
        "1. 最终统一分析已是 claim-eligible；中心主张本来就限定为 environment/regime-dependent，不依赖 10B 才能成立。",
        "2. 275M GPT 在 2.42 tokens/parameter（低于 LLaMA-1B 的 3.21）仍支持低状态 route，已经否定‘只是 token 太少’这一单因素解释。",
        "3. 机制收尾后的主要缺口是缺少简单、跨 origin 的定量解释；10B full training 不会修复这个机制缺口。",
        "4. 单 seed 最有价值的结果——Selective 后期重新反超——反而会立即触发两个额外 seed，不能在一个 screening 后直接写成正式结论。",
        "5. 当前训练栈和数据仍有七个 hard blockers，首先需要新 LR 合同、101+ unique shards 审计、no-wrap loader、wrap_count、forced milestones 与 exact-resume controller。",
        "",
        "## 成本",
        "",
        f"- 四方法累计训练 tokens：`{cost['four_method_aggregate_training_tokens']:,}`。",
        f"- 单 seed：`{cost['one_seed_raw_h100_hours']:.2f}` raw H100-hours；两张 H100 约 `{cost['two_h100_wall_days_with_overhead']:.2f}` 天。",
        f"- 每方法 validation tokens：`{cost['validation_tokens_per_method']:,}`。",
        f"- 三个 milestone checkpoint 约 `{cost['retained_checkpoint_gb']:.2f}` GB；建议至少 `{cost['recommended_free_disk_gb']:.0f}` GB 空闲。",
        f"- 若 screening 触发三 seed 确认，总量约 `{cost['three_seed_raw_h100_hours_if_confirmation_triggered']:.2f}` raw H100-hours、两卡约 `{cost['three_seed_two_h100_wall_days_estimate']:.2f}` 天。",
        "",
        "## Gate 逐项判定",
        "",
        "| Gate | 状态 | 判定依据 |",
        "|---|---|---|",
    ]
    for row in gates["rows"]:
        lines.append(f"| `{row['gate_id']}` | **{row['status']}** | {row['rationale']} |")
    lines.extend(
        [
            "",
            "## 唯一重开条件",
            "",
            decision["recommendation"]["reopen_rule"],
            "",
            "若该条件真的出现，先做远程 header-only 数据审计和 LR red-team；它们通过后再写独立 launch contract。当前产物不授权训练。",
            "",
        ]
    )
    return "\n".join(lines)


def build(output: Path, workspace: Path) -> None:
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    contract = load_json(CONTRACT)
    decision = build_decision(contract, workspace)
    output.mkdir(parents=True)
    dump_json(output / "decision.json", decision)
    (output / "LLAMA1B_10B_VALUE_ASSESSMENT.md").write_text(markdown(decision), encoding="utf-8")
    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.is_file():
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    dump_json(
        output / "value_assessment_manifest.json",
        {
            "schema_version": "llama1b_10b_value_assessment_manifest_v1",
            "status": "completed",
            "passed": True,
            "decision": decision["decision"],
            "launch_authorized": False,
            "contract_sha256": sha256_file(CONTRACT),
            "artifacts": artifacts,
        },
    )


def validate(output: Path) -> dict[str, Any]:
    manifest = load_json(output / "value_assessment_manifest.json")
    if manifest["passed"] is not True or manifest["launch_authorized"] is not False:
        raise RuntimeError("assessment manifest status")
    for name, item in manifest["artifacts"].items():
        path = output / name
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"artifact validation failed: {name}")
    return {"passed": True, "decision": manifest["decision"], "artifact_count": len(manifest["artifacts"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "build", "validate"))
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    contract = load_json(CONTRACT)
    if args.mode == "check":
        result = build_decision(contract, args.workspace.resolve())
        print(json.dumps({"passed": True, "decision": result["decision"], "gates": result["gate_summary"]}, indent=2))
        return
    if args.output_dir is None:
        parser.error(f"{args.mode} requires --output-dir")
    if args.mode == "build":
        build(args.output_dir.resolve(), args.workspace.resolve())
    print(json.dumps(validate(args.output_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
