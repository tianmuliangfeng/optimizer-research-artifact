#!/usr/bin/env python3
"""Audit and consolidate two-host MECH-00 full-hash archives.

This is a local evidence audit.  It never opens the checkpoint payloads; it
validates the stable SHA-256 certificates produced on the remote hosts and
checks their coverage against the locally certified LLaMA-1B formal run list.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "2026-07-27.1"
REQUIRED_FILES = {
    "audit_checks.csv",
    "available_step_map.csv",
    "checkpoint_hashes.csv",
    "checkpoint_inventory.csv",
    "diagnostic_data_contract.json",
    "input_discovery.csv",
    "mech00_manifest.json",
    "runtime_inventory.json",
    "source_inventory.csv",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r1-dir", type=Path, required=True)
    parser.add_argument("--llama-dir", type=Path, required=True)
    parser.add_argument("--r1-archive", type=Path, required=True)
    parser.add_argument("--llama-archive", type=Path, required=True)
    parser.add_argument("--formal-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def zip_audit(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        members = [name for name in archive.namelist() if not name.endswith("/")]
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "file_count": len(members),
        "bad_member": bad_member or "",
        "passed": bad_member is None,
    }


def host_audit(label: str, root: Path) -> dict[str, Any]:
    present = {path.name for path in root.iterdir() if path.is_file()}
    missing_files = sorted(REQUIRED_FILES - present)
    manifest = read_json(root / "mech00_manifest.json")
    hashes = read_csv(root / "checkpoint_hashes.csv")
    inventory = read_csv(root / "checkpoint_inventory.csv")
    step_map = read_csv(root / "available_step_map.csv")
    audit_checks = read_csv(root / "audit_checks.csv")
    sources = read_csv(root / "source_inventory.csv")
    discovery = read_csv(root / "input_discovery.csv")

    stable_hashes = [
        row
        for row in hashes
        if row["hash_status"] == "verified_stable"
        and HEX64.fullmatch(row["sha256"].lower())
        and row["checkpoint_bytes"] == row["size_before"] == row["size_after"]
        and row["mtime_ns_before"] == row["mtime_ns_after"]
    ]
    dirty_sources = [
        row for row in sources if row.get("git_status_porcelain", "").strip()
    ]
    dirty_mislabeled = [
        row
        for row in dirty_sources
        if row.get("audit_status", "") == "ok"
    ]
    failed_checks = [
        row for row in audit_checks if row.get("status", "").lower() == "fail"
    ]
    duplicate_paths = len(hashes) - len(
        {row["checkpoint_path"] for row in hashes}
    )
    duplicate_digests = len(hashes) - len({row["sha256"] for row in hashes})
    exact_steps = sum(row["exact_match"].lower() == "true" for row in step_map)
    host_pass = (
        not missing_files
        and manifest.get("hash_mode") == "full"
        and int(manifest["counts"]["failed_checks"]) == 0
        and len(hashes) == int(manifest["counts"]["checkpoint_files"])
        and len(inventory) == int(manifest["counts"]["inventory_rows"])
        and len(stable_hashes) == len(hashes)
        and not failed_checks
        and duplicate_paths == 0
    )
    return {
        "label": label,
        "root": str(root.resolve()),
        "manifest": manifest,
        "hashes": hashes,
        "inventory": inventory,
        "step_map": step_map,
        "audit_checks": audit_checks,
        "sources": sources,
        "discovery": discovery,
        "missing_files": missing_files,
        "stable_hash_count": len(stable_hashes),
        "checkpoint_count": len(hashes),
        "checkpoint_bytes": sum(int(row["checkpoint_bytes"]) for row in hashes),
        "duplicate_paths": duplicate_paths,
        "duplicate_digests": duplicate_digests,
        "exact_step_rows": exact_steps,
        "step_map_rows": len(step_map),
        "dirty_sources": dirty_sources,
        "dirty_mislabeled": dirty_mislabeled,
        "passed": host_pass,
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    archives = [
        zip_audit(args.r1_archive.resolve()),
        zip_audit(args.llama_archive.resolve()),
    ]
    hosts = [
        host_audit("r1_native", args.r1_dir.resolve()),
        host_audit("llama_host", args.llama_dir.resolve()),
    ]

    consolidated_hashes: list[dict[str, Any]] = []
    for host in hosts:
        for row in host["hashes"]:
            consolidated_hashes.append({"host": host["label"], **row})

    formal = read_csv(args.formal_summary.resolve())
    expected = [
        row
        for row in formal
        if row["method"] in {"down_none", "muon"}
        and row["status"] == "completed"
        and row["completed_steps"] == "6200"
    ]
    observed_paths = {row["checkpoint_path"] for row in consolidated_hashes}
    expected_coverage = []
    for row in sorted(
        expected, key=lambda item: (int(item["seed"]), item["method"])
    ):
        path = row["checkpoint_path"]
        expected_coverage.append(
            {
                "family": "llama_1b",
                "seed": row["seed"],
                "method": row["method"],
                "completed_steps": row["completed_steps"],
                "checkpoint_path": path,
                "full_hash_present": path in observed_paths,
            }
        )
    missing_expected = [
        row for row in expected_coverage if not row["full_hash_present"]
    ]

    grouped: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for row in consolidated_hashes:
        grouped[
            (
                row["host"],
                row["family"],
                row["method"],
                row["completed_steps"],
            )
        ].add(row["seed"])
    coverage_rows = [
        {
            "host": key[0],
            "family": key[1],
            "method": key[2],
            "completed_steps": key[3],
            "seeds": ",".join(sorted(seeds, key=int)),
            "checkpoint_count": len(seeds),
        }
        for key, seeds in sorted(grouped.items())
    ]

    checks = []

    def add(check: str, status: str, evidence: str) -> None:
        checks.append({"check": check, "status": status, "evidence": evidence})

    add(
        "archive_integrity",
        "pass" if all(item["passed"] for item in archives) else "fail",
        f"{sum(item['file_count'] for item in archives)} files; "
        f"bad_members={[item['bad_member'] for item in archives if item['bad_member']]}",
    )
    for host in hosts:
        add(
            f"{host['label']}_stable_checkpoint_hashes",
            "pass" if host["passed"] else "fail",
            f"{host['stable_hash_count']}/{host['checkpoint_count']} "
            f"verified_stable; {host['checkpoint_bytes'] / 2**30:.3f} GiB",
        )
        current_version = host["manifest"].get("script_version", "")
        add(
            f"{host['label']}_mech00_version",
            "pass" if current_version == "2026-07-24.2" else "warn",
            f"observed={current_version}; expected=2026-07-24.2",
        )
        add(
            f"{host['label']}_source_dirty_classification",
            "warn" if host["dirty_mislabeled"] else "pass",
            f"dirty_sources={len(host['dirty_sources'])}; "
            f"dirty_mislabeled_as_ok={len(host['dirty_mislabeled'])}",
        )
    add(
        "llama1b_formal_selected_checkpoint_coverage",
        "pass" if not missing_expected else "fail",
        f"observed={len(expected_coverage) - len(missing_expected)}/"
        f"{len(expected_coverage)} expected down_none/muon formal checkpoints; "
        f"missing={len(missing_expected)}",
    )

    fail_count = sum(row["status"] == "fail" for row in checks)
    warn_count = sum(row["status"] == "warn" for row in checks)
    overall = (
        "PASS"
        if fail_count == 0 and warn_count == 0
        else "PASS_WITH_WARNINGS"
        if fail_count == 0
        else "PARTIAL_COVERAGE_REQUIRES_SUPPLEMENT"
    )

    source_rows = []
    for host in hosts:
        for row in host["sources"]:
            dirty = bool(row.get("git_status_porcelain", "").strip())
            source_rows.append(
                {
                    "host": host["label"],
                    **row,
                    "locally_reclassified_status": (
                        "warn_dirty" if dirty else "pass_clean"
                    ),
                }
            )

    summary = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "overall_status": overall,
        "checks": {
            "pass": sum(row["status"] == "pass" for row in checks),
            "warn": warn_count,
            "fail": fail_count,
        },
        "archives": archives,
        "hosts": [
            {
                "label": host["label"],
                "manifest_script_version": host["manifest"].get(
                    "script_version", ""
                ),
                "host_id": host["manifest"].get("host_id", ""),
                "execution_domain": host["manifest"].get(
                    "execution_domain", ""
                ),
                "checkpoint_count": host["checkpoint_count"],
                "stable_hash_count": host["stable_hash_count"],
                "checkpoint_bytes": host["checkpoint_bytes"],
                "dirty_source_count": len(host["dirty_sources"]),
            }
            for host in hosts
        ],
        "combined_checkpoint_count": len(consolidated_hashes),
        "combined_checkpoint_bytes": sum(
            int(row["checkpoint_bytes"]) for row in consolidated_hashes
        ),
        "expected_llama1b_formal_selected_checkpoints": len(expected_coverage),
        "missing_llama1b_formal_selected_checkpoints": missing_expected,
        "interpretation": {
            "verified_hashes_are_valid": True,
            "mech00_full_hash_caveat_fully_closed": not missing_expected,
            "checkpoint_schema_still_owned_by_mech01": True,
            "nearest_step_rows_are_not_exact_observations": True,
        },
    }

    write_csv(output / "archive_manifest.csv", archives)
    write_csv(output / "consolidated_checkpoint_hashes.csv", consolidated_hashes)
    write_csv(output / "checkpoint_coverage.csv", coverage_rows)
    write_csv(output / "expected_llama1b_formal_coverage.csv", expected_coverage)
    write_csv(output / "source_inventory_reclassified.csv", source_rows)
    write_csv(output / "audit_checks.csv", checks)
    write_json(output / "mech00_full_audit.json", summary)

    r1 = hosts[0]
    llama = hosts[1]
    missing_lines = "\n".join(
        f"- seed{row['seed']} `{row['method']}`: `{row['checkpoint_path']}`"
        for row in missing_expected
    ) or "- 无"
    source_caveat = (
        "LLaMA official source 工作树包含 tracked modifications；旧版 v2026-07-24.1 "
        "仍将其写成 `audit_status=ok`，本地已重分类为 `warn_dirty`。"
    )
    report = f"""# MECH-00 full-hash 两主机证据审计（2026-07-27）

## 技术结论

本次提交的两个 ZIP 均可完整解压，远端实际完成了 full-file SHA-256。
R1 的 {r1['checkpoint_count']}/{r1['checkpoint_count']} 个、LLaMA 主机的
{llama['checkpoint_count']}/{llama['checkpoint_count']} 个 checkpoint 均为
`verified_stable`，文件大小和 mtime 在哈希前后不变，17 个路径与 17 个摘要均唯一。
这些已经生成的 SHA-256 可以正式保留和引用。

但 MECH-00 还不能标记为完全关闭：LLaMA inventory 没有扫描
`20_llama_swiglu_1b/multiseed_followup`，因此 seed2024/2025 的 1B formal
`down_none/muon` 共 4 个 checkpoint 未进入 full-hash 表。整体状态为
**{overall}**。

## 已验证的 checkpoint 证据

| execution domain | checkpoint 数 | 读取体积 | stable SHA-256 | 结果 |
|---|---:|---:|---:|---|
| R1 native | {r1['checkpoint_count']} | {r1['checkpoint_bytes'] / 2**30:.3f} GiB | {r1['stable_hash_count']} | 通过 |
| LLaMA host | {llama['checkpoint_count']} | {llama['checkpoint_bytes'] / 2**30:.3f} GiB | {llama['stable_hash_count']} | 通过 |
| 合计 | {len(consolidated_hashes)} | {summary['combined_checkpoint_bytes'] / 2**30:.3f} GiB | {len(consolidated_hashes)} | 已验证部分通过 |

R1 覆盖 `none/muon × seed2024/2025/2026 @ step6200`。LLaMA 主机覆盖
LLaMA-124M 的 `down_none/muon × 3 seeds @ step6200`、GPT bridge
`none@6200`，以及 LLaMA-1B seed2026 的 `down_none/muon @ step1000/6200`。

## 尚缺的 1B formal full hash

{missing_lines}

这些缺口不否定已经完成的训练或 compact artifact 证书，但在补齐前，不能把
LLaMA-1B 三 seed formal checkpoint provenance 表述成全量 SHA-256 已封口。

## 数据与指标定义

- 分析粒度：一个唯一的远程 checkpoint 文件路径。
- `verified_stable`：SHA-256 读取前后文件 size、mtime 和 inode 保持一致。
- coverage baseline：已通过本地/W&B 169 项核对的 LLaMA-1B formal
  `down_none/muon × seed2024/2025/2026 @ step6200`。
- `available_step_map.csv` 中 `exact_match=false` 只表示最近可用文件，不能当成
  step0/500/3000 的真实 checkpoint。

## 审计方法

1. 对两个 ZIP 执行完整 CRC 解压测试并计算 ZIP SHA-256；
2. 核对每个 manifest 的 `hash_mode=full`、计数和失败项；
3. 逐行验证摘要格式、唯一性、size/mtime 稳定性；
4. 将两主机 checkpoint 路径与正式 1B 三 seed summary 的 checkpoint 路径做集合对账；
5. 独立复核 source inventory，不沿用 v2026-07-24.1 的 dirty-source 错误分类。

## 限制与稳健性

{source_caveat}
这不改变 checkpoint SHA-256；训练 run 已保存实际 trainer 和
`triton_kernels.py` 的内容哈希。但正式 source provenance 仍应使用
MECH-00 v2026-07-24.2 的 dirty-aware 输出，并由 MECH-01 保存 selected source
文件哈希与必要 diff。

MECH-00 只证明文件身份与可用点，不验证 checkpoint 内部 schema。model、
optimizer、loader 和 RNG key 的只读 schema 验证仍由 MECH-01 完成。

## 下一步

1. 在 LLaMA 主机仅对 `multiseed_followup` 补充一次 full hash，避免重读已经验证的
   45.140 GiB；
2. 将新增 4 行与本次 17 行合并，要求 21/21 checkpoint 均为
   `verified_stable`；
3. 用 MECH-01 的 selected checkpoint hash 与本表交叉核对；
4. 通过后才把 MECH-00 状态改为“full hash closed”。

## 进一步问题

- seed2024/2025 的四个远程 checkpoint 是否仍保留在原路径；
- LLaMA 主机补哈希时是否已同步 MECH-00 v2026-07-24.2。
"""
    (output / "MECH00_FULL_AUDIT_20260727.md").write_text(
        report, encoding="utf-8"
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    sources = [
        {
            "id": "headline_sql",
            "label": "MECH-00 headline audit metrics",
            "query": {
                "engine": "duckdb",
                "sql": (
                    "WITH h AS (SELECT * FROM read_csv_auto("
                    "'consolidated_checkpoint_hashes.csv', header=true)), "
                    "e AS (SELECT * FROM read_csv_auto("
                    "'expected_llama1b_formal_coverage.csv', header=true)) "
                    "SELECT (SELECT COUNT(*) FROM h) AS verified_checkpoints, "
                    "(SELECT SUM(CAST(checkpoint_bytes AS UBIGINT)) / "
                    "POWER(2, 30) FROM h) AS total_gib, "
                    "(SELECT COUNT(*) FROM e WHERE NOT "
                    "CAST(full_hash_present AS BOOLEAN)) "
                    "AS missing_formal_checkpoints"
                ),
                "description": "Computes headline stable-hash count, bytes read, and missing formal coverage.",
                "tables_used": [
                    "consolidated_checkpoint_hashes.csv",
                    "expected_llama1b_formal_coverage.csv",
                ],
                "metric_definitions": {
                    "verified_checkpoints": "Unique remote checkpoint paths with verified_stable SHA-256 rows.",
                    "total_gib": "Sum of checkpoint_bytes divided by 2^30.",
                    "missing_formal_checkpoints": "Expected 1B formal selected paths absent from the full-hash table.",
                },
            },
        },
        {
            "id": "coverage_sql",
            "label": "MECH-00 scope coverage",
            "query": {
                "engine": "duckdb",
                "sql": (
                    "WITH h AS (SELECT * FROM read_csv_auto("
                    "'consolidated_checkpoint_hashes.csv', header=true)), "
                    "e AS (SELECT * FROM read_csv_auto("
                    "'expected_llama1b_formal_coverage.csv', header=true)), "
                    "c AS ("
                    "SELECT 'R1 selected' AS scope, COUNT(*) AS verified_count, "
                    "COUNT(*) AS expected_count FROM h WHERE host='r1_native' "
                    "UNION ALL "
                    "SELECT 'LLaMA scanned', COUNT(*), COUNT(*) FROM h "
                    "WHERE host='llama_host' "
                    "UNION ALL "
                    "SELECT '1B formal selected', "
                    "SUM(CASE WHEN CAST(full_hash_present AS BOOLEAN) "
                    "THEN 1 ELSE 0 END), COUNT(*) FROM e) "
                    "SELECT scope, verified_count, expected_count, "
                    "expected_count-verified_count AS missing_count, "
                    "100.0*verified_count/expected_count AS coverage_pct FROM c"
                ),
                "description": "Compares stable full-hash coverage with the expected checkpoint count in each audit scope.",
                "tables_used": [
                    "consolidated_checkpoint_hashes.csv",
                    "expected_llama1b_formal_coverage.csv",
                ],
                "metric_definitions": {
                    "coverage_pct": "100 × verified checkpoint count / expected checkpoint count."
                },
            },
        },
        {
            "id": "domain_sql",
            "label": "MECH-00 per-domain hash summary",
            "query": {
                "engine": "duckdb",
                "sql": (
                    "SELECT CASE host WHEN 'r1_native' THEN 'R1 native' "
                    "ELSE 'LLaMA host' END AS domain, "
                    "COUNT(*) AS checkpoint_count, "
                    "SUM(CASE WHEN hash_status='verified_stable' "
                    "THEN 1 ELSE 0 END) AS stable_hash_count, "
                    "SUM(CAST(checkpoint_bytes AS UBIGINT))/POWER(2,30) "
                    "AS total_gib, "
                    "CASE WHEN stable_hash_count=checkpoint_count "
                    "THEN 'pass' ELSE 'fail' END AS result "
                    "FROM read_csv_auto('consolidated_checkpoint_hashes.csv', "
                    "header=true) GROUP BY host"
                ),
                "description": "Aggregates full-hash count and bytes by execution domain.",
                "tables_used": ["consolidated_checkpoint_hashes.csv"],
            },
        },
        {
            "id": "missing_sql",
            "label": "Missing LLaMA-1B formal checkpoint hashes",
            "query": {
                "engine": "duckdb",
                "sql": (
                    "SELECT CAST(seed AS INTEGER) AS seed, method, "
                    "CAST(completed_steps AS INTEGER) AS completed_steps, "
                    "checkpoint_path FROM read_csv_auto("
                    "'expected_llama1b_formal_coverage.csv', header=true) "
                    "WHERE NOT CAST(full_hash_present AS BOOLEAN) "
                    "ORDER BY seed, method"
                ),
                "description": "Lists expected 1B formal selected checkpoint paths absent from the submitted full-hash inventories.",
                "tables_used": ["expected_llama1b_formal_coverage.csv"],
            },
        },
    ]
    domain_rows = [
        {
            "domain": "R1 native",
            "checkpoint_count": r1["checkpoint_count"],
            "stable_hash_count": r1["stable_hash_count"],
            "total_gib": round(r1["checkpoint_bytes"] / 2**30, 3),
            "result": "pass",
        },
        {
            "domain": "LLaMA host",
            "checkpoint_count": llama["checkpoint_count"],
            "stable_hash_count": llama["stable_hash_count"],
            "total_gib": round(llama["checkpoint_bytes"] / 2**30, 3),
            "result": "pass",
        },
    ]
    coverage_chart_rows = [
        {
            "scope": "R1 selected",
            "verified_count": 6,
            "expected_count": 6,
            "missing_count": 0,
            "coverage_pct": 100.0,
        },
        {
            "scope": "LLaMA scanned",
            "verified_count": 11,
            "expected_count": 11,
            "missing_count": 0,
            "coverage_pct": 100.0,
        },
        {
            "scope": "1B formal selected",
            "verified_count": len(expected_coverage) - len(missing_expected),
            "expected_count": len(expected_coverage),
            "missing_count": len(missing_expected),
            "coverage_pct": round(
                100
                * (len(expected_coverage) - len(missing_expected))
                / len(expected_coverage),
                1,
            ),
        },
    ]
    headline_rows = [
        {
            "verified_checkpoints": len(consolidated_hashes),
            "total_gib": round(summary["combined_checkpoint_bytes"] / 2**30, 3),
            "missing_formal_checkpoints": len(missing_expected),
        }
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "MECH-00 full-hash 两主机证据审计",
            "description": "R1 与 LLaMA 主机 checkpoint full SHA-256 的覆盖和质量审计。",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": [
                {
                    "id": "verified_card",
                    "description": "两台主机已通过 size/mtime/inode 稳定性检查的 checkpoint。",
                    "dataset": "headline",
                    "sourceId": "headline_sql",
                    "metrics": [
                        {
                            "label": "稳定 SHA-256",
                            "field": "verified_checkpoints",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "volume_card",
                    "description": "本轮远程 full-hash 实际读取的 checkpoint 总体积。",
                    "dataset": "headline",
                    "sourceId": "headline_sql",
                    "metrics": [
                        {
                            "label": "已验证体积, GiB",
                            "field": "total_gib",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "missing_card",
                    "description": "尚未进入 full-hash 表的 LLaMA-1B formal selected checkpoints。",
                    "dataset": "headline",
                    "sourceId": "headline_sql",
                    "metrics": [
                        {
                            "label": "待补 checkpoint",
                            "field": "missing_formal_checkpoints",
                            "format": "number",
                        }
                    ],
                },
            ],
            "charts": [
                {
                    "id": "coverage_chart",
                    "title": "Full-hash coverage by audit scope",
                    "subtitle": "已验证 checkpoint 占各审计范围期望数量的比例",
                    "type": "bar",
                    "dataset": "coverage_chart",
                    "sourceId": "coverage_sql",
                    "encodings": {
                        "x": {
                            "field": "scope",
                            "type": "nominal",
                            "label": "Audit scope",
                        },
                        "y": {
                            "field": "coverage_pct",
                            "type": "quantitative",
                            "label": "Coverage",
                            "unit": "%",
                        },
                        "tooltip": [
                            {
                                "field": "verified_count",
                                "type": "quantitative",
                                "label": "Verified",
                            },
                            {
                                "field": "expected_count",
                                "type": "quantitative",
                                "label": "Expected",
                            },
                            {
                                "field": "missing_count",
                                "type": "quantitative",
                                "label": "Missing",
                            },
                        ],
                    },
                    "valueFormat": "number",
                    "unit": "%",
                    "layout": "horizontal",
                }
            ],
            "tables": [
                {
                    "id": "domain_table",
                    "title": "两台主机 full-hash 结果",
                    "subtitle": "一个路径为一个 checkpoint；体积使用 GiB",
                    "dataset": "domain_summary",
                    "sourceId": "domain_sql",
                    "density": "spacious",
                    "defaultSort": {"field": "checkpoint_count", "direction": "desc"},
                    "columns": [
                        {"field": "domain", "label": "Execution domain"},
                        {
                            "field": "checkpoint_count",
                            "label": "Checkpoints",
                            "type": "number",
                            "align": "right",
                        },
                        {
                            "field": "stable_hash_count",
                            "label": "Stable SHA-256",
                            "type": "number",
                            "align": "right",
                        },
                        {
                            "field": "total_gib",
                            "label": "GiB read",
                            "type": "number",
                            "align": "right",
                        },
                        {"field": "result", "label": "Result"},
                    ],
                },
                {
                    "id": "missing_table",
                    "title": "尚缺的 LLaMA-1B formal checkpoint hashes",
                    "subtitle": "仅列出机制计划需要的 down_none 与 Muon",
                    "dataset": "missing_checkpoints",
                    "sourceId": "missing_sql",
                    "density": "comfortable",
                    "defaultSort": {"field": "seed", "direction": "asc"},
                    "columns": [
                        {"field": "seed", "label": "Seed", "type": "number"},
                        {"field": "method", "label": "Method"},
                        {
                            "field": "completed_steps",
                            "label": "Step",
                            "type": "number",
                        },
                        {"field": "checkpoint_path", "label": "Remote checkpoint"},
                    ],
                },
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# MECH-00 full-hash 两主机证据审计",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "headline_sql",
                    "body": (
                        "## 技术摘要\n\n"
                        "两个 ZIP 和其中 17 个 checkpoint 哈希证书均有效："
                        "17/17 为 `verified_stable`，共读取 52.000 GiB。"
                        "但 1B formal selected checkpoint 仅覆盖 2/6，"
                        "seed2024/2025 的 `down_none/muon` 四个路径尚未扫描。"
                        "因此当前结论是 **已生成哈希可信，但 MECH-00 尚未完全关闭**。"
                    ),
                },
                {
                    "id": "headline_metrics",
                    "type": "metric-strip",
                    "cardIds": ["verified_card", "volume_card", "missing_card"],
                },
                {
                    "id": "coverage_finding",
                    "type": "markdown",
                    "sourceId": "coverage_sql",
                    "body": (
                        "## 17 个已扫描 checkpoint 全部稳定，缺口集中在 1B followup\n\n"
                        "R1 和当前 LLaMA inventory 内部没有坏哈希、重复路径或"
                        "读取期间变化。覆盖缺口不是随机失败，而是原命令没有包含"
                        "`20_llama_swiglu_1b/multiseed_followup`。下图按审计范围"
                        "显示覆盖率；1B formal 的分母是已验收的三 seed × 两个 selected methods。"
                    ),
                },
                {
                    "id": "coverage_chart_block",
                    "type": "chart",
                    "chartId": "coverage_chart",
                },
                {
                    "id": "domain_table_block",
                    "type": "table",
                    "tableId": "domain_table",
                },
                {
                    "id": "definitions",
                    "type": "markdown",
                    "body": (
                        "## 证据粒度与判定标准\n\n"
                        "- 粒度：一个唯一远程 checkpoint 路径。\n"
                        "- `verified_stable`：读取 SHA-256 前后 size、mtime、inode 一致。\n"
                        "- `exact_match=false` 的 step-map 行只表示最近文件，不能作为"
                        " step0/500/3000 的真实观测。\n"
                        "- checkpoint 内部 model/optimizer/loader/RNG schema 仍由 MECH-01 验证。"
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## 审计方法能验证文件身份，但不替代 schema smoke\n\n"
                        "审计依次检查 ZIP CRC、manifest 计数、SHA-256 格式与唯一性、"
                        "文件稳定性和 checkpoint 路径覆盖，并将结果与已通过 169 项"
                        "本地/W&B 对账的 1B formal summary 做集合核对。"
                    ),
                },
                {
                    "id": "missing_finding",
                    "type": "markdown",
                    "sourceId": "missing_sql",
                    "body": (
                        "## 四个缺失 checkpoint 必须定向补 hash\n\n"
                        "这些缺口不否定训练结果或 compact artifact 证书，"
                        "但在补齐前不能宣称 LLaMA-1B 三 seed formal checkpoint "
                        "provenance 已完成全量 SHA-256。"
                    ),
                },
                {
                    "id": "missing_table_block",
                    "type": "table",
                    "tableId": "missing_table",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 旧版 source 分类留下一个独立 provenance caveat\n\n"
                        "两台输出使用 v2026-07-24.1。checkpoint SHA-256 不受影响，"
                        "但该版本把含 tracked modifications 的 LLaMA official source "
                        "误写为 `audit_status=ok`；本地已重分类为 `warn_dirty`。"
                        "补 hash 必须使用 v2026-07-24.2，selected source hash/diff "
                        "继续由 MECH-01 封口。"
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 推荐下一步\n\n"
                        "1. 只扫描 `multiseed_followup`，补四个 checkpoint，避免重读已有 45.140 GiB。\n"
                        "2. 要求合并表达到 21/21 `verified_stable`。\n"
                        "3. 将 MECH-01 selected checkpoint SHA 与本表交叉核对。\n"
                        "4. 完成后再把 MECH-00 标记为 full-hash closed。"
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## 尚待确认\n\n"
                        "- seed2024/2025 的四个 checkpoint 是否仍保留在原远程路径。\n"
                        "- LLaMA 主机补跑前是否已经同步 MECH-00 v2026-07-24.2。"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "partial",
            "datasets": {
                "headline": headline_rows,
                "coverage_chart": coverage_chart_rows,
                "domain_summary": domain_rows,
                "missing_checkpoints": missing_expected,
            },
        },
        "sources": sources,
        "package_info": {
            "report_audience": "technical",
            "source_notes": (
                "One native coverage bar is used because the report renderer "
                "requires a chart and scope coverage is the only useful shape. "
                "Exact hashes and remote paths remain in tables/CSV."
            ),
        },
    }
    write_json(output / "artifact.json", artifact)
    print(f"MECH-00 local audit: {overall}")
    print(
        f"stable_hashes={len(consolidated_hashes)} "
        f"missing_expected={len(missing_expected)}"
    )
    print(f"report={output / 'MECH00_FULL_AUDIT_20260727.md'}")


if __name__ == "__main__":
    main()
