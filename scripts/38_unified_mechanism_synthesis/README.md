# 38 — 统一机制主分析

这是 CPU-only、只读的统一机制分析。它把以下已验收证据连接为一条可复算证据链：

- OWT 与 WikiText-103 的 24 层 GPT 模块 K 状态结构实验；
- 两个数据集上的互补模块 bridge；
- 实验 40 的 LLaMA `block4` 坐标分块不变性审计；
- MECH-01 至 MECH-09R；
- 三架构正式训练主比较；
- R1 block 与 dense-full alpha 三种子确认。

分析不会读取 checkpoint、不会调用在线 W&B API，也不会运行训练。比较优先级、统计单位和可声明边界冻结在
`UNIFIED_MECHANISM_CONTRACT.md`，输入只来自 `source_registry.json`。

## 运行

```bash
bash commands/38_unified_mechanism_synthesis/20260729_unified_mechanism_synthesis.sh
```

脚本依次运行单元测试、写入分析产物，并执行独立只读复算验收。结果位于：

```text
${SNM_RESULTS_ROOT}/
  38_unified_mechanism_synthesis/<UTC timestamp>/
```

## 新纳入的历史证据

`foundational_module_structure.csv` 记录 OWT/WikiText-103 三种子结构排序、K 状态规模和显存。
`none` 表示移除 `c_proj` K、保留非 `c_proj` K（旧资料曾称 `release84`）；`dense_full` 是稠密机制控制，不等同于官方
block4 收缩。WikiText 的 Muon 行来自同配方历史参照，不与 dual-alpha cohort 强行做配对差值。

`complementary_bridge_summary.csv` 对比相反的模块分配：仅保留 `c_proj` K、移除非 `c_proj` K。
它与 `none` 按 seed 严格配对，用来判断有用的历史 K 贡献位于哪一类模块。

这些结果是 24 层 GPT 上的 supportive evidence，不能代替待运行的 41 号 R1 模块 2×2 因子实验。

`architecture_transfer_boundary.csv` 记录实验 40 的限制证据：LLaMA
连续 `block4` 更新对隐藏坐标分块呈强非不变性，因此它不能被当作跨架构通用的
原始 Newton–Muon 或 LLaMA 主基线。LLaMA 的官方原始 Newton–Muon 对照仍是
`newton_full`。该审计不构成完整训练 loss 排名。

## 主要输出

- `source_audit.csv` / `source_audit.json`
- `foundational_module_structure.csv`
- `complementary_bridge_summary.csv`
- `architecture_transfer_boundary.csv`
- `primary_training_contrasts.csv`
- `rollout_cluster_bootstrap.csv`
- `prediction_rollout_alignment.csv`
- `alpha_synthesis.csv`
- `mechanism_chain.csv`
- `claim_evidence_matrix.csv`
- `UNIFIED_MECHANISM_SYNTHESIS_REPORT.md`
- `artifact.json`
- `unified_mechanism_manifest.json`
