# 38 号统一机制综合：冻结分析合同

状态：2026-07-29 修订，新增历史模块分配证据；正式比较优先级不变。

## 研究问题

把已验收的 OWT/WikiText-103 模块 K 状态实验、MECH-01 至 MECH-09R、三架构正式训练主比较和
R1 alpha 实验连接为一条可审计证据链。分析只读取既有产物，不训练模型、不读取 checkpoint，也不修改
任何上游结果。

## 正式方法比较优先级

每个正式模型族只允许以下主比较：

1. `selective_diag_vs_muon`
2. `selective_none_vs_muon`
3. `selective_diag_vs_original_newton_muon`
4. `selective_none_vs_original_newton_muon`

`original_newton_muon_vs_muon` 仅作为家族基线。`selective_diag_vs_selective_none` 不是主比较，不得进入
正式主结果、主结论或成功门。所有 loss 差值定义为 `left - right`，负值表示左侧更好。

历史 OWT/WikiText 结构表可以同时列出各结构模式和描述性排序，但不能把 diag-versus-none 提升为论文主轴。

## 历史模块分配证据合同

- 架构固定为 24 层 GPT，种子固定为 2024、2025、2026。
- `none`：移除 `c_proj` K，保留非 `c_proj` K；旧资料中的 `release84` 仅作为来源别名。
- `diag`：`c_proj` 仅保留逐坐标对角尺度。
- `block4`：四块结构控制。
- `dense_full`：稠密 `c_proj` 机制控制；不得把它误称为官方 block4 原始方法。
- `cproj_only` bridge：保留 `c_proj` K，移除非 `c_proj` K。
- WikiText dual-alpha 汇总中的四种结构模式可以同 cohort 配对；Muon 仅作为同配方历史参照，不能伪装成
  dual-alpha 同批次运行。
- 模块 bridge 必须与同数据集、同 seed 的 `none` 逐种子配对。
- 历史模块结论为 `supportive`，不能直接外推到 R1；R1 模块定位由 41 号 2×2 因子实验决定。
- 历史 timing 不得用于投稿效率结论；K 状态大小和已验收 peak memory 可保留为结构/资源证据，并由 39 号
  投稿效率审计决定最终可引用范围。

## 证据等级

- `confirmatory`：预先冻结比较、独立种子或严格共享前缀因果干预支持。
- `supportive`：真实轨迹、跨数据集或跨架构结果支持，但设计不足以单独承担广义因果主张。
- `descriptive`：几何、代理损失、运行时诊断或事后汇总。
- `limiting`：失败门、不确定结果或明确限制更强主张的反证。

不得因为结果不利而删除 `limiting` 证据。

## LLaMA block4 跨架构边界

- 实验 40 只回答连续四块 `block4` 是否对隐藏坐标重排保持等变；
- 验收分类必须为 `strong_non_invariance`；
- 该结果属于 `limiting` 证据，支持“连续 `block4` 依赖任意坐标分块”；
- 不得据此声称 `block4` 在完整训练中优于或劣于任何优化器；
- `block4` 不得被称为 LLaMA 的原始 Newton–Muon 或主基线；
- LLaMA 的官方原始 Newton–Muon 对照固定为 `newton_full`。

## 输入边界

只读取 `source_registry.json` 明确列出的已验收分析文件。原始训练日志、checkpoint、W&B 在线接口和未验收
临时目录不在输入边界内。每个输入记录 SHA-256、大小、行数/键以及验收条件。

## 短程轨迹统计

MECH-08 的配对单位为 `(checkpoint_cell, data_replica)`：

1. 外层以 `checkpoint_cell` 为簇进行有放回抽样；
2. 内层在抽中的簇内对 replicas 有放回抽样；
3. 固定随机种子 `20260729` 和 5000 次 bootstrap；
4. 输出均值、95% percentile interval、簇数、配对单位数和 origin 方向一致性；
5. 区间完全小于零为 `left_better`，完全大于零为 `left_worse`，否则为 `uncertain`。

MECH-08 并发 timing 明确不可用于投稿效率结论。

## 机制主张边界

- 历史 OWT/WikiText 模块实验支持“昂贵的 dense/block `c_proj` K 并非唯一有用来源”，不证明 R1 必然同向。
- 实验 40 将连续 `block4` 的可迁移范围限制在其坐标分块定义内，不提供完整训练 loss 排名。
- MECH-01 只支持实现与运行域一致性。
- MECH-02 只支持 K 几何具有结构；几何门不能自动推出训练优势。
- MECH-03/06 的失败或不确定预测限制“一步代理可预测长程结果”的主张。
- MECH-07 是冻结 checkpoint 上的反事实 shadow 证据，不等同于真实训练轨迹。
- MECH-08 是 128 步真实 rollout 的轨迹证据。
- MECH-09R 只识别其冻结因果树中的 down-projection refresh 中介作用。
- alpha 实验支持被测网格上的非单调剂量响应；不得宣称统一最优 alpha。
- 正式训练结果决定最终方法表现；机制实验解释表现，不替代正式训练比较。

## 必需输出

- `source_audit.csv`
- `source_audit.json`
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
- `report_source_notes.json`
- `unified_mechanism_manifest.json`

任一必需输入缺失、验收失败、主比较集合污染、重复配对键、种子覆盖不完整或输出为空，分析必须以非零状态退出。
