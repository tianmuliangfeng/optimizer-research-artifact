# 实验 48：LLaMA-1B 三预算长 token 正式分析

完整性状态：**passed**。冻结分类：`persistent_muon_lead`。

三个预算点均来自相同 peak-LR 语义和 1800-step cooldown 的独立分叉终点；它们不是同一最终轨迹上的普通中途 checkpoint。

| budget | down_none | down_diag | newton_full | muon |
|---|---:|---:|---:|---:|
| tokens_3p2506b | 2.974236 | 2.976024 | 2.976155 | 2.969889 |
| tokens_6p9694b | 2.862803 | 2.863701 | 2.864014 | 2.858799 |
| tokens_approximately_10b | 2.817789 | 2.818116 | 2.818708 | 2.814068 |

解释边界：本实验检验同一 LLaMA-1B 架构内部的训练阶段效应；不能单独识别纯架构因果，也不解释 refresh harm 的来源。

配对差异、逐 seed 终点、tail-5 和 normalized AUC 分别见 `paired_contrasts.csv` 与 `endpoint_results.csv`。
