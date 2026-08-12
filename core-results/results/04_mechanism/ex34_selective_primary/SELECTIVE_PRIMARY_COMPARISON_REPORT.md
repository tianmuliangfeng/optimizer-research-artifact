# Selective Newton–Muon：以两个外部基线为中心的统一主分析

## 技术摘要

本报告纠正比较层级：`diag` 与 `none` 都是本文提出的 Selective Newton–Muon 方案。主比较是每个方案分别对 Muon 和原始 Newton–Muon；`diag vs none` 不属于主对比合同。

- GPT-R1：两个 Selective 方案均优于 Muon；diag 基本保留原始 block4，none 则相对 block4 有可辨认损失。
- LLaMA-124M：两个 Selective 方案、Newton-full 与 Muon 处于紧密核心组，当前三 seed 不能支持稳定质量排序。
- LLaMA-1B：两个 Selective 方案均稳定优于 Newton-full，但后期均稳定落后于 Muon。因此 Selective 改善了原始 Newton–Muon，却没有消除1B 后期的 family-level gap。

负 delta 表示左侧方法 loss 更低。均值与 SD 均来自 seeds 2024/2025/2026 的配对差；practical margin 为 0.002 loss。

## 主证据：每个 Selective 方案分别对两个基线

| 架构 | 主对比 | Final delta mean ± SD | 左侧更优 seeds | Tail-5 delta | AUC delta |
|---|---|---:|---:|---:|---:|
| GPT-R1 | `diag − muon` | -0.016033 ± 0.001002 | 3/3 | -0.016440 | -0.018227 |
| GPT-R1 | `none − muon` | -0.010467 ± 0.000503 | 3/3 | -0.010747 | -0.009260 |
| GPT-R1 | `diag − block4` | -0.001100 ± 0.000557 | 3/3 | -0.001180 | -0.000809 |
| GPT-R1 | `none − block4` | +0.004467 ± 0.000503 | 0/3 | +0.004513 | +0.008158 |
| LLaMA-124M | `down_diag − muon` | -0.000652 ± 0.000812 | 2/3 | -0.000666 | -0.002049 |
| LLaMA-124M | `down_none − muon` | -0.000605 ± 0.000947 | 2/3 | -0.000600 | -0.002263 |
| LLaMA-124M | `down_diag − newton_full` | +0.000073 ± 0.000620 | 2/3 | +0.000048 | -0.000957 |
| LLaMA-124M | `down_none − newton_full` | +0.000119 ± 0.000779 | 2/3 | +0.000114 | -0.001172 |
| LLaMA-1B | `down_diag − muon` | +0.005623 ± 0.000944 | 0/3 | +0.005640 | -0.001004 |
| LLaMA-1B | `down_none − muon` | +0.004471 ± 0.000805 | 0/3 | +0.004521 | +0.000523 |
| LLaMA-1B | `down_diag − newton_full` | -0.000964 ± 0.000330 | 3/3 | -0.000908 | -0.002746 |
| LLaMA-1B | `down_none − newton_full` | -0.002116 ± 0.000645 | 3/3 | -0.002027 | -0.001219 |

这张表是论文主比较。`diag − none` 被有意排除，因为两者都是本文方案；它只能出现在补充消融中。

## 基线关系决定 Selective 方案的解释

| 架构 | 原始 Newton–Muon − Muon | Final 同方向 seeds | 解释 |
|---|---:|---:|---|
| GPT-R1 | -0.014933 ± 0.000924 | 3/3 left-better, 0/3 left-worse | 原始 Newton–Muon 本身优于 Muon；Selective 目标是保留收益并降低状态。 |
| LLaMA-124M | -0.000725 ± 0.000328 | 3/3 left-better, 0/3 left-worse | 两条基线近似打平；Selective 的主要价值是状态效率而非明确质量提升。 |
| LLaMA-1B | +0.006587 ± 0.001176 | 0/3 left-better, 3/3 left-worse | 原始 Newton–Muon 后期落后于 Muon；Selective 虽改善原始方法，仍未反超 Muon。 |

## 状态成本：两个方案都比原始 Newton–Muon 更轻

| 架构 | 方法角色 | 方法 | K-state MiB | Final loss mean ± SD |
|---|---|---|---:|---:|
| GPT-R1 | `selective_diag` | `diag` | 162.281 | 3.261100 ± 0.001114 |
| GPT-R1 | `selective_none` | `none` | 162.000 | 3.266667 ± 0.000666 |
| GPT-R1 | `original_newton_muon` | `block4` | 378.000 | 3.262200 ± 0.001136 |
| GPT-R1 | `muon` | `muon` | 0.000 | 3.277133 ± 0.000252 |
| LLaMA-124M | `selective_diag` | `down_diag` | 162.188 | 3.266926 ± 0.000642 |
| LLaMA-124M | `selective_none` | `down_none` | 162.000 | 3.266973 ± 0.001899 |
| LLaMA-124M | `original_newton_muon` | `newton_full` | 546.000 | 3.266853 ± 0.001149 |
| LLaMA-124M | `muon` | `muon` | 0.000 | 3.267578 ± 0.001192 |
| LLaMA-1B | `selective_diag` | `down_diag` | 1728.756 | 2.975748 ± 0.000622 |
| LLaMA-1B | `selective_none` | `down_none` | 1728.000 | 2.974596 ± 0.000848 |
| LLaMA-1B | `original_newton_muon` | `newton_full` | 5888.250 | 2.976712 ± 0.000517 |
| LLaMA-1B | `muon` | `muon` | 0.000 | 2.970125 ± 0.000954 |

K-state 只表示输入预条件器状态，不等于总 optimizer state 或峰值显存。不同架构的绝对 loss 也不可直接横向比较。

## 范围、数据与指标定义

- 架构：GPT-R1、LLaMA-124M、LLaMA-1B。
- population：每个架构的正式 seeds 2024/2025/2026，固定训练预算的最终验证点。
- Selective 方案：R1 的 `diag/none`；LLaMA 的 `down_diag/down_none`。
- 原始 Newton–Muon：R1 的官方 `block4`；LLaMA 的 `newton_full`。
- Muon：独立的 reference Muon baseline；`none` 不等于 Muon。
- Primary metric：paired final validation loss delta；negative 表示左侧更好。
- Supporting metrics：paired tail-5 delta 与 normalized validation AUC delta。

## 方法与验证

所有 delta 都由逐 run authoritative summary 重新计算，而不是从旧报告文字或名义排名抄录。每个 family 强制要求 4 methods × 3 seeds 恰好12 个唯一 method/seed cells；缺失或重复会终止分析。95% t 区间使用n=3、df=2，仅作为不确定性描述，不以显著性替代 effect size 与三 seed 方向一致性。

本报告使用精确表格而非跨架构柱状图：三个架构的绝对 loss 尺度与训练语义不同，主问题又要求精确读取十多个配对 contrast；合并图形容易把跨架构数值误读成同一量纲的排名。

## 局限、稳健性与可写边界

- 每个架构只有 3 seeds，区间较宽；结论依赖配对设计、方向一致性和预先使用的 0.002 practical margin。
- LLaMA-1B 的 W&B timing 与 memory 不作为性能证据；质量结论来自固定 token budget 的 validation metrics。
- MECH-05/06 只回答 Newton–Muon 家族内部 K-state 选择，不能替代本报告的 family-level 主比较。
- 可以写“Selective 改善/压缩原始 Newton–Muon”；只有当其对 Muon contrast 同时满足证据时，才可以写“相对 Muon 保持或改进”。
- 1B 必须如实写成：Selective 优于 Newton-full，但后期仍落后 Muon。

## 下一步

无需新增长训练。唯一建议补充的是 LLaMA-1B seed2026 的 4 methods × early/late checkpoints 只读 family-level diagnostic，用于解释 early-to-late family reversal；它不能替代三 seed 长程因果比较。

## 进一步问题

若只读 family diagnostic 能稳定定位某类 update-direction 或held-out shadow-loss 反转，再条件式决定是否在 GPT-R1 与 LLaMA-124M endpoint 复现；否则停止扩实验。

## Source audit

- r1: `15_official_newton_muon_r1/analysis/wandb_20260721_multiseed_factorial/r1_multiseed_run_summary.csv`, SHA-256 `ab91afd37db5559c031ed1ffac5441b57856386e62175d94f3f55b3f2568dcc3`, rows=12.
- llama124: `17_llama_swiglu_validation/analysis/wandb_20260722_multiseed/llama_multiseed_run_summary.csv`, SHA-256 `249bfdca2e15417fc639b134afa4d7b3ded08a32026987c7b024aab38e26f7d7`, rows=15.
- llama1b: `20_llama_swiglu_1b/analysis/formal6200_multiseed_20260727/llama1b_formal_multiseed_run_summary.csv`, SHA-256 `66fdd4d3619f2b33d0621257c85139e6803058bbe0051d5894720ca0ee687a83`, rows=12.
