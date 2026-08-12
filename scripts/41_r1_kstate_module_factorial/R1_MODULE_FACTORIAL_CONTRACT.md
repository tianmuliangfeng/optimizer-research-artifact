# 41 号 R1 模块 K 状态 2×2 因子实验合同

状态：代码编写前冻结。39 号实验结束后才能启动远程训练。

## 目的

检验固定 R1 Newton 配方中两类 MLP K 状态的独立作用与交互：

- `c_fc K ∈ {full, none}`
- `c_proj K ∈ {block4, none}`

四个 cell 必须使用相同 R1 架构、数据、初始化种子、训练步数和 Newton 学习率。`diag` 不属于该 2×2；
它作为已提出的 Selective 方法在主结果中单独对 Muon 和原始 Newton–Muon 比较。

## 四个 cell

| cell | c_fc K | c_proj K | 来源 | 方法解释 |
| --- | --- | --- | --- | --- |
| `both` | full | block4 | 复用 15 号正式结果 | 原始 block4 Newton–Muon 控制 |
| `fc_only` | full | none | 复用 15 号正式结果 | Selective `none` |
| `cproj_only` | none | block4 | 新训练 | 互补模块 bridge |
| `neither` | none | none | 新训练 | Newton 配方 all-none；不得称为正式 Muon baseline |

旧资料中的 `release84` 在本实验和新文档中统一称为 `none`。

## 复用边界

只允许复用
`15_official_newton_muon_r1/analysis/wandb_20260721_multiseed_factorial/r1_multiseed_run_summary.csv`
中 seeds 2024、2025、2026 的 `block4` 和 `none` 行。其六行只读子集保存在
`existing_cells_reference.csv`；canonical 文件与子集文件的 SHA-256 都冻结在
`factorial_contract.json`。新代码不得重跑这两个 cell，也不得用 Muon 行替代 `neither`。

## 新训练边界

只训练 `cproj_only` 与 `neither`，每个 cell 使用 seeds 2024、2025、2026，共 6 个正式训练。
每个 seed 先通过 34 步 exact-shape smoke，跨过 step 32 的第一次 K refresh。正式运行必须上传 W&B，
同时保留本地 manifest、派生源码、源码 diff、逐步指标和 checkpoint。

本实验固定使用 R1 主机物理 GPU0 和 GPU1，在两张卡上并行不同 seed；每个
训练进程通过 `CUDA_VISIBLE_DEVICES` 只能看到一张卡。同一 seed 的两个 cell
在该 seed 所分配的卡上顺序运行。双卡并发 timing 不可用于论文效率结论。

## 主要估计量

对每个 seed 计算：

- `c_fc main = mean(Y_full,* - Y_none,*)`
- `c_proj main = mean(Y_*,block4 - Y_*,none)`
- `interaction = Y_full,block4 - Y_full,none - Y_none,block4 + Y_none,none`

其中 `Y` 是 validation loss，负值表示左侧启用 K 后 loss 更低。主指标为 step 6200
`final_val_loss`；`tail5_val_loss_mean` 与 `normalized_val_auc` 为稳健性指标。

三种子汇总报告均值、样本标准差、df=2 的 95% t 区间、方向计数和 practical margin 0.002。
不得只按单个 seed 或单个 cell 选择结论。

## 资源验收

预期 K 状态：

- `both`: 378 MiB
- `fc_only`: 162 MiB
- `cproj_only`: 216 MiB
- `neither`: 0 MiB

任何新 cell 偏离预期超过 0.01 MiB，或 c_fc/c_proj 模式与 manifest 不一致，正式分析必须失败。

## 结果分支

- 历史分配复现：c_fc 主效应有益且 c_proj block4 主效应有害；停止新增机制训练，写入架构限定结论。
- 主效应复现但交互显著：先解释两个 simple effects；只有确有必要才补一个对角扩展。
- R1 分配方向相反：把 OWT/WikiText 结果限定为架构/配方特异，不进行宽泛事后 sweep。
- 全部落在 practical margin：结合 39 号投稿审计后决定是否值得增加两个 seed。

后续实验由 41 号结果决定，不在结果出现前自动扩展。
