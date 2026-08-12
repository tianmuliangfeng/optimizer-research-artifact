# 完整 60-run 数据核查与初步结论

生成时间：2026-07-18（Asia/Shanghai）

## 1. 完整性结论

- 计划命令：60 runs。
- 旧批次：40 个唯一 run。
- 本次补充：20 个唯一 run。
- 新旧重叠：0。
- 合并后：60 个唯一 run。
- 相对计划：缺失 0，意外 0。
- 设计覆盖：5 suites × 3 seeds × 4 methods = 60。
- seeds：2024、2025、2026。
- methods：`muon_blog`、`diag_cproj_k`、`no_cproj_k`、`paper_block4`。
- 原始 W&B 指标导出：55 个文件，SHA-256 重复 0。
- 标准化数据：615 个唯一 run-metric 组合；质量概览也是 615 行。
- 全部 run 的 `lr/matrix` 均为常数 0.01。

验证集轨迹覆盖：

| Suite | Runs | 每个 run 的 val/loss 点数 | 首个 step | 最后观测 step |
|---|---:|---:|---:|---:|
| OWT 12L / 100M | 12 | 25 | 0 | 12000 |
| WikiText 12L / 100M | 12 | 25 | 0 | 12000 |
| OWT 18L / 3k | 12 | 6 | 0 | 2500 |
| OWT 24L / 12k | 12 | 24 | 0 | 11500 |
| WikiText 24L / 12k | 12 | 24 | 0 | 11500 |

注意：18L 和 24L 的表中“final”是最后一个共同观测 checkpoint，并非优化器执行完毕后的终端 step。

## 2. 跨 seed 最后观测验证损失

数值为 3 seeds 均值；同一 suite 内越低越好。不要直接比较不同 suite 的绝对 loss。

| Suite | Muon | diag | none | block4 | 最优方法 |
|---|---:|---:|---:|---:|---|
| OWT 12L / 100M | **4.071494** | 4.100724 | 4.109021 | 4.105402 | Muon |
| WikiText 12L / 100M | **3.474540** | 3.509995 | 3.513991 | 3.512089 | Muon |
| OWT 18L / 3k | 5.101494 | **5.085648** | 5.095412 | 5.094326 | diag |
| OWT 24L / 12k | 6.064825 | **5.595164** | 5.642381 | 5.920393 | diag |
| WikiText 24L / 12k | 5.583172 | **5.066822** | 5.115395 | 5.393732 | diag |

Muon 相对 diag 的最后观测 loss 差值（Muon − diag）：

| Suite | 差值 | 解释 |
|---|---:|---|
| OWT 12L | **-0.029230** | Muon 更好 |
| WikiText 12L | **-0.035454** | Muon 更好 |
| OWT 18L | +0.015846 | diag 小幅更好 |
| OWT 24L | +0.469661 | diag 大幅更好 |
| WikiText 24L | +0.516351 | diag 大幅更好 |

## 3. 这次补齐数据后最重要的新结论

### 3.1 12L 的 Muon 优势是稳定结果，不是单 seed 偶然

- 在 OWT 12L 和 WikiText 12L 上，Muon 对 diag、none、block4 的最后观测 loss 均为 3/3 seeds 获胜。
- 排除 step 0 后，两个 12L suite 共 6 个 seed、24 个观测 checkpoint、3 个对手：Muon 在 **432/432** 个配对比较中 loss 更低。
- 因而可以把结论升级为：在当前 12L、batch=16、约 100M token、matrix LR=0.01 的口径下，Muon 对三个 Newton 变体取得跨数据集、跨 seed、贯穿训练轨迹的稳定优势。

### 3.2 优势在 18L 已反转，在 24L 强烈反转

- OWT 18L 中 diag 对 Muon 的优势较小，但最后观测点仍是 3/3 seeds 获胜。
- 两个 24L suite 中，diag 对 Muon 的最后观测优势分别为 0.4697 和 0.5164，并且用 best loss 代替最后 loss 后排序仍不改变。
- 因而“Muon 有时压倒性领先、有时明显落后”是真实且可重复的 regime reversal，不再能用缺 seed 或单次噪声解释。

### 3.3 diag 是四种方案里跨 regime 最稳健的 Newton 变体

- diag 在全部 5 个 suite 中都优于 none 的 3/3 seeds。
- 12L 中 diag 不及 Muon，但优于 none，并通常略优于 block4。
- 18L 中 diag 小幅第一；24L 中 diag 明显第一。
- 当前证据支持“对 c_proj 使用 diagonal K 比 full block4 更稳健”，不支持 block4 在更大模型上天然更好。

## 4. 为什么 Muon 会发生反转

当前证据最支持“实验 regime 与固定超参数共同导致”，而不是 Muon 核心代码不合规。

1. **模型规模与 batch 同时改变，不能单独归因于深度。** 12L、18L、24L 的 micro-batch 分别为 16、8、2；每步 token 分别为 8192、4096、1024。Muon 的正交化更新对梯度噪声与更新频率敏感，这些变量足以改变最优学习率和方法排序。
2. **matrix LR 固定为 0.01，没有按规模或方法重新调优。** 这是一张“统一参考 LR”比较表，不是每种方法 best-of-grid 后的公平上限比较。12L 的结果说明 0.01 很适合当前 Muon 口径；24L 的结果只说明 0.01 在该 regime 下明显不合适，不能直接推出 Muon 方法本身失效。
3. **24L 在相同名义 12.288M token 下执行了 4 倍于 18L 的优化器步数。** 18L 为 3000 steps × 4096 tokens；24L 为 12000 steps × 1024 tokens。每 token 更新密度改变了。
4. **24L 的 AdamW 与矩阵参数学习率调度失配。** AdamW 在 step 3000 已衰减到最低 LR，而矩阵参数仍以 0.01 更新到 step 12000；后 9000 步名义 matrix/AdamW LR 比达到 100:1。24L 的后期反弹与该失配一致。
5. **方法名不能掩盖真实参数分组差异。** `no_cproj_k` 不是纯 Muon：它只在 `mlp.c_proj` 不用 K，其他隐藏二维矩阵仍使用 full K。因此 Muon vs none 不只是单个 c_proj 消融。

这些是有数据和配置支持的高可信诊断，但 batch、深度、宽度、步数密度和调度被绑定变化，当前 60-run 仍不能给出单一因素的严格因果占比。

## 5. Muon 实现合规性结论

静态对照本仓库 vendored 的公开参考实现后，没有发现能够解释结果反转的核心实现违规：

- EMA-form momentum 与 Nesterov 路径一致；
- BF16 Newton–Schulz 5 步及系数一致；
- packed Q/K/V 分块后分别正交化；
- shape-aware 缩放存在；
- hidden 2D matrix 使用 Muon，embedding/head/1D 参数走 AdamW；
- matrix weight decay 为 0；
- 四个实验方法共享 momentum、QKV split、shape scaling 和 NS dtype 口径。

因此目前不应把 24L 的落后定性为“Muon 代码作弊或不合规”。更准确的说法是：核心实现基本合规，但实验配置并非跨规模调优公平，固定 LR 和更新密度对 Muon 极其不利。

限制：本地可用 Python 环境没有 PyTorch，未重跑仓库内对齐单元测试；实现结论来自静态源码对照和成功训练轨迹。

## 6. 下一步最有信息量的实验

1. 在 24L 固定其他配置，对四种方法分别扫描 matrix LR：0.003、0.005、0.01。
2. 解耦 24L 的 micro-batch 与有效 batch：用 gradient accumulation 得到有效 batch 8 或 16，并明确同时控制总 token 与 optimizer steps。
3. 给矩阵参数加入与训练长度一致的 LR decay 对照，避免 step 3000 后 matrix/AdamW 比例持续漂移。
4. 记录每类矩阵的 update RMS、parameter RMS、update/parameter ratio，以及梯度噪声代理指标。
5. 论文中分开报告“统一 LR 主表”与“各方法 best-of-grid 表”。

## 7. 可复核数据文件

- `run_summary.csv`：60 个 run 的逐 run 汇总。
- `method_seed_aggregate.csv`：suite × method 的跨 seed 汇总。
- `paired_seed_summary.csv`：同 seed 的方法配对差值。
- `normalized_history_long.csv`：全部标准化长表轨迹。
- `validation_loss_aligned.csv`：验证 loss 对齐表。
- `pairwise_validation_summary.csv`：逐 checkpoint 配对胜负汇总。
- `source_manifest.csv`：55 个原始导出文件、SHA-256 与归档路径。
- `batch_metadata.json`：完整数据集元信息。
