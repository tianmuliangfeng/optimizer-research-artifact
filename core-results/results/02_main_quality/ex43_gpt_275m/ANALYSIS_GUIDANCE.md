# 实验 43 最终分析指导

> 文档性质：基于 seed 2024 部分结果形成的分析备忘录  
> 记录日期：2026-07-31  
> 适用实验：43，约 275M Modded-NanoGPT，Record #28 对齐  
> 权威边界：本文件不修改实验合同、预注册指标、方法、seed 或停止规则；若与冻结合同或正式分析器冲突，以冻结合同和正式分析器为准。

## 1. 用途与证据边界

本文件用于在实验 43 的四个 formal seed 全部完成后，提醒分析者重点核验已经在 seed 2024 中出现的质量—状态模式。seed 2024 的结果已经被查看，因此下文提出的“增益分解”和轨迹解释只能作为明确标注的阶段性、探索性分析线索，不能替代预注册的 paired contrasts，也不能据此选择性排除 seed、修改方法或追加超参数搜索。

正式结论必须来自冻结分析器对 seeds `2024/2025/2026/2027` 的统一分析。不得只使用本文件中的 seed 2024 数字生成论文主表。

## 2. 冻结实验语义

四种方法为：

- `muon`：不分配 K state。
- `original_newton_muon`：attention 与 `c_fc` 使用 full K，`c_proj` 使用上游 block4 K。
- `selective_none`：保持其他 Newton 路由不变，仅移除 `c_proj` K。
- `selective_diag`：保持其他 Newton 路由不变，仅将 `c_proj` K 替换为四个 block 对齐的 diagonal scales。

因此，`original_newton_muon` 的 contraction/`c_proj` 模式是 `block4`，但整个方法不应简称为“只有 block4”。`original_newton_muon` 与两个 Selective 方法之间唯一允许的算法差异是 `c_proj` 的 K 表示。

冻结分析口径：

- formal seeds：`2024/2025/2026/2027`；
- 主终点：step `1695` 的 final validation loss；
- common target loss：`3.3`；
- validation grid：step `0, 50, ..., 1650, 1695`，共 35 点；
- final-loss practical margin：`±0.002`；
- 主对比：`none-muon`、`none-original`、`diag-muon`、`diag-original`；
- 强制 benchmark anchor：`original-muon`；
- `diag-none` 仅作次要描述；
- 并发条件下的 wall time、tokens/s 和 steps/s 不得作为论文效率证据。

## 3. seed 2024 阶段性结果

### 3.1 完整性状态

- 四个 formal cell 均为 `scientifically_complete`，本地验收检查全真。
- 包内列出的工件哈希均匹配。
- 四种方法共享相同初始化哈希和参数结构哈希。
- 35 个预期验证点齐全，final step 和 token budget 正确。
- W&B 当时均为 `pending_upload`；这不否定本地科学完成，但论文交接前必须完成上传。
- `timing_eligible=false`。
- 收到的压缩包：`ex43 seed2024.zip`；SHA-256：`0904a94cec84eeccad62e5b04b21cacb707e5d4fdb20feac605ea08d98268bc6`。本地未保留解压副本。

### 3.2 Loss 汇总

| 方法 | Final/Best | Tail-5 mean | Normalized AUC | 插值 steps to 3.3 | 首个观测 step <= 3.3 |
|---|---:|---:|---:|---:|---:|
| Muon | 3.276400089 | 3.295685577 | 3.775533730 | 1575.153831 | 1600 |
| Original | 3.274687290 | 3.293531370 | **3.762282530** | 1564.417506 | 1600 |
| Selective-diag | 3.273820639 | 3.292728710 | 3.763419196 | **1561.190114** | 1600 |
| Selective-none | **3.273376465** | **3.292686605** | 3.767818929 | 1561.756034 | 1600 |

所有方法的 best loss 都出现在 final step。seed 2024 上，Original 在前中期轨迹占优，Selective-none 在最后三个验证点取得最低 loss。因而必须区分“全程 AUC 最好”和“固定预算终点最好”。

### 3.3 Final-loss paired differences

差值定义为 `candidate - reference`，负值表示 candidate 更好。

| 对比 | seed 2024 差值 | 按 ±0.002 的单 seed 描述 |
|---|---:|---|
| `none - muon` | **-0.003023624** | favorable，超过 margin |
| `none - original` | -0.001310825 | margin 内，方向有利 |
| `diag - muon` | **-0.002579451** | favorable，超过 margin |
| `diag - original` | -0.000866652 | margin 内，方向有利 |
| `original - muon` | -0.001712799 | margin 内，方向有利 |
| `diag - none` | +0.000444174 | margin 内，none 略好 |

单个 seed 不得被报告为多 seed practical-margin 结论。

### 3.4 需要复核的探索性“增益分解”

对每个 seed `s` 定义：

```text
anchor_gain_s    = loss_muon_s - loss_original_s
selective_gain_s = loss_original_s - loss_none_s
total_gain_s     = loss_muon_s - loss_none_s
```

seed 2024：

- `anchor_gain = 0.001712799`；
- `selective_gain = 0.001310825`；
- `selective_gain / anchor_gain = 0.765312`；
- 在 Muon→None 总 final-loss 收益中，anchor 占 56.65%，移除 `c_proj` block4 K 的额外收益占 43.35%。

这是积极但尚未确认的线索：一个很窄、且使算法更简单的 contraction 路由改动，带来了与完整 Original→Muon 增益同量级的额外终点改善。

最终分析时：

1. 必须先完成预注册 contrasts；
2. 再逐 seed 列出上述三个 gain；
3. 报告 `mean(selective_gain) / mean(anchor_gain)`，不要把不稳定的逐 seed ratio 直接平均；
4. 同时报告 `selective_gain - anchor_gain` 的逐 seed值；
5. 若某 seed 的 `anchor_gain` 接近零或变号，ratio 只作描述，不得作为主结论；
6. 该分解必须标为查看 seed 2024 后提出的 exploratory analysis。

### 3.5 轨迹限制

seed 2024 中：

- Tail-5：Muon→Original 改善 `0.002154207`，Original→None 额外改善 `0.000844765`，后者约为前者的 39.21%。
- AUC：Original 比 Muon 好 `0.013251199`，但 None 比 Original 差 `0.005536399`。
- steps-to-target：Original 比 Muon 快约 `10.74` steps，None 又比 Original 快约 `2.66` steps。

因此不能把 final-loss 的近似等量增益外推成“None 在全轨迹同样增益”。如果其他 seed 复现该动态，应表述为：Original 改善前中期轨迹；None 保留主体收益，并改善后期/终点质量。

### 3.6 状态和峰值显存

| 方法 | Optimizer state | Total preconditioner | Peak allocated |
|---|---:|---:|---:|
| Muon | 1.1711 GiB | 0 | 32.6946 GiB |
| Original | 2.1643 GiB | 1397.25 MiB | 34.0592 GiB |
| Selective-diag | 1.8482 GiB | 848.67 MiB | 33.5234 GiB |
| Selective-none | 1.7424 GiB | 740.25 MiB | 33.4176 GiB |

相对 Original，Selective-none 减少：

- optimizer state：432.00 MiB，`19.49%`；
- total preconditioner：657.00 MiB，`47.02%`；
- peak allocated：657.01 MiB，`1.88%`。

相对 Original，Selective-diag 减少：

- optimizer state：323.72 MiB，`14.61%`；
- total preconditioner：548.58 MiB，`39.26%`；
- peak allocated：548.58 MiB，`1.57%`。

## 4. 四 seed 最终分析的强制顺序

1. 验证 16/16 formal cells 的本地科学完整性、工件哈希、相同 seed 内初始化哈希、参数结构、final step、token budget 和验证 grid。
2. 使用冻结 `analyze_record28.py` 运行正式统一分析，不得手工挑选 checkpoint。
3. 对每个预注册 contrast 报告四个原始 paired differences、paired mean、样本标准差、df=3 的双侧 95% paired-t 区间、方向计数和 practical-margin 分类。
4. 分别分析 final、tail-5、normalized AUC、steps/tokens-to-target；final 是主指标，其余不得代替主指标，也不得隐藏反向信息。
5. 报告 optimizer state、K state、total preconditioner、peak allocated/reserved；并发诊断时间只归档，不进入效率结论。
6. 完成预注册结果后再运行第 3.4 节的探索性增益分解。
7. 单独记录 W&B 完成状态；上传失败不得触发重训已验收 cell。
8. 实验 43 自身分析完成后，才与 124M、455M 做跨规模综合；跨规模一致性不能替代实验 43 的四 seed 配对统计。

## 5. 结果解释决策表

| 最终多 seed 结果 | 推荐解释 |
|---|---|
| None 对 Original final 方向一致、均值在 margin 内，同时状态显著下降 | 最稳健的积极结论：质量等效或略优，状态明显更低，形成更好的 Pareto |
| None 对 Original 与 Original 对 Muon 的 final 改善同号且量级接近 | 作为探索性强化：窄范围 contraction 路由修正贡献了与 benchmark anchor 同量级的额外终点收益 |
| None final 更好，但 AUC 稳定差于 Original | 限定为后期/固定预算终点优势；承认 Original 的前中期优化更强 |
| None 对 Original 超过 `-0.002` 且 paired CI 支持 | 可以报告实用改善，但仍需同时报告 AUC、tail-5 和状态 |
| None 对 Original 在不同 seed 频繁变号 | 结论退回质量等效/不确定，核心价值主要来自状态节省 |
| None 对 Original 明确劣于 `+0.002` | 记录 275M 的尺度边界，不得选择性追加方法或改参数 |

## 6. 写作边界

推荐的保守主张是：

> Selective-none retains the useful Newton routing on attention and expansion while removing contraction block4 K, reaching comparable or slightly better fixed-budget validation loss with substantially less preconditioner state.

除非四 seed 正式统计支持，否则不得写：

- “None 全程支配 Original”；
- “移除 block4 K 带来显著大幅 loss 改善”；
- “wall-clock 更快”；
- “seed 2024 已证明跨 seed 稳健性”。

实验 45 的 Mousse 结果无论方向如何，都不得改写实验 43 的方法集合。若冻结 gate 最终允许 275M Mousse extension，它应属于实验 45 命名空间，并作为独立外部近邻扩展分析。
