# 实验 44 最终分析指导

> 文档性质：基于 seed 2024 部分结果形成的分析备忘录  
> 记录日期：2026-07-31  
> 适用实验：44，约 455M Modded-NanoGPT，Record #17 单 H100 适配  
> 权威边界：本文件不修改实验合同、预注册指标、方法、seed 或停止规则；若与冻结合同或正式分析器冲突，以冻结合同和正式分析器为准。

## 1. 用途与证据边界

本文件用于在实验 44 的三个 formal seed 全部完成后，提醒分析者重点核验已经在 seed 2024 中出现的质量—状态模式。seed 2024 已被查看，下文的“增益分解”和轨迹解释因此是阶段性、探索性分析线索，不能替代预注册 paired contrasts，不能用于挑选 seed、修改方法、重调超参数或增加 optimizer。

正式结论必须来自冻结分析器对 seeds `2024/2025/2026` 的统一分析。实验是上游 Record #17 配方的单 H100、8 次梯度累积适配，不得称为 unchanged official reproduction。

## 2. 冻结实验语义

四种方法为：

- `muon`：不分配 K state。
- `original_newton_muon`：QKV、O 和 MLP expansion 保持冻结的 full K 路由；MLP contraction 使用四个对齐的 `1024×1024` block，即 `cproj_k_mode=block4`。
- `selective_none`：保持其他 Newton 路由不变，仅删除 contraction K。
- `selective_diag`：保持其他 Newton 路由不变，仅把 contraction K 替换成四个对齐的 1024 维 diagonal scales。

所以，“Original 是 block4”只对 contraction/`c_proj` 的 K 表示成立；Original 的其余 Newton 路由仍为 full K。三个 Newton 方法的受控干预只发生在 contraction K。

冻结分析口径：

- formal seeds：`2024/2025/2026`；
- 主终点：step `5960` 的 final validation loss；
- common target loss：`2.95`；
- validation grid：step `0, 125, ..., 5875, 5960`，共 49 点；
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
- 49 个预期验证点齐全，final step、token budget、单卡梯度累积与 compiled-autograd 禁用检查通过。
- W&B 当时均为 `pending_upload`；论文交接前必须完成上传。
- `timing_eligible=false`。
- 收到的压缩包：`ex44 seed2024.zip`；SHA-256：`6788274196618af7b935d454cc203113c03f08a0f1bcfae2f15cefd8baa071b0`。本地未保留解压副本。

### 3.2 Loss 汇总

| 方法 | Final/Best | Tail-5 mean | Normalized AUC | 插值 steps to 2.95 | 首个观测 step <= 2.95 |
|---|---:|---:|---:|---:|---:|
| Muon | 2.919156313 | 2.926203537 | 3.259968924 | 5256.711974 | 5375 |
| Original | 2.917922735 | 2.924476147 | **3.250293501** | **5207.702012** | 5250 |
| Selective-diag | 2.918477297 | 2.925115490 | 3.250815552 | 5222.588581 | 5250 |
| Selective-none | **2.916889191** | **2.923807192** | 3.253629078 | 5214.427448 | 5250 |

所有方法的 best loss 都出现在 final step。seed 2024 上，Original/Diag 在训练前中期占优，Selective-none 从最后约 500 steps 开始取得最低 loss。必须区分 Original 的全程 AUC 优势和 None 的固定预算终点优势。

### 3.3 Final-loss paired differences

差值定义为 `candidate - reference`，负值表示 candidate 更好。

| 对比 | seed 2024 差值 | 按 ±0.002 的单 seed 描述 |
|---|---:|---|
| `none - muon` | **-0.002267122** | favorable，略超过 margin |
| `none - original` | -0.001033545 | margin 内，方向有利 |
| `diag - muon` | -0.000679016 | margin 内，方向有利 |
| `diag - original` | +0.000554562 | margin 内，方向不利 |
| `original - muon` | -0.001233578 | margin 内，方向有利 |
| `diag - none` | +0.001588106 | margin 内，none 较好 |

单个 seed 不得被报告为三 seed practical-margin 结论。

### 3.4 需要复核的探索性“增益分解”

对每个 seed `s` 定义：

```text
anchor_gain_s    = loss_muon_s - loss_original_s
selective_gain_s = loss_original_s - loss_none_s
total_gain_s     = loss_muon_s - loss_none_s
```

seed 2024：

- `anchor_gain = 0.001233578`；
- `selective_gain = 0.001033545`；
- `selective_gain / anchor_gain = 0.837843`；
- 在 Muon→None 总 final-loss 收益中，anchor 占 54.41%，移除 contraction block4 K 的额外收益占 45.59%。

该比例与实验 43 seed 2024 的 76.53% 同方向且量级接近，是当前最值得在剩余 seed 中复核的跨规模线索：一个局部的 subtraction-style intervention，可能贡献与完整 Original→Muon benchmark anchor 同量级的额外终点收益。

最终分析时：

1. 必须先完成预注册 contrasts；
2. 再逐 seed 列出上述三个 gain；
3. 报告 `mean(selective_gain) / mean(anchor_gain)`，不要直接平均逐 seed ratio；
4. 同时报告 `selective_gain - anchor_gain` 的逐 seed值；
5. 若某 seed 的 `anchor_gain` 接近零或变号，ratio 只能描述，不能成为主统计；
6. 该分解必须标为查看 seed 2024 后提出的 exploratory analysis；
7. 与实验 43 的比例相似只能作为跨规模一致性线索，不能增加实验 44 的名义样本量。

### 3.5 轨迹限制

seed 2024 中：

- Tail-5：Muon→Original 改善 `0.001727390`，Original→None 额外改善 `0.000668955`，后者约为前者的 38.73%。
- AUC：Original 比 Muon 好 `0.009675423`，但 None 比 Original 差 `0.003335577`。
- steps-to-target：Original 比 Muon 快约 `49.01` steps；None 比 Original慢约 `6.73` steps，但仍比 Muon 快约 `42.28` steps。

所以 final-loss 的近似等量增益不是全程优化优势。如果其他 seed 复现，应表述为：Original 在前中期和 target crossing 上更强，None 在后期取得更好的固定预算终点。

### 3.6 状态和峰值显存

| 方法 | Optimizer state | Total preconditioner | Peak allocated |
|---|---:|---:|---:|
| Muon | 2.2522 GiB | 0 | 51.4758 GiB |
| Original | 4.6272 GiB | 3316.00 MiB | 54.7141 GiB |
| Selective-diag | 3.8777 GiB | 2036.75 MiB | 53.4648 GiB |
| Selective-none | 3.6272 GiB | 1780.00 MiB | 53.2141 GiB |

相对 Original，Selective-none 减少：

- optimizer state：1024.00 MiB，`21.61%`；
- total preconditioner：1536.00 MiB，`46.32%`；
- peak allocated：1536.01 MiB，`2.74%`。

相对 Original，Selective-diag 减少：

- optimizer state：767.50 MiB，`16.20%`；
- total preconditioner：1279.25 MiB，`38.58%`；
- peak allocated：1279.25 MiB，`2.28%`。

## 4. 三 seed 最终分析的强制顺序

1. 验证 12/12 formal cells 的本地科学完整性、工件哈希、相同 seed 内初始化哈希、参数结构、final step、token budget、验证 grid、梯度累积和 compiled-autograd 状态。
2. 使用冻结 `analyze_record17.py` 运行正式统一分析，不得手工挑选 checkpoint。
3. 对每个预注册 contrast 报告三个原始 paired differences、paired mean、样本标准差、df=2 的双侧 95% paired-t 区间、方向计数和 practical-margin 分类。
4. 分别分析 final、tail-5、normalized AUC、steps/tokens-to-target；final 是主指标，但必须完整披露 AUC 和 target crossing 的潜在反向。
5. 报告 optimizer state、K state、total preconditioner、peak allocated/reserved；并发诊断时间只归档，不进入效率结论。
6. 完成预注册结果后再运行第 3.4 节的探索性增益分解。
7. 单独记录 W&B 完成状态；上传失败不得触发重训已验收 cell。
8. 实验 44 自身分析完成后，才与 124M、275M 做跨规模综合；不得把不同规模当作额外随机 seed 合并统计。

## 5. 结果解释决策表

| 最终多 seed 结果 | 推荐解释 |
|---|---|
| None 对 Original final 方向一致、均值在 margin 内，同时状态显著下降 | 最稳健的积极结论：在最大 paper-aligned 规模上保持质量并改善质量—状态 Pareto |
| None 对 Original 与 Original 对 Muon 的 final 改善同号且量级接近 | 探索性强化：contraction block4 K 的移除贡献了与 benchmark anchor 同量级的额外终点收益 |
| None final 更好，但 AUC/target crossing 稳定差于 Original | 限定为后期/固定预算终点优势，明确 Original 的前中期收敛更强 |
| None 对 Original 超过 `-0.002` 且 paired CI 支持 | 可以报告实用改善，但不得省略 AUC、tail-5 和状态结果 |
| None 对 Original 在不同 seed 频繁变号 | 结论退回质量等效/不确定，主要价值是状态节省及相对 Muon 的总体位置 |
| None 对 Original 明确劣于 `+0.002` | 记录 455M 尺度边界，不得选择性重跑或临时加入新 optimizer |

## 6. 写作边界

推荐的保守主张是：

> At the 455M paper-aligned scale, removing contraction block4 K preserves the useful Newton routing elsewhere and reaches comparable or slightly better fixed-budget validation loss with substantially less preconditioner state.

除非三个 seed 的正式统计支持，否则不得写：

- “None 全程支配 Original”；
- “455M 已证明普遍尺度律”；
- “单 H100 adaptation 是 unchanged official reproduction”；
- “wall-clock 或吞吐更快”；
- “seed 2024 已证明跨 seed 稳健性”。

实验 45 的 Mousse 结果无论方向如何，都不得改写实验 44 的方法集合。冻结计划没有自动的 455M Mousse extension；若未来确有新实验，必须建立独立命名空间和新的预注册合同。
