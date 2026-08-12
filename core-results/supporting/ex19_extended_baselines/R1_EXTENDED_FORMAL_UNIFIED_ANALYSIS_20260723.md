# R1 补充优化器三 seed 正式实验与 core R1 合并分析（2026-07-23）

## 证据状态

- 补充基线：3 methods × 3 seeds = 9 个 6200-step formal runs。
- 合并后：7 methods × 3 seeds = 21 个 R1 runs。
- W&B 导出覆盖：8 个预期指标、全部 run、全部预期 step；195 项检查全部 PASS。
- 证据等级：`PASS_WITH_CAVEATS`。W&B CSV 可以证明曲线、步数、LR、状态显存和初始 loss 的一致性，但不能单独证明远端 source/runtime/init/resume/checkpoint manifests；这些仍需从原运行目录归档。
- seed2026 用于三点 pilot 选 LR；seeds2024/2025 是未参与选参的 confirmatory seeds。正式 timing 不可用，只保留描述性数值。

## 统一三 seed 结果

Primary endpoint 是 step-6200 validation loss；tail-5 与 normalized validation AUC 是预先指定的 secondary endpoints，均为越低越好。

| 方法 | final mean ± seed SD | tail-5 mean | normalized AUC | optimizer state MiB | peak MiB | final rank |
|---|---:|---:|---:|---:|---:|---:|
| Newton–Muon diag | 3.261100 ± 0.001114 | 3.269300 | 3.615159 | 780.756 | 38304 | 1 |
| Newton–Muon block4 | 3.262200 ± 0.001136 | 3.270480 | 3.615968 | 996.475 | 39168 | 2 |
| Newton–Muon none | 3.266667 ± 0.000666 | 3.274993 | 3.624126 | 780.475 | 38304 | 3 |
| Moonlight Muon | 3.274967 ± 0.000451 | 3.286727 | 3.652248 | 618.475 | 37703 | 4 |
| Muon | 3.277133 ± 0.000252 | 3.285740 | 3.633386 | 618.475 | 37703 | 5 |
| NorMuon | 3.334467 ± 0.000924 | 3.342713 | 3.727410 | 618.791 | 37703 | 6 |
| AdamW | 3.400433 ± 0.004692 | 3.410087 | 3.860811 | 942.475 | 38026 | 7 |

该排序在未参与选参的 seeds2024/2025 上不变：`diag < block4 < none < Moonlight < Muon < NorMuon < AdamW`。

## 同 seed 配对差

下表使用 `candidate − reference`，负数表示补充基线更好。

| Candidate | Reference | final 配对均值 ± SD | 三 seed 胜负 | confirmatory seeds2024/2025 均值 |
|---|---|---:|---:|---:|
| Moonlight | diag | +0.013867 ± 0.001365 | 0–3 | +0.014600 |
| Moonlight | block4 | +0.012767 ± 0.001193 | 0–3 | +0.013250 |
| Moonlight | none | +0.008300 ± 0.000854 | 0–3 | +0.008750 |
| Moonlight | Muon | **−0.002167 ± 0.000379** | **3–0** | **−0.001950** |
| NorMuon | diag | +0.073367 ± 0.001553 | 0–3 | +0.073600 |
| NorMuon | Muon | +0.057333 ± 0.001159 | 0–3 | +0.057050 |
| AdamW | diag | +0.139333 ± 0.005372 | 0–3 | +0.142400 |
| AdamW | Muon | +0.123300 ± 0.004590 | 0–3 | +0.125850 |

## Moonlight 与 Muon 的晚期交叉

Moonlight 是唯一需要谨慎解释的补充基线：

- primary final：Moonlight 在三个 seed 均优于 Muon，平均 −0.002167；
- tail-3：Moonlight 仍平均优于 Muon −0.000922；
- tail-5：Moonlight 反而平均差 +0.000987；
- normalized AUC：Moonlight 平均差 +0.018863；
- 在每个 seed 的 62 个 post-initial validation checkpoints 中，Moonlight 只有最后两个（step6100/6200）优于 Muon，前 60 个均更差。

因此，Moonlight 对 Muon 的优势是稳定但高度 endpoint-specific 的 terminal-warmdown crossover。按预注册 primary endpoint，它是最强的非 Newton 补充基线；但现有数据不支持“Moonlight 全程支配 Muon”或“整体优化效率更高”。

## 结论与实验决策

1. 昨天的肉眼观察得到正式确认：即使 Moonlight、NorMuon 和 AdamW 各自获得了三个 LR pilot cells，Newton–Muon 的 diag、block4、none 仍在 step6200 final 上对三种补充基线全部取得 3/3 seed 一致优势。
2. 调参不对称现在更有利于竞争基线而不是我们的 core：Moonlight/NorMuon/AdamW 是 pilot-tuned，Newton trio 的主配置保持冻结；未参与选参的 seeds2024/2025 仍复现相同排序。不过这不是等预算全局超参数搜索，论文应披露候选 LR 数量与 tuned/prespecified 身份。
3. Moonlight 通过 R1 竞争性 gate：它明显落后于 Newton trio，但在相同 optimizer-state/peak memory 下，final 稳定优于 Muon。它应继续进入 LLaMA-124M 架构适配 pilot；是否进入 1B 必须由 LLaMA-124M seed2026 formal 决定。
4. NorMuon 没有通过 R1 竞争性 gate：它在全曲线和 final 上均明显落后于 Muon/diag，默认不进入 1B。若 LLaMA-124M pilot 已经在跑，可让它完成以形成架构审计证据，但不自动追加 6200-step formal 或多 seed。
5. AdamW 仍有论文价值，因为它是标准一阶基线；它不靠 R1 竞争性晋级，而是凭基线必要性和已有 LLaMA-124M 三 seed 证据进入独立 1B pilot。
6. 状态—质量权衡更清晰：Moonlight/Muon 具有最低状态与峰值显存；diag 相对 Moonlight 多约 162.28 MiB optimizer state、601 MiB peak，却换得 final −0.01387、tail-5 −0.01743、AUC −0.03709。这个结果应写成 Pareto/trade-off，而不是声称 diag 在所有资源维度支配现代基线。

## 论文表述边界

可写：在给 Moonlight、NorMuon 和 AdamW 每种三个预设 LR 的小型选参预算后，Newton–Muon 的三个 K 表示在 tuned seed 和两个 confirmatory seeds 上仍取得更低的 step-6200 validation loss。

不可写：完成了所有优化器的等预算全局调参；Moonlight 全程优于 Muon；现有 wall-clock 是正式性能结论；n=3 已足以支持大样本显著性声明。
