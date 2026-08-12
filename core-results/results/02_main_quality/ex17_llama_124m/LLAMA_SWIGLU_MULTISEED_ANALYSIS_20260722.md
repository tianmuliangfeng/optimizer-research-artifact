# LLaMA/SwiGLU-124M 三 seed 实验分析

日期：2026-07-22  
范围：seeds 2024/2025/2026，五种方法，6200 optimizer steps，3,250,585,600 tokens/run。

## 技术摘要

LLaMA/SwiGLU 上没有复现 GPT-R1 中 `diag` 对 `none` 的稳定优势。LLaMA 三 seed 的主终点配对差 `down_diag - down_none` 为 `-0.000310`、`-0.001187`、`+0.001357`，均值仅 `-0.000047`，95% t 区间为 `[-0.003257, +0.003164]`。方向为 2:1，但量级远小于 GPT-R1 三 seed 稳定的均值 `-0.005567`。

用相同 seed 做架构交互描述，`LLaMA(diag-none) - GPT(diag-none)` 三个值均为正，均值 `+0.005520`，95% t 区间 `[+0.003581, +0.007459]`。结合已经完成的 GPT-on-LLaMA-host bridge（`-0.0049`，原 GPT 主机为 `-0.0050`），最合适的结论是：**diag/none 的相对收益具有明显的 architecture-associated interaction，不能从 GPT-GELU 直接迁移到 LLaMA-SwiGLU。**

五方法中，`newton_full` 的三 seed 平均终点最低（3.266853），但它没有赢得任何单 seed，且只比 `down_diag` 低 0.000073、比 `down_none` 低 0.000119；这些差异远小于跨 seed 波动，不能据此声称质量优势。`down_diag`、`down_none`、`newton_full`、`muon` 在当前预算下形成一个非常紧的核心组。AdamW 每个 seed 都显著落后，平均高约 0.096 loss。

## 主终点：step 6200 validation loss

| seed | down_diag | down_none | newton_full | muon | adamw | 当 seed 最优 |
|---:|---:|---:|---:|---:|---:|---|
| 2024 | 3.266753 | 3.267064 | 3.267097 | 3.268193 | 3.370099 | down_diag |
| 2025 | 3.267637 | 3.268824 | 3.267861 | 3.268336 | 3.356022 | down_diag |
| 2026 | 3.266387 | 3.265030 | 3.265602 | 3.266205 | 3.364128 | down_none |

| 方法 | 三 seed 均值 | seed SD | 平均名次 | seed 胜场 | normalized AUC 均值 |
|---|---:|---:|---:|---:|---:|
| newton_full | 3.266853 | 0.001149 | 2.333 | 0 | 3.612401 |
| down_diag | 3.266926 | 0.000642 | 2.000 | 2 | 3.611443 |
| down_none | 3.266973 | 0.001899 | 2.333 | 1 | 3.611228 |
| muon | 3.267578 | 0.001192 | 3.333 | 0 | 3.613492 |
| adamw | 3.363416 | 0.007065 | 5.000 | 0 | 3.814964 |

`newton_full` 的终点均值名义上第一，而 `down_none` 的整程 normalized AUC 名义上第一；这说明核心组的微小排序随时间窗口改变。应报告估计量和不确定性，不应把 0.0001 量级的名义排名写成方法胜负。

此前已经使用的固定 loss targets 给出同样结论：到 3.4 loss 的三 seed 平均插值 step 为 down_none 4134.5、down_diag 4135.5、newton_full 4149.4、muon 4191.9、AdamW 5569.2。diag 与 none 只差约 1 step，核心组差距远小于 AdamW。

## diag 与 none 的配对结果

差值统一定义为 `diag - none`，负数偏向 diag。

| seed | LLaMA/SwiGLU | GPT-GELU R1 | LLaMA − GPT 交互差 |
|---:|---:|---:|---:|
| 2024 | -0.000310 | -0.005700 | +0.005390 |
| 2025 | -0.001187 | -0.006000 | +0.004813 |
| 2026 | +0.001357 | -0.005000 | +0.006357 |
| 均值 | -0.000047 | -0.005567 | +0.005520 |

LLaMA 的 tail-5 配对均值为 `-0.000066`，与主终点一致地接近零；AUC 配对均值为 `+0.000215`，方向轻微偏向 none。主终点和 AUC 的符号不同，再次说明这里没有稳定、可迁移的 LLaMA 赢家。

## 资源与可扩展性含义

`newton_full` 相比 `down_diag` 没有可辨认的质量收益，但每 run 的 optimizer state 多约 383.8 MiB，峰值 allocated 显存多约 596.7 MiB。`muon` 的 optimizer state 与峰值显存最低，但终点均值比 `newton_full` 高 0.000725；三 seed 都是 newton_full 更低，不过 n=3 的 95% 区间仍轻微跨零（上界约 +0.000090）。

这些运行的 wall-clock 和 ms/step 只保留作诊断，不进入性能结论。1B fixed-memory/OOM 实验仍有必要，因为 124M 的固定 batch 不能直接回答最大可运行 batch/context。

## 数据质量与方法

- seed2024/2025 的两个 formal manifest 均为 completed，5/5 方法完成，无 failed methods，W&B 上传完整。
- 所有新 run 均到达 6200 steps 和 3,250,585,600 tokens，`resume_count=0`。
- 10/10 新摘要文件的 SHA-256 与 manifest 完全一致；15/15 run 的 W&B final loss 与本地摘要精确相等。
- seed2024/2025 使用同一 H100 80GB、Torch 2.8.0+cu126、CUDA 12.6、同一数据 fingerprint 与脚本 hash；每个 seed 内五方法共享同一初始化 hash。
- 本次新增及合并检查 72/72 通过；seed2026 继续沿用上一批已经完成的 137/137 本地证据审计。
- primary endpoint 为固定预算下 step-6200 validation loss；tail-5、0–6200 trapezoidal normalized AUC、配对差为 secondary/robustness 指标。
- n=3 的 95% 区间使用 seed 级样本 SD 与 t(2) 临界值，只用于展示不确定性，统计功效有限。

## 限制

项目在查看 seed2024/2025 结果前没有冻结 practical non-inferiority margin，因此不能事后把本结果包装成正式的等价性或非劣检验。可以说“观测差异接近零且不稳定”，不能说“已经统计证明等价”。此外，架构交互是经过 host bridge 支持的描述性估计，不是随机化的抽象架构因果效应。

## 建议的下一步

1. 结束 LLaMA-124M 扩 seed；现有三 seed 已足以否定“GPT 的 diag 优势会原样迁移”这一简单假设。
2. 在查看 1B 正式结果前，先冻结 1B analysis contract 和 practical margin；这是现在必须完成、且不能补做的步骤。
3. 按既定顺序进行 1B 0/1-step probe、34-step smoke、fixed-memory/OOM 容量实验，再决定 medium/formal 放行。
4. 1B 质量主比较至少保留 `down_diag`、`down_none` 和 `muon`；`newton_full` 是否进入长跑，应由 fixed-memory 容量结果和是否需要完整状态对照共同决定。AdamW 可保留为低成本 sanity baseline，但当前 124M 结果不支持把它当竞争方法。
5. 论文表述使用“architecture-associated interaction / architecture-dependent optimizer behavior”，不要写成“none 在 LLaMA 上获胜”或“核心四方法已被证明等价”。

## 产物索引

- `llama_multiseed_run_summary.csv`：15 个 run 的终点、tail、AUC、显存和证据字段。
- `llama_multiseed_method_summary.csv`：五方法三 seed 聚合。
- `llama_multiseed_pairwise_by_seed.csv` / `llama_multiseed_pairwise_summary.csv`：配对差及 t 区间。
- `architecture_interaction_by_seed.csv` / `architecture_interaction_summary.csv`：GPT 与 LLaMA 架构交互。
- `llama_multiseed_mean_validation_curves.csv`：三 seed 平均验证曲线。
- `llama_multiseed_steps_tokens_to_fixed_targets.csv` / `llama_multiseed_fixed_target_summary.csv`：此前固定 targets 4.0/3.8/3.6/3.5/3.4 的逐 seed 与聚合效率。
- `data_quality_checks.csv`、`source_manifest.csv`、`analysis_manifest.json`：审计与来源。
- `analyze_multiseed.py`：可复现分析脚本。
