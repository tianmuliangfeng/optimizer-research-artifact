# GPT R1 on LLaMA-host bridge：seed2026 分析

生成时间：2026-07-22T09:39:00.798499+08:00  
数据状态：**PASS_WITH_CAVEATS**

## 技术摘要

- **跨硬件 bridge 几乎精确复现原 GPT R1 的 `diag-none` 差异。** 新主机上 final validation loss 为 diag **3.2620**、none **3.2669**，配对差为 **-0.0049**；原主机配对差为 **-0.0050**，只变化 **+0.0001**。
- **LLaMA seed2026 的方向反转不能由 H100 主机/驱动/runtime 差异解释。** 在同一 LLaMA 主机/runtime 上，GPT bridge 为 **-0.0049**，LLaMA/SwiGLU 为 **+0.001357**；架构相关的同主机残差为 **+0.006257**。
- **当前支持“architecture-associated interaction”，但还不是跨 seed 的架构定律。** bridge 只有 seed2026；W&B 导出完整，但本地 formal manifest、checkpoint、source/runtime/init fingerprint 尚未随本批数据提供。
- **timing 不可用于论文性能结论。** bridge 与 GPU0 LLaMA 训练并发；时间和 step-ms 仅保留作诊断。

## 关键结果

| Context | diag final | none final | diag-none final | diag-none tail-5 | diag-none AUC |
|---|---:|---:|---:|---:|---:|
| GPT R1 original host | 3.262100 | 3.267100 | -0.005000 | -0.005200 | -0.008674 |
| GPT R1 LLaMA-host bridge | 3.262000 | 3.266900 | -0.004900 | -0.004940 | -0.008628 |
| LLaMA/SwiGLU same host | 3.266387 | 3.265030 | +0.001357 | +0.001314 | +0.001759 |


## 硬件效应与架构效应的隔离

原始跨 regime 方向变化为 `+0.001357 - (-0.005000) = +0.006357`。加入 GPT-on-LLaMA-host bridge 后：

- host/runtime 对 GPT 配对差的变化：**+0.000100**；
- 同 host/runtime 下从 GPT 到 LLaMA 的剩余变化：**+0.006257**；
- 因此，主机/runtime 混杂只占观测方向变化的约 **1.6%**，其余约 **98.4%** 与架构/实现路径变化相关。

这里的百分比是 seed2026 的描述性分解，不是方差分析或因果效应估计。由于 GPT 导出保留到 4 位小数，`0.0001` 的 host 差异也接近日志舍入分辨率。

## 范围与指标定义

- 每个方法 6200 optimizer steps，每步 524,288 tokens，总计 3,250,585,600 tokens。
- primary endpoint：step-6200 validation loss。
- secondary：最后五个验证点均值、0–6200 trapezoidal normalized AUC、固定 loss 阈值的 steps/tokens-to-target。
- 配对差统一定义为 `diag - none`；负数表示 diag 更低、更好。
- bridge 只包含 GPT `diag` 与 `none`，符合预设最小设计。

## 数据质量与稳健性

- 9/9 指标导出齐全，run 集合精确为 diag/none seed2026。
- 每个 run 有 63 个 validation 点（0–6200，每 100 steps），310 个训练 loss 点和完整 LR schedule；无非有限值或重复 method/metric/step。
- 两个方法初始 validation loss 均为 10.979；AdamW LR 0.004、matrix LR 0.0004，step 4400 后按 1800-step warmdown 降至 0。
- 每种方法的 bridge-vs-original 曲线 RMSE、最大绝对偏差保存在 `bridge_vs_original_gpt.csv`。

## 限制

1. 只有 seed2026 bridge，不能估计 bridge 配对差的跨 seed 方差。
2. 当前输入只有 W&B CSV；在纳入论文最终证据包前，还需核对远端 formal manifest、checkpoint、official commit、derived source hash、runtime fingerprint、init hash、并发标记和 resume count。
3. LLaMA 与 GPT 的“架构相关残差”同时包含架构所必然带来的模型实现路径差异；它排除了本次所测 host/runtime 混杂，但不是对抽象架构变量的随机化因果估计。
4. 并发节点负载使 timing 永久不合格；显存 allocated 与 exact optimizer/K-state 可保留，但最好用本地 manifest/state bytes 最终对账。

## 建议

1. **无需补 GPT bridge seeds2024/2025。** 预设触发条件是方向改变或差距明显缩小；本次 `-0.0049` 与原 `-0.0050` 基本相同。
2. **继续收口 LLaMA seeds2024/2025。** 真正需要估计的是 LLaMA 上 `diag-none` 是否稳定接近零或反转。
3. 论文可以从“architecture/system regime dependent”升级为更精确的表述：**在所测 LLaMA host/runtime 上，GPT 仍复现 diag 优于 none，而 LLaMA seed2026 未复现，支持 architecture-associated interaction。** 三 seed LLaMA 完成前不要写成普遍架构结论。
4. 将远端 bridge artifact 目录同步回来，完成 manifest/W&B/checkpoint 对账后再标记为主表可用。

## 待回答问题

- LLaMA seeds2024/2025 的 `diag-none` 配对差是否继续围绕 0，还是回到 GPT 的负向差异？
- 三 seed LLaMA 的置信区间或配对分布是否排除 practical non-inferiority margin？
- bridge 的本地 state bytes、checkpoint 与 resume 证据能否和 W&B 逐项一致？
