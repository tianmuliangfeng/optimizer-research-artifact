# 机制收尾报告（2026-08-05）

## 最终裁决

机制实验线在此收尾，不启动 GEO-01C，也不再以“补救”名义增加 GPU 机制实验。当前证据支持一个清晰但克制的结论：**在冻结的 LLaMA-1B 匹配回放树内，scheduled down-projection refresh 会稳定地产生短时 held-out loss impulse；但我们测试的单一 shock 标量和局部方向几何，均没有成为跨 checkpoint origin 稳健的定量解释器。**

这不是“机制实验没用”。正结果回答了 *where/when the harm is injected*；负结果回答了 *which tempting scalar explanations should not be claimed*。对论文而言，这比把探索性 pooled correlation 包装成机制更可靠。

## 证据完整性

- GEO-01B ZIP、12 个本地冻结输入均通过精确 SHA-256 检查。
- GEO-01B handoff 中 73 个文件全部通过字节数与 SHA-256 检查；12/12 discovery units、24 outcome rows、96 geometry rows 完整。
- 关键 GEO-01B 统计由本地脚本从 JSONL 独立重算，并逐项与远程 analysis manifest 对照通过。
- GEO-01A 只承担工程可行性角色；其 ZIP 当前不在本地，因此本包明确记录为“accepted lineage only / not reverified”，不用于科学收尾结论。
- MDP-04 的 12/12 单元已恢复，但原冻结数值 gate 失败（6/432 residual rows > 0.01），因此只作为诊断与复现教训，不作为正式证据。

## 稳健的正发现：refresh harm 重复出现

| 数据包 | 事件 | 正向单元 | normalized harm mean | median | range |
|---|---|---:|---:|---:|---:|
| MDP-05 | production refresh | 12/12 | 0.004484 | 0.004143 | [0.002929, 0.006641] |
| MDP-05 | delayed refresh | 12/12 | 0.002325 | 0.002292 | [0.001711, 0.003088] |
| GEO-01B | production refresh | 12/12 | 0.004580 | 0.004526 | [0.002918, 0.006555] |
| GEO-01B | delayed refresh | 12/12 | 0.002396 | 0.002341 | [0.001770, 0.003230] |

两套独立包在两个事件上都得到 12/12 同号，而且 mean 非常接近。这使“refresh 本身在冻结树内造成短时 loss harm”成为可以保留的因果证据；它不等价于对所有模型、训练阶段或优化器的普遍定律。

## 局部曲率究竟告诉了我们什么

GEO-01B 的 full Taylor 对即时 counterfactual line loss 的拟合非常准：production 的中位相对误差为 0.010415%，delayed 为 0.027509%；相对一阶近似，加入曲率后本地误差分别下降 99.728% 与 99.646%。

但这只说明 Hessian-vector product 与 Taylor 分解在**即时、固定 batch、给定方向**上数值闭合。它没有转化成跨 origin 的 16-step endpoint harm 预测：GEO-01B 的 confirmation candidate=false，curvature increment 也不满足门槛。因此“局部曲率能预测后续训练伤害”必须列为禁止表述。

## 为什么 pooled correlation 不能升级为机制

GEO-01B production full-Taylor 的 pooled Spearman 很高（0.853），但 origin-centered Spearman 只有 0.168；delayed 分别为 0.434 与 0.063。相反，简单 norm 在 production 的 centered Spearman 是 0.720，说明更复杂的 Taylor 标量没有在关键的 within-origin 检验中稳定胜出。

checkpoint origin 对 endpoint harm 的描述性解释比例为 production 95.9%、delayed 98.0%；两个事件的 unit-level harm 相关为 r=0.963。MDP-05 也得到约 96% 的 origin share。最合理的解释是：当前 pooled 排序主要携带 checkpoint/stage/method 状态，而不是一个可转移的单标量局部机制。

## 论文使用边界

正文可简洁表述 refresh loss impulse 的冻结树内因果证据；详细数据放 appendix。MDP-05 的 null mediation 与 GEO-01B 的 failed origin-independent gate 应主动进入 appendix/limitations，显示我们对机制主张做了强检验。

不得声称：普适 refresh 定律；局部曲率预测多步 harm；已经得到自动 layer selector；GEO-01B 是 confirmatory evidence；或机制已被完全解释。逐条措辞见 `claim_boundary.csv`。

## 后续动作

1. 冻结本目录，论文机制段落只从本包取数。
2. 不运行 GEO-01C；新的 trajectory/state-dependent 解释只能作为未来独立项目。
3. 下一项单独评估 LLaMA-1B 10B-token 实验的科学增益、计算成本与对投稿风险的净贡献；该决定不属于本机制包。
