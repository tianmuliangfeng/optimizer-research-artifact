# 实验 45 大纲：controlled 124M R1 Mousse 强近邻审计

> 状态：只冻结科学设计，尚未编写实验代码。  
> 依据：`docs/RELATED_WORK_COLLISION_REVIEW_20260730.md` 与实验 19 的既有
> 124M R1 扩展基线结果。  
> 命名：本文统一使用官方名称 `NorMuon`。  
> 边界：本实验不修改实验 43、44，也不触碰 `HANDOFF.md`。

## 1. 纠正后的实验定位

实验 45 不是 275M/455M 的通用强基线扩展，也不是重新运行 Moonlight 和
NorMuon。它只在已经完成且证据最完整的 **124M controlled R1** 平台上新增
一个外部强基线：

> **Mousse: Rectifying the Geometry of Muon with Curvature-Aware
> Preconditioning**。

Mousse 是碰撞审查中识别出的最强技术近邻。它使用 Shampoo/Kronecker
梯度统计构造白化坐标中的 Muon 更新；本文方法使用输入 activation second
moment，并只对 contraction `c_proj` 的 Newton–Muon K-state 选择
`block4`、`diag` 或 `none`。实验 45 的核心问题是：

> 在完全对齐的 124M R1 训练合同下，加入 Mousse 后，Selective
> Newton–Muon 相对 Muon 和原始 Newton–Muon 的结论是否仍成立？

实验 45 是论文的 **external-neighbor robustness panel**。它不能替代 Muon
和原始 Newton–Muon 两个核心 control，也不能把论文主问题改写成
Selective 方法与 Mousse 的单一胜负。

## 2. 已有基线及其角色

实验 19 已经在同一 124M R1 设置中完成 Moonlight、NorMuon 和 AdamW 的
三 seed、6200-step 正式实验，因此实验 45 不重跑这些方法。

既有 step-6200 validation loss 如下：

| 方法 | 三 seed mean ± seed SD | 角色 |
|---|---:|---|
| Newton–Muon `diag` | 3.261100 ± 0.001114 | Selective 主方法 |
| 原始 Newton–Muon `block4` | 3.262200 ± 0.001136 | 原始方法 control |
| Newton–Muon `none` | 3.266667 ± 0.000666 | Selective 主方法 |
| Moonlight Muon | 3.274967 ± 0.000451 | 已有现代 Muon 强基线 |
| Muon | 3.277133 ± 0.000252 | 核心外部 control |
| NorMuon | 3.334467 ± 0.000924 | 已有谱优化器基线 |
| AdamW | 3.400433 ± 0.004692 | 标准一阶基线 |

该排序在 confirmatory seeds 2024/2025 上保持不变。Moonlight 在最终端点上
比 Muon 平均低 0.002167，但优势只在 terminal warmdown 最后两个验证点出现；
它不是全程支配 Muon。NorMuon 明显落后于 Muon 和三个 Newton–Muon 路由。

上述结果只在以下 identity certificate 全部通过时进入实验 45 的联合表：

- 模型架构、参数量、初始化与 seed 一致；
- FineWeb10B shards、数据顺序、batch、sequence length 和 token budget 一致；
- 6200-step schedule、1800-step terminal warmdown 和 validation grid 一致；
- auxiliary parameter routing/optimizer、packed-QKV 逻辑和 runtime family 一致；
- 每个复用 run 的 source、manifest、曲线及终点数据可追溯。

如 identity audit 失败，相关历史行只能作为非配对背景证据，不得与 Mousse
形成 paired claim。实验 45 默认只补必要的 matched control，不重建整个
optimizer zoo。

## 3. 实验方法集合

### 3.1 唯一新增训练方法

- `mousse`

### 3.2 复用的核心比较方法

- `muon`
- `original_newton_muon`，即 124M 官方 `c_proj=block4` 路由
- `selective_diag`
- `selective_none`

### 3.3 复用的外部背景基线

- `moonlight`
- `normuon`
- `adamw`

Moonlight 和 NorMuon 用于说明 Mousse 相对现有现代优化器基线的位置；它们
不是实验 45 新增的训练任务。SOAP-Muon、NAMO-D、SkewAdam 和其他候选方法
不自动进入本实验。

## 4. 45A：Mousse 来源、实现与数值审计

正式训练前必须完成以下审计：

1. 冻结 Mousse 论文版本、官方代码 URL、commit、license、配置与核心文件
   SHA-256，并保存 source snapshot、适配 patch 和 hash manifest。
2. 将实现明确标记为 `Mousse-R1 adaptation`：只把 Mousse hidden-matrix
   update 移植到现有 124M R1 scaffold；模型、数据、初始化、schedule、
   validation、auxiliary optimizer 和 packed-QKV 逻辑保持 R1 合同。
3. 除非与官方代码逐项一致，不得写成 `unchanged official reproduction`。
4. 从固定官方来源一次性冻结 trace normalization、spectral tempering、
   damping、factor EMA、preconditioner interval、grafting、update ordering
   和状态 dtype；除 matrix LR 外不展开联合超参数搜索。
5. 用小矩阵 reference test 审计单步与多步更新、factor/inverse 数值、
   route coverage、状态 schema/bytes、有限性、refresh 计数和更新顺序。
6. 证明 Mousse cell 中不存在 Newton–Muon activation-K state 泄漏。
7. 审计初始化 hash、loader/RNG、warmup reset、模型梯度清空和训练前后
   parameter routing。
8. smoke 必须跨过第一次 Mousse preconditioner refresh，并至少完成一次
   refresh 后参数更新。

优先使用同一 R1 scaffold 来隔离 optimizer 本身。如果只能使用 Mousse 官方
scaffold，且 parameter grouping、auxiliary optimizer 或更新顺序与 R1 不同，
则必须在该 scaffold 内重跑 matched Muon；不能把不等价的 Mousse 结果与历史
R1 Muon 做配对比较。

## 5. 45B：冻结的三点 LR pilot

- 仅使用 seed2026；
- 最多三个预注册 matrix LR：官方映射中心值的 `0.8× / 1.0× / 1.2×`；
- 其余 Mousse 超参数全部冻结；
- 使用实验 19 的 1000-update R1 pilot budget 和相同 validation protocol；
- 只按 step-1000 validation loss 选择一个 formal LR；
- 若最佳两个候选落入 `±0.002`，优先选择官方中心值；
- 不按 wall-clock、显存或对主方法是否有利来选择；
- pilot 后不得追加 damping、interval、EMA、grafting 或其他网格。

seed2026 是 tuned-seed screen；seeds 2024/2025 才是未参与选参的
confirmatory seeds。最终论文必须披露这一非对称角色。

## 6. 45C：124M 三 seed 正式实验

冻结配置：

- 模型：Record #4-derived 124M Modded-NanoGPT GPT；
- 协议：controlled R1；
- 数据：与现有 R1 完全相同的 FineWeb10B shards 和顺序；
- 方法：唯一 pilot-selected Mousse 配方；
- seeds：2026、2024、2025；
- updates：6200；
- tokens/update：`512 × 1024`；
- 总训练 tokens：`3,250,585,600`；
- validation：每 100 step；
- schedule：保留 1800-step terminal warmdown；
- 每 seed 正式训练前运行跨 refresh 的 exact-shape smoke；
- 本地 artifacts 先通过完整性检查，再上传 W&B；
- W&B 失败只重试上传，不重跑已经通过本地验收的训练。

实验 45 默认只新增 3 个正式 Mousse run，而不是重新运行全部八个方法。只有
identity audit 失败且缺少相应 matched control 时，才允许增加必要的 control
run，并必须在 manifest 中说明原因。

## 7. 预注册比较优先级

主分析继续遵守本文的比较逻辑：两个 Selective 方法分别对外部 control，而
不是强调 `diag-vs-none`。

### 7.1 实验 45 的 primary comparisons

- `selective_diag - mousse`
- `selective_none - mousse`

### 7.2 必须同时报告的 anchors

- `mousse - muon`
- `mousse - original_newton_muon`

### 7.3 外部基线背景比较

- `mousse - moonlight`
- `mousse - normuon`
- `mousse - adamw`

### 7.4 非主比较

- `selective_diag - selective_none` 只作描述；
- 不以 Mousse、Moonlight 和 NorMuon 之间的排名替代 Selective 分别对 Muon
  与原始 Newton–Muon 的主结论；
- 不根据 Mousse 的结果重新选择 diag、none 或原始方法的超参数。

## 8. 终点与统计合同

主要终点：

- step-6200 final validation loss。

次要终点：

- tail-5 validation loss；
- normalized validation AUC；
- best validation loss；
- optimizer state、factor/inverse/workspace bytes；
- peak allocated 和 peak reserved memory；
- 数值稳定性、refresh 次数和每个 route 的状态覆盖。

统计规则：

- seed 是推断单位；
- practical equivalence margin 固定为 `±0.002` validation loss；
- 对每个预注册差值报告三 seed 原始值、paired mean、sample SD、
  paired-t confidence interval 和方向计数；
- 不因 `n=3` 使用夸大的大样本显著性语言；
- formal quality run 的 wall-clock 只作诊断，尤其不得使用双卡并发训练时间
  作为 paper-ready throughput。

只有 Mousse 形成有竞争力的质量—状态点时，才允许追加一次隔离效率审计：
单 GPU、邻卡空闲、顺序平衡、固定测量窗口，并与实验 39/42 的系统测量合同
保持一致。

## 9. 结果解释矩阵

| 可能结果 | 论文解释 |
|---|---|
| Selective 优于或等效于 Mousse | 外部强近邻不能解释掉 contraction-selective activation-K routing 的收益 |
| Mousse 优于 Selective，但 Selective 状态明显更低 | 写成质量—状态 Pareto，不掩盖 Mousse 的质量优势 |
| Mousse 同时优于 Muon 和原始 Newton–Muon | 承认 gradient-space curvature hybrid 在该设置更强，并收紧性能主张 |
| Mousse 不优于 Muon | 只陈述 controlled 124M R1 的负结果，不能外推为 Mousse 普遍无效 |
| Mousse 与 Selective 接近 | 强调两条不同曲率路线形成相近质量点，再比较状态与系统成本 |

无论结果方向如何，Mousse 都必须进入论文的主实验或外部强近邻面板，并在
Related Work 中正面区分：

- Mousse：gradient-space Kronecker factors 与 whitened-coordinate Muon；
- 本文：activation-space K-state 与 contraction-specific full/diag/none routing。

不得使用“首次 curvature-aware Muon”或“Mousse 与本文无关”之类表述。

## 10. 停止规则

1. 默认在 124M 三 seed formal 与统一分析完成后停止。
2. Mousse 胜、平或负都不触发对 43/44 的修改。
3. 不因结果不利而追加 optimizer zoo、扩大超参数网格或选择性重跑旧基线。
4. 只有同时满足以下三个条件，才允许在实验 45 命名空间内增加一个 275M
   Mousse extension：
   - 所有 45 完整性检查通过；
   - Mousse 与预先存在的最佳 124M Selective（当前为 `diag`）在 final
     上落入 `±0.002`，且 tail-5/AUC 没有明确反向；
   - 缺少规模证据会实质改变摘要或主 claim。
5. 可选 275M extension 只增加与实验 43 对齐的 Mousse cells；不得修改实验
   43，也不自动扩展到 455M、LLaMA 或更多外部 optimizer。

## 11. 未来实现目录与交付物

正式实现时使用：

```text
scripts/45_r1_mousse_strong_baseline/
commands/45_r1_mousse_strong_baseline/20260730_r1_mousse_strong_baseline.sh
${SNM_RESULTS_ROOT}/
  45_r1_mousse_strong_baseline/
```

预期交付物：

- 固定实验 contract；
- source、patch、license 与 SHA-256 provenance；
- 45A reference tests 和实现审计；
- 45B pilot manifest、曲线和唯一 LR 选择证书；
- 45C 每 seed manifest、scalar CSV、完整 log、checkpoint metadata 和 W&B
  upload status；
- 历史行 identity/reuse certificate；
- 八方法统一 124M 表；
- 预注册 paired contrasts、状态—质量 Pareto 和论文可引用结论；
- 明确列出 evidence level、限制条件及未通过的任何检查。

本文件只定义实验大纲，不包含运行代码或远程实验指令。
