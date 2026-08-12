# 实验 43：约 275M Modded‑NanoGPT、near Record #28 质量实验合同

`record28_contract.json` 是机器可读的权威合同；本文档用于人工审阅。实验开始后，不得依据中途结果修改方法、种子、预算、主指标或对比优先级。

## 1. 研究问题与表述边界

实验比较四种固定方法：

- `muon`
- `original_newton_muon`
- `selective_none`
- `selective_diag`

核心问题是：在上游 Newton–Muon‑2 配方附近，Selective‑none 和 Selective‑diag 是否分别优于 Muon 与原始 Newton–Muon。

论文中应表述为“约 275M 参数的 Modded‑NanoGPT，采用接近上游 Short Track Record #28 的配方”，不能称为 Record #28 的精确复现。原因是目标主机使用 H100 而非上游 L40S，并加入了新方法、配对多种子统计、来源审计和本地证据封存。

并行运行只可支持质量和显存结论。并发产生的 wall time、tokens/s 和 steps/s 不具备论文效率证据资格。

## 2. 上游来源

远程仓库固定为：

```text
${SNM_OFFICIAL_REPO}
```

提交固定为：

```text
df78af0db523d8bceb25af4919a3e3e7082b80f3
```

控制器从 Git blob 读取上游源码，不直接执行带 CRLF 变化的工作树文件。LF 规范化哈希为：

| 文件 | SHA‑256 |
| --- | --- |
| `train_gpt_muon_2.py` | `04e25b21067e55247e603b09c90dcce2616f3418f41b5480dc9bc6ed8ed781c7` |
| `train_gpt_newton_muon_2.py` | `d30d31e3a01a18ea19050ea8aba04609d4825af2d26594955c148af83b07c4b6` |
| `triton_kernels.py` | `b51ac50c699b05306619d92cb9ec6edadd266d8118c53f5b9726db76480ea16d` |
| `data/cached_fineweb10B.py` | `adcc9f7d81ed1ac115a66d08d94d8d3e5c7425cabaf856da1f1fb106af87d09b` |

## 3. 模型、数据与训练预算

模型为 Modded‑NanoGPT：12 层、6 个注意力头、宽度 768、MLP 宽度 3072、padded vocabulary 50304，第 7 个 block 跳过 attention，精确参数量为 275,743,572。

FineWeb10B 数据必须恰好包含 50 个训练 shard 和 1 个验证 shard。preflight 会校验文件名、绝对路径、二进制 header、大小和每个文件的完整 SHA‑256，并生成所有 cell 共享的数据指纹。

正式训练固定为：

- seeds：2024、2025、2026、2027；
- 每个 seed 跑四种方法，共 16 个正式 cell；
- 单卡 train sequence length：49,152；
- gradient accumulation：8；
- 每次 optimizer update：393,216 tokens；
- 1,695 次 update，合计 666,501,120 个训练 tokens；
- validation 使用 10,485,760 tokens；
- 在 step 0、每 50 steps 和最终 step 1695 验证，共 35 个点；
- compile warm-up 为 17 次 update，随后恢复模型、优化器、数据加载器、RNG、梯度、K 累积器和 preconditioner step，再从 step 0 开始计数训练。

优化器配方固定为：

- DistAdam：LR 0.008，betas `(0.8, 0.95)`，epsilon `1e-10`；
- Muon matrix LR：0.05；
- momentum：前 300 updates 从 0.85 线性升到 0.95；
- LR：前 55% 保持，后 45% 线性降到 0.1 倍；
- Newton K：每 16 updates 刷新，EWMA 0.8，初始对角 `1e-3`，ridge multiplier 0.2。

## 4. 方法差异

`muon` 必须来自上游 Muon‑2 trainer，不得以 Newton 配方的 all‑none 版本替代。

另外三种方法共享同一个由环境变量选择模式的 Newton‑Muon‑2 派生程序：

- `original_newton_muon`：attention 与 `c_fc` 使用 full K，`c_proj` 使用上游 block4 K；
- `selective_none`：仅移除 `c_proj` K；
- `selective_diag`：仅将 `c_proj` K 替换为四个 block 对齐的 diagonal scales。

三种 Newton‑Muon 方法唯一允许的算法差异是 `c_proj` 的 K 表示。

## 5. smoke、环境与并行

正式训练前，seed 2026 的四种方法各运行 18 个 counted updates。smoke 必须跨过第 16 个 update 的首次 K refresh，并使用 1,695‑step 正式调度的前 18 步，不能因缩短循环而重缩放 LR、momentum 或 attention window。smoke 仅是完整性门禁，不进入质量统计。

远程环境固定为：

```text
controller: ${SNM_CONTROLLER_PYTHON}
training:   ${SNM_TRAINING_PYTHON}
GPU:        NVIDIA H100 80GB HBM3
Torch:      2.8.0+cu126
Triton:     3.4.0
```

controller 负责 preflight、调度、分析和 W&B；training 环境不需要安装 W&B。每个训练子进程只看见一张物理卡，因此其日志显示 `cuda:0` 是正常现象。

实验 43 默认只占物理 GPU0，可与占用物理 GPU1 的实验 44 并行。两者必须使用共享物理 GPU lock：

```text
${SNM_RESULTS_ROOT}/.physical_gpu_locks
```

## 6. 来源快照、恢复与 W&B

首次启动时，live repository 只负责创建并封存 `RUN_DIR/source_snapshot`。其中包含 controller、worker、analyzer、测试、合同、上游 Git blobs、四个派生 trainer、diff 和完整文件哈希。此后同一 run 的执行与恢复均以该快照为权威；live source 的后续变化只能进入审计备注。

恢复规则：

- 已完成 cell 只有在本地 manifest、artifact hash 和预注册身份全部通过时才复用；
- 中断 cell 原样保留，并从初始化开始建立新的编号 attempt；
- 不从半程 checkpoint 恢复；
- 恢复 shell 优先直接调用 run 目录中的快照 controller，不依赖当前 live Python 文件。

本地科学证据先封存，随后完成 16‑cell 配对分析，最后才尝试 W&B 上传。单次上传有超时上限；W&B 断网、超时或 controller 被终止都只能形成待上传清单，不能阻断分析或触发训练重跑。`offline` 与正式实验中的 `disabled` 均不得被标记为在线上传完成。

正式 checkpoint 只保存 raw model `state_dict` 和哈希，不宣称可以恢复优化器；大文件可留在远程主机，不要求进入交接包。

## 7. 统计优先级

主指标为 step 1695 的 final validation loss。四个主对比是：

1. `selective_none - muon`
2. `selective_none - original_newton_muon`
3. `selective_diag - muon`
4. `selective_diag - original_newton_muon`

负差值表示 candidate 更好。`original_newton_muon - muon` 是必须报告的 benchmark anchor；`selective_diag - selective_none` 只作次要描述。

每个对比以 seed 为配对单位，报告四个原始差值、paired mean、样本标准差、df=3 的双侧 95% paired‑t 区间、方向计数和 practical‑margin 分类。final-loss practical margin 固定为 ±0.002。

次要指标包括 normalized validation AUC、最后 5 个验证点均值、best validation loss、共同目标 loss 3.30 的 steps/tokens（仅当所有方法满足资格）、K/优化器状态字节和峰值 CUDA allocated/reserved bytes。

只有冻结分析触发 ambiguity gate，且该不确定性会改变跨尺度论文结论时，才允许按顺序增加 seed 2028、2029；最多增加两个 seed，且四种方法必须一起补跑。

## 8. 验收含义

`passed=true` 只表示来源、环境、数据、初始化配对、16 个正式 cell、训练预算、35 个验证点、K-state schema、checkpoint hash、本地 artifact 和统计重算全部通过完整性验收；它不表示任何算法已经获胜。
