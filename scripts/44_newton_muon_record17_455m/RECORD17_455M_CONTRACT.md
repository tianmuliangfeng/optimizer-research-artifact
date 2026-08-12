# 实验 44：Modded‑NanoGPT Medium Track Record #17（455M）实验合同

`record17_contract.json` 是机器可读的权威合同。本文档用于人工审阅；实验开始后不得依据中途结果修改方法、种子、预算、目标 loss 或对比优先级。

## 1. 目的与表述边界

实验在同一份 Record #17 单 H100 适配配方下比较：

- `muon`
- `original_newton_muon`
- `selective_none`
- `selective_diag`

主问题是：`selective_none` 和 `selective_diag` 是否分别优于 Muon 和原始 Newton–Muon。原始 Newton–Muon 对 Muon 是必须报告的 benchmark anchor；`diag` 对 `none` 仅为次要描述性比较。

论文应称其为“upstream Modded‑NanoGPT Medium Track Record #17 recipe 的单 H100 适配”。上游使用 8 张 H100，本实验通过单卡 8 次梯度累积保持原始 global batch，因此不能称为 unchanged official reproduction。并发运行所得 wall time、tokens/s、steps/s 不进入效率结论。

## 2. 冻结来源

Record #17 历史源：

| 字段 | 值 |
| --- | --- |
| repository | `https://github.com/KellerJordan/modded-nanogpt` |
| commit | `9e7218468ea864a33053142c196d90bbf3ed48e1` |
| path | `records/track_2_medium/2025-11-12_BlockMaskRedundantOp/train_gpt_medium.py` |
| Git blob | `8504813a5ba0b1bf981fd6ad9d6348bfa1754b0f` |
| canonical LF SHA‑256 | `03d91174eed5e8cbf57063a1e997eb98570dde7a09ba9b2c94aa36e9d5eb94cb` |

仓库内 vendored 文件必须通过上述哈希，随后才允许生成训练程序。`Newton-Muon-official-r0` 的 commit `df78af0db523d8bceb25af4919a3e3e7082b80f3` 只承担 Newton 实现与 FineWeb10B 数据来源审计；它不包含 455M Record #17 trainer。

### Runtime provenance amendment

The immutable upstream execution log
`records/track_2_medium/2025-11-12_BlockMaskRedundantOp/000_3b22a9d4-b52e-4916-99bf-3d48b38747a7.txt`
records Python `3.10.12`, PyTorch `2.7.0+cu126`, CUDA build `12.6`, and
driver `560.35.03`; this remains upstream provenance rather than the execution
contract for our controlled adaptation. Experiment 44 uses the existing pinned
runtime `${SNM_TRAINING_PYTHON}` with PyTorch
`2.8.0+cu126` and Triton `3.4.0`, because the added FP32 activation-statistics
GEMMs use the PyTorch 2.8 `out_dtype` API.

The generated unified source count-checks one compiler-policy change:
`torch._dynamo.config.compiled_autograd = False`. FlexAttention,
`torch.compile(model)`, model/data/optimizer mathematics, and all four methods'
shared execution domain remain enabled and unchanged. This avoids the observed
PyTorch 2.8 nested FX/FlexAttention failure, which happened before any counted
update and is not scientific evidence.

## 3. 模型、数据和预算

- 模型：16 层、宽度 1024、8 heads、MLP 宽度 4096，第 7 个 block 跳过 attention。
- 精确参数量：454,496,336。
- 数据：50 个 FineWeb10B train shards、1 个 validation shard，逐文件完整 SHA‑256。
- 正式 seeds：2024、2025、2026。
- 正式 cell：3 seeds × 4 methods = 12。
- optimizer updates：5,960。
- 每个 update：524,288 tokens。
- 总训练量：3,124,756,480 tokens。
- 单卡 microbatch：65,536 tokens，gradient accumulation = 8。
- validation：10,485,760 tokens；step 0、每 125 steps 及最终 step 5960，共 49 点。

单卡 loader 必须先读取与原 8-rank 执行相同的 524,288-token global batch，再切成 8 个连续 microbatch。禁止连续调用 8 次 local-size loader，因为 shard 末尾丢弃规则会改变数据顺序。

Newton 家族的 K 统计固定聚合这 8 个顺序 microbatch，即每个 counted update
使用全部 524,288 tokens 的激活。本规则是单 H100 适配的一部分；它不声称严格
等价于一个假设的 8-rank Newton 实现中“参数 owner 只使用本 rank 激活”的
K 统计路径。四种方法仍共享完全相同的数据、预算和单卡执行域。

## 4. 优化器与 K 合同

Record #17 配方保持：

- AdamW：head LR `1/320`，embedding LR `0.3`，scalar LR `0.015`，betas `(0.8, 0.95)`，epsilon `1e-10`。
- Muon：LR `0.025`，momentum 在前 300 updates 从 `0.85` 升至 `0.95`，weight decay `0.01`，MLP 矩阵乘数 `2`。
- LR schedule：前 30% 恒定，后 70% 线性降至 0；所有 smoke 也使用正式 5960-step 分母。
- Newton：`beta=0.9`、`ridge=0.2`、K 初始值 `1e-3 I`、每 24 updates 刷新；正式运行应刷新 248 次。
- inverse 与 K 对 raw gradient 的应用都必须在 FP32 完成，并且发生在 momentum 与 Newton–Schulz 之前。

三种 Newton 家族方法除 MLP contraction `proj_w` 的 K 表示外完全相同：

- `original_newton_muon`：四个对齐的 `1024×1024` block。
- `selective_none`：仅删除 contraction K。
- `selective_diag`：四个对齐的 1024 维 diagonal scale。

QKV、O 和 MLP expansion 的 K 路由不得改变。Muon 不得分配任何 K state。

## 5. warmup、smoke 与验收

instrumentation warmup 为 26 updates，用于编译并执行首次 refresh；随后必须恢复 model、AdamW、Muon、K、activation accumulators、loader、RNG 和 gradients，清空缓存并重置 peak memory。正式计数从 update 0 开始。

smoke 固定 seed 2026、四方法、27 counted updates，必须跨过 update 24 的首次 refresh。smoke 通过后才能启动正式 cell。

每个正式 cell 必须：

- 产生 49 个精确 validation 点；
- 完成 5,960 updates 和精确 token 预算；
- 通过初始化配对、数据指纹、source hash、FP32 application、K schema、有限性和 warmup rollback 检查；
- 保存并哈希 `state_step005960.pt`；
- 本地科学证据先封存，再由 controller Python 上传 W&B。

动态 CUDA 峰值的口径固定为
`counted_run_after_warmup_reset_including_validation`：从 warmup 完整回滚并
重置峰值统计之后开始，包含所有周期验证和最终验证。不得把它表述为仅训练
step 的峰值。

## 6. 统计合同

主指标为 step 5960 的 final validation loss。每个 Selective 方法分别对 Muon 和 original Newton–Muon做 seed 配对差值；差值定义为 candidate minus comparator，负值更优。

- `n=3`，95% paired Student‑t CI 使用 `df=2`。
- final loss practical margin 固定为 `±0.002`。
- common target 固定为 validation loss `2.95`，只作为次要轨迹效率指标；用相邻 validation 点线性插值首次向下穿越。
- `diag_vs_none` 不得触发主结论或独立扩 seed。
- 只有主对比不确定性会改变跨尺度论文结论时，才允许成套追加四方法 seed 2027，必要时再追加 seed 2028；不得自动追加。

分析通过只表示数据、配对和完整性合同通过，不等于任何算法被判定为更优。
