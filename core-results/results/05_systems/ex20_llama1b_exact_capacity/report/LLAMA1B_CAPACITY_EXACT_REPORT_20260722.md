# LLaMA-SwiGLU 1B 精确显存容量边界

## 技术摘要

预注册的四个奇数 batch confirmation 全部按计划完成：`newton_full@35` 发生真实 CUDA OOM，`down_none@37`、`down_diag@37` 和 `muon@39` 均完整运行 34 steps。合并父细扫端点后，四种方法的最大成功 device batch 分别为 **34、37、37、39**，首个 OOM 分别为 **35、38、38、40**，边界均精确到 1。

相对 dense Newton，`down_none/down_diag` 将最大 microbatch 从 34 提升到 37，即 **+3 sequences / +8.82%**；Muon 达到 39，即 **+14.71%**。压缩方案与 Muon 仍相差 2 sequences，但明显缩小了 dense Newton 的容量劣势。

## 精确边界

| 方法 | 最大成功 device batch | 首个 OOM | 最大成功 global batch | 相对 Newton-full |
|---|---:|---:|---:|---:|
| Newton-full | 34 | 35 | 272 | baseline |
| down-none | 37 | 38 | 296 | +8.82% |
| down-diag | 37 | 38 | 296 | +8.82% |
| Muon | 39 | 40 | 312 | +14.71% |

## Confirmation cells

| 方法 | 测试 batch | 结果 | completed steps | peak allocated (MiB) | peak reserved (MiB) |
|---|---:|---|---:|---:|---:|
| Newton-full | 35 | CUDA OOM | — | — | — |
| down-none | 37 | success | 34 | 76,115.93 | 79,234 |
| down-diag | 37 | success | 34 | 76,117.00 | 79,262 |
| Muon | 39 | success | 34 | 76,677.34 | 80,472 |

不同方法的 confirmation cell 使用了不同 device/global batch，因此上表峰值只用于证明各自边界，不用于同 batch 显存差值。`down_none/down_diag` 相对 full 和 Muon 的主显存对照仍应引用共同 batch 32 的细扫结果。

## 数据质量与结论边界

- exact controller 状态为 completed；4 个计划 cell 各有唯一结果。
- 三个成功 cell 均完成 34 steps，并覆盖 step-32 K refresh；唯一失败为可解析的 `torch.OutOfMemoryError`。
- 每个 cell 开始前可用显存完全一致（84,620,541,952 bytes），GPU 总显存也完全一致。
- exact manifest 记录的父 manifest SHA-256 与本地归档的细扫 manifest 完全匹配。
- `newton_full@35` OOM 于一次 3.36 GiB 大块分配；失败时报告 2.14 GiB free、74.10 GiB allocated、2.36 GiB reserved-but-unallocated。它仍是当前 allocator/编译配置下的操作性边界，不是硬件理论上限。
- 本实验固定 accumulation=8，global batch 随 device batch 改变，因此属于 `capacity_only`；loss、时间和吞吐量不能进入质量或性能比较。

## 决策

核心 1B OOM 实验在这里收口。当前证据已经给出 full、none、diag、Muon 的精确整数边界。以后只有在 LLaMA-1B 正式加入 AdamW、Moonlight 或 NorMuon 等额外优化器时，才为对应方法单独增加匹配的细粒度容量附录；不回溯改变当前核心四方法的主结论。
