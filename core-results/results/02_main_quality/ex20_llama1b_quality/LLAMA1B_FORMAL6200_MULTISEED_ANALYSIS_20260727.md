# LLaMA/SwiGLU-1B formal-6200 三 seed 合并分析（2026-07-27）

## 结论先行

Muon 在 seed2024、2025、2026 的冻结 step-6200 主终点和 tail-5 均为四方法第一，因此 seed2026 的反转不是孤立 seed。三种 Newton 路径在 step1000 仍全部优于 Muon，但在约 step1400--2500 的 seed/method 相关窗口内被 Muon 持续反超。当前固定 recipe 下，1B 结果明确不支持 Selective Newton-Muon 相对 Muon 的 Pareto 改进。

家族内结论则更稳定：down-none 与 down-diag 在三个 seed 的最终 loss、tail-5 和 AUC 上均优于 Newton-full。去掉或对角化 down-projection K 没有损害 full Newton-Muon 的结果，反而改善了结果并减少了 K-state；这部分是当前 1B 数据真正 支持的正结论。

## 证据完整性

- 数据检查：PASS=235，FAIL=0，WARN=4。
- 每个 run 完成 6200 updates，即 3,250,585,600 tokens（3.2506B）。
- 3 seeds × 4 methods × 7 metrics 均齐全；没有缺 run、截断曲线或 endpoint 插补。
- 每个 seed 内四方法 step0 validation loss 完全相同；12 条 run 的 matrix/backup LR 和 tokens/step 逐点一致。
- W&B 导出足以形成 loss-vs-step/token 的正式质量证据，但不含正式 manifest、checkpoint/resume 证书和实测显存字段；这些仍需用远程 compact artifacts 对账。
- 双卡并行条件下 time/train_s 与 step_avg_ms 只作描述，不进入论文性能结论。

## 三 seed 聚合

| 方法 | Final mean ± SD | Tail-5 mean ± SD | AUC mean ± SD | Final 3-seed rank |
|---|---:|---:|---:|---:|
| muon | 2.970125 ± 0.000954 | 2.978224 ± 0.000919 | 3.370644 ± 0.001550 | 3/3 wins |
| down_none | 2.974596 ± 0.000848 | 2.982745 ± 0.000841 | 3.371167 ± 0.001945 | 0/3 wins |
| down_diag | 2.975748 ± 0.000622 | 2.983864 ± 0.000636 | 3.369640 ± 0.001465 | 0/3 wins |
| newton_full | 2.976712 ± 0.000517 | 2.984772 ± 0.000524 | 3.372386 ± 0.002498 | 0/3 wins |

## 配对差值（正数表示该方法比参考方法更差）

| 对比 | Final delta mean ± SD | Tail-5 delta | AUC delta | Final 同方向 seed |
|---|---:|---:|---:|---:|
| down_none-muon | +0.004471 ± 0.000805 | +0.004521 | +0.000523 | 0/3 negative; 3/3 positive |
| down_diag-muon | +0.005623 ± 0.000944 | +0.005640 | -0.001004 | 0/3 negative; 3/3 positive |
| newton_full-muon | +0.006587 ± 0.001176 | +0.006548 | +0.001742 | 0/3 negative; 3/3 positive |
| down_none-newton_full | -0.002116 ± 0.000645 | -0.002027 | -0.001219 | 3/3 negative; 0/3 positive |
| down_diag-newton_full | -0.000964 ± 0.000330 | -0.000908 | -0.002746 | 3/3 negative; 0/3 positive |
| down_diag-down_none | +0.001152 ± 0.000315 | +0.001119 | -0.001527 | 0/3 negative; 3/3 positive |

三个 Newton 变体相对 Muon 的 final delta 分别为：down-none +0.004471、down-diag +0.005623、Newton-full +0.006587。三者在每个 seed 都超过冻结的 0.0020 practical margin，因此不能称为 “相对 Muon 基本无损”。

down-diag 相对 down-none 的最终 loss 平均高 0.001152，但 AUC 平均低 0.001527。这说明 diag 更偏早期优化，none 更偏冻结终点；两者不是简单的全程单调排序。

## 对论文叙事的含义

1. 可以写：在 1B、3.25B-token 固定 recipe 下，Muon 的后期质量优势跨 3 seeds 稳定复现；Newton 的早期优势会发生训练阶段反转。
2. 可以写：对 Newton-Muon 家族，down-projection K 的边际价值很低，down-none/down-diag 以更少状态稳定优于 full。
3. 不可以写：Selective 在 1B 相对 Muon 质量无损、优于 Muon，或 Newton-full 是质量上界。
4. 当前结果把下一步重点从继续堆长跑，转向定位反转机制，以及独立的 Newton LR/ridge/refresh 探索分支；调参结果不能替换冻结主表。

## 尚缺的本地证书

请补充 seed2024、seed2025 与 seed2026 三个 formal-6200 批次的 `llama_manifest.json`、`llama_plan.json`、`llama_swiglu_summary.csv`，以及四个 run 的 `summary.json` 和 `metrics.csv`。不需要上传约 10GB 的 checkpoint；manifest 中的 checkpoint 路径、大小、hash/resume 元数据即可。若 summary 含 optimizer/K-state/peak CUDA memory，也可完成曲线与实测内存的对账。

## 逐 seed 主终点

| Seed | Muon | down-none | down-diag | Newton-full |
|---:|---:|---:|---:|---:|
| 2024 | 2.971026 | 2.975529 | 2.976336 | 2.976945 |
| 2025 | 2.970222 | 2.973872 | 2.975097 | 2.976119 |
| 2026 | 2.969126 | 2.974385 | 2.975809 | 2.977071 |
