# R1 三种子正式实验与完整学习率 factorial 分析（2026-07-21）

## 证据状态

- 新导出的 9 个 W&B 指标覆盖 seed2024/2025 × 4 方法，共 8 个正式 run。
- 与已审计 seed2026 合并后，正式 R1 为 3 seeds × 4 methods = 12 个 run。
- W&B 数据质量检查：PASS=180，WARN=0，FAIL=0。
- 当前结论为 `PASS_WITH_CAVEATS`：CSV 不能证明本地 source/runtime fingerprint、`resume_count`、checkpoint 完整性，也不能单独证明 seed2025 diag 是 step0 clean retry。

## 三种子正式 R1

| 方法 | final mean | seed SD | tail-5 mean | normalized AUC | Peak MiB | K-state MiB |
|---|---:|---:|---:|---:|---:|---:|
| diag | 3.261100 | 0.001114 | 3.269300 | 3.615159 | 38304 | 162.28125 |
| block4 | 3.262200 | 0.001136 | 3.270480 | 3.615968 | 39168 | 378 |
| none | 3.266667 | 0.000666 | 3.274993 | 3.624126 | 38304 | 162 |
| Muon | 3.277133 | 0.000252 | 3.285740 | 3.633386 | 37703 | 0 |

三 seed 配对 final 差：

- diag − block4：-0.001100 ± 0.000557。
- diag − none：-0.005567 ± 0.000513。
- diag − Muon：-0.016033 ± 0.001002。

## 三种子完整 2×2：diag/Muon × 0.9x/1.0x LR

- 方法主效应（diag − Muon）：-0.015583 ± 0.000907。
- LR 主效应（1.0x − 0.9x）：-0.000450 ± 0.000600。
- 方法×LR 交互：-0.001833 ± 0.001155。
- 方法主效应绝对值约为 LR 主效应绝对值的 34.6 倍。

## 结论

1. diag 与 block4 的三种子平均终点基本持平；现有证据支持“质量保持、状态更省”，不支持“diag 显著优于 block4”。
2. diag 相对 none 的优势跨 seed 为同一方向，说明逐坐标尺度不是纯粹冗余；但效应量仍小，应结合 tail/AUC 和曲线支配性表述。
3. diag 相对 Muon 的优势在完整 2×2 后仍远大于 10% LR 主效应，学习率混杂这一主要替代解释已经基本排除。
4. block4 的完整 c_proj block covariance 没有显示出相对 diag 的稳定质量收益，却需要额外约 215.72 MiB K-state 和 864 MiB 实测峰值显存。
5. wall-clock/step-time 不进入论文性能结论；这批 run 与其他 GPU 任务并发，且 W&B CSV 不包含完整节点隔离证据。

## 投稿前剩余门禁

- 收到并审计 seed2024/2025 的本地 formal manifests、summary、source/runtime/init hashes 和 checkpoint 完整性。
- 明确核验 seed2025 diag 为从 step0 开始的 clean retry，且没有拼接旧曲线。
- 将 12-run 正式 R1 与 6-run LR-cross 原始证据迁入 paper evidence 包后，再把本报告升级为 `READY_FOR_MAIN_TABLE`。
