# 实验 48 本地独立验收（2026-08-12）

小型归档与统计链验收：**passed**。冻结分类独立重算：`persistent_muon_lead`。

## 三 seed 平均验证损失

| token 预算 | token/parameter | down_none | down_diag | newton_full | muon |
|---|---:|---:|---:|---:|---:|
| tokens_3p2506b | 3.2067 | 2.974236 | 2.976024 | 2.976155 | 2.969889 |
| tokens_6p9694b | 6.8752 | 2.862803 | 2.863701 | 2.864014 | 2.858799 |
| tokens_approximately_10b | 9.8647 | 2.817789 | 2.818116 | 2.818708 | 2.814068 |

## 关键结论

- Muon 在 3 个预算、3 个 seed、3 个 Newton-family 对手的全部 27 个配对终点上 loss 更低。
- 到约 10B token 时，`down_none - muon`、`down_diag - muon`、`newton_full - muon` 的平均差分别为 +0.003721、+0.004048、+0.004640。
- 随 token 增长，Newton-family 与 Muon 的差距整体收窄，但没有翻转；长训练不支持“LLaMA-1B 先前结果主要只是训练不足”的解释。
- `down_none` 在全部 9 个 seed×预算配对中优于 `newton_full`，但其平均优势从 3.25B 的 0.001919 缩小到 10B 的 0.000919。
- 10B 时 `down_none` 与 `down_diag` 的均值只差 0.000327，方向也在 seed 间混合，应视为实践等价，不应声称 none 稳定优于 diag。
- 3.25B 的 normalized AUC 排名与最终 loss 不一致；这是因为 AUC 包含共享早期轨迹且终点 cooldown 改变后期排序。正式结论应以预注册终点 loss 为主，tail-5 为稳健性旁证。

## W&B 与 checkpoint 边界

- 归档中有 36/36 条上传成功凭证；本次收到的 UI 导出覆盖 36/36 条 run。
- 收到的 36 条 run 的 6 类指标共 120,732 个轨迹值，与正式 metrics.csv 逐点一致：`True`（缺失/额外/数值不符均为 0）。
- 小型归档记录了 36 个 checkpoint 证书，总逻辑大小约 439.1 GB；真实 checkpoint 未包含在 ZIP 中。
- 远程 `verify --full-checkpoint-hash` JSON 回执已经独立核验并保存：`True`；共 1 份回执，JSON 自身声明、全部布尔检查和 sidecar SHA-256 均通过。

## 论文解释边界

实验 48 是同一 LLaMA-1B 架构内的训练阶段确认，不是架构因果实验，也不解释 refresh harm 的来源。它最直接支持的是：在约 3.2、6.9、9.9 tokens/parameter 的范围内，Muon 的质量优势持续存在；选择性 K 路由相对 full Newton 有较小但一致的终点收益与明显状态节省。
