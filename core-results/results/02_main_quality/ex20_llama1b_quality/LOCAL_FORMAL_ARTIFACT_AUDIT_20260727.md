# LLaMA/SwiGLU-1B formal-6200 本地证书审计（2026-07-27）

## 结论

三个 seed、四种方法的 compact formal artifacts 均完整可读，manifest、summary、status、metrics 与 W&B 曲线逐点一致。质量证据和实测训练状态可以正式封口。唯一保留 caveat 是约 8--11GB checkpoint 本体未随压缩包复制，因此本地无法验证 checkpoint 文件内容 hash；MECH-01 将在远程原地只读加载并对选中 checkpoint 计算 SHA-256。

- 压缩包 SHA-256：`7225a005515c18ab85cc83fcba0dc666287faea950b151663316fc22b69fdccb`；
- 检查结果：PASS=169，FAIL=0，WARN=1；
- 有效 run：12；所有 run 均为 6200 steps、3,250,585,600 tokens、resume_count=0；
- seed2026 分成两个双方法子批次，但两批 runtime、data、source、初始化和 formal 配置一致，可以合法合并。

## 实测显存与状态

| 方法 | K-state GiB | Optimizer state GiB | Peak allocated GiB | Peak vs Muon GiB |
|---|---:|---:|---:|---:|
| muon | 0.000 | 4.160 | 26.276 | +0.000 |
| down_none | 1.688 | 5.848 | 28.867 | +2.591 |
| down_diag | 1.688 | 5.848 | 28.868 | +2.592 |
| newton_full | 5.750 | 9.910 | 35.139 | +8.862 |

down-none 相对 Newton-full 精确减少 4.063 GiB K-state，并在 formal batch-8 实测降低约 6.271 GiB peak allocated。diag 与 none 的 K-state 几乎相同；full 的额外 dense down-K 没有换来更低 loss。

这些 formal peak 是进程内 batch-8 训练峰值，不能替代已封口的 exact OOM capacity boundary；二者应分别作为固定配置显存和容量边界证据。

## MECH gate

本地证书没有发现阻止机制实验的问题。按冻结队列，下一步是：

1. R1-native seed2026 none@6200 的 MECH-01 preflight；
2. 通过后运行三层 numerical smoke 并导出 fixed tensor bundle；
3. 在另一套 runtime replay 同一 bundle，完成 host/runtime equivalence；
4. 选中 checkpoint 完成 full SHA-256 与 schema gate 后，才进入 MECH-02-R1/L124 科学数据采集。
