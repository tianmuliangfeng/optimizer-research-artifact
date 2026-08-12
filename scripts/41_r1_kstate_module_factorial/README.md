# 41 — R1 模块 K 状态 2×2 因子实验

本目录只补 R1 因子表中缺失的两个 cell：

- `cproj_only`: `c_fc=none, c_proj=block4`
- `neither`: `c_fc=none, c_proj=none`

`both` 与 `fc_only` 从已验收的 15 号三种子结果只读复用；六行冻结引用位于
`existing_cells_reference.csv`，并同时保留 canonical source hash。实验不重跑已有 cell，也不把
Newton 配方下的 `neither` 冒充 Muon baseline。

唯一远程入口：

```bash
bash commands/41_r1_kstate_module_factorial/20260729_r1_kstate_module_factorial.sh
```

请等 39 号实验完全结束、R1 主机物理 GPU0 和 GPU1 都空闲后再运行。唯一
入口会在两张卡上并行调度不同 seed；每个训练进程仍只看到一张卡。入口执行
本地合同测试、三种子 smoke、六个正式训练、W&B 上传和最终因子分析。恢复
同一个 run 使用脚本打印的 `RESUME_COMMAND`。

入口明确分离两套解释器：`CTRL_PY=${SNM_CONTROLLER_PYTHON}`
负责 W&B、校验和调度；`TRAIN_PY=${SNM_TRAINING_PYTHON}`
只负责隔离训练。不得用 `TRAIN_PY` 启动 formal 控制器。官方源码默认使用
`${SNM_OFFICIAL_REPO}`。

结果目录：

```text
${SNM_RESULTS_ROOT}/
  41_r1_kstate_module_factorial/<UTC timestamp>/
```

正式 timing 一律标记为不可用于论文；质量、K 状态和峰值显存按合同验收。
