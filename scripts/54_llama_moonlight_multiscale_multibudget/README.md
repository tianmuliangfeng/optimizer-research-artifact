# Experiment 54 — LLaMA Moonlight multiscale / non-10B multibudget

This is the fresh Moonlight replacement for the retired EX54 Mousse baseline.
Old Mousse run directories are engineering records only and must not be resumed
into this experiment.

## Scientific scope

EX54 is a **self-contained experiment**. It runs:

- LLaMA-124M at the 3.25B-token endpoint;
- LLaMA-1B at the 3.25B- and 6.97B-token endpoints.

EX54 deliberately stops at 6.97B for 1B. It does not hand off a checkpoint,
selection, contract, controller, or manifest to EX57. EX57 is a separate fresh
experiment.

## Moonlight rule transferred from Experiment 19

The matrix optimizer is the audited Experiment-19 Moonlight Muon rule:

- Muon momentum `0.95`, Nesterov on;
- 5 Newton–Schulz steps with coefficients `(3.4445, -4.7750, 2.0315)`;
- logical packed-QKV splitting retained from the EX19 audit;
- per-logical-matrix scaling `0.2 * sqrt(max(rows, cols))`;
- decoupled weight decay `0.1` using the unadjusted base LR;
- no Mousse factor matrices, eigendecomposition state, or activation-K state.

The LLaMA non-matrix / tied embedding-head route remains the accepted backup
AdamW route. Following the EX19 Moonlight formal winner, auxiliary and matrix
LRs are tied within each tuning cell.

Frozen 3-cell LR screen: `0.0010`, `0.0018`, `0.0030`. The transferred EX19
winner `0.0018` is the frozen center. Tuning seeds are 5401 (124M) and 5402
(1B); formal seeds are 2024/2025/2026.

## Fairness relative to EX48

The 1B track restores EX48's exact training geometry:

- sequence length: 1024;
- device batch: 8;
- gradient accumulation: 64;
- global batch: 512;
- tokens/update: 524,288;
- accepted EX48 103-shard data projection unchanged.

Quality-run timing remains ineligible. GPUs 0 and 1 are two independent
single-GPU workers; there is no DDP.

## Run

After installing the package on the **separate EX54 host**, the normal interface
is only:

```bash
cd /path/to/source-code
bash commands/54_llama_moonlight_multiscale_multibudget/20260819_ex54_llama_moonlight_multiscale_multibudget.sh all
```

`all` always creates a **fresh v4 scientific run**, prepares the frozen 50-shard
124M view, then executes `preflight -> tuning -> formal -> verify`. After an
interruption or recoverable worker failure, continue the exact run with:

```bash
bash commands/54_llama_moonlight_multiscale_multibudget/20260819_ex54_llama_moonlight_multiscale_multibudget.sh resume
```

The launcher remembers the active run in `LATEST_EX54_MOONLIGHT_RUN.txt`; no
`RUN=...` or GPU environment variables are required. Never reuse retired Mousse
or pre-v4 Moonlight run directories.  V4 additionally hash-pins the accepted
Experiment 19 Moonlight source and requires exact AST equality for the three
transferred algorithm subtrees; it keeps the accepted LLaMA AdamW backup route.


## 2026-08-19 fairness / GPU scheduling freeze

- EX54 is its own comparison family on its own host. Tuning seeds are **5401 (124M)** and **5402 (1B)**; formal seeds **2024/2025/2026 are never used for tuning**.
- The tuning budget remains exactly three LR cells × one tuning seed × 1000 updates per scale; no extra search was added for Moonlight.
- Physical GPUs **0/1** run independent single-GPU tuning jobs concurrently from one shared queue, and the six formal units are likewise load-balanced across GPU0/1.
- This is **not DDP**. Every 1B job retains device batch 8, accumulation 64, global batch 512; parallel execution changes wall-clock time only.
- Quality-run timing is ineligible, so cross-GPU I/O contention cannot be used as throughput evidence.
- EX54 has no EX57 runtime/checkpoint/selection/contract dependency.


## Compiler-cache isolation

GPU0/1 use separate `RUN_DIR/_compile_cache/gpuN` TorchInductor cache roots. This prevents concurrent single-GPU jobs from sharing one cold-start compiler-cache path; timing remains ineligible.
