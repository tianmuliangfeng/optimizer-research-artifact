# Experiment 57 — independent LLaMA-1B Moonlight 10B baseline

This is a fresh replacement for the retired EX57 Mousse run. It is **not a
continuation of Experiment 54**.

## Independence guarantee

EX57 owns its own:

- `ex57_contract.json`;
- frozen EX48 data-projection copy and frozen controls;
- Moonlight optimizer implementation;
- source builder and derived trainer snapshot;
- tuning selection;
- formal checkpoints and manifests;
- analysis and completion receipt.

The controller never imports `scripts/54_*`, never reads an EX54 run directory,
and never consumes an EX54 checkpoint or selection. The only shared scientific
provenance is the accepted upstream LLaMA/EX48 parent source hashes and the
Experiment-19 Moonlight algorithmic prior.

## Moonlight definition

The matrix route is the Experiment-19 audited Moonlight Muon baseline:

- momentum 0.95, Nesterov on;
- five quintic Newton–Schulz steps;
- logical packed-QKV splitting retained from EX19;
- update scaling `0.2 * sqrt(max(rows, cols))`;
- decoupled weight decay 0.1 using the unadjusted base LR;
- no Mousse factor/eigen state and no activation-K state.

The frozen LLaMA tuning screen is `0.0010 / 0.0018 / 0.0030`, with auxiliary
and matrix LR tied. Seed 5701 is tuning-only. Formal seeds are
2024/2025/2026.

## Fairness to EX48

EX57 uses the exact EX48 1B training geometry:

- device batch = 8;
- accumulation = 64;
- global batch = 512;
- sequence length = 1024;
- 524,288 tokens/update;
- identical accepted 103-shard EX48 data projection;
- identical formal seeds;
- identical 3.25B, 6.97B, and approximately-10B endpoint graph.

GPU allocation remains physical GPUs `0 1 2`, with one independent formal seed
per GPU and no DDP. Quality-run timing is not a scientific endpoint.

## Run

After installing the package on the EX57 host, the normal interface is only:

```bash
cd /path/to/source-code
bash commands/57_llama1b_10b_moonlight/20260819_ex57_llama1b_10b_moonlight.sh all
```

`all` always creates a **fresh v4 scientific run** and executes
`preflight -> tuning -> formal -> verify`. If the process is interrupted or a
recoverable worker fails, continue that exact run with:

```bash
bash commands/57_llama1b_10b_moonlight/20260819_ex57_llama1b_10b_moonlight.sh resume
```

The launcher remembers the active run in `LATEST_EX57_MOONLIGHT_RUN.txt`; no
`RUN=...` or GPU environment variables are required. Never reuse the retired
Mousse or pre-v4 Moonlight run directories.  V4 additionally hash-pins the
accepted Experiment 19 Moonlight source and requires exact AST equality for the
three transferred algorithm subtrees; it keeps the accepted LLaMA AdamW backup
route.


## 2026-08-19 fairness / GPU scheduling freeze

- Tuning uses **seed 5701 only**. Formal seeds **2024/2025/2026 are forbidden from tuning**.
- The three LR cells are launched concurrently on physical GPUs **0/1/2**, one independent single-GPU job per device.
- Formal seeds 2024/2025/2026 are likewise launched concurrently on GPUs 0/1/2. This is **not DDP**; each scientific run retains device batch 8, accumulation 64, global batch 512.
- Parallelism changes wall-clock time only. Quality-run timing remains ineligible.
- EX57 is a separate comparison family and has no EX54 runtime/checkpoint/selection dependency.


## Compiler-cache isolation

GPU0/1/2 use separate `RUN_DIR/_compile_cache/gpuN` TorchInductor cache roots. This is an execution-only wall-clock optimization for concurrent single-GPU jobs; scientific timing is not reported.
