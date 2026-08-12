# R1 Muon/diag 2x2 learning-rate cross

This protocol supplies only the two cells missing from formal R1. It must be
analyzed together with the matching controlled-seed R1 cells; it is not a
replacement for R1.

| Method | Muon LR: base 0.0036 / matrix 0.00036 | Newton LR: base 0.0040 / matrix 0.00040 |
|---|---|---|
| Muon | reuse formal R1 | **run here** |
| diag | **run here** | reuse formal R1 |

The resulting three-seed factorial separates:

- method (`muon` versus `diag`);
- absolute LR level (`0.9x` versus `1.0x`);
- method-by-LR interaction.

## Isolation and evidence rules

- The entry point delegates to the audited R1 machinery with `--lr-cross`.
- It uses a distinct family, protocol, results directory, run prefix, and W&B
  project, so official-recipe R1 runs cannot be mislabeled as LR-cross runs.
- Each crossed source is derived from the pinned official source with exactly
  one additional textual change to `Hyperparameters.learning_rate`; source
  hashes and the complete official-to-derived patch are saved per run.
- Muon and diag retain their existing implementations, initialization audit,
  exact-formal-shape smoke certificate, finite-value gate, checkpoint checks,
  W&B upload checks, and batch resume behavior.
- Use the same seed in R1 and LR-cross. The primary endpoint is paired final
  validation loss, with the full validation curve as supporting evidence.
- Running on physical GPU 1 via `CUDA_VISIBLE_DEVICES=1` is compute-isolated
  from R1 on physical GPU 0. Because the GPUs still share a host, do not pool
  wall-clock/throughput measurements from overlapping runs; quality and memory
  evidence remain in scope.

## Defaults

- methods: `muon diag`;
- W&B project: `Selective-Newton-Muon-MainConf-OfficialR1-LRCross-20260720`;
- results family: `16_official_newton_muon_r1_lr_cross`;
- formal profile: 6200 updates, identical to R1;
- supported seeds for the planned analysis: 2024, 2025, and 2026.

Every seed requires its own exact-shape numerical smoke because the smoke
certificate binds the seed, runtime, methods, and derived-source fingerprints.

## H100 GPU 1 launcher

`run_all_seeds_gpu1.sh` runs seed 2024, 2025, and 2026 sequentially on physical
GPU 1. It creates a matching smoke before each formal batch and uploads only
formal evidence to W&B. The method order is reversed relative to the matching
R1 endpoints within each seed.

The launcher is restart-safe at the batch level. Re-running the same launcher
after a power interruption finds the latest per-seed smoke/formal batch,
revalidates completed methods, retries an incomplete W&B upload when possible,
and restarts only an interrupted training method from step zero. The official
program does not support mid-method checkpoint continuation.
