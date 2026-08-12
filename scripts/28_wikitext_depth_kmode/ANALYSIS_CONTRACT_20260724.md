# WikiText-103 12L depth × c_proj K-mode analysis contract

This contract is frozen before the formal WikiText depth results are read.
It is the cross-dataset confirmation of family 25, not a new search for the
best layer mask.

## Frozen data and training configuration

- dataset: `wikitext103_gpt2_50m`, WikiText-103 raw, GPT-2 tokenizer;
- pinned train/validation binaries:
  - train SHA-256:
    `58c04ef835efade28c303561b99873eed64ac6a4060c5d715b4fb6538ae3cd34`;
  - validation SHA-256:
    `397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979`;
- model: local NanoGPT GPT/GELU, 12L/D768/H12;
- sequence length 512, device batch 16, gradient accumulation 1;
- 5,000 updates, LR decay 5,000, matrix LR 0.02;
- `input_beta=0.95`, `input_ridge=0.2`, `input_refresh=32`,
  `input_max_samples=2048`;
- seeds: 2024, 2025, 2026;
- validation every 500 steps with 20 batches; train logging every 20 steps.

The selected `mlp.c_proj` layers use `none` or `diag`. Every unselected
`mlp.c_proj` and every non-`mlp.c_proj` matrix remains on local dense-full
Newton-Muon.

## Frozen cells

- early: h0-h7;
- center: h2-h9;
- late: h4-h11;
- edge: h0-h3 and h8-h11;
- all: h0-h11.

For every rule and seed, run both `none` and `diag`. Rerun full Newton-Muon
and Muon anchors under the same launcher. Total:

`5 rules × 2 modes × 3 seeds + 2 anchors × 3 seeds = 36 runs`.

## Primary endpoints and estimands

The primary endpoint is validation loss at the last common checkpoint,
step 4,500. For every rule and seed:

`Delta_Wiki(rule, seed) = L_diag - L_none`.

Negative values favor diagonal K. Report mean, sample SD, individual seed
values, and sign count. The all-depth contrast remains the quality-facing
primary contrast.

Depth interaction is:

`I_Wiki(rule, seed) = Delta_Wiki(rule, seed) - Delta_Wiki(all, seed)`.

The OWT result fixed the external prediction before WikiText results are
read:

- edge interaction is negative;
- center and late interactions are positive;
- early is not assigned a directional interaction prediction.

The cross-dataset transfer contrast is descriptive:

`T(rule) = mean_seed Delta_Wiki(rule) - mean_seed Delta_OWT(rule)`.

Absolute validation losses must not be compared between datasets.

## Decision rules

1. **Uniform direction replication**
   - all five WikiText `Delta` means are negative; and
   - at least four rules have 3/3 negative seed deltas.

2. **Depth-magnitude pattern replication**
   - mean `I_Wiki(edge) <= -0.002`;
   - mean `I_Wiki(center) >= +0.002`;
   - mean `I_Wiki(late) >= +0.002`; and
   - each predicted interaction has the expected sign in at least 2/3 seeds.

3. **Direction-only replication**
   - rule 1 passes but rule 2 fails. The allowed conclusion is that diag
     benefit transfers, while the exact depth-amplitude ordering is
     dataset-dependent.

4. **No cross-dataset replication**
   - the all-depth sign conflicts across seeds, or fewer than four rules
     show negative means. Report the boundary; do not select a different
     post-hoc rule.

The `0.002` interaction margin is frozen as a practical materiality
threshold. Three-seed t intervals are descriptive only.

## Secondary endpoints

1. normalized validation AUC on steps 0:500:4,500;
2. mean of the final three validation checkpoints;
3. best validation loss, for continuity only;
4. exact K-state bytes and released fraction;
5. peak allocated memory;
6. elapsed time, descriptive only.

Timing is ineligible if another process shares the same physical GPU.

## Integrity checks

- exactly 36 unique formal run names and 12 cells per seed;
- exact validation grid 0:500:4,500 and training grid 0:20:4,980;
- seed-matched step-0 validation losses are equal within numerical tolerance;
- target layer and `none/diag/full` tensor counts match each rule;
- retained plus released K-state equals the full anchor;
- binary hashes match the pinned data fingerprints;
- all values are finite and no metric step is duplicated.

This experiment does not test attention, `mlp.c_fc`, LLaMA/SwiGLU, or
official R1 block4 routing.
