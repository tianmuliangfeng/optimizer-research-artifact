# R1 depth × c_proj K-mode: frozen analysis contract

This is the official-R1 architecture/implementation transfer of the completed
OWT and WikiText-103 depth experiments. It is confirmatory for the direction
of the paired `diag - none` effect and diagnostic for its depth modulation.

## Frozen run matrix

- Architecture and training recipe: official R1 12L/D768/H12, 6,200 steps.
- Seeds: 2024, 2025, 2026.
- Depth rules:
  - early: layers 0–7;
  - center: layers 2–9;
  - late: layers 4–11;
  - edge: layers 0–3 and 8–11;
  - all: layers 0–11.
- Selected `mlp.c_proj` mode: `none` or efficient `diag`.
- Every unselected `mlp.c_proj`: official `block4`.
- All other Newton-Muon matrix families: unchanged official behavior.
- Anchors rerun within every seed: official `block4` and official Muon.
- Total: `5 × 2 × 3 + 2 × 3 = 36` formal runs.

The four partial rules select eight layers and leave four block4 layers, so
their pairwise differences can be interpreted as position effects. The
all-depth rule changes twelve layers and is interpreted separately because it
also changes coverage.

## Controls

- The ten depth treatments and block4 use the official Newton-Muon LR:
  base LR 0.0040 and matrix LR 0.00040.
- Muon retains its official recipe LR: base LR 0.0036 and matrix LR 0.00036.
  Muon comparisons are therefore recipe-level, not shared-LR causal effects.
- Within a seed and shard, initialization hashes must be identical.
- A 34-step exact-shape numerical smoke must cross the step-32 K refresh.
- Generated-source hashes, source diffs, runtime fingerprints, data audit,
  commands, metric rows, memory accounting, and W&B histories are retained.
- Formal checkpoints are disabled because they are not an estimand and cost
  approximately 10 GiB per run.
- Two GPUs may run concurrently. Quality and per-process memory remain usable;
  timing is explicitly ineligible.
- Every within-rule none/diag pair stays on one physical GPU. The all-depth
  pair and both official anchors share one shard, and shards cross over GPUs
  between seeds to avoid complete method–GPU confounding.

## Frozen endpoints

Primary endpoint:

- validation loss at step 6,200.

Primary contrast:

- within-seed, within-rule `diag - none`.

Primary reporting:

- the three seed-level deltas;
- mean and sample SD across seeds;
- number of negative deltas out of three.

Secondary quality endpoints:

- mean of the final five validation points;
- normalized validation-loss AUC over the common validation grid.

Secondary systems endpoints:

- K-state bytes;
- optimizer-state bytes;
- peak allocated GPU memory.

## Pre-specified transfer checks

1. Direction transfer: report whether each rule's mean `diag - none` is
   negative and whether at least two of three seeds agree.
2. All-depth transfer: compare the all-depth paired effect with the completed
   OWT and WikiText results.
3. Frozen magnitude pattern: test, without selecting a new mask, whether edge
   is stronger than all and center/late are weaker than all.
4. Official-baseline trade-off: compare all-none and all-diag against the
   newly rerun block4 anchor within seed.

With three seeds, uncertainty is descriptive. Do not claim conventional
statistical significance, do not select a new best depth mask post hoc, and
do not treat Muon differences as shared-LR comparisons.
