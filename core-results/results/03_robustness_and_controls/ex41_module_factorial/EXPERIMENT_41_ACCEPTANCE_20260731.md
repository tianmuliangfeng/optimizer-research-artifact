# Experiment 41 acceptance — R1 K-state module 2×2 factorial

Date: 2026-07-31  
Run: `20260730T043222+0000`  
Status: **training complete; quality and memory evidence accepted**

## Acceptance decision

All six newly trained formal runs completed at step 6200 and all six reused
formal runs have frozen source evidence. Seeds are 2024, 2025, and 2026 for
every factorial cell. Initial validation losses match within seed. The ZIP
contains 428 entries and no checkpoint files.

The run-level manifest and the first corrected analysis both passed. The
accepted scientific classification is **`r1_allocation_diverges`**, replacing
the emitted `partial_non_cproj_support` label. The old label came from a branch
ordering bug: when both `c_fc` and `c_proj block4` were beneficial, the
classifier stopped at the `c_fc`-beneficial branch before testing the
`c_proj`-beneficial branch. This affected only the label and suggested next
step, not training, loss values, factorial effects, confidence intervals, or
memory measurements. Analyzer version `2026-07-31.4` fixes the ordering and has
a regression test.

## Mean final validation loss

| Cell | c_fc K | c_proj K | Mean final val loss | K-state MiB | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|
| both / original Newton–Muon block4 | full | block4 | 3.262200 | 378 | 39168 |
| cproj_only | none | block4 | 3.265567 | 324 | 38923 |
| fc_only / Selective none | full | none | 3.266667 | 162 | 38304 |
| neither / Newton-recipe all-none | none | none | 3.270267 | 108 | 38059 |

`neither` is not the formal Muon baseline. Diag is not part of this factorial.

## Factorial conclusions

Effects use enabled-minus-disabled coding, so negative means enabling the
factor lowers loss.

| Effect | Mean | 95% t interval, df=2 | Seed direction | Interpretation |
|---|---:|---:|---:|---|
| c_fc full main effect | -0.003483 | [-0.004167, -0.002799] | 3/3 beneficial | supported |
| c_proj block4 main effect | -0.004583 | [-0.005267, -0.003899] | 3/3 beneficial | supported |
| interaction | +0.000233 | [-0.001222, +0.001689] | mixed/zero | below the 0.002 materiality margin |

The simple effects agree with the main effects:

- Removing c_proj block4 raises final loss by 0.004467 with c_fc full and by
  0.004700 with c_fc none.
- Removing c_fc full raises final loss by 0.003367 with c_proj block4 and by
  0.003600 with c_proj none.
- Tail-five validation loss reproduces both beneficial main effects and a
  non-material interaction.
- Normalized validation AUC also favors both factors; its positive interaction
  is secondary evidence and does not overturn the primary final-loss decision.

Thus the earlier OWT/WikiText allocation result does not transfer literally to
R1 Modded-NanoGPT: on R1, both MLP K factors improve quality and the effects are
approximately additive. No extra rescue sweep is justified.

## Quality–memory trade-off

The c_proj block4 factor contributes the larger absolute final-loss gain, but
it costs 216 MiB of persistent K state, versus 54 MiB for c_fc full. On the
main-effect scale, c_fc delivers about 3.04× more loss reduction per MiB of
persistent K state. This makes Selective none a defensible memory–quality
Pareto point, but not the best-loss cell:

- Selective none versus original block4 saves 216 MiB K state (57.14%) and
  864 MiB peak allocated memory (2.21%), at +0.004467 mean final loss.
- cproj_only versus original block4 saves 54 MiB K state (14.29%) and 245 MiB
  peak allocated memory (0.63%), at +0.003367 mean final loss.

The paper must not claim that removing c_proj K improves R1 loss. The defensible
claim is a measured quality–memory trade-off and architecture-dependent K-state
allocation.

## K-state contract erratum

The frozen contract omitted the invariant 108 MiB attention K state from the
two newly trained cells. The accepted additive decomposition, verified for all
three seeds, is:

- shared attention K state: 108 MiB;
- c_fc full increment: 54 MiB;
- c_proj block4 increment: 216 MiB;
- corrected totals: both 378, fc_only 162, cproj_only 324, neither 108 MiB.

This erratum does not affect training or quality results.

## W&B reconciliation

The nine exports contain exactly the six newly trained formal runs. Run names
match the ZIP summaries.

- All 378 validation-loss points, all 1,860 sampled training-loss points, all
  3,732 learning-rate points, and all 18 memory observations match the local
  artifacts exactly (5,988 observations total across seven eligible metrics).
- Peak allocated memory is 38,059 MiB for neither and 38,923 MiB for
  cproj_only, identically across seeds.
- K state is 108 versus 324 MiB; optimizer state is 726.474613 versus
  942.474613 MiB.
- W&B timing endpoints match local summaries exactly. Intermediate exported
  timing samples differ by at most 0.47 ms for step average and 1.026 s for
  cumulative time (under 0.04%), consistent with chart export
  sampling/rounding.

Timing is ineligible because the two physical GPUs trained concurrently.
Experiment 39 remains the source for isolated efficiency claims.

## Evidence boundaries

- Primary statistical unit: seed; `n=3`.
- Confidence intervals are t intervals with 2 degrees of freedom and should be
  reported as small-sample evidence.
- The two `c_fc=full` cells are reused from the frozen experiment-15 summary;
  the two `c_fc=none` cells are new training.
- This factorial isolates K-state allocation inside the fixed Newton recipe; it
  is not a four-optimizer benchmark.
- No checkpoint is needed for this acceptance because the manifest, summaries,
  full metric CSVs, initialization hashes, source hashes, and W&B curves are
  present.

## Source integrity

- Input ZIP SHA-256:
  `eb70af40da8345b08536210fbe044cfaa04c89910b7a4b2c1aa574a334b3f360`
- Frozen reused-summary SHA-256:
  `21737784b9fccc69053bf7e507bd91a0a8a8ea26d02579d5c3d87f512e0ecb13`
- Frozen factorial contract SHA-256:
  `86635002eebf38050208f838840ce6fabb13a545befd504b021f3a727781077e`
- Corrected analyzer SHA-256:
  `9401b438c7fb6b11e8f5f8db8eaf9edffcc0a9dbd7a408fced9bac479577e5ef`
- Regression tests SHA-256:
  `41e3a2c4a19dd6378730b6c375b48fa147d88575a72239223a5058385a2b66ad`
- Regression tests: 13/13 passed.

