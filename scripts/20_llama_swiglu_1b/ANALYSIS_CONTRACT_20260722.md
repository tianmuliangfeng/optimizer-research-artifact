# LLaMA/SwiGLU-1B analysis contract

Frozen on 2026-07-22 before inspecting any 1B medium or formal loss curve.

## Scope and evidence classes

- Model: pinned `llama_swiglu_1b_v1` (1,013,690,368 parameters).
- Data/tokenizer/validation: the pinned FineWeb10B protocol inherited from the
  audited 124M trainer.
- Core methods: `down_none`, `down_diag`, `newton_full`, and `muon`.
- Seed2026 is the first scale screen. Additional seeds are conditional and are
  not silently pooled with seed2026.
- Probe, smoke, capacity, and medium are screening evidence. Only the 6200-step
  stage is formal quality evidence.
- Concurrent runs on disjoint GPUs remain eligible for loss-vs-step/token and
  per-process CUDA memory. Wall-clock, step time, throughput, and energy are
  ineligible.

## Formal primary endpoint

The primary endpoint is validation loss at optimizer step 6200, corresponding
to exactly 3,250,585,600 training tokens with global batch 512 and context
1024. Lower is better. No best-checkpoint selection replaces the fixed-budget
endpoint.

The primary structural contrasts are `down_diag - down_none` and
`newton_full - down_none`. `muon - down_none` is the core optimizer reference.

## Practical margin

The practical loss margin is frozen at **0.0020 validation-loss units**.  A
memory-saving method is descriptively practically non-inferior to its reference
only when its fixed-budget loss is no more than 0.0020 higher.  With seed2026
alone this is a decision threshold, not a population-level statistical
non-inferiority test. A paper-level uncertainty claim requires additional
independent seeds.

The margin was chosen before 1B medium/formal curves were observed. It is of
the same order as, but slightly larger than, the seed-level fluctuations among
the 124M structural core and remains far below the roughly 0.096 AdamW gap.

## Secondary and robustness endpoints

- Tail-5 validation-loss mean and sample SD over steps 5800--6200.
- Normalized trapezoidal validation-loss AUC over steps 0--6200.
- Best observed validation loss, reported only as secondary.
- First discrete and linearly interpolated steps/tokens to fixed targets 4.0,
  3.8, and 3.6.
- A deterministic family-common target equal to the worst valid core-method
  final loss; this rule is frozen before results and guarantees reachability
  without choosing a favorable method-specific target.
- Exact model, optimizer, and K-state bytes; peak allocated CUDA memory.
  Peak reserved and OOM boundary come only from the capacity track.

## Fine capacity-boundary endpoint

The fixed-global-batch coarse capacity grid established batch 32 success and
batch 64 OOM for all three structural methods.  Because 512 has no additional
integer divisors in that interval, the predeclared fine-capacity track fixes
gradient accumulation at 8, context at 1024, seed at 2026, and runs 34 updates
per cell.  Global batch is defined mechanically as eight times device batch;
therefore this track is capacity-only and its loss, tokens/update, and timing
are not quality or performance evidence.

The first-pass device-batch grid is `32, 34, 36, 38, 40, 42, 44`, evaluated in
ascending order for `muon`, `newton_full`, `down_none`, and `down_diag`.  Muon
is the matched optimizer-memory reference for the selective Newton methods;
its older batch-8 smoke is not substituted for this comparison. Batch 32 is a
required cross-protocol anchor.  Each method stops at its first genuine CUDA
OOM or other failure.  The primary endpoint for each method is the ordered
pair `(maximum tested successful device batch, first tested OOM device batch)`;
a success must complete all 34 updates and the step-32 K refresh.  Peak
allocated, peak reserved, pre-run free memory, exact state bytes, failure
class, and failure step are supporting endpoints.  Every cell requires at
least 98% of the target GPU's physical memory to be free immediately before
launch; failure of this guard aborts the experiment and is not classified as
an OOM endpoint.

The predeclared memory contrasts at each common successful device batch are
`down_none - muon`, `down_diag - muon`, and `newton_full - muon` for peak
allocated bytes, peak reserved bytes, optimizer-state bytes, and K-state
bytes.  Capacity-boundary differences are reported separately and are not
converted into quality or throughput claims.

If the successful and OOM endpoints differ by exactly two, the single odd
batch between them may be run afterward as the predeclared exact-boundary
confirmation.  No other result-dependent batch selection is permitted.  If a
method succeeds through batch 44, the even grid may be extended upward before
any odd-batch confirmation, retaining the same fixed-accumulation protocol.

## Medium-1000 gate

Medium uses 1000 plateau-LR updates (524,288,000 tokens) and is not a shortened
formal schedule. A method passes when it:

1. completes 1000 updates with finite train/validation loss;
2. preserves the pinned source, runtime, data, initialization, model profile,
   global batch, context, and device batch 8;
3. has no unexplained loss spike, persistent divergence, skipped update, data
   error, or inexact resume;
4. produces a complete manifest, checkpoint, metrics history, and summary.

Medium is not used to select the winner or redefine the 0.0020 margin. No LR
change is permitted merely because another method is ahead at step 1000. A
method with nonfinite or clearly divergent behavior is stopped and diagnosed;
any changed-LR retry is labeled a separate exploratory branch and cannot be
substituted silently into the matched formal comparison.

## Missing runs and stopping rules

- No imputation or last-observation carry-forward.
- Interrupted medium/formal runs may use exact checkpoint resume; resume count
  is reported and timing remains ineligible.
- A formal comparison requires all included methods to complete the identical
  6200-step/token budget. Incomplete methods are reported as failures, not
  ranked by an earlier checkpoint.
- Additional seeds or an approximately 10B-token extension are considered only
  after the 6200-step seed2026 result shows a clear, stable, decision-relevant
  contrast under this contract.
