# Experiments 54, 56, and 57: accepted remote-result synthesis

## Acceptance summary

All three experiments passed their frozen scientific gates and are ready for use in the next manuscript revision.

| Experiment | Scientific result | Acceptance | W&B effect | Timing |
|---|---|---|---|---|
| EX54 | Moonlight at LLaMA-124M and independent 1B non-10B budgets | accepted | not provided; not required | ineligible |
| EX56 | LLaMA-1B global diagonal through approximately 10B tokens | accepted | not provided; not required | ineligible |
| EX57 | Independent LLaMA-1B Moonlight through approximately 10B tokens | accepted | not provided; not required | ineligible |

The independent audit verified completion/handoff state, analysis and source-snapshot hashes, checkpoint verification receipts, formal-unit counts, and 216 reported statistical fields. The primary evidence is the local sealed CSV/JSON package, so the absence of W&B exports does not create a missing-data problem.

The received bundle is the intentional `no-pt` variant: checkpoint byte payloads are absent, while the remote full-checkpoint hash receipts and per-unit checkpoint hashes are retained. This is sufficient for the present result-package acceptance, but the bundle must not be described as a locally replayable full-checkpoint archive.

## Paper-level conclusions

1. **Moonlight has a clear environment/stage boundary.** It is strongly favorable at LLaMA-124M, but at LLaMA-1B it trails Muon from 3.25B tokens onward and trails every tested core route by 6.97B and approximately 10B tokens.
2. **Global diagonal changes the within-family ordering at LLaMA-1B.** It beats local diagonal, identity, and full-K at every one of 27 seed-budget comparisons, with only 1.600 MiB of retained diagonal K state.
3. **Muon remains the LLaMA-1B long-budget leader.** Muon beats global diagonal in 9/9 paired comparisons. Together with the three previously tested curvature-state routes, this expands the descriptive lead to 36/36 seed-method-budget endpoints, while preserving the independent experiment graphs.
4. **The evidence does not support a universal optimizer or state-allocation ranking.** Scale, architecture, and training stage condition the observed ordering.

## Mandatory reporting boundaries

- EX54 and EX57 overlap at the 3.25B and 6.97B budgets but are independent experiments. Report their directional agreement as replication; do not concatenate them into an $n=6$ estimate.
- The $\pm0.002$ band is a practical descriptive margin, not an equivalence test. In particular, EX56 does not license a formal equivalence claim between global diagonal and Muon.
- No wall-clock or throughput claim may be derived from EX54, EX56, or EX57.
- The source artifacts use `none`; manuscript-facing terminology should use **identity**, with a one-time definition that the retained right-side second-moment factor is the identity.

## Canonical data locations

- `54_llama_moonlight_multiscale_multibudget/received_20260831/source_artifacts/`
- `56_llama1b_10b_global_diag/received_20260831/source_artifacts/`
- `57_llama1b_10b_moonlight/received_20260831/source_artifacts/`

The `source_artifacts` trees preserve exact remote files and relative structure. Their companion `independent_review` directories contain the audit verdicts and paper-safe summaries.
