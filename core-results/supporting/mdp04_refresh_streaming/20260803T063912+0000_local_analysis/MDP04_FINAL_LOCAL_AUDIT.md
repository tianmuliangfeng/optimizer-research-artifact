# MDP-04 final local audit

Date: 2026-08-03  
Source run: `20260803T063912+0000`  
Computation: **complete (12/12 units)**  
Formal adjudication: **`numeric_gate_failed`**  
Matrix evidence status: **descriptive partial / non-claim-eligible**

## 1. Delivery and integrity

The handoff contains the exact frozen coverage: 12 selected origin--replica units, 24 unit-event summaries, 432 layer-event rows, 12 accepted loss-outcome rows, and six validation slices. All selected unit manifests pass. The final lineage is 9 inherited stricter-v3 units plus 3 current-v4 units.

All summary artifacts and all scientific files named by the 12 unit manifests match their recorded SHA-256 values. The only provenance diagnostic is that `worker.log` differs from its unit-manifest hash in 12/12 units, consistent with logs continuing to append after manifest creation. Logs are not scientific inputs and this does not alter any CSV/JSON/NPZ value.

## 2. Why formal validation failed

The frozen full-run threshold was not changed. `runtime_resolvent_relative_residual <= 0.01` failed in 6/432 nested layer rows (1.39%). The maximum is 0.01492673. Every failure is late-stage layer 3: five rows from `late_muon` and one from `late_newton_full`. The other 408 non-layer-3 rows have maximum 0.00705758, below the gate.

The six failing rows have condition proxies around 26.3k--26.6k before refresh and 25.0k--25.4k after refresh. Across all nested layer rows the residual is strongly associated with the condition proxy (post-hoc numerical diagnostic, not seed-level inference). This localizes the problem to an ill-conditioned late layer rather than missing rows, non-finite values, source drift, or a failed shadow-to-actual update audit.

The registered 128-coordinate float64 slices have maximum resolvent residual 1.289e-15. They calibrate the implementation only, do not include layer 3, and were pre-labelled non-claim-eligible; they cannot rescue the full-run gate.

## 3. Descriptive matrix-to-loss alignment

All correlations below use the 12 nested origin--replica units separately for each event. They have no seed-level p-value. `within-origin` removes the four origin means; LOOO is the range after leaving out each origin in turn.

### Oriented endpoint loss harm

| Event | Metric | Spearman | Pearson | Within-origin Pearson | LOOO Spearman range |
|---|---|---:|---:|---:|---:|
| production@32 | matched-G median | 0.720 | 0.758 | 0.871 | [0.467, 0.883] |
| production@32 | NS5 median | 0.699 | 0.739 | 0.749 | [0.300, 0.950] |
| production@32 | matched-G pooled ratio | 0.930 | 0.918 | 0.836 | [0.850, 0.950] |
| production@32 | NS5 pooled ratio | 0.818 | 0.890 | 0.871 | [0.583, 0.967] |
| delayed@64 | matched-G median | 0.650 | 0.590 | 0.687 | [0.350, 0.933] |
| delayed@64 | NS5 median | 0.734 | 0.578 | 0.773 | [0.367, 1.000] |
| delayed@64 | matched-G pooled ratio | 0.825 | 0.823 | 0.551 | [0.683, 0.850] |
| delayed@64 | NS5 pooled ratio | 0.853 | 0.773 | 0.716 | [0.650, 0.983] |

### Oriented AUC harm (primary layer medians)

| Event | Metric | Spearman | Pearson | Within-origin Pearson | LOOO Spearman range |
|---|---|---:|---:|---:|---:|
| production@32 | matched-G median | 0.881 | 0.847 | 0.888 | [0.767, 0.917] |
| production@32 | NS5 median | 0.769 | 0.784 | 0.764 | [0.450, 0.917] |
| delayed@64 | matched-G median | 0.601 | 0.547 | 0.729 | [0.283, 0.817] |
| delayed@64 | NS5 median | 0.657 | 0.521 | 0.815 | [0.183, 0.883] |

The downstream matched-gradient and actual source-pinned NS5 measures retain the same sign in every leave-one-origin-out check for both events and both loss outcomes. The pooled Frobenius ratios are secondary pre-registered aggregations; the 18-layer median is the primary aggregation.

### Upstream magnitude checks

| Event | Metric | Spearman | Pearson | Within-origin Pearson | LOOO Spearman range |
|---|---|---:|---:|---:|---:|
| production@32 | Delta-K median | -0.042 | -0.045 | -0.578 | [-0.617, 0.183] |
| production@32 | Delta-A median | 0.000 | -0.052 | -0.575 | [-0.500, 0.233] |
| production@32 | runtime-inverse median | -0.266 | 0.098 | 0.911 | [-0.717, 0.583] |
| delayed@64 | Delta-K median | 0.161 | 0.137 | -0.494 | [-0.550, 0.500] |
| delayed@64 | Delta-A median | 0.161 | 0.116 | -0.487 | [-0.550, 0.500] |
| delayed@64 | runtime-inverse median | -0.245 | -0.058 | 0.677 | [-0.650, 0.683] |

Raw K/A change and runtime-inverse change magnitude do not track loss harm consistently; their leave-one-origin-out signs change. By contrast, the shock after applying the same gradient and after the production NS5 pipeline is consistently aligned with harm. This favors a gradient- and update-conditioned mechanism over a simple `larger covariance change is worse` account.

The failing resolvent proxy itself is not the observed loss mediator: its median Spearman correlation with loss harm is 0.105 at production and 0.343 at delayed refresh. The numerical-gate failure and the downstream update-alignment signal therefore need to be reported as two distinct facts.

## 4. Scientific adjudication

1. Experiment 37 remains accepted evidence that the scheduled down-projection refresh causes the short-horizon loss impulse under its frozen intervention tree.
2. This replay adds a coherent descriptive signal: matched-G preconditioned-gradient shock and actual NS5 update shock track both endpoint and AUC harm across both registered events, including within-origin and leave-one-origin-out checks.
3. MDP-04 cannot be promoted to formal claim-eligible evidence because one frozen hard gate failed. The threshold must not be relaxed and layer 3 must not be removed post hoc.
4. No long-horizon optimizer ranking, universal route ranking, or automatic selector follows from these 12 nested replay units.

## 5. Project decision

Default action is to stop remote MDP-04 computation, preserve this result as a numerically limited but scientifically informative diagnostic, and proceed with local evidence freezing and paper writing. The accepted paper-level mechanism remains the loss-level causal refresh result from experiment 37; the matrix alignment may guide discussion, limitations, and a future independently pre-registered confirmation, but must not be presented as a passed formal MDP-04 claim.

A future confirmation is optional, not queued. If explicitly authorized, it needs a new contract and independent data fixed before observation; reusing this run or changing the 0.01 gate cannot upgrade the present evidence.
