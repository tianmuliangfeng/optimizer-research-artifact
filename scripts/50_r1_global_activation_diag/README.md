# Experiment 50: R1 global activation-diagonal control

This confirmatory control changes only the representation of the activation
second-moment preconditioner in the accepted R1 124M Newton recipe. Every
eligible matrix parameter uses a diagonal input-side K; the MLP down projection
keeps the same `[4, d]` diagonal grouping already audited in Experiment 15.

Formal evidence consists of exactly three runs (seeds 2024, 2025, 2026) at
6200 steps. Existing block4, selective-diag, selective-none, and Muon rows are
read-only frozen controls. The primary contrast is global-diag minus
selective-diag final validation loss.

Run stages through `commands/50_r1_global_activation_diag/20260814_ex50_r1_global_activation_diag.sh`:

1. `preflight`
2. `pilot`
3. `formal`
4. `verify`

Reuse the same `EX50_RUN_DIR` for every stage. Re-running a stage with the same
directory skips accepted units and restarts only incomplete seed-level units.
There is no mid-step checkpoint resume. W&B is secondary; local manifests and
CSV files are the scientific record. Timing from concurrent formal runs is not
eligible evidence.
