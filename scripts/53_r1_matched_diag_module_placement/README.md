# Experiment 53: matched diagonal module placement

This directory implements the frozen five-arm Modded-GPT-124M placement
experiment in `matched_diag_contract.json`.

The target families are MLP `c_fc`, MLP `c_proj`, and attention `o_proj`.
Every retained state uses the same activation-second-moment diagonal
representation, including the same coordinatewise statistic, initialization,
EWMA, all-coordinate ridge reduction, epsilon, and refresh cadence, in every
transformer layer. Module input dimension changes the number of diagonal
coordinates (and therefore state bytes), but not the representation or training
budget. QKV is frozen to `none`; no dense
activation covariance, inverse, Cholesky path, or temporary dense workspace is
allowed. Consequently `all_none` is a genuine state-free anchor.

The first four arms are a strict `c_fc x c_proj` diagonal/none factorial. The
fifth arm places the same generic diagonal state at attention `o_proj`.

Seed roles are immutable:

- engineering pilot: seed 2053, 34 steps, five arms, outcome-ineligible;
- formal: seeds 2024, 2025, and 2026, five arms per seed (15 units).

The suite is fail-closed and supports unit-level retry. It snapshots all source
dependencies into the run directory before GPU work. A formal unit consumes an
accepted seed-2053 engineering certificate for the same arm; the certificate
does not select learning rates or arms and is deliberately independent of all
formal seeds. Local CSV/JSON/checkpoints are primary. W&B is secondary and
quality-run timing is explicitly ineligible. The remote execution contract
reserves physical GPU 0 exclusively for Experiment 53 and runs all units
serially; GPU 1 remains available to a separately isolated suite.

Remote entry point:

```bash
export EX53_RUN_DIR="${SNM_RESULTS_ROOT:-$PWD/runs}/53_r1_matched_diag_module_placement/$(date -u +%Y%m%dT%H%M%S+0000)"
export EX53_GPUS="0"
bash commands/53_r1_matched_diag_module_placement/20260817_ex53_r1_matched_diag_module_placement.sh all
```

`all` performs the four stages in order. To run stages separately, keep the
same exported `EX53_RUN_DIR` for `preflight`, `pilot`, `formal`, and `verify`.
`resume` requires that same variable and continues the frozen source snapshot,
skipping accepted units.

CPU checks:

```bash
python -B scripts/53_r1_matched_diag_module_placement/test_matched_diag_source.py
python -B scripts/53_r1_matched_diag_module_placement/test_analyze_matched_diag.py
python -B scripts/53_r1_matched_diag_module_placement/test_matched_diag_suite.py
```
