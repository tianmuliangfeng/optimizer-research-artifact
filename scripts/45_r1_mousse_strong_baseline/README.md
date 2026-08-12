# Experiment 45: controlled 124M R1 Mousse baseline

This directory implements the experiment-45 outline without modifying
experiments 43/44 or `HANDOFF.md`. It adds exactly one new training method:
`Mousse-R1 adaptation`.

The trainer is derived from the pinned official R1 `train_gpt_muon_1.py`.
Model construction, seed/init, FineWeb10B loader/order, 512x1024 tokens per
update, BF16 training path, validation, schedule, and the tied embedding/head
AdamW route are unchanged. Only the 48 hidden matrix tensors use Mousse.
Packed QKV tensors are split into 72 total logical matrices, as required by
the existing R1 Muon contract.

This is not an unchanged reproduction of the upstream Mousse training
scaffold. The upstream provenance, the R1 mapping, and all frozen constants
are machine-readable in `mousse_contract.json`.

`verify_mousse_upstream.py /path/to/Mousse` independently verifies the pinned
commit, MIT license identity, and byte hashes. The normalized upstream config
and snapshot manifest are under `upstream_snapshot/`.

## Frozen Mousse semantics

- momentum `0.95`, no Nesterov;
- double-sided gradient Kronecker statistics, EMA beta `0.95`;
- trace normalization and bias correction;
- factor epsilon `1e-5`, exponent `1/8`;
- eigensystem refresh on one-based steps `1, 11, 21, ...`;
- official five-stage Mousse Newton--Schulz coefficients in BF16;
- unwhitening and Frobenius-norm grafting;
- spectral-norm LR adjustment and hidden weight decay `0.01`;
- no Newton--Muon activation-K state.

The auxiliary tied embedding/head parameter stays on R1 fused AdamW with LR
`0.0036`, betas `(0.9, 0.95)`, and zero weight decay.

## 45A audit gates

`--preflight` fails closed on the pinned R1 commit/source, FineWeb shards,
accepted R1 runtime family, initialization hash, exhaustive 48+1 parameter
routing, 12 packed QKV tensors / 72 logical matrices, and a 12-step small
matrix audit. The small audit verifies state schema, finite factors/eigensystems,
QKV splitting, refreshes at steps 1 and 11, and zero activation-K routes.

The 34-step exact-shape smoke is a separate formal gate. It must report four
refreshes for every logical matrix and a total refresh count of `72 * 4`.

## 45B pilot and selection

Seed 2026 runs exactly three 1000-update cells:

| cell | mapped matrix LR |
|---|---:|
| `mousse_lr080` | 0.012 |
| `mousse_lr100` | 0.015 |
| `mousse_lr120` | 0.018 |

The center maps the upstream `0.06` recipe from global batch 2048 to the R1
global batch 512 by linear scaling: `0.06 * 512 / 2048 = 0.015`. This is a
predeclared mapping, not an observed-result adjustment.

Selection uses only step-1000 validation loss. The center is selected if it is
within `0.002` of the lowest loss; otherwise the lowest-loss cell is selected.
The controller writes `pilot_selection.json` only after all three local cells
are valid. `analyze_mousse_pilot.py` independently recomputes the certificate.

## 45C formal profile

The unique selected LR is run at seeds 2026, 2024, and 2025. Every formal run
has 6200 updates, 3,250,585,600 tokens, validation every 100 updates, and an
1800-update terminal warmdown. A same-seed 34-step smoke certificate is
required before each formal run. Formal checkpoints are required.

Local logs, curves, source/diff, summary, state schema/bytes, checkpoint
metadata, and manifest are validated before W&B upload. A failed W&B upload
can be retried with `--resume-batch`; a locally accepted training run is not
rerun. Timing is always marked ineligible because two-GPU concurrency is used
for turnaround, not for a paper throughput comparison.

The ready-to-run R1-host workflow is
`commands/45_r1_mousse_strong_baseline/20260730_r1_mousse_strong_baseline.sh`. It uses GPU 0 for preflight
and the pilot, then runs seeds 2026 and 2024 concurrently on GPUs 0 and 1,
followed by seed 2025.

## Unified analysis

After the three formal batches finish, run:

```bash
python scripts/45_r1_mousse_strong_baseline/analyze_mousse_formal.py \
  --mousse-summaries FORMAL_BATCH_2026 FORMAL_BATCH_2024 FORMAL_BATCH_2025 \
  --core-summary "${SNM_RESULTS_ROOT}/15_official_newton_muon_r1/analysis/wandb_20260721_multiseed_factorial/r1_multiseed_run_summary.csv" \
  --extended-summary "${SNM_RESULTS_ROOT}/19_r1_extended_baselines/analysis/wandb_20260723_formal_multiseed_unified/extended_formal_run_summary.csv" \
  --output-dir "${SNM_RESULTS_ROOT}/45_r1_mousse_strong_baseline/analysis/FINAL_DIRECTORY"
```

The analyzer first issues a historical identity/reuse certificate. It requires
all three new local formal manifests and the frozen experiment-15/19 analysis
manifests. The current historical bundle consists of accepted W&B exports and
analysis manifests, not the original per-run local manifests, so the
certificate says `passed_with_caveats`, records that evidence limitation, and
keeps timing comparisons prohibited. The practical-equivalence margin is
`±0.002`; paired-t intervals with three seeds are reported descriptively.
