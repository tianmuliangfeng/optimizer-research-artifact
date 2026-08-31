# Experiment 55 — fresh-seed 124M baseline-fairness repair

EX55 repairs one narrowly defined selection concern in the Modded-GPT 124M
comparison.  Six external baselines used seed 2026 during their historical
configuration selection.  EX55 does **not** retune them and does not discard
any historical result.  It runs one new, common formal seed (2027) for the ten
frozen methods and constructs a paired `2024/2025/2027` panel.

The ten mandatory methods are local block-4, local diagonal, local none, Muon,
AdamW, NorMuon, Moonlight, Mousse, MALT, and MALTER-Eq17.  Their configurations
are the exact previously accepted winners.  The accepted 2024/2025/2026 panel
is copied unchanged into the analysis output and remains separately auditable.
`Moonlight` is the canonical analysis label; the frozen EX19 worker exposes the
same method as `moonlight_muon` on its command line and in its manifests.  EX55
records this interface-only alias without changing the selected
`moonlight_r1scale` cell.

## Evidence roles

- Seed 5501 is a 34-step outcome-free engineering smoke.  It cannot select a
  method, learning rate, route, or reporting rule.
- Seed 2027 first passes a same-seed exact-shape smoke and is then trained for
  the frozen 6200-step R1 protocol.
- W&B is secondary.  Locally complete, hash-bound evidence remains valid when
  an online W&B upload is incomplete.  `resume` restores local training and
  verification only; this suite intentionally has no independent W&B re-upload
  stage, and W&B completeness is not a scientific acceptance gate.
- Timing is ineligible because GPU 1 may run Experiment 55 while the separately
  isolated GPU 0 runs Experiment 53. Experiment 55 itself uses only physical
  GPU 1 and schedules all of its units serially.
- The leave-selection-seed-out `2024/2025` table is descriptive (`n=2`); it
  deliberately reports no confidence interval or p-value.

## Stages and recovery

The command wrapper exposes `preflight`, `pilot`, `formal`, `verify`, `resume`,
and `all`.  `resume` requires the exact existing `EX55_RUN_DIR` and advances
through only the incomplete stages.  Reuse that directory for every later stage.  Both the suite
and each underlying batch are resumable: accepted units are skipped and an
interrupted batch is resumed in place.

The inherited runner families expose slightly different formal-smoke summary
schemas and artifact filenames.  The core runner writes `r1_summary.json` /
`r1_metrics.csv`, whereas newer family runners may write `summary.json` /
`metrics.csv`.  EX55 accepts either an aggregate `initial_val_loss` /
`val_loss_step_0` field or, when an accepted runner omits it, the unique
accepted child's validation row at step 0.  The child manifest, summary, and
metrics SHA-256 are sealed into `formal_smoke_pairing_manifest.json`.  This is
evidence extraction only: it never reruns or changes an accepted smoke,
selected cell, formal seed, or training configuration.

The accepted EX15 numerical-smoke source deliberately evaluates one 65,536-token
validation batch, while the other frozen formal-smoke runners retain the full
10,485,760-token validation budget.  Consequently, the pairing certificate
hard-gates the exact common parameter-initialization hash across all ten methods
and checks step-0 loss equality only within each validation-token stratum.  It
does not compare the two smoke losses as if they used the same estimator.  All
ten 6,200-step formal runs use the full 10,485,760-token validation budget; the
final analyzer separately requires their full-budget initial losses to agree.

The inherited **formal** aggregate summaries are heterogeneous as well.  In
particular, a valid core aggregate may expose fewer than five `val_loss_step_*`
fields even though its accepted child `r1_metrics.csv` contains the complete
100-step validation trajectory.  The final analyzer therefore resolves the
unique accepted formal child for every method, reconstructs the required
`5800/5900/6000/6100/6200` tail-5 directly from `r1_metrics.csv` or
`metrics.csv`, and cross-checks step 0, step 6200, and every aggregate milestone
that is present.  It writes each metrics path, byte count, and SHA-256—together
with the accepted child manifest/summary lineage—to
`analysis/formal_metrics_tail5_lineage.json`.  Final `verify` rehashes those
external evidence files, so this compatibility repair neither relaxes the
tail-5 requirement nor fabricates missing milestones.

For a run whose immutable `source_snapshot` predates this repair, the wrapper
uses the live amended controller/analyzer without editing the snapshot.  A
hash-bound `analysis_amendment_tail5_metrics_*.json` receipt records the old and
new analyzer hashes and explicitly states that no seed, method,
hyperparameter, checkpoint, aggregate manifest, or metrics file changed.

`preflight` copies and SHA-256 verifies four accepted inputs:

1. the authoritative EX49 ten-method historical panel;
2. the accepted EX19 three-method extended-baseline selection table;
3. the authoritative EX45 Mousse selection certificate;
4. the authoritative EX49 MALT/MALTER selection certificate.

These four small authoritative files are stored byte-exactly as portable base64
payloads under `accepted_inputs_encoded/`. The wrapper decodes them into the
new run directory and verifies their original byte counts and accepted SHA-256
values before the controller starts; the remote host therefore does not need
the later local-analysis directory layout. The first preflight still validates
the deep EX49 pilot/source lineage referenced by the packaged MALT certificate
against the original accepted EX49 run, then copies that lineage into the EX55
run for self-contained resume.

Immediately after those inputs are frozen, the controller writes
`analysis_preformal/leave_selection_seed_out_2024_2025.csv` and a hash-bound
manifest.  This is the zero-GPU, descriptive `n=2` sensitivity; it is available
before seed 2027 training and is never promoted to an inferential test.

It also freezes the first 50 FineWeb training shards plus the validation shard,
computes a full SHA-256 for every one of those 51 files, audits all four
implementation families, and snapshots all executable source.  The resulting
inventory is byte-compatible with the accepted EX49 MALT pilot inventory.  At
the first preflight, EX55 copies the accepted MALT aggregate and all twelve
source manifests into the run and rebinds paths without changing any selected
cell; later resume therefore does not depend on the original EX49 run paths.

Typical execution:

```bash
EX55_RUN_DIR="${SNM_RESULTS_ROOT:-$PWD/runs}/55_r1_fresh_seed_baseline_fairness/$(date -u +%Y%m%dT%H%M%S+0000)"
EX55_GPUS="1" RUN_DIR="$EX55_RUN_DIR" bash commands/55_r1_fresh_seed_baseline_fairness/20260817_ex55_r1_fresh_seed_baseline_fairness.sh all
```

After any interruption:

```bash
EX55_RUN_DIR=/absolute/path/to/the/existing/run \
bash commands/55_r1_fresh_seed_baseline_fairness/20260817_ex55_r1_fresh_seed_baseline_fairness.sh resume
```

The formal checkpoint artifacts are large.  Do not delete them before
`verify`; after formal, the controller records a full SHA-256 for each of the
ten checkpoints.  `verify` independently rechecks path, byte count, and full
content hash for all ten, and also rehashes the 51-file data inventory.
