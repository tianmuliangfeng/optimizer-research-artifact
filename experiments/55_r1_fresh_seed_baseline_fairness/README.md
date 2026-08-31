# 55 R1 Fresh Seed Baseline Fairness

- Status: `implemented`
- Code: `../../scripts/55_r1_fresh_seed_baseline_fairness`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `false`
- One-click rerun: `false`
- Native resume: `true`
- Native verification: `true`

## Accepted result

A fresh-seed controlled panel confirms that the principal baseline
comparisons are not artifacts of reusing the earlier formal seeds.

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 55_r1_fresh_seed_baseline_fairness
```

A fresh rerun is not declared for this archived experiment.
Use the native archival validator below when available.

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  55_r1_fresh_seed_baseline_fairness --results-root /path/to/results \
  --run-dir /path/to/results/55_r1_fresh_seed_baseline_fairness/RUN_ID
```

Resume an interrupted native run:

```bash
python reproducibility/reproduce.py resume \
  55_r1_fresh_seed_baseline_fairness --results-root /path/to/results \
  --run-dir /path/to/results/55_r1_fresh_seed_baseline_fairness/RUN_ID
```

Run the experiment-specific native validator:

```bash
python reproducibility/reproduce.py native-verify \
  55_r1_fresh_seed_baseline_fairness --results-root /path/to/results \
  --run-dir /path/to/results/55_r1_fresh_seed_baseline_fairness/RUN_ID
```

## Entrypoints

- `command:20260817_ex55_r1_fresh_seed_baseline_fairness` → `commands/55_r1_fresh_seed_baseline_fairness/20260817_ex55_r1_fresh_seed_baseline_fairness.sh` (native-verify, resume)
- `command:20260819_ex55_repair_tail5_analysis` → `commands/55_r1_fresh_seed_baseline_fairness/20260819_ex55_repair_tail5_analysis.sh` (native-verify, resume)
- `script:run_fresh_seed_suite` → `scripts/55_r1_fresh_seed_baseline_fairness/run_fresh_seed_suite.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
