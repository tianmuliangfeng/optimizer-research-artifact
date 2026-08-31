# 56 Llama1B 10B Global Diag

- Status: `implemented`
- Code: `../../scripts/56_llama1b_10b_global_diag`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `false`
- One-click rerun: `false`
- Native resume: `true`
- Native verification: `true`

## Accepted result

Across three long-budget LLaMA-1B checkpoints and three seeds, global
diagonal beats the local curvature-state routes in every paired endpoint;
Muon remains better than global diagonal in every paired endpoint.

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 56_llama1b_10b_global_diag
```

A fresh rerun is not declared for this archived experiment.
Use the native archival validator below when available.

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  56_llama1b_10b_global_diag --results-root /path/to/results \
  --run-dir /path/to/results/56_llama1b_10b_global_diag/RUN_ID
```

Resume an interrupted native run:

```bash
python reproducibility/reproduce.py resume \
  56_llama1b_10b_global_diag --results-root /path/to/results \
  --run-dir /path/to/results/56_llama1b_10b_global_diag/RUN_ID
```

Run the experiment-specific native validator:

```bash
python reproducibility/reproduce.py native-verify \
  56_llama1b_10b_global_diag --results-root /path/to/results \
  --run-dir /path/to/results/56_llama1b_10b_global_diag/RUN_ID
```

## Entrypoints

- `command:20260817_ex56_llama1b_10b_global_diag` → `commands/56_llama1b_10b_global_diag/20260817_ex56_llama1b_10b_global_diag.sh` (native-verify, resume)
- `script:run_formal` → `scripts/56_llama1b_10b_global_diag/run_formal.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
