# 57 Llama1B 10B Moonlight

- Status: `implemented`
- Code: `../../scripts/57_llama1b_10b_moonlight`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `false`
- One-click rerun: `false`
- Native resume: `true`
- Native verification: `true`

## Accepted result

At the same long LLaMA-1B budgets, Moonlight trails Muon at every paired
endpoint and trails all core curvature-state routes at the two longer budgets.

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 57_llama1b_10b_moonlight
```

A fresh rerun is not declared for this archived experiment.
Use the native archival validator below when available.

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  57_llama1b_10b_moonlight --results-root /path/to/results \
  --run-dir /path/to/results/57_llama1b_10b_moonlight/RUN_ID
```

Resume an interrupted native run:

```bash
python reproducibility/reproduce.py resume \
  57_llama1b_10b_moonlight --results-root /path/to/results \
  --run-dir /path/to/results/57_llama1b_10b_moonlight/RUN_ID
```

Run the experiment-specific native validator:

```bash
python reproducibility/reproduce.py native-verify \
  57_llama1b_10b_moonlight --results-root /path/to/results \
  --run-dir /path/to/results/57_llama1b_10b_moonlight/RUN_ID
```

## Entrypoints

- `command:20260819_ex57_llama1b_10b_moonlight` → `commands/57_llama1b_10b_moonlight/20260819_ex57_llama1b_10b_moonlight.sh` (native-verify, resume)
- `script:run_suite` → `scripts/57_llama1b_10b_moonlight/run_suite.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
