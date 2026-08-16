# 52 Llama Global Diag Scale

- Status: `implemented`
- Code: `../../scripts/52_llama_global_diag_scale`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `false`
- One-click rerun: `false`
- Native resume: `true`
- Native verification: `true`

## Accepted result

Across LLaMA 124M and 1B, global activation diagonals are a strong low-state
alternative, while Muon remains marginally best at 1B. The reused 1B W&B
run history stitches the screen and formal continuation, so formal claims
remain bound to local accepted CSVs and manifests.

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 52_llama_global_diag_scale
```

A fresh rerun is not declared for this archived experiment.
Use the native archival validator below when available.

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  52_llama_global_diag_scale --results-root /path/to/results \
  --run-dir /path/to/results/52_llama_global_diag_scale/RUN_ID
```

Resume an interrupted native run:

```bash
python reproducibility/reproduce.py resume \
  52_llama_global_diag_scale --results-root /path/to/results \
  --run-dir /path/to/results/52_llama_global_diag_scale/RUN_ID
```

Run the experiment-specific native validator:

```bash
python reproducibility/reproduce.py native-verify \
  52_llama_global_diag_scale --results-root /path/to/results \
  --run-dir /path/to/results/52_llama_global_diag_scale/RUN_ID
```

## Entrypoints

- `command:20260814_ex52_llama_global_diag_scale` → `commands/52_llama_global_diag_scale/20260814_ex52_llama_global_diag_scale.sh` (native-verify, resume)
- `script:run_llama_global_diag_suite` → `scripts/52_llama_global_diag_scale/run_llama_global_diag_suite.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
