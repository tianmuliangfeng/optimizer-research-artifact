# 54 Llama Moonlight Multiscale Multibudget

- Status: `implemented`
- Code: `../../scripts/54_llama_moonlight_multiscale_multibudget`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `false`
- One-click rerun: `false`
- Native resume: `true`
- Native verification: `true`

## Accepted result

Moonlight is strong at LLaMA 124M but does not preserve that ordering at
LLaMA 1B. Timing is excluded and the 1B result is not pooled with EX57.

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 54_llama_moonlight_multiscale_multibudget
```

A fresh rerun is not declared for this archived experiment.
Use the native archival validator below when available.

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  54_llama_moonlight_multiscale_multibudget --results-root /path/to/results \
  --run-dir /path/to/results/54_llama_moonlight_multiscale_multibudget/RUN_ID
```

Resume an interrupted native run:

```bash
python reproducibility/reproduce.py resume \
  54_llama_moonlight_multiscale_multibudget --results-root /path/to/results \
  --run-dir /path/to/results/54_llama_moonlight_multiscale_multibudget/RUN_ID
```

Run the experiment-specific native validator:

```bash
python reproducibility/reproduce.py native-verify \
  54_llama_moonlight_multiscale_multibudget --results-root /path/to/results \
  --run-dir /path/to/results/54_llama_moonlight_multiscale_multibudget/RUN_ID
```

## Entrypoints

- `command:20260819_ex54_llama_moonlight_multiscale_multibudget` → `commands/54_llama_moonlight_multiscale_multibudget/20260819_ex54_llama_moonlight_multiscale_multibudget.sh` (native-verify, resume)
- `script:run_suite` → `scripts/54_llama_moonlight_multiscale_multibudget/run_suite.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
