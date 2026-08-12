# Commands for 17 Llama Swiglu Validation

- Status: `implemented`
- Code: `../../scripts/17_llama_swiglu_validation`
- Source-freeze tier: `legacy_command_or_live_source`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `false`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 17_llama_swiglu_validation
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 17_llama_swiglu_validation --arg=--official-repo --arg=/path/to/value --arg=--python-exe --arg=/path/to/value
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  17_llama_swiglu_validation --results-root /path/to/results \
  --run-dir /path/to/results/17_llama_swiglu_validation/RUN_ID
```

## Entrypoints

- `script:run_llama_swiglu_validation` -> `scripts/17_llama_swiglu_validation/run_llama_swiglu_validation.py` (reproduce; requires receipt-bound --arg values)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
