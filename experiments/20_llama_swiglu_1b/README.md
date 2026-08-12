# 20 Llama Swiglu 1B

- Status: `implemented`
- Code: `../../scripts/20_llama_swiglu_1b`
- Source-freeze tier: `legacy_command_or_live_source`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `false`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 20_llama_swiglu_1b
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 20_llama_swiglu_1b --entrypoint script:run_llama_swiglu_1b --arg=--stage --arg=formal --arg=--official-repo --arg=/path/to/value --arg=--python-exe --arg=/path/to/value
python reproducibility/reproduce.py reproduce 20_llama_swiglu_1b --entrypoint script:run_llama_swiglu_1b_capacity --arg=--official-repo --arg=/path/to/value --arg=--python-exe --arg=/path/to/value
python reproducibility/reproduce.py reproduce 20_llama_swiglu_1b --entrypoint script:run_llama_swiglu_1b_capacity_cell --arg=--stage --arg=formal --arg=--official-repo --arg=/path/to/value --arg=--python-exe --arg=/path/to/value
python reproducibility/reproduce.py reproduce 20_llama_swiglu_1b --entrypoint script:run_llama_swiglu_1b_capacity_exact --arg=--fine-manifest --arg=/path/to/value --arg=--official-repo --arg=/path/to/value --arg=--python-exe --arg=/path/to/value
python reproducibility/reproduce.py reproduce 20_llama_swiglu_1b --entrypoint script:run_llama_swiglu_1b_capacity_fine --arg=--official-repo --arg=/path/to/value --arg=--python-exe --arg=/path/to/value
python reproducibility/reproduce.py reproduce 20_llama_swiglu_1b --entrypoint script:run_llama_swiglu_1b_capacity_fine_cell --arg=--stage --arg=formal --arg=--official-repo --arg=/path/to/value --arg=--python-exe --arg=/path/to/value --arg=--capacity-accumulation-steps --arg=8 --arg=--device-batch-size --arg=16
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  20_llama_swiglu_1b --results-root /path/to/results \
  --run-dir /path/to/results/20_llama_swiglu_1b/RUN_ID
```

## Entrypoints

- `script:run_llama_swiglu_1b` → `scripts/20_llama_swiglu_1b/run_llama_swiglu_1b.py` (reproduce; requires receipt-bound --arg values)
- `script:run_llama_swiglu_1b_capacity` → `scripts/20_llama_swiglu_1b/run_llama_swiglu_1b_capacity.py` (reproduce; requires receipt-bound --arg values)
- `script:run_llama_swiglu_1b_capacity_cell` → `scripts/20_llama_swiglu_1b/run_llama_swiglu_1b_capacity_cell.py` (reproduce; requires receipt-bound --arg values)
- `script:run_llama_swiglu_1b_capacity_exact` → `scripts/20_llama_swiglu_1b/run_llama_swiglu_1b_capacity_exact.py` (reproduce; requires receipt-bound --arg values)
- `script:run_llama_swiglu_1b_capacity_fine` → `scripts/20_llama_swiglu_1b/run_llama_swiglu_1b_capacity_fine.py` (reproduce; requires receipt-bound --arg values)
- `script:run_llama_swiglu_1b_capacity_fine_cell` → `scripts/20_llama_swiglu_1b/run_llama_swiglu_1b_capacity_fine_cell.py` (reproduce; requires receipt-bound --arg values)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
