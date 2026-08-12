# Commands for 40 Llama Block Partition Invariance Audit

- Status: `implemented`
- Code: `../../scripts/40_llama_block_partition_invariance_audit`
- Source-freeze tier: `partial_or_run_specific_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `false`
- Native verification: `false`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 40_llama_block_partition_invariance_audit
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 40_llama_block_partition_invariance_audit
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  40_llama_block_partition_invariance_audit --results-root /path/to/results \
  --run-dir /path/to/results/40_llama_block_partition_invariance_audit/RUN_ID
```

## Entrypoints

- `command:20260729_llama_block_partition_invariance_audit` -> `commands/40_llama_block_partition_invariance_audit/20260729_llama_block_partition_invariance_audit.sh` (reproduce)
- `script:run_llama_block_partition_audit` -> `scripts/40_llama_block_partition_invariance_audit/run_llama_block_partition_audit.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
