# 48 Llama1B 10B Multibudget

- Status: `implemented`
- Code: `../../scripts/48_llama1b_10b_multibudget`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `true`
- Native verification: `true`

## Accepted execution geometry

The replacement formal protocol uses one physical host with exactly four H100
80GB GPUs (`EX48_GPUS="0 1 2 3"` by default). The interrupted two-GPU attempt
was deleted, is not resumable, and is not accepted as evidence. This amendment
changes scheduling and wall-clock only; methods, seeds, token budgets, data
order, and analysis rules remain frozen.

## Independent acceptance audit

After a formal run has been copied locally, independently rebuild the endpoint
and paired tables and bind the persisted remote full-checkpoint re-hash receipt:

```bash
python scripts/48_llama1b_10b_multibudget/audit_received_results.py \
  --run-dir /path/to/results/48_llama1b_10b_multibudget/RUN_ID \
  --received-dir /path/to/received-files \
  --output-dir /path/to/final-acceptance
```

The receipt is a semantic certificate for the frozen verifier's full re-hash
of all 36 retained endpoints. It is not a forensic execution log: the compact
JSON does not itself record the command, process exit code, host, or Python
environment, so the audit binds its semantics to the hash-frozen verifier.

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect 48_llama1b_10b_multibudget
```

Build a fresh reproduction plan (read-only by default).
Machine-specific settings must use repeated `--env KEY=VALUE`
arguments, while runner-specific options use repeated `--arg`
values, so both are covered by the plan receipt. A plan may be
inspected with missing required arguments, but execution is refused
until every declared argument group is satisfied:

```bash
python reproducibility/reproduce.py reproduce 48_llama1b_10b_multibudget
```

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  48_llama1b_10b_multibudget --results-root /path/to/results \
  --run-dir /path/to/results/48_llama1b_10b_multibudget/RUN_ID
```

Resume an interrupted native run:

```bash
python reproducibility/reproduce.py resume \
  48_llama1b_10b_multibudget --results-root /path/to/results \
  --run-dir /path/to/results/48_llama1b_10b_multibudget/RUN_ID
```

Run the experiment-specific native validator:

```bash
python reproducibility/reproduce.py native-verify \
  48_llama1b_10b_multibudget --results-root /path/to/results \
  --run-dir /path/to/results/48_llama1b_10b_multibudget/RUN_ID
```

## Entrypoints

- `command:20260805_ex48_llama1b_10b_multibudget` → `commands/48_llama1b_10b_multibudget/20260805_ex48_llama1b_10b_multibudget.sh` (native-verify, resume)
- `command:reproduce_full` → `commands/48_llama1b_10b_multibudget/reproduce_full.sh` (reproduce)
- `script:audit_received_results` → `scripts/48_llama1b_10b_multibudget/audit_received_results.py` (explicit selection only)
- `script:run_formal` → `scripts/48_llama1b_10b_multibudget/run_formal.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
