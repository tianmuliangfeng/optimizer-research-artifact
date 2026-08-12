# Commands for 48 Llama1B 10B Multibudget

- Status: `implemented`
- Code: `../../scripts/48_llama1b_10b_multibudget`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `true`
- One-click rerun: `true`
- Native resume: `true`
- Native verification: `true`

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
python reproducibility/reproduce.py reproduce \
  48_llama1b_10b_multibudget \
  --env 'EX48_GPUS=0 1 2 3'
```

The formal contract requires one host with four H100 GPUs and
`EX48_GPUS="0 1 2 3"`. Repeat the exact receipt-bound environment when
executing the printed plan:

```bash
python reproducibility/reproduce.py reproduce \
  48_llama1b_10b_multibudget \
  --env 'EX48_GPUS=0 1 2 3' \
  --execute --receipt <plan_sha256>
```

An incomplete run produced under the earlier two-GPU setup is not admissible
evidence and must not be resumed. Start a new run directory
under the four-GPU contract instead.

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

- `command:20260805_ex48_llama1b_10b_multibudget` -> `commands/48_llama1b_10b_multibudget/20260805_ex48_llama1b_10b_multibudget.sh` (native-verify, resume)
- `command:reproduce_full` -> `commands/48_llama1b_10b_multibudget/reproduce_full.sh` (reproduce)
- `script:run_formal` -> `scripts/48_llama1b_10b_multibudget/run_formal.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
