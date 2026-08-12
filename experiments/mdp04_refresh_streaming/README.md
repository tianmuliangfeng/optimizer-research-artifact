# Mdp-04 Refresh Streaming

- Status: `implemented`
- Code: `../../scripts/mdp_refresh_streaming`
- Source-freeze tier: `sealed_source_snapshot`
- Generic artifact verification: `true`
- Fresh rerun: `false`
- One-click rerun: `false`
- Native resume: `false`
- Native verification: `true`

Inspect the frozen public entrypoints:

```bash
python reproducibility/reproduce.py inspect mdp04_refresh_streaming
```

A fresh rerun is not declared for this archived experiment.
Use the native archival validator below when available.

Verify an existing result without training. The run path must
match a declared legacy result root or carry a matching sealed
source-snapshot lineage:

```bash
python reproducibility/reproduce.py verify \
  mdp04_refresh_streaming --results-root /path/to/results \
  --run-dir /path/to/results/_shared/analysis/method_deepening_mdp04_refresh_replay/RUN_ID
```

Run the experiment-specific native validator:

```bash
python reproducibility/reproduce.py native-verify \
  mdp04_refresh_streaming --results-root /path/to/results \
  --run-dir /path/to/results/_shared/analysis/method_deepening_mdp04_refresh_replay/RUN_ID
```

## Entrypoints

- `command:20260803_mdp04_refresh_streaming` → `commands/mdp04_refresh_streaming/20260803_mdp04_refresh_streaming.sh` (native-verify)
- `command:reproduce_archived` → `commands/mdp04_refresh_streaming/reproduce_archived.sh` (explicit selection only)
- `script:run_stream_replay` → `scripts/mdp_refresh_streaming/run_stream_replay.py` (explicit selection only)

Historical results are not stored in this source repository. See
`../../docs/REPRODUCIBILITY.md` for evidence tiers and limitations.
