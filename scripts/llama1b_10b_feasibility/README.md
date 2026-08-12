# LLaMA-1B approximately-10B-token feasibility package

This directory is a CPU-only planning and audit package. It is deliberately
unnumbered, contains no training worker or remote launch command, and assigns
no experiment number. Passing its checks does not authorize a 10B run.

The package freezes only facts that are already implied by the accepted
LLaMA-1B profile and roadmap:

- four methods: `down_none`, `down_diag`, `newton_full`, and `muon`;
- seed 2026 screening only;
- global batch 512, context 1024, and 524,288 training tokens/update;
- milestone steps 6200, 13293, and 19073, corresponding to 3,250,585,600,
  6,969,360,384, and 9,999,745,024 actual training tokens;
- at least 101 consecutively numbered unique train shards, zero wrap, and one
  prefetched microbatch included in physical stream capacity;
- planning estimates derived from accepted three-seed step times and accepted
  checkpoint sizes;
- a disabled geometry interface with no selected formula or layers.

## Why the current trainer is not launch-ready

The audit intentionally reports hard blockers:

1. the current data controller requires exactly 50 train shards;
2. the current sequential loader advances with modulo and records no
   `wrap_count`;
3. the current formal stage is fixed to 6200 steps;
4. validation cannot force the non-round step 13293;
5. the long-horizon LR schedule is unresolved. The accepted 6200-step run
   warms down for its final 1800 steps and reaches zero LR, so it cannot be
   continued. A from-scratch 19073-step run with only a final warmdown would
   not be schedule-matched to the accepted step-6200 endpoint.

The current planning recommendation, still deliberately unselected, is a new
from-scratch 19073-step schedule with one final 1800-step warmdown. Under that
candidate, the whole long trajectory is a new experiment schedule: its
step-6200 checkpoint is useful as an internal milestone but is not a
schedule-matched replication of the accepted 6200-step run. This choice must
be red-teamed and frozen in a later launch contract before any training.

These are feasibility findings, not instructions to modify the trainer. A
future authorized experiment needs a new preregistered launch contract and a
separate controller.

## Local commands

Contract and source compatibility check:

```bash
python scripts/llama1b_10b_feasibility/audit_10b_feasibility.py check
```

Generate a planning-only report without inspecting remote data:

```bash
python scripts/llama1b_10b_feasibility/audit_10b_feasibility.py plan \
  --output-dir /tmp/llama1b_10b_plan
```

Header-only remote data feasibility audit (CPU and filesystem only):

```bash
python scripts/llama1b_10b_feasibility/audit_10b_feasibility.py audit-data \
  --data-dir ${SNM_OFFICIAL_REPO}/data/fineweb10B \
  --output-dir /tmp/llama1b_10b_data_audit
```

The data audit reads each 1024-byte FineWeb header and file metadata. It does
not load token payloads, initialize CUDA, launch training, use W&B, or modify
the dataset. Full content hashes remain a future launch-contract decision.

Run tests with any standard Python; PyTorch is not required:

```bash
python -m unittest scripts/llama1b_10b_feasibility/test_feasibility.py -v
```

Every generated report and manifest contains `launch_authorized=false` and
`scientific_evidence_class=none_planning_only`.

## Post-mechanism-closure value decision

The original feasibility contract remains unchanged and planning-only. After
the final unified analysis and mechanism closure were accepted, a separate
decision layer was added without granting launch authority:

```bash
python scripts/llama1b_10b_feasibility/evaluate_10b_value.py check
python scripts/llama1b_10b_feasibility/evaluate_10b_value.py build \
  --output-dir /tmp/llama1b_10b_value_assessment
```

The frozen 2026-08-05 decision is `do_not_launch_now_keep_reviewer_triggered_contingency`.
Only a blind paper red-team identifying post-3.25B LLaMA undertraining as an
acceptance-critical blocker may reopen the gate. If reopened, the read-only
101+ shard audit and LR-policy review happen before any launch code is written.
