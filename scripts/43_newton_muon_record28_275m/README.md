# 43 - Modded-NanoGPT approximately 275M near-Record-28 quality experiment

This directory implements the paired multi-seed quality comparison for:

- Muon;
- the original upstream Newton-Muon control;
- Selective-none;
- Selective-diag.

It is aligned to the upstream `train_gpt_muon_2.py` and
`train_gpt_newton_muon_2.py` recipe at commit
`df78af0db523d8bceb25af4919a3e3e7082b80f3`. It must be described as an
approximately 275M-parameter Modded-NanoGPT experiment near upstream Short
Track Record #28, not as an exact Record #28 reproduction.

The frozen formal design is four methods by seeds 2024, 2025, 2026, and 2027:
16 formal cells. Every cell performs 1,695 optimizer updates with 393,216
tokens per update, for exactly 666,501,120 training tokens. A four-cell
18-update smoke gate crosses the first Newton K refresh before formal
training.

The frozen execution domain is the otherwise-idle LLaMA H100 host. This is a
pre-formal provenance-only amendment: it does not change the model, methods,
seeds, data order, hyperparameters, endpoints, or analysis. The only remote
entry point is:

```bash
bash commands/43_newton_muon_record28_275m/20260730_newton_muon_record28_275m.sh
```

The default official checkout is:

```text
${SNM_OFFICIAL_REPO}
```

The controller and trainer deliberately use different environments:

```text
CTRL_PY=${SNM_CONTROLLER_PYTHON}
TRAIN_PY=${SNM_TRAINING_PYTHON}
```

The controller owns preflight, scheduling, analysis, and post-training W&B
uploads. The training environment does not need W&B.

To reserve only physical GPU0 for experiment 43 while experiment 44 uses
physical GPU1, launch with:

```bash
EXP43_GPUS=0 bash commands/43_newton_muon_record28_275m/20260730_newton_muon_record28_275m.sh
```

Both experiments must use their shared physical-GPU locks. Running 43 and 44
in parallel is valid for optimizer-quality results, but makes all wall-time,
tokens/s, and steps/s observations ineligible as paper efficiency evidence.

At first launch, the live source bootstraps a timestamped run and seals its
complete runtime dependency tree in `RUN_DIR/source_snapshot`. All subsequent
work and recovery are executed from that snapshot. Once the snapshot exists,
the printed recovery command invokes its controller directly and does not
import the live Python files. Before snapshot sealing only, recovery falls
back to the checked-in shell entry point. A shorthand equivalent is:

```bash
RUN_DIR="${SNM_RESULTS_ROOT}/43_newton_muon_record28_275m/<UTC timestamp>" \
EXP43_GPUS=0 \
bash commands/43_newton_muon_record28_275m/20260730_newton_muon_record28_275m.sh
```

A completed cell is reused only after its local manifest passes. An interrupted
cell is retained and restarted from initialization as a new numbered attempt.
The paired analysis is sealed before any W&B network operation. W&B uploads
then run with a bounded timeout. If an upload fails, use the upload-only
recovery emitted by the controller; a W&B failure must never block the
scientific analysis or retrain an accepted local cell. Formal `offline` or
`disabled` modes remain explicitly pending and are never reported as a
completed online handoff.

The primary outcome is final validation loss. Selective-none and
Selective-diag are each compared separately with Muon and the original
Newton-Muon control. Original Newton-Muon versus Muon is the mandatory
benchmark anchor; Selective-diag versus Selective-none is secondary only.
See `record28_contract.json` and `RECORD28_275M_CONTRACT.md` for the complete
frozen protocol and claim boundaries.
