# MECH-05 frozen K-representation selection rule

MECH-05 is a CPU-only analysis. It does not train a model, load a checkpoint,
write W&B, or run MECH-04. It combines:

- MECH-02 endpoint geometry and stability;
- MECH-03 layer-scope held-out cross-fit shadow loss;
- existing seed2026 formal endpoint losses for R1 and LLaMA-124M.

The discovery set is exactly R1 and LLaMA-124M at seed2026/step6200. GPT bridge
is a runtime control and does not vote in the selection decision.

The frozen rule returns one of:

- `diag`;
- `full_or_block`;
- `none_or_muon_sufficient`;
- `uncertain`.

Geometry magnitude alone never selects a complex K representation. Stable
held-out functional gain is required. A conflict between endpoint diagnostics
and long-run quality returns `uncertain`.

The LLaMA-1B three-seed formal rankings were already known before this rule was
frozen. They are therefore retrospective context only and are not accepted as
an analyzer input. Only future MECH-06-L1B diagnostic artifacts can test the
frozen diagnostic rule prospectively; the pre-existing training rankings
remain retrospective.

Run the contract tests:

```bash
python scripts/32_mech05_frozen_selection_rule/test_mech05_contract.py
```

Run the canonical analysis with:

```bash
bash commands/32_mech05_frozen_selection_rule/20260727_mech05_freeze_selection_rule.sh
```

The command writes directly under experiment family `32`; it creates no code
archive.
