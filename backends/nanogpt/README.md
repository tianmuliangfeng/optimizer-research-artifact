# Selective Newton-Muon

This folder is the selective layer-gating experiment line. It keeps Muon and
Newton-Muon baselines, then evaluates Selective Newton-Muon variants that use
input-side Newton-Muon only on selected matrix layers.

The Sigma-side experiments live in:

```text
${SNM_WORKSPACE_ROOT}/block-sigma-newton-muon
```

## Layout

```text
config/
  baselines/
    00_muon.py
    13_newton_muon_fast.py
  selective/
    10_selective_v2_top75_fast_warmup100.py
    14_selective_v2_top75_storage_warmup100.py
  ablations/
    07_selective_v2_top50_fast_warmup100.py
    08_selective_v2_top50_fast_warmup50.py
    09_selective_v2_top50_fast_warmup25.py
    11_selective_v2_top75_fast_warmup50.py
    12_selective_v2_top75_fast_warmup25.py
  train_shakespeare_char.py
  smoke_cpu.py
scripts/
  run_selective_warmup_matrix.py
  run_storage_matrix.py
```

## Current Variants

The current loss/time candidate is:

```text
config/selective/10_selective_v2_top75_fast_warmup100.py
```

The storage candidate is:

```text
config/selective/14_selective_v2_top75_storage_warmup100.py
```

It uses:

```text
selective_fraction = 0.75
selective_warmup_steps = 100
selective_score_mode = "gain_logcond_cost_power"
selective_cost_power = 0.25
selective_freeze_after_warmup = True
selective_log_diagnostics = False
selective_release_inactive_k_state = True  # storage candidate only
```

## Main Commands

Prepare data if needed:

```powershell
python data/shakespeare_char/prepare.py
```

Print the selective warmup matrix:

```powershell
python scripts/run_selective_warmup_matrix.py --dry-run
```

Run the selective warmup matrix:

```powershell
python scripts/run_selective_warmup_matrix.py
```

Run the storage comparison:

```powershell
python scripts/run_storage_matrix.py --dry-run
python scripts/run_storage_matrix.py
```

Prepare or refresh the OpenWebText GPT-2 subset:

```powershell
python data/openwebtext_gpt2/prepare.py
```

Run the OpenWebText storage matrix on model tier 1:

```powershell
python scripts/run_owt_storage_matrix.py --dry-run
python scripts/run_owt_storage_matrix.py --tiers tier1
```

Run both OpenWebText model tiers:

```powershell
python scripts/run_owt_storage_matrix.py --tiers tier1 tier2
```

Run the OpenWebText oracle-static selective check. This uses the saved
`selective_layer_report` mask and skips allocating Newton K-state for
Muon-only layers from step 0:

```powershell
python scripts/run_owt_oracle_static.py --dry-run --tiers tier1 --seeds 1337 2024
python scripts/run_owt_oracle_static.py --tiers tier1 --seeds 1337 2024
```

Run the tier2 oracle-static check after generating the tier2 dynamic layer
report:

```powershell
python scripts/run_owt_oracle_static.py --dry-run --tiers tier2 --seeds 1337 2024
python scripts/run_owt_oracle_static.py --tiers tier2 --seeds 1337 2024
```

Run the practical shape-prior static selector. This does not use an oracle mask;
it releases low-priority high-cost MLP projection layers from step 0:

```powershell
python scripts/run_owt_shape_prior.py --dry-run --tiers tier2 --seeds 1337 2024
python scripts/run_owt_shape_prior.py --tiers tier2 --seeds 1337 2024
```

Run the tier2 band-prior sweep. This compares early/middle/late/edge MLP
projection bands at the same release40 K-state budget:

```powershell
python scripts/run_owt_band_prior.py --dry-run --tiers tier2 --seeds 1337 2024
python scripts/run_owt_band_prior.py --tiers tier2 --seeds 1337 2024
```

Run only the main candidate:

```powershell
python train.py config/train_shakespeare_char.py config/selective/10_selective_v2_top75_fast_warmup100.py
```

Run only the storage candidate:

```powershell
python train.py config/train_shakespeare_char.py config/selective/14_selective_v2_top75_storage_warmup100.py
```

The default W&B project for this folder is:

```text
Selective-Newton-Muon
```

The OpenWebText storage experiments use:

```text
Selective-Newton-Muon-OWT
```

## Notes

Old v1 selective configs and the earlier diagnostic top50/top75 configs were
removed from this folder. The remaining ablations are the warmup/fraction sweep
that led to the current top75/w100 candidate.
