# Experiment Script Registry

Scripts stay in this directory so existing commands keep working. Use this
registry to find the right runner or builder by experiment family.

For new publication experiments, prefer the model-agnostic artifact root:

```text
scripts/
```

Keep this directory for existing paper-version scripts and method-level helpers.

## Static Release and Pareto

| Script | Use |
|---|---|
| `run_owt_strong_static_baseline.py` | Tier3 strong static baselines. |
| `build_static_center_sweep_masks.py` | Build center-band release-ratio masks. |
| `run_static_center_sweep.py` | Run static center release-ratio sweep. |
| `run_tier3_50m_static_recheck.py` | 50M-token Tier3 static recheck and release-point runs. |
| `build_model_center_cproj_mask.py` | Build center `mlp.c_proj` masks for larger model tiers. |
| `run_owt_large_model_static.py` | Larger-model static release runs. |

## Dynamic and Probe Rules

| Script | Use |
|---|---|
| `build_cheap_muon_probe_masks.py` | Build masks from cheap Muon probe reports. |
| `run_cheap_muon_probe_replay.py` | Replay cheap-probe masks. |
| `run_update_similarity_probe.py` | Collect update-similarity probe metrics. |

## Mechanism and Counterfactuals

| Script | Use |
|---|---|
| `analyze_existing_static_mechanism.py` | Analyze existing static-rule evidence. |
| `build_mechanism_counterfactual_masks.py` | Build mechanism counterfactual masks. |
| `run_mechanism_counterfactuals.py` | Run Tier3 mechanism counterfactuals. |
| `run_tier3_50m_mechanism_counterfactuals.py` | Run 50M-token mechanism counterfactuals. |

## Legacy or Early Exploration

| Script | Use |
|---|---|
| `run_owt_storage_matrix.py` | Early OWT storage accounting matrix. |
| `run_owt_band_prior.py` | Early band-prior experiments. |
| `run_owt_oracle_static.py` | Early oracle static experiments. |
| `run_owt_shape_prior.py` | Early shape-prior experiments. |
| `run_selective_warmup_matrix.py` | Tiny Shakespeare warmup matrix. |
| `run_storage_matrix.py` | Tiny Shakespeare storage matrix. |
| `run_storage_pareto_matrix.py` | Tiny Shakespeare Pareto matrix. |

## Naming Rule for Future Runners

For new publication experiments, use names like:

```text
scripts/01_scale_up/run_scale_up_300m.py
scripts/02_long_token_budget/run_long_budget_100m.py
scripts/03_fixed_memory/run_fixed_memory_batch_sweep.py
```

Default outputs should point outside the source tree, under:

```text
runs/
```
