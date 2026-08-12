# Publication Experiment Scripts

Put new experiment scripts here, grouped by the experiment family from the
publication experiment plan.

| Folder | Experiment family |
|---|---|
| `01_scale_up/` | Larger model scale-up experiments. |
| `02_long_token_budget/` | Longer token budget experiments. |
| `03_fixed_memory/` | Fixed-memory practical value experiments. |
| `04_dataset_generalization/` | Dataset generalization experiments. |
| `05_memory_efficient_baselines/` | Adafactor, 8-bit optimizer, GaLore, and related baselines. |
| `06_kstate_spectrum/` | K-state spectrum, effective-rank, and covariance diagnostics. |
| `07_random_irregular_masks/` | Random, periodic, and irregular mask controls. |
| `08_layer_sensitivity_restoration/` | One-layer release, restore, and warmup-then-release tests. |
| `09_uniformity_vs_irregularity/` | Contiguous vs scattered release-set geometry. |
| `10_budget_aware_rules/` | Budget-aware rule family. |
| `11_dynamic_release/` | Dynamic release probes and negative/positive results. |
| `12_paper_block4_correction/` | Reference-aligned Muon/Newton-Muon reset across datasets and depths. |
| `13_reference_lr_sanity/` | Short reference-LR, stability, and ordering sanity grid. |
| `14_official_newton_muon_r0/` | Pinned two-run official Newton-Muon-1 H100 reproduction gate. |
| `15_official_newton_muon_r1/` | Controlled official-architecture Muon/block4/none/diag comparison. |
| `22_r1_block_alpha/` | R1 block-local alpha pilot and preregistered multi-seed confirmation. |
| `25_owt_depth_kmode/` | Paired OWT depth-rule comparison of c_proj `none` versus `diag`. |
| `26_mech00_inventory/` | Read-only MECH-00 artifact, checkpoint, runtime, and source inventory. |
| `27_mech01_unified_k_diagnostics/` | Read-only checkpoint/schema and GPU numerical validation gate for unified K diagnostics. |
| `28_wikitext_depth_kmode/` | Three-seed WikiText-103 replication of the paired c_proj depth × `none/diag` experiment. |
| `29_r1_depth_kmode/` | Three-seed official-R1 depth transfer with selected c_proj `none/diag` and block4 fallback. |
| `_shared/` | Shared helpers. |

Prefer script names like:

```text
run_scale_up_300m.py
run_long_budget_100m.py
run_fixed_memory_batch_sweep.py
analyze_kstate_spectrum.py
```

Each runner should write generated commands or local artifacts to the matching
folder under:

```text
runs/
```

Set `SNM_RESULTS_ROOT` when results live outside this repository.

## Current Runners

```text
01_scale_up/run_scale_up_300m.py
01_scale_up/run_24l_module_bridge.py
02_long_token_budget/run_long_budget_100m.py
03_fixed_memory/run_fixed_memory_batch_sweep.py
04_dataset_generalization/run_wikitext103_100m.py
04_dataset_generalization/run_wikitext103_24l_12k_lr_sweep.py
06_kstate_spectrum/run_cproj_k_structure.py
06_kstate_spectrum/run_quadratic_score_probe.py
06_kstate_spectrum/run_targeted_diag_generalization.py
12_paper_block4_correction/run_paper_block4_correction.py
13_reference_lr_sanity/run_reference_lr_sanity.py
14_official_newton_muon_r0/run_official_newton_muon_r0.py
15_official_newton_muon_r1/run_official_newton_muon_r1.py
22_r1_block_alpha/run_r1_block_alpha.py
22_r1_block_alpha/run_confirmatory_batch.py
25_owt_depth_kmode/run_owt_depth_kmode.py
26_mech00_inventory/run_mech00_inventory.py
27_mech01_unified_k_diagnostics/run_mech01.py
28_wikitext_depth_kmode/run_wikitext_depth_kmode.py
29_r1_depth_kmode/run_r1_depth_kmode.py
29_r1_depth_kmode/run_three_seed_batch.py
```

All runners support `--dry-run`, `--wandb-mode disabled`, and
`--python-exe <path-to-training-python>`. Use `--python-exe` when the runner is
started from a lightweight Python but training should run inside the PyTorch
environment.

For a cheap smoke pass, run each script with `--dry-run` first. Then remove
`--dry-run` only on a machine whose Python environment has the training
dependencies installed.

`26_mech00_inventory` is the exception to the training-runner interface: it is
a standard-library-only, read-only audit and therefore has no `--dry-run`,
W&B, or training-Python arguments. Its safe active-training mode is
`--hash-mode none`; checkpoint SHA-256 can be completed later with
`--hash-mode full`.

`27_mech01_unified_k_diagnostics` is also an audit rather than a training
runner. Its standard-library controller delegates read-only checkpoint and
numerical work to the pinned `--python-exe`; it never uploads W&B data or
calls an optimizer step.
