"""Newton-Muon with the paper's MLP-contraction four-block K structure.

The QKV, attention-output, and MLP-expansion matrices retain one full input
second-moment matrix.  Only ``mlp.c_proj`` is split into four contiguous
``d x d`` blocks, matching the public Newton-Muon implementation.
"""

optimizer_type = "cproj_k_mode_newton_muon"
out_dir = "out_14_newton_muon_paper_block4"
wandb_run_name = "14_newton_muon_paper_block4"

cproj_k_mode = "block4"
cproj_k_blocks = 4
cproj_k_reference_mode = "block4"

input_beta = 0.95
input_ridge = 0.2
input_refresh = 32
input_max_samples = 0
input_first_refresh_step = 31
input_init_scale = 0.001
input_init_inverse_scale = 1.0

# Use the same public-Muon post-preconditioner pipeline as the reference Muon
# baseline: EMA-form momentum, Nesterov lookahead, separate Q/K/V
# orthogonalization, aspect-ratio scaling, and BF16 Newton-Schulz.
muon_nesterov = True
muon_momentum_ema = True
muon_split_qkv = True
muon_adjust_lr_for_shape = True
muon_ns_compute_dtype = "bfloat16"

diagnostic_interval = 0
