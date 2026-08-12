"""Muon aligned with Keller Jordan's public implementation and blog guidance."""

optimizer_type = "muon"
out_dir = "out_15_muon_blog_reference"
wandb_run_name = "15_muon_blog_reference"

muon_momentum = 0.95
muon_ns_steps = 5
muon_nesterov = True
muon_momentum_ema = True
muon_split_qkv = True
muon_adjust_lr_for_shape = True
muon_ns_compute_dtype = "bfloat16"

matrix_weight_decay = 0.0
