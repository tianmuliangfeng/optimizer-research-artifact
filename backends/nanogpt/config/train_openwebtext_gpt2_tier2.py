# OpenWebText GPT-2-tokenized subset, model tier 2.

out_dir = "out_openwebtext_gpt2_tier2"
eval_interval = 200
eval_iters = 20
log_interval = 10
always_save_checkpoint = False

wandb_log = True
wandb_project = "Selective-Newton-Muon-OWT"
wandb_run_name = "owt_tier2_newton_muon"
wandb_mode = "online"
wandb_group = "owt_tier2"
wandb_tags = "formal,openwebtext_gpt2,tier2,selective_newton_muon,storage_pareto"

dataset = "openwebtext_gpt2"
require_real_tiny_shakespeare = False
gradient_accumulation_steps = 1
batch_size = 16
block_size = 512

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

optimizer_type = "newton_muon"
learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
warmup_iters = 20

muon_learning_rate = 0.02
muon_momentum = 0.95
muon_ns_steps = 5
matrix_weight_decay = 0.0
matrix_eps = 1e-8

input_beta = 0.95
input_ridge = 0.2
input_refresh = 32
input_max_samples = 2048

selective_fraction = 1.0
selective_min_active = 1
selective_warmup_steps = 100
selective_score_interval = 25
selective_score_beta = 0.9
selective_score_threshold = 0.0
selective_score_mode = "gain_logcond_cost_power"
selective_cond_power = 1.0
selective_cost_power = 0.25
selective_freeze_after_warmup = True
selective_log_diagnostics = False
selective_release_inactive_k_state = True
selective_selection_mode = "k_release_budget"
selective_release_k_fraction = 0.0

diagnostic_interval = 0
