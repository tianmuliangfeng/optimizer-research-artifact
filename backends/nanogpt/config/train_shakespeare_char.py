# Shared tiny Shakespeare character-level experiment settings.

out_dir = "out_selective_newton_muon"
eval_interval = 50
eval_iters = 20
log_interval = 5
always_save_checkpoint = False

wandb_log = True
wandb_project = "Selective-Newton-Muon"
wandb_run_name = "tiny_shakespeare_selective_default"
wandb_mode = "online"
wandb_group = "tiny_shakespeare_selective"
wandb_tags = "formal,tiny_shakespeare_real,selective_newton_muon"

dataset = "shakespeare_char"
require_real_tiny_shakespeare = True
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

optimizer_type = "selective_newton_muon"
learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 20

muon_learning_rate = 0.02
muon_momentum = 0.95
muon_ns_steps = 5

input_beta = 0.95
input_ridge = 0.2
input_refresh = 16
input_max_samples = 2048

selective_fraction = 0.75
selective_min_active = 1
selective_warmup_steps = 100
selective_score_interval = 25
selective_score_beta = 0.9
selective_score_threshold = 0.0
selective_score_mode = "gain_over_cost"
selective_cond_power = 1.0
selective_cost_power = 0.25
selective_freeze_after_warmup = True
selective_log_diagnostics = False
selective_release_inactive_k_state = False
selective_selection_mode = "fraction"
selective_release_k_fraction = 0.0

sigma_block_size = 64
sigma_beta = 0.95
sigma_lambda_max = 0.25
sigma_lambda_start = 50
sigma_lambda_warmup = 200
sigma_eps = 1e-4
sigma_refresh = 8
sigma_stat_source = "base"
match_base_fro_norm = True
diagnostic_interval = 25
