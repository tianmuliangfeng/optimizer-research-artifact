out_dir = "out_smoke_cpu"
eval_interval = 1
eval_iters = 1
log_interval = 1
always_save_checkpoint = False
wandb_log = False
wandb_mode = "disabled"

dataset = "shakespeare_char"
require_real_tiny_shakespeare = False
gradient_accumulation_steps = 1
batch_size = 4
block_size = 32

n_layer = 2
n_head = 4
n_embd = 32
dropout = 0.0

optimizer_type = "selective_newton_muon"
learning_rate = 1e-3
max_iters = 2
lr_decay_iters = 2
warmup_iters = 1
min_lr = 1e-4

muon_learning_rate = 0.01
input_refresh = 1
input_max_samples = 256
selective_fraction = 0.5
selective_warmup_steps = 1
selective_score_interval = 1
selective_score_mode = "gain_logcond_cost_power"
selective_cost_power = 0.25
selective_freeze_after_warmup = True
sigma_block_size = 16
sigma_lambda_start = 0
sigma_lambda_warmup = 1
sigma_refresh = 1
diagnostic_interval = 1

device = "cpu"
dtype = "float32"
compile = False
