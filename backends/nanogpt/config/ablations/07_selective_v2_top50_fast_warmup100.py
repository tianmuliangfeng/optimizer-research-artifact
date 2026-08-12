optimizer_type = "selective_newton_muon"
out_dir = "out_07_selective_v2_top50_fast_warmup100"
wandb_run_name = "tiny_shakespeare_real_07_selective_v2_top50_fast_warmup100"

selective_fraction = 0.5
selective_warmup_steps = 100
selective_score_interval = 25
selective_score_beta = 0.9
selective_score_mode = "gain_logcond_cost_power"
selective_cond_power = 1.0
selective_cost_power = 0.25
selective_freeze_after_warmup = True
selective_log_diagnostics = False
diagnostic_interval = 0
