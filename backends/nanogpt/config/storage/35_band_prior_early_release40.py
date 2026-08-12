optimizer_type = "selective_newton_muon"
out_dir = "out_35_band_prior_early_release40"
wandb_run_name = "band_prior_early_release40"
wandb_tags = "formal,openwebtext_gpt2,selective_newton_muon,band_prior,storage_pareto"

selective_selection_mode = "shape_prior"
selective_shape_prior_policy = "mlp_c_proj_early_band"
selective_shape_prior_tiebreak = "early_layers_first"

selective_release_k_fraction = 0.40
selective_fraction = 1.0
selective_min_active = 1
selective_warmup_steps = 0
selective_score_interval = 25
selective_score_beta = 0.9
selective_score_mode = "gain_logcond_cost_power"
selective_cond_power = 1.0
selective_cost_power = 0.25
selective_freeze_after_warmup = True
selective_log_diagnostics = False
selective_release_inactive_k_state = True
diagnostic_interval = 0
