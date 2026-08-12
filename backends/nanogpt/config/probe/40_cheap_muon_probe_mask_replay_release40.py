optimizer_type = "selective_newton_muon"
out_dir = "out_40_cheap_muon_probe_mask_replay_release40"
wandb_run_name = "cheap_muon_probe_mask_replay_release40"
wandb_tags = "formal,openwebtext_gpt2,selective_newton_muon,cheap_muon_probe,mask_replay,storage_pareto"

selective_selection_mode = "oracle_static"
selective_static_mask_path = ""
selective_static_mask_seed = -1
selective_static_mask_run_name = ""
selective_static_mask_target_release_k_fraction = 0.40

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
