optimizer_type = "selective_newton_muon"
out_dir = "out_38_static_center_mask_replay"
wandb_run_name = "static_center_mask_replay"
wandb_tags = "formal,openwebtext_gpt2,selective_newton_muon,static_center_sweep,mask_replay,storage_pareto"

selective_selection_mode = "oracle_static"
selective_static_mask_path = ""
selective_static_mask_seed = 2024
selective_static_mask_run_name = ""
selective_static_mask_target_release_k_fraction = -1.0

selective_release_k_fraction = 0.0
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
