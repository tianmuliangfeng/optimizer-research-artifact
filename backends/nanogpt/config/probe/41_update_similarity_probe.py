optimizer_type = "selective_newton_muon"
out_dir = "out_41_update_similarity_probe"
wandb_run_name = "update_similarity_probe"
wandb_tags = "formal,openwebtext_gpt2,tier3,selective_newton_muon,mechanism,update_similarity_probe"

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
selective_release_inactive_k_state = False
selective_selection_mode = "k_release_budget"
selective_release_k_fraction = 0.0

update_similarity_probe_enabled = True
update_similarity_probe_interval = 25
update_similarity_probe_start_step = 0
update_similarity_probe_stop_step = -1
