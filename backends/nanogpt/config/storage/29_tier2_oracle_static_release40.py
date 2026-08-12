optimizer_type = "selective_newton_muon"
out_dir = "out_29_tier2_oracle_static_release40"
wandb_run_name = "tier2_oracle_static_release40"
wandb_tags = "formal,openwebtext_gpt2,tier2,selective_newton_muon,oracle_static,storage_pareto"

selective_selection_mode = "oracle_static"
selective_static_mask_path = os.path.join(
    os.environ.get("SNM_RESULTS_ROOT", "../../runs"),
    "analysis_exports/owt_tier2_dynamic_20260706/wandb_export_2026-07-06T14_17_38.134+08_00.csv",
)
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
import os
