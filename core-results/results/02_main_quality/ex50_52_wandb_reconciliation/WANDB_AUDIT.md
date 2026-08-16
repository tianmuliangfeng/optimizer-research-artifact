# Experiments 50--52 W&B reconciliation

- Status: `passed_with_documented_ex52_stitched_wandb_history`
- Reconciliation rows: 111/111 passed
- Identity-free W&B validation-loss rows retained: 476

## Coverage

| Experiment | W&B coverage | Acceptance interpretation |
|---|---|---|
| 50 | 3 formal seeds; val/train loss, LR, state/memory and auxiliary timing fields | All provided points match the accepted R1 artifacts. |
| 51 | 275M four seeds and 455M three seeds; validation loss, tokens and auxiliary training time | All provided points match the accepted formal metrics. |
| 52 | 124M and 1B, three seeds each, grouped by display name; loss, LR, tokens and auxiliary performance/time | All fields match the exact W&B lineage. The reused 1B run id retains the medium screen through step 1000 before the formal continuation. |

## Claim boundary

W&B is a secondary display mirror. Experiments 50 and 51 have clean pointwise loss reconciliation. Experiment 52 also reconciles, but its 1B W&B run id was reused: the displayed history contains the medium screen through step 1000 and the formal continuation thereafter. The accepted Experiment 52 endpoint and paired-contrast conclusions therefore continue to come from the sealed local formal CSVs. Timing remains excluded from paper efficiency claims.
