# Final unified analysis: experiments 38–45

Date: 2026-08-03  
Status: **claim-eligible with explicit scope caveats**  
Practical final-loss margin: **±0.002**

## Executive decision

The completed evidence supports the paper's central allocation claim, but not a universal optimizer-route ranking. Selective K-state routing preserves most or all of the original Newton–Muon quality benefit while substantially reducing state. The preferred low-state route changes with environment: diagonal is the clear 124M choice, diag and none are practically interchangeable at 275M, and none has the best endpoint mean at 455M. Because model size, architecture details, and token/parameter ratios are not jointly controlled, this is evidence for **environment-dependent allocation**, not a monotonic scaling law.

Mousse is a useful 124M external neighbor. It beats Muon, Moonlight, NorMuon, and AdamW in final loss, but ranks fourth of eight: diag, block4, and none all have lower means. It is also dominated on the final-loss/optimizer-state plane. These results do not trigger a 275M Mousse extension.

The earlier provisional hypothesis that the original-over-Muon gain and the none-over-original gain are of similar size does **not** survive as a general cross-scale statement. At 455M both increments favor none/original in sequence but the second is smaller; at 275M none is essentially tied with original; at 124M none is materially worse than original.

## Cross-scale quality evidence

| Environment | Parameters | Tokens/parameter | Best endpoint mean | Best mean loss | none vs original | diag vs original |
|---|---|---|---|---|---|---|
| gpt124m | 124M | 26.21 | diag | 3.261100 | robust_material_degradation | inconclusive |
| gpt275m | 276M | 2.42 | none | 3.274724 | practical_equivalence_supported | inconclusive |
| gpt455m | 454M | 6.88 | none | 2.917476 | practical_equivalence_supported | inconclusive |

Key paired results (seed is the unit; scales are never pooled):

- 124M: diag−Muon = **−0.016033** [−0.018522, −0.013545]; none−Muon = **−0.010467** [−0.011717, −0.009216]. Diag improves materially over none by **−0.005567** [−0.006841, −0.004292].
- 275M: none−original = **−0.000074** [−0.001937, +0.001788], satisfying the preregistered practical-equivalence interval. Diag−none = **+0.000239** [−0.001060, +0.001538], also practically equivalent.
- 455M: none−Muon = **−0.002335** [−0.002764, −0.001905] and none−original = **−0.000821** [−0.001904, +0.000262]. The former is directional and slightly beyond the mean materiality threshold; its interval straddles −0.002, so it is not labelled a robust material improvement.

The endpoint preference changes from diag at 124M to an effectively interchangeable diag/none pair at 275M and none at 455M. Token/parameter ratios are 26.21, 2.42, and 6.88 respectively, so scale and training budget are confounded.

## 124M external-neighbor panel

| Rank | Method | Final loss mean ± SD | Δ vs Mousse | Optimizer state MiB | Peak MiB | Pareto |
|---|---|---|---|---|---|---|
| 1 | Newton-Muon diag | 3.261100 ± 0.001114 | -0.006933 | 780.8 | 38304 | yes |
| 2 | Newton-Muon block4 | 3.262200 ± 0.001136 | -0.005833 | 996.5 | 39168 | no |
| 3 | Newton-Muon none | 3.266667 ± 0.000666 | -0.001367 | 780.5 | 38304 | yes |
| 4 | Mousse-R1 | 3.268033 ± 0.000252 | +0.000000 | 2887.1 | 39985 | no |
| 5 | Moonlight Muon | 3.274967 ± 0.000451 | +0.006933 | 618.5 | 37703 | yes |
| 6 | Muon | 3.277133 ± 0.000252 | +0.009100 | 618.5 | 37703 | no |
| 7 | NorMuon | 3.334467 ± 0.000924 | +0.066433 | 618.8 | 37703 | no |
| 8 | AdamW | 3.400433 ± 0.004692 | +0.132400 | 942.5 | 38026 | no |

Mousse−Muon is **-0.009100** and Mousse−block4 is **+0.005833**. Diag−Mousse is **-0.006933**; none−Mousse is **-0.001367**. The none advantage is within the practical margin by mean but remains directional in all three seeds; it should not be described as established equivalence.

Relative to Mousse, diag saves **2106.4 MiB** of optimizer state (73.0%) and **1681 MiB** peak memory while lowering loss. None has nearly identical state cost. Block4 also lowers loss and uses **1890.6 MiB** less optimizer state.

## Mechanistic synthesis

The evidence chain is now coherent at three levels:

1. **Allocation:** Experiment 41 finds both full `c_fc` K and block4 `c_proj` K beneficial on R1, with approximately additive main effects: -0.003483 and -0.004583. This rules out the literal claim that removing `c_proj` K universally improves quality.
2. **Compression of useful information:** Experiment 41D shows that diagonal `c_proj` K improves over none by -0.005567 and matches block4 within the ±0.002 practical margin, while adding only 0.28125 MiB over none. The defensible interpretation is that per-coordinate scale information carries high value in this R1 slice, whereas dense/block cross-coordinate state is not needed to recover block4-level quality here.
3. **Dynamics and boundary:** Experiment 38 supports stage-dependent short-horizon behavior and a causal role for scheduled down-projection refresh in its frozen 1B intervention tree. Experiment 40 shows that contiguous LLaMA block4 has pooled median update drift 0.3447, 21.64× the equivariant control. Thus block4 is not an architecture-neutral definition of original Newton–Muon.

The analytic state calculation is consistent with this interpretation: at width 3072, block routing stores 25.00% of full matrix state, while diag stores 0.03255%. The local equivariance check passes its expected invariances and measures cross-block block-route drift 0.1519.

These diagnostics strengthen the method narrative but do not replace end-to-end training comparisons. Refresh mediation remains a short-horizon causal result, not a full-training quality claim.

## Efficiency and memory

- R1 isolated H100 evidence (Experiment 39): median tokens/s are Muon 445563, original block4 436682, none 437830, and diag 437468. None/diag save 216 MiB K state and 864 MiB measured peak memory versus block4, with a small throughput advantage over block4 but remaining slower than Muon.
- LLaMA-1B isolated H100 evidence (Experiment 42): none and diag are about 4.0% faster than `newton_full`, cut K state by about 70.65%, and reduce timed peak allocated memory by about 17.85%; both remain about 1.5% slower than Muon. These are technical repeats on one host, not cross-seed quality evidence.
- Timing from Experiments 41, 43, 44, and 45 is excluded because the protocols do not provide isolated, paper-eligible timing. Raw throughput must not be compared numerically between Experiments 39 and 42.

## Submission-level conclusions

1. **Supported central claim:** selective allocation can retain Newton–Muon quality benefits at much lower persistent state; the strongest route is environment dependent.
2. **Supported 124M deployed choice:** diag is the best-supported quality/state choice and outperforms Mousse in every paired seed.
3. **Supported larger-GPT choice:** none is the most economical route at 275M and 455M; it is practically equivalent to original at both scales and has the best endpoint mean at 455M.
4. **Supported external-baseline statement:** Mousse is stronger than Muon and the other external optimizer baselines tested at 124M, but it does not match the selective/original Newton–Muon group and costs substantially more optimizer state.
5. **Supported mechanistic statement:** useful K information is module- and architecture-dependent; diagonal scale information can recover the deployed R1 `c_proj` benefit at near-none incremental state.
6. **Not supported:** a universal diag or none ranking, a monotonic scale law, a general equality of the two sequential loss gains, architecture-neutral block4, or 43/44/45 timing comparisons.

## Gate decision

The final unified evidence analysis is complete and claim-eligible. No immediate new large training run is justified by Experiments 43–45. The next planned step is the separately scoped method-deepening package (refresh/stability and related local diagnostics), followed by freezing paper tables, figures, limitations, and wording. A 275M Mousse run remains unnecessary unless the paper's positioning changes and requires cross-scale external-baseline coverage as an explicit reviewer-facing claim.
