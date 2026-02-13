# Rounds Analysis: Are We Running Too Many Rounds?

## Summary

**Yes.** Previous experiments show that:

1. **Best performance often occurs mid-training (rounds 5–15), not at round 19.**  
   Several runs peak then decline by round 19, so running 20 rounds can yield *worse* reported performance than stopping earlier.

2. **Win rate vs GapMaximizer typically plateaus (or peaks) by round 5–10** for the architectures that learn. Extra rounds beyond that add little or hurt.

3. **Recommendation:** For screening experiments, **10–12 rounds** are likely enough. For final evaluation, consider **early stopping** or **best-checkpoint-by-validation** instead of always using the final round.

---

## Evidence

### Architecture comparison (20 rounds, vs GapMaximizer)

| Run | Round 5 | Round 10 | Round 15 | Round 19 | Peak round | Peak WR |
|-----|---------|----------|----------|----------|------------|---------|
| **large_hidden_embedding_run_01** | 57.6% | 52.8% | 59.2% | **52.8%** | **7** | **63.2%** |
| large_hidden_boolean_run_05 | 20.8% | 20.0% | 24.0% | 20.8% | ~5 | ~21% |
| **linear_embedding_run_01** | 41.6% | 60.0% | **60.8%** | 58.4% | **15** | **60.8%** |
| game_based_embedding_run_01 | 23.2% | 13.6% | 12.0% | 8.0% | early | collapsed |

- **large_hidden_embedding:** Peak at **round 7 (63.2%)**; by round 19 win rate is **52.8%**. So 12 extra rounds (8–19) do not improve and may worsen the final number.
- **linear_embedding:** Peak at round 15 (60.8%); round 19 is 58.4%. Slight drop after peak.
- **game_based_embedding:** Collapsed after round 5; final 8%.

### Game-based scaling screening (20 rounds)

| Run | Round 5 | Round 10 | Round 19 | Peak (approx) |
|-----|---------|----------|----------|----------------|
| game_based_embedding_scale_11 | 60.0% | **60.8%** | 49.6% | round 10 |
| game_based_embedding_scale_21 | **52.8%** | 30.4% | 29.6% | round 5 |
| game_based_embedding_scale_05 | 44.0% | 36.0% | 20.0% | early |
| game_based_embedding_scale_22 | 53.6% | 56.0% | 39.2% | round 10 |

- Scale 11: best at **round 10 (60.8%)**, then drops to 49.6% by round 19.
- Scale 21: best at **round 5 (52.8%)**, then drops to 29.6% by round 19.

So for these game-based scaling runs, **more rounds often mean lower final win rate**, not higher.

---

## Plateau definition (95% of final)

- **large_hidden_embedding:** Reaches 95% of final (52.8%) by **round 5** (57.6% ≥ 0.95×52.8%).
- **large_hidden_boolean:** Plateau by **round 5**.
- **linear_embedding:** Plateau by **round 10** (60% ≥ 0.95×58.4%).

So for runs that learn, **95% of final performance is typically reached by round 5–10**. Rounds 11–19 add little or reduce performance.

---

## Possible causes of decline after peak

1. **Overfitting** to self-play or training distribution.
2. **Epsilon decay:** By later rounds exploration is low; policy may have converged to a bad local optimum.
3. **Instability:** Learning rate or updates causing late training degradation.
4. **Validation opponent (GapMaximizer):** Not identical to training; agent may adapt in a way that hurts vs this opponent.

---

## Recommendations

1. **Screening experiments:** Use **10 rounds** (or at most 12) instead of 20. You will see peak or near-peak performance and save roughly half the time per run.
2. **Evaluation:** Prefer **best checkpoint by validation win rate** (e.g. best round by vs GapMaximizer) rather than always reporting round 19. That aligns reported numbers with “best we got” and avoids the “extra rounds hurt” effect.
3. **Early stopping:** Consider stopping when validation win rate has not improved for 3–5 rounds (or when it drops below a fraction of the best so far), instead of always running 20 rounds.
4. **Hyperparameters:** If you keep 20 rounds, consider reviewing learning-rate schedule and epsilon decay so that later rounds do not systematically degrade performance.

---

## Time impact

- Current: 20 rounds × 250 episodes/round = 5,000 training episodes per run, plus validation.
- With 10 rounds: 2,500 training episodes per run. **~50% less time per run** for screening, with evidence that you are not losing (and may gain) by avoiding the post-peak decline.

---

*Analysis based on: experiment_20260123_173115_architecture_comparison, experiment_20260126_112849_game_based_scaling_screening. Win rates vs GapMaximizer, last line of each `metrics_round_N_vs_gapmaximizer_*.jsonl`.*
