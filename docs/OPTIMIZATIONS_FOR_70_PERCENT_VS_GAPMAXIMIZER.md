# Optimizations to Reach ~70% Win Rate vs GapMaximizer

**Context:** Validation win rate is the average over **both positions**: half of validation episodes are trainee as P1, half as trainee as P2. So overall win rate = (trainee wins as P1 + trainee wins as P2) / (2 × episodes_per_position). To hit 70% overall, the agent must perform well in both seats (e.g. 70% in each, or 80% in one and 60% in the other).

**Change made:** Validation now logs **per-position** win rates so you can see imbalance, e.g.  
`[as P1: 75.2%, as P2: 58.4%]`.

---

## 1. **Turn on validation and use per-position stats** (quick win)

- Your gapmaximizer experiment had `skip_validation: true`. For tuning toward 70%, **enable validation** so you get a stable, position-balanced metric each round.
- Use the new **as P1 / as P2** breakdown to see if the agent is weak in one seat (e.g. much worse as P2). If so, consider:
  - More training with trainee as the weak position (e.g. `trainee_first_only: false` and enough rounds so both positions get many episodes), or
  - Slightly more episodes per position for the weak seat (would require a small code change to optionally weight positions).

---

## 2. **Prioritized Experience Replay (PER)** (already implemented, toggle on)

- Set `"use_prioritized_replay": true` in your config.
- Your logs showed high variance in loss and rare wins vs GapMaximizer; PER upweights high-TD-error transitions (e.g. wins and near-wins), which can improve sample efficiency and stability.
- Compare runs with and without PER on the same setup to measure impact.

---

## 3. **Reward shaping: try `normalized_score_diff`**

- Config: `"reward_mode": "normalized_score_diff"`.
- Gives a dense reward in [-1, 1] from (my_score - opp_score) / 21 instead of only WIN/LOSS/DRAW. This can speed up learning against a strong opponent by rewarding “getting closer” and punishing “falling behind” within a game.
- If the agent becomes too risk-averse or Q-values drift, you can mix (e.g. terminal reward binary + smaller normalized_score_diff) or revert to binary.

---

## 4. **Double DQN** (not yet implemented)

- You **already use a target network** (policy + target with soft or hard updates). That’s standard DQN.
- **Double DQN** keeps both networks but changes how the bootstrap target is computed: right now the **target** network is used for both (1) choosing the best next action and (2) evaluating its Q-value, which tends to overestimate. Double DQN uses the **policy** network to choose the best next action and the **target** network only to evaluate that action’s Q-value, reducing overestimation bias.
- Change in `optimize()`: instead of `self.target(non_final_next_states).max(1).values`, use:
  - `next_actions = self.policy(non_final_next_states).argmax(1)`
  - `next_state_values[non_final_mask] = self.target(non_final_next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)`
- Often improves stability and final performance with minimal extra cost.

---

## 5. **Curriculum: mix self-play and vs GapMaximizer**

- Config has `curriculum_vs_gapmaximizer_ratio` (currently 0). If you add support for it:
  - e.g. 20–30% of episodes vs GapMaximizer and the rest self-play (or vs a weaker opponent) early on, then increase the vs-GapMax fraction over rounds.
- Reduces early training on a very hard distribution and can improve stability and final win rate.

---

## 6. **Capacity and training length**

- **Network:** `game_based_scale: 1` → try **2** (or explicit `game_based_hidden_layers`) for more capacity if you see a plateau.
- **Episodes / rounds:** 250 × 10 = 2500 episodes; try more rounds (e.g. 15–20) or more episodes per round (e.g. 350–500) to see if win rate keeps improving.
- **Epsilon:** With only vs-GapMaximizer, exploration matters. Current decay over 30k steps is reasonable; if you increase total steps, consider slightly slower decay (e.g. 40k–50k) so the agent keeps exploring in later rounds.

---

## 7. **Learning rate and target network**

- You already use LR decay and soft/hard target updates. If loss is stable but win rate plateaus:
  - Slightly **higher** initial LR (e.g. 7e-5 or 1e-4) with the same decay can sometimes push performance; watch for instability.
  - **Target update:** Hard updates every 500 steps are reasonable; if you switch to Double DQN, you can try slightly less frequent updates (e.g. 750) for more stability.

---

## 8. **Validation episodes**

- `validation_episodes_ratio: 1` → 250 total (125 per position). For a more reliable 70% estimate, consider **1.5–2** (375–500 total) so variance is lower and per-position rates are more stable.

---

## Suggested order of trials

1. **Enable validation** and run with current setup; note **as P1 / as P2** win rates.
2. **Enable PER** and compare validation win rate (and per-position) vs same setup without PER.
3. **Switch to `normalized_score_diff`** (or a mix) and compare again.
4. **Add Double DQN** and re-run the same config.
5. If still below 70%, try **more capacity** (scale 2), **more rounds/episodes**, and optionally **curriculum** (if you implement the ratio).

Using the per-position validation output will tell you whether to focus on P1, P2, or both when tuning.

---

## 9. **Escaping local minima / plateaus**

If win rate improves for a few rounds then flattens or drops and stays stuck, the policy is likely in a local optimum (e.g. a narrow strategy that loses to GapMaximizer). Try these in order:

- **Exploration boost on regression** (implemented): Set `"exploration_boost_on_regression_steps": 5000` under `training`. When validation win rate **drops for 2 consecutive rounds**, epsilon is boosted for the **next** round only (then cleared). Requires 2 consecutive regressions to avoid boosting on every small dip (which can cause a death spiral). If you see major regression after enabling this, try **disabling it** (`0`) to see if the cause is elsewhere.
- **Slower epsilon decay**: Increase `eps_decay` (e.g. 40k–50k) so epsilon stays higher for more rounds. Reduces the chance of locking into a bad policy too early.
- **Higher minimum epsilon**: Set `eps_end` to 0.2–0.25 so the agent never fully stops exploring. Useful if the game has many distinct lines and the agent needs to keep trying alternatives.
- **Reward shaping**: Use `normalized_score_diff` so the agent gets a learning signal even when it loses (e.g. “losing by less” is better). Reduces reliance on rare wins and can smooth the landscape.
- **Curriculum**: Implement and use `curriculum_vs_gapmaximizer_ratio` (e.g. 0.2–0.3 early, then increase). Mix of self-play (or vs random) and vs GapMaximizer so the agent sees more wins early and learns basics before facing the hard opponent full time.
- **PER + Double DQN**: PER focuses on high-TD-error transitions; Double DQN reduces overestimation. Together they often improve stability and final performance, which can help escape bad basins.
- **Learning rate warm restarts**: Periodically bump LR back up (e.g. every 5 rounds) to allow larger steps and escape shallow minima. Experiment with a small multiplier (e.g. 1.5×) to avoid divergence.
- **Multiple seeds and checkpoint selection**: Run 3–5 seeds and keep the best checkpoint by validation win rate; some seeds will get stuck, others may find a better basin.
- **Larger capacity**: Try `game_based_scale: 2` so the policy has more representational capacity to learn a more complex winning strategy.

---

## 10. **Log analysis: extend training now or wait?**

Analysis of **experiment_20260217_190352_double_dqn_gapmaximizer** (scale_11, Double DQN, binary reward, 10 rounds, no validation, no PER):

| Round | Trainee position | End-of-round win rate (in-training) |
|-------|------------------|-------------------------------------|
| 0     | P1               | ~17%                                |
| 1     | P2               | ~28%                                |
| 2     | P1               | ~30%                                |
| 4     | P1               | ~30%                                |
| 6     | P1               | ~30%                                |
| 9     | P2               | **~48%**                            |

**Conclusion:** Current duration **does show promise**. As P2 the agent improved from ~28% to ~48%; as P1 it improved from ~17% to ~30% but then plateaued. We are not at 70%, but learning is happening and P2 in particular has room to improve further.

**Recommendation:** **Extend training with all improvements** rather than waiting. Use the full-improvements experiment (PER, Double DQN, `normalized_score_diff`, validation on, exploration boost on regression) and **15 rounds** with slower epsilon decay (40k) and higher minimum epsilon (0.2) so both positions get more data and exploration. If after 15 rounds with these settings win rate is still flat, then consider curriculum or larger capacity next.

---

## 11. **Major validation regression: causes and mitigations**

If validation win rate improves for several rounds then drops sharply (e.g. 49% → 32% over 2–4 rounds), check the following:

- **Exploration boost on regression:** If enabled, it now triggers only after **2 consecutive** regressions and applies for one round then clears. If you still see runaway regression, set `exploration_boost_on_regression_steps: 0` and re-run; if regression stops, the boost was the cause. If regression continues, the cause is elsewhere.
- **PER (Prioritized Replay):** High-TD-error transitions are replayed more often. As the policy changes, old “surprising” (often bad) transitions can dominate and push the policy backward (forgetting). Try **lower `per_alpha`** (e.g. 0.4) so sampling is closer to uniform, or ensure **`per_beta`** anneals to 1.0 so importance-sampling weights correct for the bias. In the worst case, disable PER (`use_prioritized_replay: false`) to test.
- **Reward shaping:** With `normalized_score_diff`, the agent gets a dense reward. If it starts optimizing “lose by less” instead of “win”, validation win rate can drop. Try **binary** reward for a run to compare.
- **Single policy, two positions:** One network plays both P1 and P2. Alternating rounds can pull the policy in two directions; later rounds might overwrite what worked for one position. Consider more rounds, or (if you add it) curriculum that balances both positions.
- **Checkpoint selection:** Use **best validation checkpoint** (`model_best.pt`) for evaluation and deployment, not the final round. The training script already saves the best round by validation; avoid relying on the last round if you see late regression.

---

## 12. **Why “going first” (P1) can poison training**

One policy is used for both positions. Observations are always from the *current* player’s view (“Current Zones” = my hand/field) and **do not** explicitly say “I am P1” or “I am P2”. So the network has to infer position from state. That makes it easy for P1 updates to overwrite or conflict with what worked for P2.

**What’s going wrong:**

1. **Whole rounds of P1-only updates**  
   On even rounds you add 250 episodes of P1 experience and do 250 `optimize()` steps in a row. Each step samples from the replay buffer; by the end of the round the buffer is heavily P1. So you get a long run of gradients that are mostly from P1. If P1 is harder or noisier, those updates can move the shared policy in a bad direction and **undo** good P2 behavior (catastrophic forgetting).

2. **PER amplifies P1**  
   If P1 transitions have higher TD error (e.g. you lose more as P1, or Q is wrong there), PER replays them more often. So even on P2 rounds, batches can be dominated by P1 data. The policy keeps being pulled toward fitting P1 and can stay unstable on P1 while hurting P2.

3. **Same input shape, different roles**  
   P1 (first mover) and P2 (reactor) need different strategies. With no position in the observation, the network has to use the same representation for both. That can make learning unstable and make P1 gradients “poison” the shared representation.

**Mitigations:**

- **Add a position indicator to the observation** (recommended): e.g. a 2-dim one-hot “I am P1” / “I am P2” so the network can condition on position. Implemented below as an optional env + network change; use it for new runs and compare validation (especially as P1) and stability.
- **Train P2 more than P1**: e.g. 2 rounds as P2, 1 as P1, or use `trainee_first_only: false` and increase total rounds so P2 still gets more data. Reduces how much one block of P1 updates can overwrite P2.
- **Soften P1 rounds**: lower learning rate or fewer episodes on P1-only rounds (would need a small code change to pass “position” or round type into the training step).
- **Larger replay buffer**: so that after a P1 round, the buffer stays more mixed and batches are less dominated by P1 (helps a bit; PER can still over-sample P1 if TD error is higher there).

---

## 13. **What to do when you've flatlined (e.g. ~45%)**

If validation peaks (e.g. round 6 at ~45.6%) then flatlines or regresses for the rest of training, that **indicates a local minimum or plateau**: the policy found a basin that beats ~45% but further updates don’t improve (or overwrite) it.

**Do this first:**

1. **Use the best checkpoint.** Evaluate and report results with `model_best.pt` (or `model_round_6.pt` in the flatline example), not the final round. The training script already saves the best round by validation.
2. **Don’t assume more of the same training will help.** More rounds with the same setup often just keep you in the same basin.

**Then try, in order:**

- **Larger capacity:** `game_based_scale: 2` (or explicit larger hidden layers). More parameters can represent a better policy; train from scratch or fine-tune from the best checkpoint.
- **Curriculum:** Implement and use `curriculum_vs_gapmaximizer_ratio` (e.g. 20–30% vs GapMaximizer early, rest self-play or vs random), then increase. Gets more wins early and can lead to a better basin.
- **Multiple seeds:** Run 3–5 seeds; keep the best by validation. Some seeds plateau lower, others may reach a higher basin.
- **Change the signal:** Try **binary** reward instead of `normalized_score_diff` (or vice versa), or **disable PER** for one run. Different gradient distribution can land in a different minimum.
- **LR warm restarts:** Every 5 rounds, multiply learning rate by 1.2–1.5 (then decay again) to allow occasional larger steps and escape shallow minima.
- **Exploration:** Slightly higher `eps_end` (e.g. 0.25) or slower `eps_decay` so the agent keeps exploring a bit in later rounds.

If you implement curriculum, that’s the highest-lever next step; otherwise start with larger capacity and multiple seeds.
