# Research Methodology: Rising Loss vs. Improving Performance

## Is It Acceptable for Research?

**Short Answer**: Yes, **if and only if** you properly document, explain, and justify it.

## When Rising Loss with Improving Performance is Acceptable

### ✅ Acceptable Scenarios:

1. **Self-Play Non-Stationarity** (Your Case)
   - Opponent is constantly improving (it's yourself)
   - Data distribution shifts over time
   - Old experiences become less relevant
   - **This is well-documented in RL literature**

2. **Exploration vs. Exploitation Trade-off**
   - Agent explores new strategies
   - Initial performance may dip, then improve
   - Loss may rise temporarily during exploration

3. **Curriculum Learning**
   - Agent moves from easy to hard scenarios
   - Loss naturally increases on harder problems
   - But overall capability improves

### ❌ NOT Acceptable Scenarios:

1. **True Divergence**
   - Loss → ∞ (unbounded growth)
   - Q-values exploding
   - Network weights becoming NaN
   - **This indicates a bug, not a feature**

2. **Performance Not Improving**
   - Loss rising AND win rates not improving
   - Agent getting worse over time
   - **This indicates failed training**

3. **Unstable Training**
   - Loss oscillating wildly
   - No clear learning signal
   - **This indicates hyperparameter issues**

## Research Best Practices

### 1. **Report Both Metrics**

Always report:
- **Loss curves** (with explanation)
- **Win rates** (primary performance metric)
- **Other performance metrics** (score, game length, etc.)

Example in paper:
> "While training loss increased from 0.5 to 2.1 over 5000 episodes, win rates improved from 45% to 68%. This discrepancy is expected in self-play due to non-stationary data distribution..."

### 2. **Explain the Phenomenon**

In your methodology section, explain:

```markdown
## Training Dynamics

In self-play reinforcement learning, the training environment is inherently 
non-stationary. As the agent improves, it faces increasingly skilled opponents 
(previous versions of itself), leading to:

1. **Distribution Shift**: State-action pairs from early training become 
   less representative of current gameplay
2. **Replay Memory Contamination**: Older experiences, while still valid, 
   represent a different skill level
3. **Rising Loss with Improving Performance**: The network struggles to 
   simultaneously fit old (easier) and new (harder) experiences, causing 
   loss to rise even as win rates improve

This phenomenon is well-documented in self-play RL [cite AlphaGo, AlphaZero, etc.]
and is considered normal when performance metrics are improving.
```

### 3. **Provide Evidence**

Show:
- **Loss vs. Win Rate Plot**: Demonstrate inverse relationship
- **Recent vs. Old Experience Loss**: Show loss is higher on old experiences
- **Validation Against Fixed Opponents**: Show improvement against baseline agents
- **Smoothed Metrics**: Use rolling averages to show trends

### 4. **Compare with Baselines**

- Compare your agent against:
  - Random baseline
  - Heuristic baselines
  - Previous training checkpoints
  - Fixed opponents (not self-play)

This shows the agent is genuinely improving, not just playing against a harder opponent.

### 5. **Document Thresholds**

Define what you consider "acceptable":
- Loss rising but bounded (e.g., < 10.0)
- Win rates improving consistently
- No NaN or infinite values
- Stable Q-value ranges

### 6. **Address in Limitations**

If loss rising is a concern, discuss it in limitations:

```markdown
## Limitations

One limitation of our approach is the rising training loss despite improving 
performance. While this is expected in self-play scenarios, it makes it 
difficult to use loss as a stopping criterion. Future work could explore:

- Prioritized Experience Replay to focus on recent experiences
- Periodic memory clearing to reduce distribution shift
- Alternative loss functions that account for non-stationarity
```

## What to Report in Your Thesis/Paper

### Essential Metrics:

1. **Primary**: Win rates over training (with confidence intervals)
2. **Secondary**: Loss curves (with explanation)
3. **Validation**: Performance against fixed opponents
4. **Ablation**: Comparison of different hyperparameters

### Example Figure Caption:

> "Figure X: Training dynamics showing (a) win rate improving from 45% to 68% 
> over 5000 episodes, and (b) corresponding loss increase from 0.5 to 2.1. 
> The rising loss is expected in self-play due to non-stationary data 
> distribution, as the agent faces increasingly skilled opponents. The 
> improvement in win rates confirms the agent is learning despite the 
> increasing loss metric."

## Red Flags to Watch For

Stop training and investigate if:

1. **Loss → ∞** (unbounded growth)
2. **Win rates decreasing** (agent getting worse)
3. **NaN or Inf values** (numerical instability)
4. **Q-values exploding** (beyond reasonable bounds)
5. **No learning signal** (random performance)

## Recommendations for Your Research

### ✅ DO:

1. **Monitor win rates as primary metric**
2. **Document the loss/performance relationship**
3. **Cite relevant literature** (AlphaGo, AlphaZero, etc.)
4. **Show validation against fixed opponents**
5. **Use smoothed/rolling averages** for clearer trends
6. **Set reasonable loss thresholds** (e.g., stop if loss > 50.0)

### ❌ DON'T:

1. **Ignore rising loss without explanation**
2. **Use loss as sole stopping criterion**
3. **Claim success without performance metrics**
4. **Ignore if loss is truly diverging** (→ ∞)

## Conclusion

**Yes, it's acceptable to accept rising loss if win rates improve**, but you must:

1. ✅ Document and explain why
2. ✅ Report both metrics
3. ✅ Provide evidence of improvement
4. ✅ Compare with baselines
5. ✅ Address in limitations
6. ✅ Monitor for true divergence

This is standard practice in self-play RL research and is well-supported by literature.

## References to Cite

- AlphaGo (Nature 2016) - Self-play training
- AlphaZero (Science 2018) - Self-play with rising loss
- "Non-stationary Bandits" literature
- "Distribution Shift in RL" papers

