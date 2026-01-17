# Fast Hyperparameter Experimentation Guide

This guide explains how to quickly test different hyperparameter configurations without editing code.

## Quick Start

1. **Edit `hyperparams_config.json`** - Change any hyperparameter values
2. **Enable Quick Test Mode** - Set `"quick_test_mode": true` for fast iteration (3 rounds, 100 episodes each)
3. **Run training** - The script automatically loads from the config file
4. **Early stopping** - Training stops automatically if loss diverges, saving time

## Configuration File

The `hyperparams_config.json` file contains all hyperparameters:

### Key Hyperparameters for Loss Issues

- **`target_update_frequency`**: How often to update target network (lower = more stable, but slower)
  - Recommended: 500-1000 for stability
  - Too high (2000+) causes rising loss due to stale targets
  
- **`learning_rate`**: Learning rate for optimizer
  - Recommended: 3e-5 to 8e-5
  - Too high causes divergence, too low causes slow learning

- **`gradient_clip_norm`**: Maximum gradient norm (prevents exploding gradients)
  - Recommended: 3.0-10.0
  - Lower = more stable but slower learning

- **`q_value_clip`**: Maximum Q-value (prevents Q-value explosion)
  - Recommended: 10.0-20.0
  - Lower = more stable but may limit learning

- **`replay_buffer_size`**: Size of experience replay memory
  - Recommended: 25,000-100,000
  - Smaller = faster aging of old experiences (helps with self-play non-stationarity)
  - Larger = more diverse experiences but may include outdated strategies
  - For self-play: Try 25,000-50,000 to reduce impact of old experiences

### Quick Test Mode

For fast hyperparameter testing:

```json
"training": {
  "quick_test_mode": true,
  "quick_test_rounds": 3,
  "quick_test_eps_per_round": 100
}
```

This runs only 300 episodes total (vs 5000 normally), allowing you to quickly test if hyperparameters are working.

### Early Stopping

Automatically stops training if loss is diverging:

```json
"early_stopping": {
  "enabled": true,
  "check_interval": 50,        # Check every N episodes
  "window_size": 100,           # Look at last N losses
  "divergence_threshold": 0.5,  # Minimum slope to consider divergence
  "min_episodes": 200,          # Don't stop before this many episodes
  "max_loss": 50.0              # Stop if loss exceeds this
}
```

## Workflow for Experimentation

1. **Start with Quick Test Mode**:
   ```json
   "quick_test_mode": true
   ```
   Run training and check if loss is stable after ~300 episodes.

2. **If loss is stable**, disable quick test mode and run full training:
   ```json
   "quick_test_mode": false
   ```

3. **If loss is rising**, try:
   - Reduce `learning_rate` by 50%
   - Reduce `target_update_frequency` to 500
   - Reduce `gradient_clip_norm` to 3.0
   - Reduce `q_value_clip` to 10.0

4. **If loss is too high**, try:
   - Increase `target_update_frequency` to 1000
   - Increase `learning_rate` slightly
   - Check if `max_loss` threshold is too low

## Example: Testing Different Learning Rates

1. Edit `hyperparams_config.json`:
   ```json
   "learning_rate": 3e-5,
   "quick_test_mode": true
   ```

2. Run training:
   ```bash
   python train_no_features.py
   ```

3. Check loss trend in output or graphs

4. If stable, try higher LR:
   ```json
   "learning_rate": 5e-5
   ```

5. Repeat until you find optimal value

## Tips

- **Early stopping saves time**: Bad hyperparameters are detected within 200-300 episodes
- **Quick test mode is 16x faster**: 300 episodes vs 5000 for initial testing
- **Monitor loss trends**: Rising loss = bad hyperparameters, stable/decreasing = good
- **Start conservative**: Lower learning rates and more frequent target updates are safer

