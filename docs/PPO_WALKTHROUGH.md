# PPO Training Walkthrough (with implementation references)

This document walks through the full PPO pipeline in this codebase, with concrete file/line references and code snippets.

---

## 1. Entry point and config (train.py)

- **Config**: `hyperparams_config.json` is loaded; `algorithm` is read (e.g. `"ppo"`).
- **PPO branch** (around 159–189):

```python
# train.py (condensed)
if algorithm == "ppo":
    ppo_config = config.get("ppo", {})
    ppo_hidden = ppo_config.get("hidden_layers", [128, 128])
    use_position_indicator = ppo_config.get("use_position_indicator", False)
    model = EmbeddingActorCritic(
        env.observation_space,
        num_actions=actions,
        embedding_dim=52, zone_encoded_dim=52,
        hidden_layers=ppo_hidden,
        use_position_indicator=use_position_indicator,
    )
    trainee = Players.PPOAgent(
        "PlayerAgent", model,
        lr=ppo_config.get("learning_rate", 3e-4),
        gamma=ppo_config.get("gamma", 0.99),
        clip_eps=ppo_config.get("clip_eps", 0.2),
        ppo_epochs=ppo_config.get("ppo_epochs", 4),
        value_coef=ppo_config.get("value_coef", 0.5),
        entropy_coef=ppo_config.get("entropy_coef", 0.01),
    )
```

- **Batching**: `optimize_every_n_episodes = config.get("ppo", {}).get("optimize_every_n_episodes", 50)` is computed and passed into all `selfPlayTraining` calls so PPO optimizes every 50 episodes (and on the last episode of the round).

---

## 2. High-level training loop (train.py)

- **Rounds**: `for round_num in range(rounds)` (e.g. 10 rounds).
- **Who is trainee**: If training vs GapMaximizer, trainee alternates P1/P2 by round: `trainee_is_p1 = trainee_first_only or (round_num % 2 == 0)`.
- **Call into training** (e.g. trainee as P1):

```python
# train.py
_, _, steps_done = Training.selfPlayTraining(
    trainee, gapmaximizer_opponent, eps_per_round,  # e.g. 250 episodes
    model_id=f"round_{round_num}_vs_gapmaximizer",
    initial_steps=steps_done,
    round_number=round_num,
    initial_total_time=total_time,
    early_stopping_config=early_stopping_config,
    reward_mode=reward_mode,
    optimize_every_n_episodes=optimize_every_n_episodes,  # 50 for PPO
)
```

So each round runs `eps_per_round` episodes (e.g. 250) in `selfPlayTraining`.

---

## 3. Episode loop and turn execution (training.py)

- **Episode loop**: `selfPlayTraining` runs `for episode in range(episodes)`.
- Each episode has a **game loop** over turns. For each turn:
  - **P1’s turn**: `execute_player_turn(env, p1, p2, ...)` → then possibly `update_replay_memory(p1, p1_states, p1_actions, reward, ...)` for intermediate or terminal.
  - **P2’s turn**: same with p2.
- **Observation source**: At the start of a player’s turn, the env gives an observation from the **current** player’s perspective:

```python
# training.py execute_player_turn
observation = env.get_obs()
player_states.append(observation)
# ...
action = player.getAction(observation, valid_actions, actions, steps, force_greedy=validating)
player_actions.append(action)
```

- **Environment observation** (`environment.py`):

```python
# environment.py get_obs()
def get_obs(self):
    self.updateRevealed()
    is_p1 = self.current_zones is self.player_zones
    position = np.array([1.0, 0.0], dtype=np.float32) if is_p1 else np.array([0.0, 1.0], dtype=np.float32)
    obs = {
        "Current Zones": self.current_zones,
        "Off-Player Field": self.off_zones["Field"],
        "Off-Player Revealed": self.off_zones["Revealed"],
        "Deck": self.deck, "Scrap": self.scrap, "Stack": self.stack,
        "Effect-Shown": self.effect_shown,
        "position": position,
    }
    return obs
```

So each step the agent sees: zones (hand, field, revealed, deck, scrap, stack, effect-shown) plus a P1/P2 position one-hot (or neutral if `use_position_indicator` is false).

---

## 4. PPO agent: action and per-step storage (players.py)

- **Per-turn reset**: At the start of each turn, the PPO agent clears its per-turn lists so (state, action, log_prob, value) stay aligned:

```python
# training.py execute_player_turn
if hasattr(player, "current_turn_log_probs"):
    player.current_turn_log_probs.clear()
    player.current_turn_values.clear()
```

- **Getting an action**: The agent gets `observation`, runs the model, samples from the categorical policy, and stores log_prob and value for this step:

```python
# players.py PPOAgent.getAction
def getAction(self, observation, valid_actions, total_actions, steps_done, force_greedy=False):
    with torch.no_grad():
        logits, value = self.model(observation)
        # mask invalid actions with -inf
        mask = torch.full((1, total_actions), float("-inf"), ...)
        for a in valid_actions:
            mask[0, a] = 0
        logits = logits + mask
        dist = Categorical(logits=logits)
        action = dist.sample() if not force_greedy else logits.argmax(dim=1)
        action = action.item()
        log_prob = dist.log_prob(torch.tensor([action], ...)).squeeze(0)
        val = value.squeeze(0)
    self.current_turn_log_probs.append(log_prob.detach())
    self.current_turn_values.append(val.detach())
    return action
```

So for every (state, action) we have a stored `log_prob` and `value` from the **current** policy at the time of acting.

---

## 5. Storing trajectory (training.py → players.py)

- After each turn (or at game end), the training loop calls `update_replay_memory(player, states, actions, reward, next_state, score_change, gap_change)`.
- For PPO, that function **does not** use replay memory; it forwards to the agent’s trajectory buffer:

```python
# training.py update_replay_memory
if hasattr(player, "store_ppo_trajectory") and callable(getattr(player, "store_ppo_trajectory")):
    player.store_ppo_trajectory(states, actions, reward)
    return
```

- **Rewards** passed in from the game loop are:
  - **Terminal**: `REWARD_WIN` (1.0), `REWARD_LOSS` (-1.0), or `REWARD_DRAW` (0.0).
  - **Intermediate** (end of a non-terminal turn): `REWARD_INTERMEDIATE` (0.0). (Score/gap shaping is used for DQN; for PPO only this scalar is passed.)

- The agent appends one **chunk** per turn (or terminal block):

```python
# players.py PPOAgent.store_ppo_trajectory
def store_ppo_trajectory(self, states, actions, reward, log_probs=None, values=None):
    log_probs = log_probs if log_probs is not None else self.current_turn_log_probs
    values = values if values is not None else self.current_turn_values
    if len(states) != len(actions) or len(states) != len(log_probs) or len(states) != len(values):
        return
    self._trajectory.append((
        list(states), list(actions), list(log_probs), list(values), reward,
    ))
```

So `_trajectory` is a list of **chunks**, each chunk = (states, actions, log_probs, values, reward). One chunk = one turn (possibly several steps if there are counters/stack resolution).

---

## 6. When optimization runs (training.py)

- At **end of each episode**, the loop decides whether to call `optimize()`:

```python
# training.py (end of episode)
should_optimize = (
    optimize_every_n_episodes is None
    or (episode + 1) % optimize_every_n_episodes == 0
    or episode == episodes - 1
)
if should_optimize:
    if hasattr(p1, "optimize") and callable(getattr(p1, "optimize")):
        loss = p1.optimize()
    elif hasattr(p2, "optimize") and callable(getattr(p2, "optimize")):
        loss = p2.optimize()
```

- So with `optimize_every_n_episodes=50`, we call `optimize()` at episodes 50, 100, 150, 200, 250 (and on 249 if the round length isn’t a multiple of 50). Until then, **only the trainee** (p1 or p2) is collecting trajectory; the other player is e.g. GapMaximizer and has no `optimize()`. So we batch **50 episodes** of the trainee’s experience, then run one `optimize()`.

---

## 7. PPO optimize() (players.py): flatten, returns, advantages, loss

- **Guard**: If `_trajectory` is empty, return `None`.
- **Chunk info**: We build `chunk_info = [(start_idx, length, reward), ...]` while flattening, so we know which indices belong to which turn and what reward that turn got.

**Flatten and assign rewards for returns:**

```python
# players.py PPOAgent.optimize
all_states, all_actions, all_log_probs, all_values, all_rewards = [], [], [], [], []
for chunk_idx, (states, actions, log_probs, values, reward) in enumerate(self._trajectory):
    start = len(all_states)
    for i in range(len(states)):
        all_states.append(states[i])
        all_actions.append(actions[i])
        all_log_probs.append(log_probs[i])
        all_values.append(values[i])
        if i < len(states) - 1:
            all_rewards.append(0.0)
        else:
            all_rewards.append(reward)
    chunk_info.append((start, len(states), reward))
self._trajectory.clear()
self._clear_turn_aux()
```

So in the flattened list, only the **last** step of each chunk gets a non-zero reward in `all_rewards`; the rest are 0. That avoids double-counting when we overwrite returns by chunk.

**Monte Carlo returns, then overwrite by chunk:**

```python
# Monte Carlo
returns = []
R = 0.0
for r in reversed(all_rewards):
    R = r + self.gamma * R
    returns.append(R)
returns = list(reversed(returns))

# Overwrite: every step in a chunk gets that chunk’s reward as its return
for start, length, reward in chunk_info:
    for i in range(start, min(start + length, len(returns))):
        returns[i] = reward
```

So:
- **Terminal chunk**: all steps in that turn get the terminal reward (WIN/LOSS/DRAW) as their return.
- **Intermediate chunks**: all steps in that turn get the intermediate reward (e.g. 0.0) as their return.

**Advantages and normalization:**

```python
returns_t = torch.tensor(returns, device=device, dtype=torch.float32)
old_log_probs = torch.stack([lp.to(device) for lp in all_log_probs])
old_values = torch.stack([v.to(device) for v in all_values])
actions_t = torch.tensor(all_actions, device=device, dtype=torch.long)

advantages = returns_t - old_values.squeeze(-1)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**PPO epochs (same batch, multiple gradient steps):**

```python
for _ in range(self.ppo_epochs):
    logits, values = self.model(all_states)
    if values.dim() == 1:
        values = values.unsqueeze(1)
    dist = Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions_t)
    entropy = dist.entropy().mean()
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = torch.nn.functional.mse_loss(values.squeeze(-1), returns_t)
    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
    self.optimizer.step()
```

So we use **clipped surrogate** (min of ratio*adv and clipped ratio*adv), **value MSE** vs returns, and **entropy bonus**; then one gradient step per epoch. The same flattened batch (all steps from the last 50 episodes) is reused for `ppo_epochs` (e.g. 4) passes.

---

## 8. Network: observation → logits and value (networks.py)

- **EmbeddingActorCritic** is used when `use_embeddings` is true (default for PPO).
- **Single observation** → vector:
  - Zones (Hand, Field, Revealed, Off-Player Field/Revealed, Deck, Scrap, Stack, Effect-Shown) are each 52-dim boolean; we take card indices, embed them, max-pool per zone, then pass through a small MLP to get `zone_encoded_dim` per zone. All zone encodings are concatenated.
  - Then we append a 2-dim position vector: either the env’s `obs["position"]` (P1/P2 one-hot) or a neutral `[0.5, 0.5]` when `use_position_indicator` is false.

```python
# networks.py EmbeddingActorCritic._preprocess_single
fusion = torch.cat(zone_encodings, dim=0)
if self.use_position_indicator:
    position = obs.get("position", np.array([1.0, 0.0], dtype=np.float32))
else:
    position = np.array([0.5, 0.5], dtype=np.float32)
position_t = torch.from_numpy(np.asarray(position, dtype=np.float32)).to(device=fusion.device)
return torch.cat([fusion, position_t], dim=0)
```

- **Forward**: The concatenated vector goes through a shared backbone (e.g. [128, 128]), then a policy head (logits) and a value head (scalar):

```python
# networks.py EmbeddingActorCritic.forward
x = self._preprocess_observation(observation)
if x.dim() == 1:
    x = x.unsqueeze(0)
features = self.backbone(x)
logits = self.policy_head(features)
value = self.value_head(features)
# squeeze back to 1D if single obs
return logits, value
```

---

## 9. End-to-end flow summary

| Stage | Where | What happens |
|-------|--------|--------------|
| Config & model | `train.py` | `algorithm=="ppo"` → `EmbeddingActorCritic` + `PPOAgent`; `optimize_every_n_episodes=50`. |
| Rounds | `train.py` | For each round: trainee is P1 or P2; call `selfPlayTraining(trainee, opponent, eps_per_round, ..., optimize_every_n_episodes=50)`. |
| Episodes | `training.py` | For each episode: game loop over turns; each turn calls `execute_player_turn` for P1 then P2. |
| Observation | `environment.py` | `get_obs()` returns dict with zones + `position` (P1/P2 one-hot). |
| Action | `players.py` | `PPOAgent.getAction` → `model(obs)` → sample action, store `log_prob` and `value` in per-turn lists. |
| Store chunk | `training.py` → `players.py` | After each turn or at terminal: `update_replay_memory` → `store_ppo_trajectory(states, actions, reward)`; one chunk per turn. |
| Optimize? | `training.py` | Every 50 episodes and on last episode: call `trainee.optimize()`. |
| Optimize | `players.py` | Flatten chunks → returns (MC then overwrite by chunk) → advantages (normalized) → 4 PPO epochs (clip + value + entropy), gradient steps. |
| Network | `networks.py` | Obs → zones (embed + aggregate) + position → backbone → policy head (logits) + value head. |

That is the full PPO process in this implementation, from config and env observation through trajectory collection, batching every 50 episodes, and the PPO update with chunk-based returns and advantage normalization.
