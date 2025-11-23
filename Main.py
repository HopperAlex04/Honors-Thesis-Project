import torch
from pathlib import Path

import Players
from GameEnvironment import CuttleEnvironment
from Networks import NeuralNetwork
import Training

user_ended = False

env = CuttleEnvironment()
actions = env.actions
model = NeuralNetwork(env.observation_space, 2, actions, None)

# Eventually make these adjustable as well
BATCH_SIZE = 128
GAMMA = 0.4
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 10000
TAU = 0.005
LR = 3e-4
trainee = Players.Agent(
    "PlayerAgent", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR
)

valid_agent = Players.Agent("ValidAgent", model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR)

# Ensure models directory exists
models_dir = Path("./models")
models_dir.mkdir(parents=True, exist_ok=True)
print(f"Models directory ready: {models_dir}")

# Save initial checkpoint
try:
    checkpoint_path = models_dir / f"checkpoint{0}.pt"
    torch.save(model, checkpoint_path)
    print(f"Saved initial checkpoint: {checkpoint_path}")
except Exception as e:
    print(f"Error saving initial checkpoint: {e}")
    raise

validation01 = Players.HeuristicHighCard("HighCard")

rounds = 10
eps_per_round = 100

for x in range(rounds):
    checkpoint_path = models_dir / f"checkpoint{x}.pt"
    
    # Check if checkpoint exists before loading
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint {checkpoint_path} not found. Skipping round {x}.")
        continue
    
    # Load previous checkpoint and update trainee's model
    try:
        prev_checkpoint = torch.load(checkpoint_path, weights_only=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
        
        # Update trainee's model with the loaded checkpoint
        trainee.model.load_state_dict(prev_checkpoint.state_dict())
        trainee.policy.load_state_dict(prev_checkpoint.state_dict())
        print(f"Updated trainee model with checkpoint {x}")
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        print(f"Skipping round {x}.")
        continue
    
    try:
        Training.selfPlayTraining(trainee, trainee, eps_per_round)
    except Exception as e:
        print(f"Error during self-play training in round {x}: {e}")
        continue

    # Create validation agent with previous checkpoint for comparison
    # Load state dict into a new model instance to avoid optimizer issues
    validation_model = NeuralNetwork(env.observation_space, 2, actions, None)
    validation_model.load_state_dict(prev_checkpoint.state_dict())
    valid_agent = Players.Agent("ValidAgent", validation_model, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR)

    try:
        p1w, p2w = Training.selfPlayTraining(trainee, valid_agent, eps_per_round, True)
    except Exception as e:
        print(f"Error during validation training in round {x}: {e}")
        continue

    # Save checkpoint for next round (always save, regardless of win/loss)
    new_checkpoint_path = models_dir / f"checkpoint{x + 1}.pt"
    try:
        if p2w < p1w:
            # New model won - save it
            torch.save(trainee.model, new_checkpoint_path)
            print(f"Saved new checkpoint: {new_checkpoint_path} (new model won: p1w={p1w} vs p2w={p2w})")
        else:
            # Previous model won - save it instead
            torch.save(prev_checkpoint, new_checkpoint_path)
            # Also update trainee to use the previous model
            trainee.model.load_state_dict(prev_checkpoint.state_dict())
            trainee.policy.load_state_dict(prev_checkpoint.state_dict())
            print(f"Saved previous checkpoint: {new_checkpoint_path} (previous model won: p1w={p1w} vs p2w={p2w})")
    except Exception as e:
        print(f"Error saving checkpoint {new_checkpoint_path}: {e}")

    try:
        p1w, p2w = Training.selfPlayTraining(trainee, validation01, eps_per_round, True)
        print(f"Round {x}: Validation vs heuristic - p1w: {p1w}, p2w: {p2w}")
    except Exception as e:
        print(f"Error during heuristic validation in round {x}: {e}")
        continue
