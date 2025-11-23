"""
Neural network architecture for DQN agent in Cuttle game environment.

This module provides a neural network that processes game observations
and outputs Q-values for all possible actions.
"""

from typing import Dict, List, Union, Optional, Any, Tuple
import numpy as np
import torch
from torch import nn


# Constants
EMBEDDING_VOCAB_SIZE = 54  # 0 = empty, 1-52 = cards, 53+ = special values
STACK_SIZE = 5  # Stack depth for action history
EFFECT_SHOWN_SIZE = 2  # Number of revealed cards


class NeuralNetwork(nn.Module):
    """
    Deep Q-Network (DQN) for Cuttle card game.
    
    Processes game observations (dictionary of zones and game state)
    and outputs Q-values for all possible actions.
    
    The network:
    1. Concatenates boolean zone arrays (hand, field, deck, scrap, etc.)
    2. Embeds discrete values (stack, effect_shown) using embedding layers
    3. Passes through a linear layer to output Q-values
    
    Args:
        observation_space: Dictionary representing the observation space structure
        embedding_size: Size of embedding vectors for discrete values
        num_actions: Number of possible actions in the environment
        custom_network: Optional custom network to replace default architecture
    """
    
    def __init__(
        self,
        observation_space: Dict[str, Any],
        embedding_size: int,
        num_actions: int,
        custom_network: Optional[nn.Sequential] = None
    ) -> None:
        """
        Initialize the neural network.
        
        Args:
            observation_space: Dictionary representing observation structure
            embedding_size: Dimension of embedding vectors
            num_actions: Total number of possible actions
            custom_network: Optional custom network architecture
        """
        super().__init__()
        
        self.embedding_size = embedding_size
        self.num_actions = num_actions
        
        if custom_network is not None:
            self.linear_relu_stack = custom_network
            # Still need embedding for stack and effect_shown
            self.embedding = nn.Embedding(EMBEDDING_VOCAB_SIZE, embedding_size)
        else:
            # Calculate input dimension from observation space
            input_length = self._calculate_input_dimension(observation_space, embedding_size)
            
            # Embedding layer for discrete values (stack, effect_shown)
            # Vocab size 54: 0 = empty, 1-52 = cards, 53+ = special values
            self.embedding = nn.Embedding(EMBEDDING_VOCAB_SIZE, embedding_size)
            
            # Default network: two hidden layers with ReLU activations
            # Conservative architecture: 256 → 128 → num_actions
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_length, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions),
                nn.Tanh()
            )
    
    def _calculate_input_dimension(
        self,
        observation_space: Dict[str, Any],
        embedding_size: int
    ) -> int:
        """
        Calculate the input dimension from observation space structure.
        
        Args:
            observation_space: Dictionary representing observation structure
            embedding_size: Dimension of embedding vectors
            
        Returns:
            Total input dimension for the network
        """
        input_length = 0
        
        for item in observation_space.values():
            if isinstance(item, dict):
                # Nested dictionary (e.g., "Current Zones")
                for nested_item in item.values():
                    if isinstance(nested_item, np.ndarray):
                        input_length += len(nested_item)
            elif isinstance(item, np.ndarray):
                # Direct numpy array
                input_length += len(item)
            elif isinstance(item, list):
                # List items are embedded (stack, effect_shown)
                input_length += len(item) * embedding_size
        
        return input_length
    
    def forward(
        self,
        observation: Union[Dict[str, Any], List[Dict[str, Any]], Tuple[Dict[str, Any], ...]]
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            observation: Single observation dict, list of observations, or tuple of observations for batching
            
        Returns:
            Q-values tensor of shape [batch_size, num_actions] or [num_actions]
        """
        state = self._preprocess_observation(observation)
        q_values = self.linear_relu_stack(state)
        return q_values
    
    def _preprocess_observation(
        self,
        observation: Union[Dict[str, Any], List[Dict[str, Any]], Tuple[Dict[str, Any], ...]]
    ) -> torch.Tensor:
        """
        Preprocess observation(s) into network input tensor.
        
        Args:
            observation: Single observation dict, list of observations, or tuple of observations
            
        Returns:
            Preprocessed tensor ready for network input
        """
        if isinstance(observation, dict):
            return self._preprocess_single(observation)
        elif isinstance(observation, (list, tuple)):
            return self._preprocess_batch(list(observation))
        else:
            raise TypeError(
                f"Expected dict, list, or tuple of dicts, got {type(observation)}"
            )
    
    def _preprocess_single(self, obs: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess a single observation.
        
        Args:
            obs: Single observation dictionary
            
        Returns:
            Preprocessed tensor of shape [feature_dim]
        """
        # Validate observation structure
        self._validate_observation(obs)
        
        # Concatenate boolean zone arrays
        zone_arrays = [
            obs["Current Zones"]["Hand"],
            obs["Current Zones"]["Field"],
            obs["Current Zones"]["Revealed"],
            obs["Off-Player Field"],
            obs["Off-Player Revealed"],
            obs["Deck"],
            obs["Scrap"],
        ]
        state = np.concatenate(zone_arrays, axis=0)
        state_tensor = torch.from_numpy(state).float()
        
        # Embed discrete values (stack and effect_shown)
        device = next(self.parameters()).device
        stack_tensor = torch.tensor(obs["Stack"], dtype=torch.long, device=device)
        effect_tensor = torch.tensor(obs["Effect-Shown"], dtype=torch.long, device=device)
        
        embed_stack = self.embedding(stack_tensor).flatten()
        embed_effect = self.embedding(effect_tensor).flatten()
        
        # Concatenate all features
        final = torch.cat([state_tensor.to(device), embed_stack, embed_effect])
        
        return final
    
    def _preprocess_batch(
        self, 
        observations: Union[List[Dict[str, Any]], Tuple[Dict[str, Any], ...]]
    ) -> torch.Tensor:
        """
        Preprocess a batch of observations.
        
        Args:
            observations: List or tuple of observation dictionaries
            
        Returns:
            Preprocessed tensor of shape [batch_size, feature_dim]
        """
        # Convert tuple to list for consistent processing
        if isinstance(observations, tuple):
            observations = list(observations)
        if not observations:
            # Return empty tensor with correct shape
            device = next(self.parameters()).device
            # Calculate feature dim from first observation if available
            if hasattr(self, '_cached_feature_dim'):
                feature_dim = self._cached_feature_dim
            else:
                # Use a sample observation to determine shape
                sample = self._preprocess_single(observations[0] if observations else {})
                feature_dim = sample.shape[0]
                self._cached_feature_dim = feature_dim
            return torch.empty(0, feature_dim, device=device)
        
        # Process each observation and stack
        processed = [self._preprocess_single(obs) for obs in observations]
        return torch.stack(processed)
    
    def _validate_observation(self, obs: Dict[str, Any]) -> None:
        """
        Validate observation structure.
        
        Args:
            obs: Observation dictionary to validate
            
        Raises:
            ValueError: If observation structure is invalid
        """
        required_keys = [
            "Current Zones",
            "Off-Player Field",
            "Off-Player Revealed",
            "Deck",
            "Scrap",
            "Stack",
            "Effect-Shown"
        ]
        
        for key in required_keys:
            if key not in obs:
                raise ValueError(f"Missing required observation key: {key}")
        
        # Validate nested structure
        if "Hand" not in obs["Current Zones"]:
            raise ValueError("Missing 'Hand' in 'Current Zones'")
        if "Field" not in obs["Current Zones"]:
            raise ValueError("Missing 'Field' in 'Current Zones'")
        if "Revealed" not in obs["Current Zones"]:
            raise ValueError("Missing 'Revealed' in 'Current Zones'")
        
        # Validate list lengths
        if len(obs["Stack"]) != STACK_SIZE:
            raise ValueError(
                f"Stack must have length {STACK_SIZE}, got {len(obs['Stack'])}"
            )
        if len(obs["Effect-Shown"]) != EFFECT_SHOWN_SIZE:
            raise ValueError(
                f"Effect-Shown must have length {EFFECT_SHOWN_SIZE}, "
                f"got {len(obs['Effect-Shown'])}"
            )
