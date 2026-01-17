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
# Stack and effect_shown are now boolean arrays of length 52 (no embeddings needed)


class NeuralNetwork(nn.Module):
    """
    Deep Q-Network (DQN) for Cuttle card game.
    
    Processes game observations (dictionary of zones and game state)
    and outputs Q-values for all possible actions.
    
    The network:
    1. Concatenates boolean zone arrays (hand, field, deck, scrap, stack, effect_shown)
    2. Passes through a linear layer to output Q-values
    
    Args:
        observation_space: Dictionary representing the observation space structure
        embedding_size: Size of embedding vectors for discrete values
        num_actions: Number of possible actions in the environment
        custom_network: Optional custom network to replace default architecture
    """
    
    def __init__(
        self,
        observation_space: Dict[str, Any],
        embedding_size: int,  # Kept for backward compatibility, no longer used (no embeddings)
        num_actions: int,
        custom_network: Optional[nn.Sequential] = None
    ) -> None:
        """
        Initialize the neural network.
        
        Args:
            observation_space: Dictionary representing observation structure
            embedding_size: Deprecated - kept for backward compatibility, no longer used
            num_actions: Total number of possible actions
            custom_network: Optional custom network architecture
        """
        super().__init__()
        
        self.num_actions = num_actions
        
        if custom_network is not None:
            self.linear_relu_stack = custom_network
        else:
            # Calculate input dimension from observation space
            input_length = self._calculate_input_dimension(observation_space)
            
            # Default network: two hidden layers with ReLU activations
            # Conservative architecture: 256 → 128 → num_actions
            # NOTE: No activation on output layer - Q-values should be unbounded
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_length, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions)
                # No activation function - Q-values need to be unbounded to represent
                # proper expected future rewards (can be > 1.0 with intermediate rewards)
            )
    
    def _calculate_input_dimension(
        self,
        observation_space: Dict[str, Any]
    ) -> int:
        """
        Calculate the input dimension from observation space structure.
        
        All inputs are boolean arrays: zones (52 each), stack (52), effect_shown (52).
        
        Args:
            observation_space: Dictionary representing observation structure
            
        Returns:
            Total input dimension for the network
        """
        input_length = 0
        
        for key, item in observation_space.items():
            if isinstance(item, dict):
                # Nested dictionary (e.g., "Current Zones")
                for nested_item in item.values():
                    if isinstance(nested_item, np.ndarray):
                        input_length += len(nested_item)
            elif isinstance(item, np.ndarray):
                # Direct numpy array (stack, effect_shown, zones)
                input_length += len(item)
        
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
    
    def get_state(
        self,
        observation: Union[Dict[str, Any], List[Dict[str, Any]], Tuple[Dict[str, Any], ...]]
    ) -> torch.Tensor:
        """
        Get preprocessed state from observation(s).
        
        This is an alias for _preprocess_observation for backward compatibility.
        
        Args:
            observation: Single observation dict, list of observations, or tuple of observations
            
        Returns:
            Preprocessed tensor ready for network input
        """
        return self._preprocess_observation(observation)
    
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
        
        # Concatenate boolean zone arrays (all are boolean arrays of length 52)
        zone_arrays = [
            obs["Current Zones"]["Hand"],
            obs["Current Zones"]["Field"],
            obs["Current Zones"]["Revealed"],
            obs["Off-Player Field"],
            obs["Off-Player Revealed"],
            obs["Deck"],
            obs["Scrap"],
            obs["Stack"],      # Boolean array of length 52
            obs["Effect-Shown"],  # Boolean array of length 52
        ]
        state = np.concatenate(zone_arrays, axis=0)
        device = next(self.parameters()).device
        state_tensor = torch.from_numpy(state).float().to(device)
        
        return state_tensor
    
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
            "Effect-Shown",
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
        
        # Validate array lengths (stack and effect_shown are now boolean arrays)
        if not isinstance(obs["Stack"], np.ndarray) or len(obs["Stack"]) != 52:
            raise ValueError(
                f"Stack must be a boolean array of length 52, got {type(obs['Stack'])} with length {len(obs['Stack']) if hasattr(obs['Stack'], '__len__') else 'N/A'}"
            )
        if not isinstance(obs["Effect-Shown"], np.ndarray) or len(obs["Effect-Shown"]) != 52:
            raise ValueError(
                f"Effect-Shown must be a boolean array of length 52, got {type(obs['Effect-Shown'])} with length {len(obs['Effect-Shown']) if hasattr(obs['Effect-Shown'], '__len__') else 'N/A'}"
            )
