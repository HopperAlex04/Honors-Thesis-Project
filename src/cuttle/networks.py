"""
Neural network architecture for DQN agent in Cuttle game environment.

This module provides a neural network that processes game observations
and outputs Q-values for all possible actions.

For dimension calculation utilities, see cuttle.network_dimensions module.
"""

from typing import Dict, List, Union, Optional, Any, Tuple
import numpy as np
import torch
from torch import nn


# Constants
# Stack and effect_shown are now boolean arrays of length 52 (no embeddings needed)


def init_weights(module: nn.Module) -> None:
    """
    Initialize network weights using Kaiming/He initialization.
    
    This initialization is recommended for networks using ReLU activation
    functions. It helps prevent vanishing/exploding gradients and ensures
    proper signal propagation through deep networks.
    
    Args:
        module: Neural network module to initialize
    """
    if isinstance(module, nn.Linear):
        # Kaiming/He initialization for ReLU networks
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Standard normal initialization for embeddings
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


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
            
            # Game-based architecture: 52-neuron hidden layer (one per card)
            # This matches the design of embedding and multi-encoder networks
            # for fair comparison in input representation experiments.
            # NOTE: No activation on output layer - Q-values should be unbounded
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_length, 52),  # Game-based: one neuron per card
                nn.ReLU(),
                nn.Linear(52, num_actions)
                # No activation function - Q-values need to be unbounded to represent
                # proper expected future rewards (can be > 1.0 with intermediate rewards)
            )
            
            # Apply Kaiming initialization for ReLU networks
            self.apply(init_weights)
    
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


class EmbeddingBasedNetwork(nn.Module):
    """
    Embedding-based DQN network for Cuttle card game.
    
    Uses learned card embeddings with zone aggregation to process observations.
    Architecture: Embedding → Zone aggregation → 52 neurons → num_actions
    
    Args:
        observation_space: Dictionary representing the observation space structure
        embedding_dim: Dimension of card embeddings (default: 32)
        num_actions: Number of possible actions in the environment
        zone_encoded_dim: Dimension of each zone encoding after aggregation (default: 32)
    """
    
    def __init__(
        self,
        observation_space: Dict[str, Any],
        embedding_dim: int = 52,
        num_actions: int = 3157,
        zone_encoded_dim: int = 52
    ) -> None:
        """
        Initialize the embedding-based network.
        
        Args:
            observation_space: Dictionary representing observation structure
            embedding_dim: Dimension of card embeddings
            num_actions: Total number of possible actions
            zone_encoded_dim: Dimension of each zone encoding after aggregation
        """
        super().__init__()
        
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.zone_encoded_dim = zone_encoded_dim
        
        # Card embedding layer: 52 cards → embedding_dim
        self.card_embedding = nn.Embedding(52, embedding_dim)
        
        # Zone aggregation: Use max pooling to aggregate cards in each zone
        # Each zone will be aggregated to zone_encoded_dim
        # We'll use a small MLP to process the aggregated embeddings
        self.zone_aggregator = nn.Sequential(
            nn.Linear(embedding_dim, zone_encoded_dim),
            nn.ReLU()
        )
        
        # Number of zones: Hand, Field, Revealed (current), Field, Revealed (off), Deck, Scrap, Stack, Effect-Shown = 9
        num_zones = 9
        fusion_dim = num_zones * zone_encoded_dim
        
        # Shared game-based hidden layer: 52 neurons (one per card)
        self.hidden_layer = nn.Sequential(
            nn.Linear(fusion_dim, 52),
            nn.ReLU()
        )
        
        # Output layer: 52 neurons → num_actions
        self.output_layer = nn.Linear(52, num_actions)
        
        # Apply Kaiming initialization for ReLU networks
        self.apply(init_weights)
    
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
        preprocessed = self._preprocess_observation(observation)
        hidden = self.hidden_layer(preprocessed)
        q_values = self.output_layer(hidden)
        return q_values
    
    def _preprocess_observation(
        self,
        observation: Union[Dict[str, Any], List[Dict[str, Any]], Tuple[Dict[str, Any], ...]]
    ) -> torch.Tensor:
        """
        Preprocess observation(s) using embeddings and zone aggregation.
        
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
        Preprocess a single observation using embeddings.
        
        Args:
            obs: Single observation dictionary
            
        Returns:
            Preprocessed tensor of shape [fusion_dim]
        """
        # Validate observation structure
        self._validate_observation(obs)
        
        # Get all zones
        zones = [
            obs["Current Zones"]["Hand"],
            obs["Current Zones"]["Field"],
            obs["Current Zones"]["Revealed"],
            obs["Off-Player Field"],
            obs["Off-Player Revealed"],
            obs["Deck"],
            obs["Scrap"],
            obs["Stack"],
            obs["Effect-Shown"],
        ]
        
        device = next(self.parameters()).device
        zone_encodings = []
        
        for zone in zones:
            # Get indices of present cards in this zone
            card_indices = np.where(zone)[0]
            
            if len(card_indices) == 0:
                # Empty zone: use zero embedding
                zone_embedding = torch.zeros(self.embedding_dim, device=device)
            else:
                # Embed cards and aggregate using max pooling
                card_indices_tensor = torch.from_numpy(card_indices).long().to(device)
                card_embeddings = self.card_embedding(card_indices_tensor)  # [num_cards, embedding_dim]
                # Max pooling across cards in zone
                zone_embedding = torch.max(card_embeddings, dim=0)[0]  # [embedding_dim]
            
            # Process through zone aggregator
            zone_encoding = self.zone_aggregator(zone_embedding)  # [zone_encoded_dim]
            zone_encodings.append(zone_encoding)
        
        # Concatenate all zone encodings
        fusion = torch.cat(zone_encodings, dim=0)  # [fusion_dim]
        return fusion
    
    def _preprocess_batch(
        self,
        observations: Union[List[Dict[str, Any]], Tuple[Dict[str, Any], ...]]
    ) -> torch.Tensor:
        """
        Preprocess a batch of observations.
        
        Args:
            observations: List or tuple of observation dictionaries
            
        Returns:
            Preprocessed tensor of shape [batch_size, fusion_dim]
        """
        if isinstance(observations, tuple):
            observations = list(observations)
        if not observations:
            device = next(self.parameters()).device
            fusion_dim = 9 * self.zone_encoded_dim
            return torch.empty(0, fusion_dim, device=device)
        
        processed = [self._preprocess_single(obs) for obs in observations]
        return torch.stack(processed)
    
    def _validate_observation(self, obs: Dict[str, Any]) -> None:
        """Validate observation structure (same as NeuralNetwork)."""
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
        
        if "Hand" not in obs["Current Zones"]:
            raise ValueError("Missing 'Hand' in 'Current Zones'")
        if "Field" not in obs["Current Zones"]:
            raise ValueError("Missing 'Field' in 'Current Zones'")
        if "Revealed" not in obs["Current Zones"]:
            raise ValueError("Missing 'Revealed' in 'Current Zones'")


class MultiEncoderNetwork(nn.Module):
    """
    Multi-encoder DQN network for Cuttle card game.
    
    Uses separate encoders for each zone type, then fuses representations.
    Architecture: Zone encoders → Fusion → 52 neurons → num_actions
    
    Args:
        observation_space: Dictionary representing the observation space structure
        num_actions: Number of possible actions in the environment
        encoder_hidden_dim: Hidden dimension for zone encoders (default: 26, game-based: 2×13 ranks)
        encoder_output_dim: Output dimension for each zone encoder (default: 13, game-based: one per rank)
    """
    
    def __init__(
        self,
        observation_space: Dict[str, Any],
        num_actions: int = 3157,
        encoder_hidden_dim: int = 26,  # Game-based: 2×13 (ranks)
        encoder_output_dim: int = 13   # Game-based: one per rank
    ) -> None:
        """
        Initialize the multi-encoder network.
        
        Args:
            observation_space: Dictionary representing observation structure
            num_actions: Total number of possible actions
            encoder_hidden_dim: Hidden dimension for zone encoders (default: 26, game-based: 2×13 ranks)
            encoder_output_dim: Output dimension for each zone encoder (default: 13, game-based: one per rank)
        """
        super().__init__()
        
        self.num_actions = num_actions
        self.encoder_output_dim = encoder_output_dim
        
        # Separate encoders for each zone type
        # Each processes a 52-dim boolean array
        self.hand_encoder = nn.Sequential(
            nn.Linear(52, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_output_dim)
        )
        self.field_encoder = nn.Sequential(
            nn.Linear(52, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_output_dim)
        )
        self.revealed_encoder = nn.Sequential(
            nn.Linear(52, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_output_dim)
        )
        self.off_field_encoder = nn.Sequential(
            nn.Linear(52, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_output_dim)
        )
        self.off_revealed_encoder = nn.Sequential(
            nn.Linear(52, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_output_dim)
        )
        self.deck_encoder = nn.Sequential(
            nn.Linear(52, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_output_dim)
        )
        self.scrap_encoder = nn.Sequential(
            nn.Linear(52, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_output_dim)
        )
        self.stack_encoder = nn.Sequential(
            nn.Linear(52, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_output_dim)
        )
        self.effect_shown_encoder = nn.Sequential(
            nn.Linear(52, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_output_dim)
        )
        
        # Number of zones: 9
        num_zones = 9
        fusion_dim = num_zones * encoder_output_dim
        
        # Shared game-based hidden layer: 52 neurons (one per card)
        self.hidden_layer = nn.Sequential(
            nn.Linear(fusion_dim, 52),
            nn.ReLU()
        )
        
        # Output layer: 52 neurons → num_actions
        self.output_layer = nn.Linear(52, num_actions)
        
        # Apply Kaiming initialization for ReLU networks
        self.apply(init_weights)
    
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
        preprocessed = self._preprocess_observation(observation)
        hidden = self.hidden_layer(preprocessed)
        q_values = self.output_layer(hidden)
        return q_values
    
    def _preprocess_observation(
        self,
        observation: Union[Dict[str, Any], List[Dict[str, Any]], Tuple[Dict[str, Any], ...]]
    ) -> torch.Tensor:
        """
        Preprocess observation(s) using zone-specific encoders.
        
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
        Preprocess a single observation using zone encoders.
        
        Args:
            obs: Single observation dictionary
            
        Returns:
            Preprocessed tensor of shape [fusion_dim]
        """
        # Validate observation structure
        self._validate_observation(obs)
        
        device = next(self.parameters()).device
        
        # Encode each zone using its specific encoder
        hand_encoded = self.hand_encoder(
            torch.from_numpy(obs["Current Zones"]["Hand"]).float().to(device)
        )
        field_encoded = self.field_encoder(
            torch.from_numpy(obs["Current Zones"]["Field"]).float().to(device)
        )
        revealed_encoded = self.revealed_encoder(
            torch.from_numpy(obs["Current Zones"]["Revealed"]).float().to(device)
        )
        off_field_encoded = self.off_field_encoder(
            torch.from_numpy(obs["Off-Player Field"]).float().to(device)
        )
        off_revealed_encoded = self.off_revealed_encoder(
            torch.from_numpy(obs["Off-Player Revealed"]).float().to(device)
        )
        deck_encoded = self.deck_encoder(
            torch.from_numpy(obs["Deck"]).float().to(device)
        )
        scrap_encoded = self.scrap_encoder(
            torch.from_numpy(obs["Scrap"]).float().to(device)
        )
        stack_encoded = self.stack_encoder(
            torch.from_numpy(obs["Stack"]).float().to(device)
        )
        effect_shown_encoded = self.effect_shown_encoder(
            torch.from_numpy(obs["Effect-Shown"]).float().to(device)
        )
        
        # Concatenate all zone encodings
        fusion = torch.cat([
            hand_encoded,
            field_encoded,
            revealed_encoded,
            off_field_encoded,
            off_revealed_encoded,
            deck_encoded,
            scrap_encoded,
            stack_encoded,
            effect_shown_encoded,
        ], dim=0)
        
        return fusion
    
    def _preprocess_batch(
        self,
        observations: Union[List[Dict[str, Any]], Tuple[Dict[str, Any], ...]]
    ) -> torch.Tensor:
        """
        Preprocess a batch of observations.
        
        Args:
            observations: List or tuple of observation dictionaries
            
        Returns:
            Preprocessed tensor of shape [batch_size, fusion_dim]
        """
        if isinstance(observations, tuple):
            observations = list(observations)
        if not observations:
            device = next(self.parameters()).device
            fusion_dim = 9 * self.encoder_output_dim
            return torch.empty(0, fusion_dim, device=device)
        
        processed = [self._preprocess_single(obs) for obs in observations]
        return torch.stack(processed)
    
    def _validate_observation(self, obs: Dict[str, Any]) -> None:
        """Validate observation structure (same as NeuralNetwork)."""
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
        
        if "Hand" not in obs["Current Zones"]:
            raise ValueError("Missing 'Hand' in 'Current Zones'")
        if "Field" not in obs["Current Zones"]:
            raise ValueError("Missing 'Field' in 'Current Zones'")
        if "Revealed" not in obs["Current Zones"]:
            raise ValueError("Missing 'Revealed' in 'Current Zones'")
