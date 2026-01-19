"""
Utility functions for calculating optimal network dimensions.

Provides methods to rigorously determine encoder dimensions based on:
- Parameter count matching
- Compression ratios
- Game-based heuristics
- Information-theoretic principles
"""

from typing import Dict, Tuple, Optional
import math


def count_parameters_boolean_network(
    input_dim: int,
    num_actions: int,
    hidden_layers: Tuple[int, ...] = (512, 256)
) -> int:
    """
    Count parameters in boolean network architecture.
    
    Args:
        input_dim: Input dimension (typically 468)
        num_actions: Number of actions
        hidden_layers: Tuple of hidden layer sizes
        
    Returns:
        Total parameter count
    """
    params = 0
    prev_dim = input_dim
    
    for hidden_dim in hidden_layers:
        # Linear layer: (prev_dim + 1) * hidden_dim (including bias)
        params += (prev_dim + 1) * hidden_dim
        prev_dim = hidden_dim
    
    # Output layer
    params += (prev_dim + 1) * num_actions
    
    return params


def count_parameters_embedding_network(
    embedding_dim: int,
    zone_encoded_dim: int,
    num_zones: int,
    num_actions: int,
    hidden_dim: int = 52
) -> int:
    """
    Count parameters in embedding-based network.
    
    Args:
        embedding_dim: Card embedding dimension
        zone_encoded_dim: Zone encoding dimension after aggregation
        num_zones: Number of zones (typically 9)
        num_actions: Number of actions
        hidden_dim: Hidden layer size (typically 52)
        
    Returns:
        Total parameter count
    """
    params = 0
    
    # Card embedding: 52 cards × embedding_dim
    params += 52 * embedding_dim
    
    # Zone aggregator: embedding_dim → zone_encoded_dim
    params += (embedding_dim + 1) * zone_encoded_dim
    
    # Fusion to hidden: (num_zones * zone_encoded_dim) → hidden_dim
    fusion_dim = num_zones * zone_encoded_dim
    params += (fusion_dim + 1) * hidden_dim
    
    # Hidden to output: hidden_dim → num_actions
    params += (hidden_dim + 1) * num_actions
    
    return params


def count_parameters_multi_encoder_network(
    encoder_hidden_dim: int,
    encoder_output_dim: int,
    num_zones: int,
    num_actions: int,
    hidden_dim: int = 52
) -> int:
    """
    Count parameters in multi-encoder network.
    
    Args:
        encoder_hidden_dim: Hidden dimension in zone encoders
        encoder_output_dim: Output dimension of zone encoders
        num_zones: Number of zones (typically 9)
        num_actions: Number of actions
        hidden_dim: Hidden layer size (typically 52)
        
    Returns:
        Total parameter count
    """
    params = 0
    
    # Each zone encoder: 52 → encoder_hidden_dim → encoder_output_dim
    encoder_params = (52 + 1) * encoder_hidden_dim + (encoder_hidden_dim + 1) * encoder_output_dim
    params += num_zones * encoder_params
    
    # Fusion to hidden: (num_zones * encoder_output_dim) → hidden_dim
    fusion_dim = num_zones * encoder_output_dim
    params += (fusion_dim + 1) * hidden_dim
    
    # Hidden to output: hidden_dim → num_actions
    params += (hidden_dim + 1) * num_actions
    
    return params


def calculate_dimensions_by_parameter_matching(
    target_params: int,
    num_zones: int = 9,
    num_actions: int = 3157,
    hidden_dim: int = 52,
    method: str = "embedding"
) -> Dict[str, int]:
    """
    Calculate encoder dimensions to match target parameter count.
    
    Args:
        target_params: Target total parameter count
        num_zones: Number of zones
        num_actions: Number of actions
        hidden_dim: Hidden layer size (fixed at 52)
        method: "embedding" or "multi_encoder"
        
    Returns:
        Dictionary with calculated dimensions
    """
    if method == "embedding":
        # For embedding network: solve for embedding_dim and zone_encoded_dim
        # Parameters = 52*emb + (emb+1)*zone + (9*zone+1)*52 + (52+1)*actions
        
        # Fixed parts: hidden layer and output layer
        fixed_params = (hidden_dim + 1) * num_actions  # Output layer
        
        # Variable parts: embedding + aggregator + fusion
        # We'll use a heuristic: embedding_dim = zone_encoded_dim (game-based)
        # This simplifies to: 52*d + (d+1)*d + (9*d+1)*52 = target - fixed
        
        available_params = target_params - fixed_params - (hidden_dim + 1) * hidden_dim  # Fusion layer
        
        # Solve: 52*d + d^2 + d = available (approximately)
        # d^2 + 53*d - available = 0
        # Using quadratic formula
        a = 1
        b = 53
        c = -available_params
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            # Fallback to reasonable defaults
            dim = 32
        else:
            dim = int((-b + math.sqrt(discriminant)) / (2 * a))
            dim = max(16, min(128, dim))  # Clamp to reasonable range
        
        return {
            "embedding_dim": dim,
            "zone_encoded_dim": dim
        }
    
    elif method == "multi_encoder":
        # For multi-encoder: solve for encoder_hidden_dim and encoder_output_dim
        # Parameters = 9*[(52+1)*hidden + (hidden+1)*output] + (9*output+1)*52 + (52+1)*actions
        
        fixed_params = (hidden_dim + 1) * num_actions  # Output layer
        
        # Per encoder: (52+1)*hidden + (hidden+1)*output
        # We'll use heuristic: encoder_hidden_dim = 2 * encoder_output_dim
        # This gives: 9*[53*2*out + (2*out+1)*out] = 9*[106*out + 2*out^2 + out] = 9*[107*out + 2*out^2]
        # Fusion: (9*output+1)*52
        
        # Solve: 9*[107*out + 2*out^2] + (9*out+1)*52 = available
        # Approximate: 9*2*out^2 + 9*107*out + 9*52*out ≈ available
        # 18*out^2 + 1431*out - available ≈ 0
        
        # First, estimate without fusion layer
        available_for_encoders = target_params - fixed_params - (1 * hidden_dim)  # Minimal fusion
        
        # Solve quadratic: 18*out^2 + 1431*out - available ≈ 0
        a = 18
        b = 1431
        c = -available_for_encoders
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            encoder_output_dim = 32  # Fallback
        else:
            encoder_output_dim = int((-b + math.sqrt(discriminant)) / (2 * a))
            encoder_output_dim = max(16, min(64, encoder_output_dim))  # Clamp
        
        encoder_hidden_dim = 2 * encoder_output_dim
        
        # Recalculate fusion with actual encoder_output_dim
        fusion_params = (num_zones * encoder_output_dim + 1) * hidden_dim
        encoder_params = num_zones * ((52 + 1) * encoder_hidden_dim + (encoder_hidden_dim + 1) * encoder_output_dim)
        total_estimated = fixed_params + fusion_params + encoder_params
        
        # If we're way off, adjust
        if total_estimated > target_params * 1.2:
            # Too many params, reduce
            encoder_output_dim = max(16, int(encoder_output_dim * 0.8))
            encoder_hidden_dim = 2 * encoder_output_dim
        elif total_estimated < target_params * 0.8:
            # Too few params, increase
            encoder_output_dim = min(64, int(encoder_output_dim * 1.2))
            encoder_hidden_dim = 2 * encoder_output_dim
        
        return {
            "encoder_hidden_dim": encoder_hidden_dim,
            "encoder_output_dim": encoder_output_dim
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_dimensions_by_compression_ratio(
    input_dim: int,
    target_compression: float,
    num_zones: int = 9,
    method: str = "embedding"
) -> Dict[str, int]:
    """
    Calculate dimensions based on desired compression ratio.
    
    Compression ratio = input_dim / fusion_dim
    
    Args:
        input_dim: Input dimension (468 for boolean)
        target_compression: Desired compression ratio (e.g., 1.0 = no compression)
        num_zones: Number of zones
        method: "embedding" or "multi_encoder"
        
    Returns:
        Dictionary with calculated dimensions
    """
    target_fusion_dim = int(input_dim / target_compression)
    
    if method == "embedding":
        # fusion_dim = num_zones * zone_encoded_dim
        zone_encoded_dim = target_fusion_dim // num_zones
        
        # Embedding dim should be similar to zone_encoded_dim (game-based)
        embedding_dim = zone_encoded_dim
        
        return {
            "embedding_dim": max(16, min(128, embedding_dim)),
            "zone_encoded_dim": max(16, min(128, zone_encoded_dim))
        }
    
    elif method == "multi_encoder":
        # fusion_dim = num_zones * encoder_output_dim
        encoder_output_dim = target_fusion_dim // num_zones
        
        # Hidden dim typically 2x output dim
        encoder_hidden_dim = 2 * encoder_output_dim
        
        return {
            "encoder_hidden_dim": max(32, min(128, encoder_hidden_dim)),
            "encoder_output_dim": max(16, min(64, encoder_output_dim))
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_dimensions_game_based(
    num_zones: int = 9,
    method: str = "embedding"
) -> Dict[str, int]:
    """
    Calculate dimensions based on game structure.
    
    Uses game-based heuristics:
    - Embedding: 52 (one per card)
    - Zone encoding: 52 (one per card) or 13 (one per rank)
    - Encoder output: 13 (one per rank) or 52 (one per card)
    
    Args:
        num_zones: Number of zones
        method: "embedding" or "multi_encoder"
        
    Returns:
        Dictionary with calculated dimensions
    """
    if method == "embedding":
        # Game-based: 52 cards, so embedding_dim = 52
        # Zone encoding: could be 52 (card-level) or 13 (rank-level)
        # Using 52 for consistency with game structure
        return {
            "embedding_dim": 52,
            "zone_encoded_dim": 52
        }
    
    elif method == "multi_encoder":
        # Game-based: 13 ranks, so encoder_output_dim = 13
        # Hidden dim: 2x output for capacity
        return {
            "encoder_hidden_dim": 26,  # 2 * 13
            "encoder_output_dim": 13   # One per rank
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_dimensions_information_theoretic(
    input_dim: int,
    num_zones: int = 9,
    method: str = "embedding",
    information_ratio: float = 0.5
) -> Dict[str, int]:
    """
    Calculate dimensions based on information-theoretic principles.
    
    Uses the idea that we need to preserve a certain ratio of information
    from the input to the fusion layer.
    
    Args:
        input_dim: Input dimension (468 for boolean)
        num_zones: Number of zones
        method: "embedding" or "multi_encoder"
        information_ratio: Ratio of information to preserve (0.0-1.0)
        
    Returns:
        Dictionary with calculated dimensions
    """
    # Effective dimensionality considering sparsity
    # Boolean arrays are sparse, so effective dim < input_dim
    effective_dim = int(input_dim * information_ratio)
    
    if method == "embedding":
        # Distribute effective dim across zones
        zone_encoded_dim = effective_dim // num_zones
        
        # Embedding should capture card relationships
        # Use sqrt of zone_encoded_dim as heuristic
        embedding_dim = int(math.sqrt(zone_encoded_dim * 52))
        
        return {
            "embedding_dim": max(16, min(128, embedding_dim)),
            "zone_encoded_dim": max(16, min(128, zone_encoded_dim))
        }
    
    elif method == "multi_encoder":
        # Each encoder should preserve information_ratio of its input
        encoder_output_dim = int(52 * information_ratio)
        encoder_hidden_dim = 2 * encoder_output_dim  # Standard 2x expansion
        
        return {
            "encoder_hidden_dim": max(32, min(128, encoder_hidden_dim)),
            "encoder_output_dim": max(16, min(64, encoder_output_dim))
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


def recommend_dimensions(
    method: str = "embedding",
    strategy: str = "parameter_matching",
    target_params: Optional[int] = None,
    input_dim: int = 468,
    num_zones: int = 9,
    num_actions: int = 3157,
    hidden_dim: int = 52,
    **kwargs
) -> Dict[str, int]:
    """
    Recommend encoder dimensions using specified strategy.
    
    Args:
        method: "embedding" or "multi_encoder"
        strategy: "parameter_matching", "compression_ratio", "game_based", or "information_theoretic"
        target_params: Target parameter count (for parameter_matching)
        input_dim: Input dimension (for compression_ratio)
        num_zones: Number of zones
        num_actions: Number of actions
        hidden_dim: Hidden layer size (fixed at 52)
        **kwargs: Additional arguments for specific strategies
        
    Returns:
        Dictionary with recommended dimensions
    """
    if strategy == "parameter_matching":
        if target_params is None:
            # Default: match boolean network parameter count
            target_params = count_parameters_boolean_network(input_dim, num_actions)
        
        return calculate_dimensions_by_parameter_matching(
            target_params, num_zones, num_actions, hidden_dim, method
        )
    
    elif strategy == "compression_ratio":
        compression = kwargs.get("compression_ratio", 1.0)
        return calculate_dimensions_by_compression_ratio(
            input_dim, compression, num_zones, method
        )
    
    elif strategy == "game_based":
        return calculate_dimensions_game_based(num_zones, method)
    
    elif strategy == "information_theoretic":
        info_ratio = kwargs.get("information_ratio", 0.5)
        return calculate_dimensions_information_theoretic(
            input_dim, num_zones, method, info_ratio
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def print_dimension_analysis(
    method: str = "embedding",
    num_zones: int = 9,
    num_actions: int = 3157,
    hidden_dim: int = 52
) -> None:
    """
    Print analysis of different dimension strategies.
    
    Args:
        method: "embedding" or "multi_encoder"
        num_zones: Number of zones
        num_actions: Number of actions
        hidden_dim: Hidden layer size
    """
    print(f"\n{'='*60}")
    print(f"Dimension Analysis for {method.upper()} Network")
    print(f"{'='*60}\n")
    
    # Current dimensions
    if method == "embedding":
        current_emb = 52
        current_zone = 52
        current_params = count_parameters_embedding_network(
            current_emb, current_zone, num_zones, num_actions, hidden_dim
        )
        print(f"Current Dimensions:")
        print(f"  embedding_dim: {current_emb}")
        print(f"  zone_encoded_dim: {current_zone}")
        print(f"  Total parameters: {current_params:,}\n")
    else:
        current_hidden = 64
        current_output = 32
        current_params = count_parameters_multi_encoder_network(
            current_hidden, current_output, num_zones, num_actions, hidden_dim
        )
        print(f"Current Dimensions:")
        print(f"  encoder_hidden_dim: {current_hidden}")
        print(f"  encoder_output_dim: {current_output}")
        print(f"  Total parameters: {current_params:,}\n")
    
    # Boolean network baseline
    bool_params = count_parameters_boolean_network(468, num_actions)
    print(f"Boolean Network (baseline): {bool_params:,} parameters\n")
    
    # Different strategies
    strategies = [
        ("parameter_matching", {"target_params": bool_params}),
        ("compression_ratio", {"compression_ratio": 1.0}),
        ("game_based", {}),
        ("information_theoretic", {"information_ratio": 0.5}),
    ]
    
    print("Recommended Dimensions by Strategy:")
    print("-" * 60)
    
    for strategy_name, kwargs in strategies:
        dims = recommend_dimensions(
            method=method,
            strategy=strategy_name,
            num_zones=num_zones,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            **kwargs
        )
        
        if method == "embedding":
            params = count_parameters_embedding_network(
                dims["embedding_dim"], dims["zone_encoded_dim"],
                num_zones, num_actions, hidden_dim
            )
            print(f"{strategy_name:25s}: emb={dims['embedding_dim']:3d}, zone={dims['zone_encoded_dim']:3d}, params={params:7,}")
        else:
            params = count_parameters_multi_encoder_network(
                dims["encoder_hidden_dim"], dims["encoder_output_dim"],
                num_zones, num_actions, hidden_dim
            )
            print(f"{strategy_name:25s}: hidden={dims['encoder_hidden_dim']:3d}, output={dims['encoder_output_dim']:3d}, params={params:7,}")
    
    print()
