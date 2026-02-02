#!/usr/bin/env python3
"""
Calculate game_based scale to match large_hidden parameter count.

This script calculates the parameter counts for different network architectures
and finds the scale factor for game_based that matches large_hidden's parameters.
"""

# Constants
num_actions = 3157
embedding_dim = 52
zone_encoded_dim = 52
num_zones = 9
fusion_dim = num_zones * zone_encoded_dim  # 468


def count_params_embedding_network(hidden_layers):
    """Count parameters in embedding-based network."""
    params = 0
    
    # Embedding layer: 52 cards × embedding_dim
    params += 52 * embedding_dim
    
    # Zone aggregator: embedding_dim → zone_encoded_dim
    params += (embedding_dim + 1) * zone_encoded_dim
    
    # Fusion to first hidden: fusion_dim → hidden_layers[0]
    prev_dim = fusion_dim
    for hidden_dim in hidden_layers:
        params += (prev_dim + 1) * hidden_dim
        prev_dim = hidden_dim
    
    # Output layer: last_hidden → num_actions
    params += (prev_dim + 1) * num_actions
    
    return params


def main():
    # Calculate large_hidden parameters
    large_hidden_layers = [512]
    large_hidden_params = count_params_embedding_network(large_hidden_layers)
    
    print(f"{'='*70}")
    print("Parameter Count Analysis")
    print(f"{'='*70}\n")
    print(f"Large hidden ([512]): {large_hidden_params:,} parameters\n")
    
    # Calculate game_based at different scales
    print("Game-based scaling:")
    print("-" * 70)
    
    target_scale = None
    for scale in range(1, 10):
        hidden_layers = [52 * scale, 13 * scale, 15 * scale]
        params = count_params_embedding_network(hidden_layers)
        ratio = params / large_hidden_params
        match = "✓ MATCH" if abs(params - large_hidden_params) < 1000 else ""
        print(f"Scale {scale:2d}: [{52*scale:3d}, {13*scale:3d}, {15*scale:3d}] → {params:10,} params ({ratio:.2f}x) {match}")
        
        if params >= large_hidden_params and target_scale is None:
            target_scale = scale
    
    print(f"\n{'='*70}")
    if target_scale:
        print(f"Scale {target_scale} is the first to match or exceed large_hidden")
    else:
        print("No scale up to 9 matches large_hidden")
    print(f"{'='*70}\n")
    
    # Find exact match (if possible with non-integer scale)
    print("Finding optimal scale for exact match:")
    print("-" * 70)
    
    # Try to find a scale that gets close
    best_scale = None
    best_diff = float('inf')
    
    for scale in range(1, 20):
        hidden_layers = [52 * scale, 13 * scale, 15 * scale]
        params = count_params_embedding_network(hidden_layers)
        diff = abs(params - large_hidden_params)
        if diff < best_diff:
            best_diff = diff
            best_scale = scale
    
    if best_scale:
        hidden_layers = [52 * best_scale, 13 * best_scale, 15 * best_scale]
        params = count_params_embedding_network(hidden_layers)
        print(f"Best match: Scale {best_scale} → {params:,} params (diff: {best_diff:,})")
        print(f"Hidden layers: {hidden_layers}")
        print(f"Ratio to large_hidden: {params / large_hidden_params:.4f}")


if __name__ == "__main__":
    main()
