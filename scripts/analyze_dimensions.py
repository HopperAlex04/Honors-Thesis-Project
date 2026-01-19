#!/usr/bin/env python3
"""
Analyze and recommend network dimensions using various strategies.

This script helps choose optimal encoder dimensions for embedding-based
and multi-encoder networks using rigorous methods.
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from cuttle.network_dimensions import (
    print_dimension_analysis,
    recommend_dimensions,
    count_parameters_boolean_network,
    count_parameters_embedding_network,
    count_parameters_multi_encoder_network
)


def main():
    """Print dimension analysis for both network types."""
    print("=" * 70)
    print("NETWORK DIMENSION ANALYSIS")
    print("=" * 70)
    print("\nThis analysis compares different strategies for choosing encoder dimensions.")
    print("Use these recommendations to select dimensions that match your experimental goals.\n")
    
    # Boolean network baseline
    bool_params = count_parameters_boolean_network(468, 3157)
    print(f"Boolean Network (baseline): {bool_params:,} parameters\n")
    
    # Embedding network analysis
    print_dimension_analysis("embedding")
    
    # Multi-encoder network analysis
    print_dimension_analysis("multi_encoder")
    
    # Example recommendations
    print("\n" + "=" * 70)
    print("EXAMPLE RECOMMENDATIONS")
    print("=" * 70)
    print("\nFor fair parameter count comparison:")
    print("-" * 70)
    
    emb_dims = recommend_dimensions(
        method="embedding",
        strategy="parameter_matching",
        target_params=bool_params
    )
    print(f"Embedding Network: embedding_dim={emb_dims['embedding_dim']}, "
          f"zone_encoded_dim={emb_dims['zone_encoded_dim']}")
    
    multi_dims = recommend_dimensions(
        method="multi_encoder",
        strategy="parameter_matching",
        target_params=bool_params
    )
    print(f"Multi-Encoder Network: encoder_hidden_dim={multi_dims['encoder_hidden_dim']}, "
          f"encoder_output_dim={multi_dims['encoder_output_dim']}")
    
    print("\nFor game-based alignment:")
    print("-" * 70)
    emb_dims = recommend_dimensions(method="embedding", strategy="game_based")
    print(f"Embedding Network: embedding_dim={emb_dims['embedding_dim']}, "
          f"zone_encoded_dim={emb_dims['zone_encoded_dim']}")
    
    multi_dims = recommend_dimensions(method="multi_encoder", strategy="game_based")
    print(f"Multi-Encoder Network: encoder_hidden_dim={multi_dims['encoder_hidden_dim']}, "
          f"encoder_output_dim={multi_dims['encoder_output_dim']}")
    
    print("\n" + "=" * 70)
    print("See obsidian-vault/04-Neural-Networks/Dimension Selection.md for details")
    print("=" * 70)


if __name__ == "__main__":
    main()
